"""Build the common visible teacher request and retain every raw model exchange."""
from __future__ import annotations

import dataclasses
import json
from importlib import resources
from pathlib import Path
from typing import Sequence

from turnitover.core.actions import Observation
from turnitover.core.serde import to_dict
from turnitover.models.client import ModelError, is_normal_finish_reason
from turnitover.teacher.providers import Teacher
from turnitover.verifier.contracts import Context, DecisionError, parse_json


class TeacherPolicy:
    def __init__(self, teacher: Teacher, output: Path, references: list[Path], calls: list[dict] | None = None):
        self.teacher, self.output, self.references = teacher, output, references
        self.calls = list(calls or [])

    def decide(self, context: Context, history: Sequence[Observation], remaining: int,
               validation_feedback: str | None = None) -> dict:
        images = list(self.references)
        order = [{"image": i, "kind": "reference"} for i in range(len(images))]
        for observation in history:
            if observation.image_ref:
                order.append({"image": len(images), "kind": "observation", "step": observation.step})
                images.append(self.output / observation.image_ref)
        visible = {"context": dataclasses.asdict(context), "remaining": remaining,
                   "stop_required": remaining == 0,
                   "image_order": order, "history": [to_dict(o) for o in history]}
        protocol = resources.files("turnitover.teacher").joinpath("prompt.txt").read_text()
        prompt = protocol + "\nVisible task state:\n" + json.dumps(visible, ensure_ascii=False)
        if validation_feedback:
            prompt += "\nValidation feedback:\n" + validation_feedback

        call_dir = self.output / "model_calls" / f"{len(self.calls):03d}"
        call_dir.mkdir(parents=True, exist_ok=False)
        (call_dir / "prompt.txt").write_text(prompt)
        (call_dir / "request.json").write_text(json.dumps(visible, indent=2))
        (call_dir / "images.json").write_text(json.dumps([str(p.relative_to(self.output)) for p in images]))
        record = {"path": str(call_dir.relative_to(self.output)), "status": "started",
                  "remaining": remaining, "image_count": len(images)}
        self.calls.append(record)
        try:
            response = self.teacher(prompt, images)
            (call_dir / "response.txt").write_text(response.text)
            record.update({k: v for k, v in dataclasses.asdict(response).items() if k != "text"})
            if not is_normal_finish_reason(response.finish_reason):
                raise ModelError("Teacher response is incomplete")
            value = parse_json(response.text)
            if not isinstance(value, dict):
                raise DecisionError("Teacher decision must be a JSON object")
            record["status"] = "received"
            return value
        except Exception as exc:
            record.update(status="failed", error_type=type(exc).__name__, error_message=str(exc))
            raise
        finally:
            (call_dir / "metadata.json").write_text(json.dumps(record, indent=2))
