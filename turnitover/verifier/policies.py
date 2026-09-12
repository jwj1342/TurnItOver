"""Active VLM, fixed/random observation baselines, and an honest offline runtime baseline."""
from __future__ import annotations

import dataclasses
import json
from importlib import resources
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from turnitover.core.actions import Observation, QueryRuntime, RuntimeProperty
from turnitover.core.serde import to_dict
from turnitover.models.client import ModelError, ModelResult
from turnitover.taxonomy import load_taxonomy
from turnitover.verifier.contracts import Context, parse_json


def action_plan(context: Context) -> list[dict]:
    actions = []
    views = sorted(context.views, key=lambda v: 0 if v == "front" else 1 if v == "oblique_fr" else 2)
    if "query_runtime" in context.action_types and "stats" in context.runtime_properties:
        actions.append({"type": "query_runtime", "property": "stats"})
    if "request_view" in context.action_types:
        actions.extend({"type": "request_view", "view_id": view} for view in views[:2])
    if "actuate_joint" in context.action_types:
        for joint in context.joints:
            actions.append({"type": "actuate_joint", "joint_id": joint, "detent": "limit"})
            if "query_runtime" in context.action_types and "state_delta" in context.runtime_properties:
                actions.append({"type": "query_runtime", "property": "state_delta"})
            actions.append({"type": "actuate_joint", "joint_id": joint, "detent": "zero"})
    if "request_view" in context.action_types:
        actions.extend({"type": "request_view", "view_id": view} for view in views[2:])
    return actions


class VisionPolicy:
    def __init__(self, call: Callable[[str, list[Path]], ModelResult], output: Path,
                 references: list[Path], mode: str = "active", seed: int = 0):
        self.call, self.output, self.references, self.mode = call, output, references, mode
        self.rng = np.random.default_rng(seed)
        self.calls: list[dict] = []
        self._plan = None

    def decide(self, context: Context, history: Sequence[Observation], remaining: int) -> dict:
        if self._plan is None:
            self._plan = action_plan(context)
            if self.mode == "random":
                self.rng.shuffle(self._plan)
        if self.mode in ("fixed", "random") and remaining and len(history) < len(self._plan):
            return {"action": self._plan[len(history)]}
        images = list(self.references)
        order = [{"image": i, "kind": "reference"} for i in range(len(images))]
        for obs in history:
            if obs.image_ref:
                order.append({"image": len(images), "kind": "observation", "step": obs.step})
                images.append(self.output / obs.image_ref)
        data = {"context": dataclasses.asdict(context), "remaining": remaining,
                "taxonomy_ids": load_taxonomy().ids, "image_order": order,
                "history": [to_dict(o) for o in history]}
        final_only = remaining == 0 or self.mode != "active"
        protocol = resources.files("turnitover.verifier").joinpath("final_prompt.txt" if final_only else "prompt.txt").read_text()
        prompt = protocol + "\nObserved task data:\n" + json.dumps(data, ensure_ascii=False)
        if self.mode != "active":
            prompt += "\nFixed observation collection has finished. Return a final verdict now."
        call_dir = self.output / "model_calls" / f"{len(self.calls):03d}"
        call_dir.mkdir(parents=True)
        (call_dir / "prompt.txt").write_text(prompt)
        (call_dir / "images.json").write_text(json.dumps([str(p.relative_to(self.output)) for p in images]))
        record = {"path": str(call_dir.relative_to(self.output)), "status": "started"}
        self.calls.append(record)
        try:
            response = self.call(prompt, images)
            (call_dir / "response.txt").write_text(response.text)
            record.update({k: v for k, v in dataclasses.asdict(response).items() if k != "text"})
            if response.finish_reason not in {"completed", "end_turn", "STOP", "stop"}:
                raise ModelError("Judge response is incomplete")
            parsed = parse_json(response.text)
            if self.mode != "active" and "verdict" not in parsed:
                from turnitover.verifier.contracts import DecisionError

                raise DecisionError("Fixed/random baseline must return a final verdict")
            record["status"] = "received"
            return parsed
        except Exception as exc:
            record.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            (call_dir / "metadata.json").write_text(json.dumps(record, indent=2))


class RuntimePolicy:
    """Checks observable budget violations only; never claims reference/geometry validity."""
    calls: tuple = ()

    def decide(self, context: Context, history: Sequence[Observation], remaining: int) -> dict:
        plan = action_plan(context)
        if remaining and len(history) < len(plan):
            return {"action": plan[len(history)]}
        findings = []
        for obs in history:
            if not isinstance(obs.action, QueryRuntime) or obs.action.property != RuntimeProperty.STATS:
                continue
            for key, limit, defect in (("triangles_scene", context.max_triangles, "runtime.triangle_budget"),
                                       ("draw_calls", context.max_draw_calls, "runtime.draw_calls")):
                value = (obs.payload or {}).get(key)
                if isinstance(value, (int, float)) and value > limit:
                    findings.append({"defect_id": defect, "parts": [], "severity": min(1, (value-limit)/limit),
                        "confidence": 1.0, "evidence_steps": [obs.step],
                        "description": f"{key}={value} exceeds limit {limit}.",
                        "suggested_fix": "Reduce subdivisions or consolidate draw calls, then recheck."})
        return {"verdict": {"status": "fail" if findings else "uncertain", "confidence": 1.0 if findings else 0.0,
                            "findings": findings, "summary": "Runtime budget violations found." if findings else "No budget violation established; visual verification remains unavailable.",
                            "limitations": ["Offline runtime baseline does not judge visual, kinematic or physical correctness."]}}
