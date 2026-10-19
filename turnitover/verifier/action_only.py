"""Action-only acquisition followed by an independent frozen verifier readout."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from turnitover.core.serde import to_dict
from turnitover.core.actions import QueryRuntime
from turnitover.models.client import ModelError, is_normal_finish_reason
from turnitover.oracle.supervision import action_examples, action_json
from turnitover.verifier.contracts import DecisionError, parse_json, parse_decision
from turnitover.verifier.policies import VisionPolicy


class ActionOnlyPolicy:
    """Keep acquisition inputs identical to the existing causal action export."""

    def __init__(self, call, diagnose, output: Path, reference: str, actions=None):
        self.call, self.output, self.reference, self.actions = call, output, reference, actions
        self.final = VisionPolicy(diagnose, output, [output / reference], mode="fixed")
        self.calls = []
        self.action_attempts = 0
        self.legal_actions = 0

    def decide(self, context, history, remaining, validation_feedback=None):
        if remaining == 0 or (self.actions is not None and len(history) >= len(self.actions)):
            # Calling final with zero ensures VisionPolicy cannot collect extra actions.
            return self.final.decide(context, history, 0, validation_feedback)
        if self.actions is not None:
            value = {"action": self.actions[len(history)]}
            self.action_attempts += 1
            parse_decision(value, context, {o.step for o in history})
            self.legal_actions += 1
            return value
        public = []
        for observation in history:
            row = to_dict(observation)
            row.pop("elapsed_ms", None)
            row["action"] = ({"type": "query_runtime", "property": observation.action.property.value}
                             if isinstance(observation.action, QueryRuntime) else action_json(observation.action))
            row["payload"] = {k: v for k, v in (row.get("payload") or {}).items() if k != "render_ms"}
            public.append({k: row[k] for k in ("step", "action", "image_ref", "payload")})
        # action_examples constructs only the preceding prefix. The dummy action is
        # validated but never enters the model-visible user content.
        dummy = {"step": len(public), "action": {"type": "request_view", "view_id": context.views[0]}}
        row = list(action_examples("online", asdict(context), self.reference, public + [dummy],
                                   remaining + len(public)))[-1]
        prompt = row["messages"][0]["content"]
        if any(image is None for image in row["images"]):
            # Runtime observations retain their payload and step but have no PNG.
            protocol, payload_text = prompt.rsplit("\n", 1)
            payload = json.loads(payload_text)
            row["images"] = [image for image in row["images"] if image is not None]
            payload["image_order"] = [image for image in payload["image_order"] if image is not None]
            prompt = protocol + "\n" + json.dumps(payload)
        if validation_feedback:
            prompt += "\n\nValidation feedback from your previous response:\n" + validation_feedback
        folder = self.output / "action_calls" / f"{len(self.calls):03d}"
        folder.mkdir(parents=True)
        (folder / "prompt.txt").write_text(prompt)
        (folder / "images.json").write_text(json.dumps(row["images"]))
        self.action_attempts += 1
        result = self.call(prompt, [self.output / image for image in row["images"]])
        (folder / "response.txt").write_text(result.text)
        meta = {k: v for k, v in asdict(result).items() if k != "text"}
        self.calls.append(meta)
        try:
            if not is_normal_finish_reason(result.finish_reason):
                raise ModelError("Action model response is incomplete")
            value = parse_json(result.text)
            if not isinstance(value, dict) or set(value) != {"action"}:
                raise DecisionError("Action-only acquisition requires exactly one action")
            meta["action_only_json"] = True
            parse_decision(value, context, {o.step for o in history})
            meta["legal_action"] = True
            self.legal_actions += 1
            return value
        finally:
            (folder / "metadata.json").write_text(json.dumps(meta, indent=2))
