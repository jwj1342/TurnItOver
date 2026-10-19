"""Action-only acquisition followed by an independent frozen verifier readout."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

from turnitover.core.serde import to_dict
from turnitover.core.actions import QueryRuntime
from turnitover.models.client import ModelError, is_normal_finish_reason
from turnitover.oracle.supervision import action_examples, action_json
from turnitover.verifier.contracts import DecisionError, parse_json, parse_decision
from turnitover.verifier.policies import VisionPolicy


ACTION_STAGE_INSTRUCTION = (
    "This is the acquisition-only stage. Choose the next legal observation action. "
    "Return exactly one JSON object with only the action key and its action object. "
    "Do not return a verdict or stop early; a separate frozen diagnoser runs after acquisition."
)


def complete_final_readout(policy, context, episode):
    """Diagnose the acquired prefix after acquisition failure without more actions.

    Preserve acquisition termination, errors, trajectory and spent budget. A final
    readout already attempted by the ordinary Loop is never called a second time.
    """
    from turnitover.verifier.loop import MAX_DECISION_ATTEMPTS, _validation_feedback

    if policy.final.calls:
        failed = episode.termination in ("invalid_decision", "model_error")
        return episode, dict(supplemental=False, calls=len(policy.final.calls),
                             status="failed" if failed else "complete",
                             error_type=episode.error_type if failed else None,
                             error=episode.error_message if failed else None)
    feedback = None
    for attempt in range(MAX_DECISION_ATTEMPTS):
        raw = None
        try:
            raw = policy.final.decide(context, episode.trajectory, 0, feedback)
            verdict = parse_decision(raw, context, {o.step for o in episode.trajectory}, final_only=True)
            return replace(episode, verdict=verdict), dict(supplemental=True, calls=attempt+1, status="complete")
        except DecisionError as exc:
            if attempt + 1 < MAX_DECISION_ATTEMPTS:
                feedback = _validation_feedback(exc, raw, context, episode.trajectory, 0)
                continue
            error = exc
        except ModelError as exc:
            error = exc
        return episode, dict(supplemental=True, calls=attempt+1, status="failed",
                             error_type=type(error).__name__, error=str(error))
    raise AssertionError("Final readout did not terminate")


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
        protocol, payload_text = row["messages"][0]["content"].rsplit("\n", 1)
        prompt = protocol + "\n" + ACTION_STAGE_INSTRUCTION + "\n" + payload_text
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
        try:
            result = self.call(prompt, [self.output / image for image in row["images"]])
        except ModelError as exc:
            meta = dict(error_type=type(exc).__name__, error=str(exc), legal_action=False)
            self.calls.append(meta)
            (folder / "metadata.json").write_text(json.dumps(meta, indent=2))
            raise
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
