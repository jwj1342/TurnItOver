"""Verifier result v1, separate from the existing synthetic Sample schema."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

from turnitover.core.actions import ActuateJoint, Detent, QueryRuntime, RequestView, RuntimeProperty, ViewId
from turnitover.taxonomy import load_taxonomy

VERIFIER_SCHEMA_VERSION = 1


class DecisionError(ValueError):
    pass


@dataclass(frozen=True)
class Context:
    task: str
    parts: tuple[str, ...]
    joints: tuple[str, ...]
    views: tuple[str, ...]
    runtime_properties: tuple[str, ...]
    action_types: tuple[str, ...]
    max_triangles: int
    max_draw_calls: int


@dataclass(frozen=True)
class Finding:
    defect_id: str
    parts: tuple[str, ...]
    severity: float
    confidence: float
    evidence_steps: tuple[int, ...]
    description: str
    suggested_fix: str


@dataclass(frozen=True)
class Verdict:
    status: str
    confidence: float
    findings: tuple[Finding, ...]
    summary: str
    limitations: tuple[str, ...]


def uncertain(message: str) -> Verdict:
    return Verdict("uncertain", 0.0, (), message, ("Verification is incomplete; do not treat this as a pass.",))


def _keys(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise DecisionError("Decision fields do not match the required schema")


def _unit(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise DecisionError("Confidence and severity must be finite numbers in [0, 1]")
    return float(value)


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise DecisionError("Description fields must be nonempty strings")
    return value


def _strings(value):
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise DecisionError("Expected a list of strings")
    return tuple(value)


def parse_json(text: str) -> dict:
    # Accept a single enclosing JSON fence, never extract a convenient substring from prose.
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise DecisionError("Duplicate JSON key")
            result[key] = value
        return result
    try:
        return json.loads(text, object_pairs_hook=pairs)
    except (ValueError, TypeError):
        raise DecisionError("Model must return one valid JSON decision") from None


def parse_decision(value: dict, context: Context, observed_steps: set[int], *, final_only: bool = False):
    if not isinstance(value, dict) or len(value) != 1:
        raise DecisionError("Return exactly one action or verdict")
    if "action" in value:
        if final_only:
            raise DecisionError("Observation budget is exhausted; a verdict is required")
        a = value["action"]
        if not isinstance(a, dict) or a.get("type") not in context.action_types:
            raise DecisionError("Action type is unavailable")
        if a["type"] == "request_view":
            _keys(a, ("type", "view_id"))
            if a["view_id"] not in context.views:
                raise DecisionError("Unknown view")
            return RequestView(ViewId(a["view_id"]))
        if a["type"] == "actuate_joint":
            _keys(a, ("type", "joint_id", "detent"))
            if a["joint_id"] not in context.joints or a["detent"] not in tuple(d.value for d in Detent):
                raise DecisionError("Unknown joint or detent")
            return ActuateJoint(a["joint_id"], Detent(a["detent"]))
        if a["type"] == "query_runtime":
            _keys(a, ("type", "property"))
            if a["property"] not in context.runtime_properties:
                raise DecisionError("Unavailable runtime property")
            return QueryRuntime(RuntimeProperty(a["property"]))
        raise DecisionError("Unsupported action")
    if "verdict" not in value:
        raise DecisionError("Missing verdict")
    v = value["verdict"]
    _keys(v, ("status", "confidence", "findings", "summary", "limitations"))
    if v["status"] not in ("pass", "fail", "uncertain") or not isinstance(v["findings"], list):
        raise DecisionError("Invalid verdict status or findings")
    findings = []
    for f in v["findings"]:
        _keys(f, ("defect_id", "parts", "severity", "confidence", "evidence_steps", "description", "suggested_fix"))
        if f["defect_id"] not in load_taxonomy().ids:
            raise DecisionError("Unknown defect taxonomy ID")
        parts = _strings(f["parts"])
        if not set(parts) <= set(context.parts):
            raise DecisionError("Unknown part ID; describe a missing part in text and use parts=[]")
        evidence = f["evidence_steps"]
        if (not isinstance(evidence, list) or not evidence or any(type(i) is not int for i in evidence)
                or not set(evidence) <= observed_steps):
            raise DecisionError("Findings must cite existing successful observation steps")
        findings.append(Finding(f["defect_id"], parts, _unit(f["severity"]), _unit(f["confidence"]),
                                tuple(evidence), _text(f["description"]), _text(f["suggested_fix"])))
    if v["status"] == "pass" and (findings or not observed_steps):
        raise DecisionError("Pass requires observations and no findings")
    if v["status"] == "fail" and not findings:
        raise DecisionError("Fail requires at least one evidence-linked finding")
    limitations = _strings(v["limitations"])
    if v["status"] == "uncertain" and not limitations:
        raise DecisionError("Uncertain verdict must explain its limitations")
    return Verdict(v["status"], _unit(v["confidence"]), tuple(findings), _text(v["summary"]), limitations)
