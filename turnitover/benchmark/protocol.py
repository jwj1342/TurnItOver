"""Benchmark v1 protocol: difficulty tiers, input conditions and the prediction contract.

Pure data and validation, no I/O. Everything here is serialised to ``benchmark/v1/protocol.json`` by the
freeze script, so the frozen protocol and the code that enforces it cannot drift apart silently.
"""
from __future__ import annotations

from dataclasses import dataclass

from turnitover.taxonomy import load_taxonomy

BENCHMARK_VERSION = "benchmark-v1"
PROTOCOL_SCHEMA_VERSION = 1

DECISIONS = ("pass", "fail", "uncertain", "continue")
ABSTAIN = ("uncertain", "continue")
IMAGE_ROLES = ("reference_rest", "reference_limit", "candidate_rest", "candidate_limit")
LABEL_CLEAN = "clean"

# Operational definitions. ``min_joints``/``max_joints`` count articulated joints of the source object.
DIFFICULTIES = {
    "easy": {
        "object": "single articulated joint", "min_joints": 1, "max_joints": 1,
        "defects": ["kinematics.joint_axis"], "n_errors": 1,
        "visibility": "Target joint moves visibly rest->limit in the frozen camera; reference and candidate "
                      "differ at limit by the Level 0 pixel gates (human-reviewed, no occlusion or clipping).",
        "example": "A plain door swings up instead of sideways.",
    },
    "medium": {
        "object": "two or more articulated joints", "min_joints": 2, "max_joints": None,
        "defects": ["kinematics.joint_axis", "structure.part_offset"], "n_errors": 1,
        "visibility": "Same visibility gates as easy for joint_axis. structure.part_offset on a single-joint "
                      "object is also medium and is static_visible. Part-offset samples do not exist yet.",
        "example": "One door of a multi-door wardrobe swings up instead of sideways.",
    },
    "hard": {
        "object": "two or more articulated joints", "min_joints": 2, "max_joints": None,
        "defects": ["kinematics.joint_axis", "structure.part_offset"], "n_errors": 1,
        "visibility": "Target part is occluded or its motion is below the default-view pixel gate, so the "
                      "decision needs a view or state beyond the default four images. Overlap with medium "
                      "is decided by measured observation difficulty, not by defect type.",
        "example": "An occluded door of a multi-door wardrobe rotates about the wrong axis.",
    },
    "expert": {
        "object": "two or more articulated joints", "min_joints": 2, "max_joints": None,
        "defects": ["kinematics.joint_axis", "structure.part_offset"], "n_errors": 2,
        "visibility": "Two distinct injected errors, each independently supported by observed evidence.",
        "example": "One door is offset while another door rotates about the wrong axis.",
    },
}
CLEAN_CONTROL = "Every tier carries clean controls: the same object, no corruption, gold decision pass."

# Which manifest fields each baseline may read. Gold is never listed; ``fields`` is the whole boundary.
# ``n_evidence`` bounds the legal evidence indices a prediction may cite under that condition.
CONDITIONS = {
    "rgb": {
        "fields": ["sample_id", "prompt", "context", "images"], "n_evidence": 4, "evidence_sufficient": True,
        "note": "Four frozen RGB frames. No spec/code, no pixel statistics, no gold.",
    },
    "no_image": {
        "fields": ["sample_id", "prompt", "context"], "n_evidence": 0, "evidence_sufficient": False,
        "note": "Text only. Evidence is insufficient by construction, so the correct output is "
                "uncertain/continue; any pass or fail is an unsupported claim.",
    },
    "pixel_diff": {
        "fields": ["sample_id", "prompt", "context", "images"], "n_evidence": 4, "evidence_sufficient": True,
        "note": "Deterministic rule on the same four frames. Thresholds are fitted on Train only and frozen "
                "before Validation is touched. No model, no spec, no gold.",
    },
    "code_read": {
        "fields": ["sample_id", "prompt", "context", "code"], "n_evidence": 2, "evidence_sufficient": True,
        "note": "Reference then candidate AssetSpec JSON (index 0, 1). No images, no gold.",
    },
}

# Defects whose localisation target is a joint; every other defect is localised by its part set.
JOINT_LEVEL_LAYERS = ("kinematics",)
# Evidence a finding must cite to be consistent: indices into the condition's evidence list.
MOTION_EVIDENCE = {"rgb": (1, 3), "pixel_diff": (1, 3), "code_read": (0, 1), "no_image": ()}


def is_joint_level(defect_id: str) -> bool:
    return defect_id.split(".", 1)[0] in JOINT_LEVEL_LAYERS


def target_key(defect_id: str, joint_id: str | None, parts) -> tuple:
    """Localisation identity: (defect, joint) for kinematics, (defect, sorted parts) otherwise."""
    return (defect_id, joint_id) if is_joint_level(defect_id) else (defect_id, tuple(sorted(parts)))


@dataclass(frozen=True)
class Finding:
    defect_id: str
    joint_id: str | None
    parts: tuple[str, ...]
    evidence: tuple[int, ...]

    @property
    def key(self) -> tuple:
        return target_key(self.defect_id, self.joint_id, self.parts)


@dataclass(frozen=True)
class Prediction:
    case_id: str
    decision: str
    findings: tuple[Finding, ...]


class InvalidOutput(ValueError):
    pass


def _is_str_list(value) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


def parse_prediction(obj, *, condition: str, joint_ids, part_ids, taxonomy_ids=None) -> Prediction:
    """Strictly validate one prediction object; raise InvalidOutput on any deviation.

    Contract: ``{"case_id", "decision", "findings": [{"defect_id", "joint_id", "parts", "evidence"}]}``.
    ``fail`` needs at least one finding; every other decision needs none. No field is repaired or guessed.
    """
    taxonomy_ids = taxonomy_ids if taxonomy_ids is not None else set(load_taxonomy().ids)
    if not isinstance(obj, dict) or set(obj) != {"case_id", "decision", "findings"}:
        raise InvalidOutput("top-level keys must be exactly case_id, decision, findings")
    if not isinstance(obj["case_id"], str):
        raise InvalidOutput("case_id must be a string")
    decision, raw = obj["decision"], obj["findings"]
    if decision not in DECISIONS:
        raise InvalidOutput(f"decision must be one of {DECISIONS}")
    if not isinstance(raw, list):
        raise InvalidOutput("findings must be a list")
    if (decision == "fail") != bool(raw):
        raise InvalidOutput("fail requires findings; pass/uncertain/continue forbid them")
    n_evidence = CONDITIONS[condition]["n_evidence"]
    findings = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"defect_id", "joint_id", "parts", "evidence"}:
            raise InvalidOutput("finding keys must be exactly defect_id, joint_id, parts, evidence")
        defect_id, joint_id, parts, evidence = item["defect_id"], item["joint_id"], item["parts"], item["evidence"]
        if defect_id not in taxonomy_ids:
            raise InvalidOutput(f"unknown defect_id {defect_id!r}")
        if is_joint_level(defect_id):
            if joint_id not in joint_ids:
                raise InvalidOutput("joint-level finding needs a joint_id from the case context")
        elif joint_id is not None:
            raise InvalidOutput("non-joint finding must set joint_id to null")
        if not _is_str_list(parts) or any(p not in part_ids for p in parts):
            raise InvalidOutput("parts must be a list of case part ids")
        if (not isinstance(evidence, list) or not evidence or any(type(e) is not int for e in evidence)
                or any(not 0 <= e < n_evidence for e in evidence)):
            raise InvalidOutput(f"evidence must be a nonempty list of indices in [0, {n_evidence})")
        findings.append(Finding(defect_id, joint_id, tuple(parts), tuple(sorted(set(evidence)))))
    return Prediction(obj["case_id"], decision, tuple(findings))


def finding_consistent(finding: Finding, condition: str) -> bool:
    """A kinematics claim must cite the motion pair; other claims only need valid, nonempty evidence."""
    if finding.defect_id.startswith("kinematics."):
        return set(MOTION_EVIDENCE[condition]) <= set(finding.evidence)
    return bool(finding.evidence)


def protocol_document() -> dict:
    return {"schema_version": PROTOCOL_SCHEMA_VERSION, "benchmark": BENCHMARK_VERSION,
            "difficulties": DIFFICULTIES, "clean_control": CLEAN_CONTROL, "conditions": CONDITIONS,
            "decisions": list(DECISIONS), "abstain": list(ABSTAIN), "image_roles": list(IMAGE_ROLES),
            "motion_evidence": {k: list(v) for k, v in MOTION_EVIDENCE.items()},
            "prediction_format": {
                "case_id": "str", "decision": "pass|fail|uncertain|continue",
                "findings": [{"defect_id": "taxonomy id", "joint_id": "str|null (required for kinematics.*)",
                              "parts": ["part id"], "evidence": ["index into the condition's evidence list"]}]},
            "metrics": {
                "macro_f1": "mean per-class F1 over {clean} U defect ids in gold or predicted; pass predicts clean, "
                            "fail predicts its finding defect ids, abstain/invalid predict nothing",
                "localization_f1": "micro F1 over localisation keys (defect+joint for kinematics, defect+parts "
                                   "otherwise) pooled over all samples, from valid fail predictions",
                "exact_verdict": "decision matches expected and finding keys equal gold keys; under insufficient "
                                 "evidence or gold uncertain the expected decision is abstain",
                "evidence_consistency": "share of fail predictions whose findings all cite valid evidence "
                                        "including the motion pair for kinematics claims",
                "false_pass": "pass predictions / gold-defect samples",
                "invalid_output": "invalid or missing predictions / samples"}}
