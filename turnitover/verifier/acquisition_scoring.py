"""Score frozen active-observation runs against a hash-bound private exposure sidecar."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from turnitover.core.jsonio import write_json_atomic


def score_acquisition(runs: dict[str, Path], sidecar_path: Path, output_path: Path) -> dict:
    """Compute exposure curves without inferring visibility from diagnostic Gold.

    The sidecar is private evaluation data.  It maps every observed action prefix to
    whether the case's Gold defect is exposed, and supplies the privileged Oracle
    curve.  Missing prefixes fail closed rather than being treated as unexposed.
    """
    if output_path.exists():
        raise ValueError("Scoring output must not exist")
    sidecar_bytes = sidecar_path.read_bytes()
    sidecar = json.loads(sidecar_bytes)
    if sidecar.get("schema_version") != 1:
        raise ValueError("Unsupported exposure sidecar schema")
    expected_manifest_sha = _required_string(sidecar, "benchmark_manifest_sha256")
    cases = _case_index(sidecar.get("cases"))
    oracle = _curve(sidecar.get("oracle"), "oracle")
    report_runs = {}
    for label, root in runs.items():
        manifest = json.loads((root / "batch-manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("score_eligible") is not True:
            raise ValueError(f"{label}: expected a complete, full frozen batch")
        if manifest.get("benchmark_manifest_sha256") != expected_manifest_sha:
            raise ValueError(f"{label}: batch and exposure sidecar use different benchmarks")
        case_ids = {task.get("id") for task in manifest.get("tasks", [])}
        if case_ids != set(cases):
            raise ValueError(f"{label}: batch case IDs do not match exposure sidecar")
        budget = manifest.get("config", {}).get("budget")
        if type(budget) is not int or budget < 1:
            raise ValueError(f"{label}: batch needs a positive configured budget")
        report_runs[label] = _score_run(root, cases, budget)
    report = {
        "schema_version": 1, "status": "complete",
        "benchmark_manifest_sha256": expected_manifest_sha,
        "exposure_sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
        "runs": report_runs, "oracle": oracle,
    }
    write_json_atomic(output_path, report)
    return report


def _score_run(root: Path, cases: dict[str, dict], budget: int) -> dict:
    per_case = []
    for case_id, case in cases.items():
        result = json.loads((root / case_id / "result.json").read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            raise ValueError(f"{case_id}: incomplete result")
        actions = [_action_key(item) for item in result.get("trajectory", [])]
        prefix_map = {tuple(item["actions"]): item["exposed"] for item in case["prefixes"]}
        prefix = tuple(actions)
        if prefix not in prefix_map:
            raise ValueError(f"{case_id}: missing exposure record for prefix {prefix!r}")
        per_case.append({"id": case_id, "multi_joint": case["multi_joint"],
                         "actions": actions, "exposed": prefix_map[prefix]})
    return {"n": len(per_case), "budget": budget,
            "exposure_rate": _rate(per_case),
            "multi_joint_exposure_rate": _rate([row for row in per_case if row["multi_joint"]]),
            "per_case": per_case}


def _rate(rows: list[dict]) -> float | None:
    return sum(row["exposed"] for row in rows) / len(rows) if rows else None


def _case_index(value) -> dict[str, dict]:
    if not isinstance(value, list) or not value:
        raise ValueError("Exposure sidecar needs non-empty cases")
    indexed = {}
    for case in value:
        case_id = _required_string(case, "id")
        if case_id in indexed or type(case.get("multi_joint")) is not bool:
            raise ValueError("Invalid or duplicate exposure case")
        prefixes = case.get("prefixes")
        if not isinstance(prefixes, list) or not prefixes:
            raise ValueError(f"{case_id}: missing exposure prefixes")
        for prefix in prefixes:
            if (not isinstance(prefix, dict) or not isinstance(prefix.get("actions"), list)
                    or any(not isinstance(action, str) for action in prefix["actions"])
                    or type(prefix.get("exposed")) is not bool):
                raise ValueError(f"{case_id}: invalid exposure prefix")
        indexed[case_id] = case
    return indexed


def _curve(value, name: str) -> dict[str, float]:
    if not isinstance(value, dict) or not isinstance(value.get("exposure_rate"), dict):
        raise ValueError(f"Exposure sidecar needs {name} curve")
    curve = value["exposure_rate"]
    if not curve or any(not isinstance(key, str) or type(rate) not in (int, float) for key, rate in curve.items()):
        raise ValueError(f"Invalid {name} curve")
    return {key: float(rate) for key, rate in curve.items()}


def _required_string(value: dict, key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"Missing {key}")
    return result


def _action_key(observation: dict) -> str:
    action = observation.get("action")
    if not isinstance(action, dict):
        raise ValueError("Observation lacks serialized action")
    kind = action.get("__type__")
    if kind == "RequestView" and isinstance(action.get("view_id"), str):
        return f"view:{action['view_id']}"
    if kind == "ActuateJoint" and isinstance(action.get("joint_id"), str) and isinstance(action.get("detent"), str):
        return f"joint:{action['joint_id']}:{action['detent']}"
    if kind == "QueryRuntime" and isinstance(action.get("property"), str):
        return f"runtime:{action['property']}"
    raise ValueError("Unsupported serialized observation action")
