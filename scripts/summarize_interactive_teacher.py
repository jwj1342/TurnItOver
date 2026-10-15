"""Tabulate saved interactive-teacher Episodes and their independent audits (no model or browser calls)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from turnitover.teacher.structure import moving_children, part_relation


def summarize(source: Path, audit_name: str, manifest_root: Path | None = None) -> dict:
    episode = json.loads((source / "episode.json").read_text())
    audit = json.loads((source / audit_name).read_text())
    evidence = audit.get("independent_evidence_audit", {})
    manifest = json.loads((manifest_root / episode["episode_id"] / "manifest.json" if manifest_root
                           else Path(episode["frozen_witness_manifest"]["path"])).read_text())
    hierarchy = (manifest.get("rest_hierarchy") or {}).get("candidate")
    verdict = episode.get("verdict") or {}
    identity = episode.get("data_identity", {})
    findings = []
    for finding in verdict.get("findings", []):
        child = moving_children(hierarchy["root"], finding["joint_id"]) if hierarchy and finding.get("joint_id") else None
        findings.append({key: finding.get(key) for key in ("defect_id", "joint_id", "parts", "evidence_steps")}
                        | {"parts_equal_visible_joint_child": None if child is None
                           else sorted(finding.get("parts", [])) == child,
                           "part_relation": part_relation(hierarchy["root"], finding["joint_id"], finding.get("parts", []))
                           if hierarchy and finding.get("joint_id") else None})
    usage = episode.get("token_usage", {})
    return {
        "dir": str(source), "id": episode["episode_id"], "asset_id": identity.get("asset_id"),
        "mode": identity.get("experiment_mode"), "reference_protocol": episode.get("reference_protocol"),
        "common_input": identity.get("common_input", "none"),
        "prompt_addenda": identity.get("prompt_addenda", []),
        "objective_defect_ids": evidence.get("objective_defect_ids"),
        "termination": episode.get("termination"),
        "actions": [{"step": row["step"], "source": row.get("decision_source"), "action": row["decision"]}
                    for row in episode.get("trajectory", [])],
        "teacher_queried_hierarchy": any(row.get("decision_source") == "teacher"
                                         and row["decision"].get("property") == "hierarchy"
                                         for row in episode.get("trajectory", [])),
        "status": verdict.get("status"), "findings": findings, "summary": verdict.get("summary"),
        "localization": evidence.get("localization"),
        "evidence_standard": evidence.get("evidence_standard"),
        "actions_legal": evidence.get("actions_legal"), "evidence_sufficient": evidence.get("evidence_sufficient"),
        "objective_exposure_imperceptible_only": evidence.get("objective_exposure_imperceptible_only"),
        "first_new_evidence_step": next((row["step"] for row in evidence.get("action_audit", [])
                                         if row.get("new_objective_evidence")), None),
        "diagnosis_outcome": evidence.get("diagnosis_outcome"), "accepted": evidence.get("accepted"),
        "rejection_reasons": evidence.get("rejection_reasons"),
        "observation_sha256": [row.get("live_sha256") for row in evidence.get("action_audit", [])],
        "real_api_requests": episode.get("real_api_requests"), "tokens": usage.get("total_tokens"),
        "episode_sha256": hashlib.sha256((source / "episode.json").read_bytes()).hexdigest(),
        "audit_sha256": hashlib.sha256((source / audit_name).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", action="append", type=Path, required=True)
    parser.add_argument("--audit-name", default="audit.json")
    parser.add_argument("--manifest-root", type=Path, help="Superseding manifests (e.g. for the rest hierarchy)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [summarize(source, args.audit_name, args.manifest_root) for source in args.episode]
    args.out.write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(json.dumps({key: row[key] for key in ("dir", "status", "diagnosis_outcome", "accepted")}
                         | {"findings": row["findings"]}))
