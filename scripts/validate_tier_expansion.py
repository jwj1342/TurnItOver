"""Validate object-level pairing, split quotas, and provenance for a tier expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


if __package__:
    from .tier_expansion_evidence import validate_case_evidence
    from .tier_expansion_pins import validate_pins
else:
    from tier_expansion_evidence import validate_case_evidence
    from tier_expansion_pins import validate_pins


TIERS = ("easy", "medium", "hard")
SPLITS = ("train", "validation", "test")
KINDS = {"clean", "defect"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _baseline_digest(rows: list[dict]) -> str:
    inventory = []
    for row in rows:
        if row["membership"] != "baseline":
            continue
        cases = [
            {"case_id": case["case_id"], "kind": case["kind"], "defect_id": case.get("defect_id")}
            for case in row["cases"] if case["membership"] == "baseline"
        ]
        inventory.append({
            "source_identity": row["source_identity"],
            "original_split": row.get("original_split"),
            "cases": sorted(cases, key=lambda case: case["case_id"]),
        })
    inventory.sort(key=lambda row: row["source_identity"])
    payload = json.dumps(inventory, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate(manifest_path: Path, contract_path: Path) -> dict:
    contract = _read_json(contract_path)
    manifest = _read_json(manifest_path)
    if manifest.get("version") != 1 or not isinstance(manifest.get("objects"), list):
        raise ValueError("Expansion manifest must have version 1 and an objects list")
    if contract.get("status") != "candidate_contract":
        raise ValueError("Expansion contract is not an active candidate contract")

    objects = manifest["objects"]
    frozen_identities = manifest.get("frozen_evaluation_source_identities")
    if not isinstance(frozen_identities, list) or any(not isinstance(value, str) for value in frozen_identities):
        raise ValueError("Manifest must include the frozen evaluation source identity list")
    if len(set(frozen_identities)) != len(frozen_identities):
        raise ValueError("Frozen evaluation source identity list contains duplicates")
    frozen = contract["frozen_evaluation_exclusion"]
    frozen_payload = "\n".join(sorted(frozen_identities))
    if frozen_identities:
        frozen_payload += "\n"
    frozen_digest = hashlib.sha256(frozen_payload.encode()).hexdigest()
    if len(frozen_identities) != frozen["objects"] or frozen_digest != frozen["source_identity_set_sha256"]:
        raise ValueError("Frozen evaluation exclusion set differs from its pinned inventory")
    frozen_set = set(frozen_identities)
    by_tier = {tier: [] for tier in TIERS}
    source_occurrences: dict[str, list[dict]] = defaultdict(list)
    fingerprint_occurrences: dict[str, list[dict]] = defaultdict(list)
    case_ids: set[str] = set()
    case_totals = Counter()
    case_kind_totals = Counter()
    baseline_objects = Counter()
    baseline_cases = Counter()
    split_totals = {tier: Counter() for tier in TIERS}
    added_cases = 0

    for row in objects:
        if not isinstance(row, dict):
            raise ValueError("Each object entry must be an object")
        tier, split = row.get("tier"), row.get("split")
        source = row.get("source_identity")
        fingerprint = row.get("geometry_fingerprint")
        membership = row.get("membership")
        original_split = row.get("original_split")
        cases = row.get("cases")
        if tier not in TIERS or split not in SPLITS:
            raise ValueError(f"Invalid tier or split for object: {row}")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Every object requires a stable source_identity")
        if not isinstance(fingerprint, str) or not SHA256.fullmatch(fingerprint):
            raise ValueError(f"Invalid geometry_fingerprint for {source}")
        if membership not in ("baseline", "new"):
            raise ValueError(f"Invalid object membership for {source}")
        if membership == "new" and source in frozen_set:
            raise ValueError(f"New object overlaps frozen evaluation v2: {source}")
        if membership == "baseline" and original_split is not None:
            if original_split not in SPLITS or split != original_split:
                raise ValueError(f"Baseline split changed for {source}")
        if membership == "new" and original_split is not None:
            raise ValueError(f"New object has an original split: {source}")
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"Object has no cases: {source}")

        kinds = set()
        for case in cases:
            if not isinstance(case, dict):
                raise ValueError(f"Invalid case entry for {source}")
            case_id, kind = case.get("case_id"), case.get("kind")
            if not isinstance(case_id, str) or not case_id.strip() or case_id in case_ids:
                raise ValueError(f"Missing or duplicate case_id for {source}: {case_id}")
            if kind not in KINDS:
                raise ValueError(f"Case kind must be clean or defect: {case_id}")
            defect_id = case.get("defect_id")
            if (kind == "clean" and defect_id is not None) or (kind == "defect" and not isinstance(defect_id, str)):
                raise ValueError(f"Invalid defect_id for {case_id}")
            case_ids.add(case_id)
            kinds.add(kind)
            case_totals[tier] += 1
            case_kind_totals[(tier, kind)] += 1
            case_membership = case.get("membership")
            if case_membership not in ("baseline", "added"):
                raise ValueError(f"Invalid case membership: {case_id}")
            if case_membership == "added":
                added_cases += 1
                if case.get("automatic_status") != "passed":
                    raise ValueError(f"Added case lacks automatic pass: {case_id}")
                if not SHA256.fullmatch(str(case.get("automatic_evidence_sha256", ""))):
                    raise ValueError(f"Added case lacks automatic evidence hash: {case_id}")
                review = case.get("difficulty_review")
                if not isinstance(review, dict) or review.get("status") != "passed":
                    raise ValueError(f"Added case lacks human difficulty approval: {case_id}")
                expected_difficulty = "clean" if kind == "clean" else tier
                if review.get("verdict") != kind or review.get("difficulty") != expected_difficulty:
                    raise ValueError(f"Human review disagrees with case labels: {case_id}")
                if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
                    raise ValueError(f"Added case lacks a human reviewer: {case_id}")
                if not isinstance(review.get("reviewed_at"), str) or not review["reviewed_at"].strip():
                    raise ValueError(f"Added case lacks a review date: {case_id}")
            elif membership == "new":
                raise ValueError(f"New object contains a baseline case: {case_id}")
            else:
                baseline_cases[tier] += 1
        if not KINDS.issubset(kinds):
            raise ValueError(f"Object lacks a Clean or defect case: {source}")

        normalized = {**row, "cases": cases}
        by_tier[tier].append(normalized)
        split_totals[tier][split] += 1
        if membership == "baseline":
            baseline_objects[tier] += 1
            source_occurrences[source].append(normalized)
            fingerprint_occurrences[fingerprint].append(normalized)
        else:
            source_occurrences[source].append(normalized)
            fingerprint_occurrences[fingerprint].append(normalized)

    for identity, rows in source_occurrences.items():
        if len({row['split'] for row in rows}) > 1:
            raise ValueError(f'Source identity crosses dataset splits: {identity}')
        if len(rows) > 1:
            tiers = [row["tier"] for row in rows]
            allowed_pairs = {
                frozenset(pair)
                for pair in contract["baseline_reuse_allowlist"]["allowed_tier_pairs"]
            }
            permitted = (
                all(row["membership"] == "baseline" for row in rows)
                and len(rows) == 2
                and frozenset(tiers) in allowed_pairs
            )
            if not permitted:
                raise ValueError(f"Source identity overlaps outside the baseline allowlist: {identity}")
        fingerprints = {row["geometry_fingerprint"] for row in rows}
        if len(fingerprints) > 1:
            raise ValueError(f"Source identity maps to multiple geometries: {identity}")
    for fingerprint, rows in fingerprint_occurrences.items():
        if len(rows) > 1 and len({row["source_identity"] for row in rows}) > 1:
            raise ValueError(f"Duplicate geometry fingerprint: {fingerprint}")
        if len(rows) > 1 and not all(row["membership"] == "baseline" for row in rows):
            raise ValueError(f"New object reuses an existing geometry: {fingerprint}")

    shared_baseline = sum(
        len(rows) == 2 and all(row["membership"] == "baseline" for row in rows)
        for rows in source_occurrences.values()
    )
    expected_shared = contract["baseline_reuse_allowlist"]["expected_shared_identities"]
    if shared_baseline != expected_shared:
        raise ValueError(f"Baseline shared identities {shared_baseline}; expected {expected_shared}")
    shared_ids = sorted(
        identity for identity, rows in source_occurrences.items()
        if len(rows) == 2 and all(row["membership"] == "baseline" for row in rows)
    )
    shared_digest = hashlib.sha256(("\n".join(shared_ids) + ("\n" if shared_ids else "")).encode()).hexdigest()
    if shared_digest != contract["baseline_reuse_allowlist"]["identity_set_sha256"]:
        raise ValueError("Baseline shared identity set differs from the immutable allowlist")

    tier_reports = {}
    for tier in TIERS:
        minimum = contract["tier_minima"][tier]
        rows = by_tier[tier]
        split_counts = split_totals[tier]
        baseline = contract["baseline_counts"][tier]
        if baseline_objects[tier] != baseline["objects"] or baseline_cases[tier] != baseline["cases"]:
            raise ValueError(
                f"{tier} immutable baseline is {baseline_objects[tier]} objects/"
                f"{baseline_cases[tier]} cases; expected {baseline['objects']}/{baseline['cases']}"
            )
        if _baseline_digest(rows) != baseline["inventory_sha256"]:
            raise ValueError(f"{tier} immutable baseline inventory changed")
        if len(rows) < minimum["objects"]:
            raise ValueError(f"{tier} has {len(rows)} objects; requires {minimum['objects']}")
        if case_totals[tier] < minimum["cases"]:
            raise ValueError(f"{tier} has {case_totals[tier]} cases; requires {minimum['cases']}")
        for split in SPLITS:
            if split_counts[split] < minimum[split]:
                raise ValueError(f"{tier} {split} has {split_counts[split]}; requires {minimum[split]}")
        tier_reports[tier] = {
            "objects": len(rows),
            "cases": case_totals[tier],
            "clean_cases": case_kind_totals[(tier, "clean")],
            "defect_cases": case_kind_totals[(tier, "defect")],
            "splits": dict(split_counts),
            "new_objects": sum(row["membership"] == "new" for row in rows),
        }

    unique_objects = len(source_occurrences)
    required_unique = contract["global_minima"]["independent_source_objects"]
    if unique_objects < required_unique:
        raise ValueError(f"Global unique sources {unique_objects}; requires {required_unique}")
    validate_pins(manifest_path.parent, manifest, contract)
    for row in objects:
        for case in row["cases"]:
            if case["membership"] == "added":
                validate_case_evidence(manifest_path.parent, row, case)
    return {
        "status": "passed",
        "validation_scope": "inventory, custody baseline/v2 byte preservation, conversion-bound original spec/archive geometry and local evidence bindings; no human identity authentication or checker replay",
        "tiers": tier_reports,
        "independent_source_objects": unique_objects,
        "added_cases_with_automatic_and_human_pass": added_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=Path("benchmark/releases/tier-expansion-v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output must be a new path")
    report = validate(args.manifest, args.contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
