"""Validate a portable Hard candidate without programs or source archives."""

from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import re

if __package__:
    from .hard_bundle_contract import KIND_COUNTS, KINDS, SOURCE_KEYS, STATES
    from .hard_bundle_inventory import validate_manifest
    from .prepare_hard_review import validate_review_bundle
    from .validate_easy_bundle import _png_size
else:
    from hard_bundle_contract import KIND_COUNTS, KINDS, SOURCE_KEYS, STATES
    from hard_bundle_inventory import validate_manifest
    from prepare_hard_review import validate_review_bundle
    from validate_easy_bundle import _png_size


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _index(path):
    rows = _rows(path)
    result = {row["case_id"]: row for row in rows}
    if len(rows) != len(result):
        raise ValueError(f"Duplicate Hard cases: {path}")
    return result


def _contains_absolute_path(value):
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    return isinstance(value, str) and (
        value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value) is not None
    )


def validate_bundle(root: Path) -> dict:
    root = root.resolve()
    inventory = validate_manifest(root)
    public, gold, checks = (
        _index(root / name)
        for name in (
            "pool_manifest.jsonl",
            "private/gold.jsonl",
            "private/checks.jsonl",
        )
    )
    review, blank, returned = (
        _rows(root / name)
        for name in (
            "review_manifest.jsonl",
            "human_review.jsonl",
            "private/human_spot_review.jsonl",
        )
    )
    if not (
        len(public) == len(gold) == len(checks) == 75
        and set(public) == set(gold) == set(checks)
    ):
        raise ValueError("Hard candidate ledgers must align at 75 cases")
    if (
        len(review) != 10
        or len(blank) != 10
        or [row["case_id"] for row in review] != [row["case_id"] for row in blank]
        or [row["case_id"] for row in returned] != [row["case_id"] for row in review]
    ):
        raise ValueError("Hard random-ten ledgers disagree")
    kinds = Counter()
    identities = {key: set() for key in SOURCE_KEYS}
    media = set()
    for cid, row in gold.items():
        if (
            not re.fullmatch(r"hp-[0-9a-f]{14}", cid)
            or row.get("kind") not in KINDS
            or row.get("training_eligibility") != "pending_human_review"
            or row.get("construction_program") != "local_custody"
            or "program" in row
        ):
            raise ValueError("Hard identity, eligibility or custody differs")
        source = row["source"]
        if any(
            key.startswith("source_") and key in source
            for key in (
                "source_archive",
                "source_spec",
                "source_urdf",
                "source_catalog",
            )
        ):
            raise ValueError("Hard portable Gold leaks an absolute source path")
        for key in SOURCE_KEYS:
            value = source.get(key)
            if not value or value in identities[key]:
                raise ValueError(f"Duplicate or missing Hard source identity: {key}")
            identities[key].add(value)
        kinds[row["kind"]] += 1
        check = checks[cid]
        if (
            check.get("candidate_sha") != row["candidate_sha"]
            or check.get("reference_sha") != row["reference_sha"]
            or check.get("code_correct") is not False
            or check.get("used_by_student") is not False
            or check.get("privileged") is not True
            or check.get("clean_check", {}).get("code_correct") is not True
            or check.get("source_authenticity", {}).get("passed") is not True
        ):
            raise ValueError(
                "Hard checker, clean control or authenticity evidence differs"
            )
        if any(
            set(item) != {"logical_name", "sha256"}
            for item in check["source_authenticity"]["source_files"]
        ):
            raise ValueError("Hard authenticity files must use logical names only")
        p = public[cid]
        if (
            p.get("states") != list(STATES)
            or p.get("image_size") != [256, 256]
            or p.get("status") != "pending_human_review"
        ):
            raise ValueError("Hard public schema differs")
        for view in (p["initial_view"], p["supplementary_view"]):
            for role in ("reference", "candidate"):
                for state in STATES:
                    name = f"native/{cid}/{view}/{role}_{state}.png"
                    if _png_size(root / name) != (256, 256):
                        raise ValueError("Hard native image invalid")
                    media.add(name)
    if dict(kinds) != KIND_COUNTS:
        raise ValueError("Hard corruption balance differs")
    if (root / "train_manifest.jsonl").read_text().strip():
        raise ValueError("Hard candidate must not contain training exports")
    decisions = Counter(row.get("final_decision") or "pending" for row in returned)
    if decisions != Counter({"hard": 6, "clean": 3, "pending": 1}):
        raise ValueError("Frozen Hard human observation differs")
    metadata = json.loads((root / "source/metadata.json").read_text())
    strict = json.loads((root / "source/strict-audit.json").read_text())
    validation = json.loads((root / "source/validation.json").read_text())
    registry = json.loads((root / "construction_registry.json").read_text())
    if (
        metadata.get("benchmark_frozen") is not False
        or metadata.get("training_ready") is not False
        or strict.get("errors") != []
        or validation.get("objects") != 75
        or registry.get("availability") != "local_custody"
        or len(registry.get("programs", [])) != 75
    ):
        raise ValueError("Hard source evidence or custody boundary differs")
    review_report = validate_review_bundle(root / "review_bundle")
    allowed = {
        "README.md",
        "pool_manifest.jsonl",
        "review_manifest.jsonl",
        "human_review.jsonl",
        "sample.json",
        "train_manifest.jsonl",
        "review.html",
        "review-all.html",
        "private/gold.jsonl",
        "private/checks.jsonl",
        "private/human_spot_review.jsonl",
        "source/metadata.json",
        "source/validation.json",
        "source/strict-audit.json",
        "construction_registry.json",
        *media,
    }
    if {name for name in inventory if not name.startswith("review_bundle/")} != allowed:
        raise ValueError("Hard candidate contains unexpected or missing files")
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in (".json", ".jsonl"):
            continue
        values = (
            [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if path.suffix == ".jsonl"
            else json.loads(path.read_text())
        )
        if _contains_absolute_path(values):
            raise ValueError("Hard candidate leaks a local absolute path")
    return {
        "objects": 75,
        "cases": 75,
        "defect_cases": 75,
        "corruption_counts": dict(sorted(kinds.items())),
        "native_images": len(media),
        "human_spot": dict(decisions),
        "training_eligible_cases": 0,
        "archive_availability": "local_custody",
        "program_replay_performed": False,
        "strict_audit_status": strict["status"],
        "strict_audit_warnings": len(strict.get("warnings", [])),
        "review_bundle": review_report,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_bundle(args.root), indent=2, sort_keys=True))
