"""Assemble immutable v1 and new compositional v1.1 Hard candidates."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from scripts.hard_construction.hard_artifacts import page


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def rows(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values))


def copy_files(source: Path, out: Path, record: dict) -> None:
    for relative, digest in record["files"].items():
        src, dst = source / relative, out / relative
        if sha(src) != digest:
            raise ValueError("Changed source artifact: " + relative)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copyfile(src, dst)
        if sha(dst) != digest:
            raise ValueError("Copied artifact mismatch: " + relative)


def add_composite_clean(out: Path, record: dict) -> None:
    gold, public = record["gold"], record["public"]
    cid = "hp-" + hashlib.sha256(("hard-v11-clean:" + gold["case_id"]).encode()).hexdigest()[:14]
    clean_public = {**public, "case_id": cid, "native_root": f"native/{cid}"}
    for view in (public["initial_view"], public["supplementary_view"]):
        for state in public["states"]:
            src = out / public["native_root"] / view / f"reference_{state}.png"
            for role in ("reference", "candidate"):
                dst = out / f"native/{cid}/{view}/{role}_{state}.png"; dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst); record["files"][dst.relative_to(out).as_posix()] = sha(dst)
    program = out / f"private/programs/{cid}.json"; program.parent.mkdir(parents=True, exist_ok=True)
    spec = Path(gold["source"]["source_spec"]); shutil.copyfile(spec, program); record["files"][program.relative_to(out).as_posix()] = sha(program)
    clean_gold = {**gold, "case_id": cid, "fault_count": 0, "fault_signature": "clean", "defects": [],
                  "candidate_sha": gold["reference_sha"], "program": program.relative_to(out).as_posix(),
                  "expected_status": "pass", "training_eligibility": "pending_human_review"}
    clean_check = {"case_id": cid, **record["checks"]["clean_check"], "source_authenticity": record["checks"]["source_authenticity"]}
    clean_review = {"case_id": cid, "status": "pending_human_review", "reviewer": None, "reviewed_at": None,
                    "human_judgment": {"both_faults_identified": None, "notes": None}}
    record["paired_control"] = {"public": clean_public, "gold": clean_gold, "checks": clean_check, "review": clean_review}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-formal", type=Path, required=True)
    parser.add_argument("--composite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise ValueError("Output must be new")
    baseline_ids = {json.loads(path.read_text())["asset_id"] for path in (args.baseline_formal / "private/objects").glob("*/record.json")}
    old = [json.loads(path.read_text()) for path in (args.baseline / "private/objects").glob("*/record.json") if json.loads(path.read_text())["asset_id"] in baseline_ids]
    new = [json.loads(path.read_text()) for path in (args.composite / "private/objects").glob("*/record.json")]
    if len(old) != 75 or len(new) != 75: raise ValueError("Expected 75 v1 and 75 compositional objects")
    records = []
    for record in old:
        record = copy.deepcopy(record); copy_files(args.baseline, args.out, record)
        record["gold"].update(construction_track="single_motion_v1", fault_count=1)
        record["paired_control"]["gold"].update(construction_track="single_motion_v1", fault_count=0, faults=[])
        records.append(record)
    for record in new:
        record = copy.deepcopy(record); copy_files(args.composite, args.out, record); add_composite_clean(args.out, record); records.append(record)
    identities = [record["gold"]["source"]["source_identity"] for record in records]
    if len(set(identities)) != 150: raise ValueError("Source identity overlap")
    public = [entry["public"] for record in records for entry in (record, record["paired_control"])]
    gold = [entry["gold"] for record in records for entry in (record, record["paired_control"])]
    checks = [entry["checks"] for record in records for entry in (record, record["paired_control"])]
    reviews = [entry["review"] for record in records for entry in (record, record["paired_control"])]
    for record in records: write(args.out / f"private/objects/{record['asset_id']}/record.json", record)
    rows(args.out / "pool_manifest.jsonl", public); rows(args.out / "private/gold.jsonl", gold)
    rows(args.out / "private/checks.jsonl", checks); rows(args.out / "human_review.jsonl", reviews); rows(args.out / "train_manifest.jsonl", [])
    page(args.out, public, "review.html")
    report = {"status": "ready_pending_human_review", "policy": "hard-full-motion-formal-v1.1", "objects": 150,
              "cases": 300, "tracks": {"single_motion_v1": 75, "compositional_v1_1": 75},
              "training_eligible_cases": 0, "human_review_complete": False,
              "splits": {"train": 100, "validation": 20, "test": 30}}
    write(args.out / "validation.json", report)
    write(args.out / "manifest.json", {**report, "inputs": {str(args.baseline.resolve()): sha(args.baseline / "manifest.json"), str(args.composite.resolve()): sha(args.composite / "manifest.json")}, "files": {p.relative_to(args.out).as_posix(): sha(p) for p in args.out.rglob("*") if p.is_file() and p.name != "manifest.json"}})
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__": main()
