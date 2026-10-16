"""Re-audit saved Kimi Episodes and export a new, hash-verified Round2--5 dataset."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shlex

from scripts.hash_teacher_export import write_hash_manifest
from turnitover.teacher.audit import audit_episode
from turnitover.teacher.export import export_episodes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "data/benchmarks/partnet-mobility-formal-v1/prepared"
DEFAULT_ANNOTATIONS = (ROOT / "data/benchmarks/partnet-mobility-formal-v1/"
                       "oracle-first-failure-v4/annotations.private.jsonl")
DEFAULT_SOURCE_ROOT = ROOT / "output/interactive-teacher"
DEFAULT_BASELINES = (
    "kimi-round2-sft-export-v2", "kimi-round3-sft-export-perceptible",
    "kimi-round4-sft-export", "kimi-round5-sft-export",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _by_id(paths: list[Path]) -> dict[str, dict]:
    return {row["id"]: row for path in paths for row in _rows(path)}


def _discover(root: Path, rounds: tuple[int, ...]) -> list[Path]:
    sources = []
    for round_number in rounds:
        sources.extend(path.parent for path in root.glob(f"kimi-round{round_number}-*/episode.json"))
    return sorted(set(sources))


def _dataset_stats(datasets: list[Path]) -> dict:
    stats = Counter()
    statuses, reasons, fail_assets = Counter(), Counter(), set()
    for dataset in datasets:
        if not (dataset / "manifest.json").is_file():
            continue
        manifest = json.loads((dataset / "manifest.json").read_text())
        stats.update(manifest.get("counts", {}))
        for split in ("train", "validation"):
            path = dataset / f"{split}.diagnosis.jsonl"
            if not path.is_file():
                continue
            for row in _rows(path):
                verdict = json.loads(row["messages"][1]["content"])["verdict"]
                statuses[verdict["status"]] += 1
                if row.get("audit_conclusion", {}).get("diagnosis_outcome") == "correct_fail":
                    identity = row["data_identity"]
                    fail_assets.add((identity.get("source_dataset"), identity.get("source_asset_id")))
        rejected = dataset / "rejected.jsonl"
        if rejected.is_file():
            for row in _rows(rejected):
                reasons.update(row.get("reasons") or [])
    return {"datasets": [str(path) for path in datasets], "counts": dict(stats),
            "diagnosis_statuses": dict(statuses), "correct_fail_independent_assets": len(fail_assets),
            "rejection_reasons": dict(reasons)}


def _validate_export(output: Path) -> dict:
    rows = {kind: [] for kind in ("actions", "diagnosis")}
    for split in ("train", "validation"):
        for kind in rows:
            rows[kind].extend(_rows(output / f"{split}.{kind}.jsonl"))
    image_bindings, image_hashes = 0, set()
    for row in rows["actions"] + rows["diagnosis"]:
        if not (len(row["images"]) == len(row["image_sha256"]) == len(row["image_provenance"])):
            raise ValueError(f"Image provenance roster differs: {row['id']}")
        for relative, expected, provenance in zip(
                row["images"], row["image_sha256"], row["image_provenance"], strict=True):
            actual = _sha(output / relative)
            if actual != expected or provenance["source_sha256"] != expected \
                    or provenance["exported_sha256"] != expected or provenance["exported"] != relative:
                raise ValueError(f"Image identity mismatch: {row['id']} {relative}")
            image_bindings += 1
            image_hashes.add(actual)

    split_rows = {split: [row for kind in rows for row in _rows(output / f"{split}.{kind}.jsonl")]
                  for split in ("train", "validation")}
    isolation = {}
    for key in ("split_group", "source_asset_id", "reference_sha"):
        left = {row["data_identity"].get(key) for row in split_rows["train"]}
        right = {row["data_identity"].get(key) for row in split_rows["validation"]}
        overlap = sorted((left & right) - {None})
        isolation[key] = {"overlap": overlap, "ok": not overlap}
        if overlap:
            raise ValueError(f"Train/Validation leakage in {key}: {overlap}")

    statuses, fail_assets, source_assets, candidate_versions, episode_ids = Counter(), set(), set(), set(), set()
    source_identity = {}
    for row in rows["actions"] + rows["diagnosis"]:
        identity = row["data_identity"]
        asset = (identity.get("source_dataset"), identity.get("source_asset_id"))
        source_assets.add(asset)
        candidate_versions.add((asset, identity.get("candidate_sha")))
        episode_ids.add(row["episode_id"])
        fingerprint = (identity.get("source_identity_sha256"), identity.get("source_content_sha256"))
        previous = source_identity.setdefault(asset, fingerprint)
        if previous != fingerprint:
            raise ValueError(f"One source asset has inconsistent identity: {asset}")
    for row in rows["diagnosis"]:
        verdict = json.loads(row["messages"][1]["content"])["verdict"]
        statuses[verdict["status"]] += 1
        if row["audit_conclusion"]["diagnosis_outcome"] == "correct_fail":
            identity = row["data_identity"]
            fail_assets.add((identity.get("source_dataset"), identity.get("source_asset_id")))
    reasons = Counter(reason for row in _rows(output / "rejected.jsonl")
                      for reason in (row.get("reasons") or []))
    return {"image_bindings": image_bindings, "unique_image_sha256": len(image_hashes),
            "image_identity": "all source, row, and exported SHA-256 values agree",
            "split_isolation": isolation, "action_rows": len(rows["actions"]),
            "diagnosis_rows": len(rows["diagnosis"]), "diagnosis_statuses": dict(statuses),
            "independent_source_assets": len(source_assets),
            "source_assets": sorted(f"{source}:{asset}" for source, asset in source_assets),
            "candidate_versions": len(candidate_versions), "episode_ids": len(episode_ids),
            "source_identity_consistent": True,
            "correct_fail_independent_assets": len(fail_assets),
            "correct_fail_assets": sorted(f"{source}:{asset}" for source, asset in fail_assets),
            "rejection_reasons": dict(reasons)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--oracle-annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--manifest-root", type=Path,
                        help="Use a field-preserving superseding witness manifest when present")
    parser.add_argument("--round", dest="rounds", type=int, action="append", choices=(2, 3, 4, 5))
    parser.add_argument("--audit-name", default="audit.correctness-v1.json")
    parser.add_argument("--diagnosis-objective", choices=("first_failure", "comprehensive_within_budget"),
                        default="first_failure")
    args = parser.parse_args()
    rounds = tuple(args.rounds or (2, 3, 4, 5))
    sources = _discover(args.source_root, rounds)
    if not sources:
        raise ValueError("No saved Kimi Episodes found")
    if args.out.exists():
        raise ValueError("Output must not exist")

    gold = _by_id([args.benchmark / "train.gold.jsonl", args.benchmark / "validation.gold.jsonl"])
    annotations = _by_id([args.oracle_annotations])
    cases = []
    for source in sources:
        episode_path = source / "episode.json"
        episode = json.loads(episode_path.read_text())
        uid = episode["episode_id"]
        manifest = None
        manifest_path = None
        superseding = args.manifest_root / uid / "manifest.json" if args.manifest_root else None
        recorded = Path((episode.get("frozen_witness_manifest") or {}).get("path", ""))
        if superseding and superseding.is_file():
            manifest_path = superseding
        elif recorded.is_file():
            manifest_path = recorded
        if manifest_path:
            manifest = json.loads(manifest_path.read_text())
        audit = audit_episode(source, gold[uid], annotations.get(uid), manifest=manifest,
                              audit_name=args.audit_name, diagnosis_objective=args.diagnosis_objective)
        evidence = audit["independent_evidence_audit"]
        cases.append({"source": str(source), "round": int(source.name.split("-round", 1)[1][0]),
                      "episode_id": uid, "episode_sha256": _sha(episode_path),
                      "original_audit_sha256": _sha(source / "audit.json"),
                      "reaudit_sha256": _sha(source / args.audit_name),
                      "manifest": str(manifest_path) if manifest_path else None,
                      "manifest_sha256": _sha(manifest_path) if manifest_path else None,
                      "split": episode.get("data_identity", {}).get("split"),
                      "split_group": episode.get("data_identity", {}).get("split_group"),
                      "source_asset_id": episode.get("data_identity", {}).get("source_asset_id"),
                      "termination": episode.get("termination"), "pipeline_ok": audit["pipeline_ok"],
                      "accepted": evidence["accepted"], "status": evidence["status"],
                      "diagnosis_outcome": evidence["diagnosis_outcome"],
                      "rejection_reasons": evidence["rejection_reasons"]})

    manifest = export_episodes(sources, args.out, args.audit_name)
    validation = _validate_export(args.out)
    baseline_paths = [args.source_root / name for name in DEFAULT_BASELINES]
    report = {"schema_version": 1, "status": "complete", "scope": "saved_kimi_round2_through_round5",
              "diagnosis_objective": args.diagnosis_objective,
              "no_paid_api_calls": True, "no_new_collection": True, "no_sft_training": True,
              "raw_episode_count": len(sources), "round_counts": dict(Counter(row["round"] for row in cases)),
              "before": _dataset_stats(baseline_paths),
              "after": {"manifest_counts": manifest["counts"], **validation}, "cases": cases}
    (args.out / "audit-report.json").write_text(json.dumps(report, indent=2))
    command = ["python", "-m", "scripts.reexport_interactive_teacher_rounds",
               "--source-root", str(args.source_root), "--out", str(args.out),
               "--diagnosis-objective", args.diagnosis_objective]
    if args.manifest_root:
        command.extend(("--manifest-root", str(args.manifest_root)))
    (args.out / "reproduction.json").write_text(json.dumps({
        "command": " ".join(shlex.quote(part) for part in command),
        "finalize_hashes_after_loader": (
            f"python -m scripts.hash_teacher_export --dataset {shlex.quote(str(args.out))}"),
        "audit_name": args.audit_name, "rounds": list(rounds),
        "benchmark": str(args.benchmark), "oracle_annotations": str(args.oracle_annotations),
        "baseline_datasets": [str(path) for path in baseline_paths],
    }, indent=2))
    write_hash_manifest(args.out)
    print(json.dumps({"out": str(args.out), "raw_episodes": len(sources), **validation}, indent=2))


if __name__ == "__main__":
    main()
