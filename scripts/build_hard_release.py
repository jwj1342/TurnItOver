"""Build code-pinned Hard candidate and answer-free review ZIP assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

if __package__:
    from .easy_release_archive import sha256, write_zip
    from .validate_hard_bundle import validate_bundle
else:
    from easy_release_archive import sha256, write_zip
    from validate_hard_bundle import validate_bundle


def build_release(
    root: Path,
    output: Path,
    tag: str,
    base_commit: str,
    code_ref="refs/heads/laq/dataset-release",
) -> dict:
    if not re.fullmatch(r"hard-candidate-v\d+\.\d+\.\d+-rc\.\d+", tag):
        raise ValueError("Use hard-candidate-vX.Y.Z-rc.N")
    if not re.fullmatch(r"[0-9a-f]{40}", base_commit):
        raise ValueError("Base code commit must be complete")
    if output.exists() or output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Release output must be new and outside candidate")
    report = validate_bundle(root)
    metadata = json.loads((root / "source/metadata.json").read_text())
    output.mkdir(parents=True)
    assets = {
        f"{tag}-data.zip": write_zip(root, output / f"{tag}-data.zip", "hard"),
        f"{tag}-review.zip": write_zip(
            root / "review_bundle", output / f"{tag}-review.zip", "hard_review"
        ),
    }
    preserved = (
        "pool_manifest.jsonl",
        "review_manifest.jsonl",
        "human_review.jsonl",
        "private/gold.jsonl",
        "private/checks.jsonl",
        "private/human_spot_review.jsonl",
        "source/metadata.json",
        "source/validation.json",
        "source/strict-audit.json",
        "construction_registry.json",
    )
    contract = {
        "version": 1,
        "dataset": "hard",
        "repository": "mmk1209/TIO-datasets",
        "tag": tag,
        "code_ref": code_ref,
        "base_code_commit": base_commit,
        "source_manifest_sha256": metadata["source_manifest_sha256"],
        "status": "candidate_owner_approved_pending_full_review",
        "benchmark_frozen": False,
        "training_ready": False,
        "archive_availability": "local_custody",
        "assets": assets,
        "expected_report": report,
        "preserved_files": {name: sha256(root / name) for name in preserved},
        "distribution": {
            "data": "Coordinator/research only; includes Gold, checks and the raw human spot return.",
            "review": "Separate answer-free random-ten review asset; no Gold or source identities.",
            "access_control": "Private repository mmk1209/TIO-datasets; owner invitation and upstream terms apply.",
            "programs": "Programs, mesh/URDF, caches and rejected candidates remain local_custody.",
        },
    }
    manifest = output / "release-manifest.json"
    manifest.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    names = sorted([*assets, manifest.name])
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(output/name)}  {name}\n" for name in names)
    )
    return contract


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--base-code-commit", required=True)
    parser.add_argument("--code-ref", default="refs/heads/laq/dataset-release")
    args = parser.parse_args()
    print(
        json.dumps(
            build_release(
                args.root, args.output, args.tag, args.base_code_commit, args.code_ref
            ),
            indent=2,
        )
    )
