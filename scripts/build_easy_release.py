"""Build immutable Easy candidate assets from an explicitly supplied snapshot."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

if __package__:
    from .easy_release_archive import sha256, write_zip
    from .validate_easy_bundle import validate_bundle
else:
    from easy_release_archive import sha256, write_zip
    from validate_easy_bundle import validate_bundle


def build_release(root: Path, output: Path, tag: str, source_commit: str, base_commit: str) -> dict:
    if not re.fullmatch(r"easy-candidate-v\d+\.\d+\.\d+-rc\.\d+", tag):
        raise ValueError("use an explicit easy-candidate-vX.Y.Z-rc.N tag")
    if any(not re.fullmatch(r"[0-9a-f]{40}", value) for value in (source_commit, base_commit)):
        raise ValueError("source and base commits must be complete Git object IDs")
    if output.exists() or output.resolve().is_relative_to(root.resolve()):
        raise ValueError("release output must be new and outside the source bundle")
    report = validate_bundle(root)
    output.mkdir(parents=True)
    assets = {
        f"{tag}-data.zip": write_zip(root, output / f"{tag}-data.zip", "easy"),
        f"{tag}-validation-review.zip": write_zip(
            root / "validation_review", output / f"{tag}-validation-review.zip", "validation_review"),
    }
    preserved = ["private/gold.jsonl", "private/import_checks.jsonl", "human_review.jsonl",
                 "review_manifest.jsonl", "splits.json", "provenance.json", "collection_approval.json"]
    contract = dict(
        version=1, repository="jwj1342/TurnItOver", tag=tag,
        code_ref=f"refs/tags/{tag}", base_code_commit=base_commit,
        source_dataset_commit=source_commit,
        status="candidate_pending_review", benchmark_frozen=False, training_ready=False,
        archive_availability="local_custody", assets=assets,
        expected_report=report,
        preserved_files={name: sha256(root / name) for name in preserved},
        distribution={
            "data": "Contains Gold, checker evidence and historical reviews; coordinator/research use, not blind-review input.",
            "validation_review": "Distribute this separate ZIP to reviewers before either independent return is compared.",
            "access_control": "A public pre-release does not prevent access to Gold or provide benchmark holdout secrecy.",
            "programs": "Sealed construction archive is not included; this is not a standalone replay or training export.",
        },
    )
    manifest = output / "release-manifest.json"
    manifest.write_text(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    names = sorted([*assets, manifest.name])
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(output / name)}  {name}\n" for name in names), encoding="utf-8")
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--base-code-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(build_release(args.root, args.output, args.tag, args.source_commit,
                                   args.base_code_commit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
