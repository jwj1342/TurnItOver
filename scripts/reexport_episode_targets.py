#!/usr/bin/env python3
"""Re-export assistant and stopping targets while preserving audited observations."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil

from turnitover.oracle.episodes import episode_examples
from turnitover.oracle.stopping import stopping_examples


SPLITS = ("train", "validation", "test")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload(record: dict) -> dict:
    return json.loads(record["messages"][0]["content"].rsplit("\n", 1)[-1])


def _attach_from_prefix(record: dict, source: dict, *, same_prompt: bool = True) -> dict:
    if (same_prompt and record["messages"][0] != source["messages"][0]) or record["images"] != source["images"]:
        raise ValueError(f"Public prefix changed during target re-export: {record['id']}")
    record["data_identity"] = source["data_identity"]
    record["image_sha256"] = source["image_sha256"]
    return record


def reexport(source: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"Output must not exist: {output}")
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Complete source supervision is required")
    budget = manifest["budget"]
    objective = manifest["episode_objective"]
    output.mkdir(parents=True)
    shutil.copytree(source / "images", output / "images", copy_function=os.link)
    shutil.copy2(source / "annotations.private.jsonl", output / "annotations.private.jsonl")

    annotations = {row["id"]: row for row in _rows(source / "annotations.private.jsonl")}
    episode_counts, stopping_counts, stop_targets = Counter(), Counter(), Counter()
    for split in SPLITS:
        shutil.copy2(source / f"{split}.actions.jsonl", output / f"{split}.actions.jsonl")
        old_rows = _rows(source / f"{split}.episodes.jsonl")
        prefixes = {(row["episode_id"], row["decision_index"]): row for row in old_rows}
        episode_ids = list(dict.fromkeys(row["episode_id"] for row in old_rows))
        with (output / f"{split}.episodes.jsonl").open("w", encoding="utf-8") as episode_stream, (
                output / f"{split}.stopping.jsonl").open("w", encoding="utf-8") as stopping_stream:
            for episode_id in episode_ids:
                annotation = annotations[episode_id]
                if annotation["split"] != split:
                    raise ValueError(f"Annotation split differs: {episode_id}")
                first = prefixes[(episode_id, 0)]
                payload = _payload(first)
                trajectory = annotation["trajectories"]["oracle"]
                labels = annotation["labels"]
                for record in episode_examples(
                        episode_id, payload["context"], first["images"][0], trajectory["history"],
                        trajectory["witness_indices"], annotation["exposures"], labels, budget, objective):
                    source_record = prefixes[(episode_id, record["decision_index"])]
                    _attach_from_prefix(record, source_record)
                    episode_stream.write(json.dumps(record) + "\n")
                    episode_counts[split] += 1
                for record in stopping_examples(
                        episode_id, payload["context"], first["images"][0], trajectory["history"],
                        trajectory["witness_indices"], annotation["exposures"], labels, budget, objective):
                    source_record = prefixes[(episode_id, record["decision_index"])]
                    _attach_from_prefix(record, source_record, same_prompt=False)
                    stopping_stream.write(json.dumps(record) + "\n")
                    stopping_counts[split] += 1
                    stop_targets[f"{record['stop_target']}:{record['stop_outcome'] or 'none'}"] += 1

    root = Path(__file__).resolve().parents[1]
    target_sources = [
        Path(__file__).resolve(), root / "turnitover/oracle/episodes.py",
        root / "turnitover/oracle/stopping.py", root / "turnitover/verifier/prompt.txt",
    ]
    report = {
        **manifest,
        "version": 2,
        "status": "complete",
        "parent_export": str(source),
        "parent_manifest_sha256": _sha256(manifest_path),
        "reexport_scope": "assistant_and_stop_targets_only",
        "observations_reused_by_hardlink": True,
        "public_episode_inputs_preserved": True,
        "episode_examples": dict(episode_counts),
        "stopping_examples": dict(stopping_counts),
        "stop_targets": dict(stop_targets),
        "target_source_hashes": {str(path): _sha256(path) for path in target_sources},
        "limitations": [
            *manifest.get("limitations", []),
            "Assistant and stopping targets were re-exported from the audited private annotations; "
            "no observation, Gold label, or pixel was regenerated.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reexport(args.source.resolve(), args.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
