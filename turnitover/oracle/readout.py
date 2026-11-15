"""Load paired readout evidence and select a privileged strongest observation."""
import json
from pathlib import Path


def load_readout_data(oracle: Path, paired: Path) -> tuple[dict, dict, list[dict]]:
    manifest = json.loads((oracle / "manifest.json").read_text())
    if manifest["status"] != "complete" or json.loads((paired / "manifest.json").read_text())["status"] != "complete":
        raise ValueError("Complete evidence and paired export required")
    pairs = {r["id"]: r for r in map(json.loads, (paired / "test.diagnosis.jsonl").read_text().splitlines())}
    annotations = [r for r in map(json.loads, (oracle / "annotations.private.jsonl").read_text().splitlines()) if r["split"] == "test"]
    return manifest, pairs, annotations


def strongest_step(annotation: dict, indices: list[int]) -> int:
    return max(range(len(indices)), key=lambda i: annotation["signals"]["against_reference"][indices[i]]["image_l1"])
