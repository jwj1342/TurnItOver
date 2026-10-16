"""Load and validate matched-state motion-diagnosis supervision."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


CLASSES = ("axis", "offset", "clean", "uncertain")


def load_motion_examples(root: Path, split: str, supervision: str = "evidence"):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("formal_training") is not True:
        raise ValueError("Refusing incomplete or non-formal motion supervision")
    if split not in ("train", "validation") or supervision not in ("ordinary", "evidence"):
        raise ValueError("Unknown split or supervision mode")
    for line in (root / f"{split}.jsonl").read_text().splitlines():
        source = json.loads(line)
        record = {key: source[key] for key in (
            "id", "asset_id", "condition", "images", "image_sha256", "messages")}
        record["messages"] = [dict(record["messages"][0]), {
            "role": "assistant", "content": json.dumps(source[f"{supervision}_target"])}]
        images = []
        for relative, expected in zip(record["images"], record["image_sha256"], strict=True):
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Invalid motion image: {relative}")
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        yield record, images, source


def validate_target(target: dict) -> None:
    if set(target) != {"defect_type", "joint", "part", "evidence_steps"}:
        raise ValueError("Unexpected diagnosis target fields")
    kind = target["defect_type"]
    if kind not in CLASSES or not isinstance(target["evidence_steps"], list):
        raise ValueError("Invalid diagnosis target")
    if kind == "axis" and not all(isinstance(target[key], str) for key in ("joint", "part")):
        raise ValueError("Axis target needs joint and part")
    if kind == "offset" and (target["joint"] is not None or not isinstance(target["part"], str)):
        raise ValueError("Offset target needs only part")
    if kind in ("clean", "uncertain") and (target["joint"] is not None or target["part"] is not None):
        raise ValueError("Clean/uncertain targets cannot localize a defect")
