"""Build and validate the answer-free random-ten Hard review asset."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil

if __package__:
    from .easy_release_archive import sha256
    from .hard_bundle_contract import STATES
    from .hard_bundle_inventory import validate_manifest, write_manifest
    from .validate_easy_bundle import _png_size
else:
    from easy_release_archive import sha256
    from hard_bundle_contract import STATES
    from hard_bundle_inventory import validate_manifest, write_manifest
    from validate_easy_bundle import _png_size

ASSETS = Path(__file__).with_name("hard_review_assets")


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_review_bundle(root: Path) -> dict:
    root = root.resolve()
    files = validate_manifest(root)
    public, blank = _rows(root / "review_manifest.jsonl"), _rows(
        root / "blank_return.jsonl"
    )
    ids = [row["case_id"] for row in public]
    if (
        len(ids) != 10
        or len(set(ids)) != 10
        or {row["case_id"] for row in blank} != set(ids)
        or any(not re.fullmatch(r"hp-[0-9a-f]{14}", cid) for cid in ids)
    ):
        raise ValueError("Hard review must cover ten unique opaque cases")
    media = set()
    for row in public:
        if (
            row.get("states") != list(STATES)
            or row.get("image_size") != [256, 256]
            or row.get("status") != "pending_human_review"
        ):
            raise ValueError("Hard review public schema differs")
        for view in (row["initial_view"], row["supplementary_view"]):
            for role in ("reference", "candidate"):
                for state in STATES:
                    name = f'native/{row["case_id"]}/{view}/{role}_{state}.png'
                    if _png_size(root / name) != (256, 256):
                        raise ValueError(
                            "Hard review native frame is missing or invalid"
                        )
                    media.add(name)
    allowed = {
        "review.html",
        "review_manifest.jsonl",
        "blank_return.jsonl",
        "README.md",
        *media,
    }
    if files != allowed:
        raise ValueError("Hard review contains unexpected or missing files")
    text = (root / "review.html").read_text()
    if any(
        token in text
        for token in (
            "candidate_sha",
            "reference_sha",
            "asset_id",
            "joint_id",
            "part_id",
            "kind",
        )
    ):
        raise ValueError("Hard review page leaks construction answers")
    for row in blank:
        if (
            row.get("reviewer") is not None
            or row.get("reviewed_at") is not None
            or row.get("status") != "pending_human_review"
        ):
            raise ValueError("Hard review return must remain blank")
    return {"cases": 10, "media_files": len(media), "blank_returns": 10}


def prepare_review(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError("Hard review output must be new and outside source")
    public = _rows(source / "review_manifest.jsonl")
    blank = _rows(source / "human_review.jsonl")
    if len(public) != 10 or len(blank) != 10:
        raise ValueError("Source must contain the fixed random-ten review")
    output.mkdir(parents=True)
    shutil.copyfile(source / "review.html", output / "review.html")
    shutil.copyfile(source / "review_manifest.jsonl", output / "review_manifest.jsonl")
    (output / "blank_return.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in blank)
    )
    shutil.copyfile(ASSETS / "README.md", output / "README.md")
    by_id = {row["case_id"]: row for row in public}
    for cid, row in by_id.items():
        for view in (row["initial_view"], row["supplementary_view"]):
            for role in ("reference", "candidate"):
                for state in STATES:
                    name = f"native/{cid}/{view}/{role}_{state}.png"
                    target = output / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / name, target)
    write_manifest(output)
    return validate_review_bundle(output)
