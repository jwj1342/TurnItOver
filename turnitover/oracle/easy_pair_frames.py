"""Recover native Easy midpoint frames from an exactly matched review board."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image


BOARD_SIZE = (912, 630)
_ORIGINAL_TILES = {
    "reference_rest": (0, 0), "reference_limit": (0, 2),
    "candidate_rest": (1, 0), "candidate_limit": (1, 2),
}


def _crop(row: int, column: int) -> tuple[int, int, int, int]:
    x, y = 114 + column * 266, 98 + row * 266
    return x, y, x + 256, y + 256


def _rgb(raw: bytes) -> Image.Image:
    with Image.open(io.BytesIO(raw)) as image:
        return image.convert("RGB")


def easy_mid_frames(case_id: str, existing_frames: dict[str, bytes],
                    board_roots: list[Path]) -> tuple[dict[str, bytes], dict]:
    """Extract mid tiles only when all four endpoint tiles match the release.

    Boards are the unscaled layout written by ``prepare_level0_axis_review.board``.
    A differently rendered or stale board is never used to supply missing frames.
    """
    if not case_id or Path(case_id).name != case_id or case_id in {".", ".."}:
        raise ValueError("Invalid Easy case id")
    originals = {role: _rgb(existing_frames[role]) for role in _ORIGINAL_TILES}
    if any(image.size != (256, 256) for image in originals.values()):
        raise ValueError(f"Easy endpoint frames must be native 256x256: {case_id}")
    seen = set()
    for root in board_roots:
        for relative in (Path(f"{case_id}.png"), Path("boards") / f"{case_id}.png",
                         Path("easy/boards") / f"{case_id}.png"):
            path = Path(root) / relative
            if not path.is_file() or path.resolve() in seen:
                continue
            seen.add(path.resolve())
            raw = path.read_bytes()
            try:
                board = _rgb(raw)
            except (OSError, ValueError):
                continue
            if board.size != BOARD_SIZE:
                continue
            if any(board.crop(_crop(*tile)).tobytes() != originals[role].tobytes()
                   for role, tile in _ORIGINAL_TILES.items()):
                continue
            frames, crops = {}, {}
            for row, role in enumerate(("reference_mid", "candidate_mid")):
                box = _crop(row, 1)
                output = io.BytesIO()
                board.crop(box).save(output, format="PNG")
                frames[role] = output.getvalue()
                crops[role] = list(box)
            audit = {
                "method": "native_board_crop_exact_endpoint_match",
                "board_name": path.name,
                "board_sha256": hashlib.sha256(raw).hexdigest(),
                "board_size": list(BOARD_SIZE), "crops": crops,
                "validated": True,
                "matched_original_sha256": {
                    role: hashlib.sha256(existing_frames[role]).hexdigest()
                    for role in _ORIGINAL_TILES
                },
            }
            return frames, audit
    raise ValueError(f"No exact matching native Easy board for midpoint frames: {case_id}")
