from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from src.formatting import format_html_for_display, formatted_html_diff


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"会话 JSON 无效：{path.name} 第 {exc.lineno} 行：{exc.msg}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        raise ValueError(f"{path} 不是受支持的 Chatbox session.json")
    return data


def _content_parts(message: dict[str, Any], part_type: str) -> list[dict[str, Any]]:
    return [
        part
        for part in (message.get("contentParts") or [])
        if isinstance(part, dict) and part.get("type") == part_type
    ]


def _message_text(message: dict[str, Any]) -> str:
    values = [str(part.get("text") or "").strip() for part in _content_parts(message, "text")]
    return "\n\n".join(value for value in values if value)


def _reasoning_text(message: dict[str, Any]) -> str:
    values = [str(part.get("text") or "").strip() for part in _content_parts(message, "reasoning")]
    return "\n\n".join(value for value in values if value)


def _has_image(message: dict[str, Any]) -> bool:
    return bool(_content_parts(message, "image"))


def extract_html(text: str) -> str:
    """Extract a complete HTML snapshot from a Chatbox assistant response."""
    if not text:
        return ""

    fenced = re.search(r"```(?:html)?\s*(<!doctype\s+html\b.*?</html>)\s*```", text, flags=re.I | re.S)
    if fenced:
        return fenced.group(1).strip() + "\n"

    fenced_any = re.search(r"```(?:html)?\s*(.*?)\s*```", text, flags=re.I | re.S)
    if fenced_any and "<html" in fenced_any.group(1).lower():
        return fenced_any.group(1).strip() + "\n"

    start = re.search(r"<!doctype\s+html\b|<html\b", text, flags=re.I)
    if not start:
        return ""
    end = list(re.finditer(r"</html\s*>", text, flags=re.I))
    if end:
        return text[start.start() : end[-1].end()].strip() + "\n"
    return text[start.start() :].strip() + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_reference_image(session_dir: str | Path) -> Path | None:
    session_dir = Path(session_dir)
    resources = session_dir / "resources"
    if not resources.exists():
        return None

    files = [p for p in resources.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    # Chatbox normally exports input.*. One supplied session contains the typo
    # inpuy.jpg, so keep source files untouched and tolerate that historical name.
    for stem in ("input", "inpuy"):
        for path in files:
            if path.stem.lower() == stem:
                return path
    return None


def discover_render_images(session_dir: str | Path, reference: Path | None = None) -> list[Path]:
    session_dir = Path(session_dir)
    resources = session_dir / "resources"
    if not resources.exists():
        return []

    def sort_key(path: Path) -> tuple[int, str]:
        match = re.search(r"(\d+)$", path.stem)
        return (int(match.group(1)) if match else 10**9, path.name.lower())

    candidates = sorted(
        [
            p
            for p in resources.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_SUFFIXES
            and p.stem.lower().startswith("resource-")
        ],
        key=sort_key,
    )

    # Chatbox may export a duplicate copy of the input image as resource-*. It
    # is not a render and would shift every later checkpoint by one.
    if reference and reference.exists():
        ref_hash = _sha256(reference)
        candidates = [p for p in candidates if _sha256(p) != ref_hash]
    return candidates


def discover_sessions(root: str | Path) -> list[Path]:
    """Discover Chatbox sessions below ``root`` at any grouping depth.

    Session fixtures are now grouped under paths such as
    ``data/sessions/human_demo/<session-id>/``. Using ``rglob`` keeps the
    viewer compatible with that layout and with future grouping directories
    while still identifying a session only by the presence of ``session.json``.
    """
    root = Path(root)
    if not root.exists():
        return []

    session_dirs = {
        path.parent.resolve(): path.parent
        for path in root.rglob("session.json")
        if path.is_file()
    }
    return sorted(
        session_dirs.values(),
        key=lambda path: (path.name.lower(), str(path).lower()),
    )


def display_session_name(session_dir: str | Path) -> str:
    path = Path(session_dir) / "session.json"
    try:
        data = _read_json(path)
        return str(data.get("name") or data.get("threadName") or Path(session_dir).name)
    except Exception:
        return Path(session_dir).name


def parse_session(session_dir: str | Path) -> dict[str, Any]:
    session_dir = Path(session_dir)
    data = _read_json(session_dir / "session.json")
    messages = data.get("messages") or []
    reference = find_reference_image(session_dir)
    render_images = discover_render_images(session_dir, reference)

    # Image-bearing user messages are chronological. The first is the true RGB
    # reference. Each later image is the render that the user fed back for the
    # most recent assistant code state.
    image_user_indices = [
        idx
        for idx, message in enumerate(messages)
        if isinstance(message, dict) and message.get("role") == "user" and _has_image(message)
    ]
    user_image_map: dict[int, Path | None] = {}
    if image_user_indices:
        user_image_map[image_user_indices[0]] = reference
        for idx, image_path in zip(image_user_indices[1:], render_images):
            user_image_map[idx] = image_path

    checkpoints: list[dict[str, Any]] = []
    current_code = ""
    pending_prompt = ""
    pending_input_render: str | None = None
    previous_checkpoint: dict[str, Any] | None = None

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")

        if role == "user":
            pending_prompt = _message_text(message)
            mapped_image = user_image_map.get(message_index)
            pending_input_render = None
            if mapped_image is not None and reference is not None and mapped_image != reference:
                pending_input_render = str(mapped_image)
                if previous_checkpoint is not None:
                    previous_checkpoint["final_render_path"] = str(mapped_image)
                    previous_checkpoint["render_source"] = "uploaded_in_next_user_turn"
            continue

        if role != "assistant":
            continue

        response_text = _message_text(message)
        html = extract_html(response_text)
        if not html:
            continue

        reasoning = _reasoning_text(message)
        usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
        before = current_code
        after = html
        checkpoint_index = len(checkpoints)

        checkpoint = {
            "checkpoint": checkpoint_index,
            "message_id": message.get("id"),
            "timestamp": message.get("timestamp"),
            "model": message.get("model") or message.get("modelId") or (data.get("settings") or {}).get("modelId"),
            "provider": message.get("aiProvider") or (data.get("settings") or {}).get("provider"),
            "prompt": pending_prompt,
            "reasoning_trace": reasoning,
            "reasoning_kind": "full_visible_trace" if reasoning else "none",
            "reasoning_tokens": usage.get("reasoningTokens")
            or (usage.get("outputTokenDetails") or {}).get("reasoningTokens"),
            "input_tokens": usage.get("inputTokens"),
            "output_tokens": usage.get("outputTokens"),
            "total_tokens": usage.get("totalTokens"),
            "generation_duration_ms": message.get("generationDuration"),
            "first_token_latency_ms": message.get("firstTokenLatency"),
            "finish_reason": message.get("finishReason"),
            "code_before": before,
            "code_after": after,
            "code_before_pretty": format_html_for_display(before),
            "code_after_pretty": format_html_for_display(after),
            "code_diff": formatted_html_diff(before, after),
            "changed_files": ["index.html"] if before != after else [],
            "input_render_path": pending_input_render,
            "final_render_path": None,
            "render_source": None,
        }
        checkpoints.append(checkpoint)
        previous_checkpoint = checkpoint
        current_code = after

    missing_mapped_images = max(0, (len(image_user_indices) - 1) - len(render_images))
    unmatched_render_count = max(0, len(render_images) - max(0, len(image_user_indices) - 1))

    return {
        "session": {
            "session_id": str(data.get("id") or session_dir.name),
            "name": data.get("name") or data.get("threadName") or session_dir.name,
            "thread_name": data.get("threadName"),
            "provider": (data.get("settings") or {}).get("provider"),
            "model": (data.get("settings") or {}).get("modelId"),
            "checkpoint_count": len(checkpoints),
            "reference_path": str(reference) if reference else None,
            "render_count": len(render_images),
            "missing_mapped_images": missing_mapped_images,
            "unmatched_render_count": unmatched_render_count,
            "source_dir": str(session_dir),
        },
        "checkpoints": checkpoints,
        "render_images": [str(path) for path in render_images],
    }
