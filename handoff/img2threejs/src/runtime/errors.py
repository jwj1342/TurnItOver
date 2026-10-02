from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RenderFailure:
    kind: str
    message: str


def classify_render_failure(
    *,
    page_loaded: bool,
    canvas_found: bool,
    canvas_nonempty: bool,
    load_errors: list[str],
    page_errors: list[str],
    console_errors: list[str],
) -> RenderFailure | None:
    if load_errors or not page_loaded:
        return RenderFailure("load_error", "page failed to load")
    if page_errors:
        return RenderFailure("page_error", page_errors[0])
    if console_errors:
        return RenderFailure("console_error", console_errors[0])
    if not canvas_found:
        return RenderFailure("missing_canvas", "no visible canvas was found")
    if not canvas_nonempty:
        return RenderFailure("blank_canvas", "canvas appears blank or uniform")
    return None
