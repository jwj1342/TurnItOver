"""Headless browser runtime for active inspection of generated Three.js HTML."""

from .actions import ViewAction, parse_action, pointer_motion
from .browser_session import (
    ActionResult,
    BrowserConfig,
    BrowserRuntimeError,
    BrowserSession,
    LoadResult,
)
from .errors import RenderFailure, classify_render_failure
from .render_check import RenderCheckResult, check_render

__all__ = [
    "ActionResult",
    "BrowserConfig",
    "BrowserRuntimeError",
    "BrowserSession",
    "LoadResult",
    "RenderCheckResult",
    "RenderFailure",
    "ViewAction",
    "check_render",
    "classify_render_failure",
    "parse_action",
    "pointer_motion",
]
