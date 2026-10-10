"""Backward-compatible name for privileged GT code validation."""
from turnitover.verifier.gt_validator import validate_code


def audit(candidate, reference, render, views, max_triangles, max_draw_calls):
    return validate_code(candidate, reference, render, views, max_triangles, max_draw_calls)
