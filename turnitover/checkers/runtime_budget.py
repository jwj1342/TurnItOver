"""RuntimeBudgetChecker: triangle and draw-call budgets from the runtime stats probe."""
from __future__ import annotations

from typing import ClassVar

from turnitover.checkers.base import Evidence
from turnitover.core.sample import CheckResult, Invariant


class RuntimeBudgetChecker:
    checker_id: ClassVar[str] = "runtime.budget"
    needs_reference: ClassVar[bool] = False

    def __init__(self, max_triangles: int = 5000, max_draw_calls: int = 32):
        self.max_triangles = max_triangles
        self.max_draw_calls = max_draw_calls

    def check(self, evidence: Evidence, reference: Evidence | None = None) -> CheckResult:
        if not evidence.compile_ok:
            return CheckResult(self.checker_id, False, (Invariant("compile", 0.0, 0.0, False),), 0.0)
        tri = float(evidence.stats.get("triangles_scene", 0))
        calls = float(evidence.stats.get("draw_calls", 0))
        inv = (
            Invariant("triangles", tri, float(self.max_triangles), tri <= self.max_triangles),
            Invariant("draw_calls", calls, float(self.max_draw_calls), calls <= self.max_draw_calls),
        )
        passed = all(i.passed for i in inv)
        return CheckResult(self.checker_id, passed, inv, 1.0 if passed else 0.0)
