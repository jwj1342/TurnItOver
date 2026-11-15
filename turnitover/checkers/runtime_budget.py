"""RuntimeBudgetChecker: triangle and draw-call budgets from the runtime stats probe."""
from __future__ import annotations

from typing import ClassVar
import math

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
        # Missing/invalid measurements cannot become a zero-cost passing program.
        values = [evidence.stats.get(name) for name in ('triangles_scene', 'draw_calls')]
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
               for value in values):
            return CheckResult(self.checker_id, False,
                               (Invariant('runtime_evidence_complete', 0.0, 1.0, False),), None)
        tri, calls = map(float, values)
        inv = (
            Invariant("triangles", tri, float(self.max_triangles), tri <= self.max_triangles),
            Invariant("draw_calls", calls, float(self.max_draw_calls), calls <= self.max_draw_calls),
        )
        passed = all(i.passed for i in inv)
        return CheckResult(self.checker_id, passed, inv, 1.0 if passed else 0.0)
