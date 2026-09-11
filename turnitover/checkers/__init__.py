"""Deterministic checkers over browser-exported evidence."""
from turnitover.checkers.base import Checker, Evidence, PartMesh
from turnitover.checkers.geometry import GeometryAlignmentChecker
from turnitover.checkers.runtime_budget import RuntimeBudgetChecker

CHECKERS: dict[str, type] = {
    GeometryAlignmentChecker.checker_id: GeometryAlignmentChecker,
    RuntimeBudgetChecker.checker_id: RuntimeBudgetChecker,
}


def make_checkers(cfg: dict[str, dict]) -> tuple[Checker, ...]:
    return tuple(CHECKERS[cid](**(params or {})) for cid, params in cfg.items())


__all__ = ["Checker", "Evidence", "PartMesh", "CHECKERS", "make_checkers"]
