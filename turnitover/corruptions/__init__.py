"""Explicit implementations keyed by taxonomy defect id."""
from turnitover.corruptions.base import Corruption
from turnitover.corruptions.kinematics import JointAxis
from turnitover.corruptions.runtime import TriangleBudget
from turnitover.corruptions.structure import PartOffset

CORRUPTIONS = {cls.defect_id: cls for cls in (JointAxis, TriangleBudget, PartOffset)}


def get_corruption(defect_id: str, **params) -> Corruption:
    return CORRUPTIONS[defect_id](**params)


def registered_ids() -> tuple[str, ...]:
    return tuple(sorted(CORRUPTIONS))


__all__ = ["Corruption", "get_corruption", "registered_ids"]
