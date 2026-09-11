"""Corruption registry. Importing this package registers all implemented corruptions."""
from turnitover.corruptions.base import Corruption, get_corruption, register, registered_ids
from turnitover.corruptions import kinematics, runtime, structure  # noqa: F401  (registration side effect)

__all__ = ["Corruption", "get_corruption", "register", "registered_ids"]
