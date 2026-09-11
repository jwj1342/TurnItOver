from __future__ import annotations

import math

import numpy as np

from turnitover.core.program import ObjectProgram
from turnitover.core.sample import CorruptionSpec
from turnitover.corruptions.base import register, require_spec, unit_vector


@register("kinematics.joint_axis")
class JointAxis:
    """Rotate one joint's axis by a random angle about a random perpendicular. Rest pose unchanged."""

    def __init__(self, min_deg: float = 15.0, max_deg: float = 60.0):
        self.min_deg, self.max_deg = min_deg, max_deg

    def applicable(self, program: ObjectProgram) -> bool:
        return program.spec is not None and len(program.spec.joints) > 0

    def apply(self, program: ObjectProgram, rng: np.random.Generator):
        spec = require_spec(program)
        joint = spec.joints[rng.integers(len(spec.joints))]
        axis = np.asarray(joint.axis, dtype=float)
        axis /= np.linalg.norm(axis)
        perp = unit_vector(rng)
        perp -= perp.dot(axis) * axis
        perp /= np.linalg.norm(perp)
        ang = math.radians(float(rng.uniform(self.min_deg, self.max_deg)))
        new_axis = axis * math.cos(ang) + perp * math.sin(ang)
        new_axis /= np.linalg.norm(new_axis)
        new_spec = spec.replace_joint(joint.id, axis=tuple(float(x) for x in new_axis))
        cs = CorruptionSpec(
            defect_id=self.defect_id, parts=(joint.part,), severity=math.degrees(ang) / self.max_deg,
            params={"joint": joint.id, "angle_deg": math.degrees(ang), "original_axis": list(joint.axis)},
            label={"part": joint.part, "axis": new_axis.tolist()},
        )
        return ObjectProgram.from_spec(new_spec), cs
