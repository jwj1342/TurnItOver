from __future__ import annotations

import math

import numpy as np

from turnitover.core.program import ObjectProgram
from turnitover.core.sample import CorruptionSpec
from turnitover.corruptions.base import register, require_spec, unit_vector


AXIS_RANDOM_ANGLE = "axis_random_angle"
AXIS_PERMUTATION = "axis_permutation"
AXIS_TOLERANCE = 1e-6


def coordinate_axis(axis, tol: float = AXIS_TOLERANCE) -> tuple[int, int] | None:
    """(index, sign) when ``axis`` is +-e_i within ``tol`` on every component, else None."""
    v = np.asarray(axis, dtype=float)
    i = int(np.argmax(np.abs(v)))
    if abs(np.linalg.norm(v) - 1.0) > tol or np.any(np.abs(np.delete(v, i)) > tol):
        return None
    return i, 1 if v[i] > 0 else -1


@register("kinematics.joint_axis")
class JointAxis:
    """Replace one joint's axis; rest pose unchanged. ``mode`` is the stratum recorded in the label params.

    ``axis_random_angle``: rotate by a random angle about a random perpendicular. The result is almost never a
    coordinate axis, so the candidate code alone gives the defect away (issue #8 item 9); keep it as its own stratum.
    ``axis_permutation``: swap the coordinate axis +-e_i for +-e_j (j != i), sign drawn from ``rng``. The candidate
    stays a clean-looking coordinate axis. Joints whose axis is not a coordinate axis are not eligible.
    """

    def __init__(self, min_deg: float = 15.0, max_deg: float = 60.0, mode: str = AXIS_RANDOM_ANGLE):
        if mode not in (AXIS_RANDOM_ANGLE, AXIS_PERMUTATION):
            raise ValueError(f"unknown joint_axis mode {mode!r}")
        self.min_deg, self.max_deg, self.mode = min_deg, max_deg, mode

    def eligible(self, program: ObjectProgram) -> list:
        if program.spec is None:
            return []
        if self.mode == AXIS_RANDOM_ANGLE:
            return list(program.spec.joints)
        return [j for j in program.spec.joints if coordinate_axis(j.axis) is not None]

    def applicable(self, program: ObjectProgram) -> bool:
        return bool(self.eligible(program))

    def apply(self, program: ObjectProgram, rng: np.random.Generator):
        spec = require_spec(program)
        joints = self.eligible(program)
        if not joints:
            raise ValueError(f"no joint is eligible for joint_axis mode {self.mode}")
        joint = joints[rng.integers(len(joints))]
        if self.mode == AXIS_PERMUTATION:
            return self._permute(spec, joint, rng)
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
            params={"joint": joint.id, "mode": self.mode, "angle_deg": math.degrees(ang),
                    "original_axis": list(joint.axis)},
            label={"part": joint.part, "axis": new_axis.tolist()},
        )
        return ObjectProgram.from_spec(new_spec), cs

    def _permute(self, spec, joint, rng: np.random.Generator):
        source, source_sign = coordinate_axis(joint.axis)
        targets = [(j, s) for j in range(3) if j != source for s in (1, -1)]
        target, target_sign = targets[rng.integers(len(targets))]
        new_axis = [0.0, 0.0, 0.0]
        new_axis[target] = float(target_sign)
        new_spec = spec.replace_joint(joint.id, axis=tuple(new_axis))
        cs = CorruptionSpec(
            defect_id=self.defect_id, parts=(joint.part,), severity=1.0,
            params={"joint": joint.id, "mode": self.mode, "angle_deg": 90.0, "original_axis": list(joint.axis),
                    "from_index": source, "from_sign": source_sign, "to_index": target, "to_sign": target_sign},
            label={"part": joint.part, "axis": new_axis},
        )
        return ObjectProgram.from_spec(new_spec), cs


@register("kinematics.direction_flip")
class DirectionFlip:
    """Negate one joint's axis, keeping ``sign``: same anchor and limits, opposite motion. Rest pose unchanged.

    Negating the axis and negating ``sign`` move the part identically; the axis form is used because clean axes
    already take both signs while clean ``sign`` is always +1, so ``sign = -1`` would mark the defect in the code.
    The flip is undetectable for a motion symmetric about rest; callers screen that with observed pixels.
    """

    mode = "axis_negation"

    def applicable(self, program: ObjectProgram) -> bool:
        return program.spec is not None and len(program.spec.joints) > 0

    def apply(self, program: ObjectProgram, rng: np.random.Generator):
        spec = require_spec(program)
        joint = spec.joints[rng.integers(len(spec.joints))]
        new_axis = tuple(0.0 - x for x in joint.axis)  # 0.0 - x never produces a "-0.0" literal in the emitted code
        new_spec = spec.replace_joint(joint.id, axis=new_axis)
        cs = CorruptionSpec(
            defect_id=self.defect_id, parts=(joint.part,), severity=1.0,
            params={"joint": joint.id, "mode": self.mode, "original_axis": list(joint.axis),
                    "original_sign": joint.sign},
            label={"part": joint.part, "flipped": True},
        )
        return ObjectProgram.from_spec(new_spec), cs
