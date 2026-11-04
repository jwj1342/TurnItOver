"""Truthful two-field mutations for ``hard-full-motion-formal-v1.1``."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from scripts.hard_construction.hard_full_motion_corruptions import (
    changed_fields, cone_directions, joint_anchor, joint_axis, part_rotation, part_translation, unit_directions)

POLICY = "hard-full-motion-formal-v1.1"
FIELDS = ("joint_axis", "joint_anchor", "part_offset", "part_rotation")
SIGNATURES = tuple("+".join(pair) for pair in combinations(FIELDS, 2))


def expected_fields(joint_id: str, part_id: str, signature: str) -> set[str]:
    paths = {"joint_axis": f"joints.{joint_id}.axis", "joint_anchor": f"joints.{joint_id}.anchor",
             "part_offset": f"parts.{part_id}.origin", "part_rotation": f"parts.{part_id}.rotation"}
    fields = signature.split("+")
    if signature not in SIGNATURES or len(fields) != 2:
        raise ValueError("Unsupported composite signature")
    return {paths[field] for field in fields}


def apply(reference, joint_id: str, part_id: str, signature: str, diagonal: float, variant: int):
    """Apply two deterministic Easy/Medium-range field changes in signature order."""
    program, defects = reference, []
    for index, field in enumerate(signature.split("+")):
        choice = (variant + index) % 8
        if field == "joint_axis":
            axis = cone_directions(program.spec.joint(joint_id).axis, 20)[choice]
            before = list(program.spec.joint(joint_id).axis)
            program = joint_axis(program, joint_id, axis)
            defects.append({"field": field, "difficulty": "medium", "joint_id": joint_id, "part_id": part_id,
                            "parameters": {"angle_deg": 20., "original_axis": before, "candidate_axis": list(axis)}})
        elif field == "joint_anchor":
            vector = np.asarray(unit_directions()[choice]) * (diagonal * .01)
            program = joint_anchor(program, joint_id, vector)
            defects.append({"field": field, "difficulty": "medium", "joint_id": joint_id, "part_id": part_id,
                            "parameters": {"offset_fraction": .01, "vector": vector.tolist()}})
        elif field == "part_offset":
            vector = np.asarray(unit_directions()[choice]) * (diagonal * .01)
            program = part_translation(program, part_id, vector)
            defects.append({"field": field, "difficulty": "easy", "joint_id": joint_id, "part_id": part_id,
                            "parameters": {"offset_fraction": .01, "vector": vector.tolist()}})
        else:
            axis = unit_directions()[choice]
            program = part_rotation(program, part_id, axis, 2.)
            defects.append({"field": field, "difficulty": "easy", "joint_id": joint_id, "part_id": part_id,
                            "parameters": {"angle_deg": 2., "axis": list(axis)}})
    if set(changed_fields(reference, program)) != expected_fields(joint_id, part_id, signature):
        raise ValueError("Composite candidate changed fields beyond declared faults")
    return program, defects
