from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from turnitover.assets.uniphys40k import UnsupportedAsset, convert_archive


def _box(extents, center=(0, 0, 0)) -> str:
    ex, ey, ez = (value / 2 for value in extents)
    cx, cy, cz = center
    vertices = [(cx + x, cy + y, cz + z) for x in (-ex, ex)
                for y in (-ey, ey) for z in (-ez, ez)]
    faces = [(1, 2, 4), (1, 4, 3), (5, 7, 8), (5, 8, 6),
             (1, 5, 6), (1, 6, 2), (3, 4, 8), (3, 8, 7),
             (1, 3, 7), (1, 7, 5), (2, 6, 8), (2, 8, 4)]
    return "".join(f"v {x} {y} {z}\n" for x, y, z in vertices) + "".join(
        f"f {a} {b} {c}\n" for a, b, c in faces)


def _merge_objs(*values: str) -> str:
    output, offset = [], 0
    for value in values:
        for line in value.splitlines():
            if line.startswith("v "):
                output.append(line)
            elif line.startswith("f "):
                output.append("f " + " ".join(str(int(item) + offset) for item in line.split()[1:]))
        offset += sum(line.startswith("v ") for line in value.splitlines())
    return "\n".join(output) + "\n"


def _archive(path: Path, *, extra_motion: bool = False, full_model_scale: bool = False,
             prismatic: bool = False) -> Path:
    asset = "UP_00000042"
    static = _box((2.0, 1.0, 0.2))
    moving = _box((2.0, 0.1, 1.0), center=(0, 0.45, 0.4))
    metadata = {
        "id": asset,
        "source": {"dataset": "PartNet", "original_id": "fixture", "license_ref": "PartNet"},
        "object": {"category": "fixture/laptop", "object_name": "Fixture Laptop"},
        "geometry": {"asset_type": "decomposed_parts", "num_parts": 2, "format": "obj"},
    }
    fixed_annotation = {
        "part_level": {"basic_description": (
            "Base." if full_model_scale else "Base, approx. 100*50*10 cm.")},
        "kinematic_info": {"motion_types": ["D"], "motion_info": {}},
    }
    code = "B" if prismatic else "C"
    moving_annotation = {
        "part_level": {"basic_description": "Lid, approx. 100*50*5 cm."},
        "kinematic_info": {"motion_types": [code], "motion_info": {
            "dependency": [2],
            code: {"axis": [1, 0, 0], "pos": ([99, 99, 99] if prismatic
                                                else [0, 0.5, 0.1]), "range": [0, 1.0]},
        }},
    }
    if extra_motion:
        fixed_annotation["kinematic_info"] = {"motion_types": ["B"], "motion_info": {
            "dependency": [1],
            "B": {"axis": [0, 1, 0], "pos": [0, 0, 0], "range": [0, 0.1]},
        }}
    with zipfile.ZipFile(path, "w") as zf:
        def add(name, value):
            zf.writestr(f"{asset}/{name}", value if isinstance(value, str) else json.dumps(value))
        add("meta_data.json", metadata)
        add("annotations/object.json", {"object_name": "Fixture Laptop", "category": "fixture/laptop",
                                         "volume": ([100, 50, 50] if full_model_scale
                                                    else [200, 200, 200]), "mass": 1})
        add("annotations/part_1.json", fixed_annotation)
        add("annotations/part_2.json", moving_annotation)
        add("parts/part_1.obj", static)
        add("parts/part_2.obj", moving)
        add("full_model/model.obj", _merge_objs(static, moving))
    return path


def test_convert_archive_derives_scale_and_dependency_group(tmp_path):
    converted = convert_archive(_archive(tmp_path / "fixture.zip"))
    assert converted.report["scale_validation"]["metres_per_obj_unit"] == pytest.approx(0.5)
    assert converted.report["scale_validation"]["reference_part_annotation_dimensions_cm"] == [10, 50, 100]
    assert converted.report["joint_validation"]["moving_parts"] == ["2"]
    assert converted.report["joint_validation"]["static_parts"] == ["1"]
    spec = converted.spec
    assert spec.joint("joint_part_2").anchor == pytest.approx((0, 0.25, 0.05))
    assert spec.part("part_2").parent is None
    assert np.max(np.asarray(spec.part("part_1").mesh.vertices), axis=0) == pytest.approx((0.5, 0.25, 0.05))


def test_convert_archive_rejects_ambiguous_multiple_motions(tmp_path):
    with pytest.raises(UnsupportedAsset, match="exactly one independent"):
        convert_archive(_archive(tmp_path / "fixture.zip", extra_motion=True))


def test_convert_archive_can_derive_scale_from_full_model_dimensions(tmp_path):
    converted = convert_archive(_archive(tmp_path / "fixture.zip", full_model_scale=True))
    assert converted.report["scale_validation"]["metres_per_obj_unit"] == pytest.approx(0.5)
    assert converted.report["scale_validation"]["method"].startswith("object_annotation")


def test_convert_archive_does_not_apply_revolute_pivot_gate_to_prismatic(tmp_path):
    converted = convert_archive(_archive(tmp_path / "fixture.zip", prismatic=True))
    assert converted.report["joint_validation"]["pivot_validation"] == "not_applicable_for_prismatic"
