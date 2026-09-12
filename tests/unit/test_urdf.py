import numpy as np
import pytest

from turnitover.assets.urdf import UrdfModel


URDF = '''<robot name="fixture">
<link name="root"/>
<link name="arm"><visual><origin xyz="1 0 0"/><geometry><box size="0.2 0.3 0.4"/></geometry></visual></link>
<joint name="hinge" type="revolute"><parent link="root"/><child link="arm"/>
<origin xyz="2 0 0" rpy="0 0 1.5707963267948966"/><axis xyz="0 0 1"/><limit lower="-1" upper="1"/></joint>
<link name="tip"><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>
<joint name="slide" type="prismatic"><parent link="arm"/><child link="tip"/>
<origin xyz="1 0 0" rpy="0 1.5707963267948966 0"/><axis xyz="0 0 1"/><limit lower="0" upper="0.5"/></joint>
</robot>'''


def fixture_model(tmp_path):
    path = tmp_path / "fixture.urdf"
    path.write_text(URDF)
    return UrdfModel(path)


def test_fk_with_nested_nonidentity_joint_frames(tmp_path):
    model = fixture_model(tmp_path)
    rest = model.world_vertices({})
    assert np.allclose(rest["arm"].mean(axis=0), [2, 1, 0])
    assert np.allclose(rest["tip"].mean(axis=0), [2, 1, 0])
    moved = model.world_vertices({"slide": .5})
    assert np.allclose(moved["tip"].mean(axis=0), [2, 1.5, 0])
    combined = model.world_vertices({"hinge": .5, "slide": .5})
    assert np.allclose(combined["tip"].mean(axis=0), [2-1.5*np.sin(.5), 1.5*np.cos(.5), 0])
    spec = model.spec("fixture", "fixture")
    assert not spec.part("root").mesh.faces
    assert len(spec.joints) == 2
    assert spec.joint("hinge").limits == (-1, 1)


def test_unsupported_joint_is_not_silently_frozen(tmp_path):
    path = tmp_path / "bad.urdf"
    path.write_text(URDF.replace('type="revolute"', 'type="continuous"'))
    with pytest.raises(ValueError, match="Unsupported joint"):
        UrdfModel(path)
