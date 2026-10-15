import json

from turnitover.core.actions import Observation, QueryRuntime, RuntimeProperty
from turnitover.teacher.policy import TeacherPolicy
from turnitover.teacher.providers import MockTeacher
from turnitover.teacher.structure import joint_structure, part_relation
from turnitover.verifier.contracts import Context


# Shape of the PM_103275 rest hierarchy: base -> link_2 -> joint_1 -> link_1 -> joint_0 -> link_0.
ROOT = {"name": "asset", "children": [{"name": "base", "children": [{"name": "link_2", "children": [
    {"name": "joint:joint_1", "children": [{"name": "link_1", "children": [
        {"name": "joint:joint_0", "children": [{"name": "link_0", "children": []}]}]}]}]}]}]}


def test_joint_structure_names_parent_and_moving_child():
    assert joint_structure(ROOT) == {
        "joint_1": {"parent_link": "link_2", "moving_child_links": ["link_1"]},
        "joint_0": {"parent_link": "link_1", "moving_child_links": ["link_0"]}}


def test_part_relation_separates_parent_and_descendant_errors():
    assert part_relation(ROOT, "joint_1", ["link_1"]) == "direct_child"
    assert part_relation(ROOT, "joint_1", ["link_1", "link_2"]) == "includes_static_part_outside_joint_subtree"
    assert part_relation(ROOT, "joint_1", ["link_1", "link_0"]) == "includes_moving_descendant"


def test_visible_state_carries_structure_only_after_hierarchy_was_observed(tmp_path):
    context = Context("inspect", ("link_0", "link_1", "link_2"), ("joint_0", "joint_1"), ("front",),
                      ("hierarchy",), ("request_view", "actuate_joint", "query_runtime"), 100, 10)
    verdict = {"verdict": {"status": "uncertain", "confidence": .1, "findings": [], "summary": "s",
                           "limitations": ["l"]}}
    policy = TeacherPolicy(MockTeacher([verdict, verdict]), tmp_path, [])
    policy.decide(context, [], 1)
    query = Observation(0, QueryRuntime(RuntimeProperty.HIERARCHY), None,
                        {"property": "hierarchy", "root": ROOT}, 0)
    policy.decide(context, [query], 0)
    first, second = (json.loads((tmp_path / f"model_calls/{i:03d}/request.json").read_text()) for i in (0, 1))
    assert "joint_structure" not in first
    assert second["joint_structure"]["joint_1"]["moving_child_links"] == ["link_1"]
    assert "Joint-axis diagnosis rules" not in (tmp_path / "model_calls/001/prompt.txt").read_text()
