"""Joint/link relations read from a visible ``query_runtime hierarchy`` payload.

The runtime hierarchy nests each articulated part group under a ``joint:<id>`` group,
which itself sits under the static parent part. These helpers only reshape what the
student already observed; they never consult Gold, labels, or the reference program.
"""
from __future__ import annotations

JOINT_PREFIX = "joint:"


def _is_joint(node: dict) -> bool:
    return node.get("name", "").startswith(JOINT_PREFIX)


def find(node: dict, name: str) -> dict | None:
    if node.get("name") == name:
        return node
    return next((found for child in node.get("children", []) if (found := find(child, name))), None)


def moving_children(root: dict, joint: str) -> list[str] | None:
    """Part groups directly under ``joint:<id>``: the links that joint itself carries."""
    node = find(root, f"{JOINT_PREFIX}{joint}")
    if node is None:
        return None
    return sorted(child["name"] for child in node.get("children", []) if not _is_joint(child))


def links(node: dict) -> list[str]:
    names = [] if _is_joint(node) else [node["name"]]
    return names + [name for child in node.get("children", []) for name in links(child)]


def joint_structure(root: dict) -> dict[str, dict]:
    """``joint_id -> {parent_link, moving_child_links}`` for every joint group in the hierarchy."""
    out: dict[str, dict] = {}

    def walk(node: dict, parent: str | None) -> None:
        if _is_joint(node):
            joint = node["name"][len(JOINT_PREFIX):]
            out[joint] = {"parent_link": parent, "moving_child_links": moving_children(node, joint)}
        for child in node.get("children", []):
            walk(child, parent if _is_joint(node) else node["name"])

    walk(root, None)
    return out


def part_relation(root: dict, joint: str, parts: list[str]) -> str | None:
    """How reported parts relate to the joint in the visible hierarchy (auditor-side analysis only)."""
    node = find(root, f"{JOINT_PREFIX}{joint}")
    if node is None:
        return None
    child, subtree, reported = set(moving_children(root, joint)), set(links(node)), set(parts)
    if reported == child:
        return "direct_child"
    if reported - subtree:
        return "includes_static_part_outside_joint_subtree"
    return "includes_moving_descendant" if child <= reported else "descendant_without_direct_child"
