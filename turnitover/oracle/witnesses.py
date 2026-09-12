"""Finite observation witnesses and state-correct, budgeted acquisition recipes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from turnitover.core.actions import ActuateJoint, Detent, RequestView, detent_value


@dataclass(frozen=True)
class Witness:
    view: str
    joint: str | None = None
    detent: Detent | None = None

    @property
    def key(self):
        return f"{self.view}:{self.joint or 'rest'}:{self.detent.value if self.detent else 'rest'}"


def witness_grid(views, joints, motion_views=("front", "oblique_fr")):
    result = []
    for view in views:
        result.append(Witness(view))
        if view not in motion_views:
            continue
        for name, limits in joints.items():
            seen = {0.}
            for detent in Detent:
                value = detent_value(detent, limits)
                if value not in seen:
                    result.append(Witness(view, name, detent))
                    seen.add(value)
    return tuple(result)


def zero_detent(limits):
    for detent in Detent:
        if detent_value(detent, limits) == 0:
            return detent
    raise ValueError("Witness planner requires a discrete detent at rest=0; cannot insert a free reset")


def recipe(current: Witness, target: Witness, joints):
    actions = []
    same_joint = current.joint == target.joint
    if current.joint is not None and not same_joint:
        actions.append(ActuateJoint(current.joint, zero_detent(joints[current.joint])))
    if current.view != target.view:
        actions.append(RequestView(target.view))
    if target.joint is not None and (not same_joint or current.detent != target.detent):
        actions.append(ActuateJoint(target.joint, target.detent))
    return tuple(actions)


def next_witness(current, action, joints):
    if isinstance(action, RequestView):
        return Witness(str(action.view_id), current.joint, current.detent)
    if isinstance(action, ActuateJoint):
        if detent_value(action.detent, joints[action.joint_id]) == 0:
            if current.joint not in (None, action.joint_id):
                raise ValueError("Cannot silently drop another active joint")
            return Witness(current.view)
        if current.joint not in (None, action.joint_id):
            raise ValueError("Witness recipes must reset the previous joint explicitly")
        return Witness(current.view, action.joint_id, action.detent)
    raise TypeError(action)


def plan(grid, exposures, joints, budget, mode="oracle", seed=0):
    """Greedy exposure/cost teacher; fixed/random witness orders are readout controls.

    This is not a globally optimal POMDP policy. Every restoration and view change
    is an actual observation action and consumes one unit of the same budget.
    """
    if budget < 1 or mode not in ("oracle", "fixed", "random"):
        raise ValueError("Positive budget and oracle/fixed/random mode required")
    lookup = {w.key: i for i, w in enumerate(grid)}
    if len(lookup) != len(grid) or len(exposures) != len(grid):
        raise ValueError("Unique witnesses and one exposure row per witness required")
    current = Witness("front")
    if current.key not in lookup:
        raise ValueError("Front rest witness required")
    trajectory = [(RequestView("front"), lookup[current.key])]
    visited = {lookup[current.key]}
    covered = set(exposures[lookup[current.key]])
    order = list(range(len(grid)))
    if mode == "random":
        np.random.default_rng(seed).shuffle(order)
    while len(trajectory) < budget:
        candidates = []
        for index in order:
            if index in visited:
                continue
            actions = recipe(current, grid[index], joints)
            if not actions or len(actions) > budget-len(trajectory):
                continue
            state, acquired = current, set()
            for action in actions:
                state = next_witness(state, action, joints)
                if state.key not in lookup:
                    raise ValueError(f"Recipe left the audited witness grid: {state.key}")
                acquired.update(exposures[lookup[state.key]])
            gains = len(acquired-covered)
            candidates.append((index, actions, gains))
        if not candidates:
            break
        # Zero-gain ties prefer the cheapest next witness, then stable grid order.
        index, actions, _ = (max(candidates, key=lambda c: (c[2]/len(c[1]), c[2], -len(c[1]), -c[0]))
                             if mode == "oracle" else candidates[0])
        for action in actions:
            current = next_witness(current, action, joints)
            # Intermediate views at a still-active joint may not be in the original grid.
            # Reset first for cross-view motion recipes so every step has audited evidence.
            if current.key not in lookup:
                raise ValueError(f"Recipe left the audited witness grid: {current.key}")
            witness_index = lookup[current.key]
            trajectory.append((action, witness_index))
            visited.add(witness_index)
            covered.update(exposures[witness_index])
        visited.add(index)
    return tuple(trajectory)
