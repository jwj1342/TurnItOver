"""ScriptedPolicy: replay a fixed action list, then Stop."""
from __future__ import annotations

from typing import Sequence

from turnitover.core.actions import Action, Observation, Stop
from turnitover.policy.base import PolicyContext


class ScriptedPolicy:
    def __init__(self, actions: Sequence[Action]):
        self._actions = tuple(actions)

    def reset(self, context: PolicyContext) -> None:
        pass

    def act(self, history: Sequence[Observation]) -> Action:
        i = len(history)
        return self._actions[i] if i < len(self._actions) else Stop()
