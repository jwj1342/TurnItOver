"""JudgePolicy: history of observations -> next action. The single seam for oracle / VLM later."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from turnitover.core.actions import Action, Observation, ViewId


@dataclass(frozen=True)
class PolicyContext:
    joints: tuple[str, ...]
    views: tuple[ViewId, ...]
    budget: int | None


class JudgePolicy(Protocol):
    def reset(self, context: PolicyContext) -> None: ...

    def act(self, history: Sequence[Observation]) -> Action: ...
