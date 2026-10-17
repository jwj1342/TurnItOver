from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PipelineRound:
    round_index: int
    source: str
    render_success: bool
    render_attempts: int
    verifier_verdict: str | None = None
    feedback: str | None = None
    selected_view_ids: list[str] = field(default_factory=list)
    evidence_paths: list[str] = field(default_factory=list)
    round_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    status: str
    accepted: bool
    final_html: str
    visual_revisions: int
    rounds: list[PipelineRound] = field(default_factory=list)

    def to_dict(self, *, include_html: bool = False) -> dict[str, Any]:
        data = {
            "status": self.status,
            "accepted": self.accepted,
            "visual_revisions": self.visual_revisions,
            "rounds": [item.to_dict() for item in self.rounds],
        }
        if include_html:
            data["final_html"] = self.final_html
        return data
