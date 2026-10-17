from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.runtime import BrowserSession

from .action_space import CameraActionSpace, RelativeDiscreteActionSpace
from .observation_runtime import ObservationRuntime, RelativeBrowserRuntime


VALID_VERDICTS = {"revise", "accept"}


@dataclass
class VerifierObservation:
    view_id: str
    action: str
    screenshot_path: str
    remaining_budget: int
    action_payload: dict[str, Any] = field(default_factory=dict)
    runtime_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifierEpisodeResult:
    feedback: str
    selected_view_ids: list[str]
    verdict: str = "revise"
    observations: list[VerifierObservation] = field(default_factory=list)
    exhausted_budget: bool = False
    action_space: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "feedback": self.feedback,
            "selected_view_ids": list(self.selected_view_ids),
            "verdict": self.verdict,
            "observations": [asdict(item) for item in self.observations],
            "exhausted_budget": self.exhausted_budget,
            "action_space": dict(self.action_space),
        }


class VerifierEpisode:
    """Stateful active-view episode over one rendered candidate HTML."""

    def __init__(
        self,
        session_or_runtime: BrowserSession | ObservationRuntime,
        html: str,
        output_dir: str | Path,
        *,
        action_budget: int,
        action_space: CameraActionSpace | None = None,
        page_already_loaded: bool = False,
    ) -> None:
        if action_budget < 1:
            raise ValueError("action_budget must be >= 1")

        if isinstance(session_or_runtime, BrowserSession):
            resolved_space = action_space or RelativeDiscreteActionSpace()
            if not isinstance(resolved_space, RelativeDiscreteActionSpace):
                raise TypeError(
                    "Passing BrowserSession directly is only supported for relative_discrete_v1; "
                    "construct the matching ObservationRuntime for other action spaces."
                )
            runtime: ObservationRuntime = RelativeBrowserRuntime(session_or_runtime, resolved_space)
        else:
            runtime = session_or_runtime
            if action_space is None:
                action_space = getattr(runtime, "action_space", None)
            resolved_space = action_space
            if resolved_space is None:
                raise ValueError("VerifierEpisode requires an action_space")

        self.runtime = runtime
        self.action_space = resolved_space
        self.html = html
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.action_budget = action_budget
        self.actions_used = 0
        self.observations: list[VerifierObservation] = []
        self.finished = False
        self.feedback = ""
        self.verdict = "revise"
        self.selected_view_ids: list[str] = []
        self.page_already_loaded = page_already_loaded

    @property
    def remaining_budget(self) -> int:
        return max(self.action_budget - self.actions_used, 0)

    def start(self) -> VerifierObservation:
        if self.page_already_loaded:
            self.runtime.prepare_loaded()
        else:
            self.runtime.reset(self.html)
        return self._capture_initial()

    def act(self, action: str, *, target_view_id: str | None = None) -> VerifierObservation:
        if self.finished:
            raise RuntimeError("verifier episode is already finished")
        if self.remaining_budget <= 0:
            raise RuntimeError("verifier action budget exhausted")

        camera_action = self.action_space.resolve(action, view_id=target_view_id)
        self.actions_used += 1
        view_id = f"view_{len(self.observations):02d}"
        suffix = target_view_id if target_view_id else camera_action.name
        safe_suffix = suffix.replace("/", "_").replace(" ", "_")
        screenshot_path = self.output_dir / f"{view_id}_{safe_suffix}.png"
        metadata = self.runtime.execute(camera_action, screenshot_path)
        observation = VerifierObservation(
            view_id=view_id,
            action=camera_action.name,
            screenshot_path=str(screenshot_path),
            remaining_budget=self.remaining_budget,
            action_payload=camera_action.to_dict(),
            runtime_metadata=dict(metadata or {}),
        )
        self.observations.append(observation)
        return observation

    def finish(
        self,
        feedback: str,
        selected_view_ids: list[str] | None = None,
        *,
        verdict: str = "revise",
    ) -> VerifierEpisodeResult:
        if self.finished:
            raise RuntimeError("verifier episode is already finished")
        feedback = feedback.strip()
        if not feedback:
            raise ValueError("finish requires non-empty feedback")
        verdict = verdict.strip().lower()
        if verdict not in VALID_VERDICTS:
            raise ValueError(f"unknown verifier verdict {verdict!r}; expected revise or accept")

        known = {item.view_id for item in self.observations}
        selected = list(dict.fromkeys(selected_view_ids or []))
        unknown = [item for item in selected if item not in known]
        if unknown:
            raise ValueError(f"unknown selected view ids: {', '.join(unknown)}")

        self.finished = True
        self.feedback = feedback
        self.verdict = verdict
        self.selected_view_ids = selected
        result = self.result()
        (self.output_dir / "trace.json").write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result

    def result(self) -> VerifierEpisodeResult:
        return VerifierEpisodeResult(
            feedback=self.feedback,
            selected_view_ids=list(self.selected_view_ids),
            verdict=self.verdict,
            observations=list(self.observations),
            exhausted_budget=self.remaining_budget == 0,
            action_space=self.action_space.to_dict(),
        )

    def _capture_initial(self) -> VerifierObservation:
        view_id = f"view_{len(self.observations):02d}"
        screenshot_path = self.output_dir / f"{view_id}_initial.png"
        capture_action = self.action_space.resolve("capture")
        metadata = self.runtime.execute(capture_action, screenshot_path)
        observation = VerifierObservation(
            view_id=view_id,
            action="initial",
            screenshot_path=str(screenshot_path),
            remaining_budget=self.remaining_budget,
            action_payload={"type": "initial"},
            runtime_metadata=dict(metadata or {}),
        )
        self.observations.append(observation)
        return observation
