from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agents import CodeAgent, CodeAgentResult, VerifierAgent, VerifierAgentResult
from src.config import Settings
from src.runtime import BrowserSession
from src.verifier import (
    CameraActionSpace,
    PoseBrowserRuntime,
    PoseGridActionSpace,
    RelativeBrowserRuntime,
    RelativeDiscreteActionSpace,
    VerifierEpisode,
)

from .retry import RenderAttemptResult, run_with_render_retry
from .state import PipelineResult, PipelineRound


class PipelineController:
    """Deterministic orchestration for CodeAgent <-> VerifierAgent iteration.

    The controller makes all retry/round/stopping decisions. Model reasoning text
    is recorded for inspection but is never parsed to control execution.
    """

    def __init__(
        self,
        settings: Settings,
        code_agent: CodeAgent,
        verifier_agent: VerifierAgent,
        browser_session: BrowserSession,
        action_space: CameraActionSpace,
    ) -> None:
        self.settings = settings
        self.code_agent = code_agent
        self.verifier_agent = verifier_agent
        self.browser_session = browser_session
        self.action_space = action_space

    def run(self, reference_image: str | Path, run_dir: str | Path) -> PipelineResult:
        reference = Path(reference_image)
        if not reference.is_file():
            raise FileNotFoundError(reference)

        root = Path(run_dir)
        root.mkdir(parents=True, exist_ok=True)
        rounds: list[PipelineRound] = []
        current_html = ""
        visual_revisions = 0
        pending_result: CodeAgentResult | None = None

        for round_index in range(self.settings.max_visual_revisions + 1):
            round_dir = root / f"round_{round_index:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)

            if round_index == 0:
                pending_result = self.code_agent.generate(
                    reference,
                    round_dir / "workspace_generate",
                )
                source = "generate"
            else:
                if pending_result is None:
                    raise RuntimeError("visual revision result missing before pipeline round")
                source = "visual_revise"

            assert pending_result is not None
            current_html = pending_result.html
            self._write_code_result(round_dir / f"code_agent_{source}", source, pending_result)

            render_result = self._render_with_repairs(
                reference,
                current_html,
                round_dir,
            )
            current_html = render_result.final_html
            if not render_result.success:
                round_state = PipelineRound(
                    round_index=round_index,
                    source=source,
                    render_success=False,
                    render_attempts=len(render_result.attempts),
                    round_dir=str(round_dir),
                )
                rounds.append(round_state)
                result = PipelineResult(
                    status="render_failed",
                    accepted=False,
                    final_html=current_html,
                    visual_revisions=visual_revisions,
                    rounds=rounds,
                )
                self._finalize(root, result)
                return result

            verifier_result = self._run_fresh_verifier(
                reference,
                current_html,
                round_dir,
            )
            self._write_json(round_dir / "verifier_agent.json", verifier_result.to_dict())
            evidence_paths = self._resolve_evidence_paths(verifier_result)

            round_state = PipelineRound(
                round_index=round_index,
                source=source,
                render_success=True,
                render_attempts=len(render_result.attempts),
                verifier_verdict=verifier_result.verdict,
                feedback=verifier_result.feedback,
                selected_view_ids=list(verifier_result.selected_view_ids),
                evidence_paths=[str(path) for path in evidence_paths],
                round_dir=str(round_dir),
            )
            rounds.append(round_state)

            if verifier_result.verdict == "accept":
                result = PipelineResult(
                    status="accepted",
                    accepted=True,
                    final_html=current_html,
                    visual_revisions=visual_revisions,
                    rounds=rounds,
                )
                self._finalize(root, result)
                return result

            if round_index >= self.settings.max_visual_revisions:
                result = PipelineResult(
                    status="max_visual_revisions",
                    accepted=False,
                    final_html=current_html,
                    visual_revisions=visual_revisions,
                    rounds=rounds,
                )
                self._finalize(root, result)
                return result

            pending_result = self.code_agent.revise_visual(
                reference,
                current_html,
                verifier_result.feedback,
                evidence_paths,
                root / f"round_{round_index + 1:02d}" / "workspace_visual_revise",
            )
            visual_revisions += 1

        raise AssertionError("pipeline loop terminated unexpectedly")

    def _render_with_repairs(
        self,
        reference: Path,
        initial_html: str,
        round_dir: Path,
    ) -> RenderAttemptResult:
        current_html = initial_html

        def candidate(previous, attempt_index: int) -> str:
            nonlocal current_html
            if previous is None:
                return current_html
            repair = self.code_agent.repair_runtime(
                reference,
                current_html,
                previous,
                round_dir / "runtime_repairs" / f"attempt_{attempt_index:02d}",
            )
            current_html = repair.html
            self._write_code_result(
                round_dir / "runtime_repairs" / f"call_{attempt_index:02d}",
                "runtime_repair",
                repair,
            )
            return current_html

        return run_with_render_retry(
            self.browser_session,
            candidate,
            round_dir / "render_attempts",
            # The initial candidate is checked before the bounded repair retries.
            max_attempts=1 + min(3, max(0, self.settings.max_render_retries)),
        )

    def _run_fresh_verifier(
        self,
        reference: Path,
        html: str,
        round_dir: Path,
    ) -> VerifierAgentResult:
        # A new runtime + episode is created for every renderable revision. This
        # deliberately prevents stale views from being reused across code edits.
        if isinstance(self.action_space, RelativeDiscreteActionSpace):
            runtime = RelativeBrowserRuntime(self.browser_session, self.action_space)
        elif isinstance(self.action_space, PoseGridActionSpace):
            runtime = PoseBrowserRuntime(self.browser_session, self.action_space)
        else:
            raise TypeError(f"unsupported action space type: {type(self.action_space).__name__}")

        episode = VerifierEpisode(
            runtime,
            html,
            round_dir / "verifier",
            action_budget=self.settings.verifier_action_budget,
            action_space=self.action_space,
            # _render_with_repairs leaves the successful candidate loaded in
            # this same BrowserSession. Reusing it preserves the render gate's
            # guarantee and avoids a second, failure-prone CDN fetch.
            page_already_loaded=True,
        )
        return self.verifier_agent.verify(reference, episode)

    @staticmethod
    def _resolve_evidence_paths(result: VerifierAgentResult) -> list[Path]:
        by_id = {
            str(item.get("view_id")): Path(str(item.get("screenshot_path")))
            for item in result.observations
            if item.get("view_id") and item.get("screenshot_path")
        }
        selected = [by_id[view_id] for view_id in result.selected_view_ids if view_id in by_id]
        if selected:
            return selected
        # Robust fallback for an otherwise valid revise verdict: give CodeAgent
        # the initial current render rather than silently sending no visual evidence.
        if result.observations:
            first = result.observations[0]
            path = first.get("screenshot_path")
            if path:
                return [Path(str(path))]
        return []

    @staticmethod
    def _write_code_result(directory: Path, kind: str, result: CodeAgentResult) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "code.html").write_text(result.html, encoding="utf-8")
        (directory / "reasoning.txt").write_text(result.reasoning or "", encoding="utf-8")
        payload = result.to_dict(include_html=False)
        payload["kind"] = kind
        PipelineController._write_json(directory / "agent.json", payload)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _finalize(root: Path, result: PipelineResult) -> None:
        (root / "final.html").write_text(result.final_html, encoding="utf-8")
        PipelineController._write_json(root / "trajectory.json", result.to_dict())
        PipelineController._write_json(
            root / "summary.json",
            {
                "status": result.status,
                "accepted": result.accepted,
                "visual_revisions": result.visual_revisions,
                "rounds": len(result.rounds),
                "final_html": "final.html",
                "trajectory": "trajectory.json",
            },
        )
