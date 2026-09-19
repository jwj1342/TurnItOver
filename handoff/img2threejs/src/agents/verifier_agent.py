from __future__ import annotations

import base64
import json
import mimetypes
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.config import Settings
from src.models import build_qwen_llm
from src.verifier import VerifierEpisode, VerifierEpisodeResult


VerifierRunner = Callable[[VerifierEpisode, Path, str], dict[str, Any]]


def _run_with_finish_retry(
    conversation: Any,
    episode: VerifierEpisode,
    finish_message: Any,
) -> bool:
    """Run once and allow one bounded finish-only recovery turn.

    Returns whether the recovery turn was needed. The caller still validates
    ``episode.finished`` so this helper never fabricates a verifier verdict.
    """
    conversation.run()
    if episode.finished:
        return False
    conversation.send_message(finish_message)
    conversation.run()
    return True


@dataclass
class VerifierAgentResult:
    feedback: str
    selected_view_ids: list[str]
    verdict: str = "revise"
    observations: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str | None = None
    final_answer: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VerifierAgent:
    """Independent OpenHands verifier with observation-only camera tools."""

    def __init__(
        self,
        settings: Settings,
        *,
        prompt_dir: str | Path | None = None,
        runner: VerifierRunner | None = None,
    ) -> None:
        self.settings = settings
        self.prompt_dir = Path(prompt_dir or Path(__file__).parents[1] / "prompts")
        self._runner = runner or self._run_openhands

    def verify(self, reference_image: str | Path, episode: VerifierEpisode) -> VerifierAgentResult:
        reference = Path(reference_image)
        if not reference.is_file():
            raise FileNotFoundError(reference)

        episode.start()
        base_prompt = (self.prompt_dir / "verifier_step.md").read_text(encoding="utf-8")
        prompt = base_prompt + "\n\n" + episode.action_space.tool_instructions()
        trace = self._runner(episode, reference, prompt)

        if not episode.finished:
            failure = {
                "error": "VerifierAgent ended without calling finish after one finish-only retry",
                "finish_retry_attempted": bool(trace.get("finish_retry_attempted", False)),
                "remaining_budget": episode.remaining_budget,
                "observations": [asdict(item) for item in episode.observations],
                "reasoning": trace.get("reasoning") or None,
                "final_answer": trace.get("final_answer") or None,
                "events": list(trace.get("events") or []),
                "usage": dict(trace.get("usage") or {}),
            }
            (episode.output_dir / "incomplete_agent.json").write_text(
                json.dumps(failure, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            raise RuntimeError(failure["error"])

        result: VerifierEpisodeResult = episode.result()
        return VerifierAgentResult(
            feedback=result.feedback,
            selected_view_ids=list(result.selected_view_ids),
            verdict=result.verdict,
            observations=[asdict(item) for item in result.observations],
            reasoning=trace.get("reasoning") or None,
            final_answer=trace.get("final_answer") or None,
            events=list(trace.get("events") or []),
            usage=dict(trace.get("usage") or {}),
        )

    def _run_openhands(self, episode: VerifierEpisode, reference_image: Path, prompt: str) -> dict[str, Any]:
        try:
            from openhands.sdk import (
                Action, Agent, Conversation, ImageContent, Message, Observation,
                TextContent, Tool, ToolDefinition,
            )
            from openhands.sdk.tool import ToolExecutor, register_tool
            from pydantic import Field
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "OpenHands agent dependencies are missing. Install requirements-agent.txt."
            ) from exc

        class ThreeJSViewAction(Action):
            action: str = Field(
                description=(
                    "Camera action supported by the configured action space, or 'finish'. "
                    "For V1 use orbit/zoom/capture; for V2 use goto_pose/capture."
                )
            )
            target_view_id: str | None = Field(
                default=None,
                description="V2 only: required when action='goto_pose'; one configured pose-grid view id",
            )
            feedback: str | None = Field(
                default=None,
                description="Required only for finish: concise actionable visual assessment",
            )
            verdict: str | None = Field(
                default=None,
                description="For finish: 'revise' if CodeAgent should modify the candidate, 'accept' if no important actionable mismatch remains",
            )
            selected_view_ids: list[str] = Field(
                default_factory=list,
                description="Evidence observation ids such as view_01 to keep when finishing",
            )

        class ThreeJSViewObservation(Observation):
            message: str
            view_id: str | None = None
            screenshot_data_url: str | None = None
            remaining_budget: int
            finished: bool = False

            @property
            def to_llm_content(self):
                content = [TextContent(text=self.message)]
                if self.screenshot_data_url:
                    content.append(ImageContent(image_urls=[self.screenshot_data_url]))
                return content

        class ThreeJSViewExecutor(ToolExecutor[ThreeJSViewAction, ThreeJSViewObservation]):
            def __call__(self, action: ThreeJSViewAction, conversation=None) -> ThreeJSViewObservation:  # noqa: ARG002
                action_name = action.action.strip().lower()
                if action_name == "finish":
                    if not (action.feedback or "").strip():
                        return ThreeJSViewObservation(
                            message="finish requires non-empty feedback",
                            remaining_budget=episode.remaining_budget,
                            finished=False,
                        )
                    try:
                        result = episode.finish(
                            action.feedback or "",
                            selected_view_ids=list(action.selected_view_ids or []),
                            verdict=action.verdict or "revise",
                        )
                    except ValueError as exc:
                        return ThreeJSViewObservation(
                            message=str(exc),
                            remaining_budget=episode.remaining_budget,
                            finished=False,
                        )
                    return ThreeJSViewObservation(
                        message=(
                            f"Verification finished with verdict={result.verdict}. "
                            f"Selected views: {', '.join(result.selected_view_ids) or 'none'}"
                        ),
                        remaining_budget=episode.remaining_budget,
                        finished=True,
                    )

                if episode.remaining_budget <= 0:
                    return ThreeJSViewObservation(
                        message="Observation budget exhausted. Call finish now.",
                        remaining_budget=0,
                        finished=False,
                    )

                try:
                    observation = episode.act(action_name, target_view_id=action.target_view_id)
                except (ValueError, RuntimeError) as exc:
                    return ThreeJSViewObservation(
                        message=str(exc),
                        remaining_budget=episode.remaining_budget,
                        finished=False,
                    )
                return ThreeJSViewObservation(
                    message=(
                        f"Observed {observation.view_id} after {observation.action}. "
                        f"Remaining observation budget: {observation.remaining_budget}."
                    ),
                    view_id=observation.view_id,
                    screenshot_data_url=_image_data_url(Path(observation.screenshot_path)),
                    remaining_budget=observation.remaining_budget,
                    finished=False,
                )

        class ThreeJSViewTool(ToolDefinition[ThreeJSViewAction, ThreeJSViewObservation]):
            @classmethod
            def create(cls, conv_state, **params):  # noqa: ARG003
                return [cls(
                    description=(
                        "Inspect the current Three.js render using the configured restricted camera action space. "
                        "Every non-finish observation consumes budget. Finish with verdict, feedback and selected evidence ids."
                    ),
                    action_type=ThreeJSViewAction,
                    observation_type=ThreeJSViewObservation,
                    executor=ThreeJSViewExecutor(),
                )]

        tool_name = f"ThreeJSViewTool_{uuid.uuid4().hex}"
        register_tool(tool_name, ThreeJSViewTool)

        llm = build_qwen_llm(self.settings, usage_id="verifier-agent")
        agent = Agent(
            llm=llm,
            tools=[Tool(name=tool_name)],
            include_default_tools=["ThinkTool"],
        )

        events: list[dict[str, Any]] = []
        reasoning_chunks: list[str] = []
        assistant_texts: list[str] = []

        def on_event(event: Any) -> None:
            reasoning = getattr(event, "reasoning_content", None)
            if isinstance(reasoning, str) and reasoning.strip():
                reasoning_chunks.append(reasoning.strip())
            try:
                llm_message = event.to_llm_message()
            except Exception:
                llm_message = None
            if llm_message is not None and getattr(llm_message, "role", None) == "assistant":
                text_parts = []
                for part in getattr(llm_message, "content", []) or []:
                    text = getattr(part, "text", None)
                    if isinstance(text, str) and text.strip():
                        text_parts.append(text.strip())
                if text_parts:
                    assistant_texts.append("\n".join(text_parts))
                msg_reasoning = getattr(llm_message, "reasoning_content", None)
                if isinstance(msg_reasoning, str) and msg_reasoning.strip():
                    reasoning_chunks.append(msg_reasoning.strip())
            if hasattr(event, "model_dump"):
                try:
                    events.append(event.model_dump(mode="json"))
                    return
                except Exception:
                    pass
            events.append({"type": type(event).__name__, "repr": repr(event)})

        initial_view = episode.observations[0]
        conversation = Conversation(
            agent=agent,
            workspace=str(episode.output_dir),
            callbacks=[on_event],
            max_iteration_per_run=self.settings.verifier_agent_max_steps,
        )
        conversation.send_message(Message(role="user", content=[
            TextContent(text=prompt),
            TextContent(text="Reference image:"),
            ImageContent(image_urls=[_image_data_url(reference_image)]),
            TextContent(text=f"Initial candidate render ({initial_view.view_id}):"),
            ImageContent(image_urls=[_image_data_url(Path(initial_view.screenshot_path))]),
        ]))
        finish_message = Message(role="user", content=[TextContent(text=(
            "你尚未通过观察工具提交结构化结论。不要继续观察，也不要输出普通文本。"
            "现在必须调用 ThreeJSViewTool 的 finish 动作：根据已经获得的视图设置 "
            "verdict、非空 feedback 和有效的 selected_view_ids。"
        ))])
        finish_retry_attempted = _run_with_finish_retry(
            conversation,
            episode,
            finish_message,
        )

        unique_reasoning = list(dict.fromkeys(chunk for chunk in reasoning_chunks if chunk))
        usage: dict[str, Any] = {}
        metrics = getattr(llm, "metrics", None)
        if metrics is not None:
            cost = getattr(metrics, "accumulated_cost", None)
            if cost is not None:
                usage["accumulated_cost"] = cost

        return {
            "reasoning": "\n\n".join(unique_reasoning),
            "final_answer": assistant_texts[-1] if assistant_texts else "",
            "events": events,
            "usage": usage,
            "finish_retry_attempted": finish_retry_attempted,
        }


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
