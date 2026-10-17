from __future__ import annotations

import base64
import mimetypes
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.config import Settings
from src.models import build_qwen_llm
from src.runtime.render_check import RenderCheckResult


AgentRunner = Callable[[Path, str, Path], dict[str, Any]]
VisualAgentRunner = Callable[[Path, str, Path, list[Path]], dict[str, Any]]


@dataclass
class CodeAgentResult:
    html: str
    reasoning: str | None = None
    final_answer: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_html: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not include_html:
            data.pop("html", None)
        return data


class CodeAgent:
    """OpenHands-backed code-writing agent for image-to-Three.js reconstruction."""

    def __init__(
        self,
        settings: Settings,
        *,
        prompt_dir: str | Path | None = None,
        runner: AgentRunner | None = None,
        visual_runner: VisualAgentRunner | None = None,
    ) -> None:
        self.settings = settings
        self.prompt_dir = Path(prompt_dir or Path(__file__).parents[1] / "prompts")
        self._runner = runner or self._run_openhands
        self._visual_runner = visual_runner or self._run_openhands_visual

    def generate(self, reference_image: str | Path, workspace: str | Path) -> CodeAgentResult:
        workspace_path = self._prepare_workspace(reference_image, workspace)
        prompt = (self.prompt_dir / "code_generate.md").read_text(encoding="utf-8")
        trace = self._runner(workspace_path, prompt, Path(reference_image))
        return self._result_from_workspace(workspace_path, trace)

    def repair_runtime(
        self,
        reference_image: str | Path,
        current_html: str,
        render_error: RenderCheckResult,
        workspace: str | Path,
    ) -> CodeAgentResult:
        workspace_path = self._prepare_workspace(reference_image, workspace)
        (workspace_path / "candidate.html").write_text(current_html, encoding="utf-8")
        template = (self.prompt_dir / "code_repair.md").read_text(encoding="utf-8")
        prompt = template.format(repair_message=render_error.repair_message())
        trace = self._runner(workspace_path, prompt, Path(reference_image))
        return self._result_from_workspace(workspace_path, trace)

    def revise_visual(
        self,
        reference_image: str | Path,
        current_html: str,
        feedback: str,
        evidence_images: list[str | Path],
        workspace: str | Path,
    ) -> CodeAgentResult:
        """Revise a renderable candidate using verifier-selected visual evidence."""
        feedback = feedback.strip()
        if not feedback:
            raise ValueError("visual revision requires non-empty verifier feedback")

        workspace_path = self._prepare_workspace(reference_image, workspace)
        (workspace_path / "candidate.html").write_text(current_html, encoding="utf-8")
        evidence_paths = [Path(path) for path in evidence_images]
        missing = [str(path) for path in evidence_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing verifier evidence images: {', '.join(missing)}")

        template = (self.prompt_dir / "code_visual_revise.md").read_text(encoding="utf-8")
        prompt = template.format(feedback=feedback)
        trace = self._visual_runner(
            workspace_path,
            prompt,
            Path(reference_image),
            evidence_paths,
        )
        return self._result_from_workspace(workspace_path, trace)

    def _prepare_workspace(self, reference_image: str | Path, workspace: str | Path) -> Path:
        source = Path(reference_image)
        if not source.is_file():
            raise FileNotFoundError(source)
        target = Path(workspace)
        target.mkdir(parents=True, exist_ok=True)
        copied = target / f"reference{source.suffix.lower() or '.png'}"
        if source.resolve() != copied.resolve():
            shutil.copy2(source, copied)
        return target

    def _result_from_workspace(self, workspace: Path, trace: dict[str, Any]) -> CodeAgentResult:
        candidate = workspace / "candidate.html"
        if not candidate.is_file():
            raise RuntimeError("CodeAgent finished without creating candidate.html")
        html = candidate.read_text(encoding="utf-8")
        if not html.strip():
            raise RuntimeError("CodeAgent created an empty candidate.html")
        return CodeAgentResult(
            html=html,
            reasoning=trace.get("reasoning") or None,
            final_answer=trace.get("final_answer") or None,
            events=list(trace.get("events") or []),
            usage=dict(trace.get("usage") or {}),
        )

    def _run_openhands(self, workspace: Path, prompt: str, reference_image: Path) -> dict[str, Any]:
        return self._run_openhands_with_images(workspace, prompt, [reference_image])

    def _run_openhands_visual(
        self,
        workspace: Path,
        prompt: str,
        reference_image: Path,
        evidence_images: list[Path],
    ) -> dict[str, Any]:
        return self._run_openhands_with_images(
            workspace,
            prompt,
            [reference_image, *evidence_images],
        )

    def _run_openhands_with_images(
        self,
        workspace: Path,
        prompt: str,
        images: list[Path],
    ) -> dict[str, Any]:
        """Run one isolated OpenHands conversation and return an auditable trace."""
        try:
            from openhands.sdk import Agent, Conversation, ImageContent, Message, TextContent, Tool
            from openhands.tools.file_editor import FileEditorTool
            from openhands.tools.terminal import TerminalTool
        except ImportError as exc:  # pragma: no cover - requires agent environment
            raise RuntimeError(
                "OpenHands agent dependencies are missing. Install requirements-agent.txt."
            ) from exc

        llm = build_qwen_llm(self.settings, usage_id="code-agent")
        agent = Agent(
            llm=llm,
            tools=[Tool(name=FileEditorTool.name), Tool(name=TerminalTool.name)],
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

        conversation = Conversation(
            agent=agent,
            workspace=str(workspace),
            callbacks=[on_event],
            max_iteration_per_run=self.settings.code_agent_max_steps,
        )
        content = [TextContent(text=prompt)]
        labels = ["Reference image", *[f"Verifier evidence {index + 1}" for index in range(len(images) - 1)]]
        for label, image in zip(labels, images):
            content.append(TextContent(text=f"{label}:"))
            content.append(ImageContent(image_urls=[_image_data_url(image)]))
        conversation.send_message(Message(role="user", content=content))
        conversation.run()

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
        }


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
