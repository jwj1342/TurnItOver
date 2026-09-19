from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.formatting import format_html_for_display, formatted_html_diff


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Pipeline JSON 无效：{path} 第 {exc.lineno} 行：{exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Pipeline JSON 顶层必须是 object：{path}")
    return data


def discover_pipeline_runs(root: str | Path) -> list[Path]:
    """Discover pipeline sessions below ``root`` at any grouping depth."""
    root = Path(root)
    if not root.exists():
        return []

    run_dirs = {
        path.parent.resolve(): path.parent
        for path in root.rglob("trajectory.json")
        if path.is_file()
    }
    return sorted(
        run_dirs.values(),
        key=lambda path: (path.name.lower(), str(path).lower()),
    )


def display_pipeline_name(run_dir: str | Path) -> str:
    return Path(run_dir).name


def _find_reference(run_dir: Path) -> Path | None:
    candidates = sorted(
        [
            path
            for path in run_dir.iterdir()
            if path.is_file()
            and path.stem.lower() == "reference"
            and path.suffix.lower() in IMAGE_SUFFIXES
        ],
        key=lambda path: path.name.lower(),
    )
    return candidates[0] if candidates else None


def _load_render_attempts(round_dir: Path) -> list[dict[str, Any]]:
    attempts_root = round_dir / "render_attempts"
    if not attempts_root.exists():
        return []

    attempts: list[dict[str, Any]] = []
    for attempt_dir in sorted(attempts_root.glob("attempt_*"), key=lambda path: path.name):
        if not attempt_dir.is_dir():
            continue
        attempts.append(
            {
                "name": attempt_dir.name,
                "check": _read_json(attempt_dir / "check.json"),
                "candidate_path": attempt_dir / "candidate.html",
                "render_path": attempt_dir / "render.png",
            }
        )
    return attempts


def _authoritative_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    successful = [item for item in attempts if item["check"].get("success") is True]
    if successful:
        return successful[-1]
    return attempts[-1] if attempts else None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _local_artifact_path(base_dir: Path, recorded_path: Any) -> Path | None:
    if not recorded_path:
        return None
    name = Path(str(recorded_path)).name
    if not name:
        return None
    local = base_dir / name
    return local if local.is_file() else None


def _usage_value(usage: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = usage.get(key)
        if value is not None:
            return value
    return None


def parse_pipeline_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    trajectory = _read_json(run_dir / "trajectory.json")
    summary = _read_json(run_dir / "summary.json")
    rounds_meta = trajectory.get("rounds")
    if not isinstance(rounds_meta, list):
        raise ValueError(f"{run_dir / 'trajectory.json'} 缺少 rounds 数组")

    reference = _find_reference(run_dir)
    checkpoints: list[dict[str, Any]] = []
    render_images: list[str] = []
    previous_code = ""
    previous_render: str | None = None

    for ordinal, raw_round in enumerate(rounds_meta):
        if not isinstance(raw_round, dict):
            continue
        round_index = int(raw_round.get("round_index", ordinal))
        source = str(raw_round.get("source") or ("generate" if round_index == 0 else "visual_revise"))
        round_dir = run_dir / f"round_{round_index:02d}"
        code_agent_dir = round_dir / f"code_agent_{source}"
        agent = _read_json(code_agent_dir / "agent.json")
        usage = agent.get("usage") if isinstance(agent.get("usage"), dict) else {}
        reasoning = _read_text(code_agent_dir / "reasoning.txt") or str(agent.get("reasoning") or "")

        attempts = _load_render_attempts(round_dir)
        authoritative = _authoritative_attempt(attempts)
        render_check: dict[str, Any] = {}
        final_render_path: str | None = None

        if authoritative is not None:
            candidate_path = authoritative["candidate_path"]
            code_after = _read_text(candidate_path)
            render_check = authoritative["check"]
            render_path = authoritative["render_path"]
            if render_path.is_file():
                final_render_path = str(render_path)
                render_images.append(final_render_path)
        else:
            code_after = ""

        if not code_after:
            code_after = _read_text(code_agent_dir / "code.html")

        verifier_dir = round_dir / "verifier"
        trace = _read_json(verifier_dir / "trace.json")
        verifier_agent = _read_json(round_dir / "verifier_agent.json")
        selected_view_ids = list(
            trace.get("selected_view_ids")
            or verifier_agent.get("selected_view_ids")
            or raw_round.get("selected_view_ids")
            or []
        )
        selected_set = {str(view_id) for view_id in selected_view_ids}
        observations = trace.get("observations") or verifier_agent.get("observations") or []
        verifier_views: list[dict[str, Any]] = []
        if isinstance(observations, list):
            for observation in observations:
                if not isinstance(observation, dict):
                    continue
                view_id = str(observation.get("view_id") or "")
                screenshot = _local_artifact_path(verifier_dir, observation.get("screenshot_path"))
                verifier_views.append(
                    {
                        "view_id": view_id or None,
                        "action": observation.get("action"),
                        "path": str(screenshot) if screenshot else None,
                        "selected": view_id in selected_set,
                        "remaining_budget": observation.get("remaining_budget"),
                        "action_payload": observation.get("action_payload"),
                        "runtime_metadata": observation.get("runtime_metadata"),
                    }
                )

        feedback = str(
            trace.get("feedback")
            or verifier_agent.get("feedback")
            or raw_round.get("feedback")
            or ""
        )
        verdict = (
            trace.get("verdict")
            or verifier_agent.get("verdict")
            or raw_round.get("verifier_verdict")
        )
        prompt = (
            "Generate Three.js reconstruction from the reference image."
            if round_index == 0
            else str(checkpoints[-1].get("verifier_feedback") or "Visual revision")
        )

        checkpoint = {
            "checkpoint": round_index,
            "message_id": None,
            "timestamp": None,
            "model": None,
            "provider": None,
            "prompt": prompt,
            "coder_reasoning": reasoning,
            "coder_final_answer": str(agent.get("final_answer") or ""),
            "coder_usage": usage,
            # Compatibility aliases used by the human-session-oriented viewer fields.
            "reasoning_trace": reasoning,
            "reasoning_kind": "pipeline_agent_reasoning" if reasoning else "none",
            "reasoning_tokens": _usage_value(usage, "reasoning_tokens", "reasoningTokens"),
            "input_tokens": _usage_value(usage, "input_tokens", "inputTokens"),
            "output_tokens": _usage_value(usage, "output_tokens", "outputTokens"),
            "total_tokens": _usage_value(usage, "total_tokens", "totalTokens"),
            "generation_duration_ms": agent.get("generation_duration_ms"),
            "first_token_latency_ms": agent.get("first_token_latency_ms"),
            "finish_reason": agent.get("finish_reason"),
            "code_before": previous_code,
            "code_after": code_after,
            "code_before_pretty": format_html_for_display(previous_code),
            "code_after_pretty": format_html_for_display(code_after),
            "code_diff": formatted_html_diff(previous_code, code_after),
            "changed_files": ["candidate.html"] if previous_code != code_after else [],
            "input_render_path": previous_render,
            "final_render_path": final_render_path,
            "render_source": "pipeline_render_attempt" if final_render_path else None,
            "source": source,
            "round_dir": str(round_dir),
            "render_success": bool(raw_round.get("render_success")),
            "render_attempts": attempts,
            "render_check": render_check,
            "verifier_verdict": verdict,
            "verifier_feedback": feedback,
            "verifier_views": verifier_views,
            "selected_view_ids": selected_view_ids,
            "verifier_reasoning": str(verifier_agent.get("reasoning") or ""),
            "verifier_final_answer": str(verifier_agent.get("final_answer") or ""),
            "verifier_usage": verifier_agent.get("usage") or {},
            "agent_events": agent.get("events") or [],
            "verifier_events": verifier_agent.get("events") or [],
        }
        checkpoints.append(checkpoint)
        previous_code = code_after
        if final_render_path:
            previous_render = final_render_path

    accepted = summary.get("accepted") if "accepted" in summary else trajectory.get("accepted")
    status = summary.get("status") or trajectory.get("status")
    return {
        "session": {
            "source_type": "pipeline",
            "session_id": run_dir.name,
            "name": run_dir.name,
            "thread_name": None,
            "provider": None,
            "model": None,
            "status": status,
            "accepted": accepted,
            "visual_revisions": summary.get("visual_revisions", trajectory.get("visual_revisions")),
            "checkpoint_count": len(checkpoints),
            "reference_path": str(reference) if reference else None,
            "render_count": len(render_images),
            "missing_mapped_images": 0,
            "unmatched_render_count": 0,
            "source_dir": str(run_dir),
        },
        "checkpoints": checkpoints,
        "render_images": render_images,
    }
