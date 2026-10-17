import json
from pathlib import Path

from PIL import Image

from src.pipeline_parser import discover_pipeline_runs, parse_pipeline_run


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_round(
    run_dir: Path,
    index: int,
    source: str,
    code: str,
    feedback: str,
    selected_view: str,
) -> None:
    round_dir = run_dir / f"round_{index:02d}"
    code_dir = round_dir / f"code_agent_{source}"
    code_dir.mkdir(parents=True)
    (code_dir / "code.html").write_text(f"<html>agent-{index}</html>", encoding="utf-8")
    (code_dir / "reasoning.txt").write_text(f"reasoning {index}", encoding="utf-8")
    _write_json(code_dir / "agent.json", {"reasoning": f"reasoning {index}", "events": [], "usage": {}})

    attempt_dir = round_dir / "render_attempts" / "attempt_00"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "candidate.html").write_text(code, encoding="utf-8")
    Image.new("RGB", (8, 8), color="white").save(attempt_dir / "render.png")
    _write_json(
        attempt_dir / "check.json",
        {
            "success": True,
            "canvas_nonempty": True,
            "image_stddev": 12.0,
            "screenshot_path": f"data/runs/old-copy/round_{index:02d}/render_attempts/attempt_00/render.png",
        },
    )

    verifier_dir = round_dir / "verifier"
    verifier_dir.mkdir(parents=True)
    view_name = f"{selected_view}_orbit_left.png"
    Image.new("RGB", (8, 8), color="black").save(verifier_dir / view_name)
    trace = {
        "feedback": feedback,
        "selected_view_ids": [selected_view],
        "verdict": "revise",
        "observations": [
            {
                "view_id": selected_view,
                "action": "orbit_left",
                "screenshot_path": f"data/runs/old-copy/round_{index:02d}/verifier/{view_name}",
                "remaining_budget": 3,
            }
        ],
    }
    _write_json(verifier_dir / "trace.json", trace)
    _write_json(
        round_dir / "verifier_agent.json",
        {**trace, "reasoning": f"verifier reasoning {index}", "events": [], "usage": {}},
    )


def test_parse_pipeline_run_normalizes_rounds_and_rebases_paths(tmp_path: Path) -> None:
    sessions_root = tmp_path / "sessions"
    run_dir = sessions_root / "agent_demo" / "pipeline-demo"
    run_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color="white").save(run_dir / "reference.png")

    _write_round(
        run_dir,
        0,
        "generate",
        "<html>authoritative-0</html>",
        "fix blade material",
        "view_00",
    )
    _write_round(
        run_dir,
        1,
        "visual_revise",
        "<html>authoritative-1</html>",
        "fix blade thickness",
        "view_01",
    )

    trajectory = {
        "status": "max_visual_revisions",
        "accepted": False,
        "visual_revisions": 1,
        "rounds": [
            {
                "round_index": 0,
                "source": "generate",
                "render_success": True,
                "render_attempts": 1,
                "verifier_verdict": "revise",
                "feedback": "fix blade material",
            },
            {
                "round_index": 1,
                "source": "visual_revise",
                "render_success": True,
                "render_attempts": 1,
                "verifier_verdict": "revise",
                "feedback": "fix blade thickness",
            },
        ],
    }
    _write_json(run_dir / "trajectory.json", trajectory)
    _write_json(
        run_dir / "summary.json",
        {"status": "max_visual_revisions", "accepted": False, "visual_revisions": 1},
    )

    assert discover_pipeline_runs(sessions_root) == [run_dir]

    parsed = parse_pipeline_run(run_dir)
    assert parsed["session"]["source_type"] == "pipeline"
    assert parsed["session"]["accepted"] is False
    assert parsed["session"]["checkpoint_count"] == 2
    assert len(parsed["render_images"]) == 2

    first, second = parsed["checkpoints"]
    assert first["code_after"] == "<html>authoritative-0</html>"
    assert second["code_before"] == first["code_after"]
    assert second["code_after"] == "<html>authoritative-1</html>"
    assert second["prompt"] == "fix blade material"
    assert first["verifier_views"][0]["selected"] is True
    assert Path(first["verifier_views"][0]["path"]).parent == run_dir / "round_00" / "verifier"
    assert Path(first["final_render_path"]).parent == run_dir / "round_00" / "render_attempts" / "attempt_00"
