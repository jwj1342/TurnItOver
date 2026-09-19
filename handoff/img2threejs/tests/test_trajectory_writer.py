import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from src.trajectory import CodegenTrajectoryWriter


def test_codegen_trajectory_writer(tmp_path: Path) -> None:
    reference = tmp_path / "input.png"
    Image.new("RGB", (8, 8), color="white").save(reference)
    run_dir = tmp_path / "run"
    writer = CodegenTrajectoryWriter(run_dir, reference)

    result = SimpleNamespace(
        html="<html>ok</html>",
        reasoning="visible reasoning",
        usage={"tokens": 10},
        to_dict=lambda include_html=False: {
            "reasoning": "visible reasoning",
            "final_answer": "done",
            "events": [],
            "usage": {"tokens": 10},
        },
    )
    writer.record_call(0, "generate", result)
    trajectory = writer.finalize(success=True, attempts=1, final_html=result.html)

    payload = json.loads(trajectory.read_text(encoding="utf-8"))
    assert payload["success"] is True
    assert payload["calls"][0]["kind"] == "generate"
    assert (run_dir / "call_00_generate" / "reasoning.txt").read_text() == "visible reasoning"
    assert (run_dir / "final.html").read_text() == "<html>ok</html>"
