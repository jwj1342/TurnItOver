import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts.run_uniphys_iterate_batch import run


def test_batch_continues_after_normal_rejection_and_summarizes(tmp_path):
    references = tmp_path / "references"
    objects = []
    for index in range(20):
        object_id = f"UPB_{index:08d}"
        image = references / object_id / "screenshots" / "06.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(f"image-{index}".encode())
        objects.append({"object_id": object_id})
    (references / "manifest.json").write_text(json.dumps({"status": "complete", "objects": objects}))
    env_file = tmp_path / ".env"
    env_file.write_text("\n".join([
        "TIO_GENERATOR_PROVIDER=openai_compatible",
        "TIO_GENERATOR_MODEL=qwen3.7-plus",
        "TIO_JUDGE_PROVIDER=openai_compatible",
        "TIO_JUDGE_MODEL=qwen3.7-plus",
    ]))

    def execute(command, **kwargs):
        output = Path(command[command.index("--out") + 1])
        reference = Path(command[command.index("--reference-image") + 1])
        output.mkdir(parents=True)
        accepted = output.name != "UPB_00000000"
        result = {
            "status": "accepted" if accepted else "max_visual_revisions",
            "accepted": accepted,
            "termination": None if accepted else "revision_limit",
            "rounds": [{}],
            "model_calls": {"generation": 1, "runtime_repair": 0, "visual_revision": 0, "judge": 1},
            "total_tokens": 10,
            "references": [{"path": "reference-00.png", "sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}],
        }
        (output / "result.json").write_text(json.dumps(result))
        return subprocess.CompletedProcess(command, 0 if accepted else 1, "done\n", "")

    result = run(references, tmp_path / "output", env_file, execute=execute)
    assert result["status"] == "complete"
    assert result["totals"] == {
        "planned": 20,
        "completed": 20,
        "accepted": 19,
        "model_calls": {"generation": 20, "runtime_repair": 0, "visual_revision": 0, "judge": 20},
        "total_tokens": 200,
    }
    assert result["tasks"][0]["status"] == "max_visual_revisions"
    assert (tmp_path / "output/UPB_00000000/runner.stdout.log").read_text() == "done\n"


def test_batch_rejects_incomplete_reference_manifest(tmp_path):
    references = tmp_path / "references"
    references.mkdir()
    (references / "manifest.json").write_text(json.dumps({"status": "complete", "objects": []}))
    env_file = tmp_path / ".env"
    env_file.write_text("\n".join([
        "TIO_GENERATOR_PROVIDER=openai_compatible",
        "TIO_GENERATOR_MODEL=qwen3.7-plus",
        "TIO_JUDGE_PROVIDER=openai_compatible",
        "TIO_JUDGE_MODEL=qwen3.7-plus",
    ]))

    with pytest.raises(ValueError, match="complete 20-object"):
        run(references, tmp_path / "output", env_file)
