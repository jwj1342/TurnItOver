import json

from PIL import Image

from turnitover.core.program import ObjectProgram
from turnitover.models.client import ModelResult
from turnitover.models.generation import generate_program
from turnitover.render.gate import RenderGateResult, image_stddev
from turnitover.render.session import RenderConfig
from turnitover.render.views import load_views
from turnitover.repair.iterative import IterativeConfig, run_iterative
from turnitover.repair.loop import Feedback, RepairProposal, propose_repair
from turnitover.verifier.runner import VerifyConfig


SOURCE = "export default function createObject(THREE) { return {root:new THREE.Group(), joints:{}}; }\n"


def _photo(path):
    Image.new("RGB", (8, 8), "red").save(path)
    return path


def _render(tmp_path):
    return RenderConfig(tmp_path / "dist", tmp_path / "esbuild", 32, 32)


def _views(repo_root):
    return load_views(repo_root / "configs/views.yaml")


def test_generate_program_records_program_and_rejects_truncation(tmp_path):
    photo = _photo(tmp_path / "photo.png")

    def call(prompt, images):
        assert prompt == "build" and images[0].parent.name == "generation"
        return ModelResult(f"```ts\n{SOURCE}```", "fixture", {"total_tokens": 7}, "completed", 1.0)

    generated = generate_program("build", (photo,), tmp_path / "generation", call)
    assert generated.program.source == SOURCE
    assert json.loads((tmp_path / "generation/result.json").read_text())["status"] == "complete"

    def truncated(prompt, images):
        return ModelResult(SOURCE, "fixture", {}, "length", 1.0)

    try:
        generate_program("build", (photo,), tmp_path / "truncated", truncated)
    except RuntimeError:
        pass
    else:
        raise AssertionError("truncated response was accepted")
    failed = json.loads((tmp_path / "truncated/result.json").read_text())
    assert failed["status"] == "failed"
    assert failed.get("error"), "生成失败时应落盘错误详情以便诊断"


def test_repair_proposal_reuses_public_feedback_contract(tmp_path):
    response = json.dumps({"edits": [{"old": "red", "new": "blue"}]})
    proposal = propose_repair("red", lambda *_: ModelResult(response, "fixture", {"total_tokens": 3},
                                                              "completed", 1.0),
                              Feedback({"phase": "visual_revision"}, (), 0), tmp_path / "repair")
    assert proposal.source == "blue" and proposal.outcome == "applied"
    assert json.loads((tmp_path / "repair/result.json").read_text())["usage"]["total_tokens"] == 3


def test_image_stddev_distinguishes_constant_and_varied_images():
    import io

    constant = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(constant, format="PNG")
    varied = Image.new("RGB", (4, 4), "white")
    varied.putpixel((0, 0), (0, 0, 0))
    varied_bytes = io.BytesIO()
    varied.save(varied_bytes, format="PNG")
    assert image_stddev(constant.getvalue()) == 0
    assert image_stddev(varied_bytes.getvalue()) > 1


def test_iterative_loop_gates_and_fresh_verifies_final_revision(tmp_path):
    photo = _photo(tmp_path / "photo.png")
    repo_root = __import__("pathlib").Path(__file__).resolve().parents[2]
    verified = []
    gated = []

    def gate(program, output, render, views, threshold):
        output.mkdir(parents=True)
        gated.append(program.source)
        return RenderGateResult(True, "complete", image_ref=None, image_stddev=4.0)

    def verifier(program, output, render, views, cfg, **kwargs):
        output.mkdir(parents=True)
        verified.append((program.source, output.name))
        if len(verified) == 1:
            return {"status": "complete", "termination": "judge_finished", "spent": 1,
                    "model_calls": [{}], "trajectory": [{"step": 0, "payload": {"stats": {}}}],
                    "verdict": {"status": "fail", "confidence": 1.0,
                                "findings": [{"defect_id": "geometry.missing_part", "parts": [],
                                              "severity": 1.0, "confidence": 1.0, "evidence_steps": [0],
                                              "description": "missing", "suggested_fix": "add it"}],
                                "summary": "revise", "limitations": []}}
        return {"status": "complete", "termination": "judge_finished", "spent": 1,
                "model_calls": [{}], "trajectory": [{"step": 0}],
                "verdict": {"status": "pass", "confidence": 1.0, "findings": [],
                            "summary": "ok", "limitations": []}}

    def repair(source, generator, packet, output, **kwargs):
        output.mkdir(parents=True)
        assert packet.public["verdict"]["status"] == "fail"
        return RepairProposal(source + "// revised\n", "applied", "{}", None, {"total_tokens": 5},
                              "fixture", "completed", 1.0)

    cfg = IterativeConfig(max_runtime_repairs=0, max_visual_revisions=1,
                          verifier=VerifyConfig(budget=1))
    result = run_iterative((photo,), tmp_path / "run", _render(tmp_path), _views(repo_root), cfg,
                           lambda *_: ModelResult(SOURCE, "fixture", {}, "completed", 1.0),
                           generation_prompt="build", gate_call=gate, verifier_call=verifier,
                           repair_call=repair)
    assert result["status"] == "accepted" and result["accepted"]
    assert len(gated) == len(verified) == 2
    assert verified[1][0].endswith("// revised\n")
    assert (tmp_path / "run/round-001/verifier").is_dir()


def test_render_failure_has_independent_bounded_repair_budget(tmp_path):
    photo = _photo(tmp_path / "photo.png")
    repo_root = __import__("pathlib").Path(__file__).resolve().parents[2]
    calls = {"gate": 0, "repair": 0}

    def gate(program, output, *args):
        output.mkdir(parents=True)
        calls["gate"] += 1
        return RenderGateResult(False, "compile", "CompileError", "bad program")

    def repair(source, generator, packet, output, **kwargs):
        output.mkdir(parents=True)
        calls["repair"] += 1
        return RepairProposal(None, "invalid_patch", "bad", "invalid", {}, "fixture", "completed", 1.0)

    cfg = IterativeConfig(max_runtime_repairs=2, max_visual_revisions=9,
                          verifier=VerifyConfig(mode="runtime"))
    result = run_iterative((photo,), tmp_path / "run", _render(tmp_path), _views(repo_root), cfg,
                           lambda *_: ModelResult(SOURCE, "fixture", {}, "completed", 1.0),
                           generation_prompt="build", gate_call=gate,
                           verifier_call=lambda *a, **k: (_ for _ in ()).throw(AssertionError("verified")),
                           repair_call=repair)
    assert result["status"] == "render_failed" and not result["accepted"]
    assert calls == {"gate": 3, "repair": 2}
