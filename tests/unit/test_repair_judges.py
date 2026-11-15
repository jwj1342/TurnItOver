import json
import os
from pathlib import Path

import pytest
from PIL import Image

from turnitover.core.program import ObjectProgram
from turnitover.core.sample import Framing
from turnitover.repair import judges
from turnitover.repair.judges import (PACK_FORMAT, Case, Cell, GoldJudge, JudgeModel, NoneJudge, PixelJudge, Sweep, VlmJudge,
                                      cell_name, judge_prompt, load_pack, sweep_layout, validate_verdict)

TIO_RUNS = Path(os.environ.get("TIO_RUNS", Path(__file__).resolve().parents[2] / "tio-runs"))
PACK = TIO_RUNS / "data900/repair58/three-tier-900-validation-repair-58-v1"
LABELS = TIO_RUNS / "data900/three-tier-900-sft-v1/episode_export/annotations.private.jsonl"
EPISODES = TIO_RUNS / "data900/three-tier-900-sft-v1/episode_export/validation.episodes.jsonl"
real_pack = pytest.mark.skipif(not (PACK.exists() and LABELS.exists() and EPISODES.exists()),
                               reason="her repair pack and the 900 export are not under $TIO_RUNS (default <repo>/tio-runs)")
INFO = {"parts": ["base", "door"], "joints": ["hinge"]}


def make_case(tmp_path, views=("front",), states=("rest", "mid", "limit"), labels=None):
    case = Case("case1", "medium", "initial", ObjectProgram("export function build(){}"), None,
                labels or [{"defect_id": "kinematics.joint_axis", "parts": ["door"], "severity": 0.7, "params": {"joint": "hinge"}}],
                Sweep(tuple(views), ("hinge",), tuple(states)))
    cells, i = [], 0
    single = len(views) == 1
    for view in views:
        for kind, color in (("reference", (10, 10, 10)), ("candidate", (10, 10, 10))):
            for state in states:
                path = tmp_path / f"{kind}_{view}_{state}.png"
                Image.new("RGB", (8, 8), (200, 0, 0) if (kind == "candidate" and state == "limit") else color).save(path)
                cells.append(Cell(view, state, kind, cell_name(i, view, kind, state, single), path)); i += 1
    return case, cells


def test_cell_names_follow_the_training_export():
    assert cell_name(0, "front", "reference", "rest", True) == "00-reference_rest.png"
    assert cell_name(9, "left", "candidate", "rest", False) == "09-left_candidate_rest.png"


def test_prompt_has_training_layout(tmp_path):
    case, cells = make_case(tmp_path)
    prompt, images = judge_prompt(case, {"parts": ["base", "door"], "joints": ["hinge"]}, cells)
    assert prompt.startswith(judges.PROMPT_PREFIX + "{\"context\"")
    payload = json.loads(prompt[prompt.index('{"context"'):])
    assert list(payload) == ["context", "remaining", "history", "image_order", "taxonomy_ids", "training_objective"]
    assert payload["remaining"] == 0 and len(payload["taxonomy_ids"]) == 29
    assert payload["image_order"] == [f"images/case1/{c.name}" for c in cells]
    assert [h["step"] for h in payload["history"]] == list(range(len(cells) - 1))
    assert payload["history"][0]["image_ref"] == "images/case1/01-reference_mid.png"
    assert images == [c.path for c in cells]


def test_prompt_lists_the_case_roster_not_the_harness_order(tmp_path):
    case, cells = make_case(tmp_path)
    case.parts, case.joints = ("door", "base"), ("hinge",)
    prompt, _ = judge_prompt(case, INFO, cells)
    assert json.loads(prompt[prompt.index('{"context"'):])["context"]["parts"] == ["door", "base"]


@real_pack
def test_prompt_matches_the_export_byte_for_byte_for_every_pack_case(tmp_path):
    exported = {}
    for line in EPISODES.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            exported[record["id"].split(":")[0]] = record["messages"][0]["content"]
    cases = load_pack(PACK, LABELS, EPISODES)
    assert len(cases) == 58
    for case in cases:
        cells = sweep_layout(case, {"reference": tmp_path / "reference", "candidate": tmp_path / "candidate"})
        harness = {"parts": sorted(case.parts, reverse=True), "joints": sorted(case.joints, reverse=True)}   # not the export order
        prompt, _ = judge_prompt(case, harness, cells)
        assert prompt == exported[case.uid], case.uid


def test_verdict_rules():
    parts = ["base", "door"]
    ok = {"verdict": {"status": "fail", "confidence": 0.9, "findings": [{"defect_id": "kinematics.joint_axis", "parts": ["door"],
          "joint_id": "hinge", "severity": 0.5, "confidence": 0.9, "evidence_steps": [1, 2], "description": "d", "suggested_fix": "f"}],
          "summary": "s", "limitations": []}}
    verdict, error = validate_verdict("```json\n" + json.dumps(ok) + "```", parts, {0, 1, 2})
    assert error is None and verdict["findings"][0]["joint_id"] == "hinge" and verdict["findings"][0]["evidence_steps"] == [1, 2]
    bad = [("not json", "invalid JSON"),
           (json.dumps({"verdict": {"status": "fail", "findings": []}}), "fail without findings"),
           (json.dumps({"verdict": {"status": "pass", "findings": ok["verdict"]["findings"]}}), "pass with findings or without observations"),
           (json.dumps({"verdict": {"status": "fail", "findings": [{**ok["verdict"]["findings"][0], "parts": ["ghost"]}]}}), "unknown part id"),
           (json.dumps({"verdict": {"status": "fail", "findings": [{**ok["verdict"]["findings"][0], "evidence_steps": [7]}]}}),
            "finding cites a step that does not exist"),
           (json.dumps({"action": {"type": "request_view"}}), "expected one verdict object")]
    for text, reason in bad:
        verdict, error = validate_verdict(text, parts, {0, 1, 2})
        assert error == reason and verdict["status"] == "uncertain" and verdict["findings"] == []


class FakeModel:
    def __init__(self, response: str):
        self.response, self.identity, self.calls = response, {"base": "fake"}, 0

    def __call__(self, prompt, images):
        self.calls += 1
        return self.response, {"input_tokens": 1, "output_tokens": 1}


FINDING = {"defect_id": "kinematics.joint_axis", "parts": ["door"], "joint_id": "hinge", "severity": 0.5, "confidence": 0.9,
           "evidence_steps": [0], "description": "d", "suggested_fix": "f"}


def test_every_judge_hands_the_generator_the_same_images(tmp_path, monkeypatch):
    case, cells = make_case(tmp_path, views=("front", "left"))
    monkeypatch.setattr(judges, "sweep_cells", lambda session, case, source, cache: (cells, INFO))
    model = FakeModel(json.dumps({"verdict": {"status": "fail", "confidence": 0.9, "findings": [FINDING], "summary": "s", "limitations": []}}))
    packets = {}
    for judge in (NoneJudge(), PixelJudge(), GoldJudge(), VlmJudge("zs8b_sweep", model, "sweep"),
                  VlmJudge("zs8b_screenshot", model, "screenshot")):
        folder = tmp_path / judge.judge_id; folder.mkdir()
        packets[judge.judge_id] = judge.bind(None, case, tmp_path)("src", {"passed": False, "checks": []}, folder)
    by_key = {(c.view, c.state, c.kind): c for c in cells}
    group = [by_key[(v, s, k)] for v in ("front", "left") for s in ("rest", "mid", "limit") for k in ("reference", "candidate")]
    common = {"task", "observations", "reference_condition"}
    sections = {"none": set(), "pixel": {"pixel_difference"}, "gold": {"verdict", "evidence_step_cells"},
                "zs8b_sweep": {"verdict", "evidence_step_cells"}, "zs8b_screenshot": {"verdict", "evidence_step_cells"}}
    for judge_id, packet in packets.items():
        assert packet.images == tuple(c.path for c in group), judge_id
        assert [o["image_ref"] for o in packet.public["observations"]] == [c.name for c in group], judge_id
        assert packet.public["observations"] == packets["none"].public["observations"], judge_id
        assert set(packet.public) == common | sections[judge_id], judge_id
        assert packet.observation_cost == 6
    # a verdict's step k is the judge's image k+1: the sweep judge saw the front reference mid, the screenshot judge the candidate rest
    assert packets["zs8b_sweep"].public["evidence_step_cells"] == [{"step": 0, "view": "front", "state": "mid", "kind": "reference"}]
    assert packets["zs8b_screenshot"].public["evidence_step_cells"] == [{"step": 0, "view": "front", "state": "rest", "kind": "candidate"}]


def test_pixel_judge_reports_cells_without_defect_types(tmp_path, monkeypatch):
    case, cells = make_case(tmp_path)
    monkeypatch.setattr(judges, "sweep_cells", lambda session, case, source, cache: (cells, INFO))
    folder = tmp_path / "pixel"; folder.mkdir()
    packet = PixelJudge(0.01).bind(None, case, tmp_path)("src", {}, folder)
    report = packet.public["pixel_difference"]["cells"]
    assert [(c["view"], c["state"], c["differs"]) for c in report] == [("front", "rest", False), ("front", "mid", False), ("front", "limit", True)]
    assert "defect" not in packet.public["pixel_difference"]["summary"].lower().replace("no defect", "")


def test_gold_judge_uses_audit_and_cites_candidate_steps(tmp_path, monkeypatch):
    case, cells = make_case(tmp_path)
    monkeypatch.setattr(judges, "sweep_cells", lambda session, case, source, cache: (cells, INFO))
    failing = {"passed": False, "checks": [{"invariants": [{"name": "chamfer_m", "passed": False, "parts": ["door"], "state": "joint:hinge:limit"}]}]}
    folder = tmp_path / "gold"; folder.mkdir()
    packet = GoldJudge().bind(None, case, tmp_path)("src", failing, folder)
    verdict = packet.public["verdict"]
    assert verdict["status"] == "fail" and verdict["findings"][0]["joint_id"] == "hinge"
    assert verdict["findings"][0]["evidence_steps"] == [2, 3, 4]      # candidate rest, mid, limit
    assert [c["state"] for c in packet.public["evidence_step_cells"]] == ["rest", "mid", "limit"]
    assert "hinge" in verdict["findings"][0]["suggested_fix"]
    folder = tmp_path / "gold-pass"; folder.mkdir()
    packet = GoldJudge().bind(None, case, tmp_path)("src", {"passed": True, "checks": []}, folder)
    assert packet.public["verdict"]["status"] == "pass" and packet.public["verdict"]["findings"] == []
    case.labels = []
    with pytest.raises(ValueError, match="labels"):
        GoldJudge().bind(None, case, tmp_path)


def test_verdict_cache_is_keyed_by_adapter_content(tmp_path, monkeypatch):
    case, cells = make_case(tmp_path)
    monkeypatch.setattr(judges, "sweep_cells", lambda session, case, source, cache: (cells, INFO))
    adapter = tmp_path / "adapter"; adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"seed 1")
    first = JudgeModel(tmp_path / "Qwen3-VL-2B-Instruct", adapter)
    (adapter / "adapter_model.safetensors").write_bytes(b"seed 2")
    second = JudgeModel(tmp_path / "Qwen3-VL-2B-Instruct", adapter)
    calls = []
    monkeypatch.setattr(JudgeModel, "__call__", lambda self, prompt, images: (
        calls.append(self.identity["adapter_sha256"]) or json.dumps({"verdict": {"status": "uncertain", "findings": []}}), {}))
    for index, model in enumerate((first, second, first)):
        folder = tmp_path / f"round-{index}"; folder.mkdir()
        VlmJudge("ours_2b_evidence", model).bind(None, case, tmp_path)("src", {}, folder)
    assert calls == [first.identity["adapter_sha256"], second.identity["adapter_sha256"]]


def test_pack_loader_on_a_synthetic_pack(tmp_path):
    import hashlib
    cand, ref = "export default function createObject(T){ /* bad */ }", "export default function createObject(T){ /* good */ }"
    (tmp_path / "cases" / "ms-1").mkdir(parents=True)
    (tmp_path / "cases" / "ms-1" / "candidate.ts").write_text(cand)
    (tmp_path / "cases" / "ms-1" / "reference.ts").write_text(ref)
    (tmp_path / "cases" / "ms-1" / "reference.spec.json").write_text(json.dumps({"parts": [{"id": "base"}, {"id": "link_1"}, {"id": "link_0"}],
                                                                                   "joints": [{"id": "joint_1"}, {"id": "joint_0"}]}))
    (tmp_path / "manifest.json").write_text(json.dumps({"format": PACK_FORMAT}))
    entry = {"case_id": "ms-1", "difficulty": "medium", "target_joint_id": "joint_0", "views": ["oblique_fr"],
             "state_values": {"rest": 0.0, "mid": 0.3, "limit": 0.6},
             "observation_cells": [{"view_id": "oblique_fr", "state": s, "candidate_image": f"images/ms-1/0{i+3}-candidate_{s}.png",
                                    "reference_image": f"images/ms-1/0{i}-reference_{s}.png"} for i, s in enumerate(("rest", "mid", "limit"))],
             "programs": {k: {"typescript": f"cases/ms-1/{k}.ts", "spec": f"cases/ms-1/{k}.spec.json",
                              "typescript_sha256": hashlib.sha256(t.encode()).hexdigest()} for k, t in (("candidate", cand), ("reference", ref))}}
    (tmp_path / "cases.json").write_text(json.dumps([entry]))
    (tmp_path / "labels.jsonl").write_text(json.dumps({"id": "ms-1", "labels": [{"defect_id": "kinematics.direction_flip", "parts": ["link_0"], "severity": 0.9, "params": {"joint": "joint_0"}}]}) + "\n")
    (tmp_path / "framing.json").write_text(json.dumps({"cases": {"ms-1": {"center": [0.0, 0.1, 0.2], "radius": 1.5}}}))
    case, = load_pack(tmp_path, tmp_path / "labels.jsonl", framing_path=tmp_path / "framing.json")
    assert case.initial == cand and case.reference.source == ref and case.tier == "medium"
    assert case.sweep.views == ("oblique_fr",) and case.sweep.joints == ("joint_0",) and case.sweep.states == ("rest", "mid", "limit")
    assert dict(case.sweep.values) == {"mid": 0.3, "limit": 0.6}
    assert case.labels[0]["defect_id"] == "kinematics.direction_flip" and case.grid[1] == {"view": "oblique_fr", "joint": "joint_0", "detent": "mid"}
    assert case.framing == Framing((0.0, 0.1, 0.2), 1.5)
    assert case.parts == ("base", "link_1", "link_0") and case.joints == ("joint_1", "joint_0")
    (tmp_path / "cases" / "ms-1" / "candidate.ts").write_text(cand + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_pack(tmp_path, tmp_path / "labels.jsonl", framing_path=tmp_path / "framing.json")


@real_pack
def test_real_pack_loads_with_training_cell_names():
    cases = load_pack(PACK, LABELS, EPISODES)
    assert len(cases) == 58 and all(c.labels for c in cases)
    entries = {e["case_id"]: e for e in json.loads((PACK / "cases.json").read_text())}
    for case in cases:
        names = [c.name for c in sweep_layout(case, {"reference": Path("r"), "candidate": Path("c")})]
        expected = sorted(p.split("/")[-1] for c in entries[case.uid]["observation_cells"] for p in (c["reference_image"], c["candidate_image"]))
        assert sorted(names) == expected, case.uid
        assert any(len(x) for x in case.exposures), case.uid     # every pack case has an exposed label (index 0 is falsy)
