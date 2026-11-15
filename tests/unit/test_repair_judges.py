import json
from pathlib import Path

import pytest
from PIL import Image

from turnitover.core.program import ObjectProgram
from turnitover.repair import judges
from turnitover.repair.judges import (Case, Cell, GoldJudge, NoneJudge, PixelJudge, Sweep, cell_name, judge_prompt,
                                      validate_verdict)

EXPORT = Path("/scratch/jwj/tio-runs/data900/three-tier-900-sft-v1/episode_export/validation.episodes.jsonl")


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


@pytest.mark.skipif(not EXPORT.exists(), reason="three-tier-900 export not available")
def test_prompt_matches_a_real_export_record_byte_for_byte(tmp_path):
    records = [json.loads(l) for l in EXPORT.read_text().splitlines() if l.strip()]
    for difficulty in ("medium", "hard"):
        record = next(r for r in records if r["difficulty"] == difficulty)
        content = record["messages"][0]["content"]
        payload = json.loads(content[content.index('{"context"'):])
        uid = record["id"].split(":")[0]
        views = payload["context"]["views"]
        states = ["rest", "mid", "limit"]
        cells = []
        for name in payload["image_order"]:
            base = name.split("/")[-1]
            view = views[0] if len(views) == 1 else base.split("-", 1)[1].rsplit("_", 2)[0]   # NN-<view>_<kind>_<state>.png
            kind = "reference" if "reference" in base else "candidate"
            state = base.rsplit("_", 1)[1][:-4]
            cells.append(Cell(view, state, kind, base, tmp_path / base))
        case = Case(uid, difficulty, "", ObjectProgram("x"), None, [], Sweep(tuple(views), ("joint_0",), tuple(states)))
        prompt, _ = judge_prompt(case, {"parts": payload["context"]["parts"], "joints": payload["context"]["joints"]}, cells)
        assert prompt == content, difficulty


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


def test_pixel_and_none_judges_shape_feedback_without_defect_types(tmp_path, monkeypatch):
    case, cells = make_case(tmp_path)
    monkeypatch.setattr(judges, "sweep_cells", lambda session, case, source, cache: (cells, {"parts": ["base", "door"], "joints": ["hinge"]}))
    folder = tmp_path / "pixel"; folder.mkdir()
    packet = PixelJudge(0.01).bind(None, case, tmp_path)("src", {}, folder)
    report = packet.public["pixel_difference"]["cells"]
    assert [c["differs"] for c in report] == [False, False, True]
    assert "verdict" not in packet.public and "defect" not in json.dumps(packet.public["pixel_difference"]["summary"]).lower().replace("no defect", "")
    assert [p.name for p in packet.images] == ["reference_front_rest.png", "candidate_front_limit.png"]
    assert packet.public["observations"][0]["step"] == -1 and packet.public["observations"][1]["image_ref"] == "05-candidate_limit.png"
    folder = tmp_path / "none"; folder.mkdir()
    packet = NoneJudge().bind(None, case, tmp_path)("src", {}, folder)
    assert [p.name for p in packet.images] == ["reference_front_rest.png", "candidate_front_rest.png"]
    assert packet.observation_cost == 1


def test_gold_judge_uses_audit_and_cites_candidate_steps(tmp_path, monkeypatch):
    case, cells = make_case(tmp_path)
    monkeypatch.setattr(judges, "sweep_cells", lambda session, case, source, cache: (cells, {"parts": ["base", "door"], "joints": ["hinge"]}))
    failing = {"passed": False, "checks": [{"invariants": [{"name": "chamfer_m", "passed": False, "parts": ["door"], "state": "joint:hinge:limit"}]}]}
    folder = tmp_path / "gold"; folder.mkdir()
    packet = GoldJudge().bind(None, case, tmp_path)("src", failing, folder)
    verdict = packet.public["verdict"]
    assert verdict["status"] == "fail" and verdict["findings"][0]["joint_id"] == "hinge"
    assert verdict["findings"][0]["evidence_steps"] == [2, 3, 4]      # candidate rest, mid, limit
    assert "hinge" in verdict["findings"][0]["suggested_fix"]
    assert packet.images[0] == cells[0].path and len(packet.images) == 4
    folder = tmp_path / "gold-pass"; folder.mkdir()
    packet = GoldJudge().bind(None, case, tmp_path)("src", {"passed": True, "checks": []}, folder)
    assert packet.public["verdict"]["status"] == "pass" and packet.public["verdict"]["findings"] == []


PACK = Path("/scratch/jwj/tio-runs/data900/repair58/three-tier-900-validation-repair-58-v1")
LABELS = Path("/scratch/jwj/tio-runs/data900/three-tier-900-sft-v1/episode_export/annotations.private.jsonl")
EPISODES = EXPORT


def test_pack_loader_on_a_synthetic_pack(tmp_path):
    import hashlib
    from turnitover.repair.judges import PACK_FORMAT, load_pack
    cand, ref = "export default function createObject(T){ /* bad */ }", "export default function createObject(T){ /* good */ }"
    (tmp_path / "cases" / "ms-1").mkdir(parents=True)
    (tmp_path / "cases" / "ms-1" / "candidate.ts").write_text(cand)
    (tmp_path / "cases" / "ms-1" / "reference.ts").write_text(ref)
    (tmp_path / "manifest.json").write_text(json.dumps({"format": PACK_FORMAT}))
    entry = {"case_id": "ms-1", "difficulty": "medium", "target_joint_id": "joint_0", "views": ["oblique_fr"],
             "state_values": {"rest": 0.0, "mid": 0.3, "limit": 0.6},
             "observation_cells": [{"view_id": "oblique_fr", "state": s, "candidate_image": f"images/ms-1/0{i+3}-candidate_{s}.png",
                                    "reference_image": f"images/ms-1/0{i}-reference_{s}.png"} for i, s in enumerate(("rest", "mid", "limit"))],
             "programs": {k: {"typescript": f"cases/ms-1/{k}.ts", "typescript_sha256": hashlib.sha256(t.encode()).hexdigest()}
                          for k, t in (("candidate", cand), ("reference", ref))}}
    (tmp_path / "cases.json").write_text(json.dumps([entry]))
    (tmp_path / "labels.jsonl").write_text(json.dumps({"id": "ms-1", "labels": [{"defect_id": "kinematics.direction_flip", "parts": ["link_0"], "severity": 0.9, "params": {"joint": "joint_0"}}]}) + "\n")
    case, = load_pack(tmp_path, tmp_path / "labels.jsonl")
    assert case.initial == cand and case.reference.source == ref and case.tier == "medium"
    assert case.sweep.views == ("oblique_fr",) and case.sweep.joints == ("joint_0",) and case.sweep.states == ("rest", "mid", "limit")
    assert dict(case.sweep.values) == {"mid": 0.3, "limit": 0.6}
    assert case.labels[0]["defect_id"] == "kinematics.direction_flip" and case.grid[1] == {"view": "oblique_fr", "joint": "joint_0", "detent": "mid"}
    (tmp_path / "cases" / "ms-1" / "candidate.ts").write_text(cand + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_pack(tmp_path, tmp_path / "labels.jsonl")


@pytest.mark.skipif(not (PACK.exists() and LABELS.exists() and EPISODES.exists()), reason="her repair pack not available")
def test_real_pack_loads_with_training_cell_names():
    from turnitover.repair.judges import load_pack
    cases = load_pack(PACK, LABELS, EPISODES)
    assert len(cases) == 58 and all(c.labels for c in cases)
    entries = {e["case_id"]: e for e in json.loads((PACK / "cases.json").read_text())}
    for case in cases:
        single = len(case.sweep.views) == 1
        names, index = [], 0
        for view in case.sweep.views:
            for kind in ("reference", "candidate"):
                for state in case.sweep.states:
                    names.append(cell_name(index, view, kind, state, single)); index += 1
        expected = sorted(p.split("/")[-1] for c in entries[case.uid]["observation_cells"] for p in (c["reference_image"], c["candidate_image"]))
        assert sorted(names) == expected, case.uid
        assert any(len(x) for x in case.exposures), case.uid     # every pack case has an exposed label (index 0 is falsy)
