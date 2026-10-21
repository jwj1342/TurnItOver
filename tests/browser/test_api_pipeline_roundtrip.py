"""Scripted API fixture + real Chromium integration; this is not an API experiment."""
import dataclasses
import json
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from turnitover.api_pipeline.fixture import TASK, controlled_cases, reference_program
from turnitover.api_pipeline.paired import prepare, run_pairs
from turnitover.api_pipeline.scoring import score_pairs
from turnitover.models.client import ModelResult
from turnitover.models.config import ModelConfig


pytestmark = pytest.mark.browser


class ScriptedAPIFixture:
    """Returns authored code and scripted verdicts; never sends network requests."""

    def __init__(self, ledger_root):
        self.config = ModelConfig("judge", "openai_compatible", "scripted-api-fixture",
                                  "http://localhost", "")
        self.returned_model = self.config.model
        self.public_config = {**self.config.public(), "role": "shared", "temperature": 0,
                              "scripted_fixture_not_experiment": True}
        self.ledger_root = ledger_root
        self.ledger_root.mkdir()
        self.calls = []
        self.verdicts = Counter()

    def scoped(self, label):
        def call(prompt, images):
            assert all(Path(path).is_file() for path in images)
            if label == "shared/generator":
                # Authored reference reuse is intentional only inside this test fixture.
                text = "```ts\n" + reference_program().source + "\n```"
            elif label.endswith("/generator"):
                assert "initialX + 0.15" in prompt
                text = json.dumps({"edits": [{"old": "initialX + 0.15",
                                              "new": "initialX + 0.0"}]})
            else:
                assert label.endswith("/judge") or label.endswith("/judge/recovery")
                assert "current_program" not in prompt and "gt_validation" not in prompt
                data = json.loads(prompt.split("Observed task data:\n", 1)[1].split("\n", 1)[0])
                if data["remaining"]:
                    assert "/active/" in label
                    action = ({"type": "request_view", "view_id": "front"} if not data["history"]
                              else {"type": "actuate_joint", "joint_id": "drawer_slide", "detent": "mid"})
                    text = json.dumps({"action": action})
                else:
                    first_round = self.verdicts[label] == 0
                    self.verdicts[label] += 1
                    findings = [{"defect_id": "structure.part_offset", "parts": ["drawer"],
                                 "joint_id": None, "severity": 0.5, "confidence": 1,
                                 "evidence_steps": [0 if "/active/" in label else 1],
                                 "description": "The drawer is offset on X in the observed view.",
                                 "suggested_fix": "Restore the public drawer rest X position."}] if first_round else []
                    text = json.dumps({"verdict": {
                        "status": "fail" if first_round else "pass", "confidence": 1,
                        "findings": findings, "summary": "Scripted API fixture verdict.",
                        "limitations": ["Scripted API fixture; no model capability is measured."]}})
            result = ModelResult(text, self.returned_model, {"total_tokens": 7}, "completed", 1)
            self.calls.append({"scope": label, "usage": result.usage, "elapsed_ms": 1,
                               "provider_cost": None, "scripted_fixture_not_experiment": True})
            (self.ledger_root / "ledger.json").write_text(json.dumps({"calls": self.calls, "returned_model": self.returned_model, "request_count": len(self.calls)}))
            return result
        return call


def test_scripted_api_fixture_real_chromium_paired_roundtrip(tmp_path, render_config, views):
    prepared, paired = tmp_path / "prepared", tmp_path / "paired"
    api = ScriptedAPIFixture(tmp_path / "paired-api")
    base = prepare(prepared, render_config, views, api, TASK,
                   "SCRIPTED API FIXTURE: return the authored reference; this is not an experiment.",
                   reference_program(), runtime_repairs=0)
    cases = {"c01": controlled_cases(base)["c01"]}
    # The common initial is fixed; comparison transport/output limits may differ,
    # but both arms and their repairs must use the same comparison configuration.
    api.config = dataclasses.replace(api.config, max_tokens=8192, timeout=300)
    api.public_config.update(max_tokens=8192, timeout=300)
    result = run_pairs(prepared, paired, cases, render_config, views, api,
                       budget=2, revisions=1, runtime_repairs=0)
    assert result["status"] == "complete"
    assert result["model"]["max_tokens"] == 8192
    assert result["shared_generation_model"]["max_tokens"] == 4096
    assert result["pairs"]["c01"]["initial_inputs_equal"]

    for mode in ("active", "fixed"):
        root = paired / "c01" / mode
        arm = json.loads((root / "result.json").read_text())
        assert arm["accepted"] and arm["status"] == "accepted"
        assert len(arm["rounds"]) == 2
        assert arm["model_calls"]["visual_revision"] == 1
        assert arm["model_calls"]["runtime_repair"] == 0
        assert "initialX + 0.0" in (root / "final.ts").read_text()
        for index in range(2):
            folder = root / f"round-{index:03d}" / "verifier"
            verifier = json.loads((folder / "result.json").read_text())
            trajectory = [json.loads(line) for line in (folder / "trajectory.jsonl").read_text().splitlines()]
            assert verifier["spent"] == 2 and len(trajectory) == 2
            assert verifier["verdict"]["status"] == ("fail" if index == 0 else "pass")
            for observation in trajectory:
                if observation["image_ref"]:
                    with Image.open(folder / observation["image_ref"]) as image:
                        assert image.format == "PNG"
                        assert image.size == (render_config.width, render_config.height)
                        image.verify()
            if mode == "active":
                assert trajectory[0]["action"]["view_id"] == "front"
                assert trajectory[1]["action"]["joint_id"] == "drawer_slide"
                assert trajectory[1]["action"]["detent"] == "mid"
            else:
                assert trajectory[0]["action"]["property"] == "stats"
                assert trajectory[1]["action"]["view_id"] == "front"
                assert (folder / "observation_plan.json").is_file()

    scores = score_pairs(prepared, paired, render_config, views)
    assert scores["labels_valid"] and scores["base_quality"]["code_correct"]
    for mode in ("active", "fixed"):
        score = scores["pairs"]["c01"][mode]
        assert score["visual_repair_calls"] == 1 and score["first_correct_round"] == 1
        assert score["final_quality"]["code_correct"]
        assert score["total_observations"] == 4
        assert score["api_cost"]["requests"] == (7 if mode == "active" else 3)
        assert score["api_cost"]["tokens"] == (49 if mode == "active" else 21)
        assert score["api_cost"]["provider_cost_total"] is None
        initial = score["rounds"][0]
        assert initial["labels_valid"] and not initial["quality"]["code_correct"]
        assert initial["diagnosis"]["detection_correct"] and initial["diagnosis"]["type_exact"]
        assert initial["diagnosis"]["part_localization"] == {"tp": 1, "fp": 0, "fn": 0}
        assert initial["diagnosis"]["citations_exist"]
        assert initial["diagnosis"]["geometric_witness_consistent_findings"] == 1

    calls_before = len(api.calls)
    api.public_config['temperature'] = 1
    with pytest.raises(ValueError, match='API configurations differ'):
        run_pairs(prepared, tmp_path / 'changed-model-config', cases, render_config, views, api, budget=2)
    assert len(api.calls) == calls_before
    api.public_config['temperature'] = 0
    (prepared / "task.txt").write_text(TASK + "\nTampered after preparation.")
    with pytest.raises(ValueError, match="Prepared input changed: task.txt"):
        run_pairs(prepared, tmp_path / "tampered-run", cases, render_config, views, api, budget=2)
    assert len(api.calls) == calls_before
    assert not (tmp_path / "tampered-run").exists()
