"""Reject dataset mutation during input loading before initializing a model."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.unit.test_frozen_evaluation_cli import make_run, sha
from turnitover.oracle.evaluation_snapshot import load_snapshot


@pytest.mark.parametrize("changed", ["annotations", "split", "manifest"])
def test_dataset_changes_during_loading_fail_before_model_initialization(tmp_path, monkeypatch, changed):
    dataset, existing = make_run(tmp_path)
    frozen = load_snapshot(existing)
    records = frozen["records"]
    originals = {row["id"]: row["original_images"] for row in frozen["image_paths"]}
    for record in records:
        record["images"] = originals[record["id"]]
    split_file = dataset / "validation.episodes.jsonl"
    split_file.write_text("".join(json.dumps(record) + "\n" for record in records))
    expected_manifest_hash = sha(dataset / "manifest.json")
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}")

    path = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_qwen_episode_sft.py"
    spec = importlib.util.spec_from_file_location("evaluation_capture_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def load_and_mutate(*args):
        for record in records:
            yield record, []
        if changed == "manifest":
            manifest_path = dataset / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["changed_during_loading"] = True
            manifest_path.write_text(json.dumps(manifest))
        else:
            mutation_path = dataset / "annotations.private.jsonl" if changed == "annotations" else split_file
            with mutation_path.open("a") as stream:
                stream.write("\n")

    monkeypatch.setattr(module, "load_training_examples", load_and_mutate)
    torch = ModuleType("torch")
    torch.manual_seed = Mock()
    torch.cuda = SimpleNamespace(manual_seed_all=Mock())
    torch.float32 = "float32"
    monkeypatch.setitem(sys.modules, "torch", torch)

    class FakeGenerationConfig:
        @classmethod
        def from_model_config(cls, config):
            return cls()

        def to_dict(self):
            return {"max_new_tokens": self.max_new_tokens, "do_sample": self.do_sample}

    processor_init = Mock(side_effect=AssertionError("Processor must not initialize after dataset mutation"))
    model_init = Mock(side_effect=AssertionError("Model must not initialize after dataset mutation"))
    transformers = ModuleType("transformers")
    transformers.AutoConfig = SimpleNamespace(from_pretrained=Mock(return_value=object()))
    transformers.AutoProcessor = SimpleNamespace(from_pretrained=processor_init)
    transformers.GenerationConfig = FakeGenerationConfig
    transformers.Qwen3VLForConditionalGeneration = SimpleNamespace(from_pretrained=model_init)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    output = tmp_path / "captured-evaluation"
    with pytest.raises(ValueError, match="Dataset changed"):
        module.evaluate(dataset, model_path, output, adapter=None, max_new_tokens=512,
                        dataset_manifest_sha256=expected_manifest_hash, device="cpu")
    processor_init.assert_not_called()
    model_init.assert_not_called()
    torch.manual_seed.assert_not_called()
    report = json.loads((output / "metrics.json").read_text())
    assert report["status"] == "incomplete"
    assert report["completed_cases"] == 0
    assert not (output / "predictions.jsonl").exists()
