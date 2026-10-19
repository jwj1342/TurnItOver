import base64
from dataclasses import asdict
import io
import json
from types import SimpleNamespace

import pytest

from scripts.action_sft_metrics import merge, summarize_rows
from scripts.evaluate_action_sft_learning import target_mass
from turnitover.models.action_qwen import RemoteModel
from turnitover.models.client import ModelError, ModelResult
from turnitover.oracle.supervision import action_examples
from turnitover.verifier.contracts import Context


def test_learning_credits_conflicting_targets_by_probability():
    context = Context("inspect", ("body",), (), ("front", "top"), (), ("request_view",), 100, 10)
    group = []
    for view, weight in (("front", .25), ("top", .75)):
        row = next(action_examples("uid", asdict(context), "ref.png", [
            {"step": 0, "action": {"type": "request_view", "view_id": view}}], 6))
        row["loss_weight"] = weight
        group.append((row, []))
    assert target_mass('{"action":{"type":"request_view","view_id":"front"}}', group) == .25
    assert target_mass('{"action":{"type":"request_view","view_id":"top"}}', group) == .75
    with pytest.raises(ValueError):
        target_mass('{"verdict":{"status":"pass"}}', group)


def row(uid="a", mode="fixed", labels=1, exposed=1):
    return dict(id=uid, asset_id="object1", mode=mode, budget=2, labels=labels,
                exposed=exposed, valid_evidence=bool(exposed), all_evidence=labels == exposed,
                diagnostic_exact=True, spent=2, action_attempts=2, legal_actions=2, folder=uid)


def test_evidence_denominators_separate_clean_and_defective():
    summary, _, _ = summarize_rows([row(labels=2), row("b", labels=0, exposed=0)])
    fixed = summary["k2/fixed"]
    assert fixed["valid_evidence_rate"] == .5
    assert fixed["nonclean_any_evidence_rate"] == 1.
    assert fixed["nonclean_all_evidence_rate"] == 0.
    assert fixed["exposure_recall"] == .5
    assert fixed["independent_objects"] == 1


def test_merge_requires_matched_cases_and_preserves_pairing(tmp_path):
    runs = [tmp_path / "base", tmp_path / "sft"]
    config = dict(source_sha256={"raw": "abc"}, reference_sha256={"a": "xyz"},
                  budgets=[2], action_tokens=128, final_tokens=1024)
    for root, modes in zip(runs, (("zero_shot", "fixed", "random"), ("action_sft",))):
        root.mkdir()
        (root / "run_config.json").write_text(json.dumps(config))
        (root / "metrics.json").write_text(json.dumps(dict(status="complete", score_eligible=True,
            episodes=[row(mode=mode, exposed=int(mode == "action_sft")) for mode in modes],
            offline_privileged_oracle=[])))
    result = merge(runs, tmp_path / "merged.json")
    assert len(result["per_object_paired_deltas"]) == 3
    assert all(r["delta_action_sft_minus_baseline"]["valid_evidence"] == 1.
               for r in result["per_object_paired_deltas"])
    config["reference_sha256"]["a"] = "wrong"
    (runs[1] / "run_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="different public"):
        merge(runs, tmp_path / "bad.json")


def test_remote_transport_sends_only_public_prompt_images_and_flags(tmp_path, monkeypatch):
    expected = ModelResult("output", "base", {}, "completed", 1.)
    def urlopen(request, timeout):
        payload = json.loads(request.data)
        assert set(payload) == {"prompt", "images", "enabled", "final"}
        assert payload["prompt"] == "public"
        assert base64.b64decode(payload["images"][0]) == b"png-bytes"
        assert payload["enabled"] is False and payload["final"] is True
        return io.BytesIO(json.dumps(asdict(expected)).encode())
    def build_opener(proxy):
        assert proxy.proxies == {}
        return SimpleNamespace(open=urlopen)
    monkeypatch.setattr("urllib.request.build_opener", build_opener)
    image = tmp_path / "public.png"
    image.write_bytes(b"png-bytes")
    assert RemoteModel("http://localhost").call(False, final=True)("public", [image]) == expected
    def failed(*args, **kwargs):
        raise OSError("transport failed")
    monkeypatch.setattr("urllib.request.build_opener", lambda proxy: SimpleNamespace(open=failed))
    with pytest.raises(ModelError, match="transport failed"):
        RemoteModel("http://localhost").call(False)("public", [])


def test_remote_transport_rejects_external_unauthenticated_endpoints():
    for endpoint in ("http://example.com", "http://127.0.0.1.example.com", "https://localhost", "http://user@localhost"):
        with pytest.raises(ValueError, match="loopback"):
            RemoteModel(endpoint)
