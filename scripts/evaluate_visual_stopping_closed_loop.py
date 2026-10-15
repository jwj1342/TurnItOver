#!/usr/bin/env python3
"""Run a small real-browser v5 loop with fixed actions and model-chosen stopping."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image

from scripts.run_oracle_supervision import _benchmark_provenance
from turnitover.assets.source import make_source
from turnitover.config import load_generate_config, parse_action
from turnitover.core.actions import Detent, detent_value
from turnitover.core.program import ObjectProgram
from turnitover.dataset.prepare import records
from turnitover.oracle.evidence import replay_corruptions
from turnitover.oracle.supervision import action_json
from turnitover.oracle.visual_decisions import BUDGET, decide, image_evidence, make_row
from turnitover.policy.loop import execute_observation
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views
from turnitover.verifier.contracts import parse_json


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def select_traces(traces: list[dict], count: int = 6) -> list[dict]:
    """Predeclare late/early evidence strata without using any model prediction."""
    eligible = []
    for trace in traces:
        if trace["split"] != "validation" or any(
                row["state"] == "ambiguous" for row in trace["evidence"]):
            continue
        states = [row["state"] for row in trace["evidence"]]
        first = states.index("sufficient") + 1 if "sufficient" in states else BUDGET
        kind = ("active" if "sufficient" in states and first < BUDGET else
                "forced_fail" if "sufficient" in states else "forced_uncertain")
        eligible.append((trace, first, kind))
    selected, assets = [], set()
    active = [(trace, step) for trace, step, kind in eligible if kind == "active"]
    for step in sorted({step for _, step in active}):
        candidates = sorted((trace for trace, value in active if value == step), key=lambda row: row["id"])
        choice = next((row for row in candidates if row["data_identity"]["asset_id"] not in assets), candidates[0])
        selected.append(choice)
        assets.add(choice["data_identity"]["asset_id"])
        if len(selected) == count // 2:
            break
    forced = sorted((trace for trace, _, kind in eligible if kind == "forced_uncertain"),
                    key=lambda row: row["id"])
    for prefer_new in (True, False):
        for trace in forced:
            if trace in selected:
                continue
            asset = trace["data_identity"]["asset_id"]
            if prefer_new and asset in assets:
                continue
            selected.append(trace)
            assets.add(asset)
            if len(selected) == count:
                return selected
    if len(selected) != count:
        raise ValueError(f"Could not select {count} closed-loop traces")
    return selected


class StopModel:
    def __init__(self, model_path: Path, adapter: Path, max_new_tokens: int):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            adapter, local_files_only=True, trust_remote_code=False)
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
        self.model = PeftModel.from_pretrained(base, adapter, local_files_only=True).eval()
        self.max_new_tokens = max_new_tokens

    def predict(self, record: dict, images: list[Image.Image]) -> tuple[str | None, str | None, str, str | None]:
        from turnitover.oracle.training import tokenize_prompt

        _, inputs = tokenize_prompt(self.processor, record, images)
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = output[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        try:
            value = parse_json(text)
            decision = value["stop_decision"]["decision"]
            outcome = value["stop_decision"].get("outcome")
            if decision not in ("continue", "stop"):
                raise ValueError("invalid stop decision")
            if decision == "continue" and outcome is not None:
                raise ValueError("continue has outcome")
            if decision == "stop" and outcome not in ("fail", "uncertain"):
                raise ValueError("stop lacks outcome")
            return decision, outcome, text, None
        except (KeyError, TypeError, ValueError) as exc:
            return None, None, text, str(exc)


def run(args) -> dict:
    if args.out.exists():
        raise ValueError(f"Output must not exist: {args.out}")
    _benchmark_provenance(args.benchmark)
    cfg = load_generate_config(args.config, repo_root=args.asset_workspace)
    cfg = replace(cfg, render=replace(
        cfg.render, harness_dir=args.harness_workspace / "web/dist",
        esbuild_bin=args.harness_workspace / "web/node_modules/.bin/esbuild"))
    source = make_source(**cfg.asset_source)
    traces = select_traces(_read_jsonl(args.dataset / "trajectories.jsonl"), args.episodes)
    raw = {hashlib.sha256(f"{sample.source_id}:{sample.sample_id}".encode()).hexdigest()[:24]: sample
           for sample, _ in records(sorted((args.benchmark / "raw").glob("*.tar")))}
    args.out.mkdir(parents=True)
    model = StopModel(args.model, args.adapter, args.max_new_tokens)
    results = []
    started = time.perf_counter()
    with ObservationSession(cfg.render, load_views(cfg.views_path)) as session:
        browser = session.browser_version()
        for trace in traces:
            uid, order_text = trace["id"].split(":order")
            order_id = int(order_text)
            sample = raw[uid]
            reference = ObjectProgram.from_spec(source.get(sample.asset_id))
            candidate = replay_corruptions(reference, sample.corruptions)
            if (reference.sha, candidate.sha) != (
                    sample.reference_sha, sample.program_sha):
                raise ValueError("Source replay mismatch")
            folder = args.out / trace["id"].replace(":", "-")
            (folder / "observations").mkdir(parents=True)
            references, reference_bytes, reference_hash_match = [], [], []
            joint = reference.spec.joints[0]
            session.load(reference, framing=sample.framing)
            for index, state in enumerate(trace["context"]["reference_states"]):
                session.reset_joints()
                session.set_joints({joint.id: detent_value(Detent(state["detent"]), joint.limits)})
                captured = session.request_view(state["view"]).png
                path = folder / f"reference-{index}.png"
                path.write_bytes(captured)
                references.append(str(path.relative_to(args.out)))
                reference_bytes.append(captured)
                reference_hash_match.append(
                    _sha256_bytes(captured) == _sha256_bytes((args.dataset / trace["references"][index]).read_bytes()))
            session.dispose()
            session.load(candidate, framing=sample.framing)
            history, evidence, observation_bytes, calls = [], [], [], []
            model_stop_step = None
            predicted_outcome = None
            for step, frozen in enumerate(trace["history"]):
                action = parse_action(frozen["action"])
                captured: list[bytes] = []
                def sink(index, png):
                    path = folder / "observations" / f"{index:03d}.png"
                    path.write_bytes(png)
                    captured.append(png)
                    return str(path.relative_to(args.out))
                observation = execute_observation(session, action, step, sink)
                payload = {key: value for key, value in observation.payload.items() if key != "render_ms"}
                history.append({"step": step, "action": action_json(action),
                                "image_ref": observation.image_ref, "payload": payload,
                                "reference_index": frozen["reference_index"]})
                observation_bytes.append(captured[0])
                evidence.append(image_evidence(reference_bytes[frozen["reference_index"]], captured[0]))
                row = make_row(uid, order_id, trace["context"], references, history, evidence,
                               trace["data_identity"])
                if row is None:
                    raise ValueError("Selected unambiguous trace became ambiguous on replay")
                images = [Image.open(args.out / path).convert("RGB") for path in references]
                images.extend(Image.open(args.out / item["image_ref"]).convert("RGB") for item in history)
                decision, outcome, raw_text, error = model.predict(row, images)
                frozen_hash_match = _sha256_bytes(captured[0]) == _sha256_bytes(
                    (args.dataset / frozen["image_ref"]).read_bytes())
                calls.append({"step": step + 1, "remaining": BUDGET - step - 1,
                              "decision": decision, "outcome": outcome, "error": error,
                              "raw": raw_text, "evidence": evidence[-1],
                              "frozen_image_sha_match": frozen_hash_match})
                if decision == "stop" or decision is None:
                    model_stop_step = step + 1 if decision == "stop" else None
                    predicted_outcome = outcome
                    break
            session.dispose()
            expected = decide(trace["evidence"], 0)
            expected_step = (min(expected[3]) + 1) if expected[3] else BUDGET
            expected_outcome = "fail" if expected[3] else "uncertain"
            sufficient_at_stop = bool(model_stop_step and any(
                item["state"] == "sufficient" for item in evidence[:model_stop_step]))
            result = {
                "episode_id": trace["id"], "asset_id": trace["data_identity"]["asset_id"],
                "expected_stop_step": expected_step, "expected_outcome": expected_outcome,
                "model_stop_step": model_stop_step, "predicted_outcome": predicted_outcome,
                "stop_timing": ("invalid" if model_stop_step is None else
                                "early" if model_stop_step < expected_step else
                                "late" if model_stop_step > expected_step else "exact"),
                "termination_kind": ("invalid" if model_stop_step is None else
                                     "forced" if model_stop_step == BUDGET else "active"),
                "sufficient_evidence_at_active_stop": (
                    sufficient_at_stop if model_stop_step and model_stop_step < BUDGET else None),
                "reference_sha_match": all(reference_hash_match),
                "observation_sha_match": all(call["frozen_image_sha_match"] for call in calls),
                "calls": calls,
            }
            (folder / "episode.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            results.append(result)
            print(json.dumps({key: value for key, value in result.items() if key != "calls"}), flush=True)
    counts = Counter(row["stop_timing"] for row in results)
    active = [row for row in results if row["termination_kind"] == "active"]
    per_asset = defaultdict(Counter)
    for row in results:
        bucket = per_asset[row["asset_id"]]
        bucket["episodes"] += 1
        bucket[f"timing:{row['stop_timing']}"] += 1
        bucket["supported_active_stop"] += row["sufficient_evidence_at_active_stop"] is True
    report = {
        "version": 1, "status": "complete", "scope": "real_browser_fixed_actions_model_stopping",
        "model": str(args.model), "adapter": str(args.adapter), "dataset": str(args.dataset),
        "selection_rule": "validation traces without ambiguous pixels; one each at first evidence steps 1/2/3, then forced-no-evidence traces preferring new assets",
        "action_policy": "frozen public v5 schedule; model controls only continue/stop",
        "episodes": len(results), "assets": len({row["asset_id"] for row in results}),
        "stop_timing": dict(counts),
        "active_stops": len(active),
        "supported_active_stops": sum(row["sufficient_evidence_at_active_stop"] is True for row in active),
        "active_stop_support_rate": (sum(row["sufficient_evidence_at_active_stop"] is True for row in active)
                                     / len(active) if active else None),
        "exact_outcome": sum(row["predicted_outcome"] == row["expected_outcome"] for row in results),
        "all_reference_sha_match": all(row["reference_sha_match"] for row in results),
        "all_observation_sha_match": all(row["observation_sha_match"] for row in results),
        "per_asset": {asset: dict(value) for asset, value in sorted(per_asset.items())},
        "episode_results": results, "browser_version": browser,
        "elapsed_seconds": time.perf_counter() - started,
        "frozen_test_used": False,
        "limitations": [
            "Actions are a fixed public schedule; autonomous action selection is not evaluated.",
            "Six stratified episodes are a smoke test, not a population estimate.",
            "The evidence gate measures aligned silhouette mismatch, not general articulation diagnosis.",
        ],
    }
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--asset-workspace", type=Path, required=True)
    parser.add_argument("--harness-workspace", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/generate_partnet_mobility_supervision.yaml"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--browser-site-packages", type=Path,
                        help="Append an existing Playwright environment without overriding model packages")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.browser_site_packages:
        sys.path.append(str(args.browser_site_packages.resolve()))
    for name in ("dataset", "benchmark", "asset_workspace", "harness_workspace", "config", "model", "adapter", "out"):
        setattr(args, name, getattr(args, name).resolve())
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
