#!/usr/bin/env python3
"""Run protocol-separated v4/v5 verifier modules in real browser Episodes."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import random
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
from turnitover.oracle.training import tokenize_prompt
from turnitover.oracle.visual_decisions import BUDGET, image_evidence, make_row
from turnitover.policy.loop import execute_observation
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views
from turnitover.verifier.modular import (
    annotation_witness_indices,
    build_single_reference_record,
    finding_signature,
    gate_findings,
    label_signature,
    parse_stop,
    parse_verdict,
    visible_label_steps,
)


CASE_IDS = (
    "3f5511f422151139238be0c9",  # clean, PM_100202
    "c69a463b095f51d73c34e4ce",  # axis-only, PM_102252
    "2d951ffce23f8a1759623d4f",  # offset-only, PM_102590
    "53fca423bddb7c50450f1075",  # offset-only, PM_10797
    "873a775b1dedfd3be942d055",  # offset+axis, PM_1280
    "811b17e40a22e964b694cd74",  # axis-only, PM_34610
    "0f564689a9d115effdbb8f82",  # clean, PM_45759
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _images(items: list[bytes]) -> list[Image.Image]:
    return [Image.open(io.BytesIO(value)).convert("RGB") for value in items]


def _stratum(annotation: dict) -> str:
    labels = {row["defect_id"] for row in annotation["labels"]}
    if not labels:
        return "clean"
    if labels == {"structure.part_offset"}:
        return "offset_only"
    if labels == {"kinematics.joint_axis"}:
        return "axis_only"
    return "offset_and_axis"


def _finding_dict(finding) -> dict:
    return asdict(finding)


class QwenModule:
    def __init__(self, model_path: Path, adapter: Path, stop_max_new_tokens: int,
                 diagnosis_max_new_tokens: int):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            adapter, local_files_only=True, trust_remote_code=False)
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
        self.model = PeftModel.from_pretrained(
            base, adapter, local_files_only=True).eval()
        self.max_new_tokens = {"stop": stop_max_new_tokens, "diagnosis": diagnosis_max_new_tokens}

    def generate(self, record: dict, image_bytes: list[bytes], task: str) -> tuple[str, int, int, float]:
        _, inputs = tokenize_prompt(self.processor, record, _images(image_bytes))
        inputs = inputs.to(self.model.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens[task], do_sample=False)
        generated = output[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return (text, int(inputs["input_ids"].shape[1]), int(generated.shape[1]),
                (time.perf_counter() - started) * 1000)


def _templates(v4_root: Path) -> tuple[dict, dict]:
    diagnoses, stops = {}, {}
    for row in _read_jsonl(v4_root / "validation.episodes.jsonl"):
        if row.get("terminal"):
            diagnoses[row["episode_id"]] = row
    for row in _read_jsonl(v4_root / "validation.stopping.jsonl"):
        stops.setdefault(row["episode_id"], row)
    missing = set(CASE_IDS) - diagnoses.keys() | set(CASE_IDS) - stops.keys()
    if missing:
        raise ValueError(f"Missing v4 protocol templates: {sorted(missing)}")
    return diagnoses, stops


def _trace_plan(v5_root: Path, seed: int) -> list[tuple[str, str, dict]]:
    traces = {row["id"]: row for row in _read_jsonl(v5_root / "trajectories.jsonl")
              if row["split"] == "validation"}
    plan = []
    for uid in CASE_IDS:
        plan.append((uid, "fixed", traces[f"{uid}:order0"]))
        choice = random.Random(f"{seed}:{uid}").choice((1, 2))
        plan.append((uid, "random", traces[f"{uid}:order{choice}"]))
    return plan


def _stop_call(system: str, module: QwenModule, v4_template: dict, trace: dict,
               references: list[bytes], observations: list[bytes], history: list[dict],
               evidence: list[dict]) -> dict:
    remaining = BUDGET - len(history)
    if system == "v4_only":
        record = build_single_reference_record(v4_template, history, remaining)
        roster = [references[0], *observations]
        protocol = "v4_single_reference_stop_decision"
    else:
        record = make_row(trace["id"].split(":")[0], int(trace["id"].rsplit("order", 1)[1]),
                          trace["context"], [f"reference_{i}.png" for i in range(8)],
                          history, evidence, trace["data_identity"])
        if record is None:
            return {"protocol": "v5_eight_reference_prefix_stopping", "decision": None,
                    "outcome": None, "error": "ambiguous_visual_difference_abstention",
                    "raw": None, "input_tokens": 0, "output_tokens": 0, "elapsed_ms": 0.0}
        roster = [*references, *observations]
        protocol = "v5_eight_reference_prefix_stopping"
    text, input_tokens, output_tokens, elapsed = module.generate(record, roster, "stop")
    decision = outcome = error = None
    try:
        decision, outcome = parse_stop(text)
    except ValueError as exc:
        error = str(exc)
    return {"protocol": protocol, "decision": decision, "outcome": outcome,
            "error": error, "raw": text, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "elapsed_ms": elapsed}


def _diagnosis_call(module: QwenModule, template: dict, reference: bytes,
                    observations: list[bytes], history: list[dict]) -> tuple[dict, object | None]:
    record = build_single_reference_record(template, history, BUDGET - len(history))
    text, input_tokens, output_tokens, elapsed = module.generate(
        record, [reference, *observations], "diagnosis")
    payload = json.loads(record["messages"][0]["content"].rsplit("\n", 1)[-1])
    verdict = error = None
    try:
        verdict = parse_verdict(text, payload["context"], history)
    except ValueError as exc:
        error = str(exc)
    call = {"protocol": "v4_single_reference_terminal_diagnosis",
            "status": verdict.status if verdict else None,
            "findings": [_finding_dict(row) for row in verdict.findings] if verdict else [],
            "error": error, "raw": text, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "elapsed_ms": elapsed}
    return call, verdict


def _run_episode(system: str, trace: dict, sample, source, cfg, session,
                 stop_module: QwenModule, diagnosis_module: QwenModule,
                 stop_template: dict, diagnosis_template: dict, folder: Path) -> dict:
    reference = ObjectProgram.from_spec(source.get(sample.asset_id))
    candidate = replay_corruptions(reference, sample.corruptions)
    if (reference.sha, candidate.sha) != (sample.reference_sha, sample.program_sha):
        raise ValueError("Source replay mismatch")
    joint = reference.spec.joints[0]
    references = []
    reference_hash_match = []
    session.load(reference, framing=sample.framing)
    for index, state in enumerate(trace["context"]["reference_states"]):
        session.reset_joints()
        session.set_joints({joint.id: detent_value(Detent(state["detent"]), joint.limits)})
        captured = session.request_view(state["view"]).png
        (folder / f"reference-{index}.png").write_bytes(captured)
        references.append(captured)
        frozen = (trace["_dataset_root"] / trace["references"][index]).read_bytes()
        reference_hash_match.append(_sha(captured) == _sha(frozen))
    v4_frozen_reference = (trace["_v4_root"] / diagnosis_template["images"][0]).read_bytes()
    v4_reference_match = _sha(references[0]) == _sha(v4_frozen_reference)
    session.dispose()
    session.load(candidate, framing=sample.framing)
    observations, history, evidence, calls = [], [], [], []
    final_status, accepted_findings = None, []
    termination_reason = None
    for step, frozen in enumerate(trace["history"]):
        captured = []

        def sink(index, png):
            path = folder / "observations" / f"{index:03d}.png"
            path.write_bytes(png)
            captured.append(png)
            return str(path.relative_to(folder))

        action = parse_action(frozen["action"])
        observation = execute_observation(session, action, step, sink)
        payload = {key: value for key, value in observation.payload.items() if key != "render_ms"}
        history.append({"step": step, "action": action_json(action),
                        "image_ref": f"observation_{step}.png", "payload": payload,
                        "reference_index": frozen["reference_index"]})
        observations.append(captured[0])
        evidence.append(image_evidence(references[frozen["reference_index"]], captured[0]))
        stop_call = _stop_call(system, stop_module, stop_template, trace,
                               references, observations, history, evidence)
        diagnosis_call = None
        rejected = []
        if stop_call["decision"] == "stop" and stop_call["outcome"] == "fail":
            diagnosis_call, verdict = _diagnosis_call(
                diagnosis_module, diagnosis_template, references[0], observations, history)
            accepted_findings, rejected = gate_findings(
                verdict, history, evidence, require_aligned_visual=system != "v4_only")
            if accepted_findings:
                final_status = "fail"
                termination_reason = "supported_diagnosis"
        remaining = BUDGET - len(history)
        if final_status is None and remaining == 0:
            if diagnosis_call is None:
                diagnosis_call, verdict = _diagnosis_call(
                    diagnosis_module, diagnosis_template, references[0], observations, history)
                _, rejected = gate_findings(
                    verdict, history, evidence, require_aligned_visual=system != "v4_only")
            final_status = "uncertain"
            accepted_findings = []
            termination_reason = "budget_exhausted_uncertain"
        calls.append({"step": step + 1, "action": frozen["action"],
                      "remaining": remaining, "visual_evidence": evidence[-1],
                      "stop": stop_call, "diagnosis": diagnosis_call,
                      "rejected_findings": rejected,
                      "frozen_observation_sha_match": _sha(captured[0]) == _sha(
                          (trace["_dataset_root"] / frozen["image_ref"]).read_bytes())})
        if final_status is not None:
            break
    session.dispose()
    return {
        "system": system, "policy": trace["_policy"], "episode_id": trace["id"],
        "source_sample_id": trace["id"].split(":")[0],
        "asset_id": trace["data_identity"]["asset_id"], "order_id": trace["id"].rsplit("order", 1)[1],
        "observation_count": len(history), "final_status": final_status,
        "accepted_findings": [_finding_dict(row) for row in accepted_findings],
        "termination_reason": termination_reason, "history": history,
        "visual_evidence": evidence, "calls": calls,
        "all_reference_sha_match": all(reference_hash_match),
        "v4_single_reference_sha_match": v4_reference_match,
        "all_observation_sha_match": all(row["frozen_observation_sha_match"] for row in calls),
    }


def _score_episode(row: dict, annotation: dict, trace: dict) -> dict:
    history = row["history"]
    states = trace["context"]["reference_states"]
    witnesses = annotation_witness_indices(annotation, history, states)
    visible = visible_label_steps(annotation, history, states)
    expected = {label_signature(annotation["labels"][index])
                for index, steps in visible.items() if steps}
    predicted = {(item["defect_id"], item["joint_id"], tuple(sorted(item["parts"])))
                 for item in row["accepted_findings"]}
    unsupported = 0
    for item in row["accepted_findings"]:
        signature = (item["defect_id"], item["joint_id"], tuple(sorted(item["parts"])))
        index = next((i for i, label in enumerate(annotation["labels"])
                      if label_signature(label) == signature), None)
        if index is None or not set(item["evidence_steps"]) <= set(visible[index]):
            unsupported += 1
    full_visible = visible_label_steps(annotation, trace["history"], states)
    first_steps = [min(steps) + 1 for steps in full_visible.values() if steps]
    expected_stop = min(first_steps) if first_steps else BUDGET
    if row["final_status"] == "fail":
        if not first_steps or row["observation_count"] < expected_stop:
            timing = "early"
        elif row["observation_count"] > expected_stop:
            timing = "late"
        else:
            timing = "exact"
    else:
        timing = "missed" if first_steps else "exact_uncertain"
    scored = dict(row)
    scored.update(
        gold_stratum=_stratum(annotation), expected_visible_findings=[list(item) for item in sorted(expected)],
        expected_first_stop_step=expected_stop, stop_timing=timing,
        unsupported_findings=unsupported,
        v4_unmapped_observations=sum(witness is None for witness in witnesses),
        evidence_sufficient_at_fail=(bool(predicted) and unsupported == 0 and predicted <= expected
                                     if row["final_status"] == "fail" else None),
        tp=len(predicted & expected), fp=len(predicted - expected), fn=len(expected - predicted),
    )
    return scored


def _prf(tp: int, fp: int, fn: int) -> dict:
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def _summarize(rows: list[dict]) -> dict:
    counts = Counter(row["stop_timing"] for row in rows)
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    fail_rows = [row for row in rows if row["final_status"] == "fail"]
    rejected = sum(len(call["rejected_findings"]) for row in rows for call in row["calls"])
    return {
        "episodes": len(rows), "assets": len({row["asset_id"] for row in rows}),
        "finding_micro": _prf(tp, fp, fn),
        "unsupported_findings": sum(row["unsupported_findings"] for row in rows),
        "coordinator_rejected_findings": rejected,
        "v4_unmapped_observations": sum(row["v4_unmapped_observations"] for row in rows),
        "premature_stop": counts["early"],
        "missed_stop": counts["late"] + counts["missed"],
        "stop_timing": dict(counts), "uncertain": sum(row["final_status"] == "uncertain" for row in rows),
        "average_observations": sum(row["observation_count"] for row in rows) / len(rows),
        "fail_stops": len(fail_rows),
        "evidence_sufficient_fail_stops": sum(row["evidence_sufficient_at_fail"] is True for row in fail_rows),
        "evidence_sufficient_stop_rate": (
            sum(row["evidence_sufficient_at_fail"] is True for row in fail_rows) / len(fail_rows)
            if fail_rows else None),
    }


def run(args) -> dict:
    if args.out.exists():
        raise ValueError(f"Output must not exist: {args.out}")
    _benchmark_provenance(args.benchmark)
    cfg = load_generate_config(args.config, repo_root=args.asset_workspace)
    cfg = replace(cfg, render=replace(
        cfg.render, harness_dir=args.harness_workspace / "web/dist",
        esbuild_bin=args.harness_workspace / "web/node_modules/.bin/esbuild"))
    source = make_source(**cfg.asset_source)
    diagnosis_templates, stop_templates = _templates(args.v4_dataset)
    annotations = {row["id"]: row for row in _read_jsonl(args.v4_dataset / "annotations.private.jsonl")
                   if row["split"] == "validation"}
    plan = _trace_plan(args.v5_dataset, args.seed)
    raw = {hashlib.sha256(f"{sample.source_id}:{sample.sample_id}".encode()).hexdigest()[:24]: sample
           for sample, _ in records(sorted((args.benchmark / "raw").glob("*.tar")))}
    args.out.mkdir(parents=True)
    (args.out / "episodes").mkdir()
    stop_module = QwenModule(args.model, args.stop_adapter, args.stop_max_new_tokens,
                             args.diagnosis_max_new_tokens)
    diagnosis_module = (stop_module if args.stop_adapter == args.diagnosis_adapter else
                        QwenModule(args.model, args.diagnosis_adapter, args.stop_max_new_tokens,
                                   args.diagnosis_max_new_tokens))
    rows = []
    started = time.perf_counter()
    with ObservationSession(cfg.render, load_views(cfg.views_path)) as session:
        browser = session.browser_version()
        for uid, policy, original_trace in plan:
            trace = dict(original_trace, _dataset_root=args.v5_dataset, _v4_root=args.v4_dataset,
                         _policy=policy)
            folder = args.out / "episodes" / f"{uid}-{policy}"
            (folder / "observations").mkdir(parents=True)
            actual = _run_episode(
                args.system, trace, raw[uid], source, cfg, session, stop_module, diagnosis_module,
                stop_templates[uid], diagnosis_templates[uid], folder)
            scored = _score_episode(actual, annotations[uid], trace)
            (folder / "episode.json").write_text(json.dumps(scored, indent=2), encoding="utf-8")
            rows.append(scored)
            print(json.dumps({key: value for key, value in scored.items()
                              if key not in ("calls", "history", "visual_evidence")}), flush=True)
    with (args.out / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    report = {
        "version": 1, "status": "complete", "scope": "modular_real_browser_verifier",
        "system": args.system, "model": str(args.model),
        "stop_adapter": str(args.stop_adapter), "diagnosis_adapter": str(args.diagnosis_adapter),
        "protocols": {"stopping": ("v4_single_reference_stop_decision" if args.system == "v4_only"
                                     else "v5_eight_reference_prefix_stopping"),
                      "diagnosis": "v4_single_reference_terminal_diagnosis"},
        "coordination": {
            "fail": "stop/fail suggestion plus supported diagnosis; modular systems require cited aligned v5 mismatch",
            "axis": "requires a cited aligned same-joint actuation; static mismatch alone is rejected",
            "continue": "continue while budget and no supported diagnosis",
            "forced": "budget exhaustion without accepted fail becomes uncertain; pass is never emitted",
        },
        "selection": {"case_ids": list(CASE_IDS), "assets": 7,
                      "strata": dict(Counter(_stratum(annotations[uid]) for uid in CASE_IDS)),
                      "gold_use": "predeclared stratification and post-hoc scoring only; never model/coordinator input"},
        "policies": {"fixed": "v5 order0", "random": "seeded choice from v5 order1/order2",
                     "seed": args.seed, "active": False},
        "overall": _summarize(rows),
        "per_policy": {policy: _summarize([row for row in rows if row["policy"] == policy])
                       for policy in ("fixed", "random")},
        "per_asset": {asset: _summarize([row for row in rows if row["asset_id"] == asset])
                      for asset in sorted({row["asset_id"] for row in rows})},
        "per_stratum": {stratum: _summarize([row for row in rows if row["gold_stratum"] == stratum])
                        for stratum in sorted({row["gold_stratum"] for row in rows})},
        "all_reference_sha_match": all(row["all_reference_sha_match"] for row in rows),
        "all_v4_single_reference_sha_match": all(row["v4_single_reference_sha_match"] for row in rows),
        "all_observation_sha_match": all(row["all_observation_sha_match"] for row in rows),
        "browser_version": browser, "elapsed_seconds": time.perf_counter() - started,
        "frozen_test_used": False,
        "limitations": [
            "Fixed/random are predetermined schedules, not learned or active action policies.",
            "Seven held-out objects and two schedules are a systems smoke test, not population evidence.",
            "v5 aligned-shape evidence does not identify defect type; v4 supplies diagnoses and the coordinator only gates claims.",
            "Private labels/exposure grids are used only after each Episode for metrics.",
        ],
    }
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("v4_only", "modular", "multitask"), required=True)
    parser.add_argument("--v4-dataset", type=Path, required=True)
    parser.add_argument("--v5-dataset", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--asset-workspace", type=Path, required=True)
    parser.add_argument("--harness-workspace", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/generate_partnet_mobility_supervision.yaml"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stop-adapter", type=Path, required=True)
    parser.add_argument("--diagnosis-adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--stop-max-new-tokens", type=int, default=64)
    parser.add_argument("--diagnosis-max-new-tokens", type=int, default=512)
    parser.add_argument("--browser-site-packages", type=Path)
    args = parser.parse_args()
    if args.browser_site_packages:
        sys.path.append(str(args.browser_site_packages.resolve()))
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
