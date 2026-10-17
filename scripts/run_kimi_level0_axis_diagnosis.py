#!/usr/bin/env python3
"""One-call-per-image Kimi diagnosis of the reviewed Level 0 axis bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from turnitover.models.client import ModelError, request_payload
from turnitover.models.config import model_config, read_environment
from turnitover.teacher.providers import KimiTeacher


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/evidence/partnet-mobility-level0-axis-v1"
DEFAULT_OUT = ROOT / "docs/evidence/kimi-level0-axis-diagnosis-v1"
PROMPT = """The image is a 2x2 motion comparison. Top row: reference rest, reference limit. Bottom row: candidate rest, candidate limit. Decide whether the candidate motion matches the reference. Reply in exactly two lines:
verdict: pass or fail
motion: one sentence comparing the normal and candidate motion directions."""
PIXEL_THRESHOLD = 0.005
TILES = {"reference_rest": (10, 10), "reference_limit": (276, 10),
         "candidate_rest": (10, 276), "candidate_limit": (276, 276)}
TILE_SIZE = 256


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> dict[str, dict]:
    return {row["case_id"]: row for row in map(json.loads, path.read_text().splitlines())}


def preflight() -> tuple[list[dict], dict]:
    """Check reviewed labels and input bytes, without trusting public pending status."""
    human = _rows(SOURCE / "human_review.jsonl")
    gold = _rows(SOURCE / "private/gold.jsonl")
    checks = _rows(SOURCE / "private/checks.jsonl")
    if set(human) != set(gold) or set(gold) != set(checks) or len(gold) != 10:
        raise ValueError("Expected the same ten cases in human review, Gold, and checks")
    cases = []
    for case_id in sorted(gold):
        reviewed, truth, check = human[case_id], gold[case_id], checks[case_id]
        if reviewed["status"] != "human_review_complete":
            raise ValueError(f"Human review incomplete: {case_id}")
        judgment = reviewed["human_judgment"]
        if judgment["verdict"] != truth["expected_status"]:
            raise ValueError(f"Human/Gold verdict mismatch: {case_id}")
        if truth["kind"] == "joint_axis" and judgment["defect_id"] != truth["defect_id"]:
            raise ValueError(f"Human/Gold defect mismatch: {case_id}")
        if truth["kind"] == "clean" and judgment["defect_id"] is not None:
            raise ValueError(f"Clean review has a defect label: {case_id}")
        if bool(check["code_correct"]) != (truth["expected_status"] == "pass"):
            raise ValueError(f"Independent checker/Gold mismatch: {case_id}")
        image = SOURCE / "inputs" / f"{case_id}.png"
        with Image.open(image) as opened:
            if opened.size != (542, 542) or opened.format != "PNG":
                raise ValueError(f"Unexpected input image: {image}")
        cases.append({"case_id": case_id, "image": image, "image_sha256": _sha(image),
                      "kind": truth["kind"], "expected_status": truth["expected_status"]})
    return cases, {"human_review_sha256": _sha(SOURCE / "human_review.jsonl"),
                   "gold_sha256": _sha(SOURCE / "private/gold.jsonl"),
                   "checks_sha256": _sha(SOURCE / "private/checks.jsonl"),
                   "input_sha256": {case["case_id"]: case["image_sha256"] for case in cases}}


def pixel_baseline(image: Path) -> dict:
    pixels = np.asarray(Image.open(image).convert("RGB"), dtype=np.float32) / 255.0
    crops = {name: pixels[y:y + TILE_SIZE, x:x + TILE_SIZE] for name, (x, y) in TILES.items()}
    rest = float(np.abs(crops["reference_rest"] - crops["candidate_rest"]).mean())
    limit = float(np.abs(crops["reference_limit"] - crops["candidate_limit"]).mean())
    score = max(rest, limit)
    return {"reference_candidate_rest_mae": rest, "reference_candidate_limit_mae": limit,
            "max_mae": score, "threshold": PIXEL_THRESHOLD,
            "predicted_status": "fail" if score > PIXEL_THRESHOLD else "pass"}


def parse(text: str) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    verdict = next((line.split(":", 1)[1].strip().lower() for line in lines
                    if line.lower().startswith("verdict:") and ":" in line), None)
    motion = next((line.split(":", 1)[1].strip() for line in lines
                   if line.lower().startswith("motion:") and ":" in line), None)
    return {"verdict": verdict if verdict in {"pass", "fail"} else None,
            "motion": motion, "format_valid": len(lines) == 2 and verdict in {"pass", "fail"}
            and bool(motion)}


def run(args: argparse.Namespace) -> dict:
    if args.out.exists():
        raise ValueError(f"Output must be new: {args.out}")
    cases, preflight_record = preflight()
    env = read_environment(args.env_file)
    config = model_config("teacher", env)
    if config.model != "k3-256k":
        raise ValueError("This experiment is pinned to k3-256k")
    if not args.third_party_transmission_reviewed or not args.transmission_review_basis:
        raise ValueError("Transmission review and its basis are required")
    teacher = KimiTeacher(config, None, transmission_reviewed=True,
                          estimated_input_tokens=args.estimated_input_tokens,
                          max_real_requests=len(cases))
    args.out.mkdir(parents=True)
    results = []
    for order, case in enumerate(cases):
        case_dir = args.out / "cases" / case["case_id"]
        case_dir.mkdir(parents=True)
        (case_dir / "prompt.txt").write_text(PROMPT, encoding="utf-8")
        request = {"order": order, "model": config.public(), "prompt": PROMPT,
                   "image": "input.png", "image_sha256": case["image_sha256"],
                   "images_per_call": 1, "gold_or_review_sent": False,
                   "asset_id_sent": False, "corruption_parameters_sent": False}
        (case_dir / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
        (case_dir / "input.png").write_bytes(case["image"].read_bytes())
        _, _, payload = request_payload(config, PROMPT, [case_dir / "input.png"])
        # This is the exact API body, retained without endpoint or Authorization header.
        (case_dir / "api_request.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        result = {"case_id": case["case_id"], "kind": case["kind"],
                  "image_sha256": case["image_sha256"], "pixel_baseline": pixel_baseline(case["image"])}
        try:
            response = teacher(PROMPT, [case_dir / "input.png"])
            (case_dir / "response.txt").write_text(response.text, encoding="utf-8")
            result["model_call"] = {"status": "completed", "model": response.model,
                                    "usage": response.usage, "finish_reason": response.finish_reason,
                                    "elapsed_ms": response.elapsed_ms}
            result["prediction"] = parse(response.text)
        except ModelError as exc:
            # A failed request is retained as a failed case; it is never retried.
            (case_dir / "response_error.txt").write_text(str(exc), encoding="utf-8")
            result["model_call"] = {"status": "error", "error": str(exc)}
            result["prediction"] = {"verdict": None, "motion": None, "format_valid": False}
        (case_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
    # Gold is intentionally joined only after all ten requests have returned or failed.
    gold = _rows(SOURCE / "private/gold.jsonl")
    human = _rows(SOURCE / "human_review.jsonl")
    for result in results:
        truth, review = gold[result["case_id"]], human[result["case_id"]]["human_judgment"]
        result["posthoc"] = {"gold_status": truth["expected_status"],
                              "human_status": review["verdict"],
                              "pass_fail_correct": result["prediction"]["verdict"] == truth["expected_status"],
                              "baseline_correct": result["pixel_baseline"]["predicted_status"] == truth["expected_status"]}
    groups = {kind: [row for row in results if row["kind"] == kind] for kind in ("clean", "joint_axis")}
    summary = {"version": 1, "status": "complete", "model": config.public(),
               "source_commit": "f34b07c127266b401564592bfe38bb6c9e7121fb",
               "preflight": preflight_record,
               "protocol": {"cases": 10, "independent_assets": 5, "calls_per_case": 1,
                            "real_api_requests": teacher.calls, "retries": 0, "few_shot": False,
                            "images_per_call": 1, "prompt_identical": True,
                            "posthoc_gold_join_only": True, "sft_training": False,
                            "test_modified": False, "transmission_review_basis": args.transmission_review_basis},
               "counts": {kind: {"total": len(rows),
                                  "pass_fail_correct": sum(row["posthoc"]["pass_fail_correct"] for row in rows),
                                  "incorrect": sum(not row["posthoc"]["pass_fail_correct"] for row in rows),
                                  "format_invalid": sum(not row["prediction"]["format_valid"] for row in rows)}
                          for kind, rows in groups.items()},
               "pixel_baseline": {"rule": "fail iff max paired-tile MAE > 0.005", "threshold": PIXEL_THRESHOLD,
                                   "correct": sum(row["posthoc"]["baseline_correct"] for row in results),
                                   "total": len(results),
                                   "motion_understanding": "not measured; this baseline only detects pixel mismatch"},
               "cases": results,
               "boundaries": {"training_labels_created": False, "api_output_used_as_label": False,
                              "test_accessed": False, "old_pending_manifest_used_for_selection": False}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--estimated-input-tokens", type=int, required=True)
    parser.add_argument("--third-party-transmission-reviewed", action="store_true")
    parser.add_argument("--transmission-review-basis")
    args = parser.parse_args()
    print(json.dumps(run(args)["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
