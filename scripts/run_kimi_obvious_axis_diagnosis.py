#!/usr/bin/env python3
"""Run one fixed-evidence Kimi diagnosis call on each approved axis/clean case."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from turnitover.assets.catalog import CatalogSource
from turnitover.config import load_generate_config
from turnitover.core.actions import RuntimeProperty, detent_value
from turnitover.core.jsonio import write_json_atomic
from turnitover.core.program import ObjectProgram
from turnitover.models.config import model_config, read_environment
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views
from turnitover.teacher.audit import audit_episode
from turnitover.teacher.episode import run_teacher_episode
from turnitover.teacher.policy import TeacherPolicy
from turnitover.teacher.providers import KimiTeacher
from turnitover.verifier.contracts import Context


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/evidence/partnet-mobility-obvious-defects-v1"
REVIEW = ROOT / "docs/evidence/partnet-mobility-obvious-defects-v1-human-review"
DEFAULT_OUT = ROOT / "docs/evidence/kimi-obvious-axis-diagnosis-v1"
DEFAULT_CONFIG = ROOT / "configs/generate_partnet_mobility_supervision.yaml"
DEFAULT_SPLITS = ROOT / "data/assets/partnet-mobility-trainval-v1/splits.json"
STATES = ("rest", "mid", "limit")
BOARD_X = (122, 388, 654)
BOARD_Y = {"reference": 116, "candidate": 382}
TILE = 256
INITIAL_ACTIONS = (
    {"type": "query_runtime", "property": "hierarchy"},
    {"type": "actuate_joint", "joint_id": "{joint_id}", "detent": "rest"},
    {"type": "actuate_joint", "joint_id": "{joint_id}", "detent": "mid"},
    {"type": "actuate_joint", "joint_id": "{joint_id}", "detent": "limit"},
)


def _rows(path: Path) -> dict[str, dict]:
    return {row["case_id"]: row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _approved_cases() -> list[dict]:
    public = _rows(SOURCE / "review_manifest.jsonl")
    gold = _rows(SOURCE / "private/gold.jsonl")
    human = _rows(REVIEW / "human_review.jsonl")
    adjudication = _rows(REVIEW / "private/adjudication.jsonl")
    by_asset: dict[str, dict[str, dict]] = {}
    for case_id, private in gold.items():
        if private["kind"] not in {"clean", "joint_axis"}:
            continue
        decision = adjudication[case_id]
        if not decision["checker_consistent"]:
            raise ValueError(f"Independent checker is inconsistent: {case_id}")
        review = human[case_id]["human_judgment"]
        expected = {
            "status": private["expected_status"],
            "defect_id": private["defect_id"],
            "joint_id": private["joint_id"],
            "parts": private["parts"],
        }
        observed = {
            "status": review["verdict"],
            "defect_id": review["defect_id"],
            "joint_id": review["joint_id"],
            "parts": review["parts"],
        }
        human_gold_agree = expected == observed
        if private["kind"] == "joint_axis" and (
                decision["candidate_status"] != "review_approved_candidate" or not human_gold_agree):
            raise ValueError(f"Axis case did not pass blind review against Gold: {case_id}")
        if private["kind"] == "clean" and review["verdict"] not in {"pass", "uncertain"}:
            raise ValueError(f"Clean control was judged as a defect: {case_id}")
        by_asset.setdefault(public[case_id]["asset_id"], {})[private["kind"]] = {
            "case_id": case_id, "public": public[case_id], "gold": private, "human": review,
            "human_gold_agree": human_gold_agree,
            "human_review_gate": decision["review_gate"],
        }
    if len(by_asset) != 3 or any(set(pair) != {"clean", "joint_axis"} for pair in by_asset.values()):
        raise ValueError("Expected three approved axis/clean pairs")
    return [by_asset[asset][kind] for asset in sorted(by_asset) for kind in ("clean", "joint_axis")]


def _crop_board(case: dict, case_dir: Path) -> tuple[list[Path], list[Path], list[dict]]:
    board = SOURCE / case["public"]["board"]
    if _sha(board) != case["public"]["board_sha256"]:
        raise ValueError(f"Source board SHA changed: {case['case_id']}")
    references, candidates, records = [], [], []
    with Image.open(board) as image:
        if image.size[0] < BOARD_X[-1] + TILE or image.size[1] < BOARD_Y["candidate"] + TILE:
            raise ValueError(f"Unexpected board dimensions: {board}")
        for row_name, paths in (("reference", references), ("candidate", candidates)):
            for index, state in enumerate(STATES):
                target = case_dir / f"{row_name}-{state}.png"
                box = (BOARD_X[index], BOARD_Y[row_name], BOARD_X[index] + TILE,
                       BOARD_Y[row_name] + TILE)
                image.crop(box).save(target)
                paths.append(target)
                records.append({"kind": row_name, "state": state, "path": target.name,
                                "sha256": _sha(target), "size": [TILE, TILE], "source_crop": list(box)})
    return references, candidates, records


class FrozenBoardSession:
    """Replay the already-rendered candidate tiles as a fixed Episode trajectory."""

    def __init__(self, joint_id: str, view: str, limits: tuple[float, float], hierarchy: dict,
                 images: list[Path]):
        self.joint_id, self.view, self.limits = joint_id, view, limits
        self.hierarchy, self.images = hierarchy, dict(zip(STATES, images, strict=True))

    def request_view(self, view):  # pragma: no cover - the fixed protocol never requests a view
        raise ValueError(f"Unexpected request_view({view}) in fixed protocol")

    def actuate(self, joint_id, detent):
        if joint_id != self.joint_id or detent.value not in STATES:
            raise ValueError("Fixed protocol requested an unexpected joint or detent")
        return SimpleNamespace(png=self.images[detent.value].read_bytes(), render_ms=0.0,
                               value=detent_value(detent, self.limits))

    def query_runtime(self, prop):
        if prop != RuntimeProperty.HIERARCHY:
            raise ValueError("Fixed protocol exposes only the observed hierarchy")
        return self.hierarchy


def _hierarchies(cases: list[dict], config_path: Path) -> tuple[dict, dict]:
    cfg = load_generate_config(config_path)
    catalog = CatalogSource(cfg.asset_source["manifest"])
    views = load_views(cfg.views_path)
    result, specs = {}, {}
    with ObservationSession(cfg.render, views) as session:
        for asset_id in sorted({case["public"]["asset_id"] for case in cases}):
            spec = catalog.get(asset_id)
            session.load(ObjectProgram.from_spec(spec), framing=None)
            result[asset_id] = session.query_runtime(RuntimeProperty.HIERARCHY)
            specs[asset_id] = spec
            session.dispose()
    return result, specs


def _expected_steps(case: dict) -> list[int]:
    return [STATES.index(state) + 1 for state in case["human"]["evidence_states"]]


def _score(case: dict, episode: dict) -> dict:
    expected = case["gold"]
    expected_steps = _expected_steps(case)
    format_valid = not episode.get("teacher_verdict_missing") and bool(episode.get("verdict"))
    verdict = episode.get("verdict") or {}
    findings = verdict.get("findings") or []
    predicted_defects = sorted({row.get("defect_id") for row in findings})
    predicted_joints = sorted({row.get("joint_id") for row in findings if row.get("joint_id") is not None})
    predicted_parts = sorted({part for row in findings for part in row.get("parts", [])})
    predicted_steps = sorted({step for row in findings for step in row.get("evidence_steps", [])})
    if expected["kind"] == "clean":
        fields = {
            "verdict_exact_gold": format_valid and verdict.get("status") == "pass",
            "clean_safe": format_valid and verdict.get("status") in {"pass", "uncertain"},
            "defect_id": format_valid and not predicted_defects,
            "joint_id": format_valid and not predicted_joints,
            "parts": format_valid and not predicted_parts,
            "evidence_steps": format_valid and not predicted_steps,
        }
        category = ("format_error" if not format_valid else
                    "clean_false_positive" if verdict.get("status") == "fail" else
                    "clean_pass" if verdict.get("status") == "pass" else "clean_uncertain")
        strict = None
    else:
        fields = {
            "verdict": format_valid and verdict.get("status") == "fail",
            "defect_id": format_valid and predicted_defects == [expected["defect_id"]],
            "joint_id": format_valid and predicted_joints == [expected["joint_id"]],
            "parts": format_valid and predicted_parts == sorted(expected["parts"]),
            "evidence_steps": format_valid and predicted_steps == expected_steps,
        }
        strict = all(fields.values())
        category = ("format_error" if not format_valid else
                    "no_evidence_judgment" if verdict.get("status") == "uncertain" else
                    "defect_recognition_error" if not fields["verdict"] or not fields["defect_id"] else
                    "localization_error" if not fields["joint_id"] or not fields["parts"] else
                    "evidence_error" if not fields["evidence_steps"] else "strict_correct")
    return {
        "case_id": case["case_id"], "asset_id": case["public"]["asset_id"],
        "kind": expected["kind"], "format_valid": format_valid, "error_category": category,
        "strict_axis_correct": strict, "field_correctness": fields,
        "expected": {"status": expected["expected_status"], "defect_id": expected["defect_id"],
                     "joint_id": expected["joint_id"], "parts": expected["parts"],
                     "evidence_steps": expected_steps},
        "predicted": {"status": verdict.get("status"), "defect_ids": predicted_defects,
                      "joint_ids": predicted_joints, "parts": predicted_parts,
                      "evidence_steps": predicted_steps},
        "human_gold_agree": case["human_gold_agree"],
    }


def _write_reproduce(out: Path) -> None:
    (out / "REPRODUCE.md").write_text("""# Kimi obvious-axis fixed-diagnosis experiment

This is a six-call, no-retry Train-only experiment: three human-approved `joint_axis`
cases and their approved clean controls. The exact prompts, visible requests, 256 px
images, raw response text, Episode ledgers, image hashes, private score, and aggregate
summary are retained here. The model received no Gold, corruption parameters, or human
answers. Billing is recorded as unknown (`null`) because the gateway price is unknown.

Preflight and run from the repository root into a fresh directory:

```bash
python -m pytest tests/unit/test_kimi_obvious_axis_diagnosis.py -q
python -m scripts.run_kimi_obvious_axis_diagnosis \
  --out docs/evidence/kimi-obvious-axis-diagnosis-v1 \
  --third-party-transmission-reviewed \
  --transmission-review-basis user-requested-six-case-PartNet-Train-diagnostic \
  --estimated-input-tokens 12000
```

The command refuses an existing output directory. It makes exactly one real `k3-256k`
call per case, has no schema retry, and has a global six-request cap. Do not rerun it to
replace failed or unfavorable outputs.

Validation:

```bash
python -m scripts.run_kimi_obvious_axis_diagnosis \
  --audit-existing --out docs/evidence/kimi-obvious-axis-diagnosis-v1
python -m pytest tests/unit/test_kimi_obvious_axis_diagnosis.py tests/unit/test_teacher.py -q
python -m pytest tests/unit -q
python -m py_compile scripts/run_kimi_obvious_axis_diagnosis.py
git diff --check
```
""", encoding="utf-8")


def audit_existing(out: Path) -> dict:
    """Re-audit saved Episodes and hashes without making any model request."""
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in _approved_cases()}
    if set(cases) != {row["case_id"] for row in summary["cases"]}:
        raise ValueError("Saved result case IDs differ from the frozen six-case protocol")
    markers = ("human_judgment", "expected_status", "candidate_sha", "reference_sha",
               "angle_deg", "original_axis", "seed")
    audit_rows = []
    for saved in summary["cases"]:
        case_id = saved["case_id"]
        case_dir = out / "cases" / case_id
        episode = json.loads((case_dir / "episode.json").read_text(encoding="utf-8"))
        semantic = _score(cases[case_id], episode)
        if semantic != {key: saved[key] for key in semantic}:
            raise ValueError(f"Saved semantic score changed: {case_id}")
        call_dirs = sorted((case_dir / "model_calls").iterdir())
        if len(call_dirs) != 1:
            raise ValueError(f"Expected exactly one saved call: {case_id}")
        call_dir = call_dirs[0]
        prompt = (call_dir / "prompt.txt").read_text(encoding="utf-8")
        request = json.loads((call_dir / "request.json").read_text(encoding="utf-8"))
        image_paths = json.loads((call_dir / "images.json").read_text(encoding="utf-8"))
        actual_shas = [_sha(case_dir / relative) for relative in image_paths]
        expected_images = episode["input_images"]
        expected_shas = ([row["sha256"] for row in expected_images if row["kind"] == "reference"]
                         + [row["sha256"] for row in expected_images if row["kind"] == "candidate"])
        # The legacy auditor infers reference order by sorting ``reference-*`` filenames.
        # Preserve the raw request and use a temporary numeric-name mirror for that adapter.
        with tempfile.TemporaryDirectory(prefix="tio-kimi-axis-audit-") as temp:
            mirror = Path(temp) / case_id
            shutil.copytree(case_dir, mirror)
            mirror_images_path = mirror / "model_calls/000/images.json"
            mirror_images = json.loads(mirror_images_path.read_text(encoding="utf-8"))
            for index, relative in enumerate(mirror_images[:3]):
                source = mirror / relative
                target = mirror / f"reference-{index:02d}.png"
                source.rename(target)
                mirror_images[index] = target.name
            mirror_images_path.write_text(json.dumps(mirror_images), encoding="utf-8")
            pipeline = audit_episode(mirror)
        pipeline["input_adapter"] = (
            "Temporary numeric reference filenames were used because the legacy auditor sorts "
            "reference-* names; raw request files and image bytes were not changed."
        )
        write_json_atomic(case_dir / "audit.json", pipeline)
        row = {
            "case_id": case_id, "pipeline_ok": pipeline["pipeline_ok"],
            "one_call": len(call_dirs) == 1 and episode["real_api_requests"] == 1,
            "response_retained": (call_dir / "response.txt").is_file(),
            "six_image_shas_match": actual_shas == expected_shas and len(actual_shas) == 6,
            "fixed_terminal_request": request["remaining"] == 0 and request["stop_required"] is True,
            "history_steps": [row["step"] for row in request["history"]],
            "private_markers_in_prompt": [marker for marker in markers if marker in prompt],
            "source_board_sha_matches": _sha(SOURCE / cases[case_id]["public"]["board"])
            == cases[case_id]["public"]["board_sha256"],
        }
        row["passed"] = (row["pipeline_ok"] and row["one_call"] and row["response_retained"]
                         and row["six_image_shas_match"] and row["fixed_terminal_request"]
                         and row["history_steps"] == [0, 1, 2, 3]
                         and not row["private_markers_in_prompt"] and row["source_board_sha_matches"])
        audit_rows.append(row)
    summary["independent_audit"] = {
        "status": "passed" if all(row["passed"] for row in audit_rows) else "failed",
        "cases": audit_rows,
    }
    axis = [row for row in summary["cases"] if row["kind"] == "joint_axis"]
    clean = [row for row in summary["cases"] if row["kind"] == "clean"]
    summary["field_accuracy"] = {
        "joint_axis": {"total": len(axis), **{
            key: sum(row["field_correctness"][key] for row in axis)
            for key in ("verdict", "defect_id", "joint_id", "parts", "evidence_steps")}},
        "clean": {"total": len(clean), **{
            key: sum(row["field_correctness"][key] for row in clean)
            for key in ("verdict_exact_gold", "clean_safe", "defect_id", "joint_id", "parts",
                        "evidence_steps")}},
    }
    categories = sorted({row["error_category"] for row in summary["cases"]})
    summary["error_categories"] = {
        category: sum(row["error_category"] == category for row in summary["cases"])
        for category in categories
    }
    summary["token_usage"] = {
        key: sum((row["model_call"].get("usage") or {}).get(key, 0) for row in summary["cases"])
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    write_json_atomic(summary_path, summary)
    write_json_atomic(out / "independent_audit.json", summary["independent_audit"])
    return summary


def run(args) -> dict:
    if args.out.exists():
        raise ValueError(f"Output must not exist: {args.out}")
    cases = _approved_cases()
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    for case in cases:
        group = f"PartNet-Mobility:{case['public']['asset_id'].removeprefix('PM_')}"
        if splits["groups"].get(group) != "train":
            raise ValueError(f"Non-Train asset selected: {case['public']['asset_id']}")
    hierarchies, specs = _hierarchies(cases, args.config)
    env = read_environment(args.env_file)
    model = model_config("teacher", env)
    if model.model != "k3-256k":
        raise ValueError("This experiment is pinned to k3-256k")
    if not args.third_party_transmission_reviewed or not args.transmission_review_basis:
        raise ValueError("Transmission review and its basis are required")
    teacher = KimiTeacher(model, None, transmission_reviewed=True,
                          estimated_input_tokens=args.estimated_input_tokens,
                          max_real_requests=len(cases))
    args.out.mkdir(parents=True)
    _write_reproduce(args.out)
    case_results = []
    for order, case in enumerate(cases):
        case_dir = args.out / "cases" / case["case_id"]
        case_dir.mkdir(parents=True)
        references, candidates, images = _crop_board(case, case_dir)
        public, expected = case["public"], case["gold"]
        spec = specs[public["asset_id"]]
        joint = spec.joint(public["joint_id"])
        context = Context(
            task=("Compare candidate rest, mid, and limit motion with the same-view reference states. "
                  "Return one final structured diagnosis for the visible evidence."),
            parts=spec.part_ids, joints=spec.joint_ids, views=(public["view"],),
            runtime_properties=("hierarchy",),
            action_types=("request_view", "actuate_joint", "query_runtime"),
            max_triangles=1_000_000, max_draw_calls=1_024,
        )
        shutil.copyfile(ROOT / "turnitover/teacher/joint_axis_rules.txt", case_dir / "protocol_addendum.txt")
        policy = TeacherPolicy(
            teacher, case_dir, references,
            reference_states=[{"view": public["view"], "joint_id": public["joint_id"], "detent": state}
                              for state in STATES],
            protocol_addenda=("joint_axis_rules.txt",),
        )
        actions = tuple({key: (public["joint_id"] if value == "{joint_id}" else value)
                         for key, value in action.items()} for action in INITIAL_ACTIONS)
        identity = {
            "experiment": "kimi-obvious-axis-diagnosis-v1", "order": order,
            "case_id": case["case_id"], "asset_id": public["asset_id"], "split": "train",
            "source_board": str((SOURCE / public["board"]).relative_to(ROOT)),
            "source_board_sha256": public["board_sha256"],
            "candidate_sha": expected["candidate_sha"], "reference_sha": expected["reference_sha"],
            "human_review_gate": case["human_review_gate"],
            "human_gold_agreement": case["human_gold_agree"],
            "independent_checker_consistent": True,
            "fixed_evidence": True, "active_acquisition": False,
            "third_party_transmission_reviewed": True,
            "transmission_review_basis": args.transmission_review_basis,
        }
        session = FrozenBoardSession(public["joint_id"], public["view"], joint.limits,
                                     hierarchies[public["asset_id"]], candidates)
        episode = run_teacher_episode(
            session, policy, context, len(actions), case_dir,
            episode_id=case["case_id"], max_model_calls=1, decision_attempts=1,
            data_identity=identity, initial_actions=actions,
        )
        episode["real_api_requests"] = len(policy.calls)
        episode["usage_cost_rmb_estimate"] = None
        episode["billing_cost_rmb"] = None
        episode["input_images"] = images
        write_json_atomic(case_dir / "episode.json", episode)
        score = _score(case, episode)
        write_json_atomic(case_dir / "private_score.json", score)
        call_dir = case_dir / "model_calls/000"
        case_results.append({
            **score,
            "model_call": {"status": policy.calls[0]["status"], "usage": policy.calls[0].get("usage"),
                           "finish_reason": policy.calls[0].get("finish_reason"),
                           "elapsed_ms": policy.calls[0].get("elapsed_ms")},
            "artifacts": {"episode": str((case_dir / "episode.json").relative_to(args.out)),
                          "prompt_sha256": _sha(call_dir / "prompt.txt"),
                          "request_sha256": _sha(call_dir / "request.json"),
                          "response_sha256": _sha(call_dir / "response.txt") if (call_dir / "response.txt").exists()
                          else None,
                          "images": images},
        })
    axis = [row for row in case_results if row["kind"] == "joint_axis"]
    clean = [row for row in case_results if row["kind"] == "clean"]
    summary = {
        "version": 1, "status": "complete", "model": model.public(),
        "protocol": {"fixed_sufficient_evidence": True, "active_acquisition": False,
                     "calls_per_case": 1, "schema_retries": 0, "total_request_cap": len(cases),
                     "states": list(STATES), "candidate_evidence_steps": {"rest": 1, "mid": 2, "limit": 3},
                     "student_image_size": [TILE, TILE]},
        "counts": {"cases": len(cases), "real_api_requests": teacher.calls,
                   "format_valid": sum(row["format_valid"] for row in case_results),
                   "axis_strict_correct": sum(row["strict_axis_correct"] is True for row in axis),
                   "axis_total": len(axis),
                   "clean_false_positives": sum(row["error_category"] == "clean_false_positive" for row in clean),
                   "clean_total": len(clean)},
        "human_clear_but_kimi_not_strict": [row["case_id"] for row in axis
                                             if row["strict_axis_correct"] is not True],
        "cases": case_results,
        "boundaries": {"expanded_collection": False, "sft_exported": False, "sft_trained": False,
                       "frozen_test_read_or_modified": False, "images_rerendered": False,
                       "chromium_use": "hierarchy_query_only", "billing_cost_rmb": None},
    }
    write_json_atomic(args.out / "summary.json", summary)
    return audit_existing(args.out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--estimated-input-tokens", type=int)
    parser.add_argument("--third-party-transmission-reviewed", action="store_true")
    parser.add_argument("--transmission-review-basis")
    parser.add_argument("--audit-existing", action="store_true",
                        help="Re-audit an existing bundle without making an API request")
    args = parser.parse_args()
    if args.audit_existing:
        summary = audit_existing(args.out)
    else:
        if args.estimated_input_tokens is None:
            parser.error("--estimated-input-tokens is required for a real Kimi run")
        summary = run(args)
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
