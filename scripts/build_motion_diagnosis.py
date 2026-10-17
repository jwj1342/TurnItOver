"""Build audited same-object axis/offset/clean/uncertain matched-state supervision."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from turnitover.assets.source import make_source
from turnitover.config import load_generate_config
from turnitover.core.actions import RuntimeProperty
from turnitover.core.program import ObjectProgram
from turnitover.dataset.prepare import records
from turnitover.oracle.evidence import pixels, replay_corruptions
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views
from turnitover.teacher.witnesses import core_changed_fraction


INSTRUCTION = """Compare REFERENCE and CANDIDATE image pairs at exactly matched camera and actual joint values.
The public runtime hierarchy is shared across all variants and identifies legal joint/part IDs only.
Diagnose the tested motion: axis = wrong motion direction/plane; offset = displaced part; clean =
all shown states match; uncertain = the shown states cannot establish the motion diagnosis. A rest pose
alone cannot prove axis or clean. Return only JSON with defect_type, joint, part, evidence_steps.
evidence_steps are zero-based pair indices that visibly support the answer. Use null IDs for clean/uncertain,
and a null joint for offset. Public input: """


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target(kind, joint, part, steps):
    return {"defect_type": kind, "joint": joint if kind == "axis" else None,
            "part": part if kind in ("axis", "offset") else None, "evidence_steps": steps}


def main(args):
    root = args.out.resolve()
    if root.exists():
        raise ValueError(f"Output exists: {root}")
    root.mkdir(parents=True)
    bench = Path("../TurnItOver-dataset/data/benchmarks/partnet-mobility-formal-v1").resolve()
    annotations = [json.loads(x) for x in (bench / "oracle-first-failure-v4/annotations.private.jsonl").read_text().splitlines()]
    eligible = defaultdict(lambda: defaultdict(list))
    for row in annotations:
        for label in row["labels"]:
            if label["defect_id"] in ("kinematics.joint_axis", "structure.part_offset"):
                eligible[(row["split"], row["asset_id"])][label["defect_id"]].append((row, label))
    selected = {}
    for key, value in eligible.items():
        if args.assets and key[1] not in args.assets:
            continue
        if args.motion_v2 and key[0] == "validation" and "kinematics.joint_axis" not in value:
            selected[key] = value
            continue
        if len(value) != 2 or key[0] not in ("train", "validation"):
            continue
        combined = {row["id"] for row, _ in value["kinematics.joint_axis"]} & {
            row["id"] for row, _ in value["structure.part_offset"]}
        if combined:
            uid = sorted(combined)[0]
            selected[key] = {defect: [pair for pair in pairs if pair[0]["id"] == uid]
                             for defect, pairs in value.items()}
        else:
            selected[key] = value
    raw = {}
    for sample, _ in records(sorted((bench / "raw").glob("*.tar"))):
        uid = hashlib.sha256(f"{sample.source_id}:{sample.sample_id}".encode()).hexdigest()[:24]
        raw[uid] = sample
    cfg = load_generate_config(Path("configs/generate_partnet_mobility_supervision.yaml"), repo_root=bench.parents[2])
    harness = Path("../TurnItOver").resolve()
    cfg = replace(cfg, render=replace(cfg.render, harness_dir=harness / "web/dist",
                                      esbuild_bin=harness / "web/node_modules/.bin/esbuild"))
    source = make_source(**cfg.asset_source)
    outputs = {"train": [], "validation": []}
    rejected = []
    with ObservationSession(cfg.render, load_views(cfg.views_path)) as session:
        browser = session.browser_version()
        for (split, asset_id), labels in sorted(selected.items()):
            offset_row, offset_label = labels["structure.part_offset"][0]
            axis_row, axis_label = labels.get("kinematics.joint_axis", [(offset_row, offset_label)])[0]
            axis_sample, offset_sample = raw[axis_row["id"]], raw[offset_row["id"]]
            reference = ObjectProgram.from_spec(source.get(asset_id))
            if reference.sha not in (axis_row["reference_sha"], offset_row["reference_sha"]):
                rejected.append({"asset_id": asset_id, "reason": "reference_identity_mismatch"}); continue
            axis = next((c for c in axis_sample.corruptions if c.defect_id == "kinematics.joint_axis"), None)
            offset = next(c for c in offset_sample.corruptions if c.defect_id == "structure.part_offset")
            joint = reference.spec.joint(axis.params["joint"]) if axis else reference.spec.joints[0]
            part = joint.part
            lo, hi = joint.limits
            rest = min(hi, max(lo, 0.0))
            states = [("front", "rest", rest), ("front", "zero", lo), ("front", "mid", (lo + hi) / 2),
                      ("front", "limit", hi), ("oblique_fl", "rest", rest),
                      ("oblique_fl", "zero", lo), ("oblique_fl", "mid", (lo + hi) / 2),
                      ("oblique_fl", "limit", hi)]
            if args.motion_v2:
                end = max((lo, hi), key=lambda q: abs(q - rest))
                states = [(view, name, value) for view in ("front", "oblique_fl", "back", "right")
                          for name, value in (("rest", rest), ("actuated", end))]
            folder = root / "images" / split / asset_id
            folder.mkdir(parents=True)
            session.load(reference, framing=axis_sample.framing)
            hierarchy = session.query_runtime(RuntimeProperty.HIERARCHY)
            refs = []
            for index, (view, name, value) in enumerate(states):
                session.reset_joints(); session.set_joints({joint.id: value})
                path = folder / f"reference-{index}.png"; path.write_bytes(session.request_view(view).png); refs.append(path)
            session.dispose()
            cases = (("axis", [axis], part), ("offset", [offset], offset.parts[0]), ("clean", [], None))
            if axis is None:
                cases = cases[1:]
            rendered = {}
            for kind, corruptions, target_part in cases:
                candidate = replay_corruptions(reference, corruptions)
                session.load(candidate, framing=axis_sample.framing)
                paths, core = [], []
                for index, ((view, _, value), ref_path) in enumerate(zip(states, refs, strict=True)):
                    session.reset_joints(); session.set_joints({joint.id: value})
                    path = folder / f"{kind}-{index}.png"; path.write_bytes(session.request_view(view).png); paths.append(path)
                    core.append(core_changed_fraction(pixels(path.read_bytes()), pixels(ref_path.read_bytes())))
                session.dispose(); rendered[kind] = (paths, core, candidate.sha, target_part)
            if (axis and max(rendered["axis"][1]) < 0.001) or max(rendered["offset"][1]) < 0.001:
                rejected.append({"asset_id": asset_id, "reason": "defect_not_perceptible",
                                 "axis_max_core": max(rendered["axis"][1]) if axis else None, "offset_max_core": max(rendered["offset"][1])})
                if not args.motion_v2:
                    continue
            public = {"hierarchy": hierarchy, "joint": joint.id, "joint_type": str(joint.type),
                      "states": [{"step": i, "view": v, "state_name": n, "actual_value": q}
                                 for i, (v, n, q) in enumerate(states)],
                      "rest_value": rest, "lower_limit_value": lo, "upper_limit_value": hi}
            prompt = INSTRUCTION + json.dumps(public, sort_keys=True)
            for kind in rendered:
                paths, core, candidate_sha, target_part = rendered[kind]
                visible = [i for i, value in enumerate(core) if value >= 0.001]
                steps = visible if kind in ("axis", "offset") else list(range(len(states)))
                full_target = target(kind, joint.id, target_part, steps)
                image_paths = [p for pair in zip(refs, paths, strict=True) for p in pair]
                row = {"id": f"{asset_id}:{kind}:full", "asset_id": asset_id, "split": split,
                       "condition": "full_motion", "images": [p.relative_to(root).as_posix() for p in image_paths],
                       "image_sha256": [sha(p) for p in image_paths], "messages": [{"role": "user", "content": prompt}],
                       "ordinary_target": full_target, "evidence_target": full_target,
                       "gold": {"class": kind, "joint": joint.id if kind == "axis" else None,
                                "part": target_part, "perceptible_steps": visible, "candidate_sha": candidate_sha}}
                outputs[split].append(row)
            if axis is None:
                continue
            axis_paths, axis_core, candidate_sha, _ = rendered["axis"]
            indices = [0, 2, 4, 6] if args.motion_v2 else [0, 4]
            insufficient_paths = [p for i in indices for p in (refs[i], axis_paths[i])]
            insufficient_public = dict(public); insufficient_public["states"] = [public["states"][i] for i in indices]
            insufficient_prompt = INSTRUCTION + json.dumps(insufficient_public, sort_keys=True)
            outputs[split].append({"id": f"{asset_id}:axis:insufficient", "asset_id": asset_id, "split": split,
                "condition": "rest_only", "images": [p.relative_to(root).as_posix() for p in insufficient_paths],
                "image_sha256": [sha(p) for p in insufficient_paths], "messages": [{"role": "user", "content": insufficient_prompt}],
                "ordinary_target": target("axis", joint.id, part, [0]),
                "evidence_target": target("uncertain", None, None, []),
                "gold": {"class": "uncertain", "latent_class": "axis", "joint": joint.id, "part": part,
                         "perceptible_steps": [], "candidate_sha": candidate_sha}})
    for split, rows in outputs.items():
        (root / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    train_assets = sorted({row["asset_id"] for row in outputs["train"]})
    validation_assets = sorted({row["asset_id"] for row in outputs["validation"]})
    manifest = {"version": 1, "status": "complete", "formal_training": True,
        "protocol": "matched reference/candidate; front+oblique_fl; actual rest/lower/mid/upper values",
        "ordinary_vs_evidence": "identical inputs and row count; only rest-only axis target differs",
        "train_objects": len(train_assets), "validation_objects": len(validation_assets),
        "train_assets": train_assets, "validation_assets": validation_assets,
        "train_rows": len(outputs["train"]), "validation_rows": len(outputs["validation"]),
        "rejected": rejected, "browser": browser, "source_commit": "b64ab7b",
        "frozen_test_used": False, "builder_sha256": sha(Path(__file__)),
        "files": {f"{s}.jsonl": sha(root / f"{s}.jsonl") for s in outputs}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--motion-v2", action="store_true", help="Render four fixed views at rest and farthest legal state; retain rows for downstream observability audit")
    parser.add_argument("--assets", nargs="*", help="Optional exact asset IDs for bounded replay")
    main(parser.parse_args())
