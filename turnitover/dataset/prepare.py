"""Prepare isolated model inputs and gold targets with explicit asset-group splits."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from turnitover.core.actions import OBSERVATION_ACTIONS, RuntimeProperty
from turnitover.core.serde import to_dict
from turnitover.storage.reader import iter_samples

SPLITS = ("train", "validation", "test")


def records(shards):
    seen = set()
    for shard in sorted(map(Path, shards)):
        manifest = shard.with_suffix(".manifest.json")
        if not manifest.exists() or json.loads(manifest.read_text()).get("status") != "complete":
            raise ValueError(f"Missing or incomplete shard manifest: {shard}")
        for sample, members in iter_samples(shard):
            identity = (sample.source_id, sample.sample_id)
            if identity in seen:
                raise ValueError(f"Duplicate sample identity: {identity}")
            seen.add(identity)
            if sample.schema_version not in (2, 3) or not sample.dataset.get("split_group"):
                raise ValueError("Dataset export requires validated schema v2/v3 samples; regenerate legacy shards")
            yield sample, members


def make_splits(shards, output: Path, seed: int = 0):
    groups = {s.dataset["split_group"] for s, _ in records(shards)}
    if len(groups) < 3:
        raise ValueError("Need at least three independent split groups; never split variants of one asset across sets")
    ordered = sorted(groups, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
    n_holdout = max(1, len(ordered) // 10)
    assignment = {g: ("test" if i < n_holdout else "validation" if i < 2*n_holdout else "train")
                  for i, g in enumerate(ordered)}
    with output.open("x") as f:
        json.dump({"version": 1, "seed": seed, "unit": "split_group", "groups": assignment}, f, indent=2)
    return assignment


def diagnosis(sample):
    findings = []
    for c in sample.corruptions:
        if c.defect_id == "structure.part_offset":
            text = f"Part {c.parts[0]} is displaced by {c.params['vector']} metres relative to the reference."
        elif c.defect_id == "kinematics.joint_axis":
            text = f"Joint {c.params['joint']} on part {c.parts[0]} has an axis error of {c.params['angle_deg']:.2f} degrees."
        elif c.defect_id == "runtime.triangle_budget":
            text = f"Scene triangle count {c.params['triangles_after']} exceeds the configured budget."
        else:
            raise ValueError(f"No diagnosis template for {c.defect_id}")
        findings.append({"defect_id": c.defect_id, "parts": list(c.parts), "severity": c.severity,
                         "description": text, "parameters": c.params})
    return {"status": "fail" if findings else "pass", "findings": findings,
            "scope": "configured_synthetic_constraints", "privileged": True,
            "training_eligibility": "requires_observation_exposure_review",
            "action_supervision": None}


def prepare(shards, splits_path: Path, output: Path):
    raw = json.loads(splits_path.read_text())
    assignment = raw.get("groups", {})
    if raw.get("version") != 1 or not assignment or any(v not in SPLITS for v in assignment.values()):
        raise ValueError("Split manifest requires version=1 and group -> train/validation/test assignments")
    if output.exists():
        raise ValueError("Dataset output must not exist")
    output.mkdir(parents=True)
    counts, defects, quality_counts, groups = Counter(), Counter(), Counter(), {}
    references = {}
    streams = {}
    try:
        for split in SPLITS:
            for kind in ("inputs", "gold"):
                streams[split, kind] = (output / f"{split}.{kind}.jsonl").open("w")
        with (output / "quarantine.jsonl").open("w") as quarantine:
            for sample, members in records(shards):
                meta = sample.dataset
                group = meta["split_group"]
                if group not in assignment:
                    raise ValueError(f"Missing split assignment: {group}")
                split = assignment[group]
                # A mislabeled duplicate reference must not cross split boundaries.
                for identity in (f"asset:{sample.source_id}:{sample.asset_id}", f"sha:{sample.reference_sha}"):
                    if identity in references and references[identity] != split:
                        raise ValueError(f"Reference leakage across splits: {identity}")
                    references[identity] = split
                groups[group] = split
                quality = meta.get("quality", {}).get("status", "unvalidated")
                quality_counts[quality] += 1
                uid = hashlib.sha256(f"{sample.source_id}:{sample.sample_id}".encode()).hexdigest()[:24]
                if quality != "accepted" or not sample.compile_ok:
                    quarantine.write(json.dumps({"id": uid, "sample_id": sample.sample_id, "split": split,
                                                  "quality": meta.get("quality"), "error": sample.error}) + "\n")
                    continue
                image_map = {}
                observations = [o for o in sample.trajectory if isinstance(o.action, OBSERVATION_ACTIONS)]
                image_refs = meta["reference_images"] + [o.image_ref for o in observations if o.image_ref]
                for index, ref in enumerate(dict.fromkeys(image_refs)):
                    if ref not in members:
                        raise ValueError(f"Missing image {ref} for {sample.sample_id}")
                    target = Path("images") / uid / f"{index:03d}.png"
                    (output / target).parent.mkdir(parents=True, exist_ok=True)
                    (output / target).write_bytes(members[ref])
                    image_map[ref] = target.as_posix()
                limits = meta["constraints"].get("runtime.budget", {})
                context = {"task": "Verify the articulated candidate against the reference and configured constraints.",
                           "parts": meta["parts"], "joints": meta["joints"], "views": meta["views"],
                           "runtime_properties": [p.value for p in RuntimeProperty],
                           "action_types": ["request_view", "actuate_joint", "query_runtime"],
                           "max_triangles": limits.get("max_triangles", 5000),
                           "max_draw_calls": limits.get("max_draw_calls", 32)}
                model_input = {"id": uid, "context": context,
                               "reference_images": [image_map[r] for r in meta["reference_images"]],
                               "history": [{"step": o.step, "action": to_dict(o.action),
                                            "image_ref": image_map.get(o.image_ref), "payload": o.payload}
                                           for o in observations]}
                gold = {"id": uid, "source_id": sample.source_id, "sample_id": sample.sample_id,
                        "asset_id": sample.asset_id, "category": meta["category"], "split_group": group,
                        "provenance": {k: meta[k] for k in ("origin", "license")},
                        "quality": meta["quality"], "target": diagnosis(sample), "checks": to_dict(sample.checks),
                        "candidate_program": sample.program_source, "reference_program": meta["reference_program_source"]}
                streams[split, "inputs"].write(json.dumps(model_input) + "\n")
                streams[split, "gold"].write(json.dumps(gold) + "\n")
                counts[split] += 1
                defects.update(c.defect_id for c in sample.corruptions)
        manifest = {"version": 1, "status": "complete", "counts": dict(counts), "quality_counts": dict(quality_counts),
                    "defects": dict(defects), "groups": groups, "split_manifest": raw,
                    "shards": [{"path": str(Path(p).resolve()),
                                "manifest": json.loads(Path(p).with_suffix(".manifest.json").read_text())} for p in shards],
                    "limitations": ["Gold diagnosis is privileged, not evidence-grounded SFT.",
                                    "Scripted trajectories are not expert action supervision.",
                                    "Asset-group splits do not establish category or corruption-combination generalization."]}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest
    except Exception as exc:
        (output / "manifest.json").write_text(json.dumps({"version": 1, "status": "incomplete", "error": str(exc)}))
        raise
    finally:
        for f in streams.values():
            f.close()
