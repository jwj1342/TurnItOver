"""Synthetic portable Hard bundle fixture; never benchmark evidence."""

from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

from scripts.hard_bundle_contract import KIND_COUNTS, STATES
from scripts.hard_bundle_inventory import write_manifest
from scripts.prepare_hard_review import prepare_review


def rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in values))


def object_rows(index, kind):
    cid = f"hp-{index:014x}"
    ref = f"{index:016x}"
    cand = f"{index+1000:016x}"
    source = {
        key: f"synthetic-{key}-{index}"
        for key in (
            "source_identity",
            "model_id",
            "archive_sha256",
            "original_spec_sha256",
            "original_geometry_fingerprint",
        )
    }
    source.update(
        split="train",
        complexity={"active_joints": 3, "parts": 5, "category": "Synthetic"},
    )
    public = {
        "case_id": cid,
        "initial_view": "front",
        "supplementary_view": "left",
        "states": list(STATES),
        "native_root": f"native/{cid}",
        "image_size": [256, 256],
        "status": "pending_human_review",
    }
    gold = {
        "case_id": cid,
        "asset_id": f"asset-{index}",
        "kind": kind,
        "reference_sha": ref,
        "candidate_sha": cand,
        "training_eligibility": "pending_human_review",
        "construction_program": "local_custody",
        "source": source,
    }
    checks = {
        "case_id": cid,
        "reference_sha": ref,
        "candidate_sha": cand,
        "code_correct": False,
        "privileged": True,
        "used_by_student": False,
        "checks": [],
        "clean_check": {"code_correct": True},
        "source_authenticity": {
            "passed": True,
            "source_files": [{"logical_name": "source-file-0000", "sha256": "a" * 64}],
        },
    }
    return public, gold, checks


def build_bundle(root: Path):
    root.mkdir()
    public = []
    gold = []
    checks = []
    index = 0
    for kind, count in sorted(KIND_COUNTS.items()):
        for _ in range(count):
            p, g, c = object_rows(index, kind)
            public.append(p)
            gold.append(g)
            checks.append(c)
            for view in ("front", "left"):
                for role in ("reference", "candidate"):
                    for state in STATES:
                        path = root / f'native/{p["case_id"]}/{view}/{role}_{state}.png'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        Image.new("RGB", (256, 256), (index % 255, 40, 70)).save(path)
            index += 1
    selected = public[:10]
    blank = [
        {
            "case_id": p["case_id"],
            "status": "pending_human_review",
            "reviewer": None,
            "reviewed_at": None,
            "human_judgment": {"initial_verdict": None},
        }
        for p in selected
    ]
    returned = []
    for i, p in enumerate(selected):
        returned.append(
            {
                "case_id": p["case_id"],
                "final_decision": "hard" if i < 6 else "clean" if i < 9 else None,
            }
        )
    rows(root / "pool_manifest.jsonl", public)
    rows(root / "review_manifest.jsonl", selected)
    rows(root / "human_review.jsonl", blank)
    rows(root / "private/gold.jsonl", gold)
    rows(root / "private/checks.jsonl", checks)
    rows(root / "private/human_spot_review.jsonl", returned)
    (root / "train_manifest.jsonl").write_text("")
    (root / "sample.json").write_text("{}\n")
    case_data = [
        {
            k: p[k]
            for k in ("case_id", "initial_view", "supplementary_view", "native_root")
        }
        for p in selected
    ]
    (root / "review.html").write_text(
        '<script type="application/json">' + json.dumps(case_data) + "</script>"
    )
    (root / "review-all.html").write_text("synthetic public page")
    (root / "README.md").write_text("synthetic fixture")
    (root / "source").mkdir()
    (root / "source/validation.json").write_text('{"objects":75}\n')
    (root / "source/strict-audit.json").write_text(
        '{"status":"passed_with_warnings","errors":[],"warnings":[{},{},{}]}\n'
    )
    (root / "source/metadata.json").write_text(
        json.dumps(
            {
                "benchmark_frozen": False,
                "training_ready": False,
                "source_manifest_sha256": "a" * 64,
            }
        )
        + "\n"
    )
    (root / "construction_registry.json").write_text(
        json.dumps(
            {
                "availability": "local_custody",
                "programs": [{"case_id": p["case_id"]} for p in public],
            }
        )
        + "\n"
    )
    prepare_review(root, root.parent / (root.name + "-review"))
    (root.parent / (root.name + "-review")).rename(root / "review_bundle")
    write_manifest(root)
    return root
