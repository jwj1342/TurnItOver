import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser(description="Recompute Table 2 visibility from frozen validation metadata")
parser.add_argument('--dataset', type=Path, required=True)
parser.add_argument('--runtime', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
root = args.dataset.resolve()
out = args.out.resolve()
if out.exists():
    raise ValueError('Output directory must not already exist')
sys.path.insert(0, str(args.runtime.resolve()))

from turnitover.oracle.sft_data import _annotations, _check_annotation_identity, validate_target
from turnitover.oracle.supervision_validation import public_payload

manifest_hash = "b64e50f027a8788554f9248b6aa7da51b4010e16103f3ffafc9e6bb41806179b"
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert digest(root / "manifest.json") == manifest_hash
manifest = json.loads((root / "manifest.json").read_bytes())
for name in ("annotations.private.jsonl", "validation.episodes.jsonl"):
    assert digest(root / name) == manifest["files"][name]

annotations = _annotations(root, "validation")
records = [json.loads(line) for line in (root / "validation.episodes.jsonl").read_bytes().splitlines()]
assert len(records) == len({r["episode_id"] for r in records}) == 120
records = [r for r in records if r["admission"] == "eligible"]
assert len(records) == len(annotations) == 104
assert {r["episode_id"] for r in records} == set(annotations)
assert Counter(r["difficulty"] for r in records) == {"easy": 39, "medium": 35, "hard": 30}
assert len({r["data_identity"]["split_group"] for r in records}) == 60

rows = []
for record in records:
    case = record["episode_id"]
    annotation = annotations[case]
    _check_annotation_identity(record, annotation)
    validate_target(record, "diagnosis", "evidence", annotation)
    assert record["difficulty"] == annotation["difficulty"]
    if annotation["is_clean"]:
        continue
    assert record["evidence"]["proxy"] == "counterfactual_pixel_exposure"
    payload = public_payload(record)
    steps = {}
    for entry in payload["history"]:
        match = re.fullmatch(r"\d+-(?:(.+)_)?(candidate|reference)_(rest|mid|limit)\.png",
                             Path(entry["image_ref"]).name)
        assert match, entry
        view, role, state = match.groups()
        assert view is None or view == entry["action"]["view_id"]
        steps[entry["step"]] = (entry["action"]["view_id"], role, state)

    views = list(dict.fromkeys(view for view, _, _ in steps.values()))
    assert len(views) == (2 if record["difficulty"] == "hard" else 1)
    full = set(steps)
    target = json.loads(record["messages"][1]["content"])["verdict"]
    exposure = {step for finding in target["findings"] for step in finding["evidence_steps"]}
    assert exposure <= full
    assert all(steps[step][1] == "candidate" for step in exposure)
    assert bool(exposure) == bool(record["evidence"]["covered_label_indices"])
    masks = {
        "viewpoint_at_rest": {s for s, (_, _, state) in steps.items() if state == "rest"} if len(views) > 1 else None,
        "fixed_view_with_rest": {s for s, (view, _, _) in steps.items() if view == views[0]},
        "fixed_view_non_rest": {s for s, (view, _, state) in steps.items() if view == views[0] and state != "rest"},
        "full_sequence": full,
    }
    rows.append({
        "case": case, "difficulty": record["difficulty"], "fixed_view": views[0],
        "exposure_steps": sorted(exposure),
        "conditions": {key: None if mask is None else {
            "visible": bool(exposure & mask), "matched_steps": sorted(exposure & mask),
        } for key, mask in masks.items()},
    })

assert Counter(row["difficulty"] for row in rows) == {"easy": 19, "medium": 15, "hard": 10}
summary = {}
for condition in masks:
    summary[condition] = {}
    for tier in ("easy", "medium", "hard", "all"):
        flags = [row["conditions"][condition]["visible"] for row in rows
                 if (tier == "all" or row["difficulty"] == tier) and row["conditions"][condition] is not None]
        summary[condition][tier] = None if not flags else {
            "visible": sum(flags), "eligible": len(flags), "percent": 100 * sum(flags) / len(flags),
        }

report = {"manifest_sha256": manifest_hash, "code_commit": "89e07db82890af02ae651dcd8b86082981fb7d1b",
          "visibility": summary, "cases": rows}
out.mkdir(parents=True)
(out / "visibility.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
