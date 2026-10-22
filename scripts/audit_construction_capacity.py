#!/usr/bin/env python3
"""Capacity audit for the Easy/Medium construction target (issue #9): how many more objects can we reach?

Read-only: local conversion/screening records plus one metadata listing of the pinned upstream revision.
No rendering, no model calls, no downloads of archives.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import urllib.request

from turnitover.core.jsonio import write_json_atomic

ROOT = Path(__file__).resolve().parent.parent
V4 = ROOT / "docs/evidence/partnet-mobility-level0-axis-v4"
ORIGINAL = ROOT / "data/assets/partnet-mobility-trainval-v1"
ROUND2 = ROOT / "data/assets/partnet-mobility-level0-round2"
SPLITS = ORIGINAL / "splits.json"
REVISION = "ee0aa3ef1df16181d76d83f7415aa8c94ed1da8f"
LISTING = f"https://huggingface.co/api/datasets/sapien-sim/PartNetMobility/tree/{REVISION}"


def upstream_archives() -> set[str]:
    """All numeric archive ids at the pinned revision, following Hugging Face's Link pagination."""
    ids, url = set(), LISTING
    while url:
        with urllib.request.urlopen(url, timeout=60) as response:
            ids.update(m.group(1) for item in json.load(response) if (m := re.fullmatch(r"(\d+)\.zip", item["path"])))
            link = response.headers.get("Link", "")
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = match.group(1) if match else None
    return ids


def klass(joints: int) -> str:
    return "single" if joints == 1 else "multi"


def rate(num, den):
    return None if den == 0 else round(num / den, 4)


def audit(offline_ids: set[str] | None) -> dict:
    conversion = {}
    for base in (ORIGINAL, ROUND2):
        for row in json.loads((base / "conversion.json").read_text())["assets"]:
            if row["status"] == "structural_candidate":
                conversion.setdefault(row["asset_id"], row)
    downloaded = {r["asset_id"] for r in json.loads((ROOT / "data/partnet-mobility-level0-round2/download-manifest.json").read_text())["archives"]}
    used = {o["asset_id"].removeprefix("PM_") for o in json.loads((V4 / "object_splits.json").read_text())["objects"]}
    validation = {g.rsplit(":", 1)[1] for g, s in json.loads(SPLITS.read_text())["groups"].items() if s == "validation"}
    screened = {}
    for name in ("screening.json", "expansion_screening.json"):
        for asset in json.loads((V4 / "private" / name).read_text())["assets"]:
            screened[asset["asset_id"].removeprefix("PM_")] = asset["status"]
    for aid in used:  # the 38 inherited objects passed in earlier packages, not in these two files
        screened.setdefault(aid, "automatic_pass")

    pool = {a: r for a, r in conversion.items()
            if a not in used and a not in validation and r["frozen_test_overlap"] is False}
    unscreened = {a for a in pool if a not in screened}
    rejected_after_screen = {a for a in pool if a in screened}

    structural_by_class = Counter(klass(r["joints"]) for r in conversion.values())
    screen_yield = {}
    for cls in ("single", "multi"):
        done = [a for a, st in screened.items() if a in conversion and klass(conversion[a]["joints"]) == cls]
        passed = [a for a in done if screened[a] == "automatic_pass"]
        screen_yield[cls] = {"screened": len(done), "passed": len(passed), "rate": rate(len(passed), len(done))}
    structural_rate = rate(len(conversion), len(downloaded))
    pool_by_class = Counter(klass(conversion[a]["joints"]) for a in unscreened)
    used_by_class = Counter(klass(conversion[a]["joints"]) for a in used if a in conversion)

    report = {
        "scope": "read-only capacity audit; no rendering, model calls or archive downloads",
        "revision": REVISION, "upstream_archives": len(offline_ids) if offline_ids else None,
        "downloaded_archives": len(downloaded),
        "structural_candidates": len(conversion), "structural_rate_of_downloads": structural_rate,
        "structural_by_class": dict(structural_by_class),
        "used_in_v4": {"objects": len(used), "by_class": dict(used_by_class)},
        "protected_excluded": {"original_validation_groups": len(validation & set(conversion))},
        "screening_yield_by_class": screen_yield,
        "unscreened_local_candidates": {"objects": len(unscreened), "by_class": dict(pool_by_class)},
        "screened_but_not_passed": len(rejected_after_screen),
    }
    targets = {"easy_single_joint_objects": 75, "medium_objects": 75}
    expected_local_single = (pool_by_class["single"] * (screen_yield["single"]["rate"] or 0))
    expected_local_multi = (pool_by_class["multi"] * (screen_yield["multi"]["rate"] or 0))
    have_single, have_multi = used_by_class["single"], used_by_class["multi"]
    need_single = max(0, targets["easy_single_joint_objects"] - have_single - expected_local_single)
    need_multi = max(0, targets["medium_objects"] - have_multi - expected_local_multi)
    single_share = rate(structural_by_class["single"], len(conversion))
    pass_single = screen_yield["single"]["rate"] or 0
    new_archives, sensitivity = None, {}
    if pass_single and single_share and structural_rate:
        for factor in (1.0, 0.5):  # 0.5: the screening pass rate on new archives is half the observed one
            sensitivity[str(factor)] = round(need_single / (single_share * pass_single * factor * structural_rate))
        new_archives = sensitivity["1.0"]
    report["projection"] = {
        "targets": targets, "have": {"single": have_single, "multi": have_multi},
        "expected_from_unscreened_local": {"single": round(expected_local_single, 1), "multi": round(expected_local_multi, 1)},
        "still_needed_after_local": {"single": round(need_single, 1), "multi": round(need_multi, 1)},
        "additional_archives_to_download_estimate": new_archives,
        "estimate_by_pass_rate_factor": sensitivity,
        "undownloaded_upstream_archives": len(offline_ids - downloaded) if offline_ids else None,
        "assumptions": ["screening pass rate and single-joint share are assumed to carry over to new archives",
                        "original-pool screened-and-rejected objects are not re-screened",
                        "undownloaded archives are not yet checked against the frozen Test identities"]}
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="skip the upstream listing")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit("output exists")
    report = audit(None if args.offline else upstream_archives())
    write_json_atomic(args.out, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
