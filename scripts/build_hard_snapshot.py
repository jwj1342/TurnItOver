"""Export the sealed Hard formal candidate without programs or source archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

if __package__:
    from .easy_release_archive import sha256
    from .hard_bundle_contract import sanitize_check, sanitize_gold, STATES
    from .hard_bundle_inventory import write_manifest
    from .prepare_hard_review import prepare_review
    from .validate_hard_bundle import validate_bundle
else:
    from easy_release_archive import sha256
    from hard_bundle_contract import sanitize_check, sanitize_gold, STATES
    from hard_bundle_inventory import write_manifest
    from prepare_hard_review import prepare_review
    from validate_hard_bundle import validate_bundle


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        )
    )


def _load_return(path: Path):
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError("Hard human return must be one JSON array")
    return value


def export_snapshot(
    source: Path,
    strict_audit: Path,
    human_return: Path,
    output: Path,
    source_code_commit: str,
) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError("Hard snapshot output must be new and outside source")
    if not re.fullmatch(r"[0-9a-f]{40}", source_code_commit):
        raise ValueError("Source code commit must be complete")
    manifest_path = source / "manifest.json"
    manifest_sha = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    actual = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != set(manifest["files"]):
        raise ValueError("Hard formal seal does not exactly cover source files")
    for name, digest in manifest["files"].items():
        if sha256(source / name) != digest:
            raise ValueError(f"Hard source SHA differs: {name}")
    validation = json.loads((source / "validation.json").read_text())
    strict = json.loads(strict_audit.read_text())
    if (
        validation.get("objects") != 75
        or validation.get("cases") != 75
        or strict.get("objects") != 75
        or strict.get("errors") != []
        or strict.get("status") not in ("passed", "passed_with_warnings")
    ):
        raise ValueError("Hard source validation is incomplete")
    gold_raw, checks_raw = _rows(source / "private/gold.jsonl"), _rows(
        source / "private/checks.jsonl"
    )
    public, review = _rows(source / "pool_manifest.jsonl"), _rows(
        source / "review_manifest.jsonl"
    )
    blank = _rows(source / "human_review.jsonl")
    returned = _load_return(human_return)
    sample_ids = [row["case_id"] for row in review]
    if (
        len(gold_raw) != 75
        or len(public) != 75
        or len(review) != 10
        or len(blank) != 10
        or [row.get("case_id") for row in returned] != sample_ids
    ):
        raise ValueError("Hard source or human return coverage differs")
    decisions = {"hard": 0, "clean": 0, "pending": 0, "other": 0}
    for row in returned:
        decision = row.get("final_decision")
        decisions[
            (
                decision
                if decision in ("hard", "clean")
                else "pending" if decision is None else "other"
            )
        ] += 1
    if decisions != {"hard": 6, "clean": 3, "pending": 1, "other": 0}:
        raise ValueError("Hard human return differs from the frozen 6/3/1 observation")

    output.mkdir(parents=True)
    for name in (
        "pool_manifest.jsonl",
        "review_manifest.jsonl",
        "human_review.jsonl",
        "sample.json",
        "train_manifest.jsonl",
        "review.html",
        "review-all.html",
    ):
        shutil.copyfile(source / name, output / name)
    _write_rows(output / "private/gold.jsonl", [sanitize_gold(row) for row in gold_raw])
    _write_rows(
        output / "private/checks.jsonl", [sanitize_check(row) for row in checks_raw]
    )
    _write_rows(output / "private/human_spot_review.jsonl", returned)
    _write_json(output / "source/validation.json", validation)
    _write_json(output / "source/strict-audit.json", strict)
    programs = []
    for row in gold_raw:
        name = row["program"]
        programs.append(
            {
                "case_id": row["case_id"],
                "sha256": manifest["files"][name],
                "size": (source / name).stat().st_size,
                "candidate_sha": row["candidate_sha"],
                "reference_sha": row["reference_sha"],
            }
        )
    _write_json(
        output / "construction_registry.json",
        {
            "availability": "local_custody",
            "source_manifest_sha256": manifest_sha,
            "programs": sorted(programs, key=lambda row: row["case_id"]),
        },
    )
    _write_json(
        output / "source/metadata.json",
        {
            "version": 1,
            "source_manifest_sha256": manifest_sha,
            "source_strict_audit_sha256": sha256(strict_audit),
            "human_return_sha256": sha256(human_return),
            "source_code_commit": source_code_commit,
            "source_dirty": True,
            "generation_policy": manifest["policy"],
            "archive_availability": "local_custody",
            "benchmark_frozen": False,
            "training_ready": False,
            "human_spot": decisions,
        },
    )
    for row in public:
        cid = row["case_id"]
        for view in (row["initial_view"], row["supplementary_view"]):
            for role in ("reference", "candidate"):
                for state in STATES:
                    name = f"native/{cid}/{view}/{role}_{state}.png"
                    target = output / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / name, target)
    with tempfile.TemporaryDirectory(
        prefix=".hard-review-", dir=output.parent
    ) as temporary:
        review_root = Path(temporary) / "review"
        prepare_review(output, review_root)
        shutil.copytree(review_root, output / "review_bundle")
    (output / "README.md").write_text(
        """# Hard 候选数据\n\n75 个独立复杂多关节物体、75 条单缺陷案例、900 张两视角三状态原生图。\n这是候选包，不是冻结 benchmark 或训练导出；训练清单为空。随机 10 条原始人工回件\n按原样保留为 6 hard、3 clean、1 pending，项目所有者仍批准按 Hard 候选发布。\n\n构造程序、mesh、URDF、下载缓存与历史拒绝样本仅本地保管。Gold 中的绝对源路径已移除，\nconstruction_registry.json 只保留构造程序哈希、大小和程序身份。本包不能独立重放几何构造。\n独立审核者只应接收 hard review ZIP，不应接收本含 Gold 的协调者候选包。\n"""
    )
    write_manifest(output)
    report = validate_bundle(output)
    if sha256(manifest_path) != manifest_sha:
        raise ValueError("Hard source changed during export")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--strict-audit", type=Path, required=True)
    parser.add_argument("--human-return", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-code-commit", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export_snapshot(
                args.source,
                args.strict_audit,
                args.human_return,
                args.output,
                args.source_code_commit,
            ),
            indent=2,
        )
    )
