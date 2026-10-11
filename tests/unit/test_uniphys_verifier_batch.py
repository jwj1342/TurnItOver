import hashlib
import json

import pytest

from scripts import run_uniphys_verifier_batch
from turnitover.verifier.contracts import VERIFIER_SCHEMA_VERSION


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _benchmark(tmp_path):
    root = tmp_path / "benchmark"
    (root / "cases").mkdir(parents=True)
    (root / "objects").mkdir()
    catalog = root / "catalog.json"
    catalog.write_text("frozen catalog", encoding="utf-8")
    cases, gold = [], []
    for object_index in range(20):
        object_id = f"UPB_{object_index:08d}"
        split_group = f"uniphys:test:{object_id}"
        object_dir = root / "objects" / object_id
        object_dir.mkdir()
        reference_program = object_dir / "clean.ts"
        reference_program.write_text(f"export default 'reference-{object_index}';", encoding="utf-8")
        image = object_dir / "reference.png"
        image.write_bytes(f"png-{object_index}".encode())
        reference_sha = run_uniphys_verifier_batch.ObjectProgram(
            reference_program.read_text(encoding="utf-8")).sha
        for variant in ("clean", "axis", "offset"):
            case_id = f"{object_id}-{variant}"
            program = root / "cases" / f"{case_id}.ts"
            program.write_text(f"export default '{case_id}';", encoding="utf-8")
            program_sha = run_uniphys_verifier_batch.ObjectProgram(
                program.read_text(encoding="utf-8")).sha
            cases.append({"id": case_id, "program": program.relative_to(root).as_posix(),
                          "reference_program": reference_program.relative_to(root).as_posix(),
                          "reference_images": [image.relative_to(root).as_posix()],
                          "split_group": split_group, "split": "test", "task": "verify"})
            findings = [] if variant == "clean" else [{
                "defect_id": "structure.part_offset", "parts": ["part"], "joint_id": None}]
            gold.append({"id": case_id, "program_sha": program_sha,
                         "reference_sha": reference_sha, "split_group": split_group,
                         "split": "test", "target": {
                             "status": "pass" if variant == "clean" else "fail",
                             "findings": findings}})
    (root / "test.gold.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in gold), encoding="utf-8")
    manifest = {
        "status": "complete", "object_count": 20, "case_count": 60,
        "preflight": {"status": "complete", "validated_cases": 60},
        "verifier_schema_version": VERIFIER_SCHEMA_VERSION,
        "catalog": str(catalog.resolve()), "catalog_sha256": _sha256(catalog), "cases": cases,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _complete_verify(program, output, render, views, config, **kwargs):
    output.mkdir(parents=True)
    result = {"schema_version": VERIFIER_SCHEMA_VERSION, "status": "complete",
              "program_sha": program.sha, "verdict": {"status": "pass"},
              "termination": "judge_finished", "spent": 1}
    (output / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return result


def _score(gold, results, output):
    assert len(gold.read_text(encoding="utf-8").splitlines()) == 60
    report = {"status": "complete", "n": 60}
    output.write_text(json.dumps(report), encoding="utf-8")
    return report


def test_runs_all_frozen_cases_and_scores(tmp_path):
    benchmark = _benchmark(tmp_path)
    output = tmp_path / "results"

    report = run_uniphys_verifier_batch.run(
        benchmark, output, tmp_path / ".env", policy="runtime",
        verify_case=_complete_verify, score=_score)

    assert report["status"] == "complete"
    assert report["totals"] == {"planned": 60, "completed": 60, "errors": 0}
    assert (output / "metrics.json").is_file()
    assert all(task["status"] == "complete" for task in report["tasks"])


def test_rejects_a_case_path_outside_the_frozen_benchmark(tmp_path):
    benchmark = _benchmark(tmp_path)
    manifest_path = benchmark / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["program"] = "../escape.ts"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid program"):
        run_uniphys_verifier_batch.run(
            benchmark, tmp_path / "results", tmp_path / ".env", policy="runtime",
            verify_case=_complete_verify, score=_score)


def test_rejects_duplicate_variant_programs(tmp_path):
    benchmark = _benchmark(tmp_path)
    manifest = json.loads((benchmark / "manifest.json").read_text(encoding="utf-8"))
    first, second = manifest["cases"][:2]
    second_program = benchmark / second["program"]
    second_program.write_text((benchmark / first["program"]).read_text(encoding="utf-8"), encoding="utf-8")
    rows = [json.loads(line) for line in (benchmark / "test.gold.jsonl").read_text().splitlines()]
    rows[1]["program_sha"] = rows[0]["program_sha"]
    (benchmark / "test.gold.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate candidate program SHA"):
        run_uniphys_verifier_batch.run(
            benchmark, tmp_path / "results", tmp_path / ".env", policy="runtime",
            verify_case=_complete_verify, score=_score)


def test_preserves_completed_case_when_a_later_case_fails(tmp_path):
    benchmark = _benchmark(tmp_path)
    output = tmp_path / "results"
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("stopped")
        return _complete_verify(*args, **kwargs)

    with pytest.raises(RuntimeError, match="stopped"):
        run_uniphys_verifier_batch.run(
            benchmark, output, tmp_path / ".env", policy="runtime",
            verify_case=fail_second, score=_score)

    manifest = json.loads((output / "batch-manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["tasks"][0]["status"] == "complete"
    assert manifest["tasks"][1]["status"] == "runner_error"
