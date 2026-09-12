"""Reproducible synthetic examples; run from the repository root after setup_env.sh."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from turnitover.assets.toy import toy_cabinet
from turnitover.core.program import ObjectProgram
from turnitover.preview import export_preview
from turnitover.render.session import RenderConfig
from turnitover.render.views import load_views
from turnitover.telemetry import git_state, now_iso
from turnitover.verifier.runner import VerifyConfig, verify


def run(output: Path, *, videos: bool = True) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    spec = toy_cabinet()
    reference = ObjectProgram.from_spec(spec)
    cases = {
        "triangle-budget": ObjectProgram.from_spec(
            spec.replace_part("drawer_0", segments=(16, 16, 16))
                .replace_part("drawer_1", segments=(16, 16, 16))),
        "joint-axis": ObjectProgram.from_spec(spec.replace_joint("door", axis=(0.0, 0.0, 1.0))),
    }
    render = RenderConfig(harness_dir=root / "web/dist", esbuild_bin=root / "web/node_modules/.bin/esbuild",
                          width=384, height=384)
    views = load_views(root / "configs/views.yaml")
    summary = {"status": "running", "created_at": now_iso(), "git": git_state(root),
               "synthetic": True, "policy": "runtime", "budget": 12, "cases": []}

    def save():
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    save()
    try:
        export_preview(reference, render, views, output / "reference", fps=12, seconds=2, videos=videos)
        for name, candidate in cases.items():
            case = output / name
            export_preview(candidate, render, views, case / "preview", fps=12, seconds=2, videos=videos)
            result = verify(candidate, case / "verification", render, views,
                            VerifyConfig(mode="runtime", budget=12), reference_program=reference,
                            references=(output / "reference/screenshots/06.png",))
            if result["status"] != "complete" or result.get("audit", {}).get("status") != "complete":
                raise RuntimeError(f"Incomplete verification or audit: {name}")
            checks = result["audit"]["checks"]
            failed = [inv for check in checks for inv in check["invariants"] if not inv["passed"]]
            expected = (result["verdict"]["status"] == "fail" if name == "triangle-budget" else
                        result["verdict"]["status"] == "uncertain" and
                        any(inv["state"] == "joint:door:limit" for inv in failed) and
                        not any(inv["state"] == "rest" for inv in failed))
            if not expected:
                raise RuntimeError(f"Unexpected example outcome: {name}; inspect its audit")
            summary["cases"].append({"name": name, "program_sha": candidate.sha,
                                     "verdict": result["verdict"]["status"], "spent": result["spent"],
                                     "checks": [{"checker_id": c["checker_id"], "passed": c["passed"]}
                                                for c in checks], "failed_invariants": failed})
            save()
        rows = "".join(
            f'<tr><td>{html.escape(c["name"])}</td><td>{c["verdict"]}</td>'
            f'<td><a href="{c["name"]}/preview/index.html">Screenshots and videos</a></td>'
            f'<td><a href="{c["name"]}/verification/index.html">Verifier evidence</a></td>'
            f'<td><a href="{c["name"]}/verification/audit.json">Independent gold audit</a></td></tr>'
            for c in summary["cases"])
        (output / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>TurnItOver examples</title>'
            '<style>body{font:17px system-ui;max-width:1100px;margin:40px auto;padding:20px}'
            'td,th{padding:12px;text-align:left}img{max-width:100%}</style>'
            '<h1>Two synthetic verification examples</h1>'
            '<p>The runtime baseline detects excess triangles. It cannot judge joint correctness: '
            'the joint-axis example remains uncertain, while a separate privileged audit finds '
            'a mismatch only after the door moves. Audit evidence is never given to the judge.</p>'
            '<p><a href="reference/index.html">Reference screenshots and videos</a> · '
            '<a href="summary.json">Machine-readable summary</a></p>'
            '<table><tr><th>Case</th><th>Runtime verdict</th><th>Preview</th><th>Trace</th><th>Audit</th></tr>'
            + rows + '</table><img src="reference/overview.png" alt="Reference views">')
        summary["status"] = "complete"
    except Exception as exc:
        summary.update(status="error", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        summary["finished_at"] = now_iso()
        save()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    run(args.out.resolve(), videos=not args.no_video)
    print(args.out / "index.html")
