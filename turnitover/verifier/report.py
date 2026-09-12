"""Offline HTML presentation of saved verifier evidence; no policy or browser execution."""
from __future__ import annotations

import html
from pathlib import Path


def write_report(output: Path, data: dict, observations):
    esc = html.escape
    verdict = data["verdict"]
    findings = "".join(f'<li><b>{esc(f["defect_id"])}</b>: {esc(f["description"])}<br>'
                       f'Fix: {esc(f["suggested_fix"])} · Evidence: {f["evidence_steps"]}</li>' for f in verdict["findings"])
    cards = "".join(f'<figure><img src="{esc(o.image_ref)}"><figcaption>Step {o.step}: '
                    f'{esc(type(o.action).__name__)}</figcaption></figure>' for o in observations if o.image_ref)
    refs = "".join(f'<figure><img src="{esc(r["path"])}"><figcaption>Reference input</figcaption></figure>'
                   for r in data["references"])
    limitations = "".join(f"<li>{esc(x)}</li>" for x in verdict["limitations"])
    audit_link = '<a href="audit.json">Privileged gold audit</a>' if "audit" in data else ""
    validation = f'<p>Validation: {esc(data["error_message"])}</p>' if data.get("error_message") else ""
    calls = "".join(f'<li><a href="{esc(c["path"])}/prompt.txt">Call {i}: prompt</a> · '
                    f'<a href="{esc(c["path"])}/response.txt">raw response</a> · '
                    f'<a href="{esc(c["path"])}/metadata.json">usage/status</a></li>'
                    for i, c in enumerate(data.get("model_calls", [])))
    (output / "index.html").write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>TurnItOver verification</title>
<style>body{{max-width:1100px;margin:32px auto;padding:0 24px;background:#17171c;color:#eee;font:16px system-ui}}
a{{color:#a9ceff}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}}
figure{{margin:0;background:#24242b}}img{{width:100%}}figcaption{{padding:12px}}li{{margin:12px 0}}</style>
<h1>Verification: {esc(verdict['status'])}</h1><p>{esc(verdict['summary'])}</p>
<p>Termination: {esc(data['termination'])} · Observations spent: {data.get('spent','unknown')} / {data['config']['budget']}</p>
{validation}
<p><a href="result.json">Full result</a> · <a href="trajectory.jsonl">Observation trace</a> ·
<a href="feedback.json">Repair feedback</a> {audit_link}</p><h2>Findings</h2><ul>{findings}</ul>
<h2>Limitations</h2><ul>{limitations}</ul><h2>Reference inputs</h2><div class="grid">{refs}</div>
<h2>Acquired views</h2><div class="grid">{cards}</div><h2>Model calls</h2><ul>{calls}</ul></html>''')
