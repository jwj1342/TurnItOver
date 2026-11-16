"""Qualitative figure for the paper (fig:qualitative): real repair episodes, one per row. Each row shows the reference and
candidate renderings at the observation the verifier cites, the verifier's finding, the code change of the first applied
patch, and the outcome of the independent checks, next to the outcome of the no-verifier runs of the same case.
Usage: python qualitative_figure.py <out.pdf> <episode dir>:<no-verifier episode dir>[,...] ...
"""
from __future__ import annotations

import difflib
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread

LABELS = Path(__file__).resolve().parents[3] / "tio-runs/data900/three-tier-900-sft-v1.1/annotations.private.jsonl"
NAMES = {"kinematics.joint_axis": "joint axis", "kinematics.direction_flip": "direction flip",
         "kinematics.joint_anchor": "joint anchor", "structure.part_offset": "part offset",
         "structure.part_rotation": "part rotation"}


def first_finding(episode: Path) -> tuple[dict, dict, Path, Path]:
    """The first verdict with a finding, its cited cell and the reference/candidate images of that cell."""
    for rnd in sorted(episode.glob("round-*")):
        verdict = json.loads((rnd / "verdict.json").read_text())
        if not verdict["findings"]:
            continue
        feedback = json.loads((rnd / "feedback.json").read_text())
        cell = feedback["evidence_step_cells"][0]
        obs = {(o["view"], o["state"], o["kind"]): o["image_ref"] for o in feedback["observations"]}
        return (verdict["findings"][0], cell, rnd / obs[(cell["view"], cell["state"], "reference")],
                rnd / obs[(cell["view"], cell["state"], "candidate")])
    raise SystemExit(f"{episode}: no finding")


def change(episode: Path) -> list[str]:
    result = json.loads((episode / "result.json").read_text())
    applied = [r["round"] for r in result["rounds"] if r["outcome"] == "applied"]
    before = (episode / "initial.ts").read_text().splitlines()
    after = (episode / f"round-{applied[0]:03d}" / "accepted.ts").read_text().splitlines()
    lines = [l.strip()[:58] for l in difflib.unified_diff(before, after, lineterm="", n=0)
             if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    return lines[:4]


def crop_box(images) -> tuple[int, int, int, int]:
    """Square box around the object in both renderings (pixels unlike the corner background), with a margin."""
    import numpy as np
    mask = np.zeros(images[0].shape[:2], bool)
    for img in images:
        mask |= np.abs(img[..., :3] - img[0, 0, :3]).sum(-1) > 0.05
    rows, cols = np.where(mask)
    cy, cx = (rows.min() + rows.max()) / 2, (cols.min() + cols.max()) / 2
    half = int(max(rows.max() - rows.min(), cols.max() - cols.min()) * 0.6) + 4
    h, w = mask.shape
    top, left = int(max(0, cy - half)), int(max(0, cx - half))
    return top, int(min(h, cy + half)), left, int(min(w, cx + half))


def outcome(episode: Path) -> str:
    r = json.loads((episode / "result.json").read_text())
    return "pass" if r.get("final_passed") else "fail"


def main() -> None:
    out, rows = Path(sys.argv[1]), sys.argv[2:]
    injected = {json.loads(l)["id"]: json.loads(l)["labels"] for l in LABELS.open()}
    fig, axes = plt.subplots(len(rows), 4, figsize=(7.0, 1.75 * len(rows)),
                             gridspec_kw={"width_ratios": [1, 1, 1.45, 1.75]})
    if len(rows) == 1:
        axes = [axes]
    for letter, ax_row, spec in zip("abcdef", axes, rows):
        ep, baselines = spec.split(":")
        episode = Path(ep)
        finding, cell, ref, cand = first_finding(episode)
        state = cell["state"]
        images = [imread(ref), imread(cand)]
        box = crop_box(images)
        for ax, img, title in ((ax_row[0], images[0], f"Reference ({state})"), (ax_row[1], images[1], f"Candidate ({state})")):
            ax.imshow(img[box[0]:box[1], box[2]:box[3]])
            ax.set_title(title, fontsize=7)
        ax_row[0].text(-0.08, 0.5, f"({letter})", transform=ax_row[0].transAxes, ha="right", va="center", fontsize=8, fontweight="bold")
        for ax in ax_row[:2]:
            ax.axis("off")
        parts = ", ".join(finding.get("parts") or []) or "--"
        truth = textwrap.fill("; ".join(f"{NAMES[l['defect_id']]} ({', '.join(l.get('parts', []))})" for l in injected[episode.name]), 30)
        verdict = (f"Injected: {truth}\n\nFinding: {NAMES.get(finding['defect_id'], finding['defect_id'])}\n"
                   f"part: {parts}\ncites step {finding['evidence_steps']}\n\n"
                   + textwrap.fill(finding.get("suggested_fix", ""), 30))
        ax_row[2].text(0, 1, verdict, va="top", ha="left", fontsize=6.3, family="sans-serif")
        ax_row[2].set_title("Verifier", fontsize=7)
        ax_row[2].axis("off")
        base = [outcome(Path(b)) for b in baselines.split(",")]
        code = "\n".join(change(episode)) or "(no applied patch)"
        ax_row[3].text(0, 1, code, va="top", ha="left", fontsize=5.6, family="monospace")
        ax_row[3].text(0, 0.02, f"Repair: {outcome(episode)}   |   no verifier: {', '.join(base)}",
                       va="bottom", ha="left", fontsize=6.3, fontweight="bold")
        ax_row[3].set_title("Code change and independent check", fontsize=7)
        ax_row[3].axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".jpg"), dpi=200, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
