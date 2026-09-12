"""Counterfactual rendering measures pixel exposure, not a VLM's ability to recognize it."""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from turnitover.core.actions import detent_value
from turnitover.core.program import ObjectProgram


@dataclass(frozen=True)
class ExposureThresholds:
    image_l1: float = .001
    changed_fraction: float = .001
    pixel_delta: float = .02


def pixel_signal(a, b, thresholds=ExposureThresholds()):
    delta = np.abs(a.astype(np.float32)-b.astype(np.float32)) / 255.
    mean = float(delta.mean())
    fraction = float((delta.max(axis=2) > thresholds.pixel_delta).mean())
    return {"image_l1": mean, "changed_fraction": fraction,
            "exposed": mean >= thresholds.image_l1 and fraction >= thresholds.changed_fraction}


def pixels(png):
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


def replay_corruptions(reference, labels, omit=None):
    spec = reference.spec
    for index, label in enumerate(labels):
        if index == omit:
            continue
        if label.defect_id == "structure.part_offset":
            part = spec.part(label.parts[0])
            spec = spec.replace_part(part.id, origin=tuple(x+y for x, y in zip(part.origin, label.params["vector"], strict=True)))
        elif label.defect_id == "kinematics.joint_axis":
            spec = spec.replace_joint(label.params["joint"], axis=tuple(label.label["axis"]))
        else:
            raise ValueError(f"No audited counterfactual replay for {label.defect_id}")
    return ObjectProgram.from_spec(spec)


def render_grid(session, program, framing, grid):
    info = session.load(program, framing=framing)
    rows = []
    try:
        for witness in grid:
            session.reset_joints()
            if witness.joint is not None:
                value = detent_value(witness.detent, tuple(info.joints[witness.joint]["limits"]))
                session.set_joints({witness.joint: value})
            rows.append(session.request_view(witness.view).png)
        return rows
    finally:
        session.dispose()


def exposure_matrix(candidate, reference, counterfactuals, thresholds):
    cand, ref = [pixels(x) for x in candidate], [pixels(x) for x in reference]
    reference_signals = [pixel_signal(a, b, thresholds) for a, b in zip(cand, ref, strict=True)]
    signals = []
    for repaired in counterfactuals:
        signals.append([pixel_signal(a, pixels(b), thresholds) for a, b in zip(cand, repaired, strict=True)])
    exposure = [tuple(i for i, rows in enumerate(signals) if rows[j]["exposed"] and reference_signals[j]["exposed"])
                for j in range(len(candidate))]
    return exposure, {"against_reference": reference_signals, "leave_one_corruption_out": signals}
