"""Paired-state diagnosis targets qualified by counterfactual pixel exposure.

This condition grants reference renders at acquired states. It is separate from
single-reference action BC and cannot establish single-photo verifier capability.
"""
from __future__ import annotations

import json
from pathlib import Path

from turnitover.taxonomy import load_taxonomy
from turnitover.verifier.contracts import Context, parse_decision


def paired_example(annotation, context, mode, candidate_paths, reference_paths):
    trajectory = annotation["trajectories"][mode]
    history = trajectory["history"]
    indices = trajectory["witness_indices"]
    if len(history) != len(candidate_paths) or len(history) != len(reference_paths):
        raise ValueError("One matched image pair per observation required")
    findings = []
    for label_index, label in enumerate(annotation["labels"]):
        steps = [o["step"] for o, index in zip(history, indices, strict=True)
                 if label_index in annotation["exposures"][index]]
        if not steps:
            continue
        if label["defect_id"] == "kinematics.joint_axis":
            description = "The part moves along a different axis from the matched reference articulation."
            fix = "Correct the joint axis to match the reference motion."
        elif label["defect_id"] == "structure.part_offset":
            description = "The part is displaced relative to the matched reference configuration."
            fix = "Correct the part placement relative to its parent."
        else:
            raise ValueError("Unsupported evidence-qualified diagnosis")
        findings.append(dict(defect_id=label["defect_id"], parts=label["parts"], severity=label["severity"],
                             confidence=1., evidence_steps=steps, description=description, suggested_fix=fix))
    verdict = dict(status="fail" if findings else "uncertain", confidence=1. if findings else 0.,
                   findings=findings, summary="Observed deviations from paired references." if findings else "No qualified defect evidence in the acquired observations.",
                   limitations=["Coverage is limited to acquired views and states; this is not a proof of correctness.",
                                "Training findings use synthetic truth qualified by a pixel-difference proxy; confidence is not calibrated."])
    target = {"verdict": verdict}
    parse_decision(target, Context(**context), {o["step"] for o in history}, final_only=True)
    images, pairs = [], []
    for observation, index, candidate, reference in zip(history, indices, candidate_paths, reference_paths, strict=True):
        images.extend([reference, candidate])
        pairs.append({"step": observation["step"], "action": observation["action"],
                      "state": annotation["grid"][index], "reference_image": reference, "candidate_image": candidate})
    protocol = (Path(__file__).parents[1] / "verifier/prompt.txt").read_text()
    prompt = protocol + "\nThis experiment supplies matched reference/candidate image pairs at each observed camera and joint state. Image order is explicitly given below. Compare each pair. Return only a final verdict.\n"
    prompt += json.dumps({"context": context, "remaining": 0, "taxonomy_ids": load_taxonomy().ids,
                          "paired_observations": pairs, "image_order": images})
    return {"id": f"{annotation['id']}:{mode}:paired", "condition": "paired_reference_diagnosis",
            "images": images, "messages": [{"role": "user", "content": prompt},
                                            {"role": "assistant", "content": json.dumps(target)}]}
