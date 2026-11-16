"""Project v1.1 validation episodes to the Table 2 observation conditions."""
import copy
import json
import re
from pathlib import Path

FRAME = re.compile(r"\d+-(?:(.+)_)?(reference|candidate)_(rest|mid|limit)\.png")


def project(record, images, condition):
    prefix, raw = record["messages"][0]["content"].rsplit("\n", 1)
    payload = json.loads(raw)
    frames = [FRAME.fullmatch(Path(name).name) for name in record["images"]]
    if not all(frames):
        raise ValueError("Unrecognized frozen frame name")
    metadata = [frame.groups() for frame in frames]
    if condition == "viewpoint":
        if len({v for v, _, _ in metadata}) < 2:
            return None
        keep = [i for i, (_, _, state) in enumerate(metadata) if state == "rest"]
    elif condition == "motion":
        keep = [i for i, (view, _, _) in enumerate(metadata) if view == metadata[0][0]]
    else:
        raise ValueError(condition)
    if not keep or keep[0] != 0:
        raise ValueError("The initial reference frame must be kept")
    positions = {name: index for index, name in enumerate(record["images"])}
    new_positions = {old: new for new, old in enumerate(keep)}
    history = []
    steps = {}
    for row in payload["history"]:
        index = positions[row["image_ref"]]
        if index in new_positions:
            steps[row["step"]] = len(history)
            history.append({**row, "step": len(history)})
    result = copy.deepcopy(record)
    result.pop("pass_certificate", None)
    for field in ("images", "image_sha256"):
        result[field] = [record[field][i] for i in keep]
    payload.update(history=history, image_order=result["images"])
    result["messages"][0]["content"] = prefix + "\n" + json.dumps(payload, separators=(",", ":"))
    target = json.loads(record["messages"][1]["content"])
    verdict = target["verdict"]
    findings = []
    for finding in verdict["findings"]:
        evidence = [steps[step] for step in finding["evidence_steps"] if step in steps]
        if evidence:
            findings.append({**finding, "evidence_steps": evidence})
    verdict["findings"] = findings
    if not findings:
        # A parser-compatible empty finding set. Table 2 scores only defect F1.
        verdict.update(status="uncertain", summary="No labeled defect is exposed in this subset.",
                       limitations=["Only the selected observations are assessed."])
    result["messages"][1]["content"] = json.dumps(target)
    return result, [images[i] for i in keep]

