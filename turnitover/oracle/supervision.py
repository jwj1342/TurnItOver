"""Causal action examples from privileged pixel-exposure teacher trajectories."""
from __future__ import annotations

import json
from pathlib import Path

from turnitover.core.actions import ActuateJoint, RequestView
from turnitover.verifier.contracts import Context, parse_decision
from turnitover.taxonomy import load_taxonomy


def action_json(action):
    if isinstance(action, RequestView):
        return {"type": "request_view", "view_id": action.view_id}
    if isinstance(action, ActuateJoint):
        return {"type": "actuate_joint", "joint_id": action.joint_id, "detent": action.detent.value}
    raise TypeError(action)


def action_examples(uid, context, reference, history, budget):
    """Only preceding observations are exposed; teacher metadata stays outside messages."""
    protocol = (Path(__file__).parents[1] / "verifier/prompt.txt").read_text()
    for step, observation in enumerate(history):
        prefix = history[:step]
        target = {"action": observation["action"]}
        parse_decision(target, Context(**context), {o["step"] for o in prefix})
        images = [reference] + [o["image_ref"] for o in prefix]
        payload = {"context": context, "remaining": budget-step, "history": prefix,
                   "image_order": images, "taxonomy_ids": load_taxonomy().ids}
        yield {"id": f"{uid}:action:{step}", "condition": "single_reference_action_bc",
               "images": images, "messages": [
                   {"role": "user", "content": protocol + "\n" + json.dumps(payload)},
                   {"role": "assistant", "content": json.dumps(target)}]}


def load_examples(root, split, kind="actions"):
    """Read complete exports, resolving images for downstream multimodal processors."""
    from PIL import Image
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Refusing incomplete supervision")
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown split")
    if kind not in ("actions", "diagnosis"):
        raise ValueError("Unknown supervision kind")
    with (root / f"{split}.{kind}.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            images = []
            for relative in record["images"]:
                path = (root / relative).resolve()
                if not path.is_relative_to(root.resolve()):
                    raise ValueError("Image outside dataset")
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            yield record, images


def load_action_groups(root, split):
    """Keep all weighted variants together when forming a training batch."""
    group, current = [], None
    seen = set()
    for record, images in load_examples(root, split):
        if record["condition"] != "single_reference_empirical_action_bc":
            raise ValueError("Grouped loader requires the empirical distribution export")
        key = record["id"].rsplit(":", 1)[0]
        if current is not None and key != current:
            _validate_group(group)
            yield group
            group = []
            seen.add(current)
        if key in seen:
            raise ValueError("Group variants must be contiguous")
        current = key
        group.append((record, images))
    if group:
        _validate_group(group)
        yield group


def _validate_group(group):
    import math
    weights = [record["loss_weight"] for record, _ in group]
    if not all(math.isfinite(w) and w > 0 for w in weights) or not math.isclose(sum(weights), 1.):
        raise ValueError("Group weights must be positive and sum to one")
    first = group[0][0]
    if any(record["messages"][0] != first["messages"][0] or record["images"] != first["images"]
           for record, _ in group):
        raise ValueError("Group variants must have identical public inputs")
