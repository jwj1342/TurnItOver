"""Causal action examples from privileged pixel-exposure teacher trajectories."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

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


def load_examples(root, split, kind="actions", *, require_formal_training=False, terminal_only=False):
    """Read complete exports, resolving images for downstream multimodal processors."""
    from PIL import Image
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Refusing incomplete supervision")
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown split")
    if kind not in ("actions", "diagnosis", "episodes", "stopping"):
        raise ValueError("Unknown supervision kind")
    if require_formal_training and manifest.get("formal_training") is not True:
        raise ValueError("Refusing non-formal training data")
    with (root / f"{split}.{kind}.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            if terminal_only and not record.get("terminal"):
                continue
            identity = record.get("data_identity")
            if kind in ("episodes", "stopping"):
                if not isinstance(identity, dict):
                    raise ValueError("Episode is missing data identity")
                if identity.get("formal_training") != manifest.get("formal_training"):
                    raise ValueError("Episode data purpose differs from manifest")
                for key in ("source_identity_sha256", "source_content_sha256"):
                    value = identity.get(key)
                    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                        raise ValueError(f"Episode has invalid {key}")
                for key in ("reference_sha", "candidate_sha"):
                    value = identity.get(key)
                    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{16}", value):
                        raise ValueError(f"Episode has invalid {key}")
            images = []
            image_hashes = record.get("image_sha256")
            if kind in ("episodes", "stopping") and (
                    not isinstance(image_hashes, list) or len(image_hashes) != len(record["images"])):
                raise ValueError("Supervision image hashes are missing")
            for index, relative in enumerate(record["images"]):
                path = (root / relative).resolve()
                if not path.is_relative_to(root.resolve()):
                    raise ValueError("Image outside dataset")
                if image_hashes is not None and hashlib.sha256(path.read_bytes()).hexdigest() != image_hashes[index]:
                    raise ValueError("Episode image SHA mismatch")
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
