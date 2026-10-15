"""Auditable mixing for v4 diagnosis replay and v5 visual stopping."""
from __future__ import annotations

from collections import Counter
import copy
import random

from turnitover.oracle.sft_data import load_terminal_training_examples
from turnitover.oracle.supervision import load_examples


DIAGNOSIS_PROTOCOL = "v4_single_reference_terminal_diagnosis"
STOPPING_PROTOCOL = "v5_eight_reference_prefix_stopping"


def load_protocol_examples(v4_root, v5_root, split="train"):
    """Keep each task's original model-visible prompt and image protocol intact."""
    diagnosis = []
    for record, images in load_terminal_training_examples(v4_root, split, "evidence"):
        record = copy.deepcopy(record)
        record.update(sft_objective="diagnosis", input_protocol=DIAGNOSIS_PROTOCOL)
        diagnosis.append((record, images))
    stopping = []
    for record, images in load_examples(
            v5_root, split, "stopping", require_formal_training=True):
        record = copy.deepcopy(record)
        record.update(sft_objective="stopping", input_protocol=STOPPING_PROTOCOL)
        stopping.append((record, images))
    if not diagnosis or not stopping:
        raise ValueError("Both diagnosis replay and stopping examples are required")
    return diagnosis, stopping


def stopping_class_weights(records) -> dict[str, float]:
    counts = Counter(record["stop_target"] for record, _ in records)
    if set(counts) != {"continue", "stop"}:
        raise ValueError(f"Both stopping labels are required: {counts}")
    return {label: len(records) / (len(counts) * count) for label, count in counts.items()}


def replay_groups(n_diagnosis: int, n_stopping: int, *, seed: int,
                  stopping_per_diagnosis: int = 4) -> list[dict]:
    """Use every stopping row once and cycle diagnosis rows between optimizer steps."""
    if n_diagnosis <= 0 or n_stopping <= 0 or stopping_per_diagnosis <= 0:
        raise ValueError("Positive example counts and mixing ratio are required")
    stopping = list(range(n_stopping))
    random.Random(seed).shuffle(stopping)
    groups = []
    diagnosis_pool: list[int] = []
    diagnosis_round = 0
    for start in range(0, n_stopping, stopping_per_diagnosis):
        if not diagnosis_pool:
            diagnosis_pool = list(range(n_diagnosis))
            random.Random(seed + 10_000 + diagnosis_round).shuffle(diagnosis_pool)
            diagnosis_round += 1
        groups.append({"diagnosis": diagnosis_pool.pop(),
                       "stopping": stopping[start:start + stopping_per_diagnosis]})
    return groups
