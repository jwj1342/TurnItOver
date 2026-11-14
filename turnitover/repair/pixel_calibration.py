"""Fit the Table 1 pixel threshold using hash-bound Train observations only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from turnitover.repair.frozen_manifest import _file
from turnitover.repair.table1_judges import ObservationPair, _pixel_difference


def calibrate_pixel_threshold(manifest_path: Path) -> dict:
    """Maximize balanced accuracy; choose the smallest threshold on ties.

    The manifest declares schema_version=1, status=frozen and split=train.
    Every case has a unique id, boolean defective and three pairs ordered
    zero/mid/limit. Each pair contains reference/candidate relative paths and
    sha256={reference: ..., candidate: ...}. Declared split provenance must
    be independently checked before freezing; this loader checks hashes.
    """
    manifest_path = Path(manifest_path).resolve()
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if (not isinstance(manifest, dict) or manifest.get('schema_version') != 1
            or manifest.get('status') != 'frozen' or manifest.get('split') != 'train'):
        raise ValueError('Expected a frozen schema_version 1 Train calibration manifest')
    cases = manifest.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('Train calibration cases required')
    samples, seen = [], set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError('Calibration case must be an object')
        case_id = case.get('id')
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError('Unique nonempty calibration case ids required')
        seen.add(case_id)
        if type(case.get('defective')) is not bool:
            raise ValueError('Calibration defective label must be boolean')
        pairs = case.get('pairs')
        if (not isinstance(pairs, list) or len(pairs) != 3
                or any(not isinstance(pair, dict) for pair in pairs)
                or [pair.get('state') for pair in pairs] != ['zero', 'mid', 'limit']):
            raise ValueError('Calibration needs ordered zero/mid/limit pairs')
        scores = []
        for pair in pairs:
            hashes = pair.get('sha256')
            if not isinstance(hashes, dict):
                raise ValueError('Calibration image hashes required')
            paths = {kind: _file(manifest_path.parent, pair.get(kind), hashes.get(kind), kind)
                     for kind in ('reference', 'candidate')}
            scores.append(_pixel_difference(ObservationPair(pair['state'], **paths)))
        samples.append((max(scores), case['defective']))
    positives = sum(label for _, label in samples)
    negatives = len(samples) - positives
    if not positives or not negatives:
        raise ValueError('Balanced accuracy requires both clean and defective Train cases')
    thresholds = sorted({0.0, 1.0, *(score for score, _ in samples)})
    best = None
    for threshold in thresholds:
        tp = sum(score > threshold and label for score, label in samples)
        tn = sum(score <= threshold and not label for score, label in samples)
        numerator = tp * negatives + tn * positives
        if best is None or numerator > best[0]:
            best = (numerator, threshold, tp, tn)
    numerator, threshold, tp, tn = best
    return {'schema_version': 1, 'status': 'frozen', 'split': 'train',
            'metric': 'mean_absolute_rgb_0_1', 'aggregation': 'max_zero_mid_limit',
            'training_manifest_sha256': hashlib.sha256(raw).hexdigest(),
            'threshold': threshold, 'decision_rule': 'defective iff score > threshold',
            'selection': 'maximum_balanced_accuracy', 'tie_break': 'smallest_threshold',
            'balanced_accuracy': numerator / (2 * positives * negatives),
            'counts': {'cases': len(samples), 'defective': positives, 'clean': negatives,
                       'true_positive': tp, 'true_negative': tn}}
