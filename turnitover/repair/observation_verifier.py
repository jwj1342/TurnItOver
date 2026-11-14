"""Inject the shared trained single-observation verifier into the repair loop."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from turnitover.oracle.observation_inference import infer_observationwise
from turnitover.oracle.supervision_validation import public_payload


def make_observation_verifier(acquire, model_call):
    """Return a run_iterative verifier_call using fresh public observations.

    acquire has the existing verifier call signature and returns (record, images)
    for the CURRENT program. model_call(single_pair_record, pair_images) is the
    same callback used by recognition. Neither callback receives private gold.
    """
    def verify(program, output, render, views, cfg, **kwargs):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        record, images = acquire(program, output, render, views, cfg, **kwargs)
        result = infer_observationwise(record, images, model_call)
        paths = {}
        for index, (name, image) in enumerate(zip(record['images'], images, strict=True)):
            target = output / f'input-{index:03d}{Path(image).suffix}'
            if Path(image).resolve() != target.resolve():
                shutil.copyfile(image, target)
            paths[name] = target.name
        trajectory = []
        for row in public_payload(record)['history']:
            row = dict(row)
            for key in ('image_ref', 'reference_image_ref'):
                if row.get(key):
                    row[key] = paths[row[key]]
            trajectory.append(row)
        result.update(status='complete', termination='observationwise_aggregation',
                      trajectory=trajectory, spent=len(result['observations']),
                      model_calls=[{'usage': row['usage'], 'observation_id': row['observation_id']}
                                   for row in result['observations']])
        (output / 'result.json').write_text(json.dumps(result, indent=2))
        return result
    return verify
