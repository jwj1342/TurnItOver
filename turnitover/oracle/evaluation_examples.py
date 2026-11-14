"""Validation-only inference inputs, without training target reconstruction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from turnitover.oracle.supervision_validation import public_payload, validate_identity


def load_evaluation_examples(root, split='validation'):
    """Read only the requested Validation episodes and their public image files.

    Assistant targets remain unchanged for subsequent scoring. They do not
    determine input selection and are never passed to the observation builder.
    This loader does not audit other splits or open Test files.
    """
    from PIL import Image
    if split != 'validation':
        raise ValueError('Observation-wise evaluation is restricted to Validation')
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('Refusing incomplete evaluation dataset')
    name = f'{split}.episodes.jsonl'
    data = (root / name).read_bytes()
    files = manifest.get('files', {})
    if files.get(name) != hashlib.sha256(data).hexdigest():
        raise ValueError('Validation episodes manifest SHA mismatch')
    records = [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]
    if manifest.get('counts', {}).get(split) != len(records):
        raise ValueError('Validation manifest count mismatch')
    seen, episodes, provenance = set(), set(), {}
    for record in records:
        if record.get('terminal') is not True:
            raise ValueError('Expected terminal evaluation record')
        if not record.get('id') or record['id'] in seen or not record.get('episode_id') or record['episode_id'] in episodes:
            raise ValueError('Missing or duplicate evaluation record identity')
        seen.add(record['id']); episodes.add(record['episode_id'])
        identity = record.get('data_identity')
        validate_identity(identity, manifest)
        if manifest.get('groups', {}).get(identity['split_group']) != split:
            raise ValueError('Evaluation source split differs from manifest')
        source = identity['source_identity_sha256']
        # RP29 may derive multiple clean programs from the same source asset.
        # Each reference remains bound by the frozen row and image hashes.
        value = {key: identity[key] for key in ('source_content_sha256', 'source_dataset', 'asset_id', 'split_group')}
        if source in provenance and provenance[source] != value:
            raise ValueError('Inconsistent evaluation source identity')
        provenance[source] = value
        public_payload(record)
        paths, hashes = record.get('images'), record.get('image_sha256')
        if not isinstance(paths, list) or not isinstance(hashes, list) or len(paths) != len(hashes):
            raise ValueError('Evaluation image hashes are missing')
        images = []
        for relative, expected in zip(paths, hashes, strict=True):
            if not isinstance(relative, str) or Path(relative).is_absolute():
                raise ValueError('Image path must be relative')
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Image outside evaluation dataset')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected or files.get(relative) != digest:
                raise ValueError('Evaluation image SHA mismatch')
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        yield record, images
