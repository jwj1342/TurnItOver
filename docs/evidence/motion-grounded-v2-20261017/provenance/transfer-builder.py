"""Matched pixel-only arms: identical prompts/images, optional comparison targets."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from PIL import Image
from scripts.build_motion_grounded import read, sha, dump

PROMPT = '''Compare reference and candidate images at the public matched states.
Each image is one camera: columns REFERENCE then CANDIDATE; rows follow public state order
(rest then actuated in training; transfer can include an intermediate state).
If only rest is observed, the image has one row. Panel step IDs are listed in public input.
Compare static position and rest-to-actuated motion before diagnosing. A displaced part can
also change the moving silhouette. Occluded or missing evidence requires uncertain.
Use legal hierarchy IDs. Return JSON with defect_type (axis/offset/clean/uncertain), joint,
part, evidence_steps. joint is null for offset; both IDs null for clean/uncertain.
Axis evidence includes rest and actuated steps; offset cites visible displaced steps;
clean cites all observed steps; uncertain cites []. You may include comparison with
static_rest (matching/different), static_steps, motion_change (matching/different/unobserved),
and motion_pairs (rest/actuated step pairs with different changes).
Public input: '''


def convert(row, source, root, variant):
    row = copy.deepcopy(row)
    public = json.loads(row['messages'][0]['content'].split('Public input: ', 1)[1])
    public.pop('rgb_comparison', None)
    panels = []
    images = []
    views = list(dict.fromkeys(s['view'] for s in public['states']))
    for view in views:
        indices = [i for i, s in enumerate(public['states']) if s['view'] == view]
        tiles = [Image.open(source / row['images'][2*i+j]).convert('RGB')
                 for i in indices for j in range(2)]
        w, h = tiles[0].size
        sheet = Image.new('RGB', (2*w, len(indices)*h), 'white')
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % 2)*w, (i // 2)*h))
        identity = json.dumps([row['image_sha256'][2*i+j] for i in indices for j in range(2)])
        relative = Path('images') / (hashlib.sha256(identity.encode()).hexdigest() + '.png')
        sheet.save(root / relative)
        images.append(str(relative))
        panels.append({'view': view, 'steps': [public['states'][i]['step'] for i in indices]})
    public['panels'] = panels
    row['images'] = images
    row['image_sha256'] = [sha(root / p) for p in images]
    row['messages'] = [{'role': 'user', 'content': PROMPT + json.dumps(public, sort_keys=True)}]
    target = {k: v for k, v in row['evidence_target'].items() if k != 'comparison'}
    if variant == 'intermediate':
        target = {'comparison': row['gold']['comparison'], **target}
    row['ordinary_target'] = row['evidence_target'] = target
    return row


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'images').mkdir()
    manifest = json.loads((a.source / 'manifest.json').read_text())
    for split in ('train', 'validation'):
        rows = read(a.source / f'{split}.jsonl')
        if a.suite:
            rows = [r for r in rows if r['condition'].startswith(('extra/', 'frozen_B/', 'frozen_C/'))]
        converted = [convert(r, a.source, a.out, a.variant) for r in rows]
        (a.out / f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in converted))
        manifest[f'{split}_rows'] = len(converted)
    manifest.update(variant=a.variant, representation='pixel-only aligned panels',
                    source=str(a.source), source_manifest_sha256=sha(a.source/'manifest.json'),
                    builder_sha256=sha(__file__), identical_public_prompt=True)
    dump(a.out / 'manifest.json', manifest)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--variant', choices=['direct', 'intermediate'], required=True)
    p.add_argument('--suite', action='store_true')
    main(p.parse_args())
