#!/usr/bin/env python3
"""Pin, audit and prepare a diagnosis-only addition for the existing v4 loader."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turnitover.oracle.sft_data import load_multitask_training_examples, supervision_digest
from turnitover.oracle.supervision import load_examples
from turnitover.oracle.training import tokenize_example, tokenize_prompt
from turnitover.verifier.contracts import Context, parse_decision

SOURCE = 'docs/evidence/kimi-easy-interactive-v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def prepare(args):
    import torch
    from transformers import AutoProcessor

    args.out.mkdir(parents=True, exist_ok=False)
    args.report.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', args.commit], text=True).strip()
    archive = subprocess.check_output(['git', '-C', str(args.repo), 'archive', commit, SOURCE])
    snapshot = args.out / 'snapshot'
    snapshot.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        stream.extractall(snapshot, filter='data')
    source = snapshot / SOURCE
    kimi = source / 'sft'
    manifest = json.loads((kimi / 'manifest.json').read_text())
    assert manifest['formal_training'] and manifest['status'] == 'complete'
    new = rows(kimi / 'train.diagnosis.jsonl')
    assert len(new) == 10
    by_episode = {row['episode_id']: row for row in manifest['sources']}
    semantic = json.loads((source / 'semantic_review.json').read_text())
    assert not semantic['additional_semantic_holds_on_accepted_diagnoses']
    base_train = list(load_multitask_training_examples(args.base, 'train', 'evidence'))
    validation = list(load_examples(args.base, 'validation', 'episodes', require_formal_training=True, terminal_only=True))
    identities = ('source_asset_id', 'source_identity_sha256', 'source_content_sha256', 'split_group')
    for key in identities:
        train_ids = {r['data_identity'][key] for r, _ in base_train} | {r['data_identity'][key] for r in new}
        val_ids = {r['data_identity'][key] for r, _ in validation}
        assert not train_ids & val_ids, (key, train_ids & val_ids)
    original_assets = {r['data_identity']['source_asset_id']: r['data_identity'] for r, _ in base_train}
    identity_keys = ('source_identity_sha256', 'source_content_sha256', 'source_dataset', 'source_id', 'model_id')
    for record in new:
        identity = record['data_identity']
        original = original_assets[identity['source_asset_id']]
        assert all(identity[key] == original[key] for key in identity_keys)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    roster = []
    classes = Counter()
    # Exercise the same formal Episode loader used by v4, without action rows.
    loader_root = args.out / 'kimi-diagnosis'
    loader_root.mkdir()
    shutil.copytree(kimi / 'images', loader_root / 'images')
    (loader_root / 'train.episodes.jsonl').write_bytes((kimi / 'train.diagnosis.jsonl').read_bytes())
    (loader_root / 'manifest.json').write_bytes((kimi / 'manifest.json').read_bytes())
    for record, images in load_examples(loader_root, 'train', 'episodes', require_formal_training=True, terminal_only=True):
        identity = record['data_identity']
        assert identity['split'] == 'train' and identity['formal_training']
        assert hashlib.sha256(identity['source_identity'].encode()).hexdigest() == identity['source_identity_sha256']
        assert identity['archive_sha256'] == identity['source_content_sha256']
        entry = by_episode[record['episode_id']]
        assert entry['diagnosis_accepted'] and record['episode_diagnosis_accepted']
        assert entry['source_asset_id'] == identity['source_asset_id']
        assert entry['candidate_sha'] == identity['candidate_sha'] and entry['reference_sha'] == identity['reference_sha']
        case = source / 'cases' / record['episode_id']
        assert sha(case / 'episode.json') == record['source_episode_sha256'] == entry['episode_sha256']
        assert sha(case / 'export_audit.json') == record['source_audit_sha256'] == entry['audit_sha256']
        audit = json.loads((case / 'export_audit.json').read_text())['independent_evidence_audit']
        assert audit['accepted'] and audit['evidence_sufficient']
        payload = json.loads(record['messages'][0]['content'].rsplit('\n', 1)[-1])
        target = json.loads(record['messages'][1]['content'])
        verdict = parse_decision(target, Context(**payload['context']), {r['step'] for r in payload['history']}, final_only=True)
        assert len(payload['image_order']) == len(images)
        references = [r for r in payload['image_order'] if r['kind'] == 'reference']
        assert {r['state']['detent'] for r in references} == {'rest', 'mid', 'limit'}
        for i, provenance in enumerate(record['image_provenance']):
            assert provenance['exported'] == record['images'][i]
            assert sha(case / provenance['source']) == provenance['source_sha256'] == record['image_sha256'][i]
        for finding in verdict.findings:
            assert list(finding.parts) == payload['joint_structure'][finding.joint_id]['moving_child_links']
            audited = [r for r in audit['localization'] if r['defect_id'] == finding.defect_id and r['joint_id'] == finding.joint_id]
            assert len(audited) == 1 and all(audited[0][k] for k in ('defect_type_correct', 'joint_correct', 'parts_correct'))
            assert set(finding.evidence_steps) == set(audited[0]['evidence_steps'])
        batch = tokenize_example(processor, record, images)
        _, prefix = tokenize_prompt(processor, record, images)
        length = prefix['input_ids'].shape[1]
        assert torch.all(batch['labels'][:, :length] == -100)
        assert torch.equal(batch['labels'][:, length:], batch['input_ids'][:, length:])
        assert batch['pixel_values'].numel() and batch['image_grid_thw'].shape[0] == len(images)
        category = 'clean' if verdict.status == 'pass' else verdict.findings[0].defect_id
        classes[category] += 1
        item = {k: record[k] for k in ('id', 'episode_id', 'images', 'image_sha256', 'data_identity', 'source_episode_sha256', 'source_audit_sha256')}
        item.update(category=category, reference_states=references, input_tokens=length,
                    assistant_tokens=int((batch['labels'] != -100).sum()), assistant_only_mask=True,
                    prompt_sha256=hashlib.sha256(record['messages'][0]['content'].encode()).hexdigest(),
                    target_sha256=hashlib.sha256(record['messages'][1]['content'].encode()).hexdigest())
        roster.append(item)
        print(json.dumps({k: item[k] for k in ('id', 'category', 'input_tokens', 'assistant_tokens')}), flush=True)
    assert classes == {'clean': 7, 'kinematics.joint_axis': 3}
    combined = args.out / 'experiment-data'
    combined.mkdir()
    for name in ('images',):
        shutil.copytree(args.base / name, combined / name)
    shutil.copytree(kimi / 'images', combined / 'kimi-images')
    for name in ('manifest.json', 'train.stopping.jsonl', 'validation.episodes.jsonl', 'validation.stopping.jsonl'):
        (combined / name).write_bytes((args.base / name).read_bytes())
    with (combined / 'train.episodes.jsonl').open('w') as stream:
        stream.write((args.base / 'train.episodes.jsonl').read_text().rstrip() + '\n')
        for record in new:
            record = dict(record)
            record['images'] = [r.replace('images/', 'kimi-images/', 1) for r in record['images']]
            stream.write(json.dumps(record) + '\n')
    loaded = list(load_multitask_training_examples(combined, 'train', 'evidence'))
    assert len(loaded) == len(base_train) + 10 == 549
    for original, (actual, _) in zip(new, [r for r in loaded if r[0]['condition'] == 'visible_only_interactive_teacher'], strict=True):
        assert actual['messages'] == original['messages'] and actual['image_sha256'] == original['image_sha256']
    report = dict(status='passed', source_commit=commit, training_code_base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_diagnosis_sha256=sha(kimi / 'train.diagnosis.jsonl'), source_manifest_sha256=sha(kimi / 'manifest.json'),
                  classes=dict(classes), used=10, excluded_actions=52, excluded_old_teacher_rows=117,
                  baseline_examples=len(base_train), experiment_examples=len(loaded), validation_examples=len(validation),
                  validation_assets=sorted({r['data_identity']['source_asset_id'] for r, _ in validation}),
                  no_train_validation_overlap_checked=list(identities), original_asset_identity_match_checked=list(identity_keys), baseline_supervision_sha256=supervision_digest([r for r, _ in base_train]),
                  experiment_supervision_sha256=supervision_digest([r for r, _ in loaded]),
                  baseline_input_hashes={name: sha(args.base / name) for name in ('manifest.json','train.episodes.jsonl','train.stopping.jsonl','validation.episodes.jsonl','validation.stopping.jsonl')},
                  adaptation='Only filesystem image paths are prefixed; all model-visible messages and image bytes preserved.',
                  semantic_review=semantic, rows=roster)
    write(args.report / 'data-audit.json', report)
    write(args.report / 'used-kimi-rows.json', new)
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','semantic_review')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    for key in ('repo','base','model','out','report'):
        setattr(args, key, getattr(args, key).resolve())
    prepare(args)
