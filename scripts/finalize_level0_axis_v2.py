"""Validate a Level 0 v2 bundle and export only genuinely reviewed object pairs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.prepare_obvious_defect_review import ROOT, sha256_path, write_jsonl
from turnitover.assets.catalog import CatalogSource, decode_spec
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.core.serde import to_dict


def rows(path):
    result = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({r['case_id'] for r in result}) != len(result):
        raise ValueError(f'Duplicate case IDs: {path}')
    return result


def accepted(review, gold, inherited=False):
    j = review['human_judgment']
    complete = (review['status'] == 'human_review_complete' and
        j.get('verdict') == gold['expected_status'] and j.get('defect_id') == gold['defect_id'] and
        j.get('one_glance') is True and set(j.get('evidence_states', [])) == {'mid', 'limit'} and
        isinstance(j.get('normal_vs_candidate_motion'), str) and bool(j['normal_vs_candidate_motion'].strip()))
    if not inherited:
        complete = (complete and isinstance(review.get('reviewer'), str) and bool(review['reviewer'].strip()) and
            isinstance(review.get('reviewed_at'), str) and bool(review['reviewed_at'].strip()) and
            j.get('no_clipping_or_occlusion') is True and j.get('rest_matches_reference') is True)
    return bool(complete)


def validate_specs(gold, catalog, out):
    for g in gold:
        ref = ObjectProgram.from_spec(catalog.get(g['asset_id']))
        candidate = ObjectProgram.from_spec(decode_spec(json.loads((out / 'private' / f"{g['case_id']}.json").read_text())))
        if (ref.sha, candidate.sha) != (g['reference_sha'], g['candidate_sha']):
            raise ValueError('Program SHA mismatch')
        if g['kind'] == 'clean':
            if ref.sha != candidate.sha or g['defect_id'] is not None:
                raise ValueError('Clean program modified')
            continue
        if g['kind'] != 'joint_axis' or g['defect_id'] != 'kinematics.joint_axis':
            raise ValueError('Only joint_axis is allowed')
        expected, actual = to_dict(ref.spec), to_dict(candidate.spec)
        joint_id = g['joint_id']
        before = next(j for j in expected['joints'] if j['id'] == joint_id)
        after = next(j for j in actual['joints'] if j['id'] == joint_id)
        a, b = np.array(before['axis']), np.array(after['axis'])
        if abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))) > 1e-6:
            raise ValueError('Axis must change by exactly 90 degrees')
        after['axis'] = before['axis']
        if actual != expected:
            raise ValueError('Modification beyond a single joint axis')


def finalize(out: Path):
    from scripts.prepare_level0_axis_v2 import seal
    manifest = json.loads((out / 'manifest.json').read_text())
    # Human review is the only editable input between generation and acceptance.
    for name, digest in manifest['files'].items():
        if name != 'human_review.jsonl' and sha256_path(out / name) != digest:
            raise ValueError(f'Bundle SHA mismatch: {name}')
    for name, digest in manifest['inputs'].items():
        if sha256_path(Path(name)) != digest:
            raise ValueError(f'Generation input SHA mismatch: {name}')
    public = rows(out / 'review_manifest.jsonl')
    gold = rows(out / 'private/gold.jsonl')
    reviews = {r['case_id']: r for r in rows(out / 'human_review.jsonl')}
    checks = {r['case_id']: r for r in rows(out / 'private/checks.jsonl')}
    ids = {r['case_id'] for r in public}
    if not ids == {r['case_id'] for r in gold} == set(reviews) == set(checks):
        raise ValueError('Case ID sets differ')
    config_path = next(Path(p) for p in manifest['inputs'] if p.endswith('.yaml') and 'generate_' in p)
    splits_path = next(Path(p) for p in manifest['inputs'] if p.endswith('/splits.json'))
    catalog = CatalogSource(load_generate_config(config_path).asset_source['manifest'])
    splits = json.loads(splits_path.read_text())
    validate_specs(gold, catalog, out)
    provenance = json.loads((out / 'private/inherited_review_provenance.json').read_text())
    for case_id, source in provenance.items():
        source_dir = ROOT / source['source']
        if sha256_path(source_dir / 'human_review.jsonl') != source['review_file_sha256']:
            raise ValueError('Inherited review provenance changed')
        original = {r['case_id']: r for r in rows(source_dir / 'human_review.jsonl')}
        if reviews[case_id] != original[case_id]:
            raise ValueError('Inherited judgment must remain verbatim')
    groups = defaultdict(list)
    for g in gold:
        meta = catalog.metadata(g['asset_id'])
        if splits['groups'][meta['split_group']] != 'train':
            raise ValueError('Non-Train object')
        c = checks[g['case_id']]
        if (c['code_correct'] != (g['kind'] == 'clean') or not c['checks'][1]['passed'] or
            c['candidate_sha'] != g['candidate_sha'] or c['reference_sha'] != g['reference_sha']):
            raise ValueError('Checker disagreement')
        groups[g['asset_id']].append(g)
    for group in groups.values():
        if len(group) != 2 or {g['kind'] for g in group} != {'clean', 'joint_axis'}:
            raise ValueError('Each object must have exactly one clean/axis pair')
    for p in public:
        if {'kind', 'defect_id', 'parameters', 'expected_status', 'asset_id', 'joint_id'} & p.keys():
            raise ValueError('Private answer fields in public input')
        if Image.open(out / p['input_preview']).size != (542, 542) or Image.open(out / p['board']).size != (912, 630):
            raise ValueError('Unexpected image dimensions')
    eligible = {asset for asset, group in groups.items()
        if all(accepted(reviews[g['case_id']], g, g['case_id'] in provenance) for g in group)}
    exported, targets, object_splits = [], [], []
    by_id = {p['case_id']: p for p in public}
    for asset, group in sorted(groups.items()):
        object_splits.append({'asset_id': asset, 'split_group': catalog.metadata(asset)['split_group'],
            'split': 'train', 'case_ids': sorted(g['case_id'] for g in group),
            'training_eligible': asset in eligible})
        for g in group:
            case_id = g['case_id']
            g['training_eligibility'] = 'eligible' if asset in eligible else 'pending_or_rejected_human_review'
            by_id[case_id]['status'] = reviews[case_id]['status']
            # Four ordered 256px tiles are copied losslessly from the actual reviewed input.
            image_paths = []
            preview = Image.open(out / by_id[case_id]['input_preview'])
            for tile, (x, y) in enumerate(((10, 10), (276, 10), (10, 276), (276, 276))):
                rel = f'frames/{case_id}/{tile}.png'
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                preview.crop((x, y, x + 256, y + 256)).save(out / rel)
                image_paths.append(rel)
            by_id[case_id]['images'] = image_paths
            if asset in eligible:
                exported.append({'case_id': case_id, 'images': image_paths,
                    'input_preview': by_id[case_id]['input_preview'],
                    'prompt': 'Images: reference rest, reference limit, candidate rest, candidate limit. '
                              'Does the candidate follow the reference motion? Return pass or fail and describe the motion.'})
                targets.append({'case_id': case_id, 'expected_status': g['expected_status'],
                    'defect_id': g['defect_id'], 'motion_description': reviews[case_id]['human_judgment']['normal_vs_candidate_motion']})
    write_jsonl(out / 'review_manifest.jsonl', public)
    write_jsonl(out / 'private/gold.jsonl', gold)
    write_jsonl(out / 'train_manifest.jsonl', sorted(exported, key=lambda r: r['case_id']))
    write_jsonl(out / 'private/train_targets.jsonl', sorted(targets, key=lambda r: r['case_id']))
    (out / 'object_splits.json').write_text(json.dumps({'unit': 'independent_source_object', 'objects': object_splits}, indent=2))
    audit = json.loads((out / 'private/screening.json').read_text())['assets']
    rejected = [a for a in audit if a['status'] == 'rejected']
    validation = {'status': 'automatic_checks_passed', 'acceptance': 'complete' if len(eligible) >= 20 else 'target_not_met',
        'screened_train_objects': len(audit), 'automatic_pass_objects': len(groups), 'automatic_pass_cases': len(gold),
        'clean_cases': len(groups), 'axis_cases': len(groups), 'training_eligible_objects': len(eligible),
        'training_eligible_cases': len(exported), 'training_clean_cases': len(eligible), 'training_axis_cases': len(eligible),
        'human_review_complete': sum(r['status'] == 'human_review_complete' for r in reviews.values()),
        'pending_human_review': sum(r['status'] == 'pending_human_review' for r in reviews.values()),
        'rejected_objects': [{'asset_id': a['asset_id'], 'reasons': a['rejection_reason_counts']} for a in rejected],
        'clean_checker_passed': sum(c['code_correct'] for c in checks.values()),
        'axis_checker_rejected': sum(not c['code_correct'] for c in checks.values()),
        'axis_angle_deg': 90, 'native_image_size': [256, 256], 'part_offset_cases': 0,
        'validation_test_rows_touched': 0, 'model_calls': 0, 'sft_training_runs': 0}
    (out / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    (out / 'README.md').write_text(
        '# PartNet-Mobility Level 0 v2\n\n'
        f'筛选 {len(audit)} 个现有 Train 物体；自动通过 {len(groups)} 个物体、{len(gold)} 个样本'
        f'（clean/axis 各 {len(groups)}）。训练合格 {len(eligible)} 个物体、{len(exported)} 个样本；'
        f'待人工审核 {validation["pending_human_review"]} 个样本。目标为至少 20 个物体、40 个样本，'
        f'当前状态：{validation["acceptance"]}。\n\n'
        'train_manifest.jsonl 是唯一训练输入清单；图像路径相对于本目录。'
        'private/train_targets.jsonl 由 case_id 连接监督答案，勿作为普通模型输入。'
        'object_splits.json 按独立源物体记录 Train 分组和成对资格。'
        'review_manifest.jsonl 包括待审核样本，不能直接作为训练清单。\n\n'
        '四图顺序：参考 rest、参考 limit、候选 rest、候选 limit；frames/ 为原生 256×256 PNG，'
        'inputs/ 为同图四宫格，boards/ 额外展示 mid。private/ 保存 Gold、程序、checker、'
        '逐关节/视角/seed 筛选诊断及旧审核记录来源。SHA 清单见 manifest.json。\n\n'
        '新增样本仅能由真实人工逐图审核后填写 human_review.jsonl：status=human_review_complete，'
        'reviewer、reviewed_at、human_judgment.verdict、defect_id、one_glance、'
        'normal_vs_candidate_motion、evidence_states=[mid,limit]、no_clipping_or_occlusion、'
        'rest_matches_reference。请先盲审图像，不打开 private/。看不清、遮挡、出框或不明确时，'
        'status=human_review_rejected，并记录 notes；默认批准和自动筛选均不能替代逐图审核。'
        '只有同物体 clean/axis 两行均满足审核门禁才可导出。\n\n'
        '复现（仓库根目录、全新输出路径）：\n\n```bash\n'
        'python -m scripts.prepare_level0_axis_v2 --out /tmp/level0-axis-v2-repro\n'
        '# 人工记录为独立输入，确认图像/程序复现一致后，沿用已完成的真实审核：\n'
        'cp docs/evidence/partnet-mobility-level0-axis-v2/human_review.jsonl /tmp/level0-axis-v2-repro/human_review.jsonl\n'
        'python -m scripts.finalize_level0_axis_v2 --out /tmp/level0-axis-v2-repro\n'
        'diff -rq docs/evidence/partnet-mobility-level0-axis-v2 /tmp/level0-axis-v2-repro\n'
        '```\n\n人工填写审核文件后执行：\n\n```bash\n'
        'python -m scripts.finalize_level0_axis_v2 --out docs/evidence/partnet-mobility-level0-axis-v2\n'
        '```\n\n仅 human_review.jsonl 允许审核时编辑，验收先校验其余文件及源输入 SHA，'
        '再校验程序只改变一个关节的 90° 轴、checker、Train 划分、成对审核状态，最后更新 SHA。'
        '包含人工修改后，复现渲染应比较图像和程序；人工审核记录须单独保留，不能由生成器重建。\n',
        encoding='utf-8')
    seal(out, manifest['inputs'])
    return validation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.out.resolve())))
