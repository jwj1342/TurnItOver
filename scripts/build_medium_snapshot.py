"""Export active Medium media and evidence without construction programs or archives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

if __package__:
    from .easy_release_archive import sha256
    from .medium_bundle_inventory import write_manifest
    from .prepare_medium_review import prepare_review
    from .validate_medium_bundle import validate_bundle
else:
    from easy_release_archive import sha256
    from medium_bundle_inventory import write_manifest
    from prepare_medium_review import prepare_review
    from validate_medium_bundle import validate_bundle

PRESERVED = ('review_manifest.jsonl', 'private/gold.jsonl', 'private/checks.jsonl',
             'human_review.jsonl', 'private/human_spot_review.jsonl',
             'private/human_spot_dispositions.jsonl', 'train_manifest.jsonl',
             'private/train_targets.jsonl')
SOURCE_FILES = {'manifest.json': 'source/manifest.json', 'summary.json': 'source/summary.json',
                'private/verification/automatic-checks.json': 'source/automatic-checks.json',
                'private/tightening_policy.json': 'source/tightening_policy.json'}


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _safe(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '\\' in name or any(p in ('', '.', '..') for p in name.split('/')):
        raise ValueError(f'Unsafe source-relative path: {name!r}')
    return Path(*path.parts)


def export_snapshot(source: Path, output: Path, namespace: str, source_code_commit: str) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists() or output.is_symlink() or output.is_relative_to(source):
        raise ValueError('Snapshot output must be new and outside the source')
    if not re.fullmatch(r'[0-9a-f]{40}', source_code_commit):
        raise ValueError('Source code commit must be a complete Git object ID')
    manifest_path = source / 'manifest.json'
    original_manifest_sha = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    paths = list(source.rglob('*'))
    if source.is_symlink() or any(p.is_symlink() for p in paths):
        raise ValueError('Source may not contain symbolic links')
    actual = {p.relative_to(source).as_posix() for p in paths if p.is_file() and p != manifest_path}
    if actual != set(manifest['files']):
        raise ValueError('Source seal does not cover exactly its files')
    for name, digest in manifest['files'].items():
        if sha256(source / _safe(name)) != digest:
            raise ValueError(f'Source snapshot SHA mismatch: {name}')
    gold, public = _rows(source / 'private/gold.jsonl'), _rows(source / 'review_manifest.jsonl')
    if len(gold) != 150 or len({g['asset_id'] for g in gold}) != 75:
        raise ValueError('Expected the complete 75-object, 150-case Medium source')
    output.mkdir(parents=True)

    def copy(before, after=None):
        target = output / _safe(after or before)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / _safe(before), target)

    for name in PRESERVED:
        copy(name)
    for before, after in SOURCE_FILES.items():
        copy(before, after)
    for round_name, page in [('round-0001', 'source_spot_check_10.json'),
                             ('round-0002', 'source_review_page.html')]:
        for filename in ('raw.json', 'import.json', 'source_manifest.json', page):
            copy(f'private/human_review_history/{round_name}/{filename}')
    for row in public:
        cid = row['case_id']
        for name in [row['board'], row['input_preview'], *row['images']]:
            copy(name)
        for role in ('reference', 'candidate'):
            for state in ('rest', 'mid', 'limit'):
                copy(f'native/{cid}/{role}_{state}.png')
    programs = []
    for g in gold:
        path = source / _safe(g['program'])
        programs.append(dict(case_id=g['case_id'], source_path=g['program'],
            sha256=manifest['files'][g['program']], size=path.stat().st_size,
            candidate_sha=g['candidate_sha'], reference_sha=g['reference_sha']))
    _json(output / 'construction_registry.json', dict(availability='local_custody',
        source_manifest_sha256=original_manifest_sha, programs=sorted(programs, key=lambda g:g['case_id'])))
    _json(output / 'source/metadata.json', dict(version=1,
        source_manifest_sha256=original_manifest_sha, source_code_commit=source_code_commit,
        source_dirty=True, generation_policy='medium-strength-v3',
        selection_policy='medium-selection-tightening-v1', archive_availability='local_custody',
        benchmark_frozen=False, training_ready=False))
    with tempfile.TemporaryDirectory(prefix='.medium-review-', dir=output.parent) as temporary:
        review = Path(temporary) / 'review'
        mapping = prepare_review(output, review, namespace)
        shutil.copytree(review, output / 'review_bundle')
    mapping_path = output / 'private/review_mapping.jsonl'
    mapping_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in mapping), encoding='utf-8')
    (output / 'README.md').write_text('''# Medium 候选数据

75 个独立物体、150 条 Clean/缺陷配对样本；不是冻结 benchmark 或训练导出。
本包含 Gold、checker 证据及历史判断，仅供协调者和研究使用，不是独立盲审输入。
审核者只接收独立 medium_review ZIP；review_bundle 使用不带答案的匿名 ID，
与原样本的对应关系仅在本包 private/review_mapping.jsonl 中。

原始 Gold 的 program 和 source 路径属于历史来源字段，运行本包验收不会读取这些路径。
construction_registry.json 仅记录本地保管的构造程序 SHA、大小与原相对位置；
程序、mesh、URDF、旧版本媒体均未打包。本包不能独立重放几何检查或直接作为程序训练数据。
source/manifest.json 是原始来源清单，其历史绝对路径无需在下载者机器上存在。
source/automatic-checks.json 记录源数据的全量校验，便携验收不冒充重新执行源几何检查。

两轮真实人工观察 16 条：11 条 Medium、2 条 clean、3 条 too_obvious。
当前 64 条缺陷尚无人审通过记录，完整诊断模板均待填，训练导出为空。
来源按已有 Train 身份保留，与 Easy 重用来源不能作为独立留出评测；未重新划分 Validation。
第三方 PartNet-Mobility/ShapeNet 条款继续适用，私有访问不等于再分发许可。
''', encoding='utf-8')
    write_manifest(output)
    report = validate_bundle(output)
    if sha256(manifest_path) != original_manifest_sha:
        raise ValueError('Source manifest changed during export')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--namespace', required=True)
    parser.add_argument('--source-code-commit', required=True)
    args = parser.parse_args()
    print(json.dumps(export_snapshot(args.source, args.output, args.namespace,
                                    args.source_code_commit), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
