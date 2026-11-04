"""Extract private baseline evidence into isolated custody and build full blind-review intake."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(zip_path, output, tier):
    if output.exists():
        raise ValueError('Extraction output must be new')
    output.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        required = [tier + '/review_manifest.jsonl', tier + '/private/gold.jsonl']
        if not all(name in names for name in required):
            raise ValueError('Missing tier ledgers')
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            if name.is_absolute() or '..' in name.parts or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError('Unsafe ZIP member')
            allowed = (member.filename in required or any(member.filename.startswith(tier + '/' + p + '/')
                       for p in ('frames', 'native', 'boards')))
            if not allowed or member.is_dir():
                continue
            target = output.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst)
    root = output / tier
    public = [json.loads(line) for line in (root / 'review_manifest.jsonl').read_text().splitlines() if line.strip()]
    gold = [json.loads(line) for line in (root / 'private/gold.jsonl').read_text().splitlines() if line.strip()]
    if {r['case_id'] for r in public} != {r['case_id'] for r in gold}:
        raise ValueError('Gold/public case coverage mismatch')
    intake = []
    for row in public:
        cid = row['case_id']
        native = root / 'native' / cid
        if native.exists():
            primary = [{'path': str((native / f'{role}_{state}.png').resolve()),
                        'caption': f'{label} / 初始视角 / {state}'}
                       for role, label in (('reference', '参考'), ('candidate', '候选'))
                       for state in ('rest', 'mid', 'limit')]
        else:
            states = [('参考','rest'),('参考','limit'),('候选','rest'),('候选','limit')]
            primary = [{'path': str((root / name).resolve()),'caption':f'{role} / 初始视角 / {state}'}
                       for name,(role,state) in zip(row['images'],states,strict=True)]
        supplementary = []
        if row.get('board'):
            supplementary.append({'path':str((root/row['board']).resolve()),'caption':'三态辅助对比'})
        intake.append(dict(source_case_id=cid,bundle=str(root.resolve()),primary=primary,supplementary=supplementary))
    (output/'intake.json').write_text(json.dumps(intake,ensure_ascii=False,indent=2)+'\n')
    (output/'extraction-audit.json').write_text(json.dumps(dict(source_zip=str(zip_path.resolve()),
        source_zip_sha256=digest(zip_path),tier=tier,cases=len(public),independent_objects=len({r['asset_id'] for r in gold}),
        status='pending_human_review',media_files={p.relative_to(output).as_posix():digest(p)
            for p in output.rglob('*.png')}),indent=2)+'\n')
    return dict(tier=tier,cases=len(public),intake=str((output/'intake.json').resolve()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--tier',choices=('easy','medium'),required=True)
    args=parser.parse_args()
    print(json.dumps(extract(args.zip,args.out,args.tier)))
