import json

from PIL import Image

from scripts.tier_review_intake import easy_intake, hard_intake


def test_hard_intake_excludes_baseline_assets(tmp_path):
    bundle = tmp_path / 'hard'
    baseline = tmp_path / 'baseline'
    native = bundle / 'native'
    objects = baseline / 'private/objects'
    objects.mkdir(parents=True)
    bundle.mkdir()
    (objects / 'asset-base').mkdir()
    (objects / 'asset-base/record.json').write_text('{}')
    (bundle / 'private').mkdir()
    (bundle / 'private/gold.jsonl').write_text(
        json.dumps({'case_id': 'hp-baseline', 'asset_id': 'asset-base'}) + '\n'
        + json.dumps({'case_id': 'hp-new', 'asset_id': 'asset-new'}) + '\n')
    rows = []
    for cid in ('hp-baseline', 'hp-new'):
        rows.append({'case_id': cid, 'native_root': 'native', 'initial_view': 'v0',
                     'supplementary_view': 'v1', 'states': ['rest', 'mid', 'limit']})
        for view in ('v0', 'v1'):
            directory = native / view
            directory.mkdir(parents=True, exist_ok=True)
            for role in ('reference', 'candidate'):
                for state in ('rest', 'mid', 'limit'):
                    Image.new('RGB', (8, 8), 'white').save(directory / f'{role}_{state}.png')
    (bundle / 'review_manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))

    intake = hard_intake(bundle, baseline)

    assert [row['source_case_id'] for row in intake] == ['hp-new']
    assert len(intake[0]['primary']) == 6
    assert len(intake[0]['supplementary']) == 6


def test_easy_intake_extracts_mid_tiles_without_identifiers(tmp_path):
    bundle = tmp_path / 'easy'
    (bundle / 'frames/case-1').mkdir(parents=True)
    for index in range(4):
        Image.new('RGB', (8, 8), 'white').save(bundle / f'frames/case-1/{index}.png')
    board_path = bundle / 'board.png'
    Image.new('RGB', (820, 650), 'white').save(board_path)
    (bundle / 'review_manifest.jsonl').write_text(json.dumps(
        {'case_id': 'case-1', 'board': 'board.png'}) + '\n')

    intake = easy_intake(bundle, tmp_path / 'extracted')

    assert intake[0]['source_case_id'] == 'case-1'
    assert len(intake[0]['primary']) == 4
    assert len(intake[0]['supplementary']) == 2
