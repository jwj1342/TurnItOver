import io

import pytest
from PIL import Image

from turnitover.oracle.easy_pair_frames import easy_mid_frames


def _png(image):
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _board(root, *, altered=False):
    root.mkdir(parents=True, exist_ok=True)
    board = Image.new("RGB", (912, 630), "white")
    colors = (("red", "green", "blue"), ("yellow", "purple", "black"))
    endpoints = {}
    for row, role in enumerate(("reference", "candidate")):
        for column, state in enumerate(("rest", "mid", "limit")):
            image = Image.new("RGB", (256, 256), colors[row][column])
            board.paste(image, (114 + column * 266, 98 + row * 266))
            if state != "mid":
                endpoints[f"{role}_{state}"] = _png(image)
    if altered:
        board.putpixel((114, 98), (1, 2, 3))
    (root / "easy-case.png").write_bytes(_png(board))
    return endpoints


def test_native_midpoints_require_exact_endpoint_match(tmp_path):
    stale, correct = tmp_path / "stale", tmp_path / "correct"
    endpoints = _board(stale, altered=True)
    _board(correct)
    frames, audit = easy_mid_frames("easy-case", endpoints, [stale, stale, correct])
    assert set(frames) == {"reference_mid", "candidate_mid"}
    assert Image.open(io.BytesIO(frames["reference_mid"])).getpixel((0, 0)) == (0, 128, 0)
    assert Image.open(io.BytesIO(frames["candidate_mid"])).getpixel((0, 0)) == (128, 0, 128)
    assert audit["validated"] is True
    assert audit["crops"] == {"reference_mid": [380, 98, 636, 354],
                               "candidate_mid": [380, 364, 636, 620]}
    assert set(audit["matched_original_sha256"]) == set(endpoints)


def test_stale_or_missing_boards_cannot_supply_midpoints(tmp_path):
    endpoints = _board(tmp_path, altered=True)
    with pytest.raises(ValueError, match="No exact matching native Easy board"):
        easy_mid_frames("easy-case", endpoints, [tmp_path])
    with pytest.raises(ValueError, match="No exact matching native Easy board"):
        easy_mid_frames("easy-case", endpoints, [])


def test_board_dimensions_are_not_rescaled(tmp_path):
    endpoints = _board(tmp_path)
    path = tmp_path / "easy-case.png"
    path.write_bytes(_png(Image.open(path).resize((456, 315))))
    with pytest.raises(ValueError, match="No exact matching native Easy board"):
        easy_mid_frames("easy-case", endpoints, [tmp_path])


def test_easy_board_to_six_image_export_and_both_sft_loaders(tmp_path):
    import hashlib
    import json
    import zipfile
    from turnitover.oracle.release_sft_export import ReleaseCase, _export_cases
    from turnitover.oracle.sft_data import load_training_examples

    boards = tmp_path / 'boards'
    endpoints = _board(boards)
    assets = tmp_path / 'assets'
    assets.mkdir()
    spec = assets / 'PM_42.json'
    spec.write_text(json.dumps({'parts': [{'id': 'link_0'}], 'joints': [{'id': 'joint_0'}]}))
    source = {'source_dataset': 'sapien-sim/PartNetMobility', 'source_asset_id': '42',
              'original_spec_sha256': hashlib.sha256(spec.read_bytes()).hexdigest()}
    gold = {'case_id': 'easy-case', 'asset_id': 'PM_42', 'source': source,
            'reference_sha': 'a' * 16, 'candidate_sha': 'b' * 16,
            'defect_id': 'structure.part_offset', 'parts': ['link_0']}
    archive = tmp_path / 'easy.zip'
    roles = ['reference_rest', 'reference_limit', 'candidate_rest', 'candidate_limit']
    public = {'view': 'front', 'images': [role + '.png' for role in roles]}
    with zipfile.ZipFile(archive, 'w') as z:
        for role in roles:
            z.writestr('easy/' + role + '.png', endpoints[role])
    case = ReleaseCase('easy', archive, 'easy', gold, public)
    output = tmp_path / 'export'
    manifest = _export_cases([{'tag': 'easy-fixture'}], [case], output, 'paired-test', [assets],
                             allow_hash_split=True, easy_board_roots=[boards])
    split = manifest['groups']['sapien-sim/PartNetMobility:42']
    ordinary, evidence = [list(load_training_examples(output, split, 'diagnosis', mode))[0]
                          for mode in ('ordinary', 'evidence')]
    assert len(ordinary[1]) == len(evidence[1]) == 6
    assert ordinary[0]['messages'][0] == evidence[0]['messages'][0]
    payload = json.loads(evidence[0]['messages'][0]['content'].rsplit('\n', 1)[-1])
    assert [row['state'] for row in payload['history']] == ['zero', 'mid', 'limit']
    assert [path.rsplit('/', 1)[-1] for path in evidence[0]['images']] == [
        f'{i:02d}-{kind}_{state}.png' for i, (state, kind) in enumerate(
            (state, kind) for state in ('zero', 'mid', 'limit') for kind in ('reference', 'candidate'))]
