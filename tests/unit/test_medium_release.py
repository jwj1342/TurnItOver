"""Private release round trips and failure boundaries, using synthetic media only."""
import json
from pathlib import Path

import pytest

from scripts.build_medium_release import build_release
from scripts.easy_release_archive import sha256
from scripts.validate_medium_release import validate_release
from tests.unit.medium_bundle_fixture import build_bundle


@pytest.fixture
def released(tmp_path):
    source = build_bundle(tmp_path / 'synthetic-source')
    assets = tmp_path / 'assets'
    contract = build_release(source, assets, 'medium-candidate-v0.1.0-rc.1', '1' * 40)
    pinned = tmp_path / 'pinned.json'
    pinned.write_bytes((assets / 'release-manifest.json').read_bytes())
    return source, assets, pinned, contract


def test_complete_release_is_reproducible_and_portable(released, tmp_path):
    source, assets, pinned, contract = released
    second = tmp_path / 'second'
    build_release(source, second, contract['tag'], '1' * 40)
    assert {p.name: sha256(p) for p in assets.iterdir()} == {p.name: sha256(p) for p in second.iterdir()}
    report = validate_release(assets, pinned, tmp_path / 'accepted')
    assert report['dataset']['cases'] == 150
    assert report['dataset']['training_eligible_cases'] == 0
    assert (tmp_path / 'accepted/medium_review/review.html').is_file()
    assert not (tmp_path / 'accepted/medium/private/programs').exists()


def test_changed_zip_is_rejected_before_extraction(released, tmp_path):
    _, assets, pinned, contract = released
    path = assets / next(iter(contract['assets']))
    path.write_bytes(path.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='size or SHA-256 mismatch'):
        validate_release(assets, pinned, tmp_path / 'not-published')
    assert not (tmp_path / 'not-published').exists()


def test_wrong_pinned_report_does_not_publish_partial_output(released, tmp_path):
    _, assets, pinned, contract = released
    contract['expected_report']['cases'] = 149
    encoded = json.dumps(contract, sort_keys=True).encode()
    pinned.write_bytes(encoded)
    (assets / 'release-manifest.json').write_bytes(encoded)
    names = sorted([*contract['assets'], 'release-manifest.json'])
    (assets / 'SHA256SUMS').write_text(''.join(f'{sha256(assets / n)}  {n}\n' for n in names))
    with pytest.raises(ValueError, match='report differs'):
        validate_release(assets, pinned, tmp_path / 'not-published')
    assert not (tmp_path / 'not-published').exists()
