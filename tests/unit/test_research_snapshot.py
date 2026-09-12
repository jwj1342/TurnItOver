"""Keep the shared research claims tied to the archived measurement ledger."""
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts.summarize_repair_experiment import summarize


SNAPSHOT = Path(__file__).resolve().parents[2] / 'results/2026-09-12'


def test_shared_measurement_hashes():
    manifest = json.loads((SNAPSHOT / 'snapshot.json').read_text())
    for relative, entry in manifest['files'].items():
        data = (SNAPSHOT / relative).read_bytes()
        assert len(data) == entry['bytes'], relative
        assert hashlib.sha256(data).hexdigest() == entry['sha256'], relative


def test_repair_claims_recompute_from_archived_episodes(tmp_path):
    root = tmp_path / 'repair'
    shutil.copytree(SNAPSHOT / 'output/repair-sol-pilot-v1', root)
    original = json.loads((root / 'analysis.json').read_text())
    report = summarize(root)
    assert report == original
    assert sum(c['model_calls'] for c in report['conditions'].values()) == 30
    assert report['reported_cost_usd'] == '3.4443460'
    for name, condition in report['conditions'].items():
        assert condition['damaged_cases'] == 3
        assert condition['repaired'] == (1 if name == 'fixed' else 3)
        assert condition['clean_cases'] == condition['clean_preserved'] == 1
        assert condition['cost_missing_calls'] == 0
    summary = json.loads((root / 'summary.json').read_text())
    summary['episodes'][0]['final_passed'] = True
    (root / 'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='Summary disagrees'):
        summarize(root)
