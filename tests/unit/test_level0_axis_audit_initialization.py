"""A continuation keeps its freshly audited queue when copying a baseline."""
from contextlib import nullcontext
import json

from scripts.prepare_level0_axis_v2 import DEFAULT_OUT


def test_audit_only_initialization_preserves_new_audit_and_source_report(tmp_path, monkeypatch):
    from scripts import extend_level0_axis_v2 as expansion

    base = DEFAULT_OUT.with_name('partnet-mobility-level0-axis-v3')
    prior_audit = (base / 'private/capacity_audit.json').read_bytes()
    fresh_audit = b'{"inputs": {}, "queue": [], "round": "new expansion"}\n'
    assert fresh_audit != prior_audit
    out = tmp_path / 'continuation'
    (out / 'private').mkdir(parents=True)
    (out / 'private/capacity_audit.json').write_bytes(fresh_audit)
    fresh_report = b'Fresh expansion capacity report\n'
    (out / 'SOURCE_AUDIT.md').write_bytes(fresh_report)
    monkeypatch.setattr(expansion, 'load_pool', lambda: ([], {'groups': {}}, {}))
    monkeypatch.setattr(expansion, 'ObservationSession', lambda *args: nullcontext(object()))

    result = expansion.run(out=out, base=base, target=60)

    assert result['new_objects'] == 0
    assert (out / 'private/capacity_audit.json').read_bytes() == fresh_audit
    assert (out / 'SOURCE_AUDIT.md').read_bytes() == fresh_report
    assert (base / 'private/capacity_audit.json').read_bytes() == prior_audit
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['base_objects'] == 38
