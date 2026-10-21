"""Real Chromium history + scripted API transport failure/recovery, not model results."""
import json
import shutil
from pathlib import Path

import pytest

from tests.browser.test_api_pipeline_roundtrip import ScriptedAPIFixture
from turnitover.api_pipeline.fixture import TASK, controlled_cases, reference_program
from turnitover.api_pipeline.paired import prepare, run_pairs
from turnitover.api_pipeline.recovery import recover_final
from turnitover.api_pipeline.scoring import score_pairs
from turnitover.models.client import ModelError, request_payload

pytestmark = pytest.mark.browser


class FailingFinalFixture(ScriptedAPIFixture):
    def scoped(self, label):
        normal = super().scoped(label)
        def call(prompt, images):
            if label == 'c01/active/judge' and self.verdicts[label] == 1:
                data = json.loads(prompt.split('Observed task data:\n')[1].split('\n')[0])
                if data['remaining'] == 0:
                    directory = self.ledger_root / 'api_calls/failed'
                    directory.mkdir(parents=True)
                    (directory / 'prompt.txt').write_text(prompt)
                    _, _, payload = request_payload(self.config, prompt, images)
                    payload['temperature'] = self.public_config['temperature']
                    (directory / 'request.json').write_text(json.dumps(payload))
                    (directory / 'error.json').write_text(json.dumps({'type': 'HTTPError', 'http_code': 524}))
                    self.calls.append({'scope': label, 'directory': 'api_calls/failed',
                                       'status': 'failed', 'usage': {}, 'provider_cost': None})
                    (self.ledger_root / 'ledger.json').write_text(json.dumps({
                        'returned_model': self.returned_model, 'calls': self.calls}))
                    raise ModelError('Scripted 524; no real network call')
            return normal(prompt, images)
        return call


class RecoveryFixture(ScriptedAPIFixture):
    def __init__(self, path):
        super().__init__(path)
        self.output = self.ledger_root
        self.temperature = self.public_config['temperature']
        self.verdicts['c01/active/judge/recovery'] = 1  # The repaired fixture is correct.


def test_final_transport_recovery_preserves_real_history_and_original_failure(tmp_path, render_config, views):
    prepared, source, output = (tmp_path / p for p in ('prepared', 'failed', 'continued'))
    old_api = FailingFinalFixture(tmp_path / 'failed-api')
    base = prepare(prepared, render_config, views, old_api, TASK, 'SCRIPTED fixture only', reference_program())
    with pytest.raises(RuntimeError, match='active failed'):
        run_pairs(prepared, source, {'c01': controlled_cases(base)['c01']}, render_config, views,
                  old_api, budget=2, revisions=1, runtime_repairs=0)
    original = (source / 'result.json').read_bytes()
    api = RecoveryFixture(tmp_path / 'continued-api')
    report = recover_final(prepared, source, output, render_config, views, api)
    assert report['status'] == 'complete' and report['continuation']['new_active_observations'] == 0
    assert (source / 'result.json').read_bytes() == original
    assert json.loads(original)['status'] == 'incomplete'
    for name, digest in report['continuation']['copied_history_sha256'].items():
        from turnitover.api_pipeline.paired import file_sha
        assert file_sha(source / name) == digest
    score = score_pairs(prepared, output, render_config, views)
    active = score['pairs']['c01']['active']
    assert active['final_quality']['code_correct'] and active['visual_repair_calls'] == 1
    assert active['api_cost']['requests'] == 8  # Six successful original, one failed, one recovered.
    assert active['api_cost']['requests_without_usage'] == 1
    assert active['api_cost']['tokens'] == 49  # Only seven actual successful fixture usages.
    assert active['total_observations'] == 4
    assert api.calls[0]['scope'] == 'c01/active/judge/recovery'
