import json

import pytest

from scripts.summarize_api_pipeline import summarize


def test_accounting_includes_failed_attempts_without_inventing_usage_or_cost(tmp_path):
    ledger = tmp_path / 'ledger.json'
    ledger.write_text(json.dumps({'request_count': 2, 'calls': [
        {'scope': 'diagnostic', 'status': 'failed', 'usage': {}, 'provider_cost': None},
        {'scope': 'c02/active/judge', 'status': 'complete', 'usage': {'total_tokens': 20},
         'provider_cost': None}]}))
    result = summarize([], [ledger])['api_accounting']
    assert result['request_attempts'] == 2
    assert result['request_status'] == {'failed': 1, 'complete': 1}
    assert result['reported_tokens'] == 20 and result['attempts_without_usage'] == 1
    assert result['provider_cost_total'] is None and result['reported_cost_count'] == 0


def test_accounting_rejects_duplicate_ledger(tmp_path):
    ledger = tmp_path / 'ledger.json'
    with pytest.raises(ValueError, match='double count'):
        summarize([], [ledger, ledger])
