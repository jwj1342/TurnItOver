import json
from dataclasses import replace
from datetime import datetime, timezone
from email.utils import format_datetime
from urllib.error import HTTPError, URLError

import pytest

from turnitover.models import request_control as control
from turnitover.models.config import ModelConfig, model_config


def cfg(**kwargs):
    return replace(ModelConfig('generator', 'openai_compatible', 'model',
                               'https://endpoint.test/v1', 'private-key'), **kwargs)


@pytest.fixture
def clock(tmp_path, monkeypatch):
    class Clock:
        now = 1000.0
        waits = []
        def time(self):
            return self.now
        def sleep(self, seconds):
            self.waits.append(seconds)
            self.now += seconds
    clock = Clock()
    monkeypatch.setattr(control, 'state_directory', lambda: tmp_path)
    monkeypatch.setattr(control.time, 'time', clock.time)
    monkeypatch.setattr(control.time, 'sleep', clock.sleep)
    return clock


def test_shared_account_interval_covers_models_roles_and_failed_calls(clock, tmp_path):
    starts = []
    def operation():
        starts.append(clock.now)
        return 'ok'
    assert control.send(cfg(request_interval=60), operation) == 'ok'
    # A separate role/model still shares the same endpoint/key lane on disk.
    assert control.send(cfg(role='judge', model='different-model', request_interval=60), operation) == 'ok'
    assert starts == [1000, 1060]
    assert clock.waits == [60]
    events = [json.loads(x) for x in next(tmp_path.glob('*.jsonl')).read_text().splitlines()]
    assert [e['event'] for e in events] == ['started', 'received', 'started', 'received']
    assert events[2]['wait_seconds'] == 60


def test_retries_stop_at_four_attempts_and_persist_cooldown(clock, tmp_path):
    starts = []
    def fail():
        starts.append(clock.now)
        raise URLError('https://endpoint.test/private-key')
    with pytest.raises(URLError):
        control.send(cfg(request_interval=60, max_retries=3), fail)
    assert starts == [1000, 1060, 1180, 1420]
    assert len(starts) == 4
    events = [json.loads(x) for x in next(tmp_path.glob('*.jsonl')).read_text().splitlines()]
    failed = [e for e in events if e['event'] == 'failed']
    assert [e['will_retry'] for e in failed] == [True, True, True, False]
    # Even the exhausted final error cools down the next caller.
    called = []
    control.send(cfg(request_interval=60), lambda: called.append(clock.now))
    assert called == [1900]
    assert 'private-key' not in ''.join(p.read_text() for p in tmp_path.iterdir())
    assert 'endpoint.test' not in ''.join(p.read_text() for p in tmp_path.iterdir())


def test_retry_after_has_priority_and_403_is_not_retried(clock):
    calls = []
    def rate_limited_once():
        calls.append(clock.now)
        if len(calls) == 1:
            raise HTTPError('https://private-key.test', 429, 'secret', {'Retry-After': '180'}, None)
        return 'ok'
    assert control.send(cfg(request_interval=60, max_retries=3), rate_limited_once) == 'ok'
    assert calls == [1000, 1180]
    calls.clear()
    def forbidden():
        calls.append(clock.now)
        raise HTTPError('https://private-key.test', 403, 'secret', {}, None)
    with pytest.raises(HTTPError):
        control.send(cfg(request_interval=60, max_retries=3), forbidden)
    assert len(calls) == 1


def test_retry_after_http_date_and_invalid_values():
    future = datetime.now(timezone.utc).timestamp() + 120
    seconds = control.retry_after({'Retry-After': format_datetime(datetime.fromtimestamp(future, timezone.utc))})
    assert 118 <= seconds <= 120
    for value in ('invalid', '-2', 'nan', 'inf'):
        assert control.retry_after({'Retry-After': value}) == 0


def test_unconfigured_behavior_unchanged(clock, tmp_path):
    called = []
    assert control.send(cfg(), lambda: called.append(1)) is None
    assert called == [1]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('interval,retries', [('-1', '0'), ('nan', '0'), ('inf', '0'),
                                            ('0', '-1'), ('0', '4'), ('0', '1.5')])
def test_invalid_environment_bounds(interval, retries):
    with pytest.raises(ValueError):
        model_config('generator', {'TIO_GENERATOR_REQUEST_INTERVAL_SECONDS': interval,
                                   'TIO_GENERATOR_MAX_RETRIES': retries})


def test_config_explicit_and_defaults():
    assert cfg().request_interval == 0 and cfg().max_retries == 0
    configured = model_config('generator', {'TIO_GENERATOR_REQUEST_INTERVAL_SECONDS': '60',
                                           'TIO_GENERATOR_MAX_RETRIES': '3'})
    assert configured.request_interval == 60 and configured.max_retries == 3
    assert configured.public()['request_interval'] == 60
    with pytest.raises(ValueError):
        cfg(max_retries=4).validate()


def test_shared_lane_serializes_separate_processes(tmp_path):
    import multiprocessing
    import os
    import time
    context = multiprocessing.get_context('fork')
    entered = context.Event()
    release = context.Event()
    second_entered = context.Event()
    outcome = context.Queue()
    def first():
        control.state_directory = lambda: tmp_path
        def operation():
            entered.set()
            assert release.wait(5)
            return 'first'
        outcome.put(control.send(cfg(request_interval=.05), operation))
    def second():
        control.state_directory = lambda: tmp_path
        outcome.put(control.send(cfg(role='judge', model='other', request_interval=.05),
                                 lambda: second_entered.set() or 'second'))
    a, b = context.Process(target=first), context.Process(target=second)
    a.start()
    try:
        assert entered.wait(5)
        b.start()
        assert not second_entered.wait(.1)
        release.set()
        assert second_entered.wait(5)
        a.join(5); b.join(5)
        assert a.exitcode == b.exitcode == 0
        assert {outcome.get(timeout=1), outcome.get(timeout=1)} == {'first', 'second'}
    finally:
        release.set()
        for process in (a, b):
            if process.pid and process.is_alive():
                process.terminate(); process.join(5)
