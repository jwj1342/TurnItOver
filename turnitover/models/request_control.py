"""Cross-process request spacing and bounded transport retries, opt-in per model."""
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit

RETRYABLE = {429, 502, 503, 504}


def retry_after(headers):
    value = headers.get('Retry-After') if headers else None
    if value is None:
        return 0.0
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            seconds = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return 0.0
    return seconds if 0 <= seconds < float('inf') else 0.0


def state_directory():
    path = Path(tempfile.gettempdir()) / f'tio-request-control-{os.getuid()}'
    path.mkdir(mode=0o700, exist_ok=True)
    return path


@contextmanager
def request_slot(cfg):
    # Account-wide scope: models/roles on the same endpoint/key share one lane.
    key = hashlib.sha256((cfg.base_url + '\0' + cfg.api_key).encode()).hexdigest()
    root = state_directory()
    with (root / (key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield root / (key + '.json'), root / (key + '.jsonl')


def save_state(path, next_allowed):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'next_allowed': next_allowed}))
    temporary.replace(path)


def send(cfg, operation):
    """Hold the shared lane through response/retry; every attempt takes a slot.

    Never log URLs, credentials, payloads or provider error bodies. A timeout
    may already have incurred usage; attempts without responses remain unknown.
    """
    if cfg.request_interval == 0 and cfg.max_retries == 0:
        return operation()
    with request_slot(cfg) as (state_path, log_path):
        for attempt in range(cfg.max_retries + 1):
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            wait = max(0.0, state.get('next_allowed', 0.0) - time.time())
            if wait:
                time.sleep(wait)
            started = time.time()
            save_state(state_path, started + cfg.request_interval)
            event = {'attempt': attempt + 1, 'started_at': started, 'wait_seconds': wait}
            # Persist start before sending, so interrupted attempts remain visible.
            with log_path.open('a') as log:
                log.write(json.dumps({**event, 'event': 'started'}) + '\n')
            try:
                result = operation()
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                retryable = not isinstance(exc, HTTPError) or exc.code in RETRYABLE
                delay = max(cfg.request_interval, 30.0) * 2 ** attempt
                if isinstance(exc, HTTPError):
                    delay = max(delay, retry_after(exc.headers))
                    event['http_status'] = exc.code
                    event['request_id'] = exc.headers.get('x-request-id') if exc.headers else None
                    location = exc.headers.get('Location') if exc.headers else None
                    if location:
                        target = urlsplit(location)
                        event['redirect_host'] = target.hostname
                        event['redirect_scheme'] = target.scheme
                        # No query/path: redirect targets may contain credentials.
                event.update(error_type=type(exc).__name__, will_retry=retryable and attempt < cfg.max_retries)
                if retryable:
                    save_state(state_path, max(started + cfg.request_interval, time.time() + delay))
                with log_path.open('a') as log:
                    log.write(json.dumps({**event, 'event': 'failed', 'ended_at': time.time()}) + '\n')
                if not event['will_retry']:
                    raise
            else:
                with log_path.open('a') as log:
                    log.write(json.dumps({**event, 'event': 'received', 'ended_at': time.time()}) + '\n')
                return result
