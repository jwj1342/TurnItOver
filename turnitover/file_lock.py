"""Host-local process locks for request lanes and batch output ownership."""
from contextlib import contextmanager
import errno
from pathlib import Path
import sys
import time


@contextmanager
def file_lock(path, *, blocking=True):
    with Path(path).open('a+b') as handle:
        if sys.platform == 'win32':
            import msvcrt
            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if not blocking or exc.errno not in (errno.EACCES, errno.EDEADLK):
                        raise
                    time.sleep(.05)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            if sys.platform == 'win32':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
