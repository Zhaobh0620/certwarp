"""The read lock of the development truth of R2's segmentation-level reading (runs/20261110-r2/plan.md section 4), the way of experiments/uf/uf_lock.py (copied with the path and the words changed):
"the development set is read once for R2's segmentation level" holds across results roots.

A file at one fixed path (r2_common.DEV_READ_LOCK), not derived from any option.  eval_r2 on real data refuses at its first line if it exists, and creates it with O_EXCL immediately before the first truth
label is opened.  A run that failed before that point leaves no lock; one that failed after it keeps the lock.  Only BZ may approve another read: --bz-go "<BZ's words and the time>" appends a line.  The
file is never deleted or edited.  Synthetic runs do not touch it (the tests point DEV_READ_LOCK elsewhere).  The chain (run_r2.py) checks it before its first step, so that the training is not spent on an
evaluation it would refuse.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import r2_common as rc

DEV_READ_LOCK = rc.DEV_READ_LOCK
FIRST_READ, APPROVED_REREAD = 'first_read', 'bz_approved_reread'


class DevReadLocked(Exception):
    """The development truth was read for R2's segmentation level already; only BZ may approve another read."""
    status = 'dev_read_locked'


def describe():
    path = Path(DEV_READ_LOCK)
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []


def check(bz_go=None):
    """Before anything is read.  DevReadLocked if the lock exists without BZ's words; ValueError for words that are empty or unlock nothing."""
    path = Path(DEV_READ_LOCK)
    if bz_go is not None and not bz_go.strip():
        raise ValueError("--bz-go: BZ's words and the time are required")
    if bz_go is not None and not path.exists():
        raise ValueError(f'--bz-go: {path} does not exist, there is nothing to unlock')
    if bz_go is None and path.exists():
        first = (describe() or [{}])[0]
        raise DevReadLocked(f"{path} exists: the development truth was read for R2's segmentation level at {first.get('utc')} (out_dir {first.get('out_dir')}); another read only with BZ's approval: "
                            '--bz-go "<BZ\'s words and the time>"; the file is not deleted')


def take(record, bz_go=None):
    """The moment before the first truth label: create the lock (O_EXCL), or with BZ's words append to it.  Returns the line written."""
    path = Path(DEV_READ_LOCK)
    line = dict(record, event=APPROVED_REREAD if bz_go else FIRST_READ, utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), pid=os.getpid(), **({'bz_go': bz_go} if bz_go else {}))
    text = json.dumps(line, ensure_ascii=False)+'\n'
    if bz_go:
        with path.open('a', encoding='utf-8') as stream:
            stream.write(text)
        return line
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise DevReadLocked(f'{path} appeared just before the first truth label; not read') from None
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        stream.write(text)
    return line
