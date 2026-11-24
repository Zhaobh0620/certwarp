"""The two read locks of R3 (runs/20261111-r3/plan.md), the way of experiments/r2/r2_lock.py (copied, the path made a parameter):
  calib  <SEG>/r3_calib_read.lock   the one read of the training set's calibration truth for the composition (compose_r3.py)
  dev    <SEG>/r3_dev_read.lock     the one read of the development truth (eval_r3.py)
A file at a fixed path (r3_common), not derived from any option.  The reader on real data refuses at its first line if its lock exists, and creates it with O_EXCL immediately before the first truth label
is opened.  A run that failed before that point leaves no lock; one that failed after it keeps it.  Only BZ may approve another read: --bz-go "<BZ's words and the time>" appends a line.  Never deleted or
edited.  Synthetic runs do not touch them (the tests point the paths elsewhere).  The chains check them before their first step.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import r3_common as r3c

PATHS = dict(calib=r3c.CALIB_READ_LOCK, dev=r3c.DEV_READ_LOCK)
WORDS = dict(calib="the training set's calibration truth was read for R3's composition", dev="the development truth was read for R3's segmentation level")
FIRST_READ, APPROVED_REREAD = 'first_read', 'bz_approved_reread'


class ReadLocked(Exception):
    """The truth was read already; only BZ may approve another read."""
    status = 'read_locked'


def path_of(kind):
    return Path(PATHS[kind])


def describe(kind):
    path = path_of(kind)
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []


def check(kind, bz_go=None):
    """Before anything is read.  ReadLocked if the lock exists without BZ's words; ValueError for words that are empty or unlock nothing."""
    path = path_of(kind)
    if bz_go is not None and not bz_go.strip():
        raise ValueError("--bz-go: BZ's words and the time are required")
    if bz_go is not None and not path.exists():
        raise ValueError(f'--bz-go: {path} does not exist, there is nothing to unlock')
    if bz_go is None and path.exists():
        first = (describe(kind) or [{}])[0]
        raise ReadLocked(f"{path} exists: {WORDS[kind]} at {first.get('utc')} (out_dir {first.get('out_dir')}); another read only with BZ's approval: "
                         '--bz-go "<BZ\'s words and the time>"; the file is not deleted')


def take(kind, record, bz_go=None):
    """The moment before the first truth label: create the lock (O_EXCL), or with BZ's words append to it.  Returns the line written."""
    path = path_of(kind)
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
        raise ReadLocked(f'{path} appeared just before the first truth label; not read') from None
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        stream.write(text)
    return line
