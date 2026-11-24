"""The read lock of the development truth of work package UF (runs/20261105-uf/任务书.md), the way of experiments/ul/dev_lock.py and experiments/nn/nn_lock.py: "the development set is read once" holds across results roots.

A file at one fixed path, outside the root of the segmenter's results and not derived from any option. eval_uf on real data refuses at its first line if the file exists, and creates it with O_EXCL immediately before the first truth label is opened. A run that failed before that point leaves no lock; a run that failed after it keeps the lock (the truth may have been read).
Only BZ may approve another read: --bz-go "<BZ's words and the time>" appends a line, the first line stays. The file is never deleted or edited. Synthetic runs do not touch it. The driver (run_uf.py) checks it before the first training step, so that 10 hours of training are not spent on an evaluation it would refuse.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

DEV_READ_LOCK = Path("/home/rmcfz/seq_vp_tag_results/seg/uf_dev_read.lock")
FIRST_READ, APPROVED_REREAD = "first_read", "bz_approved_reread"


class DevReadLocked(Exception):
    """The development truth was read for UF already; only BZ may approve another read. Status `dev_read_locked`."""
    status = "dev_read_locked"


def describe():
    """The lines of the lock file, as records; empty when there is none."""
    path = Path(DEV_READ_LOCK)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def check(bz_go=None):
    """Before anything is read. DevReadLocked if the lock exists and BZ's words are not given; ValueError for words that are empty or that unlock nothing."""
    path = Path(DEV_READ_LOCK)
    if bz_go is not None and not bz_go.strip():
        raise ValueError("--bz-go: BZ's words and the time are required")
    if bz_go is not None and not path.exists():
        raise ValueError(f"--bz-go: {path} does not exist, there is nothing to unlock")
    if bz_go is None and path.exists():
        first = (describe() or [{}])[0]
        raise DevReadLocked(f"{path} exists: the development truth was read for UF at {first.get('utc')} (out_dir {first.get('out_dir')}); it is read once, whatever --out-dir or --member-root say. Another read only with BZ's approval: --bz-go \"<BZ's words and the time>\"; the file is not deleted")


def take(record, bz_go=None):
    """The moment before the first truth label: create the lock (O_EXCL: a second run loses here even if it passed `check`), or with BZ's words append to it. Returns the line written."""
    path = Path(DEV_READ_LOCK)
    line = dict(record, event=APPROVED_REREAD if bz_go else FIRST_READ, utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), pid=os.getpid(), **({"bz_go": bz_go} if bz_go else {}))
    text = json.dumps(line, ensure_ascii=False)+"\n"
    if bz_go:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(text)
        return line
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise DevReadLocked(f"{path} appeared just before the first truth label; not read") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(text)
    return line
