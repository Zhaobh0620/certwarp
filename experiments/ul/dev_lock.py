"""The read lock of the development truth of work package UL (runs/20261104-ul/plan.md, revision 1; docs/bayesreg/reviews/UL_tools_复核_20261104.md, important 3): "the development set is read once" holds across results roots.

A file at one fixed path, outside the root of the segmenter's results and not derived from any option, so that neither --out nor --out-dir nor --member-root leads around it. eval_u1 on real data refuses at its first line if the file exists, and creates it with O_EXCL immediately before the first
truth label of the arm U1 is opened. A run that failed before that point leaves no lock; a run that failed after it keeps the lock (the truth may have been read). Only BZ may approve another read: --bz-go "<BZ's words and the time>" appends a line, the first line stays. The file is never deleted or edited.
Synthetic runs and the control of the evaluator (member A0 only, whose numbers were read long ago: --control-only) do not touch it.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

DEV_READ_LOCK = Path("/home/rmcfz/seq_vp_tag_results/seg/ul_dev_read.lock")
FIRST_READ, APPROVED_REREAD = "first_read", "bz_approved_reread"


class DevReadLocked(Exception):
    """The development truth was read for U1 already; only BZ may approve another read. Status `dev_read_locked`."""
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
        raise DevReadLocked(f"{path} exists: the development truth was read for U1 at {first.get('utc')} (out_dir {first.get('out_dir')}); it is read once, whatever --out or --out-dir say. Another read only with BZ's approval: --bz-go \"<BZ's words and the time>\"; the file is not deleted")


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
