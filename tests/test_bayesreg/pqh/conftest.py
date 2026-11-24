"""The sandbox of every PQH test: holdout_common's roots in tmp_path with XB_TEST_SANDBOX set (the XB sandbox: I7, the systemd-run shim that exits 99, the authorizations in a temporary git
repository), every fixed path of pqh_pins and HI's hold-out lock moved into tmp_path, PQH's own write guard live (pqh_inputs.guard_write refuses the real roots pqh_pins.REAL_WRITE_ROOTS before a
write; review I7), and a tripwire: the list of files under each real root (names, sizes, modification times; no content read) must be the same after the test as before it (review I7: not only
whether the roots exist).  No real subject's file is read."""
import os
from pathlib import Path

import pytest

from ..xb import _world as XW
from . import _world as W


def listing():
    """{real root: sorted (relative path, size, mtime_ns)} of pqh_pins.REAL_WRITE_ROOTS (a root that is a file: its own row; a missing root: None)."""
    out = {}
    for root in W.hp.REAL_WRITE_ROOTS:
        root = Path(root)
        if not root.exists():
            out[str(root)] = None
        elif root.is_file():
            st = root.stat()
            out[str(root)] = [('.', st.st_size, st.st_mtime_ns)]
        else:
            rows = []
            for directory, dirs, files in os.walk(root):
                for name in files+[d for d in dirs if os.path.islink(os.path.join(directory, d))]:
                    path = os.path.join(directory, name)
                    st = os.lstat(path)
                    rows.append((os.path.relpath(path, root), st.st_size, st.st_mtime_ns))
            out[str(root)] = sorted(rows)
    return out


@pytest.fixture(autouse=True)
def pqh_sandbox(tmp_path, monkeypatch):
    before = listing()
    box = XW.sandbox(tmp_path, monkeypatch)
    W.move_pins(tmp_path, monkeypatch)
    yield box
    XW.sealed_module._reset_cache()
    after = listing()
    changed = {root: (before[root], after[root]) for root in before if before[root] != after[root]}
    assert not changed, f'a test wrote under a real root: {sorted(changed)}'
