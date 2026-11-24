"""The sandbox of every XB test (tests/test_bayesreg/xb/_world.py sandbox): the roots in a tempfile.mkdtemp(prefix='xbt_') directory, removed after the test (pytest's tmp_path lies under
/tmp/pytest-of-<user>/ unless --basetemp is given, and the start strings built from the roots refuse the word pytest), XB_TEST_SANDBOX set (I7: holdout_common's root functions refuse the real roots, so a test that
forgot to move a root is stopped before it writes), systemd-run a shim that exits 99; and, after the test, a tripwire: a test that created one of the real roots fails."""
import pytest

from . import _world as W


@pytest.fixture(autouse=True)
def xb_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv(W.hc.TEST_SANDBOX_ENV, '1')
    before = W.real_roots_present()
    base = W.roots_base()
    try:
        yield W.sandbox(tmp_path, monkeypatch, base=base)
        W.sealed_module._reset_cache()          # leave no copy of the lists (pinned or not) to the next test
        after = W.real_roots_present()
        assert after == before, f'a test wrote into the real results roots: {sorted(set(after)-set(before))}'
    finally:
        W.remove_roots_base(base)
