"""The sandbox of every HP test (tests/test_bayesreg/hp/_world.py sandbox): the roots of holdout_common in tmp_path, holdout_common.TEST_SANDBOX_ENV set (a real root is
refused before anything is written; review I2), the machine gate replaced (its own test uses the real one with a fake query), systemd-run a shim that exits 99, the
guard's cache of the sealed lists empty before and after; and a tripwire, the second net: a test that creates one of the real roots fails."""
import pytest

from . import _world as W


@pytest.fixture(autouse=True)
def hp_sandbox(tmp_path, monkeypatch):
    before = W.real_roots_present()
    yield W.sandbox(tmp_path, monkeypatch)
    W.S._reset_cache()
    after = W.real_roots_present()
    assert after == before, f'a test wrote into the real results roots: {sorted(set(after)-set(before))}'
