"""The hold-out module's tests use the XB sandbox (the same roots in a tempfile.mkdtemp(prefix='xbt_') directory removed after the test, the systemd-run shim)."""
import pytest

from ..xb import _world as W


@pytest.fixture(autouse=True)
def holdout_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv(W.hc.TEST_SANDBOX_ENV, '1')                # I7
    before = W.real_roots_present()
    base = W.roots_base()
    try:
        yield W.sandbox(tmp_path, monkeypatch, base=base)
        W.sealed_module._reset_cache()
        assert W.real_roots_present() == before, 'a test wrote into the real results roots'
    finally:
        W.remove_roots_base(base)
