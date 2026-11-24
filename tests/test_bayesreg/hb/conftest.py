"""The sandbox of every HB test (tests/test_bayesreg/hb/_world.py sandbox: XB's, with HE's hold-out sources among the roots), and after the test the tripwire: no real root created.  The roots lie
in a tempfile.mkdtemp(prefix='xbt_') directory removed after the test, as XB's (f027acc): pytest's tmp_path lies under /tmp/pytest-of-<user>/ unless --basetemp is given, and the start strings
built from the roots (hb_core.unit_argv, queue_launch) refuse the word pytest -- the check is not relaxed."""
import pytest

from . import _world as W


@pytest.fixture(autouse=True)
def hb_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv(W.hc.TEST_SANDBOX_ENV, '1')
    extra = W.Path('/home/rmcfz/seq_vp_tag_results/hb_dev_v1')               # hb_configc's development root under the real results root
    before = W.XW.real_roots_present()+[str(extra)]*extra.exists()
    base = W.XW.roots_base()
    try:
        yield W.sandbox(tmp_path, monkeypatch, base=base)
        W.XW.sealed_module._reset_cache()
        after = W.XW.real_roots_present()+[str(extra)]*extra.exists()
        assert after == before, f'a test wrote into the real results roots: {sorted(set(after)-set(before))}'
    finally:
        W.XW.remove_roots_base(base)
