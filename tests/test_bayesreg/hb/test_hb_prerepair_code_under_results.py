"""XF (runs/20261115-d1/实现说明_XF.md): hb_prerepair's spy when the code tree lies under the hold-out root, as the detached checkout <results>/holdout_B_v1/code does (C22.H0-4; the tool's
launch string runs from it).  Before XF every open of that tree was a breach ('code' is not one of the hold-out root's allowed entries): the start's lazy `import hb_method`, the
preregistration, the pair list, the unseal record, the sealed lists, the input lock.  Now a read whose two forms lie in the running code tree is allowed (after the name rule and the evaluation's
and data roots, before the hold-out roots' rule); a write into it, its own data/ (XF revision 2, review I-1: a link out of the checkout or a real directory), and the rest of the hold-out root
are judged as before.

Two settings: a small code tree under a temporary hold-out root (the spy's rules; the spy importing a fresh module), and this repository's tree put under the spy's hold-out root (Spy.home its
parent directory) for a whole run of the tool, made to load a module of the code tree lazily under its spy."""
import importlib.util
import os
from pathlib import Path
import sys
import uuid

import pytest

from . import _world as W
from .test_hb_prerepair import finalized_and_moved, lock_repo, pairs, run, unsealed  # noqa: F401   (unsealed: a fixture)

hc, hb, hm = W.hc, W.hb, W.hb_method
import hb_prerepair as P  # noqa: E402


def holdout_with_code(tmp_path, monkeypatch, data='link'):
    """<tmp>/results/holdout_B_v1 as the hold-out root, its code/ a small code tree (a package, a split, the input lock, and data/ -- a link out to <tmp>/seq/data (data='link', the
    production layout) or a real directory (data='dir'))."""
    results = tmp_path/'results'
    home = results/'holdout_B_v1'
    code = home/'code'
    (code/'xfprobe').mkdir(parents=True)
    (code/'xfprobe'/'__init__.py').write_text('')
    (code/'splits'/'bayesreg_v1').mkdir(parents=True)
    (code/'splits'/'bayesreg_v1'/'holdout_B_pairs.tsv').write_text('fixed\tmoving\n')
    (code/'runs'/'holdout_B_v1').mkdir(parents=True)
    (code/'runs'/'holdout_B_v1'/'hb_input_lock.json').write_text('{}')
    for root in (tmp_path/'seq'/'data', code/'data') if data == 'dir' else (tmp_path/'seq'/'data',):
        (root/'three_d'/'x').mkdir(parents=True)
        (root/'three_d'/'manifest.json').write_text('{}')
        (root/'three_d'/'x'/'pair.npz').write_bytes(b'x')
    if data == 'link':
        (code/'data').symlink_to(tmp_path/'seq'/'data')
    (home/'configc'/'p').mkdir(parents=True)
    monkeypatch.setattr(hc, 'RESULTS', results)
    monkeypatch.setattr(hc, 'HOLDOUT_ROOT', home)
    return home, code


@pytest.mark.parametrize('data', ['link', 'dir'])
def test_the_code_tree_under_the_hold_out_root_is_read_and_never_written(tmp_path, monkeypatch, data):
    home, code = holdout_with_code(tmp_path, monkeypatch, data)
    spy = P.Spy(code)
    for good in (code/'xfprobe'/'__init__.py', code/'xfprobe'/'__pycache__'/'__init__.cpython-310.pyc', code/'splits'/'bayesreg_v1'/'holdout_B_pairs.tsv',
                 code/'runs'/'holdout_B_v1'/'hb_input_lock.json', code/'bayesreg'/'io'/'sealed.py'):
        assert spy.why(good) is None, good
    for bad, write in ((code/'xfprobe'/'__init__.py', True), (code/'runs'/'holdout_B_v1'/'new.json', True),        # never written
                       (code/'data'/'three_d'/'manifest.json', False), (code/'data', False),                            # the checkout's data/ (link or real directory): not code (review I-1)
                       (code/'tests'/'aligned_seg35.nii.gz', False), (code/'x'/'pair.npz', False),                     # the name rule first, as before
                       (home/'configc'/'p'/'result.json', False), (home/'eval'/'x.json', False), (home/'code_other'/'x.py', False)):
        assert spy.why(bad, write=write), (bad, write)
    assert P.Spy().code == Path(hb.REPO).resolve()                                         # by default the tree this module is in


def test_the_audit_events_mode_and_flags_make_a_write():
    assert P.is_write('rb', 0) is False and P.is_write('r', 0) is False
    assert all(P.is_write(m, 0) for m in ('w', 'wb', 'a', 'xb', 'r+'))
    assert P.is_write(None, os.O_RDONLY) is False and P.is_write(None, os.O_WRONLY | os.O_CREAT) is True


def test_the_spy_lets_a_lazy_import_from_the_code_tree_under_the_hold_out_root(tmp_path, monkeypatch):
    home, code = holdout_with_code(tmp_path, monkeypatch, 'dir')
    W.XW.sealed_module._reset_cache()
    name = f'xfprobe_{uuid.uuid4().hex[:12]}'
    (code/'xfprobe'/f'{name}.py').write_text('VALUE = 3\n')
    sys.path.insert(0, str(code))
    spy = P.install_spy(code)
    try:
        module = __import__(f'xfprobe.{name}', fromlist=['VALUE'])
        assert module.VALUE == 3
        (code/'splits'/'bayesreg_v1'/'holdout_B_pairs.tsv').read_text()
        with pytest.raises(P.Breach):
            open(code/'xfprobe'/'written.py', 'w').close()
        with pytest.raises(P.Breach):
            os.open(code/'xfprobe'/'written2.py', os.O_WRONLY | os.O_CREAT)
        with pytest.raises(P.Breach):
            (code/'data'/'three_d'/'manifest.json').read_text()                      # the checkout's data/, a real directory here (review I-1)
        with pytest.raises(P.Breach):
            open(home/'configc'/'p'/'result.json').close()
    finally:
        P.remove_spy()
        sys.path.remove(str(code))
        sys.modules.pop(f'xfprobe.{name}', None)
        sys.modules.pop('xfprobe', None)
    assert [b['path'] for b in spy.breaches] == [str(code/'xfprobe'/'written.py'), str(code/'xfprobe'/'written2.py'), str(code/'data'/'three_d'/'manifest.json'),
                                                 str(home/'configc'/'p'/'result.json')]
    assert not (code/'xfprobe'/'written.py').exists() and not (code/'xfprobe'/'written2.py').exists()


@pytest.fixture
def code_under_the_spys_hold_out_root(monkeypatch):
    """The spy's hold-out root (Spy.home) the code tree's parent directory, so that this repository's tree lies under it as <results>/holdout_B_v1/code does; and a module of the code tree
    loaded afresh under the spy (at the start's frozen-settings read), as a lazy import there opens <code>/.../__pycache__/*.pyc and the source."""
    parent = Path(hb.REPO).resolve().parent
    real_init = P.Spy.__init__

    def init(self, code=None):
        real_init(self, code)
        self.home = parent
    monkeypatch.setattr(P.Spy, '__init__', init)
    real_settings = P.he.frozen_settings
    calls = []

    def settings(*args, **kwargs):
        spec = importlib.util.spec_from_file_location(f'xf_lazy_{uuid.uuid4().hex[:8]}', Path(hb.REPO)/'bayesreg'/'io'/'code_identity.py')
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
        calls.append(1)
        return real_settings(*args, **kwargs)
    monkeypatch.setattr(P.he, 'frozen_settings', settings)
    return calls


def test_a_run_from_a_code_tree_under_the_hold_out_root(tmp_path, monkeypatch, unsealed, code_under_the_spys_hold_out_root):  # noqa: F811
    ps = pairs(1)
    finalized_and_moved(ps, monkeypatch)
    repo = lock_repo(tmp_path, ps)
    assert run(repo) == 0
    record = sorted((P.record_root()/'runs').glob('*.json'))[-1]
    import json
    doc = json.loads(record.read_text())
    root = str(Path(hb.REPO).resolve())+'/'
    seen = [e for e in doc['spy']['events'] if e['path'].startswith(root)]
    assert code_under_the_spys_hold_out_root and doc['spy']['passed'] and doc['pairs'][ps[0]]['state'] == 'written' and doc['pairs'][ps[0]]['passed']
    assert any('code_identity' in e['path'] for e in seen) and any(e['path'].endswith(hm.PREREG_REL) for e in seen) and all(e['refused'] is None for e in seen)
