"""XF (runs/20261115-d1/实现说明_XF.md): HP's spy when the code tree lies inside the results root, as the hold-out's detached checkout <results>/holdout_B_v1/code (C22.H0-4) and the P0
checkouts <results>/p0_code_<sha> do.  hp_core's docstring (5): a module imported lazily under the spy reads code files -- allowed; before XF the results root was judged before the code, so
under the results root that read (and the import machinery's listing of a package directory) was a breach.  Now: a read of a code file and a listing of a directory whose two forms lie in a
code root are allowed after the data roots and the code trees' data/ and before the results root; a write into the code tree, a non-code file of it not allowed one by one, the rest of the
results root are judged as before.

Two settings: a small code tree inside a temporary results root (the policy's classes; the spy importing a fresh module), and this repository's tree put inside the results root as the policy
sees it (every Policy's results plus the tree's parent) for the four spied runs, each made to load a module of the code tree lazily under its spy (at the machine gate, after the admission)."""
import importlib.util
import os
from pathlib import Path
import sys
import uuid

import pytest

from . import _world as W

hc, core, prep, ah, S = W.hc, W.core, W.prep, W.ah, W.S


def code_tree_in_results(tmp_path):
    results = tmp_path/'results'
    code = results/'code'
    (code/'xfprobe').mkdir(parents=True)
    (code/'xfprobe'/'__init__.py').write_text('')
    (code/'configs').mkdir()
    (code/'configs'/'x.json').write_text('{}')
    (code/'data').mkdir()                                                   # a real data/ directory in the code tree: a forbidden root (Policy: <code root>/data)
    (code/'data'/'m.py').write_text('')
    (results/'other').mkdir()
    (results/'other'/'y.py').write_text('')
    (code/'link_out.py').symlink_to(results/'other'/'y.py')
    return results, code


def policy(tmp_path, results, code):
    return core.Policy(name='xf test', write_inside=[tmp_path/'out'], results=results, data_roots=(tmp_path/'subjects',), code_roots=(code,))


def test_a_code_tree_inside_the_results_root_is_read_as_code_and_never_written(tmp_path):
    results, code = code_tree_in_results(tmp_path)
    pol = policy(tmp_path, results, code)
    assert pol.classify(code/'xfprobe'/'__init__.py', False)[0] == 'allowed'
    assert pol.classify(code/'xfprobe'/'__pycache__'/'__init__.cpython-310.pyc', False)[0] == 'allowed'     # D1's kind of open: a cached bytecode that does not exist
    assert pol.classify(code/'bayesreg'/'eval'/'truth.py', False)[0] == 'allowed'                             # code is exempt from the name rule, as before
    assert pol.classify_listing(code/'xfprobe') == ('allowed', 'a directory of the running code tree, wherever it lies (XF)')
    assert pol.classify_listing(code)[0] == 'allowed'
    # never written; a non-code file of the code tree only one by one (forbidden under the results root, as before)
    assert pol.classify(code/'xfprobe'/'__init__.py', True)[0] == 'forbidden'
    assert pol.classify(code/'configs'/'x.json', False)[0] == 'forbidden'
    # the code tree's data/ and a link out of the tree: judged where they are
    assert pol.classify(code/'data'/'m.py', False)[0] == 'forbidden'
    assert pol.classify_listing(code/'data')[0] == 'forbidden'
    assert pol.classify(code/'link_out.py', False)[0] == 'forbidden'
    # the rest of the results root as before
    assert pol.classify(results/'other'/'y.py', False)[0] == 'forbidden'
    assert pol.classify_listing(results/'other')[0] == 'forbidden'
    assert pol.classify_listing(results)[0] == 'forbidden'


def test_outside_the_results_root_nothing_changes(tmp_path):
    """The same tree with the results root elsewhere: the classes the policy gave before XF (code read: 'code'; a write: unlisted; a listing: not watched)."""
    _, code = code_tree_in_results(tmp_path)
    pol = policy(tmp_path, tmp_path/'elsewhere', code)
    assert pol.classify(code/'xfprobe'/'__init__.py', False)[0] == 'allowed'
    assert pol.classify(code/'xfprobe'/'__init__.py', True)[0] == 'unlisted'
    assert pol.classify(code/'configs'/'x.json', False)[0] == 'unlisted'
    assert pol.classify_listing(code/'xfprobe')[0] == 'allowed'


def test_the_spy_lets_a_lazy_import_from_a_code_tree_inside_the_results_root(tmp_path):
    """A fresh module of the code tree imported under the spy: the import machinery lists the directory (os.listdir) and opens the cached bytecode and the source (io.open_code) --
    all allowed; then a write into the code tree and a read of the results root breach."""
    results, code = code_tree_in_results(tmp_path)
    name = f'xfprobe_{uuid.uuid4().hex[:12]}'
    (code/'xfprobe'/f'{name}.py').write_text('VALUE = 5\n')
    sys.path.insert(0, str(code))
    spy = core.Spy(policy(tmp_path, results, code)).install()
    try:
        module = __import__(f'xfprobe.{name}', fromlist=['VALUE'])
        assert module.VALUE == 5
        with pytest.raises(core.Breach):
            open(code/'xfprobe'/'written.py', 'w').close()
        with pytest.raises(core.Breach):
            (results/'other'/'y.py').read_text()
    finally:
        spy.remove()
        sys.path.remove(str(code))
        sys.modules.pop(f'xfprobe.{name}', None)
        sys.modules.pop('xfprobe', None)
    events = {(e['resolved'], e['write']): e['cls'] for e in spy.events}
    source = str((code/'xfprobe'/f'{name}.py').resolve())
    assert events[(source, False)] == 'allowed'
    assert any(p.startswith(str((code/'xfprobe'/'__pycache__').resolve())+'/') and c == 'allowed' for (p, _), c in events.items())
    assert [(b['resolved'], b['write'], b['cls']) for b in spy.breaches] == [(str((code/'xfprobe'/'written.py').resolve()), True, 'forbidden'),
                                                                           (str((results/'other'/'y.py').resolve()), False, 'forbidden')]
    assert not (code/'xfprobe'/'written.py').exists()


# ---- this repository's tree inside the results root, the four spied runs -------------------------------------------------------------------------------------------------
def lazy_load():
    """A module of the code tree loaded afresh (not through sys.modules): its loader opens <code>/bayesreg/io/__pycache__/code_identity.cpython-*.pyc and code_identity.py, as D1's
    driver did on its lazy import."""
    spec = importlib.util.spec_from_file_location(f'xf_lazy_{uuid.uuid4().hex[:8]}', W.ROOT/'bayesreg'/'io'/'code_identity.py')
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


@pytest.fixture
def code_inside_the_results_root(monkeypatch):
    """Every Policy's results root plus the code tree's parent (the results root above <results>/p0_code_<sha>, <results>/holdout_B_v1/code), and a lazy load of a module of the code tree
    at the machine gate, under the spy, after the admission.  Before XF each run below stopped there with a breach."""
    parent = W.ROOT.resolve().parent
    real_init = core.Policy.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        if self.results is not None:
            self.results = sorted(set(self.results) | {parent})
    monkeypatch.setattr(core.Policy, '__init__', init)
    gate = hc.machine_gate
    calls = []

    def lazy_gate(*args, **kwargs):
        lazy_load()
        calls.append(1)
        return gate(*args, **kwargs)
    monkeypatch.setattr(hc, 'machine_gate', lazy_gate)
    return calls


def code_events(record):
    root = str(W.ROOT.resolve())+'/'
    return [e for e in record['spy']['events'] if e['resolved'].startswith(root+'bayesreg/io/') and 'code_identity' in e['resolved']]


def test_the_hold_out_preparation_with_the_code_tree_inside_the_results_root(tmp_path, monkeypatch, code_inside_the_results_root):
    w = W.make_world(tmp_path, monkeypatch)
    S._reset_cache()
    record = prep.run_holdout(unseal_path=w.record_path, code=W.CODE, log=W.quiet)
    assert code_inside_the_results_root and record['passed'] and record['spy']['passed'] and not record['spy']['breaches']
    assert code_events(record) and all(e['cls'] == 'allowed' and not e['write'] for e in code_events(record))


def test_the_preparations_development_check_with_the_code_tree_inside_the_results_root(tmp_path, monkeypatch, code_inside_the_results_root):
    w = W.make_world(tmp_path, monkeypatch, decided=False)
    monkeypatch.setattr(hc, 'DEV_PAIRS', W.DEV_PAIRS)
    W.write_old_stage(hc.DEV_STAGE, w.data, W.DEV_PAIRS)
    monkeypatch.setattr(prep, 'PREP_DEVCHECK_PLAN_SHA256', W.plan_file(w, prep.DEVCHECK_PLAN_REL))
    S._reset_cache()
    record = prep.run_devcheck(code=W.CODE, log=W.quiet)
    assert code_inside_the_results_root and record['reading'] == '逐位相同 2/2' and record['spy']['passed']
    assert code_events(record) and all(e['cls'] == 'allowed' for e in code_events(record))


def test_the_hold_out_atropos_with_the_code_tree_inside_the_results_root(tmp_path, monkeypatch, code_inside_the_results_root):
    w = W.make_world(tmp_path, monkeypatch)
    S._reset_cache()
    prep.run_holdout(unseal_path=w.record_path, code=W.CODE, log=W.quiet)
    W.fake_ants(tmp_path, monkeypatch)
    S._reset_cache()
    record = ah.run_holdout(unseal_path=w.record_path, code=W.CODE, log=W.quiet)
    assert len(code_inside_the_results_root) == 2 and record['passed'] and record['spy']['passed']
    assert code_events(record) and all(e['cls'] == 'allowed' for e in code_events(record))


def test_the_atropos_development_check_with_the_code_tree_inside_the_results_root(tmp_path, monkeypatch, code_inside_the_results_root):
    from .test_hp_atropos import devcheck_world
    devcheck_world(tmp_path, monkeypatch)
    S._reset_cache()
    record = ah.run_devcheck(code=W.CODE, log=W.quiet)
    assert code_inside_the_results_root and record['reading'] == '逐位相同 2/2' and record['spy']['passed']
    assert code_events(record) and all(e['cls'] == 'allowed' for e in code_events(record))
