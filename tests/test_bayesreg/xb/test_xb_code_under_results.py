"""XF (runs/20261115-d1/实现说明_XF.md): XB's spies when the code tree lies inside the results root, as the hold-out's detached checkout <results>/holdout_B_v1/code (C22.H0-4) and the P0
checkouts <results>/p0_code_<sha> do.  D1 stopped at XB's dry run on P0: the driver's lazy import of bayesreg.io.code_identity opened <code>/bayesreg/io/__pycache__/*.pyc (io.open_code, audit
event 'open') under the spy, and the policy judged the results root (a forbidden root) before the code root.  Rule 3a of xb_policy: a read of the running code tree is allowed wherever it lies;
a write into it is not; the rest of the results root and a forbidden root at or below the code root stay forbidden.  XF revision 2 (review I-1): the code tree's own data/ is such a root
(xb_spy.code_data_root) -- in production main_repo (the main checkout) is not the code root, so without it a checkout whose data/ is a real directory would have been read through 3a; the tests
use the production layout (main_repo apart from the code root) with data/ a real directory and a link out of the checkout.

Two settings: a small code tree of its own inside a temporary results root (the policy's classes; the driver's spy importing a fresh module; spyrun running a method script that lies there, a
real subprocess), and this repository's tree put inside a forbidden root that contains it (xb_spy.forbidden_roots plus the tree's parent: the results root seen from the policy) for the
driver's own runs -- the P0 dry run of d1_expected.md section 4, a development run and a hold-out run."""
import json
import os
import subprocess
import sys
import time
import uuid

import pytest

from . import _world as W
from .test_xb_run import holdout_world, result

hc, xc, xb_spy, xb_drive = W.hc, W.xc, W.xb_spy, W.xb_drive
import xb_policy  # noqa: E402


def stage_like(root):
    """<root>/three_d/manifest.json and <root>/three_d/x/pair.npz: what a data/ holds (synthetic bytes)."""
    (root/'three_d'/'x').mkdir(parents=True)
    (root/'three_d'/'manifest.json').write_text('{}')
    (root/'three_d'/'x'/'pair.npz').write_bytes(b'x')


def code_tree_in_results(tmp_path, data='dir'):
    """The production layout: <tmp>/main (the main checkout, main_repo, its data/ a real directory standing for the link's target) apart from <tmp>/results/code (the running checkout: a
    package, a splits file, a link out to <results>/other, and data/ -- a real directory (data='dir') or a link to <tmp>/main/data (data='link', as the P0 checkout's link resolves out of it)),
    <results>/other, <results>/holdout_B_v1/stage."""
    results = tmp_path/'results'
    code = results/'code'
    pkg = code/'xfprobe'
    pkg.mkdir(parents=True)
    (pkg/'__init__.py').write_text('')
    (code/'splits'/'bayesreg_v1').mkdir(parents=True)
    (code/'splits'/'bayesreg_v1'/'holdout_B_pairs.tsv').write_text('fixed\tmoving\n')
    stage_like(tmp_path/'main'/'data')
    if data == 'dir':
        stage_like(code/'data')
    else:
        (code/'data').symlink_to(tmp_path/'main'/'data')
    (results/'other').mkdir()
    (results/'other'/'disp.npz').write_bytes(b'x')
    (code/'link_out').symlink_to(results/'other')
    (results/'holdout_B_v1'/'stage'/'p').mkdir(parents=True)
    (results/'holdout_B_v1'/'stage'/'p'/'manifest.json').write_text('{}')
    return results, code


def both_policies(tmp_path, results, code):
    """The driver's and a method process's policy with this code root and this results root; main_repo <tmp>/main, apart from the code root, as in production (hc.MAIN_REPO and xc.REPO)."""
    common = dict(code_root=code, results=results, main_repo=tmp_path/'main', holdout_root=results/'holdout_B_v1')
    driver = xb_spy.driver_policy(work=tmp_path/'work', record_dir=tmp_path/'rec', lock_dir=tmp_path/'locks', **common)
    method = xb_spy.method_policy(name='test', work=results/'work'/'a'/'p', env_prefix=None, baselines=[], **common)
    return dict(driver=driver, method=method)


@pytest.mark.parametrize('data', ['dir', 'link'])
@pytest.mark.parametrize('which', ['driver', 'method'])
def test_a_code_tree_inside_the_results_root_is_read_and_never_written(tmp_path, which, data):
    results, code = code_tree_in_results(tmp_path, data)
    document = both_policies(tmp_path, results, code)[which]
    assert document['code_roots'] == [str(code)] and str(code/'data') in document['forbidden_roots']
    pol = xb_policy.Policy(document)
    cls = lambda p, write=False: pol.classify(os.path.realpath(p), write)[0]       # noqa: E731   (the spies classify the resolved form: xb_policy.norm)
    assert cls(code/'xfprobe'/'__init__.py') == 'allowed'
    assert cls(code/'xfprobe'/'__pycache__'/'__init__.cpython-310.pyc') == 'allowed'          # D1's open: the cached bytecode, which does not exist
    assert cls(code/'splits'/'bayesreg_v1'/'holdout_B_pairs.tsv') == 'allowed'
    assert pol.classify(os.path.realpath(code/'xfprobe'/'__init__.py'))[1] == 'a read of the running code tree'
    # never written: the results root's forbidden class, as before
    assert cls(code/'xfprobe'/'__init__.py', write=True) == 'forbidden'
    assert cls(code/'xfprobe'/'new.py', write=True) == 'forbidden'
    # the rest of the results root and its hold-out stage: forbidden as before
    assert cls(results/'other'/'disp.npz') == 'forbidden'
    assert cls(results/'holdout_B_v1'/'stage'/'p'/'manifest.json') == 'forbidden'
    assert cls(results/'seg'/'x'/'probs36.npz') == 'forbidden'
    # the checkout's own data/ (review I-1): a real directory (a forbidden root at or below the code root: code_data_root) or a link out of the checkout (judged where it points: main/data)
    assert cls(code/'data'/'three_d'/'manifest.json') == 'forbidden'
    assert cls(code/'data'/'three_d'/'x'/'pair.npz') == 'forbidden'
    assert cls(code/'data') == 'forbidden'
    assert cls(code/'link_out'/'disp.npz') == 'forbidden'
    # the name rules are unchanged outside the code tree
    assert cls(results/'work'/'a'/'p'/'pair.npz') == 'forbidden'


def test_a_policy_written_without_code_roots_reads_as_before(tmp_path):
    """The key is optional: a policy document without it (written before XF) classifies as the order 1-6 did -- the results root first."""
    results, code = code_tree_in_results(tmp_path)
    document = dict(both_policies(tmp_path, results, code)['driver'])
    del document['code_roots']
    pol = xb_policy.Policy(document)
    assert pol.code_roots == () and 'code_roots' not in pol.document
    assert pol.classify(os.path.realpath(code/'xfprobe'/'__init__.py'))[0] == 'forbidden'


def test_a_code_root_that_is_a_forbidden_root_gets_no_reads(tmp_path):
    """3a only takes a forbidden root above the code root out of the way: a code root equal to a forbidden root (the results root given as the code root) is forbidden whole."""
    results, code = code_tree_in_results(tmp_path)
    document = dict(both_policies(tmp_path, results, code)['driver'], code_roots=[str(results)])
    pol = xb_policy.Policy(document)
    assert pol.classify(os.path.realpath(code/'xfprobe'/'__init__.py'))[0] == 'forbidden'
    assert pol.classify(os.path.realpath(results/'other'/'disp.npz'))[0] == 'forbidden'


def fresh_module(code):
    """A module of the code tree that no test imported: its import opens the cached bytecode path and the source through io.open_code, under the spy."""
    name = f'xfprobe_{uuid.uuid4().hex[:12]}'
    (code/'xfprobe'/f'{name}.py').write_text('VALUE = 7\n')
    return f'xfprobe.{name}'


def test_the_drivers_spy_lets_a_lazy_import_from_a_code_tree_inside_the_results_root(tmp_path):
    """D1's failure, in this process: a lazy import of a module of the code tree under DriverSpy; then a write into the code tree, a read of the checkout's data/ (a real directory, review
    I-1) and a read of the results root still breach."""
    results, code = code_tree_in_results(tmp_path)
    policy = both_policies(tmp_path, results, code)['driver']
    name = fresh_module(code)
    sys.path.insert(0, str(code))
    try:
        with xb_spy.DriverSpy(policy) as spy:
            module = __import__(name, fromlist=['VALUE'])
            assert module.VALUE == 7
            (code/'splits'/'bayesreg_v1'/'holdout_B_pairs.tsv').read_text()
            with pytest.raises(xb_spy.Breach):
                open(code/'xfprobe'/'written.txt', 'w').close()
            with pytest.raises(xb_spy.Breach):
                (code/'data'/'three_d'/'x'/'pair.npz').read_bytes()
            with pytest.raises(xb_spy.Breach):
                (results/'other'/'disp.npz').read_bytes()
        record = spy.record(expect_original=False)
    finally:
        sys.path.remove(str(code))
        sys.modules.pop(name, None)
        sys.modules.pop('xfprobe', None)
    seen = {e['path']: e['cls'] for e in spy.events}
    source = os.path.realpath(code/'xfprobe'/f'{name.split(".")[1]}.py')
    cached = [p for p in seen if p.startswith(os.path.realpath(code/'xfprobe'/'__pycache__')+'/')]
    assert seen[source] == 'allowed' and cached and all(seen[p] == 'allowed' for p in cached)          # the .pyc that D1 tripped on is seen and allowed
    assert [(b['path'], b['write'], b['cls']) for b in record['breaches']] == [(os.path.realpath(code/'xfprobe'/'written.txt'), True, 'forbidden'),
                                                                              (os.path.realpath(code/'data'/'three_d'/'x'/'pair.npz'), False, 'forbidden'),
                                                                              (os.path.realpath(results/'other'/'disp.npz'), False, 'forbidden')]
    assert not (code/'xfprobe'/'written.txt').exists()


def test_spyrun_runs_a_method_script_that_lies_in_a_code_tree_inside_the_results_root(tmp_path):
    """The method process (a real subprocess under sci310): its script and a module it imports lie in the code tree inside the results root -- allowed; a write into the code tree stops it."""
    results, code = code_tree_in_results(tmp_path)
    work = results/'work'/'a'/'p'
    work.mkdir(parents=True)
    (work/'in.txt').write_text('fixed and moving')
    policy = both_policies(tmp_path, results, code)['method']
    (work/'policy.json').write_text(json.dumps(policy))
    name = fresh_module(code)
    script = code/'xfprobe'/'m_probe.py'
    script.write_text(f'import sys\nsys.path.insert(0, {str(code)!r})\nimport {name}\nopen(sys.argv[1]).read()\nopen("out.txt", "w").write("ok")\n')
    env = dict(os.environ, BL2_T_LAUNCH=repr(time.time()))
    run = lambda: subprocess.run(xb_spy.spyrun_argv(xc.PY, work/'policy.json', work/'spy.jsonl', script, [str(work/'in.txt')]), capture_output=True, text=True, env=env, cwd=work,   # noqa: E731
                                 timeout=120)
    done = run()
    record = xb_spy.method_spy_record(work/'spy.jsonl', expect=[work/'in.txt'])
    assert done.returncode == 0 and record['passed'] and (work/'out.txt').exists(), (done.stderr, record['breaches'])
    (work/'spy.jsonl').unlink()
    script.write_text(f'import sys\nopen(sys.argv[1]).read()\nopen({str(code/"xfprobe"/"written.txt")!r}, "w").write("no")\n')
    done = run()
    record = xb_spy.method_spy_record(work/'spy.jsonl', expect=[work/'in.txt'])
    assert done.returncode == xb_spy.SPYRUN_BREACH_EXIT and record['breaches'][0]['cls'] == 'forbidden' and record['breaches'][0]['write'] is True
    assert not (code/'xfprobe'/'written.txt').exists()


# ---- this repository's tree inside a forbidden root that holds it (the results root as the policy sees it) ------------------------------------------------------------
@pytest.fixture
def code_inside_a_forbidden_root(monkeypatch):
    """xb_spy.forbidden_roots plus the code tree's parent directory: every policy of the driver and of its method processes then has a forbidden root above the code root, as the results root
    is above <results>/p0_code_<sha> and <results>/holdout_B_v1/code.  Before XF each run below breached on its first read of the code tree."""
    real = xb_spy.forbidden_roots
    parent = str(xc.REPO.resolve().parent)
    monkeypatch.setattr(xb_spy, 'forbidden_roots', lambda **k: [*real(**k), parent])
    return parent


def test_the_p0_dry_run_with_the_code_tree_inside_a_forbidden_root(monkeypatch, tmp_path, code_inside_a_forbidden_root):
    """d1_expected.md section 4 with the code tree inside a forbidden root: the same two expected refusals, no spy breach (D1 had refused_spy_breach, exit 20)."""
    W.fake_checks(monkeypatch)
    monkeypatch.setattr(hc, 'code_identity', lambda run=None, repo=None: dict(commit='c'*40, clean=True, status='', untracked=[]))
    pair = hc.pair_name(*W.holdout_rows()[0])
    out = tmp_path/'dry'
    out.mkdir()
    assert xb_drive.main(['dry-run', '--arm', 'fireants_greedy', '--pair-set', 'holdout', '--pair', pair, '--out', str(out)]) == xc.EXIT['admission_rejected']
    doc = json.loads((out/'dry_run.json').read_text())
    assert doc['refused']['stage'] == 'unseal' and doc['spy']['breaches'] == []
    assert sorted(e['kind'] for e in doc['expected']) == ['no_unseal_record', 'rule_one_unpinned'] and doc['unexpected'] == [] and doc['p0_ok'] is True
    assert doc['spy']['n_paths'] > 0


def test_a_development_run_with_the_code_tree_inside_a_forbidden_root(monkeypatch, code_inside_a_forbidden_root):
    W.fake_checks(monkeypatch)
    W.write_stage_pair(hc.DEV_STAGE, W.DEV_PAIR)
    assert W.run_cell('fireants_greedy', 'dev', W.DEV_PAIR, deps=W.deps(step=W.FakeMethod())) == 0
    doc = result('fireants_greedy')
    assert doc['status'] == 'ok' and doc['spies']['driver']['passed'] and doc['spies']['driver']['breaches'] == []


def test_a_holdout_run_with_the_code_tree_inside_a_forbidden_root(tmp_path, monkeypatch, code_inside_a_forbidden_root):
    pair, original, sim = holdout_world(tmp_path, monkeypatch)
    assert W.run_cell('fireants_greedy', 'holdout', pair, deps=W.deps()) == 0
    doc = result('fireants_greedy', 'holdout', pair)
    assert doc['status'] == 'ok' and doc['spies']['driver']['passed'] and doc['spies']['driver']['n_sealed_list_reads'] >= 8
