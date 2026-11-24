"""Shared by the XB and hold-out tests: the sandbox (every root of holdout_common and xb_core in a directory of its own made by tempfile.mkdtemp(prefix='xbt_'), never under pytest's
basetemp (/tmp/pytest-of-<user>/... by default): those roots go into the units' start strings, which refuse the word pytest (xb_chain.unit_argv, xb_limit_experiment.trial_argv); XB_TEST_SANDBOX set so that holdout_common refuses a real root (I7), the committed
authorizations in a temporary git repository; systemd-run replaced on PATH by a shim that exits 99), synthetic stage pairs, SG's real functions on a simulated unseal branch (simulated_unseal), and a
fake method step.  No real subject's file is opened: the ids are real ids (lists of ids only), the arrays synthetic."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT/'experiments'/'xb', ROOT/'experiments'/'holdout'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import holdout_common as hc  # noqa: E402
import xb_core as xc  # noqa: E402
import xb_drive  # noqa: E402
import xb_spy  # noqa: E402

import bayesreg.io.sealed as sealed_module  # noqa: E402
from jd_curl.three_d.nifti import file_sha256  # noqa: E402

REAL_HI_PRED_ROOTS = dict(xc.HI_PRED_ROOTS)     # as imported, before any fixture replaces it (A5a review m7)
SHAPE = (12, 14, 10)                            # (nz, ny, nx)
DEV_PAIR = hc.DEV_PAIRS[0]                     # OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1: ids of the real allowlist, synthetic arrays


REAL_RESULTS = Path('/home/rmcfz/seq_vp_tag_results')
REAL_ROOTS = (REAL_RESULTS/'holdout_B_v1', REAL_RESULTS/'xb_dev_v1', REAL_RESULTS/'xb_limit_v1', Path('/mnt/d/seq_vp_tag_holdout_B_v1'))


def real_roots_present():
    """Which of the runner's real roots exist (a tripwire: no test may create one)."""
    return [str(p) for p in REAL_ROOTS if p.exists()]


def holdout_rows():
    return hc.holdout_pairs()


def roots_base():
    """A new directory for the sandbox's roots (tempfile.mkdtemp(prefix='xbt_'), under TMPDIR or /tmp): every path built from it goes into start strings, so it must not hold the word pytest
    (the start-string check is not relaxed).  The conftest removes it after the test (remove_roots_base)."""
    base = Path(tempfile.mkdtemp(prefix='xbt_'))
    if 'pytest' in str(base):
        shutil.rmtree(base, ignore_errors=True)
        raise RuntimeError(f'{base}: the temporary directory holds the word pytest (TMPDIR?); the start strings would be refused')
    return base


def remove_roots_base(base):
    """The test's roots, removed after it (only a directory roots_base made: its name starts with xbt_ and it is not a real root)."""
    base = Path(base)
    if base.name.startswith('xbt_') and hc.real_root_problem(base) is None and base.parent == Path(tempfile.gettempdir()):
        shutil.rmtree(base, ignore_errors=True)


def sandbox(tmp_path, monkeypatch, base=None):
    """The roots under `base` (the conftests pass roots_base(); tmp_path when a caller gives none), the systemd-run shim in tmp_path."""
    base = tmp_path if base is None else Path(base)
    results = base/'results'
    holdout = results/'holdout_B_v1'
    monkeypatch.setattr(hc, 'RESULTS', results)
    monkeypatch.setattr(hc, 'HOLDOUT_ROOT', holdout)
    monkeypatch.setattr(hc, 'STAGE_ROOT', holdout/'stage')
    monkeypatch.setattr(hc, 'ATROPOS_ROOT', holdout/'atropos')
    monkeypatch.setattr(hc, 'LOCK_ROOT', holdout/'locks')
    monkeypatch.setattr(hc, 'BATCH_ROOT', holdout/'batches')
    monkeypatch.setattr(hc, 'BIG_ROOT', base/'d'/'seq_vp_tag_holdout_B_v1')
    monkeypatch.setattr(hc, 'DEV_STAGE', base/'dev_stage')
    monkeypatch.setattr(xc, 'DEV_ROOT', results/'xb_dev_v1')
    monkeypatch.setattr(xc, 'HI_PRED_ROOTS', dict(holdout=results/'seg'/'holdout'/'pred', dev=results/'seg'/'dev'/'pred'))
    monkeypatch.setattr(xc, 'HS_RUNS', results/'hs_v1'/'runs')
    monkeypatch.setattr(hc, 'AUTH_ROOT', base/'auth_repo')           # the committed authorizations: a git repository of the test (auth_repo, authorize)
    monkeypatch.setenv(hc.TEST_SANDBOX_ENV, '1')                         # I7: holdout_common's root functions refuse the real roots
    monkeypatch.setattr(hc, '_local_end', None)                           # A5a review I1: no unit end recorded by an earlier test of this process (the gate's local lower bound)
    monkeypatch.delenv('XB_BATCH_DIR', raising=False)
    shim = tmp_path/'bin'
    shim.mkdir()
    (shim/'systemd-run').write_text('#!/bin/sh\nexit 99\n')
    (shim/'systemd-run').chmod(0o755)
    monkeypatch.setenv('PATH', f"{shim}:{os.environ.get('PATH', '')}")
    monkeypatch.delenv('XB_UNIT', raising=False)
    sealed_module._reset_cache()                   # C23.6-5: each driver starts with the guard's cache empty (a driver is a process of its own; the tests run in one: reset, as PQ's exception (13))
    return SimpleNamespace(tmp=tmp_path, base=base, results=results, holdout=holdout, shim=shim)


def images(shape=SHAPE, seed=0):
    rng = np.random.default_rng(seed)
    z, y, x = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing='ij')
    blob = np.exp(-(((z-5.)/3.)**2+((y-6.)/4.)**2+((x-4.)/2.5)**2))+.3*np.exp(-(((z-8.)/2.)**2+((y-3.)/2.)**2+((x-7.)/2.)**2))
    fixed = (blob+.01*rng.standard_normal(shape)).astype(np.float32)
    moving = np.roll(fixed, 1, axis=2).astype(np.float32)
    return fixed, moving


def write_stage_pair(stage, pair, *, seg=True, shape=SHAPE, seed=0, initial_u=True):
    """pair.npz as a stage writes it (fixed, moving float32; the four label members unless seg=False, as the hold-out stage of prereg revision 1; an all-zero initial_u) and its manifest.json with
    the loader's fields and a label-derived record (dropped by the loader's decoder)."""
    fixed_id, moving_id = pair.split('__')
    d = Path(stage)/pair
    d.mkdir(parents=True)
    fixed, moving = images(shape, seed)
    arrays = dict(fixed=fixed, moving=moving)
    if seg:
        rng = np.random.default_rng(7)
        for side in ('fixed', 'moving'):
            arrays[f'{side}_seg4'] = rng.integers(0, 5, shape).astype(np.int16)
            arrays[f'{side}_seg35'] = rng.integers(0, 36, shape).astype(np.int16)
    if initial_u:
        arrays['initial_u'] = np.zeros(shape+(3,))
    np.savez_compressed(d/'pair.npz', **arrays)
    manifest = dict(pair=pair, fixed_subject=fixed_id, moving_subject=moving_id, shape_zyx=list(shape), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], unit='mm',
                    sources=dict(fixed=dict(sha256='1'*64), moving=dict(sha256='2'*64), fixed_seg4=dict(sha256='3'*64)), pair_sha256=file_sha256(d/'pair.npz'),
                    baseline_label_metrics=dict(seg35=dict(macro_dice=.5)), labels_policy='synthetic')
    (d/'manifest.json').write_text(json.dumps(manifest))
    return d/'pair.npz'


# ---- SG's real functions on a simulated unseal branch (contract C23.9-13 (2), as tests/test_bayesreg/hi/_world.py simulated_unseal_branch) -----------------
GIT_ENV = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1', GIT_AUTHOR_NAME='xb-test', GIT_AUTHOR_EMAIL='xb-test@example.invalid', GIT_COMMITTER_NAME='xb-test',
               GIT_COMMITTER_EMAIL='xb-test@example.invalid')
REAL_PINS = dict(REPO=sealed_module.REPO, SEALED_LISTS=sealed_module.SEALED_LISTS)


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, text=True, env=GIT_ENV)


def write_json(path, document):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(document, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
    return Path(path)


def simulated_unseal(tmp_path, monkeypatch, *, record=True, written_by="lead on BZ's behalf", subjects=None, commit=True, log=True):
    """The unseal branch rebuilt in a temporary git repository: the four real sealed list files copied byte for byte (ids only; bayesreg's pinned sha256 unchanged), the repository's allowlist with a
    never_for_training source holding exactly the 100 subjects of holdout_B_v1, a record written as the lead would write it on BZ's behalf (its words contain 解封记录你代我写), its sha256 and words on
    one line of the decision log, committed.  bayesreg.io.sealed.REPO is pinned to it (and the guard's cache reset), so bayesreg.io.unseal.load_unseal makes the unseal object from it.
    Returns the repository, the record path and the unbound allowlist."""
    from bayesreg.io import unseal as unseal_module
    from bayesreg.io.guard import load_dev_allowlist
    from bayesreg.io.pair import subject_list_sha256
    repo = Path(tmp_path)/'sim_repo'
    repo.mkdir(parents=True)
    git(repo, 'init', '-q')
    for spec in REAL_PINS['SEALED_LISTS']:
        target = repo/spec['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/spec['path']).read_bytes())
    holdout_spec = {s['name']: s for s in REAL_PINS['SEALED_LISTS']}['holdout_B_v1']
    holdout = sorted(sealed_module.load_sealed_lists(repo).subjects['holdout_B_v1'])
    listed = sorted(holdout if subjects is None else subjects)
    document = json.loads((ROOT/'configs/bayesreg/dev_subjects.json').read_text(encoding='utf-8'))
    source = 'holdout_B_v1_unsealed'
    document['sources'].append(dict(name=source, dataset='OASIS', path=holdout_spec['path'], sha256=holdout_spec['sha256'], never_for_training=True, subjects=holdout))
    document['subjects']['OASIS'] = list(document['subjects']['OASIS'])+[s for s in holdout if s not in document['subjects']['OASIS']]
    allowlist = load_dev_allowlist(write_json(repo/unseal_module.ALLOWLIST_REL, document))
    words = '（测试）解封记录你代我写；BZ 过目清单（XB 测试的模拟解封分支）'
    path = repo/unseal_module.UNSEAL_REL
    if record:
        write_json(path, dict(schema=unseal_module.UNSEAL_SCHEMA, written_by=written_by, bz_words=words, time_local='2026-11-07 11:30 PM CST', utc='2026-11-07T23:30:00Z',
                              sealed_list='holdout_B_v1', subjects_file=holdout_spec['path'], subjects_file_sha256=holdout_spec['sha256'], subjects_sha256=subject_list_sha256(listed),
                              n_subjects=len(listed), allowlist_source=source, allowlist_sha256=allowlist.sha256, segmenter_id='unet35_v4_ens', registry_sha256=xc.V4SEG_ANCHOR['registry_sha256'],
                              purpose='test: the XB admission with the real loader'))
        if log:
            decision_log = repo/unseal_module.DECISION_LOG_REL
            decision_log.parent.mkdir(parents=True, exist_ok=True)
            decision_log.write_text(f'| test | the unseal record {hashlib.sha256(path.read_bytes()).hexdigest()} | {words} |\n', encoding='utf-8')
    if commit:
        git(repo, 'add', '-A')
        git(repo, 'commit', '-q', '--allow-empty', '-m', 'the simulated unseal branch')
    monkeypatch.setattr(sealed_module, 'REPO', repo)
    monkeypatch.setattr(sealed_module, 'SEALED_LISTS', REAL_PINS['SEALED_LISTS'])
    sealed_module._reset_cache()
    return SimpleNamespace(repo=repo, record=path, allowlist=allowlist, subjects=holdout)


# ---- fakes -------------------------------------------------------------------------------------------------------------------------------------
def fake_usage(free_gib=500.):
    return lambda path: SimpleNamespace(free=free_gib*2**30, total=2*free_gib*2**30, used=free_gib*2**30)


def fake_git(commit='c'*40, status='', untracked=''):
    def run(argv, **kwargs):
        if 'rev-parse' in argv:
            return subprocess.CompletedProcess(argv, 0, commit+'\n', '')
        if 'status' in argv:
            return subprocess.CompletedProcess(argv, 0, status, '')
        return subprocess.CompletedProcess(argv, 0, untracked, '')
    return run


def warp(moving, d):
    ident = np.meshgrid(*[np.arange(n) for n in moving.shape], indexing='ij')
    return map_coordinates(moving.astype(np.float64), ident+d, order=1, mode='constant', cval=0.)


class FakeMethod:
    """A method step for the tests: reads the derived npz named in its argv (and maps.npz for ConvexAdam-v4seg), writes disp.npz, own_warped.npz, method_out.json and its spy log, as m_*.py would.
    `returncode`, `events`, `d`, `own_scale` let a test make it fail, be killed, or be suspicious."""

    def __init__(self, *, returncode=0, events=(None, None), shift=(0., 0., 1.), own_scale=None, write=True, kernel=None, memory_guard=False):
        self.returncode, self.events, self.shift, self.own_scale, self.write, self.kernel, self.memory_guard = returncode, events, shift, own_scale, write, kernel, memory_guard
        self.calls = []

    def __call__(self, name, cmd, env, cwd, logfile, gpu, *, policy=None, log=print, **kwargs):
        cmd = [str(c) for c in cmd]
        self.calls.append(dict(name=name, cmd=cmd, env=dict(env), cwd=str(cwd), policy=policy))
        work = Path(cwd)
        inputs = next((Path(c) for c in cmd if c.endswith('inputs.npz')), None)
        if self.write and self.returncode == 0 and inputs is not None:
            with np.load(inputs) as z:
                moving = np.array(z['moving'], np.float64)
            d = np.stack([np.full(moving.shape, s) for s in self.shift])
            own = warp(moving, d)
            if self.own_scale is not None:
                own = own*0+np.random.default_rng(3).standard_normal(own.shape)*self.own_scale
            np.savez_compressed(work/'disp.npz', d=d.astype(np.float32))
            np.savez_compressed(work/'own_warped.npz', w=own.astype(np.float32))
            (work/'method_out.json').write_text(json.dumps(dict(seconds_registration_cold=1., seconds_registration_warm=None, seconds_total=2., device='cuda (fake)', threads={}, config={}, warnings=[])))
            (work/'fixed.nii.gz').write_bytes(b'an intermediate the method wrote')
        log_path = next((Path(c) for c in cmd if c.endswith('spy_method.jsonl')), None)
        if log_path is not None and inputs is not None:
            rows = [dict(event='start'), dict(path=os.path.realpath(inputs), write=False, cls='allowed')]
            rows += [dict(path=os.path.realpath(c), write=False, cls='allowed') for c in cmd if c.endswith('maps.npz')]
            log_path.write_text(''.join(json.dumps(r)+'\n' for r in rows+[dict(event='end', breaches=0)]))
        before, after = self.events
        return dict(name=name, cmd=cmd, cwd=str(cwd), returncode=self.returncode, seconds_wall=.1, min_mem_available_gb=10., memory_guard_killed=self.memory_guard, memory_events_before=before,
                    memory_events_after=after, kernel_oom_lines=self.kernel, fd_spy=dict(passed=True, samples=1, breaches=[]), spy_breach=None)


def fake_precheck(gpu, log=None):
    return dict(passed=True, note='test')


PINNED = 'c'*40


def pinned_ok():
    """Rule one passing (holdout_common.pinned_code_problems's record, as far as the runners read it): P1 = PINNED.  Its own tests run the real one (tests/test_bayesreg/holdout/test_pinned_code.py)."""
    return dict(passed=True, pinned=PINNED, head=PINNED, problems=[], note='test: rule one replaced'), []


def deps(step=None, **kwargs):
    kwargs.setdefault('pinned', pinned_ok)
    return xb_drive.Deps(gate=lambda: dict(passed=True, note='the tests query no machine'), disk_usage=fake_usage(), code_run=fake_git(), step=step or FakeMethod(), precheck=fake_precheck,
                         log=lambda *a: None, probe=lambda when: dict(when=when, note='test'), **kwargs)


def pinned_repo(path):
    """A temporary code tree on which rule one passes (bayesreg.io.code_identity.unit_identity_problems): P1 a commit with a code file and the decision log; the next commit adds
    runs/holdout_B_v1/pinned_code.json naming P1 and appends its decision_log_line to the log (C24.3-9: one commit); a clean working tree.  Returns (repo, P1)."""
    from bayesreg.io import code_identity as ci
    repo = Path(path)
    (repo/'experiments').mkdir(parents=True)
    (repo/'experiments'/'tool.py').write_text('X = 1\n')
    (repo/'docs').mkdir()
    (repo/ci.LATER_DECISION_LOG).write_text('log\n')
    (repo/'runs').mkdir()
    (repo/'runs'/'README.md').write_text('records\n')            # runs/ is a top-level directory of P1 (a new one would be code)
    git(repo, 'init', '-q')
    git(repo, 'add', '-A')
    git(repo, 'commit', '-q', '-m', 'P1')
    p1 = git(repo, 'rev-parse', 'HEAD').stdout.strip()
    line = f'| 2026-11-10 | P1 = {p1} (test) |'
    write_json(repo/ci.PINNED_RECORD_REL, dict(schema=ci.PINNED_RECORD_SCHEMA, commit=p1, decision_log_line=line, written_by='lead', time_local='2026-11-10 11:30 AM CST',
                                               utc='2026-11-10T17:30:00Z', reason='test'))
    with open(repo/ci.LATER_DECISION_LOG, 'a', encoding='utf-8') as stream:
        stream.write(line+'\n')
    git(repo, 'add', '-A')
    git(repo, 'commit', '-q', '-m', 'the pinned record')
    return repo, p1


def commit_rejudgment(arm, pair, record, *, judgment, k=1, pair_set='holdout', written_by='lead', words='', commit=True, bound=None, **override):
    """A rejudgment record (prereg revision 2 item 18; holdout_common.rejudgment) at runs/holdout_B_v1/rejudgments/<arm>/<pair>_<k>.json of the sandbox's code tree (hc.AUTH_ROOT), bound to
    `record` by its sha256 now (or `bound`, the record field as given), committed unless told otherwise."""
    repo = auth_repo()
    document = dict(schema=hc.REJUDGMENT_SCHEMA, arm=arm, pair=pair, pair_set=pair_set, status='needs_lead_rejudgment',
                    record=dict(path=str(record), sha256=file_sha256(record)) if bound is None else bound,
                    evidence=[], judgment=judgment, reason='test: the evidence could not be read', written_by=written_by, words=words, time_local='2026-11-10 11:00 AM CST',
                    utc='2026-11-10T17:00:00Z')
    document.update(override)
    path = write_json(repo/hc.REJUDGMENT_DIR_REL/arm/f'{pair}_{k}.json', document)
    if commit:
        git(repo, 'add', '-A')
        git(repo, 'commit', '-q', '-m', f'rejudgment {arm} {pair} {k}')
    return path


def fake_checks(monkeypatch):
    """check_frozen and check_environment replaced (their own tests run the real ones): a run test hashes no baseline file and runs no conda."""
    monkeypatch.setattr(xc, 'check_frozen', lambda arm, **kw: dict(config=dict(path=xc.FROZEN_CONFIG_REL, sha256=xc.FROZEN_CONFIG_SHA256), arm=arm, passed=True, note='test'))
    monkeypatch.setattr(xc, 'check_environment', lambda arm, **kw: dict(arm=arm, passed=True, note='test'))


def write_prediction(root, subject, labels, *, anchor=xc.V4SEG_ANCHOR, shape=SHAPE, override=None):
    """probs36.npz (float16, one-hot-ish around `labels`) and provenance.json as HI writes them."""
    d = Path(root)/subject
    d.mkdir(parents=True)
    p = np.full((36,)+shape, .01/35, dtype=np.float64)
    for k in range(36):
        p[k][labels == k] = .99
    p = (p/p.sum(0)).astype(np.float16)
    np.savez_compressed(d/'probs36.npz', p=p)
    prov = dict(subject=subject, layer='structure', segmenter_id=anchor['segmenter_id'], policy=anchor['policy'], registry_sha256=anchor['registry_sha256'], test_labels_seen=False,
                output_sha256={'probs36.npz': file_sha256(d/'probs36.npz'), 'structure_groups.npz': '0'*64}, outputs=dict(probs36=dict(shape=[36, *shape])), hi=dict(schema='test'))
    prov.update(override or {})
    (d/'provenance.json').write_text(json.dumps(prov))
    return d


def auth_repo():
    """hc.AUTH_ROOT of the sandbox as a git repository (made on first use)."""
    repo = Path(hc.AUTH_ROOT)
    if not (repo/'.git').exists():
        repo.mkdir(parents=True, exist_ok=True)
        git(repo, 'init', '-q')
    return repo


def commit_authorization(pair_set, kind, document, *, k=None, commit=True):
    """Write an authorization where the runner looks for it (hc.authorization_dir) and commit it (commit=False: written, not committed)."""
    repo = auth_repo()
    directory = hc.authorization_dir(pair_set)
    if k is None:
        k = 1
        while (directory/f'{hc.AUTH_PREFIX[pair_set]}xb_{kind}_{k}.json').exists():
            k += 1
    path = write_json(directory/f'{hc.AUTH_PREFIX[pair_set]}xb_{kind}_{k}.json', document)
    if commit:
        git(repo, 'add', '-A')
        git(repo, 'commit', '-q', '-m', f'{kind} {k}')
    return path


def authorize(kind, arm, pair_set, pair, *, commit=True, **override):
    """The driver's own draft (xb_drive.authorization_draft, the lead on BZ's behalf), committed in the sandbox's authorization repository; `override` changes fields of the draft."""
    _, document = xb_drive.authorization_draft(kind, arm, pair_set, pair, written_by="lead on BZ's behalf", words='（测试）BZ：可以', time_local='2026-11-09 1:00 PM CST', reason='test')
    document.update(override)
    return commit_authorization(pair_set, kind, document, commit=commit)


def refusals(pair_set='dev'):
    """Every refusal record of the sandbox's record root (I1)."""
    return sorted(Path(xc.roots(pair_set)[0]).glob('batches/*/refusals/*.json'))


def method_run(pair_set, pair, labels, **override):
    """The method's T88 run.json at its pinned path (xb_core.method_t88_run) with the labels_sha256 of `labels` (the maps as the v4 argmax gives them; I4)."""
    path = xc.method_t88_run(pair_set, pair)
    fixed, moving = pair.split('__')
    digest = xc.labels_digest({'fixed': np.asarray(labels[fixed], np.int16), 'moving': np.asarray(labels[moving], np.int16)})
    return write_json(path, dict(dict(schema='test.t88.run', status='ok', labels_sha256=digest), **override))


def run_cell(*args, **kwargs):
    """xb_drive.run_cell as a fresh driver process would start it: the guard's cache of the sealed lists empty (C23.6-5; each driver runs in a unit of its own, the tests run in one process: the cache
    is reset before each call, as PQ's exception (13))."""
    sealed_module._reset_cache()
    return xb_drive.run_cell(*args, **kwargs)
