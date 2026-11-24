"""Shared by the HP tests (runs/20261107-hp/任务书.md): the sandbox (every root of holdout_common and the two tools in tmp_path; holdout_common.TEST_SANDBOX_ENV set, so
that a real root is refused before anything is written (review I2); systemd-run replaced on PATH by a shim that exits 99; a tripwire on the real roots), a synthetic hold-out world -- a temporary git repository that bayesreg's pinned values point into (bayesreg.io.sealed.REPO and
SEALED_LISTS, by monkeypatch only; C23.9 preamble) holding synthetic sealed lists, a synthetic pair list, the pinned allowlist, BZ's synthetic unseal record and its
line in the decision log, so that the unseal object comes from bayesreg's real loader (the HI tests' "simulated unseal branch", tests/test_bayesreg/hi/_world.py) --
synthetic NIfTI subjects (aligned_norm and a synthetic aligned_seg35 that no tool may open), a development stage in the old format (label members included), fake
ANTs binaries, the decisions of the two development checks (fake records) and committed resume authorizations.  Synthetic ids only (OASIS_OAS1_09xx_MR1), except where a test reads the real id lists (ids only)."""
import builtins
import hashlib
import io
import json
import os
import pathlib
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT/'experiments'/'holdout', ROOT/'experiments'/'d0'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import holdout_common as hc  # noqa: E402
import hp_core as core  # noqa: E402
import prep_stage as prep  # noqa: E402
import atropos_holdout as ah  # noqa: E402
import atropos_seg  # noqa: E402

import bayesreg.io.sealed as S  # noqa: E402
from bayesreg.io import unseal as U  # noqa: E402
from bayesreg.io.guard import load_dev_allowlist  # noqa: E402
from bayesreg.io.pair import subject_list_sha256  # noqa: E402
from bayesreg.seg.unet import synthetic  # noqa: E402
from jd_curl.three_d.nifti import file_sha256, load_scalar  # noqa: E402

REAL_PINS = dict(REPO=S.REPO, SEALED_LISTS=S.SEALED_LISTS)
REAL_HC = dict(PAIRS_TSV_SHA256=hc.PAIRS_TSV_SHA256, N_PAIRS=hc.N_PAIRS, machine_gate=hc.machine_gate, AUTH_ROOT=hc.AUTH_ROOT, STAGE_ROOT=hc.STAGE_ROOT,
               ATROPOS_ROOT=hc.ATROPOS_ROOT, LOCK_ROOT=hc.LOCK_ROOT, RESULTS=hc.RESULTS)
REAL_CORE = dict(gate_identity=core.gate_identity, pinned_identity=core.pinned_identity)         # as imported, before the sandbox replaces them (A5a)
REAL_RESULTS = Path('/home/rmcfz/seq_vp_tag_results')
REAL_ROOTS = (REAL_RESULTS/'holdout_B_v1', REAL_RESULTS/'hp_devcheck_v1', Path('/mnt/d/seq_vp_tag_holdout_B_v1'))
oasis = synthetic.oasis
HOLDOUT = [oasis(901), oasis(902), oasis(903), oasis(904)]       # the synthetic holdout_B_v1
PAIRS = [(HOLDOUT[0], HOLDOUT[1]), (HOLDOUT[2], HOLDOUT[3])]
H1 = [oasis(940), oasis(941)]
H4 = [oasis(942)]
H3 = ['IBSR_01', 'IBSR_02']
DEV = [oasis(930), oasis(931), oasis(932)]
DEV_PAIRS = tuple(f'{DEV[i]}__{DEV[i+1]}' for i in range(2))
T0 = [oasis(920)]
TRAIN = [oasis(910)]
SHAPE_XYZ = (10, 12, 8)
SHAPE_ZYX = SHAPE_XYZ[::-1]
WORDS = '（合成测试）解封合成留出集'
CODE = dict(kind='test', commit='c'*40, clean=True, pinned_code=dict(passed=True, pinned='c'*40, problems=[]))     # a hold-out run's code: rule one passed (A5a; hp_core.check_pinned)
DEV_CODE = dict(kind='test', commit='d'*40, clean=True, check=dict(problems=[]))                                    # the development checks' code (P0): main adds the clean check
GIT_ENV = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1', GIT_AUTHOR_NAME='hp-test', GIT_AUTHOR_EMAIL='hp-test@example.invalid',
               GIT_COMMITTER_NAME='hp-test', GIT_COMMITTER_EMAIL='hp-test@example.invalid')


def quiet(*args, **kwargs):
    pass


def real_roots_present():
    return [str(p) for p in REAL_ROOTS if p.exists()]


def sandbox(tmp_path, monkeypatch):
    results = tmp_path/'results'
    holdout = results/'holdout_B_v1'
    monkeypatch.setattr(hc, 'RESULTS', results)
    monkeypatch.setattr(hc, 'HOLDOUT_ROOT', holdout)
    monkeypatch.setattr(hc, 'STAGE_ROOT', holdout/'stage')
    monkeypatch.setattr(hc, 'ATROPOS_ROOT', holdout/'atropos')
    monkeypatch.setattr(hc, 'LOCK_ROOT', holdout/'locks')
    monkeypatch.setattr(hc, 'BATCH_ROOT', holdout/'batches')
    monkeypatch.setattr(hc, 'BIG_ROOT', tmp_path/'d'/'seq_vp_tag_holdout_B_v1')
    monkeypatch.setattr(hc, 'DEV_STAGE', tmp_path/'dev_stage')
    monkeypatch.setenv(hc.TEST_SANDBOX_ENV, '1')                         # review I2: the real roots are refused before anything is written
    monkeypatch.setattr(hc, '_local_end', None)                           # A5a review I1: no unit end recorded by an earlier test of this process (the gate's local lower bound)
    monkeypatch.setattr(hc, 'machine_gate', lambda **kwargs: dict(passed=True, note='the tests query no machine', **kwargs))
    gates = []                                                           # A5a: rule two of a decision's code_commit and rule one of a hold-out run, replaced (their own tests run the real ones)
    monkeypatch.setattr(core, 'gate_identity', lambda commit, **kw: (gates.append(commit), (dict(passed=True, gate=commit, note='test: rule two replaced'), []))[1])
    monkeypatch.setattr(core, 'pinned_identity', lambda: (dict(passed=True, pinned=CODE['commit'], note='test: rule one replaced'), []))
    shim = tmp_path/'bin'
    shim.mkdir()
    (shim/'systemd-run').write_text('#!/bin/sh\nexit 99\n')
    (shim/'systemd-run').chmod(0o755)
    monkeypatch.setenv('PATH', f"{shim}:{os.environ.get('PATH', '')}")
    S._reset_cache()
    return types.SimpleNamespace(tmp=tmp_path, results=results, holdout=holdout, shim=shim, gates=gates)


# ---- a temporary git repository: bayesreg's pinned tree ------------------------------------------------------------------------------------------
def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, text=True, env=GIT_ENV)


def commit_all(repo, message='test'):
    git(repo, 'add', '-A')
    git(repo, 'commit', '-q', '--allow-empty', '-m', message)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def write_json(path, document):
    return write_text(path, json.dumps(document, indent=1, ensure_ascii=False)+'\n')


def pin(monkeypatch, repo, specs):
    monkeypatch.setattr(S, 'REPO', Path(repo))
    monkeypatch.setattr(S, 'SEALED_LISTS', tuple(specs))
    S._reset_cache()


def real_pins(monkeypatch):
    monkeypatch.setattr(S, 'REPO', REAL_PINS['REPO'])
    monkeypatch.setattr(S, 'SEALED_LISTS', REAL_PINS['SEALED_LISTS'])
    monkeypatch.setattr(hc, 'PAIRS_TSV_SHA256', REAL_HC['PAIRS_TSV_SHA256'])
    monkeypatch.setattr(hc, 'N_PAIRS', REAL_HC['N_PAIRS'])
    S._reset_cache()


def tsv_text(rows):
    return 'fixed\tmoving\n'+''.join(f'{f}\t{m}\n' for f, m in rows)


def log_line(repo, record_path, words):
    sha = hashlib.sha256(Path(record_path).read_bytes()).hexdigest()
    log = Path(repo)/U.DECISION_LOG_REL
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as stream:
        stream.write(f'| 2026-11-07 | the unseal record {sha} | {words} |\n')
    return sha


def make_world(tmp_path, monkeypatch, *, rows=None, record=True, extra_sources=(), unsealed_extra=(), data_subjects=None, decided=True, **changes):
    """The synthetic tree, pinned (bayesreg.io.sealed.REPO, SEALED_LISTS; holdout_common's pair-list sha256 and count, and its AUTH_ROOT), and the synthetic data root.
    `rows`: the pair list (default PAIRS); record False: no unseal record; decided: the two development checks' decisions filled from fake records (decide); `changes`:
    fields of the record."""
    tmp_path = Path(tmp_path)
    repo = tmp_path/'repo'
    repo.mkdir(parents=True)
    git(repo, 'init', '-q')
    holdout = HOLDOUT+list(unsealed_extra)
    write_text(repo/'lists/b.txt', ''.join(f'{s}\n' for s in holdout))
    write_text(repo/'lists/h1.txt', ''.join(f'{s}\n' for s in H1))
    write_text(repo/'lists/h4.txt', '# H4 (synthetic)\n'+''.join(f'{s}\n' for s in H4))
    write_text(repo/'lists/h3.tsv', 'fixed\tmoving\n'+f'{H3[0]}\t{H3[1]}\n')
    specs = (dict(name='holdout_B_v1', path='lists/b.txt', sha256=file_sha256(repo/'lists/b.txt'), columns=None, count=len(holdout)),
             dict(name='H1', path='lists/h1.txt', sha256=file_sha256(repo/'lists/h1.txt'), columns=None, count=len(H1)),
             dict(name='H4', path='lists/h4.txt', sha256=file_sha256(repo/'lists/h4.txt'), columns=None, count=len(H4)),
             dict(name='H3', path='lists/h3.tsv', sha256=file_sha256(repo/'lists/h3.tsv'), columns=('fixed', 'moving'), count=len(H3)))
    rows = list(PAIRS if rows is None else rows)
    tsv = write_text(repo/hc.PAIRS_TSV_REL, tsv_text(rows))
    sources = [('unsealed_B', holdout, True), ('stage_L2R_val_v1', DEV, False), ('t0', T0, True), ('train', TRAIN, False)]+list(extra_sources)
    allowlist = load_dev_allowlist(write_json(repo/U.ALLOWLIST_REL, synthetic.allowlist_document(sources)))
    w = types.SimpleNamespace(repo=repo, specs=specs, allowlist=allowlist, record_path=repo/U.UNSEAL_REL, document=None, tmp=tmp_path, rows=rows, tsv=tsv,
                              holdout=holdout)
    if record:
        w.document = dict(schema=U.UNSEAL_SCHEMA, written_by='BZ', bz_words=WORDS, time_local='2026-11-07 6:00 AM CST', utc='2026-11-07T12:00:00Z', sealed_list='holdout_B_v1',
                          subjects_file='lists/b.txt', subjects_file_sha256=specs[0]['sha256'], subjects_sha256=subject_list_sha256(sorted(holdout)), n_subjects=len(holdout),
                          allowlist_source='unsealed_B', allowlist_sha256=allowlist.sha256, segmenter_id='unet35_v4_ens', registry_sha256='a'*64,
                          purpose='synthetic test of the hold-out data preparation and Atropos entry (HP)')
        w.document.update(changes)
        write_json(w.record_path, w.document)
        log_line(repo, w.record_path, w.document['bz_words'])
    commit_all(repo, 'the synthetic tree')
    pin(monkeypatch, repo, specs)
    monkeypatch.setattr(hc, 'PAIRS_TSV_SHA256', file_sha256(tsv))
    monkeypatch.setattr(hc, 'N_PAIRS', len(rows))
    monkeypatch.setattr(hc, 'AUTH_ROOT', repo)
    w.data = tmp_path/'data'
    synthetic.write_subjects(w.data, list(dict.fromkeys(holdout+DEV+H1+T0+TRAIN if data_subjects is None else data_subjects)), shape_xyz=SHAPE_XYZ)
    monkeypatch.setattr(prep, 'DATA_ROOT', w.data)
    monkeypatch.setattr(prep, 'EXPECTED_SHAPE_ZYX', SHAPE_ZYX)
    monkeypatch.setattr(ah, 'N_SUBJECTS', len({s for r in rows for s in r}))
    if decided:
        w.decisions = decide(monkeypatch)
    return w


# ---- the decisions of the development checks (review I1): fake records of the two checks, the constants filled from them ----------------------------
PLAN_SHA = hashlib.sha256(b'a synthetic plan').hexdigest()


def fake_devcheck(module, *, reading, record_extra=None, plan_sha256=PLAN_SHA, code=None):
    """A development check's record and lock as the check writes them (schema, plan, reading, lock line, the code it ran on: A5a), in the sandbox's results root; returns the record's path."""
    code = DEV_CODE if code is None else code
    line = dict(tool=f'{module.SCRIPT} devcheck', out=str(module.devcheck_root()), plan_sha256=plan_sha256, code_commit=code['commit'], utc='2026-11-09T12:00:00Z', pid=1)
    lock = module.devcheck_lock()
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps(line)+'\n', encoding='utf-8')
    path = module.devcheck_root()/module.DEVCHECK_RECORD
    write_json(path, dict(schema=module.SCHEMA+'.devcheck', plan=dict(path=module.DEVCHECK_PLAN_REL, sha256=plan_sha256), reading=reading, lock_line=line, code=code,
                          **(record_extra or {})))
    return path


def decision_for(path, *, reading, bz_words=None, plan_sha256=PLAN_SHA, code_commit=None):
    """The decision constant as the reviewed commit would write it: a plain dict (contract C24.3-6 (a)) with hp_core.DECISION_KEYS, code_commit the commit the check ran on (A5a)."""
    return dict(devcheck_json=str(path), devcheck_sha256=file_sha256(path), reading=reading, plan_sha256=plan_sha256, code_commit=DEV_CODE['commit'] if code_commit is None else code_commit,
                decided_by='test', review='test', bz_words=bz_words)


def decide(monkeypatch, *, prep_reading=None, atropos_reading=None, bz_words=None):
    """Both decisions filled (the plans' sha256 pinned to PLAN_SHA): the prep check read 逐位相同 19/19 (or prep_reading), the Atropos check 逐位相同 2/2 (or
    atropos_reading)."""
    n = len(hc.DEV_PAIRS)
    prep_reading = prep.READ_EQUAL.format(n=n) if prep_reading is None else prep_reading
    atropos_reading = ah.READ_EQUAL.format(n=len(ah.DEV_SUBJECTS)) if atropos_reading is None else atropos_reading
    monkeypatch.setattr(prep, 'PREP_DEVCHECK_PLAN_SHA256', PLAN_SHA)
    monkeypatch.setattr(ah, 'ATROPOS_DEVCHECK_PLAN_SHA256', PLAN_SHA)
    p = fake_devcheck(prep, reading=prep_reading, record_extra=dict(n=n, n_equal=n))
    a = fake_devcheck(ah, reading=atropos_reading, record_extra=dict(subjects=list(ah.DEV_SUBJECTS)))
    decisions = dict(prep=decision_for(p, reading=prep_reading), atropos=decision_for(a, reading=atropos_reading, bz_words=bz_words))
    monkeypatch.setattr(prep, 'PREP_DEVCHECK_DECISION', decisions['prep'])
    monkeypatch.setattr(ah, 'ATROPOS_DEVCHECK_DECISION', decisions['atropos'])
    return decisions


# ---- committed resume authorizations (holdout_common.load_resume) --------------------------------------------------------------------------------
def write_resume(w, module, *, lock, previous_record, k=1, commit=True, **changes):
    """splits/bayesreg_v1/holdout_B_v1_<tool>_resume_<k>.json in the synthetic tree (holdout_common.AUTH_ROOT), committed unless told otherwise."""
    previous = Path(previous_record)
    document = dict(schema=f'bayesreg.{module.TOOL}.resume.v1', written_by="lead on BZ's behalf", words='（测试）补跑一次', time_local='2026-11-09 7:00 PM CST',
                    utc='2026-11-09T12:00:00Z', reason='test: an infrastructure failure', tool=module.TOOL, arm=module.ARM, pair=core.WHOLE_RUN, pair_set=core.PAIR_SET,
                    lock_sha256=file_sha256(lock), previous_record=str(previous), previous_record_sha256=file_sha256(previous) if previous.is_file() else None)
    document.update(changes)
    path = write_json(Path(w.repo)/hc.AUTH_DIRS['holdout']/f"{hc.AUTH_PREFIX['holdout']}{module.TOOL}_resume_{k}.json", document)
    if commit:
        commit_all(w.repo, 'a resume authorization')
    return path


def label_files(root):
    return sorted(str(p) for p in Path(root).rglob('aligned_seg*'))


# ---- a test-level spy (independent of the tools' own) ---------------------------------------------------------------------------------------------
class OpenSpy:
    """Every path opened through builtins.open, io.open and pathlib's accessor (Python 3.10), as tests/test_bayesreg/hi/_world.py's Spy; plus the image reader of
    prep_stage (load_scalar) and of nibabel."""

    def __init__(self, monkeypatch):
        self.opened, self.loads = [], []
        real = prep.load_scalar

        def load(path, *args, **kwargs):
            self.loads.append(str(path))
            return real(path, *args, **kwargs)
        monkeypatch.setattr(prep, 'load_scalar', load)

        def opener(real_open):
            def wrapped(file, *args, **kwargs):
                if isinstance(file, (str, bytes, os.PathLike)):
                    self.opened.append(os.fsdecode(file))
                return real_open(file, *args, **kwargs)
            return wrapped
        monkeypatch.setattr(builtins, 'open', opener(builtins.open))
        monkeypatch.setattr(io, 'open', opener(io.open))
        accessor = getattr(pathlib, '_NormalAccessor', None)
        if accessor is not None and 'open' in vars(accessor):
            monkeypatch.setattr(accessor, 'open', staticmethod(opener(vars(accessor)['open'])))

    def labels(self):
        return [p for p in self.opened+self.loads if 'aligned_seg' in Path(p).name or 'seg35' in Path(p).name or 'seg4' in Path(p).name]

    def images(self):
        return [p for p in self.opened+self.loads if p.endswith(('.nii.gz', '.nii'))]


# ---- fake ANTs ------------------------------------------------------------------------------------------------------------------------------------
FAKE_IMAGEMATH = '''#!{python} -I
import json, os, sys
import nibabel as nib, numpy as np
log = os.environ.get('HP_FAKE_ANTS_LOG')
if log:
    with open(log, 'a') as s: s.write(json.dumps(dict(tool='ImageMath', argv=sys.argv[1:], cwd=os.getcwd(), threads=os.environ.get('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS')))+'\\n')
_, dim, out, op, inp, lo, hi, bins, mask = sys.argv
img = nib.load(inp); a = np.asarray(img.dataobj, dtype=np.float64); m = np.asarray(nib.load(mask).dataobj) > 0
q = np.quantile(a[m], [float(lo), float(hi)])
t = nib.Nifti1Image(np.clip(a, q[0], q[1]).astype(np.float32), img.affine); t.header.set_xyzt_units('mm'); nib.save(t, out)
'''
FAKE_ATROPOS = '''#!{python} -I
import json, os, sys
import nibabel as nib, numpy as np
log = os.environ.get('HP_FAKE_ANTS_LOG')
if log:
    with open(log, 'a') as s: s.write(json.dumps(dict(tool='Atropos', argv=sys.argv[1:], cwd=os.getcwd(), threads=os.environ.get('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS')))+'\\n')
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
img = nib.load(args['-a']); a = np.asarray(img.dataobj, dtype=np.float64); m = np.asarray(nib.load(args['-x']).dataobj) > 0
labels_name, post_pattern = args['-o'].strip('[]').split(',')
qc_fail = bool(os.environ.get('HP_FAKE_ATROPOS_QC_FAIL'))         # one class with about 2 % of the mask, posteriors one-hot: atropos_seg's QC (>= 5 %) stops
edges = np.quantile(a[m], [.02, .5] if qc_fail else [1/3, 2/3])
cls = np.digitize(a, edges)
labels = np.where(m, cls+1, 0).astype(np.uint8)
t = nib.Nifti1Image(labels, img.affine); t.header.set_xyzt_units('mm'); nib.save(t, labels_name)
for k in range(3):
    p = np.where(m, np.where(cls == k, 1. if qc_fail else .8, 0. if qc_fail else .1), 0.).astype(np.float32)
    t = nib.Nifti1Image(p, img.affine); t.header.set_xyzt_units('mm'); nib.save(t, post_pattern % (k+1))
'''


def fake_ants(tmp_path, monkeypatch):
    """ImageMath and Atropos replaced by Python scripts (the tests run no ANTs): truncation at the in-mask quantiles, three classes by intensity tertiles, posteriors .8
    / .1 / .1 in the mask; each call's argv, working directory and ITK thread count appended to a log."""
    directory = Path(tmp_path)/'fake_ants'
    directory.mkdir()
    for name, text in (('ImageMath', FAKE_IMAGEMATH), ('Atropos', FAKE_ATROPOS)):
        (directory/name).write_text(text.format(python=sys.executable))
        (directory/name).chmod(0o755)
    log = Path(tmp_path)/'fake_ants.log'
    monkeypatch.setattr(atropos_seg, 'ANTS', directory)
    monkeypatch.setenv('HP_FAKE_ANTS_LOG', str(log))
    return log


def ants_calls(log):
    return [json.loads(line) for line in Path(log).read_text().splitlines()] if Path(log).is_file() else []


# ---- the development stage in the old format (label members and label-derived records included) ---------------------------------------------------
def write_old_stage(stage, data_root, pairs, *, perturb=None):
    """prepare_stage_3d_l2r_val.py's layout on synthetic subjects: pair.npz with fixed, fixed_seg4, fixed_seg35, moving, moving_seg4, moving_seg35, initial_u; a
    manifest with the label-derived records; selection.json.  perturb: (pair name, key) whose array gets one ulp more at one voxel."""
    stage = Path(stage)
    stage.mkdir(parents=True)
    rows = []
    for name in pairs:
        fixed, moving = name.split('__')
        out = stage/name
        out.mkdir()
        arrays, sources = {}, {}
        rng = np.random.default_rng(5)
        for side, subject in (('fixed', fixed), ('moving', moving)):
            v = load_scalar(Path(data_root)/subject/'aligned_norm.nii.gz')
            arrays[side] = np.array(v.values)
            arrays[f'{side}_seg4'] = rng.integers(0, 5, v.values.shape).astype(np.float64)
            arrays[f'{side}_seg35'] = rng.integers(0, 36, v.values.shape).astype(np.float64)
            sources[side] = dict(v.metadata)
            sources[f'{side}_seg4'] = dict(sha256='4'*64, labels=True)
            grid = v.grid
        arrays = {k: arrays[k] for k in ('fixed', 'fixed_seg4', 'fixed_seg35', 'moving', 'moving_seg4', 'moving_seg35')}
        arrays['initial_u'] = np.zeros(grid.shape+(3,))
        if perturb and perturb[0] == name:
            a = arrays[perturb[1]]
            a.flat[a.size//2] = np.nextafter(a.flat[a.size//2], np.inf)
        np.savez_compressed(out/'pair.npz', **arrays)
        manifest = dict(dataset='synthetic old stage', pair=name, l2r_fixed='OASIS_0000', l2r_moving='OASIS_0001', fixed_subject=fixed, moving_subject=moving,
                        mapping='fixed -> moving pullback', shape_zyx=list(grid.shape), spacing_xyz=list(grid.spacing), origin_xyz=[float(a[0]) for a in grid.axes], unit='mm',
                        preprocessing='x', sources=sources, pair_sha256=file_sha256(out/'pair.npz'), baseline_label_metrics=dict(seg35=dict(macro_dice=.5)),
                        baseline_native_mse=0., weights=dict(), labels_policy='seg4 primary')
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
        rows.append(dict(pair=name, fixed=str(Path(data_root)/fixed), moving=str(Path(data_root)/moving)))
    (stage/'selection.json').write_text(json.dumps(dict(rule='synthetic', pairs=rows, no_metric_based_selection=True)))
    return stage


def plan_file(w, rel, text='synthetic plan\n'):
    """A plan in the pinned tree, committed; returns its sha256."""
    path = write_text(Path(w.repo)/rel, text)
    commit_all(w.repo, 'plan')
    return file_sha256(path)
