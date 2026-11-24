"""Shared by the HI tests: the spy, the synthetic world (tiny members, a synthetic registry, allowlist, sealed list and a temporary git repository holding BZ's synthetic unseal record and decision log), and the
things the sandbox fixture (conftest.py) replaces, kept here as they were imported so that their own tests use the real ones.  Everything is in tmp_path; no real subject's file is opened."""
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
import torch

import bayesreg.io.sealed as sealed_module
import bayesreg.seg.unet.data as data_module
from bayesreg.eval.truth import SealedLists
from bayesreg.io.guard import load_dev_allowlist
from bayesreg.io.pair import REGISTRY_SCHEMA, load_segmenter_registry, subject_list_sha256
from bayesreg.seg.unet import synthetic
from bayesreg.seg.unet.calibrate import MEMBER_FORMAT
from bayesreg.seg.unet.data import OasisFiles
from bayesreg.seg.unet.groups import DEFAULT_GROUPS_PATH, load_grouping
from bayesreg.seg.unet.model import build_model
from bayesreg.seg.unet.train import file_sha256

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT/'experiments/hi') not in sys.path:
    sys.path.insert(0, str(ROOT/'experiments/hi'))
import hi_core as hi  # noqa: E402
import run_hi  # noqa: E402

REAL = dict(check_environment=hi.check_environment, machine_gate=hi.machine_gate, decision=hi.DEVCHECK_DECISION, lock=hi.HOLDOUT_LOCK, results=hi.RESULTS, devcheck_lock=hi.DEVCHECK_LOCK,
            devcheck_out=hi.DEVCHECK_OUT, pid_alive=hi.pid_alive, pinned_code_check=hi.pinned_code_check, holdout_out=hi.HOLDOUT_OUT_SUGGESTED)     # as imported, before any fixture
REAL_PINS = dict(REPO=sealed_module.REPO, SEALED_LISTS=sealed_module.SEALED_LISTS)     # bayesreg's pinned tree and sealed lists, as imported (C23.2-3)
RECORD = (ROOT/hi.UNSEAL_REL).exists()          # diff review n4: the unseal branch has BZ's record, main has none; the hold-out assertions follow it

SUBJECTS = [synthetic.oasis(901), synthetic.oasis(902)]          # the synthetic "hold-out", unsealed by a synthetic record
CONTACT = [synthetic.oasis(910), synthetic.oasis(911), synthetic.oasis(912)]      # the synthetic training set (both folds' label contact)
T0 = [synthetic.oasis(920)]
DEV = [synthetic.oasis(930)]
H1 = [synthetic.oasis(940)]
SHAPE_XYZ = (12, 16, 16)                        # stored (nz, ny, nx) = (16, 16, 12)
TINY = dict(base_channels=4, depth=2)
# (fold, seed, temperature, window, Gaussian sigma, amp) in HS's order A0 A1 B0 B1, every member different
MEMBERS = (('A', 0, 1.3, [8, 8, 8], .125, True), ('A', 1, .7, [16, 8, 8], .25, False), ('B', 0, 1.1, [8, 16, 12], .125, False), ('B', 1, 2., [16, 16, 8], .5, True))
TEST_ENVIRONMENT = dict(gpu='NVIDIA GeForce RTX 4090', torch='2.5.1', cuda='12.1', cudnn=90100, cudnn_benchmark=False, cudnn_deterministic=False)
GIT_ENV = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1', GIT_AUTHOR_NAME='hi-test', GIT_AUTHOR_EMAIL='hi-test@example.invalid', GIT_COMMITTER_NAME='hi-test',
               GIT_COMMITTER_EMAIL='hi-test@example.invalid')


def quiet(*args, **kwargs):
    pass


def write_json(path, document):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(document, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
    return Path(path)


def sha_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_p(path):
    with np.load(path, allow_pickle=False) as npz:
        return npz['p']


# ---- a temporary git repository ------------------------------------------------------------------------------------------------------------------
def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, text=True, env=GIT_ENV)


def init_repo(repo):
    Path(repo).mkdir(parents=True, exist_ok=True)
    git(repo, 'init', '-q')
    return Path(repo)


def commit_all(repo, message='test'):
    git(repo, 'add', '-A')
    git(repo, 'commit', '-q', '--allow-empty', '-m', message)


def log_record(repo, path, words=None):
    """Write the record's sha256 into the repository's decision log, with BZ's words on the same line (D98: one line holds both; `words` None: the record's own bz_words)."""
    if words is None:
        words = json.loads(Path(path).read_text(encoding='utf-8')).get('bz_words')
    log = Path(repo)/hi.DECISION_LOG_REL
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as stream:
        stream.write(f'| test | the unseal record {sha_of(path)} | {words} |\n')


# ---- bayesreg's pinned values (C23.9-13 (5): the tests change them with monkeypatch, never through an option) -----------------------------------
def pin(monkeypatch, repo, specs):
    """bayesreg.io.sealed.REPO (the code tree of the record, of its sealed lists and of the pinned allowlist) and SEALED_LISTS point into the test's git repository; the guard's cache of the lists is reset."""
    monkeypatch.setattr(sealed_module, 'REPO', Path(repo))
    monkeypatch.setattr(sealed_module, 'SEALED_LISTS', tuple(specs))
    sealed_module._reset_cache()


def real_pins(monkeypatch):
    """Back to the pinned values of the code tree (REAL_PINS)."""
    monkeypatch.setattr(sealed_module, 'REPO', REAL_PINS['REPO'])
    monkeypatch.setattr(sealed_module, 'SEALED_LISTS', REAL_PINS['SEALED_LISTS'])
    sealed_module._reset_cache()


# ---- the spy -------------------------------------------------------------------------------------------------------------------------------------
class Spy:
    """Every path opened through builtins.open and io.open (so np.load, gzip and nibabel) and through pathlib (Path.open, read_text, read_bytes: in Python 3.10 pathlib bound io.open into
    pathlib._NormalAccessor.open when it was imported, so replacing io.open alone does not see them -- diff review 2, s3; the accessor's open is replaced too, and test_the_spy_sees_every_way_of_opening checks it),
    every call of the image loader of bayesreg.seg.unet.data, every image or label path asked of `files`.  os.open (the locks' O_EXCL) and mmap are not watched."""

    def __init__(self, monkeypatch):
        self.opened, self.loads, self.asked = [], [], []
        real_load = data_module.load_scalar

        def load(path, *args, **kwargs):
            self.loads.append(str(path))
            return real_load(path, *args, **kwargs)
        monkeypatch.setattr(data_module, 'load_scalar', load)

        def opener(real):
            def wrapped(file, *args, **kwargs):
                if isinstance(file, (str, bytes, os.PathLike)):
                    self.opened.append(os.fsdecode(file))
                return real(file, *args, **kwargs)
            return wrapped
        monkeypatch.setattr(builtins, 'open', opener(builtins.open))
        monkeypatch.setattr(io, 'open', opener(io.open))
        accessor = getattr(pathlib, '_NormalAccessor', None)
        if accessor is not None and 'open' in vars(accessor):          # Python 3.10: Path.open -> self._accessor.open(self, ...)
            monkeypatch.setattr(accessor, 'open', staticmethod(opener(vars(accessor)['open'])))

    def files(self, root):
        spy = self

        class Files(OasisFiles):
            def image_path(self, subject):
                spy.asked.append(('image', subject))
                return super().image_path(subject)

            def label_path(self, subject, kind='seg35'):
                spy.asked.append(('label', subject))
                raise AssertionError(f'a label path was asked: {subject} {kind}')
        return Files(root=Path(root))

    def images(self):
        return [p for p in self.opened+self.loads if p.endswith(('.nii.gz', '.nii'))]

    def labels(self):
        return [p for p in self.opened+self.loads if 'seg35' in Path(p).name or 'seg4' in Path(p).name or Path(p).name == 'pair.npz' or any(w in Path(p).name for w in ('metrics', 'evaluation', 'labels.json'))]

    def nothing_read(self):
        return self.asked == [] and self.loads == [] and self.images() == []


# ---- the synthetic world --------------------------------------------------------------------------------------------------------------------------
def write_member(root, fold, seed, temperature, window, sigma, amp):
    torch.manual_seed(100+2*'AB'.index(fold)+seed)
    model = build_model(TINY)
    directory = Path(root)/f'fold{fold}_s{seed}'
    directory.mkdir(parents=True)
    torch.save(dict(format=MEMBER_FORMAT, state_dict=model.state_dict(), model_config=dict(TINY), temperature=temperature, fold=fold, seed=seed, best_iteration=1, label_contact_subjects=list(CONTACT),
                    label_contact_subjects_sha256=subject_list_sha256(CONTACT), model_best_sha256='0'*64, percentile=99.5, window=list(window), amp=amp, window_gaussian_sigma=sigma, checkpoint_choice='last',
                    checkpoint_iteration=40), directory/'member.pt')
    return file_sha256(directory/'member.pt')


def write_unseal(w, document, *, commit=True, log=True):
    """BZ's synthetic record at the pinned path, its sha256 and bz_words on one line of the decision log, both committed (unless told otherwise)."""
    write_json(w.unseal_path, document)
    if log:
        log_record(w.repo, w.unseal_path)
    if commit:
        commit_all(w.repo, 'the synthetic unseal record')


def world(tmp_path, *, role='holdout', monkeypatch=None):
    """role holdout: SUBJECTS unsealed (on the sealed list holdout_B_v1, in a never_for_training allowlist source, a valid committed unseal record); role dev: SUBJECTS development subjects (on no sealed list).
    C23.9-13 (5): the temporary git repository is bayesreg's pinned code tree for the holdout role (pin: its lists/holdout.txt is holdout_B_v1, its lists/h1.txt is H1, its configs/bayesreg/dev_subjects.json the
    pinned allowlist), so the unseal object comes from bayesreg's real loader.  w.allowlist and w.files are bound to it (bayesreg.io.unseal.bind_unseal, bind_files: what the entry uses after loading the record);
    w.allowlist_unbound and w.files_unbound are what the entry is handed (holdout_kwargs).  The dev role pins nothing (SUBJECTS are on no real sealed list)."""
    data = synthetic.write_subjects(tmp_path/'data', SUBJECTS, shape_xyz=SHAPE_XYZ)       # aligned_norm and (synthetic) aligned_seg35: the spy shows the labels are never opened
    models_root = tmp_path/'members'
    shas = {f'fold{f}_s{s}': write_member(models_root, f, s, t, w, g, a) for f, s, t, w, g, a in MEMBERS}
    registry_path = write_json(tmp_path/'segmenters_hi_test.json', dict(schema=REGISTRY_SCHEMA, comment='synthetic', models={
        f'unet35_v4_ens_fold{f}': dict(segmenter_id='unet35_v4_ens', fold=i, member_model_sha256=[shas[f'fold{f}_s0'], shas[f'fold{f}_s1']], label_contact_subjects=list(CONTACT),
                                       label_contact_subjects_sha256=subject_list_sha256(CONTACT)) for i, f in enumerate('AB')}))
    registry = load_segmenter_registry(registry_path)
    anchor = dict(hi.CONTRACT_SEGMENTER, registry_sha256=registry.sha256)
    holdout = role == 'holdout'
    if holdout and monkeypatch is None:
        raise AssertionError('world(role="holdout") pins bayesreg into its repository: pass monkeypatch')
    sources = [('synthetic_unsealed' if holdout else 'synthetic_dev_stage', SUBJECTS, holdout), ('synthetic_train', CONTACT, False), ('synthetic_T0', T0, True), ('synthetic_dev', DEV, False)]
    repo = init_repo(tmp_path/'repo')
    allowlist = load_dev_allowlist(write_json(repo/'configs'/'bayesreg'/'dev_subjects.json', synthetic.allowlist_document(sources)))
    lists = repo/'lists'
    lists.mkdir(parents=True)
    (lists/'holdout.txt').write_text(''.join(f'{s}\n' for s in SUBJECTS), encoding='utf-8')
    (lists/'h1.txt').write_text(''.join(f'{s}\n' for s in H1), encoding='utf-8')
    specs = (dict(name='holdout_B_v1', path='lists/holdout.txt', sha256=file_sha256(lists/'holdout.txt'), columns=None, count=len(SUBJECTS)),
             dict(name='H1', path='lists/h1.txt', sha256=file_sha256(lists/'h1.txt'), columns=None, count=len(H1)))
    if holdout:
        pin(monkeypatch, repo, specs)
        sealed = sealed_module.load_sealed_lists(repo)
    else:
        sealed = SealedLists({'holdout_B_v1': frozenset(), 'H1': frozenset(H1)}, [dict(name='holdout_B_v1'), dict(name='H1')])
    unseal = dict(schema=hi.UNSEAL_SCHEMA, written_by='BZ', bz_words='（合成测试）解封合成留出集', time_local='2026-11-07 5:30 AM CST', utc='2026-11-07T11:30:00Z', sealed_list='holdout_B_v1',
                  subjects_file='lists/holdout.txt', subjects_file_sha256=specs[0]['sha256'], subjects_sha256=subject_list_sha256(sorted(SUBJECTS)), n_subjects=len(SUBJECTS),
                  allowlist_source='synthetic_unsealed', allowlist_sha256=allowlist.sha256, segmenter_id='unet35_v4_ens', registry_sha256=registry.sha256, purpose='synthetic test of the hold-out entry')
    files = OasisFiles(root=data)
    w = types.SimpleNamespace(data=data, models_root=models_root, shas=shas, registry=registry, anchor=anchor, allowlist=allowlist, allowlist_unbound=allowlist, repo=repo, specs=specs, sealed=sealed,
                              unseal=unseal, unseal_path=repo/hi.UNSEAL_REL, grouping=load_grouping(DEFAULT_GROUPS_PATH, 'structure13_v1'), files=files, files_unbound=files, training=list(CONTACT),
                              tmp=tmp_path, hi_unseal=None)
    write_unseal(w, unseal)
    if holdout:
        w.hi_unseal = hi.load_unseal(w.unseal_path, anchor=anchor)
        w.allowlist, w.files = hi.bind(w.hi_unseal, allowlist, files)
    return w


def simulated_unseal_branch(tmp_path, monkeypatch):
    """C23.9-13 (2) and its condition on coverage: the unseal branch of runs/20261107-hi/入口说明.md §4 rebuilt in a temporary git repository, for the counter-examples on the real ids -- the four real sealed
    list files copied byte for byte (ids only; bayesreg's pinned sha256 unchanged), the repository's allowlist with a never_for_training source holding exactly the 100 subjects of holdout_B_v1 (added unless the
    allowlist has it, as on the unseal branch), a committed record and its line in the decision log.  bayesreg's tree is pinned to it; bayesreg's real loader makes the unseal object and bind_unseal binds it.
    Returns (allowlist_unbound, allowlist, unseal: HI's view, repo, source)."""
    repo = init_repo(Path(tmp_path)/'sim_repo')
    real_specs = REAL_PINS['SEALED_LISTS']
    for spec in real_specs:
        target = repo/spec['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/spec['path']).read_bytes())
    holdout_spec = {s['name']: s for s in real_specs}['holdout_B_v1']
    holdout = sorted(sealed_module.load_sealed_lists(repo).subjects['holdout_B_v1'])
    document = json.loads((ROOT/'configs/bayesreg/dev_subjects.json').read_text(encoding='utf-8'))
    source = next((s['name'] for s in document['sources'] if sorted(s['subjects']) == holdout and s['never_for_training'] is True), None)
    if source is None:
        source = 'holdout_B_v1_unsealed'
        document['sources'].append(dict(name=source, dataset='OASIS', path=holdout_spec['path'], sha256=holdout_spec['sha256'], never_for_training=True, subjects=holdout))
        document['subjects']['OASIS'] = list(document['subjects']['OASIS'])+[s for s in holdout if s not in document['subjects']['OASIS']]
    allowlist = load_dev_allowlist(write_json(repo/'configs'/'bayesreg'/'dev_subjects.json', document))
    record = dict(schema=hi.UNSEAL_SCHEMA, written_by='BZ', bz_words='（测试）模拟解封分支', time_local='2026-11-07 6:00 AM CST', utc='2026-11-07T12:00:00Z', sealed_list='holdout_B_v1',
                  subjects_file=holdout_spec['path'], subjects_file_sha256=holdout_spec['sha256'], subjects_sha256=subject_list_sha256(holdout), n_subjects=len(holdout), allowlist_source=source,
                  allowlist_sha256=allowlist.sha256, segmenter_id=hi.CONTRACT_SEGMENTER['segmenter_id'], registry_sha256=hi.CONTRACT_SEGMENTER['registry_sha256'], purpose='test: the counter-examples on the real ids')
    path = write_json(repo/hi.UNSEAL_REL, record)
    log_record(repo, path)
    commit_all(repo, 'the simulated unseal branch')
    pin(monkeypatch, repo, real_specs)
    unseal = hi.load_unseal(path)
    bound, _ = hi.bind(unseal, allowlist)
    return types.SimpleNamespace(allowlist_unbound=allowlist, allowlist=bound, unseal=unseal, repo=repo, source=source)


def holdout_kwargs(w, *, out_root, files=None, unseal_path='default', resume_path=None):
    """The entry's arguments: the pinned allowlist and the OasisFiles unbound (the entry binds them to the record it loads; C23.8-1)."""
    return dict(unseal_path=w.unseal_path if unseal_path == 'default' else unseal_path, out_root=out_root, files=w.files_unbound if files is None else files, device='cpu', allowlist=w.allowlist_unbound,
                allowlist_record=dict(allowlist_sha256=w.allowlist_unbound.sha256), sealed=w.sealed, training=w.training, registry=w.registry, models_root=w.models_root, grouping=w.grouping,
                code=dict(kind='test', commit='c'*40, clean=True), command=['run_hi.py', 'infer', '(test)'], repo=w.repo, anchor=w.anchor, resume_path=resume_path, log=quiet)


def devcheck_document(path=None, *, reading=None, environment=None, backends=('hs_a',), plan_sha256=None, subjects=None, lock=None):
    """A devcheck.json as the check writes it, for the decision (only the keys check_decision reads); by default at DEVCHECK_OUT/devcheck.json (the sandbox points DEVCHECK_OUT into tmp_path), with the line of
    DEVCHECK_LOCK as its lock record (diff review 2, s2)."""
    path = Path(hi.DEVCHECK_OUT)/hi.DEVCHECK_RECORD if path is None else path
    environment = dict(TEST_ENVIRONMENT, device='cuda') if environment is None else environment
    lock = dict(path=str(hi.DEVCHECK_LOCK), line=(hi.lock_lines(hi.DEVCHECK_LOCK) or [None])[0]) if lock is None else lock
    return write_json(path, dict(schema=hi.DEVCHECK_SCHEMA, backends=list(backends), environment=environment, reading=dict(text=hi.READ_SELECT.format(n=20) if reading is None else reading),
                                 plan=dict(path=hi.DEVCHECK_PLAN_REL, sha256=hi.DEVCHECK_PLAN_SHA256 if plan_sha256 is None else plan_sha256), subjects=list(hi.DEV_SUBJECTS if subjects is None else subjects),
                                 lock=lock))


def decision_for(path, **overrides):
    return dict(dict(devcheck_json=str(path), devcheck_sha256=sha_of(path), reading=hi.READ_SELECT.format(n=20), backend='hs_a', environment=dict(TEST_ENVIRONMENT), decided_by='test', review='test'), **overrides)
