"""HI revision 1 (docs/bayesreg/reviews/HI_工具_复核_20261107.md): the code root (I1), the git checks of the unseal record (I2), the decision of the development check and the environment (I3), run once and the resume record
(I4), the machine gate (m4).  CPU only, tmp_path only: the lock, the results root and the decision point into tmp_path (conftest.py); the record lives in a temporary git repository.  No real subject's file is opened."""
import json
from pathlib import Path
import subprocess
import types

import pytest
import torch

from bayesreg.io.pair import subject_list_sha256
from bayesreg.seg.unet.snapshot import CodeNotClean

from ._world import (REAL, ROOT, SUBJECTS, TEST_ENVIRONMENT, Spy, commit_all, decision_for, devcheck_document, git, hi, holdout_kwargs, run_hi, sha_of, world, write_json)


@pytest.fixture
def w(tmp_path, decided, monkeypatch):
    return world(tmp_path, monkeypatch=monkeypatch)


def load(w, path=None):
    """C23.9-13 (5): through bayesreg's loader on the pinned tree (the world pins it into its repository); no repo, specs, sealed or allowlist passed."""
    return hi.load_unseal(w.unseal_path if path is None else path, anchor=w.anchor)


# ---- I1: the code root ---------------------------------------------------------------------------------------------------------------------------
def test_the_code_root_must_be_this_entrys_tree():
    assert hi.check_code_root(dict(root=str(ROOT), imports_inside_root=True))['root'] == str(ROOT.resolve())
    other = ROOT.parent/f'{ROOT.name}_another_tree_never_made'          # a tree that is not ROOT wherever the tests run (the main checkout or a worktree; lead ~6:34 PM CST 11-10)
    assert other.resolve() != ROOT.resolve()
    with pytest.raises(CodeNotClean, match='the code record is of'):
        hi.check_code_root(dict(root=str(other), imports_inside_root=True), repo=ROOT)
    with pytest.raises(CodeNotClean, match='not imported from inside'):
        hi.check_code_root(dict(root=str(ROOT), imports_inside_root=False))
    with pytest.raises(CodeNotClean):
        hi.check_code_root({})


def test_real_inputs_refuse_another_code_root_before_reading_the_inputs(monkeypatch, tmp_path):
    """real_inputs checks the root first: with a code record of another tree, the pinned allowlist (and everything after it) is never read."""
    import bayesreg.seg.unet.pins as pins
    import bayesreg.seg.unet.snapshot as snapshot
    monkeypatch.setattr(snapshot, 'code_record', lambda *a, **k: dict(kind='git_worktree', clean=True, commit='f'*40, root=str(tmp_path), imports_inside_root=True))
    monkeypatch.setattr(pins, 'load_pinned_allowlist', lambda *a, **k: (_ for _ in ()).throw(AssertionError('the allowlist was read')))
    with pytest.raises(CodeNotClean, match='the code record is of'):
        run_hi.real_inputs()
    lines = []
    assert run_hi.dry_run_infer(types.SimpleNamespace(unseal=None, resume=None, out=None, subjects_file=None), out=lines.append) == 9 and any('CodeNotClean' in line for line in lines)


# ---- I2: committed and unchanged, before a byte is read ------------------------------------------------------------------------------------------
def untrack(w):
    """The record stays on disk, git forgets it (committed without it; commit_all would add it again)."""
    git(w.repo, 'rm', '-q', '--cached', hi.UNSEAL_REL)
    git(w.repo, 'commit', '-q', '-m', 'untrack the record')


def ignore(w):
    untrack(w)
    (w.repo/'.gitignore').write_text(hi.UNSEAL_REL+'\n', encoding='utf-8')
    git(w.repo, 'add', '.gitignore')
    git(w.repo, 'commit', '-q', '-m', 'ignore it')


CASES = dict(
    untracked=untrack,
    ignored=ignore,
    staged_not_committed=lambda w: (untrack(w), git(w.repo, 'add', hi.UNSEAL_REL)),
    modified=lambda w: w.unseal_path.write_text(w.unseal_path.read_text(encoding='utf-8')+' ', encoding='utf-8'),
)


def test_a_committed_unchanged_record_passes(w):
    unseal = load(w)
    assert unseal.git['record']['path'] == hi.UNSEAL_REL and unseal.git['record']['last_commit'] and unseal.git['decision_log']['path'] == hi.DECISION_LOG_REL


@pytest.mark.parametrize('case', sorted(CASES))
def test_a_record_that_is_not_committed_and_unchanged_is_refused_before_it_is_read(w, monkeypatch, case):
    CASES[case](w)
    status = git(w.repo, 'status', '--porcelain', '--ignored', '--', hi.UNSEAL_REL).stdout.rstrip('\n')
    assert status[:2] == dict(untracked='??', ignored='!!', staged_not_committed='A ', modified=' M')[case], status
    spy = Spy(monkeypatch)
    with pytest.raises(hi.HIRefused, match='not a committed, unchanged file') as refused:
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=w.tmp/'out', files=spy.files(w.data)))
    assert refused.value.stage == 'unseal' and not [p for p in spy.opened if p.endswith('unseal_holdout_B_v1.json')] and spy.nothing_read() and not (w.tmp/'out').exists()


def test_a_record_outside_the_tree_is_a_refusal_of_stage_unseal(w, monkeypatch):
    outside = write_json(w.tmp/'elsewhere'/hi.UNSEAL_REL, w.unseal)
    spy = Spy(monkeypatch)
    with pytest.raises(hi.HIRefused) as refused:
        load(w, outside)
    assert refused.value.stage == 'unseal' and str(outside) not in spy.opened
    with pytest.raises(hi.HIRefused, match='not inside the code tree') as refused:
        hi.tracked_unchanged(outside, w.repo)
    assert refused.value.stage == 'unseal'


@pytest.mark.parametrize('name', ['*.json', ':(glob)splits', 'splits/*', '[a-z]*'])
def test_pathspec_magic_in_a_file_name_is_taken_literally(w, name):
    """--literal-pathspecs: an untracked file named like a pattern is refused (ls-files does not match the tracked files the pattern would)."""
    target = w.repo/name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{}\n', encoding='utf-8')
    with pytest.raises(hi.HIRefused, match='not a committed, unchanged file'):
        hi.tracked_unchanged(target, w.repo)
    assert hi.tracked_unchanged(w.repo/'lists/holdout.txt', w.repo)['tracked'] is True


# ---- I3: the decision of the development check, cuda, the environment ----------------------------------------------------------------------------
REAL_DEVCHECK_JSON = Path('/home/rmcfz/seq_vp_tag_results/seg/hi_devcheck_v1/devcheck.json')
DECIDED = dict(devcheck_json=str(REAL_DEVCHECK_JSON), devcheck_sha256='0deba0f8f44b98fb9f255a843d8ba3d09e2bc5f452f9ef724c0b3f45ef19b4e4', reading='逐位相同 20/20：选 hs_a', backend='hs_a',
               environment=dict(gpu='NVIDIA GeForce RTX 4090', torch='2.5.1', cuda='12.1', cudnn=90100, cudnn_benchmark=False, cudnn_deterministic=False))     # the lead's numbers of 11-07 (devcheck_result.md)


DECISION_REVIEW = 'docs/bayesreg/reviews/HI_决定_差异复核_20261107.md'       # the diff review of hi-decide (afd44a5, 0/0/2), in the repository


def test_the_decision_in_the_code_is_the_development_check():
    """Changed with the commit that fills DEVCHECK_DECISION (runs/20261107-hi/入口说明.md §1; replaces test_nothing_is_decided_in_the_code): the decision is filled with the check of 2026-11-07 (values written here again,
    from runs/20261107-hi/devcheck_result.md), and its review field names the finished diff review of that commit, a file of the repository (decision diff review u1)."""
    decision = REAL['decision']
    assert decision is not None and sorted(decision) == sorted(hi.DECISION_KEYS) and REAL['lock'] == Path('/home/rmcfz/seq_vp_tag_results/seg/hi_holdout_infer.lock')
    assert {k: (dict(decision[k]) if k == 'environment' else decision[k]) for k in DECIDED} == DECIDED
    assert sorted(decision['environment']) == sorted(hi.DECISION_ENVIRONMENT_KEYS) == sorted(['gpu', 'torch', 'cuda', 'cudnn', 'cudnn_benchmark', 'cudnn_deterministic'])
    assert decision['decided_by'] == 'lead (total responsible), 2026-11-07, per devcheck_plan.md §3'
    assert decision['review'].startswith(DECISION_REVIEW+' ') and 'pending' not in decision['review'] and (ROOT/DECISION_REVIEW).is_file()


def test_the_decision_holds_against_the_real_development_check(monkeypatch):
    """Decision diff review u1: where either the real devcheck.json or its lock is there, both must be (a missing one is lost evidence, not a reason to pass quietly) and the decision must pass check_decision
    against them, read only, their sha256 unchanged.  Where neither is (another machine, a fresh checkout), the test says so and is skipped: the constants are checked by the test above."""
    devcheck_json, lock = REAL_DEVCHECK_JSON, Path(REAL['devcheck_lock'])
    if not devcheck_json.exists() and not lock.exists():
        pytest.skip(f'neither {devcheck_json} nor {lock} is on this machine: the real development check is not here, only the constants of the decision are checked')
    assert devcheck_json.is_file() and lock.is_file(), f'only one of {devcheck_json} ({devcheck_json.exists()}) and {lock} ({lock.exists()}) is there: evidence of the check is missing'
    monkeypatch.setattr(hi, 'DEVCHECK_OUT', REAL['devcheck_out'])
    monkeypatch.setattr(hi, 'DEVCHECK_LOCK', lock)
    before = sha_of(devcheck_json), sha_of(lock)
    record = hi.check_decision(REAL['decision'])
    assert record['reading'] == '逐位相同 20/20：选 hs_a' and (sha_of(devcheck_json), sha_of(lock)) == before
    assert before == ('0deba0f8f44b98fb9f255a843d8ba3d09e2bc5f452f9ef724c0b3f45ef19b4e4', '7895bf34002aade2a2bdc8c9d77da7420964112b7b388ce6382d18428f7fa3d8')
    checked = json.loads(devcheck_json.read_text(encoding='utf-8'))['environment']
    assert {k: checked[k] for k in hi.DECISION_ENVIRONMENT_KEYS} == DECIDED['environment'] and checked['device'] == 'cuda'


def test_without_a_decision_infer_refuses_before_any_subject(w, monkeypatch):
    monkeypatch.setattr(hi, 'DEVCHECK_DECISION', None)
    spy = Spy(monkeypatch)
    with pytest.raises(hi.HIRefused, match='DEVCHECK_DECISION is None') as refused:
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=w.tmp/'out', files=spy.files(w.data)))
    assert refused.value.stage == 'decision' and spy.nothing_read() and not (w.tmp/'out').exists() and not Path(hi.HOLDOUT_LOCK).exists()


def unlinked(decision):
    Path(decision['devcheck_json']).unlink()
    return decision


BAD_DECISIONS = dict(
    sha=(lambda t: decision_for(devcheck_document(), devcheck_sha256='0'*64), 'not the sha256'),
    missing_file=(lambda t: unlinked(decision_for(devcheck_document())), 'missing'),
    elsewhere=(lambda t: decision_for(devcheck_document(t/'elsewhere'/hi.DEVCHECK_RECORD)), 'not the one development check'),
    backend=(lambda t: decision_for(devcheck_document(), backend='segment_subject'), 'runs .hs_a. only'),
    keys=(lambda t: dict(decision_for(devcheck_document()), extra=1), 'exactly the keys'),
    environment_keys=(lambda t: decision_for(devcheck_document(), environment=dict(gpu='x')), 'environment: exactly the keys'),
    reading_changed=(lambda t: decision_for(devcheck_document(reading='逐位相同 20/20：选 hs_a（改过）'), reading='逐位相同 20/20：选 hs_a（改过）'), 'only .逐位相同 20/20：选 hs_a. lets'),
    reading_not_the_checks=(lambda t: decision_for(devcheck_document(reading=hi.READ_PENDING)), 'is not the check'),
    pending=(lambda t: decision_for(devcheck_document(reading=hi.READ_PENDING), reading=hi.READ_PENDING), 'only .逐位相同 20/20：选 hs_a. lets'),
    undecided=(lambda t: decision_for(devcheck_document(reading=hi.READ_UNDECIDED), reading=hi.READ_UNDECIDED), 'only .逐位相同 20/20：选 hs_a. lets'),
    plan=(lambda t: decision_for(devcheck_document(plan_sha256='0'*64)), 'the check ran under the plan sha256'),
    subjects=(lambda t: decision_for(devcheck_document(subjects=hi.DEV_SUBJECTS[:19])), 'not the 20 development subjects'),
    check_on_cpu=(lambda t: decision_for(devcheck_document(environment=dict(TEST_ENVIRONMENT, device='cpu'))), 'environment'),
    lock_line=(lambda t: decision_for(devcheck_document(lock=dict(path=str(hi.DEVCHECK_LOCK), line=dict(out='another run')))), "lock record"),
    lock_path=(lambda t: decision_for(devcheck_document(lock=dict(path=str(t/'other.lock'), line=hi.lock_lines(hi.DEVCHECK_LOCK)[0]))), "lock record"),
    lock_missing=(lambda t: (hi.DEVCHECK_LOCK.unlink(), decision_for(devcheck_document(lock=dict(path=str(hi.DEVCHECK_LOCK), line=None))))[1], "lock record"),
    environment_not_the_checks=(lambda t: decision_for(devcheck_document(environment=dict(TEST_ENVIRONMENT, device='cuda', gpu='another GPU'))), 'environment'),
    no_hs_a_in_the_check=(lambda t: decision_for(devcheck_document(backends=('segment_subject',))), 'backends'),
)


@pytest.mark.parametrize('name', sorted(BAD_DECISIONS))
def test_a_decision_that_does_not_hold_is_refused(tmp_path, name, decided):
    """Diff review R1 included: only the select reading, the pinned plan, the 20 development subjects and the one devcheck.json at DEVCHECK_OUT."""
    make, words = BAD_DECISIONS[name]
    decision = make(tmp_path)
    with pytest.raises(hi.HIRefused, match=words) as refused:
        hi.check_decision(decision)
    assert refused.value.stage == 'decision'


def test_the_sandbox_decision_holds(decided):
    assert hi.check_decision(decided.decision)['reading'] == '逐位相同 20/20：选 hs_a'


def test_a_pending_reading_cannot_be_decided(tmp_path, decided):
    """Diff review R1 (tightened from revision 1): under the pinned plan 不过（待查） leaves DEVCHECK_DECISION None (devcheck_plan.md §4); a decision that copies it is refused.  Another way needs a new plan and its
    sha256 in the code, reviewed."""
    pending = devcheck_document(reading=hi.READ_PENDING)
    with pytest.raises(hi.HIRefused, match='only .逐位相同 20/20：选 hs_a. lets the entry run') as refused:
        hi.check_decision(decision_for(pending, reading=hi.READ_PENDING))
    assert refused.value.stage == 'decision'


def test_the_environment_check_takes_cuda_only_and_the_checks_environment(monkeypatch):
    check = REAL['check_environment']
    decision = decision_for(devcheck_document())
    with pytest.raises(hi.HIRefused, match='cuda only'):
        check(torch.device('cpu'), decision)
    monkeypatch.setattr(hi, 'environment_of', lambda device: dict(TEST_ENVIRONMENT, device=str(device), python='3.10.20'))
    assert check(torch.device('cuda'), decision)['gpu'] == TEST_ENVIRONMENT['gpu']
    for key, value in (('cudnn_benchmark', True), ('gpu', 'NVIDIA A100'), ('torch', '2.6.0'), ('cudnn', 90200)):
        monkeypatch.setattr(hi, 'environment_of', lambda device, k=key, v=value: dict(TEST_ENVIRONMENT, **{k: v}))
        with pytest.raises(hi.HIRefused, match='differs from the development check'):
            check(torch.device('cuda'), decision)


def test_infer_on_the_cpu_is_refused_by_the_real_environment_check_before_the_lock(w, monkeypatch):
    monkeypatch.setattr(hi, 'check_environment', REAL['check_environment'])
    spy = Spy(monkeypatch)
    with pytest.raises(hi.HIRefused, match='cuda only') as refused:
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=w.tmp/'out', files=spy.files(w.data)))
    assert refused.value.stage == 'decision' and spy.nothing_read() and not Path(hi.HOLDOUT_LOCK).exists() and not (w.tmp/'out').exists()


# ---- m4, n1: the machine gate --------------------------------------------------------------------------------------------------------------------
def fake_machine(units='', pids='', util='0', systemctl_rc=0, fail=None):
    def run(argv, **kwargs):
        if fail is not None and argv[0] == fail:
            raise OSError(f'{fail} is not there (made up by the test)')
        if argv[0] == 'systemctl':
            return subprocess.CompletedProcess(argv, systemctl_rc, units, 'failed' if systemctl_rc else '')
        if '--query-compute-apps=pid' in argv:
            return subprocess.CompletedProcess(argv, 0, pids, '')
        return subprocess.CompletedProcess(argv, 0, util+'\n', '')
    return run


INSIDE = '0::/user.slice/user-1000.slice/user@1000.service/app.slice/tag-hi-infer.service\n'
OUTSIDE = '0::/user.slice/user-1000.slice/user@1000.service/app.slice/app-something.scope\n'
HI_UNIT = 'tag-hi-infer.service loaded active running HI\n'


@pytest.mark.parametrize('machine, cgroup, passes', [
    (dict(), OUTSIDE, True),
    (dict(units=HI_UNIT), INSIDE, True),                     # this process runs in tag-hi-infer: its own unit is not "another"
    (dict(units=HI_UNIT), OUTSIDE, False),                   # diff review n1: a tag-hi-infer that this process is not in (stuck, or a run outside the unit)
    (dict(units=HI_UNIT), None, False),                      # the cgroup cannot be read: nothing is excluded
    (dict(units='tag-he-smoke-queue.service loaded active running HE\n'), INSIDE, False),
    (dict(units='seq-vp-chain.service loaded activating start x\n'), OUTSIDE, False),
    (dict(units=HI_UNIT+'tag-tta-queue.service loaded active running x\n'), INSIDE, False),
    (dict(pids='12345\n'), OUTSIDE, False),
    (dict(util='50'), OUTSIDE, False),
    (dict(util=''), OUTSIDE, False),
    (dict(systemctl_rc=1), OUTSIDE, False),
    (dict(fail='systemctl'), OUTSIDE, False),
    (dict(fail=str(hi.SMI)), OUTSIDE, False),
])
def test_the_machine_gate(tmp_path, machine, cgroup, passes):
    gate = REAL['machine_gate']
    path = tmp_path/'cgroup'
    if cgroup is not None:
        path.write_text(cgroup, encoding='utf-8')
    if passes:
        assert gate(run=fake_machine(**machine), cgroup=path)['passed'] is True
    else:
        with pytest.raises(hi.HIRefused) as refused:
            gate(run=fake_machine(**machine), cgroup=path)
        assert refused.value.stage == 'machine'


DEVCHECK_INSIDE = INSIDE.replace('tag-hi-infer', 'tag-hi-devcheck')
DEVCHECK_UNIT = HI_UNIT.replace('tag-hi-infer', 'tag-hi-devcheck')


@pytest.mark.parametrize('units, cgroup, passes', [
    (DEVCHECK_UNIT, DEVCHECK_INSIDE, True),                 # the check runs in its own unit
    (DEVCHECK_UNIT, OUTSIDE, False),                        # a tag-hi-devcheck this process is not in
    (HI_UNIT, DEVCHECK_INSIDE, False),                      # the hold-out entry's unit is another unit for the check
    (DEVCHECK_UNIT, INSIDE, False),                         # and the check's unit for the entry (own_units of the entry)
])
def test_the_machine_gate_of_the_check_excludes_only_its_own_unit(tmp_path, units, cgroup, passes):
    """Diff review 2, s2: the same rule as n1, with tag-hi-devcheck as the check's own unit."""
    path = tmp_path/'cgroup'
    path.write_text(cgroup, encoding='utf-8')
    own = hi.OWN_UNITS if cgroup == INSIDE else hi.DEVCHECK_UNITS
    if passes:
        assert REAL['machine_gate'](run=fake_machine(units=units), cgroup=path, own_units=own)['own_units'] == list(own)
    else:
        with pytest.raises(hi.HIRefused, match='other units are active'):
            REAL['machine_gate'](run=fake_machine(units=units), cgroup=path, own_units=own)


@pytest.mark.parametrize('own', [('tag-hi-infer',), ('tag-hi-devcheck',)])
def test_the_gates_refusal_names_the_callers_unit(tmp_path, own):
    """Diff review 3, t2: the words of a refusal name the run that is refused, the hold-out entry or the development check."""
    (tmp_path/'cgroup').write_text(OUTSIDE, encoding='utf-8')
    with pytest.raises(hi.HIRefused, match=rf'this run \({own[0]}\) runs alone'):
        REAL['machine_gate'](run=fake_machine(units='tag-x.service loaded active running x\n'), cgroup=tmp_path/'cgroup', own_units=own)


def test_this_test_process_is_not_in_the_entrys_unit():
    assert hi.units_of_this_process() == [] and hi.units_of_this_process(Path('/nowhere/cgroup')) == []


def test_a_busy_machine_stops_infer_before_the_lock(w, monkeypatch):
    monkeypatch.setattr(hi, 'machine_gate', lambda **kwargs: REAL['machine_gate'](run=fake_machine(units='tag-x.service loaded active running x\n')))
    spy = Spy(monkeypatch)
    with pytest.raises(hi.HIRefused, match='other units are active'):
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=w.tmp/'out', files=spy.files(w.data)))
    assert spy.nothing_read() and not Path(hi.HOLDOUT_LOCK).exists()


# ---- I4: once, and the resume record -------------------------------------------------------------------------------------------------------------
def root():
    """The one root of the hold-out entry (the sandbox's HOLDOUT_OUT_SUGGESTED, tmp_path/'out'; A5a review I2)."""
    return Path(hi.HOLDOUT_OUT_SUGGESTED)


def tree(path):
    return {str(p.relative_to(path)): sha_of(p) for p in sorted(Path(path).rglob('*')) if p.is_file()}


def test_the_first_run_takes_the_lock_and_a_second_is_refused_before_anything(w, monkeypatch):
    R = root()
    hi.infer_holdout(None, **holdout_kwargs(w, out_root=R))
    lock = Path(hi.HOLDOUT_LOCK)
    lines = hi.lock_lines(lock)
    assert len(lines) == 1 and lines[0]['event'] == hi.FIRST_RUN and lines[0]['out'] == str(R) and lines[0]['unseal_sha256'] == sha_of(w.unseal_path)
    assert lines[0]['subjects_sha256'] == subject_list_sha256(SUBJECTS) and lines[0]['n_subjects'] == 2 and lines[0]['code_commit'] == 'c'*40 and lines[0]['pid'] > 0 and lines[0]['resume'] is None
    assert lines[0]['devcheck_sha256'] == hi.DEVCHECK_DECISION['devcheck_sha256'] and sorted(lines[0]['members']) == ['unet35_v4_ens_foldA', 'unet35_v4_ens_foldB']
    lock_bytes, mtime = lock.read_bytes(), lock.stat().st_mtime_ns
    before = tree(R)
    spy = Spy(monkeypatch)
    with pytest.raises(hi.HIRefused, match='runs once') as refused:
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=R, files=spy.files(w.data)))
    assert refused.value.stage == 'once'
    with pytest.raises(hi.HIRefused, match='is not') as refused:                      # another root: refused first of all (A5a)
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=w.tmp/'out2', files=spy.files(w.data)))
    assert refused.value.stage == 'output'
    assert spy.nothing_read() and not (w.tmp/'out2').exists() and lock.read_bytes() == lock_bytes and lock.stat().st_mtime_ns == mtime
    assert tree(R) == before


def test_the_lock_is_taken_with_o_excl(tmp_path):
    path = tmp_path/'locks'/'x.lock'
    hi.take_lock(path, dict(a=1))
    with pytest.raises(hi.HIRefused, match='appeared just before') as refused:
        hi.take_lock(path, dict(a=2))
    assert refused.value.stage == 'once' and hi.lock_lines(path) == [dict(a=1)]


def failing_writer(monkeypatch, fail_on):
    """hs_seg.write_prediction that raises for the subjects in the set `fail_on` (made up by the test; the set can be changed between runs)."""
    real_write = hi.hs_seg.write_prediction

    def writer(subject, *args, **kwargs):
        if subject in fail_on:
            raise hi.hs_seg.HSSegFailed(f'{subject}: made up by the test')
        return real_write(subject, *args, **kwargs)
    monkeypatch.setattr(hi.hs_seg, 'write_prediction', writer)
    return fail_on


def failed_first_run(w, monkeypatch, *, process_gone=True):
    """A first run whose second subject fails: the root with one subject written and hi_stopped.json.  The test process is the run's process, so `pid_alive` is replaced by "gone" (diff review n5's own
    test uses the real one)."""
    fail_on = failing_writer(monkeypatch, {SUBJECTS[1]})
    with pytest.raises(hi.hs_seg.HSSegFailed):
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=root()))
    fail_on.clear()
    if process_gone:
        monkeypatch.setattr(hi, 'pid_alive', lambda pid, **kwargs: False)
    return root()


def resume_document(w, out, *, remaining=None, **overrides):
    remaining = sorted(set(SUBJECTS)-{p.stem for p in (out/'subjects').glob('*.json')}) if remaining is None else remaining
    stopped = out/hi.STOPPED_RECORD
    return dict(dict(schema=hi.RESUME_SCHEMA, written_by='lead', words='（合成测试）同意补跑剩下的人', time_local='2026-11-07 6:30 AM CST', utc=hi.utc(), reason='the writer failed (made up by the test)',
                     unseal_sha256=sha_of(w.unseal_path), lock_sha256=sha_of(hi.HOLDOUT_LOCK), previous_out=str(out), stopped_sha256=sha_of(stopped) if stopped.exists() else None,
                     subjects_sha256=subject_list_sha256(remaining), n_subjects=len(remaining)), **overrides)


def write_resume(w, document, k=1, *, commit=True):
    path = write_json(w.repo/hi.RESUME_DIR_REL/f'unseal_holdout_B_v1_resume_{k}.json', document)
    if commit:
        commit_all(w.repo, 'a resume record')
    return path


def test_after_a_failure_only_an_explicit_resume_record_runs_the_rest_once_into_the_same_root(w, monkeypatch):
    """A5a review I2 (the lead's ruling (5)): the resume writes the subjects left into the one root; its records are hi_*_resume_1.json beside the first run's (nothing overwritten); HI is complete when
    the subject records of both runs are every subject; the same record (or any other) cannot resume again (the resume lock is named after the root)."""
    R = failed_first_run(w, monkeypatch)
    lock = Path(hi.HOLDOUT_LOCK)
    lock_bytes = lock.read_bytes()
    first = {name: sha_of(R/name) for name in (hi.STARTED_RECORD, hi.STOPPED_RECORD)}
    written = tree(R/'pred'/SUBJECTS[0])
    spy = Spy(monkeypatch)
    with pytest.raises(hi.HIRefused, match='runs once'):
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=R, files=spy.files(w.data)))
    assert spy.nothing_read()
    assert hi.complete_subjects(lock, SUBJECTS) == (False, [SUBJECTS[1]])
    path = write_resume(w, resume_document(w, R))
    record = hi.infer_holdout(None, **holdout_kwargs(w, out_root=R, resume_path=path))
    assert sorted(record['rows']) == [SUBJECTS[1]] and record['resume']['previous_out'] == str(R) and record['resume']['n_done_before'] == 1 and record['run'] == 1
    assert record['union'] == dict(written=sorted(SUBJECTS), complete=True, missing=[], rule=record['union']['rule'])
    assert record['names'] == dict(started='hi_started_resume_1.json', stopped='hi_stopped_resume_1.json', record='hi_record_resume_1.json')
    assert {name: sha_of(R/name) for name in first} == first and not (R/hi.INFER_RECORD).exists()            # the first run's records untouched
    assert [p.name for p in hi.run_files(R)] == sorted([hi.STARTED_RECORD, hi.STOPPED_RECORD, 'hi_started_resume_1.json', 'hi_record_resume_1.json'])
    assert tree(R/'pred'/SUBJECTS[0]) == written and json.loads((R/'subjects'/f'{SUBJECTS[1]}.json').read_text())['run'] == 1
    resume_lock = hi.resume_lock_path(lock, R)
    line = hi.lock_lines(resume_lock)[0]
    assert lock.read_bytes() == lock_bytes and line['event'] == hi.AUTHORIZED_RESUME and line['resume'] == dict(path=str(path), sha256=sha_of(path), previous_out=str(R)) and line['out'] == str(R)
    started = json.loads((R/'hi_started_resume_1.json').read_text(encoding='utf-8'))
    assert started['resume']['path'] == str(path) and started['residue']['subjects'] == [] and hi.finished_subjects(lock) == set(SUBJECTS) and hi.run_outs(lock) == [str(R)]
    assert hi.complete_subjects(lock, SUBJECTS) == (True, []) and hi.run_consistency_problems(lock) == [] and hi.written_subject_problems(R) == []
    prov = json.loads((R/'pred'/SUBJECTS[1]/'provenance.json').read_text(encoding='utf-8'))
    assert prov['hi']['resume']['sha256'] == sha_of(path)
    with pytest.raises(hi.HIRefused, match='was resumed already'):            # the same record again
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=R, resume_path=path))


def test_a_second_record_after_the_resume_is_refused(w, monkeypatch):
    """Diff review R2, the reviewer's first abuse, in the one root (A5a review I2 point 4): after the resume finished, another record (other words) naming the root again is refused -- the resume lock
    is named after the root (one resume in all), no subject is left, the resume finished."""
    R = failed_first_run(w, monkeypatch)
    first = write_resume(w, resume_document(w, R))
    hi.infer_holdout(None, **holdout_kwargs(w, out_root=R, resume_path=first))
    before = tree(R)
    again = write_resume(w, resume_document(w, R, remaining=[SUBJECTS[1]], words='（合成测试）再补跑一次', lock_sha256=sha_of(hi.HOLDOUT_LOCK)), k=2)
    spy = Spy(monkeypatch)
    with pytest.raises(hi.HIRefused, match='not the latest run of the lock') as refused:
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=R, files=spy.files(w.data), resume_path=again))
    assert 'was resumed already' in str(refused.value) and 'no subject of the unseal record is left' in str(refused.value) and 'finished' in str(refused.value)
    assert refused.value.stage == 'resume' and spy.nothing_read() and tree(R) == before


def test_a_deleted_subject_record_is_seen_and_the_resume_refused(w, monkeypatch):
    """Diff review R2, the reviewer's second abuse: with subjects/<done subject>.json deleted, hi_stopped.json's done is no longer among the records and the resume is refused (it would infer that subject
    again)."""
    R = failed_first_run(w, monkeypatch)
    stopped_sha = sha_of(R/hi.STOPPED_RECORD)
    (R/'subjects'/f'{SUBJECTS[0]}.json').unlink()
    path = write_resume(w, resume_document(w, R, remaining=list(SUBJECTS), stopped_sha256=stopped_sha))
    spy = Spy(monkeypatch)
    with pytest.raises(hi.HIRefused, match='says done') as refused:
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=R, files=spy.files(w.data), resume_path=path))
    assert refused.value.stage == 'resume' and spy.nothing_read() and not hi.resume_lock_path(hi.HOLDOUT_LOCK, R).exists()


def test_a_written_subjects_file_changed_by_one_byte_refuses_the_resume(w, monkeypatch):
    """A5a review I2 point 2 and its test list: before the resume's lock, every written subject's files are checked against its record (output_sha256, provenance_sha256); one byte changed in
    pred/<written subject>/probs36.npz (or provenance.json) refuses the resume (stage resume) with nothing read, no lock taken, nothing moved."""
    R = failed_first_run(w, monkeypatch)
    path = write_resume(w, resume_document(w, R))
    for name in ('probs36.npz', 'provenance.json'):
        target = R/'pred'/SUBJECTS[0]/name
        original = target.read_bytes()
        target.write_bytes(original[:-1]+bytes([original[-1] ^ 1]))
        before = tree(R)
        spy = Spy(monkeypatch)
        with pytest.raises(hi.HIRefused, match='not as their records say') as refused:
            hi.infer_holdout(None, **holdout_kwargs(w, out_root=R, files=spy.files(w.data), resume_path=path))
        assert refused.value.stage == 'resume' and name in str(refused.value) and spy.nothing_read() and tree(R) == before
        assert not hi.resume_lock_path(hi.HOLDOUT_LOCK, R).exists() and not (R/'attempt1').exists()
        target.write_bytes(original)


def test_a_failed_resume_is_not_resumed_again(w, monkeypatch):
    """One root, one resume (A5a review I2 point 4): the resume fails as well (its own hi_stopped_resume_1.json; the first run's records untouched); no further record resumes it -- the resume lock is
    named after the root.  HI stays incomplete; that is for the lead and BZ (a code change)."""
    R = failed_first_run(w, monkeypatch)
    fail_on = failing_writer(monkeypatch, {SUBJECTS[1]})                      # the resume fails as well
    first = write_resume(w, resume_document(w, R))
    with pytest.raises(hi.hs_seg.HSSegFailed):
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=R, resume_path=first))
    fail_on.clear()
    assert json.loads((R/'hi_stopped_resume_1.json').read_text(encoding='utf-8'))['done'] == [] and json.loads((R/hi.STOPPED_RECORD).read_text(encoding='utf-8'))['done'] == [SUBJECTS[0]]
    again = write_resume(w, resume_document(w, R, remaining=[SUBJECTS[1]], lock_sha256=sha_of(hi.HOLDOUT_LOCK)), k=2)
    with pytest.raises(hi.HIRefused, match='was resumed already'):
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=R, resume_path=again))
    assert hi.complete_subjects(hi.HOLDOUT_LOCK, SUBJECTS) == (False, [SUBJECTS[1]])


def test_a_resume_while_the_runs_process_may_live_is_refused(w, monkeypatch):
    """Diff review n5: the lock line's pid still answers (here: the test process itself, which ran it), so the run may still be going; refused."""
    R = failed_first_run(w, monkeypatch, process_gone=False)
    assert REAL['pid_alive'](hi.lock_lines(hi.HOLDOUT_LOCK)[0]['pid']) is True and REAL['pid_alive'](None) is True and REAL['pid_alive'](0) is True
    path = write_resume(w, resume_document(w, R))
    spy = Spy(monkeypatch)
    with pytest.raises(hi.HIRefused, match='may still be running') as refused:
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=R, files=spy.files(w.data), resume_path=path))
    assert refused.value.stage == 'resume' and spy.nothing_read()


def check_downstream(w, R):
    """What the readers of HI's output see after the resume (A5a review I2 point 3, the readers by subject id): every subject written (records of both runs, one root), its files as recorded, XB's
    ConvexAdam-v4seg checks (xb_core.check_prediction on <root>/pred and check_hi_record on the subject's record) pass for both, the root listed once (PQH's hi_predictions reads run_outs)."""
    import sys
    if str(ROOT/'experiments'/'xb') not in sys.path:
        sys.path.insert(0, str(ROOT/'experiments'/'xb'))
    import xb_core as xc
    lock = hi.HOLDOUT_LOCK
    assert hi.complete_subjects(lock, SUBJECTS) == (True, []) and hi.run_consistency_problems(lock) == [] and hi.written_subject_problems(R) == [] and hi.run_outs(lock) == [str(R)]
    from bayesreg.io.pair import verify_segmentation_files
    shape = list(json.loads((R/'pred'/SUBJECTS[0]/'provenance.json').read_text())['outputs']['probs36']['shape'][1:])
    for s in SUBJECTS:
        checked = xc.check_prediction(R/'pred', s, shape=shape, anchor=w.anchor)                                                   # XB's ConvexAdam-v4seg, by id
        assert xc.check_hi_record(R/'pred', s, checked['probs36_sha256'], checked['provenance_sha256'])['record_sha256'] == sha_of(R/'subjects'/f'{s}.json')
        d, row = R/'pred'/s, json.loads((R/'subjects'/f'{s}.json').read_text())
        assert verify_segmentation_files(json.loads((d/'provenance.json').read_text()), d) == row['output_sha256']                 # HE (run_hx reads <pred_root>/<s>/ by id; the outputs re-hashed)
        locked = {name: sha_of(d/name) if (d/name).is_file() else None for name in ('probs36.npz', 'structure_groups.npz', 'provenance.json')}     # hb_store.input_lock's rows (PRED_FILES, by id)
        assert None not in locked.values() and locked['provenance.json'] == row['provenance_sha256'] and {k: locked[k] for k in row['output_sha256']} == row['output_sha256']


def test_a_killed_runs_unrecorded_prediction_is_moved_out_whole_and_predicted_again(w, monkeypatch):
    """A5a review I2 point 2 and its test list (the lead's ruling (5)): the first run is killed after pred/<s>/provenance.json and before subjects/<s>.json (KeyboardInterrupt stands for SIGKILL: no
    hi_stopped.json).  The subject is not written (no record): the resume writes residue.json (every file's sha256, once), moves pred/<s>/ whole to attempt1/pred/<s>/ (the same bytes,
    residue_moved.json), predicts the subject again into pred/<s>/, and HI is complete.  This supersedes diff review 2 s1's "never predicted twice"."""
    real_write = hi.write_json_once

    def killed(path, document):
        if Path(path).parent.name == 'subjects' and Path(path).stem == SUBJECTS[1]:
            raise KeyboardInterrupt('the process is killed here (made up by the test)')
        return real_write(path, document)
    monkeypatch.setattr(hi, 'write_json_once', killed)
    with pytest.raises(KeyboardInterrupt):
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=root()))
    monkeypatch.setattr(hi, 'write_json_once', real_write)
    monkeypatch.setattr(hi, 'pid_alive', lambda pid, **kwargs: False)
    R = root()
    assert not (R/hi.STOPPED_RECORD).exists() and (R/'pred'/SUBJECTS[1]/'provenance.json').is_file() and hi.subject_records(R) == {SUBJECTS[0]}
    assert hi.finished_subjects(hi.HOLDOUT_LOCK) == {SUBJECTS[0]} and hi.residue_subjects(R) == [SUBJECTS[1]] and hi.run_consistency_problems(hi.HOLDOUT_LOCK) == []
    left = tree(R/'pred'/SUBJECTS[1])
    path = write_resume(w, resume_document(w, R, remaining=[SUBJECTS[1]]))
    record = hi.infer_holdout(None, **holdout_kwargs(w, out_root=R, resume_path=path))
    assert sorted(record['rows']) == [SUBJECTS[1]] and record['residue']['subjects'] == [SUBJECTS[1]] and record['union']['complete'] is True
    residue = json.loads((R/'attempt1'/'residue.json').read_text())
    assert {r['file']: r['sha256'] for r in residue['files']} == left and tree(R/'attempt1'/'pred'/SUBJECTS[1]) == left
    assert json.loads((R/'attempt1'/'residue_moved.json').read_text())['passed'] is True and record['residue']['residue_sha256'] == sha_of(R/'attempt1'/'residue.json')
    assert (R/'pred'/SUBJECTS[1]/'provenance.json').is_file() and not (R/'pred'/'attempt1').exists() and sorted(p.name for p in (R/'pred').iterdir()) == sorted(SUBJECTS)
    check_downstream(w, R)


def killed_first_run(w, monkeypatch):
    """A first run killed (KeyboardInterrupt for SIGKILL) after pred/<second subject>/provenance.json and before its record: no hi_stopped.json, one residue."""
    real_write = hi.write_json_once

    def killed(path, document):
        if Path(path).parent.name == 'subjects' and Path(path).stem == SUBJECTS[1]:
            raise KeyboardInterrupt('the process is killed here (made up by the test)')
        return real_write(path, document)
    monkeypatch.setattr(hi, 'write_json_once', killed)
    with pytest.raises(KeyboardInterrupt):
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=root()))
    monkeypatch.setattr(hi, 'write_json_once', real_write)
    monkeypatch.setattr(hi, 'pid_alive', lambda pid, **kwargs: False)
    return root()


@pytest.mark.parametrize('failure', [RuntimeError('the residue moved to attempt1/pred changed (made up by the test)'), OSError(18, 'Invalid cross-device link (made up by the test)')])
def test_a_resume_whose_residue_cannot_be_moved_writes_its_stopped_record(w, monkeypatch, failure):
    """A5a diff review n3: keep_residue runs after the resume lock is taken; when it fails (moved bytes that differ, a rename that fails) the resume writes hi_stopped_resume_1.json (stage
    keep_residue, the error, the residue's directory and subjects) and stops -- never a taken lock without a stopped record.  The first run's records stay as they were, nothing is predicted, and
    the root is not resumed again.  Without the failure the same resume writes no stopped record (the test of the killed run)."""
    R = killed_first_run(w, monkeypatch)
    first = {p.name: sha_of(p) for p in hi.run_files(R)}
    path = write_resume(w, resume_document(w, R, remaining=[SUBJECTS[1]]))

    def failing(*args, **kwargs):
        raise failure
    monkeypatch.setattr(hi, 'keep_residue', failing)
    spy = Spy(monkeypatch)
    with pytest.raises(type(failure)):
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=R, files=spy.files(w.data), resume_path=path))
    stopped = json.loads((R/'hi_stopped_resume_1.json').read_text(encoding='utf-8'))
    assert stopped['stage'] == 'keep_residue' and type(failure).__name__ in stopped['error'] and stopped['run'] == 1 and stopped['done'] == []
    assert stopped['residue'] == dict(dir=str(R/'attempt1'), subjects=[SUBJECTS[1]]) and stopped['subject'] is None
    assert hi.resume_lock_path(hi.HOLDOUT_LOCK, R).exists() and not (R/'hi_started_resume_1.json').exists() and spy.nothing_read()
    assert {p.name: sha_of(p) for p in hi.run_files(R) if p.name in first} == first
    assert hi.complete_subjects(hi.HOLDOUT_LOCK, SUBJECTS) == (False, [SUBJECTS[1]])
    again = write_resume(w, resume_document(w, R, remaining=[SUBJECTS[1]], lock_sha256=sha_of(hi.HOLDOUT_LOCK)), k=2)
    with pytest.raises(hi.HIRefused, match='was resumed already'):
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=R, resume_path=again))


def test_a_failing_subject_record_writes_hi_stopped_and_the_resume_moves_its_prediction_out(w, monkeypatch):
    """Diff review 2, s1: the subject record is written inside the try; its failure writes hi_stopped.json (done without that subject).  A5a review I2: that subject is not written; the resume moves its
    prediction out of pred/ (kept whole) and predicts it again."""
    real_write = hi.write_json_once

    def failing(path, document):
        if Path(path).parent.name == 'subjects' and Path(path).stem == SUBJECTS[1]:
            raise OSError('the disk is full (made up by the test)')
        return real_write(path, document)
    monkeypatch.setattr(hi, 'write_json_once', failing)
    with pytest.raises(OSError):
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=root()))
    monkeypatch.setattr(hi, 'write_json_once', real_write)
    R = root()
    stopped = json.loads((R/hi.STOPPED_RECORD).read_text(encoding='utf-8'))
    assert stopped['subject'] == SUBJECTS[1] and stopped['done'] == [SUBJECTS[0]] and 'OSError' in stopped['error']
    monkeypatch.setattr(hi, 'pid_alive', lambda pid, **kwargs: False)
    record = hi.infer_holdout(None, **holdout_kwargs(w, out_root=R, resume_path=write_resume(w, resume_document(w, R, remaining=[SUBJECTS[1]]))))
    assert record['residue']['subjects'] == [SUBJECTS[1]] and (R/'attempt1'/'pred'/SUBJECTS[1]/'provenance.json').is_file()
    check_downstream(w, R)


def fake_proc(root, pid, *, started_ticks, boot='11111111-2222-3333-4444-555555555555', comm='python (x) y'):
    """A /proc with <pid>/stat (field 22 = started_ticks; a comm with spaces and parentheses, as the kernel allows) and sys/kernel/random/boot_id."""
    (root/str(pid)).mkdir(parents=True)
    fields = ['S']+['0']*18+[str(started_ticks)]+['0']*30          # fields 3 .. 52
    (root/str(pid)/'stat').write_text(f'{pid} ({comm}) '+' '.join(fields)+'\n', encoding='utf-8')
    if boot is not None:
        (root/'sys'/'kernel'/'random').mkdir(parents=True)
        (root/'sys'/'kernel'/'random'/'boot_id').write_text(boot+'\n', encoding='utf-8')
    return root


def test_a_reused_pid_is_told_apart_by_boot_id_and_start_ticks(tmp_path):
    """Diff review 3, t1: the lock line records (pid, start ticks, boot id); a pid that answers is the same process only when both agree.  Another boot or other start ticks: another process (the run's is gone).
    Anything unknown counts as alive (fail closed).  No wall clock: the ticks are counted from boot."""
    import os
    pid = os.getpid()                                                 # a pid that answers os.kill(pid, 0)
    boot = '11111111-2222-3333-4444-555555555555'
    proc = fake_proc(tmp_path/'p', pid, started_ticks=4242, boot=boot)
    assert hi.process_start_ticks(pid, proc) == 4242 and hi.boot_id(proc) == boot and hi.process_identity(pid, proc) == dict(pid=pid, start_ticks=4242, boot_id=boot)
    alive = REAL['pid_alive']
    assert alive(pid, start_ticks=4242, boot=boot, proc=proc) is True                      # the same process
    assert alive(pid, start_ticks=4243, boot=boot, proc=proc) is False                     # the number was given to another process
    assert alive(pid, start_ticks=4242, boot='another-boot', proc=proc) is False           # a reboot since the lock
    assert alive(pid, start_ticks=4242, boot=boot, proc=fake_proc(tmp_path/'noboot', pid, started_ticks=4242, boot=None)) is True     # boot id unreadable
    assert alive(pid, start_ticks=4242, boot=boot, proc=tmp_path/'nowhere') is True        # nothing readable
    assert alive(pid) is True and alive(pid, start_ticks=None, boot=boot, proc=proc) is True and alive(pid, start_ticks=4242, boot='', proc=proc) is True    # the line lacks them
    me = hi.process_identity()
    assert type(me['start_ticks']) is int and me['boot_id'] and alive(pid, start_ticks=me['start_ticks'], boot=me['boot_id']) is True     # the real /proc: this process


def test_the_lock_line_records_the_process_identity(w):
    hi.infer_holdout(None, **holdout_kwargs(w, out_root=root()))
    line = hi.lock_lines(hi.HOLDOUT_LOCK)[0]
    assert line['pid'] > 0 and line['start_ticks'] == hi.process_start_ticks(line['pid']) and line['boot_id'] == hi.boot_id()


def test_a_resume_whose_lock_pid_was_reused_runs(w, monkeypatch):
    """Diff review 3, t1, end to end: the first run's pid (this test process) answers, but a /proc in which that pid has other start ticks lets the resume through (the process that wrote the lock is gone)."""
    R = failed_first_run(w, monkeypatch, process_gone=False)
    line = hi.lock_lines(hi.HOLDOUT_LOCK)[0]
    monkeypatch.setattr(hi, 'PROC', fake_proc(w.tmp/'proc', line['pid'], started_ticks=line['start_ticks']+1, boot=line['boot_id']))
    record = hi.infer_holdout(None, **holdout_kwargs(w, out_root=R, resume_path=write_resume(w, resume_document(w, R))))
    assert sorted(record['rows']) == [SUBJECTS[1]]


def test_a_dead_pid_is_seen_as_gone():
    probe = subprocess.Popen(['true'])
    probe.wait()
    assert REAL['pid_alive'](probe.pid) is False


def resume_departures(w, out1):
    return dict(
        schema=(dict(schema='bayesreg.hi.resume.v0'), 'schema'),
        writer=(dict(written_by='someone'), 'written_by'),
        words=(dict(words=''), 'words'),
        utc=(dict(utc='2030-01-01T00:00:00Z'), 'in the future'),
        unseal=(dict(unseal_sha256='0'*64), 'unseal_sha256'),
        lock=(dict(lock_sha256='0'*64), 'lock_sha256'),
        previous_out=(dict(previous_out=str(w.tmp/'not_a_run')), 'not the output directory of a run of the lock'),
        stopped=(dict(stopped_sha256='0'*64), 'stopped_sha256'),
        stopped_null=(dict(stopped_sha256=None), 'stopped_sha256 is null'),
        subjects=(dict(subjects_sha256=subject_list_sha256(SUBJECTS)), 'subjects_sha256'),
        n_subjects=(dict(n_subjects=2), 'n_subjects'),
        extra_key=(dict(note='x'), 'exactly the keys'),
    )


@pytest.mark.parametrize('name', ['schema', 'writer', 'words', 'utc', 'unseal', 'lock', 'previous_out', 'stopped', 'stopped_null', 'subjects', 'n_subjects', 'extra_key'])
def test_a_resume_record_that_does_not_hold_is_refused_before_anything(w, monkeypatch, name):
    R = failed_first_run(w, monkeypatch)
    change, words = resume_departures(w, R)[name]
    path = write_resume(w, dict(resume_document(w, R), **change))
    before = tree(R)
    spy = Spy(monkeypatch)
    with pytest.raises(hi.HIRefused, match=words) as refused:
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=R, files=spy.files(w.data), resume_path=path))
    assert refused.value.stage == 'resume' and spy.nothing_read() and tree(R) == before and not hi.resume_lock_path(hi.HOLDOUT_LOCK, R).exists()


def test_a_resume_record_must_be_committed_and_named_as_pinned(w, monkeypatch):
    R = failed_first_run(w, monkeypatch)
    loose = write_resume(w, resume_document(w, R), commit=False)
    with pytest.raises(hi.HIRefused, match='not a committed, unchanged file'):
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=R, resume_path=loose))
    elsewhere = write_json(w.repo/'resume.json', resume_document(w, R))
    commit_all(w.repo)
    with pytest.raises(hi.HIRefused, match='unseal_holdout_B_v1_resume_<k>.json'):
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=R, resume_path=elsewhere))
    with pytest.raises(hi.HIRefused, match='a resumed run takes exactly'):
        hi.infer_holdout([SUBJECTS[0]], **holdout_kwargs(w, out_root=R, resume_path=write_resume(w, resume_document(w, R), k=2)))
    assert not (R/'hi_started_resume_1.json').exists() and not hi.resume_lock_path(hi.HOLDOUT_LOCK, R).exists()


def test_a_resume_without_a_lock_is_refused(w):
    path = write_resume(w, dict(schema=hi.RESUME_SCHEMA))
    with pytest.raises(hi.HIRefused, match='there is no run to resume') as refused:
        hi.infer_holdout(None, **holdout_kwargs(w, out_root=root(), resume_path=path))
    assert refused.value.stage == 'once'
