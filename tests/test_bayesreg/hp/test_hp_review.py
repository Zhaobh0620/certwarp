"""HP after its review (docs/bayesreg/reviews/HP_工具_复核_20261109.md; the lead's rulings, decision_log 2026-11-09 "HP 复核"): the decision constants of the two
development checks (I1), the real roots refused in the test sandbox before anything is written (I2), the resume through holdout_common.load_resume, the data
preparation's record checked by the Atropos entry (m3), the manifests checked before the lock (m4), the spy's limits (m5), the gaps of the first round's tests (m6),
the dry run's --out (m7)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from . import _world as W

hc, core, prep, ah, S = W.hc, W.core, W.prep, W.ah, W.S


def run_prep(w, **kwargs):
    S._reset_cache()
    return prep.run_holdout(unseal_path=kwargs.pop('unseal_path', w.record_path), code=W.CODE, log=W.quiet, **kwargs)


def run_atropos(w, **kwargs):
    S._reset_cache()
    return ah.run_holdout(unseal_path=kwargs.pop('unseal_path', w.record_path), code=W.CODE, log=W.quiet, **kwargs)


def data_events(exc_or_record, root):
    spy = getattr(exc_or_record, 'hp_spy', None) if not isinstance(exc_or_record, dict) else exc_or_record
    root = str(Path(root).resolve())
    return [e for e in spy['events'] if e['resolved'].startswith(root)]


# ---- I1: the decisions ------------------------------------------------------------------------------------------------------------------------------
def test_no_decision_refuses_both_hold_out_runs_before_the_admission(tmp_path, monkeypatch):
    w = W.make_world(tmp_path, monkeypatch)
    for module, run in ((prep, run_prep), (ah, run_atropos)):
        with monkeypatch.context() as m:
            m.setattr(module, 'PREP_DEVCHECK_DECISION' if module is prep else 'ATROPOS_DEVCHECK_DECISION', None)
            with pytest.raises(hc.Refused) as info:
                run(w)
            assert info.value.stage == 'decision' and 'is None' in str(info.value)
            assert data_events(info.value, w.data) == []
            assert not any(e['resolved'].endswith('holdout_B_pairs.tsv') for e in info.value.hp_spy['events'])     # refused before the admission read anything
    assert not Path(hc.STAGE_ROOT).exists() and not Path(hc.LOCK_ROOT).exists()


@pytest.mark.parametrize('case', ['sha256', 'reading', 'plan', 'lock_line', 'keys', 'path'])
def test_a_decision_that_does_not_hold_is_refused(tmp_path, monkeypatch, case):
    w = W.make_world(tmp_path, monkeypatch)
    decision = dict(w.decisions['prep'])
    if case == 'sha256':
        decision['devcheck_sha256'] = '0'*64
    elif case == 'reading':
        decision['reading'] = prep.READ_DIFFERENT
    elif case == 'plan':
        monkeypatch.setattr(prep, 'PREP_DEVCHECK_PLAN_SHA256', 'f'*64)
    elif case == 'lock_line':
        prep.devcheck_lock().write_text(json.dumps(dict(other=1))+'\n')
    elif case == 'keys':
        decision.pop('bz_words')
    else:
        decision['devcheck_json'] = str(tmp_path/'elsewhere.json')
    monkeypatch.setattr(prep, 'PREP_DEVCHECK_DECISION', decision)
    with pytest.raises(hc.Refused) as info:
        run_prep(w)
    assert info.value.stage == 'decision'
    assert not Path(hc.STAGE_ROOT).exists()


def test_the_atropos_readings_the_decision_accepts(tmp_path, monkeypatch):
    w = W.make_world(tmp_path, monkeypatch)
    S._reset_cache()
    assert ah.check_decision()['reading'] == '逐位相同 2/2'
    W.decide(monkeypatch, atropos_reading=ah.READ_NONDETERMINISTIC)
    with pytest.raises(hc.Refused, match="BZ's words"):
        ah.check_decision()
    W.decide(monkeypatch, atropos_reading=ah.READ_NONDETERMINISTIC, bz_words='（测试）BZ：可以接受')
    assert ah.check_decision()['bz_words']
    for reading in (ah.READ_INPUTS_DIFFER, ah.READ_SPY_FAILED):
        W.decide(monkeypatch, atropos_reading=reading, bz_words='（测试）BZ：可以接受')
        with pytest.raises(hc.Refused, match='does not let the hold-out run start'):
            ah.check_decision()
    W.decide(monkeypatch, prep_reading=prep.READ_DIFFERENT, bz_words='（测试）')
    with pytest.raises(hc.Refused, match='does not let the hold-out run start'):
        prep.check_decision()
    assert w


def test_a_decision_from_a_real_development_check_lets_the_run_start(tmp_path, monkeypatch):
    """The prep development check runs on the synthetic old-format stage; a decision written from its record and lock (as the reviewed commit would) is accepted, and
    the hold-out run then goes through."""
    w = W.make_world(tmp_path, monkeypatch, decided=False)
    monkeypatch.setattr(hc, 'DEV_PAIRS', W.DEV_PAIRS)
    W.write_old_stage(hc.DEV_STAGE, w.data, W.DEV_PAIRS)
    monkeypatch.setattr(prep, 'PREP_DEVCHECK_PLAN_SHA256', W.plan_file(w, prep.DEVCHECK_PLAN_REL))
    S._reset_cache()
    record = prep.run_devcheck(code=W.DEV_CODE, log=W.quiet)
    assert record['reading'] == '逐位相同 2/2' and record['lock_line']['code_commit'] == W.DEV_CODE['commit']
    path = prep.devcheck_root()/prep.DEVCHECK_RECORD
    monkeypatch.setattr(prep, 'PREP_DEVCHECK_DECISION', W.decision_for(path, reading=record['reading'], plan_sha256=prep.PREP_DEVCHECK_PLAN_SHA256))
    S._reset_cache()
    assert prep.check_decision()['devcheck_sha256'] == W.file_sha256(path)
    assert run_prep(w)['passed']


# ---- I2: the real roots ------------------------------------------------------------------------------------------------------------------------------
def test_writing_a_real_root_is_refused_before_anything_is_written(tmp_path, monkeypatch):
    """Without the sandbox's monkeypatch of the roots (the module constants back at the real paths), every run is refused by the test-sandbox guard before it writes;
    the tripwire of conftest.py checks no real root appeared."""
    w = W.make_world(tmp_path, monkeypatch)
    before = W.real_roots_present()
    for name in ('STAGE_ROOT', 'ATROPOS_ROOT', 'LOCK_ROOT'):
        monkeypatch.setattr(hc, name, W.REAL_HC[name])
    with pytest.raises(hc.RealRootInTest):
        run_prep(w)
    with pytest.raises(hc.RealRootInTest):
        run_atropos(w)
    monkeypatch.setattr(hc, 'RESULTS', W.REAL_HC['RESULTS'])
    with pytest.raises(hc.RealRootInTest, match='hp_devcheck_v1'):
        prep.run_devcheck(code=W.CODE, log=W.quiet)
    with pytest.raises(hc.RealRootInTest, match='hp_devcheck_v1'):
        ah.run_devcheck(code=W.CODE, log=W.quiet)
    with pytest.raises(hc.RealRootInTest):
        prep.prepare(W.REAL_HC['STAGE_ROOT'], W.PAIRS, None, rule='x', dataset='x')
    with pytest.raises(hc.RealRootInTest):
        hc.write_json_once(W.REAL_HC['RESULTS']/'holdout_B_v1'/'x.json', {})
    policy = core.Policy(name='test', write_inside=[W.REAL_HC['STAGE_ROOT']])
    assert policy.classify(W.REAL_HC['STAGE_ROOT']/'pair.npz', True)[0] == 'forbidden'          # the spy too, as a second net
    assert W.real_roots_present() == before
    monkeypatch.delenv(hc.TEST_SANDBOX_ENV)
    assert core.guard_write(W.REAL_HC['STAGE_ROOT']) == (W.REAL_HC['STAGE_ROOT'],)            # outside the sandbox the guard lets the production paths through


# ---- the resume ----------------------------------------------------------------------------------------------------------------------------------
def failing_prep(monkeypatch, after=1):
    real = prep.write_pair
    calls = []

    def write(out, fixed, moving, files, **kwargs):
        calls.append(fixed)
        if len(calls) > after:
            raise RuntimeError('a synthetic infrastructure failure')
        return real(out, fixed, moving, files, **kwargs)
    monkeypatch.setattr(prep, 'write_pair', write)


def test_the_data_preparation_resumes_once_with_a_committed_authorization(tmp_path, monkeypatch):
    w = W.make_world(tmp_path, monkeypatch)
    with monkeypatch.context() as m:
        failing_prep(m)
        with pytest.raises(RuntimeError):
            run_prep(w)
    stage, lock = Path(hc.STAGE_ROOT), Path(hc.LOCK_ROOT)/prep.LOCK_NAME
    stopped = stage/prep.STOPPED
    assert stopped.is_file() and lock.is_file()
    with pytest.raises(hc.Refused) as info:
        run_prep(w)
    assert info.value.stage == 'once'
    uncommitted = W.write_resume(w, prep, lock=lock, previous_record=stopped, commit=False)
    with pytest.raises(hc.Refused) as info:
        run_prep(w, resume_path=uncommitted)
    assert info.value.stage == 'resume' and 'committed' in str(info.value)
    W.commit_all(w.repo, 'the authorization committed')
    before = {p.relative_to(stage): W.file_sha256(p) for p in stage.rglob('*') if p.is_file()}
    record = run_prep(w, resume_path=uncommitted)
    kept = core.attempt_path(stage)
    assert record['passed'] and record['resume']['kept']['dst'] == str(kept)
    assert {p.relative_to(kept): W.file_sha256(p) for p in kept.rglob('*') if p.is_file()} == before          # the failed attempt kept whole
    assert Path(hc.resume_lock_path(lock)).is_file() and json.loads(Path(hc.resume_lock_path(lock)).read_text())['event'] == 'authorized_resume'
    assert (stage/prep.RECORD).is_file()
    second = W.write_resume(w, prep, lock=lock, previous_record=stopped, k=2)
    with pytest.raises(hc.Refused) as info:
        run_prep(w, resume_path=second)
    assert info.value.stage == 'resume'                                         # at most once, and the run is complete now


def test_a_complete_run_is_not_resumed(tmp_path, monkeypatch):
    w = W.make_world(tmp_path, monkeypatch)
    run_prep(w)
    lock = Path(hc.LOCK_ROOT)/prep.LOCK_NAME
    auth = W.write_resume(w, prep, lock=lock, previous_record=Path(hc.STAGE_ROOT)/prep.STOPPED)
    with pytest.raises(hc.Refused, match='complete record'):
        run_prep(w, resume_path=auth)


def test_atropos_resumes_after_its_qc_stop(tmp_path, monkeypatch):
    """m6 (c) and the resume: a QC failure (a class below 5 %) is atropos_seg's SystemExit; main exits 8 and atropos_stopped.json stays; with a committed
    authorization the run resumes once, the failed attempt kept whole."""
    w = W.make_world(tmp_path, monkeypatch)
    run_prep(w)
    W.fake_ants(tmp_path, monkeypatch)
    monkeypatch.setenv('HP_FAKE_ATROPOS_QC_FAIL', '1')
    monkeypatch.setattr(core, 'check_code', lambda code, require_clean=True: dict(checked='test'))
    S._reset_cache()
    assert ah.main(['run', '--unseal', str(w.record_path)], code=W.CODE, out=W.quiet) == 8
    out, lock = Path(hc.ATROPOS_ROOT), Path(hc.LOCK_ROOT)/ah.LOCK_NAME
    stopped = json.loads((out/ah.STOPPED).read_text())
    assert 'QC failed' in stopped['stopped']
    monkeypatch.delenv('HP_FAKE_ATROPOS_QC_FAIL')
    auth = W.write_resume(w, ah, lock=lock, previous_record=out/ah.STOPPED)
    record = run_atropos(w, resume_path=auth)
    assert record['passed'] and len(record['subjects']) == 4
    assert (core.attempt_path(out)/ah.STOPPED).is_file() and Path(hc.resume_lock_path(lock)).is_file()


# ---- m3, m4 --------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('case', ['missing', 'not_passed', 'other_unseal', 'other_sums'])
def test_atropos_checks_the_data_preparation_record(tmp_path, monkeypatch, case):
    w = W.make_world(tmp_path, monkeypatch)
    run_prep(w)
    W.fake_ants(tmp_path, monkeypatch)
    path = Path(hc.STAGE_ROOT)/prep.RECORD
    document = json.loads(path.read_text())
    if case == 'missing':
        path.unlink()
    else:
        if case == 'not_passed':
            document['passed'] = False
        elif case == 'other_unseal':
            document['admission']['unseal']['sha256'] = '0'*64
        else:
            document['stage_summary']['sha256sums_sha256'] = '0'*64
        path.write_text(json.dumps(document))
    with pytest.raises(hc.Refused) as info:
        run_atropos(w)
    assert info.value.stage == 'stage'
    assert not Path(hc.ATROPOS_ROOT).exists() and not (Path(hc.LOCK_ROOT)/ah.LOCK_NAME).exists()


def test_the_manifests_are_checked_before_the_lock(tmp_path, monkeypatch):
    from bayesreg.io.unseal import bind_unseal, load_unseal
    w = W.make_world(tmp_path, monkeypatch)
    run_prep(w)
    S._reset_cache()
    bound = bind_unseal(w.allowlist, load_unseal(w.record_path))
    name = f'{W.PAIRS[1][0]}__{W.PAIRS[1][1]}'
    manifest = Path(hc.STAGE_ROOT)/name/'manifest.json'
    document = json.loads(manifest.read_text())
    document['fixed_subject'] = W.HOLDOUT[0]
    manifest.write_text(json.dumps(document))
    entries = ah.StageEntries(hc.STAGE_ROOT, bound)
    with pytest.raises(hc.Refused, match='the manifest names'):
        entries.precheck()
    document['fixed_subject'], document['moving_subject'] = W.H1[0], W.HOLDOUT[0]
    manifest.write_text(json.dumps(document))
    renamed = ah.StageEntries(hc.STAGE_ROOT, bound)
    renamed.names = [f'{W.H1[0]}__{W.HOLDOUT[0]}' if n == name else n for n in renamed.names]
    (Path(hc.STAGE_ROOT)/f'{W.H1[0]}__{W.HOLDOUT[0]}').mkdir()
    (Path(hc.STAGE_ROOT)/f'{W.H1[0]}__{W.HOLDOUT[0]}'/'manifest.json').write_text(json.dumps(document))
    from bayesreg.io.guard import SubjectSealed
    with pytest.raises(SubjectSealed):
        renamed.precheck()                                                      # the guard on the manifest's ids, before any pair.npz is opened


# ---- m5 ------------------------------------------------------------------------------------------------------------------------------------------
def test_preload_refuses_indexed_gzip_and_code_files_are_exempt_from_the_name_rule(monkeypatch):
    from nibabel import _compression
    monkeypatch.setattr(_compression, 'HAVE_INDEXED_GZIP', True, raising=False)
    with pytest.raises(hc.Refused, match='indexed_gzip'):
        core.preload()
    monkeypatch.setattr(_compression, 'HAVE_INDEXED_GZIP', False)
    assert core.preload()['have_indexed_gzip'] is False
    policy = core.Policy(name='test')
    import sys
    site = Path(sys.prefix)/'lib'/'x'
    assert policy.classify(site/'metrics.py', False)[0] == 'allowed'
    assert policy.classify(site/'metrics.json', False)[0] == 'forbidden'
    assert policy.classify(core.CODE_TREE/'bayesreg'/'eval'/'metrics.py', False)[0] == 'allowed'
    assert policy.classify(core.CODE_TREE/'bayesreg'/'eval'/'metrics.py', True)[0] == 'forbidden'


# ---- m6 ------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('case', ['not_in_tsv', 'reversed'])
def test_a_pair_off_the_list_or_reversed_reads_nothing(tmp_path, monkeypatch, case):
    """m6 (a): admit_holdout under the run's spy: refused, and the spy saw nothing of a subject (no image, no stage)."""
    w = W.make_world(tmp_path, monkeypatch)
    pairs = [(W.HOLDOUT[0], W.HOLDOUT[2])] if case == 'not_in_tsv' else [(W.HOLDOUT[1], W.HOLDOUT[0])]
    S._reset_cache()
    core.preload()
    with pytest.raises(hc.Refused) as info:
        with core.spying(prep.holdout_policy(hc.STAGE_ROOT, Path(hc.LOCK_ROOT)/prep.LOCK_NAME)):
            core.admit_holdout(pairs, unseal_path=w.record_path, data_root=w.data)
    assert info.value.stage == 'admission'
    assert data_events(info.value, w.data) == [] and data_events(info.value, hc.STAGE_ROOT) == []
    assert info.value.hp_spy['breaches'] == []


def test_asking_a_label_member_of_a_pair_stops(tmp_path):
    """m6 (b): Spy.see_member -- a pair.npz member other than fixed, moving, initial_u stops the run."""
    path = tmp_path/'pair.npz'
    np.savez_compressed(path, fixed=np.zeros(3), fixed_seg35=np.zeros(3, np.int16))
    S._reset_cache()
    with core.spying(core.Policy(name='test', read_exact=[path])) as spy:
        with np.load(path) as npz:
            assert npz['fixed'].shape == (3,)
            with pytest.raises(core.Breach, match="'fixed_seg35'"):
                npz['fixed_seg35']
    assert [m['member'] for m in spy.members] == ['fixed', 'fixed_seg35'] and spy.breaches


@pytest.mark.parametrize('name', ['H1', 'H3'])
def test_atropos_refuses_h1_h3_before_the_stage_is_read(tmp_path, monkeypatch, name):
    """m6 (d): the refusal carries the spy's record; nothing of the stage or of a subject was opened."""
    subject = dict(H1=W.H1[0], H3=W.H3[0])[name]
    w = W.make_world(tmp_path, monkeypatch, rows=W.PAIRS+[(subject, W.HOLDOUT[0])])
    with pytest.raises(hc.Refused) as info:
        run_atropos(w)
    assert info.value.stage == 'never'
    assert data_events(info.value, hc.STAGE_ROOT) == [] and data_events(info.value, w.data) == []


# ---- m7 ------------------------------------------------------------------------------------------------------------------------------------------
def test_the_dry_run_out_must_be_outside_the_roots(tmp_path, monkeypatch):
    nonempty = tmp_path/'nonempty'
    nonempty.mkdir()
    (nonempty/'x').write_text('x')
    for out in (hc.RESULTS/'dry', hc.BIG_ROOT/'dry', hc.REAL_BIG/'dry', Path('/mnt/d/data')/'dry', core.CODE_TREE/'dry', nonempty):
        assert core.dry_run_out_problem(out), out
        assert prep.main(['run', '--dry-run', '--out', str(out)], code=W.CODE, out=W.quiet) == 2
        assert ah.main(['run', '--dry-run', '--out', str(out)], code=W.CODE, out=W.quiet) == 2
    assert core.dry_run_out_problem(tmp_path/'fresh') is None
    assert not (hc.RESULTS/'dry').exists() and not (core.CODE_TREE/'dry').exists()
    assert os.path.isdir(nonempty)
