"""PQH's resume (D123 (b); review I2 and the lead's ruling (3) of 11-09; I1 锁 / 补跑): both units killed half way and resumed through a committed resume record, all the way through, on the
synthetic world in the hold-out mode (W.holdout_world; SG's admission stood in by the development one, tested with the real loaders in test_pqh_admission.py):

- the killed run's half-written pair or subject is renamed whole (<name>.attempt<k>) and kept; only what the resume record lists is computed again; a pq_failed pair is computed again only when
  listed, a written pair never; a half-written item that is not listed refuses the resume;
- the members unit's resume passes 'once' with its authorization; the subjects the killed run finished are carried;
- after the lock nothing escapes a unit: a failing pq_failed write, a SystemExit -- the opened files and the records are written.
A kill is a KeyboardInterrupt raised inside the unit (the units do not catch it, as they cannot catch SIGKILL): the lock is there, the records are not."""
from pathlib import Path

import numpy as np
import pytest
import torch

from . import _world as W

hp, hin, hu, io_, hc = W.hp, W.hin, W.hu, W.io_, W.hc
PAIR = W.PAIR
S1, S2 = PAIR.split('__')


@pytest.fixture
def world(tmp_path, monkeypatch):
    w, layout = W.holdout_world(tmp_path, monkeypatch)
    return W.namespace(w=w, layout=layout, tmp=tmp_path)


def resume_record(layout, unit, recompute):
    small, _ = hin.run_dirs(layout, unit, 'first_run')
    lock = Path(layout.out_lock if unit == 'out' else layout.members_lock)
    previous = small/(hu.OUT_RECORD if unit == 'out' else hin.MEMBERS_RECORD)
    path = W.commit_authorization('resume', unit, reason=f'the unit was killed (test)\nrecompute: {recompute}', lock_sha256=io_.sha_file(lock), previous_record=str(previous),
                                  previous_record_sha256=io_.sha_file(previous) if previous.is_file() else None)
    return str(path.relative_to(hc.AUTH_ROOT))


def arrays(path):
    with np.load(path) as z:
        return {k: np.array(z[k]) for k in z.files}


# ---- the out unit ------------------------------------------------------------------------------------------------------------------------------------
def killed_after_pq_npz(world, monkeypatch):
    original = hu.write_once

    def dying(path, data=None, *, text=None):
        if Path(path).name == hu.PQ_JSON:
            raise KeyboardInterrupt('killed (test)')
        return original(path, data, text=text)
    monkeypatch.setattr(hu, 'write_once', dying)
    with pytest.raises(KeyboardInterrupt):
        W.run_out(world.layout, deps=W.deps())
    monkeypatch.setattr(hu, 'write_once', original)
    small, big = world.layout.pq_small, world.layout.pq_big
    assert Path(world.layout.out_lock).is_file() and (big/PAIR/hu.PQ_NPZ).is_file() and not (small/PAIR/hu.PQ_JSON).exists() and not (small/hu.OUT_RECORD).exists()


def test_the_out_unit_resumes_a_killed_pair_renamed_and_computed_again(world, monkeypatch):
    killed_after_pq_npz(world, monkeypatch)
    small, big = world.layout.pq_small, world.layout.pq_big
    half = arrays(big/PAIR/hu.PQ_NPZ)
    rel = resume_record(world.layout, 'out', PAIR)
    assert W.run_out(world.layout, deps=W.deps(), resume_rel=rel) == 0
    assert (big/f'{PAIR}.attempt1'/hu.PQ_NPZ).is_file() and not (small/f'{PAIR}.attempt1').exists()            # kept whole, never deleted
    mine = arrays(big/PAIR/hu.PQ_NPZ)
    assert sorted(mine) == sorted(half) and all(mine[k].tobytes() == half[k].tobytes() for k in mine)        # the same computation
    record = io_.read_json(small/hu.OUT_RESUME_RECORD)
    assert record['complete'] is True and record['kind'] == 'resume' and record['recompute'] == [PAIR] and record['attempts'][0]['dst'].endswith('.attempt1')
    listed = (small/hu.MANIFEST_RESUME).read_text()
    assert str(big/PAIR/hu.PQ_NPZ) in listed and '.attempt1' not in listed and (small/'opened_files_out.resume.json').is_file()
    assert W.run_out(world.layout, deps=W.deps(), resume_rel=rel) == hu.EXIT['refused']     # at most once


def test_without_a_resume_record_the_killed_unit_is_refused_once(world, monkeypatch):
    killed_after_pq_npz(world, monkeypatch)
    assert W.run_out(world.layout, deps=W.deps()) == hu.EXIT['refused']
    assert io_.read_json(W.refusals(world.layout)[-1])['stage'] == 'once'


def test_a_resume_record_that_does_not_list_the_half_written_pair_is_refused(world, monkeypatch):
    killed_after_pq_npz(world, monkeypatch)
    rel = resume_record(world.layout, 'out', 'none')
    assert W.run_out(world.layout, deps=W.deps(), resume_rel=rel) == hu.EXIT['refused']
    refusal = io_.read_json(W.refusals(world.layout)[-1])
    assert refusal['stage'] == 'resume' and 'does not list it' in refusal['reason'] and not Path(hc.resume_lock_path(world.layout.out_lock)).exists()
    assert (world.layout.pq_big/PAIR/hu.PQ_NPZ).is_file() and not list(Path(world.layout.pq_big).glob('*.attempt*'))      # nothing renamed before the lock


def failed_first_run(world, monkeypatch):
    original = hu.compute_pair

    def failing(*args, **kwargs):
        raise RuntimeError('an infrastructure failure (test)')
    monkeypatch.setattr(hu, 'compute_pair', failing)
    assert W.run_out(world.layout, deps=W.deps()) == 0
    monkeypatch.setattr(hu, 'compute_pair', original)
    small = world.layout.pq_small
    assert (small/PAIR/hu.FAILED).is_file() and io_.read_json(small/hu.OUT_RECORD)['complete'] is True


def test_a_pq_failed_pair_is_computed_again_only_when_the_resume_record_lists_it(world, monkeypatch):
    failed_first_run(world, monkeypatch)
    small, big = world.layout.pq_small, world.layout.pq_big
    rel = resume_record(world.layout, 'out', PAIR)
    assert W.run_out(world.layout, deps=W.deps(), resume_rel=rel) == 0
    assert (small/f'{PAIR}.attempt1'/hu.FAILED).is_file() and (small/PAIR/hu.PQ_JSON).is_file() and (big/PAIR/hu.PQ_NPZ).is_file()
    assert io_.read_json(small/hu.OUT_RESUME_RECORD)['written'] == [PAIR]


def test_a_pq_failed_pair_not_listed_stays_failed(world, monkeypatch):
    failed_first_run(world, monkeypatch)
    small = world.layout.pq_small
    rel = resume_record(world.layout, 'out', 'none')
    assert W.run_out(world.layout, deps=W.deps(), resume_rel=rel) == 0
    record = io_.read_json(small/hu.OUT_RESUME_RECORD)
    assert record['todo'] == [] and record['written'] == [] and (small/PAIR/hu.FAILED).is_file() and not (small/PAIR/hu.PQ_JSON).exists()


def test_a_written_pair_is_never_computed_again(world, monkeypatch):
    """With every pair written the run is complete and holdout_common.load_resume refuses it; the plan of a resume refuses a written pair on its own too (several pairs, one written)."""
    killed_after_pq_npz(world, monkeypatch)
    small, big = world.layout.pq_small, world.layout.pq_big
    W.write_json(small/PAIR/hu.PQ_JSON, dict(test='as if written'))
    with pytest.raises(hu.Refusal, match='never computed again'):
        hu.out_resume_plan(world.layout, small, big, [PAIR])
    rel = resume_record(world.layout, 'out', PAIR)
    assert W.run_out(world.layout, deps=W.deps(), resume_rel=rel) == hu.EXIT['refused']
    assert 'complete record' in io_.read_json(W.refusals(world.layout)[-1])['reason']


@pytest.mark.parametrize('reason, ok', [('x\nrecompute: none', []), ('x\nrecompute: a, b', ['a', 'b']), ('x\nrecompute:', None), ('x\nrecompute:   ', None),
                                        ('x\nrecompute: a,,b', None), ('x\nrecompute: a,', None), ('x', None), ('recompute: a\nrecompute: b', None)])
def test_the_recompute_line(reason, ok):
    """Review NM7: an empty list is refused; only 'recompute: none' means nothing."""
    if ok is None:
        with pytest.raises(hu.Refusal, match='resume'):
            hu.resume_items(dict(reason=reason))
    else:
        assert hu.resume_items(dict(reason=reason)) == ok


def test_a_resume_record_without_the_recompute_line_is_refused(world, monkeypatch):
    killed_after_pq_npz(world, monkeypatch)
    lock = Path(world.layout.out_lock)
    path = W.commit_authorization('resume', 'out', reason='killed (test)', lock_sha256=io_.sha_file(lock), previous_record=str(world.layout.pq_small/hu.OUT_RECORD),
                                  previous_record_sha256=None)
    assert W.run_out(world.layout, deps=W.deps(), resume_rel=str(path.relative_to(hc.AUTH_ROOT))) == hu.EXIT['refused']
    assert "recompute" in io_.read_json(W.refusals(world.layout)[-1])['reason']


def test_nothing_escapes_the_out_unit_after_the_lock(world, monkeypatch):
    """I2: compute_pair fails, and so does the write of pq_failed.json: the unit returns 8 with its records (the failed record not written is named); a SystemExit there is a failed pair."""
    original = hu.write_once

    def failing_compute(*args, **kwargs):
        raise SystemExit('the frozen tree refused (test)')

    def failing_write(path, data=None, *, text=None):
        if Path(path).name == hu.FAILED:
            raise OSError('disk full (test)')
        return original(path, data, text=text)
    monkeypatch.setattr(hu, 'compute_pair', failing_compute)
    monkeypatch.setattr(hu, 'write_once', failing_write)
    assert W.run_out(world.layout, deps=W.deps()) == hu.EXIT['stopped']
    small = world.layout.pq_small
    record = io_.read_json(small/hu.OUT_RECORD)
    assert record['complete'] is False and 'SystemExit' in record['failed'][PAIR] and 'disk full' in record['failed_record_not_written'][PAIR]
    assert (small/'opened_files_out.json').is_file() and not (small/hu.MANIFEST).exists()


def test_a_record_name_taken_is_written_beside_it(world, monkeypatch):
    """write_record: never overwritten, never lost (I2)."""
    small = world.layout.pq_small
    W.write_json(small/'opened_files_out.json', dict(test='there already'))
    monkeypatch.setattr(hu, 'compute_pair', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('fails (test)')))
    assert W.run_out(world.layout, deps=W.deps()) == 0
    beside = sorted(small.glob('opened_files_out.*.json'))
    assert len(beside) == 1 and io_.read_json(beside[0])['written_beside'] == str(small/'opened_files_out.json')


# ---- the members unit --------------------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def members_world(world, monkeypatch):
    """The members unit in the hold-out mode with its admission stood in (the real one: test_pqh_admission.py) and the inference replaced by the registered prediction and the development V;
    the members run copied by holdout_world removed."""
    import shutil
    import pq_members
    small, big = hin.run_dirs(world.layout, 'members', 'first_run')
    shutil.rmtree(small)
    shutil.rmtree(big)
    w = world.w

    def admission(layout, deps, checks, subjects=None, rules=None):
        subjects = list(layout.subjects) if subjects is None else list(subjects)
        if rules is not None:
            rules.allow_files(hin.hi_files(layout.hi_lock, layout.subjects), 'test: HI\'s runs')
        checks['admission'] = {s: 'test: the admission has its own tests' for s in subjects}
        plans = {s: ([('model', 'm', 'path')], {'model': {}}, 'all_folds_mean') for s in subjects}
        return subjects, None, object(), None, plans, hin.hi_predictions(layout.hi_lock, subjects), dict(reading='test')
    monkeypatch.setattr(hu, 'members_admission', admission)
    calls = []

    cached = {}
    for s in (S1, S2):                                     # read before the unit (outside its spy: in the hold-out mode the development roots are forbidden, as they should be)
        with np.load(w.ctx.pred_root/s/'probs36.npz') as z:
            mean = np.array(z['p']).astype(np.float32)
        with np.load(w.ctx.pq_root/'members'/s/io_.VAR_FILE) as z:
            cached[s] = (mean, np.array(z['V']))

    def mean_and_variance(subject, *, files, models, plan, device):
        calls.append(subject)
        mean, V = cached[subject]
        return mean.copy(), V.copy(), dict(sha256=W.PW.source_sha(subject)), dict(percentile=99.5), [dict(member_sha256='m')]
    monkeypatch.setattr(pq_members, 'mean_and_variance', mean_and_variance)
    deps = W.deps(device=lambda device, free: torch.device('cpu'), environment=lambda device, decision: dict(device=str(device)))
    return W.namespace(layout=world.layout, deps=deps, calls=calls, small=small, big=big)


def run_members(m, **kwargs):
    W.XW.sealed_module._reset_cache()
    return hu.members_unit(m.layout, device='cpu', deps=m.deps, **kwargs)


def test_the_members_unit_resumes_a_killed_subject_past_once(members_world, monkeypatch):
    import pq_members
    m = members_world
    original = pq_members.subject_record

    def dying(subject, **kwargs):
        if subject == S2:
            io_.save_npz_once(Path(kwargs['out_dir'])/io_.VAR_FILE, V=kwargs['V'].astype(np.float32))
            raise KeyboardInterrupt('killed after var.npz (test)')
        return original(subject, **kwargs)
    monkeypatch.setattr(pq_members, 'subject_record', dying)
    with pytest.raises(KeyboardInterrupt):
        run_members(m)
    monkeypatch.setattr(pq_members, 'subject_record', original)
    assert Path(m.layout.members_lock).is_file() and (m.big/S1/'member.json').is_file() and (m.big/S2/io_.VAR_FILE).is_file() and not (m.big/S2/'member.json').exists()
    rel = resume_record(m.layout, 'members', S2)
    m.calls.clear()
    assert run_members(m, resume_rel=rel) == 0
    assert m.calls == [S2] and (m.big/f'{S2}.attempt1'/io_.VAR_FILE).is_file() and (m.big/S2/'member.json').is_file()
    record = io_.read_json(m.small/hin.MEMBERS_RESUME_RECORD)
    assert record['kind'] == 'resume' and record['carried'] == [S1] and sorted(record['subjects']) == [S1, S2] and record['m5_passed'] is True
    rows, _ = hin.members_view(m.layout, [S1, S2])
    assert sorted(rows) == [S1, S2]
    assert run_members(m, resume_rel=rel) == hu.EXIT['refused']                     # at most once
    assert W.run_out(m.layout, deps=W.deps()) == 0                                  # the PQ unit on the resumed members run
    listed = (Path(m.layout.pq_small)/hu.MANIFEST).read_text()
    assert str(m.big/S2/io_.VAR_FILE) in listed and '.attempt' not in listed        # review NM5: the renamed half directory is kept, not listed


def test_a_members_resume_that_does_not_list_the_half_written_subject_is_refused(members_world, monkeypatch):
    import pq_members
    m = members_world
    original = pq_members.subject_record

    def dying(subject, **kwargs):
        if subject == S2:
            io_.save_npz_once(Path(kwargs['out_dir'])/io_.VAR_FILE, V=kwargs['V'].astype(np.float32))
            raise KeyboardInterrupt('killed after var.npz (test)')
        return original(subject, **kwargs)
    monkeypatch.setattr(pq_members, 'subject_record', dying)
    with pytest.raises(KeyboardInterrupt):
        run_members(m)
    monkeypatch.setattr(pq_members, 'subject_record', original)
    rel = resume_record(m.layout, 'members', S1)
    assert run_members(m, resume_rel=rel) == hu.EXIT['refused']
    reason = io_.read_json(W.refusals(m.layout)[-1])['reason']
    assert 'does not list it' in reason and 'is done' in reason


def test_nothing_escapes_the_members_unit_after_the_lock(members_world, monkeypatch):
    import pq_members

    def exiting(subject, **kwargs):
        raise SystemExit('a library exited (test)')
    monkeypatch.setattr(pq_members, 'mean_and_variance', exiting)
    assert run_members(members_world) == hu.EXIT['stopped']
    record = io_.read_json(members_world.small/hin.MEMBERS_RECORD)
    assert record['stage'] == 'error' and 'SystemExit' in record['error'] and (members_world.small/'opened_files_members.json').is_file()
    assert (members_world.small/'members_stopped.json').is_file() and not (members_world.small/hu.STOPPED).exists()


def test_a_members_control_rerun_after_a_stop_before_its_lock_runs_and_its_stop_record_is_not_listed(members_world):
    """Review 2 n3: a members control rerun that stopped before its lock leaves pq_refused.json in members_rerun/; with a new authorization listing both stops the rerun runs ('once' lets
    a directory holding only stop records through), and the PQ unit's list leaves the stop record out (stop records are not outputs)."""
    m = members_world
    lock = Path(m.layout.members_lock)
    hc.take_lock(lock, hc.lock_line(tool=hp.TOOL, unit='members', kind='first_run'))
    first_stop = W.write_json(m.small/hu.STOPPED, dict(unit='members', stage='hm5', error='test'))
    W.commit_authorization('rerun', 'members', refusals=[dict(path=str(first_stop), sha256=io_.sha_file(first_stop))])
    plan = hu.lock_plan(m.layout, 'members', lock, None, {}, lambda: False)
    assert plan['kind'] == 'control_rerun'
    assert hu.stop_before_lock(m.layout, 'members', None, {}, 'hm6', 'the sealed-list cache (test)', W.quiet, plan) == hu.EXIT['stopped']
    rerun_small, rerun_big = hin.run_dirs(m.layout, 'members', 'control_rerun')
    rerun_stop = rerun_small/hu.STOPPED
    assert rerun_stop.is_file() and sorted(q.name for q in rerun_small.iterdir()) == [hu.STOPPED]
    assert run_members(m) == hu.EXIT['refused']                                     # the first authorization no longer covers the stops
    W.commit_authorization('rerun', 'members', refusals=[dict(path=str(q), sha256=io_.sha_file(q)) for q in (first_stop, rerun_stop)])
    assert run_members(m) == 0
    assert (rerun_small/hin.MEMBERS_RECORD).is_file() and Path(str(lock)+'.rerun').is_file() and rerun_stop.is_file()
    assert hin.members_dirs(m.layout)[0] == rerun_small
    assert W.run_out(m.layout, deps=W.deps()) == 0
    listed = (Path(m.layout.pq_small)/hu.MANIFEST).read_text()
    assert str(rerun_big/S1/io_.VAR_FILE) in listed and 'pq_refused' not in listed
