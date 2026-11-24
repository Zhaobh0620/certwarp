"""PQH in the hold-out mode before the lock (contract C22.H3-3, C22.H4-4, C22.H0-4, C22.H8 实现门槛 准入, 次序, 拒绝之后的重跑; C23.3, C23.4-2): the real ids of the pair list and of the sealed lists
(id lists only), SG's real loaders; no unseal record (main today) -> both units are refused at the admission and open no file of a subject; with the simulated unseal branch the admission passes;
the preregistration, the checkout, the evaluation's master lock, the once rule, the control rerun and the resume through committed authorizations (holdout_common)."""
import json
from pathlib import Path

import pytest

from .. import _p0_pins as P0
from . import _world as W

hp, hin, hu, io_, hc, XW = W.hp, W.hin, W.hu, W.io_, W.hc, W.XW


@pytest.fixture
def holdout(tmp_path, monkeypatch):
    """The first pair of holdout_B_pairs.tsv; the code tree is the detached checkout <holdout root>/code (a link to this tree); the preregistration pinned to this tree's file; a finalization
    record naming the pair without output (no method file to read)."""
    row = hc.holdout_pairs()[0]
    checkout = Path(hc.HOLDOUT_ROOT)/'code'
    checkout.parent.mkdir(parents=True, exist_ok=True)
    checkout.symlink_to(W.ROOT)
    monkeypatch.setattr(hp, 'PREREG_SHA256', io_.sha_file(W.ROOT/hp.PREREG_REL))
    layout = hin.holdout_layout(checkout, pairs=[row])
    pair = layout.pairs[0]
    W.finalized(layout, {pair: W.no_output_row()})
    return W.namespace(layout=layout, pair=pair, row=row, tmp=tmp_path)


def subject_files_opened(refusal, pair):
    fixed, moving = pair.split('__')
    return [f['path'] for f in (refusal['spy'] or {}).get('files', []) if fixed in f['path'] or moving in f['path']]


def test_without_a_pinned_preregistration_the_units_refuse(holdout, monkeypatch):
    monkeypatch.setattr(hp, 'PREREG_SHA256', None)
    assert W.run_out(holdout.layout, deps=W.deps()) == hu.EXIT['refused']
    assert io_.read_json(W.refusals(holdout.layout)[-1])['stage'] == 'preregistration'


def test_the_preregistration_pin_is_the_committed_file():
    """P0: pqh_pins.PREREG_SHA256 is the sha256 of the committed preregistration; the units' check (pqh_units.prereg_problem) finds no problem on this tree."""
    P0.assert_pinned(hp.PREREG_SHA256, hp.PREREG_REL)
    assert hu.prereg_problem(W.ROOT) is None


@pytest.mark.parametrize('unit', ['out', 'members'])
def test_without_the_unseal_record_both_units_are_refused_at_the_admission(holdout, unit):
    W.XW.sealed_module._reset_cache()
    run = hu.out_unit if unit == 'out' else hu.members_unit
    assert run(holdout.layout, deps=W.deps()) == hu.EXIT['refused']
    refusal = io_.read_json(W.refusals(holdout.layout)[-1])
    assert refusal['stage'] == 'unseal' and refusal['pair'] == unit and 'unseal' in refusal['reason']
    assert subject_files_opened(refusal, holdout.pair) == [] and not Path(holdout.layout.out_lock).exists() and not Path(holdout.layout.members_lock).exists()


def test_with_the_unseal_object_the_admission_passes_and_the_order_refuses_next(holdout, monkeypatch):
    XW.simulated_unseal(holdout.tmp, monkeypatch)
    assert W.run_out(holdout.layout, deps=W.deps()) == hu.EXIT['refused']
    refusal = io_.read_json(W.refusals(holdout.layout)[-1])
    assert refusal['stage'] == 'order' and refusal['checks']['admission'][holdout.pair]['unseal']['sealed_list'] == 'holdout_B_v1'


def test_the_other_direction_of_a_pair_is_refused(holdout, monkeypatch):
    XW.simulated_unseal(holdout.tmp, monkeypatch)
    fixed, moving = holdout.row
    layout = hin.holdout_layout(holdout.layout.code, pairs=[(moving, fixed)])
    W.finalized(layout, {layout.pairs[0]: W.no_output_row()})
    assert W.run_out(layout, deps=W.deps()) == hu.EXIT['refused']
    refusal = io_.read_json(W.refusals(layout)[-1])
    assert refusal['stage'] == 'admission' and 'other way round' in refusal['reason']


def test_a_code_tree_that_is_not_the_checkout_is_refused(holdout):
    checkout = Path(hc.HOLDOUT_ROOT)/'code'
    checkout.unlink()
    checkout.mkdir()                                                                # the detached checkout is another tree than the one that runs
    layout = hin.holdout_layout(W.ROOT, pairs=[holdout.row])
    assert W.run_out(layout, deps=W.deps()) == hu.EXIT['refused']
    assert io_.read_json(W.refusals(layout)[-1])['stage'] == 'code'


def test_the_units_take_p1_from_the_pinned_record_only(holdout, monkeypatch):
    """The lead's instruction after HB's merge: rule one through holdout_common.pinned_code_problems with no pinned commit given (P1 only from the committed runs/holdout_B_v1/pinned_code.json;
    the finalization record's code.commit is recorded, not used); this tree has no such record, so the real check refuses at 'code'."""
    calls = []
    original = hc.pinned_code_problems

    def spy(pinned=None, *, repo=None, run=None):
        calls.append(dict(pinned=pinned, repo=repo))
        return original(pinned, repo=repo, run=run)
    monkeypatch.setattr(hc, 'pinned_code_problems', spy)
    assert W.run_out(holdout.layout, deps=W.deps(identity=None)) == hu.EXIT['refused']
    refusal = io_.read_json(W.refusals(holdout.layout)[-1])
    assert refusal['stage'] == 'code' and 'pinned_code.json' in refusal['reason']
    assert calls == [dict(pinned=None, repo=holdout.layout.code)]
    assert refusal['checks']['code']['commit'] is None and refusal['checks']['code']['clean'] is False and refusal['checks']['code']['finalization_record_code'] == 'c'*40


def test_a_code_tree_whose_identity_has_problems_is_refused(holdout):
    record = dict(commit='d'*40, head='e'*40, clean=False, identity=None, rule='test')
    assert W.run_out(holdout.layout, deps=W.deps(identity=lambda layout: (record, ['a code directory changed (test)']))) == hu.EXIT['refused']
    refusal = io_.read_json(W.refusals(holdout.layout)[-1])
    assert refusal['stage'] == 'code' and 'a code directory changed' in refusal['reason']


def test_code_imported_from_elsewhere_is_refused(holdout):
    """m7: bayesreg, this package, holdout_common and hi_core must come from the checkout."""
    assert hu.import_problems(holdout.layout) == []                                 # the checkout links to this tree: the imports are its own
    other = W.namespace(code=holdout.tmp/'elsewhere')
    problems = hu.import_problems(other)
    assert any(p.startswith('bayesreg is imported from') for p in problems) and any(p.startswith('jd_curl') for p in problems)


def test_the_evaluations_master_lock_refuses(holdout):
    lock = Path(holdout.layout.ev_master_lock)
    lock.parent.mkdir(parents=True)
    lock.write_text('{}\n')
    assert W.run_out(holdout.layout, deps=W.deps()) == hu.EXIT['refused']
    refusal = io_.read_json(W.refusals(holdout.layout)[-1])
    assert refusal['stage'] == 'order' and 'evaluation has started' in refusal['reason']


# ---- the code identity at PQH's entry: rule one is HB's (bayesreg.io.code_identity through holdout_common); the repositories built by HB's own test helpers --------------------------------
from ..holdout import test_pinned_code as PC  # noqa: E402


def identity_repo(tmp_path):
    """HB's test repository: P1, then the inclusion commit (the allowlist modified, the unseal record, runs/holdout_B_v1/pinned_code.json and the decision log's line, one commit)."""
    return PC.repo(tmp_path, ignore='__pycache__/\n*.pyc\n')


def test_the_entrys_record_maps_hbs_fields(tmp_path):
    """Review NM1: commit = P1 (the pinned record's), head = HEAD, clean = passed; HB's whole record under identity.  The allowlist's one allowance passes (the narrow exception)."""
    r, pin = identity_repo(tmp_path)
    code, problems = hu.unit_code_identity(r)
    head = W.XW.git(r, 'rev-parse', 'HEAD').stdout.strip()
    assert problems == [] and code['commit'] == pin and code['head'] == head != pin and code['clean'] is True and code['identity']['code_dirs_differ'] == [PC.ALLOW]
    code, problems = hu.unit_code_identity(tmp_path/'nowhere')
    assert problems and code['commit'] is None and code['clean'] is False


@pytest.mark.parametrize('change, ok', [
    ({'splits/bayesreg_v1/holdout_B_v1_pqh_rerun_1.json': '{}'}, True), ({'splits/bayesreg_v1/holdout_B_v1_pqh_resume_2.json': '{}'}, True),
    ({'splits/bayesreg_v1/holdout_B_v1_pqh_no_resume_1.json': '{}'}, True), ({'runs/holdout_B_v1/pq_release.md': '放行：x\n'}, True),
    ({'runs/holdout_B_v1/pq_outputs_manifest.sha256': 'x\n'}, True), ({'experiments/pq/pqh_units.py': 'changed\n'}, False), ({'configs/c.json': 'changed\n'}, False),
    ({'scripts/new.py': 'x\n'}, False), ({'docs/other.md': 'x\n'}, False), ({'configs/bayesreg/dev_subjects.json': '{"included": []}\n'}, False)])
def test_later_commits_at_the_entry(tmp_path, change, ok):
    """Rule one commit by commit: PQH's own authorizations and records may be added; code may not change; the allowlist only once, with the unseal record (a second change refuses)."""
    r, _ = identity_repo(tmp_path)
    for rel, text in change.items():
        PC.write(r, rel, text)
    PC.commit(r)
    code, problems = hu.unit_code_identity(r)
    assert (problems == []) == ok, problems


@pytest.mark.parametrize('rel, ok', [('scripts/new.py', False), ('experiments/new_module.py', False), ('experiments/x.pyc', False), ('jd_curl/data.bin', False),
                                     ('experiments/__pycache__/x.cpython-310.pyc', False), ('experiments/he/__pycache__/run_hx.cpython-310.pyc', True), ('sitecustomize.py', False),
                                     ('runs/other/x.txt', True)])
def test_untracked_and_ignored_files_at_the_entry(tmp_path, rel, ok):
    """HB's rule one over the working tree: a compiled file is allowed only beside its source (the source-less .pyc of the old local copy now refuses: review NM1), and an importable file at the
    code tree's root refuses too."""
    r, _ = identity_repo(tmp_path)
    PC.write(r, rel, 'x\n')
    code, problems = hu.unit_code_identity(r)
    assert (problems == []) == ok, problems


def test_the_decision_log_is_only_appended_to(tmp_path):
    r, _ = identity_repo(tmp_path)
    log = (r/'docs/decision_log.md').read_text()
    (r/'docs/decision_log.md').write_text('rewritten\n'+log)                       # the old bytes are no longer a prefix
    PC.commit(r)
    assert hu.unit_code_identity(r)[1]


# ---- after a run: once, the control rerun, the resume --------------------------------------------------------------------------------------------
commit_authorization = W.commit_authorization


def stopped_first_run(layout):
    lock = Path(layout.out_lock)
    hc.take_lock(lock, hc.lock_line(tool=hp.TOOL, unit='out', kind='first_run'))
    stop = XW.write_json(Path(layout.pq_small)/hu.STOPPED, dict(unit='out', stage='hm1', error='test'))
    return lock, stop


def test_after_a_control_failure_the_unit_reruns_only_with_an_authorization_once(holdout):
    layout = holdout.layout
    lock, stop = stopped_first_run(layout)
    with pytest.raises(hu.Refusal, match='rerun.*committed authorization'):
        hu.lock_plan(layout, 'out', lock, None, {}, lambda: False)
    commit_authorization('rerun', 'out', refusals=[dict(path=str(Path(layout.pq_small)/'other.json'), sha256='0'*64)])
    with pytest.raises(hu.Refusal, match='rerun'):
        hu.lock_plan(layout, 'out', lock, None, {}, lambda: False)                # an authorization that does not list exactly the stop record
    commit_authorization('rerun', 'out', refusals=[dict(path=str(stop), sha256=io_.sha_file(stop))])
    plan = hu.lock_plan(layout, 'out', lock, None, {}, lambda: False)
    assert plan['kind'] == 'control_rerun' and plan['lock'] == lock.with_name(lock.name+'.rerun')
    assert hin.run_dirs(layout, 'out', plan['kind'])[0] == Path(layout.pq_small)/'rerun'
    hc.take_lock(plan['lock'], hc.lock_line(tool=hp.TOOL, unit='out', kind='control_rerun'))
    with pytest.raises(hu.Refusal, match='no second rerun'):
        hu.lock_plan(layout, 'out', lock, None, {}, lambda: False)


def test_a_control_rerun_stopped_before_its_lock_needs_a_new_authorization(holdout):
    """Review NM4: a control rerun that stops before its lock writes pq_refused.json in rerun/ (stop_before_lock with the plan); that record counts as a stop, so the authorization that let the rerun
    start no longer covers the stops; a new one listing all of them lets it start (its .rerun lock not taken yet).  The members unit's records of its rerun count the same way."""
    layout = holdout.layout
    lock, stop = stopped_first_run(layout)
    commit_authorization('rerun', 'out', refusals=[dict(path=str(stop), sha256=io_.sha_file(stop))])
    plan = hu.lock_plan(layout, 'out', lock, None, {}, lambda: False)
    assert plan['kind'] == 'control_rerun'
    assert hu.stop_before_lock(layout, 'out', None, {}, 'hm6', 'the sealed-list cache (test)', W.quiet, plan) == hu.EXIT['stopped']
    rerun_stop = Path(layout.pq_small)/'rerun'/hu.STOPPED
    assert rerun_stop.is_file() and sorted(hu.previous_stop_records(layout, 'out')) == sorted([stop, rerun_stop])
    with pytest.raises(hu.Refusal, match='rerun'):
        hu.lock_plan(layout, 'out', lock, None, {}, lambda: False)
    commit_authorization('rerun', 'out', refusals=[dict(path=str(p), sha256=io_.sha_file(p)) for p in (stop, rerun_stop)])
    assert hu.lock_plan(layout, 'out', lock, None, {}, lambda: False)['kind'] == 'control_rerun'
    small, _ = hin.run_dirs(layout, 'members', 'control_rerun')
    members_stop = XW.write_json(small/hu.STOPPED, dict(unit='members', stage='hm6', error='test', before_lock=True))
    assert members_stop in hu.previous_stop_records(layout, 'members')


def test_a_second_run_needs_a_resume_record_and_resumes_once(holdout):
    layout = holdout.layout
    lock = Path(layout.out_lock)
    hc.take_lock(lock, hc.lock_line(tool=hp.TOOL, unit='out', kind='first_run'))
    with pytest.raises(hu.Refusal, match='once'):
        hu.lock_plan(layout, 'out', lock, None, {}, lambda: False)
    previous = Path(layout.pq_small)/hu.OUT_RECORD
    path = commit_authorization('resume', 'out', lock_sha256=io_.sha_file(lock), previous_record=str(previous), previous_record_sha256=None)
    rel = path.relative_to(hc.AUTH_ROOT)
    plan = hu.lock_plan(layout, 'out', lock, str(rel), {}, lambda: False)
    assert plan['kind'] == 'resume' and plan['lock'] == hc.resume_lock_path(lock)
    with pytest.raises(hu.Refusal, match='resume'):
        hu.lock_plan(layout, 'out', lock, str(rel), {}, lambda: True)              # a complete record is never resumed
    hc.take_lock(plan['lock'], hc.lock_line(tool=hp.TOOL, unit='out', kind='resume'))
    with pytest.raises(hu.Refusal, match='resume'):
        hu.lock_plan(layout, 'out', lock, str(rel), {}, lambda: False)             # at most once


def test_a_refusal_before_the_lock_needs_a_rerun_authorization(holdout):
    layout = holdout.layout
    record = hu.write_refusal(layout, 'out', dict(stage='machine', refusal_class='infrastructure', reason='test'), W.quiet)
    with pytest.raises(hu.Refusal, match='rerun'):
        hu.lock_plan(layout, 'out', layout.out_lock, None, {}, lambda: False)
    commit_authorization('rerun', 'out', refusals=[dict(path=str(record), sha256=io_.sha_file(record))])
    checks = {}
    assert hu.lock_plan(layout, 'out', layout.out_lock, None, checks, lambda: False)['kind'] == 'first_run'
    assert checks['rerun_authorization']['refusals'] == [dict(path=str(record), sha256=io_.sha_file(record))]


def test_a_members_stop_record_is_a_control_failure_and_an_error_record_is_not(holdout):
    layout = holdout.layout
    small, _ = hin.run_dirs(layout, 'members', 'first_run')
    XW.write_json(small/hin.MEMBERS_RECORD, dict(stage='error', spy=dict(m6=True), m5_passed=False))
    assert hu.previous_stop_records(layout, 'members') == []
    (small/hin.MEMBERS_RECORD).unlink()
    XW.write_json(small/hin.MEMBERS_RECORD, dict(stage='hm5', spy=dict(m6=True), m5_passed=False))
    assert hu.previous_stop_records(layout, 'members') == [small/hin.MEMBERS_RECORD]


def test_the_layout_of_the_hold_out_mode_reads_holdout_common_now(holdout):
    layout = holdout.layout
    assert layout.pq_small == Path(hc.HOLDOUT_ROOT)/'pq' and layout.pq_big == Path(hc.BIG_ROOT)/'pq' and layout.stage == Path(hc.STAGE_ROOT)
    assert layout.finalized == Path(hc.HOLDOUT_ROOT)/'review'/'finalized_outputs.json' and layout.method_records == Path(hc.HOLDOUT_ROOT)/'method' and hp.METHOD_ARM == 'method' and Path(hp.MEMBERS_LOCK) != Path(hp.DEV_MEMBERS_LOCK)
    assert Path(hc.HOLDOUT_ROOT)/hp.EV_EVAL_DIR_NAME in layout.forbidden_dirs and Path(hp.PQ_V1) in layout.forbidden_dirs
    with pytest.raises(hc.RealRootInTest):
        hc.guard_root('/home/rmcfz/seq_vp_tag_results/holdout_B_v1/pq')
