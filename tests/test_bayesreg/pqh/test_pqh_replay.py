"""PQH, the development replay gate (D119 (a), contract C22.H8; plan runs/20261109-pqh/replay_gate_plan.md) on the PQ package's synthetic world: its development reading (run_pq: out, the
release, eval, decide) is the reference; EV's three functions are stood in by a reference implementation of the interface (REFERENCE_EV below: pq_eval.evaluate_pair's per-pair part on arrays,
class B with the resampling passed in).  Passed: 逐位相同 n/n; one value off: 不过; a missing function, an unpinned plan, a changed reference file: refused before the lock; the decision constant's
check for the evaluation entry's step (0) (CPU, tmp_path)."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from .. import _p0_pins as P0
from .. import _p1_decisions as P1
from . import _world as W

hp, io_, core = W.hp, W.io_, W.core
import pq_eval as pe  # noqa: E402
import pq_stats as stats  # noqa: E402
import pqh_replay as rp  # noqa: E402


def truth_of(subject):
    T = W.PW.truth35(subject)
    return dict(seg35=T.astype(float), seg4=W.PW.LOOKUP[T].astype(float), files={})


def pair_acceptance(*, truth_fixed, truth_moving, d, pq, baselines, spacing_xyz):
    """The interface pq_pair_acceptance, as pq_eval.evaluate_pair computes it from arrays (a reference for EV, not EV's code)."""
    T_F, T_tilde = np.asarray(truth_fixed['seg35']), core.warp_labels_l2r(truth_moving['seg35'], d)
    w4 = core.warp_labels_l2r(truth_moving['seg4'], d)
    e, e4 = T_F != T_tilde, np.asarray(truth_fixed['seg4']) != w4
    masks = dict(omega_e=(T_F > 0) | (T_tilde > 0), band=pe.band_mask(T_F, T_tilde, spacing_xyz), whole=np.ones(T_F.shape, dtype=bool))
    keys = pe.score_keys(pq, baselines)
    scores = {m: {n: stats.score_record(pe.masked(k, mask), e[mask]) for n, k in keys.items()} for m, mask in masks.items()}
    return dict(e=e, e4=e4, scores=scores)


def b_class(ause_table, *, interval):
    """The interface pq_b_class with the resampling passed in (pq_stats.b_class's arithmetic)."""
    q = np.asarray(ause_table['q'], dtype=np.float64)
    rows = {}
    for k in stats.JUDGED_BASELINES:
        base = np.asarray(ause_table[k], dtype=np.float64)
        delta = base-q
        lo = interval(delta)['lo']
        rows[k] = dict(delta=delta.tolist(), LB=lo, R=float(delta.sum()/base.sum()))
    return dict(baselines=rows, passed=all(r['LB'] > 0 and r['R'] >= stats.R_MIN for r in rows.values()))


REFERENCE_EV = dict(truth=truth_of, pair=pair_acceptance, b_class=b_class)


@pytest.fixture
def world(tmp_path, monkeypatch):
    import run_pq
    w, layout, entry = W.dev_world(tmp_path, monkeypatch)
    W.XW.sealed_module._reset_cache()
    assert run_pq.out_unit(w.ctx, gate=W.GATE, code=W.CODE, log=W.quiet) == 0
    W.PW.release(w, io_.sha_file(w.ctx.pq_root/io_.MANIFEST_FILE))
    W.XW.sealed_module._reset_cache()
    assert run_pq.eval_unit(w.ctx, gate=W.GATE, code=W.CODE, ev=W.PW.EV, read_truth=w.truth, guard=W.PW.guard_of(w), log=W.quiet) == 0
    assert run_pq.decide_unit(w.ctx, log=W.quiet) == 0
    reference = {rel: io_.sha_file(w.ctx.pq_root/rel) for rel in [rp.reference_rel('manifest'), rp.reference_rel('decision'), rp.reference_rel('record')]
                 + [rp.reference_rel(k, s) for s in w.ctx.shorts for k in ('eval_json', 'e_npz')]}
    return W.namespace(w=w, reference=reference, tmp=tmp_path)


def run(world, ev=None, **kwargs):
    return rp.replay(world.w.ctx, ev=ev or REFERENCE_EV, out_root=world.tmp/'replay', lock=world.tmp/'replay.lock', guard=lambda: ['dev pairs (test)'], reference=world.reference,
                     plan=dict(path=hp.REPLAY_PLAN_REL, sha256='p'*64), log=W.quiet, **kwargs)


def test_the_reference_implementation_replays_bit_for_bit(world):
    result = run(world)
    assert result['reading'] == hp.REPLAY_READING_PASS.replace('19', str(len(world.w.ctx.shorts))) and result['error'] is None
    assert all(r['equal'] for r in result['pairs'].values()) and result['b_class']['differences'] == []
    assert all(r['reference_nonnull'] > 0 and r['compared_nonnull'] == r['reference_nonnull'] for r in result['pairs'].values())          # m6
    head = W.XW.git(world.w.repo, 'rev-parse', 'HEAD').stdout.strip()
    assert result['code']['commit'] == head and result['code']['clean'] is True and sorted(result['code']['files']) == sorted(hp.REPLAY_CODE_FILES)      # I4
    assert sorted(result['environment']) == sorted(hp.REPLAY_ENVIRONMENT_KEYS) and result['environment'] == rp.environment()
    line = json.loads((world.tmp/'replay.lock').read_text().splitlines()[0])
    assert line['code_commit'] == head and line['clean'] is True and line['environment'] == rp.environment()
    assert Path(world.tmp/'replay.lock').is_file() and io_.read_json(world.tmp/'replay'/rp.RESULT_FILE)['reading'] == result['reading']
    with pytest.raises(rp.ReplayRefused, match='once'):
        run(world)


def test_one_value_off_is_not_passed(world):
    def off(**kwargs):
        out = pair_acceptance(**kwargs)
        out['scores']['band']['H2']['auroc'] = np.nextafter(out['scores']['band']['H2']['auroc'], 2.)
        return out
    result = run(world, ev=dict(REFERENCE_EV, pair=off))
    assert result['reading'] == hp.REPLAY_READING_FAIL and any(d['item'] == 'band.H2.auroc' for r in result['pairs'].values() for d in r['differences'])


def test_another_resampling_is_not_passed(world):
    def other(ause_table, *, interval):
        return b_class(ause_table, interval=lambda delta: dict(lo=float(np.percentile(delta, 2.5))))
    result = run(world, ev=dict(REFERENCE_EV, b_class=other))
    assert result['reading'] == hp.REPLAY_READING_FAIL and result['b_class']['differences']


def test_an_ev_function_that_fails_after_the_lock_is_undecided(world):
    def broken(**kwargs):
        raise RuntimeError('not implemented')
    result = run(world, ev=dict(REFERENCE_EV, pair=broken))
    assert result['reading'] == hp.REPLAY_READING_UNDECIDED and 'not implemented' in result['error']


def test_an_empty_comparison_is_not_passed():
    """m6: two nulls are the same, but a pair passes only when the reference has non-null items and all of them were compared."""
    e = dict(e=np.zeros(2, dtype=np.uint8), e4=np.zeros(2, dtype=np.uint8))
    counts = {}
    assert rp.compare_pair(dict(scores={}, **e), dict(scores={}), e, counts) == [] and counts == dict(reference_nonnull=0, compared_nonnull=0)
    assert not rp.pair_equal([], counts)
    assert rp.pair_equal([], dict(reference_nonnull=3, compared_nonnull=3)) and not rp.pair_equal([], dict(reference_nonnull=3, compared_nonnull=2))


def test_a_dirty_tree_refuses_the_gate_before_the_lock(world):
    """I4: the gate runs on a clean tree (pqh_units.pinned_code_problems against HEAD); its commit goes into the result and the lock line."""
    (world.w.repo/'configs'/'bayesreg'/'extra.json').write_text('{}')
    with pytest.raises(rp.ReplayRefused, match='code'):
        run(world)
    assert not (world.tmp/'replay.lock').exists()


def test_refusals_before_the_lock(world, monkeypatch):
    with pytest.raises(rp.ReplayRefused, match='ev'):
        rp.ev_functions(module=SimpleNamespace(pq_dev_truth=truth_of))
    monkeypatch.setattr(hp, 'REPLAY_PLAN_SHA256', None)             # P0 pinned it (runs/20261114-p0/P0_pins.md); the unpinned state, set back here, as before
    with pytest.raises(rp.ReplayRefused, match='plan.*None'):
        rp.check_plan(W.ROOT)
    changed = dict(world.reference)
    changed[rp.reference_rel('decision')] = '0'*64
    with pytest.raises(rp.ReplayRefused, match='reference'):
        rp.replay(world.w.ctx, ev=REFERENCE_EV, out_root=world.tmp/'r', lock=world.tmp/'r.lock', guard=lambda: [], reference=changed, plan=dict(sha256='p'*64), log=W.quiet)
    assert not (world.tmp/'r.lock').exists() and not (world.tmp/'r').exists()


def test_the_replay_plan_pin_is_the_committed_plan():
    """P0: REPLAY_PLAN_SHA256 is the sha256 of the committed replay_gate_plan.md; check_plan accepts it on this tree and refuses another value (only the plan is read)."""
    P0.assert_pinned(hp.REPLAY_PLAN_SHA256, hp.REPLAY_PLAN_REL)
    assert rp.check_plan(W.ROOT) == dict(path=hp.REPLAY_PLAN_REL, sha256=hp.REPLAY_PLAN_SHA256)
    with pytest.raises(rp.ReplayRefused, match='is not the pinned'):
        rp.check_plan(W.ROOT, pinned='0'*64)


def test_the_pinned_references_are_the_development_readings_files():
    """The pins name the released list and the 19 pairs' evaluation files; the released list's sha256 is runs/20261107-pq/outputs_manifest.sha256's (no file of pq_v1 is opened here)."""
    shorts = W.pins.SHORTS
    want = {rp.reference_rel('manifest'), rp.reference_rel('decision'), rp.reference_rel('record')} | {rp.reference_rel(k, s) for s in shorts for k in ('eval_json', 'e_npz')}
    assert set(rp.REFERENCE_SHA256) == want
    released = (W.ROOT/'runs'/'20261107-pq'/'outputs_manifest.sha256').read_text().split()[0]
    assert rp.REFERENCE_SHA256['outputs_manifest.txt'] == released


PINS_REL = 'experiments/pq/pqh_pins.py'
FILLED = "PQ_REPLAY_DECISION = {\n    'reading': '逐位相同 19/19',\n}     # filled (test)\n"


def decision_repo(tmp_path):
    """A code tree whose experiments/pq/pqh_pins.py is this one as at the gate's commit (P1 filled PQ_REPLAY_DECISION: P1.at_gate sets it back to None, P0's bytes), and the commit that fills
    PQ_REPLAY_DECISION after it as a plain dict literal (contract revision 9 draft C24.3-6 (a))."""
    repo = tmp_path/'code'
    (repo/'experiments'/'pq').mkdir(parents=True)
    W.XW.git(repo, 'init', '-q')
    text = P1.at_gate((W.ROOT/PINS_REL).read_text(encoding='utf-8'), 'PQ_REPLAY_DECISION')
    (repo/PINS_REL).write_text(text, encoding='utf-8')
    (repo/'experiments'/'pq'/'pq_core.py').write_text('# test\n')
    W.XW.git(repo, 'add', '-A')
    W.XW.git(repo, 'commit', '-q', '-m', 'the gate ran here')
    gate = W.XW.git(repo, 'rev-parse', 'HEAD').stdout.strip()
    filled = text.replace('PQ_REPLAY_DECISION = None\n', FILLED)
    assert filled != text
    (repo/PINS_REL).write_text(filled, encoding='utf-8')
    W.XW.git(repo, 'commit', '-q', '-am', 'the decision')
    return repo, gate


def gate_problems(gate, repo, run=None):
    from bayesreg.io import code_identity
    return code_identity.gate_identity_problems(gate, repo=repo, run=run)[1]


def decision_for(tmp_path, gate):
    """The decision of a gate that ran here: its result at the gate's one path rp.result_path() (REPLAY_ROOT moved into tmp_path by the sandbox; B4 m2)."""
    env = rp.environment()
    result = rp.result_path()
    assert Path(result).is_relative_to(tmp_path)
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(dict(reading=hp.REPLAY_READING_PASS, plan=dict(sha256='p'*64), code=dict(commit=gate, head=gate, clean=True), environment=env)))
    return {'result_json': str(result), 'result_sha256': io_.sha_file(result), 'reading': hp.REPLAY_READING_PASS, 'code_commit': gate, 'plan_sha256': 'p'*64, 'environment': env,
            'decided_by': 'lead', 'review': 'r'}


def test_the_decision_check_for_the_evaluation_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(hp, 'PQ_REPLAY_DECISION', None)                    # P1 filled it (runs/20261115-p1); the P0 state, set back here: the old refusal
    assert 'None' in rp.replay_decision_problem()
    repo, gate = decision_repo(tmp_path)
    decision = decision_for(tmp_path, gate)
    env = decision['environment']
    monkeypatch.setattr(hp, 'REPLAY_PLAN_SHA256', 'p'*64)
    assert rp.replay_decision_problem(decision, repo=repo) is None
    assert 'keys' in rp.replay_decision_problem(dict(decision, extra=1), repo=repo)
    no_env = {k: v for k, v in decision.items() if k != 'environment'}
    assert 'keys' in rp.replay_decision_problem(no_env, repo=repo)                                       # I4: the environment is a key of the constant
    assert 'reading' in rp.replay_decision_problem(dict(decision, reading=hp.REPLAY_READING_FAIL), repo=repo)
    assert 'sha256' in rp.replay_decision_problem(dict(decision, result_sha256='0'*64), repo=repo)
    assert 'environment' in rp.replay_decision_problem(dict(decision, environment=dict(env, numpy='0.0')), repo=repo)
    assert 'code_commit' in rp.replay_decision_problem(dict(decision, code_commit='0'*40), repo=repo)
    monkeypatch.setattr(rp, 'environment', lambda: dict(env, numpy='9.9'))
    assert 'environment now' in rp.replay_decision_problem(decision, repo=repo)
    monkeypatch.undo()
    monkeypatch.setattr(hp, 'REPLAY_PLAN_SHA256', None)
    assert 'plan_sha256' in rp.replay_decision_problem(decision, repo=repo)


def test_the_decision_names_the_gates_own_result_file(tmp_path, monkeypatch):
    """B4 review m2 (the lead's ruling 2 of 11-14): result_json must be the gate's result REPLAY_ROOT/result/replay_result.json (rp.result_path(), where main writes it), as the truth-reader
    gate checks its own; a copy of the very bytes elsewhere (same sha256, same reading) is refused, and so is a path that only resolves elsewhere.  The written path with a '..' that resolves to the
    gate's file is the same file and passes."""
    repo, gate = decision_repo(tmp_path)
    decision = decision_for(tmp_path, gate)
    monkeypatch.setattr(hp, 'REPLAY_PLAN_SHA256', 'p'*64)
    assert rp.replay_decision_problem(decision, repo=repo) is None
    assert Path(decision['result_json']) == Path(hp.REPLAY_ROOT)/'result'/'replay_result.json' == rp.result_path()
    copy = tmp_path/'elsewhere'/'replay_result.json'
    copy.parent.mkdir()
    copy.write_bytes(Path(decision['result_json']).read_bytes())
    assert io_.sha_file(copy) == decision['result_sha256']
    problem = rp.replay_decision_problem(dict(decision, result_json=str(copy)), repo=repo)
    assert problem and "is not the gate's result" in problem
    same = Path(hp.REPLAY_ROOT)/'result'/'..'/'result'/'replay_result.json'
    assert rp.replay_decision_problem(dict(decision, result_json=str(same)), repo=repo) is None
    other_root = tmp_path/'pq_replay_v2'/'result'/'replay_result.json'                 # the right name under another root
    other_root.parent.mkdir(parents=True)
    other_root.write_bytes(copy.read_bytes())
    assert "is not the gate's result" in rp.replay_decision_problem(dict(decision, result_json=str(other_root)), repo=repo)
    monkeypatch.setattr(hp, 'REPLAY_ROOT', tmp_path/'pq_replay_v2')                   # the path follows REPLAY_ROOT (read at call time)
    assert rp.replay_decision_problem(dict(decision, result_json=str(other_root)), repo=repo) is None
    assert "is not the gate's result" in rp.replay_decision_problem(decision, repo=repo)


def test_the_result_path_is_compared_resolved_as_the_truth_gate_does(tmp_path, monkeypatch):
    """P0S revision 2 (the code review's minor 7): result_json is compared after resolve(), the truth-reader gate's way (bayesreg/eval/holdout.py's own check): a symbolic link elsewhere
    that leads to the gate's result is that file and passes; one that leads to a byte-identical copy elsewhere is refused."""
    repo, gate = decision_repo(tmp_path)
    decision = decision_for(tmp_path, gate)
    monkeypatch.setattr(hp, 'REPLAY_PLAN_SHA256', 'p'*64)
    links = tmp_path/'links'
    links.mkdir()
    (links/'to_the_result.json').symlink_to(rp.result_path())
    assert rp.replay_decision_problem(dict(decision, result_json=str(links/'to_the_result.json')), repo=repo) is None
    copy = tmp_path/'copy.json'
    copy.write_bytes(rp.result_path().read_bytes())
    (links/'to_a_copy.json').symlink_to(copy)
    assert "is not the gate's result" in rp.replay_decision_problem(dict(decision, result_json=str(links/'to_a_copy.json')), repo=repo)


def test_main_writes_the_result_where_the_decision_must_name_it(monkeypatch):
    """B4 m2: main('run') hands replay() the directory of rp.result_path() (so the file the decision names is the one the run wrote)."""
    import hi_core
    import holdout_common
    import mc0_core
    import pqh_units
    monkeypatch.setattr(rp, 'check_plan', lambda repo, pinned=None: dict(path=hp.REPLAY_PLAN_REL, sha256='p'*64))
    monkeypatch.setattr(rp, 'ev_functions', lambda module=None: dict(REFERENCE_EV))
    monkeypatch.setattr(hi_core, 'units_of_this_process', lambda units: list(units))
    monkeypatch.setattr(holdout_common, 'wait_for', lambda check, **kwargs: dict(passed=True))
    monkeypatch.setattr(pqh_units, 'import_problems', lambda layout: [])
    monkeypatch.setattr(mc0_core, 'frozen_api', lambda: SimpleNamespace(boundary_taper=W.PW.frozen_formula_taper))
    seen = {}

    def replay(ctx, *, ev, out_root, lock, **kwargs):
        seen.update(out_root=Path(out_root), lock=Path(lock))
        return dict(reading=hp.REPLAY_READING_PASS)
    monkeypatch.setattr(rp, 'replay', replay)
    assert rp.main(['run']) == 0
    assert seen['out_root']/rp.RESULT_FILE == rp.result_path() and seen['lock'] == Path(hp.REPLAY_LOCK)


def test_the_constant_filled_as_a_plain_dict_literal_is_read(tmp_path, monkeypatch):
    """Item 6 (contract revision 9 draft C24.3-6 (a)): PQ_REPLAY_DECISION filled as a plain dict literal -- what rule two lets change -- is read by replay_decision_problem() (the constant
    itself, decision=None) and by the evaluation entry's wrapper (bayesreg.eval.holdout.replay_decision_problem, which calls PQH's)."""
    from bayesreg.eval import holdout
    repo, gate = decision_repo(tmp_path)
    decision = decision_for(tmp_path, gate)
    assert type(decision) is dict
    monkeypatch.setattr(hp, 'REPLAY_PLAN_SHA256', 'p'*64)
    monkeypatch.setattr(hp, 'PQ_REPLAY_DECISION', decision)
    assert rp.replay_decision_problem(repo=repo) is None and holdout.replay_decision_problem(repo=repo) is None
    assert json.loads(json.dumps(hp.PQ_REPLAY_DECISION)) == decision


def test_the_replay_decision_is_the_gate_that_ran_on_p0pp():
    """P1 (prereg revision 2 item 9 step 5; runs/20261115-d58/result_C6.md): PQ_REPLAY_DECISION is a plain dict literal with exactly REPLAY_DECISION_KEYS and the values of the gate of
    2026-11-15 on P0'' (written again in tests/test_bayesreg/_p1_decisions.py): the pinned plan, the pass reading, the environment's keys, result_json the gate's own result under the real
    REPLAY_ROOT (a real write root of PQH); pqh_pins.py is the file of P0'' but for this constant (rule two's AST check on the file)."""
    decision = hp.PQ_REPLAY_DECISION
    P1.assert_values(decision, hp.REPLAY_DECISION_KEYS, P1.PQ)
    assert decision['plan_sha256'] == hp.REPLAY_PLAN_SHA256 and decision['reading'] == hp.REPLAY_READING_PASS and sorted(decision['environment']) == sorted(hp.REPLAY_ENVIRONMENT_KEYS)
    real_root = P1.RESULTS/'pq_replay_v1'
    assert real_root in [Path(p) for p in hp.REAL_WRITE_ROOTS] and Path(decision['result_json']) == real_root/'result'/rp.RESULT_FILE and P1.PQ_LOCK[0] == real_root/'pq_replay_gate.lock'
    now, gate = (W.ROOT/PINS_REL).read_text(encoding='utf-8'), P1.blob_at(P1.P0PP, PINS_REL)
    assert P1.at_gate(now, 'PQ_REPLAY_DECISION') == gate
    from bayesreg.io import code_identity
    assert code_identity.decision_assignment_problem(gate, now, PINS_REL) is None


def test_the_replay_decision_holds_against_the_real_gate(monkeypatch):
    """P1: replay_decision_problem() takes the constant against the real result and lock (read only: their sha256 before and after; REPLAY_ROOT set back to the real root for this read, nothing
    written), the lock's code (P0'', clean), plan and environment those of the constant; rule two is asked about P0'' (replaced here by a recorder: the range P0''..HEAD is tested in
    tests/test_bayesreg/io_seg/test_code_identity.py)."""
    before = P1.real_evidence(P1.PQ['result_json'], P1.PQ_LOCK[0])
    assert before == (P1.PQ['result_sha256'], P1.PQ_LOCK[1])
    line = P1.lock_line(P1.PQ_LOCK[0])
    assert line['code_commit'] == P1.P0PP and line['clean'] is True and line['plan_sha256'] == P1.PQ['plan_sha256'] and line['environment'] == P1.PQ['environment']
    real_root = P1.RESULTS/'pq_replay_v1'
    monkeypatch.setattr(hp, 'REPLAY_ROOT', real_root)
    monkeypatch.setattr(hp, 'REPLAY_LOCK', real_root/'pq_replay_gate.lock')
    from bayesreg.io import code_identity
    gates = []
    monkeypatch.setattr(code_identity, 'gate_identity_problems', lambda commit, *, repo=None, run=None, head='HEAD': (gates.append(commit), (dict(stub='test'), []))[1])
    assert rp.replay_decision_problem() is None and gates == [P1.P0PP]
    assert (P1.sha256_of(P1.PQ['result_json']), P1.sha256_of(P1.PQ_LOCK[0])) == before


@pytest.mark.parametrize('change', ['another constant of pqh_pins.py', 'another file of a code directory', 'uncommitted'])
def test_the_decision_is_stale_after_any_other_change_of_code(tmp_path, change):
    """I4 by rule two (bayesreg.io.code_identity.gate_identity_problems): from the gate's commit the code changes only by the decision constants' values."""
    repo, gate = decision_repo(tmp_path)
    assert gate_problems(gate, repo) == []
    if change == 'another constant of pqh_pins.py':
        text = (repo/PINS_REL).read_text(encoding='utf-8')
        (repo/PINS_REL).write_text(text.replace("TOOL = 'pqh'", "TOOL = 'pqh2'"), encoding='utf-8')
        W.XW.git(repo, 'commit', '-q', '-am', 'changed')
    elif change == 'another file of a code directory':
        (repo/'experiments'/'pq'/'pq_core.py').write_text('# changed\n')
        W.XW.git(repo, 'commit', '-q', '-am', 'changed')
    else:
        (repo/'experiments'/'pq'/'pq_core.py').write_text('# changed, not committed\n')
    assert gate_problems(gate, repo)


def test_the_run_records_a_refusal_of_the_gate(tmp_path, monkeypatch):
    """m6: main('run') writes a record under REPLAY_ROOT/refusals/ when the gate refuses (here: the lock is there), and returns 2."""
    import hi_core
    import holdout_common
    import mc0_core
    import pqh_units
    monkeypatch.setattr(rp, 'check_plan', lambda repo, pinned=None: dict(path=hp.REPLAY_PLAN_REL, sha256='p'*64))
    monkeypatch.setattr(rp, 'ev_functions', lambda module=None: dict(REFERENCE_EV))
    monkeypatch.setattr(hi_core, 'units_of_this_process', lambda units: list(units))
    monkeypatch.setattr(holdout_common, 'wait_for', lambda check, **kwargs: dict(passed=True))
    monkeypatch.setattr(pqh_units, 'import_problems', lambda layout: [])
    monkeypatch.setattr(mc0_core, 'frozen_api', lambda: SimpleNamespace(boundary_taper=W.PW.frozen_formula_taper))

    def refused(*args, **kwargs):
        raise rp.ReplayRefused('once', 'the lock exists (test)')
    monkeypatch.setattr(rp, 'replay', refused)
    assert rp.main(['run']) == 2
    records = sorted((Path(hp.REPLAY_ROOT)/'refusals').glob('*.json'))
    assert len(records) == 1 and io_.read_json(records[0])['stage'] == 'once'


@pytest.mark.parametrize('change', ['a statement next to the decision', 'an import next to the decision', 'a computed value', 'a second assignment', 'a comment line above it',
                                    'a statement on the same line', 'a mapping proxy'])
def test_only_the_decision_statement_may_differ(tmp_path, change):
    """Review NM2 and review 2 n1, now by rule two's AST check (bayesreg.io.code_identity.decision_assignment_problem): exactly the one top-level assignment to PQ_REPLAY_DECISION, a literal
    value, its lines to itself; anything else that differs from the gate's commit refuses -- a statement after a ';' on its line included (n1), and a MappingProxyType(...) value (not a literal:
    C24.3-6 (a))."""
    repo, gate = decision_repo(tmp_path)
    text = (repo/PINS_REL).read_text(encoding='utf-8')
    marker = 'REPLAY_DECISION_KEYS = '
    if change == 'a statement next to the decision':
        text = text.replace(marker, "REPLAY_READING_PASS = 'anything'\n"+marker, 1)
    elif change == 'an import next to the decision':
        text = text.replace(marker, 'import os\n'+marker, 1)
    elif change == 'a computed value':
        text = text.replace("    'reading': '逐位相同 19/19',\n", "    'reading': str(19),\n", 1)
    elif change == 'a second assignment':
        text = text.replace(marker, 'PQ_REPLAY_DECISION = None\n'+marker, 1)
    elif change == 'a comment line above it':
        text = text.replace('PQ_REPLAY_DECISION = {', '# a note (test)\nPQ_REPLAY_DECISION = {', 1)
    elif change == 'a statement on the same line':
        text = text.replace("}     # filled (test)\n", "}; REPLAY_READING_PASS = 'anything'     # filled (test)\n", 1)
    else:
        text = text.replace(FILLED, "PQ_REPLAY_DECISION = MappingProxyType(dict(reading='逐位相同 19/19'))\n", 1)
    assert text != (repo/PINS_REL).read_text(encoding='utf-8')
    (repo/PINS_REL).write_text(text, encoding='utf-8')
    W.XW.git(repo, 'commit', '-q', '-am', change)
    assert gate_problems(gate, repo), change


def test_the_replay_checks_never_raise(tmp_path):
    """Review NM3: a git that cannot run is a problem, never an exception (rule two never raises); replay_decision_problem returns its problem."""
    repo, gate = decision_repo(tmp_path)

    def broken(argv, **kwargs):
        raise OSError('no git (test)')
    assert gate_problems(gate, repo, run=broken)
    decision = decision_for(tmp_path, gate)
    import pytest as _pytest
    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(hp, 'REPLAY_PLAN_SHA256', 'p'*64)
        assert 'rule two' in rp.replay_decision_problem(decision, repo=repo, run=broken)
        mp.setattr(rp, '_replay_decision_problem', lambda *a, **k: (_ for _ in ()).throw(KeyError('code')))
        assert 'failed: KeyError' in rp.replay_decision_problem(decision, repo=repo)


def test_the_run_records_any_error_before_the_gates_own_handling(tmp_path, monkeypatch):
    """Review NM3: main('run') records every exception of the run (here an OSError of the references), with whether the lock was taken, and returns 2."""
    import hi_core
    import holdout_common
    import mc0_core
    import pqh_units
    monkeypatch.setattr(rp, 'check_plan', lambda repo, pinned=None: dict(path=hp.REPLAY_PLAN_REL, sha256='p'*64))
    monkeypatch.setattr(rp, 'ev_functions', lambda module=None: dict(REFERENCE_EV))
    monkeypatch.setattr(hi_core, 'units_of_this_process', lambda units: list(units))
    monkeypatch.setattr(holdout_common, 'wait_for', lambda check, **kwargs: dict(passed=True))
    monkeypatch.setattr(pqh_units, 'import_problems', lambda layout: [])
    monkeypatch.setattr(mc0_core, 'frozen_api', lambda: SimpleNamespace(boundary_taper=W.PW.frozen_formula_taper))
    monkeypatch.setattr(rp, 'replay', lambda *a, **k: (_ for _ in ()).throw(OSError('a reference file vanished (test)')))
    assert rp.main(['run']) == 2
    records = sorted((Path(hp.REPLAY_ROOT)/'refusals').glob('*.json'))
    document = io_.read_json(records[0])
    assert len(records) == 1 and document['stage'] == 'OSError' and 'vanished' in document['reason'] and document['lock_exists'] is False
