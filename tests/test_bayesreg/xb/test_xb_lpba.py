"""LP-B item 3 (runs/20261119-lpba/任务书_B_方法配置C与外部臂.md; contract C26.5-6, C26.4 L33, C26.13 revision 11d; LPBA40 prereg §4.3, §6.6): the external arms on an LPBA40 mode -- only the
four arms of the main family, the mode's roots (records, work, no store: nothing moves to D:), its stage, the admission on the pinned LPBA allowlist, units tag-lp-x-, P19 at the second suspicious
pair while the hold-out keeps its third.  XB's sandbox (xb/conftest.py) with the LPBA40 world of tests/test_bayesreg/hb/_lpba_world.py (LP-A's entries make the stage); the method step is XB's
FakeMethod.  No real subject's file, no unit."""
import json
from pathlib import Path

import pytest

from ..hb import _lpba_world as L
from . import _world as W

hc, xc, xb_drive = W.hc, W.xc, W.xb_drive
import xb_chain  # noqa: E402

FORMAL, FEAS, PAIR, FEAS_PAIR = L.FORMAL, L.FEAS, L.PAIR, L.FEAS_PAIR
REVERSE = 'LPBA40_S02__LPBA40_S01'


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = L.make(tmp_path, monkeypatch)
    L.prepare(w, monkeypatch, FEAS, hi=False, atropos=False)
    L.prepare(w, monkeypatch, FORMAL, hi=False, atropos=False)
    W.fake_checks(monkeypatch)
    return w


def deps(**kwargs):
    """XB's deps with the feasibility mode's code: a clean detached checkout (C26.9-1; review minor 5)."""
    d = W.deps(**kwargs)
    d.code_run = L.detached_git()
    return d


def result(arm, pair_set, pair):
    return json.loads(xc.cell_paths(arm, pair_set, pair)['result'].read_text())


def test_the_lpba_constants():
    assert xc.LPBA_ARMS == ('ants_s', 'fireants_greedy', 'fireants_syn', 'convexadam_v4seg') and set(xc.LPBA_ARMS) <= set(xc.ARMS)
    assert xc.P19_STOP_AT == 3 and xc.p19_stop_at('holdout') == xc.p19_stop_at('dev') == 3 and xc.p19_stop_at(FORMAL) == xc.p19_stop_at(FEAS) == xc.P19_STOP_AT_LPBA == 2
    assert xc.FROZEN_CONFIG_SHA256 == L.lc.FROZEN_FILES[xc.FROZEN_CONFIG_REL] == '7c4965ebed65cfe55009518abca72c021309c366f96b8452b9c622a5f53b15a2'
    assert xb_drive.PAIR_SETS == ('holdout', 'dev', FORMAL, FEAS) and xc.authorizations_problem('holdout') is None and xc.authorizations_problem('dev') is None


def test_the_hold_outs_p19_is_read_at_call_time(monkeypatch):
    monkeypatch.setattr(xc, 'P19_STOP_AT', 5)
    assert xc.p19_stop_at('holdout') == 5 and xc.p19_stop_at(FORMAL) == 2


def test_the_places_names_and_plan_of_an_lpba_mode(world):
    record, work, store = xc.roots(FORMAL)
    assert record == work == L.lc.roots(FORMAL).root and store is None and xc.pair_path(FORMAL, PAIR) == L.lc.roots(FORMAL).stage/PAIR/'pair.npz'
    paths = xc.cell_paths('ants_s', FORMAL, PAIR)
    assert paths['store'] is None and paths['work'] == record/'work'/'ants_s'/PAIR and paths['record'] == record/'ants_s'/PAIR and paths['lock'] == record/'locks'/f'xb__ants_s__{PAIR}.lock'
    assert xc.hi_pred_root(FEAS) == L.lc.roots(FEAS).pred_root and xc.subjects_of(FORMAL, PAIR) == ('S05', 'S06') and xc.subjects_of('holdout', 'a__b') == ('a', 'b')
    assert xc.unit_name('ants_s', PAIR, FORMAL) == 'tag-lp-x-ants_s-S05-S06' and xc.unit_name('fireants_syn', FEAS_PAIR, FEAS) == 'tag-lp-feas-x-fireants_syn-S01-S02'
    assert xc.chain_unit(FORMAL) == 'tag-lp-x-chain' and xc.chain_unit(FEAS) == 'tag-lp-feas-x-chain' and xc.chain_unit() == xc.CHAIN_UNIT == 'tag-hb-x-chain'
    argv = xb_chain.unit_argv('ants_s', FORMAL, PAIR)
    assert '--unit=tag-lp-x-ants_s-S05-S06' in argv and argv[argv.index('--pair-set')+1] == FORMAL and not [a for a in argv if 'tag-hb' in a or 'seq-vp' in a]
    assert xb_chain.chain_launch_string(FORMAL, ['ants_s'])[2] == '--unit=tag-lp-x-chain'
    assert xb_chain.cells(FORMAL, ['ants_s', 'fireants_greedy']) == [('ants_s', PAIR), ('fireants_greedy', PAIR)]
    with pytest.raises(xb_chain.Stop, match='not LPBA40 arms'):
        xb_chain.cells(FORMAL, ['niftyreg'])
    assert xb_drive.planned(FORMAL, 'S05', 'S06') and not xb_drive.planned(FORMAL, 'S06', 'S05') and not xb_drive.planned(FEAS, 'S05', 'S06')


def test_a_feasibility_cell_runs_and_its_field_stays_in_the_work_cell(world):
    L.S._reset_cache()
    assert W.run_cell('fireants_greedy', FEAS, FEAS_PAIR, deps=deps()) == 0
    doc = result('fireants_greedy', FEAS, FEAS_PAIR)
    work = xc.cell_paths('fireants_greedy', FEAS, FEAS_PAIR)['work']
    assert doc['status'] == 'ok' and doc['disp']['file'] == str(work/'disp.npz') and doc['disp']['sha256'] == xc.sha256(work/'disp.npz') and doc['move'] is None
    assert doc['store'] == dict(kind='home', work=str(work), d=None) and (doc['fixed_subject'], doc['moving_subject']) == ('S01', 'S02') and doc['pair_set'] == FEAS
    assert doc['checks']['admission']['admission']['subjects']['S01']['admitted_by_lpba40'] is False and doc['checks']['pins']['prereg']['sha256'] == L.lc.PREREG_SHA256
    assert doc['checks']['pair_input']['guard'].get('lpba40_guard') is None and doc['lock']['line']['pair_set'] == FEAS and doc['spy_passed']
    assert doc['preregistration'] == L.lc.PREREG_REL and doc['lock']['line']['code_commit'] == 'c'*40
    field = next(a for a in json.loads((L.LE.ROOT/'configs/frozen/lpba40_test_v1_arm_table.json').read_text())['arms'] if a['name'] == 'fireants_greedy')['field']
    assert field['file_key'] == 'disp.file' and Path(doc['disp']['file']).is_relative_to(Path(field['root'].replace('{arm}', 'fireants_greedy').replace('/home/rmcfz/seq_vp_tag_results', str(L.lc.RESULTS)).replace(
        'lpba40_test_v1', 'lpba40_feas_v1')))


def test_p19_stops_an_lpba_arm_at_its_second_suspicious_pair(world):
    for k, pair in enumerate((FEAS_PAIR, REVERSE)):
        L.S._reset_cache()
        assert W.run_cell('fireants_greedy', FEAS, pair, deps=deps(step=W.FakeMethod(own_scale=1.))) == 0
        assert result('fireants_greedy', FEAS, pair)['export_check']['suspicious'] is True
        stop = xc.cell_paths('fireants_greedy', FEAS, pair)['arm_stop']
        assert stop.exists() == (k == 1)
    doc = json.loads(stop.read_text())
    assert doc['stop_at'] == 2 and doc['suspicious_pairs'] == [FEAS_PAIR, REVERSE] and 'second pair' in doc['rule'] and 'LPBA40 prereg §6.6' in doc['rule']


def test_only_the_four_arms_and_the_modes_table(world):
    L.S._reset_cache()
    assert W.run_cell('niftyreg', FEAS, FEAS_PAIR, deps=deps()) == xc.EXIT['refused']
    L.S._reset_cache()
    assert W.run_cell('ants_s', FEAS, PAIR, deps=deps()) == xc.EXIT['admission_rejected']       # a formal pair in the feasibility mode
    stages = [json.loads(p.read_text())['stage'] for p in sorted(W.refusals(FEAS))]
    assert stages == ['admission', 'admission'] and not xc.cell_paths('ants_s', FEAS, PAIR)['lock'].exists()


def test_the_formal_mode_refuses_without_a_committed_p_l_and_only_it_has_authorizations(world):     # integration review minor 8: the name of the LP-C2 behaviour
    L.S._reset_cache()
    assert W.run_cell('ants_s', FORMAL, PAIR, deps=deps()) == xc.EXIT['refused']
    doc = json.loads(sorted(W.refusals(FORMAL))[-1].read_text())
    # LP-C2 integration: rule one is implemented now (lpba_arms.pinned_code_problems -> lpba_common, LPBA40 constants); the expected change of decision_log line 584 ruling 3:
    # the refusal is rule one's own (no committed runs/lpba40_test_v1/pinned_code.json here), and the formal mode's authorizations are known (AUTH_DIRS 'lpba40_test', D148).
    assert doc['stage'] == 'code' and 'runs/lpba40_test_v1/pinned_code.json' in doc['reason'] and not xc.cell_paths('ants_s', FORMAL, PAIR)['lock'].exists()
    assert xc.authorizations_problem(FORMAL) is None and xc.authorizations_problem(FEAS)
    with pytest.raises(hc.Refused, match='authorizations'):
        xb_drive.authorization_draft('resume', 'ants_s', FEAS, FEAS_PAIR, written_by="lead on BZ's behalf", words='x', time_local='x', reason='x')
    assert xb_chain.no_resume('ants_s', FORMAL, PAIR) is None and xb_chain.declined_refusal('ants_s', FORMAL, PAIR) is None


def test_convexadam_v4seg_waits_for_the_methods_finalized_record(world):
    L.S._reset_cache()
    assert W.run_cell('convexadam_v4seg', FEAS, FEAS_PAIR, deps=deps()) == xc.EXIT['refused']
    doc = json.loads(sorted(W.refusals(FEAS))[-1].read_text())
    assert doc['stage'] == 'method_run' and 'step 5 not done' in doc['reason']


def test_i1_the_development_arm_stop_keeps_its_keys(monkeypatch):
    """Review I1: arm_stopped.json of the hold-out and the development has its keys (no stop_at); only an LPBA arm's carries it."""
    W.fake_checks(monkeypatch)
    W.write_stage_pair(hc.DEV_STAGE, W.DEV_PAIR, seg=True)
    monkeypatch.setattr(xc, 'P19_STOP_AT', 1)                                              # read at call time: the first suspicious pair stops the development arm here
    assert W.run_cell('fireants_greedy', 'dev', W.DEV_PAIR, deps=W.deps(step=W.FakeMethod(own_scale=1.))) == 0
    stop = json.loads(xc.cell_paths('fireants_greedy', 'dev', W.DEV_PAIR)['arm_stop'].read_text())
    assert set(stop) == {'schema', 'arm', 'pair_set', 'utc', 'suspicious_pairs', 'at_pair', 'rule'} and 'third pair' in stop['rule']


def test_minor3_an_lpba_cell_with_a_spy_breach_takes_no_rerun_or_resume(world):
    """Diff review minor 3: an LPBA external cell whose driver's spy stopped a read (a refusal of stage spy_breach, a record whose spy did not pass) is refused a rerun and a resume."""
    import xb_drive
    for pair_set, pair in ((FEAS, FEAS_PAIR),):
        paths = xc.cell_paths('fireants_greedy', pair_set, pair)
        root = xc.roots(pair_set)[0]
        assert xb_drive.lpba_spy_breach_problem('fireants_greedy', pair_set, pair, paths, root) is None
        paths['result'].parent.mkdir(parents=True, exist_ok=True)
        paths['result'].write_text(json.dumps(dict(status='infra_failure', spy_passed=False, failure=dict(kind='spy_not_passed'))))
        assert 'spy breach is not rerun' in xb_drive.lpba_spy_breach_problem('fireants_greedy', pair_set, pair, paths, root)
    dev_paths = xc.cell_paths('fireants_greedy', 'dev', W.DEV_PAIR)
    assert xb_drive.lpba_spy_breach_problem('fireants_greedy', 'dev', W.DEV_PAIR, dev_paths, xc.roots('dev')[0]) is None


# ---- LP-C2 integration item 7 (LP-B quick diff review minor 1; decision_log 2026-11-19, the LP-B merge ~4:22 PM CST) ----------------------------------------------------------------
def test_integration7_the_lpba_driver_policy_forbids_archive_and_reviews(world):
    """Minor 1: XB's driver policy in an LPBA mode lists the code tree's and the main checkout's archive/ and docs/bayesreg/reviews/ as forbidden roots (xb_policy rule 4; below the code
    root, so rule 3a does not read them as code), as ArmPolicy and the review's lpba_policy do; the development mode's policy is unchanged."""
    import xb_policy
    for pair_set, pair in ((FORMAL, PAIR), (FEAS, FEAS_PAIR)):
        paths = xc.cell_paths('ants_s', pair_set, pair)
        doc = xb_drive.policies('ants_s', pair_set, pair, paths, xc.pair_path(pair_set, pair))
        want = [str(Path(t)/rel) for t in dict.fromkeys((Path(xc.REPO), Path(hc.MAIN_REPO))) for rel in ('archive', 'docs/bayesreg/reviews')]
        assert doc['forbidden_roots'][-len(want):] == want
        policy = xb_policy.Policy(doc)
        for path in (Path(xc.REPO)/'archive'/'x.py', Path(xc.REPO)/'docs'/'bayesreg'/'reviews'/'x.md', Path(hc.MAIN_REPO)/'archive'/'x.py'):
            assert policy.classify(str(path))[0] == 'forbidden', (pair_set, path)
        assert policy.classify(str(Path(xc.REPO)/'experiments'/'xb'/'xb_drive.py')) == ('allowed', 'a read of the running code tree')
    dev = xb_drive.policies('ants_s', 'dev', W.DEV_PAIR, xc.cell_paths('ants_s', 'dev', W.DEV_PAIR), xc.pair_path('dev', W.DEV_PAIR))
    assert not [r for r in dev['forbidden_roots'] if Path(r).name in ('archive', 'reviews')]
