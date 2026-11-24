"""LP-G (decision_log 2026-11-20, the feasibility stop's diagnosis and ruling; C23.6-5, C22.H4-3): the guard's four sealed lists, three of which lie under the code tree's
archive/seq_vp_3d/results/verification/3d_stage_HOLDOUT_v1/plan/, are read one by one by every LPBA40 process that admits a pair -- the method's supervisor, t88, repair and third
certification and configuration C (lpba_arms.ArmPolicy's read_exact) and the review unit (hb_review.lpba_policy's deny_except) -- while the rest of plan/, a listing of plan/ and the rest of
archive/ stay forbidden.  The world is LP-B's (tests/test_bayesreg/hb/_lpba_world.py) with the synthetic lists moved to their production paths (_lpba_world.production_lists: the module
constant's paths, synthetic contents; no real list is opened).  Before the fix these chains stopped where the feasibility chain did: the supervisor's spy at H1's list.
No real subject's file, no unit."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from . import _lpba_world as L
from . import _world as W

R, he, hc, hb, review, hm, hold, lc, G, S = L.R, L.he, L.hc, L.hb, L.review, L.hm, L.hold, L.lc, L.G, L.S
FORMAL, FEAS, PAIR, FEAS_PAIR, PLAN_REL = L.FORMAL, L.FEAS, L.PAIR, L.FEAS_PAIR, L.PLAN_REL
import firewall_c as fc  # noqa: E402
import hb_configc as C  # noqa: E402
from .test_hb_configc import Launcher, fake_prepare, kernel, probe_of  # noqa: E402

la = hold._lpba()
STAGES = ('supervisor', 't88', 'repair', 'third')
DEVCHECK = Path(__file__).resolve().parents[3]/'experiments'/'lpba'/'devcheck'             # lpdev_a, lpdev_b, lpdev_c (as tests/test_bayesreg/lpba/test_lpba_devcheck.py)
if str(DEVCHECK) not in sys.path:
    sys.path.insert(0, str(DEVCHECK))


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(R.core, 'frozen_api', lambda: SimpleNamespace(PreoptConfig=lambda **k: dict(k), four_level_grids=None, downsample_array=None))     # no frozen tree here
    monkeypatch.setattr(R, 'enable_jit', lambda ctx: False)
    w = L.make(tmp_path, monkeypatch)
    w.prod = L.production_lists(w, monkeypatch)                                         # before any entry runs: LP-A's entries admit with the lists at their production paths too
    return w


@pytest.fixture
def staged(world, monkeypatch):
    L.prepare(world, monkeypatch, FORMAL, hi=False, atropos=False)
    return world


@pytest.fixture
def both(staged, monkeypatch):
    L.prepare(staged, monkeypatch, FEAS, hi=False, atropos=False)
    return staged


@pytest.fixture
def linked(world, tmp_path):
    """A link to the synthetic code tree: a path to a list written through it (the as-given form differs from the resolved one)."""
    link = tmp_path/'linked_tree'
    link.symlink_to(world.repo, target_is_directory=True)
    return link


def sealed():
    """The files the guard reads (holdout_common.sealed_list_paths: resolved)."""
    return [Path(p) for p in hc.sealed_list_paths()]


def forms(world, link):
    """Each sealed list as the guard opens it (<sealed.REPO>/<path>), resolved, and through the link."""
    return [p for spec in S.SEALED_LISTS for p in (Path(S.REPO)/spec['path'], (Path(S.REPO)/spec['path']).resolve(), link/spec['path'])]


def others(world, link):
    """What stays forbidden: plan/'s other file (also through the link), a name that only begins like a list's, a .py under plan/, a file of archive/ outside plan/."""
    h1 = next(spec['path'] for spec in S.SEALED_LISTS if spec['name'] == 'H1')
    return [world.prod.other, link/PLAN_REL/world.prod.other.name, world.repo/(h1+'.bak'), world.repo/PLAN_REL/'x.py', world.prod.stray]


def h1_path():
    return str((Path(S.REPO)/next(spec['path'] for spec in S.SEALED_LISTS if spec['name'] == 'H1')).resolve())


def arm_policies():
    """(name, ArmPolicy) of every user: the method's four processes and the configuration C driver, both modes."""
    out = []
    for pair_set, pair in ((FORMAL, PAIR), (FEAS, FEAS_PAIR)):
        ctx = L.ctx_of(pair_set, pair)
        out += [(f'{pair_set} {stage}', R.lpba_policy_of(ctx, stage)) for stage in STAGES]
        paths = C.cell_paths(pair_set, pair)
        out.append((f'{pair_set} configc', C.lpba_policy_of(pair_set, pair, paths, paths['batches']/'b')))
    return out


def list_reads(events):
    """The resolved paths of the reads an ArmSpy listed as inputs."""
    return {e['resolved'] for e in events if not e['write'] and e['why'] == 'input'}


def test_the_world_puts_three_lists_under_the_code_trees_archive(world):
    lists = sealed()
    assert len(lists) == 4 and sum(p.is_relative_to((world.repo/'archive').resolve()) for p in lists) == 3
    assert set(lists) == {(Path(S.REPO)/spec['path']).resolve() for spec in S.SEALED_LISTS} and all(p.is_file() for p in lists)
    assert {p.parent for p in lists if p.is_relative_to((world.repo/'archive').resolve())} == {(world.repo/PLAN_REL).resolve()}


# ---- (i) ArmPolicy: the four lists read one by one, nothing else of plan/ or archive/ -------------------------------------------------------------------
def test_lpg_i_every_arm_policy_reads_the_four_lists_and_nothing_else_of_archive(both, linked):
    lists = sealed()
    for name, policy in arm_policies():
        doc = policy.document()
        assert {str(p) for p in lists} <= set(doc['read_exact']) and doc['sealed_lists'] == sorted(map(str, lists)), name
        for path in forms(both, linked):
            assert policy.classify(path, False) == ('allowed', 'input'), (name, path)
            assert policy.classify(path, True)[0] == 'forbidden', (name, path)
        for path in others(both, linked):
            assert policy.classify(path, False)[0] == 'forbidden', (name, path)
        for path in (both.prod.plan, linked/PLAN_REL, both.prod.archive):
            assert policy.classify_listing(path)[0] == 'forbidden', (name, path)


def test_lpg_i_the_lists_come_from_the_guards_paths_at_construction(staged, monkeypatch):
    """The read_exact entries are holdout_common.sealed_list_paths' (hp_core.allowed_list_paths), read when the policy is made: no other file of plan/ enters with them."""
    ctx = L.ctx_of()
    policy = R.lpba_policy_of(ctx, 'supervisor')
    assert policy.read_exact == {Path(p).resolve() for p in hc.sealed_list_paths()} | {(Path(ctx['stage_dir'])/PAIR/n).resolve() for n in ('pair.npz', 'manifest.json')}
    monkeypatch.setattr(lc.hp, 'allowed_list_paths', lambda: [])                       # the fix taken out
    bare = R.lpba_policy_of(ctx, 'supervisor')
    assert bare.sealed_lists == set() and bare.classify(Path(h1_path()), False)[0] == 'forbidden'


# ---- (ii) the read chains under the spy, in the production layout -----------------------------------------------------------------------------------
def t88_chain(world, monkeypatch):
    """test_hb_lpba.test_i3a_the_t88_read_chain_runs_under_the_spy_without_a_breach's chain (LP-A's stage, Atropos and HI; stopped where the T88 arithmetic starts).  Returns (ctx, rc, run, reached)."""
    import bayesreg.seg.structure_layer as sl
    registry = L.prepare(world, monkeypatch, FORMAL)
    reg_repo = L.git_repo(world.tmp/'registry_tree')
    registry_path = reg_repo/'segmenters_test.json'
    registry_path.write_bytes(Path(registry.path).read_bytes())
    L.commit(reg_repo, registry_path.name)
    monkeypatch.setattr(sl, 'FORMAL_REGISTRIES', sl.FORMAL_REGISTRIES+(registry_path,))
    ctx = L.ctx_of()
    ctx['paths'] = dict(ctx['paths'], registry=str(registry_path), models_root=str(world.tmp/'members'))
    monkeypatch.setattr(he, 'segmenter_start_problems', lambda s, registry=None: ([], dict(test='the synthetic registry is not the contract anchor')))
    monkeypatch.setattr(he, 'channel_source_problems', lambda sources: [])
    mi = R.core.make_inputs()
    real_qc = mi.alignment_qc
    monkeypatch.setattr(mi, 'alignment_qc', lambda u_wm, a_wm, **k: dict(real_qc(u_wm, a_wm, **k), passed=True, test='the synthetic images: the QC computed, passed by the test'))
    Path(ctx['out']).mkdir(parents=True)
    reached = []

    def stop(*args, **kwargs):
        reached.append(True)
        raise SystemExit('test: stopped where the T88 arithmetic starts (after the channels and the colour table)')
    import fl_channels
    monkeypatch.setattr(fl_channels, 'prepare_stacks', stop)
    S._reset_cache()
    rc = R.t88_main(ctx)
    return ctx, rc, he.read_json(Path(ctx['out'])/'t88'/'run.json'), reached


def test_lpg_ii_the_t88_read_chain_reads_the_four_lists_without_a_breach(world, monkeypatch):
    ctx, rc, run, reached = t88_chain(world, monkeypatch)
    out = Path(ctx['out'])
    opened = he.read_json(out/'t88'/'opened_files_t88_0.json')
    assert rc == 0 and opened['passed'] is True and opened['breaches'] == [] and not list(out.glob('**/spy_breach*.json')), (run.get('reason'), opened['breaches'])
    assert reached == [True] and 'after the channels' in run['reason']
    assert {str(p) for p in sealed()} <= list_reads(opened['events'])
    assert all(e['cls'] == 'allowed' for e in opened['events'])
    assert not [e for e in opened['events'] if e['resolved'].startswith(str(world.prod.archive.resolve())+'/') and e['resolved'] not in {str(p) for p in sealed()}]


class StopChild:
    """The supervisor's child (run_hx.SubprocessChild's place): t88 reports a refusal, so the supervisor writes line.json (rejected) and stops right after its own admission."""
    kind = 'test'

    def __init__(self):
        self.calls = []

    def __call__(self, kind, ctx):
        self.calls.append(kind)
        d = Path(ctx['out'])/kind
        d.mkdir(parents=True, exist_ok=True)
        he.write_json(d/'run.json', dict(status=he.T88_REJECTED, reason='test (LP-G): the supervisor stops after its admission'))
        return 0


def supervisor(monkeypatch, pair_set=FORMAL, pair=PAIR):
    monkeypatch.setattr(he, 'segmenter_start_problems', lambda s, registry=None: ([], dict(test=True)))
    ctx = L.ctx_of(pair_set, pair)
    child = StopChild()
    S._reset_cache()
    return ctx, child, R.run_pair(ctx, child=child, probe=W.probe)


@pytest.mark.parametrize('pair_set, pair', [(FORMAL, PAIR), (FEAS, FEAS_PAIR)])
def test_lpg_ii_the_supervisors_admission_reads_the_four_lists(both, monkeypatch, pair_set, pair):
    ctx, child, rc = supervisor(monkeypatch, pair_set, pair)
    out = Path(ctx['out'])
    opened = he.read_json(out/'opened_files_supervisor_0.json')
    assert rc == 0 and child.calls == ['t88'] and opened['passed'] is True and not list(out.glob('**/spy_breach*.json')), opened['breaches']
    assert (out/'admission.json').is_file() and he.read_json(out/'line.json')['line_status'] == he.REJECTED
    assert {str(p) for p in sealed()} <= list_reads(opened['events']) and opened['policy']['sealed_lists'] == sorted(map(str, sealed()))


class FallbackChild(StopChild):
    """t88 reports no field (a failure of the entry): the supervisor goes on to configuration C's fallback in its own process (run_hx.configc_fallback)."""

    def __call__(self, kind, ctx):
        self.calls.append(kind)
        d = Path(ctx['out'])/kind
        d.mkdir(parents=True, exist_ok=True)
        he.write_json(d/'run.json', dict(status='estimate_failed', reason='test (LP-G): no field, configuration C follows', oracle=None, field=None))
        return 0


def test_lpg_ii_the_supervisors_configuration_c_fallback_reads_nothing_else_of_archive(staged, monkeypatch):
    """The supervisor's in-process part of configuration C's fallback under its spy (derive_pair, the firewall gate, the candidate certified again; the frozen run is Launcher's stand-in and
    firewall_c.prepare test_hb_configc's, as in the configuration C arm's tests): no breach, the four lists read, nothing else of archive/."""
    monkeypatch.setattr(fc, 'prepare', fake_prepare)
    monkeypatch.setattr(he, 'segmenter_start_problems', lambda s, registry=None: ([], dict(test=True)))
    ctx = L.ctx_of()
    S._reset_cache()
    pin, _ = R.admitted_images(ctx)
    monkeypatch.setattr(W, 'grid', lambda shape=None: pin.grid)
    child = FallbackChild()
    S._reset_cache()
    rc = R.run_pair(ctx, child=child, launcher=Launcher('ok'), certification=W.Certify().pair(), probe=W.probe)
    out = Path(ctx['out'])
    opened = he.read_json(out/'opened_files_supervisor_0.json')
    line = he.read_json(out/'output'/'line.json')
    assert rc == 0 and child.calls == ['t88'] and opened['passed'] is True and line['line_status'] == he.FALLBACK_C, (opened['breaches'], line.get('fallback'))
    assert {str(p) for p in sealed()} <= list_reads(opened['events'])
    archive = str(staged.prod.archive.resolve())+'/'
    assert {e['resolved'] for e in opened['events'] if e['resolved'].startswith(archive)} <= {str(p) for p in sealed()}


def test_lpg_ii_without_the_fix_the_supervisor_stops_at_h1(staged, monkeypatch):
    """The feasibility stop reproduced (stage supervisor, the H1 list): the read_exact union taken out (hp_core.allowed_list_paths answering nothing to ArmPolicy)."""
    monkeypatch.setattr(lc.hp, 'allowed_list_paths', lambda: [])
    ctx, child, rc = supervisor(monkeypatch)
    out = Path(ctx['out'])
    breach = he.read_json(out/'spy_breach_0.json')
    assert rc == he.STOP_QUEUE_EXIT and child.calls == [] and breach['stage'] == 'supervisor'
    assert breach['breaches'][0]['resolved'] == h1_path() and breach['breaches'][0]['cls'] == 'forbidden'
    line = he.read_json(out/'line.json')
    assert line['line_status'] == he.IMPLEMENTATION and line['implementation_error']['spy_breach']['stage'] == 'supervisor'


def test_lpg_ii_the_repair_and_the_third_certification_read_the_four_lists(staged):
    ctx = L.ctx_of()
    out = Path(ctx['out'])
    out.mkdir(parents=True)
    S._reset_cache()
    assert R.repair_main(ctx) == 0
    rec = he.read_json(out/'repair'/'repair.json')
    opened = he.read_json(out/'repair'/'opened_files_repair_0.json')
    assert rec['status'] == 'error' and rec['admission_second_lock']['rule'].startswith('he_holdout.lpba_admitted_images')     # admitted; no T88 field here (as test_hb_lpba)
    assert opened['passed'] is True and {str(p) for p in sealed()} <= list_reads(opened['events'])
    (out/'repair').rename(out/'repair_first')
    S._reset_cache()
    pin, _ = R.admitted_images(ctx)
    L.write_cell(ctx, pin)
    S._reset_cache()
    assert R.third_cert_main(ctx, certification=W.Certify().pair()) == 0
    third = he.read_json(out/'output'/'third_certificate.json')
    opened = he.read_json(out/'opened_files_third_0.json')
    assert third['passed'] is True and opened['passed'] is True and {str(p) for p in sealed()} <= list_reads(opened['events'])


def test_lpg_ii_the_third_certifications_fallback_reads_nothing_else_of_archive(staged, monkeypatch):
    """A third certificate that fails: configuration C's fallback in the third certification's own process (in-process part; the frozen run and firewall_c.prepare the configuration C tests'
    stand-ins) under its spy: no breach, the four lists read, nothing else of archive/; the new output is configuration C's."""
    monkeypatch.setattr(fc, 'prepare', fake_prepare)
    ctx = L.ctx_of()
    out = Path(ctx['out'])
    out.mkdir(parents=True)
    S._reset_cache()
    pin, _ = R.admitted_images(ctx)
    monkeypatch.setattr(W, 'grid', lambda shape=None: pin.grid)
    u, _ = L.write_cell(ctx, pin)
    S._reset_cache()
    assert R.third_cert_main(ctx, launcher=Launcher('ok'), certification=W.Certify(fail={he.array_sha256(u)}).pair()) == 0
    opened = he.read_json(out/'opened_files_third_0.json')
    line = he.read_json(out/'output'/'line.json')
    assert opened['passed'] is True and line['line_status'] == he.FALLBACK_C and (out/'output_superseded_0'/'third_certificate.json').is_file(), (opened['breaches'], line.get('fallback'))
    assert {str(p) for p in sealed()} <= list_reads(opened['events'])
    archive = str(staged.prod.archive.resolve())+'/'
    assert {e['resolved'] for e in opened['events'] if e['resolved'].startswith(archive)} <= {str(p) for p in sealed()}


@pytest.mark.parametrize('stage', ['t88', 'repair', 'third'])
def test_lpg_ii_without_the_fix_the_children_stop_at_h1(staged, monkeypatch, stage):
    import fast_channels
    monkeypatch.setattr(fast_channels, 'read_predicted', lambda pin, **kwargs: (_ for _ in ()).throw(AssertionError('not reached: the admission comes first')))
    monkeypatch.setattr(he, 'segmenter_start_problems', lambda s, registry=None: ([], dict(test=True)))
    ctx = L.ctx_of()
    out = Path(ctx['out'])
    out.mkdir(parents=True)
    if stage == 'third':
        S._reset_cache()
        pin, _ = R.admitted_images(ctx)
        L.write_cell(ctx, pin)
    monkeypatch.setattr(lc.hp, 'allowed_list_paths', lambda: [])
    S._reset_cache()
    main = dict(t88=lambda: R.t88_main(ctx), repair=lambda: R.repair_main(ctx), third=lambda: R.third_cert_main(ctx, certification=W.Certify().pair()))[stage]
    assert main() == la.SPY_BREACH_EXIT
    record_dir = out if stage == 'third' else out/stage
    breach = he.read_json(record_dir/'spy_breach_0.json')
    assert breach['stage'] == stage and breach['breaches'][0]['resolved'] == h1_path()


@pytest.fixture
def configc(staged, monkeypatch):
    monkeypatch.setattr(fc, 'prepare', fake_prepare)
    monkeypatch.setattr(he, 'frozen_identity', lambda *a, **k: (dict(test=True), []))
    monkeypatch.setattr(la, 'code_record', lambda pair_set, pinned=None, **k: (dict(commit='c'*40, clean=True, pair_set=pair_set), []))
    S._reset_cache()
    pin, _ = R.admitted_images(L.ctx_of())
    monkeypatch.setattr(W, 'grid', lambda shape=None: pin.grid)
    batch = C.roots(FORMAL)[0]/'batches'/'test'
    batch.mkdir(parents=True)
    return batch


def test_lpg_ii_the_configuration_c_driver_reads_the_four_lists(configc):
    S._reset_cache()
    rc = C.run(FORMAL, PAIR, batch=configc, launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(), kernel=kernel())
    paths = C.cell_paths(FORMAL, PAIR)
    r = he.read_json(paths['result'])
    assert rc == 0 and r['status'] == 'ok', (r.get('error'), r.get('traceback'))
    opened = he.read_json(sorted(paths['record'].glob('opened_files*.json'))[0])
    assert opened['passed'] is True and {str(p) for p in sealed()} <= list_reads(opened['events']) and not sorted(paths['record'].glob('spy_breach*.json'))


def test_lpg_ii_without_the_fix_the_configuration_c_driver_is_refused_at_h1(configc, monkeypatch):
    monkeypatch.setattr(lc.hp, 'allowed_list_paths', lambda: [])
    S._reset_cache()
    assert C.run(FORMAL, PAIR, batch=configc, launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(), kernel=kernel()) == C.EXIT['refused']
    paths = C.cell_paths(FORMAL, PAIR)
    breach = he.read_json(sorted(paths['record'].glob('spy_breach*.json'))[0])
    assert breach['breaches'][0]['resolved'] == h1_path()
    assert [json.loads(p.read_text())['stage'] for p in sorted((configc/'refusals').glob('*.json'))] == ['spy']


# ---- (iii) the review unit -------------------------------------------------------------------------------------------------------------------------
def code_tree_review_mode(monkeypatch, world, pair_set=FORMAL):
    """The review's mode made where production makes it: hb.REPO the code tree that holds the lists (sealed.REPO; one checkout, <formal root>/code).  The tests' hb.REPO is the worktree."""
    with monkeypatch.context() as m:
        m.setattr(hb, 'REPO', world.repo)
        return review.lpba_mode(pair_set, None)


def test_lpg_iii_the_review_policy_lets_the_four_lists_through_and_nothing_else_of_archive(both, monkeypatch, linked, tmp_path):
    lists = sealed()
    for pair_set in (FORMAL, FEAS):
        policy = code_tree_review_mode(monkeypatch, both, pair_set).policy
        for path in forms(both, linked):
            assert policy.why(path) is None, (pair_set, path)
        for path in others(both, linked)+[both.prod.plan]:
            assert policy.why(path) and 'forbidden root' in policy.why(path), (pair_set, path)
        archive = set(review._paths([both.repo/'archive']))
        exceptions = [(d, s) for d, s in policy.deny_except if d in archive]
        assert {s for _, s in exceptions} == {str(p) for p in lists if p.is_relative_to((both.repo/'archive').resolve())} | {
            os.path.abspath(Path(S.REPO)/spec['path']) for spec in S.SEALED_LISTS if spec['path'].startswith('archive/')}
        assert all(Path(s).is_file() for _, s in exceptions)                            # files only: no directory of archive/ is exempt
    with monkeypatch.context() as m:                                                   # the tests' own layout (hb.REPO the worktree): no list under its archive/, nothing added
        m.setattr(hb, 'REPO', tmp_path/'another_tree')
        assert review.sealed_list_exceptions(Path(hb.REPO)/'archive') == () and review.lpba_policy(FORMAL).deny_except == ()
    assert review.holdout_policy().deny_except == () and review.dev_policy(tmp_path/'he_smoke_x'/'seg', None).deny_except == ()


def review_world(staged):
    ctx = L.ctx_of()
    S._reset_cache()
    pin, _ = R.admitted_images(ctx)
    L.write_cell(ctx, pin)
    rec = hb.record_dir('method', PAIR, FORMAL)
    rec.mkdir(parents=True)
    he.write_json(rec/'step4.json', dict(status=he.HX, went_on=True, written_by='test'))
    return ctx


def write_manifest(mode):
    mode.manifest().parent.mkdir(parents=True, exist_ok=True)
    mode.manifest().write_text(json.dumps(dict(schema='test', pairs=review.input_manifest([PAIR], FORMAL))))


def test_lpg_iii_the_review_unit_admits_and_finalizes_reading_the_four_lists(staged, monkeypatch):
    ctx = review_world(staged)
    mode = code_tree_review_mode(monkeypatch, staged)
    write_manifest(mode)
    S._reset_cache()
    assert review.review_pair(ctx, mode, batch='test-batch', certification=W.Certify().pair(), probe=W.probe) == 0
    opened = he.read_json(Path(ctx['out'])/'review'/review.OPENED)
    assert opened['passed'] is True and opened['sealed_lists_opened'] == sorted(map(str, sealed())) and opened['n_sealed_list_opens'] >= 4
    assert [d for d, _ in opened['policy']['deny_except']] and he.read_json(Path(ctx['out'])/'review'/review.REVIEW_RECORD)['finalized'] is True


def test_lpg_iii_without_the_fix_the_review_unit_stops_at_h1(staged, monkeypatch):
    ctx = review_world(staged)
    monkeypatch.setattr(review, 'sealed_list_exceptions', lambda root: ())          # the fix taken out
    mode = code_tree_review_mode(monkeypatch, staged)
    write_manifest(mode)
    S._reset_cache()
    assert review.review_pair(ctx, mode, batch='test-batch', certification=W.Certify().pair(), probe=W.probe) == review.STOP_EXIT['spy']
    stop = he.read_json(Path(ctx['out'])/'review'/review.NOT_FINALIZED)
    assert stop['kind'] == 'spy' and h1_path() in stop['reason']


def test_lpg_iii_the_reviews_third_certification_reads_the_four_lists_under_both_spies(staged, monkeypatch):
    """The third certification in the review unit's child: ReviewSpy (lpba_policy) and, inside run_hx.third_cert_main, the ArmSpy of stage third -- both let the four lists through."""
    ctx = review_world(staged)
    mode = code_tree_review_mode(monkeypatch, staged)
    S._reset_cache()
    assert review.third_child_main(ctx, mode, certification=W.Certify().pair()) == 0
    rev = Path(ctx['out'])/'review'
    outer = he.read_json(rev/'opened_files_third.json')                                # hb_core.numbered (the review's records)
    inner = he.read_json(Path(ctx['out'])/'opened_files_third_0.json')                  # run_hx.numbered (lpba_spied's)
    assert outer['passed'] is True and outer['sealed_lists_opened'] == sorted(map(str, sealed())) and inner['passed'] is True
    assert {str(p) for p in sealed()} <= list_reads(inner['events']) and he.read_json(Path(ctx['out'])/'output'/'third_certificate.json')['passed'] is True


# ---- LP-G revision 1: the review unit's spy refuses a write to a sealed list (the LP-G review's minor 1) ----------------------------------------------------
def test_lpg_r1_the_review_spy_refuses_a_write_to_a_sealed_list(staged, monkeypatch):
    """lpba_policy's read_only: the four sealed lists may be read, never opened for writing (open, io.open, Path.open, Path.write_text); the files stay as they were.  The hold-out's and the
    development's policies have no read_only: their why(path, write=True) is why(path)."""
    import dataclasses
    policy = code_tree_review_mode(monkeypatch, staged).policy
    lists = sealed()
    for spec in S.SEALED_LISTS:
        for path in (Path(S.REPO)/spec['path'], (Path(S.REPO)/spec['path']).resolve()):
            assert policy.why(path) is None and 'read only' in policy.why(path, write=True), path
    assert dataclasses.replace(policy, read_only=()).why(lists[0], write=True) is None             # the write rule is read_only's alone (taken out: the exception lets the write through)
    ctx = L.ctx_of()
    before = {p: hb.sha(p) for p in lists}
    spy = review.ReviewSpy(R.pair_path(ctx), policy).install()
    refused = 0
    try:
        for path in lists:
            assert path.read_text(encoding='utf-8') and open(path, encoding='utf-8').read()     # noqa: SIM115  (reads pass)
            for attempt in (lambda: open(path, 'a'), lambda: open(path, mode='r+'), lambda: path.open('a'), lambda: path.write_text('x')):     # noqa: SIM115
                with pytest.raises(review.spying.Breach, match='read only'):
                    attempt()
                refused += 1
    finally:
        spy.remove()
    assert {p: hb.sha(p) for p in lists} == before and len(spy.breaches) == refused == 4*len(lists)
    for other_policy in (review.holdout_policy(), review.dev_policy(Path(hc.RESULTS)/'he_smoke_x'/'seg', None)):
        assert other_policy.read_only == ()
        for path in [*lists, staged.prod.other, Path(hb.REPO)/'experiments'/'hb'/'hb_review.py']:
            assert other_policy.why(path, write=True) == other_policy.why(path), path


# ---- LP-G revision 1: the checkout under the formal root, the feasibility mode's two exceptions together (the LP-G review's minor 3) ------------------------------
def test_lpg_r1_the_checkout_under_the_formal_root_takes_both_exceptions_together(both, monkeypatch):
    """C26.9-5: both modes run from <formal root>/code.  The feasibility review policy forbids the formal root except its code/, and code/archive/ except the sealed lists: a list passes both
    layers; the rest of plan/ and archive/, and the rest of the formal root, stay forbidden.  The arm policy of every process (both modes) reads the lists there too."""
    import shutil
    formal = Path(la.root(FORMAL))
    code = formal/'code'
    shutil.copytree(both.repo, code, symlinks=True)
    L.SW.pin(monkeypatch, code, [dict(spec) for spec in S.SEALED_LISTS])                 # sealed.REPO = <formal root>/code, the same lists (paths, sha256)
    monkeypatch.setattr(hb, 'REPO', code)
    h1 = next(spec['path'] for spec in S.SEALED_LISTS if spec['name'] == 'H1')
    lists = [code/spec['path'] for spec in S.SEALED_LISTS]
    rest = [code/PLAN_REL/both.prod.other.name, code/PLAN_REL/'x.py', code/(h1+'.bak'), code/'archive'/'x.json', code/PLAN_REL]
    for pair_set in (FEAS, FORMAL):
        policy = review.lpba_policy(pair_set)
        for path in lists:
            assert policy.why(path) is None and policy.why(path, write=True), (pair_set, path)
        for path in rest:
            assert policy.why(path) and 'forbidden root' in policy.why(path), (pair_set, path)
        assert policy.why(code/'experiments'/'hb'/'hb_review.py') is None and policy.why(code/'data'/'x') and policy.why(code/'docs'/'bayesreg'/'reviews'/'x.md')
    feas = review.lpba_policy(FEAS)
    roots = {d for d, _ in feas.deny_except}
    assert set(review._paths([formal])) <= roots and set(review._paths([code/'archive'])) <= roots     # the two layers
    assert {s for d, s in feas.deny_except if d in review._paths([code/'archive'])} == {str(p) for p in lists if 'archive' in p.parts}
    for path in (formal/'work'/'method'/PAIR/'output'/'final.npz', formal/'stage'/PAIR/'pair.npz', formal/'codex'/'x.py'):
        assert feas.why(path), path
    for pair_set, pair in ((FORMAL, PAIR), (FEAS, FEAS_PAIR)):
        ctx = L.ctx_of(pair_set, pair)
        for stage in STAGES:
            arm = R.lpba_policy_of(ctx, stage)
            assert all(arm.classify(p, False) == ('allowed', 'input') for p in lists) and not any(arm.classify(p, False)[0] == 'allowed' for p in rest[:-1])
            assert arm.classify_listing(code/PLAN_REL)[0] == 'forbidden'


# ---- (iv) across the LPBA entries (LP-G revision 1: two statements, the LP-G review's minor 2) ---------------------------------------------------------------
def test_lpg_iv_every_lpba_entry_reads_what_it_needs_of_archive_and_nothing_else(both, monkeypatch, linked):
    """For every LPBA entry with a spy: (1) every file of archive/ it rightly reads is allowed -- the three sealed lists there (all entries), and for XB's LPBA driver also the archive/ files of
    check_frozen and check_environment (xb_core.archive_reads, per arm); (2) every other file of archive/ is not -- the rest of plan/ (a file, a .py, a name beginning like a list's),
    archive/x.json, BL2's frozen files (but for XB), another group's smoke_cmd.md and another harness file (XB).  The code tree is the lists' (production: one checkout).
    Two kinds keep a documented difference in (2), recorded here, not changed by LP-G:
      hp_core.Policy (lpba_common.Policy and lpdev_common.Policy: prep, Atropos, HI, the admission dry run, lpdev_a/b/c) has no archive/ root: a non-code file there is 'unlisted' (refused),
        but a .py of the code tree is code -- shown on its own code tree (hp_core.CODE_TREE), the tests' and production's checkout alike, paths only;
      the evaluation entry's spy (bayesreg.eval.lpba_eval) and make_eval_manifest_lpba's guard watch the data roots and the results root's hold-out only: archive/ is not watched."""
    import admission_dry_run
    import atropos_lpba
    import lpdev_a
    import lpdev_b
    import lpdev_c
    import make_eval_manifest_lpba as M
    import prep_lpba
    import xb_drive
    import xb_policy
    from bayesreg.eval import lpba_eval
    xc = W.XW.xc
    repo = both.repo
    archive_lists = [str(p) for p in hc.sealed_list_paths() if p.startswith(os.path.realpath(repo/'archive')+'/')]
    h1 = next(spec['path'] for spec in S.SEALED_LISTS if spec['name'] == 'H1')
    plan_rest = [both.prod.other, linked/PLAN_REL/both.prod.other.name, repo/PLAN_REL/'x.py', repo/(h1+'.bak'), both.prod.stray]
    bl2 = [repo/xc.ARCHIVE_HARNESS/'drive.py', repo/xc.BL2_PROTOCOL, repo/xc.ENV_RECORDS_REL/'ants'/'versions.json']
    rest = [str(p) for p in plan_rest+bl2]
    assert len(archive_lists) == 3
    entries = []                                                                         # (name, kind, allowed(path) -> bool, rightly read, the rest)
    for name, policy in arm_policies():
        entries.append((f'ArmPolicy {name}', 'deny', lambda p, policy=policy: policy.classify(p, False)[0] == 'allowed', archive_lists, rest))
    for pair_set in (FORMAL, FEAS):
        policy = code_tree_review_mode(monkeypatch, both, pair_set).policy
        entries.append((f'review {pair_set}', 'deny', lambda p, policy=policy: policy.why(p) is None, archive_lists, rest))
    with monkeypatch.context() as m:                                                   # XB's LPBA driver as run_cell makes it, the code tree the lists' (production); the frozen configuration a copy
        m.setattr(xc, 'REPO', repo)
        for pair_set, pair in ((FORMAL, PAIR), (FEAS, FEAS_PAIR)):
            for arm in xc.LPBA_ARMS:
                paths = xc.cell_paths(arm, pair_set, pair)
                policy = xb_policy.Policy(xb_drive.policies(arm, pair_set, pair, paths, xc.pair_path(pair_set, pair), extra_exact=hc.sealed_list_paths()))
                checks = [str(p) for p in xc.archive_reads(arm)]
                assert len(checks) > 3, arm                                              # the configuration was read (the drive.py, the protocol, a smoke_cmd.md and more)
                others = [p for p in rest if p not in checks]+[str(repo/xc.SMOKE_CMD_REL/'niftyreg'/'smoke_cmd.md'), str(repo/xc.ARCHIVE_HARNESS/'other.py')]
                entries.append((f'xb driver {pair_set} {arm}', 'deny', lambda p, policy=policy: policy.classify(os.path.realpath(p))[0] == 'allowed', archive_lists+checks, others))
    hp_policies = []
    for pair_set, pairs in ((FORMAL, [('S05', 'S06')]), (FEAS, [('S01', 'S02'), ('S02', 'S01')])):
        hp_policies += [(f'prep {pair_set}', prep_lpba.policy(pair_set)), (f'atropos {pair_set}', atropos_lpba.policy(pair_set, pairs)),
                        (f'hi {pair_set}', L.hi_lpba.policy(pair_set, pairs))]
    hp_policies += [('admission dry run', lc.Policy(name='lpba admission dry run (C26.9-3)', read_exact=admission_dry_run.code_tree_reads(), write_inside=[], results=hc.RESULTS)),     # its run()'s
                    ('lpdev_a', lpdev_a.policy()), ('lpdev_b', lpdev_b.policy(['S01', 'S02'], [('S01', 'S02')])), ('lpdev_c', lpdev_c.policy())]
    for name, policy in hp_policies:
        entries.append((name, 'hp', lambda p, policy=policy: policy.classify(p, False)[0] == 'allowed', archive_lists, rest))
    S._reset_cache()
    spy = lpba_eval.start_spy(lpba_eval.spy_roots())
    try:
        for p in archive_lists+rest:
            spy.saw(p, 'rb')
        spy.allow([])
        entries.append(('lpba_eval spy', 'watch', lambda p: (spy.saw(p, 'rb'), True)[1], archive_lists, rest))     # SpyRefused would raise: a refusal fails the test
        verdicts = {name: ([allowed(p) for p in good], [allowed(p) for p in bad]) for name, kind, allowed, good, bad in entries}
    finally:
        lpba_eval.stop_spy()
    assert {e['kind'] for e in spy.events if e['path'] in archive_lists} == {'sealed_list'} and not [e for e in spy.events if e['path'] not in archive_lists] and spy.refused == []
    M.guard_on(lpba_eval.spy_roots())
    try:
        for p in archive_lists+rest:
            M.guard_path(p)                                                              # GuardViolation would raise
    finally:
        hits = M.guard_off()
    assert hits == []
    kinds = {name: kind for name, kind, *_ in entries}
    for name, (good, bad) in verdicts.items():
        assert all(good), (name, good)
        if kinds[name] in ('deny', 'hp'):
            assert not any(bad), (name, bad)
        else:
            assert all(bad), (name, bad)                                                 # 'watch': archive/ is not watched (documented above)
    for name, policy in hp_policies:                                                     # hp_core's own code tree: a .py under archive/ is code there, a non-code file is not
        tree = Path(L.LE.hp.CODE_TREE)
        assert policy.classify(tree/PLAN_REL/'x.py', False) == ('allowed', 'code (the running code tree, wherever it lies: XF)'), name
        assert policy.classify(tree/xc.ARCHIVE_HARNESS/'drive.py', False) == ('allowed', 'code (the running code tree, wherever it lies: XF)'), name
        assert policy.classify(tree/PLAN_REL/'other_plan_file.tsv', False)[0] != 'allowed', name
