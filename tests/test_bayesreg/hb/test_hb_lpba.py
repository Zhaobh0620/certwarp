"""LP-B (runs/20261119-lpba/任务书_B_方法配置C与外部臂.md; contract C26.6, C26.4 L30-L50, C26.5-5, C26.8, C26.10 T8, T16 (this package's links), T19): the LPBA40 modes of the method arm
(HE: run_hx, he_holdout; its queue, review and step 7 in experiments/hb) and of the configuration C arm.  The world is tests/test_bayesreg/hb/_lpba_world.py's (HB's sandbox, LP-A's synthetic LPBA40
world and entries); HE's children are run in this process with stand-ins only where a frozen run, a GPU or the formal mode's rule one (LP-C2's) would be needed.  No real subject's file, no unit."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from . import _lpba_world as L
from . import _world as W

R, he, hc, hb, review, hm, hold, lc, G, S = L.R, L.he, L.hc, L.hb, L.review, L.hm, L.hold, L.lc, L.G, L.S
FORMAL, FEAS, PAIR, FEAS_PAIR = L.FORMAL, L.FEAS, L.PAIR, L.FEAS_PAIR
import firewall_c as fc  # noqa: E402
import hb_configc as C  # noqa: E402
import hb_store  # noqa: E402

la = hold._lpba()


def quiet(*a, **k):
    pass


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(R.core, 'frozen_api', lambda: SimpleNamespace(PreoptConfig=lambda **k: dict(k), four_level_grids=None, downsample_array=None))     # no frozen tree here
    monkeypatch.setattr(R, 'enable_jit', lambda ctx: False)                             # the frozen tree's JIT cache (jd_curl.three_d.jit_cache is the frozen tree's): not used here
    return L.make(tmp_path, monkeypatch)


@pytest.fixture
def staged(world, monkeypatch):
    L.prepare(world, monkeypatch, FORMAL, hi=False, atropos=False)
    return world


# ---- T8: the pair sets, the contexts, the names ----------------------------------------------------------------------------------------------------
def test_the_pair_sets_and_the_roots_are_lpba_commons():
    import xb_core
    assert hold.LPBA_PAIR_SETS == lc.PAIR_SETS == xb_core.LPBA_PAIR_SETS == la.PAIR_SETS and set(hold.LPBA_PAIR_SETS) <= set(hold.PAIR_SETS)
    assert hold.PAIR_SETS[:2] == ('dev', 'holdout') and hm.PAIR_SET_CHOICES == ('holdout',)+hold.LPBA_PAIR_SETS and C.PAIR_SETS == ('holdout', 'dev')+hold.LPBA_PAIR_SETS
    real = Path('/home/rmcfz/seq_vp_tag_results')
    assert R.LPBA_ROOT_NAMES == ('lpba40_test_v1', 'lpba40_feas_v1')
    for name in R.LPBA_ROOT_NAMES:                                                     # the dev subcommand refuses the LPBA roots (only lpba-pair runs an LPBA pair)
        assert R.holdout_roots_problem(real/name/'stage', '/nonexistent/out') and R.holdout_roots_problem('/nonexistent/stage', real/name/'work'/'method'/PAIR)
    assert la.P19_STOP_AT == 2


def test_the_lpba_context_is_fixed_and_checked_by_pair_set(staged):
    ctx = L.ctx_of()
    r = lc.roots(FORMAL)
    assert ctx['pair_set'] == FORMAL and Path(ctx['stage_dir']) == r.stage and Path(ctx['out']) == r.root/'work'/'method'/PAIR == hb.cell(PAIR, FORMAL)
    assert ctx['paths']['pred_root'] == str(r.pred_root) and ctx['paths']['atropos_root'] == str(r.atropos) and ctx['paths']['allowlist'] == str(G.lpba40_allowlist_path())
    assert R.holdout_context_problem(ctx) is None and ctx['settings_sha256'] == la.SETTINGS_SHA256 == W.hb_method.SETTINGS_SHA256
    feas = lc.roots(FEAS)
    for change in (dict(paths=dict(ctx['paths'], pred_root=str(feas.pred_root))), dict(paths=dict(ctx['paths'], allowlist='/x/dev_subjects.json')), dict(out=str(Path(ctx['out']).with_name('other'))),
                   dict(stage_dir=str(feas.stage)), dict(launcher=dict(kind='direct')), dict(pair_set=FEAS), dict(pair_set='holdout'), dict(pair='LPBA40_S06__LPBA40_S05'),
                   dict(settings_sha256='0'*64, test_context=None)):
        bad = dict(ctx, **change)
        assert R.holdout_context_problem(bad) and R.cli_context_problem(bad), change
        assert R.run_pair(bad) == 2 and not Path(bad['out']).exists()
    assert R.evaluate_main(ctx) == 2                                                    # C26.6-2, L50: never evaluated by run_hx


@pytest.mark.parametrize('pair_set, pair', [(FORMAL, FEAS_PAIR), (FEAS, PAIR), (FORMAL, 'LPBA40_S06__LPBA40_S05'), (FEAS, 'LPBA40_S02__LPBA40_S03'), (FORMAL, 'OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1')])
def test_a_context_takes_only_its_modes_table(world, pair_set, pair):
    """C26.6-2 (revision 11b): the formal context refuses the feasibility pairs and the reverse; the direction is the table's; nothing else."""
    with pytest.raises(ValueError):
        L.ctx_of(pair_set, pair)


def test_rule_one_of_the_formal_mode_fails_closed_and_feasibility_is_heads(staged, monkeypatch, tmp_path):
    # LP-C2 integration: rule one is implemented now (lpba_arms.pinned_code_problems -> lpba_common, LPBA40 constants); the expected change of decision_log line 584 ruling 3:
    # this code tree has no committed runs/lpba40_test_v1/pinned_code.json, so the formal mode is refused by rule one itself (no P_L), not by the old stand-in.
    with pytest.raises(ValueError, match='runs/lpba40_test_v1/pinned_code.json is not committed'):
        R.make_lpba_context(FORMAL, PAIR, device='cpu', check_identity=True)
    ctx = L.ctx_of()
    monkeypatch.setattr(R.he, 'identity', lambda root, frozen=None: (dict(repo=dict(head=ctx['code_head'])), []))
    _, problems = R.check_identity(dict(ctx, check_identity=True), tmp_path, 'identity.json')
    assert len(problems) == 1 and 'runs/lpba40_test_v1/pinned_code.json is not committed' in problems[0]
    feas = L.ctx_of(FEAS, FEAS_PAIR)
    assert R.check_identity(dict(feas, check_identity=True), tmp_path, 'identity_f.json')[1] == []          # C26.9-1: not rule one; HEAD as the context's
    monkeypatch.setattr(R.he, 'identity', lambda root, frozen=None: (dict(repo=dict(head='d'*40)), []))
    assert R.check_identity(dict(feas, check_identity=True), tmp_path, 'identity_g.json')[1]
    with pytest.raises(ValueError, match='HEAD'):
        R.make_lpba_context(FEAS, FEAS_PAIR, device='cpu', check_identity=False, pinned='e'*40)


def test_steps_1_to_3_open_no_manifest_and_the_branch_never_uses_sealed_admission(staged, monkeypatch):
    import spies as spying
    ctx = L.ctx_of()
    S._reset_cache()
    spy = spying.Spies(R.pair_path(ctx)).install()
    try:
        record, allowlist = hold.lpba40_pre_manifest(FORMAL, PAIR, ctx['stage_dir'])
    finally:
        spy.remove()
    assert spy.loader_calls == [] and not [p for p in spy.opened if p.startswith(str(lc.roots(FORMAL).stage))], spy.opened
    assert record['pair_table'] == dict(path=lc.TEST_PAIRS_REL, sha256=lc.LIST_SHA256[lc.TEST_PAIRS_REL], row=1, line='S05\tS06')
    assert {s: (r['step'], r['admitted_by_lpba40']) for s, r in record['check_subject'].items()} == dict(S05=(None, True), S06=(None, True))
    assert record['allowlist']['path'] == G.LPBA40_ALLOWLIST_REL and record['reject_constant_sha256'] == G.LPBA40_TEST_REJECT_SHA256 and allowlist.path == str(G.lpba40_allowlist_path())
    with pytest.raises(hold.HoldoutAdmissionRefused, match='stage'):
        hold.lpba40_pre_manifest(FORMAL, PAIR, lc.roots(FEAS).stage)
    calls = []
    real = hold.lpba40_pre_manifest
    monkeypatch.setattr(hold, 'lpba40_pre_manifest', lambda *a, **k: calls.append('pre') or real(*a, **k))
    monkeypatch.setattr(R, 'sealed_admission', lambda ctx: (_ for _ in ()).throw(AssertionError('sealed_admission is not the LPBA branch')))
    S._reset_cache()
    pin, rec = R.admitted_images(ctx)
    assert calls == ['pre'] and pin.provenance['keys_read'] == ['fixed', 'moving', 'initial_u'] and rec['manifest_subjects'] == dict(fixed='S05', moving='S06')
    assert rec['lpba40_guard']['fixed'] == rec['lpba40_guard']['moving'] == dict(admitted_by_lpba40=True) and rec['rule'].startswith('he_holdout.lpba_admitted_images')
    assert hold.allowlist_of(ctx, pin) == str(G.lpba40_allowlist_path())
    with pytest.raises(hold.HoldoutAdmissionRefused, match='allowlist'):
        hold.allowlist_of(ctx, SimpleNamespace())
    with pytest.raises(hold.HoldoutAdmissionRefused, match='allowlist'):
        hold.allowlist_of(dict(ctx, paths=dict(ctx['paths'], allowlist='/x.json')), pin)
    monkeypatch.setattr(G, 'LPBA40_ALLOWLIST_SHA256', 'f'*64)                         # the pinned allowlist does not hold: refused before the manifest
    S._reset_cache()
    with pytest.raises(hold.HoldoutAdmissionRefused, match='allowlist'):
        R.admitted_images(ctx)


def test_the_t88_child_records_the_lpba_admission_and_hands_on_the_pinned_path(staged, monkeypatch):
    import fast_channels
    ctx = L.ctx_of()
    Path(ctx['out']).mkdir(parents=True)
    seen = {}

    def stop(pin, *, unet, seg_root, allowlist, workers):
        seen.update(allowlist=allowlist, seg_root=str(seg_root), pred_root=str(unet.pred_root))
        raise SystemExit('test: stopped after the admission')
    monkeypatch.setattr(fast_channels, 'read_predicted', stop)
    monkeypatch.setattr(he, 'segmenter_start_problems', lambda s, registry=None: ([], dict(test=True)))
    S._reset_cache()
    assert R.t88_main(ctx) == 0
    run = he.read_json(Path(ctx['out'])/'t88'/'run.json')
    assert run['status'] == he.T88_REJECTED and run['admission_second_lock']['rule'].startswith('he_holdout.lpba_admitted_images')
    assert run['admission_second_lock']['lpba40_guard']['fixed'] == dict(admitted_by_lpba40=True) and run['data']['subject_guard']['lpba40_guard']['moving'] == dict(admitted_by_lpba40=True)
    assert seen == dict(allowlist=str(G.lpba40_allowlist_path()), seg_root=str(lc.roots(FORMAL).atropos), pred_root=str(lc.roots(FORMAL).pred_root))


def test_the_repair_and_the_third_certification_admit_again(staged, monkeypatch):
    """C26.6-2 (L45, L47): the repair child admits the pair in its own process (no T88 field here: an error, a fallback -- never 'rejected'); with the pinned allowlist broken it is refused, and
    repair_decision judges 'rejected'.  The third certification admits again and records it."""
    ctx = L.ctx_of()
    out = Path(ctx['out'])
    out.mkdir(parents=True)
    S._reset_cache()
    assert R.repair_main(ctx) == 0
    rec = he.read_json(out/'repair'/'repair.json')
    assert rec['status'] == 'error' and rec['admission_second_lock']['rule'].startswith('he_holdout.lpba_admitted_images') and he.repair_decision(rec, None)[0] == 'fallback'
    (out/'repair').rename(out/'repair_first')
    with monkeypatch.context() as m:
        m.setattr(G, 'LPBA40_ALLOWLIST_SHA256', 'f'*64)
        S._reset_cache()
        assert R.repair_main(ctx) == 0
    rec = he.read_json(out/'repair'/'repair.json')
    assert rec['status'] == he.T88_REJECTED and he.repair_decision(rec, None)[0] == 'rejected'
    (out/'repair').rename(out/'repair_second')
    S._reset_cache()
    pin, _ = R.admitted_images(ctx)
    L.write_cell(ctx, pin)
    S._reset_cache()
    assert R.third_cert_main(ctx, certification=W.Certify().pair()) == 0
    third = he.read_json(out/'output'/'third_certificate.json')
    assert third['passed'] is True and third['admission_second_lock']['rule'].startswith('he_holdout.lpba_admitted_images')


def test_the_unit_names_are_tag_lp(staged):
    names = {hb.unit_name('method', PAIR, FORMAL): 'tag-lp-he-S05-S06', hb.unit_name('review', PAIR, FORMAL): 'tag-lp-review-S05-S06',
             hb.unit_name('configc', PAIR, FORMAL): 'tag-lp-c-S05-S06', hb.unit_name('method', FEAS_PAIR, FEAS): 'tag-lp-feas-he-S01-S02',
             hb.queue_unit('method', FORMAL): 'tag-lp-he-queue', hb.queue_unit('configc', FEAS): 'tag-lp-feas-c-queue', hb.queue_unit('review', FORMAL): 'tag-lp-review-queue'}
    assert names == {v: v for v in names.values()}
    ctx, feas = L.ctx_of(), L.ctx_of(FEAS, FEAS_PAIR)
    assert R.make_launcher(ctx).unit_prefix == 'tag-lp-he-c' and R.third_launcher(ctx).unit_prefix == 'tag-lp-he-c3'
    assert R.make_launcher(feas).unit_prefix == 'tag-lp-feas-he-c' and R.third_launcher(feas).unit_prefix == 'tag-lp-feas-he-c3'
    argv = hb.unit_argv('method', PAIR, [hb.REPO/'experiments/he/run_hx.py', 'lpba-pair', '--pair-set', FORMAL, '--pair', PAIR], code=hb.REPO, unit_dir=Path('/r/u'),
                        res=dict(he.SUPERVISOR_UNIT), gpu=True, log=Path('/r/u.log'), pair_set=FORMAL)
    assert '--unit=tag-lp-he-S05-S06' in argv and not [a for a in argv if 'tag-hb' in a or 'seq-vp' in a]
    assert hb.queue_launch('method', ['queue', '--pair-set', FORMAL], pair_set=FORMAL)[2] == '--unit=tag-lp-he-queue'
    assert hb.unit_name('method', W.first_pair()) == f'tag-hb-he-{hc.short_pair(W.first_pair())}'           # the hold-out's names unchanged


# ---- T16: the whole admission chain on a pair of the 34 (this package's links) -----------------------------------------------------------------
def test_t16_the_admission_chain_of_the_method_on_s05_s06(world, monkeypatch, tmp_path):
    registry = L.prepare(world, monkeypatch, FORMAL)
    from bayesreg.io.guard import SubjectSealed, load_dev_allowlist
    from bayesreg.io.pair import load_pair, validate_segmentation_provenance
    from bayesreg.seg.structure_layer import ENSEMBLE_FILE, StructureLayer, StructureSource
    ctx = L.ctx_of()
    r = lc.roots(FORMAL)
    S._reset_cache()
    pin, rec = R.admitted_images(ctx)                                                   # L30, L40: the LPBA branch
    path = hold.allowlist_of(ctx, pin)
    mi = he.core.make_inputs()
    unet = SimpleNamespace(folds=L.LE.ROOT/'splits/bayesreg_v1/folds.json', folds_sha256=None)
    assert mi.a4_admission(['S05', 'S06'], path, unet)['subjects'] == ['S05', 'S06']     # L42: the development branch on the LPBA allowlist (never_for_training false, D143)
    registry_path = Path(registry.path)
    (registry_path.parent/ENSEMBLE_FILE).write_text(json.dumps(dict(registry_sha256=registry.sha256, member_check=dict(passed=True, format='test'))))     # the synthetic registry as a staging one
    src = StructureSource(root=r.pred_root, registry_path=registry_path, allowlist_path=Path(path), models_root=tmp_path/'members', image_root=tmp_path/'none', segmenter_id='unet35_v4_ens',
                          infer='never', allow_staging=True)
    layer = StructureLayer(pin, src)                                                    # L43: the allowlist from the pinned path, its sha256 load_pair's, the D24 preflight
    assert layer.preflight['contact_union_contains_pair_subject'] is False and layer.allowlist.sha256 == pin.provenance['subject_guard']['allowlist_sha256']
    for side in ('fixed', 'moving'):
        _, accepted, _ = layer.load(side)                                               # the structure provenance (recheck_sealed_pair: None for the 34, L19)
        assert list(accepted['input_sha256'].values()) == [pin.source_sha256[side]]
        wm, _ = mi.atropos_wm(r.atropos, pin, side)                                     # L44: the tissue layer, no allowlist needed
        assert wm.shape == pin.grid.shape
    allowlist = load_dev_allowlist(path)
    for side in ('fixed', 'moving'):
        tissue = json.loads((r.atropos/getattr(pin, side+'_subject')/'provenance.json').read_text())
        validate_segmentation_provenance(tissue, pin, side=side, layer='tissue', allowlist=allowlist)
    assert {s: v['policy'] for s, v in L.hi_lpba.select_models(registry, ['S05', 'S06'], allowlist).items()} == dict(S05='all_folds_mean', S06='all_folds_mean')     # L48
    out = Path(ctx['out'])
    out.mkdir(parents=True)
    S._reset_cache()
    assert R.repair_main(ctx) == 0 and he.repair_decision(he.read_json(out/'repair'/'repair.json'), None)[0] != 'rejected'      # L45
    fw = fc.derive_pair(pin, tmp_path/'derived', pair=PAIR)                              # L46: configuration C's derived pair from the admitted pin
    gate, problems = fc.firewall_check(pin, tmp_path/'derived', pair=PAIR)
    manifest = json.loads((tmp_path/'derived'/'manifest.json').read_text())
    assert problems == [] and gate['passed'] and 'label_sets' not in manifest and manifest['pair'] == fc.slot_name(PAIR) == PAIR and fw['seg_keys_zero'] == list(fc.SEG_KEYS)
    assert manifest['weights'] == fc.frozen_weights(pin.grid)
    import xb_core
    import xb_drive
    record = xb_core.check_pair_input(r.stage/PAIR/'pair.npz', fixed='S05', moving='S06', allowlist=allowlist)          # L33
    assert record['guard']['lpba40_guard']['fixed'] == dict(admitted_by_lpba40=True)
    S._reset_cache()
    admitted, _ = xb_drive.lpba_admission(FORMAL, 'S05', 'S06')
    assert all(v['admitted_by_lpba40'] for v in admitted['admission']['subjects'].values())
    # the counter-examples: never_for_training true (refused in a4_admission), the development allowlist (refused by name in load_pair)
    doc = json.loads(world.allowlist_path.read_text(encoding='utf-8'))
    doc['sources'][-1]['never_for_training'] = True
    never = L.LE.W.write_json(tmp_path/'never.json', doc)
    with pytest.raises(SystemExit, match='never_for_training'):
        mi.a4_admission(['S05', 'S06'], never, unet)
    dev = load_dev_allowlist(L.LE.W.write_json(tmp_path/'dev_only.json', dict(L.LE.lpba_document(world.HW.CONTACT), sources=L.LE.lpba_document(world.HW.CONTACT)['sources'][:3],
                                                                                 subjects=dict(OASIS=list(L.LE.W.DEV)+list(world.HW.CONTACT), LPBA40=[f'S{n:02d}' for n in range(1, 25)]))))
    with pytest.raises(SubjectSealed, match='LPBA40_TEST_REJECT'):
        load_pair(r.stage/PAIR/'pair.npz', allowlist=dev)


# ---- the method queue, the review, step 7 (feasibility mode, end to end with HE played) -------------------------------------------------------
class LpbaSystem(W.hb.System):
    """The units of an LPBA mode played (as test_hb_queues.FakeSystem for the hold-out): the method unit writes HE's cell (its own LPBA context), the review unit its records."""

    def __init__(self, pair_set, outcomes):
        self.pair_set, self.outcomes, self.runs, self.gates = pair_set, list(outcomes), [], []

    def run(self, argv, log_path):
        self.runs.append(argv)
        outcome = self.outcomes.pop(0)
        unit_dir = Path(argv[argv.index('--record-dir')+1])
        unit_dir.mkdir(parents=True, exist_ok=True)
        (unit_dir/'unit_start.json').write_text('{}')
        inner = argv[argv.index('--')+1:]
        pair = inner[inner.index('--pair')+1]
        assert inner[inner.index('--pair-set')+1] == self.pair_set
        if Path(inner[2]).name == 'run_hx.py':
            assert inner[3] == 'lpba-pair'
            line = outcome.get('line')
            ctx = R.make_lpba_context(self.pair_set, pair, device='cpu', check_identity=False)
            if line in he.OUTPUT_STATUSES:
                u, _ = W.write_cell(ctx, status=line, step4=False)
                if line != he.HX:
                    np.savez(Path(ctx['out'])/'repair'/'certified.npz', u=u)            # a fallback pair whose repair also left a field: kept (as test_hb_queues.FakeSystem)
            elif line is not None:
                Path(ctx['out']).mkdir(parents=True)
                he.write_json(Path(ctx['out'])/'line.json', dict(line_status=line, rejection=dict(stage='admission')))
        else:
            cell = hb.cell(pair, self.pair_set)
            rev = cell/'review'
            rev.mkdir(exist_ok=True)
            line = he.read_json(cell/'output'/'line.json')
            path = hb.write_once(rev/review.REVIEW_RECORD, dict(review.finalized_part(cell, line=line, review_doc=dict(consistent=True)), finalized=True))
            hb.write_terminal(cell, pair=pair, status=line['line_status'], written_by='test', output_dir='output', review=dict(path='review/review.json', sha256=hb.sha(path)))
        (unit_dir/'unit_end.json').write_text(json.dumps(dict(exit_code=0, killed_by_signal=None, killed_by_supervisor=False, memory_events=dict(oom=0, oom_kill=0), dmesg_readable=True,
                                                              oom_lines=[], dxg_lines=0)))
        Path(log_path).write_text('Finished with result: success\n')
        return 0

    def uptime(self):
        return 10.

    def dmesg(self):
        return ''

    def journal(self, unit, since):
        return ''

    def gate(self, kind, pair_set='holdout'):
        self.gates.append((kind, pair_set))
        return dict(passed=True)

    def disk(self, need_d, need_home):
        return dict(passed=True, free_gib={})

    def sleep(self, seconds):
        raise AssertionError('no hold file in the tests')


def code_run():
    return L.detached_git('c'*40)


def no_pinned_record(run):
    """LP-C2 integration review minor 8: `run` (the stand-in git, which answers every command), except that `git show <rev>:runs/lpba40_test_v1/pinned_code.json` fails as a git
    without that committed record does (exit 128), so that rule one's refusal is the one a code tree without P_L gets ("is not committed", as at test_rule_one_of_... :75)."""
    import subprocess

    def wrapped(argv, **kwargs):
        if 'show' in argv and any(str(a).endswith(':runs/lpba40_test_v1/pinned_code.json') for a in argv):
            return subprocess.CompletedProcess(argv, 128, '', "fatal: path 'runs/lpba40_test_v1/pinned_code.json' does not exist (test)")
        return run(argv, **kwargs)
    return wrapped


def test_the_feasibility_queue_review_and_step_7(world, monkeypatch, tmp_path):
    L.prepare(world, monkeypatch, FEAS, atropos=False)
    pairs = [FEAS_PAIR, 'LPBA40_S02__LPBA40_S01']
    system = LpbaSystem(FEAS, [dict(line=he.HX), dict(line=he.FALLBACK_ID)])
    assert hm.queue(pair_set=FEAS, system=system, echo=quiet, code_run=code_run()) == 0
    units = [a for argv in system.runs for a in argv if a.startswith('--unit=')]
    assert units == ['--unit=tag-lp-feas-he-S01-S02', '--unit=tag-lp-feas-he-S02-S01'] and system.gates == [('method', FEAS)]*2
    for pair, status in zip(pairs, (he.HX, he.FALLBACK_ID)):
        step = he.read_json(hb.record_dir('method', pair, FEAS)/'step4.json')
        assert step['status'] == status and step['went_on'] and hb.lock_path('method', pair, FEAS).is_file() and hm.state(pair, FEAS) == 'done'
    assert not (hb.cell(pairs[0], FEAS)/'repair'/'certified.npz').exists() and (hb.cell(pairs[1], FEAS)/'repair'/'certified.npz').exists()     # C26.6-7 (fallback: kept)
    assert hb.lpba_hi(FEAS, ['S01', 'S02'])[1] == []
    review_system = LpbaSystem(FEAS, [dict(), dict()])
    assert review.queue(pair_set=FEAS, system=review_system, echo=quiet, code_run=code_run()) == 0
    final = he.read_json(hb.review_root(FEAS)/'finalized_outputs.json')
    assert final['pair_set'] == FEAS and sorted(final['pairs']) == sorted(pairs) and (hb.review_root(FEAS)/'input_manifest.json').is_file()
    assert [a for argv in review_system.runs for a in argv if a.startswith('--unit=')] == ['--unit=tag-lp-feas-review-S01-S02', '--unit=tag-lp-feas-review-S02-S01']
    import xb_core                                                                       # ConvexAdam-v4seg's pin of the method's t88/run.json: the finalized record and the cell (no move)
    found, why = xb_core.method_t88_lookup(FEAS_PAIR, FEAS)
    assert why is None and found[0] == hb.cell(FEAS_PAIR, FEAS)/'t88'/'run.json' and found[1] == hb.sha(found[0]) == xb_core.method_t88_sha256(FEAS, FEAS_PAIR)
    repo = L.git_repo(tmp_path/'code_tree')                                              # step 7, feasibility (review I2): the lock to the feasibility root, nothing committed, no field deleted
    head = W.XW.git(repo, 'rev-parse', 'HEAD').stdout.strip()
    lock = hb_store.lpba_input_lock(FEAS, repo=repo)
    assert lock == L.lc.roots(FEAS).root/'hb_input_lock.json' and W.XW.git(repo, 'rev-parse', 'HEAD').stdout.strip() == head and not (repo/'runs').exists()
    rows = L.read(lock)['rows']
    assert {r['kind'] for r in rows} == {'prediction', 'hi_record', 'method_field'} and len([r for r in rows if r['kind'] == 'prediction']) == 6
    with pytest.raises(hc.Refused, match='input_lock'):
        hb_store.lpba_input_lock(FEAS, 'runs/lpba40_feas_v1/x.json', repo=repo)
    with pytest.raises(hc.Refused, match='feasibility mode deletes no field'):
        hb_store.lpba_delete_fields(FEAS, 'runs/lpba40_feas_v1/hb_input_lock.json', repo=repo)
    assert all((hb.cell(p, FEAS)/'t88'/he.FIELD_FILE).exists() for p in pairs)


def test_the_formal_step_7_deletes_the_certified_fields_after_the_committed_lock(world, monkeypatch, tmp_path):
    """C26.6-7, prereg §8.2 step 7 (formal): the lock in runs/lpba40_test_v1/, committed, then the field of each hx_certified pair deleted (its record written first); rule one checked by the
    store itself (review minor 7) -- refused until LP-C2's hook passes (stubbed here)."""
    L.prepare(world, monkeypatch, FORMAL, atropos=False)
    repo = L.git_repo(tmp_path/'code_tree')
    with pytest.raises(hc.Refused, match='rule one'):
        hb_store.lpba_input_lock(FORMAL, repo=repo)
    monkeypatch.setattr(la, 'pinned_code_problems', lambda pinned=None, *, repo=None, run=None: (dict(pinned='c'*40, passed=True), []))     # LP-C2's rule one (stubbed)
    assert hm.queue(pair_set=FORMAL, system=LpbaSystem(FORMAL, [dict(line=he.HX)]), echo=quiet, code_run=code_run()) == 0
    assert review.queue(pair_set=FORMAL, system=LpbaSystem(FORMAL, [dict()]), echo=quiet, code_run=code_run()) == 0
    lock = hb_store.lpba_input_lock(FORMAL, repo=repo)
    rel = str(lock.relative_to(repo))
    assert rel == 'runs/lpba40_test_v1/hb_input_lock.json'
    rows = L.read(lock)['rows']
    with pytest.raises(hc.Refused, match='committed'):
        hb_store.lpba_delete_fields(FORMAL, rel, repo=repo)
    L.commit(repo, rel)
    done = hb_store.lpba_delete_fields(FORMAL, rel, repo=repo)
    assert [d['deleted'] for d in done] == [True] and not (hb.cell(PAIR, FORMAL)/'t88'/he.FIELD_FILE).exists()
    deleted = he.read_json(hb.record_dir('method', PAIR, FORMAL)/hb_store.LPBA_FIELD_DELETED)
    assert deleted['sha256'] == next(r['sha256'] for r in rows if r.get('pair') == PAIR) and deleted['lock']['path'] == rel


def test_a_unit_without_line_json_is_infra_failure_and_a_rejection_stops(world, monkeypatch):
    L.prepare(world, monkeypatch, FEAS, atropos=False)
    assert hm.queue(pair_set=FEAS, system=LpbaSystem(FEAS, [dict()]), echo=quiet, code_run=code_run()) == hb.EXIT['infra']        # C26.6-8
    step = he.read_json(hb.record_dir('method', FEAS_PAIR, FEAS)/'step4.json')
    assert step['status'] == 'infra_failure' and not step['went_on'] and hm.state(FEAS_PAIR, FEAS) == 'stopped'
    with pytest.raises(hc.Refused, match='authorizations'):                            # the feasibility mode has no resume (C26.6-2, C26.8-5)
        hm.load_resume(FEAS_PAIR, 'splits/x.json', FEAS)
    assert hm.queue(pair_set=FEAS, pairs=['LPBA40_S02__LPBA40_S01'], system=LpbaSystem(FEAS, [dict(line=he.REJECTED)]), echo=quiet, code_run=code_run()) == hb.EXIT['implementation']
    assert he.read_json(hb.terminal_path(hb.cell('LPBA40_S02__LPBA40_S01', FEAS)))['line_status'] == he.REJECTED


def test_the_formal_queues_refuse_without_rule_one(staged):
    for queue in (lambda: hm.queue(pair_set=FORMAL, system=LpbaSystem(FORMAL, []), echo=quiet, code_run=code_run()),
                  lambda: review.queue(pair_set=FORMAL, system=LpbaSystem(FORMAL, []), echo=quiet, code_run=code_run()),
                  lambda: C.queue(FORMAL, system=LpbaSystem(FORMAL, []), echo=quiet, code_run=code_run())):
        assert queue() == hb.EXIT['refused']
    assert not list(hb.lpba().batch_root(FORMAL).glob('hb_*'))
    record, problems = hm.lpba_start_checks(FORMAL, code_run=no_pinned_record(code_run()))
    # LP-C2 integration: rule one is implemented now (lpba_arms.pinned_code_problems -> lpba_common, LPBA40 constants); the expected change of decision_log line 584 ruling 3:
    # the stand-in git (code_run) gives no runs/lpba40_test_v1/pinned_code.json, so rule one refuses on that record.
    # Integration review minor 8: matched as :75 matches it ("is not committed"); the stand-in answers `git show` of the record as a git without it does (no_pinned_record).
    assert any('runs/lpba40_test_v1/pinned_code.json is not committed' in p for p in problems) and record['settings'] is None and record['pins'] is None


def test_the_review_unit_finalizes_an_lpba_cell_under_its_policy(staged):
    ctx = L.ctx_of()
    S._reset_cache()
    pin, _ = R.admitted_images(ctx)
    L.write_cell(ctx, pin)
    rec = hb.record_dir('method', PAIR, FORMAL)
    rec.mkdir(parents=True)
    he.write_json(rec/'step4.json', dict(status=he.HX, went_on=True, written_by='test'))
    mode = review.lpba_mode(FORMAL, None)
    mode.manifest().parent.mkdir(parents=True, exist_ok=True)
    mode.manifest().write_text(json.dumps(dict(schema='test', pairs=review.input_manifest([PAIR], FORMAL))))
    S._reset_cache()                                                                    # the hold-out's mode does not review an LPBA context (nor an LPBA mode another pair set's)
    assert review.review_pair(ctx, review.holdout_mode(None), batch='b', certification=W.Certify().pair(), probe=W.probe) == review.STOP_EXIT['not_finalized']
    assert he.read_json(Path(ctx['out'])/'review'/'review_not_finalized.json')['kind'] == 'pair_set'
    S._reset_cache()
    assert review.review_pair(ctx, mode, batch='test-batch', certification=W.Certify().pair(), probe=W.probe) == 0
    doc = he.read_json(Path(ctx['out'])/'review'/review.REVIEW_RECORD)
    assert doc['finalized'] is True and doc['mode'] == FORMAL and doc['admission']['rule'].startswith('he_holdout.lpba_admitted_images')
    assert he.read_json(hb.terminal_path(ctx['out']))['line_status'] == he.HX
    opened = he.read_json(Path(ctx['out'])/'review'/review.OPENED)
    assert opened['passed'] is True and opened['policy']['name'] == f'{FORMAL} review' and opened['policy']['seg_name'] == 'lpba40_test_v1'


def test_the_lpba_review_policy(staged):
    policy = review.lpba_policy(FORMAL)
    root = Path(la.root(FORMAL))
    for denied in (root/'eval'/'x.json', root/'fireants_greedy'/PAIR/'result.json', root/'work'/'configc'/PAIR/'output'/'disp.npz', Path(la.root(FEAS))/'stage'/'x',
                   Path(hc.HOLDOUT_ROOT)/'review'/'x', Path(hc.RESULTS)/'seg'/'unet35_v4_ens'/'pred'/'S05'/'probs36.npz', root/'locks'/'xb__ants_s__x.lock', Path('/mnt/d/data/LPBA40/x')):
        assert policy.why(denied), denied
    for allowed in (root/'stage'/PAIR/'pair.npz', root/'work'/'method'/PAIR/'output'/'final.npz', root/'review'/'input_manifest.json', root/'locks'/f'hb__method__{PAIR}.lock',
                    Path(hc.RESULTS)/'seg'/'unet35_v4_ens_lpba40_test_v1'/'pred'/'S05'/'provenance.json'):
        assert policy.why(allowed) is None, allowed
    assert review.holdout_policy().not_development == review.RESULTS_NOT_DEVELOPMENT and review.holdout_policy().seg_name == 'holdout_B_v1'      # the hold-out's unchanged


# ---- configuration C (C26.5-5; T19) -----------------------------------------------------------------------------------------------------------------
from .test_hb_configc import Launcher, fake_prepare, kernel, probe_of  # noqa: E402


@pytest.fixture
def configc(staged, monkeypatch):
    monkeypatch.setattr(fc, 'prepare', fake_prepare)
    monkeypatch.setattr(he, 'frozen_identity', lambda *a, **k: (dict(test=True), []))
    monkeypatch.setattr(la, 'code_record', lambda pair_set, pinned=None, **k: (dict(commit='c'*40, clean=True, pair_set=pair_set), []))     # the code identity (formal: rule one) is LP-C2's
    S._reset_cache()
    pin, _ = R.admitted_images(L.ctx_of())
    monkeypatch.setattr(W, 'grid', lambda shape=None: pin.grid)                          # the frozen run's stand-in writes a field of the LPBA pair's grid
    batch = C.roots(FORMAL)[0]/'batches'/'test'
    batch.mkdir(parents=True)
    return batch


def arm_table_field(results):
    """The LPBA40 arm table's configuration C field (configs/frozen/lpba40_test_v1_arm_table.json) with the real results root mapped to the sandbox's."""
    table = json.loads((L.LE.ROOT/'configs/frozen/lpba40_test_v1_arm_table.json').read_text(encoding='utf-8'))
    field = next(a for a in table['arms'] if a['name'] == 'configc')['field']
    return field, Path(field['root'].replace('{arm}', 'configc').replace('/home/rmcfz/seq_vp_tag_results', str(results)))


def resolve(document, field, root):
    """The evaluation's reading of the arm table's field keys (bayesreg.eval.holdout's dotted keys and its checks: a disp.npz under the root, its sha256)."""
    from bayesreg.eval.holdout import _inside, dotted
    path, digest = dotted(document, field['file_key']), dotted(document, field['sha256_key'])
    assert isinstance(path, str) and Path(path).name == 'disp.npz' and _inside(path, root) and Path(path).is_file() and hb.sha(path) == digest, (path, digest, root)
    return Path(path)


def test_t19_configc_writes_its_output_store_without_a_move_and_the_arm_table_finds_it(configc, monkeypatch):
    S._reset_cache()
    rc = C.run(FORMAL, PAIR, batch=configc, launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(), kernel=kernel())
    paths = C.cell_paths(FORMAL, PAIR)
    r = he.read_json(paths['result'])
    assert rc == 0, (r.get('error'), r.get('traceback'))
    assert r['status'] == 'ok' and 'move' not in r
    assert paths['store'] is None and r['output_store']['disp']['file'] == str(paths['work']/'output'/'disp.npz') and r['admission']['check_subject']['S05']['admitted_by_lpba40']
    assert r['firewall']['seg_keys_zero'] == list(fc.SEG_KEYS) and r['firewall']['gate']['passed'] and paths['work'] == Path(la.root(FORMAL))/'work'/'configc'/PAIR
    field, root = arm_table_field(lc.RESULTS)
    assert resolve(r, field, root) == paths['work']/'output'/'disp.npz'
    # the resume (C26.8-3: the formal mode's authorizations, LP-C2's -- simulated here: AUTH_DIRS and AUTH_PREFIX of D148): result_resume.json, work <pair>.resume, output_store there
    over = dict(r, status='infra_failure', output=None, output_store=None)
    hb.write_once(paths['overridden'], over)
    monkeypatch.setattr(hc, 'AUTH_DIRS', dict(hc.AUTH_DIRS, lpba40_test='splits/lpba40_test_v1'))
    monkeypatch.setattr(hc, 'AUTH_PREFIX', dict(hc.AUTH_PREFIX, lpba40_test='lpba40_test_v1_'))
    repo = W.XW.auth_repo()
    directory = hc.authorization_dir(FORMAL)
    # LP-C2 integration (decision_log line 584 ruling 3; contract C26.8-3): a resume written on BZ's behalf quotes BZ's words of line 520 word for word (holdout_common.LPBA40_BZ_WORDS)
    doc = dict(schema='bayesreg.hb.resume.v1', written_by="lead on BZ's behalf", words=hc.LPBA40_BZ_WORDS, time_local='2026-11-19 9:00 AM CST', utc='2026-11-19T15:00:00Z',
               reason='test', tool='hb', arm=C.ARM, pair=PAIR, pair_set=FORMAL, lock_sha256=hb.sha(paths['lock']), previous_record=str(paths['overridden']),
               previous_record_sha256=hb.sha(paths['overridden']))
    path = W.XW.write_json(directory/'lpba40_test_v1_hb_resume_1.json', doc)
    W.git(repo, 'add', '-A')
    W.git(repo, 'commit', '-q', '-m', 'resume')
    S._reset_cache()
    rc = C.run(FORMAL, PAIR, batch=configc, resume_rel=str(path.relative_to(hc.AUTH_ROOT)), launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(), kernel=kernel())
    assert rc == 0, [json.loads(p.read_text()).get('reason') for p in sorted((configc/'refusals').glob('*.json'))]
    resumed = he.read_json(paths['record']/'result_resume.json')
    work = paths['work'].with_name(paths['work'].name+'.resume')
    assert resumed['status'] == 'ok' and resumed['output_store']['disp']['file'] == str(work/'output'/'disp.npz') and C.effective(paths) == resumed
    assert resolve(resumed, field, root) == work/'output'/'disp.npz'


def test_configc_on_lpba_refuses_outside_its_table_and_only_the_formal_mode_has_authorizations(configc):     # integration review minor 8: the name of the LP-C2 behaviour
    S._reset_cache()
    assert C.run(FORMAL, 'LPBA40_S06__LPBA40_S05', batch=configc, launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(), kernel=kernel()) == C.EXIT['refused']
    refusals = sorted((configc/'refusals').glob('*.json'))
    assert [json.loads(p.read_text())['stage'] for p in refusals] == ['admission']
    with pytest.raises(hc.Refused, match='rerun|authorizations'):
        C.rerun_authorization(FORMAL, 'LPBA40_S06__LPBA40_S05')
    # LP-C2 integration (decision_log line 584 ruling 3): holdout_common knows the formal mode's authorizations now (AUTH_DIRS 'lpba40_test', D148); the feasibility mode has none
    assert C.declined(FORMAL, PAIR) is None and C.authorizations_problem(FORMAL) is None and C.authorizations_problem(FEAS) and C.authorizations_problem('holdout') is None
    assert C.hold_file(FORMAL) == C.hold_file(FEAS) == lc.roots(FORMAL).hold and hb.unit_name('configc', PAIR, FORMAL) == 'tag-lp-c-S05-S06'


# ---- review round (LP-B review 0/4/9; decision_log 2026-11-19 ~3:13 PM CST) ----------------------------------------------------------------------
from .test_hb_queues import FakeSystem, method_done, pairs as holdout_pairs, unsealed  # noqa: E402,F401   (the hold-out queues' own fixture and fakes)

BATCH_KEYS_HOLDOUT = {'existing_pins', 'code', 'settings', 'preregistration', 'unseal', 'schema', 'kind', 'settings_sha256', 'prereg_sha256', 'utc_start', 'disk_start', 'cells', 'gate',
                      'unit', 'hi', 'disk_batch', 'utc_end', 'disk_end', 'exit'}
MANIFEST_KEYS_HOLDOUT = {'schema', 'batch', 'code', 'settings_sha256', 'pairs', 'utc'}


def test_i1_the_hold_outs_batch_and_input_manifest_keep_their_keys(tmp_path, monkeypatch, unsealed):
    """Review I1: batch.json of the method queue and input_manifest.json of the review queue have the hold-out's keys (no pair_set); only an LPBA mode's carry it."""
    ps = holdout_pairs(1)
    monkeypatch.setattr(hb, 'rows', lambda: list(ps))
    assert hm.queue(pairs=ps, system=FakeSystem([dict(line=he.HX)]), echo=quiet) == 0
    batch = he.read_json(next(Path(hc.HOLDOUT_ROOT).glob('batches/hb_method_*/batch.json')))
    assert set(batch) == BATCH_KEYS_HOLDOUT
    assert review.queue(system=FakeSystem([dict(review='finalize')]), echo=quiet) == 0
    assert set(he.read_json(hb.review_root()/'input_manifest.json')) == MANIFEST_KEYS_HOLDOUT


def test_i1_an_lpba_batch_and_manifest_carry_the_pair_set(world, monkeypatch):
    L.prepare(world, monkeypatch, FEAS, atropos=False)
    assert hm.queue(pair_set=FEAS, system=LpbaSystem(FEAS, [dict(line=he.HX), dict(line=he.HX)]), echo=quiet, code_run=code_run()) == 0
    batch = he.read_json(next(la.batch_root(FEAS).glob('hb_method_*/batch.json')))
    assert set(batch) == BATCH_KEYS_HOLDOUT-{'unseal', 'preregistration'}|{'pins', 'pair_set'} and batch['pair_set'] == FEAS      # the LPBA pins (lpba_common.check_pins) hold its preregistration
    assert review.queue(pair_set=FEAS, system=LpbaSystem(FEAS, [dict(), dict()]), echo=quiet, code_run=code_run()) == 0
    assert set(he.read_json(hb.review_root(FEAS)/'input_manifest.json')) == MANIFEST_KEYS_HOLDOUT|{'pair_set'}


class ConfigcSystem(LpbaSystem):
    """The configuration C units of an LPBA mode, its driver run in this process (hb_configc.run with test_hb_configc's stand-ins of the frozen run)."""

    def run(self, argv, log_path):
        self.runs.append(argv)
        unit_dir = Path(argv[argv.index('--record-dir')+1])
        unit_dir.mkdir(parents=True, exist_ok=True)
        (unit_dir/'unit_start.json').write_text('{}')
        inner = argv[argv.index('--')+1:]
        get = lambda k: inner[inner.index(k)+1] if k in inner else None     # noqa: E731
        S._reset_cache()
        rc = C.run(get('--pair-set'), get('--pair'), batch=Path(get('--batch')), pinned=get('--pinned'), launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(), kernel=kernel())
        (unit_dir/'unit_end.json').write_text(json.dumps(dict(exit_code=rc, killed_by_signal=None, killed_by_supervisor=False, memory_events=dict(oom=0, oom_kill=0), dmesg_readable=True,
                                                              oom_lines=[], dxg_lines=0)))
        Path(log_path).write_text('Finished with result: success\n')
        return 0


import xb_chain  # noqa: E402


class XbSystem(xb_chain.System):
    """The external arms' units of an LPBA mode, the driver run in this process (xb_drive.run_cell with XB's FakeMethod), its supervisor's records as xb_unit writes them."""

    def __init__(self):
        self.runs = []

    def run(self, argv, log_path):
        import xb_drive
        self.runs.append(argv)
        unit_dir = Path(argv[argv.index('--record-dir')+1])
        unit_dir.mkdir(parents=True, exist_ok=True)
        (unit_dir/'unit_start.json').write_text('{}')
        get = lambda k: argv[argv.index(k)+1] if k in argv else None     # noqa: E731
        batch = next(a.split('=', 2)[2] for a in argv if a.startswith('--setenv=XB_BATCH_DIR='))
        deps = W.XW.deps()
        deps.code_run = L.detached_git('c'*40)
        S._reset_cache()
        rc = xb_drive.run_cell(get('--arm'), get('--pair-set'), get('--pair'), deps=deps, batch_dir=batch, pinned=get('--pinned'))
        (unit_dir/'unit_end.json').write_text(json.dumps(dict(exit_code=rc, killed_by_signal=None, killed_by_supervisor=False, memory_events=dict(oom=0, oom_kill=0), dmesg_readable=True,
                                                              oom_lines=[], dxg_lines=0)))
        Path(log_path).write_text('Finished with result: success\n')
        return rc

    def uptime(self):
        return 10.

    def dmesg(self):
        return ''

    def journal(self, unit, since):
        return ''

    def gate(self, pair_set=None):
        return dict(passed=True)

    def disk(self, need_d, need_home):
        return dict(passed=True, free_gib={})

    def sleep(self, seconds):
        raise AssertionError('no hold file in the tests')

    def now(self):
        return 0.


def test_i2_the_feasibility_chain_runs_on_one_commit(world, monkeypatch):
    """Review I2: the feasibility chain on one detached checkout (HEAD c...c): the method queue, the review queue, the input lock by hb_store's CLI (to the feasibility root, nothing committed,
    no field deleted), the configuration C queue, the external arms' chain -- every start finds the same commit; a commit made after the method arm (HEAD moved) is refused."""
    L.prepare(world, monkeypatch, FEAS, atropos=False)
    W.XW.fake_checks(monkeypatch)
    head = dict(commit='c'*40)
    monkeypatch.setattr(hc, 'code_identity', lambda repo=None, run=None: dict(root=str(hb.REPO), commit=head['commit'], clean=True, status='', untracked=[]))
    monkeypatch.setattr(la, 'detached_problem', lambda root, run=None: None)               # the tests' checkout is a worktree on a branch; production's is detached (tested on its own)
    monkeypatch.setattr(fc, 'prepare', fake_prepare)
    monkeypatch.setattr(he, 'frozen_identity', lambda *a, **k: (dict(test=True), []))
    S._reset_cache()
    pin, _ = R.admitted_images(L.ctx_of(FEAS, FEAS_PAIR))
    monkeypatch.setattr(W, 'grid', lambda shape=None: pin.grid)
    assert hm.queue(pair_set=FEAS, system=LpbaSystem(FEAS, [dict(line=he.HX), dict(line=he.FALLBACK_ID)]), echo=quiet) == 0
    assert review.queue(pair_set=FEAS, system=LpbaSystem(FEAS, [dict(), dict()]), echo=quiet) == 0
    assert hb_store.main(['lpba-input-lock', '--pair-set', FEAS]) == 0
    assert (la.root(FEAS)/'hb_input_lock.json').is_file() and all((hb.cell(p, FEAS)/'t88'/he.FIELD_FILE).exists() for p in (FEAS_PAIR, 'LPBA40_S02__LPBA40_S01'))
    with pytest.raises(SystemExit):
        hb_store.main(['lpba-delete-fields', '--pair-set', FEAS, '--lock', 'runs/lpba40_feas_v1/hb_input_lock.json'])
    configc = ConfigcSystem(FEAS, [])
    assert C.queue(FEAS, system=configc, echo=quiet) == 0
    assert [a for argv in configc.runs for a in argv if a.startswith('--unit=')] == ['--unit=tag-lp-feas-c-S01-S02', '--unit=tag-lp-feas-c-S02-S01']
    r = he.read_json(C.cell_paths(FEAS, FEAS_PAIR)['result'])
    assert r['status'] == 'ok' and r['code']['commit'] == 'c'*40 and sorted(C.cell_paths(FEAS, FEAS_PAIR)['record'].glob('opened_files*.json'))
    xb = XbSystem()
    assert xb_chain.queue(FEAS, ['fireants_greedy'], system=xb, log=quiet) == 0
    doc = json.loads(W.XW.xc.cell_paths('fireants_greedy', FEAS, FEAS_PAIR)['result'].read_text())
    assert doc['status'] == 'ok' and doc['lock']['line']['code_commit'] == 'c'*40 and all('--pinned' in argv and argv[argv.index('--pinned')+1] == 'c'*40 for argv in xb.runs)
    head['commit'] = 'd'*40                                                                  # a committed input lock would move HEAD: every later start is refused
    record, problems = hm.lpba_start_checks(FEAS)
    assert problems and 'HB locks name' in ' '.join(problems)


# ---- I3: the allow-list spy ------------------------------------------------------------------------------------------------------------------------
def test_i3_the_arm_policy_judges_the_data_and_results_roots_only(staged, tmp_path):
    ctx = L.ctx_of()
    policy = R.lpba_policy_of(ctx, 't88')
    r = lc.roots(FORMAL)
    out = Path(ctx['out'])
    allowed = [(r.stage/PAIR/'pair.npz', False), (r.pred_root/'S05'/'probs36.npz', False), (r.atropos/'S06'/'labels.nii.gz', False), (out/'t88'/'run.json', True),
               (out/'fallback_configc'/'run'/'task'/'labels.json', True), (Path(R.core.ROOT)/'experiments'/'he'/'run_hx.py', False), (Path('/tmp')/'x.bin', True),
               (Path(R.core.ROOT)/'jd_curl'/'__pycache__'/'x.nbi', True)]
    forbidden = [(r.stage/'LPBA40_S07__LPBA40_S08'/'pair.npz', False), (r.root/'eval'/'x.json', False), (r.pred_root/'S07'/'probs36.npz', False), (Path('/mnt/d/data/LPBA40/x'), False),
                 (Path('/tmp')/'S05_label.nii.gz', False), (Path(R.core.ROOT)/'experiments'/'x.json', True), (Path(hc.RESULTS)/'holdout_B_v1'/'x.json', False)]
    for path, write in allowed:
        assert policy.classify(path, write)[0] == 'allowed', path
    for path, write in forbidden:
        assert policy.classify(path, write)[0] == 'forbidden', path
    assert policy.classify_listing(r.root)[0] == 'forbidden' and policy.classify_listing(out)[0] == 'allowed'
    assert la.SPY_BREACH_EXIT == review.SPY_BREACH_EXIT


def test_i3_a_child_breach_stops_the_pair(staged, monkeypatch):
    import fast_channels
    ctx = L.ctx_of()
    out = Path(ctx['out'])
    out.mkdir(parents=True)
    secret = lc.roots(FORMAL).root/'eval'/'secret.json'
    secret.parent.mkdir(parents=True)
    secret.write_text('{}')

    def peek(pin, **kwargs):
        secret.read_text()                                                               # a read outside the allow list
    monkeypatch.setattr(fast_channels, 'read_predicted', peek)
    monkeypatch.setattr(he, 'segmenter_start_problems', lambda s, registry=None: ([], dict(test=True)))
    S._reset_cache()
    assert R.t88_main(ctx) == la.SPY_BREACH_EXIT
    breach = he.read_json(out/'t88'/'spy_breach_0.json')
    assert breach['stage'] == 't88' and breach['breaches'][0]['resolved'] == str(secret.resolve()) and he.read_json(out/'t88'/'opened_files_t88_0.json')['passed'] is False
    with pytest.raises(R.LpbaSpyBreach):
        R.child_breach(out/'t88')
    assert R.supervisor_breach(ctx, breach, t0=0.) == he.STOP_QUEUE_EXIT
    line = he.read_json(out/'line.json')
    assert line['line_status'] == he.IMPLEMENTATION and line['stop_queue'] is True and line['implementation_error']['spy_breach']['stage'] == 't88'


def test_i3_the_queue_stops_on_a_reported_breach_whatever_line_json_says(world, monkeypatch):
    L.prepare(world, monkeypatch, FEAS, atropos=False)

    class Breached(LpbaSystem):
        def run(self, argv, log_path):
            rc = super().run(argv, log_path)
            pair = argv[argv.index('--pair')+1]
            he.write_json(hb.cell(pair, FEAS)/'repair'/'spy_breach_0.json', dict(stage='repair'))
            return rc
    assert hm.queue(pair_set=FEAS, system=Breached(FEAS, [dict(line=he.HX)]), echo=quiet, code_run=code_run()) == hb.EXIT['stop']
    step = he.read_json(hb.record_dir('method', FEAS_PAIR, FEAS)/'step4.json')
    assert step['went_on'] is False and step['spy_breach']


def test_i3_the_hold_out_processes_install_no_spy(tmp_path, monkeypatch):
    monkeypatch.setattr(R.core, 'frozen_api', lambda: SimpleNamespace(PreoptConfig=lambda **k: dict(k)))
    monkeypatch.setattr(R, 'lpba_spied', lambda *a, **k: (_ for _ in ()).throw(AssertionError('a hold-out process is not spied by LP-B')))
    pair, ctx, _ = W.holdout_ctx(tmp_path, monkeypatch, unseal=False)
    Path(ctx['out']).mkdir(parents=True)
    assert R.t88_main(ctx) == 0 and he.read_json(Path(ctx['out'])/'t88'/'run.json')['status'] == he.T88_REJECTED
    assert not list(Path(ctx['out']).rglob('opened_files_*'))


def test_i3_a_breach_of_the_configuration_c_driver_is_infra_failure_and_stops(configc, monkeypatch):
    secret = la.root(FORMAL)/'eval'/'secret.json'
    secret.parent.mkdir(parents=True)
    secret.write_text('{}')

    def peeking_prepare(*args, **kwargs):
        secret.read_text()
    monkeypatch.setattr(fc, 'prepare', peeking_prepare)
    S._reset_cache()
    assert C.run(FORMAL, PAIR, batch=configc, launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(), kernel=kernel()) == C.EXIT['infra_failure']
    paths = C.cell_paths(FORMAL, PAIR)
    r = he.read_json(paths['result'])
    assert (r['status'], r['kind']) == ('infra_failure', 'spy_breach') and sorted(paths['record'].glob('spy_breach*.json'))


# ---- I4, minors ---------------------------------------------------------------------------------------------------------------------------------
def test_i4_and_minor1_the_review_policy_keeps_the_code_tree_and_refuses_label_nifti(staged):
    policy = review.lpba_policy(FORMAL)
    assert set(review._paths([hb.REPO])) <= set(policy.keep) and policy.label_name == review.LPBA_LABEL_NAME
    for denied in ('/x/S05/delineation/label.nii.gz', str(la.root(FORMAL)/'stage'/'_nifti_mm'/'S05_label.nii.gz')):
        assert policy.why(denied) and 'label' in policy.why(denied)
    assert policy.why(hb.REPO/'scripts'/'check_labels_stage_3d_e10.py') is None and policy.why(hb.REPO/'experiments'/'he'/'run_hx.py') is None
    assert policy.why(hb.REPO/'data'/'three_d'/'x') is not None                              # the forbidden roots before keep
    assert review.holdout_policy().label_name == '' and review.holdout_policy().keep == ()


def test_minor3_a_name_that_is_not_plain_is_an_admission_refusal(staged):
    with pytest.raises(hold.HoldoutAdmissionRefused):
        hold.lpba40_pre_manifest(FORMAL, 'LPBA40_S05__LPBA40_S06/../x')
    with pytest.raises(hold.HoldoutAdmissionRefused):
        hold.lpba40_pre_manifest(FORMAL, 'work')


def test_minor5_the_feasibility_code_record_wants_a_detached_checkout_that_is_not_main(world, monkeypatch):
    record, problems = la.code_record(FEAS, run=L.detached_git('c'*40))
    assert problems == [] and record['detached'] is True and record['commit'] == 'c'*40
    record, problems = la.code_record(FEAS, run=W.XW.fake_git(commit='c'*40))                # on a branch
    assert problems and 'branch' in problems[0]
    assert la.code_record(FEAS, 'd'*40, run=L.detached_git('c'*40))[1]                         # the chain's commit is not this unit's HEAD
    assert la.detached_problem(hc.MAIN_REPO, run=L.detached_git()) and 'main checkout' in la.detached_problem(hc.MAIN_REPO, run=L.detached_git())


def test_minor2_the_lpba_chains_batch_records_the_lpba_preregistration(world):
    pre = xb_chain.batch_record(FEAS, ['fireants_greedy'])['preregistration']
    assert pre['path'] == lc.PREREG_REL and pre['pinned_sha256'] == lc.PREREG_SHA256 and pre['problem'] is None
    assert xb_chain.batch_record('holdout', ['fireants_greedy'])['preregistration']['path'] == W.XW.xc.PREREG_REL
    argv = xb_chain.unit_argv('fireants_greedy', FEAS, FEAS_PAIR, pinned='c'*40)
    assert argv[argv.index('--pinned')+1] == 'c'*40 and '--pinned' not in xb_chain.unit_argv('fireants_greedy', 'dev', W.XW.DEV_PAIR, pinned='c'*40)


# ---- diff review (0/1/4; decision_log 2026-11-19 ~3:51 PM CST) -----------------------------------------------------------------------------------
def test_i3a_d_and_the_code_trees_records_are_forbidden_and_code_is_read(staged):
    """Diff review I3-a: the whole D: is a forbidden root; in a code tree only code (hp_core.CODE_SUFFIXES, a __pycache__) and CODE_CONFIGS escape the name rule, its archive/ and
    docs/bayesreg/reviews/ (the frozen tree's results/, docs/, figures/ too) are forbidden; every open but a read of a .py is listed."""
    ctx = L.ctx_of()
    tree, frozen = Path(R.core.ROOT), Path(ctx['paths']['frozen'])
    configc_paths = C.cell_paths(FORMAL, PAIR)
    for policy in (R.lpba_policy_of(ctx, 't88'), R.lpba_policy_of(ctx, 'repair'), C.lpba_policy_of(FORMAL, PAIR, configc_paths, configc_paths['batches']/'b')):
        for path in (Path('/mnt/d/zz/final.npz'), Path('/mnt/d/seq_vp_3d_results/x/disp.npz'), Path('/mnt/d/seq_vp_tag_holdout_B_v1/x.npz'), frozen/'results'/'zz'/'metrics.json',
                     frozen/'results'/'zz'/'final.npz', frozen/'docs'/'x.md', frozen/'figures'/'x.png', tree/'archive'/'zz'/'evaluation.json', tree/'archive'/'zz'/'final.npz',
                     tree/'docs'/'bayesreg'/'reviews'/'x.md', hc.MAIN_REPO/'archive'/'x.json', tree/'configs'/'x_metrics.json', tree/'runs'/'x'/'per_label_dice.csv', tree/'data'/'x'):
            assert policy.classify(path, False)[0] == 'forbidden', (policy.name, path)
        for path in (Path('/mnt/d'), frozen/'results', tree/'archive'):
            assert policy.classify_listing(path)[0] == 'forbidden', (policy.name, path)
        assert policy.classify(frozen/'jd_curl'/'three_d'/'feasibility.py', False) == ('allowed', 'code')
        assert policy.classify(tree/'experiments'/'he'/'run_hx.py', False) == ('allowed', 'code')
        assert policy.classify(tree/'configs'/'frozen'/'lpba40_test_v1_arm_table.json', False) == ('allowed', 'code tree file (listed)')
        assert policy.classify(frozen/'jd_curl'/'__pycache__'/'x.label-1.py310.nbi', False) == ('allowed', 'code (listed)')
        assert policy.classify(tree/'jd_curl'/'__pycache__'/'x.nbi', True) == ('allowed', 'code cache')
        assert policy.classify(tree/'experiments'/'x.json', True)[0] == 'forbidden'
    spy = la.ArmSpy(R.lpba_policy_of(ctx, 't88'))
    spy.see_open((str(tree/'experiments'/'he'/'run_hx.py'), 'r', 0))
    spy.see_open((str(tree/'configs'/'frozen'/'lpba40_test_v1_arm_table.json'), 'r', 0))
    with pytest.raises(la.lc.hp.Breach):
        spy.see_open(('/mnt/d/zz/final.npz', 'rb', 0))
    assert [(Path(e['path']).name, e['cls']) for e in spy.events] == [('lpba40_test_v1_arm_table.json', 'allowed'), ('final.npz', 'forbidden')]     # the .py read counted, not listed
    assert spy.n_opens == 3 and len(spy.breaches) == 1


def test_i3a_the_t88_read_chain_runs_under_the_spy_without_a_breach(world, monkeypatch, record_property):
    """Diff review I3-a (the reviewer's run, again): the t88 child's real read chain on the synthetic world (LP-A's stage, Atropos, HI) -- the admission, the segmenter anchor, read_predicted,
    the colour table and prepare_stacks -- under the tightened ArmPolicy, stopped where the T88 arithmetic would start: no breach; the inputs read are the allowed ones."""
    import bayesreg.seg.structure_layer as sl
    registry = L.prepare(world, monkeypatch, FORMAL)
    reg_repo = L.git_repo(world.tmp/'registry_tree')                                     # the synthetic registry, committed and clean, stands in for the contract's (a repository registry)
    registry_path = reg_repo/'segmenters_test.json'
    registry_path.write_bytes(Path(registry.path).read_bytes())
    L.commit(reg_repo, registry_path.name)
    monkeypatch.setattr(sl, 'FORMAL_REGISTRIES', sl.FORMAL_REGISTRIES+(registry_path,))
    ctx = L.ctx_of()
    ctx['paths'] = dict(ctx['paths'], registry=str(registry_path), models_root=str(world.tmp/'members'))     # the reader pointed at the synthetic registry and members (HI's)
    monkeypatch.setattr(he, 'segmenter_start_problems', lambda s, registry=None: ([], dict(test='the synthetic registry is not the contract anchor')))
    monkeypatch.setattr(he, 'channel_source_problems', lambda sources: [])           # the anchor after the channels (the same synthetic registry)
    mi = R.core.make_inputs()
    real_qc = mi.alignment_qc
    monkeypatch.setattr(mi, 'alignment_qc', lambda u_wm, a_wm, **k: dict(real_qc(u_wm, a_wm, **k), passed=True, test='the synthetic images: the QC computed, passed by the test'))
    out = Path(ctx['out'])
    out.mkdir(parents=True)
    reached = []

    def stop(*args, **kwargs):
        reached.append(True)
        raise SystemExit('test: stopped where the T88 arithmetic starts (after the channels and the colour table)')
    import fl_channels
    monkeypatch.setattr(fl_channels, 'prepare_stacks', stop)                             # the synthetic labels have no cerebral WM to colour; nothing is read after this point
    S._reset_cache()
    assert R.t88_main(ctx) == 0
    run = he.read_json(out/'t88'/'run.json')
    opened = he.read_json(out/'t88'/'opened_files_t88_0.json')
    assert opened['passed'] is True and opened['breaches'] == [] and not list(out.glob('**/spy_breach*.json')), (run.get('reason'), opened['breaches'])
    assert reached == [True] and 'after the channels' in run['reason'], (run['status'], run.get('reason'))
    r = lc.roots(FORMAL)
    read = {e['resolved'] for e in opened['events'] if not e['write']}
    assert {str((r.stage/PAIR/'pair.npz').resolve()), str((r.stage/PAIR/'manifest.json').resolve())} <= read
    for subject in ('S05', 'S06'):
        assert any(p.startswith(str((r.pred_root/subject).resolve())+'/') for p in read) and any(p.startswith(str((r.atropos/subject).resolve())+'/') for p in read)
    assert all(e['cls'] == 'allowed' for e in opened['events']) and opened['n_opens'] >= len(opened['events'])
    whys = {}
    for e in opened['events']:
        whys[f"{e['why']}{' (write)' if e['write'] else ''}"] = whys.get(f"{e['why']}{' (write)' if e['write'] else ''}", 0)+1
    record_property('t88_spy', json.dumps(dict(n_opens=opened['n_opens'], listed=len(opened['events']), breaches=len(opened['breaches']), by_why=whys,
                                               code_listed=sorted({e['resolved'] for e in opened['events'] if e['why'].startswith('code')}),
                                               pair_members=sorted({m['member'] for m in opened['members'] if m['file'].endswith('pair.npz')}))))     # shown with --junitxml


def test_minor1_and_3_a_breach_keeps_the_method_cells_files_and_takes_no_resume(world, monkeypatch):
    """Diff review minor 1: the method queue reads the breach records before certified_check, so an hx_certified cell with a breach keeps repair/certified.npz; minor 3: its resume is refused."""
    L.prepare(world, monkeypatch, FEAS, atropos=False)

    class Breached(LpbaSystem):
        def run(self, argv, log_path):
            rc = super().run(argv, log_path)
            he.write_json(hb.cell(argv[argv.index('--pair')+1], FEAS)/'repair'/'spy_breach_0.json', dict(stage='repair'))
            return rc
    assert hm.queue(pair_set=FEAS, system=Breached(FEAS, [dict(line=he.HX)]), echo=quiet, code_run=code_run()) == hb.EXIT['stop']
    step = he.read_json(hb.record_dir('method', FEAS_PAIR, FEAS)/'step4.json')
    assert step['certified']['kept'] is True and (hb.cell(FEAS_PAIR, FEAS)/'repair'/'certified.npz').is_file() and not (hb.record_dir('method', FEAS_PAIR, FEAS)/'certified_deleted.json').exists()
    with pytest.raises(hc.Refused, match='spy breach is not rerun'):
        hm.load_resume(FEAS_PAIR, 'splits/x.json', FEAS)
    assert 'spy breach is not rerun' in review.lpba_spy_stop_problem(FEAS_PAIR, FEAS) and review.lpba_spy_stop_problem(FEAS_PAIR, 'holdout') is None


def test_minor3_a_review_stop_by_a_spy_takes_no_redo(staged):
    assert review.lpba_spy_stop_problem(PAIR, FORMAL) is None
    home = review.home_records(PAIR, hb.review_root(FORMAL))
    home.mkdir(parents=True)
    he.write_json(home/'review_not_finalized.json', dict(seq=1, kind='third_spy', reason='test'))
    assert 'name a spy' in review.lpba_spy_stop_problem(PAIR, FORMAL)


def test_minor1_and_3_a_swallowed_breach_of_configc_keeps_the_intermediates_and_takes_no_rerun(configc, monkeypatch):
    """Diff review minor 1: a breach after the lock that some code swallowed: infra_failure (spy_breach, the driver's status superseded), the intermediate commits kept; minor 3: no rerun, no
    resume."""
    secret = la.root(FORMAL)/'eval'/'secret.json'
    secret.parent.mkdir(parents=True)
    secret.write_text('{}')
    real_check = fc.firewall_check

    def swallowing_check(*args, **kwargs):
        try:
            secret.read_text()
        except Exception:                                                              # noqa: BLE001 - a library that swallows the spy's exception
            pass
        return real_check(*args, **kwargs)
    monkeypatch.setattr(fc, 'firewall_check', swallowing_check)
    S._reset_cache()
    assert C.run(FORMAL, PAIR, batch=configc, launcher=Launcher('ok'), certify=W.Certify(), probe=probe_of(), kernel=kernel()) == C.EXIT['infra_failure']
    paths = C.cell_paths(FORMAL, PAIR)
    r = he.read_json(paths['result'])
    assert (r['status'], r['kind']) == ('infra_failure', 'spy_breach') and r['superseded'] == dict(status='ok', kind='ok') and r['intermediates']['kept'] is True
    task = fc.task_dir(paths['work']/'run')
    assert sorted(p.name for p in task.glob('committed_*.npz'))+sorted(p.name for p in task.glob('level_*.npz')) == ['committed_0001.npz', 'level_2_commit.npz']
    assert sorted(paths['record'].glob('spy_breach*.json')) and not (paths['record']/'intermediates.tsv').exists()
    with pytest.raises(hc.Refused, match='spy breach is not rerun'):
        C.rerun_authorization(FORMAL, PAIR)
    with pytest.raises(hc.Refused, match='spy breach is not rerun'):
        C.load_resume(FORMAL, PAIR, 'splits/x.json')
    assert C.spy_breach_problem('holdout', PAIR) is None


def test_minor3_the_configc_driver_writes_only_its_own_batch(staged):
    paths = C.cell_paths(FORMAL, PAIR)
    batch = paths['batches']/'hb_configc_1'
    policy = C.lpba_policy_of(FORMAL, PAIR, paths, batch)
    assert policy.classify(batch/'refusals'/'x.json', True) == ('allowed', 'own')
    assert policy.classify(paths['batches']/'hb_configc_0'/'refusals'/'x.json', True)[0] == 'forbidden'
    assert policy.classify(paths['batches']/'hb_configc_0'/'refusals'/'x.json', False) == ('allowed', 'input')     # the earlier refusal records: read (their numbers, the rerun check)
    assert policy.classify(paths['batches']/'hb_method_0'/'batch.json', True)[0] == 'forbidden'


def test_minor2_the_checkout_under_the_formal_root_is_code_for_both_modes(staged, monkeypatch):
    """Diff review minor 2 (C26.9-5): the detached checkout of both modes is <formal root>/code; the review policy (both modes) and the arm policy (both modes) read it as code; its data/,
    archive/ and reviews stay forbidden; the rest of the formal root stays forbidden to the feasibility mode."""
    formal = Path(la.root(FORMAL))
    code = formal/'code'
    L.prepare(staged, monkeypatch, FEAS, hi=False, atropos=False)
    ctxs = {FORMAL: L.ctx_of(FORMAL, PAIR), FEAS: L.ctx_of(FEAS, FEAS_PAIR)}                     # made from this checkout; the policies below see the checkout at <formal root>/code
    monkeypatch.setattr(hb, 'REPO', code)
    monkeypatch.setattr(R.core, 'ROOT', code)
    for pair_set in (FORMAL, FEAS):
        policy = review.lpba_policy(pair_set)
        for path in (code/'experiments'/'hb'/'hb_review.py', code/'splits'/'lpba40_test_v1'/'test_pairs.tsv', code/'configs'/'frozen'/'lpba40_test_v1_arm_table.json'):
            assert policy.why(path) is None, (pair_set, path)
        for path in (code/'data'/'three_d'/'x', code/'docs'/'bayesreg'/'reviews'/'x.md'):
            assert policy.why(path), (pair_set, path)
    feas_policy = review.lpba_policy(FEAS)
    assert feas_policy.why(formal/'work'/'method'/PAIR/'output'/'final.npz') and feas_policy.why(formal/'stage'/PAIR/'pair.npz') and feas_policy.why(formal/'codex'/'x.py')
    for pair_set, pair in ((FORMAL, PAIR), (FEAS, FEAS_PAIR)):
        policy = R.lpba_policy_of(ctxs[pair_set], 't88')
        assert policy.classify(code/'experiments'/'he'/'run_hx.py', False) == ('allowed', 'code')
        assert policy.classify(code/'splits'/'lpba40_test_v1'/'test_pairs.tsv', False) == ('allowed', 'code tree file (listed)')
        for path in (code/'data'/'x', code/'archive'/'x.py', formal/'work'/'configc'/PAIR/'output'/'disp.npz'):
            assert policy.classify(path, False)[0] == 'forbidden', (pair_set, path)
        cpaths = C.cell_paths(pair_set, pair)
        assert C.lpba_policy_of(pair_set, pair, cpaths, cpaths['batches']/'b').classify(code/'experiments'/'hb'/'hb_configc.py', False) == ('allowed', 'code')


# ---- LP-C2 integration item 7 (LP-B quick diff review minors 1, 2; decision_log 2026-11-19, the LP-B merge ~4:22 PM CST) ------------------------------------------------------------
def test_integration7_the_review_policy_forbids_archive_and_reviews_of_both_trees(staged):
    """Minor 1: the review unit's lpba_policy lists the code tree's and the main checkout's archive/ and docs/bayesreg/reviews/ as forbidden roots (lpba_arms.CODE_TREE_FORBIDDEN), as
    ArmPolicy and XB's LPBA driver policy do; the code itself stays readable; the hold-out's policy has no such root (unchanged)."""
    tree = Path(hb.REPO)
    for pair_set in (FORMAL, FEAS):
        policy = review.lpba_policy(pair_set)
        for path in (tree/'archive'/'x.py', tree/'archive'/'seq_vp_3d'/'results'/'x.json', tree/'docs'/'bayesreg'/'reviews'/'x.md', hc.MAIN_REPO/'archive'/'x.py',
                     hc.MAIN_REPO/'docs'/'bayesreg'/'reviews'/'x.md'):
            assert policy.why(path) and 'forbidden root' in policy.why(path), (pair_set, path)
        assert policy.why(tree/'experiments'/'hb'/'hb_review.py') is None
    assert la.CODE_TREE_FORBIDDEN == ('archive', 'docs/bayesreg/reviews')
    assert not [d for d in review.holdout_policy().deny if Path(d).name == 'archive']


def test_integration7_the_arm_policy_forbids_the_main_checkout_and_result_documents(staged, tmp_path):
    """Minor 2: ArmPolicy forbids the whole main checkout but the code trees inside it (an LPBA process runs from <formal root>/code, C26.9-5), and a result document by its name
    (result*.md, hb_review.RESULT_DOC_NAME's rule) wherever the name rule applies; the run's own files keep no name rule."""
    code = tmp_path/'formal'/'code'
    own = tmp_path/'own'
    policy = la.ArmPolicy(name='integration 7', code_roots=[code], write_inside=[own], results=[tmp_path/'results'])
    for path in (hc.MAIN_REPO/'experiments'/'he'/'run_hx.py', hc.MAIN_REPO/'splits'/'lpba40_test_v1'/'test_pairs.tsv', hc.MAIN_REPO/'README.md'):
        assert policy.classify(path, False) == ('forbidden', 'the main checkout outside the code trees (an LPBA process runs from <formal root>/code, C26.9-5)'), path
    assert policy.classify_listing(hc.MAIN_REPO/'runs')[0] == 'forbidden'
    assert policy.classify(code/'experiments'/'he'/'run_hx.py', False) == ('allowed', 'code')
    for path in (code/'runs'/'20261119-hb-result'/'result.md', code/'docs'/'result_lpba.md', tmp_path/'elsewhere'/'result.md'):
        assert policy.classify(path, False)[0] == 'forbidden', path
    assert policy.classify(code/'runs'/'x'/'notes_result.md', False) == ('allowed', 'code tree file (listed)')       # only a name that begins with result
    assert policy.classify(own/'result.md', True) == ('allowed', 'own')
    inside = la.ArmPolicy(name='integration 7, a worktree', code_roots=[hc.MAIN_REPO/'wt'/'worktrees'/'w'])
    assert inside.classify(hc.MAIN_REPO/'wt'/'worktrees'/'w'/'experiments'/'he'/'run_hx.py', False) == ('allowed', 'code')
    assert inside.classify(hc.MAIN_REPO/'experiments'/'he'/'run_hx.py', False)[0] == 'forbidden'
    import hb_review
    assert la.RESULT_DOC_NAME.pattern == hb_review.RESULT_DOC_NAME.pattern
    assert policy.document()['main_checkout'] and policy.document()['result_doc_name'] == la.RESULT_DOC_NAME.pattern
