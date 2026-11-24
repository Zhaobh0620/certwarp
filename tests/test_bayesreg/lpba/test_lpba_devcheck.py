"""LP-D (runs/20261119-lpba/任务书_D_开发集逐位对照.md; contract C26.5-11 (a)(b)(c), C26.9-1; prereg §12 item 4): the three development bitwise checks on synthetic worlds only.
(a) on LP-A's synthetic LPBA world (test_lpba_entries.world: a temporary code tree pinned into bayesreg, the LPBA allowlist, tiny LPBA40-like NIfTI files) with a synthetic "old stage" written the
way paper 1's prepare_stage_3d_lpba40.py wrote it (label members and label_sets included, so that the tests can show they are never asked); (b) on a synthetic development stage (synthetic OASIS
ids, synthetic aligned_norm, the stage written as paper 1 wrote it: sources.<side>.sha256 the NIfTI file's) with tiny CPU members and "registered" predictions made by hs_a from the NIfTI files;
(c) on synthetic run records and fields.  No real image, label, stage, prediction or result root is touched (holdout_common's test sandbox is set; every root monkeypatched into tmp_path)."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import types

import numpy as np
import pytest

import bayesreg.io.sealed as S
from bayesreg.io.guard import load_dev_allowlist
from bayesreg.io.pair import load_segmenter_registry, subject_list_sha256
from bayesreg.seg.unet import synthetic
from jd_curl.three_d.nifti import file_sha256, load_scalar

from ..sg import _sg_world as W
from .test_lpba_entries import CODE, PINNED, run_prep, world  # noqa: F401  (world: LP-A's synthetic LPBA world, a fixture)

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT/'experiments'/'lpba'/'devcheck', ROOT/'experiments'/'lpba', ROOT/'experiments'/'hi', ROOT/'experiments'/'holdout', ROOT/'experiments'/'he'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import lpdev_common as dc  # noqa: E402
import lpdev_a  # noqa: E402
import lpdev_b  # noqa: E402
import lpdev_c  # noqa: E402
import hi_lpba  # noqa: E402

lc, hc, hp = dc.lc, dc.hc, dc.hp
hi = hi_lpba.hi
he = lpdev_c.he


def quiet(*args, **kwargs):
    pass


# ==== (a) ===========================================================================================================================================
LABEL_MEMBERS = ('fixed_lpba56', 'moving_lpba56', 'fixed_brain', 'moving_brain')


def write_old_stage(root, stage, *, transform=None):
    """A synthetic stage_LPBA40_v1/LPBA40_S01__LPBA40_S02 written as prepare_stage_3d_lpba40.py wrote it: the mm copy read by load_scalar (no flip), / its p99.5 of v > 0, clipped to [0, 1];
    the label members, the brain members and an all-zero initial_u; a manifest with label_sets and baseline_label_metrics.  `transform(side, v, p)` replaces the stored image (counter-examples)."""
    out = Path(root)/'LPBA40_S01__LPBA40_S02'
    out.mkdir(parents=True)
    arrays, sources = {}, {}
    for side, subject in (('fixed', 'S01'), ('moving', 'S02')):
        copy = Path(stage)/lc.MM_DIR/f'{subject}_image.nii.gz'
        vol = load_scalar(copy)
        v = np.asarray(vol.values, dtype=np.float64)
        p = float(np.percentile(v[v > 0], 99.5))
        arrays[side] = np.clip(v/p, 0., 1.) if transform is None else transform(side, v, p)
        arrays[f'{side}_lpba56'] = (v > np.median(v[v > 0])).astype(np.float64)*21
        arrays[f'{side}_brain'] = (v > 0).astype(np.float64)
        sources[side] = dict(vol.metadata)
        sources[f'{side}_lpba56'] = dict(source=str(copy).replace('image', 'label'), sha256='e'*64)
        grid = vol.grid
    arrays['initial_u'] = np.zeros(grid.shape+(3,))
    np.savez_compressed(out/'pair.npz', **arrays)
    manifest = dict(dataset='LPBA40 (synthetic)', pair=out.name, role='tune', fixed_subject='S01', moving_subject='S02', shape_zyx=list(grid.shape), spacing_xyz=list(grid.spacing),
                    origin_xyz=[float(a[0]) for a in grid.axes], unit='mm', sources=sources, pair_sha256=file_sha256(out/'pair.npz'), baseline_label_metrics=dict(lpba56=dict(macro_dice=.5)),
                    label_sets=dict(lpba56=[21], brain=[1]), label_primary='lpba56')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    return out


@pytest.fixture
def a_world(world, monkeypatch):
    run_prep()
    old = world.tmp/'old_stage'
    monkeypatch.setattr(lpdev_a, 'OLD_STAGE', old)
    S._reset_cache()
    return types.SimpleNamespace(world=world, old=old, stage=lc.roots('lpba40_feas').stage)


def run_a(**kwargs):
    S._reset_cache()
    return lpdev_a.run(code=CODE, log=quiet, **kwargs)


def test_a_passes_on_paper1s_way_and_reads_only_fixed_moving(a_world):
    write_old_stage(a_world.old, a_world.stage)
    record = run_a()
    assert record['passed'] and record['conclusion'] == 'passed' and record['spy']['passed']
    c = record['comparison']
    for side in ('fixed', 'moving'):
        s = c['sides'][side]
        assert s['passed'] and all(s['checks'].values()) and s['old']['bitwise_equal'] and s['p5']['bitwise_equal']
        assert s['recomputed_p5_sha256'] == s['prep_input_sha256'] == s['manifest_source_sha256']
        assert s['p']['hex'] == s['prep_p']['hex']
    assert c['old_keys_read'] == ['fixed', 'moving', 'initial_u']                    # load_pair's whitelist; no label member
    asked = {(Path(m['file']).parent.name, m['member']) for m in record['spy']['members']}
    assert not [m for m in asked if m[1] in LABEL_MEMBERS]
    assert ('LPBA40_S01__LPBA40_S02', 'fixed') in asked
    opened = [e['resolved'] for e in record['spy']['events']]
    old_opened = sorted({p for p in opened if p.startswith(str(a_world.old))})
    assert old_opened == sorted(str(p) for p in lpdev_a.old_paths())                 # of the old stage, the one pair's two files only
    assert not [p for p in opened if 'label' in Path(p).name]
    assert not [p for p in opened if p.startswith(str(a_world.world.raw))]           # no LPBA40 original: v_new from the stage's mm copy
    saved = json.loads((dc.roots().a/lpdev_a.RECORD).read_text(encoding='utf-8'))
    assert saved['passed'] and saved['code']['commit'] == CODE['commit'] and {f['path'] for f in c['files_read']} >= {str(p) for p in lpdev_a.old_paths()}
    assert all(len(f['sha256']) == 64 for f in c['files_read'])


@pytest.mark.parametrize('transform, what', [
    (lambda side, v, p: np.clip(np.flip(v, 2)/p, 0., 1.), 'flipped'),
    (lambda side, v, p: np.clip(v/(p*(1+1e-6)), 0., 1.), 'another p'),
    (lambda side, v, p: v/p if side == 'moving' else np.clip(v/p, 0., 1.), 'not clipped'),
])
def test_a_does_not_pass_when_the_old_array_is_not_np_clip_np_flip(a_world, transform, what):
    write_old_stage(a_world.old, a_world.stage, transform=transform)
    record = run_a()
    assert not record['passed'] and record['conclusion'] == 'not passed', what
    sides = record['comparison']['sides']
    assert not all(s['checks']['old_bitwise'] for s in sides.values())
    assert all(s['checks']['p5_bitwise'] and s['checks']['recomputed_sha256_is_prep_s'] for s in sides.values())
    assert any(s['old']['n_elements_differ'] > 0 for s in sides.values())


def test_a_prep_scale_is_checked_against_the_literal_formula(a_world, monkeypatch):
    """Review minor 1: prep_lpba.scale (the code under check) off the formula at the check's time: the literal p and P5 still match the stage, prep_scale_is_literal does not."""
    write_old_stage(a_world.old, a_world.stage)
    real = lpdev_a.prep.scale
    monkeypatch.setattr(lpdev_a.prep, 'scale', lambda v, c: real(v, c*(1+1e-9)))
    record = run_a()
    assert not record['passed']
    for s in record['comparison']['sides'].values():
        assert not s['checks']['prep_scale_is_literal'] and s['checks']['p5_bitwise'] and s['checks']['recomputed_sha256_is_prep_s'] and s['checks']['old_bitwise']


@pytest.mark.parametrize('deviation', ['percentile 99.4', 'c x (1 + 1e-9)'])
def test_a_a_stage_and_a_tool_both_off_the_formula_do_not_pass(world, monkeypatch, deviation):
    """Review minor 1, the lead's counter-example: the stage made with a formula other than P5 and prep_lpba.scale the same at the check -- before, a check through prep.scale would have
    passed; the literal formula of the tool does not."""
    def scale(v, c):
        brain = v[v > 0]
        p = float(np.percentile(brain, 99.4 if deviation.startswith('percentile') else 99.5))
        return v / p * (c*(1+1e-9) if deviation.startswith('c') else c), p
    monkeypatch.setattr(lpdev_a.prep, 'scale', scale)
    run_prep()
    monkeypatch.setattr(lpdev_a, 'OLD_STAGE', world.tmp/'old_stage')
    write_old_stage(world.tmp/'old_stage', lc.roots('lpba40_feas').stage)
    record = run_a()
    assert not record['passed']
    for s in record['comparison']['sides'].values():
        assert not s['checks']['p5_bitwise'] and not s['checks']['recomputed_sha256_is_prep_s'] and not s['checks']['prep_scale_is_literal']
        assert s['checks']['old_bitwise']                                                  # the literal p is paper 1's p
        assert s['checks']['p_hex_is_prep_s'] == (deviation.startswith('c'))


def test_a_runs_once_and_the_dry_run_reads_no_array(a_world, tmp_path):
    write_old_stage(a_world.old, a_world.stage)
    dry = run_a(dry_run=True, out=tmp_path/'dry')
    assert dry['spy']['passed'] and {p['exists'] for p in dry['planned']} == {True}
    assert not [e for e in dry['spy']['events'] if Path(e['resolved']).suffix in ('.npz', '.gz')] and not dc.roots().a.exists()
    assert 'tag-lp-feas-deva' in dry['launch']
    run_a()
    with pytest.raises(hc.Refused, match='output'):
        run_a()
    shutil.rmtree(dc.roots().a)
    with pytest.raises(hc.Refused, match='once'):
        run_a()


def test_a_a_failure_after_the_lock_stops_and_keeps_the_record(a_world, monkeypatch):
    write_old_stage(a_world.old, a_world.stage)

    def broken(path):
        raise RuntimeError('a synthetic failure')
    monkeypatch.setattr(lpdev_a.prep, 'stage_image', broken)
    with pytest.raises(dc.Stopped, match='synthetic failure'):
        run_a()
    stopped = json.loads((dc.roots().a/lpdev_a.STOPPED).read_text(encoding='utf-8'))
    assert stopped['passed'] is False and 'synthetic failure' in stopped['stopped'] and dc.roots().a_lock.is_file()


def test_a_a_failure_after_the_comparison_also_stops(a_world, monkeypatch):
    """Review minor 5: a step after the comparison, outside compare (here the record's own write): the stopped record is written all the same."""
    write_old_stage(a_world.old, a_world.stage)
    real = hc.write_json_once

    def failing(path, document):
        if Path(path).name == lpdev_a.RECORD:
            raise OSError('a synthetic failure of the record\'s write')
        return real(path, document)
    monkeypatch.setattr(hc, 'write_json_once', failing)
    with pytest.raises(dc.Stopped, match='OSError'):
        run_a()
    assert (dc.roots().a/lpdev_a.STOPPED).is_file() and not (dc.roots().a/lpdev_a.RECORD).exists()


def test_a_failure_of_the_stopped_record_itself_still_stops_with_both(a_world, monkeypatch):
    """LP-D diff review minor (LP-C2 integration item 6): the record's write fails and so does the stopped record's -- after_lock still leaves as Stopped, carrying both exceptions (its
    message, .original and __cause__ the first, .record_error and __context__ the second); the lock stays."""
    write_old_stage(a_world.old, a_world.stage)
    real = hc.write_json_once

    def failing(path, document):
        if Path(path).name in (lpdev_a.RECORD, lpdev_a.STOPPED):
            raise OSError(f'a synthetic failure of {Path(path).name}')
        return real(path, document)
    monkeypatch.setattr(hc, 'write_json_once', failing)
    with pytest.raises(dc.Stopped) as info:
        run_a()
    stopped = info.value
    assert f'failure of {lpdev_a.RECORD}' in str(stopped) and f'failure of {lpdev_a.STOPPED}' in str(stopped)
    assert isinstance(stopped.original, OSError) and stopped.__cause__ is stopped.original and lpdev_a.RECORD in str(stopped.original)
    assert isinstance(stopped.record_error, OSError) and stopped.__context__ is stopped.record_error and lpdev_a.STOPPED in str(stopped.record_error)
    assert not (dc.roots().a/lpdev_a.STOPPED).exists() and dc.roots().a_lock.is_file()


def test_after_lock_keeps_both_exceptions_whatever_the_first_was(tmp_path, monkeypatch):
    """Integration item 6 on the context itself: a KeyboardInterrupt whose stopped record cannot be written leaves as Stopped with both; one whose record is written leaves as itself."""
    class Spy:
        removed = 0

        def record(self):
            return dict(passed=True)

        def remove(self):
            Spy.removed += 1
    real = hc.write_json_once

    def failing(path, document):
        raise PermissionError('a synthetic refusal of the stopped record')
    monkeypatch.setattr(hc, 'write_json_once', failing)
    with pytest.raises(dc.Stopped) as info:
        with dc.after_lock({}, Spy(), tmp_path, tmp_path/'x.lock', 'x_stopped.json'):
            raise KeyboardInterrupt('a synthetic interrupt')
    assert isinstance(info.value.original, KeyboardInterrupt) and isinstance(info.value.record_error, PermissionError) and Spy.removed == 1
    assert 'KeyboardInterrupt' in str(info.value) and 'PermissionError' in str(info.value)
    monkeypatch.setattr(hc, 'write_json_once', real)
    with pytest.raises(KeyboardInterrupt):
        with dc.after_lock({}, Spy(), tmp_path, tmp_path/'x.lock', 'x_stopped.json'):
            raise KeyboardInterrupt('a synthetic interrupt')
    assert (tmp_path/'x_stopped.json').is_file()


def test_a_the_hold_and_a_missing_stage_refuse(a_world):
    write_old_stage(a_world.old, a_world.stage)
    hold = lc.roots('lpba40_feas').hold
    hold.parent.mkdir(parents=True, exist_ok=True)
    hold.write_text('')
    with pytest.raises(hc.Refused, match='hold'):
        run_a()
    hold.unlink()
    shutil.rmtree(a_world.old)
    with pytest.raises(hc.Refused, match='old pair'):
        run_a()
    assert not dc.roots().a_lock.exists()


def test_a_a_label_member_of_the_old_pair_is_a_breach(a_world):
    write_old_stage(a_world.old, a_world.stage)
    S._reset_cache()
    with pytest.raises(hp.Breach, match='fixed_lpba56'):
        with dc.spying(lpdev_a.policy()):
            with np.load(lpdev_a.old_paths()[0], allow_pickle=False) as npz:
                npz['fixed_lpba56']


def test_a_main_exit_codes(a_world, monkeypatch, tmp_path):
    write_old_stage(a_world.old, a_world.stage)
    monkeypatch.setattr(hp, 'check_code', lambda code, require_clean=True: dict(problems=[]))
    monkeypatch.setattr(dc, 'bytecode_problem', lambda: None)
    lines = []
    assert lpdev_a.main(['dry-run', '--out', str(tmp_path/'dry')], code=dict(CODE), out=lines.append) == 0
    assert json.loads((tmp_path/'dry'/'dry_run.json').read_text(encoding='utf-8'))['dry_run'] is True
    S._reset_cache()
    assert lpdev_a.main(['run'], code=dict(CODE), out=lines.append) == 0
    S._reset_cache()
    assert lpdev_a.main(['run'], code=dict(CODE), out=lines.append) == 8                     # once
    assert lpdev_a.main(['dry-run', '--out', str(tmp_path/'dry')], code=dict(CODE), out=lines.append) == 2      # the out directory is not empty


def test_a_main_not_passed_stopped_and_an_error_before_the_lock(a_world, monkeypatch, tmp_path):
    """Review minors 6, 7: 3 (ran, not passed), 9 (stopped after the lock), 8 (any exception before the lock, not a traceback's 1; a dry run writes it)."""
    write_old_stage(a_world.old, a_world.stage, transform=lambda side, v, p: np.clip(np.flip(v, 2)/p, 0., 1.))
    monkeypatch.setattr(hp, 'check_code', lambda code, require_clean=True: dict(problems=[]))
    monkeypatch.setattr(dc, 'bytecode_problem', lambda: None)
    lines = []
    S._reset_cache()
    assert lpdev_a.main(['run'], code=dict(CODE), out=lines.append) == 3
    shutil.rmtree(dc.roots().a)
    dc.roots().a_lock.unlink()                                       # a test only: the lock is never removed by the tools
    monkeypatch.setattr(lpdev_a.prep, 'stage_image', lambda path: (_ for _ in ()).throw(RuntimeError('broken')))
    S._reset_cache()
    assert lpdev_a.main(['run'], code=dict(CODE), out=lines.append) == 9
    monkeypatch.setattr(lc, 'check_hold', lambda pair_set: (_ for _ in ()).throw(RuntimeError('not a Refused')))
    S._reset_cache()
    assert lpdev_a.main(['dry-run', '--out', str(tmp_path/'dry2')], code=dict(CODE), out=lines.append) == 8
    assert 'RuntimeError' in json.loads((tmp_path/'dry2'/'dry_run.json').read_text(encoding='utf-8'))['refused']['stage']


# ==== (b) ===========================================================================================================================================
DEV_SUBJ = [synthetic.oasis(985), synthetic.oasis(986), synthetic.oasis(987)]
DEV_PAIRS = [(DEV_SUBJ[0], DEV_SUBJ[1]), (DEV_SUBJ[1], DEV_SUBJ[2])]


def write_dev_pair(stage, data, fixed, moving, *, change=None):
    """A pair directory as paper 1's OASIS stage writer made it (synthetic.write_pair's way): fixed, moving = load_scalar(aligned_norm).values, an all-zero initial_u, a seg member (never asked),
    a manifest whose sources.<side>.sha256 is the NIfTI file's.  change(side, v): a counter-example's array."""
    out = Path(stage)/hc.pair_name(fixed, moving)
    out.mkdir(parents=True)
    vols = {side: load_scalar(Path(data)/s/'aligned_norm.nii.gz') for side, s in (('fixed', fixed), ('moving', moving))}
    arrays = {side: (v.values if change is None else change(side, v.values)) for side, v in vols.items()}
    grid = vols['fixed'].grid
    np.savez_compressed(out/'pair.npz', fixed=arrays['fixed'], moving=arrays['moving'], initial_u=np.zeros(grid.shape+(3,)), fixed_seg35=np.ones(grid.shape))
    manifest = dict(fixed_subject=fixed, moving_subject=moving, shape_zyx=list(grid.shape), spacing_xyz=list(grid.spacing), origin_xyz=[float(a[0]) for a in grid.axes], unit='mm',
                    pair_sha256=file_sha256(out/'pair.npz'), sources={side: dict(sha256=v.metadata['sha256']) for side, v in vols.items()}, label_sets=dict(seg35=[1]))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    return out


@pytest.fixture
def dev_world(tmp_path, monkeypatch):
    """A synthetic development world: bayesreg's tree pinned into a temporary repository whose allowlist source stage_L2R_val_v1 holds DEV_SUBJ (on no sealed list), the training source the
    members' label contact; the results root in tmp_path; the development pairs and subjects of hp_core and the tools monkeypatched to the synthetic ones."""
    from ..hi import _world as HW
    w = W.make_world(tmp_path, monkeypatch, record=False, sources=[('stage_L2R_val_v1', DEV_SUBJ, False), ('train', HW.CONTACT, False), ('t0', W.T0, True)])
    for rel in PINNED:                                               # lpba_common.check_pins reads them from bayesreg's (pinned) tree
        (w.repo/rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, w.repo/rel)
    results = tmp_path/'results'
    results.mkdir()
    monkeypatch.setenv(hc.TEST_SANDBOX_ENV, '1')
    for module in (lc, hc):
        monkeypatch.setattr(module, 'RESULTS', results)
    monkeypatch.setattr(hc, 'DEV_PAIRS', tuple(hc.pair_name(f, m) for f, m in DEV_PAIRS))
    monkeypatch.setattr(hc, 'machine_gate', lambda **kwargs: dict(passed=True, note='test: no machine queried', **{k: v for k, v in kwargs.items() if k != 'run'}))
    monkeypatch.setattr(hc, 'disk_guard', lambda **kwargs: dict(passed=True, note='test', **kwargs))
    S._reset_cache()
    yield types.SimpleNamespace(w=w, repo=w.repo, results=results, tmp=tmp_path, HW=HW)
    S._reset_cache()


@pytest.fixture
def b_world(dev_world, monkeypatch):
    import torch
    from bayesreg.seg.unet.data import OasisFiles
    from bayesreg.seg.unet.groups import DEFAULT_GROUPS_PATH, load_grouping, require_inference_grouping
    HW, tmp = dev_world.HW, dev_world.tmp
    data = synthetic.write_subjects(tmp/'data', DEV_SUBJ, shape_xyz=HW.SHAPE_XYZ)
    stage = tmp/'stage_L2R_val_v1'
    for f, m in DEV_PAIRS:
        write_dev_pair(stage, data, f, m)
    models_root = tmp/'members'
    shas = {f'fold{f}_s{s}': HW.write_member(models_root, f, s, t, w_, g, a) for f, s, t, w_, g, a in HW.MEMBERS}
    registry = load_segmenter_registry(HW.write_json(tmp/'registry.json', dict(schema='bayesreg.segmenters.v1', comment='synthetic', models={
        f'unet35_v4_ens_fold{f}': dict(segmenter_id='unet35_v4_ens', fold=i, member_model_sha256=[shas[f'fold{f}_s0'], shas[f'fold{f}_s1']], label_contact_subjects=list(HW.CONTACT),
                                       label_contact_subjects_sha256=subject_list_sha256(HW.CONTACT)) for i, f in enumerate('AB')})))
    anchor = dict(hi.CONTRACT_SEGMENTER, registry_sha256=registry.sha256)
    real_check = hi.check_registry
    grouping = load_grouping(DEFAULT_GROUPS_PATH, 'structure13_v1')
    monkeypatch.setattr(hi, 'check_registry', lambda r, anchor_=None: real_check(r, anchor))
    monkeypatch.setattr(hi_lpba, 'inputs', lambda: (registry, grouping, Path(DEFAULT_GROUPS_PATH)))
    monkeypatch.setattr(hi, 'MODELS_ROOT', models_root)
    monkeypatch.setattr(hi, 'check_decision', lambda decision: dict(devcheck_json='/test/devcheck.json', devcheck_sha256='d'*64, note='test'))
    monkeypatch.setattr(hi, 'check_environment', lambda device, decision: hi.environment_of(device))
    # the "registered" predictions: hs_a from the NIfTI files (HS's seg phase's way), before the check
    allowlist = load_dev_allowlist(dev_world.repo/hp.ALLOWLIST_REL)
    available = hi.member_files(registry, models_root)
    reference = tmp/'reference_pred'
    device = torch.device('cpu')
    for s in DEV_SUBJ:
        plan, models, policy_name = hi.plan_of(registry, s, allowlist, available)
        hi.predict_hs_a(s, files=OasisFiles(root=data), registry=registry, models=models, plan=plan, policy=policy_name, device=device, grouping=grouping,
                        grouping_check=require_inference_grouping(grouping), pred_root=reference, code=dict(CODE), environment=hi.environment_of(device), command=['test'],
                        allowlist=allowlist, hi=dict(purpose='test: the registered predictions'))
    monkeypatch.setattr(lpdev_b, 'DEV_STAGE', stage)
    monkeypatch.setattr(lpdev_b, 'DEV_PAIRS', tuple(DEV_PAIRS))
    monkeypatch.setattr(lpdev_b, 'DEV_SUBJECTS', tuple(DEV_SUBJ))
    monkeypatch.setattr(lpdev_b, 'REFERENCE_ROOT', reference)
    S._reset_cache()
    return types.SimpleNamespace(dev=dev_world, data=data, stage=stage, reference=reference, registry=registry, allowlist=allowlist)


def run_b(**kwargs):
    S._reset_cache()
    return lpdev_b.run(code=dict(CODE), device='cpu', log=quiet, **kwargs)


def test_b_the_stage_array_path_reproduces_the_registered_predictions_bitwise(b_world):
    record = run_b()
    assert record['passed'] and record['reading']['passed'] and record['reading']['n_bitwise_equal'] == len(DEV_SUBJ), record['reading']
    for s, row in record['rows'].items():
        c = row['comparison']
        assert c['probs36']['bitwise_equal'] and c['structure_groups']['bitwise_equal'] and c['probs36']['argmax_agreement'] == 1.
        assert row['manifest_source_is_reference_image']                                # the OASIS stage names the NIfTI file the reference was made from
        assert row['image']['input_sha256'] != row['image']['manifest_source_sha256']      # the array's sha256 is not the file's (why P7's check is not asked here)
        assert c['input_sha256'] == {'aligned_norm.nii.gz': row['image']['input_sha256']}
    opened = [e['resolved'] for e in record['spy']['events']]
    assert not [p for p in opened if p.startswith(str(b_world.data))]                    # no aligned_norm, no aligned_seg35: the stage arrays only
    members = {(Path(m['file']).name, m['member']) for m in record['spy']['members']}
    assert {k for _, k in members if _ == 'pair.npz'} <= {'fixed', 'moving', 'initial_u'} and ('pair.npz', 'fixed_seg35') not in members
    assert {k for f, k in members if f in ('probs36.npz', 'structure_groups.npz')} == {'p'}
    assert record['spy']['passed'] and record['spy']['breaches'] == []
    assert {f['path'] for f in record['files_read']} >= {str(b_world.stage/hc.pair_name(*p)/'pair.npz') for p in DEV_PAIRS}
    assert (dc.roots().b/lpdev_b.RECORD).is_file() and sorted(p.name for p in (dc.roots().b/'subjects').iterdir()) == sorted(f'{s}.json' for s in DEV_SUBJ)


def test_b_a_stage_array_other_than_the_image_does_not_pass(b_world):
    shutil.rmtree(b_world.stage)
    brighter = lambda side, v: v+.5*(v > 0) if side == 'moving' else v          # noqa: E731
    for f, m in DEV_PAIRS:                                       # the last subject (moving of the second pair only) a little brighter in the stage than in its aligned_norm
        write_dev_pair(b_world.stage, b_world.data, f, m, change=brighter if m == DEV_SUBJ[2] else None)
    record = run_b()
    assert not record['passed'] and record['reading']['text'] == '不过' and record['reading']['not_equal'] == [DEV_SUBJ[2]]
    assert record['rows'][DEV_SUBJ[0]]['comparison']['bitwise_equal'] and record['rows'][DEV_SUBJ[1]]['comparison']['bitwise_equal']
    assert not record['rows'][DEV_SUBJ[2]]['comparison']['probs36']['bitwise_equal']


def test_b_a_subject_with_two_arrays_stops(b_world):
    npz = b_world.stage/hc.pair_name(*DEV_PAIRS[1])/'pair.npz'
    with np.load(npz) as z:
        arrays = {k: z[k] for k in z.files}
    arrays['fixed'] = arrays['fixed']*(1+1e-12)                    # DEV_SUBJ[1] is moving of the first pair and fixed of the second
    np.savez_compressed(npz, **arrays)
    manifest = json.loads((npz.parent/'manifest.json').read_text(encoding='utf-8'))
    manifest['pair_sha256'] = file_sha256(npz)
    (npz.parent/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(dc.Stopped, match='two different arrays'):
        run_b()
    stopped = json.loads((dc.roots().b/lpdev_b.STOPPED).read_text(encoding='utf-8'))
    assert stopped['done'] == [DEV_SUBJ[0]]


def rewrite_manifest(pair_dir, **changes):
    manifest = json.loads((pair_dir/'manifest.json').read_text(encoding='utf-8'))
    manifest.update(changes)
    manifest['pair_sha256'] = file_sha256(pair_dir/'pair.npz')
    (pair_dir/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')


def test_b_a_subject_whose_manifests_name_two_source_files_stops(b_world):
    """Review minor 7: dev_stage_array's second stop -- one array, but the two manifests name two NIfTI files for it."""
    pair_dir = b_world.stage/hc.pair_name(*DEV_PAIRS[1])
    manifest = json.loads((pair_dir/'manifest.json').read_text(encoding='utf-8'))
    rewrite_manifest(pair_dir, sources=dict(manifest['sources'], fixed=dict(sha256='a'*64)))       # DEV_SUBJ[1] is moving of the first pair and fixed of the second
    with pytest.raises(dc.Stopped, match='two source files'):
        run_b()
    assert json.loads((dc.roots().b/lpdev_b.STOPPED).read_text(encoding='utf-8'))['done'] == [DEV_SUBJ[0]]


def test_b_a_seg_key_of_a_stage_pair_is_a_breach(b_world):
    S._reset_cache()
    with pytest.raises(hp.Breach, match='fixed_seg35'):
        with dc.spying(lpdev_b.policy(list(DEV_SUBJ), list(DEV_PAIRS))):
            with np.load(b_world.stage/hc.pair_name(*DEV_PAIRS[0])/'pair.npz', allow_pickle=False) as npz:
                npz['fixed_seg35']


def test_b_main_exit_codes(b_world, monkeypatch, tmp_path):
    """Review minor 7: 0, 8 (once), 2, 3, 9."""
    monkeypatch.setattr(hp, 'check_code', lambda code, require_clean=True: dict(problems=[]))
    monkeypatch.setattr(dc, 'bytecode_problem', lambda: None)
    lines = []
    S._reset_cache()
    assert lpdev_b.main(['dry-run', '--out', str(tmp_path/'dry')], code=dict(CODE), out=lines.append) == 0
    assert lpdev_b.main(['dry-run', '--out', str(tmp_path/'dry')], code=dict(CODE), out=lines.append) == 2
    S._reset_cache()
    assert lpdev_b.main(['run', '--device', 'cpu'], code=dict(CODE), out=lines.append) == 0
    record = json.loads((dc.roots().b/lpdev_b.RECORD).read_text(encoding='utf-8'))
    assert record['pins']['prereg']['sha256'] == lc.PREREG_SHA256 and sorted(record['pred_files']) == sorted(DEV_SUBJ)
    S._reset_cache()
    assert lpdev_b.main(['run', '--device', 'cpu'], code=dict(CODE), out=lines.append) == 8
    shutil.rmtree(dc.roots().b)
    dc.roots().b_lock.unlink()                                       # a test only
    shutil.rmtree(b_world.stage)
    brighter = lambda side, v: v+.5*(v > 0) if side == 'moving' else v          # noqa: E731
    for f, m in DEV_PAIRS:
        write_dev_pair(b_world.stage, b_world.data, f, m, change=brighter if m == DEV_SUBJ[2] else None)
    S._reset_cache()
    assert lpdev_b.main(['run', '--device', 'cpu'], code=dict(CODE), out=lines.append) == 3
    shutil.rmtree(dc.roots().b)
    dc.roots().b_lock.unlink()
    monkeypatch.setattr(lpdev_b, 'compare_subject', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('broken')))
    S._reset_cache()
    assert lpdev_b.main(['run', '--device', 'cpu'], code=dict(CODE), out=lines.append) == 9


def test_b_the_metadata_is_hi_lpba_stage_arrays_on_an_lpba_stage(world):
    run_prep()
    from bayesreg.io.guard import load_dev_allowlist as load
    allowlist = load(world.allowlist_path)
    stage, pairs = lc.roots('lpba40_feas').stage, lc.table('lpba40_feas')[1]
    S._reset_cache()
    for subject in ('S01', 'S02'):
        a1, m1, r1 = hi_lpba.stage_arrays(subject, pairs, stage, allowlist)
        a2, m2, r2 = lpdev_b.dev_stage_array(subject, pairs, stage, allowlist, name_of=lc.pair_name)
        assert a1.tobytes() == a2.tobytes() and r1['input_sha256'] == r2['input_sha256']
        assert {k: v for k, v in m1.items() if k != 'reader'} == {k: v for k, v in m2.items() if k != 'reader'} and set(m1) == set(m2)


def test_b_once_dry_run_and_admission(b_world, tmp_path, monkeypatch):
    dry = run_b(dry_run=True, out=tmp_path/'dry')
    assert dry['spy']['passed'] and {p['exists'] for p in dry['planned']['stage']+dry['planned']['reference']} == {True}
    assert not [e for e in dry['spy']['events'] if Path(e['resolved']).suffix in ('.npz', '.gz', '.pt')] and 'tag-lp-devb' in dry['launch']
    assert sorted(dry['admission']['subjects']) == sorted(DEV_SUBJ)
    run_b()
    with pytest.raises(hc.Refused, match='output'):
        run_b()
    shutil.rmtree(dc.roots().b)
    with pytest.raises(hc.Refused, match='once'):
        run_b()


def test_b_a_subject_off_the_development_source_is_refused(b_world, monkeypatch):
    monkeypatch.setattr(lpdev_b, 'DEV_SUBJECTS', tuple(DEV_SUBJ)+(W.T0[0],))
    monkeypatch.setattr(lpdev_b, 'DEV_PAIRS', tuple(DEV_PAIRS)+((DEV_SUBJ[2], W.T0[0]),))
    monkeypatch.setattr(hc, 'DEV_PAIRS', hc.DEV_PAIRS+(hc.pair_name(DEV_SUBJ[2], W.T0[0]),))
    with pytest.raises(hc.Refused, match='admission'):
        run_b()
    assert not dc.roots().b_lock.exists()


# ==== (c) ===========================================================================================================================================
FIX, MOV = synthetic.oasis(985), synthetic.oasis(986)


def systemctl(state='inactive', returncode=0):
    calls = []

    def run(argv, **kwargs):
        calls.append(list(argv))
        return types.SimpleNamespace(returncode=returncode, stdout=f'{state}\n', stderr='')
    run.calls = calls
    return run


@pytest.fixture
def c_world(dev_world, monkeypatch):
    monkeypatch.setattr(lpdev_c, 'FIXED', FIX)
    monkeypatch.setattr(lpdev_c, 'MOVING', MOV)
    monkeypatch.setattr(lpdev_c, 'PAIR_NAME', hc.pair_name(FIX, MOV))
    stage = dev_world.tmp/'stage_L2R_val_v1'
    stage.mkdir()
    monkeypatch.setattr(lpdev_c, 'DEV_STAGE', stage)
    monkeypatch.setattr(lpdev_c, 'unit_state', lambda run=None: 'inactive')
    rng = np.random.default_rng(3)
    u = rng.normal(size=(6, 5, 4, 3))
    write_hs_reference(u, monkeypatch)
    write_run(u)
    S._reset_cache()
    return types.SimpleNamespace(dev=dev_world, u=u, stage=stage, monkeypatch=monkeypatch)


def write_field(directory, u):
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory/lpdev_c.FIELD_FILE, u=u)
    return dict(file=lpdev_c.FIELD_FILE, sha256=file_sha256(directory/lpdev_c.FIELD_FILE), u_sha256=hashlib.sha256(np.ascontiguousarray(u).tobytes()).hexdigest())


def write_hs_reference(u, monkeypatch, *, pin=True, **changes):
    """HS's synthetic reference, its two sha256 pinned (HS_RUN_SHA256, HS_FIELD_SHA256) unless pin is False."""
    ref = lpdev_c.hs_reference()
    if ref.exists():
        shutil.rmtree(ref)
    field = write_field(ref, u)
    run = dict(status='done', pair=lpdev_c.PAIR_NAME, run_id=lpdev_c.HS_RUN_ID, oracle=False, field=field, checkpoints={'600': dict(field, step=600, seconds_task=50.)}, labels_sha256='9'*64)
    run.update(changes)
    (ref/'run.json').write_text(json.dumps(run), encoding='utf-8')
    (ref/'metrics_0600.json').write_text(json.dumps(dict(dice=.9)), encoding='utf-8')          # an evaluation beside it: never opened
    if pin:
        monkeypatch.setattr(lpdev_c, 'HS_RUN_SHA256', file_sha256(ref/'run.json'))
        monkeypatch.setattr(lpdev_c, 'HS_FIELD_SHA256', file_sha256(ref/lpdev_c.FIELD_FILE))
    return ref


def write_run(u, *, line_status='hx_certified', context=None, t88=None, output=True, identity=None, field=True):
    o = lpdev_c.run_out()
    if o.exists():
        shutil.rmtree(o)
    settings = he.frozen_settings()
    ctx = dict(schema=lpdev_c.CONTEXT_SCHEMA, pair=lpdev_c.PAIR_NAME, stage_dir=str(lpdev_c.DEV_STAGE), out=str(o), pair_set='dev', check_identity=True, settings=settings,
               settings_sha256=he.settings_sha256(settings), settings_frozen=True, test_context=None, code_head=CODE['commit'])
    ctx.update(context or {})
    o.mkdir(parents=True)
    (o/'context.json').write_text(json.dumps(ctx), encoding='utf-8')
    for name in ('identity.json', 'identity_t88.json', 'identity_repair.json', 'identity_end.json'):
        (o/name).write_text(json.dumps(dict(problems=(identity or {}).get(name, []))), encoding='utf-8')
    if t88 is not False:
        rec = dict(schema=lpdev_c.T88_SCHEMA, pair=lpdev_c.PAIR_NAME, status='done', oracle=False, field=write_field(o/'t88', u), seconds_task=60., labels_sha256='9'*64)
        rec.update(t88 or {})
        (o/'t88'/'run.json').write_text(json.dumps(rec), encoding='utf-8')
        if not field:
            (o/'t88'/lpdev_c.FIELD_FILE).unlink()
    line = dict(pair=lpdev_c.PAIR_NAME, line_status=line_status, repair=dict(status='certified', passed=True), fallback=None)
    if output:
        (o/'output').mkdir()
        (o/'output'/'line.json').write_text(json.dumps(line), encoding='utf-8')
        (o/'output'/'cert.json').write_text(json.dumps(dict(process='decision', status='passed', state_hash='f'*64, min_jd_lower_bound=.1)), encoding='utf-8')
    elif line_status is not None:
        (o/'line.json').write_text(json.dumps(line), encoding='utf-8')
    return o


def run_c(**kwargs):
    S._reset_cache()
    return lpdev_c.compare(code=dict(CODE), log=quiet, **kwargs)


def test_c_passes_on_the_same_field_and_hx_certified(c_world):
    record = run_c()
    c = record['comparison']
    assert record['passed'] and c['conditions'] == dict(t88_field_bitwise_hs=True, line_status_hx_certified=True) and c['reasons'] == []
    assert all(c['consistent'].values()) and c['context']['code_head_is_this_commit'] and all(c['identity'].values())
    assert record['pins']['prereg']['sha256'] == lc.PREREG_SHA256 and record['reference_check']['sha256']['run.json'] == lpdev_c.HS_RUN_SHA256
    assert record['run_ended']['active_state'] == 'inactive' and record['run_ended']['terminal_files'] == ['output/line.json']
    assert c['described']['labels_sha256'] == dict(run='9'*64, hs='9'*64, equal=True)
    opened = [Path(e['resolved']).name for e in record['spy']['events']]
    assert 'metrics_0600.json' not in opened and not [n for n in opened if 'evaluation' in n]
    assert {(Path(m['file']).parent.name, m['member']) for m in record['spy']['members']} == {('t88', 'u'), (lpdev_c.SHORT, 'u')}
    assert c['described']['note'].startswith('the repaired field is not compared')
    assert (dc.roots().c/lpdev_c.RECORD).is_file() and dc.roots().c_lock.is_file()


def test_c_one_element_off_does_not_pass(c_world):
    u = c_world.u.copy()
    u[0, 0, 0, 0] = np.nextafter(u[0, 0, 0, 0], np.inf)
    write_hs_reference(u, c_world.monkeypatch)
    record = run_c()
    c = record['comparison']
    assert not record['passed'] and c['conditions']['t88_field_bitwise_hs'] is False and c['fields']['n_elements_differ'] == 1 and c['conditions']['line_status_hx_certified']
    assert c['reasons'] == ['condition not met: t88_field_bitwise_hs']


@pytest.mark.parametrize('kwargs', [dict(line_status='hx_fallback_configC'), dict(line_status='hx_fallback_identity'), dict(line_status='hx_provenance_rejected', output=False)])
def test_c_any_other_line_status_does_not_pass(c_world, kwargs):
    write_run(c_world.u, **kwargs)
    record = run_c()
    assert not record['passed'] and record['comparison']['conditions']['line_status_hx_certified'] is False and record['comparison']['conditions']['t88_field_bitwise_hs']


@pytest.mark.parametrize('kwargs, what', [
    (dict(context=dict(settings_frozen=False)), 'context'), (dict(context=dict(pair_set='holdout')), 'context'), (dict(context=dict(test_context='x')), 'context'),
    (dict(context=dict(check_identity=False)), 'context'), (dict(context=dict(code_head='d'*40)), 'context'), (dict(context=dict(stage_dir='/elsewhere')), 'context'),
    (dict(identity={'identity.json': ['HEAD moved']}), 'identity'), (dict(identity={'identity_end.json': ['the frozen tree changed']}), 'identity'),
    (dict(identity={'identity_t88.json': ['x']}), 'identity'),
    (dict(t88=dict(status='time_cap')), 't88_record'), (dict(t88=dict(oracle=True)), 't88_record'), (dict(t88=dict(schema='other')), 't88_record'),
    (dict(t88=dict(field=dict(file='field_0600.npz', sha256='0'*64, u_sha256='0'*64))), 't88_field'),
])
def test_c_inconsistent_records_do_not_pass(c_world, kwargs, what):
    write_run(c_world.u, **kwargs)
    record = run_c()
    assert not record['passed'] and record['comparison']['consistent'][what] is False and f'not consistent: {what}' in record['comparison']['reasons']


@pytest.mark.parametrize('changes', [dict(run_id='hs_SHx24_first5_0439-0440'), dict(oracle=True), dict(status='time_cap')])
def test_c_the_reference_must_be_hs_s_record(c_world, changes):
    write_hs_reference(c_world.u, c_world.monkeypatch, **changes)
    record = run_c()
    assert not record['passed'] and record['comparison']['consistent']['hs_record'] is False


def test_c_the_reference_must_have_the_pinned_sha256(c_world):
    """Review minor 8: a reference whose bytes are not the pinned ones refuses before the lock."""
    write_hs_reference(c_world.u, c_world.monkeypatch, pin=False, labels_sha256='8'*64)
    with pytest.raises(hc.Refused, match='reference'):
        run_c()
    assert not dc.roots().c_lock.exists()


def test_c_once_dry_run(c_world, tmp_path):
    dry = run_c(dry_run=True, out=tmp_path/'dry', launch_code='/code')
    assert dry['spy']['passed'] and dry['planned']['run_out_exists'] and 'tag-lp-devc' in dry['launch']
    assert not [e for e in dry['spy']['events'] if Path(e['resolved']).suffix == '.npz']
    run_c()
    with pytest.raises(hc.Refused, match='output'):
        run_c()
    shutil.rmtree(dc.roots().c)
    with pytest.raises(hc.Refused, match='once'):
        run_c()


# review I1: the end of the run before the lock; after the lock a missing file is "not passed", never a refusal
@pytest.mark.parametrize('state', ['active', 'activating', 'deactivating', 'reloading'])
def test_c_i1_a_running_unit_refuses_before_the_lock(c_world, monkeypatch, state):
    """T88 done, the repair still running (the unit active): no lock is spent."""
    shutil.rmtree(lpdev_c.run_out()/'output')
    monkeypatch.setattr(lpdev_c, 'unit_state', lambda run=None: state)
    with pytest.raises(hc.Refused, match='has not ended'):
        run_c()
    assert not dc.roots().c_lock.exists() and not dc.roots().c.exists()


def test_c_i1_no_terminal_file_refuses_before_the_lock(c_world):
    shutil.rmtree(lpdev_c.run_out()/'output')
    with pytest.raises(hc.Refused, match='has the run ended'):
        run_c()
    assert not dc.roots().c_lock.exists()


@pytest.mark.parametrize('kwargs, terminal, missing', [
    (dict(t88=False, line_status='hx_provenance_rejected', output=False), ['line.json'], ['t88/run.json', 't88/field_0600.npz']),        # refused at the start: no T88
    (dict(field=False, line_status='hx_fallback_configC'), ['output/line.json'], ['t88/field_0600.npz']),                              # T88 failed, a fallback's output
    (dict(t88=False, line_status=None, output=False), ['line_not_written.json'], ['t88/run.json', 't88/field_0600.npz']),             # a check failed at the end
])
def test_c_i1_after_the_lock_a_missing_field_is_not_passed_with_its_reason(c_world, kwargs, terminal, missing):
    write_run(c_world.u, **kwargs)
    if terminal == ['line_not_written.json']:
        (lpdev_c.run_out()/'line_not_written.json').write_text(json.dumps(dict(reason='the frozen tree or the code changed during the run: [...]', line_status='hx_certified')), encoding='utf-8')
    record = run_c()
    c = record['comparison']
    assert dc.roots().c_lock.is_file() and (dc.roots().c/lpdev_c.RECORD).is_file()
    assert not record['passed'] and record['conclusion'] == 'not passed' and record['run_ended']['terminal_files'] == terminal
    assert c['missing'] == missing and all(f'missing: {m}' in c['reasons'] for m in missing) and 'condition not met: t88_field_bitwise_hs' in c['reasons']
    assert c['conditions']['line_status_hx_certified'] is False and 'reason' in c['fields']


def test_c_unit_state_reads_systemctl_and_fails_closed():
    run = systemctl('inactive')
    assert lpdev_c.unit_state(run=run) == 'inactive' and run.calls == [['systemctl', '--user', 'show', 'tag-lp-devc', '--property=ActiveState', '--value']]
    assert lpdev_c.unit_state(run=systemctl('active')) == 'active'
    for bad in (systemctl('inactive', returncode=1), systemctl('')):
        with pytest.raises(hc.Refused, match='unit'):
            lpdev_c.unit_state(run=bad)

    def broken(argv, **kwargs):
        raise OSError('no systemctl')
    with pytest.raises(hc.Refused, match='unit'):
        lpdev_c.unit_state(run=broken)


def test_c_main_exit_codes(c_world, monkeypatch, tmp_path):
    """Review minors 4, 6, 7: compare 0, 8 (once), 3, 9; dry-run 0; gate 0 and 8 (a disk guard's InfrastructureFailure too); launch-string 0."""
    monkeypatch.setattr(hp, 'check_code', lambda code, require_clean=True: dict(problems=[]))
    monkeypatch.setattr(dc, 'bytecode_problem', lambda: None)
    lines = []
    S._reset_cache()
    assert lpdev_c.main(['dry-run', '--out', str(tmp_path/'dry')], code=dict(CODE), out=lines.append) == 0
    assert lpdev_c.main(['launch-string', '--code', '/some/checkout'], out=lines.append) == 0 and '--unit=tag-lp-devc' in lines[-1]
    S._reset_cache()
    assert lpdev_c.main(['compare'], code=dict(CODE), out=lines.append) == 0
    S._reset_cache()
    assert lpdev_c.main(['compare'], code=dict(CODE), out=lines.append) == 8
    shutil.rmtree(dc.roots().c)
    dc.roots().c_lock.unlink()                                       # a test only
    write_run(c_world.u, line_status='hx_fallback_configC')
    S._reset_cache()
    assert lpdev_c.main(['compare'], code=dict(CODE), out=lines.append) == 3
    shutil.rmtree(dc.roots().c)
    dc.roots().c_lock.unlink()
    monkeypatch.setattr(lpdev_c, 'compare_run', lambda **kwargs: (_ for _ in ()).throw(RuntimeError('broken')))
    S._reset_cache()
    assert lpdev_c.main(['compare'], code=dict(CODE), out=lines.append) == 9
    assert (dc.roots().c/lpdev_c.STOPPED).is_file()
    shutil.rmtree(lpdev_c.run_out())
    assert lpdev_c.main(['gate'], out=lines.append) == 0
    monkeypatch.setattr(hc, 'disk_guard', lambda **kwargs: (_ for _ in ()).throw(hc.InfrastructureFailure('disk_guard', 'C: free 10 GiB < need 2 + 20')))
    assert lpdev_c.main(['gate'], out=lines.append) == 8 and 'infrastructure' in lines[-1]


def test_c_the_start_string_is_hes_supervisor_unit(dev_world):
    argv = lpdev_c.launch_argv('/some/checkout')
    assert he.unit_argv_problems(argv, he.SUPERVISOR_UNIT) == [] and '--unit=tag-lp-devc' in argv
    assert f'--setenv=PYTHONPATH={he.core.FROZEN}:/some/checkout' in argv
    tail = argv[argv.index('/some/checkout/experiments/he/run_hx.py'):]
    assert tail == ['/some/checkout/experiments/he/run_hx.py', 'pair', '--stage-dir', str(lpdev_c.DEV_STAGE), '--pair', lpdev_c.PAIR_NAME, '--out', str(lpdev_c.run_out()), '--device', 'cuda',
                    '--launcher', 'unit']
    assert lpdev_c.PAIR_NAME == 'OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1' and lpdev_c.HS_REFERENCE_REL == 'hs_v1/runs/SHx24_first5/0438-0439'
    assert lpdev_c.run_out() == dev_world.results/'lpba40_devcheck_v1'/'c_run'/'0438-0439'


def test_c_the_gate(dev_world, monkeypatch):
    seen = {}
    monkeypatch.setattr(hc, 'machine_gate', lambda **kwargs: seen.update(kwargs) or dict(passed=True))
    assert lpdev_c.gate()['machine'] == dict(passed=True) and seen['mem_min_gib'] == he.FROZEN_RUN_GATE_GIB
    lpdev_c.run_out().mkdir(parents=True)
    with pytest.raises(hc.Refused, match='output'):
        lpdev_c.gate()
    lpdev_c.run_out().rmdir()
    hold = lc.roots('lpba40_feas').hold
    hold.parent.mkdir(parents=True)
    hold.write_text('')
    with pytest.raises(hc.Refused, match='hold'):
        lpdev_c.gate()


# ==== shared ==========================================================================================================================================
def test_the_roots_are_new_and_under_the_results_root():
    r = dc.roots()
    base = Path('/home/rmcfz/seq_vp_tag_results')
    assert (r.a, r.a_lock) == (base/'lpba40_feas_v1'/'devcheck_a', base/'lpba40_feas_v1'/'locks'/'lpba_devcheck_a.lock')
    assert (r.b, r.b_lock, r.c, r.c_lock, r.c_run) == (base/'lpba40_devcheck_v1'/'b', base/'lpba40_devcheck_v1'/'locks'/'devcheck_b.lock', base/'lpba40_devcheck_v1'/'c',
                                                        base/'lpba40_devcheck_v1'/'locks'/'devcheck_c.lock', base/'lpba40_devcheck_v1'/'c_run')
    assert lpdev_a.OLD_STAGE == hc.MAIN_REPO/'data'/'three_d'/'stage_LPBA40_v1' and lpdev_b.DEV_STAGE == lpdev_c.DEV_STAGE == hc.DEV_STAGE
    assert len(lpdev_b.DEV_SUBJECTS) == 20 and len(lpdev_b.DEV_PAIRS) == 19 and {s for p in lpdev_b.DEV_PAIRS for s in p} == set(lpdev_b.DEV_SUBJECTS)
    assert lpdev_b.REFERENCE_ROOT == base/'seg'/'unet35_v4_ens'/'pred'
    assert [lpdev_a.lc.unit_name('lpba40_feas', lpdev_a.UNIT_TOOL), lpdev_b.UNIT, lpdev_c.UNIT] == ['tag-lp-feas-deva', 'tag-lp-devb', 'tag-lp-devc']


def test_arrays_equal_is_bitwise():
    a = np.array([0., 1., 2.])
    assert dc.arrays_equal(a, a.copy())['bitwise_equal']
    assert not dc.arrays_equal(a, a.astype(np.float32))['bitwise_equal']
    assert not dc.arrays_equal(np.array([0.]), np.array([-0.]))['bitwise_equal']          # -0.0 == 0.0 but not the same bits
    r = dc.arrays_equal(a, a+[0, 0, 1e-12])
    assert not r['bitwise_equal'] and r['n_elements_differ'] == 1


ORDER = (
    ('experiments/lpba/devcheck/lpdev_a.py', 'run', ['admit', 'load_c_record', 'check_hold', 'check_stage', 'machine_gate', 'take_lock', 'after_lock', 'compare']),
    ('experiments/lpba/devcheck/lpdev_b.py', 'run', ['admit_dev', 'check_pins', 'check_registry', 'check_contact', 'check_hold', 'select_models', 'check_decision', 'plan_of',
                                                     'reference_record', 'machine_gate', 'take_lock', 'after_lock', 'dev_stage_array', 'predict', 'compare_subject']),
    ('experiments/lpba/devcheck/lpdev_c.py', 'compare', ['admit_dev', 'check_pins', 'check_hold', 'check_reference', 'run_ended', 'take_lock', 'after_lock', 'compare_run']),
)


def test_the_checks_admit_before_they_read():
    """C26.5-1 次序 (admit, then read), pinned by hand as test_lpba_census.ORDER does."""
    for rel, name, calls in ORDER:
        tree = ast.parse((ROOT/rel).read_text(encoding='utf-8'))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        lines = [min(n.lineno for n in ast.walk(node) if isinstance(n, ast.Call) and (getattr(n.func, 'id', None) == c or getattr(n.func, 'attr', None) == c)) for c in calls]
        assert lines == sorted(lines), (rel, name, list(zip(calls, lines)))
