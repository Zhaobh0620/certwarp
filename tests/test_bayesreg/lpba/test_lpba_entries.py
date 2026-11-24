"""LP-A items 2-4 (contract C26.5-2 to -4, C26.10 T8, T9, T10 (the image half), T16 (this package's links)): the LPBA40 data preparation, HI's LPBA mode and the Atropos entry on
a synthetic world only -- a temporary code tree pinned into bayesreg (its sealed lists synthetic, the pinned files copied, a synthetic LPBA allowlist pinned by monkeypatch), the
results root and the LPBA40 data root in tmp_path, tiny synthetic NIfTI files with LPBA40's kind of header (an LAS affine, units unknown), tiny synthetic U-Net members, ANTs
replaced by a stand-in.  No real image, label or result root is touched (holdout_common's test sandbox is set)."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import types

import nibabel as nib
import numpy as np
import pytest

import bayesreg.io.guard as G
import bayesreg.io.sealed as S
from bayesreg.io.guard import SubjectSealed, load_dev_allowlist
from bayesreg.io.pair import load_pair, recheck_sealed_pair, subject_list_sha256, validate_segmentation_provenance
from jd_curl.three_d.nifti import load_scalar

from ..sg import _sg_world as W

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT/'experiments'/'lpba', ROOT/'experiments'/'hi', ROOT/'experiments'/'d0', ROOT/'experiments'/'holdout', ROOT/'experiments'/'xb'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import lpba_common as lc  # noqa: E402
import prep_lpba as prep  # noqa: E402
import atropos_lpba as atr  # noqa: E402
import atropos_seg  # noqa: E402
import hi_lpba  # noqa: E402

hc, hp = lc.hc, lc.hp
SHAPE_XYZ = (12, 16, 16)                            # native; stored (nz, ny, nx) = (16, 16, 12) (the HI world's size: the tiny members' windows fit)
STORED = (16, 16, 12)
FEAS = ('S01', 'S02')
PINNED = (lc.PREREG_REL, *lc.LIST_SHA256, *lc.FROZEN_FILES, lc.C_RECORD_REL)


def las_image(subject, *, units='unknown', label=False):
    """An LPBA40-like header: an LAS affine (x to the left), qform and sform coded, the spatial unit not mm; a subject-dependent image, or a label map."""
    rng = np.random.default_rng(int(subject[1:]) + (100 if label else 0))
    data = rng.uniform(0., 300., SHAPE_XYZ)
    data[:2] = 0.                                    # background slabs: v > 0 is not everything
    data[:, :, -2:] = 0.
    if label:
        data = (data > 150).astype(np.int16)*21+(data > 250).astype(np.int16)
    affine = np.array([[-1., 0, 0, 90], [0, 1., 0, -126], [0, 0, 1., -72], [0, 0, 0, 1]])
    img = nib.Nifti1Image(data.astype(np.int16 if label else np.float32), affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    img.header.set_xyzt_units(units)
    return img


def lpba_document(contact):
    sources = [dict(name='synthetic_dev', dataset='OASIS', path='lists/dev.txt', sha256='a'*64, never_for_training=False, subjects=list(W.DEV)),
               dict(name='synthetic_train', dataset='OASIS', path='lists/train.txt', sha256='b'*64, never_for_training=False, subjects=list(contact)),
               dict(name='lpba_tune', dataset='LPBA40', path='lists/tune.txt', sha256='c'*64, never_for_training=False, subjects=[f'S{n:02d}' for n in range(1, 25)]),
               dict(name='lpba40_test_v1', dataset='LPBA40', path=lc.TEST_SUBJECTS_REL, sha256=lc.LIST_SHA256[lc.TEST_SUBJECTS_REL], never_for_training=False,
                    subjects=[f'S{n:02d}' for n in range(3, 37)])]
    return dict(schema=G.ALLOWLIST_SCHEMA, subjects=dict(OASIS=list(W.DEV)+list(contact), LPBA40=[f'S{n:02d}' for n in range(1, 37)]), sources=sources)


@pytest.fixture
def world(tmp_path, monkeypatch):
    from ..hi import _world as HW
    w = W.make_world(tmp_path, monkeypatch, record=False)
    for rel in PINNED:
        (w.repo/rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, w.repo/rel)
    path = W.write_json(w.repo/G.LPBA40_ALLOWLIST_REL, lpba_document(HW.CONTACT))
    monkeypatch.setattr(G, 'LPBA40_ALLOWLIST_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    results, raw = tmp_path/'results', tmp_path/'lpba_raw'
    results.mkdir()
    for subject in [f'S{n:02d}' for n in range(1, 37)]+['S37']:
        directory = raw/subject/'delineation'
        directory.mkdir(parents=True)
        nib.save(las_image(subject), str(directory/'image.nii.gz'))
        nib.save(las_image(subject, label=True), str(directory/'label.nii.gz'))       # never opened (the spy refuses the name)
    monkeypatch.setenv(hc.TEST_SANDBOX_ENV, '1')
    monkeypatch.setattr(lc, 'RESULTS', results)
    monkeypatch.setattr(hc, 'RESULTS', results)
    monkeypatch.setattr(lc, 'RAW_ROOT', raw)
    monkeypatch.setattr(prep, 'EXPECTED_SHAPE_ZYX', STORED)
    monkeypatch.setattr(hc, 'machine_gate', lambda **kwargs: dict(passed=True, note='test: no machine queried', **kwargs))
    monkeypatch.setattr(hc, 'disk_guard', lambda **kwargs: dict(passed=True, note='test', **kwargs))
    S._reset_cache()
    yield types.SimpleNamespace(repo=w.repo, results=results, raw=raw, allowlist_path=path, tmp=tmp_path, HW=HW)
    S._reset_cache()


CODE = dict(commit='c'*40, clean=True, note='test')


def run_prep(pair_set='lpba40_feas', **kwargs):
    S._reset_cache()
    return prep.run(pair_set, code=CODE, log=lambda text: None, **kwargs)


# ---- T8: the names, the pins, the code ----------------------------------------------------------------------------------------------------------
def test_t8_the_formal_mode_takes_only_the_34_and_the_17(world):
    admitted = lc.admit('lpba40_test')
    assert admitted.pairs == [(f'S{n:02d}', f'S{n+1:02d}') for n in range(3, 37, 2)] and len(admitted.subjects) == 34
    assert all(r['admitted_by_lpba40'] for r in admitted.record['subjects'].values())
    for subjects in (['S01'], ['S02'], ['S37'], ['S40'], ['OASIS_OAS1_0438_MR1']):
        with pytest.raises(hc.Refused, match='not a subject of lpba40_test'):
            lc.admit('lpba40_test', subjects=subjects)
    with pytest.raises(hc.Refused, match='other way round'):
        lc.admit('lpba40_test', pairs=[('S04', 'S03')])
    with pytest.raises(hc.Refused, match='not a row'):
        lc.admit('lpba40_test', pairs=[('S03', 'S05')])


def test_t8_the_feasibility_mode_takes_only_s01_s02(world):
    admitted = lc.admit('lpba40_feas')
    assert admitted.pairs == [('S01', 'S02'), ('S02', 'S01')]
    assert {s: r['admitted_by_lpba40'] for s, r in admitted.record['subjects'].items()} == dict(S01=False, S02=False)
    for subject in ('S03', 'S36', 'S37'):
        with pytest.raises(hc.Refused):
            lc.admit('lpba40_feas', subjects=[subject])
    with pytest.raises(hc.Refused, match='pair_set'):
        lc.admit('holdout')


def test_t8_the_pins_refuse(world, monkeypatch):
    monkeypatch.setattr(lc, 'FROZEN_FILES', dict(lc.FROZEN_FILES, **{'experiments/d0/configC_reference.json': '0'*64}))
    with pytest.raises(hc.Refused, match='pins'):
        lc.admit('lpba40_feas')


@pytest.mark.parametrize('name, value', [('SETTINGS_SHA256', '1'*64), ('PREREG_SHA256', '2'*64), ('C_RECORD_SHA256', '3'*64)])
def test_t8_settings_prereg_and_c_record_pins(world, monkeypatch, name, value):
    monkeypatch.setattr(lc, name, value)
    with pytest.raises(hc.Refused, match='pins'):
        lc.admit('lpba40_feas')


def test_t8_only_the_pinned_lpba_allowlist(world, monkeypatch):
    monkeypatch.setattr(G, 'LPBA40_ALLOWLIST_SHA256', 'f'*64)
    with pytest.raises(hc.Refused, match='allowlist'):
        lc.load_lpba_allowlist()


def test_t8_the_lpba_allowlist_with_never_for_training_true_is_refused(world, monkeypatch):
    doc = json.loads(world.allowlist_path.read_text(encoding='utf-8'))
    doc['sources'][-1]['never_for_training'] = True
    path = W.write_json(world.allowlist_path, doc)
    monkeypatch.setattr(G, 'LPBA40_ALLOWLIST_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(hc.Refused, match='never_for_training'):
        lc.load_lpba_allowlist()


def test_t8_a_formal_run_refuses_without_rule_one(world, monkeypatch):
    monkeypatch.setattr(hp, 'check_code', lambda code, require_clean=True: dict(problems=[]))
    with pytest.raises(hc.Refused, match='rule one'):
        lc.check_code(dict(CODE), 'lpba40_test')
    assert lc.check_code(dict(CODE), 'lpba40_feas')['check'] == dict(problems=[])
    for run in (lambda: prep.run('lpba40_test', code=CODE), lambda: atr.run('lpba40_test', code=CODE), lambda: hi_lpba.run('lpba40_test', code=CODE, device='cpu')):
        with pytest.raises(hc.Refused, match='code'):
            run()
    assert not (world.results/'lpba40_test_v1').exists()


def test_t8_units_are_tag_lp():
    assert [lc.unit_name(p, t) for p in lc.PAIR_SETS for t in ('prep', 'atropos', 'hi')] == [
        'tag-lp-prep', 'tag-lp-atropos', 'tag-lp-hi', 'tag-lp-feas-prep', 'tag-lp-feas-atropos', 'tag-lp-feas-hi']


def test_the_roots_are_the_contracts():
    r, f = lc.roots('lpba40_test'), lc.roots('lpba40_feas')
    base = Path('/home/rmcfz/seq_vp_tag_results')
    assert (r.root, r.pred_root, r.hold) == (base/'lpba40_test_v1', base/'seg/unet35_v4_ens_lpba40_test_v1/pred', base/'lpba40_test_v1/hold')
    assert (f.root, f.pred_root) == (base/'lpba40_feas_v1', base/'seg/unet35_v4_ens_lpba40_feas_v1/pred')
    assert lc.image_path('S05') == Path('/mnt/d/data/LPBA40/nii/subjects/S05/delineation/image.nii.gz')
    assert prep.EXPECTED_SHAPE_ZYX == (181, 217, 181) and prep.PERCENTILE == 99.5
    assert lc.C_RECORD_SHA256 == hashlib.sha256((ROOT/lc.C_RECORD_REL).read_bytes()).hexdigest()


def test_the_c_record_is_read_from_its_hex(world):
    c = lc.load_c_record()
    doc = json.loads((ROOT/lc.C_RECORD_REL).read_text(encoding='utf-8'))
    assert c.c == float.fromhex(doc['c']['hex']) == float(doc['c']['decimal']) and c.W_dev == float.fromhex(doc['W_dev']['hex'])


# ---- T9: the data preparation -----------------------------------------------------------------------------------------------------------------------
def test_t9_the_stage_holds_no_label_and_binds_p7(world):
    record = run_prep()
    assert record['passed'] and record['spy']['passed'] and not record['spy']['breaches']
    stage = lc.roots('lpba40_feas').stage
    opened = [e['resolved'] for e in record['spy']['events']]
    assert not [p for p in opened if 'label' in Path(p).name] and not [p for p in opened if p.startswith(str(world.raw)) and not p.endswith('image.nii.gz')]
    assert sorted({p for p in opened if p.startswith(str(world.raw))}) == [str(world.raw/s/'delineation/image.nii.gz') for s in FEAS]
    c = lc.load_c_record().c
    arrays = {}
    for fixed, moving in (('S01', 'S02'), ('S02', 'S01')):
        name = lc.pair_name(fixed, moving)
        with np.load(stage/name/'pair.npz', allow_pickle=False) as npz:
            assert npz.files == ['fixed', 'moving', 'initial_u']
            members = {k: npz[k] for k in npz.files}
        assert members['initial_u'].shape == STORED+(3,) and not members['initial_u'].any() and members['fixed'].dtype == np.float64
        manifest = json.loads((stage/name/'manifest.json').read_text(encoding='utf-8'))
        assert set(manifest) == {'shape_zyx', 'spacing_xyz', 'origin_xyz', 'unit', 'fixed_subject', 'moving_subject', 'sources', 'source_files', 'intensity', 'flip', 'weights', 'pair_sha256'}
        assert 'label' not in json.dumps(list(manifest)) and 'label_sets' not in json.dumps(manifest) and 'baseline_label_metrics' not in json.dumps(manifest)
        for side, subject in (('fixed', fixed), ('moving', moving)):
            digest = hashlib.sha256(np.ascontiguousarray(members[side]).tobytes()).hexdigest()
            assert manifest['sources'][side] == dict(sha256=digest)                                  # P7: the stage array's sha256, not a file's
            files = manifest['source_files'][side]
            assert files['original']['path'] == str(world.raw/subject/'delineation/image.nii.gz') and files['mm_copy']['sha256'] != digest
            copy = Path(files['mm_copy']['path'])
            v = np.flip(load_scalar(copy).values, 2).copy()                                           # P2, P3
            p = float(np.percentile(v[v > 0], 99.5))
            assert np.array_equal(members[side], v / p * c) and members[side].tobytes() == (v / p * c).tobytes()      # P5, no clipping
            assert manifest['intensity']['p'][side]['p99_5']['hex'] == p.hex()
            arrays.setdefault(subject, set()).add(digest)
    assert all(len(d) == 1 for d in arrays.values())                                                   # one array per subject in both directions
    assert set(record['stage_summary']['read_back']['subjects']) == set(FEAS)
    assert (stage/prep.SUMS).is_file() and json.loads((stage/prep.RECORD).read_text(encoding='utf-8'))['passed']


def test_t9_the_mm_copy_keeps_data_affine_and_forms(world, tmp_path):
    out = tmp_path/'copy'/'x.nii.gz'
    rec = prep.mm_copy(world.raw/'S05/delineation/image.nii.gz', out)
    original, copy = nib.load(str(world.raw/'S05/delineation/image.nii.gz')), nib.load(str(out))
    assert copy.header.get_xyzt_units()[0] == 'mm' and original.header.get_xyzt_units()[0] == 'unknown'
    assert np.asanyarray(copy.dataobj).tobytes() == np.asanyarray(original.dataobj).tobytes() and np.array_equal(copy.affine, original.affine)
    assert rec['units_before'][0] == 'unknown' and rec['units_after'][0] == 'mm'
    with pytest.raises(FileExistsError):
        prep.mm_copy(world.raw/'S05/delineation/image.nii.gz', out)


def test_t9_a_subject_with_two_different_arrays_stops(world, monkeypatch):
    real, calls = prep.scale, []

    def drifting(v, c):
        calls.append(1)
        img, p = real(v, c)
        return (img*(1+1e-12) if len(calls) == 3 else img), p             # the third array: S02 in its second pair
    monkeypatch.setattr(prep, 'scale', drifting)
    with pytest.raises(prep.StageCheckFailed, match='two different arrays'):
        run_prep()
    stage = lc.roots('lpba40_feas').stage
    assert (stage/prep.STOPPED).is_file() and not (stage/prep.RECORD).exists()


def test_t9_once_the_lock_and_the_stage(world):
    run_prep()
    with pytest.raises(hc.Refused, match='output'):
        run_prep()
    shutil.rmtree(lc.roots('lpba40_feas').stage)
    with pytest.raises(hc.Refused, match='once'):
        run_prep()


def test_t9_a_label_read_is_a_breach(world, monkeypatch):
    monkeypatch.setattr(lc, 'image_path', lambda subject: Path(lc.RAW_ROOT)/subject/'delineation'/'label.nii.gz')
    with pytest.raises(hp.Breach, match='label'):
        run_prep()


def test_t9_the_grid_check(world, monkeypatch):
    monkeypatch.setattr(prep, 'EXPECTED_SHAPE_ZYX', (181, 217, 181))
    with pytest.raises(prep.StageCheckFailed, match='P4'):
        run_prep()


def test_t9_the_hold_pauses(world):
    hold = lc.roots('lpba40_feas').hold
    hold.parent.mkdir(parents=True)
    hold.write_text('')
    with pytest.raises(hc.Refused, match='hold'):
        run_prep()


def test_t9_the_dry_run_reads_no_image(world, tmp_path):
    record = run_prep(dry_run=True, out=tmp_path/'dry')
    assert [p['exists'] for p in record['planned_images']] == [True, True]
    assert not [e for e in record['spy']['events'] if e['resolved'].startswith(str(world.raw))] and not lc.roots('lpba40_feas').stage.exists()


# ---- T10 (the image half): the same transform keeps voxel correspondence --------------------------------------------------------------------------
def test_t10_image_and_label_through_the_same_transform_correspond(tmp_path):
    rng = np.random.default_rng(7)
    native_labels = rng.integers(0, 4, SHAPE_XYZ).astype(np.int16)
    native_image = (native_labels*100+rng.uniform(1, 50, SHAPE_XYZ)).astype(np.float32)       # the label is a function of the image: floor(image / 100)
    affine = np.array([[-1., 0, 0, 90], [0, 1., 0, -126], [0, 0, 1., -72], [0, 0, 0, 1]])
    paths = {}
    for kind, data in (('image', native_image), ('label', native_labels)):
        img = nib.Nifti1Image(data, affine)
        img.set_qform(affine, code=1)
        img.set_sform(affine, code=1)
        img.header.set_xyzt_units('unknown')
        nib.save(img, str(tmp_path/f'{kind}.nii.gz'))
        paths[kind] = prep.mm_copy(tmp_path/f'{kind}.nii.gz', tmp_path/'mm'/f'{kind}.nii.gz')['mm_copy']['path']
    image, _ = prep.stage_image(paths['image'])
    label = np.flip(load_scalar(paths['label'], labels=True).values, 2).copy()        # the evaluation's half (C26.7-3), here only to show the correspondence
    assert image.shape == label.shape == STORED
    assert np.array_equal(np.floor(image/100), label)
    # the left-right reading: x of the LAS file runs to the left; RAS storage flips it, the stage's np.flip(., 2) flips it back -- axis 2 of the stage is native x
    assert np.array_equal(image, native_image.astype(np.float64).transpose(2, 1, 0))


# ---- T16 (this package's links) and the provenance bindings ---------------------------------------------------------------------------------------
def test_t16_load_pair_a4_admission_and_xb_on_the_lpba_allowlist(world, monkeypatch):
    run_prep()
    allowlist = load_dev_allowlist(world.allowlist_path)
    stage = lc.roots('lpba40_feas').stage
    pair = load_pair(stage/lc.pair_name('S01', 'S02')/'pair.npz', allowlist=allowlist)
    assert recheck_sealed_pair(pair, allowlist, where='t16') is None
    import make_inputs
    unet = types.SimpleNamespace(folds=ROOT/'splits/bayesreg_v1/folds.json', folds_sha256=None)
    from bayesreg.seg.unet.pins import PINNED_FOLDS_SHA256
    if hashlib.sha256(unet.folds.read_bytes()).hexdigest() != PINNED_FOLDS_SHA256:
        pytest.fail('splits/bayesreg_v1/folds.json is not the pinned folds file')
    assert make_inputs.a4_admission(['S05', 'S06'], world.allowlist_path, unet)['subjects'] == ['S05', 'S06']
    doc = json.loads(world.allowlist_path.read_text(encoding='utf-8'))
    doc['sources'][-1]['never_for_training'] = True
    never = W.write_json(world.tmp/'never.json', doc)
    with pytest.raises(SystemExit, match='never_for_training'):
        make_inputs.a4_admission(['S05', 'S06'], never, unet)
    import xb_core
    record = xb_core.check_pair_input(stage/lc.pair_name('S01', 'S02')/'pair.npz', fixed='S01', moving='S02', allowlist=allowlist)
    assert record['pair_sha256'] == pair.provenance['pair_sha256']


def fake_ants(monkeypatch):
    """ANTs replaced: ImageMath copies t1 to t1_trunc, Atropos writes labels and three posteriors by intensity tertiles in the mask (darkest first)."""
    def run(argv, cwd, env, stdout, stderr):
        out = Path(cwd)
        if Path(argv[0]).name == 'ImageMath':
            shutil.copyfile(out/'t1.nii.gz', out/'t1_trunc.nii.gz')
        else:
            t1, mask = nib.load(str(out/'t1.nii.gz')), np.asanyarray(nib.load(str(out/'mask.nii.gz')).dataobj) > 0
            v = np.asanyarray(t1.dataobj).astype(np.float64)
            q = np.quantile(v[mask], [1/3, 2/3])
            classes = np.where(mask, 1+(v > q[0])+(v > q[1]), 0)
            nib.save(nib.Nifti1Image(classes.astype(np.uint8), t1.affine), str(out/'labels.nii.gz'))
            for k in (1, 2, 3):
                nib.save(nib.Nifti1Image((classes == k).astype(np.float32), t1.affine), str(out/f'posteriors{k:02d}.nii.gz'))
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(atropos_seg, 'subprocess', types.SimpleNamespace(run=run, STDOUT=-2))


def test_atropos_binds_p7_and_records_the_wm_ratio(world, monkeypatch):
    run_prep()
    fake_ants(monkeypatch)
    S._reset_cache()
    record = atr.run('lpba40_feas', code=CODE, log=lambda text: None)
    assert record['spy']['passed'] and record['passed'] == (not record['feasibility_stop']['stop'])
    out = lc.roots('lpba40_feas').atropos
    allowlist = load_dev_allowlist(world.allowlist_path)
    stage = lc.roots('lpba40_feas').stage
    W_dev = lc.load_c_record().W_dev
    for subject in FEAS:
        prov = json.loads((out/subject/'provenance.json').read_text(encoding='utf-8'))
        for fixed, moving in (('S01', 'S02'), ('S02', 'S01')):
            pair = load_pair(stage/lc.pair_name(fixed, moving)/'pair.npz', allowlist=allowlist)
            side = 'fixed' if fixed == subject else 'moving'
            validate_segmentation_provenance(prov, pair, side=side, layer='tissue', allowlist=allowlist)
            digest = hashlib.sha256(np.ascontiguousarray(getattr(pair, side)).tobytes()).hexdigest()
            assert list(prov['input_sha256'].values()) == [digest] == [pair.source_sha256[side]] and prov['image_source']['array_sha256_float64'] == digest
        image = getattr(pair, side)
        wm = atropos_seg.load_zyx(out/subject/prov['posterior_files'][prov['wm_index']]) > .5
        assert record['wm'][subject]['wm_ratio']['hex'] == (float(image[wm].mean())/W_dev).hex()
    assert record['shared_subjects'] == ['S01', 'S02']
    opened = [e['resolved'] for e in record['spy']['events']]
    assert not [p for p in opened if 'label' in Path(p).name and 'atropos' not in p]


def test_atropos_qc_stops(world, monkeypatch):
    run_prep()

    def lopsided(argv, cwd, env, stdout, stderr):
        out = Path(cwd)
        if Path(argv[0]).name == 'ImageMath':
            shutil.copyfile(out/'t1.nii.gz', out/'t1_trunc.nii.gz')
            return types.SimpleNamespace(returncode=0)
        t1, mask = nib.load(str(out/'t1.nii.gz')), np.asanyarray(nib.load(str(out/'mask.nii.gz')).dataobj) > 0
        classes = np.where(mask, 3, 0)
        classes[np.unravel_index(np.flatnonzero(mask)[:2], mask.shape)] = 1
        classes[np.unravel_index(np.flatnonzero(mask)[2:4], mask.shape)] = 2
        nib.save(nib.Nifti1Image(classes.astype(np.uint8), t1.affine), str(out/'labels.nii.gz'))
        for k in (1, 2, 3):
            nib.save(nib.Nifti1Image((classes == k).astype(np.float32), t1.affine), str(out/f'posteriors{k:02d}.nii.gz'))
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(atropos_seg, 'subprocess', types.SimpleNamespace(run=lopsided, STDOUT=-2))
    S._reset_cache()
    with pytest.raises(SystemExit, match='QC failed'):
        atr.run('lpba40_feas', code=CODE, log=lambda text: None)
    assert (lc.roots('lpba40_feas').atropos/atr.STOPPED).is_file()


def test_atropos_refuses_a_changed_stage(world):
    run_prep()
    stage = lc.roots('lpba40_feas').stage
    manifest = stage/lc.pair_name('S02', 'S01')/'manifest.json'
    manifest.write_text(manifest.read_text(encoding='utf-8')+' ', encoding='utf-8')
    S._reset_cache()
    with pytest.raises(hc.Refused, match='differs from SHA256SUMS'):
        atr.run('lpba40_feas', code=CODE, log=lambda text: None)


def test_hi_stage_reader_is_scoped_and_restored(world):
    from bayesreg.seg.unet import data
    original = data.read_image
    files = hi_lpba.StageFiles('lpba40_feas', '/x')
    array = np.zeros(STORED)
    with hi_lpba.stage_reader(files, 'S01', array, dict(sha256='a'*64)) as calls:
        assert data.read_image(files, 'S01')[0] is array
        with pytest.raises(hc.Refused, match='reader'):
            data.read_image(files, 'S02')
        with pytest.raises(hc.Refused, match='reader'):
            data.read_image(types.SimpleNamespace(), 'S01')
    assert data.read_image is original and calls == ['S01']
    with pytest.raises(RuntimeError):
        with hi_lpba.stage_reader(files, 'S01', array, {}):
            raise RuntimeError('inside')
    assert data.read_image is original


def hi_world(world, monkeypatch):
    """The HI world's tiny members, a registry of them (contact: the synthetic training set on the LPBA allowlist) and its anchor; HI's gates replaced as in the HI tests."""
    import torch
    HW = world.HW
    hi = hi_lpba.hi
    models_root = world.tmp/'members'
    shas = {f'fold{f}_s{s}': HW.write_member(models_root, f, s, t, w_, g, a) for f, s, t, w_, g, a in HW.MEMBERS}
    from bayesreg.io.pair import load_segmenter_registry
    from bayesreg.seg.unet.groups import DEFAULT_GROUPS_PATH, load_grouping
    registry = load_segmenter_registry(HW.write_json(world.tmp/'registry.json', dict(schema='bayesreg.segmenters.v1', comment='synthetic', models={
        f'unet35_v4_ens_fold{f}': dict(segmenter_id='unet35_v4_ens', fold=i, member_model_sha256=[shas[f'fold{f}_s0'], shas[f'fold{f}_s1']], label_contact_subjects=list(HW.CONTACT),
                                       label_contact_subjects_sha256=subject_list_sha256(HW.CONTACT)) for i, f in enumerate('AB')})))
    anchor = dict(hi.CONTRACT_SEGMENTER, registry_sha256=registry.sha256)
    real_check = hi.check_registry
    monkeypatch.setattr(hi, 'check_registry', lambda r, anchor_=None: real_check(r, anchor))
    monkeypatch.setattr(hi_lpba, 'inputs', lambda: (registry, load_grouping(DEFAULT_GROUPS_PATH, 'structure13_v1'), Path(DEFAULT_GROUPS_PATH)))
    monkeypatch.setattr(hi, 'MODELS_ROOT', models_root)
    monkeypatch.setattr(hi, 'check_decision', lambda decision: dict(devcheck_json='/test/devcheck.json', devcheck_sha256='d'*64, note='test'))
    monkeypatch.setattr(hi, 'check_environment', lambda device, decision: hi.environment_of(device))
    return registry, torch


def test_hi_lpba_mode_binds_p7_end_to_end(world, monkeypatch):
    run_prep()
    registry, torch = hi_world(world, monkeypatch)
    S._reset_cache()
    record = hi_lpba.run('lpba40_feas', code=CODE, device='cpu', log=lambda text: None)
    assert record['passed'] and record['spy']['passed'] and sorted(record['rows']) == list(FEAS)
    r = lc.roots('lpba40_feas')
    allowlist = load_dev_allowlist(world.allowlist_path)
    for subject in FEAS:
        prov = json.loads((r.pred_root/subject/'provenance.json').read_text(encoding='utf-8'))
        assert prov['allowlist_sha256'] == allowlist.sha256 and prov['policy'] == 'all_folds_mean' and prov['hi']['pair_set'] == 'lpba40_feas'
        for fixed, moving in (('S01', 'S02'), ('S02', 'S01')):
            pair = load_pair(r.stage/lc.pair_name(fixed, moving)/'pair.npz', allowlist=allowlist)
            side = 'fixed' if fixed == subject else 'moving'
            accepted = validate_segmentation_provenance(prov, pair, side=side, layer='structure', registry=registry, allowlist=allowlist)
            assert list(accepted['input_sha256'].values()) == [pair.source_sha256[side]]           # P7: the HI provenance's input_sha256 is the stage array's
    opened = [e['resolved'] for e in record['spy']['events']]
    assert not [p for p in opened if p.startswith(str(world.raw))]                                    # the stage arrays, not the NIfTI files


def test_hi_select_models_on_the_lpba_allowlist(world, monkeypatch):
    registry, _ = hi_world(world, monkeypatch)
    allowlist = load_dev_allowlist(world.allowlist_path)
    assert {s: v['policy'] for s, v in hi_lpba.select_models(registry, ['S05', 'S06', 'S01'], allowlist).items()} == dict(S05='all_folds_mean', S06='all_folds_mean', S01='all_folds_mean')
    dev = load_dev_allowlist(W.write_json(world.tmp/'dev.json', dict(lpba_document(world.HW.CONTACT), sources=lpba_document(world.HW.CONTACT)['sources'][:3],
                                                                     subjects=dict(OASIS=list(W.DEV)+list(world.HW.CONTACT), LPBA40=[f'S{n:02d}' for n in range(1, 25)]))))
    with pytest.raises(hc.Refused, match='SubjectSealed'):
        hi_lpba.select_models(registry, ['S05'], dev)
    with pytest.raises(SubjectSealed):
        G.check_lpba40_names(['LPBA40_S05__LPBA40_S06'], dev)


# ---- review I2 (C26.10 T16, this package's links on a pair of the 34): prep -> Atropos -> HI -> xb_core.check_pair_input, formal mode with a one-row table ---------
ONE = [('S05', 'S06')]


@pytest.fixture
def formal_one(world, monkeypatch):
    """The formal mode (lpba40_test) with its table cut to S05 <- S06 (lc.table), and rule one stubbed (formal_code_problem, rule_one_problem): the code identity of
    C26.9-4 is another package's; everything else -- the LPBA allowlist's exception, the roots, the spy -- is the real one."""
    real_table = lc.table
    monkeypatch.setattr(lc, 'table', lambda pair_set, lists=None: (['S05', 'S06'], list(ONE)) if pair_set == 'lpba40_test' else real_table(pair_set, lists))
    monkeypatch.setattr(lc, 'formal_code_problem', lambda code, pair_set, dry_run=False: None)
    monkeypatch.setattr(lc, 'rule_one_problem', lambda code: None)
    return world


def test_i2_one_pair_of_the_34_through_every_entry(formal_one, monkeypatch):
    world = formal_one
    rec = run_prep('lpba40_test')
    assert rec['passed'] and rec['stage_summary']['read_back']['pairs']['LPBA40_S05__LPBA40_S06']['lpba40_guard']['fixed'] == dict(admitted_by_lpba40=True)
    assert all(r['admitted_by_lpba40'] for r in rec['admission']['subjects'].values())
    fake_ants(monkeypatch)
    S._reset_cache()
    atropos = atr.run('lpba40_test', code=CODE, log=lambda text: None)
    assert atropos['passed'] and atropos['stage_precheck']['records']['LPBA40_S05__LPBA40_S06']['lpba40_guard']['moving'] == dict(admitted_by_lpba40=True)
    assert atropos['feasibility_stop']['applies'] is False and all(w['input_bound'] for w in atropos['wm'].values())
    registry, _ = hi_world(world, monkeypatch)
    S._reset_cache()
    hirec = hi_lpba.run('lpba40_test', code=CODE, device='cpu', log=lambda text: None)
    assert hirec['passed'] and sorted(hirec['rows']) == ['S05', 'S06']
    assert all(r['image']['pairs'][0]['lpba40_guard']['fixed']['admitted_by_lpba40'] for r in hirec['rows'].values())
    r = lc.roots('lpba40_test')
    allowlist = load_dev_allowlist(world.allowlist_path)
    pair = load_pair(r.stage/'LPBA40_S05__LPBA40_S06'/'pair.npz', allowlist=allowlist)
    assert pair.provenance['subject_guard']['lpba40_guard']['fixed']['admitted_by_lpba40']
    for side, subject in (('fixed', 'S05'), ('moving', 'S06')):
        structure = json.loads((r.pred_root/subject/'provenance.json').read_text(encoding='utf-8'))
        accepted = validate_segmentation_provenance(structure, pair, side=side, layer='structure', registry=registry, allowlist=allowlist)
        tissue = json.loads((r.atropos/subject/'provenance.json').read_text(encoding='utf-8'))
        accepted_t = validate_segmentation_provenance(tissue, pair, side=side, layer='tissue', allowlist=allowlist)
        assert list(accepted['input_sha256'].values()) == list(accepted_t['input_sha256'].values()) == [pair.source_sha256[side]]
    import xb_core
    record = xb_core.check_pair_input(r.stage/'LPBA40_S05__LPBA40_S06'/'pair.npz', fixed='S05', moving='S06', allowlist=allowlist)
    assert record['guard']['lpba40_guard']['moving'] == dict(admitted_by_lpba40=True)
    # the counter-example: the development allowlist is refused by name in load_pair, and by check_pair_subjects in check_pair_input
    dev = load_dev_allowlist(W.write_json(world.tmp/'dev_only.json', dict(lpba_document(world.HW.CONTACT), sources=lpba_document(world.HW.CONTACT)['sources'][:3],
                                                                          subjects=dict(OASIS=list(W.DEV)+list(world.HW.CONTACT), LPBA40=[f'S{n:02d}' for n in range(1, 25)]))))
    with pytest.raises(SubjectSealed, match='LPBA40_TEST_REJECT'):
        load_pair(r.stage/'LPBA40_S05__LPBA40_S06'/'pair.npz', allowlist=dev)
    with pytest.raises(hc.Refused, match='admission'):
        xb_core.check_pair_input(r.stage/'LPBA40_S05__LPBA40_S06'/'pair.npz', fixed='S05', moving='S06', allowlist=dev)


# ---- review minors 2-4 ------------------------------------------------------------------------------------------------------------------------------
def test_minor2_hi_reads_only_the_member_files_of_the_models_root(world, monkeypatch, tmp_path):
    models = tmp_path/'members_root'
    (models/'foldA_s0').mkdir(parents=True)
    monkeypatch.setattr(hi_lpba.hi, 'MODELS_ROOT', models)
    policy = hi_lpba.policy('lpba40_feas', lc.table('lpba40_feas')[1])
    assert policy.classify(models/'foldA_s0'/'member.pt', False)[0] == 'allowed'
    assert policy.classify_listing(models)[0] == 'allowed' and policy.classify_listing(models/'foldA_s0')[0] == 'allowed'
    for other in ('member.json', 'train_log.txt', 'eval.json'):
        assert policy.classify(models/'foldA_s0'/other, False)[0] != 'allowed', other
    assert policy.classify(models/'foldA_s0'/'member.pt', True)[0] != 'allowed'


def test_minor3_check_stage_wants_exactly_the_stages_files(world):
    run_prep()
    stage = lc.roots('lpba40_feas').stage
    pairs = lc.table('lpba40_feas')[1]
    assert lc.check_stage('lpba40_feas', pairs)['n_files'] == 2+2*2
    extra = stage/'LPBA40_S01__LPBA40_S02'/'extra.json'
    extra.write_text('{}')
    with pytest.raises(hc.Refused, match='exactly'):
        lc.check_stage('lpba40_feas', pairs)
    extra.unlink()
    copy = stage/lc.MM_DIR/'S01_image.nii.gz'
    copy.write_bytes(copy.read_bytes()+b'\0')
    with pytest.raises(hc.Refused, match='differs from SHA256SUMS'):
        lc.check_stage('lpba40_feas', pairs)
    assert lc.MM_DIR == prep.MM_DIR


def test_minor4_one_array_sha256():
    import he_core
    assert prep.array_sha256 is atr.array_sha256 is hi_lpba.array_sha256 is lc.array_sha256 is he_core.array_sha256
