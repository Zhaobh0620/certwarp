"""LP-C2 item 1 (contract C26.7, C26.10 T10, T11, T19, T16's evaluation link; prereg §2.2 last part, §5, §8.4, §12 items 4 (d), 12): bayesreg.eval.lpba_eval on the synthetic
world of _eval_world only (temporary code tree and git repository, tmp_path roots, tiny NIfTI files with LPBA40's kind of header).  No real file of a subject is opened."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import types

import nibabel as nib
import numpy as np
import pytest

import bayesreg.io.sealed as S
from bayesreg.eval import fields as F
from bayesreg.eval import holdout as H
from bayesreg.eval import holdout_decide as HD
from bayesreg.eval import lpba_decide as D
from bayesreg.eval import lpba_eval as E
from jd_curl.three_d.nifti import load_scalar

from . import _eval_world as EW

lc, hc, prep = EW.lc, EW.hc, EW.prep


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = EW.make(tmp_path, monkeypatch)
    yield w
    E.stop_spy()
    S._reset_cache()


def formal(world, **kwargs):
    rows = EW.build_stage(world, 'lpba40_test')
    EW.write_cells(world, 'lpba40_test', rows, **kwargs)
    return rows


def open_formal(world, name='run1', **kwargs):
    S._reset_cache()
    return E.open_lpba(E.eval_dir('lpba40_test')/name, 'lpba40_test', run=EW.no_units, repo=world.repo, **kwargs)


def spy_events():
    spy = E._LAST_SPY
    return [] if spy is None else spy.events


def labels_opened(events):
    return sorted({e['path'] for e in events if Path(e['path']).name in ('label.nii.gz', 'label_mm.nii.gz')})


# ---- the constants ---------------------------------------------------------------------------------------------------------------------------------------
def test_the_constants_are_the_preregistrations():
    import lpba_arm_table
    assert len(E.LPBA40_IDS) == 56 and len(set(E.LPBA40_IDS)) == 56
    assert set(E.LPBA40_IDS)-set(E.LPBA54_IDS) == {181, 182} and len(E.LPBA54_IDS) == 54
    assert E.LPBA40_IDS == tuple(range(21, 35))+tuple(range(41, 51))+tuple(range(61, 69))+tuple(range(81, 93))+(101, 102, 121, 122)+tuple(range(161, 167))+(181, 182)
    assert len(E.LR_PAIRS) == 27 and all(r == l+1 and l % 2 == 1 for l, r in E.LR_PAIRS) and {i for p in E.LR_PAIRS for i in p} == set(E.LPBA54_IDS)
    scheme = E.SCHEME
    assert scheme.primary == 'lpba56' and dict((m, ids) for m, _, ids in scheme.macro_sets)['lpba54'] == E.LPBA54_IDS
    assert E.PREREG_SHA256 == lc.PREREG_SHA256 and E.ARM_TABLE_SHA256 == lc.FROZEN_FILES[E.ARM_TABLE_REL]
    assert E.REAL_FORMAL_ROOT == lpba_arm_table.LPBA_ROOT
    assert E.BZ_WORDS == hc.LPBA40_BZ_WORDS
    assert E.OWN_UNITS['lpba40_test'] == (lc.unit_name('lpba40_test', 'eval'),) and E.OWN_UNITS['lpba40_feas'] == (lc.unit_name('lpba40_feas', 'eval'),)
    assert all(u.startswith('tag-lp-') for units in E.OWN_UNITS.values() for u in units)
    log = (EW.ROOT/'docs'/'decision_log.md').read_text(encoding='utf-8').splitlines()
    assert E.BZ_WORDS in log[519] and log[519].startswith('| 2026-11-19 | **BZ：24 四面体代理选 (乙)')       # decision_log line 520, word for word


# ---- T10: the label and the image through the same transform (prereg §12 item 4 (d)) -----------------------------------------------------------------------
def test_t10_label_and_image_correspond_voxel_by_voxel(tmp_path, monkeypatch):
    """An LAS default affine and units unknown: the image through prep_lpba's mm copy, load_scalar and np.flip(., 2); the label through the entry's mm copy, load_scalar
    (labels=True) and the same flip: voxel by voxel the image is 10 x label + 5.  (The admission check of _label_volume is tested in the world; here it is stood in.)"""
    lab = EW.label_data('S05')
    monkeypatch.setattr(E, '_check_admission', lambda admission, subject, pair_name: None)
    for affine in (EW.AFFINE, np.diag([-1., 1., 1., 1.])):            # LPBA40's kind of header: x to the left (LAS), qform and sform coded, units unknown
        img = EW.las(lab*10.+5., affine=affine)
        raw = tmp_path/f'raw{int(affine[0, 3])}'
        monkeypatch.setattr(E, 'RAW_ROOT', raw)
        src_i, src_l = tmp_path/f'image{int(affine[0, 3])}.nii.gz', raw/'S05'/E.LABEL_REL
        src_l.parent.mkdir(parents=True)
        nib.save(img, str(src_i))
        nib.save(EW.las(lab, dtype=np.int16, affine=affine), str(src_l))
        copy = prep.mm_copy(src_i, tmp_path/f'mm{int(affine[0, 3])}'/'image.nii.gz')
        v, _ = prep.stage_image(copy['mm_copy']['path'])
        folder = tmp_path/f'priv{int(affine[0, 3])}'
        folder.mkdir()
        volume, (aff, shape), _ = E._label_volume(src_l, folder, admission=None, subject='S05', pair_name='LPBA40_S05__LPBA40_S06')
        labels = np.flip(np.asarray(volume.values), 2).copy()
        assert labels.shape == v.shape == EW.STORED and shape == EW.SHAPE_XYZ and np.array_equal(aff, nib.load(str(src_i)).affine)
        assert np.array_equal(v, labels*10.+5.)
        lr = E.left_right_record(labels)
        known = [p for p in lr['pairs'] if p['sign_left_minus_right'] is not None]
        assert {(p['left'], p['right']) for p in known} == {(21, 22), (41, 42)} and all(p['sign_left_minus_right'] == -1 for p in known)
        assert lr['n_expected_sign'] == lr['n_known'] == 2 and lr['expected_sign'] == -1
    mirrored = np.flip(labels, 2)
    assert all(p['sign_left_minus_right'] == 1 for p in E.left_right_record(mirrored)['pairs'] if p['sign_left_minus_right'] is not None)


def test_t10_load_scalar_needs_the_mm_copy(tmp_path):
    nib.save(EW.las(EW.label_data('S05'), dtype=np.int16), str(tmp_path/'label.nii.gz'))
    with pytest.raises(Exception):
        load_scalar(tmp_path/'label.nii.gz', labels=True)                # units unknown: the reason for the mm copy (prereg §2.2 P1)


def test_t11_id_record():
    values = np.zeros((4, 4, 4))
    values[0, 0, :2] = 21
    values[1, 1, 1] = 7
    values[2, 2, 2] = 300
    record = E.id_record(values)
    assert record['n_missing'] == 55 and 21 not in record['missing_ids'] and record['unexpected_values'] == {'7': 1, '300': 1} and record['n_unexpected_voxels'] == 2


# ---- T11 and T19: the formal evaluation end to end ----------------------------------------------------------------------------------------------------------------
def test_t11_formal_end_to_end_reads_labels_once_after_every_check_and_keeps_no_truth_copy(world):
    rows = formal(world, resumed=('LPBA40_S05__LPBA40_S06',), suspicious={('ants_s', 'LPBA40_S07__LPBA40_S08')})
    EW.write_manifest(world, 'lpba40_test', rows)
    admission, context = open_formal(world)
    before = spy_events()
    assert labels_opened(before) == []                                               # nothing of the truth before the locks
    assert E.master_lock('lpba40_test').is_file()
    line = H.lock_lines(E.master_lock('lpba40_test'))[0]
    assert line['prereg_sha256'] == E.PREREG_SHA256 and line['arm_table_sha256'] == E.ARM_TABLE_SHA256 and line['pinned_commit'] == 'p'*40 and line['pair_set'] == 'lpba40_test'
    assert all(r['admitted_by_lpba40'] for r in context['record']['admission']['subjects'].values())
    code = E.evaluate_lpba(admission, context)
    out = context['out']
    assert code == 0, (out/'stopped.json').read_text() if (out/'stopped.json').exists() else code
    records = {p.stem: json.loads(p.read_text()) for p in (out/'pairs').glob('*.json')}
    assert len(records) == 17 and all(r['complete'] and r['truth_read'] for r in records.values())
    summary = json.loads((out/'lpba_eval.json').read_text())
    events = summary['opened_files']['events']
    opened = labels_opened(events)
    assert {Path(p).parent.parent.name for p in opened if Path(p).name == 'label.nii.gz'} == {f'S{n:02d}' for n in range(3, 37)}
    assert all(e['phase'] == 'evaluation' and e['kind'] == 'truth' for e in events if Path(e['path']).name in ('label.nii.gz', 'label_mm.nii.gz'))
    assert summary['opened_files']['refused'] == []
    assert not any('label' in p.name for p in world.results.rglob('*') if p.is_file()), 'no truth copy under the results root'
    assert list(world.tmp.iterdir()) == [], 'the private temporary directories are removed'
    rec = records['LPBA40_S03__LPBA40_S04']
    raw = world.raw/'S03'/'delineation'/'label.nii.gz'
    assert rec['truth']['fixed']['sha256'] == hashlib.sha256(raw.read_bytes()).hexdigest() and rec['truth']['fixed']['copy']['removed'] is True
    assert rec['truth']['fixed']['left_right']['n_expected_sign'] == rec['truth']['fixed']['left_right']['n_known'] == 2
    assert rec['truth']['fixed']['ids']['n_missing'] == 51 and rec['truth']['fixed']['ids']['unexpected_values'] == {}       # 21, 22, 41, 42, 181 present
    for arm in ('method', 'configc', 'ants_s', 'fireants_greedy', 'fireants_syn', 'convexadam_v4seg'):
        entry = rec['arms'][arm]
        assert entry['outcome'] == 'output' and {'lpba54', 'lpba56', 'dice_official', 'hd95', 'sdlogj_official', 'sdlogj_brain'} <= set(entry['metrics'])
        cert = entry['certificate']
        assert cert['mode'] == 'jacobian_only' and cert['configuration'] == 'BL1' and 'faces_max_abs_u_mm' in cert and 'domain' in cert
        assert entry['tet24']['u_sha256_equals_bl1_state_hash'] is True
    assert 'decision_certificate' in rec['arms']['method'] and rec['arms']['method']['decision_certificate']['certificate_state_hash_matches_u'] is True
    assert rec['identity']['metrics']['sdlogj_official'] == 0. and rec['identity']['certificate'] is None
    assert records['LPBA40_S07__LPBA40_S08']['arms']['ants_s']['export_suspicious'] is True
    assert rec['brain']['nodes'] > 0
    # T19: configuration C's output_store, the first run's and a resumed cell's under <pair>.resume
    resumed = records['LPBA40_S05__LPBA40_S06']['arms']['configc']
    assert resumed['terminal'].endswith('result_resume.json') and '/work/configc/LPBA40_S05__LPBA40_S06.resume/output' in resumed['field']['disp']['path']
    assert '/work/configc/LPBA40_S03__LPBA40_S04/output' in rec['arms']['configc']['field']['disp']['path']
    # the 24 tetrahedra: the same as calling tet_ratio_summary directly on u = h d of the evaluated field
    from bayesreg.eval.holdout_tet24 import folding_module
    d, _ = F.read_disp(rec['arms']['ants_s']['field']['disp']['path'])
    u = F.d_to_u(d, (1., 1., 1.))
    direct = folding_module().tet_ratio_summary(u, (1., 1., 1.))
    assert json.dumps(direct, sort_keys=True) == json.dumps(rec['arms']['ants_s']['tet24']['tet_ratio_summary'], sort_keys=True)
    assert rec['arms']['ants_s']['certificate']['state_hash'] == F.array_sha256(u)
    # the decision code reads these records through the lock lines
    found = HD.collect([line['out_dir']], [r for r in E.pair_rows('lpba40_test', lc.read_lists())])
    table, _ = E.load_arm_table('lpba40_test', world.repo)
    decision = D.decide(found, table, E.pair_rows('lpba40_test', lc.read_lists()))
    assert decision['n_used'] == 17 and set(decision['L1']['arms']) == set(D.MAIN_ORDER)
    # once: a second run is refused before anything is read
    with pytest.raises(H.HoldoutRefused, match='once'):
        open_formal(world, 'run2')


def test_t11_every_condition_before_the_labels_refuses(world, monkeypatch):
    rows = formal(world)
    table, _ = E.load_arm_table('lpba40_test', world.repo)
    # no committed manifest
    with pytest.raises(H.HoldoutRefused, match='manifest'):
        open_formal(world)
    assert labels_opened(spy_events()) == []
    # a manifest without a line the entry reads
    roles = EW.write_manifest(world, 'lpba40_test', rows, drop=(str(table['stage_root'])+'/LPBA40_S03__LPBA40_S04/pair.npz',))
    with pytest.raises(H.HoldoutRefused, match='not in the manifest'):
        open_formal(world)
    # a line outside the files read
    EW.write_manifest(world, 'lpba40_test', rows, extra=f"{'0'*64}  {world.raw}/S03/delineation/label.nii.gz")
    with pytest.raises(H.HoldoutRefused, match='outside the files this entry reads'):
        open_formal(world)
    EW.write_manifest(world, 'lpba40_test', rows)
    # an uncommitted change of the manifest
    (world.repo/E.MANIFEST_REL).write_text((world.repo/E.MANIFEST_REL).read_text()+'\n', encoding='utf-8')
    with pytest.raises(H.HoldoutRefused, match='manifest'):
        open_formal(world)
    EW.W.git(world.repo, 'checkout', '--', E.MANIFEST_REL)
    # another unit running
    busy = lambda args, **kw: subprocess.CompletedProcess(args, 0, 'tag-hb-tet24.service loaded active running x\n', '')
    with pytest.raises(H.HoldoutRefused, match='machine'):
        S._reset_cache()
        E.open_lpba(E.eval_dir('lpba40_test')/'x', 'lpba40_test', run=busy, repo=world.repo)
    # a cell not written
    method = Path(table['arms'][0]['record_dir'].format(pair='LPBA40_S35__LPBA40_S36'))
    (method/'terminal.json').write_text(json.dumps(dict(line_status='hx_provenance_rejected')))
    with pytest.raises(H.HoldoutRefused, match='not written'):
        open_formal(world)
    (method/'terminal.json').write_text(json.dumps(dict(line_status='hx_certified')))
    EW.write_finalized('lpba40_test', rows, table)                        # the terminal record's bytes changed: the finalized record names them again (decision_log line 584 ruling 5)
    EW.write_manifest(world, 'lpba40_test', rows)
    # the out dir outside eval/
    with pytest.raises(H.HoldoutRefused, match='output'):
        S._reset_cache()
        E.open_lpba(world.results/'elsewhere', 'lpba40_test', run=EW.no_units, repo=world.repo)
    # the master lock there
    E.master_lock('lpba40_test').parent.mkdir(parents=True, exist_ok=True)
    E.master_lock('lpba40_test').write_text('{}\n')
    with pytest.raises(H.HoldoutRefused, match='once'):
        open_formal(world)
    assert labels_opened(spy_events()) == []
    assert not (E.eval_dir('lpba40_test')/'run1').exists()


def test_t11_the_code_identity_and_the_pins_come_first(world, monkeypatch):
    formal(world)
    def refuse(pair_set, repo=None):
        raise H.HoldoutRefused('code_identity', 'test: rule one does not hold')
    monkeypatch.setattr(E, 'check_code_identity', refuse)
    with pytest.raises(H.HoldoutRefused, match='code_identity'):
        open_formal(world)
    assert spy_events() == [] or all(e['kind'] == 'sealed_list' for e in spy_events())
    monkeypatch.setattr(E, 'check_code_identity', lambda pair_set, repo=None: dict(rule_one=dict(pinned='p'*40), code=dict(head='h'*40, clean=True, status='')))
    monkeypatch.setattr(lc, 'FROZEN_FILES', dict(lc.FROZEN_FILES, **{'experiments/d0/configC_reference.json': '0'*64}))
    with pytest.raises(H.HoldoutRefused, match='admission'):
        open_formal(world)
    assert not any(Path(e['path']).is_relative_to(world.raw) for e in spy_events())


def test_t11_a_label_off_the_images_grid_stops_before_any_dice(world):
    rows = formal(world)
    bad = world.raw/'S05'/'delineation'/'label.nii.gz'
    shifted = EW.AFFINE.copy()
    shifted[0, 3] += 2.
    nib.save(EW.las(EW.label_data('S05'), dtype=np.int16, affine=shifted), str(bad))
    EW.write_manifest(world, 'lpba40_test', rows)
    admission, context = open_formal(world)
    assert E.evaluate_lpba(admission, context) == 4
    out = context['out']
    stopped = json.loads((out/'stopped.json').read_text())
    assert stopped['pair'] == 'LPBA40_S05__LPBA40_S06' and 'TruthMismatch' in stopped['reason']
    partial = json.loads((out/'pairs'/'LPBA40_S05__LPBA40_S06.json').read_text())
    assert partial['complete'] is False and partial['arms'] == {} and 'identity' not in partial
    assert (out/'pairs'/'LPBA40_S03__LPBA40_S04.json').is_file() and not (out/'pairs'/'LPBA40_S07__LPBA40_S08.json').exists()
    assert list(world.tmp.iterdir()) == []


def test_t11_label_volume_has_its_own_guard(world, tmp_path):
    """Review minor 6: _label_volume refuses without the issued admission (and its locks), whoever calls it."""
    folder = tmp_path/'priv'
    folder.mkdir()
    with pytest.raises(H.HoldoutRefused, match='not an admission'):
        E._label_volume(world.raw/'S03'/E.LABEL_REL, folder, admission=None, subject='S03', pair_name='LPBA40_S03__LPBA40_S04')
    assert list(folder.iterdir()) == []


@pytest.mark.parametrize('values', ['half', 'negative', 'nan'])
def test_t11_label_values_not_finite_non_negative_integers_stop_before_any_dice(world, values):
    """Review minor 7: a label map of non-integers, negatives or non-finite values stops the evaluation at that subject (TruthMismatch, exit 4), before any Dice."""
    rows = formal(world)
    lab = EW.label_data('S05').astype(np.float32)
    lab[5, 5, 5] = {'half': 21.5, 'negative': -1., 'nan': np.nan}[values]
    nib.save(EW.las(lab, dtype=np.float32), str(world.raw/'S05'/E.LABEL_REL))
    EW.write_manifest(world, 'lpba40_test', rows)
    admission, context = open_formal(world)
    assert E.evaluate_lpba(admission, context) == 4
    stopped = json.loads((context['out']/'stopped.json').read_text())
    assert stopped['pair'] == 'LPBA40_S05__LPBA40_S06' and 'TruthMismatch' in stopped['reason']
    assert json.loads((context['out']/'pairs'/'LPBA40_S05__LPBA40_S06.json').read_text())['arms'] == {}
    assert list(world.tmp.iterdir()) == []


def test_t11_read_label_refuses_without_its_admission_and_pair_lock(world):
    rows = formal(world)
    EW.write_manifest(world, 'lpba40_test', rows)
    admission, context = open_formal(world)
    meta = context['grids']['LPBA40_S03__LPBA40_S04']
    kwargs = dict(grid_metadata=meta, pair_name='LPBA40_S03__LPBA40_S04', image=context['originals']['S03'], listed=context['listed'])
    with pytest.raises(H.HoldoutRefused, match='pair lock'):
        E.read_label('S03', admission=admission, **kwargs)
    forged = E.Admission(admission.mode, admission.subjects, admission.pairs, admission.record)
    with pytest.raises(H.HoldoutRefused, match='not an admission'):
        E.read_label('S03', admission=forged, **kwargs)
    E.take_pair_lock(admission, rows[0], context['line'])
    with pytest.raises(H.HoldoutRefused, match='not a subject'):
        E.read_label('S05', admission=admission, **kwargs)
    values, record = E.read_label('S03', admission=admission, **kwargs)
    assert values.shape == EW.STORED and record['affine_equal_to_image'] is True
    E.stop_spy()


def test_t11_reevaluation_authorization(world):
    rows = formal(world)
    EW.write_manifest(world, 'lpba40_test', rows)
    admission, context = open_formal(world)
    E.stop_spy()
    master_sha = hashlib.sha256(E.master_lock('lpba40_test').read_bytes()).hexdigest()
    manifest_sha = context['line']['manifest_sha256']

    def authorize(k, **changes):
        doc = dict(schema=E.SCHEMA_REEVAL, written_by="lead on BZ's behalf", words=E.BZ_WORDS, time_local='2026-11-20 1:00 AM CST', utc=H.utc(),
                   reason='the evaluation unit was killed (test)', reason_class='evaluation_killed', master_lock_sha256=master_sha, manifest_sha256=manifest_sha,
                   pairs=['LPBA40_S03__LPBA40_S04'], previous_out=[str(context['out'])])
        doc.update(changes)
        path = EW.write_json(world.repo/E.REEVAL_DIR_REL/f'lpba40_test_v1_reeval_{k}.json', doc)
        EW.W.commit_all(world.repo, f'reeval {k}')
        return path
    ok = authorize(1)
    n, doc, record = E.load_reeval(ok, manifest_sha256=manifest_sha, rows=rows, repo=world.repo)
    assert n == 1 and doc['pairs'] == ['LPBA40_S03__LPBA40_S04']
    for k, changes, match in ((2, dict(words='预登记写好'), 'word for word'), (3, dict(written_by='lead'), 'written_by'), (4, dict(reason_class='method_failure'), 'reason_class'),
                              (5, dict(manifest_sha256='0'*64), 'manifest')):
        with pytest.raises(H.HoldoutRefused, match=match):
            E.load_reeval(authorize(k, pairs=['LPBA40_S07__LPBA40_S08'], **changes), manifest_sha256=manifest_sha, rows=rows, repo=world.repo)
    with pytest.raises(H.HoldoutRefused, match='reevaluation authorization is'):
        E.load_reeval(world.repo/'splits'/'lpba40_test_v1'/'holdout_B_v1_reeval_1.json', manifest_sha256=manifest_sha, rows=rows, repo=world.repo)


# ---- T11: the no-truth mode (the feasibility pairs; C26.7-6) ---------------------------------------------------------------------------------------------------------
def test_t11_the_no_truth_mode_opens_no_label(world):
    rows = EW.build_stage(world, 'lpba40_feas')
    EW.write_cells(world, 'lpba40_feas', rows)
    S._reset_cache()
    admission, context = E.open_lpba(E.eval_dir('lpba40_feas')/'feas1', 'lpba40_feas', run=EW.no_units, repo=world.repo)
    assert context['record']['manifest']['path'] is None and context['record']['truth'] is False
    assert not E.master_lock('lpba40_test').exists() and E.master_lock('lpba40_feas').is_file()
    with pytest.raises(H.HoldoutRefused, match='no-truth'):
        E.read_label('S01', admission=admission, grid_metadata=context['grids'][rows[0][3]], pair_name=rows[0][3], image=context['originals']['S01'], listed=context['listed'])
    assert E.evaluate_lpba(admission, context) == 0
    summary = json.loads((context['out']/'lpba_eval.json').read_text())
    assert summary['label_files_opened'] == [] and labels_opened(summary['opened_files']['events']) == [] and summary['truth_mode'] is False
    rec = json.loads((context['out']/'pairs'/f'{rows[0][3]}.json').read_text())
    assert rec['truth_read'] is False and 'truth' not in rec
    entry = rec['arms']['method']
    assert {'sdlogj_official', 'central_det_nonpositive', 'central_det_nonpositive_brain', 'sdlogj_brain', 'logj_q01_brain'} <= set(entry['metrics'])
    assert 'lpba54' not in entry['metrics'] and entry['official'] is None and entry['certificate']['configuration'] == 'BL1' and entry['tet24'] is not None
    assert all(Path(e['path']).name != 'label.nii.gz' for e in summary['opened_files']['events'])
    assert '/lpba40_feas_v1/' in entry['field']['disp']['path']


def test_t11_the_no_truth_mode_spy_refuses_a_label(world, monkeypatch):
    rows = EW.build_stage(world, 'lpba40_feas')
    EW.write_cells(world, 'lpba40_feas', rows)
    S._reset_cache()
    admission, context = E.open_lpba(E.eval_dir('lpba40_feas')/'feas2', 'lpba40_feas', run=EW.no_units, repo=world.repo)
    original = E.score_d

    def nosy(d, grid, truth, brain, **kwargs):
        open(world.raw/'S01'/'delineation'/'label.nii.gz', 'rb').close()
        return original(d, grid, truth, brain, **kwargs)
    monkeypatch.setattr(E, 'score_d', nosy)
    assert E.evaluate_lpba(admission, context) == 4
    stopped = json.loads((context['out']/'stopped.json').read_text())
    assert 'SpyRefused' in stopped['reason']


# ---- T16 (the evaluation link) and the cells --------------------------------------------------------------------------------------------------------------------
def test_t16_the_evaluation_admission_of_a_test_pair_on_the_lpba_allowlist(world):
    admitted = E._admit('lpba40_test')
    assert admitted.record['subjects']['S05']['admitted_by_lpba40'] is True and admitted.record['subjects']['S06']['admitted_by_lpba40'] is True
    with pytest.raises(H.HoldoutRefused, match='admission'):
        E._admit('holdout')


def test_t19_classify_configc_output_store_and_its_place(world):
    rows = formal(world, resumed=('LPBA40_S09__LPBA40_S10',))
    table, _ = E.load_arm_table('lpba40_test', world.repo)
    arm = next(a for a in table['arms'] if a['name'] == 'configc')
    first = E.classify_cell(arm, rows[0], repo=world.repo)
    assert first.written and first.outcome == 'output' and first.field_dir.endswith('/work/configc/LPBA40_S03__LPBA40_S04/output')
    resumed = E.classify_cell(arm, rows[3], repo=world.repo)
    assert resumed.written and resumed.field_dir.endswith('/work/configc/LPBA40_S09__LPBA40_S10.resume/output') and resumed.terminal.endswith('result_resume.json')
    base = Path(arm['record_dir'].format(pair=rows[1][3], arm='configc'))
    doc = json.loads((base/'result.json').read_text())
    elsewhere = world.results/'elsewhere'/'disp.npz'
    elsewhere.parent.mkdir()
    np.savez(elsewhere, d=np.zeros((3,)+EW.STORED))
    doc['output_store']['disp'] = dict(file=str(elsewhere), sha256=EW.sha(elsewhere))
    (base/'result.json').write_text(json.dumps(doc))
    off = E.classify_cell(arm, rows[1], repo=world.repo)
    assert not off.written and 'a disp.npz under' in off.problem


def test_the_cells_needing_authorizations_follow_holdouts_rules(world):
    rows = formal(world)
    table, _ = E.load_arm_table('lpba40_test', world.repo)
    arm = next(a for a in table['arms'] if a['name'] == 'ants_s')
    base = Path(arm['record_dir'].format(pair=rows[0][3], arm='ants_s'))
    (base/'result.json').write_text(json.dumps(dict(status='infra_failure')))
    cell = E.classify_cell(arm, rows[0], repo=world.repo)
    assert not cell.written and 'infrastructure' in cell.problem
    record = base/'result.json'
    doc = dict(schema='bayesreg.xb.no_resume.v1', written_by="lead on BZ's behalf", words=hc.LPBA40_BZ_WORDS, time_local='2026-11-20 1:00 AM CST', utc=H.utc(),
               reason='未批补跑 (test)', tool='xb', arm='ants_s', pair=rows[0][3], pair_set='lpba40_test', record=str(record), record_sha256=EW.sha(record), note=hc.NO_RESUME_NOTE)
    EW.write_json(world.repo/'splits'/'lpba40_test_v1'/'lpba40_test_v1_xb_no_resume_1.json', doc)
    EW.W.commit_all(world.repo, 'no_resume (test)')
    cell = E.classify_cell(arm, rows[0], repo=world.repo)
    assert cell.written and cell.no_output_class == 'infrastructure' and cell.authorization['kind'] == 'no_resume'
    # the P19 stop: two suspicious pairs (LPBA40) and XB's own record
    other = next(a for a in table['arms'] if a['name'] == 'fireants_syn')
    base = Path(other['record_dir'].format(pair=rows[5][3], arm='fireants_syn'))
    for name in ('result.json',):
        (base/name).unlink()
    xb = Path(H.template_root(other['record_dir'].replace('{arm}', 'fireants_syn')))/H.XB_STOP_NAME
    fields = dict(arm='fireants_syn', pair_set='lpba40_test', utc=H.utc(), suspicious_pairs=[rows[1][3], rows[2][3]], at_pair=rows[2][3], rule='P19_STOP_AT 2')
    EW.write_json(xb, dict(schema=H.XB_STOP_SCHEMA, **fields))
    stop = dict(schema=H.SCHEMA_STOP, arm='fireants_syn', written_by='lead', unrun_pairs=[rows[5][3]], arm_stopped=dict(path=str(xb), sha256=EW.sha(xb)), **{k: v for k, v in fields.items() if k != 'arm'})
    EW.write_json(world.repo/'runs'/'lpba40_test_v1'/'arm_stop_fireants_syn.json', stop)
    EW.W.commit_all(world.repo, 'stop (test)')
    cell = E.classify_cell(other, rows[5], repo=world.repo)
    assert cell.written and cell.no_output_class == 'arm_stopped', cell.problem


def test_relocate_maps_only_the_formal_root(world):
    table, record = E.load_arm_table('lpba40_feas', world.repo)
    text = json.dumps({k: v for k, v in table.items() if k != 'comment'})          # the comment's prose names the formal root as text, not as a place
    assert '/lpba40_test_v1/' not in text.replace('runs/lpba40_test_v1/', '') and str(world.results/'lpba40_feas_v1') in text
    assert table['arms'][0]['stop_record'] == 'runs/lpba40_test_v1/arm_stop_{arm}.json'
    assert record['sha256'] == E.ARM_TABLE_SHA256


def test_the_dry_run_reads_only_the_code_tree(world, tmp_path, capsys):
    out = tmp_path/'dry'
    assert E.dry_run(out, 'lpba40_test', repo=world.repo) == 0
    report = json.loads((out/'dry_run.json').read_text())
    assert report['truth_read'] is False and report['first_refusal'] == 'step-10 manifest committed'
    assert report['conditions']['admission (lpba_common.admit: pins, names, check_subject)'] == 'passed'


# ---- review I3: the spy watches the whole data root, data/three_d and the results root's holdout_B_v1 ----------------------------------------------------------------------
def nosy_world(world, monkeypatch, tmp_path):
    """Files under the three places the spy now watches (stand-ins in tmp_path: E.DATA_ROOT, E.THREE_D_ROOTS; <RESULTS>/holdout_B_v1 is the world's)."""
    data, three_d = tmp_path/'mnt_d_data', tmp_path/'repo_data'/'three_d'
    monkeypatch.setattr(E, 'DATA_ROOT', data)
    monkeypatch.setattr(E, 'THREE_D_ROOTS', (three_d,))
    files = dict(data=data/'OASIS_OAS1_0960_MR1'/'aligned_norm.nii.gz', three_d=three_d/'stage_LPBA40_v1'/'LPBA40_S03__LPBA40_S04'/'manifest.json',
                 holdout=world.results/'holdout_B_v1'/'eval'/'holdout_eval.json')
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n')
    return files


@pytest.mark.parametrize('place', ['data', 'three_d', 'holdout'])
def test_i3_the_formal_mode_refuses_an_open_of_another_place(world, monkeypatch, tmp_path, place):
    files = nosy_world(world, monkeypatch, tmp_path)
    rows = formal(world)
    EW.write_manifest(world, 'lpba40_test', rows)
    admission, context = open_formal(world)
    original = E.score_d

    def nosy(d, grid, truth, brain, **kwargs):
        open(files[place], 'rb').close()
        return original(d, grid, truth, brain, **kwargs)
    monkeypatch.setattr(E, 'score_d', nosy)
    assert E.evaluate_lpba(admission, context) == 4
    stopped = json.loads((context['out']/'stopped.json').read_text())
    assert 'SpyRefused' in stopped['reason'] and str(files[place]) in json.dumps(stopped) or 'SpyRefused' in stopped['reason']
    summary = json.loads((context['out']/'lpba_eval.json').read_text())
    assert any(e['path'] == str(files[place].resolve()) for e in summary['opened_files']['events'])


@pytest.mark.parametrize('place', ['data', 'three_d', 'holdout'])
def test_i3_the_no_truth_mode_refuses_an_open_of_another_place(world, monkeypatch, tmp_path, place):
    files = nosy_world(world, monkeypatch, tmp_path)
    rows = EW.build_stage(world, 'lpba40_feas')
    EW.write_cells(world, 'lpba40_feas', rows)
    S._reset_cache()
    admission, context = E.open_lpba(E.eval_dir('lpba40_feas')/f'feas_{place}', 'lpba40_feas', run=EW.no_units, repo=world.repo)
    original = E.score_d

    def nosy(d, grid, truth, brain, **kwargs):
        open(files[place], 'rb').close()
        return original(d, grid, truth, brain, **kwargs)
    monkeypatch.setattr(E, 'score_d', nosy)
    assert E.evaluate_lpba(admission, context) == 4
    assert 'SpyRefused' in json.loads((context['out']/'stopped.json').read_text())['reason']


def test_i3_an_open_there_before_the_list_refuses_the_admission(world, monkeypatch, tmp_path):
    """An open under a watched place during the admission (before the list is allowed) refuses the run before its lock (HoldoutRefused 'spy')."""
    files = nosy_world(world, monkeypatch, tmp_path)
    rows = EW.build_stage(world, 'lpba40_feas')
    EW.write_cells(world, 'lpba40_feas', rows)
    original = E.check_cells

    def nosy(table, rows, **kwargs):
        open(files['holdout'], 'rb').close()
        return original(table, rows, **kwargs)
    monkeypatch.setattr(E, 'check_cells', nosy)
    S._reset_cache()
    with pytest.raises(H.HoldoutRefused, match='spy'):
        E.open_lpba(E.eval_dir('lpba40_feas')/'feas_x', 'lpba40_feas', run=EW.no_units, repo=world.repo)
    assert not E.master_lock('lpba40_feas').exists()
    assert set(map(str, E.spy_roots())) >= {str(E.DATA_ROOT), str(world.results/'holdout_B_v1')}


def test_minor5_the_feasibility_lock_is_refused_first(world, monkeypatch):
    E.master_lock('lpba40_feas').parent.mkdir(parents=True)
    E.master_lock('lpba40_feas').write_text('{}\n')
    monkeypatch.setattr(E, 'load_arm_table', lambda *a, **k: (_ for _ in ()).throw(AssertionError('read after the lock was found')))
    S._reset_cache()
    with pytest.raises(H.HoldoutRefused, match='once'):
        E.open_lpba(E.eval_dir('lpba40_feas')/'feas_y', 'lpba40_feas', run=EW.no_units, repo=world.repo)


def test_minor9_tet24_checks_the_schemes_and_keys(monkeypatch):
    import bayesreg.eval.holdout_tet24 as T
    u = np.zeros((6, 7, 8, 3))
    good = E.tet24(u, (1., 1., 1.), F.array_sha256(u))
    assert sorted(good['tet_ratio_summary']) == ['tet24', 'tet5', 'tet6']
    monkeypatch.setattr(T, 'folding_module', lambda repo=None: types.SimpleNamespace(tet_ratio_summary=lambda u, spacing: dict(tet24=dict(min_ratio=1.))))
    with pytest.raises(RuntimeError, match='not the schemes'):
        E.tet24(u, (1., 1., 1.), F.array_sha256(u))


# ---- decision_log line 584 ruling 5: the method cell against LP-B's finalized record --------------------------------------------------------------------------------------
@pytest.mark.parametrize('case', ['missing', 'pair_left_out', 'sha_differs', 'line_status', 'other_cell', 'review_row', 'schema'])
def test_ruling5_the_method_cell_is_crossed_with_the_finalized_record(world, case):
    rows = formal(world)
    table, _ = E.load_arm_table('lpba40_test', world.repo)
    arm = next(a for a in table['arms'] if a['name'] == 'method')
    row = rows[2]
    assert E.classify_cell(arm, row, repo=world.repo).written                   # as written: crossed and pinned
    cell = Path(arm['record_dir'].format(pair=row[3]))
    if case == 'missing':
        E.finalized_path('lpba40_test').unlink()
    elif case == 'pair_left_out':
        EW.write_finalized('lpba40_test', rows, table, drop=(row[3],))
    elif case == 'sha_differs':                                                   # a record the entry reads (terminal.json) at another sha256 than the finalized record's
        EW.write_finalized('lpba40_test', rows, table, changes={row[3]: dict(terminal=dict(path=str(cell/'terminal.json'), sha256='0'*64))})
    elif case == 'line_status':
        EW.write_finalized('lpba40_test', rows, table, changes={row[3]: dict(line_status='hx_fallback_configC')})
    elif case == 'other_cell':
        EW.write_finalized('lpba40_test', rows, table, changes={row[3]: dict(cell=str(cell)+'_moved')})
    elif case == 'review_row':
        EW.write_finalized('lpba40_test', rows, table, changes={row[3]: dict(review=dict(path=str(cell/'review'/'review.json'), sha256='1'*64))})
    else:
        doc = json.loads(E.finalized_path('lpba40_test').read_text())
        E.finalized_path('lpba40_test').write_text(json.dumps(dict(doc, pair_set='lpba40_feas')))
    got = E.classify_cell(arm, row, repo=world.repo)
    assert not got.written and 'finalized' in got.problem, got.problem
    with pytest.raises(H.HoldoutRefused, match='finalized'):
        E.check_cells(table, rows, repo=world.repo)


def test_ruling5_the_finalized_sha256_go_to_the_list(world):
    rows = formal(world)
    table, _ = E.load_arm_table('lpba40_test', world.repo)
    cells = E.check_cells(table, rows, repo=world.repo)
    cell = cells[('method', rows[0][3])]
    assert set(cell.pinned_sha256) >= {str(Path(cell.field_dir)/'disp.npz'), cell.terminal} and str(E.finalized_path('lpba40_test')) in map(str, cell.needed)
    assert E.input_roles(table, rows, cells, E.image_originals(table, rows))[str(E.finalized_path('lpba40_test'))] == 'cell'
    # an output file (not read at the admission) whose finalized sha256 is not the file's: the list (the files' own sha256) refuses it
    method = Path(table['arms'][0]['record_dir'].format(pair=rows[0][3]))
    EW.write_finalized('lpba40_test', rows, table, changes={rows[0][3]: dict(part=dict(output={'output/final.npz': '0'*64, 'output/disp.npz': EW.sha(method/'output'/'disp.npz'),
                                                                                                'output/cert.json': EW.sha(method/'output'/'cert.json'),
                                                                                                'output/line.json': EW.sha(method/'output'/'line.json')}))})
    EW.write_manifest(world, 'lpba40_test', rows)                               # the list holds the files' own sha256
    cells = E.check_cells(table, rows, repo=world.repo)
    with pytest.raises(H.HoldoutRefused, match='finalized record names'):
        E.check_manifest(table, rows, cells, E.image_originals(table, rows), prereg_sha256=E.PREREG_SHA256, arm_table_sha256=E.ARM_TABLE_SHA256, repo=world.repo)
