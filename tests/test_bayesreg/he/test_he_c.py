"""HE2 (runs/20261107-he2/任务书.md item 4; contract C22.B1, C22.B4, D83, D86): the tissue-dependent geometry prior C of the HE entry, experiments/he/he_c.py.  CPU only, synthetic worlds in a temporary
directory (tests/test_bayesreg/he/_world.py); no real image, label, prediction or field is read; nothing starts a unit.

- the three passes and the tissue layer against run_p2's own read_tissue, tissue_levels and run_field (arm C), bit for bit, and C off (one optimize_k) against pass 0 (a subprocess with the frozen tree);
- the constants of the frozen settings against p2_core's and bayesreg.seg.channels' own;
- D86, each class with an example on the real functions: (1) the input (the tissue layer's provenance, the class names), (2) the implementation (a coarse grid), (3) the data (an empty brain mask, a level
  without an Omega_B cell; estimate_failed), (4) anything else (the simplex check, an OSError); the table of steps against C22.B1-8, p2_core's two messages read from its source.
"""
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from ._world import FROZEN, GRID, PAIR, PY, ROOT, make_world

for _d in ('he', 'fl', 'ef', 'mc0'):
    sys.path.insert(0, str(ROOT/'experiments'/_d))

import he_c  # noqa: E402
import he_core as he  # noqa: E402

p2 = he_c.p2
needs_frozen = pytest.mark.skipif(not (FROZEN/'jd_curl/three_d/preopt.py').is_file(), reason='the frozen worktree is not on this machine')
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
           PYTHONPATH=f'{FROZEN}:{ROOT}')
SMALL = dict(iterations=[2, 2, 2, 4])


@pytest.fixture
def base():
    path = Path(tempfile.mkdtemp(prefix='he_c_world_'))
    yield path
    shutil.rmtree(path, ignore_errors=True)


# ---- bit for bit against run_p2 ----------------------------------------------------------------------------------------------------------------------
@needs_frozen
def test_the_three_passes_and_the_tissue_layer_are_run_p2s_bit_for_bit(tmp_path, base, monkeypatch):
    world = make_world(base/'w', monkeypatch)
    spec = dict(repo=str(ROOT), world=world, out=str(base/'he_out'), p2_out=str(base/'p2_out'), t88_pre=SMALL)
    (tmp_path/'spec.json').write_text(json.dumps(spec))
    done = subprocess.run([PY, '-B', str(Path(__file__).with_name('_c_compare.py')), str(tmp_path/'spec.json'), str(tmp_path/'result.json')], env=ENV, cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout[-3000:]+done.stderr[-5000:]
    r = json.loads((tmp_path/'result.json').read_text())
    assert r['status'] == 'done' and r['n_he'] == r['n_p2'] == 3 and r['field_moved'] > .01, r
    assert all(r['tissue_same'].values()), r['tissue_same']
    for row in r['rows']:
        assert all(row[k] for k in ('u_sha256', 'tissue', 'estimate', 'scale', 'levels', 'hooks', 'cap_hits')), row
    assert len({row['he_u'] for row in r['rows']}) == 3                                         # C acts: three different fields (the comparison is not of three copies of one)
    assert r['final_is_pass2'] and r['off_is_pass0']                                            # the field kept is pass 2's; C off is pass 0 (C22.B1-6, C22.B4-6 (a) on a synthetic pair)
    assert r['ratios_used'][0] is None and r['ratios_used'][1] == r['estimates'][0] and r['ratios_used'][2] == r['estimates'][1] and r['estimates'][2] is not None
    assert r['summary']['ratios_used'] == r['ratios_used'][1:] and r['summary']['last_estimate'] == r['estimates'][2] and r['summary']['n_passes'] == 3
    assert r['p2_status'] == 'checkpoint_mismatch'                                              # run_p2 keeps its field at step 600 only; with 4 native steps there is none (the passes are compared)


# ---- the constants -------------------------------------------------------------------------------------------------------------------------------------
def test_the_constants_in_the_frozen_settings_are_the_codes():
    from bayesreg.seg import channels
    s = he.frozen_settings()
    c = s['c']
    assert s['tissue_prior'] == he.TISSUE_PRIOR == he_c.OFF and he_c.MODES == ('off', 'moment3') and c == he_c.constants()      # HB: C off (C22 revision 8); C's constants stay frozen as the ablation's definition
    assert he.he2_settings()['tissue_prior'] == he_c.MOMENT3 and he.he2_settings()['c'] == c
    assert c['passes'] == p2.C_PASSES == 3 and c['ratios_learned'] == list(p2.C_CLASSES) == ['CSF', 'GM'] and c['levels'] == 4
    t = inspect.signature(channels.tissue_channels).parameters
    assert c['smoothing'] == dict(sigma_voxels=t['sigma_voxels'].default, truncate=t['truncate'].default, mode=t['mode'].default, eps=t['eps'].default) == dict(sigma_voxels=1., truncate=4., mode='nearest', eps=1e-3)
    assert c['brain_threshold'] == inspect.signature(channels.brain_mask).parameters['threshold'].default == .5
    assert 'tau_WM' in c['tau_background'] and 'not learned' in c['b'] and 'untapered, unrepaired' in c['estimated_on'] and 'Omega_B' in c['normalization'] and '090c985' in c['code']
    changed = he.frozen_settings()
    changed['c']['passes'] = 2
    assert not he.is_frozen(changed)


def test_the_statuses_are_he_cores():
    assert (he_c.REJECTED, he_c.IMPLEMENTATION, he_c.DATA, he_c.OTHER, he_c.ESTIMATE_FAILED) == (he.T88_REJECTED, he.T88_IMPLEMENTATION, he.T88_TISSUE_DATA, he.T88_TISSUE_ERROR, he.T88_ESTIMATE)


# ---- D86 ----------------------------------------------------------------------------------------------------------------------------------------------
def test_the_d86_table_and_p2_cores_messages():
    """C22.B1-8: (1) provenance or file check, class names -> provenance_rejected; (2) a coarse grid -> implementation_error; (3) an empty brain mask, a level without an Omega_B cell -> tissue_data; the rest
    tissue_error.  p2_core.tissue_levels' two ValueError messages are the patterns he_c matches (read from its source: a change there fails this test)."""
    from bayesreg.io.pair import ProvenanceRejected
    from jd_curl.validation import InputValidationError
    table = he_c.refusals()
    assert table['tissue_layer'] == [(ProvenanceRejected, he_c.REJECTED)] and table['class_index'] == [(ValueError, he_c.REJECTED)] and table['brain_mask'] == [(InputValidationError, he_c.DATA)]
    assert table['simplex'] == [] and table['tissue_channels'] == [] and table['grids'] == []                      # the lead's ruling of 11-07 (HE2 review minors 1, 2) and minor 5: all theirs is (4)
    assert set(table) == {'tissue_layer', 'simplex', 'brain_mask', 'tissue_channels', 'class_index', 'grids', 'tissue_levels'}
    assert '2026-11-07' in he_c.__doc__ and 'simplex' in he_c.__doc__ and 'JSONDecodeError' in he_c.__doc__ and "load_scalar's" in he_c.__doc__
    assert [(k, s) for k, s, _ in table['tissue_levels']] == [(ValueError, he_c.IMPLEMENTATION), (ValueError, he_c.DATA)]
    source = inspect.getsource(p2.tissue_levels)
    texts = re.findall(r"raise ValueError\((f?)'([^']*)'\)", source)
    assert len(texts) == 2
    grid_message = texts[0][1]
    cell_message = texts[1][1].replace('{level}', '2')
    assert he_c.COARSE_GRID.match(grid_message) and not he_c.NO_BRAIN_CELL.match(grid_message)
    assert he_c.NO_BRAIN_CELL.match(cell_message) and not he_c.COARSE_GRID.match(cell_message)
    for status, fallback, rejudge, stop in ((he_c.REJECTED, False, False, False), (he_c.IMPLEMENTATION, False, False, True), (he_c.DATA, True, False, False), (he_c.OTHER, True, True, False)):
        r = he_c.TissueRefusal(status, 's', ValueError('x')).record()
        assert (r['fallback'], r['needs_lead_rejudgment'], r['stop_queue']) == (fallback, rejudge, stop) and r['rule'] == he_c.D86_RULE


def pin_of(world):
    from bayesreg.io.pair import load_pair
    return load_pair(Path(world['stage_dir'])/world['pair']/'pair.npz', allowlist=world['paths']['allowlist'])


def refused(call):
    with pytest.raises(he_c.TissueRefusal) as info:
        call()
    return info.value


def test_the_tissue_layer_of_the_fixed_image_is_read_and_its_levels_built(base, monkeypatch):
    from bayesreg.grids import four_level_grids                                         # the copy of the frozen four_level_grids, bit for bit (bayesreg/grids.py)
    from jd_curl.three_d.nifti import downsample_array
    world = make_world(base/'w', monkeypatch)
    pin = pin_of(world)
    tissue = he_c.read_tissue(pin, Path(world['paths']['atropos_root']))
    assert tissue.names == ['CSF', 'GM', 'WM'] and tissue.wm_index == 2 and tissue.index == dict(background=0, CSF=1, GM=2, WM=3) and tissue.ptilde.shape == (4,)+GRID and tissue.brain.shape == GRID
    assert tissue.record['accepted']['segmenter_id'] == 'atropos' and tissue.record['n_brain'] == int(tissue.brain.sum()) > 0 and Path(tissue.record['directory']).name == pin.fixed_subject
    levels = he_c.levels_of(tissue, pin.grid, four_level_grids(pin.grid), downsample_array)
    assert [lv.level for lv in levels] == [0, 1, 2, 3] and all(lv.n_brain_cells > 0 for lv in levels)


def test_d86_1_the_input_provenance_and_the_class_names(base, monkeypatch):
    world = make_world(base/'p', monkeypatch, atropos=dict(test_labels_seen=True))
    r = refused(lambda: he_c.read_tissue(pin_of(world), Path(world['paths']['atropos_root'])))
    assert r.status == he_c.REJECTED and r.step == 'tissue_layer' and 'test_labels_seen' in str(r) and r.record()['fallback'] is False
    world = make_world(base/'n', monkeypatch, atropos=dict(class_names=('CSF', 'GRAY', 'WM')))
    r = refused(lambda: he_c.read_tissue(pin_of(world), Path(world['paths']['atropos_root'])))
    assert r.status == he_c.REJECTED and r.step == 'class_index' and 'CSF, GM and white matter' in str(r)


def test_d86_2_a_coarse_grid_is_an_implementation_error(base, monkeypatch):
    from jd_curl.three_d.grid import make_grid_3d
    from bayesreg.grids import four_level_grids                                         # the copy of the frozen four_level_grids, bit for bit (bayesreg/grids.py)
    from jd_curl.three_d.nifti import downsample_array
    world = make_world(base/'w', monkeypatch)
    pin = pin_of(world)
    tissue = he_c.read_tissue(pin, Path(world['paths']['atropos_root']))

    def other_grid(array, grid, shape):
        values, got, record = downsample_array(array, grid, shape)
        return values, make_grid_3d(nx=got.shape[2], ny=got.shape[1], nz=got.shape[0], hx=2., hy=2., hz=2., unit='mm'), record
    r = refused(lambda: he_c.levels_of(tissue, pin.grid, four_level_grids(pin.grid), other_grid))
    assert r.status == he_c.IMPLEMENTATION and r.step == 'tissue_levels' and r.record()['stop_queue'] is True and r.record()['fallback'] is False


def test_d86_3_the_data_an_empty_brain_mask_a_level_without_a_brain_cell_and_estimate_failed(base, monkeypatch):
    from bayesreg.grids import four_level_grids                                         # the copy of the frozen four_level_grids, bit for bit (bayesreg/grids.py)
    from jd_curl.three_d.nifti import downsample_array
    world = make_world(base/'e', monkeypatch, fixed_posteriors=np.zeros((3,)+GRID))
    r = refused(lambda: he_c.read_tissue(pin_of(world), Path(world['paths']['atropos_root'])))
    assert r.status == he_c.DATA and r.step == 'brain_mask' and r.record()['fallback'] is True and r.record()['needs_lead_rejudgment'] is False
    world = make_world(base/'w', monkeypatch)
    pin = pin_of(world)
    tissue = he_c.read_tissue(pin, Path(world['paths']['atropos_root']))
    tissue.brain = np.zeros(GRID, dtype=bool)
    tissue.brain[17, 18, 19] = True                                                             # one voxel: no cell of level 0 inside the mask
    r = refused(lambda: he_c.levels_of(tissue, pin.grid, four_level_grids(pin.grid), downsample_array))
    assert r.status == he_c.DATA and r.step == 'tissue_levels' and 'no cell inside the brain mask' in str(r)
    with pytest.raises(p2.EstimateFailed):                                                      # the passes: a field without any (J - 1)^2 has no ratio (status estimate_failed, D86 (3))
        p2.moment_ratios(np.zeros(GRID+(3,)), (1., 1., 1.), tissue.ptilde, he_c.read_tissue(pin, Path(world['paths']['atropos_root'])).brain, tissue.names, tissue.wm_index, .2)
    kind, f = he.t88_decision(dict(status=he.T88_ESTIMATE, reason='x', oracle=False, field=None))
    assert kind == 'fallback' and f['class'] == 'entry' and 'needs_lead_rejudgment' not in f
    kind, f = he.t88_decision(dict(status=he.T88_TISSUE_DATA, reason='x', oracle=False, field=None))
    assert kind == 'fallback' and f['class'] == 'entry' and 'needs_lead_rejudgment' not in f


def test_d86_4_anything_else_falls_back_for_the_leads_rejudgment(base, monkeypatch):
    world = make_world(base/'s', monkeypatch, fixed_posteriors=np.full((3,)+GRID, .6))          # not a simplex: simplex_with_background refuses (no class of C22.B1-8 names it)
    r = refused(lambda: he_c.read_tissue(pin_of(world), Path(world['paths']['atropos_root'])))
    assert r.status == he_c.OTHER and r.step == 'simplex' and r.record()['needs_lead_rejudgment'] is True and r.record()['fallback'] is True
    # the lead's ruling of 2026-11-07 (HE2 review minors 1, 2): a provenance.json that is not JSON, and a posterior NIfTI load_scalar refuses (InputValidationError), are (4) too
    world = make_world(base/'j', monkeypatch)
    pin = pin_of(world)
    (Path(world['paths']['atropos_root'])/pin.fixed_subject/'provenance.json').write_text('{not json')
    r = refused(lambda: he_c.read_tissue(pin, Path(world['paths']['atropos_root'])))
    assert r.status == he_c.OTHER and r.step == 'tissue_layer' and r.record()['exception'] == 'JSONDecodeError' and r.record()['needs_lead_rejudgment'] is True
    import nibabel as nib
    from jd_curl.validation import InputValidationError
    from ._world import a4
    world = make_world(base/'k', monkeypatch)
    pin = pin_of(world)
    directory = Path(world['paths']['atropos_root'])/pin.fixed_subject
    nib.Nifti1Image(np.zeros(GRID[::-1], dtype=np.float32), np.eye(4)).to_filename(str(directory/'posteriors01.nii.gz'))      # no units, no qform/sform: the reader refuses its header
    a4.rehash(directory)                                                                                                       # the file check passes: the hashes are the files'
    r = refused(lambda: he_c.read_tissue(pin, Path(world['paths']['atropos_root'])))
    assert r.status == he_c.OTHER and r.step == 'tissue_layer' and isinstance(r.error, InputValidationError) and r.record()['fallback'] is True
    world = make_world(base/'o', monkeypatch)
    from bayesreg.cli import run_pair

    def unreadable(*args, **kwargs):
        raise OSError('an Atropos file could not be read (test)')
    monkeypatch.setattr(run_pair, 'load_tissue_layer', unreadable)
    r = refused(lambda: he_c.read_tissue(pin_of(world), Path(world['paths']['atropos_root'])))
    assert r.status == he_c.OTHER and r.step == 'tissue_layer' and r.record()['exception'] == 'OSError'
    kind, f = he.t88_decision(dict(status=he.T88_TISSUE_ERROR, reason='x', oracle=False, field=None))
    assert kind == 'fallback' and f['class'] == 'entry' and f['needs_lead_rejudgment'] is True
    kind, info = he.t88_decision(dict(status=he.T88_IMPLEMENTATION, reason='x', rejection_stage='tissue'))
    assert kind == 'rejected' and info == dict(stage='tissue', kind='implementation_error', reason='x', stop_queue=True)
    kind, info = he.t88_decision(dict(status=he.T88_REJECTED, reason='x', rejection_stage='tissue', rejection_kind='tissue_input'))
    assert kind == 'rejected' and info == dict(stage='tissue', reason='x', kind='tissue_input') and 'stop_queue' not in info


def test_the_summary_and_the_mark():
    rows = [dict(u_sha256='a', tissue=dict(ratios_used=None), estimate=dict(ratios=dict(CSF=.8, GM=1.4))), dict(u_sha256='b', tissue=dict(ratios_used=dict(CSF=.8, GM=1.4)), estimate=dict(ratios=dict(CSF=.7, GM=1.5))),
            dict(u_sha256='c', tissue=dict(ratios_used=dict(CSF=.7, GM=1.5)), estimate=dict(ratios=dict(CSF=.6, GM=1.6)))]
    s = he_c.summary(rows)
    assert s['u_sha256'] == ['a', 'b', 'c'] and s['ratios_used'] == [dict(CSF=.8, GM=1.4), dict(CSF=.7, GM=1.5)] and s['last_estimate'] == dict(CSF=.6, GM=1.6) and he_c.summary([]) is None
    sides = {side: dict(segmenter_id='unet35_v4_ens', policy='all_folds_mean') for side in ('fixed', 'moving')}
    assert he_c.mark('unet35_v4_ens', ['fixed', 'moving', 'initial_u'], sides) == dict(oracle=False, truth_labels_read=False)
    assert he_c.mark('unet35_v1_ens', ['fixed', 'moving', 'initial_u'], sides)['oracle'] is True                         # another id: not the predictions it names
    assert he_c.mark('unet35_v4_ens', ['fixed', 'moving', 'fixed_seg35'], sides) == dict(oracle=True, truth_labels_read=True)
