"""Synthetic worlds for HE's tests (tmp only): a stage of any name with one pair whose subjects are development ids of the real allowlist (synthetic data only: no file of those subjects is read), predictions under
the repository's registry, Atropos posteriors; the truth label members differ from the predictions, as a real stage's do.  The pipeline itself runs in a subprocess (_pipeline.py) with PYTHONPATH=<frozen>:<repo>."""
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.ndimage import gaussian_filter

from bayesreg.io import pair as pair_module
from bayesreg.seg import structure_layer
from jd_curl.three_d.nifti import dice_labels, file_sha256

from ..io_seg._helpers import source_sha
from ..solve import test_d0_a4_tools as a4

ROOT = Path(__file__).resolve().parents[3]
HE = ROOT/'experiments/he'
FROZEN = Path('/home/rmcfz/seq_vp_tag_results/frozen_eb808be')
PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'
GRID = (34, 36, 38)                             # (nz, ny, nx): four T88 levels and the frozen pyramid's 8x level (experiments/seed/k1_control.py: >= 34 nodes per axis)
FIXED, MOVING = 'OASIS_OAS1_0438_MR1', 'OASIS_OAS1_0439_MR1'
PAIR = f'{FIXED}__{MOVING}'
STAGE = 'stage_he_synthetic'                    # any name: the entry does not know the stage names
LABELS = (1, 20, 2, 21, 7, 11, 13, 14)


def jit_cache():
    """A content-addressed numba cache shared by the tests of a session (a cached entry is the code an identical compile produced)."""
    path = Path(tempfile.gettempdir())/'he_tests_jit_cache'
    path.mkdir(exist_ok=True)
    return path


def label_map(grid=GRID):
    L = np.zeros(grid, dtype=np.int64)
    L[3:-3, 3:-3, 3:-3] = 1
    for k, label in enumerate(LABELS):
        L[4+k*3:7+k*3, 4:-4, 4:-4] = label
    return L


def write_pair(stage, pair, maps, *, truth_seed=3, truth_shift=1, manifest_extra=None, subjects=None):
    """pair.npz with images, the truth label members (not the predictions: a shift and a seed), an all-zero initial_u; manifest.json with the loader's fields and label-derived records, as a real stage has them."""
    fixed, moving = pair.split('__') if subjects is None else subjects
    d = Path(stage)/pair
    d.mkdir(parents=True)
    rng = np.random.default_rng(3)
    images = {s: gaussian_filter(rng.uniform(.1, .7, GRID), 1.) for s in (fixed, moving)}
    rng_truth = np.random.default_rng(truth_seed)
    truth = {s: np.roll(m, truth_shift, axis=2).astype(np.int16) for s, m in maps.items()}
    if truth_seed != 3:
        for s in truth:
            flip = rng_truth.uniform(size=GRID) < .02
            truth[s] = np.where(flip, 0, truth[s]).astype(np.int16)
    arrays = dict(fixed=images[fixed], fixed_seg4=(truth[fixed] % 5).astype(np.int16), fixed_seg35=truth[fixed], moving=images[moving], moving_seg4=(truth[moving] % 5).astype(np.int16), moving_seg35=truth[moving],
                  initial_u=np.zeros(GRID+(3,)))
    np.savez_compressed(d/'pair.npz', **arrays)
    baseline = {k: dice_labels(arrays['fixed_'+k].astype(float), arrays['moving_'+k].astype(float), labels=list(range(1, n+1))) for k, n in (('seg4', 4), ('seg35', 35))}
    manifest = dict(pair=pair, fixed_subject=fixed, moving_subject=moving, shape_zyx=list(GRID), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], unit='mm',
                    sources=dict(fixed=dict(sha256=source_sha(fixed)), moving=dict(sha256=source_sha(moving)), fixed_seg4=dict(sha256='0'*64)), pair_sha256=file_sha256(d/'pair.npz'),
                    baseline_label_metrics=baseline, weights=dict(lam=.1, kappa=.002), labels_policy='synthetic', preprocessing='synthetic', **(manifest_extra or {}))
    (d/'manifest.json').write_text(json.dumps(manifest, default=float))
    return d


def sealed_subjects():
    """(a B-line hold-out subject, an H1 subject): the first of each sealed list of bayesreg.eval.truth (subject ids only; the lists are pinned by sha256; no data of these subjects is read)."""
    from bayesreg.eval.truth import load_sealed_lists
    sealed = load_sealed_lists()
    return sorted(sealed.subjects['holdout_B_v1'])[0], sorted(sealed.subjects['H1'])[0]


def mistaken_allowlist(path, subjects):
    """The repository's allowlist with `subjects` wrongly added (a source of its own): the first lock would let them through."""
    document = json.loads((ROOT/'configs/bayesreg/dev_subjects.json').read_text())
    document['subjects']['OASIS'] = list(document['subjects']['OASIS'])+[s for s in subjects if s not in document['subjects']['OASIS']]
    document['sources'] = list(document['sources'])+[dict(name='he_test_mistake', dataset='OASIS', path='lists/he_test_mistake.txt', sha256='0'*64, never_for_training=False, subjects=list(subjects))]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(document))
    return Path(path)


V4_REGISTRY = ROOT/'configs/bayesreg/segmenters_hs_v1.json'      # the anchor's registry (C22.9-4 (xv)): the real file, read only
SEGMENTERS = dict(v4=(V4_REGISTRY, 'unet35_v4_ens'), v1=(structure_layer.REPO_REGISTRY, 'unet35_v1_ens'))


def as_atropos_seg_writes(directory, grid_zyx=GRID):
    """The posterior files rewritten as experiments/d0/atropos_seg.py writes them (units mm, qform and sform: the tissue layer's reader, run_pair.load_tissue_layer, requires them; P2's test does the same),
    the provenance's output hashes again."""
    import nibabel as nib
    from jd_curl.three_d.grid import make_grid_3d
    from jd_curl.three_d.nifti import save_scalar
    grid = make_grid_3d(nx=grid_zyx[2], ny=grid_zyx[1], nz=grid_zyx[0], hx=1., hy=1., hz=1., unit='mm')
    for name in a4.ATROPOS_FILES:
        p = np.asarray(nib.load(str(directory/name)).get_fdata()).transpose(2, 1, 0)
        (directory/name).unlink()
        save_scalar(directory/name, p, grid)
    a4.rehash(directory)


def make_world(tmp, monkeypatch, *, qc_fail=False, provenance=None, truth_seed=3, pair=PAIR, stage=STAGE, subjects=None, allowlist=None, segmenter='v4', atropos=None, fixed_posteriors=None):
    """The world of one pair; returns the JSON-able description _pipeline.py takes.  subjects: the manifest's (default: the pair's name); allowlist: another allowlist file (default: the repository's).
    HE2: segmenter 'v4' (default; the predictions name unet35_v4_ens and the models of the anchor's registry, configs/bayesreg/segmenters_hs_v1.json) or 'v1' (unet35_v1_ens: a reader that finds another
    segmenter's predictions); atropos: changes of the FIXED image's Atropos provenance (e.g. class_names); fixed_posteriors: the fixed image's Atropos posteriors (3, nz, ny, nx).  The reader's registry in
    paths is always the anchor's."""
    tmp = Path(tmp)
    monkeypatch.setattr(a4, 'SHAPE', GRID)
    registry, segmenter_id = SEGMENTERS[segmenter]
    models = pair_module.load_segmenter_registry(registry).models
    L = label_map()
    fixed, moving = pair.split('__') if subjects is None else subjects
    maps = {fixed: L, moving: np.roll(L, 1, axis=0)}
    stage_dir = tmp/'data/three_d'/stage
    write_pair(stage_dir, pair, maps, truth_seed=truth_seed, subjects=(fixed, moving))
    for subject, m in maps.items():
        noise = np.random.default_rng(int(hashlib.sha256(subject.encode()).hexdigest()[:8], 16)).uniform(0, 1, (36,)+GRID)
        probs = .6*(np.arange(36)[:, None, None, None] == m[None])+.4*noise/noise.sum(axis=0)
        d = a4.write_prediction(tmp/'pred', subject, models, probs=(probs/probs.sum(axis=0)).astype(np.float32))
        prov = json.loads((d/'provenance.json').read_text())
        prov.update(segmenter_id=segmenter_id, **(provenance or {}))
        (d/'provenance.json').write_text(json.dumps(prov))
        wm = np.full(GRID, .9) if qc_fail else np.where(np.isin(m, a4.MI.CEREBRAL_WM), .9, .05)
        csf = .3*(1-wm)
        posteriors = fixed_posteriors if (subject == fixed and fixed_posteriors is not None) else np.stack([csf, 1-wm-csf, wm])
        directory = a4.write_atropos(tmp/'atropos', subject, posteriors=posteriors, **((atropos or {}) if subject == fixed else {}))
        as_atropos_seg_writes(directory)
    paths = dict(pred_root=str(tmp/'pred'), models_root=str(tmp/'models'), image_root=str(tmp/'images'), registry=str(V4_REGISTRY), folds=str(a4.MI.FOLDS), folds_sha256=None, atropos_root=str(tmp/'atropos'),
                 allowlist=str(ROOT/'configs/bayesreg/dev_subjects.json' if allowlist is None else allowlist), frozen=str(FROZEN), cache=str(jit_cache()))
    return dict(stage_dir=str(stage_dir), pair=pair, paths=paths, root=str(tmp))
