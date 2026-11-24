"""D0 (docs/bayesreg/05_D0协议.md §3, §12, §14): build the recoloured input stages of arms A1, A2, A3 and of the
supplementary arms A4p (A4' of §12: self-trained U-Net posterior colouring), A1Ph, A1Ps (§13), A5, A5P (§14).

For each arm and each of the 19 pairs of data/three_d/stage_L2R_val_v1 this writes
  <inputs>/<arm>/stage_L2R_val_v1/<pair>/{pair.npz, manifest.json}
plus a byte copy of the stage's selection.json and completion.json, so that the frozen configuration-C entry point
(eb808be scripts/run_stage_3d_e48_pg.py, resolve_pair) reads it through the worktree link
data/three_d/stage_L2R_val_v1 (run_queue.sh switches that link per arm during `prepare` only).
pair.npz: the original members in the original order; only `fixed` and `moving` are replaced (float64, same shape);
`fixed_seg4`, `fixed_seg35`, `moving_seg4`, `moving_seg35`, `initial_u` are copied bitwise so that the frozen run can
compute its own labels.json / smoothness.json. manifest.json: the original with pair_sha256 recomputed,
baseline_native_mse moved into preprocessing_d0 (it was computed on the original images) and a preprocessing_d0 record.

Colourings (each image with its own labels / posteriors and its own original intensities I):
  A1 (oracle, seg4):  c_k = mean(I[seg4 == k]), k = 1..4;  J = sum_k c_k 1[seg4 == k] (0 elsewhere);  G_.75 * J
  A2 (Atropos):       P_k = Atropos posterior k (k = 1..3, CSF/GM/WM; zero outside its mask);  c_k = sum P_k I / sum P_k;
                      J = sum_k c_k P_k  (no smoothing)
  A3 (oracle, seg35): delta = .05 * mean(I[seg4 == 3]) (white matter);  z = default_rng(20261030).uniform(-1, 1, 35);
                      c_l = mean(I[seg35 == l]) + delta * z[l-1], l = 1..35;  J = sum_l c_l 1[seg35 == l] (0 elsewhere);  G_.75 * J
  A4p (U-Net, §12):   P_l = U-Net ensemble posterior of seg35 class l (probs36.npz, l = 0..35, 0 = background), read back
                      by bayesreg.seg.channels.read_back_probabilities (raw channel sum on every voxel within 1e-3 of 1
                      for float16 storage, else STOP; then divided by the sum);
                      w = sum (P_1 + P_20) I / sum (P_1 + P_20) (cerebral white matter, seg35 labels 1 and 20);
                      delta = .05 * w;  z as A3;  c_l = sum P_l I / sum P_l + delta * z[l-1], l = 1..35 (a class with
                      sum P_l = 0 is skipped);  J = sum_l c_l P_l (background class weight 0);  G_.75 * J
                      (the A3 formula with the one-hot replaced by the posterior; only delta's white-matter mean differs:
                      A3 takes the truth seg4 == 3, A4p the segmenter's own cerebral white matter)
  A1Ph, A1Ps (A1P, runs/20261102-a1p/plan.md): the A1 colouring with the truth seg4 replaced by the U-Net prediction.
                      P_l as A4p; S_k = sum over l with seg4_of_seg35(l) = k of P_l, k = 0..4 (0 = background; the table
                      configs/bayesreg/seg4_of_seg35_v1.json, read from the development labels by check_seg4_map.py, not
                      the unverified tissue4_ref);
                      A1Ph (hard): L = argmax_k S_k (ties: the smallest k);  c_k = mean(I[L == k]), k = 1..4;
                                   J = sum_k c_k 1[L == k] (0 elsewhere);  G_.75 * J
                      A1Ps (soft): c_k = sum S_k I / sum S_k, k = 1..4 (a class with sum S_k = 0 is skipped);
                                   J = sum_k c_k S_k;  G_.75 * J
                      with a one-hot posterior both are A1 of the labels seg4_of_seg35(seg35). No label reaches the colouring;
                      every A4p check applies (subject admission, registry, D24, provenance, read-back, group sums, alignment QC).
  A5, A5P (runs/20261103-a5/plan.md, protocol §14): the fixed colour table f_l of configs/bayesreg/colour_table_a5_v1.json (pinned
                      sha256) on a label map L of the image: A5 (oracle) L = seg35; A5P L = argmax_l P_l (P_l as A4p, ties: the
                      smallest l), no label; w = mean(I[L in {1, 20}]) (the image's own cerebral white matter);
                      J = sum_{l=1..35} f_l w 1[L == l] (0 elsewhere);  G_.75 * J.  colour_from_table takes the image and a label
                      map, never a pair; with L one-hot for the same labels A5P equals A5 bit for bit.  Every A4p check applies to A5P.
  G_.75 = scipy.ndimage.gaussian_filter(sigma=.75 voxel per axis, mode='nearest', truncate=4.0), float64.
Labels are read here only for the oracle arms A1/A3/A5 (contract C03.1-2: oracle arm) and are copied for evaluation; A2 reads
no label for its colouring (its images depend only on I and the Atropos posteriors, whose provenance is checked with
bayesreg.io.pair.validate_segmentation_provenance / verify_segmentation_files before use). A4p reads no label for its
colouring either. Its predictions pass, per pair, every check of bayesreg.seg.structure_layer.StructureLayer (the run_pair
checks: grouping status; the repository registry only, no staging registry; D24 pair-level label contact; provenance
with model / member sha256; input sha256 = the manifest's source image; grid; output hashes of probs36.npz and
structure_groups.npz; read-back raw sums); the repository registry must be tracked by git and clean (no staged,
unstaged or untracked change; A4 review item 3); both images must have policy all_folds_mean; every subject must be on
the allowlist, not never_for_training and not a B-line training subject of splits/bayesreg_v1/folds.json (pinned
sha256); the 36 classes summed by the grouping (structure13_v1) must equal structure_groups.npz within 2e-3 (A4 review
item 6; first-order bound of the two float16 roundings and read-back normalizations about 3.5 * 2^-11 = 1.7e-3); and
the label-free alignment QC (A4 review item 5; protocol §12.4 item 5, revision 2): for each image
Dice({U-Net cerebral_wm group > .5}, {Atropos WM posterior > .5}) >= .80, the U-Net group read from the verified
structure_groups.npz (structure13_v1 channel cerebral_wm, seg35 labels 1 and 20), the Atropos WM posterior read through
the A2 path `posteriors` (tissue-layer provenance bound to the manifest's source image, output files re-hashed, xyz ->
zyx, same grid; the file named by wm_index); a missing or refused Atropos output STOPs (checked for every subject
before anything is written, and again per image). Revision 2 (before any A4p result; runs/20261030-d0-a4/
qc_survey_20261030/): on the 20 development subjects this Dice is .853-.881 aligned and at most .792 under the
flips (left-right, anterior-posterior, superior-inferior) and shifts of 2 or more voxels along any axis; a 1-voxel
shift is not caught (up to .854, qc_wm_shifts.out). Revision 1's
Dice({I > 0}, {P_0 < .5}) (I its own image, P_0 the U-Net background posterior) is still computed and recorded per
image, but no longer stops: it was .79-.85 aligned ({I > 0} also holds CSF the labels do not cover) and hardly moved
under a left-right flip or an 8-voxel shift.
A record <inputs>/<arm>/d0_inputs.json lists every written file with its sha256 and the colour constants.
An existing arm directory is never overwritten (STOP). A STOP leaves the arm directory without d0_inputs.json (kept as
evidence; the queue refuses it); to rebuild, the lead first renames it (e.g. A4p.stopped_<UTC>, kept) and then runs
make_inputs again (protocol §12.8). A4p is written under /home/rmcfz/seq_vp_tag_results/d0_v1_a4/inputs by default and
never under d0_v1/. A1Ph and A1Ps are written under /home/rmcfz/seq_vp_tag_results/d0_v1_a1p/inputs by default, never under
d0_v1/ or d0_v1_a4/. A5 and A5P are written under /home/rmcfz/seq_vp_tag_results/d0_v1_a5/inputs by default, never under d0_v1/,
d0_v1_a4/ or d0_v1_a1p/. --only-pair and --stage are for dry runs: the record then says partial true and the queue refuses it.
A5Pv2 (runs/20261103-unv2/plan.md): --arms A5P with --segmenter-id unet35_v2_full, --registry, --pred-root and --inputs, all four: the
colour table, the formula and every check of §12.4 are those of A5P; the segmenter's repository registry is the file --registry names
(one of bayesreg.seg.structure_layer.FORMAL_REGISTRIES, committed and clean, as for v1); the models root recorded is the parent of
--pred-root; the inputs go under the --inputs given, never under d0_v1/, d0_v1_a4/, d0_v1_a1p/ or d0_v1_a5/. The preregistered rule
is enforced here: <parent of --pred-root>/eval_seg/eval_seg.json of these predictions must say decision.run_a5pv2 true, otherwise
STOP, unless --bz-decision-run-a5pv2 names BZ's decision (recorded). Without these options nothing changes, bit for bit.
Usage: python -B experiments/d0/make_inputs.py [--arms A1 A2 A3 | --arms A4p | --arms A1Ph A1Ps | --arms A5 A5P] [--inputs DIR] [--seg-root DIR]
                                               [--unet-pred-root DIR] [--stage DIR] [--only-pair NAME]
       python -B experiments/d0/make_inputs.py --arms A5P --segmenter-id unet35_v2_full --registry FILE --pred-root DIR --inputs DIR [--seg-root DIR]
                                               [--bz-decision-run-a5pv2 "where BZ decided it"]
       python -B experiments/d0/make_inputs.py --write-a4-subjects FILE [--only-missing]   (subject ids only)
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bayesreg.io.pair import load_pair, validate_segmentation_provenance, verify_segmentation_files  # noqa: E402

STAGE_NAME = 'stage_L2R_val_v1'
STAGE = ROOT/'data/three_d'/STAGE_NAME
ALLOWLIST = ROOT/'configs/bayesreg/dev_subjects.json'
SEED = 20261030
SIGMA = .75
DELTA_FRAC = .05
A1P_ARMS = ('A1Ph', 'A1Ps')                                 # runs/20261102-a1p/plan.md
A5_ARMS = ('A5', 'A5P')                                     # runs/20261103-a5/plan.md (fixed colour table)
UNET_ARMS = ('A4p',)+A1P_ARMS+('A5P',)                      # coloured from the U-Net posterior, no label
ARMS = ('A1', 'A2', 'A3', 'A4p')+A1P_ARMS+A5_ARMS
ORACLE_ARMS = ('A1', 'A3', 'A5')                            # coloured from the truth labels (contract C03.1-2)
SEEDED = ('A3', 'A4p')                                      # arms that use the z offsets
D0V1 = Path('/home/rmcfz/seq_vp_tag_results/d0_v1')
D0A4 = Path('/home/rmcfz/seq_vp_tag_results/d0_v1_a4')     # A4p (protocol §12); never written under d0_v1/
D0A1P = Path('/home/rmcfz/seq_vp_tag_results/d0_v1_a1p')   # A1P; never written under d0_v1/ or d0_v1_a4/
D0A5 = Path('/home/rmcfz/seq_vp_tag_results/d0_v1_a5')     # A5, A5P; never written under d0_v1/, d0_v1_a4/ or d0_v1_a1p/
COLOUR_TABLE_A5 = ROOT/'configs/bayesreg/colour_table_a5_v1.json'
COLOUR_TABLE_A5_SHA256 = 'ee1d8160bf0d6966df8dc8bfe9bf367e7c6069b34a47a3edf18c58839ed85289'     # add-only file, pinned here
SEG4_MAP = ROOT/'configs/bayesreg/seg4_of_seg35_v1.json'
N_SEG4 = 5                                                  # seg4 labels 0..4, 0 = background
UNET_ROOT = Path('/home/rmcfz/seq_vp_tag_results/seg/unet35_v1')
FOLDS = ROOT/'splits/bayesreg_v1/folds.json'
N_CLASSES = 36                                              # U-Net classes: seg35 label l at channel l, 0 = background
CEREBRAL_WM = (1, 20)                                       # seg35 label table: Left/Right-Cerebral-White-Matter
GROUP_TOL = 2e-3                                            # |sum of probs36 over a group - structure_groups.npz|
#   (A4 review item 6; was 1e-2): both files are float16 roundings of the same float32 ensemble probabilities, each
#   read back divided by its own channel sum; first-order worst case about 3.5 * 2^-11 = 1.7e-3 at a group value near 1
ALIGN_QC_MIN = .80                  # protocol §12.4 item 5 (revision 2): Dice(U-Net cerebral_wm > .5, Atropos WM > .5)
ALIGN_QC_LEVEL = .5                 # both posteriors thresholded at .5
UNET_WM_GROUP = 'cerebral_wm'       # structure13_v1 group of seg35 labels 1, 20 (= CEREBRAL_WM)
SUPPORT_REVISION1_MIN = .8          # revision 1's stop threshold of Dice({I > 0}, {P_0 < .5}); since revision 2 a record
KEY_ORDER = ('fixed', 'fixed_seg4', 'fixed_seg35', 'moving', 'moving_seg4', 'moving_seg35', 'initial_u')
RECIPES = {
    'A1': 'oracle seg4 colouring: c_k = mean(I[seg4==k]) (k=1..4), J = sum_k c_k 1[seg4==k], background 0, '
          'gaussian sigma .75 voxel (mode nearest, truncate 4)',
    'A2': 'Atropos posterior colouring: c_k = sum P_k I / sum P_k (k=1..3), J = sum_k c_k P_k, no smoothing',
    'A3': 'oracle seg35 colouring: c_l = mean(I[seg35==l]) + .05 mean(I[seg4==3]) z_l, z = default_rng(20261030)'
          '.uniform(-1,1,35), J = sum_l c_l 1[seg35==l], background 0, gaussian sigma .75 voxel (mode nearest, truncate 4)',
    'A4p': "U-Net posterior colouring (protocol section 12, A4'; the A3 formula with the one-hot replaced by the posterior): "
           'P_l = unet35_v1_ens posterior of seg35 class l (probs36.npz via read_back_probabilities, l=0..35, 0 = background), '
           'c_l = sum P_l I / sum P_l + .05 w z_l with w = sum (P_1+P_20) I / sum (P_1+P_20), z = default_rng(20261030)'
           '.uniform(-1,1,35), J = sum_{l=1..35} c_l P_l, gaussian sigma .75 voxel (mode nearest, truncate 4)',
    'A1Ph': 'predicted seg4 hard colouring (A1P; the A1 formula on a predicted label): S_k = sum of the unet35_v1_ens posterior '
            'over the seg35 classes of seg4 label k (configs/bayesreg/seg4_of_seg35_v1.json), L = argmax_k S_k (k=0..4, ties: '
            'smallest k), c_k = mean(I[L==k]) (k=1..4), J = sum_k c_k 1[L==k], background 0, gaussian sigma .75 voxel (mode '
            'nearest, truncate 4)',
    'A1Ps': 'predicted seg4 soft colouring (A1P): S_k as A1Ph, c_k = sum S_k I / sum S_k (k=1..4), J = sum_k c_k S_k, gaussian '
            'sigma .75 voxel (mode nearest, truncate 4)',
    'A5': 'oracle seg35 table colouring (A5): L = seg35, w = mean(I[L in {1,20}]), J = sum_{l=1..35} f_l w 1[L==l] with f_l of '
          'configs/bayesreg/colour_table_a5_v1.json, background 0, gaussian sigma .75 voxel (mode nearest, truncate 4)',
    'A5P': 'predicted seg35 table colouring (A5P; the A5 formula on a predicted label): L = argmax_l P_l (unet35_v1_ens posterior, '
           'l=0..35, 0 = background, ties: smallest l), w = mean(I[L in {1,20}]), J = sum_{l=1..35} f_l w 1[L==l] with f_l of '
           'configs/bayesreg/colour_table_a5_v1.json, background 0, gaussian sigma .75 voxel (mode nearest, truncate 4)',
}
RECIPE_SEGMENTER = 'unet35_v1_ens'                          # the segmenter named in RECIPES; A5Pv2 names its own
A5PV2_SEGMENTER = 'unet35_v2_full'                          # the only segmenter A5Pv2 takes (UNV2 review, important 2)
EVAL_SEG_FORMAT = 'bayesreg.unet35.eval_seg.v1'             # experiments/unv2/eval_seg.py


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def smooth(J):
    return gaussian_filter(J, sigma=SIGMA, mode='nearest', truncate=4.0)


def z_offsets():
    return np.random.default_rng(SEED).uniform(-1., 1., 35)


def colour_labels(I, lab, ids, offsets=None):
    J = np.zeros_like(I)
    consts = {}
    for i, l in enumerate(ids):
        m = lab == l
        if not m.any():
            consts[int(l)] = None
            continue
        c = float(I[m].mean()) + (0. if offsets is None else float(offsets[i]))
        J[m] = c
        consts[int(l)] = c
    return J, consts


def posteriors(seg_root, subject, pair, side, allowlist=None):
    """HB (C23.4-8): `allowlist`, for a hold-out pair the bound allowlist load_pair admitted it through, is handed to validate_segmentation_provenance (its sealed recheck); None as before."""
    d = seg_root/subject
    prov = json.loads((d/'provenance.json').read_text())
    accepted = validate_segmentation_provenance(prov, pair, side=side, layer='tissue') if allowlist is None else \
        validate_segmentation_provenance(prov, pair, side=side, layer='tissue', allowlist=allowlist)
    verify_segmentation_files(prov, d)
    P = np.stack([np.asarray(nib.load(str(d/name)).dataobj).transpose(2, 1, 0).astype(np.float64)
                  for name in prov['posterior_files']])
    if P.shape[1:] != pair.fixed.shape:
        raise SystemExit(f'{subject}: posterior grid {P.shape[1:]} != image grid {pair.fixed.shape}')
    return P, prov, accepted


def colour_from_table(I, L, table):
    """A5, A5P (protocol §14): J = G_.75 * sum_{l=1..35} f_l w 1[L == l], background 0, w = mean(I[L in {1, 20}]).  I: the image's own
    intensities; L: a label map on the same grid (the truth seg35 for A5, the argmax of the posterior for A5P), integer or float
    storage; table: {l: f_l}.  The same I and the same labels give the same bytes, whatever L's storage.  STOP on a value outside
    0..35 or without cerebral white matter (w undefined)."""
    if L.shape != I.shape:
        raise SystemExit(f'colour_from_table: label map shape {L.shape} != image shape {I.shape}')
    if not np.isin(L, np.arange(N_CLASSES)).all():
        raise SystemExit(f'colour_from_table: label map has values outside 0..{N_CLASSES-1}')
    white = np.isin(L, CEREBRAL_WM)
    if not white.any():
        raise SystemExit('colour_from_table: no cerebral white matter in the label map (w undefined)')
    w = float(I[white].mean())
    del white
    J = np.zeros_like(I)
    consts, voxels = {}, {0: int(np.count_nonzero(L == 0))}
    for l in range(1, N_CLASSES):
        m = L == l
        voxels[l] = int(np.count_nonzero(m))
        consts[l] = table[l]*w
        J[m] = consts[l]
    return smooth(J), dict(wm_mean=w, wm_labels=list(CEREBRAL_WM), c_by_seg35_label=consts, voxels_by_seg35_label=voxels)


def colour(arm, I, seg4, seg35, P, table=None):
    if arm == 'A1':
        J, consts = colour_labels(I, seg4, range(1, 5))
        return smooth(J), dict(c_by_seg4_label=consts)
    if arm == 'A2':
        consts = [float((P[k]*I).sum()/P[k].sum()) for k in range(P.shape[0])]
        return np.tensordot(np.asarray(consts), P, axes=1), dict(c_by_posterior_file=consts)
    if arm == 'A3':
        wm = float(I[seg4 == 3].mean())
        delta = DELTA_FRAC*wm
        J, consts = colour_labels(I, seg35, range(1, 36), delta*z_offsets())
        return smooth(J), dict(c_by_seg35_label=consts, delta=delta, wm_mean=wm)
    if arm == 'A4p':                                    # P: (36, nz, ny, nx) U-Net posterior; no label is passed
        if seg4 is not None or seg35 is not None:
            raise ValueError('A4p: the colouring takes no label')
        if P.shape != (N_CLASSES,)+I.shape:
            raise SystemExit(f'A4p: posterior shape {P.shape} != {(N_CLASSES,)+I.shape}')
        w = P[CEREBRAL_WM[0]]+P[CEREBRAL_WM[1]]
        if not w.sum() > 0:
            raise SystemExit('A4p: no cerebral white matter in the posterior (delta undefined)')
        wm = float((w*I).sum()/w.sum())
        del w
        delta = DELTA_FRAC*wm
        z = z_offsets()
        J = np.zeros_like(I)
        consts = {}
        for l in range(1, N_CLASSES):
            mass = float(P[l].sum())
            if mass == 0:
                consts[l] = None
                continue
            c = float((P[l]*I).sum())/mass + delta*float(z[l-1])
            J += c*P[l]
            consts[l] = c
        return smooth(J), dict(c_by_seg35_class=consts, delta=delta, wm_mean_soft=wm, wm_labels=list(CEREBRAL_WM))
    if arm in A1P_ARMS:                                 # P: (5, nz, ny, nx) the posterior merged to seg4; no label is passed
        if seg4 is not None or seg35 is not None:
            raise ValueError(f'{arm}: the colouring takes no label')
        if P.shape != (N_SEG4,)+I.shape:
            raise SystemExit(f'{arm}: merged posterior shape {P.shape} != {(N_SEG4,)+I.shape}')
        if arm == 'A1Ph':
            L = np.argmax(P, axis=0)                    # ties: the smallest k
            J, consts = colour_labels(I, L, range(1, N_SEG4))
            voxels = {k: int(np.count_nonzero(L == k)) for k in range(N_SEG4)}
            return smooth(J), dict(c_by_seg4_label=consts, predicted_voxels_by_seg4_label=voxels)
        J = np.zeros_like(I)
        consts, mass = {}, {}
        for k in range(1, N_SEG4):
            mass[k] = float(P[k].sum())
            if mass[k] == 0:
                consts[k] = None
                continue
            consts[k] = float((P[k]*I).sum())/mass[k]
            J += consts[k]*P[k]
        return smooth(J), dict(c_by_seg4_label=consts, posterior_mass_by_seg4_label=mass)
    if arm == 'A5':
        J, consts = colour_from_table(I, seg35, table)
        return J, dict(consts, label_source='truth seg35 (oracle)')
    if arm == 'A5P':                                    # P: (36, nz, ny, nx) U-Net posterior; no label is passed
        if seg4 is not None or seg35 is not None:
            raise ValueError('A5P: the colouring takes no label')
        if P.shape != (N_CLASSES,)+I.shape:
            raise SystemExit(f'A5P: posterior shape {P.shape} != {(N_CLASSES,)+I.shape}')
        J, consts = colour_from_table(I, np.argmax(P, axis=0), table)      # ties: the smallest label
        return J, dict(consts, label_source='argmax of the U-Net posterior')
    raise ValueError(arm)


# ---- A5, A5P: the colour table (runs/20261103-a5/plan.md) -----------------------------------------------------------------------
def load_colour_table(path=COLOUR_TABLE_A5, sha256=COLOUR_TABLE_A5_SHA256):
    """({l: f_l}, record).  STOP unless the file has the pinned sha256, every seg35 label 1..35 is in exactly one row, and each row is
    one midline label or the left and right label of one structure (structure_groups_v1 label table: 1..10 <-> 20..29, 14..19 <-> 30..35)."""
    path = Path(path)
    if sha_file(path) != sha256:
        raise SystemExit(f'{path}: sha256 {sha_file(path)} != pinned {sha256} (the table is add-only: a new table is a new file)')
    doc = json.loads(path.read_text())
    table = {}
    for row in doc['rows']:
        labels = row['labels']
        if len(labels) == 2:
            if labels[1] != labels[0]+(19 if labels[0] <= 10 else 16) or not (1 <= labels[0] <= 10 or 14 <= labels[0] <= 19):
                raise SystemExit(f'{path}: row {labels} is not the left and right label of one structure')
        elif labels not in ([11], [12], [13]):
            raise SystemExit(f'{path}: row {labels} is not a midline label')
        for l in labels:
            if l in table:
                raise SystemExit(f'{path}: label {l} in two rows')
            table[l] = float(row['f'])
    if sorted(table) != list(range(1, N_CLASSES)) or not all(np.isfinite(v) and v > 0 for v in table.values()):
        raise SystemExit(f'{path}: f must be finite and positive on exactly the labels 1..{N_CLASSES-1}')
    return table, dict(table_file=str(path), table_sha256=sha_file(path), schema=doc['schema'], rows=doc['rows'],
                       provenance=doc['provenance'], f_by_seg35_label={str(l): table[l] for l in sorted(table)})


# ---- A1P: the seg35 -> seg4 table (read from the development labels, runs/20261102-a1p/seg4_label_check.json) ----------------
def load_seg4_map(path=SEG4_MAP, root=ROOT):
    """(lookup (36,) int: seg4 label of each seg35 class, record).  STOP unless the table covers 0..35 with values in 0..4,
    only 0 maps to 0, and the check file it names has the pinned sha256 and says the same: every seg35 label in the table's
    seg4 label, share 1 on every subject, seg35 = 0 exactly where seg4 = 0."""
    path = Path(path)
    doc = json.loads(path.read_text())
    table = {int(k): int(v) for k, v in doc['seg4_of_seg35'].items()}
    if sorted(table) != list(range(N_CLASSES)) or any(not 0 <= v < N_SEG4 for v in table.values()):
        raise SystemExit(f'{path}: seg4_of_seg35 must map 0..{N_CLASSES-1} into 0..{N_SEG4-1}')
    if table[0] != 0 or any(table[k] == 0 for k in range(1, N_CLASSES)) or set(table.values()) != set(range(N_SEG4)):
        raise SystemExit(f'{path}: only the background class maps to seg4 label 0, and every seg4 label is reached')
    check_path = Path(root)/doc['check_file']
    if sha_file(check_path) != doc['check_file_sha256']:
        raise SystemExit(f'{check_path}: sha256 differs from the one pinned in {path}')
    check = json.loads(check_path.read_text())
    rows = check['per_seg35_label']
    bad = [k for k in range(1, N_CLASSES) if rows[str(k)]['seg4'] != table[k] or rows[str(k)]['share'] != 1.
           or rows[str(k)]['min_subject_share'] != 1.]
    if bad or check['voxels_seg4_positive_seg35_0'] != 0 or check['voxels_seg35_positive_seg4_0'] != 0:
        raise SystemExit(f'{path}: not what {check_path} found (labels {bad}; seg4 > 0 with seg35 = 0: '
                         f"{check['voxels_seg4_positive_seg35_0']}; seg35 > 0 with seg4 = 0: {check['voxels_seg35_positive_seg4_0']})")
    record = dict(table_file=str(path), table_sha256=sha_file(path), check_file=str(check_path),
                  check_sha256=doc['check_file_sha256'], check_subjects=len(check['subjects']), seg4_names=doc['seg4_names'],
                  seg4_of_seg35={str(k): table[k] for k in sorted(table)})
    return np.array([table[k] for k in range(N_CLASSES)]), record


def merge_seg4(P, lookup):
    """S (5, nz, ny, nx): S_k = sum over the seg35 classes l with lookup[l] = k of P_l (P: (36, nz, ny, nx))."""
    if P.shape[0] != N_CLASSES:
        raise SystemExit(f'merge_seg4: {N_CLASSES} classes required, got {P.shape[0]}')
    S = np.zeros((N_SEG4,)+P.shape[1:], dtype=P.dtype)
    for label in range(N_CLASSES):
        S[lookup[label]] += P[label]
    return S


# ---- A4p: the self-trained structure-layer segmenter (protocol §12) ---------------------------------------------------
@dataclass(frozen=True)
class UNetSource:
    """Where the A4p posteriors come from: predictions <pred_root>/<subject>/ written by bayesreg.seg.unet.infer; the
    registry is always the repository one (StructureLayer refuses any other); the folds file and its pinned sha256 for
    the independent D24 subject check (folds_sha256 None: bayesreg.seg.unet.pins.PINNED_FOLDS_SHA256). A5Pv2:
    segmenter_id and registry (that file is then the repository registry of the structure layer), gate (the record of
    a5pv2_gate); all None for everything else, and then the record is the one it always was."""
    pred_root: Path = UNET_ROOT/'pred'
    folds: Path = FOLDS
    folds_sha256: str = None
    models_root: Path = UNET_ROOT
    image_root: Path = Path('/mnt/d/data')
    segmenter_id: str = None
    registry: Path = None
    gate: dict = None

    def record(self):
        record = dict(pred_root=str(self.pred_root), folds=str(self.folds), models_root=str(self.models_root),
                      image_root=str(self.image_root))
        if self.segmenter_id is not None:
            record.update(segmenter_id=self.segmenter_id, registry=str(self.registry), a5pv2_gate=self.gate)
        return record


def a5pv2_gate(pred_root, bz_decision=None):
    """The preregistered rule of runs/20261103-unv2/plan.md, enforced before anything is read or written: the evaluation of exactly these
    predictions (<parent of pred_root>/eval_seg/eval_seg.json, experiments/unv2/eval_seg.py) says decision.run_a5pv2 true. A missing or
    foreign evaluation counts as not allowed. STOP otherwise, unless `bz_decision` (--bz-decision-run-a5pv2) names BZ's decision, which is
    recorded with the reason it was needed. Returns the record."""
    path = Path(pred_root).parent/'eval_seg'/'eval_seg.json'
    record = dict(rule='mean Dice35 of v2 more than .002 below that of v1: A5Pv2 does not run (runs/20261103-unv2/plan.md)',
                  eval_file=str(path), run_a5pv2=None, v2_minus_v1=None, eval_sha256=None)
    if not path.is_file():
        problem = f'no evaluation at {path}'
    else:
        document = json.loads(path.read_text())
        decision = document.get('decision') or {}
        record.update(eval_sha256=sha_file(path), run_a5pv2=decision.get('run_a5pv2'), v2_minus_v1=decision.get('v2_minus_v1'))
        evaluated = (document.get('roots') or {}).get('v2')
        if document.get('format') != EVAL_SEG_FORMAT or evaluated is None or Path(evaluated).resolve() != Path(pred_root).resolve():
            problem = f'{path} is not an evaluation of the predictions {pred_root} (it evaluated {evaluated})'
        elif decision.get('run_a5pv2') is not True:
            problem = f"the rule does not allow A5Pv2: decision.run_a5pv2 {decision.get('run_a5pv2')!r}, v2 - v1 {decision.get('v2_minus_v1')}"
        else:
            problem = None
    if problem is not None and not bz_decision:
        raise SystemExit(f'A5Pv2: {problem}; hand over to BZ (only --bz-decision-run-a5pv2 "<where BZ decided it>" goes past this); nothing written')
    return dict(record, passed_by='the rule' if problem is None else 'BZ decision', problem=problem, bz_decision=bz_decision)


def stage_subjects(stage):
    """Canonical subjects of the stage's pairs (manifest fields only), in order of first appearance."""
    out = []
    for row in json.loads((stage/'selection.json').read_text())['pairs']:
        man = json.loads((stage/row['pair']/'manifest.json').read_text())
        for s in (man['fixed_subject'], man['moving_subject']):
            if s not in out:
                out.append(s)
    return out


def unsealed_allowlist(allowlist):
    """HB (contract C23.8-2, C23.4-1): True when `allowlist` is a DevAllowlist carrying an unseal object (bayesreg.io.unseal.bind_unseal's), the hold-out's; False for a path (the development's)."""
    from bayesreg.io.guard import DevAllowlist
    return isinstance(allowlist, DevAllowlist) and getattr(allowlist, 'unseal', None) is not None


def a4_admission_unsealed(subjects, allowlist, split):
    """HB (contract C23.8-2, revision 6c; hi_core.other_layers' rule): D24 for a hold-out pair, the admission only.  Every subject passes bayesreg.io.guard.check_subject on the bound allowlist through
    the unseal object (C23.4-2 in full); never_for_training only in the record's own source (the unsealed one); in no other allowlist source; not a B-line training subject of the folds file; on the
    sealed list the record unseals and no other (H1-H4 never).  STOP otherwise.  Reads subject ids only (no image, no label); no number of the method changes."""
    from bayesreg.io.guard import check_subject
    from bayesreg.io.sealed import guard_sealed_lists, lists_of
    from bayesreg.io.unseal import bound_problem
    problem = 'no unseal object on the allowlist' if getattr(allowlist, 'unseal', None) is None else bound_problem(allowlist)
    if problem:
        raise SystemExit(f'A4p: the bound allowlist does not hold ({problem})')
    unseal = allowlist.unseal
    sealed = guard_sealed_lists()
    bad, checks = {}, {}
    for s in subjects:
        check = check_subject(s, allowlist)
        on = sorted(lists_of(sealed, s))
        never = sorted(x['name'] for x in allowlist.sources if s in x['subjects'] and x['never_for_training'] and x['name'] != unseal.allowlist_source)
        holding = sorted(x['name'] for x in allowlist.sources if s in x['subjects'])
        why = [w for w, hit in (('refused by bayesreg.io.guard.check_subject', not check.admitted), ('admitted without the unseal object', check.admitted and not check.admitted_by_unseal),
                                (f'never_for_training sources {never} (T0)', bool(never)), ('a B-line training subject (folds train/calib)', s in split.training_subjects),
                                (f'allowlist sources {holding}, not only the unsealed {unseal.allowlist_source!r}', holding != [unseal.allowlist_source]),
                                (f'sealed lists {on}, not exactly [{unseal.sealed_list!r}]', on != [unseal.sealed_list])) if hit]
        if why:
            bad[s] = why
        checks[s] = dict(step=check.step, item=check.item, admitted=check.admitted, admitted_by_unseal=check.admitted_by_unseal, sealed_lists=on, allowlist_sources=holding)
    if bad:
        raise SystemExit(f'A4p: subjects refused (D24, hold-out): {bad}')
    return dict(check='D24 subject admission of a hold-out pair (C23.8-2): bayesreg.io.guard.check_subject through the unseal object; never_for_training only in the unsealed source; not in the folds '
                      'training set (train and calib, both folds); subject ids only',
                subjects=list(subjects), folds_path=None, folds_sha256=split.sha256, n_training_subjects=len(split.training_subjects), allowlist_path=str(allowlist.path),
                allowlist_sha256=allowlist.sha256, unseal=unseal.record(), unsealed_source=unseal.allowlist_source, guard=checks)


def a4_admission(subjects, allowlist, unet):
    """D24 for A4p, independent of the registry: every subject on the development allowlist, not never_for_training,
    and not a B-line training subject (train or calib, either fold) of the folds file, whose sha256 must be the pinned
    one. STOP otherwise. Reads subject ids only (no image, no label).
    HB: an allowlist bound to an unseal object (a hold-out pair, C23.8-2) goes to a4_admission_unsealed."""
    from bayesreg.io.guard import load_dev_allowlist
    from bayesreg.seg.unet.data import load_folds
    from bayesreg.seg.unet.pins import PINNED_FOLDS_SHA256
    want = unet.folds_sha256 or PINNED_FOLDS_SHA256
    split = load_folds(unet.folds)
    if split.sha256 != want:
        raise SystemExit(f'A4p: {unet.folds} sha256 {split.sha256} != pinned {want}')
    if unsealed_allowlist(allowlist):
        return dict(a4_admission_unsealed(subjects, allowlist, split), folds_path=str(unet.folds))
    allow = load_dev_allowlist(allowlist)
    bad = {}
    for s in subjects:
        why = [w for w, hit in (('not on the development allowlist', not allow.allows(s)),
                                ('never_for_training', s in allow.never_for_training),
                                ('a B-line training subject (folds train/calib)', s in split.training_subjects)) if hit]
        if why:
            bad[s] = why
    if bad:
        raise SystemExit(f'A4p: subjects refused (D24): {bad}')
    return dict(check='D24 subject admission: on the development allowlist, not never_for_training, not in the folds '
                      'training set (train and calib, both folds); subject ids only',
                subjects=list(subjects), folds_path=str(unet.folds), folds_sha256=split.sha256,
                n_training_subjects=len(split.training_subjects), allowlist_path=str(allowlist),
                allowlist_sha256=allow.sha256)


def pair_structure_layer(pin, allowlist, unet):
    """StructureLayer of the pair (checks 0-2 of bayesreg.seg.structure_layer: grouping, repository registry, D24
    pair-level preflight), predictions never generated here (infer 'never'); STOP on a refusal.
    HB (C23.8-2, C23.4-8): an allowlist bound to an unseal object is handed to StructureLayer as `allowlist=` (the very object load_pair admitted the pair through); its file is the source's path."""
    from jd_curl.validation import InputValidationError
    from bayesreg.io.pair import ProvenanceRejected
    from bayesreg.seg.structure_layer import REGISTERED, StructureLayer, StructureSource
    other = {} if unet.segmenter_id is None else dict(segmenter_id=unet.segmenter_id, formal_registry=Path(unet.registry))
    bound = allowlist if unsealed_allowlist(allowlist) else None
    src = StructureSource(root=Path(unet.pred_root), registry_path=_registry_of(unet), allowlist_path=Path(allowlist if bound is None else bound.path),
                          models_root=Path(unet.models_root), image_root=Path(unet.image_root), infer='never', **other)
    try:
        layer = StructureLayer(pin, src) if bound is None else StructureLayer(pin, src, allowlist=bound)
    except (ProvenanceRejected, InputValidationError) as exc:
        raise SystemExit(f'A4p {pin.fixed_subject} <- {pin.moving_subject}: structure layer refused '
                         f'({type(exc).__name__}: {exc})')
    if layer.registry_record['status'] != REGISTERED:
        raise SystemExit(f"A4p: registry {layer.registry_record['path']} is not the repository registry")
    git = layer.registry_record.get('git') or {}
    if not (git.get('tracked') is True and git.get('clean') is True):           # A4 review item 3
        raise SystemExit(f"A4p: the repository registry {layer.registry_record['path']} must be committed and clean "
                         f"(git tracked {git.get('tracked')}, clean {git.get('clean')}, status {git.get('porcelain')!r}"
                         f"{', ' + git['note'] if git.get('note') else ''})")
    return layer


def _dice(a, b, what):
    if a.shape != b.shape:
        raise SystemExit(f'A4p {what}: arrays of shape {a.shape} and {b.shape} differ')
    na, nb, both = int(a.sum()), int(b.sum()), int(np.count_nonzero(a & b))
    return (2.*both/(na+nb) if na+nb else None), na, nb, both


def alignment_qc(u_wm, a_wm, *, threshold=ALIGN_QC_MIN, level=ALIGN_QC_LEVEL):
    """Label-free alignment QC of one image (protocol §12.4 item 5, revision 2): Dice of {U-Net cerebral_wm group > .5}
    and {Atropos WM posterior > .5}, both on the image's zyx grid (two segmenters of the same image; Atropos is
    unsupervised). Returns the record; `passed` is Dice >= threshold (both sets empty: Dice undefined, not passed)."""
    dice, nu, na, both = _dice(np.asarray(u_wm) > level, np.asarray(a_wm) > level, 'alignment QC')
    return dict(check=f'Dice({{U-Net {UNET_WM_GROUP} > {level:g}}}, {{Atropos WM > {level:g}}}), label-free '
                      '(protocol §12.4 item 5, revision 2)', dice=dice, threshold=threshold, level=level,
                passed=dice is not None and dice >= threshold, n_unet_wm=nu, n_atropos_wm=na, n_both=both)


def support_dice_record(I, p0):
    """Revision 1's QC, since revision 2 a record only (protocol §12.4 item 5): Dice of {I > 0} (the skull-stripped
    image's support) and {P_0 < .5} (the voxels the U-Net does not call background). Never stops."""
    dice, ni, nb, both = _dice(np.asarray(I) > 0, np.asarray(p0) < .5, 'support Dice')
    return dict(check='Dice({I > 0}, {P_0 < .5}), label-free; record only since revision 2 (protocol §12.4 item 5)',
                dice=dice, stops=False, revision1_threshold=SUPPORT_REVISION1_MIN,
                below_revision1_threshold=dice is None or dice < SUPPORT_REVISION1_MIN, n_image_positive=ni,
                n_posterior_brain=nb, n_both=both)


def atropos_wm(seg_root, pin, side, allowlist=None):
    """(Atropos WM posterior (nz, ny, nx) float64, record) of one image for the A4p alignment QC, read through the A2
    path `posteriors`: tissue-layer provenance validated (atropos, no label, wm_index naming WM) and bound to the
    manifest's source image, output files re-hashed, posterior files transposed xyz -> zyx, grid equal to the image's.
    Here also: every posterior file is among the hash-verified outputs and there is one per class name. STOP when the
    output is missing or refused (protocol §12.4 item 5, revision 2)."""
    from jd_curl.validation import InputValidationError
    from bayesreg.io.pair import ProvenanceRejected
    subject = getattr(pin, side+'_subject')
    if seg_root is None:
        raise SystemExit('A4p: the alignment QC needs the Atropos posteriors (--seg-root); none given')
    directory = Path(seg_root)/subject
    if not (directory/'provenance.json').is_file():
        raise SystemExit(f'A4p {side} {subject}: no Atropos posteriors at {directory} (no provenance.json; alignment QC, '
                         'protocol §12.4 item 5)')
    try:
        P, prov, accepted = posteriors(Path(seg_root), subject, pin, side, allowlist=allowlist)
    except (ProvenanceRejected, InputValidationError, OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f'A4p {side} {subject}: Atropos posteriors refused ({type(exc).__name__}: {exc}; alignment QC, '
                         'protocol §12.4 item 5)')
    files, names = list(prov['posterior_files']), list(accepted['class_names'])
    if len(files) != len(names) or P.shape[0] != len(names) or not set(files) <= set(prov['output_sha256']):
        raise SystemExit(f'A4p {side} {subject}: Atropos posterior files {files} must be one per class {names}, each '
                         'among the hash-verified outputs')
    k = int(accepted['wm_index'])
    wm = np.array(P[k])
    del P
    record = dict(directory=str(directory), provenance_sha256=sha_file(directory/'provenance.json'),
                  output_sha256=dict(prov['output_sha256']), segmenter_id=accepted['segmenter_id'],
                  model_ids=list(accepted['model_ids']), class_names=names, wm_index=k,
                  wm_class_name=accepted['wm_class_name'], wm_file=files[k],
                  read='experiments/d0/make_inputs.py posteriors (the A2 path): nibabel, transpose(2, 1, 0), float64')
    return wm, record


def _repo_registry():
    from bayesreg.seg import structure_layer as sl
    return sl.REPO_REGISTRY


def _registry_of(unet):
    """The repository registry of this source: configs/bayesreg/segmenters.json unless A5Pv2 names another."""
    return _repo_registry() if unet.registry is None else Path(unet.registry)


def _recipe(arm, unet):
    return RECIPES[arm] if unet is None or unet.segmenter_id is None else RECIPES[arm].replace(RECIPE_SEGMENTER, unet.segmenter_id)


def unet_posterior(layer, side):
    """(P (36, nz, ny, nx) float64 divided by the channel sum, record, cerebral_wm group (nz, ny, nx) of the read-back
    structure_groups.npz) of one image of the pair after every StructureLayer check (checks 3-4), the all_folds_mean
    policy, the read-back raw-sum check of probs36.npz and the agreement of its group sums with structure_groups.npz;
    STOP otherwise."""
    from jd_curl.validation import InputValidationError
    from bayesreg.io.pair import ProvenanceRejected
    from bayesreg.seg.channels import read_back_probabilities
    subject = layer.subjects[side]
    ids, policy = layer.expected[side]
    if policy != 'all_folds_mean':
        raise SystemExit(f'A4p {side} {subject}: policy {policy} != all_folds_mean (D24: a development subject is in no '
                         "fold model's label contact)")
    try:
        groups, accepted, rec = layer.load(side)
    except (ProvenanceRejected, InputValidationError) as exc:
        raise SystemExit(f'A4p {side} {subject}: structure layer refused ({type(exc).__name__}: {exc})')
    directory = Path(rec['directory'])
    if 'probs36.npz' not in rec['verified_outputs']:
        raise SystemExit(f'A4p {side} {subject}: probs36.npz is not among the hash-verified outputs')
    try:
        P, read_back = read_back_probabilities(directory/'probs36.npz', shape=(N_CLASSES,)+groups.shape[1:])
    except InputValidationError as exc:
        raise SystemExit(f'A4p {side} {subject}: probs36.npz refused by the read-back ({exc})')
    g = layer.grouping
    if UNET_WM_GROUP not in g.group_names \
            or tuple(g.group_labels[g.group_names.index(UNET_WM_GROUP)]) != CEREBRAL_WM:
        raise SystemExit(f'A4p {side} {subject}: grouping {g.name} has no group {UNET_WM_GROUP} of seg35 labels '
                         f'{CEREBRAL_WM} (alignment QC, protocol §12.4 item 5)')
    wm_channel = 1+g.group_names.index(UNET_WM_GROUP)                  # channel 0 = background group
    u_wm = np.array(groups[wm_channel])
    deviation = 0.
    for k, labels in enumerate((g.background_labels,)+tuple(g.group_labels)):
        deviation = max(deviation, float(np.max(np.abs(P[list(labels)].sum(axis=0)-groups[k]))))
    del groups
    if not deviation <= GROUP_TOL:
        raise SystemExit(f'A4p {side} {subject}: probs36 summed by {g.name} differs from structure_groups.npz by '
                         f'{deviation:.3e} > {GROUP_TOL:g}')
    record = dict(unet_provenance_sha256=rec['provenance_sha256'], unet_output_sha256=rec['verified_outputs'],
                  unet_directory=str(directory), segmenter_id=accepted['segmenter_id'], model_ids=list(ids),
                  policy=policy, members=rec['models']['members'], registry_path=layer.registry.path,
                  registry_sha256=layer.registry.sha256, registry_status=layer.registry_record['status'],
                  registry_git={k: (layer.registry_record.get('git') or {}).get(k) for k in ('tracked', 'clean',
                                                                                             'last_commit')},
                  read_back_probs36=read_back, read_back_groups=rec['read_back'],
                  group_sum_max_abs_deviation=deviation, group_sum_tol=GROUP_TOL, grouping=rec['grouping'],
                  prediction_code=rec['prediction_code'],
                  qc_wm_group=dict(file='structure_groups.npz', name=UNET_WM_GROUP, channel=wm_channel,
                                   labels=list(CEREBRAL_WM)))
    return P, record, u_wm


def build_arm(arm, inputs, seg_root, *, stage=STAGE, allowlist=ALLOWLIST, unet=None, only_pair=None, seg4_map=SEG4_MAP,
              colour_table=COLOUR_TABLE_A5, colour_table_sha256=COLOUR_TABLE_A5_SHA256):
    """only_pair: dry run, one pair of the stage's selection (the record says partial true; the queue refuses it)."""
    out = inputs/arm/STAGE_NAME
    if (inputs/arm).exists():
        raise SystemExit(f'{inputs/arm} exists; refusing to overwrite (STOP). To rebuild after a STOP, rename it first '
                         f'(e.g. {arm}.stopped_<UTC>; kept as evidence, protocol §12.8), then run make_inputs again')
    admission = None
    lookup, seg4_record = (load_seg4_map(seg4_map) if arm in A1P_ARMS else (None, None))    # before anything is written
    table, table_record = (load_colour_table(colour_table, colour_table_sha256) if arm in A5_ARMS else (None, None))
    if arm in UNET_ARMS:                                # subject and registry checks before anything is written
        if unet is None:
            raise SystemExit(f'{arm}: a UNetSource is required')
        admission = a4_admission(stage_subjects(stage), allowlist, unet)
        from bayesreg.seg.structure_layer import is_formal_registry, registry_git_state
        if unet.registry is not None and not is_formal_registry(unet.registry):      # UNV2 review, important 2
            raise SystemExit(f'A5Pv2: {unet.registry} is not one of the repository registries '
                             '(bayesreg.seg.structure_layer.FORMAL_REGISTRIES); nothing written')
        if not Path(_registry_of(unet)).is_file():
            raise SystemExit(f'A4p: no repository registry at {_registry_of(unet)} (no other registry is ever read); '
                             'nothing written')
        git = registry_git_state(_registry_of(unet))
        if not (git.get('tracked') is True and git.get('clean') is True):       # A4 review item 3
            raise SystemExit(f'A4p: the repository registry {_registry_of(unet)} must be committed and clean (git tracked '
                             f"{git.get('tracked')}, clean {git.get('clean')}, status {git.get('porcelain')!r}); "
                             'nothing written')
        missing = [s for s in admission['subjects']
                   if seg_root is None or not (Path(seg_root)/s/'provenance.json').is_file()]
        if missing:                                     # protocol §12.4 item 5 (revision 2): the QC needs Atropos
            raise SystemExit(f'A4p: no Atropos posteriors under {seg_root} for {missing} (the alignment QC of protocol '
                             '§12.4 item 5 needs them); nothing written')
    out.mkdir(parents=True)
    files = {}
    for name in ('selection.json', 'completion.json'):
        shutil.copyfile(stage/name, out/name)
        if sha_file(out/name) != sha_file(stage/name):
            raise SystemExit(f'{name}: copy differs')
        files[f'{STAGE_NAME}/{name}'] = sha_file(out/name)
    sel = json.loads((stage/'selection.json').read_text())
    pairs = {}
    if only_pair is not None and only_pair not in [row['pair'] for row in sel['pairs']]:
        raise SystemExit(f'--only-pair {only_pair}: not in {stage}/selection.json')
    for row in sel['pairs']:
        name = row['pair']
        if only_pair is not None and name != only_pair:
            continue
        src = stage/name
        manifest = json.loads((src/'manifest.json').read_text())
        if sha_file(src/'pair.npz') != manifest['pair_sha256']:
            raise SystemExit(f'{name}: source pair.npz sha256 differs from its manifest')
        pin = load_pair(src/'pair.npz', allowlist=allowlist)            # subject guard + firewall view (A2, A4p colour from it)
        with np.load(src/'pair.npz') as z:
            if tuple(z.files) != KEY_ORDER:
                raise SystemExit(f'{name}: unexpected members {z.files}')
            arrays = {k: np.array(z[k]) for k in z.files}
        record = dict(source_pair_sha256=manifest['pair_sha256'], source_manifest_sha256=sha_file(src/'manifest.json'))
        layer = pair_structure_layer(pin, allowlist, unet) if arm in UNET_ARMS else None
        if layer is not None:
            record['structure_layer'] = layer.record()
        for side in ('fixed', 'moving'):
            I = arrays[side]
            if not np.array_equal(I, getattr(pin, side)):
                raise SystemExit(f'{name}:{side}: firewall view differs from the archive member')
            P = prov = None
            if arm == 'A2':
                subject = getattr(pin, side+'_subject')
                P, prov, _ = posteriors(seg_root, subject, pin, side)
                J, consts = colour(arm, getattr(pin, side), None, None, P)          # no label is passed for A2
                consts.update(atropos_provenance_sha256=sha_file(seg_root/subject/'provenance.json'),
                              atropos_output_sha256=prov['output_sha256'])
            elif arm in UNET_ARMS:
                P, unet_record, u_wm = unet_posterior(layer, side)
                support = support_dice_record(getattr(pin, side), P[0])           # record only (revision 2)
                a_wm, atropos_record = atropos_wm(seg_root, pin, side)
                qc = alignment_qc(u_wm, a_wm)
                del u_wm, a_wm
                if not qc['passed']:
                    raise SystemExit(f"{arm} {name}:{side}: alignment QC Dice({{U-Net {UNET_WM_GROUP} > .5}}, "
                                     f"{{Atropos WM > .5}}) = {qc['dice']} < {qc['threshold']} (protocol §12.4 item 5, "
                                     'revision 2; the pair is refused)')
                if arm in A1P_ARMS:
                    P = merge_seg4(P, lookup)
                J, consts = colour(arm, getattr(pin, side), None, None, P, table=table)      # no label is passed for these arms
                consts.update(unet_record, alignment_qc=qc, support_dice_record=support, atropos_qc=atropos_record)
            else:
                J, consts = colour(arm, I, arrays[side+'_seg4'], arrays[side+'_seg35'], None, table=table)
            del P
            if J.shape != I.shape or J.dtype != np.float64 or not np.all(np.isfinite(J)):
                raise SystemExit(f'{name}:{side}: bad coloured image')
            arrays[side] = J
            record[side] = dict(subject=getattr(pin, side+'_subject'), **consts,
                                image_min=float(J.min()), image_max=float(J.max()),
                                original_image_mean_in_nonzero=float(I[I > 0].mean()),
                                coloured_mean_in_nonzero=float(J[J > 0].mean()))
        dest = out/name
        dest.mkdir()
        np.savez_compressed(dest/'pair.npz', **{k: arrays[k] for k in KEY_ORDER})
        with np.load(dest/'pair.npz') as z:                                  # copied members bitwise, order kept
            if tuple(z.files) != KEY_ORDER or any(not np.array_equal(z[k], arrays[k]) or z[k].dtype != arrays[k].dtype
                                                  for k in KEY_ORDER):
                raise SystemExit(f'{name}: written pair.npz does not reproduce the arrays')
        new = dict(manifest)
        new['pair_sha256'] = sha_file(dest/'pair.npz')
        new['preprocessing_d0'] = dict(
            arm=arm, recipe=_recipe(arm, unet), oracle=arm in ORACLE_ARMS, protocol='docs/bayesreg/05_D0协议.md',
            script='experiments/d0/make_inputs.py', script_sha256=sha_file(Path(__file__)),
            seed=SEED if arm in SEEDED else None, z_offsets=z_offsets().tolist() if arm in SEEDED else None,
            source_pair_sha256=record['source_pair_sha256'], source_manifest_sha256=record['source_manifest_sha256'],
            source_preprocessing=manifest.get('preprocessing'), source_baseline_native_mse=new.pop('baseline_native_mse', None),
            colour=dict(fixed=record['fixed'], moving=record['moving']),
            note='fixed/moving replaced by the coloured images; labels and initial_u copied bitwise; baseline_label_metrics '
                 'unchanged (identity map, same labels)')
        if table_record is not None:
            new['preprocessing_d0']['colour_table'] = table_record
        new['preprocessing'] = f"D0 {arm}: {_recipe(arm, unet)}; on top of: {manifest.get('preprocessing')}"
        (dest/'manifest.json').write_text(json.dumps(new, indent=1)+'\n')
        files[f'{STAGE_NAME}/{name}/pair.npz'] = new['pair_sha256']
        files[f'{STAGE_NAME}/{name}/manifest.json'] = sha_file(dest/'manifest.json')
        pairs[name] = record
        print(f"{arm} {name[11:15]}<-{name[-8:-4]}: fixed [{record['fixed']['image_min']:.3f}, {record['fixed']['image_max']:.3f}] "
              f"moving [{record['moving']['image_min']:.3f}, {record['moving']['image_max']:.3f}]", flush=True)
    rec = dict(arm=arm, recipe=_recipe(arm, unet), oracle=arm in ORACLE_ARMS, stage=STAGE_NAME, source_stage=str(stage),
               created_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
               script_sha256=sha_file(Path(__file__)), seed=None if arm in A5_ARMS else SEED, sigma_voxel=SIGMA if arm != 'A2' else None,
               z_offsets=z_offsets().tolist() if arm in SEEDED else None, files_sha256=files, pairs=pairs)
    if only_pair is not None:                           # the queue's input check needs every pair: it refuses this record
        rec['partial'] = True
    if seg4_record is not None:
        rec['seg4_map'] = seg4_record
    if table_record is not None:
        rec['colour_table'] = table_record
    if arm in UNET_ARMS:
        regs = {(p[s]['registry_path'], p[s]['registry_sha256'], p[s]['registry_status'],
                 tuple(sorted(p[s]['registry_git'].items()))) for p in pairs.values() for s in ('fixed', 'moving')}
        if len(regs) != 1:
            raise SystemExit(f'A4p: the pairs were checked against different registries {sorted(regs)}')
        (path, sha, status, git), = regs
        images = [p[s] for p in pairs.values() for s in ('fixed', 'moving')]
        support = [c['support_dice_record']['dice'] for c in images]
        rec.update(subject_admission=admission, unet_source=unet.record(),
                   structure_registry=dict(path=path, sha256=sha, status=status, git=dict(git)),
                   atropos_root=str(seg_root),
                   alignment_qc=dict(rule=f'Dice({{U-Net {UNET_WM_GROUP} > .5}}, {{Atropos WM > .5}}) >= threshold per '
                                          'image, else STOP (protocol §12.4 item 5, revision 2)',
                                     threshold=ALIGN_QC_MIN, level=ALIGN_QC_LEVEL,
                                     min_dice=min(c['alignment_qc']['dice'] for c in images),
                                     basis='runs/20261030-d0-a4/qc_survey_20261030/ (aligned .853-.881; flips and shifts '
                                           '>= 2 voxels <= .792; 1-voxel shifts not caught)'),
                   support_dice_record=dict(rule='Dice({I > 0}, {P_0 < .5}) per image, recorded, never stops (revision '
                                                 '1 stop rule >= .8, a record since revision 2)',
                                            min_dice=None if None in support else min(support),
                                            max_dice=None if None in support else max(support),
                                            n_below_revision1_threshold=sum(c['support_dice_record']
                                                                            ['below_revision1_threshold']
                                                                            for c in images)),
                   group_sum_tol=GROUP_TOL)
    (inputs/arm/'d0_inputs.json').write_text(json.dumps(rec, indent=1)+'\n')


def write_a4_subjects(path, *, stage=STAGE, allowlist=ALLOWLIST, unet=UNetSource(), only_missing=False):
    """The subject list for `python -m bayesreg.seg.unet.cli infer --subjects-file`: the stage's subjects after the
    D24 admission (ids only; no image, no label read). only_missing: leave out subjects whose prediction directory
    already has provenance.json (inference never overwrites). Never overwrites `path`; never under d0_v1/."""
    path = Path(path)
    if _inside(path, D0V1):
        raise SystemExit(f'{path}: A4p files are never written under {D0V1}')
    if path.exists():
        raise SystemExit(f'{path} exists; refusing to overwrite (STOP)')
    subjects = stage_subjects(stage)
    admission = a4_admission(subjects, allowlist, unet)
    have = [s for s in subjects if (Path(unet.pred_root)/s/'provenance.json').is_file()]
    chosen = [s for s in subjects if not (only_missing and s in have)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(s+'\n' for s in chosen))
    print(json.dumps(dict(admission, written=str(path), n_written=len(chosen), already_predicted=have,
                          only_missing=only_missing), indent=1))
    return chosen


def _inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=['A1', 'A2', 'A3'], choices=list(ARMS))
    ap.add_argument('--inputs', type=Path, default=None,
                    help=f'default {D0V1}/inputs for A1-A3, {D0A4}/inputs for A4p, {D0A1P}/inputs for A1Ph and A1Ps, {D0A5}/inputs for A5 and A5P')
    ap.add_argument('--stage', type=Path, default=STAGE, help=f'the source stage (default {STAGE})')
    ap.add_argument('--only-pair', default=None, help='dry run: one pair; the record says partial true and the queue refuses it')
    ap.add_argument('--seg-root', type=Path, default=Path('/home/rmcfz/seq_vp_tag_results/seg/atropos_v1'),
                    help='Atropos outputs <root>/<subject>/: the A2 colouring and the A4p alignment QC')
    ap.add_argument('--unet-pred-root', '--pred-root', dest='unet_pred_root', type=Path, default=None,
                    help=f'A4p, A1P, A5P: U-Net predictions <root>/<subject>/ (default {UNET_ROOT/"pred"})')
    ap.add_argument('--segmenter-id', default=None, help='A5Pv2 only: the segmenter of the predictions (default unet35_v1_ens)')
    ap.add_argument('--registry', type=Path, default=None,
                    help='A5Pv2 only: its repository registry, committed and clean (default configs/bayesreg/segmenters.json)')
    ap.add_argument('--bz-decision-run-a5pv2', default=None, metavar='WHERE_BZ_DECIDED_IT',
                    help='BZ 决定 (BZ decision), A5Pv2 only: go past the preregistered rule of runs/20261103-unv2/plan.md; recorded')
    ap.add_argument('--write-a4-subjects', type=Path, default=None,
                    help='write the A4p subject list for the U-Net inference and exit')
    ap.add_argument('--only-missing', action='store_true', help='with --write-a4-subjects: subjects without a prediction')
    a = ap.parse_args(argv)
    pred_given = a.unet_pred_root is not None
    a.unet_pred_root = a.unet_pred_root if pred_given else UNET_ROOT/'pred'
    other_segmenter = a.segmenter_id is not None or a.registry is not None
    a5pv2 = {}
    if a.bz_decision_run_a5pv2 is not None and not other_segmenter:
        raise SystemExit('--bz-decision-run-a5pv2 goes with A5Pv2 only (--segmenter-id, --registry)')
    if other_segmenter:                                 # A5Pv2: each of these named, or nothing of it
        if a.write_a4_subjects is not None or a.arms != ['A5P']:
            raise SystemExit('--segmenter-id and --registry (A5Pv2, runs/20261103-unv2/plan.md) go with --arms A5P only')
        if a.segmenter_id is None or a.registry is None or not pred_given or a.inputs is None:
            raise SystemExit('A5Pv2 names all of --segmenter-id, --registry, --pred-root and --inputs')
        if a.segmenter_id != A5PV2_SEGMENTER:
            raise SystemExit(f'A5Pv2: --segmenter-id must be {A5PV2_SEGMENTER}, got {a.segmenter_id!r}')
        if any(_inside(a.inputs, r) for r in (D0V1, D0A4, D0A1P, D0A5)):
            raise SystemExit(f'{a.inputs}: A5Pv2 inputs are never written under {D0V1}, {D0A4}, {D0A1P} or {D0A5}')
        a5pv2 = dict(segmenter_id=a.segmenter_id, registry=a.registry, models_root=a.unet_pred_root.parent,
                     gate=a5pv2_gate(a.unet_pred_root, a.bz_decision_run_a5pv2))
    unet = UNetSource(pred_root=a.unet_pred_root, **a5pv2)
    if a.write_a4_subjects is not None:
        write_a4_subjects(a.write_a4_subjects, stage=a.stage, unet=unet, only_missing=a.only_missing)
        return 0
    if 'A4p' in a.arms and len(a.arms) > 1:
        raise SystemExit('A4p is the supplementary arm of protocol §12: build it on its own (--arms A4p)')
    if set(a.arms) & set(A5_ARMS) and not set(a.arms) <= set(A5_ARMS):
        raise SystemExit('A5 and A5P (runs/20261103-a5/plan.md): build them without any other arm (--arms A5 A5P)')
    if set(a.arms) & set(A1P_ARMS) and not set(a.arms) <= set(A1P_ARMS):
        raise SystemExit('A1Ph and A1Ps (runs/20261102-a1p/plan.md): build them without A1-A3 (--arms A1Ph A1Ps)')
    for arm in a.arms:
        root = D0A5 if arm in A5_ARMS else D0A1P if arm in A1P_ARMS else D0A4 if arm == 'A4p' else D0V1
        inputs = a.inputs if a.inputs is not None else root/'inputs'
        if arm == 'A4p' and _inside(inputs, D0V1):
            raise SystemExit(f'{inputs}: A4p inputs are never written under {D0V1}')
        if arm in A1P_ARMS and (_inside(inputs, D0V1) or _inside(inputs, D0A4)):
            raise SystemExit(f'{inputs}: A1P inputs are never written under {D0V1} or {D0A4}')
        if arm in A5_ARMS and any(_inside(inputs, r) for r in (D0V1, D0A4, D0A1P)):
            raise SystemExit(f'{inputs}: A5 inputs are never written under {D0V1}, {D0A4} or {D0A1P}')
        build_arm(arm, inputs, a.seg_root, stage=a.stage, unet=unet if arm in UNET_ARMS else None, only_pair=a.only_pair)
    return 0


if __name__ == '__main__':
    sys.exit(main())
