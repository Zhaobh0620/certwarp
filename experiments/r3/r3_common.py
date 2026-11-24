"""The fixed names and numbers of work package R3a (runs/20261111-r3/任务书.md, plan.md): development round 2, second item -- the segmentation side as a composition of
L (the recipe of UF, v3_large, trained twice as long: 80000 steps), N (nnU-Net 3d_fullres, the recipe of runs/20261105-nn, two models per fold of our two folds) and, if R3b's calibration
diagnosis passes, D (an expected-Dice decoding of the small structures, bayesreg.seg.decode).  The composition is chosen on the training set's calibration subjects, out of fold, by a rule
written here before any reading; then one segmentation-level reading on the 20 development subjects (the last of round 2), and, past its gate, a 19-pair registration reading (another package).

Standard library only (imported by every R3 tool and its tests).  Nothing here reads a file but the helpers that are called; no constant is derived from a splits/ list.
UF's pins, the subjects of the development set, the label groups and the gates of R2 are taken from experiments/r2/r2_common.py (the same objects; a test checks them).
"""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE.parent/'r2') not in sys.path:
    sys.path.insert(0, str(HERE.parent/'r2'))
import r2_common as r2c  # noqa: E402

RESULTS, SEG, PY, MAIN_REPO = r2c.RESULTS, r2c.SEG, r2c.PY, r2c.MAIN_REPO
PLAN_REL = 'runs/20261111-r3/plan.md'

# ---- roots (all new; everything else is only read) ------------------------------------------------------------------------------------------------
OUT = SEG/'unet35_r3_full'                      # the chain: snapshot/, r3l/fold<F>_s3/, control_check/, nn_calib/, compose/, nn_dev/, eval_r3/, analysis/
SMOKE = SEG/'unet35_r3_smoke'                   # the GPU smoke before the chain: snapshot/, member/ (L, 400 steps), nnUNet_results_smoke/ (N, a few epochs), synthetic/
NN_ROOT = RESULTS/'nnunet_r3'                   # nnU-Net's three roots (raw links, preprocessed with our splits_final.json, results), train_runs/, models/<model>/train_record.json
NN_REFERENCE = RESULTS/'nnunet_v1'              # work package NN, only read: its plans are the recipe (runs/20261105-nn/prepare_record.md)
DEV_READ_LOCK = SEG/'r3_dev_read.lock'          # the one read of the development truth (eval_r3.py)
CALIB_READ_LOCK = SEG/'r3_calib_read.lock'      # the one read of the training set's calibration truth for the composition (compose_r3.py; BZ 11-11 ~2:22 PM CST, hard rule 6's reading (b))
DECODE_SWITCH = 'decode_switch.json'            # <OUT>/decode_switch.json: D on or off, written once by the lead after R3b's diagnosis, before the composition reads anything
R3B_DECISION = RESULTS/'r3b_calib_v1'/'analysis'/'decision_calib.json'     # R3b's decision record (r3b_decision.DEFAULT_PATH); D on only with its verdict 'pass'
DECODE_PY_REL = 'bayesreg/seg/decode.py'        # R3b's decoder: its sha256 is pinned in the switch and checked in the snapshot that composes (the review's I1)
DECISION_DECODE_KEY = 'decode_py_sha256'        # the key of decode.py's sha256 in R3b's decision record (R3b's review I1, the lead 11-11 ~3:57 PM CST); to align with R3b's final record

# ---- what R3 builds on, only read, pinned (R2's objects) -----------------------------------------------------------------------------------------
UF_OUT, UF_EVAL, UF_EVAL_SHA256, UF_ORDER, UF_MEMBER_SHA256 = r2c.UF_OUT, r2c.UF_EVAL, r2c.UF_EVAL_SHA256, r2c.UF_ORDER, r2c.UF_MEMBER_SHA256
SUBJECTS, SHORTS, N_SUBJECTS, N_PAIRS = r2c.SUBJECTS, r2c.SHORTS, r2c.N_SUBJECTS, r2c.N_PAIRS
LABELS, SMALL_SIX, BIG_SIX, OTHER_23 = r2c.LABELS, r2c.SMALL_SIX, r2c.BIG_SIX, r2c.OTHER_23
FOCUS_SUBJECT = r2c.FOCUS_SUBJECT
FOLDS = ('A', 'B')
N_CALIB_PER_FOLD = 8

# ---- L: UF's recipe, twice the steps ---------------------------------------------------------------------------------------------------------------
L_PROFILE = 'v3_large'                          # cli --profile v3_large, exactly UF's (no new profile, no new loss or sampling)
L_FAMILY = 'r3l'                                # <OUT>/r3l/fold<F>_s3
L_SEED = 3                                      # UF has seeds 0 and 1, R2 seed 2: new initial weights and a new data order
L_MEMBERS = (('A', L_SEED), ('B', L_SEED))
L_STEPS = 80000                                 # UF's 40000 doubled (the brief); the poly schedule over all of them, no early stop, the last step
L_WALL_SECONDS = 28800.                         # 8 h per member, about 1.6 x the estimate (UF's largest .2218 s x 80000 = 4.9 h); a member stopped by it did not finish, the chain stops
EVAL_EVERY = 500                                # UF's
UF_CHANGES = ('max_iters', 'min_iters', 'patience', 'max_wall_seconds')     # the only keys where an L member's config may differ from its UF namesake's (UF's eval_every, checkpoint stay)


def member_name(fold, seed):
    return f'fold{fold}_s{seed}'


L_NAMES = tuple(member_name(f, s) for f, s in L_MEMBERS)


def l_train_config():
    """The --set values of L: UF's six keys (uf_common.TRAIN_CONFIG) with the schedule doubled and the wall stop of 8 h."""
    return dict(max_iters=L_STEPS, min_iters=L_STEPS, patience=L_STEPS, eval_every=EVAL_EVERY, checkpoint='last', max_wall_seconds=L_WALL_SECONDS)


# ---- N: nnU-Net, the recipe of runs/20261105-nn, our two folds, two seeded models per fold --------------------------------------------------------------
NN_BASE_TRAINER = 'nnUNetTrainer_250epochs_NoMirroring'       # NN's trainer (no mirroring: 32 of 35 labels are Left-/Right-; nn_common.py), 250 epochs
NN_EPOCHS = 250
NN_SEEDS = {'A': (0, 1), 'B': (2, 3)}           # the seed of the initial weights (ext_trainers/nn_trainers_r3.py): four distinct seeds, so no two of the four models start alike
NN_FOLD_INDEX = {'A': 0, 'B': 1}                 # splits_final.json: fold 0 = our fold A (98 train, 8 val = its calibration subjects), fold 1 = fold B
NN_WALL_SECONDS = 36000                          # 10 h per model (NN measured 5.84 h for 250 epochs): nn_r3.train_model stops nnUNetv2_train (its own session) itself past it, records it, the chain stops;
                                                 # the step's limit (run_r3.TIMEOUTS) is 30 minutes longer, so the inner one comes first (the review's m2, m9)
NN_INFER_WALL_SECONDS = 6600                     # the inference child of one set (nn_r3.infer_set stops it itself); the step's limit is 7200 s
NN_DEVICE_CAP_GIB = 9.                           # the inference child's device cap (experiments/fl/nx_infer_child.py's LIMIT_BYTES)


def nn_trainer(seed):
    return f'{NN_BASE_TRAINER}_r3s{seed}'


def nn_name(fold, seed):
    return f'nn{fold}_s{seed}'


NN_MODELS = tuple((f, s) for f in FOLDS for s in NN_SEEDS[f])     # nnA_s0, nnA_s1, nnB_s2, nnB_s3: the order of training and of the inference's sum
NN_NAMES = tuple(nn_name(f, s) for f, s in NN_MODELS)


def nn_models_of_fold(fold):
    return [nn_name(fold, s) for s in NN_SEEDS[fold]]


def nn_model_of(name):
    """(fold, seed) of a model name nn<F>_s<seed>."""
    fold, seed = name.removeprefix('nn').split('_s')
    if (fold, int(seed)) not in NN_MODELS:
        raise ValueError(f'{name}: not a model of R3 ({NN_NAMES})')
    return fold, int(seed)


# ---- the candidates of the composition (closed; written before any reading) -----------------------------------------------------------------------------
# A candidate is a recipe of members per fold; the composition delivered is the winner's type over both folds.  Weights: every model equal (an L member counts as a UF member, an nnU-Net model as
# one model): the U-Net family's mean P_U (in the order below, as ensemble_probabilities adds), then P = (1 - w) P_U + w P_N with w = n_N / (n_U + n_N), P_N the mean of the nnU-Net models.
# The family shares are the same within a fold and over both folds (each count doubles), so a candidate read out of fold is the same mixture as the composition delivered.
CANDIDATE_BASES = {
    'UF':      dict(uf=('s0', 's1'), l=False, nn=False, words='UF 本折两个成员（UF4 在本折的对应）'),
    'UF+L':    dict(uf=('s0', 's1'), l=True,  nn=False, words='UF 本折两个成员 + L 本折一个成员，等权'),
    'Lswap':   dict(uf=('s0',),      l=True,  nn=False, words='L 替换 UF 本折种子 1 的成员（事先定死；UF 两个种子的成员地位相同）'),
    'UF+N':    dict(uf=('s0', 's1'), l=False, nn=True,  words='UF 本折两个成员 + nnU-Net 本折两个模型，每个模型等权（两家族各 1/2）'),
    'UF+L+N':  dict(uf=('s0', 's1'), l=True,  nn=True,  words='UF 本折两个成员 + L 一个 + nnU-Net 两个，每个模型等权（U-Net 3/5，nnU-Net 2/5）'),
}
BASE_ORDER = ('UF', 'UF+L', 'Lswap', 'UF+N', 'UF+L+N')
DECODE_SUFFIX = '+D'
REFERENCE = 'UF'                                 # the seg4 floor is against it; a winner equal to it means "no change": no development reading


def candidate_order(decode_on):
    """The closed list in its order: the five bases, then (only with D on) the same five with D."""
    return BASE_ORDER+(tuple(b+DECODE_SUFFIX for b in BASE_ORDER) if decode_on else ())


def base_of(candidate):
    return candidate[:-len(DECODE_SUFFIX)] if candidate.endswith(DECODE_SUFFIX) else candidate


def decodes(candidate):
    return candidate.endswith(DECODE_SUFFIX)


def unet_members(candidate, fold):
    """[(family, member name)] of the U-Net family of one fold, in the order of addition: UF's (seed 0, then 1), then L's."""
    base = CANDIDATE_BASES[base_of(candidate)]
    out = [('uf', f'fold{fold}_{s}') for s in base['uf']]
    if base['l']:
        out.append((L_FAMILY, member_name(fold, L_SEED)))
    return out


def nn_members(candidate, fold):
    return nn_models_of_fold(fold) if CANDIDATE_BASES[base_of(candidate)]['nn'] else []


def nn_weight(candidate, fold='A'):
    """w = n_N / (n_U + n_N) of a candidate (the same in each fold and over both)."""
    n_u, n_n = len(unet_members(candidate, fold)), len(nn_members(candidate, fold))
    return n_n/(n_u+n_n)


def n_models(candidate):
    """Models per fold: the simplicity of the tie rule."""
    return len(unet_members(candidate, 'A'))+len(nn_members(candidate, 'A'))


def delivered(candidate):
    """The composition over both folds (what the development reading evaluates and a registration would register): U-Net members fold A then fold B, nnU-Net models, w, D."""
    return dict(candidate=candidate, unet=[f'{f}/{n}' for fold in FOLDS for f, n in unet_members(candidate, fold)], nn=[n for fold in FOLDS for n in nn_members(candidate, fold)],
                nn_weight=nn_weight(candidate), decode=decodes(candidate), words=CANDIDATE_BASES[base_of(candidate)]['words']+('，再做 D（期望 Dice 解码）' if decodes(candidate) else ''))


# ---- the rule (written before any reading; the brief) ---------------------------------------------------------------------------------------------------
SEG4_TOLERANCE = .001                            # a candidate whose 16-subject mean seg4 is more than .001 below REFERENCE's is not eligible
TIE_BAND = .0005                                 # eligible candidates within .0005 of the best seg35 form the band; the simplest of the band wins


def finite(value):
    import math
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def choose(means, order):
    """The composition rule on the 16 out-of-fold calibration subjects' means ({candidate: dict(seg35, seg4)}), the candidates in `order` (candidate_order):
      1. eligible: seg35 and seg4 defined, and seg4 >= REFERENCE's seg4 - .001 (REFERENCE itself always, if defined);
      2. best: the highest mean seg35 of the eligible (ties: the earlier in `order`);
      3. the band: the eligible with seg35 > best - .0005 (strictly: a difference of .0005 or more is not 'less than .0005');
      4. the winner: the simplest of the band -- fewer models per fold first, then without D before with D (the brief's order: 成员更少、不带 D 的优先), then REFERENCE's base (UF, UF+D)
         before the others (the lead's ruling on the review's I3, 11-11 ~3:43 PM CST: no change wins a tie of size and D), then the higher seg35, then the earlier in `order`.
    Returns dict(winner, best, eligible, excluded {candidate: reason}, band, ranking).  REFERENCE's seg35 or seg4 undefined: nothing is eligible, winner None."""
    ref = means.get(REFERENCE) or {}
    excluded, eligible = {}, []
    for c in order:
        m = means.get(c) or {}
        if not (finite(ref.get('seg35')) and finite(ref.get('seg4'))):
            excluded[c] = f'{REFERENCE} seg35 or seg4 undefined: nothing is chosen'
        elif not (finite(m.get('seg35')) and finite(m.get('seg4'))):
            excluded[c] = 'seg35 or seg4 undefined'
        elif c != REFERENCE and m['seg4'] < ref['seg4']-SEG4_TOLERANCE:
            excluded[c] = f"seg4 {m['seg4']:.5f} more than {SEG4_TOLERANCE} below {REFERENCE}'s {ref['seg4']:.5f}"
        else:
            eligible.append(c)
    if not eligible:
        return dict(winner=None, best=None, eligible=[], excluded=excluded, band=[], ranking=[])
    position = {c: i for i, c in enumerate(order)}
    ranking = sorted(eligible, key=lambda c: (-means[c]['seg35'], position[c]))
    best = ranking[0]
    band = [c for c in ranking if means[best]['seg35']-means[c]['seg35'] < TIE_BAND]
    winner = sorted(band, key=lambda c: (n_models(c), decodes(c), base_of(c) != REFERENCE, -means[c]['seg35'], position[c]))[0]
    return dict(winner=winner, best=best, eligible=eligible, excluded=excluded, band=band, ranking=ranking)


# ---- the development reading's gate: BZ's (b), 11-11 ~3:38 PM CST (decision_log; fixed before any R3 training or reading; R2's bars are not used) -------------
SEG_BAR = .002                                  # COMBO - UF4, seg35, the mean over the 20 paired subjects
SEG_POSITIVE_MIN = 13                           # at least 13 of 20 subjects positive
SEG4_FLOOR = -.001                              # COMBO - UF4, seg4, mean
UF_PAIRED_SE = r2c.UF_PAIRED_SE
SEG_BOOT_DRAWS, SEG_BOOT_SEED = 20000, 20261111
CONTROL_TOL_FORWARD = r2c.CONTROL_TOL_FORWARD
OX_SLOPE = r2c.OX_SLOPE
C_FLOOR_GIB = r2c.C_FLOOR_GIB
NN_GATE_TOLERANCE = 1e-4                         # nx_seg.GATE_TOLERANCE: each model's stored argmax against nnU-Net's own label file, per subject (an orientation check)


# ---- small helpers (r2_common's) ---------------------------------------------------------------------------------------------------------------------
sha256_file, write_text_once, write_json_once, read_json, no_nan = r2c.sha256_file, r2c.write_text_once, r2c.write_json_once, r2c.read_json, r2c.no_nan
pair_subjects = r2c.pair_subjects


def plan_record(repo):
    path = Path(repo)/PLAN_REL
    return dict(path=PLAN_REL, sha256=sha256_file(path) if path.is_file() else None)


def json_sha256(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode('utf-8')).hexdigest()
