"""The fixed names and numbers of work package R2 (runs/20261110-r2/任务书.md, plan.md): development round 2, first item -- the pp diagnosis, a segmenter recipe with an emphasis on the small and
subcortical structures (one recipe, r2s: the lead's ruling of 11-10 ~10:42 PM CST), one segmentation-level reading, and, past its gate, one 19-pair registration reading.  Standard library only: imported by every R2 tool and its tests.

Nothing here reads a file but the helpers that are called; no constant is derived from a splits/ list.
"""
import hashlib
import json
from pathlib import Path

RESULTS = Path('/home/rmcfz/seq_vp_tag_results')
SEG = RESULTS/'seg'
PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'
MAIN_REPO = Path('/home/rmcfz/code/python_projects/seq_vp_tag')
PLAN_REL = 'runs/20261110-r2/plan.md'

# ---- roots (all new; the others are only read) ----------------------------------------------------------------------------------------------------
PP_ROOT = RESULTS/'r2_pp_v1'                    # the pp diagnosis (pp_r2.py): pp/, opened_files.json, pp_r2.json; then pp_split.json, targets.json, pp_report.md
OUT = SEG/'unet35_r2_full'                      # the training chain (run_r2.py): snapshot/, <recipe>/fold<F>_s<seed>/, control_check/, eval_r2/, analysis/
FIELDS_ROOT_NAME = 'r2_v1'                      # the 19-pair registration reading (run_r2f.py), RESULTS/r2_v1
DEV_READ_LOCK = SEG/'r2_dev_read.lock'          # the one read of the development truth by eval_r2 (r2_lock.py)
SMOKE = SEG/'unet35_r2_smoke'                   # the GPU smoke before the training (smoke_r2.py): another root, its own snapshot, the training set only, no lock
REGISTRY_REL = 'configs/bayesreg/segmenters_r2_v1.json'      # the candidate's registry, written once after the segmentation-level gate and the review (r2_seg.py)

# ---- what R2 builds on, only read, pinned ---------------------------------------------------------------------------------------------------------
UF_OUT = SEG/'unet35_v4_full'                   # UF: the four members of unet35_v4_ens (UF4)
UF_EVAL = UF_OUT/'eval_uf'/'eval_uf.json'
UF_EVAL_SHA256 = '400e0736a263ba7de9b822591c67e653e93a6809b28feb4b207b411c437e3a39'
UF_ORDER = ('foldA_s0', 'foldA_s1', 'foldB_s0', 'foldB_s1')     # eval_uf.ENSEMBLE_ORDER, the registry order of unet35_v4_ens
UF_MEMBER_SHA256 = {'foldA_s0': '52522d1f9cb6ad46f295542b94746d48f4f504a2db07b2dad1d7e4154d2852fd', 'foldA_s1': '15dea2900bc9acc453a423bb4405345557dd67f627e813cfc2b38c3715da55cb',
                    'foldB_s0': '28d8e7e142ca0efa0c340014a158dae2d65b7c4dae9423f7af8f82271378ee46', 'foldB_s1': '181e8473b09a4ca7bdf1157096ea3036ed32184e4d14dd0e515e88da4ed447ea'}
V4_SEGMENTER = 'unet35_v4_ens'
V4_REGISTRY_REL = 'configs/bayesreg/segmenters_hs_v1.json'
V4_REGISTRY_SHA256 = 'b7682421bf6e8d07c606ad55cd47db7936a293c5e79d00839c82e3f8c2edefa1'
V4_PRED = SEG/'unet35_v4_ens'/'pred'
HS_ROOT = RESULTS/'hs_v1'                       # HS: the frozen Hx24-RF05-F1200 on v4's channels, 19/19 hx_certified, .8778 / .9194 (runs/20261106-hs/result.md)
HS_DECISION_SHA256 = '0b85dd06fb2d859174bcefd61c9bd5e6238760b8aa8ea205de31bb62e1c435c7'      # hs_v1/analysis/decision_hs.json
HS_REFERENCE_REL = 'experiments/p2/reference_unet35_v4_ens.json'                             # HS's 19 pairs by their records' sha256 (run_hs.py write-reference; committed)
HS_REFERENCE_SHA256 = 'fd157a2bc48522d8ee18bcfd4e6425b5551914c2c02d74c7ad8ddc4f152531df'

SUBJECTS = tuple(f'OASIS_OAS1_{n:04d}_MR1' for n in range(438, 458))
SHORTS = tuple(f'{n:04d}-{n+1:04d}' for n in range(438, 457))
N_SUBJECTS, N_PAIRS = 20, 19

# ---- the labels ------------------------------------------------------------------------------------------------------------------------------------
LABELS = tuple(range(1, 36))
SMALL_SIX = (4, 18, 19, 23, 34, 35)             # the inferior lateral ventricles, the vessels, the choroid plexus (experiments/unv2/eval_seg.SMALL_SIX, sorted)
BIG_SIX = (1, 2, 5, 20, 21, 24)
OTHER_23 = tuple(k for k in LABELS if k not in SMALL_SIX+BIG_SIX)

# ---- the target rule (written before the pp diagnosis ran; plan.md section 2) -----------------------------------------------------------------------
TARGET_TOP_STRUCTURES = 3                        # at most three structures of the other 23 join the small six
TARGET_SHARE_MIN = .5                           # a structure joins only if the segmentation share of its error (R3b, mean over its labels) is at least this
# the segmentation share of label l: b^2 / (a^2 + b^2), a = 1 - pp_l (the predicted labels not aligned), b = sqrt(2) (1 - s_l) (the predicted labels wrong; s_l = UF4's segmentation-level Dice, a pair the
# mean of its two subjects); R3b (runs/20261103-n0 (c)), a quadrature model, an order-of-magnitude reference, not an additive split (gap analysis review, important 6)

# ---- the recipe and the members ----------------------------------------------------------------------------------------------------------------------
PROFILE = 'r2_small'                            # bayesreg/seg/unet/small.py; the cli profile
SEED = 2                                        # UF's members have seeds 0 and 1: new initial weights and data order
MEMBERS = (('A', SEED), ('B', SEED))            # one member per fold of folds_full_v2.json (98 train + 8 calib each, UF's folds)
P_TARGET, TARGET_DICE_WEIGHT = .5, 1.            # fixed before any reading, never tuned
EVAL_EVERY = 500
# One arm (the lead's ruling, decision_log 2026-11-10 ~10:42 PM and ~10:59 PM CST: train r2s only; r2sl, the same with twice the steps, is not trained, its question left for later).
RECIPES = {
    'r2s': dict(max_iters=40000, max_wall_seconds=14400.),     # UF's schedule (poly to its end, the last step, no early stop) with the target emphasis
}
RECIPE_ORDER = ('r2s',)
CANDIDATES = {                                  # the candidate segmenter: UF4's four members and r2s's two, an equal-weight mean (all_folds_mean of two fold models of three members)
    'R2S': dict(recipe='r2s', segmenter_id='unet35_r2s_ens', words='R2S：UF4 的四个成员加 r2s 的两个成员（各成员 softmax(z/T) 等权平均）'),
}
CANDIDATE_ORDER = ('R2S',)                      # the one arm of the gate


def member_name(fold, seed=SEED):
    return f'fold{fold}_s{seed}'


MEMBER_NAMES = tuple(member_name(f, s) for f, s in MEMBERS)


def train_config(recipe, targets):
    """The --set values of one recipe: UF's three changes (the poly schedule to its end, no early stop, the last step), the recipe's schedule, the target emphasis."""
    r = RECIPES[recipe]
    return dict(max_iters=r['max_iters'], min_iters=r['max_iters'], patience=r['max_iters'], eval_every=EVAL_EVERY, checkpoint='last', max_wall_seconds=r['max_wall_seconds'],
                target_labels=[int(t) for t in targets], p_target=P_TARGET, target_dice_weight=TARGET_DICE_WEIGHT)


def candidate_plan(candidate):
    """[(fold model suffix, [member names in order])]: fold A = UF foldA_s0, foldA_s1, the recipe's foldA_s2; fold B likewise.  The order is the registry's and the evaluation's (bit for bit the same running sum)."""
    recipe = CANDIDATES[candidate]['recipe']
    return [('foldA', [('uf', 'foldA_s0'), ('uf', 'foldA_s1'), (recipe, member_name('A'))]), ('foldB', [('uf', 'foldB_s0'), ('uf', 'foldB_s1'), (recipe, member_name('B'))])]


# ---- the gates (the brief; the lead's ruling of 8:25 PM CST 11-10: not lowered, the pass probability written) -----------------------------------------
SEG_BAR = .005                                  # primary - UF4, seg35, the mean over the 20 paired subjects
SEG_POSITIVE_MIN = 15                           # at least 15 of 20 subjects positive
SEG4_FLOOR = -.001                              # primary - UF4, seg4, mean
UF_PAIRED_SE = .00076                           # 〔推断〕the standard error of a paired seg35 difference over these 20 subjects: UF4 - v1's t interval (runs/20261110-gap2/gap2_rev1.out section D, as the gap analysis revision 1)
REG_BAR = .003                                  # R2 - HS, seg35, 19 pairs, method level: mean >= +.003
REG_LOWER = -.003                               # and the subject-block 95 % lower bound > -.003
BETTER_LOWER = 0.                               # "better" only with the block lower bound > 0 (about Delta >= +.0045 at HS's lower half width .0045)
SDLOGJ_MAX = .10
BOOT_N, BOOT_SEED = 10000, 20261105             # the block bootstrap of decide_hs (runs/20261105-fd2/recompute.py)
SEG_BOOT_DRAWS, SEG_BOOT_SEED = 20000, 20261110
CONTROL_TOL_FORWARD = 2e-4                      # UF4 recomputed on the GPU against eval_uf.json's rows (uf_common.CONTROL_TOL_FORWARD)
OX_SLOPE = .890                                 # 〔模型〕the segmentation-level to registration slope HS used (runs/20261105-fx OX 读法 2); the gap analysis and its review give .84-.92
FOCUS_SUBJECT = 'OASIS_OAS1_0440_MR1'
C_FLOOR_GIB = 40.                               # the brief: no new run below 40 GiB free on C:


# ---- small helpers ------------------------------------------------------------------------------------------------------------------------------------
def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_once(path, text):
    """Written to <path>.tmp, then moved; FileExistsError if the file is there (never overwritten)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'{path} exists; never overwritten')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)
    return path


def no_nan(node):
    """NaN and infinities as null, recursively (JSON without NaN)."""
    import math
    if isinstance(node, float):
        return node if math.isfinite(node) else None
    if isinstance(node, dict):
        return {k: no_nan(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [no_nan(v) for v in node]
    return node


def write_json_once(path, document):
    return write_text_once(path, json.dumps(no_nan(document), indent=1, ensure_ascii=False, allow_nan=False)+'\n')


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def plan_record(repo):
    """{path, sha256} of plan.md in this code tree (None when it is not there): recorded in members_plan.json, eval_r2.json and decision_r2.json, so that a reading names the preregistration it
    was made under (review m8; described, not a gate)."""
    path = Path(repo)/PLAN_REL
    return dict(path=PLAN_REL, sha256=sha256_file(path) if path.is_file() else None)


def pair_subjects(short):
    f, m = short.split('-')
    return f'OASIS_OAS1_{f}_MR1', f'OASIS_OAS1_{m}_MR1'


def pair_name(short):
    f, m = pair_subjects(short)
    return f'{f}__{m}'
