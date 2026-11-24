"""HS (runs/20261106-hs/任务书.md, plan.md): the candidate segmenter of UF registered, and the frozen Hx24 configuration (Hx24-RF05-F1200, C22.0-4) run on its channels, 19 development pairs.  The first of the three selection readings (D71, D72).
Not F_B: no certified descent, only the final field passes the certificate; development pairs only, in no result table.

The constants, the candidates, the cells and the guards.  The batch below stands where run_fl.py and decide_fl.py expect the module fl_core (fx_core.Batch, as NX's); fx_core, fd2_core, fl_core are not edited.

The candidate is what UF's reading names (seg/unet35_v4_full/analysis/decision_uf.json, decide_uf.py: `candidate` when the reading is worth_hx24): (a) UF4, the four UF members, or (b) UF4_nn, 1/2 of UF4's mean probability and 1/2 of nnU-Net's.  The file is written once by UF and only read here.
    UF4     segmenter_id unet35_v4_ens, fold models unet35_v4_ens_foldA (fold 0: foldA_s0, foldA_s1) and _foldB (fold 1: foldB_s0, foldB_s1)
    UF4_nn  segmenter_id uf_nn_v1, fold models uf_nn_v1_foldA, uf_nn_v1_foldB (the same members) and uf_nn_v1_nnall (fold 2: nnU-Net's checkpoint_final.pth; the schema's fold is an integer, the id says fold all)
Both in one new registry file, configs/bayesreg/segmenters_hs_v1.json (only the candidate's entries; append-only), its predictions in /home/rmcfz/seq_vp_tag_results/seg/<segmenter_id>/pred.
Cells: Cell('SHx24', 'first5' or 'rest600', pair) the T88 field (fx_core.SPEC, unchanged), Cell('RSHx24', 'c600' or 'rest600', pair) its repair with RW's fuse; all 19 pairs in the plan.  No oracle arm.
No function of this module reads a label except read_truth (fl_core's), for the evaluation.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import fl_core as fl
import fx_core as fx
import mc0_core as core

RESULTS = fx.RESULTS
NATIVE = fx.NATIVE
ALL_PAIRS = fx.ALL_PAIRS
SPEC = fx.SPEC                                  # the frozen Hx24: K = 35 argmax one-hot, sigma_field .5, 600 native steps, Q3h, D factor 24 (C22.7)
HS_ARM = 'SHx24'                                # S: the segmenter swapped, nothing else
ROOT_NAME = 'hs_v1'
SEG_CODE_DIR = 'seg'                            # the seg phase's checkout: $ROOT/seg/code; the fields phase's: $ROOT/code (the registration is committed between them, as NX's steps 1 and 3)

# UF: written once by UF's chain, only read here (never before UF's reading: the dry run only asks whether the file is there)
UF_OUT = RESULTS/'seg'/'unet35_v4_full'
UF_DECISION_REL = 'analysis/decision_uf.json'
UF_EVAL_REL = 'eval_uf/eval_uf.json'
UF_NN_REL = 'nn'                                # probs/<subject>/probs36.npz and convert_record.json: the nnU-Net probabilities UF's (b) read
UF_WORTH = 'worth_hx24'                         # decide_uf.WORTH
UF_MEMBERS = (('A', 0), ('A', 1), ('B', 0), ('B', 1))      # foldA_s0, foldA_s1, foldB_s0, foldB_s1: eval_uf.ENSEMBLE_ORDER, v2's registry order
NN_OUT = RESULTS/'nnunet_v1'                    # NN's weights and records, only read
NN_WEIGHT = .5                                  # uf_common.NN_WEIGHT: fixed before any reading, never tuned

CANDIDATES = dict(
    UF4=dict(segmenter_id='unet35_v4_ens', prefix='unet35_v4_ens', nn_model=None, nn_fold=None, uf_arm='UF4', words='(a) UF4：UF 四个成员的集成（各成员 softmax(z/T) 的平均）'),
    UF4_nn=dict(segmenter_id='uf_nn_v1', prefix='uf_nn_v1', nn_model='uf_nn_v1_nnall', nn_fold=2, uf_arm='UF4_nn', words='(b) UF4 + nnU-Net 各 ½（UF4 的 float16 概率读回 + nnU-Net 读回的概率）'))
REGISTRY_REL = 'configs/bayesreg/segmenters_hs_v1.json'
STAGED_REGISTRY = 'registry_staged.json'        # under the candidate's seg directory: the bytes the registration will commit, written before the inference
INFER_RECORD, SEGCHECK_RECORD = 'infer_record.json', 'segcheck.json'
SEGCHECK_TOL = 2e-4                             # uf_common.CONTROL_TOL_FORWARD: the U1_control of UF, "a GPU forward pass, AMP and cuDNN may move a few voxels"; the mean seg35 and seg4 of the 20 subjects

# The reading (runs/20261106-hs/任务书.md, plan.md)
READ_ADOPT, READ_NOT, UNDECIDABLE = '采用新分割器', '不采用', '不可判'
ADOPT_MEAN_MIN = 0.                             # Delta mean >= 0
ADOPT_LOWER = -.003                             # the subject-block 95 % lower bound > -.003
SDLOGJ_MAX = .10                                # the method's mean official SDlogJ <= .10
BOOT_N, BOOT_SEED = 10000, 20261105             # runs/20261105-fd2/recompute.py block, experiments/p1/p1_core.block_interval
N_SUBJECTS = 20
OX_SLOPE = .890                                 # 〔模型〕runs/20261105-fx/result.md OX 读法 2: (.9362 - .8625) / (1 - .9172); segmentation level to the final seg35 of this line
GOAL = .88

# The line status of a pair (C22.5-4)
HX_CERTIFIED, HX_A0, HX_IDENTITY, HX_REJECTED = 'hx_certified', 'hx_fallback_configC', 'hx_fallback_identity', 'hx_provenance_rejected'
FALLBACK_DIR, ID_DIR = 'fallback', 'id'
REFUSAL_FILE = 'entry_refusal.json'             # written beside run.json by the field run when the channels were refused (run_hs.HSFast): which kind


def uf_paths(uf_out=None):
    out = Path(UF_OUT if uf_out is None else uf_out)
    return dict(out=out, decision=out/UF_DECISION_REL, eval=out/UF_EVAL_REL, nn=out/UF_NN_REL)


def uf_reading(uf_out=None):
    """(waiting, stop, candidate key, the document): what HS waits for (not a problem), why it does not run, the candidate.  Reads decision_uf.json only when it is there (after UF's reading)."""
    paths = uf_paths(uf_out)
    if not paths['decision'].is_file():
        return ['等 UF：没有 analysis/decision_uf.json（UF 还没有读数）'], None, None, None
    document = json.loads(paths['decision'].read_text(encoding='utf-8'))
    reading, candidate = document.get('reading'), document.get('candidate')
    if reading != UF_WORTH:
        if reading == 'not_connected':
            return [], f"UF 的读数是 {reading}（{document.get('reading_text')}）：不接进 Hx24，HS 不跑，本包作废", None, document
        return [], f"UF 的读数是 {reading}（{document.get('reading_text')}）：HS 不跑，交总负责人", None, document     # 不可判 and anything else (revision 1, review minor 14)
    if candidate not in CANDIDATES:
        return [], f'UF 的读数是值得接进 Hx24，但候选 {candidate!r} 不是 {sorted(CANDIDATES)} 之一', None, document
    if not paths['eval'].is_file():
        return [], f"UF 有读数却没有 {UF_EVAL_REL}", None, document
    return [], None, candidate, document


def seg_dir(candidate, results=None):
    return Path(RESULTS if results is None else results)/'seg'/CANDIDATES[candidate]['segmenter_id']


def model_ids(candidate):
    """{'A': id of fold 0, 'B': id of fold 1, 'nn': id or None}."""
    c = CANDIDATES[candidate]
    return dict(A=f"{c['prefix']}_foldA", B=f"{c['prefix']}_foldB", nn=c['nn_model'])


def member_name(fold, seed):
    return f'fold{fold}_s{seed}'


@dataclass(frozen=True)
class HSCell(fx._Cell):
    PREFIX = 'hs'
    ARMS = {HS_ARM: SPEC}


class HSBatch(fx.Batch):
    PREFIX, ROOT_NAME, ARMS, Cell, FIRST_PHASE = 'hs', ROOT_NAME, {HS_ARM: SPEC}, HSCell, (HS_ARM,)
    ROLE = 'HS: Hx24@600 (frozen, Hx24-RF05-F1200) on the channels of the UF candidate segmenter, repair with the fuse of RW: the first selection reading (D71, D72), development pairs, NOT oracle, NOT F_B, NOT in any result table'
    REPAIR_ARM = 'R'+HS_ARM
    CELLS_TEXT = f'{HS_ARM} tier first5 and {"R"+HS_ARM} tier c600 on the first 5 development pairs; both tier rest600 on the other 14'

    def __init__(self):
        super().__init__()
        self.UF_OUT = UF_OUT

    def require_cell(self, arm, tier, pair):
        """Checked before any file of the pair is opened."""
        first, rest = pair in core.PAIRS, pair in fl.REST_PAIRS
        if arm == HS_ARM:
            ok = (tier == self.FIRST and first) or (tier == f'rest{NATIVE}' and rest)
        elif arm == self.REPAIR_ARM:
            ok = (tier == f'c{NATIVE}' and first) or (tier == f'rest{NATIVE}' and rest)
        else:
            ok = False
        if not ok:
            self.refuse(arm, tier, pair)
        return self.Cell(arm, tier, pair)

    def field_cell(self, pair):
        return self.Cell(HS_ARM, self.FIRST if pair in core.PAIRS else f'rest{NATIVE}', pair)

    def repair_cell(self, pair):
        return self.Cell(self.REPAIR_ARM, f'c{NATIVE}' if pair in core.PAIRS else f'rest{NATIVE}', pair)

    def plan_fields(self):
        """decide_fl.collect asks: the fields of the first 5 pairs (the other 14 are rest_cells)."""
        return [self.field_cell(p) for p in core.PAIRS]

    def plan_all_fields(self):
        return [self.field_cell(p) for p in ALL_PAIRS]

    def plan_all_repairs(self):
        return [self.repair_cell(p) for p in ALL_PAIRS]

    def chosen_repairs(self, root):
        """decide_fl.collect asks: the repairs of the first 5 pairs are always in the plan."""
        return [dict(arm=HS_ARM, step=NATIVE)]

    def rest_cells(self, root):
        return [self.field_cell(p) for p in fl.REST_PAIRS], [self.repair_cell(p) for p in fl.REST_PAIRS]

    def all_cells(self, root):
        return self.plan_all_fields()+self.plan_all_repairs()

    def planned_pairs(self, root=None):
        return ALL_PAIRS

    def registry_path(self, repo=None):
        return Path(core.ROOT if repo is None else repo)/REGISTRY_REL

    def candidate(self, uf_out=None):
        """The candidate key, or SystemExit: every unit asks UF's reading again (the file is written once)."""
        waiting, stop, candidate, _ = uf_reading(self.UF_OUT if uf_out is None else uf_out)
        if candidate is None:
            raise SystemExit('；'.join(waiting+([stop] if stop else [])))
        return candidate

    def segmenter_id(self, uf_out=None):
        return CANDIDATES[self.candidate(uf_out)]['segmenter_id']

    def seg_dir(self, candidate=None):
        return seg_dir(self.candidate() if candidate is None else candidate, self.RESULTS)

    def pred_root(self, candidate=None):
        return self.seg_dir(candidate)/'pred'

    def fallback_root(self, root=None):
        return Path(self.DEFAULT_ROOT if root is None else root)/FALLBACK_DIR

    def id_root(self, root=None):
        return Path(self.DEFAULT_ROOT if root is None else root)/ID_DIR


HS = HSBatch()
