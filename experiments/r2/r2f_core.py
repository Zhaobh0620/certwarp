"""R2 step 3b, the 19-pair registration reading (runs/20261110-r2/plan.md section 5): the core that stands where HS's modules expect `hs_core`.

run_r2f.py loads experiments/hs/decide_hs.py, hs_seg.py and run_hs.py once more under other names with this module in the place of hs_core (the way run_hs.py itself loads run_fl.py for HS), so that
HS's code paths -- the field runs, the repairs with RW's fuse, the two certificates, the fallbacks of D74/D75, the third certificate, the reading of each pair -- run unchanged on R2's candidate.  hs_core,
hs_seg, decide_hs, run_hs, fx_core, run_fl are not edited.

What differs from hs_core (everything else is hs_core's own object):
  - the root r2_v1 (cells r2_SHx24_<tier>_<pair>, units tag-r2-*); the arm is HS's SHx24, the frozen Hx24 (fx_core.SPEC) on the candidate's channels;
  - the upstream is R2's segmentation-level reading (seg/unet35_r2_full/analysis/decision_r2.json, reading 'pass'), not UF's; its evaluation eval_r2/eval_r2.json holds the candidate's rows;
  - the candidate R2S (unet35_r2s_ens; one arm, the lead's ruling of 11-10 ~10:42 PM CST): UF4's four members and r2s's two, two fold models of three members, no nnU-Net;
  - the registry configs/bayesreg/segmenters_r2_v1.json;
  - the reading (decide in run_r2f.py): Delta = R2 - HS, seg35, mean >= +.003 and the subject-block lower bound > -.003, 19/19 with an output, SDlogJ <= .10; "better" only with the lower bound > 0.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import hs_core as _hs
import r2_common as rc
from hs_core import (ALL_PAIRS, FALLBACK_DIR, HS_ARM, HX_A0, HX_CERTIFIED, HX_IDENTITY, HX_REJECTED, ID_DIR, INFER_RECORD, NATIVE, REFUSAL_FILE, RESULTS, SEG_CODE_DIR, SEGCHECK_RECORD,  # noqa: F401
                     SPEC, STAGED_REGISTRY, UNDECIDABLE, BOOT_N, BOOT_SEED, SDLOGJ_MAX, GOAL, N_SUBJECTS, NN_OUT, NN_WEIGHT, fl, fx, core, member_name)

ROOT_NAME = rc.FIELDS_ROOT_NAME
UF_OUT = rc.OUT                                 # the upstream: R2's chain root (its members, eval_r2/, analysis/); the name is hs_core's
UF_DECISION_REL = 'analysis/decision_r2.json'
UF_EVAL_REL = 'eval_r2/eval_r2.json'
UF_NN_REL = 'nn'                                # not used: no candidate of R2 has nnU-Net
UF_WORTH = 'pass'                               # decide_r2.PASS
UF_MEMBERS = (('A', 0), ('A', 1), ('B', 0), ('B', 1))
CANDIDATES = {k: dict(segmenter_id=v['segmenter_id'], prefix=v['segmenter_id'], nn_model=None, nn_fold=None, uf_arm=k, recipe=v['recipe'], words=v['words']) for k, v in rc.CANDIDATES.items()}
REGISTRY_REL = rc.REGISTRY_REL
SEGCHECK_TOL = rc.CONTROL_TOL_FORWARD
READ_ADOPT, READ_NOT = '采用新分割器', '不采用'
ADOPT_MEAN_MIN = rc.REG_BAR                     # Delta mean >= +.003
ADOPT_LOWER = rc.REG_LOWER                      # the subject-block 95 % lower bound > -.003
BETTER_LOWER = rc.BETTER_LOWER                  # "better": the lower bound > 0
OX_SLOPE = rc.OX_SLOPE


def uf_paths(uf_out=None):
    out = Path(UF_OUT if uf_out is None else uf_out)
    return dict(out=out, decision=out/UF_DECISION_REL, eval=out/UF_EVAL_REL, nn=out/UF_NN_REL)


def uf_reading(uf_out=None):
    """(waiting, stop, candidate, document) from R2's segmentation-level reading: only 'pass' with the candidate R2S lets the registration reading run (any other name, R2SL among them, is refused)."""
    paths = uf_paths(uf_out)
    if not paths['decision'].is_file():
        return ['等 R2 的分割级读数：没有 analysis/decision_r2.json'], None, None, None
    document = json.loads(paths['decision'].read_text(encoding='utf-8'))
    reading, candidate = document.get('reading'), document.get('candidate')
    if reading != UF_WORTH:
        return [], f"R2 的分割级读数是 {reading}（{document.get('reading_text')}）：配准读数不跑，交 BZ", None, document
    if candidate not in CANDIDATES:
        return [], f'R2 的分割级读数过门，但候选 {candidate!r} 不是 {sorted(CANDIDATES)} 之一', None, document
    if not paths['eval'].is_file():
        return [], f'R2 有读数却没有 {UF_EVAL_REL}', None, document
    return [], None, candidate, document


def seg_dir(candidate, results=None):
    return Path(RESULTS if results is None else results)/'seg'/CANDIDATES[candidate]['segmenter_id']


def model_ids(candidate):
    c = CANDIDATES[candidate]
    return dict(A=f"{c['prefix']}_foldA", B=f"{c['prefix']}_foldB", nn=None)


@dataclass(frozen=True)
class R2Cell(fx._Cell):
    PREFIX = 'r2'
    ARMS = {HS_ARM: SPEC}


class R2Batch(_hs.HSBatch):
    PREFIX, ROOT_NAME, ARMS, Cell = 'r2', ROOT_NAME, {HS_ARM: SPEC}, R2Cell
    ROLE = ('R2: Hx24@600 (frozen, Hx24-RF05-F1200) on the channels of R2\'s candidate segmenter, repair with the fuse of RW: development round 2\'s first registration reading (of 2), development pairs, '
            'NOT oracle, NOT F_B, NOT in any result table')

    def __init__(self):
        super().__init__()
        self.UF_OUT = UF_OUT

    def registry_path(self, repo=None):
        return Path(core.ROOT if repo is None else repo)/REGISTRY_REL

    def candidate(self, uf_out=None):
        waiting, stop, candidate, _ = uf_reading(self.UF_OUT if uf_out is None else uf_out)
        if candidate is None:
            raise SystemExit('；'.join(waiting+([stop] if stop else [])))
        return candidate

    def segmenter_id(self, uf_out=None):
        return CANDIDATES[self.candidate(uf_out)]['segmenter_id']

    def seg_dir(self, candidate=None):
        return seg_dir(self.candidate() if candidate is None else candidate, self.RESULTS)


HS = R2Batch()                                  # the name HS's modules import
R2 = HS


def load_copy(name, path, substitutes):
    """The module at `path` executed once more under `name`, with sys.modules[k] = v for each substitute while it imports (so that its `import hs_core` binds this module, and so on); sys.modules is
    restored afterwards (HS's own modules, when imported in the same process, keep theirs).  The copy is registered under `name`."""
    import importlib.util
    import sys
    saved = {k: sys.modules.get(k) for k in substitutes}
    try:
        sys.modules.update(substitutes)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return module


def replace_names(module, **names):
    """setattr for each name, after checking that `module` has it (review m1): if HS renames or drops a function, R2 stops here instead of leaving a new attribute nobody calls."""
    missing = sorted(n for n in names if not hasattr(module, n))
    if missing:
        raise AttributeError(f"{getattr(module, '__name__', module)} has no {missing}: a module of HS changed under R2; R2 does not run until it is looked at")
    for name, value in names.items():
        setattr(module, name, value)
    return module
