"""R3b's constants and small helpers (runs/20261111-r3/calibration_plan.md; task brief runs/20261111-r3/任务书.md, R3b).

The calibration diagnosis of the v4 ensemble and the "calibration passes" criterion of the small-structure decoding
(bayesreg.seg.decode): the 16 calibration subjects of the training set, each with the two members of its own fold (out of the
members' training data), argmax against the decoding, read once.  Everything a later step or R3a needs to pin is here.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from bayesreg.seg import decode as decode_module

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path("/home/rmcfz/seq_vp_tag_results")
OUT = RESULTS/"r3b_calib_v1"                                   # plan section 4
REHEARSAL_ROOT = RESULTS/"r3b_calib_rehearsal"                 # plan revision 2 item 4: the rehearsal of the real inputs under the spy (never OUT)
READ_LOCK = RESULTS/"r3b_calib_truth_read.lock"                # one fixed path, not derived from any option (plan section 4)
UF_MODELS = RESULTS/"seg"/"unet35_v4_full"                     # the four members of v4 (UF), only read
DATA_ROOT = Path("/mnt/d/data")
REGISTRY_REL = "configs/bayesreg/segmenters_hs_v1.json"         # contract C22.1-2 revision 4
REGISTRY_SHA256 = "b7682421bf6e8d07c606ad55cd47db7936a293c5e79d00839c82e3f8c2edefa1"
FOLDS_REL = "splits/bayesreg_v1/folds_full_v2.json"            # the folds of UF (98 train + 8 calib per fold); pinned in bayesreg.seg.unet.pins
ALLOWLIST_REL = "configs/bayesreg/dev_subjects.json"
SEG4_MAP_REL = "configs/bayesreg/seg4_of_seg35_v1.json"
PLAN_REL = "runs/20261111-r3/calibration_plan.md"
PLAN_SHA256 = "801c39d28f9a5f87b2038bedf33d297e18d58acc09684d9d91644cf2b8e0ac64"   # calibration_plan.md with revisions 1 and 2 (revision 2: R3b revision round, review r3b-review 1be9f07); a new plan revision changes this pin in the same commit
SEGMENTER_ID = "unet35_v4_ens"
FOLDS = ("A", "B")
SEEDS = (0, 1)
GPU_CAP_GIB = 20.
C_FLOOR_GIB = 40.                                              # plan revision 2 item 3: the infer gate's C: floor (R2's)

# plan section 2: copied from runs/20261105-uf/plan.md "成员" (ids only); a real run checks them against the pinned folds file
CALIB_SUBJECTS = {
    "A": tuple(f"OASIS_OAS1_{n:04d}_MR1" for n in (63, 64, 81, 107, 157, 190, 384, 434)),
    "B": tuple(f"OASIS_OAS1_{n:04d}_MR1" for n in (70, 88, 199, 226, 301, 328, 416, 426)),
}
MEMBER_SHA256 = {                                               # the registry's members per fold (plan section 2)
    "A": ("52522d1f9cb6ad46f295542b94746d48f4f504a2db07b2dad1d7e4154d2852fd", "15dea2900bc9acc453a423bb4405345557dd67f627e813cfc2b38c3715da55cb"),
    "B": ("28d8e7e142ca0efa0c340014a158dae2d65b7c4dae9423f7af8f82271378ee46", "181e8473b09a4ca7bdf1157096ea3036ed32184e4d14dd0e515e88da4ed447ea"),
}

# plan section 3: the one setting the criterion judges; "ratio" is described, never selectable
DECODE_LABELS = decode_module.SMALL_SIX
DECODE_SPEC = decode_module.decoder_spec(DECODE_LABELS, objective="expected_dice", candidate_floor=1e-3)
DESCRIPTION_SPEC = decode_module.decoder_spec(DECODE_LABELS, objective="ratio", candidate_floor=1e-3)

# plan revision 2 item 8: the volumes of every target label of R2 (decision_log 11-11 ~2:45 AM CST); beyond the small six, description only
TARGET_LABELS = (4, 15, 16, 17, 18, 19, 23, 31, 32, 33, 34, 35)

# plan section 6: inclusive, no rounding
CRITERION = dict(n_subjects=16, small6_mean_min=.005, small6_positive_min=10, seg35_mean_min=0., seg4_mean_min=-.0005)
BOOTSTRAP = dict(n=20000, seed=20261111)
ECE_BINS = 15
ECE_FLOORS = (1e-3, .01)
NEIGHBOURS = (3, 22, 14, 30, 17, 33, 1, 20, 2, 21)

PROBS_DIR, DECODED_DIR, EVAL_DIR, ANALYSIS_DIR = "probs", "decoded", "eval", "analysis"
INFER_RECORD, DECODE_RECORD = "infer_record.json", "decode_record.json"
OPENED_INFER, OPENED_DECODE, OPENED_EVAL = "opened_files_infer.json", "opened_files_decode.json", "opened_files_eval.json"
EVAL_FILE, DECISION_FILE, SUMMARY_FILE = "eval_calib.json", "decision_calib.json", "summary_calib.md"
DECISION_FORMAT = "bayesreg.r3b.calibration_decision.v1"
EVAL_FORMAT = "bayesreg.r3b.eval_calib.v1"
VERDICTS = ("pass", "not_pass", "undecidable")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def no_nan(node):
    if isinstance(node, dict):
        return {str(k): no_nan(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [no_nan(v) for v in node]
    if isinstance(node, (np.floating, float)):
        return None if not np.isfinite(node) else float(node)
    if isinstance(node, np.integer):
        return int(node)
    return node


def write_new(path, text):
    """Write a new file atomically (tmp, then replace); refuse to overwrite."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path)+".tmp")
    with open(tmp, "w", encoding="utf-8") as stream:
        stream.write(text)
    os.replace(tmp, path)
    return path


def write_json_new(path, document):
    return write_new(path, json.dumps(no_nan(document), indent=1, ensure_ascii=False)+"\n")


def save_npz_new(path, **arrays):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path)+".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return path


def plan_check(repo=ROOT):
    """(path, sha256) of the plan; ValueError when it is not the pinned one (a real run with PLAN_SHA256 None is refused).  The pin is read at call time."""
    path = Path(repo)/PLAN_REL
    sha = sha256_file(path)
    pinned = PLAN_SHA256
    if pinned is None:
        raise ValueError(f"{PLAN_REL}: no pinned sha256 in r3b_common.PLAN_SHA256 yet (now {sha}); a real run is refused")
    if sha != pinned:
        raise ValueError(f"{PLAN_REL}: sha256 {sha} is not the pinned {pinned}")
    return path, sha


def decode_py_path():
    """The file of the bayesreg.seg.decode module this process imported (plan revision 2 item 1)."""
    return Path(decode_module.__file__).resolve()


def implementation_record():
    """The sha256 of the decoder's implementation and of this file (plan revision 2 item 1; review I1): written by decode, checked by eval, copied into the
    evaluation and the decision, checked again by r3b_decision.load_calibration_decision."""
    return dict(decode_py_sha256=sha256_file(decode_py_path()), decode_py_path=str(decode_py_path()),
                r3b_common_py_sha256=sha256_file(Path(__file__).resolve()))


def all_subjects():
    return [s for f in FOLDS for s in CALIB_SUBJECTS[f]]
