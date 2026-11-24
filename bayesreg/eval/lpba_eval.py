"""bayesreg.eval.lpba_eval -- the one evaluation entry of the LPBA40 second test set (contract C26.7, revision 11/11a/11b/11c; prereg
docs/bayesreg/notes/LPBA40第二测试集预登记_20261119.md (sha256 06121278...) §2.2 last part, §5, §8.4, §12 item 12; D147).

Work package LP-C2 (runs/20261119-lpba/任务书_C2_评测判定可行性与代码身份.md; runs/20261119-lpba/实现说明_C2.md).  It is the only process that reads an LPBA40 S03-S36 label
(census class T, D147): the label NIfTI <RAW_ROOT>/<s>/delineation/label.nii.gz, through read_label only (and its helper _label_volume).
Its relation to bayesreg.eval.truth (contract C26.3-6, revision 11c-1): truth refuses these 34 subjects on every allowlist -- check_name and recheck_guard by the pair
directory's name (as given and resolved), admit_pair and recheck_guard again by the manifest's subjects after the whitelisted decoding (refuse_lpba40_subjects) -- so
load_truth and bayesreg/eval/cli never read them, and this entry calls none of truth's functions: it reads the labels its own way (C26.7-3, prereg §2.2 last part).
The stage's pair.npz is read through bayesreg.io.pair.load_pair with the pinned LPBA allowlist (C26.3's one exception), fixed / moving / initial_u only.
bayesreg.eval.holdout (C23.12) is not changed and never reads LPBA40.

The structure is bayesreg/eval/holdout.py's (C26.7-1): its helpers are imported, never changed (take_lock, write_once, read_committed, machine_gate, OpenedFiles, the arm table's
fill / record_dir / arm_roots / dotted / template_root, verify_files, certify_field_bl1, decision_certificate, cell_authorization's rules), and the stages are:
  1. admission   open_lpba(out_dir, pair_set): the spy (C23.6-5's rule: the guard's cache of the sealed lists empty; RAW_ROOT watched from the start); the code identity (formal:
                 rule one with the LPBA40 constants, C26.9-4, bayesreg.io.code_identity.LPBA40; feasibility: a clean tree, recorded, C26.9-1); the pins and the admission of
                 experiments/lpba/lpba_common.admit (the preregistration PREREG_SHA256, the lists, the frozen files C26.6-3 -- the method's settings_sha256, the external arms'
                 frozen file, configC_reference.json, the LPBA40 arm table --, the c record, the pinned LPBA allowlist, the mode's names (formal: the 34 of test_subjects.txt and
                 the 17 rows of test_pairs.tsv in their direction; S01, S02 and S37-S40 refused), the sealed lists (H1-H4 and the B-line hold-out never), check_subject on the
                 LPBA allowlist per subject, admitted_by_lpba40 recorded: L34); the preregistration and the arm table committed and unchanged (the table validated by
                 experiments/lpba/lpba_arm_table.load_bytes); every (arm, pair) cell written (classify_cell, D107's classes as the hold-out's); the machine gate (no other tag-* /
                 seq-vp-* unit); (formal) the step-10 sha256 manifest committed, unchanged, exactly the files read (check_manifest); each pair's stage manifest in the row's
                 direction; the out-dir rules; (formal) the master lock (O_EXCL; its line: preregistration, arm table, manifest and its commit, P_L, utc, pid) after every
                 check, or a reevaluation authorization (load_reeval, D109).
  2. truth       read_label(subject, admission=..., ...) (prereg §2.2 last part, C26.7-3): the admission object issued here, its lock and the pair's lock taken; the label's
                 sha256 into the record; in a private temporary directory (tempfile.mkdtemp, never under the results root or the code tree) a copy with only the xyzt units set
                 to mm (data, affine, qform and sform checked equal to the original's), load_scalar(labels=True), np.flip(., 2) -- the stage's own transform (prep_lpba P1-P3);
                 the copy and its directory removed at once (nothing of the truth is left under the results root).  Checks: the label's affine and shape equal the same
                 subject's image's -- the original image.nii.gz whose path and sha256 the stage manifest names (source_files.<side>.original), its sha256 checked, its header
                 read (nibabel; an input of the manifest, role image_header); the grid after the transform the stage manifest's; values finite, non-negative, integers.  Not
                 so: TruthMismatch, the subject's evaluation stops before any Dice (to the lead; exit 4).  The 56 ids are code constants (LPBA40_IDS; never read from voxels): a
                 missing id is recorded and dice_labels' rule applies; values outside 0 and the 56 ids are recorded (count and values) and are in no label set.  The left-right
                 record (only recorded): for the 27 pairs (odd id left, even right) the sign of the difference of the centroids on the array's axis 2 (left - right; the
                 OASIS convention: left at the lower index, sign -1).
  3. score       score_cell: fields.describe_run / load_field (as holdout.score_cell), metrics.evaluate_field with metrics.lpba40_scheme(LPBA40_IDS) (lpba56; lpba54 without
                 181, 182), the brain mask = the stage's fixed image > 0 (no label), the Jacobian-only certificate of BL1 (holdout.certify_field_bl1, tile 8192: faces and
                 sampling domain recorded beside it, not folded into its status), and the 24-tetrahedron proxy: bayesreg.eval.holdout_tet24.folding_module (scripts/
                 folding_metrics.py of this tree, default schemes (24, 6, 5)), u = fields.d_to_u(d, spacing), sha256(u) equal to the same cell's BL1 state_hash (else
                 FieldMismatch, the evaluation stops), tet_ratio_summary(u, spacing) as it is (C24.12-4's order; holdout_tet24.py is not changed).  The method arm also the
                 decision configuration's certificate of output/final.npz (holdout.decision_certificate; recorded only).  The identity row (d = 0) through evaluate_field,
                 in no family.
The no-truth mode (the feasibility pairs, pair_set 'lpba40_feas'; prereg §7.1, C26.7-6): no label file is opened (the spy refuses one; no read_label call); only the field's
quantities: official SDlogJ, the central-difference det counts (all and brain), the brain log J statistics (mask = fixed image > 0), BL1 with faces and domain, the 24
tetrahedra.  It has no committed manifest (it runs before P_L): every file it reads is hashed at the admission (the run's own list, recorded) and again before each read.

Cells (holdout's D107 classes; the arm table configs/frozen/lpba40_test_v1_arm_table.json, c47bbbe2...): the terminal record (the first of terminal.files there), its class;
output -> the complete files, the method's review record finalized, the field (the method: field.dir/disp.npz; the others: file_key / sha256_key of the terminal record,
under field.root -- configuration C's output_store.disp.file, a resumed cell's under <pair>.resume, C26.5-5, T19); the no-output classes written as the hold-out's: a
wrapper exception or infra_failure only with a committed resume that ran (or a no_resume, 未批补跑), hx_implementation_error only with a committed no_resume whose reason states
BZ's (甲), hx_provenance_rejected (P20) only with BZ's no_resume naming P20 (holdout_common, pair set 'lpba40_test'; splits/lpba40_test_v1/, D148); an external or
configuration C arm's needs_lead_rejudgment through its committed rejudgment record (bayesreg.io.rejudgment, runs/lpba40_test_v1/rejudgments/, C26.8-4); a P19 stop
(runs/lpba40_test_v1/arm_stop_<arm>.json, at least P19_STOP_AT = 2 suspicious pairs, C26.8-2); anything else stops before any truth.  Nothing is moved on LPBA40 (prereg §11):
the method cell is crossed with LP-B's finalized record <root>/review/finalized_outputs.json without the move part (method_cross_check; decision_log line 584 ruling 5), and that
record is an input of the list (role cell).

Roots: the code constant RESULTS (read at call time; the tests point it into tmp_path): formal lpba40_test_v1/, feasibility lpba40_feas_v1/ (C26.2-3).  The committed arm table
names the formal root; relocate() maps it onto the mode's root (feasibility: lpba40_feas_v1/; the identity map for the formal mode in production).  Outputs: --out-dir, a new
directory under <root>/eval/ (C26.7-7): <out>/pairs/<pair>.json once per pair, <out>/lpba_eval.json; the decision code bayesreg.eval.lpba_decide reads them through the lock
lines (run_lines).  It never returns to the methods (C26.7-9): no output selects a pair, triggers a fallback or reruns an arm.
"""
import os

if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("NUMBA_NUM_THREADS", "2")

from dataclasses import dataclass                                          # noqa: E402
import json                                                                # noqa: E402
from pathlib import Path                                                   # noqa: E402
import re                                                                  # noqa: E402
import shutil                                                              # noqa: E402
import sys                                                                 # noqa: E402
import tempfile                                                            # noqa: E402
from types import MappingProxyType                                         # noqa: E402

import numpy as np                                                         # noqa: E402

from jd_curl.three_d.nifti import file_sha256, load_scalar                 # noqa: E402

from bayesreg.io import rejudgment as rejudgment_io                        # noqa: E402
from bayesreg.io.guard import SubjectSealed, is_hex64                      # noqa: E402
from bayesreg.io.pair import _read_manifest                                # noqa: E402
from bayesreg.io.sealed import SealedListError, sealed_cache_is_empty      # noqa: E402
from bayesreg.io import sealed as sealed_module                            # noqa: E402

from . import holdout as H                                                 # noqa: E402
from . import metrics                                                      # noqa: E402
from .fields import array_sha256, d_to_u, describe_run, load_field         # noqa: E402
from .holdout import HoldoutRefused, ManifestChanged, SpyRefused, TruthMismatch      # noqa: E402

SCHEMA_PAIR = "bayesreg.lpba40.eval.pair.v1"
SCHEMA_RUN = "bayesreg.lpba40.eval.v1"
SCHEMA_MANIFEST = "bayesreg.lpba40.eval_manifest.v1"
SCHEMA_REEVAL = "bayesreg.lpba40.reeval.v1"

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path("/home/rmcfz/seq_vp_tag_results")                     # read at call time (roots); the tests point it into tmp_path
REAL_RESULTS = Path("/home/rmcfz/seq_vp_tag_results")                # never moved: a label copy is never written under it (C26.7-3)
REAL_FORMAL_ROOT = "/home/rmcfz/seq_vp_tag_results/lpba40_test_v1"   # the root the committed arm table names (experiments/lpba/lpba_arm_table.LPBA_ROOT; a test compares)
PAIR_SETS = ("lpba40_test", "lpba40_feas")
ROOT_NAMES = MappingProxyType(dict(lpba40_test="lpba40_test_v1", lpba40_feas="lpba40_feas_v1"))
RAW_ROOT = Path("/mnt/d/data/LPBA40/nii/subjects")                     # prereg §1.1; read only: image.nii.gz headers and (formal) label.nii.gz
# LP-C2 review I3: the spy watches, from its start, the whole data root (the B-line hold-out's originals among it), the code trees' data/three_d (stage_LPBA40_v1's S03-S24 manifests and
# _nifti_mm label copies: C26.5-1 禁读) and the results root's holdout_B_v1; under them only the run's list (and, formal, the pair's labels while read_label reads them) may be opened.
DATA_ROOT = Path("/mnt/d/data")
THREE_D_ROOTS = (REPO/"data"/"three_d", Path("/home/rmcfz/code/python_projects/seq_vp_tag/data/three_d"))     # this tree's data link and the main checkout's
HOLDOUT_RESULTS_NAME = "holdout_B_v1"                                 # <RESULTS>/holdout_B_v1 (read at call time)
IMAGE_REL = "delineation/image.nii.gz"
LABEL_REL = "delineation/label.nii.gz"
MASTER_LOCK_NAME = "lpba_eval.lock"                                  # D109's one master lock per mode, under <root>/eval/ (fixed, never from an option)
PAIR_LOCK_DIR_NAME = "pair_locks"
OWN_UNITS = MappingProxyType(dict(lpba40_test=("tag-lp-eval",), lpba40_feas=("tag-lp-feas-eval",)))       # C26.2-5
UNIT_MEMORY_MAX = "10G"          # the evaluation unit's MemoryMax: the hold-out's tag-hb-eval (decision_log 5edd3857: 10G, swap 4G, 72 h); the feasibility chain's step must equal it
# LP-C2 integration review minor 3: the evaluation unit's whole spec, the hold-out's tag-hb-eval (decision_log 2026-11-17 line 465, kept by the review of line 466) with this entry
# and its unit names; launch_line builds the start string from it (the feasibility chain runs the feasibility mode's with its log: the same spec as the formal one's).
UNIT_SPEC = MappingProxyType(dict(memory_max=UNIT_MEMORY_MAX, memory_swap_max="4G", oom_policy="continue", cpu_quota="800%", runtime_s=259200, nice=5, numba_threads=2))
PY = "/home/rmcfz/miniconda3/envs/sci310/bin/python"

PREREG_REL = "docs/bayesreg/notes/LPBA40第二测试集预登记_20261119.md"
PREREG_SHA256 = "06121278d8b229401afea7cb86c841f80ba02f23bc40f8c5f5594c7da8d95643"
ARM_TABLE_REL = "configs/frozen/lpba40_test_v1_arm_table.json"
ARM_TABLE_SHA256 = "c47bbbe25d748fa14115c251ea1ceb93514eae9136ae8319ba669eb5453a29ad"      # decision_log line 534 (C26.2-4)
TEST_PAIRS_REL = "splits/lpba40_test_v1/test_pairs.tsv"
FEAS_PAIRS_REL = "splits/lpba40_test_v1/feasibility_pairs.tsv"
MANIFEST_REL = "runs/lpba40_test_v1/eval_manifest.txt"                # prereg §8.2 step 10: the input lines; RECORDS_REL the record-only lines (never read here)
RECORDS_REL = "runs/lpba40_test_v1/eval_records.txt"
MANIFEST_HEADER = ("prereg_sha256", "arm_table_sha256")
REEVAL_DIR_REL = "splits/lpba40_test_v1"                              # D148
REEVAL_NAME = re.compile(r"lpba40_test_v1_reeval_([0-9]{1,3})\.json", re.ASCII)
REEVAL_LOCK = re.compile(r"reeval_([0-9]{1,3})\.lock", re.ASCII)
REEVAL_KEYS = ("schema", "written_by", "words", "time_local", "utc", "reason", "reason_class", "master_lock_sha256", "manifest_sha256", "pairs", "previous_out")
REEVAL_WRITERS = ("BZ", "lead on BZ's behalf")
BZ_WORDS = "预登记写好、复核过后批准你直接跑, 不用等我"                    # decision_log line 520 (C26.8-3, -4; = holdout_common.LPBA40_BZ_WORDS, a test compares)
AUTH_PAIR_SET = "lpba40_test"                                         # holdout_common's pair set of the LPBA40 authorizations (the formal mode only, C26.6-2)
P19_STOP_AT = 2                                                       # C26.8-2, prereg §6.6 (裁决 (8)): an arm stops at its second suspicious pair

# the 56 LPBA40 label ids (prereg §5.1; C26.2-4): code constants, never read from voxels
LPBA40_IDS = tuple(range(21, 35))+tuple(range(41, 51))+tuple(range(61, 69))+tuple(range(81, 93))+(101, 102, 121, 122)+tuple(range(161, 167))+(181, 182)
LPBA54_IDS = tuple(i for i in LPBA40_IDS if i not in metrics.LPBA40_EXCLUDED_54)
LR_PAIRS = tuple((i, i+1) for i in LPBA54_IDS if i % 2 == 1)         # 27 pairs: an odd id is a left structure, the next even one its right (prereg notes, label table)
LR_EXPECTED_SIGN = -1                                                 # left at the lower index of axis 2 (the OASIS convention; prereg §2.2)
SCHEME = metrics.lpba40_scheme(LPBA40_IDS)


# ---- the mode's roots and the arm table --------------------------------------------------------------------------------------------------------------
def check_pair_set(pair_set):
    if pair_set not in PAIR_SETS:
        raise HoldoutRefused("pair_set", f"{pair_set!r}: one of {PAIR_SETS} (C26.6-2)")
    return pair_set


def root(pair_set):
    return Path(RESULTS)/ROOT_NAMES[check_pair_set(pair_set)]


def eval_dir(pair_set):
    return root(pair_set)/"eval"


def master_lock(pair_set):
    return eval_dir(pair_set)/MASTER_LOCK_NAME


def pair_lock_dir(pair_set):
    return eval_dir(pair_set)/PAIR_LOCK_DIR_NAME


def relocate(value, pair_set):
    """Every string of the committed arm table that names the formal root REAL_FORMAL_ROOT, mapped onto the mode's root (C26.2-3: one group of roots per pair set)."""
    if isinstance(value, str):
        head = REAL_FORMAL_ROOT+"/"
        if value == REAL_FORMAL_ROOT:
            return str(root(pair_set))
        return str(root(pair_set))+"/"+value[len(head):] if value.startswith(head) else value
    if isinstance(value, list):
        return [relocate(v, pair_set) for v in value]
    if isinstance(value, dict):
        return {k: relocate(v, pair_set) for k, v in value.items()}
    return value


def _lpba_module(name):
    """experiments/lpba/<name>.py of this code tree, imported by its flat name, its origin checked (holdout._experiment_module)."""
    return H._experiment_module("lpba", name)


def load_arm_table(pair_set, repo=None):
    """The pinned LPBA40 arm table (ARM_TABLE_SHA256), committed and unchanged, validated by experiments/lpba/lpba_arm_table.load_bytes (the form, then the content: build_table of
    the pinned hold-out table), relocated onto the mode's root.  (table, record)."""
    rootdir = Path(REPO if repo is None else repo)
    raw, digest, git = H.read_committed(ARM_TABLE_REL, "arms", rootdir)
    if digest != ARM_TABLE_SHA256:
        raise HoldoutRefused("arms", f"{ARM_TABLE_REL}: sha256 {digest} is not the pinned {ARM_TABLE_SHA256}")
    try:
        table = _lpba_module("lpba_arm_table").load_bytes(raw, rootdir)
    except ValueError as exc:
        raise HoldoutRefused("arms", f"{ARM_TABLE_REL}: {exc}") from None
    return relocate(table, pair_set), dict(path=ARM_TABLE_REL, sha256=digest, git=git, relocated_to=str(root(pair_set)))


def check_prereg(repo=None):
    raw, digest, git = H.read_committed(PREREG_REL, "prereg", Path(REPO if repo is None else repo))
    if digest != PREREG_SHA256:
        raise HoldoutRefused("prereg", f"{PREREG_REL}: sha256 {digest} is not the pinned {PREREG_SHA256}")
    return dict(path=PREREG_REL, sha256=digest, git=git)


def pair_rows(pair_set, lists):
    """((index, fixed, moving, name), ...) of the mode's table in its row order (holdout's row form; index from 0)."""
    pairs = lists.test_pairs if pair_set == "lpba40_test" else lists.feas_pairs
    return tuple((i, f, m, f"LPBA40_{f}__LPBA40_{m}") for i, (f, m) in enumerate(pairs))


# ---- the code identity (C26.9) ---------------------------------------------------------------------------------------------------------------------------
def rule_one(repo=None, run=None):
    """C26.9-4: rule one with the LPBA40 constants (bayesreg.io.code_identity.LPBA40), before anything else is read.  The record; HoldoutRefused('code_identity')."""
    from bayesreg.io import code_identity as ci
    record, problems = ci.unit_identity_problems(repo=Path(REPO if repo is None else repo), run=run, profile=ci.LPBA40)
    if problems:
        raise HoldoutRefused("code_identity", f"rule one for LPBA40 does not hold (C26.9-4): {problems[:5]}; nothing is read")
    return json.loads(json.dumps(record, default=H.json_default))


def check_code_identity(pair_set, repo=None):
    """Formal: rule one (C26.9-4).  Feasibility (C26.9-1: before P_L, from a clean detached checkout): the code tree clean, its commit recorded.  Returns the record."""
    rootdir = Path(REPO if repo is None else repo)
    code = H.code_record(rootdir)
    if pair_set == "lpba40_test":
        return dict(rule_one=rule_one(rootdir), code=code)
    if not code["clean"]:
        raise HoldoutRefused("code", f"the code tree is not clean ({list(H.CLEAN_DIRS)}): {code['status'][:200]!r} (C26.9-1)")
    return dict(rule_one=None, code=code, note="feasibility: not under rule one (C26.9-1); code_commit and clean recorded")


# ---- the spy (C23.6-5's rule; holdout.OpenedFiles) ---------------------------------------------------------------------------------------------------------
_SPY = None
_HOOKED = False


def _audit(event, args):
    spy = _SPY
    if spy is None or event != "open" or not args:
        return
    path = args[0]
    if path is None or isinstance(path, int):
        return
    spy.saw(path, args[1] if len(args) > 1 else None)


def start_spy(dirs):
    global _SPY, _HOOKED
    if not sealed_cache_is_empty():
        raise HoldoutRefused("spy", "the sealed lists are already cached in this process: the spy would not see the guard read them (C23.6-5); start a new process")
    files = [Path(sealed_module.REPO)/spec["path"] for spec in sealed_module.SEALED_LISTS]
    _SPY = H.OpenedFiles(dirs, files)
    if not _HOOKED:
        sys.addaudithook(_audit)
        _HOOKED = True
    return _SPY


_LAST_SPY = None


def stop_spy():
    global _SPY, _LAST_SPY
    spy, _SPY = _SPY, None
    if spy is not None:
        _LAST_SPY = spy
    return spy


# ---- admission objects: issued here only, checked by identity (holdout's way) ------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Admission:
    mode: str                 # the pair set
    subjects: tuple
    pairs: tuple              # ((index, fixed, moving, name), ...)
    record: MappingProxyType


_ISSUED, _STATE = {}, {}


def _issue(mode, subjects, pairs, record):
    admission = Admission(mode, tuple(subjects), tuple(pairs), MappingProxyType(dict(record)))
    _ISSUED[id(admission)] = admission
    _STATE[id(admission)] = dict(locked=False, pair_locks={})
    return admission


def _state(admission, stage="truth"):
    if not (type(admission) is Admission and _ISSUED.get(id(admission)) is admission):
        raise HoldoutRefused(stage, "not an admission issued by bayesreg.eval.lpba_eval in this process (copies, pickles and hand-made objects are refused)")
    return _STATE[id(admission)]


# ---- cells (holdout's D107 classes; LPBA40's records) -----------------------------------------------------------------------------------------------------
def _stop_cell(arm, row, repo, cell):
    """P19 (C26.8-2): the arm's committed stop record (runs/lpba40_test_v1/arm_stop_<arm>.json) lists the pair among its unrun pairs, names at least P19_STOP_AT suspicious
    pairs and XB's own record at its sha256 (holdout.stop_cell's rules with LPBA40's threshold).  (written, problem)."""
    stop_rel = H.fill(arm["stop_record"], arm, row)
    if not (Path(repo)/stop_rel).is_file():
        return False, None
    document, git = H._committed_record(stop_rel, H.SCHEMA_STOP, arm, row, repo, writers=H.STOP_WRITERS)
    unrun = document.get("unrun_pairs")
    if not (isinstance(unrun, list) and row[3] in unrun and isinstance(document.get("suspicious_pairs"), list) and len(document["suspicious_pairs"]) >= P19_STOP_AT):
        return False, f"{stop_rel} does not list {row[3]} among the unrun pairs of a P19 stop ({P19_STOP_AT} suspicious pairs)"
    xb = document.get("arm_stopped")
    expected = H.template_root(arm["record_dir"].replace("{arm}", arm["name"]))/H.XB_STOP_NAME
    if not (isinstance(xb, dict) and set(xb) == {"path", "sha256"} and is_hex64(xb.get("sha256"))):
        return False, f"{stop_rel}: arm_stopped {{path, sha256}} of XB's {H.XB_STOP_NAME} required"
    if os.path.realpath(str(xb["path"])) != os.path.realpath(str(expected)) or not Path(expected).is_file():
        return False, f"{stop_rel}: arm_stopped.path {xb['path']!r} is not XB's record {expected} (or it is not there)"
    original = H._read_record(expected, cell.records)
    if cell.records[str(expected)][0] != xb["sha256"]:
        return False, f"{expected}: sha256 {cell.records[str(expected)][0]} is not the {xb['sha256']} {stop_rel} names"
    differ = [k for k in H.XB_STOP_FIELDS if not isinstance(original, dict) or original.get(k) != document.get(k)]
    if not isinstance(original, dict) or original.get("schema") != H.XB_STOP_SCHEMA or differ:
        return False, f"{expected}: schema {H.XB_STOP_SCHEMA} and the fields {differ} as {stop_rel} copies them required"
    cell.needed.append(expected)
    cell.written, cell.outcome, cell.no_output_class = True, "no_output", "arm_stopped"
    cell.stop_record = dict(git, document=document, arm_stopped=dict(path=str(expected), sha256=xb["sha256"]))
    return True, None


def _authorization(arm, row, kind, terminal_path, terminal_sha, records, needed):
    """The committed holdout_common records of one cell (pair set 'lpba40_test', splits/lpba40_test_v1/; holdout.cell_authorization's rules without the hold-out's finalized
    record): resume -- exactly one, its previous record kept inside authorization.records at its sha256, the terminal record now another file and sha256 (the resume ran);
    no_resume -- its note 未批补跑, its record the terminal record at record_sha256; for hx_implementation_error its reason states BZ's (甲) of C22.5-6, for hx_provenance_rejected
    (p20) it names P20.  (record or None, problem or None); a record of the cell holdout_common refuses stops the evaluation."""
    hc = H._holdout_common()
    spec = arm["authorization"]
    allowed = Path(H.fill(spec["records"], arm, row))
    found = {}
    for kind_ in ("resume", "no_resume"):
        valid, refused = hc.find_authorizations(kind=kind_, tool=spec["tool"], arm=spec["arm"], pair=row[3], pair_set=AUTH_PAIR_SET)
        if refused:
            raise HoldoutRefused("arms", f"{arm['name']} {row[3]}: holdout_common refuses these {kind_} records of the cell: {refused}")
        found[kind_] = valid
    problems, have = [], dict(resume=None, no_resume=None, no_resume_jia=None, no_resume_p20=None)
    if len(found["resume"]) > 1:
        problems.append(f"{len(found['resume'])} resume records: one resume per cell (prereg §8.3)")
    for auth in found["resume"][:1]:
        doc = auth.document
        previous, digest = doc.get("previous_record"), doc.get("previous_record_sha256")
        problem = H._record_place_problem(previous, allowed)
        if problem is None and os.path.realpath(previous) == os.path.realpath(str(terminal_path)):
            problem = f"{previous}: the resume's previous record is the terminal record now (the resume did not run)"
        if problem is None and digest is not None:
            if not Path(previous).is_file() or H._hash_record(previous, records) != digest:
                problem = f"{previous}: missing or not at the resume's previous_record_sha256 {digest}"
            elif digest == terminal_sha:
                problem = f"{terminal_path} has the sha256 of the resume's previous record: the resume did not run"
            else:
                needed.append(Path(previous))
        if problem:
            problems.append(f"resume {auth.path}: {problem}")
        else:
            have["resume"] = dict(kind="resume", path=auth.path, sha256=auth.sha256, previous_record=previous, previous_record_sha256=digest)
    for auth in found["no_resume"]:
        row_, problem = H._no_resume_checked(hc, auth, allowed, records, needed, expected_record=terminal_path)
        if problem:
            problems.append(problem)
            continue
        reason = row_["reason"]
        if kind == "p20":
            why = rejudgment_io.p20_reason_problem(reason)
            if why is None:
                have["no_resume_p20"] = have["no_resume_p20"] or dict(row_, kind="no_resume_p20")
        elif kind == "implementation_error":
            why = rejudgment_io.decision_reason_problem(reason, None, "(甲)")
            if why is None:
                have["no_resume_jia"] = have["no_resume_jia"] or dict(row_, kind="no_resume_jia")
        else:
            why = None
            have["no_resume"] = have["no_resume"] or row_
        if why:
            problems.append(f"no_resume {auth.path}: {why}")
    for name in H.AUTHORIZATION_NEEDED[kind]:
        if have[name] is not None:
            return have[name], None
    return None, f"{kind}: no committed record of this cell writes it (holdout's D107 rules; {AUTH_PAIR_SET}, {hc.AUTH_DIRS[AUTH_PAIR_SET]}){'; ' + str(problems) if problems else ''}"


def _rejudged_cell(arm, row, cell, terminal_path, repo):
    """An external or configuration C arm's needs_lead_rejudgment (C26.8-4): its committed rejudgment record (bayesreg.io.rejudgment, runs/lpba40_test_v1/rejudgments/<arm>/
    <pair>_<k>.json, pair set 'lpba40_test') judging this terminal record at its sha256; holdout.rejudged_cell's classes.  (written, problem)."""
    spec = arm["authorization"]
    found, problems = rejudgment_io.rejudgment(spec["arm"], row[3], record=terminal_path, pair_set=AUTH_PAIR_SET, root=Path(repo))     # runs/lpba40_test_v1/rejudgments by the pair set
    if found is None:
        return False, f"{terminal_path}: needs_lead_rejudgment and no committed rejudgment record that holds ({rejudgment_io.LPBA40_REJUDGMENT_DIR_REL}): {problems}"
    document = found["document"]
    if document["status"] != cell.status:
        return False, f"{found['path']}: status {document['status']!r} is not the terminal status {cell.status!r}"
    cell.rejudgment = dict(path=found["path"], sha256=found["sha256"], k=found["k"], judgment=document["judgment"], git=found["git"])
    if document["judgment"] == "infrastructure":
        record, problem = _authorization(arm, row, "resumed", terminal_path, cell.records[str(terminal_path)][0], cell.records, cell.needed)
        if record is None:
            return False, f"{found['path']}: infrastructure before the resume ran only classifies the cell: {problem}"
        cell.authorization = record
    cell.written, cell.outcome, cell.cause = True, "no_output", "rejudgment"
    cell.no_output_class = dict(unit_limit="unit_limit", no_output="bz_no_rerun", infrastructure="infrastructure")[document["judgment"]]
    return True, None


FINALIZED_NAME = "finalized_outputs.json"          # LP-B's step-5 record of the method arm (experiments/hb/hb_review.finalize): <root>/review/finalized_outputs.json
FINALIZED_SCHEMA = H.FINALIZED_SCHEMA              # bayesreg.hb.finalized_outputs.v1


def finalized_path(pair_set):
    return root(pair_set)/"review"/FINALIZED_NAME


def finalized_record(pair_set, cache, records):
    """The mode's finalized record (C26.6-6, C22.H1-3; decision_log line 584 ruling 5), read once per run (`cache`): strict JSON, schema FINALIZED_SCHEMA, pair_set the mode's, pairs a dict; its bytes
    hashed into `records` (an input of the list).  (document or None, problem or None)."""
    path = finalized_path(pair_set)
    if "finalized" not in cache:
        if not path.is_file():
            cache["finalized"] = (None, None, f"{path}: not there: step 5 has not finalized the method arm (C26.6-6)")
        else:
            raw = path.read_bytes()
            try:
                doc, problem = H.strict_json(raw, str(path)), None
            except HoldoutRefused as exc:
                doc, problem = None, str(exc)
            if problem is None and (not isinstance(doc, dict) or doc.get("schema") != FINALIZED_SCHEMA or doc.get("pair_set") != pair_set or not isinstance(doc.get("pairs"), dict)):
                problem = f"{path}: schema {FINALIZED_SCHEMA}, pair_set {pair_set!r} and pairs required"
            cache["finalized"] = (doc, H.sha_bytes(raw), problem)
    doc, digest, problem = cache["finalized"]
    if digest is not None:
        records[str(path)] = (digest, doc)
    return (None if problem else doc), problem


def method_cross_check(arm, row, cell, pair_set, cache):
    """Decision_log line 584 ruling 5 (C26.6-6; holdout.method_cross_check without the move -- LPBA40 moves nothing, prereg §11): the method cell against LP-B's finalized record of the
    mode.  The pair's row names this very cell (no move); its line status is the terminal record's; its review row is the review.json read (finalized true, at that sha256), and a
    pair whose review.json is finalized but that the finalized record leaves out is not written; every file this entry reads in the cell has the sha256 the finalized record names
    (terminal.json and review/review.json by their rows, the output files by part.output: holdout.finalized_sha256), compared now with what was read and later with the list
    (cell.pinned_sha256); a no_resume of BZ that writes the cell is the one its bz_decision names.  None, or the problem."""
    final, problem = finalized_record(pair_set, cache, cell.records)
    path = finalized_path(pair_set)
    if final is None:
        return problem
    cell.needed.append(path)
    frow = final["pairs"].get(row[3])
    base = H.record_dir(arm, row)
    if not isinstance(frow, dict):
        return f"{path}: no row for {row[3]} (the pair is not finalized)"
    if os.path.realpath(str(frow.get("cell"))) != os.path.realpath(str(base)):
        return f"{path}: the row's cell {frow.get('cell')!r} is not the arm table's {base} (nothing moves on LPBA40)"
    if frow.get("line_status") != cell.status:
        return f"{path}: line_status {frow.get('line_status')!r} is not the terminal record's {cell.status!r}"
    if cell.review is not None:
        review_row = frow.get("review") or {}
        read = cell.records.get(cell.review["path"])
        if os.path.realpath(str(review_row.get("path"))) != os.path.realpath(cell.review["path"]) or read is None or review_row.get("sha256") != read[0]:
            return f"{path}: its review row {review_row} is not the review.json read ({cell.review['path']}, finalized {cell.review['value']!r})"
    pinned = {}
    for p in cell.needed:
        if not H._inside(p, base):
            continue
        rel = Path(os.path.abspath(str(p))).relative_to(os.path.abspath(str(base))).as_posix()
        want = H.finalized_sha256(frow, rel)
        if not is_hex64(want):
            return f"{path}: names no sha256 for {rel} of {row[3]} (the entry reads {p})"
        read = cell.records.get(str(p))
        if read is not None and read[0] != want:
            return f"{p}: read at sha256 {read[0]}, the finalized record names {want}"
        pinned[str(p)] = want
    if cell.authorization is not None and cell.authorization.get("kind") in ("no_resume_jia", "no_resume_p20"):
        decided = frow.get("bz_decision")
        if not isinstance(decided, dict) or decided.get("sha256") != cell.authorization["sha256"]:
            return f"{path}: bz_decision {decided!r} of {row[3]} is not the no_resume record {cell.authorization['path']} the entry found"
    cell.pinned_sha256 = pinned
    cell.move = dict(finalized=dict(path=str(path), line_status=frow.get("line_status"), bz_decision=frow.get("bz_decision"), cell=frow.get("cell")),
                     rule="decision_log line 584 ruling 5: the method cell against the finalized record (no move on LPBA40)")
    return None


def classify_cell(arm, row, *, repo=None, pair_set="lpba40_test", cache=None):
    """One (arm, pair) cell by D107's classes (module docstring).  Reads only the cell's terminal record, the method's review record, the records its authorizations name
    (hashed), XB's stop record and committed records of the code tree; no field, no truth.  Returns a holdout.Cell."""
    rootdir = Path(REPO if repo is None else repo)
    name = row[3]
    base = H.record_dir(arm, row)
    records = {}
    cell = H.Cell(arm["name"], name, row[0], False, None, None, None, None, [], records)
    terminal_path = next((base/H.fill(rel, arm, row) for rel in arm["terminal"]["files"] if (base/H.fill(rel, arm, row)).is_file()), None)
    if terminal_path is None:
        if arm["family"] != "method":
            written, problem = _stop_cell(arm, row, rootdir, cell)
            if written:
                cell.needed = list(dict.fromkeys(cell.needed))
                return cell
            if problem is not None:
                cell.problem = problem
                return cell
        cell.problem = f"{base}: no terminal record ({arm['terminal']['files']}) and no committed stop record of the arm"
        return cell
    cell.terminal = str(terminal_path)
    document = H._read_record(terminal_path, records)
    cell.needed.append(terminal_path)
    status = H.dotted(document, arm["terminal"]["status_key"])
    cell.status = status
    classes = {s: key for key, values in arm["terminal"]["classes"].items() for s in values}
    classes.update(H.FIXED_STATUSES)
    kind = classes.get(status) if isinstance(status, str) else None
    if status == "needs_lead_rejudgment" and arm["family"] != "method":
        written, problem = _rejudged_cell(arm, row, cell, terminal_path, rootdir)
        if not written:
            cell.problem = problem
        cell.needed = list(dict.fromkeys(cell.needed))
        return cell
    if status == H.METHOD_P20 and arm["family"] == "method":
        kind = "p20"
    if kind is None or kind == "not_written":
        cell.problem = f"{terminal_path}: {arm['terminal']['status_key']} = {status!r} is not a written terminal state (D107; C26.8-5: stop, to BZ)"
        return cell
    if kind in H.AUTHORIZATION_NEEDED:
        record, problem = _authorization(arm, row, kind, terminal_path, records[str(terminal_path)][0], records, cell.needed)
        if record is None:
            cell.problem = f"{terminal_path}: {status!r}: {problem}"
            return cell
        cell.authorization = record
    if arm["export_check"] is not None and kind == "output":
        flag = H.dotted(document, arm["export_check"]["key"])
        if not isinstance(flag, bool):
            cell.problem = f"{terminal_path}: {arm['export_check']['key']} is not a bool (the export check of a complete record)"
            return cell
        cell.export_suspicious = flag
    if kind == "output":
        missing = [rel for rel in arm["complete_files"] if not (base/H.fill(rel, arm, row)).is_file()]
        review_path = H.review_record_path(arm, row) if arm["family"] == "method" else None
        if review_path is not None and not review_path.is_file():
            missing.append(str(review_path))
        if missing:
            cell.problem = f"{base}: the complete record lacks {missing}"
            return cell
        if review_path is not None:                    # C26.6-6: only a finalized pair is written
            key = H.fill(arm["review_record"]["status_key"], arm, row)
            review_doc = H._read_record(review_path, records)
            value = H.dotted(review_doc, key)
            cell.review = dict(path=str(review_path), key=key, value=value, final=list(arm["review_record"]["final"]))
            if not H._final(value, arm["review_record"]["final"]):
                cell.problem = f"{review_path}: {key} = {value!r} is not one of {arm['review_record']['final']}: the pair is not finalized (C26.6-6)"
                return cell
            cell.needed.append(review_path)
        cell.needed += [base/H.fill(rel, arm, row) for rel in arm["complete_files"]]
        field = arm["field"]
        if "dir" in field:
            directory = Path(H.fill(field["dir"], arm, row))
        else:
            path, digest = H.dotted(document, field["file_key"]), H.dotted(document, field["sha256_key"])
            froot = H.template_root(field["root"].replace("{arm}", arm["name"]))
            if not (isinstance(path, str) and path.startswith("/") and Path(path).name == "disp.npz" and H._inside(path, froot) and is_hex64(digest)):
                cell.problem = f"{terminal_path}: {field['file_key']} = {path!r} / {field['sha256_key']} = {digest!r}: a disp.npz under {froot} and its sha256 required"
                return cell
            if not Path(path).is_file():
                cell.problem = f"{path}: the field the terminal record names is not there"
                return cell
            directory = Path(path).parent
            cell.field_sha256 = {str(path): digest}
        cell.field_dir = str(directory)
        cell.needed += H.kind_files(arm, directory)
        cell.written, cell.outcome = True, "output"
    else:
        cell.written, cell.outcome = True, "no_output"
        cell.no_output_class = H.NO_OUTPUT_CLASS[kind]
        if kind == "p20":
            cell.cause = H.P20_WORD
    cell.needed = list(dict.fromkeys(cell.needed))
    if arm["family"] == "method":                                       # decision_log line 584 ruling 5: LP-B's finalized record
        problem = method_cross_check(arm, row, cell, pair_set, {} if cache is None else cache)
        if problem:
            cell.written, cell.outcome, cell.no_output_class, cell.problem = False, None, None, f"{row[3]}: the method cell against the finalized record: {problem}"
            return cell
        cell.needed = list(dict.fromkeys(cell.needed))
    return cell


def check_cells(table, rows, *, repo=None, pair_set="lpba40_test"):
    """Every arm on every pair written (C26.7-2 "场都已写完"); HoldoutRefused('arms') listing those that are not (no pair is evaluated then)."""
    cells, problems, cache = {}, [], {}
    for arm in table["arms"]:
        for row in rows:
            cell = classify_cell(arm, row, repo=repo, pair_set=pair_set, cache=cache)
            cells[(arm["name"], row[3])] = cell
            if not cell.written:
                problems.append(f"{arm['name']} {row[3]}: {cell.problem}")
    if problems:
        raise HoldoutRefused("arms", f"{len(problems)} cells are not written (D107): {problems[:10]}{' ...' if len(problems) > 10 else ''}")
    return cells


def spy_roots():
    """The places watched from the spy's start (review I3): RAW_ROOT, the whole DATA_ROOT, THREE_D_ROOTS and <RESULTS>/holdout_B_v1 (all read at call time)."""
    return list(dict.fromkeys([Path(RAW_ROOT), Path(DATA_ROOT), *map(Path, THREE_D_ROOTS), Path(RESULTS)/HOLDOUT_RESULTS_NAME]))


def watched_roots(table):
    """The spy's places: spy_roots(), the stage, every arm's record, field and authorization-record roots."""
    roots = spy_roots()+[Path(table["stage_root"]), Path(table["stage_root"]).parent/"review"]+[r for arm in table["arms"] for r in H.arm_roots(arm)]
    return list(dict.fromkeys(roots))


# ---- the stage manifests and the images' headers ----------------------------------------------------------------------------------------------------------
def stage_manifest(table, row):
    return Path(table["stage_root"])/row[3]/"manifest.json"


def stage_document(path, listed):
    """The whole stage manifest (it holds no field derived from or naming a label: prep_lpba P6), from bytes hashed against the list just before."""
    return H.strict_json(H.verified_bytes(path, listed), str(path))


def image_originals(table, rows):
    """{subject: (path, sha256)} of the original image.nii.gz each stage manifest names (source_files.<side>.original), read from the manifests at their present bytes
    (admission: before the list holds them, they are hashed and compared with the list after).  Every path under RAW_ROOT/<subject>/delineation/image.nii.gz; one per
    subject (two pairs naming one subject must agree).  HoldoutRefused('stage')."""
    out = {}
    for row in rows:
        path = stage_manifest(table, row)
        document = H.strict_json(path.read_bytes(), str(path))
        files = document.get("source_files") if isinstance(document, dict) else None
        for side, subject in (("fixed", row[1]), ("moving", row[2])):
            original = (((files or {}).get(side) or {}).get("original") or {})
            want = Path(RAW_ROOT)/subject/IMAGE_REL
            if (files or {}).get(side, {}).get("subject") != subject or os.path.abspath(str(original.get("path"))) != os.path.abspath(str(want)) \
                    or not is_hex64(original.get("sha256")):
                raise HoldoutRefused("stage", f"{path}: source_files.{side} does not name {subject}'s original {want} with a sha256")
            if subject in out and out[subject] != (str(want), original["sha256"]):
                raise HoldoutRefused("stage", f"{subject}: two stage manifests name different originals {out[subject]} / {original}")
            out[subject] = (str(want), original["sha256"])
    return out


def check_stage(table, rows, listed):
    """Each pair's stage manifest (hashed against the list first): its subjects the row's, in the row's direction; the grid read (_read_manifest), no member of pair.npz
    opened.  {pair: grid metadata}."""
    grids = {}
    for row in rows:
        path = stage_manifest(table, row)
        H.verify_files([path], listed)
        _, metadata, selected = _read_manifest(path)
        if (selected["fixed_subject"], selected["moving_subject"]) != (row[1], row[2]):
            raise HoldoutRefused("pairs", f"{path}: {selected['fixed_subject']} <- {selected['moving_subject']} is not the row {row[1]} <- {row[2]}")
        grids[row[3]] = metadata
    return grids


# ---- the step-10 manifest (formal) and the run's own list (feasibility) ----------------------------------------------------------------------------------
def input_roles(table, rows, cells, originals):
    """{path: role} of every file this entry reads: the cells' records and field files, each pair's stage manifest.json and pair.npz, each subject's original image.nii.gz
    (its header: the affine the label is checked against, C26.7-3)."""
    roles = {str(p): "cell" for cell in cells.values() for p in cell.needed}
    stage = Path(table["stage_root"])
    roles.update({str(stage/row[3]/name): "stage" for row in rows for name in ("manifest.json", "pair.npz")})
    roles.update({path: "image_header" for path, _ in originals.values()})
    return roles


def parse_manifest(raw, where=MANIFEST_REL):
    """The step-10 file (prereg §8.2 step 10; holdout.parse_manifest's form): '# bayesreg.lpba40.eval_manifest.v1', '# prereg_sha256 <hex>', '# arm_table_sha256 <hex>' (once
    each), then '<64 hex>  <absolute path>' per input.  HoldoutRefused('manifest') on any other line or a duplicate path."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HoldoutRefused("manifest", f"{where}: not UTF-8") from None
    lines = text.splitlines()
    if not lines or lines[0] != f"# {SCHEMA_MANIFEST}":
        raise HoldoutRefused("manifest", f"{where}: the first line must be '# {SCHEMA_MANIFEST}'")
    header, inputs = {}, {}
    for number, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        parts = line.split(" ")
        if line.startswith("# ") and len(parts) == 3 and parts[1] in MANIFEST_HEADER:
            if parts[1] in header or not is_hex64(parts[2]):
                raise HoldoutRefused("manifest", f"{where}:{number}: header {parts[1]} once, with a 64-hex value")
            header[parts[1]] = parts[2]
            continue
        match = H._ENTRY.fullmatch(line)
        if not match or match.group(2) in inputs:
            raise HoldoutRefused("manifest", f"{where}:{number}: only input lines '<64 hex>  <absolute path>' (one per path); record-only lines go to {RECORDS_REL}")
        inputs[match.group(2)] = match.group(1)
    if sorted(header) != sorted(MANIFEST_HEADER):
        raise HoldoutRefused("manifest", f"{where}: the header lines {list(MANIFEST_HEADER)} are required")
    return header, inputs


def check_manifest(table, rows, cells, originals, *, prereg_sha256, arm_table_sha256, repo=None):
    """C26.7-2: the manifest committed and unchanged; its header the pinned preregistration and arm table; its lines exactly the files this entry reads (all there); the records
    read in the admission at their listed sha256; a field whose terminal record names its sha256 listed at that sha256; an original image at the sha256 its stage manifest
    names; the record-only file committed with it.  (record, inputs)."""
    rootdir = Path(REPO if repo is None else repo)
    raw, digest, git = H.read_committed(MANIFEST_REL, "manifest", rootdir)
    header, inputs = parse_manifest(raw)
    want = dict(prereg_sha256=prereg_sha256, arm_table_sha256=arm_table_sha256)
    if header != want:
        raise HoldoutRefused("manifest", f"{MANIFEST_REL}: header {header} is not {want}")
    roles = input_roles(table, rows, cells, originals)
    outside = [p for p in inputs if p not in roles]
    if outside:
        raise HoldoutRefused("manifest", f"{MANIFEST_REL}: {len(outside)} input lines are outside the files this entry reads (C26.7-2): {outside[:5]}")
    missing = sorted(set(roles)-set(inputs))
    if missing:
        raise HoldoutRefused("manifest", f"{len(missing)} files this entry reads are not in the manifest: {missing[:5]}")
    absent = sorted(p for p in inputs if not Path(p).is_file())
    if absent:
        raise HoldoutRefused("manifest", f"{len(absent)} input files are not there: {absent[:5]}")
    _cell_sha_checks(cells, originals, inputs, MANIFEST_REL)
    return dict(path=MANIFEST_REL, sha256=digest, git=git, header=header, n_inputs=len(inputs), records_file=records_file_state(git, rootdir)), inputs


def records_file_state(manifest_git, repo=None):
    """The record-only file RECORDS_REL (prereg §8.2 step 10: "只记录"的行另放一个文件，同一次提交): tracked, unchanged, last changed in the manifest's commit; never read
    (holdout.records_file_state's rule with this test set's path)."""
    rootdir = Path(REPO if repo is None else repo)
    record = H.tracked_unchanged(rootdir/RECORDS_REL, "manifest", rootdir)
    if record["last_commit"] != manifest_git.get("last_commit"):
        raise HoldoutRefused("manifest", f"{RECORDS_REL} was last committed in {record['last_commit']}, the manifest in {manifest_git.get('last_commit')}: one commit")
    blob = H._git(rootdir, "ls-files", "-s", "--", RECORDS_REL).stdout.split()
    return dict(record, blob=blob[1] if len(blob) > 1 else None, read=False)


def _cell_sha_checks(cells, originals, inputs, where):
    for cell in cells.values():
        for path, (sha, _) in cell.records.items():
            if inputs.get(path) != sha:
                raise HoldoutRefused("manifest", f"{path}: read in the admission at sha256 {sha}, {where} lists {inputs.get(path)}")
        for path, sha in (cell.field_sha256 or {}).items():
            if inputs.get(path) != sha:
                raise HoldoutRefused("manifest", f"{path}: the terminal record {cell.terminal} names sha256 {sha}, {where} lists {inputs.get(path)}")
        for path, sha in (cell.pinned_sha256 or {}).items():
            if inputs.get(path) != sha:
                raise HoldoutRefused("manifest", f"{path}: the finalized record names sha256 {sha}, {where} lists {inputs.get(path)}")
    for path, sha in originals.values():
        if inputs.get(path) != sha:
            raise HoldoutRefused("manifest", f"{path}: the stage manifest names sha256 {sha}, {where} lists {inputs.get(path)}")


def own_list(table, rows, cells, originals):
    """The feasibility mode (no committed manifest, C26.9-1): every file this entry reads hashed now; the same checks as check_manifest against this list.  (record, inputs)."""
    roles = input_roles(table, rows, cells, originals)
    absent = sorted(p for p in roles if not Path(p).is_file())
    if absent:
        raise HoldoutRefused("manifest", f"{len(absent)} files this entry reads are not there: {absent[:5]}")
    inputs = {p: file_sha256(p) for p in roles}
    _cell_sha_checks(cells, originals, inputs, "the run's own list")
    text = "".join(f"{sha}  {path}\n" for path, sha in sorted(inputs.items()))
    return dict(path=None, sha256=H.sha_bytes(text.encode("utf-8")), git=None, n_inputs=len(inputs), roles=roles, rule="feasibility: the run's own list, hashed at admission"), inputs


# ---- the locks and the reevaluation authorization (D109; C26.7-8) ------------------------------------------------------------------------------------------
def run_lines(pair_set="lpba40_test"):
    """[(lock path, line)] of the master lock and the reevaluation locks reeval_<n>.lock in the order of n (the decision code's only source of the runs)."""
    out = [(master_lock(pair_set), line) for line in H.lock_lines(master_lock(pair_set))]
    found = []
    folder = eval_dir(pair_set)
    for path in folder.glob("reeval_*.lock") if folder.is_dir() else []:
        match = REEVAL_LOCK.fullmatch(path.name)
        if match:
            found.append((int(match.group(1)), path))
    for _, path in sorted(found):
        out += [(path, line) for line in H.lock_lines(path)]
    return out


def completed_pairs(pair_set="lpba40_test"):
    done = {}
    for _, line in run_lines(pair_set):
        out = line.get("out_dir")
        folder = Path(out)/"pairs" if out else None
        for path in sorted(folder.glob("*.json")) if folder is not None and folder.is_dir() else []:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if record.get("complete") is True:
                done.setdefault(record.get("pair"), out)
    return done


def load_reeval(path, *, manifest_sha256, rows, repo=None):
    """A committed reevaluation authorization (D109, C26.7-8, C26.8-4): splits/lpba40_test_v1/lpba40_test_v1_reeval_<n>.json, unchanged; written by BZ (its own words) or "lead
    on BZ's behalf" (the words BZ_WORDS of decision_log line 520 word for word); a reason of holdout.REEVAL_REASONS (the evaluation's own infrastructure failures); the master
    lock's sha256 and the master lock's manifest only; pairs of the list not complete and not reevaluated before.  (n, document, record); HoldoutRefused('reeval')."""
    rootdir = Path(REPO if repo is None else repo)
    target = Path(path).resolve()
    match = REEVAL_NAME.fullmatch(target.name)
    if target.parent != (rootdir/REEVAL_DIR_REL).resolve() or not match:
        raise HoldoutRefused("reeval", f"{path}: a reevaluation authorization is {REEVAL_DIR_REL}/lpba40_test_v1_reeval_<n>.json")
    rel = target.relative_to(rootdir.resolve()).as_posix()
    raw, digest, git = H.read_committed(rel, "reeval", rootdir)
    document = H.strict_json(raw, rel)
    if not isinstance(document, dict) or sorted(document) != sorted(REEVAL_KEYS) or document["schema"] != SCHEMA_REEVAL:
        raise HoldoutRefused("reeval", f"{rel}: exactly the keys {list(REEVAL_KEYS)} and schema {SCHEMA_REEVAL}")
    problems = []
    if document["written_by"] not in REEVAL_WRITERS:
        problems.append(f"written_by {document['written_by']!r}: one of {REEVAL_WRITERS}")
    for key in ("words", "time_local", "reason"):
        if not isinstance(document[key], str) or not document[key].strip():
            problems.append(f"{key}: non-empty")
    if document["written_by"] == REEVAL_WRITERS[1] and document["words"] != BZ_WORDS:
        problems.append(f"words {document['words']!r}: on BZ's behalf, BZ's words of decision_log line 520 word for word ({BZ_WORDS!r})")
    if document["reason_class"] not in H.REEVAL_REASONS:
        problems.append(f"reason_class {document['reason_class']!r}: one of {H.REEVAL_REASONS}")
    problems += [p for p in (H.utc_problem(document["utc"], "utc"),) if p]
    if not isinstance(document["previous_out"], list) or not all(isinstance(o, str) for o in document["previous_out"]):
        problems.append("previous_out: a list of the earlier runs' output directories")
    lock = master_lock("lpba40_test")
    if not lock.is_file():
        raise HoldoutRefused("reeval", f"{lock}: there is no master lock; a reevaluation follows a first run")
    master = H.lock_lines(lock)[0]
    if document["master_lock_sha256"] != file_sha256(lock):
        problems.append(f"master_lock_sha256 {document['master_lock_sha256']!r} is not the master lock's")
    if document["manifest_sha256"] != master.get("manifest_sha256") or manifest_sha256 != master.get("manifest_sha256"):
        problems.append(f"the manifest {manifest_sha256} is not the master lock's {master.get('manifest_sha256')} (D109: the same manifest only)")
    names = {row[3] for row in rows}
    asked = document["pairs"]
    if not isinstance(asked, list) or not asked or len(set(asked)) != len(asked) or not set(asked) <= names:
        problems.append("pairs: a non-empty list of distinct pairs of the list")
        asked = []
    done = completed_pairs("lpba40_test")
    earlier = {}
    for other in sorted((rootdir/REEVAL_DIR_REL).glob("lpba40_test_v1_reeval_*.json")):
        m = REEVAL_NAME.fullmatch(other.name)
        if not m or int(m.group(1)) == int(match.group(1)):
            continue
        try:
            for name in json.loads(other.read_text(encoding="utf-8")).get("pairs") or []:
                earlier.setdefault(name, str(other))
        except (OSError, ValueError, AttributeError):
            raise HoldoutRefused("reeval", f"{other}: an earlier reevaluation authorization that cannot be read") from None
    for lock_path, line in run_lines("lpba40_test"):
        if lock_path != lock:
            for name in line.get("pairs") or []:
                earlier.setdefault(name, str(lock_path))
    for name in asked:
        if name in done:
            problems.append(f"{name} was evaluated completely by {done[name]}")
        if (pair_lock_dir("lpba40_test")/f"{name}.reeval.lock").exists() or name in earlier:
            problems.append(f"{name} was reevaluated once already or is named by {earlier.get(name, 'its .reeval.lock')} (D109: at most once)")
    if problems:
        raise HoldoutRefused("reeval", f"{rel}: {problems}")
    return int(match.group(1)), document, dict(path=rel, sha256=digest, git=git)


def take_pair_lock(admission, row, line):
    """The pair lock before anything of its truth is read (D109): <pair>.lock on a first read; on an authorized reevaluation <pair>.reeval.lock."""
    state = _state(admission, "once")
    if not state["locked"]:
        raise HoldoutRefused("once", "the master (or reevaluation) lock is not taken by this admission")
    name = row[3]
    folder = pair_lock_dir(admission.mode)
    first = folder/f"{name}.lock"
    reeval = admission.record.get("reeval")
    if reeval is not None and name not in reeval["pairs"]:
        raise HoldoutRefused("once", f"{name} is not a pair of the reevaluation authorization")
    if reeval is None or not first.exists():
        H.take_lock(first, dict(line, pair=name, read="first"))
        state["pair_locks"][name] = str(first)
    else:
        path = folder/f"{name}.reeval.lock"
        H.take_lock(path, dict(line, pair=name, read="reevaluation"))
        state["pair_locks"][name] = str(path)
    return state["pair_locks"][name]


# ---- the out dir ------------------------------------------------------------------------------------------------------------------------------------
def check_out_dir(out_dir, pair_set, table, rows, cells=None):
    """--out-dir: new, a directory under the mode's eval/ (not eval/ itself, its locks or pair_locks); not in the code tree, a seq_vp_3d tree, RAW_ROOT; not inside a
    cell, a field directory or a root the spy watches; the lock directories not under a watched root either."""
    out = Path(out_dir)
    folder = eval_dir(pair_set)
    if out.exists():
        raise HoldoutRefused("output", f"{out} exists: a new --out-dir for every run; nothing is overwritten")
    if H._under_read_only(out) or H._inside(out, REPO) or H._inside(out, RAW_ROOT):
        raise HoldoutRefused("output", f"{out}: not under a seq_vp_3d tree, the data root or inside the code tree")
    if not H._inside(out, folder) or out.resolve() == folder.resolve() or H._inside(out, pair_lock_dir(pair_set)) or REEVAL_LOCK.fullmatch(out.name) \
            or out.name in (MASTER_LOCK_NAME, PAIR_LOCK_DIR_NAME):
        raise HoldoutRefused("output", f"{out}: --out-dir is a new directory under {folder} (C26.7-7)")
    watched = watched_roots(table)
    if any(H._inside(folder, r) for r in watched):
        raise HoldoutRefused("output", f"{folder} is under a directory the spy watches: the lock could not be written")
    places = [H.record_dir(arm, row) for arm in table["arms"] for row in rows]+[Path(c.field_dir) for c in (cells or {}).values() if c.field_dir]
    if any(H._inside(out, p) for p in places) or any(H._inside(out, r) for r in watched):
        raise HoldoutRefused("output", f"{out}: inside an evaluated cell, its field directory or a watched root")
    return out


# ---- stage 1: the admission ----------------------------------------------------------------------------------------------------------------------------------
def open_lpba(out_dir, pair_set="lpba40_test", *, reeval_path=None, run=None, cgroup=H.PROC_CGROUP, repo=None):
    """Stage 1 (module docstring).  Returns (admission, context); nothing of a subject is read.  On a refusal the spy is stopped."""
    check_pair_set(pair_set)
    rootdir = Path(REPO if repo is None else repo)
    stop_spy()
    spy = start_spy(spy_roots())                                        # review I3: under these roots only the run's list may be opened
    try:
        return _open_lpba(out_dir, pair_set, rootdir, spy, reeval_path=reeval_path, run=run, cgroup=cgroup)
    except BaseException:
        stop_spy()
        raise


def _admit(pair_set):
    """experiments/lpba/lpba_common.admit (pins, the LPBA allowlist, the mode's names, sealed lists, check_subject per subject: L34).  Its Refused become HoldoutRefused."""
    lc = _lpba_module("lpba_common")
    try:
        return lc.admit(pair_set)
    except lc.Refused as exc:
        raise HoldoutRefused("admission", f"lpba_common.admit ({getattr(exc, 'stage', '?')}): {exc}; nothing is read") from None


def _open_lpba(out_dir, pair_set, rootdir, spy, *, reeval_path, run, cgroup):
    identity = check_code_identity(pair_set, rootdir)                  # C26.9: before anything else
    admitted = _admit(pair_set)                                         # C26.7-2: pins, names, sealed lists, check_subject (admitted_by_lpba40)
    if reeval_path is None and master_lock(pair_set).exists():           # review minor 5: both modes, before the arm table and the cells
        raise HoldoutRefused("once", f"{master_lock(pair_set)} exists: evaluated once; a reevaluation needs a committed authorization (D109)")
    prereg = check_prereg(rootdir)
    table, table_record = load_arm_table(pair_set, rootdir)
    rows = pair_rows(pair_set, admitted.lists)
    if [(r[1], r[2]) for r in rows] != [tuple(p) for p in admitted.pairs]:
        raise HoldoutRefused("pairs", "the admitted pairs are not the mode's table in its order")
    spy.watch(watched_roots(table))
    cells = check_cells(table, rows, repo=rootdir, pair_set=pair_set)
    machine = H.machine_gate(run=run, cgroup=cgroup, own_units=OWN_UNITS[pair_set])
    originals = image_originals(table, rows)
    if pair_set == "lpba40_test":
        manifest, listed = check_manifest(table, rows, cells, originals, prereg_sha256=prereg["sha256"], arm_table_sha256=table_record["sha256"], repo=rootdir)
    else:
        manifest, listed = own_list(table, rows, cells, originals)
    spy.allow(list(listed)+[p for p in (Path(sealed_module.REPO)/spec["path"] for spec in sealed_module.SEALED_LISTS)])
    grids = check_stage(table, rows, listed)
    out = check_out_dir(out_dir, pair_set, table, rows, cells)
    reeval = None
    if reeval_path is not None:
        if pair_set != "lpba40_test":
            raise HoldoutRefused("reeval", "a reevaluation is the formal mode's only (C26.6-2: the feasibility mode has no authorization)")
        n, document, record = load_reeval(reeval_path, manifest_sha256=manifest["sha256"], rows=rows, repo=rootdir)
        reeval = dict(n=n, pairs=list(document["pairs"]), record=record, reason=document["reason"], reason_class=document["reason_class"])
    elif master_lock(pair_set).exists():
        raise HoldoutRefused("once", f"{master_lock(pair_set)} exists: evaluated once (D109)")
    code = identity["code"]
    if not code["clean"]:
        raise HoldoutRefused("code", f"the code tree is not clean ({list(H.CLEAN_DIRS)}): {code['status'][:200]!r}")
    line = dict(schema=SCHEMA_RUN, pair_set=pair_set, prereg_sha256=prereg["sha256"], arm_table_sha256=table_record["sha256"], manifest_sha256=manifest["sha256"],
                manifest_commit=(manifest.get("git") or {}).get("last_commit"), code_commit=code["head"],
                pinned_commit=(identity.get("rule_one") or {}).get("pinned"), out_dir=str(out.resolve()), truth=pair_set == "lpba40_test", utc=H.utc(), pid=os.getpid())
    if reeval is None:
        lock = dict(path=str(master_lock(pair_set)), line=H.take_lock(master_lock(pair_set), line))
    else:
        path = eval_dir(pair_set)/f"reeval_{reeval['n']}.lock"
        lock = dict(path=str(path), line=H.take_lock(path, dict(line, reeval=reeval["record"], pairs=reeval["pairs"])))
    record = dict(pair_set=pair_set, identity=identity, admission=admitted.record, prereg=prereg, arm_table=table_record, machine=machine,
                  manifest={k: v for k, v in manifest.items() if k != "roles"}, reeval=reeval, lock=lock, truth=pair_set == "lpba40_test")
    admission = _issue(pair_set, [s for row in rows for s in (row[1], row[2])], rows, record)
    _STATE[id(admission)]["locked"] = True
    out.mkdir(parents=True)
    return admission, dict(table=table, cells=cells, listed=listed, grids=grids, originals=originals, out=out, line=line, record=record, spy=spy,
                           allowlist=admitted.allowlist)


# ---- stage 2: the truth (class T, D147) ---------------------------------------------------------------------------------------------------------------------------
def _check_admission(admission, subject, pair_name):
    state = _state(admission)
    if admission.mode != "lpba40_test":
        raise HoldoutRefused("truth", "the no-truth mode reads no label (C26.7-6)")
    if not state["locked"]:
        raise HoldoutRefused("truth", "the lock of this admission is not taken: no truth is read before it")
    row = next((r for r in admission.pairs if r[3] == pair_name), None)
    if row is None or subject not in (row[1], row[2]):
        raise HoldoutRefused("truth", f"{subject} is not a subject of the admitted pair {pair_name}")
    lock = state["pair_locks"].get(pair_name)
    if lock is None or not Path(lock).is_file():
        raise HoldoutRefused("truth", f"the pair lock of {pair_name} is not taken")
    return row


def private_temp_dir():
    """A private temporary directory (tempfile.mkdtemp, mode 0700) that is not under the results root, the code tree or RAW_ROOT (C26.7-3: no truth copy under the results root)."""
    folder = Path(tempfile.mkdtemp(prefix="lpba_label_"))
    if any(H._inside(folder, r) for r in (Path(RESULTS), REAL_RESULTS, REPO, RAW_ROOT)):
        shutil.rmtree(folder, ignore_errors=True)
        raise HoldoutRefused("truth", f"{folder}: the temporary directory is under the results root, the code tree or the data root (set TMPDIR elsewhere)")
    return folder


def _label_volume(src, folder, *, admission, subject, pair_name):
    """The label's mm copy in `folder` (only the xyzt units set to mm: nibabel Nifti1Image(np.asanyarray(dataobj), affine, header.copy()); data, affine, qform and sform with
    their codes checked equal to the original's), then load_scalar(labels=True) of the copy.  Returns (volume, the original's affine and stored shape, copy record)."""
    import nibabel as nib
    _check_admission(admission, subject, pair_name)                      # review minor 6: its own guard (the issued admission, the locks, the pair)
    if Path(src) != Path(RAW_ROOT)/subject/LABEL_REL:
        raise HoldoutRefused("truth", f"{src} is not {subject}'s label")
    original = nib.load(str(src))
    data = np.asanyarray(original.dataobj)
    out = nib.Nifti1Image(data, original.affine, original.header.copy())
    out.header.set_xyzt_units("mm")
    copy = Path(folder)/"label_mm.nii.gz"
    nib.save(out, str(copy))
    back = nib.load(str(copy))
    problems = []
    if not (np.asanyarray(back.dataobj).dtype == data.dtype and back.shape == data.shape and np.asanyarray(back.dataobj).tobytes() == data.tobytes()):
        problems.append("the data differ")
    if not np.array_equal(back.affine, original.affine):
        problems.append("the affine differs")
    for name, get in (("qform", lambda im: im.header.get_qform(coded=True)), ("sform", lambda im: im.header.get_sform(coded=True))):
        (a, ca), (b, cb) = get(original), get(back)
        if ca != cb or (a is None) != (b is None) or (a is not None and not np.array_equal(a, b)):
            problems.append(f"the {name} or its code differs")
    if problems:
        raise TruthMismatch("truth_copy", f"{src}: the mm copy {problems}; the evaluation stops")
    try:
        volume = load_scalar(copy, labels=True)
    except ValueError as exc:                                            # review minor 7: load_scalar's own label check (finite, non-negative, integer) stops as TruthMismatch
        raise TruthMismatch("truth_grid", f"{src}: {exc}; the evaluation stops before any Dice") from None
    return volume, (np.asarray(original.affine, dtype=np.float64), tuple(int(n) for n in original.shape)), dict(units_before=list(original.header.get_xyzt_units()),
                                                                                                                stored_dtype=str(original.get_data_dtype()))


def _image_header(path, sha256, listed):
    """The original image's affine and stored shape (nibabel header), its sha256 checked against the list and the stage manifest before the read."""
    import nibabel as nib
    H.verify_files([Path(path)], listed)
    if listed.get(str(path)) != sha256:
        raise ManifestChanged("manifest_changed", f"{path}: listed {listed.get(str(path))}, the stage manifest names {sha256}")
    image = nib.load(str(path))
    return np.asarray(image.affine, dtype=np.float64), tuple(int(n) for n in image.shape)


def id_record(values):
    """C26.7-3 "ID": the 56 ids (code constants) missing in this label map, and the values that are neither 0 nor one of the 56 (count and voxels)."""
    found, counts = np.unique(values.astype(np.int64), return_counts=True)
    present = {int(v): int(c) for v, c in zip(found, counts)}
    missing = [i for i in LPBA40_IDS if i not in present]
    unexpected = {str(v): c for v, c in present.items() if v != 0 and v not in LPBA40_IDS}
    return dict(missing_ids=missing, n_missing=len(missing), unexpected_values=unexpected, n_unexpected_values=len(unexpected),
                n_unexpected_voxels=int(sum(unexpected.values())), rule="a missing id: recorded, dice_labels' rule; an unexpected value: recorded, in no label set")


def left_right_record(values):
    """C26.7-3 (only recorded): for each of the 27 left-right pairs the centroid of each label on the array's axis 2 and the sign of (left - right); the expected sign -1 (left at
    the lower index, as OASIS).  A pair with a side missing: sign None."""
    pairs, agree, known = [], 0, 0
    axis = np.arange(values.shape[2], dtype=np.float64)
    for left, right in LR_PAIRS:
        cents = []
        for label in (left, right):
            mask = values == label
            n = int(mask.sum())
            cents.append(None if n == 0 else float((mask.sum(axis=(0, 1))*axis).sum()/n))
        sign = None if None in cents else int(np.sign(cents[0]-cents[1]))
        if sign is not None:
            known += 1
            agree += sign == LR_EXPECTED_SIGN
        pairs.append(dict(left=left, right=right, centroid_axis2_left=cents[0], centroid_axis2_right=cents[1], sign_left_minus_right=sign))
    return dict(axis=2, expected_sign=LR_EXPECTED_SIGN, pairs=pairs, n_pairs=len(LR_PAIRS), n_known=known, n_expected_sign=agree,
                role="recorded only (C26.7-3): nothing is changed by it")


def read_label(subject, *, admission, grid_metadata, pair_name, image, listed, spy=None):
    """Class T (D147; C26.7-1, -3): the one reader of an LPBA40 S03-S36 label.  `image`: (path, sha256) of the subject's original image the stage manifest names.  Returns
    (labels float64 on the stage's grid, record)."""
    _check_admission(admission, subject, pair_name)
    spy = _SPY if spy is None else spy                                 # the entry's spy, when one is on, lets this label (and its private copy) through for this read only
    src = Path(RAW_ROOT)/subject/LABEL_REL
    affine, shape = _image_header(image[0], image[1], listed)
    folder = private_temp_dir()
    try:
        if spy is not None:
            spy.watch([folder])
            spy.truth = frozenset(os.path.realpath(p) for p in (src, folder, folder/"label_mm.nii.gz"))      # the folder itself: its removal opens it
        original_sha256 = file_sha256(src)
        volume, (label_affine, label_shape), copy = _label_volume(src, folder, admission=admission, subject=subject, pair_name=pair_name)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        if spy is not None:
            spy.truth = frozenset()
    if label_shape != shape or not np.array_equal(label_affine, affine):
        raise TruthMismatch("truth_grid", f"{src}: the label's affine / shape {label_shape} differ from the image's {shape}; the evaluation of {subject} stops before any Dice")
    values = np.flip(np.asarray(volume.values, dtype=np.float64), 2).copy()
    grid = volume.grid
    own = dict(shape_zyx=list(grid.shape), spacing_xyz=[float(h) for h in grid.spacing], origin_xyz=[float(a[0]) for a in grid.axes])
    want = {k: [float(v) for v in grid_metadata[k]] if k != "shape_zyx" else list(grid_metadata[k]) for k in ("shape_zyx", "spacing_xyz", "origin_xyz")}
    if own != want:
        raise TruthMismatch("truth_grid", f"{src}: grid {own} differs from the stage manifest's {want}; the evaluation of {subject} stops before any Dice")
    if not np.all(np.isfinite(values)) or np.any(values < 0) or not np.array_equal(values, np.rint(values)):
        raise TruthMismatch("truth_grid", f"{src}: labels must be finite non-negative integers; the evaluation of {subject} stops before any Dice")
    record = dict(path=str(src), sha256=original_sha256, copy=dict(copy, place="a private temporary directory, removed after the read", removed=not folder.exists()),
                  transform="mm copy -> load_scalar(labels=True) -> np.flip(., 2) (prereg §2.2; the stage's P1-P3)", image=dict(path=image[0], sha256=image[1]),
                  affine_equal_to_image=True, grid=own, ids=id_record(values), left_right=left_right_record(values))
    return values, record


# ---- stage 3: the metrics ----------------------------------------------------------------------------------------------------------------------------------------
def tet24(u, spacing, state_hash, repo=None):
    """C26.7-5 (C24.12-4's order): sha256(u) equal to the same cell's BL1 state_hash (else FieldMismatch: stop), then folding_metrics.tet_ratio_summary(u, spacing) as it is,
    the module through bayesreg.eval.holdout_tet24.folding_module (this tree's scripts/folding_metrics.py, default schemes (24, 6, 5))."""
    from .holdout_tet24 import FieldMismatch, folding_module
    if array_sha256(u) != state_hash:
        raise FieldMismatch("field", f"sha256(u) is not the BL1 certificate's state_hash {state_hash}")
    from .holdout_tet24 import KEYS, SCHEMES
    fm = folding_module(REPO if repo is None else repo)
    summary = fm.tet_ratio_summary(u, tuple(float(h) for h in spacing))
    if sorted(summary) != sorted(f"tet{s}" for s in SCHEMES) or any(sorted(v) != sorted(KEYS) for v in summary.values()):      # review minor 9 (holdout_tet24.compute_cell's check)
        raise RuntimeError(f"tet_ratio_summary returned {sorted(summary)}: not the schemes {SCHEMES} with {KEYS}")
    return dict(tet_ratio_summary=summary, u_sha256_equals_bl1_state_hash=True,
                role="described only (prereg §5.3; C24.12-1): neither it nor BL1 implies the other")


def field_metrics(d, grid, brain):
    """The no-truth mode's metrics of a field (C26.7-6; metrics' definitions): official SDlogJ (and its folding count), the central-difference det counts (all, brain), the
    brain log J statistics; the brain mask = the stage's fixed image > 0."""
    det = metrics.central_det(d)
    sd, folds, neg = metrics.sdlogj_official(d, det)
    nonpos = det <= 0
    out = dict(sdlogj_official=sd, num_foldings_official=folds, central_det_nonpositive_fraction=neg, central_det_nonpositive=int(nonpos.sum()),
               central_det_nonpositive_brain=int(np.count_nonzero(nonpos & brain[2:-2, 2:-2, 2:-2])))
    out.update(metrics.brain_jacobian_stats(d_to_u(d, grid.spacing), tuple(float(h) for h in grid.spacing), brain))
    return out


def score_d(d, grid, truth, brain, *, certify=True, repo=None):
    """One field: with truth, metrics.evaluate_field (lpba56, lpba54, official Dice, HD95, SDlogJ, brain log J); without, field_metrics; with certify, BL1 (faces, domain)
    and the 24 tetrahedra on u = h d."""
    if truth is not None:
        scalars, per_label, official = metrics.evaluate_field(d, truth, SCHEME, grid.spacing)
        record = dict(metrics=scalars, per_label=per_label, official=official)
    else:
        record = dict(metrics=field_metrics(d, grid, brain), per_label=None, official=None)
    record["certificate"] = record["tet24"] = None
    if certify:
        u = d_to_u(d, grid.spacing)
        cert = H.certify_field_bl1(u, grid, brain_nodes=brain)
        cert["certified_field"] = "u = h d (voxels_to_displacement of the evaluated d)"
        record["certificate"] = cert
        record["tet24"] = tet24(u, grid.spacing, cert["state_hash"], repo)
    return record


def evaluate_pair(admission, context, row, record=None):
    """Stages 2 and 3 for one pair (its lock taken here), filling `record` as it goes (a failure keeps the arms already evaluated)."""
    table, cells, listed = context["table"], context["cells"], context["listed"]
    index, fixed, moving, name = row
    truth_mode = admission.mode == "lpba40_test"
    record = dict(schema=SCHEMA_PAIR, pair=name, index=index, fixed=fixed, moving=moving, complete=False, arms={}) if record is None else record
    record.update(lock=take_pair_lock(admission, row, context["line"]), started_utc=H.utc(), truth_read=False)
    stage = Path(table["stage_root"])/name
    H.verify_files([stage/"manifest.json", stage/"pair.npz"], listed)
    grid, metadata, selected = _read_manifest(stage/"manifest.json")
    if (selected["fixed_subject"], selected["moving_subject"]) != (fixed, moving):
        raise HoldoutRefused("pairs", f"{stage}/manifest.json: {selected['fixed_subject']} <- {selected['moving_subject']} is not the row {fixed} <- {moving}")
    from bayesreg.io.pair import load_pair
    pin = load_pair(stage/"pair.npz", allowlist=context["allowlist"])           # the stage through the admission (G): fixed, moving, initial_u only
    H.verify_files([stage/"manifest.json", stage/"pair.npz"], listed)
    brain = np.asarray(pin.fixed) > 0                                             # C26.7-4: the brain mask = the stage's fixed image > 0, no label
    record.update(grid=metadata, brain=dict(rule="fixed image > 0 (prereg §5.3)", nodes=int(brain.sum())))
    truth = None
    if truth_mode:
        labels, truth_record = {}, {}
        for side, subject in (("fixed", fixed), ("moving", moving)):
            labels[side], truth_record[side] = read_label(subject, admission=admission, grid_metadata=metadata, pair_name=name, image=context["originals"][subject],
                                                          listed=listed, spy=context.get("spy"))
        truth = dict(fixed_lpba56=labels["fixed"], moving_lpba56=labels["moving"], brain=brain)
        record.update(truth=truth_record, truth_read=True)
    record["identity"] = score_d(np.zeros((3,)+tuple(grid.shape)), grid, truth, brain, certify=False)
    for arm in table["arms"]:
        cell = cells[(arm["name"], name)]
        entry = dict(outcome=cell.outcome, no_output_class=cell.no_output_class, cause=cell.cause, terminal_status=cell.status, terminal=cell.terminal,
                     export_suspicious=cell.export_suspicious, authorization=cell.authorization, rejudgment=cell.rejudgment,
                     stop_record=None if cell.stop_record is None else {k: v for k, v in cell.stop_record.items() if k != "document"}, review=cell.review, finalized=cell.move)
        if cell.outcome == "output":
            H.verify_files(cell.needed, listed)
            info = describe_run(Path(cell.field_dir), pair=name, kind=arm["kind"])
            d, _, field = load_field(info, grid.shape, grid.spacing)
            H.verify_files(cell.needed, listed)
            if (field.get("disp") or {}).get("file_sha256") not in (None, listed.get(str(Path(cell.field_dir)/"disp.npz"))):
                raise ManifestChanged("manifest_changed", f"{cell.field_dir}/disp.npz changed while it was read")
            entry.update(score_d(d, grid, truth, brain), field=field, field_role=info.field_role)
            if arm["family"] == "method":
                H.verify_files(cell.needed, listed)
                entry["decision_certificate"] = H.decision_certificate(arm, row, grid, listed, dict(brain=brain), d)
        record["arms"][arm["name"]] = entry
    record.update(complete=True, ended_utc=H.utc())
    return record


def evaluate_lpba(admission, context):
    """Every admitted pair in list order (a reevaluation: its authorized pairs).  An error of a pair is kept in its record and the next pair follows (exit 1); a changed
    listed file, a truth check that fails, a field that is not BL1's, or an open the spy refuses stops everything (exit 4)."""
    from .holdout_tet24 import FieldMismatch
    out, spy = context["out"], context.get("spy")
    reeval = admission.record.get("reeval")
    rows = [r for r in admission.pairs if reeval is None or r[3] in reeval["pairs"]]
    done, failed, stopped = [], [], None
    try:
        for row in rows:
            path = out/"pairs"/f"{row[3]}.json"
            record = dict(schema=SCHEMA_PAIR, pair=row[3], index=row[0], fixed=row[1], moving=row[2], complete=False, arms={})
            try:
                evaluate_pair(admission, context, row, record)
            except (ManifestChanged, TruthMismatch, SpyRefused, FieldMismatch) as exc:
                stopped = f"{type(exc).__name__}: {exc}"
            except HoldoutRefused as exc:
                if exc.stage == "once":
                    raise
                record.update(complete=False, error=f"{type(exc).__name__}: {exc}", utc=H.utc())
                failed.append(row[3])
            except Exception as exc:                                   # noqa: BLE001 - evidence kept; a reevaluation needs an authorization (D109)
                import traceback
                record.update(complete=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(), utc=H.utc())
                failed.append(row[3])
            if stopped is None and spy is not None and spy.refused:
                stopped = f"SpyRefused: {spy.refused}"
            if stopped is not None:
                H.write_once(path, dict(record, complete=False, error=stopped, utc=H.utc()))
                H.write_once(out/"stopped.json", dict(pair=row[3], reason=stopped, utc=H.utc()))
                break
            H.write_once(path, record)
            if record.get("complete"):
                done.append(row[3])
    finally:
        if spy is not None and spy is _SPY:
            stop_spy()
    opened = None if spy is None else spy.record()
    labels_opened = [] if opened is None else sorted({e["path"] for e in opened["events"] if Path(e["path"]).name == "label.nii.gz"})
    summary = dict(schema=SCHEMA_RUN, record=admission.record, evaluated=done, failed=failed, stopped=stopped, not_reached=[r[3] for r in rows if r[3] not in done+failed],
                   arm_table=context["table"], opened_files=opened, label_files_opened=labels_opened, truth_mode=admission.mode == "lpba40_test", ended_utc=H.utc())
    H.write_once(out/"lpba_eval.json", summary)
    return 4 if stopped else (1 if failed else 0)


# ---- the command line -----------------------------------------------------------------------------------------------------------------------------------------------
def launch_line(pair_set, out_dir, *, code=None, log=None):
    """The start string of the evaluation unit of `pair_set` (OWN_UNITS: tag-lp-eval, tag-lp-feas-eval) for `evaluate --out-dir out_dir`, in hp_core.launch_line's form (--wait, the
    log appended; default <RESULTS>/<unit>_unit.log): UNIT_SPEC -- MemoryMax 10G, MemorySwapMax 4G, OOMPolicy continue, CPUQuota 800%, RuntimeMaxSec 259200 (72 h), nice 5, one BLAS
    thread, NUMBA 2, no GPU, PYTHONPATH the code tree (the hold-out's tag-hb-eval, decision_log line 465).  Nothing is started here (LP-C2 integration review minor 3)."""
    check_pair_set(pair_set)
    unit = OWN_UNITS[pair_set][0]
    code = str(REPO if code is None else code)
    log = Path(RESULTS)/f"{unit}_unit.log" if log is None else log
    spec = UNIT_SPEC
    env = dict(OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, NUMBA_NUM_THREADS=spec["numba_threads"], CUDA_VISIBLE_DEVICES="",
               PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=code)
    props = [f"--property=MemoryMax={spec['memory_max']}", f"--property=MemorySwapMax={spec['memory_swap_max']}", f"--property=OOMPolicy={spec['oom_policy']}",
             f"--property=CPUQuota={spec['cpu_quota']}", f"--property=RuntimeMaxSec={spec['runtime_s']}", f"--property=StandardOutput=append:{log}",
             f"--property=StandardError=append:{log}"]
    return " ".join(["systemd-run", "--user", f"--unit={unit}", "--wait", "--collect", f"--working-directory={code}", f"--nice={spec['nice']}", *props,
                     *[f"--setenv={k}={v}" for k, v in env.items()], PY, "-B", "-m", "bayesreg.eval.lpba_eval", "evaluate", "--pair-set", pair_set, "--out-dir", str(out_dir)])


def dry_run(out_dir, pair_set="lpba40_test", *, repo=None):
    """Each admission condition on its own, reading only code-tree files (no subject file, no stage, no field): the first refusal is what `evaluate` would stop at.  Writes
    <out>/dry_run.json."""
    out = Path(out_dir)
    if out.exists():
        raise SystemExit(f"{out} exists")
    rootdir = Path(REPO if repo is None else repo)
    report = dict(schema="bayesreg.lpba40.eval.dry_run.v1", utc=H.utc(), pair_set=pair_set, conditions={})
    steps = [("code identity (C26.9)", lambda: check_code_identity(pair_set, rootdir)), ("admission (lpba_common.admit: pins, names, check_subject)", lambda: _admit(pair_set)),
             ("preregistration committed", lambda: check_prereg(rootdir)), ("arm table committed and valid", lambda: load_arm_table(pair_set, rootdir))]
    if pair_set == "lpba40_test":
        steps.append(("step-10 manifest committed", lambda: H.read_committed(MANIFEST_REL, "manifest", rootdir)))
    first = None
    for label, step in steps:
        try:
            step()
            report["conditions"][label] = "passed"
        except (HoldoutRefused, SubjectSealed, SealedListError, OSError, ValueError) as exc:
            report["conditions"][label] = f"refused: {type(exc).__name__}: {exc}"
            first = first or label
    report.update(first_refusal=first, truth_read=False)
    out.mkdir(parents=True)
    H.write_once(out/"dry_run.json", report)
    for label, text in report["conditions"].items():
        print(f"{label}: {text}")
    print(f"first refusal: {first}; no subject file was opened")
    return 0


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog="python -m bayesreg.eval.lpba_eval", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--pair-set", required=True, choices=PAIR_SETS)
    e.add_argument("--out-dir", required=True, type=Path, help="a new directory under <the mode's root>/eval/")
    e.add_argument("--reeval", type=Path, help="a committed reevaluation authorization (D109; the formal mode only)")
    d = sub.add_parser("dry-run")
    d.add_argument("--pair-set", required=True, choices=PAIR_SETS)
    d.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "dry-run":
        return dry_run(args.out_dir, args.pair_set)
    try:
        admission, context = open_lpba(args.out_dir, args.pair_set, reeval_path=args.reeval)
    except SubjectSealed as exc:
        print(f"subject_sealed: {exc}; nothing was read", file=sys.stderr)
        return 3
    except HoldoutRefused as exc:
        print(f"refused at {exc.stage}: {exc}; no truth was read", file=sys.stderr)
        return 2
    except (SealedListError, OSError, ValueError) as exc:
        print(f"refused: {type(exc).__name__}: {exc}; no truth was read", file=sys.stderr)
        return 2
    return evaluate_lpba(admission, context)


__all__ = ["Admission", "DATA_ROOT", "LPBA40_IDS", "LPBA54_IDS", "LR_PAIRS", "SCHEME", "check_cells", "check_manifest", "classify_cell", "evaluate_lpba", "evaluate_pair", "field_metrics",
           "id_record", "launch_line", "left_right_record", "load_arm_table", "load_reeval", "open_lpba", "parse_manifest", "read_label", "relocate", "rule_one", "run_lines", "score_d",
           "spy_roots", "start_spy", "stop_spy", "tet24"]


if __name__ == "__main__":
    sys.exit(main())
