"""HI (runs/20261107-hi/任务书.md): the structure-layer inference of unet35_v4_ens for subjects of the B-line hold-out, only after BZ unseals them, and the development check that decides between the entry's two backends.
Nothing here runs on the hold-out by itself: the entry refuses without BZ's unseal record (none exists; its format is in runs/20261107-hi/入口说明.md), and the allowlist does not hold a hold-out subject today.  The decision of the
development check is filled (DEVCHECK_DECISION: the check of 2026-11-07, 逐位相同 20/20：选 hs_a; branch hi-decide).  Revision 1 follows docs/bayesreg/reviews/HI_工具_复核_20261107.md (I1-I5, m1-m13); revision 2 the
diff review HI_工具_差异复核_20261107.md (R1, R2, n1-n5); then diff reviews 2 (s1-s4) and 3 (t1-t3).

Contract: C22.1-2 revision 4, the items 登记, 成员, 上游推断的规格 and 留出集 (BZ 2026-11-07); C22.9-4 (xv) "留出集的 v4 推断入口"; project rules, hard rules 4 (no rerun after a failure), 6 (the hold-out is never used for
development, BZ unseals, once per subject) and 7 (the label firewall); docs/bayesreg/04_分割器.md §1-5 (the inference guard: bayesreg.seg.unet.infer.check_subject_admitted on the pinned allowlist,
bayesreg.seg.unet.pins.load_pinned_allowlist).

The two backends (C22.1-2 revision 4 留出集: hs_seg's path (a), or infer.segment_subject with the v4 registry; chosen after a bitwise check against the registered probs36 of the 20 development subjects):
  hs_a             experiments/hs/hs_seg.py (a), the code that wrote seg/unet35_v4_ens/pred (HS's seg phase, a73c01c; hs_seg.py and fast_channels.py have no diff since; the SG package (C23.8-1) added the sealed
                   admission to infer.check_subject_admitted, data.OasisFiles.image_path / label_path, StructureLayer, synthetic.write_pair and nx_seg.read_labels_file -- admission only, no arithmetic:
                   tests/test_bayesreg/hi/test_hs_a_bitwise.py runs hs_a on synthetic members against c027447's code, bit for bit): hs_seg.uf_plan (the
                   same guard again, then D24: all_folds_mean over every registered model, the members found by sha256, A0 A1 B0 B1), hs_seg.uf4_probabilities (infer.load_member re-hashing each member.pt, the members'
                   own percentile, amp, sliding Gaussian window, temperature; ensemble_probabilities) and hs_seg.write_prediction (probs36 float16, structure_groups, the Hx line's reader on the stored file, provenance.json
                   last, never overwritten).  Its member records carry checkpoint_choice, checkpoint_iteration, amp (as used: predict_logits uses AMP only on cuda), window and percentile (上游推断的规格).
  segment_subject  bayesreg.seg.unet.infer.segment_subject(segmenter_id='unet35_v4_ens') on the formal registry: the same functions in the same order and probs.astype(float16).  hs_seg (a) stores
                   float16_keeping_argmax(probs, argmax of probs.astype(float16)), which touches no voxel (the labels are that array's own argmax, ties to the smallest label), so the two store the same bits by
                   construction (the tests: bitwise on synthetic members, CPU).  Its provenance records best_iteration (04_分割器.md §16 未决 2) and no checkpoint_choice, checkpoint_iteration, amp or window per member,
                   so it falls short of 上游推断的规格 without a change to bayesreg/, which this package does not make.  It is a backend of the development check only; the hold-out entry runs hs_a and refuses the other.
The choice is the lead's, after the development check, read by the rule written before it runs (runs/20261107-hi/devcheck_plan.md, pinned by DEVCHECK_PLAN_SHA256), and is written into DEVCHECK_DECISION by a reviewed
commit; the entry refuses while it is None, on another device than cuda, and in an environment other than the check's (review I3).

The order of the hold-out entry (infer_holdout); no file of a subject is opened before step 11, and the image of the first subject only after every subject of the run passed:
  1. the backend is hs_a; the code is the pinned commit's (rule one, pinned_code_check: P1 from the committed runs/holdout_B_v1/pinned_code.json, contract C24.3; A5a), and the run's code record
     names P1 (the lock line's code_commit);
  2. BZ's unseal record (load_unseal, through bayesreg's one loader bayesreg.io.unseal.load_unseal, contract C23.3-2): the pinned path splits/bayesreg_v1/unseal_holdout_B_v1.json inside the code tree, committed
     and unchanged against HEAD (git, literal pathspecs) before a byte of it is read, schema bayesreg.hi.unseal.v1, written_by "BZ" or "lead on BZ's behalf" (then bz_words holds "解封记录你代我写", D98), its
     list a sealed list that may be unsealed (UNSEALABLE: holdout_B_v1 only; H1-H4 never), the list file and its subjects' sha256 those pinned in bayesreg.io.sealed, the pinned allowlist's sha256 and the source
     that admits them named, never_for_training, holding exactly them and nobody else's subjects, and one line of the committed docs/decision_log.md holding the record's sha256 and bz_words; then the anchor's
     segmenter and registry sha256 (HI's own).  No record -> HIRefused (stage unseal): no file of any subject is read.  Then the pinned allowlist and the OasisFiles are bound to the record's unseal object
     (bind: bayesreg.io.unseal.bind_unseal, bind_files; C23.8-1); only the bound ones are used below;
  3. run once (review I4, diff review R2, n5): the fixed lock HOLDOUT_LOCK must not exist; a run after a failure needs an explicit, committed resume record (load_resume) naming the latest run of the lock, whose
     process is gone, for exactly the subjects no run of the lock finished (all runs taken together, each run's records consistent); a failed run is resumed at most once.  A5a review I2 (the lead's
     ruling of 2026-11-10 ~12:05 PM CST (5)): the one root HOLDOUT_OUT_SUGGESTED serves the first run and its resume (the resume lock is named after it, so one resume in all); a subject is written when
     it has subjects/<s>.json (finished_subjects); the resume checks the written subjects' files against their records (written_subject_problems) and moves what a killed run left in pred/ of the
     others to attempt<k>/ whole (keep_residue, after the lock); its own records are hi_*_resume_<k>.json (run_names); HI is complete when the records of both runs are every subject (complete_subjects);
  4. the decision of the development check (check_decision: DEVCHECK_DECISION not None, the reading "逐位相同 20/20：选 hs_a", the devcheck.json at DEVCHECK_OUT with its sha256, made under the pinned plan on the
     20 development subjects);
  5. the registry against the anchor;
  6. every subject: infer.check_subject_admitted(subject, allowlist) with the bound allowlist (bayesreg's sealed guard, C23.4: SubjectSealed off the allowlist, not canonical, or on a sealed list without this
     record's unseal object), the allowlist carrying this record's unseal object, one of the record's subjects, the sealed lists (bayesreg.eval.truth.load_sealed_lists, HI's second lock: only the record's list,
     H1-H4 refused), the allowlist sources exactly the record's (T0, the training set, the development stages refused);
  7. D24 (hs_seg.uf_plan -> infer.select_fold_models: all_folds_mean; a training subject is refused even if it got this far), the members re-hashed;
  8. the output directory: HOLDOUT_OUT_SUGGESTED only (checked first of all, A5a), under /home/rmcfz/seq_vp_tag_results, not in the code tree (review m6); new for the first run, there for a resume;
  9. the machine gate (no other active tag-* or seq-vp-* unit, no GPU compute process, utilization <= 20 %; review m4), then the device (cuda only, free memory, the process capped) and its environment against the decision;
  10. the lock taken with O_EXCL;
  11. subject after subject: the image (aligned_norm.nii.gz only), the prediction, a record (with provenance.json's sha256 and the run); the run's final record at the end, or its stopped record on a
      failure (kept, not rerun; run_names).
The firewall: only aligned_norm.nii.gz of each subject is read (bayesreg.seg.unet.data.read_image); no label file, no pair.npz, no evaluation record.

The development check (devcheck): the 20 subjects of stage_L2R_val_v1 (experiments/unv2/eval_seg.admit_subjects, as run_hs.seg_inputs), the plan's sha256, the same guard on each, the registered predictions' provenance
checked before any image (this segmenter, this registry file, these members in this order, all_folds_mean, test_labels_seen false), then per backend and subject the stored float16 arrays (hs_a in memory with the
writer's own transform; segment_subject's files as it writes them) against the registered files after their hashes are checked: bitwise equal, the elements that differ, the largest difference, the argmax maps and their
agreement; the same for structure_groups; the reading of the plan.  hs_a on all 20 first, segment_subject after it (diff review n3).  No truth label; the output in a new directory (DEVCHECK_OUT), never overwritten;
once (diff review R1: the fixed lock DEVCHECK_LOCK, O_EXCL), only into DEVCHECK_OUT, and after the machine gate with tag-hi-devcheck as its own unit (diff review 2, s2).

What the code cannot tell: whether BZ wrote the unseal record.  Every commit is made by the same git user and anyone can write "written_by": "BZ"; "only BZ unseals" rests on the process (BZ's words in the decision
log, the record's sha256 written there, the commit reviewed independently).  runs/20261107-hi/入口说明.md §3, §4.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import MappingProxyType

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REPO = EXPERIMENTS.parent
for _p in (EXPERIMENTS/'hs', EXPERIMENTS/'unv2'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))                  # last, as mc0_core does; the units run with PYTHONPATH=<code tree>

import hs_seg  # noqa: E402
import hs_core as hc  # noqa: E402
from bayesreg.eval.truth import SEALED_LISTS, SealedListError, SealedLists  # noqa: E402
from bayesreg.io import unseal as bayes_unseal  # noqa: E402
from bayesreg.io.guard import DevAllowlist, SubjectSealed, canonical_subject_list, is_hex64, strict_json_object  # noqa: E402
from bayesreg.io.pair import SegmenterRegistry, subject_list_sha256, verify_segmentation_files  # noqa: E402
from jd_curl.three_d.nifti import file_sha256  # noqa: E402
from jd_curl.validation import InputValidationError  # noqa: E402

# ---- constants ----------------------------------------------------------------------------------------------------------------------------------
CONTRACT_SEGMENTER = MappingProxyType(dict(segmenter_id='unet35_v4_ens', policy='all_folds_mean', registry='configs/bayesreg/segmenters_hs_v1.json',
                                           registry_sha256='b7682421bf6e8d07c606ad55cd47db7936a293c5e79d00839c82e3f8c2edefa1'))     # C22.1-2 修订 4 (登记); written here, not taken from he_core or from any setting
CANDIDATE = 'UF4'                               # hs_core.CANDIDATES key of (a): segmenter_id unet35_v4_ens, no nnU-Net model
RESULTS = Path('/home/rmcfz/seq_vp_tag_results')    # every output of this entry is under it (project rules 运行结果写到); read at call time
MODELS_ROOT = RESULTS/'seg'/'unet35_v4_full'    # the four UF members (experiments/p2/reference_unet35_v4_ens.json channels.models_root); only read
REGISTERED_PRED = RESULTS/'seg'/'unet35_v4_ens'/'pred'     # the registered predictions of the 20 development subjects (HS's seg phase); only read
DATA_ROOT = Path('/mnt/d/data')                 # <subject>/aligned_norm.nii.gz (04_分割器.md §1-3); only read
DEVCHECK_OUT = RESULTS/'seg'/'hi_devcheck_v1'   # the brief: a new directory, never overwritten
HOLDOUT_OUT_SUGGESTED = RESULTS/'seg'/'unet35_v4_ens_holdout_B_v1'      # the one --out of the hold-out entry (A5a; prereg revision 2 list 5 (l); he_core.HOLDOUT_SOURCES, xb_core.HI_PRED_ROOTS read its pred/): run_hi refuses any other
HOLDOUT_LOCK = Path('/home/rmcfz/seq_vp_tag_results/seg/hi_holdout_infer.lock')    # review I4: one fixed path, not derived from any option; created with O_EXCL, never edited or deleted
DEVCHECK_LOCK = Path('/home/rmcfz/seq_vp_tag_results/seg/hi_devcheck.lock')         # diff review R1: the development check runs once (as tta_dev_read.lock); O_EXCL, never edited or deleted
MAIN_REPO = Path('/home/rmcfz/code/python_projects/seq_vp_tag')     # data/ is an untracked link of the main checkout (the allowlist's stage lists): the source root of load_pinned_allowlist; a worktree has none
DEV_SOURCE = 'stage_L2R_val_v1'
DEV_SUBJECTS = tuple(f'OASIS_OAS1_{n:04d}_MR1' for n in range(438, 458))      # the 20 subjects of the 19 development pairs
UNSEAL_REL = bayes_unseal.UNSEAL_REL           # review m7: the one path of BZ's record (C23.3-1: defined in bayesreg.io.unseal)
DECISION_LOG_REL = bayes_unseal.DECISION_LOG_REL    # review m7: the record's sha256 must be written there (committed), on the line of BZ's words (D98)
UNSEAL_SCHEMA = bayes_unseal.UNSEAL_SCHEMA
UNSEAL_KEYS = bayes_unseal.UNSEAL_KEYS
UNSEALABLE = bayes_unseal.UNSEALABLE            # C23.3-2 (v): defined once, in bayesreg.io.unseal (holdout_B_v1 only; H1-H4 never); the same object
RESUME_SCHEMA = 'bayesreg.hi.resume.v1'
RESUME_DIR_REL = 'splits/bayesreg_v1'
RESUME_NAME = re.compile(r'unseal_holdout_B_v1_resume_[0-9]{1,3}\.json', re.ASCII)
RESUME_KEYS = ('schema', 'written_by', 'words', 'time_local', 'utc', 'reason', 'unseal_sha256', 'lock_sha256', 'previous_out', 'stopped_sha256', 'subjects_sha256', 'n_subjects')
RESUME_WRITERS = ('BZ', 'lead')                 # review I4: "由总负责人或 BZ 另给明确的授权"
EARLIEST_UTC = datetime(2026, 10, 30, tzinfo=timezone.utc)      # the draw of the B-line hold-out (splits/bayesreg_v1, 044426f): no record is older
UTC_FORMAT = '%Y-%m-%dT%H:%M:%SZ'
UTC_FORM = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z', re.ASCII)
BACKENDS = ('hs_a', 'segment_subject')
PRODUCTION_BACKEND = 'hs_a'
GPU_CAP_GIB = hs_seg.GPU_CAP_GIB                # 20: HS's seg phase capped itself there (hs_seg.infer_all)
GPU_FREE_MIN_GIB = 21.                          # run_hs.SEG_GPU_FREE_MIB: free device memory before a cuda run starts
GATE_UNIT_PATTERNS = ('seq-vp-*', 'tag-*')      # review m4: any other active unit of either line stops the hold-out run before it starts
OWN_UNITS = ('tag-hi-infer',)                   # the entry's own unit (run_hi.py's start string); excluded from the gate only when this process runs in it (diff review n1)
DEVCHECK_UNITS = ('tag-hi-devcheck',)           # the development check's own unit (diff review 2, s2), the same rule
PROC_CGROUP = Path('/proc/self/cgroup')
PROC = Path('/proc')                            # diff review 3, t1: a process is told by (boot_id, pid, start ticks): /proc/sys/kernel/random/boot_id and /proc/<pid>/stat field 22, no wall clock
SMI = Path('/usr/lib/wsl/lib/nvidia-smi')       # bayesreg.seg.unet.cli.SMI
GPU_UTIL_MAX = 20                               # bayesreg.seg.unet.cli.GPU_UTIL_MAX
SLAB = 8
DEVCHECK_PLAN_REL = 'runs/20261107-hi/devcheck_plan.md'
DEVCHECK_PLAN_SHA256 = '168606db18fa526cf60e7252b0bb42a9a72b33b04819947b836e8415e32b46ba'     # review I3: the reading is fixed before the run; another plan needs this constant changed and reviewed
READ_SELECT, READ_PENDING, READ_UNDECIDED = '逐位相同 {n}/{n}：选 hs_a', '不过（待查）', '不判'          # devcheck_plan.md §3
# Review I3: filled by a reviewed commit after the development check and the lead's choice (devcheck_plan.md §4), with exactly DECISION_KEYS (None -> infer refuses).  Filled on branch hi-decide from the check
# tag-hi-devcheck, 7:22:34-7:50:10 AM CST 2026-11-07, code 1b1321f, reading 逐位相同 20/20：选 hs_a (runs/20261107-hi/devcheck_result.md, main 86db1ba); the values copied from that devcheck.json, read only
# (its lock /home/rmcfz/seq_vp_tag_results/seg/hi_devcheck.lock has sha256 7895bf34002aade2a2bdc8c9d77da7420964112b7b388ce6382d18428f7fa3d8; check_decision compares the lock line itself).
DEVCHECK_DECISION = MappingProxyType(dict(
    devcheck_json='/home/rmcfz/seq_vp_tag_results/seg/hi_devcheck_v1/devcheck.json',
    devcheck_sha256='0deba0f8f44b98fb9f255a843d8ba3d09e2bc5f452f9ef724c0b3f45ef19b4e4',
    reading='逐位相同 20/20：选 hs_a',
    backend='hs_a',
    environment=MappingProxyType(dict(gpu='NVIDIA GeForce RTX 4090', torch='2.5.1', cuda='12.1', cudnn=90100, cudnn_benchmark=False, cudnn_deterministic=False)),
    decided_by='lead (total responsible), 2026-11-07, per devcheck_plan.md §3',
    review='docs/bayesreg/reviews/HI_决定_差异复核_20261107.md (the independent diff review of branch hi-decide d0cf87b, afd44a5: blocking 0, important 0, minor 2; can merge)'))
DECISION_KEYS = ('devcheck_json', 'devcheck_sha256', 'reading', 'backend', 'environment', 'decided_by', 'review')
DECISION_ENVIRONMENT_KEYS = ('gpu', 'torch', 'cuda', 'cudnn', 'cudnn_benchmark', 'cudnn_deterministic')
HI_PREDICTION_SCHEMA, HI_INFER_SCHEMA, DEVCHECK_SCHEMA = 'bayesreg.hi.prediction.v1', 'bayesreg.hi.infer.v1', 'bayesreg.hi.devcheck.v1'
INFER_RECORD, STARTED_RECORD, STOPPED_RECORD, DEVCHECK_RECORD = 'hi_record.json', 'hi_started.json', 'hi_stopped.json', 'devcheck.json'
FIRST_RUN, AUTHORIZED_RESUME = 'first_run', 'authorized_resume'
COMPOSITION = dict(rule='(a) UF4: the mean over the four members of softmax(z_m / T_m) (ensemble_probabilities), float16 as v2 stores probs36', weights=dict(UF4=1.))     # hs_seg.predict_subject's (a), word for word
REGISTRY_NOTE = ('the formal registry configs/bayesreg/segmenters_hs_v1.json (C22.1-2 修订 4, 登记), its sha256 checked against the anchor CONTRACT_SEGMENTER of experiments/hi/hi_core.py before any image')
AMP_RULE = 'amp: the member.pt flag as used, i.e. true only on cuda (bayesreg.seg.unet.train.predict_logits: use_amp = amp and device_type == "cuda")'


class HIRefused(Exception):
    """The entry refuses before any image of the subject (or of the run) is read; `stage` says which check.  Status hi_refused."""
    status = 'hi_refused'

    def __init__(self, stage, message):
        super().__init__(f'{stage}: {message}')
        self.stage = stage


def utc():
    return time.strftime(UTC_FORMAT, time.gmtime())


def write_json_once(path, document):
    """hs_seg.write_json_once: a file that is there is never overwritten (FileExistsError)."""
    return hs_seg.write_json_once(path, document)


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (tuple, set, frozenset)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'not JSON serializable: {type(value)}')


def plain(document):
    """The document as JSON gives it back (numpy scalars, tuples, paths and mapping proxies made plain), so that hs_seg's writers can dump it."""
    return json.loads(json.dumps(document, default=lambda v: dict(v) if isinstance(v, MappingProxyType) else json_default(v)))


def check_out_location(path, stage='output', *, resume=False):
    """Review m6: an output directory under RESULTS (read at call time), outside the code tree.  A first run: not there yet (FileExistsError: never overwritten).  A resume (A5a review I2, the
    lead's ruling of 2026-11-10 ~12:05 PM CST (5)): the one root of the hold-out entry, there and a directory -- the resume writes the subjects left into the same root (resume_layout), never over a
    file.  Nothing is created here."""
    target = Path(path).resolve()
    if not target.is_relative_to(Path(RESULTS).resolve()) or target.is_relative_to(REPO.resolve()):
        raise HIRefused(stage, f'{path}: an output directory must be under {RESULTS} and outside the code tree {REPO}')
    if resume:
        if not target.is_dir():
            raise HIRefused('resume', f'{path} is not there: a resume writes into the root of the run it resumes')
        return target
    if target.exists():
        raise FileExistsError(f'{path} exists: the first run writes a new root; after a failure only a committed resume record writes into it (the subjects left)')
    return target


def holdout_out_problem(out_root):
    """A5a (prereg revision 2 list 5 (l); A5a review I2 point 1): the hold-out entry writes only into HOLDOUT_OUT_SUGGESTED (read at call time; he_core.HOLDOUT_SOURCES and xb_core.HI_PRED_ROOTS
    read its pred/), the first run and a resume alike.  None, or why not (compared resolved)."""
    if Path(out_root).resolve() == Path(HOLDOUT_OUT_SUGGESTED).resolve():
        return None
    return f'{out_root} is not {HOLDOUT_OUT_SUGGESTED}: the hold-out entry writes only there (he_core.HOLDOUT_SOURCES and xb_core.HI_PRED_ROOTS read its pred/)'


def run_names(k=0):
    """A5a review I2 point 3: the records of a run in the one root, never overwritten -- the first run (k = 0) hi_started.json, hi_stopped.json, hi_record.json; the k-th resume the same names with
    _resume_<k> (hi_started_resume_1.json, ...).  The subjects' records (subjects/<s>.json) and predictions (pred/<s>/) of every run share the root."""
    if k == 0:
        return dict(started=STARTED_RECORD, stopped=STOPPED_RECORD, record=INFER_RECORD)
    return {key: name.replace('.json', f'_resume_{k}.json') for key, name in (('started', STARTED_RECORD), ('stopped', STOPPED_RECORD), ('record', INFER_RECORD))}


RUN_FILE = re.compile(r'hi_(started|stopped|record)(_resume_[0-9]{1,3})?\.json', re.ASCII)
RESIDUE_DIR = 'attempt{k}'                       # A5a review I2 point 2: what a killed run left in pred/ of the subjects it did not finish, moved out of pred/ before the k-th resume (kept whole)


def run_files(out):
    """The run records of a root (RUN_FILE: the first run's and every resume's started, stopped and final record), sorted: what a reader of HI's runs may open beside subjects/ and pred/ (PQH's list)."""
    out = Path(out)
    return sorted(p for p in out.iterdir() if p.is_file() and RUN_FILE.fullmatch(p.name)) if out.is_dir() else []


def written_subject_problems(out):
    """A5a review I2 point 2: every subject with a record subjects/<s>.json is written: its prediction pred/<s>/ holds each output file the record names with that sha256 (output_sha256), and
    provenance.json (with the record's provenance_sha256 when it has one).  Hashes the files; [] or the problems."""
    out, problems = Path(out), []
    for subject in sorted(subject_records(out)):
        path = out/'subjects'/f'{subject}.json'
        try:
            row = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            problems.append(f'{path}: {type(exc).__name__}: {exc}')
            continue
        outputs = row.get('output_sha256') if isinstance(row, dict) else None
        if not isinstance(outputs, dict) or not outputs:
            problems.append(f'{path}: no output_sha256')
            continue
        pred = out/'pred'/subject
        for name, want in sorted(outputs.items()):
            got = file_sha256(pred/name) if (pred/name).is_file() else None
            if got != want:
                problems.append(f'{pred/name}: sha256 {got} is not its record\'s {want}')
        provenance = pred/hs_seg.PROVENANCE_FILE
        if not provenance.is_file():
            problems.append(f'{provenance} is missing')
        elif row.get('provenance_sha256') is not None and file_sha256(provenance) != row['provenance_sha256']:
            problems.append(f'{provenance}: sha256 is not its record\'s {row["provenance_sha256"]}')
    return problems


def residue_subjects(out):
    """A5a review I2 point 2: the subjects whose directory is in pred/ but who have no record subjects/<s>.json (a killed run's half or whole prediction): not written; moved out before a resume."""
    pred = Path(out)/'pred'
    return sorted(p.name for p in pred.iterdir() if p.is_dir() and p.name not in subject_records(out)) if pred.is_dir() else []


def keep_residue(out, subjects, k):
    """A5a review I2 point 2 (the lead's ruling (5): the half files kept whole): after the resume's lock, a sha256 list of every file of pred/<s>/ of `subjects` written once to
    <root>/attempt<k>/residue.json, then each directory moved whole to <root>/attempt<k>/pred/<s>/ (a rename on one file system), and the sha256 of every moved file checked again (residue_moved.json).
    Returns the record; RuntimeError when a moved file's sha256 differs (both lists kept)."""
    out = Path(out)
    base = out/RESIDUE_DIR.format(k=k)
    rows = []
    for subject in subjects:
        src = out/'pred'/subject
        rows += [dict(subject=subject, file=str(p.relative_to(src)), bytes=p.stat().st_size, sha256=file_sha256(p)) for p in sorted(src.rglob('*')) if p.is_file()]
    base.mkdir(parents=True, exist_ok=True)
    write_json_once(base/'residue.json', dict(schema=HI_INFER_SCHEMA+'.residue', subjects=list(subjects), files=rows, utc=utc(),
                                              rule='A5a review I2: what a killed run left in pred/ of the subjects it did not finish (no subjects/<s>.json), kept whole, moved out of pred/ before the resume'))
    (base/'pred').mkdir(exist_ok=True)
    for subject in subjects:
        os.rename(out/'pred'/subject, base/'pred'/subject)
    after = [dict(r, sha256_after=file_sha256(base/'pred'/r['subject']/r['file'])) for r in rows]
    bad = [r for r in after if r['sha256_after'] != r['sha256']]
    write_json_once(base/'residue_moved.json', dict(files=after, passed=not bad, utc=utc()))
    if bad:
        raise RuntimeError(f'the residue moved to {base}/pred changed: {bad[:3]}')
    return dict(dir=str(base), subjects=list(subjects), n_files=len(rows), residue_sha256=file_sha256(base/'residue.json'))


def fresh_root(path):
    """The first run's output root, made here (it must not exist; FileExistsError otherwise)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'{path} exists: the first run writes a new root; nothing is overwritten (a resume writes the subjects left into it, check_out_location)')
    path.mkdir(parents=True)
    return path


def pinned_code_check():
    """Rule one on this entry's code tree (contract C24.3; decision_log 2026-11-09 df8541d, 2026-11-10 ~2:28 AM CST n8, ~2:59 AM CST (1)): bayesreg.io.code_identity.unit_identity_problems -- P1
    from the committed runs/holdout_B_v1/pinned_code.json at HEAD, never HEAD by default; every commit of P1..HEAD only adds authorizations and records, appends to the decision log, changes the
    allowlist once with the unseal record; the code unchanged; a clean working tree.  (record, problems).  Read at call time (the tests replace it; its own tests run on temporary repositories)."""
    from bayesreg.io import code_identity
    return code_identity.unit_identity_problems(repo=REPO)


def pinned_code(code):
    """The hold-out entry's code record after rule one (pinned_code_check, read at call time): HIRefused (stage code) on a problem -- nothing of the hold-out read yet; otherwise the record with
    `head` (HEAD), `commit` (P1: the lock line's code_commit) and `pinned_code` (the check)."""
    record, problems = pinned_code_check()
    if problems:
        raise HIRefused('code', f'rule one (the pinned commit of runs/holdout_B_v1/pinned_code.json): {problems}')
    code = dict(code or {})
    return dict(code, head=code.get('commit'), commit=record.get('pinned'), pinned_code=record)


def check_code_root(code, repo=None):
    """Review I1: the code record (snapshot.code_record, the tree bayesreg was imported from, whose cleanliness pins.load_pinned_allowlist checks) is the tree of this module, of hs_seg and hs_core, of the registry,
    the sealed lists and the unseal record, and bayesreg and jd_curl were imported from inside it.  CodeNotClean otherwise (nothing of the inputs read yet)."""
    from bayesreg.seg.unet.snapshot import CodeNotClean
    repo = Path(REPO if repo is None else repo).resolve()
    root = code.get('root') if isinstance(code, dict) else None
    problems = []
    if not root or Path(root).resolve() != repo:
        problems.append(f'the code record is of {root}, this entry is in {repo}')
    if not isinstance(code, dict) or code.get('imports_inside_root') is not True:
        problems.append(f"bayesreg and jd_curl are not imported from inside the code root ({(code or {}).get('imported_from') if isinstance(code, dict) else code})")
    for module in (hs_seg, hc):
        if not Path(module.__file__).resolve().is_relative_to(repo):
            problems.append(f'{module.__name__} is imported from {module.__file__}, outside {repo}')
    if problems:
        raise CodeNotClean(f'the clean-tree check would not cover the code that runs: {problems}')
    return dict(root=str(repo), checked='code record root = hi_core.REPO, imports inside it, hs_seg and hs_core from it (review I1)')


# ---- git: committed and unchanged ---------------------------------------------------------------------------------------------------------------
def _git(repo, *args):
    """git with literal pathspecs (review I2): a file name is never a pattern or a magic signature."""
    try:
        return subprocess.run(['git', '--literal-pathspecs', '-C', str(repo), *args], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(args, 127, '', f'{type(exc).__name__}: {exc}')


def tracked_unchanged(path, repo=None, *, stage='unseal'):
    """The file is inside the code tree, tracked by git and without any change against HEAD (not untracked, ignored, only staged or modified).  Reads nothing of the file itself.  HIRefused(stage) otherwise."""
    root = Path(REPO if repo is None else repo).resolve()
    target = Path(path).resolve()
    if not target.is_relative_to(root):
        raise HIRefused(stage, f'{path} is not inside the code tree {root}')
    rel = target.relative_to(root).as_posix()
    tracked = _git(root, 'ls-files', '--error-unmatch', '--', rel)
    changed = _git(root, 'status', '--porcelain', '--ignored', '--untracked-files=all', '--', rel)
    if tracked.returncode != 0 or changed.returncode != 0 or changed.stdout.strip() or tracked.stdout.strip() != rel:
        raise HIRefused(stage, f'{rel}: not a committed, unchanged file of the code tree (ls-files exit {tracked.returncode} {tracked.stdout.strip()!r}; status {changed.stdout.strip()!r})')
    last = _git(root, 'log', '-1', '--format=%H', '--', rel).stdout.strip()
    return dict(path=rel, tracked=True, unchanged=True, last_commit=last or None)


def unseal_tracked(path, repo=None):
    return tracked_unchanged(path, repo, stage='unseal')


def utc_problem(value, key):
    """None if `value` is a real UTC time in YYYY-MM-DDTHH:MM:SSZ between the draw of the hold-out and now (10 minutes of slack); the problem otherwise (review m8)."""
    if not (isinstance(value, str) and UTC_FORM.fullmatch(value)):
        return f'{key} {value!r}: YYYY-MM-DDTHH:MM:SSZ required'
    try:
        when = datetime.strptime(value, UTC_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return f'{key} {value!r}: not a real time'
    if not EARLIEST_UTC <= when <= datetime.now(timezone.utc)+timedelta(minutes=10):
        return f'{key} {value!r}: before the draw of the hold-out or in the future'
    return None


def _is_count(value):
    return type(value) is int and value >= 0


def _nonempty(value):
    return isinstance(value, str) and value.strip() != ''


def _strict_json(path, stage):
    raw = Path(path).read_bytes()
    try:
        document = json.loads(raw.decode('utf-8'), object_pairs_hook=strict_json_object)
    except (UnicodeDecodeError, ValueError) as exc:
        raise HIRefused(stage, f'{path}: not UTF-8 JSON without duplicate keys ({exc})') from None
    return raw, document


# ---- the registry and the members ---------------------------------------------------------------------------------------------------------------
def check_registry(registry, anchor=CONTRACT_SEGMENTER):
    """The registry is the anchor's: its file sha256 is the anchor's, every model is of the anchor's segmenter, and the segmenter is HS's candidate (a).  HIRefused otherwise (stage registry)."""
    if not isinstance(registry, SegmenterRegistry):
        raise HIRefused('registry', 'a SegmenterRegistry from load_segmenter_registry is required')
    problems = []
    if registry.sha256 != anchor['registry_sha256']:
        problems.append(f"sha256 {registry.sha256} is not the anchor's {anchor['registry_sha256']}")
    ids = sorted({e['segmenter_id'] for e in registry.models.values()})
    if ids != [anchor['segmenter_id']]:
        problems.append(f"the models are of {ids}, not only {anchor['segmenter_id']!r}")
    if hc.CANDIDATES[CANDIDATE]['segmenter_id'] != anchor['segmenter_id'] or hc.CANDIDATES[CANDIDATE]['nn_model'] is not None:
        problems.append(f"hs_core's candidate {CANDIDATE} is not {anchor['segmenter_id']} without an nnU-Net model")
    if anchor['policy'] != 'all_folds_mean':
        problems.append(f"the anchor's policy {anchor['policy']!r} is not all_folds_mean")
    if problems:
        raise HIRefused('registry', f'{registry.path}: {problems}')
    return dict(path=registry.path, sha256=registry.sha256, anchor=dict(anchor), models=sorted(registry.models),
                members={m: list(e['member_model_sha256']) for m, e in sorted(registry.models.items())}, rule='file sha256 = the anchor; every model of the anchor segmenter (C22.1-2 修订 4)')


def member_files(registry, models_root):
    """{sha256: member.pt} of every registered member under models_root (bayesreg.seg.unet.registry.find_member_files re-hashes each); HIRefused if one is missing."""
    from bayesreg.seg.unet.registry import find_member_files
    available = find_member_files(models_root)
    missing = sorted(sha[:12] for e in registry.models.values() for sha in e['member_model_sha256'] if sha not in available)
    if missing:
        raise HIRefused('members', f'registered members {missing} not found under {models_root}')
    return available


def plan_of(registry, subject, allowlist, available):
    """hs_seg.uf_plan: infer.check_subject_admitted again, then D24 (infer.select_fold_models: all_folds_mean over every registered model), the members by sha256 in HS's order A0 A1 B0 B1."""
    return hs_seg.uf_plan(registry, CANDIDATE, subject, allowlist, available)


# ---- BZ's unseal record -------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Unseal:
    """HI's view of BZ's unseal record (C23.8-1): made only by load_unseal, around the unseal object of bayesreg's one loader (`base`, bayesreg.io.unseal.load_unseal); the guard takes `base`, bound to the allowlist."""
    path: str
    sha256: str
    sealed_list: str
    subjects: tuple
    allowlist_source: str
    allowlist_sha256: str
    document: MappingProxyType
    git: MappingProxyType
    base: object

    def record(self):
        return dict(path=self.path, sha256=self.sha256, sealed_list=self.sealed_list, n_subjects=len(self.subjects), subjects_sha256=subject_list_sha256(list(self.subjects)),
                    allowlist_source=self.allowlist_source, allowlist_sha256=self.allowlist_sha256, written_by=self.document.get('written_by'), bz_words=self.document.get('bz_words'),
                    time_local=self.document.get('time_local'), utc=self.document.get('utc'), purpose=self.document.get('purpose'), git=plain(dict(self.git)))


def load_unseal(path, *, anchor=CONTRACT_SEGMENTER):
    """BZ's unseal record (runs/20261107-hi/入口说明.md §3), through bayesreg's one loader (bayesreg.io.unseal.load_unseal, C23.3-2 (i)-(ix): the pinned path in bayesreg's code tree, committed and unchanged
    before a byte is read, the fields, written_by "BZ" or "lead on BZ's behalf" with BZ's words (D98), the sealed list, the pinned allowlist and its source, the decision log); then HI's own check on top of it:
    the record's segmenter_id and registry_sha256 are the anchor's.  Every refusal is HIRefused (stage unseal); nothing of a subject is read.  No second way to load the record is left here (C23.8-1): the tree,
    the sealed lists and the allowlist are bayesreg's pinned ones (the tests change them with monkeypatch)."""
    if path is None:
        raise HIRefused('unseal', 'no unseal record: the hold-out is sealed until BZ writes one (runs/20261107-hi/入口说明.md); no file of any subject is read')
    try:
        base = bayes_unseal.load_unseal(path)
    except (SubjectSealed, SealedListError, InputValidationError, OSError, ValueError) as exc:
        raise HIRefused('unseal', f'{path}: {type(exc).__name__}: {exc}') from None
    document = base.document
    if document['segmenter_id'] != anchor['segmenter_id'] or document['registry_sha256'] != anchor['registry_sha256']:
        raise HIRefused('unseal', f"{path}: segmenter {document['segmenter_id']!r} / registry {document['registry_sha256']!r} are not the anchor's ({anchor['segmenter_id']}, {anchor['registry_sha256']})")
    return Unseal(path=base.path, sha256=base.sha256, sealed_list=base.sealed_list, subjects=base.subjects, allowlist_source=base.allowlist_source, allowlist_sha256=base.allowlist_sha256,
                  document=base.document, git=base.git, base=base)


def bind(unseal, allowlist, files=None):
    """C23.8-1: the allowlist (pins.load_pinned_allowlist) and the OasisFiles bound to the record's unseal object (bayesreg.io.unseal.bind_unseal, bind_files), for admit, plan_of (hs_seg.uf_plan) and the
    image reads.  HIRefused (stage unseal) when they cannot be bound (another allowlist than the record's, an object not of the loader)."""
    if not isinstance(unseal, Unseal):
        raise HIRefused('unseal', 'no unseal record: the hold-out is sealed until BZ writes one; no file of any subject is read')
    try:
        bound = bayes_unseal.bind_unseal(allowlist, unseal.base)
        bound_files = None if files is None else bayes_unseal.bind_files(files, bound)
    except SubjectSealed as exc:
        raise HIRefused('unseal', f'the unseal record cannot be bound to the allowlist and the files: {exc}') from None
    return bound, bound_files


# ---- run once: the lock and the resume record ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Resume:
    path: str
    sha256: str
    previous_out: str
    subjects: tuple
    done: tuple
    lock_sha256: str
    lock_path: str                               # the resume lock of the run it resumes (once per failed run)
    document: MappingProxyType
    git: MappingProxyType

    def record(self):
        return dict(path=self.path, sha256=self.sha256, previous_out=self.previous_out, n_subjects=len(self.subjects), subjects_sha256=subject_list_sha256(list(self.subjects)), n_done_before=len(self.done),
                    lock_sha256=self.lock_sha256, resume_lock=self.lock_path, written_by=self.document.get('written_by'), words=self.document.get('words'), time_local=self.document.get('time_local'),
                    utc=self.document.get('utc'), reason=self.document.get('reason'), git=dict(self.git))


def lock_lines(path):
    """The JSON lines of a lock file (empty if there is none)."""
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.is_file() else []


def resume_lock_path(lock, previous_out):
    """Diff review R2: named after the run it resumes, so one failed run is resumed at most once, whatever record authorizes it."""
    lock = Path(lock)
    return lock.with_name(f'{lock.name}.resume_{hashlib.sha256(str(previous_out).encode("utf-8")).hexdigest()[:16]}')


def run_lines(lock):
    """Every run of the lock: the first run's line and the line of every authorized resume (each resume lock holds one)."""
    lock = Path(lock)
    runs = [dict(line, lock_file=str(lock)) for line in lock_lines(lock)]
    for other in sorted(lock.parent.glob(f'{lock.name}.resume_*')):
        runs += [dict(line, lock_file=str(other)) for line in lock_lines(other)]
    return [r for r in runs if r.get('out')]


def run_outs(lock):
    """The output roots of the lock's runs, each once, in the lock's order (A5a review I2: the first run and its resume share HOLDOUT_OUT_SUGGESTED, so a root is listed once)."""
    return list(dict.fromkeys(r['out'] for r in run_lines(lock)))


def latest_run(lock):
    """The run that no authorized resume names as its previous_out (the runs make one chain: first <- resume <- resume ...); None unless there is exactly one."""
    runs = run_lines(lock)
    resumed = {(r.get('resume') or {}).get('previous_out') for r in runs}
    last = [r for r in runs if r['out'] not in resumed]
    return last[0] if len(last) == 1 else None


def subject_records(out):
    """The subjects with a record subjects/<subject>.json in a run's output directory."""
    out = Path(out)
    return {p.name[:-5] for p in (out/'subjects').glob('*.json')} if (out/'subjects').is_dir() else set()


def predicted_subjects(out):
    """Diff review 2, s1: the subjects with a complete prediction in a run's output directory (pred/<subject>/provenance.json: hs_seg.write_prediction writes it last)."""
    pred = Path(out)/'pred'
    return {p.parent.name for p in pred.glob(f'*/{hs_seg.PROVENANCE_FILE}')} if pred.is_dir() else set()


def finished_subjects(lock):
    """What the runs of the lock wrote, taken together (A5a review I2 point 2; the lead's ruling (5)): the subjects with a record subjects/<s>.json in a run root -- the first run's and the resume's
    in the one root.  A prediction without its record (a run killed between the two) is not written: it is residue, moved out before the resume (keep_residue) and predicted again."""
    done = set()
    for out in run_outs(lock):
        done |= subject_records(out)
    return done


def complete_subjects(lock, subjects):
    """A5a review I2 point 3: HI's hold-out inference is complete when the records of the first run and the resume, taken together (finished_subjects), are every subject of `subjects` (the unseal
    record's).  (complete, missing)."""
    missing = sorted(set(subjects)-finished_subjects(lock))
    return not missing, missing


def run_consistency_problems(lock):
    """Diff review R2 (as A5a review I2 changed it): in every run root of the lock, the subjects each run says it finished -- the done of its stopped record and the rows of its final record, the first
    run's and every resume's (run_names) -- have a record subjects/<s>.json (a deleted record shows), and every record has its prediction's provenance.json.  A prediction without a record is residue
    (residue_subjects), not a problem here: the resume moves it out.  Reads the run records and lists the directories; no prediction is hashed (written_subject_problems does that, before a resume)."""
    problems = []
    for out in run_outs(lock):
        out = Path(out)
        records = subject_records(out)
        for path in run_files(out):
            kind = RUN_FILE.fullmatch(path.name).group(1)
            if kind == 'started':
                continue
            try:
                doc = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError) as exc:
                problems.append(f'{path}: {type(exc).__name__}: {exc}')
                continue
            said = set(doc.get('done') or []) if kind == 'stopped' else set(doc.get('rows') or {})
            if not said <= records:
                problems.append(f'{path} says done {sorted(said)}, subjects/ holds {sorted(records)}: a record is missing')
        lost = sorted(s for s in records if not (out/'pred'/s/hs_seg.PROVENANCE_FILE).is_file())
        if lost:
            problems.append(f'{out}: subjects with a record but no complete prediction: {lost}')
    return problems


def process_start_ticks(pid, proc=None):
    """Diff review 3, t1: when the process `pid` started, in clock ticks since boot (/proc/<pid>/stat field 22); None when it cannot be read.  Counted from boot, it does not move with the wall clock."""
    proc = Path(PROC if proc is None else proc)
    try:
        stat = (proc/str(pid)/'stat').read_text(encoding='utf-8')
        return int(stat[stat.rindex(')')+2:].split()[19])          # after "pid (comm) ": field 3 is [0], field 22 is [19]
    except (OSError, ValueError, IndexError):
        return None


def boot_id(proc=None):
    """Diff review 3, t1: the id of this boot (/proc/sys/kernel/random/boot_id); None when it cannot be read."""
    proc = Path(PROC if proc is None else proc)
    try:
        text = (proc/'sys'/'kernel'/'random'/'boot_id').read_text(encoding='utf-8').strip()
    except OSError:
        return None
    return text or None


def process_identity(pid=None, proc=None):
    """What a lock line records of the process that writes it (diff review 3, t1): pid, start ticks, boot id."""
    pid = os.getpid() if pid is None else pid
    return dict(pid=pid, start_ticks=process_start_ticks(pid, proc), boot_id=boot_id(proc))


def pid_alive(pid, start_ticks=None, boot=None, proc=None):
    """Diff review n5 and diff review 3 t1: whether the process of a lock line may still run.  The pid does not answer -> gone.  It answers: with the line's start ticks and boot id, another boot id, or the same boot
    with other start ticks, is another process that got the number (gone); the same two are the same process (alive).  Anything unknown -- no ticks or boot id in the line, /proc unreadable, a bad pid -- counts as
    alive (fail closed).  No wall clock is used.  Holds only within one PID namespace (the processes of this machine's user units; runs/20261107-hi/入口说明.md §3.2)."""
    if type(pid) is not int or pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    if type(start_ticks) is not int or not isinstance(boot, str) or not boot:
        return True
    now_boot, now_ticks = boot_id(proc), process_start_ticks(pid, proc)
    if now_boot is None:
        return True
    if now_boot != boot:
        return False
    if now_ticks is None:
        return True
    return now_ticks == start_ticks


def check_once(resume_path, lock):
    """Before anything of a run: no lock and no resume record (the first run), or a lock and a resume record (load_resume decides).  HIRefused (stage once) otherwise."""
    lock = Path(lock)
    if resume_path is None and lock.exists():
        first = (lock_lines(lock) or [{}])[0]
        raise HIRefused('once', f"{lock} exists: the hold-out inference ran at {first.get('utc')} (out {first.get('out')}); it runs once, whatever --out says. After a failure only with an explicit resume record "
                                f"({RESUME_DIR_REL}/unseal_holdout_B_v1_resume_<k>.json, committed); the lock is never edited or deleted")
    if resume_path is not None and not lock.exists():
        raise HIRefused('once', f'{lock} does not exist: there is no run to resume')


def load_resume(path, *, unseal, lock, repo=None):
    """The explicit authorization of a run after a failure (review I4, diff review R2 and n5): a committed record in splits/bayesreg_v1/ named unseal_holdout_B_v1_resume_<k>.json, of schema bayesreg.hi.resume.v1,
    written by BZ or the lead, for this unseal record and this lock (their sha256), naming the latest run of the lock (no authorized resume names it yet), which did not finish (no hi_record.json; hi_stopped.json
    with its sha256, or null when the process was killed and wrote none) and whose process is gone (pid_alive false), every run of the lock consistent (run_consistency_problems), and exactly the subjects of the
    unseal record that no run of the lock finished (finished_subjects).  A failed run is resumed at most once (the resume lock is named after it).  HIRefused (stage resume) otherwise."""
    repo = Path(REPO if repo is None else repo)
    path, lock = Path(path), Path(lock)
    if path.resolve().parent != (repo/RESUME_DIR_REL).resolve() or not RESUME_NAME.fullmatch(path.name):
        raise HIRefused('resume', f'{path} is not {RESUME_DIR_REL}/unseal_holdout_B_v1_resume_<k>.json in the code tree {repo}')
    if not path.is_file():
        raise HIRefused('resume', f'{path}: no such file')
    tracked = tracked_unchanged(path, repo, stage='resume')
    raw, document = _strict_json(path, 'resume')
    if not isinstance(document, dict) or sorted(document) != sorted(RESUME_KEYS):
        raise HIRefused('resume', f'{path}: exactly the keys {list(RESUME_KEYS)} are required')
    sha = hashlib.sha256(raw).hexdigest()
    problems = []
    if document['schema'] != RESUME_SCHEMA:
        problems.append(f"schema {document['schema']!r}")
    if document['written_by'] not in RESUME_WRITERS:
        problems.append(f"written_by {document['written_by']!r} is not one of {RESUME_WRITERS}")
    for key in ('words', 'time_local', 'reason'):
        if not _nonempty(document[key]):
            problems.append(f'{key}: a non-empty string is required')
    problem = utc_problem(document['utc'], 'utc')
    if problem:
        problems.append(problem)
    if document['unseal_sha256'] != unseal.sha256:
        problems.append(f"unseal_sha256 {document['unseal_sha256']!r} is not the unseal record's {unseal.sha256}")
    if not lock.is_file() or document['lock_sha256'] != file_sha256(lock):
        problems.append(f"lock_sha256 {document['lock_sha256']!r} is not the sha256 of {lock}")
    previous = Path(str(document['previous_out']))
    latest = latest_run(lock)
    if str(previous) not in run_outs(lock):
        problems.append(f'previous_out {previous} is not the output directory of a run of the lock')
    elif latest is None or latest['out'] != str(previous):
        problems.append(f"previous_out {previous} is not the latest run of the lock ({None if latest is None else latest['out']}): a run is resumed once, and only the last one")
    elif pid_alive(latest.get('pid'), start_ticks=latest.get('start_ticks'), boot=latest.get('boot_id')):
        problems.append(f"the process of the run {previous} (pid {latest.get('pid')}) may still be running: no resume while it is")
    finals = [p.name for p in run_files(previous) if RUN_FILE.fullmatch(p.name).group(1) == 'record']
    if finals:
        problems.append(f'{previous} finished ({finals} there): nothing to resume')
    started = json.loads((previous/STARTED_RECORD).read_text(encoding='utf-8')) if (previous/STARTED_RECORD).is_file() else None
    if started is None or (started.get('unseal') or {}).get('sha256') != unseal.sha256:
        problems.append(f'{previous}: no {STARTED_RECORD} of this unseal record')
    stopped = previous/STOPPED_RECORD
    if document['stopped_sha256'] is None:
        if stopped.exists():
            problems.append(f'stopped_sha256 is null but {stopped} is there')
    elif not stopped.is_file() or file_sha256(stopped) != document['stopped_sha256']:
        problems.append(f"stopped_sha256 {document['stopped_sha256']!r} is not the sha256 of {stopped}")
    problems += run_consistency_problems(lock)
    done = sorted(finished_subjects(lock))
    remaining = sorted(set(unseal.subjects)-set(done))
    if not remaining:
        problems.append('no subject of the unseal record is left')
    if document['subjects_sha256'] != subject_list_sha256(remaining) or not _is_count(document['n_subjects']) or document['n_subjects'] != len(remaining):
        problems.append(f"subjects_sha256 / n_subjects are not those of the {len(remaining)} subjects no run finished ({subject_list_sha256(remaining)})")
    own = resume_lock_path(lock, previous)
    if own.exists():
        problems.append(f'{own} exists: the run {previous} was resumed already')
    if problems:
        raise HIRefused('resume', f'{path}: {problems}')
    return Resume(path=str(path), sha256=sha, previous_out=str(previous), subjects=tuple(remaining), done=tuple(done), lock_sha256=document['lock_sha256'], lock_path=str(own),
                  document=MappingProxyType(dict(document)), git=MappingProxyType(tracked))


def take_lock(path, line):
    """O_EXCL: a second run loses here even if it passed every check before.  The file is written once, never edited or deleted.  HIRefused (stage once) if it is there."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise HIRefused('once', f'{path} appeared just before the first image; not read') from None
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        stream.write(json.dumps(plain(line), ensure_ascii=False)+'\n')
    return line


# ---- the decision of the development check ------------------------------------------------------------------------------------------------------
def check_decision(decision):
    """Review I3 and diff review R1: the development check was read and the lead's choice written (DEVCHECK_DECISION, a reviewed commit): exactly DECISION_KEYS, backend hs_a, the reading "逐位相同 20/20：选 hs_a"
    (devcheck_plan.md §4: any other reading leaves the decision None; another way needs a new plan and its sha256 in the code, reviewed), the devcheck.json at DEVCHECK_OUT with this sha256, made under the pinned
    plan (its plan.sha256 = DEVCHECK_PLAN_SHA256) on exactly the 20 development subjects, its reading, backends and environment the decision's, on cuda.  HIRefused (stage decision) otherwise; nothing of a subject is
    read.  DEVCHECK_OUT and DEVCHECK_PLAN_SHA256 are read at call time (the tests point the first into tmp_path)."""
    if decision is None:
        raise HIRefused('decision', f'DEVCHECK_DECISION is None: the development check has not been read and decided ({DEVCHECK_PLAN_REL} §4); the hold-out entry does not run')
    if not isinstance(decision, (dict, MappingProxyType)) or sorted(decision) != sorted(DECISION_KEYS):
        raise HIRefused('decision', f'DEVCHECK_DECISION: exactly the keys {list(DECISION_KEYS)} are required')
    problems = []
    if decision['backend'] != PRODUCTION_BACKEND:
        problems.append(f"backend {decision['backend']!r}: the entry runs {PRODUCTION_BACKEND!r} only")
    select = READ_SELECT.format(n=len(DEV_SUBJECTS))
    if decision['reading'] != select:
        problems.append(f"reading {decision['reading']!r}: only {select!r} lets the entry run under the pinned plan ({DEVCHECK_PLAN_REL} §4)")
    environment = decision['environment']
    if not isinstance(environment, (dict, MappingProxyType)) or sorted(environment) != sorted(DECISION_ENVIRONMENT_KEYS):
        raise HIRefused('decision', f'DEVCHECK_DECISION.environment: exactly the keys {list(DECISION_ENVIRONMENT_KEYS)} are required')
    path = Path(str(decision['devcheck_json']))
    if path.resolve() != (Path(DEVCHECK_OUT)/DEVCHECK_RECORD).resolve():
        raise HIRefused('decision', f'{path} is not the one development check, {Path(DEVCHECK_OUT)/DEVCHECK_RECORD}')
    if not is_hex64(decision['devcheck_sha256']) or not path.is_file() or file_sha256(path) != decision['devcheck_sha256']:
        raise HIRefused('decision', f"{path}: missing or not the sha256 {decision['devcheck_sha256']!r}")
    record = json.loads(path.read_text(encoding='utf-8'))
    checked = record.get('environment') or {}
    if record.get('schema') != DEVCHECK_SCHEMA or PRODUCTION_BACKEND not in (record.get('backends') or []):
        problems.append(f"{path}: schema {record.get('schema')!r}, backends {record.get('backends')}")
    if (record.get('plan') or {}).get('sha256') != DEVCHECK_PLAN_SHA256:
        problems.append(f"the check ran under the plan sha256 {(record.get('plan') or {}).get('sha256')!r}, not the pinned {DEVCHECK_PLAN_SHA256}")
    if sorted(record.get('subjects') or []) != sorted(DEV_SUBJECTS):
        problems.append(f"the check's subjects are not the 20 development subjects ({len(record.get('subjects') or [])})")
    if (record.get('reading') or {}).get('text') != decision['reading']:
        problems.append(f"reading {decision['reading']!r} is not the check's {(record.get('reading') or {}).get('text')!r}")
    lines = lock_lines(DEVCHECK_LOCK)                        # diff review 2, s2: the check's own record of its lock is the lock's line
    if not lines or (record.get('lock') or {}).get('line') != lines[0] or Path(str((record.get('lock') or {}).get('path'))).resolve() != Path(DEVCHECK_LOCK).resolve():
        problems.append(f"the check's lock record {(record.get('lock') or {}).get('line')} is not the line of {DEVCHECK_LOCK} ({lines[:1]})")
    if checked.get('device') != 'cuda' or any(checked.get(k) != environment[k] for k in DECISION_ENVIRONMENT_KEYS):
        problems.append(f'the environment {dict(environment)} is not the check\'s {({k: checked.get(k) for k in ("device",)+DECISION_ENVIRONMENT_KEYS})}')
    if problems:
        raise HIRefused('decision', f'DEVCHECK_DECISION: {problems}')
    return dict(plain(dict(decision)), checked='devcheck.json at DEVCHECK_OUT, its sha256, the select reading, the pinned plan, the 20 subjects, the lock line, backends and environment')


def check_environment(device, decision):
    """Review I3: the hold-out runs on cuda only, in the environment of the development check (gpu, torch, cuda, cudnn, cudnn_benchmark, cudnn_deterministic).  Returns the environment; HIRefused (stage decision)."""
    import torch
    device = torch.device(device)
    if device.type != 'cuda':
        raise HIRefused('decision', f'device {device}: the hold-out inference runs on cuda only (the development check\'s numerics; AMP is off elsewhere)')
    environment = environment_of(device)
    differ = {k: (environment.get(k), decision['environment'][k]) for k in DECISION_ENVIRONMENT_KEYS if environment.get(k) != decision['environment'][k]}
    if differ:
        raise HIRefused('decision', f'the environment differs from the development check\'s (now, then): {differ}')
    return environment


# ---- the machine gate ---------------------------------------------------------------------------------------------------------------------------
def units_of_this_process(cgroup=PROC_CGROUP, units=OWN_UNITS):
    """Diff review n1: the units of `units` this process really runs in (its cgroup path names <unit>.service); none when it runs outside them or the file cannot be read."""
    try:
        text = Path(cgroup).read_text(encoding='utf-8')
    except OSError:
        return []
    return [u for u in units if re.search(rf'/{re.escape(u)}\.service(/|$)', text, re.MULTILINE)]


def machine_gate(*, run=None, smi=SMI, cgroup=PROC_CGROUP, own_units=OWN_UNITS):
    """Review m4, after bayesreg.seg.unet.cli.machine_gate: no active or activating user unit of GATE_UNIT_PATTERNS but the one this process runs in (diff review n1: tag-hi-infer is excluded only when /proc/self/cgroup
    shows this process inside it; a stuck tag-hi-infer and a run outside it refuse), no GPU compute process, utilization <= GPU_UTIL_MAX.  A query that fails refuses.  Runs before this process touches the GPU.
    HIRefused (stage machine) otherwise.  Under WSL `nvidia-smi --query-compute-apps` often lists no process at all (the same limit as cli.machine_gate); the utilization reading still counts."""
    run = subprocess.run if run is None else run
    own = units_of_this_process(cgroup, own_units)        # tag-hi-infer for infer, tag-hi-devcheck for the check (diff review 2, s2)
    record = dict(check='HI machine gate (review m4, n1)', unit_patterns=list(GATE_UNIT_PATTERNS), own_units=list(own), cgroup=str(cgroup), checked_utc=utc())
    try:
        units = run(['systemctl', '--user', 'list-units', '--state=active,activating', '--no-legend', '--plain', *GATE_UNIT_PATTERNS], capture_output=True, text=True, timeout=60)
        if units.returncode != 0:
            raise HIRefused('machine', f'systemctl --user list-units failed ({units.stderr.strip()})')
        active = sorted(line.split()[0] for line in units.stdout.splitlines() if line.strip())
        others = [u for u in active if u.removesuffix('.service') not in own]
        record.update(active_units=active, other_units=others)
        if others:
            raise HIRefused('machine', f"other units are active ({others}): this run ({'/'.join(own_units)}) runs alone")     # diff review 3, t2: the caller's unit
        apps = run([str(smi), '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True, timeout=60)
        util = run([str(smi), '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise HIRefused('machine', f'the machine could not be queried ({type(exc).__name__}: {exc})') from None
    pids = [p.strip() for p in apps.stdout.splitlines() if p.strip().isdigit()]
    reading = (util.stdout.strip().splitlines() or [''])[0].strip()
    if apps.returncode != 0 or util.returncode != 0 or not reading.isdigit():
        raise HIRefused('machine', f'nvidia-smi gave no reading (compute apps exit {apps.returncode}, utilization {reading!r})')
    record.update(gpu_compute_pids=pids, gpu_utilization_percent=int(reading), gpu_util_max=GPU_UTIL_MAX)
    if pids or int(reading) > GPU_UTIL_MAX:
        raise HIRefused('machine', f'GPU busy (compute pids {pids or "none"}, utilization {reading}%)')
    record['passed'] = True
    return record


# ---- the guard ----------------------------------------------------------------------------------------------------------------------------------
def allowlist_guard(subject, allowlist):
    """bayesreg.seg.unet.infer.check_subject_admitted: the guard of 04_分割器.md §1-5, the one infer.segment_subject and hs_seg.uf_plan call (SubjectSealed)."""
    from bayesreg.seg.unet.infer import check_subject_admitted
    if not isinstance(allowlist, DevAllowlist):
        raise HIRefused('admission', 'a DevAllowlist (pins.load_pinned_allowlist) is required')
    check_subject_admitted(subject, allowlist)


def other_layers(subject, *, unseal_list, unseal_source, allowlist, sealed, training):
    """The sealed lists and the allowlist sources (T0, the training set, the development stages) for a subject the allowlist admitted.  HIRefused (stage admission)."""
    which = sealed.which(subject)
    on = sorted(name for name, ids in sealed.subjects.items() if subject in ids)
    if which in ('H2', 'H3') or any(name != unseal_list for name in on):
        raise HIRefused('admission', f'{subject} is on the sealed lists {on or [which]}: only {unseal_list!r} is unsealed by this record (H1-H4 never by this entry)')
    holding = sorted(s['name'] for s in allowlist.sources if subject in s['subjects'])
    never = sorted(s['name'] for s in allowlist.sources if subject in s['subjects'] and s['never_for_training'] and s['name'] != unseal_source)
    if never:
        raise HIRefused('admission', f'{subject} is in the never_for_training sources {never} (T0): never segmented by the hold-out entry')
    if subject in set(training):
        raise HIRefused('admission', f'{subject} is a B-line training subject (folds): never segmented by the hold-out entry (D24 would refuse it for v4 too)')
    if holding != [unseal_source]:
        raise HIRefused('admission', f'{subject} is in the allowlist sources {holding}, not only the unsealed source {unseal_source!r}')
    if on != [unseal_list]:
        raise HIRefused('admission', f'{subject} is not on the sealed list {unseal_list!r} that the record unseals')
    return dict(sealed_list=which, allowlist_sources=holding)


def admit(subject, *, unseal, allowlist, sealed, training):
    """Step 6 of the module docstring for one subject, before any of its files is opened: the guard on the allowlist bound to the record (C23.4: a hold-out subject passes it only through the record's unseal
    object), the allowlist carrying this record's unseal object, then HI's own layers.  Returns the record; HIRefused or SubjectSealed otherwise."""
    if not isinstance(unseal, Unseal):
        raise HIRefused('unseal', 'no unseal record: the hold-out is sealed until BZ writes one; no file of any subject is read')
    allowlist_guard(subject, allowlist)
    if getattr(allowlist, 'unseal', None) is not unseal.base:
        raise HIRefused('unseal', f"{subject}: the allowlist does not carry this record's unseal object (bayesreg.io.unseal.bind_unseal, C23.8-1)")
    if subject not in unseal.subjects:
        raise HIRefused('admission', f'{subject} is not one of the {len(unseal.subjects)} subjects of the unseal record {unseal.path}')
    layers = other_layers(subject, unseal_list=unseal.sealed_list, unseal_source=unseal.allowlist_source, allowlist=allowlist, sealed=sealed, training=training)
    return dict(subject=subject, guard='bayesreg.seg.unet.infer.check_subject_admitted (pinned allowlist)', allowlist_sha256=allowlist.sha256, unseal_sha256=unseal.sha256, **layers)


def guard_census(groups, *, allowlist, sealed, training, registry=None):
    """The dry run's view of the guard without an unseal record: for each named group of ids, which layer refuses each id first (the allowlist guard, the sealed lists, the sources and T0, the training set, and,
    with a registry, D24).  Opens no file of any subject.  Returns {group: {layer: count}}."""
    from bayesreg.seg.unet.infer import ModelSelectionRefused, select_fold_models
    out = {}
    for group, ids in groups.items():
        counts = {}
        for subject in ids:
            try:
                allowlist_guard(subject, allowlist)
                other_layers(subject, unseal_list=None, unseal_source=None, allowlist=allowlist, sealed=sealed, training=training)
                layer = 'admitted (would still need the unseal record)'
            except SubjectSealed:
                layer = 'allowlist guard (SubjectSealed)'
            except HIRefused as refused:
                text = str(refused)
                layer = 'sealed lists' if 'sealed lists' in text else 'never_for_training (T0)' if 'never_for_training' in text else 'training set' if 'training subject' in text else 'allowlist sources'
            if registry is not None and layer.startswith(('training', 'never')):
                try:
                    select_fold_models(registry, subject, allowlist, segmenter_id=CONTRACT_SEGMENTER['segmenter_id'])
                except ModelSelectionRefused:
                    layer += ' + D24 (ModelSelectionRefused)'
            counts[layer] = counts.get(layer, 0)+1
        out[group] = counts
    return out


# ---- one prediction -----------------------------------------------------------------------------------------------------------------------------
def environment_of(device):
    """bayesreg.seg.unet.train.environment_record without its code record (given separately), with cuDNN's switches, as hs_seg.infer_all records it."""
    import torch
    from bayesreg.seg.unet.train import environment_record
    environment = {k: v for k, v in environment_record(device).items() if k != 'code'}
    environment.update(cudnn_benchmark=bool(torch.backends.cudnn.benchmark), cudnn_deterministic=bool(torch.backends.cudnn.deterministic))
    return environment


def prepare_device(device, *, free_min_gib=GPU_FREE_MIN_GIB):
    """torch.device; on cuda the free memory checked and the process capped at GPU_CAP_GIB (hs_seg.infer_all)."""
    import torch
    device = torch.device(device)
    if device.type == 'cuda':
        from bayesreg.seg.unet.large import cap_gpu_memory
        free, _ = torch.cuda.mem_get_info(device)
        if free < free_min_gib*2**30:
            raise HIRefused('machine', f'{free/2**30:.1f} GiB free on {device}, {free_min_gib:g} GiB required; nothing is read')
        cap_gpu_memory(device, GPU_CAP_GIB)
    return device


def effective_amp(requested, device):
    """bayesreg.seg.unet.train.predict_logits: use_amp = amp and device_type == "cuda" (review I3: the provenance records what was used)."""
    import torch
    return bool(requested) and torch.device(device).type == 'cuda'


def hs_a_probabilities(subject, *, files, registry, models, plan, device):
    """hs_seg.uf4_probabilities, as HS's seg phase calls it: (float32 (36, ...), image metadata, normalization, member records)."""
    return hs_seg.uf4_probabilities(subject, files=files, registry=registry, models=models, plan=plan, device=device)


def stored_arrays(probs, grouping):
    """(probs36 float16, structure_groups float16): what hs_seg.write_prediction stores for (a), in memory (float16_keeping_argmax with the labels of hs_seg.predict_subject (a): the argmax of probs.astype(float16))."""
    from bayesreg.seg.unet.groups import group_probabilities
    labels = probs.astype(np.float16).argmax(axis=0)
    p16, n_ties = hs_seg.nx_seg.float16_keeping_argmax(probs, labels)
    del labels
    return p16, group_probabilities(probs, grouping, dtype=np.float32).astype(np.float16), n_ties


def provenance_for(subject, *, registry, models, policy, members, meta, norm, grouping, grouping_check, code, environment, command, allowlist, seconds, hi, anchor=CONTRACT_SEGMENTER):
    """hs_seg.provenance_of with the formal registry (its repository path and sha256), the note of this entry instead of the staged registry's, and the block `hi` (schema, backend, purpose, the unseal record, the
    admission, the anchor).  Every field the registered predictions of the development subjects carry, so the Hx line's readers take it the same way."""
    composition = dict(COMPOSITION, sources=dict(uf4=dict(members=[m['member_sha256'] for m in members])))
    document = hs_seg.provenance_of(subject, candidate=CANDIDATE, registry=registry, models=models, policy=policy, members=members, meta=meta, norm=norm, grouping=grouping, grouping_check=grouping_check,
                                    registry_rel=anchor['registry'], staged_sha256=registry.sha256, code=code, environment=environment, command=list(command), composition=composition, seconds=seconds, allowlist=allowlist)
    document['registry_note'] = REGISTRY_NOTE
    document['hi'] = dict(hi, schema=HI_PREDICTION_SCHEMA, amp_rule=AMP_RULE)
    return plain(document)


def predict_hs_a(subject, *, files, registry, models, plan, policy, device, grouping, grouping_check, pred_root, code, environment, command, allowlist, hi):
    """One subject through hs_a: the probabilities, then hs_seg.write_prediction (probs36 float16 with the argmax kept, structure_groups, the reader on the stored file, provenance.json last; never overwritten)."""
    started = time.perf_counter()
    probs, meta, norm, members = hs_a_probabilities(subject, files=files, registry=registry, models=models, plan=plan, device=device)
    seconds_forward = time.perf_counter()-started
    members = [dict(m, amp=effective_amp(m['amp'], device)) for m in members]
    labels = probs.astype(np.float16).argmax(axis=0)            # hs_seg.predict_subject (a): ties to the smallest label
    provenance = provenance_for(subject, registry=registry, models=models, policy=policy, members=members, meta=meta, norm=norm, grouping=grouping, grouping_check=grouping_check, code=code,
                                environment=environment, command=command, allowlist=allowlist, seconds=time.perf_counter()-started, hi=hi)
    row = hs_seg.write_prediction(subject, probs, labels, out_root=pred_root, grouping=grouping, provenance=provenance)
    return dict(row, seconds_forward=seconds_forward, seconds_total=time.perf_counter()-started, image_sha256=meta['sha256'], members=[m['member_sha256'] for m in members])


# ---- the hold-out entry -------------------------------------------------------------------------------------------------------------------------
def infer_holdout(subjects, *, unseal_path, out_root, files, device, allowlist, allowlist_record, sealed, training, registry, models_root, grouping, code, command, repo=None, backend=PRODUCTION_BACKEND,
                  anchor=CONTRACT_SEGMENTER, resume_path=None, free_min_gib=GPU_FREE_MIN_GIB, log=print):
    """The hold-out inference, in the order of the module docstring (steps 1-11).  subjects None: every subject of the unseal record (of the resume record when resuming).  A failure after the lock writes
    hi_stopped.json and stops: the evidence is kept, nothing is rerun.  DEVCHECK_DECISION, HOLDOUT_LOCK, RESULTS, check_environment and machine_gate are read at call time (the tests replace them).
    `allowlist` and `files`: the pinned allowlist and the OasisFiles, unbound; right after the record is loaded they are bound to its unseal object (bind, C23.8-1) and only the bound ones are used.  `repo`: the
    tree of the resume records (None: REPO); the record itself is read by bayesreg's loader from bayesreg's pinned tree."""
    from bayesreg.seg.unet.groups import require_inference_grouping
    if backend != PRODUCTION_BACKEND:
        raise HIRefused('backend', f"{backend!r}: the hold-out entry runs {PRODUCTION_BACKEND!r} only (segment_subject's provenance lacks checkpoint_choice, checkpoint_iteration, amp and window per member: "
                                   'C22.1-2 修订 4 上游推断的规格; bayesreg/ is not changed by this package)')
    problem = holdout_out_problem(out_root)     # A5a review I2 point 1: the one root, checked here too (not only on the command line)
    if problem:
        raise HIRefused('output', problem)
    code = pinned_code(code)                    # A5a: rule one before the record is read (contract C24.3; the lock line's code_commit is P1)
    unseal = load_unseal(unseal_path, anchor=anchor)
    allowlist, files = bind(unseal, allowlist, files)
    lock = Path(HOLDOUT_LOCK)
    check_once(resume_path, lock)
    resume = None if resume_path is None else load_resume(resume_path, unseal=unseal, lock=lock, repo=repo)
    if resume is not None and Path(resume.previous_out).resolve() != Path(out_root).resolve():
        raise HIRefused('resume', f'the resume record names {resume.previous_out}, not the one root {out_root}: a resume writes the subjects left into the root of the run it resumes')
    k = 0 if resume is None else 1+sum(1 for r in run_lines(lock) if r.get('event') == AUTHORIZED_RESUME)     # the run's number in the root (run_names)
    decision = check_decision(DEVCHECK_DECISION)
    registry_record = check_registry(registry, anchor)
    allowed = list(unseal.subjects) if resume is None else list(resume.subjects)
    subjects = allowed if subjects is None else subjects
    if not isinstance(subjects, (list, tuple)) or not subjects:
        raise HIRefused('admission', 'a non-empty list of subjects is required')
    try:
        subjects = canonical_subject_list(list(subjects), 'subjects')
    except InputValidationError as exc:         # a non-canonical id or a duplicate is refused, never repaired
        raise SubjectSealed(f'the subjects of the run: {exc}') from None
    if resume is not None and sorted(subjects) != sorted(resume.subjects):
        raise HIRefused('resume', f'a resumed run takes exactly the {len(resume.subjects)} subjects of its resume record')
    admissions = {s: admit(s, unseal=unseal, allowlist=allowlist, sealed=sealed, training=training) for s in subjects}
    grouping_check = require_inference_grouping(grouping)
    available = member_files(registry, models_root)
    plans = {s: plan_of(registry, s, allowlist, available) for s in subjects}
    check_out_location(out_root, resume=resume is not None)
    residue = []
    if resume is not None:                      # A5a review I2 point 2: before the lock, the subjects already written are as their records say; what is not written is residue
        problems = written_subject_problems(out_root)
        if problems:
            raise HIRefused('resume', f'the subjects written in {out_root} are not as their records say (subjects/<s>.json): {problems[:5]}')
        residue = residue_subjects(out_root)
        written = set(residue) & finished_subjects(lock)
        if written or set(subjects) & finished_subjects(lock):
            raise HIRefused('resume', f'subjects of this resume already written: {sorted(written | (set(subjects) & finished_subjects(lock)))}')
    names = run_names(k)
    gate = machine_gate()
    device = prepare_device(device, free_min_gib=free_min_gib)
    environment = check_environment(device, decision)
    line = dict(unseal_path=unseal.path, unseal_sha256=unseal.sha256, out=str(Path(out_root)), subjects_sha256=subject_list_sha256(list(subjects)), n_subjects=len(subjects), members=registry_record['members'],
                models_root=str(models_root), code_commit=(code or {}).get('commit'), command=list(command), devcheck_sha256=decision['devcheck_sha256'],
                event=FIRST_RUN if resume is None else AUTHORIZED_RESUME, resume=None if resume is None else dict(path=resume.path, sha256=resume.sha256, previous_out=resume.previous_out), utc=utc(),
                **process_identity())     # pid, start_ticks, boot_id (diff review 3, t1)
    take_lock(lock if resume is None else Path(resume.lock_path), line)
    root = fresh_root(out_root) if resume is None else Path(out_root)
    kept = None
    if resume is not None:                      # after the lock: the residue moved out of pred/, kept whole (the lead's ruling (5))
        try:
            kept = keep_residue(root, residue, k)
        except (Exception, SystemExit) as failed:       # A5a diff review n3: the resume lock is taken (one resume per root): never a lock without a stopped record
            write_json_once(root/names['stopped'], plain(dict(
                subject=None, stage='keep_residue', error=f'{type(failed).__name__}: {failed}', done=[], run=k, utc=utc(),
                residue=dict(dir=str(root/RESIDUE_DIR.format(k=k)), subjects=list(residue)),
                note='stopped while the residue was moved out of pred/ (attempt<k>/residue.json and residue_moved.json hold the sha256 before and after, as far as they were written); '
                     'kept, not rerun; this root was resumed once (its resume lock is taken): the lead and BZ decide')))
            raise
    common = dict(unseal=unseal.record(), resume=None if resume is None else resume.record(), lock=dict(path=str(lock if resume is None else resume.lock_path), line=line), decision=decision,
                  registry=registry_record, allowlist=allowlist_record, sealed_lists=sealed.record, subjects=list(subjects), subjects_sha256=subject_list_sha256(list(subjects)), backend=backend,
                  models_root=str(models_root), data_root=str(getattr(files, 'root', None)), machine_gate=gate, code=code, command=list(command), environment=environment, run=k, names=names,
                  residue=kept)
    write_json_once(root/names['started'], plain(dict(common, schema=HI_INFER_SCHEMA, utc=utc())))
    done = root/'subjects'
    done.mkdir(exist_ok=resume is not None)
    rows = {}
    for subject in subjects:
        plan, models, policy = plans[subject]
        hi = dict(backend=backend, purpose='hold-out', unseal=unseal.record(), resume=None if resume is None else resume.record(), admission=admissions[subject], anchor=dict(anchor),
                  devcheck=dict(json=decision['devcheck_json'], sha256=decision['devcheck_sha256']), contract='C22.1-2 修订 4 (登记、成员、上游推断的规格、留出集); C22.9-4 (xv)', task='runs/20261107-hi/任务书.md')
        try:
            row = predict_hs_a(subject, files=files, registry=registry, models=models, plan=plan, policy=policy, device=device, grouping=grouping, grouping_check=grouping_check, pred_root=root/'pred', code=code,
                               environment=environment, command=command, allowlist=allowlist, hi=hi)
            row = dict(row, provenance_sha256=file_sha256(root/'pred'/subject/hs_seg.PROVENANCE_FILE), run=k)     # A5a review I2: the record binds provenance.json too, and names its run
            write_json_once(done/f'{subject}.json', plain(row))     # diff review 2, s1: inside the try, so that a failure here writes the stopped record too
        except (Exception, SystemExit) as failed:       # hs_seg.HSSegFailed (the reader's check), SystemExit (fast_channels' refusals), FileExistsError, anything else
            write_json_once(root/names['stopped'], plain(dict(subject=subject, error=f'{type(failed).__name__}: {failed}', done=sorted(rows), run=k, utc=utc(),
                                                              note='kept, not rerun; the subjects left run again only with an explicit resume record (load_resume), into this root')))
            raise
        rows[subject] = row
        log(f"{subject}: {row['seconds_total']:.1f} s (forward {row['seconds_forward']:.1f} s), the reader's argmax the stored argmax on every voxel, group deviation {row['group_max_abs_deviation']:.2e}")
    import torch
    complete, missing = complete_subjects(lock, unseal.subjects)
    record = dict(common, schema=HI_INFER_SCHEMA, rows=rows, n_reader_not_own_total=sum(r['n_reader_not_own'] for r in rows.values()), passed=all(r['n_reader_not_own'] == 0 for r in rows.values()) and len(rows) == len(subjects),
                  gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else None,
                  union=dict(written=sorted(finished_subjects(lock)), complete=complete, missing=missing,
                             rule='A5a review I2: HI is complete when the first run\'s and the resume\'s subjects/<s>.json in the one root, taken together, are every subject of the unseal record'),
                  utc=utc())
    write_json_once(root/names['record'], plain(record))
    return record


# ---- the development check ----------------------------------------------------------------------------------------------------------------------
def check_plan():
    """Review I3: the reading of the check is the one written before it ran (DEVCHECK_PLAN_SHA256).  HIRefused (stage plan) otherwise."""
    path = REPO/DEVCHECK_PLAN_REL
    sha = file_sha256(path) if path.is_file() else None
    if sha != DEVCHECK_PLAN_SHA256:
        raise HIRefused('plan', f'{DEVCHECK_PLAN_REL}: sha256 {sha} is not the pinned {DEVCHECK_PLAN_SHA256}: the reading is fixed before the run')
    return dict(path=DEVCHECK_PLAN_REL, sha256=sha)


def reference_record(reference_root, subject, *, registry, plan, anchor=CONTRACT_SEGMENTER):
    """Before any image: the registered prediction of the subject is of this segmenter, this registry file and these members in HS's order, all_folds_mean, test_labels_seen false, its two outputs named with their
    sha256.  Returns the record (the provenance's sha256, the image's sha256, the code and environment it was made with); HIRefused (stage reference) otherwise.  The outputs are re-hashed when they are read."""
    directory = Path(reference_root)/subject
    path = directory/hs_seg.PROVENANCE_FILE
    try:
        provenance = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise HIRefused('reference', f'{path}: {exc}') from None
    problems = []
    if provenance.get('subject') != subject or provenance.get('layer') != 'structure' or provenance.get('segmenter_id') != anchor['segmenter_id']:
        problems.append(f"subject {provenance.get('subject')!r}, layer {provenance.get('layer')!r}, segmenter {provenance.get('segmenter_id')!r}")
    if provenance.get('registry_sha256') != registry.sha256:
        problems.append(f"registry sha256 {provenance.get('registry_sha256')}, the registry file's {registry.sha256}")
    if [m.get('member_sha256') for m in provenance.get('members') or []] != [sha for _, sha, _ in plan]:
        problems.append("the members are not HS's plan, in order")
    if provenance.get('policy') != anchor['policy'] or provenance.get('test_labels_seen') is not False:
        problems.append(f"policy {provenance.get('policy')!r}, test_labels_seen {provenance.get('test_labels_seen')!r}")
    outputs = provenance.get('output_sha256') or {}
    if sorted(outputs) != sorted([hs_seg.PROBS_FILE, hs_seg.GROUPS_FILE]) or not all(is_hex64(v) for v in outputs.values()):
        problems.append(f'output_sha256 {outputs}')
    if problems:
        raise HIRefused('reference', f'{directory}: {problems}')
    environment = provenance.get('environment') or {}
    return dict(directory=str(directory), provenance_sha256=file_sha256(path), image_sha256=(provenance.get('input_sha256') or {}).get(hs_seg.IMAGE_FILE), output_sha256=dict(outputs),
                code_commit=(environment.get('code') or {}).get('commit'), environment={k: environment.get(k) for k in ('torch', 'cuda', 'cudnn', 'gpu', 'device', 'cudnn_benchmark', 'cudnn_deterministic')})


def read_reference(reference_root, subject):
    """(probs36 float16, structure_groups float16) of the registered prediction, after its outputs are re-hashed against its provenance (bayesreg.io.pair.verify_segmentation_files)."""
    directory = Path(reference_root)/subject
    provenance = json.loads((directory/hs_seg.PROVENANCE_FILE).read_text(encoding='utf-8'))
    verify_segmentation_files(provenance, directory)
    arrays = []
    for name in (hs_seg.PROBS_FILE, hs_seg.GROUPS_FILE):
        with np.load(directory/name, allow_pickle=False) as npz:
            arrays.append(npz['p'])
    return arrays[0], arrays[1]


def compare(mine, theirs, *, slab=SLAB):
    """Two stored float16 arrays (C, Z, Y, X): bitwise equal, the elements that differ (bit patterns), the largest absolute difference, and the argmax maps (ties to the smallest channel, np.argmax) compared voxel by
    voxel, with their agreement 1 - differing voxels / voxels."""
    record = dict(shape=list(mine.shape), dtype=str(mine.dtype), reference_shape=list(theirs.shape), reference_dtype=str(theirs.dtype))
    if mine.shape != theirs.shape or mine.dtype != np.float16 or theirs.dtype != np.float16:
        return dict(record, bitwise_equal=False, comparable=False)
    n_differ, largest = 0, 0.
    for z0 in range(0, mine.shape[1], slab):
        a, b = mine[:, z0:z0+slab], theirs[:, z0:z0+slab]
        bits = a.view(np.uint16) != b.view(np.uint16)
        n = int(bits.sum())
        if n:
            n_differ += n
            largest = max(largest, float(np.nanmax(np.abs(a[bits].astype(np.float64)-b[bits].astype(np.float64)))))
    top_mine, top_theirs = mine.argmax(axis=0), theirs.argmax(axis=0)
    n_top = int((top_mine != top_theirs).sum())
    return dict(record, comparable=True, bitwise_equal=n_differ == 0, n_elements=int(mine.size), n_elements_differ=n_differ, max_abs_difference=largest, n_voxels=int(top_mine.size),
                argmax_equal=n_top == 0, n_voxels_argmax_differ=n_top, argmax_agreement=1.-n_top/top_mine.size)


def devcheck_subject(subject, backend, *, files, registry, models, plan, policy, device, grouping, allowlist, models_root, out_root, reference_root, command, log=print):
    """One subject, one backend: the stored arrays against the registered files.  hs_a: in memory (stored_arrays: the bytes write_prediction would store); segment_subject: its own files under
    <out_root>/segment_subject/pred (it writes them itself; never overwritten)."""
    started = time.perf_counter()
    if backend == 'hs_a':
        probs, meta, _, members = hs_a_probabilities(subject, files=files, registry=registry, models=models, plan=plan, device=device)
        seconds_forward = time.perf_counter()-started
        p16, g16, n_ties = stored_arrays(probs, grouping)
        del probs
        image_sha = meta['sha256']
        member_order = [m['member_sha256'] for m in members]
        extra = dict(n_float16_ties_fixed=n_ties)
    elif backend == 'segment_subject':
        from bayesreg.seg.unet.infer import segment_subject
        pred_root = Path(out_root)/'segment_subject'/'pred'
        pred_root.mkdir(parents=True, exist_ok=True)
        provenance = segment_subject(subject, files=files, registry=registry, allowlist=allowlist, models_root=models_root, grouping=grouping, pred_root=pred_root, device=str(device),
                                     segmenter_id=CONTRACT_SEGMENTER['segmenter_id'], command=list(command), log=log)
        seconds_forward = None
        directory = pred_root/subject
        verify_segmentation_files(provenance, directory)
        with np.load(directory/hs_seg.PROBS_FILE, allow_pickle=False) as npz:
            p16 = npz['p']
        with np.load(directory/hs_seg.GROUPS_FILE, allow_pickle=False) as npz:
            g16 = npz['p']
        image_sha = provenance['input_sha256'][hs_seg.IMAGE_FILE]
        member_order = [m['member_sha256'] for m in provenance['members']]
        extra = dict(policy=provenance['policy'], provenance_sha256=file_sha256(directory/hs_seg.PROVENANCE_FILE), output_sha256=provenance['output_sha256'])
    else:
        raise HIRefused('backend', f'{backend!r} is not one of {BACKENDS}')
    seconds_inference = time.perf_counter()-started
    ref16, refg16 = read_reference(reference_root, subject)
    row = dict(subject=subject, backend=backend, image_sha256=image_sha, member_order=member_order, members_as_planned=member_order == [sha for _, sha, _ in plan], probs36=compare(p16, ref16),
               structure_groups=compare(g16, refg16), seconds_forward=seconds_forward, seconds_inference=seconds_inference, seconds_total=time.perf_counter()-started, **extra)
    del p16, g16, ref16, refg16
    return row


def summarize(rows, backends, subjects):
    out = {}
    for backend in backends:
        mine = [rows[backend][s] for s in subjects if s in rows.get(backend, {})]
        p = [r['probs36'] for r in mine]
        g = [r['structure_groups'] for r in mine]
        out[backend] = dict(n_subjects=len(mine), n_probs36_bitwise_equal=sum(x['bitwise_equal'] for x in p), n_probs36_argmax_equal=sum(bool(x.get('argmax_equal')) for x in p),
                            n_groups_bitwise_equal=sum(x['bitwise_equal'] for x in g), n_image_same_as_registered=sum(bool(r['image_same_as_registered']) for r in mine),
                            max_elements_differ=max((x.get('n_elements_differ', -1) for x in p), default=None), max_voxels_argmax_differ=max((x.get('n_voxels_argmax_differ', -1) for x in p), default=None),
                            min_argmax_agreement=min((x.get('argmax_agreement', 0.) for x in p), default=None), all_bitwise_equal=len(mine) == len(subjects) and all(x['bitwise_equal'] for x in p+g))
    return out


def devcheck_reading(rows, subjects, plan):
    """devcheck_plan.md §3, judged on hs_a only: every subject's probs36 and structure_groups bitwise equal -> READ_SELECT; any not -> READ_PENDING with the elements that differ and the argmax agreement of each
    subject; hs_a not done on every subject -> READ_UNDECIDED.  segment_subject is described in the summary, not judged."""
    mine = (rows or {}).get('hs_a') or {}
    rule = 'devcheck_plan.md §3: hs_a, every subject, probs36 and structure_groups bitwise equal -> select hs_a; otherwise 不过（待查）, handed to the lead; hs_a not done on every subject -> 不判'
    if sorted(mine) != sorted(subjects):
        return dict(text=READ_UNDECIDED, rule=rule, plan=plan, n_done=len(mine), n_subjects=len(subjects))
    per = {s: dict(probs36_bitwise_equal=mine[s]['probs36']['bitwise_equal'], probs36_elements_differ=mine[s]['probs36'].get('n_elements_differ'),
                   probs36_argmax_agreement=mine[s]['probs36'].get('argmax_agreement'), groups_bitwise_equal=mine[s]['structure_groups']['bitwise_equal'],
                   groups_elements_differ=mine[s]['structure_groups'].get('n_elements_differ'), image_same_as_registered=mine[s].get('image_same_as_registered')) for s in subjects}
    equal = all(r['probs36_bitwise_equal'] and r['groups_bitwise_equal'] for r in per.values())
    agreements = [r['probs36_argmax_agreement'] for r in per.values() if r['probs36_argmax_agreement'] is not None]
    return dict(text=READ_SELECT.format(n=len(subjects)) if equal else READ_PENDING, rule=rule, plan=plan, n_subjects=len(subjects), n_bitwise_equal=sum(r['probs36_bitwise_equal'] and r['groups_bitwise_equal'] for r in per.values()),
                min_probs36_argmax_agreement=min(agreements) if len(agreements) == len(per) else None, per_subject=per)


def devcheck(subjects, *, backends, out_root, files, device, allowlist, allowlist_record, sealed, training, registry, models_root, grouping, reference_root, code, command, anchor=CONTRACT_SEGMENTER,
             free_min_gib=GPU_FREE_MIN_GIB, expected_subjects=DEV_SUBJECTS, log=print):
    """The development check, once (diff review R1: the fixed lock DEVCHECK_LOCK; there -> HIRefused('once') before anything is read).  Before any image: the plan's sha256; the registry against the anchor; each
    subject a development subject of stage_L2R_val_v1 (the caller's eval_seg.admit_subjects) admitted by the same guard and planned (D24); every registered prediction's provenance; the output directory (new, under
    RESULTS); the device; then the lock with O_EXCL.  Then hs_a on every subject first, segment_subject after it (diff review n3: a failure of the described backend does not touch the judged one); a record per row;
    devcheck.json with the reading of the plan at the end (devcheck_stopped.json with 不判 when hs_a fails).  No truth label.  DEVCHECK_LOCK is read at call time (the tests point it into tmp_path)."""
    from bayesreg.seg.unet.groups import require_inference_grouping
    lock = Path(DEVCHECK_LOCK)
    if lock.exists():
        first = (lock_lines(lock) or [{}])[0]
        raise HIRefused('once', f"{lock} exists: the development check ran at {first.get('utc')} (out {first.get('out')}); it runs once (devcheck_plan.md §2), whatever --out says; the lock is never edited or deleted")
    backends = list(backends)
    if not backends or any(b not in BACKENDS for b in backends) or len(set(backends)) != len(backends):
        raise HIRefused('backend', f'{backends}: one or both of {BACKENDS}')
    backends = [b for b in BACKENDS if b in backends]          # hs_a first (n3)
    if Path(out_root).resolve() != Path(DEVCHECK_OUT).resolve():     # diff review 2 (optional): the one directory the decision takes, at the API too
        raise HIRefused('output', f'{out_root} is not {DEVCHECK_OUT}: the decision takes only {Path(DEVCHECK_OUT)/DEVCHECK_RECORD}; nothing is read and the lock is not spent')
    plan_record = check_plan()
    registry_record = check_registry(registry, anchor)
    subjects = list(subjects)
    if sorted(subjects) != sorted(expected_subjects):     # the tests' synthetic subjects stand in for the 20 through this argument; run_hi.py never passes it
        raise HIRefused('admission', f'the development check runs on the 20 development subjects {DEV_SUBJECTS[0]}..{DEV_SUBJECTS[-1]}, got {len(subjects)}')
    for subject in subjects:
        allowlist_guard(subject, allowlist)
        if sealed.which(subject) is not None or subject in allowlist.never_for_training or subject in set(training):
            raise HIRefused('admission', f'{subject}: on a sealed list, never_for_training or a training subject')
    grouping_check = require_inference_grouping(grouping)
    available = member_files(registry, models_root)
    plans = {s: plan_of(registry, s, allowlist, available) for s in subjects}
    references = {s: reference_record(reference_root, s, registry=registry, plan=plans[s][0], anchor=anchor) for s in subjects}
    check_out_location(out_root)
    gate = machine_gate(own_units=DEVCHECK_UNITS)          # diff review 2, s2: before this process touches the GPU and before the one lock is spent
    device = prepare_device(device, free_min_gib=free_min_gib)
    line = dict(out=str(Path(out_root)), plan_sha256=plan_record['sha256'], backends=backends, subjects_sha256=subject_list_sha256(sorted(subjects)), n_subjects=len(subjects),
                code_commit=(code or {}).get('commit'), command=list(command), event=FIRST_RUN, utc=utc(), pid=os.getpid())
    take_lock(lock, line)
    root = fresh_root(out_root)
    environment = environment_of(device)
    common = dict(schema=DEVCHECK_SCHEMA, subjects=subjects, backends=backends, plan=plan_record, lock=dict(path=str(lock), line=plain(line)), machine_gate=gate, registry=registry_record, allowlist=allowlist_record,
                  sealed_lists=sealed.record, grouping_check=grouping_check, reference_root=str(reference_root), models_root=str(models_root), data_root=str(getattr(files, 'root', None)), references=references,
                  code=code, command=list(command), environment=environment)
    write_json_once(root/'devcheck_started.json', plain(dict(common, utc=utc())))
    (root/'subjects').mkdir()
    rows = {b: {} for b in backends}
    failures = {}
    for backend in backends:
        for subject in subjects:
            plan, models, policy = plans[subject]
            try:
                row = devcheck_subject(subject, backend, files=files, registry=registry, models=models, plan=plan, policy=policy, device=device, grouping=grouping, allowlist=allowlist, models_root=models_root,
                                       out_root=root, reference_root=reference_root, command=command, log=log)
            except (Exception, SystemExit) as failed:
                failure = dict(subject=subject, backend=backend, error=f'{type(failed).__name__}: {failed}', done=sorted(rows[backend]), utc=utc())
                if backend == 'hs_a':
                    write_json_once(root/'devcheck_stopped.json', plain(dict(failure, reading=READ_UNDECIDED, note='kept, not rerun; the lead decides (devcheck_plan.md §3: 不判)')))
                    raise
                write_json_once(root/f'devcheck_{backend}_stopped.json', plain(dict(failure, note='the described backend stopped; the reading of hs_a stands (devcheck_plan.md §3); kept, not rerun')))
                failures[backend] = failure
                log(f'{subject} {backend}: {failure["error"]}; {backend} stops here, the reading of hs_a is not touched')
                break
            row['image_same_as_registered'] = row['image_sha256'] == references[subject]['image_sha256']
            write_json_once(root/'subjects'/f'{backend}__{subject}.json', plain(row))
            rows[backend][subject] = row
            p = row['probs36']
            log(f"{subject} {backend}: probs36 bitwise {p['bitwise_equal']} ({p.get('n_elements_differ')} elements differ), argmax {p.get('argmax_equal')} ({p.get('n_voxels_argmax_differ')} voxels); "
                f"groups bitwise {row['structure_groups']['bitwise_equal']}; {row['seconds_total']:.1f} s")
    import torch
    record = dict(common, rows=rows, summary=summarize(rows, backends, subjects), described_failures=failures, reading=devcheck_reading(rows, subjects, plan_record),
                  gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else None,
                  decision=dict(rule='C22.1-2 修订 4 留出集 and devcheck_plan.md §4: the lead confirms the reading; a reviewed commit writes hi_core.DEVCHECK_DECISION', made=False), utc=utc())
    write_json_once(root/DEVCHECK_RECORD, plain(record))
    return record


__all__ = ['BACKENDS', 'CONTRACT_SEGMENTER', 'DEVCHECK_DECISION', 'DEVCHECK_LOCK', 'DEVCHECK_OUT', 'HIRefused', 'HOLDOUT_LOCK', 'PRODUCTION_BACKEND', 'UNSEALABLE', 'UNSEAL_KEYS', 'UNSEAL_REL', 'UNSEAL_SCHEMA',
           'Unseal', 'admit', 'bind', 'check_code_root', 'check_decision', 'check_environment', 'check_registry', 'compare', 'devcheck', 'devcheck_reading', 'finished_subjects', 'guard_census', 'infer_holdout', 'predicted_subjects', 'process_identity', 'process_start_ticks', 'boot_id',
           'latest_run', 'load_resume', 'load_unseal', 'machine_gate', 'pid_alive', 'stored_arrays', 'tracked_unchanged', 'units_of_this_process', 'unseal_tracked']
