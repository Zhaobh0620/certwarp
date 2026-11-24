"""bayesreg.eval.holdout -- the one evaluation entry of the B-line hold-out (contract C23.12, revision 6, BZ 2026-11-07; D106-D109).

Work package EV (runs/20261107-ev/任务书.md and the lead's changes of 11-07; runs/20261107-ev/实现说明.md; the interface alignment of HB, PQH and EV,
runs/20261109-align/接口对齐.md, the lead ~9:10 PM CST 11-09).  It is the only code that reads a truth label of a hold-out subject (C23.6-3 class E): the label
NIfTI files aligned_seg35.nii.gz / aligned_seg4.nii.gz under the pinned DATA_ROOT, through read_truth only.  Every other entry (truth.admit_pair / load_truth,
eval.cli, eval_seg, OasisFiles.label_path) refuses sealed subjects as before (C23.6-1, C23.8-3).

Three stages (the lead, 11-07, change 2): admission -> truth -> the shared metric function.
  1. admission   open_holdout: (0) the truth-reader gate (TRUTH_READER_DECISION, C23.12-10) and the PQ replay gate (experiments/pq/pqh_replay.py's
                 replay_decision_problem, C22.H8, D119; contract revision 7), each with the code identity of the global rule (decision_log df8541d: the code
                 directories CODE_DIRS of the gate's commit byte for byte the HEAD's); (a) BZ's unseal object (SG, C23.3) bound to the pinned allowlist
                 (C23.4-1), every pair one row of the pinned holdout_B_pairs.tsv in its direction, both subjects admitted by SG's check_subject (C23.4-2), on the
                 sealed list holdout_B_v1 and no other (H1-H4 never); (b) the preregistration (PREREG_SHA256) and its machine-readable arm table (ARM_TABLE_SHA256,
                 the sha256 written in the preregistration), every (arm, pair) cell written by the D107 classes (below), the segmentation reading not requested,
                 then the machine gate (no other tag-* / seq-vp-* unit); (c) the sha256 manifest (D108) committed and unchanged against HEAD, holding exactly the
                 C23.12-4 inputs, every needed file listed, the records read in (b) at their listed sha256, PQ's release (C22.H4-5; Ev1) or BZ's choice (ii) and
                 PQ's outputs found through its list; (d) the master lock (O_EXCL, fixed path), written only after all of these and the clean code tree.  A
                 reevaluation (D109) needs a committed authorization instead of a free master lock (load_reeval).
                 The truth-reader gate has its own admission (admit_truth_gate: the 20 development subjects of stage_L2R_val_v1, the development allowlist and a
                 second lock as experiments/unv2/eval_seg.admit_subjects; no unseal object) and its own one-time lock (take_gate_lock); the PQ replay gate reads
                 development truth through pq_dev_truth (the same reader).
  2. truth       read_truth(subject, admission=..., grid_metadata=...): its first step checks the admission (an object issued by stage 1 of this process, its
                 lock taken, the pair lock of this subject's pair taken); then load_scalar(labels=True) of the two NIfTI files, the grid (shape, spacing,
                 origin) equal to the pair manifest's, finite non-negative integers; every file's sha256 into the record.
  3. score       score_cell(info, grid, truth, cert_mode=...): fields.load_field -> metrics.evaluate_field -> the certificate, the calls of
                 bayesreg.eval.cli.evaluate in the same order (C23.12-5: no metric of its own).  score_d is its inner part on a given d (the intention-to-treat
                 identity rows use it with d = 0).  The evaluation-entry control (preregistration 5.1 item 8) calls this same function object.

Cells (the alignment, items 1-2): every arm has its record directory (record_dir: the terminal record and the small records; the method arm's is its whole
cell, /mnt/d/seq_vp_tag_holdout_B_v1/method/<pair>/ after the move, with terminal.json, output/, review/review.json) and its field place apart (field: a fixed
directory, or the path and sha256 its terminal record names under the arm's field root on D: or /home).  What a cell is "written" means (D107; the lead's
change 3, I2): every cell's terminal status is mapped by the arm table to one class:
  output              the field and its complete record (the arm's complete_files); the method arm also the step-5 review record (review_record:
                      review/review.json, finalized true; C22.H1, D111: a pair not finalized is not written);
  method_failure      the method's own failure with its terminal record;
  unit_limit          the fixed status unit_limit: class (3), a method failure on every arm (written, no output).  Preregistration revision 1c item 1 (乙):
                      only a RuntimeMaxSec timeout without any OOM evidence is unit_limit; any OOM evidence is written infra_failure by the runner;
  wrapper_exception   our wrapper's code exception: the first one is NOT written (stop, fix, rerun); written (no output, class (2)) only after a committed
                      resume of this cell (holdout_common, kind resume) whose run ended in a wrapper exception again (alignment item 3);
  implementation_error  the method arm's hx_implementation_error (C22.5-6): written (no output, imputed as identity) only with a committed no_resume record of
                      this cell whose reason states BZ's (甲) (alignment item 3; preregistration revision 1c item 3); (乙) is a resume, the cell is rerun;
  infrastructure      the fixed status infra_failure: written (no output) only after a committed resume whose run failed again, or with a committed no_resume
                      record (未批补跑) (revision 1c item 3);
  arm stopped         no terminal record, and the arm's committed stop record (P19, stop_record) lists the pair among its unrun pairs (F1/F2: p = 1, 不判);
  not_written         admission / provenance refusals (P20), code_not_clean, list errors, machine-gate refusals, the fixed status needs_lead_rejudgment and
                      every status the table does not map: the evaluation stops before any truth (nothing is evaluated while one cell is not written).
The resume and no_resume records are holdout_common's (experiments/holdout/holdout_common.py: find_authorizations, load_authorization; schema
bayesreg.<tool>.<kind>.v1 in splits/bayesreg_v1/, committed, written by BZ or "lead on BZ's behalf" with BZ's words, the time and a reason; the alignment item
3: no other kind).  This entry adds: a resume's previous record (if any) is kept, hashed at its sha256 and is not the terminal record now (review2 m1); a
no_resume's record is there at its sha256 with the note 未批补跑; both under the arm's authorization.records directory, never under DATA_ROOT (review2 m2).
PQ (contract C22.H, revision 7, BZ 2026-11-09) is the method arm's added output, not an arm: its files are found through PQ's sha256 list
(pq_outputs_manifest.txt, named with its sha256 by runs/holdout_B_v1/pq_outputs_manifest.sha256 and released by runs/holdout_B_v1/pq_release.md in one
commit; alignment item 5), never by a path template; a missing PQ never stops F1-F3.  The PQ acceptance (C22.H7-1) is computed in this same unit, per pair,
from the truth arrays this entry read (pq_pair_acceptance, experiments/pq's functions called as they are); Ev1 and Ev2 (pq.json's sampling.d_sha256) are
recorded.  pq_dev_truth, pq_pair_acceptance and pq_b_class are the three functions PQH's replay gate calls (alignment item 6).

The manifest (D108; the lead's change 5, preregistration revision 1c item 1): MANIFEST_REL holds the input lines only (parse_manifest: header lines naming
the preregistration, arm table and unseal sha256 (and, only when BZ chose (ii), the record of that choice), then "<64 hex>  <absolute path>" for exactly the
files this entry reads, each hashed again just before the read).  The record-only lines (deleted files, movement records, derived inputs, spy records, the
step-8 input lock, the annex of superseded refusals of C23.12-6) are in RECORDS_REL, committed in the same commit; this entry checks only git's view of that
file and never reads it.  Nothing outside the manifest is read (the truth files excepted); a difference stops the evaluation, keeping what it wrote.

Certificates (preregistration §3, D112 (a); the lead's rulings of 11-09): every arm gets the Jacobian-only certificate in BL1's configuration (tile 8192) on
its evaluated field; the method arm also the decision configuration's certificate (configuration C, tile 65536) on output/final.npz's u, always, recorded only
(the method's certificate statistics are this one).

Code identity (the global rule, decision_log df8541d; review2 N1, m3; A5b: contract revision 9 C24.3): rule one first (rule_one:
bayesreg.io.code_identity.unit_identity_problems -- P1 from the committed runs/holdout_B_v1/pinned_code.json, every commit of P1..HEAD only adding
authorizations and records and appending to the decision log); then at (0) rule two for each gate (code_identity_problem: gate_identity_problems from the gate's
code_commit -- only the four decision constants' values and the allowlist's one allowance may change in the code), and for the truth gate its code_commit the
commit it ran on (the result's code.head, clean, the gate lock's first line); the clean-tree check covers CODE_DIRS (bayesreg, experiments, jd_curl, configs,
scripts) and splits; the modules imported from experiments/ and scripts/ (PQ's, holdout_common, evaluate_l2r_style) must come from this code tree.
A5b also (runs/20261110-a5/实现说明_A5b.md): the method cell through HB's finalized record and the move records (method_cross_check); the step-5 (甲) and a
withdrawal (no_resume of arm review, its record on /home under REVIEW_ROOT); P20 (the method's hx_provenance_rejected, and a cell of another arm refused before its
lock, by BZ's no_resume naming P20; class bz_no_rerun); the rejudgment records of the external and configuration C arms (bayesreg.io.rejudgment); XB's own stop
record checked by its sha256; BZ's (ii) only at PQ_METHOD_SIDE_REL; --out-dir a new directory under EVAL_DIR; the repair ablation (check_ablation before any
truth, evaluate_ablation after it: described only, never certified, not an arm).

It never returns to the methods (C23.12-6): no output selects a pair, triggers a fallback or reruns an arm; the certificates computed here are recorded only.
Outputs: --out-dir (required, new, never inside an evaluated cell, a PQ directory, the stage, a watched root or a seq_vp_3d tree; nothing is overwritten):
<out>/pairs/<pair>.json once per pair, <out>/holdout_eval.json at the end; the decision code (bayesreg.eval.holdout_decide) reads them.

SG (C23.2-C23.4, merged in ecf4621): (a) calls bayesreg.io.unseal.load_unseal and bind_unseal (this file's admit_holdout is on the call-site list of
C23.9-12) and bayesreg.io.guard.check_subject; the sealed lists are the guard's copy (bayesreg.io.sealed.guard_sealed_lists).  read_truth and
_load_label_volume are the E rows of the census of C23.6-3.

The spy (C23.6-5): open_holdout first checks bayesreg.io.sealed.sealed_cache_is_empty() and refuses otherwise, then records every file opened under the
watched places -- DATA_ROOT, the stage, every arm's record and field roots, PQ's two roots and the files of the sealed lists (read by the guard; each read
recorded) -- through a Python audit hook.  After (c) whatever it saw must be an input of the manifest or a sealed list; from then on an open there of
anything else (a truth file of an admitted subject excepted) is refused at the open (the evaluation stops).  The record is written into holdout_eval.json.
"""
import os

if __name__ == "__main__":
    # as bayesreg.eval.cli (EVF): command-line defaults only, before numba is imported; importing this module changes no environment variable
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("NUMBA_NUM_THREADS", "2")

from dataclasses import dataclass                                          # noqa: E402
from datetime import datetime, timedelta, timezone                         # noqa: E402
import hashlib                                                             # noqa: E402
import io                                                                  # noqa: E402
import json                                                                # noqa: E402
from pathlib import Path                                                   # noqa: E402
import re                                                                  # noqa: E402
import subprocess                                                          # noqa: E402
import sys                                                                 # noqa: E402
import time                                                                # noqa: E402
from types import MappingProxyType                                         # noqa: E402

import numpy as np                                                         # noqa: E402

from jd_curl.three_d.nifti import file_sha256, load_scalar                 # noqa: E402

from bayesreg.io import rejudgment as rejudgment_io                        # noqa: E402
from bayesreg.io import sealed as sealed_module                            # noqa: E402
from bayesreg.io.code_identity import gate_identity_problems, unit_identity_problems      # noqa: E402
from bayesreg.io.guard import SubjectSealed, check_subject, is_hex64, load_dev_allowlist, strict_json_object      # noqa: E402
from bayesreg.io.pair import _read_manifest                                # noqa: E402
from bayesreg.io.sealed import PAPER1_LISTS, SealedListError, guard_sealed_lists, lists_of, sealed_cache_is_empty      # noqa: E402
from bayesreg.io.unseal import bind_unseal, load_unseal                    # noqa: E402
from bayesreg.solve.backend import CERT_TILE_SIZE                          # noqa: E402

from . import cert_eval, metrics                                           # noqa: E402
from .fields import array_sha256, d_to_u, describe_run, load_field, read_disp, read_final_native      # noqa: E402

SCHEMA_PAIR = "bayesreg.holdout.pair.v1"
SCHEMA_RUN = "bayesreg.holdout.eval.v1"
SCHEMA_ARMS = "bayesreg.holdout.arm_table.v3"                # v3: the interface alignment of 11-09 (record and field places apart, holdout_common's records)
SCHEMA_MANIFEST = "bayesreg.holdout.eval_manifest.v3"         # v3: PQ's release is read from the code tree (alignment item 5)
SCHEMA_REEVAL = "bayesreg.holdout.reeval.v1"
SCHEMA_STOP = "bayesreg.holdout.arm_stop.v1"
SCHEMA_PQ_METHOD_SIDE = "bayesreg.holdout.pq_method_side.v1"  # BZ's committed choice (ii) after a method-side control of PQ failed (C22.H4-2, C22.H7-3 row 1)
SCHEMA_GATE = "bayesreg.holdout.truth_gate.v1"

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path("/home/rmcfz/seq_vp_tag_results")
# holdout_common's record root (experiments/holdout/holdout_common.HOLDOUT_ROOT, merged with XB); the evaluation's locks are below it.
HOLDOUT_RESULTS = RESULTS/"holdout_B_v1"
EVAL_DIR = HOLDOUT_RESULTS/"eval"
MASTER_LOCK = EVAL_DIR/"holdout_eval.lock"            # D109: one master lock, a fixed path, never derived from an option; O_EXCL, never edited or deleted
PAIR_LOCK_DIR = EVAL_DIR/"pair_locks"                 # D109: one lock per pair (<pair>.lock; a reevaluation <pair>.reeval.lock, at most once)
DATA_ROOT = Path("/mnt/d/data")                       # C23.12-4: the data root of the data preparation (<subject>/aligned_seg35.nii.gz); a constant
LABEL_FILES = MappingProxyType({"seg35": "aligned_seg35.nii.gz", "seg4": "aligned_seg4.nii.gz"})     # = bayesreg.seg.unet.data.LABEL_FILES (tested)
READ_ONLY_ROOTS = (Path("/home/rmcfz/code/python_projects/seq_vp_3d"), Path("/mnt/d/seq_vp_3d_results"))
CODE_DIRS = ("bayesreg", "experiments", "jd_curl", "configs", "scripts")     # the global rule (decision_log df8541d): the code identity
CLEAN_DIRS = CODE_DIRS+("splits",)                    # the clean-tree check: the code and the committed authorizations (review2 N1)

PAIRS_REL = "splits/bayesreg_v1/holdout_B_pairs.tsv"
PAIRS_SHA256 = "8088916b103a4ddd8b5589164a62ea6048bac1c1799fd367f1deec154151a9ee"   # C23.12-2 (a)
N_PAIRS = 50
PAIR_NAME = "{fixed}__{moving}"                       # the stage's pair directory (the development stages' form)
PAIR_SET = "holdout"                                  # holdout_common's pair set of the hold-out cells
UNSEALED_LIST = "holdout_B_v1"
NEVER_UNSEALED = ("H1", "H2", "H3", "H4")
UNSEAL_REL = "splits/bayesreg_v1/unseal_holdout_B_v1.json"     # C23.3-1
ALLOWLIST_REL = "configs/bayesreg/dev_subjects.json"
PREREG_REL = "docs/bayesreg/notes/留出集预登记_20261107.md"
PREREG_SHA256 = "df7b497887dc7b8290493c040635edf7334e9ae3be584e31a89177080836f493"     # P0 (11-14): the BZ-adopted preregistration (revisions 1-2, adopted 11-10) as committed; None or another file: the entry refuses at (b)
ARM_TABLE_REL = "configs/frozen/holdout_B_v1_arm_table.json"
ARM_TABLE_SHA256 = "69cb1ddf3e2006116c3d2237f3a4912e68a7238f2ce813f7d67723e5eeb8d661"     # P0 (11-14): the arm table of revision 2 (its sha256 is also written in the preregistration); None or another file: refused at (b)
MANIFEST_REL = "runs/holdout_B_v1/eval_manifest.txt"     # D108: the inputs only (revision 1c item 1; the lead's path, 11-07); read and checked here
RECORDS_REL = "runs/holdout_B_v1/eval_records.txt"       # the record-only lines, the same commit; never read here (git state only)
REEVAL_DIR_REL = "splits/bayesreg_v1"
REEVAL_NAME = re.compile(r"holdout_B_v1_reeval_([0-9]{1,3})\.json", re.ASCII)
REEVAL_WRITERS = ("BZ", "lead on BZ's behalf")
AUTHORIZATION_WRITERS = REEVAL_WRITERS                # PQ's method side (ii); holdout_common's records check their own (RESUME_WRITERS, the same two)
STOP_WRITERS = REEVAL_WRITERS+("lead",)               # the P19 stop record goes to the lead (preregistration 4.5)
REEVAL_REASONS = ("evaluation_killed", "evaluation_oom", "io_error")       # D109: infrastructure failures of the evaluation itself only (review m18)
SEG4_MAP_REL = "configs/bayesreg/seg4_of_seg35_v1.json"
SEG4_MAP_SHA256 = "273743b9b396668d8f0539da33008f2a163b1d659ce64245f24314795c22fa4e"    # G of C22.Q2-6 (the lead's change 4)
EXCLUDED_ARMS = ("transmorph", "vpreg", "synthmorph")    # not in the arm table (review I2; preregistration 2.5)
RESERVED_NAMES = ("eval", "pq", "review", "work", "stage", "atropos", "locks", "batches", "prerepair")     # holdout_common.RESERVED and this entry's own directories

# ---- the method arm's step-5 records, the move to D: (A5b items 5, 8; the interface alignment, revisions 1 and 2; HB 实现说明 §6.1) ------------------------------
# The method cell is known from the finalized record's `cell` and the move records (move.json, or its ok rows with a passed move_resume.json), never by a path template alone: every file this
# entry reads in the method cell on D: is the `dst` of exactly one ok move row whose sha256_after is the finalized record's sha256 of that file (and the manifest's).
REVIEW_ROOT = HOLDOUT_RESULTS/"review"                # hb_core.review_root(): finalized_outputs.json, and <pair>/ the step-5 records kept on /home (hb_review.home_records)
FINALIZED_NAME = "finalized_outputs.json"
FINALIZED_SCHEMA = "bayesreg.hb.finalized_outputs.v1"
MOVE_NAME, MOVE_RESUME_NAME = "move.json", "move_resume.json"      # in the method arm's record directory (its authorization.records): holdout_common.move_files' records
MOVE_SCHEMA = "bayesreg.holdout.move.v1"
REVIEW_ARM = "review"                                 # holdout_common's arm name of the step-5 records (hb_review: (甲) of a step-5 hx_implementation_error, a withdrawal)
P20_WORD = rejudgment_io.P20_WORD                      # a no_resume of a P20 cell names P20 in its reason (prereg revision 2 items 17 (f), 18; bayesreg.io.rejudgment.p20_reason_problem)
REFUSAL_COPIES = "refusals"                           # holdout_common.REFUSAL_COPIES (A5a) / hb_configc.keep_refusals: <record dir>/refusals/<batch>__<name>, <copy>.sha256 beside
# ---- the repair ablation (prereg revision 2 item 4; contract revision 9 draft C24.2; A5b item 3): described only ----------------------------------------------------
PREREPAIR_ROOT = HOLDOUT_RESULTS/"prerepair"          # <pair>/prerepair.json: experiments/hb/hb_prerepair.py's record (the three checks, the inputs' and the output's sha256, the commit)
PREREPAIR_BIG = Path("/mnt/d/seq_vp_tag_holdout_B_v1/prerepair")      # <pair>/prerepair_disp.npz: u_to_d(boundary_taper(field_0600's u)), key d, read as a plain displacement
PREREPAIR_RECORD, PREREPAIR_DISP = "prerepair.json", "prerepair_disp.npz"
PREREPAIR_SCHEMA = "bayesreg.hb.prerepair.v1"
PREREPAIR_CHECKS = ("field_sha256", "u_sha256", "tapered_u_sha256")
ABLATION_KEYS = ("seg35", "seg4", "sdlogj_official")   # the numbers reported (after = the method arm's output; difference after - before)
# ---- the P19 stop: XB's own record beside the committed one (prereg revision 2 item 11; the lead's ruling (5), 1:18 AM 11-10) -------------------------------------
XB_STOP_NAME = "arm_stopped.json"                     # xb_core: <record root>/<arm>/arm_stopped.json
XB_STOP_SCHEMA = "bayesreg.xb.arm_stop.v1"
XB_STOP_FIELDS = ("arm", "pair_set", "utc", "suspicious_pairs", "at_pair", "rule")     # copied as they are into runs/holdout_B_v1/arm_stop_<arm>.json
PQ_METHOD_SIDE_REL = "runs/holdout_B_v1/pq_method_side_decision.json"      # BZ's (ii): this path and no other (the interface alignment revision 1 item 3)

# ---- PQ's outputs and release (alignment item 5; PQH's pqh_units.RELEASE_SHA_REL / RELEASE_NOTE_REL) ---------------------------------------------------
PQ_RELEASE_SHA_REL = "runs/holdout_B_v1/pq_outputs_manifest.sha256"      # one line: "<sha256>  <absolute path of pq_outputs_manifest.txt>"
PQ_RELEASE_NOTE_REL = "runs/holdout_B_v1/pq_release.md"                  # one line "放行：<sha256>" (experiments/pq/pq_eval.release_note_problem's rule)
PQ_LIST_NAMES = ("pq_outputs_manifest.txt", "pq_outputs_manifest.resume.txt")
PQ_D_SHA256_KEY = "sampling.d_sha256"                 # Ev2: pq.json's d (float64, C-order; = output/line.json's output.disp.d_sha256)
PQ_FILES = MappingProxyType({"pq.json": "small_root", "pq.npz": "big_root", "baselines.npz": "big_root"})     # C22.H6-1: small files and big files
PQ_NOTES = MappingProxyType({"pq_failed.json": "pq_failed", "pq_absent.json": "pq_absent"})                  # C22.H6-3: listed, never read
# The step-8 input lock (preregistration 5.2 step 8; HB's experiments/hb/hb_store.py input-lock, DEFAULT_LOCK_REL and LOCK_SCHEMA): the 100 subjects' HI outputs
# and the method arm's field_0600.npz by sha256, a record committed under runs/holdout_B_v1/.  C22.H4-2 Hm2 (revision 7a m2): pq.json's input sha256 are
# checked against it after the lock -- by this entry (the lead's ruling (10) on the PQH review I9, decision_log 2026-11-09 ~10:30 PM CST).
INPUT_LOCK_REL = "runs/holdout_B_v1/hb_input_lock.json"
INPUT_LOCK_SCHEMA = "bayesreg.hb.input_lock.v1"
PQ_INPUT_KEYS = MappingProxyType({"probs36.npz": "probs36_sha256", "structure_groups.npz": "structure_groups_sha256", "provenance.json": "provenance_sha256"})

GATE_UNIT_PATTERNS = ("seq-vp-*", "tag-*")
OWN_UNITS = ("tag-hb-eval",)
GATE_OWN_UNITS = ("tag-ev-truthgate",)
PROC_CGROUP = Path("/proc/self/cgroup")
EARLIEST_UTC = datetime(2026, 10, 30, tzinfo=timezone.utc)
UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
UTC_FORM = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", re.ASCII)

# ---- the truth-reader gate (C23.12-10) ----------------------------------------------------------------------------------------------------------------
DEV_STAGE = "stage_L2R_val_v1"
DEV_SUBJECTS = tuple(f"OASIS_OAS1_{n:04d}_MR1" for n in range(438, 458))
DEV_PAIRS = tuple((DEV_SUBJECTS[i], DEV_SUBJECTS[i+1]) for i in range(19))
FOLDS_REL = "splits/bayesreg_v1/folds.json"
DEV_DATA_ROOT = Path("/home/rmcfz/code/python_projects/seq_vp_tag")/"data"/"three_d"     # the main checkout's data/ link (holdout_common.MAIN_REPO): a worktree has none
TRUTH_GATE_PLAN_REL = "runs/20261107-ev/truth_gate_plan.md"
TRUTH_GATE_PLAN_SHA256 = "eb76887ec71ca3626a6af600e78d87953ec560f02f82d3be58798d20178ba302"     # P0 (11-14): the final plan (PT 5ef8243); None or another plan: the gate does not run
TRUTH_GATE_OUT = RESULTS/"ev"/"truth_gate_v1"
TRUTH_GATE_LOCK = RESULTS/"ev"/"truth_gate.lock"      # once (O_EXCL, fixed path)
GATE_READING_PASS = "逐位相同 20/20（seg35、seg4）；指标逐位相同 19/19"
GATE_READING_FAIL = "不过（交 BZ）"
GATE_READING_UNDECIDED = "不判（交 BZ）"
# Filled by a reviewed commit after the gate ran and read GATE_READING_PASS (C23.12-10), with exactly DECISION_KEYS; None -> (0) refuses.
TRUTH_READER_DECISION = {
    "result_json": "/home/rmcfz/seq_vp_tag_results/ev/truth_gate_v1/truth_gate.json",
    "result_sha256": "3abeee225a5e27dc1e8ad31a6017b804775d6d8487162d7be07b6c750a131eb0",
    "reading": "逐位相同 20/20（seg35、seg4）；指标逐位相同 19/19",
    "code_commit": "318e72ad3d097bcbd03db180a56c0144ee4344bc",
    "environment": {"numpy": "2.0.2", "nibabel": "5.4.0"},
    "counts": {"reader": {"subjects": 20, "identical": 20}, "metrics": {"pairs": 19, "identical": 19}},
    "plan_sha256": "eb76887ec71ca3626a6af600e78d87953ec560f02f82d3be58798d20178ba302",
    "decided_by": "lead (total responsible), 2026-11-15, per truth_gate_plan.md §3, §4",
    "review": "docs/bayesreg/reviews/D58_复核_20261115.md (sha256 9629f2495a4f099347871e4eb3d606a22f2484838745e65e494fe1145f946c4b; 0 blocking, 1 important, 8 minor; reading holds)",
}     # P1 (11-15): filled after the gate ran on P0'' 318e72ad (C4, D7); code_commit = truth_gate.json code.head (clean) = the lock's first code_head
DECISION_KEYS = ("result_json", "result_sha256", "reading", "code_commit", "environment", "counts", "plan_sha256", "decided_by", "review")
DECISION_ENVIRONMENT_KEYS = ("numpy", "nibabel")
GATE_COUNTS = MappingProxyType(dict(reader=dict(subjects=20, identical=20), metrics=dict(pairs=19, identical=19)))
DECISION_TILE = 65536            # the decision configuration's certificate (configuration C, C22.4-1): the method arm's D112 (a) certificate


class HoldoutRefused(Exception):
    """Refused before any truth label (or, for stage 'manifest_changed', before the next one) is read; `stage` names the condition."""
    status = "holdout_refused"

    def __init__(self, stage, message):
        super().__init__(f"{stage}: {message}")
        self.stage = stage


class ManifestChanged(HoldoutRefused):
    """A listed file differs from its manifest sha256 when it is about to be read: the evaluation stops; what it wrote is kept."""


class TruthMismatch(HoldoutRefused):
    """A truth volume's grid differs from the pair manifest's, or its labels are not finite non-negative integers (C23.12-4 "不同就停"; review m3): the
    evaluation stops as for ManifestChanged (exit 4, to BZ); a reevaluation (D109) cannot repair it."""


def utc():
    return time.strftime(UTC_FORMAT, time.gmtime())


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw, where):
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=strict_json_object)
    except (UnicodeDecodeError, ValueError) as exc:
        raise HoldoutRefused("record", f"{where}: not UTF-8 JSON without duplicate keys ({exc})") from None


def _inside(path, root):
    resolved, root = Path(path).resolve(), Path(root).resolve()
    return resolved == root or root in resolved.parents


def _under_read_only(path):
    return any(_inside(path, root) for root in READ_ONLY_ROOTS)


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (tuple, set, frozenset)):
        return list(value)
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, MappingProxyType):
        return dict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)}")


def dumps(document):
    """bayesreg.solve.export.dumps (the cli's writer: non-finite values as null), with the extra types of this module made plain first."""
    from bayesreg.solve.export import dumps as export_dumps
    return export_dumps(json.loads(json.dumps(document, default=json_default, allow_nan=True)), indent=1)


def write_once(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as stream:
        stream.write(dumps(document)+"\n")
    return path


def take_lock(path, line, stage="once"):
    """O_EXCL: written once, after every check (hi_core.take_lock); HoldoutRefused(stage) if it is there."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise HoldoutRefused(stage, f"{path} exists: evaluated once (D109); nothing read") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(json.loads(json.dumps(line, default=json_default)), ensure_ascii=False, sort_keys=True)+"\n")
    return line


def lock_lines(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---- git: committed and unchanged (hi_core.tracked_unchanged, literal pathspecs) ----------------------------------------------------------------------
def _git(repo, *args):
    try:
        # core.quotePath=false: the preregistration's file name is not ASCII and must come back as itself (ls-files would quote it)
        return subprocess.run(["git", "--literal-pathspecs", "-c", "core.quotePath=false", "-C", str(repo), *args], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(args, 127, "", f"{type(exc).__name__}: {exc}")


def tracked_unchanged(path, stage, repo=None):
    """The file is inside the code tree, tracked and without any change against HEAD; nothing of it is read here.  HoldoutRefused(stage) otherwise."""
    root = Path(REPO if repo is None else repo).resolve()
    target = Path(path).resolve()
    if not target.is_relative_to(root):
        raise HoldoutRefused(stage, f"{path} is not inside the code tree {root}")
    rel = target.relative_to(root).as_posix()
    tracked = _git(root, "ls-files", "--error-unmatch", "--", rel)
    changed = _git(root, "status", "--porcelain", "--ignored", "--untracked-files=all", "--", rel)
    if tracked.returncode != 0 or changed.returncode != 0 or changed.stdout.strip() or tracked.stdout.strip() != rel:
        raise HoldoutRefused(stage, f"{rel}: not a committed, unchanged file of the code tree (ls-files exit {tracked.returncode}; status {changed.stdout.strip()!r})")
    last = _git(root, "log", "-1", "--format=%H", "--", rel).stdout.strip()
    return dict(path=rel, tracked=True, unchanged=True, last_commit=last or None)


def read_committed(rel, stage, repo=None):
    """(raw bytes, sha256, git record) of a committed, unchanged file of the code tree."""
    root = Path(REPO if repo is None else repo)
    record = tracked_unchanged(root/rel, stage, root)
    raw = (root/rel).read_bytes()
    return raw, sha_bytes(raw), record


def code_record(repo=None):
    """The code tree's state: HEAD and whether CLEAN_DIRS (the global rule's code directories bayesreg, experiments, jd_curl, configs, scripts, and splits) have
    no change against HEAD, untracked files included (review2 N1)."""
    root = Path(REPO if repo is None else repo)
    status = _git(root, "status", "--porcelain", "--untracked-files=all", "--", *CLEAN_DIRS)
    import nibabel
    import scipy
    return dict(root=str(root), head=_git(root, "rev-parse", "HEAD").stdout.strip() or None, clean=status.returncode == 0 and not status.stdout.strip(),
                status=status.stdout, dirs=list(CLEAN_DIRS), python=sys.version.split()[0], numpy=np.__version__, nibabel=nibabel.__version__, scipy=scipy.__version__)


def code_identity_problem(commit, repo=None):
    """Rule two of a gate (contract revision 9 draft C24.3-6; prereg revision 2 item 9 step 8; decision_log 2026-11-10 ~1:53 AM CST B1, ~2:45 AM CST N3; A5b item 1): `commit` (the gate's
    code_commit) is 40 hex and bayesreg.io.code_identity.gate_identity_problems(commit) holds -- from the gate's commit to HEAD the code (the five code directories, the root's importable files,
    new top-level directories) changes only by the four decision constants' values (judged on the AST, each in its own file) and by the allowlist's one allowance; outside the code nothing is
    checked.  None, or the problem."""
    root = Path(REPO if repo is None else repo)
    if not (isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit, re.ASCII)):
        return f"code_commit {commit!r}: 40 lowercase hex required"
    _, problems = gate_identity_problems(commit, repo=root)
    if problems:
        return f"the code tree is not the gate's commit {commit} by rule two (only the decision constants' values and the allowlist's allowance may change in the code): {problems[:5]}"
    return None


def rule_one(repo=None):
    """Rule one for this process (contract revision 9 draft C24.3-4, -7; decision_log 2026-11-10 ~2:59 AM CST (7); A5b item 1): bayesreg.io.code_identity.unit_identity_problems on the code tree --
    P1 from the committed runs/holdout_B_v1/pinned_code.json (never HEAD), every commit of P1..HEAD only adding authorizations and records and appending to the decision log, the code P1's, the
    working tree clean.  Called before any subject file is read.  Returns the record; HoldoutRefused('code_identity') otherwise."""
    root = Path(REPO if repo is None else repo)
    record, problems = unit_identity_problems(repo=root)
    if problems:
        raise HoldoutRefused("code_identity", f"rule one does not hold (C24.3-4): {problems[:5]}; nothing is read")
    return json.loads(json.dumps(record, default=json_default))


def utc_problem(value, key):
    if not (isinstance(value, str) and UTC_FORM.fullmatch(value)):
        return f"{key} {value!r}: YYYY-MM-DDTHH:MM:SSZ required"
    try:
        when = datetime.strptime(value, UTC_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return f"{key} {value!r}: not a real time"
    if not EARLIEST_UTC <= when <= datetime.now(timezone.utc)+timedelta(minutes=10):
        return f"{key} {value!r}: before the draw of the hold-out or in the future"
    return None


# ---- the machine gate (C23.12-2 (b); hi_core.machine_gate, units only: this is a CPU evaluation) -------------------------------------------------------
def units_of_this_process(cgroup=PROC_CGROUP, units=OWN_UNITS):
    try:
        text = Path(cgroup).read_text(encoding="utf-8")
    except OSError:
        return []
    return [u for u in units if re.search(rf"/{re.escape(u)}\.service(/|$)", text, re.MULTILINE)]


def machine_gate(*, run=None, cgroup=PROC_CGROUP, own_units=OWN_UNITS):
    """No active or activating user unit of GATE_UNIT_PATTERNS but the one this process runs in (excluded only when /proc/self/cgroup shows this process inside
    it).  A query that fails refuses.  HoldoutRefused('machine') otherwise."""
    run = subprocess.run if run is None else run
    own = units_of_this_process(cgroup, own_units)
    record = dict(check="hold-out evaluation machine gate (C23.12-2 (b), hi_core.machine_gate)", unit_patterns=list(GATE_UNIT_PATTERNS), own_units=own, checked_utc=utc())
    try:
        units = run(["systemctl", "--user", "list-units", "--state=active,activating", "--no-legend", "--plain", *GATE_UNIT_PATTERNS], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise HoldoutRefused("machine", f"the machine could not be queried ({type(exc).__name__}: {exc})") from None
    if units.returncode != 0:
        raise HoldoutRefused("machine", f"systemctl --user list-units failed ({units.stderr.strip()})")
    active = sorted(line.split()[0] for line in units.stdout.splitlines() if line.strip())
    others = [u for u in active if u.removesuffix(".service") not in own]
    record.update(active_units=active, other_units=others)
    if others:
        raise HoldoutRefused("machine", f"other units are active ({others}): a cell may still be running; the evaluation runs alone")
    record["passed"] = True
    return record


# ---- the spy (C23.6-5) -------------------------------------------------------------------------------------------------------------------------------------
class SpyRefused(PermissionError):
    """An open the spy does not allow (after (c)): raised at the open itself, so nothing is read."""


class OpenedFiles:
    """Every open (Python audit event "open") of a file under the watched directories or of a sealed list, in order, with the phase.  `allowed` is None
    while it only records (stage 1 up to (c)); afterwards an open of a watched file that is not allowed raises SpyRefused."""

    def __init__(self, dirs, sealed_files):
        self.dirs = []
        self.watch(dirs)
        self.sealed = {os.path.realpath(p) for p in sealed_files}
        self.events, self.refused, self.allowed, self.truth, self.phase = [], [], None, frozenset(), "admission"

    def watch(self, dirs):
        """Each directory by its resolved and its unresolved absolute path (review m17: a subject directory that is a link elsewhere is still watched)."""
        for d in dirs:
            for form in (os.path.realpath(d), os.path.abspath(d)):
                if form not in self.dirs:
                    self.dirs.append(form)

    def watched(self, path):
        return path in self.sealed or any(path == d or path.startswith(d+os.sep) for d in self.dirs)

    def saw(self, path, mode):
        raw = os.fsdecode(path)
        path = os.path.realpath(raw)
        if not (self.watched(path) or self.watched(os.path.abspath(raw))):
            return
        kind = "sealed_list" if path in self.sealed else "truth" if path in self.truth else "input"
        self.events.append(dict(path=path, mode=None if mode is None else str(mode), kind=kind, phase=self.phase))
        if self.allowed is not None and kind == "input" and path not in self.allowed:
            self.refused.append(path)
            raise SpyRefused(f"the evaluation entry's spy refuses {path}: not an input of the manifest, a sealed list or a truth file of the pair (C23.12-2 (c))")

    def allow(self, inputs):
        """After (c): every file seen so far is an input of the manifest or a sealed list; from now on only those (and truth files) may be opened."""
        self.allowed = frozenset(os.path.realpath(p) for p in inputs)
        stray = sorted({e["path"] for e in self.events if e["kind"] == "input" and e["path"] not in self.allowed})
        if stray:
            raise HoldoutRefused("spy", f"files outside the manifest were opened before (c): {stray[:5]}")
        self.phase = "evaluation"

    def record(self):
        return dict(watched_dirs=self.dirs, sealed_lists=sorted(self.sealed), events=self.events, refused=self.refused,
                    sealed_list_reads=sum(e["kind"] == "sealed_list" for e in self.events))


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
    """C23.6-5: the guard's copy of the sealed lists must be empty (bayesreg.io.sealed.sealed_cache_is_empty), so that its first read of the lists happens
    under the spy; the lists' files (bayesreg.io.sealed.REPO and SEALED_LISTS, read at call time) are watched and allowed, every read recorded."""
    global _SPY, _HOOKED
    if not sealed_cache_is_empty():
        raise HoldoutRefused("spy", "the sealed lists are already cached in this process: the spy would not see the guard read them (C23.6-5); start a new process")
    files = [Path(sealed_module.REPO)/spec["path"] for spec in sealed_module.SEALED_LISTS]
    _SPY = OpenedFiles(dirs, files)
    if not _HOOKED:
        sys.addaudithook(_audit)                                    # stays for the process; inactive while _SPY is None
        _HOOKED = True
    return _SPY


_LAST_SPY = None              # the spy of the last stop_spy (the tests read its events after a refusal; review m6)


def stop_spy():
    global _SPY, _LAST_SPY
    spy, _SPY = _SPY, None
    if spy is not None:
        _LAST_SPY = spy
    return spy


# ---- admission objects: issued here only, checked by identity (as C23.3-5) -----------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Admission:
    mode: str                 # "holdout" or "truth_gate"
    subjects: tuple
    pairs: tuple              # ((index, fixed, moving, name), ...)
    record: MappingProxyType


_ISSUED = {}                  # id -> admission (strong references; `is` comparisons only)
_STATE = {}                   # id -> {"locked": bool, "pair_locks": {name: lock path}}


def _issue(mode, subjects, pairs, record):
    admission = Admission(mode, tuple(subjects), tuple(pairs), MappingProxyType(dict(record)))
    _ISSUED[id(admission)] = admission
    _STATE[id(admission)] = dict(locked=False, pair_locks={})
    return admission


def _issued(admission):
    return type(admission) is Admission and _ISSUED.get(id(admission)) is admission


def _state(admission, stage="truth"):
    if not _issued(admission):
        raise HoldoutRefused(stage, "not an admission issued by bayesreg.eval.holdout in this process (copies, pickles and hand-made objects are refused)")
    return _STATE[id(admission)]


# ---- (0) the truth-reader gate's and the PQ replay gate's decisions ---------------------------------------------------------------------------------------
def _is_hex40(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value, re.ASCII) is not None


def _gate_ran_problems(result, decision, lock_path):
    """B4 review, 3rd plan, I1 (A5b item 2): the constant's code_commit is the commit the gate ran on -- the result's code.head, its code.clean true, and the code_head of the gate lock's first
    line (the lock the gate took once, O_EXCL) the same commit.  [problems]."""
    problems = []
    code = result.get("code") if isinstance(result.get("code"), dict) else {}
    if code.get("head") != decision["code_commit"] or code.get("clean") is not True:
        problems.append(f"the gate ran on {code.get('head')!r} (clean {code.get('clean')!r}), not the decision's code_commit {decision['code_commit']!r} on a clean tree")
    try:
        first = lock_lines(lock_path)[:1]
    except (OSError, ValueError) as exc:
        first = []
        problems.append(f"{lock_path}: not readable ({type(exc).__name__})")
    if not first or first[0].get("code_head") != decision["code_commit"]:
        problems.append(f"{lock_path}: its first line's code_head {(first[0] if first else {}).get('code_head')!r} is not the decision's code_commit {decision['code_commit']!r}")
    return problems


def _check_gate_constant(decision, *, name, stage, reading, counts, plan_sha256, result_path, schema, environment_keys, plan_rel, repo=None, lock_path=None):
    """One pinned conclusion of a development-set gate (the truth-reader gate, C23.12-10): not None; exactly DECISION_KEYS; the reading and counts the pass;
    the plan pinned and the same; the result file at its fixed path with its sha256, saying the same; the environment this one; decided_by and review
    non-empty (review m8); the code_commit the commit the gate ran on (the result's code.head, clean, the gate lock's first line; B4 review I1) and rule two
    against it (code_identity_problem: gate_identity_problems; A5b item 1).  HoldoutRefused(stage)."""
    if decision is None:
        raise HoldoutRefused(stage, f"{name} is None: the gate ({plan_rel}) has not been read and pinned; nothing is read")
    if not isinstance(decision, (dict, MappingProxyType)) or sorted(decision) != sorted(DECISION_KEYS):
        raise HoldoutRefused(stage, f"{name}: exactly the keys {list(DECISION_KEYS)} are required")
    problems = []
    if decision["reading"] != reading:
        problems.append(f"reading {decision['reading']!r} is not {reading!r}")
    got = decision["counts"]
    if not isinstance(got, (dict, MappingProxyType)) or dict(got) != dict(counts):
        problems.append(f"counts {got!r}: {dict(counts)} required")
    if plan_sha256 is None or decision["plan_sha256"] != plan_sha256:
        problems.append(f"plan_sha256 {decision['plan_sha256']!r} is not the pinned plan's {plan_sha256!r}")
    for key in ("decided_by", "review"):
        if not isinstance(decision[key], str) or not decision[key].strip():
            problems.append(f"{key}: a non-empty string required")
    problems += [p for p in (code_identity_problem(decision["code_commit"], repo),) if p]
    path = Path(str(decision["result_json"]))
    if path.resolve() != Path(result_path).resolve():
        problems.append(f"{path} is not the gate's result {result_path}")
    elif not is_hex64(decision["result_sha256"]) or not path.is_file() or file_sha256(path) != decision["result_sha256"]:
        problems.append(f"{path}: missing or not the sha256 {decision['result_sha256']!r}")
    else:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("schema") != schema or result.get("reading") != decision["reading"] or (result.get("plan") or {}).get("sha256") != decision["plan_sha256"]:
            problems.append(f"{path}: schema, reading or plan differ from the decision")
        if result.get("counts") != (dict(got) if isinstance(got, (dict, MappingProxyType)) else got):
            problems.append(f"{path}: counts {result.get('counts')} differ from the decision")
        if lock_path is not None:
            problems += _gate_ran_problems(result, decision, lock_path)
    environment = decision["environment"]
    import nibabel
    import scipy
    versions = dict(numpy=np.__version__, nibabel=nibabel.__version__, scipy=scipy.__version__)
    now = {k: versions[k] for k in environment_keys}
    if not isinstance(environment, (dict, MappingProxyType)) or sorted(environment) != sorted(environment_keys) or dict(environment) != now:
        problems.append(f"environment {environment!r} is not this one {now}")
    if problems:
        raise HoldoutRefused(stage, f"{name}: {problems}")
    return dict(json.loads(json.dumps(decision, default=json_default)), checked="result file, sha256, reading, counts, plan, environment, code identity, review")


def check_decision(decision=None, repo=None):
    """C23.12-2 (0): TRUTH_READER_DECISION is filled (a reviewed commit), its reading is GATE_READING_PASS with the two counts 20/20 and 19/19, the result
    file exists with its sha256 and says the same, it ran under the pinned plan, numpy and nibabel are the versions it ran with, its code commit has the
    code directories of HEAD.  HoldoutRefused('gate')."""
    return _check_gate_constant(TRUTH_READER_DECISION if decision is None else decision, name="TRUTH_READER_DECISION", stage="gate",
                                reading=GATE_READING_PASS, counts=GATE_COUNTS, plan_sha256=TRUTH_GATE_PLAN_SHA256, result_path=Path(TRUTH_GATE_OUT)/"truth_gate.json",
                                schema=SCHEMA_GATE, environment_keys=DECISION_ENVIRONMENT_KEYS, plan_rel=TRUTH_GATE_PLAN_REL, repo=repo, lock_path=TRUTH_GATE_LOCK)


# ---- the modules of experiments/ this entry calls (PQ's arithmetic, holdout_common's records): from this code tree only (review2 N1) --------------------
def check_module_origin(module, folder):
    """The module was imported from `folder` of this code tree (a module of the same flat name imported from elsewhere earlier in the process would be the
    cached one).  HoldoutRefused('code') otherwise."""
    origin = getattr(module, "__file__", None)
    if origin is None or not _inside(origin, folder):
        raise HoldoutRefused("code", f"the module {module.__name__} was imported from {origin}, not from {folder} of this code tree (the global rule)")
    return str(Path(origin).resolve())


def _experiment_module(directory, name):
    """experiments/<directory>/<name>.py of this code tree, imported by its flat name (the experiments/ convention), its origin checked."""
    import importlib
    folder = Path(REPO)/"experiments"/directory
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    return _checked(importlib.import_module(name), folder)


def _checked(module, folder):
    check_module_origin(module, folder)
    return module


def _pq_modules():
    """experiments/pq's pq_core, pq_eval, pq_stats, called as they are (C22.H8: the arithmetic is the development set's; never copied), and the scripts/
    module pq_core takes the warp from (evaluate_l2r_style), all from this code tree."""
    pq_core, pq_eval, pq_stats = (_experiment_module("pq", name) for name in ("pq_core", "pq_eval", "pq_stats"))
    _checked(sys.modules["evaluate_l2r_style"], Path(REPO)/"scripts")
    return pq_core, pq_eval, pq_stats


def _holdout_common():
    """experiments/holdout/holdout_common.py (XB's, merged): the authorization records (resume, rerun, no_resume) every hold-out tool shares."""
    return _experiment_module("holdout", "holdout_common")


def helper_origins():
    """The origins of the helper modules this entry imports, each checked to be of this code tree (recorded with the code identity): PQ's arithmetic, the
    warp it takes from scripts/, holdout_common and the hi_core it calls for git."""
    _pq_modules()
    hc = _holdout_common()
    _checked(hc._hi(), Path(REPO)/"experiments"/"hi")
    names = ("pq_core", "pq_eval", "pq_stats", "evaluate_l2r_style", "holdout_common", "hi_core")
    return {n: str(Path(sys.modules[n].__file__).resolve()) for n in names}


def replay_decision_problem(decision=None, *, repo=None):
    """experiments/pq/pqh_replay.replay_decision_problem's signature (alignment item 6: the entry calls PQH's function, no second pin).  PQH is not merged into
    this branch's base (73723d7), so this is a local stand-in: while experiments/pq/pqh_replay.py is not in the code tree it returns a problem (the entry
    refuses at (0)); once it is, PQH's function is called as it is (its pin: pqh_pins.PQ_REPLAY_DECISION).  None, or the problem."""
    if not (Path(REPO)/"experiments"/"pq"/"pqh_replay.py").is_file():
        return ("experiments/pq/pqh_replay.py is not in the code tree (PQH not merged): the development replay gate's decision (C22.H8, D119) cannot be checked; "
                "the evaluation entry does not run")
    return _experiment_module("pq", "pqh_replay").replay_decision_problem(decision, repo=repo)


def replay_decision(decision=None):
    """The replay gate's decision the code identity is checked against: `decision`, or pqh_pins.PQ_REPLAY_DECISION (PQH's one pin)."""
    if decision is not None:
        return decision
    if not (Path(REPO)/"experiments"/"pq"/"pqh_pins.py").is_file():
        return None
    return _experiment_module("pq", "pqh_pins").PQ_REPLAY_DECISION


def check_pq_replay(decision=None, repo=None):
    """C23.12-2 (0) of contract revision 7 (C22.H7-1 last item, C22.H8, D119 (a)): replay_decision_problem() is None, and rule two from the decision's code commit
    (code_identity_problem: gate_identity_problems; PQH's function checks it too, this entry again with its own repository); otherwise the whole entry refuses,
    nothing read (fail closed).  HoldoutRefused('pq_replay')."""
    problem = replay_decision_problem(decision, repo=repo)
    pinned = None
    if problem is None:
        pinned = replay_decision(decision)
        commit = pinned.get("code_commit") if isinstance(pinned, (dict, MappingProxyType)) else None
        problem = code_identity_problem(commit, repo)
    if problem is not None:
        raise HoldoutRefused("pq_replay", f"the PQ replay gate: {problem}; nothing is read")
    return dict(json.loads(json.dumps(pinned, default=json_default)), checked="pqh_replay.replay_decision_problem, code identity")


# ---- (a) unseal object, pairs, subjects --------------------------------------------------------------------------------------------------------------------
def load_pairs(repo=None):
    """The 50 rows of the pinned holdout_B_pairs.tsv: ((index, fixed, moving, name), ...), index from 0 in row order (preregistration 4.1)."""
    raw, digest, git = read_committed(PAIRS_REL, "pairs", repo)
    if digest != PAIRS_SHA256:
        raise HoldoutRefused("pairs", f"{PAIRS_REL}: sha256 {digest} is not the pinned {PAIRS_SHA256}")
    lines = [line for line in raw.decode("utf-8").splitlines() if line.strip()]
    if lines[0].split("\t") != ["fixed", "moving"]:
        raise HoldoutRefused("pairs", f"{PAIRS_REL}: header fixed<TAB>moving required")
    rows = []
    for index, line in enumerate(lines[1:]):
        fixed, moving = line.split("\t")
        rows.append((index, fixed, moving, PAIR_NAME.format(fixed=fixed, moving=moving)))
    if len(rows) != N_PAIRS or len({r[1] for r in rows} | {r[2] for r in rows}) != 2*N_PAIRS:
        raise HoldoutRefused("pairs", f"{PAIRS_REL}: {len(rows)} rows of distinct subjects, {N_PAIRS} expected")
    return tuple(rows), dict(path=PAIRS_REL, sha256=digest, git=git)


def check_pair_row(fixed, moving, rows):
    """The pair is one row of the list in this direction; HoldoutRefused('pairs') otherwise (a reversed pair is not on the list)."""
    for row in rows:
        if (row[1], row[2]) == (fixed, moving):
            return row
    reversed_ = any((row[2], row[1]) == (fixed, moving) for row in rows)
    raise HoldoutRefused("pairs", f"{fixed} <- {moving} is not a row of {PAIRS_REL}" + (" (it is a row in the other direction)" if reversed_ else ""))


def admit_subjects(subjects, unseal, bound):
    """This entry's own second lock (on the sealed list holdout_B_v1 and no other, H1-H4 never; one of the unseal object's subjects), then C23.4-2:
    bayesreg.io.guard.check_subject on the allowlist bind_unseal returned.  HoldoutRefused('admission') / SubjectSealed; only ids are used."""
    sealed = guard_sealed_lists()
    rows = []
    for subject in subjects:
        on = lists_of(sealed, subject)
        if set(on) & PAPER1_LISTS or any(name != UNSEALED_LIST for name in on):
            raise HoldoutRefused("admission", f"{subject} is on the sealed lists {list(on)}: only {UNSEALED_LIST} is evaluated here, H1-H4 never")
        if on != (UNSEALED_LIST,):
            raise HoldoutRefused("admission", f"{subject} is not on the sealed list {UNSEALED_LIST}")
        if subject not in tuple(unseal.subjects):
            raise HoldoutRefused("admission", f"{subject} is not one of the subjects of the unseal object")
        verdict = check_subject(subject, bound)
        if not verdict.admitted:
            raise SubjectSealed(f"{subject}: refused by bayesreg.io.guard.check_subject at step {verdict.step} ({verdict.item}): {verdict.message}")
        rows.append(dict(subject=subject, sealed_lists=list(on), admitted_by_unseal=verdict.admitted_by_unseal))
    return rows, sealed


def admit_holdout(*, repo=None):
    """Condition (a): the unseal object of bayesreg.io.unseal.load_unseal (the one pinned path, checked (i)-(ix)), bind_unseal on the pinned allowlist, the
    pinned pairs list, every subject through admit_subjects.  Returns (unseal object, bound allowlist, pairs, record); nothing of a subject is read."""
    root = Path(REPO if repo is None else repo)
    try:
        unseal = load_unseal(root/UNSEAL_REL)
    except (SubjectSealed, SealedListError, OSError, ValueError) as exc:
        raise HoldoutRefused("unseal", f"no valid unseal object ({type(exc).__name__}: {exc}); nothing is read") from None
    if unseal.sealed_list != UNSEALED_LIST:
        raise HoldoutRefused("unseal", f"the unseal object unseals {unseal.sealed_list!r}; this entry evaluates {UNSEALED_LIST} only")
    allowlist = load_dev_allowlist(root/ALLOWLIST_REL)
    try:
        bound = bind_unseal(allowlist, unseal)
    except SubjectSealed as exc:
        raise HoldoutRefused("unseal", f"bind_unseal refused ({exc}); nothing is read") from None
    pairs, pairs_record = load_pairs(root)
    subjects = [s for row in pairs for s in (row[1], row[2])]
    rows, sealed = admit_subjects(subjects, unseal, bound)
    record = dict(unseal=unseal.record(), allowlist_sha256=allowlist.sha256, pairs=pairs_record, subjects=rows, sealed_lists=sealed.record)
    return unseal, bound, pairs, record


# ---- (b) preregistration, arm table, cells ---------------------------------------------------------------------------------------------------------
def check_prereg(repo=None):
    if PREREG_SHA256 is None:
        raise HoldoutRefused("prereg", "PREREG_SHA256 is None: the BZ-adopted preregistration (after revision 2) is not pinned; nothing is read")
    raw, digest, git = read_committed(PREREG_REL, "prereg", repo)
    if digest != PREREG_SHA256:
        raise HoldoutRefused("prereg", f"{PREREG_REL}: sha256 {digest} is not the pinned {PREREG_SHA256}")
    return raw, dict(path=PREREG_REL, sha256=digest, git=git)


_SAFE_ARM = re.compile(r"[a-z0-9][a-z0-9_]{0,63}", re.ASCII)
_TEMPLATE_FIELDS = re.compile(r"\{([a-z_]+)\}")
FAMILIES = ("method", "main", "descriptive", "tier_90s")      # tier_90s: the 90 s tier of preregistration 2.6 (described only, F3)
CLASS_KEYS = ("output", "method_failure", "unit_limit", "wrapper_exception", "implementation_error", "infrastructure", "not_written")
# Terminal statuses every arm's runner writes with one meaning (the lead, 11-07, revision 1b item 6); the arm table cannot map them otherwise.
FIXED_STATUSES = MappingProxyType({"unit_limit": "unit_limit",                    # class (3): RuntimeMaxSec without OOM evidence (revision 1c item 1, 乙)
                                   "infra_failure": "infrastructure",              # written (no output) only after a committed resume or with a no_resume record
                                   "needs_lead_rejudgment": "not_written"})        # not written: the lead decides first
NO_OUTPUT_CLASS = MappingProxyType(dict(method_failure="method", unit_limit="unit_limit", wrapper_exception="wrapper", implementation_error="implementation",
                                        infrastructure="infrastructure", p20="bz_no_rerun"))
# bz_no_rerun (判定（BZ 不再跑）, reported apart from the classes (1)-(4) of preregistration 4.5): BZ's no_resume of a P20 cell (the method's hx_provenance_rejected; an external or
# configuration C cell refused before its lock again after a rerun), or a rejudgment record's no_output (prereg revision 2 items 17 (f), 18).
METHOD_P20 = "hx_provenance_rejected"                 # the method arm's P20 status: not written, unless BZ's committed no_resume (P20) writes it (no rejudgment record, N2 (ii))
# The committed holdout_common records a class needs before its cell counts as written (alignment item 3; review I5, I6, I10)
AUTHORIZATION_NEEDED = MappingProxyType(dict(wrapper_exception=("resume",), infrastructure=("resume", "no_resume"), implementation_error=("no_resume_jia",),
                                             p20=("no_resume_p20",), resumed=("resume",)))
JIA, YI = rejudgment_io.CHOICE_WORDS["(甲)"], rejudgment_io.CHOICE_WORDS["(乙)"]       # C22.5-6: BZ's choice in a record's reason (bayesreg.io.rejudgment.decision_reason_problem)
KIND_FILES = dict(disp=("disp.npz",))                 # the files fields.describe_run / load_field read per kind (relative to the field directory)
ARM_CERTIFICATE = ("jacobian_only", "BL1")            # every arm (preregistration §3, revision 1a item 7, D112 (a); the lead's ruling (5) of 11-09)
ARM_KEYS = ("name", "family", "kind", "record_dir", "field", "cert_mode", "jacobian_config", "terminal", "complete_files", "review_record", "decision_certificate",
            "export_check", "stop_record", "authorization")


def forbidden_path(path):
    """Under DATA_ROOT (the hold-out subjects' images and labels) or a seq_vp_3d tree, by the unresolved or the resolved path (review2 m2: no input of this
    entry but read_truth's may lie there)."""
    roots = [Path(DATA_ROOT), *READ_ONLY_ROOTS]
    forms = {os.path.abspath(str(path)), os.path.realpath(str(path))}
    for root in roots:
        for r in {os.path.abspath(str(root)), os.path.realpath(str(root))}:
            if any(f == r or f.startswith(r+os.sep) for f in forms):
                return True
    return False


def _template(text, where, allowed=("pair", "fixed", "moving", "arm")):
    if not isinstance(text, str) or not text:
        raise HoldoutRefused("arms", f"{where}: a non-empty string required")
    bad = [f for f in _TEMPLATE_FIELDS.findall(text) if f not in allowed]
    if bad:
        raise HoldoutRefused("arms", f"{where}: unknown template fields {bad}")
    return text


def _relative(text, where):
    _template(text, where)
    if text.startswith("/") or any(part in ("", ".", "..") for part in text.split("/")):
        raise HoldoutRefused("arms", f"{where}: a relative path without '.', '..' required")
    return text


def _absolute(text, where, *, pair=True, fields=True):
    """An absolute path (template), without '..', not under DATA_ROOT or a seq_vp_3d tree (review2 m2); with {pair} when `pair`; no field at all unless
    `fields`."""
    _template(text, where)
    plain = _TEMPLATE_FIELDS.sub("x", text)
    if not text.startswith("/") or ".." in text.split("/") or (pair and "{pair}" not in text) or (not fields and "{" in text) or forbidden_path(plain):
        raise HoldoutRefused("arms", f"{where}: an absolute path{' template with {pair}' if pair else ''} without '..', not under {DATA_ROOT} or a seq_vp_3d tree")
    return text


def _dotted_key(text, where):
    _template(text, where)
    if any(not part for part in text.split(".")):
        raise HoldoutRefused("arms", f"{where}: a dotted key required")
    return text


def validate_pq_section(pq):
    """The arm table's pq section (required: PQ is in the hold-out method entry, contract revision 7; review I1): the two roots of PQ's outputs (C22.H6-1;
    PQH): small_root /home/rmcfz/seq_vp_tag_results/holdout_B_v1/pq, big_root /mnt/d/seq_vp_tag_holdout_B_v1/pq.  The files themselves are found through
    PQ's sha256 list (alignment item 5); the roots are where they must lie and what the spy watches."""
    if not isinstance(pq, dict) or set(pq) != {"small_root", "big_root"}:
        raise HoldoutRefused("arms", "pq: exactly {small_root, big_root} (PQ is the method arm's added output, not an arm; a null pq is refused: F4 must be computable)")
    for key in ("small_root", "big_root"):
        _absolute(pq[key], f"pq.{key}", pair=False, fields=False)
    if pq["small_root"] == pq["big_root"]:
        raise HoldoutRefused("arms", "pq: two different roots")
    return pq


def _final_values(values, where):
    if not isinstance(values, list) or not values or not all(isinstance(v, (str, bool, int)) for v in values):
        raise HoldoutRefused("arms", f"{where}: a non-empty list of strings, bools or ints")
    return values


def _validate_field(arm, where):
    """field: {"dir": an absolute template with {pair}} (the method arm: its cell's output/), or {"file_key", "sha256_key", "root"}: the terminal record names
    the field file and its sha256 (XB's disp, HB configuration C's output_store; a resumed attempt writes beside the first), which must lie under the root
    (an absolute template without {pair}, e.g. /mnt/d/seq_vp_tag_holdout_B_v1/{arm}); the kind disp then (the file is <dir>/disp.npz)."""
    field = arm["field"]
    if isinstance(field, dict) and set(field) == {"dir"}:
        _absolute(field["dir"], f"{where}.field.dir")
    elif isinstance(field, dict) and set(field) == {"file_key", "sha256_key", "root"}:
        _dotted_key(field["file_key"], f"{where}.field.file_key")
        _dotted_key(field["sha256_key"], f"{where}.field.sha256_key")
        _absolute(field["root"], f"{where}.field.root", pair=False)
        if "{pair}" in field["root"] or arm["kind"] != "disp":
            raise HoldoutRefused("arms", f"{where}.field: a root without {{pair}}, and kind disp, when the terminal record names the field file")
    else:
        raise HoldoutRefused("arms", f"{where}.field: {{dir}} or {{file_key, sha256_key, root}}")


def validate_arm_table(table):
    """The arm table's form (runs/20261107-ev/实现说明.md §11, the proposal for revision 2, v3 after the interface alignment).  HoldoutRefused('arms')."""
    keys = {"schema", "stage_root", "main_family", "arms", "pq", "segmentation"}
    if not isinstance(table, dict) or set(table) - {"comment"} != keys:
        raise HoldoutRefused("arms", f"arm table: exactly the keys {sorted(keys)} (and an optional comment) are required")
    if table["schema"] != SCHEMA_ARMS:
        raise HoldoutRefused("arms", f"arm table: schema {SCHEMA_ARMS} required")
    _absolute(table["stage_root"], "stage_root", pair=False, fields=False)
    validate_pq_section(table["pq"])
    if table["segmentation"] is not None:
        raise HoldoutRefused("arms", "segmentation: the segmentation-level reading is not implemented in this entry (null required)")
    arms = table["arms"]
    if not isinstance(arms, list) or not arms:
        raise HoldoutRefused("arms", "arms: a non-empty list required")
    names = []
    for k, arm in enumerate(arms):
        where = f"arms[{k}]"
        if not isinstance(arm, dict) or set(arm) != set(ARM_KEYS):
            raise HoldoutRefused("arms", f"{where}: exactly the keys {sorted(ARM_KEYS)} are required")
        name = arm["name"]
        if not isinstance(name, str) or not _SAFE_ARM.fullmatch(name) or name in names or name in RESERVED_NAMES:
            raise HoldoutRefused("arms", f"{where}.name {name!r}: a distinct lowercase id, none of {RESERVED_NAMES}")
        if any(x in name for x in EXCLUDED_ARMS):
            raise HoldoutRefused("arms", f"{where}.name {name!r}: TransMorph, VPreg and SynthMorph are not in the arm table (preregistration 2.5)")
        names.append(name)
        if arm["family"] not in FAMILIES:
            raise HoldoutRefused("arms", f"{where}.family: one of {FAMILIES}")
        if arm["kind"] not in KIND_FILES and arm["kind"] != "configC":
            raise HoldoutRefused("arms", f"{where}.kind: 'disp' or 'configC'")
        _absolute(arm["record_dir"], f"{where}.record_dir")
        _validate_field(arm, where)
        if (arm["cert_mode"], arm["jacobian_config"]) != ARM_CERTIFICATE:
            raise HoldoutRefused("arms", f"{where}: cert_mode {ARM_CERTIFICATE[0]!r} with jacobian_config {ARM_CERTIFICATE[1]!r} on every arm (preregistration §3, "
                                         "D112 (a): the Jacobian-only certificate all arms share; the method's decision certificate is decision_certificate)")
        terminal = arm["terminal"]
        if not isinstance(terminal, dict) or set(terminal) != {"files", "status_key", "classes"} or not isinstance(terminal["files"], list) or not terminal["files"]:
            raise HoldoutRefused("arms", f"{where}.terminal: exactly files (the terminal records in precedence order, the first there counts), status_key, classes")
        for j, rel in enumerate(terminal["files"]):
            _relative(rel, f"{where}.terminal.files[{j}]")
        if not isinstance(terminal["status_key"], str) or not terminal["status_key"]:
            raise HoldoutRefused("arms", f"{where}.terminal.status_key: a dotted key required")
        classes = terminal["classes"]
        if not isinstance(classes, dict) or set(classes) != set(CLASS_KEYS):
            raise HoldoutRefused("arms", f"{where}.terminal.classes: exactly {list(CLASS_KEYS)}")
        seen = []
        for key in CLASS_KEYS:
            values = classes[key]
            if not isinstance(values, list) or not all(isinstance(v, str) and v for v in values):
                raise HoldoutRefused("arms", f"{where}.terminal.classes.{key}: a list of statuses")
            seen += values
        if len(seen) != len(set(seen)) or not classes["output"]:
            raise HoldoutRefused("arms", f"{where}.terminal.classes: every status in one class only; at least one output status")
        wrong = [s for key in CLASS_KEYS for s in classes[key] if s in FIXED_STATUSES and FIXED_STATUSES[s] != key]
        if wrong:
            raise HoldoutRefused("arms", f"{where}.terminal.classes: {wrong} have a fixed class ({dict(FIXED_STATUSES)}) on every arm")
        if classes["implementation_error"] and (arm["family"] != "method" or classes["implementation_error"] != ["hx_implementation_error"]):
            raise HoldoutRefused("arms", f"{where}.terminal.classes.implementation_error: ['hx_implementation_error'] on the method arm only (C22.5-6, "
                                         "revision 8b); a wrapper's code exception is wrapper_exception")
        if not isinstance(arm["complete_files"], list):
            raise HoldoutRefused("arms", f"{where}.complete_files: the files of a complete record (preregistration 5.4), relative to record_dir")
        for j, rel in enumerate(arm["complete_files"]):
            _relative(rel, f"{where}.complete_files[{j}]")
        if arm["family"] == "method":
            review = arm["review_record"]
            if not isinstance(review, dict) or set(review) != {"file", "status_key", "final"}:
                raise HoldoutRefused("arms", f"{where}.review_record: {{file, status_key, final}}: the step-5 review record and its finalized values "
                                             "(C22.H1, D111; review I7; HB: review/review.json, finalized true)")
            if isinstance(review["file"], str) and review["file"].startswith("/"):      # a batch record (absolute) or a file of the cell (relative)
                _absolute(review["file"], f"{where}.review_record.file", pair=False)
            else:
                _relative(review["file"], f"{where}.review_record.file")
            _dotted_key(review["status_key"], f"{where}.review_record.status_key")
            _final_values(review["final"], f"{where}.review_record.final")
            certificate = arm["decision_certificate"]
            if not isinstance(certificate, dict) or set(certificate) != {"final", "cert"}:
                raise HoldoutRefused("arms", f"{where}.decision_certificate: {{final, cert}} (output/final.npz and output/cert.json; D112 (a), review I3, m15)")
            for key in ("final", "cert"):
                _relative(certificate[key], f"{where}.decision_certificate.{key}")
                if certificate[key] not in arm["complete_files"]:
                    raise HoldoutRefused("arms", f"{where}.decision_certificate.{key}: one of complete_files (read from the manifest)")
        elif arm["review_record"] is not None or arm["decision_certificate"] is not None:
            raise HoldoutRefused("arms", f"{where}: review_record and decision_certificate on the method arm only")
        if arm["export_check"] is not None:
            check = arm["export_check"]
            if not isinstance(check, dict) or set(check) != {"key"}:
                raise HoldoutRefused("arms", f"{where}.export_check: null or {{key}} (a bool of the terminal record)")
            _dotted_key(check["key"], f"{where}.export_check.key")
        _relative(arm["stop_record"], f"{where}.stop_record")
        auth = arm["authorization"]
        if not isinstance(auth, dict) or set(auth) != {"tool", "arm", "records"} or not all(isinstance(auth[k], str) and _SAFE_ARM.fullmatch(auth[k]) for k in ("tool", "arm")):
            raise HoldoutRefused("arms", f"{where}.authorization: {{tool, arm, records}}: holdout_common's names of this arm's resume / no_resume records "
                                         "(splits/bayesreg_v1/holdout_B_v1_<tool>_<kind>_<k>.json) and the directory their records lie in")
        _absolute(auth["records"], f"{where}.authorization.records")
    methods = [a["name"] for a in arms if a["family"] == "method"]
    main = table["main_family"]
    if len(methods) != 1:
        raise HoldoutRefused("arms", f"exactly one method arm required, got {methods}")
    if not isinstance(main, list) or len(main) != 5 or len(set(main)) != 5 or sorted(main) != sorted(a["name"] for a in arms if a["family"] == "main"):
        raise HoldoutRefused("arms", "main_family: the 5 arms of family main, in the preregistration's order (preregistration 4.2)")
    return table


def method_arm(table):
    return next(a for a in table["arms"] if a["family"] == "method")


def load_arm_table(prereg_raw, repo=None):
    """The pinned arm table (ARM_TABLE_SHA256), committed and unchanged, its sha256 written in the pinned preregistration (the preregistration pins it)."""
    if ARM_TABLE_SHA256 is None:
        raise HoldoutRefused("arms", "ARM_TABLE_SHA256 is None: the machine-readable arm table (preregistration revision 2, item 16) is not pinned")
    raw, digest, git = read_committed(ARM_TABLE_REL, "arms", repo)
    if digest != ARM_TABLE_SHA256:
        raise HoldoutRefused("arms", f"{ARM_TABLE_REL}: sha256 {digest} is not the pinned {ARM_TABLE_SHA256}")
    if ARM_TABLE_SHA256.encode("ascii") not in prereg_raw:
        raise HoldoutRefused("arms", f"the preregistration does not name the arm table's sha256 {ARM_TABLE_SHA256}")
    table = validate_arm_table(strict_json(raw, ARM_TABLE_REL))
    return table, dict(path=ARM_TABLE_REL, sha256=digest, git=git)


def fill(template, arm, row):
    _, fixed, moving, name = row
    return template.format(pair=name, fixed=fixed, moving=moving, arm=arm["name"])


def record_dir(arm, row):
    return Path(fill(arm["record_dir"], arm, row))


def arm_roots(arm):
    """The directories above every cell of the arm: its record root, its field root and the root of its authorizations' records (the templates up to their first per-pair field)."""
    field = arm["field"]
    templates = (arm["record_dir"], field["dir"] if "dir" in field else field["root"], arm["authorization"]["records"])
    return [template_root(t.replace("{arm}", arm["name"])) for t in templates]


def watched_roots(table):
    """What the spy watches (C23.6-5): the data root, the stage, every arm's record, field and authorization-record roots, PQ's two roots, the step-5 records on /home
    (REVIEW_ROOT: the finalized record, the kept copies) and the repair ablation's two roots (A5b)."""
    roots = [Path(DATA_ROOT), Path(table["stage_root"])]+[r for arm in table["arms"] for r in arm_roots(arm)]
    return list(dict.fromkeys(roots+[Path(table["pq"]["small_root"]), Path(table["pq"]["big_root"]), Path(REVIEW_ROOT), Path(PREREPAIR_ROOT), Path(PREREPAIR_BIG)]))


def review_record_path(arm, row):
    """The method arm's step-5 review record: a file of the cell (HB: review/review.json) or an absolute batch file."""
    rel = fill(arm["review_record"]["file"], arm, row)
    return Path(rel) if rel.startswith("/") else record_dir(arm, row)/rel


def dotted(document, key):
    value = document
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def kind_files(arm, base, protocol=None):
    """The files of the field directory `base` that fields.describe_run / load_field read for the arm's kind (configC: the task named by protocol.json)."""
    base = Path(base)
    if arm["kind"] == "disp":
        return [base/name for name in KIND_FILES["disp"]]
    files = [base/"protocol.json", base/"exit_code", base/"completion.json"]
    if protocol is not None:
        task = (protocol.get("task") or {})
        name, pair = task.get("name"), task.get("pair")
        if not isinstance(name, str) or not isinstance(pair, str) or "/" in name or "/" in pair:
            raise HoldoutRefused("arms", f"{base}/protocol.json: task.name and task.pair plain names required")
        files += [base/name/"summary.json", base/name/"worker_completion.json", base/name/"final_native.json", base/name/"final_native.npz",
                  base/"inputs"/pair/"manifest.json"]
        if (base/"run_id").is_file():
            files.append(base/"run_id")
    return files


@dataclass
class Cell:
    arm: str
    pair: str
    index: int
    written: bool
    outcome: str | None              # "output" | "no_output" | None (not written)
    no_output_class: str | None      # method | unit_limit | wrapper | implementation | infrastructure | arm_stopped
    status: str | None
    export_suspicious: bool | None
    needed: list                     # files this entry reads for the cell (all in the manifest)
    records: dict                    # path -> (sha256, document or None) of the small records read in (b)
    problem: str | None = None
    authorization: dict | None = None
    stop_record: dict | None = None
    review: dict | None = None       # the method arm's review record: path, key, value
    terminal: str | None = None      # the terminal record that counts (the first of terminal.files there)
    field_dir: str | None = None     # the field directory given to fields.describe_run
    field_sha256: dict | None = None  # {path: sha256} the terminal record names for its field (compared with the manifest in (c))
    cause: str | None = None         # bz_no_rerun's cause: P20 (BZ's no_resume of a cell refused before its lock, or of the method's hx_provenance_rejected) or rejudgment
    rejudgment: dict | None = None   # the committed rejudgment record that wrote the cell (external arms and configuration C only; prereg revision 2 item 18)
    move: dict | None = None         # the method arm: the finalized record's row and the move rows of the files read (A5b item 8)
    pinned_sha256: dict | None = None  # the method arm: {path: sha256} the finalized record names for each file read in the cell on D: (compared with the manifest in (c))


def _read_record(path, records):
    """A small JSON record read once: its bytes hashed and kept (the manifest check of (c) compares this sha256)."""
    raw = Path(path).read_bytes()
    document = strict_json(raw, str(path))
    records[str(path)] = (sha_bytes(raw), document)
    return document


def _hash_record(path, records):
    raw = Path(path).read_bytes()
    records[str(path)] = (sha_bytes(raw), None)
    return records[str(path)][0]


def _committed_record(rel, schema, arm, row, repo, stage="arms", writers=AUTHORIZATION_WRITERS):
    raw, digest, git = read_committed(rel, stage, repo)
    document = strict_json(raw, rel)
    schemas = (schema,) if isinstance(schema, str) else tuple(schema)
    if not isinstance(document, dict) or document.get("schema") not in schemas or document.get("arm") != arm["name"]:
        raise HoldoutRefused(stage, f"{rel}: schema {schemas} for arm {arm['name']} required")
    if document.get("written_by") not in writers:
        raise HoldoutRefused(stage, f"{rel}: written_by {document.get('written_by')!r} is not one of {list(writers)}")
    problem = utc_problem(document.get("utc"), "utc")
    if problem:
        raise HoldoutRefused(stage, f"{rel}: {problem}")
    return document, dict(path=rel, sha256=digest, git=git)


def _record_place_problem(path, allowed):
    """A record a holdout_common authorization names: an absolute path inside the allowed directory (the arm's authorization.records, or for arm review the
    step-5 records kept on /home, REVIEW_ROOT/<pair>), not under DATA_ROOT (review2 m2)."""
    if not isinstance(path, str) or not path.startswith("/"):
        return f"{path!r}: an absolute path required"
    if forbidden_path(path) or not _inside(path, allowed):
        return f"{path}: not inside {allowed}, or under {DATA_ROOT} / a seq_vp_3d tree"
    return None


def _no_resume_checked(hc, auth, allowed, records, needed, *, expected_record=None):
    """(row or None, problem or None): one holdout_common no_resume record of a cell: its note 未批补跑, its record inside `allowed` (and the very file
    `expected_record` when given) at record_sha256 (read: an input of the manifest)."""
    doc = auth.document
    record = doc.get("record")
    problem = _record_place_problem(record, allowed)
    if problem is None and expected_record is not None and os.path.realpath(str(record)) != os.path.realpath(str(expected_record)):
        problem = f"its record {record} is not {expected_record}"
    if problem is None and doc.get("note") != hc.NO_RESUME_NOTE:
        problem = f"note {doc.get('note')!r} is not {hc.NO_RESUME_NOTE!r}"
    if problem is None and (not Path(record).is_file() or _hash_record(record, records) != doc.get("record_sha256")):
        problem = f"{record}: missing or not at the record_sha256 {doc.get('record_sha256')!r}"
    if problem:
        return None, f"no_resume {auth.path}: {problem}"
    needed.append(Path(record))
    return dict(kind="no_resume", path=auth.path, sha256=auth.sha256, arm=doc.get("arm"), record=record, record_sha256=doc.get("record_sha256"), reason=doc.get("reason")), None


def cell_authorization(arm, row, kind, terminal_path, terminal_sha, records, needed, *, decided=None):
    """The committed holdout_common records of one cell (alignment item 3; preregistration 5.4, revision 1c items 3, 4): find_authorizations of kinds resume
    and no_resume for (authorization.tool, authorization.arm, the pair, holdout) -- committed, unchanged, written by BZ or "lead on BZ's behalf" with BZ's
    words, the time and a reason, checked there.  This entry adds (review2 m1, m2):
      resume     exactly one; its previous record (when it names one with a sha256) is kept, inside authorization.records, at that sha256 (read: an input of
                 the manifest), and the terminal record now is another file with another sha256 -- the resume ran;
      no_resume  its note is 未批补跑, its record is there inside authorization.records at record_sha256 (read: an input); for hx_implementation_error its
                 reason states BZ's (甲) of C22.5-6 and not (乙); on the method arm also a no_resume of arm review (step 5: BZ's (甲) on a step-5
                 hx_implementation_error, or a withdrawal; hb_review), whose record is one of the step-5 records kept on /home (REVIEW_ROOT/<pair>/;
                 prereg revision 2 item 14, A5b item 5); for the method arm's hx_provenance_rejected (kind p20) its reason names P20 (prereg revision 2
                 items 17 (f), 18: BZ does not let a P20 cell run again; the method takes no rejudgment record).
                 A5b review I2 (the lead's ruling 11-10 ~9:22 PM CST): on the method arm the entry does not pick a (甲) or P20 record itself: it takes the one record at
                 the sha256 HB's finalized record names (`decided`: its bz_decision, hb_core.bz_decision_auth's choice), of the arm its step names (4: the
                 method arm's, 5: arm review's), and checks it against the same criterion (bayesreg.io.rejudgment: decision_reason_problem with (甲) and one case of
                 C22.5-6, p20_reason_problem); another record of the cell, sound or not, does not count.
    kind 'resumed': a resume that ran (an infrastructure rejudgment after the resume, prereg revision 2 item 18).
    Returns (record or None, problem or None).  A record of this cell that holdout_common refuses stops the evaluation (HoldoutRefused('arms'))."""
    hc = _holdout_common()
    spec = arm["authorization"]
    allowed = Path(fill(spec["records"], arm, row))
    places = [(spec["arm"], allowed)]
    if arm["family"] == "method" and kind == "implementation_error":
        places.append((REVIEW_ARM, Path(REVIEW_ROOT)/row[3]))
    found = {"resume": [], "no_resume": []}
    for kind_ in ("resume", "no_resume"):
        for name, place in (places if kind_ == "no_resume" else places[:1]):
            valid, refused = hc.find_authorizations(kind=kind_, tool=spec["tool"], arm=name, pair=row[3], pair_set=PAIR_SET)
            if refused:
                raise HoldoutRefused("arms", f"{arm['name']} {row[3]}: holdout_common refuses these {kind_} records of the cell (arm {name}): {refused}")
            found[kind_] += [(auth, place) for auth in valid]
    problems = []
    if len(found["resume"]) > 1:
        problems.append(f"{len(found['resume'])} resume records: one resume per cell (preregistration 5.4)")
    resume = None
    for auth, _ in found["resume"][:1]:
        doc = auth.document
        previous, digest = doc.get("previous_record"), doc.get("previous_record_sha256")
        problem = _record_place_problem(previous, allowed)
        if problem is None and os.path.realpath(previous) == os.path.realpath(terminal_path):
            problem = f"{previous}: the resume's previous record is the terminal record now (the resume did not run; review2 m1)"
        if problem is None and digest is not None:
            if not Path(previous).is_file() or _hash_record(previous, records) != digest:
                problem = f"{previous}: missing or not at the resume's previous_record_sha256 {digest}"
            elif digest == terminal_sha:
                problem = f"{terminal_path} has the sha256 of the resume's previous record: the resume did not run (review2 m1)"
            else:
                needed.append(Path(previous))
        if problem:
            problems.append(f"resume {auth.path}: {problem}")
        else:
            resume = dict(kind="resume", path=auth.path, sha256=auth.sha256, previous_record=previous, previous_record_sha256=digest)
    no_resume, jia, p20 = None, None, None
    by_decision = arm["family"] == "method" and kind in ("implementation_error", "p20")
    decided_ok = isinstance(decided, dict) and is_hex64(decided.get("sha256")) and decided.get("step") in (4, 5)
    if by_decision and not decided_ok:
        problems.append(f"the finalized record names no bz_decision {{step, path, sha256}} for {row[3]} ({decided!r}): BZ's decision is the record HB took (A5b review I2)")
    for auth, place in found["no_resume"]:
        if by_decision and (not decided_ok or auth.sha256 != decided["sha256"]):
            continue                                                 # A5b review I2: only the record the finalized bz_decision names
        row_, problem = _no_resume_checked(hc, auth, place, records, needed)
        if problem:
            problems.append(problem)
            continue
        reason = row_["reason"]
        if row_["arm"] == spec["arm"]:
            no_resume = no_resume or row_
        if by_decision:
            step_arm = spec["arm"] if decided.get("step") == 4 else REVIEW_ARM
            why = (rejudgment_io.p20_reason_problem(reason) if kind == "p20" else rejudgment_io.decision_reason_problem(reason, None, "(甲)"))
            if row_["arm"] != step_arm:
                why = f"its arm {row_['arm']!r} is not step {decided.get('step')}'s ({step_arm!r})"
            if why:
                problems.append(f"no_resume {auth.path} (the finalized bz_decision): {why}")
            elif kind == "p20":
                p20 = dict(row_, kind="no_resume_p20")
            else:
                jia = dict(row_, kind="no_resume_jia")
    have = dict(resume=resume, no_resume=no_resume, no_resume_jia=jia, no_resume_p20=p20)
    for name in AUTHORIZATION_NEEDED[kind]:
        if have[name] is not None:
            return have[name], None
    want = {"wrapper_exception": "a first wrapper exception is not written (revision 1a item 2, 1b item 5): only after a committed resume of the cell whose run "
                                 "ended in a wrapper exception again",
            "infrastructure": "infra_failure is written only after a committed resume of the cell that failed again, or with a committed no_resume record "
                              "(未批补跑) (revision 1c item 3)",
            "implementation_error": "hx_implementation_error is written only with a committed no_resume record (arm method at step 4, arm review at step 5) whose "
                                    "reason states BZ's (甲) of C22.5-6 (alignment item 3; revision 1c item 3; prereg revision 2 item 14), the one the finalized "
                                    "record's bz_decision names (A5b review I2)",
            "p20": "hx_provenance_rejected (P20) is written only with BZ's committed no_resume record of the cell, its reason naming P20 (prereg revision 2 items 17 (f), "
                   "18; no rejudgment record for the method arm), the one the finalized record's bz_decision names (A5b review I2)",
            "resumed": "an infrastructure rejudgment writes the cell only after a committed resume of it ran (prereg revision 2 item 18)"}[kind]
    return None, want+(f"; {problems}" if problems else "")


def _final(value, allowed):
    return any(type(value) is type(v) and value == v for v in allowed)


# ---- the method cell on D: (A5b items 5, 8): the finalized record and the move records ---------------------------------------------------------------------
def finalized_record(cache, records):
    """REVIEW_ROOT/finalized_outputs.json (hb_review.finalize: C22.H1-3), read once per run (`cache`), strict JSON, schema FINALIZED_SCHEMA, pairs a dict;
    its bytes hashed into `records` (an input of the manifest).  (document or None, problem or None)."""
    path = Path(REVIEW_ROOT)/FINALIZED_NAME
    if "finalized" not in cache:
        if not path.is_file():
            cache["finalized"] = (None, None, f"{path}: not there: step 5 has not finalized the method arm (C22.H1-3; no move before it)")
        else:
            raw = path.read_bytes()
            try:
                doc, problem = strict_json(raw, str(path)), None
            except HoldoutRefused as exc:
                doc, problem = None, str(exc)
            if problem is None and (not isinstance(doc, dict) or doc.get("schema") != FINALIZED_SCHEMA or not isinstance(doc.get("pairs"), dict)):
                problem = f"{path}: schema {FINALIZED_SCHEMA} with pairs required"
            cache["finalized"] = (doc, sha_bytes(raw), problem)
    doc, digest, problem = cache["finalized"]
    if digest is not None:
        records[str(path)] = (digest, doc)
    return (None if problem else doc), problem


def finalized_sha256(frow, rel):
    """The sha256 the finalized record names for the file `rel` of the method cell: terminal.json and review/review.json by their rows, the output and T88/repair
    files by part.output / part.t88_repair (hb_review.finalized_part); None when it names none."""
    if rel == "terminal.json":
        return (frow.get("terminal") or {}).get("sha256")
    if rel == "review/review.json":
        return (frow.get("review") or {}).get("sha256")
    part = frow.get("part") if isinstance(frow.get("part"), dict) else {}
    for key in ("output", "t88_repair"):
        rows = part.get(key) if isinstance(part.get(key), dict) else {}
        if rel in rows:
            return rows[rel]
    return None


MOVE_ROW_KEYS, MOVE_OK_ROW_KEYS = ("file", "src", "dst", "ok"), ("sha256_before", "sha256_after")       # holdout_common.move_files writes these in every row / every ok row


def _move_rows(path, document, cell_dir, dst_dir):
    """The rows of one move record (holdout_common.move_files' format) and its problems: schema MOVE_SCHEMA, files a list, src_dir the finalized cell, dst_dir the arm table's cell on D:;
    every row with MOVE_ROW_KEYS and every ok row also with MOVE_OK_ROW_KEYS (A5b diff review minor 3: a row without dst is named as the record's format, not as a missing move)."""
    problems = []
    if not isinstance(document, dict) or document.get("schema") != MOVE_SCHEMA or not isinstance(document.get("files"), list):
        return [], [f"{path}: schema {MOVE_SCHEMA} with files required"]
    for i, r in enumerate(document["files"]):
        missing = [k for k in MOVE_ROW_KEYS+(MOVE_OK_ROW_KEYS if isinstance(r, dict) and r.get("ok") is True else ()) if not isinstance(r, dict) or k not in r]
        if missing:
            problems.append(f"{path}: files[{i}] ({r.get('file') if isinstance(r, dict) else r!r}) has no {missing}: the move record's format (holdout_common.move_files: "
                            f"{', '.join(MOVE_ROW_KEYS)} in every row, {', '.join(MOVE_OK_ROW_KEYS)} in an ok row)")
    if os.path.realpath(str(document.get("src_dir"))) != os.path.realpath(str(cell_dir)):
        problems.append(f"{path}: src_dir {document.get('src_dir')!r} is not the finalized record's cell {cell_dir}")
    if os.path.realpath(str(document.get("dst_dir"))) != os.path.realpath(str(dst_dir)):
        problems.append(f"{path}: dst_dir {document.get('dst_dir')!r} is not the arm table's cell {dst_dir}")
    return [r for r in document["files"] if isinstance(r, dict)], problems


def method_moves(arm, row, frow, records, needed):
    """(files, problem): the interface alignment revision 2's reading of the move records in the method arm's record directory (authorization.records):
    move.json passed true -> its rows; passed false -> its ok rows and the rows of move_resume.json, which must have passed; otherwise refused.  Every file in
    exactly one ok row.  files: {file (relative to the cell): dict(src, dst, sha256_before, sha256_after, record)}.  Both records are read (inputs of the
    manifest)."""
    rec_dir = Path(fill(arm["authorization"]["records"], arm, row))
    cell_dir, dst_dir = frow.get("cell"), record_dir(arm, row)
    path = rec_dir/MOVE_NAME
    if not path.is_file():
        return None, f"{path}: no move record: the method cell was not moved to D: (hb_store.py move; the arm table's cell is {dst_dir})"
    move = _read_record(path, records)
    needed.append(path)
    rows, problems = _move_rows(path, move, cell_dir, dst_dir)
    sources = [(path, r) for r in rows if r.get("ok") is True]
    if move.get("passed") is not True:
        resumed_path = rec_dir/MOVE_RESUME_NAME
        if move.get("passed") is not False or not resumed_path.is_file():
            return None, f"{path}: passed {move.get('passed')!r} and no {MOVE_RESUME_NAME}: a half-moved cell (repair: hb_store.py move-resume)"
        resumed = _read_record(resumed_path, records)
        needed.append(resumed_path)
        more, more_problems = _move_rows(resumed_path, resumed, cell_dir, dst_dir)
        problems += more_problems
        if resumed.get("passed") is not True:
            problems.append(f"{resumed_path}: passed {resumed.get('passed')!r}: the move-resume did not pass")
        sources += [(resumed_path, r) for r in more if r.get("ok") is True]
    files = {}
    for where, r in sources:
        name = r.get("file")
        if name in files:
            problems.append(f"{name}: in two ok rows ({files[name]['record']}, {where}): every file in exactly one (the interface alignment revision 2)")
            continue
        files[name] = dict(src=r.get("src"), dst=r.get("dst"), sha256_before=r.get("sha256_before"), sha256_after=r.get("sha256_after"), record=str(where))
    if problems:
        return None, "; ".join(problems)
    return files, None


def method_cross_check(arm, row, cell, cache):
    """A5b item 8 (the lead's ruling (8); the interface alignment revisions 1, 2): from the finalized record's row of the pair (its `cell`) through the move
    records to the arm table's cell on D:: every file this entry reads under that cell is the dst of an ok move row whose src is the finalized cell's file,
    whose sha256_after equals the finalized record's sha256 of it (and, for a record read in (b), the sha256 of the bytes read); the finalized sha256 is kept
    (cell.pinned_sha256) and compared with the manifest in (c).  None, or the problem (the cell is then not written)."""
    final, problem = finalized_record(cache, cell.records)
    finalized_path = Path(REVIEW_ROOT)/FINALIZED_NAME
    if final is None:
        return problem
    cell.needed.append(finalized_path)
    frow = final["pairs"].get(row[3])
    if not isinstance(frow, dict) or not isinstance(frow.get("cell"), str):
        return f"{finalized_path}: no row with a cell for {row[3]}"
    files, problem = method_moves(arm, row, frow, cell.records, cell.needed)
    if problem:
        return problem
    base = record_dir(arm, row)
    pinned, rows = {}, {}
    for path in cell.needed:
        if not _inside(path, base):
            continue
        rel = Path(os.path.abspath(str(path))).relative_to(os.path.abspath(str(base))).as_posix()
        want = finalized_sha256(frow, rel)
        mv = files.get(rel)
        if not is_hex64(want):
            return f"{finalized_path}: names no sha256 for {rel} of {row[3]} (the entry reads {path})"
        if mv is None or os.path.realpath(str(mv["dst"])) != os.path.realpath(str(path)) or mv["sha256_after"] != want:
            return f"{rel}: no ok move row with dst {path} and sha256_after {want} (the finalized record's): {mv}"
        if os.path.realpath(str(mv["src"])) != os.path.realpath(str(Path(frow["cell"])/rel)):
            return f"{rel}: the move row's src {mv['src']!r} is not the finalized cell's {Path(frow['cell'])/rel}"
        read = cell.records.get(str(path))
        if read is not None and read[0] != want:
            return f"{path}: read at sha256 {read[0]}, the finalized record names {want}"
        pinned[str(path)] = want
        rows[rel] = dict(mv, finalized_sha256=want)
    cell.pinned_sha256 = pinned
    cell.move = dict(finalized=dict(path=str(finalized_path), line_status=frow.get("line_status"), bz_decision=frow.get("bz_decision"), cell=frow["cell"]),
                     files=rows, rule="the interface alignment revisions 1, 2: finalized cell -> move.json (or its ok rows and a passed move_resume.json) dst; sha256 = the finalized record's")
    if cell.authorization is not None and cell.authorization.get("kind") in ("no_resume_jia", "no_resume_p20"):
        decided = frow.get("bz_decision")
        if not isinstance(decided, dict) or decided.get("sha256") != cell.authorization["sha256"]:
            return f"{finalized_path}: bz_decision {decided!r} of {row[3]} is not the no_resume record {cell.authorization['path']} the entry found (sha256 {cell.authorization['sha256']})"
    return None


# ---- the cells of the other arms that have no terminal record: P19, P20 ------------------------------------------------------------------------------------------
def stop_cell(arm, row, root, cell, repo):
    """P19 (prereg revision 2 item 11; the lead's ruling (5)): the arm's committed stop record (stop_record: runs/holdout_B_v1/arm_stop_<arm>.json) lists the
    pair among its unrun pairs, names at least three suspicious pairs, and names XB's own record (arm_stopped: {path, sha256}) -- <the arm's record root>/
    arm_stopped.json, read here at that sha256 (an input of the manifest), schema XB_STOP_SCHEMA, its fields XB_STOP_FIELDS equal to the committed record's.
    (written, problem)."""
    stop_rel = fill(arm["stop_record"], arm, row)
    if not (root/stop_rel).is_file():
        return False, None
    document, git = _committed_record(stop_rel, SCHEMA_STOP, arm, row, repo, writers=STOP_WRITERS)
    unrun = document.get("unrun_pairs")
    if not (isinstance(unrun, list) and row[3] in unrun and isinstance(document.get("suspicious_pairs"), list) and len(document["suspicious_pairs"]) >= 3):
        return False, f"{stop_rel} does not list {row[3]} among the unrun pairs of a P19 stop (three suspicious pairs)"
    xb = document.get("arm_stopped")
    expected = template_root(arm["record_dir"].replace("{arm}", arm["name"]))/XB_STOP_NAME
    if not (isinstance(xb, dict) and set(xb) == {"path", "sha256"} and is_hex64(xb.get("sha256"))):
        return False, f"{stop_rel}: arm_stopped {{path, sha256}} of XB's {XB_STOP_NAME} required (prereg revision 2 item 11)"
    if os.path.realpath(str(xb["path"])) != os.path.realpath(str(expected)) or not Path(expected).is_file():
        return False, f"{stop_rel}: arm_stopped.path {xb['path']!r} is not XB's record {expected} (or it is not there)"
    original = _read_record(expected, cell.records)
    if cell.records[str(expected)][0] != xb["sha256"]:
        return False, f"{expected}: sha256 {cell.records[str(expected)][0]} is not the {xb['sha256']} {stop_rel} names"
    differ = [k for k in XB_STOP_FIELDS if not isinstance(original, dict) or original.get(k) != document.get(k)]
    if not isinstance(original, dict) or original.get("schema") != XB_STOP_SCHEMA or differ:
        return False, f"{expected}: schema {XB_STOP_SCHEMA} and the fields {differ} as {stop_rel} copies them required"
    cell.needed.append(expected)
    cell.written, cell.outcome, cell.no_output_class = True, "no_output", "arm_stopped"
    cell.stop_record = dict(git, document=document, arm_stopped=dict(path=str(expected), sha256=xb["sha256"]))
    return True, None


def p20_cell(arm, row, cell):
    """P20 of an external arm or the configuration C arm (prereg revision 2 item 17 (f); decision_log 2026-11-10 ~2:52, ~2:53 AM CST; A5b item 7): a cell
    refused before its lock, again after a rerun, with no terminal record, is written (no output, class bz_no_rerun, cause P20) by BZ's committed no_resume
    record whose record is the newest copy (by its sequence number seq) of the cell's refusal records in its record directory (<record dir>/refusals/*.json:
    holdout_common.keep_refusals / hb_configc.keep_refusals), at that copy's sha256, its reason naming P20.  Every copy is read (inputs of the manifest).
    (written, problem)."""
    hc = _holdout_common()
    folder = record_dir(arm, row)/REFUSAL_COPIES
    copies = sorted(folder.glob("*.json")) if folder.is_dir() else []
    if not copies:
        return False, None
    seqs = {}
    for path in copies:
        doc = _read_record(path, cell.records)
        cell.needed.append(path)
        seqs[str(path)] = doc.get("seq") if isinstance(doc, dict) else None
    ordered, problem = rejudgment_io.by_seq(copies, seq_of=lambda q: seqs[str(q)] if isinstance(seqs[str(q)], int) and not isinstance(seqs[str(q)], bool) and seqs[str(q)] >= 1 else None)
    if problem:
        return False, f"{folder}: {problem}"
    newest = ordered[-1][1]
    spec = arm["authorization"]
    valid, refused = hc.find_authorizations(kind="no_resume", tool=spec["tool"], arm=spec["arm"], pair=row[3], pair_set=PAIR_SET)
    if refused:
        raise HoldoutRefused("arms", f"{arm['name']} {row[3]}: holdout_common refuses these no_resume records of the cell: {refused}")
    problems = []
    for auth in valid:
        found, problem = _no_resume_checked(hc, auth, Path(fill(spec["records"], arm, row)), cell.records, cell.needed, expected_record=newest)
        if problem is None and rejudgment_io.p20_reason_problem(found["reason"]) is not None:
            problem = f"no_resume {auth.path}: {rejudgment_io.p20_reason_problem(found['reason'])}"
        if problem:
            problems.append(problem)
            continue
        cell.authorization = dict(found, kind="no_resume_p20", newest_refusal_copy=dict(path=str(newest), seq=ordered[-1][0]))
        cell.written, cell.outcome, cell.no_output_class, cell.cause = True, "no_output", "bz_no_rerun", P20_WORD
        return True, None
    return False, (f"{folder}: refused before its lock ({len(copies)} refusal records copied); no committed no_resume record of BZ names the newest copy {newest} with P20 "
                   f"(prereg revision 2 item 17 (f)){'; ' + str(problems) if problems else ''}")


def rejudged_cell(arm, row, cell, terminal_path, root):
    """Prereg revision 2 item 18 (decision_log 2026-11-10 ~2:28 AM CST NI3, ~2:45 AM CST N1, N2; A5b item 6): an external arm's or the configuration C arm's
    cell whose terminal record is needs_lead_rejudgment: the committed rejudgment record of the cell (bayesreg.io.rejudgment.rejudgment, the one reader; arm
    authorization.arm) whose record is this terminal record at its sha256 now and whose status is the terminal status.  unit_limit -> written, no output, class
    unit_limit; no_output -> written, no output, class bz_no_rerun (判定（BZ 不再跑）), cause rejudgment; infrastructure -> written (no output, class
    infrastructure) only when the cell's committed resume ran (this terminal record is the resumed attempt's), otherwise not written.  (written, problem)."""
    spec = arm["authorization"]
    found, problems = rejudgment_io.rejudgment(spec["arm"], row[3], record=terminal_path, pair_set=PAIR_SET, root=root)
    if found is None:
        return False, (f"{terminal_path}: needs_lead_rejudgment and no committed rejudgment record that holds (runs/holdout_B_v1/rejudgments/{spec['arm']}/{row[3]}_<k>.json, "
                       f"prereg revision 2 item 18): {problems}")
    document = found["document"]
    if document["status"] != cell.status:
        return False, f"{found['path']}: status {document['status']!r} is not the terminal status {cell.status!r}"
    cell.rejudgment = dict(path=found["path"], sha256=found["sha256"], k=found["k"], judgment=document["judgment"], git=found["git"])
    judgment = document["judgment"]
    if judgment == "infrastructure":
        record, problem = cell_authorization(arm, row, "resumed", terminal_path, cell.records[str(terminal_path)][0], cell.records, cell.needed)
        if record is None:
            return False, f"{found['path']}: infrastructure before the resume ran only classifies the cell: {problem}"
        cell.authorization = record
    cell.written, cell.outcome = True, "no_output"
    cell.no_output_class = dict(unit_limit="unit_limit", no_output="bz_no_rerun", infrastructure="infrastructure")[judgment]
    cell.cause = "rejudgment"
    return True, None


def classify_cell(arm, row, table, *, repo=None, cache=None):
    """One (arm, pair) cell by the D107 classes of the module docstring.  Reads only the cell's terminal record (the first of terminal.files there), the method
    arm's review record, the records the cell's holdout_common authorizations name (hashed), the method arm's finalized and move records, XB's stop record,
    the cell's refusal copies, and committed records of the code tree; no field, no truth.  `cache`: shared by the cells of one run (the finalized record)."""
    _, _, _, name = row
    root = Path(REPO if repo is None else repo)
    cache = {} if cache is None else cache
    base = record_dir(arm, row)
    records = {}
    cell = Cell(arm["name"], name, row[0], False, None, None, None, None, [], records)
    terminal_path = next((base/fill(rel, arm, row) for rel in arm["terminal"]["files"] if (base/fill(rel, arm, row)).is_file()), None)
    if terminal_path is None:
        if arm["family"] != "method":
            written, problem = stop_cell(arm, row, root, cell, repo)
            if not written and problem is None:
                written, problem = p20_cell(arm, row, cell)
            if written:
                cell.needed = list(dict.fromkeys(cell.needed))
                return cell
            if problem is not None:
                cell.problem = problem
                return cell
        cell.problem = f"{base}: no terminal record ({arm['terminal']['files']}) and no committed stop record of the arm or BZ's P20 no_resume"
        return cell
    cell.terminal = str(terminal_path)
    document = _read_record(terminal_path, records)
    cell.needed.append(terminal_path)
    status = dotted(document, arm["terminal"]["status_key"])
    cell.status = status
    classes = {s: key for key, values in arm["terminal"]["classes"].items() for s in values}
    classes.update(FIXED_STATUSES)
    kind = classes.get(status) if isinstance(status, str) else None
    if status == "needs_lead_rejudgment" and arm["family"] != "method":
        written, problem = rejudged_cell(arm, row, cell, terminal_path, root)
        if not written:
            cell.problem = problem
        cell.needed = list(dict.fromkeys(cell.needed))
        return cell
    if status == METHOD_P20 and arm["family"] == "method":
        kind = "p20"
    if kind is None or kind == "not_written":
        cell.problem = (f"{terminal_path}: {arm['terminal']['status_key']} = {status!r} is not a written terminal state (D107: admission, provenance, code_not_clean, list and machine "
                        "refusals, needs_lead_rejudgment and unmapped statuses stop the evaluation)")
        return cell
    if kind in AUTHORIZATION_NEEDED:
        decided = None
        if arm["family"] == "method" and kind in ("implementation_error", "p20"):        # A5b review I2: the decision HB took, by its sha256 in the finalized record
            final, _ = finalized_record(cache, records)
            frow = ((final or {}).get("pairs") or {}).get(name)
            decided = frow.get("bz_decision") if isinstance(frow, dict) else None
        record, problem = cell_authorization(arm, row, kind, terminal_path, records[str(terminal_path)][0], records, cell.needed, decided=decided)
        if record is None:
            cell.problem = f"{terminal_path}: {status!r}: {problem}"
            return cell
        cell.authorization = record
    if arm["export_check"] is not None and kind == "output":
        flag = dotted(document, arm["export_check"]["key"])
        if not isinstance(flag, bool):
            cell.problem = f"{terminal_path}: {arm['export_check']['key']} is not a bool (the export check of a complete record)"
            return cell
        cell.export_suspicious = flag
    if kind == "output":
        missing = [rel for rel in arm["complete_files"] if not (base/fill(rel, arm, row)).is_file()]
        review_path = review_record_path(arm, row) if arm["family"] == "method" else None
        if review_path is not None and not review_path.is_file():
            missing.append(str(review_path))
        if missing:
            cell.problem = f"{base}: the complete record lacks {missing}"
            return cell
        if review_path is not None:                    # C22.H1-3, D111: only a finalized pair is written (review I7)
            key = fill(arm["review_record"]["status_key"], arm, row)
            review_doc = _read_record(review_path, records)
            value = dotted(review_doc, key)
            cell.review = dict(path=str(review_path), key=key, value=value, final=list(arm["review_record"]["final"]),
                               redo=review_doc.get("redo") if isinstance(review_doc, dict) else None)      # prereg revision 2 item 14, 9 (5 (i)): how often the review was redone
            if not _final(value, arm["review_record"]["final"]):
                cell.problem = f"{review_path}: {key} = {value!r} is not one of {arm['review_record']['final']}: the pair is not finalized (C22.H1, D111)"
                return cell
            cell.needed.append(review_path)
        cell.needed += [base/fill(rel, arm, row) for rel in arm["complete_files"]]
        field = arm["field"]
        if "dir" in field:
            directory = Path(fill(field["dir"], arm, row))
        else:
            path, digest = dotted(document, field["file_key"]), dotted(document, field["sha256_key"])
            froot = template_root(field["root"].replace("{arm}", arm["name"]))
            if not (isinstance(path, str) and path.startswith("/") and Path(path).name == "disp.npz" and _inside(path, froot) and not forbidden_path(path)
                    and is_hex64(digest)):
                cell.problem = f"{terminal_path}: {field['file_key']} = {path!r} / {field['sha256_key']} = {digest!r}: a disp.npz under {froot} and its sha256 required"
                return cell
            if not Path(path).is_file():
                cell.problem = f"{path}: the field the terminal record names is not there"
                return cell
            directory = Path(path).parent
            cell.field_sha256 = {str(path): digest}
        protocol = None
        if arm["kind"] == "configC":
            protocol = _read_record(directory/"protocol.json", records)
        cell.field_dir = str(directory)
        cell.needed += kind_files(arm, directory, protocol)
        cell.written, cell.outcome = True, "output"
    else:
        cell.written, cell.outcome = True, "no_output"
        cell.no_output_class = NO_OUTPUT_CLASS[kind]
        if kind == "p20":
            cell.cause = P20_WORD
    cell.needed = list(dict.fromkeys(cell.needed))
    if arm["family"] == "method":
        problem = method_cross_check(arm, row, cell, cache)
        if problem:
            cell.written, cell.outcome, cell.no_output_class, cell.problem = False, None, None, f"{name}: the method cell's files against the finalized and move records: {problem}"
            return cell
        cell.needed = list(dict.fromkeys(cell.needed))
    return cell


def check_cells(table, pairs, *, repo=None, cache=None):
    """Condition (b): every arm on every pair written; HoldoutRefused('arms') listing the cells that are not (no pair is evaluated then)."""
    cells, problems = {}, []
    cache = {} if cache is None else cache
    for arm in table["arms"]:
        for row in pairs:
            cell = classify_cell(arm, row, table, repo=repo, cache=cache)
            cells[(arm["name"], row[3])] = cell
            if not cell.written:
                problems.append(f"{arm['name']} {row[3]}: {cell.problem}")
    if problems:
        raise HoldoutRefused("arms", f"{len(problems)} cells are not written (D107): {problems[:10]}{' ...' if len(problems) > 10 else ''}")
    return cells


# ---- the repair ablation (prereg revision 2 item 4; contract revision 9 draft C24.2; A5b item 3): described only, decided before any truth -----------------------
def prerepair_paths(row):
    return Path(PREREPAIR_ROOT)/row[3]/PREREPAIR_RECORD, Path(PREREPAIR_BIG)/row[3]/PREREPAIR_DISP


def _ablation_problem(document, row, frow, disp_path):
    """Why the preparation record does not put the pair into the ablation, or None: schema, pair, passed, the three checks (C24.2-3), the output
    prerepair_disp.npz at its fixed place with a sha256, and the inputs' sha256 (t88/field_0600.npz, t88/run.json, repair/repair.json) the finalized record's
    part.t88_repair values (C24.2-2 "已定"; decision_log 2026-11-10 ~2:33 AM CST (4))."""
    if not isinstance(document, dict) or document.get("schema") != PREREPAIR_SCHEMA or document.get("pair") != row[3]:
        return f"not a preparation record of {row[3]} (schema {PREREPAIR_SCHEMA})"
    checks = document.get("checks") if isinstance(document.get("checks"), dict) else {}
    failed = [k for k in PREREPAIR_CHECKS if checks.get(k) is not True]
    if document.get("passed") is not True or failed:
        return f"the preparation did not pass: checks {failed or 'passed'} not passed; problems {document.get('problems')}"
    output = document.get("output") if isinstance(document.get("output"), dict) else {}
    if os.path.realpath(str(output.get("path"))) != os.path.realpath(str(disp_path)) or not is_hex64(output.get("sha256")):
        return f"output {output!r}: {disp_path} with its sha256 required"
    if not isinstance(frow, dict):
        return "no row of the finalized record"
    inputs = document.get("inputs") if isinstance(document.get("inputs"), dict) else {}
    part = frow.get("part") if isinstance(frow.get("part"), dict) else {}
    pinned = part.get("t88_repair") if isinstance(part.get("t88_repair"), dict) else {}
    for name in ("t88/field_0600.npz", "t88/run.json", "repair/repair.json"):
        got = (inputs.get(name) or {}).get("sha256") if isinstance(inputs.get(name), dict) else None
        if not is_hex64(got) or got != pinned.get(name):
            return f"inputs[{name}].sha256 {got!r} is not the finalized record's part.t88_repair {pinned.get(name)!r}"
    return None


def check_ablation(table, pairs, cells, cache):
    """(b) of the ablation, before any truth (C24.2-4 "消融的对在读真值之前定"): for each pair, the preparation record (PREREPAIR_ROOT/<pair>/prerepair.json,
    read: an input of the manifest) puts it into the ablation when _ablation_problem holds none and the method arm has an output there; its prerepair_disp.npz
    (PREREPAIR_BIG/<pair>/) is then an input too, at the record's output.sha256.  A pair without a record, or whose record does not hold, is listed with the
    reason and does not stop F1-F4 (C24.2-2).  Not an arm (the arm table is pinned), never certified, in no family.  Returns {pairs: {pair: ...}, excluded:
    {pair: reason}, records: {path: (sha256, document)}, needed: [paths], pinned_sha256: {disp path: sha256}}."""
    method = method_arm(table)
    final = (cache.get("finalized") or (None,))[0] or {}
    out = dict(pairs={}, excluded={}, records={}, needed=[], pinned_sha256={})
    for row in pairs:
        record_path, disp_path = prerepair_paths(row)
        if not record_path.is_file():
            out["excluded"][row[3]] = f"no preparation record {record_path} (experiments/hb/hb_prerepair.py did not write one)"
            continue
        document = _read_record(record_path, out["records"])
        out["needed"].append(record_path)
        cell = cells[(method["name"], row[3])]
        problem = _ablation_problem(document, row, (final.get("pairs") or {}).get(row[3]), disp_path)
        if problem is None and cell.outcome != "output":
            problem = f"the method arm has no output ({cell.status}): no repaired field to compare with"
        if problem is None and not disp_path.is_file():
            problem = f"{disp_path}: not there"
        if problem:
            out["excluded"][row[3]] = f"{record_path}: {problem}"
            continue
        out["needed"].append(disp_path)
        out["pinned_sha256"][str(disp_path)] = document["output"]["sha256"]
        out["pairs"][row[3]] = dict(record=str(record_path), record_sha256=out["records"][str(record_path)][0], disp=str(disp_path), disp_sha256=document["output"]["sha256"],
                                    line_status=cell.status, inputs={k: (v or {}).get("sha256") for k, v in document["inputs"].items() if isinstance(v, dict)})
    return out


def ablation_summary(ablation):
    """What the master lock line holds of the ablation (C24.2-4: written before the first truth, never changed): the pairs and the excluded pairs with reasons."""
    return dict(pairs=sorted(ablation["pairs"]), excluded=dict(sorted(ablation["excluded"].items())),
                rule="prereg revision 2 item 4, C24.2: described only; not an arm, not certified, in no family, no Holm, no change of any decision")


# ---- (c) the manifest ----------------------------------------------------------------------------------------------------------------------------------------
_ENTRY = re.compile(r"([0-9a-f]{64})  (/.+)", re.ASCII)
MANIFEST_HEADER = ("prereg_sha256", "arm_table_sha256", "unseal_sha256")
PQ_HEADER_BZ_II = "pq_method_side_decision"           # C22.H4-2 (ii): BZ's committed choice, F4 only described (C22.H7-3 row 1); otherwise PQ's release is read


@dataclass
class Manifest:
    sha256: str
    git: dict
    header: dict
    inputs: dict                # path -> sha256: exactly the files this entry reads, each hashed again just before the read
    records_file: dict          # the record-only file of the same commit (never read here: path, git blob id, last commit)
    text: str


def _header_value(key, value):
    if key.endswith("_sha256"):
        return is_hex64(value)
    return isinstance(value, str) and not value.startswith("/") and all(part not in ("", ".", "..") for part in value.split("/"))


def parse_manifest(raw, where=MANIFEST_REL):
    """The D108 file: input lines only (the lead, preregistration revision 1c item 1).
        # bayesreg.holdout.eval_manifest.v3                    (first line)
        # prereg_sha256 <64 hex> / # arm_table_sha256 <64 hex> / # unseal_sha256 <64 hex>   (once each)
        # pq_method_side_decision <repository-relative path>   (only when a method-side control of PQ failed and BZ chose (ii): F4 only described)
        <64 hex>  <absolute path>                               one per file this entry reads (PQ's list and outputs among them when PQ is released)
    PQ's release is not in this file: it is read from the code tree (PQ_RELEASE_SHA_REL, PQ_RELEASE_NOTE_REL; alignment item 5).  The record-only lines are in
    RECORDS_REL, committed with it; this entry does not read that file.  HoldoutRefused('manifest') on any other line or a duplicate path."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HoldoutRefused("manifest", f"{where}: not UTF-8") from None
    lines = text.splitlines()
    if not lines or lines[0] != f"# {SCHEMA_MANIFEST}":
        raise HoldoutRefused("manifest", f"{where}: the first line must be '# {SCHEMA_MANIFEST}'")
    header, inputs = {}, {}
    keys = MANIFEST_HEADER+(PQ_HEADER_BZ_II,)
    for number, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        parts = line.split(" ")
        if line.startswith("# ") and len(parts) == 3 and parts[1] in keys:
            if parts[1] in header or not _header_value(parts[1], parts[2]):
                raise HoldoutRefused("manifest", f"{where}:{number}: header {parts[1]} once, with a valid value")
            header[parts[1]] = parts[2]
            continue
        match = _ENTRY.fullmatch(line)
        if not match or match.group(2) in inputs:
            raise HoldoutRefused("manifest", f"{where}:{number}: only input lines '<64 hex>  <absolute path>' (one per path); record-only lines go to {RECORDS_REL}")
        inputs[match.group(2)] = match.group(1)
    if sorted(set(header) & set(MANIFEST_HEADER)) != sorted(MANIFEST_HEADER):
        raise HoldoutRefused("manifest", f"{where}: the header lines {list(MANIFEST_HEADER)} are required ({PQ_HEADER_BZ_II} only after BZ's (ii))")
    return header, inputs


def input_roles(table, pairs, cells, pq_states=None, pq_list=None, ablation=None):
    """{path: role} of every file this entry reads (C23.12-4): a cell's records and field files (the method's finalized and move records among them, A5b), the
    stage's manifest.json and pair.npz, (PQ released) PQ's sha256 list and the PQ outputs of the pairs where they are present (C22.H7-1), and the repair
    ablation's preparation records and prerepair_disp.npz (C24.2-2; A5b)."""
    roles = {str(p): "cell" for cell in cells.values() for p in cell.needed}
    stage = Path(table["stage_root"])
    roles.update({str(stage/row[3]/name): "stage" for row in pairs for name in ("manifest.json", "pair.npz")})
    for state in (pq_states or {}).values():
        roles.update({path: "pq" for path in state["files"].values()})
    if pq_list is not None:
        roles[str(pq_list)] = "pq_list"
    roles.update({str(p): "ablation" for p in (ablation or {}).get("needed", [])})
    return roles


def classify_listed(path, roles):
    """The C23.12-4 role of an input path (input_roles), or None (then the manifest holds a file outside C23.12-4)."""
    return roles.get(str(Path(path)))


def needed_inputs(table, pairs, cells, pq_states=None, pq_list=None, ablation=None):
    return set(input_roles(table, pairs, cells, pq_states, pq_list, ablation))


def records_file_state(manifest_git, repo=None):
    """The record-only file: tracked, unchanged, last changed in the manifest's commit.  Its content is not read here (only git's view of it)."""
    root = Path(REPO if repo is None else repo)
    record = tracked_unchanged(root/RECORDS_REL, "manifest", root)
    if record["last_commit"] != manifest_git.get("last_commit"):
        raise HoldoutRefused("manifest", f"{RECORDS_REL} was last committed in {record['last_commit']}, the manifest in {manifest_git.get('last_commit')}: one commit (D108)")
    blob = _git(root, "ls-files", "-s", "--", RECORDS_REL).stdout.split()
    return dict(record, blob=blob[1] if len(blob) > 1 else None, read=False)


def _method_side_decision(rel, repo):
    """BZ's (ii) (C22.H4-2): a committed record of SCHEMA_PQ_METHOD_SIDE written by BZ or "lead on BZ's behalf", with words, times, a reason and choice "ii";
    a record that does not hold refuses."""
    raw, digest, git = read_committed(rel, "manifest", repo)
    document = strict_json(raw, rel)
    keys = ("schema", "written_by", "words", "time_local", "utc", "reason", "choice")
    problems = []
    if not isinstance(document, dict) or set(document) - {"comment"} != set(keys) or document.get("schema") != SCHEMA_PQ_METHOD_SIDE:
        problems.append(f"exactly the keys {list(keys)} and schema {SCHEMA_PQ_METHOD_SIDE}")
    else:
        if document["written_by"] not in AUTHORIZATION_WRITERS:
            problems.append(f"written_by {document['written_by']!r}")
        problems += [f"{k}: non-empty" for k in ("words", "time_local", "reason") if not isinstance(document[k], str) or not document[k].strip()]
        problems += [p for p in (utc_problem(document["utc"], "utc"),) if p]
        if document["choice"] != "ii":
            problems.append(f"choice {document['choice']!r}: 'ii' (F4 only described); after (i) the PQ rerun is released instead")
    if problems:
        raise HoldoutRefused("manifest", f"{rel}: {problems}")
    return dict(method_side="bz_ii", decision=dict(path=rel, sha256=digest, git=git, document=document), passed=None, entries={}, list=None)


def check_pq_release(table, header, inputs, repo=None):
    """PQ's half of (c) (alignment item 5; C22.H4-5, C22.H7-1 Ev1).  BZ's (ii): _method_side_decision, no PQ file read.  Otherwise the release, from the code
    tree: PQ_RELEASE_SHA_REL committed and unchanged, one line '<sha256>  <absolute path of pq_outputs_manifest.txt>' (the list under pq.small_root; neither
    file there refuses: PQ's outputs could not be found) -- then Ev1's global part, recorded and never refused: PQ_RELEASE_NOTE_REL committed, unchanged, in the
    same commit, experiments/pq/pq_eval.release_note_problem's rule (one line 放行：<that sha256>, no word of refusal); the list an input of this manifest at
    the released sha256.  The list is read (its bytes at this manifest's sha256) and parsed: {absolute path: sha256}."""
    if PQ_HEADER_BZ_II in header:
        if header[PQ_HEADER_BZ_II] != PQ_METHOD_SIDE_REL:          # the interface alignment revision 1 item 3; prereg revision 2 item 12 (A5b item 9)
            raise HoldoutRefused("manifest", f"{MANIFEST_REL}: {PQ_HEADER_BZ_II} {header[PQ_HEADER_BZ_II]!r}: BZ's (ii) record is {PQ_METHOD_SIDE_REL} and no other path")
        return _method_side_decision(header[PQ_HEADER_BZ_II], repo)
    _, pq_eval, _ = _pq_modules()
    try:
        raw, digest, git = read_committed(PQ_RELEASE_SHA_REL, "manifest", repo)
    except HoldoutRefused as exc:
        raise HoldoutRefused("manifest", f"neither PQ's release ({exc}) nor BZ's (ii) in the manifest header: PQ's outputs cannot be found (alignment item 5)") from None
    lines = [line.strip() for line in raw.decode("utf-8-sig", errors="replace").splitlines() if line.strip()]
    match = _ENTRY.fullmatch(lines[0]) if len(lines) == 1 else None
    small = Path(table["pq"]["small_root"])
    if match is None or Path(match.group(2)).name not in PQ_LIST_NAMES or not _inside(match.group(2), small) or forbidden_path(match.group(2)):
        raise HoldoutRefused("manifest", f"{PQ_RELEASE_SHA_REL}: one line '<64 hex>  <absolute path of {PQ_LIST_NAMES[0]} under {small}>' required")
    list_sha, list_path = match.group(1), match.group(2)
    record = dict(method_side="released", sha_file=dict(path=PQ_RELEASE_SHA_REL, sha256=digest, last_commit=git.get("last_commit")), list=list_path,
                  released_sha256=list_sha, problems=[], entries={})
    try:
        note_raw, note_digest, note_git = read_committed(PQ_RELEASE_NOTE_REL, "manifest", repo)
        record["note_file"] = dict(path=PQ_RELEASE_NOTE_REL, sha256=note_digest, last_commit=note_git.get("last_commit"))
        if note_git.get("last_commit") != git.get("last_commit"):
            record["problems"].append(f"{PQ_RELEASE_SHA_REL} and {PQ_RELEASE_NOTE_REL} are not in one commit ({git.get('last_commit')}, {note_git.get('last_commit')})")
        problem = pq_eval.release_note_problem(note_raw.decode("utf-8-sig", errors="replace"), list_sha)
        if problem:
            record["problems"].append(problem)
    except HoldoutRefused as exc:
        record["problems"].append(f"{PQ_RELEASE_NOTE_REL}: {exc}")
    if inputs.get(list_path) is None:
        raise HoldoutRefused("manifest", f"PQ's list {list_path} is not an input of {MANIFEST_REL}")
    if inputs[list_path] != list_sha:
        record["problems"].append(f"{list_path}: this manifest lists {inputs[list_path]}, the release {list_sha}")
    text = verified_bytes(list_path, inputs).decode("utf-8", errors="replace")
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        entry = _ENTRY.fullmatch(line)
        if not entry or entry.group(2) in record["entries"]:
            record["problems"].append(f"{list_path}:{number}: not '<64 hex>  <absolute path>' (one per path)")
            continue
        record["entries"][entry.group(2)] = entry.group(1)
    record["passed"] = not record["problems"]
    return record


def pq_states(table, pairs, cells, release):
    """{pair: PQ state} from PQ's list (alignment item 5: never from a path template; PQH's §4.2): present (pq.json under small_root and pq.npz, baselines.npz
    under big_root, each once, in the pair's own directory <root>/[...]/<pair>/), no_method_output, pq_failed / pq_absent (listed; C22.H6-3, never read) or
    missing.  BZ's (ii): not_read.  Ev1 per present pair: the release holds and each file's sha256 is the same in the list and in this manifest (written in
    (c), compared again when the bytes are read: verified_bytes)."""
    method = method_arm(table)
    small, big = Path(table["pq"]["small_root"]), Path(table["pq"]["big_root"])
    by_pair = {}
    for path in release["entries"]:
        p = Path(path)
        if p.name in PQ_FILES or p.name in PQ_NOTES:
            by_pair.setdefault(p.parent.name, []).append(p)
    states = {}
    for row in pairs:
        cell = cells[(method["name"], row[3])]
        if release["method_side"] != "released":
            states[row[3]] = dict(state="not_read", reason="BZ chose (ii) after a method-side control of PQ failed (C22.H4-2): F4 only described", files={})
            continue
        if cell.outcome != "output":
            states[row[3]] = dict(state="no_method_output", reason=f"the method has no output ({cell.status})", files={})
            continue
        listed = by_pair.get(row[3], [])
        names = [p.name for p in listed]
        files, misplaced = {}, []
        for p in listed:
            root = small if PQ_FILES.get(p.name, "small_root") == "small_root" else big
            if not _inside(p, root) or forbidden_path(p):
                misplaced.append(str(p))
            elif p.name in PQ_FILES:
                files[p.name] = str(p)
        if misplaced or any(names.count(n) > 1 for n in names):
            states[row[3]] = dict(state="missing", reason=f"PQ's list holds misplaced or repeated files of the pair: {misplaced or sorted(names)}", files={})
        elif set(files) == set(PQ_FILES):
            states[row[3]] = dict(state="present", reason=None, files=files)
        else:
            note = next((state for name, state in PQ_NOTES.items() if name in names), None)
            states[row[3]] = dict(state=note or "missing", reason=f"PQ's list holds {sorted(names) or 'nothing'} for the pair (C22.H6-3)", files={})
    return states


def read_input_lock(repo=None):
    """The step-8 input lock (INPUT_LOCK_REL; HB's format): committed and unchanged, schema INPUT_LOCK_SCHEMA, rows of kind prediction (subject, path,
    sha256) and method_field (pair, path, sha256).  A lock that is not there or does not hold is recorded, not refused: every pair's PQ then stops at the
    lock check (F4 不判（评测侧）), F1-F3 go on (the lead's ruling (10), 11-09).  Returns {path, sha256, last_commit, problem, predictions, fields}."""
    out = dict(path=INPUT_LOCK_REL, sha256=None, last_commit=None, problem=None, predictions={}, fields={})
    try:
        raw, digest, git = read_committed(INPUT_LOCK_REL, "manifest", repo)
    except HoldoutRefused as exc:
        return dict(out, problem=str(exc))
    out.update(sha256=digest, last_commit=git.get("last_commit"))
    try:
        document = strict_json(raw, INPUT_LOCK_REL)
    except HoldoutRefused as exc:
        return dict(out, problem=str(exc))
    if not isinstance(document, dict) or document.get("schema") != INPUT_LOCK_SCHEMA or not isinstance(document.get("rows"), list):
        return dict(out, problem=f"{INPUT_LOCK_REL}: schema {INPUT_LOCK_SCHEMA} with rows required")
    for row in document["rows"]:
        if not isinstance(row, dict):
            return dict(out, problem=f"{INPUT_LOCK_REL}: a row that is not an object")
        if row.get("kind") == "prediction" and isinstance(row.get("path"), str):
            key = (row.get("subject"), Path(row["path"]).name)
            if key in out["predictions"]:
                return dict(out, problem=f"{INPUT_LOCK_REL}: two rows for {key}")
            out["predictions"][key] = row.get("sha256")
        elif row.get("kind") == "method_field":
            out["fields"][row.get("pair")] = row.get("sha256")
    return out


def pq_input_lock_check(document, row, lock):
    """C22.H4-2 Hm2 after the lock (revision 7a m2; the lead's ruling (10) on the PQH review I9): each side's probs36, structure_groups and provenance sha256
    that pq.json records (inputs.<side>.probs36_sha256, ...) equal the step-8 input lock's row of that subject and file; the method's field_0600.npz too when
    pq.json names it (method_files['t88/field_0600.npz']).  Any difference, a missing value or a lock that does not hold: not passed."""
    rows, problems = [], []
    if lock.get("problem"):
        problems.append(lock["problem"])
    for side, subject in (("fixed", row[1]), ("moving", row[2])):
        recorded = dotted(document, f"inputs.{side}") or {}
        for name, key in PQ_INPUT_KEYS.items():
            mine, locked = recorded.get(key) if isinstance(recorded, dict) else None, lock["predictions"].get((subject, name))
            same = is_hex64(mine) and mine == locked
            rows.append(dict(side=side, subject=subject, file=name, pq_json=mine, input_lock=locked, same=same))
            if not same:
                problems.append(f"{side} {subject} {name}: pq.json {mine!r}, the input lock {locked!r}")
    field = ((document.get("method_files") or {}).get("t88/field_0600.npz") or {}).get("sha256") if isinstance(document.get("method_files"), dict) else None
    if field is not None:
        locked = lock["fields"].get(row[3])
        rows.append(dict(file="t88/field_0600.npz", pq_json=field, input_lock=locked, same=field == locked))
        if field != locked:
            problems.append(f"t88/field_0600.npz: pq.json {field!r}, the input lock {locked!r}")
    return dict(passed=not problems, rows=rows, problems=problems, lock=dict(path=lock.get("path"), sha256=lock.get("sha256"), last_commit=lock.get("last_commit")),
                rule="C22.H4-2 Hm2 after the step-8 input lock (revision 7a m2), checked by the evaluation entry (the lead, 11-09, PQH review I9)")


def pq_ev1(states, release, listed):
    """Ev1 of each pair with PQ outputs (C22.H7-1): the release holds and every PQ file's sha256 is the same in PQ's list and in this manifest."""
    summary = {k: release.get(k) for k in ("sha_file", "note_file", "list", "released_sha256", "passed", "problems")}
    for state in states.values():
        if state["state"] != "present":
            continue
        files = {name: dict(path=path, manifest_sha256=listed.get(path), pq_list_sha256=release["entries"].get(path)) for name, path in state["files"].items()}
        same = all(f["manifest_sha256"] is not None and f["manifest_sha256"] == f["pq_list_sha256"] for f in files.values())
        state["ev1"] = dict(passed=bool(release["passed"] is True and same), release=summary, files=files)
    return states


def check_manifest(table, pairs, cells, *, prereg_sha256, arm_table_sha256, unseal_sha256, repo=None, ablation=None):
    """Condition (c): the manifest committed and unchanged; its header the pinned preregistration, arm table and unseal object; PQ's release (or BZ's choice
    (ii)) and PQ's states from its list; its lines exactly the files this entry reads (all present); the records read in (b) at their listed sha256; a field
    whose terminal record names its sha256 listed at that sha256, a method cell's file at the finalized record's sha256 (A5b), a prerepair_disp.npz at its
    preparation record's; the record-only file committed with it.  Returns (Manifest, inputs, PQ release, PQ states)."""
    raw, digest, git = read_committed(MANIFEST_REL, "manifest", repo)
    header, inputs = parse_manifest(raw)
    want = dict(prereg_sha256=prereg_sha256, arm_table_sha256=arm_table_sha256, unseal_sha256=unseal_sha256)
    if {k: header[k] for k in MANIFEST_HEADER} != want:
        raise HoldoutRefused("manifest", f"{MANIFEST_REL}: header {header} is not {want}")
    records = records_file_state(git, repo)
    release = check_pq_release(table, header, inputs, repo)
    states = pq_states(table, pairs, cells, release)
    roles = input_roles(table, pairs, cells, states, release.get("list"), ablation)
    unclassified = [p for p in inputs if classify_listed(p, roles) is None]
    if unclassified:
        raise HoldoutRefused("manifest", f"{MANIFEST_REL}: {len(unclassified)} input lines are outside the files this entry reads (C23.12-4, D108): {unclassified[:5]}")
    missing = sorted(set(roles)-set(inputs))
    if missing:
        raise HoldoutRefused("manifest", f"{len(missing)} files this entry reads are not in the manifest: {missing[:5]}")
    absent = sorted(p for p in inputs if not Path(p).is_file())
    if absent:
        raise HoldoutRefused("manifest", f"{len(absent)} input files are not there: {absent[:5]}")
    for cell in cells.values():
        for path, (sha, _) in cell.records.items():
            if inputs.get(path) != sha:
                raise HoldoutRefused("manifest", f"{path}: read in (b) at sha256 {sha}, listed {inputs.get(path)}")
        for path, sha in (cell.field_sha256 or {}).items():
            if inputs.get(path) != sha:
                raise HoldoutRefused("manifest", f"{path}: the terminal record {cell.terminal} names sha256 {sha}, the manifest lists {inputs.get(path)}")
        for path, sha in (cell.pinned_sha256 or {}).items():
            if inputs.get(path) != sha:
                raise HoldoutRefused("manifest", f"{path}: the finalized record names sha256 {sha} (through the move records), the manifest lists {inputs.get(path)}")
    for path, (sha, _) in ((ablation or {}).get("records") or {}).items():
        if inputs.get(path) != sha:
            raise HoldoutRefused("manifest", f"{path}: the preparation record read in (b) at sha256 {sha}, listed {inputs.get(path)}")
    for path, sha in ((ablation or {}).get("pinned_sha256") or {}).items():
        if inputs.get(path) != sha:
            raise HoldoutRefused("manifest", f"{path}: its preparation record names sha256 {sha}, the manifest lists {inputs.get(path)}")
    if release["method_side"] == "released":
        pq_ev1(states, release, inputs)
        release["input_lock"] = read_input_lock(repo)
    return Manifest(digest, git, header, inputs, records, raw.decode("utf-8")), inputs, release, states


def check_stage(table, pairs, listed):
    """Each pair's stage manifest (an input of the manifest, hashed first): its subjects are the row's, in the row's direction (C23.12-2 (a)); the grid is
    read here, no member of pair.npz is opened.  HoldoutRefused('pairs') otherwise.
    Why after (c) and not inside (a) (review m9): the stage manifest is a subject's file and is read only at its listed sha256, so the manifest of (c) comes
    first; (a) itself never opens a subject file.  Every pair is a row of the pinned list because the pairs are taken from it (load_pairs), and this check
    adds the stage's direction; nothing of a subject is read before it, and no truth before (d)."""
    stage = Path(table["stage_root"])
    grids = {}
    for row in pairs:
        path = stage/row[3]/"manifest.json"
        verify_files([path], listed)
        grid, metadata, selected = _read_manifest(path)
        if (selected["fixed_subject"], selected["moving_subject"]) != (row[1], row[2]):
            raise HoldoutRefused("pairs", f"{path}: {selected['fixed_subject']} <- {selected['moving_subject']} is not the list's row {row[1]} <- {row[2]}")
        grids[row[3]] = metadata
    return grids


def verified_bytes(path, listed):
    """Bytes of a listed file, hashed just before use; ManifestChanged if they differ from the manifest (the evaluation stops)."""
    raw = Path(path).read_bytes()
    if listed.get(str(path)) != sha_bytes(raw):
        raise ManifestChanged("manifest_changed", f"{path}: sha256 {sha_bytes(raw)} differs from the manifest's {listed.get(str(path))}; the evaluation stops")
    return raw


def verify_files(paths, listed):
    for path in paths:
        if listed.get(str(path)) is None or file_sha256(path) != listed[str(path)]:
            raise ManifestChanged("manifest_changed", f"{path}: missing from the manifest or not its sha256; the evaluation stops")


# ---- (d) the locks and the reevaluation authorization (D109) ------------------------------------------------------------------------------------------
REEVAL_LOCK = re.compile(r"reeval_([0-9]{1,3})\.lock", re.ASCII)


def earlier_reeval_pairs(n, repo=None):
    """{pair: where} of every pair named by another committed reevaluation authorization or by a reevaluation lock line (review m10: a pair is reevaluated
    at most once, also a pair the first run never reached)."""
    root = Path(REPO if repo is None else repo)
    seen = {}
    folder = root/REEVAL_DIR_REL
    for path in sorted(folder.glob("holdout_B_v1_reeval_*.json")) if folder.is_dir() else []:
        match = REEVAL_NAME.fullmatch(path.name)
        if not match or int(match.group(1)) == n:
            continue
        try:
            pairs = json.loads(path.read_text(encoding="utf-8")).get("pairs")
        except (OSError, ValueError, AttributeError):
            raise HoldoutRefused("reeval", f"{path}: an earlier reevaluation authorization that cannot be read") from None
        for name in pairs if isinstance(pairs, list) else []:
            seen.setdefault(name, str(path))
    for path, line in run_lines():
        if path == Path(MASTER_LOCK):
            continue
        if not isinstance(line.get("pairs"), list):
            raise HoldoutRefused("reeval", f"{path}: a reevaluation lock line without its pairs")
        for name in line["pairs"]:
            seen.setdefault(name, str(path))
    return seen


def load_reeval(path, *, manifest_sha256, pairs, repo=None):
    """A committed reevaluation authorization (D109): splits/bayesreg_v1/holdout_B_v1_reeval_<n>.json, unchanged against HEAD; written by BZ or the lead
    on BZ's behalf, with words, time, a reason and its class (REEVAL_REASONS: the evaluation's own infrastructure failure only, review m18); the master
    lock's sha256 and the manifest of the master lock line (the same manifest only); pairs of the list that no earlier run of the lock completed and that
    no other authorization or reevaluation lock names (review m10).  Returns (n, document, record); HoldoutRefused('reeval')."""
    root = Path(REPO if repo is None else repo)
    target = Path(path).resolve()
    match = REEVAL_NAME.fullmatch(target.name)
    if target.parent != (root/REEVAL_DIR_REL).resolve() or not match:
        raise HoldoutRefused("reeval", f"{path}: a reevaluation authorization is {REEVAL_DIR_REL}/holdout_B_v1_reeval_<n>.json")
    rel = target.relative_to(root.resolve()).as_posix()
    raw, digest, git = read_committed(rel, "reeval", root)
    document = strict_json(raw, rel)
    keys = ("schema", "written_by", "words", "time_local", "utc", "reason", "reason_class", "master_lock_sha256", "manifest_sha256", "pairs", "previous_out")
    if not isinstance(document, dict) or sorted(document) != sorted(keys) or document["schema"] != SCHEMA_REEVAL:
        raise HoldoutRefused("reeval", f"{rel}: exactly the keys {list(keys)} and schema {SCHEMA_REEVAL}")
    problems = []
    if document["written_by"] not in REEVAL_WRITERS:
        problems.append(f"written_by {document['written_by']!r}: one of {REEVAL_WRITERS}")
    for key in ("words", "time_local", "reason"):
        if not isinstance(document[key], str) or not document[key].strip():
            problems.append(f"{key}: non-empty")
    if document["reason_class"] not in REEVAL_REASONS:
        problems.append(f"reason_class {document['reason_class']!r}: one of {REEVAL_REASONS} (D109: the evaluation's own infrastructure failures only)")
    problems += [p for p in (utc_problem(document["utc"], "utc"),) if p]
    if not isinstance(document["previous_out"], list) or not all(isinstance(o, str) for o in document["previous_out"]):
        problems.append("previous_out: a list of the earlier runs' output directories (recorded; the lock lines decide)")
    if not Path(MASTER_LOCK).is_file():
        raise HoldoutRefused("reeval", f"{MASTER_LOCK}: there is no master lock; a reevaluation follows a first run")
    master_sha = file_sha256(MASTER_LOCK)
    master = lock_lines(MASTER_LOCK)[0]
    if document["master_lock_sha256"] != master_sha:
        problems.append(f"master_lock_sha256 {document['master_lock_sha256']!r} is not the master lock's {master_sha}")
    if document["manifest_sha256"] != master.get("manifest_sha256") or manifest_sha256 != master.get("manifest_sha256"):
        problems.append(f"the manifest {manifest_sha256} is not the master lock's {master.get('manifest_sha256')} (D109: the same manifest only)")
    names = {row[3] for row in pairs}
    asked = document["pairs"]
    if not isinstance(asked, list) or not asked or len(set(asked)) != len(asked) or not set(asked) <= names:
        problems.append("pairs: a non-empty list of distinct pairs of the list")
        asked = []
    done = completed_pairs()
    earlier = earlier_reeval_pairs(int(match.group(1)), root)
    for name in asked:
        if name in done:
            problems.append(f"{name} was evaluated completely by {done[name]}")
        if (Path(PAIR_LOCK_DIR)/f"{name}.reeval.lock").exists() or name in earlier:
            problems.append(f"{name} was reevaluated once already or is named by {earlier.get(name, 'its .reeval.lock')} (D109: at most once)")
    if problems:
        raise HoldoutRefused("reeval", f"{rel}: {problems}")
    return int(match.group(1)), document, dict(path=rel, sha256=digest, git=git)


def run_lines():
    """[(lock path, line)] of the master lock and the reevaluation locks reeval_<n>.lock in the order of n (review I8: the decision code's only source of
    the runs)."""
    out = [(Path(MASTER_LOCK), line) for line in lock_lines(MASTER_LOCK)]
    found = []
    for path in Path(EVAL_DIR).glob("reeval_*.lock") if Path(EVAL_DIR).is_dir() else []:
        match = REEVAL_LOCK.fullmatch(path.name)
        if match:
            found.append((int(match.group(1)), path))
    for _, path in sorted(found):
        out += [(path, line) for line in lock_lines(path)]
    return out


def run_outs():
    """The output directories of every run of the lock (master lock and reevaluation locks), in order."""
    return [line.get("out_dir") for _, line in run_lines() if line.get("out_dir")]


def completed_pairs():
    done = {}
    for out in run_outs():
        for path in sorted((Path(out)/"pairs").glob("*.json")) if (Path(out)/"pairs").is_dir() else []:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if record.get("complete") is True:
                done.setdefault(record.get("pair"), out)
    return done


def take_pair_lock(admission, row, line):
    """The pair lock before its truth is read (D109): <pair>.lock on a first read; on an authorized reevaluation <pair>.reeval.lock when <pair>.lock exists."""
    state = _state(admission, "once")
    if not state["locked"]:
        raise HoldoutRefused("once", "the master (or reevaluation) lock is not taken by this admission")
    name = row[3]
    first = Path(PAIR_LOCK_DIR)/f"{name}.lock"
    reeval = admission.record.get("reeval")
    if reeval is not None and name not in reeval["pairs"]:
        raise HoldoutRefused("once", f"{name} is not a pair of the reevaluation authorization")
    if reeval is None or not first.exists():
        take_lock(first, dict(line, pair=name, read="first"))
        state["pair_locks"][name] = str(first)
    else:
        path = Path(PAIR_LOCK_DIR)/f"{name}.reeval.lock"
        take_lock(path, dict(line, pair=name, read="reevaluation"))
        state["pair_locks"][name] = str(path)
    return state["pair_locks"][name]


# ---- stage 1: the whole admission ---------------------------------------------------------------------------------------------------------------------
def check_out_dir(out_dir, table, pairs, cells=None):
    """--out-dir: new; not under a seq_vp_3d tree or the code tree; not inside an evaluated cell, a field directory, PQ's directories, the stage or any root
    the spy watches (the entry could not write there).  The evaluation's lock directories not under a watched root either (review2 m4: the master lock is
    written there last, under the spy)."""
    out = Path(out_dir)
    if out.exists():
        raise HoldoutRefused("output", f"{out} exists: a new --out-dir for every run; nothing is overwritten")
    if _under_read_only(out) or _inside(out, REPO) or forbidden_path(out):
        raise HoldoutRefused("output", f"{out}: not under a seq_vp_3d tree, the data root or inside the code tree")
    if not _inside(out, EVAL_DIR) or Path(out).resolve() == Path(EVAL_DIR).resolve() or _inside(out, PAIR_LOCK_DIR) or REEVAL_LOCK.fullmatch(out.name) \
            or out.name in (Path(MASTER_LOCK).name, Path(PAIR_LOCK_DIR).name):
        raise HoldoutRefused("output", f"{out}: --out-dir is a new directory under {EVAL_DIR} (not the directory itself, its locks or pair_locks; contract revision 9 draft "
                                       "C24.8 item 15, decision_log 2026-11-10 ~2:33 AM CST (6)): the review spy of step 5 forbids that root")
    watched = watched_roots(table)
    locks = [d for d in (Path(EVAL_DIR), Path(PAIR_LOCK_DIR)) if any(_inside(d, root) for root in watched)]
    if locks:
        raise HoldoutRefused("output", f"the lock directories {[str(d) for d in locks]} are under a directory the spy watches: the master lock could not be written")
    roots = [record_dir(arm, row) for arm in table["arms"] for row in pairs]+[Path(c.field_dir) for c in (cells or {}).values() if c.field_dir]
    if any(_inside(out, root) for root in roots):
        raise HoldoutRefused("output", f"{out}: inside an evaluated cell or its field directory (hard constraint 5)")
    if any(_inside(out, root) for root in watched):
        raise HoldoutRefused("output", f"{out}: under a directory the spy watches ({[str(r) for r in watched if _inside(out, r)]}): the entry could not write there")
    return out


def template_root(template):
    """The directory above every directory of a template (the template up to its first field); a plain path is its own root."""
    if "{" not in template:
        return Path(template)
    head = template.split("{", 1)[0]
    return Path(head) if head.endswith("/") else Path(head).parent


def imported_from(root):
    """Whether the bayesreg package and the helper modules this process imported (experiments/pq, experiments/holdout, scripts) are the ones of the code tree
    checked (review m7, review2 N1)."""
    import bayesreg
    if not _inside(Path(bayesreg.__file__).parent, Path(root)/"bayesreg"):
        return False
    return all(_inside(origin, Path(root)) for origin in helper_origins().values())


def open_holdout(out_dir, *, reeval_path=None, run=None, cgroup=PROC_CGROUP, repo=None, decision=None, pq_replay=None):
    """Stage 1: the spy (C23.6-5), (0), (a), (b) with the machine gate, (c), the out-dir rules, (d).  Returns (admission, context); nothing of a subject
    is read.  On a refusal the spy is stopped; otherwise it stays on until evaluate_holdout ends (context["spy"])."""
    root = Path(REPO if repo is None else repo)
    stop_spy()
    spy = start_spy([DATA_ROOT])
    try:
        return _open_holdout(out_dir, root, spy, reeval_path=reeval_path, run=run, cgroup=cgroup, decision=decision, pq_replay=pq_replay)
    except BaseException:
        stop_spy()
        raise


def _open_holdout(out_dir, root, spy, *, reeval_path, run, cgroup, decision, pq_replay):
    identity = rule_one(root)                                                     # rule one (C24.3-4, -7): P1 from runs/holdout_B_v1/pinned_code.json; before anything else
    gate = check_decision(decision, root)                                        # (0)
    replay = check_pq_replay(pq_replay, root)
    origins = helper_origins()                                                    # experiments/pq, holdout_common, scripts: this tree's (review2 N1)
    unseal, bound, pairs, admitted = admit_holdout(repo=root)                     # (a)
    prereg_raw, prereg = check_prereg(root)                                       # (b)
    table, table_record = load_arm_table(prereg_raw, root)
    spy.watch(watched_roots(table))
    cache = {}
    cells = check_cells(table, pairs, repo=root, cache=cache)
    ablation = check_ablation(table, pairs, cells, cache)                         # the repair ablation's pairs, before any truth (C24.2-4)
    machine = machine_gate(run=run, cgroup=cgroup)
    manifest, listed, release, states = check_manifest(table, pairs, cells, prereg_sha256=prereg["sha256"],                                # (c)
                                                       arm_table_sha256=table_record["sha256"], unseal_sha256=unseal.sha256, repo=root, ablation=ablation)
    spy.allow(listed)
    check_stage(table, pairs, listed)
    out = check_out_dir(out_dir, table, pairs, cells)
    reeval = None
    if reeval_path is None:
        if Path(MASTER_LOCK).exists():
            raise HoldoutRefused("once", f"{MASTER_LOCK} exists: the hold-out is evaluated once; a reevaluation needs a committed authorization (D109)")
    else:
        n, document, record = load_reeval(reeval_path, manifest_sha256=manifest.sha256, pairs=pairs, repo=root)
        reeval = dict(n=n, pairs=list(document["pairs"]), record=record, reason=document["reason"], reason_class=document["reason_class"])
    code = code_record(root)
    if not code["clean"]:
        raise HoldoutRefused("code", f"the code tree is not clean ({list(CLEAN_DIRS)}): {code['status'][:200]!r}")
    if not imported_from(root):
        raise HoldoutRefused("code", f"the imported bayesreg or a helper module ({origins}) is not of the code tree {root}")
    code["helper_modules"] = origins
    line = dict(schema=SCHEMA_RUN, unseal_sha256=unseal.sha256, prereg_sha256=prereg["sha256"], arm_table_sha256=table_record["sha256"],
                manifest_sha256=manifest.sha256, manifest_commit=manifest.git.get("last_commit"), code_commit=code["head"], pinned_commit=identity.get("pinned"),
                out_dir=str(out.resolve()), ablation=ablation_summary(ablation), utc=utc(), pid=os.getpid())
    if reeval is None:
        lock = dict(path=str(MASTER_LOCK), line=take_lock(MASTER_LOCK, line))                       # (d), after every check
    else:
        path = Path(EVAL_DIR)/f"reeval_{reeval['n']}.lock"
        lock = dict(path=str(path), line=take_lock(path, dict(line, reeval=reeval["record"], pairs=reeval["pairs"])))
    record = dict(gate=gate, pq_replay=replay, admission=admitted, prereg=prereg, arm_table=table_record, machine=machine,
                  manifest=dict(path=MANIFEST_REL, sha256=manifest.sha256, git=manifest.git, n_inputs=len(manifest.inputs), records_file=manifest.records_file),
                  pq_release={k: (v if k != "input_lock" else {kk: vv for kk, vv in v.items() if kk not in ("predictions", "fields")})
                              for k, v in release.items() if k != "entries"}, code=code, rule_one=identity, reeval=reeval, lock=lock,
                  ablation=ablation_summary(ablation))
    subjects = [s for row in pairs for s in (row[1], row[2])]
    admission = _issue("holdout", subjects, pairs, record)
    _STATE[id(admission)]["locked"] = True
    out.mkdir(parents=True)
    return admission, dict(table=table, cells=cells, listed=listed, out=out, unseal=unseal, bound=bound, line=line, record=record, spy=spy,
                           pq_states=states, pq_release=release, ablation=ablation)


# ---- the truth-reader gate's admission (C23.12-10; development subjects only) ---------------------------------------------------------------------------
def admit_truth_gate(*, data_root=None, repo=None, allowlist_path=None):
    """The 20 development subjects of stage_L2R_val_v1 and their 19 pairs: the development allowlist source DEV_STAGE (not never_for_training, on the allowlist,
    on no sealed list, no B-line training subject; experiments/unv2/eval_seg.admit_subjects) and the bench's pair guard (truth.admit_pair: names, manifest,
    check_pair_subjects).  No unseal object.  Returns (admission, pair rows with grid and guard).  SubjectSealed / HoldoutRefused('dev')."""
    from bayesreg.eval.truth import admit_pair
    from bayesreg.seg.unet.data import load_folds
    root = Path(REPO if repo is None else repo)
    allowlist_path = root/ALLOWLIST_REL if allowlist_path is None else Path(allowlist_path)
    allowlist = load_dev_allowlist(allowlist_path)
    sealed = guard_sealed_lists()
    training = set(load_folds(root/FOLDS_REL).training_subjects)
    sources = {s["name"]: s for s in allowlist.sources}
    if DEV_STAGE not in sources or sorted(sources[DEV_STAGE]["subjects"]) != sorted(DEV_SUBJECTS):
        raise HoldoutRefused("dev", f"the allowlist source {DEV_STAGE} is not exactly the 20 development subjects")
    for subject in DEV_SUBJECTS:
        if lists_of(sealed, subject):
            raise SubjectSealed(f"{subject} is on the sealed list {list(lists_of(sealed, subject))}")
        if subject in allowlist.never_for_training or not allowlist.allows(subject) or subject in training:
            raise SubjectSealed(f"{subject}: not a development subject of the gate (never_for_training, off the allowlist, or a B-line training subject)")
        holding = sorted(s["name"] for s in allowlist.sources if subject in s["subjects"])
        if holding != [DEV_STAGE]:
            raise SubjectSealed(f"{subject} is in the allowlist sources {holding}, not only {DEV_STAGE}")
    data = Path(root/"data"/"three_d" if data_root is None else data_root)
    rows = []
    for index, (fixed, moving) in enumerate(DEV_PAIRS):
        name = PAIR_NAME.format(fixed=fixed, moving=moving)
        pair_dir, grid, metadata, guard = admit_pair(data, DEV_STAGE, name, allowlist, sealed)
        if (guard["fixed_subject"], guard["moving_subject"]) != (fixed, moving):
            raise HoldoutRefused("dev", f"{name}: the manifest's subjects are not {fixed} <- {moving}")
        rows.append(dict(row=(index, fixed, moving, name), pair_dir=pair_dir, grid=grid, metadata=metadata, guard=guard))
    admission = _issue("truth_gate", DEV_SUBJECTS, tuple(r["row"] for r in rows),
                       dict(allowlist_path=str(allowlist_path), allowlist_sha256=allowlist.sha256, stage=DEV_STAGE, data_root=str(data), sealed_lists=sealed.record))
    return admission, rows


def take_gate_lock(admission, line):
    state = _state(admission, "once")
    if admission.mode != "truth_gate":
        raise HoldoutRefused("once", "the gate lock belongs to a truth-gate admission")
    take_lock(TRUTH_GATE_LOCK, line)
    state["locked"] = True
    return dict(path=str(TRUTH_GATE_LOCK), line=line)


# ---- stage 2: the one reader of the truth (C23.6-3 class E) -------------------------------------------------------------------------------------------
def _load_label_volume(path):
    """load_scalar(labels=True) of one label NIfTI and the stored dtype from its header.  Called by read_truth only."""
    import nibabel
    volume = load_scalar(path, labels=True)
    stored = str(nibabel.load(str(path)).get_data_dtype())
    return volume, stored


def _check_admission(admission, subject, pair_name):
    state = _state(admission)
    if not state["locked"]:
        raise HoldoutRefused("truth", "the lock of this admission is not taken: no truth is read before it")
    if subject not in admission.subjects:
        raise HoldoutRefused("truth", f"{subject} is not a subject of this admission")
    row = next((r for r in admission.pairs if r[3] == pair_name), None)
    if row is None or subject not in (row[1], row[2]):
        raise HoldoutRefused("truth", f"{subject} is not a subject of the admitted pair {pair_name}")
    if admission.mode == "holdout":
        lock = state["pair_locks"].get(pair_name)
        if lock is None or not Path(lock).is_file():
            raise HoldoutRefused("truth", f"the pair lock of {pair_name} is not taken")
    return row


def read_truth(subject, *, admission, grid_metadata, pair_name):
    """The aligned_seg35 / aligned_seg4 of one subject, float64 (load_scalar's values), after the admission check; the grid of each volume (shape, spacing,
    origin) must equal the pair manifest's (grid_metadata from bayesreg.io.pair._read_manifest), values finite, non-negative, integer.  Returns
    ({seg35, seg4}, record with each file's path, sha256, stored dtype)."""
    _check_admission(admission, subject, pair_name)
    out, record = {}, {}
    for kind in ("seg35", "seg4"):
        path = Path(DATA_ROOT)/subject/LABEL_FILES[kind]
        volume, stored = _load_label_volume(path)
        grid = volume.grid
        own = dict(shape_zyx=list(grid.shape), spacing_xyz=[float(h) for h in grid.spacing], origin_xyz=[float(a[0]) for a in grid.axes])
        want = {k: [float(v) for v in grid_metadata[k]] if k != "shape_zyx" else list(grid_metadata[k]) for k in ("shape_zyx", "spacing_xyz", "origin_xyz")}
        if own != want:
            raise TruthMismatch("truth_grid", f"{path}: grid {own} differs from the pair manifest's {want}; the evaluation stops")
        values = np.asarray(volume.values, dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0) or not np.array_equal(values, np.rint(values)):
            raise TruthMismatch("truth_grid", f"{path}: labels must be finite non-negative integers; the evaluation stops")
        out[kind] = values
        record[kind] = dict(path=str(path), sha256=volume.metadata["sha256"], stored_dtype=stored, shape_zyx=own["shape_zyx"])
    return out, record


def seg4_map(repo=None):
    """G of C22.Q2-6: configs/bayesreg/seg4_of_seg35_v1.json at its pinned sha256, as a lookup array over 0..35."""
    raw = (Path(REPO if repo is None else repo)/SEG4_MAP_REL).read_bytes()
    if sha_bytes(raw) != SEG4_MAP_SHA256:
        raise HoldoutRefused("seg4_map", f"{SEG4_MAP_REL}: sha256 {sha_bytes(raw)} is not the pinned {SEG4_MAP_SHA256}")
    table = {int(k): int(v) for k, v in json.loads(raw.decode("utf-8"))["seg4_of_seg35"].items()}
    if sorted(table) != list(range(36)):
        raise HoldoutRefused("seg4_map", f"{SEG4_MAP_REL}: labels 0..35 required")
    return np.array([table[k] for k in range(36)], dtype=np.float64)


def seg4_equals_g(seg35, seg4, lookup):
    """Whether the truth seg4 equals G(truth seg35) voxel by voxel (C23.12-4); with the count of differing voxels."""
    labels = seg35.astype(np.int64)
    if labels.min() < 0 or labels.max() >= lookup.size:
        return dict(equal=False, differing=None, note="seg35 labels outside 0..35")
    differing = int(np.count_nonzero(lookup[labels] != seg4))
    return dict(equal=differing == 0, differing=differing)


def truth_of_pair(fixed, moving):
    """The truth dictionary of evaluate_field for the OASIS scheme (keys of truth.load_truth) from two subjects' arrays."""
    truth = dict(fixed_seg35=fixed["seg35"], moving_seg35=moving["seg35"], fixed_seg4=fixed["seg4"], moving_seg4=moving["seg4"])
    truth["brain"] = truth["fixed_seg4"] > 0
    return truth


# ---- stage 3: the shared metric function ----------------------------------------------------------------------------------------------------------------
def _bitwise(a, b):
    a, b = np.ascontiguousarray(a, dtype=np.float64), np.ascontiguousarray(b, dtype=np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


BL1_CERT_CONFIG = MappingProxyType(dict(delta=0., max_depth=48, max_boxes=255, tile_size=8192, policy="lowest_bound_first_v1", prepass="edge_hull_v1",
                                         fallback_workers=2))     # bl1_cert.py:53-54 (eb808be): ParallelEnclosureCertificateConfig(...)
JACOBIAN_CONFIGS = ("C07-2", "BL1")


def bl1_config():
    from jd_curl.three_d.feasibility import ParallelEnclosureCertificateConfig
    return ParallelEnclosureCertificateConfig(**BL1_CERT_CONFIG)


def certify_field_bl1(u, grid, *, brain_nodes=None, split_cells=True):
    """The jacobian_only record of cert_eval.certify_field under BL1's configuration (preregistration §3; the lead, 11-07: "外部臂的 Jacobian 证书用 BL1 配置"):
    the certificate is cert_eval.certify_jacobian_only (the copy of BL1's certify_jd_only) with bl1_config(), the per-cell pass cert_eval.cell_pass and
    cell_summary; the record has certify_field's keys, its config written out field by field (backend.certificate_record knows the C07-2 class only)."""
    import time
    from jd_curl.three_d import feasibility as F
    from jd_curl.three_d import interval as iv
    from jd_curl.three_d.interpolation import check_map_domain
    from jd_curl.validation import as_float64_array
    config = bl1_config()
    u = np.ascontiguousarray(as_float64_array(u, "u", grid.shape+(3,), check_finite=False))
    tick = time.perf_counter()
    result = cert_eval.certify_jacobian_only(u, grid, config)
    seconds = time.perf_counter()-tick
    record = {key: cert_eval._value(getattr(result, key, None)) for key in cert_eval.RESULT_FIELDS}
    lowest = record["min_jd_lower_bound"]
    record.update(mode="jacobian_only", configuration="BL1", n_cells=int(grid.n_cells), seconds=seconds,
                  config=dict(config_class=type(config).__name__, certificate_version=F.certificate_version(config), delta_hex=config.delta.hex(),
                              max_depth=config.max_depth, max_boxes=config.max_boxes, tile_size=config.tile_size, policy=config.policy, prepass=config.prepass,
                              kernel=config.kernel, fallback_workers=config.fallback_workers, precision="CPU FP64, outward rounding",
                              source="archive/seq_vp_3d/results/verification/3d_stage_BL1_v1/analysis/bl1_cert.py:53-54"),
                  passed=bool(result.status == "certified_feasible" and lowest is not None and lowest > 0 and result.n_infeasible == 0 and result.n_uncertain == 0))
    finite = bool(np.all(np.isfinite(u)))
    record["faces_max_abs_u_mm"] = cert_eval._faces_max(u) if finite else None
    if finite:
        domain = check_map_domain(u, grid)
        record["domain"] = dict(ok=domain is None, reason=None if domain is None else domain.reason, count=None if domain is None else int(domain.invalid_count))
    else:
        record["domain"] = dict(ok=False, reason="NONFINITE_STATE", count=None)
    if split_cells and finite and iv.arithmetic_supported():
        tick = time.perf_counter()
        code, lower = cert_eval.cell_pass(u, grid, config)
        record["cells"] = dict(cert_eval.cell_summary(code, lower, result, brain_nodes), seconds=time.perf_counter()-tick)
    else:
        record["cells"] = None
    return record


def score_d(d, grid, truth, *, cert_mode=None, u_run=None, run_certificate=None, tile_size=CERT_TILE_SIZE, split_cells=True, jacobian_config="C07-2"):
    """metrics.evaluate_field and, with a cert_mode, cert_eval.certify_field of u = h d: the calls and record fields of bayesreg.eval.cli.evaluate.
    jacobian_config "BL1" (jacobian_only only): certify_field_bl1 instead of the C07-2 configuration."""
    from .cli import nonfinite_metrics
    if jacobian_config not in JACOBIAN_CONFIGS or (jacobian_config == "BL1" and cert_mode != "jacobian_only"):
        raise ValueError(f"jacobian_config {jacobian_config!r}: one of {JACOBIAN_CONFIGS}; BL1 with cert_mode jacobian_only only")
    scalars, per_label, official = metrics.evaluate_field(d, truth, metrics.OASIS, grid.spacing)
    record = dict(metrics=scalars, per_label=per_label, official=official, nonfinite_metrics=nonfinite_metrics(scalars),
                  units=dict(spacing_xyz=list(grid.spacing), hd95="voxel units (surface_distance spacing ones(3)); equal to mm only when every spacing is 1 mm",
                             hd95_equals_mm=all(float(h) == 1. for h in grid.spacing)), certificate=None)
    if cert_mode is not None:
        u_cert = d_to_u(d, grid.spacing)
        if jacobian_config == "BL1":
            cert = certify_field_bl1(u_cert, grid, brain_nodes=truth["brain"], split_cells=split_cells)
        else:
            cert = cert_eval.certify_field(u_cert, grid, mode=cert_mode, tile_size=tile_size, brain_nodes=truth["brain"], split_cells=split_cells)
        cert["certified_field"] = "u = h d (voxels_to_displacement of the evaluated d)"
        cert["u_equals_run_u_bitwise"] = None if u_run is None else _bitwise(u_cert, u_run)
        if run_certificate is not None:
            cert["run_certificate_state_hash_matches"] = run_certificate.get("state_hash") == cert["state_hash"]
            cert["run_certificate_status_matches"] = run_certificate.get("status") == cert["status"]
        record["certificate"] = cert
    return record


def score_cell(info, grid, truth, *, cert_mode, tile_size=CERT_TILE_SIZE, split_cells=True, jacobian_config="C07-2"):
    """Stage 3 for one described run (fields.describe_run): fields.load_field, then score_d.  The evaluation-entry control calls this same object."""
    d, u_run, field_record = load_field(info, grid.shape, grid.spacing)
    record = score_d(d, grid, truth, cert_mode=cert_mode, u_run=u_run, run_certificate=info.run_certificate, tile_size=tile_size, split_cells=split_cells,
                     jacobian_config=jacobian_config)
    record["field"] = field_record
    return record


# ---- the method arm's decision certificate (D112 (a)) and PQ's acceptance (C22.H7-1) ----------------------------------------------------------------------
def decision_certificate(arm, row, grid, listed, truth, d):
    """D112 (a) (C22.H1-6; the lead's ruling (4) of 11-09; review I3, m15): the decision configuration's certificate of output/final.npz's u -- its stored
    state_hash checked against u (fields.read_final_native) and compared with cert.json's (the Hm1 chain) -- by cert_eval.certify_field(mode='full',
    tile_size=65536) with certify_field's defaults (split_cells; brain_nodes as bayesreg.eval.cli gives them, for the cell summary).  Always on the method
    arm, recorded only: it changes no output and triggers no fallback; the method's certificate statistics are this one."""
    base = record_dir(arm, row)
    final_path = base/fill(arm["decision_certificate"]["final"], arm, row)
    cert_path = base/fill(arm["decision_certificate"]["cert"], arm, row)
    verify_files([final_path], listed)
    u, final_record = read_final_native(final_path, grid.shape)
    verify_files([final_path], listed)
    decision = strict_json(verified_bytes(cert_path, listed), str(cert_path))
    cert = cert_eval.certify_field(u, grid, mode="full", tile_size=DECISION_TILE, brain_nodes=truth["brain"])
    return dict(certificate=cert, certified_field=f"{arm['decision_certificate']['final']}: u (mm)", final=final_record,
                role="recorded only (C22.H1-6, D112 (a)): it changes no output and triggers no fallback; the method's certificate statistics",
                cert_json=dict(path=str(cert_path), state_hash=decision.get("state_hash"), status=decision.get("status")),
                state_hash_matches_cert_json=decision.get("state_hash") == final_record["u_sha256"],
                certificate_state_hash_matches_u=cert.get("state_hash") == final_record["u_sha256"],
                status_matches_cert_json=decision.get("status") == cert["status"],
                u_equals_h_d_bitwise=_bitwise(u, d_to_u(d, grid.spacing)))


def read_pq_outputs(state, listed, shape_zyx):
    """pq.json, pq.npz (q, q4) and baselines.npz (pq_core.BASELINE_KEYS) of one pair, from bytes hashed against the manifest just before (verified_bytes;
    numpy reads those bytes, not the file again).  Every array on the pair's grid."""
    pq_core, _, _ = _pq_modules()
    document = strict_json(verified_bytes(state["files"]["pq.json"], listed), state["files"]["pq.json"])
    arrays = {}
    for name, keys in (("pq.npz", ("q", "q4")), ("baselines.npz", pq_core.BASELINE_KEYS)):
        with np.load(io.BytesIO(verified_bytes(state["files"][name], listed)), allow_pickle=False) as z:
            missing = [k for k in keys if k not in z.files]
            if missing:
                raise HoldoutRefused("pq", f"{state['files'][name]}: members {missing} missing")
            arrays[name] = {k: np.array(z[k]) for k in keys}
    wrong = [f"{name}:{k} {v.shape}" for name, members in arrays.items() for k, v in members.items() if tuple(v.shape) != tuple(shape_zyx)]
    if wrong:
        raise HoldoutRefused("pq", f"PQ arrays not on the pair's grid {tuple(shape_zyx)}: {wrong}")
    return document, arrays["pq.npz"], arrays["baselines.npz"]


def pq_acceptance(pq, base, T_F, T_tilde, F4, T4_tilde, spacing_xyz, *, errors=False):
    """The PQ acceptance numbers of one pair (C22.H7-1), from arrays only (it opens no file): e = 1[T_F != T~_M] (seg35 truth), e4 likewise on seg4
    (described); the masks Ω_e = {T_F > 0} ∪ {T~_M > 0} (judged), the two-sided boundary band and the whole grid (described); every score's AUSE, AUROC and
    largest tie share on each mask (pq_stats.score_record, called as it is), q's reliability on Ω_e (two binnings), q4 against e4 on Ω_e.  The same
    functions and order as experiments/pq/pq_eval.evaluate_pair (pq_eval.score_keys, masked, band_mask; never copied).  errors=True also returns e, e4."""
    _, pq_eval, pq_stats = _pq_modules()
    T_F, T_tilde = np.asarray(T_F), np.asarray(T_tilde)
    e = T_F != T_tilde
    e4 = np.asarray(F4) != np.asarray(T4_tilde)
    masks = dict(omega_e=(T_F > 0) | (T_tilde > 0), band=pq_eval.band_mask(T_F, T_tilde, tuple(float(s) for s in spacing_xyz)),
                 whole=np.ones(T_F.shape, dtype=bool))
    keys = pq_eval.score_keys(pq, base)
    scores = {}
    for mask_name in pq_eval.MASKS:
        mask = masks[mask_name]
        scores[mask_name] = {}
        for name, k in keys.items():
            record = pq_stats.score_record(pq_eval.masked(k, mask), e[mask])
            if mask_name != "omega_e":
                record.pop("curve")
            scores[mask_name][name] = record
    omega = masks["omega_e"]
    out = dict(masks={k: int(m.sum()) for k, m in masks.items()}, n_errors={k: int(e[m].sum()) for k, m in masks.items()},
               n_errors_seg4=int(e4[omega].sum()), scores=scores, seg4_scale=dict(q4=pq_stats.score_record([pq["q4"][omega]], e4[omega])),
               reliability=dict(width=pq_stats.reliability(pq["q"][omega], e[omega], bins="width"),
                                decile=pq_stats.reliability(pq["q"][omega], e[omega], bins="decile")))
    return dict(out, e=e, e4=e4) if errors else out


def pq_pair_acceptance(*, truth_fixed, truth_moving, d, pq, baselines, spacing_xyz):
    """PQH's replay gate interface (alignment item 6; pqh_replay.EV_FUNCTIONS['pair']): one pair's acceptance numbers from arrays only -- the truth dicts
    {'seg35', 'seg4'} (this entry's reader), d, PQ's q, q4 and the baselines.  T~_M = metrics.warp_labels(moving, d) (order 0, outside 0: the function
    evaluate_field warps with).  The hold-out evaluation (evaluate_pq) calls this same function.  Returns pq_acceptance's numbers with e and e4."""
    T_tilde = metrics.warp_labels(np.asarray(truth_moving["seg35"]), d)
    T4_tilde = metrics.warp_labels(np.asarray(truth_moving["seg4"]), d)
    return pq_acceptance(pq, baselines, truth_fixed["seg35"], T_tilde, truth_fixed["seg4"], T4_tilde, spacing_xyz, errors=True)


def pq_b_class(ause_table, *, interval, r_min=None, baselines=None):
    """PQH's replay gate interface (alignment item 6; pqh_replay.EV_FUNCTIONS['b_class']) and F4's class B on the hold-out (holdout_decide.decide_f4):
    experiments/pq/pq_stats.b_class's arithmetic with the resampling passed in (`interval(delta) -> {'lo': ...}`; the development gate passes p1_core's
    block_interval, the hold-out its resampling on 𝒩 restricted to 𝒫).  ause_table: {score: [AUSE per pair]} with 'q' and the 8 judged baselines.  Per
    baseline Δ = AUSE_k - AUSE_q, the interval, LB = its 'lo', R = ΣΔ / ΣAUSE_k (None when ΣAUSE_k = 0: not passed), passed = LB > 0 and R >= r_min;
    not judged when a value is missing.  The replay gate compares Δ, LB, R with the development reading bit for bit."""
    _, _, pq_stats = _pq_modules()
    r_min = pq_stats.R_MIN if r_min is None else r_min
    baselines = pq_stats.JUDGED_BASELINES if baselines is None else tuple(baselines)
    missing = sorted(k for k in ("q",)+baselines if k not in ause_table or any(v is None for v in ause_table[k]))
    if missing:
        return dict(judged=False, reason=f"AUSE missing for {missing}", passed=None, baselines={})
    q = np.asarray(ause_table["q"], dtype=np.float64)
    rows = {}
    for k in baselines:
        base = np.asarray(ause_table[k], dtype=np.float64)
        delta = base-q
        iv = interval(delta)
        total = float(base.sum())
        ratio = float(delta.sum()/total) if total > 0 else None
        lo = iv.get("lo")
        ok = bool(lo is not None and lo > 0 and ratio is not None and ratio >= r_min)
        rows[k] = dict(delta=delta.tolist(), interval=iv, LB=lo, R=ratio, passed=ok, n_positive=int((delta > 0).sum()))
    return dict(judged=True, passed=all(r["passed"] for r in rows.values()), failed=[k for k, r in rows.items() if not r["passed"]], baselines=rows, r_min=r_min,
                rule="LB_k > 0 (interval(Δ_k)['lo'], Δ_k = AUSE_k - AUSE_q) and R_k = ΣΔ_k / ΣAUSE_k >= .05, for every one of the 8 (pq_stats.b_class's arithmetic)")


_PQ_DEV = None


def pq_dev_truth(subject):
    """PQH's replay gate interface (alignment item 6; pqh_replay.EV_FUNCTIONS['truth']; C22.H8 "真值数组由评测入口自己的唯一读取函数在开发集上读出"): the
    truth of one of the 20 development subjects of stage_L2R_val_v1 through read_truth, admitted as the truth-reader gate admits them (admit_truth_gate: the
    development allowlist source, on no sealed list, no B-line training subject, the bench's pair guard; data root DEV_DATA_ROOT).  The replay gate's own lock
    (pqh_replay's REPLAY_LOCK, O_EXCL before its first truth) is the once-guard of that run, so the admission is marked locked here.  A hold-out subject is
    refused (SubjectSealed / HoldoutRefused('dev')).  Returns {'seg35', 'seg4', 'files': {kind: sha256}}."""
    global _PQ_DEV
    if subject not in DEV_SUBJECTS:
        raise HoldoutRefused("dev", f"{subject} is not one of the 20 development subjects of {DEV_STAGE}")
    if _PQ_DEV is None:
        admission, rows = admit_truth_gate(data_root=DEV_DATA_ROOT)
        _STATE[id(admission)]["locked"] = True
        _PQ_DEV = (admission, rows)
    admission, rows = _PQ_DEV
    row = next(r for r in rows if subject in (r["row"][1], r["row"][2]))
    values, record = read_truth(subject, admission=admission, grid_metadata=row["metadata"], pair_name=row["row"][3])
    return dict(seg35=values["seg35"], seg4=values["seg4"], files={k: v["sha256"] for k, v in record.items()})


def evaluate_pq(context, row, metadata, truth, method_output, record):
    """The PQ evaluation record of one pair (C22.H7-1), written into record['pq'] as it goes (review2 m10: a failure leaves an incomplete PQ record, which
    the decision lists as such): its state (pq_states), Ev1 (from (c)), Ev2 (the d of the metrics, the d of the entry and pq.json's sampling.d_sha256 are one:
    float64 C-order, bayesreg.eval.fields.array_sha256), then the acceptance numbers on the truth arrays this entry read (pq_pair_acceptance).  Ev2 failing
    stops this pair's PQ evaluation (no numbers; C22.H8) and F4 reads 不判（评测侧）; F1-F3 are not touched."""
    tick = time.perf_counter()
    release, state = context["pq_release"], context["pq_states"][row[3]]
    out = dict(method_side=release["method_side"], state=state["state"], reason=state["reason"], files=state["files"], ev1=state.get("ev1"), ev2=None,
               acceptance=None, complete=False)
    record["pq"] = out
    if release["method_side"] != "released" or state["state"] != "present":
        out["complete"] = True
        return out
    if method_output is None:
        raise HoldoutRefused("pq", f"{row[3]}: PQ outputs without the method's output")
    d, d_sha256_metrics = method_output
    document, pq, base = read_pq_outputs(state, context["listed"], tuple(metadata["shape_zyx"]))
    d_pq = dotted(document, PQ_D_SHA256_KEY)
    d_entry = array_sha256(d)
    out["ev2"] = dict(passed=bool(isinstance(d_pq, str) and d_pq == d_entry == d_sha256_metrics), d_sha256=d_entry, d_sha256_of_the_metrics=d_sha256_metrics,
                      d_sha256_in_pq_json=d_pq, key=PQ_D_SHA256_KEY, rule="float64 C-order bytes (bayesreg.eval.fields.array_sha256, Hm4)")
    out["input_lock"] = pq_input_lock_check(document, row, release.get("input_lock") or dict(problem="no input lock read", predictions={}, fields={}))
    if not out["ev2"]["passed"]:
        out.update(complete=True, note="Ev2: the d of q is not the d of the metrics: this pair's PQ evaluation stops (C22.H8); F4 不判（评测侧）",
                   seconds=time.perf_counter()-tick)
        return out
    if not out["input_lock"]["passed"]:
        out.update(complete=True, note="the inputs pq.json records are not the step-8 input lock's (C22.H4-2 Hm2): this pair's PQ evaluation stops; "
                                       "F4 不判（评测侧）", seconds=time.perf_counter()-tick)
        return out
    numbers = pq_pair_acceptance(truth_fixed=dict(seg35=truth["fixed_seg35"], seg4=truth["fixed_seg4"]),
                                 truth_moving=dict(seg35=truth["moving_seg35"], seg4=truth["moving_seg4"]), d=d, pq=pq, baselines=base,
                                 spacing_xyz=metadata["spacing_xyz"])
    out.update(acceptance={k: v for k, v in numbers.items() if k not in ("e", "e4")}, complete=True, seconds=time.perf_counter()-tick)
    return out


# ---- the evaluation of the admitted hold-out --------------------------------------------------------------------------------------------------------------
def evaluate_pair(admission, context, row, lookup, record=None):
    """Stages 2 and 3 for one pair (its lock taken here), filling `record` (the record of <out>/pairs/<pair>.json) as it goes, so that a failure keeps the
    arms already evaluated (a partial record; a reevaluation never overwrites it, revision 1b item 2).  Every arm: score_cell with the arm's certificate
    (BL1 Jacobian-only); the method arm also its decision certificate; then the pair's PQ evaluation."""
    table, cells, listed = context["table"], context["cells"], context["listed"]
    index, fixed, moving, name = row
    record = dict(schema=SCHEMA_PAIR, pair=name, index=index, fixed=fixed, moving=moving, complete=False, arms={}) if record is None else record
    lock = take_pair_lock(admission, row, context["line"])
    record.update(lock=lock, started_utc=utc())
    stage = Path(table["stage_root"])/name
    verify_files([stage/"manifest.json", stage/"pair.npz"], listed)
    grid, metadata, selected = _read_manifest(stage/"manifest.json")
    if (selected["fixed_subject"], selected["moving_subject"]) != (fixed, moving):
        raise HoldoutRefused("pairs", f"{stage}/manifest.json: {selected['fixed_subject']} <- {selected['moving_subject']} is not the row {fixed} <- {moving}")
    if file_sha256(stage/"pair.npz") != selected["pair_sha256"]:
        raise ManifestChanged("manifest_changed", f"{stage}/pair.npz: sha256 differs from manifest.pair_sha256")
    arrays, truth_record = {}, {}
    spy = context.get("spy")
    if spy is not None:                                             # the spy lets this pair's truth files through, for the reads below only
        spy.truth = frozenset(os.path.realpath(Path(DATA_ROOT)/s/LABEL_FILES[k]) for s in (fixed, moving) for k in LABEL_FILES)
    try:
        for side, subject in (("fixed", fixed), ("moving", moving)):
            arrays[side], truth_record[side] = read_truth(subject, admission=admission, grid_metadata=metadata, pair_name=name)
    finally:
        if spy is not None:
            spy.truth = frozenset()
    seg4_check = {side: seg4_equals_g(arrays[side]["seg35"], arrays[side]["seg4"], lookup) for side in ("fixed", "moving")}
    truth = truth_of_pair(arrays["fixed"], arrays["moving"])
    record.update(grid=metadata, truth=truth_record, seg4_equals_G=seg4_check)
    record["identity"] = score_d(np.zeros((3,)+tuple(grid.shape)), grid, truth)
    method_output = None
    for arm in table["arms"]:
        cell = cells[(arm["name"], name)]
        entry = dict(outcome=cell.outcome, no_output_class=cell.no_output_class, cause=cell.cause, terminal_status=cell.status, terminal=cell.terminal,
                     export_suspicious=cell.export_suspicious, authorization=cell.authorization, rejudgment=cell.rejudgment, move=cell.move,
                     stop_record=None if cell.stop_record is None else {k: v for k, v in cell.stop_record.items() if k != "document"}, review=cell.review)
        if cell.outcome == "output":
            verify_files(cell.needed, listed)
            info = describe_run(Path(cell.field_dir), pair=name, kind=arm["kind"])
            scored = score_cell(info, grid, truth, cert_mode=arm["cert_mode"], jacobian_config=arm["jacobian_config"])
            verify_files(cell.needed, listed)
            if scored["field"].get("disp", {}).get("file_sha256") not in (None, listed.get(str(Path(cell.field_dir)/"disp.npz"))):
                raise ManifestChanged("manifest_changed", f"{cell.field_dir}/disp.npz changed while it was read")
            entry.update(scored, field_role=info.field_role, run_status=info.run_status)
            if arm["family"] == "method":
                d, _, _ = load_field(info, grid.shape, grid.spacing)
                verify_files(cell.needed, listed)
                entry["decision_certificate"] = decision_certificate(arm, row, grid, listed, truth, d)
                method_output = (d, (scored["field"].get("disp") or {}).get("d_sha256", scored["field"].get("d_sha256")))   # review2 m6: the metrics' d sha256
        record["arms"][arm["name"]] = entry
    evaluate_pq(context, row, metadata, truth, method_output, record)           # A5b review I1: PQ first, the ablation after it (described only, never in PQ's way)
    evaluate_ablation(context, row, grid, truth, record)
    record.update(complete=True, ended_utc=utc())
    return record


def evaluate_ablation(context, row, grid, truth, record):
    """The repair ablation of one pair (prereg revision 2 item 4; C24.2-1, -2; A5b item 3 (b)), in this pair's lock, on the truth arrays already read: the
    pair's prerepair_disp.npz read as a plain displacement (fields.read_disp, hashed against the manifest before and after), score_d without cert_mode (no
    certificate, never "certified"), its numbers beside the method arm's output (the repaired field): before, after, after - before for ABLATION_KEYS.  Kept in
    its own section of the pair's record (record['ablation']), in no arm's numbers; a pair not in the ablation keeps the reason decided before any truth.
    A5b review I1 (the lead's ruling, 11-10 ~9:22 PM CST): run after the pair's PQ evaluation; an error here other than the three that stop everything
    (ManifestChanged, TruthMismatch, SpyRefused) is written into record['ablation'] (complete false, error) and the pair goes on -- the ablation is described
    only and never makes the pair failed (C24.2-1, -2)."""
    ablation = context.get("ablation") or dict(pairs={}, excluded={})
    name = row[3]
    if name not in ablation["pairs"]:
        record["ablation"] = dict(in_ablation=False, reason=ablation["excluded"].get(name, "not decided"), role="described only (prereg revision 2 item 4)")
        return record["ablation"]
    entry = ablation["pairs"][name]
    out = dict(in_ablation=True, line_status=entry["line_status"], record=dict(path=entry["record"], sha256=entry["record_sha256"]), inputs=entry["inputs"], complete=False,
               role="described only (prereg revision 2 item 4, C24.2): not certified, not an arm, in no family, no Holm, no decision changed")
    record["ablation"] = out
    listed = context["listed"]
    try:
        verify_files([entry["disp"], entry["record"]], listed)
        d, field = read_disp(entry["disp"], grid.shape)
        verify_files([entry["disp"]], listed)
        scored = score_d(d, grid, truth)                              # no cert_mode: no certificate (C24.2-1)
    except (ManifestChanged, TruthMismatch, SpyRefused):
        raise
    except Exception as exc:                                          # noqa: BLE001 - A5b review I1: described only, the pair is not failed by it
        out.update(error=f"{type(exc).__name__}: {exc}", note="the ablation of this pair did not complete (A5b review I1): recorded here only; the pair, its arms "
                                                              "and its PQ evaluation are not touched")
        return out
    after = ((record["arms"].get(method_arm(context["table"])["name"]) or {}).get("metrics") or {})
    before = scored["metrics"]
    out.update(prerepair=dict(scored, field=field), before={k: before.get(k) for k in ABLATION_KEYS}, after={k: after.get(k) for k in ABLATION_KEYS},
               difference={k: (None if not isinstance(after.get(k), (int, float)) or not isinstance(before.get(k), (int, float)) else float(after[k])-float(before[k]))
                           for k in ABLATION_KEYS}, difference_written="after - before (after: the method arm's output)", complete=True)
    return out


def evaluate_holdout(admission, context):
    """Every admitted pair in list order (a reevaluation: its authorized pairs).  An evaluation error of a pair is kept in its record and the next pair
    follows (exit 1; a reevaluation needs an authorization, D109); a changed listed file, a truth volume off the pair's grid (review m3) or an open the spy
    refuses stops everything (exit 4).  The spy is stopped at the end and its record written into holdout_eval.json."""
    out, lookup, spy = context["out"], seg4_map(), context.get("spy")
    reeval = admission.record.get("reeval")
    rows = [r for r in admission.pairs if reeval is None or r[3] in reeval["pairs"]]
    done, failed, stopped = [], [], None
    try:
        for row in rows:
            path = out/"pairs"/f"{row[3]}.json"
            record = dict(schema=SCHEMA_PAIR, pair=row[3], index=row[0], fixed=row[1], moving=row[2], complete=False, arms={})
            try:
                evaluate_pair(admission, context, row, lookup, record)
            except (ManifestChanged, TruthMismatch, SpyRefused) as exc:
                stopped = f"{type(exc).__name__}: {exc}"
            except HoldoutRefused as exc:
                if exc.stage == "once":
                    raise
                record.update(complete=False, error=f"{type(exc).__name__}: {exc}", utc=utc())
                failed.append(row[3])
            except Exception as exc:                                   # evidence kept; a reevaluation needs an authorization (D109)
                import traceback
                record.update(complete=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(), utc=utc())
                failed.append(row[3])
            if stopped is None and spy is not None and spy.refused:        # a refusal some code swallowed still stops the evaluation
                stopped = f"SpyRefused: {spy.refused}"
            if stopped is not None:
                write_once(path, dict(record, complete=False, error=stopped, utc=utc()))       # the partial record is kept
                write_once(out/"stopped.json", dict(pair=row[3], reason=stopped, utc=utc()))
                break
            write_once(path, record)
            if record.get("complete"):
                done.append(row[3])
    finally:
        if spy is not None and spy is _SPY:
            stop_spy()
    summary = dict(schema=SCHEMA_RUN, record=admission.record, evaluated=done, failed=failed, stopped=stopped, not_reached=[r[3] for r in rows if r[3] not in done+failed],
                   arm_table=context["table"], opened_files=None if spy is None else spy.record(), ended_utc=utc())
    write_once(out/"holdout_eval.json", summary)
    return 4 if stopped else (1 if failed else 0)


# ---- the command line -----------------------------------------------------------------------------------------------------------------------------------
def dry_run(out_dir, pair=None):
    """Each condition on its own, reading only code-tree files (no subject file): the first refusal is what `evaluate` would stop at.  Writes <out>/dry_run.json."""
    out = Path(out_dir)
    if out.exists():
        raise SystemExit(f"{out} exists")
    report = dict(schema="bayesreg.holdout.dry_run.v1", utc=utc(), pair=pair, conditions={})
    steps = [("rule one (pinned_code.json; C24.3-4)", lambda: rule_one()),          # expected refused before the unsealing (no runs/holdout_B_v1/pinned_code.json): recorded, the rest go on
             ("(0) truth-reader gate", lambda: check_decision()), ("(0) PQ replay gate", lambda: check_pq_replay()),
             ("(a) unseal object and pairs", lambda: admit_holdout()), ("(b) preregistration", lambda: check_prereg()),
             ("(c) manifest committed", lambda: read_committed(MANIFEST_REL, "manifest"))]
    if pair is not None:
        rows, _ = load_pairs()
        fixed, _, moving = pair.partition("__")
        steps.insert(3, ("(a) pair on the list", lambda: check_pair_row(fixed, moving, rows)))
    first = None
    for label, step in steps:
        try:
            step()
            report["conditions"][label] = "passed"
        except (HoldoutRefused, SubjectSealed, SealedListError, OSError, ValueError) as exc:
            report["conditions"][label] = f"refused: {type(exc).__name__}: {exc}"
            first = first or label
    report["first_refusal"] = first
    report["truth_read"] = False
    out.mkdir(parents=True)
    write_once(out/"dry_run.json", report)
    for label, text in report["conditions"].items():
        print(f"{label}: {text}")
    print(f"first refusal: {first}; no subject file was opened")
    return 0


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog="python -m bayesreg.eval.holdout", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--out-dir", required=True, type=Path, help=f"a new directory under {EVAL_DIR} (C24.8 item 15)")
    e.add_argument("--reeval", type=Path, help="a committed reevaluation authorization (D109)")
    d = sub.add_parser("dry-run")
    d.add_argument("--out-dir", required=True, type=Path)
    d.add_argument("--pair", help="<fixed>__<moving>: also check that this pair is a row of the list")
    args = parser.parse_args(argv)
    if args.command == "dry-run":
        return dry_run(args.out_dir, args.pair)
    try:
        admission, context = open_holdout(args.out_dir, reeval_path=args.reeval)
    except SubjectSealed as exc:
        print(f"subject_sealed: {exc}; nothing was read", file=sys.stderr)
        return 3
    except HoldoutRefused as exc:
        print(f"refused at {exc.stage}: {exc}; no truth was read", file=sys.stderr)
        return 2
    except (SealedListError, OSError, ValueError) as exc:        # fail closed, before any truth: the same exit code as a refusal (review m16)
        print(f"refused: {type(exc).__name__}: {exc}; no truth was read", file=sys.stderr)
        return 2
    return evaluate_holdout(admission, context)


__all__ = ["Admission", "HoldoutRefused", "ManifestChanged", "OpenedFiles", "SpyRefused", "TruthMismatch", "admit_holdout", "admit_truth_gate",
           "certify_field_bl1", "check_cells", "check_decision", "check_manifest", "check_pair_row", "check_pq_release", "check_pq_replay", "classify_cell",
           "code_identity_problem", "decision_certificate", "evaluate_holdout", "evaluate_pair", "evaluate_pq", "load_arm_table", "load_pairs", "load_reeval",
           "machine_gate", "open_holdout", "parse_manifest", "pq_acceptance", "pq_b_class", "pq_dev_truth", "pq_pair_acceptance", "pq_states",
           "read_pq_outputs", "read_truth", "replay_decision_problem", "run_lines", "run_outs", "score_cell", "score_d", "seg4_equals_g", "seg4_map",
           "start_spy", "stop_spy", "take_gate_lock", "truth_of_pair", "validate_arm_table", "watched_roots"]


if __name__ == "__main__":
    sys.exit(main())
