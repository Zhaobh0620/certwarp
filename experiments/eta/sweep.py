"""C05-7 eta initial sweep under P2 (work packages ETA, ETA2; runs/20261030-eta-sweep/plan.md; contract C05-7 and appendix
7a, pending review). Development set only (stage_T0_tune_v1); the chosen eta is an EB initial value, never the final eta.

    python -B experiments/eta/sweep.py [--root DIR] [--code DIR] [--commit SHA]
        [--until precheck|atropos|probe|s1|s2|s3]
        [--channel-source structure|tissue|none] [--logvol-points gauss27+closed27|gauss27+corners|gauss27]
        [--logvol-closed27-share BETA] [--metric-jd-scaling affine_v1|off]
        [--jd-cell-estimate certificate|gauss27+corners|closed27] [--structure-root DIR] [--structure-grouping NAME]
        [--segmenter-id ID] [--segmenter-registry PATH (dry run only)] [--backend torch64|numpy] [--device cuda|cpu]
        [--outer a] [--rounds 1,1,1,1] [--wall-seconds 900] [--extra "ARGS"] [--data-root DIR] [--allowlist PATH]
        [--tissue-root DIR] [--memory-max 9G] [--eval-memory-max 6G] [--probe-memory-max 9G] [--atropos-memory-max 4G]
        [--atropos-threads 8] [--min-gpu-free-mib 22528] [--home-floor-gb 100] [--wait-minutes 0] [--dry-run]
        [--sigma-domain brain|support|support_shared] [--support-threshold THETA]

Pairs (fixed rule, written before any run): stage_T0_tune_v1/selection.json holds 12 pairs, 4 per ventricle-volume stratum
(its own pre-registered rule; no registration outcome was looked at). Per stratum in the order "moving larger", "moving
smaller", "balanced", the first pair by directory name: 0012-0013, 0016-0017, 0026-0027. No other stage is accepted
(the L2R validation pairs must not become a tuning set); a T0 problem stops the sweep.
Configuration (default since ETA2: the method's final configuration, as the H arms of the M2 GPU smoke v2 apart from eta):
--channel-source structure (the registered segmenter unet35_v1_ens, predictions under --structure-root, default
/home/rmcfz/seq_vp_tag_results/seg/unet35_v1/pred, read with --structure-infer never; the registry is the code tree's
configs/bayesreg/segmenters.json, "registered"; tau^2, the brain mask and the normalization from the tissue layer, C03.1-3),
--jd-cell-estimate certificate (D47; run_pair's default since DEF), --logvol-points gauss27+closed27 with
--logvol-closed27-share BETA (C05-2b, (A') adopted by BZ 10-30; default 1 = D48, run_pair's default since DEF; the beta
of the method was decided after smoke v2) and the Simpson rule
(D48; the uniform rule is an ablation that Laplace / EB refuse, so not offered), --backend torch64 --device cuda with the
OPT1 switches at run_pair's defaults, --metric-jd-scaling affine_v1, --outer a, --rounds 1,1,1,1, --wall-seconds 900
(the development switch of C13-3 / D44: every run ends by "预算收尾" budget_exhausted or another C13-3 ending, never
"B 线收敛" unless its status says so), --cpu-threads 4. The first plan's G2 configuration is --channel-source none
--logvol-points gauss27+corners --jd-cell-estimate gauss27+corners. One configuration per --root: the probes' records,
nu_star.json and the selections carry it, and a restart under another configuration STOPs (the probe comparison, the
recomputed records). --extra must not repeat an option the sweep sets itself (prefixes included).
The probe runs under the same configuration (its data-term gradient includes the structure-layer channels); its record's
`configuration` must equal the sweep's, else STOP. Every launch record, every selection, nu_star.json, summary.json and
eta_selected.json carry the configuration; every run's config.json args are checked against it (pair_row; mismatch: STOP).
sigma_domain (C04-6, C05-7a appended note; work package SWLI): --sigma-domain brain (default) | support | support_shared
and --support-threshold THETA (default .05 with support / support_shared; run_pair's checks) are passed to the probe and
to every run_pair (model_args) and belong to the configuration: under support / support_shared the probe configuration
(and so the root's configuration) gains sigma_domain and support_threshold, the runs' config.json args are checked for
them, and a root started under another sigma_domain or theta STOPs (the probe comparison and the recomputed records,
as for every other key).  Under brain nothing is passed and every record is the one before SWLI, key for key (roots
started before SWLI resume unchanged).  A brain nu* or eta is never carried to support_shared: a new root.

Order (each step in its own systemd unit, in series, `systemd-run --user --wait --collect`):
  precheck (no unit; channel_source structure only): the registry -- real runs: the code tree's
           configs/bayesreg/segmenters.json, tracked by git and clean (no staged, unstaged or untracked change;
           bayesreg.seg.structure_layer.registry_git_state, the flags run_pair records in status.json, required as
           experiments/d0/make_inputs.py requires them), on top of the code tree's own cleanliness (below); a dry run:
           --segmenter-registry (synthetic staging registry, passed with --structure-registry-staging) -- loads and
           registers at least two folds of --segmenter-id (all_folds_mean, D24); then the prediction of each of the six
           subjects: --structure-root/<subject>/{provenance.json, structure_groups.npz} with segmenter_id, subject,
           grouping and policy all_folds_mean (the T0 subjects are in no fold model's label contact, D24) and model ids
           registered (a dry run: the provenance names experiments/eta/dry_inputs.py). Missing predictions: their ids
           go to a new file $ROOT/structure_infer_subjects_<UTC>.txt and the sweep STOPs with the exact command
             cd <repository> && <python> -B -m bayesreg.seg.unet.cli infer --subjects-file <that file>
           (+ --pred-out DIR when --structure-root is not the inference's default; the inference has its own machine
           gate, needs a clean repository and the GPU, and never overwrites). An incomplete prediction directory or a
           prediction that does not fit STOPs (kept; the lead decides). The full validation (D24 preflight,
           provenance, hashes, read-back) is run_pair's, first met by the probe.
  atropos  the tissue layer of the six subjects, only where --tissue-root/<subject>/provenance.json is missing:
           experiments/d0/atropos_seg.py --stage stage_T0_tune_v1 --subjects <one> (the D0 commands and checks), unit
           <prefix>-atropos-<subject>, Nice 19, one subject per unit; afterwards provenance.json with
           checks.qc_min_class_fraction_ge_0.05 true is required for every subject (else STOP).
  probe    experiments/eta/probe.py per pair (CPU FP64, unit <prefix>-probe-<pair>): nu*_p at u_1 (C05-7). The centre
           nu* = geometric mean of the three nu*_p (probe/nu_star.json, written once); tau^(0) from the probes (all equal).
  s1       nu in {nu*/10, nu*, 10 nu*}, s_tau = 1                    labels x0p1, x1, x10
  s2       nu in {nu_1/4, nu_1, 4 nu_1}, s_tau = 1 (nu_1: s1 choice)  labels x0p25, x1, x4  (x1 = the s1 choice, reused)
  s3       nu = nu_2 (s2 choice), s_tau in {1/2, 1, 2}                labels st0p5, st1, st2 (st1 = the s2 choice, reused)
  eta of a candidate (eta/<stage>_<label>.json, run_pair --eta-json): tau_0 = tau_0^(0) (fixed, C05-6), tau_k = s_tau
  tau_k^(0) (k >= 1), alpha = alpha_nug(tau) = 1e-2 min_k 1/tau_k^2 (P2: not swept). A reused candidate is not rerun (same
  configuration; the wall budget makes a rerun differ, and constraint 4 forbids reruns); its rows are listed again.
  Every run: unit <prefix>-<stage>-<label>-<pair> -> runs/<stage>/<label>/<pair>/ (bayesreg.cli.run_pair), then its
  evaluation by bayesreg.eval.cli only (unit <prefix>-eval-<stage>-<label>-<pair> -> eval/<stage>/<label>/<pair>/, run id
  eta_<stage>_<label>_<pair>, method bayesreg_eta_sweep, arm <stage>-<label>; done.json at the end).
Selection after each stage (selection_<stage>.json / .md, written once; on a restart recomputed and compared, a mismatch
STOPs): a candidate is eligible when all three pairs have an evaluation without error whose field_role is comparable
(bayesreg.eval.fields.COMPARABLE_ROLES: native_formal, disp_only), whose full certificate passed and whose seg35 is finite.
Score = the mean seg35 over the three pairs. Ties: every eligible candidate within TIE_TOL = 1e-3 of the best score is
tied; among them the one closest to the stage centre (smallest |log factor|), then the higher mean seg4, then the smaller
factor. No eligible candidate -> STOP (the lead decides). seg4 is reported next to seg35 everywhere.
End: summary.json / summary.md (one table per stage: seg35, seg4, HD95, official and brain SDlogJ, certificate, status,
field_role; the chosen eta and the selection path) and eta_selected.json, each written once.

Resume (hard constraint 4: nothing is rerun, no evidence overwritten), per run as in experiments/m2/gpu_smoke.sh:
eval done.json present -> skip; unit still active -> STOP; run directory with status.json -> evaluate only; run directory
without status.json -> STOP; an earlier launch record without status.json -> STOP; an earlier evaluation attempt without
done.json -> STOP; a probe directory without probe.json -> STOP; an Atropos directory without provenance.json -> STOP.
Gate before every unit (copied from gpu_smoke.sh; polled every 60 s for at most --wait-minutes, then STOP): hold file
$ROOT/hold absent; tag-d0-queue not active; no other seq-vp-* / tag-* unit active, activating or deactivating
(tag-unet35-train and the unit running this script excepted); GPU free >= --min-gpu-free-mib (runs on cuda only);
MemAvailable >= MemoryMax of the unit + 2 GiB; /home free >= --home-floor-gb. The gate's computation and the training
unit's memory go to sweep.log and to the launch record launch/<kind>_<name>_<UTC>.txt (unit, code, systemd command, state
before and after, exit code, OOM count).
Code version: --commit SHA (default the HEAD at start) resolved once; without --code the units run in a detached git worktree
$ROOT/code of that commit (code_commit.txt; verified on every restart, as gpu_smoke.sh). Log: $ROOT/sweep.log (UTC, appended).
--dry-run: synthetic inputs of experiments/eta/dry_inputs.py only (--data-root, --tissue-root, --allowlist required, and
for channel_source structure --structure-root and --segmenter-registry; none under data/ or the real segmentation root,
manifests and structure provenance naming dry_inputs.py, allowlist source synthetic_dry_run); unit prefix tag-etadry,
device cpu, one thread, Nice 19, --root outside the real roots; every gate is enforced (GPU excepted). --segmenter-registry
outside a dry run is refused (real runs use the code tree's registered registry).
Default root /home/rmcfz/seq_vp_tag_results/eta_sweep_v2 (ETA2; eta_sweep_v1 holds only the Atropos step of 10-30 for the
first plan's configuration, which never ran).
Exit codes: 0 done (or --until reached), 2 usage, 8 STOP.

Design v4 (work package ETA4, runs/20261101-eta-sweep-v4/plan.md; contract C05-7 item 7b, D56; --design v4; the
default --design v3 is everything above, unchanged: the same commands, configuration and records for the same command
line, experiments/eta/check_v3_records.py):
  scope    sigma_domain brain only (structure_repr sqrt_prob: the key absent from run_pair while SDF is a draft); unit
           prefix tag-eta4 (tag-eta4dry), told apart from v3's tag-eta in the journal.
  centre   no probe: the centre is the eta the v3 sweep chose (--centre-eta, default eta_sweep_v3/eta_selected.json; a
           real run pins its sha256 to V3_ETA_SHA256): nu_centre = its nu, tau^(0) = its tau (s_tau = 1), alpha =
           alpha_nug(tau) checked. Checks: its summary.json beside it agrees (eta_selected, stage_set, the same three
           pairs); its configuration equals this root's apart from the path keys (jd_cell_estimate,
           max_accepted_per_level, design, path; structure root resolved, registry by sha256 / folds / status); the
           model code (bayesreg/, jd_curl/, configs/bayesreg/) of its code_sha against this code tree: changed files
           recorded, --accept-code-change required (real runs); the first run's argument list through run_pair's own
           parse_args and args_record in the code tree (C13-4a: a conflict is refused here, not in a unit), and that
           config.json args record against the v3 run of the same pair (<centre dir>/runs/*/*/<pair>/config.json,
           args_differences): only the per-run keys, the path's keys (equal to the path's values), paths inside each
           code tree (equal relative to it) and SWLI's sigma_domain brain / support_threshold null may differ, else
           STOP. centre/centre.json (written once; a restart recomputes and compares) records all of it, the grid and
           the selection rule (the SDlogJ condition and its note). --until centre stops there.
  s1       nu in {nu_c/16, nu_c/4, nu_c, 4 nu_c}, s_tau = 1          labels x0p0625, x0p25, x1, x4
  s2       nu in {nu_1/2, nu_1, 2 nu_1}, s_tau = 1                   labels x0p5, x1, x2 (x1 = the s1 choice, reused)
  s3       nu = nu_2, s_tau in {1/2, 1, 2}                           labels st0p5, st1, st2 (st1 reused), as v3
  24 new runs. The selection rule is v3's plus the SDlogJ condition (below); the edge of each choice (lower / upper /
  interior of its stage), whether the chosen nu is the smallest ("仍在下端") or the largest ("仍在上端") nu run, the
  lowest and highest eligible nu and the candidates the SDlogJ condition excluded below and above the chosen nu are
  recorded, never acted on (no automatic extension).
  --sdlogj-official-max (v4; default SDLOGJ_V4 = .10, contract 7b / D56, mandatory): a candidate is eligible only if
  also the three official SDlogJ are finite and their mean <= .10; another value or `none` only with --sdlogj-note
  TEXT, recorded in centre.json.
  path (the solver path; only path switches, F_B unchanged): one of PATHS -- current (certificate, no per-level caps),
  p1 (--max-accepted-per-level 500,500,100,500), p2 (--jd-cell-estimate certificate_bernstein), p3 (both) -- taken
  either mechanically from the path pilot's summary (--path-from-pilot SUMMARY_JSON, pilot_path(): the pilot's
  preregistered "η 重扫路径的建议") or explicitly (--path NAME --path-note TEXT, the lead's decision; with
  --path-from-pilot as well the pilot record is kept beside the override). A v4 root needs one of them; then
  --jd-cell-estimate and --max-accepted-per-level are not given (one source). pilot_path gives "resolved" (p1 / p2 /
  p3), "lead" (the pilot's rule ends in its branch 4: the current path, the lead decides) or "hold": the pilot not
  ended (a tag-pilot* unit or run_pilot.py process, or its queue.log not ending with "end" after a summary line naming
  this file); its experiments/pilot/summarize.py not loadable, or differing from the copy in the pilot's code
  worktree; the summary missing, unreadable or not the newest of its directory; another schema, root or baseline; arm
  switches or eta not the expected ones; the P1-P3 verdicts not reproduced by the pilot's arm_flags on the summary's
  own rows and comparisons, or the recommendation not reproduced by its recommend; a certificate_bernstein check-(b)
  violation anywhere in the pilot; a verdict 待审核 / 待核对 among P1-P3. "lead" and "hold" write
  path_hold_<UTC>.json and STOP before any unit (no automatic start; the lead gives --path explicitly). path.json
  (written once, compared on a restart) records the path, its switches, the pilot record and the summarize.py sha256;
  the configuration gains design, path and (with caps) max_accepted_per_level.
  After each run (pair_row): config.json args checked for the path (max_accepted null / caps / level_time_shares null,
  or max_accepted 500 -- the --extra value in a dry run -- and no SCH key), sigma_domain brain, support_threshold
  null, structure_repr absent or sqrt_prob, and config.json eta equal to the candidate's eta file.
  Hard constraint 1 (contract 7b): an evaluated field (metrics.json field_present) whose evaluator certificate did not
  pass (zero or undecided) STOPs at once -- after the evaluation, on a restart that meets done.json, and for every
  selection row (experiments/m2v1/run_batch.py hard_constraint_1).
  certificate_bernstein (C09.1-3a check (b), D53, contract 7b "审核停止"): after each run's evaluation, on a restart that
  meets done.json and for every selection row the run's check-(b) count -- the larger of the run total
  bernstein_j53_violations_total (anywhere in status.json, hyper.json or timing.json) and the sum of the log.jsonl
  jd_scaling rows' n_bernstein_j53_violations, whichever are recorded, as the pilot's summarize.py -- must be 0; > 0
  (待审核) or not recorded (待核对) STOPs, evidence kept, until audit/<stage>-<label>-<pair>.json passes
  audit_problems(): state_hash = the run's status.json "exported" (and the evaluator certificate's state_hash),
  violations = the current count (an integer, never a bool; null when not recorded), certificate_config
  CompiledFallbackCertificateConfig (configuration C without the prescreen), recertified_passed true,
  min_jd_lower_bound finite > 0, auditor, utc with a time zone, later than the run's end (run_end_utc: the
  evaluation's done.json; an end that cannot be determined never releases, R3-1) and not later than the moment the
  file is read plus 10 minutes (R3-3), and recertification_record: the output JSON of experiments/audit/recertify.py
  (R3-2: tool, schema and script_sha256 of the repository's script, a 40-digit code_sha, code_dirty false unless
  the audit's code_dirty_note; RECERT review I1: run_dir = the run, export_file.file_sha256 = the run's exported file
  (final_native.npz, or final_level{L}.npz for a run stopped before the native level), grid.rebuilt_grid_id = the
  evaluator certificate's grid_id; S2, S3: status certified_feasible, certificate_state_hash = state_hash,
  certificate_version of this jd_curl, n_prescreen_cells 0, tile_size 8192; state_hash = the exported state;
  certificate_config {"class": "CompiledFallbackCertificateConfig", "delta": 0.0, "max_depth": 48, "max_boxes": 255,
  ...} -- a valid instance of that class; passed true; no uncertain and no infeasible cell; min_jd_lower_bound > 0,
  equal to the audit's; started after the run's end, ended not after the audit's utc) (ETA4 review rounds 2 and 3).
  A run without an exported state is never released by an audit; only by a
  no-export confirmation (no_export_problems: run = <stage>-<label>-<pair>, no_export true, state_hash null, auditor,
  utc as above; R2-5), after which the run counts as ineligible ("不合格（无导出）") and the sweep goes on. The file's,
  and the recertification record's, path and sha256 go into the selection row and the readouts.
  The model code changed since the centre's code_sha without --accept-code-change: the list goes to
  centre/model_code_changed_pending_<UTC>.json and the sweep STOPs before centre.json (R2-7). path.json and the
  "path:" log line carry the pilot summary's counts ran / not_run / failed_no_status (R2-6).
  readouts_<stage>.json / .md (written once, descriptive, not part of the selection): per run seg35, seg4, HD95, the two
  SDlogJ, brain log J quantiles .01 / .99, % J < .5, |log J| > log 2, the evaluator certificate (status, bound,
  prescreen and fallback cells, seconds), the check-(b) count and audit, and the per-level readouts of
  experiments/m2v1/level_table.py (ending, accepted steps, cap, GF_B/T_B, seconds, blocked levels).
  Gate additions (v4): no other driver process (python running experiments/eta/sweep.py, experiments/m2v1/run_batch.py
  or experiments/pilot/run_pilot.py; this process and its ancestors excepted). A real v4 call that starts runs
  (--until s1 / s2 / s3) needs --no-start-after. Default root eta_sweep_v4; a real v4 root never lies under
  eta_sweep_v1..v3.
--max-accepted-per-level a0,a1,a2,a3 (both designs; C13-4a, D51): run_pair's switch, passed to every run (never to the
probe, whose cap is 1), recorded in the configuration; --extra must then not name --max-accepted (run_pair refuses
both). --extra never names --max-accepted-per-level or --level-time-shares (prefixes included).
--no-start-after ISO (both designs; a date-time with an explicit UTC offset, e.g. 2026-11-02T06:00-06:00): no run unit
starts at or after it (checked before the gate and while it waits) -> STOP (exit 8); evaluations of ended runs go on;
a restart with a later deadline resumes.
"""
import argparse
from datetime import datetime, timezone
import glob
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bayesreg.eval.fields import COMPARABLE_ROLES  # noqa: E402
from bayesreg.io import ProvenanceRejected, load_segmenter_registry  # noqa: E402
from jd_curl.validation import InputValidationError  # noqa: E402
from bayesreg.model import CLOSED27_SIMPSON, LOGVOL_GAUSS27_CLOSED27, LOGVOL_POINTS  # noqa: E402
from bayesreg.model.support import BRAIN, DEFAULT_SUPPORT_THRESHOLD, SIGMA_DOMAINS, check_theta  # noqa: E402
from bayesreg.seg.unet.groups import DEFAULT_GROUPS_PATH, load_grouping  # noqa: E402
from bayesreg.seg.structure_layer import (DEFAULT_GROUPING, GROUPS_FILE, PROVENANCE_FILE, SEGMENTER_ID,  # noqa: E402
                                          registry_git_state)
from jd_curl.three_d import feasibility as jd_feasibility  # noqa: E402
from bayesreg.solve.jd_scaling import (CELL_ESTIMATES as JD_CELL_ESTIMATES,  # noqa: E402
                                       CERTIFICATE_BERNSTEIN, DEFAULT_CELL_ESTIMATE as JD_DEFAULT,
                                       MODES as JD_SCALING_MODES, NONE as JD_NONE, OFF as JD_OFF)

PY = "/home/rmcfz/miniconda3/envs/sci310/bin/python"
SMI = "/usr/lib/wsl/lib/nvidia-smi"
RESULTS = Path("/home/rmcfz/seq_vp_tag_results")
REAL_ROOT = Path("/home/rmcfz/seq_vp_tag_results/eta_sweep_v2")
V3_ROOT = RESULTS/"eta_sweep_v3"
REAL_ROOT_V4 = RESULTS/"eta_sweep_v4"
# a v4 root never lies under these (the roots of v1-v3; read only)
PROTECTED_V4 = (RESULTS/"eta_sweep_v1", REAL_ROOT, V3_ROOT)
REAL_ROOTS = (Path("/home/rmcfz/seq_vp_tag_results/eta_sweep_v1"), REAL_ROOT, V3_ROOT, REAL_ROOT_V4)
REAL_SEG = Path("/home/rmcfz/seq_vp_tag_results/seg")
# v4's centre: the eta the v3 sweep chose (runs/20261030-eta-sweep/result.md "fd3eb2dda679"; the file M2V1 and the path
# pilot read)
V3_ETA = V3_ROOT/"eta_selected.json"
V3_ETA_SHA256 = "fd3eb2dda6798cf129705026c15438c7791aa74f0a541ad8c5d0e16eb1c2d5e1"
# where `python -m bayesreg.seg.unet.cli infer` writes by default (its --out/pred) and the M2 smoke v2 read them
DEFAULT_STRUCTURE_ROOT = REAL_SEG/"unet35_v1"/"pred"
SEQ_VP_3D_DATA = Path("/home/rmcfz/code/python_projects/seq_vp_3d/data")
STAGE = "stage_T0_tune_v1"
CHANNEL_SOURCES = ("structure", "tissue", "none")
INFER_POLICY = "all_folds_mean"          # D24: subjects outside every fold model's label contact (the T0 subjects)
DRY_SCRIPT = "experiments/eta/dry_inputs.py"
# options the sweep sets itself (run_pair and probe); --extra must not repeat them (one configuration per root)
MANAGED = ("--pair", "--out", "--channel-source", "--logvol-points", "--logvol-closed27-share", "--closed27-weights",
           "--metric-jd-scaling", "--jd-cell-estimate", "--backend", "--device", "--outer", "--rounds", "--wall-seconds",
           "--cpu-threads", "--data-root", "--allowlist", "--tissue-root", "--eta-json", "--structure-root",
           "--segmenter-registry", "--structure-registry-staging", "--structure-grouping", "--segmenter-id",
           "--structure-infer", "--structure-groups", "--sigma-domain", "--support-threshold")
INFER_ENV = "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1"
STRATA = ("moving larger", "moving smaller", "balanced")
TRAIN_UNIT = "tag-unet35-train"
D0_QUEUE = "tag-d0-queue"
MEM_MARGIN_MIB = 2048
POLL_SECONDS = 60
EVAL_MAX_SEC = 7200
PROBE_MAX_SEC = 3600
ATROPOS_MAX_SEC = 3600
METHOD = "bayesreg_eta_sweep"
SCHEMA = "bayesreg.eta_sweep.v2"
ALPHA_NUG_FACTOR = 1e-2          # C05-6 / D9 (= bayesreg.solve.multilevel.ALPHA_NUG_FACTOR; checked in the tests)
TIE_TOL = 1e-3
STAGES = ("s1", "s2", "s3")
UNTIL = ("precheck", "atropos", "probe", "centre")+STAGES
LABELS = dict(s1=(("x0p1", .1), ("x1", 1.), ("x10", 10.)),
              s2=(("x0p25", .25), ("x1", 1.), ("x4", 4.)),
              s3=(("st0p5", .5), ("st1", 1.), ("st2", 2.)))
REUSED = dict(s2="x1", s3="st1")          # the previous stage's choice, not rerun
# design v4 (ETA4; module docstring): the centre is the v3 choice, 4x steps in s1 (two below, one anchor above), 2x in
# s2, s3 as v3
DESIGNS = ("v3", "v4")
LABELS_V4 = dict(s1=(("x0p0625", .0625), ("x0p25", .25), ("x1", 1.), ("x4", 4.)),
                 s2=(("x0p5", .5), ("x1", 1.), ("x2", 2.)),
                 s3=LABELS["s3"])
# solver paths (only path switches; C13-4a / D51, C09.1-3a / D53) and the path pilot's arms that define them
# (experiments/pilot/run_pilot.py ARMS; runs/20261031-path-pilot/plan.md)
CAPS_P1 = (500, 500, 100, 500)
PATHS = dict(current=dict(caps=None, cell=JD_DEFAULT), p1=dict(caps=CAPS_P1, cell=JD_DEFAULT),
             p2=dict(caps=None, cell=CERTIFICATE_BERNSTEIN), p3=dict(caps=CAPS_P1, cell=CERTIFICATE_BERNSTEIN))
PILOT_SCHEMA = "bayesreg.path_pilot.summary.v1"
PILOT_PATH = dict(p0="current", p1="p1", p2="p2", p3="p3")             # the pilot's recommendation -> our path
PILOT_ARM = dict(current="p0n", p1="p1", p2="p2", p3="p3")              # our path -> the pilot arm that ran it
PILOT_PENDING = ("待审核", "待核对")                                     # verdict prefixes that hold the path
PILOT_DRIVER = "experiments/pilot/run_pilot.py"
PILOT_SUMMARIZE = "experiments/pilot/summarize.py"
PILOT_UNITS = "tag-pilot*"                                              # tag-pilot-*, tag-pilotdry-*
SDLOGJ_V4 = .10               # contract C05-7 item 7b / D56: mandatory under v4 (another value or none: --sdlogj-note)
MAX_ACCEPTED_DEFAULT = 500    # run_pair's --max-accepted default (C13-4, D19)
# run_pair config.json args a v4 run may record differently from the v3 run (contract 7b, ETA4 review M7): per run;
# the path's keys (checked against the path); paths inside each code tree (compared relative to it); SWLI's keys
# (absent in v3, brain / null now)
ARGS_PER_RUN = ("pair", "out", "eta_json")
ARGS_PATH = ("jd_cell_estimate", "max_accepted", "max_accepted_per_level", "level_time_shares")
ARGS_CODE_TREE = ("allowlist", "segmenter_registry", "structure_groups")
ARGS_SWLI = dict(sigma_domain="brain", support_threshold=None)
# an audit file (contract 7b "审核停止"): every key required (ETA4 review I3; round 2 R2-1: the recertification
# script's output file)
AUDIT_KEYS = ("state_hash", "violations", "certificate_config", "recertified_passed", "min_jd_lower_bound", "auditor",
              "utc", "recertification_record")
# the recertification (D53: the final state recertified without the prescreen; ETA4 review round 2 R2-2): configuration
# C of the main line without the prescreen -- the parent class of PrescreenCertificateConfig with delta 0, max_depth
# 48, max_boxes 255 (project rules "证书")
RECERT_CLASS = "CompiledFallbackCertificateConfig"
RECERT_FIELDS = dict(delta=0., max_depth=48, max_boxes=255)
RECERT_KEYS = ("tool", "schema", "script_sha256", "code_sha", "code_dirty", "run_dir", "export_file", "grid", "state_hash",
               "certificate_state_hash", "certificate_config", "certificate_version", "status", "passed",
               "min_jd_lower_bound", "n_uncertain", "n_infeasible", "n_prescreen_cells", "started_utc", "ended_utc")
# the recertification output must come from the repository's tool (ETA4 review round 3 R3-2; RECERT review)
RECERT_TOOL = "experiments/audit/recertify.py"
RECERT_SCHEMA = "bayesreg.audit.recertify.v1"
RECERT_TILE_SIZE = 8192                     # C07-2's T, the only one recorded for the audit (RECERT review S3)
SHA1_HEX = re.compile(r"^[0-9a-f]{40}$")
# an audit's utc may not lie later than the moment the sweep reads the file plus this (round 3 R3-3)
AUDIT_FUTURE_SLACK = 600.
# a no-export confirmation (ETA4 review round 2 R2-5): a run that needs the audit but has no exported state
NO_EXPORT_KEYS = ("run", "no_export", "state_hash", "auditor", "utc")
PER_LEVEL_CAPS = re.compile(r"^[0-9]+(,[0-9]+){3}$")                    # run_pair's grammar (C13-4a)
# run_pair options the sweep never passes through --extra (prefixes included); --max-accepted itself only with caps
SCH_OPTIONS = ("--max-accepted-per-level", "--level-time-shares")
DRIVER_SCRIPTS = ("experiments/eta/sweep.py", "experiments/m2v1/run_batch.py", "experiments/pilot/run_pilot.py")
MODEL_PATHS = ("bayesreg", "jd_curl", "configs/bayesreg")
# configuration keys in which a v4 root may differ from the v3 centre's configuration (the path; the design record)
PATH_KEYS = ("jd_cell_estimate", "max_accepted_per_level", "design", "path")
# readouts (v4; descriptive): evaluator metrics and certificate fields
READOUT_METRICS = ("seg35", "seg4", "hd95", "sdlogj_official", "sdlogj_brain", "logj_q01_brain", "logj_q99_brain",
                   "pct_J_lt_half_brain", "frac_abs_logj_gt_log2_brain", "central_det_nonpositive_brain", "min_J_centre")
READOUT_CERT = ("status", "passed", "min_jd_lower_bound", "n_cells", "n_prescreen_cells", "n_fallback_cells",
                "n_uncertain", "seconds")
MEASURES = (("seg35", "seg35"), ("seg4", "seg4"), ("hd95", "HD95"), ("sdlogj_official", "SDlogJ 官方"),
            ("sdlogj_brain", "SDlogJ 脑内"))
NOTE = ("开发集（stage_T0_tune_v1，3 对）；结果只作经验贝叶斯的初值（合同 C05-7、D10），不是最终 η。每次运行以开发开关 "
        "--wall-seconds {wall:g} 的“预算收尾”或其他 C13-3 收尾结束，不是“B 线收敛”，除非状态写明。单次观测。")
RULE = ("合格：三对都有无错误的评测、field_role ∈ {roles}、评测全证书通过、seg35 有限；得分 = 三对 seg35 的平均；"
        "与最高分相差 ≤ {tol:g} 的合格候选视为平局，依次取：离本阶段中心最近（|log 倍数| 最小）、seg4 平均高、倍数小。"
        "无合格候选则 STOP。")
RULE_SDLOGJ = "另加资格条件（v4，总负责人定）：三对官方 SDlogJ 的平均有限且 ≤ {cap:g}。"
EDGE_TEXT = dict(lower="网格下端", upper="网格上端", interior="网格内部")


class Stop(Exception):
    """The sweep stops here; evidence kept, nothing rerun (the lead decides)."""


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def write_once(path, document):
    """Write JSON (or text) to a new file; never overwrite."""
    path = Path(path)
    with open(path, "x", encoding="utf-8") as stream:
        if isinstance(document, str):
            stream.write(document)
        else:
            json.dump(document, stream, indent=1, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    return path


def to_mib(text):
    """systemd size (bytes, or K / M / G / T with base 1024; integers only) -> MiB (gpu_smoke.sh to_mib)."""
    match = re.fullmatch(r"([0-9]+)([KkMmGgTt]?)", str(text))
    if not match:
        raise ValueError(f"{text!r}: an integer size such as 7G")
    n, unit = int(match.group(1)), match.group(2).upper()
    return dict(K=n//1024, M=n, G=n*1024, T=n*1048576).get(unit, n//1048576)


def short_name(pair):
    """OASIS_OAS1_0012_MR1__OASIS_OAS1_0013_MR1 -> 0012-0013."""
    match = re.fullmatch(r"OASIS_OAS1_([0-9]{4})_MR1__OASIS_OAS1_([0-9]{4})_MR1", pair)
    if not match:
        raise Stop(f"{pair}: not an OASIS pair name")
    return f"{match.group(1)}-{match.group(2)}"


def subjects_of(pair):
    fixed, _, moving = pair.partition("__")
    return fixed, moving


def select_pairs(selection):
    """The fixed pair rule (module docstring): per stratum in STRATA order the first pair by name."""
    by_stratum = {}
    for row in selection["pairs"]:
        by_stratum.setdefault(row.get("stratum"), []).append(row["pair"])
    if set(by_stratum) != set(STRATA):
        raise Stop(f"selection.json strata {sorted(map(str, by_stratum))} are not {list(STRATA)}")
    return [min(by_stratum[s]) for s in STRATA]


def scaled(center, factor):
    """center x factor, written as the contract does (nu*/10, nu/4: a division for factors below 1)."""
    return center*factor if factor >= 1 else center/round(1/factor)


def alpha_nug(tau):
    """C05-6 (P2): 1e-2 min_k 1/tau_k^2, k = 0..K_T."""
    return ALPHA_NUG_FACTOR*min(1/(float(t)**2) for t in tau)


def candidate_eta(stage, label, factor, *, nu_center, tau0):
    """eta of one candidate (run_pair --eta-json reads alpha, nu, tau; the rest is a record)."""
    if stage == "s3":
        nu, s_tau = nu_center, factor
    else:
        nu, s_tau = scaled(nu_center, factor), 1.
    tau = [float(tau0[0])]+[s_tau*float(t) for t in tau0[1:]]
    return dict(alpha=alpha_nug(tau), nu=nu, tau=tau, stage=stage, label=label, factor=factor, s_tau=s_tau,
                nu_center=nu_center, tau0=list(tau0),
                rule=("C05-7 P2 sweep: tau_0 fixed (C05-6), tau_k = s_tau tau_k^(0); alpha = alpha_nug(tau) = "
                      "1e-2 min_k 1/tau_k^2 (not swept)"),
                source=f"experiments/eta/sweep.py {stage} {label}")


def geometric_mean(values):
    values = [float(v) for v in values]
    if not values or any(not (v > 0 and math.isfinite(v)) for v in values):
        raise Stop(f"nu*: positive finite values required, got {values}")
    return math.exp(sum(math.log(v) for v in values)/len(values))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def finite_only(doc):
    """A JSON document with every non-finite float replaced by None (readouts; write_once refuses NaN)."""
    if isinstance(doc, dict):
        return {k: finite_only(v) for k, v in doc.items()}
    if isinstance(doc, list):
        return [finite_only(v) for v in doc]
    if isinstance(doc, float) and not math.isfinite(doc):
        return None
    return doc


def number(value):
    """A float as a command-line value that parses back to the same float ('900', '1', '0.5'; repr otherwise)."""
    text = f"{float(value):g}"
    return text if float(text) == float(value) else repr(float(value))


def same_probe_configuration(got, want):
    """A probe record's `configuration` (experiments/eta/probe.py) against the sweep's probe_configuration(): equal,
    with the registry paths compared after resolving (the probe resolves the code tree's registry itself)."""
    if not isinstance(got, dict) or set(got) != set(want):
        return False
    for key, value in want.items():
        if key != "structure" and got[key] != value:
            return False
    a, b = got["structure"], want["structure"]
    if a is None or b is None:
        return a is None and b is None
    if not isinstance(a, dict) or not set(b) <= set(a):
        return False
    for key, value in b.items():
        if key == "registry":
            if a[key] is None or value is None or Path(a[key]).resolve() != Path(value).resolve():
                return False
        elif a[key] != value:
            return False
    return True


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, rel):
    """A repository script as a private module (experiments/m2v1/level_table.py, experiments/pilot/summarize.py), loaded
    only when needed (they are not imported at module level: experiments/m2v1 and experiments/pilot load this file)."""
    spec = importlib.util.spec_from_file_location(name, REPO/rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_caps(text):
    """--max-accepted-per-level a0,a1,a2,a3 (run_pair's grammar, C13-4a) -> list of four ints; ValueError otherwise."""
    if not isinstance(text, str) or PER_LEVEL_CAPS.fullmatch(text) is None:
        raise ValueError(f"{text!r}: four decimal integers >= 0, comma separated, nothing else (a0,a1,a2,a3; C13-4a)")
    return [int(part) for part in text.split(",")]


def parse_deadline(text):
    """--no-start-after: an ISO date-time with a date and an explicit UTC offset (as the path pilot: the window crosses
    midnight and DST changes) -> aware UTC datetime; ValueError otherwise."""
    try:
        moment = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        raise ValueError(f"{text!r}: an ISO date-time with a date and a UTC offset, e.g. 2026-11-02T06:00-06:00") from None
    if "T" not in text or moment.tzinfo is None:
        raise ValueError(f"{text!r}: a date, a time and an explicit UTC offset are required (e.g. 2026-11-02T06:00-06:00)")
    return moment.astimezone(timezone.utc)


def extra_conflicts(extra, caps):
    """--extra flags that the sweep refuses: run_pair's SCH switches (any prefix longer than --max-accepted, which is an
    option of its own), and with per-level caps --max-accepted too (run_pair refuses both, C13-4a)."""
    flags = {a.split("=", 1)[0] for a in shlex.split(extra) if a.startswith("--")}
    bad = {f for f in flags if any(m.startswith(f) for m in SCH_OPTIONS) and f != "--max-accepted"
           and not "--max-accepted".startswith(f)}
    if caps is not None:
        bad |= {f for f in flags if SCH_OPTIONS[0].startswith(f)}
    return sorted(bad)


def edge_of(candidates, chosen, key):
    """lower / upper / interior: where the chosen candidate's `key` (nu or s_tau) lies among the stage's candidates."""
    values = [c[key] for c in candidates]
    value = next(c[key] for c in candidates if c["label"] == chosen)
    if value == min(values):
        return "lower"
    if value == max(values):
        return "upper"
    return "interior"


def find_key(doc, key):
    """Every value of `key` anywhere in a JSON document (experiments/pilot/summarize.py find_key)."""
    out = []
    if isinstance(doc, dict):
        for k, v in doc.items():
            if k == key:
                out.append(v)
            out += find_key(v, key)
    elif isinstance(doc, list):
        for v in doc:
            out += find_key(v, key)
    return out


def bernstein_count(run_dir):
    """The run's certificate_bernstein check-(b) count (C09.1-3a, D53; the path pilot's reading, summarize.py
    bernstein_readouts): the larger of the run total (bernstein_j53_violations_total anywhere in status.json, hyper.json
    or timing.json) and the sum of n_bernstein_j53_violations over the log.jsonl jd_scaling rows of that cell estimate,
    whichever are recorded; None when neither is."""
    run_dir = Path(run_dir)
    totals = [v for name in ("status.json", "hyper.json", "timing.json")
              for v in find_key(read_json(run_dir/name), "bernstein_j53_violations_total")
              if isinstance(v, int) and not isinstance(v, bool)]
    known = [max(totals)] if totals else []
    rows = []
    if (run_dir/"log.jsonl").is_file():
        with open(run_dir/"log.jsonl", encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("kind") == "jd_scaling" \
                        and row.get("cell_estimate") == CERTIFICATE_BERNSTEIN:
                    rows.append(row)
    if rows and all(isinstance(r.get("n_bernstein_j53_violations"), int) for r in rows):
        known.append(sum(r["n_bernstein_j53_violations"] for r in rows))
    return max(known) if known else None


def parse_utc(text):
    """An ISO date-time with a date, a time and a time zone ("Z" or an offset) -> aware datetime; None otherwise."""
    if not isinstance(text, str) or "T" not in text:
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else None


def time_problems(value, run_end, now=None):
    """utc of an audit or a no-export confirmation (ETA4 review rounds 2 and 3, R2-3, R3-1, R3-3): a date, a time and a
    time zone; later than the run's end (run_end, aware; not determinable -> a problem, never a release); not later
    than `now` (the moment the sweep reads the file; aware) plus AUDIT_FUTURE_SLACK seconds."""
    moment = parse_utc(value)
    if moment is None:
        return [f"utc {value!r} is not an ISO date-time with a date, a time and a time zone"]
    out = []
    if run_end is None:
        out.append("the run's end cannot be determined (no done.json done_utc, no launch record, no status.json): the "
                   "time cannot be checked, not released")
    elif moment <= run_end:
        out.append(f"utc {value!r} is not later than the run's end {run_end.isoformat()}")
    now = datetime.now(timezone.utc) if now is None else now
    if (moment-now).total_seconds() > AUDIT_FUTURE_SLACK:
        out.append(f"utc {value!r} lies more than {AUDIT_FUTURE_SLACK:g} s after the moment the file was read "
                   f"({now.isoformat()})")
    return out


def recertification_problems(record, *, state_hash, run_end=None, audit_utc=None, tool_sha256=None, run_dir=None,
                             export_sha256=None, eval_grid_id=None, dirty_note=None):
    """What the recertification output (an audit's recertification_record; ETA4 review rounds 2 and 3, R2-1, R2-2,
    R3-2; RECERT review I1, S1-S3) lacks: every key of RECERT_KEYS;
      - written by the repository's tool: tool RECERT_TOOL, schema RECERT_SCHEMA, script_sha256 equal to the sha256 of
        REPO/RECERT_TOOL (or `tool_sha256` when given), code_sha 40 hex digits, code_dirty false (true only with an
        audit's `dirty_note`, recorded);
      - bound to the run (I1): run_dir equal to `run_dir` resolved, export_file.file_sha256 equal to the sha256 of the
        run's exported file (`export_sha256`: u, grid and status as one file), grid.rebuilt_grid_id equal to the
        evaluator certificate's grid_id (`eval_grid_id`, from the pair's grid, when recorded), state_hash equal to the
        run's exported state and certificate_state_hash equal to state_hash;
      - the configuration: certificate_config a JSON object whose "class" is RECERT_CLASS (configuration C without the
        prescreen) with RECERT_FIELDS, tile_size RECERT_TILE_SIZE, a positive integer fallback_workers (recorded), and
        fields that make a valid instance of that class; certificate_version that of this instance in this jd_curl;
        n_prescreen_cells 0;
      - the result: status certified_feasible, passed true, a finite min_jd_lower_bound > 0, no uncertain and no
        infeasible cell (integers 0);
      - started_utc and ended_utc with a zone, started after the run's end (when known) and ended not after the
        audit's utc (when given)."""
    if not isinstance(record, dict):
        return ["the recertification record is missing or not a JSON object"]
    out = [f"recertification record: missing {k}" for k in RECERT_KEYS if k not in record]
    if record.get("tool") != RECERT_TOOL:
        out.append(f"recertification record: tool {record.get('tool')!r} is not {RECERT_TOOL}")
    if record.get("schema") != RECERT_SCHEMA:
        out.append(f"recertification record: schema {record.get('schema')!r} is not {RECERT_SCHEMA}")
    if tool_sha256 is None:
        script = REPO/RECERT_TOOL
        tool_sha256 = file_sha256(script) if script.is_file() else None
    if tool_sha256 is None or record.get("script_sha256") != tool_sha256:
        out.append(f"recertification record: script_sha256 {record.get('script_sha256')!r} is not that of the "
                   f"repository's {RECERT_TOOL} ({tool_sha256!r})")
    if not (isinstance(record.get("code_sha"), str) and SHA1_HEX.fullmatch(record["code_sha"])):
        out.append(f"recertification record: code_sha {record.get('code_sha')!r} is not a 40-digit commit")
    if record.get("code_dirty") is not False and not (isinstance(dirty_note, str) and dirty_note.strip()):
        out.append(f"recertification record: code_dirty {record.get('code_dirty')!r} (false required, or the audit's "
                   "code_dirty_note)")
    if run_dir is not None and record.get("run_dir") != str(Path(run_dir).resolve()):
        out.append(f"recertification record: run_dir {record.get('run_dir')!r} is not {Path(run_dir).resolve()}")
    if export_sha256 is not None and (record.get("export_file") or {}).get("file_sha256") != export_sha256:
        out.append(f"recertification record: export_file sha256 {(record.get('export_file') or {}).get('file_sha256')!r}"
                   f" is not that of the run's exported file {export_sha256!r}")
    if eval_grid_id is not None and (record.get("grid") or {}).get("rebuilt_grid_id") != eval_grid_id:
        out.append(f"recertification record: grid_id {(record.get('grid') or {}).get('rebuilt_grid_id')!r} is not the "
                   f"evaluator certificate's {eval_grid_id!r}")
    if record.get("certificate_state_hash") != record.get("state_hash"):
        out.append(f"recertification record: certificate_state_hash {record.get('certificate_state_hash')!r} differs "
                   f"from state_hash {record.get('state_hash')!r}")
    if record.get("status") != "certified_feasible":
        out.append(f"recertification record: status {record.get('status')!r} is not certified_feasible")
    v = record.get("n_prescreen_cells")
    if not (isinstance(v, int) and not isinstance(v, bool) and v == 0):
        out.append(f"recertification record: n_prescreen_cells {v!r} (0: no prescreen)")
    for key in ("n_uncertain", "n_infeasible"):
        value = record.get(key)
        if not (isinstance(value, int) and not isinstance(value, bool) and value == 0):
            out.append(f"recertification record: {key} {value!r} (0 required: zero and uncertain do not pass)")
    started, ended = parse_utc(record.get("started_utc")), parse_utc(record.get("ended_utc"))
    if started is None or ended is None:
        out.append(f"recertification record: started_utc {record.get('started_utc')!r} / ended_utc "
                   f"{record.get('ended_utc')!r} are not ISO date-times with a zone")
    else:
        if run_end is not None and started <= run_end:
            out.append(f"recertification record: started {started.isoformat()} before the run's end "
                       f"{run_end.isoformat()}")
        moment = parse_utc(audit_utc) if audit_utc is not None else None
        if moment is not None and ended > moment:
            out.append(f"recertification record: ended {ended.isoformat()} after the audit's utc {audit_utc}")
    if state_hash is None or record.get("state_hash") != state_hash:
        out.append(f"recertification record: state_hash {record.get('state_hash')!r} is not the run's exported state "
                   f"{state_hash!r}")
    config = record.get("certificate_config")
    if not isinstance(config, dict) or config.get("class") != RECERT_CLASS:
        out.append(f"recertification record: certificate_config {config!r} is not a {RECERT_CLASS} object (configuration "
                   "C without the prescreen)")
    else:
        fields = {k: v for k, v in config.items() if k != "class"}
        for key, value in dict(RECERT_FIELDS, tile_size=RECERT_TILE_SIZE).items():
            if key not in fields or isinstance(fields[key], bool) or fields[key] != value:
                out.append(f"recertification record: certificate_config {key} {fields.get(key)!r} (configuration C "
                           f"without the prescreen: {value!r})")
        workers = fields.get("fallback_workers")
        if not (isinstance(workers, int) and not isinstance(workers, bool) and workers >= 1):
            out.append(f"recertification record: certificate_config fallback_workers {workers!r} (a positive integer)")
        try:
            instance = getattr(jd_feasibility, RECERT_CLASS)(**fields)
        except Exception as exc:      # noqa: BLE001 -- any invalid field set is a problem
            out.append(f"recertification record: certificate_config fields are not a valid {RECERT_CLASS} "
                       f"({type(exc).__name__}: {exc})")
        else:
            version = jd_feasibility.certificate_version(instance)
            if record.get("certificate_version") != version:
                out.append(f"recertification record: certificate_version {record.get('certificate_version')!r} is not "
                           f"this jd_curl's {version!r} for that configuration")
    if record.get("passed") is not True:
        out.append("recertification record: passed is not true")
    bound = record.get("min_jd_lower_bound")
    if not (finite(bound) and bound > 0):
        out.append(f"recertification record: min_jd_lower_bound {bound!r} is not a finite number > 0")
    return out


def audit_problems(audit, *, state_hash, eval_state_hash, count, run_end=None, record=None, now=None, binding=None):
    """What an audit file (contract 7b "审核停止", ETA4 review I3 and round 2) lacks for this run: every key of
    AUDIT_KEYS; the state_hash equal to the run's exported state (status.json "exported") and to the evaluator
    certificate's state_hash when recorded; violations an integer (never a bool) equal to the run's current check-(b)
    count (null when the count is not recorded); certificate_config RECERT_CLASS; recertified_passed true and a finite
    min_jd_lower_bound > 0; an auditor; utc with a time zone and later than the run's end (`run_end`); the
    recertification record (`record`, the JSON file named by recertification_record, loaded by the caller) passing
    recertification_problems -- bound to the run through `binding` (run_dir, export_sha256, eval_grid_id; RECERT review
    I1) -- and agreeing with the audit (class, bound); an optional code_dirty_note (non-empty) accepts a record made on a
    dirty code tree (RECERT review S1). Returns the list of problems (empty: released)."""
    if not isinstance(audit, dict):
        return ["no audit file (or not a JSON object)"]
    out = [f"missing {k}" for k in AUDIT_KEYS if k not in audit]
    if state_hash is None:
        out.append("the run has no exported state (status.json exported null): nothing to recertify; a no-export "
                   "confirmation (no_export) is the only way on, and the run counts as ineligible")
    elif audit.get("state_hash") != state_hash:
        out.append(f"state_hash {audit.get('state_hash')!r} is not the run's exported state {state_hash!r}")
    if eval_state_hash is not None and audit.get("state_hash") != eval_state_hash:
        out.append(f"state_hash {audit.get('state_hash')!r} is not the evaluator certificate's {eval_state_hash!r}")
    if "violations" in audit:
        v = audit["violations"]
        good = (v is None and count is None) or (isinstance(v, int) and not isinstance(v, bool) and v == count)
        if not good:
            out.append(f"violations {v!r} is not the run's count {count!r} (an integer, never a bool)")
    if audit.get("certificate_config") != RECERT_CLASS:
        out.append(f"certificate_config {audit.get('certificate_config')!r} is not {RECERT_CLASS} (configuration C "
                   "without the prescreen)")
    if audit.get("recertified_passed") is not True:
        out.append("recertified_passed is not true")
    bound = audit.get("min_jd_lower_bound")
    if not (finite(bound) and bound > 0):
        out.append(f"min_jd_lower_bound {bound!r} is not a finite number > 0")
    if not (isinstance(audit.get("auditor"), str) and audit["auditor"].strip()):
        out.append("auditor missing or empty")
    out += time_problems(audit.get("utc"), run_end, now)
    if "code_dirty_note" in audit and not (isinstance(audit["code_dirty_note"], str) and audit["code_dirty_note"].strip()):
        out.append("code_dirty_note empty (give it only to accept a recertification made on a dirty code tree)")
    if "recertification_record" in audit:
        out += recertification_problems(record, state_hash=state_hash, run_end=run_end, audit_utc=audit.get("utc"),
                                        dirty_note=audit.get("code_dirty_note"), **(binding or {}))
        if isinstance(record, dict) and record.get("min_jd_lower_bound") != bound:
            out.append(f"min_jd_lower_bound {bound!r} differs from the recertification record's "
                       f"{record.get('min_jd_lower_bound')!r}")
    return out


def no_export_problems(doc, *, name, state_hash, run_end, now=None):
    """What a no-export confirmation (ETA4 review round 2 R2-5) lacks: every key of NO_EXPORT_KEYS; run equal to this
    run's name (<stage>-<label>-<pair>); no_export true; state_hash null; the run itself without an exported state (a
    confirmation never releases a run that has one); an auditor; utc with a time zone and later than the run's end."""
    if not isinstance(doc, dict):
        return ["no confirmation file (or not a JSON object)"]
    out = [f"missing {k}" for k in NO_EXPORT_KEYS if k not in doc]
    if state_hash is not None:
        out.append(f"the run has an exported state {state_hash!r}: a no-export confirmation cannot release it (the "
                   "recertification audit is required)")
    if doc.get("run") != name:
        out.append(f"run {doc.get('run')!r} is not {name!r}")
    if doc.get("no_export") is not True:
        out.append("no_export is not true")
    if doc.get("state_hash", "") is not None:
        out.append(f"state_hash {doc.get('state_hash')!r} is not null")
    if not (isinstance(doc.get("auditor"), str) and doc["auditor"].strip()):
        out.append("auditor missing or empty")
    out += time_problems(doc.get("utc"), run_end, now)
    return out


def code_relative(value, code):
    """A path argument relative to its code tree ("<code>/..."), other values unchanged (args comparison, M7)."""
    if isinstance(value, str) and code is not None:
        prefix = str(code).rstrip("/")+"/"
        if value.startswith(prefix):
            return "<code>/"+value[len(prefix):]
    return value


def args_differences(v3_args, ours, *, v3_code, our_code, expected_path):
    """run_pair config.json args of a v4 run against the v3 run's (contract 7b; ETA4 review M7): per-run keys ignored;
    the path's keys must equal `expected_path` (absent keys read as null); code-tree paths compared relative to each
    tree; SWLI's keys absent in v3 must be brain / null; every other key equal. Returns (unexpected, allowed) as dicts
    key -> [v3 value, our value]."""
    unexpected, allowed = {}, {}
    for key in sorted(set(v3_args) | set(ours)):
        if key in ARGS_PER_RUN:
            continue
        a, b = v3_args.get(key), ours.get(key)
        if key in ARGS_PATH:
            if b != expected_path.get(key, a):
                unexpected[key] = [a, b]
            elif a != b:
                allowed[key] = [a, b]
        elif key in ARGS_CODE_TREE:
            ra, rb = code_relative(a, v3_code), code_relative(b, our_code)
            if ra != rb:
                unexpected[key] = [ra, rb]
            elif a != b:
                allowed[key] = [ra, rb]
        elif key in ARGS_SWLI and key not in v3_args:
            if b != ARGS_SWLI[key]:
                unexpected[key] = [a, b]
            else:
                allowed[key] = ["<absent>", b]
        elif a != b:
            unexpected[key] = [a, b]
    return unexpected, allowed


def pilot_ended(root, summary_path):
    """The path pilot's driver log says its last call ended and wrote this summary (contract 7b: the pilot's final
    summary): the last line of <root>/queue.log is "<UTC> end" and the last "summary:" line names `summary_path`.
    Returns the problems (empty: ended)."""
    log = Path(root)/"queue.log"
    if not log.is_file():
        return [f"{log} missing"]
    lines = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    out = []
    if not lines or re.fullmatch(r"\S+ end", lines[-1]) is None:
        out.append(f"the last line of {log} is not the driver's end ({(lines or [''])[-1][:120]!r})")
    summaries = [line for line in lines if re.match(r"\S+ summary: ", line)]
    named = summaries[-1].split(" summary: ", 1)[1].split(", ", 1)[0] if summaries else None
    if named is None or Path(named).resolve() != Path(summary_path).resolve():
        out.append(f"the last summary line of {log} names {named!r}, not {summary_path}")
    return out


def config_differences(centre, ours, allowed=PATH_KEYS):
    """Differences between the centre's configuration (the v3 eta_selected.json) and this root's, apart from the keys in
    `allowed`; structure root compared after resolving, the registry by sha256 / folds / status (its path is the code
    worktree of each root), as experiments/m2v1/run_batch.py config_differences."""
    if not isinstance(centre, dict):
        return [f"the centre's configuration is {centre!r}"]
    out = []
    for key in sorted(set(centre) | set(ours)):
        if key in allowed:
            continue
        a, b = centre.get(key), ours.get(key)
        if key == "structure":
            if not (isinstance(a, dict) and isinstance(b, dict)):
                if a != b:
                    out.append(f"structure: centre {a!r}, ours {b!r}")
                continue
            for sub in sorted((set(a) | set(b)) - {"root", "registry"}):
                if a.get(sub) != b.get(sub):
                    out.append(f"structure.{sub}: centre {a.get(sub)!r}, ours {b.get(sub)!r}")
            if a.get("root") is None or b.get("root") is None or Path(a["root"]).resolve() != Path(b["root"]).resolve():
                out.append(f"structure.root: centre {a.get('root')!r}, ours {b.get('root')!r}")
        elif key == "registry":
            a, b = a or {}, b or {}
            for sub in ("sha256", "folds", "status"):
                if a.get(sub) != b.get(sub):
                    out.append(f"registry.{sub}: centre {a.get(sub)!r}, ours {b.get(sub)!r}")
        elif a != b:
            out.append(f"{key}: centre {a!r}, ours {b!r}")
    return out


def pilot_path(summary_path, *, eta=None, dry=False, summarize=None):
    """The eta re-sweep path taken mechanically from the path pilot's summary (runs/20261031-path-pilot/plan.md "η 重扫路径
    的建议"; contract C05-7 item 7b; module docstring). Returns a record: status "resolved" (path p1 / p2 / p3), "lead"
    (the rule's branch 4: the current path, the lead decides) or "hold" (the path cannot be taken; path "current" is
    what stays), with the reasons. `eta` (alpha, nu, tau) must be the pilot's eta when given. `summarize` is the
    pilot's experiments/pilot/summarize.py module: its arm_flags, applied to the summary's own rows, comparisons and arm
    switches (no pilot run directory is read), must reproduce the recorded P1-P3 verdicts, and its recommend the
    recorded recommendation; without it (not loadable) -> hold. The pilot's queue.log must end with its driver's "end"
    after a summary line naming this file (pilot_ended)."""
    path = Path(summary_path)
    record = dict(source=str(path), sha256=None, status="hold", path="current", pilot_recommendation=None,
                  pilot_reason=None, verdicts={}, verdicts_recomputed={}, bernstein_violation_runs=None,
                  other_summaries=[], reasons=[])

    def hold(reason):
        record["reasons"].append(reason)
        return record

    if summarize is None:
        return hold(f"PILOT 的汇总工具 {PILOT_SUMMARIZE} 载入失败，无法重算读数标签与建议")
    if not path.is_file():
        return hold(f"{path}: 汇总缺失（PILOT 尚未写出 summary.json）")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return hold(f"{path}: 读不了（{exc}）")
    record["sha256"] = file_sha256(path)
    # PILOT writes summary.json once and a differing recomputation to summary_<UTC>.json: the newest is the last stamp
    stamped = sorted(path.parent.glob("summary_*.json"))
    newest = stamped[-1] if stamped else path.parent/"summary.json"
    record["other_summaries"] = sorted(str(p) for p in path.parent.glob("summary*.json")
                                       if p.resolve() != path.resolve())
    if path.resolve() != newest.resolve():
        return hold(f"{path} 不是该目录最新的汇总（最新 {newest.name}；重算不同时 PILOT 另写 summary_<UTC>.json）")
    if not isinstance(doc, dict) or doc.get("schema") != PILOT_SCHEMA:
        return hold(f"schema {doc.get('schema') if isinstance(doc, dict) else None!r}（应为 {PILOT_SCHEMA}）")
    # ETA4 review round 2 R2-6: a pilot stopped by its deadline also ends with "end"; how complete it is shows here
    counts = doc.get("counts") or {}
    record["pilot_counts"] = {k: counts.get(k) for k in ("ran", "not_run", "failed_no_status")}
    if doc.get("root") is None or Path(doc["root"]).resolve() != path.parent.resolve():
        return hold(f"汇总记录的根目录 {doc.get('root')} 不是 {path.parent}")
    under_results = path.resolve() == RESULTS or RESULTS in path.resolve().parents
    if dry and under_results:
        return hold(f"{path}: dry run 只接受 {RESULTS} 之外的合成 PILOT 汇总")
    if not dry and not under_results:
        return hold(f"{path}: 真实运行只接受 {RESULTS} 下的 PILOT 汇总")
    ended = pilot_ended(path.parent, path)
    if ended:
        return hold(f"PILOT 尚未结束，或这不是它最后写出的汇总：{'; '.join(ended)}")
    if doc.get("baseline") != "p0n":
        return hold(f"基线 {doc.get('baseline')!r}（应为 p0n）")
    arms = doc.get("arms") or {}
    for name, spec in PATHS.items():
        arm = PILOT_ARM[name]
        want = {}
        if spec["cell"] != JD_DEFAULT:
            want["jd_cell_estimate"] = spec["cell"]
        if spec["caps"] is not None:
            want["max_accepted_per_level"] = list(spec["caps"])
        if arm not in (doc.get("arm_order") or []) or (arms.get(arm) or {}).get("changes") != want:
            return hold(f"PILOT 臂 {arm} 缺失或开关不是 {want}（{(arms.get(arm) or {}).get('changes')!r}）")
    if eta is not None and doc.get("eta") != eta:
        return hold(f"PILOT 的 η {doc.get('eta')} 不是中心 η {eta}")
    recommendation = doc.get("recommendation") or {}
    record.update(pilot_recommendation=recommendation.get("path"), pilot_reason=recommendation.get("reason"))
    if recommendation.get("path") not in PILOT_PATH:
        return hold(f"建议 {recommendation.get('path')!r} 不在 {sorted(PILOT_PATH)} 中")
    flags = doc.get("flags") or {}
    verdicts = {a: (flags.get(a) or {}).get("verdict") for a in ("p1", "p2", "p3")}
    record["verdicts"] = verdicts
    if any(not isinstance(v, str) for v in verdicts.values()):
        return hold(f"P1–P3 的读数标签不全：{verdicts}")
    # the pilot's own rules on the summary's own data (contract 7b "重算"; ETA4 review M2): the P1-P3 verdicts, then
    # the recommendation from them
    try:
        rows, comparisons, base = doc["rows"], doc["comparisons"], doc["baseline"]
        again = {}
        for a in ("p1", "p2", "p3"):
            changes = arms[a]["changes"]
            again[a] = summarize.arm_flags(a, rows[a], rows[base], comparisons[a],
                                           changes.get("jd_cell_estimate", JD_DEFAULT),
                                           changes.get("max_accepted_per_level"))["verdict"]
    except Exception as exc:          # noqa: BLE001 -- any failure of the recomputation holds the path
        return hold(f"读数标签重算失败（{type(exc).__name__}: {exc}）")
    record["verdicts_recomputed"] = again
    if again != verdicts:
        return hold(f"按汇总数据重算的 P1–P3 读数标签 {again} 与记录的 {verdicts} 不同")
    try:
        rec = summarize.recommend(dict({a: (f or {}).get("verdict") for a, f in flags.items()}, **again),
                                  (doc.get("extra_comparisons") or {}).get("p3_vs_p1"))[0]
    except Exception as exc:          # noqa: BLE001
        return hold(f"建议重算失败（{type(exc).__name__}: {exc}）")
    if rec != recommendation.get("path"):
        return hold(f"按读数标签重算的建议 {rec!r} 与记录的 {recommendation.get('path')!r} 不同")
    runs = (doc.get("counts") or {}).get("bernstein_violation_runs")
    record["bernstein_violation_runs"] = runs
    if not isinstance(runs, list):
        return hold("counts.bernstein_violation_runs 缺失")
    if runs:
        return hold(f"PILOT 有 certificate_bernstein 核对 (b) 违反 > 0 的运行 {runs}（D53：审核完成前其数字不进结论）")
    pending = {a: v for a, v in verdicts.items() if v.startswith(PILOT_PENDING)}
    if pending:
        return hold(f"P1–P3 有读数标签为 待审核 / 待核对：{pending}")
    name = PILOT_PATH[recommendation["path"]]
    if name == "current":
        record.update(status="lead", path="current")
        record["reasons"].append("PILOT 规则第 4 条：没有臂倾向列入候选，维持现行路径，交总负责人")
        return record
    record.update(status="resolved", path=name)
    return record


def driver_processes(proc, self_pid, scripts=DRIVER_SCRIPTS):
    """PIDs of driver processes under the /proc tree `proc`: argv[0] is a python interpreter and some argument ends with
    one of `scripts`; this process and its ancestors excepted (experiments/m2v1/run_batch.py sweep_processes,
    experiments/pilot/run_pilot.py driver_processes)."""
    def argv_of(pid):
        try:
            raw = (Path(proc)/str(pid)/"cmdline").read_bytes()
        except OSError:
            return []
        return [a.decode("utf-8", "replace") for a in raw.split(b"\0") if a]

    def ppid_of(pid):
        try:
            text = (Path(proc)/str(pid)/"stat").read_text(encoding="utf-8", errors="replace")
            return int(text[text.rindex(")")+1:].split()[1])
        except (OSError, ValueError, IndexError):
            return None

    ancestors, pid = set(), int(self_pid)
    while pid and pid not in ancestors:
        ancestors.add(pid)
        pid = ppid_of(pid)
    found = []
    for entry in sorted(Path(proc).glob("[0-9]*"), key=lambda p: int(p.name) if p.name.isdigit() else -1):
        if not entry.name.isdigit() or int(entry.name) in ancestors:
            continue
        argv = argv_of(entry.name)
        if argv and Path(argv[0]).name.startswith("python") and any(a.endswith(s) for a in argv[1:] for s in scripts):
            found.append(int(entry.name))
    return found


def eligibility(rows):
    """(eligible, reasons) of one candidate's per-pair rows (module docstring)."""
    reasons = []
    for row in rows:
        tag = row["pair"]
        if row.get("eval_error"):
            reasons.append(f"{tag}: evaluation error")
        elif row.get("field_role") not in COMPARABLE_ROLES:
            reasons.append(f"{tag}: field_role {row.get('field_role')}")
        elif (row.get("certificate") or {}).get("passed") is not True:
            reasons.append(f"{tag}: certificate not passed")
        elif not finite((row.get("metrics") or {}).get("seg35")):
            reasons.append(f"{tag}: seg35 not finite")
    return not reasons, reasons


def mean_of(rows, key):
    values = [(r.get("metrics") or {}).get(key) for r in rows]
    values = [v for v in values if finite(v)]
    return sum(values)/len(values) if len(values) == len(rows) and values else None


def choose(candidates, tol=TIE_TOL):
    """The selection rule: (chosen candidate or None, tied labels, reason)."""
    pool = [c for c in candidates if c["eligible"]]
    if not pool:
        return None, [], "no eligible candidate"
    best = max(c["mean_seg35"] for c in pool)
    tied = [c for c in pool if c["mean_seg35"] >= best-tol]
    seg4 = (lambda c: c["mean_seg4"] if c["mean_seg4"] is not None else -math.inf)
    # |log factor| rounded: log(.1) and log(10) differ in the last bit and must tie
    tied.sort(key=lambda c: (round(abs(math.log(c["factor"])), 9), -seg4(c), c["factor"]))
    chosen = tied[0]
    if len(tied) == 1:
        reason = f"best mean seg35 {chosen['mean_seg35']:.6f}"
    else:
        reason = (f"tie within {tol:g} of the best mean seg35 {best:.6f} among {[c['label'] for c in tied]}: closest "
                  "to the centre, then mean seg4, then the smaller factor")
    return chosen, [c["label"] for c in tied], reason


def fmt(value, nd=4):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{nd}g}" if abs(value) < 1e-3 and value != 0 else f"{value:.{nd}f}"
    return str(value)


def configuration_text(configuration):
    """One line of the configuration for the Markdown tables (the JSON records carry the full dict)."""
    if not configuration:
        return "—"
    c = configuration
    parts = [f"channel_source {c.get('channel_source')}", f"logvol_points {c.get('logvol_points')}"]
    if c.get("logvol_closed27_share") is not None:
        parts.append(f"β_S {c['logvol_closed27_share']:g}（{c.get('closed27_weights')}）")
    parts += [f"metric_jd_scaling {c.get('metric_jd_scaling')}", f"jd_cell_estimate {c.get('jd_cell_estimate')}"]
    if c.get("max_accepted_per_level") is not None:  # ETA4: only with per-level caps
        parts.append(f"max_accepted_per_level {','.join(str(a) for a in c['max_accepted_per_level'])}")
    if c.get("design") is not None:                  # ETA4: only in a v4 configuration
        parts.append(f"design {c['design']}，路径 {c.get('path')}")
    if c.get("sigma_domain") is not None:             # SWLI: only in a support / support_shared configuration
        parts.append(f"sigma_domain {c['sigma_domain']}（θ {c.get('support_threshold'):g}）")
    for key in ("backend", "device", "outer", "rounds", "wall_seconds"):
        if key in c:
            parts.append(f"{key} {c[key]}")
    structure = c.get("structure")
    if structure:
        parts.append(f"结构层 {structure.get('segmenter_id')}（{structure.get('grouping')}，登记表 "
                     f"{structure.get('registry_status')}，{structure.get('root')}）")
    if c.get("extra"):
        parts.append(f"extra {c['extra']!r}")
    return "，".join(parts)


def stage_markdown(selection, wall):
    head = ["候选", "ν", "s_τ", "对", *(t for _, t in MEASURES), "证书（评测）", "状态", "field_role"]
    lines = [f"## 阶段 {selection['stage']}（中心 ν = {selection['nu_center']:.6g}，"
             f"{'s_τ 扫描' if selection['stage'] == 's3' else 's_τ = 1'}）", "",
             NOTE.format(wall=wall), "", f"配置：{configuration_text(selection.get('configuration'))}", "",
             "| "+" | ".join(head)+" |", "|"+"---|"*len(head)]
    for c in selection["candidates"]:
        label = c["label"]+(f"（复用 {c['reused_from']}）" if c.get("reused_from") else "")
        for r in c["rows"]:
            m, cert = r.get("metrics") or {}, r.get("certificate") or {}
            cells = [label, f"{c['nu']:.6g}", fmt(c["s_tau"], 3), r["pair"],
                     *(fmt(m.get(k), 3 if k == "hd95" else 4) for k, _ in MEASURES),
                     f"{cert.get('status')} / {cert.get('passed')} / {fmt(cert.get('min_jd_lower_bound'), 3)}",
                     f"{r.get('run_status')}（{r.get('wording')}，exit {r.get('exit_code')}）", str(r.get("field_role"))]
            lines.append("| "+" | ".join(str(x).replace("|", "/") for x in cells)+" |")
    v4 = selection.get("design") == "v4"
    if not v4:
        lines += ["", "| 候选 | ν | s_τ | seg35 平均 | seg4 平均 | 合格 | 不合格原因 | 选中 |", "|---|---|---|---|---|---|---|---|"]
    else:                                   # ETA4: the mean official SDlogJ next to the Dice means
        lines += ["", "| 候选 | ν | s_τ | seg35 平均 | seg4 平均 | 官方 SDlogJ 平均 | 合格 | 不合格原因 | 选中 |",
                  "|---|---|---|---|---|---|---|---|---|"]
    for c in selection["candidates"]:
        sd = f" {fmt(c.get('mean_sdlogj_official'))} |" if v4 else ""
        lines.append(f"| {c['label']} | {c['nu']:.6g} | {fmt(c['s_tau'], 3)} | {fmt(c['mean_seg35'])} | "
                     f"{fmt(c['mean_seg4'])} |{sd} {c['eligible']} | {'; '.join(c['ineligible']) or '—'} | "
                     f"{'**是**' if c['label'] == selection['chosen'] else ''} |")
    lines += ["", f"选择：{selection['chosen']}（{selection['reason']}）；规则：{selection['rule']}", ""]
    if v4 and selection.get("edge"):
        lines += [f"选中值在本段{EDGE_TEXT[selection['edge']]}（只记录，不自动扩展网格）。", ""]
    return "\n".join(lines)


class System:
    """The machine: systemd, nvidia-smi, /proc, df (replaced by a fake in the tests)."""

    def __init__(self):
        cgroup = Path("/proc/self/cgroup")
        match = re.search(r"^0::.*/([^/]*\.service)$", cgroup.read_text() if cgroup.is_file() else "", re.M)
        self.self_unit = match.group(1) if match else ""

    @staticmethod
    def _out(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
        except OSError as exc:
            return f"({exc})"

    def unit_active(self, unit):
        return subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit], check=False).returncode == 0

    def other_units(self):
        out = self._out(["systemctl", "--user", "list-units", "--state=active,activating,deactivating", "--no-legend",
                         "--plain", "seq-vp-*", "tag-*"])
        units = [line.split()[0] for line in out.splitlines() if line.strip()]
        return [u for u in units if u not in (f"{TRAIN_UNIT}.service", self.self_unit)]

    def active_units(self):
        out = self._out(["systemctl", "--user", "list-units", "--state=active", "--no-legend", "--plain", "tag-*",
                         "seq-vp-*"])
        return [line.split()[0] for line in out.splitlines() if line.strip()]

    def gpu_free_mib(self):
        text = self._out([SMI, "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).strip().splitlines()
        return int(text[0].strip()) if text and text[0].strip().isdigit() else None

    def gpu_state(self):
        text = self._out([SMI, "--query-gpu=name,memory.used,memory.free,utilization.gpu", "--format=csv,noheader"])
        return (text.splitlines() or [""])[0]

    def mem_available_mib(self):
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])//1024
        return 0

    def home_free_gb(self):
        text = self._out(["df", "--output=avail", "-B1G", "/home"]).split()
        return int(text[-1]) if text and text[-1].isdigit() else 0

    def train_state(self):
        state = self._out(["systemctl", "--user", "show", TRAIN_UNIT, "-p", "ActiveState", "-p", "MemoryCurrent", "-p",
                           "MemoryMax"]).replace("\n", " ")
        current = self._out(["systemctl", "--user", "show", TRAIN_UNIT, "-p", "MemoryCurrent", "--value"]).strip()
        return f"{TRAIN_UNIT}: {state}" + (f"(MemoryCurrent {int(current)//1048576} MiB)" if current.isdigit() else "")

    def uptime(self):
        return float(Path("/proc/uptime").read_text().split()[0])

    def loadavg(self):
        return " ".join(Path("/proc/loadavg").read_text().split()[:3])

    def dmesg_since(self, t0, pattern):
        out = self._out(["dmesg"])
        count = 0
        for line in out.splitlines():
            match = re.match(r"^\[ *([0-9.]+)\]", line)
            if match and float(match.group(1)) >= t0 and re.search(pattern, line):
                count += 1
        return count

    def run(self, cmd, record):
        """Run the systemd-run command, its own output appended to `record`; returns the exit code."""
        with open(record, "a", encoding="utf-8") as stream:
            return subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=False).returncode

    def sleep(self, seconds):
        time.sleep(seconds)

    # ETA4 (design v4 and --no-start-after only)
    def now(self):
        return datetime.now(timezone.utc)

    def drivers_running(self):
        """PIDs of other driver processes (eta sweep, M2V1, path pilot) on the real /proc."""
        return driver_processes(Path("/proc"), os.getpid())

    def pilot_state(self):
        """(active / activating / deactivating tag-pilot* units, PIDs of experiments/pilot/run_pilot.py processes)."""
        out = self._out(["systemctl", "--user", "list-units", "--state=active,activating,deactivating", "--no-legend",
                         "--plain", PILOT_UNITS])
        return ([line.split()[0] for line in out.splitlines() if line.strip()],
                driver_processes(Path("/proc"), os.getpid(), (PILOT_DRIVER,)))

    def check_run_args(self, code, args):
        """run_pair's own parse_args and args_record on `args` in the code tree (CPU; nothing is read or written):
        (ok, message, the config.json args record or None)."""
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        out = subprocess.run([PY, "-B", "-c", "import json, sys\nfrom bayesreg.cli.run_pair import args_record, "
                              "parse_args\nprint('ARGS ' + json.dumps(args_record(parse_args(sys.argv[1:]))))",
                              *[str(a) for a in args]], cwd=code, env=env, capture_output=True, text=True, check=False)
        lines = [line for line in out.stdout.splitlines() if line.startswith("ARGS ")]
        record = json.loads(lines[-1][5:]) if out.returncode == 0 and lines else None
        return out.returncode == 0 and record is not None, (out.stderr or out.stdout).strip()[-2000:], record


class Sweep:
    def __init__(self, opts, system=None):
        self.o = opts
        self.sys = System() if system is None else system
        self.root = Path(opts.root)
        self.dry = opts.dry_run
        self.prefix = "tag-etadry" if self.dry else "tag-eta"
        self.device = "cpu" if self.dry else opts.device
        self.threads = 1 if self.dry else 4
        self.quota = "100%" if self.dry else "600%"
        self.data_root = Path(opts.data_root) if opts.data_root else REPO/"data"/"three_d"
        self.tissue_root = Path(opts.tissue_root) if opts.tissue_root else REAL_SEG/"atropos_v1"
        self.allowlist = opts.allowlist
        self.memmax_mib = to_mib(opts.memory_max)
        self.eval_mib = to_mib(opts.eval_memory_max)
        self.probe_mib = to_mib(opts.probe_memory_max)
        self.atropos_mib = to_mib(opts.atropos_memory_max)
        self.structure = opts.channel_source == "structure"
        self.closed27 = opts.logvol_points == LOGVOL_GAUSS27_CLOSED27
        # C04-6 sigma_domain (SWLI); getattr: subclasses with their own options (experiments/m2v1/run_batch.py ArmSweep)
        # run brain
        self.sigma_domain = getattr(opts, "sigma_domain", None) or BRAIN
        self.support_threshold = getattr(opts, "support_threshold", None)
        self.support = self.sigma_domain != BRAIN
        self.structure_root = Path(opts.structure_root)
        self.registry_path = None          # set by structure_precheck (the code tree's registry; a dry run's staging one)
        self.registry_record = None
        self.code = None
        self.code_sha = "n/a"
        self.pairs = []
        # ETA4; getattr: the subclasses of experiments/m2v1 and experiments/pilot have neither (design v3, no caps, no
        # deadline: everything as before)
        self.v4 = getattr(opts, "design", "v3") == "v4"
        if self.v4:                        # ETA4 review M9: v4 units are told apart from v3's in the journal
            self.prefix = "tag-eta4dry" if self.dry else "tag-eta4"
        self.labels = LABELS_V4 if self.v4 else LABELS
        caps = getattr(opts, "max_accepted_per_level", None)
        self.level_caps = None if caps is None else list(caps)
        self.path_name = None              # v4: set by resolve_path
        self.no_start_after = getattr(opts, "no_start_after_utc", None)
        self.deadline_run = False          # True while the gate of a run unit waits (--no-start-after)

    # ---- configuration -------------------------------------------------------------------------------------------
    def probe_configuration(self):
        """The configuration that the probe records (experiments/eta/probe.py configuration(); compared in
        probe_phase): beta_S and the closed27 rule null unless gauss27+closed27, structure null unless structure."""
        structure = None
        if self.structure:
            structure = dict(root=str(self.structure_root), grouping=self.o.structure_grouping,
                             segmenter_id=self.o.segmenter_id,
                             registry=None if self.registry_path is None else str(self.registry_path),
                             registry_status="staging_unregistered" if self.dry else "registered",
                             staging_opt_in=self.dry, infer="never")
        out = dict(channel_source=self.o.channel_source, logvol_points=self.o.logvol_points,
                   logvol_closed27_share=float(self.o.logvol_closed27_share) if self.closed27 else None,
                   closed27_weights=CLOSED27_SIMPSON if self.closed27 else None,
                   metric_jd_scaling=self.o.metric_jd_scaling, jd_cell_estimate=self.o.jd_cell_estimate,
                   structure=structure)
        if self.support:                    # SWLI: only under support / support_shared (brain: the record before)
            out.update(sigma_domain=self.sigma_domain, support_threshold=float(self.support_threshold))
        return out

    def configuration(self):
        """The full configuration of this root (nu_star.json, every selection, summary.json, eta_selected.json and
        every launch record)."""
        out = dict(self.probe_configuration(), backend=self.o.backend,
                   device=self.device if self.o.backend == "torch64" else None, outer=self.o.outer,
                   rounds=self.o.rounds, wall_seconds=self.o.wall_seconds, cpu_threads=self.threads,
                   opt1="run_pair defaults (--jacobian-cache on, --jacobian-cache-log off, --speculative-gradient on)"
                   if self.o.backend == "torch64" else None,
                   extra=self.o.extra, registry=self.registry_record)
        if self.level_caps is not None:     # ETA4: per-level caps (run_pair only; the probe's cap is 1)
            out["max_accepted_per_level"] = list(self.level_caps)
        if self.v4:                         # ETA4: the design and the path (path.json holds its source)
            out.update(design="v4", path=self.path_name)
        return out

    def expected_run_args(self):
        """run_pair's argument values (config.json args) that this configuration implies (pair_row checks them)."""
        expected = dict(channel_source=self.o.channel_source, logvol_points=self.o.logvol_points,
                        logvol_closed27_share=float(self.o.logvol_closed27_share) if self.closed27 else 1.,
                        closed27_weights=CLOSED27_SIMPSON, metric_jd_scaling=self.o.metric_jd_scaling,
                        jd_cell_estimate=self.o.jd_cell_estimate, backend=self.o.backend, outer=self.o.outer,
                        rounds=self.o.rounds, wall_seconds=float(self.o.wall_seconds))
        if self.o.backend == "torch64":
            expected["device"] = self.device
        if self.support:                    # SWLI (brain: not checked, so runs of the code before SWLI still pass)
            expected.update(sigma_domain=self.sigma_domain, support_threshold=float(self.support_threshold))
        if self.structure:
            expected.update(structure_root=str(self.structure_root), structure_grouping=self.o.structure_grouping,
                            segmenter_id=self.o.segmenter_id, structure_infer="never",
                            structure_registry_staging=self.dry)
        # ETA4 (C16 SCH): with caps run_pair records max_accepted null, the caps and level_time_shares null; without them
        # in a v4 root neither SCH key is written (an absent key reads as null)
        if self.level_caps is not None:
            expected.update(max_accepted=None, max_accepted_per_level=list(self.level_caps), level_time_shares=None)
        elif self.v4:
            expected.update(max_accepted=self.extra_max_accepted(), max_accepted_per_level=None, level_time_shares=None)
        if self.v4:                         # contract 7b scope (ETA4 review M6): SWLI's brain default, nothing else
            expected.update(sigma_domain=BRAIN, support_threshold=None)
        return expected

    def extra_max_accepted(self):
        """run_pair's max_accepted without per-level caps: the --extra value when given (dry runs), else 500 (D19)."""
        words = shlex.split(self.o.extra)
        for k, word in enumerate(words):
            if word == "--max-accepted" and k+1 < len(words):
                return int(words[k+1])
            if word.startswith("--max-accepted="):
                return int(word.split("=", 1)[1])
        return MAX_ACCEPTED_DEFAULT

    def path_expected_args(self):
        """The path's run_pair args (config.json) for the comparison with the v3 run (args_differences)."""
        if self.level_caps is not None:
            return dict(jd_cell_estimate=self.o.jd_cell_estimate, max_accepted=None,
                        max_accepted_per_level=list(self.level_caps), level_time_shares=None)
        return dict(jd_cell_estimate=self.o.jd_cell_estimate, max_accepted=self.extra_max_accepted(),
                    max_accepted_per_level=None, level_time_shares=None)

    def model_args(self):
        """The model / path options shared by the probe and run_pair (the same values for both)."""
        args = ["--channel-source", self.o.channel_source, "--logvol-points", self.o.logvol_points]
        if self.closed27:
            args += ["--logvol-closed27-share", number(self.o.logvol_closed27_share), "--closed27-weights",
                     CLOSED27_SIMPSON]
        args += ["--metric-jd-scaling", self.o.metric_jd_scaling]
        if self.o.jd_cell_estimate != JD_NONE:          # "none" (metric off): not passed; run_pair records "none" (DEF)
            args += ["--jd-cell-estimate", self.o.jd_cell_estimate]
        if self.support:                                # SWLI; brain: nothing passed (the commands before SWLI)
            args += ["--sigma-domain", self.sigma_domain, "--support-threshold", number(self.support_threshold)]
        if self.structure:
            args += ["--structure-root", str(self.structure_root), "--structure-grouping", self.o.structure_grouping,
                     "--segmenter-id", self.o.segmenter_id]
            if self.dry:
                args += ["--segmenter-registry", str(self.registry_path), "--structure-registry-staging"]
        return args

    # ---- log -------------------------------------------------------------------------------------------------
    def log(self, text):
        line = f"{utc()} {text}"
        print(line, flush=True)
        with open(self.root/"sweep.log", "a", encoding="utf-8") as stream:
            stream.write(line+"\n")

    # ---- start checks --------------------------------------------------------------------------------------------
    def dry_checks(self):
        if not (self.o.data_root and self.o.tissue_root and self.o.allowlist):
            raise Stop("--dry-run: --data-root, --tissue-root and --allowlist are required "
                       "(experiments/eta/dry_inputs.py prints them)")
        paths = [self.o.data_root, self.o.tissue_root, self.o.allowlist]
        if self.structure:
            if not self.o.segmenter_registry or self.o.structure_root == str(DEFAULT_STRUCTURE_ROOT):
                raise Stop("--dry-run with --channel-source structure: --structure-root and --segmenter-registry are "
                           "required (experiments/eta/dry_inputs.py prints them)")
            paths += [self.o.structure_root, self.o.segmenter_registry]
        forbidden = [REPO/"data", (REPO/"data").resolve(), SEQ_VP_3D_DATA, REAL_SEG]
        for path in paths:
            for form in (Path(os.path.abspath(path)), Path(path).resolve()):
                for bad in forbidden:
                    if form == bad or bad in form.parents:
                        raise Stop(f"--dry-run: {path} lies under {bad} (real data); synthetic inputs only")
        # ETA4: the centre eta and the pilot summary of a dry run are synthetic, never a real result
        for path in (getattr(self.o, "centre_eta", None), getattr(self.o, "path_from_pilot", None)):
            if path is None:
                continue
            for form in (Path(os.path.abspath(path)), Path(path).resolve()):
                if form == RESULTS or RESULTS in form.parents:
                    raise Stop(f"--dry-run: {path} lies under {RESULTS} (a real result); synthetic inputs only")
        if "synthetic_dry_run" not in Path(self.o.allowlist).read_text(encoding="utf-8"):
            raise Stop(f"--dry-run: {self.o.allowlist} is not the dry_inputs.py allowlist")

    def start_checks(self):
        if self.o.segmenter_registry and not self.dry:
            raise Stop("--segmenter-registry: dry runs only (real runs use the code tree's registered "
                       "configs/bayesreg/segmenters.json)")
        if self.dry:
            self.dry_checks()
        stage_dir = self.data_root/STAGE
        selection = read_json(stage_dir/"selection.json")
        if selection is None:
            raise Stop(f"{stage_dir}/selection.json missing (T0 not usable as expected; reported, no other stage used)")
        self.pairs = select_pairs(selection)
        for pair in self.pairs:
            for name in ("pair.npz", "manifest.json"):
                if not (stage_dir/pair/name).is_file():
                    raise Stop(f"{stage_dir/pair/name} missing (T0 not usable; no other stage used)")
            manifest = read_json(stage_dir/pair/"manifest.json")
            if [manifest.get("fixed_subject"), manifest.get("moving_subject")] != list(subjects_of(pair)):
                raise Stop(f"{pair}: manifest subjects {manifest.get('fixed_subject')}, {manifest.get('moving_subject')}")
            if self.dry and "experiments/eta/dry_inputs.py" not in json.dumps(manifest):
                raise Stop(f"--dry-run: {stage_dir/pair}/manifest.json is not a dry_inputs.py pair")
        self.log(f"pairs (rule: per stratum {list(STRATA)} the first by name in {STAGE}/selection.json): "
                 + ", ".join(f"{p} ({short_name(p)})" for p in self.pairs))

    def git(self, *args, cwd=REPO):
        return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=False)

    def prepare_code(self):
        explicit = bool(self.o.commit)
        if not self.o.code:
            commit = self.o.commit or "HEAD"
            out = self.git("rev-parse", "--verify", f"{commit}^{{commit}}")
            if out.returncode:
                raise Stop(f"commit {commit} not found in {REPO}")
            sha = out.stdout.strip()
            self.log(f"commit: --commit {commit} -> {sha} (explicit {int(explicit)}; resolved in {REPO} at start)")
            code, record = self.root/"code", self.root/"code_commit.txt"
            if code.exists():
                if not record.is_file():
                    raise Stop(f"{code} exists without {record}")
                want = record.read_text().strip()
                if self.git("rev-parse", "HEAD", cwd=code).stdout.strip() != want:
                    raise Stop(f"{code} is not at the recorded commit {want}")
                if self.git("status", "--porcelain", cwd=code).stdout.strip():
                    raise Stop(f"{code} has changes (git status --porcelain not empty)")
                if sha != want:
                    if explicit:
                        raise Stop(f"--commit {sha} differs from the commit {want} of the existing worktree {code}")
                    self.log(f"code: the repository HEAD is now {sha}; the existing worktree keeps {want} (resume)")
                self.log(f"code: reusing worktree {code} at {want}")
            else:
                if record.exists():
                    raise Stop(f"{record} exists without {code} (kept; the lead decides)")
                out = self.git("worktree", "add", "--detach", str(code), sha)
                with open(self.root/"sweep.log", "a", encoding="utf-8") as stream:
                    stream.write(out.stdout+out.stderr)
                if out.returncode:
                    raise Stop(f"git worktree add {code} {sha} failed")
                write_once(record, sha+"\n")
                self.log(f"code: new detached worktree {code} at {sha}")
            self.code = code
        else:
            code = Path(self.o.code)
            head = self.git("rev-parse", "HEAD", cwd=code)
            if not code.is_dir() or head.returncode:
                raise Stop(f"--code {code} is not a git tree")
            if explicit:
                sha = self.git("rev-parse", "--verify", f"{self.o.commit}^{{commit}}", cwd=code).stdout.strip()
                if sha != head.stdout.strip():
                    raise Stop(f"--code {code} is at {head.stdout.strip()}, not at --commit {sha}")
            if self.git("status", "--porcelain", "--untracked-files=no", cwd=code).stdout.strip():
                raise Stop(f"--code {code} has tracked changes")
            self.code = code
        self.code_sha = self.git("rev-parse", "HEAD", cwd=self.code).stdout.strip() or "n/a"
        if not self.allowlist:
            self.allowlist = str(self.code/"configs"/"bayesreg"/"dev_subjects.json")
        if not Path(self.allowlist).is_file():
            raise Stop(f"allowlist {self.allowlist} missing")
        self.check_code_tree()
        self.log(f"code {self.code}: HEAD {self.code_sha}; data root {self.data_root}; tissue root {self.tissue_root}; "
                 f"allowlist {self.allowlist}")

    def check_code_tree(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        need = {("-m", "bayesreg.cli.run_pair"): ("--eta-json", "--wall-seconds", "--logvol-points", "gauss27+closed27",
                                                  "--logvol-closed27-share", "--jd-cell-estimate", "certificate",
                                                  "--structure-root", "--structure-registry-staging"),
                ("experiments/eta/probe.py",): ("--channel-source", "--logvol-points", "gauss27+closed27",
                                                "--logvol-closed27-share", "--jd-cell-estimate", "--structure-root",
                                                "--structure-registry-staging"),
                ("experiments/d0/atropos_seg.py",): ("--stage", "--data-root", "--allowlist")}
        if self.support:                    # SWLI: a code tree without the switches cannot run this configuration
            for target in (("-m", "bayesreg.cli.run_pair"), ("experiments/eta/probe.py",)):
                need[target] += ("--sigma-domain", "--support-threshold")
        # ETA4: a v4 root may take any path (resolved after this check); explicit switches need their code
        if self.v4 or self.level_caps is not None:
            need[("-m", "bayesreg.cli.run_pair")] += ("--max-accepted-per-level",)
        if self.v4 or self.o.jd_cell_estimate == CERTIFICATE_BERNSTEIN:
            need[("-m", "bayesreg.cli.run_pair")] += (CERTIFICATE_BERNSTEIN,)
        for target, flags in need.items():
            out = subprocess.run([PY, "-B", *target, "--help"], cwd=self.code, env=env, capture_output=True, text=True,
                                 check=False).stdout
            missing = [f for f in flags if f not in out]
            if missing:
                raise Stop(f"{' '.join(target)} in {self.code} lacks {missing} (merge the ETA and ETA2 tooling first)")

    def write_env(self):
        path = self.root/f"env_{stamp()}.txt"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        versions = subprocess.run([PY, "-B", "-c", "import importlib\nfor m in ('numpy','scipy','numba','torch','nibabel'):"
                                   "\n    print(m, importlib.import_module(m).__version__)\nimport torch\n"
                                   "print('torch.version.cuda', torch.version.cuda)"], cwd=self.code, env=env,
                                  capture_output=True, text=True, check=False)
        lines = [f"recorded: {utc()}", f"code: {self.code} at {self.code_sha}", f"python: {PY}", versions.stdout.strip(),
                 f"nvidia-smi: {self.sys.gpu_state()}", f"memory available: {self.sys.mem_available_mib()} MiB",
                 "seed: none (run_pair and the probe are deterministic; the wall budget is not)"]
        write_once(path, "\n".join(lines)+"\n")
        self.log(f"env: {path}")

    # ---- structure precheck ----------------------------------------------------------------------------------------
    def infer_command(self, subjects_file):
        """The inference command for the missing predictions (bayesreg.seg.unet.cli infer: pinned allowlist, the
        repository registry, all_folds_mean for subjects outside every label contact, its own machine gate, GPU;
        never overwrites). Run in the repository (a clean work tree; its data/ link verifies the allowlist sources)."""
        command = [PY, "-B", "-m", "bayesreg.seg.unet.cli", "infer", "--subjects-file", str(subjects_file)]
        if os.path.abspath(self.structure_root) != os.path.abspath(DEFAULT_STRUCTURE_ROOT):
            command += ["--pred-out", str(self.structure_root)]
        return f"cd {shlex.quote(str(REPO))} && {INFER_ENV} " + " ".join(shlex.quote(c) for c in command)

    def check_registry(self):
        """The registry the runs will use: real runs the code tree's configs/bayesreg/segmenters.json, tracked and clean
        (registry_git_state: no staged, unstaged or untracked change); a dry run the synthetic staging registry. It must
        load and register >= 2 folds of the segmenter (all_folds_mean). Returns the SegmenterRegistry."""
        if self.dry:
            path = Path(self.o.segmenter_registry)
        else:
            path = Path(self.code)/"configs"/"bayesreg"/"segmenters.json"
        if not path.is_file():
            raise Stop(f"structure precheck: no segmenter registry {path}")
        git = registry_git_state(path)
        if not self.dry and not (git.get("tracked") is True and git.get("clean") is True):
            raise Stop(f"structure precheck: the registry {path} must be committed and clean (git tracked "
                       f"{git.get('tracked')}, clean {git.get('clean')}, status {git.get('porcelain')!r}); kept, the lead "
                       "decides")
        try:
            registry = load_segmenter_registry(path)
        except ProvenanceRejected as exc:
            raise Stop(f"structure precheck: registry {path}: {exc}") from None
        folds = sorted({e["fold"] for e in registry.models.values() if e["segmenter_id"] == self.o.segmenter_id})
        if len(folds) < 2:
            raise Stop(f"structure precheck: registry {path} has folds {folds} of {self.o.segmenter_id}; all_folds_mean "
                       "needs at least 2 (D24)")
        self.registry_path = path
        self.registry_record = dict(path=str(path), sha256=registry.sha256, folds=folds,
                                    status="staging_unregistered" if self.dry else "registered",
                                    git=dict(tracked=git.get("tracked"), clean=git.get("clean"),
                                             last_commit=git.get("last_commit")))
        self.log(f"structure precheck: registry {path} ({self.registry_record['status']}; sha256 {registry.sha256[:12]}; "
                 f"folds {folds} of {self.o.segmenter_id}; git tracked {git.get('tracked')}, clean {git.get('clean')}"
                 + ("; dry run: synthetic staging registry, passed with --structure-registry-staging" if self.dry
                    else "") + ")")
        return registry

    def structure_precheck(self):
        """Before any unit: the registry and the six subjects' predictions (module docstring, precheck)."""
        if not self.structure:
            self.log(f"structure precheck: channel_source {self.o.channel_source}, no structure layer")
            return
        registry = self.check_registry()
        subjects = sorted({s for p in self.pairs for s in subjects_of(p)})
        missing, bad = [], []
        for subject in subjects:
            directory = self.structure_root/subject
            if not ((directory/PROVENANCE_FILE).is_file() and (directory/GROUPS_FILE).is_file()):
                if directory.exists() and any(directory.iterdir()):
                    bad.append(f"{directory}: no {PROVENANCE_FILE} / {GROUPS_FILE} but not empty (incomplete prediction "
                               "kept; the inference refuses a non-empty directory)")
                else:
                    missing.append(subject)
                continue
            prov = read_json(directory/PROVENANCE_FILE) or {}
            want = dict(segmenter_id=self.o.segmenter_id, subject=subject, grouping=self.o.structure_grouping,
                        policy=INFER_POLICY)
            wrong = [f"{k} {prov.get(k)!r} (expected {v!r})" for k, v in want.items() if prov.get(k) != v]
            unknown = sorted(set(prov.get("model_ids") or []) - set(registry.models))
            if unknown or not prov.get("model_ids"):
                wrong.append(f"model_ids {prov.get('model_ids')} not all in the registry {self.registry_path}")
            if self.dry and DRY_SCRIPT not in json.dumps(prov):
                wrong.append(f"not a {DRY_SCRIPT} prediction (dry run)")
            if wrong:
                bad.append(f"{directory}/{PROVENANCE_FILE}: {'; '.join(wrong)}")
        if bad:
            raise Stop("structure precheck: " + " | ".join(bad) + " (kept; the lead decides)")
        if missing:
            when, k = stamp(), 0
            subjects_file = self.root/f"structure_infer_subjects_{when}.txt"
            while subjects_file.exists():                  # a restart within the same second: a new file, never overwritten
                k += 1
                subjects_file = self.root/f"structure_infer_subjects_{when}_{k}.txt"
            write_once(subjects_file, "".join(subject+"\n" for subject in missing))
            command = self.infer_command(subjects_file)
            self.log(f"structure precheck: inference command (GPU; after the GPU smoke and every timed run; the "
                     f"inference has its own machine gate; {INFER_POLICY} for the T0 subjects, D24): {command}")
            raise Stop(f"structure precheck: no prediction for {len(missing)} of {len(subjects)} subjects under "
                       f"{self.structure_root}: {', '.join(missing)} (ids in {subjects_file}); run the inference first: "
                       f"{command}" + ("  [dry run: synthetic inputs, experiments/eta/dry_inputs.py writes every "
                                       "prediction]" if self.dry else ""))
        self.log(f"structure precheck: predictions of {len(subjects)} subjects present under {self.structure_root} "
                 f"({INFER_POLICY}, {self.o.segmenter_id}, {self.o.structure_grouping})")

    # ---- gate ------------------------------------------------------------------------------------------------------
    def busy_reason(self, memmax_mib, gpu):
        if (self.root/"hold").exists():
            return f"hold file {self.root/'hold'} present"
        if self.v4:                          # ETA4: drivers start units in each other's gaps (the path pilot's rule)
            drivers = self.sys.drivers_running()
            if drivers:
                return (f"driver processes running: {drivers} ({', '.join(DRIVER_SCRIPTS)}; this process and its "
                        "ancestors excepted)")
        if self.sys.unit_active(D0_QUEUE):
            return f"{D0_QUEUE} is active"
        units = self.sys.other_units()
        if units:
            return f"units active: {' '.join(units)} (only {TRAIN_UNIT} may run alongside)"
        if gpu:
            free = self.sys.gpu_free_mib()
            if free is None:
                return "nvidia-smi gave no free-memory reading"
            if free < self.o.min_gpu_free_mib:
                return f"GPU free {free} MiB < {self.o.min_gpu_free_mib} MiB"
        avail, need = self.sys.mem_available_mib(), memmax_mib+MEM_MARGIN_MIB
        if avail < need:
            return (f"MemAvailable {avail} MiB < MemoryMax {memmax_mib} MiB + {MEM_MARGIN_MIB} MiB = {need} MiB "
                    f"({self.sys.train_state()})")
        home = self.sys.home_free_gb()
        if home < self.o.home_floor_gb:
            return f"/home free {home} GB < {self.o.home_floor_gb} GB"
        return ""

    def gate_text(self, label, memmax_text, memmax_mib):
        avail, need = self.sys.mem_available_mib(), memmax_mib+MEM_MARGIN_MIB
        units = self.sys.other_units()
        return (f"gate {label}: MemAvailable {avail} MiB >= MemoryMax {memmax_text} ({memmax_mib} MiB) + {MEM_MARGIN_MIB} "
                f"MiB = {need} MiB: {'yes' if avail >= need else 'NO'}; {self.sys.train_state()}; /home free "
                f"{self.sys.home_free_gb()} GB (floor {self.o.home_floor_gb}); gpu {self.sys.gpu_state()}; other units: "
                f"{' '.join(units) or 'none'}")

    def deadline_reached(self):
        """--no-start-after (ETA4): True when a run unit may no longer start."""
        return self.no_start_after is not None and self.sys.now() >= self.no_start_after

    def gate(self, label, memmax_mib, gpu):
        waited = 0
        while True:
            if self.deadline_run and self.deadline_reached():     # ETA4: run units only, also while waiting
                raise Stop(f"before {label}: --no-start-after {self.no_start_after.isoformat()} reached (waited "
                           f"{waited} min; not run, not a failure; a restart with a later deadline resumes)")
            reason = self.busy_reason(memmax_mib, gpu)
            if not reason:
                return
            if waited >= self.o.wait_minutes:
                raise Stop(f"before {label}: {reason} (waited {waited} min, --wait-minutes {self.o.wait_minutes})")
            if waited % 30 == 0:
                self.log(f"waiting before {label}: {reason}")
            waited += 1
            self.sys.sleep(POLL_SECONDS)

    # ---- one unit --------------------------------------------------------------------------------------------------
    def env_args(self, gpu_visible):
        t = self.threads
        out = [f"--setenv={k}={t}" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                             "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")]
        out += ["--setenv=PYTHONDONTWRITEBYTECODE=1", f"--setenv=NUMBA_CACHE_DIR={self.root/'cache'/'numba'}"]
        if not gpu_visible:
            out.append("--setenv=CUDA_VISIBLE_DEVICES=")
        return out

    def systemd_command(self, *, kind, unit, command, memmax_text, gpu, nice, runtime, quota, wrapped=None, env=None):
        """The systemd-run argument list of one unit (launch; split out by ETA4 so that the tests can compare commands
        without launching; the list is the one launch built before)."""
        sd = ["systemd-run", "--user", f"--unit={unit}", "--wait", "--collect", "--expand-environment=no",
              f"--working-directory={self.code}", f"--property=MemoryMax={memmax_text}", "--property=MemorySwapMax=4G",
              f"--property=CPUQuota={quota}", f"--property=RuntimeMaxSec={runtime}"]
        sd += env if env is not None else self.env_args(gpu)
        if nice is not None:
            sd.append(f"--nice={nice}")
        if wrapped is not None:
            sd += ["/bin/bash", "-c", '"${@:3}" >> "$1" 2>&1; rc=$?; echo $rc > "$2"; exit $rc', f"{kind}-wrapper",
                   str(wrapped[0]), str(wrapped[1])]
        sd += [str(c) for c in command]
        return sd

    def launch(self, *, kind, name, unit, command, memmax_text, memmax_mib, gpu, nice, runtime, quota, wrapped=None,
               env=None):
        """Gate, then one systemd unit with --wait; returns (exit code, launch record). `wrapped` = (log, exit file):
        the command's output goes to the log and its exit code to the exit file (gpu_smoke.sh's evaluation wrapper)."""
        if self.sys.unit_active(unit):
            raise Stop(f"{unit} is active")
        self.deadline_run = kind == "run"
        try:
            self.gate(f"{kind} {name}", memmax_mib, gpu)
        finally:
            self.deadline_run = False
        (self.root/"launch").mkdir(exist_ok=True)
        (self.root/"cache"/"numba").mkdir(parents=True, exist_ok=True)
        record = self.root/"launch"/f"{kind}_{name}_{stamp()}.txt"
        sd = self.systemd_command(kind=kind, unit=unit, command=command, memmax_text=memmax_text, gpu=gpu, nice=nice,
                                  runtime=runtime, quota=quota, wrapped=wrapped, env=env)
        gate = self.gate_text(f"{kind} {name}", memmax_text, memmax_mib)
        with open(record, "x", encoding="utf-8") as stream:
            stream.write(f"{kind}: {name}\nunit: {unit}\ncode: {self.code} (HEAD {self.code_sha})\n"
                         f"configuration: {json.dumps(self.configuration(), ensure_ascii=False)}\nsystemd command:\n"
                         + " ".join(shlex.quote(c) for c in sd)+"\n"+gate+"\n"
                         + f"before {utc()}: uptime {self.sys.uptime()}; loadavg {self.sys.loadavg()}; MemAvailable "
                         f"{self.sys.mem_available_mib()} MiB; /home free {self.sys.home_free_gb()} GB; gpu "
                         f"{self.sys.gpu_state()}\nunits active: {' '.join(self.sys.active_units())}\n")
        self.log(gate)
        self.log(f"{kind} {name}: unit {unit} (record {record})")
        t0 = self.sys.uptime()
        rc = self.sys.run(sd, record)
        if wrapped is not None and Path(wrapped[1]).is_file():
            rc = int(Path(wrapped[1]).read_text().strip() or rc)
        with open(record, "a", encoding="utf-8") as stream:
            stream.write(f"after {utc()}: exit {rc}; uptime {self.sys.uptime()}; MemAvailable "
                         f"{self.sys.mem_available_mib()} MiB; gpu {self.sys.gpu_state()}; {self.sys.train_state()}\n"
                         f"dmesg since start: oom {self.sys.dmesg_since(t0, 'Out of memory|oom-kill|Killed process')}, "
                         f"dxg {self.sys.dmesg_since(t0, 'create_allocation failed')}\n")
        return rc, record

    def earlier(self, kind, name):
        return sorted(glob.glob(str(self.root/"launch"/f"{kind}_{name}_*.txt")))

    # ---- Atropos ---------------------------------------------------------------------------------------------------
    @staticmethod
    def tissue_ok(directory):
        prov = read_json(Path(directory)/"provenance.json")
        return prov is not None and (prov.get("checks") or {}).get("qc_min_class_fraction_ge_0.05") is True

    def atropos_phase(self):
        subjects = sorted({s for p in self.pairs for s in subjects_of(p)})
        done = []
        for subject in subjects:
            directory = self.tissue_root/subject
            if (directory/"provenance.json").is_file():
                if not self.tissue_ok(directory):
                    raise Stop(f"{directory}/provenance.json: QC qc_min_class_fraction_ge_0.05 not true (kept)")
                done.append(subject)
                continue
            if directory.exists():
                raise Stop(f"{directory} exists without provenance.json (incomplete Atropos attempt kept)")
            name = subject
            if self.earlier("atropos", name):
                raise Stop(f"Atropos of {subject} was launched before ({self.earlier('atropos', name)}) and left no "
                           "provenance.json (kept; the lead decides)")
            (self.root/"logs").mkdir(exist_ok=True)
            s = stamp()
            log, exit_file = self.root/"logs"/f"atropos_{name}_{s}.log", self.root/"logs"/f"atropos_{name}_exit_{s}.txt"
            command = [PY, "-B", "experiments/d0/atropos_seg.py", "--stage", STAGE, "--data-root", self.data_root,
                       "--allowlist", self.allowlist, "--out-root", self.tissue_root, "--subjects", subject,
                       "--threads", self.o.atropos_threads]
            env = [f"--setenv=ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={self.o.atropos_threads}",
                   "--setenv=OPENBLAS_NUM_THREADS=1", "--setenv=OMP_NUM_THREADS=1", "--setenv=MKL_NUM_THREADS=1",
                   "--setenv=PYTHONDONTWRITEBYTECODE=1", "--setenv=CUDA_VISIBLE_DEVICES="]
            rc, record = self.launch(kind="atropos", name=name, unit=f"{self.prefix}-atropos-{subject_number(subject)}",
                                     command=command, memmax_text=self.o.atropos_memory_max, memmax_mib=self.atropos_mib,
                                     gpu=False, nice=19, runtime=ATROPOS_MAX_SEC,
                                     quota=f"{100*int(self.o.atropos_threads)}%", wrapped=(log, exit_file), env=env)
            if rc != 0 or not self.tissue_ok(directory):
                raise Stop(f"Atropos of {subject}: exit {rc}, provenance/QC {'ok' if self.tissue_ok(directory) else 'missing'}"
                           f" (see {log}, {record}; kept, nothing rerun)")
            self.log(f"Atropos {subject}: done ({directory})")
        self.log(f"atropos: {len(done)} of {len(subjects)} subjects already present; tissue layer complete")

    # ---- probe -----------------------------------------------------------------------------------------------------
    def probe_phase(self):
        records = {}
        for pair in self.pairs:
            short = short_name(pair)
            directory = self.root/"probe"/short
            unit = f"{self.prefix}-probe-{short}"
            if not (directory/"probe.json").is_file():
                if self.sys.unit_active(unit):
                    raise Stop(f"{unit} is still active")
                status = read_json(directory/"status.json")
                if status is not None:
                    raise Stop(f"probe {short} ended {status.get('status')} without probe.json (kept; the lead decides)")
                if directory.exists():
                    raise Stop(f"{directory} exists without status.json (incomplete probe kept)")
                if self.earlier("probe", short):
                    raise Stop(f"probe {short} was launched before and left no status.json (kept)")
                (self.root/"probe").mkdir(exist_ok=True)
                (self.root/"logs").mkdir(exist_ok=True)
                s = stamp()
                log, exit_file = self.root/"logs"/f"probe_{short}_{s}.log", self.root/"logs"/f"probe_{short}_exit_{s}.txt"
                command = [PY, "-B", "experiments/eta/probe.py", "--pair", f"{STAGE}/{pair}", *self.model_args(),
                           "--out", directory, "--data-root", self.data_root, "--allowlist", self.allowlist,
                           "--tissue-root", self.tissue_root, "--cpu-threads", self.threads]
                rc, record = self.launch(kind="probe", name=short, unit=unit, command=command,
                                         memmax_text=self.o.probe_memory_max, memmax_mib=self.probe_mib, gpu=False,
                                         nice=19 if self.dry else 10, runtime=PROBE_MAX_SEC, quota=self.quota,
                                         wrapped=(log, exit_file))
                if not (directory/"probe.json").is_file():
                    status = read_json(directory/"status.json") or {}
                    raise Stop(f"probe {short}: exit {rc}, status {status.get('status')} ({status.get('reason')}); no "
                               f"probe.json (see {log}, {record})")
            records[short] = read_json(directory/"probe.json")
            self.log(f"probe {short}: nu* = {records[short]['nu_star']:.6g} (u_1 {records[short]['u1_state_hash'][:12]})")
        taus = {json.dumps(r["eta_tmp"]["tau"]) for r in records.values()}
        if len(taus) != 1:
            raise Stop(f"the probes' tau^(0) differ: {sorted(taus)}")
        want = self.probe_configuration()
        for short, record in records.items():
            if not same_probe_configuration(record.get("configuration"), want):
                raise Stop(f"probe {short}: configuration {record.get('configuration')} differs from this sweep's {want} "
                           "(one configuration per --root)")
        values = [records[short_name(p)]["nu_star"] for p in self.pairs]
        center = dict(schema=SCHEMA+".nu_star", contract="C05-7 尺度探针; appendix 7a (pending review)",
                      per_pair={short_name(p): records[short_name(p)]["nu_star"] for p in self.pairs},
                      alpha_star_per_pair={short_name(p): records[short_name(p)]["alpha_star"] for p in self.pairs},
                      combination="geometric mean of the per-pair nu* (C05-7 appendix 7a)",
                      nu_star=geometric_mean(values), spread_max_over_min=max(values)/min(values),
                      tau0=records[short_name(self.pairs[0])]["eta_tmp"]["tau"],
                      nu_tmp={short_name(p): records[short_name(p)]["eta_tmp"]["nu"] for p in self.pairs},
                      model_channels={short_name(p): records[short_name(p)].get("model_channels") for p in self.pairs},
                      configuration=self.configuration())
        self.compare_or_write(self.root/"probe"/"nu_star.json", center)
        self.log(f"nu* centre {center['nu_star']:.6g} (geometric mean; per pair {center['per_pair']}; spread "
                 f"{center['spread_max_over_min']:.3g}x); tau^(0) {center['tau0']}")
        return center

    def compare_or_write(self, path, document, ignore=("written_utc",)):
        """Write once; on a restart the recomputed document must equal the stored one (else STOP)."""
        stored = read_json(path)
        if stored is None:
            write_once(path, dict(document, written_utc=utc()))
            return document
        strip = (lambda d: {k: v for k, v in d.items() if k not in ignore})
        if json.loads(json.dumps(strip(document))) != strip(stored):
            raise Stop(f"{path} differs from the recomputed record (kept; the lead decides)")
        return stored

    # ---- v4: path, centre, preflight (ETA4) ---------------------------------------------------------------------------
    def centre_eta(self):
        """(alpha, nu, tau) of --centre-eta, checked (positive finite; four tau; alpha = alpha_nug(tau))."""
        doc = read_json(self.o.centre_eta) if Path(self.o.centre_eta).is_file() else None
        if not isinstance(doc, dict):
            raise Stop(f"--centre-eta {self.o.centre_eta}: missing or not a JSON object")
        value = dict(alpha=doc.get("alpha"), nu=doc.get("nu"), tau=doc.get("tau"))
        good = (finite(value["alpha"]) and value["alpha"] > 0 and finite(value["nu"]) and value["nu"] > 0
                and isinstance(value["tau"], list) and len(value["tau"]) == 4
                and all(finite(t) and t > 0 for t in value["tau"]))
        if not good:
            raise Stop(f"--centre-eta {self.o.centre_eta}: eta {value} (alpha, nu > 0 finite; four positive tau)")
        if value["alpha"] != alpha_nug(value["tau"]):
            raise Stop(f"--centre-eta {self.o.centre_eta}: alpha {value['alpha']!r} is not alpha_nug(tau) = "
                       f"{alpha_nug(value['tau'])!r} (P2, C05-6)")
        return doc, value

    def resolve_path(self):
        """v4: the solver path (module docstring): --path-from-pilot (pilot_path) and / or --path with --path-note; sets
        the cell estimate and the per-level caps; path.json written once (a restart recomputes and compares). A pilot
        record that is not "resolved" without an explicit --path writes path_hold_<UTC>.json and STOPs (nothing
        started; the current path is what stays, the lead decides)."""
        o = self.o
        record = None
        if o.path_from_pilot:
            eta = None
            if o.centre_eta and Path(o.centre_eta).is_file():
                eta = self.centre_eta()[1]
            # the pilot must have ended (contract 7b; ETA4 review M3): no tag-pilot* unit, no run_pilot.py process
            # (its queue.log is read by pilot_path)
            units, drivers = self.sys.pilot_state()
            # the pilot's own summarize.py (arm_flags, recommend); not loadable -> hold (contract 7b; review M1); its
            # sha256 and that of the copy in the pilot's code worktree are recorded, a difference holds
            ours = REPO/PILOT_SUMMARIZE
            theirs = Path(o.path_from_pilot).parent/"code"/PILOT_SUMMARIZE
            tool = dict(path=str(ours), sha256=file_sha256(ours) if ours.is_file() else None,
                        pilot_code_copy=str(theirs) if theirs.is_file() else None,
                        pilot_code_copy_sha256=file_sha256(theirs) if theirs.is_file() else None)
            try:
                summarize = load_module("_eta_pilot_summarize", PILOT_SUMMARIZE)
            except Exception as exc:       # noqa: BLE001 -- recorded; pilot_path holds without it
                summarize = None
                tool["load_error"] = f"{type(exc).__name__}: {exc}"
            record = pilot_path(o.path_from_pilot, eta=eta, dry=self.dry, summarize=summarize)
            record["summarize_tool"] = tool
            extra_holds = []
            if units or drivers:
                extra_holds.append(f"PILOT 仍在运行（单元 {units}，驱动进程 {drivers}）")
            if tool["pilot_code_copy_sha256"] is not None and tool["pilot_code_copy_sha256"] != tool["sha256"]:
                extra_holds.append(f"重算所用的 {ours} 与 PILOT 代码 worktree 中的 {theirs} 不同")
            if extra_holds:
                record.update(status="hold", path="current")
                record["reasons"] = extra_holds+record["reasons"]
            self.log(f"path: pilot summary {o.path_from_pilot} -> {record['status']} {record['path']} "
                     f"(pilot recommendation {record['pilot_recommendation']!r}; pilot counts "
                     f"{record.get('pilot_counts')}; {'; '.join(record['reasons']) or 'ok'})")
        if o.path:
            name, source = o.path, "explicit (--path)" if record is None else "override (--path over --path-from-pilot)"
        elif record is not None and record["status"] == "resolved":
            name, source = record["path"], "pilot (--path-from-pilot)"
        else:
            hold = dict(schema=SCHEMA+".path_hold", written_utc=utc(), pilot=record,
                        rule="取不到路径或规则交总负责人时：维持现行路径并交总负责人，不自动开跑（plan.md）；"
                             "总负责人用 --path NAME --path-note TEXT 显式给出后再启动")
            when, k = stamp(), 0
            target = self.root/f"path_hold_{when}.json"
            while target.exists():
                k += 1
                target = self.root/f"path_hold_{when}_{k}.json"
            write_once(target, hold)
            raise Stop(f"path: {record['status'] if record else 'none'} -- {'; '.join(record['reasons']) if record else ''}"
                       f" (record {target}); the current path stays, nothing started; the lead gives --path explicitly")
        spec = PATHS[name]
        conflicts = extra_conflicts(o.extra, spec["caps"])
        if conflicts:
            raise Stop(f"path {name}: --extra names {conflicts}, which the path's switches exclude (C13-4a)")
        o.jd_cell_estimate = spec["cell"]
        self.level_caps = None if spec["caps"] is None else list(spec["caps"])
        self.path_name = name
        document = dict(schema=SCHEMA+".path", path=name, source=source, note=o.path_note,
                        switches=dict(jd_cell_estimate=spec["cell"],
                                      max_accepted_per_level=None if spec["caps"] is None else list(spec["caps"])),
                        run_pair_switches=self.path_switches(), pilot=record,
                        pilot_counts=None if record is None else record.get("pilot_counts"),
                        rule=("路径只改求解路径（C13-4a / D51、C09.1-3a / D53），不改 F_B、证书、接受与停止阈值；"
                              "来源见 source；PILOT 的建议规则见 runs/20261031-path-pilot/plan.md 判定一节"))
        self.compare_or_write(self.root/"path.json", document)
        self.log(f"path: {name} ({source}): run_pair switches {self.path_switches() or 'none (current path)'}")

    def path_switches(self):
        """The run_pair switches of the path beyond the v3 command (the cell estimate is always passed)."""
        out = []
        if self.o.jd_cell_estimate != JD_DEFAULT:
            out += ["--jd-cell-estimate", self.o.jd_cell_estimate]
        if self.level_caps is not None:
            out += ["--max-accepted-per-level", ",".join(str(a) for a in self.level_caps)]
        return out

    def code_difference(self, other_sha):
        """Model files (bayesreg/, jd_curl/, configs/bayesreg/) changed between `other_sha` and this code tree."""
        if other_sha == self.code_sha:
            return []
        out = self.git("diff", "--name-only", str(other_sha), self.code_sha, "--", *MODEL_PATHS, cwd=self.code)
        if out.returncode:
            raise Stop(f"cannot compare the model code of {other_sha} and {self.code_sha}: {out.stderr.strip()}")
        return [line for line in out.stdout.splitlines() if line.strip()]

    def model_code_gate(self, changed, centre_sha, args_check=None):
        """Contract 7b scope (ETA4 review round 2 R2-7): model files changed since the centre's code must be accepted
        by the lead after seeing them. Changed and not accepted (--accept-code-change absent): the list goes to a new
        centre/model_code_changed_pending_<UTC>.json (with the run_pair args check) and the sweep STOPs before
        centre.json; the lead reads it and reruns with --accept-code-change. Returns whether a change was accepted."""
        if not changed:
            return False
        if self.o.accept_code_change:
            return True
        (self.root/"centre").mkdir(exist_ok=True)
        when, k = stamp(), 0
        target = self.root/"centre"/f"model_code_changed_pending_{when}.json"
        while target.exists():
            k += 1
            target = self.root/"centre"/f"model_code_changed_pending_{when}_{k}.json"
        write_once(target, dict(schema=SCHEMA+".model_code_pending", written_utc=utc(), centre_code_sha=centre_sha,
                                code_sha=self.code_sha, model_paths=list(MODEL_PATHS), model_code_changed=changed,
                                run_pair_args_check=args_check,
                                rule="总负责人核对此清单后，带 --accept-code-change 重跑 --until centre，再进行第 3 步"
                                     "（合同 C05-7 第 7b 条适用范围；plan.md 启动第 2 步）"))
        raise Stop(f"the model code differs from the centre's ({centre_sha}) in {len(changed)} files, listed in {target}; "
                   "the lead reads the list, then reruns with --accept-code-change (contract 7b; plan.md)")

    def centre_phase(self):
        """v4: the centre from the v3 choice (module docstring); centre/centre.json written once."""
        path = Path(self.o.centre_eta)
        doc, value = self.centre_eta()
        sha = file_sha256(path)
        if not self.dry and sha != V3_ETA_SHA256:
            raise Stop(f"--centre-eta {path}: sha256 {sha} is not the pinned {V3_ETA_SHA256} (the v3 choice)")
        spath = path.parent/"summary.json"
        summary = read_json(spath)
        if not isinstance(summary, dict):
            raise Stop(f"{spath} missing (the summary of the sweep that chose the centre)")
        problems = []
        if summary.get("eta_selected") != value:
            problems.append(f"eta_selected {summary.get('eta_selected')} (file {value})")
        if summary.get("stage_set") != STAGE:
            problems.append(f"stage_set {summary.get('stage_set')!r}")
        if list(summary.get("pairs") or []) != list(self.pairs):
            problems.append(f"pairs {summary.get('pairs')} (ours {self.pairs})")
        if problems:
            raise Stop(f"{spath} does not fit the centre: {'; '.join(problems)}")
        differ = config_differences(doc.get("configuration"), self.configuration())
        if differ:
            raise Stop(f"the centre's configuration ({path}) differs from this root's beyond the path keys "
                       f"{list(PATH_KEYS)}: {'; '.join(differ)}")
        args_check = self.preflight(path.parent)
        if self.dry:
            changed, accepted = None, None
        else:
            changed = self.code_difference(doc.get("code_sha"))
            accepted = self.model_code_gate(changed, doc.get("code_sha"), args_check)
        tie = dict(tie_tol=TIE_TOL, sdlogj_official_max=self.o.sdlogj_official_max,
                   sdlogj_contract_value=SDLOGJ_V4, sdlogj_note=self.o.sdlogj_note)
        centre = dict(schema=SCHEMA+".centre", design="v4",
                      contract="C05-7 第 7b 条与 D56；v4 不跑探针，中心取 v3 选中的 η（ETA4 plan.md）",
                      nu_star=value["nu"], tau0=list(value["tau"]), alpha=value["alpha"],
                      source=str(path), source_sha256=sha, pinned_sha256=None if self.dry else V3_ETA_SHA256,
                      summary=str(spath), summary_sha256=file_sha256(spath), centre_code_sha=doc.get("code_sha"),
                      centre_configuration=doc.get("configuration"), path_keys_allowed_to_differ=list(PATH_KEYS),
                      configuration_check="equal apart from the path keys", run_pair_args_check=args_check,
                      model_code_changed=changed, accept_code_change=accepted,
                      grid={s: [[label, f] for label, f in self.labels[s]] for s in STAGES},
                      reused=REUSED, selection=tie, probe="not run (the centre is known; C05-7a's probe is v3's)",
                      configuration=self.configuration())
        (self.root/"centre").mkdir(exist_ok=True)
        self.compare_or_write(self.root/"centre"/"centre.json", centre)
        self.log(f"centre: nu {value['nu']!r}, tau^(0) {value['tau']}, alpha {value['alpha']!r} from {path} (sha256 "
                 f"{sha[:12]}); model code changed since {doc.get('code_sha')}: "
                 f"{'not checked (dry run)' if changed is None else len(changed)} files; SDlogJ condition "
                 f"{self.o.sdlogj_official_max if self.o.sdlogj_official_max is not None else 'none'}")
        return centre

    def preflight(self, v3_root):
        """v4: the first run's argument list through run_pair's own parse_args and args_record in the code tree (C13-4a:
        a conflict is refused here, before any unit), and the resulting config.json args against those of the v3 run
        of the same pair (args_differences; contract 7b, ETA4 review M7: only the path's keys, paths inside the code
        trees and SWLI's brain / null may differ). Returns the record kept in centre.json."""
        args = ["--out", str(self.root/"preflight_out_never_created"),
                *self.run_args(self.pairs[0], self.root/"eta"/"preflight_eta_never_read.json")]
        ok, message, ours = self.sys.check_run_args(self.code, args)
        if not ok:
            raise Stop(f"run_pair parse_args refused the run arguments ({' '.join(map(str, args))}): {message}")
        found = sorted(glob.glob(str(Path(v3_root)/"runs"/"*"/"*"/short_name(self.pairs[0])/"config.json")))
        if not found:
            raise Stop(f"no v3 run config.json of {short_name(self.pairs[0])} under {v3_root}/runs (needed for the "
                       "run_pair args comparison, contract 7b)")
        v3_args = (read_json(found[0]) or {}).get("args")
        if not isinstance(v3_args, dict):
            raise Stop(f"{found[0]}: no args")
        # the v3 run's code tree: the first word of its cmd.txt is <code tree>/bayesreg/cli/run_pair.py
        words = (Path(found[0]).parent/"cmd.txt").read_text(encoding="utf-8").split() \
            if (Path(found[0]).parent/"cmd.txt").is_file() else []
        v3_code = Path(words[0]).parents[2] if words and words[0].endswith("bayesreg/cli/run_pair.py") else None
        unexpected, allowed = args_differences(v3_args, ours, v3_code=v3_code, our_code=self.code,
                                               expected_path=self.path_expected_args())
        if unexpected:
            raise Stop(f"run_pair args differ from the v3 run {found[0]} beyond the path, the code trees and SWLI's "
                       f"brain / null: {json.dumps(unexpected, ensure_ascii=False)}")
        self.log(f"preflight: run_pair parse_args accepts the run arguments (path switches "
                 f"{self.path_switches() or 'none'}); args against the v3 run {found[0]}: differences allowed "
                 f"{sorted(allowed)}, none other")
        return dict(v3_run_config=found[0], v3_run_config_sha256=file_sha256(found[0]),
                    v3_code=None if v3_code is None else str(v3_code), allowed_differences=allowed, unexpected={},
                    rule="per-run keys ignored; path keys equal to the path's; code-tree paths equal relative to each "
                         "tree; sigma_domain brain / support_threshold null where v3 had none; every other key equal")

    def bernstein_gate(self, name, run, ev=None):
        """certificate_bernstein (D53, contract 7b "审核停止"): a run's check-(b) count must be 0; else (> 0: 待审核, not
        recorded: 待核对) its file audit/<name>.json must release it. Two kinds (ETA4 review round 2):
          - an audit (audit_problems): bound to this run's exported state_hash, the evaluator certificate's state_hash,
            the current count, the run's end (run_end_utc) and the recertification record it names
            (recertification_record: a path, relative to audit/ unless absolute; its sha256 recorded);
          - a no-export confirmation (no_export_problems; "no_export" in the file): only for a run without an exported
            state; the run then counts as ineligible ("不合格（无导出）") and the sweep goes on.
        Returns dict(count, audit, audit_sha256, recertification_record, recertification_sha256, no_export) (None
        without certificate_bernstein)."""
        if self.o.jd_cell_estimate != CERTIFICATE_BERNSTEIN:
            return None
        count = bernstein_count(run)
        base = dict(count=count, audit=None, audit_sha256=None, recertification_record=None,
                    recertification_sha256=None, no_export=False, code_dirty_note=None)
        if count == 0:
            return base
        target = self.root/"audit"/f"{name}.json"
        status_doc = read_json(Path(run)/"status.json") or {}
        exported = status_doc.get("exported")
        eval_hash = eval_grid = None
        if ev is not None:
            found = sorted(glob.glob(str(Path(ev)/"*"/"metrics.json")))
            if found:
                cert = (read_json(found[0]) or {}).get("certificate") or {}
                eval_hash, eval_grid = cert.get("state_hash"), cert.get("grid_id")
        # RECERT review I1: the recertification is bound to this run's directory, its exported file (u, grid and
        # status as one file; the same rule as experiments/audit/recertify.py exported_file) and the evaluator's grid
        export = status_doc.get("export") if isinstance(status_doc.get("export"), dict) else {}
        level = export.get("level")
        export_path = Path(run)/(f"final_level{level}.npz" if isinstance(level, int) and not isinstance(level, bool)
                                 else "final_native.npz")
        binding = dict(run_dir=run, export_sha256=file_sha256(export_path) if export_path.is_file() else "<missing>",
                       eval_grid_id=eval_grid)
        read_at = datetime.now(timezone.utc)          # round 3 R3-3: the audit's time may not lie beyond this + 10 min
        try:
            audit = read_json(target)
        except ValueError:
            audit = None
        run_end = self.run_end_utc(name, run, ev)
        state = "待核对（计数未记录）" if count is None else f"待审核（违反 {count}）"
        if isinstance(audit, dict) and "no_export" in audit:
            problems = no_export_problems(audit, name=name, state_hash=exported, run_end=run_end, now=read_at)
            if not problems:
                sha = file_sha256(target)
                self.log(f"{name}: certificate_bernstein check (b) {state}; no exported state; confirmation {target} "
                         f"(sha256 {sha[:12]}) by {audit['auditor']} at {audit['utc']}: the run counts as ineligible "
                         "(不合格（无导出）), the sweep goes on")
                return dict(base, audit=str(target), audit_sha256=sha, no_export=True)
            kind = "no-export confirmation"
        else:
            record_path, record, record_sha = None, None, None
            if isinstance(audit, dict) and isinstance(audit.get("recertification_record"), str):
                record_path = Path(audit["recertification_record"])
                if not record_path.is_absolute():
                    record_path = target.parent/record_path
                try:
                    record = read_json(record_path)
                except ValueError:
                    record = None
                record_sha = file_sha256(record_path) if record_path.is_file() else None
            problems = audit_problems(audit, state_hash=exported, eval_state_hash=eval_hash, count=count,
                                      run_end=run_end, record=record, now=read_at, binding=binding)
            if not problems:
                sha = file_sha256(target)
                self.log(f"{name}: certificate_bernstein check (b) {state}; audit {target} (sha256 {sha[:12]}) released "
                         f"it: recertified with {audit['certificate_config']} {RECERT_FIELDS}, bound "
                         f"{audit['min_jd_lower_bound']!r}, record {record_path} (sha256 {record_sha[:12]}), by "
                         f"{audit['auditor']} at {audit['utc']}"
                         + (f"; code_dirty_note {audit['code_dirty_note']!r}" if audit.get("code_dirty_note") else ""))
                return dict(base, audit=str(target), audit_sha256=sha, recertification_record=str(record_path),
                            recertification_sha256=record_sha, code_dirty_note=audit.get("code_dirty_note"))
            kind = "audit file"
        raise Stop(f"{name}: certificate_bernstein check (b) {state} (D53: report to the lead on its own; the final state "
                   f"recertified without the prescreen; its numbers enter no conclusion before). Evidence kept. The {kind} "
                   f"{target} is not complete: {'; '.join(problems)} (audit keys {list(AUDIT_KEYS)}; no-export keys "
                   f"{list(NO_EXPORT_KEYS)}; contract 7b)")

    def run_end_utc(self, name, run, ev):
        """The run's end for the audit's utc (ETA4 review round 2 R2-3): its evaluation's done.json done_utc (the
        evaluation follows the run), else the last "after <UTC>:" line of its launch records, else the mtime of its
        status.json."""
        done = read_json(Path(ev)/"done.json") if ev is not None else None
        moment = parse_utc((done or {}).get("done_utc"))
        if moment is not None:
            return moment
        for record in reversed(self.earlier("run", name)):
            for line in reversed(Path(record).read_text(encoding="utf-8").splitlines()):
                if line.startswith("after "):            # "after 2026-10-31T15:05:50Z: exit 0; ..." (launch)
                    moment = parse_utc(line[len("after "):len("after ")+20])
                    if moment is not None:
                        return moment
        status = Path(run)/"status.json"
        return datetime.fromtimestamp(status.stat().st_mtime, timezone.utc) if status.is_file() else None

    def hard_constraint_1(self, name, ev):
        """v4 (contract 7b, ETA4 review I2; experiments/m2v1/run_batch.py hard_constraint_1): an evaluated field
        (field_present) whose evaluator certificate did not pass (zero or undecided) STOPs the sweep (run_pair exports
        certified states only, so this would be a defect). Checked after the evaluation, whenever a restart meets
        done.json, and for every row of a selection."""
        if not self.v4:
            return
        found = sorted(glob.glob(str(Path(ev)/"*"/"metrics.json")))
        if len(found) != 1:
            raise Stop(f"{name}: {len(found)} metrics.json under {ev} (one expected); kept, the lead decides")
        doc = read_json(found[0]) or {}
        cert = doc.get("certificate") or {}
        if doc.get("field_present") and cert.get("passed") is not True:
            raise Stop(f"{name}: the evaluator's certificate of the exported field did not pass ({cert.get('status')}, "
                       f"{cert.get('min_jd_lower_bound')}): hard constraint 1; kept, the lead decides")

    # ---- sweep runs ------------------------------------------------------------------------------------------------
    def run_args(self, pair, eta_file):
        args = ["--pair", f"{STAGE}/{pair}", *self.model_args()]
        if self.structure:
            args += ["--structure-infer", "never"]          # predictions are made before the sweep (precheck)
        args += ["--backend", self.o.backend]
        if self.o.backend == "torch64":
            args += ["--device", self.device]
        args += ["--outer", self.o.outer, "--rounds", self.o.rounds, "--wall-seconds", number(self.o.wall_seconds),
                 "--cpu-threads", str(self.threads), "--data-root", str(self.data_root), "--allowlist", self.allowlist,
                 "--tissue-root", str(self.tissue_root), "--eta-json", str(eta_file)]
        if self.level_caps is not None:     # ETA4: run_pair only (C13-4a); nothing without caps (the v3 command)
            args += ["--max-accepted-per-level", ",".join(str(a) for a in self.level_caps)]
        return args+shlex.split(self.o.extra)

    def one_run(self, stage, label, pair, eta_file):
        short = short_name(pair)
        name = f"{stage}-{label}-{short}"
        run, ev = self.root/"runs"/stage/label/short, self.root/"eval"/stage/label/short
        unit = f"{self.prefix}-{name}"
        if (ev/"done.json").is_file():
            self.log(f"skip {name} (evaluated)")
            self.hard_constraint_1(name, ev)          # ETA4 (v4): re-checked on every restart
            self.bernstein_gate(name, run, ev)        # ETA4 / D53: a restart before the audit stops here again
            return
        if self.sys.unit_active(unit):
            raise Stop(f"{unit} is still active")
        if (run/"status.json").is_file():
            self.log(f"resume: {run} has status.json; evaluating")
            self.evaluate(stage, label, short, run, ev)
            self.hard_constraint_1(name, ev)
            self.bernstein_gate(name, run, ev)
            return
        if run.exists():
            raise Stop(f"{run} exists without status.json (incomplete run kept whole; the lead decides)")
        if self.earlier("run", name):
            raise Stop(f"{name} was launched before ({self.earlier('run', name)}) and has no status.json (kept whole)")
        run.parent.mkdir(parents=True, exist_ok=True)
        cuda = self.o.backend == "torch64" and self.device == "cuda"
        command = [PY, "-B", "-m", "bayesreg.cli.run_pair", "--out", run, *self.run_args(pair, eta_file)]
        rc, record = self.launch(kind="run", name=name, unit=unit, command=command, memmax_text=self.o.memory_max,
                                 memmax_mib=self.memmax_mib, gpu=cuda, nice=19 if self.dry else None,
                                 runtime=int(self.o.wall_seconds)+1800, quota=self.quota)
        status = read_json(run/"status.json")
        if status is None:
            raise Stop(f"{name} ended (systemd-run exit {rc}) without {run}/status.json (killed? see {record})")
        self.log(f"ended {name}: systemd-run exit {rc}; {status.get('status')} {status.get('wording')} exit "
                 f"{status.get('exit_code')}")
        self.evaluate(stage, label, short, run, ev)
        self.hard_constraint_1(name, ev)
        self.bernstein_gate(name, run, ev)

    def evaluate(self, stage, label, short, run, ev):
        name = f"{stage}-{label}-{short}"
        earlier = sorted(glob.glob(str(ev/"eval_*.log")))+sorted(glob.glob(str(ev/"long.csv")))+self.earlier("eval", name)
        if earlier:
            raise Stop(f"evaluation of {name}: earlier attempt without done.json ({' '.join(earlier)}); kept, not rerun")
        ev.mkdir(parents=True, exist_ok=True)
        s = stamp()
        log, exit_file = ev/f"eval_{s}.log", ev/f"eval_exit_{s}.txt"
        rid = f"eta_{stage}_{label}_{short}"
        command = [PY, "-B", "-m", "bayesreg.eval.cli", run, "--pairs-stage", STAGE, "--out", "metrics.json", "--long",
                   ev/"long.csv", "--summary", ev/"summary.json", "--out-dir", ev, "--method", METHOD, "--arm",
                   f"{stage}-{label}", "--run-id", rid, "--data-root", self.data_root, "--allowlist", self.allowlist]
        rc, record = self.launch(kind="eval", name=name, unit=f"{self.prefix}-eval-{name}", command=command,
                                 memmax_text=self.o.eval_memory_max, memmax_mib=self.eval_mib, gpu=False,
                                 nice=19 if self.dry else 10, runtime=EVAL_MAX_SEC, quota=self.quota,
                                 wrapped=(log, exit_file))
        if rc != 0:
            raise Stop(f"evaluation of {name} exit {rc} (see {log}, {record}; kept whole, nothing rerun)")
        metrics = sorted(glob.glob(str(ev/"*"/"metrics.json")))
        write_once(ev/"done.json", dict(done_utc=utc(), run=str(run), run_id=rid, metrics=metrics, log=str(log),
                                        code_sha=self.code_sha))
        text = Path(log).read_text(encoding="utf-8") if Path(log).is_file() else ""
        tail = (text.strip().splitlines() or [""])[-1][:260]
        self.log(f"evaluated {name}: {tail}")

    def pair_row(self, stage, label, pair):
        short = short_name(pair)
        run, ev = self.root/"runs"/stage/label/short, self.root/"eval"/stage/label/short
        status, config, hyper = read_json(run/"status.json") or {}, read_json(run/"config.json") or {}, \
            read_json(run/"hyper.json") or {}
        found = sorted(glob.glob(str(ev/"*"/"metrics.json")))
        if not (ev/"done.json").is_file() or len(found) != 1:
            raise Stop(f"{stage}-{label}-{short}: evaluation missing ({len(found)} metrics.json, done.json "
                       f"{(ev/'done.json').is_file()})")
        record = read_json(found[0])
        m, c = record.get("metrics") or {}, record.get("certificate") or {}
        levels = [dict(level=r.get("level"), status=r.get("status"), accepted=r.get("accepted"))
                  for r in hyper.get("rounds") or []]
        # the run's own record of its configuration (run_pair config.json args; absent when the run ended before it,
        # e.g. provenance_rejected): every value this sweep implies must be there unchanged
        args = config.get("args")
        if isinstance(args, dict):
            expected = self.expected_run_args()
            differ = {k: (args.get(k), v) for k, v in expected.items() if args.get(k) != v}
            if differ:
                raise Stop(f"{stage}-{label}-{short}: run_pair config.json args differ from this sweep's configuration "
                           f"(got, expected): {differ}")
            config_check = "config.json args = the sweep's configuration"
            # ETA4 review M6 (v4): the candidate's eta, and structure_repr sqrt_prob (contract 7b scope; the key is
            # absent while SDF is not implemented)
            if self.v4:
                eta_doc = read_json(self.root/"eta"/f"{stage}_{label}.json") or {}
                want = {k: eta_doc.get(k) for k in ("alpha", "nu", "tau")}
                got = {k: (config.get("eta") or {}).get(k) for k in ("alpha", "nu", "tau")}
                if got != want:
                    raise Stop(f"{stage}-{label}-{short}: config.json eta {got} is not the candidate's {want}")
                if args.get("structure_repr", "sqrt_prob") != "sqrt_prob":
                    raise Stop(f"{stage}-{label}-{short}: structure_repr {args.get('structure_repr')!r} (contract 7b: "
                               "sqrt_prob only)")
                config_check += "; eta = the candidate's; structure_repr sqrt_prob (absent = sqrt_prob)"
        else:
            config_check = "no config.json args (run ended before writing them)"
        if self.v4:
            self.hard_constraint_1(f"{stage}-{label}-{short}", ev)
        row = dict(pair=short, run_dir=str(run), eval_metrics=found[0], run_id=record.get("run_id"),
                   run_status=status.get("status"), wording=status.get("wording"), exit_code=status.get("exit_code"),
                   field_role=record.get("field_role"), eval_error=record.get("error"),
                   metrics={k: (m.get(k) if finite(m.get(k)) else None) for k, _ in MEASURES},
                   certificate=dict(mode=c.get("mode"), status=c.get("status"), passed=c.get("passed"),
                                    min_jd_lower_bound=c.get("min_jd_lower_bound")) if c else None,
                   eta=(config.get("eta") or {}), levels=levels,
                   wall_seconds=(status.get("wall_budget") or {}).get("seconds"),
                   structure_registry=status.get("structure_registry"), configuration_check=config_check)
        if self.o.jd_cell_estimate == CERTIFICATE_BERNSTEIN:      # ETA4 / D53: a count > 0 needs the lead's audit
            gate = self.bernstein_gate(f"{stage}-{label}-{short}", run, ev)
            row.update(bernstein_violations=gate["count"], bernstein_audit=gate["audit"],
                       bernstein_audit_sha256=gate["audit_sha256"],
                       bernstein_recertification=gate["recertification_record"],
                       bernstein_recertification_sha256=gate["recertification_sha256"],
                       bernstein_no_export=gate["no_export"], bernstein_code_dirty_note=gate["code_dirty_note"])
        return row

    def candidate(self, stage, label, factor, center, tau0, previous):
        reused = None
        if REUSED.get(stage) == label:
            reused = previous
        if reused is not None:
            prev_stage, prev = reused["stage"], reused["chosen_candidate"]
            eta_doc = read_json(prev["eta_file"])
            if eta_doc is None or eta_doc["nu"] != prev["nu"] or eta_doc["s_tau"] != 1.:
                raise Stop(f"{stage}-{label}: the reused eta {prev['eta_file']} does not match the {prev_stage} choice")
            return dict(label=label, factor=factor, nu=eta_doc["nu"], s_tau=eta_doc["s_tau"], eta_file=prev["eta_file"],
                        reused_from=prev.get("reused_from") or f"{prev_stage}-{prev['label']}", rows=prev["rows"],
                        alpha=eta_doc["alpha"], tau=eta_doc["tau"])
        eta_doc = candidate_eta(stage, label, factor, nu_center=center, tau0=tau0)
        (self.root/"eta").mkdir(exist_ok=True)
        eta_file = self.root/"eta"/f"{stage}_{label}.json"
        self.compare_or_write(eta_file, eta_doc)
        return dict(label=label, factor=factor, nu=eta_doc["nu"], s_tau=eta_doc["s_tau"], eta_file=str(eta_file),
                    reused_from=None, rows=None, alpha=eta_doc["alpha"], tau=eta_doc["tau"])

    def stage(self, stage, center, tau0, previous):
        candidates = [self.candidate(stage, label, factor, center, tau0, previous)
                      for label, factor in self.labels[stage]]
        self.log(f"stage {stage}: centre nu {center:.6g}; candidates "
                 + ", ".join(f"{c['label']} nu {c['nu']:.6g} s_tau {c['s_tau']:g}"
                             + (f" (reused {c['reused_from']})" if c["reused_from"] else "") for c in candidates))
        for c in candidates:
            if c["reused_from"]:
                continue
            for pair in self.pairs:
                self.one_run(stage, c["label"], pair, c["eta_file"])
        cap = getattr(self.o, "sdlogj_official_max", None)
        for c in candidates:
            if not c["reused_from"]:
                c["rows"] = [self.pair_row(stage, c["label"], p) for p in self.pairs]
            c["eligible"], c["ineligible"] = eligibility(c["rows"])
            c["mean_seg35"], c["mean_seg4"] = mean_of(c["rows"], "seg35"), mean_of(c["rows"], "seg4")
            if self.v4:                     # ETA4: the official SDlogJ mean, and the eligibility condition
                c["mean_sdlogj_official"] = mean_of(c["rows"], "sdlogj_official")
                c["mean_sdlogj_brain"] = mean_of(c["rows"], "sdlogj_brain")
                c["excluded_by_sdlogj"] = False
                # contract 7b (round 2 R2-4): the three official SDlogJ all finite and their mean <= the condition
                odd = [r["pair"] for r in c["rows"] if not finite((r.get("metrics") or {}).get("sdlogj_official"))]
                if cap is not None and c["eligible"] and (odd or c["mean_sdlogj_official"] > cap):
                    c["eligible"], c["excluded_by_sdlogj"] = False, True
                    c["ineligible"] = c["ineligible"]+[
                        f"official SDlogJ not finite on {odd} (v4 eligibility condition)" if odd else
                        f"mean official SDlogJ {fmt(c['mean_sdlogj_official'])} > {cap:g} (v4 eligibility condition)"]
                # round 2 R2-5: a run released by a no-export confirmation counts as ineligible
                confirmed = [r["pair"] for r in c["rows"] if r.get("bernstein_no_export")]
                if confirmed:
                    c["eligible"] = False
                    c["ineligible"] = c["ineligible"]+[f"{confirmed}: 不合格（无导出）, check (b) confirmed without an "
                                                       "exported state"]
        chosen, tied, reason = choose(candidates)
        rule = RULE.format(roles="/".join(COMPARABLE_ROLES), tol=TIE_TOL)
        if self.v4 and cap is not None:
            rule += RULE_SDLOGJ.format(cap=cap)
        selection = dict(schema=SCHEMA+".selection", stage=stage, nu_center=center, tau0=list(tau0),
                         note=NOTE.format(wall=self.o.wall_seconds), configuration=self.configuration(),
                         rule=rule, tie_tol=TIE_TOL,
                         candidates=candidates, chosen=None if chosen is None else chosen["label"],
                         chosen_candidate=chosen, tied=tied, reason=reason)
        if self.v4:
            selection.update(design="v4", sdlogj_official_max=cap,
                             excluded_by_sdlogj=[c["label"] for c in candidates if c["excluded_by_sdlogj"]],
                             edge=None if chosen is None else edge_of(candidates, chosen["label"],
                                                                      "s_tau" if stage == "s3" else "nu"))
        selection = self.compare_or_write(self.root/f"selection_{stage}.json", selection)
        if self.v4:
            self.write_readouts(stage, selection)
        md = self.root/f"selection_{stage}.md"
        if not md.exists():
            write_once(md, stage_markdown(selection, self.o.wall_seconds))
        if chosen is None:
            raise Stop(f"stage {stage}: no eligible candidate ({'; '.join(c['label']+': '+', '.join(c['ineligible']) for c in candidates)}); see {md}")
        self.log(f"stage {stage}: chose {selection['chosen']} (nu {chosen['nu']:.6g}, s_tau {chosen['s_tau']:g}; "
                 f"mean seg35 {chosen['mean_seg35']:.4f}, seg4 {fmt(chosen['mean_seg4'])}; {reason}); table {md}")
        return selection

    def write_readouts(self, stage, selection):
        """v4: readouts_<stage>.json / .md (module docstring; descriptive, written once, never part of the selection)."""
        target_json, target_md = self.root/f"readouts_{stage}.json", self.root/f"readouts_{stage}.md"
        if target_json.exists() and target_md.exists():
            return
        lt = load_module("_eta_level_table", "experiments/m2v1/level_table.py")
        rows, records = [], []
        for c in selection["candidates"]:
            for r in c["rows"]:
                metrics_doc = read_json(r["eval_metrics"]) or {}
                m, cert = metrics_doc.get("metrics") or {}, metrics_doc.get("certificate") or {}
                record = lt.run_record(r["run_dir"], f"{stage}-{c['label']}-{r['pair']}"
                                       + (f"（复用 {c['reused_from']}）" if c.get("reused_from") else ""))
                records.append(record)
                native = [x for x in record["levels"] if x["level"] == 3]
                rows.append(dict(
                    candidate=c["label"], reused_from=c.get("reused_from"), nu=c["nu"], s_tau=c["s_tau"],
                    pair=r["pair"], run_dir=r["run_dir"], run_status=r.get("run_status"), wording=r.get("wording"),
                    field_role=r.get("field_role"), metrics={k: m.get(k) for k in READOUT_METRICS},
                    certificate={k: cert.get(k) for k in READOUT_CERT},
                    bernstein_violations=r.get("bernstein_violations"), bernstein_audit=r.get("bernstein_audit"),
                    bernstein_audit_sha256=r.get("bernstein_audit_sha256"),
                    bernstein_recertification_sha256=r.get("bernstein_recertification_sha256"),
                    bernstein_no_export=r.get("bernstein_no_export"),
                    levels=[dict(level=x["level"], round=x["round"], ending=x["ending"], status=x["status"],
                                 accepted=x["accepted"], cap=x["cap"], GF_B_over_T_B=x["GF_B_over_T_B"],
                                 seconds=x["seconds"], min_jd_lower_bound=x["min_jd_lower_bound"])
                            for x in record["levels"]],
                    native_accepted=sum(x["accepted"] or 0 for x in native) if native else None,
                    native_ending=native[-1]["ending"] if native else None,
                    native_GF_B_over_T_B=native[-1]["GF_B_over_T_B"] if native else None,
                    n_blocked=record["n_blocked"]))
        document = dict(schema=SCHEMA+".readouts", stage=stage, note=NOTE.format(wall=self.o.wall_seconds),
                        descriptive="只描述；不参与选择（选择见 selection_<stage>.json）", level_table_note=lt.NOTE,
                        configuration=selection["configuration"], rows=rows)
        head = ["候选", "ν", "s_τ", "对", "seg35", "seg4", "SDlogJ 官方", "SDlogJ 脑内", "log J q01 / q99（脑内）",
                "J < .5 %", "证书（状态 / 下界 / 预筛 / 回退）", "各层接受步", "各层收尾", "原生末 GF_B/T_B", "受阻层",
                "bernstein 违反"]
        lines = [f"# 阶段 {stage} 的读数（只描述）", "", NOTE.format(wall=self.o.wall_seconds), "",
                 f"配置：{configuration_text(selection['configuration'])}", "", "| "+" | ".join(head)+" |",
                 "|"+"---|"*len(head)]
        for row in rows:
            m, cert = row["metrics"], row["certificate"]
            cells = [row["candidate"]+(f"（复用 {row['reused_from']}）" if row["reused_from"] else ""),
                     f"{row['nu']:.6g}", fmt(row["s_tau"], 3), row["pair"], fmt(m.get("seg35")), fmt(m.get("seg4")),
                     fmt(m.get("sdlogj_official")), fmt(m.get("sdlogj_brain")),
                     f"{fmt(m.get('logj_q01_brain'), 3)} / {fmt(m.get('logj_q99_brain'), 3)}",
                     fmt(m.get("pct_J_lt_half_brain"), 3),
                     f"{cert.get('status')} / {fmt(cert.get('min_jd_lower_bound'), 3)} / {cert.get('n_prescreen_cells')} / "
                     f"{cert.get('n_fallback_cells')}",
                     "/".join(str(x["accepted"]) for x in row["levels"]) or "—",
                     "/".join(x["ending"] for x in row["levels"]) or "—", fmt(row["native_GF_B_over_T_B"]),
                     row["n_blocked"], "—" if row["bernstein_violations"] is None else
                     f"{row['bernstein_violations']}"
                     + (f"（审核 {row['bernstein_audit_sha256'][:12]}）" if row["bernstein_audit_sha256"] else "")]
            lines.append("| "+" | ".join(str(x).replace("|", "/") for x in cells)+" |")
        text = "\n".join(lines)+"\n\n"+lt.markdown(records, title=f"阶段 {stage} 各层读数（experiments/m2v1/level_table.py）")
        if not target_json.exists():
            write_once(target_json, finite_only(document))
        if not target_md.exists():
            write_once(target_md, text)

    def finish(self, center, selections):
        final = selections["s3"]["chosen_candidate"]
        path = [dict(stage=s, centre_nu=selections[s]["nu_center"], chosen=selections[s]["chosen"],
                     nu=selections[s]["chosen_candidate"]["nu"], s_tau=selections[s]["chosen_candidate"]["s_tau"],
                     mean_seg35=selections[s]["chosen_candidate"]["mean_seg35"],
                     mean_seg4=selections[s]["chosen_candidate"]["mean_seg4"], reason=selections[s]["reason"])
                for s in STAGES]
        eta = dict(alpha=final["alpha"], nu=final["nu"], tau=final["tau"])
        document = dict(schema=SCHEMA+".summary", note=NOTE.format(wall=self.o.wall_seconds), stage_set=STAGE,
                        pairs=self.pairs, nu_star=center, eta_selected=eta, selection_path=path,
                        configuration=self.configuration(),
                        code_sha=self.code_sha, selections={s: str(self.root/f"selection_{s}.json") for s in STAGES})
        selected = dict(eta, use="EB 初值（C15-6 的 η⁽⁰⁾）；开发集结果，不是最终 η", selection_path=path,
                        nu_star=center["nu_star"], configuration=self.configuration(), code_sha=self.code_sha)
        if self.v4:                         # ETA4: edges (recorded, never acted on), the path and its record
            cands = [c for s in ("s1", "s2") for c in selections[s]["candidates"]]
            nus = [c["nu"] for c in cands]
            eligible = [c["nu"] for c in cands if c["eligible"]]
            # ETA4 review M5: both ends alike, and where the SDlogJ condition bound (candidates excluded on either side
            # of the chosen nu; below it is the side the sweep set out to explore)
            excluded = sorted({(c["nu"], c["label"]) for c in cands if c.get("excluded_by_sdlogj")})
            edges = dict({s: selections[s].get("edge") for s in STAGES},
                         nu_lowest_run=min(nus), nu_highest_run=max(nus),
                         nu_lowest_eligible=min(eligible) if eligible else None,
                         nu_highest_eligible=max(eligible) if eligible else None,
                         still_at_lower_end=final["nu"] == min(nus), still_at_upper_end=final["nu"] == max(nus),
                         at_lower_end_of_eligible=bool(eligible) and final["nu"] == min(eligible),
                         excluded_by_sdlogj_below_chosen=[dict(nu=n, label=lab) for n, lab in excluded
                                                          if n < final["nu"]],
                         excluded_by_sdlogj_above_chosen=[dict(nu=n, label=lab) for n, lab in excluded
                                                          if n > final["nu"]])
            extra = dict(design="v4", path=self.path_name, path_record=str(self.root/"path.json"),
                         path_record_sha256=file_sha256(self.root/"path.json"),
                         centre_record=str(self.root/"centre"/"centre.json"), edges=edges,
                         sdlogj_official_max=getattr(self.o, "sdlogj_official_max", None),
                         excluded_by_sdlogj={s: selections[s].get("excluded_by_sdlogj") for s in STAGES},
                         readouts={s: str(self.root/f"readouts_{s}.json") for s in STAGES},
                         unverified=("在带 --wall-seconds 的开发路径下选出（C13-5a）：正式运行不用时间预算，"
                                     "η 能否沿用未验证"))
            document.update(extra)
            selected.update(design="v4", path=self.path_name, edges=edges, unverified=extra["unverified"])
        for name, doc in (("eta_selected.json", selected), ("summary.json", document)):
            if not (self.root/name).exists():
                write_once(self.root/name, doc)
        if not (self.root/"summary.md").exists():
            lines = ["# η 初值扫描汇总（合同 C05-7，P2）", "", NOTE.format(wall=self.o.wall_seconds), "",
                     f"配对：{', '.join(self.pairs)}；代码 {self.code_sha}", "",
                     f"配置：{configuration_text(document['configuration'])}", ""]
            if self.v4:
                lines += [f"中心（不跑探针）：v3 选中的 ν = {center['nu_star']:.6g}，τ⁽⁰⁾ {center['tau0']}；来源 "
                          f"{center['source']}（sha256 {center['source_sha256'][:12]}）", "",
                          f"路径：{self.path_name}（{self.root/'path.json'}）", ""]
            else:
                lines += [f"ν*（几何平均）= {center['nu_star']:.6g}；各对 {center['per_pair']}", ""]
            for s in STAGES:
                lines.append(stage_markdown(selections[s], self.o.wall_seconds))
            lines += ["## 选中的 η 与选择路径", "", f"η = α {eta['alpha']:.6g}，ν {eta['nu']:.6g}，τ {eta['tau']}", ""]
            lines += [f"- {p['stage']}：中心 ν {p['centre_nu']:.6g} → {p['chosen']}（ν {p['nu']:.6g}，s_τ {p['s_tau']:g}；"
                      f"seg35 平均 {fmt(p['mean_seg35'])}，seg4 平均 {fmt(p['mean_seg4'])}；{p['reason']}）" for p in path]
            if self.v4:
                e = document["edges"]
                ends = [t for t, flag in (("仍在下端（本次运行过的最小 ν）", e["still_at_lower_end"]),
                                          ("仍在上端（本次运行过的最大 ν）", e["still_at_upper_end"])) if flag]
                below = "、".join(f"{x['label']}（ν {x['nu']:.6g}）" for x in e["excluded_by_sdlogj_below_chosen"])
                above = "、".join(f"{x['label']}（ν {x['nu']:.6g}）" for x in e["excluded_by_sdlogj_above_chosen"])
                lines += ["", f"- 网格位置：s1 {EDGE_TEXT.get(e['s1'], e['s1'])}，s2 {EDGE_TEXT.get(e['s2'], e['s2'])}，"
                          f"s3 {EDGE_TEXT.get(e['s3'], e['s3'])}；选中 ν {'，'.join(ends) or '不在运行过的 ν 的两端'}。"
                          "只记录，不自动扩展网格，交总负责人。",
                          f"- SDlogJ 条件（{document['sdlogj_official_max']}）排除的候选：选中 ν 下侧 {below or '无'}；"
                          f"上侧 {above or '无'}；最低合格 ν {fmt(e['nu_lowest_eligible'])}"
                          f"{'（选中值即最低合格 ν）' if e['at_lower_end_of_eligible'] else ''}。",
                          f"- {document['unverified']}"]
            write_once(self.root/"summary.md", "\n".join(lines)+"\n")
        self.log(f"selected eta: alpha {eta['alpha']:.6g}, nu {eta['nu']:.6g}, tau {eta['tau']} (EB initial value); "
                 f"{self.root/'summary.md'}")

    # ---- main --------------------------------------------------------------------------------------------------------
    def main(self):
        if self.dry and any(self.root.resolve() == real or real in self.root.resolve().parents for real in REAL_ROOTS):
            print(f"--dry-run: --root must lie outside {', '.join(map(str, REAL_ROOTS))}", file=sys.stderr)
            return 2
        if self.v4 and any(self.root.resolve() == p or p in self.root.resolve().parents for p in PROTECTED_V4):
            print(f"--design v4: --root must lie outside {', '.join(map(str, PROTECTED_V4))} (read only)",
                  file=sys.stderr)
            return 2
        self.root.mkdir(parents=True, exist_ok=True)
        if self.v4 or self.level_caps is not None or self.no_start_after is not None:     # ETA4 (v3: the line before)
            self.log(f"start (ETA4): design {'v4' if self.v4 else 'v3'}; path "
                     + (f"from pilot {self.o.path_from_pilot}" if getattr(self.o, "path_from_pilot", None) else "")
                     + (f" explicit {self.o.path} ({self.o.path_note!r})" if getattr(self.o, "path", None) else "")
                     + (f"; per-level caps {self.level_caps}" if self.level_caps is not None else "")
                     + (f"; centre {self.o.centre_eta}" if self.v4 else "")
                     + (f"; sdlogj_official_max {self.o.sdlogj_official_max}" if self.v4 else "")
                     + (f"; no start after {self.no_start_after.isoformat()}" if self.no_start_after else ""))
        self.log(f"start: root {self.root}; until {self.o.until}; configuration channel_source {self.o.channel_source}, "
                 f"logvol_points {self.o.logvol_points}"
                 + (f" (beta_S {self.o.logvol_closed27_share:g}, {CLOSED27_SIMPSON})" if self.closed27 else "")
                 + f", metric_jd_scaling {self.o.metric_jd_scaling}, jd_cell_estimate {self.o.jd_cell_estimate}"
                 + (f", sigma_domain {self.sigma_domain} (theta {self.support_threshold:g})" if self.support else "")
                 + (f", structure root {self.structure_root} ({self.o.segmenter_id}, {self.o.structure_grouping})"
                    if self.structure else "")
                 + f", backend {self.o.backend}/{self.device}, outer {self.o.outer}, rounds {self.o.rounds}, wall "
                 f"{self.o.wall_seconds:g} s (development budget, 预算收尾), extra {self.o.extra!r}; MemoryMax "
                 f"{self.o.memory_max} (eval {self.o.eval_memory_max}, probe {self.o.probe_memory_max}, atropos "
                 f"{self.o.atropos_memory_max}); min GPU free {self.o.min_gpu_free_mib} MiB; dry-run {int(self.dry)}; "
                 f"self unit {self.sys.self_unit or 'none'}")
        try:
            if self.deadline_reached():       # ETA4: a deadline already past -> nothing read, nothing started
                raise Stop(f"--no-start-after {self.no_start_after.isoformat()} has already passed")
            self.start_checks()
            self.prepare_code()
            self.write_env()
            self.structure_precheck()
            if self.o.until == "precheck":
                self.log("until precheck: end")
                return 0
            if self.v4:
                self.resolve_path()          # before any unit: a held path starts nothing
            self.atropos_phase()
            if self.o.until == "atropos":
                self.log("until atropos: end")
                return 0
            center = self.centre_phase() if self.v4 else self.probe_phase()
            if self.o.until in ("probe", "centre"):
                self.log(f"until {self.o.until}: end")
                return 0
            selections, previous = {}, None
            nu = center["nu_star"]
            for stage in STAGES:
                selections[stage] = previous = self.stage(stage, nu, center["tau0"], previous)
                nu = previous["chosen_candidate"]["nu"]
                if self.o.until == stage and stage != STAGES[-1]:
                    self.log(f"until {stage}: end")
                    return 0
            self.finish(center, selections)
            self.log("end")
            return 0
        except Stop as exc:
            self.log(f"STOP: {exc}")
            return 8


def subject_number(subject):
    match = re.fullmatch(r"OASIS_OAS1_([0-9]{4})_MR1", subject)
    return match.group(1) if match else re.sub(r"[^A-Za-z0-9]", "", subject)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -B experiments/eta/sweep.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default=None, help=f"default {REAL_ROOT} (--design v3) or {REAL_ROOT_V4} (v4)")
    parser.add_argument("--code")
    parser.add_argument("--commit")
    parser.add_argument("--until", choices=UNTIL, default="s3",
                        help="precheck: start checks, code tree, structure precheck only; probe (v3) / centre (v4)")
    v4 = parser.add_argument_group("design v4 and the solver path (ETA4; module docstring)")
    v4.add_argument("--design", choices=DESIGNS, default="v3",
                    help="v3 (default: the C05-7 three-stage sweep from the probe, unchanged) or v4 (centre = the v3 "
                    "choice, grid LABELS_V4, a path from the pilot or explicit)")
    v4.add_argument("--centre-eta", default=None, help=f"v4: the centre eta (default {V3_ETA}, sha256 pinned in real "
                    "runs; a dry run gives a synthetic one)")
    v4.add_argument("--accept-code-change", action="store_true",
                    help="v4: accept (and record) model-code changes since the centre's code_sha")
    v4.add_argument("--path-from-pilot", default=None, help="v4: the path pilot's summary.json (pilot_path: the "
                    "preregistered suggestion taken mechanically; not resolved -> STOP, the lead decides)")
    v4.add_argument("--path", choices=tuple(PATHS), default=None,
                    help="v4: the path given explicitly (the lead's decision; needs --path-note)")
    v4.add_argument("--path-note", default=None, help="v4: who decided the explicit --path, when, why (recorded)")
    v4.add_argument("--sdlogj-official-max", default=None,
                    help=f"v4: eligibility also needs the three official SDlogJ finite and their mean <= this (default "
                    f"{SDLOGJ_V4:g}, contract C05-7 item 7b / D56, mandatory under v4); another value or 'none' only "
                    "with --sdlogj-note (recorded in centre.json)")
    v4.add_argument("--sdlogj-note", default=None,
                    help="v4: who decided a --sdlogj-official-max other than the contract's, when, why (recorded)")
    v4.add_argument("--max-accepted-per-level", default=None,
                    help="run_pair's C13-4a caps a0,a1,a2,a3 for every run (not the probe); with --path / "
                    "--path-from-pilot the path sets them")
    v4.add_argument("--no-start-after", default=None,
                    help="no run unit starts at or after this ISO date-time with a UTC offset (e.g. "
                    "2026-11-02T06:00-06:00); STOP, resumable")
    parser.add_argument("--channel-source", choices=CHANNEL_SOURCES, default="structure")
    parser.add_argument("--logvol-points", choices=LOGVOL_POINTS, default=LOGVOL_GAUSS27_CLOSED27,
                        help="C05-2 / C05-2a / C05-2b point set of the log-volume prior (default gauss27+closed27, (A'))")
    parser.add_argument("--logvol-closed27-share", type=float, default=1.,
                        help="C05-2b beta_S > 0 (default 1 = D48; the method's beta is decided after the GPU smoke v2: "
                        "pass it); only with --logvol-points gauss27+closed27. The closed27 rule is Simpson (D48)")
    parser.add_argument("--metric-jd-scaling", choices=JD_SCALING_MODES, default="affine_v1")
    parser.add_argument("--jd-cell-estimate", choices=JD_CELL_ESTIMATES, default=None,
                        help="C09.1-3a cell estimate of the JD scaling (default certificate, D47 as set by the lead "
                        "10-31; run_pair's default since DEF); no effect with --metric-jd-scaling off: not given -> none "
                        "(not passed to run_pair / probe), given -> refused, as run_pair")
    parser.add_argument("--structure-root", default=str(DEFAULT_STRUCTURE_ROOT),
                        help="structure predictions <root>/<subject>/ (channel_source structure; never generated here)")
    parser.add_argument("--structure-grouping", default=DEFAULT_GROUPING)
    parser.add_argument("--segmenter-id", default=SEGMENTER_ID)
    parser.add_argument("--segmenter-registry", default=None,
                        help="dry runs only: the synthetic staging registry of experiments/eta/dry_inputs.py (passed "
                        "with --structure-registry-staging); real runs use the code tree's configs/bayesreg/segmenters.json")
    parser.add_argument("--backend", choices=("torch64", "numpy"), default="torch64")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--outer", choices=("a",), default="a", help="option (a) (the probe's definition of u_1)")
    parser.add_argument("--rounds", default="1,1,1,1")
    parser.add_argument("--wall-seconds", type=float, default=900., help="development budget per run (预算收尾; default "
                        "900)")
    parser.add_argument("--extra", default="", help="more run_pair arguments (e.g. '--max-accepted 3' in a dry run); "
                        "never an option the sweep sets itself")
    parser.add_argument("--data-root")
    parser.add_argument("--allowlist")
    parser.add_argument("--tissue-root")
    parser.add_argument("--memory-max", default="9G", help="run units (default 9G, the H arms of the GPU smoke v2)")
    parser.add_argument("--eval-memory-max", default="6G")
    parser.add_argument("--probe-memory-max", default="9G", help="probe units (default 9G: the probe prepares the "
                        "native channels of the structure layer as a run does)")
    parser.add_argument("--atropos-memory-max", default="4G")
    parser.add_argument("--atropos-threads", type=int, default=8)
    parser.add_argument("--min-gpu-free-mib", type=int, default=22528,
                        help="run units on cuda (default 22528: the H arms of the GPU smoke v2, closed27 19.9 GiB + 2 GiB)")
    parser.add_argument("--home-floor-gb", type=int, default=100)
    parser.add_argument("--wait-minutes", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sigma-domain", choices=SIGMA_DOMAINS, default=None,
                        help="C04-6 / C05-7a (SWL draft; work package SWLI): run_pair's --sigma-domain for the probe "
                        "and every run (default brain: nothing passed, the records before SWLI); part of the root's "
                        "configuration")
    parser.add_argument("--support-threshold", type=float, default=None,
                        help="C04-6 / D50 theta (default .05 with support / support_shared; refused with brain); "
                        "run_pair's checks")
    args = parser.parse_args(argv)
    for name in ("memory_max", "eval_memory_max", "probe_memory_max", "atropos_memory_max"):
        try:
            if to_mib(getattr(args, name)) <= 0:
                raise ValueError
        except ValueError:
            parser.error(f"--{name.replace('_', '-')} {getattr(args, name)}: an integer size such as 7G")
    if not (args.wall_seconds > 0 and math.isfinite(args.wall_seconds)):
        parser.error("--wall-seconds: a positive number")
    if not (args.logvol_closed27_share > 0 and math.isfinite(args.logvol_closed27_share)):
        parser.error("--logvol-closed27-share: beta_S must be finite and > 0 (C05-2b)")
    if args.logvol_points != LOGVOL_GAUSS27_CLOSED27 and args.logvol_closed27_share != 1.:
        parser.error("--logvol-closed27-share: only with --logvol-points gauss27+closed27 (C05-2b)")
    # ETA4: design v4, the path, the caps, the deadline (v3 without them: the parse before ETA4)
    v4 = args.design == "v4"
    if not v4:
        given = [f"--{n.replace('_', '-')}" for n in ("centre_eta", "path_from_pilot", "path", "path_note",
                                                       "sdlogj_official_max", "sdlogj_note")
                 if getattr(args, n) is not None]
        given += ["--accept-code-change"] if args.accept_code_change else []
        given += ["--until centre"] if args.until == "centre" else []
        if given:
            parser.error(f"{', '.join(given)}: --design v4 only")
    else:
        if args.until == "probe":
            parser.error("--until probe: v4 runs no probe (--until centre)")
        if args.path is None and args.path_from_pilot is None:
            parser.error("--design v4: give --path-from-pilot SUMMARY or --path NAME --path-note TEXT (the path record)")
        if args.path is not None and not (args.path_note or "").strip():
            parser.error("--path: --path-note TEXT is required (who decided, when, why)")
        if args.jd_cell_estimate is not None or args.max_accepted_per_level is not None:
            parser.error("--design v4: the path sets --jd-cell-estimate and --max-accepted-per-level (one source)")
        if args.metric_jd_scaling != "affine_v1":
            parser.error("--design v4: the paths need --metric-jd-scaling affine_v1")
        if args.centre_eta is None:
            if args.dry_run:
                parser.error("--design v4 --dry-run: --centre-eta (a synthetic centre) is required")
            args.centre_eta = str(V3_ETA)
        # contract 7b (ETA4 review I1): the SDlogJ condition is part of v4's eligibility; off or another value only
        # explicitly, with a note
        text = SDLOGJ_V4 if args.sdlogj_official_max is None else args.sdlogj_official_max
        if str(text).strip().lower() == "none":
            value = None
        else:
            try:
                value = float(text)
            except ValueError:
                parser.error(f"--sdlogj-official-max {text!r}: a positive finite number or none")
            if not (value > 0 and math.isfinite(value)):
                parser.error("--sdlogj-official-max: a positive finite number or none")
        if value != SDLOGJ_V4 and not (args.sdlogj_note or "").strip():
            parser.error(f"--sdlogj-official-max {text}: not the contract's {SDLOGJ_V4:g} (C05-7 item 7b, D56); "
                         "--sdlogj-note TEXT is required (who decided, when, why)")
        if value == SDLOGJ_V4 and args.sdlogj_note is not None:
            parser.error("--sdlogj-note: only with a --sdlogj-official-max other than the contract's")
        args.sdlogj_official_max = value
        # contract 7b scope: the centre was chosen under brain / sqrt_prob
        if (args.sigma_domain or BRAIN) != BRAIN or args.support_threshold is not None:
            parser.error("--design v4: sigma_domain brain only (contract C05-7 item 7b scope; another domain is a new "
                         "sweep)")
        # ETA4 review M8: a real v4 run that starts runs has a deadline
        if not args.dry_run and args.until in STAGES and args.no_start_after is None:
            parser.error("--design v4: --no-start-after ISO is required for a real run that starts runs (--until "
                         f"{args.until}); the BZ-authorised window")
    if args.path_note is not None and args.path is None:
        parser.error("--path-note: only with --path")
    if args.root is None:
        args.root = str(REAL_ROOT_V4 if v4 else REAL_ROOT)
    if args.max_accepted_per_level is not None:
        try:
            args.max_accepted_per_level = parse_caps(args.max_accepted_per_level)
        except ValueError as exc:
            parser.error(f"--max-accepted-per-level: {exc}")
    args.no_start_after_utc = None
    if args.no_start_after is not None:
        try:
            args.no_start_after_utc = parse_deadline(args.no_start_after)
        except ValueError as exc:
            parser.error(f"--no-start-after: {exc}")
    # run_pair.resolve_jd_cell_estimate's rule (C09.1-3a last item, DEF)
    if args.metric_jd_scaling == JD_OFF:
        if args.jd_cell_estimate is not None:
            parser.error(f"--jd-cell-estimate {args.jd_cell_estimate}: no effect with --metric-jd-scaling off; "
                         "leave it out (recorded as none) (C09.1-3a)")
        args.jd_cell_estimate = JD_NONE
    elif args.jd_cell_estimate is None:
        args.jd_cell_estimate = JD_DEFAULT
    # run_pair's sigma_domain rules (C04-6): brain refuses an explicit theta; none refuses support; theta finite in (0, 1)
    # and above the floor (structure: K_S of the grouping; tissue: the K-independent bound, as run_pair)
    domain = args.sigma_domain or BRAIN
    if domain == BRAIN:
        if args.support_threshold is not None:
            parser.error(f"--support-threshold {args.support_threshold:g}: not with --sigma-domain brain (C04-6)")
        args.support_threshold = None
    else:
        if args.channel_source == "none":
            parser.error(f"--sigma-domain {domain}: no structure channel with --channel-source none (C04-6)")
        theta = DEFAULT_SUPPORT_THRESHOLD if args.support_threshold is None else args.support_threshold
        try:
            k = load_grouping(DEFAULT_GROUPS_PATH, args.structure_grouping).k if args.channel_source == "structure" \
                else 1
            args.support_threshold = check_theta(theta, k)
        except (OSError, ValueError, InputValidationError) as exc:
            parser.error(f"--support-threshold: {exc}")
    args.sigma_domain = domain
    try:
        extra = shlex.split(args.extra)
    except ValueError as exc:
        parser.error(f"--extra: {exc}")
    # argparse also takes unambiguous prefixes (allow_abbrev): a prefix of a managed option is refused too
    flags = {a.split("=", 1)[0] for a in extra if a.startswith("--")}
    repeated = sorted(f for f in flags if any(m.startswith(f) for m in MANAGED))
    if repeated:
        parser.error(f"--extra must not repeat options the sweep sets itself: {repeated}")
    # ETA4: the SCH switches only through --max-accepted-per-level / the path (recorded); with caps no --max-accepted
    # (a v4 path's caps are known after path resolution, which repeats this check)
    conflicts = extra_conflicts(args.extra, args.max_accepted_per_level)
    if conflicts:
        parser.error(f"--extra must not name {conflicts} (run_pair's SCH switches go through --max-accepted-per-level "
                     "or the path; --max-accepted is refused with per-level caps, C13-4a)")
    return args


def main(argv=None, system=None):
    return Sweep(parse_args(argv), system).main()


if __name__ == "__main__":
    sys.exit(main())
