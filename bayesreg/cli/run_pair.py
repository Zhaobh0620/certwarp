"""One pair through the B-line solver (contract C03, C07-C13, C16).

    python -m bayesreg.cli.run_pair --pair <stage>/<pair> --channel-source none|tissue|structure --out <dir>
                                    [--max-levels N] [--outer a|b] [--rounds 1,1,1,3] [--max-accepted 500]
                                    [--max-outer 60] [--data-root DIR] [--allowlist PATH] [--tissue-root DIR]
                                    [--backend numpy|torch64] [--device cpu|cuda]
                                    [--metric-jd-scaling affine_v1|off] [--cpu-threads 4]
                                    [--jd-cell-estimate certificate|gauss27+corners|closed27]
                                    [--structure-root DIR] [--segmenter-registry PATH] [--segmenter-models-root DIR]
                                    [--structure-image-root DIR] [--structure-groups PATH]
                                    [--structure-grouping structure13_v1] [--structure-infer never|missing]
                                    [--structure-device cpu|cuda] [--structure-synthetic]
                                    [--structure-registry-staging] [--wall-seconds S]
                                    [--jacobian-cache on|off] [--jacobian-cache-log on|off]
                                    [--speculative-gradient on|off]
                                    [--logvol-points gauss27+closed27|gauss27|gauss27+corners]
                                    [--logvol-closed27-share BETA_S] [--closed27-weights simpson|uniform]
                                    [--witness-estimate on|off]
                                    [--sigma-domain brain|support|support_shared] [--support-threshold THETA]
                                    [--max-accepted-per-level a0,a1,a2,a3] [--level-time-shares s0,s1,s2]
                                    [--inner pg|lm]
                                    [--init-field PATH --start-level 3 [--init-field-oracle-ok]]
                                    [--checkpoint-accepted N | --checkpoint-seconds S]

Order: the run directory is created (it must not exist or be empty: nothing is overwritten, hard
constraint 4) and cmd.txt, git.txt, env.txt, config.json are written; then load_pair (manifest ->
subject guard on the development allowlist -> pair.npz members fixed / moving only; a sealed subject ends
the run with status subject_sealed before any image is decoded); the tissue layer P_T of both images
(Atropos posteriors under --tissue-root/<subject>/, provenance validated and file hashes verified; any
rejection -> provenance_rejected); C03.2-C03.6 (simplex with background, brain mask of the fixed image,
white-matter normalization, smooth -> floor, four-level pyramid); the multilevel solve; export.
channel_source: tissue -> the likelihood channels are the tissue layer (P_S := P_T, the label-free
ablation arm); none -> intensity only; structure -> the structure layer P_S of the registered self-trained
segmenter (bayesreg.seg.structure_layer; contract C03.1-2/3, C03.2-4a, D24, D37, D39): per image
--structure-root/<subject>/{structure_groups.npz, provenance.json} of bayesreg.seg.unet.infer; a missing one is
refused (--structure-infer never, the default) or generated there (--structure-infer missing; occupies the GPU
unless --structure-device cpu, so not during timed runs) with the fold models that never saw the subject's labels
(training subject: the other fold; everyone else: both folds, mean, which needs 2 registered folds); grouping
--structure-grouping (default structure13_v1; an unverified grouping is refused); registry --segmenter-registry
(default the repository configs/bayesreg/segmenters.json, recorded as registered). Any other registry, e.g. the
staging copy of the training chain (the default path once --structure-registry-staging is given), is admitted
only with --structure-registry-staging: recorded as staging_unregistered = 未正式登记 in provenance.json and
status.json and flagged by bayesreg.eval, development diagnostics only; ensemble.json beside it must carry the
registry's sha256 and member_check.passed true. Before any prediction is read or made, the D24 pair-level check (the label contact
of the models used for either image contains neither subject); then the provenance (test_labels_seen false, model
and member sha256 equal to the registry, model selection, grouping, grid, hashes) and the read-back (raw channel
sum within 1e-3 of 1 on every voxel for float16 storage, 1e-6 for float32/64, before the normalization); the
(K_S + 1)-channel groups enter C03.2 through simplex_from_background_class (C03.2-4a; channel 0 = background, not
a likelihood channel, D2), then smooth -> floor -> sqrt and the pyramid. Any refusal -> provenance_rejected.
Whatever the source, tau^2, the brain mask and the normalization come from P_T (C03.1-3).
eta: the C05-7 temporary values (P2) unless --eta-json gives {"alpha", "nu", "tau"} (tau_0..tau_K_T).
sigma: the Gaussian M step of C11-2 at every level start and after every round (EM proper: later package);
the sigma trajectory and the C11-5 collapse flag go to hyper.json (sigma_monitor) and qc.json (flag only).
backend: numpy (CPU FP64 reference, default) or torch64 (device FP64, --device cuda|cpu, default cuda when
available; the GPU-path stable comparison of C08-8 with CPU fallback; its statistics in hyper.json
gpu_comparison; the certificate is the CPU call either way).
metric_jd_scaling (C09.1-3a, path only): affine_v1 (default) uses p = -S~ M^-1 S~ g in PG; off is the C09.1 metric
bit for bit; the computations of S~ (min J_a, fraction r_a < 1) and jd_scaling_refresh are in hyper.json (round
metric.jd_scaling) and log.jsonl (kind jd_scaling).
jd_cell_estimate (C09.1-3a last item, D47, path only): the cell estimate m_K of S~ -- certificate (default since work
package DEF, 2026-10-31: min(m_K, the per-cell certified lower bound of the accepted state)), gauss27+corners (the
default before DEF), closed27 (also the {0, 1/2, 1}^3 points), certificate_bernstein (JDTI, D53, not the default:
on edge-hull cells with b_K < J_act the bound max(b_K, jd_curl prescreen bound); its check (b) violations are warned in
log.jsonl (kind warning) and their run total is status.json bernstein_j53_violations_total, written with this option
only: the sum over the round records -- an exception row included -- or, when an exception ended the run outside every
round, over the warning rows of log.jsonl, with bernstein_j53_violations_source saying so).  With --metric-jd-scaling off the option
has no effect: not given, it is recorded as "none"; given (any value), the run is refused (usage error, exit 2, no run
directory; before DEF only the default was allowed).  Recorded in config.json (args, solver.pg), hyper.json and
provenance.json switches.
logvol_points (C05-2 / C05-2a / C05-2b, model option): gauss27+closed27 (C05-2b, BZ adopted (A') 10-30; the default
since DEF, 2026-10-31: C05-2b first paragraph, D43, D48) puts the log-volume prior on Gauss27 U closed27 by shares
1/(1 + beta_S) and beta_S/(1 + beta_S): --logvol-closed27-share beta_S (default 1), --closed27-weights simpson (default,
D48) | uniform (ablation); both only with gauss27+closed27 (refused otherwise).  gauss27 (C05-2, the default before
DEF) and gauss27+corners (C05-2a: adds the cell-corner term of the log-volume prior to F_B on every level) are explicit
options.  Only these contract names (C05-2b "记录与 CLI"; the IMPL27 aliases were removed in IMPL27b).  Recorded in
config.json (args, solver) and provenance.json switches (logvol_points, logvol_closed27_share, closed27_weights; the
last two null unless gauss27+closed27).
Reproducing a run of the code before DEF (MIN1; review IMPL27b_EBX, EBX minor 1; general rule FUP1, review MIN1
minor 4): DEF changed two defaults, --logvol-points (gauss27 -> gauss27+closed27) and --jd-cell-estimate
(gauss27+corners -> certificate).  General rule: take the command line of the run before DEF and add each of these two
switches it did not give, with its value before DEF: --logvol-points gauss27; with the metric on (--metric-jd-scaling
affine_v1, the default) --jd-cell-estimate gauss27+corners (with --metric-jd-scaling off the cell estimate has no
effect and must not be given: it is refused; an old command line with off that gave --jd-cell-estimate, which before
DEF admitted only gauss27+corners there, drops it).  Every other switch keeps the value the old command line gave.
The run is then bitwise the code before DEF; with only one of the two added, the other runs with its new default,
which is not.
Examples: the default run before DEF -> --logvol-points gauss27 --jd-cell-estimate gauss27+corners; a C05-2a arm
(--logvol-points gauss27+corners given) -> add --jd-cell-estimate gauss27+corners; an IMPL27b-era run with
--logvol-points gauss27+closed27 given -> add --jd-cell-estimate gauss27+corners; a run with --metric-jd-scaling off ->
add --logvol-points gauss27 only.  The first, second and last are the golden arms pre_def, corners and pre_def_off of
tests/test_bayesreg/solve/_run_pair_digest.py (checked against 8a4d060).
Witness record (C05-2b "记录与 CLI", C16; always, a record only): every log.jsonl candidate row carries witness_cell,
witness_xi, witness_type (the exact class of witness_xi) and witness_cell_min_j53 (bayesreg.solve.witness; null without
a certificate witness point); transfer rows and cert.json through the certificate record; status.json witness_summary
counts the JACOBIAN_VIOLATION outcomes by type.  --witness-estimate on (default off; a diagnostic beyond the contract)
adds witness_estimate, an estimate of the witness cell's minimum J (dense sampling + L-BFGS-B; costs time per
rejection).  The time is in timing.json (witness_seconds, witness_records), not in certificate_seconds.

sigma_domain (contract C04-6, C11-8; work package SWLI; draft SWL 2026-10-31 -- D49 / D50 are BZ's, the default is
unchanged): --sigma-domain brain (default; bitwise the code before SWLI) | support | support_shared, the estimation domain
of the structure-channel noise variances; --support-threshold THETA (D50, default .05; finite, 0 < THETA < 1, and above
the C03.5-9 floor eps/(1 + (K + 1) eps), eps = 1e-3).  Refused at argument parsing, before the run directory exists
(usage error, exit 2): an explicit --support-threshold with brain; support / support_shared with --channel-source none;
THETA not finite or outside (0, 1); THETA at or below the floor -- K = K_S of --structure-grouping in --structure-groups
(structure), and for tissue, where K_T is known only from the Atropos provenance read after the run directory exists, the
K-independent bound eps/(1 + 2 eps) = max_K of the floor (stricter than K_T's by eps/(1 + 2 eps) - eps/(1 + (K_T + 1)
eps), 2.0e-6 for K_T = 3; the solver checks K_T exactly).
Recorded in config.json (args; solver and sigma: sigma_domain, support_threshold, support_rule), provenance.json switches
(the same three keys), hyper.json (algorithm; non-brain only: support_levels, the per-level support statistics, and the
round rows' sigma2_S_used / sigma2_S_post / sigma_tilde2, null for an empty-support channel) and timing.json
(solve_levels[].support_cpu_seconds, non-brain only).

wall_seconds (development switch, default none; contract C13-3 appendix, C20 D44, pending review -- C13-5 keeps "no time
budget" for formal runs): a wall-clock budget of S seconds counted from the start of main(); the solver checks it only
at safe points (PG iteration top, before every further line-search candidate, option (b) outer iteration top, before
every transfer). Used up on the native level -> status budget_exhausted ("预算收尾", exit 0): the last formal commit is
exported and passes the fresh CPU certificate like any ending (else final_certificate_rejected); used up before the
native level -> stopped_before_native (final_level<ell>.npz, no disp.npz, exit 2). The export and its certificate run
after the budget. The record (wall_seconds, where and when it was found used up, checks) is in config.json
(wall_budget, at the start of the solve), provenance.json (switches.wall_seconds), hyper.json (wall_budget, at the end
of the solve), the ending round row (budget), status.json and timing.json (wall_budget, at the end of the run).

Per-level caps and coarse-level shares (work package SCHI; contract C13-4a, C13-5a, C16 SCH item, D51 / D52; draft SCH
2026-10-31, path only, defaults unchanged; development runs only):
--max-accepted-per-level a0,a1,a2,a3 (C13-4a): the accepted-step cap of every (level, round) of level ell is a_ell (the
whole string must match ^[0-9]+(,[0-9]+){3}$: four decimal integers >= 0; "+3", " 3", "3_0" are refused); --max-accepted
N stays N on every level; giving both is refused.  --level-time-shares s0,s1,s2 (C13-5a; only with --wall-seconds W):
three finite floats, 0 < s < 1, s0 + s1 + s2 < 1; coarse level ell ends its round level_budget_reached ("粗层预算转层")
once (s0 + ... + s_ell) W seconds from the start of main() have passed, checked only at PG / option (b) outer iteration
tops; the level's remaining rounds are skipped and the run goes on to the transfer; the native level only has W.
Every refusal happens in parse_args (usage error, exit 2): no run directory, no file, no input read.
Records only when either switch is given (C16 SCH; without them every record file is byte-identical to the code before
SCH up to the measured mask of tests/test_bayesreg/_golden/golden_11409eb_sch_records.json): config.json args
max_accepted_per_level and level_time_shares (the one not given null; with --max-accepted-per-level, args.max_accepted is
null), solver (also in provenance.json solver and hyper.json algorithm) max_accepted_per_level, max_accepted_source,
level_time_shares, level_deadlines_seconds (max_accepted_per_round null when the four caps differ); provenance.json
switches max_accepted_per_level and level_time_shares; hyper.json / log.jsonl round rows cap_accepted,
accepted_committed, level_time_share, level_cumulative_share, level_deadline_seconds, level_elapsed_seconds,
run_elapsed_seconds, ended_by_level_share, level_share_hit, rounds_skipped; log.jsonl kind "level_budget" rows;
timing.json solve_levels.levels[] deadline_seconds and ended_by_level_share.  qc.json and status.json never change.

Inner solver (work package LMI, 2026-11-01; contract C09.3, path only, default unchanged): --inner pg (default; run_pg,
bitwise the code before LMI) | lm (bayesreg.solve.lm.run_lm: the C09.3 LM-PCG inner, options (a) and (b)).  Under lm the
JD affine scaling and its cell estimate (C09.1-3a, PG direction only) have no effect: --metric-jd-scaling and
--jd-cell-estimate not given -> recorded as "none" (config.json args, provenance.json switches); given (any value) ->
refused; lm needs --backend numpy (the LM class of the C08-8 GPU-path comparison has not passed the C08-9 L1 gate).
Every refusal happens in parse_args (usage error, exit 2): no run directory.  --wall-seconds and --level-time-shares are
checked at the LM iteration tops only (C13-5a LM item), never between direction attempts.  Records: config.json args
`inner` (whenever --inner is given); under lm only: solver.inner and solver.lm (config.json, provenance.json solver,
hyper.json algorithm; solver.pg null), provenance.json switches inner "LM", the round rows' metric.lm, and log.jsonl rows
of kind lm_start / candidate / accept with gamma, the PCG iterations and residuals, pred and the model ratio.

Native-level warm start (work packages WST, WARM; contract C12-4a, C16 WST item, D59; path only, default off; BZ 11-03: a seed may enter
the method, whether it becomes the default is open, and every run so far is a development diagnostic): --init-field PATH --start-level 3 (both or neither) starts the run at the
native level from a seed, the final_native.npz of a finished run on this pair (run_pair or the frozen configuration C solver;
bayesreg.solve.init_field).  Order: after load_pair and before any channel, the seed is bound to this pair and native grid
and its provenance read; a refusal ends the run init_field_rejected (exit 5; run directory with cmd.txt, git.txt, env.txt,
log.txt, timing.json, status.json).  An oracle seed (D0 arms A1, A3, A5; any source that saw truth labels, or was itself seeded by
one) needs --init-field-oracle-ok and is then flagged in every record, and bayesreg.eval gives it field_role oracle_seeded (not
compared by default, never averaged with others).  The native entry state is the first of s_k u_seed, s_k = .9^k, k = 0..6,
that passes the configuration C certificate (C12-2 without the prolongation); seven rejections -> initialization_failed
(exit 2), u = 0 is never used.  Levels 0..2 are not run (their pyramid arrays are built by C03.6 as always and dropped at
once); eta as without the seed (--eta-json or the C05-7 values; the seed does not enter them); sigma: the level-start M step at
the entry state (C11-3), then rounds[3] rounds; G0 at u = 0 (C13-2); sigma_domain != brain: the frozen support at the entry
state (C04-6); --max-accepted-per-level: only a3 acts.  Refused in parse_args (usage error, exit 2, no run directory):
one of the two without the other, --start-level other than 3, --init-field-oracle-ok without --init-field, a PATH that is
not a file, --max-levels < 4, --level-time-shares (no coarse level to share), --inner lm and --outer b (not implemented for
the warm start).  The learn CLI, the probe and the sweep tools have no such switch.
Records only under the switch (without it every record file is byte-identical to the code before WST under the measured
mask of tests/test_bayesreg/_golden/golden_c16ca4c_wst_records.json): config.json args init_field, start_level,
init_field_oracle_ok and top-level init_field (the seed record); solver (config.json, provenance.json, hyper.json algorithm)
start_level, levels_skipped, levels_skipped_reason; provenance.json init_field and switches init_field, start_level,
init_field_oracle; hyper.json init_field (with entry_rows, accepted_k, scale, entry_state_hash); log.jsonl one row of kind
init_field; qc.json init_field_oracle; timing.json init_field_load and init_field; status.json init_field; cert.json,
final_native.npz and disp.npz meta init_field (path, source_format, arm, oracle, state hashes).
Profile and checkpoints (C12-4a item 8, records only; bayesreg.solve.checkpoints): under --init-field, and under --checkpoint-accepted N
(every N-th accepted step of the native level) or --checkpoint-seconds S (the first accepted step S seconds after the last checkpoint;
needs --wall-seconds W with W/S >= 3), hyper.json carries `profile`: L (the profile objective) and its fields at the native entry and
end state, the checkpoint rows; log.jsonl rows of kind profile (entry) and checkpoint; status.json n_checkpoints.  A checkpoint is a
formal commit, written as <out>/checkpoints/ckpt_<kkk>/disp.npz (bayesreg.eval.cli reads it as kind disp, with --pair); with --checkpoint-*
the native entry state is written too, as <out>/checkpoints/entry/disp.npz (not a checkpoint).  A failure of these records lands in
hyper.json profile.errors; the run goes on.  L compares within one run only (C12-4a item 8).  One --checkpoint-*
at most; refused in parse_args (exit 2, no run directory): --checkpoint-* with --inner lm, --outer b or --max-levels < 4, and a
--checkpoint-seconds that cannot give 3 checkpoints in W.  The states and the decisions of the solve do not change.

OPT1 device switches (--backend torch64 only; 数值逐位不变；只影响显存与速度): --jacobian-cache on|off (default on; off =
TorchModelState jacobian_cache_slots 0), --jacobian-cache-log on|off (default off), --speculative-gradient on|off (default
on). A non-default value with another backend is refused. Recorded in config.json (args, opt1, backend
state_options_effective), provenance.json (switches; backend.state_options_effective) and hyper.json
(backend_after_solve.device_states: the provenance of every live TorchModelState -- since MEM only those of the last
level grid, as preparing a state on a new grid drops the device states of the other grids; FUP1, review MEM minor 3).
status.json device_anomalies: the
C08-8(e) device anomaly counts of the run (evaluate, compare; torch64 only, None otherwise); nonzero ones are also
logged as a warning. With a GPU shared with training, check before the start that the free device memory is at least
bayesreg.backends.torch64.estimate_peak_bytes + 2 GB; otherwise --jacobian-cache off (review OPT1_复核_20261030).  Call
it with the run's point set: closed27=True for gauss27+closed27 (the default run since DEF), corners=True for
gauss27+corners; its defaults (closed27=False, corners=False) estimate a gauss27 run and underestimate the default run
(MIN1, review DEF minor 4).

cpu_threads (CPUPAR; default 4 = bayesreg.geom.parallel.DEFAULT_CPU_THREADS, leaving cores and memory for a
concurrent run or training; every value is bitwise the same for every count): recorded in config.json, provenance.json
(resources) and, with the peak RSS of the process (resource.getrusage ru_maxrss, self and children), the user and
system CPU seconds, in status.json and timing.json (resources) at the end of the run.  env.txt and provenance.json env
carry memtrace_env: BAYESREG_MEMTRACE and BAYESREG_MEMTRACE_RESET_HWM as found ("未设" when unset); with the HWM reset
on, peak_rss_mb is not the run's peak (FUP1, review MEM minor 4).

Files (exp-log procedure + C16): cmd.txt, git.txt, env.txt, config.json, data.txt, provenance.json (written before the
solve), log.txt, log.jsonl, hyper.json (with backend_after_solve, the backend record after the solve),
qc.json, timing.json, status.json (with the export record, resources and structure_registry: the registration
state of the structure layer's registry with its git tracked / clean flags, None unless channel_source structure); when the
native level was reached also final_native.npz and cert.json (C16; a level_failed on the native level, an
exception inside a round included, still exports its last formal commit, C13-7) and -- only when the fresh
CPU certificate of the exported state passes (hard constraint 1) -- disp.npz; otherwise the run ends
final_certificate_rejected with final_native.npz / cert.json (rejected = true) kept as evidence.
With --max-levels < 4 the last level's commit goes to final_level<ell>.npz (development switch).
Exit codes: 0 run finished with a C13-3 ending, 2 run failure (level_failed, transfer_rejected,
initialization_failed, final_certificate_rejected) or stopped_before_native, 3 subject_sealed,
4 provenance_rejected, 5 init_field_rejected, 1 exception.
"""
import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import re
import resource
import shlex
import subprocess
import sys
import time
import traceback

import numpy as np

from jd_curl.validation import InputValidationError
from jd_curl.three_d.nifti import load_scalar

from .. import memtrace
from ..heap import release_free_heap
from ..geom.parallel import DEFAULT_CPU_THREADS
from ..io import ProvenanceRejected, SubjectSealed, load_pair, validate_segmentation_provenance, verify_segmentation_files
from ..io.pair import recheck_sealed_pair
from ..model import CLOSED27_SIMPSON, CLOSED27_WEIGHTS, LOGVOL_DEFAULT, LOGVOL_GAUSS27_CLOSED27, LOGVOL_POINTS
from ..model.support import (BRAIN, DEFAULT_SUPPORT_THRESHOLD, SIGMA_DOMAINS, check_theta, check_theta_coarse,
                             domain_record)
from ..seg import (brain_mask, channel_pyramid, normalize_intensity, segmentation_qc, simplex_from_background_class,
                   simplex_with_background, tissue_channels)
from ..seg.structure_layer import (DEFAULT_GROUPING, INFER_MODES, REPO_REGISTRY, SEGMENTER_ID, StructureLayer,
                                   StructureSource, registry_label, registry_status)
from ..seg.unet.groups import DEFAULT_GROUPS_PATH
from ..solve import (BACKENDS, CHANNEL_SOURCES, Caps, Eta, GaussianMStep, PGConfig, ProximalParams, SolverConfig,
                     default_lambda, eta_temporary, export_native, level_inputs, make_backend, solve_multilevel)
from ..solve.checkpoints import CheckpointConfig, ProfileRecorder
from ..solve.export import JsonlLog, dumps, grid_record, write_final_state, write_json
from ..solve.init_field import (INIT_FIELD_REJECTED, INIT_FIELD_REJECTED_WORDING, InitFieldRejected, load_init_field,
                                status_summary as init_field_summary)
from ..solve.lm import INNER_LM, INNER_PG, INNER_SOLVERS, LMConfig
from ..solve.jd_scaling import (AFFINE_V1, CELL_ESTIMATES as JD_CELL_ESTIMATES, CERTIFICATE_BERNSTEIN,
                                DEFAULT_CELL_ESTIMATE as JD_DEFAULT, MODES as JD_SCALING_MODES, NONE as JD_NONE,
                                OFF as JD_OFF, JDScalingConfig, j53_violations_logged, j53_violations_total)
from ..solve.stopping import (BUDGET_EXHAUSTED, CAP_REACHED, CONVERGED_B, FINAL_CERTIFICATE_REJECTED,
                              INITIALIZATION_FAILED, LEVEL_FAILED, LINE_SEARCH_FAILED, N_COARSE, OUTER_INNER_FAILED,
                              OUTER_NO_PROGRESS, OUTER_STAGNATED, PROVENANCE_REJECTED, RUN_WORDING, STAGNATED,
                              STOPPED_BEFORE_NATIVE, SUBJECT_SEALED, TRANSFER_REJECTED, TRIAL_BUDGET_EXHAUSTED,
                              LevelShares, WallBudget)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO/"data"/"three_d"
DEFAULT_ALLOWLIST = REPO/"configs"/"bayesreg"/"dev_subjects.json"
DEFAULT_TISSUE_ROOT = Path("/home/rmcfz/seq_vp_tag_results/seg/atropos_v1")
# Structure layer (docs/bayesreg/04_分割器.md, experiments/segnet/run_train.sh: $OUT): members under $OUT/fold<F>_s<seed>/,
# the staging registry $OUT/segmenters.json (until the lead registers into configs/bayesreg/segmenters.json),
# predictions of run_pair under $OUT/predictions/<subject>/.
DEFAULT_SEGMENTER_OUT = Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v1")
DEFAULT_STRUCTURE_ROOT = DEFAULT_SEGMENTER_OUT/"predictions"
DEFAULT_SEGMENTER_REGISTRY = DEFAULT_SEGMENTER_OUT/"segmenters.json"
DEFAULT_STRUCTURE_IMAGE_ROOT = Path("/mnt/d/data")
ENDINGS = (CONVERGED_B, CAP_REACHED, STAGNATED, OUTER_STAGNATED, LINE_SEARCH_FAILED, TRIAL_BUDGET_EXHAUSTED,
           OUTER_NO_PROGRESS, OUTER_INNER_FAILED, BUDGET_EXHAUSTED)
EXIT = {SUBJECT_SEALED: 3, PROVENANCE_REJECTED: 4, LEVEL_FAILED: 2, TRANSFER_REJECTED: 2, INITIALIZATION_FAILED: 2,
        STOPPED_BEFORE_NATIVE: 2, FINAL_CERTIFICATE_REJECTED: 2, INIT_FIELD_REJECTED: 5}
NATIVE_START_LEVEL = N_COARSE                 # C12-4a: the only --start-level (levels 0..N_COARSE - 1 are the coarse ones)
PACKAGES = ("numpy", "scipy", "numba", "nibabel", "torch", "antspyx", "pytest")
ON_OFF = ("on", "off")
OPT1_NOTE = "数值逐位不变；只影响显存与速度"
OPT1_DEFAULTS = dict(jacobian_cache="on", jacobian_cache_log="off", speculative_gradient="on")


def _positive_seconds(text):
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: a number of seconds") from None
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"{text!r}: a finite positive number of seconds")
    return value


def _positive_share(text):
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: a number") from None
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"{text!r}: beta_S must be finite and > 0 (C05-2b)")
    return value


PER_LEVEL_CAPS = re.compile(r"[0-9]+(,[0-9]+){3}")       # C13-4a grammar (whole string; ASCII digits only)
SCH_SOURCE = "_sch_max_accepted_source"     # args attribute (SCH switches only; never written to config.json args)


def _per_level_caps(text):
    """--max-accepted-per-level (C13-4a): the whole string matches ^[0-9]+(,[0-9]+){3}$ (four decimal integers >= 0)."""
    if not isinstance(text, str) or PER_LEVEL_CAPS.fullmatch(text) is None:
        raise argparse.ArgumentTypeError(f"{text!r}: four decimal integers >= 0, comma separated, nothing else "
                                         "(a0,a1,a2,a3; C13-4a)")
    return [int(part) for part in text.split(",")]


def _level_shares(text):
    """--level-time-shares (C13-5a): exactly three comma-separated items, each parsed by float(), finite, in (0, 1); the
    sum (left to right) < 1 and the --wall-seconds requirement are checked after parsing (resolve_schedule_args)."""
    parts = text.split(",")
    if len(parts) != N_COARSE:
        raise argparse.ArgumentTypeError(f"{text!r}: exactly {N_COARSE} shares s0,s1,s2 (C13-5a)")
    values = []
    for part in parts:
        try:
            value = float(part)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r}: {part!r} is not a number (C13-5a)") from None
        if not np.isfinite(value) or not 0 < value < 1:
            raise argparse.ArgumentTypeError(f"{text!r}: every share finite and in (0, 1) (C13-5a)")
        values.append(value)
    return values


class _NoteGiven(argparse.Action):
    """Store the value and note on the parser that the option was given (C13-4a: --max-accepted given or default)."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        parser.sch_given.add(self.dest)


class _NoteGivenLMI(argparse.Action):
    """Store the value and note on the parser that the option was given (LMI: --metric-jd-scaling refused under --inner
    lm when given; the stored value and the default are unchanged)."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        parser.lmi_given.add(self.dest)


def resolve_inner_args(parser, args):
    """LMI (C09.3) at argument parsing (before the run directory): --inner lm refuses --backend torch64, an explicit
    --metric-jd-scaling and an explicit --jd-cell-estimate (PG-only, C09.1-3a), and records both as "none"; --inner pg or
    no --inner leaves the PG resolution (resolve_jd_cell_estimate) exactly as before LMI."""
    if getattr(args, "inner", INNER_PG) != INNER_LM:
        args.jd_cell_estimate = resolve_jd_cell_estimate(parser, args.metric_jd_scaling, args.jd_cell_estimate)
        return args
    if args.backend != "numpy":
        parser.error("--inner lm: --backend numpy only (C08-9: the LM class of the GPU-path comparison has not passed "
                     "the L1 gate; LMI)")
    if "metric_jd_scaling" in getattr(parser, "lmi_given", set()):
        parser.error(f"--metric-jd-scaling {args.metric_jd_scaling}: no effect with --inner lm (the JD scaling of "
                     "C09.1-3a changes only the PG direction); leave it out (recorded as none)")
    if args.jd_cell_estimate is not None:
        parser.error(f"--jd-cell-estimate {args.jd_cell_estimate}: no effect with --inner lm (C09.1-3a, PG only); "
                     "leave it out (recorded as none)")
    args.metric_jd_scaling = JD_NONE
    args.jd_cell_estimate = JD_NONE
    return args


def inner_options(args):
    """The SolverConfig options of --inner: {} for pg or without it (SolverConfig as before LMI)."""
    if getattr(args, "inner", INNER_PG) != INNER_LM:
        return dict(pg=pg_config(args.metric_jd_scaling, args.jd_cell_estimate))
    return dict(inner=INNER_LM, lm=LMConfig())


def resolve_schedule_args(parser, args):
    """C13-4a / 5a at argument parsing (before the run directory): refuse --max-accepted with --max-accepted-per-level,
    shares without --wall-seconds, a share sum >= 1.  Without either new switch args is left exactly as before SCH (no
    new attribute: config.json args unchanged).  With one: args.max_accepted_per_level and args.level_time_shares both
    set (the one not given None), args.max_accepted None when per-level caps are given (C16 SCH), and the private
    SCH_SOURCE attribute (max_accepted_source) for the solver record."""
    per_level = getattr(args, "max_accepted_per_level", None)
    shares = getattr(args, "level_time_shares", None)
    if per_level is None and shares is None:
        return args
    given = getattr(parser, "sch_given", set())
    if per_level is not None and "max_accepted" in given:
        parser.error("--max-accepted and --max-accepted-per-level: give one of them (C13-4a: refused, no precedence "
                     "rule; --max-accepted N is --max-accepted-per-level N,N,N,N)")
    if shares is not None:
        if args.wall_seconds is None:
            parser.error("--level-time-shares: only with --wall-seconds (C13-5a)")
        try:
            LevelShares(tuple(shares), args.wall_seconds)
        except InputValidationError as exc:
            parser.error(f"--level-time-shares: {exc}")
    for name in ("max_accepted_per_level", "level_time_shares"):       # both at the end of args, in this order
        if hasattr(args, name):
            delattr(args, name)
    args.max_accepted_per_level = None if per_level is None else list(per_level)
    args.level_time_shares = None if shares is None else list(shares)
    if per_level is not None:
        args.max_accepted = None
    setattr(args, SCH_SOURCE, "max_accepted_per_level" if per_level is not None
            else "max_accepted" if "max_accepted" in given else "default")
    return args


def schedule_options(args):
    """The SolverConfig options of the SCH switches: {} without them (SolverConfig as before SCH)."""
    if not hasattr(args, SCH_SOURCE):
        return {}
    shares = args.level_time_shares
    return dict(level_time_shares=None if shares is None else LevelShares(tuple(shares), args.wall_seconds),
                max_accepted_source=getattr(args, SCH_SOURCE))


def solver_caps(args):
    """C13-4 / 4a: Caps(--max-accepted, --max-outer), or the per-level caps of --max-accepted-per-level."""
    per_level = getattr(args, "max_accepted_per_level", None)
    if per_level is None:
        return Caps(args.max_accepted, args.max_outer)
    return Caps.per_level(per_level, max_outer=args.max_outer)


def schedule_switches(args):
    """provenance.json switches of SCH (next to wall_seconds): {} without the new switches."""
    if not hasattr(args, SCH_SOURCE):
        return {}
    return dict(max_accepted_per_level=args.max_accepted_per_level, level_time_shares=args.level_time_shares)


def args_record(args):
    """config.json args: vars(args) without the private (underscore) attributes (SCH_SOURCE)."""
    return {key: value for key, value in vars(args).items() if not key.startswith("_")}


def resolve_init_field_args(parser, args):
    """C12-4a item 6 at argument parsing (before the run directory): without the switches args is left exactly as before
    WST (no new attribute: config.json args unchanged); with them init_field, start_level and init_field_oracle_ok are all
    set, at the end of args, in this order."""
    path = getattr(args, "init_field", None)
    start = getattr(args, "start_level", None)
    oracle_ok = getattr(args, "init_field_oracle_ok", False)
    if path is None and start is None and not oracle_ok:
        return args
    if path is None or start is None:
        parser.error("--init-field and --start-level go together; --init-field-oracle-ok only with them (C12-4a)")
    if start != NATIVE_START_LEVEL:
        parser.error(f"--start-level {start}: only {NATIVE_START_LEVEL}, a native seed enters at the native level (C12-4a)")
    if not Path(path).is_file():
        parser.error(f"--init-field {path}: not a file")
    if args.max_levels != 4:
        parser.error(f"--max-levels {args.max_levels}: --init-field runs the native level (C12-4a)")
    if args.outer != "a":
        parser.error("--init-field: option (a) only (C12-4a item 6; option (b) is not implemented for the warm start)")
    if getattr(args, "inner", INNER_PG) != INNER_PG:
        parser.error("--init-field: --inner pg only (C12-4a item 6; lm is not implemented for the warm start)")
    if getattr(args, "level_time_shares", None) is not None:
        parser.error("--level-time-shares: no coarse level runs with --init-field (C12-4a item 6)")
    for name in ("init_field", "start_level", "init_field_oracle_ok"):
        if hasattr(args, name):
            delattr(args, name)
    args.init_field, args.start_level, args.init_field_oracle_ok = str(path), start, bool(oracle_ok)
    return args


def _positive_int(text):
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: a positive integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r}: a positive integer")
    return value


def resolve_checkpoint_args(parser, args):
    """C12-4a item 8 at argument parsing: without the switches args is left as before WST (no new attribute)."""
    accepted = getattr(args, "checkpoint_accepted", None)
    seconds = getattr(args, "checkpoint_seconds", None)
    if accepted is None and seconds is None:
        return args
    if accepted is not None and seconds is not None:
        parser.error("--checkpoint-accepted and --checkpoint-seconds: give one of them (C12-4a item 8)")
    if args.max_levels != 4 or args.outer != "a" or getattr(args, "inner", INNER_PG) != INNER_PG:
        parser.error("--checkpoint-*: the native level of option (a) with --inner pg only (C12-4a item 8)")
    if seconds is not None and (args.wall_seconds is None or args.wall_seconds/seconds < 3):
        parser.error(f"--checkpoint-seconds {seconds:g}: needs --wall-seconds W with W/S >= 3, at least 3 checkpoints (C12-4a item 8)")
    return args


def checkpoint_config(args):
    """The CheckpointConfig of the switches, or None."""
    accepted, seconds = getattr(args, "checkpoint_accepted", None), getattr(args, "checkpoint_seconds", None)
    return None if accepted is None and seconds is None else CheckpointConfig(every_accepted=accepted, every_seconds=seconds)


def init_field_options(args):
    """The SolverConfig options of --start-level and --checkpoint-*: {} without them (SolverConfig as before WST)."""
    options = dict(start_level=args.start_level) if getattr(args, "init_field", None) is not None else {}
    config = checkpoint_config(args)
    return options if config is None else dict(options, checkpoint=config)


def init_field_switches(seed):
    """provenance.json switches of C12-4a: {} without a seed."""
    return {} if seed is None else dict(init_field=seed.record["path"], start_level=NATIVE_START_LEVEL,
                                        init_field_oracle=seed.oracle)


def _support_threshold(text):
    """--support-threshold: the coarse check of C04-6 (finite, 0 < theta < 1); the floor bound needs K (resolve)."""
    try:
        return check_theta_coarse(float(text))
    except (ValueError, InputValidationError) as exc:
        raise argparse.ArgumentTypeError(f"{text!r}: {exc}") from None


def resolve_sigma_domain_args(parser, args):
    """C04-6 at argument parsing (before the run directory): sigma_domain (default brain) and theta (brain: an explicit
    one refused, recorded null; otherwise default .05 and the exact floor bound with K known before any image is read --
    structure: K_S of the grouping; tissue: the K-independent bound K = 1, see the module docstring).  Sets
    args.sigma_domain and args.support_threshold to the values in effect."""
    domain = BRAIN if args.sigma_domain is None else args.sigma_domain
    theta = args.support_threshold
    if domain == BRAIN:
        if theta is not None:
            parser.error(f"--support-threshold {theta}: not with --sigma-domain brain (C04-6: refused so that the record "
                         "does not mislead; recorded as null)")
    else:
        if args.channel_source == "none":
            parser.error(f"--sigma-domain {domain}: no structure channel with --channel-source none (C04-6: the switch "
                         "would have no effect)")
        theta = DEFAULT_SUPPORT_THRESHOLD if theta is None else theta
        if args.channel_source == "structure":
            from ..seg.unet.groups import load_grouping
            try:
                k = load_grouping(args.structure_groups, args.structure_grouping).k
            except (OSError, ValueError, InputValidationError) as exc:
                parser.error(f"--sigma-domain {domain}: K_S of --structure-grouping is needed for the --support-threshold "
                             f"bound (C04-6): {exc}")
        else:
            k = 1                          # tissue: K_T unknown before the run directory; max_K eps/(1 + (K + 1) eps)
        try:
            theta = check_theta(theta, k)
        except InputValidationError as exc:
            parser.error(f"--support-threshold: {exc}")
    args.sigma_domain, args.support_threshold = domain, theta
    return args


def sigma_domain_options(args):
    """The SolverConfig options of --sigma-domain / --support-threshold ({} under brain: SolverConfig as before)."""
    domain = getattr(args, "sigma_domain", None) or BRAIN
    if domain == BRAIN:
        return {}
    return dict(sigma_domain=domain, support_threshold=args.support_threshold)


def sigma_domain_switches(config):
    """provenance.json switches of C04-6: sigma_domain, support_threshold, support_rule (null under brain)."""
    return domain_record(config.sigma_domain, config.support_threshold)


def sigma_policy_record(config, sigma):
    """The sigma policy record of config.json (sigma) and hyper.json (sigma_policy): the GaussianMStep record under brain
    (unchanged), the SupportMStep record (C11-8) otherwise."""
    if config.sigma_domain == BRAIN:
        return sigma.record()
    from ..solve.multilevel import SupportMStep
    return SupportMStep.from_policy(sigma, config.sigma_domain).record()


def make_budget(seconds):
    """The run's WallBudget (counted from now) or None (no --wall-seconds)."""
    return None if seconds is None else WallBudget(seconds)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m bayesreg.cli.run_pair", description=__doc__.split("\n\n")[0])
    parser.sch_given = set()                   # options noted by _NoteGiven (C13-4a: --max-accepted given or default)
    parser.lmi_given = set()                   # options noted by _NoteGivenLMI (LMI: --metric-jd-scaling given)
    parser.add_argument("--pair", required=True, help="<stage>/<pair>, e.g. stage_L2R_val_v1/OASIS_..__OASIS_..")
    parser.add_argument("--channel-source", required=True, choices=CHANNEL_SOURCES)
    parser.add_argument("--out", required=True, help="run directory (must not exist or be empty)")
    parser.add_argument("--max-levels", type=int, default=4, help="run levels 0..N-1 (development switch; < 4 "
                        "ends with stopped_before_native and no disp.npz)")
    parser.add_argument("--outer", choices=("a", "b"), default="a", help="C10 outer option")
    parser.add_argument("--rounds", default="1,1,1,3", help="EM rounds per level (C11-3, D15)")
    parser.add_argument("--max-accepted", type=int, default=500, action=_NoteGiven,
                        help="C13-4 accepted steps per (level, round)")
    parser.add_argument("--max-outer", type=int, default=60, help="C13-4 outer commits per (level, round), option (b)")
    parser.add_argument("--backend", choices=BACKENDS, default="numpy", help="numpy (CPU FP64 reference) or torch64 "
                        "(device FP64 with the C08-8 GPU-path comparison; certificate on the CPU either way)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None, help="torch64 device (default: cuda when "
                        "available)")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--allowlist", default=str(DEFAULT_ALLOWLIST))
    parser.add_argument("--tissue-root", default=str(DEFAULT_TISSUE_ROOT))
    parser.add_argument("--eta-json", default=None, help='{"alpha": .., "nu": .., "tau": [tau_0, .., tau_K_T]}')
    parser.add_argument("--cpu-threads", type=int, default=DEFAULT_CPU_THREADS, help="threads of the CPU FP64 "
                        f"reference (CPUPAR; default {DEFAULT_CPU_THREADS}, leaving room for a concurrent run; every "
                        "value is bitwise the same for every count; memory grows with the count, see status.json "
                        "resources)")
    parser.add_argument("--metric-jd-scaling", choices=JD_SCALING_MODES, default=AFFINE_V1, action=_NoteGivenLMI,
                        help="C09.1-3a JD affine-scaling PG metric (path only): affine_v1 (default) or off (the C09.1 "
                        "metric, bitwise); refused with --inner lm (recorded as none)")
    parser.add_argument("--jd-cell-estimate", choices=JD_CELL_ESTIMATES, default=None,
                        help="C09.1-3a cell estimate m_K of the JD scaling (path only, D47): certificate (default "
                        "since DEF: min(m_K, per-cell certified lower bound of the accepted state)), gauss27+corners (the "
                        "default before DEF), closed27 (adds the {0, 1/2, 1}^3 points) or certificate_bernstein (D53, "
                        "not the default: hull cells below J_act take max(b_K, prescreen bound)). No effect with "
                        "--metric-jd-scaling off: not given -> recorded as none; given -> refused. To reproduce a run "
                        "before DEF: with the metric on (affine_v1), give both switches it did not give with their "
                        "old values (--logvol-points gauss27, --jd-cell-estimate gauss27+corners); with "
                        "--metric-jd-scaling off only --logvol-points gauss27 (this switch is refused there); see the "
                        "module docstring")
    parser.add_argument("--logvol-points", choices=LOGVOL_POINTS, default=LOGVOL_DEFAULT,
                        help="point set of the log-volume prior (changes F_B): gauss27+closed27 (default since DEF, "
                        "C05-2b: Gauss27 and closed27 rules by shares; BZ adopted (A') 10-30), gauss27 (C05-2, the "
                        "default before DEF) or gauss27+corners (C05-2a: adds the 8 cell corners, beta_c = 1). The "
                        "default run before DEF, bit for bit: --logvol-points gauss27 --jd-cell-estimate "
                        "gauss27+corners (both; with --metric-jd-scaling off, --logvol-points gauss27 alone)")
    parser.add_argument("--logvol-closed27-share", dest="logvol_closed27_share", type=_positive_share, default=1.,
                        help="C05-2b share beta_S > 0 (Gauss27 1/(1 + beta_S), closed27 beta_S/(1 + beta_S); "
                        "default 1); only with --logvol-points gauss27+closed27")
    parser.add_argument("--closed27-weights", dest="closed27_weights", choices=CLOSED27_WEIGHTS,
                        default=CLOSED27_SIMPSON,
                        help="C05-2b / D48 closed27 rule: simpson (default, (1/6, 2/3, 1/6)^3) or uniform (1/27; "
                        "ablation switch, Laplace and EB refuse it); only with --logvol-points gauss27+closed27")
    parser.add_argument("--witness-estimate", choices=ON_OFF, default="off",
                        help="diagnostic beyond the contract (default off): add an estimate of the witness cell's "
                        "minimum J (dense sampling + L-BFGS-B, not a bound) to every JACOBIAN_VIOLATION record; costs "
                        "time per rejection.  The contract witness fields are always recorded")
    parser.add_argument("--sigma-domain", choices=SIGMA_DOMAINS, default=None,
                        help="C04-6 / C11-8 (draft SWL; D49 for BZ): estimation domain of the structure-channel noise "
                        "variances: brain (default; bitwise the code before SWLI), support (one sigma_c^2 per channel on "
                        "its frozen support) or support_shared (one sigma_S^2 on the supports); changes F_B")
    parser.add_argument("--support-threshold", type=_support_threshold, default=None,
                        help="C04-6 / D50 support threshold theta (default .05 with support / support_shared; refused "
                        "with brain): finite, 0 < theta < 1, above the floor eps/(1 + (K + 1) eps)")
    parser.add_argument("--wall-seconds", type=_positive_seconds, default=None,
                        help="development switch (C13-3 appendix, D44): wall-clock budget in seconds from the start of "
                        "the run, checked at safe points; native level -> budget_exhausted (预算收尾, last formal commit "
                        "exported and certified), before it -> stopped_before_native (default: none)")
    parser.add_argument("--max-accepted-per-level", type=_per_level_caps, default=argparse.SUPPRESS,
                        help="C13-4a (SCH, development runs): accepted-step caps a0,a1,a2,a3 of every (level, round) of "
                        "levels 0..3 (decimal integers >= 0); not with --max-accepted (default: --max-accepted on "
                        "every level)")
    parser.add_argument("--level-time-shares", type=_level_shares, default=argparse.SUPPRESS,
                        help="C13-5a (SCH, development switch, needs --wall-seconds W): shares s0,s1,s2 of W; coarse "
                        "level ell ends level_budget_reached (粗层预算转层) after (s0 + ... + s_ell) W s, checked at PG "
                        "and outer iteration tops; 0 < s < 1, s0 + s1 + s2 < 1 (default: none)")
    parser.add_argument("--inner", choices=INNER_SOLVERS, default=argparse.SUPPRESS,
                        help="C09 inner solver (LMI, path only): pg (default; PG + BB2 + Armijo, the code before LMI) or "
                        "lm (C09.3 LM-PCG; --backend numpy only; --metric-jd-scaling / --jd-cell-estimate refused, "
                        "recorded as none)")
    parser.add_argument("--init-field", default=argparse.SUPPRESS,
                        help="C12-4a (WST, path only, default off): the final_native.npz of a finished run on this "
                        "pair (run_pair or the frozen configuration C solver) as the native entry state, with the C12-2 "
                        ".9^k back-off; needs --start-level 3 (default: none, the run starts at u = 0)")
    parser.add_argument("--start-level", type=int, default=argparse.SUPPRESS,
                        help="C12-4a: 3, the native level; levels 0..2 are not run; only with --init-field")
    parser.add_argument("--init-field-oracle-ok", action="store_true", default=argparse.SUPPRESS,
                        help="C12-4a: admit a seed from a run that used truth labels (D0 arms A1, A3, A5); every record then "
                        "carries the oracle flag and the result is not a method result. Default: refused")
    parser.add_argument("--checkpoint-accepted", type=_positive_int, default=argparse.SUPPRESS,
                        help="C12-4a item 8 (records only): write every N-th accepted step of the native level as "
                        "<out>/checkpoints/ckpt_<kkk>/disp.npz and its entry state as <out>/checkpoints/entry/disp.npz, with L in "
                        "hyper.json. Default: none")
    parser.add_argument("--checkpoint-seconds", type=_positive_seconds, default=argparse.SUPPRESS,
                        help="C12-4a item 8: the same at the first accepted step S seconds after the last checkpoint; needs "
                        "--wall-seconds W with W/S >= 3. Default: none")
    opt1 = parser.add_argument_group("torch64 device memory (OPT1; " + OPT1_NOTE + ")")
    opt1.add_argument("--jacobian-cache", choices=ON_OFF, default="on",
                      help="OPT1 C2a: keep the J that evaluate computed (base and newest trial) for the C08-8 device "
                      "comparison; off = jacobian_cache_slots 0 (about 1.9 GB less per slot at the native size with "
                      f"corners). {OPT1_NOTE}. --backend torch64 only (default on)")
    opt1.add_argument("--jacobian-cache-log", choices=ON_OFF, default="off",
                      help="OPT1 C2a: also keep log J (saves two torch.log per compared chunk, doubles the cache). "
                      f"{OPT1_NOTE}. --backend torch64 only, needs --jacobian-cache on (default off)")
    opt1.add_argument("--speculative-gradient", choices=ON_OFF, default="on",
                      help="OPT1 C3: evaluate also scatters the gradient flux (8 B x 3 N per live evaluation), so the "
                      f"gradient of an accepted step needs no geometry pass. {OPT1_NOTE}. --backend torch64 only "
                      "(default on)")
    structure = parser.add_argument_group("structure layer (--channel-source structure)")
    structure.add_argument("--structure-root", default=str(DEFAULT_STRUCTURE_ROOT),
                           help="predictions <root>/<subject>/ (structure_groups.npz, provenance.json); missing ones "
                           "are generated here")
    structure.add_argument("--segmenter-registry", default=None,
                           help="segmenters.json: default the repository configs/bayesreg/segmenters.json "
                           "(registered); "
                           f"with --structure-registry-staging the staging copy {DEFAULT_SEGMENTER_REGISTRY}")
    structure.add_argument("--structure-registry-staging", action="store_true",
                           help="admit a registry other than the repository one (the staging copy of the training "
                           "chain, 未正式登记): development diagnostics only, recorded as staging_unregistered in "
                           "provenance.json and status.json and flagged by bayesreg.eval; ensemble.json beside it must "
                           "carry the registry's sha256 and member_check.passed true. Default: refused")
    structure.add_argument("--segmenter-models-root", default=str(DEFAULT_SEGMENTER_OUT),
                           help="member.pt files under <root>/*/ (inference only)")
    structure.add_argument("--structure-image-root", default=str(DEFAULT_STRUCTURE_IMAGE_ROOT),
                           help="<root>/<subject>/aligned_norm.nii.gz (inference only)")
    structure.add_argument("--structure-groups", default=str(DEFAULT_GROUPS_PATH), help="grouping table")
    structure.add_argument("--structure-grouping", default=DEFAULT_GROUPING,
                           help="grouping name (status default / ablation / verified only)")
    structure.add_argument("--segmenter-id", default=SEGMENTER_ID)
    structure.add_argument("--structure-infer", choices=INFER_MODES, default="never",
                           help="never (default): refuse a missing prediction; missing: generate absent predictions "
                           "(torch; 会占 GPU unless --structure-device cpu, 计时运行不要用)")
    structure.add_argument("--structure-device", choices=("cpu", "cuda"), default=None,
                           help="inference device (default: cuda when available; 会占 GPU，计时运行不要用 -- run the "
                           "inference as a separate step or use cpu)")
    structure.add_argument("--structure-synthetic", action="store_true",
                           help="tests / dry runs: inference admits only synthetic ids (>= OASIS_OAS1_0900_MR1) and a "
                           "non-real image root instead of the pinned repository allowlist")
    args = parser.parse_args(argv)
    resolve_inner_args(parser, args)           # LMI; without --inner lm: resolve_jd_cell_estimate as before
    if args.logvol_points != LOGVOL_GAUSS27_CLOSED27 and (args.logvol_closed27_share != 1.
                                                          or args.closed27_weights != CLOSED27_SIMPSON):
        parser.error("--logvol-closed27-share / --closed27-weights: only with --logvol-points gauss27+closed27 "
                     "(C05-2b)")
    if args.segmenter_registry is None:
        args.segmenter_registry = str(DEFAULT_SEGMENTER_REGISTRY if args.structure_registry_staging else REPO_REGISTRY)
    if args.backend != "torch64":
        changed = [f"--{name.replace('_', '-')} {getattr(args, name)}" for name, value in OPT1_DEFAULTS.items()
                   if getattr(args, name) != value]
        if changed:
            parser.error(f"{', '.join(changed)}: OPT1 device switches apply to --backend torch64 only")
    if args.jacobian_cache == "off" and args.jacobian_cache_log == "on":
        parser.error("--jacobian-cache-log on needs --jacobian-cache on")
    resolve_sigma_domain_args(parser, args)
    resolve_schedule_args(parser, args)
    resolve_init_field_args(parser, args)
    resolve_checkpoint_args(parser, args)
    return args


def resolve_jd_cell_estimate(parser, metric_jd_scaling, given):
    """The jd_cell_estimate in effect (C09.1-3a last item, D47; DEF): under --metric-jd-scaling off the option has no
    effect -- not given: "none"; given (any value): parser.error (免得记录误导); otherwise the given value or the
    default "certificate"."""
    if metric_jd_scaling == JD_OFF:
        if given is not None:
            parser.error(f"--jd-cell-estimate {given}: no effect with --metric-jd-scaling off; leave it out (recorded "
                         "as none) (C09.1-3a)")
        return JD_NONE
    return JD_DEFAULT if given is None else given


def pg_config(metric_jd_scaling, jd_cell_estimate):
    """PGConfig of the resolved options: no JDScalingConfig under "off" (jd_cell_estimate "none")."""
    if metric_jd_scaling == JD_OFF:
        return PGConfig(metric_jd_scaling=metric_jd_scaling)
    return PGConfig(metric_jd_scaling=metric_jd_scaling, jd=JDScalingConfig(cell_estimate=jd_cell_estimate))


def device_state_options(args):
    """The TorchModelState options of the OPT1 switches (--backend torch64): --jacobian-cache off ->
    jacobian_cache_slots 0 (on: the TorchModelState default), --jacobian-cache-log, --speculative-gradient.
    Bitwise neutral (数值逐位不变；只影响显存与速度); the effective values are in the backend record."""
    options = dict(jacobian_cache_log=args.jacobian_cache_log == "on",
                   speculative_gradient=args.speculative_gradient == "on")
    if args.jacobian_cache == "off":
        options["jacobian_cache_slots"] = 0
    return options


def backend_of(args):
    """make_backend of the run's arguments (the OPT1 options for torch64; and the run's --logvol-points, so that the
    torch64 record written before the solve -- config.json, provenance.json backend.comparison -- describes the run's
    point set, not the model default: a record only, MIN1, review DEF minor 1)."""
    estimate = getattr(args, "witness_estimate", "off") == "on"
    if args.backend == "torch64":
        return make_backend(args.backend, device=args.device, state_options=device_state_options(args),
                            witness_estimate=estimate, logvol_points=getattr(args, "logvol_points", None))
    return make_backend(args.backend, device=args.device, witness_estimate=estimate)


def closed27_switches(config):
    """provenance.json switches of C05-2b: beta_S and the closed27 rule, null unless logvol_points is gauss27+closed27
    (review IMPL27 suggestion: a gauss27 run does not record values that do not act)."""
    closed = config.logvol_points == LOGVOL_GAUSS27_CLOSED27
    return dict(logvol_closed27_share=config.logvol_closed27_share if closed else None,
                closed27_weights=config.closed27_weights if closed else None)


def opt1_switches(args):
    """provenance.json switches: the three OPT1 values as given, and whether they are in effect (torch64)."""
    return dict(jacobian_cache=args.jacobian_cache, jacobian_cache_log=args.jacobian_cache_log,
                speculative_gradient=args.speculative_gradient, opt1_in_effect=args.backend == "torch64",
                opt1_note=OPT1_NOTE)


def structure_source(args, command=()):
    """The StructureSource of the run's arguments (channel_source structure)."""
    return StructureSource(root=Path(args.structure_root), registry_path=Path(args.segmenter_registry),
                           allowlist_path=Path(args.allowlist), models_root=Path(args.segmenter_models_root),
                           image_root=Path(args.structure_image_root), groups_path=Path(args.structure_groups),
                           grouping=args.structure_grouping, segmenter_id=args.segmenter_id, infer=args.structure_infer,
                           device=args.structure_device, synthetic=args.structure_synthetic, command=tuple(command),
                           allow_staging=args.structure_registry_staging)


def structure_registry_label(args):
    """status.json structure_registry (review PS item 2): the registration state of the structure layer's registry
    (registered / staging_unregistered) and its git tracked / clean flags (review PS1_PS2 item b), None unless
    channel_source structure. The full record (sha256, staging ensemble check, git state) is in provenance.json
    channels.structure.registry."""
    if args.channel_source != "structure":
        return None
    return registry_label(dict(registry_status(args.segmenter_registry), staging_opt_in=args.structure_registry_staging),
                          path=args.segmenter_registry)


# ---- run directory bookkeeping (exp-log procedure) --------------------------------------------------------------

def _prepare_run_dir(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"{path}: the run directory exists and is not empty (nothing is overwritten)")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_logger(path):
    logger = logging.getLogger(f"bayesreg.run_pair.{path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in (logging.FileHandler(path/"log.txt", encoding="utf-8"), logging.StreamHandler(sys.stderr)):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logging.captureWarnings(True)
    return logger


def _close_logger(logger):
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False).stdout
    except OSError as exc:
        return f"(git unavailable: {exc})"


def code_record():
    head = _git("rev-parse", "HEAD").strip()
    status = _git("status", "--porcelain")
    return dict(repo=str(REPO), head=head, dirty=bool(status.strip()), status=status)


def jd_curl_snapshot():
    """sha256 of every jd_curl/*.py file (C16 'jd_curl 快照 SHA 清单')."""
    root = REPO/"jd_curl"
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*.py"))}


MEMTRACE_UNSET = "未设"


def memtrace_env():
    """The memory-trace switches of bayesreg.memtrace as found in the environment (FUP1, review MEM minor 4): the raw
    values of BAYESREG_MEMTRACE (the marks file; empty = off; on, every mark also resets torch's CUDA peak statistics)
    and BAYESREG_MEMTRACE_RESET_HWM ("1" with the marks on: VmHWM is reset at every mark, so timing.json / status.json
    peak_rss_mb (ru_maxrss) no longer covers the run), "未设" (unset) when absent.  A record only (env.txt,
    provenance.json env)."""
    return {name: os.environ.get(name, MEMTRACE_UNSET) for name in (memtrace.ENV, memtrace.ENV_RESET)}


def env_record(backend=None):
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    threads = {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                              "NUMBA_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(), cpu=cpu,
                packages=versions, thread_env=threads, memtrace_env=memtrace_env(),
                longdouble_nmant=int(np.finfo(np.longdouble).nmant),
                backend=None if backend is None else backend.name,
                device="cpu" if backend is None else backend.device,
                gpu_used=backend is not None and backend.device == "cuda",
                device_info=None if backend is None or not hasattr(backend, "dev") else backend.dev.describe())


def resource_record(cpu_threads):
    """Thread count of the CPU reference and the resources of this process so far: peak resident set size
    (resource.getrusage ru_maxrss, KiB on Linux, of this process and of its waited-for children) and CPU seconds."""
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(cpu_threads=cpu_threads, peak_rss_mb=own.ru_maxrss/1024, peak_rss_children_mb=children.ru_maxrss/1024,
                user_seconds=own.ru_utime, system_seconds=own.ru_stime,
                source="resource.getrusage(RUSAGE_SELF | RUSAGE_CHILDREN).ru_maxrss / 1024 (Linux: KiB -> MiB), "
                       "peak over the process lifetime up to this record")


def _write_text(path, text):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text, encoding="utf-8")


# ---- inputs ---------------------------------------------------------------------------------------------

def resolve_pair(data_root, pair):
    parts = Path(pair).parts
    if len(parts) != 2 or any(p in (".", "..") for p in parts):
        raise InputValidationError("--pair: <stage>/<pair> required")
    return Path(data_root)/parts[0]/parts[1]/"pair.npz"


def _same_grid(a, b):
    return a.shape == b.shape and a.unit == b.unit and all(np.array_equal(x, y) for x, y in zip(a.axes, b.axes))


def load_tissue_layer(pair, side, tissue_root, *, allowlist=None):
    """C03.1-2/3: the Atropos posteriors of one image as (K_T, nz, ny, nx) with the accepted provenance.

    The record must pass validate_segmentation_provenance (layer tissue) and verify_segmentation_files;
    posterior_files must be posteriors01..0K (Atropos numbers from 1: file k+1 is class index k), each
    listed in output_sha256; every NIfTI must lie on the pair's grid exactly.
    `allowlist` (C23.4-8, D103): for a pair with a sealed subject the very bound allowlist load_pair admitted it
    through, checked before any file is read; None for every other pair (as before)."""
    recheck_sealed_pair(pair, allowlist, where=f"tissue layer ({side})")
    subject = pair.fixed_subject if side == "fixed" else pair.moving_subject
    directory = Path(tissue_root)/subject
    path = directory/"provenance.json"
    if not path.is_file():
        raise ProvenanceRejected(f"{side}: no tissue-layer provenance at {path}")
    prov = json.loads(path.read_text(encoding="utf-8"))
    accepted = validate_segmentation_provenance(prov, pair, side=side, layer="tissue", allowlist=allowlist)
    verified = verify_segmentation_files(prov, directory)
    k_t = len(accepted["class_names"])
    expected = [f"posteriors{k+1:02d}.nii.gz" for k in range(k_t)]
    if prov.get("posterior_files") != expected or prov.get("posterior_file_index_base") != 1 \
            or not set(expected) <= set(verified):
        raise ProvenanceRejected(f"{side}: posterior_files must be {expected} (index base 1), each hash-verified")
    channels = []
    for name in expected:
        volume = load_scalar(directory/name)
        if not _same_grid(volume.grid, pair.grid):
            raise ProvenanceRejected(f"{side}: {name} is not on the pair's grid")
        if volume.metadata["sha256"] != verified[name]:
            raise ProvenanceRejected(f"{side}: {name} changed after verification")
        channels.append(volume.values)
    record = dict(accepted=dict(accepted), directory=str(directory), provenance_sha256=hashlib.sha256(
        path.read_bytes()).hexdigest(), verified_outputs=verified, posterior_files=expected)
    return np.stack(channels), accepted, record


def structure_ptilde(layer, side):
    """C03.1-2 + C03.2-4a + C03.5 for one image: P_S of the registered segmenter (StructureLayer.load: provenance,
    D24, hashes, read-back with the raw channel-sum check) -> simplex_from_background_class (|sum - 1| <= 1e-6, else
    provenance_rejected) -> smooth -> floor. Returns (ptilde (K_S + 1 channels), accepted, record)."""
    p, accepted, record = layer.load(side)
    try:
        pbar, simplex = simplex_from_background_class(p)
    except InputValidationError as exc:
        raise ProvenanceRejected(f"structure layer ({side}): C03.2-4a: {exc}") from None
    del p
    ptilde, smoothing = tissue_channels(pbar)
    del pbar
    return ptilde, accepted, dict(record, simplex=simplex, smoothing=smoothing)


def read_eta(path, k_t):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    tau = document["tau"]
    if len(tau) != k_t+1:
        raise InputValidationError(f"--eta-json: tau needs K_T + 1 = {k_t+1} values (tau_0 first)")
    return Eta(document["alpha"], document["nu"], tuple(float(t)**2 for t in tau), source=f"file {path}")


def prepare_channels(pair, channel_source, tissue_root, logger, structure=None):
    """C03.2-C03.6 for one pair: returns (LevelInput tuple, fixed tissue ptilde, mask, provenance dict).

    `structure`: the StructureSource of channel_source "structure" (ignored otherwise). The structure layer is
    checked (grouping, registry, D24 preflight) before any tissue or structure array is read."""
    layer = None
    if channel_source == "structure":
        if not isinstance(structure, StructureSource):
            raise InputValidationError("channel_source 'structure' needs a StructureSource")
        layer = StructureLayer(pair, structure, log=logger.info)
    p_f, acc_f, rec_f = load_tissue_layer(pair, "fixed", tissue_root)
    p_m, acc_m, rec_m = load_tissue_layer(pair, "moving", tissue_root)
    if acc_f["class_names"] != acc_m["class_names"]:
        raise ProvenanceRejected("tissue layer: fixed and moving class_names differ")
    pbar_f, simplex_f = simplex_with_background(p_f)
    pbar_m, simplex_m = simplex_with_background(p_m)
    del p_f, p_m
    mask_f, mask_prov_f = brain_mask(pbar_f)
    mask_m, mask_prov_m = brain_mask(pbar_m)
    fixed, norm_f = normalize_intensity(pair.fixed, pbar_f, mask_f, acc_f)
    moving, norm_m = normalize_intensity(pair.moving, pbar_m, mask_m, acc_m)
    qc = segmentation_qc(pbar_f, mask_f)
    pt_f, smooth_prov = tissue_channels(pbar_f)
    pt_m, _ = tissue_channels(pbar_m)
    del pbar_f, pbar_m
    memtrace.mark("channels.tissue")
    structure_record = None if channel_source != "tissue" else "P_S := P_T (label-free ablation arm, C03.1-3)"
    if layer is not None:
        structure_f, sacc_f, srec_f = structure_ptilde(layer, "fixed")
        structure_m, sacc_m, srec_m = structure_ptilde(layer, "moving")
        if sacc_f["class_names"] != sacc_m["class_names"] or structure_f.shape != structure_m.shape:
            raise ProvenanceRejected("structure layer: fixed and moving class_names differ")
        structure_record = dict(layer.record(), fixed=srec_f, moving=srec_m)
        memtrace.mark("channels.structure")
    else:
        structure_f, structure_m = pt_f, pt_m      # tissue: P_S := P_T; none: unused by the model
    levels, pyramid = channel_pyramid(pair.grid, fixed=fixed, moving=moving, tissue_fixed=pt_f, tissue_moving=pt_m,
                                      structure_fixed=structure_f, structure_moving=structure_m, mask=mask_f)
    memtrace.mark("channels.pyramid")
    if layer is not None:
        structure_record["background_channel"] = pyramid["background_channel"]    # D2, as the pyramid used it
    k_s = None if channel_source == "none" else structure_f.shape[0]-1
    del structure_f, structure_m, fixed, moving, pt_m
    # MEM (memory only, same values): level by level, each ChannelLevel dropped as soon as its LevelInput exists (the
    # pyramid's square-root and structure stacks are not all alive next to the (14, N) channel stacks)
    levels = list(levels)
    inputs = []
    for k in range(len(levels)):
        level, levels[k] = levels[k], None
        inputs.append(level_inputs((level,), channel_source)[0])
        del level
    inputs = tuple(inputs)
    del levels
    memtrace.mark("channels.level_inputs")
    logger.info("channels: K_T = %d, K_S = %s, %d model channels, mask %d nodes, levels %s", pt_f.shape[0]-1, k_s,
                inputs[0].fixed.shape[0], int(mask_f.sum()), [lv.grid.shape for lv in inputs])
    provenance = dict(tissue=dict(fixed=rec_f, moving=rec_m), structure=structure_record,
                      simplex=dict(fixed=simplex_f, moving=simplex_m), mask=dict(fixed=mask_prov_f, moving=mask_prov_m),
                      normalization=dict(fixed=norm_f, moving=norm_m), qc=qc, smoothing=smooth_prov, pyramid=pyramid,
                      K_T=pt_f.shape[0]-1, K_S=k_s)
    return inputs, pt_f, mask_f, provenance


# ---- main ------------------------------------------------------------------------------------------------

def run(args, out, logger, timing, budget=None):
    pair_path = resolve_pair(args.data_root, args.pair)
    tick = time.perf_counter()
    try:
        pair = load_pair(pair_path, allowlist=args.allowlist)
    except SubjectSealed as exc:
        logger.error("subject_sealed: %s", exc)
        return dict(status=SUBJECT_SEALED, wording=RUN_WORDING[SUBJECT_SEALED], reason=str(exc))
    timing["load_pair"] = time.perf_counter()-tick
    memtrace.mark("run.load_pair")
    init_seed = None
    if getattr(args, "init_field", None) is not None:             # C12-4a: bound and read before any channel
        tick = time.perf_counter()
        try:
            init_seed = load_init_field(args.init_field, pair=pair, pair_name=args.pair,
                                        oracle_ok=args.init_field_oracle_ok)
        except InitFieldRejected as exc:
            logger.error("init_field_rejected: %s", exc)
            return dict(status=INIT_FIELD_REJECTED, wording=INIT_FIELD_REJECTED_WORDING, reason=str(exc))
        timing["init_field_load"] = time.perf_counter()-tick
        timing["init_field"] = init_field_summary(init_seed.record)     # status.json too, whatever ends the run
        logger.info("init field: %s, arm %s, oracle %s, state %s", init_seed.record["source_format"],
                    init_seed.record["arm"], init_seed.oracle, init_seed.record["state_hash"])
    structure = None
    if args.channel_source == "structure":
        structure = structure_source(args, command=("python", "-m", "bayesreg.cli.run_pair", "--pair", args.pair,
                                                    "--out", str(out)))
    _write_text(out/"data.txt", dumps(dict(pair=args.pair, pair_path=str(pair_path), pair_provenance=pair.provenance,
                                           allowlist=args.allowlist, tissue_root=args.tissue_root,
                                           channel_source=args.channel_source,
                                           structure=None if structure is None else structure.record()),
                                      indent=1)+"\n")
    tick = time.perf_counter()
    try:
        inputs, pt_f, mask_f, channel_prov = prepare_channels(pair, args.channel_source, args.tissue_root, logger,
                                                              structure=structure)
    except ProvenanceRejected as exc:
        logger.error("provenance_rejected: %s", exc)
        return dict(status=PROVENANCE_REJECTED, wording=RUN_WORDING[PROVENANCE_REJECTED], reason=str(exc))
    timing["channels_and_pyramid"] = time.perf_counter()-tick
    k_t = pt_f.shape[0]-1
    eta = read_eta(args.eta_json, k_t) if args.eta_json else eta_temporary(pt_f, mask_f, pair.grid)
    del pt_f, mask_f
    release_free_heap()                        # MEM: the channel preparation's freed arena memory goes back to the OS
    memtrace.mark("run.channels_ready")
    rounds = tuple(int(r) for r in args.rounds.split(","))
    proximal = ProximalParams(lam=default_lambda(pair.grid)) if args.outer == "b" else None
    config = SolverConfig(outer=args.outer, rounds=rounds, caps=solver_caps(args), **inner_options(args),
                          proximal=proximal,
                          max_levels=args.max_levels, cpu_threads=args.cpu_threads, logvol_points=args.logvol_points,
                          logvol_closed27_share=args.logvol_closed27_share, closed27_weights=args.closed27_weights,
                          **sigma_domain_options(args), **schedule_options(args), **init_field_options(args))
    backend = backend_of(args)
    sigma = GaussianMStep()
    seed_records = {} if init_seed is None else dict(init_field=init_seed.record)       # C16 WST item
    write_json(out/"config.json", dict(args=args_record(args), solver=config.record(), eta=eta.record(),
                                       sigma=sigma_policy_record(config, sigma),
                                       backend=backend.record(), opt1=opt1_switches(args), seed="none (deterministic)",
                                       wall_budget=None if budget is None else budget.record(), **seed_records))
    # provenance before the solve (S1 review item 7): an exception later still leaves it; exports go to status.json
    write_json(out/"provenance.json", dict(code=code_record(), jd_curl_snapshot=jd_curl_snapshot(),
                                           pair=pair.provenance, channels=channel_prov, channel_source=args.channel_source,
                                           omega0=pair.grid.omega, native_grid=grid_record(pair.grid),
                                           solver=config.record(), backend=backend.record(), eta=eta.record(),
                                           **seed_records,
                                           switches=dict(outer=config.outer,
                                                         inner="LM" if config.inner == INNER_LM else "PG",
                                                         metric=config.metric_mode,
                                                         metric_jd_scaling=config.metric_jd_scaling,
                                                         jd_cell_estimate=config.jd_cell_estimate,
                                                         logvol_points=config.logvol_points,
                                                         **closed27_switches(config),
                                                         witness_estimate=args.witness_estimate,
                                                         **sigma_domain_switches(config),
                                                         noise="gaussian", parametrization="P2", tau0_fixed=True,
                                                         probability_space_residual=False, background_channel=False,
                                                         preoptimizer=False, inner_forcing=False,
                                                         gf_normalization="sqrt(omega_ell/omega0)",
                                                         backend=args.backend, wall_seconds=args.wall_seconds,
                                                         **schedule_switches(args), **init_field_switches(init_seed),
                                                         **opt1_switches(args)),
                                           env=env_record(backend),
                                           resources=dict(cpu_threads=config.cpu_threads,
                                                          peak_rss="status.json resources (known at the end of the "
                                                                   "run; provenance.json is written before the solve)")))
    logger.info("solve: backend %s (%s), outer %s, rounds %s, eta %s, wall budget %s", backend.name, backend.device,
                config.outer, rounds, eta.record(),
                "none" if budget is None else f"{budget.seconds:g} s ({budget.elapsed():.1f} s used)")
    jsonl = JsonlLog(out/"log.jsonl")
    memtrace.mark("run.solve_start")
    tick = time.perf_counter()
    grids = tuple(lv.grid for lv in inputs)
    levels = list(inputs)                      # MEM: handed over; solve_multilevel releases each level's arrays
    del inputs
    try:
        # init_seed, recorder: only under their switches, so that the default call is the code before WST
        recorder = None
        if init_seed is not None or config.checkpoint is not None:
            recorder = ProfileRecorder(grid=grids[-1], directory=None if config.checkpoint is None else out/"checkpoints",
                                       config=config.checkpoint, log=jsonl,
                                       tag=None if init_seed is None else init_field_summary(init_seed.record))
        wst = {} if init_seed is None else dict(init_seed=init_seed)
        if recorder is not None:
            wst["recorder"] = recorder
        result = solve_multilevel(backend, levels, eta=eta, sigma=sigma, config=config, log=jsonl, budget=budget,
                                  release_inputs=True, **wst)
    finally:
        jsonl.close()
        del levels
    timing["solve"] = time.perf_counter()-tick
    timing["solve_levels"] = result.timing
    memtrace.mark("run.solve_end")
    logger.info("run status %s (%s): %s", result.status, result.wording, result.reason)
    exported = None
    status, wording, reason = result.status, result.wording, result.reason
    tick = time.perf_counter()
    seed_tag = None
    if init_seed is not None:                                     # C16 WST item: also after a failed entry (no commit)
        seed_tag = init_field_summary(result.init_field or init_seed.record)
        timing["init_field"] = seed_tag
    if result.native and result.commit is not None:
        exported = export_native(out, u=result.commit.u, grid=grids[-1], status=result.status, backend=backend,
                                 solver_cert=result.commit.cert,
                                 **({} if seed_tag is None else dict(init_field=seed_tag)))
        if not exported["certificate_passed"]:                    # hard constraint 1: no disp.npz, a failure
            status, wording = FINAL_CERTIFICATE_REJECTED, RUN_WORDING[FINAL_CERTIFICATE_REJECTED]
            reason = (f"the CPU certificate of the output state is {exported['cert']['status']} "
                      f"({exported['cert']['reason']}); solver status {result.status}: {result.reason}")
            logger.error("final certificate rejected: %s", reason)
    elif result.status == STOPPED_BEFORE_NATIVE and result.commit is not None:
        exported = dict(state_hash=write_final_state(out/f"final_level{result.level}.npz", result.commit.u,
                                                     grids[result.level], status=result.status), level=result.level)
    timing["export"] = time.perf_counter()-tick
    memtrace.mark("run.export_end")
    timing["certificate_seconds"] = backend.certificate_seconds
    timing["certificate_calls"] = backend.certificate_calls
    timing["witness_seconds"] = backend.witness_seconds
    timing["witness_records"] = backend.witness_records
    witness_summary = backend.witness_summary.record()
    comparison = (backend.comparison_summary() if hasattr(backend, "comparison_summary")
                  else dict(path="CPU reference (C08-1..7)", comparisons=None))
    timing["comparison_seconds"] = comparison.get("seconds")
    device_anomalies = comparison.get("device_anomalies")         # torch64 only (status.json, review OPT1 item 4)
    if device_anomalies and any(device_anomalies.values()):
        compare_reasons = [r["fallback_reason"] for r in backend.comparator.stats.records
                           if str(r["fallback_reason"]).startswith("device")]
        logger.warning("device anomalies %s, decided on the CPU (C08-8(e)); the trajectory is not the pure device "
                       "path's: evaluate %s; compare %s", device_anomalies, comparison.get("evaluate_anomalies", [])[:3],
                       compare_reasons[:3])
    brain = config.sigma_domain == BRAIN
    if result.state is None:
        sigma_used = None
    else:
        sigma_used = np.asarray(result.state.sigma2).tolist() if brain else result.state.sigma2_record()
    sigma_record = result.sigma or {}
    final_rows = [r for r in result.rounds if r.get("final")]
    # C04-6 "记录": the per-level support statistics only when sigma_domain != brain (brain: hyper.json unchanged
    # except the three keys of the configuration record in `algorithm`)
    support_records = {} if brain else dict(support_levels=result.support)
    write_json(out/"hyper.json", dict(eta=eta.record(), eta_tmp=eta.record() if eta.source.startswith("C05-7") else None,
                                      sigma_policy=sigma_policy_record(config, sigma), sigma_used=sigma_used,
                                      sigma_post=result.sigma_post,
                                      noise_model="gaussian", nu_t=None, inverse_gamma=dict(a=0., b=0.),
                                      rounds=result.rounds, transfers=result.transfers, algorithm=config.record(),
                                      sigma_collapse_suspected=sigma_record.get("sigma_collapse_suspected"),
                                      sigma_monitor=sigma_record, gpu_comparison=comparison,
                                      GF_B_cpu_final=final_rows[-1].get("GF_B_cpu") if final_rows else None,
                                      backend_after_solve=backend.record(), wall_budget=result.budget,
                                      solver_status=result.status, status=status, reason=reason, **support_records,
                                      **({} if init_seed is None else dict(init_field=result.init_field)),
                                      **({} if result.profile is None else dict(profile=result.profile))))
    write_json(out/"qc.json", dict(
        sigma_collapse_suspected=sigma_record.get("sigma_collapse_suspected"),
        sigma_collapse_events=sigma_record.get("events"), sigma_trajectory=sigma_record.get("trace"),
        sigma_collapse_criterion=sigma_record.get("criterion"), posterior_skipped="C14 not implemented",
        final_certificate=None if exported is None or "cert" not in exported else exported["cert"],
        not_implemented=["pass_rate", "pcg_not_converged", "S_conv", "posterior_cert_pass_rate_low",
                         "sigma_u quantiles", "log J quantiles"],
        **({} if init_seed is None else dict(init_field_oracle=init_seed.oracle))))
    summary = dict(status=status, wording=wording, reason=reason, native=result.native, solver_status=result.status,
                   exported=None if exported is None else exported.get("state_hash"),
                   export=None if exported is None else {k: v for k, v in exported.items() if k != "disp_meta"},
                   sigma_collapse_suspected=sigma_record.get("sigma_collapse_suspected"),
                   device_anomalies=device_anomalies, witness_summary=witness_summary)
    if result.profile is not None:                                # C16 WST item
        summary["n_checkpoints"] = result.profile["n_checkpoints"]
    if config.pg.jd_cell_estimate == CERTIFICATE_BERNSTEIN:       # C09.1-3a check (b): the run total (JDTI)
        summary["bernstein_j53_violations_total"] = j53_violations_total(result.rounds)
        if summary["bernstein_j53_violations_total"]:
            logger.warning("certificate_bernstein check (b): %d violation(s) in this run (log.jsonl kind warning); "
                           "the audit recertifies the final state without the prescreen before any use of its numbers",
                           summary["bernstein_j53_violations_total"])
    return summary


def _j53_total_from_log(path, logger):
    """status.json fields of a certificate_bernstein run that an exception ended outside the round records (JDTI review
    minor 1): the check-(b) total from the warning rows already written to log.jsonl."""
    source = "log.jsonl warning rows (the run ended with an exception; no round records)"
    try:
        rows = []
        if Path(path).is_file():
            with open(path, encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
        total = j53_violations_logged(rows)
    except (OSError, ValueError) as exc:
        logger.error("certificate_bernstein: the check-(b) total could not be read from %s: %s", path, exc)
        return dict(bernstein_j53_violations_total=None, bernstein_j53_violations_source=f"{source}: unreadable ({exc})")
    if total:
        logger.warning("certificate_bernstein check (b): %d violation(s) in this run (log.jsonl kind warning); the audit "
                       "recertifies the final state without the prescreen before any use of its numbers", total)
    return dict(bernstein_j53_violations_total=total, bernstein_j53_violations_source=source)


def main(argv=None):
    args = parse_args(argv)
    budget = make_budget(args.wall_seconds)                       # counted from here (development switch, D44)
    out = _prepare_run_dir(args.out)
    logger = _run_logger(out)
    started = time.time()
    timing = dict(started_unix=started)
    argv_text = sys.argv if argv is None else ["python", "-m", "bayesreg.cli.run_pair", *argv]
    _write_text(out/"cmd.txt", " ".join(shlex.quote(a) for a in argv_text)+f"\n# cwd {os.getcwd()}\n"
                f"# python {sys.executable}\n")
    code = code_record()
    _write_text(out/"git.txt", f"HEAD {code['head']}\ndirty {code['dirty']}\n\n# git status --porcelain\n{code['status']}"
                f"\n# git diff HEAD\n{_git('diff', 'HEAD')}")
    _write_text(out/"env.txt", dumps(env_record(), indent=1)+"\n")
    try:
        summary = run(args, out, logger, timing, budget)
    except Exception as exc:                                      # evidence kept, never retried (constraint 4)
        logger.error("exception: %s\n%s", exc, traceback.format_exc())
        summary = dict(status="exception", wording="运行异常", reason=f"{type(exc).__name__}: {exc}")
        if args.jd_cell_estimate == CERTIFICATE_BERNSTEIN:
            summary.update(_j53_total_from_log(out/"log.jsonl", logger))
        code_out = 1
    else:
        code_out = 0 if summary["status"] in ENDINGS else EXIT.get(summary["status"], 1)
    summary = dict(summary, structure_registry=structure_registry_label(args))
    if "init_field" in timing:                                    # C16 WST item (only under --init-field)
        summary["init_field"] = timing["init_field"]
    timing["total_seconds"] = time.time()-started
    memtrace.mark("run.end")
    resources = resource_record(args.cpu_threads)
    timing["peak_rss_mb"] = resources["peak_rss_mb"]
    timing["resources"] = resources
    # development wall-clock budget (D44): its record at the end of the run (elapsed includes the export and the final
    # certificate, which run after the budget); hyper.json keeps the record at the end of the solve
    summary["wall_budget"] = None if budget is None else budget.record()
    timing["wall_budget"] = summary["wall_budget"]
    write_json(out/"timing.json", timing)
    write_json(out/"status.json", dict(summary, exit_code=code_out, resources=resources))
    logger.info("status %s, exit %d", summary["status"], code_out)
    _close_logger(logger)
    return code_out


if __name__ == "__main__":
    sys.exit(main())
