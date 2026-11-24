#!/usr/bin/env bash
# D0 queue (docs/bayesreg/05_D0协议.md §4-§6): configuration C (frozen eb808be worktree, 90 s) on the 19 Learn2Reg validation
# pairs x 4 input arms, strictly serial, in the pre-registered order (pair-major; the arm order rotates with the pair index:
# pair i runs A[(i+0)%4], A[(i+1)%4], A[(i+2)%4], A[(i+3)%4] with A = (A0 A1 A2 A3)).
#
# Per run: hold file -> wait while a paper-1 unit (seq-vp-*) or another tag-* unit is active, and while the GPU has
# another compute process or > 20% utilization -> quiet machine
# (1-min loadavg < 1.5, at most 30 min) -> /home free >= 100 GB -> point the worktree link data/three_d/stage_L2R_val_v1 at
# the arm's stage (flock; restored to the original stage right after) -> e48 prepare -> check_prepared.py (configuration C
# == the freeze reference; inputs == the arm's files) -> records (git.txt env.txt data.txt seed cmd.txt d0_run.json) ->
# systemd-run --user --unit tag-d0-<arm>-<ffff>-<mmmm> --wait --collect (resources as eb808be scripts/launch_stage_3d_e48.sh:
# CPUQuota 1200%, MemoryMax 12G, MemorySwapMax 4G, 8 BLAS/OMP/Numba threads, watchdog timeout 300 s + 15 s kill grace)
# -> classify_run.py -> ok / handover_only / method_failure: evaluate.py -> metrics.json; ok/handover_only: delete the
# intermediate commit maps (committed_*.npz, level_*.npz) of the task directory only; method_failure: kept whole ->
# queue_done.json, one line in queue.log; infrastructure / unsure / any check failure: STOP (run kept whole, nothing rerun).
# Resume: a run with queue_done.json is skipped; a run directory without it whose unit has ended and which has an exit_code
# is post-processed; otherwise STOP. One rerun of an infrastructure failure, in a new directory <pair>__r1, happens only if
# the lead creates <run>/RETRY_APPROVED (protocol §6).
#
# --a4: the supplementary arm A4p (A4' of protocol §12, U-Net posterior colouring): the same frozen solver, options,
# resources, gates, checks, classification, evaluation and cleanup, but only arm A4p, the 19 pairs in selection.json order,
# everything under /home/rmcfz/seq_vp_tag_results/d0_v1_a4 (inputs/A4p, runs/A4p, cache, queue.log, analysis; nothing is
# written under d0_v1/), units tag-d0a4-A4p-<ffff>-<mmmm>, run ids d0a4_*. Extra start checks (STOP if not met):
# tag-unet35-train and every tag-d0-* unit inactive; the GPU without compute process and <= 20% utilization; the
# repository registry configs/bayesreg/segmenters.json tracked by git and clean (A4 review item 3); the A4p inputs record
# checked against it (status registered, git tracked and clean when the inputs were made, sha256 still the same). The
# per-run wait no longer exempts tag-d0-* units. At the end: evaluate.py summarize-a4 (A0-A3 read from d0_v1/runs).
# --a1p: the arms A1Ph, A1Ps (runs/20261102-a1p/plan.md, predicted seg4 colouring): the same frozen solver, options, resources,
# gates, checks, classification, evaluation and cleanup; the 19 pairs in selection.json order, the arm order rotating with the
# pair index (pair i runs A[(i+0)%2], A[(i+1)%2] with A = (A1Ph A1Ps)); everything under
# /home/rmcfz/seq_vp_tag_results/d0_v1_a1p (nothing under d0_v1/ or d0_v1_a4/), units tag-d0a1p-<arm>-<ffff>-<mmmm>, run ids
# d0a1p_*. The start checks of --a4 for both arms, and no tag-d0a4-* unit. At the end: evaluate.py summarize-a1p (A0-A3 read
# from d0_v1/runs, A4p from d0_v1_a4/runs).
# --x1: gate A, X1 (runs/20261102-gate-a/plan.md, the frozen solver with absolute weights): the arms X1h (--native-reference
# identity --native-penalty-scale .5) and X1f (--native-reference identity, x1), the original images, the first 5 pairs of
# selection.json, the arm order rotating with the pair index; everything else as configuration C (same 90 s, options, resources,
# gates, classification, evaluation, cleanup except that the native seed level_3_level_prepared.npz is kept and evaluated;
# check_prepared.py is told the two switches and compares every other key with
# the freeze reference). Everything under /home/rmcfz/seq_vp_tag_results/gate_a_x1, units tag-gax1-<arm>-<ffff>-<mmmm>, run ids
# gax1_*, stage GA-<arm>. Start checks: no tag-unet35-train, tag-d0-*, tag-d0a4-*, tag-d0a1p-* unit, the GPU free. At the end:
# evaluate.py summarize-x1 (A0 read from d0_v1/runs, P0n from path_pilot_v1).
# --a5: the arms A5, A5P (runs/20261103-a5/plan.md, protocol §14, the fixed colour table on the truth / predicted seg35): the same
# frozen solver, options, resources, gates, checks, classification, evaluation and cleanup; pair-major, the arm order rotating with
# the pair index (pair i runs A[(i+0)%2], A[(i+1)%2] with A = (A5 A5P)); everything under /home/rmcfz/seq_vp_tag_results/d0_v1_a5
# (nothing under d0_v1/, d0_v1_a4/, d0_v1_a1p/), units tag-d0a5-<arm>-<ffff>-<mmmm>, run ids d0a5_*. --first-5: the first 5 pairs only
# (10 runs, the plan's first gate). An item of pair index >= 5 needs analysis/decision_a5.json with gate1.expand true (evaluate.py
# summarize-a5 writes it at the end of --first-5), else STOP; a run with queue_done.json is skipped, so plain --a5 after --first-5
# runs the other 14 pairs (28 runs) and reruns nothing. The start checks of --a1p (the registry check for A5P), no tag-d0a4-*,
# tag-d0a1p-*, tag-gax1-* unit, the colour table committed and clean, its sha256 in both inputs records, neither record partial.
# At the end: evaluate.py summarize-a5 (each reference from the long.csv of its own batch).
# --b1 SPEC [--pairs first-5|rest-14|all]: the option arms of the frozen solver (runs/20261103-b1/plan.md, protocol §15). The arms come from a
# specification (experiments/d0/b1_spec.py: name, inputs root, changed options, --seconds): the same frozen solver, resources, gates, classification,
# evaluation and cleanup; only the listed options differ from configuration C (check_prepared.py --b1-spec compares every other key with the freeze
# reference), and the watchdog grows with the budget (300 s for 90 s, 1100 s for 330 s). Pair-major, the arm order rotating with the pair index
# (pair i runs A[(i+j) % n], j = 0..n-1). Everything under /home/rmcfz/seq_vp_tag_results/b1_v1, units tag-b1-<arm>-<ffff>-<mmmm>, run ids b1_*,
# stage B1-<arm>. Start checks: the ones of --a5 per arm, on the record of the arm's inputs root (colour table; unless oracle inputs, the registry of the entry, default
# configs/bayesreg/segmenters.json), earlier runs of the arm with the same configuration, no tag-d0a4-*, tag-d0a1p-*, tag-gax1-*, tag-d0a5-*, tag-unv2-* unit. An item of pair
# index >= 5, an arm marked expansion, oracle inputs, B1W_A5 and B1W_v2 need analysis/decision_b1.json, made from the repository's arms_selection.json, with this arm's
# configuration as the winner's, else STOP. A run with queue_done.json is skipped. --runs-root is refused: the summary reads $D0/runs.
# With --dry-run the command lines of every item are printed too; nothing is written. At the end: evaluate.py summarize-b1.
# All modes switch the same worktree link under one lock file, $RESULTS/.frozen_eb808be_link.lock (A4 review item 1).
# Usage: run_queue.sh [--a4 | --a1p | --x1 | --a5 [--first-5] | --b1 SPEC [--pairs first-5|rest-14|all]] [--only ARM:INDEX[,ARM:INDEX...]] [--runs-root DIR] [--queue-log FILE] [--unit-prefix P] [--dry-run]
# Tests only (tests/test_bayesreg/solve/test_d0_a4_tools.py, synthetic tmp trees): D0Q_REPO, D0Q_RESULTS, D0Q_SMI and D0Q_POLL replace
# the repository, the results root, nvidia-smi and the poll interval; unset (always, for real runs) they are the paths and the value below.
set -u
REPO=${D0Q_REPO:-/home/rmcfz/code/python_projects/seq_vp_tag}
RESULTS=${D0Q_RESULTS:-/home/rmcfz/seq_vp_tag_results}
W=$RESULTS/frozen_eb808be
D0=$RESULTS/d0_v1
D0V1=$D0
PY=/home/rmcfz/miniconda3/envs/sci310/bin/python
SMI=${D0Q_SMI:-/usr/lib/wsl/lib/nvidia-smi}
FROZEN=eb808be
LINK_LOCK=$RESULTS/.frozen_${FROZEN}_link.lock           # one lock for the link switch of every mode (A4 review item 1)
REAL_STAGE=$(readlink -f $REPO/data/three_d/stage_L2R_val_v1)
LINK=$W/data/three_d/stage_L2R_val_v1
WATCHDOG=300; QUOTA=1200; FLOOR_GB=100; POLL=${D0Q_POLL:-60}
RUNS=""; QLOG=""; UNITP=""; ONLY=""; DRY=0; MODE=d0; FIRST5=0; B1SPEC=""; B1PAIRS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2;; --runs-root) RUNS="$2"; shift 2;; --queue-log) QLOG="$2"; shift 2;;
    --unit-prefix) UNITP="$2"; shift 2;; --dry-run) DRY=1; shift;; --a4) MODE=a4; shift;; --a1p) MODE=a1p; shift;;
    --x1) MODE=x1; shift;; --a5) MODE=a5; shift;; --first-5) FIRST5=1; shift;;
    --b1) MODE=b1; B1SPEC="$2"; shift 2;; --pairs) B1PAIRS="$2"; shift 2;;
    *) echo "unknown option $1"; exit 2;;
  esac
done
[ $FIRST5 -eq 0 ] || [ $MODE = a5 ] || { echo "--first-5 goes with --a5"; exit 2; }
[ -z "$B1PAIRS" ] || [ $MODE = b1 ] || { echo "--pairs goes with --b1"; exit 2; }
[ -z "$RUNS" ] || [ $MODE != b1 ] || { echo "--runs-root: not with --b1 (the summary reads the runs of b1_v1)"; exit 2; }
EXTRA_BUSY=(); STAGE_PREFIX=D0; KEEP_NPZ=(); CHECK_OPTS=(); EVAL_CMD=(run)
declare -A B1_ROOT=() B1_SECONDS=() B1_WD=() B1_GW=() B1_IT=() B1_EXP=() B1_INPUT_ARM=()
if [ $MODE = a4 ]; then
  D0=$RESULTS/d0_v1_a4; ARMS=(A4p); SELF_RE='^tag-d0a4-'; RIDP=d0a4; UNITP=${UNITP:-tag-d0a4}
elif [ $MODE = a1p ]; then
  D0=$RESULTS/d0_v1_a1p; ARMS=(A1Ph A1Ps); SELF_RE='^tag-d0a1p-'; RIDP=d0a1p; UNITP=${UNITP:-tag-d0a1p}; EXTRA_BUSY=('tag-d0a4-*')
elif [ $MODE = x1 ]; then
  D0=$RESULTS/gate_a_x1; ARMS=(X1h X1f); SELF_RE='^tag-gax1-'; RIDP=gax1; UNITP=${UNITP:-tag-gax1}; EXTRA_BUSY=('tag-d0a4-*' 'tag-d0a1p-*'); STAGE_PREFIX=GA
  KEEP_NPZ=(level_3_level_prepared.npz)                    # the native seed stays: evaluate.py reports its Dice (plan revision 1, review I3)
elif [ $MODE = a5 ]; then
  D0=$RESULTS/d0_v1_a5; ARMS=(A5 A5P); SELF_RE='^tag-d0a5-'; RIDP=d0a5; UNITP=${UNITP:-tag-d0a5}; EXTRA_BUSY=('tag-d0a4-*' 'tag-d0a1p-*' 'tag-gax1-*')
elif [ $MODE = b1 ]; then
  D0=$RESULTS/b1_v1; SELF_RE='^tag-b1-'; RIDP=b1; UNITP=${UNITP:-tag-b1}; STAGE_PREFIX=B1
  EXTRA_BUSY=('tag-d0a4-*' 'tag-d0a1p-*' 'tag-gax1-*' 'tag-d0a5-*' 'tag-unv2-*')
  case "${B1PAIRS:=all}" in first-5|rest-14|all) ;; *) echo "--pairs: first-5, rest-14 or all, not $B1PAIRS"; exit 2;; esac
  B1SPEC=$(readlink -f "$B1SPEC")
  B1LINES=$($PY -B $REPO/experiments/d0/b1_spec.py shell "$B1SPEC") || { echo "$B1LINES"; echo "--b1: the specification $B1SPEC is refused"; exit 2; }
  ARMS=()
  while IFS=$'\t' read -r a root sec wd gw it exp input; do
    ARMS+=("$a"); B1_ROOT[$a]=$root; B1_SECONDS[$a]=$sec; B1_WD[$a]=$wd; B1_GW[$a]=$gw; B1_IT[$a]=$it; B1_EXP[$a]=$exp; B1_INPUT_ARM[$a]=$input
  done <<< "$B1LINES"
  EVAL_CMD=(run-b1 --spec "$B1SPEC")
else
  ARMS=(A0 A1 A2 A3); SELF_RE='^tag-d0-'; RIDP=d0v1; UNITP=${UNITP:-tag-d0}
fi
REG_ARMS=("${ARMS[@]}"); [ $MODE = a5 ] && REG_ARMS=(A5P)   # the arms coloured from the segmenter (A5 is the oracle arm)
RUNS=${RUNS:-$D0/runs}; QLOG=${QLOG:-$D0/queue.log}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PRE="fp32-v1:iterations=50/50/100/60,lr=0.5,cap=0.5,sigma_grad=1,sigma_field=1,mu=1/1/0.01/0.003,alpha=1e-3/1e-3/1e-3/0,kappa_pre=10,j0=0.2,seconds=60,taper=4"
OPTS=(--strength weak --lean-commits --resolution-exit
      --pg-metric-policy-levels legacy_q1_jd_affine_always_v1,legacy_q1_jd_affine_always_v1,image_shift_ggn_equilibrated_jd_affine_v3,image_shift_ggn_equilibrated_jd_affine_v3
      --level-shares 0.020833,0.0625,0.145833,1 --geometry-weight-levels 1,1,0.1,0.03 --curvature-weight-levels 1e-3,1e-3,1e-3,0
      --throughput v41 --export-reserve 1 --seconds 90 --pre-optimizer "$PRE")
OPTS_C=("${OPTS[@]}")
[ $DRY -eq 1 ] || mkdir -p "$RUNS" "$(dirname "$QLOG")" $D0/cache/numba_jit $D0/cache/first_use
utc() { date -u +%FT%TZ; }
log() { echo "$(utc) $*" | tee -a "$QLOG"; }
homefree() { df --output=avail -B1G /home | tail -1 | tr -d ' '; }
gpustate() { $SMI --query-gpu=name,clocks.sm,pstate,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' '; }
dmesg_since() { dmesg 2>/dev/null | grep -E "$2" | sed -E 's/^\[ *([0-9.]+)\].*/\1/' | awk -v t0="$1" '$1+0 >= t0+0 {c++} END {print c+0}'; }
original_images() { case "$1" in A0|X1h|X1f) return 0;; esac; return 1; }           # arms without a recoloured stage
stage_of() { if [ $MODE = b1 ]; then echo "${B1_ROOT[$1]}/stage_L2R_val_v1"; return; fi; original_images "$1" && echo "$REAL_STAGE" || echo "$D0/inputs/$1/stage_L2R_val_v1"; }
input_arm() { if [ $MODE = b1 ]; then echo "${B1_INPUT_ARM[$1]}"; else echo "$1"; fi; }                  # the arm named by the inputs' records
input_record() { if [ $MODE = b1 ]; then echo "${B1_ROOT[$1]}/d0_inputs.json"; else echo "$D0/inputs/$1/d0_inputs.json"; fi; }
b1_opts() {                                                                        # configuration C with only the changes of the arm's entry
  local arm=$1 prev="" o; OPTS=()
  for o in "${OPTS_C[@]}"; do
    case "$prev" in
      --seconds) o=${B1_SECONDS[$arm]};;
      --geometry-weight-levels) [ "${B1_GW[$arm]}" = - ] || o=${B1_GW[$arm]};;
      --pre-optimizer) [ "${B1_IT[$arm]}" = - ] || o=${o/iterations=50\/50\/100\/60,/iterations=${B1_IT[$arm]},};;
    esac
    OPTS+=("$o"); prev=$o
  done
}
set_arm_opts() { case "$1" in                                                      # the arm's switches of `prepare` and check_prepared
    X1h) ARM_OPTS=(--native-reference identity --native-penalty-scale .5);;
    X1f) ARM_OPTS=(--native-reference identity);;
    *) ARM_OPTS=();; esac
  if [ $MODE = b1 ]; then b1_opts "$1"; CHECK_OPTS=(--b1-spec "$B1SPEC" --b1-arm "$1"); WATCHDOG=${B1_WD[$1]}; fi; }
build_prep_cmd() {  # uses arm, pair, run of the caller
  prep_cmd=($PY -B $W/scripts/run_stage_3d_e48_pg.py prepare "$run" "${OPTS[@]}" "${ARM_OPTS[@]}" --stage "$STAGE_PREFIX-$arm" --pair "$pair")
}
build_sd_cmd() {  # uses run, unit of the caller
  local inner="printf '%s\n' \"\$(date -u +%FT%TZ)\" > '$run/supervisor_started_utc.txt'; timeout --signal=TERM --kill-after=15s ${WATCHDOG}s $PY -B '$run/snapshot/scripts/run_stage_3d_e48_pg.py' run '$run' > '$run/run.log' 2>&1; ec=\$?; printf '%s\n' \"\$ec\" > '$run/exit_code'; printf '%s\n' \"\$(date -u +%FT%TZ)\" > '$run/supervisor_finished_utc.txt'; exit \$ec"
  sd_cmd=(systemd-run --user --unit="$unit" --wait --collect --working-directory="$run/snapshot"
      --property=CPUQuota=${QUOTA}% --property=MemoryMax=12G --property=MemorySwapMax=4G
      --setenv=OPENBLAS_NUM_THREADS=8 --setenv=OMP_NUM_THREADS=8 --setenv=MKL_NUM_THREADS=8 --setenv=NUMEXPR_NUM_THREADS=8
      --setenv=NUMBA_NUM_THREADS=8 --setenv=PYTHONDONTWRITEBYTECODE=1
      --setenv=JD_CURL_JIT_CACHE=$D0/cache/numba_jit --setenv=JD_CURL_FIRST_USE_CACHE=$D0/cache/first_use
      /bin/bash -c "$inner")
}
short_of() { echo "$1" | sed -E 's/OASIS_OAS1_([0-9]+)_MR1__OASIS_OAS1_([0-9]+)_MR1/\1-\2/'; }
mapfile -t PAIRS < <($PY -c "import json,sys;[print(p['pair']) for p in json.load(open(sys.argv[1]))['pairs']]" "$REAL_STAGE/selection.json")
[ ${#PAIRS[@]} -eq 19 ] || { log "STOP: expected 19 validation pairs, got ${#PAIRS[@]}"; exit 4; }

# ---- order ------------------------------------------------------------------------------------------------------------------
ORDER=()
if [ $MODE = a4 ]; then
  for i in $(seq 0 18); do ORDER+=("A4p:$i"); done                  # one arm: the selection.json order (protocol §12.5)
elif [ $MODE = a1p ]; then
  for i in $(seq 0 18); do for j in 0 1; do ORDER+=("${ARMS[$(( (i+j) % 2 ))]}:$i"); done; done     # plan: the arm order rotates
elif [ $MODE = x1 ]; then
  for i in $(seq 0 4); do for j in 0 1; do ORDER+=("${ARMS[$(( (i+j) % 2 ))]}:$i"); done; done      # gate A plan: the first 5 pairs
elif [ $MODE = a5 ]; then
  last=18; [ $FIRST5 -eq 1 ] && last=4                             # A5 plan: --first-5, then all (done runs are skipped)
  for i in $(seq 0 $last); do for j in 0 1; do ORDER+=("${ARMS[$(( (i+j) % 2 ))]}:$i"); done; done
elif [ $MODE = b1 ]; then
  case $B1PAIRS in first-5) lo=0; hi=4;; rest-14) lo=5; hi=18;; all) lo=0; hi=18;; esac
  n_arms=${#ARMS[@]}
  for i in $(seq $lo $hi); do for j in $(seq 0 $((n_arms-1))); do ORDER+=("${ARMS[$(( (i+j) % n_arms ))]}:$i"); done; done
else
  for i in $(seq 0 18); do for j in 0 1 2 3; do ORDER+=("${ARMS[$(( (i+j) % 4 ))]}:$i"); done; done
fi
if [ -n "$ONLY" ]; then
  SEL=(); IFS=, read -ra WANT <<< "$ONLY"
  for x in "${ORDER[@]}"; do for y in "${WANT[@]}"; do [ "$x" = "$y" ] && SEL+=("$x"); done; done
  ORDER=("${SEL[@]}")
fi
if [ $DRY -eq 1 ]; then
  n=0; for x in "${ORDER[@]}"; do n=$((n+1)); echo "$n ${x%%:*} ${PAIRS[${x##*:}]}"; done
  if [ $MODE = b1 ]; then
    n=0
    for x in "${ORDER[@]}"; do
      n=$((n+1)); arm=${x%%:*}; pair=${PAIRS[${x##*:}]}; run=$RUNS/$arm/$pair; unit=$UNITP-$arm-$(short_of "$pair")
      set_arm_opts "$arm"; build_prep_cmd; build_sd_cmd
      echo "# item $n: arm $arm, pair index ${x##*:}, inputs $(stage_of "$arm")/$pair (arm of the record: $(input_arm "$arm")), watchdog ${WATCHDOG} s; link during prepare: $LINK -> $(stage_of "$arm")"
      printf '%q ' "${prep_cmd[@]}"; echo
      printf '%q ' $PY -B $REPO/experiments/d0/check_prepared.py "$run" --pair "$pair" --stage "$STAGE_PREFIX-$arm" "${ARM_OPTS[@]}" "${CHECK_OPTS[@]}" --expect-pair-sha256 '<pair.npz sha256>' --expect-manifest-sha256 '<manifest.json sha256>'; echo
      printf '%q ' "${sd_cmd[@]}"; echo
    done
  fi
  exit 0
fi

# ---- start checks -----------------------------------------------------------------------------------------------------------
log "start: runs $RUNS; protocol sha256 $(sha256sum $REPO/docs/bayesreg/05_D0协议.md 2>/dev/null | cut -c1-16); repo HEAD $(git -C $REPO rev-parse --short HEAD); ${#ORDER[@]} runs in order; mode $MODE (arms ${ARMS[*]})"
if [ $MODE != d0 ]; then                                   # protocol §12.5: the finished, registered segmenter; a free GPU
  busy=$(systemctl --user list-units --state=active,activating --no-legend --plain 'tag-unet35-train*' 'tag-d0-*' "${EXTRA_BUSY[@]}" 2>/dev/null \
         | awk '{print $1}' | tr '\n' ' ')
  [ -z "${busy// /}" ] || { log "STOP: units still active: $busy (${ARMS[*]} need the finished segmenter and no D0 queue)"; exit 1; }
  apps=$($SMI --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' | tr '\n' ' ')
  util=$($SMI --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  { [ -z "${apps// /}" ] && [ "${util:-100}" -le 20 ]; } || { log "STOP: GPU not idle at start (compute pids: ${apps:-none}; utilization ${util:-?}%)"; exit 1; }
fi
if [ $MODE = a5 ]; then                                    # A5 plan, first gate: decided by evaluate.py summarize-a5 from A5 and A1 only
  beyond=0; for x in "${ORDER[@]}"; do [ "${x##*:}" -ge 5 ] && beyond=1; done
  if [ $beyond -eq 1 ]; then
    $PY -B - "$D0/analysis/decision_a5.json" <<'EOF' >> "$QLOG" 2>&1 || { log "STOP: the first gate has not opened the pairs after the first 5 ($D0/analysis/decision_a5.json; see queue.log)"; exit 1; }
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
gate = (json.loads(path.read_text()) if path.is_file() else {}).get('gate1') or {}
print(f"A5 first gate: {path}: expand {gate.get('expand')}; {gate.get('text')}")
sys.exit(0 if gate.get('expand') is True else 1)
EOF
  fi
  TABREL=configs/bayesreg/colour_table_a5_v1.json
  { git -C "$REPO" ls-files --error-unmatch -- $TABREL >/dev/null 2>&1 && [ -z "$(git -C "$REPO" status --porcelain -- $TABREL 2>&1)" ]; } \
    || { log "STOP: the colour table $REPO/$TABREL is not committed or not clean (git status: $(git -C "$REPO" status --porcelain -- $TABREL 2>&1 | tr '\n' ' '))"; exit 1; }
  for arm in "${ARMS[@]}"; do
  $PY -B - "$D0/inputs/$arm/d0_inputs.json" "$REPO/$TABREL" "$arm" <<'EOF' >> "$QLOG" 2>&1 || { log "STOP: the $arm inputs were not made with the current colour table, or are partial (see queue.log)"; exit 1; }
import hashlib, json, sys
from pathlib import Path
rec, table, arm = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
doc = json.loads(rec.read_text()) if rec.is_file() else {}
sha = hashlib.sha256(table.read_bytes()).hexdigest() if table.is_file() else None
recorded = (doc.get('colour_table') or {}).get('table_sha256')
ok = sha is not None and recorded == sha and doc.get('arm') == arm and not doc.get('partial')      # partial: a dry run's inputs
print(f"{arm} colour table check: inputs record {rec} table sha256 {recorded}; partial {doc.get('partial')}; repository table {table} sha256 {sha}; {'ok' if ok else 'FAILED'}")
sys.exit(0 if ok else 1)
EOF
  done
fi
if [ $MODE = b1 ]; then                                    # B1 plan: the selection decides what runs beyond the first 5 pairs
  gated=(); for x in "${ORDER[@]}"; do if [ "${x##*:}" -ge 5 ] || [ "${B1_EXP[${x%%:*}]}" = 1 ]; then gated+=("${x%%:*}"); fi; done
  if [ ${#gated[@]} -gt 0 ]; then
    $PY -B $REPO/experiments/d0/b1_spec.py gate --selection-spec $REPO/runs/20261103-b1/arms_selection.json "$B1SPEC" "$D0/analysis/decision_b1.json" "${gated[@]}" >> "$QLOG" 2>&1 \
      || { log "STOP: the selection has not opened these items: the arm is not the winner's configuration, or no decision ($D0/analysis/decision_b1.json; see queue.log)"; exit 1; }
  fi
  $PY -B $REPO/experiments/d0/b1_spec.py check-inputs "$B1SPEC" "$REPO" "$RUNS" >> "$QLOG" 2>&1 \
    || { log "STOP: the inputs of an arm do not pass the colour table and registry checks, or an earlier run of the arm has another configuration (see queue.log)"; exit 1; }
fi
if [ $MODE = a4 ] || [ $MODE = a1p ] || [ $MODE = a5 ]; then         # the arms coloured from the segmenter
  REGREL=configs/bayesreg/segmenters.json                  # A4 review item 3: committed and clean, not only the same sha256
  { git -C "$REPO" ls-files --error-unmatch -- $REGREL >/dev/null 2>&1 && [ -z "$(git -C "$REPO" status --porcelain -- $REGREL 2>&1)" ]; } \
    || { log "STOP: the repository registry $REPO/$REGREL is not committed or not clean (git status: $(git -C "$REPO" status --porcelain -- $REGREL 2>&1 | tr '\n' ' '))"; exit 1; }
  for arm in "${REG_ARMS[@]}"; do
  $PY -B - "$D0/inputs/$arm/d0_inputs.json" "$REPO/$REGREL" "$arm" <<'EOF' >> "$QLOG" 2>&1 || { log "STOP: the $arm inputs were not made with the current repository registry (see queue.log)"; exit 1; }
import hashlib, json, sys
from pathlib import Path
rec, reg, arm = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
doc = json.loads(rec.read_text()) if rec.is_file() else {}
r = doc.get('structure_registry') or {}
sha = hashlib.sha256(reg.read_bytes()).hexdigest() if reg.is_file() else None
git = r.get('git') or {}
ok = r.get('status') == 'registered' and sha is not None and r.get('sha256') == sha and Path(r.get('path', '')).resolve() == reg.resolve() \
     and git.get('tracked') is True and git.get('clean') is True and not doc.get('partial')      # partial: a dry run's inputs
print(f"{arm} registry check: inputs record {rec} registry {r}; partial {doc.get('partial')}; repository registry {reg} sha256 {sha}; {'ok' if ok else 'FAILED'}")
sys.exit(0 if ok else 1)
EOF
  done
fi
[ "$(git -C $W rev-parse HEAD)" = "$(git -C $REPO rev-parse $FROZEN)" ] || { log "STOP: worktree HEAD is not $FROZEN"; exit 1; }
[ -z "$(git -C $W status --porcelain)" ] || { log "STOP: frozen worktree has tracked changes or untracked files outside the ignored data/ and results/"; exit 1; }
$PY -B - "$W" "$REPO/experiments/d0/configC_reference.json" <<'EOF' || { log "STOP: worktree source differs from the freeze"; exit 1; }
import hashlib, json, sys
from pathlib import Path
w = Path(sys.argv[1]); ref = json.loads(Path(sys.argv[2]).read_text())['freeze_source_py_sha256']
live = {str(p.relative_to(w)): hashlib.sha256(p.read_bytes()).hexdigest() for d in ('jd_curl', 'scripts') for p in (w/d).rglob('*.py') if '__pycache__' not in p.parts}
sys.exit(0 if live == ref else 1)
EOF
ENVTXT=$D0/env_$(date -u +%Y%m%dT%H%M%SZ).txt
{ echo "python: $($PY -c 'import sys;print(sys.version.replace(chr(10)," "))')"; echo "conda env: $(dirname $(dirname $PY))"
  $PY -c "
import importlib
for m in ('numpy','scipy','numba','torch','triton','nibabel','surface_distance'):
    try: x=importlib.import_module(m); print(m, getattr(x,'__version__','?'))
    except Exception as e: print(m, 'n/a', type(e).__name__)
import torch; print('torch.cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version(), 'device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)" 2>&1
  echo "nvidia-smi: $($SMI --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1)"
  echo "cpu: $(lscpu | sed -n 's/^Model name:[ ]*//p'); nproc $(nproc)"; echo "kernel: $(uname -r)"; echo "memory: $(free -g | awk '/Mem:/{print $2" GiB total"}')"
  echo "ANTs: 2.6.5-gfdce4d2 (/home/rmcfz/opt/baselines/ants/ants-2.6.5/bin; Atropos for A2)"
  if [ $MODE = a4 ] || [ $MODE = a1p ] || [ $MODE = a5 ]; then echo "segmenter registry (${REG_ARMS[*]}): $REPO/configs/bayesreg/segmenters.json sha256 $(sha256sum $REPO/configs/bayesreg/segmenters.json | cut -c1-64)"; fi
  if [ $MODE = a5 ]; then echo "colour table (${ARMS[*]}): $REPO/configs/bayesreg/colour_table_a5_v1.json sha256 $(sha256sum $REPO/configs/bayesreg/colour_table_a5_v1.json | cut -c1-64)"; fi
  if [ $MODE = b1 ]; then echo "b1 specification: $B1SPEC sha256 $(sha256sum "$B1SPEC" | cut -c1-64); pairs $B1PAIRS"; fi
  echo "jit cache: $D0/cache/numba_jit; first-use cache: $D0/cache/first_use (not ~/.cache/seq_vp_3d)"; } > "$ENVTXT"
declare -A EXP_PAIR EXP_MAN
for arm in "${ARMS[@]}"; do
  st=$(stage_of $arm)
  out=$($PY -B - "$st" "$(input_arm "$arm")" "$(input_record "$arm")" <<'EOF'
import hashlib, json, sys
from pathlib import Path
st, arm, rec = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
original = arm in ('A0', 'X1h', 'X1f')
listed = json.loads(rec.read_text())['files_sha256'] if not original else None
bad = 0
for row in json.loads((st/'selection.json').read_text())['pairs']:
    d = st/row['pair']; p, m = sha(d/'pair.npz'), sha(d/'manifest.json')
    man = json.loads((d/'manifest.json').read_text())
    ok = man['pair_sha256'] == p and man['pair'] == row['pair'] and original == ('preprocessing_d0' not in man)
    if listed is not None:
        ok = ok and listed.get(f"stage_L2R_val_v1/{row['pair']}/pair.npz") == p and listed.get(f"stage_L2R_val_v1/{row['pair']}/manifest.json") == m \
             and man['preprocessing_d0']['arm'] == arm
    bad += not ok
    print(row['pair'], p, m)
sys.exit(1 if bad else 0)
EOF
) || { log "STOP: arm $arm inputs do not verify ($st)"; exit 1; }
  while read -r p ps ms; do EXP_PAIR[$arm:$p]=$ps; EXP_MAN[$arm:$p]=$ms; done <<< "$out"
done
log "start checks passed: worktree $FROZEN clean and == freeze source; inputs of ${ARMS[*]} verified; env $ENVTXT"

restore_link() { ln -sfn "$REAL_STAGE" "$LINK.tmp" && mv -Tf "$LINK.tmp" "$LINK"; }
mkdir -p "$(dirname "$LINK")"; [ -e "$LINK" ] || restore_link

wait_turn() {
  [ -e $D0/hold ] && { log "hold file $D0/hold present; waiting"; while [ -e $D0/hold ]; do sleep $POLL; done; log "hold released"; }
  local said=0
  while :; do
    local busy; busy=$(systemctl --user list-units --state=active,activating --no-legend --plain 'seq-vp-*' 'tag-*' 2>/dev/null \
                       | awk '{print $1}' | grep -v "^${UNITP}-" | grep -v "$SELF_RE" | tr '\n' ' ')
    [ -z "${busy// /}" ] && break
    [ $said -eq 0 ] && log "waiting: other units active: $busy"; said=1; sleep $POLL
  done
  [ $said -eq 1 ] && log "other units ended"
  said=0                                                  # GPU jobs that are not systemd units (e.g. a training)
  while :; do
    local apps util; apps=$($SMI --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' | tr '\n' ' ')
    util=$($SMI --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    [ -z "${apps// /}" ] && [ "${util:-0}" -le 20 ] && break
    if [ $said -eq 0 ] || [ $((said % 30)) -eq 0 ]; then
      log "waiting: GPU busy (compute pids: ${apps:-none}; utilization ${util}%; $(for p in $apps; do ps -o args= -p $p 2>/dev/null | cut -c1-120; done | tr '\n' ';'))"
    fi
    said=$((said+1)); sleep $POLL
  done
  [ $said -gt 0 ] && log "GPU free"
  local w=0; until awk '{exit !($1 < 1.5)}' /proc/loadavg || [ $w -ge 30 ]; do sleep $POLL; w=$((w+1)); done
}

records() {  # records <run> <arm> <idx> <pair> <attempt> <unit> <run_id>
  local run=$1 arm=$2 idx=$3 pair=$4 att=$5 unit=$6 rid=$7
  { echo "repo: $REPO"; echo "repo HEAD: $(git -C $REPO rev-parse HEAD)"; echo "repo git diff --stat:"; git -C $REPO diff --stat
    echo "repo untracked (experiments/d0, docs/bayesreg, configs/bayesreg, runs):"; git -C $REPO status --porcelain -- experiments/d0 docs/bayesreg configs/bayesreg runs
    echo "frozen worktree: $W"; echo "frozen HEAD: $(git -C $W rev-parse HEAD)"; echo "frozen git status --porcelain (must be empty):"; git -C $W status --porcelain
    echo "experiments/d0 file sha256:"; (cd $REPO && sha256sum experiments/d0/*.py experiments/d0/*.sh experiments/d0/*.json); } > "$run/git.txt" 2>&1
  cp "$ENVTXT" "$run/env.txt"
  [ $MODE != b1 ] || $PY -B $REPO/experiments/d0/b1_spec.py arm-record "$B1SPEC" "$arm" > "$run/b1_arm.json"
  $PY -B - "$run" "$arm" "$idx" "$pair" "$(stage_of $arm)" "$(input_arm "$arm")" "$(input_record "$arm")" "$att" "$unit" "$rid" <<'EOF' > "$run/data.txt"
import hashlib, json, sys
from pathlib import Path
run, arm, idx, pair, st, in_arm, in_rec, att, unit, rid = sys.argv[1:]
run, st = Path(run), Path(st)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
man = json.loads((run/'inputs'/pair/'manifest.json').read_text())
lines = [f'run_id: {rid}', f'arm: {arm}', f'pair index: {idx}', f'pair: {pair}', f'attempt: {att}', f'unit: {unit}',
         f"fixed subject: {man['fixed_subject']}", f"moving subject: {man['moving_subject']}",
         f'arm stage directory (prepare source): {st}',
         f'arm input pair.npz: {st/pair/"pair.npz"} sha256 {sha(st/pair/"pair.npz")}',
         f'arm input manifest.json: {st/pair/"manifest.json"} sha256 {sha(st/pair/"manifest.json")}',
         f'run copy pair.npz: {run/"inputs"/pair/"pair.npz"} sha256 {sha(run/"inputs"/pair/"pair.npz")}',
         f'run copy manifest.json: sha256 {sha(run/"inputs"/pair/"manifest.json")}']
pd = man.get('preprocessing_d0')
if pd:
    lines += [f"original pair.npz sha256: {pd['source_pair_sha256']}", f"original manifest sha256: {pd['source_manifest_sha256']}",
              f"colouring: {pd['recipe']}", f"oracle (truth labels used for the colouring): {pd['oracle']}",
              f"inputs record: {in_rec} sha256 {sha(in_rec)}"]
    if in_arm == 'A2':
        for side in ('fixed', 'moving'):
            c = pd['colour'][side]
            lines.append(f"atropos {side} {c['subject']}: provenance sha256 {c['atropos_provenance_sha256']}")
    if 'colour_table' in pd:
        lines.append(f"colour table: {pd['colour_table']['table_file']} sha256 {pd['colour_table']['table_sha256']}")
    if 'unet_provenance_sha256' in (pd.get('colour') or {}).get('fixed', {}):
        for side in ('fixed', 'moving'):
            c = pd['colour'][side]
            lines.append(f"unet {side} {c['subject']}: provenance sha256 {c['unet_provenance_sha256']}; policy {c['policy']}; "
                         f"models {c['model_ids']}; registry {c['registry_status']} sha256 {c['registry_sha256']}")
else:
    lines += [f'colouring: none ({arm}, original images; the stage files of data/three_d/stage_L2R_val_v1)']
if in_arm != arm:                                          # B1: the arm's inputs are those of an earlier arm; b1_arm.json has its options
    lines.append(f'arm of the inputs record: {in_arm}')
lines.append('truth labels for evaluation: the original stage pair.npz (evaluate.py checks its sha256)')
print('\n'.join(lines))
EOF
  { echo "colouring seed: 20261030 (numpy default_rng; used only by A3: z_l = uniform(-1, 1, 35)); A0/A1/A2 colourings have no random step"
    if [ "$arm" = A4p ]; then echo "A4p (protocol §12.3): the same z_l as A3; the U-Net inference has no random step (eval mode)"; fi
    if [ "$arm" = A1Ph ] || [ "$arm" = A1Ps ]; then echo "$arm (runs/20261102-a1p/plan.md): no z_l; the U-Net inference has no random step (eval mode)"; fi
    if [ "$arm" = A5 ] || [ "$arm" = A5P ]; then echo "$arm (runs/20261103-a5/plan.md): no z_l, the fixed colour table; the U-Net inference has no random step (eval mode)"; fi
    if [ $MODE = b1 ]; then echo "$arm (runs/20261103-b1/plan.md): the inputs of $(input_arm "$arm"), no new colouring, no random step"; fi
    echo "Atropos (A2 posteriors): -r 0 (constant internal seed)"
    echo "solver: configuration C has no random number generator or seed option; GPU reductions may still differ between runs"; } > "$run/seed"
}

postprocess() {  # postprocess <run> <arm> <pair> <t0_uptime> <rid> -> 0 continue, 1 STOP
  local run=$1 arm=$2 pair=$3 t0=$4 rid=$5 nd no cls line
  nd=$(dmesg_since "$t0" 'create_allocation failed'); no=$(dmesg_since "$t0" 'Out of memory|oom-kill|Killed process')
  [ -e "$run/classification.json" ] || $PY -B $REPO/experiments/d0/classify_run.py "$run" --watchdog $WATCHDOG --dxg "$nd" --oom "$no" --out "$run/classification.json" >> "$run/log.txt" 2>&1
  cls=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['class'])" "$run/classification.json" 2>/dev/null)
  echo "$(utc) class $cls (dxg $nd, oom $no)" >> "$run/log.txt"
  case "$cls" in
    ok|handover_only|method_failure)
      $PY -B $REPO/experiments/d0/evaluate.py "${EVAL_CMD[@]}" "$run" --arm "$arm" --pair "$pair" --run-id "$rid" >> "$run/log.txt" 2>&1 \
        || { log "STOP: evaluation failed for $run (see log.txt)"; return 1; }
      if [ "$cls" != method_failure ]; then
        $PY -B - "$run" "${KEEP_NPZ[@]}" <<'EOF' >> "$run/log.txt" 2>&1 || { log "STOP: cleanup failed for $run"; return 1; }
import json, sys
from pathlib import Path
run = Path(sys.argv[1]); proto = json.loads((run/'protocol.json').read_text()); task = run/proto['task']['name']
keep = set(sys.argv[2:])                                   # --x1: the native seed (gate A plan, revision 1); otherwise empty
gone, kept = [], []
for p in sorted(task.glob('*.npz')):
    if p.name.startswith(('committed_', 'level_')) and p.name != 'final_native.npz' and p.name not in keep:
        gone.append(dict(name=p.name, bytes=p.stat().st_size)); p.unlink()
    else:
        kept.append(dict(name=p.name, bytes=p.stat().st_size))
rec = dict(rule='D0 protocol section 6: only the intermediate commit maps committed_*.npz and level_*.npz of the task directory are deleted (runs classified ok or handover_only); everything else is kept',
           deleted=gone, deleted_bytes=sum(g['bytes'] for g in gone), kept_npz=kept)
if keep:                                                   # the rule above says level_*.npz are deleted; the other modes' record is unchanged
    rec['rule_exception'] = 'gate A plan, revision 1 (--x1): ' + ', '.join(sorted(keep)) + ' matches level_*.npz and is kept, the native seed that evaluate.py reports'
(run/'cleanup.json').write_text(json.dumps(rec, indent=1)+'\n'); print('cleanup', len(gone), 'files', rec['deleted_bytes'], 'bytes; kept', [k['name'] for k in kept])
EOF
      fi
      line=$($PY -c "
import json,sys
m=json.load(open(sys.argv[1]+'/metrics.json')); x=m.get('metrics') or {}; c=m.get('certificate') or {}
f=lambda v,n=4: '-' if v is None else f'{v:.{n}f}'
print(f\"{m['classification']['class']} status {m.get('status')} seg35 {f(x.get('seg35'))} seg4 {f(x.get('seg4'))} hd95 {f(x.get('hd95'),3)} sdlogj {f(x.get('sdlogj_official'))} brain {f(x.get('sdlogj_brain'),3)} cert {c.get('status')}/{c.get('passed')} selfcheck {(m.get('self_check') or {}).get('passed')} e2e {f(m.get('e2e_seconds'),1)} native {m.get('native_commits')}\")" "$run")
      $PY -c "import json,sys,datetime;json.dump(dict(done_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),summary=sys.argv[2]),open(sys.argv[1]+'/queue_done.json','w'),indent=1)" "$run" "$line"
      log "done $arm $(basename "$run"): exit $(cat "$run/exit_code" 2>/dev/null) $line; /home free $(homefree) GB"
      return 0;;
    *) log "STOP: $run classified ${cls:-unknown} ($(sed -n 's/.*"reason": "\(.*\)",*/\1/p' "$run/classification.json" | head -1)); kept whole, nothing rerun"; return 1;;
  esac
}

one_run() {  # one_run <arm> <idx>
  local arm=$1 idx=$2 pair=${PAIRS[$2]} short run att unit rid t0 rc prep_cmd sd_cmd
  short=$(short_of "$pair")
  [ $MODE != b1 ] || WATCHDOG=${B1_WD[$arm]}                 # also for the classification of a resumed run
  local a0=$RUNS/$arm/$pair a1=$RUNS/$arm/${pair}__r1
  if [ -f "$a1/queue_done.json" ] || [ -f "$a0/queue_done.json" ]; then echo "$(utc) skip $arm $pair (done)"; return 0; fi
  run=$a0; att=0; unit=$UNITP-$arm-$short
  if [ -e "$a0" ]; then
    if [ -e "$a1" ]; then run=$a1; att=1; unit=$unit-r1
    elif [ -e "$a0/RETRY_APPROVED" ]; then run=$a1; att=1; unit=$unit-r1; log "retry approved for $a0: rerun once in $a1"
    fi
  fi
  if [ -e "$run" ]; then                                   # resume an interrupted queue
    while systemctl --user is-active --quiet "$unit"; do sleep 10; done
    if [ -f "$run/exit_code" ]; then
      log "resume: $run has an exit_code; post-processing"
      postprocess "$run" "$arm" "$pair" "$(cat "$run/uptime_before" 2>/dev/null || echo 0)" "$(cat "$run/run_id" 2>/dev/null)"; return $?
    fi
    log "STOP: $run exists without exit_code or queue_done.json (incomplete attempt kept whole; the lead may create $a0/RETRY_APPROVED for one rerun)"
    return 1
  fi
  wait_turn
  local free; free=$(homefree); [ "$free" -ge $FLOOR_GB ] || { log "STOP: /home free $free GB < $FLOOR_GB GB before $arm $pair"; return 1; }
  rid=${RIDP}_${arm}_${short}_$(date -u +%Y%m%dT%H%M%SZ)$([ $att -eq 1 ] && echo _r1)
  mkdir -p "$RUNS/$arm"
  set_arm_opts "$arm"
  build_prep_cmd
  exec 9> "$LINK_LOCK"; flock 9                            # shared by the D0 and --a4 modes (A4 review item 1)
  local target; target=$(stage_of $arm)
  ln -sfn "$target" "$LINK.tmp" && mv -Tf "$LINK.tmp" "$LINK" && [ "$(readlink "$LINK")" = "$target" ] || { flock -u 9; log "STOP: cannot point $LINK at $target"; return 1; }
  local prep_out; prep_out=$("${prep_cmd[@]}" 2>&1); rc=$?
  restore_link; flock -u 9
  [ -d "$run" ] || { log "STOP: prepare failed for $arm $pair (exit $rc): $(echo "$prep_out" | tail -2 | tr '\n' ' ')"; return 1; }
  { echo "$(utc) prepare (link $LINK -> $target), exit $rc"; echo "$prep_out"; } > "$run/log.txt"
  [ $rc -eq 0 ] || { log "STOP: prepare exit $rc for $run"; return 1; }
  $PY -B $REPO/experiments/d0/check_prepared.py "$run" --pair "$pair" --stage "$STAGE_PREFIX-$arm" "${ARM_OPTS[@]}" \
      "${CHECK_OPTS[@]}" --expect-pair-sha256 "${EXP_PAIR[$arm:$pair]}" --expect-manifest-sha256 "${EXP_MAN[$arm:$pair]}" > "$run/prepared_check.json" 2>&1 \
      || { log "STOP: prepared run does not match configuration C / the arm inputs: $(cat "$run/prepared_check.json" | cut -c1-300)"; return 1; }
  echo "$rid" > "$run/run_id"
  records "$run" "$arm" "$idx" "$pair" "$att" "$unit" "$rid"
  build_sd_cmd
  { echo "# prepare (run from any directory; the frozen script resolves its root from its own path)"; printf '%q ' "${prep_cmd[@]}"; echo
    echo "# worktree link during prepare: $LINK -> $target"
    echo "# run"; printf '%q ' "${sd_cmd[@]}"; echo; } > "$run/cmd.txt"
  echo "$unit" > "$run/service_unit.txt"
  t0=$(cut -d' ' -f1 /proc/uptime); echo "$t0" > "$run/uptime_before"
  echo "$(utc) launch $unit; loadavg $(cut -d' ' -f1-3 /proc/loadavg); MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB; gpu $(gpustate)" >> "$run/log.txt"
  log "run $arm $pair (index $idx, attempt $att) unit $unit; /home free $free GB"
  "${sd_cmd[@]}" >> "$run/log.txt" 2>&1; rc=$?
  echo "$(utc) unit ended, systemd-run exit $rc; gpu $(gpustate)" >> "$run/log.txt"
  postprocess "$run" "$arm" "$pair" "$t0" "$rid"
}

n=0
for x in "${ORDER[@]}"; do
  n=$((n+1)); one_run "${x%%:*}" "${x##*:}" || { log "queue stopped at item $n/${#ORDER[@]} (${x%%:*} ${PAIRS[${x##*:}]})"; exit 8; }
done
log "queue finished: ${#ORDER[@]} items"
if [ -z "$ONLY" ] && [ $MODE = a4 ]; then
  $PY -B $REPO/experiments/d0/evaluate.py summarize-a4 --a4-runs-root "$RUNS" --d0-runs-root $D0V1/runs --out-dir $D0/analysis \
      > $D0/analysis_summarize.log 2>&1
  log "summary exit $? -> $D0/analysis/{long.csv,summary_a4.md,decision_a4.json} (A0-A3 read from $D0V1/runs)"
elif [ -z "$ONLY" ] && [ $MODE = a1p ]; then
  $PY -B $REPO/experiments/d0/evaluate.py summarize-a1p --a1p-runs-root "$RUNS" --d0-runs-root $D0V1/runs --a4-runs-root $RESULTS/d0_v1_a4/runs \
      --out-dir $D0/analysis > $D0/analysis_summarize.log 2>&1
  log "summary exit $? -> $D0/analysis/{long.csv,summary_a1p.md,decision_a1p.json} (A0-A3 read from $D0V1/runs, A4p from $RESULTS/d0_v1_a4/runs)"
elif [ -z "$ONLY" ] && [ $MODE = x1 ]; then
  $PY -B $REPO/experiments/d0/evaluate.py summarize-x1 --x1-runs-root "$RUNS" --d0-runs-root $D0V1/runs --pilot-root $RESULTS/path_pilot_v1 \
      --out-dir $D0/analysis > $D0/analysis_summarize.log 2>&1
  log "summary exit $? -> $D0/analysis/{long.csv,summary_x1.md,decision_x1.json} (A0 read from $D0V1/runs, P0n from $RESULTS/path_pilot_v1)"
elif [ -z "$ONLY" ] && [ $MODE = a5 ]; then
  $PY -B $REPO/experiments/d0/evaluate.py summarize-a5 --a5-runs-root "$RUNS" --d0-root $D0V1 --a4-root $RESULTS/d0_v1_a4 --a1p-root $RESULTS/d0_v1_a1p \
      --out-dir $D0/analysis >> $D0/analysis_summarize.log 2>&1
  log "summary exit $? -> $D0/analysis/{long.csv,summary_a5.md,decision_a5.json} (A0, A1, A3 from $D0V1/analysis/long.csv, A4p from d0_v1_a4, A1Ph from d0_v1_a1p)"
  $PY -B -c "import json,sys;d=json.load(open(sys.argv[1]));print('first gate:',d['gate1']['text']);print('second gate:',d['gate2']['text']);print('reading:',d['gate2']['reading_text'])" "$D0/analysis/decision_a5.json" 2>&1 | tee -a "$QLOG"
elif [ -z "$ONLY" ] && [ $MODE = b1 ]; then
  $PY -B $REPO/experiments/d0/evaluate.py summarize-b1 --b1-root "$D0" --selection-spec $REPO/runs/20261103-b1/arms_selection.json --a5-root $RESULTS/d0_v1_a5 \
      --out-dir $D0/analysis >> $D0/analysis_summarize.log 2>&1
  log "summary exit $? -> $D0/analysis/{long.csv,summary_b1.md,decision_b1.json} (A5P and A5 from $RESULTS/d0_v1_a5/analysis/long.csv)"
  $PY -B -c "import json,sys;d=json.load(open(sys.argv[1]));print('selection:',d['selection']['text']);print('winner:',d['winner']);print('reading:',d['readings']['text'])" "$D0/analysis/decision_b1.json" 2>&1 | tee -a "$QLOG"
elif [ -z "$ONLY" ]; then
  $PY -B $REPO/experiments/d0/evaluate.py summarize --runs-root "$RUNS" --out-dir $D0/analysis > $D0/analysis_summarize.log 2>&1
  log "summary exit $? -> $D0/analysis/{long.csv,summary.md,decision.json}"
fi
log "end"
