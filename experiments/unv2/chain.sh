#!/usr/bin/env bash
# The chain of the full-training-set segmenter unet35_v2_full (work package UNV2, runs/20261103-unv2/任务书.md, plan.md;
# docs/bayesreg/04_分割器.md §14). Same network, loss, augmentation and steps as unet35_v1; only the training data differ.
# Members (fold, seed), strictly serial in this order: A0 B0 A1 B1.
#
#   chain.sh launch [--no-start-after ISO] [--no-start] [--out DIR] [--repo DIR]     run by the lead; returns after the checks
#   chain.sh status [--out DIR]
#   chain.sh dry-run --work DIR [--no-start-after ISO] [--stub-cli FILE]             synthetic, CPU, no unit, no snapshot
#   (internal) chain.sh wait-and-start --out DIR ...     the detached waiter started by launch
#   (internal) chain.sh chain --out DIR ...              inside systemd unit tag-unv2-chain
#
# launch, in the foreground (any failure: exit 1, nothing started):
#   1. the repository is clean: `git status --porcelain` empty;
#   2. the code snapshot $OUT/snapshot = `git archive HEAD` of the code paths (snapshot --profile v2_full), made once, verified on
#      every later launch and reused whatever the repository HEAD is now, so all members run the same code;
#   3. `cli check-inputs --profile v2_full --require-snapshot` from the snapshot: folds_full_v2.json pinned and re-derived from
#      folds.json, the repository allowlist with its sources re-hashed, training source bayesreg_train_v1 == the 106, label guard
#      constructible; no label or image is read;
#   4. env record $OUT/env_<UTC>.txt;
#   5. unless --no-start: detaches the waiter (setsid nohup; log $OUT/launch.log) and prints its PID.
# waiter: waits while any tag-* unit other than tag-unv2-* or any seq-vp-* unit is active, while the GPU has < MIN_FREE_MIB MiB free
#   or > 20 % utilization, and while MemAvailable < MIN_AVAIL_GIB GiB (polls every 60 s, logs every 30 min); then checks /home free
#   >= HOME_FLOOR_GB and starts `systemd-run --user --unit tag-unv2-chain --collect` with the snapshot as working directory, running
#   the snapshot's copy of this script. The waiter is deliberately not a tag-* unit: the D0 queues wait for every active tag-* unit
#   other than their own, so a unit that itself waited for them would deadlock with them (as run_train.sh of v1). A deadline that
#   passes while it waits: nothing is started. Whenever the waiter ends without starting the unit it writes the time and the reason to
#   $OUT/waiter_gave_up and exits 1, so that whoever waits for it can see it; a new waiter removes an old marker.
# chain (log $OUT/chain.log; the first failure stops the chain, nothing is rerun automatically, hard rule 4):
#   check-inputs; for A0 B0 A1 B1: [deadline passed: no new member] [wait: no seq-vp-* unit, GPU free] train -> calibrate;
#   registrable only with at least one calibrated member per fold, otherwise STOP (the members are kept); check-members -> register
#   into $OUT/segmenters.json (staging; after review the lead registers into the repository: the last line of the chain prints the
#   command) -> infer the 20 development subjects (before its step, a probe of the inference's machine gate: a refusal stops the chain
#   without a marker, so it can be launched again) -> eval_seg -> $OUT/eval_seg/, with the preregistered decision.run_a5pv2 written to
#   chain.log and $OUT/decision_run_a5pv2.txt, shown by `status`. Idempotent: markers in $OUT/chain_state/ skip what is finished, a step that started and did not finish stops the
#   chain and is kept whole (the lead decides); after the registration no new member is started (the registry is append-only).
#   --no-start-after ISO (with an explicit offset, e.g. 2026-11-04T06:00:00-06:00): at or after it no new member starts; a member
#   already started runs to the end of its calibration.
# dry-run: the steps of `chain` on synthetic subjects (experiments/unv2/synthetic_tree.py), the training device cpu and
#   CUDA_VISIBLE_DEVICES empty; no waiting, no unit, no snapshot, no real data; --stub-cli replaces the Python command line (tests).
set -u
REPO=/home/rmcfz/code/python_projects/seq_vp_tag
OUT=/home/rmcfz/seq_vp_tag_results/seg/unet35_v2_full
V1_PRED=/home/rmcfz/seq_vp_tag_results/seg/unet35_v1/pred
PY=/home/rmcfz/miniconda3/envs/sci310/bin/python
SMI=${UNV2_SMI:-/usr/lib/wsl/lib/nvidia-smi}   # UNV2_SMI: tests only (tests/test_bayesreg/segnet/test_unv2.py); unset for real runs
UNIT=tag-unv2-chain
MEMBERS=(A0 B0 A1 B1)
MIN_FREE_MIB=${MIN_FREE_MIB:-14336}      # 14 GiB, as v1
MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-8}        # 106 subjects in memory, about 3.7 GB (v1: 53 subjects, 6 GiB)
HOME_FLOOR_GB=${HOME_FLOOR_GB:-100}      # project rules
MAX_UTIL=20
POLL=60
START=1
DEADLINE=
WORK=
STUB=
CMD=${1:-}
[ $# -gt 0 ] && shift
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2;; --repo) REPO="$2"; shift 2;; --no-start) START=0; shift;;
    --no-start-after) DEADLINE="$2"; shift 2;;
    --work) WORK="$2"; shift 2;; --stub-cli) STUB="$2"; shift 2;;
    *) echo "unknown option $1"; exit 2;;
  esac
done
[ "$CMD" = dry-run ] && OUT=$WORK/out
SNAP=$OUT/snapshot
STATE=$OUT/chain_state
utc() { date -u +%FT%TZ; }
log() { echo "$(utc) $*"; }
die() { log "STOP: $*"; exit 1; }

if [ -n "$DEADLINE" ]; then  # a value without an explicit offset is refused (no guessing of the time zone)
  case "$DEADLINE" in
    *Z|*[+-][0-9][0-9]:[0-9][0-9]) ;;
    *) die "--no-start-after $DEADLINE: an ISO time with an explicit offset is required (e.g. 2026-11-04T06:00:00-06:00)";;
  esac
  DEADLINE_EPOCH=$(date -d "$DEADLINE" +%s 2>/dev/null) || die "--no-start-after $DEADLINE: not a time"
fi
past_deadline() {  # 0 when the deadline has passed
  [ -n "$DEADLINE" ] || return 1
  [ "$(date +%s)" -ge "$DEADLINE_EPOCH" ]
}

busy_reason() {  # busy_reason launch|member -> prints why the GPU job may not start yet (empty: may start)
  local units free util avail
  if [ "$1" = launch ]; then
    units=$(systemctl --user list-units --state=active,activating --no-legend --plain 'tag-*' 'seq-vp-*' 2>&1) \
      || { echo "systemctl failed: $units"; return; }
    units=$(echo "$units" | awk '{print $1}' | grep -v '^tag-unv2-' | tr '\n' ' ')
  else
    units=$(systemctl --user list-units --state=active,activating --no-legend --plain 'seq-vp-*' 2>&1) \
      || { echo "systemctl failed: $units"; return; }
    units=$(echo "$units" | awk '{print $1}' | tr '\n' ' ')
  fi
  [ -n "${units// /}" ] && { echo "units active: $units"; return; }
  free=$($SMI --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  util=$($SMI --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  case "$free" in ''|*[!0-9]*) echo "nvidia-smi gave no free-memory reading ($free)"; return;; esac
  [ "$free" -lt "$MIN_FREE_MIB" ] && { echo "GPU free ${free} MiB < ${MIN_FREE_MIB} MiB"; return; }
  case "$util" in ''|*[!0-9]*) echo "nvidia-smi gave no utilization reading ($util)"; return;; esac
  [ "$util" -gt "$MAX_UTIL" ] && { echo "GPU utilization ${util}% > ${MAX_UTIL}%"; return; }
  avail=$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)
  [ "$avail" -lt "$MIN_AVAIL_GIB" ] && { echo "MemAvailable ${avail} GiB < ${MIN_AVAIL_GIB} GiB"; return; }
  return 0
}

wait_until_free() {  # wait_until_free launch|member [stop-at-deadline]: returns 3 when the deadline passes while waiting
  local reason n=0
  while :; do
    reason=$(busy_reason "$1")
    [ -z "$reason" ] && break
    if [ "${2:-}" = stop-at-deadline ] && past_deadline; then log "deadline $DEADLINE passed while waiting ($reason)"; return 3; fi
    [ $((n % 30)) -eq 0 ] && log "waiting ($1): $reason"
    n=$((n+1)); sleep $POLL
  done
  [ $n -gt 0 ] && log "free after $n polls"
  log "GPU: $($SMI --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader 2>&1 | head -1)"
  return 0
}

env_record() {
  { echo "recorded: $(utc)"; echo "python: $($PY -c 'import sys;print(sys.version.replace(chr(10)," "))')"
    echo "conda env: $(dirname "$(dirname "$PY")")"
    (cd "$SNAP" && CUDA_VISIBLE_DEVICES= $PY -B -c "
import importlib
for m in ('numpy', 'scipy', 'torch', 'nibabel'):
    print(m, importlib.import_module(m).__version__)
import torch
print('torch.version.cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version())" 2>&1)
    echo "nvidia-smi: $($SMI --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader 2>&1)"
    echo "cpu: $(lscpu | sed -n 's/^Model name:[ ]*//p'); nproc $(nproc)"; echo "kernel: $(uname -r)"
    echo "memory: $(free -g | awk '/Mem:/{print $2" GiB total, "$7" GiB available"}')"
    echo "disk: $(df -h --output=avail,target /home | tail -1)"
    echo "snapshot: $(sed -n 's/^ *"commit": "\(.*\)",/\1/p' "$SNAP/SNAPSHOT.json")"
    echo "no start after: ${DEADLINE:-none}"; } > "$1"
}

passed_on() {  # the options of the chain, for the waiter and the unit
  [ -n "$DEADLINE" ] && printf '%s\n' --no-start-after "$DEADLINE"
  return 0
}

launch() {
  [ -z "${UNV2_SMI:-}" ] || die "UNV2_SMI is set ($UNV2_SMI): a test hook, the waiter would inherit it; unset it"
  mkdir -p "$OUT" || die "cannot create $OUT"
  exec 9> "$OUT/.launch.lock"; flock -n 9 || die "another launch holds $OUT/.launch.lock"
  systemctl --user is-active --quiet $UNIT && die "unit $UNIT is already active"
  pgrep -f "chain.sh wait-and-start --out $OUT" > /dev/null && die "a waiter for $OUT is already running"
  local dirty; dirty=$(git -C "$REPO" status --porcelain) || die "git status failed in $REPO"
  [ -z "$dirty" ] || die "repository $REPO is not clean (git status --porcelain):
$dirty"
  local head; head=$(git -C "$REPO" rev-parse HEAD)
  if [ -e "$SNAP" ]; then
    (cd "$SNAP" && $PY -B -m bayesreg.seg.unet.snapshot verify --root "$SNAP" > /dev/null) || die "existing snapshot $SNAP does not verify"
    log "reusing snapshot $SNAP (commit $(sed -n 's/^ *"commit": "\(.*\)",/\1/p' "$SNAP/SNAPSHOT.json"); repository HEAD $head)"
  else
    (cd "$REPO" && $PY -B -m bayesreg.seg.unet.snapshot make --repo "$REPO" --dest "$SNAP" --profile v2_full) || die "snapshot failed"
    log "snapshot $SNAP made from commit $head"
  fi
  (cd "$SNAP" && CUDA_VISIBLE_DEVICES= $PY -B -m bayesreg.seg.unet.cli check-inputs --profile v2_full --require-snapshot) || die "pinned-input check failed"
  local envtxt; envtxt=$OUT/env_$(date -u +%Y%m%dT%H%M%SZ).txt
  env_record "$envtxt"; log "env record $envtxt"
  [ $START -eq 1 ] || { log "--no-start: checks passed, nothing started"; return 0; }
  mapfile -t PASS < <(passed_on)
  setsid nohup /bin/bash "$SNAP/experiments/unv2/chain.sh" wait-and-start --out "$OUT" "${PASS[@]}" >> "$OUT/launch.log" 2>&1 < /dev/null &
  log "waiter started (pid $!; log $OUT/launch.log); unit $UNIT starts when no other tag-* unit is active and the GPU is free; no start after: ${DEADLINE:-none}"
}

give_up() {  # the waiter ends without starting the unit: a marker for whoever waits for it, then exit 1
  echo "$(utc) $*" > "$OUT/waiter_gave_up"
  die "$*; marker $OUT/waiter_gave_up"
}

wait_and_start() {
  mkdir -p "$OUT" || die "cannot create $OUT"
  rm -f "$OUT/waiter_gave_up"
  log "waiter pid $$: snapshot $SNAP; MIN_FREE_MIB $MIN_FREE_MIB, MIN_AVAIL_GIB $MIN_AVAIL_GIB; no start after: ${DEADLINE:-none}"
  wait_until_free launch stop-at-deadline
  [ $? -eq 3 ] && give_up "deadline $DEADLINE passed while waiting; nothing started"
  past_deadline && give_up "deadline $DEADLINE has passed; nothing started"
  local home; home=$(df --output=avail -B1G /home | tail -1 | tr -d ' ')
  [ "$home" -ge "$HOME_FLOOR_GB" ] || give_up "/home free $home GB < $HOME_FLOOR_GB GB; nothing started"
  systemctl --user is-active --quiet $UNIT && give_up "unit $UNIT is already active; nothing started"
  systemctl --user reset-failed $UNIT > /dev/null 2>&1   # a failed unit of the same name would block systemd-run
  mapfile -t PASS < <(passed_on)
  systemd-run --user --unit $UNIT --collect -p WorkingDirectory="$SNAP" -p MemoryMax=12G -p MemorySwapMax=4G \
    -E PYTHONDONTWRITEBYTECODE=1 -E OMP_NUM_THREADS=4 -E OPENBLAS_NUM_THREADS=4 -E MKL_NUM_THREADS=4 \
    -E NUMEXPR_NUM_THREADS=4 -E MIN_FREE_MIB="$MIN_FREE_MIB" -E MIN_AVAIL_GIB="$MIN_AVAIL_GIB" \
    /bin/bash "$SNAP/experiments/unv2/chain.sh" chain --out "$OUT" "${PASS[@]}" || give_up "systemd-run failed; nothing started"
  log "unit $UNIT started (chain log $OUT/chain.log)"
}

step() {  # step <name> <command...>: run once; a started-but-unfinished step stops the chain
  local name=$1; shift
  if [ -f "$STATE/$name.done" ]; then log "skip $name (done)"; return 0; fi
  [ -f "$STATE/$name.started" ] && die "$name started at $(cat "$STATE/$name.started") and did not finish; kept whole, not rerun"
  utc > "$STATE/$name.started"
  log "start $name: $*"
  "$@" || die "$name failed (exit $?)"
  utc > "$STATE/$name.done"
  log "done $name"
}

members_and_registration() {  # uses CLI, COMMON, TRAIN_EXTRA, DEVICE, GATE
  local m f s complete_a=0 complete_b=0
  "${CLI[@]}" check-inputs --require-snapshot "${COMMON[@]}" || die "input check failed"
  for m in "${MEMBERS[@]}"; do
    f=${m:0:1}; s=${m:1:1}
    if [ ! -f "$STATE/train_$m.done" ] && [ ! -f "$STATE/train_$m.started" ]; then
      if [ -f "$STATE/register.done" ]; then log "registered already: $m is not started (the registry is append-only)"; continue; fi
      if [ "$GATE" = 1 ]; then
        wait_until_free member stop-at-deadline
        [ $? -eq 3 ] && { log "no new member after the deadline: $m is not started"; continue; }
      fi
      if past_deadline; then log "deadline $DEADLINE has passed: $m is not started"; continue; fi
    fi
    step "train_$m" "${CLI[@]}" train --fold "$f" --seed "$s" "${COMMON[@]}" --out "$OUT" --device "$DEVICE" "${TRAIN_EXTRA[@]}"
    step "calibrate_$m" "${CLI[@]}" calibrate --fold "$f" --seed "$s" "${COMMON[@]}" --out "$OUT" --device "$DEVICE"
  done
  for m in "${MEMBERS[@]}"; do
    if [ -f "$STATE/calibrate_$m.done" ]; then
      case "$m" in A*) complete_a=$((complete_a+1));; B*) complete_b=$((complete_b+1));; esac
    fi
  done
  log "calibrated members: fold A $complete_a, fold B $complete_b"
  { [ $complete_a -ge 1 ] && [ $complete_b -ge 1 ]; } || die "not registrable: at least one calibrated member per fold is required (fold A $complete_a, fold B $complete_b); the members are kept"
  [ -f "$STATE/register.done" ] || "${CLI[@]}" check-members "${COMMON[@]}" --out "$OUT" || die "member check failed (nothing registered)"
  step register "${CLI[@]}" register "${COMMON[@]}" --out "$OUT" --registry "$REGISTRY"
}

report_decision() {  # the preregistered rule, in the log and for `status`
  local line
  line=$("$PY" -B -c 'import json, sys
d = json.load(open(sys.argv[1]))["decision"]
print("decision.run_a5pv2 =", str(d["run_a5pv2"]).lower()+"; v2 - v1 =", d["v2_minus_v1"], "(Dice35 means);", d["text"])' "$OUT/eval_seg/eval_seg.json" 2>&1) \
    || die "no preregistered decision readable from $OUT/eval_seg/eval_seg.json: $line"
  [ -f "$OUT/decision_run_a5pv2.txt" ] || echo "$(utc) $line" > "$OUT/decision_run_a5pv2.txt"   # a rerun changes nothing
  log "$line"
}

inference_and_evaluation() {  # uses CLI, INFER_COMMON, DEVICE, GATE, EVAL
  if [ "$GATE" = 1 ] && [ ! -f "$STATE/infer.done" ] && [ ! -f "$STATE/infer.started" ]; then
    wait_until_free member
    "${CLI[@]}" infer "${INFER_COMMON[@]}" --all-dev --gate-only --registry "$REGISTRY" --out "$OUT" --device "$DEVICE" \
      || die "the inference's machine gate refuses; infer not started, no marker left: launch again when the machine is free"
  fi
  step infer "${CLI[@]}" infer "${INFER_COMMON[@]}" --all-dev --registry "$REGISTRY" --out "$OUT" --device "$DEVICE"
  step eval_seg "${EVAL[@]}" --v2 "$OUT/pred" --v1 "$V1_PRED" --out-dir "$OUT/eval_seg"
  [ -n "$STUB" ] || report_decision
}

chain() {
  exec >> "$OUT/chain.log" 2>&1
  local here; here=$(cd "$(dirname "$0")/../.." && pwd)
  [ "$here" = "$(cd "$SNAP" 2>/dev/null && pwd)" ] || die "chain must run from the snapshot $SNAP (runs from $here)"
  cd "$SNAP" || die "cannot cd $SNAP"
  mkdir -p "$STATE"
  log "chain start: unit $UNIT, snapshot commit $(sed -n 's/^ *"commit": "\(.*\)",/\1/p' "$SNAP/SNAPSHOT.json"); no start after: ${DEADLINE:-none}"
  CLI=("$PY" -B -m bayesreg.seg.unet.cli)
  COMMON=(--profile v2_full)
  TRAIN_EXTRA=()
  INFER_COMMON=(--profile v2_full)
  DEVICE=cuda
  GATE=1
  REGISTRY=$OUT/segmenters.json
  EVAL=("$PY" -B experiments/unv2/eval_seg.py)
  members_and_registration
  inference_and_evaluation
  log "chain finished: staged registry $REGISTRY, predictions $OUT/pred, evaluation $OUT/eval_seg"
  log "after review, from a clean main: python -m bayesreg.seg.unet.cli register --profile v2_full --out $OUT (writes configs/bayesreg/segmenters_v2.json; then commit it)"
}

dry_run() {
  [ -n "$WORK" ] || { echo "dry-run: --work DIR is required"; exit 2; }
  local root; root=$(cd "$(dirname "$0")/../.." && pwd)
  mkdir -p "$OUT/chain_state" || die "cannot create $OUT"
  export CUDA_VISIBLE_DEVICES=
  cd "$root" || die "cannot cd $root"
  log "dry-run start: work $WORK; synthetic subjects, no unit, no snapshot, CPU; no start after: ${DEADLINE:-none}"
  if [ -n "$STUB" ]; then
    CLI=(/bin/bash "$STUB" cli); EVAL=(/bin/bash "$STUB" cli eval_seg)
  else
    [ -f "$WORK/folds_full_v2.json" ] || $PY -B experiments/unv2/synthetic_tree.py --work "$WORK" || die "synthetic tree failed"
    CLI=("$PY" -B -m bayesreg.seg.unet.cli); EVAL=("$PY" -B experiments/unv2/eval_seg.py --allowlist "$WORK/dev_subjects.json" --folds "$WORK/folds.json" --dev-source synthetic_dev --data-root "$WORK/data")
  fi
  COMMON=(--profile v2_full --synthetic --folds "$WORK/folds_full_v2.json" --allowlist "$WORK/dev_subjects.json" --training-source synthetic_bline_train --expect-counts 5,1 --data-root "$WORK/data")
  TRAIN_EXTRA=(--set base_channels=4 --set depth=2 --set max_iters=4 --set eval_every=2 --set warmup_iters=1 --set min_iters=4 --set log_every=1 --set amp=false)
  INFER_COMMON=(--profile v2_full --synthetic --allowlist "$WORK/dev_subjects.json" --dev-source synthetic_dev --data-root "$WORK/data")
  DEVICE=cpu
  GATE=0
  REGISTRY=$OUT/segmenters.json
  V1_PRED=$WORK/pred_v1
  members_and_registration
  inference_and_evaluation
  log "dry-run finished: staged registry $REGISTRY, predictions $OUT/pred, evaluation $OUT/eval_seg"
}

status() {
  systemctl --user status $UNIT --no-pager 2>&1 | head -5
  pgrep -af "chain.sh wait-and-start" || echo "no waiter running"
  ls -1 "$STATE" 2>/dev/null | tr '\n' ' '; echo
  [ -f "$OUT/waiter_gave_up" ] && echo "WAITER GAVE UP: $(cat "$OUT/waiter_gave_up")"
  [ -f "$OUT/decision_run_a5pv2.txt" ] && echo "PREREGISTERED RULE: $(cat "$OUT/decision_run_a5pv2.txt")"
  for f in "$OUT/launch.log" "$OUT/chain.log"; do [ -f "$f" ] && { echo "== $f"; tail -5 "$f"; }; done
}

case "$CMD" in
  launch) launch;;
  wait-and-start) wait_and_start;;
  chain) chain;;
  dry-run) dry_run;;
  status) status;;
  *) sed -n '2,10p' "$0"; exit 2;;
esac
