#!/usr/bin/env bash
# Real training of the structure-layer segmenter unet35_v1 (docs/bayesreg/04_分割器.md §9; review
# docs/bayesreg/reviews/segnet_复核_20261030.md items 1 and 2). Four members (fold A/B x seed 0/1), strictly serial.
#
#   run_train.sh launch [--out DIR] [--repo DIR] [--no-start]   run by the lead; returns after the checks
#   run_train.sh status [--out DIR]
#   (internal) run_train.sh wait-and-start --out DIR            the detached waiter started by launch
#   (internal) run_train.sh chain --out DIR                     inside systemd unit tag-unet35-train
#
# launch, in the foreground (any failure: exit 1, nothing started):
#   1. the repository is clean: `git status --porcelain` empty (tracked changes or untracked files refuse);
#   2. the code snapshot $OUT/snapshot = `git archive HEAD` of the code paths (bayesreg.seg.unet.snapshot), made once
#      and verified on every later launch; an existing snapshot is reused whatever the repository HEAD is now, so all
#      members run the same code (the SHA in $OUT/snapshot/SNAPSHOT.json is the run's code version);
#   3. `cli check-inputs --require-snapshot` from the snapshot: folds sha256 pinned, the repository allowlist with its
#      sources re-hashed (reads the stage lists under data/), training source bayesreg_train_v1 == the folds, label
#      guard constructible; no label or image is read;
#   4. env record $OUT/env_<UTC>.txt (python, numpy, scipy, torch, CUDA, cuDNN, GPU, memory, disk);
#   5. unless --no-start: detaches the waiter (setsid nohup; log $OUT/launch.log) and prints its PID.
# waiter: waits while tag-d0-queue or any tag-d0-* / seq-vp-* unit is active, while the GPU has < MIN_FREE_MIB MiB
#   free or > 20 % utilization, and while MemAvailable < MIN_AVAIL_GIB GiB (polls every 60 s, logs every 30 min); then
#   checks /home free >= HOME_FLOOR_GB and starts `systemd-run --user --unit tag-unet35-train --collect` with the
#   snapshot as working directory, running the snapshot's copy of this script (`chain`). The waiter is deliberately
#   not a tag-* unit: the D0 queue waits for every active tag-* unit other than tag-d0-*, so a tag-unet35-train unit
#   that itself waited for the D0 queue would deadlock with it.
# chain (log $OUT/chain.log; set -e: the first failure stops the chain, nothing is rerun automatically):
#   check-inputs; for A0 A1 B0 B1: [wait: no seq-vp-* unit active, GPU free >= MIN_FREE_MIB] train -> calibrate;
#   check-members -> register into $OUT/segmenters.json (staging copy; after review the lead registers into the
#   repository with `cli register --registry configs/bayesreg/segmenters.json`) -> evaluate-calibration.
#   Markers in $OUT/chain_state/ let a relaunch resume: finished steps are skipped; a step that started and did not
#   finish stops the chain (its directory is kept whole; the lead decides, hard rule 4).
set -u
REPO=/home/rmcfz/code/python_projects/seq_vp_tag
OUT=/home/rmcfz/seq_vp_tag_results/seg/unet35_v1
PY=/home/rmcfz/miniconda3/envs/sci310/bin/python
SMI=/usr/lib/wsl/lib/nvidia-smi
UNIT=tag-unet35-train
MIN_FREE_MIB=${MIN_FREE_MIB:-14336}      # 14 GiB
MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-6}
HOME_FLOOR_GB=${HOME_FLOOR_GB:-20}
MAX_UTIL=20
POLL=60
START=1
CMD=${1:-}
[ $# -gt 0 ] && shift
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2;; --repo) REPO="$2"; shift 2;; --no-start) START=0; shift;;
    *) echo "unknown option $1"; exit 2;;
  esac
done
SNAP=$OUT/snapshot
STATE=$OUT/chain_state
utc() { date -u +%FT%TZ; }
log() { echo "$(utc) $*"; }
die() { log "STOP: $*"; exit 1; }

busy_reason() {  # busy_reason launch|member -> prints why the GPU job may not start yet (empty: may start)
  local units free util avail
  if [ "$1" = launch ]; then
    systemctl --user is-active --quiet tag-d0-queue && { echo "tag-d0-queue is active"; return; }
    units=$(systemctl --user list-units --state=active,activating --no-legend --plain 'tag-d0-*' 'seq-vp-*' 2>/dev/null | awk '{print $1}' | tr '\n' ' ')
  else
    units=$(systemctl --user list-units --state=active,activating --no-legend --plain 'seq-vp-*' 2>/dev/null | awk '{print $1}' | tr '\n' ' ')
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

wait_until_free() {  # wait_until_free launch|member
  local reason n=0
  while :; do
    reason=$(busy_reason "$1")
    [ -z "$reason" ] && break
    [ $((n % 30)) -eq 0 ] && log "waiting ($1): $reason"
    n=$((n+1)); sleep $POLL
  done
  [ $n -gt 0 ] && log "free after $n polls"
  log "GPU: $($SMI --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader 2>&1 | head -1)"
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
    echo "snapshot: $(sed -n 's/^ *"commit": "\(.*\)",/\1/p' "$SNAP/SNAPSHOT.json")"; } > "$1"
}

launch() {
  mkdir -p "$OUT" || die "cannot create $OUT"
  exec 9> "$OUT/.launch.lock"; flock -n 9 || die "another launch holds $OUT/.launch.lock"
  systemctl --user is-active --quiet $UNIT && die "unit $UNIT is already active"
  pgrep -f "run_train.sh wait-and-start --out $OUT" > /dev/null && die "a waiter for $OUT is already running"
  local dirty; dirty=$(git -C "$REPO" status --porcelain) || die "git status failed in $REPO"
  [ -z "$dirty" ] || die "repository $REPO is not clean (git status --porcelain):
$dirty"
  local head; head=$(git -C "$REPO" rev-parse HEAD)
  if [ -e "$SNAP" ]; then
    (cd "$SNAP" && $PY -B -m bayesreg.seg.unet.snapshot verify --root "$SNAP" > /dev/null) || die "existing snapshot $SNAP does not verify"
    log "reusing snapshot $SNAP (commit $(sed -n 's/^ *"commit": "\(.*\)",/\1/p' "$SNAP/SNAPSHOT.json"); repository HEAD $head)"
  else
    (cd "$REPO" && $PY -B -m bayesreg.seg.unet.snapshot make --repo "$REPO" --dest "$SNAP") || die "snapshot failed"
    log "snapshot $SNAP made from commit $head"
  fi
  (cd "$SNAP" && CUDA_VISIBLE_DEVICES= $PY -B -m bayesreg.seg.unet.cli check-inputs --require-snapshot) || die "pinned-input check failed"
  local envtxt; envtxt=$OUT/env_$(date -u +%Y%m%dT%H%M%SZ).txt
  env_record "$envtxt"; log "env record $envtxt"
  [ $START -eq 1 ] || { log "--no-start: checks passed, nothing started"; return 0; }
  setsid nohup /bin/bash "$SNAP/experiments/segnet/run_train.sh" wait-and-start --out "$OUT" >> "$OUT/launch.log" 2>&1 < /dev/null &
  log "waiter started (pid $!; log $OUT/launch.log); unit $UNIT starts when tag-d0-queue has ended and the GPU is free"
}

wait_and_start() {
  log "waiter pid $$: snapshot $SNAP; MIN_FREE_MIB $MIN_FREE_MIB, MIN_AVAIL_GIB $MIN_AVAIL_GIB"
  wait_until_free launch
  local home; home=$(df --output=avail -B1G /home | tail -1 | tr -d ' ')
  [ "$home" -ge "$HOME_FLOOR_GB" ] || die "/home free $home GB < $HOME_FLOOR_GB GB"
  systemctl --user is-active --quiet $UNIT && die "unit $UNIT is already active"
  systemctl --user reset-failed $UNIT > /dev/null 2>&1   # a failed unit of the same name would block systemd-run
  systemd-run --user --unit $UNIT --collect -p WorkingDirectory="$SNAP" -p MemoryMax=12G -p MemorySwapMax=4G \
    -E PYTHONDONTWRITEBYTECODE=1 -E OMP_NUM_THREADS=4 -E OPENBLAS_NUM_THREADS=4 -E MKL_NUM_THREADS=4 \
    -E NUMEXPR_NUM_THREADS=4 -E MIN_FREE_MIB="$MIN_FREE_MIB" -E MIN_AVAIL_GIB="$MIN_AVAIL_GIB" \
    /bin/bash "$SNAP/experiments/segnet/run_train.sh" chain --out "$OUT" || die "systemd-run failed"
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

chain() {
  exec >> "$OUT/chain.log" 2>&1
  local here; here=$(cd "$(dirname "$0")/../.." && pwd)
  [ "$here" = "$(cd "$SNAP" && pwd)" ] || die "chain must run from the snapshot $SNAP (runs from $here)"
  cd "$SNAP" || die "cannot cd $SNAP"
  mkdir -p "$STATE"
  log "chain start: unit $UNIT, snapshot commit $(sed -n 's/^ *"commit": "\(.*\)",/\1/p' "$SNAP/SNAPSHOT.json")"
  local CLI=("$PY" -B -m bayesreg.seg.unet.cli)
  "${CLI[@]}" check-inputs --require-snapshot || die "pinned-input check failed"
  local f s
  for f in A B; do
    for s in 0 1; do
      if [ ! -f "$STATE/train_$f$s.done" ]; then
        [ -f "$STATE/train_$f$s.started" ] || wait_until_free member
      fi
      step "train_$f$s" "${CLI[@]}" train --fold $f --seed $s --out "$OUT"
      step "calibrate_$f$s" "${CLI[@]}" calibrate --fold $f --seed $s --out "$OUT"
    done
  done
  "${CLI[@]}" check-members --out "$OUT" || die "member check failed (nothing registered)"
  step register "${CLI[@]}" register --out "$OUT" --registry "$OUT/segmenters.json"
  step evaluate_calibration "${CLI[@]}" evaluate-calibration --out "$OUT"
  log "chain finished: 4 members trained and calibrated, staged registry $OUT/segmenters.json, cross-fold calibration $OUT/calibration_crossfit.json"
}

status() {
  systemctl --user status $UNIT --no-pager 2>&1 | head -5
  pgrep -af "run_train.sh wait-and-start" || echo "no waiter running"
  ls -1 "$STATE" 2>/dev/null | tr '\n' ' '; echo
  for f in "$OUT/launch.log" "$OUT/chain.log"; do [ -f "$f" ] && { echo "== $f"; tail -5 "$f"; }; done
}

case "$CMD" in
  launch) launch;;
  wait-and-start) wait_and_start;;
  chain) chain;;
  status) status;;
  *) sed -n '2,6p' "$0"; exit 2;;
esac
