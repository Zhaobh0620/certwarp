#!/usr/bin/env bash
# One native-scale memory measurement of run_pair (work package MEM; runs/20261031-mem/plan.md).
#
#   measure.sh --id M01 --commit SHA [--max-accepted 20] [--rounds 1,1,1,1] [--wall-seconds S] [--share 1]
#              [--reset-hwm 1] [--extra "ARGS"] [--wait-minutes 180] [--min-avail-mib 12000] [--dry-show]
#
# Configuration: the H2 arm of the M2 GPU smoke v2 (runs/20261030-m2-gpu-smoke-v2; m2_gpu_smoke_v2/H2/cmd.txt) on the
# development pair stage_L2R_val_v1 0438 <- 0439, every option explicit (structure channels from unet35_v1/pred,
# tissue layer atropos_v1, --structure-infer never, gauss27+closed27 beta_S --share (default 1, H2) Simpson,
# torch64 cuda, affine_v1, jd_cell_estimate certificate, outer a), with --max-accepted N (default 20) per (level, round)
# so that the run reaches the native level quickly and ends there with cap_reached (上限收尾) and the export.
# Code: a detached git worktree $ROOT/code_<sha12> of --commit (made once, must be clean at that commit).
# Unit: tag-mem-<id>, MemoryMax=12G, MemorySwapMax=4G, RuntimeMaxSec=3600, CPUQuota=600%, 4 BLAS / OpenMP / numba
# threads as H2, NUMBA_CACHE_DIR=$ROOT/cache/numba, BAYESREG_MEMTRACE=$OUT/memtrace.jsonl (bayesreg.memtrace marks),
# BAYESREG_MEMTRACE_RESET_HWM=--reset-hwm (default 1: per-segment host peaks).
# Gate (polled every 60 s for at most --wait-minutes, else STOP): no active / activating tag-* or seq-vp-* unit except
# tag-cvg-* (lead 10-31 ~7:05 am CDT: the CVG diagnostic units -- CPU only, MemoryMax 4G -- may run in parallel; they are
# listed in the launch record); no pytest process; MemAvailable >= --min-avail-mib; /home free >= 100 GB.  Memory is
# not affected by the CPU contention; wall clocks of a measurement next to a CVG unit are not timing results.
# Side channel: experiments/mem/sample.py (1 s) -> $OUT/samples.csv.  Launch record $OUT/launch.txt.
# Output: $OUT (must not exist): run/ (run_pair's directory), memtrace.jsonl, samples.csv, launch.txt.
# Nothing is rerun or overwritten (hard constraint 4): a failed measurement keeps its directory; a new id is used.
set -u
REPO=/home/rmcfz/code/python_projects/seq_vp_tag
PY=/home/rmcfz/miniconda3/envs/sci310/bin/python
SMI=/usr/lib/wsl/lib/nvidia-smi
ROOT=/home/rmcfz/seq_vp_tag_results/mem_v1
SEG=/home/rmcfz/seq_vp_tag_results/seg
PAIR=stage_L2R_val_v1/OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1
ID=""
COMMIT=""
MAXACC=20
ROUNDS=1,1,1,1
WALL=""
SHARE=1
RESET=1
EXTRA=""
WAIT_MIN=180
MIN_AVAIL=12000
SHOW=0
while [ $# -gt 0 ]; do
  case "$1" in
    --id) ID="$2"; shift 2;; --commit) COMMIT="$2"; shift 2;; --max-accepted) MAXACC="$2"; shift 2;;
    --rounds) ROUNDS="$2"; shift 2;; --wall-seconds) WALL="$2"; shift 2;; --share) SHARE="$2"; shift 2;;
    --reset-hwm) RESET="$2"; shift 2;; --extra) EXTRA="$2"; shift 2;; --wait-minutes) WAIT_MIN="$2"; shift 2;;
    --min-avail-mib) MIN_AVAIL="$2"; shift 2;; --dry-show) SHOW=1; shift;;
    *) echo "unknown option $1"; exit 2;;
  esac
done
[ -n "$ID" ] && [ -n "$COMMIT" ] || { echo "--id and --commit are required"; exit 2; }
case "$ID" in *[!A-Za-z0-9]*) echo "--id: letters and digits only"; exit 2;; esac
SHA=$(git -C "$REPO" rev-parse --verify "$COMMIT^{commit}") || { echo "--commit $COMMIT: unknown"; exit 2; }
CODE=$ROOT/code_${SHA:0:12}
OUT=$ROOT/$ID
UNIT=tag-mem-$ID
ARGS=(--out "$OUT/run" --pair "$PAIR" --channel-source structure --logvol-points gauss27+closed27
      --logvol-closed27-share "$SHARE" --closed27-weights simpson --backend torch64 --device cuda
      --metric-jd-scaling affine_v1 --jd-cell-estimate certificate --outer a --rounds "$ROUNDS"
      --max-accepted "$MAXACC" --cpu-threads 4 --data-root "$REPO/data/three_d"
      --allowlist "$CODE/configs/bayesreg/dev_subjects.json" --tissue-root "$SEG/atropos_v1"
      --structure-root "$SEG/unet35_v1/pred" --structure-infer never)
[ -n "$WALL" ] && ARGS+=(--wall-seconds "$WALL")
# shellcheck disable=SC2206
[ -n "$EXTRA" ] && ARGS+=($EXTRA)
if [ $SHOW -eq 1 ]; then echo "$UNIT $CODE"; printf '%q ' "${ARGS[@]}"; echo; exit 0; fi
[ -e "$OUT" ] && { echo "$OUT exists (nothing is overwritten; use a new --id)"; exit 2; }

mkdir -p "$ROOT/cache/numba"
if [ ! -d "$CODE" ]; then
  git -C "$REPO" worktree add --detach "$CODE" "$SHA" >/dev/null || { echo "worktree add failed"; exit 2; }
fi
[ "$(git -C "$CODE" rev-parse HEAD)" = "$SHA" ] || { echo "$CODE is not at $SHA"; exit 2; }
[ -z "$(git -C "$CODE" status --porcelain --untracked-files=no)" ] || { echo "$CODE has tracked changes"; exit 2; }

memavail_mib() { awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo; }
home_free_gb() { df -BG --output=avail /home | tail -1 | tr -dc 0-9; }
busy_units() { systemctl --user list-units --state=active,activating --no-legend --plain 'seq-vp-*' 'tag-*' 2>/dev/null \
                 | awk '{print $1}' | tr '\n' ' '; }
blocking_units() { busy_units | tr ' ' '\n' | grep -v '^tag-cvg-' | tr '\n' ' '; }
tests_running() { pgrep -f "pytest" | tr '\n' ' '; }
gate_text() {
  echo "MemAvailable $(memavail_mib) MiB (min $MIN_AVAIL); /home free $(home_free_gb) GB (min 100);" \
       "units [$(busy_units)] (blocking [$(blocking_units)]); pytest [$(tests_running)];" \
       "gpu $($SMI --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null)"
}
gate_ok() {
  [ "$(memavail_mib)" -ge "$MIN_AVAIL" ] && [ "$(home_free_gb)" -ge 100 ] && [ -z "$(blocking_units | tr -d ' ')" ] \
    && [ -z "$(tests_running | tr -d ' ')" ]
}
deadline=$(( $(date +%s) + WAIT_MIN*60 ))
until gate_ok; do
  if [ "$(date +%s)" -ge "$deadline" ]; then echo "STOP: gate not met within $WAIT_MIN min: $(gate_text)"; exit 3; fi
  echo "$(date -u +%FT%TZ) waiting: $(gate_text)"
  sleep 60
done
mkdir -p "$OUT"
REC=$OUT/launch.txt
{
  echo "id: $ID"; echo "unit: $UNIT"; echo "code: $CODE (HEAD $SHA)"
  echo "gate $(date -u +%FT%TZ): $(gate_text)"
  echo "uptime $(cut -d' ' -f1 /proc/uptime); loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
} > "$REC"
"$PY" -B "$CODE/experiments/mem/sample.py" --unit "$UNIT" --out "$OUT/samples.csv" > "$OUT/sampler.log" 2>&1 &
SAMPLER=$!
CMD=(systemd-run --user --unit="$UNIT" --wait --collect --expand-environment=no --working-directory="$CODE"
     --property=MemoryMax=12G --property=MemorySwapMax=4G --property=CPUQuota=600% --property=RuntimeMaxSec=3600
     --setenv=OPENBLAS_NUM_THREADS=4 --setenv=OMP_NUM_THREADS=4 --setenv=MKL_NUM_THREADS=4
     --setenv=NUMEXPR_NUM_THREADS=4 --setenv=NUMBA_NUM_THREADS=4 --setenv=PYTHONDONTWRITEBYTECODE=1
     --setenv=NUMBA_CACHE_DIR="$ROOT/cache/numba" --setenv=BAYESREG_MEMTRACE="$OUT/memtrace.jsonl"
     --setenv=BAYESREG_MEMTRACE_RESET_HWM="$RESET"
     "$PY" -B -m bayesreg.cli.run_pair "${ARGS[@]}")
{ echo "systemd command:"; printf '%q ' "${CMD[@]}"; echo; } >> "$REC"
"${CMD[@]}" >> "$REC" 2>&1
RC=$?
wait $SAMPLER
{
  echo "systemd-run exit $RC"
  echo "after $(date -u +%FT%TZ): MemAvailable $(memavail_mib) MiB; gpu $($SMI --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
  echo "units after: [$(busy_units)]"
  echo "dmesg oom lines since start: $(dmesg 2>/dev/null | grep -ci 'oom' || true)"
} >> "$REC"
echo "$ID done: exit $RC ($OUT)"
exit $RC
