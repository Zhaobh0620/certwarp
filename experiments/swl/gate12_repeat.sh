#!/bin/bash
# SWLM: native re-measurement of C04-6 gate 12 (fixed state; review reviews/SWLM_复核_20261101.md rounds 1 and 2; the
# gate's wording is the contract's, adopted 11-01 in d7c7476).  For the lead; a native-scale series on the GPU.
#
#   setsid nohup experiments/swl/gate12_repeat.sh <SHA> [--pairs 5] [--threads 4] [--wait-minutes 180] \
#                [--min-avail-mib 12000] [--max-gpu-mib 2000] [--dry-show] > <log> 2>&1 &
#
# Launch it with setsid, OUTSIDE any systemd unit (review R2-6; chosen over excluding the own unit from the idle gate,
# so that the gate stays the plain rule of measure.sh): the idle gate counts every active tag-* / seq-vp-* unit as busy,
# so a driver inside such a unit would wait --wait-minutes and STOP.  The script refuses to start when its own cgroup is
# a tag-* or seq-vp-* unit.  Each run is its own transient unit tag-swl-g12-t<threads>-<tag>, started and awaited here.
#
# Design: one uncounted warm-up run (support_shared, which runs every step brain runs but GaussianMStep.level_start,
# so the numba cache and the page cache are warm for both arms), then --pairs pairs (>= 5) in ABBA order: pair i odd
# brain then support_shared, pair i even support_shared then brain (B S S B B S ...).  One process per run, the same
# native seed (mem_v1/M03/run/final_native.npz, as the first run).  Threads: OPENBLAS / OMP / MKL / NUMEXPR /
# NUMBA_NUM_THREADS and --cpu-threads all follow --threads (4 is the production setting and the criterion; a 1-thread
# series of 2 pairs is descriptive only, into its own directory: its summary says "只作描述").  Every run is
# reported; nothing is added, rerun or overwritten (hard constraint 4).  A failed run (nonzero exit, or new OOM lines in
# dmesg during the run: the count after minus the count before) is recorded in runs.tsv and the series goes on; the
# summary judges the complete pairs and reports the failures per arm as a difference -- "不通过（持有）" if (a) or (b)
# already fails, else "未决（运行失败）" -- never skipped silently.  A failed warm-up counts the same way (contract:
# "含预热"; R3-2).  Only a series without failed runs and with fewer than 5 complete pairs is "只作描述" (the
# contract's order, 577b49b: holding, then failed runs, then the 5-pair rule).
# Idle gate before every run (polled every 60 s for at most --wait-minutes, else STOP), as experiments/mem/measure.sh:
# no active / activating tag-* or seq-vp-* unit, no pytest process, MemAvailable >= --min-avail-mib, /home free >=
# 100 GB, GPU memory used <= --max-gpu-mib (no other GPU job).  Launch record per run in launch.txt: the gate reading,
# free -m, /proc/vmstat thp_* and compact_*, nvidia-smi, loadavg; after the run the exit code and the new OOM lines.
# Code: a snapshot of <SHA> by `git archive` into $ROOT/code (no worktree is registered; the SHA is in
# code_commit.txt); data root and allowlist are passed explicitly (the snapshot has no data/ link).
# Output: /home/rmcfz/seq_vp_tag_results/swl_gate12_v2_t<threads>/ (must not exist): warmup.json, <arm>-<i>.json,
# their .memtrace.jsonl, runs.tsv (tag, arm, exit, oom_new), launch.txt, run.log, summary.json
# (experiments/swl/gate12_summary.py: the three criteria and the verdict, host peaks from the swl.* segments only).  Do not use the "Memory peak" line systemd-run prints (548 KiB in the first run): the memtrace fields
# and the summary are the record.
set -u
SHA=""; PAIRS=5; THREADS=4; WAIT_MIN=180; MIN_AVAIL=12000; MAX_GPU=2000; SHOW=0
SHA=${1:-}; [ $# -gt 0 ] && shift
while [ $# -gt 0 ]; do
  case "$1" in
    --pairs) PAIRS="$2"; shift 2;; --threads) THREADS="$2"; shift 2;; --wait-minutes) WAIT_MIN="$2"; shift 2;;
    --min-avail-mib) MIN_AVAIL="$2"; shift 2;; --max-gpu-mib) MAX_GPU="$2"; shift 2;; --dry-show) SHOW=1; shift;;
    *) echo "unknown option $1"; exit 2;;
  esac
done
REPO=/home/rmcfz/code/python_projects/seq_vp_tag
PY=/home/rmcfz/miniconda3/envs/sci310/bin/python
SMI=/usr/lib/wsl/lib/nvidia-smi
SEED=/home/rmcfz/seq_vp_tag_results/mem_v1/M03/run/final_native.npz      # the seed of the first run (10-31)
ROOT=/home/rmcfz/seq_vp_tag_results/swl_gate12_v2_t$THREADS
[ -n "$SHA" ] || { echo "usage: $0 <SHA> [--pairs 5] [--threads 4] ..."; exit 2; }
FULL=$(git -C "$REPO" rev-parse --verify "$SHA^{commit}") || { echo "$SHA: unknown commit"; exit 2; }
case "$PAIRS" in ''|*[!0-9]*) echo "--pairs: a number"; exit 2;; esac
[ "$THREADS" -ge 4 ] && [ "$PAIRS" -lt 5 ] && { echo "--pairs >= 5 for the criterion series (4 threads)"; exit 2; }

ORDER=(warmup:support_shared)
for i in $(seq 1 "$PAIRS"); do
  if [ $((i % 2)) -eq 1 ]; then ORDER+=("brain-$i:brain" "support_shared-$i:support_shared")
  else ORDER+=("support_shared-$i:support_shared" "brain-$i:brain"); fi
done
if [ $SHOW -eq 1 ]; then echo "$ROOT ($FULL), threads $THREADS: ${ORDER[*]}"; exit 0; fi
if grep -Eq '/(tag-|seq-vp-)[^/]*\.service' /proc/self/cgroup 2>/dev/null; then
  echo "running inside a tag-*/seq-vp-* unit ($(cut -d: -f3 /proc/self/cgroup)): start with setsid outside units"; exit 2
fi
[ "$THREADS" -eq 4 ] || echo "threads $THREADS: a descriptive series only (the gate's criteria use 4 threads)"
[ -e "$ROOT" ] && { echo "$ROOT exists (nothing is overwritten)"; exit 2; }
[ -e "$SEED" ] || { echo "$SEED missing"; exit 2; }

memavail_mib() { awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo; }
home_free_gb() { df -BG --output=avail /home | tail -1 | tr -dc 0-9; }
gpu_used_mib() { $SMI --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9; }
busy_units() { systemctl --user list-units --state=active,activating --no-legend --plain 'seq-vp-*' 'tag-*' 2>/dev/null \
                 | awk '{print $1}' | tr '\n' ' '; }
tests_running() { pgrep -f "pytest" | tr '\n' ' '; }
gate_text() {
  echo "MemAvailable $(memavail_mib) MiB (min $MIN_AVAIL); /home free $(home_free_gb) GB (min 100);" \
       "units [$(busy_units)]; pytest [$(tests_running)]; gpu used $(gpu_used_mib) MiB (max $MAX_GPU)"
}
gate_ok() {
  local gpu; gpu=$(gpu_used_mib)
  [ "$(memavail_mib)" -ge "$MIN_AVAIL" ] && [ "$(home_free_gb)" -ge 100 ] && [ -z "$(busy_units | tr -d ' ')" ] \
    && [ -z "$(tests_running | tr -d ' ')" ] && [ -n "$gpu" ] && [ "$gpu" -le "$MAX_GPU" ]
}

mkdir -p "$ROOT/code" "$ROOT/cache/numba"
git -C "$REPO" archive "$FULL" | tar -x -C "$ROOT/code" || { echo "git archive failed"; exit 2; }
echo "$FULL" > "$ROOT/code_commit.txt"
printf 'tag\tarm\texit\toom_new\n' > "$ROOT/runs.tsv"
oom_lines() { dmesg 2>/dev/null | grep -ci 'out of memory\|oom-kill\|oom_kill' || true; }
REC=$ROOT/launch.txt
echo "series $ROOT: code $FULL, threads $THREADS, pairs $PAIRS, order ${ORDER[*]}" > "$REC"
for entry in "${ORDER[@]}"; do
  tag=${entry%%:*}; arm=${entry##*:}
  deadline=$(( $(date +%s) + WAIT_MIN*60 ))
  until gate_ok; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "STOP before $tag: gate not met within $WAIT_MIN min: $(gate_text)" | tee -a "$REC"; exit 3
    fi
    echo "$(date -u +%FT%TZ) waiting before $tag: $(gate_text)" >> "$ROOT/run.log"
    sleep 60
  done
  {
    echo "== $tag ($arm) $(date -u +%FT%TZ)"
    echo "gate: $(gate_text)"
    echo "loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
    free -m
    grep -E '^(thp_|compact_)' /proc/vmstat | tr '\n' ' '; echo
    $SMI --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
  } >> "$REC"
  oom_before=$(oom_lines)
  systemd-run --user --wait --collect --unit "tag-swl-g12-t$THREADS-$tag" \
    -p MemoryMax=12G -p MemorySwapMax=4G -p CPUQuota=600% --working-directory="$ROOT/code" \
    -E OPENBLAS_NUM_THREADS="$THREADS" -E OMP_NUM_THREADS="$THREADS" -E MKL_NUM_THREADS="$THREADS" \
    -E NUMEXPR_NUM_THREADS="$THREADS" -E NUMBA_NUM_THREADS="$THREADS" -E NUMBA_CACHE_DIR="$ROOT/cache/numba" \
    -E PYTHONDONTWRITEBYTECODE=1 -E BAYESREG_MEMTRACE="$ROOT/$tag.memtrace.jsonl" -E BAYESREG_MEMTRACE_RESET_HWM=1 \
    "$PY" -B experiments/swl/fixed_state_memory.py --arm "$arm" --out "$ROOT/$tag.json" --seed-npz "$SEED" \
    --device cuda --cpu-threads "$THREADS" --data-root "$REPO/data/three_d" \
    --allowlist "$ROOT/code/configs/bayesreg/dev_subjects.json" >> "$ROOT/run.log" 2>&1
  rc=$?
  oom_new=$(( $(oom_lines) - oom_before ))
  printf '%s\t%s\t%s\t%s\n' "$tag" "$arm" "$rc" "$oom_new" >> "$ROOT/runs.tsv"
  echo "exit $rc $(date -u +%FT%TZ); new OOM lines in dmesg during the run: $oom_new" >> "$REC"
  if [ $rc -ne 0 ] || [ "$oom_new" -gt 0 ]; then
    echo "$tag failed (exit $rc, new OOM lines $oom_new): kept in runs.tsv, not rerun; the summary reports it" \
      | tee -a "$REC"
  fi
done
"$PY" -B "$ROOT/code/experiments/swl/gate12_summary.py" "$ROOT" --pairs "$PAIRS" | tee -a "$REC"
