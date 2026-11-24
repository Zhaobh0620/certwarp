#!/usr/bin/env bash
# AUD queue (runs/20261102-fb-audit/plan.md): the five development pairs, one process at a time, CPU only, single thread,
# nice 19.  Run from a git archive snapshot: experiments/aud/queue.sh <results root> (the snapshot's own code is used).
# A failing pair stops the queue; nothing is restarted automatically (hard constraint 4).
set -u
ROOT=${1:?results root}
CODE=$(cd "$(dirname "$0")/../.." && pwd)
PY=/home/rmcfz/miniconda3/envs/sci310/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$ROOT/logs"
{
  echo "code $CODE"
  echo "snapshot $(cat "$CODE/SNAPSHOT_SHA" 2>/dev/null)"
  echo "started_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cmd $0 $*"
  $PY -c 'import sys, numpy, scipy; print("python", sys.version.split()[0], "numpy", numpy.__version__, "scipy", scipy.__version__)'
  free -g
  df -h /home
} >> "$ROOT/env.txt" 2>&1
cd "$CODE" || exit 1
for pair in 0438-0439 0439-0440 0440-0441 0441-0442 0442-0443; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start $pair" >> "$ROOT/queue.log"
  nice -n 19 "$PY" -B experiments/aud/fb_audit.py --pair "$pair" --out "$ROOT" >> "$ROOT/logs/$pair.log" 2>&1
  rc=$?
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end $pair exit $rc" >> "$ROOT/queue.log"
  if [ "$rc" -ne 0 ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) STOP" >> "$ROOT/queue.log"
    exit "$rc"
  fi
done
nice -n 19 "$PY" -B experiments/aud/analyze.py --root "$ROOT" >> "$ROOT/logs/analyze.log" 2>&1
rc=$?
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) analyze exit $rc; done" >> "$ROOT/queue.log"
exit "$rc"
