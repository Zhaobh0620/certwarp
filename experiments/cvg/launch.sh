#!/bin/bash
# CVG: launch one diagnostic process as the systemd user unit tag-cvg-<id> (work package CVG).
#   bash experiments/cvg/launch.sh <id> <script.py> [script args...]
# - code: the pinned `git archive` snapshot of bayesreg/, jd_curl/, configs/ (commit in $RES/code/PINNED; made once,
#   never a worktree); the scripts are copied to $RES/launch/<id>/scripts/ and run from there (sha256 in cmd.txt);
# - unit: MemoryMax=4G MemorySwapMax=1G CPUQuota=400% Nice=19 RuntimeMaxSec=3600; BLAS 1 thread, numba 4 threads,
#   CUDA hidden; one tag-cvg unit at a time (refused if another is active); --wait (the caller runs it in the
#   background) and --collect.
set -euo pipefail
REPO=/home/rmcfz/code/python_projects/seq_vp_tag
RES=/home/rmcfz/seq_vp_tag_results/cvg_v1
PY=/home/rmcfz/miniconda3/envs/sci310/bin/python
ID=$1; SCRIPT=$2; shift 2
mkdir -p "$RES/code" "$RES/launch"
if [ ! -f "$RES/code/PINNED" ]; then git -C "$REPO" rev-parse HEAD > "$RES/code/PINNED"; fi
SHA=$(cat "$RES/code/PINNED")
SNAP=$RES/code/$SHA
if [ ! -d "$SNAP" ]; then
  mkdir -p "$SNAP.tmp"
  git -C "$REPO" archive "$SHA" bayesreg jd_curl configs | tar -x -C "$SNAP.tmp"
  echo "$SHA" > "$SNAP.tmp/SNAPSHOT_SHA"
  mv "$SNAP.tmp" "$SNAP"
fi
if systemctl --user list-units --all --no-legend 'tag-cvg-*' | grep -q 'active\|activating'; then
  echo "another tag-cvg unit is active; refused" >&2; exit 3
fi
AVAIL=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
if [ "$AVAIL" -lt 6144 ]; then
  echo "MemAvailable $AVAIL MiB < 6144 MiB (4G unit + 2G margin); refused, try later" >&2; exit 5
fi
L=$RES/launch/$ID
if [ -e "$L" ]; then echo "$L exists" >&2; exit 4; fi
mkdir -p "$L/scripts"
cp "$REPO"/experiments/cvg/*.py "$REPO"/experiments/cvg/launch.sh "$L/scripts/"
{
  echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ) unit tag-cvg-$ID snapshot $SHA"
  echo "CVG_CODE=$SNAP $PY -B $L/scripts/$SCRIPT $*"
  (cd "$L/scripts" && sha256sum ./*)
  echo "# free -m"; free -m
  echo "# other units"; systemctl --user list-units --all --no-legend 'tag-*' 'seq-vp-*' || true
} > "$L/cmd.txt"
set +e
systemd-run --user --unit="tag-cvg-$ID" --wait --collect --working-directory="$RES" \
  -p MemoryMax=4G -p MemorySwapMax=1G -p CPUQuota=400% -p Nice=19 -p RuntimeMaxSec=3600 \
  -p StandardOutput="file:$L/stdout.txt" -p StandardError="file:$L/stderr.txt" \
  --setenv=CVG_CODE="$SNAP" --setenv=CUDA_VISIBLE_DEVICES= --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
  --setenv=NUMBA_NUM_THREADS=4 --setenv=PYTHONDONTWRITEBYTECODE=1 \
  $PY -B "$L/scripts/$SCRIPT" "$@" > "$L/systemd.txt" 2>&1
code=$?
echo "exit $code $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$L/cmd.txt"
exit $code
