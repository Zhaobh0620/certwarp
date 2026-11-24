#!/usr/bin/env bash
# KKT diagnosis launcher: plain processes (no systemd units), nice 19, single-thread BLAS, no GPU.
# usage: experiments/kkt/run.sh <run id> <state file name> <k> <label>
set -euo pipefail
RUN_ID=$1; STATE=$2; K=$3; LABEL=$4
CVG=/home/rmcfz/seq_vp_tag_results/cvg_v1
OUT=/home/rmcfz/seq_vp_tag_results/kkt_v1
REPO=/home/rmcfz/code/python_projects/seq_vp_tag
PY=/home/rmcfz/miniconda3/envs/sci310/bin/python
mkdir -p "$OUT/launch/$LABEL"
cp "$REPO/experiments/kkt/kkt.py" "$OUT/launch/$LABEL/kkt.py"
sha256sum "$REPO/experiments/kkt/kkt.py" "$REPO/experiments/cvg/common.py" > "$OUT/launch/$LABEL/sha256.txt"
export CVG_CODE=$CVG/code/be844cdf757fa62e74449c2879302f076f336b0c
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
cd "$REPO"
nice -n 19 "$PY" -B experiments/kkt/kkt.py --run "$CVG/runs/$RUN_ID" --state "$CVG/runs/$RUN_ID/states/$STATE" \
    --k "$K" --label "$LABEL" --out "$OUT" 2>&1 | tee "$OUT/launch/$LABEL/stdout.txt"
