#!/usr/bin/env bash
# LMI L0 demo (开发对 L0 诊断): the PG arm, then the LM arm, serially, as ordinary niced processes (no systemd unit).
#   bash experiments/lmi/launch.sh <root> [wall_seconds] [start_state.npz]
# (start_state: the supplementary warm-start arms, l0_demo.py --start-state)
# Each arm directory gets cmd.txt, git.txt (HEAD and git diff), env.txt, data.txt (cache path and sha256) and stdout.txt.
set -u
ROOT=${1:?root directory}
WALL=${2:-900}
START=${3:-}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=/home/rmcfz/miniconda3/envs/sci310/bin/python
CACHE=/home/rmcfz/seq_vp_tag_results/cvg_v1/cache/0438-0439.npz
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
mkdir -p "$ROOT/launch"
for INNER in pg lm; do
  OUT="$ROOT/$INNER"
  META="$ROOT/launch/$INNER"
  mkdir -p "$META"
  CMD=(nice -n 19 "$PY" -B "$REPO/experiments/lmi/l0_demo.py" --inner "$INNER" --out "$OUT" --wall-seconds "$WALL"
       --cache "$CACHE")
  if [ -n "$START" ]; then CMD+=(--start-state "$START"); fi
  printf '# %s UTC\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${CMD[*]}" > "$META/cmd.txt"
  { echo "HEAD $(git -C "$REPO" rev-parse HEAD)"; echo "# git status --porcelain"; git -C "$REPO" status --porcelain;
    echo "# git diff HEAD"; git -C "$REPO" diff HEAD; } > "$META/git.txt"
  { "$PY" -c 'import sys, numpy, scipy, numba; print(sys.version); print("numpy", numpy.__version__, "scipy", scipy.__version__, "numba", numba.__version__)';
    env | grep -E '^(OPENBLAS|OMP|MKL|NUMEXPR|NUMBA)_NUM_THREADS|^CUDA_VISIBLE_DEVICES|^PYTHONDONTWRITEBYTECODE' | sort;
    uname -a; free -m; } > "$META/env.txt" 2>&1
  { echo "cache $CACHE"; sha256sum "$CACHE"; echo "pair stage_L2R_val_v1/OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1 (development, L2R validation)";
    if [ -n "$START" ]; then echo "start state $START"; sha256sum "$START"; fi
    echo "nothing under data/ is read"; } > "$META/data.txt"
  "${CMD[@]}" > "$META/stdout.txt" 2>&1
  echo "exit $? at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$META/stdout.txt"
done
