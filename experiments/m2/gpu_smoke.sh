#!/usr/bin/env bash
# M2 v0 GPU smoke (work package GS; runs/20261030-m2-gpu-smoke/plan.md): one development pair, three arms in series, each in
# its own systemd unit, the unified evaluation (bayesreg.eval.cli) of every arm and of the D0 A0 run in their own units,
# then one comparison table.
#
#   G1  --backend torch64 --channel-source none   --logvol-points gauss27         --metric-jd-scaling affine_v1 --outer a
#       --rounds 1,1,1,1 --wall-seconds 2700
#   G2  as G1 with --logvol-points gauss27+corners
#   G3  as G2 with --channel-source tissue
#   (DEF, 2026-10-31: the G arms also pass --jd-cell-estimate gauss27+corners, the cell estimate they ran with as
#   run_pair's default then -- run_pair's default is certificate since DEF; a code tree whose run_pair has no
#   --jd-cell-estimate, i.e. before JDC, where gauss27+corners was the only estimate, gets it left out, logged)
#
# Second batch (work package SMK2; runs/20261030-m2-gpu-smoke-v2/plan.md; contract C05-2b, D47, D48): three arms that
# differ only in the point set of the log-volume prior, all with the structure layer and the certificate cell estimate:
#   common  --backend torch64 --device cuda --channel-source structure --metric-jd-scaling affine_v1
#           --jd-cell-estimate certificate --outer a --rounds 1,1,1,1 --wall-seconds 2700 --structure-root DIR
#           --structure-infer never; OPT1 switches at their defaults (J cache on, log J cache off, speculative
#           gradient on); tau^2, the brain mask and the normalization come from the tissue layer P_T as run_pair does
#           for every channel source (C03.1-3: --tissue-root is passed as for G3)
#   H1  --logvol-points gauss27+corners                                                   (C05-2a)
#   H2  --logvol-points gauss27+closed27 --logvol-closed27-share 1 --closed27-weights simpson   ((A'), beta_S = 1)
#   H3  --logvol-points gauss27+closed27 --logvol-closed27-share 3 --closed27-weights simpson   ((A'), beta_S = 3)
# The structure predictions are read from --structure-root (default the registered segmenter's predictions
# $REAL_SEG/unet35_v1/pred; never generated here); the registry is the repository one of the code tree
# (configs/bayesreg/segmenters.json, "registered"); only a dry run passes --segmenter-registry (a synthetic staging
# registry, with --structure-registry-staging). G and H arms are never mixed in one start: the batch of the arms picks
# the default --root ($REAL_ROOT for G, $REAL_ROOT_V2 for H) and the default --min-gpu-free-mib (9216 for G; 22528
# for H: bayesreg.backends.torch64.estimate_peak_bytes at the native shape (192, 224, 160) with the v1 mask size
# 2098661 voxels, 14 channels, OPT1 defaults, 4 live evaluations: H1 17.1 GiB, H2/H3 19.9 GiB, + 2 GiB margin as
# run_pair's docstring asks). An H start reads the v1 records of --ref-root (default $REAL_ROOT; read only) for the
# table: the arms --ref-arms (default G2,G3) and, when $REF_ROOT/eval_A0/done.json exists, the A0 re-evaluation
# (--a0-eval, default that directory; otherwise A0 is re-evaluated into $ROOT/eval_A0 as before).
#
# Not a timing experiment: the GPU may be shared with the segmenter training (tag-unet35-train); wall clocks are recorded,
# never reported as speed. --wall-seconds is the development switch of contract C13-3 (appendix) / C20 D44 (预算收尾).
#
# Usage: gpu_smoke.sh [--root DIR] [--code DIR] [--commit SHA] [--arms G1,G2,G3|H1,H2,H3] [--pair STAGE/PAIR]
#                     [--data-root DIR] [--allowlist PATH] [--tissue-root DIR] [--structure-root DIR]
#                     [--segmenter-registry PATH (dry run only)] [--ref-root DIR] [--ref-arms G2,G3] [--a0-eval DIR]
#                     [--a0-run DIR] [--a0-metrics PATH] [--device cuda|cpu]
#                     [--wall-seconds S] [--extra "ARGS"] [--memory-max 10G] [--eval-memory-max 6G]
#                     [--min-gpu-free-mib N] [--home-floor-gb 100] [--wait-minutes 0] [--a0-only] [--dry-run]
#                     [--show-args]
# --pair: the development pair <stage>/<pair> (default the 0438<-0439 pair of stage_L2R_val_v1). The script does not
# check it against the allowlist itself; run_pair refuses a subject outside it (hard constraint 6 still applies to the
# lead's choice: development pairs only).
# --show-args: print the run_pair arguments of every arm and exit (nothing created, no check of the code tree).
#
# Code version (GS review item 9): --commit SHA (default: the HEAD at start) is resolved once and logged. Without --code the
# code tree is a detached git worktree $ROOT/code of that commit, made once ($ROOT/code_commit.txt) and verified on every
# later start (clean, at the recorded commit; an explicit --commit that differs -> STOP; a moved HEAD without --commit is
# logged and the recorded commit kept). With --code DIR the tree must have no tracked changes (and be at --commit when one
# is given). The SHA in use is written to smoke.log and to every launch record.
#
# Order. Start checks (any failure: STOP, nothing started): the tissue posteriors of both subjects, the A0 metrics and run
# directory exist; the code tree has run_pair --wall-seconds; --dry-run inputs are synthetic (below). Then:
#   A0 re-evaluation (GS review item 8; skipped in a dry run): the D0 A0 run directory (read only; nothing is written into
#   it) evaluated by bayesreg.eval.cli into $ROOT/eval_A0 (unit <prefix>-eval-A0), so that A0 and the arms share one
#   evaluation entry. A failed re-evaluation, or an earlier attempt without done.json, is kept and not rerun; the table
#   then uses the experiments/d0/evaluate.py numbers and says so (compare_gpu_smoke.py). A busy gate still STOPs.
#   Per arm, in order -- resume rule (hard constraint 4: nothing is rerun, no evidence overwritten):
#     eval_<arm>/done.json present -> skip; unit still active -> STOP; run directory with status.json -> evaluate only;
#     run directory without status.json -> STOP; a launch record launch/<arm>_*.txt (or launch/<arm>.txt) without
#     status.json -> STOP (launched before; kept whole, the lead decides); otherwise
#     gate (polled every 60 s for at most --wait-minutes, then STOP): hold file $ROOT/hold absent; tag-d0-queue not
#     active; no other seq-vp-* / tag-* unit active or activating (tag-unet35-train and the unit running this script
#     excepted: sharing the GPU with the training is planned); GPU free >= --min-gpu-free-mib (cuda only); MemAvailable >=
#     MemoryMax of the unit + 2 GiB; /home free >= --home-floor-gb. Before every unit start the gate's computation and
#     the training unit's MemoryCurrent / MemoryMax (systemctl --user show tag-unet35-train) go to smoke.log and the
#     launch record;
#     systemd-run --user --unit <prefix>-<arm> --wait --collect (MemoryMax --memory-max, MemorySwapMax 4G, CPUQuota,
#     RuntimeMaxSec = wall seconds + 1800 as a safety net) running `python -B -m bayesreg.cli.run_pair` in the code tree;
#     the run directory $ROOT/<arm> must not exist (run_pair creates it and writes cmd/git/env/data/config/provenance/
#     log/hyper/qc/timing/status and, on the native level, final_native.npz, cert.json and -- certificate PASS only --
#     disp.npz); launch record $ROOT/launch/<arm>_<UTC>.txt (new file per start, appended to: unit, commit, systemd
#     command, gate, GPU / memory / load before and after, exit code, OOM count); no status.json after the unit -> STOP;
#     evaluation: gate (no GPU condition; MemoryMax --eval-memory-max) and unit <prefix>-eval-<arm> (MemoryMax,
#     MemorySwapMax 4G, Nice 10, RuntimeMaxSec 7200) running `python -m bayesreg.eval.cli <run> --out-dir
#     $ROOT/eval_<arm>` (metrics.json under <out-dir>/<run_id>/, long.csv, summary.json); its output goes to
#     eval_<arm>/eval_<UTC>.log and its exit code to eval_<arm>/eval_exit_<UTC>.txt, the unit record to
#     launch/eval_<arm>_<UTC>.txt; an earlier evaluation attempt without done.json or an evaluation error -> STOP.
# Finally experiments/m2/compare_gpu_smoke.py writes $ROOT/comparison.{json,md} (once; never overwritten): A0 and the arms
# with their field_role; only native_formal rows sit next to A0, handover_only and the rest are listed apart. An H start
# also lists the --ref-arms of --ref-root and adds the witness-type counts (status.json witness_summary), the rejected
# candidates per level and the levels that ended blocked (受阻收尾: hyper.json rounds).
# Log: $ROOT/smoke.log (UTC, appended).
# --a0-only: the start checks and the A0 re-evaluation, nothing else (no arm, no comparison).
# --dry-run (GS review item 2): synthetic inputs only -- --data-root, --tissue-root and --allowlist must be given and
# none may lie under data/ (the repository link, its target or seq_vp_3d/data) or the real segmentation root; the pair's
# manifest must name experiments/m2/dry_inputs.py and the allowlist its synthetic_dry_run source. Unit prefix tag-m2dry,
# device cpu, one thread, Nice 19, no A0 re-evaluation; every gate is enforced (GPU excepted); --root must lie outside
# the real roots. H arms (SMK2): --structure-root and --segmenter-registry must be given, the same path checks apply,
# the fixed subject's structure provenance must name experiments/m2/dry_inputs.py, the registry is passed with
# --structure-registry-staging, and --ref-root (and --a0-eval, when given) must lie outside the real roots too.
set -u
REPO=/home/rmcfz/code/python_projects/seq_vp_tag
PY=/home/rmcfz/miniconda3/envs/sci310/bin/python
SMI=/usr/lib/wsl/lib/nvidia-smi
REAL_ROOT=/home/rmcfz/seq_vp_tag_results/m2_gpu_smoke
REAL_ROOT_V2=/home/rmcfz/seq_vp_tag_results/m2_gpu_smoke_v2
REAL_SEG=/home/rmcfz/seq_vp_tag_results/seg
ROOT=""
CODE=""
COMMIT=""
ARMS_TEXT=G1,G2,G3
PAIR=stage_L2R_val_v1/OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1
DATA_ROOT=""
ALLOWLIST=""
TISSUE_ROOT=""
STRUCTURE_ROOT=""
SEG_REGISTRY=""
REF_ROOT=""
REF_ARMS=G2,G3
A0_EVAL=""
SHOW=0
G_JD_EXPLICIT=1                      # DEF: G arms pass --jd-cell-estimate gauss27+corners (header)
DEFAULT_PAIR=$PAIR
D0_RUNS=/home/rmcfz/seq_vp_tag_results/d0_v1/runs
A0_RUN=""                            # default $D0_RUNS/A0/<pair> (the D0 configuration C run of the same pair)
A0=""
DEVICE=cuda
WALL=2700
EXTRA=""
MEMMAX=10G
EVAL_MEMMAX=6G
MIN_GPU_MIB=""                       # default by batch: G 9216, H 22528 (header)
HOME_FLOOR_GB=100
WAIT_MIN=0
DRY=0
A0_ONLY=0
POLL=60
THREADS=4
QUOTA=600%
MEM_MARGIN_MIB=2048                  # GS review item 1: MemAvailable >= MemoryMax + 2 GiB
TRAIN_UNIT=tag-unet35-train          # planned to share the GPU; never blocks the gate, always logged
EVAL_MAX_SEC=7200
while [ $# -gt 0 ]; do
  case "$1" in
    --root) ROOT="$2"; shift 2;; --code) CODE="$2"; shift 2;; --commit) COMMIT="$2"; shift 2;;
    --arms) ARMS_TEXT="$2"; shift 2;; --pair) PAIR="$2"; shift 2;; --data-root) DATA_ROOT="$2"; shift 2;;
    --allowlist) ALLOWLIST="$2"; shift 2;; --tissue-root) TISSUE_ROOT="$2"; shift 2;;
    --structure-root) STRUCTURE_ROOT="$2"; shift 2;; --segmenter-registry) SEG_REGISTRY="$2"; shift 2;;
    --ref-root) REF_ROOT="$2"; shift 2;; --ref-arms) REF_ARMS="$2"; shift 2;; --a0-eval) A0_EVAL="$2"; shift 2;;
    --show-args) SHOW=1; shift;;
    --a0-run) A0_RUN="$2"; shift 2;; --a0-metrics) A0="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;; --wall-seconds) WALL="$2"; shift 2;; --extra) EXTRA="$2"; shift 2;;
    --memory-max) MEMMAX="$2"; shift 2;; --eval-memory-max) EVAL_MEMMAX="$2"; shift 2;;
    --min-gpu-free-mib) MIN_GPU_MIB="$2"; shift 2;; --home-floor-gb) HOME_FLOOR_GB="$2"; shift 2;;
    --wait-minutes) WAIT_MIN="$2"; shift 2;; --a0-only) A0_ONLY=1; shift;; --dry-run) DRY=1; shift;;
    *) echo "unknown option $1"; exit 2;;
  esac
done

to_mib() {  # systemd size (bytes, or K / M / G / T with base 1024; integers only) -> MiB
  local v=$1 n u
  n=${v%[KkMmGgTt]}; u=${v#"$n"}
  case "$n" in ''|*[!0-9]*) return 1;; esac
  case "$u" in K|k) echo $((n/1024));; M|m) echo "$n";; G|g) echo $((n*1024));; T|t) echo $((n*1048576));;
               '') echo $((n/1048576));; *) return 1;; esac
}
MEMMAX_MIB=$(to_mib "$MEMMAX") && [ "$MEMMAX_MIB" -gt 0 ] || { echo "--memory-max $MEMMAX: an integer size such as 10G"; exit 2; }
EVAL_MEMMAX_MIB=$(to_mib "$EVAL_MEMMAX") && [ "$EVAL_MEMMAX_MIB" -gt 0 ] \
  || { echo "--eval-memory-max $EVAL_MEMMAX: an integer size such as 6G"; exit 2; }
STAGE=${PAIR%%/*}
PAIRNAME=${PAIR#*/}
SHORT=$(echo "$PAIRNAME" | sed -E 's/OASIS_OAS1_([0-9]+)_MR1__OASIS_OAS1_([0-9]+)_MR1/\1-\2/')
FIXED=${PAIRNAME%%__*}
MOVING=${PAIRNAME#*__}
[ -n "$A0_RUN" ] || A0_RUN=$D0_RUNS/A0/$PAIRNAME
[ -z "$A0" ] && A0=$A0_RUN/metrics.json

IFS=, read -ra ARMS <<< "$ARMS_TEXT"
BATCH=""
for arm in "${ARMS[@]}"; do
  case "$arm" in G1|G2|G3) b=G;; H1|H2|H3) b=H;; *) echo "unknown arm $arm (G1, G2, G3 | H1, H2, H3)"; exit 2;; esac
  [ -z "$BATCH" ] || [ "$BATCH" = "$b" ] \
    || { echo "--arms $ARMS_TEXT: G and H arms are never mixed in one start (separate roots)"; exit 2; }
  BATCH=$b
done
[ -n "$BATCH" ] || { echo "--arms: no arm given"; exit 2; }
REFS=()
if [ "$BATCH" = H ]; then
  # another pair gets its own default root (the arms' run directories carry no pair name)
  [ -n "$ROOT" ] || { [ "$PAIR" = "$DEFAULT_PAIR" ] && ROOT=$REAL_ROOT_V2 || ROOT=${REAL_ROOT_V2}_$SHORT; }
  [ -n "$MIN_GPU_MIB" ] || MIN_GPU_MIB=22528
  [ -n "$REF_ROOT" ] || REF_ROOT=$REAL_ROOT
  [ -z "$REF_ARMS" ] || IFS=, read -ra REFS <<< "$REF_ARMS"
  for arm in "${REFS[@]}"; do
    case "$arm" in G1|G2|G3) ;; *) echo "--ref-arms: unknown reference arm $arm (G1, G2, G3)"; exit 2;; esac
  done
else
  [ -n "$ROOT" ] || ROOT=$REAL_ROOT
  [ -n "$MIN_GPU_MIB" ] || MIN_GPU_MIB=9216
fi
case "$MIN_GPU_MIB" in *[!0-9]*) echo "--min-gpu-free-mib $MIN_GPU_MIB: an integer (MiB)"; exit 2;; esac
if [ -n "$SEG_REGISTRY" ] && { [ $DRY -eq 0 ] || [ "$BATCH" != H ]; }; then
  echo "--segmenter-registry: H arms of a dry run only (real runs use the registered repository registry of the code tree)"
  exit 2
fi

UNITP=tag-m2
if [ $DRY -eq 1 ]; then
  UNITP=tag-m2dry; DEVICE=cpu; THREADS=1; QUOTA=100%
  outside_real_roots() {  # outside_real_roots <option> <path>: exit 2 when the path lies under a real root
    local form       # $REAL_ROOT* covers v1, v2 and the per-pair v2 roots
    for form in "$(realpath -sm "$2")" "$(realpath -m "$2")"; do
      case "$form" in "$REAL_ROOT"*) echo "--dry-run: $1 must lie outside $REAL_ROOT* (the real roots)"; exit 2;; esac
    done
  }
  outside_real_roots --root "$ROOT"
  [ -n "$DATA_ROOT" ] && [ -n "$TISSUE_ROOT" ] && [ -n "$ALLOWLIST" ] \
    || { echo "--dry-run: --data-root, --tissue-root and --allowlist are required (experiments/m2/dry_inputs.py prints them)"; exit 2; }
  dry_paths=("$DATA_ROOT" "$TISSUE_ROOT" "$ALLOWLIST")
  if [ "$BATCH" = H ]; then
    [ -n "$STRUCTURE_ROOT" ] && [ -n "$SEG_REGISTRY" ] \
      || { echo "--dry-run: H arms need --structure-root and --segmenter-registry (experiments/m2/dry_inputs.py prints them)"; exit 2; }
    dry_paths+=("$STRUCTURE_ROOT" "$SEG_REGISTRY")
    outside_real_roots --ref-root "$REF_ROOT"
    [ -z "$A0_EVAL" ] || outside_real_roots --a0-eval "$A0_EVAL"
  fi
  tree=$(realpath -m "$(dirname "$0")/../..")          # the tree holding this script (its data/ link, if any)
  forbidden=("$REPO/data" "$(realpath -m "$REPO/data")" "$tree/data" "$(realpath -m "$tree/data")"
             /home/rmcfz/code/python_projects/seq_vp_3d/data "$REAL_SEG")
  for path in "${dry_paths[@]}"; do
    for form in "$(realpath -sm "$path")" "$(realpath -m "$path")"; do
      for bad in "${forbidden[@]}"; do
        case "$form/" in "$bad"/*) echo "--dry-run: $path lies under $bad (real data); synthetic inputs only"; exit 2;; esac
      done
    done
  done
  grep -q 'experiments/m2/dry_inputs.py' "$DATA_ROOT/$PAIR/manifest.json" 2>/dev/null \
    || { echo "--dry-run: $DATA_ROOT/$PAIR/manifest.json is not a dry_inputs.py pair"; exit 2; }
  grep -q '"synthetic_dry_run"' "$ALLOWLIST" 2>/dev/null \
    || { echo "--dry-run: $ALLOWLIST is not the dry_inputs.py allowlist"; exit 2; }
  if [ "$BATCH" = H ]; then
    grep -q 'experiments/m2/dry_inputs.py' "$STRUCTURE_ROOT/$FIXED/provenance.json" 2>/dev/null \
      || { echo "--dry-run: $STRUCTURE_ROOT/$FIXED/provenance.json is not a dry_inputs.py prediction"; exit 2; }
  fi
fi
[ -n "$DATA_ROOT" ] || DATA_ROOT=$REPO/data/three_d
[ -n "$TISSUE_ROOT" ] || TISSUE_ROOT=$REAL_SEG/atropos_v1
[ -n "$STRUCTURE_ROOT" ] || STRUCTURE_ROOT=$REAL_SEG/unet35_v1/pred

arm_args() {  # arm_args <arm> -> the run_pair arguments of the arm (one per line)
  local source=none points=gauss27 share=""
  case "$1" in
    G1) ;; G2) points=gauss27+corners;; G3) points=gauss27+corners; source=tissue;;
    H1) source=structure; points=gauss27+corners;;
    H2) source=structure; points=gauss27+closed27; share=1;;
    H3) source=structure; points=gauss27+closed27; share=3;;
    *) return 1;;
  esac
  printf '%s\n' --pair "$PAIR" --channel-source "$source" --logvol-points "$points"
  [ -n "$share" ] && printf '%s\n' --logvol-closed27-share "$share" --closed27-weights simpson
  printf '%s\n' --backend torch64 --device "$DEVICE" --metric-jd-scaling affine_v1
  if [ "$source" = structure ]; then
    printf '%s\n' --jd-cell-estimate certificate
  elif [ "$G_JD_EXPLICIT" -eq 1 ]; then
    printf '%s\n' --jd-cell-estimate gauss27+corners      # DEF: the G arms' estimate, explicit (header)
  fi
  printf '%s\n' --outer a --rounds 1,1,1,1 --wall-seconds "$WALL" --cpu-threads "$THREADS" \
    --data-root "$DATA_ROOT" --allowlist "${ALLOWLIST:-<code tree>/configs/bayesreg/dev_subjects.json}" \
    --tissue-root "$TISSUE_ROOT"
  if [ "$source" = structure ]; then
    printf '%s\n' --structure-root "$STRUCTURE_ROOT" --structure-infer never
    [ -n "$SEG_REGISTRY" ] && printf '%s\n' --segmenter-registry "$SEG_REGISTRY" --structure-registry-staging
  fi
  return 0
}

if [ $SHOW -eq 1 ]; then          # --show-args: nothing created, no code-tree check
  for arm in "${ARMS[@]}"; do
    mapfile -t args < <(arm_args "$arm")
    echo "$arm: python -B -m bayesreg.cli.run_pair --out $ROOT/$arm ${args[*]} $EXTRA"
  done
  echo "root $ROOT; batch $BATCH; min GPU free ${MIN_GPU_MIB} MiB; MemoryMax $MEMMAX; ref root ${REF_ROOT:-none} (${REFS[*]:-no reference arms})"
  exit 0
fi
mkdir -p "$ROOT" "$ROOT/launch" || { echo "cannot create $ROOT"; exit 1; }
LOG=$ROOT/smoke.log
utc() { date -u +%FT%TZ; }
stamp() { date -u +%Y%m%dT%H%M%SZ; }
log() { echo "$(utc) $*" | tee -a "$LOG"; }
stop() { log "STOP: $*"; exit 1; }
dmesg_since() { dmesg 2>/dev/null | grep -E "$2" | sed -E 's/^\[ *([0-9.]+)\].*/\1/' | awk -v t0="$1" '$1+0 >= t0+0 {c++} END {print c+0}'; }
gpustate() { $SMI --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader 2>&1 | head -1; }
homefree() { df --output=avail -B1G /home | tail -1 | tr -d ' '; }
memavail_mib() { awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo; }
train_state() {  # the training unit's state and memory (MemoryCurrent in bytes; [not set] when it is not running)
  local s; s=$(systemctl --user show "$TRAIN_UNIT" -p ActiveState -p MemoryCurrent -p MemoryMax 2>&1 | tr '\n' ' ')
  local cur; cur=$(systemctl --user show "$TRAIN_UNIT" -p MemoryCurrent --value 2>/dev/null)
  case "$cur" in ''|*[!0-9]*) echo "$TRAIN_UNIT: $s";; *) echo "$TRAIN_UNIT: $s(MemoryCurrent $((cur/1048576)) MiB)";; esac
}
# the unit running this script (if any) is not "another" unit
SELF_UNIT=$(sed -n 's|^0::.*/\([^/]*\.service\)$|\1|p' /proc/$$/cgroup 2>/dev/null | head -1)
other_units() {
  systemctl --user list-units --state=active,activating,deactivating --no-legend --plain 'seq-vp-*' 'tag-*' 2>/dev/null \
    | awk -v train="$TRAIN_UNIT.service" -v self="$SELF_UNIT" '$1 != train && $1 != self {printf "%s ", $1}' | sed 's/ $//'
}

# ---- start checks ---------------------------------------------------------------------------------------------------------
log "start: root $ROOT; arms ${ARMS[*]}; pair $PAIR; device $DEVICE; wall ${WALL}s; MemoryMax $MEMMAX (eval $EVAL_MEMMAX); dry-run $DRY; a0-only $A0_ONLY; repo HEAD $(git -C $REPO rev-parse --short HEAD 2>/dev/null); self unit ${SELF_UNIT:-none}"
for arm in "${ARMS[@]}"; do arm_args "$arm" > /dev/null || stop "unknown arm $arm (G1, G2, G3 | H1, H2, H3)"; done
[ "$STAGE" != "$PAIR" ] && [ -n "$PAIRNAME" ] || stop "--pair must be <stage>/<pair>"
for subject in "$FIXED" "$MOVING"; do
  [ -f "$TISSUE_ROOT/$subject/provenance.json" ] || stop "no tissue posteriors for $subject under $TISSUE_ROOT"
done
[ -f "$A0" ] || stop "A0 metrics $A0 missing"
[ $DRY -eq 1 ] || [ -d "$A0_RUN" ] || stop "A0 run directory $A0_RUN missing"
if [ "$BATCH" = H ]; then
  # structure layer (C03.1-2/3): predictions of the registered segmenter, never generated here (--structure-infer never)
  for subject in "$FIXED" "$MOVING"; do
    [ -f "$STRUCTURE_ROOT/$subject/provenance.json" ] && [ -f "$STRUCTURE_ROOT/$subject/structure_groups.npz" ] \
      || stop "no structure prediction (provenance.json, structure_groups.npz) for $subject under $STRUCTURE_ROOT"
  done
  [ -z "$SEG_REGISTRY" ] || [ -f "$SEG_REGISTRY" ] || stop "--segmenter-registry $SEG_REGISTRY missing"
  [ -d "$REF_ROOT" ] || log "reference root $REF_ROOT missing: the table lists the reference arms as missing"
  if [ -z "$A0_EVAL" ]; then      # the v1 re-evaluation of A0 (read only) when it is the same A0 run
    ref_run=$($PY -c "import json,sys;print(json.load(open(sys.argv[1])).get('run',''))" "$REF_ROOT/eval_A0/done.json" 2>/dev/null)
    if [ -n "$ref_run" ] && [ "$(realpath -m "$ref_run")" = "$(realpath -m "$A0_RUN")" ]; then
      A0_EVAL=$REF_ROOT/eval_A0
    fi
  fi
fi
[ -n "$A0_EVAL" ] || A0_EVAL=$ROOT/eval_A0
log "batch $BATCH; structure root $STRUCTURE_ROOT; segmenter registry ${SEG_REGISTRY:-configs/bayesreg/segmenters.json of the code tree}; reference ${REF_ROOT:-none} (${REFS[*]:-no arms}); A0 evaluation $A0_EVAL; min GPU free ${MIN_GPU_MIB} MiB"
explicit_commit=0
[ -n "$COMMIT" ] && explicit_commit=1
if [ -z "$CODE" ]; then
  [ -n "$COMMIT" ] || COMMIT=HEAD
  sha=$(git -C $REPO rev-parse --verify "$COMMIT^{commit}" 2>/dev/null) || stop "commit $COMMIT not found in $REPO"
  log "commit: --commit ${COMMIT} -> $sha (explicit $explicit_commit; resolved in $REPO at start)"
  CODE=$ROOT/code
  if [ -e "$CODE" ]; then
    want=$(cat "$ROOT/code_commit.txt" 2>/dev/null) || stop "$CODE exists without $ROOT/code_commit.txt"
    [ "$(git -C "$CODE" rev-parse HEAD 2>/dev/null)" = "$want" ] || stop "$CODE is not at the recorded commit $want"
    [ -z "$(git -C "$CODE" status --porcelain)" ] || stop "$CODE has changes (git status --porcelain not empty)"
    if [ "$sha" != "$want" ]; then
      [ $explicit_commit -eq 1 ] && stop "--commit $sha differs from the commit $want of the existing worktree $CODE"
      log "code: the repository HEAD is now $sha; the existing worktree keeps the recorded commit $want (resume)"
    fi
    log "code: reusing worktree $CODE at $want"
  else
    [ -e "$ROOT/code_commit.txt" ] && stop "$ROOT/code_commit.txt exists without $CODE (kept; the lead decides)"
    git -C $REPO worktree add --detach "$CODE" "$sha" >> "$LOG" 2>&1 || stop "git worktree add $CODE $sha failed"
    echo "$sha" > "$ROOT/code_commit.txt"
    log "code: new detached worktree $CODE at $sha (repo status: $(git -C $REPO status --porcelain | wc -l) changed/untracked paths, not used)"
  fi
else
  [ -d "$CODE" ] || stop "--code $CODE is not a directory"
  head_sha=$(git -C "$CODE" rev-parse HEAD 2>/dev/null) || stop "--code $CODE is not a git tree"
  if [ $explicit_commit -eq 1 ]; then
    sha=$(git -C "$CODE" rev-parse --verify "$COMMIT^{commit}" 2>/dev/null) || stop "commit $COMMIT not found in $CODE"
    [ "$sha" = "$head_sha" ] || stop "--code $CODE is at $head_sha, not at --commit $sha"
  fi
  [ -z "$(git -C "$CODE" status --porcelain --untracked-files=no)" ] || stop "--code $CODE has tracked changes"
  log "commit: --code $CODE at $head_sha (explicit --commit $explicit_commit; untracked paths: $(git -C "$CODE" status --porcelain | wc -l))"
fi
CODE_SHA=$(git -C "$CODE" rev-parse HEAD 2>/dev/null || echo n/a)
[ -n "$ALLOWLIST" ] || ALLOWLIST=$CODE/configs/bayesreg/dev_subjects.json
[ -f "$ALLOWLIST" ] || stop "allowlist $ALLOWLIST missing"
HELP=$(cd "$CODE" && CUDA_VISIBLE_DEVICES= $PY -B -m bayesreg.cli.run_pair --help 2>/dev/null)
grep -q -- --wall-seconds <<< "$HELP" || stop "run_pair in $CODE has no --wall-seconds (merge the GS tooling first)"
if [ "$BATCH" = G ] && ! grep -q -- --jd-cell-estimate <<< "$HELP"; then
  G_JD_EXPLICIT=0                    # before JDC: gauss27+corners was the only cell estimate (header, DEF)
  log "run_pair in $CODE has no --jd-cell-estimate (before JDC): the G arms run without it" \
      "(gauss27+corners, the only estimate there)"
fi
if [ "$BATCH" = H ]; then
  for option in --jd-cell-estimate --logvol-closed27-share --closed27-weights --structure-root gauss27+closed27; do
    grep -q -- "$option" <<< "$HELP" || stop "run_pair in $CODE has no $option (C05-2b / D47 tooling missing)"
  done
  [ -n "$SEG_REGISTRY" ] || [ -f "$CODE/configs/bayesreg/segmenters.json" ] \
    || stop "no registered segmenter registry $CODE/configs/bayesreg/segmenters.json"
fi
log "code $CODE: HEAD $CODE_SHA; data root $DATA_ROOT; tissue root $TISSUE_ROOT; allowlist $ALLOWLIST; A0 $A0 (run $A0_RUN)"
ENVTXT=$ROOT/env_$(stamp).txt          # exp-log procedure, env: CUDA, driver, GPU (run_pair's env.txt has the packages)
{ echo "recorded: $(utc)"; echo "code: $CODE at $CODE_SHA"; echo "python: $($PY -c 'import sys;print(sys.version.replace(chr(10)," "))')"
  echo "conda env: $(dirname "$(dirname "$PY")")"
  (cd "$CODE" && CUDA_VISIBLE_DEVICES= $PY -B -c "
import importlib
for m in ('numpy', 'scipy', 'numba', 'torch', 'nibabel'):
    print(m, importlib.import_module(m).__version__)
import torch
print('torch.version.cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version())" 2>&1)
  echo "nvidia-smi: $($SMI --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader 2>&1)"
  echo "cpu: $(lscpu | sed -n 's/^Model name:[ ]*//p'); nproc $(nproc)"; echo "kernel: $(uname -r)"
  echo "memory: $(free -g | awk '/Mem:/{print $2" GiB total, "$7" GiB available"}')"
  echo "seed: none (run_pair is deterministic; config.json seed)"; } >> "$ENVTXT"
log "env: $ENVTXT"

busy_reason() {  # busy_reason <MemoryMax MiB> <gpu 0|1>: why the next unit may not start (empty: may start)
  [ -e "$ROOT/hold" ] && { echo "hold file $ROOT/hold present"; return; }
  systemctl --user is-active --quiet tag-d0-queue && { echo "tag-d0-queue is active"; return; }
  local units; units=$(other_units)
  [ -n "$units" ] && { echo "units active: $units (only $TRAIN_UNIT may run alongside)"; return; }
  if [ "$2" = 1 ]; then
    local free; free=$($SMI --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    case "$free" in ''|*[!0-9]*) echo "nvidia-smi gave no free-memory reading ($free)"; return;; esac
    [ "$free" -lt "$MIN_GPU_MIB" ] && { echo "GPU free ${free} MiB < ${MIN_GPU_MIB} MiB"; return; }
  fi
  local avail need; avail=$(memavail_mib); need=$(( $1 + MEM_MARGIN_MIB ))
  [ "$avail" -lt "$need" ] && { echo "MemAvailable ${avail} MiB < MemoryMax $1 MiB + ${MEM_MARGIN_MIB} MiB = ${need} MiB ($(train_state))"; return; }
  local home; home=$(homefree)
  [ "$home" -lt "$HOME_FLOOR_GB" ] && { echo "/home free ${home} GB < ${HOME_FLOOR_GB} GB"; return; }
  return 0
}

gate_text() {  # gate_text <label> <MemoryMax text> <MemoryMax MiB>: the gate's computation at this moment (one line)
  local avail need units; avail=$(memavail_mib); need=$(( $3 + MEM_MARGIN_MIB )); units=$(other_units)
  echo "gate $1: MemAvailable ${avail} MiB >= MemoryMax $2 ($3 MiB) + ${MEM_MARGIN_MIB} MiB = ${need} MiB: $([ "$avail" -ge "$need" ] && echo yes || echo NO); $(train_state); /home free $(homefree) GB (floor $HOME_FLOOR_GB); gpu $(gpustate); other units: ${units:-none}"
}

gate() {  # gate <label> <MemoryMax MiB> <gpu 0|1>: 0 may start, 1 STOP
  local reason n=0 limit=$((WAIT_MIN))
  while :; do
    reason=$(busy_reason "$2" "$3")
    [ -z "$reason" ] && break
    [ $n -ge "$limit" ] && { log "STOP before $1: $reason (waited $n min, --wait-minutes $WAIT_MIN)"; return 1; }
    [ $((n % 30)) -eq 0 ] && log "waiting before $1: $reason"
    n=$((n+1)); sleep $POLL
  done
  return 0
}

evaluate() {  # evaluate <name> <run> <run id or ""> <method> -> 0 evaluated, 1 evaluation failed / earlier attempt, 2 gate STOP
  local name=$1 run=$2 rid=$3 method=$4 ev=$ROOT/eval_$1 unit=$UNITP-eval-$1 rc t0 s
  local earlier
  earlier=$(ls -d "$ev"/eval_*.log "$ev"/eval.log "$ev"/long.csv "$ROOT"/launch/eval_"$name"_*.txt 2>/dev/null | tr '\n' ' ')
  if [ -n "$earlier" ]; then
    log "evaluation of $name: earlier attempt without done.json ($earlier); kept whole, not rerun"
    return 1
  fi
  gate "eval_$name" "$EVAL_MEMMAX_MIB" 0 || return 2
  systemctl --user is-active --quiet "$unit" && { log "STOP: $unit is active"; return 2; }
  mkdir -p "$ev" "$ROOT/cache/numba"
  s=$(stamp)
  local elog=$ev/eval_$s.log erc=$ev/eval_exit_$s.txt rec=$ROOT/launch/eval_${name}_$s.txt
  local eval_cmd=($PY -B -m bayesreg.eval.cli "$run" --pairs-stage "$STAGE" --out metrics.json --long "$ev/long.csv"
                  --summary "$ev/summary.json" --out-dir "$ev" --method "$method" --arm "$name"
                  --data-root "$DATA_ROOT" --allowlist "$ALLOWLIST")
  [ -n "$rid" ] && eval_cmd+=(--run-id "$rid")
  local sd_cmd=(systemd-run --user --unit="$unit" --wait --collect --expand-environment=no --working-directory="$CODE"
                --nice=$([ $DRY -eq 1 ] && echo 19 || echo 10)
                --property=MemoryMax=$EVAL_MEMMAX --property=MemorySwapMax=4G --property=CPUQuota=$QUOTA
                --property=RuntimeMaxSec=$EVAL_MAX_SEC
                --setenv=CUDA_VISIBLE_DEVICES= --setenv=OPENBLAS_NUM_THREADS=$THREADS --setenv=OMP_NUM_THREADS=$THREADS
                --setenv=MKL_NUM_THREADS=$THREADS --setenv=NUMEXPR_NUM_THREADS=$THREADS --setenv=NUMBA_NUM_THREADS=$THREADS
                --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=NUMBA_CACHE_DIR=$ROOT/cache/numba
                /bin/bash -c '"${@:3}" >> "$1" 2>&1; rc=$?; echo $rc > "$2"; exit $rc' eval-wrapper "$elog" "$erc"
                "${eval_cmd[@]}")
  { echo "evaluation: $name"; echo "unit: $unit"; echo "run: $run (read only)"; echo "out: $ev"
    echo "code: $CODE (HEAD $CODE_SHA)"; echo "systemd command:"; printf '%q ' "${sd_cmd[@]}"; echo
    echo "$(gate_text "eval_$name" "$EVAL_MEMMAX" "$EVAL_MEMMAX_MIB")"
    echo "before $(utc): uptime $(cut -d' ' -f1 /proc/uptime); loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
  } >> "$rec"
  log "$(gate_text "eval_$name" "$EVAL_MEMMAX" "$EVAL_MEMMAX_MIB")"
  log "evaluate $name: unit $unit -> $ev (log $elog)"
  t0=$(cut -d' ' -f1 /proc/uptime)
  "${sd_cmd[@]}" >> "$rec" 2>&1; rc=$?
  [ -f "$erc" ] && rc=$(cat "$erc")
  { echo "after $(utc): exit $rc; uptime $(cut -d' ' -f1 /proc/uptime); MemAvailable $(memavail_mib) MiB; $(train_state)"
    echo "dmesg since start: oom $(dmesg_since "$t0" 'Out of memory|oom-kill|Killed process')"; } >> "$rec"
  [ "$rc" = 0 ] || { log "evaluation of $name exit $rc (see $elog, $rec; kept whole, nothing rerun)"; return 1; }
  $PY -c "import glob,json,sys,datetime;m=sorted(glob.glob(sys.argv[2]+'/*/metrics.json'));json.dump(dict(done_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),run=sys.argv[3],run_id=sys.argv[4] or None,metrics=m,log=sys.argv[5],code_sha=sys.argv[6]),open(sys.argv[1],'x'),indent=1)" \
    "$ev/done.json" "$ev" "$run" "$rid" "$elog" "$CODE_SHA" || { log "evaluation of $name: done.json not written"; return 1; }
  log "evaluated $name: $(tail -1 "$elog" | cut -c1-260)"
  return 0
}

one_arm() {  # one_arm <arm> -> 0 continue, 1 STOP
  local arm=$1 run=$ROOT/$1 unit=$UNITP-$1 rec rc t0 earlier
  if [ -f "$ROOT/eval_$arm/done.json" ]; then log "skip $arm (evaluated)"; return 0; fi
  systemctl --user is-active --quiet "$unit" && { log "STOP: $unit is still active"; return 1; }
  if [ -f "$run/status.json" ]; then
    log "resume: $run has status.json; evaluating"
    evaluate "$arm" "$run" "m2gpu_${arm}_$SHORT" bayesreg_m2_gpu_smoke || { log "STOP: evaluation of $arm"; return 1; }
    return 0
  fi
  [ -e "$run" ] && { log "STOP: $run exists without status.json (incomplete run kept whole; the lead decides)"; return 1; }
  earlier=$(ls "$ROOT"/launch/"$arm"_*.txt "$ROOT"/launch/"$arm".txt 2>/dev/null | tr '\n' ' ')
  [ -n "$earlier" ] && { log "STOP: $arm was launched before ($earlier) and has no status.json (kept whole; not restarted, the lead decides)"; return 1; }
  gate "$arm" "$MEMMAX_MIB" $([ "$DEVICE" = cuda ] && echo 1 || echo 0) || return 1
  rec=$ROOT/launch/${arm}_$(stamp).txt
  mapfile -t args < <(arm_args "$arm")
  # shellcheck disable=SC2206
  local extra=($EXTRA)
  local inner_env=(--setenv=OPENBLAS_NUM_THREADS=$THREADS --setenv=OMP_NUM_THREADS=$THREADS --setenv=MKL_NUM_THREADS=$THREADS
                   --setenv=NUMEXPR_NUM_THREADS=$THREADS --setenv=NUMBA_NUM_THREADS=$THREADS --setenv=PYTHONDONTWRITEBYTECODE=1
                   --setenv=NUMBA_CACHE_DIR=$ROOT/cache/numba)
  [ "$DEVICE" = cpu ] && inner_env+=(--setenv=CUDA_VISIBLE_DEVICES=)
  local sd_cmd=(systemd-run --user --unit="$unit" --wait --collect --expand-environment=no --working-directory="$CODE"
                --property=MemoryMax=$MEMMAX --property=MemorySwapMax=4G --property=CPUQuota=$QUOTA
                --property=RuntimeMaxSec=$(( ${WALL%.*} + 1800 )) "${inner_env[@]}")
  [ $DRY -eq 1 ] && sd_cmd+=(--nice=19)
  sd_cmd+=($PY -B -m bayesreg.cli.run_pair --out "$run" "${args[@]}" "${extra[@]}")
  mkdir -p "$ROOT/cache/numba"
  { echo "arm: $arm"; echo "unit: $unit"; echo "run: $run"; echo "code: $CODE (HEAD $CODE_SHA)"
    echo "systemd command:"; printf '%q ' "${sd_cmd[@]}"; echo
    echo "$(gate_text "$arm" "$MEMMAX" "$MEMMAX_MIB")"
    echo "before $(utc): uptime $(cut -d' ' -f1 /proc/uptime); loadavg $(cut -d' ' -f1-3 /proc/loadavg); MemAvailable $(memavail_mib) MiB; /home free $(homefree) GB; gpu $(gpustate)"
    echo "units active: $(systemctl --user list-units --state=active --no-legend --plain 'tag-*' 'seq-vp-*' 2>/dev/null | awk '{print $1}' | tr '\n' ' ')"
  } >> "$rec"
  log "$(gate_text "$arm" "$MEMMAX" "$MEMMAX_MIB")"
  t0=$(cut -d' ' -f1 /proc/uptime)
  log "run $arm: unit $unit -> $run (record $rec)"
  "${sd_cmd[@]}" >> "$rec" 2>&1; rc=$?
  { echo "after $(utc): systemd-run exit $rc; uptime $(cut -d' ' -f1 /proc/uptime); MemAvailable $(memavail_mib) MiB; gpu $(gpustate); $(train_state)"
    echo "dmesg since start: oom $(dmesg_since "$t0" 'Out of memory|oom-kill|Killed process'), dxg $(dmesg_since "$t0" 'create_allocation failed')"; } >> "$rec"
  [ -f "$run/status.json" ] || { log "STOP: $arm ended (systemd-run exit $rc) without $run/status.json (killed? see $rec)"; return 1; }
  log "ended $arm: systemd-run exit $rc; $($PY -c "import json,sys;s=json.load(open(sys.argv[1]));print(s.get('status'), s.get('wording'), 'exit', s.get('exit_code'))" "$run/status.json")"
  evaluate "$arm" "$run" "m2gpu_${arm}_$SHORT" bayesreg_m2_gpu_smoke || { log "STOP: evaluation of $arm"; return 1; }
  return 0
}

a0_eval() {  # the D0 A0 run through bayesreg.eval.cli (GS review item 8) -> 0 continue, 1 STOP
  if [ "$A0_EVAL" != "$ROOT/eval_A0" ]; then        # an H start reusing the v1 record, or an explicit --a0-eval
    log "A0 re-evaluation: $A0_EVAL (read only; not evaluated here; done.json $([ -f "$A0_EVAL/done.json" ] && echo present || echo MISSING -- the table then uses experiments/d0/evaluate.py))"
    return 0
  fi
  if [ $DRY -eq 1 ]; then log "dry-run: A0 re-evaluation skipped (real data); the table uses the D0 evaluate.py numbers"; return 0; fi
  if [ -f "$ROOT/eval_A0/done.json" ]; then log "A0 re-evaluation present ($ROOT/eval_A0/done.json)"; return 0; fi
  evaluate A0 "$A0_RUN" "" configC_frozen_eb808be_90s
  case $? in
    0) return 0;;
    1) log "A0 re-evaluation not available; the table uses experiments/d0/evaluate.py ($A0) and says so"; return 0;;
    *) return 1;;
  esac
}

a0_eval || { log "smoke stopped at the A0 re-evaluation"; exit 8; }
if [ $A0_ONLY -eq 1 ]; then log "a0-only: end"; exit 0; fi
for arm in "${ARMS[@]}"; do
  one_arm "$arm" || { log "smoke stopped at $arm"; exit 8; }
done
if [ -e "$ROOT/comparison.json" ] || [ -e "$ROOT/comparison.md" ]; then
  log "comparison exists ($ROOT/comparison.md); not rewritten"
else
  ref_opts=()
  [ "$BATCH" = H ] && [ ${#REFS[@]} -gt 0 ] && ref_opts=(--ref-root "$REF_ROOT" --ref-arms "$REF_ARMS")
  (cd "$CODE" && CUDA_VISIBLE_DEVICES= $PY -B experiments/m2/compare_gpu_smoke.py --root "$ROOT" --arms "$ARMS_TEXT" \
     "${ref_opts[@]}" --a0 "$A0" --a0-eval "$A0_EVAL" --out-json "$ROOT/comparison.json" --out-md "$ROOT/comparison.md") \
     >> "$LOG" 2>&1 || stop "comparison failed (see $LOG)"
  log "comparison: $ROOT/comparison.md"
fi
log "end"
