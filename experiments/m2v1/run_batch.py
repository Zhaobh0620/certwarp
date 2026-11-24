"""M2V1: the development batch after the C05-7 eta sweep (runs/20261031-m2v1-dev/plan.md). Development set only
(stage_L2R_val_v1); every number it yields is a development-set single observation.

    python -B experiments/m2v1/run_batch.py [--root DIR] [--code DIR] [--commit SHA] [--sweep-root DIR]
        [--after-sweep] [--sweep-wait-minutes 480] [--until precheck|eta|d47|dev] [--d47-last]
        [--no-start-after HH:MM|ISO]
        [--wall-seconds 900] [--extra "ARGS"] [--data-root DIR] [--allowlist PATH] [--tissue-root DIR]
        [--structure-root DIR] [--segmenter-registry PATH (dry run only)] [--memory-max 9G] [--eval-memory-max 6G]
        [--min-gpu-free-mib 22528] [--home-floor-gb 100] [--wait-minutes 0] [--max-consecutive-failures 3]
        [--accept-code-change] [--d0-long F] [--a4-long F] [--smoke-root DIR] [--ref-reeval DIR]
        [--d47-pair PAIR (dry run only)]
        [--dry-run]

Two arms, in this order (--d47-last: the dev batch first; the order is recorded in batch.json), each run in its own
systemd user unit (one at a time, `systemd-run --user --wait --collect`):
  d47  the D47 control: 1 run on OASIS_OAS1_0438_MR1 <- OASIS_OAS1_0439_MR1 (the smoke pair) with the method's final
       configuration and --jd-cell-estimate gauss27+corners (the only difference from the dev arm);
  dev  the development batch: the 19 Learn2Reg validation pairs of stage_L2R_val_v1/selection.json, once each, in that
       order, with the method's final configuration (--jd-cell-estimate certificate, D47).
Configuration (fixed, not options; the eta sweep's configuration apart from eta): --channel-source structure (registered
unet35_v1_ens, predictions under --structure-root read with --structure-infer never; the registry is the code tree's
configs/bayesreg/segmenters.json, tracked and clean), --logvol-points gauss27+closed27 --logvol-closed27-share 1
--closed27-weights simpson, --metric-jd-scaling affine_v1, --backend torch64 --device cuda (OPT1 switches at run_pair's
defaults), --outer a --rounds 1,1,1,1 --cpu-threads 4, --wall-seconds 900 (the development switch of C13-3 / D44: every
run ends 预算收尾 or another C13-3 ending, never "B 线收敛" unless its status says so). The machinery (code worktree,
registry check, gate, launch records, evaluation wrapper, write-once records) is experiments/eta/sweep.py's: its Sweep
class is subclassed (ArmSweep), one object per arm, so that model_args / configuration / expected_run_args are the
sweep's own code; the changes are listed at ArmSweep.

eta (from the sweep, never chosen here): <sweep-root>/eta_selected.json (alpha, nu, tau; sha256 recorded) and
<sweep-root>/summary.json (sha256 recorded). Checks (any failure: STOP): alpha, nu > 0 finite, tau K_T + 1 = 4 positive
values; summary.json eta_selected equals the file's eta; the sweep's stage set is stage_T0_tune_v1 and none of its pairs
or subjects is a pair or subject of this batch; the sweep's configuration equals each arm's configuration
(eta.Sweep.configuration()) key by key -- the structure root compared after resolving, the registry by sha256, folds and
status (its path is the sweep's own code worktree) -- apart from jd_cell_estimate on the d47 arm (sweep certificate, arm
gauss27+corners); the model code (bayesreg/, jd_curl/, configs/bayesreg/) of this batch's commit equals that of the
sweep's code_sha (`git diff --name-only`), unless --accept-code-change (recorded). The eta record eta/eta_m2v1.json
(alpha, nu, tau, the source, its sha256, the checks) is written once and is what every run reads with --eta-json; on a
restart it is recomputed and must agree.

--after-sweep: after the start checks, the code worktree and the precheck (so that a problem shows at launch, not
hours later), wait (polling every 60 s, at most --sweep-wait-minutes) for the sweep to end: eta_selected.json
present, no tag-eta-* unit active / activating / deactivating and no experiments/eta/sweep.py process -> go on; no
eta_selected.json, no such unit or process and the last line of <sweep-root>/sweep.log a STOP -> do nothing, exit 9;
the wait limit or the deadline reached first -> STOP (exit 8), nothing started.

Before any unit (precheck): the pairs (19 in a real run, the D47 pair among them; manifests and their subjects), the
registry (eta.Sweep.check_registry: real runs the code tree's registry, tracked and clean, >= 2 folds), the structure
predictions of the 20 subjects (<structure-root>/<subject>/{provenance.json, structure_groups.npz}; segmenter id,
subject, grouping, policy all_folds_mean, model ids registered; nothing is inferred here), the tissue layer
(<tissue-root>/<subject>/provenance.json with the QC flag), the subject pools (every subject on the development
allowlist, on no sealed list -- B-line hold-out, H1-H4 (bayesreg.eval.truth.load_sealed_lists) --, not a B-line
training subject (splits/bayesreg_v1/train_subjects.txt)). The run reads only fixed and moving (run_pair's loader);
labels are read by bayesreg.eval.cli only.

Per run (unit <prefix>-<arm>-<short>, MemoryMax --memory-max (9G), MemorySwapMax 4G, CPUQuota 600%, RuntimeMaxSec
wall + 1800) -> runs/<arm>/<short>/ (bayesreg.cli.run_pair); then its evaluation (unit <prefix>-eval-<arm>-<short>,
MemoryMax --eval-memory-max (6G)) by bayesreg.eval.cli only -> eval/<arm>/<short>/<run_id>/metrics.json (run id
m2v1_<arm>_<short>, method bayesreg_m2v1, arm dev / d47_gauss27_corners), long.csv, summary.json, done.json. After a run:
its config.json args must equal the arm's expected values (eta.Sweep.expected_run_args plus pair and eta_json) and its
config.json eta the eta record -- else STOP. An exported field whose evaluator certificate did not pass: STOP (hard
constraint 1; run_pair exports certified states only, so this would be a defect).
Gate before every unit (eta.Sweep.busy_reason; polled every 60 s for at most --wait-minutes, then STOP): hold file
$ROOT/hold absent; no other seq-vp-* / tag-* unit active, activating or deactivating (only the unit running this script
excepted; unlike the sweep, tag-unet35-train is not excepted); GPU free >= --min-gpu-free-mib (run units on cuda);
MemAvailable >= MemoryMax of the unit + 2 GiB; /home free >= --home-floor-gb.
Deadline: --no-start-after T (local time, HH:MM today or an ISO date-time; recorded in UTC): no run unit starts at or after
T (also checked while the gate waits); those pairs are logged as not run (未运行, not a failure) and the queue goes on to
the summary. Evaluations of runs that have ended are always done.
Resume (hard constraint 4: nothing rerun, nothing overwritten): eval done.json -> skip; eval_failed.json -> skip; unit
still active -> STOP; run directory with status.json -> evaluate only (if not yet); run directory without status.json,
or a launch record without run directory -> a failed run (recorded once under failures/, never relaunched); an earlier
evaluation attempt without done.json -> a failed evaluation (eval_failed.json, never rerun).
Failures (a run unit that left no status.json, a nonzero run exit code, a failed evaluation) are recorded, counted in the
summary and the queue goes on; --max-consecutive-failures (3) failures of newly launched runs in a row -> STOP.
End: experiments/m2v1/summarize.py writes summary.json / summary.md (written once; a different recomputation goes to
summary_<UTC>.*); its A0 / A4' numbers are the bayesreg.eval.cli re-evaluation in --ref-reeval (default
$ROOT/ref_reeval, experiments/m2v1/reeval_refs.py), the D0 long.csv numbers a reference column. batch.json (stage,
pairs, arms, configurations, expected args, eta record) is written once at the start.
The driver may run on a newer commit than the run code: the plan launches it from main with --commit 68721d1 (the eta
sweep's code); every unit runs run_pair / bayesreg.eval.cli in the worktree of that commit, and the eta check compares
the sweep's code_sha with that commit.
One driver per root: an exclusive flock on $ROOT/driver.lock (a second call on the same root exits 2 at once).
Code version: --commit SHA (default the repository HEAD at start, whose tracked files must be clean) in a detached git
worktree $ROOT/code (code_commit.txt; reused and verified on a restart; eta.Sweep.prepare_code). Log: $ROOT/queue.log (UTC).
--dry-run: synthetic inputs of experiments/m2v1/dry_inputs.py only (every input path outside data/ and the real result
roots), unit prefix tag-m2v1dry, device cpu, one thread, Nice 19; --segmenter-registry (staging) and --d47-pair are
dry-run options.
Exit codes: 0 done (or --until reached), 2 usage or another driver on the root, 8 STOP, 9 the sweep ended with STOP (nothing done).
"""
import argparse
from datetime import datetime, timezone
import fcntl
import glob
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bayesreg.eval.truth import SealedListError, load_sealed_lists  # noqa: E402
from bayesreg.io.guard import load_dev_allowlist  # noqa: E402
from bayesreg.model import LOGVOL_GAUSS27_CLOSED27  # noqa: E402
from bayesreg.seg.structure_layer import DEFAULT_GROUPING, GROUPS_FILE, PROVENANCE_FILE, SEGMENTER_ID  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO/rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eta = _load("_m2v1_eta_sweep", "experiments/eta/sweep.py")
summarize = _load("_m2v1_summarize", "experiments/m2v1/summarize.py")
Stop = eta.Stop
read_json, write_once, utc, stamp, short_name, subjects_of = (eta.read_json, eta.write_once, eta.utc, eta.stamp,
                                                             eta.short_name, eta.subjects_of)

PY = eta.PY
STAGE = "stage_L2R_val_v1"
SWEEP_STAGE = eta.STAGE                                     # stage_T0_tune_v1
D47_PAIR = "OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1"
N_PAIRS = 19
RESULTS = Path("/home/rmcfz/seq_vp_tag_results")
REAL_ROOT = RESULTS/"m2v1_dev"
SWEEP_ROOT = RESULTS/"eta_sweep_v3"
# never a --root (results of other work packages; read only)
PROTECTED = tuple(RESULTS/n for n in ("eta_sweep_v1", "eta_sweep_v2", "eta_sweep_v3", "d0_v1", "d0_v1_a4", "m2_gpu_smoke",
                                      "m2_gpu_smoke_v2", "seg", "cvg_v1", "mem_v1", "chain_20261031"))
ORDER = ("d47", "dev")
CELL = dict(d47="gauss27+corners", dev="certificate")
EVAL_ARM = dict(d47="d47_gauss27_corners", dev="dev")
SWEEP_CELL = "certificate"
METHOD = "bayesreg_m2v1"
SCHEMA = "bayesreg.m2v1.v1"
MODEL_PATHS = ("bayesreg", "jd_curl", "configs/bayesreg")
TRAIN_LIST = "splits/bayesreg_v1/train_subjects.txt"
DRY_SCRIPT = "experiments/m2v1/dry_inputs.py"
K_T_PLUS_1 = 4
EXIT_SWEEP_STOPPED = 9
POLL_SECONDS = eta.POLL_SECONDS


class Deadline(Exception):
    """--no-start-after reached before a run unit started (the pair is not run; not a failure)."""


SWEEP_SCRIPT = "experiments/eta/sweep.py"


def _argv(proc, pid):
    try:
        raw = (Path(proc)/str(pid)/"cmdline").read_bytes()
    except OSError:
        return []
    return [a.decode("utf-8", "replace") for a in raw.split(b"\0") if a]


def _ppid(proc, pid):
    """The parent PID from /proc/<pid>/stat (the fields after the last ')': state, ppid, ...); None if unreadable."""
    try:
        text = (Path(proc)/str(pid)/"stat").read_text(encoding="utf-8", errors="replace")
        return int(text[text.rindex(")")+1:].split()[1])
    except (OSError, ValueError, IndexError):
        return None


def sweep_processes(proc, self_pid):
    """PIDs of eta sweep drivers under the /proc tree `proc` (M2V1 review I2): argv[0] is a python interpreter (its base
    name starts with "python") and some argument ends with experiments/eta/sweep.py. This process and its ancestors are
    excluded (a shell that started the driver may carry the script name in its own command line), and so is any
    process whose argv[0] is not python (e.g. the `bash -c "... setsid nohup ... sweep.py ... & disown"` shell that
    launched the sweep and waits for it)."""
    ancestors, pid = set(), int(self_pid)
    while pid and pid not in ancestors:
        ancestors.add(pid)
        pid = _ppid(proc, pid)
    found = []
    for entry in sorted(Path(proc).glob("[0-9]*"), key=lambda p: int(p.name) if p.name.isdigit() else -1):
        if not entry.name.isdigit() or int(entry.name) in ancestors:
            continue
        argv = _argv(proc, entry.name)
        if (argv and Path(argv[0]).name.startswith("python")
                and any(a.endswith(SWEEP_SCRIPT) for a in argv[1:])):
            found.append(int(entry.name))
    return found


class QueueSystem(eta.System):
    """eta.System with: no unit excepted from the gate but the one running this script; the sweep's units and driver
    process (--after-sweep); the clock."""

    def other_units(self):
        out = self._out(["systemctl", "--user", "list-units", "--state=active,activating,deactivating", "--no-legend",
                         "--plain", "seq-vp-*", "tag-*"])
        units = [line.split()[0] for line in out.splitlines() if line.strip()]
        return [u for u in units if u != self.self_unit]

    def eta_units(self):
        out = self._out(["systemctl", "--user", "list-units", "--state=active,activating,deactivating", "--no-legend",
                         "--plain", "tag-eta-*"])
        return [line.split()[0] for line in out.splitlines() if line.strip()]

    def sweep_driver_running(self):
        """PIDs of running eta sweep drivers (sweep_processes on the real /proc)."""
        return sweep_processes(Path("/proc"), os.getpid())

    def now(self):
        return datetime.now(timezone.utc)


class ArmSweep(eta.Sweep):
    """One arm on eta.Sweep's machinery. Changes: the log is $ROOT/queue.log; the code-tree check asks run_pair and
    bayesreg.eval.cli; busy_reason words the unit rule for this batch (the rule itself is QueueSystem.other_units: no
    unit excepted but this script's); the gate also stops at the deadline (Deadline) while deadline_active; run_args
    name stage_L2R_val_v1; expected_args adds the per-run values. model_args, configuration, expected_run_args,
    prepare_code, write_env, check_registry, dry_checks, launch, env_args, earlier and compare_or_write are the
    sweep's."""

    def __init__(self, opts, system, arm):
        super().__init__(opts, system)
        self.arm = arm
        self.prefix = "tag-m2v1dry" if self.dry else "tag-m2v1"
        self.deadline = None
        self.deadline_active = False

    def log(self, text):
        line = f"{utc()} {text}"
        print(line, flush=True)
        with open(self.root/"queue.log", "a", encoding="utf-8") as stream:
            stream.write(line+"\n")

    def check_code_tree(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        need = {("-m", "bayesreg.cli.run_pair"): ("--eta-json", "--wall-seconds", "--logvol-points", "gauss27+closed27",
                                                  "--logvol-closed27-share", "--jd-cell-estimate", "certificate",
                                                  "gauss27+corners", "--structure-root", "--structure-infer"),
                ("-m", "bayesreg.eval.cli"): ("--pairs-stage", "--out-dir", "--run-id", "--method", "--arm")}
        for target, flags in need.items():
            out = subprocess.run([PY, "-B", *target, "--help"], cwd=self.code, env=env, capture_output=True, text=True,
                                 check=False).stdout
            missing = [f for f in flags if f not in out]
            if missing:
                raise Stop(f"{' '.join(target)} in {self.code} lacks {missing}")

    def busy_reason(self, memmax_mib, gpu):
        """eta.Sweep.busy_reason with this batch's wording: no unit may run alongside (QueueSystem.other_units excepts
        only the unit running this script)."""
        reason = super().busy_reason(memmax_mib, gpu)
        return reason.replace(f" (only {eta.TRAIN_UNIT} may run alongside)", " (no other tag-* / seq-vp-* unit may run "
                              "alongside)")

    def gate(self, label, memmax_mib, gpu):
        """eta.Sweep.gate (same polling and STOP) with the deadline checked before every poll of a run unit's gate."""
        waited = 0
        while True:
            if self.deadline_active and self.deadline is not None and self.sys.now() >= self.deadline:
                raise Deadline(f"before {label}: --no-start-after {self.deadline.isoformat()} reached (waited {waited} "
                               "min)")
            reason = self.busy_reason(memmax_mib, gpu)
            if not reason:
                return
            if waited >= self.o.wait_minutes:
                raise Stop(f"before {label}: {reason} (waited {waited} min, --wait-minutes {self.o.wait_minutes})")
            if waited % 30 == 0:
                self.log(f"waiting before {label}: {reason}")
            waited += 1
            self.sys.sleep(POLL_SECONDS)

    def run_args(self, pair, eta_file):
        args = ["--pair", f"{STAGE}/{pair}", *self.model_args(), "--structure-infer", "never", "--backend", self.o.backend]
        if self.o.backend == "torch64":
            args += ["--device", self.device]
        args += ["--outer", self.o.outer, "--rounds", self.o.rounds, "--wall-seconds", eta.number(self.o.wall_seconds),
                 "--cpu-threads", str(self.threads), "--data-root", str(self.data_root), "--allowlist", self.allowlist,
                 "--tissue-root", str(self.tissue_root), "--eta-json", str(eta_file)]
        return args+shlex.split(self.o.extra)

    def expected_args(self, pair, eta_file):
        """Every config.json args value this arm implies for one run (checked after the run; mismatch: STOP)."""
        expected = dict(self.expected_run_args(), structure_infer="never", cpu_threads=self.threads,
                        data_root=str(self.data_root), allowlist=str(self.allowlist),
                        tissue_root=str(self.tissue_root), eta_json=str(eta_file))
        if pair is not None:
            expected["pair"] = f"{STAGE}/{pair}"
        return expected


def arm_namespace(o, arm):
    """The eta.Sweep option namespace of one arm (the fixed configuration of the module docstring)."""
    return argparse.Namespace(
        root=o.root, code=o.code, commit=o.commit, until=None, channel_source="structure",
        logvol_points=LOGVOL_GAUSS27_CLOSED27, logvol_closed27_share=1., metric_jd_scaling="affine_v1",
        jd_cell_estimate=CELL[arm], structure_root=o.structure_root, structure_grouping=DEFAULT_GROUPING,
        segmenter_id=SEGMENTER_ID, segmenter_registry=o.segmenter_registry, backend="torch64", device="cuda", outer="a",
        rounds="1,1,1,1", wall_seconds=o.wall_seconds, extra=o.extra, data_root=o.data_root, allowlist=o.allowlist,
        tissue_root=o.tissue_root, memory_max=o.memory_max, eval_memory_max=o.eval_memory_max,
        probe_memory_max=o.memory_max, atropos_memory_max=o.eval_memory_max, atropos_threads=1,
        min_gpu_free_mib=o.min_gpu_free_mib, home_floor_gb=o.home_floor_gb, wait_minutes=o.wait_minutes,
        dry_run=o.dry_run)


def parse_deadline(text, now_local=None):
    """--no-start-after: 'HH:MM' (today, local time) or an ISO date-time (naive = local time) -> aware UTC datetime."""
    if text is None:
        return None
    now_local = now_local or datetime.now().astimezone()
    match = re.fullmatch(r"([0-9]{1,2}):([0-9]{2})", text)
    if match:
        moment = now_local.replace(hour=int(match.group(1)), minute=int(match.group(2)), second=0, microsecond=0)
    else:
        moment = datetime.fromisoformat(text)
        if moment.tzinfo is None:
            moment = moment.astimezone()
    return moment.astimezone(timezone.utc)


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def last_line(path):
    path = Path(path)
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    return lines[-1] if lines else ""


def config_differences(sweep, ours, *, d47):
    """Differences between the sweep's configuration (eta_selected.json) and one arm's (module docstring)."""
    out = []
    if not isinstance(sweep, dict):
        return [f"the sweep's configuration is {sweep!r}"]
    for key in sorted(set(sweep) | set(ours)):
        a, b = sweep.get(key), ours.get(key)
        if key == "jd_cell_estimate" and d47:
            if (a, b) != (SWEEP_CELL, CELL["d47"]):
                out.append(f"jd_cell_estimate: sweep {a!r}, arm {b!r} (expected {SWEEP_CELL!r} and {CELL['d47']!r})")
        elif key == "structure":
            if not (isinstance(a, dict) and isinstance(b, dict)):
                if a != b:
                    out.append(f"structure: sweep {a!r}, arm {b!r}")
                continue
            for sub in ("grouping", "segmenter_id", "registry_status", "staging_opt_in", "infer"):
                if a.get(sub) != b.get(sub):
                    out.append(f"structure.{sub}: sweep {a.get(sub)!r}, arm {b.get(sub)!r}")
            if a.get("root") is None or Path(a["root"]).resolve() != Path(b["root"]).resolve():
                out.append(f"structure.root: sweep {a.get('root')!r}, arm {b.get('root')!r}")
        elif key == "registry":
            a, b = a or {}, b or {}
            for sub in ("sha256", "folds", "status"):
                if a.get(sub) != b.get(sub):
                    out.append(f"registry.{sub}: sweep {a.get(sub)!r}, arm {b.get(sub)!r}")
        elif a != b:
            out.append(f"{key}: sweep {a!r}, arm {b!r}")
    return out


class Queue:
    def __init__(self, opts, system=None):
        self.o = opts
        self.sys = QueueSystem() if system is None else system
        self.root = Path(opts.root)
        self.dry = opts.dry_run
        self.arms = {arm: ArmSweep(arm_namespace(opts, arm), self.sys, arm) for arm in ORDER}
        self.order = tuple(reversed(ORDER)) if opts.d47_last else ORDER
        self.base = self.arms["dev"]
        self.deadline = parse_deadline(opts.no_start_after)
        for arm in self.arms.values():
            arm.deadline = self.deadline
        self.d47_pair = opts.d47_pair or D47_PAIR
        self.pairs = []
        self.eta_file = None
        self.eta = None
        self.eta_record = None
        self.in_row = 0

    def log(self, text):
        self.base.log(text)

    # ---- sweep -----------------------------------------------------------------------------------------------------
    def wait_for_sweep(self):
        sweep = Path(self.o.sweep_root)
        selected, waited = sweep/"eta_selected.json", 0
        while True:
            units, driver = self.sys.eta_units(), self.sys.sweep_driver_running()
            last = last_line(sweep/"sweep.log")
            if selected.is_file() and not units and not driver:
                self.log(f"after-sweep: {selected} present, no tag-eta-* unit, no sweep process (last log line: "
                         f"{last[:200]!r}); going on")
                return "selected"
            if not selected.is_file() and not units and not driver and " STOP: " in last:
                self.log(f"after-sweep: the sweep ended with STOP and no eta_selected.json ({last[:300]!r}); nothing "
                         "is done (exit 9)")
                return "stopped"
            if self.deadline is not None and self.sys.now() >= self.deadline:
                raise Stop(f"after-sweep: --no-start-after {self.deadline.isoformat()} reached while waiting for the "
                           "sweep; nothing started")
            if waited >= self.o.sweep_wait_minutes:
                raise Stop(f"after-sweep: the sweep has not ended after {waited} min (--sweep-wait-minutes "
                           f"{self.o.sweep_wait_minutes}); units {units}, sweep processes {driver}, eta_selected.json "
                           f"{selected.is_file()}; nothing started")
            if waited % 30 == 0:
                self.log(f"after-sweep: waiting (eta_selected.json {selected.is_file()}; tag-eta-* units "
                         f"{' '.join(units) or 'none'}; sweep processes {driver or 'none'}; last log line {last[:160]!r})")
            waited += 1
            self.sys.sleep(POLL_SECONDS)

    # ---- start ---------------------------------------------------------------------------------------------------------
    def root_checks(self):
        root = self.root.resolve()
        for bad in PROTECTED+((Path(self.o.sweep_root).resolve(),) if self.o.sweep_root else ()):
            if root == bad or bad in root.parents or root in bad.parents:
                return f"--root {root} overlaps {bad} (results of other work; read only)"
        if self.dry and (root == REAL_ROOT or REAL_ROOT in root.parents or RESULTS in root.parents):
            return f"--dry-run: --root must lie outside {RESULTS}"
        if self.dry:
            if self.o.ref_reeval is None:
                return "--dry-run: --ref-reeval is required (the synthetic re-evaluation of experiments/m2v1/dry_inputs.py)"
            for name in ("sweep_root", "d0_long", "a4_long", "smoke_root", "ref_reeval"):
                path = Path(getattr(self.o, name)).resolve()
                if path == RESULTS or RESULTS in path.parents:
                    return f"--dry-run: --{name.replace('_', '-')} {path} lies under {RESULTS} (synthetic inputs only)"
        return ""

    def start_checks(self):
        if self.o.segmenter_registry and not self.dry:
            raise Stop("--segmenter-registry: dry runs only (real runs use the code tree's registered registry)")
        if self.o.d47_pair and not self.dry:
            raise Stop("--d47-pair: dry runs only (the D47 pair is OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1)")
        if self.dry:
            self.base.dry_checks()
        stage_dir = self.base.data_root/STAGE
        selection = read_json(stage_dir/"selection.json")
        if selection is None:
            raise Stop(f"{stage_dir}/selection.json missing")
        self.pairs = [row["pair"] for row in selection.get("pairs") or []]
        if len(set(self.pairs)) != len(self.pairs):
            raise Stop(f"{stage_dir}/selection.json lists a pair twice")
        if not self.dry and len(self.pairs) != N_PAIRS:
            raise Stop(f"{stage_dir}/selection.json has {len(self.pairs)} pairs, expected {N_PAIRS}")
        if self.d47_pair not in self.pairs:
            raise Stop(f"the D47 pair {self.d47_pair} is not in {stage_dir}/selection.json")
        for pair in self.pairs:
            for name in ("pair.npz", "manifest.json"):
                if not (stage_dir/pair/name).is_file():
                    raise Stop(f"{stage_dir/pair/name} missing")
            manifest = read_json(stage_dir/pair/"manifest.json")
            if [manifest.get("fixed_subject"), manifest.get("moving_subject")] != list(subjects_of(pair)):
                raise Stop(f"{pair}: manifest subjects {manifest.get('fixed_subject')}, {manifest.get('moving_subject')}")
            if self.dry and DRY_SCRIPT not in json.dumps(manifest):
                raise Stop(f"--dry-run: {stage_dir/pair}/manifest.json is not a {DRY_SCRIPT} pair")
        if not self.dry and not self.o.code:
            dirty = self.base.git("status", "--porcelain", "--untracked-files=no").stdout.strip()
            if dirty:
                raise Stop(f"the repository {REPO} has tracked changes (git status --porcelain): {dirty[:300]}")
        self.log(f"pairs ({STAGE}/selection.json order): {len(self.pairs)}; D47 pair {self.d47_pair}")

    def share_code(self):
        for arm in self.arms.values():
            if arm is not self.base:
                arm.code, arm.code_sha, arm.allowlist = self.base.code, self.base.code_sha, self.base.allowlist

    # ---- precheck ------------------------------------------------------------------------------------------------------
    def precheck(self):
        registry = self.base.check_registry()
        for arm in self.arms.values():
            arm.registry_path, arm.registry_record = self.base.registry_path, self.base.registry_record
        subjects = sorted({s for p in self.pairs for s in subjects_of(p)})
        bad, missing = [], []
        for subject in subjects:
            directory = self.base.structure_root/subject
            if not ((directory/PROVENANCE_FILE).is_file() and (directory/GROUPS_FILE).is_file()):
                missing.append(subject)
                continue
            prov = read_json(directory/PROVENANCE_FILE) or {}
            want = dict(segmenter_id=SEGMENTER_ID, subject=subject, grouping=DEFAULT_GROUPING, policy=eta.INFER_POLICY)
            wrong = [f"{k} {prov.get(k)!r} (expected {v!r})" for k, v in want.items() if prov.get(k) != v]
            unknown = sorted(set(prov.get("model_ids") or []) - set(registry.models))
            if unknown or not prov.get("model_ids"):
                wrong.append(f"model_ids {prov.get('model_ids')} not all registered")
            if self.dry and DRY_SCRIPT not in json.dumps(prov):
                wrong.append(f"not a {DRY_SCRIPT} prediction (dry run)")
            if wrong:
                bad.append(f"{directory}: {'; '.join(wrong)}")
            if not eta.Sweep.tissue_ok(self.base.tissue_root/subject):
                bad.append(f"{self.base.tissue_root/subject}: no tissue provenance.json with the QC flag")
        if missing:
            raise Stop(f"precheck: no structure prediction for {missing} under {self.base.structure_root} (this tool never "
                       "infers; the lead decides)")
        if bad:
            raise Stop("precheck: " + " | ".join(bad))
        allow = load_dev_allowlist(self.base.allowlist)
        try:
            sealed = load_sealed_lists(REPO)
        except SealedListError as exc:
            raise Stop(f"precheck: sealed lists: {exc}") from None
        train = {line.strip() for line in (REPO/TRAIN_LIST).read_text(encoding="utf-8").splitlines() if line.strip()}
        pool = []
        for subject in subjects:
            if not allow.allows(subject):
                pool.append(f"{subject} not on the development allowlist {self.base.allowlist}")
            if sealed.which(subject) is not None:
                pool.append(f"{subject} on the sealed list {sealed.which(subject)}")
            if subject in train:
                pool.append(f"{subject} is a B-line training subject ({TRAIN_LIST})")
        if pool:
            raise Stop("precheck (hard constraint 6): " + "; ".join(pool))
        self.pool_record = dict(subjects=subjects, allowlist=allow.record(), sealed_lists=sealed.record,
                                train_list=dict(path=TRAIN_LIST, sha256=sha256_of(REPO/TRAIN_LIST)))
        self.log(f"precheck: {len(subjects)} subjects: structure predictions ({eta.INFER_POLICY}, {SEGMENTER_ID}) and "
                 "tissue layer present; on the development allowlist, on no sealed list (B-line hold-out, H1-H4), no "
                 "training subject")

    # ---- eta ---------------------------------------------------------------------------------------------------------
    def code_difference(self, sweep_sha):
        if sweep_sha == self.base.code_sha:
            return []
        out = self.base.git("diff", "--name-only", sweep_sha, self.base.code_sha, "--", *MODEL_PATHS)
        if out.returncode:
            raise Stop(f"cannot compare the model code of {sweep_sha} and {self.base.code_sha}: {out.stderr.strip()}")
        return [line for line in out.stdout.splitlines() if line.strip()]

    def eta_phase(self):
        sweep = Path(self.o.sweep_root)
        path, spath = sweep/"eta_selected.json", sweep/"summary.json"
        if not path.is_file():
            raise Stop(f"{path} missing (the sweep has not selected an eta)")
        if not spath.is_file():
            raise Stop(f"{spath} missing")
        doc, summary = json.loads(path.read_text(encoding="utf-8")), json.loads(spath.read_text(encoding="utf-8"))
        value = dict(alpha=doc.get("alpha"), nu=doc.get("nu"), tau=doc.get("tau"))
        good = (eta.finite(value["alpha"]) and value["alpha"] > 0 and eta.finite(value["nu"]) and value["nu"] > 0
                and isinstance(value["tau"], list) and len(value["tau"]) == K_T_PLUS_1
                and all(eta.finite(t) and t > 0 for t in value["tau"]))
        if not good:
            raise Stop(f"{path}: eta {value} (alpha, nu > 0 finite; tau {K_T_PLUS_1} positive values)")
        if summary.get("eta_selected") != value:
            raise Stop(f"{spath} eta_selected {summary.get('eta_selected')} differs from {path} {value}")
        if summary.get("stage_set") != SWEEP_STAGE:
            raise Stop(f"{spath}: stage_set {summary.get('stage_set')!r}, expected {SWEEP_STAGE}")
        sweep_pairs = list(summary.get("pairs") or [])
        sweep_subjects = {s for p in sweep_pairs for s in subjects_of(p)}
        ours = {s for p in self.pairs for s in subjects_of(p)}
        if not sweep_pairs or set(sweep_pairs) & set(self.pairs) or sweep_subjects & ours:
            raise Stop(f"the sweep's pairs {sweep_pairs} overlap this batch (pairs {sorted(set(sweep_pairs) & set(self.pairs))}, "
                       f"subjects {sorted(sweep_subjects & ours)}) or are missing")
        checks = {}
        for name, arm in self.arms.items():
            differ = config_differences(doc.get("configuration"), arm.configuration(), d47=name == "d47")
            if differ:
                raise Stop(f"arm {name}: configuration differs from the sweep's ({path}): {'; '.join(differ)}")
            checks[name] = ("equal to the sweep's" if name == "dev" else
                            f"equal to the sweep's apart from jd_cell_estimate ({SWEEP_CELL} -> {CELL['d47']})")
        sweep_sha = doc.get("code_sha")
        changed = self.code_difference(sweep_sha)
        if changed and not self.o.accept_code_change:
            raise Stop(f"the model code differs between the sweep ({sweep_sha}) and this batch ({self.base.code_sha}): "
                       f"{changed} (--accept-code-change records and accepts it; the lead decides)")
        record = dict(schema=SCHEMA+".eta", alpha=value["alpha"], nu=value["nu"], tau=value["tau"],
                      use=("η 由 C05-7 扫描在 T0 调参对上选出（EB 初值）；本批照用，不在 L2R 验证对上调"),
                      source=str(path), source_sha256=sha256_of(path), summary=str(spath), summary_sha256=sha256_of(spath),
                      sweep_stage_set=SWEEP_STAGE, sweep_pairs=sweep_pairs, sweep_code_sha=sweep_sha,
                      code_sha=self.base.code_sha, model_code_changed=changed,
                      accept_code_change=bool(changed and self.o.accept_code_change),
                      selection_path=doc.get("selection_path"), nu_star=doc.get("nu_star"),
                      sweep_configuration=doc.get("configuration"), configuration_check=checks)
        (self.root/"eta").mkdir(exist_ok=True)
        self.eta_file = self.root/"eta"/"eta_m2v1.json"
        self.base.compare_or_write(self.eta_file, record)
        self.eta, self.eta_record = value, record
        self.log(f"eta: alpha {value['alpha']:.6g}, nu {value['nu']:.6g}, tau {value['tau']} from {path} (sha256 "
                 f"{record['source_sha256'][:12]}; sweep code {sweep_sha}; model code changed {changed or 'no'}); "
                 f"configuration: {checks}; record {self.eta_file}")

    def arm_pairs(self, arm):
        return [self.d47_pair] if arm == "d47" else list(self.pairs)

    def write_batch(self):
        arms = {name: dict(pairs=self.arm_pairs(name), configuration=arm.configuration(),
                           expected_args=arm.expected_args(None, self.eta_file), eval_arm=EVAL_ARM[name],
                           unit_prefix=f"{arm.prefix}-{name}-", run_id=f"m2v1_{name}_<short>")
                for name, arm in self.arms.items()}
        document = dict(schema=SCHEMA+".batch", stage=STAGE, pairs=self.pairs, d47_pair=self.d47_pair, order=list(self.order),
                        arms=arms, method=METHOD, wall_seconds=self.o.wall_seconds, code_sha=self.base.code_sha,
                        eta=self.eta, eta_file=str(self.eta_file),
                        eta_source={k: self.eta_record[k] for k in ("source", "source_sha256", "summary",
                                                                    "summary_sha256", "sweep_code_sha", "sweep_pairs")},
                        pools=self.pool_record, dry_run=self.dry)
        self.base.compare_or_write(self.root/"batch.json", document)
        self.log(f"batch: {self.root/'batch.json'}; deadline "
                 + (f"{self.deadline.isoformat()} (no run unit starts at or after it)" if self.deadline else "none"))

    # ---- runs ----------------------------------------------------------------------------------------------------------
    def record_failure(self, name, document):
        (self.root/"failures").mkdir(exist_ok=True)
        path = self.root/"failures"/f"run_{name}.json"
        if not path.exists():
            write_once(path, dict(document, recorded_utc=utc()))
        return path

    def check_run(self, arm, pair, run, name):
        config = read_json(run/"config.json") or {}
        args = config.get("args")
        if not isinstance(args, dict):
            self.log(f"{name}: no config.json args (the run ended before writing them)")
            return
        expected = arm.expected_args(pair, self.eta_file)
        differ = {k: (args.get(k), v) for k, v in expected.items() if args.get(k) != v}
        if differ:
            raise Stop(f"{name}: run_pair config.json args differ from the plan (got, expected): {differ}")
        got = config.get("eta") or {}
        if summarize.eta_check(got, self.eta) is not True:
            raise Stop(f"{name}: config.json eta {got} differs from the eta record {self.eta}")

    def one(self, arm_name, pair):
        """'skipped' (done before), 'ok' or 'failed' (newly ended); Deadline when the run would start too late."""
        arm, short = self.arms[arm_name], short_name(pair)
        name = f"{arm_name}-{short}"
        run, ev = self.root/"runs"/arm_name/short, self.root/"eval"/arm_name/short
        unit = f"{arm.prefix}-{name}"
        if (ev/"done.json").is_file():
            self.hard_constraint_1(name, ev)                  # re-checked on every call (M2V1 review minor 1)
            self.log(f"skip {name} (evaluated)")
            return "skipped"
        if (ev/"eval_failed.json").is_file():
            self.log(f"skip {name} (evaluation failed earlier; kept, not rerun)")
            return "skipped"
        if self.sys.unit_active(unit):
            raise Stop(f"{unit} is still active")
        if (run/"status.json").is_file():
            self.log(f"resume: {run} has status.json; evaluating")
            self.check_run(arm, pair, run, name)
            return "failed" if self.evaluate(arm, name, short, run, ev) == "failed" else "skipped"
        earlier = arm.earlier("run", name)
        if run.exists() or earlier:
            path = self.record_failure(name, dict(arm=arm_name, pair=pair, run_dir=str(run), launch_records=earlier,
                                                  reason="run directory without status.json, or a launch record "
                                                  "without run directory (killed?); kept, never relaunched"))
            self.log(f"skip {name}: launched before without status.json (failure {path}; not rerun)")
            return "skipped"
        if self.deadline is not None and self.sys.now() >= self.deadline:
            raise Deadline(f"{name}: --no-start-after {self.deadline.isoformat()} reached")
        run.parent.mkdir(parents=True, exist_ok=True)
        cuda = arm.o.backend == "torch64" and arm.device == "cuda"
        command = [PY, "-B", "-m", "bayesreg.cli.run_pair", "--out", run, *arm.run_args(pair, self.eta_file)]
        arm.deadline_active = True
        try:
            rc, record = arm.launch(kind="run", name=name, unit=unit, command=command, memmax_text=self.o.memory_max,
                                    memmax_mib=arm.memmax_mib, gpu=cuda, nice=19 if self.dry else None,
                                    runtime=int(self.o.wall_seconds)+1800, quota=arm.quota)
        finally:
            arm.deadline_active = False
        status = read_json(run/"status.json")
        if status is None:
            path = self.record_failure(name, dict(arm=arm_name, pair=pair, run_dir=str(run), systemd_exit=rc,
                                                  launch_record=str(record), reason="the unit ended without "
                                                  "status.json (killed? see the launch record); kept, not rerun"))
            self.log(f"FAILED {name}: systemd-run exit {rc}, no status.json (failure {path}; not rerun)")
            return "failed"
        self.log(f"ended {name}: systemd-run exit {rc}; {status.get('status')} {status.get('wording')} exit "
                 f"{status.get('exit_code')}")
        self.check_run(arm, pair, run, name)
        outcome = self.evaluate(arm, name, short, run, ev)
        return "failed" if status.get("exit_code") != 0 or outcome == "failed" else "ok"

    def hard_constraint_1(self, name, ev):
        """The evaluator's record of one evaluated run (eval/<arm>/<short>/<run_id>/metrics.json): a field that was
        evaluated (field_present) whose certificate did not pass -> STOP (hard constraint 1; run_pair exports certified
        states only, so this would be a defect). Checked after the evaluation and again whenever a call meets its
        done.json. Returns the record."""
        found = sorted(glob.glob(str(ev/"*"/"metrics.json")))
        if len(found) != 1:
            raise Stop(f"{name}: {len(found)} metrics.json under {ev} (one expected); kept, the lead decides")
        doc = read_json(found[0]) or {}
        cert = doc.get("certificate") or {}
        if doc.get("field_present") and cert.get("passed") is not True:
            raise Stop(f"{name}: the evaluator's certificate of the exported field did not pass ({cert.get('status')}, "
                       f"{cert.get('min_jd_lower_bound')}): hard constraint 1; kept, the lead decides")
        return doc

    def evaluate(self, arm, name, short, run, ev):
        """eta.Sweep.evaluate with this batch's stage, run id, method and arm; a failed evaluation is recorded
        (eval_failed.json) and the queue goes on; an earlier attempt without done.json is not rerun."""
        earlier = sorted(glob.glob(str(ev/"eval_*.log")))+sorted(glob.glob(str(ev/"long.csv")))+arm.earlier("eval", name)
        if earlier:
            if not (ev/"eval_failed.json").exists():
                write_once(ev/"eval_failed.json", dict(reason="earlier evaluation attempt without done.json; kept, not "
                                                        "rerun", files=earlier, recorded_utc=utc()))
            self.log(f"evaluation of {name}: earlier attempt without done.json ({' '.join(earlier)}); recorded as failed")
            return "failed"
        ev.mkdir(parents=True, exist_ok=True)
        s = stamp()
        log, exit_file = ev/f"eval_{s}.log", ev/f"eval_exit_{s}.txt"
        rid = f"m2v1_{arm.arm}_{short}"
        command = [PY, "-B", "-m", "bayesreg.eval.cli", run, "--pairs-stage", STAGE, "--out", "metrics.json", "--long",
                   ev/"long.csv", "--summary", ev/"summary.json", "--out-dir", ev, "--method", METHOD, "--arm",
                   EVAL_ARM[arm.arm], "--run-id", rid, "--data-root", arm.data_root, "--allowlist", arm.allowlist]
        rc, record = arm.launch(kind="eval", name=name, unit=f"{arm.prefix}-eval-{name}", command=command,
                                memmax_text=self.o.eval_memory_max, memmax_mib=arm.eval_mib, gpu=False,
                                nice=19 if self.dry else 10, runtime=eta.EVAL_MAX_SEC, quota=arm.quota,
                                wrapped=(log, exit_file))
        metrics = sorted(glob.glob(str(ev/"*"/"metrics.json")))
        if rc != 0 or len(metrics) != 1:
            write_once(ev/"eval_failed.json", dict(exit=rc, log=str(log), launch_record=str(record), metrics=metrics,
                                                    recorded_utc=utc()))
            self.log(f"FAILED evaluation of {name}: exit {rc}, {len(metrics)} metrics.json (see {log}; kept, not rerun)")
            return "failed"
        write_once(ev/"done.json", dict(done_utc=utc(), run=str(run), run_id=rid, metrics=metrics, log=str(log),
                                        code_sha=arm.code_sha))
        doc = self.hard_constraint_1(name, ev)
        cert = doc.get("certificate") or {}
        m = doc.get("metrics") or {}
        self.log(f"evaluated {name}: field_role {doc.get('field_role')}; seg35 {m.get('seg35')}, seg4 {m.get('seg4')}; "
                 f"certificate {cert.get('status')} {cert.get('min_jd_lower_bound')}")
        return "ok"

    def batch(self):
        not_run = []
        for arm in self.order:
            for pair in self.arm_pairs(arm):
                try:
                    outcome = self.one(arm, pair)
                except Deadline as exc:
                    not_run.append(f"{arm}-{short_name(pair)}")
                    self.log(f"not run: {exc} (未运行, not a failure)")
                    continue
                if outcome == "failed":
                    self.in_row += 1
                    if self.in_row >= self.o.max_consecutive_failures:
                        raise Stop(f"{self.in_row} failed runs in a row (--max-consecutive-failures "
                                   f"{self.o.max_consecutive_failures}); kept, the lead decides")
                elif outcome == "ok":
                    self.in_row = 0
            if self.o.until == arm and arm != self.order[-1]:
                self.log(f"until {arm}: end (not run in this call: the later arms)")
                return not_run, False
        return not_run, True

    # ---- main ----------------------------------------------------------------------------------------------------------
    def lock(self):
        """One driver per root: an exclusive, non-blocking flock on $ROOT/driver.lock held until main() returns (the
        kernel drops it if the process dies). None when another driver holds it."""
        stream = open(self.root/"driver.lock", "a+", encoding="utf-8")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            return None
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid {os.getpid()} since {utc()}\n")
        stream.flush()
        return stream

    def main(self):
        problem = self.root_checks()
        if problem:
            print(problem, file=sys.stderr)
            return 2
        self.root.mkdir(parents=True, exist_ok=True)
        lock = self.lock()
        if lock is None:
            print(f"{self.root/'driver.lock'} is held by another run_batch.py on this root; nothing done", file=sys.stderr)
            return 2
        try:
            return self.run()
        finally:
            lock.close()

    def run(self):
        self.log(f"start: root {self.root}; until {self.o.until}; sweep root {self.o.sweep_root} (after-sweep "
                 f"{int(self.o.after_sweep)}); arms in order {list(self.order)} (jd_cell_estimate {CELL}); wall "
                 f"{self.o.wall_seconds:g} s (预算收尾); extra {self.o.extra!r}; MemoryMax {self.o.memory_max} (eval "
                 f"{self.o.eval_memory_max}); min GPU free {self.o.min_gpu_free_mib} MiB; deadline "
                 f"{self.deadline.isoformat() if self.deadline else 'none'} (--no-start-after {self.o.no_start_after}); "
                 f"dry-run {int(self.dry)}; self unit {getattr(self.sys, 'self_unit', '') or 'none'}")
        try:
            self.start_checks()
            self.base.prepare_code()
            self.share_code()
            self.base.write_env()
            self.precheck()
            if self.o.until == "precheck":
                self.log("until precheck: end")
                return 0
            if self.o.after_sweep and self.wait_for_sweep() == "stopped":
                return EXIT_SWEEP_STOPPED
            self.eta_phase()
            self.write_batch()
            if self.o.until == "eta":
                self.log("until eta: end")
                return 0
            not_run, complete = self.batch()
            if not_run:
                self.log(f"not run (deadline; 未运行, not failures): {', '.join(not_run)}")
            if complete:
                try:
                    js, md, state = summarize.write(self.root, d0_long=self.o.d0_long, a4_long=self.o.a4_long,
                                                    smoke_root=self.o.smoke_root, ref_reeval=self.o.ref_reeval)
                except summarize.SummaryError as exc:
                    raise Stop(f"summary: {exc}") from None
                self.log(f"summary: {js}, {md} ({state})")
            self.log("end")
            return 0
        except Stop as exc:
            self.log(f"STOP: {exc}")
            return 8


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -B experiments/m2v1/run_batch.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default=str(REAL_ROOT))
    parser.add_argument("--code")
    parser.add_argument("--commit")
    parser.add_argument("--sweep-root", default=str(SWEEP_ROOT))
    parser.add_argument("--after-sweep", action="store_true")
    parser.add_argument("--sweep-wait-minutes", type=int, default=480)
    parser.add_argument("--until", choices=("precheck", "eta", *ORDER), default=None,
                        help="stop after this step or arm (default: both arms and the summary)")
    parser.add_argument("--d47-last", action="store_true", help="run the dev batch first and the D47 arm after it "
                        "(default: the D47 arm first, plan.md); recorded in batch.json (order)")
    parser.add_argument("--no-start-after", help="local time HH:MM (today) or an ISO date-time; no run unit starts "
                        "at or after it")
    parser.add_argument("--wall-seconds", type=float, default=900.)
    parser.add_argument("--extra", default="")
    parser.add_argument("--data-root")
    parser.add_argument("--allowlist")
    parser.add_argument("--tissue-root")
    parser.add_argument("--structure-root", default=str(eta.DEFAULT_STRUCTURE_ROOT))
    parser.add_argument("--segmenter-registry", default=None)
    parser.add_argument("--memory-max", default="9G")
    parser.add_argument("--eval-memory-max", default="6G")
    parser.add_argument("--min-gpu-free-mib", type=int, default=22528)
    parser.add_argument("--home-floor-gb", type=int, default=100)
    parser.add_argument("--wait-minutes", type=int, default=0)
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--accept-code-change", action="store_true")
    parser.add_argument("--d0-long", default=str(summarize.D0_LONG))
    parser.add_argument("--a4-long", default=str(summarize.A4_LONG))
    parser.add_argument("--smoke-root", default=str(summarize.SMOKE_ROOT))
    parser.add_argument("--ref-reeval", default=None, help="the bayesreg.eval.cli re-evaluation of A0 / A4' "
                        "(experiments/m2v1/reeval_refs.py; default <root>/ref_reeval)")
    parser.add_argument("--d47-pair", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("memory_max", "eval_memory_max"):
        try:
            if eta.to_mib(getattr(args, name)) <= 0:
                raise ValueError
        except ValueError:
            parser.error(f"--{name.replace('_', '-')} {getattr(args, name)}: an integer size such as 9G")
    if not (args.wall_seconds > 0 and math.isfinite(args.wall_seconds)):
        parser.error("--wall-seconds: a positive number")
    if args.max_consecutive_failures < 1:
        parser.error("--max-consecutive-failures: at least 1")
    try:
        extra = shlex.split(args.extra)
        parse_deadline(args.no_start_after)
    except ValueError as exc:
        parser.error(str(exc))
    flags = {a.split("=", 1)[0] for a in extra if a.startswith("--")}
    repeated = sorted(f for f in flags if any(m.startswith(f) for m in eta.MANAGED))
    if repeated:
        parser.error(f"--extra must not repeat options the queue sets itself: {repeated}")
    return args


def main(argv=None, system=None):
    return Queue(parse_args(argv), system).main()


if __name__ == "__main__":
    sys.exit(main())
