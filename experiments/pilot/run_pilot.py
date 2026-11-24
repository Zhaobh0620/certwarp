"""PILOT: the path-improvement pilot (runs/20261031-path-pilot/plan.md). Development set only (stage_L2R_val_v1, the
first five pairs of selection.json); every number it yields is a development-set single observation (exploratory).

    python -B experiments/pilot/run_pilot.py [--root DIR] [--code DIR] [--commit SHA] [--m2v1-root DIR]
        [--sweep-root DIR] [--arms p0n,p1,p2,p3,p4] [--until precheck|eta] [--after-m2v1] [--m2v1-wait-minutes 600]
        [--no-start-after HH:MM|ISO] [--wall-seconds 900] [--data-root DIR] [--allowlist PATH] [--tissue-root DIR]
        [--structure-root DIR] [--segmenter-registry PATH (dry run only)] [--memory-max 9G] [--eval-memory-max 6G]
        [--min-gpu-free-mib 22528] [--home-floor-gb 100] [--wait-minutes 0] [--max-consecutive-failures 3]
        [--accept-code-change] [--no-j53] [--dry-run]

Arms. Each arm is the M2V1 dev arm (experiments/m2v1/run_batch.py, runs/20261031-m2v1-dev/plan.md: the method's final
configuration, the eta of the C05-7 sweep, 900 s) with the switches below and nothing else changed:
  p0n  none of them: the current path on the pilot's code, in the same period -- the primary baseline of every paired
       difference and reading (the lead, 10-31 ~6:15 pm CDT, before any pilot result);
  p1   --max-accepted-per-level 500,500,100,500 (C13-4a, D51; SCH);
  p2   --jd-cell-estimate certificate_bernstein (C09.1-3a, D53; JDT);
  p3   both;
  p4   P3 and --sigma-domain support_shared --support-threshold 0.05 (C04-6, D49 / D50; SWLI): exploratory arm added by
       BZ 10-31 ~9:55 pm CDT (revision 3, before any pilot result), described only, its eta the same (not re-swept for
       support_shared), its native memory not measured (SWLI gate 12): MemoryMax 9G as for every arm, an OOM is a
       failure, never rerun;
  p0   (not run) the M2V1 dev run of the same pair (code 68721d1), reused as a secondary reference column (checked:
       run id, pair, eta, config.json args against M2V1's own expected values, --max-accepted 500, no per-level caps,
       certificate).
--arms (default p0n,p1,p2,p3,p4) selects the arms to run; the order is pair-major: for each pair (selection.json order),
the arms in the --arms order, so that a deadline leaves complete pairs. One arm set per root (batch.json; a restart
with another set STOPs). Without p0n the readings fall back to P0 as the baseline (summarize.py).
The machinery is M2V1's: run_batch.ArmSweep (itself eta.Sweep) per arm, run_batch.Queue for the precheck (registry,
structure predictions, tissue layer, development allowlist, sealed lists, B-line training list), the gate, launch
records, resume rules, config.json checks, the hard-constraint-1 check and the driver lock; the changes are listed at
PilotArm and PilotQueue. Nothing under experiments/m2v1/ or experiments/eta/ is modified.

eta: <sweep-root>/eta_selected.json (real runs: sha256 pinned to ETA_SHA256, the file the M2V1 batch read) and
<sweep-root>/summary.json; the checks of M2V1 (positive finite alpha, nu, four tau; summary agrees; sweep stage
stage_T0_tune_v1, no pair or subject shared with the pilot); plus: M2V1's batch.json records the same eta and the same
source sha256, the same stage, the pilot's pairs as its first pairs, the same wall seconds, an empty --extra, the same
dry-run flag; the base configuration (the dev arm rebuilt on the pilot's code tree) equals the sweep's and M2V1's dev
configuration (run_batch.config_differences: structure root resolved, registry by sha256 / folds / status); each arm's
configuration equals the base configuration with exactly its changes. The model code (bayesreg/, jd_curl/,
configs/bayesreg/) of the pilot's commit differs from 68721d1 (the sweep's and M2V1's code): the changed files are
recorded and --accept-code-change is required (the plan says why the default path is unchanged). The eta record
eta/eta_pilot.json is written once and read by every run (--eta-json); a restart recomputes and compares it.
P0 record p0.json (written once, compared on a restart): per pair the M2V1 dev run's state (evaluated, eval_failed,
not_evaluated, run_failed, missing), its run id and launch records, after the checks above; a mismatch STOPs; an M2V1
evaluated field whose certificate did not pass STOPs (hard constraint 1). M2V1 must have ended: a tag-m2v1-* unit or an
experiments/m2v1/run_batch.py process -> STOP (--after-m2v1 waits for it, polling every 60 s, at most
--m2v1-wait-minutes; the deadline stops the wait). Missing P0 runs are recorded, the pilot goes on; their pairs have no
paired difference.

Per run (unit <prefix>-<arm>-<short>, MemoryMax 9G, MemorySwapMax 4G, CPUQuota 600%, RuntimeMaxSec wall + 1800) ->
runs/<arm>/<short>/; evaluation by bayesreg.eval.cli only (unit <prefix>-eval-<arm>-<short>, MemoryMax 6G, Nice 10) ->
eval/<arm>/<short>/pilot_<arm>_<short>/metrics.json (method bayesreg_path_pilot, arm P1 / P2 / P3 / P0n). After a run
its config.json args must equal the arm's expected values (eta.Sweep.expected_run_args plus the per-run values, plus
max_accepted 500 / max_accepted_per_level null / level_time_shares null, or max_accepted null / max_accepted_per_level
[500, 500, 100, 500] / level_time_shares null for p1, p3 and p4; sigma_domain brain, support_threshold null -- SWLI,
PILOT review minor 10 --, for p4 sigma_domain support_shared, support_threshold 0.05) and its eta the eta record, else
STOP.
Gate before every unit: M2V1's (hold file $ROOT/hold absent; no other tag-* / seq-vp-* unit; GPU free >=
--min-gpu-free-mib; MemAvailable >= MemoryMax + 2 GiB; /home >= --home-floor-gb) and no driver process of M2V1, of
the eta sweep or of another pilot (python with experiments/m2v1/run_batch.py, experiments/eta/sweep.py or
experiments/pilot/run_pilot.py; this process and its ancestors excepted); polled every 60 s for at most --wait-minutes.
Deadline (--no-start-after): an ISO date-time with a date and an explicit UTC offset only, e.g.
2026-11-01T05:00-06:00 (HH:MM and a naive date-time are refused at parsing: the window crosses midnight and the end of
DST on 11-01; PILOT review I2); a deadline already past at the start STOPs before anything is read. Resume rules,
failures (recorded, the queue goes on; --max-consecutive-failures in a row STOP, counted across arms) and the
hard-constraint-1 STOP as M2V1 (run_batch.py docstring).
End: experiments/pilot/summarize.py writes summary.json / summary.md (written once; a different recomputation goes to
summary_<UTC>.*), including the 53-point minimum J of every final_native.npz (CPU, read only; --no-j53: not computed,
recorded as "审核时另算").
Code version: --commit SHA (default the repository HEAD at start, tracked files clean) in a detached worktree $ROOT/code
(eta.Sweep.prepare_code). The code tree must offer --max-accepted-per-level, and certificate_bernstein when p2 or p3
is selected. Log: $ROOT/queue.log (UTC).
--dry-run: synthetic inputs only (experiments/m2v1/dry_inputs.py and a dry M2V1 root made from them by
experiments/m2v1/run_batch.py --dry-run --until dev --d47-last); unit prefix tag-pilotdry, device cpu, one thread,
Nice 19; every input and the root outside /home/rmcfz/seq_vp_tag_results; the eta sha256 is not pinned.
Exit codes: 0 done (or --until reached), 2 usage or another driver on the root, 8 STOP.
"""
import argparse
from datetime import datetime
import glob
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bayesreg.model import LOGVOL_GAUSS27_CLOSED27  # noqa: E402
from bayesreg.seg.structure_layer import DEFAULT_GROUPING, SEGMENTER_ID  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO/rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rb = _load("_pilot_m2v1_run_batch", "experiments/m2v1/run_batch.py")
eta = rb.eta
psum = _load("_pilot_summarize", "experiments/pilot/summarize.py")
Stop, Deadline = rb.Stop, rb.Deadline
read_json, write_once, utc, short_name, subjects_of = rb.read_json, rb.write_once, rb.utc, rb.short_name, rb.subjects_of
sha256_of, config_differences, parse_deadline = rb.sha256_of, rb.config_differences, rb.parse_deadline

PY = eta.PY
STAGE = rb.STAGE                                        # stage_L2R_val_v1
SWEEP_STAGE = eta.STAGE                                 # stage_T0_tune_v1
N_SELECTION = 19
N_PAIRS = 5
RESULTS = rb.RESULTS
REAL_ROOT = RESULTS/"path_pilot_v1"
M2V1_ROOT = rb.REAL_ROOT                                # /home/rmcfz/seq_vp_tag_results/m2v1_dev
SWEEP_ROOT = rb.SWEEP_ROOT                              # /home/rmcfz/seq_vp_tag_results/eta_sweep_v3
# the eta file the M2V1 batch read (eta_m2v1.json source_sha256; runs/20261030-eta-sweep/result.md "fd3eb2dda679")
ETA_SHA256 = "fd3eb2dda6798cf129705026c15438c7791aa74f0a541ad8c5d0e16eb1c2d5e1"
PROTECTED = rb.PROTECTED+(M2V1_ROOT,)
BASE_CELL = "certificate"
BERNSTEIN = "certificate_bernstein"
CAPS_P1 = (500, 500, 100, 500)
SUPPORT_SHARED = "support_shared"                       # C04-6 sigma_domain of the exploratory arm P4 (revision 3)
SUPPORT_THETA = .05                                     # C04-6 / D50 default theta, passed explicitly
MAX_ACCEPTED = 500                                      # run_pair's --max-accepted default (D19)
# arm id -> label, the switches that differ from the M2V1 dev arm, note
ARMS = dict(
    p1=dict(label="P1", caps=CAPS_P1, cell=BASE_CELL, note="按层接受步上限 500,500,100,500（C13-4a，D51）"),
    p2=dict(label="P2", caps=None, cell=BERNSTEIN, note="单元估计 certificate_bernstein（C09.1-3a，D53）"),
    p3=dict(label="P3", caps=CAPS_P1, cell=BERNSTEIN, note="P1 + P2"),
    p4=dict(label="P4", caps=CAPS_P1, cell=BERNSTEIN, sigma=SUPPORT_SHARED, theta=SUPPORT_THETA,
            note="P3 + --sigma-domain support_shared（θ .05；探索臂，只描述；BZ 10-31 约 9:55 pm，修订 3）"),
    p0n=dict(label="P0n", caps=None, cell=BASE_CELL, note="现行路径在本试验代码与时段上（主要对照；总负责人 10-31）"))
DEFAULT_ARMS = "p0n,p1,p2,p3,p4"
METHOD = "bayesreg_path_pilot"
SCHEMA = "bayesreg.path_pilot.v1"
M2V1_SCRIPT = "experiments/m2v1/run_batch.py"
DRIVER_SCRIPTS = (M2V1_SCRIPT, "experiments/eta/sweep.py", "experiments/pilot/run_pilot.py")
DRY_SCRIPT = rb.DRY_SCRIPT                              # experiments/m2v1/dry_inputs.py (the synthetic inputs)
POLL_SECONDS = eta.POLL_SECONDS


def driver_processes(proc, self_pid, scripts):
    """PIDs of driver processes under the /proc tree `proc`: argv[0] is a python interpreter and some argument ends with
    one of `scripts`; this process and its ancestors excepted (experiments/m2v1/run_batch.py sweep_processes, generalised
    to several scripts; its helpers _argv and _ppid are used as they are)."""
    ancestors, pid = set(), int(self_pid)
    while pid and pid not in ancestors:
        ancestors.add(pid)
        pid = rb._ppid(proc, pid)
    found = []
    for entry in sorted(Path(proc).glob("[0-9]*"), key=lambda p: int(p.name) if p.name.isdigit() else -1):
        if not entry.name.isdigit() or int(entry.name) in ancestors:
            continue
        argv = rb._argv(proc, entry.name)
        if argv and Path(argv[0]).name.startswith("python") and any(a.endswith(s) for a in argv[1:] for s in scripts):
            found.append(int(entry.name))
    return found


class PilotSystem(rb.QueueSystem):
    """run_batch.QueueSystem (no unit excepted but this script's) with the driver processes and M2V1's state."""

    def drivers_running(self):
        return driver_processes(Path("/proc"), os.getpid(), DRIVER_SCRIPTS)

    def m2v1_state(self):
        """(active tag-m2v1-* units, PIDs of experiments/m2v1/run_batch.py processes)."""
        out = self._out(["systemctl", "--user", "list-units", "--state=active,activating,deactivating", "--no-legend",
                         "--plain", "tag-m2v1-*"])
        units = [line.split()[0] for line in out.splitlines() if line.strip()]
        return units, driver_processes(Path("/proc"), os.getpid(), (M2V1_SCRIPT,))


def pilot_namespace(o, spec):
    """The eta.Sweep option namespace of one arm: run_batch.arm_namespace with the arm's cell estimate and no --extra."""
    return argparse.Namespace(
        root=o.root, code=o.code, commit=o.commit, until=None, channel_source="structure",
        logvol_points=LOGVOL_GAUSS27_CLOSED27, logvol_closed27_share=1., metric_jd_scaling="affine_v1",
        jd_cell_estimate=spec["cell"], structure_root=o.structure_root, structure_grouping=DEFAULT_GROUPING,
        segmenter_id=SEGMENTER_ID, segmenter_registry=o.segmenter_registry, backend="torch64", device="cuda", outer="a",
        rounds="1,1,1,1", wall_seconds=o.wall_seconds, extra="", data_root=o.data_root, allowlist=o.allowlist,
        tissue_root=o.tissue_root, memory_max=o.memory_max, eval_memory_max=o.eval_memory_max,
        probe_memory_max=o.memory_max, atropos_memory_max=o.eval_memory_max, atropos_threads=1,
        min_gpu_free_mib=o.min_gpu_free_mib, home_floor_gb=o.home_floor_gb, wait_minutes=o.wait_minutes,
        dry_run=o.dry_run, sigma_domain=spec.get("sigma"), support_threshold=spec.get("theta"))


def changes_of(spec):
    """The configuration keys in which an arm differs from the M2V1 dev arm."""
    out = {}
    if spec["cell"] != BASE_CELL:
        out["jd_cell_estimate"] = spec["cell"]
    if spec["caps"] is not None:
        out["max_accepted_per_level"] = list(spec["caps"])
    if spec.get("sigma") is not None:                   # eta.Sweep.probe_configuration's keys under a non-brain domain
        out.update(sigma_domain=spec["sigma"], support_threshold=float(spec["theta"]))
    return out


class PilotArm(rb.ArmSweep):
    """run_batch.ArmSweep with: the unit prefix tag-pilot / tag-pilotdry; the per-level caps (configuration key
    max_accepted_per_level, the run_pair switch, the config.json args check); the code-tree check of the pilot's
    switches (need_flags, set by the queue on the base arm, which alone prepares the code); the gate also waits for
    driver processes (M2V1, the eta sweep, another pilot)."""

    def __init__(self, opts, system, arm, spec):
        super().__init__(opts, system, arm)
        self.prefix = "tag-pilotdry" if self.dry else "tag-pilot"
        self.spec = spec
        self.caps = None if spec["caps"] is None else list(spec["caps"])
        self.need_flags = ()

    def configuration(self):
        conf = super().configuration()
        if self.caps is not None:
            conf["max_accepted_per_level"] = list(self.caps)
        return conf

    def run_args(self, pair, eta_file):
        args = super().run_args(pair, eta_file)
        if self.caps is not None:
            args += ["--max-accepted-per-level", ",".join(str(c) for c in self.caps)]
        return args

    def expected_args(self, pair, eta_file):
        expected = super().expected_args(pair, eta_file)
        if not self.support:                                               # SWLI default (PILOT review minor 10)
            expected.update(sigma_domain="brain", support_threshold=None)
        # P4: eta.Sweep.expected_run_args already holds sigma_domain support_shared and support_threshold 0.05
        if self.caps is not None:          # run_pair resolve_schedule_args (C16 SCH): max_accepted null with per-level caps
            expected.update(max_accepted=None, max_accepted_per_level=list(self.caps), level_time_shares=None)
        else:                              # no SCH switch: max_accepted the default, the SCH keys absent (read as null)
            expected.update(max_accepted=MAX_ACCEPTED, max_accepted_per_level=None, level_time_shares=None)
        return expected

    def check_code_tree(self):
        super().check_code_tree()
        if not self.need_flags:
            return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        out = subprocess.run([PY, "-B", "-m", "bayesreg.cli.run_pair", "--help"], cwd=self.code, env=env,
                                 capture_output=True, text=True, check=False).stdout
        missing = [f for f in self.need_flags if f not in out]
        if missing:
            raise Stop(f"bayesreg.cli.run_pair in {self.code} lacks {missing} (SCHI / JDTI not in this code tree)")

    def busy_reason(self, memmax_mib, gpu):
        drivers = self.sys.drivers_running()
        if drivers:
            return (f"driver processes running: {drivers} (experiments/m2v1/run_batch.py, experiments/eta/sweep.py or "
                    "another experiments/pilot/run_pilot.py)")
        return super().busy_reason(memmax_mib, gpu)


class PilotQueue(rb.Queue):
    """run_batch.Queue with the pilot's arms, pairs (the first five of selection.json), eta checks (plus M2V1's batch),
    the P0 record, pair-major order, evaluation names and summary. Inherited as they are: precheck, share_code,
    record_failure, check_run, one, hard_constraint_1, code_difference, lock, main."""

    def __init__(self, opts, system=None):              # noqa: super().__init__ builds M2V1's two arms; not called
        self.o = opts
        self.sys = PilotSystem() if system is None else system
        self.root = Path(opts.root)
        self.dry = opts.dry_run
        self.arm_ids = [a.strip() for a in opts.arms.split(",")]
        self.base = PilotArm(pilot_namespace(opts, dict(cell=BASE_CELL)), self.sys, "base",
                             dict(label="base", caps=None, cell=BASE_CELL, note="M2V1 dev arm"))
        self.arms = {a: PilotArm(pilot_namespace(opts, ARMS[a]), self.sys, a, ARMS[a]) for a in self.arm_ids}
        flags = ["--max-accepted-per-level"]
        if any(ARMS[a]["cell"] == BERNSTEIN for a in self.arm_ids):
            flags.append(BERNSTEIN)
        if any(ARMS[a].get("sigma") for a in self.arm_ids):
            flags += ["--sigma-domain", SUPPORT_SHARED, "--support-threshold"]
        self.base.need_flags = tuple(flags)
        self.deadline = parse_deadline(opts.no_start_after)
        for arm in (self.base, *self.arms.values()):
            arm.deadline = self.deadline
        self.pairs, self.eta_file, self.eta, self.eta_record = [], None, None, None
        self.m2v1_batch, self.p0_record, self.pool_record = None, None, None
        self.in_row = 0

    # ---- start ---------------------------------------------------------------------------------------------------
    def root_checks(self):
        root = self.root.resolve()
        others = [Path(self.o.m2v1_root).resolve(), Path(self.o.sweep_root).resolve()]
        for bad in PROTECTED+tuple(others):
            if root == bad or bad in root.parents or root in bad.parents:
                return f"--root {root} overlaps {bad} (results of other work; read only)"
        if self.dry:
            if root == RESULTS or RESULTS in root.parents:
                return f"--dry-run: --root must lie outside {RESULTS}"
            for name in ("m2v1_root", "sweep_root"):
                path = Path(getattr(self.o, name)).resolve()
                if path == RESULTS or RESULTS in path.parents:
                    return f"--dry-run: --{name.replace('_', '-')} {path} lies under {RESULTS} (synthetic inputs only)"
        return ""

    def start_checks(self):
        if self.o.segmenter_registry and not self.dry:
            raise Stop("--segmenter-registry: dry runs only (real runs use the code tree's registered registry)")
        if self.dry:
            self.base.dry_checks()
        stage_dir = self.base.data_root/STAGE
        selection = read_json(stage_dir/"selection.json")
        if selection is None:
            raise Stop(f"{stage_dir}/selection.json missing")
        listed = [row["pair"] for row in selection.get("pairs") or []]
        if len(set(listed)) != len(listed):
            raise Stop(f"{stage_dir}/selection.json lists a pair twice")
        if not self.dry and len(listed) != N_SELECTION:
            raise Stop(f"{stage_dir}/selection.json has {len(listed)} pairs, expected {N_SELECTION}")
        self.pairs = listed[:N_PAIRS]
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
        self.log(f"pairs (the first {N_PAIRS} of {STAGE}/selection.json, its order): "
                 + ", ".join(short_name(p) for p in self.pairs) + f"; arms {self.arm_ids} (pair-major)")

    # ---- M2V1 ----------------------------------------------------------------------------------------------------
    def wait_for_m2v1(self):
        waited = 0
        while True:
            units, pids = self.sys.m2v1_state()
            if not units and not pids:
                self.log("after-m2v1: no tag-m2v1-* unit and no experiments/m2v1/run_batch.py process; going on")
                return
            if self.deadline is not None and self.sys.now() >= self.deadline:
                raise Stop(f"after-m2v1: --no-start-after {self.deadline.isoformat()} reached while waiting for M2V1; "
                           "nothing started")
            if waited >= self.o.m2v1_wait_minutes:
                raise Stop(f"after-m2v1: M2V1 has not ended after {waited} min (--m2v1-wait-minutes "
                           f"{self.o.m2v1_wait_minutes}); units {units}, processes {pids}; nothing started")
            if waited % 30 == 0:
                self.log(f"after-m2v1: waiting (units {' '.join(units) or 'none'}; processes {pids or 'none'})")
            waited += 1
            self.sys.sleep(POLL_SECONDS)

    # ---- eta -----------------------------------------------------------------------------------------------------
    def eta_phase(self):
        sweep, m2 = Path(self.o.sweep_root), Path(self.o.m2v1_root)
        path, spath = sweep/"eta_selected.json", sweep/"summary.json"
        if not path.is_file() or not spath.is_file():
            raise Stop(f"{path} or {spath} missing")
        doc, summary = read_json(path), read_json(spath)
        value = dict(alpha=doc.get("alpha"), nu=doc.get("nu"), tau=doc.get("tau"))
        good = (eta.finite(value["alpha"]) and value["alpha"] > 0 and eta.finite(value["nu"]) and value["nu"] > 0
                and isinstance(value["tau"], list) and len(value["tau"]) == rb.K_T_PLUS_1
                and all(eta.finite(t) and t > 0 for t in value["tau"]))
        if not good:
            raise Stop(f"{path}: eta {value} (alpha, nu > 0 finite; tau {rb.K_T_PLUS_1} positive values)")
        source_sha = sha256_of(path)
        if not self.dry and source_sha != ETA_SHA256:
            raise Stop(f"{path}: sha256 {source_sha} is not the pinned {ETA_SHA256} (the file M2V1 read)")
        if summary.get("eta_selected") != value:
            raise Stop(f"{spath} eta_selected {summary.get('eta_selected')} differs from {path} {value}")
        if summary.get("stage_set") != SWEEP_STAGE:
            raise Stop(f"{spath}: stage_set {summary.get('stage_set')!r}, expected {SWEEP_STAGE}")
        sweep_pairs = list(summary.get("pairs") or [])
        sweep_subjects = {s for p in sweep_pairs for s in subjects_of(p)}
        ours = {s for p in self.pairs for s in subjects_of(p)}
        if not sweep_pairs or set(sweep_pairs) & set(self.pairs) or sweep_subjects & ours:
            raise Stop(f"the sweep's pairs {sweep_pairs} overlap the pilot or are missing")
        mpath = m2/"batch.json"
        mb = read_json(mpath)
        if mb is None:
            raise Stop(f"{mpath} missing (the M2V1 batch is the P0 arm)")
        mdev = (mb.get("arms") or {}).get("dev") or {}
        problems = []
        if bool(mb.get("dry_run")) != self.dry:
            problems.append(f"dry_run {mb.get('dry_run')} (pilot {self.dry})")
        if mb.get("stage") != STAGE:
            problems.append(f"stage {mb.get('stage')!r}")
        if mb.get("eta") != value:
            problems.append(f"eta {mb.get('eta')} (sweep file {value})")
        if (mb.get("eta_source") or {}).get("source_sha256") != source_sha:
            problems.append(f"eta source sha256 {(mb.get('eta_source') or {}).get('source_sha256')} (pilot {source_sha})")
        if list(mb.get("pairs") or [])[:len(self.pairs)] != self.pairs:
            problems.append(f"its first pairs {list(mb.get('pairs') or [])[:len(self.pairs)]} are not the pilot's")
        if mb.get("wall_seconds") != self.o.wall_seconds:
            problems.append(f"wall_seconds {mb.get('wall_seconds')} (pilot {self.o.wall_seconds})")
        if (mdev.get("configuration") or {}).get("extra") != "":
            problems.append(f"dev arm --extra {(mdev.get('configuration') or {}).get('extra')!r} (the pilot's arms are "
                            "defined against a dev arm without --extra)")
        if not mdev.get("expected_args"):
            problems.append("no dev arm expected_args")
        if problems:
            raise Stop(f"{mpath} does not fit the pilot: {'; '.join(problems)}")
        base = self.base.configuration()
        checks = {}
        for name, other in (("sweep", doc.get("configuration")), ("m2v1_dev", mdev.get("configuration"))):
            differ = config_differences(other, base, d47=False)
            if differ:
                raise Stop(f"the base configuration differs from the {name} configuration: {'; '.join(differ)}")
            checks[name] = "equal to the base configuration"
        for aid, arm in self.arms.items():
            differ = config_differences(dict(base, **changes_of(arm.spec)), arm.configuration(), d47=False)
            if differ:
                raise Stop(f"arm {aid}: configuration is not the base configuration with exactly "
                           f"{changes_of(arm.spec)}: {'; '.join(differ)}")
            checks[aid] = f"the base configuration with {changes_of(arm.spec) or 'no change'}"
        changed_sweep = self.code_difference(doc.get("code_sha"))
        changed_m2v1 = self.code_difference(mb.get("code_sha"))
        if (changed_sweep or changed_m2v1) and not self.o.accept_code_change:
            raise Stop(f"the model code differs from the sweep's ({doc.get('code_sha')}: {changed_sweep}) or M2V1's "
                       f"({mb.get('code_sha')}: {changed_m2v1}); --accept-code-change records and accepts it (plan: "
                       "the default path is bitwise unchanged by the golden tests)")
        record = dict(schema=SCHEMA+".eta", alpha=value["alpha"], nu=value["nu"], tau=value["tau"],
                      use="η 由 C05-7 扫描在 T0 调参对上选出（EB 初值）；与 M2V1 相同，本试验照用，不在 L2R 验证对上调",
                      source=str(path), source_sha256=source_sha, pinned_sha256=None if self.dry else ETA_SHA256,
                      summary=str(spath), summary_sha256=sha256_of(spath), sweep_stage_set=SWEEP_STAGE,
                      sweep_pairs=sweep_pairs, sweep_code_sha=doc.get("code_sha"), m2v1_root=str(m2),
                      m2v1_batch_sha256=sha256_of(mpath), m2v1_code_sha=mb.get("code_sha"),
                      m2v1_eta_file=mb.get("eta_file"), code_sha=self.base.code_sha,
                      model_code_changed_vs_sweep=changed_sweep, model_code_changed_vs_m2v1=changed_m2v1,
                      accept_code_change=bool((changed_sweep or changed_m2v1) and self.o.accept_code_change),
                      sweep_configuration=doc.get("configuration"), configuration_check=checks)
        (self.root/"eta").mkdir(exist_ok=True)
        self.eta_file = self.root/"eta"/"eta_pilot.json"
        self.base.compare_or_write(self.eta_file, record)
        self.eta, self.eta_record, self.m2v1_batch = value, record, mb
        self.log(f"eta: alpha {value['alpha']:.6g}, nu {value['nu']:.6g}, tau {value['tau']} from {path} (sha256 "
                 f"{source_sha[:12]}{'' if self.dry else ', pinned'}); M2V1 {mpath} agrees; configuration {checks}; "
                 f"model code changed vs sweep {len(changed_sweep)} files, vs M2V1 {len(changed_m2v1)} files "
                 f"(accepted {int(record['accept_code_change'])}); record {self.eta_file}")

    # ---- P0 --------------------------------------------------------------------------------------------------------
    def p0_phase(self):
        units, pids = self.sys.m2v1_state()
        if units or pids:
            raise Stop(f"M2V1 is still running (units {units}, processes {pids}); P0 is read after it ends "
                       "(--after-m2v1 waits)")
        m2, mb = Path(self.o.m2v1_root), self.m2v1_batch
        expected_base = dict(mb["arms"]["dev"]["expected_args"], max_accepted=MAX_ACCEPTED, max_accepted_per_level=None,
                             level_time_shares=None)
        if expected_base.get("jd_cell_estimate") != BASE_CELL:
            raise Stop(f"M2V1 dev expected jd_cell_estimate {expected_base.get('jd_cell_estimate')!r}")
        rows = {}
        for pair in self.pairs:
            short = short_name(pair)
            run, ev = m2/"runs"/"dev"/short, m2/"eval"/"dev"/short
            rid = f"m2v1_dev_{short}"
            launches = [Path(p).name for p in sorted(glob.glob(str(m2/"launch"/f"run_dev-{short}_*.txt")))]
            entry = dict(pair=pair, run_dir=str(run), eval_dir=str(ev), run_id=rid, launches=launches)
            status, config = read_json(run/"status.json"), read_json(run/"config.json")
            if status is None:
                entry["state"] = "missing" if not run.exists() and not launches else "run_failed"
            else:
                args = (config or {}).get("args")
                if not isinstance(args, dict):
                    raise Stop(f"P0 {short}: {run}/config.json has no args")
                differ = rb.summarize.args_check(args, dict(expected_base, pair=f"{STAGE}/{pair}"))
                if differ:
                    raise Stop(f"P0 {short}: config.json args differ from M2V1's dev arm (got, expected): {differ}")
                if rb.summarize.eta_check((config or {}).get("eta"), self.eta) is not True:
                    raise Stop(f"P0 {short}: config.json eta {(config or {}).get('eta')} differs from {self.eta}")
                entry.update(run_status=status.get("status"), exit_code=status.get("exit_code"))
                if (ev/"done.json").is_file():
                    done = read_json(ev/"done.json") or {}
                    doc = self.hard_constraint_1(f"P0 {short}", ev)
                    if done.get("run_id") != rid or doc.get("run_id") != rid or doc.get("pair") != pair:
                        raise Stop(f"P0 {short}: run id {done.get('run_id')} / {doc.get('run_id')} or pair "
                                   f"{doc.get('pair')} (expected {rid}, {pair})")
                    entry.update(state="evaluated", field_role=doc.get("field_role"),
                                 certificate_passed=(doc.get("certificate") or {}).get("passed"))
                elif (ev/"eval_failed.json").is_file():
                    entry["state"] = "eval_failed"
                else:
                    entry["state"] = "not_evaluated"
            rows[short] = entry
        record = dict(schema=SCHEMA+".p0", m2v1_root=str(m2), m2v1_batch_sha256=sha256_of(m2/"batch.json"),
                      m2v1_code_sha=mb.get("code_sha"), pilot_code_sha=self.base.code_sha,
                      model_code_changed=self.eta_record["model_code_changed_vs_m2v1"],
                      rule=("P0 = M2V1 开发批同一对的 dev 运行，复用、不重跑；核对 run id、pair、η、config.json args "
                            "（M2V1 自己的应有值，另加 max_accepted 500、无按层上限、certificate）与硬约束 1"),
                      pairs=rows)
        self.p0_record = self.base.compare_or_write(self.root/"p0.json", record)
        states = {k: v["state"] for k, v in rows.items()}
        self.log(f"P0 (M2V1 dev runs, code {mb.get('code_sha')}): {states}; record {self.root/'p0.json'}")

    # ---- batch -----------------------------------------------------------------------------------------------------
    def write_batch(self):
        arms = {aid: dict(label=arm.spec["label"], note=arm.spec["note"], changes=changes_of(arm.spec),
                          configuration=arm.configuration(), expected_args=arm.expected_args(None, self.eta_file),
                          eval_arm=arm.spec["label"], unit_prefix=f"{arm.prefix}-{aid}-", run_id=f"pilot_{aid}_<short>",
                          pairs=list(self.pairs))
                for aid, arm in self.arms.items()}
        document = dict(schema=SCHEMA+".batch", stage=STAGE, pairs=self.pairs, arm_order=self.arm_ids,
                        order="pair-major: for each pair (selection.json order) the arms in arm_order", arms=arms,
                        base_configuration=self.base.configuration(), method=METHOD, wall_seconds=self.o.wall_seconds,
                        code_sha=self.base.code_sha, eta=self.eta, eta_file=str(self.eta_file),
                        eta_source={k: self.eta_record[k] for k in ("source", "source_sha256", "summary",
                                                                    "summary_sha256", "sweep_code_sha", "sweep_pairs")},
                        m2v1=dict(root=str(self.o.m2v1_root), batch_sha256=self.eta_record["m2v1_batch_sha256"],
                                  code_sha=self.eta_record["m2v1_code_sha"], arm="dev",
                                  p0_record=str(self.root/"p0.json")),
                        pools=self.pool_record, dry_run=self.dry)
        self.base.compare_or_write(self.root/"batch.json", document)
        self.log(f"batch: {self.root/'batch.json'}; deadline "
                 + (f"{self.deadline.isoformat()} (no run unit starts at or after it)" if self.deadline else "none"))

    def evaluate(self, arm, name, short, run, ev):
        """run_batch.Queue.evaluate with the pilot's run id (pilot_<arm>_<short>), method and arm label."""
        earlier = sorted(glob.glob(str(ev/"eval_*.log")))+sorted(glob.glob(str(ev/"long.csv")))+arm.earlier("eval", name)
        if earlier:
            if not (ev/"eval_failed.json").exists():
                write_once(ev/"eval_failed.json", dict(reason="earlier evaluation attempt without done.json; kept, not "
                                                        "rerun", files=earlier, recorded_utc=utc()))
            self.log(f"evaluation of {name}: earlier attempt without done.json ({' '.join(earlier)}); recorded as failed")
            return "failed"
        ev.mkdir(parents=True, exist_ok=True)
        s = eta.stamp()
        log, exit_file = ev/f"eval_{s}.log", ev/f"eval_exit_{s}.txt"
        rid = f"pilot_{arm.arm}_{short}"
        command = [PY, "-B", "-m", "bayesreg.eval.cli", run, "--pairs-stage", STAGE, "--out", "metrics.json", "--long",
                   ev/"long.csv", "--summary", ev/"summary.json", "--out-dir", ev, "--method", METHOD, "--arm",
                   arm.spec["label"], "--run-id", rid, "--data-root", arm.data_root, "--allowlist", arm.allowlist]
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
        self.log(f"evaluated {name}: field_role {doc.get('field_role')}; certificate {cert.get('status')} "
                 f"{cert.get('min_jd_lower_bound')}")
        return "ok"

    def batch(self):
        """Pair-major: for each pair, the arms in --arms order (run_batch.Queue.one per run; deadline and consecutive
        failures as M2V1)."""
        not_run = []
        for pair in self.pairs:
            for aid in self.arm_ids:
                try:
                    outcome = self.one(aid, pair)
                except Deadline as exc:
                    not_run.append(f"{aid}-{short_name(pair)}")
                    self.log(f"not run: {exc} (未运行, not a failure)")
                    continue
                if outcome == "failed":
                    self.in_row += 1
                    if self.in_row >= self.o.max_consecutive_failures:
                        raise Stop(f"{self.in_row} failed runs in a row (--max-consecutive-failures "
                                   f"{self.o.max_consecutive_failures}); kept, the lead decides")
                elif outcome == "ok":
                    self.in_row = 0
        return not_run

    # ---- main ----------------------------------------------------------------------------------------------------
    def run(self):
        self.log(f"start: root {self.root}; until {self.o.until}; arms {self.arm_ids} "
                 f"({ {a: changes_of(ARMS[a]) for a in self.arm_ids} }); M2V1 root {self.o.m2v1_root} (after-m2v1 "
                 f"{int(self.o.after_m2v1)}); sweep root {self.o.sweep_root}; wall {self.o.wall_seconds:g} s (预算收尾); "
                 f"MemoryMax {self.o.memory_max} (eval {self.o.eval_memory_max}); min GPU free {self.o.min_gpu_free_mib} "
                 f"MiB; deadline {self.deadline.isoformat() if self.deadline else 'none'} (--no-start-after "
                 f"{self.o.no_start_after}); dry-run {int(self.dry)}; self unit "
                 f"{getattr(self.sys, 'self_unit', '') or 'none'}")
        try:
            if self.deadline is not None and self.sys.now() >= self.deadline:
                raise Stop(f"--no-start-after {self.deadline.isoformat()} is already past (now "
                           f"{self.sys.now().isoformat()}); nothing read, nothing started (PILOT review I2)")
            self.start_checks()
            self.base.prepare_code()
            self.share_code()
            self.base.write_env()
            self.precheck()
            if self.o.until == "precheck":
                self.log("until precheck: end")
                return 0
            if self.o.after_m2v1:
                self.wait_for_m2v1()
            self.eta_phase()
            self.p0_phase()
            self.write_batch()
            if self.o.until == "eta":
                self.log("until eta: end")
                return 0
            not_run = self.batch()
            if not_run:
                self.log(f"not run (deadline; 未运行, not failures): {', '.join(not_run)}")
            try:
                js, md, state = psum.write(self.root, j53=not self.o.no_j53)
            except psum.SummaryError as exc:
                raise Stop(f"summary: {exc}") from None
            self.log(f"summary: {js}, {md} ({state})")
            self.log("end")
            return 0
        except Stop as exc:
            self.log(f"STOP: {exc}")
            return 8


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -B experiments/pilot/run_pilot.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default=str(REAL_ROOT))
    parser.add_argument("--code")
    parser.add_argument("--commit")
    parser.add_argument("--m2v1-root", default=str(M2V1_ROOT))
    parser.add_argument("--sweep-root", default=str(SWEEP_ROOT))
    parser.add_argument("--arms", default=DEFAULT_ARMS, help=f"comma-separated arm ids among {sorted(ARMS)} (default "
                        f"{DEFAULT_ARMS}; p0n is the baseline of the readings)")
    parser.add_argument("--until", choices=("precheck", "eta"), default=None)
    parser.add_argument("--after-m2v1", action="store_true")
    parser.add_argument("--m2v1-wait-minutes", type=int, default=600)
    parser.add_argument("--no-start-after", help="ISO date-time with a date and a UTC offset, e.g. "
                        "2026-11-01T05:00-06:00 (HH:MM and naive date-times are refused); no run unit starts at or "
                        "after it")
    parser.add_argument("--wall-seconds", type=float, default=900., help="must equal the M2V1 batch's (checked)")
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
    parser.add_argument("--no-j53", action="store_true", help="summary: do not compute the 53-point minimum J of the "
                        "final states (recorded as to be computed at the audit)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    arms = [a.strip() for a in args.arms.split(",")]
    if not arms or any(a not in ARMS for a in arms) or len(set(arms)) != len(arms):
        parser.error(f"--arms {args.arms!r}: distinct ids among {sorted(ARMS)}")
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
    if args.no_start_after is not None:
        try:
            moment = datetime.fromisoformat(args.no_start_after)
        except ValueError:
            moment = None
        if moment is None or moment.tzinfo is None or re.fullmatch(r"[0-9]{1,2}:[0-9]{2}", args.no_start_after):
            parser.error(f"--no-start-after {args.no_start_after!r}: an ISO date-time with a date and a UTC offset, e.g. "
                         "2026-11-01T05:00-06:00 (PILOT review I2: the window crosses midnight and the end of DST)")
    return args


def main(argv=None, system=None):
    return PilotQueue(parse_args(argv), system).main()


if __name__ == "__main__":
    sys.exit(main())
