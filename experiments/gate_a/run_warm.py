"""WARM (runs/20261103-warm/plan.md, 任务书.md; contract C12-4a, D59): X2d's objective continued from the A1Ph field, native level only.
Development set only (stage_L2R_val_v1); every number it yields is a development-set single observation; n = 5 gives direction only.

    python -B experiments/gate_a/run_warm.py --phase probe|pairs [--root DIR] [--code DIR] [--commit SHA] [--until precheck|eta]
        [--dry-run] [--bz-go TEXT] [--seed-root DIR] [--eta-json PATH] [--x2-root DIR] [--no-start-after ISO] [--data-root DIR]
        [--allowlist PATH] [--tissue-root DIR] [--structure-root DIR] [--memory-max 9G] [--eval-memory-max 6G] [--min-gpu-free-mib 22528]
        [--home-floor-gb 100] [--wait-minutes 30] [--max-consecutive-failures 3]
        [--synthetic --segmenter-registry PATH --wall-seconds S --checkpoint-seconds S (the tests' synthetic inputs only)]

Objective (the same as X2d, experiments/gate_a/run_x2.py arm d; its eta file is read, not rewritten): channel source structure,
unet35_v1_ens, structure13_v1, --sigma-domain support_shared --support-threshold 0.05, nu 1.37, tau (1,1,1,1), alpha .01, A′.  Still a
diagnostic point, not an eta the contract selected (C04-6).
Seed: the A1Ph final field <seed-root>/<pair>/pg_weak__four__90s/final_native.npz, key u; --init-field PATH --start-level 3.  The seed's
source reads no truth label (D0 A1Ph: U-Net predicted colouring); its hashes go into seeds.json and every record of the run.
Phases (one arm, one root, the same code; every pair runs once and the probe's run is that pair's result; the lead's revision before any run: the
probe is 900 s, not 300 s, because the bars of the verdict are written for 900 s):
  probe  pair 0441-0442, --wall-seconds 900, --checkpoint-seconds 225: runs/warm/0441-0442/.  The summary's verdict
         (experiments/gate_a/summarize_warm.py): stagnation stops here and goes to BZ, also "moved but below the bar".
  pairs  the other four of the first five pairs of selection.json (0438-0439, 0439-0440, 0440-0441, 0442-0443), the same arguments: runs/warm/<short>/.
         Never launches 0441-0442 (rb.Queue.one also refuses to relaunch anything with a launch record, a run directory or a done.json); its summary
         covers all five.  Starts only after a probe whose verdict is go, or with --bz-go "<BZ's words and time>" (recorded in batch_pairs.json and queue.log).
Per run: unit tag-warm-warm-<short> (MemoryMax 9G, MemorySwapMax 4G, CPUQuota 600%, RuntimeMaxSec wall + 1800) -> runs/warm/<short>/; one
evaluation by bayesreg.eval.cli of the final field (unit tag-warm-eval-warm-<short> -> eval/warm/<short>/gawarm_warm_<short>/metrics.json) and
of the entry and each checkpoint (unit tag-warm-eval-warm-<short>-<field> -> eval_fields/warm/<short>/<field>/gawarm_warm_<short>_<field>/
metrics.json).  Method bayesreg_warm_v1, arm WARM (fields: WARM-<field>).  After a run: config.json args must equal the expected values
(init_field, start_level, init_field_oracle_ok, checkpoint_seconds, sigma_domain, support_threshold included), its eta the eta record, its seed
record seeds.json, its solver record support_shared, start level 3 and the checkpoint rule, else STOP.  init_field_rejected: STOP (the precheck
passed, so a defect).  An evaluated field whose evaluator certificate did not pass: STOP (hard constraint 1).
Single, never rerun: resume rules, failures (recorded, not rerun; --max-consecutive-failures in a row STOP), the gate (hold file $ROOT/hold; no
other tag-* / seq-vp-* unit; no driver process of M2V1, the eta sweep, the pilot, X2 or another WARM; GPU free; MemAvailable; /home; before every
GPU run unit a 1-minute load < 1.5), the precheck (registry, structure predictions, tissue layer, allowlist, sealed lists, training list) and
the driver lock are X2's (experiments/gate_a/run_x2.py), which are M2V1's and the pilot's; only the unit prefix, the seed and the evaluations of
the entry and the checkpoints are added here.  Before every launch of a real root the command must not contain FORBIDDEN_WORD below: an idle check
matched that word in a start command once (runs/20261101-swl-gate12/result.md).
Records: batch_<phase>.json, seeds.json (written once, compared on a restart), queue.log (UTC), summary_<phase>.json / .md (written once; a
different recomputation goes to summary_<phase>_<UTC>.*).
Code version: --commit SHA (default the repository HEAD at start, tracked files clean) in a detached worktree $ROOT/code; the phase pairs reuses
it (the tree must be the probe's).
--dry-run: prints every command line of the phase (systemd-run and the command, the evaluations with the checkpoints of the plan) after the real
run_pair parser accepted the arguments and they matched the expected values; the seeds and the eta record are checked; nothing is written, no unit
or solver starts, no GPU.  --synthetic: synthetic inputs only (the tests' fake machine): unit prefix tag-warmdry, device cpu, one thread, Nice 19; every
input and the root outside /home/rmcfz/seq_vp_tag_results; eta sha256 not pinned.
Exit codes: 0 done (or --until reached; a probe verdict of stagnation is a result, not a failure), 2 usage or another driver on the root, 8 STOP.
"""
import argparse
from datetime import datetime
import glob
import importlib.util
import math
import os
from pathlib import Path
import re
import shlex
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bayesreg.solve.init_field import INIT_FIELD_REJECTED, _read_manifest as read_seed_manifest  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO/rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


x2 = _load("_warm_run_x2", "experiments/gate_a/run_x2.py")
sw = _load("_warm_summarize", "experiments/gate_a/summarize_warm.py")
pl, rb, eta = x2.pl, x2.rb, x2.eta
Stop = x2.Stop
read_json, write_once, utc, short_name = x2.read_json, x2.write_once, x2.utc, x2.short_name
parse_deadline, sha256_of = x2.parse_deadline, x2.sha256_of

PY = x2.PY
STAGE = x2.STAGE
RESULTS = x2.RESULTS
REAL_ROOT = RESULTS/"warm_v1"
PROTECTED = x2.PROTECTED+(x2.REAL_ROOT,)
METHOD = "bayesreg_warm_v1"
SCHEMA = "bayesreg.warm.v1"
TASK = "pg_weak__four__90s"                         # the frozen solver's task directory
SEED_ARM = "A1Ph"
SEED_STAGE = "D0-A1Ph"
SEED_ROLE = "formal_native_outer_commit"
SEED_ROOT = RESULTS/"d0_v1_a1p"/"runs"/SEED_ARM
ETA_JSON = x2.REAL_ROOT/"eta"/"eta_d.json"
ETA_SHA256 = "6b0b8a85dd6490044aa1866b908522cfafc2e4d1df5350e50ffc7022b6a4154d"     # gate_a_x2/eta/eta_d.json, read by the X2d runs
START_LEVEL = 3
PROBE_SHORT = "0441-0442"
FIRST_FIVE = ("0438-0439", "0439-0440", "0440-0441", "0441-0442", "0442-0443")
FORBIDDEN_WORD = "py"+"test"                        # gate 12 incident: an idle check matched it in a start command (spelled apart: this file may be grepped)
_D = x2.ARMS["d"]                                   # X2d's objective, nothing else
ARM = "warm"
ARMS = {ARM: dict(_D, label="WARM", wall_seconds=900., checkpoint_seconds=225.,
                  note="A1Ph 种子 + X2d 目标（诊断点，不是合同选出的 η），900 s；前 5 对每对单次，探针那一次就是 0441-0442 的结果")}
PHASES = ("probe", "pairs")
NEED_FLAGS = ("--init-field", "--start-level", "--checkpoint-seconds", "--sigma-domain", pl.SUPPORT_SHARED, "--support-threshold")
DRIVER_SCRIPTS = (*x2.DRIVER_SCRIPTS, "experiments/gate_a/run_warm.py")
# The checkpoints to expect (--dry-run): every S seconds of native time; a planning figure, the runs write what they reach
PLANNED_CHECKPOINTS = 3


class WarmSystem(x2.X2System):
    def drivers_running(self):
        return pl.driver_processes(Path("/proc"), os.getpid(), DRIVER_SCRIPTS)


def namespace(o, spec):
    """X2's option namespace of one arm, with the phase's wall seconds (o.wall_seconds, set by parse_args), the seed root and the checkpoint rule."""
    space = x2.x2_namespace(o, spec)
    space.seed_root, space.checkpoint_seconds = o.seed_root, o.checkpoint_seconds
    return space


class WarmArm(x2.X2Arm):
    """run_x2.X2Arm (X2d's model and path arguments, the gate with the 1-minute load) with the unit prefix tag-warm, the seed and the checkpoints."""

    def __init__(self, opts, system, arm, spec):
        super().__init__(opts, system, arm, spec)
        self.prefix = "tag-warmdry" if self.dry else "tag-warm"
        self.seed_root = Path(opts.seed_root)
        self.checkpoint_seconds = float(opts.checkpoint_seconds if opts.checkpoint_seconds is not None else spec["checkpoint_seconds"])

    def seed_of(self, pair):
        return self.seed_root/pair/TASK/"final_native.npz"

    def configuration(self):
        return dict(super().configuration(), start_level=START_LEVEL, checkpoint_seconds=self.checkpoint_seconds, seed_arm=SEED_ARM,
                    seed_root=str(self.seed_root))

    def run_args(self, pair, eta_file):
        return [*super().run_args(pair, eta_file), "--init-field", str(self.seed_of(pair)), "--start-level", str(START_LEVEL),
                "--checkpoint-seconds", eta.number(self.checkpoint_seconds)]

    def expected_args(self, pair, eta_file):
        expected = dict(super().expected_args(pair, eta_file), start_level=START_LEVEL, init_field_oracle_ok=False,
                        checkpoint_seconds=self.checkpoint_seconds)
        if pair is not None:
            expected["init_field"] = str(self.seed_of(pair))
        return expected

    def launch(self, *, command, **kwargs):
        if not self.dry and any(FORBIDDEN_WORD in str(c) for c in command):
            raise Stop(f"the command of {kwargs.get('name')} contains the word {FORBIDDEN_WORD!r} (gate 12 incident); nothing started")
        return super().launch(command=command, **kwargs)


class WarmQueue(x2.X2Queue):
    """run_x2.X2Queue with one arm, the two phases, the seeds, X2d's eta file, the evaluations of the entry and the checkpoints, the probe's gate.
    Inherited as they are: share_code, precheck, record_failure, hard_constraint_1, lock, main (rb.Queue, pl.PilotQueue, x2.X2Queue)."""

    def __init__(self, opts, system=None):              # noqa: super().__init__ builds X2's arms; not called
        self.o = opts
        self.sys = WarmSystem() if system is None else system
        self.root = Path(opts.root)
        self.dry = opts.dry_run
        self.phase = opts.phase
        self.arm_ids = [ARM]
        self.base = WarmArm(namespace(opts, ARMS[ARM]), self.sys, "base", dict(ARMS[ARM], label="base"))
        self.arms = {ARM: WarmArm(namespace(opts, ARMS[ARM]), self.sys, ARM, ARMS[ARM])}
        self.base.need_flags = NEED_FLAGS
        self.deadline = parse_deadline(opts.no_start_after)
        for arm in (self.base, *self.arms.values()):
            arm.deadline = self.deadline
        self.pairs, self.five, self.eta_file, self.eta, self.eta_record = [], [], None, None, None
        self.pool_record, self.probe_record = None, None
        self.arm_eta, self.seeds = {}, {}
        self.in_row = 0

    # ---- start ---------------------------------------------------------------------------------------------------
    def root_checks(self):
        root = self.root.resolve()
        others = [Path(self.o.seed_root).resolve(), Path(self.o.eta_json).resolve().parent, Path(self.o.x2_root).resolve()]
        for bad in PROTECTED+tuple(others):
            if root == bad or bad in root.parents or root in bad.parents:
                return f"--root {root} overlaps {bad} (results of other work; read only)"
        if self.dry:
            if root == RESULTS or RESULTS in root.parents:
                return f"--synthetic: --root must lie outside {RESULTS}"
            for name in ("seed_root", "eta_json", "x2_root"):
                path = Path(getattr(self.o, name)).resolve()
                if path == RESULTS or RESULTS in path.parents:
                    return f"--synthetic: --{name.replace('_', '-')} {path} lies under {RESULTS} (synthetic inputs only)"
        return ""

    def start_checks(self):
        super().start_checks()                 # PilotQueue: the first five pairs of selection.json, subjects, a clean tree
        self.five = list(self.pairs)
        self.check_five()

    def check_five(self):
        if not self.dry and tuple(short_name(p) for p in self.five) != FIRST_FIVE:
            raise Stop(f"the first five pairs are {[short_name(p) for p in self.five]}, expected {list(FIRST_FIVE)} (the plan)")

    def probe_pair(self):
        """The probe's pair among the first five (synthetic inputs: the last one plays it)."""
        return self.five[-1] if self.dry else next(p for p in self.five if short_name(p) == PROBE_SHORT)

    def phase_pairs(self):
        """The pairs this phase launches: the probe's pair, or the others; a pair belongs to one phase, so no phase repeats another's run."""
        probe = self.probe_pair()
        return [probe] if self.phase == "probe" else [p for p in self.five if p != probe]

    def summary_pairs(self):
        """The pairs the phase's summary covers: the probe's pair, or all five (the probe's run is that pair's result)."""
        return [self.probe_pair()] if self.phase == "probe" else list(self.five)

    # ---- seeds ---------------------------------------------------------------------------------------------------
    def seed_record(self, pair):
        """What the source's small records say about a seed (the full binding happens in run_pair, C12-4a item 2); a mismatch STOPs before any
        unit.  The manifest goes through init_field's filter (no key naming labels); the source's pair.npz is not opened; `warped` is not read."""
        path = self.base.seed_of(pair)
        task_dir, run_dir = path.parent, path.parent.parent
        manifest_path = run_dir/"inputs"/pair/"manifest.json"
        try:                                                     # review WARM_tools M5: an unreadable record is a STOP, not a traceback
            final, protocol = read_json(task_dir/"final_native.json"), read_json(run_dir/"protocol.json")
            manifest = read_seed_manifest(manifest_path) if manifest_path.is_file() else None
        except ValueError as exc:                                # InitFieldRejected and JSONDecodeError
            raise Stop(f"seed of {pair}: {exc}") from None
        if not path.is_file() or not all(isinstance(d, dict) for d in (final, protocol, manifest)):
            raise Stop(f"seed of {pair}: {path}, final_native.json, protocol.json or inputs/{pair}/manifest.json missing or not a JSON object")
        file_sha = sha256_of(path)
        cert = final.get("certificate") if isinstance(final.get("certificate"), dict) else {}
        pre = manifest.get("preprocessing_d0") if isinstance(manifest.get("preprocessing_d0"), dict) else {}
        problems = []
        if final.get("sha256") != file_sha:
            problems.append(f"final_native.json sha256 {final.get('sha256')} is not the file's {file_sha}")
        if final.get("role") != SEED_ROLE:
            problems.append(f"role {final.get('role')!r}, expected {SEED_ROLE!r}")
        if cert.get("status") != "certified_feasible":
            problems.append(f"certificate {cert.get('status')!r}")
        if protocol.get("stage") != SEED_STAGE:
            problems.append(f"stage {protocol.get('stage')!r}, expected {SEED_STAGE!r}")
        if pre.get("arm") != SEED_ARM or pre.get("oracle") is not False:
            problems.append(f"preprocessing_d0 arm {pre.get('arm')!r}, oracle {pre.get('oracle')!r} (expected {SEED_ARM!r}, false)")
        if problems:
            raise Stop(f"seed of {pair} ({path}): {'; '.join(problems)}")
        return dict(path=str(path), file_sha256=file_sha, state_hash=cert.get("state_hash"), grid_id=cert.get("grid_id"),
                    min_jd_lower_bound=cert.get("min_jd_lower_bound"), stage=protocol.get("stage"), arm=pre.get("arm"), oracle=pre.get("oracle"),
                    source_pair_sha256=pre.get("source_pair_sha256"), source_seconds=(protocol.get("task") or {}).get("wall_seconds"))

    def seeds_document(self):
        return dict(schema=SCHEMA+".seeds", seed_arm=SEED_ARM, seed_root=str(self.base.seed_root), task=TASK,
                    seeds={pair: self.seed_record(pair) for pair in self.five})

    def seeds_phase(self):
        document = self.seeds_document()
        self.base.compare_or_write(self.root/"seeds.json", document)
        self.seeds = document["seeds"]
        self.log("seeds: " + ", ".join(f"{short_name(p)} {r['file_sha256'][:12]} (bound {r['min_jd_lower_bound']:.3g})" for p, r in self.seeds.items())
                 + f"; record {self.root/'seeds.json'}")

    # ---- eta -----------------------------------------------------------------------------------------------------
    def eta_value(self):
        """X2d's eta record, checked: (alpha, nu, tau); nothing is selected here and the file is not rewritten."""
        path = Path(self.o.eta_json)
        doc = read_json(path)
        if doc is None:
            raise Stop(f"{path} missing (X2d's eta record)")
        if not self.dry and sha256_of(path) != ETA_SHA256:
            raise Stop(f"{path}: sha256 {sha256_of(path)} is not the pinned {ETA_SHA256} (the file X2d read)")
        want = dict(x2.arm_eta(_D), channel_source=_D["channel"], sigma_domain=_D["sigma"], support_threshold=_D["theta"])
        differ = {k: (doc.get(k), v) for k, v in want.items() if doc.get(k) != v}
        if differ:
            raise Stop(f"{path} differs from X2d's objective (got, expected): {differ}")
        return path, dict(alpha=doc["alpha"], nu=doc["nu"], tau=doc["tau"])

    def eta_phase(self):
        path, value = self.eta_value()
        for aid in self.arms:
            self.arm_eta[aid] = (path, value)
        self.log(f"eta: {value}; read from {path} (X2d's record, a diagnostic point, not an eta the contract selected)")

    # ---- batch ---------------------------------------------------------------------------------------------------
    def write_batch(self):
        arms = {aid: dict(label=arm.spec["label"], note=arm.spec["note"], channel_source=arm.spec["channel"], **x2.arm_domain(arm),
                          eta=self.arm_eta[aid][1], eta_file=str(self.arm_eta[aid][0]), configuration=arm.configuration(),
                          expected_args=arm.expected_args(None, self.arm_eta[aid][0]), eval_arm=arm.spec["label"],
                          unit_prefix=f"{arm.prefix}-{aid}-", run_id=f"gawarm_{aid}_<short>", pairs=list(self.pairs))
                for aid, arm in self.arms.items()}
        document = dict(schema=SCHEMA+".batch", phase=self.phase, stage=STAGE, pairs=self.pairs, summary_pairs=self.summary_pairs(), probe_pair=self.probe_pair(),
                        arm_order=self.arm_ids,
                        order="selection.json order; one arm", arms=arms, method=METHOD, wall_seconds=self.o.wall_seconds,
                        checkpoint_seconds=self.arms[self.arm_ids[0]].checkpoint_seconds, code_sha=self.base.code_sha, pools=self.pool_record,
                        seed_root=str(self.base.seed_root), seeds_file=str(self.root/"seeds.json"), x2_root=str(self.o.x2_root),
                        probe_gate=self.probe_record, dry_run=self.dry)
        self.base.compare_or_write(self.root/f"batch_{self.phase}.json", document)
        self.log(f"batch: {self.root/f'batch_{self.phase}.json'}; deadline "
                 + (f"{self.deadline.isoformat()} (no run unit starts at or after it)" if self.deadline else "none"))

    # ---- the probe's gate ------------------------------------------------------------------------------------------
    def probe_gate(self):
        """The phase pairs needs a probe verdict "go" or BZ's words (--bz-go); the probe's code tree is the one in use (prepare_code keeps it)."""
        if self.phase != "pairs":
            return
        doc = read_json(self.root/sw.summary_name("probe", "json"))
        batch = read_json(self.root/"batch_probe.json")
        if doc is None or batch is None:
            raise Stop(f"{self.root}: no probe summary or no batch_probe.json; the five pairs start after the probe")
        if batch.get("code_sha") != self.base.code_sha:
            raise Stop(f"the probe ran on {batch.get('code_sha')}, this tree is {self.base.code_sha}")
        verdict = doc.get("probe") or {}
        record = dict(verdict=verdict.get("label"), go=verdict.get("go"), accepted=verdict.get("accepted"), rms_all_voxels=verdict.get("rms_all_voxels"),
                      rms_brain_voxels=verdict.get("rms_brain_voxels"), bz_go=self.o.bz_go)
        if verdict.get("go") is not True and not self.o.bz_go:
            raise Stop(f"the probe's verdict is {verdict.get('label')!r} (accepted {verdict.get('accepted')}, RMS over all nodes {verdict.get('rms_all_voxels')} voxels): "
                       "stop and hand to BZ; --bz-go \"<BZ's words and time>\" records BZ's decision to go on")
        self.probe_record = record
        self.log(f"probe gate: {verdict.get('label')}; go {verdict.get('go')}; bz-go {self.o.bz_go!r}")

    # ---- evaluation ----------------------------------------------------------------------------------------------
    def eval_unit(self, arm, *, name, unit, target, ev, rid, label, pair=None):
        """One bayesreg.eval.cli unit on a run or (with `pair`) a disp.npz directory; 'ok' or 'failed' (kept, not rerun); hard constraint 1 STOPs."""
        earlier = sorted(glob.glob(str(ev/"eval_*.log")))+sorted(glob.glob(str(ev/"long.csv")))+arm.earlier("eval", name)
        if earlier:
            if not (ev/"eval_failed.json").exists():
                write_once(ev/"eval_failed.json", dict(reason="earlier evaluation attempt without done.json; kept, not rerun", files=earlier,
                                                        recorded_utc=utc()))
            self.log(f"evaluation of {name}: earlier attempt without done.json ({' '.join(earlier)}); recorded as failed")
            return "failed"
        ev.mkdir(parents=True, exist_ok=True)
        s = eta.stamp()
        log, exit_file = ev/f"eval_{s}.log", ev/f"eval_exit_{s}.txt"
        rc, record = arm.launch(kind="eval", name=name, unit=unit, command=self.eval_command(arm, target, ev, rid, label, pair),
                                memmax_text=self.o.eval_memory_max, memmax_mib=arm.eval_mib, gpu=False, nice=19 if self.dry else 10,
                                runtime=eta.EVAL_MAX_SEC, quota=arm.quota, wrapped=(log, exit_file))
        metrics = sorted(glob.glob(str(ev/"*"/"metrics.json")))
        if rc != 0 or len(metrics) != 1:
            write_once(ev/"eval_failed.json", dict(exit=rc, log=str(log), launch_record=str(record), metrics=metrics, recorded_utc=utc()))
            self.log(f"FAILED evaluation of {name}: exit {rc}, {len(metrics)} metrics.json (see {log}; kept, not rerun)")
            return "failed"
        write_once(ev/"done.json", dict(done_utc=utc(), run=str(target), run_id=rid, metrics=metrics, log=str(log), code_sha=arm.code_sha))
        doc = self.hard_constraint_1(name, ev)
        cert = doc.get("certificate") or {}
        self.log(f"evaluated {name}: field_role {doc.get('field_role')}; certificate {cert.get('status')} {cert.get('min_jd_lower_bound')}")
        return "ok"

    @staticmethod
    def eval_command(arm, target, ev, rid, label, pair=None):
        command = [PY, "-B", "-m", "bayesreg.eval.cli", target, "--pairs-stage", STAGE, "--out", "metrics.json", "--long", ev/"long.csv", "--summary",
                   ev/"summary.json", "--out-dir", ev, "--method", METHOD, "--arm", label, "--run-id", rid, "--data-root", arm.data_root,
                   "--allowlist", arm.allowlist]
        return command if pair is None else [*command, "--pair", pair]

    def evaluate(self, arm, name, short, run, ev):
        """The final field: run_x2.X2Queue.evaluate with WARM's run id, method and arm label."""
        return self.eval_unit(arm, name=name, unit=f"{arm.prefix}-eval-{name}", target=run, ev=ev, rid=f"gawarm_{arm.arm}_{short}", label=arm.spec["label"])

    def evaluate_fields(self, arm, pair, short, run):
        """The entry and each checkpoint of a finished run (C12-4a item 8), in that order; 'ok' or 'failed'.  A restart evaluates what is missing."""
        name = f"{arm.arm}-{short}"
        profile = (read_json(run/"hyper.json") or {}).get("profile") or {}
        directories = {p.name: p for p in (run/"checkpoints").glob("*") if (p/"disp.npz").is_file()}
        expected = ["entry"]+[f"ckpt_{c['k']:03d}" for c in profile.get("checkpoints") or []]
        missing = [f for f in expected if f not in directories]
        if missing and (run/"status.json").is_file() and (read_json(run/"status.json") or {}).get("native"):
            self.log(f"{name}: no disp.npz for {missing} (profile errors: {profile.get('errors')})")
        if profile.get("errors"):
            self.log(f"{name}: records with errors (the solve did not notice): {profile['errors']}")
        outcome = "ok"
        for field in sorted(directories, key=lambda f: (f != "entry", f)):
            tag = field.replace("_", "")
            field_name = f"{name}-{tag}"
            ev = self.root/"eval_fields"/arm.arm/short/field
            if (ev/"done.json").is_file():
                self.hard_constraint_1(field_name, ev)
                continue
            if (ev/"eval_failed.json").is_file():
                outcome = "failed"
                continue
            got = self.eval_unit(arm, name=field_name, unit=f"{arm.prefix}-eval-{field_name}", target=directories[field], ev=ev,
                                 rid=f"gawarm_{arm.arm}_{short}_{field}", label=f"{arm.spec['label']}-{field}", pair=pair)
            outcome = "failed" if got == "failed" else outcome
        return outcome

    def check_run(self, arm, pair, run, name):
        status = read_json(run/"status.json") or {}
        if status.get("status") == INIT_FIELD_REJECTED:
            raise Stop(f"{name}: the seed was refused ({status.get('reason')}) although the precheck passed; kept, the lead decides")
        rb.Queue.check_run(self, arm, pair, run, name)
        config = read_json(run/"config.json") or {}
        if not config:
            return
        seed, solver, want = config.get("init_field") or {}, config.get("solver") or {}, self.seeds[pair]
        problems = [f"seed {key} {seed.get(key)!r} (seeds.json {want[key]!r})" for key in ("file_sha256", "state_hash") if seed.get(key) != want[key]]
        if seed.get("oracle") is not False or seed.get("arm") != SEED_ARM:
            problems.append(f"seed arm {seed.get('arm')!r}, oracle {seed.get('oracle')!r}")
        if solver.get("sigma_domain") != _D["sigma"] or solver.get("start_level") != START_LEVEL:
            problems.append(f"solver sigma_domain {solver.get('sigma_domain')!r}, start_level {solver.get('start_level')!r}")
        if solver.get("checkpoint") != dict(every_accepted=None, every_seconds=arm.checkpoint_seconds):
            problems.append(f"solver checkpoint {solver.get('checkpoint')}")
        if problems:
            raise Stop(f"{name}: config.json differs from the plan: {'; '.join(problems)}")

    def one(self, arm_name, pair):
        """rb.Queue.one with X2d's eta record, then the entry and the checkpoints."""
        self.eta_file, self.eta = self.arm_eta[arm_name]
        outcome = rb.Queue.one(self, arm_name, pair)
        arm, short = self.arms[arm_name], short_name(pair)
        run = self.root/"runs"/arm_name/short
        if (run/"status.json").is_file() and self.evaluate_fields(arm, pair, short, run) == "failed" and outcome == "ok":
            outcome = "failed"
        return outcome

    # ---- run -----------------------------------------------------------------------------------------------------
    def run(self):
        self.log(f"start: root {self.root}; phase {self.phase}; until {self.o.until}; arm {self.arm_ids[0]}; wall {self.o.wall_seconds:g} s (预算收尾); "
                 f"checkpoint {self.arms[self.arm_ids[0]].checkpoint_seconds:g} s; MemoryMax {self.o.memory_max} (eval {self.o.eval_memory_max}); min GPU free "
                 f"{self.o.min_gpu_free_mib} MiB; deadline {self.deadline.isoformat() if self.deadline else 'none'}; synthetic {int(self.dry)}; self unit "
                 f"{getattr(self.sys, 'self_unit', '') or 'none'}")
        try:
            if self.deadline is not None and self.sys.now() >= self.deadline:
                raise Stop(f"--no-start-after {self.deadline.isoformat()} is already past; nothing read, nothing started")
            self.start_checks()
            self.base.prepare_code()
            self.share_code()
            self.base.write_env()
            self.precheck()
            self.seeds_phase()
            self.pairs = self.phase_pairs()
            if self.o.until == "precheck":
                self.log("until precheck: end")
                return 0
            self.probe_gate()
            self.eta_phase()
            self.write_batch()
            if self.o.until == "eta":
                self.log("until eta: end")
                return 0
            not_run = self.batch()
            if not_run:
                self.log(f"not run (deadline; 未运行, not failures): {', '.join(not_run)}")
            try:
                js, md, state = sw.write(self.root, self.phase)
            except sw.SummaryError as exc:
                raise Stop(f"summary: {exc}") from None
            self.log(f"summary: {js}, {md} ({state})")
            if self.phase == "probe":
                verdict = (read_json(js) or {}).get("probe") or {}
                self.log(f"probe verdict: {verdict.get('label')} (accepted {verdict.get('accepted')}, RMS over all nodes {verdict.get('rms_all_voxels')} voxels); go {verdict.get('go')}"
                         + ("" if verdict.get("go") else "; stop here, BZ decides"))
            self.log("end")
            return 0
        except Stop as exc:
            self.log(f"STOP: {exc}")
            return 8

    # ---- --dry-run: command lines only ---------------------------------------------------------------------------
    def print_commands(self, out=None):
        """Every command line of the phase; the real parser and the expected values are checked; nothing is written, nothing starts."""
        out = sys.stdout if out is None else out
        print(f"# run_warm.py --dry-run: phase {self.phase}, root {self.root}; nothing written, no unit, no solver, no GPU", file=out)
        problem = self.root_checks()
        if problem:
            print(problem, file=sys.stderr)
            return 2
        try:
            selection = read_json(self.base.data_root/STAGE/"selection.json")
            if selection is None:
                raise Stop(f"{self.base.data_root/STAGE}/selection.json missing")
            self.five = [row["pair"] for row in selection.get("pairs") or []][:pl.N_PAIRS]
            self.check_five()
            self.pairs = self.phase_pairs()
            self.seeds = self.seeds_document()["seeds"]
            path, value = self.eta_value()
        except Stop as exc:
            print(f"STOP: {exc}", file=sys.stderr)
            return 8
        code = Path(self.o.code) if self.o.code else self.root/"code"
        from bayesreg.cli import run_pair
        aid = self.arm_ids[0]
        arm = self.arms[aid]
        arm.code, arm.allowlist = code, self.o.allowlist or str(code/"configs"/"bayesreg"/"dev_subjects.json")
        arm.registry_path = self.o.segmenter_registry            # --synthetic only; the real runs use the code tree's registered registry
        print(f"# eta {value} from {path}; sha256 {sha256_of(path)}", file=out)
        print(f"# code: {'--code '+str(code) if self.o.code else f'{code} (detached worktree of --commit or HEAD, made at the start)'}", file=out)
        print(f"# planning figure for the evaluations: entry plus {PLANNED_CHECKPOINTS} checkpoints; a run writes what it reaches", file=out)
        print(f"# this phase launches {[short_name(p) for p in self.pairs]}; its summary covers {[short_name(p) for p in self.summary_pairs()]}", file=out)
        for pair in self.pairs:
            short = short_name(pair)
            name, run = f"{aid}-{short}", self.root/"runs"/aid/short
            args = arm.run_args(pair, path)
            parsed = {k: v for k, v in vars(run_pair.parse_args(["--out", str(run), *args])).items() if not k.startswith("_")}
            differ = {k: (parsed.get(k), v) for k, v in arm.expected_args(pair, path).items() if parsed.get(k) != v}
            if differ:
                print(f"STOP: {name}: the parser's values differ from the expected ones (got, expected): {differ}", file=sys.stderr)
                return 8
            seed = self.seeds[pair]
            print(f"\n# ---- {name}: seed {seed['file_sha256']} state {seed['state_hash']} (bound {seed['min_jd_lower_bound']}); "
                  "run_pair's parser accepted these arguments; expected values match", file=out)
            command = [PY, "-B", "-m", "bayesreg.cli.run_pair", "--out", run, *args]
            sd = arm.systemd_command(kind="run", unit=f"{arm.prefix}-{name}", command=command, memmax_text=self.o.memory_max, gpu=True, nice=None,
                                     runtime=int(self.o.wall_seconds)+1800, quota=arm.quota)
            print(shlex.join(sd), file=out)
            fields = [("final", None)]+[(f, pair) for f in ("entry", *(f"ckpt_{k:03d}" for k in range(PLANNED_CHECKPOINTS)))]
            for field, field_pair in fields:
                if field_pair is None:
                    target, ev, rid = run, self.root/"eval"/aid/short, f"gawarm_{aid}_{short}"
                    unit, label = f"{arm.prefix}-eval-{name}", arm.spec["label"]
                else:
                    target, ev, rid = run/"checkpoints"/field, self.root/"eval_fields"/aid/short/field, f"gawarm_{aid}_{short}_{field}"
                    unit, label = f"{arm.prefix}-eval-{name}-{field.replace('_', '')}", f"{arm.spec['label']}-{field}"
                sd = arm.systemd_command(kind="eval", unit=unit, command=self.eval_command(arm, target, ev, rid, label, field_pair),
                                         memmax_text=self.o.eval_memory_max, gpu=False, nice=10, runtime=eta.EVAL_MAX_SEC, quota=arm.quota,
                                         wrapped=(ev/"eval_<stamp>.log", ev/"eval_exit_<stamp>.txt"))
                print(shlex.join(sd), file=out)
        return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -B experiments/gate_a/run_warm.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--root", default=str(REAL_ROOT))
    parser.add_argument("--code")
    parser.add_argument("--commit")
    parser.add_argument("--until", choices=("precheck", "eta"), default=None)
    parser.add_argument("--dry-run", dest="print_commands", action="store_true", help="print every command line of the phase; nothing written, no unit, no "
                        "solver, no GPU (the seeds, the eta record and run_pair's parser are checked)")
    parser.add_argument("--bz-go", default=None, help="phase pairs after a probe verdict of stagnation: BZ's words and time (recorded)")
    parser.add_argument("--seed-root", default=str(SEED_ROOT), help="the A1Ph run directories (<root>/<pair>/pg_weak__four__90s/final_native.npz; read only)")
    parser.add_argument("--eta-json", default=str(ETA_JSON), help="X2d's eta record (read only)")
    parser.add_argument("--x2-root", default=str(x2.REAL_ROOT), help="X2's root (side reference: X2d's Dice; read only)")
    parser.add_argument("--no-start-after", help="ISO date-time with a date and a UTC offset, e.g. 2026-11-04T05:00-06:00; no run unit starts at or after it")
    parser.add_argument("--data-root")
    parser.add_argument("--allowlist")
    parser.add_argument("--tissue-root")
    parser.add_argument("--structure-root", default=str(eta.DEFAULT_STRUCTURE_ROOT))
    parser.add_argument("--segmenter-registry", default=None)
    parser.add_argument("--memory-max", default="9G")
    parser.add_argument("--eval-memory-max", default="6G")
    parser.add_argument("--min-gpu-free-mib", type=int, default=22528)
    parser.add_argument("--home-floor-gb", type=int, default=100)
    parser.add_argument("--wait-minutes", type=int, default=30, help="how long the gate polls before a STOP")
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--synthetic", dest="dry_run", action="store_true", help="synthetic inputs only (the tests' fake machine)")
    parser.add_argument("--wall-seconds", type=float, default=None, help="--synthetic only (the plan fixes 900 s)")
    parser.add_argument("--checkpoint-seconds", type=float, default=None, help="--synthetic only (the plan fixes 225 s)")
    args = parser.parse_args(argv)
    spec = ARMS[ARM]
    if not args.dry_run and (args.wall_seconds is not None or args.checkpoint_seconds is not None):
        parser.error("--wall-seconds and --checkpoint-seconds: --synthetic only (the plan fixes them)")
    if not args.dry_run and args.segmenter_registry:
        parser.error("--segmenter-registry: --synthetic only")
    args.wall_seconds = spec["wall_seconds"] if args.wall_seconds is None else args.wall_seconds
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
    if args.bz_go is not None and (args.phase != "pairs" or not args.bz_go.strip()):
        parser.error("--bz-go: phase pairs only, with BZ's words and time")
    if args.no_start_after is not None:
        try:
            moment = datetime.fromisoformat(args.no_start_after)
        except ValueError:
            moment = None
        if moment is None or moment.tzinfo is None or re.fullmatch(r"[0-9]{1,2}:[0-9]{2}", args.no_start_after):
            parser.error(f"--no-start-after {args.no_start_after!r}: an ISO date-time with a date and a UTC offset")
    return args


def main(argv=None, system=None):
    args = parse_args(argv)
    queue = WarmQueue(args, system)
    return queue.print_commands() if args.print_commands else queue.main()


if __name__ == "__main__":
    sys.exit(main())
