"""Gate A, X2 (runs/20261102-gate-a/plan.md): bayesreg from u = 0 at lower absolute weights, tissue and structure channels.
Development set only (stage_L2R_val_v1, the first five pairs of selection.json); every number it yields is a development-set
single observation; n = 5 gives direction only.  No new solver code: run_pair with --eta-json and --channel-source.

    python -B experiments/gate_a/run_x2.py [--root DIR] [--code DIR] [--commit SHA] [--arms a,b,c,d] [--until precheck|eta]
        [--no-start-after ISO] [--wall-seconds 900] [--data-root DIR] [--allowlist PATH] [--tissue-root DIR]
        [--structure-root DIR] [--segmenter-registry PATH (dry run only)] [--memory-max 9G] [--eval-memory-max 6G]
        [--min-gpu-free-mib 22528] [--home-floor-gb 100] [--wait-minutes 30] [--max-consecutive-failures 3]
        [--a0-runs-root DIR] [--pilot-root DIR] [--dry-run]

Arms (the plan; written before any run; nothing here chooses eta, and these pairs are not used to tune it):
  a  --channel-source tissue,    nu 3.87, tau (1, .5, .5, .5), alpha_nug(tau) = .01    (the G3 weights, on the current defaults)
  b  --channel-source tissue,    nu 1.37, tau (1, 1, 1, 1),    alpha_nug(tau) = .01
  c  --channel-source structure, nu 1.37, tau (1, 1, 1, 1),    alpha_nug(tau) = .01
  d  arm c with --sigma-domain support_shared --support-threshold 0.05 (C04-6, D50; passed as for the pilot's P4; plan revision 2, BZ
     11-02 ~10:33 pm CST).  A diagnostic point, not an eta the contract selected: C04-6 does not carry an eta selected under brain to
     support_shared, so a good result calls for a new probe and sweep.
Everything else is the M2V1 dev arm (experiments/m2v1/run_batch.py; its machinery is used as it is through the pilot's
subclasses, experiments/pilot/run_pilot.py, which are not modified): --logvol-points gauss27+closed27 --logvol-closed27-share 1
--closed27-weights simpson, --metric-jd-scaling affine_v1 --jd-cell-estimate certificate, --backend torch64 --device cuda,
--outer a --rounds 1,1,1,1 --cpu-threads 4, --wall-seconds 900; structure: the registered unet35_v1_ens, --structure-infer never.
Order: pair-major (selection.json order), the arm order rotating with the pair index (pair i runs the arms from position
i mod n; the four arms over five pairs: first position a 2, b 1, c 1, d 1), so that time of day does not follow an arm; one unit at a
time.
Per run: unit <prefix>-<arm>-<short> (MemoryMax 9G, MemorySwapMax 4G, CPUQuota 600%, RuntimeMaxSec wall + 1800) ->
runs/<arm>/<short>/; evaluation by bayesreg.eval.cli only (unit <prefix>-eval-<arm>-<short>) -> eval/<arm>/<short>/
gax2_<arm>_<short>/metrics.json (method bayesreg_gate_a_x2, arm X2a / X2b / X2c / X2d).  After a run its config.json args must equal
the arm's expected values (sigma_domain and support_threshold too: brain and null for a, b, c; support_shared and 0.05 for d) and its
eta the arm's record, else STOP; an exported field whose evaluator certificate did not pass:
STOP (hard constraint 1), as M2V1.  Resume rules, failures (recorded, not rerun; --max-consecutive-failures in a row STOP), the
gate (hold file $ROOT/hold; no other tag-* / seq-vp-* unit; no driver process of M2V1, the eta sweep, the pilot or another
X2; GPU free; MemAvailable; /home), the precheck (registry, structure predictions, tissue layer, allowlist, sealed lists,
training list) and the driver lock: as M2V1 and the pilot; and, only here, before every GPU run unit a 1-minute load < 1.5 (plan
revision 1), polled for --wait-minutes (default 30) and then STOP.
eta records: eta/eta_<arm>.json, written once, read by every run of the arm through --eta-json (run_pair reads alpha, nu, tau only); a
restart recomputes and compares.  They and batch.json name each arm's sigma domain and the role of its eta (arm d: 诊断点).
End: experiments/gate_a/summarize_x2.py writes summary.json / summary.md (written once; a different recomputation goes to
summary_<UTC>.*).
Code version: --commit SHA (default the repository HEAD at start, tracked files clean) in a detached worktree $ROOT/code.  Log:
$ROOT/queue.log (UTC).
--dry-run: synthetic inputs only (experiments/m2v1/dry_inputs.py); unit prefix tag-gax2dry, device cpu, one thread, Nice 19; every
input and the root outside /home/rmcfz/seq_vp_tag_results.
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
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO/rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pl = _load("_gax2_run_pilot", "experiments/pilot/run_pilot.py")
sx2 = _load("_gax2_summarize", "experiments/gate_a/summarize_x2.py")
rb, eta = pl.rb, pl.eta
Stop, Deadline = pl.Stop, pl.Deadline
read_json, write_once, utc, short_name = pl.read_json, pl.write_once, pl.utc, pl.short_name
parse_deadline, sha256_of = pl.parse_deadline, pl.sha256_of

PY = pl.PY
STAGE = pl.STAGE
RESULTS = pl.RESULTS
REAL_ROOT = RESULTS/"gate_a_x2"
PROTECTED = pl.PROTECTED+(pl.REAL_ROOT, RESULTS/"gate_a_x1", RESULTS/"d0_v1_a1p")+tuple(RESULTS/n for n in (     # GATEA_rev2 review N6
    "frozen_eb808be", "fb_audit_v1", "gap_decomp_v1", "swl_gate12", "swl_gate12_v2_t4", "kkt_v1", "lmi_v1", "sig_v1", "jdt_v1", "jdti_v1",
    "sg_l1_gate", "reviews", "logs", "tests"))
METHOD = "bayesreg_gate_a_x2"
SCHEMA = "bayesreg.gate_a_x2.v1"
BASE_CELL = pl.BASE_CELL
TISSUE_TAU = (1., .5, .5, .5)
FLAT_TAU = (1., 1., 1., 1.)
ARMS = dict(
    a=dict(label="X2a", channel="tissue", nu=3.87, tau=TISSUE_TAU, caps=None, cell=BASE_CELL,
           note="tissue、ν 3.87、τ (1,.5,.5,.5)：G3 的权重，用现行默认（A′、certificate 单元估计、900 s）"),
    b=dict(label="X2b", channel="tissue", nu=1.37, tau=FLAT_TAU, caps=None, cell=BASE_CELL,
           note="tissue、ν 1.37、τ (1,1,1,1)"),
    c=dict(label="X2c", channel="structure", nu=1.37, tau=FLAT_TAU, caps=None, cell=BASE_CELL,
           note="structure（14 通道）、ν 1.37、τ (1,1,1,1)"),
    d=dict(label="X2d", channel="structure", nu=1.37, tau=FLAT_TAU, caps=None, cell=BASE_CELL, sigma=pl.SUPPORT_SHARED,
           theta=pl.SUPPORT_THETA, eta_role=sx2.DIAGNOSTIC_NOTE,
           note="structure（14 通道）、ν 1.37、τ (1,1,1,1)、support_shared（θ .05）：诊断点，不是合同选出的 η（C04-6）"))
ROLE_BRAIN = "brain 下的固定值，诊断用；不是合同选出的 η"
DEFAULT_ARMS = "a,b,c,d"
DRIVER_SCRIPTS = (*pl.DRIVER_SCRIPTS, "experiments/gate_a/run_x2.py")
A0_RUNS = RESULTS/"d0_v1"/"runs"/"A0"
PILOT_ROOT = pl.REAL_ROOT
LOAD_LIMIT = 1.5


def arm_eta(spec):
    """(alpha, nu, tau) of an arm: alpha = alpha_nug(tau) (C05-6, P2)."""
    return dict(alpha=eta.alpha_nug(spec["tau"]), nu=spec["nu"], tau=list(spec["tau"]))


def arm_domain(arm):
    """The sigma domain of an arm as batch.json and the eta record state it: the values the arm's run_pair arguments carry (null theta under
    brain), and the role of its eta."""
    return dict(sigma_domain=arm.sigma_domain, support_threshold=arm.support_threshold, eta_role=arm.spec.get("eta_role", ROLE_BRAIN))


class X2System(pl.PilotSystem):
    def drivers_running(self):
        return pl.driver_processes(Path("/proc"), os.getpid(), DRIVER_SCRIPTS)


def x2_namespace(o, spec):
    """The eta.Sweep option namespace of one arm: the pilot's (sigma_domain and support_threshold from the arm, as for P4) with the arm's
    channel source."""
    space = pl.pilot_namespace(o, spec)
    space.channel_source = spec["channel"]
    return space


class X2Arm(pl.PilotArm):
    """run_pilot.PilotArm with the unit prefix tag-gax2 / tag-gax2dry (the channel source comes through the namespace; no caps) and, for
    the GPU run units only, a quiet machine: 1-minute load < 1.5 (plan revision 1, review I8: the budget is wall clock, and the rest of the
    gate does not see an ordinary CPU job; m2v1 and the pilot are not changed)."""

    def __init__(self, opts, system, arm, spec):
        super().__init__(opts, system, arm, spec)
        self.prefix = "tag-gax2dry" if self.dry else "tag-gax2"

    def busy_reason(self, memmax_mib, gpu):
        reason = super().busy_reason(memmax_mib, gpu)
        if reason or not gpu:
            return reason
        load = float(self.sys.loadavg().split()[0])
        if load >= LOAD_LIMIT:
            return f"1-minute load {load:.2f} >= {LOAD_LIMIT} (a timed run; plan revision 1)"
        return ""


class X2Queue(pl.PilotQueue):
    """run_pilot.PilotQueue without M2V1 and P0: one eta record per arm, the rotating order, X2's evaluation names and summary.
    Inherited as they are: start_checks (the first five pairs), share_code, precheck, record_failure, check_run, hard_constraint_1,
    lock, main."""

    def __init__(self, opts, system=None):              # noqa: super().__init__ builds the pilot's arms; not called
        self.o = opts
        self.sys = X2System() if system is None else system
        self.root = Path(opts.root)
        self.dry = opts.dry_run
        self.arm_ids = [a.strip() for a in opts.arms.split(",")]
        # The base arm runs nothing: it prepares the code, the environment record, the registry and batch.json.  It has arm c's namespace in
        # every case, so its sigma domain (brain) reaches no arm: each arm below has its own namespace and takes only the code, the allowlist
        # and the registry from it (share_code, precheck).  It alone checks the code tree, hence the switches of arm d here (as PilotQueue).
        self.base = X2Arm(x2_namespace(opts, dict(ARMS["c"])), self.sys, "base", dict(ARMS["c"], label="base"))
        self.arms = {a: X2Arm(x2_namespace(opts, ARMS[a]), self.sys, a, ARMS[a]) for a in self.arm_ids}
        if any(ARMS[a].get("sigma") for a in self.arm_ids):
            self.base.need_flags = ("--sigma-domain", pl.SUPPORT_SHARED, "--support-threshold")
        self.deadline = parse_deadline(opts.no_start_after)
        for arm in (self.base, *self.arms.values()):
            arm.deadline = self.deadline
        self.pairs, self.eta_file, self.eta, self.eta_record = [], None, None, None
        self.pool_record = None
        self.arm_eta = {}
        self.in_row = 0

    def root_checks(self):
        root = self.root.resolve()
        for bad in PROTECTED:
            if root == bad or bad in root.parents or root in bad.parents:
                return f"--root {root} overlaps {bad} (results of other work; read only)"
        if self.dry and (root == RESULTS or RESULTS in root.parents):
            return f"--dry-run: --root must lie outside {RESULTS}"
        return ""

    def eta_phase(self):
        """eta/eta_<arm>.json, once: the plan's fixed values, nothing selected from any run."""
        (self.root/"eta").mkdir(exist_ok=True)
        for aid, arm in self.arms.items():
            value = arm_eta(arm.spec)
            record = dict(schema=SCHEMA+".eta", **value, channel_source=arm.spec["channel"], **arm_domain(arm),
                          use="runs/20261102-gate-a/plan.md X2 的固定值，诊断用；不是在这些配对上选的，也不是选 η",
                          alpha_rule="alpha_nug(tau) = 1e-2 min_k 1/tau_k^2 (C05-6, P2)", code_sha=arm.code_sha)
            path = self.root/"eta"/f"eta_{aid}.json"
            arm.compare_or_write(path, record)
            self.arm_eta[aid] = (path, value)
            self.log(f"eta {aid}: alpha {value['alpha']:g}, nu {value['nu']:g}, tau {value['tau']}, channel source "
                     f"{arm.spec['channel']}, sigma domain {arm.sigma_domain}; record {path}")

    def write_batch(self):
        arms = {aid: dict(label=arm.spec["label"], note=arm.spec["note"], channel_source=arm.spec["channel"], **arm_domain(arm),
                          eta=self.arm_eta[aid][1], eta_file=str(self.arm_eta[aid][0]), configuration=arm.configuration(),
                          expected_args=arm.expected_args(None, self.arm_eta[aid][0]), eval_arm=arm.spec["label"],
                          unit_prefix=f"{arm.prefix}-{aid}-", run_id=f"gax2_{aid}_<short>", pairs=list(self.pairs))
                for aid, arm in self.arms.items()}
        document = dict(schema=SCHEMA+".batch", stage=STAGE, pairs=self.pairs, arm_order=self.arm_ids,
                        order="pair-major; pair i runs the arms of arm_order from position i mod n", arms=arms, method=METHOD,
                        wall_seconds=self.o.wall_seconds, code_sha=self.base.code_sha, pools=self.pool_record,
                        a0_runs_root=str(self.o.a0_runs_root), pilot_root=str(self.o.pilot_root), dry_run=self.dry)
        self.base.compare_or_write(self.root/"batch.json", document)
        self.log(f"batch: {self.root/'batch.json'}; deadline "
                 + (f"{self.deadline.isoformat()} (no run unit starts at or after it)" if self.deadline else "none"))

    def one(self, arm_name, pair):
        """rb.Queue.one with this arm's eta record (run arguments, config.json check)."""
        self.eta_file, self.eta = self.arm_eta[arm_name]
        return rb.Queue.one(self, arm_name, pair)

    def evaluate(self, arm, name, short, run, ev):
        """run_pilot.PilotQueue.evaluate with X2's run id (gax2_<arm>_<short>), method and arm label."""
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
        rid = f"gax2_{arm.arm}_{short}"
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

    def order_of(self, index):
        """The arms of pair `index`: arm_order from position index mod n."""
        k = index % len(self.arm_ids)
        return self.arm_ids[k:]+self.arm_ids[:k]

    def batch(self):
        not_run = []
        for index, pair in enumerate(self.pairs):
            for aid in self.order_of(index):
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

    def run(self):
        self.log(f"start: root {self.root}; until {self.o.until}; arms {self.arm_ids} "
                 f"({ {a: arm_eta(ARMS[a]) for a in self.arm_ids} }); wall {self.o.wall_seconds:g} s (预算收尾); MemoryMax "
                 f"{self.o.memory_max} (eval {self.o.eval_memory_max}); min GPU free {self.o.min_gpu_free_mib} MiB; deadline "
                 f"{self.deadline.isoformat() if self.deadline else 'none'}; dry-run {int(self.dry)}; self unit "
                 f"{getattr(self.sys, 'self_unit', '') or 'none'}")
        try:
            if self.deadline is not None and self.sys.now() >= self.deadline:
                raise Stop(f"--no-start-after {self.deadline.isoformat()} is already past; nothing read, nothing started")
            self.start_checks()
            self.base.prepare_code()
            self.share_code()
            self.base.write_env()
            self.precheck()
            if self.o.until == "precheck":
                self.log("until precheck: end")
                return 0
            self.eta_phase()
            self.write_batch()
            if self.o.until == "eta":
                self.log("until eta: end")
                return 0
            not_run = self.batch()
            if not_run:
                self.log(f"not run (deadline; 未运行, not failures): {', '.join(not_run)}")
            try:
                js, md, state = sx2.write(self.root)
            except sx2.SummaryError as exc:
                raise Stop(f"summary: {exc}") from None
            self.log(f"summary: {js}, {md} ({state})")
            self.log("end")
            return 0
        except Stop as exc:
            self.log(f"STOP: {exc}")
            return 8


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -B experiments/gate_a/run_x2.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default=str(REAL_ROOT))
    parser.add_argument("--code")
    parser.add_argument("--commit")
    parser.add_argument("--arms", default=DEFAULT_ARMS, help=f"comma-separated arm ids among {sorted(ARMS)}")
    parser.add_argument("--until", choices=("precheck", "eta"), default=None)
    parser.add_argument("--no-start-after", help="ISO date-time with a date and a UTC offset, e.g. 2026-11-03T05:00-06:00; "
                        "no run unit starts at or after it")
    parser.add_argument("--wall-seconds", type=float, default=900.)
    parser.add_argument("--data-root")
    parser.add_argument("--allowlist")
    parser.add_argument("--tissue-root")
    parser.add_argument("--structure-root", default=str(eta.DEFAULT_STRUCTURE_ROOT))
    parser.add_argument("--segmenter-registry", default=None)
    parser.add_argument("--memory-max", default="9G")
    parser.add_argument("--eval-memory-max", default="6G")
    parser.add_argument("--min-gpu-free-mib", type=int, default=22528)
    parser.add_argument("--home-floor-gb", type=int, default=100)
    parser.add_argument("--wait-minutes", type=int, default=30, help="how long the gate polls before a STOP (default 30: the load limit "
                        "normally needs a minute or two after the previous evaluation)")
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--a0-runs-root", default=str(A0_RUNS), help="A0's run directories (reference; metrics.json read only)")
    parser.add_argument("--pilot-root", default=str(PILOT_ROOT), help="the path pilot's root (reference P0n; read only)")
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
            parser.error(f"--no-start-after {args.no_start_after!r}: an ISO date-time with a date and a UTC offset")
    return args


def main(argv=None, system=None):
    return X2Queue(parse_args(argv), system).main()


if __name__ == "__main__":
    sys.exit(main())
