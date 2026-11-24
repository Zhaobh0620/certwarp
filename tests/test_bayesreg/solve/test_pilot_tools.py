"""The PILOT tools (experiments/pilot; runs/20261031-path-pilot/plan.md), on synthetic files only (tmp_path; no data/,
no unit, no real segmentation, no real result file):

- run_pilot.py with a fake machine (units emulated; run_pair's arguments parsed by its real parser): a synthetic M2V1
  root made by experiments/m2v1/run_batch.py itself (dry run, dev arm only, fake units) is the P0 arm; the pilot's arms
  p1, p2, p3 pair-major, one unit each, evaluations; launch records with each arm's switches; config.json checks
  (per-level caps, cell estimate, max_accepted); the eta record (M2V1 agrees); p0.json; batch.json; the summary (paired
  differences against P0, readings, recommendation); a restart launches nothing; P0 mismatches and a running M2V1 STOP;
  --after-m2v1 waits; the gate waits for driver processes; the deadline; --arms changed on a restart STOPs; the model-code
  change needs --accept-code-change; the code-tree flags;
- summarize's readings (preregistered thresholds) on hand-made rows; driver process matching on a fake /proc;
- slow (-m slow): the dry run in process (real run_pair, CPU torch64, and bayesreg.eval.cli on the synthetic structure
  layer; no systemd unit): an M2V1 dry root (dev arm) and the pilot's p1 arm, plus p2 / p3 when run_pair offers
  certificate_bernstein (JDTI).
"""
import contextlib
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
PAIRS = ("OASIS_OAS1_0901_MR1__OASIS_OAS1_0902_MR1", "OASIS_OAS1_0902_MR1__OASIS_OAS1_0903_MR1",
         "OASIS_OAS1_0903_MR1__OASIS_OAS1_0904_MR1")
SHORTS = ("0901-0902", "0902-0903", "0903-0904")
T0 = datetime(2026, 11, 1, 15, 0, tzinfo=timezone.utc)
OLD = "68721d1b89e9af0038f0f33a0a5a198856d9bbe6"        # the sweep's and M2V1's code (model code differs from HEAD)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def head():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


def bernstein_offered():
    from bayesreg.solve.jd_scaling import CELL_ESTIMATES
    return "certificate_bernstein" in CELL_ESTIMATES


@pytest.fixture(scope="module")
def pl():
    return load("pilot_run_pilot", "experiments/pilot/run_pilot.py")


@pytest.fixture(scope="module")
def m2():
    return load("pilot_test_m2v1_run_batch", "experiments/m2v1/run_batch.py")


@pytest.fixture(scope="module")
def dry_mod():
    return load("pilot_test_m2v1_dry_inputs", "experiments/m2v1/dry_inputs.py")


def m2v1_common(dry):
    return ["--dry-run", "--data-root", str(dry/"data"), "--allowlist", str(dry/"dev_subjects.json"), "--tissue-root",
            str(dry/"seg"), "--structure-root", str(dry/"structure"), "--segmenter-registry",
            str(dry/"registry"/"segmenters.json"), "--sweep-root", str(dry/"sweep"), "--d0-long",
            str(dry/"d0"/"long.csv"), "--a4-long", str(dry/"a4"/"long.csv"), "--smoke-root", str(dry/"smoke"),
            "--ref-reeval", str(dry/"ref_reeval"), "--d47-pair", PAIRS[0]]


def pilot_common(dry, m2root, wall="60"):
    return ["--dry-run", "--data-root", str(dry/"data"), "--allowlist", str(dry/"dev_subjects.json"), "--tissue-root",
            str(dry/"seg"), "--structure-root", str(dry/"structure"), "--segmenter-registry",
            str(dry/"registry"/"segmenters.json"), "--sweep-root", str(dry/"sweep"), "--m2v1-root", str(m2root),
            "--wall-seconds", wall]


class FakeSystem:
    """Emulates the units: run_pair (arguments through its real parser; certificate_bernstein mapped to an existing
    choice for parsing while run_pair does not offer it, and recorded as requested) and bayesreg.eval.cli."""

    def __init__(self, *, busy=(), drivers=(), m2v1=(), drop=(), mutate=None, clock=T0, step=60., on_sleep=None,
                 violations=0, no_total=False):
        self.self_unit, self.units, self.busy, self.drivers, self.m2v1 = "", [], list(busy), list(drivers), list(m2v1)
        self.drop, self.mutate, self.clock, self.step, self.on_sleep = set(drop), mutate, clock, step, on_sleep
        self.violations, self.no_total, self.slept = violations, no_total, 0

    def unit_active(self, unit):
        return False

    def other_units(self):
        return list(self.busy)

    def active_units(self):
        return []

    def eta_units(self):
        return []

    def sweep_driver_running(self):
        return []

    def drivers_running(self):
        return list(self.drivers)

    def m2v1_state(self):
        return list(self.m2v1), []

    def now(self):
        return self.clock

    def gpu_free_mib(self):
        return 24000

    def gpu_state(self):
        return "fake gpu"

    def mem_available_mib(self):
        return 13000

    def home_free_gb(self):
        return 500

    def train_state(self):
        return "tag-unet35-train: fake"

    def uptime(self):
        return 1.

    def loadavg(self):
        return "0 0 0"

    def dmesg_since(self, t0, pattern):
        return 0

    def sleep(self, seconds):
        self.slept += 1
        self.clock += timedelta(seconds=seconds)
        if self.on_sleep is not None:
            self.on_sleep(self)

    @staticmethod
    def split(cmd):
        unit = next(c for c in cmd if c.startswith("--unit=")).split("=", 1)[1]
        if "/bin/bash" in cmd:
            k = cmd.index("/bin/bash")
            return unit, Path(cmd[k+4]), Path(cmd[k+5]), cmd[k+6:]
        return unit, None, None, cmd[cmd.index(next(c for c in cmd if str(c).endswith("python"))):]

    def run(self, cmd, record):
        unit, log, exit_file, argv = self.split(cmd)
        argv = [str(a) for a in argv]
        self.units.append(unit)
        self.clock += timedelta(seconds=self.step)
        arg = (lambda flag: argv[argv.index(flag)+1])
        if "-eval-" in unit:
            run, ev, rid = Path(argv[argv.index("bayesreg.eval.cli")+1]), Path(arg("--out-dir")), arg("--run-id")
            status = json.loads((run/"status.json").read_text())
            config = json.loads((run/"config.json").read_text())
            a = config["args"]
            seg35 = (.70 + (.004 if a.get("max_accepted_per_level") else 0.)
                     + (.002 if a["jd_cell_estimate"] == "certificate_bernstein" else 0.) + .01*SHORTS.index(rid[-9:]))
            (ev/rid).mkdir(parents=True)
            (ev/rid/"metrics.json").write_text(json.dumps(dict(
                run_id=rid, pair=a["pair"].split("/", 1)[1], field_role="native_formal", field_present=True,
                run_status=status["status"], metrics=dict(seg35=seg35, seg4=seg35+.05, hd95=2., sdlogj_official=.05,
                                                          sdlogj_brain=.2),
                certificate=dict(mode="full", status="certified_feasible", passed=True, min_jd_lower_bound=.1))))
            Path(arg("--long")).write_text("x\n")
        else:
            from bayesreg.cli import run_pair as rp
            assert argv[2:4] == ["-m", "bayesreg.cli.run_pair"]
            given = argv[4:]
            cell = given[given.index("--jd-cell-estimate")+1] if "--jd-cell-estimate" in given else None
            if cell == "certificate_bernstein" and not bernstein_offered():
                given = [("certificate" if g == "certificate_bernstein" else g) for g in given]
            args = rp.parse_args(given)
            run = Path(args.out)
            if unit in self.drop:
                return 137
            eta = json.loads(Path(args.eta_json).read_text())
            run.mkdir()
            recorded = {k: v for k, v in vars(args).items() if not k.startswith("_")}
            if cell is not None:
                recorded["jd_cell_estimate"] = cell
            if self.mutate and self.mutate[0] in unit:
                recorded = dict(recorded, **self.mutate[1])
            (run/"config.json").write_text(json.dumps(dict(args=recorded, eta=dict(
                alpha=eta["alpha"], nu=eta["nu"], tau=eta["tau"]))))
            bern = recorded["jd_cell_estimate"] == "certificate_bernstein"
            status = dict(status="budget_exhausted", wording="预算收尾", exit_code=0, native=True, exported="f"*64,
                          wall_budget=dict(wall_seconds=args.wall_seconds))
            if bern and not self.no_total:
                status["jd_scaling"] = dict(bernstein_j53_violations_total=self.violations)
            (run/"status.json").write_text(json.dumps(status))
            caps = recorded.get("max_accepted_per_level") or [500]*4
            rounds = [dict(level=lv, round=0, status="cap_reached", accepted=caps[lv], cap=caps[lv], commits=caps[lv],
                           GF_B=2., T_B=1., GF_B_over_T_B=2., min_jd_lower_bound=.01, seconds=10. if lv < 2 else
                           (80. if caps[2] < 500 else 400.),
                           counters=dict(candidates=caps[lv]+10, rejections=dict(JACOBIAN_VIOLATION=10)),
                           metric=dict(jd_scaling=dict(mode="affine_v1", cell_estimate=recorded["jd_cell_estimate"],
                                                       jd_scaling_refresh=3, min_J_a=.02, max_fraction_r_lt_1=.05,
                                                       computations=[])))
                      for lv in range(3)]
            native = 45 if caps[2] < 500 else 20
            rounds.append(dict(rounds[-1], level=3, status="budget_exhausted", accepted=native, cap=500, seconds=300.,
                               GF_B_over_T_B=50. if caps[2] < 500 else 100.))
            if recorded.get("max_accepted_per_level"):          # run_pair records cap_accepted with SCH switches only
                for lv, r in enumerate(rounds):
                    r["cap_accepted"] = caps[lv]
            hyper = dict(rounds=rounds)
            if recorded.get("sigma_domain") not in (None, "brain"):   # SWLI: support_levels for non-brain runs only
                for r in rounds:
                    r.update(sigma2_S_used=.004, sigma2_S_post=.005)
                hyper["support_levels"] = [dict(level=lv, N_q=1000., N_c_over_N_q=[.05, .01], N_c=[50., 10.],
                                                empty_support_channels=[1] if lv == 0 else [], N_S=60.,
                                                sigma2_S_start=.003) for lv in range(4)]
            (run/"hyper.json").write_text(json.dumps(hyper))
            if bern:
                (run/"log.jsonl").write_text("".join(json.dumps(dict(
                    kind="jd_scaling", level=lv, round=0, at="round_start", cell_estimate="certificate_bernstein",
                    min_J_a=.02, fraction_r_lt_1=.01, n_cells_raised=7, n_cells_bernstein=9,
                    n_bernstein_j53_violations=self.violations if lv == 0 else 0))+"\n" for lv in range(4)))
        if log is not None:
            log.write_text(f"fake {unit}\n")
            exit_file.write_text("0\n")
        return 0


def patch_m2v1(m2, monkeypatch, sha):
    def prepare(self):
        self.code, self.code_sha = ROOT, sha
        self.allowlist = self.allowlist or self.o.allowlist
    monkeypatch.setattr(m2.ArmSweep, "prepare_code", prepare)
    monkeypatch.setattr(m2.ArmSweep, "write_env", lambda self: None)


def make_inputs(dry_mod, m2, out, monkeypatch, *, code_sha=None, wall="60"):
    """m2v1 dry inputs plus an M2V1 dry root (dev arm only) made by experiments/m2v1/run_batch.py with fake units."""
    sha = code_sha or head()
    dry_mod.main(["--out", str(out), "--wall-seconds", wall, "--extra", "", "--code-sha", sha])
    patch_m2v1(m2, monkeypatch, sha)
    m2root = out/"m2v1"
    rc = m2.main(["--root", str(m2root), *m2v1_common(out), "--wall-seconds", wall, "--d47-last", "--until", "dev"],
                 system=FakeSystem())
    assert rc == 0, (m2root/"queue.log").read_text()[-3000:]
    assert sorted(p.name for p in (m2root/"runs"/"dev").iterdir()) == list(SHORTS)
    return m2root


@pytest.fixture
def inputs(dry_mod, m2, tmp_path, monkeypatch):
    out = tmp_path/"dry"
    return out, make_inputs(dry_mod, m2, out, monkeypatch)


@pytest.fixture
def driver(pl, monkeypatch):
    sha = head()

    def prepare(self):
        self.code, self.code_sha = ROOT, sha
        self.allowlist = self.allowlist or self.o.allowlist
    monkeypatch.setattr(pl.PilotArm, "prepare_code", prepare)
    monkeypatch.setattr(pl.PilotArm, "write_env", lambda self: None)

    def go(root, system, dry, m2root, *extra):
        return pl.main(["--root", str(root), *pilot_common(dry, m2root), *extra], system=system)
    return go


def runs_of(system):
    return [u for u in system.units if "-eval-" not in u]


# ---- end to end --------------------------------------------------------------------------------------------------------

def test_five_arms_pair_major_p0n_baseline_p0_reused_summary_and_restart(driver, inputs, tmp_path):
    dry, m2root = inputs
    root = tmp_path/"pilot"
    system = FakeSystem()
    assert driver(root, system, dry, m2root) == 0, (root/"queue.log").read_text()[-3000:]
    assert runs_of(system) == [f"tag-pilotdry-{a}-{s}" for s in SHORTS for a in ("p0n", "p1", "p2", "p3", "p4")]
    assert len(system.units) == 30 and all(f"tag-pilotdry-eval-{a}-{s}" in system.units for s in SHORTS
                                           for a in ("p0n", "p1", "p2", "p3", "p4"))
    launches = {a: next((root/"launch").glob(f"run_{a}-0902-0903_*.txt")).read_text() for a in ("p0n", "p1", "p2", "p3", "p4")}
    for a, text in launches.items():
        caps = a in ("p1", "p3", "p4")
        cell = "certificate_bernstein" if a in ("p2", "p3", "p4") else "certificate"
        assert ("--sigma-domain support_shared --support-threshold 0.05" in text) == (a == "p4")
        assert (f"--jd-cell-estimate {cell}" in text) and (("--max-accepted-per-level 500,500,100,500" in text) == caps)
        for piece in ("--channel-source structure", "--logvol-points gauss27+closed27", "--structure-infer never",
                      "--wall-seconds 60", "--device cpu", f"--eta-json {root/'eta'/'eta_pilot.json'}",
                      "--pair stage_L2R_val_v1/", "--nice=19", "--property=MemoryMax=9G"):
            assert piece in text, (a, piece)
        assert ('"max_accepted_per_level"' in text) == caps        # the configuration line of the launch record
        assert "--extra" not in text and "--max-accepted 2" not in text
    record = json.loads((root/"eta"/"eta_pilot.json").read_text())
    assert (record["alpha"], record["nu"], record["tau"]) == (.01, 50., [1., .5, .5, .5])
    assert record["pinned_sha256"] is None and record["model_code_changed_vs_m2v1"] == []
    assert record["configuration_check"]["sweep"] == record["configuration_check"]["m2v1_dev"]
    assert "max_accepted_per_level" in record["configuration_check"]["p3"]
    p0 = json.loads((root/"p0.json").read_text())
    assert {k: v["state"] for k, v in p0["pairs"].items()} == {s: "evaluated" for s in SHORTS}
    assert p0["pairs"][SHORTS[0]]["run_id"] == f"m2v1_dev_{SHORTS[0]}"
    batch = json.loads((root/"batch.json").read_text())
    assert batch["arm_order"] == ["p0n", "p1", "p2", "p3", "p4"] and batch["pairs"] == list(PAIRS)
    assert batch["arms"]["p4"]["changes"] == dict(jd_cell_estimate="certificate_bernstein",
                                                  max_accepted_per_level=[500, 500, 100, 500],
                                                  sigma_domain="support_shared", support_threshold=.05)
    assert (batch["arms"]["p4"]["expected_args"]["sigma_domain"], batch["arms"]["p4"]["expected_args"]["support_threshold"]
            ) == ("support_shared", .05)
    assert (batch["arms"]["p3"]["expected_args"]["sigma_domain"], batch["arms"]["p3"]["expected_args"]["support_threshold"]
            ) == ("brain", None)
    assert batch["arms"]["p0n"]["changes"] == {} and batch["arms"]["p0n"]["expected_args"]["max_accepted"] == 500
    assert batch["arms"]["p1"]["expected_args"]["max_accepted"] is None
    assert batch["arms"]["p2"]["expected_args"]["max_accepted"] == 500
    assert batch["arms"]["p3"]["changes"] == dict(jd_cell_estimate="certificate_bernstein",
                                                  max_accepted_per_level=[500, 500, 100, 500])
    s = json.loads((root/"summary.json").read_text())
    assert s["counts"]["ran"] == 15 and s["counts"]["args_mismatch"] == 0 and s["counts"]["p0_comparable"] == 3
    # the baseline is P0n (the lead 10-31); P0 (M2V1) only in the secondary columns, P0n - P0 included
    assert s["baseline"] == "p0n" and sorted(s["comparisons"]) == ["p1", "p2", "p3", "p4"] and sorted(s["flags"]) == [
        "p1", "p2", "p3", "p4"]
    assert sorted(s["secondary_comparisons"]) == ["p0n", "p1", "p2", "p3", "p4"]
    # P4 (revision 3): exploratory, described only; its own P4 - P3 comparison; the support records
    assert s["flags"]["p4"]["verdict"].startswith("只描述（探索臂 P4") and "N3_criterion" not in s["flags"]["p4"]["checks"]
    assert s["flags"]["p4"]["direction"]["reading"].startswith("可比对数 3 < 4")
    assert s["extra_comparisons"]["p4_vs_p3"]["seg4"]["n"] == 3
    sup = s["rows"]["p4"][0]["support"]
    assert [e["n_empty"] for e in sup] == [1, 0, 0, 0] and sup[0]["sigma2_S_used"] == [.004]
    assert s["rows"]["p3"][0]["support"] is None
    assert s["secondary_comparisons"]["p0n"]["seg35"]["median"] == pytest.approx(0.)
    assert s["counts"]["j53_computed"] == 0 and s["rows"]["p1"][0]["j53_final"]["state"] == "无 final_native.npz"
    assert s["rows"]["p1"][0]["caps"] == [500, 500, 100, 500] and s["rows"]["p0n"][0]["caps"] == [500]*4
    assert s["rows"]["p0"][0]["caps"] == [500]*4 and s["counts"]["bernstein_violation_runs"] == []
    assert batch["arms"]["p2"]["expected_args"]["sigma_domain"] == "brain"
    c1 = s["comparisons"]["p1"]
    assert c1["native_accepted"]["median_ratio"] == pytest.approx(45/20) and c1["seg35"]["n"] == 3
    assert c1["seg35"]["median"] == pytest.approx(.004)
    # three pairs: fewer than MIN_N = 4 comparable pairs -> describe only
    assert all(s["flags"][a]["verdict"].startswith("只描述") for a in ("p1", "p2", "p3"))
    assert s["flags"]["p2"]["checks"]["S3_bernstein_violations_zero"] is True
    b = s["rows"]["p3"][0]["bernstein"]
    assert (b["computations"], b["violations_total"], b["violations_in_rows"], b["violations"]) == (4, 0, 0, 0)
    assert s["recommendation"]["path"] == "p0"
    md = (root/"summary.md").read_text()
    assert md.startswith(f"# 路径改进试验汇总（{root}）\n\ncertificate_bernstein 核对 (b)：没有违反计数 > 0 的运行。")
    for piece in ("P4 − P3 的配对差（σ 估计域的效应", "σ 估计域的支撑记录", "方向假设（预登记，描述用）", "seg35", "seg4",
                  "P1 − P0n", "P0n − P0（M2V1）", "次要参照", "P3 − P1", "原生接受步", "500/500/100/500",
                  "certificate_bernstein 记录", "各层读数", "53 点最小 J", "单次观测", "预算收尾", "主要对照（基线）：**P0n**"):
        assert piece in md, piece
    again = FakeSystem()
    assert driver(root, again, dry, m2root) == 0 and again.units == []
    assert "(unchanged)" in (root/"queue.log").read_text()


def test_p0_mismatch_and_running_m2v1_stop_before_any_unit(driver, inputs, tmp_path):
    dry, m2root = inputs
    system = FakeSystem(m2v1=["tag-m2v1-dev-0903-0904.service"])
    assert driver(tmp_path/"a", system, dry, m2root) == 8 and system.units == []
    assert "M2V1 is still running" in (tmp_path/"a"/"queue.log").read_text()
    # --after-m2v1: waits while M2V1 runs, then goes on
    waiting = FakeSystem(m2v1=["tag-m2v1-dev-0903-0904.service"])
    waiting.on_sleep = lambda s: s.m2v1.clear() if s.slept >= 2 else None
    assert driver(tmp_path/"b", waiting, dry, m2root, "--after-m2v1", "--until", "eta") == 0 and waiting.units == []
    assert waiting.slept == 2 and (tmp_path/"b"/"p0.json").is_file()
    # a P0 run whose config.json differs from M2V1's dev arm
    bad = tmp_path/"m2bad"
    shutil.copytree(m2root, bad, ignore=shutil.ignore_patterns("driver.lock"))
    config = bad/"runs"/"dev"/SHORTS[1]/"config.json"
    doc = json.loads(config.read_text())
    doc["args"]["max_accepted"] = 400
    config.write_text(json.dumps(doc))
    system = FakeSystem()
    assert driver(tmp_path/"c", system, dry, bad) == 8 and system.units == []
    assert "P0 0902-0903: config.json args differ" in (tmp_path/"c"/"queue.log").read_text()
    # M2V1 recorded another eta source
    other = tmp_path/"m2eta"
    shutil.copytree(m2root, other, ignore=shutil.ignore_patterns("driver.lock"))
    batch = json.loads((other/"batch.json").read_text())
    batch["eta_source"]["source_sha256"] = "0"*64
    (other/"batch.json").write_text(json.dumps(batch))
    assert driver(tmp_path/"d", FakeSystem(), dry, other) == 8
    assert "eta source sha256" in (tmp_path/"d"/"queue.log").read_text()


def test_missing_p0_run_is_recorded_and_the_pilot_goes_on(driver, inputs, tmp_path):
    dry, m2root = inputs
    gone = tmp_path/"m2gone"
    shutil.copytree(m2root, gone, ignore=shutil.ignore_patterns("driver.lock"))
    shutil.rmtree(gone/"runs"/"dev"/SHORTS[2])
    shutil.rmtree(gone/"eval"/"dev"/SHORTS[2])
    for path in (gone/"launch").glob(f"*dev-{SHORTS[2]}_*.txt"):
        path.unlink()
    root = tmp_path/"pilot"
    system = FakeSystem()
    assert driver(root, system, dry, gone, "--arms", "p1") == 0, (root/"queue.log").read_text()[-2000:]
    assert runs_of(system) == [f"tag-pilotdry-p1-{s}" for s in SHORTS]
    p0 = json.loads((root/"p0.json").read_text())
    assert p0["pairs"][SHORTS[2]]["state"] == "missing"
    s = json.loads((root/"summary.json").read_text())
    assert s["comparisons"]["p1"]["seg35"]["n"] == 2 and s["counts"]["p0_comparable"] == 2


def test_config_mismatch_stops_and_arms_are_fixed_per_root(driver, inputs, tmp_path):
    dry, m2root = inputs
    root = tmp_path/"pilot"
    system = FakeSystem(mutate=("p1-0901", dict(max_accepted_per_level=[500, 500, 500, 500])))
    assert driver(root, system, dry, m2root) == 8
    assert runs_of(system) == ["tag-pilotdry-p0n-0901-0902", "tag-pilotdry-p1-0901-0902"]
    assert "config.json args differ" in (root/"queue.log").read_text()
    # P4's config.json must say support_shared (revision 3)
    other = tmp_path/"pilot4"
    system = FakeSystem(mutate=("p4-0901", dict(sigma_domain="brain", support_threshold=None)))
    assert driver(other, system, dry, m2root) == 8
    assert runs_of(system) == [f"tag-pilotdry-{a}-0901-0902" for a in ("p0n", "p1", "p2", "p3", "p4")]
    log = (other/"queue.log").read_text()
    assert "config.json args differ" in log and "'sigma_domain': ('brain', 'support_shared')" in log
    # another arm set on the same root: the recomputed records (eta_pilot.json's configuration check, batch.json)
    # differ -> STOP, nothing launched
    again = FakeSystem()
    assert driver(root, again, dry, m2root, "--arms", "p1,p3") == 8 and again.units == []
    assert "eta_pilot.json differs from the recomputed record" in (root/"queue.log").read_text()


def test_gate_waits_for_driver_processes_and_deadline(driver, inputs, tmp_path):
    dry, m2root = inputs
    system = FakeSystem(drivers=[4242])
    assert driver(tmp_path/"a", system, dry, m2root, "--arms", "p1") == 8 and system.units == []
    assert "driver processes running: [4242]" in (tmp_path/"a"/"queue.log").read_text()
    waiting = FakeSystem(drivers=[4242])
    waiting.on_sleep = lambda s: s.drivers.clear()
    assert driver(tmp_path/"b", waiting, dry, m2root, "--arms", "p1", "--wait-minutes", "5") == 0
    assert len(runs_of(waiting)) == 3
    # deadline after the first run: the rest are not run; a later call runs them
    root = tmp_path/"c"
    late = FakeSystem(step=3600.)
    deadline = (T0+timedelta(minutes=30)).isoformat()
    assert driver(root, late, dry, m2root, "--arms", "p1", "--no-start-after", deadline) == 0
    assert runs_of(late) == ["tag-pilotdry-p1-0901-0902"]
    assert "not run (deadline" in (root/"queue.log").read_text()
    s = json.loads((root/"summary.json").read_text())
    assert s["counts"]["not_run"] == 2
    later = FakeSystem(clock=T0+timedelta(days=1))
    assert driver(root, later, dry, m2root, "--arms", "p1") == 0
    assert runs_of(later) == ["tag-pilotdry-p1-0902-0903", "tag-pilotdry-p1-0903-0904"]


def test_model_code_change_needs_accept(dry_mod, m2, driver, tmp_path, monkeypatch):
    out = tmp_path/"dry"
    m2root = make_inputs(dry_mod, m2, out, monkeypatch, code_sha=OLD)
    root = tmp_path/"pilot"
    assert driver(root, FakeSystem(), out, m2root, "--until", "eta") == 8
    assert "--accept-code-change" in (root/"queue.log").read_text()
    assert driver(tmp_path/"ok", FakeSystem(), out, m2root, "--until", "eta", "--accept-code-change") == 0
    record = json.loads((tmp_path/"ok"/"eta"/"eta_pilot.json").read_text())
    assert record["accept_code_change"] is True and any(f.startswith("bayesreg/")
                                                         for f in record["model_code_changed_vs_m2v1"])


def test_code_tree_flags_and_arm_parsing(pl, tmp_path):
    def queue(arms):
        return pl.PilotQueue(pl.parse_args(["--root", str(tmp_path/"x"), "--dry-run", "--arms", arms]),
                             system=FakeSystem())
    assert queue("p1").base.need_flags == ("--max-accepted-per-level",)
    assert queue("p4").base.need_flags == ("--max-accepted-per-level", "certificate_bernstein", "--sigma-domain",
                                           "support_shared", "--support-threshold")
    q = queue("p1,p2")
    assert q.base.need_flags == ("--max-accepted-per-level", "certificate_bernstein")
    q.base.code = ROOT
    if bernstein_offered():
        q.base.check_code_tree()
    else:
        with pytest.raises(pl.Stop, match="certificate_bernstein"):
            q.base.check_code_tree()
    for bad in ("p5", "p1,p1", ""):
        with pytest.raises(SystemExit):
            pl.parse_args(["--arms", bad])
    a = queue("p3").arms["p3"]
    a.registry_record = None
    assert a.configuration()["max_accepted_per_level"] == [500, 500, 100, 500]
    assert a.configuration()["jd_cell_estimate"] == "certificate_bernstein"
    assert pl.changes_of(pl.ARMS["p0n"]) == {}


def test_driver_process_matching(pl, tmp_path):
    def proc(pid, ppid, *argv):
        d = tmp_path/"proc"/str(pid)
        d.mkdir(parents=True)
        (d/"cmdline").write_bytes(b"\0".join(a.encode() for a in argv)+b"\0")
        (d/"stat").write_text(f"{pid} (x) S {ppid} 0 0")
    proc(10, 1, "/usr/bin/python3", "-B", "experiments/m2v1/run_batch.py", "--d47-last")
    proc(11, 1, "/bin/bash", "-c", "setsid nohup python experiments/m2v1/run_batch.py")
    proc(12, 1, "/x/python", "-B", "experiments/eta/sweep.py")
    proc(13, 1, "/x/python", "-B", "experiments/pilot/run_pilot.py")
    proc(14, 13, "/x/python", "-B", "experiments/pilot/run_pilot.py")
    proc(15, 1, "/x/python", "-B", "experiments/m2v1/run_batch.py.bak")
    found = pl.driver_processes(tmp_path/"proc", 14, pl.DRIVER_SCRIPTS)
    assert found == [10, 12]
    assert pl.driver_processes(tmp_path/"proc", 1, (pl.M2V1_SCRIPT,)) == [10]


# ---- readings ------------------------------------------------------------------------------------------------------

def row(short, *, comparable=True, seg35=.80, seg4=.85, native=20, gf=100., blocked=0, state="ran", total=0):
    return dict(pair=f"P_{short}", short=short, state=state, comparable=comparable, n_blocked=blocked, exported=True,
                field_role="native_formal",
                metrics=dict(seg35=seg35, seg4=seg4, hd95=2., sdlogj_official=.05, sdlogj_brain=.2),
                native_reached=True, native_accepted=native, native_seconds=300., native_gf_over_tb=gf,
                coarse_reached=0, jv_rejections=10, bernstein=dict(violations=total))


def test_readings_follow_the_preregistered_thresholds(pl):
    ps = pl.psum
    p0 = [row(f"{k}", native=20) for k in range(5)]

    def flags(rows, cell="certificate", caps=(500, 500, 100, 500), aid="p1"):
        return ps.arm_flags(aid, rows, p0, ps.compare(rows, p0), cell, caps)
    good = [row(f"{k}", native=45, seg35=.803, seg4=.852) for k in range(5)]
    assert flags(good)["verdict"].startswith("倾向列入候选")
    assert flags([row(f"{k}", native=35) for k in range(5)])["verdict"].startswith("无法区分")   # ratio 1.75 < 2
    assert flags([dict(r, n_blocked=1) if k == 0 else r for k, r in enumerate(good)])["verdict"] == \
        "不列入候选（安全条件不满足）"
    assert flags([dict(r, comparable=False) if k == 4 else r for k, r in enumerate(good)])["verdict"] == \
        "不列入候选（安全条件不满足）"
    assert flags([row(f"{k}", native=45, seg4=.835) for k in range(5)])["verdict"] == "不列入候选（Dice 降低）"
    assert flags([row(f"{k}", native=45, seg35=.794) for k in range(5)])["verdict"].startswith("无法区分")  # -.006
    assert flags(good[:3]+[row(f"{k}", state="not_run", comparable=False) for k in (3, 4)])["verdict"].startswith("只描述")
    # P2: steps ratio 1.5, or the native GF_B/T_B halved on >= 4 of 5 pairs
    p2 = (lambda rows: flags(rows, cell="certificate_bernstein", caps=None, aid="p2")["verdict"])
    assert p2([row(f"{k}", native=30) for k in range(5)]).startswith("倾向")
    assert p2([row(f"{k}", native=20, gf=40.) for k in range(5)]).startswith("倾向")
    assert p2([row(f"{k}", native=20, gf=80.) for k in range(5)]).startswith("无法区分")
    assert p2([row(f"{k}", native=30, total=1 if k == 2 else 0) for k in range(5)]).startswith("待审核")
    assert p2([row(f"{k}", native=30, total=None) for k in range(5)]).startswith("待核对")
    # PILOT review I1 (a): a violation is never hidden by a missing count elsewhere
    mixed = flags([row(f"{k}", native=30, total=t) for k, t in enumerate((0, 3, None, 0, 0))], cell="certificate_bernstein",
                  caps=None, aid="p2")
    assert mixed["verdict"].startswith("待审核") and mixed["checks"]["S3_bernstein_violations_zero"] is False
    assert any("[0, 3, None, 0, 0]" in x and "重新认证" in x for x in mixed["reasons"])
    n3 = (lambda rows: flags(rows, cell="certificate_bernstein", caps=None, aid="p2")["checks"]["N3_criterion"])
    assert n3([row(f"{k}") for k in range(5)]) is True                       # no benefit needed
    # pair by pair (review minor 1): one pair worse is not offset by another pair better
    p0[1] = dict(p0[1], n_blocked=1)
    assert n3([row(f"{k}", blocked=1 if k == 0 else 0) for k in range(5)]) is False
    p0[1] = dict(p0[1], n_blocked=0)
    # under the audit rule (review I1 (b)): S3 False -> 待审核, S3 None -> 待核对, whatever S1 and S2 say
    assert n3([row(f"{k}", total=t) for k, t in enumerate((0, 3, None, 0, 0))]) == "待审核"
    assert n3(good[:3]+[row(f"{k}", total=1) for k in (3, 4)]) == "待审核"
    assert n3([row(f"{k}", total=None) for k in range(5)]) == "待核对"
    # S1 reasons tell a missing export (budget) from a non-comparable export (review minor 2)
    nox = flags([dict(r, comparable=False, exported=False) if k == 4 else r for k, r in enumerate(good)])
    assert any("预算或运行失败" in x for x in nox["reasons"])
    cert = flags([dict(r, comparable=False) if k == 4 else r for k, r in enumerate(good)])
    assert any("证书未通过或评测失败" in x for x in cert["reasons"])
    assert "N3_criterion" not in flags(good)["checks"]
    # P4 direction hypothesis (revision 3), descriptive
    base4 = [row(f"{k}", seg35=.80, seg4=.80) for k in range(5)]
    up = ps.p4_direction(ps.compare([row(f"{k}", seg35=.797, seg4=.83) for k in range(5)], base4))
    assert up["reading"].startswith("与方向假设一致") and up["median_seg4"] == pytest.approx(.03)
    assert ps.p4_direction(ps.compare([row(f"{k}", seg35=.79, seg4=.83) for k in range(5)], base4))["reading"].startswith(
        "与方向假设不一致")                                                       # seg35 -.01
    assert ps.p4_direction(ps.compare([row(f"{k}", seg4=.81) for k in range(5)], base4))["reading"].startswith(
        "与方向假设不一致")                                                       # seg4 +.01 < +.02
    assert ps.p4_direction(ps.compare(base4[:3], base4[:3]))["reading"].startswith("可比对数 3")
    # recommendation (P4 never enters it: its verdict is never a candidate label)
    lean, no = "倾向列入候选（交总负责人）", "无法区分（只描述）"
    p3_vs_p1 = ps.compare(good, [row(f"{k}", native=40, seg35=.803, seg4=.852) for k in range(5)])
    assert ps.recommend(dict(p1=lean, p2=no, p3=lean), p3_vs_p1)[0] == "p3"
    worse = ps.compare([row(f"{k}", native=30) for k in range(5)], good)
    assert ps.recommend(dict(p1=lean, p2=no, p3=lean), worse)[0] == "p1"
    assert ps.recommend(dict(p1=no, p2=lean, p3=no), None)[0] == "p2"
    assert ps.recommend(dict(p1=no, p2=no, p3=no), None)[0] == "p0"
    assert ps.recommend(dict(p1=no, p2=no, p3=no, p4="只描述（探索臂 P4：σ 估计域 support_shared；不贴候选标签）"),
                        None)[0] == "p0"


# ---- the dry run in process (slow) ------------------------------------------------------------------------------------

class InlineSystem(FakeSystem):
    """Runs the units' commands in this process instead of systemd: bayesreg.cli.run_pair and bayesreg.eval.cli with
    the exact argument lists the drivers built (CPU torch64, CUDA hidden by the caller)."""

    def run(self, cmd, record):
        unit, log, exit_file, argv = self.split(cmd)
        self.units.append(unit)
        argv = [str(a) for a in argv]
        if argv[2:4] == ["-m", "bayesreg.cli.run_pair"]:
            from bayesreg.cli import run_pair
            entry = run_pair.main
        elif argv[2:4] == ["-m", "bayesreg.eval.cli"]:
            from bayesreg.eval import cli
            entry = cli.main
        else:
            raise AssertionError(f"unexpected unit {unit}: {argv}")
        if log is None:
            return entry(argv[4:])
        with open(log, "a", encoding="utf-8") as stream, contextlib.redirect_stdout(stream), \
                contextlib.redirect_stderr(stream):
            rc = entry(argv[4:])
        exit_file.write_text(f"{rc}\n")
        return rc


@pytest.mark.slow
def test_dry_run_in_process(pl, m2, dry_mod, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")              # CPU only, whatever the caller set (review minor 12)
    deadline = (datetime.now(timezone.utc)+timedelta(hours=6)).isoformat(timespec="minutes")
    wall = "60"
    out = tmp_path/"dry"
    sha = head()
    dry_mod.main(["--out", str(out), "--wall-seconds", wall, "--extra", "", "--code-sha", sha])

    def prepare_m2(self):
        self.code, self.code_sha = ROOT, sha
        self.allowlist = self.allowlist or str(out/"dev_subjects.json")
        self.check_code_tree()
    monkeypatch.setattr(m2.ArmSweep, "prepare_code", prepare_m2)
    monkeypatch.setattr(m2.ArmSweep, "write_env", lambda self: None)
    m2root = out/"m2v1"
    clock = datetime.now(timezone.utc)
    assert m2.main(["--root", str(m2root), *m2v1_common(out), "--wall-seconds", wall, "--d47-last", "--until", "dev",
                    "--no-start-after", deadline], system=InlineSystem(clock=clock)) == 0, \
        (m2root/"queue.log").read_text()[-4000:]
    monkeypatch.setattr(pl.PilotArm, "prepare_code", prepare_m2)
    monkeypatch.setattr(pl.PilotArm, "write_env", lambda self: None)
    arms = "p0n,p1,p2,p3,p4" if bernstein_offered() else "p0n,p1"
    root = tmp_path/"pilot"
    args = ["--root", str(root), *pilot_common(out, m2root, wall), "--arms", arms, "--no-start-after", deadline]
    system = InlineSystem(clock=datetime.now(timezone.utc))
    assert pl.main(args, system=system) == 0, (root/"queue.log").read_text()[-4000:]
    n_arms = len(arms.split(","))
    assert len(runs_of(system)) == 3*n_arms and len(system.units) == 6*n_arms
    for config in (root/"runs").glob("*/*/config.json"):
        a = json.loads(config.read_text())["args"]
        arm = config.parent.parent.name
        caps = arm in ("p1", "p3", "p4")
        assert (a["channel_source"], a["logvol_points"], a["structure_infer"], a["structure_registry_staging"]) == (
            "structure", "gauss27+closed27", "never", True)
        assert a["jd_cell_estimate"] == ("certificate_bernstein" if arm in ("p2", "p3", "p4") else "certificate")
        assert (a["sigma_domain"], a["support_threshold"]) == (("support_shared", .05) if arm == "p4" else ("brain", None))
        # without an SCH switch run_pair writes neither SCH key (C16 SCH: records only when given)
        assert a.get("max_accepted_per_level") == ([500, 500, 100, 500] if caps else None)
        assert ("max_accepted_per_level" in a) == caps and ("level_time_shares" in a) == caps
        assert a["max_accepted"] == (None if caps else 500) and a.get("level_time_shares") is None
        hyper = json.loads((config.parent/"hyper.json").read_text())
        if caps:
            assert [r.get("cap_accepted") for r in hyper["rounds"]] == [500, 500, 100, 500]
    s = json.loads((root/"summary.json").read_text())
    rows = [r for a in ["p0", *s["arm_order"]] for r in s["rows"][a]]
    # hard constraint 1: every exported field passes the evaluator's certificate. The wall budget on the synthetic pairs
    # depends on the machine's load: a run may end before the native level (stopped_before_native, no field) or reach
    # it with no accepted step (handover_only, not comparable) -- recorded, never a violation (dry run 22:38 UTC 10-31)
    for r in rows:
        if r["exported"]:
            assert r["certificate"]["passed"] is True, r["short"]
            # the 53-point minimum J of the final state (N3 record) is computed and >= the certified lower bound
            assert r["j53_final"]["state"] == "computed" and r["j53_final"]["ge_certified_bound"] is True, r["short"]
        else:
            assert r["run_status"] == "stopped_before_native", (r["short"], r["run_status"])
        assert r["comparable"] == (r["field_role"] == "native_formal" and (r["certificate"] or {}).get("passed") is True)
    assert sum(r["field_role"] == "native_formal" for r in rows) >= len(rows)-2
    assert s["counts"]["args_mismatch"] == 0 and s["counts"]["eta_mismatch"] == 0 and s["counts"]["ran"] == 3*n_arms
    assert s["baseline"] == "p0n" and s["comparisons"]["p1"]["seg35"]["n"] >= 2
    if "p4" in s["arm_order"]:
        assert all(r["support"] and len(r["support"]) == 4 for r in s["rows"]["p4"] if r["native_reached"])
        assert all(r["support"] is None for a in ("p0n", "p1", "p2", "p3") for r in s["rows"][a])
    assert s["counts"]["j53_below_certified_bound"] == 0 and s["counts"]["j53_computed"] >= len(rows)-2
    md = (root/"summary.md").read_text()
    assert "seg35" in md and "seg4" in md and "各层读数" in md
    again = InlineSystem(clock=datetime.now(timezone.utc))
    assert pl.main(args, system=again) == 0 and again.units == []
    assert "(unchanged)" in (root/"queue.log").read_text()


def test_bernstein_violation_count_from_rows_or_run_total(pl, tmp_path):
    ps = pl.psum
    run = tmp_path/"run"
    run.mkdir()
    assert ps.bernstein_readouts(run)["violations"] is None
    rows = [dict(kind="jd_scaling", level=lv, round=0, cell_estimate="certificate_bernstein", n_cells_raised=3+lv,
                 n_bernstein_j53_violations=v) for lv, v in ((0, 0), (3, 1))]
    (run/"log.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    b = ps.bernstein_readouts(run)
    assert (b["violations_in_rows"], b["total_found"], b["violations"]) == (1, False, 1)
    assert b["per_level"]["3"]["max_cells_raised"] == 6
    (run/"hyper.json").write_text(json.dumps(dict(algorithm=dict(jd=dict(bernstein_j53_violations_total=0)))))
    assert ps.bernstein_readouts(run)["violations"] == 1          # the larger of the recorded counts
    (run/"log.jsonl").write_text(json.dumps(dict(rows[0], n_bernstein_j53_violations=None))+"\n")
    b = ps.bernstein_readouts(run)
    assert (b["rows_field"], b["violations_total"], b["violations"]) == (False, 0, 0)


def test_j53_of_the_final_state(pl, tmp_path):
    """summarize.j53_final: read only, the file's own grid, closed27_cell_estimate; --no-j53 and failures recorded."""
    import numpy as np
    from bayesreg.solve.export import write_final_state
    from bayesreg.solve.jd_scaling import closed27_cell_estimate
    from tests.test_bayesreg.solve import _synthetic as syn
    ps = pl.psum
    grid = syn.cube_grid(7, h=1.5, origin=(2., -1., .5))
    run = tmp_path/"run"
    run.mkdir()
    assert ps.j53_final(run)["state"] == "无 final_native.npz"
    u = np.zeros(grid.shape+(3,))
    z, y, x = np.meshgrid(*(np.linspace(0., np.pi, n) for n in grid.shape), indexing="ij")
    bump = np.sin(z)*np.sin(y)*np.sin(x)
    u[..., 0], u[..., 1], u[..., 2] = .9*bump, -.4*bump*np.cos(x), .3*bump
    u[[0, -1]] = 0.
    u[:, [0, -1]] = 0.
    u[:, :, [0, -1]] = 0.
    write_final_state(run/"final_native.npz", u, grid, status="budget_exhausted")
    before = (run/"final_native.npz").read_bytes()
    got = ps.j53_final(run)
    want, default = closed27_cell_estimate(u, grid)
    assert got["state"] == "computed" and got["min_j53"] == float(want.min())
    assert got["min_gauss27_corners"] == float(default.min()) and got["min_j53"] <= got["min_gauss27_corners"]
    assert tuple(got["argmin_cell_zyx"]) == tuple(int(i) for i in np.unravel_index(int(np.argmin(want)), want.shape))
    assert (run/"final_native.npz").read_bytes() == before
    assert ps.j53_final(run, compute=False)["state"].startswith("审核时另算")
    # a non-cubic, anisotropic grid with a non-zero origin (review minor 12): axis order and the grid_id
    from jd_curl.three_d.grid import make_grid_3d
    aniso = make_grid_3d(nx=9, ny=7, nz=5, hx=1., hy=1.5, hz=2., x0=-3., y0=4., z0=1.25, unit="mm")
    v = np.zeros(aniso.shape+(3,))
    zz, yy, xx = np.meshgrid(*(np.sin(np.linspace(0., np.pi, n)) for n in aniso.shape), indexing="ij")
    v[..., 0], v[..., 1], v[..., 2] = .5*zz*yy*xx, -.3*zz*yy*xx**2, .2*zz**2*yy*xx
    v[[0, -1]], v[:, [0, -1]], v[:, :, [0, -1]] = 0., 0., 0.        # sin(pi) is not exactly 0: six zero faces
    other = tmp_path/"aniso"
    other.mkdir()
    write_final_state(other/"final_native.npz", v, aniso, status="budget_exhausted")
    got = ps.j53_final(other)
    want, _ = closed27_cell_estimate(v, aniso)
    assert got["state"] == "computed" and got["min_j53"] == float(want.min()) and got["grid_id"] == aniso.grid_id
    # a final_native.npz without state_hash: not silently computed (review minor 3)
    bare = tmp_path/"bare"
    bare.mkdir()
    np.savez(bare/"final_native.npz", u=v)
    assert ps.j53_final(bare)["state"].startswith("待核对")
    (run/"final_native.npz").write_bytes(b"not an npz")
    assert ps.j53_final(run)["state"].startswith("审核时另算")


def test_deadline_must_carry_a_date_and_an_offset_and_be_ahead(pl, driver, inputs, tmp_path):
    """PILOT review I2: HH:MM and naive date-times refused at parsing; a deadline already past STOPs before anything is
    read or started."""
    for bad in ("05:00", "23:59", "2026-11-01T05:00", "tomorrow"):
        with pytest.raises(SystemExit):
            pl.parse_args(["--no-start-after", bad])
    assert pl.parse_args(["--no-start-after", "2026-11-01T05:00-06:00"]).no_start_after == "2026-11-01T05:00-06:00"
    dry, m2root = inputs
    root = tmp_path/"pilot"
    system = FakeSystem()
    past = (T0-timedelta(minutes=1)).isoformat()
    assert driver(root, system, dry, m2root, "--no-start-after", past) == 8 and system.units == []
    assert "already past" in (root/"queue.log").read_text() and not (root/"eta").exists()


def test_bernstein_violation_runs_are_listed_first(driver, inputs, tmp_path):
    """PILOT review I1 (c): every run with a violation count > 0 is listed in counts and at the top of summary.md,
    whatever the arm's label (here n = 3: describe only)."""
    dry, m2root = inputs
    root = tmp_path/"pilot"
    assert driver(root, FakeSystem(violations=2), dry, m2root) == 0
    s = json.loads((root/"summary.json").read_text())
    listed = s["counts"]["bernstein_violation_runs"]
    assert sorted((b["arm"], b["short"]) for b in listed) == sorted((a, x) for a in ("p2", "p3", "p4") for x in SHORTS)
    assert all(b["violations"] == 2 for b in listed)
    assert s["flags"]["p2"]["verdict"].startswith("只描述") and s["flags"]["p2"]["checks"]["N3_criterion"] == "待审核"
    md = (root/"summary.md").read_text()
    head = md.split("\n")[2]
    assert head.startswith("**certificate_bernstein 核对 (b) 有违反的运行（D53）：") and "重新认证终态" in head
    assert "P2 0901-0902：2" in head and "P3 0903-0904：2" in head
