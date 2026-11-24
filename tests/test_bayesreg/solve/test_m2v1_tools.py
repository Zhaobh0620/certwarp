"""The M2V1 tools (experiments/m2v1; runs/20261031-m2v1-dev/plan.md), on synthetic files only (tmp_path; no data/, no
unit, no real segmentation, no real result file):

- run_batch.py with a fake machine (units emulated; run_pair's arguments parsed by its real parser): the D47 arm first and
  the dev batch after it, one unit each, evaluations; launch records with the arm's configuration and arguments; the eta
  record (source sha256, configuration checks); batch.json; the summary (paired differences against A0 and A4' for seg35
  and seg4, convergence table, D47 table); a restart launches nothing; the deadline (none started after it, listed as not
  run, a later call runs them; a deadline reached while the gate waits); a killed run is recorded and never relaunched,
  the batch goes on; consecutive failures STOP; a config.json mismatch STOPs; eta checks (configuration, overlapping
  sweep pairs, model code changed, a changed eta file on restart); --after-sweep (waits then runs; the sweep's STOP: exit
  9, nothing done; wait limit); the gate (busy unit, training unit not excepted); dry-run path refusals; the precheck
  (missing prediction, pool check); the summary's reference pair check;
- level_table.py on fake and killed runs; summarize's D47 flags and reading labels; the deadline parser;
- slow (-m slow): the dry run in process (real run_pair, CPU torch64, and bayesreg.eval.cli on the synthetic structure
  layer; no systemd unit).
"""
import contextlib
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
P1, P2, P3 = ("OASIS_OAS1_0901_MR1__OASIS_OAS1_0902_MR1", "OASIS_OAS1_0902_MR1__OASIS_OAS1_0903_MR1",
              "OASIS_OAS1_0903_MR1__OASIS_OAS1_0904_MR1")
T0 = datetime(2026, 10, 31, 21, 0, tzinfo=timezone.utc)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def head():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


@pytest.fixture(scope="module")
def q():
    return load("m2v1_run_batch", "experiments/m2v1/run_batch.py")


@pytest.fixture(scope="module")
def lt():
    return load("m2v1_level_table", "experiments/m2v1/level_table.py")


@pytest.fixture(scope="module")
def dry_mod():
    return load("m2v1_dry_inputs", "experiments/m2v1/dry_inputs.py")


@pytest.fixture(scope="module")
def dry(tmp_path_factory, dry_mod):
    out = tmp_path_factory.mktemp("m2v1")/"dry"
    dry_mod.main(["--out", str(out), "--wall-seconds", "60", "--extra", "", "--code-sha", head()])
    return out


def common(dry, sweep=None):
    return ["--dry-run", "--data-root", str(dry/"data"), "--allowlist", str(dry/"dev_subjects.json"), "--tissue-root",
            str(dry/"seg"), "--structure-root", str(dry/"structure"), "--segmenter-registry",
            str(dry/"registry"/"segmenters.json"), "--sweep-root", str(sweep or dry/"sweep"), "--d0-long",
            str(dry/"d0"/"long.csv"), "--a4-long", str(dry/"a4"/"long.csv"), "--smoke-root", str(dry/"smoke"),
            "--ref-reeval", str(dry/"ref_reeval"), "--d47-pair", P1, "--wall-seconds", "60"]


class FakeSystem:
    """Emulates the units: run_pair (arguments through its real parser) and bayesreg.eval.cli write what they would."""

    def __init__(self, *, busy=(), drop=(), fail=(), mutate=None, eta_units=(), driver=(), clock=T0, step=60.,
                 on_sleep=None, noexport=None, badcert=()):
        self.self_unit, self.units, self.busy, self.drop, self.fail = "", [], list(busy), set(drop), set(fail)
        # noexport {unit: status}: run_pair ends without disp.npz but writes the state hash into status.json "exported"
        # (68721d1: final_certificate_rejected, stopped_before_native; M2V1 review I1); badcert: run ids whose
        # evaluated field fails the evaluator's certificate (hard constraint 1)
        self.noexport, self.badcert = dict(noexport or {}), set(badcert)
        self.mutate, self.eta, self.driver, self.clock, self.step, self.on_sleep = (mutate, list(eta_units), list(driver),
                                                                                 clock, step, on_sleep)
        self.slept = 0

    def unit_active(self, unit):
        return False

    def other_units(self):
        return list(self.busy)

    def active_units(self):
        return []

    def eta_units(self):
        return list(self.eta)

    def sweep_driver_running(self):
        return list(self.driver)

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
        return unit, None, None, cmd[cmd.index(next(c for c in cmd if c.endswith("python"))):]

    def run(self, cmd, record):
        unit, log, exit_file, argv = self.split(cmd)
        self.units.append(unit)
        self.clock += timedelta(seconds=self.step)
        arg = (lambda flag: argv[argv.index(flag)+1])
        if "-eval-" in unit:
            run, ev, rid = Path(argv[argv.index("bayesreg.eval.cli")+1]), Path(arg("--out-dir")), arg("--run-id")
            status = json.loads((run/"status.json").read_text())
            config = json.loads((run/"config.json").read_text())
            cell = config["args"]["jd_cell_estimate"]
            seg35 = .55+(.01 if cell == "certificate" else 0.)+.001*len(rid)
            present = status["exit_code"] == 0
            (ev/rid).mkdir(parents=True)
            (ev/rid/"metrics.json").write_text(json.dumps(dict(
                run_id=rid, field_role="native_formal" if present else "no_native_result", field_present=present,
                run_status=status["status"],
                metrics=dict(seg35=seg35, seg4=seg35+.1, hd95=2., sdlogj_official=.05, sdlogj_brain=.2) if present else {},
                certificate=(dict(mode="full", status="certified_infeasible", passed=False, min_jd_lower_bound=-.1)
                             if rid in self.badcert else
                             dict(mode="full", status="certified_feasible", passed=True, min_jd_lower_bound=.1))
                if present else {})))
            Path(arg("--long")).write_text("x\n")
        else:
            from bayesreg.cli import run_pair as rp
            assert argv[2:4] == ["-m", "bayesreg.cli.run_pair"]
            args = rp.parse_args([str(a) for a in argv[4:]])
            run = Path(args.out)
            if unit in self.drop:
                return 137
            eta = json.loads(Path(args.eta_json).read_text())
            run.mkdir()
            recorded = vars(args)
            if self.mutate:
                recorded = dict(recorded, **self.mutate)
            (run/"config.json").write_text(json.dumps(dict(args=recorded, eta=dict(
                alpha=eta["alpha"], nu=eta["nu"], tau=eta["tau"]))))
            failed = unit in self.fail
            if unit in self.noexport:
                (run/"status.json").write_text(json.dumps(dict(
                    status=self.noexport[unit], wording="运行失败", exit_code=2,
                    native=self.noexport[unit] == "final_certificate_rejected", exported="e"*64,
                    wall_budget=dict(wall_seconds=args.wall_seconds))))
            else:
                (run/"status.json").write_text(json.dumps(dict(
                    status="level_failed" if failed else "budget_exhausted", wording="运行失败" if failed else "预算收尾",
                    exit_code=2 if failed else 0, native=not failed, exported=None if failed else "f"*64,
                    wall_budget=dict(wall_seconds=args.wall_seconds))))
            gc = args.jd_cell_estimate == "gauss27+corners"
            rounds = [dict(level=lv, round=0, status="converged_gradient" if gc and lv == 0 else "cap_reached",
                           accepted=100 if gc and lv == 0 else 500, commits=500, GF_B=2., T_B=1., GF_B_over_T_B=2.,
                           min_jd_lower_bound=.01, seconds=10.,
                           counters=dict(candidates=510, rejections=dict(JACOBIAN_VIOLATION=10)),
                           metric=dict(jd_scaling=dict(mode="affine_v1", cell_estimate=args.jd_cell_estimate,
                                                       jd_scaling_refresh=3, min_J_a=.02, max_fraction_r_lt_1=.05,
                                                       computations=[])))
                      for lv in range(3)]
            rounds.append(dict(rounds[-1], level=3, status="budget_exhausted", accepted=40, GF_B_over_T_B=50.))
            (run/"hyper.json").write_text(json.dumps(dict(rounds=rounds)))
        if log is not None:
            log.write_text(f"fake {unit}\n")
            exit_file.write_text("0\n")
        return 0


@pytest.fixture
def driver(q, dry, monkeypatch):
    sha = head()

    def prepare(self):
        self.code, self.code_sha = ROOT, sha
        self.allowlist = self.allowlist or str(dry/"dev_subjects.json")
    monkeypatch.setattr(q.ArmSweep, "prepare_code", prepare)
    monkeypatch.setattr(q.ArmSweep, "write_env", lambda self: None)

    def go(root, system, *extra, sweep=None):
        return q.main(["--root", str(root), *common(dry, sweep), *extra], system=system)
    return go


def runs_of(system):
    return [u for u in system.units if "-eval-" not in u]


def evals_of(system):
    return [u for u in system.units if "-eval-" in u]


# ---- queue end to end ------------------------------------------------------------------------------------------------

def test_two_arms_end_to_end_summary_and_restart(driver, dry, tmp_path):
    root = tmp_path/"m2v1"
    system = FakeSystem()
    assert driver(root, system) == 0, (root/"queue.log").read_text()[-3000:]
    assert system.units == ["tag-m2v1dry-d47-0901-0902", "tag-m2v1dry-eval-d47-0901-0902",
                            "tag-m2v1dry-dev-0901-0902", "tag-m2v1dry-eval-dev-0901-0902",
                            "tag-m2v1dry-dev-0902-0903", "tag-m2v1dry-eval-dev-0902-0903",
                            "tag-m2v1dry-dev-0903-0904", "tag-m2v1dry-eval-dev-0903-0904"]
    record = json.loads((root/"eta"/"eta_m2v1.json").read_text())
    source = dry/"sweep"/"eta_selected.json"
    assert record["source_sha256"] == q_sha(source) and (record["alpha"], record["nu"]) == (.01, 50.)
    assert record["tau"] == [1., .5, .5, .5] and record["model_code_changed"] == []
    assert record["configuration_check"]["dev"] == "equal to the sweep's"
    assert "gauss27+corners" in record["configuration_check"]["d47"]
    batch = json.loads((root/"batch.json").read_text())
    assert batch["order"] == ["d47", "dev"] and batch["arms"]["d47"]["pairs"] == [P1] and len(batch["pairs"]) == 3
    assert batch["arms"]["d47"]["configuration"]["jd_cell_estimate"] == "gauss27+corners"
    assert batch["arms"]["dev"]["configuration"]["jd_cell_estimate"] == "certificate"
    d47 = next((root/"launch").glob("run_d47-0901-0902_*.txt")).read_text()
    dev = next((root/"launch").glob("run_dev-0901-0902_*.txt")).read_text()
    for text, cell in ((d47, "gauss27+corners"), (dev, "certificate")):
        for piece in (f"--jd-cell-estimate {cell}", f'"jd_cell_estimate": "{cell}"', "--channel-source structure",
                      "--logvol-points gauss27+closed27", "--logvol-closed27-share 1", "--closed27-weights simpson",
                      "--structure-infer never", "--structure-registry-staging", "--wall-seconds 60", "--device cpu",
                      f"--eta-json {root/'eta'/'eta_m2v1.json'}", "--pair stage_L2R_val_v1/", "--nice=19",
                      "--property=MemoryMax=9G", "--property=MemorySwapMax=4G", "MemAvailable"):
            assert piece in text, (piece, cell)
    ev = next((root/"launch").glob("eval_dev-0902-0903_*.txt")).read_text()
    for piece in ("--property=MemoryMax=6G", "bayesreg.eval.cli", "--pairs-stage stage_L2R_val_v1", "--method bayesreg_m2v1",
                  "--arm dev", "--run-id m2v1_dev_0902-0903"):
        assert piece in ev, piece
    summary = json.loads((root/"summary.json").read_text())
    assert summary["counts"]["ran"] == 4 and summary["counts"]["not_run"] == 0
    assert summary["counts"]["evaluator_certificate_passed"] == 4 and summary["counts"]["args_mismatch"] == 0
    a0 = summary["comparisons"]["A0"]
    assert a0["seg35"]["n"] == 3 and a0["seg4"]["n"] == 3 and a0["seg35"]["positive"] == 3
    assert summary["reference_values"]["A0"][P2]["seg35"] == pytest.approx(.51)
    assert summary["comparisons"]["A4p"]["seg35"]["median"] == pytest.approx(
        summary["comparisons"]["A0"]["seg35"]["median"]-.005)
    flags = summary["d47"]["flags"]
    assert any("L0" in x for x in flags["lean_gc"]) and flags["verdict"].startswith("倾向")
    md = (root/"summary.md").read_text()
    for piece in ("seg35", "seg4", "A0", "A4′", "D47 对照臂", "收敛表", "不是等时比较", "预算收尾", "H2 冒烟 v2（参照） | 缺"):
        assert piece in md, piece
    # restart: nothing launched again, the summary recomputes equal
    again = FakeSystem()
    assert driver(root, again) == 0
    assert again.units == [] and "(unchanged)" in (root/"queue.log").read_text()


def q_sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_deadline_not_run_then_later_call(driver, tmp_path):
    root = tmp_path/"m2v1"
    system = FakeSystem(step=20*60.)                       # every unit takes 20 min on the fake clock
    deadline = (T0+timedelta(minutes=50)).astimezone().strftime("%Y-%m-%dT%H:%M")
    assert driver(root, system, "--no-start-after", deadline) == 0
    assert runs_of(system) == ["tag-m2v1dry-d47-0901-0902", "tag-m2v1dry-dev-0901-0902"]
    assert len(evals_of(system)) == 2                       # evaluations after the deadline still run
    log = (root/"queue.log").read_text()
    assert "not run (deadline" in log and "dev-0902-0903" in log
    summary = json.loads((root/"summary.json").read_text())
    assert summary["counts"]["not_run"] == 2 and summary["counts"]["failed_no_status"] == 0
    assert "未运行" in (root/"summary.md").read_text()
    later = FakeSystem()
    assert driver(root, later) == 0
    assert runs_of(later) == ["tag-m2v1dry-dev-0902-0903", "tag-m2v1dry-dev-0903-0904"]
    assert list(root.glob("summary_*.json"))                 # the earlier summary kept, a new one written


def test_deadline_reached_while_the_gate_waits(driver, tmp_path):
    root = tmp_path/"m2v1"

    def free(system):
        if system.slept >= 3:
            system.busy = []
    system = FakeSystem(busy=["tag-other.service"], on_sleep=free)
    deadline = (T0+timedelta(minutes=2)).astimezone().strftime("%Y-%m-%dT%H:%M:%S")
    assert driver(root, system, "--no-start-after", deadline, "--wait-minutes", "10") == 0
    assert system.units == []
    assert "reached (waited 2 min)" in (root/"queue.log").read_text()


def test_d47_last_runs_the_dev_batch_first(driver, tmp_path):
    root = tmp_path/"m2v1"
    system = FakeSystem()
    assert driver(root, system, "--d47-last") == 0
    assert runs_of(system) == ["tag-m2v1dry-dev-0901-0902", "tag-m2v1dry-dev-0902-0903", "tag-m2v1dry-dev-0903-0904",
                               "tag-m2v1dry-d47-0901-0902"]
    assert json.loads((root/"batch.json").read_text())["order"] == ["dev", "d47"]
    assert (root/"summary.json").is_file()
    other = FakeSystem()
    assert driver(root, other) == 8 and other.units == []      # another order on the same root: batch.json differs


def test_busy_gate_recovers_within_wait_minutes(driver, tmp_path):
    """--wait-minutes 30 (the plan's launch commands, M2V1 review B1): a unit busy for two polls delays, not stops."""
    def free(system):
        if system.slept >= 2:
            system.busy = []
    system = FakeSystem(busy=["tag-eta-eval-s3-st2-0026-0027.service"], on_sleep=free)
    root = tmp_path/"m2v1"
    assert driver(root, system, "--wait-minutes", "30", "--d47-last") == 0
    assert system.slept == 2 and len(runs_of(system)) == 4
    assert "waiting before run dev-0901-0902" in (root/"queue.log").read_text()


def test_killed_run_is_recorded_not_rerun_and_the_batch_goes_on(driver, tmp_path):
    root = tmp_path/"m2v1"
    system = FakeSystem(drop={"tag-m2v1dry-dev-0902-0903"})
    assert driver(root, system) == 0
    assert runs_of(system)[-1] == "tag-m2v1dry-dev-0903-0904" and len(evals_of(system)) == 3
    failure = json.loads((root/"failures"/"run_dev-0902-0903.json").read_text())
    assert "without status.json" in failure["reason"] and failure["systemd_exit"] == 137
    summary = json.loads((root/"summary.json").read_text())
    assert summary["counts"]["failed_no_status"] == 1 and summary["comparisons"]["A0"]["seg35"]["n"] == 2
    assert "n = 2 < 3" in summary["comparisons"]["A0"]["seg35"]["label"]
    again = FakeSystem()
    assert driver(root, again) == 0 and again.units == []


def test_runs_without_disp_are_failures_not_hard_constraint_violations(driver, tmp_path):
    """M2V1 review I1 (the reviewer's repro): stopped_before_native and final_certificate_rejected write the state hash
    into status.json "exported" but leave no disp.npz; the evaluator reports field_present false and no certificate.
    They count as runs without an export (failures), never as a hard-constraint-1 violation."""
    root = tmp_path/"m2v1"
    system = FakeSystem(noexport={"tag-m2v1dry-dev-0902-0903": "stopped_before_native",
                                  "tag-m2v1dry-dev-0903-0904": "final_certificate_rejected"})
    assert driver(root, system) == 0
    summary = json.loads((root/"summary.json").read_text())
    c = summary["counts"]
    assert (c["ran"], c["exported"], c["evaluator_certificate_passed"], c["ran_without_export"]) == (4, 2, 2, 2)
    assert c["hard_constraint_1"].startswith("通过")
    rows = {r["short"]: r for r in summary["rows"]["dev"]}
    assert rows["0902-0903"]["status_exported_hash"] == "e"*64 and rows["0902-0903"]["exported"] is False
    assert rows["0902-0903"]["comparable"] is False and summary["comparisons"]["A0"]["seg35"]["n"] == 1
    md = (root/"summary.md").read_text()
    assert "**违反**" not in md and "没有导出场的 2 次计为失败" in md
    # without an evaluation the run directory decides: no disp.npz -> not exported
    s = load("m2v1_summarize_direct", "experiments/m2v1/summarize.py")
    batch = json.loads((root/"batch.json").read_text())
    shutil.rmtree(root/"eval"/"dev"/"0903-0904")
    assert s.pair_row(root, "dev", P3, batch)["exported"] is False


def test_hard_constraint_1_stops_and_stops_again_on_restart(driver, tmp_path):
    root = tmp_path/"m2v1"
    system = FakeSystem(badcert={"m2v1_dev_0902-0903"})
    assert driver(root, system) == 8
    assert runs_of(system)[-1] == "tag-m2v1dry-dev-0902-0903" and evals_of(system)[-1] == "tag-m2v1dry-eval-dev-0902-0903"
    log = (root/"queue.log").read_text()
    assert "hard constraint 1" in log and (root/"eval"/"dev"/"0902-0903"/"done.json").is_file()
    again = FakeSystem()
    assert driver(root, again) == 8 and again.units == []          # re-checked from done.json, not skipped
    assert (root/"queue.log").read_text().count("hard constraint 1") == 2
    assert not (root/"summary.json").exists()


def test_consecutive_failures_stop(driver, tmp_path):
    root = tmp_path/"m2v1"
    system = FakeSystem(fail={"tag-m2v1dry-d47-0901-0902", "tag-m2v1dry-dev-0901-0902", "tag-m2v1dry-dev-0902-0903"})
    assert driver(root, system) == 8
    assert runs_of(system) == ["tag-m2v1dry-d47-0901-0902", "tag-m2v1dry-dev-0901-0902", "tag-m2v1dry-dev-0902-0903"]
    assert "3 failed runs in a row" in (root/"queue.log").read_text()


def test_config_mismatch_stops(driver, tmp_path):
    root = tmp_path/"m2v1"
    system = FakeSystem(mutate=dict(logvol_closed27_share=3.))
    assert driver(root, system) == 8
    assert system.units == ["tag-m2v1dry-d47-0901-0902"]
    assert "config.json args differ" in (root/"queue.log").read_text()


# ---- eta checks ------------------------------------------------------------------------------------------------------

def copy_sweep(dry, tmp_path, edit):
    sweep = tmp_path/"sweep"
    shutil.copytree(dry/"sweep", sweep)
    for name, change in edit.items():
        doc = json.loads((sweep/name).read_text())
        change(doc)
        (sweep/name).write_text(json.dumps(doc))
    return sweep


def test_eta_configuration_mismatch_stops_before_any_unit(driver, dry, tmp_path):
    sweep = copy_sweep(dry, tmp_path, {"eta_selected.json": lambda d: d["configuration"].update(
        logvol_closed27_share=3.)})
    system = FakeSystem()
    assert driver(tmp_path/"m2v1", system, sweep=sweep) == 8
    assert system.units == [] and "logvol_closed27_share" in (tmp_path/"m2v1"/"queue.log").read_text()


def test_eta_sweep_pairs_overlapping_the_batch_stop(driver, dry, tmp_path):
    sweep = copy_sweep(dry, tmp_path, {"summary.json": lambda d: d.update(pairs=[P2])})
    system = FakeSystem()
    assert driver(tmp_path/"m2v1", system, sweep=sweep) == 8
    assert system.units == [] and "overlap this batch" in (tmp_path/"m2v1"/"queue.log").read_text()


def test_eta_summary_disagreeing_with_the_file_stops(driver, dry, tmp_path):
    sweep = copy_sweep(dry, tmp_path, {"summary.json": lambda d: d["eta_selected"].update(nu=51.)})
    assert driver(tmp_path/"m2v1", FakeSystem(), sweep=sweep) == 8


def test_model_code_change_stops_unless_accepted(driver, dry, tmp_path):
    touched = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", "bayesreg"], capture_output=True,
                             text=True, check=True).stdout.strip()
    sweep = copy_sweep(dry, tmp_path, {"eta_selected.json": lambda d: d.update(code_sha=touched+"~1")})
    system = FakeSystem()
    assert driver(tmp_path/"a", system, "--until", "eta", sweep=sweep) == 8
    assert "the model code differs" in (tmp_path/"a"/"queue.log").read_text()
    assert driver(tmp_path/"b", FakeSystem(), "--until", "eta", "--accept-code-change", sweep=sweep) == 0
    record = json.loads((tmp_path/"b"/"eta"/"eta_m2v1.json").read_text())
    assert record["accept_code_change"] is True and any(p.startswith("bayesreg/") for p in record["model_code_changed"])


def test_changed_eta_file_on_restart_stops(driver, dry, tmp_path):
    sweep = copy_sweep(dry, tmp_path, {})
    root = tmp_path/"m2v1"
    assert driver(root, FakeSystem(), "--until", "eta", sweep=sweep) == 0
    for name in ("eta_selected.json", "summary.json"):
        doc = json.loads((sweep/name).read_text())
        (doc if name == "eta_selected.json" else doc["eta_selected"]).update(nu=60.)
        (sweep/name).write_text(json.dumps(doc))
    system = FakeSystem()
    assert driver(root, system, sweep=sweep) == 8
    assert system.units == [] and "differs from the recomputed record" in (root/"queue.log").read_text()


# ---- after-sweep -------------------------------------------------------------------------------------------------

def test_after_sweep_waits_for_the_selection_then_runs(driver, dry, dry_mod, tmp_path):
    waiting = tmp_path/"w"
    waiting.mkdir()
    sweep = waiting/"sweep"
    sweep.mkdir()
    (sweep/"sweep.log").write_text("2026-10-31T20:00:00Z gate run s3-st2-0905-0906: ...\n")

    def finish(system):
        if system.slept == 2:
            for name in ("eta_selected.json", "summary.json", "sweep.log"):
                shutil.copy(dry/"sweep"/name, sweep/name)
            system.eta = []
    system = FakeSystem(eta_units=["tag-eta-s3-st2-0905-0906.service"], on_sleep=finish)
    root = tmp_path/"m2v1"
    assert driver(root, system, "--after-sweep", "--until", "d47", sweep=sweep) == 0
    assert system.slept == 2 and runs_of(system) == ["tag-m2v1dry-d47-0901-0902"]
    log = (root/"queue.log").read_text()
    assert "after-sweep: waiting" in log and "going on" in log and "until d47: end" in log


def test_after_sweep_stop_does_nothing(driver, dry_mod, tmp_path, q):
    out = tmp_path/"stopdry"
    dry_mod.main(["--out", str(out), "--sweep-outcome", "stop", "--wall-seconds", "60", "--extra", ""])
    system = FakeSystem()
    root = tmp_path/"m2v1"
    assert driver(root, system, "--after-sweep", sweep=out/"sweep") == q.EXIT_SWEEP_STOPPED
    assert system.units == [] and not (root/"batch.json").exists() and not (root/"eta").exists()
    assert "ended with STOP" in (root/"queue.log").read_text()


def test_after_sweep_running_driver_and_wait_limit(driver, dry, tmp_path):
    system = FakeSystem(driver=[4242])                     # eta_selected.json present but the sweep still running
    root = tmp_path/"m2v1"
    assert driver(root, system, "--after-sweep", "--sweep-wait-minutes", "3") == 8
    assert system.slept == 3 and system.units == [] and "has not ended after 3 min" in (root/"queue.log").read_text()


# ---- gate, paths, precheck -----------------------------------------------------------------------------------------

def test_busy_gate_stops_and_training_is_not_excepted(driver, tmp_path, q):
    for busy in ("tag-eta-s1-x1-0012-0013.service", "tag-unet35-train.service"):
        system = FakeSystem(busy=[busy])
        root = tmp_path/busy
        assert driver(root, system) == 8
        assert system.units == [] and busy in (root/"queue.log").read_text()
    # the real system excepts only its own unit
    real = q.QueueSystem()
    real.self_unit = "tag-m2v1-queue.service"
    real._out = lambda cmd: "tag-m2v1-queue.service loaded active running x\ntag-unet35-train.service loaded active running y\n"
    assert real.other_units() == ["tag-unet35-train.service"]


def test_dry_run_and_real_mode_path_refusals(q, dry, tmp_path, monkeypatch):
    assert q.main(["--root", str(tmp_path/"r"), *common(dry), "--sweep-root",
                   "/home/rmcfz/seq_vp_tag_results/eta_sweep_v3"], system=FakeSystem()) == 2
    assert q.main(["--root", "/home/rmcfz/seq_vp_tag_results/eta_sweep_v3/x", *common(dry)], system=FakeSystem()) == 2
    monkeypatch.setattr(q.ArmSweep, "write_env", lambda self: None)
    real = ["--root", str(tmp_path/"real"), "--sweep-root", str(dry/"sweep")]
    assert q.main(real+["--segmenter-registry", str(dry/"registry"/"segmenters.json")], system=FakeSystem()) == 8
    assert "dry runs only" in (tmp_path/"real"/"queue.log").read_text()
    assert q.main(["--root", str(tmp_path/"real2"), "--sweep-root", str(dry/"sweep"), "--d47-pair", P1],
                  system=FakeSystem()) == 8
    with pytest.raises(SystemExit):
        q.parse_args(["--extra=--jd-cell"])                # a prefix of a managed option


def test_precheck_missing_prediction_and_pool(driver, dry, tmp_path, q, monkeypatch):
    structure = tmp_path/"structure"
    shutil.copytree(dry/"structure", structure)
    shutil.rmtree(structure/"OASIS_OAS1_0903_MR1")
    system = FakeSystem()
    root = tmp_path/"a"
    args = common(dry)
    args[args.index("--structure-root")+1] = str(structure)
    assert q.main(["--root", str(root), *args], system=system) == 8
    assert system.units == [] and "OASIS_OAS1_0903_MR1" in (root/"queue.log").read_text()
    train = tmp_path/"train.txt"
    train.write_text("OASIS_OAS1_0904_MR1\n")
    monkeypatch.setattr(q, "TRAIN_LIST", str(train))
    assert driver(tmp_path/"b", FakeSystem(), "--until", "precheck") == 8
    assert "training subject" in (tmp_path/"b"/"queue.log").read_text()


def test_one_driver_per_root(driver, tmp_path):
    import fcntl
    root = tmp_path/"m2v1"
    root.mkdir()
    with open(root/"driver.lock", "a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)        # another driver (open file description) holds it
        system = FakeSystem()
        assert driver(root, system) == 2
        assert system.units == [] and not (root/"queue.log").exists()
    system = FakeSystem()
    assert driver(root, system, "--until", "precheck") == 0     # released: the next call runs
    assert "pid " in (root/"driver.lock").read_text()


def test_until_precheck_launches_nothing(driver, tmp_path):
    system = FakeSystem()
    assert driver(tmp_path/"m2v1", system, "--until", "precheck") == 0
    assert system.units == [] and not (tmp_path/"m2v1"/"eta").exists()


def test_summary_reference_pairs_must_match(driver, dry, tmp_path):
    a4 = tmp_path/"a4.csv"
    a4.write_text("".join(line for line in (dry/"a4"/"long.csv").read_text().splitlines(True)
                          if "0903_MR1__OASIS_OAS1_0904" not in line or ",A4p," not in line))
    root = tmp_path/"m2v1"
    assert driver(root, FakeSystem(), "--a4-long", str(a4)) == 8
    assert "pairs of" in (root/"queue.log").read_text() and not (root/"summary.json").exists()


def test_summary_uses_the_reevaluation_and_checks_it(driver, dry, tmp_path):
    root = tmp_path/"m2v1"
    assert driver(root, FakeSystem()) == 0
    summary = json.loads((root/"summary.json").read_text())
    assert summary["references"]["A0"]["source"].startswith("bayesreg.eval.cli")
    assert summary["reference_values"]["A4p"][P1]["route"] == "disp_from_final_native"
    assert summary["original_reference_values"]["A0"][P1]["seg35"] == summary["reference_values"]["A0"][P1]["seg35"]
    md = (root/"summary.md").read_text()
    assert "参照：重评与原数字" in md and "逐位相同" in md and "原数字（experiments/d0/evaluate.py）只作对照列" in md
    # a changed re-evaluated value is what the comparison uses (the original stays a reference column)
    reeval = tmp_path/"reeval"
    shutil.copytree(dry/"ref_reeval", reeval)
    path = next((reeval/"A0"/"0901-0902").glob("*/metrics.json"))
    doc = json.loads(path.read_text())
    doc["metrics"]["seg35"] = .4
    path.write_text(json.dumps(doc))
    root2 = tmp_path/"m2v1b"
    assert driver(root2, FakeSystem(), "--ref-reeval", str(reeval)) == 0
    s2 = json.loads((root2/"summary.json").read_text())
    per = {x["pair"]: x for x in s2["comparisons"]["A0"]["seg35"]["per_pair"]}
    assert per[P1]["reference"] == .4 and s2["original_reference_values"]["A0"][P1]["seg35"] == .5
    assert ".4000 / 0.5000" in (root2/"summary.md").read_text()
    # a missing re-evaluation, or one of another run id, stops the summary
    for change in ("missing", "run_id"):
        bad = tmp_path/f"bad_{change}"
        shutil.copytree(dry/"ref_reeval", bad)
        ev = bad/"A4p"/"0902-0903"
        if change == "missing":
            (ev/"done.json").unlink()
        else:
            path = next(ev.glob("*/metrics.json"))
            doc = json.loads(path.read_text())
            doc["run_id"] = "another"
            path.write_text(json.dumps(doc))
        r = tmp_path/f"m2v1_{change}"
        assert driver(r, FakeSystem(), "--ref-reeval", str(bad)) == 8
        assert "re-evaluation" in (r/"queue.log").read_text() and not (r/"summary.json").exists()


def test_dry_run_requires_a_synthetic_reevaluation(q, dry, tmp_path):
    args = common(dry)
    k = args.index("--ref-reeval")
    del args[k:k+2]
    assert q.main(["--root", str(tmp_path/"r"), *args], system=FakeSystem()) == 2
    assert q.main(["--root", str(tmp_path/"r2"), *args, "--ref-reeval",
                   "/home/rmcfz/seq_vp_tag_results/m2v1_dev/ref_reeval"], system=FakeSystem()) == 2


# ---- level table, summary pieces, deadline parser -----------------------------------------------------------------

def test_level_table_fake_and_killed_runs(lt, tmp_path):
    run = tmp_path/"run"
    run.mkdir()
    rounds = [dict(level=0, round=0, status="converged_gradient", accepted=7, GF_B=.5, T_B=1., GF_B_over_T_B=.5,
                   min_jd_lower_bound=.2, seconds=1., counters=dict(rejections=dict(JACOBIAN_VIOLATION=2,
                                                                                    NO_STRICT_DECREASE=1)),
                   metric=dict(jd_scaling=dict(jd_scaling_refresh=1, computations=[
                       dict(min_J_a=.3, fraction_r_lt_1=.01), dict(min_J_a=.1, fraction_r_lt_1=.04)]))),
              dict(level=1, round=0, status="line_search_failed", accepted=3, GF_B=9., T_B=1., min_jd_lower_bound=1e-7,
                   counters={}, metric={})]
    (run/"hyper.json").write_text(json.dumps(dict(rounds=rounds)))
    (run/"status.json").write_text(json.dumps(dict(status="line_search_failed", wording="受阻收尾", exit_code=0)))
    (run/"config.json").write_text(json.dumps(dict(args=dict(pair="s/p", jd_cell_estimate="certificate"),
                                                   eta=dict(nu=5.))))
    rec = lt.run_record(run)
    l0, l1 = rec["levels"]
    assert (l0["ending"], l0["reached_threshold"], l0["rejected"], l0["min_J_a"], l0["max_fraction_r_lt_1"]) == \
        ("达阈", True, 3, .1, .04)
    assert l1["ending"] == "受阻收尾" and l1["GF_B_over_T_B"] == 9. and rec["n_blocked"] == 1
    killed = tmp_path/"killed"
    killed.mkdir()
    rows = [dict(rounds[0], kind="round"), dict(kind="jd_scaling", level=1, round=0, min_J_a=.05, fraction_r_lt_1=.02),
            dict(kind="accept", level=1, round=0, min_jd_lower_bound=.03), dict(kind="accept", level=1, round=0,
                                                                                min_jd_lower_bound=.02)]
    (killed/"log.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows)+'{"cut by a kill')
    rec = lt.run_record(killed)
    assert [r["source"] for r in rec["levels"]] == ["log.jsonl (no hyper.json)", "log.jsonl (unfinished: no round record)"]
    assert rec["levels"][1]["accepted"] == 2 and rec["levels"][1]["min_jd_lower_bound"] == .02
    assert lt.main([str(run), str(killed), "--out-md", str(tmp_path/"t.md"), "--out-json", str(tmp_path/"t.json")]) == 0
    assert "受阻收尾" in (tmp_path/"t.md").read_text()
    with pytest.raises(SystemExit):
        lt.main([str(run), "--out-md", str(run/"inside.md")])
    sweep = tmp_path/"sweep"
    shutil.copytree(run, sweep/"runs"/"s1"/"x1"/"0012-0013")
    assert [label for _, label in lt.sweep_runs(sweep)] == ["s1-x1-0012-0013"]


def test_d47_flags_and_reading_labels(q):
    s = q.summarize
    level = (lambda lv, ending, acc, ratio: dict(level=lv, round=0, ending=ending, accepted=acc, GF_B_over_T_B=ratio))
    row = (lambda levels, blocked=(), lb=.1: dict(state="ran", comparable=True, run_status="budget_exhausted",
                                                  field_role="native_formal", certificate=dict(min_jd_lower_bound=lb),
                                                  levels=dict(levels=levels, n_blocked=len(blocked), blocked=list(blocked))))
    cert = row([level(0, "上限收尾", 500, 2e4), level(3, "预算收尾", 40, 7e3)])
    gc = row([level(0, "达阈", 120, .9), level(3, "预算收尾", 70, 500.)])
    flags = s.d47_flags(gc, cert)
    assert flags["verdict"].startswith("倾向") and len(flags["lean_gc"]) == 3
    blocked = row([level(0, "达阈", 120, .9)], blocked=[dict(level=2, round=0, status="line_search_failed")])
    assert s.d47_flags(blocked, cert)["verdict"] == "支持维持 D47"
    assert s.d47_flags(row([level(3, "预算收尾", 41, 6e3)]), cert)["verdict"] == "无法区分（只描述）"
    assert s.d47_flags(row([level(0, "达阈", 120, .9)], lb=1e-8), cert)["verdict"] == "支持维持 D47"
    stats = (lambda med, pos, n=19: dict(n=n, median=med, positive=pos))
    assert s.reading_label(stats(.012, 14), 19).startswith("高于")
    assert s.reading_label(stats(.012, 13), 19).startswith("中间")
    assert s.reading_label(stats(-.02, 5), 19).startswith("低于")
    assert s.reading_label(stats(-.02, 6), 19).startswith("中间")
    assert s.reading_label(stats(.0099, 19), 19).startswith("中间")
    # fewer comparable pairs (the lead, 10-31, preregistered): n >= 15 of 19 -> thresholds ceil(14n/19), floor(5n/19)
    assert s.min_pairs(19) == 15 and s.min_pairs(3) == 3
    assert s.reading_label(stats(.05, 14, n=18), 19).startswith("高于")          # ceil(14*18/19) = 14
    assert s.reading_label(stats(.05, 13, n=18), 19).startswith("中间")
    assert s.reading_label(stats(.05, 12, n=15), 19).startswith("高于")          # ceil(14*15/19) = 12
    assert s.reading_label(stats(.05, 11, n=15), 19).startswith("中间")
    assert s.reading_label(stats(-.05, 3, n=15), 19).startswith("低于")          # floor(5*15/19) = 3
    assert s.reading_label(stats(-.05, 4, n=15), 19).startswith("中间")
    assert s.reading_label(stats(-.05, 4, n=17), 19).startswith("低于")          # floor(5*17/19) = 4
    assert s.reading_label(stats(.05, 14, n=14), 19) == "n = 14 < 15：只描述"


def test_real_mode_run_arguments_are_the_sweeps(q, tmp_path):
    """The dev arm passes run_pair the sweep's own argument list (experiments/eta/sweep.py run_args at its defaults with
    beta_S 1, as eta_sweep_v3 was started) apart from the stage of --pair and the eta file; the d47 arm differs only in
    --jd-cell-estimate. Real-mode objects, nothing launched."""
    sweep = q.eta.Sweep(q.eta.parse_args(["--root", str(tmp_path/"s"), "--logvol-closed27-share", "1"]), FakeSystem())
    opts = q.parse_args(["--root", str(tmp_path/"m")])
    batch = q.Queue(opts, FakeSystem())
    for obj in (sweep, *batch.arms.values()):
        obj.allowlist = "/code/configs/bayesreg/dev_subjects.json"
    pair = "OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1"
    want = sweep.run_args(pair, "/eta.json")
    want[want.index("--pair")+1] = f"stage_L2R_val_v1/{pair}"
    assert batch.arms["dev"].run_args(pair, "/eta.json") == want
    d47 = batch.arms["d47"].run_args(pair, "/eta.json")
    assert [i for i, (a, b) in enumerate(zip(d47, want)) if a != b] == [want.index("--jd-cell-estimate")+1]
    assert d47[want.index("--jd-cell-estimate")+1] == "gauss27+corners" and len(d47) == len(want)
    assert sweep.configuration() == dict(batch.arms["dev"].configuration())


def fake_proc(root, processes):
    """A /proc-like tree: {pid: (ppid, argv)} -> <root>/<pid>/{cmdline, stat}."""
    for pid, (ppid, argv) in processes.items():
        (root/str(pid)).mkdir(parents=True)
        (root/str(pid)/"cmdline").write_bytes(b"\0".join(a.encode() for a in argv)+b"\0")
        (root/str(pid)/"stat").write_text(f"{pid} (x y) S {ppid} {pid} {pid} 0 -1\n")
    (root/"self").mkdir()                                          # non-numeric entries are ignored
    return root


def test_sweep_process_matching(q, tmp_path):
    """M2V1 review I2: only a python process with an argument ending in experiments/eta/sweep.py is a sweep driver; the
    bash shell that launched the sweep (its command line names the script), other scripts, and this process's own
    ancestors are not."""
    py = "/home/rmcfz/miniconda3/envs/sci310/bin/python"
    sweep = [py, "-B", "experiments/eta/sweep.py", "--root", "/r/eta_sweep_v3", "--wait-minutes", "30"]
    proc = fake_proc(tmp_path/"proc", {
        1: (0, ["/sbin/init"]),
        10: (1, ["/bin/bash", "-c", "setsid nohup " + " ".join(sweep) + " > log 2>&1 & disown; sleep 1; date"]),
        11: (10, sweep),                                                     # the sweep driver: counted
        12: (1, ["python3.10", "/abs/repo/experiments/eta/sweep.py"]),     # another python sweep: counted
        13: (1, [py, "-B", "experiments/eta/sweep.py.bak"]),               # not ending in sweep.py
        14: (1, ["/usr/bin/tail", "-f", "/r/eta_sweep_v3/sweep.log", "experiments/eta/sweep.py"]),
        20: (1, ["/bin/bash", "-c", "pgrep -f experiments/eta/sweep.py; " + py + " run_batch.py"]),
        21: (20, [py, "-B", "experiments/m2v1/run_batch.py", "--after-sweep"]),   # this driver
        30: (1, [py, "-c", "x", "experiments/eta/sweep.py"]),               # python ancestor below
        31: (30, [py, "-B", "experiments/m2v1/run_batch.py"]),
    })
    assert q.sweep_processes(proc, 21) == [11, 12, 30]
    assert q.sweep_processes(proc, 31) == [11, 12]                # its own python ancestor 30 is excluded
    assert q.sweep_processes(proc, 999) == [11, 12, 30]           # unknown self pid: no ancestors, still no crash
    # the real /proc: this test process is no sweep driver and the call does not fail
    assert os.getpid() not in q.QueueSystem().sweep_driver_running()


def test_parse_deadline(q):
    local = datetime(2026, 10, 31, 9, 0).astimezone()
    got = q.parse_deadline("21:50", local)
    assert got.tzinfo is not None and got.astimezone(local.tzinfo).strftime("%Y-%m-%d %H:%M") == "2026-10-31 21:50"
    assert q.parse_deadline("2026-10-31T21:50:00-05:00") == datetime(2026, 11, 1, 2, 50, tzinfo=timezone.utc)
    assert q.parse_deadline(None) is None
    with pytest.raises(SystemExit):
        q.parse_args(["--no-start-after", "late"])


# ---- the dry run in process (slow) ----------------------------------------------------------------------------------

class InlineSystem(FakeSystem):
    """Runs the units' commands in this process instead of systemd: bayesreg.cli.run_pair and bayesreg.eval.cli with
    the exact argument lists the queue built (CPU torch64, CUDA hidden by the caller)."""

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
def test_dry_run_in_process(q, dry_mod, tmp_path, monkeypatch):
    out = tmp_path/"dry"
    dry_mod.main(["--out", str(out), "--wall-seconds", "120", "--extra", "--max-accepted 2", "--code-sha", head()])
    sha = head()

    def prepare(self):
        self.code, self.code_sha = ROOT, sha
        self.allowlist = self.allowlist or str(out/"dev_subjects.json")
        self.check_code_tree()
    monkeypatch.setattr(q.ArmSweep, "prepare_code", prepare)
    monkeypatch.setattr(q.ArmSweep, "write_env", lambda self: None)
    root = tmp_path/"m2v1"
    args = ["--root", str(root), *common(out), "--wall-seconds", "120", "--extra", "--max-accepted 2", "--after-sweep",
            "--no-start-after", "23:59"]
    system = InlineSystem(clock=datetime.now(timezone.utc))
    assert q.main(args, system=system) == 0, (root/"queue.log").read_text()[-4000:]
    assert len(runs_of(system)) == 4 and len(evals_of(system)) == 4
    for config in (root/"runs").glob("*/*/config.json"):
        a = json.loads(config.read_text())["args"]
        arm = config.parent.parent.name
        assert (a["channel_source"], a["logvol_points"], a["logvol_closed27_share"], a["structure_infer"],
                a["structure_registry_staging"], a["jd_cell_estimate"]) == (
            "structure", "gauss27+closed27", 1., "never", True, "gauss27+corners" if arm == "d47" else "certificate")
        prov = json.loads((config.parent/"provenance.json").read_text())
        assert prov["channels"]["K_S"] == 13 and prov["switches"]["jd_cell_estimate"] == a["jd_cell_estimate"]
    summary = json.loads((root/"summary.json").read_text())
    rows = [r for arm in summary["rows"].values() for r in arm]
    assert all(r["field_role"] == "native_formal" and r["certificate"]["passed"] is True for r in rows)
    assert summary["counts"]["args_mismatch"] == 0 and summary["counts"]["eta_mismatch"] == 0
    assert summary["comparisons"]["A0"]["seg35"]["n"] == 3 and summary["d47"]["gauss27_corners"]["levels"]["levels"]
    md = (root/"summary.md").read_text()
    assert "seg35" in md and "seg4" in md and "D47 各层读数" in md
    again = InlineSystem(clock=datetime.now(timezone.utc))
    assert q.main(args, system=again) == 0 and again.units == []
    assert re.search(r"summary: .*\(unchanged\)", (root/"queue.log").read_text())
