"""SPD native-level profiler (experiments/spd/profile_native.py), on synthetic inputs only (no data/, no GPU):

- statistics, the gate's nvidia-smi parsing and busy rule, the iteration segmentation;
- the timing proxies are value-neutral: run_pg through TimedBackend + TimedJDScaledMetric gives bitwise the same
  trajectory (accepted thetas, energies, log rows) as the plain backend and metric, on numpy and torch-cpu;
- end to end on a synthetic pair (experiments/m2/dry_inputs.py: synthetic subjects, allowlist and Atropos-like
  posteriors under the test's tmp dir) and a fabricated run record: host, torch-cpu and numpy workers in series,
  CUDA hidden, every item timed, profile.json / profile.md written, trial.npz removed, the run record untouched.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from jd_curl.three_d.boundary import restrict_free

from bayesreg.solve import DirectObjective, JDScaledMetric, NumpyBackend, PGConfig, run_pg
from bayesreg.solve.stopping import state_hash

from . import _synthetic as syn
from .test_pg import small_state

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT/"experiments"/"spd"/"profile_native.py"
DRY = ROOT/"experiments"/"m2"/"dry_inputs.py"
PAIR = "stage_L2R_val_v1/OASIS_OAS1_0901_MR1__OASIS_OAS1_0902_MR1"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def spd():
    return load(TOOL, "profile_native")


# ---- small helpers --------------------------------------------------------------------------------------

def test_summarize(spd):
    s = spd.summarize([4., 1., 3., 2., 5.])
    assert (s["n"], s["median"], s["q25"], s["q75"], s["iqr"], s["min"], s["max"]) == (5, 3., 2., 4., 2., 1., 5.)
    assert spd.summarize([]) is None


def test_gpu_query_and_busy_rule(spd):
    rows = spd.parse_gpu_query("NVIDIA GeForce RTX 4090, 560.94, 10648, 24564, 92\n"
                               "NVIDIA GeForce RTX 4090, 560.94, 412, 24564, 0\n")
    assert rows[0]["memory_used_mib"] == 10648. and rows[0]["utilization_pct"] == 92.
    assert spd.gpu_busy(rows[1:], []) == []
    reasons = spd.gpu_busy(rows[:1], [])
    assert any("utilization" in r for r in reasons) and any("memory.used" in r for r in reasons)
    assert spd.gpu_busy(rows[1:], ["123, python, 900 MiB"]) == ["1 compute process(es) listed"]
    assert spd.gpu_busy(rows[:1], [], util=95, mib=11000) == []          # --gpu-busy-util / --gpu-busy-mib
    assert spd.parse_gpu_query("garbage\n") == []


def test_segment_iterations(spd):
    rows = [dict(t=1., kind="candidate", iteration=1, reason="JACOBIAN_VIOLATION", rule="initial"),
            dict(t=2., kind="candidate", iteration=1, reason=None, rule="halving"),
            dict(t=3., kind="accept", iteration=1, tau=.5),
            dict(t=5., kind="candidate", iteration=2, reason=None, rule="bb2"),
            dict(t=6., kind="accept", iteration=2, tau=.25)]
    events = [("certify", .1, .4), ("certify", 1.1, 1.5), ("evaluate", 1.5, 1.7), ("gradient", 2.1, 2.9),
              ("metric_solve", 3.2, 3.4), ("certify", 4., 4.5)]
    out = spd.segment_iterations(0., rows, events)
    assert [it["index"] for it in out] == [1, 2] and [it["wall"] for it in out] == [3., 3.]
    first = out[0]
    assert first["candidates"] == 2 and first["reasons"] == {"JACOBIAN_VIOLATION": 1, "None": 1}
    assert first["components"]["certify"] == pytest.approx(.7) and first["calls"]["certify"] == 2
    assert first["host_other"] == pytest.approx(3.-.7-.2-.8)
    assert out[1]["components"]["metric_solve"] == pytest.approx(.2) and out[1]["rule"] == "bb2"
    summary = spd.summarize_iterations([out, out])
    assert summary["first"]["wall"]["n"] == 2 and summary["bb2"]["components"]["certify"]["median"] == pytest.approx(.5)


# ---- the proxies are value-neutral ------------------------------------------------------------------------

def _trajectory(backend, state, metric_factory, *, steps=3):
    zero = np.zeros(state.grid.shape+(3,))
    base = backend.evaluate(state, zero)
    cert = backend.certify(zero, state.grid)
    metric = backend.build_metric(state, c_dat=backend.c_dat(state, base), b_P=backend.metric_bP(state))
    rows = []
    result = run_pg(DirectObjective(backend, state), base, theta=restrict_free(zero, state.grid), cert=cert,
                    metric=metric_factory(metric, zero), certify=backend.certify, target=0., max_steps=steps,
                    config=PGConfig(), log=rows.append)
    clean = [{k: v for k, v in r.items() if k not in ("comparison",)} for r in rows]
    return result, clean


@pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
def test_timing_proxies_are_bitwise_neutral(spd, backend_name):
    # DEF: the model and the cell estimate the SPD profiler was written for (the defaults before DEF), explicit;
    # the profiler itself uses spd.SPD_JD ("gauss27+corners")
    state = small_state(9, logvol_points="gauss27")
    cfg = PGConfig(jd=spd.SPD_JD)
    assert spd.SPD_JD.cell_estimate == "gauss27+corners"

    def make():
        if backend_name == "numpy":
            return NumpyBackend()
        from bayesreg.solve import make_backend
        return make_backend("torch64", device="cpu")
    plain_backend = make()
    plain, plain_rows = _trajectory(plain_backend, state,
                                    lambda m, u: JDScaledMetric(m, state.grid, u, config=cfg.jd))
    clock = spd.CallClock(lambda: None)
    timed_backend = spd.TimedBackend(make(), clock)
    timed, timed_rows = _trajectory(timed_backend, state,
                                    lambda m, u: spd.TimedJDScaledMetric(m, state.grid, u, clock=clock, config=cfg.jd))
    assert timed.status == plain.status and timed.accepted == plain.accepted >= 1
    assert timed.theta.tobytes() == plain.theta.tobytes()
    assert [float(e).hex() for e in timed.energies] == [float(e).hex() for e in plain.energies]
    assert timed.gradient.tobytes() == plain.gradient.tobytes()
    assert timed_rows == plain_rows
    names = {name for name, _, _ in clock.events}
    assert {"certify", "evaluate", "gradient", "metric_solve"} <= names
    assert ("stable_compare" in names) == (backend_name != "numpy")
    assert ("stable_reduction" in names) == (backend_name == "numpy")
    assert all(b >= a for _, a, b in clock.events)
    # a missing backend attribute stays missing behind the proxy (DirectObjective's getattr(..., None))
    assert (getattr(timed_backend, "stable_compare", None) is None) == (backend_name == "numpy")


# ---- end to end on synthetic inputs ------------------------------------------------------------------------

N = 26                  # four_level_grids needs >= 5 nodes per axis on the coarsest level


def fabricate(tmp):
    """Synthetic pair, allowlist and tissue posteriors (dry_inputs.py) + a fabricated run record."""
    inputs = tmp/"inputs"
    done = subprocess.run([sys.executable, "-B", str(DRY), "--out", str(inputs), "--n", str(N)], capture_output=True,
                          text=True, cwd=str(ROOT))
    assert done.returncode == 0, done.stdout+done.stderr
    run = tmp/"run"
    run.mkdir()
    grid = syn.cube_grid(N)
    u = syn.solenoidal_displacement(grid, .3)
    np.savez(run/"final_native.npz", u=u, state_hash=np.array(state_hash(u)), status=np.array("budget_exhausted"))
    args = dict(pair=PAIR, channel_source="none", logvol_points="gauss27+corners", metric_jd_scaling="affine_v1",
                cpu_threads=1, data_root=str(inputs/"data"), allowlist=str(inputs/"dev_subjects.json"),
                tissue_root=str(inputs/"seg"), backend="torch64", device="cuda")
    (run/"config.json").write_text(json.dumps(dict(
        args=args, eta=dict(alpha=.01, nu=3.87, tau2=[1., .25, .25, .25], source="test"),
        solver=dict(gauss_tile_size=1024))), encoding="utf-8")
    (run/"hyper.json").write_text(json.dumps(dict(rounds=[
        dict(level=2, round=0, sigma2_used=[.05]),
        dict(level=3, round=0, sigma2_used=[.03], T_B=1e-9, G0=1., accepted=3, status="budget_exhausted",
             counters=dict(candidates=5, certificate_calls=5, energy_evaluations=3, gradients=4, metric_solves=4,
                           rejections=dict(JACOBIAN_VIOLATION=2)),
             metric=dict(c_dat=.2, b_P=3.9, jd_scaling=dict(computations=[dict(at="round_start", seconds=.1)],
                                                             seconds=.1)))],
        gpu_comparison=dict(comparisons=3, cpu_fallback=0, seconds=dict(gpu=1., cpu_fallback=0.)))),
        encoding="utf-8")
    (run/"timing.json").write_text(json.dumps(dict(solve_levels=dict(levels=[dict(level=3, seconds=12.)]))),
                                   encoding="utf-8")
    rows = []
    for it, tau in ((1, .05), (2, .04), (3, .03)):
        if it < 3:
            rows.append(dict(level=3, kind="candidate", iteration=it, tau=2*tau, cert_status="infeasible",
                             reason="JACOBIAN_VIOLATION", energy=None))
        rows.append(dict(level=3, kind="candidate", iteration=it, tau=tau, cert_status="certified_feasible",
                         reason=None, energy=1., comparison=dict(seconds_gpu=.3, seconds_cpu=0., fallback=False)))
        rows.append(dict(level=3, kind="accept", iteration=it, tau=tau))
    (run/"log.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    return inputs, run


def digest_tree(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.rglob("*"))
            if p.is_file()}


def run_tool(tmp_path, paths, *, hide_cuda=True, extra=()):
    inputs, run = fabricate(tmp_path)
    before = digest_tree(run)
    out = tmp_path/"profile"
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    if hide_cuda:
        env["CUDA_VISIBLE_DEVICES"] = ""
    command = [sys.executable, "-B", str(TOOL), "--out", str(out), "--run-dir", str(run), "--paths", paths,
               "--repeats", "2", "--warmup", "1", "--iteration-steps", "2", "--skip-gate", "--allow-dirty", *extra]
    done = subprocess.run(command, capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=900)
    logs = "".join((out/f"worker_{p}.log").read_text() for p in ("host", "cuda", "torch-cpu", "numpy")
                   if (out/f"worker_{p}.log").exists())
    assert done.returncode == 0, done.stdout+done.stderr+logs
    assert digest_tree(run) == before                                   # the run record is only read
    return out, json.loads((out/"profile.json").read_text())


@pytest.mark.skipif(os.environ.get("SPD_TEST_CUDA") != "1",
                    reason="cuda dry run only on request (SPD_TEST_CUDA=1, GPU idle; runs/20261030-spd/plan.md step 0)")
def test_end_to_end_synthetic_cuda(tmp_path):
    """The cuda branch (synchronize, device memory, sync count, profiler, fallback items) on the synthetic pair."""
    out, profile = run_tool(tmp_path, "cuda", hide_cuda=False, extra=["--torch-profiler", "--with-fallback-items"])
    doc = profile["paths"]["cuda"]
    assert doc["status"] == "done", doc.get("error")
    assert "synchronize" in doc["sync"] and doc["torch"]["gpu"]
    for name in ("energy.evaluate", "energy.evaluate_device", "gradient.total", "metric.jd_solve", "compare.gpu",
                 "compare.device_gauss_changes", "compare.cpu_fallback_cold", "compare.cpu_fallback_warm",
                 "transfer.upload_u", "transfer.download_free"):
        item = doc["items"][name]
        assert item["error"] is None and item["n"] == 2 and item["device_peak_allocated_mib"] >= 0, name
        assert item["sync_events"]["count"] is not None, name
    assert doc["items"]["transfer.upload_u"]["transfers_per_call"]["upload_calls"] == 1
    assert {c["path"] for c in doc["items"]["compare.gpu"]["per_call"]} == {"gpu"}
    assert doc["iteration"]["device_peak_allocated_mib"] > 0 and not doc["iteration"]["errors"]
    assert "GPU 计时" in (out/"profile.md").read_text()


def test_end_to_end_synthetic(tmp_path):
    out, profile = run_tool(tmp_path, "torch-cpu,numpy",
                            extra=["--no-gpu-query", "--cert-thread-sweep", "1", "--numba-threads", "1"])
    assert not (out/"trial.npz").exists()                               # removed without --keep-trial
    for name in ("cmd.txt", "git.txt", "env.txt", "data.txt", "config.json", "profile.json", "profile.md"):
        assert (out/name).is_file(), name
    assert profile["schema"] == "spd.profile_native.v1"
    assert [w["path"] for w in profile["workers"]] == ["host", "torch-cpu", "numpy"]
    assert all(w["cuda_visible_devices"] == "" for w in profile["workers"])
    host = profile["paths"]["host"]
    assert host["status"] == "done"
    assert host["inputs"]["fixed_subject"] == "OASIS_OAS1_0901_MR1" and host["inputs"]["grid_shape_zyx"] == [N]*3
    for name in ("certificate.pass", "certificate.reject", "jd.scaling_factors", "pg.bb2_proposal", "pg.domain_bound",
                 "pg.inject_free", "pg.state_hash", "certificate.pass@threads=1"):
        item = host["items"][name]
        assert item["error"] is None and item["n"] == 2 and len(item["warmup_samples"]) == 1, name
    assert host["host"]["cert_thread_sweep"] == [dict(threads=1, identical_outcome=True)]
    assert host["env"]["numba"]["threads"] == 1 and profile["config"]["numba_threads"] == 1
    assert host["items"]["pg.bb2_proposal"]["per_call"][0]["rule"] == "bb2"
    trial = profile["trial"]
    assert trial["rejected_tau"] is not None and trial["tau"] < trial["rejected_tau"] and trial["cpu_ared"] is not None
    expected = ["energy.data", "energy.geometry_gauss", "energy.diffusion", "energy.logvol_gauss", "energy.curl",
                "energy.geometry_corners", "energy.logvol_corners", "kq.apply", "energy.evaluate", "gradient.channel",
                "gradient.diffusion", "gradient.logvol", "gradient.curl", "gradient.total", "metric.c_dat",
                "metric.build", "metric.solve", "metric.jd_solve"]
    for path, extra in (("torch-cpu", ["energy.data_device", "energy.evaluate_device", "gradient.total_device",
                                       "metric.solve_device", "compare.gpu", "compare.device_gauss_changes",
                                       "compare.channel_host", "compare.decision"]),
                        ("numpy", ["compare.cpu_reference"])):
        doc = profile["paths"][path]
        assert doc["status"] == "done", doc.get("error")
        for name in expected+extra:
            item = doc["items"][name]
            assert item["error"] is None and item["n"] == 2 and item["median"] > 0, (path, name, item.get("error"))
        it = doc["iteration"]
        assert not it["errors"] and len(it["runs"]) == 2
        assert all(r["accepted"] == 2 and len(r["iterations"]) == 2 for r in it["runs"])
        assert len({r["final_state_hash"] for r in it["runs"]}) == 1        # the same trajectory every repeat
        for r in it["runs"]:
            for row in r["iterations"]:
                assert row["host_other"] > -1e-6 and row["components"]["certify"] > 0
        assert set(it["summary"]) == {"first", "bb2"}
        assert it["host_peak_mb"] is None or it["host_peak_mb"] > 0
        assert it["device_peak_allocated_mib"] is None
    assert "cuda" not in profile["paths"] and profile["config"]["with_fallback_items"] is False
    gpu = profile["paths"]["torch-cpu"]["items"]["compare.gpu"]
    assert {c["path"] for c in gpu["per_call"]} <= {"gpu", "cpu"}
    assert profile["paths"]["torch-cpu"]["items"]["energy.evaluate"]["transfers_per_call"]["upload_calls"] >= 1
    model = profile["model"]
    assert model["weights"]["n_candidates"] == 5 and model["weights"]["n_cert_reject"] == 2
    assert set(model["paths"]) == {"torch-cpu", "numpy"} and model["observed_native_seconds"] == 12.
    assert profile["bottleneck"] and any("BB2" in line for line in profile["bottleneck"])
    md = (out/"profile.md").read_text()
    assert "## 主要瓶颈" in md and "compare.gpu" in md and "不是测量" in md and "GPU 计时" not in md


def test_refuses_nonempty_output_and_bare_worker(tmp_path, spd):
    out = tmp_path/"busy"
    out.mkdir()
    (out/"x").write_text("keep")
    with pytest.raises(SystemExit, match="not empty"):
        spd.main(["--out", str(out), "--run-dir", str(tmp_path), "--skip-gate", "--no-gpu-query", "--allow-dirty"])
    assert (out/"x").read_text() == "keep"
    with pytest.raises(SystemExit, match="driver.json"):
        spd.main(["--out", str(tmp_path), "--worker", "host"])
