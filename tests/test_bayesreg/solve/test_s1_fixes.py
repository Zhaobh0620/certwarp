"""Fixes from the S1 solver review (docs/bayesreg/reviews/S1_solve_复核_20261030.md), synthetic data only.

1. hard constraint 1: an output state whose fresh CPU certificate does not pass is never exported as disp.npz;
   final_native.npz and cert.json stay as failure evidence (status final_certificate_rejected, rejected = true) and
   run_pair exits non-zero;
2. hard constraint 4: an exception inside a round keeps the round's formal commits (option (a): the last accepted
   step) -- a FloatingPointError in the stable comparison (run_pg ends nonfinite_evaluation) and any other
   round exception (RuntimeError, the commit tracker); on the native level the last commit is exported;
3. C11-5 / D15: sigma^2 collapse is flagged (only flagged), with the reviewer's synthetic observation reproduced
   (noisy fixed image: sigma^2 orders of magnitude below the noise variance);
plus review items 5 (a failing diagnostic sigma_post M step keeps the ending) and 8 (disp.npz direction against
the repository evaluator).
"""
from dataclasses import replace
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from jd_curl.three_d.grid import make_grid_3d

from bayesreg.cli import run_pair
from bayesreg.solve import (Caps, Eta, FixedSigma, GaussianMStep, LevelInput, NumpyBackend, SigmaMonitor, SolverConfig,
                            displacement_to_voxels, export_native, solve_multilevel)
from bayesreg.solve.stopping import FINAL_CERTIFICATE_REJECTED, LEVEL_FAILED, STOPPED_BEFORE_NATIVE, state_hash

from . import _synthetic as syn
from ..io_seg._helpers import DEV
from .test_run_pair import read, setup, write_synthetic_pair, write_tissue  # noqa: F401  (fixture)

REPO = Path(__file__).resolve().parents[3]


class FailingCertificate(NumpyBackend):
    """The certificate of the next call(s) reports uncertain/subdivision_budget (as the reviewer's probe A)."""

    def __init__(self, fail=True):
        super().__init__()
        self.fail = fail

    def certify(self, u, grid):
        out = super().certify(u, grid)
        if not self.fail:
            return out
        return replace(out, passed=False, status="uncertain", cert_reason="subdivision_budget",
                       rejection="FEASIBILITY_UNCERTAIN")


# ---- 1. final certificate ---------------------------------------------------------------------------------

def test_export_refuses_a_failing_final_certificate(tmp_path):
    grid = syn.cube_grid(9)
    u = syn.solenoidal_displacement(grid, .3)
    good = NumpyBackend().certify(u, grid)
    out = export_native(tmp_path, u=u, grid=grid, status="cap_reached", backend=FailingCertificate(),
                        solver_cert=good)
    assert out["certificate_passed"] is False and out["status"] == FINAL_CERTIFICATE_REJECTED
    assert out["disp_meta"] is None and not (tmp_path/"disp.npz").exists()
    with np.load(tmp_path/"final_native.npz") as z:
        assert str(z["status"]) == FINAL_CERTIFICATE_REJECTED and z["u"].tobytes() == u.tobytes()
    cert = json.loads((tmp_path/"cert.json").read_text())
    assert cert["rejected"] is True and cert["disp_written"] is False and cert["status"] == "uncertain"
    assert cert["solver_status"] == "cap_reached" and cert["state_hash"] == state_hash(u)
    ok = tmp_path/"ok"
    ok.mkdir()
    passed = export_native(ok, u=u, grid=grid, status="cap_reached", backend=NumpyBackend(), solver_cert=good)
    assert passed["certificate_passed"] and (ok/"disp.npz").is_file()
    assert json.loads((ok/"cert.json").read_text())["rejected"] is False


def test_run_pair_final_certificate_rejected(setup, monkeypatch):
    """run_pair: the export's certificate fails -> status final_certificate_rejected, exit 2, no disp.npz,
    final_native.npz / cert.json kept; hyper.json and status.json record the solver status and the export."""
    root, args = setup
    grid, pair = write_synthetic_pair(root, DEV[0], DEV[1])
    write_tissue(root, grid, DEV[0])
    write_tissue(root, grid, DEV[1], shift=(.7, -.5, .4))
    real_export = run_pair.export_native

    def failing_export(out, *, backend, **kwargs):
        wrapped = FailingCertificate()
        wrapped.cert_config = backend.cert_config
        return real_export(out, backend=wrapped, **kwargs)
    monkeypatch.setattr(run_pair, "export_native", failing_export)
    out = root/"run_rejected_cert"
    assert run_pair.main(args(pair, out, "--rounds", "1,1,1,1", "--max-accepted", "1")) == 2
    status = read(out/"status.json")
    assert status["status"] == FINAL_CERTIFICATE_REJECTED and status["solver_status"] == "cap_reached"
    assert status["export"]["certificate_passed"] is False
    assert not (out/"disp.npz").exists()
    assert (out/"final_native.npz").is_file() and read(out/"cert.json")["rejected"] is True
    hyper = read(out/"hyper.json")
    assert hyper["status"] == FINAL_CERTIFICATE_REJECTED and hyper["solver_status"] == "cap_reached"
    assert hyper["gpu_comparison"]["path"].startswith("CPU reference")
    assert (out/"qc.json").is_file() and (out/"provenance.json").is_file()


# ---- 2. exceptions inside a round ---------------------------------------------------------------------------

class RaisingComparison(NumpyBackend):
    """Raises `error` at the (after+1)-th stable comparison on the grid of `shape` (None: any grid)."""

    def __init__(self, after, error, shape=None):
        super().__init__()
        self.after, self.error, self.shape, self.calls = after, error, shape, 0

    def stable_reduction(self, state, base, trial):
        if self.shape is None or state.grid.shape == self.shape:
            self.calls += 1
            if self.calls > self.after:
                raise self.error
        return super().stable_reduction(state, base, trial)


@pytest.mark.parametrize("error", [FloatingPointError("injected overflow"), RuntimeError("injected device failure")])
def test_exception_in_a_round_keeps_the_last_formal_commit(error):
    """Option (a), level 0: the 4th comparison raises; the 3 accepted steps are formal commits and the run's commit
    is the 3rd (its state_hash is the last accept row's), status level_failed with the exception in the reason."""
    native, levels, _ = syn.four_levels(26)
    backend = RaisingComparison(3, error)
    rows = []
    result = solve_multilevel(backend, levels, eta=Eta(1., 1., (1., 1., 1., 1.)), sigma=FixedSigma([1.]),
                              config=SolverConfig(rounds=(1, 1, 1, 1), max_levels=1), log=rows.append)
    accepts = [r for r in rows if r["kind"] == "accept"]
    assert len(accepts) == 3 and result.status == LEVEL_FAILED
    assert result.commit.cert.state_hash == accepts[-1]["state_hash"] == state_hash(result.commit.u)
    assert np.any(result.commit.u)
    row = result.rounds[-1]
    if isinstance(error, FloatingPointError):        # caught in run_pg: nonfinite_evaluation, C13-7 terminate
        assert row["status"] == "nonfinite_evaluation" and "injected overflow" in row["reason"]
        assert row["last_iteration"][-1][0] == "COMPARISON_EXCEPTION" and row["accepted"] == 3
    else:                                            # the commit tracker of the level loop
        assert row["status"] == "exception" and "RuntimeError: injected device failure" in row["reason"]
        assert row["commits"] == 3 and row["state_hash"] == accepts[-1]["state_hash"]
    assert "injected" in result.reason


class RaisingGradient(NumpyBackend):
    """Raises `error` at the (after+1)-th gradient on the grid of `shape`."""

    def __init__(self, after, error, shape):
        super().__init__()
        self.after, self.error, self.shape, self.calls = after, error, shape, 0

    def gradient(self, state, ev, **kwargs):
        if state.grid.shape == self.shape:
            self.calls += 1
            if self.calls > self.after:
                raise self.error
        return super().gradient(state, ev, **kwargs)


def test_native_exception_exports_the_last_commit(tmp_path):
    """A RuntimeError on the native level right after the first accepted step (the gradient there; G0 and the
    start gradient are the first two native gradients): level_failed on the native level with that step as the
    commit (not the transferred seed), exported as usual -- fresh CPU certificate passes, disp.npz written (C13-7)."""
    native, levels, _ = syn.four_levels(26)
    backend = RaisingGradient(2, RuntimeError("injected at native"), native.shape)
    rows = []
    result = solve_multilevel(backend, levels, eta=Eta(1., 1., (1., 1., 1., 1.)), sigma=FixedSigma([1.]),
                              config=SolverConfig(rounds=(1, 1, 1, 1), caps=Caps(max_accepted=3)), log=rows.append)
    accepted = [r for r in rows if r["kind"] == "candidate" and r["level"] == 3 and r["accepted"]]
    seed_hash = result.transfers[-1]["rows"][-1]["cert"]["state_hash"]
    assert result.status == LEVEL_FAILED and result.native and len(accepted) == 1
    assert result.rounds[-1]["status"] == "exception" and result.rounds[-1]["commits"] == 1
    assert "RuntimeError: injected at native" in result.reason
    assert result.commit.cert.state_hash == state_hash(result.commit.u) != seed_hash
    out = export_native(tmp_path, u=result.commit.u, grid=native, status=result.status, backend=backend,
                        solver_cert=result.commit.cert)
    assert out["certificate_passed"] and (tmp_path/"disp.npz").is_file()
    with np.load(tmp_path/"final_native.npz") as z:
        assert str(z["status"]) == LEVEL_FAILED


# ---- 3. sigma collapse ---------------------------------------------------------------------------------------

def test_sigma_monitor_flags_adjacent_round_drops_only():
    """C11-5: an M step used by the next round (or the native sigma_post) that lowers some sigma_c^2 by more than
    10x sets the flag; level starts and a coarse level's unused last M step are recorded, not flagged."""
    m = SigmaMonitor()
    m.update([1., 2.], level=0, round_=0, kind="level_start", used=True)
    m.update([.05, 1.], level=0, round_=0, kind="m_step", used=False)       # coarse, unused: recorded only
    m.update([.04, 1.], level=1, round_=0, kind="level_start", used=True)
    m.update([.01, .9], level=1, round_=0, kind="m_step", used=True)        # 4x: no flag
    m.update([1e-3, .9], level=1, round_=1, kind="m_step", used=True)       # 10x exactly: no flag (more than 10x)
    m.update([1e-3, .05], level=1, round_=2, kind="m_step", used=True)      # channel 1 down 18x: flag
    record = m.record()
    assert record["sigma_collapse_suspected"] is True
    assert [(e["level"], e["round"], e["channels"]) for e in record["events"]] == [(1, 2, [1])]
    assert [(e["level"], e["channels"]) for e in record["unused_m_step_drops"]] == [(0, [0])]
    assert record["trace"][3]["ratio_to_intensity"] == [90.]
    assert record["trace"][0]["ratio_to_previous"] is None
    clean = SigmaMonitor()
    clean.update([1.], level=0, round_=0, kind="level_start", used=True)
    clean.update([.5], level=0, round_=0, kind="m_step", used=True)
    assert clean.record()["sigma_collapse_suspected"] is False


def test_sigma_collapse_on_the_noisy_synthetic_pair_is_flagged(record_property):
    """The reviewer's observation (probe C), at level 0 for runtime: fixed image + Gaussian noise (std .02,
    variance 4e-4), Gaussian M step with a = b = 0, three rounds of 40 accepted steps: sigma^2 falls more than
    1000x below the noise variance and every adjacent-round drop > 10x is flagged.  (The native-level version of
    the probe -- 26^3, rounds 1,1,1,2 -- was run once by hand: 8.2e-4 -> 2.7e-6 -> 2.7e-7, both drops flagged.)"""
    native, levels, _ = syn.four_levels(26)
    rng = np.random.default_rng(0)
    noise = .02
    levels = tuple(LevelInput(lv.level, lv.grid, lv.fixed+noise*rng.normal(size=lv.fixed.shape), lv.moving, lv.mask,
                              lv.tissue) for lv in levels)
    result = solve_multilevel(NumpyBackend(), levels, eta=Eta(.01, 1., (1., .25, .25, .25)), sigma=GaussianMStep(),
                              config=SolverConfig(rounds=(3, 1, 1, 1), caps=Caps(max_accepted=40), max_levels=1))
    used = [r["sigma2_used"][0] for r in result.rounds]
    record_property("sigma2_used", used)
    assert result.status == STOPPED_BEFORE_NATIVE
    assert used[-1] < noise**2/1000
    sigma = result.sigma
    assert sigma["sigma_collapse_suspected"] is True
    assert [(e["level"], e["round"]) for e in sigma["events"]] == [(0, 0), (0, 1)]
    assert all(e["ratios"][0] < .1 for e in sigma["events"])


def test_failing_sigma_post_keeps_the_ending():
    """Review item 5 (C11-6): the M step after the last round is a diagnostic; when it fails, sigma2_post is null
    with the error and the run's ending is unchanged."""
    native, levels, _ = syn.four_levels(26)

    class FailingPost(FixedSigma):
        def m_step(self, state, point):
            raise FloatingPointError("M step: injected")
    result = solve_multilevel(NumpyBackend(), levels, eta=Eta(1., 1., (1., 1., 1., 1.)), sigma=FailingPost([1.]),
                              config=SolverConfig(rounds=(1, 1, 1, 1), caps=Caps(max_accepted=2), max_levels=1))
    assert result.status == STOPPED_BEFORE_NATIVE and result.sigma_post is None
    assert result.rounds[-1]["sigma2_post"] is None and "injected" in result.rounds[-1]["sigma2_post_error"]


# ---- 8. disp direction against the repository evaluator --------------------------------------------------------

def test_disp_direction_matches_the_evaluator():
    """C16 / review probe D: on an anisotropic grid with a nonzero origin, d equals scripts/evaluate_l2r_style
    disp_voxels bitwise, and the Learn2Reg pull-back map_coordinates(moving, identity + d) equals sampling the
    moving voxel array at the physical x + u (to round-off); the opposite sign does not."""
    sys.path.insert(0, str(REPO/"scripts"))
    try:
        import evaluate_l2r_style as evaluator
    finally:
        sys.path.remove(str(REPO/"scripts"))
    grid = make_grid_3d(nx=21, ny=18, nz=15, hx=.9, hy=1.2, hz=1.4, x0=-7., y0=3., z0=.5, unit="mm")
    u = syn.true_displacement(grid, (1.1, -.9, .8))
    d = displacement_to_voxels(u, grid)
    assert np.array_equal(d, evaluator.disp_voxels(u, grid.spacing))
    moving = syn.moving_image(grid.identity())
    ident = np.meshgrid(*[np.arange(n) for n in grid.shape], indexing="ij")
    l2r = map_coordinates(moving, [ident[k]+d[k] for k in range(3)], order=1, mode="nearest")
    x = grid.identity()+u
    index = [(x[..., 2-k]-grid.axes[2-k][0])/grid.spacing[2-k] for k in range(3)]
    physical = map_coordinates(moving, index, order=1, mode="nearest")
    wrong = map_coordinates(moving, [ident[k]-d[k] for k in range(3)], order=1, mode="nearest")
    assert np.max(np.abs(l2r-physical)) < 1e-12 and np.max(np.abs(wrong-physical)) > .1
