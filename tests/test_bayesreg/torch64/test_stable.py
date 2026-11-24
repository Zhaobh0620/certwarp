"""Layer-0 tests of the GPU-path stable comparison (contract C08-8, C08-9 "层 0 测试"; work package SG).

- the unified fallback rule on synthetic (A_hat, s_hat, r): PG, LM and outer, inside the band, on its edge, outside;
- the GPU path against the CPU reference on the torch64 L0 cases: |A_hat - A| <= s_hat, blocks close;
- device anomalies (J_q <= 0, nonfinite, OOM) in the comparison and in the evaluation: the CPU path decides and
  the reason is the CPU's;
- the CPU cache: every key component (state hash, ModelState identity: level, sigma, w, eta, tile; energy type;
  reference v) forces a recomputation;
- the GF_B double-sided 1.01 recheck;
- L1 along a numpy trajectory (the same base / trial states on both paths: identical decisions, |A_hat - A| <= s_hat,
  |F_B(device) - F_B(CPU)| <= s_hat) and the torch64-driven solver (options (a) and (b)) on small synthetic problems;
- cuda: one small GPU-path comparison (skipped without cuda; the D0 timing queue shares the GPU).
Everything runs on the CPU torch device unless marked cuda.
"""
import numpy as np
import pytest
import torch

from jd_curl.types import InvalidTrial
from jd_curl.validation import InputValidationError
from jd_curl.three_d.boundary import restrict_free

from bayesreg import model as M
from bayesreg.backends.torch64 import stable as S
from bayesreg.model import ModelState, build_samplers
from bayesreg.solve import (Caps, Commit, DirectObjective, NumpyBackend, ProximalParams, Torch64Backend, default_lambda,
                            g0_of, make_backend, solve_direct, solve_proximal, threshold)

from ._cases import case, smooth_field
from ..solve import _synthetic as syn

EPS = np.finfo(float).eps


# ----------------------------------------------------------------------------------------------
# Guard, depth, fallback rule
# ----------------------------------------------------------------------------------------------

def test_guard_depth_and_required_decrease():
    """C08-8(c): s_hat = 4096 eps max(1, E_b, E_t); depth m = 34 + tile - 1 (8225 for 8192), m <= 1.4e4;
    r = c1 tau |g.p| (PG), 1e-4 pred (LM), 0 (outer)."""
    assert S.gpu_guard(.5, .25) == 4096*EPS
    assert S.gpu_guard(3., 7.) == 4096*EPS*7.
    assert S.reduction_depth(8192) == 8225
    assert S.reduction_depth(1) == 34
    assert S.reduction_depth(13967) == 14000
    with pytest.raises(InputValidationError):
        S.reduction_depth(13968)
    assert S.required_decrease("pg", c1=1e-4, tau=.5, slope=-3.) == 1e-4*.5*3.
    assert S.required_decrease("lm", pred=2.) == 2e-4
    assert S.required_decrease("outer_E") == S.required_decrease("outer_F") == 0.
    with pytest.raises(InputValidationError):
        S.required_decrease("barrier")
    record = S.reduction_provenance(8192)
    assert record["depth_bound"] == 8225 and record["depth_limit"] == 14000


S_HAT = 2.**-33                     # powers of two: the band edges are exact
BAND = 8*S_HAT                       # 2^-30
UP = np.nextafter                    # one ulp further out
R_PG = S.required_decrease("pg", c1=2.**-13, tau=1., slope=-2.**-13)    # c1 tau |g.p| = 2^-26 = 16 bands
R_LM = S.required_decrease("lm", pred=2.**-28/1e-4)                     # 1e-4 pred ~ 2^-28 = 4 bands
FALLBACK_TABLE = [
    # context, A_hat, r, fallback, reason
    ("pg", 1e-6, R_PG, False, None),                              # far out of band, accepted side
    ("pg", -1e-6, R_PG, False, None),                             # far out of band, rejected side
    ("pg", R_PG+BAND, R_PG, True, "near_required"),               # |A - r| = 8 s_hat: edge, in
    ("pg", UP(R_PG+BAND, 1.), R_PG, False, None),                 # one ulp beyond the edge
    ("pg", R_PG-BAND, R_PG, True, "near_required"),
    ("pg", UP(R_PG-BAND, 0.), R_PG, False, None),
    ("pg", R_PG+32*S_HAT, R_PG, False, None),                     # near out of band (C08-9: 8 s < |A - r| <= 32 s)
    ("pg", BAND, R_PG, True, "near_zero"),                        # |A| = 8 s_hat: edge, in
    ("pg", UP(BAND, 1.), R_PG, False, None),
    ("pg", -BAND, R_PG, True, "near_zero"),
    ("pg", 0., R_PG, True, "near_zero"),
    ("lm", R_LM+BAND, R_LM, True, "near_required"),
    ("lm", UP(R_LM+BAND, 1.), R_LM, False, None),
    ("lm", R_LM/2, R_LM, False, None),                            # 2 bands from both 0 and r
    ("lm", BAND, R_LM, True, "near_zero"),
    ("lm", 1.5*BAND, R_LM, False, None),
    ("lm", 3.5*BAND, R_LM, True, "near_required"),
    ("outer_E", BAND, 0., True, "near_zero+near_required"),       # r = 0: both conditions coincide
    ("outer_E", UP(BAND, 1.), 0., False, None),
    ("outer_E", -BAND, 0., True, "near_zero+near_required"),
    ("outer_E", UP(-BAND, -1.), 0., False, None),
    ("outer_F", 32*S_HAT, 0., False, None),
    ("outer_F", 0., 0., True, "near_zero+near_required"),
    ("outer_F", 1e-3, 0., False, None),
    ("pg", float("nan"), R_PG, True, "nonfinite"),
]


@pytest.mark.parametrize("row", FALLBACK_TABLE, ids=lambda r: f"{r[0]}:{r[1]:.4g}")
def test_fallback_rule_table(row):
    """C08-8(d): fall back iff |A_hat| <= 8 s_hat or |A_hat - r| <= 8 s_hat (edges included); r from the
    context (PG c1 tau |g.p|, LM 1e-4 pred, outer 0)."""
    context, A, r, fallback, reason = row
    decision = S.fallback_rule(A, S_HAT, r)
    assert (decision.fallback, decision.reason) == (fallback, reason), (row, decision)
    assert S.fallback_rule(1., float("inf"), 0.).reason == "nonfinite"


def test_out_of_band_gpu_decision_equals_the_cpu_decision_under_the_premise():
    """C08-8(c)(d) arithmetic: out of the band, any CPU value A with |A - A_hat| < 8 s_hat - s gives the same
    accept / reject decision (sign of certified - r) as the GPU value -- checked on a dense synthetic grid."""
    rng = np.random.default_rng(0)
    E = 3.7
    s_hat, s = S.gpu_guard(E, E), 16*EPS*E
    for r in (0., 1e-4*3e-10, 5e-11):
        for A_hat in r+s_hat*np.concatenate((np.linspace(-40, 40, 801), rng.uniform(-40, 40, 400))):
            if S.fallback_rule(A_hat, s_hat, r).fallback:
                continue
            gpu = abs(A_hat) > s_hat and A_hat-s_hat > 0 and A_hat-s_hat >= r
            for A in A_hat+(8*s_hat-s)*np.array([-.999, -.5, 0., .5, .999]):
                cpu = abs(A) > s and A-s > 0 and A-s >= r
                assert gpu == cpu, (r, A_hat, A)


# ----------------------------------------------------------------------------------------------
# GPU path against the CPU reference
# ----------------------------------------------------------------------------------------------

def _backend(name, **options):
    state, u, state_options = case(name)
    backend = Torch64Backend(device="cpu", state_options=dict(state_options, **options))
    backend.prepare(state)
    return backend, state, u


@pytest.mark.parametrize("name", ["small", "box12", "medium"])
def test_gpu_path_matches_cpu_reference(name, record_property):
    """C08-8(a)(b): A_hat (device elementwise + tile sums + host longdouble) against the CPU reference A:
    |A_hat - A| <= s_hat (the C08-9 gate) and every block within 1e-12 max(1, E); the channel blocks bitwise
    those of the CPU formula on the downloaded residuals."""
    backend, state, u = _backend(name)
    u_t = u+smooth_field(state.grid, .02, 3)
    base, trial = backend.evaluate(state, u), backend.evaluate(state, u_t)
    assert base.device is not None and trial.device is not None
    model_id = backend.model_identity(state)
    dstate = base.device.state
    delta, blocks = S.gpu_fb_change(dstate, base.device, trial.device)
    cpu = M.stable_reduction(state, M.evaluate(state, u), M.evaluate(state, u_t))
    A_hat, s_hat = float(-delta), S.gpu_guard(base.energy, trial.energy)
    E = max(1., base.energy, trial.energy)
    record_property("gpu_path", dict(A_hat=A_hat, A=cpu.ared, err_over_s_hat=abs(A_hat-cpu.ared)/s_hat))
    assert abs(A_hat-cpu.ared) <= s_hat
    for key in ("diffusion", "logvol", "curl"):
        assert abs(float(blocks[key]-cpu.blocks[key])) <= 1e-12*E, key
    for c, value in enumerate(blocks["channel"]):
        expected = M.energy._node_change_sum(state, base.device.r_host[c], trial.device.r_host[c])/2
        assert value == expected
    # the comparator: far from the band the GPU path decides, with the GPU value and guard
    red = backend.stable_compare(state, base, trial, required=0., context="outer_F")
    assert red.path == ("gpu" if abs(A_hat) > 8*s_hat else "cpu")
    if red.path == "gpu":
        assert (red.ared, red.guard) == (A_hat, s_hat) and red.reason == cpu.reason
    assert backend.comparison_summary()["comparisons"] == 1
    assert model_id == S.model_identity(state)


def test_band_sends_the_comparison_to_the_cpu_reference():
    """C08-8(d): a required decrease within 8 s_hat of A_hat (or a tiny change) falls back; the result is the CPU
    reference (ared, guard = 16 eps max E) and the record keeps A_hat, s_hat, r and the reason."""
    backend, state, u = _backend("box12")
    base = backend.evaluate(state, u)
    trial = backend.evaluate(state, u+smooth_field(state.grid, .02, 3))
    A_hat = backend.stable_compare(state, base, trial, context="outer_F").info["A_hat"]
    near = backend.stable_compare(state, base, trial, required=A_hat, context="pg")
    assert near.path == "cpu" and near.info["fallback_reason"] == "near_required"
    cpu = M.stable_reduction(state, M.evaluate(state, base.u), M.evaluate(state, trial.u))
    assert (near.ared, near.guard) == (cpu.ared, cpu.guard)
    record = near.record()
    assert record["A_hat"] == A_hat and record["required"] == A_hat and record["fallback"]
    same = backend.evaluate(state, u)                              # Delta = 0 exactly: |A_hat| <= 8 s_hat
    red = backend.stable_compare(state, base, same, context="outer_F")
    assert red.info["A_hat"] == 0. and red.path == "cpu"
    assert red.info["fallback_reason"] == "near_zero+near_required" and red.reason == "ENERGY_RESOLUTION"
    summary = backend.comparison_summary()
    assert summary["comparisons"] == 3 and summary["cpu_fallback"] == 2
    assert summary["by_context"]["pg"] == dict(comparisons=1, cpu_fallback=1, fallback_ratio=1.)


# ----------------------------------------------------------------------------------------------
# Device anomalies (C08-8(e))
# ----------------------------------------------------------------------------------------------

def _patched_geometry(kind, calls):
    """OPT1: the comparison takes each state's geometry from `S._compared_geometry` (G recomputed; J, log J from the
    evaluation's Jacobian cache, or the full `_geometry` without one); the injection hits the trial's first chunk (the
    second call), as it hit `S._geometry` before OPT1."""
    original = S._compared_geometry

    def geometry(dstate, ev, chunk, number, corners=False):
        G, A, J, logJ = original(dstate, ev, chunk, number, corners)
        calls.append(1)
        if len(calls) == 2:                                # the trial's geometry of the first chunk
            if kind == "nonpositive":
                J = J.clone()
                J[0, 0] = -1e-3
                logJ = torch.log(J)
            elif kind == "nonfinite":
                G = G.clone()
                G[0, 0, 0, 0] = float("nan")
            elif kind == "oom":
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (injected)")
        return G, A, J, logJ
    return geometry


@pytest.mark.parametrize("slots", [2, 0], ids=["jacobian_cache", "no_cache"])
@pytest.mark.parametrize("kind", ["nonpositive", "nonfinite", "oom"])
def test_device_anomaly_in_the_comparison_falls_back(kind, slots, monkeypatch):
    """C08-8(e): a device J_q <= 0, a nonfinite device value or an OOM in the comparison sends it to the CPU
    reference; the decision and its reason are the CPU's (with and without the OPT1 Jacobian cache)."""
    backend, state, u = _backend("small", jacobian_cache_slots=slots)
    base = backend.evaluate(state, u)
    trial = backend.evaluate(state, u+smooth_field(state.grid, .02, 3))
    assert bool(trial.device.jacobian_entry(0)) == bool(slots)
    calls = []
    monkeypatch.setattr(S, "_compared_geometry", _patched_geometry(kind, calls))
    red = backend.stable_compare(state, base, trial, required=0., context="outer_F")
    cpu = M.stable_reduction(state, M.evaluate(state, base.u), M.evaluate(state, trial.u))
    assert red.path == "cpu" and red.info["fallback_reason"].startswith("device anomaly")
    assert (red.ared, red.guard, red.reason) == (cpu.ared, cpu.guard, cpu.reason)
    assert red.info["A_hat"] is None
    assert backend.comparison_summary()["device_anomalies"]["compare"] == 1


def test_device_anomaly_in_the_evaluation_is_decided_on_the_cpu(monkeypatch):
    """C08-8(e) at the evaluation: a device InvalidTrial (NONPOSITIVE_GAUSS_JD), FloatingPointError or OOM is
    recomputed on the CPU; a CPU-evaluable state comes back CPU-backed (and its comparisons take the CPU path),
    a really folded state comes back with the CPU's InvalidTrial (reason and count)."""
    backend, state, u = _backend("small")
    T = backend._T
    original = T.evaluate
    injected = iter([InvalidTrial("NONPOSITIVE_GAUSS_JD", "injected", 3), FloatingPointError("injected"),
                     torch.cuda.OutOfMemoryError("injected")])

    def failing(dstate, value):
        failure = next(injected, None)
        if isinstance(failure, BaseException):
            raise failure
        return failure if failure is not None else original(dstate, value)
    base = backend.evaluate(state, u)
    monkeypatch.setattr(T, "evaluate", failing)
    for _ in range(3):
        point = backend.evaluate(state, u)
        assert point.device is None and point.anomaly and point.cpu is not None
        assert point.energy == M.evaluate(state, u).energy
    red = backend.stable_compare(state, base, point, context="outer_F")
    assert red.path == "cpu" and "device evaluation unavailable" in red.info["fallback_reason"]
    assert red.reason == M.stable_reduction(state, M.evaluate(state, u), M.evaluate(state, u)).reason
    g = backend.gradient(state, point)
    assert np.array_equal(g, M.gradient(state, point.cpu))
    folded = np.zeros_like(u)
    h = state.grid.spacing
    folded[2, 2, 2, 0] = 1.6*h[0]
    folded[2, 3, 4, 0] = -1.6*h[0]
    monkeypatch.setattr(T, "evaluate", lambda dstate, value: InvalidTrial("NONPOSITIVE_GAUSS_JD", "device", 999))
    got, expected = backend.evaluate(state, folded), M.evaluate(state, folded)
    assert isinstance(got, InvalidTrial) and (got.reason, got.invalid_count) == (expected.reason,
                                                                                expected.invalid_count)
    anomalies = backend.comparison_summary()["evaluate_anomalies"]
    assert len(anomalies) == 4 and anomalies[-1]["cpu"] == "NONPOSITIVE_GAUSS_JD"


def test_trial_rejected_by_the_cpu_keeps_the_cpu_reason(monkeypatch):
    """C08-8(e): when the CPU reference cannot evaluate the trial, the comparison rejects it with the CPU's
    InvalidTrial reason (no energy values); a base the CPU cannot evaluate raises FloatingPointError."""
    backend, state, u = _backend("small")
    base = backend.evaluate(state, u)
    trial = backend.evaluate(state, u+smooth_field(state.grid, .02, 3))
    real = M.evaluate

    class CpuModel:
        stable_reduction = staticmethod(M.stable_reduction)
        gradient = staticmethod(M.gradient)

        @staticmethod
        def evaluate(st, value):
            if np.array_equal(value, trial.u):
                return InvalidTrial("NONPOSITIVE_GAUSS_JD", "cpu says no", 2)
            return real(st, value)
    backend.comparator.cpu_model = CpuModel
    calls = []
    monkeypatch.setattr(S, "_compared_geometry", _patched_geometry("nonpositive", calls))
    red = backend.stable_compare(state, base, trial, context="pg", required=1e-12)
    assert red.reason == "NONPOSITIVE_GAUSS_JD" and red.ared is None and red.certified is None
    with pytest.raises(FloatingPointError):
        calls.clear()
        backend.stable_compare(state, trial, base, context="pg")


# ----------------------------------------------------------------------------------------------
# Cache key (C08-8(d))
# ----------------------------------------------------------------------------------------------

def test_cache_key_every_component_forces_a_recomputation():
    """Key = state_hash + ModelState identity (level, sigma, w, eta, tile) + energy type + reference v hash: a hit
    for the same key and ModelState; a recomputation when any component changes."""
    backend, state, u = _backend("small")
    comparator = S.StableComparator(cache_size=16)
    point = backend.evaluate(state, u)
    other = backend.evaluate(state, u+smooth_field(state.grid, .02, 3))
    mid = S.model_identity(state)

    def evaluations(st, p, **kw):
        before = comparator.stats.cpu_evaluations
        comparator.cpu_point(st, p, model_id=S.model_identity(st), **kw)
        return comparator.stats.cpu_evaluations-before

    assert evaluations(state, point) == 1
    assert evaluations(state, point) == 0 and comparator.stats.cache_hits == 1        # same key: cached
    assert evaluations(state, other) == 1                                             # state_hash
    assert evaluations(state, point, energy="E_v", reference="v1") == 1               # energy type
    assert evaluations(state, point, energy="E_v", reference="v1") == 0
    assert evaluations(state, point, energy="E_v", reference="v2") == 1               # reference v
    changed = dict(sigma=state.replace(sigma2=state.sigma2*1.5),
                   w=state.replace(weights=state.weights*.9),
                   alpha=state.replace(alpha=state.alpha*2), nu=state.replace(nu=state.nu*2),
                   tau=state.replace(tau2=state.tau2*1.1), tile=state.replace(tile_size=5))
    identities = {mid}
    for name, st in changed.items():
        identities.add(S.model_identity(st))
        assert evaluations(st, point) == 1, name
    level, _, _ = case("box12")
    identities.add(S.model_identity(level))                                             # level (grid)
    assert len(identities) == len(changed)+2
    twin = state.replace()                                         # same content, another ModelState object
    assert S.model_identity(twin) == mid
    assert evaluations(twin, point) == 1                           # the cached Evaluation belongs to `state`
    cached = list(comparator.cache.values())
    assert all(isinstance(v, M.Evaluation) for v in cached)


def test_cache_is_lru_bounded():
    backend, state, u = _backend("small")
    comparator = S.StableComparator(cache_size=2)
    points = [backend.evaluate(state, u*s) for s in (1., .9, .8)]
    for p in points:
        comparator.cpu_point(state, p, model_id="m")
    assert len(comparator.cache) == 2
    comparator.cpu_point(state, points[0], model_id="m")
    assert comparator.stats.cpu_evaluations == 4


# ----------------------------------------------------------------------------------------------
# GF_B recheck (C08-8(f))
# ----------------------------------------------------------------------------------------------

def test_gf_recheck_double_sided_band():
    """T_B/1.01 <= G <= 1.01 T_B -> the CPU value decides (and is recorded, with whether it changed the
    decision); outside the band G is returned unchanged without CPU work."""
    from bayesreg.solve.stopping import gf_b
    backend, state, u = _backend("box12")
    point = backend.evaluate(state, u)
    G = gf_b(backend.gradient(state, point), state.grid.omega, state.omega0)
    G_cpu = gf_b(M.gradient(state, M.evaluate(state, u)), state.grid.omega, state.omega0)
    assert abs(G-G_cpu) <= 1e-12*G_cpu
    for target, inside in ((G*1.0101, False), (G/1.0101, False), (G*1.01, True), (G/1.01, True), (G, True)):
        value, row = backend.confirm_gradient_norm(state, point, G, target)
        if inside:
            assert value == G_cpu and row["G_gpu"] == G and row["changed_decision"] == ((G <= target)
                                                                                       != (G_cpu <= target))
        else:
            assert value == G and row is None
    checks = backend.comparison_summary()["gf_rechecks"]
    assert checks["count"] == 3
    value, row = backend.confirm_gradient_norm(state, point, G, G, extra_gradient=lambda: np.zeros(
        3*state.grid.n_free_scalar), energy="E_v", reference="v")
    assert value == G_cpu


# ----------------------------------------------------------------------------------------------
# L1 along a trajectory, and the torch64-driven solver
# ----------------------------------------------------------------------------------------------

def _problem(n=9, sigma2=1., alpha=1.):
    grid = syn.cube_grid(n)
    u_true = syn.solenoidal_displacement(grid, .4)
    moving = syn.moving_image(grid.identity())
    fixed = syn.warp_with_spline(moving, grid, u_true)
    state = ModelState(grid=grid, omega0=grid.omega, fixed=fixed[None], samplers=build_samplers(moving[None], grid),
                       mask=syn.brain_box(grid, 1), sigma2=np.array([sigma2]), alpha=alpha, nu=alpha,
                       tau2=np.full(4, 1.), tissue=syn.uniform_tissue(grid))
    return grid, state


def _start(backend, grid):
    zero = np.zeros(grid.shape+(3,))
    return Commit(zero, restrict_free(zero, grid), backend.certify(zero, grid))


class ShadowBackend(NumpyBackend):
    """The numpy trajectory; every comparison is also run on the torch64 GPU path with the same two states
    (the L1 snapshot: decisions and values on identical inputs)."""

    def __init__(self):
        super().__init__()
        self.torch = Torch64Backend(device="cpu")
        self.rows = []

    def prepare(self, state):
        self.torch.prepare(state)
        return state

    def stable_compare(self, state, base, trial, *, required=0., energy="F_B", reference=None, extra=None,
                       context="unspecified", gf_ratio=None):
        reference_red = M.stable_reduction(state, base, trial)
        tb, tt = self.torch.evaluate(state, base.u), self.torch.evaluate(state, trial.u)
        red = self.torch.stable_compare(state, tb, tt, required=required, context=context, gf_ratio=gf_ratio)

        def decision(r):
            return r.reason is None and r.certified >= required
        self.rows.append(dict(cpu=decision(reference_red), gpu=decision(red), path=red.path, A=reference_red.ared,
                              A_hat=red.info["A_hat"], s_hat=red.info["s_hat"],
                              dE=max(abs(tb.energy-base.energy), abs(tt.energy-trial.energy)),
                              s_E=S.gpu_guard(base.energy, trial.energy)))
        return reference_red


def test_l1_along_the_numpy_trajectory(record_property):
    """C08-9 (layer 0, CPU torch): along a numpy PG trajectory to convergence, every comparison is repeated on
    the GPU path with the same states: identical accept decisions, |A_hat - A| <= s_hat, F_B of the device
    and of the CPU within s_hat; both the GPU decision and the CPU fallback occur."""
    grid, state = _problem()
    backend = ShadowBackend()
    backend.prepare(state)
    target = threshold(g0_of(backend, state))
    result = solve_direct(backend, state, _start(backend, grid), target=target, caps=Caps())
    rows = backend.rows
    assert result.status == "converged_gradient" and len(rows) >= result.accepted
    assert all(r["cpu"] == r["gpu"] for r in rows)
    assert all(abs(r["A_hat"]-r["A"]) <= r["s_hat"] for r in rows if r["A_hat"] is not None)
    assert all(r["dE"] <= r["s_E"] for r in rows)
    paths = {r["path"] for r in rows}
    record_property("l1", dict(comparisons=len(rows), paths=sorted(paths),
                               max_err=max(abs(r["A_hat"]-r["A"])/r["s_hat"] for r in rows)))
    assert paths == {"gpu", "cpu"}


def test_torch64_solver_option_a_matches_numpy(record_property):
    """The torch64 backend drives solve_direct on the 9^3 problem to converged_gradient with every candidate
    decision identical to the numpy run and the final u within 1e-6 mm (free-running trajectories differ by
    rounding; the decisions do not here); records: comparison per candidate, summary bins, CPU G0."""
    grid, state = _problem()
    results = {}
    for name in ("numpy", "torch64"):
        backend = make_backend(name, device="cpu")
        st = backend.prepare(state)
        rows = []
        target = threshold(g0_of(backend, st))
        results[name] = (solve_direct(backend, st, _start(backend, grid), target=target, caps=Caps(),
                                      log=rows.append), rows, backend, target)
    (a, rows_a, _, t_a), (b, rows_b, tb, t_b) = results["numpy"], results["torch64"]
    assert t_a == t_b                                                       # G0 on the CPU reference
    assert a.status == b.status == "converged_gradient" and a.accepted == b.accepted
    ca = [(r["iteration"], r["attempt"], r["accepted"], r["reason"]) for r in rows_a if r["kind"] == "candidate"]
    cb = [(r["iteration"], r["attempt"], r["accepted"], r["reason"]) for r in rows_b if r["kind"] == "candidate"]
    assert ca == cb
    assert np.max(np.abs(a.commit.u-b.commit.u)) < 1e-6
    assert all("comparison" in r for r in rows_b if r["kind"] == "candidate" and r["energy"] is not None)
    summary = tb.comparison_summary()
    record_property("summary", {k: summary[k] for k in ("comparisons", "cpu_fallback", "fallback_reasons")})
    assert summary["comparisons"] == sum(1 for r in rows_b if r["kind"] == "candidate" and r["energy"] is not None)
    assert sum(v["comparisons"] for v in summary["by_gf_over_tb"].values()) == summary["comparisons"]
    assert isinstance(b.commit.u, np.ndarray) and b.commit.cert.passed
    assert tb.record()["comparison"]["depth_bound"] == 8225


def test_torch64_solver_option_b_commits(record_property):
    """Option (b) on the torch64 backend: E_v comparisons (GPU-path F_B plus the exact host proximal blocks)
    and the independent F_B comparison of every commit are logged with their records; the run reaches
    converged_gradient like the numpy run and agrees with it to 1e-5 mm."""
    grid, state = _problem()
    out = {}
    for name in ("numpy", "torch64"):
        backend = make_backend(name, device="cpu")
        st = backend.prepare(state)
        rows = []
        target = threshold(g0_of(backend, st))
        out[name] = (solve_proximal(backend, st, _start(backend, grid), target=target, caps=Caps(),
                                    params=ProximalParams(lam=default_lambda(grid)), log=rows.append), rows, backend)
    (a, _, _), (b, rows, backend) = out["numpy"], out["torch64"]
    assert a.status == b.status == "converged_gradient"
    checks = [r for r in rows if r["kind"] == "commit_check"]
    assert checks and all(r["committed"] for r in checks)
    for r in checks:
        assert r["E_v"]["comparison"]["energy"] == "E_v" and r["E_v"]["comparison"]["context"] == "outer_E"
        assert r["F_B"]["comparison"]["energy"] == "F_B" and r["F_B"]["comparison"]["context"] == "outer_F"
    inner = [r for r in rows if r["kind"] == "candidate" and r["energy"] is not None]
    assert all(r["comparison"]["energy"] == "E_v" for r in inner)
    assert np.max(np.abs(a.commit.u-b.commit.u)) < 1e-5
    contexts = backend.comparison_summary()["by_context"]
    record_property("contexts", contexts)
    assert set(contexts) == {"pg", "outer_E", "outer_F"}


def test_make_backend_and_state_reuse():
    """--backend numpy | torch64; prepare reuses the uploaded splines for a ModelState with the same samplers and
    keeps at most max_device_states device states."""
    assert make_backend("numpy").name == "numpy"
    with pytest.raises(InputValidationError):
        make_backend("numpy", device="cuda")
    with pytest.raises(InputValidationError):
        make_backend("cupy")
    backend = Torch64Backend(device="cpu", max_device_states=2)
    grid, state = _problem(7)
    backend.prepare(state)
    first = backend._lookup(state)[0]
    second_state = state.replace(sigma2=np.array([.5]))
    backend.prepare(second_state)
    assert backend._lookup(second_state)[0].splines is first.splines
    backend.prepare(state.replace(alpha=.5))
    assert len(backend._states) == 2
    assert backend.prepare(state) is state
    record = backend.record()
    assert record["backend"] == "torch64" and record["device"] == "cpu"
    assert record["certificate"]["delta_hex"] == "0x0.0p+0"


# ----------------------------------------------------------------------------------------------
# cuda (minimal)
# ----------------------------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")
def test_cuda_gpu_path_minimal(record_property):
    """C08-8 on cuda, one small case (box12): |A_hat - A| <= s_hat against the CPU reference, the comparison
    decided on the GPU path out of the band and on the CPU inside it."""
    state, u, options = case("box12")
    backend = Torch64Backend(device="cuda", state_options=options)
    backend.prepare(state)
    base = backend.evaluate(state, u)
    trial = backend.evaluate(state, u+smooth_field(state.grid, .02, 3))
    assert base.device.u_cm.device.type == "cuda"
    red = backend.stable_compare(state, base, trial, context="outer_F")
    cpu = M.stable_reduction(state, M.evaluate(state, base.u), M.evaluate(state, trial.u))
    record_property("cuda", dict(A_hat=red.info["A_hat"], A=cpu.ared, s_hat=red.info["s_hat"]))
    assert abs(red.info["A_hat"]-cpu.ared) <= red.info["s_hat"] and red.path == "gpu"
    near = backend.stable_compare(state, base, trial, required=red.info["A_hat"], context="pg")
    assert near.path == "cpu" and near.ared == cpu.ared
