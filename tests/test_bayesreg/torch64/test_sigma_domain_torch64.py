"""sigma_domain on the torch64 backend (work package SWLI; contract C04-6 gates 2 (torch64) and 9; draft SWL 2026-10-31).

(2)  torch64 (cpu device) on a support / support_shared state equals torch64 on the "v as a float64 weight" reference
     bit for bit (evaluate, gradient, HVP, GN diagonal, c_dat).
(9)  (i) torch64 (cpu; the cuda case runs only with SWLI_TEST_CUDA=1, GPU idle -- the lead runs it) and numpy give the
     same v at the same seed bytes and level arrays, with nodes planted at |Ghat - t_theta| <= 1e-14;
     (ii) a 1e-13 perturbation of the device rho does not change v (v never reads the device rho);
     (iv) the step sequence of numpy and torch64 (cpu) is the same (C09.1-3a gate's measure: every candidate's level,
     round, iteration, attempt, decision and reason; F_B to rounding);
     (v) two ModelStates with the same sigma, w, eta and different v have different C08-8(d) identities; the brain
     identity is the formula before SWLI.
Synthetic data only.
"""
import hashlib
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bayesreg.backends import torch64 as T                          # noqa: E402
from bayesreg.backends.torch64 import stable as ST                   # noqa: E402
from bayesreg.backends.torch64.stable import array_sha256            # noqa: E402
from bayesreg.model import support_level                             # noqa: E402
from bayesreg.solve import (Caps, GaussianMStep, NumpyBackend, SolverConfig, Torch64Backend,  # noqa: E402
                            eta_temporary, solve_multilevel)

from ..model import _swl_cases as cases                              # noqa: E402
from ..solve import _swl_synthetic as S                              # noqa: E402

CUDA = pytest.mark.skipif(os.environ.get("SWLI_TEST_CUDA") != "1",
                          reason="cuda only on request (SWLI_TEST_CUDA=1, GPU idle; the lead runs it)")


def field(grid, seed=4, scale=.08):
    from jd_curl.three_d.boundary import inject_free
    rng = np.random.default_rng(seed)
    return inject_free(scale*min(grid.spacing)*rng.normal(size=3*grid.n_free_scalar), grid)


# ---- gate 2 (torch64) --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("domain", ["support", "support_shared"])
def test_torch64_v_path_equals_the_weight_reference_bitwise(domain):
    grid, state, _, _ = cases.template()
    _, st = cases.with_support(state, np.zeros(grid.shape+(3,)), sigma_domain=domain)
    ref = cases.weight_reference(st)
    da, db = T.TorchModelState(st, "cpu"), T.TorchModelState(ref, "cpu")
    assert torch.equal(da.beta, db.beta) and torch.equal(da.cdat_weight, db.cdat_weight)
    for u in (np.zeros(grid.shape+(3,)), field(grid)):
        a, b = T.evaluate(da, u), T.evaluate(db, u)
        assert a.energy == b.energy and a.energies == b.energies
        assert torch.equal(T.gradient(da, a), T.gradient(db, b))
        v = field(grid, seed=9)
        assert torch.equal(T.hvp(da, a, v), T.hvp(db, b, v))
        assert torch.equal(T.gn_diagonal(da, a), T.gn_diagonal(db, b))
        assert T.c_dat(da, a) == T.c_dat(db, b)
    assert da.provenance()["sigma_domain"] == domain


# ---- gate 9 -------------------------------------------------------------------------------------------------------

def planted_levels(n_plant=6):
    """four_levels_structure with nodes of level 0 planted at the threshold: fixed g = t and moving s_M = t (channel 1)
    at n_plant mask nodes, so that Ghat = g + rho(0) is t up to the spline's interpolation rounding."""
    _, levels, _ = S.four_levels_structure()
    t = float(np.sqrt(np.float64(.05)))
    lv = levels[0]
    fixed, moving = np.array(lv.fixed), np.array(lv.moving)
    nodes = np.flatnonzero(lv.mask.ravel())[:n_plant]
    for array in (fixed, moving):
        array.reshape(array.shape[0], -1)[1, nodes] = t
    planted = lv.__class__(lv.level, lv.grid, fixed, moving, lv.mask, lv.tissue)
    return (planted,)+tuple(levels[1:])


def solve_l0(backend, levels, domain="support", max_levels=1, rounds=(1, 1, 1, 1), caps=Caps(2, 1), log=None):
    eta = eta_temporary(levels[-1].tissue, levels[-1].mask, levels[-1].grid)
    config = SolverConfig(rounds=rounds, caps=caps, sigma_domain=domain, max_levels=max_levels)
    return solve_multilevel(backend, list(levels), eta=eta, sigma=GaussianMStep(), config=config, log=log)


def test_planted_nodes_are_near_the_threshold():
    levels = planted_levels()
    lv = levels[0]
    from bayesreg.solve.multilevel import _template
    eta = eta_temporary(levels[-1].tissue, levels[-1].mask, levels[-1].grid)
    host = _template(lv, eta, levels[-1].grid.omega, 8192, lv.fixed.shape[0], 1)
    level = support_level(host, np.zeros(lv.grid.shape+(3,)), .05, sigma_domain="support")
    assert level.near_threshold[0] >= 6                     # |Ghat - t| <= 1e-12
    from bayesreg.model import channel_point
    point = channel_point(host, np.zeros(lv.grid.shape+(3,)))
    ghat = host.fixed_masked[1, :6]+point.rho[1, :6]
    assert np.max(np.abs(ghat-np.sqrt(np.float64(.05)))) <= 1e-14             # the planted nodes (gate 9 (i))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("domain", ["support", "support_shared"])
def test_v_is_the_same_on_both_backends(device, domain):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda not available")
    levels = planted_levels()
    a = solve_l0(NumpyBackend(), levels, domain)
    b = solve_l0(Torch64Backend(device=device), levels, domain)
    ra, rb = a.support[0], b.support[0]
    assert ra["n_near_threshold"] >= 6
    for key in ("v_sha256", "support_nodes", "N_c", "n_near_threshold", "seed_state_hash", "sigma2_start"):
        assert ra[key] == rb[key], key


def test_device_rho_perturbation_does_not_change_v(monkeypatch):
    levels = planted_levels()
    clean = solve_l0(Torch64Backend(device="cpu"), levels)
    original = T.model._channel_point
    calls = []

    def perturbed(dstate, u_cm):
        point = original(dstate, u_cm)
        if not isinstance(point, T.model.TorchChannelPoint):
            return point
        calls.append(1)
        rho = point.rho+1e-13
        return T.model.TorchChannelPoint(rho, point.grad, dstate.beta*rho)
    monkeypatch.setattr(T.model, "_channel_point", perturbed)
    dirty = solve_l0(Torch64Backend(device="cpu"), levels)
    # the perturbation did reach the device evaluations (review SWLI minor 2: a renamed function would make the
    # monkeypatch, and so this test, silently vacuous)
    assert calls
    assert (dirty.rounds[0]["F_B_trace"] != clean.rounds[0]["F_B_trace"]
            or dirty.rounds[0]["sigma2_post"] != clean.rounds[0]["sigma2_post"])
    assert dirty.support[0]["v_sha256"] == clean.support[0]["v_sha256"]
    assert dirty.support[0]["sigma2_start"] == clean.support[0]["sigma2_start"]   # the level start is CPU too
    # a v computed from the perturbed device rho would differ at the planted nodes
    lv = levels[0]
    from bayesreg.solve.multilevel import _template
    eta = eta_temporary(levels[-1].tissue, levels[-1].mask, levels[-1].grid)
    host = _template(lv, eta, levels[-1].grid.omega, 8192, lv.fixed.shape[0], 1)
    from bayesreg.model import channel_point
    point = channel_point(host, np.zeros(lv.grid.shape+(3,)))
    shifted = support_level(host, np.zeros(lv.grid.shape+(3,)), .05, sigma_domain="support",
                            rows=lambda c: np.array(point.rho[c])+1e-13)
    assert shifted.v_sha256 != clean.support[0]["v_sha256"]


@pytest.mark.parametrize("domain", ["support", "support_shared"])
def test_step_sequence_is_the_same(domain):
    _, levels, _ = S.four_levels_structure()
    rows = {}
    results = {}
    for name, backend in (("numpy", NumpyBackend()), ("torch64", Torch64Backend(device="cpu"))):
        log = []
        results[name] = solve_l0(backend, levels, domain, max_levels=2, caps=Caps(4, 2), log=log.append)
        rows[name] = log
    decisions = [[(r.get("level"), r.get("round"), r.get("outer"), r["iteration"], r["attempt"], r["accepted"],
                   r["reason"]) for r in log if r["kind"] == "candidate"] for log in rows.values()]
    assert decisions[0] == decisions[1] and decisions[0]
    a, b = results["numpy"], results["torch64"]
    assert (a.status, len(a.rounds)) == (b.status, len(b.rounds))
    for ra, rb in zip(a.rounds, b.rounds):
        assert np.allclose(ra["F_B_trace"], rb["F_B_trace"], rtol=1e-10, atol=0.)
    assert a.support[0]["v_sha256"] == b.support[0]["v_sha256"]


def _old_identity(state):
    """model_identity as written before SWLI (bayesreg/backends/torch64/stable.py at ddcbfce)."""
    digest = hashlib.sha256()
    parts = [state.grid.grid_id, repr(float(state.omega0)), repr(int(state.tile_size)),
             repr(float(state.alpha)), repr(float(state.nu)), array_sha256(state.sigma2), array_sha256(state.tau2),
             "gaussian" if state.weights is None else array_sha256(state.weights), array_sha256(state.fixed),
             array_sha256(state.mask), array_sha256(state.tissue), *[s.image_id for s in state.samplers]]
    if state.logvol_corners:
        parts += [state.logvol_points, repr(float(state.corner_weight))]
    elif state.logvol_closed27:
        parts += [state.logvol_points, repr(float(state.logvol_closed27_share)), state.closed27_weights]
    for part in parts:
        digest.update(part.encode())
        digest.update(b"|")
    return digest.hexdigest()


def test_cache_identity_contains_v():
    grid, state, _, _ = cases.template()
    assert ST.model_identity(state) == _old_identity(state)                  # brain: unchanged
    _, a = cases.with_support(state, np.zeros(grid.shape+(3,)))
    flipped = np.array(a.support_v)
    flipped[0, 0] ^= 1
    b = a.replace(support_v=flipped)
    assert a.support_v_sha256 != b.support_v_sha256 and np.array_equal(a.sigma2, b.sigma2)
    assert ST.model_identity(a) != ST.model_identity(b) != _old_identity(b)
    c = a.replace(support_threshold=.06)
    assert ST.model_identity(c) != ST.model_identity(a)
    d = a.replace(sigma_domain="support_shared")
    assert ST.model_identity(d) != ST.model_identity(a)
    backend = Torch64Backend(device="cpu")
    backend.prepare(a)
    backend.prepare(b)
    key_a = backend.comparator.cache_key(type("P", (), dict(state_hash="h"))(), model_id=backend.model_identity(a))
    key_b = backend.comparator.cache_key(type("P", (), dict(state_hash="h"))(), model_id=backend.model_identity(b))
    assert key_a != key_b
