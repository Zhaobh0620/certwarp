"""C13 stopping: copied helpers bit for bit, GF_B normalization, threshold, G0, status strings, the C13-7
continuation table, cap / threshold endings of an inner solve."""
import ast
import hashlib
from pathlib import Path
import subprocess

import numpy as np
import pytest
from scipy.linalg import norm

from jd_curl.three_d.boundary import restrict_free
from jd_curl.validation import as_float64_array, require_finite_result

from bayesreg.model import ModelState, build_samplers
from bayesreg.solve import Caps, Commit, NumpyBackend, solve_direct
from bayesreg.solve import stopping as st
from bayesreg.solve.backend import CertOutcome

from . import _synthetic as syn

REPO = Path(__file__).resolve().parents[3]


def original(commit, path, name, sha256):
    """Execute one function of an eb808be blob in a namespace with the helpers it names."""
    raw = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=REPO, capture_output=True, check=True).stdout
    assert hashlib.sha256(raw).hexdigest() == sha256
    text = raw.decode()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(np=np, norm=norm, as_float64_array=as_float64_array, require_finite_result=require_finite_result,
                     FloatArray=np.ndarray, __name__="eb808be_copy", __package__="jd_curl.three_d.solvers")
    exec(compile(ast.get_source_segment(text, node), f"{commit}:{path}", "exec"), namespace)
    return namespace[name], (node.lineno, node.end_lineno)


def test_copied_gradient_norm_mass_is_bitwise():
    commit, path, first, last = st.GNM_SOURCE
    fn, lines = original(commit, path, "gradient_norm_mass", st.GNM_SOURCE_SHA256)
    assert lines == (first, last)
    rng = np.random.default_rng(3)
    for scale in (1e-300, 1e-150, 1., 1e150):
        for omega in (1., .125, 7.3, 1e-9):
            g = scale*rng.normal(size=777)
            assert st.gradient_norm_mass(g, omega) == fn(g, omega)
    assert st.gradient_norm_mass(np.zeros(5), 2.) == fn(np.zeros(5), 2.) == 0.


def test_copied_step_ratio_is_bitwise():
    commit, path, first, last = st.STEP_SOURCE
    fn, lines = original(commit, path, "step_ratio", st.STEP_SOURCE_SHA256)
    assert lines == (first, last)
    rng = np.random.default_rng(4)
    for grid in (syn.cube_grid(7), syn.cube_grid(9, h=.7, origin=(-3., 1., 2.))):
        m = 3*grid.n_free_scalar
        for scale in (1e-14, 1., 1e3):
            delta, theta = scale*rng.normal(size=m), rng.normal(size=m)*grid.spacing[0]*40
            assert st.step_ratio(delta, theta, grid) == fn(delta, theta, grid)


def small_state(n=9, *, omega0=None, sigma2=1., alpha=1.):
    grid = syn.cube_grid(n)
    moving = syn.moving_image(grid.identity())
    fixed = syn.warp_with_spline(moving, grid, syn.solenoidal_displacement(grid, .4))
    return ModelState(grid=grid, omega0=grid.omega if omega0 is None else omega0, fixed=fixed[None],
                      samplers=build_samplers(moving[None], grid), mask=syn.brain_box(grid, 1),
                      sigma2=np.array([sigma2]), alpha=alpha, nu=alpha, tau2=np.full(4, 1.),
                      tissue=syn.uniform_tissue(grid))


def test_gf_b_normalization_and_threshold():
    """C13 gate: omega0 = 1 -> GF_B equals gradient_norm_mass(g, omega_ell) bit for bit; omega0 != 1 ->
    sqrt(omega0) times it (1 ulp); T_B = 1e-8 + 1e-6 G0."""
    g = np.random.default_rng(5).normal(size=300)
    assert st.gf_b(g, .7, 1.) == st.gradient_norm_mass(g, .7)
    for omega0 in (8., .3):
        value, reference = st.gf_b(g, .7, omega0), np.sqrt(omega0)*st.gradient_norm_mass(g, .7)
        assert abs(value-reference) <= 2*np.spacing(reference)
    assert st.threshold(0.) == 1e-8
    assert st.threshold(5.) == 1e-8+1e-6*5.
    with pytest.raises(Exception):
        st.threshold(-1.)


def test_g0_prior_blocks_vanish_at_identity():
    """C13 gate: at u = 0 the three prior blocks' gradients are <= 1e-14 G0 (K_Q 0 = 0, log 1 = 0, C 0 = 0)."""
    backend, state = NumpyBackend(), small_state()
    g0 = st.g0_of(backend, state)
    assert g0 > 0
    for block in ("diffusion", "logvol", "curl"):
        g = st.identity_gradient(backend, state, blocks=(block,))
        assert st.gf_b(g, state.grid.omega, state.omega0) <= 1e-14*g0
    channel = st.identity_gradient(backend, state, blocks=("channel",))
    assert st.gf_b(channel, state.grid.omega, state.omega0) == pytest.approx(g0, rel=1e-14)


def test_status_strings_are_closed():
    assert set(st.RUN_WORDING) == set(st.RUN_STATUSES)
    assert st.RUN_WORDING[st.CONVERGED_B] == "B 线收敛" and st.RUN_WORDING[st.CAP_REACHED] == "上限收尾"
    assert [s for s, w in st.RUN_WORDING.items() if "收敛" in w] == [st.CONVERGED_B]
    assert set(st.FINAL_STATUS) == set(st.CONTINUE_ALWAYS+st.CONTINUE_IF_SAFE)
    assert set(st.FINAL_STATUS.values()) <= set(st.RUN_STATUSES)
    assert set(st.CONTINUE_ALWAYS+st.CONTINUE_IF_SAFE) <= set(st.ROUND_STATUSES)


def fake_cert(u, *, bound=.5, digest=None):
    return CertOutcome(True, "certified_feasible", "positive_enclosure_cover", None, bound, bound, 0, 0, 1, None, None,
                       st.state_hash(u) if digest is None else digest, "v", 0.)


SAFE_ROWS = [("JACOBIAN_VIOLATION", "JACOBIAN_VIOLATION"), ("ENERGY_RESOLUTION", None), ("ARMIJO_REJECTED", None),
             ("FEASIBILITY_UNCERTAIN", "subdivision_budget"), ("INVALID_DOMAIN", "INVALID_DOMAIN"),
             ("NO_STRICT_DECREASE", None), ("MODEL_RATIO_REJECTED", None)]
UNSAFE_ROWS = [("NONPOSITIVE_GAUSS_JD", None), ("NONFINITE_STATE", "NONFINITE_STATE"),
               ("FEASIBILITY_UNCERTAIN", "arithmetic_uncertain"), ("FEASIBILITY_UNCERTAIN", "unsupported_arithmetic")]


@pytest.mark.parametrize("status", st.CONTINUE_ALWAYS)
def test_continuation_always_rows(status):
    decision = st.continuation(status, last_iteration=UNSAFE_ROWS)
    assert decision.proceed and st.final_status(status, decision) == st.FINAL_STATUS[status]


@pytest.mark.parametrize("status", [s for s in st.CONTINUE_IF_SAFE if s != st.OUTER_INNER_FAILED])
def test_continuation_conditional_rows(status):
    u = np.zeros((5, 5, 5, 3))
    ok = st.continuation(status, last_iteration=SAFE_ROWS, commit_u=u, commit_cert=fake_cert(u))
    assert ok.proceed and st.final_status(status, ok) == st.FINAL_STATUS[status]
    for row in UNSAFE_ROWS:
        bad = st.continuation(status, last_iteration=SAFE_ROWS+[row], commit_u=u, commit_cert=fake_cert(u))
        assert not bad.proceed and st.final_status(status, bad) == st.LEVEL_FAILED
    assert not st.continuation(status, last_iteration=SAFE_ROWS, commit_u=u,
                               commit_cert=fake_cert(u, digest="0"*64)).proceed          # state_hash mismatch
    assert not st.continuation(status, last_iteration=SAFE_ROWS, commit_u=u, commit_cert=fake_cert(u, bound=0.)).proceed
    assert not st.continuation(status, last_iteration=SAFE_ROWS).proceed                  # no commit


def test_continuation_outer_inner_failed_needs_search_failure():
    u = np.zeros((5, 5, 5, 3))
    for inner, expect in (("line_search_failed", True), ("trial_budget_exhausted", True),
                          ("nonfinite_evaluation", False), ("non_descent_direction", False)):
        decision = st.continuation(st.OUTER_INNER_FAILED, last_iteration=SAFE_ROWS, inner_status=inner, commit_u=u,
                                   commit_cert=fake_cert(u))
        assert decision.proceed is expect


@pytest.mark.parametrize("status", ["nonfinite_evaluation", "non_descent_direction", "exception", "unknown"])
def test_continuation_terminates_other_rows(status):
    u = np.zeros((5, 5, 5, 3))
    decision = st.continuation(status, last_iteration=[], commit_u=u, commit_cert=fake_cert(u))
    assert not decision.proceed and st.final_status(status, decision) == st.LEVEL_FAILED


def start_commit(backend, grid):
    zero = np.zeros(grid.shape+(3,))
    return Commit(zero, restrict_free(zero, grid), backend.certify(zero, grid))


def test_threshold_and_cap_endings():
    """C13 gate: a solve stops at once when GF_B <= T_B (zero steps, C13-6), at the cap with cap_reached,
    and at the threshold with converged_gradient (strongly regularized 9^3 problem)."""
    backend, state = NumpyBackend(), small_state()
    start = start_commit(backend, state.grid)
    g0 = st.g0_of(backend, state)
    zero_steps = solve_direct(backend, state, start, target=2*g0, caps=Caps())
    assert zero_steps.status == "converged_gradient" and zero_steps.accepted == 0
    assert np.array_equal(zero_steps.commit.u, start.u) and zero_steps.commit.cert is start.cert
    capped = solve_direct(backend, state, start, target=st.threshold(g0), caps=Caps(max_accepted=2))
    assert capped.status == st.CAP_REACHED and capped.accepted == 2 and capped.cap == "max_accepted"
    assert capped.gradient_norm > capped.target
    done = solve_direct(backend, state, start, target=st.threshold(g0), caps=Caps())
    assert done.status == "converged_gradient" and 0 < done.accepted < 60
    assert done.gradient_norm <= done.target == st.threshold(g0)
    assert done.commit.cert.passed and done.commit.cert.state_hash == st.state_hash(done.commit.u)
