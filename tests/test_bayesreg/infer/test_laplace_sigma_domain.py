"""Laplace under sigma_domain support / support_shared (work package SWLI; contract C04-6 gate 10, C14 appended note;
draft SWL 2026-10-31).  The brain samples' golden (gate 10, second half) is in solve/test_run_pair_sigma_domain.py
(GOLDEN_DDCBFCE_SWLI["laplace"]).

- a row with v = 0 contributes exactly 0 to K_c^T eps;
- the channel noise keeps its dimension and order: the right-hand side, the perturbations and their PCG records equal
  those of the "v as a float64 weight" reference bit for bit (the same draws, the same B);
- the system record names sigma_domain, theta and the sha256 of v.
"""
import numpy as np
import pytest

from bayesreg import model as M
from bayesreg.infer.laplace import LaplaceSystem, perturbation_rhs, sample_perturbations
from bayesreg.model import channel_vjp

from ..model import _swl_cases as cases
from ._cases import RecordingDraw


def _u(grid, seed=3, scale=.05):
    from jd_curl.three_d.boundary import inject_free
    rng = np.random.default_rng(seed)
    return inject_free(scale*min(grid.spacing)*rng.normal(size=3*grid.n_free_scalar), grid)


@pytest.mark.parametrize("domain", ["support", "support_shared"])
def test_rows_outside_the_support_contribute_exactly_zero(domain):
    grid, state, _, _ = cases.template()
    _, st = cases.with_support(state, np.zeros(grid.shape+(3,)), sigma_domain=domain)
    ev = M.evaluate(st, _u(grid))
    eps = np.random.default_rng(1).standard_normal(ev.channel.r.shape)
    eps[0] = 0.
    eps[1:][st.support_v == 1] = 0.                         # noise only on rows with v = 0
    assert np.count_nonzero(eps) > 0
    assert np.all(channel_vjp(st, ev.channel, eps) == 0.)


@pytest.mark.parametrize("domain", ["support", "support_shared"])
def test_samples_equal_the_weight_reference(domain):
    grid, state, _, _ = cases.template()
    _, st = cases.with_support(state, np.zeros(grid.shape+(3,)), sigma_domain=domain)
    ref = cases.weight_reference(st)
    u = _u(grid)
    draws = [RecordingDraw(5), RecordingDraw(5)]
    b = [perturbation_rhs(s, M.evaluate(s, u), d) for s, d in zip((st, ref), draws)]
    assert np.array_equal(b[0], b[1])
    assert [(n, s) for n, s, _ in draws[0].calls] == [(n, s) for n, s, _ in draws[1].calls]
    assert draws[0].calls[0][:2] == ("channel", (4, st.node_index.size))      # N rows per channel, as under brain
    a = sample_perturbations(st, u, np.random.default_rng(7), S=2)
    r = sample_perturbations(ref, u, np.random.default_rng(7), S=2)
    assert all(np.array_equal(x, y) for x, y in zip(a.deltas, r.deltas))
    assert list(a.iterations) == list(r.iterations)
    record = LaplaceSystem(st, u).record()
    assert record["sigma_domain"] == domain and record["support_v_sha256"] == st.support_v_sha256
    assert record["support_empty_channels"] == [3]
    assert "sigma_domain" not in LaplaceSystem(state.replace(sigma2=st.sigma2), u).record()      # brain: unchanged
