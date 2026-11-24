"""E12 allocation/reference reuse: independent layout and stream regressions."""
from dataclasses import replace

import numpy as np
import pytest

from jd_curl.three_d.cell_q1 import CellQ1Operators3D, cell_tiles
from jd_curl.three_d.geometry import cofactor, compute_cell_geometry
from jd_curl.three_d.problem import Controls3D
from jd_curl.three_d.residuals import (evaluate_trial, residual_blocks,
                                     _residual_block_pairs)
from jd_curl.validation import InputValidationError, StateMismatchError
from .b_support import case


@pytest.mark.parametrize("location", ["gauss27", "corners8", "closed27_uniform"])
@pytest.mark.parametrize("tile_size", [1, 7, 8192])
def test_geometry_exact_scalar_operator_and_cross_oracle(grid3d, location, tile_size):
    ops = CellQ1Operators3D(grid3d, location=location)
    u = np.random.default_rng(12).normal(scale=.01, size=grid3d.shape+(3,))
    before = u.copy()
    for tile in cell_tiles(grid3d, tile_size):
        observed = compute_cell_geometry(u, ops, tile=tile)
        F = np.stack([np.stack([ops.derivative(u[..., r], axis, tile=tile)
                               for axis in ("x", "y", "z")], axis=-1)
                      for r in range(3)], axis=-2)+np.eye(3)
        A = np.stack([np.cross(F[..., j, :], F[..., k, :])
                      for j, k in ((1, 2), (2, 0), (0, 1))], axis=-2)
        J = np.sum(F[..., 0, :]*A[..., 0, :], axis=-1)
        curl = np.stack((F[..., 2, 1]-F[..., 1, 2], F[..., 0, 2]-F[..., 2, 0],
                         F[..., 1, 0]-F[..., 0, 1]), axis=-1)
        for name, expected in zip(("F", "A", "J", "curl"), (F, A, J, curl)):
            actual = getattr(observed, name)
            np.testing.assert_array_equal(actual, expected)
            assert actual.flags.owndata and actual.flags.c_contiguous
            assert not actual.flags.writeable and not np.shares_memory(actual, u)
    np.testing.assert_array_equal(u, before)
    assert u.flags.writeable


@pytest.mark.parametrize("layout", ["scalar", "strided", "fortran"])
def test_cofactor_layout_and_extreme_values(layout):
    F = np.random.default_rng(3).normal(size=(4, 7, 3, 3))
    if layout == "scalar": F = F[0, 0]
    elif layout == "strided": F = F[:, ::2, ::-1, :]
    else: F = np.asfortranarray(F)
    expected = np.stack([np.cross(F[..., j, :], F[..., k, :])
                         for j, k in ((1, 2), (2, 0), (0, 1))], axis=-2)
    np.testing.assert_array_equal(cofactor(F), expected)
    with pytest.raises(FloatingPointError): cofactor(np.full((3, 3), 1e308))
    with pytest.raises(InputValidationError): cofactor(np.full((3, 3), np.nan))


@pytest.mark.parametrize("mu,nu", [(1., 1.), (.001, .001), (.7, .03)])
@pytest.mark.parametrize("outer_only", [False, True])
@pytest.mark.parametrize("step", [0., 1e-10, .2])
def test_paired_stream_matches_canonical_and_reference_evaluated_once(mu, nu, outer_only, step, monkeypatch):
    p, b, _ = case(mu=mu, nu=nu, tile=7)
    t = evaluate_trial(p, (1-step)*b.theta)
    expected = list(zip(residual_blocks(p, b), residual_blocks(p, t), strict=True))
    calls = []
    original = Controls3D._gauss_reference

    def observed(self, tile):
        calls.append((tile.start, tile.stop))
        return original(self, tile)

    monkeypatch.setattr(Controls3D, "_gauss_reference", observed)
    actual = list(_residual_block_pairs(p, b, t, outer_only=outer_only))
    if outer_only: expected = expected[:1]
    assert len(actual) == len(expected)
    for pair, reference in zip(actual, expected):
        for a, e in zip(pair, reference):
            assert (a.name, a.tile, a.problem_id, a.eval_id) == (e.name, e.tile, e.problem_id, e.eval_id)
            np.testing.assert_array_equal(a.values, e.values)
    assert calls == ([] if outer_only else [(s.start, s.stop) for s in cell_tiles(p.grid, 7)])
    with pytest.raises(StateMismatchError):
        list(_residual_block_pairs(p, b, replace(t, problem_id="stale")))


def test_outer_stream_does_not_compute_proximity():
    p, b, _ = case()
    # Deliberately invalid unused soft reference: only the image block belongs
    # to F. This is an internal stream test, not a factory-valid problem.
    ref = np.full_like(p.controls.u_ref, np.nan)
    bad = replace(p, controls=replace(p.controls, u_ref=ref))
    assert len(list(_residual_block_pairs(bad, b, b, outer_only=True))) == 1
    with pytest.raises(FloatingPointError):
        list(_residual_block_pairs(bad, b, b))
