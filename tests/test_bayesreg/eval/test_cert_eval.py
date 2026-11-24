"""bayesreg.eval.cert_eval: the C07 certificate in the full and jacobian_only modes, and the per-cell pass."""
import hashlib

import numpy as np
import pytest

from jd_curl.three_d import feasibility as F
from jd_curl.three_d.grid import make_grid_3d

from bayesreg.eval import cert_eval as C

N = 7


def cube(n=N, h=1.):
    return make_grid_3d(nx=n, ny=n, nz=n, hx=h, hy=h, hz=h, unit="mm")


def b3_2_field(grid):
    """02_理论 B3(2): cell (2, 2, 2), local vertices (1,0,1), (1,1,1) moved by (0, 0, -1.05 h_z): J < 0 on a face."""
    u = np.zeros(grid.shape+(3,))
    u[3, 2, 3, 2] = u[3, 3, 3, 2] = -1.05*grid.spacing[2]
    return u


def smooth(grid, amplitude=.3):
    t = [np.linspace(0., 1., n) for n in grid.shape]
    z, y, x = np.meshgrid(*t, indexing="ij")
    bump = np.sin(np.pi*z)*np.sin(np.pi*y)*np.sin(np.pi*x)
    u = np.stack([amplitude*bump, -.8*amplitude*bump*np.cos(np.pi*z), .6*amplitude*bump], axis=-1)
    u[[0, -1]] = 0.                             # sin(pi) is not exactly 0 in floating point
    u[:, [0, -1]] = 0.
    u[:, :, [0, -1]] = 0.
    return u


def brain_mask(grid, inside=True):
    """Nodes 1..4 per axis (the lower corners of every cell the B3(2) field touches), or nodes 5..6 only."""
    mask = np.zeros(grid.shape, bool)
    if inside:
        mask[1:5, 1:5, 1:5] = True
    else:
        mask[5:7, 5:7, 5:7] = True
    return mask


def test_zero_field_full_and_counts():
    grid = cube()
    zero = np.zeros(grid.shape+(3,))
    rec = C.certify_field(zero, grid, mode="full", brain_nodes=brain_mask(grid))
    assert rec["passed"] and rec["status"] == "certified_feasible" and 0 < rec["min_jd_lower_bound"] <= 1
    assert rec["state_hash"] == hashlib.sha256(zero.tobytes(order="C")).hexdigest()
    assert rec["config"]["delta_hex"] == "0x0.0p+0" and rec["config"]["tile_size"] == 8192
    assert rec["faces_max_abs_u_mm"] == 0. and rec["domain"]["ok"]
    cells = rec["cells"]
    assert cells["matches_map"] and not cells["mismatch"]
    regions = cells["regions"]
    assert regions["all"]["cells"] == grid.n_cells == regions["brain"]["cells"]+regions["background"]["cells"]
    assert regions["brain"]["cells"] == 64 and regions["all"]["proven"] == grid.n_cells
    assert regions["all"]["infeasible"] == regions["all"]["uncertain"] == 0


@pytest.mark.parametrize("inside", [True, False])
def test_fold_is_infeasible_and_located(inside):
    grid = cube()
    u = b3_2_field(grid)
    rec = C.certify_field(u, grid, mode="full", brain_nodes=brain_mask(grid, inside=inside))
    assert not rec["passed"] and (rec["status"], rec["reason"]) == ("infeasible", "JACOBIAN_VIOLATION")
    cells = rec["cells"]
    assert cells["matches_map"] and cells["cell_pass"]["n_infeasible"] == rec["n_infeasible"] >= 1
    regions = cells["regions"]
    assert regions["brain"]["infeasible"]+regions["background"]["infeasible"] == rec["n_infeasible"]
    assert rec["witness_cell"] is not None
    assert regions["brain" if inside else "background"]["infeasible"] == rec["n_infeasible"]
    assert regions["brain"]["min_lower_bound"] is not None


def test_jacobian_only_equals_full_on_a_zero_face_field():
    grid = cube(9)
    for u in (smooth(grid), b3_2_field(grid)):
        full = C.certify_field(u, grid, mode="full", brain_nodes=brain_mask(grid))
        jac = C.certify_field(u, grid, mode="jacobian_only", brain_nodes=brain_mask(grid))
        for key in C.RESULT_FIELDS:
            assert full[key] == jac[key], key
        assert full["cells"]["regions"] == jac["cells"]["regions"]
    assert full["mode"] == "full" and jac["mode"] == "jacobian_only"


def test_nonzero_faces_and_domain():
    grid = cube()
    u = np.zeros(grid.shape+(3,))
    u[0, 3, 3, 2] = -.3                                  # a z = z0 face node pushed below the box
    full = C.certify_field(u, grid, mode="full", brain_nodes=brain_mask(grid))
    assert (full["status"], full["reason"], full["passed"]) == ("infeasible", "NONZERO_BOUNDARY", False)
    assert full["cells"]["map_reached_cells"] is False and full["cells"]["matches_map"] is None
    jac = C.certify_field(u, grid, mode="jacobian_only", brain_nodes=brain_mask(grid))
    assert jac["passed"] and jac["status"] == "certified_feasible" and jac["cells"]["matches_map"]
    assert jac["faces_max_abs_u_mm"] == .3 and jac["domain"] == dict(ok=False, reason="INVALID_DOMAIN", count=1)
    assert full["state_hash"] == jac["state_hash"]
    interior = np.zeros(grid.shape+(3,))
    interior[3, 3, 3, 0] = 50.                           # an interior node mapped outside the box
    rec = C.certify_field(interior, grid, mode="full")
    assert (rec["status"], rec["reason"]) == ("infeasible", "INVALID_DOMAIN")
    rec = C.certify_field(interior, grid, mode="jacobian_only")
    assert rec["status"] == "infeasible" and rec["reason"] == "JACOBIAN_VIOLATION" and not rec["domain"]["ok"]


def test_nonfinite_state():
    grid = cube()
    u = np.zeros(grid.shape+(3,))
    u[3, 3, 3, 1] = np.nan
    for mode in C.MODES:
        rec = C.certify_field(u, grid, mode=mode)
        assert (rec["status"], rec["reason"], rec["passed"]) == ("uncertain", "NONFINITE_STATE", False)
        assert rec["cells"] is None and rec["domain"]["reason"] == "NONFINITE_STATE"


def test_small_tiles_and_mode_check():
    grid = cube(9)
    u = smooth(grid, amplitude=.9)
    u[4, 4, 4] += (.4, -.3, .5)
    rec = C.certify_field(u, grid, mode="full", tile_size=16, brain_nodes=brain_mask(grid))
    assert rec["config"]["tile_size"] == 16 and rec["cells"]["matches_map"]
    with pytest.raises(ValueError):
        C.certify_field(u, grid, mode="partial")


def test_cell_pass_matches_certify_map_with_small_budgets():
    """With tiny subdivision budgets (test-only configurations) some cells stay uncertain; the per-cell pass counts
    them exactly as certify_map does."""
    rng = np.random.default_rng(7)
    grid = cube(9)
    u = np.zeros(grid.shape+(3,))
    u[1:-1, 1:-1, 1:-1] = rng.uniform(-.42, .42, (7, 7, 7, 3))
    seen = 0
    for config in (F.PrescreenCertificateConfig(delta=0., max_boxes=1), F.PrescreenCertificateConfig(delta=0., max_depth=1),
                   F.PrescreenCertificateConfig(delta=0., tile_size=32)):
        result = F.certify_map(u, grid, config=config)
        code, lower = C.cell_pass(u, grid, config)
        summary = C.cell_summary(code, lower, result)
        assert summary["matches_map"], summary["mismatch"]
        seen += result.n_uncertain+result.n_infeasible
    assert seen > 0
