"""C16 export: u (mm, xyz) <-> d (voxels, zyx) round trip, file keys and shapes, state_hash, no overwrite."""
import hashlib
import json

import numpy as np
import pytest

from jd_curl.validation import InputValidationError
from jd_curl.three_d.grid import make_grid_3d

from bayesreg.solve import NumpyBackend, displacement_to_voxels, voxels_to_displacement, write_disp, write_final_state
from bayesreg.solve.export import JsonlLog, write_json

from . import _synthetic as syn


def anisotropic():
    return make_grid_3d(nx=9, ny=8, nz=7, hx=.9, hy=1.1, hz=1.3, x0=-4., y0=2., z0=.5, unit="mm")


def test_round_trip_unit_spacing_is_bitwise():
    grid = syn.cube_grid(9)
    u = syn.solenoidal_displacement(grid, .7)
    d = displacement_to_voxels(u, grid)
    assert d.shape == (3,)+grid.shape and d.dtype == np.float64
    assert np.array_equal(d[0], u[..., 2]) and np.array_equal(d[2], u[..., 0])        # d is zyx, u is xyz
    assert voxels_to_displacement(d, grid).tobytes() == u.tobytes()


def test_round_trip_anisotropic_within_one_ulp():
    grid = anisotropic()
    u = syn.true_displacement(grid, (.5, -.4, .3))
    d = displacement_to_voxels(u, grid)
    hz, hy, hx = grid.spacing[::-1]
    assert np.array_equal(d[0], u[..., 2]/hz) and np.array_equal(d[1], u[..., 1]/hy) and np.array_equal(d[2], u[..., 0]/hx)
    back = voxels_to_displacement(d, grid)
    assert np.all(np.abs(back-u) <= np.spacing(np.abs(u)))


def test_files_keys_hash_and_no_overwrite(tmp_path):
    grid = anisotropic()
    u = syn.true_displacement(grid, (.5, -.4, .3))
    digest = write_final_state(tmp_path/"final_native.npz", u, grid, status="cap_reached")
    assert digest == hashlib.sha256(u.tobytes(order="C")).hexdigest() == NumpyBackend().certify(u, grid).state_hash
    with np.load(tmp_path/"final_native.npz") as z:
        assert set(z.files) == {"u", "state_hash", "grid", "status"}
        assert z["u"].dtype == np.float64 and z["u"].tobytes() == u.tobytes() and str(z["status"]) == "cap_reached"
        assert json.loads(str(z["grid"]))["spacing_xyz"] == list(grid.spacing)
    meta = write_disp(tmp_path/"disp.npz", u, grid, state_hash=digest)
    with np.load(tmp_path/"disp.npz") as z:
        assert set(z.files) == {"d", "meta"} and z["d"].shape == (3, 7, 8, 9)
        assert json.loads(str(z["meta"]))["u_state_hash"] == digest and meta["exact_inverse"] is False
    for writer in (lambda: write_final_state(tmp_path/"final_native.npz", u, grid, status="x"),
                   lambda: write_disp(tmp_path/"disp.npz", u, grid, state_hash=digest),
                   lambda: write_json(tmp_path/"disp.npz", {}), lambda: JsonlLog(tmp_path/"disp.npz")):
        with pytest.raises(FileExistsError):
            writer()
    face = u.copy()
    face[0, 1, 1, 0] = .1
    with pytest.raises(InputValidationError):
        write_final_state(tmp_path/"other.npz", face, grid, status="x")


def test_json_is_strict(tmp_path):
    path = write_json(tmp_path/"a.json", dict(x=np.float64(np.nan), y=np.int64(3), z=(1., np.inf), w=np.longdouble(.5)))
    assert json.loads(path.read_text()) == dict(x=None, y=3, z=[1., None], w=.5)
