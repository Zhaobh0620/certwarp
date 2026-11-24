"""C03.6 / C12: bayesreg/grids.py against the eb808be originals, plus hand-computed checks."""
import ast
import hashlib
import inspect
from pathlib import Path
import subprocess

import numpy as np
import pytest

from bayesreg import grids
from jd_curl.validation import InputValidationError, as_float64_array, readonly_copy
from jd_curl.three_d.grid import Grid3D, make_grid_3d
from jd_curl.three_d.nifti import coarse_grid
from jd_curl.three_d.validation import zero_boundary

REPO = Path(__file__).resolve().parents[3]
NAMES = ("FACTORS", "four_level_grids", "prolong_displacement")
LINES = dict(FACTORS=(25, 25), four_level_grids=(45, 52), prolong_displacement=(55, 86))


@pytest.fixture(scope="module")
def original():
    """Only the two functions (and FACTORS) of eb808be multiresolution.py, executed in a namespace
    holding the helpers they use; the rest of that module imports archived solvers."""
    raw = subprocess.run(["git", "show", f"{grids.SOURCE_COMMIT}:{grids.SOURCE_PATH}"], cwd=REPO,
                         capture_output=True, check=True).stdout
    assert hashlib.sha256(raw).hexdigest() == grids.SOURCE_SHA256
    text = raw.decode()
    segments = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name in NAMES:
            segments[node.name] = (ast.get_source_segment(text, node), node.lineno, node.end_lineno)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FACTORS" for t in node.targets):
            segments["FACTORS"] = (ast.get_source_segment(text, node), node.lineno, node.end_lineno)
    assert set(segments) == set(NAMES)
    namespace = dict(np=np, Grid3D=Grid3D, InputValidationError=InputValidationError,
                     as_float64_array=as_float64_array, readonly_copy=readonly_copy,
                     coarse_grid=coarse_grid, zero_boundary=zero_boundary)
    exec(compile("\n\n".join(segments[name][0] for name in NAMES), "eb808be:multiresolution.py", "exec"), namespace)
    return segments, namespace


HELPERS = (InputValidationError, as_float64_array, readonly_copy, Grid3D, coarse_grid, zero_boundary)


def git_blob(path):
    return subprocess.run(["git", "show", f"{grids.SOURCE_COMMIT}:{path}"], cwd=REPO, capture_output=True,
                          check=True).stdout


def jd_curl_dependencies(relpath):
    """Repository paths of the jd_curl modules and packages that importing `relpath` loads directly:
    absolute and relative imports (TYPE_CHECKING blocks included), plus the parent __init__.py files."""
    package = relpath.split("/")[:-1]
    found = {"/".join(package[:k])+"/__init__.py" for k in range(1, len(package)+1)}
    for node in ast.walk(ast.parse((REPO/relpath).read_text())):
        if isinstance(node, ast.Import):
            dotted = [alias.name.split(".") for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = package[:len(package)-node.level+1] if node.level else []
            module = base+(node.module.split(".") if node.module else [])
            dotted = [module]+[module+[alias.name] for alias in node.names]
        else:
            continue
        for parts in dotted:
            if parts and parts[0] == "jd_curl":
                for candidate in ("/".join(parts)+".py", "/".join(parts)+"/__init__.py"):
                    if (REPO/candidate).is_file():
                        found.add(candidate)
    return found


def test_helper_modules_are_the_eb808be_blobs():
    """The exec namespace of `original` (and bayesreg/grids.py) uses jd_curl helpers from this working
    tree; every module they come from, followed transitively through its jd_curl imports, must be
    byte-identical to its eb808be blob. The archive copy of multiresolution.py must equal it too."""
    pending = set()
    for helper in HELPERS:
        source = Path(inspect.getsourcefile(helper)).resolve()
        assert source.is_relative_to(REPO), f"{helper.__name__} is imported from outside this tree: {source}"
        pending.add(source.relative_to(REPO).as_posix())
    checked = set()
    while pending:
        relpath = pending.pop()
        checked.add(relpath)
        assert (REPO/relpath).read_bytes() == git_blob(relpath), f"{relpath} differs from {grids.SOURCE_COMMIT}"
        pending |= jd_curl_dependencies(relpath)-checked
    assert {"jd_curl/validation.py", "jd_curl/three_d/grid.py", "jd_curl/three_d/nifti.py",
            "jd_curl/three_d/validation.py", "jd_curl/__init__.py", "jd_curl/three_d/__init__.py"} <= checked
    archive = REPO/"archive/seq_vp_3d"/grids.SOURCE_PATH
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == grids.SOURCE_SHA256


def oasis_native(origin=(0., 0., 0.)):
    return make_grid_3d(nx=160, ny=192, nz=224, hx=1., hy=1., hz=1.,
                        x0=origin[0], y0=origin[1], z0=origin[2], unit="mm")


def test_copy_is_verbatim_with_recorded_line_numbers(original):
    segments, namespace = original
    for name in NAMES:
        assert segments[name][1:] == LINES[name]
    assert namespace["FACTORS"] == grids.FACTORS == (8, 4, 2, 1)
    for name in ("four_level_grids", "prolong_displacement"):
        assert inspect.getsource(getattr(grids, name)).rstrip("\n") == segments[name][0]


@pytest.mark.parametrize("origin", [(0., 0., 0.), (-79.5, -95.5, -111.5), (3.25, -7., 12.125)])
def test_oasis_four_levels_match_formula_and_keep_fov(original, origin):
    native = oasis_native(origin)
    assert native.shape == (224, 192, 160)
    levels = grids.four_level_grids(native)
    # hand-computed (n - 2 + s)//s + 1 for (nz, ny, nx) = (224, 192, 160)
    assert [g.shape for g in levels] == [(29, 25, 21), (57, 49, 41), (113, 97, 81), (224, 192, 160)]
    for grid, s in zip(levels, grids.FACTORS):
        assert grid.shape == tuple((n-2+s)//s+1 for n in native.shape)
        assert grid.unit == "mm"
        for coarse_axis, native_axis in zip(grid.axes, native.axes):
            assert coarse_axis[0] == native_axis[0]
            tolerance = 32*np.finfo(float).eps*max(1., abs(native_axis[-1]))
            assert abs(coarse_axis[-1]-native_axis[-1]) <= tolerance
    assert levels[-1] is native
    reference = original[1]["four_level_grids"](native)
    assert [g.grid_id for g in levels] == [g.grid_id for g in reference]
    assert all(g.spacing == r.spacing and all(np.array_equal(a, b) for a, b in zip(g.axes, r.axes))
               for g, r in zip(levels, reference))


@pytest.mark.parametrize("n", [(26, 27, 28), (33, 26, 41)])
def test_four_level_grids_match_original_on_anisotropic_grids(original, n):
    native = make_grid_3d(nx=n[0], ny=n[1], nz=n[2], hx=.9, hy=1.1, hz=1.3, x0=-2., y0=5., z0=.5, unit="mm")
    ours, theirs = grids.four_level_grids(native), original[1]["four_level_grids"](native)
    assert [g.grid_id for g in ours] == [g.grid_id for g in theirs]


def test_four_level_grids_rejections_match_original(original):
    too_small = make_grid_3d(nx=25, ny=30, nz=30, hx=1., hy=1., hz=1., unit="mm")   # (25-2+8)//8+1 = 4 < 5
    not_mm = make_grid_3d(nx=30, ny=30, nz=30, hx=1., hy=1., hz=1., unit="unit")
    for grid in (too_small, not_mm):
        for function in (grids.four_level_grids, original[1]["four_level_grids"]):
            with pytest.raises(InputValidationError):
                function(grid)


def small_pair():
    native = make_grid_3d(nx=13, ny=12, nz=11, hx=.9, hy=1.1, hz=1.3, x0=-2., y0=5., z0=.5, unit="mm")
    return coarse_grid(native, (6, 7, 7)), native


def random_zero_boundary(shape, seed):
    u = np.random.default_rng(seed).normal(size=shape+(3,))
    u[[0, -1]] = 0
    u[:, [0, -1]] = 0
    u[:, :, [0, -1]] = 0
    return u


def test_prolong_matches_original_bitwise(original):
    source, target = small_pair()
    middle = coarse_grid(target, (8, 9, 10))
    for seed, (a, b) in enumerate([(source, target), (source, middle), (middle, target), (target, target)]):
        u = random_zero_boundary(a.shape, seed)
        ours = grids.prolong_displacement(u, a, b)
        theirs = original[1]["prolong_displacement"](u, a, b)
        assert ours.dtype == theirs.dtype == np.float64 and not ours.flags.writeable
        assert np.array_equal(ours, theirs)
        assert zero_boundary(ours)


def test_prolong_reproduces_q1_product_field():
    """A tensor product of per-axis piecewise-linear nodal profiles (zero at both ends) is a Q1 field
    of the source grid; its value at any target node is the product of 1D linear interpolants."""
    source, target = small_pair()
    rng = np.random.default_rng(7)
    profiles = []
    for axis in source.axes:                       # x, y, z
        g = rng.normal(size=len(axis))
        g[[0, -1]] = 0
        profiles.append(g)
    c = np.array([.7, -1.3, .4])
    gx, gy, gz = profiles
    u = c*(gz[:, None, None]*gy[None, :, None]*gx[None, None, :])[..., None]
    fine = [np.interp(t, s, g) for t, s, g in zip(target.axes, source.axes, profiles)]
    expected = c*(fine[2][:, None, None]*fine[1][None, :, None]*fine[0][None, None, :])[..., None]
    result = grids.prolong_displacement(u, source, target)
    assert np.max(np.abs(result-expected)) <= 1e-14*np.max(np.abs(expected))


def test_prolong_reproduces_affine_field_on_inner_box():
    """u_a = (A x_a + b) on interior source nodes, 0 on the six faces. Inside the box spanned by the
    second and second-to-last source nodes the Q1 interpolant is exactly A x + b."""
    source, target = small_pair()
    A = np.array([[.02, -.01, .03], [.015, -.02, .01], [-.03, .005, .025]])
    b = np.array([.1, -.2, .05])
    u = (source.identity() @ A.T + b)
    u[[0, -1]] = 0
    u[:, [0, -1]] = 0
    u[:, :, [0, -1]] = 0
    result = grids.prolong_displacement(u, source, target)
    x = target.identity()
    inner = np.ones(target.shape, dtype=bool)
    for j, axis in enumerate(source.axes):
        inner &= (x[..., j] >= axis[1]) & (x[..., j] <= axis[-2])
    assert inner.sum() >= 8*7*6
    expected = x @ A.T + b
    assert np.max(np.abs(result[inner]-expected[inner])) <= 1e-14*np.max(np.abs(expected))


def test_prolong_rejections_match_original(original):
    source, target = small_pair()
    bad_boundary = random_zero_boundary(source.shape, 3)
    bad_boundary[0, 2, 2, 1] = 1e-3
    other_fov = make_grid_3d(nx=13, ny=12, nz=11, hx=.9, hy=1.1, hz=1.31, x0=-2., y0=5., z0=.5, unit="mm")
    u = random_zero_boundary(source.shape, 4)
    for function in (grids.prolong_displacement, original[1]["prolong_displacement"]):
        with pytest.raises(InputValidationError):
            function(bad_boundary, source, target)
        with pytest.raises(InputValidationError):
            function(u, source, other_fov)
        with pytest.raises(InputValidationError):
            function(random_zero_boundary(target.shape, 5), target, source)       # coarsening
