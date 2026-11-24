"""Digests of the model, torch64 and solver paths for the C05-2a bitwise gate (path "pre_def") and for the defaults
of work package DEF (path "default").

Each digest is the sha256 of a JSON list of exact representations: float64 by float.hex, longdouble by
np.format_float_scientific(unique=True), arrays by the sha256 of their C-order float64 bytes.  The golden values in
the tests (GOLDEN_481E33B in model/test_logvol_corners.py, torch64/test_torch64_corners.py and
solve/test_logvol_points.py) are these functions on the code before C05-2a (commit 481e33b), numpy 2.0.2, torch
2.5.1, single-threaded BLAS / OpenMP, torch device cpu, this machine.  Generation (FB1; review LVC_CPUPAR important
item 1): the code of 481e33b is extracted into a temporary directory, this module is copied in, and a child Python
computes the seven digests there:
    python tests/test_bayesreg/_golden/make_golden.py --ref 481e33b --out tests/test_bayesreg/_golden/golden_481e33b.json
The record golden_481e33b.json (command, commits, environment, digests) is compared with the three test constants by
model/test_logvol_corners.py::test_golden_constants_match_the_generation_record.

DEF (2026-10-31; contract 8a4d060: C05-2b first paragraph, C09.1-3a last item, D43, D47, D48): the ModelState default
became logvol_points "gauss27+closed27" (beta_S 1, Simpson) and the JDScalingConfig default jd_cell_estimate
"certificate".  The GOLDEN_481E33B path -- the default before DEF -- is now given explicitly (`path_options("pre_def")`:
`pre_def_model()` = logvol_points "gauss27", `pre_def_pg()` = jd_cell_estimate "gauss27+corners"); on code that
predates an option (481e33b has neither) the explicit form is empty, the code's only path.  The record of this module
after the DEF edit, generated again on 481e33b with the same seven digests, is golden_481e33b_def.json (the record
before the edit, golden_481e33b.json, is kept).  Path "def_explicit" gives the new defaults explicitly
(gauss27+closed27, beta_S 1, Simpson; jd_cell_estimate certificate); its digests were generated once on the code
before DEF (8a4d060, where these options already existed): golden_8a4d060_def_explicit.json, constants GOLDEN_DEF.
Path "default" (no options: the code's defaults) must give the same digests (tests test_def_default_*).
"""
from dataclasses import fields
import hashlib
import json

import numpy as np

from jd_curl.three_d.boundary import inject_free, restrict_free
from jd_curl.three_d.grid import make_grid_3d

from bayesreg.model import BLOCK_NAMES, evaluate, gradient, hvp, stable_reduction

from ._helpers import aniso, feasible_field, free_random, make_state

GAUSS_NAMES = ("diffusion", "logvol", "curl")
PATHS = ("pre_def", "default", "def_explicit")


def _has_field(cls, name):
    return name in {f.name for f in fields(cls)}


def pre_def_model():
    """ModelState options of the GOLDEN_481E33B path: logvol_points "gauss27" (C05-2, the default before DEF), explicit
    since DEF; {} on code without the option (481e33b: C05-2 is its only log-volume prior)."""
    from bayesreg.model import ModelState
    return dict(logvol_points="gauss27") if _has_field(ModelState, "logvol_points") else {}


def pre_def_pg():
    """PGConfig of the GOLDEN_481E33B path: jd_cell_estimate "gauss27+corners" (the default before DEF), explicit since
    DEF; the default PGConfig on code without the option (481e33b: that estimate is its only one)."""
    from bayesreg.solve import JDScalingConfig, PGConfig
    if _has_field(JDScalingConfig, "cell_estimate"):
        return PGConfig(jd=JDScalingConfig(cell_estimate="gauss27+corners"))
    return PGConfig()


def def_explicit_model():
    """ModelState options of the DEF defaults given explicitly: gauss27+closed27, beta_S 1, Simpson (C05-2b, D48)."""
    return dict(logvol_points="gauss27+closed27", logvol_closed27_share=1., closed27_weights="simpson")


def def_explicit_pg():
    """PGConfig of the DEF defaults given explicitly: affine_v1 with jd_cell_estimate "certificate" (D47)."""
    from bayesreg.solve import JDScalingConfig, PGConfig
    return PGConfig(metric_jd_scaling="affine_v1", jd=JDScalingConfig(cell_estimate="certificate"))


def path_options(path):
    """(ModelState options, PGConfig or None = the solver's default) of a digest path: "pre_def" the defaults before
    DEF given explicitly (empty where the code predates them), "default" the code's defaults (no options),
    "def_explicit" the defaults since DEF given explicitly (computable on the code before DEF, 8a4d060: GOLDEN_DEF)."""
    if path == "pre_def":
        return pre_def_model(), pre_def_pg()
    if path == "default":
        return {}, None
    if path == "def_explicit":
        return def_explicit_model(), def_explicit_pg()
    raise ValueError(f"path: one of {PATHS}")


def ld(value):
    return np.format_float_scientific(np.longdouble(value), unique=True)


def array_digest(array):
    array = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def finish(parts):
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def small_case(**options):
    """The aniso 7 x 6 x 5 grid, Student-t, tile_size 7 (18 tiles) -- the torch64 'small' case."""
    g = aniso()
    state = make_state(g, seed=0, student_t=True, tile_size=7)
    if options:
        state = state.replace(**options)
    return state, feasible_field(g, np.random.default_rng(1))


def box12_case(**options):
    """The 12 x 11 x 10 grid, Gaussian, default tile (1 tile) -- the 'box12' case of model/test_logvol_corners (moved
    here in FB1 so that make_golden.py builds it on the old code)."""
    g = make_grid_3d(nx=12, ny=11, nz=10, hx=.5, hy=.45, hz=.6, x0=.1, y0=-.4, z0=.3, unit="mm")
    state = make_state(g, seed=3, student_t=False)
    if options:
        state = state.replace(**options)
    return state, feasible_field(g, np.random.default_rng(4), scale=.12)


def directions(state, u, seed):
    g = state.grid
    rng = np.random.default_rng(seed)
    v = free_random(g, rng)
    direction = inject_free(free_random(g, rng), g)
    direction *= np.max(np.abs(u))/np.max(np.abs(direction))
    return v, u+.3*direction


def evaluation_parts(ev):
    return [ld(ev.total), ev.energy.hex(), [e.hex() for e in ev.energies["channel"]],
            [ev.energies[name].hex() for name in GAUSS_NAMES]]


def reduction_parts(red):
    return [ld(red.delta), float(red.ared).hex(), float(red.guard).hex(), [ld(c) for c in red.blocks["channel"]],
            [ld(red.blocks[name]) for name in GAUSS_NAMES]]


def model_digest(state, u, seed=7):
    """evaluate, gradient (total and per block), GN HVP (total and per block), stable_reduction of the numpy model."""
    v, u_trial = directions(state, u, seed)
    ev = evaluate(state, u)
    parts = evaluation_parts(ev)
    parts.append([array_digest(gradient(state, ev))]+[array_digest(gradient(state, ev, blocks=(b,)))
                                                       for b in BLOCK_NAMES])
    parts.append([array_digest(hvp(state, ev, v))]+[array_digest(hvp(state, ev, v, blocks=(b,))) for b in BLOCK_NAMES])
    trial = evaluate(state, u_trial)
    parts.append(evaluation_parts(trial))
    parts.append(reduction_parts(stable_reduction(state, ev, trial)))
    return finish(parts)


def torch_digest(state, u, seed=7, **options):
    """The same on the torch64 backend on the cpu device, plus the GPU-path stable comparison (C08-8(a)(b))."""
    import bayesreg.backends.torch64 as T
    from bayesreg.backends.torch64.stable import gpu_fb_change
    v, u_trial = directions(state, u, seed)
    dstate = T.TorchModelState(state, "cpu", **options)
    ev = T.evaluate(dstate, u)
    parts = evaluation_parts(ev)+[array_digest(ev.tile_sums)]
    parts.append([array_digest(T.to_numpy(T.gradient(dstate, ev)))]
                 + [array_digest(T.to_numpy(T.gradient(dstate, ev, blocks=(b,)))) for b in BLOCK_NAMES])
    parts.append([array_digest(T.to_numpy(T.hvp(dstate, ev, v)))]
                 + [array_digest(T.to_numpy(T.hvp(dstate, ev, v, blocks=(b,)))) for b in BLOCK_NAMES])
    trial = T.evaluate(dstate, u_trial)
    parts.append(evaluation_parts(trial)+[array_digest(trial.tile_sums)])
    delta, blocks = gpu_fb_change(dstate, ev, trial)
    parts.append([ld(delta), [ld(c) for c in blocks["channel"]], [ld(blocks[name]) for name in GAUSS_NAMES]])
    return finish(parts)


def trajectory_digest(rows, result):
    """The candidate / accept / commit rows of one solve (as tests/test_bayesreg/solve/test_jd_scaling.digest)."""
    keep = []
    for r in rows:
        if r["kind"] == "candidate":
            keep.append(["c", r["iteration"], r["attempt"], float(r["tau"]).hex(), r["reason"],
                         None if r["energy"] is None else float(r["energy"]).hex(),
                         None if r["min_jd_lower_bound"] is None else float(r["min_jd_lower_bound"]).hex()])
        elif r["kind"] == "accept":
            keep.append(["a", r["iteration"], float(r["tau"]).hex(), float(r["energy"]).hex(), r["state_hash"]])
        elif r["kind"] in ("commit", "commit_check"):
            keep.append([r["kind"], r["outer"], r.get("state_hash")])
    return finish(dict(rows=keep, status=result.status, hash=result.commit.cert.state_hash,
                       energies=[float(e).hex() for e in result.energies]))


def mild_problem(**options):
    """The 9^3 near-fold synthetic problem of the JDS tests with a mild u* (amplitude (.3, -.2, .25) voxels)."""
    from bayesreg.model import ModelState, build_samplers
    from ..solve import _synthetic as syn
    grid = syn.cube_grid(9)
    moving = syn.moving_image(grid.identity())
    fixed = syn.warp_with_spline(moving, grid, syn.true_displacement(grid, (.3, -.2, .25)))
    state = ModelState(grid=grid, omega0=grid.omega, fixed=fixed[None], samplers=build_samplers(moving[None], grid),
                       mask=syn.brain_box(grid, 1), sigma2=np.array([1e-2]), alpha=1e-3, nu=1e-3,
                       tau2=np.full(4, 1.), tissue=syn.uniform_tissue(grid), **options)
    return grid, state


def solve_run(option, backend, steps, *, pg=None, **options):
    """solve_direct (option a) or solve_proximal (option b) from u = 0 with `pg` (None: the default PGConfig)."""
    from bayesreg.solve import (Caps, Commit, ProximalParams, default_lambda, g0_of, solve_direct, solve_proximal,
                                threshold)
    grid, state = mild_problem(**options)
    state = backend.prepare(state)
    zero = np.zeros(grid.shape+(3,))
    start = Commit(zero, restrict_free(zero, grid), backend.certify(zero, grid))
    target = threshold(g0_of(backend, state))
    rows = []
    given = {} if pg is None else dict(pg=pg)
    if option == "a":
        result = solve_direct(backend, state, start, target=target, caps=Caps(max_accepted=steps), log=rows.append,
                              **given)
    else:
        result = solve_proximal(backend, state, start, target=target, caps=Caps(max_accepted=steps, max_outer=3),
                                params=ProximalParams(lam=default_lambda(grid)), log=rows.append, **given)
    return result, rows


def torch_case(name, path):
    """torch64/_cases.case(name) on a digest path: (state, u, torch-state options)."""
    from ..torch64._cases import case
    model_options, _ = path_options(path)
    state, u, options = case(name)
    if model_options:
        state = state.replace(**model_options)
    return state, u, options


def solve_key(key, path):
    """One trajectory digest of a path: key numpy_a, numpy_b or torch64_a (6 accepted steps, cap_reached required)."""
    from bayesreg.solve import NumpyBackend, make_backend
    model_options, pg = path_options(path)
    backend_name, option = key.split("_")
    backend = NumpyBackend() if backend_name == "numpy" else make_backend("torch64", device="cpu")
    result, rows = solve_run(option, backend, 6, pg=pg, **model_options)
    if result.status != "cap_reached" or result.accepted != 6:
        raise RuntimeError(f"{path} {key}: {result.status}, {result.accepted} accepted (6 cap_reached expected)")
    return trajectory_digest(rows, result)


def all_digests(path, skip_solve=False):
    """The seven digests of a path (make_golden.py --compute): model small / box12, torch small / box12 (cpu device),
    solve numpy_a / numpy_b / torch64_a (skipped with skip_solve: they call the CPU certificate)."""
    model_options, _ = path_options(path)
    digests = dict(model={}, torch={}, solve={})
    for name, build in (("small", small_case), ("box12", box12_case)):
        state, u = build(**model_options)
        digests["model"][name] = model_digest(state, u)
    for name in ("small", "box12"):
        state, u, options = torch_case(name, path)
        digests["torch"][name] = torch_digest(state, u, **options)
    if not skip_solve:
        for key in ("numpy_a", "numpy_b", "torch64_a"):
            digests["solve"][key] = solve_key(key, path)
    return digests
