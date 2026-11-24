"""Wall clock of the CPU FP64 reference, serial against thread-parallel (work package CPUPAR; bench rules).

Question: how long do evaluate, gradient, hvp, stable_reduction, the proximal value and one CPU fallback of the
GPU-path comparison (C08-8(d): both states re-evaluated, then C08-1..7) take at a given grid size with
cpu_threads = 1 and cpu_threads = N, on one fixed synthetic state?

    python experiments/cpupar/bench.py --out DIR [--shape 160,192,224] [--channels 4] [--threads 4] [--check]

Input: a synthetic ModelState (no data/): unit spacing, `--shape` nodes (nz, ny, nx), C channels (smooth
textures), an ellipsoid mask (semi-axes .4 of the extent: ~27 % of the nodes, about a brain at 1 mm), a smooth
tissue simplex, Gaussian noise; fixed smooth U0 fields u (base), w1, w2 (trials, about 1 voxel) and a random
direction v.  Every timing is ONE observation of a fixed state (no repetition): it may not be read as a stable
number, and a per-function time is not a whole-registration speed-up.

Order: per function, the serial run and then the parallel run (alternating, same process, same inputs).  With
--check every parallel result is compared bitwise with the serial one (float64 bytes, longdouble ==).
Fallback rows use the StableComparator CPU path (points without a device evaluation, C08-8(e)): "cold" = a new
comparator (base and trial evaluated, reduction), "warm" = the next trial against the same base (base from the
base slot: trial evaluated, reduction).  Breakdown rows: channel_point (spline sampling of the C channels) and
stiffness_apply (K_Q u, the diffusion term of gradient / hvp).

Writes DIR/summary.json (code SHA, configuration, environment, seconds, peak RSS) and prints a table.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jd_curl.three_d.boundary import inject_free  # noqa: E402
from jd_curl.three_d.grid import make_grid_3d  # noqa: E402

from bayesreg import model as M  # noqa: E402
from bayesreg.geom import stiffness_apply  # noqa: E402
from bayesreg.model import ModelState, build_samplers  # noqa: E402
from bayesreg.solve import proximal as prox  # noqa: E402

LD = np.longdouble


def synthetic_state(shape, channels, tile_size=8192):
    nz, ny, nx = shape
    g = make_grid_3d(nx=nx, ny=ny, nz=nz, hx=1., hy=1., hz=1., unit="mm")
    X = g.identity()
    x, y, z = np.moveaxis(X, -1, 0)

    def images(shift):
        a, b, c = x+shift[0], y+shift[1], z+shift[2]
        out = [.5+.3*np.sin(.31*a+.2*b)*np.cos(.27*c)+.2*np.cos(.19*a-.23*b+.29*c)]
        for k in range(channels-1):
            out.append(.5+.4*np.sin(.11*(k+1)*a-.07*b+.13*c+k))
        return np.stack(out)

    moving = images((0., 0., 0.))
    fixed = images((.3, -.2, .25))
    del X
    centre = [axis.mean() for axis in g.axes]
    extent = [axis[-1]-axis[0] for axis in g.axes]
    mask = sum(((c-m)/(.4*e))**2 for c, m, e in zip((x, y, z), centre, extent)) <= 1.
    logits = np.stack((np.cos(.15*x)+np.sin(.12*y), np.sin(.2*z-.1*x), np.cos(.18*y+.05*z), .5*np.sin(.1*(x+y+z))))
    del x, y, z
    e = np.exp(logits-logits.max(axis=0))
    del logits
    tissue = e/e.sum(axis=0)
    del e
    samplers = build_samplers(moving, g)
    del moving
    # DEF (2026-10-31): explicit "gauss27", the model default of the CPUPAR measurements; the ModelState default is
    # gauss27+closed27 since DEF -- the explicit old value keeps the benchmark's problem unchanged
    state = ModelState(grid=g, omega0=g.omega, fixed=fixed, samplers=samplers, mask=mask,
                       sigma2=np.full(channels, .05), alpha=.13, nu=.07, tau2=np.array([1., .25, .36, .16]),
                       tissue=tissue, tile_size=tile_size, cpu_threads=1, logvol_points="gauss27")
    return state


def smooth(grid, amplitude, seed):
    rng = np.random.default_rng(seed)
    s = [(a-a[0])/(a[-1]-a[0]) for a in grid.axes]
    bump = np.sin(np.pi*s[2])[:, None, None]*np.sin(np.pi*s[1])[None, :, None]*np.sin(np.pi*s[0])[None, None, :]
    theta = []
    for r in range(3):
        k = rng.uniform(.05, .2, 3)
        phase = rng.uniform(0, 2*np.pi)
        field = amplitude*bump*np.cos(k[0]*np.arange(grid.shape[2])[None, None, :]
                                       + k[1]*np.arange(grid.shape[1])[None, :, None]
                                       + k[2]*np.arange(grid.shape[0])[:, None, None]+phase)
        theta.append(field[1:-1, 1:-1, 1:-1].ravel())
    return inject_free(np.concatenate(theta), grid)


def same(a, b):
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    return a == b


def sha():
    try:
        head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "bayesreg"], text=True)
        return head+("+dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def peak_rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shape", default="160,192,224", help="nodes nz,ny,nx")
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check", action="store_true", help="compare every parallel result bitwise with the serial")
    parser.add_argument("--skip", default="", help="comma list of rows to skip")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out}: exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    shape = tuple(int(s) for s in args.shape.split(","))
    skip = set(filter(None, args.skip.split(",")))
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tick = time.perf_counter()
    base_state = synthetic_state(shape, args.channels)
    setup_seconds = time.perf_counter()-tick
    g = base_state.grid
    u, w1, w2 = smooth(g, 1., 1), smooth(g, .97, 2), smooth(g, 1.02, 3)
    v = np.random.default_rng(4).normal(size=3*g.n_free_scalar)
    states = {1: base_state, args.threads: base_state.replace(cpu_threads=args.threads)}
    params = prox.ProximalParams(lam=prox.default_lambda(g))
    rows, results, mismatches = [], {}, []
    print(f"grid {shape} nodes, {g.n_cells} cells, {base_state.node_index.size} mask nodes, C = {args.channels}, "
          f"tile {base_state.tile_size}; setup {setup_seconds:.1f} s; peak RSS {peak_rss_gib():.2f} GiB", flush=True)

    def run(name, function):
        if name in skip:
            return
        row = dict(name=name)
        for threads in (1, args.threads):
            tick = time.perf_counter()
            value = function(states[threads])
            row[f"seconds_t{threads}"] = time.perf_counter()-tick
            if args.check:
                results[(name, threads)] = value
        row["speedup"] = row["seconds_t1"]/row[f"seconds_t{args.threads}"]
        row["peak_rss_gib"] = peak_rss_gib()
        if args.check:
            a, b = results.pop((name, 1)), results.pop((name, args.threads))
            ok = check(a, b)
            row["bitwise_equal"] = ok
            if not ok:
                mismatches.append(name)
        rows.append(row)
        print(f"{name:28s} t1 {row['seconds_t1']:9.2f} s   t{args.threads} {row[f'seconds_t{args.threads}']:9.2f} s   "
              f"x{row['speedup']:.2f}   rss {row['peak_rss_gib']:.2f} GiB"
              + ("" if not args.check else f"   bitwise {row['bitwise_equal']}"), flush=True)

    def check(a, b):
        if isinstance(a, M.Evaluation):
            return (same(a.total, b.total) and a.energies == b.energies
                    and all(same(getattr(a.channel, k), getattr(b.channel, k)) for k in ("rho", "grad", "r")))
        if isinstance(a, M.Reduction):
            return same(a.delta, b.delta) and a.ared == b.ared and a.blocks == b.blocks
        if isinstance(a, prox.ProximalValue):
            return same(a.total, b.total) and all(same(a.parts[k], b.parts[k]) for k in a.parts)
        if isinstance(a, tuple):
            return all(check(x, y) for x, y in zip(a, b))
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if hasattr(a, "ared"):
            return a.ared == b.ared and same(a.delta, b.delta) and a.guard == b.guard
        return same(a, b)

    evaluations = {}

    def evaluate(st):
        ev = M.evaluate(st, u)
        evaluations[st.cpu_threads] = ev
        return ev

    run("channel_point", lambda st: M.channel_point(st, u).r)
    run("evaluate", evaluate)
    run("stiffness_apply", lambda st: stiffness_apply(u, g))
    run("gradient", lambda st: M.gradient(st, evaluations[st.cpu_threads]))
    run("hvp", lambda st: M.hvp(st, evaluations[st.cpu_threads], v))
    trials = {}

    def reduction(st):
        trials[st.cpu_threads] = M.evaluate(st, w1)
        tick = time.perf_counter()
        red = M.stable_reduction(st, evaluations[st.cpu_threads], trials[st.cpu_threads])
        rows_extra.setdefault("stable_reduction_only", {})[st.cpu_threads] = time.perf_counter()-tick
        return red

    rows_extra = {}
    run("evaluate+stable_reduction", reduction)
    evaluations.clear()
    trials.clear()
    run("prox_evaluate", lambda st: prox.evaluate(st, w1, u, params))
    if "fallback" not in skip:
        from bayesreg.backends.torch64 import stable as S
        row = dict(name="fallback cold / warm")
        for threads in (1, args.threads):
            st = states[threads]
            comparator = S.StableComparator(cache_size=2)
            mid = S.model_identity(st)

            def host(field):
                field = np.array(field, dtype=np.float64, order="C")
                field.flags.writeable = False
                return S.HostEvaluation(st, field, LD(0), 0., {}, anomaly="CPU timing (no device)")

            base = host(u)
            _ = base.state_hash
            t1, t2 = host(w1), host(w2)
            _ = t1.state_hash, t2.state_hash
            tick = time.perf_counter()
            cold = comparator.compare(st, None, base, t1, model_id=mid, context="pg")
            row[f"cold_seconds_t{threads}"] = time.perf_counter()-tick
            tick = time.perf_counter()
            warm = comparator.compare(st, None, base, t2, model_id=mid, context="pg")
            row[f"warm_seconds_t{threads}"] = time.perf_counter()-tick
            row[f"cpu_evaluations_t{threads}"] = comparator.stats.cpu_evaluations
            if args.check:
                results[("fallback", threads)] = (cold, warm)
            del comparator, cold, warm, base, t1, t2
        row["peak_rss_gib"] = peak_rss_gib()
        if args.check:
            ok = check(results.pop(("fallback", 1)), results.pop(("fallback", args.threads)))
            row["bitwise_equal"] = ok
            if not ok:
                mismatches.append("fallback")
        rows.append(row)
        print(f"fallback cold t1 {row['cold_seconds_t1']:9.2f} s t{args.threads} {row[f'cold_seconds_t{args.threads}']:9.2f} s;"
              f" warm t1 {row['warm_seconds_t1']:9.2f} s t{args.threads} {row[f'warm_seconds_t{args.threads}']:9.2f} s"
              + ("" if not args.check else f"   bitwise {row['bitwise_equal']}"), flush=True)
    summary = dict(question="CPU reference wall clock, cpu_threads 1 vs N, one fixed synthetic state (CPUPAR)",
                   started_utc=started, code_sha=sha(), shape=list(shape), n_cells=int(g.n_cells),
                   n_mask_nodes=int(base_state.node_index.size), channels=args.channels,
                   tile_size=base_state.tile_size, threads=args.threads, setup_seconds=setup_seconds, rows=rows,
                   stable_reduction_only=rows_extra.get("stable_reduction_only"), bitwise_mismatches=mismatches,
                   checked=args.check, observations="one per row and thread count (single observation)",
                   env=dict(python=platform.python_version(), numpy=np.__version__, cpu_count=os.cpu_count(),
                            nice=os.nice(0), load_avg=os.getloadavg(),
                            blas_threads={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                                                         "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}),
                   peak_rss_gib=peak_rss_gib(), finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (out/"summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(dict(stable_reduction_only=summary["stable_reduction_only"], mismatches=mismatches)))
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
