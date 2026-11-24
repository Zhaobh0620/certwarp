"""JDC demo (work package JDC, 2026-10-30): the three C09.1-3a jd_cell_estimate options on the same synthetic problems.

Synthetic data only (no data/ access, no GPU, numpy backend, CPU FP64 certificate).  Start state: the edge-dip field of
tests/test_bayesreg/solve/_edge_dip.py (one cell K whose Q1 J minimum, about .0019, lies inside a z edge; corners
>= .15, Gauss points >= .18) on an n^3 unit grid, K at the centre.  Push w: at K's 8 vertices the unit direction that
lowers J at the dip point fastest (-dJ/du, normalized), elsewhere a smooth bump of amplitude .25 mm.

  quadratic  E(theta) = 1/2 |theta - theta*|^2, theta* = theta0 + w; the identity as the base metric; run_pg with the
             real certificate; reduction = E_b - E_t (a test double of the C08 comparison, no guard).
  fb         F_B of bayesreg.model (one intensity channel, sigma^2 = 1e-2, alpha = nu = 1e-3, tau^2 = 1) with
             fixed = moving o (x + u*), u* = u0 + w (u* folds K: the data term pulls through the dip); solve_direct
             (option (a), the C09.1 metric, JD scaling affine_v1) from the certified u0, the C08 stable comparison.
Per option: status, accepted steps, candidates and rejection reasons, S~ refreshes, the energy after every accepted
step, the certificate lower bound of the last accepted state, the S~ computation records, wall time.

    python experiments/jdc/demo_cell_estimate.py --out OUT.json [--steps 30] [--n 8]

A single synthetic observation of the path; not a convergence statement and not a timing benchmark.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jd_curl.three_d.boundary import restrict_free  # noqa: E402

from bayesreg.model import ModelState, build_samplers  # noqa: E402
from bayesreg.solve import (Caps, Commit, JDScaledMetric, JDScalingConfig, NumpyBackend, PGConfig, g0_of,  # noqa: E402
                            run_pg, solve_direct, threshold)
from bayesreg.solve import jd_scaling as jds  # noqa: E402
from tests.test_bayesreg.solve import _edge_dip as dip, _synthetic as syn  # noqa: E402

# The three options of JDC, by name (JDTI review minor 5): jds.CELL_ESTIMATES also holds certificate_bernstein since
# JDTI; this demo and its schema v1 stay those of JDC.  certificate_bernstein on the same problems (the "mid" / "off"
# quadratics, the F_B edge dip) is gate 6 of tests/test_bayesreg/solve/test_jd_cell_estimate_bernstein.py.
ESTIMATES = (jds.GAUSS_CORNERS, jds.CLOSED27, jds.CERTIFICATE)


def git_record():
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return dict(head=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")))


def problem(n):
    grid = syn.cube_grid(n)
    c = (n-2)//2
    cell = (c, c, c)
    u0 = dip.edge_dip_field(grid, cell)
    J, dJ = dip.cell_point_jacobian(u0, grid, cell, dip.EDGE_DIP_POINT)
    t = [(a-a[0])/(a[-1]-a[0]) for a in grid.axes]
    tz, ty, tx = np.meshgrid(t[2], t[1], t[0], indexing="ij")
    bump = np.sin(np.pi*tx)*np.sin(np.pi*ty)*np.sin(np.pi*tz)
    w = np.stack([.25*bump*np.cos(2*tx+k) for k in range(3)], axis=-1)
    from jd_curl.three_d.cell_q1 import VERTICES
    push = -dJ/np.linalg.norm(dJ)
    for v, (x, y, z) in enumerate(VERTICES):
        w[cell[0]+z, cell[1]+y, cell[2]+x] = push[v]
    for face in (0, -1):
        w[face], w[:, face], w[:, :, face] = 0., 0., 0.
    return grid, cell, u0, w, J


@dataclass
class Point:
    u: np.ndarray
    energy: float


@dataclass
class Reduction:
    ared: float
    guard: float
    certified: float
    reason: str


class Quadratic:
    name = "quadratic"

    def __init__(self, grid, theta_star):
        self.grid, self.theta_star = grid, theta_star
        self.state = type("Holder", (), dict(grid=grid, omega0=1.))()

    def evaluate(self, u):
        d = restrict_free(u, self.grid)-self.theta_star
        return Point(u, .5*float(d@d))

    def gradient(self, point):
        return restrict_free(point.u, self.grid)-self.theta_star

    def reduction(self, base, trial):
        a = base.energy-trial.energy
        return Reduction(a, 0., a, None if a > 0 else "NO_STRICT_DECREASE")


class Identity:
    a, shift = 0., 0.

    @staticmethod
    def solve(g):
        return g.copy()

    @staticmethod
    def matvec(v):
        return v.copy()


def summary(rows, status, accepted, energies, computations, seconds):
    cands = [r for r in rows if r.get("kind") == "candidate"]
    acc = [r for r in rows if r.get("kind") == "accept"]
    return dict(status=status, accepted=accepted, candidates=len(cands),
                reasons=dict(Counter(str(r["reason"]) for r in cands)), energies=[float(e) for e in energies],
                taus=[r["tau"] for r in acc], last_min_jd_lower_bound=acc[-1]["min_jd_lower_bound"] if acc else None,
                refreshes=sum(1 for c in computations if c.get("at") == "refresh"),
                computations=computations, seconds=seconds)


def run_quadratic(grid, u0, w, steps, backend):
    cert0 = backend.certify(u0, grid)
    theta0 = restrict_free(u0, grid)
    objective = Quadratic(grid, theta0+restrict_free(w, grid))
    out = {}
    for estimate in ESTIMATES:
        jds.clear_cell_bounds_cache()
        tick = time.perf_counter()
        metric = JDScaledMetric(Identity(), grid, u0, config=JDScalingConfig(cell_estimate=estimate), cert=cert0,
                                cert_config=backend.cert_config, at="freeze")
        rows = []
        base = objective.evaluate(u0)
        result = run_pg(objective, base, theta=theta0, cert=cert0, metric=metric, certify=backend.certify, target=0.,
                        max_steps=steps, log=rows.append)
        out[estimate] = summary(rows, result.status, result.accepted, result.energies, metric.computations,
                                time.perf_counter()-tick)
    return out


def run_fb(grid, u0, w, steps, backend):
    moving = syn.moving_image(grid.identity())
    u_star = u0+w
    fixed = syn.warp_with_spline(moving, grid, u_star)
    # DEF (2026-10-31): explicit "gauss27", the model default when this demo was written (JDC); the ModelState
    # default is gauss27+closed27 since DEF -- the explicit old value keeps the demo's problem unchanged
    state = ModelState(grid=grid, omega0=grid.omega, fixed=fixed[None], samplers=build_samplers(moving[None], grid),
                       mask=syn.brain_box(grid, 1), sigma2=np.array([1e-2]), alpha=1e-3, nu=1e-3, tau2=np.full(4, 1.),
                       tissue=syn.uniform_tissue(grid), logvol_points="gauss27")
    cert0 = backend.certify(u0, grid)
    start = Commit(u0, restrict_free(u0, grid), cert0)
    target = threshold(g0_of(backend, state))
    out = {}
    for estimate in ESTIMATES:
        jds.clear_cell_bounds_cache()
        tick = time.perf_counter()
        rows = []
        result = solve_direct(backend, state, start, target=target, caps=Caps(max_accepted=steps), log=rows.append,
                              pg=PGConfig(jd=JDScalingConfig(cell_estimate=estimate)))
        rec = result.metric["jd_scaling"]
        out[estimate] = summary(rows, result.status, result.accepted, result.energies, rec["computations"],
                                time.perf_counter()-tick)
        out[estimate]["GF_B_end"] = result.gradient_norm
        out[estimate]["T_B"] = target
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--problems", default="quadratic,fb")
    args = p.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists (nothing is overwritten)")
    grid, cell, u0, w, J = problem(args.n)
    backend = NumpyBackend()
    m = jds.cell_jd_estimate(u0, grid)
    m27, _ = jds.closed27_cell_estimate(u0, grid)
    record = dict(schema="jdc.demo_cell_estimate.v1", git=git_record(), python=sys.version, numpy=np.__version__,
                  platform=platform.platform(), argv=sys.argv, n=args.n, steps=args.steps, cell=list(cell),
                  start=dict(J_at_dip_point=J, m_K_default=float(m[cell]), m_K_closed27=float(m27[cell]),
                             certificate=backend.certify(u0, grid).record()))
    for name in [s.strip() for s in args.problems.split(",") if s.strip()]:
        tick = time.perf_counter()
        record[name] = (run_quadratic if name == "quadratic" else run_fb)(grid, u0, w, args.steps, backend)
        for estimate, s in record[name].items():
            print(f"{name} {estimate}: {s['status']} accepted {s['accepted']} candidates {s['candidates']} "
                  f"E {s['energies'][0]:.6g} -> {s['energies'][-1]:.6g} refreshes {s['refreshes']} "
                  f"last LB {s['last_min_jd_lower_bound']} reasons {s['reasons']} ({s['seconds']:.2f} s)", flush=True)
        record[name+"_seconds"] = time.perf_counter()-tick
    out.write_text(json.dumps(record, indent=1, default=str)+"\n")


if __name__ == "__main__":
    main()
