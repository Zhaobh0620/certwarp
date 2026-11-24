"""CVG counterfactual probe (反事实探针, NOT a solver and not an implementation of C09.3): what a damped Gauss-Newton
direction of the C09.3 form would do at a given state, compared with the PG run that produced the state.

    CVG_CODE=<snapshot> python -B experiments/cvg/gn_probe.py --run <l0_run dir> --state <npz> --label NAME --out DIR
        [--steps 10] [--level 0]

The model state and the metric M are rebuilt as in stiffness.py (the run's sigma, eta, frozen c_dat, b_P).  From u, up
to --steps probe steps; each: g = grad F_B(u); for gamma = gamma0 = 1e-3 (C09.3-12) and then 4x on every failure (at
most 12 directions): p from PCG on (H + gamma (omega_ell/omega0) I) p = -g with preconditioner M^-1 (C09.1 with
gamma in a), rtol 1e-3, maxiter 200, x0 = 0 (scipy cg; H the GN Hessian at u); pred = -g.p - 1/2 p.H p; candidate
u + p -> CPU certificate (C07-2 configuration) -> F_B -> stable comparison (C08 CPU reference) -> accepted when
certified > 0 and certified/pred >= 1e-4; gamma halves after a ratio > .75.  Recorded per direction: PCG iterations,
true relative residual, pred, ared, ratio, certificate status and min J lower bound, GF_B after.  The probe never
feeds anything back into a run; it only reports how far such steps move GF_B from the state, next to the PG trace of
the run from the same state (trace.csv).
"""
import argparse
import csv
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import numpy as np  # noqa: E402
from scipy.sparse.linalg import LinearOperator, cg  # noqa: E402

from jd_curl.three_d.boundary import inject_free, restrict_free  # noqa: E402
from bayesreg.model import build_metric, evaluate, gradient, hvp, stable_reduction  # noqa: E402
from bayesreg.solve.stopping import gf_b, state_hash  # noqa: E402

import stiffness  # noqa: E402  (setup: the run's level state and metric)

GAMMA0, GAMMA_MIN, GAMMA_MAX = 1e-3, 1e-12, 1e12


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--steps", type=int, default=10)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    run = Path(a.run)
    config, result, row, state, metric = stiffness.setup(run, a.level)
    grid = state.grid
    n = 3*grid.n_free_scalar
    scale = grid.omega/state.omega0
    u = np.load(a.state)["u"]
    k0 = int(np.load(a.state)["accepted_steps"]) if "accepted_steps" in np.load(a.state) else None
    backend = common.CaptureBackend()
    base = evaluate(state, u)
    cert = backend.certify(u, grid)
    if not cert.passed:
        raise RuntimeError("the state is not certified")
    gamma = GAMMA0
    steps, rows = 0, []
    G = gf_b(gradient(state, base), grid.omega, state.omega0)
    doc = dict(label=a.label, run=str(run), state=a.state, state_hash=state_hash(u), start_k=k0, GF_B_start=G,
               T_B=row["T_B"], F_B_start=base.energy, note="反事实探针 (counterfactual probe), not a solver",
               code=common.code_record())
    tick0 = time.perf_counter()
    while steps < a.steps:
        g = gradient(state, base)
        G = gf_b(g, grid.omega, state.omega0)
        accepted = False
        for trial in range(12):
            tick = time.perf_counter()
            M = build_metric(state, c_dat=row["metric"]["c_dat"], b_P=row["metric"]["b_P"], gamma=gamma)
            A = LinearOperator((n, n), matvec=lambda v: hvp(state, base, v)+gamma*scale*v, dtype=np.float64)
            P = LinearOperator((n, n), matvec=M.solve, dtype=np.float64)
            it = [0]
            x, info = cg(A, -g, rtol=1e-3, atol=0., maxiter=200, M=P, callback=lambda _: it.__setitem__(0, it[0]+1))
            Hx = hvp(state, base, x)
            resid = float(np.linalg.norm(Hx+gamma*scale*x+g)/np.linalg.norm(g))
            pred = float(-(g@x)-.5*(x@Hx))
            entry = dict(step=steps, trial=trial, gamma=gamma, pcg_iterations=it[0], pcg_info=int(info),
                         true_relative_residual=resid, pred=pred, max_abs_p=float(np.max(np.abs(x))),
                         GF_B_base=G, cert=None, ared=None, certified=None, ratio=None, accepted=False)
            ok = resid <= 1e-3 and pred > 0
            if ok:
                cand = inject_free(restrict_free(u, grid)+x, grid)
                oc = backend.certify(cand, grid)
                entry["cert"] = dict(passed=oc.passed, reason=oc.cert_reason, min_jd_lower_bound=oc.min_jd_lower_bound)
                if oc.passed:
                    trial_point = evaluate(state, cand)
                    if hasattr(trial_point, "energy"):
                        red = stable_reduction(state, base, trial_point)
                        ratio = red.certified/pred
                        entry.update(ared=red.ared, certified=red.certified, ratio=ratio, F_B=trial_point.energy)
                        if red.certified > 0 and ratio >= 1e-4:
                            accepted = True
                            entry["accepted"] = True
                            entry["GF_B_after"] = gf_b(gradient(state, trial_point), grid.omega, state.omega0)
                            u, base, cert = cand, trial_point, oc
                    else:
                        entry["invalid"] = trial_point.reason
            entry["seconds"] = time.perf_counter()-tick
            rows.append(entry)
            print({k: entry[k] for k in ("step", "trial", "gamma", "pcg_iterations", "pred", "ared", "ratio",
                                         "accepted")}, entry.get("GF_B_after"), flush=True)
            if accepted:
                ratio = entry["ratio"]
                if ratio > .75:
                    gamma = max(GAMMA_MIN, gamma/2)
                elif ratio < .25:
                    gamma = GAMMA_MAX if gamma >= GAMMA_MAX/4 else 4*gamma
                break
            gamma = GAMMA_MAX if gamma >= GAMMA_MAX/4 else 4*gamma
        if not accepted:
            doc["ended"] = "12 directions rejected"
            break
        steps += 1
        doc["rows"] = rows
        common.write_json(out/"gn_probe.json", doc)
    doc.update(rows=rows, steps_accepted=steps, GF_B_end=gf_b(gradient(state, base), grid.omega, state.omega0),
               F_B_end=base.energy, seconds=time.perf_counter()-tick0)
    # the PG run from the same state: GF_B after the same number of accepted steps, and after 10x / 100x as many
    trace = list(csv.DictReader(open(run/"trace.csv")))
    if k0 is not None:
        pg = {}
        for extra in (steps, 10*steps, 100*steps, 1000*steps):
            hit = [r for r in trace if r["level"] == str(a.level) and int(r["k"]) == k0+extra]
            pg[str(extra)] = float(hit[0]["GF_B"]) if hit else None
        doc["pg_same_run_GF_B_after_extra_steps"] = pg
    common.write_json(out/"gn_probe.json", doc)
    print("done", doc["steps_accepted"], doc["GF_B_start"], "->", doc["GF_B_end"], doc.get("pg_same_run_GF_B_after_extra_steps"))


if __name__ == "__main__":
    main()
