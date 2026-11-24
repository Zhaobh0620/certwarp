"""CVG run: the coarse levels of one pair with bayesreg.solve.solve_multilevel, CPU FP64 numpy, a raised step cap.

    CVG_CODE=<snapshot> python -B experiments/cvg/l0_run.py --cache <cache.npz> --pair <stage>/<pair> --out DIR
        --channel-source structure|tissue|none [--sigma em|intensity] [--nu NU] [--max-accepted N]
        [--wall-seconds S] [--max-levels 1|2] [--save-at 1,500] [--logvol-points ...] [--jd-cell-estimate ...]
        [--metric-jd-scaling affine_v1|off] [--cpu-threads 4]

Everything is bayesreg's own: solve_multilevel (option (a), rounds (1,1,1,1), the C13 threshold T_B = 1e-8 + 1e-6 G0,
level-start M step for sigma), run_pg (PG + BB2 + certificate-gated stable Armijo, C09.1-3a JD scaling), the CPU
certificate (C07-2 configuration).  The diagnostic changes are only: the accepted-step cap C13-4 (500 -> --max-accepted,
default 20000; "上限收尾" when hit), an optional wall-clock budget (WallBudget, "预算收尾"), max_levels (1 = L0 only;
the run ends stopped_before_native, never a native state), the sigma policy (--sigma intensity is the diagnostic arm
S, 诊断臂，不是方法配置) and nu (--nu: eta_tmp with that nu, alpha and tau unchanged).  Inputs come from prep.py's
cache (coarse levels of run_pair's pyramid).  The log is filtered (common.RunLog); accepted states at --save-at step
counts and the final commit of the last level are saved under states/.

Outputs: config.json, log.jsonl, states/, trace.csv (one row per accepted step: level, k, tau, F_B, GF_B, step_ratio,
min_jd_lower_bound, iteration, candidates of that iteration), result.json (status, rounds with F_B / GF_B traces, the
metric record, counters, sigma, timing, candidates and rejection counts).
"""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import numpy as np  # noqa: E402

from bayesreg.io import load_pair  # noqa: E402
from bayesreg.cli import run_pair as rp  # noqa: E402
from bayesreg.model import LOGVOL_GAUSS27_CLOSED27, LOGVOL_POINTS  # noqa: E402
from bayesreg.solve import Caps, PGConfig, SolverConfig, solve_multilevel  # noqa: E402
from bayesreg.solve.jd_scaling import (AFFINE_V1, CELL_ESTIMATES, CERTIFICATE, CERTIFICATE_BERNSTEIN, MODES,  # noqa: E402
                                       JDScalingConfig, j53_violations_total)
from bayesreg.solve.stopping import MAX_OUTER, WallBudget  # noqa: E402


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--pair", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--channel-source", required=True, choices=common.CHANNEL_SOURCES)
    p.add_argument("--sigma", default="em", choices=common.SIGMA_MODES)
    p.add_argument("--nu", type=float, default=None)
    p.add_argument("--max-accepted", type=int, default=20000)
    p.add_argument("--wall-seconds", type=float, default=None)
    p.add_argument("--max-levels", type=int, default=1)
    p.add_argument("--save-at", default="1,50,500,2000")
    p.add_argument("--logvol-points", choices=LOGVOL_POINTS, default=LOGVOL_GAUSS27_CLOSED27)
    p.add_argument("--jd-cell-estimate", choices=CELL_ESTIMATES, default=CERTIFICATE)
    p.add_argument("--metric-jd-scaling", choices=MODES, default=AFFINE_V1)
    p.add_argument("--cpu-threads", type=int, default=4)
    return p.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    cache, meta = common.load_cache(args.cache)
    pair = load_pair(rp.resolve_pair(common.DATA_ROOT, args.pair), allowlist=rp.DEFAULT_ALLOWLIST)
    if meta["pair"] != args.pair:
        raise RuntimeError("cache of another pair")
    grids = common.grids_of(pair, meta)
    inputs, kept = common.level_inputs_from_cache(cache, meta, grids, args.channel_source)
    if args.max_levels > len(kept):
        raise RuntimeError(f"max_levels {args.max_levels} > cached levels {kept}")
    eta = common.eta_of(meta, args.nu)
    sigma = common.sigma_policy(args.sigma)
    closed = args.logvol_points == LOGVOL_GAUSS27_CLOSED27
    config = SolverConfig(outer="a", rounds=(1, 1, 1, 1), caps=Caps(max_accepted=args.max_accepted, max_outer=MAX_OUTER),
                          pg=PGConfig(metric_jd_scaling=args.metric_jd_scaling,
                                      jd=JDScalingConfig(cell_estimate=args.jd_cell_estimate)),
                          max_levels=args.max_levels, cpu_threads=args.cpu_threads, logvol_points=args.logvol_points,
                          **(dict(logvol_closed27_share=1., closed27_weights="simpson") if closed else {}))
    backend = common.CaptureBackend()
    budget = None if args.wall_seconds is None else WallBudget(args.wall_seconds)
    save_at = [int(x) for x in args.save_at.split(",") if x]
    common.write_json(out/"config.json", dict(
        args=vars(args), solver=config.record(), eta=eta.record(), sigma=sigma.record(), backend=backend.record(),
        code=common.code_record(), cache=dict(path=args.cache, sha256=common.sha256_file(args.cache)),
        model_channels=int(inputs[0].fixed.shape[0]), levels_run=list(range(args.max_levels)),
        note=("CVG diagnostic run (not a method result): " +
              ("sigma arm S, " + common.DIAGNOSTIC_ONLY + "; " if args.sigma == "intensity" else "") +
              f"C13-4 cap {args.max_accepted}, wall {args.wall_seconds}, CPU FP64 numpy")))
    log = common.RunLog(out, backend, save_at)
    tick = time.perf_counter()
    try:
        result = solve_multilevel(backend, inputs, eta=eta, sigma=sigma, config=config, log=log, budget=budget)
    finally:
        log.close()
    seconds = time.perf_counter()-tick
    if result.commit is not None:
        u = np.asarray(result.commit.u)
        np.savez(out/"states"/f"u_final_L{result.level}.npz", u=u, state_hash=result.commit.cert.state_hash,
                 level=result.level, min_jd_lower_bound=result.commit.cert.min_jd_lower_bound)
    with open(out/"trace.csv", "w", encoding="utf-8") as handle:
        handle.write("level,k,tau,F_B,GF_B,step_ratio,min_jd_lower_bound,iteration,candidates,seconds\n")
        for row in log.accepts:
            handle.write(",".join("" if v is None else repr(v) for v in row)+"\n")
    rounds = []
    for row in result.rounds:
        rounds.append({k: v for k, v in row.items()})
    import resource
    common.write_json(out/"result.json", dict(
        status=result.status, wording=result.wording, reason=result.reason, level=result.level, rounds=rounds,
        transfers=result.transfers, timing=dict(result.timing, solve_seconds=seconds,
                                                total_seconds=time.time()-started),
        counters=result.counters, sigma=result.sigma, budget=result.budget, candidates=log.candidates,
        rejections=log.rejected, saved_states=log.saved, certificate_seconds=backend.certificate_seconds,
        certificate_calls=backend.certificate_calls,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        # JDTI: C09.1-3a certificate_bernstein check (b), the run total (this option only)
        **(dict(bernstein_j53_violations_total=j53_violations_total(rounds))
           if args.metric_jd_scaling == AFFINE_V1 and args.jd_cell_estimate == CERTIFICATE_BERNSTEIN else {})))
    for row in result.rounds:
        print(f"L{row['level']} r{row['round']}: {row['status']} accepted {row.get('accepted')} G0 {row.get('G0')} "
              f"T_B {row.get('T_B')} GF_B {row.get('GF_B')} ratio {row.get('GF_B_over_T_B')} "
              f"{row.get('seconds', 0):.1f} s")
    print(f"run {result.status}: {result.reason}; {seconds:.1f} s")


if __name__ == "__main__":
    main()
