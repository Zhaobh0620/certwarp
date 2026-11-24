"""LMI small CPU demo (开发对 L0 诊断, not a method result): PG vs LM-PCG on level 0 of one development pair.

    python -B experiments/lmi/l0_demo.py --inner pg|lm --out DIR [--wall-seconds 900] [--cpu-threads 4]
        [--cache /home/rmcfz/seq_vp_tag_results/cvg_v1/cache/0438-0439.npz]

Input: the CVG prep cache only (read-only; coarse levels of run_pair's pyramid for stage_L2R_val_v1
OASIS_OAS1_0438_MR1 <- OASIS_OAS1_0439_MR1, a development pair); nothing under data/ is read.  The native grid is
rebuilt from its record (shape, spacing, origin of the pair's run provenance, e.g.
/home/rmcfz/seq_vp_tag_results/m2_smoke/cpu_L2_tissue_0438_0439/provenance.json native_grid) and every cached level's
grid id is checked against the cache before anything runs.

Configuration = CVG R1 (runs/20261031-cvg/result.md; experiments/cvg/l0_run.py): channel source structure, sigma by the
Gaussian M step, eta_tmp of the cache (nu_tmp), logvol_points gauss27+closed27 (beta_S 1, Simpson), option (a), rounds
(1, 1, 1, 1), max_levels 1 (L0 only; the run ends stopped_before_native), accepted-step cap 20000 (diagnostic, as R1), a
wall-clock budget (default 900 s, "预算收尾"), CPU FP64 numpy, cpu_threads 4.  --inner pg: the PG inner with the JD
scaling and jd_cell_estimate certificate (R1); --inner lm: the C09.3 LM-PCG inner (bayesreg.solve.lm; JD scaling not
used).  Everything else is bayesreg's own solve_multilevel.

Supplementary warm start (--start-state FILE.npz with u and state_hash, e.g. CVG R1's states/u_L0_k500.npz; added
after the main arms, see runs/20261101-lmi-l0/plan.md): the level-0 round starts from that state instead of u = 0.
Everything before the round is unchanged (sigma from the level-start M step at the seed u = 0, G0 and T_B of the round,
as in R1); the state is re-certified with the C07-2 configuration (it must pass; its state_hash must match the file) and
handed to solve_round in place of the seed commit (bayesreg.solve.multilevel.solve_round wrapped for this process
only).  PG then starts with an empty BB2 history and S~ frozen at the state; LM with gamma0.

Outputs (DIR): config.json, log.jsonl (accept, round, lm_start, transfer/budget rows, every LM candidate row, the
rejected PG candidate rows), trace.csv (per accepted step: k, seconds, F_B, GF_B, GF_B/T_B, min certified J lower bound,
tau or gamma, PCG iterations (LM), attempts in that iteration), result.json (status, the round row, counters, rejection
counts, certificate rejections, PCG statistics, peak RSS), states/u_final_L0.npz.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from jd_curl.three_d.grid import make_grid_3d  # noqa: E402
from jd_curl.validation import readonly_copy  # noqa: E402
from bayesreg.grids import four_level_grids  # noqa: E402
from bayesreg.model import LOGVOL_GAUSS27_CLOSED27  # noqa: E402
from bayesreg.solve import (Caps, Eta, GaussianMStep, LevelInput, NumpyBackend, PGConfig, SolverConfig,  # noqa: E402
                            solve_multilevel)
from bayesreg.solve.jd_scaling import CERTIFICATE, JDScalingConfig  # noqa: E402
from bayesreg.solve.stopping import MAX_OUTER, WallBudget  # noqa: E402

CACHE = "/home/rmcfz/seq_vp_tag_results/cvg_v1/cache/0438-0439.npz"
PAIR = "stage_L2R_val_v1/OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1"
# native grid record of the pair (provenance.json native_grid of /home/rmcfz/seq_vp_tag_results/m2_smoke/
# cpu_L2_tissue_0438_0439; grid_id 1ea40c55...); the cached levels' grid ids are checked below
NATIVE = dict(shape_zyx=(192, 224, 160), spacing_xyz=(1., 1., 1.), origin_xyz=(-79., -112., -95.),
              grid_id="1ea40c55a1b0904f524314f273f4fffd3265cbdd3686074e6d2cd8396aa7038a")
NOTE = "开发对 L0 诊断 (development pair, level 0 only, single run per arm; not a method result, not a timing claim)"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path, document):
    Path(path).write_text(json.dumps(document, indent=1, default=_default)+"\n", encoding="utf-8")


def native_grid():
    nz, ny, nx = NATIVE["shape_zyx"]
    hx, hy, hz = NATIVE["spacing_xyz"]
    x0, y0, z0 = NATIVE["origin_xyz"]
    grid = make_grid_3d(nx=nx, ny=ny, nz=nz, hx=hx, hy=hy, hz=hz, x0=x0, y0=y0, z0=z0, unit="mm")
    if grid.grid_id != NATIVE["grid_id"]:
        raise RuntimeError("native grid record does not reproduce its grid id")
    return grid


def level_inputs(cache, meta, grids):
    """LevelInput tuple of the cached levels for channel source structure (as experiments/cvg/common.py
    channel_arrays / level_inputs_from_cache: [F~] + sqrt(P~_S^k), k >= 1; tissue of the fixed image; placeholders
    carrying only the grid for the levels not cached)."""
    kept = sorted(int(k) for k in meta["grid_ids"])
    out = []
    for ell, grid in enumerate(grids):
        if ell in kept:
            if grid.grid_id != meta["grid_ids"][str(ell)]:
                raise RuntimeError(f"level {ell}: grid id differs from the cache")
            fixed, moving = cache[f"fixed_L{ell}"], cache[f"moving_L{ell}"]
            s_fixed = np.sqrt(cache[f"structure_fixed_L{ell}"][1:])
            s_moving = np.sqrt(cache[f"structure_moving_L{ell}"][1:])
            mask = cache[f"mask_L{ell}"].astype(bool)
            mask.flags.writeable = False
            out.append(LevelInput(ell, grid, readonly_copy(np.concatenate([fixed[None], s_fixed])),
                                  readonly_copy(np.concatenate([moving[None], s_moving])), mask,
                                  readonly_copy(cache[f"tissue_fixed_L{ell}"])))
        else:
            out.append(LevelInput(ell, grid, None, None, None, None))
    return tuple(out), kept


class Log:
    """The solver's log: a filtered log.jsonl and the accepted-step trace."""

    def __init__(self, out, inner):
        self.inner = inner
        self.handle = open(Path(out)/"log.jsonl", "w", encoding="utf-8")
        self.t0 = time.perf_counter()
        self.trace, self.attempts, self.rejected, self.candidates = [], 0, {}, 0
        self.cert_rejections = 0
        self.pcg = []                      # (pcg_iterations, true_residual) of every LM direction

    def __call__(self, row):
        kind = row.get("kind")
        if kind == "candidate":
            self.candidates += 1
            self.attempts += 1
            if row.get("pcg_iterations") is not None:
                self.pcg.append((row["pcg_iterations"], row.get("true_residual")))
            if not row.get("accepted"):
                reason = row.get("reason")
                self.rejected[reason] = self.rejected.get(reason, 0)+1
                if row.get("cert_status") not in (None, "certified_feasible"):
                    self.cert_rejections += 1
            if self.inner == "lm" or not row.get("accepted"):
                self._write(row)
            return
        if kind != "gf_recheck":
            self._write(row)
        if kind == "accept":
            self.trace.append(dict(k=row["accepted_steps"], seconds=round(time.perf_counter()-self.t0, 3),
                                   F_B=row["energy"], GF_B=row["gradient_norm"],
                                   min_jd_lower_bound=row["min_jd_lower_bound"], tau=row.get("tau"),
                                   gamma=row.get("gamma"), pcg_iterations=row.get("pcg_iterations"),
                                   attempts=self.attempts))
            self.attempts = 0
            if row["accepted_steps"] % 50 == 0:
                self.handle.flush()

    def _write(self, row):
        self.handle.write(json.dumps(row, default=_default)+"\n")

    def close(self):
        self.handle.close()


def warm_start(path, backend, grid, log):
    """Replace the level-0 round's start commit by the saved state (re-certified, hash checked); a log row records it."""
    from jd_curl.three_d.boundary import restrict_free
    from bayesreg.solve import multilevel
    from bayesreg.solve.outer import Commit
    from bayesreg.solve.stopping import state_hash
    saved = np.load(path, allow_pickle=False)
    u = readonly_copy(np.asarray(saved["u"], dtype=np.float64))
    if state_hash(u) != str(saved["state_hash"]):
        raise RuntimeError("start state: state_hash differs from the file")
    cert = backend.certify(u, grid)
    if not cert.passed or cert.state_hash != str(saved["state_hash"]):
        raise RuntimeError(f"start state: certificate {cert.status} ({cert.cert_reason})")
    commit = Commit(u, readonly_copy(restrict_free(u, grid)), cert)
    original = multilevel.solve_round

    def solve_round(backend_, state, start, **kwargs):
        if kwargs.get("context", {}).get("level") == 0 and kwargs["context"].get("round") == 0:
            log(dict(kind="warm_start", path=str(path), state_hash=cert.state_hash, replaced=start.cert.state_hash,
                     min_jd_lower_bound=cert.min_jd_lower_bound,
                     accepted_steps=int(saved["accepted_steps"]) if "accepted_steps" in saved.files else None))
            start = commit
        return original(backend_, state, start, **kwargs)
    multilevel.solve_round = solve_round


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--inner", choices=("pg", "lm"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default=CACHE)
    p.add_argument("--wall-seconds", type=float, default=900.)
    p.add_argument("--max-accepted", type=int, default=20000)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--start-state", default=None, help="warm start of the level-0 round (supplementary arms)")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out/"states").mkdir()
    started = time.time()
    cache = np.load(args.cache, allow_pickle=False)
    meta = json.loads(str(cache["meta_json"]))
    if meta["pair"] != PAIR:
        raise RuntimeError("cache of another pair")
    grids = four_level_grids(native_grid())
    inputs, kept = level_inputs(cache, meta, grids)
    record = meta["eta_tmp"]
    eta = Eta(record["alpha"], record["nu"], tuple(record["tau2"]), source=record["source"])
    sigma = GaussianMStep()
    common = dict(outer="a", rounds=(1, 1, 1, 1), caps=Caps(max_accepted=args.max_accepted, max_outer=MAX_OUTER),
                  max_levels=1, cpu_threads=args.cpu_threads, logvol_points=LOGVOL_GAUSS27_CLOSED27,
                  logvol_closed27_share=1., closed27_weights="simpson")
    if args.inner == "pg":
        config = SolverConfig(pg=PGConfig(jd=JDScalingConfig(cell_estimate=CERTIFICATE)), **common)
    else:
        config = SolverConfig(inner="lm", **common)
    backend = NumpyBackend()
    budget = WallBudget(args.wall_seconds)
    import bayesreg
    write_json(out/"config.json", dict(
        args=vars(args), note=NOTE, pair=PAIR, solver=config.record(), eta=eta.record(), sigma=sigma.record(),
        backend=backend.record(), cache=dict(path=args.cache, sha256=sha256_file(args.cache)), native_grid=NATIVE,
        levels_cached=kept, levels_run=[0], model_channels=int(inputs[0].fixed.shape[0]),
        code=dict(bayesreg=str(Path(bayesreg.__file__).resolve().parent), script_sha256=sha256_file(__file__)),
        reference="CVG R1 configuration (runs/20261031-cvg/result.md)"))
    log = Log(out, args.inner)
    if args.start_state is not None:
        warm_start(args.start_state, backend, grids[0], log)
    tick = time.perf_counter()
    try:
        result = solve_multilevel(backend, list(inputs), eta=eta, sigma=sigma, config=config, log=log, budget=budget)
    finally:
        log.close()
    seconds = time.perf_counter()-tick
    row = result.rounds[0] if result.rounds else {}
    target = row.get("T_B")
    with open(out/"trace.csv", "w", encoding="utf-8") as handle:
        handle.write("k,seconds,F_B,GF_B,GF_B_over_T_B,min_jd_lower_bound,tau,gamma,pcg_iterations,attempts\n")
        for t in log.trace:
            ratio = None if target is None else t["GF_B"]/target
            values = [t["k"], t["seconds"], t["F_B"], t["GF_B"], ratio, t["min_jd_lower_bound"], t["tau"], t["gamma"],
                      t["pcg_iterations"], t["attempts"]]
            handle.write(",".join("" if v is None else repr(v) for v in values)+"\n")
    if result.commit is not None:
        np.savez(out/"states"/"u_final_L0.npz", u=np.asarray(result.commit.u), state_hash=result.commit.cert.state_hash,
                 min_jd_lower_bound=result.commit.cert.min_jd_lower_bound)
    its = [i for i, _ in log.pcg]
    res = [r for _, r in log.pcg if r is not None]
    write_json(out/"result.json", dict(
        note=NOTE, status=result.status, wording=result.wording, reason=result.reason, round=row,
        transfers=result.transfers, counters=result.counters, sigma=result.sigma, budget=result.budget,
        candidates=log.candidates, rejections=log.rejected, certificate_rejections=log.cert_rejections,
        accepted=len(log.trace), pcg=dict(directions=len(its), iterations_total=int(sum(its)),
                                          iterations_median=float(np.median(its)) if its else None,
                                          iterations_max=max(its) if its else None,
                                          true_residual_max=max(res) if res else None),
        certificate_seconds=backend.certificate_seconds, certificate_calls=backend.certificate_calls,
        solve_seconds=seconds, total_seconds=time.time()-started,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    print(f"{args.inner}: {result.status} ({result.reason}); L0 {row.get('status')} accepted {row.get('accepted')} "
          f"GF_B {row.get('GF_B')} T_B {target} ratio {row.get('GF_B_over_T_B')}; {seconds:.1f} s")


if __name__ == "__main__":
    main()
