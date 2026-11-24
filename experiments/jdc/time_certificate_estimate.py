"""JDC1 timing tool (work package JDC1, 2026-10-30): the cost of one S~ computation (C09.1-3a) per jd_cell_estimate
option on one fixed, certified native-level state.  Written in JDC1, NOT run there: the lead schedules the run (D47:
the default is chosen after this measurement).

What is timed (wall clock, time.perf_counter), per option gauss27+corners | closed27 | certificate |
certificate_bernstein (JDTI review minor 4, for N1 of C09.1-3a; schema v2): one S~ computation
exactly as JDScaledMetric._compute does it at a freeze point or a refresh -- jd_scaling.scaling_factors (cell estimate,
node minimum, factors, record) + free_scaling.  `--warmup` untimed calls per option first (numba kernels, page cache),
then `--repeats` rounds, each round timing every option once in turn (interleaved, so that load drift hits all options
alike).  "certificate" is timed twice:
  certificate          uncached: the bounds cache cleared before each call (untimed) -- a refresh at a new accepted
                       state, the solver's usual case;
  certificate_cached   the same state again (cache hit) -- e.g. a round that restarts where a line search failed.
"certificate_bernstein" likewise (N1: uncached and cached, the same state and threads as certificate):
  certificate_bernstein         both cache levels cleared before each call (untimed): b_K by cell_pass, S and l_S by
                                the prescreen, J53 by the closed27 slab pass, check (b);
  certificate_bernstein_cached  both levels hit (b_K and (S, l_S)); J53 and check (b) are recomputed on every call, as
                                in the solver.  Recorded besides: |S| (n_cells_bernstein), n_cells_raised, the check-(b)
                                violations, bernstein_seconds per call.
Memory: with BAYESREG_MEMTRACE=<path> (bayesreg.memtrace; add BAYESREG_MEMTRACE_RESET_HWM=1 for the peak per call) every
timed call is followed by a mark "jdc_timing.call" (variant, seconds), untimed; without it the marks cost nothing.
Per variant: every call's seconds; median, first and third quartile, IQR, min, max; the inner `seconds` of the record
and (certificate) `cell_bounds_seconds`; the S~ record of the first timed call (min J_a, n_r_lt_1, n_nodes_r_lowered,
n_cells_lowered, ...), so that the cost sits next to the effect (how many more nodes the option freezes).
Untimed, once: certify(u) with the C07-2 configuration (PrescreenCertificateConfig, delta 0, max_depth 48, max_boxes
255, tile --tile-size): its CertOutcome feeds "certificate" (in the solver this is the accepted candidate's own
certificate call, not part of the refresh); its seconds are reported for scale.

Input (read only): --state, a final_native.npz written by bayesreg (keys u, state_hash, grid; default the GPU smoke G3
arm, /home/rmcfz/seq_vp_tag_results/m2_gpu_smoke/G3/final_native.npz).  Read through bayesreg.eval.fields
.read_final_native (float64 (nz, ny, nx, 3), finite, sha256(u) == state_hash); the grid is rebuilt from the stored grid
record and its grid_id checked.  No image, label, pair.npz or channel is read.  --count-refreshes also reads the
log.jsonl next to --state (small file) and counts its native-level S~ computations (rows kind "jd_scaling" at the
native level, if the rows carry `level`), to put the per-call cost in the run's terms (an estimate, labelled as such).
--synthetic N instead builds a certified synthetic state on an N^3 unit grid (small smooth bump, the "mid" edge-dip cell
of tests/test_bayesreg/solve/_edge_dip.py at the centre, so that all certificate paths occur) -- for dry runs and the
tool test; no data access.

Threads: gauss27+corners and closed27 are single-threaded numpy; the certificate's edge hull and prescreen kernels run
in jd_curl's thread chunks (numba NUMBA_NUM_THREADS workers), the per-cell fallback sequentially.  Run with the thread
environment of the solver runs being compared (recorded in env.txt and timing.json).  One CPU-only process: no GPU.

Outputs (--out, a new directory; nothing is overwritten): timing.json, cmd.txt, git.txt, env.txt.
Single run on a shared machine: the numbers depend on the load (loadavg before / after recorded); a per-call timing of a
fixed state is not a statement about whole-registration speed.

    python -B experiments/jdc/time_certificate_estimate.py --out /home/rmcfz/seq_vp_tag_results/jdc_timing/<run_id>
    python -B experiments/jdc/time_certificate_estimate.py --synthetic 24 --repeats 3 --out /tmp/x     # dry run
"""
import argparse
import hashlib
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

from jd_curl.three_d.grid import make_grid_3d  # noqa: E402

from bayesreg import memtrace  # noqa: E402
from bayesreg.solve import certificate_config, certify  # noqa: E402
from bayesreg.solve import jd_scaling as jds  # noqa: E402

SCHEMA = "jdc.time_certificate_estimate.v2"      # v2 (JDTI review minor 4): the certificate_bernstein variants
DEFAULT_STATE = Path("/home/rmcfz/seq_vp_tag_results/m2_gpu_smoke/G3/final_native.npz")
BERN_CACHED = "certificate_bernstein_cached"
VARIANTS = (jds.GAUSS_CORNERS, jds.CLOSED27, jds.CERTIFICATE, "certificate_cached", jds.CERTIFICATE_BERNSTEIN,
            BERN_CACHED)
ESTIMATE_OF = {"certificate_cached": jds.CERTIFICATE, BERN_CACHED: jds.CERTIFICATE_BERNSTEIN}
NATIVE_LEVEL = 3
THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "CUDA_VISIBLE_DEVICES")


# ---- inputs -------------------------------------------------------------------------------------------------

def grid_from_record(record):
    """The Grid3D of a bayesreg grid record (export.grid_record), its grid_id checked."""
    nz, ny, nx = record["shape_zyx"]
    hx, hy, hz = record["spacing_xyz"]
    x0, y0, z0 = record["origin_xyz"]
    grid = make_grid_3d(nx=nx, ny=ny, nz=nz, hx=hx, hy=hy, hz=hz, x0=x0, y0=y0, z0=z0, unit=record["unit"])
    if grid.grid_id != record["grid_id"]:
        raise SystemExit(f"grid record: rebuilt grid_id {grid.grid_id} differs from the stored {record['grid_id']}")
    return grid


def load_state(path):
    """(u read-only, grid, input record) from a bayesreg final_native.npz."""
    from bayesreg.eval.fields import read_final_native
    u, record = read_final_native(path)
    if record["grid"] is None or record["stored_state_hash"] is None:
        raise SystemExit(f"{path}: a bayesreg final_native.npz (with state_hash and grid) is required")
    grid = grid_from_record(record["grid"])
    if u.shape != grid.shape+(3,):
        raise SystemExit(f"{path}: u {u.shape} does not fit the grid {grid.shape}")
    u.flags.writeable = False
    return u, grid, dict(source="final_native", path=str(path), file_sha256=record["file_sha256"],
                         state_hash=record["u_sha256"], state_hash_matches=record["state_hash_matches"],
                         status=record["status"], grid=record["grid"])


def synthetic_state(n):
    """A certified synthetic state on an n^3 unit grid: a smooth bump of amplitude <= .07 mm (zero faces), set to zero
    on the 4^3 nodes of the centre cell and its neighbours, plus the "mid" edge-dip cell there."""
    from tests.test_bayesreg.solve import _edge_dip as dip, _synthetic as syn
    from bayesreg.solve.stopping import state_hash
    if n < 6:
        raise SystemExit("--synthetic: N >= 6")
    grid = syn.cube_grid(n)
    u = .1*syn.true_displacement(grid, (.7, -.6, .5))
    c = (n-2)//2
    u[c-1:c+3, c-1:c+3, c-1:c+3] = 0.                       # the dip cell and its 26 neighbours as constructed
    u = u+dip.edge_dip_field(grid, (c, c, c))
    u.flags.writeable = False
    return u, grid, dict(source="synthetic", n=n, state_hash=state_hash(u), grid=dict(shape_zyx=list(grid.shape),
                                                                                     grid_id=grid.grid_id))


def count_refreshes(state_path):
    """Native-level S~ computations in the run's log.jsonl (next to the state), by `at`; None without a log."""
    log = Path(state_path).parent/"log.jsonl"
    if not log.is_file():
        return None
    counts, total = {}, 0
    with log.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("kind") == "jd_scaling" and row.get("level", NATIVE_LEVEL) == NATIVE_LEVEL:
                counts[row.get("at")] = counts.get(row.get("at"), 0)+1
                total += 1
    return dict(log=str(log), native_level=NATIVE_LEVEL, n_computations=total, by_at=counts)


# ---- timing -------------------------------------------------------------------------------------------------

def one_call(variant, u, grid, cert, cert_config):
    """Seconds and record of one S~ computation (scaling_factors + free_scaling) for a variant."""
    estimate = ESTIMATE_OF.get(variant, variant)
    if variant in (jds.CERTIFICATE, jds.CERTIFICATE_BERNSTEIN):
        jds.clear_cell_bounds_cache()                     # untimed: an uncached refresh (both cache levels)
    elif variant == "certificate_cached":
        jds.certificate_cell_bounds(u, grid, cert, cert_config)   # untimed: make sure the entry is there
    elif variant == BERN_CACHED:
        jds.bernstein_cell_bounds(u, grid, cert, cert_config, jds.J_ACT)  # untimed: both levels' entries
    tick = time.perf_counter()
    r, record = jds.scaling_factors(u, grid, estimate=estimate, cert=cert, cert_config=cert_config)
    jds.free_scaling(r)
    seconds = time.perf_counter()-tick
    memtrace.mark("jdc_timing.call", variant=variant, seconds=seconds)
    if variant in ("certificate_cached", BERN_CACHED) and not record["cell_bounds_cached"]:
        raise RuntimeError(f"{variant}: the bounds were not taken from the cache")
    if variant in (jds.CERTIFICATE, jds.CERTIFICATE_BERNSTEIN) and record["cell_bounds_cached"]:
        raise RuntimeError(f"{variant}: the bounds came from the cache")
    if variant == BERN_CACHED and not record["bernstein_cached"]:
        raise RuntimeError(f"{variant}: (S, l_S) were not taken from the cache")
    if variant == jds.CERTIFICATE_BERNSTEIN and record["bernstein_cached"]:
        raise RuntimeError(f"{variant}: (S, l_S) came from the cache")
    return seconds, record


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    q1, median, q3 = (float(v) for v in np.percentile(values, [25, 50, 75]))
    return dict(n=int(values.size), median=median, q1=q1, q3=q3, iqr=q3-q1, min=float(values.min()),
                max=float(values.max()))


def run_timing(u, grid, cert, cert_config, variants, repeats, warmup):
    out = {v: dict(seconds=[], inner_seconds=[], cell_bounds_seconds=[], bernstein_seconds=[], record=None)
           for v in variants}
    for _ in range(warmup):
        for variant in variants:
            one_call(variant, u, grid, cert, cert_config)
    for _ in range(repeats):
        for variant in variants:
            seconds, record = one_call(variant, u, grid, cert, cert_config)
            entry = out[variant]
            entry["seconds"].append(seconds)
            entry["inner_seconds"].append(record["seconds"])
            if "cell_bounds_seconds" in record:
                entry["cell_bounds_seconds"].append(record["cell_bounds_seconds"])
            if "bernstein_seconds" in record:
                entry["bernstein_seconds"].append(record["bernstein_seconds"])
            if entry["record"] is None:
                entry["record"] = {k: v for k, v in record.items() if not k.endswith("seconds")}
    for entry in out.values():
        entry["summary"] = summary(entry["seconds"])
        entry["inner_summary"] = summary(entry["inner_seconds"])
        entry["cell_bounds_summary"] = summary(entry["cell_bounds_seconds"]) if entry["cell_bounds_seconds"] else None
        entry["bernstein_summary"] = summary(entry["bernstein_seconds"]) if entry["bernstein_seconds"] else None
    return out


# ---- bookkeeping --------------------------------------------------------------------------------------------

def git_record():
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return dict(head=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")), branch=git("branch",
                                                                                                    "--show-current"))


def environment():
    try:
        from numba import config as numba_config
        numba_threads = int(numba_config.NUMBA_NUM_THREADS)
    except ImportError:
        numba_threads = None
    return dict(python=sys.version, numpy=np.__version__, platform=platform.platform(), cpu_count=os.cpu_count(),
                numba_threads=numba_threads, thread_env={k: os.environ.get(k) for k in THREAD_ENV})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--state", default=str(DEFAULT_STATE), help="bayesreg final_native.npz (read only)")
    source.add_argument("--synthetic", type=int, default=None, metavar="N", help="synthetic N^3 state (dry run)")
    parser.add_argument("--out", required=True, help="new output directory")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tile-size", type=int, default=None, help="certificate tile size T (default: C07-2's)")
    parser.add_argument("--variants", default=",".join(VARIANTS), help=f"comma list of {VARIANTS}")
    parser.add_argument("--count-refreshes", action="store_true",
                        help="also count the native-level S~ computations in the log.jsonl next to --state")
    args = parser.parse_args(argv)
    variants = tuple(v for v in args.variants.split(",") if v)
    if not variants or any(v not in VARIANTS for v in variants) or len(set(variants)) != len(variants):
        parser.error(f"--variants: distinct members of {VARIANTS}")
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats >= 1, --warmup >= 0")
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out}: exists (nothing is overwritten)")
    out.mkdir(parents=True)
    (out/"cmd.txt").write_text(" ".join([sys.executable, "-B", *sys.argv])+"\n", encoding="utf-8")
    git = git_record()
    (out/"git.txt").write_text(json.dumps(git, indent=1)+"\n", encoding="utf-8")
    env = environment()
    (out/"env.txt").write_text(json.dumps(env, indent=1)+"\n", encoding="utf-8")

    started = time.time()
    load_before = os.getloadavg()
    if args.synthetic is not None:
        u, grid, state_record = synthetic_state(args.synthetic)
    else:
        u, grid, state_record = load_state(args.state)
    cert_config = certificate_config() if args.tile_size is None else certificate_config(args.tile_size)
    cert = certify(u, grid, cert_config)
    if not cert.passed:
        raise SystemExit(f"the state does not pass the C07-2 certificate ({cert.status}, {cert.cert_reason})")
    timing = run_timing(u, grid, cert, cert_config, variants, args.repeats, args.warmup)
    default = timing.get(jds.GAUSS_CORNERS)
    ratios = None if default is None else {
        v: timing[v]["summary"]["median"]/default["summary"]["median"] for v in variants if v != jds.GAUSS_CORNERS}
    refreshes = count_refreshes(args.state) if args.count_refreshes and args.synthetic is None else None
    estimate = None
    if refreshes and default is not None and jds.CERTIFICATE in timing:
        extra = timing[jds.CERTIFICATE]["summary"]["median"]-default["summary"]["median"]
        estimate = dict(note="estimate, not a measurement: (median certificate - median default) x the native-level "
                             "S~ computations of the run's log; the certificate option changes the trajectory, so its "
                             "own run would compute S~ a different number of times",
                        extra_seconds_per_computation=extra,
                        extra_seconds_total=extra*refreshes["n_computations"])
    record = dict(schema=SCHEMA, purpose="JDC1: S~ computation cost per jd_cell_estimate option (D47 default decision); "
                                         "JDTI: certificate_bernstein for N1 of C09.1-3a (D53)",
                  memtrace=memtrace.enabled(),
                  script=str(Path(__file__).resolve().relative_to(ROOT)),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), git=git, environment=env,
                  started_unix=started, wall_seconds=time.time()-started, loadavg_before=load_before,
                  loadavg_after=os.getloadavg(), peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  state=state_record, n_cells=int(grid.n_cells), n_interior_nodes=int(np.prod(grid.interior_shape)),
                  certificate=dict(config=dict(tile_size=int(cert_config.tile_size),
                                               certificate_version=cert.certificate_version),
                                   outcome=cert.record()),
                  repeats=args.repeats, warmup=args.warmup, variants=list(variants), timing=timing,
                  median_ratio_to_default=ratios, native_refreshes=refreshes, run_cost_estimate=estimate,
                  caveats=["single run on a shared machine; see loadavg", "fixed state; not whole-registration speed",
                           "certify(u) is not part of the S~ computation (the solver has it from the accepted step)"])
    (out/"timing.json").write_text(json.dumps(record, indent=1)+"\n", encoding="utf-8")
    for v in variants:
        s = timing[v]["summary"]
        print(f"{v:20s} median {s['median']:.4f} s  IQR {s['iqr']:.4f} s  (q1 {s['q1']:.4f}, q3 {s['q3']:.4f}, "
              f"n {s['n']})", flush=True)
    return record


if __name__ == "__main__":
    main()
