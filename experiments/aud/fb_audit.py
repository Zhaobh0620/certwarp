"""AUD (F_B audit; the lead's brief of 2026-11-02 ~5 pm CST, gate A item 2; plan: runs/20261102-fb-audit/plan.md): which of
the existing final fields does the current objective F_B prefer, and in which term?  Evaluation at fixed fields only: no
solver, no GPU; bayesreg/ and jd_curl/ are not modified.

    python -B experiments/aud/fb_audit.py --pair 0438-0439 --out /home/rmcfz/seq_vp_tag_results/fb_audit_v1 [--fields ...]

One process = one pair; its fields are evaluated one after the other.  The model is the M2V1 configuration (14 channels,
gauss27+closed27, beta_S 1, Simpson; eta of m2v1_dev/eta/eta_m2v1.json), the CPU numpy reference of bayesreg.model, FP64,
1 thread.  Per field it writes raw/<short>/<field>.json, never overwriting (an existing file is skipped):

  (a) sigma = the sigma2_used of the M2V1 run's native level, the same for every field, so F_B is comparable: F_B; channel,
      diffusion, log-volume (Gauss27 / closed27) and curl blocks; the unnormalized ||Cu||^2, u^T K_Q u, sum (log J)^2/tau^2;
      <rho_c^2>_B; the prior blocks split into brain / non-brain cells; F_vol on the s_tau grid (tau_0 fixed, tau_k = s_tau
      tau_k^(0), the sweep's definition of experiments/eta/sweep.py).
  (b) one Gaussian M step at the field (C11-2): sigma~_c^2 = <rho_c^2>_B/(N_q + 2); descriptive, F_B is not comparable
      across fields then; the joint objective L(u, sigma~(u)) of C11-1 is (C11-3: across rounds only L is comparable).

Checks that stop the run (plan): the M2V1 field reproduces hyper.json F_B_end within 1e-10 (relative; that value is the
torch64 path's) and its state_hash; the P3 field, with the data blocks rescaled to P3's own sigma2_used, reproduces P3's
F_B_end; every field's state_hash and grid_id match its record; the tile pass reproduces evaluate's prior blocks (1e-12).
Label firewall: the model reads pair.npz through bayesreg.io.load_pair (fixed, moving, initial_u only).  Hard constraint 6:
development pairs of stage_L2R_val_v1 only (PAIRS below; load_pair's allowlist guard as well).
"""
import argparse
import json
import logging
import os
from pathlib import Path
import resource
import sys
import threading
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

RESULTS = Path("/home/rmcfz/seq_vp_tag_results")
DATA_ROOT = "/home/rmcfz/code/python_projects/seq_vp_tag/data/three_d"     # the repository's data symlink (read only)
SEG = RESULTS/"seg"
ETA_JSON = RESULTS/"m2v1_dev"/"eta"/"eta_m2v1.json"
PAIRS = {"0438-0439": "OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1", "0439-0440": "OASIS_OAS1_0439_MR1__OASIS_OAS1_0440_MR1",
         "0440-0441": "OASIS_OAS1_0440_MR1__OASIS_OAS1_0441_MR1", "0441-0442": "OASIS_OAS1_0441_MR1__OASIS_OAS1_0442_MR1",
         "0442-0443": "OASIS_OAS1_0442_MR1__OASIS_OAS1_0443_MR1"}
FIELDS = ("M2V1", "P3", "P4", "A0", "A1", "A4p", "u0")          # M2V1 first: the consistency check stops the run early
S_GRID = tuple(2.**(k/2) for k in range(-8, 9))                 # s_tau grid of the F_vol table
LOGJ_QUANTILES = (.01, .05, .25, .5, .75, .95, .99)
REL_TOL_F = 1e-10
REL_TOL_TILE = 1e-12
START_MEM_GB, FIELD_MEM_GB, RSS_LIMIT_GB = 9., 3., 12.
LOGVOL = dict(logvol_points="gauss27+closed27", logvol_closed27_share=1., closed27_weights="simpson")
LD = np.longdouble


class AuditStop(RuntimeError):
    """A plan check failed: stop, find out why first (nothing further is computed)."""


# ---- the computation at one field (tested on toy grids, tests/test_bayesreg/aud) -----------------------------------------

def prior_pass(state, u, s_grid=S_GRID):
    """The Gauss27 and closed27 tile loop of the prior blocks, with what evaluate does not give: the split into brain cells
    (lower-corner node in the mask, the convention of bayesreg.eval.metrics) and non-brain cells; the unnormalized sums
    grad = sum w |grad u|^2 (= u^T K_Q u/omega0), curl = sum w |c|^2 (= ||Cu||^2), with w = omega w_g/omega0; the logvol
    sums (log J)^2/tau^2 of both parts; and the logvol energy on the s_tau grid: tau^2(s) = A + s^2 B with the node fields
    A = Ptilde^0 tau_0^2, B = sum_{k>=1} Ptilde^k tau_k^2 (Q1 interpolation is linear, so at the points too)."""
    from jd_curl.three_d.cell_q1 import cell_tiles
    from bayesreg.geom import node_field, scalar_field, tile_geometry, tile_interpolate
    from bayesreg.readonly import adopt_readonly

    grid = state.grid
    u = adopt_readonly(node_field(u, grid))
    if not state.logvol_closed27:
        raise ValueError("prior_pass: gauss27+closed27 state required (the M2V1 model)")
    s2 = np.asarray(s_grid, dtype=np.float64)**2
    A = scalar_field(state.tissue[0]*state.tau2[0], grid, name="A")
    B = scalar_field(np.tensordot(state.tau2[1:], state.tissue[1:], axes=1), grid, name="B")
    brain = np.asarray(state.mask, dtype=bool)[:-1, :-1, :-1].ravel()
    w = state.ops.weights/state.omega0
    w_gauss = state.logvol_gauss_weights/state.omega0
    w_closed = state.closed27_point_weights/state.omega0
    grad, curl = [LD(0), LD(0)], [LD(0), LD(0)]                 # index 0 brain cells, 1 the other cells
    vol_g, vol_s = [LD(0), LD(0)], [LD(0), LD(0)]               # the (log J)^2/tau^2 sums at s_tau = 1
    table_g, table_s = np.zeros(s2.size, dtype=LD), np.zeros(s2.size, dtype=LD)

    def table(acc, weights, log2, a, b):
        for j, value in enumerate(s2):
            acc[j] += np.sum(weights*log2/(a+value*b), dtype=LD)

    for tile in cell_tiles(grid, state.tile_size):
        inside = brain[tile]
        geo = tile_geometry(u, state.ops, tile=tile)
        per_cell = np.sum(w*np.sum(geo.grad**2, axis=(-1, -2)), axis=-1)
        grad[0] += np.sum(per_cell[inside], dtype=LD)
        grad[1] += np.sum(per_cell[~inside], dtype=LD)
        per_cell = np.sum(w*np.sum(geo.curl**2, axis=-1), axis=-1)
        curl[0] += np.sum(per_cell[inside], dtype=LD)
        curl[1] += np.sum(per_cell[~inside], dtype=LD)
        log2 = np.log(geo.J)**2
        a, b = tile_interpolate(A, state.ops, tile=tile), tile_interpolate(B, state.ops, tile=tile)
        per_cell = np.sum(w_gauss*log2/(a+b), axis=-1)
        vol_g[0] += np.sum(per_cell[inside], dtype=LD)
        vol_g[1] += np.sum(per_cell[~inside], dtype=LD)
        table(table_g, w_gauss, log2, a, b)
        geo = tile_geometry(u, state.closed27_ops, tile=tile)
        log2 = np.log(geo.J)**2
        a, b = tile_interpolate(A, state.closed27_ops, tile=tile), tile_interpolate(B, state.closed27_ops, tile=tile)
        per_cell = np.sum(w_closed*log2/(a+b), axis=-1)
        vol_s[0] += np.sum(per_cell[inside], dtype=LD)
        vol_s[1] += np.sum(per_cell[~inside], dtype=LD)
        table(table_s, w_closed, log2, a, b)
    return dict(grad_sq=dict(brain=float(grad[0]), other=float(grad[1])),
                curl_sq=dict(brain=float(curl[0]), other=float(curl[1])),
                logvol_gauss_sq=dict(brain=float(vol_g[0]), other=float(vol_g[1])),
                logvol_closed27_sq=dict(brain=float(vol_s[0]), other=float(vol_s[1])),
                s_grid=[float(s) for s in s_grid],
                F_vol_gauss_of_s=[float(.5*v) for v in table_g], F_vol_closed27_of_s=[float(.5*v) for v in table_s])


def log_j_stats(state, u):
    """log J at the cell centres over the brain cells of the MODEL's mask (Atropos; no label read): quantiles etc., with the
    cell-centre J of bayesreg.eval.metrics."""
    from bayesreg.eval.metrics import cell_jacobian_centres

    J = cell_jacobian_centres(u, state.grid.spacing)
    cells = np.asarray(state.mask, dtype=bool)[:-1, :-1, :-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        lj = np.log(np.clip(J[cells], 1e-12, None))
    q = np.quantile(lj, LOGJ_QUANTILES)
    return dict(n_cells=int(lj.size), mask="model (Atropos), lower-corner node", quantiles=[float(x) for x in LOGJ_QUANTILES],
                values=[float(x) for x in q], mean=float(lj.mean()), std=float(lj.std()),
                pct_J_lt_half=float(100*np.mean(J[cells] < .5)), frac_abs_gt_log2=float(np.mean(np.abs(lj) > np.log(2.))),
                min_J_centre_all=float(J.min()))


def m_step_report(state, point):
    """(b): one C11-2 Gaussian M step at the field, F_B at sigma~ and the C11-1 joint objective L(u, sigma~) (a = b = 0)."""
    from bayesreg.solve.multilevel import GaussianMStep, n_q

    sigma2 = GaussianMStep().level_start(state, point)
    nq = n_q(state)
    priors = point.energies["diffusion"]+point.energies["logvol"]+point.energies["curl"]
    data = state.n_channels*(nq+2.)/2.                          # (N_q + 2)/2 per channel, by the M step
    log_term = (nq/2.+1.)*np.log(sigma2)
    return dict(sigma2=[float(x) for x in sigma2], F_data=float(data), F_B=float(data+priors),
                L=float(data+log_term.sum()+priors), L_log_channel=[float(x) for x in log_term], N_q=float(nq))


def data_at_sigma(energies, sigma_from, sigma_to):
    """The channel energies at sigma_to from those at sigma_from: E_c is linear in 1/sigma_c^2 (Gaussian, brain)."""
    return [float(e*a/b) for e, a, b in zip(energies["channel"], sigma_from, sigma_to)]


def field_report(state, u, s_grid=S_GRID):
    """Everything of one field at the frozen state; {"invalid": ...} when F_B has no value there."""
    from bayesreg.model import evaluate
    from bayesreg.solve.multilevel import channel_mean_square

    point = evaluate(state, u)
    if not hasattr(point, "energy"):
        return dict(invalid=str(getattr(point, "reason", point)))
    en = point.energies
    parts = en["logvol_parts"]
    channel = list(en["channel"])
    rho2 = channel_mean_square(state, point)
    passes = prior_pass(state, u, s_grid)
    report = dict(F_B=point.energy, F_data=float(sum(channel)), F_intensity=channel[0], F_structure=float(sum(channel[1:])),
                  channel=channel, F_diff=en["diffusion"], F_vol=en["logvol"], F_vol_gauss=parts["gauss"],
                  F_vol_closed27=parts["closed27"], F_curl=en["curl"], closed27_min_J=parts["closed27_min_J"],
                  rho2_B=[float(x) for x in rho2], m_step=m_step_report(state, point), prior_pass=passes,
                  u_max_abs_mm=float(np.abs(u).max()),
                  u_mean_norm_brain_mm=float(np.linalg.norm(np.asarray(u)[np.asarray(state.mask, dtype=bool)], axis=-1).mean()))
    del point
    report["grad_sq_total"] = passes["grad_sq"]["brain"]+passes["grad_sq"]["other"]
    report["curl_sq_total"] = passes["curl_sq"]["brain"]+passes["curl_sq"]["other"]
    report["logvol_sq_total"] = (passes["logvol_gauss_sq"]["brain"]+passes["logvol_gauss_sq"]["other"]
                                 + passes["logvol_closed27_sq"]["brain"]+passes["logvol_closed27_sq"]["other"])
    report["uKu"] = report["grad_sq_total"]*state.omega0        # K_D^T K_D = (alpha/omega0) K_Q (energy.py)
    report["log_j"] = log_j_stats(state, u)
    report["checks"] = tile_checks(state, report)
    return report


def tile_checks(state, report):
    """The tile pass against evaluate's blocks (plan, check 3); AuditStop beyond REL_TOL_TILE."""
    one = report["prior_pass"]["s_grid"].index(1.)
    pairs = dict(diffusion=(.5*state.alpha*report["grad_sq_total"], report["F_diff"]),
                 curl=(.5*state.nu*report["curl_sq_total"], report["F_curl"]),
                 logvol=(.5*report["logvol_sq_total"], report["F_vol"]),
                 logvol_table_s1=(report["prior_pass"]["F_vol_gauss_of_s"][one]+report["prior_pass"]["F_vol_closed27_of_s"][one],
                                  report["F_vol"]))
    out = {}
    for name, (mine, theirs) in pairs.items():
        out[name] = abs(mine-theirs)/max(abs(theirs), 1e-300) if theirs else abs(mine)
        if out[name] > REL_TOL_TILE and abs(mine-theirs) > 1e-9:          # (u = 0: both exactly 0)
            raise AuditStop(f"tile pass {name}: {mine!r} vs evaluate {theirs!r}")
    return out


# ---- the machine rules -----------------------------------------------------------------------------------------------

def mem_available_gb():
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])/2**20
    raise RuntimeError("no MemAvailable")


def wait_for_memory(gb, log, poll=60.):
    waited = 0.
    while True:
        have = mem_available_gb()
        if have >= gb:
            if waited:
                log.info("memory: %.1f GB available after waiting %.0f s", have, waited)
            return waited
        log.info("memory: %.1f GB available < %.1f GB: waiting", have, gb)
        time.sleep(poll)
        waited += poll


def rss_watchdog(limit_gb=RSS_LIMIT_GB):
    def run():
        while True:
            try:
                with open("/proc/self/status") as handle:
                    for line in handle:
                        if line.startswith("VmRSS:") and int(line.split()[1])/2**20 > limit_gb:
                            sys.stderr.write(f"RSS above {limit_gb} GB: abort\n")
                            os._exit(3)
            except OSError:
                pass
            time.sleep(.5)
    threading.Thread(target=run, daemon=True).start()


# ---- the fields ------------------------------------------------------------------------------------------------------

def field_source(field, short):
    """(npz path, expected state_hash, grid_id, record description) of an existing field; u0: (None, None, None, ...)."""
    long = PAIRS[short]
    if field == "u0":
        return None, None, None, "u = 0"
    if field in ("M2V1", "P3", "P4"):
        run = dict(M2V1=RESULTS/"m2v1_dev"/"runs"/"dev"/short, P3=RESULTS/"path_pilot_v1"/"runs"/"p3"/short,
                   P4=RESULTS/"path_pilot_v1"/"runs"/"p4"/short)[field]
        path = run/"final_native.npz"
        with np.load(path, allow_pickle=False) as data:
            digest, grid = str(data["state_hash"]), json.loads(str(data["grid"]))
        return path, digest, grid["grid_id"], "final_native.npz state_hash, grid"
    root = dict(A0=RESULTS/"d0_v1"/"runs"/"A0", A1=RESULTS/"d0_v1"/"runs"/"A1", A4p=RESULTS/"d0_v1_a4"/"runs"/"A4p")[field]
    run = root/long/"pg_weak__four__90s"
    record = json.loads((run/"final_native.json").read_text())
    return run/"final_native.npz", record["certificate"]["state_hash"], record["certificate"]["grid_id"], \
        "final_native.json certificate"


def load_field(field, short, grid):
    from bayesreg.solve.stopping import state_hash

    path, digest, grid_id, source = field_source(field, short)
    if path is None:
        return np.zeros(grid.shape+(3,)), dict(path=None, source=source)
    with np.load(path, allow_pickle=False) as data:
        u = np.array(data["u"])
    actual = state_hash(u)
    if actual != digest:
        raise AuditStop(f"{field} {short}: state_hash {actual} != record {digest} ({source})")
    if grid_id != grid.grid_id:
        raise AuditStop(f"{field} {short}: grid_id {grid_id} != model {grid.grid_id}")
    return u, dict(path=str(path), source=source, state_hash=actual, grid_id=grid_id)


def native_row(hyper, level=3):
    rows = [r for r in hyper["rounds"] if r["level"] == level]
    if len(rows) != 1:
        raise AuditStop(f"expected one round at level {level}, found {len(rows)}")
    return rows[0]


def consistency(field, short, report, sigma_a, eta):
    """Plan checks 1 and 2; returns the record, raises AuditStop on a mismatch."""
    if field not in ("M2V1", "P3"):
        return None
    run = dict(M2V1=RESULTS/"m2v1_dev"/"runs"/"dev"/short, P3=RESULTS/"path_pilot_v1"/"runs"/"p3"/short)[field]
    hyper = json.loads((run/"hyper.json").read_text())
    row = native_row(hyper)
    if hyper["eta"]["nu"] != eta["nu"] or hyper["eta"]["alpha"] != eta["alpha"] or hyper["eta"]["tau2"] != eta["tau2"]:
        raise AuditStop(f"{field} {short}: eta of the run {hyper['eta']} != audit {eta}")
    if report["field_record"]["state_hash"] != row["state_hash"]:
        raise AuditStop(f"{field} {short}: state_hash differs from hyper.json")
    if field == "M2V1":
        mine = report["F_B"]
    else:
        channel = data_at_sigma(dict(channel=report["channel"]), sigma_a, row["sigma2_used"])
        mine = float(sum(channel))+report["F_diff"]+report["F_vol"]+report["F_curl"]
    theirs = row["F_B_end"]
    rel = abs(mine-theirs)/abs(theirs)
    record = dict(field=field, F_B_recomputed=mine, F_B_end_hyper=theirs, rel_diff=rel, tolerance=REL_TOL_F,
                  backend_of_hyper_value=hyper["backend_after_solve"]["backend"], note=(
                      "CPU numpy reference vs the run's value" if field == "M2V1" else
                      "data blocks rescaled linearly to the run's own sigma2_used"))
    if rel > REL_TOL_F:
        raise AuditStop(f"{field} {short}: F_B {mine!r} vs hyper.json F_B_end {theirs!r}, rel {rel:.3e}")
    return record


def write_new(path, document):
    """Write a new JSON file via a temporary name; an existing file is an error (nothing is overwritten)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(document, indent=1, default=str)+"\n", encoding="utf-8")
    tmp.rename(path)


# ---- main ------------------------------------------------------------------------------------------------------------

def build_state(short, log):
    """The M2V1 native model of one pair: (state, pair record), run_pair's own input functions, sigma = sigma2_used."""
    from bayesreg.cli import run_pair
    from bayesreg.heap import release_free_heap
    from bayesreg.io import load_pair
    from bayesreg.solve.multilevel import _template, n_q

    name = f"stage_L2R_val_v1/{PAIRS[short]}"
    args = run_pair.parse_args(["--pair", name, "--channel-source", "structure", "--out", "/nonexistent/aud.unused",
                                "--logvol-points", LOGVOL["logvol_points"], "--logvol-closed27-share", "1",
                                "--closed27-weights", LOGVOL["closed27_weights"],
                                "--structure-root", str(SEG/"unet35_v1"/"pred"), "--structure-grouping", "structure13_v1",
                                "--segmenter-id", "unet35_v1_ens", "--structure-infer", "never",
                                "--data-root", DATA_ROOT, "--tissue-root", str(SEG/"atropos_v1"),
                                "--eta-json", str(ETA_JSON), "--cpu-threads", "1"])
    pair = load_pair(run_pair.resolve_pair(args.data_root, name), allowlist=args.allowlist)
    structure = run_pair.structure_source(args)
    inputs, pt_f, mask_f, provenance = run_pair.prepare_channels(pair, "structure", args.tissue_root, log, structure=structure)
    eta = run_pair.read_eta(args.eta_json, pt_f.shape[0]-1)
    native = inputs[-1]
    grid = native.grid
    del inputs, pt_f, mask_f
    release_free_heap()
    hyper = json.loads((RESULTS/"m2v1_dev"/"runs"/"dev"/short/"hyper.json").read_text())
    row = native_row(hyper)
    sigma_a = np.asarray(row["sigma2_used"], dtype=np.float64)
    host = _template(native, eta, grid.omega, 8192, native.fixed.shape[0], 1, LOGVOL["logvol_points"],
                     logvol_closed27_share=LOGVOL["logvol_closed27_share"], closed27_weights=LOGVOL["closed27_weights"])
    del native
    state = host.replace(sigma2=sigma_a)
    del host
    release_free_heap()
    q = grid.node_weights()[np.asarray(state.mask, dtype=bool)]
    mass = [float(np.sum(q*state.fixed_masked[c]**2, dtype=LD)/np.sum(q, dtype=LD)) for c in range(state.n_channels)]
    class_names = provenance["structure"]["fixed"]["accepted"]["class_names"]
    record = dict(short=short, pair=name, omega0=state.omega0, grid_id=grid.grid_id, grid_shape=list(grid.shape),
                  n_mask_nodes=int(state.node_index.size), N_q=float(n_q(state)), n_channels=int(state.n_channels),
                  channel_names=["intensity"]+list(class_names), channel_mean_fixed_square=mass,
                  sigma2_a=[float(x) for x in sigma_a], sigma2_a_source=f"m2v1_dev/runs/dev/{short}/hyper.json rounds level 3",
                  eta=eta.record(), logvol=LOGVOL, cpu_threads=int(state.cpu_threads), tile_size=int(state.tile_size),
                  channels_provenance=dict(structure_grouping="structure13_v1", segmenter_id="unet35_v1_ens"),
                  pair_provenance=pair.provenance, s_grid=[float(s) for s in S_GRID],
                  model=state.provenance(), code=code_record())
    return state, record


def code_record():
    import bayesreg
    root = Path(bayesreg.__file__).resolve().parents[1]
    sha = root/"SNAPSHOT_SHA"
    return dict(root=str(root), snapshot_sha=sha.read_text().strip() if sha.is_file() else None, argv=sys.argv,
                python=sys.version.split()[0], numpy=np.__version__,
                threads={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                        "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "CUDA_VISIBLE_DEVICES")})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pair", required=True, choices=sorted(PAIRS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--fields", default=",".join(FIELDS))
    ns = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    logging.Formatter.converter = time.gmtime
    log = logging.getLogger("aud")
    fields = [f for f in ns.fields.split(",") if f]
    if any(f not in FIELDS for f in fields):
        raise SystemExit(f"--fields: from {FIELDS}")
    out = Path(ns.out)/"raw"/ns.pair
    out.mkdir(parents=True, exist_ok=True)
    wanted = [f for f in fields if not (out/f"{f}.json").exists()]
    if not wanted:
        log.info("%s: nothing to do", ns.pair)
        return 0
    wait_for_memory(START_MEM_GB, log)
    rss_watchdog()
    tick = time.perf_counter()
    state, record = build_state(ns.pair, log)
    record["prepare_seconds"] = time.perf_counter()-tick
    log.info("%s: model ready in %.0f s, %d mask nodes, N_q %.1f", ns.pair, record["prepare_seconds"],
             record["n_mask_nodes"], record["N_q"])
    if not (out/"pair.json").exists():
        write_new(out/"pair.json", record)
    sigma_a = record["sigma2_a"]
    eta = dict(alpha=state.alpha, nu=state.nu, tau2=state.tau2.tolist())
    for field in wanted:
        wait_for_memory(FIELD_MEM_GB, log)
        tick = time.perf_counter()
        u, source = load_field(field, ns.pair, state.grid)
        report = field_report(state, u)
        if "invalid" in report:
            raise AuditStop(f"{field} {ns.pair}: F_B has no value: {report['invalid']}")
        report.update(field=field, short=ns.pair, field_record=source)
        report["consistency"] = consistency(field, ns.pair, report, sigma_a, eta)
        report["seconds"] = time.perf_counter()-tick
        report["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
        write_new(out/f"{field}.json", report)
        log.info("%s %s: F_B %.6e (data %.4e, diff %.3e, vol %.4e, curl %.4e) %.0f s, peak RSS %.1f GB%s", ns.pair, field,
                 report["F_B"], report["F_data"], report["F_diff"], report["F_vol"], report["F_curl"], report["seconds"],
                 report["peak_rss_gb"], "" if report["consistency"] is None else
                 f", check rel {report['consistency']['rel_diff']:.2e}")
        del u, report
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AuditStop as exc:
        sys.stderr.write(f"AUD STOP: {exc}\n")
        sys.exit(9)
