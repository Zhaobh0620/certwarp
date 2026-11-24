"""C08-9 L1 gate of the GPU-path stable comparison (contract C08-8/C08-9; work package SG).

Not run on real data by its author: the lead schedules the real run (after the D0 queue).  `selftest` runs on a
synthetic problem only.  Output directories must not exist or be empty (nothing is overwritten).

Three subcommands
-----------------
1. bases -- base points from the bayesreg solver on one development pair:
       python experiments/sg/l1_gate.py bases --run-dir RUN --steps 0,5 --out DIR [--backend torch64 --device cuda]
   RUN is a run_pair output directory of a development pair that reached the native level (final_native.npz; any
   ending).  The native ModelState is rebuilt exactly as run_pair built it (config.json args: pair, data root,
   allowlist -- the development guard applies --, channel source, tissue root; hyper.json: eta and sigma2_used of the
   last native round; the run's Gauss tile size), then the PG solver (outer option (a), C09.2) runs from the run's
   final_native u for max(steps) more accepted steps; the state after k accepted steps (k in --steps; 0 = the run's
   own final state) is saved as DIR/base_<pair>_k<k>.npz (u, state_hash, run_dir, k).  "Different stages" come from
   different k and from runs made with different --max-accepted; "different pairs" from runs of different pairs.
   C08-9 asks for at least 3 base points (different pairs, different stages).  Base points too close to convergence
   (GF_B/T_B ~ 1) have every candidate inside the fallback band; take earlier stages (e.g. a run_pair run with
   --rounds 1,1,1,1 --max-accepted 5..20).

2. gate -- the C08-9 check on those bases:
       python experiments/sg/l1_gate.py gate --bases DIR/base_*.npz --out /home/rmcfz/seq_vp_tag_results/sg_l1_gate/<tag>
           [--device cuda] [--directions +,-] [--near 4] [--far 1] [--inside 1] [--tiles 16] [--certify all|none]
   Per base point (native ModelState, torch64 backend on --device):
     * GF_B from the device gradient against the CPU gradient: relative difference (gate <= 1e-3);
     * PG direction p = -M^-1 g (CPU gradient, C09.1 metric at the base) and slope g.p;
     * candidates by step shrinking along +p and -p: tau_j = tau_hi 2^(-j/4) with tau_hi chosen so that the first-order
       change tau |g.p| is 32 s_hat (one secant correction after the first candidate): j = 0..near-1 aim at the near
       out-of-band interval 8 s_hat < |A_hat - r| <= 32 s_hat, `far` values above it (j = -8, -16, ...) and `inside`
       values below 8 s_hat (j = 10, 14, ...); every candidate is certified (CPU, C07-2) unless --certify none;
     * each candidate is compared for three classes on the same two states: pg (F_B, r = c1 tau |g.p|), outer_F (F_B,
       r = 0), outer_E (E_v with v = the base, default proximal parameters, r = 0); LM is not implemented in
       bayesreg.solve (PG only) and is reported as the missing class.  Per class: A_hat (GPU path, raw, no fallback),
       s_hat, A and s (CPU reference C08-1..7), the band, the GPU decision (|A_hat| > s_hat, A_hat - s_hat > 0,
       A_hat - s_hat >= r) and the CPU decision (same with A, s), and the decision of the full comparator (with its
       fallback) which must equal the CPU decision;
     * |A_hat - A| over every residual block (FB1): A_hat and A are each reassembled from their per-block changes --
       the channels, diffusion, log-volume (Gauss points and, under C05-2a, the cell corners: one block in both
       paths), curl, and for outer_E the proximal blocks prox_lambda, prox_mu, prox_nu2 -- in the order of
       gpu_fb_change / stable_reduction / E_v and must equal the totals bit for bit (gate `blocks_cover_total`); the
       per-block errors |A_hat_i - A_i| / s_hat and their sum are recorded per row and their maxima per class
       (reported; the gate is the contract's total |A_hat - A| / s_hat <= 1);
     * premise (i) of C08-8(c) over every residual block (FB1; reviews SG item 2, LVC item 2):
       sum_i (|r_i^t| |dr_i^t| + |r_i^b| |dr_i^b|), dr the device-minus-CPU residuals, for the base and one near-band
       trial: on sampled tiles (evenly spaced, --tiles) for the Gauss-point blocks -- diffusion, log-volume at the
       Gauss points and, under C05-2a, at the cell corners, curl -- divided by max(1, E_tile) (E_tile of all these
       blocks on the tile); over the whole mask for the channel blocks (node based; per channel), divided by
       max(1, E_channel); the proximal blocks of E_v are computed on the host from the same arrays in both paths
       (ExtraBlocks), so dr = 0 exactly (listed, value 0).  The gated value is the largest of the tile values and
       the channel value (<= 4e-12).
   Gates (C08-9): >= 3 bases; per class >= 10 near out-of-band candidates; zero decision mismatches out of the band;
   max |A_hat - A| / s_hat <= 1 over all candidates; premise <= 4e-12 (all blocks); GF_B relative difference <= 1e-3;
   the comparator's decision equals the CPU decision for every candidate; plus the tool check blocks_cover_total.
   summary.json has each gate's value and verdict, candidates.jsonl one row per (candidate, class), bases.json one
   row per base.  The native ModelState of a run is rebuilt with the run's logvol_points (C05-2a; C05-2b with its
   share and rule) and tile size.

3. selftest -- `gate` on base points of a synthetic problem (no data), CPU torch by default:
       python experiments/sg/l1_gate.py selftest --out DIR [--device cpu] [--n 21] [--near 4]
           [--logvol-points gauss27|gauss27+corners|gauss27+closed27]
   (also checks the proximal-change shortcut against bayesreg.solve.proximal.change on every candidate).

Cost of `gate` (an estimate, not measured at native size): per candidate one CPU certificate, one device and one
CPU evaluation of F_B, one CPU stable_reduction (two CPU geometries), the CPU proximal value P(u_t, v) and its change
of the outer_E class (one pass, `prox_value_and_change`: v is the base), and the three comparator calls (the CPU evaluations are
reused; a CPU stable_reduction again for the inside-band ones); the CPU parts dominate.  Timed on the 40 x 38 x 36
synthetic case (single BLAS thread) and scaled linearly by cells to the native 6.77e6 cells: CPU F_B ~85 s,
stable_reduction ~130 s, proximal value ~100 s, certificate a few s, i.e. ~5-6 min per candidate; with the defaults
(2 directions x (4 near + 1 far + 1 inside) = 12 candidates per base) about 1-1.5 h per base point, 3.5-4.5 h for 3
bases, single-threaded.  `selftest` prints seconds per candidate.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jd_curl.three_d.boundary import inject_free, restrict_free  # noqa: E402
from jd_curl.three_d.cell_q1 import cell_tiles  # noqa: E402

from bayesreg import model as M  # noqa: E402
from bayesreg.geom import tile_geometry  # noqa: E402
from bayesreg.model import ModelState, build_samplers  # noqa: E402
from bayesreg.model.blocks import GAUSS_BLOCKS, logvol_extra_block  # noqa: E402
from bayesreg.solve import (DirectObjective, PGConfig, ProximalParams, Torch64Backend, default_lambda,  # noqa: E402
                            g0_of, make_backend, run_pg, sampling_step_bound, threshold)
from bayesreg.solve import proximal as prox  # noqa: E402
from bayesreg.solve.export import dumps  # noqa: E402
from bayesreg.solve.stopping import gf_b, state_hash  # noqa: E402

LD = np.longdouble
EPS = float(np.finfo(np.float64).eps)
CLASSES = ("pg", "outer_F", "outer_E")
MISSING_CLASSES = {"lm": "LM-PCG (C09.3) is not implemented in bayesreg.solve (PG only): class not covered"}
DEFAULT_ROOT = Path("/home/rmcfz/seq_vp_tag_results/sg_l1_gate")
GATES = dict(min_bases=3, min_near_per_class=10, max_err_over_s_hat=1., max_premise=4e-12, max_gf_rel=1e-3)
RHO = 2.**-.25


# ----------------------------------------------------------------------------------------------
# Bookkeeping
# ----------------------------------------------------------------------------------------------

def prepare_out(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"{path}: exists and is not empty (nothing is overwritten)")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, document):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(dumps(document, indent=1)+"\n", encoding="utf-8")


def records(out, argv):
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout)
    (out/"cmd.txt").write_text(" ".join(argv)+f"\n# cwd {os.getcwd()}\n# python {sys.executable}\n")
    import torch
    write_json(out/"env.json", dict(git_head=head, git_dirty=dirty, python=sys.version, platform=platform.platform(),
                                    torch=torch.__version__, cuda=torch.version.cuda,
                                    threads={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                                                             "NUMBA_NUM_THREADS")}))


class Rows:
    def __init__(self, path):
        self.stream = open(path, "w", encoding="utf-8")

    def __call__(self, row):
        self.stream.write(dumps(row)+"\n")
        self.stream.flush()

    def close(self):
        self.stream.close()


# ----------------------------------------------------------------------------------------------
# Native ModelState of a run_pair run directory (subcommand bases / gate)
# ----------------------------------------------------------------------------------------------

def native_state_of_run(run_dir):
    """The native ModelState with which run_pair produced RUN/final_native.npz (sigma2_used of its last native
    round), and that u.  Reads data through load_pair with the run's allowlist (development guard)."""
    import logging
    from bayesreg.cli.run_pair import prepare_channels, resolve_pair
    from bayesreg.io import load_pair
    run_dir = Path(run_dir)
    config = json.loads((run_dir/"config.json").read_text())
    hyper = json.loads((run_dir/"hyper.json").read_text())
    args = config["args"]
    pair = load_pair(resolve_pair(args["data_root"], args["pair"]), allowlist=args["allowlist"])
    inputs, _, _, _ = prepare_channels(pair, args["channel_source"], args["tissue_root"], logging.getLogger("l1_gate"))
    native_rounds = [r for r in hyper["rounds"] if r.get("level") == len(inputs)-1 and "sigma2_used" in r]
    if not native_rounds:
        raise ValueError(f"{run_dir}: no native-level round in hyper.json")
    eta = hyper["eta"]
    lv = inputs[-1]
    state = ModelState(grid=lv.grid, omega0=pair.grid.omega, fixed=lv.fixed, samplers=build_samplers(lv.moving, lv.grid),
                       mask=lv.mask, sigma2=np.asarray(native_rounds[-1]["sigma2_used"]), alpha=eta["alpha"],
                       nu=eta["nu"], tau2=np.asarray(eta["tau2"]), tissue=lv.tissue,
                       tile_size=int(config["solver"].get("gauss_tile_size", 8192)),
                       logvol_points=config["solver"].get("logvol_points", "gauss27"),
                       **{k: config["solver"][k] for k in ("logvol_closed27_share", "closed27_weights")
                          if k in config["solver"]})
    with np.load(run_dir/"final_native.npz") as z:
        u = np.array(z["u"])
        stored = str(z["state_hash"])
    if state_hash(u) != stored:
        raise ValueError(f"{run_dir}: final_native.npz state_hash mismatch")
    return state, u, args["pair"]


def pg_states(backend, state, u0, steps, *, log=None):
    """Run PG (option (a), C09.2) from u0 for max(steps) accepted steps; {k: u after k accepted steps}."""
    grid = state.grid
    state = backend.prepare(state)
    wanted = sorted(set(int(k) for k in steps))
    out = {0: np.array(u0)} if 0 in wanted else {}
    if max(wanted) == 0:
        return out
    objective = DirectObjective(backend, state)
    point = objective.evaluate(u0)
    cert = backend.certify(u0, grid)
    if not cert.passed:
        raise ValueError("the start state does not pass the certificate")
    metric = backend.build_metric(state, c_dat=backend.c_dat(state, point), b_P=backend.metric_bP(state))
    target = threshold(g0_of(backend, state))
    counter = [0]

    def on_accept(theta, accepted, outcome):
        counter[0] += 1
        if counter[0] in wanted:
            out[counter[0]] = np.array(accepted.u)
    run_pg(objective, point, theta=restrict_free(u0, grid), cert=cert, metric=metric, certify=backend.certify,
           target=target, max_steps=max(wanted), config=PGConfig(), log=log, on_accept=on_accept)
    return out


def cmd_bases(ns):
    out = prepare_out(ns.out)
    records(out, sys.argv)
    state, u0, pair = native_state_of_run(ns.run_dir)
    backend = make_backend(ns.backend, device=ns.device)
    steps = [int(k) for k in ns.steps.split(",")]
    rows = Rows(out/"pg_log.jsonl")
    try:
        states = pg_states(backend, state, u0, steps, log=rows)
    finally:
        rows.close()
    written = []
    tag = pair.replace("/", "__")
    for k, u in sorted(states.items()):
        path = out/f"base_{tag}_k{k}.npz"
        np.savez(path, u=u, state_hash=np.array(state_hash(u)), run_dir=np.array(str(Path(ns.run_dir).resolve())),
                 k=np.array(k), pair=np.array(pair))
        written.append(dict(path=str(path), k=k, state_hash=state_hash(u)))
    missing = sorted(set(steps)-set(states))
    write_json(out/"bases.json", dict(run_dir=str(ns.run_dir), pair=pair, steps=steps, written=written,
                                      missing_steps=missing, note="missing: the PG solve ended before that step"))
    print(json.dumps(dict(written=len(written), missing=missing)))


# ----------------------------------------------------------------------------------------------
# The gate on one base point
# ----------------------------------------------------------------------------------------------

def prox_value_and_change(state, u, v, params):
    """P(u, v) exactly as bayesreg.solve.proximal.evaluate and its change from P(v, v) exactly as proximal.change, in
    one pass of the same residual generator: with the reference v as the base every base residual is 0, so change's
    (t - 0)(t + 0) in longdouble is t*t in longdouble (evaluate squares in float64 first; the two differ).  Halves the
    CPU geometry work of the outer_E class; checked bitwise against both functions by `selftest`."""
    u, v, d = prox._fields(state, u, v)
    value = {name: LD(0) for name in prox.PROX_BLOCKS}
    change = {name: LD(0) for name in prox.PROX_BLOCKS}
    zero = LD(0)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for name, r in prox._residuals(state, params, u, v, d):
            value[name] += np.sum(r*r, dtype=LD)
            b = r.astype(LD)
            change[name] += np.sum((b-zero)*(b+zero), dtype=LD)
        half = LD(.5)
        parts = {name: half*value[name] for name in prox.PROX_BLOCKS}
        total = LD(0)
        for name in prox.PROX_BLOCKS:
            total += parts[name]
        changes = {name: half*change[name] for name in prox.PROX_BLOCKS}
    return prox.ProximalValue(u, v, parts, total), changes


def _decision(ared, guard, required):
    return bool(abs(ared) > guard and ared-guard > 0 and ared-guard >= required)


def _band(A_hat, s_hat, r):
    d = min(abs(A_hat), abs(A_hat-r))
    if d <= 8*s_hat:
        return "inside"
    return "near" if abs(A_hat-r) <= 32*s_hat else "far"


EXTRA_RESIDUAL_NAMES = dict(corners="logvol_corners", closed27="logvol_closed27")


def gauss_block_names(state):
    """The Gauss-point residual blocks of state's F_B in device_residuals' names (C05-2a: plus logvol_corners; C05-2b:
    plus logvol_closed27)."""
    extra = EXTRA_RESIDUAL_NAMES.get(state.logvol_extra)
    return ("diffusion", "logvol", "curl")+((extra,) if extra else ())


def _cpu_residuals(state, u, tile):
    """The CPU (reference) weighted residuals of every Gauss-point block on one tile, in device_residuals' layouts:
    diffusion, logvol, curl and, under C05-2a, logvol_corners (cells, 8) (LogVolumeCorners), under C05-2b
    logvol_closed27 (cells, 27) (LogVolumeClosed27)."""
    geo = tile_geometry(u, state.ops, tile=tile)
    out = {block.name: block.residual(block.scale(state, tile), geo) for block in GAUSS_BLOCKS}
    extra = logvol_extra_block(state)
    if extra is not None:
        points = tile_geometry(u, extra.operators(state), tile=tile)
        out[EXTRA_RESIDUAL_NAMES[state.logvol_extra]] = extra.residual(extra.scale(state, tile), points)
    return out


PROXIMAL_PREMISE_NOTE = ("the proximal blocks of E_v (option (b)) are computed on the host from the same u and v arrays "
                         "in the GPU path (ExtraBlocks) and in the CPU reference: dr = 0 exactly")


def premise(backend, state, points, n_tiles, *, device_residuals=None):
    """C08-8(c)(i) over every residual block of the compared energies, for the (base, trial) device points:
      Gauss-point blocks (diffusion, log-volume at the Gauss points and, under C05-2a, at the cell corners, curl) on
        evenly spaced tiles: sum_blocks sum_points ||r|| ||dr|| / max(1, E_tile), E_tile = the larger of the points'
        energies of these blocks on the tile;
      channel blocks (node based; over the whole mask): sum_c sum_points ||r_c|| ||dr_c|| / max(1, E_channel);
      proximal blocks of E_v: dr = 0 (PROXIMAL_PREMISE_NOTE), listed with value 0.
    `max` is the largest of the tile values and the channel value (the C08-9 premise gate).
    device_residuals: the device residual function (default stable.device_residuals; tests inject a perturbed one)."""
    from bayesreg.backends.torch64 import stable as S
    device_residuals = S.device_residuals if device_residuals is None else device_residuals
    names = gauss_block_names(state)
    tiles = list(cell_tiles(state.grid, state.tile_size))
    picks = sorted(set(np.linspace(0, len(tiles)-1, min(n_tiles, len(tiles))).round().astype(int).tolist()))
    worst = dict(value=0., tile=None)
    per_tile = []
    block_max = {name: 0. for name in names}
    for index in picks:
        tile = tiles[index]
        parts, energies = {name: 0. for name in names}, []
        for point in points:
            cpu = _cpu_residuals(state, point.u, tile)
            dev = device_residuals(point.device.state, point.device, tile.start, tile.stop)
            if set(cpu) != set(names) or set(dev) != set(names):
                raise RuntimeError(f"premise: residual blocks differ (cpu {sorted(cpu)}, device {sorted(dev)}, "
                                   f"expected {sorted(names)})")
            energy = 0.
            for name in names:
                r, d = cpu[name].ravel(), (dev[name]-cpu[name]).ravel()
                parts[name] += float(np.linalg.norm(r))*float(np.linalg.norm(d))
                energy += .5*float(r@r)
            energies.append(energy)
        scale = max(1., *energies)
        value = sum(parts.values())/scale
        blocks = {name: parts[name]/scale for name in names}
        for name in names:
            block_max[name] = max(block_max[name], blocks[name])
        per_tile.append(dict(tile=index, value=value, E_tile=max(energies), blocks=blocks))
        if value >= worst["value"]:
            worst = dict(value=value, tile=index)
    n_channels = state.n_channels
    channel_parts, energies = [0.]*n_channels, []
    for point in points:
        cpu = point.cpu if point.cpu is not None else M.evaluate(state, point.u)
        r_cpu, r_dev = np.asarray(cpu.channel.r), np.asarray(point.device.r_host)
        if r_cpu.shape != r_dev.shape or r_cpu.shape[0] != n_channels:
            raise RuntimeError(f"premise: channel residual shapes differ ({r_cpu.shape} vs {r_dev.shape})")
        energy = 0.
        for c in range(n_channels):
            r, d = r_cpu[c].ravel(), (r_dev[c]-r_cpu[c]).ravel()
            channel_parts[c] += float(np.linalg.norm(r))*float(np.linalg.norm(d))
            energy += .5*float(r@r)
        energies.append(energy)
    scale = max(1., *energies)
    channel = sum(channel_parts)/scale
    return dict(max=max(worst["value"], channel), tiles_max=worst["value"], worst_tile=worst["tile"], tiles=per_tile,
                channel_block=channel, channel_per_channel=[v/scale for v in channel_parts], E_channel=max(energies),
                gauss_blocks=list(names), gauss_block_max=block_max,
                proximal=dict(blocks=list(prox.PROX_BLOCKS), value=0., note=PROXIMAL_PREMISE_NOTE),
                n_tiles_total=len(tiles), tiles_sampled=len(picks))


def _reassemble(blocks, extra=None):
    """The longdouble total of per-block changes in the order of gpu_fb_change / stable_reduction (channels,
    diffusion, logvol, curl), then the proximal blocks of E_v in PROX_BLOCKS order (the order of the comparator)."""
    total = LD(0)
    for change in blocks["channel"]:
        total += change
    for name in ("diffusion", "logvol", "curl"):
        total += blocks[name]
    if extra is not None:
        for name in prox.PROX_BLOCKS:
            total += extra.changes[name]
    return total


def block_errors(blocks_hat, blocks_cpu, s_hat, extra=None):
    """|A_hat_i - A_i| / s_hat per residual block (channels summed over c; log-volume incl. the C05-2a corners as in
    both paths; the proximal blocks of E_v from the host values each side uses), and their sum."""
    out = dict(channel=float(sum(abs(LD(a)-LD(b)) for a, b in zip(blocks_hat["channel"], blocks_cpu["channel"])))
               / s_hat)
    for name in ("diffusion", "logvol", "curl"):
        out[name] = float(abs(LD(blocks_hat[name])-LD(blocks_cpu[name])))/s_hat
    if extra is not None:
        for name in prox.PROX_BLOCKS:            # the same ExtraBlocks on both sides (host): exactly 0
            out[name] = float(abs(extra.changes[name]-extra.changes[name]))/s_hat
    return out, float(sum(out.values()))


def gate_base(name, state, u, backend, *, rows, directions=("+", "-"), near=4, far=1, inside=1, n_tiles=16,
              certify=True, params=None, c1=PGConfig().c1, check_prox_shortcut=False):
    """C08-9 on one base point; returns the per-base summary and writes one row per (candidate, class)."""
    from bayesreg.backends.torch64 import stable as S
    grid = state.grid
    state = backend.prepare(state)
    tick = time.perf_counter()
    base = backend.evaluate(state, u)
    if base.device is None:
        raise RuntimeError(f"{name}: device evaluation of the base failed ({base.anomaly})")
    cpu_b = M.evaluate(state, base.u)
    base = replace(base, cpu=cpu_b)
    g_dev = backend.gradient(state, base)
    g_cpu = M.gradient(state, cpu_b)
    G_dev, G_cpu = gf_b(g_dev, grid.omega, state.omega0), gf_b(g_cpu, grid.omega, state.omega0)
    metric = M.build_metric(state, c_dat=M.c_dat(state, cpu_b), b_P=M.metric_bP(state))
    p = -metric.solve(g_cpu)
    slope = float(g_cpu@p)
    theta0 = restrict_free(base.u, grid)
    bound = sampling_step_bound(grid, base.u, p)
    params = params or ProximalParams(lam=default_lambda(grid))
    prox_b = prox.evaluate(state, base.u, base.u, params)
    s_hat0 = S.gpu_guard(base.energy, base.energy)
    tau_hi = 32*s_hat0/abs(slope)
    js = list(range(near))+[-8*(k+1) for k in range(far)]+[10+4*k for k in range(inside)]
    summary = dict(name=name, state_hash=state_hash(base.u), E=base.energy, G_gpu=G_dev, G_cpu=G_cpu,
                   gf_rel=abs(G_dev-G_cpu)/G_cpu if G_cpu > 0 else 0., slope=slope, tau_hi=tau_hi, domain_bound=bound,
                   candidates=0, rejected_by_certificate=0, device_anomalies=0, near_trial=None)
    premise_trial = None
    for sign in directions:
        direction = p if sign == "+" else -p
        corrected = False
        for j in js:
            tau = min(tau_hi*RHO**j, .9*bound)
            u_t = inject_free(theta0+tau*direction, grid)
            row = dict(base=name, direction=sign, j=j, tau=tau)
            if certify:
                outcome = backend.certify(u_t, grid)
                if not outcome.passed:
                    summary["rejected_by_certificate"] += 1
                    rows(dict(row, cls=None, certificate=outcome.cert_reason))
                    continue
            trial = backend.evaluate(state, u_t)
            if getattr(trial, "device", None) is None:
                summary["device_anomalies"] += 1
                rows(dict(row, cls=None, device_anomaly=getattr(trial, "anomaly", getattr(trial, "reason", None))))
                continue
            cpu_t = M.evaluate(state, trial.u)
            trial = replace(trial, cpu=cpu_t)
            summary["candidates"] += 1
            delta_f, blocks_f = S.gpu_fb_change(trial.device.state, base.device, trial.device)
            cpu_red = M.stable_reduction(state, cpu_b, cpu_t)
            expected = {"channel", "diffusion", "logvol", "curl"}
            if set(blocks_f) != expected or set(cpu_red.blocks) != expected or \
                    len(blocks_f["channel"]) != state.n_channels or len(cpu_red.blocks["channel"]) != state.n_channels:
                raise RuntimeError(f"block sets differ: GPU {sorted(blocks_f)}, CPU {sorted(cpu_red.blocks)}")
            prox_t, changes = prox_value_and_change(state, trial.u, base.u, params)
            if check_prox_shortcut:
                reference_value = prox.evaluate(state, trial.u, base.u, params)
                reference = prox.change(state, prox_b, reference_value, params)
                if (reference_value.total != prox_t.total or any(reference[k] != changes[k] for k in prox.PROX_BLOCKS)
                        or any(reference_value.parts[k] != prox_t.parts[k] for k in prox.PROX_BLOCKS)):
                    raise AssertionError("proximal one-pass value/change differs from prox.evaluate / prox.change")
            extra = S.ExtraBlocks({k: changes[k] for k in prox.PROX_BLOCKS}, prox_b.total, prox_t.total)
            first = None
            for cls in CLASSES:
                if cls == "outer_E":
                    d_hat, d_cpu = LD(delta_f), LD(cpu_red.delta)
                    for value in extra.changes.values():
                        d_hat += value
                        d_cpu += value
                    E_hat = (float(LD(base.total)+prox_b.total), float(LD(trial.total)+prox_t.total))
                    E_cpu = (float(LD(cpu_b.total)+prox_b.total), float(LD(cpu_t.total)+prox_t.total))
                else:
                    d_hat, d_cpu = delta_f, cpu_red.delta
                    E_hat, E_cpu = (base.energy, trial.energy), (cpu_b.energy, cpu_t.energy)
                r = c1*tau*abs(slope) if cls == "pg" else 0.
                A_hat, s_hat = float(-d_hat), S.gpu_guard(*E_hat)
                A, s = float(-d_cpu), float(16*EPS*max(E_cpu))
                cls_extra = extra if cls == "outer_E" else None
                covered = bool(_reassemble(blocks_f, cls_extra) == LD(d_hat)
                               and _reassemble(cpu_red.blocks, cls_extra) == LD(d_cpu))
                per_block, block_sum = block_errors(blocks_f, cpu_red.blocks, s_hat, cls_extra)
                band = _band(A_hat, s_hat, r)
                gpu, cpu = _decision(A_hat, s_hat, r), _decision(A, s, r)
                red = backend.stable_compare(state, base, trial, required=r, context="pg" if cls == "pg" else cls,
                                             energy="E_v" if cls == "outer_E" else "F_B",
                                             reference=state_hash(base.u) if cls == "outer_E" else None,
                                             extra=extra if cls == "outer_E" else None)
                pipeline = bool(red.reason is None and red.certified >= r)
                rows(dict(row, cls=cls, required=r, A_hat=A_hat, s_hat=s_hat, A=A, s=s, band=band,
                          dist_over_s_hat=abs(A_hat-r)/s_hat, err_over_s_hat=abs(A_hat-A)/s_hat,
                          block_err_over_s_hat=per_block, block_err_sum_over_s_hat=block_sum,
                          blocks_cover_total=covered, gpu_decision=gpu,
                          cpu_decision=cpu, comparator_decision=pipeline, comparator_path=red.path,
                          mismatch=bool(band != "inside" and gpu != cpu), pipeline_mismatch=pipeline != cpu))
                if cls == "outer_F":
                    first = (band, abs(A_hat))
            if first is not None and not corrected and j == 0 and first[1] > 0:
                # one secant correction: aim the j = 0 candidate at |A_hat| = 32 s_hat
                tau_hi = min(tau_hi*32*s_hat0/first[1], .9*bound/RHO**min(js))
                corrected = True
            if premise_trial is None and first is not None and first[0] == "near":
                premise_trial = trial
    summary["near_trial"] = None if premise_trial is None else state_hash(premise_trial.u)
    summary["premise"] = premise(backend, state, [base] if premise_trial is None else [base, premise_trial], n_tiles)
    summary["seconds"] = time.perf_counter()-tick
    summary["seconds_per_candidate"] = summary["seconds"]/max(1, summary["candidates"])
    return summary


def summarize(bases, candidates_path, missing=MISSING_CLASSES):
    rows = [json.loads(line) for line in Path(candidates_path).read_text().splitlines()]
    rows = [r for r in rows if r.get("cls")]
    per_class = {}
    for cls in CLASSES:
        sel = [r for r in rows if r["cls"] == cls]
        per_class[cls] = dict(candidates=len(sel), near=sum(r["band"] == "near" for r in sel),
                              far=sum(r["band"] == "far" for r in sel), inside=sum(r["band"] == "inside" for r in sel),
                              mismatches=sum(r["mismatch"] for r in sel),
                              pipeline_mismatches=sum(r["pipeline_mismatch"] for r in sel),
                              max_err_over_s_hat=max((r["err_over_s_hat"] for r in sel), default=None),
                              max_block_err_over_s_hat={name: max(r["block_err_over_s_hat"][name] for r in sel)
                                                        for name in (sel[0]["block_err_over_s_hat"] if sel else ())},
                              max_block_err_sum_over_s_hat=max((r["block_err_sum_over_s_hat"] for r in sel),
                                                               default=None),
                              blocks_not_covering_total=sum(not r["blocks_cover_total"] for r in sel))
    errs = [r["err_over_s_hat"] for r in rows]
    premise_max = max((b["premise"]["max"] for b in bases), default=None)
    gf_max = max((b["gf_rel"] for b in bases), default=None)
    gates = dict(
        bases=dict(value=len(bases), limit=GATES["min_bases"], passed=len(bases) >= GATES["min_bases"]),
        near_per_class={cls: dict(value=v["near"], limit=GATES["min_near_per_class"],
                                  passed=v["near"] >= GATES["min_near_per_class"]) for cls, v in per_class.items()},
        out_of_band_mismatches=dict(value=sum(v["mismatches"] for v in per_class.values()), limit=0,
                                    passed=all(v["mismatches"] == 0 for v in per_class.values())),
        comparator_mismatches=dict(value=sum(v["pipeline_mismatches"] for v in per_class.values()), limit=0,
                                   passed=all(v["pipeline_mismatches"] == 0 for v in per_class.values())),
        max_err_over_s_hat=dict(value=max(errs, default=None), limit=GATES["max_err_over_s_hat"],
                                passed=bool(errs) and max(errs) <= GATES["max_err_over_s_hat"]),
        premise=dict(value=premise_max, limit=GATES["max_premise"],
                     passed=premise_max is not None and premise_max <= GATES["max_premise"]),
        gf_relative=dict(value=gf_max, limit=GATES["max_gf_rel"], passed=gf_max is not None and gf_max <= GATES["max_gf_rel"]),
        blocks_cover_total=dict(value=sum(v["blocks_not_covering_total"] for v in per_class.values()), limit=0,
                                passed=all(v["blocks_not_covering_total"] == 0 for v in per_class.values()),
                                note="tool check (FB1): A_hat and A are the exact sums of the per-block changes "
                                     "(channels, diffusion, log-volume incl. C05-2a corners, curl; E_v: proximal)"))
    flat = [gates["bases"], gates["out_of_band_mismatches"], gates["comparator_mismatches"], gates["max_err_over_s_hat"],
            gates["premise"], gates["gf_relative"], gates["blocks_cover_total"], *gates["near_per_class"].values()]
    verdict = "PASS" if all(g["passed"] for g in flat) else "FAIL"
    if missing:
        verdict += " (incomplete: " + "; ".join(f"{k}: {v}" for k, v in missing.items()) + ")"
    return dict(contract="C08-9", verdict=verdict, gates=gates, per_class=per_class, missing_classes=missing,
                premise_blocks=dict(gauss_on_sampled_tiles=sorted({n for b in bases for n in b["premise"]["gauss_blocks"]}),
                                    channel="whole mask, per channel", proximal=PROXIMAL_PREMISE_NOTE),
                bases=[{k: v for k, v in b.items() if k != "premise"} | dict(
                    premise_max=b["premise"]["max"], premise_tiles_max=b["premise"]["tiles_max"],
                    premise_channel=b["premise"]["channel_block"], premise_gauss_block_max=b["premise"]["gauss_block_max"])
                       for b in bases])


def run_gate(named_states, out, *, device, **options):
    """named_states: [(name, ModelState, u)]; writes candidates.jsonl, bases.json, summary.json into out."""
    backend = Torch64Backend(device=device, cache_size=4)
    rows = Rows(out/"candidates.jsonl")
    bases = []
    try:
        for name, state, u in named_states:
            summary = gate_base(name, state, u, backend, rows=rows, **options)
            bases.append(summary)
            print(json.dumps(dict(base=name, candidates=summary["candidates"], seconds=round(summary["seconds"], 1),
                                  seconds_per_candidate=round(summary["seconds_per_candidate"], 2))), flush=True)
    finally:
        rows.close()
    write_json(out/"bases.json", bases)
    result = summarize(bases, out/"candidates.jsonl")
    result.update(device=backend.dev.describe(), options={k: v for k, v in options.items() if k != "params"},
                  comparator=backend.comparison_summary() | dict(evaluate_anomalies=None))
    write_json(out/"summary.json", result)
    return result


def cmd_gate(ns):
    out = prepare_out(ns.out)
    records(out, sys.argv)
    named, cache = [], {}
    for path in ns.bases:
        with np.load(path) as z:
            u, digest, run_dir = np.array(z["u"]), str(z["state_hash"]), str(z["run_dir"])
            k = int(z["k"])
        if state_hash(u) != digest:
            raise ValueError(f"{path}: state_hash mismatch")
        if run_dir not in cache:
            cache[run_dir] = native_state_of_run(run_dir)[0]
        named.append((f"{Path(path).stem}", cache[run_dir], u))
    result = run_gate(named, out, device=ns.device, directions=tuple(ns.directions.split(",")), near=ns.near,
                      far=ns.far, inside=ns.inside, n_tiles=ns.tiles, certify=ns.certify == "all")
    print(result["verdict"])
    return 0 if result["verdict"].startswith("PASS") else 1


# ----------------------------------------------------------------------------------------------
# Self-test on a synthetic problem
# ----------------------------------------------------------------------------------------------

def synthetic_state(n=21, *, tile_size=2048, logvol_points="gauss27"):
    """A textured synthetic pair (intensity + 3 soft channels, Student-t free), ellipsoid mask, smooth tissue;
    logvol_points: C05-2 (default), C05-2a ("gauss27+corners") or C05-2b ("gauss27+closed27", beta_S 1, Simpson)."""
    from jd_curl.three_d.grid import make_grid_3d
    grid = make_grid_3d(nx=n, ny=n+2, nz=n-2, hx=1., hy=1.1, hz=.9, x0=-3., y0=2., z0=1., unit="mm")
    X = grid.identity()

    def images(points):
        x, y, z = np.moveaxis(points, -1, 0)
        intensity = .5+.3*np.sin(.31*x+.2*y)*np.cos(.27*z)+.2*np.cos(.19*x-.23*y+.29*z)
        logits = np.stack((np.sin(.21*x)+np.cos(.17*y+.1*z), np.cos(.23*z-.11*x)+.5*np.sin(.3*y),
                           np.sin(.13*x+.19*y-.07*z)))
        e = np.exp(logits-logits.max(axis=0))
        return np.concatenate((intensity[None], np.sqrt(e/e.sum(axis=0))))
    rng = np.random.default_rng(11)
    moving = images(X)
    fixed = images(X+np.array([.3, -.2, .25]))+.01*rng.standard_normal((4,)+grid.shape)
    centre = np.array([a.mean() for a in grid.axes])
    extent = np.array([a[-1]-a[0] for a in grid.axes])
    mask = np.sum(((X-centre)/(.4*extent))**2, axis=-1) <= 1.
    x, y, z = np.moveaxis(X, -1, 0)
    logits = np.stack((np.cos(.15*x)+np.sin(.12*y), np.sin(.2*z-.1*x), np.cos(.18*y+.05*z), .5*np.sin(.1*(x+y+z))))
    e = np.exp(logits-logits.max(axis=0))
    return ModelState(grid=grid, omega0=grid.omega, fixed=fixed, samplers=build_samplers(moving, grid), mask=mask,
                      sigma2=np.array([.04, .09, .02, .05]), alpha=.13, nu=.07, tau2=np.array([1., .25, .36, .16]),
                      tissue=e/e.sum(axis=0), tile_size=tile_size, logvol_points=logvol_points)


def selftest_bases(state, steps=(2, 6, 12), *, device="cpu"):
    backend = make_backend("torch64", device=device)
    zero = np.zeros(state.grid.shape+(3,))
    states = pg_states(backend, state, zero, steps)
    return [(f"synthetic_k{k}", state, u) for k, u in sorted(states.items())]


def cmd_selftest(ns):
    out = prepare_out(ns.out)
    records(out, sys.argv)
    state = synthetic_state(ns.n, logvol_points=ns.logvol_points)
    named = selftest_bases(state, device=ns.device)
    result = run_gate(named, out, device=ns.device, directions=("+", "-"), near=ns.near, far=1, inside=1,
                      n_tiles=8, certify=True, check_prox_shortcut=True)
    print(result["verdict"])
    return 0 if result["verdict"].startswith("PASS") else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bases", help="base points from a run_pair run directory")
    b.add_argument("--run-dir", required=True)
    b.add_argument("--steps", default="0,5")
    b.add_argument("--out", required=True)
    b.add_argument("--backend", choices=("numpy", "torch64"), default="numpy")
    b.add_argument("--device", default=None)
    g = sub.add_parser("gate", help="the C08-9 gate on base points")
    g.add_argument("--bases", nargs="+", required=True)
    g.add_argument("--out", required=True, help=f"run directory, e.g. {DEFAULT_ROOT}/<tag>")
    g.add_argument("--device", default=None, help="cuda (default when available) or cpu")
    g.add_argument("--directions", default="+,-")
    g.add_argument("--near", type=int, default=4)
    g.add_argument("--far", type=int, default=1)
    g.add_argument("--inside", type=int, default=1)
    g.add_argument("--tiles", type=int, default=16)
    g.add_argument("--certify", choices=("all", "none"), default="all")
    s = sub.add_parser("selftest", help="the gate on a synthetic problem (no data)")
    s.add_argument("--out", required=True)
    s.add_argument("--device", default="cpu")
    s.add_argument("--n", type=int, default=21)
    s.add_argument("--near", type=int, default=4)
    s.add_argument("--logvol-points", choices=("gauss27", "gauss27+corners", "gauss27+closed27"), default="gauss27")
    ns = parser.parse_args(argv)
    return {"bases": cmd_bases, "gate": cmd_gate, "selftest": cmd_selftest}[ns.command](ns) or 0


if __name__ == "__main__":
    sys.exit(main())
