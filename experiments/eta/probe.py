"""C05-7 scale probe nu* of one pair (work packages ETA, ETA2; contract C05-7 and its appendix 7a, pending review).

    python -B experiments/eta/probe.py --pair <stage>/<pair> --channel-source none|tissue|structure --out DIR
        [--logvol-points gauss27+closed27|gauss27|gauss27+corners] [--logvol-closed27-share BETA]
        [--closed27-weights simpson|uniform] [--metric-jd-scaling affine_v1|off]
        [--jd-cell-estimate certificate|gauss27+corners|closed27]
        [--data-root DIR] [--allowlist PATH] [--tissue-root DIR] [--cpu-threads N]
        [--structure-root DIR] [--segmenter-registry PATH] [--structure-registry-staging] [--structure-groups PATH]
        [--structure-grouping NAME] [--segmenter-id ID] [--sigma-domain brain|support|support_shared]
        [--support-threshold THETA]

sigma_domain (C04-6, C05-7a appended note; work package SWLI): run_pair's --sigma-domain / --support-threshold, names,
defaults and checks (run_pair.resolve_sigma_domain_args).  The probe's solve_multilevel gets them in its SolverConfig, so
the round state at u_1 carries the level's frozen support v and its sigma (C11-8 level start) and nu*'s numerator
||sum_c K_c^T r_c|| contains v.  `configuration` gains sigma_domain and support_threshold only under support /
support_shared (brain: the record before SWLI, key for key); provenance.json switches likewise; probe.json sigma2_used
has null for an empty-support channel.  A nu* under brain is not a nu* under support_shared (C05-7a).

The model and path options are run_pair's, with run_pair's defaults (since work package DEF, 2026-10-31, the point set
too: gauss27+closed27 with beta_S 1 and the Simpson rule, and the cell estimate certificate; before DEF the probe's
point set defaulted to gauss27+corners, the first sweep plan's -- experiments/eta/sweep.py always passes it), its value
checks (run_pair.resolve_jd_cell_estimate: with --metric-jd-scaling off --jd-cell-estimate is refused when given and
recorded as "none" otherwise; --logvol-closed27-share / --closed27-weights only with gauss27+closed27) and its meaning,
so that nu* is computed under
the configuration of the runs it centres (ETA2: with --channel-source structure the data-term gradient below includes the
structure-layer channels). The structure layer is read exactly as run_pair reads it (run_pair.structure_source ->
prepare_channels: registry, D24 preflight, provenance, read-back); a missing prediction is refused (structure_infer is
fixed to "never": the probe never runs the segmenter; the inference is a separate step whose command
experiments/eta/sweep.py prints).

What it computes (C05-7, "尺度探针"): u_1 = the state after the first accepted step on level 0 with the temporary values
eta_tmp (P2: tau^(0), nu_tmp = <1/tau^(0)2>_B, alpha_tmp = alpha_nug), and at u_1, with the level-0 state of that step
(sigma_c^2 of the level start, eta_tmp),
    nu*    = ||sum_c K_c^T r_c|| / ||C^T W C u_1 / omega0||,
    alpha* = ||sum_c K_c^T r_c|| / ||K_Q u_1 / omega0||        (P1 / P3 only; recorded, not used under P2),
Euclidean norms of the free vectors (3 N_int,); the ratio is unchanged by any common normalization (e.g. GF_B's
sqrt(omega_ell/omega0)). sum_c K_c^T r_c is bayesreg.model.gradient(blocks=("channel",)); C^T W C u_1 / omega0 is the curl
block's gradient of the same state with nu = 1 (alpha = 1 and the diffusion block for K_Q u_1 / omega0).

How (the smallest change that leaves the default path alone: bayesreg.cli.run_pair is not modified): the run_pair inputs
(load_pair with the subject guard and the label firewall, the tissue layer, C03.2-C03.6, eta_tmp of C05-7; run_pair's own
functions) and bayesreg.solve.solve_multilevel on the CPU FP64 reference (numpy backend) with max_levels = 1 and the C13-4
cap max_accepted = 1. Option (a) makes every accepted step a formal commit, and the PG loop checks the cap only before a
new iteration, so u_1 is the first accepted state of an uncapped run bit for bit (tests: its state_hash equals the first
"accept" row of an uncapped level-0 solve). The certificate is the CPU one, as always.

Status (status.json, probe.json): probe_done (nu* computed; exit 0); probe_failed (level 0 made no accepted step, or u_1 is
not usable; exit 2); subject_sealed (exit 3); provenance_rejected (exit 4); exception (exit 1). The solver status is
recorded as solver_status: stopped_before_native (max_levels = 1, "未到原生层的停止") -- or level_failed when C13-7 ends the
capped round, which leaves u_1 unchanged; never a disp.npz, never a native state. u_1 is kept as final_level0.npz.
Files: cmd.txt, git.txt, env.txt, data.txt, config.json, provenance.json, log.txt, log.jsonl, final_level0.npz, probe.json,
status.json. The run directory must not exist or be empty (nothing is overwritten). probe.json and status.json carry
`configuration` (channel source; point set with beta_S and the closed27 rule, both null unless gauss27+closed27; JD
scaling and its cell estimate; the structure source: root, grouping, segmenter id, registry path and registration
state), the number of model channels, K_T and K_S; status.json also structure_registry as run_pair writes it.  With
--jd-cell-estimate certificate_bernstein both also carry bernstein_j53_violations_total (the run total of the C09.1-3a
check (b), as run_pair's status.json; JDTI review minor 6).
"""
import argparse
import os
from pathlib import Path
import shlex
import sys
import time
import traceback

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bayesreg.cli import run_pair as rp  # noqa: E402
from bayesreg.geom.parallel import DEFAULT_CPU_THREADS  # noqa: E402
from bayesreg.io import ProvenanceRejected, SubjectSealed, load_pair  # noqa: E402
from bayesreg.model import (CLOSED27_SIMPSON, CLOSED27_WEIGHTS, LOGVOL_DEFAULT, LOGVOL_GAUSS27_CLOSED27,  # noqa: E402
                            LOGVOL_POINTS, evaluate, gradient)
from bayesreg.model.support import BRAIN, SIGMA_DOMAINS  # noqa: E402
from bayesreg.seg.structure_layer import DEFAULT_GROUPING, SEGMENTER_ID  # noqa: E402
from bayesreg.seg.unet.groups import DEFAULT_GROUPS_PATH  # noqa: E402
from bayesreg.solve import (CHANNEL_SOURCES, Caps, GaussianMStep, SolverConfig, eta_temporary,  # noqa: E402
                            make_backend, solve_multilevel)
from bayesreg.solve.export import JsonlLog, dumps, grid_record, write_final_state, write_json  # noqa: E402
from bayesreg.solve.jd_scaling import (AFFINE_V1, CELL_ESTIMATES as JD_CELL_ESTIMATES, CERTIFICATE_BERNSTEIN,  # noqa: E402
                                       MODES as JD_SCALING_MODES, j53_violations_total)
from bayesreg.solve.stopping import MAX_OUTER, RUN_WORDING, state_hash  # noqa: E402

SCHEMA = "bayesreg.eta_probe.v2"
PROBE_DONE, PROBE_FAILED = "probe_done", "probe_failed"
EXIT = {PROBE_DONE: 0, PROBE_FAILED: 2, "subject_sealed": 3, "provenance_rejected": 4, "exception": 1}
ROUNDS = (1, 1, 1, 1)          # only level 0, round 0 runs (max_levels = 1)
PROBE_CAPS = Caps(max_accepted=1, max_outer=MAX_OUTER)
STRUCTURE_INFER = "never"      # the probe never runs the segmenter (the inference is a separate step)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -B experiments/eta/probe.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--pair", required=True, help="<stage>/<pair>")
    parser.add_argument("--channel-source", required=True, choices=CHANNEL_SOURCES)
    parser.add_argument("--out", required=True, help="probe directory (must not exist or be empty)")
    parser.add_argument("--logvol-points", choices=LOGVOL_POINTS, default=LOGVOL_DEFAULT,
                        help="C05-2 / C05-2a / C05-2b point set (run_pair's option and default since DEF: "
                        "gauss27+closed27; before DEF the probe's default was gauss27+corners, the first sweep plan's; "
                        "experiments/eta/sweep.py always passes it)")
    parser.add_argument("--logvol-closed27-share", dest="logvol_closed27_share", type=rp._positive_share, default=1.,
                        help="C05-2b beta_S > 0 (run_pair's option, default 1); only with --logvol-points "
                        "gauss27+closed27")
    parser.add_argument("--closed27-weights", dest="closed27_weights", choices=CLOSED27_WEIGHTS,
                        default=CLOSED27_SIMPSON, help="C05-2b / D48 closed27 rule (run_pair's option, default "
                        "simpson); only with --logvol-points gauss27+closed27")
    parser.add_argument("--metric-jd-scaling", choices=JD_SCALING_MODES, default=AFFINE_V1)
    parser.add_argument("--jd-cell-estimate", choices=JD_CELL_ESTIMATES, default=None,
                        help="C09.1-3a cell estimate of the JD scaling (run_pair's option and default since DEF: "
                        "certificate; D47; path only, but u_1 lies on that path); no effect with --metric-jd-scaling "
                        "off: not given -> recorded as none, given -> refused")
    parser.add_argument("--data-root", default=str(rp.DEFAULT_DATA_ROOT))
    parser.add_argument("--allowlist", default=str(rp.DEFAULT_ALLOWLIST))
    parser.add_argument("--tissue-root", default=str(rp.DEFAULT_TISSUE_ROOT))
    parser.add_argument("--cpu-threads", type=int, default=DEFAULT_CPU_THREADS)
    structure = parser.add_argument_group("structure layer (--channel-source structure; run_pair's options, read only)")
    structure.add_argument("--structure-root", default=str(rp.DEFAULT_STRUCTURE_ROOT),
                           help="predictions <root>/<subject>/ (structure_groups.npz, provenance.json); a missing one "
                           "is refused (the probe never generates predictions)")
    structure.add_argument("--segmenter-registry", default=None,
                           help="segmenters.json: default the repository configs/bayesreg/segmenters.json "
                           f"(registered); with --structure-registry-staging the staging copy "
                           f"{rp.DEFAULT_SEGMENTER_REGISTRY}")
    structure.add_argument("--structure-registry-staging", action="store_true",
                           help="admit a registry other than the repository one (未正式登记; dry runs and development "
                           "diagnostics only), as run_pair")
    structure.add_argument("--structure-groups", default=str(DEFAULT_GROUPS_PATH), help="grouping table")
    structure.add_argument("--structure-grouping", default=DEFAULT_GROUPING,
                           help="grouping name (status default / ablation / verified only)")
    structure.add_argument("--segmenter-id", default=SEGMENTER_ID)
    parser.add_argument("--sigma-domain", choices=SIGMA_DOMAINS, default=None,
                        help="C04-6 / C05-7a (SWL draft; work package SWLI): run_pair's --sigma-domain (default brain); "
                        "under support / support_shared the data-term gradient of nu* contains the frozen support "
                        "weights v (the round state of solve_multilevel)")
    parser.add_argument("--support-threshold", type=rp._support_threshold, default=None,
                        help="C04-6 / D50 theta, run_pair's option and checks (default .05 with support / "
                        "support_shared; refused with brain)")
    args = parser.parse_args(argv)
    # run_pair.parse_args's value checks, with its messages
    args.jd_cell_estimate = rp.resolve_jd_cell_estimate(parser, args.metric_jd_scaling, args.jd_cell_estimate)
    if args.logvol_points != LOGVOL_GAUSS27_CLOSED27 and (args.logvol_closed27_share != 1.
                                                          or args.closed27_weights != CLOSED27_SIMPSON):
        parser.error("--logvol-closed27-share / --closed27-weights: only with --logvol-points gauss27+closed27 "
                     "(C05-2b)")
    if args.segmenter_registry is None:
        args.segmenter_registry = str(rp.DEFAULT_SEGMENTER_REGISTRY if args.structure_registry_staging
                                      else rp.REPO_REGISTRY)
    rp.resolve_sigma_domain_args(parser, args)          # run_pair's rules (C04-6): none / explicit theta refused, floor
    # the inference-only fields of run_pair's StructureSource (never used: structure_infer "never")
    args.structure_infer, args.structure_device, args.structure_synthetic = STRUCTURE_INFER, None, False
    args.segmenter_models_root = str(rp.DEFAULT_SEGMENTER_OUT)
    args.structure_image_root = str(rp.DEFAULT_STRUCTURE_IMAGE_ROOT)
    return args


def configuration(args):
    """The configuration the probe ran under (probe.json / status.json / config.json `configuration`;
    experiments/eta/sweep.py compares it with its own): beta_S and the closed27 rule are null unless gauss27+closed27
    (as run_pair.closed27_switches records them); structure is null unless channel_source structure."""
    closed = args.logvol_points == LOGVOL_GAUSS27_CLOSED27
    structure = None
    if args.channel_source == "structure":
        structure = dict(root=str(args.structure_root), grouping=args.structure_grouping,
                         groups=str(args.structure_groups), segmenter_id=args.segmenter_id,
                         registry=str(args.segmenter_registry),
                         registry_status=rp.structure_registry_label(args)["status"],
                         staging_opt_in=bool(args.structure_registry_staging), infer=args.structure_infer)
    out = dict(channel_source=args.channel_source, logvol_points=args.logvol_points,
               logvol_closed27_share=args.logvol_closed27_share if closed else None,
               closed27_weights=args.closed27_weights if closed else None,
               metric_jd_scaling=args.metric_jd_scaling, jd_cell_estimate=args.jd_cell_estimate, structure=structure)
    domain = getattr(args, "sigma_domain", None) or BRAIN
    if domain != BRAIN:
        # C05-7a appended note (SWLI): the estimation domain and theta are part of the configuration; under brain the
        # record is that of the code before SWLI, key for key (existing sweep roots compare it on a restart)
        out.update(sigma_domain=domain, support_threshold=args.support_threshold)
    return out


def domain_switches(config):
    """provenance.json switches of C04-6 (sigma_domain, support_threshold, support_rule) -- only under support /
    support_shared, so that the brain record is the one before SWLI."""
    return {} if config.sigma_domain == BRAIN else rp.sigma_domain_switches(config)


def sigma2_used(state):
    """sigma^2 of the probe's round state as recorded: null for an empty-support channel (C04-6); brain as before."""
    return state.sigma2_record() if state.has_support else np.asarray(state.sigma2).tolist()


def _norm(vector):
    vector = np.asarray(vector, dtype=np.float64).ravel()
    return float(np.sqrt(np.sum(vector*vector, dtype=np.longdouble)))


def scale_probe(state, u1):
    """nu*, alpha* and their norms at u1 with `state` (a ModelState of the numpy backend; module docstring)."""
    point = evaluate(state, u1)
    if not hasattr(point, "energy"):
        raise FloatingPointError(f"F_B at u_1 is not defined ({point.reason})")
    data = _norm(gradient(state, point, blocks=("channel",)))
    unit_nu = state.replace(nu=1.0)
    curl_unit = gradient(unit_nu, evaluate(unit_nu, u1), blocks=("curl",))
    unit_alpha = state.replace(alpha=1.0)
    diffusion_unit = gradient(unit_alpha, evaluate(unit_alpha, u1), blocks=("diffusion",))
    curl, diffusion = _norm(curl_unit), _norm(diffusion_unit)
    # linearity check (record only): the state's own curl gradient is nu_tmp times the unit one
    curl_state = gradient(state, point, blocks=("curl",))
    scale = max(_norm(curl_state), np.finfo(float).tiny)
    linearity = _norm(curl_state-state.nu*curl_unit)/scale
    if not (data > 0 and curl > 0 and diffusion > 0):
        raise FloatingPointError(f"zero norm: data {data}, curl {curl}, diffusion {diffusion}")
    return dict(nu_star=data/curl, alpha_star=data/diffusion,
                norms=dict(data_gradient=data, curl_gradient_per_unit_nu=curl,
                           diffusion_gradient_per_unit_alpha=diffusion),
                definitions=dict(
                    nu_star="||sum_c K_c^T r_c|| / ||C^T W C u_1 / omega0|| (C05-7)",
                    alpha_star="||sum_c K_c^T r_c|| / ||K_Q u_1 / omega0|| (C05-7, P1 / P3 only; record)",
                    data_gradient="bayesreg.model.gradient(state, ., blocks=('channel',)) at u_1",
                    curl_gradient_per_unit_nu="gradient(state.replace(nu=1), ., blocks=('curl',)) at u_1",
                    diffusion_gradient_per_unit_alpha="gradient(state.replace(alpha=1), ., blocks=('diffusion',)) at u_1",
                    norm="Euclidean norm of the free vector (3 N_int,), longdouble sum of squares"),
                checks=dict(curl_linearity_relative=linearity, F_B_at_u1=point.energy))


def run(args, out, logger, timing):
    pair_path = rp.resolve_pair(args.data_root, args.pair)
    tick = time.perf_counter()
    try:
        pair = load_pair(pair_path, allowlist=args.allowlist)
    except SubjectSealed as exc:
        logger.error("subject_sealed: %s", exc)
        return dict(status="subject_sealed", reason=str(exc))
    timing["load_pair"] = time.perf_counter()-tick
    structure = None
    if args.channel_source == "structure":
        structure = rp.structure_source(args, command=("python", "-B", "experiments/eta/probe.py", "--pair", args.pair,
                                                       "--out", str(out)))
    rp._write_text(out/"data.txt", dumps(dict(pair=args.pair, pair_path=str(pair_path), pair_provenance=pair.provenance,
                                              allowlist=args.allowlist, tissue_root=args.tissue_root,
                                              channel_source=args.channel_source,
                                              structure=None if structure is None else structure.record()),
                                         indent=1)+"\n")
    tick = time.perf_counter()
    try:
        inputs, pt_f, mask_f, channel_prov = rp.prepare_channels(pair, args.channel_source, args.tissue_root, logger,
                                                                 structure=structure)
    except ProvenanceRejected as exc:
        logger.error("provenance_rejected: %s", exc)
        return dict(status="provenance_rejected", reason=str(exc))
    timing["channels_and_pyramid"] = time.perf_counter()-tick
    eta = eta_temporary(pt_f, mask_f, pair.grid)
    del pt_f, mask_f
    # the likelihood channels of the probe's F_B (ETA2: intensity + K_S structure groups for channel_source structure)
    channels = dict(model_channels=int(inputs[0].fixed.shape[0]), K_T=channel_prov.get("K_T"),
                    K_S=channel_prov.get("K_S"))
    config = SolverConfig(outer="a", rounds=ROUNDS, caps=PROBE_CAPS,
                          pg=rp.pg_config(args.metric_jd_scaling, args.jd_cell_estimate),
                          max_levels=1, cpu_threads=args.cpu_threads, logvol_points=args.logvol_points,
                          logvol_closed27_share=args.logvol_closed27_share, closed27_weights=args.closed27_weights,
                          **rp.sigma_domain_options(args))
    backend = make_backend("numpy")
    sigma = GaussianMStep()
    probe_config = dict(max_levels=1, caps=PROBE_CAPS.record(), backend="numpy (CPU FP64 reference)", outer="a",
                        eta="C05-7 temporary values eta_tmp (P2)", wall_seconds=None,
                        note="bayesreg.cli.run_pair unchanged; the probe calls its input functions and solve_multilevel")
    write_json(out/"config.json", dict(args=vars(args), probe=probe_config, configuration=configuration(args),
                                       solver=config.record(), eta=eta.record(),
                                       sigma=rp.sigma_policy_record(config, sigma), backend=backend.record(),
                                       seed="none (deterministic)"))
    write_json(out/"provenance.json", dict(code=rp.code_record(), jd_curl_snapshot=rp.jd_curl_snapshot(),
                                           pair=pair.provenance, channels=channel_prov, channel_source=args.channel_source,
                                           omega0=pair.grid.omega, native_grid=grid_record(pair.grid),
                                           solver=config.record(), backend=backend.record(), eta=eta.record(),
                                           probe=probe_config, env=rp.env_record(backend),
                                           switches=dict(logvol_points=config.logvol_points,
                                                         **rp.closed27_switches(config),
                                                         metric_jd_scaling=config.pg.metric_jd_scaling,
                                                         jd_cell_estimate=config.pg.jd_cell_estimate,
                                                         parametrization="P2", tau0_fixed=True,
                                                         **domain_switches(config))))
    jsonl = JsonlLog(out/"log.jsonl")
    tick = time.perf_counter()
    try:
        result = solve_multilevel(backend, inputs, eta=eta, sigma=sigma, config=config, log=jsonl)
    finally:
        jsonl.close()
    timing["solve_level0"] = time.perf_counter()-tick
    row = result.rounds[0] if result.rounds else None
    summary = dict(solver_status=result.status, solver_wording=RUN_WORDING[result.status], solver_reason=result.reason,
                   eta_tmp=eta.record(), level0_grid=grid_record(inputs[0].grid), native=False, **channels)
    if row is not None:
        summary["level0_round"] = {k: row.get(k) for k in ("status", "reason", "inner_status", "accepted", "commits",
                                                           "G0", "T_B", "GF_B", "F_B_start", "F_B_end", "sigma2_used",
                                                           "state_hash", "min_jd_lower_bound", "seconds")}
    if args.jd_cell_estimate == CERTIFICATE_BERNSTEIN:
        # C09.1-3a check (b): the run total, as run_pair's status.json (JDTI review minor 6)
        summary["bernstein_j53_violations_total"] = j53_violations_total(result.rounds)
    if result.commit is None or row is None or row.get("accepted") != 1 or row.get("commits") != 1:
        reason = (f"level 0 made {None if row is None else row.get('accepted')} accepted steps (1 required); solver "
                  f"{result.status}: {result.reason}")
        logger.error("probe_failed: %s", reason)
        return dict(summary, status=PROBE_FAILED, reason=reason)
    u1 = result.commit.u
    digest = write_final_state(out/"final_level0.npz", u1, inputs[0].grid, status=result.status)
    tick = time.perf_counter()
    probe = scale_probe(result.state, u1)
    timing["probe"] = time.perf_counter()-tick
    document = dict(schema=SCHEMA, contract="C05-7 尺度探针 (appendix 7a, pending review)", pair=args.pair,
                    channel_source=args.channel_source, logvol_points=args.logvol_points,
                    metric_jd_scaling=args.metric_jd_scaling, jd_cell_estimate=args.jd_cell_estimate,
                    configuration=configuration(args), u1_state_hash=digest,
                    sigma2_used=sigma2_used(result.state), omega0=pair.grid.omega, **probe, **summary)
    write_json(out/"probe.json", document)
    logger.info("nu* = %.6g (data %.6g / curl %.6g), alpha* = %.6g; nu_tmp %.6g", probe["nu_star"],
                probe["norms"]["data_gradient"], probe["norms"]["curl_gradient_per_unit_nu"], probe["alpha_star"],
                eta.nu)
    return dict(summary, status=PROBE_DONE, reason=f"nu* = {probe['nu_star']:.6g}", nu_star=probe["nu_star"],
                alpha_star=probe["alpha_star"], u1_state_hash=digest)


def main(argv=None):
    args = parse_args(argv)
    out = rp._prepare_run_dir(args.out)
    logger = rp._run_logger(out)
    started = time.time()
    timing = dict(started_unix=started)
    argv_text = sys.argv if argv is None else ["python", "-B", "experiments/eta/probe.py", *argv]
    rp._write_text(out/"cmd.txt", " ".join(shlex.quote(a) for a in argv_text)+f"\n# cwd {os.getcwd()}\n"
                   f"# python {sys.executable}\n")
    code = rp.code_record()
    rp._write_text(out/"git.txt", f"HEAD {code['head']}\ndirty {code['dirty']}\n\n# git status --porcelain\n"
                   f"{code['status']}\n# git diff HEAD\n{rp._git('diff', 'HEAD')}")
    rp._write_text(out/"env.txt", dumps(rp.env_record(), indent=1)+"\n")
    try:
        summary = run(args, out, logger, timing)
    except Exception as exc:                                      # evidence kept, never retried (constraint 4)
        logger.error("exception: %s\n%s", exc, traceback.format_exc())
        summary = dict(status="exception", reason=f"{type(exc).__name__}: {exc}")
    code_out = EXIT[summary["status"]]
    timing["total_seconds"] = time.time()-started
    resources = rp.resource_record(args.cpu_threads)
    write_json(out/"timing.json", dict(timing, resources=resources))
    write_json(out/"status.json", dict(summary, exit_code=code_out, resources=resources, configuration=configuration(args),
                                       structure_registry=rp.structure_registry_label(args)))
    logger.info("probe status %s, exit %d", summary["status"], code_out)
    rp._close_logger(logger)
    return code_out


if __name__ == "__main__":
    sys.exit(main())
