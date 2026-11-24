"""Population empirical Bayes over the B-line EB training pairs (contract C15; parameterization C05-6).

    python -m bayesreg.learn.cli --out /home/rmcfz/seq_vp_tag_results/eb_v1/<run_id> --pairs-root DIR \\
        --eta0-json F [--pairs all | <fixed>__<moving>,...] [--parametrization P2] [--learn-tau0]
        [--max-rounds 3] [--rel-tol .05] [--n-samples 16] [--sigma-prior moment_match|pseudo_count|none]
        [--eb-count gull_mackay|dim]
        [--logvol-points gauss27+closed27|gauss27] [--logvol-closed27-share BETA_S] [--closed27-weights simpson]
        [--jd-cell-estimate certificate|gauss27+corners|closed27]
        [--sigma-domain brain|support|support_shared] [--support-threshold THETA]
        [--kappa0 K] [--outer a|b] [--rounds 1,1,1,3] [--max-accepted 500] [--max-outer 60]
        [--eb-pairs splits/bayesreg_v1/eb_pairs.tsv] [--folds splits/bayesreg_v1/folds.json]
        [--allowlist configs/bayesreg/dev_subjects.json] [--registry configs/bayesreg/segmenters.json]
        [--structure-registry-staging] [--tissue-root DIR] [--structure-root DIR] [--run-id ID] [--dry-run]
        [--synthetic]

Order (nothing is overwritten, failures are kept, hard constraint 4): the run directory (must not exist or be
empty) gets cmd.txt, git.txt, env.txt; then the leak guard before any image is read (`admit_eb_pairs`: only rows
of eb_pairs.tsv, same fold in folds.json, both subjects in the allowlist source bayesreg_train_v1) and the
segmenter registry; data.txt and config.json; with --dry-run only the presence of every input file is checked
(plan.json, status dry_run, exit 0).  Otherwise every round and pair: load_pair (subject guard on the allowlist)
-> tissue layer P_T (Atropos, as run_pair) -> structure layer P_S of both images from --structure-root/<subject>/
(structure_groups.npz + provenance.json written by bayesreg.seg.unet.infer; provenance validated against the
registry with the pair-level label-contact check, file hashes verified, and `check_structure_models`: only the
OTHER fold's models, D24) -> C03.2-4a (D39), C03.5, C03.6 -> `multilevel_map` (four levels, Gaussian EM with
a = b = 0) -> C14 with S = 16 -> statistics; then the C15 update.  The pair data are expected at
--pairs-root/<fixed>__<moving>/{pair.npz, manifest.json} (the EB pairs have no prepared stage yet).

Segmenter registry (review PS1_PS2 item c; the same rule as run_pair, bayesreg.seg.structure_layer.open_registry):
--registry defaults to the repository configs/bayesreg/segmenters.json (registered). Any other registry, e.g. the
staging copy of the training chain (the default path once --structure-registry-staging is given), is refused unless
--structure-registry-staging is given; it is then recorded as staging_unregistered (未正式登记, development
diagnostics only) and ensemble.json beside it must carry the registry file's sha256 and member_check.passed true.
The registry record (with the registry file's git state) is in data.txt; status.json structure_registry carries the
label (status, path, staging_opt_in, development_diagnostic_only, git tracked / clean). A dry run without a
registry file reports registry_present false; with a registry file it applies the same rule.

--eb-count (contract C15-9, D45; default gull_mackay with P2, adopted by BZ 2026-10-30): gull_mackay uses the traces
of the joint prior precision Pi_0 as effective counts (16 extra Pi_0 PCG solves per pair and round; P2 only for now:
P1 is refused, and P3 is refused until a P3 gate and an alpha lower bound exist, C15-9 / review R2 important 1); its
5% rule is reported as "步长判据满足" (a step criterion, not convergence).  dim (C15-1, bitwise the code before C15-9)
is the explicit option, and --parametrization P1 / P3 need --eb-count dim: with gull_mackay (given or by default)
they are refused when the arguments are parsed (usage error, exit 2, no run directory).

Model and path switches of every pair's MAP, its Laplace samples and the prior side (contract C20: every pending item
is a switch; work package EBX after review DEF important 1).  The same names, defaults and value checks as
bayesreg.cli.run_pair (the defaults since DEF, 2026-10-31):
  --logvol-points          gauss27+closed27 (default; C05-2b, D43) | gauss27 (C05-2, the default before DEF).
                           gauss27+corners (C05-2a) is refused here: C14 / C15 do not cover it (no corner noise for the
                           Laplace samples, no tau moment update; C14-7, C15-10);
  --logvol-closed27-share  beta_S > 0 (default 1), only with gauss27+closed27 (as run_pair);
  --closed27-weights       simpson (default, D48), only with gauss27+closed27; uniform is refused here (scope limit of
                           C14-7 / C15-10, not a mathematical obstacle; an ablation switch of run_pair only);
  --jd-cell-estimate       certificate (default, D47) | gauss27+corners (the default before DEF) | closed27
                           (C09.1-3a, path only).
The EB entry point before DEF ran gauss27 with the cell estimate gauss27+corners (SolverConfig(pg=PGConfig()) of the
code before DEF): only both old values given together, --logvol-points gauss27 --jd-cell-estimate gauss27+corners, give
that model and path (tests/test_bayesreg/learn/test_cli_switches.py); either one alone runs with the other switch's
new default, which is not the code before DEF (MIN1).
The PG metric is affine_v1 here (no --metric-jd-scaling; run_pair's resolve rule with the metric on).  Every refusal is
a usage error when the arguments are parsed (exit 2, no run directory).  config.json `switches` records the values in
effect (logvol_closed27_share and closed27_weights null unless gauss27+closed27, as run_pair records them; the MAP
records carry the full solver record), so that reading a record does not depend on the code's defaults.
sigma_domain (C04-6, C11-8, C15-5 appended notes; work package SWLI; draft SWL, D49 / D50 for BZ): --sigma-domain
brain (default; bitwise the code before SWLI) | support | support_shared and --support-threshold THETA, run_pair's names
and rules (an explicit THETA with brain refused; THETA finite in (0, 1) and above the K-independent floor bound here, the
exact bound with K_S in the solver).  Every training pair's MAP and Laplace use them (H contains the native level's v);
the sigma prior is matched on the same domain (population.sigma_prior_support: per structure channel on the pairs with a
non-empty native support, or one (a_S, b_S); D29 counts N_c / N_S) and records sigma_domain and theta.  config.json
`sigma_domain` (sigma_domain_record) carries sigma_domain, support_threshold, support_rule; `switches` is unchanged.

eta0 (--eta0-json): {"nu": .., "tau": [tau_0, .., tau_K_T]} plus "alpha" for P1/P3; under P2 alpha is
alpha_nug(tau0) (C05-6) and a given "alpha" must equal it.  The C05-7 sweep that produces eta0 is not part of
this entry point.

Files: cmd.txt, git.txt, env.txt, data.txt, config.json, log.txt, log.jsonl (one row per pair and round),
pairs/round_<r>/<pair>.json (MAP record, posterior qc, statistics) and <pair>.log.jsonl (solver log),
rounds/round_<r>.json (eta in/out, numerators/denominators, relative changes), eb_result.json (status,
trajectory, sigma prior), timing.json, status.json.  u_hat is not stored (about 165 MB per pair and round at the
native size); its state_hash and certificate are in the MAP record.
Real runs require eb_pairs.tsv and folds.json to be the committed split files (sha256 pins); --synthetic drops
the pins and admits only synthetic ids (OASIS_OAS1_09xx_MR1).
Exit codes: 0 eb_criterion_met / eb_round_cap / dry_run, 2 eb_failed, 3 subject_sealed, 4 provenance_rejected,
5 eb_pair_refused, 6 pinned_input_refused, 1 exception.
"""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import resource
import shlex
import sys
import time
import traceback

from jd_curl.validation import InputValidationError

from ..cli.run_pair import (_positive_share, _support_threshold, closed27_switches, code_record, env_record,
                            jd_curl_snapshot, load_tissue_layer, pg_config, resolve_jd_cell_estimate,
                            sigma_domain_options, sigma_domain_switches)
from ..model.support import BRAIN, DEFAULT_SUPPORT_THRESHOLD, SIGMA_DOMAINS, check_theta
from ..io import (ProvenanceRejected, SubjectSealed, load_dev_allowlist, load_pair, validate_segmentation_provenance,
                  verify_segmentation_files)
from ..model import (CLOSED27_SIMPSON, CLOSED27_UNIFORM, CLOSED27_WEIGHTS, LOGVOL_DEFAULT, LOGVOL_GAUSS27_CLOSED27,
                     LOGVOL_GAUSS27_CORNERS, LOGVOL_POINTS)
from ..seg import (brain_mask, channel_pyramid, normalize_intensity, segmentation_qc, simplex_with_background,
                   tissue_channels)
from ..seg.channels import read_back_probabilities, simplex_from_background_class
from ..seg.structure_layer import open_registry, registry_label, registry_status
from ..seg.unet.data import load_folds
from ..solve import Caps, Eta, NumpyBackend, ProximalParams, SolverConfig, default_lambda, level_inputs
from ..solve.export import JsonlLog, dumps, write_json
from ..solve.jd_scaling import AFFINE_V1, CELL_ESTIMATES as JD_CELL_ESTIMATES
from .population import (DEFAULT_EB_COUNT, EB_COUNTS, EB_FAILED, PARAMETRIZATIONS, SIGMA_PRIOR_RULES, TRAINING_SOURCE,
                         EBConfig, EBPairRefused, alpha_nugget, admit_eb_pairs, check_structure_models, multilevel_map,
                         run_population_eb)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_EB_PAIRS = REPO/"splits"/"bayesreg_v1"/"eb_pairs.tsv"
DEFAULT_FOLDS = REPO/"splits"/"bayesreg_v1"/"folds.json"
DEFAULT_ALLOWLIST = REPO/"configs"/"bayesreg"/"dev_subjects.json"
DEFAULT_REGISTRY = REPO/"configs"/"bayesreg"/"segmenters.json"
DEFAULT_STAGING_REGISTRY = Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v1/segmenters.json")  # = run_pair's
DEFAULT_TISSUE_ROOT = Path("/home/rmcfz/seq_vp_tag_results/seg/atropos_v1")
DEFAULT_STRUCTURE_ROOT = Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v1/pred")
# Pinned inputs of real runs (splits/bayesreg_v1/SHA256SUMS; docs/bayesreg/06_数据划分.md; the folds pin equals
# bayesreg.seg.unet.pins.PINNED_FOLDS_SHA256).  --synthetic (dry runs, tests) drops the pins but then admits only
# synthetic ids OASIS_OAS1_09xx_MR1 (real OASIS ids end at 0457).
PINNED_EB_PAIRS_SHA256 = "65431f6e5ef61062224d09df0ea61eccf898635c172470cef7cba47deab62b22"
PINNED_FOLDS_SHA256 = "98e9e676053d4e9386bc775c5d4cfc6be097afca9919c655ccb554d6236bf245"
SYNTHETIC_MIN = 900
STRUCTURE_GROUPS_FILE = "structure_groups.npz"   # = bayesreg.seg.unet.infer.GROUPS_FILE (that module imports torch)
STRUCTURE_PROVENANCE_FILE = "provenance.json"
EXIT = {EB_FAILED: 2, "subject_sealed": 3, "provenance_rejected": 4, "eb_pair_refused": 5,
        "pinned_input_refused": 6}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m bayesreg.learn.cli", description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, help="run directory (must not exist or be empty); "
                        "/home/rmcfz/seq_vp_tag_results/eb_v1/<run_id> for real runs")
    parser.add_argument("--pairs-root", required=True, help="directory holding <fixed>__<moving>/pair.npz")
    parser.add_argument("--eta0-json", required=True, help='{"nu": .., "tau": [tau_0, ..], "alpha": .. (P1/P3)}')
    parser.add_argument("--pairs", default="all", help="all, or comma-separated <fixed>__<moving> rows of eb_pairs.tsv")
    parser.add_argument("--run-id", default=None, help="seed source (C14-4); default: the --out directory name")
    parser.add_argument("--parametrization", choices=PARAMETRIZATIONS, default="P2")
    parser.add_argument("--learn-tau0", action="store_true", help="P2 only: learn tau_0 too (D28 option)")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--rel-tol", type=float, default=.05)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--sigma-prior", choices=SIGMA_PRIOR_RULES, default="moment_match")
    parser.add_argument("--kappa0", type=float, default=None, help="pseudo_count only (D29; BZ decides)")
    parser.add_argument("--eb-count", choices=EB_COUNTS, default=DEFAULT_EB_COUNT,
                        help="C15-9 effective counts: gull_mackay (default with P2; D45, BZ 10-30) or dim (C15-1, "
                        "bitwise the code before C15-9). gull_mackay runs with P2 only: P1 is refused, and P3 is "
                        "refused until a P3 gate and an alpha lower bound exist (C15-9, review R2 important 1); "
                        "P1 and P3 need --eb-count dim")
    model = parser.add_argument_group("model and path switches of every pair (as run_pair; defaults since DEF)")
    model.add_argument("--logvol-points", choices=LOGVOL_POINTS, default=LOGVOL_DEFAULT,
                       help="point set of the log-volume prior (changes F_B, H and Pi_0): gauss27+closed27 (default "
                       "since DEF, C05-2b) or gauss27 (C05-2, the default before DEF; the code before DEF only "
                       "together with --jd-cell-estimate gauss27+corners); gauss27+corners (C05-2a) is refused here: "
                       "C14 / C15 do not cover it (C14-7, C15-10)")
    model.add_argument("--logvol-closed27-share", dest="logvol_closed27_share", type=_positive_share, default=1.,
                       help="C05-2b share beta_S > 0 (default 1); only with --logvol-points gauss27+closed27")
    model.add_argument("--closed27-weights", dest="closed27_weights", choices=CLOSED27_WEIGHTS,
                       default=CLOSED27_SIMPSON,
                       help="C05-2b / D48 closed27 rule: simpson (default); uniform is refused here (scope limit of "
                       "C14-7 / C15-10); only with --logvol-points gauss27+closed27")
    model.add_argument("--jd-cell-estimate", choices=JD_CELL_ESTIMATES, default=None,
                       help="C09.1-3a cell estimate m_K of the JD scaling of every pair's PG (path only, D47): "
                       "certificate (default since DEF), gauss27+corners (the default before DEF; the code before DEF "
                       "only together with --logvol-points gauss27), closed27 or certificate_bernstein (D53, not the "
                       "default; every MAP record carries bernstein_j53_violations_total)")
    model.add_argument("--sigma-domain", choices=SIGMA_DOMAINS, default=None,
                       help="C04-6 / C11-8 / C15-5 (draft SWL; D49 for BZ): the estimation domain of the structure-"
                       "channel noise variances of every pair's MAP and Laplace (as run_pair): brain (default), "
                       "support or support_shared; the sigma prior is matched on the same domain")
    model.add_argument("--support-threshold", type=_support_threshold, default=None,
                       help="C04-6 / D50 theta (default .05 with support / support_shared; refused with brain); "
                       "checked here against the K-independent floor bound eps/(1 + 2 eps), exactly (K_S) by the solver")
    parser.add_argument("--outer", choices=("a", "b"), default="a")
    parser.add_argument("--rounds", default="1,1,1,3", help="EM rounds per level (C11-3)")
    parser.add_argument("--max-accepted", type=int, default=500)
    parser.add_argument("--max-outer", type=int, default=60)
    parser.add_argument("--eb-pairs", default=str(DEFAULT_EB_PAIRS))
    parser.add_argument("--folds", default=str(DEFAULT_FOLDS))
    parser.add_argument("--allowlist", default=str(DEFAULT_ALLOWLIST))
    parser.add_argument("--registry", default=None,
                        help=f"segmenters.json: default the repository {DEFAULT_REGISTRY} (registered); with "
                        f"--structure-registry-staging the staging copy {DEFAULT_STAGING_REGISTRY}")
    parser.add_argument("--structure-registry-staging", action="store_true",
                        help="admit a registry other than the repository one (the staging copy of the training chain, "
                        "未正式登记): development diagnostics only, recorded as staging_unregistered; ensemble.json "
                        "beside it must carry the registry's sha256 and member_check.passed true. Default: refused")
    parser.add_argument("--tissue-root", default=str(DEFAULT_TISSUE_ROOT))
    parser.add_argument("--structure-root", default=str(DEFAULT_STRUCTURE_ROOT))
    parser.add_argument("--dry-run", action="store_true", help="guard + file presence only; no image is read")
    parser.add_argument("--synthetic", action="store_true", help="tests / dry runs on synthetic ids (>= 0900): the "
                        "eb_pairs.tsv and folds.json pins are not required")
    args = parser.parse_args(argv)
    if args.eb_count == "gull_mackay" and args.parametrization != "P2":
        parser.error(f"--parametrization {args.parametrization} needs --eb-count dim: eb_count gull_mackay (the "
                     "default since BZ 10-30; C15-9 / D45) runs with P2 only (P1 refused; P3 refused until a P3 gate "
                     "and an alpha lower bound exist, review R2 important 1)")
    resolve_switches(parser, args)
    if args.registry is None:
        args.registry = str(DEFAULT_STAGING_REGISTRY if args.structure_registry_staging else DEFAULT_REGISTRY)
    return args


def structure_registry_label(args):
    """status.json structure_registry: the registration label of --registry (as run_pair's)."""
    return registry_label(dict(registry_status(args.registry), staging_opt_in=args.structure_registry_staging),
                          path=args.registry)


def eb_config(args):
    learn = True if args.learn_tau0 else None
    return EBConfig(parametrization=args.parametrization, learn_tau0=learn, max_rounds=args.max_rounds,
                    rel_tol=args.rel_tol, n_samples=args.n_samples, sigma_prior=args.sigma_prior, kappa0=args.kappa0,
                    eb_count=args.eb_count)


def read_eta0(path, config):
    """eta0 from JSON (tau as standard deviations, tau_0 first); P2: alpha = alpha_nug(tau0) (C05-6)."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) - {"alpha", "nu", "tau", "comment"} or "nu" not in document \
            or "tau" not in document:
        raise InputValidationError('--eta0-json: {"nu", "tau", optional "alpha", "comment"} required')
    tau = document["tau"]
    if not isinstance(tau, list) or len(tau) < 2:
        raise InputValidationError("--eta0-json: tau = [tau_0, ..., tau_K_T] (K_T >= 1)")
    tau2 = tuple(float(t)**2 for t in tau)
    nug = alpha_nugget(tau2, config.alpha_nug_factor)
    if config.parametrization == "P2":
        alpha = document.get("alpha", nug)
        if abs(float(alpha)-nug) > 1e-12*nug:
            raise InputValidationError(f"--eta0-json: P2 needs alpha = alpha_nug(tau0) = {nug} (C05-6), got {alpha}")
        alpha = nug
    else:
        if "alpha" not in document:
            raise InputValidationError(f"--eta0-json: {config.parametrization} needs alpha")
        alpha = float(document["alpha"])
    return Eta(alpha, float(document["nu"]), tau2, source=f"eta0 file {path}")


def pin_problem(args, admission, pairs):
    """Real runs: eb_pairs.tsv and folds.json must be the committed split files (sha256 pins).  --synthetic: no
    pins, but every admitted subject must be a synthetic id.  Returns None or the reason."""
    if args.synthetic:
        real = sorted(s for p in pairs for s in (p.fixed, p.moving) if int(s.split("_")[2]) < SYNTHETIC_MIN)
        return f"--synthetic admits only synthetic ids (>= {SYNTHETIC_MIN}); got {real}" if real else None
    if admission["eb_pairs_sha256"] != PINNED_EB_PAIRS_SHA256:
        return f"eb_pairs.tsv sha256 {admission['eb_pairs_sha256']} is not the pinned {PINNED_EB_PAIRS_SHA256}"
    if admission["folds_sha256"] != PINNED_FOLDS_SHA256:
        return f"folds.json sha256 {admission['folds_sha256']} is not the pinned {PINNED_FOLDS_SHA256}"
    return None


def resolve_switches(parser, args):
    """The model and path switches of the EB pairs (EBX; run_pair's value checks, then the EB scope of C14 / C15):
    --jd-cell-estimate resolved by run_pair's rule with the metric on (affine_v1: the given value or the default
    certificate); --logvol-closed27-share / --closed27-weights only with gauss27+closed27 (run_pair's rule); refused
    for EB: gauss27+corners (C14 / C15 do not cover C05-2a) and the uniform closed27 rule (scope limit of C14-7 /
    C15-10).  parser.error: exit 2 before any run directory."""
    args.jd_cell_estimate = resolve_jd_cell_estimate(parser, AFFINE_V1, args.jd_cell_estimate)
    if args.logvol_points != LOGVOL_GAUSS27_CLOSED27 and (args.logvol_closed27_share != 1.
                                                          or args.closed27_weights != CLOSED27_SIMPSON):
        parser.error("--logvol-closed27-share / --closed27-weights: only with --logvol-points gauss27+closed27 "
                     "(C05-2b)")
    if args.logvol_points == LOGVOL_GAUSS27_CORNERS:
        parser.error("--logvol-points gauss27+corners: not covered by C14 / C15 (the C05-2a corner term has no Laplace "
                     "noise and no tau moment update; C14-7, C15-10); EB runs gauss27+closed27 (default) or gauss27")
    if args.closed27_weights == CLOSED27_UNIFORM:
        parser.error("--closed27-weights uniform: refused for Laplace and EB (scope limit of C14-7 / C15-10: their "
                     "premises were checked for the Simpson weights only; uniform is an ablation switch of run_pair)")
    # C04-6 / C15-5 (SWLI): --sigma-domain / --support-threshold with run_pair's rules; the EB pairs always have the
    # structure layer, so only the K-independent floor bound is checked here (the solver checks K_S exactly)
    domain = getattr(args, "sigma_domain", None) or BRAIN
    theta = getattr(args, "support_threshold", None)
    if domain == BRAIN:
        if theta is not None:
            parser.error(f"--support-threshold {theta}: not with --sigma-domain brain (C04-6; recorded as null)")
    else:
        try:
            theta = check_theta(DEFAULT_SUPPORT_THRESHOLD if theta is None else theta, 1)
        except InputValidationError as exc:
            parser.error(f"--support-threshold: {exc}")
    args.sigma_domain, args.support_threshold = domain, theta


def solver_options(args):
    """The SolverConfig options of the model and path switches (shared by solver_config and solver_switches, so the
    record cannot drift from the run): the PG metric affine_v1 with the resolved jd_cell_estimate, logvol_points
    with beta_S and the closed27 rule (their defaults unless gauss27+closed27, as run_pair passes them), and the
    sigma_domain with theta (C04-6; nothing under brain, so the SolverConfig is the one before SWLI)."""
    return dict(pg=pg_config(AFFINE_V1, args.jd_cell_estimate), logvol_points=args.logvol_points,
                logvol_closed27_share=args.logvol_closed27_share, closed27_weights=args.closed27_weights,
                **sigma_domain_options(args))


def solver_switches(args):
    """config.json `switches`: the model and path switches of solver_config (they do not depend on the grid) in effect
    -- logvol_points with beta_S and the closed27 rule (null unless gauss27+closed27, as run_pair records them),
    metric_jd_scaling and jd_cell_estimate."""
    config = SolverConfig(**solver_options(args))          # the switches of solver_config(args, grid)
    return dict(outer=args.outer, logvol_points=config.logvol_points, **closed27_switches(config),
                metric_jd_scaling=config.pg.metric_jd_scaling, jd_cell_estimate=config.pg.jd_cell_estimate)


def sigma_domain_record(args):
    """config.json `sigma_domain` (C04-6 "记录", C15-5 appended note; SWLI): sigma_domain, support_threshold and
    support_rule in effect (brain: theta and rule null).  Kept out of `switches`, whose content is that of the code
    before SWLI (learn/test_cli_switches.py); every MAP record's solver record carries the same three keys."""
    return sigma_domain_switches(SolverConfig(**solver_options(args)))


def solver_config(args, grid):
    rounds = tuple(int(r) for r in args.rounds.split(","))
    proximal = ProximalParams(lam=default_lambda(grid)) if args.outer == "b" else None
    return SolverConfig(outer=args.outer, rounds=rounds, caps=Caps(args.max_accepted, args.max_outer),
                        proximal=proximal, **solver_options(args))


# ---- structure layer of an EB pair ------------------------------------------------------------------------

def load_structure_layer(pair_input, side, structure_root, registry, allowlist):
    """C03.1-2: P_S of one image (K_S + 1 channels, channel 0 = background) from the registered fold models:
    provenance validated (layer structure, registry, pair-level label contact), output hashes verified, grid
    equal to the pair's, read back by seg.channels.read_back_probabilities, the read-back shared with
    seg.unet.infer.read_probabilities and seg.structure_layer.read_structure_groups (C03.2-4a, D39: float64, raw
    channel sum checked on every voxel before the division by it -- 1e-3 for float16 storage, 1e-6 for float32 /
    float64; its refusals are ProvenanceRejected here)."""
    subject = pair_input.fixed_subject if side == "fixed" else pair_input.moving_subject
    directory = Path(structure_root)/subject
    path = directory/STRUCTURE_PROVENANCE_FILE
    if not path.is_file():
        raise ProvenanceRejected(f"{side}: no structure-layer provenance at {path}")
    prov = json.loads(path.read_text(encoding="utf-8"))
    accepted = validate_segmentation_provenance(prov, pair_input, side=side, layer="structure", registry=registry,
                                                allowlist=allowlist)
    verified = verify_segmentation_files(prov, directory)
    if STRUCTURE_GROUPS_FILE not in verified:
        raise ProvenanceRejected(f"{side}: {STRUCTURE_GROUPS_FILE} is not among the hash-verified outputs")
    grid = prov.get("grid")
    meta = pair_input.grid_metadata
    if not isinstance(grid, dict) or [list(grid.get(k, [])) for k in ("shape_zyx", "spacing_xyz", "origin_xyz")] != \
            [list(meta[k]) for k in ("shape_zyx", "spacing_xyz", "origin_xyz")]:
        raise ProvenanceRejected(f"{side}: the structure layer is not on the pair's grid")
    try:
        array, read_back = read_back_probabilities(directory/STRUCTURE_GROUPS_FILE,
                                                   shape=(len(accepted["class_names"])+1,)+tuple(pair_input.grid.shape))
    except InputValidationError as exc:
        raise ProvenanceRejected(f"{side}: {exc}") from exc
    record = dict(accepted=dict(accepted), directory=str(directory),
                  provenance_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), verified_outputs=verified,
                  read_back=read_back)
    return array, accepted, record


def prepare_eb_channels(pair_input, eb_pair, *, tissue_root, structure_root, registry, allowlist, logger=None):
    """C03.2-C03.6 for one EB pair with channel_source structure (mirrors cli.run_pair.prepare_channels, whose
    structure path is not implemented yet): tissue layer P_T for tau^2, the mask and the normalization; the
    structure layer P_S of the other fold's models for the likelihood channels."""
    p_f, acc_f, rec_f = load_tissue_layer(pair_input, "fixed", tissue_root)
    p_m, acc_m, rec_m = load_tissue_layer(pair_input, "moving", tissue_root)
    if acc_f["class_names"] != acc_m["class_names"]:
        raise ProvenanceRejected("tissue layer: fixed and moving class_names differ")
    s_f, sacc_f, srec_f = load_structure_layer(pair_input, "fixed", structure_root, registry, allowlist)
    s_m, sacc_m, srec_m = load_structure_layer(pair_input, "moving", structure_root, registry, allowlist)
    if sacc_f["class_names"] != sacc_m["class_names"]:
        raise ProvenanceRejected("structure layer: fixed and moving class_names differ")
    other_fold = dict(fixed=check_structure_models(sacc_f, eb_pair, registry),
                      moving=check_structure_models(sacc_m, eb_pair, registry))
    pbar_f, simplex_f = simplex_with_background(p_f)
    pbar_m, simplex_m = simplex_with_background(p_m)
    del p_f, p_m
    mask_f, mask_prov_f = brain_mask(pbar_f)
    mask_m, mask_prov_m = brain_mask(pbar_m)
    fixed, norm_f = normalize_intensity(pair_input.fixed, pbar_f, mask_f, acc_f)
    moving, norm_m = normalize_intensity(pair_input.moving, pbar_m, mask_m, acc_m)
    qc = segmentation_qc(pbar_f, mask_f)
    pt_f, smooth_prov = tissue_channels(pbar_f)
    pt_m, _ = tissue_channels(pbar_m)
    del pbar_f, pbar_m
    sbar_f, sprov_f = simplex_from_background_class(s_f)
    sbar_m, sprov_m = simplex_from_background_class(s_m)
    del s_f, s_m
    ps_f, s_smooth = tissue_channels(sbar_f)
    ps_m, _ = tissue_channels(sbar_m)
    del sbar_f, sbar_m
    levels, pyramid = channel_pyramid(pair_input.grid, fixed=fixed, moving=moving, tissue_fixed=pt_f, tissue_moving=pt_m,
                                      structure_fixed=ps_f, structure_moving=ps_m, mask=mask_f)
    inputs = level_inputs(levels, "structure")
    if logger is not None:
        logger.info("%s channels: K_T = %d, K_S = %d, %d model channels, mask %d nodes", eb_pair.pair_id,
                    pt_f.shape[0]-1, ps_f.shape[0]-1, inputs[0].fixed.shape[0], int(mask_f.sum()))
    provenance = dict(tissue=dict(fixed=rec_f, moving=rec_m), structure=dict(fixed=srec_f, moving=srec_m),
                      other_fold_check=other_fold, simplex=dict(fixed=simplex_f, moving=simplex_m),
                      structure_simplex=dict(fixed=sprov_f, moving=sprov_m), mask=dict(fixed=mask_prov_f,
                      moving=mask_prov_m), normalization=dict(fixed=norm_f, moving=norm_m), qc=qc,
                      smoothing=dict(tissue=smooth_prov, structure=s_smooth), pyramid=pyramid, K_T=pt_f.shape[0]-1,
                      K_S=ps_f.shape[0]-1, channel_source="structure")
    return inputs, provenance


def input_files(pair, pairs_root, tissue_root, structure_root):
    """The files one EB pair needs (for --dry-run; nothing is opened)."""
    root = Path(pairs_root)/pair.pair_id
    files = [root/"pair.npz", root/"manifest.json"]
    for subject in (pair.fixed, pair.moving):
        files.append(Path(tissue_root)/subject/"provenance.json")
        files.append(Path(structure_root)/subject/STRUCTURE_PROVENANCE_FILE)
        files.append(Path(structure_root)/subject/STRUCTURE_GROUPS_FILE)
    return files


# ---- run directory ---------------------------------------------------------------------------------------

def _prepare_run_dir(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"{path}: the run directory exists and is not empty (nothing is overwritten)")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_text(path, text):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text, encoding="utf-8")


def _logger(out):
    logger = logging.getLogger(f"bayesreg.learn.cli.{out}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in (logging.FileHandler(out/"log.txt", encoding="utf-8"), logging.StreamHandler(sys.stderr)):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def _close(logger):
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def run(args, out, logger, timing):
    config = eb_config(args)
    allowlist = load_dev_allowlist(args.allowlist)
    split = load_folds(args.folds)
    requested = None if args.pairs == "all" else [p for p in args.pairs.split(",") if p]
    try:
        pairs, admission = admit_eb_pairs(requested, eb_pairs_path=args.eb_pairs, split=split, allowlist=allowlist,
                                          training_source=TRAINING_SOURCE)
    except EBPairRefused as exc:
        logger.error("eb_pair_refused: %s", exc)
        return dict(status="eb_pair_refused", reason=str(exc))
    problem = pin_problem(args, admission, pairs)
    if problem:
        logger.error("pinned_input_refused: %s", problem)
        return dict(status="pinned_input_refused", reason=problem)
    eta0 = read_eta0(args.eta0_json, config)
    registry_path = Path(args.registry)
    registry = registry_record = None
    if registry_path.is_file() or not args.dry_run:                  # same rule as run_pair (review PS1_PS2 item c)
        registry, registry_record = open_registry(registry_path, allow_staging=args.structure_registry_staging)
    run_id = args.run_id or out.name
    _write_text(out/"data.txt", dumps(dict(admission=admission, pairs=[p.record() for p in pairs],
                                           pairs_root=args.pairs_root, tissue_root=args.tissue_root,
                                           structure_root=args.structure_root, registry_path=str(registry_path),
                                           registry_sha256=None if registry is None else registry.sha256,
                                           structure_registry=registry_record,
                                           eta0_json=args.eta0_json,
                                           eta0_json_sha256=hashlib.sha256(Path(args.eta0_json).read_bytes())
                                           .hexdigest()), indent=1)+"\n")
    write_json(out/"config.json", dict(args=vars(args), eb=config.record(logvol_points=args.logvol_points),
                                       eta0=eta0.record(), run_id=run_id,
                                       switches=solver_switches(args), sigma_domain=sigma_domain_record(args),
                                       code=code_record(), jd_curl_snapshot=jd_curl_snapshot(),
                                       backend=NumpyBackend().record(), seeds="round_seed(run_id, round, pair_id): "
                                       "sha256-derived (C14-4)"))
    files = {p.pair_id: [str(f) for f in input_files(p, args.pairs_root, args.tissue_root, args.structure_root)]
             for p in pairs}
    missing = {pid: [f for f in fs if not Path(f).is_file()] for pid, fs in files.items()}
    missing = {pid: fs for pid, fs in missing.items() if fs}
    if args.dry_run:
        write_json(out/"plan.json", dict(pairs=[p.pair_id for p in pairs], files=files, missing=missing,
                                         registry_present=registry is not None,
                                         rounds_max=config.max_rounds, samples_per_pair=config.n_samples,
                                         work="per round: one multilevel MAP + S PCG solves + S + 1 certificates "
                                              "per pair" + (" + S prior PCG solves with Pi_0 (C15-9)"
                                                            if config.gull_mackay else "")))
        logger.info("dry run: %d pairs admitted, %d with missing inputs", len(pairs), len(missing))
        return dict(status="dry_run", pairs=len(pairs), missing=len(missing), registry_present=registry is not None)
    if missing:
        raise FileNotFoundError(f"inputs missing for {sorted(missing)} (see --dry-run)")
    backend = NumpyBackend()
    jsonl = JsonlLog(out/"log.jsonl")

    def solve_pair(pair, eta, *, round_index):
        directory = out/"pairs"/f"round_{round_index}"
        directory.mkdir(parents=True, exist_ok=True)
        pair_input = load_pair(Path(args.pairs_root)/pair.pair_id/"pair.npz", allowlist=allowlist)
        if (pair_input.fixed_subject, pair_input.moving_subject) != (pair.fixed, pair.moving):
            raise EBPairRefused(f"{pair.pair_id}: the manifest names {pair_input.fixed_subject}, "
                                f"{pair_input.moving_subject}")
        inputs, provenance = prepare_eb_channels(pair_input, pair, tissue_root=args.tissue_root,
                                                 structure_root=args.structure_root, registry=registry,
                                                 allowlist=allowlist, logger=logger)
        solver_log = JsonlLog(directory/f"{pair.pair_id}.log.jsonl")
        try:
            outcome = multilevel_map(inputs, eta, solver_config=solver_config(args, pair_input.grid), backend=backend,
                                     log=solver_log)
        finally:
            solver_log.close()
        outcome.record.update(pair_provenance=pair_input.provenance, channels=provenance)
        logger.info("round %d %s: %s (%s)", round_index, pair.pair_id, outcome.status, outcome.reason)
        return outcome

    try:
        result = run_population_eb(pairs, solve_pair, eta0=eta0, config=config, run_id=run_id, out_dir=out,
                                   backend=backend, log=jsonl, logvol_points=args.logvol_points,
                                   sigma_domain=args.sigma_domain, support_threshold=args.support_threshold)
    finally:
        jsonl.close()
    timing["certificate_seconds"] = backend.certificate_seconds
    timing["certificate_calls"] = backend.certificate_calls
    logger.info("EB %s (%s): eta %s", result.status, result.wording, result.eta.record())
    return dict(status=result.status, wording=result.wording, eta=result.eta.record(), rounds=len(result.rounds))


def main(argv=None):
    args = parse_args(argv)
    out = _prepare_run_dir(args.out)
    logger = _logger(out)
    started = time.time()
    timing = dict(started_unix=started)
    argv_text = sys.argv if argv is None else ["python", "-m", "bayesreg.learn.cli", *argv]
    _write_text(out/"cmd.txt", " ".join(shlex.quote(a) for a in argv_text)+f"\n# cwd {os.getcwd()}\n"
                f"# python {sys.executable}\n")
    code = code_record()
    _write_text(out/"git.txt", f"HEAD {code['head']}\ndirty {code['dirty']}\n\n# git status --porcelain\n{code['status']}")
    _write_text(out/"env.txt", dumps(env_record(), indent=1)+"\n")
    try:
        summary = run(args, out, logger, timing)
    except SubjectSealed as exc:
        logger.error("subject_sealed: %s", exc)
        summary = dict(status="subject_sealed", reason=str(exc))
    except ProvenanceRejected as exc:
        logger.error("provenance_rejected: %s", exc)
        summary = dict(status="provenance_rejected", reason=str(exc))
    except EBPairRefused as exc:
        logger.error("eb_pair_refused: %s", exc)
        summary = dict(status="eb_pair_refused", reason=str(exc))
    except Exception as exc:                                      # evidence kept, never retried (constraint 4)
        logger.error("exception: %s\n%s", exc, traceback.format_exc())
        summary = dict(status="exception", reason=f"{type(exc).__name__}: {exc}")
    code_out = EXIT.get(summary["status"], 0 if summary["status"] in ("eb_criterion_met", "eb_round_cap", "dry_run")
                        else 1)
    timing["total_seconds"] = time.time()-started
    timing["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    write_json(out/"timing.json", timing)
    write_json(out/"status.json", dict(summary, structure_registry=structure_registry_label(args), exit_code=code_out))
    logger.info("status %s, exit %d", summary["status"], code_out)
    _close(logger)
    return code_out


if __name__ == "__main__":
    sys.exit(main())
