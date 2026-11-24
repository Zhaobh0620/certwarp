"""The model and path switches of the EB entry point (work package EBX, 2026-10-31; review DEF important 1; contract C20
last paragraph: every pending item is a switch, so that changing a default changes the configuration, not the code).

  --logvol-points / --logvol-closed27-share / --closed27-weights (C05-2b, D43, D48) and --jd-cell-estimate (C09.1-3a,
  D47) with run_pair's names, defaults and value checks; refused for EB (C14-7, C15-10): gauss27+corners and the
  uniform closed27 rule.

(a) defaults: the values of run_pair (gauss27+closed27, beta_S 1, simpson, certificate), the SolverConfig of the model
    defaults, config.json `switches` and args of a dry run;
(b) the old options given explicitly (--logvol-points gauss27 --jd-cell-estimate gauss27+corners) are the code before
    DEF bit for bit:
      * the SolverConfig equals the one run_pair builds for its pre_def arm (whose whole-run digest equals the code of
        8a4d060, solve/test_def_defaults.py, golden_8a4d060_run_pair.json) and the SolverConfig of the EB entry point
        before DEF (PGConfig() and logvol_points of 8a4d060: gauss27 with the cell estimate gauss27+corners);
      * the EB pair statistics (C14 samples -> posterior maps -> Gauss sums; _eb_digest's pair case) on a state built
        with the entry point's model options give the committed golden digest of the code before C15-9 / DEF
        (GOLDEN_7579645_EB["pair"], golden_7579645_eb_def.json);
(c) illegal values and combinations are usage errors (exit 2) before any run directory.
No data/, no GPU; the pair case scripts its certificates (a few seconds).
"""
import json

import numpy as np
import pytest

from bayesreg.cli import run_pair
from bayesreg.learn import EBConfig, pair_statistics
from bayesreg.learn import cli
from bayesreg.seg import structure_layer
from bayesreg.solve import Caps, JDScalingConfig, PGConfig, SolverConfig

from ..infer import _cases as cs
from ..solve import _run_pair_digest as RD
from ..solve import _synthetic as syn
from . import _eb_digest as D
from . import _guard_fixtures as gf
from .test_cli import argv, eta_file
from .test_eb_count import GOLDEN_7579645_EB

BASE = ["--out", "o", "--pairs-root", "p", "--eta0-json", "e"]
RUN_PAIR_BASE = ["--pair", "s/p", "--channel-source", "none", "--out", "x"]
CLOSED, CERT = "gauss27+closed27", "certificate"
OLD = ("--logvol-points", "gauss27", "--jd-cell-estimate", "gauss27+corners")
NEW_EXPLICIT = ("--logvol-points", CLOSED, "--logvol-closed27-share", "1", "--closed27-weights", "simpson",
                "--jd-cell-estimate", CERT)
DEFAULT_SWITCHES = dict(outer="a", logvol_points=CLOSED, logvol_closed27_share=1., closed27_weights="simpson",
                        metric_jd_scaling="affine_v1", jd_cell_estimate=CERT)
OLD_SWITCHES = dict(outer="a", logvol_points="gauss27", logvol_closed27_share=None, closed27_weights=None,
                    metric_jd_scaling="affine_v1", jd_cell_estimate="gauss27+corners")
NAMES = ("logvol_points", "logvol_closed27_share", "closed27_weights", "jd_cell_estimate")


def values(args):
    return tuple(getattr(args, name) for name in NAMES)


def run_pair_config(rp_args, rounds, caps):
    """The SolverConfig of run_pair's model / path options (run_pair.run builds it the same way; max_levels and
    cpu_threads at their defaults, as the EB entry point leaves them)."""
    return SolverConfig(rounds=rounds, caps=caps, pg=run_pair.pg_config(rp_args.metric_jd_scaling,
                                                                          rp_args.jd_cell_estimate),
                        logvol_points=rp_args.logvol_points, logvol_closed27_share=rp_args.logvol_closed27_share,
                        closed27_weights=rp_args.closed27_weights)


def test_defaults_are_run_pairs_and_recorded(tmp_path, monkeypatch):
    args = cli.parse_args(BASE)
    rp = run_pair.parse_args(RUN_PAIR_BASE)
    assert values(args) == values(rp) == (CLOSED, 1., "simpson", CERT)
    grid = syn.cube_grid(9)
    rounds, caps = (1, 1, 1, 3), Caps(500, 60)
    config = cli.solver_config(args, grid)
    assert config == SolverConfig(rounds=rounds, caps=caps, pg=PGConfig())          # the model defaults (DEF)
    assert config == run_pair_config(rp, rounds, caps)
    assert config.pg.jd == JDScalingConfig() and config.logvol_options() == dict(logvol_closed27_share=1.,
                                                                                 closed27_weights="simpson")
    assert cli.solver_switches(args) == DEFAULT_SWITCHES
    explicit = cli.parse_args(BASE+list(NEW_EXPLICIT))                             # the DEF defaults given explicitly
    assert values(explicit) == values(args) and cli.solver_config(explicit, grid) == config
    assert cli.solver_switches(explicit) == DEFAULT_SWITCHES
    paths = gf.write_files(tmp_path/"split")
    monkeypatch.setattr(structure_layer, "REPO_REGISTRY", paths["registry"])     # the repository registry of this test
    out = tmp_path/"eb_default"
    assert cli.main(argv(paths, out, "--dry-run", eta0=eta_file(tmp_path))) == 0
    record = json.loads((out/"config.json").read_text())
    assert record["switches"] == DEFAULT_SWITCHES
    assert tuple(record["args"][name] for name in NAMES) == (CLOSED, 1., "simpson", CERT)


def test_explicit_old_options_are_the_code_before_def(tmp_path, monkeypatch):
    args = cli.parse_args(BASE+list(OLD))
    assert values(args) == ("gauss27", 1., "simpson", "gauss27+corners")      # share / rule inert, recorded null
    grid = syn.cube_grid(9)
    rounds, caps = (1, 1, 1, 3), Caps(500, 60)
    config = cli.solver_config(args, grid)
    rp = run_pair.parse_args(RUN_PAIR_BASE+list(RD.ARM_ARGS["pre_def"]))           # golden-checked against 8a4d060
    assert config == run_pair_config(rp, rounds, caps)
    # the EB entry point before DEF: SolverConfig(pg=PGConfig()) with the defaults of 8a4d060
    assert config == SolverConfig(rounds=rounds, caps=caps, logvol_points="gauss27",
                                  pg=PGConfig(jd=JDScalingConfig(cell_estimate="gauss27+corners")))
    assert config.logvol_options() == {} and config.record()["pg"]["jd_cell_estimate"] == "gauss27+corners"
    assert cli.solver_switches(args) == OLD_SWITCHES
    paths = gf.write_files(tmp_path/"split")
    monkeypatch.setattr(structure_layer, "REPO_REGISTRY", paths["registry"])
    out = tmp_path/"eb_old"
    assert cli.main(argv(paths, out, "--dry-run", *OLD, eta0=eta_file(tmp_path))) == 0
    record = json.loads((out/"config.json").read_text())
    assert record["switches"] == OLD_SWITCHES
    assert (record["args"]["logvol_points"], record["args"]["jd_cell_estimate"]) == ("gauss27", "gauss27+corners")


def pair_digest(config):
    """_eb_digest.pair_case with the state's log-volume options taken from the entry point's SolverConfig (as
    solve_multilevel builds every level's ModelState: logvol_points plus config.logvol_options())."""
    grid, state = cs.cube6(factor=400.)
    state = state.replace(logvol_points=config.logvol_points, **config.logvol_options())
    u = cs.u_hat_of(grid)
    passes = [True]*16
    for k in (3, 9, 14):
        passes[k] = False
    backend = cs.ScriptedCertificate(passes)
    stats, record = pair_statistics(state, u, pair_id="digest_pair", config=EBConfig(n_samples=16, **D.DIM),
                                    seed=20261030, backend=backend, sigma_post=(1e-3, 2e-3, 3e-3))
    return D.digest([stats.record(), record]), record


@pytest.mark.skipif(np.__version__ != "2.0.2", reason="bitwise golden digests recorded with numpy 2.0.2")
def test_explicit_gauss27_pair_statistics_equal_the_committed_golden():
    grid = syn.cube_grid(9)
    old = cli.solver_config(cli.parse_args(BASE+list(OLD)), grid)
    digest, record = pair_digest(old)
    assert digest == GOLDEN_7579645_EB["pair"]                     # golden_7579645_eb_def.json (code before C15-9)
    assert "n_closed27" not in record
    new, record = pair_digest(cli.solver_config(cli.parse_args(BASE), grid))
    assert new != GOLDEN_7579645_EB["pair"] and record["n_closed27"] > 0        # the default acts (closed27 points)


@pytest.mark.parametrize("extra, message", [
    (["--logvol-points", "gauss27+corners"], "C14 / C15"),
    (["--logvol-points", "gauss27+corners", "--jd-cell-estimate", "gauss27+corners"], "C14 / C15"),
    (["--closed27-weights", "uniform"], "C14-7 / C15-10"),
    (["--logvol-points", CLOSED, "--closed27-weights", "uniform"], "C14-7 / C15-10"),
    (["--logvol-points", "gauss27", "--logvol-closed27-share", "2"], "only with --logvol-points gauss27+closed27"),
    (["--logvol-points", "gauss27", "--closed27-weights", "uniform"], "only with --logvol-points gauss27+closed27"),
    (["--logvol-closed27-share", "0"], "beta_S"),
    (["--logvol-closed27-share", "-1"], "beta_S"),
    (["--logvol-closed27-share", "nan"], "beta_S"),
    (["--logvol-closed27-share", "x"], "a number"),
    (["--closed27-weights", "gauss"], "invalid choice"),
    (["--logvol-points", "gauss28"], "invalid choice"),
    (["--jd-cell-estimate", "none"], "invalid choice"),
])
def test_illegal_switches_are_refused_before_the_run_directory(tmp_path, capsys, extra, message):
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(BASE+extra)
    assert exc.value.code == 2 and message in capsys.readouterr().err
    paths = gf.write_files(tmp_path/"split")
    out = tmp_path/"refused"
    with pytest.raises(SystemExit) as exc:
        cli.main(argv(paths, out, "--dry-run", *extra, eta0=eta_file(tmp_path)))
    assert exc.value.code == 2 and not out.exists()


def test_accepted_switch_values():
    grid = syn.cube_grid(9)
    args = cli.parse_args(BASE+["--logvol-closed27-share", "3", "--jd-cell-estimate", "closed27"])
    assert values(args) == (CLOSED, 3., "simpson", "closed27")
    config = cli.solver_config(args, grid)
    assert config.logvol_closed27_share == 3. and config.pg.jd == JDScalingConfig(cell_estimate="closed27")
    assert cli.solver_switches(args) == dict(DEFAULT_SWITCHES, logvol_closed27_share=3., jd_cell_estimate="closed27")
    args = cli.parse_args(BASE+["--logvol-points", "gauss27"])                    # cell estimate stays the default
    assert cli.solver_switches(args) == dict(OLD_SWITCHES, jd_cell_estimate=CERT)
    args = cli.parse_args(BASE+["--outer", "b", "--jd-cell-estimate", "gauss27+corners"])
    config = cli.solver_config(args, grid)
    assert config.outer == "b" and config.proximal is not None and config.pg.jd_cell_estimate == "gauss27+corners"
    assert cli.solver_switches(args) == dict(DEFAULT_SWITCHES, outer="b", jd_cell_estimate="gauss27+corners")
