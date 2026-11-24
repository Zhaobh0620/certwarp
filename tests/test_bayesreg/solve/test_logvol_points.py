"""C05-2a in the solver and the CLI (work package LVC).

(i) logvol_points = "gauss27" with jd_cell_estimate "gauss27+corners" (the defaults before DEF; explicit since DEF,
    2026-10-31: D.path_options("pre_def")) leaves the solve trajectory bitwise that of the code before C05-2a
    (commit 481e33b): golden digests (every candidate's tau, reason, energy and certificate bound, every accepted and
    committed state hash, the status and F_B trace) of 6 accepted steps on the mild 9^3 synthetic problem with the
    default PGConfig: numpy options (a) and (b), torch64 (cpu device) option (a).  Source (FB1):
    tests/test_bayesreg/_golden/golden_481e33b.json, generated on the code of 481e33b by
    `python tests/test_bayesreg/_golden/make_golden.py --ref 481e33b --out tests/test_bayesreg/_golden/golden_481e33b.json`
    (checked against the constants by model/test_logvol_corners.py::test_golden_constants_match_the_generation_record).
The corner option runs through solve_direct on the same problem (every accepted state certified, F_B decreasing,
a different trajectory); SolverConfig validates and records the option and passes it to every level's ModelState;
run_pair has --logvol-points.
"""
import numpy as np
import pytest
import torch

from jd_curl.validation import InputValidationError

from bayesreg.cli import run_pair
from bayesreg.solve import Eta, NumpyBackend, PGConfig, SolverConfig, make_backend
from bayesreg.solve import jd_scaling as jds
from bayesreg.solve.multilevel import _template

from ..model import _logvol_digest as D
from . import _synthetic as syn

CORNERS = "gauss27+corners"
GOLDEN_481E33B = dict(numpy_a="796fc35ec51ea98231e150dad525f1545c405a613a59e846826e28527abd716b",
                      numpy_b="188635287d49686d00b0b74e788762e3c99047cf1ac0dc6b4d5cdc90d5bbc1e6",
                      torch64_a="3e620f1070c26b75944c0ed0f6bb61c1dc254533d19edb45fa4b6d4866b89945")
# DEF (2026-10-31): digests of the defaults since DEF (gauss27+closed27, beta_S 1, Simpson; jd_cell_estimate
# certificate) GIVEN EXPLICITLY on the code before DEF (commit 8a4d060, where those options existed), numpy 2.0.2, torch
# 2.5.1, single-threaded.  Source: tests/test_bayesreg/_golden/golden_8a4d060_def_explicit.json, generated once by
#   python tests/test_bayesreg/_golden/make_golden.py --path def_explicit --ref 8a4d060 \
#       --out tests/test_bayesreg/_golden/golden_8a4d060_def_explicit.json
# The code's defaults (no option) must give them.
GOLDEN_DEF = dict(numpy_a="6ce277ffddf9f7f8456152cce888d462784b1790335832b9fa7f40bab1c23aa7",
                  numpy_b="3b62634ca0470db9c76025e3487d40be69e4e79d252e685078f88a14c6fa96c0",
                  torch64_a="bc01e1b2a3e84861149c265f63432207934a2a7d48560b3dc4edf6d823e15046")


@pytest.mark.skipif(np.__version__ != "2.0.2" or not torch.__version__.startswith("2.5.1"),
                    reason="bitwise golden digests recorded with numpy 2.0.2 and torch 2.5.1")
@pytest.mark.parametrize("key", sorted(GOLDEN_481E33B))
def test_default_trajectory_is_bitwise_the_pre_change_code(key):
    backend_name, option = key.split("_")
    backend = NumpyBackend() if backend_name == "numpy" else make_backend("torch64", device="cpu")
    model_options, pg = D.path_options("pre_def")
    assert model_options == dict(logvol_points="gauss27") and pg.jd.cell_estimate == "gauss27+corners"
    result, rows = D.solve_run(option, backend, 6, pg=pg, **model_options)
    assert result.status == "cap_reached" and result.accepted == 6
    assert D.trajectory_digest(rows, result) == GOLDEN_481E33B[key]


@pytest.mark.skipif(np.__version__ != "2.0.2" or not torch.__version__.startswith("2.5.1"),
                    reason="bitwise golden digests recorded with numpy 2.0.2 and torch 2.5.1")
@pytest.mark.parametrize("key", sorted(GOLDEN_DEF))
def test_def_default_trajectory_is_bitwise_the_explicit_options_before_def(key):
    """DEF: 6 accepted steps of the mild 9^3 problem with no option (gauss27+closed27, beta_S 1, Simpson; the default
    PGConfig: jd_cell_estimate certificate) give the trajectory digest of these options given explicitly on the code
    before DEF (8a4d060)."""
    jds.clear_cell_bounds_cache()                   # process-global per-state bounds (records only; not digested)
    _, pg = D.path_options("default")
    assert pg is None and PGConfig().jd_cell_estimate == "certificate"
    assert D.solve_key(key, "default") == GOLDEN_DEF[key]


def test_corner_option_solves():
    """solve_direct with the corner term: 6 accepted steps, every one certified with a certified decrease of the
    corner-including F_B; the trajectory differs from the default one."""
    backend = NumpyBackend()
    _, pg = D.path_options("pre_def")                   # DEF: the cell estimate of this test, explicit
    result, rows = D.solve_run("a", backend, 6, pg=pg, logvol_points=CORNERS)
    default, default_rows = D.solve_run("a", NumpyBackend(), 6, pg=pg, logvol_points="gauss27")
    assert result.status == "cap_reached" and result.accepted == 6
    accepted = [r for r in rows if r["kind"] == "candidate" and r["accepted"]]
    assert len(accepted) == 6 and all(r["cert_status"] == "certified_feasible" and r["certified"] > 0
                                      for r in accepted)
    assert all(b < a for a, b in zip(result.energies, result.energies[1:]))
    assert D.trajectory_digest(rows, result) != D.trajectory_digest(default_rows, default)
    torch_backend = make_backend("torch64", device="cpu")
    other, other_rows = D.solve_run("a", torch_backend, 6, pg=pg, logvol_points=CORNERS)
    decisions = [[(r["iteration"], r["attempt"], r["accepted"], r["reason"]) for r in table if r["kind"] == "candidate"]
                 for table in (rows, other_rows)]
    assert decisions[0] == decisions[1] and np.max(np.abs(other.commit.u-result.commit.u)) < 1e-6
    assert "logvol_corners" in torch_backend.record()["comparison"]["per_cell"]


def test_solver_config_template_and_cli():
    # DEF (2026-10-31, contract 8a4d060 C05-2b / D43 / D48): the default point set is gauss27+closed27; gauss27 explicit
    assert SolverConfig().logvol_points == "gauss27+closed27"
    assert SolverConfig().record()["logvol_points"] == "gauss27+closed27"
    assert SolverConfig(logvol_points="gauss27").record()["logvol_points"] == "gauss27"
    config = SolverConfig(logvol_points=CORNERS)
    assert config.record()["logvol_points"] == CORNERS
    with pytest.raises(InputValidationError):
        SolverConfig(logvol_points="corners")
    grid, level, _ = syn.single_level(n=7)
    eta = Eta(1e-3, 1e-3, (1., .25, .25, .25))
    host = _template(level, eta, grid.omega, 8192, 1, 1, config.logvol_points)
    assert host.logvol_points == CORNERS and host.replace(sigma2=np.array([.5])).logvol_corners
    assert _template(level, eta, grid.omega, 8192, 1).logvol_points == "gauss27+closed27"         # DEF default
    assert _template(level, eta, grid.omega, 8192, 1, 1, "gauss27").logvol_points == "gauss27"
    base = ["--pair", "s/p", "--channel-source", "none", "--out", "x"]
    assert run_pair.parse_args(base).logvol_points == "gauss27+closed27"                         # DEF default
    assert run_pair.parse_args(base+["--logvol-points", "gauss27"]).logvol_points == "gauss27"
    assert run_pair.parse_args(base+["--logvol-points", CORNERS]).logvol_points == CORNERS
    with pytest.raises(SystemExit):
        run_pair.parse_args(base+["--logvol-points", "corners8"])
