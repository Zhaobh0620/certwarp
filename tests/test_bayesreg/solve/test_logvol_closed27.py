"""C05-2b in the solver and the CLI (work package IMPL27).

- solve_direct with logvol_points = "gauss27+closed27" on the mild 9^3 synthetic problem: 6 accepted steps, each
  certified with a certified decrease, F_B decreasing, every evaluated candidate row carries the logvol_parts record;
  numpy and torch64 (cpu device) take the same decisions and nearly the same states; the trajectory differs from the
  default one (whose digests stay the pre-change goldens: test_logvol_points.py).
- SolverConfig: validation, record, the level template carries beta_S and the rule; run_pair: --logvol-points
  gauss27+closed27, --logvol-closed27-share, --closed27-weights (the contract names only: the IMPL27 aliases
  --logvol-closed27-beta, --logvol-closed27-weights are gone, IMPL27b), refused without gauss27+closed27;
  --witness-estimate (default off); provenance switches of beta_S and the rule are null unless gauss27+closed27.
DEF (2026-10-31; contract 8a4d060): gauss27+closed27 with beta_S 1 and Simpson is the default; the "default" of the
first test is the pre-DEF one, given explicitly (D.path_options("pre_def"): gauss27, jd_cell_estimate gauss27+corners,
the estimate of the closed27 solves here too); the refusals of beta_S / the rule are checked with an explicit
non-closed27 point set.
- run_pair end to end on the 26^3 synthetic pair with gauss27+closed27, beta_S = 3: config.json and provenance.json
  record the option, the hyper.json backend record the witness switch, timing.json the witness time, every log.jsonl
  candidate row the four C16 witness fields, status.json the witness summary.
"""
import json

import numpy as np
import pytest

from jd_curl.validation import InputValidationError

from bayesreg.cli import run_pair
from bayesreg.solve import Eta, NumpyBackend, SolverConfig, make_backend
from bayesreg.solve.multilevel import _template

from ..io_seg._helpers import DEV
from ..model import _logvol_digest as D
from . import _synthetic as syn
from .test_run_pair import setup, write_synthetic_pair, write_tissue  # noqa: F401  (setup is a fixture)

CLOSED = "gauss27+closed27"


@pytest.mark.parametrize("share", [1., 3.])
def test_closed27_option_solves(share):
    backend = NumpyBackend()
    old_model, old_pg = D.path_options("pre_def")              # DEF: the pre-DEF default path, explicit
    result, rows = D.solve_run("a", backend, 6, pg=old_pg, logvol_points=CLOSED, logvol_closed27_share=share)
    default, default_rows = D.solve_run("a", NumpyBackend(), 6, pg=old_pg, **old_model)
    assert result.status == "cap_reached" and result.accepted == 6
    candidates = [r for r in rows if r["kind"] == "candidate"]
    accepted = [r for r in candidates if r["accepted"]]
    assert len(accepted) == 6 and all(r["cert_status"] == "certified_feasible" and r["certified"] > 0
                                      for r in accepted)
    assert all(b < a for a, b in zip(result.energies, result.energies[1:]))
    evaluated = [r for r in candidates if r["energy"] is not None]
    assert evaluated and all(set(r["logvol_parts"]) >= {"gauss", "closed27", "closed27_min_J", "closed27_min_cell"}
                             for r in evaluated)
    assert all("logvol_parts" not in r for r in default_rows if r["kind"] == "candidate")
    assert D.trajectory_digest(rows, result) != D.trajectory_digest(default_rows, default)
    torch_backend = make_backend("torch64", device="cpu")
    other, other_rows = D.solve_run("a", torch_backend, 6, pg=old_pg, logvol_points=CLOSED,
                                    logvol_closed27_share=share)
    decisions = [[(r["iteration"], r["attempt"], r["accepted"], r["reason"]) for r in table if r["kind"] == "candidate"]
                 for table in (rows, other_rows)]
    assert decisions[0] == decisions[1] and np.max(np.abs(other.commit.u-result.commit.u)) < 1e-6
    assert "logvol_closed27" in torch_backend.record()["comparison"]["per_cell"]


def test_solver_config_template_and_cli():
    config = SolverConfig(logvol_points=CLOSED, logvol_closed27_share=3., closed27_weights="uniform")
    record = config.record()
    assert record["logvol_points"] == CLOSED and record["logvol_closed27_share"] == 3.
    assert record["closed27_weights"] == "uniform"
    g27 = SolverConfig(logvol_points="gauss27")
    assert "logvol_closed27_share" not in g27.record() and g27.logvol_options() == {}
    default = SolverConfig()                                    # DEF: gauss27+closed27, beta_S 1, Simpson
    assert (default.record()["logvol_points"], default.record()["logvol_closed27_share"],
            default.record()["closed27_weights"]) == (CLOSED, 1., "simpson")
    assert default.logvol_options() == dict(logvol_closed27_share=1., closed27_weights="simpson")
    for bad in (dict(logvol_points="gauss27", logvol_closed27_share=2.),
                dict(logvol_points="gauss27", closed27_weights="uniform"),
                dict(logvol_points=CLOSED, logvol_closed27_share=0.), dict(logvol_points=CLOSED, closed27_weights="x"),
                dict(logvol_points="gauss27+corners", logvol_closed27_share=2.)):
        with pytest.raises(InputValidationError):
            SolverConfig(**bad)
    grid, level, _ = syn.single_level(n=7)
    eta = Eta(1e-3, 1e-3, (1., .25, .25, .25))
    host = _template(level, eta, grid.omega, 8192, 1, 1, config.logvol_points, **config.logvol_options())
    assert host.logvol_closed27 and host.logvol_closed27_share == 3. and host.closed27_weights == "uniform"
    again = host.replace(sigma2=np.array([.5]))
    assert again.logvol_closed27_share == 3. and np.array_equal(again.closed27_point_weights, host.closed27_point_weights)
    base = ["--pair", "s/p", "--channel-source", "none", "--out", "x"]
    args = run_pair.parse_args(base)
    assert (args.logvol_points, args.logvol_closed27_share, args.closed27_weights, args.witness_estimate) == \
        (CLOSED, 1., "simpson", "off")                                              # DEF defaults
    assert run_pair.backend_of(args).witness_estimate is False
    assert run_pair.closed27_switches(g27) == dict(logvol_closed27_share=None, closed27_weights=None)
    assert run_pair.closed27_switches(SolverConfig()) == dict(logvol_closed27_share=1., closed27_weights="simpson")
    assert run_pair.closed27_switches(config) == dict(logvol_closed27_share=3., closed27_weights="uniform")
    args = run_pair.parse_args(base+["--logvol-points", CLOSED, "--logvol-closed27-share", "3",
                                     "--closed27-weights", "uniform", "--witness-estimate", "on"])
    assert (args.logvol_points, args.logvol_closed27_share, args.closed27_weights, args.witness_estimate) == \
        (CLOSED, 3., "uniform", "on")
    assert run_pair.backend_of(args).witness_estimate is True
    for extra in (["--logvol-points", CLOSED, "--logvol-closed27-beta", "2"],          # IMPL27 aliases removed
                  ["--logvol-points", CLOSED, "--logvol-closed27-weights", "uniform"],
                  ["--witness-record", "off"],
                  ["--logvol-points", "gauss27", "--logvol-closed27-share", "2"],
                  ["--logvol-points", "gauss27", "--closed27-weights", "uniform"],
                  ["--logvol-points", "gauss27+corners", "--logvol-closed27-share", "2"],
                  ["--logvol-points", CLOSED, "--logvol-closed27-share", "0"],
                  ["--logvol-points", CLOSED, "--logvol-closed27-share", "nan"],
                  ["--logvol-points", CLOSED, "--closed27-weights", "gauss"]):
        with pytest.raises(SystemExit):
            run_pair.parse_args(base+extra)


def test_end_to_end_closed27(setup):  # noqa: F811
    root, args = setup
    grid, pair = write_synthetic_pair(root, DEV[0], DEV[1])
    write_tissue(root, grid, DEV[0])
    write_tissue(root, grid, DEV[1], shift=(.7, -.5, .4))
    out = root/"run_closed27"
    assert run_pair.main(args(pair, out, "--rounds", "1,1,1,1", "--max-accepted", "2", "--logvol-points", CLOSED,
                              "--logvol-closed27-share", "3")) == 0
    read = lambda name: json.loads((out/name).read_text())                       # noqa: E731
    status = read("status.json")
    assert status["status"] == "cap_reached" and status["export"]["certificate_passed"] is True
    config = read("config.json")
    assert config["solver"]["logvol_points"] == CLOSED and config["solver"]["logvol_closed27_share"] == 3.
    assert config["solver"]["closed27_weights"] == "simpson" and config["args"]["logvol_closed27_share"] == 3.
    switches = read("provenance.json")["switches"]
    assert (switches["logvol_points"], switches["logvol_closed27_share"], switches["closed27_weights"],
            switches["witness_estimate"]) == (CLOSED, 3., "simpson", "off")
    assert read("hyper.json")["backend_after_solve"]["witness_estimate"] is False
    timing = read("timing.json")
    assert timing["witness_records"] >= 0 and timing["witness_seconds"] >= 0.
    rows = [json.loads(line) for line in (out/"log.jsonl").read_text().splitlines()]
    candidates = [r for r in rows if r["kind"] == "candidate"]
    evaluated = [r for r in candidates if r["energy"] is not None]
    assert evaluated and all("logvol_parts" in r for r in evaluated)
    fields = ("witness_cell", "witness_xi", "witness_type", "witness_cell_min_j53")
    assert all(set(fields) <= set(r) for r in candidates)
    assert all(all(r[k] is None for k in fields) for r in candidates if r["reason"] != "JACOBIAN_VIOLATION")
    violations = [r for r in candidates if r["reason"] == "JACOBIAN_VIOLATION"]
    assert all(r["witness_cell"] is not None and r["witness_cell_min_j53"] is not None for r in violations)
    summary = status["witness_summary"]
    assert summary["jacobian_violations"] >= len(violations)
    assert sum(summary["witness_type_counts"].values()) == summary["jacobian_violations"]
    cert = read("cert.json")
    assert cert["status"] == "certified_feasible" and cert["witness_type"] is None and "witness" not in cert
    assert cert["witness_cell_min_j53"] is None and cert["witness_estimate"] is None
