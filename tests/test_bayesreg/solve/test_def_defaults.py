"""Work package DEF (2026-10-31): the code defaults follow the contract text of 8a4d060 (decision_log 10-31).

  logvol_points     "gauss27" -> "gauss27+closed27", beta_S 1, closed27_weights "simpson"   (C05-2b first paragraph,
                                                                                            D43, D48)
  jd_cell_estimate  "gauss27+corners" -> "certificate"                                     (C09.1-3a last item, D47)
  metric_jd_scaling "off": jd_cell_estimate has no effect -- not given, it is recorded as "none"; given (any value),
                    the run is refused (before DEF: only the default was allowed).

(a) run_pair without any model / path option records gauss27+closed27, beta_S 1, simpson, affine_v1 and certificate in
    config.json (args, solver), provenance.json (switches), hyper.json and log.jsonl; the object and parser defaults
    agree (ModelState, SolverConfig, PGConfig, JDScalingConfig, the level template, the torch64 record before any
    state, run_pair, the eta probe and sweep, the EB entry point's config switches).
(b) --metric-jd-scaling off without --jd-cell-estimate runs and records "none"; any explicit value is refused before
    the run directory exists (usage error, exit 2).
(d) bitwise: the old options given explicitly -- pre_def (gauss27, gauss27+corners), pre_def_off (gauss27, metric off,
    no cell estimate) and corners (gauss27+corners, gauss27+corners) -- give the run digests (_run_pair_digest: every
    candidate / accept / round / transfer / S~ row, the status and the exported state) that the code before DEF
    (8a4d060) gave; the new default (no option) gives the digest that 8a4d060 gave with the new defaults explicit
    (def_explicit).  Golden: GOLDEN_8A4D060_RUN_PAIR, record tests/test_bayesreg/_golden/golden_8a4d060_run_pair.json,
    generated once by
        python tests/test_bayesreg/_golden/make_golden_run_pair.py --ref 8a4d060 \\
            --out tests/test_bayesreg/_golden/golden_8a4d060_run_pair.json
    (git archive of 8a4d060, the digest module of this tree copied in; numpy 2.0.2, single-threaded).
The model / torch64 / solver-function digests of the new default: GOLDEN_DEF in model/test_logvol_corners.py,
torch64/test_torch64_corners.py and solve/test_logvol_points.py (record golden_8a4d060_def_explicit.json).
26^3 synthetic pair, numpy backend, no data/, no GPU; about a minute (the default run is the slowest, ~25 s).
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from jd_curl.validation import InputValidationError

from bayesreg.cli import run_pair
from bayesreg.model import LOGVOL_DEFAULT, ModelState
from bayesreg.solve import JDScalingConfig, PGConfig, SolverConfig
from bayesreg.solve import jd_scaling as jds
from bayesreg.solve.multilevel import _template

from . import _run_pair_digest as RD
from . import _synthetic as syn
from ..model._helpers import aniso, make_state

ROOT = Path(__file__).resolve().parents[3]
GOLDEN_RECORD = ROOT/"tests"/"test_bayesreg"/"_golden"/"golden_8a4d060_run_pair.json"
# Source: tests/test_bayesreg/_golden/golden_8a4d060_run_pair.json (make_golden_run_pair.py --ref 8a4d060)
GOLDEN_8A4D060_RUN_PAIR = dict(
    corners="c36283f5cc4f7cf15a5092e707e723cf436a6a177a00252e17f17a58d1b53f9c",
    def_explicit="3d4f455f95564a71ae1c401b3713dfd66a4d6218197ac6132f51f6ac6a444767",
    pre_def="66c1c86e65149c8ca47aab1d84ddf63c4cc18ce1b14a74e8d2729a5f8cd8ef40",
    pre_def_off="f8805f5a2a9eb2a230c2b0bd22655a35447d81ba2ea28c38c541e57d99e7fe25")
BITWISE = pytest.mark.skipif(np.__version__ != "2.0.2", reason="bitwise golden digests recorded with numpy 2.0.2")
CLOSED, CERT = "gauss27+closed27", "certificate"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Runs:
    """The synthetic inputs (written once) and the arms, each run once (run directories under one tmp root)."""

    def __init__(self, root):
        self.root = root
        self.base = RD.write_inputs(root)
        self.done = {}

    def __call__(self, arm):
        if arm not in self.done:
            options = RD.DEFAULT_ARM if arm == "default" else RD.ARM_ARGS[arm]
            self.done[arm] = RD.run_arm(self.base, self.root/f"run_{arm}", options)
        return self.done[arm]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    return Runs(tmp_path_factory.mktemp("def_runs"))


# ---- (a) the defaults ---------------------------------------------------------------------------------------------

def test_run_pair_without_options_records_the_def_defaults(runs):
    code, out = runs("default")
    assert code == 0
    config, prov, hyper = read(out/"config.json"), read(out/"provenance.json"), read(out/"hyper.json")
    args = config["args"]
    assert (args["logvol_points"], args["logvol_closed27_share"], args["closed27_weights"], args["metric_jd_scaling"],
            args["jd_cell_estimate"]) == (CLOSED, 1., "simpson", "affine_v1", CERT)
    solver = config["solver"]
    assert (solver["logvol_points"], solver["logvol_closed27_share"], solver["closed27_weights"]) == (CLOSED, 1.,
                                                                                                      "simpson")
    assert solver["pg"]["metric_jd_scaling"] == "affine_v1" and solver["pg"]["jd_cell_estimate"] == CERT
    assert solver["pg"]["jd_scaling"]["cell_estimate"] == CERT
    assert prov["solver"] == solver
    switches = prov["switches"]
    assert (switches["logvol_points"], switches["logvol_closed27_share"], switches["closed27_weights"],
            switches["metric_jd_scaling"], switches["jd_cell_estimate"]) == (CLOSED, 1., "simpson", "affine_v1", CERT)
    rounds = hyper["rounds"]
    assert [r["level"] for r in rounds] == [0, 1, 2, 3]
    assert all(r["metric"]["jd_scaling"]["cell_estimate"] == CERT for r in rounds)
    assert hyper["algorithm"]["pg"]["jd_cell_estimate"] == CERT
    rows = [json.loads(line) for line in (out/"log.jsonl").read_text().splitlines()]
    scaling = [r for r in rows if r["kind"] == "jd_scaling"]
    assert scaling and all(r["cell_estimate"] == CERT and r["cert_min_matches"] for r in scaling)
    evaluated = [r for r in rows if r["kind"] == "candidate" and r["energy"] is not None]
    assert evaluated and all("closed27" in r["logvol_parts"] for r in evaluated)        # C05-2b in F_B
    status = read(out/"status.json")
    assert status["export"]["certificate_passed"] is True and (out/"disp.npz").is_file()


def test_object_and_parser_defaults_agree():
    assert LOGVOL_DEFAULT == CLOSED
    state = make_state(aniso())                                         # ModelState default (helpers pass none)
    assert (state.logvol_points, state.logvol_closed27_share, state.closed27_weights) == (CLOSED, 1., "simpson")
    assert state.logvol_closed27 and state.logvol_theta == (.5, .5)
    defaults = {f.name: f.default for f in ModelState.__dataclass_fields__.values()}
    assert (defaults["logvol_points"], defaults["logvol_closed27_share"], defaults["closed27_weights"]) == (
        CLOSED, 1., "simpson")
    config = SolverConfig()
    assert (config.logvol_points, config.logvol_closed27_share, config.closed27_weights) == (CLOSED, 1., "simpson")
    assert config.logvol_options() == dict(logvol_closed27_share=1., closed27_weights="simpson")
    assert JDScalingConfig().cell_estimate == CERT == jds.DEFAULT_CELL_ESTIMATE
    pg = PGConfig()
    assert pg.metric_jd_scaling == "affine_v1" and pg.jd == JDScalingConfig() and pg.jd_cell_estimate == CERT
    assert pg.record()["jd_cell_estimate"] == CERT and pg.record()["jd_scaling"]["cell_estimate"] == CERT
    grid, level, _ = syn.single_level(n=7)
    from bayesreg.solve import Eta
    host = _template(level, Eta(1e-3, 1e-3, (1., .25, .25, .25)), grid.omega, 8192, 1)
    assert host.logvol_points == CLOSED and host.logvol_closed27_share == 1. and host.closed27_weights == "simpson"
    base = ["--pair", "s/p", "--channel-source", "none", "--out", "x"]
    args = run_pair.parse_args(base)
    assert (args.logvol_points, args.logvol_closed27_share, args.closed27_weights, args.metric_jd_scaling,
            args.jd_cell_estimate) == (CLOSED, 1., "simpson", "affine_v1", CERT)
    pg = run_pair.pg_config(args.metric_jd_scaling, args.jd_cell_estimate)
    assert pg == PGConfig() and pg.jd_cell_estimate == CERT
    probe = load("def_eta_probe", "experiments/eta/probe.py")
    p = probe.parse_args(base)
    assert (p.logvol_points, p.logvol_closed27_share, p.closed27_weights, p.metric_jd_scaling, p.jd_cell_estimate) == \
        (args.logvol_points, args.logvol_closed27_share, args.closed27_weights, args.metric_jd_scaling,
         args.jd_cell_estimate)
    sweep = load("def_eta_sweep", "experiments/eta/sweep.py")
    s = sweep.parse_args([])
    assert (s.logvol_points, s.logvol_closed27_share, s.metric_jd_scaling, s.jd_cell_estimate) == (CLOSED, 1.,
                                                                                                   "affine_v1", CERT)
    from bayesreg.learn import cli as eb_cli
    eb_args = eb_cli.parse_args(["--out", "x", "--pairs-root", "y", "--eta0-json", "z"])
    assert eb_cli.solver_switches(eb_args) == dict(outer="a", logvol_points=CLOSED, logvol_closed27_share=1.,
                                                   closed27_weights="simpson", metric_jd_scaling="affine_v1",
                                                   jd_cell_estimate=CERT)
    pytest.importorskip("torch")
    from bayesreg.backends.torch64 import stable
    from bayesreg.solve import Torch64Backend
    assert "logvol_closed27" in stable.reduction_provenance(8192)["per_cell"]
    record = Torch64Backend(device="cpu").record()                      # before any state: the model default
    assert "logvol_closed27" in record["comparison"]["per_cell"]


def test_explicit_old_options_still_parse_and_record():
    base = ["--pair", "s/p", "--channel-source", "none", "--out", "x"]
    args = run_pair.parse_args(base+["--logvol-points", "gauss27", "--jd-cell-estimate", "gauss27+corners"])
    assert (args.logvol_points, args.jd_cell_estimate) == ("gauss27", "gauss27+corners")
    config = SolverConfig(logvol_points=args.logvol_points,
                          pg=run_pair.pg_config(args.metric_jd_scaling, args.jd_cell_estimate))
    assert run_pair.closed27_switches(config) == dict(logvol_closed27_share=None, closed27_weights=None)
    assert config.record()["pg"]["jd_cell_estimate"] == "gauss27+corners" and config.logvol_options() == {}


# ---- (b) metric off ---------------------------------------------------------------------------------------------

def test_metric_off_records_none_and_runs(runs):
    code, out = runs("pre_def_off")
    assert code == 0
    config, prov, hyper = read(out/"config.json"), read(out/"provenance.json"), read(out/"hyper.json")
    assert config["args"]["metric_jd_scaling"] == "off" and config["args"]["jd_cell_estimate"] == "none"
    assert config["solver"]["pg"]["jd_cell_estimate"] == "none" and config["solver"]["pg"]["jd_scaling"] is None
    assert prov["switches"]["metric_jd_scaling"] == "off" and prov["switches"]["jd_cell_estimate"] == "none"
    assert all(r["metric"]["jd_scaling"] == dict(mode="off") for r in hyper["rounds"])
    rows = [json.loads(line) for line in (out/"log.jsonl").read_text().splitlines()]
    assert not [r for r in rows if r["kind"] == "jd_scaling"]
    assert read(out/"status.json")["export"]["certificate_passed"] is True


@pytest.mark.parametrize("estimate", jds.CELL_ESTIMATES)
def test_metric_off_with_an_explicit_cell_estimate_is_refused(runs, estimate, capsys):
    out = runs.root/f"refused_{estimate}"
    with pytest.raises(SystemExit) as stop:
        run_pair.main([*runs.base, "--out", str(out), "--metric-jd-scaling", "off", "--jd-cell-estimate", estimate])
    assert stop.value.code == 2 and not out.exists()
    assert "no effect with --metric-jd-scaling off" in capsys.readouterr().err
    pg_off = PGConfig(metric_jd_scaling="off")
    assert pg_off.jd is None and pg_off.jd_cell_estimate == "none" and pg_off.record()["jd_cell_estimate"] == "none"


# ---- (d) bitwise: the code before DEF ---------------------------------------------------------------------------

@BITWISE
@pytest.mark.parametrize("arm", ["pre_def", "pre_def_off", "corners"])
def test_explicit_old_options_are_bitwise_the_code_before_def(runs, arm):
    code, out = runs(arm)
    assert code == 0
    assert RD.run_digest(out) == GOLDEN_8A4D060_RUN_PAIR[arm]


@BITWISE
def test_default_is_bitwise_the_explicit_def_options_of_the_code_before_def(runs):
    code, out = runs("default")
    assert code == 0
    assert RD.run_digest(out) == GOLDEN_8A4D060_RUN_PAIR["def_explicit"]


def test_golden_constants_match_the_generation_record():
    record = read(GOLDEN_RECORD)
    assert record["ref_commit"].startswith("8a4d060") and record["digests"] == GOLDEN_8A4D060_RUN_PAIR
    assert record["environment"]["numpy"] == "2.0.2"
    assert record["environment"]["thread_env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert record["arm_args"] == {k: list(v) for k, v in RD.ARM_ARGS.items()}
    assert record["common_args"] == list(RD.COMMON)
    import hashlib
    module = Path(RD.__file__).read_bytes()
    assert hashlib.sha256(module).hexdigest() == record["digest_module_sha256"]        # regenerate after edits


# ---- MIN1 (review DEF minor 1): the torch64 record written before the solve names the run's point set --------------

@pytest.mark.parametrize("points", [None, "gauss27", "gauss27+corners", CLOSED])
def test_torch64_record_before_the_solve_names_the_runs_point_set(points):
    """run_pair.backend_of passes --logvol-points to Torch64Backend, so record()["comparison"] before any state is the
    reduction provenance of the run's point set (default: the model default, as before); the record is the one of
    stable.reduction_provenance.  An unknown point set is refused."""
    pytest.importorskip("torch")
    from bayesreg.backends.torch64 import stable
    from bayesreg.solve import Torch64Backend
    base = ["--pair", "s/p", "--channel-source", "none", "--out", "x", "--backend", "torch64", "--device", "cpu"]
    args = run_pair.parse_args(base+([] if points is None else ["--logvol-points", points]))
    expected = stable.reduction_provenance(8192, LOGVOL_DEFAULT if points is None else points)
    assert run_pair.backend_of(args).record()["comparison"] == expected
    assert Torch64Backend(device="cpu", logvol_points=points).record()["comparison"] == expected
    assert ("logvol_closed27" in expected["per_cell"]) == (points in (None, CLOSED))
    with pytest.raises(InputValidationError, match="logvol_points"):      # FUP1 (review MIN1 minor 5): the exact type
        Torch64Backend(device="cpu", logvol_points="closed27")


def test_torch64_config_json_names_the_explicit_gauss27_point_set(tmp_path):
    """run_pair --backend torch64 --logvol-points gauss27 (with --jd-cell-estimate gauss27+corners, the run before DEF):
    config.json and provenance.json, written before the solve, carry backend.comparison of gauss27 (no
    logvol_closed27 row), the same as hyper.json backend_after_solve; before MIN1 they described the model default
    gauss27+closed27.  --max-levels 1 (development switch): ends stopped_before_native, exit 2; records only."""
    pytest.importorskip("torch")
    from bayesreg.backends.torch64 import stable
    base = RD.write_inputs(tmp_path)
    out = tmp_path/"run_torch64_gauss27"
    code = run_pair.main([*base, "--out", str(out), "--backend", "torch64", "--device", "cpu", "--logvol-points",
                          "gauss27", "--jd-cell-estimate", "gauss27+corners", "--rounds", "1,1,1,1", "--max-accepted",
                          "1", "--max-levels", "1"])
    assert code == 2 and read(out/"status.json")["status"] == "stopped_before_native"
    expected = stable.reduction_provenance(8192, "gauss27")
    config, prov, hyper = read(out/"config.json"), read(out/"provenance.json"), read(out/"hyper.json")
    assert config["args"]["logvol_points"] == "gauss27" and config["solver"]["logvol_points"] == "gauss27"
    assert config["backend"]["comparison"] == expected == prov["backend"]["comparison"]
    assert hyper["backend_after_solve"]["comparison"] == expected
    assert "logvol_closed27" not in config["backend"]["comparison"]["per_cell"]
    assert config["backend"]["device_states"] == []                    # written before the solve
