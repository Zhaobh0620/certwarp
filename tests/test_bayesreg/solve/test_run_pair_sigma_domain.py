"""run_pair and the learn CLI with --sigma-domain (work package SWLI; contract C04-6 gates 1, 8, 15 and the records of
C04-6 "记录" / C16; draft SWL 2026-10-31).  Synthetic files only (the 26^3 pair of solve/test_run_pair.py), numpy.

Gate 1 golden: GOLDEN_DDCBFCE_SWLI, the record tests/test_bayesreg/_golden/golden_ddcbfce_swli.json, generated once on
the code before SWLI by
    python tests/test_bayesreg/_golden/make_golden_swli.py --ref ddcbfce --out tests/test_bayesreg/_golden/golden_ddcbfce_swli.json
(git archive of ddcbfce, the digest modules of this tree copied in; numpy 2.0.2, single-threaded).  The default run and
the run with an explicit --sigma-domain brain give those digests (log rows, numeric fields of hyper.json / qc.json, the
disp and final_native arrays); the configuration records have exactly the three keys sigma_domain, support_threshold,
support_rule more (brain, null, null), and nothing else of SWLI appears.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from jd_curl.validation import InputValidationError

from bayesreg.cli import run_pair
from bayesreg.learn import check_sigma_prior_domain
from bayesreg.learn import cli as learn_cli
from bayesreg.model.support import SUPPORT_RULE, theta_lower_bound
from bayesreg.seg.unet.groups import DEFAULT_GROUPS_PATH, load_grouping
from bayesreg.solve import Caps, GaussianMStep, NumpyBackend, SolverConfig, eta_temporary, solve_multilevel

from . import _run_pair_digest as RD
from . import _swli_digest as SD
from . import _swl_synthetic as S

ROOT = Path(__file__).resolve().parents[3]
GOLDEN_RECORD = ROOT/"tests"/"test_bayesreg"/"_golden"/"golden_ddcbfce_swli.json"
# Source: tests/test_bayesreg/_golden/golden_ddcbfce_swli.json (make_golden_swli.py --ref ddcbfce)
GOLDEN_DDCBFCE_SWLI = dict(
    runs=dict(default=dict(
        disp="86858b04ccb8c3c955d24a1f40580a98e3c5dafbeb9252688869abe361132f11",
        final_native="41482da58c315b0134a85ab8696d3074014db141cddf5307f935852bbcd9cdc8",
        hyper="0c3350fd91316202de0cb726aec220760c995b2b697067ee23588471a7951155",
        qc="2f0aacc934c13cb56059b8ca17249fa4b5ff8956b69e8b10a8b860ee49b68e91",
        run="3d4f455f95564a71ae1c401b3713dfd66a4d6218197ac6132f51f6ac6a444767")),
    laplace=dict(converged=[True]*4, iterations=[12]*4,
                 deltas="72157799d5195f018273184061aed496dbb3794f23ce460c5fd1a642c9cf9429"))
BITWISE = pytest.mark.skipif(np.__version__ != "2.0.2", reason="bitwise golden digests recorded with numpy 2.0.2")
SWLI_KEYS = ("sigma_domain", "support_threshold", "support_rule")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Runs:
    def __init__(self, root):
        self.root = root
        self.base = RD.write_inputs(root)
        self.done = {}

    def __call__(self, arm, options=None):
        if arm not in self.done:
            options = SD.ARMS[arm] if options is None else options
            self.done[arm] = RD.run_arm(self.base, self.root/f"run_{arm}", options)
        return self.done[arm]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    return Runs(tmp_path_factory.mktemp("swli_runs"))


# ---- gate 1 -------------------------------------------------------------------------------------------------------

def test_golden_record_matches_the_constants():
    record = read(GOLDEN_RECORD)
    assert record["ref_commit"].startswith("ddcbfce") and record["digests"] == GOLDEN_DDCBFCE_SWLI
    assert record["environment"]["numpy"] == "2.0.2"


@BITWISE
@pytest.mark.parametrize("arm", ["default", "brain_explicit"])
def test_brain_is_bitwise_the_code_before(runs, arm):
    code, out = runs(arm)
    assert code == 0
    assert SD.run_digests(out) == GOLDEN_DDCBFCE_SWLI["runs"]["default"]


@pytest.mark.parametrize("arm", ["default", "brain_explicit"])
def test_brain_records_only_the_three_keys(runs, arm):
    code, out = runs(arm)
    assert code == 0
    config, prov, hyper, qc = (read(out/name) for name in ("config.json", "provenance.json", "hyper.json", "qc.json"))
    expected = dict(sigma_domain="brain", support_threshold=None, support_rule=None)
    for record in (config["solver"], prov["solver"], prov["switches"], hyper["algorithm"]):
        assert {k: record[k] for k in SWLI_KEYS} == expected
    assert config["args"]["sigma_domain"] == "brain" and config["args"]["support_threshold"] is None
    assert config["sigma"] == GaussianMStep().record() == hyper["sigma_policy"]
    assert "support_levels" not in hyper and "sigma_domain" not in hyper["sigma_monitor"]
    for row in hyper["rounds"]:
        assert not {"sigma2_S_used", "sigma2_S_post", "sigma_tilde2"} & set(row)
    assert all("sigma2_S" not in row for row in qc["sigma_trajectory"])
    rows = [json.loads(line) for line in (out/"log.jsonl").read_text().splitlines()]
    assert not [r for r in rows if r["kind"] == "support"]
    timing = read(out/"timing.json")
    assert all("support_cpu_seconds" not in level for level in timing["solve_levels"]["levels"])


KEYS_RECORD = ROOT/"tests"/"test_bayesreg"/"_golden"/"golden_ddcbfce_swli_keys.json"
# review SWLI minor 4: the key paths that a brain run has more than the code before SWLI (ddcbfce), per file; the args
# echo of the two CLI switches included; nothing removed
ADDED_KEYS = {"config.json": ["/args/sigma_domain", "/args/support_threshold", "/solver/sigma_domain",
                              "/solver/support_rule", "/solver/support_threshold"],
              "provenance.json": ["/solver/sigma_domain", "/solver/support_rule", "/solver/support_threshold",
                                  "/switches/sigma_domain", "/switches/support_rule", "/switches/support_threshold"],
              "hyper.json": ["/algorithm/sigma_domain", "/algorithm/support_rule", "/algorithm/support_threshold"],
              "qc.json": [], "status.json": [], "timing.json": []}


@pytest.mark.parametrize("arm", ["default", "brain_explicit"])
def test_brain_record_keys_against_the_code_before(runs, arm):
    """Gate 1 measure (review SWLI minor 4): the key sets of every JSON file of the run against those of the code
    before SWLI (golden_ddcbfce_swli_keys.json, make_golden_swli.py --ref ddcbfce --keys)."""
    from . import _swli_keys as SK
    record = read(KEYS_RECORD)
    assert record["ref_commit"].startswith("ddcbfce")
    code, out = runs(arm)
    assert code == 0
    new = SK.run_key_sets(out)
    for name in SK.FILES:
        old = set(record["digests"][name])
        assert sorted(set(new[name])-old) == ADDED_KEYS[name], name
        assert not old-set(new[name]), name


@BITWISE
def test_brain_laplace_samples_are_bitwise_the_code_before():
    assert SD.laplace_digest() == GOLDEN_DDCBFCE_SWLI["laplace"]


# ---- records of support / support_shared (C04-6 "记录") --------------------------------------------------------

@pytest.mark.parametrize("domain", ["support", "support_shared"])
def test_support_run_records(runs, domain):
    code, out = runs(domain, ("--sigma-domain", domain))
    assert code == 0
    config, prov, hyper, qc = (read(out/name) for name in ("config.json", "provenance.json", "hyper.json", "qc.json"))
    expected = dict(sigma_domain=domain, support_threshold=.05, support_rule=SUPPORT_RULE)
    for record in (config["solver"], prov["solver"], prov["switches"], hyper["algorithm"]):
        assert {k: record[k] for k in SWLI_KEYS} == expected
    assert config["args"]["support_threshold"] == .05
    assert config["sigma"]["sigma_domain"] == domain == hyper["sigma_policy"]["sigma_domain"]
    assert prov["switches"]["noise"] == "gaussian"
    levels = hyper["support_levels"]
    assert [r["level"] for r in levels] == [0, 1, 2, 3]
    for r in levels:
        assert len(r["N_c"]) == 3 and r["theta"] == .05 and r["t_theta"] == float(np.sqrt(.05))
        assert set(r) >= {"support_nodes", "fixed_only_nodes", "moving_only_nodes", "empty_support_channels",
                          "n_near_threshold", "v_sha256", "seed_state_hash", "N_q", "N_c_over_N_q", "sigma2_start"}
        assert ("N_S" in r) == (domain == "support_shared")
    for row in hyper["rounds"]:
        assert len(row["sigma_tilde2"]) == 3
        assert ("sigma2_S_used" in row) == (domain == "support_shared")
    assert hyper["sigma_monitor"]["sigma_domain"] == domain
    assert len(qc["sigma_trajectory"]) >= 4
    rows = [json.loads(line) for line in (out/"log.jsonl").read_text().splitlines()]
    assert [r["level"] for r in rows if r["kind"] == "support"] == [0, 1, 2, 3]
    timing = read(out/"timing.json")
    assert all(level["support_cpu_seconds"] >= 0 for level in timing["solve_levels"]["levels"])
    assert read(out/"status.json")["export"]["certificate_passed"] is True


# ---- gate 8: refusals before the run directory --------------------------------------------------------------------

def _refused(tmp_path, *options, channel_source="tissue"):
    out = tmp_path/"never"
    argv = ["--pair", "s/p", "--channel-source", channel_source, "--out", str(out), *options]
    with pytest.raises(SystemExit) as info:
        run_pair.main(argv)
    assert info.value.code == 2 and not out.exists()


@pytest.mark.parametrize("options,source", [
    (("--sigma-domain", "support"), "none"),
    (("--sigma-domain", "support_shared"), "none"),
    (("--support-threshold", ".05"), "tissue"),
    (("--sigma-domain", "brain", "--support-threshold", ".05"), "tissue"),
    (("--sigma-domain", "support", "--support-threshold", "0"), "tissue"),
    (("--sigma-domain", "support", "--support-threshold", "1"), "tissue"),
    (("--sigma-domain", "support", "--support-threshold", "nan"), "tissue"),
    (("--sigma-domain", "support", "--support-threshold", "inf"), "tissue"),
    (("--sigma-domain", "support", "--support-threshold", "-0.1"), "tissue"),
    (("--sigma-domain", "support", "--support-threshold", repr(theta_lower_bound(1))), "tissue"),
    (("--sigma-domain", "other"), "tissue"),
])
def test_run_pair_refusals(tmp_path, options, source):
    _refused(tmp_path, *options, channel_source=source)


def test_parse_time_floor_bound_structure_and_tissue():
    k_s = load_grouping(DEFAULT_GROUPS_PATH, None).k
    base = ["--pair", "s/p", "--out", "x"]
    with pytest.raises(SystemExit):
        run_pair.parse_args([*base, "--channel-source", "structure", "--sigma-domain", "support",
                             "--support-threshold", repr(theta_lower_bound(k_s))])
    above = repr(float(np.nextafter(theta_lower_bound(k_s), 1)))
    args = run_pair.parse_args([*base, "--channel-source", "structure", "--sigma-domain", "support",
                                "--support-threshold", above])
    assert args.support_threshold == float(above)
    args = run_pair.parse_args([*base, "--channel-source", "tissue", "--sigma-domain", "support_shared"])
    assert (args.sigma_domain, args.support_threshold) == ("support_shared", .05)
    args = run_pair.parse_args([*base, "--channel-source", "none"])
    assert (args.sigma_domain, args.support_threshold) == ("brain", None)


def test_solver_checks_theta_with_k_t():
    """gate 15: the solver's bound uses K = n_channels - 1 (K_T = 3 here, as under channel_source tissue)."""
    _, levels, _ = S.four_levels_structure()
    mid = (theta_lower_bound(3)+theta_lower_bound(13))/2
    config = SolverConfig(rounds=(1, 1, 1, 1), caps=Caps(1, 1), sigma_domain="support", support_threshold=mid,
                          max_levels=1)
    eta = eta_temporary(levels[-1].tissue, levels[-1].mask, levels[-1].grid)
    with pytest.raises(InputValidationError):
        solve_multilevel(NumpyBackend(), list(levels), eta=eta, sigma=GaussianMStep(), config=config)
    intensity_only = [lv.__class__(lv.level, lv.grid, lv.fixed[:1], lv.moving[:1], lv.mask, lv.tissue) for lv in levels]
    with pytest.raises(InputValidationError):                     # intensity only (channel_source none): refused
        solve_multilevel(NumpyBackend(), intensity_only, eta=eta, sigma=GaussianMStep(),
                         config=SolverConfig(sigma_domain="support", max_levels=1))
    with pytest.raises(InputValidationError):
        SolverConfig(sigma_domain="brain", support_threshold=.05)


def test_sigma_prior_domain_check():
    assert check_sigma_prior_domain({}, "brain", None) == ("brain", None)            # old prior files: brain
    assert check_sigma_prior_domain(dict(sigma_domain="support", support_threshold=.05), "support", .05)
    for prior, domain, theta in (({}, "support", .05), (dict(sigma_domain="support", support_threshold=.05),
                                                        "support_shared", .05),
                                 (dict(sigma_domain="support", support_threshold=.05), "support", .1),
                                 (dict(sigma_domain="brain", support_threshold=None), "support", .05)):
        with pytest.raises(InputValidationError):
            check_sigma_prior_domain(prior, domain, theta)


# ---- learn CLI switches -----------------------------------------------------------------------------------------------

def _learn_args(*extra):
    return ["--out", "x", "--pairs-root", "p", "--eta0-json", "e", *extra]


def test_learn_cli_switches_and_record():
    args = learn_cli.parse_args(_learn_args())
    assert (args.sigma_domain, args.support_threshold) == ("brain", None)
    # brain: the SolverConfig options are exactly those of the code before SWLI (review SWLI minor 1)
    assert set(learn_cli.solver_options(args)) == {"pg", "logvol_points", "logvol_closed27_share", "closed27_weights"}
    assert not set(SWLI_KEYS) & set(learn_cli.solver_switches(args))           # `switches` as before SWLI
    assert learn_cli.sigma_domain_record(args) == dict(sigma_domain="brain", support_threshold=None, support_rule=None)
    args = learn_cli.parse_args(_learn_args("--sigma-domain", "support_shared", "--support-threshold", ".1"))
    assert learn_cli.solver_options(args)["sigma_domain"] == "support_shared"
    assert learn_cli.sigma_domain_record(args) == dict(sigma_domain="support_shared", support_threshold=.1,
                                                       support_rule=SUPPORT_RULE)
    config = learn_cli.solver_config(args, S.four_levels_structure()[0])
    assert (config.sigma_domain, config.support_threshold) == ("support_shared", .1)
    for bad in (("--support-threshold", ".05"), ("--sigma-domain", "support", "--support-threshold", "2"),
                ("--sigma-domain", "support", "--support-threshold", repr(theta_lower_bound(1)))):
        with pytest.raises(SystemExit):
            learn_cli.parse_args(_learn_args(*bad))
