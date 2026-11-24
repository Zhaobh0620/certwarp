"""run_pair with the per-level caps and the coarse-level shares (work package SCHI, 2026-10-31; contract C13-4a, C13-5a,
C16 SCH item; the C13 SCH gates 1, 2, 3, 4, 10, 12, 13, 14).  Synthetic data only: the 26^3 pair of
solve/test_run_pair.py, channel source tissue, numpy backend (plus a torch64 cpu arm for gate 2).

Gate 1 (default unchanged):
  * the twelve existing goldens are untouched (their sha256 pinned below; the tests that read them are unchanged and
    run in the full suite);
  * the seven record files of this tree, run in a child process with the single-thread environment of the golden
    (make_golden_sch.run_worktree), have the masked sha256 of golden_11409eb_sch_records.json (S measured on 11409eb,
    every measured leaf in one of the five contract classes, no solve quantity in S), arm by arm: default (the contract's
    default arm, --max-accepted 2), wall (--wall-seconds, no shares), nocap_l0 (no --max-accepted at all);
  * the key sets of the six JSON files and of the log.jsonl row kinds equal the golden's; config.json args has neither
    new switch.
Gate 2: --max-accepted-per-level 2,2,2,2 and --max-accepted 2 give the same digest (numpy and torch64 cpu); the
  --max-accepted 2 run is the golden default arm (GOLDEN_8A4D060_RUN_PAIR def_explicit).
Gate 3: rounds 1,1,1,2 with --max-accepted-per-level 1,2,3,4 (option (a)) and 7,6,2,3 (option (b), n_in 5): every
  (level, round) ends cap_reached (premise), accepted = a_ell, every inner solve of (b) has min(n_in, a_ell - accepted).
Gate 4: refusals in parse_args (exit 2, no run directory).
Gate 10: shares that never come due (W = 1e9) give the digest of the same run without shares.
Gate 12: Caps(max_accepted=...) keeps its interface and record; probe.py's PROBE_CAPS; the learn CLI's SolverConfig and
  its record have no SCH key.
Gate 14: with a switch every C16 key is in its file (and none without, gate 1); level_share_hit.elapsed_seconds >= D;
  ended_by_level_share iff level_budget_reached; null elapsed keys without --wall-seconds; qc.json and status.json have
  the key sets of the golden's runs.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from bayesreg.cli import run_pair
from bayesreg.solve import Caps, SolverConfig
from bayesreg.solve.stopping import LEVEL_BUDGET_REACHED, WallBudget

from . import _run_pair_digest as RD
from . import _sch_records as R
from .test_def_defaults import GOLDEN_8A4D060_RUN_PAIR
from .test_run_pair import read

ROOT = Path(__file__).resolve().parents[3]
GOLDEN_DIR = ROOT/"tests"/"test_bayesreg"/"_golden"
SCH_GOLDEN = GOLDEN_DIR/"golden_11409eb_sch_records.json"
BITWISE = pytest.mark.skipif(np.__version__ != "2.0.2", reason="bitwise golden digests recorded with numpy 2.0.2")
# the twelve goldens of gate 1, first item (sha256 of the files as on 11409eb)
EXISTING_GOLDENS = dict(
    golden_481e33b="4912bc4d4de0fa57bb7be8fae02177d69bea349e5659badd84f05035a7933a60",
    golden_481e33b_def="61c1982843d39c0238ee8ca66fe75a579ea94c7f1b79589b5eba3d069ec1d9cc",
    golden_7579645_eb="3423b87c6b5d5d9e52093d56f715067b5294a7406eb22ee823eff224abccac35",
    golden_7579645_eb_def="dbfc5c06d8aa822929a15589150350d5f7ce191d4b5803957ab9f141376289d9",
    golden_7579645_eb_ebd="7193765feed1b5f5651a772d099f2d3291edb1cb3bf20e7870ffa5d97c9d2120",
    golden_8a4d060_def_explicit="fcbfc63ea7ec8bf3e8e8aa480e126fc08440d0c91f64f73d72d8e166ad97c1c3",
    golden_8a4d060_run_pair="7168e9921dc3f60ab128e5f2841a4fc539b0b2d4cdf3acb311ce4fa2d442af6d",
    golden_9747553_mem="9edb62a8595c945dc89b7e306fcb74913601cfcb94c06b773bae429017b37bd0",
    golden_9747553_mem_paths="ebb6cf35d22da9b1bdfdac2872125c4ede07ef5e6596c6b688c26e76a2b0cea4",
    golden_d046b77_mem_gf="5ead7aeff357f688051eefdeff903b2d4c9ee6ebc036fc4d94efd5801b5cb3bf",
    golden_ddcbfce_swli="b14c853805a263e017db6e70e3d768734c18916b0cf3cfe4fd82a27929887d59",
    golden_ddcbfce_swli_keys="cc537a8f8c1b3f181fc75c36fea570a42d832533684f1341f05505de2a780610")
SCH_SOLVER_KEYS = ("max_accepted_per_level", "max_accepted_source", "level_time_shares", "level_deadlines_seconds")
SCH_ROUND_KEYS = ("cap_accepted", "accepted_committed", "level_time_share", "level_cumulative_share",
                  "level_deadline_seconds", "level_elapsed_seconds", "run_elapsed_seconds", "ended_by_level_share",
                  "level_share_hit", "rounds_skipped")


def load(name, path):
    """Import a script by path; sys.path and the modules it adds from experiments/ (e.g. cvg's `common`) are restored
    afterwards, so that nothing leaks into later tests."""
    saved_path, saved_modules = list(sys.path), set(sys.modules)
    try:
        spec = importlib.util.spec_from_file_location(name, ROOT/path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved_path
        for added in set(sys.modules)-saved_modules:
            origin = getattr(sys.modules[added], "__file__", None) or ""
            if str(ROOT/"experiments") in origin:
                del sys.modules[added]
    return module


@pytest.fixture(scope="module")
def golden():
    return json.loads(SCH_GOLDEN.read_text(encoding="utf-8"))


class Runs:
    """The synthetic inputs written once; each arm run once (run directories under one tmp root)."""

    def __init__(self, root):
        self.root = root
        self.base = RD.write_inputs(root)
        self.done = {}

    def __call__(self, name, *options, monkeypatch=None, expect=0):
        if name not in self.done:
            from bayesreg.solve import jd_scaling as jds
            jds.clear_cell_bounds_cache()
            out = self.root/f"run_{name}"
            code = run_pair.main([*self.base, "--out", str(out), *options])
            assert code == expect, (name, code)
            self.done[name] = out
        return self.done[name]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("sch_runs")
    return Runs(root)


COMMON = ("--rounds", "1,1,1,1")


# ---- gate 1 ------------------------------------------------------------------------------------------------------------

def test_existing_goldens_are_untouched():
    for name, digest in EXISTING_GOLDENS.items():
        assert hashlib.sha256((GOLDEN_DIR/f"{name}.json").read_bytes()).hexdigest() == digest, name


def test_measured_mask_is_within_the_five_classes(golden):
    assert golden["ref_commit"].startswith("11409eb") and golden["gate_failed"] is False
    for arm, record in golden["arms"].items():
        assert record["unclassified"] == [] and record["digests_agree"] is True, arm
        assert record["solve_quantities_in_S"] == [], arm
        assert all(m["cls"] in R.CLASSES and m["kind"] == "value" for m in record["measured"]), arm
        assert all(record["roundtrip_exact"].values()), arm
        for m in record["measured"]:
            assert m["pattern"] in record["S"][m["file"]]
        # the classes, re-derived here from the recorded leaves
        roots = golden["roots"]
        for m in record["measured"]:
            cls = R.classify(m["file"], m["path"], m["a"], m["b"], roots=roots)
            assert cls == m["cls"] or (cls is None and m["cls"] == "npz_sha256"), (arm, m["path"])
    # S holds no solve quantity: energies, gradients, sigma, G0, T_B, state hashes, bounds, statuses, counts
    for arm, record in golden["arms"].items():
        flat = [p for name, patterns in record["S"].items() for p in patterns
                if not (name == "provenance.json" and p.startswith("/code/"))]        # class code (version record)
        for key in ("energy", "GF_B", "G0", "T_B", "sigma2", "state_hash", "min_jd_lower_bound", "status", "reason",
                    "accepted", "commits", "certificate_calls", "witness_records", "checks"):
            assert not [p for p in flat if R._keys(p) and R._keys(p)[-1] == key and "wall_budget" not in p], (arm, key)


@BITWISE
def test_default_record_files_are_byte_identical_under_the_mask(golden, tmp_path):
    gen = load("make_golden_sch", "tests/test_bayesreg/_golden/make_golden_sch.py")
    child = gen.run_worktree(tmp_path/"root")
    for arm, record in golden["arms"].items():
        texts = child["texts"][arm]
        assert child["codes"][arm] == record["exit_codes"][0], arm
        assert R.masked_digests(texts, record["S"]) == record["digests"], arm
        keys = R.key_sets(texts)
        assert keys == record["key_sets"], arm
        args = json.loads(texts["config.json"])["args"]
        assert "max_accepted_per_level" not in args and "level_time_shares" not in args
        for name in ("config.json", "provenance.json", "hyper.json"):
            assert not [k for k in keys[name] if any(f"/{s}" in k for s in SCH_SOLVER_KEYS+SCH_ROUND_KEYS)]
    assert child["environment"]["thread_env"] == golden["environment"]["thread_env"]


# ---- gate 2 ------------------------------------------------------------------------------------------------------------

@BITWISE
def test_uniform_per_level_caps_equal_max_accepted(runs):
    plain = runs("plain2", *COMMON, "--max-accepted", "2")
    per = runs("per2", *COMMON, "--max-accepted-per-level", "2,2,2,2")
    assert RD.run_digest(plain) == RD.run_digest(per) == GOLDEN_8A4D060_RUN_PAIR["def_explicit"]
    config = read(per/"config.json")
    assert config["args"]["max_accepted"] is None and config["args"]["max_accepted_per_level"] == [2, 2, 2, 2]
    assert config["solver"]["caps"]["max_accepted_per_round"] == 2
    assert config["solver"]["max_accepted_source"] == "max_accepted_per_level"


@BITWISE
def test_uniform_per_level_caps_equal_max_accepted_torch64_cpu(runs):
    torch = ("--backend", "torch64", "--device", "cpu", "--cpu-threads", "1")
    plain = runs("plain2_t64", *COMMON, "--max-accepted", "2", *torch)
    per = runs("per2_t64", *COMMON, "--max-accepted-per-level", "2,2,2,2", *torch)
    assert RD.run_digest(plain) == RD.run_digest(per)
    assert read(per/"status.json")["status"] == read(plain/"status.json")["status"]


# ---- gate 3 ------------------------------------------------------------------------------------------------------------

def test_per_level_caps_option_a(runs):
    out = runs("per1234", "--rounds", "1,1,1,2", "--max-accepted-per-level", "1,2,3,4")
    hyper = read(out/"hyper.json")
    rounds = hyper["rounds"]
    assert all(r["status"] == "cap_reached" for r in rounds), [r["status"] for r in rounds]       # the premise
    assert [(r["level"], r["round"], r["accepted"], r["cap_accepted"], r["cap"]) for r in rounds] == [
        (0, 0, 1, 1, "max_accepted"), (1, 0, 2, 2, "max_accepted"), (2, 0, 3, 3, "max_accepted"),
        (3, 0, 4, 4, "max_accepted"), (3, 1, 4, 4, "max_accepted")]
    assert all(r["accepted_committed"] == r["accepted"] for r in rounds)
    assert hyper["algorithm"]["max_accepted_per_level"] == [1, 2, 3, 4]
    assert hyper["algorithm"]["caps"]["max_accepted_per_round"] is None
    config = read(out/"config.json")
    assert config["args"]["max_accepted"] is None and config["solver"]["caps"]["max_accepted_per_round"] is None


def test_per_level_caps_option_b(runs):
    out = runs("per_b", "--rounds", "1,1,1,2", "--outer", "b", "--max-accepted-per-level", "7,6,2,3")
    hyper = read(out/"hyper.json")
    rounds = hyper["rounds"]
    assert all(r["status"] == "cap_reached" and r["cap"] == "max_accepted" for r in rounds), \
        [(r["status"], r["cap"]) for r in rounds]                                                    # the premise
    caps = {0: 7, 1: 6, 2: 2, 3: 3}
    assert [(r["level"], r["accepted"]) for r in rounds] == [(r["level"], caps[r["level"]]) for r in rounds]
    n_in = hyper["algorithm"]["proximal"]["n_in"]
    rows = [json.loads(line) for line in (out/"log.jsonl").read_text().splitlines()]
    for r in rounds:
        accepted = 0
        checks = [c for c in rows if c["kind"] == "commit_check" and c["level"] == r["level"]
                  and c["round"] == r["round"]]
        for check in checks:
            assert check["inner_status"] == "max_iterations" and check["committed"]                  # the premise
            assert check["inner_accepted"] == min(n_in, caps[r["level"]]-accepted)
            accepted += check["inner_accepted"]
        assert accepted == r["accepted"] == r["accepted_committed"]


# ---- gate 4 ------------------------------------------------------------------------------------------------------------

BAD = [
    ("--max-accepted", "2", "--max-accepted-per-level", "2,2,2,2"),
    ("--max-accepted", "500", "--max-accepted-per-level", "1,2,3,4"),          # explicit 500 counts as given
    ("--max-accepted-per-level", "1,2,3"), ("--max-accepted-per-level", "1,2,3,4,5"),
    ("--max-accepted-per-level", "-1,2,3,4"), ("--max-accepted-per-level", "+3,1,1,1"),
    ("--max-accepted-per-level", " 3,1,1,1"), ("--max-accepted-per-level", "3_0,1,1,1"),
    ("--max-accepted-per-level", "1,2,3,4 "), ("--max-accepted-per-level", "1,,3,4"),
    ("--max-accepted-per-level", "a,1,1,1"), ("--max-accepted-per-level", "１,1,1,1"),
    ("--max-accepted-per-level", "3,3,3,3\n"), ("--max-accepted-per-level", "1.0,1,1,1"),
    ("--level-time-shares", "0.1,0.1,0.1"),                                       # no --wall-seconds
    ("--wall-seconds", "100", "--level-time-shares", "0.1,0.2"),
    ("--wall-seconds", "100", "--level-time-shares", "0.1,0.2,0.3,0.1"),
    ("--wall-seconds", "100", "--level-time-shares", "0.1,x,0.2"),
    ("--wall-seconds", "100", "--level-time-shares", "0,0.1,0.1"),
    ("--wall-seconds", "100", "--level-time-shares", "-0.1,0.1,0.1"),
    ("--wall-seconds", "100", "--level-time-shares", "1,0.1,0.1"),
    ("--wall-seconds", "100", "--level-time-shares", "0.5,0.3,0.2"),            # sum = 1
    ("--wall-seconds", "100", "--level-time-shares", "0.4,0.4,0.3"),
    ("--wall-seconds", "100", "--level-time-shares", "nan,0.1,0.1"),
    ("--wall-seconds", "100", "--level-time-shares", "inf,0.1,0.1"),
]


@pytest.mark.parametrize("extra", BAD, ids=[" ".join(b) for b in BAD])
def test_refused_in_parse_args_without_a_run_directory(tmp_path, extra, capsys):
    out = tmp_path/"never"
    argv = ["--pair", "s/p", "--channel-source", "tissue", "--out", str(out), "--data-root", str(tmp_path/"nodata"),
            *extra]
    with pytest.raises(SystemExit) as info:
        run_pair.parse_args(argv)
    assert info.value.code == 2
    with pytest.raises(SystemExit) as info:
        run_pair.main(argv)
    assert info.value.code == 2 and not out.exists() and not (tmp_path/"nodata").exists()


def test_accepted_values_and_args_records():
    base = ["--pair", "s/p", "--channel-source", "tissue", "--out", "x"]
    plain = run_pair.parse_args(base)
    assert plain.max_accepted == 500 and not hasattr(plain, "max_accepted_per_level")
    assert not hasattr(plain, "level_time_shares") and not hasattr(plain, run_pair.SCH_SOURCE)
    assert run_pair.args_record(plain) == vars(plain) and run_pair.schedule_options(plain) == {}
    given = run_pair.parse_args(base+["--max-accepted", "7"])
    assert given.max_accepted == 7 and not hasattr(given, "max_accepted_per_level")
    per = run_pair.parse_args(base+["--max-accepted-per-level", "0,12,3,500"])
    assert per.max_accepted_per_level == [0, 12, 3, 500] and per.max_accepted is None and per.level_time_shares is None
    assert run_pair.solver_caps(per) == Caps.per_level((0, 12, 3, 500))
    shares = run_pair.parse_args(base+["--wall-seconds", "900", "--level-time-shares", "0.1,1e-1,.25"])
    assert shares.level_time_shares == [.1, .1, .25] and shares.max_accepted_per_level is None
    assert shares.max_accepted == 500 and getattr(shares, run_pair.SCH_SOURCE) == "default"
    record = run_pair.args_record(shares)
    assert list(record)[-2:] == ["max_accepted_per_level", "level_time_shares"] and run_pair.SCH_SOURCE not in record
    explicit = run_pair.parse_args(base+["--max-accepted", "500", "--wall-seconds", "900", "--level-time-shares",
                                         "0.1,0.1,0.1"])
    assert getattr(explicit, run_pair.SCH_SOURCE) == "max_accepted"
    options = run_pair.schedule_options(explicit)
    assert options["level_time_shares"].deadlines == (90., (.1+.1)*900., ((.1+.1)+.1)*900.)


# ---- gate 10 ----------------------------------------------------------------------------------------------------------

@BITWISE
def test_shares_never_due_give_the_run_without_shares(runs):
    plain = runs("wall_only", *COMMON, "--max-accepted", "2", "--wall-seconds", "1e9")
    shares = runs("wall_shares", *COMMON, "--max-accepted", "2", "--wall-seconds", "1e9", "--level-time-shares",
                  "0.2,0.2,0.2")
    assert RD.run_digest(plain) == RD.run_digest(shares) == GOLDEN_8A4D060_RUN_PAIR["def_explicit"]
    hyper = read(shares/"hyper.json")
    assert all(r["ended_by_level_share"] is False and r["level_share_hit"] is None for r in hyper["rounds"])
    assert read(shares/"config.json")["solver"]["max_accepted_source"] == "max_accepted"


# ---- gate 12 ----------------------------------------------------------------------------------------------------------

def test_other_caps_callers_are_unchanged():
    assert Caps(max_accepted=1, max_outer=60).record() == dict(max_accepted_per_round=1, max_outer_commits_per_round=60)
    assert Caps(3, 2).record() == dict(max_accepted_per_round=3, max_outer_commits_per_round=2)
    assert Caps().record() == dict(max_accepted_per_round=500, max_outer_commits_per_round=60)
    caps = Caps(max_accepted=1)
    assert caps.for_level(0) is caps and caps.max_accepted_per_level is None
    probe = load("sch_probe", "experiments/eta/probe.py")
    assert probe.PROBE_CAPS.record() == dict(max_accepted_per_round=1, max_outer_commits_per_round=60)
    assert probe.PROBE_CAPS.for_level(3) is probe.PROBE_CAPS
    from bayesreg.learn import cli
    from . import _synthetic as syn
    args = cli.parse_args(["--out", "o", "--pairs-root", "p", "--eta0-json", "e"])
    config = cli.solver_config(args, syn.cube_grid(9))
    assert config.sch_records is False and not set(SCH_SOLVER_KEYS) & set(config.record())
    assert config.caps == Caps(500, 60) and config.record()["caps"] == Caps(500, 60).record()
    l0 = load("sch_l0_run", "experiments/cvg/l0_run.py")
    assert hasattr(l0, "main")
    demo = load("sch_jdc_demo", "experiments/jdc/demo_cell_estimate.py")
    assert hasattr(demo, "main")


# ---- gate 14 ----------------------------------------------------------------------------------------------------------

class Clock:
    def __init__(self):
        self.t = 0.

    def __call__(self):
        return self.t


def share_run(runs, monkeypatch, name, *options):
    """run_pair with an injected budget clock that jumps to 15 s (D_0 = 10 < 15 < D_1 = 20) at the first level-0
    accepted step."""
    clock = Clock()
    monkeypatch.setattr(run_pair, "make_budget", lambda seconds: WallBudget(seconds, clock=clock))
    real = run_pair.solve_multilevel

    def spying(backend, inputs, *, log, **kwargs):
        def spy(row):
            log(row)
            if row.get("kind") == "accept" and row.get("level") == 0:
                clock.t = 15.
        return real(backend, inputs, log=spy, **kwargs)
    monkeypatch.setattr(run_pair, "solve_multilevel", spying)
    return runs(name, *options)


def log_rows(out):
    return [json.loads(line) for line in (out/"log.jsonl").read_text().splitlines()]


def test_records_with_a_share_ending(runs, monkeypatch, golden):
    out = share_run(runs, monkeypatch, "share14", "--rounds", "2,1,1,1", "--max-accepted-per-level", "2,2,2,2",
                    "--wall-seconds", "100", "--level-time-shares", "0.1,0.1,0.1")
    config, prov, hyper = read(out/"config.json"), read(out/"provenance.json"), read(out/"hyper.json")
    timing, status, qc = read(out/"timing.json"), read(out/"status.json"), read(out/"qc.json")
    assert config["args"]["max_accepted_per_level"] == [2, 2, 2, 2] and config["args"]["level_time_shares"] == [.1]*3
    assert config["args"]["max_accepted"] is None
    deadlines = [.1*100., (.1+.1)*100., ((.1+.1)+.1)*100.]
    for record in (config["solver"], prov["solver"], hyper["algorithm"]):
        assert {k: record[k] for k in SCH_SOLVER_KEYS} == dict(
            max_accepted_per_level=[2, 2, 2, 2], max_accepted_source="max_accepted_per_level",
            level_time_shares=[.1, .1, .1], level_deadlines_seconds=deadlines)
        assert record["caps"]["max_accepted_per_round"] == 2
    switches = list(prov["switches"])
    k = switches.index("wall_seconds")
    assert switches[k+1:k+3] == ["max_accepted_per_level", "level_time_shares"]
    assert prov["switches"]["max_accepted_per_level"] == [2, 2, 2, 2] and prov["switches"]["level_time_shares"] == [.1]*3
    rows = log_rows(out)
    round_rows = [r for r in rows if r["kind"] == "round"]
    assert len(round_rows) == len(hyper["rounds"]) == 4                   # level 0 round 1 skipped
    for row, logged in zip(hyper["rounds"], round_rows):
        assert all(key in row and key in logged for key in SCH_ROUND_KEYS)
        assert row["ended_by_level_share"] == (row["status"] == LEVEL_BUDGET_REACHED)
        assert row["level_elapsed_seconds"] is not None and row["run_elapsed_seconds"] is not None
        assert row["cap_accepted"] == 2
    first = hyper["rounds"][0]
    assert first["status"] == LEVEL_BUDGET_REACHED and first["rounds_skipped"] == 1
    assert first["level_share_hit"]["elapsed_seconds"] >= first["level_deadline_seconds"] == 10.
    assert first["level_time_share"] == .1 and first["level_cumulative_share"] == .1
    assert hyper["rounds"][-1]["level"] == 3 and hyper["rounds"][-1]["level_time_share"] is None
    budget_rows = [r for r in rows if r["kind"] == "level_budget"]
    assert len(budget_rows) == 1 and budget_rows[0]["level"] == 0 and budget_rows[0]["round"] == 0
    assert budget_rows[0]["level_deadline_seconds"] == 10. and budget_rows[0]["elapsed_seconds"] >= 10.
    assert budget_rows[0]["check"] == first["level_share_hit"]["check"]
    levels = timing["solve_levels"]["levels"]
    assert [(e["level"], e["deadline_seconds"], e["ended_by_level_share"]) for e in levels] == [
        (0, 10., True), (1, 20., False), (2, (.1+.1+.1)*100., False), (3, None, False)]
    # qc.json and status.json never gain a key (C16 SCH): the key sets of the golden's --wall-seconds run
    keys = R.key_sets({name: (out/name).read_text(encoding="utf-8") for name in R.FILES})
    assert keys["qc.json"] == golden["arms"]["wall"]["key_sets"]["qc.json"]
    assert keys["status.json"] == golden["arms"]["wall"]["key_sets"]["status.json"]
    assert status["status"] == "cap_reached" and qc["sigma_collapse_suspected"] is not None
    assert [t for t in hyper["sigma_monitor"]["trace"] if t["kind"] == "m_step" and t["level"] == 0][0]["used"] is False


def test_records_without_wall_seconds_have_null_elapsed(runs):
    out = runs("per_only", *COMMON, "--max-accepted-per-level", "1,1,2,2")
    hyper = read(out/"hyper.json")
    for row in hyper["rounds"]:
        assert row["level_elapsed_seconds"] is None and row["run_elapsed_seconds"] is None
        assert row["level_time_share"] is None and row["ended_by_level_share"] is False and row["rounds_skipped"] == 0
    config = read(out/"config.json")
    assert config["args"]["level_time_shares"] is None and config["solver"]["level_time_shares"] is None
    assert config["solver"]["level_deadlines_seconds"] is None
    levels = read(out/"timing.json")["solve_levels"]["levels"]
    assert all(e["deadline_seconds"] is None and e["ended_by_level_share"] is False for e in levels)


def test_shares_only_records_the_default_caps(runs):
    out = runs("shares_only", *COMMON, "--max-accepted", "2", "--wall-seconds", "1e9", "--level-time-shares",
               "0.2,0.2,0.2")
    config = read(out/"config.json")
    assert config["args"]["max_accepted"] == 2 and config["args"]["max_accepted_per_level"] is None
    assert config["solver"]["max_accepted_per_level"] == [2, 2, 2, 2]
    assert config["solver"]["caps"]["max_accepted_per_round"] == 2


def test_status_vocabulary_is_closed_in_the_records(runs, monkeypatch):
    out = runs("per1234", "--rounds", "1,1,1,2", "--max-accepted-per-level", "1,2,3,4")
    statuses = {r["status"] for r in read(out/"hyper.json")["rounds"]}
    from bayesreg.solve import stopping as st
    assert statuses <= set(st.ROUND_STATUSES) and read(out/"status.json")["status"] in st.RUN_STATUSES
    assert SolverConfig().sch_records is False
