"""The option arms B1 (runs/20261103-b1/plan.md, docs/bayesreg/05_D0协议.md §15; work package B1): experiments/d0/b1_spec.py, check_prepared.py --b1-spec,
evaluate.py summarize-b1 and run-b1, run_queue.sh --b1, on synthetic files only (tmp_path; no data/, no real image, no solver, no GPU).

- the specification: the repository file is the brief's table; everything else is refused; the watchdog grows with the budget;
- check_prepared lets through exactly what the entry lists, on the real configuration C reference; without the entry nothing changes;
- the selection: every branch of the plan (qualified, best, cheaper, gain, none qualified, not complete, not decidable); the readings on 19 pairs and B1W_A5;
- summarize-b1: seg35 and seg4 together, the groups, the baseline from the long.csv of its batch, the earlier summary kept;
- run_queue.sh --b1: order, commands, watchdog, gate, start checks; a synthetic frozen script and systemd-run run the whole queue: 5 pairs, then the other 14
  without rerunning or overwriting anything; a failure is kept;
- the other modes: --dry-run and one whole --a5 queue compared with the script of commit 82a9db7, before this package.
"""
import csv
import json
import os
import re
import shlex
import shutil
import subprocess

import numpy as np
import pytest

from jd_curl.three_d.nifti import file_sha256

from ..io_seg._helpers import write_json
from .test_d0_a4_tools import EV, FAKE_PAIRS, QUEUE, ROOT, load, queue_tree  # noqa: F401
from .test_d0_a5_tools import a5_tree, metric_values  # noqa: F401

SPEC = load("d0_b1_spec", ROOT/"experiments"/"d0"/"b1_spec.py")
CHECK = load("b1_check_prepared", ROOT/"experiments"/"d0"/"check_prepared.py")
SELECTION = ROOT/"runs"/"20261103-b1"/"arms_selection.json"
A5P_ROOT = "/home/rmcfz/seq_vp_tag_results/d0_v1_a5/inputs/A5P"
BEFORE = "82a9db7"                                                  # main before this package
REFERENCE = json.loads((ROOT/"experiments"/"d0"/"configC_reference.json").read_text())
ARMS = ("B1T", "B1M", "B1P", "B1TM")
GW = [1., 1., .1, .01]
IT = [50, 50, 100, 240]


def entry(arm="B1X", inputs_root="/tmp/inputs", seconds=90, changes=None, **more):
    return dict(arm=arm, inputs_root=inputs_root, seconds=seconds, changes={} if changes is None else changes, **more)


def write_spec(path, *entries):
    write_json(path, dict(schema=SPEC.SCHEMA, arms=list(entries)))
    return path


# ---- the specification --------------------------------------------------------------------------------------------------

def test_repository_specification_is_the_brief():
    got = SPEC.load_spec(SELECTION)
    assert [e["arm"] for e in got] == list(ARMS)
    assert all(e["inputs_root"] == A5P_ROOT and e["expansion"] is False for e in got)
    assert {e["arm"]: (e["seconds"], e["changes"]) for e in got} == {
        "B1T": (330, {}), "B1M": (90, dict(geometry_weight_levels=GW)), "B1P": (90, dict(pre_optimizer_iterations=IT)),
        "B1TM": (330, dict(geometry_weight_levels=GW))}


@pytest.mark.parametrize("bad,fragment", [
    (entry(changes=dict(curvature_weight_levels=[1, 1, 1, 1])), "not options of this tool"),
    (entry(changes=dict(geometry_weight_levels=[1, 1, .1, .03])), "not a change"),
    (entry(changes=dict(pre_optimizer_iterations=[50, 50, 100, 60])), "not a change"),
    (entry(changes=dict(geometry_weight_levels=[1, 1, .1])), "four positive reals"),
    (entry(changes=dict(geometry_weight_levels=[1, 1, .1, 0])), "four positive reals"),
    (entry(changes=dict(geometry_weight_levels="1,1,.1,.01")), "four positive reals"),
    (entry(changes=dict(pre_optimizer_iterations=[50, 50, 100, 2.5])), "four positive integers"),
    (entry(changes=[]), "changes must be an object"),
    (entry(seconds=91), "not a budget"),
    (entry(seconds=True), "not a budget"),
    (entry(inputs_root="relative/path"), "absolute"),
    (entry(arm="B1 T"), "bad arm name"),
    (entry(arm="A5P"), "earlier arm"),
    (entry(expansion="yes"), "expansion must be"),
    (dict(arm="B1X", seconds=90, changes={}), "missing keys"),
    (entry(lr=.5), "unknown keys"),
])
def test_specification_refusals(tmp_path, bad, fragment):
    with pytest.raises(SPEC.SpecError, match=fragment):
        SPEC.load_spec(write_spec(tmp_path/"s.json", bad))


def test_specification_file_refusals(tmp_path):
    with pytest.raises(SPEC.SpecError, match="duplicate"):
        SPEC.load_spec(write_spec(tmp_path/"s.json", entry(), entry()))
    write_json(tmp_path/"t.json", dict(schema="other", arms=[entry()]))
    with pytest.raises(SPEC.SpecError, match="specification"):
        SPEC.load_spec(tmp_path/"t.json")
    with pytest.raises(SPEC.SpecError):
        SPEC.load_spec(tmp_path/"missing.json")
    write_json(tmp_path/"u.json", dict(schema=SPEC.SCHEMA, arms=[]))
    with pytest.raises(SPEC.SpecError, match="at least one arm"):
        SPEC.load_spec(tmp_path/"u.json")


@pytest.mark.parametrize("seconds,watchdog", [(90, 300), (330, 1100), (120, 400), (600, 2000), (60, 300), (30, 300)])
def test_watchdog_grows_with_the_budget(seconds, watchdog):
    assert SPEC.watchdog_seconds(seconds) == watchdog


def test_budgets_are_the_frozen_parsers():
    frozen = "/home/rmcfz/seq_vp_tag_results/frozen_eb808be/scripts/run_stage_3d_e48_pg.py"
    if not os.path.exists(frozen):
        pytest.skip("no frozen worktree")
    text = open(frozen).read()
    assert f"BUDGETS = {SPEC.BUDGETS}" in text


def test_shell_lines_and_method_names(tmp_path):
    lines = SPEC.shell_lines(SELECTION)
    assert lines[0].split("\t")[:7] == ["B1T", A5P_ROOT, "330", "1100", "-", "-", "0"]
    assert lines[1].split("\t")[:7] == ["B1M", A5P_ROOT, "90", "300", "1,1,0.1,0.01", "-", "0"]
    assert lines[2].split("\t")[:7] == ["B1P", A5P_ROOT, "90", "300", "-", "50/50/100/240", "0"]
    names = {e["arm"]: SPEC.method_name(e) for e in SPEC.load_spec(SELECTION)}
    assert names == dict(B1T="frozen_eb808be_b1_330s", B1M="frozen_eb808be_b1_90s_gw1_1_0.1_0.01", B1P="frozen_eb808be_b1_90s_preit50_50_100_240",
                         B1TM="frozen_eb808be_b1_330s_gw1_1_0.1_0.01")
    assert all("configC" not in n for n in names.values())


# ---- check_prepared: an intended deviation, only what the entry lists -----------------------------------------------------

@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """A prepared run directory on the real reference protocol; the protocol is what the entry says unless a test changes it."""
    run = tmp_path/"run"
    (run/"inputs"/"p").mkdir(parents=True)
    (run/"snapshot").mkdir()
    (run/"inputs"/"p"/"pair.npz").write_bytes(b"pair")
    manifest = run/"inputs"/"p"/"manifest.json"
    pair_sha = CHECK.sha(run/"inputs"/"p"/"pair.npz")
    manifest.write_text(json.dumps(dict(pair="p", pair_sha256=pair_sha)))
    manifest_sha = CHECK.sha(manifest)
    (run/"snapshot"/"a.py").write_text("x\n")
    source = {"a.py": CHECK.sha(run/"snapshot"/"a.py")}
    ref_path = tmp_path/"ref.json"
    ref_path.write_text(json.dumps(dict(reference_protocol=REFERENCE["reference_protocol"], freeze_source_py_sha256=source)))
    monkeypatch.setattr(CHECK, "REF", ref_path)

    def go(e, *, as_entry="same", change=None, **task_change):
        rp = SPEC.expected_protocol(REFERENCE["reference_protocol"], SPEC.check_entry(e))
        doc = dict(rp, stage="B1-x", task=dict(rp["task"], pair="p", **task_change), input_sha256={"pair.npz": pair_sha, "manifest.json": manifest_sha},
                   pair_source="data/three_d/stage_L2R_val_v1/p", source_sha256=source)
        doc.update(change or {})
        (run/"protocol.json").write_text(json.dumps(doc))
        checked = SPEC.check_entry(e) if as_entry == "same" else as_entry
        return CHECK.check(run, "p", "B1-x", pair_sha, manifest_sha, b1_entry=checked)
    return go


def test_expected_protocol_changes_only_the_listed_items():
    rp = REFERENCE["reference_protocol"]
    assert SPEC.expected_protocol(rp, SPEC.check_entry(entry())) == rp                   # 90 s, no change: configuration C itself
    t = SPEC.expected_protocol(rp, SPEC.check_entry(entry(seconds=330)))
    assert t["task"]["wall_seconds"] == 330 and t["task"]["name"] == "pg_weak__four__330s"
    assert {k for k in rp if t[k] != rp[k]} == {"task"}
    m = SPEC.expected_protocol(rp, SPEC.check_entry(entry(changes=dict(geometry_weight_levels=GW))))
    assert {k for k in rp if m[k] != rp[k]} == {"geometry_weight_levels", "scope"} and m["geometry_weight_levels"] == GW
    assert "mu=nu=[1.0, 1.0, 0.1, 0.01]" in m["scope"] and "mu=nu=[1.0, 1.0, 0.1, 0.03]" not in m["scope"]
    p = SPEC.expected_protocol(rp, SPEC.check_entry(entry(changes=dict(pre_optimizer_iterations=IT))))
    assert {k for k in rp if p[k] != rp[k]} == {"pre_optimizer", "scope"} and p["pre_optimizer"]["iterations"] == IT
    assert {k for k in rp["pre_optimizer"] if p["pre_optimizer"][k] != rp["pre_optimizer"][k]} == {"iterations"}
    assert "'iterations': (50, 50, 100, 240)" in p["scope"] and "'iterations': (50, 50, 100, 60)" not in p["scope"]
    assert rp["geometry_weight_levels"] == [1., 1., .1, .03] and rp["pre_optimizer"]["iterations"] == [50, 50, 100, 60]      # the reference is untouched


def test_config_c_table_is_the_reference():
    rp = REFERENCE["reference_protocol"]
    assert SPEC.CONFIG_C == dict(seconds=rp["task"]["wall_seconds"], geometry_weight_levels=rp["geometry_weight_levels"],
                                 pre_optimizer_iterations=rp["pre_optimizer"]["iterations"])


@pytest.mark.parametrize("e", [entry(seconds=330), entry(changes=dict(geometry_weight_levels=GW)), entry(changes=dict(pre_optimizer_iterations=IT)),
                               entry(seconds=330, changes=dict(geometry_weight_levels=GW))])
def test_check_prepared_admits_the_listed_items(prepared, e):
    result = prepared(e)
    assert result["ok"] is True, result
    assert result["b1_intended_deviation"] == dict(seconds=e["seconds"], changes=e["changes"])


T = SPEC.check_entry(entry(seconds=330))
M = SPEC.check_entry(entry(changes=dict(geometry_weight_levels=GW)))
P = SPEC.check_entry(entry(changes=dict(pre_optimizer_iterations=IT)))


@pytest.mark.parametrize("written,listed,change,fragment", [
    (entry(seconds=330), M, None, "protocol.task"),                                       # the budget was not listed
    (entry(changes=dict(geometry_weight_levels=GW)), T, None, "geometry_weight_levels"),  # the weights were not listed
    (entry(changes=dict(pre_optimizer_iterations=IT)), T, None, "pre_optimizer"),         # the iterations were not listed
    (entry(changes=dict(geometry_weight_levels=[1, 1, .1, .02])), M, None, "geometry_weight_levels"),   # another value
    (entry(changes=dict(pre_optimizer_iterations=[50, 50, 100, 120])), P, None, "pre_optimizer"),
    (entry(changes=dict(geometry_weight_levels=GW)), M, dict(curvature_weight_levels=[1e-3, 1e-3, 1e-3, 1e-3]), "curvature_weight_levels"),
    (entry(changes=dict(geometry_weight_levels=GW)), M, dict(export_reserve_seconds=2.), "export_reserve_seconds"),
    (entry(changes=dict(geometry_weight_levels=GW)), M, dict(level_shares=[.1, .1, .1, 1.]), "level_shares"),
    (entry(changes=dict(geometry_weight_levels=GW)), M, dict(pre_optimizer=REFERENCE["reference_protocol"]["pre_optimizer"] | dict(lr=.25)), "pre_optimizer"),
    (entry(changes=dict(pre_optimizer_iterations=IT)), P, dict(pre_optimizer=REFERENCE["reference_protocol"]["pre_optimizer"] | dict(iterations=IT, seconds=120.)), "pre_optimizer"),
    (entry(changes=dict(geometry_weight_levels=GW)), M, dict(scope="other"), "scope"),
    (entry(seconds=330), T, dict(native_reference="identity"), "native_reference"),
])
def test_check_prepared_refuses_what_is_not_listed(prepared, written, listed, change, fragment):
    result = prepared(written, as_entry=listed, change=change)
    assert result["ok"] is False and any(fragment in p for p in result["problems"]), result


def test_check_prepared_task_name_must_follow_the_budget(prepared):
    result = prepared(entry(seconds=330), name="pg_weak__four__90s")
    assert result["ok"] is False and any("protocol.task" in p for p in result["problems"])


def test_check_prepared_without_the_entry_stays_configuration_c(prepared):
    assert prepared(entry())["ok"] is True                                                # no change, with an entry
    for e in (entry(seconds=330), entry(changes=dict(geometry_weight_levels=GW)), entry(changes=dict(pre_optimizer_iterations=IT))):
        result = prepared(e, as_entry=None)
        assert result["ok"] is False and "b1_intended_deviation" not in result, result


def test_check_prepared_cli_b1_options(tmp_path):
    common = [str(tmp_path), "--pair", "p", "--stage", "s", "--expect-pair-sha256", "a", "--expect-manifest-sha256", "b"]
    for extra in (["--b1-spec", "x.json"], ["--b1-arm", "B1T"], ["--b1-spec", "x.json", "--b1-arm", "B1T", "--native-reference", "identity"]):
        with pytest.raises(SystemExit):
            CHECK.main(common+extra)


def test_reference_file_is_unchanged():
    old = subprocess.run(["git", "-C", str(ROOT), "show", f"{BEFORE}:experiments/d0/configC_reference.json"], capture_output=True)
    assert old.returncode == 0 and old.stdout == (ROOT/"experiments"/"d0"/"configC_reference.json").read_bytes()


# ---- the selection -------------------------------------------------------------------------------------------------------

NAMES = [f"OASIS_OAS1_{900+i:04d}_MR1__OASIS_OAS1_{901+i:04d}_MR1" for i in range(19)]
BASE35, BASE4 = EV.B1_PINS["A5P"]
CANDIDATES = [SPEC.check_entry(e) for e in (entry("B1T", seconds=330), entry("B1M", changes=dict(geometry_weight_levels=GW)),
                                            entry("B1P", changes=dict(pre_optimizer_iterations=IT)),
                                            entry("B1TM", seconds=330, changes=dict(geometry_weight_levels=GW)))]


def fake(seg35, *, seg4=None, sdlogj=.08, passed=True, entry_of=None):
    m = metric_values(seg35, sdlogj)
    m["seg4"] = BASE4 if seg4 is None else seg4
    return dict(metrics=m, certificate=dict(passed=passed, min_jd_lower_bound=1e-6), self_check=dict(passed=True), classification={"class": "ok", "kind": None, "reason": None},
                native_commits=30, e2e_seconds=88., status="native_budget_exhausted", b1=entry_of)


def world(**values):
    """Baseline on all pairs, each named arm on the first 5, by its seg35 or a dict of fake()'s arguments."""
    runs = {("A5P", p): fake(BASE35) for p in NAMES}
    for arm in ARMS:
        kw = values.get(arm, BASE35+.0001)
        kw = kw if isinstance(kw, dict) else dict(seg35=kw)
        runs.update({(arm, p): fake(**kw) for p in NAMES[:5]})
    return runs


def select(runs, problems=()):
    return EV.decide_b1_selection(runs, NAMES, CANDIDATES, problems)


def test_selection_best_arm_wins():
    d = select(world(B1T=BASE35+.03, B1M=BASE35+.01, B1P=BASE35+.012, B1TM=BASE35+.02))
    assert d["status"] == "winner" and d["winner"] == "B1T" and d["best_arm"] == "B1T" and d["selected_arm"] == "B1T"
    assert d["gain"] == pytest.approx(.03) and d["winner_spec"]["seconds"] == 330 and d["winner_spec"]["inputs_root"] == "/tmp/inputs" and d["pairs"] == NAMES[:5]
    assert all(r["qualified"] for r in d["arms"]) and "胜出 B1T" in d["text"]


def test_selection_cheaper_arm_within_margin_wins():
    d = select(world(B1T=BASE35+.03, B1M=BASE35+.0285, B1P=BASE35+.029, B1TM=BASE35+.01))
    assert d["best_arm"] == "B1T" and d["winner"] == "B1P" and d["winner_spec"]["changes"] == dict(pre_optimizer_iterations=IT)   # the better of the two cheap arms
    d = select(world(B1T=BASE35+.03, B1M=BASE35+.0281, B1P=BASE35+.02, B1TM=BASE35+.01))
    assert d["winner"] == "B1M"


def test_selection_cheaper_margin_is_strict():
    d = select(world(B1T=BASE35+.03, B1M=BASE35+.0279, B1P=BASE35+.02, B1TM=BASE35+.01))           # .0021 apart: the dearer arm stays
    assert d["winner"] == "B1T"


def test_selection_same_price_is_not_cheaper():
    d = select(world(B1T=BASE35+.01, B1M=BASE35+.03, B1P=BASE35+.0295, B1TM=BASE35+.01))           # a 90 s arm is best: the other 90 s arm does not replace it
    assert d["winner"] == "B1M" and d["best_arm"] == "B1M"
    d = select(world(B1T=BASE35+.03, B1M=BASE35+.01, B1P=BASE35+.01, B1TM=BASE35+.0295))           # within .002 but the same 330 s
    assert d["winner"] == "B1T"


def test_selection_tie_goes_to_the_cheaper_arm():
    d = select(world(B1T=BASE35+.03, B1M=BASE35+.03, B1P=BASE35+.01, B1TM=BASE35+.01))
    assert d["best_arm"] == "B1M" and d["winner"] == "B1M"


@pytest.mark.parametrize("gain,winner,status", [(.0049, "configC", "no_clear_gain"), (.0051, "B1T", "winner"), (.0, "configC", "no_clear_gain"), (-.02, "configC", "no_clear_gain")])
def test_selection_small_gain_is_configuration_c(gain, winner, status):
    d = select(world(B1T=BASE35+gain, B1M=BASE35-.1, B1P=BASE35-.1, B1TM=BASE35-.1))
    assert d["status"] == status and d["winner"] == winner and d["selected_arm"] == "B1T"
    if winner == "configC":
        assert "选项没有明显提升" in d["text"] and d["winner_spec"]["arm"] == "configC" and d["winner_spec"]["seconds"] == 90 and d["winner_spec"]["changes"] == {}


def test_selection_gain_is_read_on_the_selected_arm():
    d = select(world(B1T=BASE35+.0055, B1M=BASE35+.0045, B1P=BASE35-.1, B1TM=BASE35-.1))           # the cheaper arm replaces it, and its gain is too small
    assert d["best_arm"] == "B1T" and d["selected_arm"] == "B1M" and d["winner"] == "configC" and d["status"] == "no_clear_gain"


@pytest.mark.parametrize("case,why", [
    (dict(seg35=.9, sdlogj=.0951), "sdlogj_ok"),
    (dict(seg35=.9, seg4=BASE4-.0031), "seg4_ok"),
    (dict(seg35=.9, passed=False), "certificates_ok"),
])
def test_selection_conditions_of_a_qualified_arm(case, why):
    d = select(world(B1T=case, B1M=BASE35+.01, B1P=BASE35+.009, B1TM=BASE35+.008))
    row = next(r for r in d["arms"] if r["arm"] == "B1T")
    assert row["qualified"] is False and row[why] is False and d["winner"] == "B1M"


def test_selection_just_inside_the_thresholds():
    d = select(world(B1T=dict(seg35=.9, sdlogj=.09499, seg4=BASE4-.0029), B1M=BASE35, B1P=BASE35, B1TM=BASE35))
    assert next(r for r in d["arms"] if r["arm"] == "B1T")["qualified"] is True and d["winner"] == "B1T"


def test_selection_one_failed_pair_disqualifies_the_arm():
    runs = world(B1T=.9, B1M=BASE35+.01)
    runs[("B1T", NAMES[2])] = dict(metrics=None, classification={"class": "method_failure"})
    d = select(runs)
    assert next(r for r in d["arms"] if r["arm"] == "B1T")["fields_ok"] is False and d["winner"] == "B1M"
    runs[("B1T", NAMES[2])] = fake(.9, passed=False)
    assert select(runs)["winner"] == "B1M"


def test_selection_no_qualified_arm_is_configuration_c():
    d = select(world(**{a: dict(seg35=.9, sdlogj=.2) for a in ARMS}))
    assert d["status"] == "no_qualified" and d["winner"] == "configC" and d["best_arm"] is None and "没有合格臂" in d["text"]


@pytest.mark.parametrize("missing", [("B1P", NAMES[0]), ("A5P", NAMES[4])])
def test_selection_not_complete(missing):
    runs = world(B1T=.9)
    del runs[missing]
    d = select(runs)
    assert d["status"] == "not_complete" and d["winner"] is None and d["winner_spec"] is None


def test_selection_uses_only_the_first_five_pairs():
    runs = world(B1T=BASE35+.03, B1M=BASE35+.01)
    runs.update({("B1M", p): fake(.99) for p in NAMES[5:]})
    runs[("A5P", NAMES[10])] = fake(.5)
    assert select(runs)["winner"] == "B1T"


def test_selection_not_decidable():
    d = select(world(B1T=.9), problems=("x",))
    assert d["status"] == "not_decidable" and d["winner"] is None
    runs = world(B1T=.9)
    for p in NAMES[:5]:
        runs[("A5P", p)] = fake(BASE35+.001)
    assert select(runs)["status"] == "not_decidable"


# ---- the readings --------------------------------------------------------------------------------------------------------

def read(*, winner="B1M", seg35=.86, sdlogj=.08, passed=True, oracle=None, n=19):
    entry_of = dict(seconds=90, changes=dict(geometry_weight_levels=GW))
    selection = dict(winner=winner, winner_spec=dict(arm=winner, **entry_of) if winner not in (None, "configC") else (None if winner is None else dict(EV.B1_CONFIG_C)))
    runs = {(winner, p): fake(seg35, sdlogj=sdlogj, passed=passed) for p in NAMES[:n]} if winner not in (None, "configC") else {}
    if oracle is not None:
        runs.update({("B1W_A5", p): fake(oracle, entry_of=entry_of) for p in NAMES[:5]})
    return runs, selection


@pytest.mark.parametrize("kw,key", [
    (dict(seg35=.86001), "target"),
    (dict(seg35=.8599), "near"),
    (dict(seg35=.85001), "near"),
    (dict(seg35=.8499), "not_enough"),
    (dict(seg35=.87, sdlogj=.1001), "near"),                        # past .86 but over the SDlogJ line: not the target
    (dict(seg35=.87, passed=False), "not_decidable"),               # a certificate failed: no reading (revision 1, M1)
    (dict(seg35=.855, passed=False), "not_decidable"),
    (dict(seg35=.84, passed=False), "not_decidable"),
    (dict(seg35=.87, n=18), "not_complete"),
    (dict(winner="configC"), "no_expansion"),
    (dict(winner=None), "not_decided"),
])
def test_readings(kw, key):
    runs, selection = read(**kw)
    r = EV.decide_b1_readings(runs, NAMES, selection)
    assert r["reading"] == key and EV.B1_READINGS[key] in r["text"]


def test_readings_too_few_final_maps():
    runs, selection = read(seg35=.9)
    for p in NAMES[:3]:
        runs[("B1M", p)] = dict(metrics=None)
    assert EV.decide_b1_readings(runs, NAMES, selection)["reading"] == "not_decidable"
    runs, selection = read(seg35=.9)
    for p in NAMES[:2]:
        runs[("B1M", p)] = dict(metrics=None)
    assert EV.decide_b1_readings(runs, NAMES, selection)["reading"] == "near"       # 17 maps: read, but "all certificates" fails


def test_reading_target_says_it_was_selected_among_five():
    runs, selection = read(seg35=.87)
    r = EV.decide_b1_readings(runs, NAMES, selection)
    assert "从 5 个配置里选出" in r["text"] and "开发集" in r["text"] and "seg4" in r["text"]


@pytest.mark.parametrize("oracle,key", [(.87801, "raised"), (.8779, "not_raised"), (None, "not_run")])
def test_oracle_reading(oracle, key):
    runs, selection = read(oracle=oracle)
    assert EV.decide_b1_readings(runs, NAMES, selection)["oracle"]["reading"] == key


def test_oracle_threshold_is_885_minus_the_a5_offset():
    assert EV.B1_ORACLE_RAISED == pytest.approx(.885-.0070)


def test_oracle_reading_other_configuration_and_failed_pair():
    runs, selection = read(oracle=.9)
    runs[("B1W_A5", NAMES[1])] = fake(.9, entry_of=dict(seconds=330, changes={}))
    assert EV.decide_b1_readings(runs, NAMES, selection)["oracle"]["reading"] == "config_mismatch"
    runs, selection = read(oracle=.9)
    entry_of = runs[("B1W_A5", NAMES[1])]["b1"]
    runs[("B1W_A5", NAMES[1])] = dict(metrics=None, b1=entry_of)
    assert EV.decide_b1_readings(runs, NAMES, selection)["oracle"]["reading"] == "not_decidable"
    runs, selection = read(oracle=.9)
    runs[("B1W_A5", NAMES[1])] = fake(.9, passed=False, entry_of=entry_of)              # revision 1, M1
    assert EV.decide_b1_readings(runs, NAMES, selection)["oracle"]["reading"] == "not_decidable"


def test_oracle_reading_compares_the_inputs_root():
    runs, selection = read(oracle=.9)
    assert EV.decide_b1_readings(runs, NAMES, selection, oracle_inputs="/x/A5")["oracle"]["reading"] == "config_mismatch"
    for p in NAMES[:5]:
        runs[("B1W_A5", p)]["b1"] = dict(runs[("B1W_A5", p)]["b1"], inputs_root="/x/A5")
    assert EV.decide_b1_readings(runs, NAMES, selection, oracle_inputs="/x/A5")["oracle"]["reading"] == "raised"


def test_oracle_reading_text_without_a_winner():
    runs, selection = read(winner="configC", oracle=.9)
    r = EV.decide_b1_readings(runs, NAMES, selection)["oracle"]
    assert r["reading"] == "not_applicable" and "计划不跑 B1W_A5" in r["text"]
    runs, selection = read(winner=None)
    r = EV.decide_b1_readings(runs, NAMES, selection)["oracle"]
    assert r["reading"] == "not_decided" and "选配未判" in r["text"]
    runs, selection = read()
    r = EV.decide_b1_readings(runs, NAMES, selection)["oracle"]
    assert r["reading"] == "not_run" and "还没有跑完" in r["text"]
    assert EV.B1_ORACLE_READINGS["not_raised"] == "B1W_A5 前 5 对的 seg35 未达 .8780 这条线"


def test_selection_flip_to_a_cheaper_arm_is_written_out():
    d = select(world(B1T=BASE35+.0060, B1M=BASE35+.0045, B1P=BASE35-.1, B1TM=BASE35-.1))
    assert d["winner"] == "configC" and d["best_arm"] == "B1T" and d["best_gain"] == pytest.approx(.006) and d["gain"] == pytest.approx(.0045)
    assert "B1T 增益 +0.0060" in d["text"] and "B1M" in d["text"] and "+0.0045" in d["text"]
    d = select(world(B1T=BASE35+.0040, B1M=BASE35+.0030, B1P=BASE35-.1, B1TM=BASE35-.1))           # the best arm was below .005 anyway: nothing to write out
    assert d["winner"] == "configC" and "B1T 增益" not in d["text"]


def test_candidate_row_counts_handover_only():
    runs = world(B1T=.9)
    runs[("B1T", NAMES[1])]["classification"] = {"class": "handover_only", "kind": "no_native_commit", "reason": "x"}
    d = select(runs)
    assert next(r for r in d["arms"] if r["arm"] == "B1T")["n_handover_only"] == 1 and next(r for r in d["arms"] if r["arm"] == "B1M")["n_handover_only"] == 0


def test_b1w_v2_reading():
    winner = dict(winner="B1M", winner_spec=dict(arm="B1M", seconds=90, changes=dict(geometry_weight_levels=GW)))
    config_c = dict(winner="configC", winner_spec=dict(EV.B1_CONFIG_C))
    runs = {("A5P", p): fake(.84, seg4=.90) for p in NAMES}
    runs.update({("B1M", p): fake(.85, seg4=.91) for p in NAMES})
    assert EV.decide_b1_v2(runs, NAMES, winner)["status"] == "not_run" and "还没有跑" in EV.decide_b1_v2(runs, NAMES, winner)["text"]
    b1 = winner["winner_spec"]
    runs.update({("B1W_v2", p): fake(.86, seg4=.93, entry_of=b1) for p in NAMES[:5]})
    r = EV.decide_b1_v2(runs, NAMES, winner)
    assert r["status"] == "read" and r["reference"] == "B1M" and r["seg35"]["mean_diff"] == pytest.approx(.01) and r["seg4"]["mean_diff"] == pytest.approx(.02)
    assert r["seg35"]["n"] == 5 and r["seg35"]["positive"] == 5 and "seg35" in r["text"] and "seg4" in r["text"] and "不是结论" in r["text"] and "B1W_v2 − B1M" in r["text"]
    assert EV.decide_b1_v2(runs, NAMES, config_c)["status"] == "config_mismatch"                     # the winner's configuration, not any
    runs.update({("B1W_v2", p): fake(.86, seg4=.93, entry_of=dict(seconds=90, changes={})) for p in NAMES})
    r = EV.decide_b1_v2(runs, NAMES, config_c)
    assert r["reference"] == "A5P" and r["seg35"]["mean_diff"] == pytest.approx(.02) and r["seg4"]["mean_diff"] == pytest.approx(.03) and r["seg35"]["n"] == 19


# ---- summarize-b1 --------------------------------------------------------------------------------------------------------

def write_run(root, arm, pair, seg35, *, seg4=BASE4, sdlogj=.08, config=None, cls="ok", final=True, passed=True, run_id=None):
    d = root/arm/pair
    d.mkdir(parents=True)
    m = metric_values(seg35, sdlogj)
    m["seg4"] = seg4
    doc = dict(run_id=run_id or f"b1_{arm}_{pair}", run_dir=str(d), arm=arm, pair=pair, method=None if config is None else SPEC.method_name(dict(config, arm=arm)),
               certificate=dict(passed=final and passed, min_jd_lower_bound=1e-6 if final else None), metrics=m if final else None, self_check=dict(passed=True),
               status="native_budget_exhausted", native_commits=30, e2e_seconds=88.5, success=False, evaluator=dict(sha256=f"sha-{arm}"),
               classification={"class": cls, "kind": None, "reason": None})
    if config is not None:
        doc["b1"] = dict(config, arm=arm, spec="s.json", spec_sha256="0")
    write_json(d/"metrics.json", doc)
    (d/"log.txt").write_text("2026-11-03T10:00:00Z launch tag-x; loadavg 0.5 0.9 1.0; x\n")


@pytest.fixture
def batch(tmp_path):
    stage = tmp_path/"stage"
    stage.mkdir()
    write_json(stage/"selection.json", dict(pairs=[dict(pair=p) for p in NAMES]))
    a5 = tmp_path/"a5"
    for arm, (seg35, seg4) in EV.B1_PINS.items():
        for p in NAMES:
            write_run(a5/"runs", arm, p, seg35, seg4=seg4, run_id=f"d0a5_{arm}_{p}")
    (a5/"analysis").mkdir()
    with open(a5/"analysis"/"long.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set", "split", "method", "arm", "pair", "measure", "value", "run_id"])
        w.writeheader()
        w.writerows(EV.long_rows(EV.chosen_runs(a5/"runs", ("A5P", "A5"))))
    spec = write_spec(tmp_path/"selection.json", *[dict(c) for c in CANDIDATES])
    return dict(stage=stage, a5=a5, b1=tmp_path/"b1", spec=spec, runs=tmp_path/"b1"/"runs", out=tmp_path/"b1"/"analysis")


def summarize(b):
    assert EV.summarize_b1(b["b1"], b["spec"], b["a5"], b["out"], stage=b["stage"]) == 0
    return json.loads((b["out"]/"decision_b1.json").read_text()), (b["out"]/"summary_b1.md").read_text()


def write_arms(b, values, pairs=NAMES[:5]):
    configs = {c["arm"]: c for c in CANDIDATES}
    for arm, value in values.items():
        for p in pairs:
            write_run(b["runs"], arm, p, value, config=configs[arm])


def test_summarize_selection_then_expansion(batch, capsys):
    write_arms(batch, dict(B1T=BASE35+.03, B1M=BASE35+.0285, B1P=BASE35+.01, B1TM=BASE35+.02))
    dec, text = summarize(batch)
    assert dec["winner"] == "B1M" and dec["selection"]["status"] == "winner" and dec["winner_spec"]["changes"] == dict(geometry_weight_levels=GW)
    assert dec["readings"]["reading"] == "not_complete" and dec["readings"]["oracle"]["reading"] == "not_run" and dec["reference_problems"] == []
    row = dec["comparisons"]["B1T - A5P (first_5)"]["seg35"]
    assert row["n"] == 5 and row["positive"] == 5 and row["mean_diff"] == pytest.approx(.03) and row["median_diff"] == pytest.approx(.03)
    assert dec["comparisons"]["B1M - A5P (first_5)"]["seg4"]["mean_diff"] == pytest.approx(0)
    assert set(dec["comparisons"]["B1P - A5P (first_5)"]) == {"seg35", "seg4", "hd95", "sdlogj_official", "sdlogj_brain", "pct_J_lt_half_brain"}
    assert set(dec["seg35_groups"]["B1T - A5P (first_5)"]) == {"small_six", "large_six", "other_23"}
    assert dec["seg4_structures"]["B1T - A5P (first_5)"]["3"]["n"] == 5
    assert dec["anomalies"] == [] and dec["sources"]["evaluator_sha256"]["B1T"] == ["sha-B1T"]
    for fragment in ("开发集", "单次观测", "n = 5 只有方向信息", "胜出是挑出来的", "选择效应", "基线", "不是同时段", "A5 用真值", "训练集标签训练的分割器", "0438–0443", "预算收尾不是收敛",
                     "seg35", "seg4", "HD95", "SDlogJ 官方", "脑内 SDlogJ", "脑内 J < .5", "小六个", "大六个", "其余 23 个", "皮层", "白质", "原生提交", "e2e", "证书", "最小下界", "收尾状态",
                     "native_budget_exhausted", "均值差", "中位差", "为正的对数", "受阻或异常", "没有。", "评测脚本版本", "不是结论", "330 s"):
        assert fragment in text, fragment
    rows = list(csv.DictReader(open(batch["out"]/"long.csv", newline="")))
    assert {r["arm"] for r in rows} == set(ARMS) | {"A5P", "A5"} and {r["pair"] for r in rows} == set(NAMES[:5])
    assert {r["method"] for r in rows if r["arm"] == "B1T"} == {"frozen_eb808be_b1_330s"}
    assert not [r for r in rows if r["arm"] in ARMS and "configC" in r["method"]]
    assert not [d for d in batch["out"].iterdir() if d.name.startswith("previous_")] and dec["previous_summary_moved_to"] is None
    first = (batch["out"]/"decision_b1.json").read_bytes()

    write_arms(batch, dict(B1M=.87), pairs=NAMES[5:])
    oracle = dict(CANDIDATES[1], arm="B1W_A5", expansion=True, inputs_root=str(batch["a5"]/"inputs"/"A5"), input_oracle=True)
    for p in NAMES[:5]:
        write_run(batch["runs"], "B1W_A5", p, .88, seg4=.93, config=oracle)
    dec, text = summarize(batch)
    assert dec["winner"] == "B1M" and dec["selection"]["pairs"] == NAMES[:5]                    # the 5 pairs still decide
    assert dec["readings"]["winner_row"]["n_ran"] == 19 and dec["readings"]["winner_row"]["seg35_mean"] == pytest.approx((5*(BASE35+.0285)+14*.87)/19)
    assert dec["readings"]["reading"] == "target"                                                # (5 x .8575 + 14 x .87)/19 = .8667
    assert dec["readings"]["oracle"]["reading"] == "raised" and dec["readings"]["oracle"]["seg35_mean"] == pytest.approx(.88)
    assert dec["comparisons"]["B1M - A5P (all_of_arm)"]["seg35"]["n"] == 19 and dec["comparisons"]["B1W_A5 - A5 (first_5)"]["seg35"]["mean_diff"] == pytest.approx(.88-EV.B1_PINS["A5"][0])
    assert "B1W_A5（oracle，只作上限）" in text and "A5（oracle，只作上限）" in text and "各臂自己跑过的全部配对" in text
    assert {r["method"] for r in csv.DictReader(open(batch["out"]/"long.csv", newline="")) if r["arm"] == "B1W_A5"} == {"frozen_eb808be_b1_90s_gw1_1_0.1_0.01_oracle"}
    assert dec["b1w_v2"]["status"] == "not_run" and "B1W_v2 还没有跑" in text and "近端变化罚" in text and ".009–.020" in text
    kept = [d for d in batch["out"].iterdir() if d.name.startswith("previous_")]
    assert len(kept) == 1 and (kept[0]/"decision_b1.json").read_bytes() == first and dec["previous_summary_moved_to"] == str(kept[0])
    assert (kept[0]/"long.csv").is_file() and (kept[0]/"summary_b1.md").is_file()


def test_summarize_partial_runs_are_not_a_selection(batch):
    write_arms(batch, dict(B1T=.9))
    dec, text = summarize(batch)
    assert dec["winner"] is None and dec["selection"]["status"] == "not_complete" and "没有 metrics.json" in text
    assert {a["arm"] for a in dec["anomalies"]} == {"B1M", "B1P", "B1TM"} and len(dec["anomalies"]) == 15


def test_summarize_reports_failed_runs(batch):
    write_arms(batch, dict(B1M=BASE35+.03, B1P=BASE35+.02, B1TM=BASE35+.02))
    configs = {c["arm"]: c for c in CANDIDATES}
    for p in NAMES[:4]:
        write_run(batch["runs"], "B1T", p, .9, config=configs["B1T"])
    write_run(batch["runs"], "B1T", NAMES[4], 0., config=configs["B1T"], cls="method_failure", final=False)
    dec, text = summarize(batch)
    assert dec["winner"] == "B1M" and "method_failure" in text and "无终态场" in text
    assert next(r for r in dec["selection"]["arms"] if r["arm"] == "B1T")["n_with_field"] == 4


def test_summarize_refuses_runs_of_another_configuration_and_a_changed_baseline(batch):
    write_arms(batch, dict(B1T=.9, B1M=.85, B1P=.85))
    for p in NAMES[:5]:
        write_run(batch["runs"], "B1TM", p, .95, config=dict(CANDIDATES[1], arm="B1TM"))             # 90 s under the 330 s name
    dec, _ = summarize(batch)
    assert dec["selection"]["status"] == "not_decidable" and dec["winner"] is None and "配置" in dec["reference_problems"][0]
    shutil.rmtree(batch["runs"]/"B1TM")
    write_arms(batch, dict(B1TM=.85))
    assert summarize(batch)[0]["selection"]["status"] == "winner"
    path = batch["a5"]/"runs"/"A5P"/NAMES[1]/"metrics.json"
    doc = json.loads(path.read_text())
    doc["metrics"]["seg35"] += .01
    write_json(path, doc)
    dec, text = summarize(batch)
    assert dec["winner"] is None and "参照与预登记或 metrics.json 不符" in text
    (batch["a5"]/"analysis"/"long.csv").unlink()
    with pytest.raises(SystemExit, match="long.csv"):
        summarize(batch)


def test_summarize_refuses_runs_from_other_inputs(batch):
    write_arms(batch, dict(B1T=.9, B1M=.85, B1P=.85))
    for p in NAMES[:5]:
        write_run(batch["runs"], "B1TM", p, .85, config=dict(CANDIDATES[3], inputs_root="/other/inputs"))
    dec, _ = summarize(batch)
    assert dec["selection"]["status"] == "not_decidable" and "/other/inputs" in dec["reference_problems"][0]


@pytest.mark.parametrize("winner_is_config_c", [False, True])
def test_summarize_b1w_v2_section(batch, winner_is_config_c):
    gain = .0001 if winner_is_config_c else .03
    write_arms(batch, dict(B1T=BASE35+gain, B1M=BASE35+gain-.0005, B1P=BASE35, B1TM=BASE35))
    dec, text = summarize(batch)
    assert dec["b1w_v2"]["status"] == "not_run" and "B1W_v2 还没有跑" in text
    config = dict(seconds=90, changes={} if winner_is_config_c else dict(geometry_weight_levels=GW), arm="B1W_v2", inputs_root="/v2/A5P", expansion=True, registry="configs/bayesreg/segmenters_v2.json")
    for p in NAMES[:5]:
        write_run(batch["runs"], "B1W_v2", p, BASE35+.05, seg4=BASE4+.01, config=config)
    dec, text = summarize(batch)
    ref = "A5P" if winner_is_config_c else "B1M"
    assert dec["winner"] == ("configC" if winner_is_config_c else "B1M")
    assert dec["b1w_v2"]["status"] == "read" and dec["b1w_v2"]["reference"] == ref and dec["b1w_v2"]["seg4"]["mean_diff"] == pytest.approx(.01)
    assert f"B1W_v2 − {ref}" in text and f"B1W_v2 对同配置的 {ref}" in text
    assert f"B1W_v2 - {ref} (first_5)" in dec["comparisons"] and dec["comparisons"]["B1W_v2 - A5P (first_5)"]["seg4"]["n"] == 5
    assert "还没有跑" not in dec["b1w_v2"]["text"] and dec["winner"] != "B1W_v2"                  # never a candidate


def test_summary_never_under_a_baseline_batch(batch):
    for root in (batch["a5"], EV.D0, EV.D0A4, EV.D0A1P, EV.D0A5):
        with pytest.raises(SystemExit, match="never written under"):
            EV.summarize_b1(batch["b1"], batch["spec"], batch["a5"], root/"analysis_b1", stage=batch["stage"])


def test_evaluate_run_b1_records_the_configuration(tmp_path, monkeypatch):
    spec = write_spec(tmp_path/"s.json", dict(CANDIDATES[0]))
    seen = {}

    def stub(run, arm, pair, run_id, method=None):
        seen.update(arm=arm, method=method)
        return dict(arm=arm, method=method)
    monkeypatch.setattr(EV, "evaluate_run", stub)
    out = tmp_path/"m.json"
    assert EV.main(["run-b1", str(tmp_path), "--spec", str(spec), "--arm", "B1T", "--pair", "p", "--run-id", "r", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert seen == dict(arm="B1T", method="frozen_eb808be_b1_330s") and doc["b1"]["seconds"] == 330 and doc["b1"]["spec_sha256"] == file_sha256(spec) and doc["b1"]["input_oracle"] is False
    seen.clear()
    assert EV.main(["run", str(tmp_path), "--arm", "A5P", "--pair", "p", "--run-id", "r", "--out", str(out)]) == 0
    assert seen == dict(arm="A5P", method=None) and "b1" not in json.loads(out.read_text())
    with pytest.raises(SystemExit):
        EV.main(["run", str(tmp_path), "--arm", "B1T", "--pair", "p", "--run-id", "r"])                # the plain mode still takes only its arms


# ---- expansion specifications and the gate --------------------------------------------------------------------------------

def decision(path, winner, spec, selection=SELECTION):
    write_json(path, dict(winner=winner, winner_spec=spec, selection=dict(text="t"), readings=dict(text="r"), sources=dict(selection_spec_sha256=file_sha256(selection))))
    return path


def test_expansion_specifications(tmp_path, capsys):
    d = decision(tmp_path/"d.json", "B1M", dict(CANDIDATES[1], inputs_root=A5P_ROOT))
    written = SPEC.expansion_specs(d, tmp_path/"specs")
    assert [p.name for p in written] == ["expand_B1M.json", "B1W_A5.json"]
    w, o = SPEC.load_spec(written[0])[0], SPEC.load_spec(written[1])[0]
    assert w["arm"] == "B1M" and w["inputs_root"] == A5P_ROOT and w["seconds"] == 90 and w["changes"] == dict(geometry_weight_levels=GW) and w["expansion"] is True
    assert o["arm"] == "B1W_A5" and o["inputs_root"].endswith("d0_v1_a5/inputs/A5") and SPEC.configuration(o) == SPEC.configuration(w) and o["expansion"] is True
    assert SPEC.gate(written[0], d, ["B1M"], SELECTION) == 0 and SPEC.gate(written[1], d, ["B1W_A5"], SELECTION) == 0
    SPEC.expansion_specs(d, tmp_path/"specs")                                                      # the same content: fine
    written[0].write_text("{}")
    with pytest.raises(SPEC.SpecError, match="nothing is overwritten"):
        SPEC.expansion_specs(d, tmp_path/"specs")
    assert written[0].read_text() == "{}"


@pytest.mark.parametrize("winner,spec", [("configC", dict(arm="configC", seconds=90, changes={})), (None, None)])
def test_no_expansion_without_a_winner(tmp_path, winner, spec):
    d = decision(tmp_path/"d.json", winner, spec)
    with pytest.raises(SPEC.SpecError, match="no expansion"):
        SPEC.expansion_specs(d, tmp_path/"specs")
    assert not (tmp_path/"specs").exists()


def test_gate(tmp_path, capsys):
    s = write_spec(tmp_path/"s.json", *[dict(c) for c in CANDIDATES])
    assert SPEC.gate(s, tmp_path/"none.json", ["B1M"], SELECTION) == 1                             # no decision yet
    d = decision(tmp_path/"d.json", "B1M", dict(CANDIDATES[1]))
    assert SPEC.gate(s, d, ["B1M"], SELECTION) == 0 and SPEC.gate(s, d, ["B1T"], SELECTION) == 1 and SPEC.gate(s, d, ["B1M", "B1P"], SELECTION) == 1
    d = decision(tmp_path/"d.json", None, None)
    assert SPEC.gate(s, d, ["B1M"], SELECTION) == 1
    d = decision(tmp_path/"d.json", "configC", dict(arm="configC", seconds=90, changes={}))
    assert SPEC.gate(s, d, ["B1M"], SELECTION) == 1
    later = write_spec(tmp_path/"v2.json", entry("B1W_v2"))                                         # a later arm on the winner's configuration
    assert SPEC.gate(later, d, ["B1W_v2"], SELECTION) == 0


def test_gate_compares_the_selection_specification(tmp_path, capsys):
    s = write_spec(tmp_path/"s.json", *[dict(c) for c in CANDIDATES])
    other = write_spec(tmp_path/"other.json", dict(CANDIDATES[0]))
    d = decision(tmp_path/"d.json", "B1M", dict(CANDIDATES[1]), selection=other)                    # another selection specification made this decision
    assert SPEC.gate(s, d, ["B1M"], SELECTION) == 1 and "REFUSED" in capsys.readouterr().out
    assert SPEC.gate(s, d, ["B1M"], other) == 0
    assert SPEC.gate(s, d, ["B1M"], tmp_path/"missing.json") == 1
    doc = json.loads(d.read_text())
    del doc["sources"]
    write_json(d, doc)
    assert SPEC.gate(s, d, ["B1M"], other) == 1


@pytest.mark.parametrize("named", [True, False])
def test_gate_refuses_oracle_when_the_winner_is_config_c(tmp_path, named):
    root = tmp_path/"inputs"
    root.mkdir()
    write_json(root/"d0_inputs.json", dict(arm="A5", oracle=True))
    s = write_spec(tmp_path/"s.json", entry("B1W_A5" if named else "Other", str(root)))
    d = decision(tmp_path/"d.json", "configC", dict(arm="configC", seconds=90, changes={}))
    assert SPEC.gate(s, d, ["B1W_A5" if named else "Other"], SELECTION) == 1


def test_is_gated_does_not_trust_the_specification(tmp_path):
    root = tmp_path/"inputs"
    root.mkdir()
    write_json(root/"d0_inputs.json", dict(arm="A5", oracle=True))
    plain = tmp_path/"plain"
    plain.mkdir()
    write_json(plain/"d0_inputs.json", dict(arm="A5P", oracle=False))
    cases = {("Other", str(root)): True, ("B1W_A5", str(plain)): True, ("B1W_v2", str(plain)): True, ("Other", str(plain)): False, ("Other", "/none"): False}
    for (arm, inputs), want in cases.items():
        assert SPEC.is_gated(SPEC.check_entry(entry(arm, inputs))) is want, (arm, inputs)
    line = SPEC.shell_lines(write_spec(tmp_path/"s.json", entry("Other", str(root))))[0].split("\t")
    assert line[6] == "1" and line[7] == "A5"


def test_oracle_is_in_the_method_and_the_arm_record(tmp_path):
    root = tmp_path/"inputs"
    root.mkdir()
    write_json(root/"d0_inputs.json", dict(arm="A5", oracle=True))
    spec = write_spec(tmp_path/"s.json", entry("Other", str(root), changes=dict(geometry_weight_levels=GW)))
    assert SPEC.method_name(SPEC.load_spec(spec)[0]) == "frozen_eb808be_b1_90s_gw1_1_0.1_0.01_oracle"
    record = SPEC.arm_record(spec, "Other")
    assert record["input_oracle"] is True and record["input_arm"] == "A5" and record["method"].endswith("_oracle")
    assert SPEC.method_name(SPEC.check_entry(entry("B1W_A5", "/none"))).endswith("_oracle")
    assert not SPEC.method_name(SPEC.check_entry(entry("Other", "/none"))).endswith("_oracle")


@pytest.mark.parametrize("winner,spec,seconds,changes", [("B1TM", dict(CANDIDATES[3]), 330, dict(geometry_weight_levels=GW)),
                                                         ("configC", dict(arm="configC", seconds=90, changes={}), 90, {})])
def test_v2_specification(tmp_path, capsys, winner, spec, seconds, changes):
    d = decision(tmp_path/"d.json", winner, spec)
    out = SPEC.v2_spec(d, "/results/v2/A5P", tmp_path/"specs"/"v2.json")
    e = SPEC.load_spec(out)[0]
    assert e["arm"] == "B1W_v2" and e["inputs_root"] == "/results/v2/A5P" and e["seconds"] == seconds and e["changes"] == changes
    assert e["registry"] == "configs/bayesreg/segmenters_v2.json" and SPEC.is_gated(e) is True and SPEC.gate(out, d, ["B1W_v2"], SELECTION) == 0
    SPEC.v2_spec(d, "/results/v2/A5P", tmp_path/"specs"/"v2.json")                                  # the same content: fine
    with pytest.raises(SPEC.SpecError, match="nothing is overwritten"):
        SPEC.v2_spec(d, "/results/other", tmp_path/"specs"/"v2.json")
    assert SPEC.main(["v2", "--decision", str(d), "--inputs-root", "/results/w", "--out", str(tmp_path/"w.json"), "--registry", "configs/x.json"]) == 0
    assert SPEC.load_spec(tmp_path/"w.json")[0]["registry"] == "configs/x.json"


def test_v2_specification_needs_a_decision_with_a_winner(tmp_path, capsys):
    with pytest.raises(SPEC.SpecError, match="no decision"):
        SPEC.v2_spec(tmp_path/"none.json", "/results/v2", tmp_path/"v2.json")
    d = decision(tmp_path/"d.json", None, None)
    with pytest.raises(SPEC.SpecError, match="no decision"):
        SPEC.v2_spec(d, "/results/v2", tmp_path/"v2.json")
    assert SPEC.main(["v2", "--decision", str(d), "--inputs-root", "/results/v2", "--out", str(tmp_path/"v2.json")]) == 2
    assert not (tmp_path/"v2.json").exists()


@pytest.mark.parametrize("bad", ["/abs/path.json", "../x.json", "configs/../../x.json", "", 3])
def test_registry_field_refusals(tmp_path, bad):
    with pytest.raises(SPEC.SpecError, match="registry must be"):
        SPEC.load_spec(write_spec(tmp_path/"s.json", entry(registry=bad)))


def test_registry_field_default():
    assert SPEC.check_entry(entry())["registry"] == "configs/bayesreg/segmenters.json"
    assert all(e["registry"] == "configs/bayesreg/segmenters.json" for e in SPEC.load_spec(SELECTION))
    assert SPEC.check_entry(entry(registry="configs/bayesreg/segmenters_v2.json"))["registry"].endswith("segmenters_v2.json")
    assert SPEC.configuration(SPEC.check_entry(entry(registry="x.json"))) == SPEC.configuration(SPEC.check_entry(entry()))


# ---- run_queue.sh --b1 ----------------------------------------------------------------------------------------------------

def tree_spec(tree, *entries, name="spec.json"):
    return write_spec(tree["results"].parent/name, *entries)


def selection_spec(tree):
    root = str(tree["a5"]/"inputs"/"A5P")
    return tree_spec(tree, *[dict(c, inputs_root=root) for c in CANDIDATES])


def run_b1(tree, spec, *args, **env):
    done = subprocess.run(["bash", str(QUEUE), "--b1", str(spec), *args], env=dict(tree["env"], CUDA_VISIBLE_DEVICES="", D0Q_POLL="0", **env),
                          capture_output=True, text=True, timeout=300)
    log = tree["results"]/"b1_v1"/"queue.log"
    return done.returncode, done.stdout, log.read_text() if log.exists() else ""


def order_lines(out):
    return [line for line in out.splitlines() if re.fullmatch(r"\d+ \S+ \S+", line)]


def expected_order(pairs, first_index=0, arms=ARMS):
    lines = []
    for i, p in enumerate(pairs, start=first_index):
        for j in range(len(arms)):
            lines.append(f"{len(lines)+1} {arms[(i+j) % len(arms)]} {p}")
    return lines


@pytest.fixture
def dry_tree(a5_tree):  # noqa: F811
    e2e_repo(a5_tree)
    return a5_tree


def test_queue_dry_run_order_rotates_with_the_pair_index(dry_tree):
    a5_tree = dry_tree
    spec = selection_spec(a5_tree)
    code, out, log = run_b1(a5_tree, spec, "--pairs", "first-5", "--dry-run")
    assert code == 0 and log == "" and order_lines(out) == expected_order(FAKE_PAIRS[:5])
    assert order_lines(out)[:5] == [f"1 B1T {FAKE_PAIRS[0]}", f"2 B1M {FAKE_PAIRS[0]}", f"3 B1P {FAKE_PAIRS[0]}", f"4 B1TM {FAKE_PAIRS[0]}", f"5 B1M {FAKE_PAIRS[1]}"]
    code, out, _ = run_b1(a5_tree, spec, "--pairs", "rest-14", "--dry-run")
    assert code == 0 and order_lines(out) == expected_order(FAKE_PAIRS[5:], first_index=5) and len(order_lines(out)) == 56
    assert order_lines(out)[0] == f"1 B1M {FAKE_PAIRS[5]}"                                         # pair 5 keeps the rotation of its index
    for args in (("--pairs", "all", "--dry-run"), ("--dry-run",)):
        code, out, _ = run_b1(a5_tree, spec, *args)
        assert code == 0 and order_lines(out) == expected_order(FAKE_PAIRS) and len(order_lines(out)) == 76
    assert not (a5_tree["results"]/"b1_v1").exists()                                                # a dry run creates nothing


def option(tokens, name):
    assert tokens.count(name) == 1, name
    return tokens[tokens.index(name)+1]


def test_queue_dry_run_prints_the_command_lines(dry_tree):
    a5_tree = dry_tree
    code, out, _ = run_b1(a5_tree, selection_spec(a5_tree), "--pairs", "first-5", "--dry-run")
    assert code == 0
    blocks = {}
    lines = out.splitlines()
    for k, line in enumerate(lines):
        match = re.match(r"# item (\d+): arm (\S+), pair index (\d+)", line)
        if match:
            blocks[(match[2], int(match[3]))] = (line, lines[k+1], lines[k+2], lines[k+3])
    assert len(blocks) == 20
    pre_c = ("fp32-v1:iterations=50/50/100/60,lr=0.5,cap=0.5,sigma_grad=1,sigma_field=1,mu=1/1/0.01/0.003,alpha=1e-3/1e-3/1e-3/0,"
             "kappa_pre=10,j0=0.2,seconds=60,taper=4")
    for (arm, _), (head, prep, check, sd) in blocks.items():
        seconds = 330 if arm in ("B1T", "B1TM") else 90
        tokens = shlex.split(prep)
        assert option(tokens, "--seconds") == str(seconds) and option(tokens, "--stage") == f"B1-{arm}" and f"b1_v1/runs/{arm}/" in tokens[tokens.index("prepare")+1]
        assert option(tokens, "--geometry-weight-levels") == ("1,1,0.1,0.01" if arm in ("B1M", "B1TM") else "1,1,0.1,0.03")
        assert option(tokens, "--pre-optimizer") == (pre_c.replace("iterations=50/50/100/60", "iterations=50/50/100/240") if arm == "B1P" else pre_c)
        assert option(tokens, "--strength") == "weak" and option(tokens, "--throughput") == "v41" and option(tokens, "--export-reserve") == "1"
        assert option(tokens, "--curvature-weight-levels") == "1e-3,1e-3,1e-3,0" and option(tokens, "--level-shares") == "0.020833,0.0625,0.145833,1"
        assert "--lean-commits" in tokens and "--resolution-exit" in tokens
        tokens = shlex.split(check)
        assert option(tokens, "--b1-arm") == arm and option(tokens, "--b1-spec").endswith("spec.json")
        watchdog = 1100 if seconds == 330 else 300
        tokens = shlex.split(sd)
        assert f"watchdog {watchdog} s" in head and f" {watchdog}s " in tokens[-1] and any(t.startswith(f"--unit=tag-b1-{arm}-") for t in tokens)
        assert "--property=MemoryMax=12G" in tokens and any(t.endswith("b1_v1/cache/numba_jit") for t in tokens)


def test_launch_strings_do_not_hold_the_word(dry_tree):
    """The launch strings must not hold the name of the test runner."""
    a5_tree = dry_tree
    word = "py"+"test"
    _, out, _ = run_b1(a5_tree, selection_spec(a5_tree), "--pairs", "first-5", "--dry-run")
    assert word not in out.replace(str(a5_tree["results"].parent), "TREE")
    assert word not in QUEUE.read_text()
    plan = ROOT/"runs"/"20261103-b1"/"plan.md"
    blocks = re.findall(r"```bash\n(.*?)```", plan.read_text(), re.S)
    assert blocks and all(word not in block for block in blocks)


@pytest.mark.parametrize("args,fragment", [(("--pairs", "first-9"), "--pairs: first-5"), (("--first-5",), "--first-5 goes with --a5")])
def test_queue_bad_options(a5_tree, args, fragment):  # noqa: F811
    code, out, _ = run_b1(a5_tree, selection_spec(a5_tree), *args, "--dry-run")
    assert code == 2 and fragment in out


def test_pairs_goes_with_b1(a5_tree):  # noqa: F811
    done = subprocess.run(["bash", str(QUEUE), "--a5", "--pairs", "first-5", "--dry-run"], env=a5_tree["env"], capture_output=True, text=True, timeout=60)
    assert done.returncode == 2 and "--pairs goes with --b1" in done.stdout


def test_runs_root_is_refused_with_b1_only(dry_tree):
    code, out, _ = run_b1(dry_tree, selection_spec(dry_tree), "--runs-root", "/tmp/elsewhere", "--dry-run")
    assert code == 2 and "--runs-root: not with --b1" in out
    done = subprocess.run(["bash", str(QUEUE), "--a5", "--runs-root", str(dry_tree["results"]/"x"), "--dry-run"], env=dry_tree["env"], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0


def test_queue_refuses_a_bad_specification(dry_tree):
    bad = tree_spec(dry_tree, entry(changes=dict(lr=[1])), name="bad.json")
    code, out, log = run_b1(dry_tree, bad, "--dry-run")
    assert code == 2 and "not options of this tool" in out and "is refused" in out


# ---- the whole queue on a synthetic frozen solver ---------------------------------------------------------------------------

FAKE_E48 = r'''import argparse, hashlib, json, os, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ap = argparse.ArgumentParser()
ap.add_argument('action'); ap.add_argument('output', type=Path)
for name in ('strength', 'pg-metric-policy-levels', 'level-shares', 'geometry-weight-levels', 'curvature-weight-levels', 'throughput', 'export-reserve', 'pre-optimizer', 'stage', 'pair'):
    ap.add_argument('--'+name)
ap.add_argument('--seconds', type=int, choices=(90, 330)); ap.add_argument('--lean-commits', action='store_true'); ap.add_argument('--resolution-exit', action='store_true')
a = ap.parse_args()
if a.action == 'prepare':
    proto = json.loads((ROOT/'scripts'/'reference_protocol.json').read_text())
    gw = [float(x) for x in a.geometry_weight_levels.split(',')]
    it = tuple(int(x) for x in dict(item.split('=') for item in a.pre_optimizer.split(':')[1].split(','))['iterations'].split('/'))
    proto['scope'] = proto['scope'].replace(f"mu=nu={proto['geometry_weight_levels']}", f'mu=nu={gw}').replace(f"'iterations': {tuple(proto['pre_optimizer']['iterations'])}", f"'iterations': {it}")
    proto['geometry_weight_levels'] = gw
    proto['pre_optimizer']['iterations'] = list(it)
    data = a.output/'inputs'/a.pair
    data.mkdir(parents=True)
    source = ROOT/'data'/'three_d'/'stage_L2R_val_v1'/a.pair
    for name in ('pair.npz', 'manifest.json'):
        shutil.copy(source/name, data/name)
    me = a.output/'snapshot'/'scripts'/'run_stage_3d_e48_pg.py'
    me.parent.mkdir(parents=True)
    shutil.copy(__file__, me)
    proto.update(stage=a.stage, task=dict(index=0, pair=a.pair, route='pg_weak', strength='weak', hierarchy='four', wall_seconds=a.seconds, name=f'pg_weak__four__{a.seconds}s'),
                 input_sha256={n: sha(data/n) for n in ('pair.npz', 'manifest.json')}, pair_source=f'data/three_d/stage_L2R_val_v1/{a.pair}',
                 source_sha256={'scripts/run_stage_3d_e48_pg.py': sha(me)})
    (a.output/'protocol.json').write_text(json.dumps(proto))
    print(a.output)
    sys.exit(0)
proto = json.loads((a.output/'protocol.json').read_text())
fail = os.environ.get('FAKE_E48_EXIT', '')
if fail and proto['stage'].endswith(fail.split(':')[0]):
    sys.exit(int(fail.split(':')[1]))
task = a.output/proto['task']['name']
task.mkdir()
(a.output/'started.json').write_text('{}')
(task/'summary.json').write_text(json.dumps(dict(status='native_budget_exhausted', success=False, metrics_role='formal_native_outer_commit', native_commits=3, e2e_seconds=proto['task']['wall_seconds']-2)))
(task/'worker_completion.json').write_text('{}')
(task/'final_native.json').write_text('{}')
for name in ('final_native.npz', 'committed_1.npz', 'level_3_x.npz'):
    (task/name).write_bytes(b'x')
(a.output/'completion.json').write_text('{}')
'''
FAKE_EVALUATE = r'''import json, os, sys
from pathlib import Path
argv = sys.argv[1:]
with open(os.environ['FAKE_LOG_DIR']+'/evaluate_calls.txt', 'a') as f:
    f.write(' '.join(argv)+'\n')
if argv[0] in ('run', 'run-b1'):
    run = Path(next(x for x in argv[1:] if '/' in x and not x.endswith('.json')))
    get = lambda k: argv[argv.index(k)+1]
    cls = json.loads((run/'classification.json').read_text())
    (run/'metrics.json').write_text(json.dumps(dict(run_id=get('--run-id'), classification={k: cls.get(k) for k in ('class', 'kind', 'reason')}, status='native_budget_exhausted', native_commits=3, e2e_seconds=88.,
        metrics=dict(seg35=.8, seg4=.9, hd95=2., sdlogj_official=.07, sdlogj_brain=.6), certificate=dict(status='certified_feasible', passed=True), self_check=dict(passed=True))))
'''
FAKE_SYSTEMD_RUN = '''#!/bin/bash
stamp=$(date +%s%N)
printf '%s\\n' "$@" > "$FAKE_LOG_DIR/sd_$stamp.args"
for a in "$@"; do inner=$a; case "$a" in --working-directory=*) wd=${a#--working-directory=};; esac; done
cd "$wd" && /bin/bash -c "$inner"
'''


def e2e_repo(tree):
    """Copies of the real b1_spec.py, check_prepared.py, classify_run.py in the fake repository, a stub evaluate.py, a reference with the fake freeze."""
    d = tree["repo"]/"experiments"/"d0"
    d.mkdir(parents=True, exist_ok=True)
    for name in ("b1_spec.py", "check_prepared.py", "classify_run.py"):
        shutil.copy(ROOT/"experiments"/"d0"/name, d/name)
    (d/"evaluate.py").write_text(FAKE_EVALUATE)
    selection = tree["repo"]/"runs"/"20261103-b1"/"arms_selection.json"
    selection.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(SELECTION, selection)
    return d


@pytest.fixture
def e2e(a5_tree, tmp_path):  # noqa: F811
    """The whole queue runs: a synthetic frozen worktree (clone at the tag eb808be) and the real prepare checks, classification and cleanup."""
    tree, repo, results = a5_tree, a5_tree["repo"], a5_tree["results"]
    d = e2e_repo(tree)
    log_dir = tmp_path/"fake_log"
    log_dir.mkdir()
    (repo/"scripts").mkdir()
    (repo/"scripts"/"run_stage_3d_e48_pg.py").write_text(FAKE_E48)
    write_json(repo/"scripts"/"reference_protocol.json", REFERENCE["reference_protocol"])
    write_json(d/"configC_reference.json", dict(reference_protocol=REFERENCE["reference_protocol"],
                                                freeze_source_py_sha256={"scripts/run_stage_3d_e48_pg.py": file_sha256(repo/"scripts"/"run_stage_3d_e48_pg.py")}))
    tree["git"]("add", "scripts")
    tree["git"]("commit", "-q", "-m", "fake frozen solver")
    tree["git"]("tag", "eb808be")
    subprocess.run(["git", "clone", "-q", str(repo), str(results/"frozen_eb808be")], check=True, capture_output=True)
    (tree["bin"]/"systemd-run").write_text(FAKE_SYSTEMD_RUN)
    selection = (repo/"data"/"three_d"/"stage_L2R_val_v1"/"selection.json").read_text()
    registry = json.loads(tree["record"].read_text())["structure_registry"]
    for arm in ("A5", "A5P"):
        stage = tree["a5"]/"inputs"/arm/"stage_L2R_val_v1"
        stage.mkdir(parents=True)
        (stage/"selection.json").write_text(selection)
        files = {}
        side = dict(subject="s") if arm == "A5" else dict(subject="s", unet_provenance_sha256="u", policy="all_folds_mean", model_ids=["m"], registry_status="registered", registry_sha256=registry["sha256"])
        for p in FAKE_PAIRS:
            e = stage/p
            e.mkdir()
            (e/"pair.npz").write_bytes(f"{arm} {p}".encode())
            write_json(e/"manifest.json", dict(pair=p, pair_sha256=file_sha256(e/"pair.npz"), fixed_subject=p[:18], moving_subject=p[-18:], preprocessing_d0=dict(
                arm=arm, oracle=arm == "A5", recipe=f"recipe {arm}", source_pair_sha256="o", source_manifest_sha256="m",
                colour_table=dict(table_file=str(tree["table"]), table_sha256=file_sha256(tree["table"])), colour=dict(fixed=side, moving=side))))
            files[f"stage_L2R_val_v1/{p}/pair.npz"] = file_sha256(e/"pair.npz")
            files[f"stage_L2R_val_v1/{p}/manifest.json"] = file_sha256(e/"manifest.json")
        doc = json.loads(tree["records"][arm].read_text())
        doc.update(files_sha256=files, oracle=arm == "A5", colour_table=dict(table_file=str(tree["table"]), table_sha256=file_sha256(tree["table"])))
        write_json(tree["records"][arm], doc)
    tree["env"].update(FAKE_LOG_DIR=str(log_dir), D0Q_POLL="0")
    return dict(tree, log_dir=log_dir)


def two_arm_spec(tree):
    root = str(tree["a5"]/"inputs"/"A5P")
    return tree_spec(tree, dict(CANDIDATES[0], inputs_root=root), dict(CANDIDATES[1], inputs_root=root), name="two.json")


def snapshot(directory):
    return {str(p): file_sha256(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def queue_decision(tree, winner, spec):
    """analysis/decision_b1.json, as summarize-b1 writes the keys the gate reads."""
    path = tree["results"]/"b1_v1"/"analysis"/"decision_b1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return decision(path, winner, spec, selection=tree["repo"]/"runs"/"20261103-b1"/"arms_selection.json")


def v2_inputs(tree, *, arm="A5P", commit=True):
    """The inputs of the new segmenter: the A5P layout under its own root, the record naming segmenters_v2.json; `arm` as in the record and the manifests."""
    root = tree["results"]/"unv2_inputs"/arm
    shutil.copytree(tree["a5"]/"inputs"/"A5P", root)
    registry = tree["repo"]/"configs"/"bayesreg"/"segmenters_v2.json"
    registry.write_text(json.dumps(dict(id="unet35_v2_full")))
    if commit:
        tree["git"]("add", "configs/bayesreg/segmenters_v2.json")
        tree["git"]("commit", "-q", "-m", "v2 registry")
    doc = json.loads((root/"d0_inputs.json").read_text())
    doc["arm"] = arm
    doc["structure_registry"] = dict(path=str(registry), sha256=file_sha256(registry), status="registered", git=dict(tracked=True, clean=True))
    for p in FAKE_PAIRS:
        path = root/"stage_L2R_val_v1"/p/"manifest.json"
        manifest = json.loads(path.read_text())
        manifest["preprocessing_d0"]["arm"] = arm
        write_json(path, manifest)
        doc["files_sha256"][f"stage_L2R_val_v1/{p}/manifest.json"] = file_sha256(path)
    write_json(root/"d0_inputs.json", doc)
    return root


def test_whole_queue_first_5_then_the_rest_reruns_nothing(e2e):
    spec = two_arm_spec(e2e)
    b1 = e2e["results"]/"b1_v1"
    code, out, log = run_b1(e2e, spec, "--pairs", "first-5")
    assert code == 0 and "queue finished: 10 items" in log and "start checks passed" in log, log
    assert "B1T inputs check:" in log and "arm A5P oracle False; ok" in log and "mode b1 (arms B1T B1M)" in log
    runs = b1/"runs"
    assert sorted(p.name for p in (runs/"B1T").iterdir()) == FAKE_PAIRS[:5] == sorted(p.name for p in (runs/"B1M").iterdir())
    p0 = FAKE_PAIRS[0]
    short = re.sub(r"OASIS_OAS1_(\d+)_MR1__OASIS_OAS1_(\d+)_MR1", r"\1-\2", p0)
    for arm, seconds, gw, watchdog in (("B1T", 330, "1,1,0.1,0.03", 1100), ("B1M", 90, "1,1,0.1,0.01", 300)):
        run = runs/arm/p0
        assert (run/"queue_done.json").is_file() and json.loads((run/"prepared_check.json").read_text())["ok"] is True
        assert json.loads((run/"prepared_check.json").read_text())["b1_intended_deviation"]["seconds"] == seconds
        lines = (run/"cmd.txt").read_text().splitlines()
        tokens = shlex.split(lines[1])
        assert option(tokens, "--seconds") == str(seconds) and option(tokens, "--geometry-weight-levels") == gw and option(tokens, "--stage") == f"B1-{arm}"
        assert f" {watchdog}s " in shlex.split(lines[4])[-1]
        assert (run/"service_unit.txt").read_text().strip() == f"tag-b1-{arm}-{short}" and (run/"run_id").read_text().startswith(f"b1_{arm}_{short}_")
        record = json.loads((run/"b1_arm.json").read_text())
        assert record["seconds"] == seconds and record["input_arm"] == "A5P" and record["spec_sha256"] == file_sha256(spec) and record["watchdog_seconds"] == watchdog
        task = run/f"pg_weak__four__{seconds}s"
        assert (task/"final_native.npz").is_file() and not (task/"committed_1.npz").exists() and not (task/"level_3_x.npz").exists()      # the cleanup of protocol §6
        data = (run/"data.txt").read_text()
        assert f"arm: {arm}\n" in data and "arm of the inputs record: A5P" in data and "colour table:" in data and "unet fixed" in data and "d0_v1_a5/inputs/A5P/d0_inputs.json" in data
        assert "the inputs of A5P" in (run/"seed").read_text()
        assert json.loads((run/"classification.json").read_text())["evidence"]["watchdog_seconds"] == watchdog
    calls = (e2e["log_dir"]/"evaluate_calls.txt").read_text().splitlines()
    assert sum(c.startswith("run-b1 --spec ") for c in calls) == 10 and not [c for c in calls if c.startswith("run ")]
    assert any(c.startswith("summarize-b1 --b1-root ") and "runs/20261103-b1/arms_selection.json" in c and "d0_v1_a5" in c for c in calls)
    before = snapshot(runs)

    code, out, log = run_b1(e2e, spec, "--pairs", "first-5")                                       # again: nothing runs
    assert code == 0 and out.count("skip ") == 10 and snapshot(runs) == before

    code, out, log = run_b1(e2e, spec, "--pairs", "rest-14")                                       # no decision yet
    assert code == 1 and "STOP: the selection has not opened these items" in log and snapshot(runs) == before

    queue_decision(e2e, "B1M", dict(CANDIDATES[1]))
    code, out, log = run_b1(e2e, spec, "--pairs", "rest-14")                                       # two arms, only one is the winner's configuration
    assert code == 1 and "B1T: configuration" in log and "REFUSED" in log and snapshot(runs) == before

    winner = tree_spec(e2e, dict(CANDIDATES[1], inputs_root=str(e2e["a5"]/"inputs"/"A5P"), expansion=True), name="expand.json")
    code, out, log = run_b1(e2e, winner, "--pairs", "rest-14")
    assert code == 0, log
    assert sorted(p.name for p in (runs/"B1M").iterdir()) == FAKE_PAIRS and sorted(p.name for p in (runs/"B1T").iterdir()) == FAKE_PAIRS[:5]
    after = snapshot(runs)
    assert all(after[k] == v for k, v in before.items())                                            # the first 5 are untouched
    code, out, log = run_b1(e2e, winner, "--pairs", "all")
    assert code == 0 and out.count("skip ") == 19 and snapshot(runs) == after

    oracle = tree_spec(e2e, dict(CANDIDATES[1], arm="B1W_A5", inputs_root=str(e2e["a5"]/"inputs"/"A5"), expansion=True), name="oracle.json")
    code, out, log = run_b1(e2e, oracle, "--pairs", "first-5")
    assert code == 0 and "arm A5 oracle True; ok" in log, log
    assert sorted(p.name for p in (runs/"B1W_A5").iterdir()) == FAKE_PAIRS[:5]
    data = (runs/"B1W_A5"/p0/"data.txt").read_text()
    assert "arm of the inputs record: A5" in data and "unet fixed" not in data and "oracle (truth labels used for the colouring): True" in data
    assert "colour table:" in data
    record = json.loads((runs/"B1W_A5"/p0/"b1_arm.json").read_text())
    assert record["input_oracle"] is True and record["method"].endswith("_oracle")


@pytest.mark.parametrize("arm_name,inputs", [("B1W_A5", "A5"), ("Other", "A5"), ("B1W_A5", "A5P")])
def test_oracle_inputs_are_gated_without_the_flag(e2e, arm_name, inputs):
    """Review I2: a hand-written specification without expansion: the oracle inputs or the name are enough."""
    spec = tree_spec(e2e, entry(arm_name, str(e2e["a5"]/"inputs"/inputs), seconds=330), name="hand.json")
    code, _, log = run_b1(e2e, spec, "--pairs", "first-5")
    assert code == 1 and "STOP: the selection has not opened these items" in log
    assert not list((e2e["results"]/"b1_v1"/"runs").glob("*/*"))
    queue_decision(e2e, "configC", dict(arm="configC", seconds=90, changes={}))                     # the winner is configC: no oracle arm, whatever the configuration
    spec = tree_spec(e2e, entry(arm_name, str(e2e["a5"]/"inputs"/inputs)), name="hand2.json")
    code, _, log = run_b1(e2e, spec, "--pairs", "first-5")
    assert code == 1 and "REFUSED" in log and not list((e2e["results"]/"b1_v1"/"runs").glob("*/*"))


def test_gate_refuses_a_decision_from_another_selection_in_the_queue(e2e):
    spec = tree_spec(e2e, dict(CANDIDATES[1], arm="B1W_A5", inputs_root=str(e2e["a5"]/"inputs"/"A5"), expansion=True), name="oracle.json")
    other = write_spec(e2e["results"].parent/"other.json", dict(CANDIDATES[0]))
    path = e2e["results"]/"b1_v1"/"analysis"/"decision_b1.json"
    path.parent.mkdir(parents=True)
    decision(path, "B1M", dict(CANDIDATES[1]), selection=other)
    code, _, log = run_b1(e2e, spec, "--pairs", "first-5")
    assert code == 1 and "made from a selection specification" in log and "STOP: the selection has not opened" in log


@pytest.mark.parametrize("winner,winner_spec,seconds", [("B1M", dict(CANDIDATES[1]), 90), ("configC", dict(arm="configC", seconds=90, changes={}), 90)])
def test_b1w_v2_first_5_then_the_rest(e2e, winner, winner_spec, seconds):
    root = v2_inputs(e2e)
    runs = e2e["results"]/"b1_v1"/"runs"
    spec = e2e["results"].parent/"v2.json"
    with pytest.raises(SPEC.SpecError):                                                             # no decision yet: no specification
        SPEC.v2_spec(e2e["results"]/"b1_v1"/"analysis"/"decision_b1.json", root, spec)
    written = tree_spec(e2e, dict(arm="B1W_v2", inputs_root=str(root), seconds=seconds, changes=winner_spec["changes"], registry="configs/bayesreg/segmenters_v2.json"), name="v2_early.json")
    code, _, log = run_b1(e2e, written, "--pairs", "first-5")                                      # the name is gated even without the flag
    assert code == 1 and "STOP: the selection has not opened" in log and not list(runs.glob("*/*"))
    d = queue_decision(e2e, winner, winner_spec)
    SPEC.v2_spec(d, root, spec)
    code, _, log = run_b1(e2e, spec, "--pairs", "first-5")
    assert code == 0 and "B1W_v2 inputs check:" in log and "oracle False; ok" in log, log
    assert sorted(p.name for p in (runs/"B1W_v2").iterdir()) == FAKE_PAIRS[:5]
    run = runs/"B1W_v2"/FAKE_PAIRS[0]
    tokens = shlex.split((run/"cmd.txt").read_text().splitlines()[1])
    assert option(tokens, "--seconds") == str(seconds) and option(tokens, "--stage") == "B1-B1W_v2"
    assert option(tokens, "--geometry-weight-levels") == ("1,1,0.1,0.01" if winner == "B1M" else "1,1,0.1,0.03")
    assert json.loads((run/"b1_arm.json").read_text())["registry"] == "configs/bayesreg/segmenters_v2.json"
    before = snapshot(runs)
    code, out, log = run_b1(e2e, spec, "--pairs", "rest-14")
    assert code == 0, log
    assert sorted(p.name for p in (runs/"B1W_v2").iterdir()) == FAKE_PAIRS
    after = snapshot(runs)
    assert all(after[k] == v for k, v in before.items())


def test_data_txt_follows_the_keys_of_the_inputs_not_the_arm_name(e2e):
    """Review M9: a record under another arm name still gets its colour table and U-Net provenance into data.txt."""
    root = v2_inputs(e2e, arm="A5Pv2")
    d = queue_decision(e2e, "configC", dict(arm="configC", seconds=90, changes={}))
    spec = e2e["results"].parent/"v2.json"
    SPEC.v2_spec(d, root, spec)
    code, _, log = run_b1(e2e, spec, "--pairs", "first-5", "--only", "B1W_v2:0")
    assert code == 0, log
    data = (e2e["results"]/"b1_v1"/"runs"/"B1W_v2"/FAKE_PAIRS[0]/"data.txt").read_text()
    assert "arm of the inputs record: A5Pv2" in data and "colour table:" in data and "unet fixed" in data and "unet moving" in data


@pytest.mark.parametrize("case,fragment", [("not_committed", "not committed or not clean"), ("modified", "sha256 differs"), ("default_entry", "the record names")])
def test_v2_registry_stops_the_queue(e2e, case, fragment):
    root = v2_inputs(e2e, commit=case != "not_committed")
    d = queue_decision(e2e, "configC", dict(arm="configC", seconds=90, changes={}))
    spec = e2e["results"].parent/"v2.json"
    SPEC.v2_spec(d, root, spec, registry=SPEC.REGISTRY if case == "default_entry" else SPEC.V2_REGISTRY)
    if case == "modified":
        registry = e2e["repo"]/"configs"/"bayesreg"/"segmenters_v2.json"
        registry.write_text(registry.read_text()+"\n")
    code, _, log = run_b1(e2e, spec, "--pairs", "first-5")
    assert code == 1 and "STOP: the inputs of an arm" in log and fragment in log, log
    assert not list((e2e["results"]/"b1_v1"/"runs").glob("*/*"))


def test_v2_registry_missing_until_it_is_trained_stops_the_queue(e2e):
    root = v2_inputs(e2e, commit=False)
    (e2e["repo"]/"configs"/"bayesreg"/"segmenters_v2.json").unlink()
    d = queue_decision(e2e, "configC", dict(arm="configC", seconds=90, changes={}))
    spec = e2e["results"].parent/"v2.json"
    SPEC.v2_spec(d, root, spec)
    code, _, log = run_b1(e2e, spec, "--pairs", "first-5")
    assert code == 1 and "registry configs/bayesreg/segmenters_v2.json: sha256 differs" in log, log


def test_expansion_arm_needs_the_decision_also_on_the_first_5(e2e):
    oracle = tree_spec(e2e, dict(CANDIDATES[1], arm="B1W_A5", inputs_root=str(e2e["a5"]/"inputs"/"A5"), expansion=True), name="oracle.json")
    code, _, log = run_b1(e2e, oracle, "--pairs", "first-5")
    assert code == 1 and "STOP: the selection has not opened these items" in log and not (e2e["results"]/"b1_v1"/"runs"/"B1W_A5").exists()


def test_failure_is_kept_and_nothing_is_rerun(e2e):
    spec = two_arm_spec(e2e)
    code, _, log = run_b1(e2e, spec, "--pairs", "first-5", FAKE_E48_EXIT="B1M:137")                 # a signal: infrastructure, the queue stops
    assert code == 8 and "classified infrastructure" in log and "queue stopped at item 2/10" in log, log
    runs = e2e["results"]/"b1_v1"/"runs"
    run = runs/"B1M"/FAKE_PAIRS[0]
    assert (run/"exit_code").read_text().strip() == "137" and not (run/"queue_done.json").exists() and (run/"snapshot").is_dir()
    before = snapshot(runs)
    code, _, log = run_b1(e2e, spec, "--pairs", "first-5")                                         # the run directory stays; the queue stops at it again
    after = snapshot(runs)
    changed = [k for k in before if after[k] != before[k]]
    assert code == 8 and "resume:" in log and sorted(after) == sorted(before)
    assert changed == [str(run/"log.txt")]                                                          # only the queue's own log line


def test_exit_before_the_task_clock_stops_the_queue(e2e):
    code, _, log = run_b1(e2e, two_arm_spec(e2e), "--pairs", "first-5", "--only", "B1M:0,B1T:0", FAKE_E48_EXIT="B1T:1")
    run = e2e["results"]/"b1_v1"/"runs"/"B1T"/FAKE_PAIRS[0]
    assert code == 8 and (run/"exit_code").read_text().strip() == "1" and "classified unsure" in log


@pytest.mark.parametrize("change,fragment", [
    ("table_modified", "colour table"), ("table_sha", "colour table"), ("registry_modified", "registry"), ("registry_status", "registry"), ("partial", "partial"),
    ("no_record", "no d0_inputs.json"), ("no_oracle_flag", "oracle true or false"), ("other_config", "this specification"),
])
def test_queue_b1_start_checks_stop(e2e, change, fragment):
    spec = two_arm_spec(e2e)
    record = e2e["records"]["A5P"]
    doc = json.loads(record.read_text())
    if change == "table_modified":
        e2e["table"].write_text(e2e["table"].read_text()+"\n")
    elif change == "table_sha":
        doc["colour_table"]["table_sha256"] = "0"*64
    elif change == "registry_modified":
        e2e["registry"].write_text(e2e["registry"].read_text()+"\n")
    elif change == "registry_status":
        doc["structure_registry"]["status"] = "draft"
    elif change == "partial":
        doc["partial"] = True
    elif change == "no_oracle_flag":
        del doc["oracle"]
    elif change == "other_config":
        old = e2e["results"]/"b1_v1"/"runs"/"B1T"/FAKE_PAIRS[0]
        old.mkdir(parents=True)
        write_json(old/"b1_arm.json", dict(seconds=90, changes={}, inputs_root=str(record.parent)))
    if change == "no_record":
        record.unlink()
    else:
        write_json(record, doc)
    code, _, log = run_b1(e2e, spec, "--pairs", "first-5")
    assert code == 1 and "STOP: the inputs of an arm" in log and fragment in log, log
    assert "worktree HEAD" not in log and not (e2e["results"]/"b1_v1"/"runs"/"B1M").exists()


@pytest.mark.parametrize("units,fragment", [("tag-d0a5-A5P-0438-0439.service loaded active running x\\n", "tag-d0a5"), ("tag-unv2-train.service loaded active running x\\n", "tag-unv2"),
                                            ("tag-unet35-train.service loaded active running x\\n", "tag-unet35-train")])
def test_queue_b1_refuses_to_start_beside_other_gpu_units(e2e, units, fragment):
    code, _, log = run_b1(e2e, two_arm_spec(e2e), "--pairs", "first-5", FAKE_UNITS=units)
    assert code == 1 and "STOP: units still active" in log and fragment in log


def test_queue_b1_refuses_a_busy_gpu(e2e):
    code, _, log = run_b1(e2e, two_arm_spec(e2e), "--pairs", "first-5", FAKE_PIDS="4242\\n")
    assert code == 1 and "STOP: GPU not idle at start" in log


def test_prepare_with_a_change_the_check_does_not_list_stops_the_queue(e2e):
    """The queue's prepare and check_prepared agree only on what the entry lists: a frozen script that changes one more option is stopped before any unit."""
    spec = two_arm_spec(e2e)
    frozen = e2e["results"]/"frozen_eb808be"/"scripts"/"reference_protocol.json"
    doc = json.loads(frozen.read_text())
    doc["curvature_weight_levels"] = [1e-3, 1e-3, 1e-3, 1e-3]
    frozen.write_text(json.dumps(doc))
    subprocess.run(["git", "-C", str(frozen.parent), "update-index", "--assume-unchanged", "reference_protocol.json"], check=True)
    code, _, log = run_b1(e2e, spec, "--pairs", "first-5")
    assert code == 8 and "does not match configuration C" in log and "curvature_weight_levels" in log, log
    assert not list(e2e["log_dir"].glob("sd_*"))


# ---- the other modes: the same bytes as the script before this package --------------------------------------------------------

@pytest.fixture
def old_queue(tmp_path):
    done = subprocess.run(["git", "-C", str(ROOT), "show", f"{BEFORE}:experiments/d0/run_queue.sh"], capture_output=True, text=True)
    if done.returncode != 0:
        pytest.skip(f"commit {BEFORE} not available")
    assert "POLL=60" in done.stdout and "--b1" not in done.stdout
    path = tmp_path/"old_queue.sh"
    path.write_text(done.stdout.replace("POLL=60", "POLL=0"))
    return path


@pytest.mark.parametrize("args", [(), ("--a4",), ("--a1p",), ("--x1",), ("--a5",), ("--a5", "--first-5"), ("--a5", "--only", "A5P:1,A5:7")])
def test_other_modes_dry_run_is_unchanged(a5_tree, old_queue, args):  # noqa: F811
    old = subprocess.run(["bash", str(old_queue), *args, "--dry-run"], env=a5_tree["env"], capture_output=True, text=True, timeout=120)
    new = subprocess.run(["bash", str(QUEUE), *args, "--dry-run"], env=a5_tree["env"], capture_output=True, text=True, timeout=120)
    assert new.returncode == old.returncode == 0 and new.stdout == old.stdout and new.stderr == old.stderr and "# item" not in new.stdout


def normal(text):
    """Time stamps, free space and the load are the only differences between two runs of the same queue."""
    text = re.sub(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", "T", text)
    text = re.sub(r"\d{8}T\d{6}Z", "T", text)
    text = re.sub(r"free \d+ GB", "free N GB", text)
    return text


def collect(tree):
    a5 = tree["a5"]
    out = {}
    for path in sorted((a5/"runs").rglob("*")):
        rel = str(path.relative_to(a5))
        if path.is_file() and path.name in ("cmd.txt", "data.txt", "seed", "prepared_check.json", "service_unit.txt", "run_id", "cleanup.json", "exit_code", "metrics.json", "protocol.json", "queue_done.json"):
            out[rel] = normal(path.read_text())
    out["queue.log"] = normal((a5/"queue.log").read_text()).replace(str(tree["log_dir"]), "LOG")
    out["sd"] = [normal(p.read_text()) for p in sorted(tree["log_dir"].glob("sd_*"))]
    calls = tree["log_dir"]/"evaluate_calls.txt"
    out["evaluate"] = normal(calls.read_text())
    return out


def test_a5_whole_queue_is_the_same_as_before_this_package(e2e, old_queue):
    args = ("--a5", "--only", "A5:0,A5P:0,A5P:1,A5:1")

    def go(script):
        done = subprocess.run(["bash", str(script), *args], env=dict(e2e["env"], CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True, timeout=180)
        assert done.returncode == 0, done.stdout+(e2e["a5"]/"queue.log").read_text()
        got = collect(e2e)
        shutil.rmtree(e2e["a5"]/"runs")
        (e2e["a5"]/"queue.log").unlink()
        for p in e2e["log_dir"].iterdir():
            p.unlink()
        for p in e2e["a5"].glob("env_*.txt"):
            p.unlink()
        return got
    old, new = go(old_queue), go(QUEUE)
    assert len(old["sd"]) == 4 and " 300s " in old["sd"][0] and "--seconds 90 " in old[f"runs/A5/{FAKE_PAIRS[0]}/cmd.txt"]
    assert "--b1" not in new[f"runs/A5/{FAKE_PAIRS[0]}/cmd.txt"] and "b1_v1" not in new["queue.log"]
    assert sorted(old) == sorted(new)
    for key in old:
        assert old[key] == new[key], key


def test_queue_settings():
    text = QUEUE.read_text()
    assert "D0=$RESULTS/b1_v1; SELF_RE='^tag-b1-'; RIDP=b1; UNITP=${UNITP:-tag-b1}; STAGE_PREFIX=B1" in text
    assert text.count('exec 9> "$LINK_LOCK"') == 1 and subprocess.run(["bash", "-n", str(QUEUE)], capture_output=True).returncode == 0


def test_plan_says_nothing_ran_and_names_the_rules():
    text = (ROOT/"runs"/"20261103-b1"/"plan.md").read_text()
    for fragment in ("一次也没有运行", "≤ .095", "− .003", "< .002", "< .005", "≥ .86", "≥ .85", ".8780", "configC", "选择效应", "预算收尾不是收敛",
                     "修订 1", "B1_tools_复核_20261103.md", "近端变化罚", "/mnt/c", ".009–.020", "B1W_v2"):
        assert fragment in text, fragment
    proto = (ROOT/"docs"/"bayesreg"/"05_D0协议.md").read_text()
    assert "\n## 15." in proto and "防泄漏" in proto.split("\n## 15.")[1] and "近端变化罚" in proto.split("\n## 15.")[1]
    assert "近端变化罚" in EV.B1_NOTE and ".009–.020" in EV.B1_NOTE
    assert np.isclose(EV.B1_SDLOGJ_MAX, .095) and EV.B1_MIN_GAIN == .005 and EV.B1_CHEAPER_MARGIN == .002 and EV.B1_SEG4_MARGIN == .003
