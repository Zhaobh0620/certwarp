"""Gate A tools (runs/20261102-gate-a/plan.md; work package WST, scope revision 11-02 ~5:40 pm CST), on synthetic files only (tmp_path;
no data/, no unit, no real result file).

X1 (experiments/d0/run_queue.sh --x1, check_prepared.py, evaluate.py summarize-x1):
- the order (first 5 pairs, rotating arms), the arm switches, nothing written by the dry run; each start check; D0, --a4, --a1p unchanged;
- check_prepared: exactly the two T92 keys and the T92 sentence of the scope; anything else, or a wrong value, is refused;
- the reading on seg4 (<= the midpoint, >= A0 - .01, middle, too few pairs), both thresholds on the arm's own pairs (revision 1, I2, with the
  real per-pair baselines), a state without native commits not comparable (M4), the pins of A0 and P0n; the labels (I4); the seed kept by the
  cleanup only for X1 and evaluated, descriptive only (I3); commits only (M1), the seed's MSE ratio (M2), comparable means (M3), the six large
  structures (M5), the launch load (M12); the summary.
X2 (experiments/gate_a/run_x2.py, summarize_x2.py) with a fake machine (units emulated; run_pair's arguments parsed by its real parser):
- the arms, the rotating order, the channel source and the eta file of each arm, eta records, batch.json, config.json checks (a
  mismatch STOPs), hard constraint 1 (STOP), a restart launches nothing; references A0 and P0n; the cortex and white matter columns;
  accepted and rejected counts; revision 1: three separate reading lines, "reached" only at 5/5 (I5, I6), the reproduction check and its
  qualifier (I6), the wording of "not reached" (I7), the 1-minute load gate on GPU run units only (I8), references over the same pairs,
  the six large structures (M5), launch loads; revision 2: arm d (support_shared, theta .05, passed as for P4; four arms, i mod 4), sigma_domain
  and support_threshold in the eta records, batch.json and the config.json check (a mismatch STOPs), the base arm's reach, 诊断点 in the text of
  X2d's readings, X2d - X2c and X2d - P4, the support records; after the review GATEA_rev2 (N3, N4, N6, N7, N8): the readings at full count at
  the level of summary.json, d's run id, the columns of the support table, the place of the GF_B/T_B note, the count of non-brain runs without
  support records, the protected roots; the reviewer's checks of failures, restarts and a missing P4 record;
- slow (-m slow): the dry run in process (real run_pair, CPU torch64, and bayesreg.eval.cli on the synthetic layers; no unit).
"""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from .test_d0_a4_tools import EV, FAKE_PAIRS, QUEUE, ROOT, load, queue_tree  # noqa: F401
from .test_pilot_tools import PAIRS, SHORTS, FakeSystem, InlineSystem, head, runs_of

CHECK = load("gate_a_check_prepared", ROOT/"experiments"/"d0"/"check_prepared.py")


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1))
    return path


# ---- X1: the queue --------------------------------------------------------------------------------------------------------

def run_x1(tree, *args, **env):  # noqa: F811
    done = subprocess.run(["bash", str(QUEUE), "--x1", *args], env=dict(tree["env"], **env), capture_output=True, text=True,
                          timeout=120)
    log = tree["results"]/"gate_a_x1"/"queue.log"
    return done.returncode, done.stdout, log.read_text() if log.exists() else ""


def test_run_queue_x1_settings():
    assert subprocess.run(["bash", "-n", str(QUEUE)], capture_output=True).returncode == 0
    text = QUEUE.read_text()
    assert "D0=$RESULTS/gate_a_x1; ARMS=(X1h X1f); SELF_RE='^tag-gax1-'" in text and "STAGE_PREFIX=GA" in text
    assert 'X1h) ARM_OPTS=(--native-reference identity --native-penalty-scale .5);;' in text
    assert 'X1f) ARM_OPTS=(--native-reference identity);;' in text
    assert '"${OPTS[@]}" "${ARM_OPTS[@]}" --stage "$STAGE_PREFIX-$arm"' in text
    assert 'check_prepared.py "$run" --pair "$pair" --stage "$STAGE_PREFIX-$arm" "${ARM_OPTS[@]}"' in text
    assert "summarize-x1 --x1-runs-root" in text and text.count('exec 9> "$LINK_LOCK"') == 1


def test_run_queue_x1_dry_run_lists_five_pairs_rotating(queue_tree):  # noqa: F811
    code, out, log = run_x1(queue_tree, "--dry-run")
    assert code == 0 and log == ""
    expected = []
    for i, p in enumerate(FAKE_PAIRS[:5]):
        for j in range(2):
            expected.append(f"{len(expected)+1} {('X1h', 'X1f')[(i+j) % 2]} {p}")
    assert out.splitlines() == expected and len(expected) == 10
    assert not (queue_tree["results"]/"gate_a_x1"/"runs").exists()


X1_START_STOPS = {
    "segmenter training": (dict(FAKE_UNITS="tag-unet35-train.service loaded active running x\\n"), "STOP: units still active: tag-unet35-train"),
    "a D0 unit": (dict(FAKE_UNITS="tag-d0-A0-0438-0439.service loaded active running x\\n"), "STOP: units still active: tag-d0-A0"),
    "an A4 unit": (dict(FAKE_UNITS="tag-d0a4-A4p-0438-0439.service loaded active running x\\n"), "STOP: units still active: tag-d0a4"),
    "an A1P unit": (dict(FAKE_UNITS="tag-d0a1p-A1Ph-0438-0439.service loaded active running x\\n"), "STOP: units still active: tag-d0a1p"),
    "GPU compute process": (dict(FAKE_PIDS="4242\\n"), "STOP: GPU not idle at start"),
    "GPU utilization": (dict(FAKE_UTIL="55"), "STOP: GPU not idle at start"),
}


@pytest.mark.parametrize("case", sorted(X1_START_STOPS))
def test_run_queue_x1_start_checks_stop(queue_tree, case):  # noqa: F811
    env, fragment = X1_START_STOPS[case]
    code, _, log = run_x1(queue_tree, **env)
    assert code == 1 and fragment in log, log
    assert "worktree HEAD" not in log


def test_run_queue_x1_needs_no_segmenter_and_stops_at_the_frozen_worktree(queue_tree):  # noqa: F811
    queue_tree["registry"].write_text(queue_tree["registry"].read_text()+"\n")      # the registry plays no part in X1
    code, _, log = run_x1(queue_tree)
    assert code == 1 and "STOP: worktree HEAD is not eb808be" in log, log
    assert "registry" not in log and "units still active" not in log
    assert not list((queue_tree["results"]/"gate_a_x1"/"runs").glob("*/*"))


# ---- X1: check_prepared ----------------------------------------------------------------------------------------------------

T92_TEXT = ("T92 (D2 ablation, user-confirmed 2026-10-19; docs/04 T92) native_reference=identity with lam, mu, nu x 0.5 (the metric "
            "follows): the native level only (the pre-optimizer and the transfer unchanged); an ablation of the model or of the path, "
            "never a recommended configuration; ")


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """A prepared run directory for check_prepared: a small synthetic reference and the arm's protocol."""
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
    reference = dict(stage="r", task=dict(name="t", wall_seconds=90), input_sha256={}, pair_source="r", source_sha256={},
                     seconds=90, scope="A; "+"B; ", weights=[1])
    ref_path = tmp_path/"ref.json"
    ref_path.write_text(json.dumps(dict(reference_protocol=reference, freeze_source_py_sha256=source)))
    monkeypatch.setattr(CHECK, "REF", ref_path)
    proto = dict(reference, stage="GA-X1h", task=dict(name="t", wall_seconds=90, pair="p"),
                 input_sha256={"pair.npz": pair_sha, "manifest.json": manifest_sha},
                 pair_source="data/three_d/stage_L2R_val_v1/p", source_sha256=source, native_reference="identity",
                 native_penalty_scale=.5, scope="A; "+T92_TEXT+"B; ")

    def go(*, change=None, **kwargs):
        doc = dict(proto, **(change or {}))
        (run/"protocol.json").write_text(json.dumps(doc))
        return CHECK.check(run, "p", "GA-X1h", pair_sha, manifest_sha, **kwargs)
    return go


def test_check_prepared_admits_exactly_the_t92_keys(prepared):
    assert prepared(native_reference="identity", native_penalty_scale=.5)["ok"] is True


@pytest.mark.parametrize("change,kwargs,fragment", [
    (dict(native_penalty_scale=.25), dict(native_reference="identity", native_penalty_scale=.5), "native_penalty_scale"),
    (dict(native_reference="fixed"), dict(native_reference="identity", native_penalty_scale=.5), "native_reference"),
    (dict(), dict(native_reference="identity"), "native_penalty_scale"),
    (dict(seconds=120), dict(native_reference="identity", native_penalty_scale=.5), "seconds"),
    (dict(weights=[2]), dict(native_reference="identity", native_penalty_scale=.5), "weights"),
    (dict(scope="A; other; B; "), dict(native_reference="identity", native_penalty_scale=.5), "scope"),
    (dict(), dict(), "native_reference"),            # without the switches every key must be configuration C's
])
def test_check_prepared_refuses_anything_else(prepared, change, kwargs, fragment):
    result = prepared(change=change, **kwargs)
    assert result["ok"] is False and any(fragment in p for p in result["problems"]), result


def test_check_prepared_cli_refuses_scale_without_reference(tmp_path, capsys):
    with pytest.raises(SystemExit):
        CHECK.main([str(tmp_path), "--pair", "p", "--stage", "s", "--expect-pair-sha256", "a", "--expect-manifest-sha256", "b",
                    "--native-penalty-scale", ".5"])


# ---- X1: reading and summary -----------------------------------------------------------------------------------------------

X1_PAIRS = [f"OASIS_OAS1_{438+i:04d}_MR1__OASIS_OAS1_{439+i:04d}_MR1" for i in range(5)]


def x1_metrics(seg35, seg4):
    return dict(seg35=seg35, seg4=seg4, dice_official=seg35, hd95=1.8, sdlogj_official=.08, sdlogj_brain=.6, num_foldings_official=0,
                pct_J_lt_half_brain=.1, cell_centre_J_nonpositive=0, seg4_per_label={"1": seg4-.05, "2": seg4, "3": seg4+.02, "4": seg4},
                seg35_per_label={str(k): seg35 for k in range(1, 36)})


def fake_x1(seg35, seg4, *, passed=True, native_commits=20):
    return dict(metrics=x1_metrics(seg35, seg4), certificate=dict(passed=passed), native_commits=native_commits)


def fake_p0n(seg35, seg4):
    return dict(metrics=x1_metrics(seg35, seg4), certificate=dict(passed=True), field_role="native_formal")


def x1_world(h=.80, f=.82):
    runs = {}
    for p in X1_PAIRS:
        runs[("A0", p)] = fake_x1(.8062, .8702)
        runs[("P0n", p)] = fake_p0n(.7886, .7825)
        runs[("X1h", p)] = fake_x1(.79, h)
        runs[("X1f", p)] = fake_x1(.78, f)
    return runs


@pytest.mark.parametrize("value,key", [(.82, "explains"), (.826, "explains"), (.8264, "middle"), (.86, "middle"), (.8602, "elsewhere"),
                                       (.88, "elsewhere")])
def test_x1_reading_on_seg4(value, key):
    dec = EV.decide_x1(x1_world(h=value), X1_PAIRS)
    assert dec["arms"]["X1h"]["reading"] == key and dec["pin_problems"] == []
    assert dec["arms"]["X1f"]["reading"] == "explains"


def test_x1_reading_needs_four_comparable_pairs_and_certificates():
    runs = x1_world(h=.82)
    runs[("X1h", X1_PAIRS[0])]["certificate"]["passed"] = False
    assert EV.decide_x1(runs, X1_PAIRS)["arms"]["X1h"]["n"] == 4
    assert EV.decide_x1(runs, X1_PAIRS)["arms"]["X1h"]["reading"] == "explains"
    del runs[("X1h", X1_PAIRS[1])]
    assert EV.decide_x1(runs, X1_PAIRS)["arms"]["X1h"]["reading"] == "few"


def test_x1_pins_are_rechecked():
    runs = x1_world()
    runs[("A0", X1_PAIRS[0])]["metrics"]["seg4"] = .80
    assert any("A0" in p for p in EV.decide_x1(runs, X1_PAIRS)["pin_problems"])


# Review I2's reproduction: the real seg4 of A0 and P0n, pair by pair (d0_v1/runs/A0, path_pilot_v1/eval/p0n; the plan's table).
A0_SEG4 = (.8902546281822892, .8328903190313758, .855959317721066, .8835548525661955, .8882398969136043)
P0N_SEG4 = (.8102924088498047, .7604432722272247, .766188364021791, .7797367795597601, .7959950458182388)


def real_baselines():
    runs = {}
    for p, a0, p0 in zip(X1_PAIRS, A0_SEG4, P0N_SEG4):
        runs[("A0", p)], runs[("P0n", p)] = fake_x1(.8062, a0), fake_p0n(.7886, p0)
    return runs


@pytest.mark.parametrize("value,key", [(.862, "middle"), (.83, "explains"), (.834, "explains"), (.8341, "middle"), (.8695, "elsewhere"),
                                       (.8694, "middle")])
def test_x1_thresholds_use_the_arms_own_pairs(value, key):
    """X1h without 0439 <- 0440, the hardest pair: over the other four A0 is .8795, A0 - .01 = .8695, the midpoint .8338 (.834 with the
    plan's three decimals).  The former tool, on the 5-pair .8702 and .826, read .862 as elsewhere and .83 as middle."""
    runs = real_baselines()
    for k, p in enumerate(X1_PAIRS):
        if k != 1:
            runs[("X1h", p)] = fake_x1(.79, value)
        runs[("X1f", p)] = fake_x1(.79, .84)
    dec = EV.decide_x1(runs, X1_PAIRS)
    r, f = dec["arms"]["X1h"], dec["arms"]["X1f"]
    assert dec["pin_problems"] == [] and r["n"] == 4 and X1_PAIRS[1] not in r["pairs"]
    assert r["A0_same_pairs"] == pytest.approx(.8795, abs=5e-5) and r["P0n_same_pairs"] == pytest.approx(.78805, abs=5e-5)
    assert (r["midpoint"], r["near_A0"], r["reading"]) == (.834, .8695, key)
    assert (f["n"], f["midpoint"], f["near_A0"], f["reading"]) == (5, .826, .8602, "middle")     # 5/5: the plan's thresholds, unchanged
    assert f["A0_same_pairs"] == pytest.approx(.8702, abs=5e-5) and f["P0n_same_pairs"] == pytest.approx(.7825, abs=5e-5)


def test_x1_a_state_without_native_commits_is_not_comparable():
    assert EV.x1_comparable(fake_x1(.8, .85)) and EV.x1_comparable(fake_p0n(.8, .85))
    assert not EV.x1_comparable(fake_x1(.8, .85, native_commits=0))      # handover_only: the seed, not the switch
    assert not EV.x1_comparable(fake_x1(.8, .85, native_commits=None))
    assert not EV.x1_comparable(fake_x1(.8, .85, passed=False))
    assert not EV.x1_comparable(dict(fake_p0n(.8, .85), field_role="handover"))
    assert not EV.x1_comparable(None) and not EV.x1_comparable(dict(metrics=None))
    runs = x1_world(h=.82)
    runs[("X1h", X1_PAIRS[2])]["native_commits"] = 0
    r = EV.decide_x1(runs, X1_PAIRS)["arms"]["X1h"]
    assert r["n"] == 4 and X1_PAIRS[2] not in r["pairs"]


def test_x1_labels_stay_within_what_x1_can_answer():
    explains, elsewhere = EV.X1_READINGS["explains"], EV.X1_READINGS["elsewhere"]
    assert "至少" in explains and "一半" in explains and "90 s" in explains
    assert "差距在别处" not in elsewhere and "没有明显压低 seg4" in elsewhere and ".0995" in elsewhere and ".03" in elsewhere and "90 s" in elsewhere
    for piece in ("绝对 μ = ν = .03", "同乘 .5", "E", "F = D + B", "90 s 截断", "不是 E 或 F 的驻点", "提前结束的运行见轨迹表的收尾列", "H2"):
        assert piece in EV.X1_SEMANTICS, piece


# ---- X1: the seed ----------------------------------------------------------------------------------------------------------

def seed_labels(shape=(12, 12, 12)):
    fixed = np.zeros(shape, dtype=np.int32)
    fixed[2:8, 2:8, 2:8], fixed[8:10, 2:8, 2:8] = 1, 3
    moving = np.roll(fixed, 1, axis=0)
    return dict(fixed_seg35=fixed, moving_seg35=moving, fixed_seg4=fixed, moving_seg4=moving)


def test_x1_seed_record(tmp_path):
    labs = seed_labels()
    assert EV.x1_seed_record(tmp_path, labs, (1., 1., 1.)) == dict(present=False, file="level_3_level_prepared.npz")
    u = np.zeros((*labs["fixed_seg4"].shape, 3))
    np.savez(tmp_path/"level_3_level_prepared.npz", u=u)
    still = EV.x1_seed_record(tmp_path, labs, (1., 1., 1.))
    assert still["present"] is True and 0 < still["seg4"] < 1 and still["seg4"] == still["seg35"]
    assert set(still) >= {"sha256", "state_hash", "seg4_per_label", "seg35_per_label"}
    u[..., 2] = 1.                                   # z component, 1 mm: warped(p) = moving(p + e_z) = fixed(p)
    np.savez(tmp_path/"level_3_level_prepared.npz", u=u)
    aligned = EV.x1_seed_record(tmp_path, labs, (1., 1., 1.))
    assert aligned["seg4"] == 1. and aligned["seg35"] == 1. and aligned["state_hash"] != still["state_hash"]


def cleanup_block():
    text = QUEUE.read_text()
    start = text.index("\n", text.index("""$PY -B - "$run" "${KEEP_NPZ[@]}" <<'EOF'"""))+1
    return text[start:text.index("\nEOF\n", start)]+"\n"


@pytest.mark.parametrize("keep,left", [((), set()), (("level_3_level_prepared.npz",), {"level_3_level_prepared.npz"})])
def test_queue_cleanup_keeps_the_seed_only_when_told(tmp_path, keep, left):
    run = tmp_path/"run"
    (run/"t").mkdir(parents=True)
    write_json(run/"protocol.json", dict(task=dict(name="t")))
    for name in ("committed_0001.npz", "level_2_level_prepared.npz", "level_3_level_prepared.npz", "level_3_other.npz", "final_native.npz", "x.npz"):
        (run/"t"/name).write_bytes(b"x")
    done = subprocess.run([sys.executable, "-B", "-", str(run), *keep], input=cleanup_block(), text=True, capture_output=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert {p.name for p in (run/"t").glob("*.npz")} == {"final_native.npz", "x.npz"} | left
    record = json.loads((run/"cleanup.json").read_text())
    assert ("level_3_level_prepared.npz" in [g["name"] for g in record["deleted"]]) == (not keep)
    if keep:
        assert "level_3_level_prepared.npz" in record["rule_exception"] and "revision 1" in record["rule_exception"]
    else:                                            # D0, --a4, --a1p: the record has the same keys as before
        assert set(record) == {"rule", "deleted", "deleted_bytes", "kept_npz"}


def test_queue_keeps_the_seed_only_in_x1_mode():
    text = QUEUE.read_text()
    assert text.count("KEEP_NPZ=") == 2 and "EXTRA_BUSY=(); STAGE_PREFIX=D0; KEEP_NPZ=()" in text
    x1 = text.index("elif [ $MODE = x1 ]; then")
    assert x1 < text.index("KEEP_NPZ=(level_3_level_prepared.npz)") < text.index("\nelse\n", x1)
    done = subprocess.run(["bash", "-c", 'set -u; K=(); printf "%s\\n" run "${K[@]}"'], capture_output=True, text=True)
    assert done.returncode == 0 and done.stdout == "run\n"          # with no seed to keep the block gets the same argv as before


# ---- X1: the summary --------------------------------------------------------------------------------------------------------

def x1_seed(seg35, seg4):
    return dict(present=True, file="level_3_level_prepared.npz", sha256="s", state_hash="h", seg35=seg35, seg4=seg4,
                seg4_per_label={"1": seg4-.05, "3": seg4+.02}, seg35_per_label={str(k): seg35 for k in range(1, 36)})


def write_frozen_run(root, arm, pair, seg35, seg4, *, scale=None, seed=None, passed=True, commits=20):
    d = root/arm/pair
    (d/"task").mkdir(parents=True)
    document = dict(run_id=f"{arm}_{pair[11:15]}", run_dir=str(d), arm=arm, pair=pair, metrics=x1_metrics(seg35, seg4),
                    certificate=dict(passed=passed, min_jd_lower_bound=1e-6), self_check=dict(passed=True),
                    native_commits=commits, e2e_seconds=88.5, success=False)
    if seed is not None:
        document["seed"] = seed
    write_json(d/"metrics.json", document)
    write_json(d/"protocol.json", dict(task=dict(name="task"), native_reference="identity", native_penalty_scale=scale))
    write_json(d/"task"/"summary.json", dict(native_status="outer_time_budget", status="native_budget_exhausted", reason="x", energy=dict(image=9.)))
    (d/"log.txt").write_text("2026-11-02T20:00:00Z launch tag-gax1-x; loadavg 0.25 0.50 0.75\n")
    rows = [dict(level=2, outer_accepted=True, E_start=99., F_start=99.),                      # another level: not this table's
            dict(level=3, outer_accepted=True, E_start=20., F_start=10., D_start=8., D_candidate=6., committed_E=18., committed_F=11.,
                 committed_mse_ratio=.3, inner_accepted_steps=5, F_decrease_certified=False),
            dict(level=3, outer_accepted=False, E_start=18., F_start=11., D_start=6., D_candidate=None, inner_accepted_steps=7),   # appended, not accepted
            dict(level=3, outer_accepted=True, E_start=18., F_start=11., D_start=6., D_candidate=5., committed_E=15., committed_F=10.5,
                 committed_mse_ratio=.25, inner_accepted_steps=5, F_decrease_certified=True)]
    (d/"task"/"history.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))


def test_x1_trajectory_counts_commits_only(tmp_path):
    write_frozen_run(tmp_path, "X1f", X1_PAIRS[0], .78, .80)
    t = EV.x1_trajectory(tmp_path/"X1f"/X1_PAIRS[0])
    assert (t["commits"], t["rows_not_accepted"], t["inner_accepted"], t["F_certified"], t["F_recorded"]) == (2, 1, 10, 1, 2)
    assert (t["E_first"], t["E_last"], t["F_first"], t["F_last"]) == (20., 15., 10., 10.5)
    assert (t["ratio_first"], t["ratio_last"]) == (.3, .25) and t["ratio_seed"] == pytest.approx(8*.3/6)


def x1_tree(tmp_path, failing=(), no_commit=()):
    stage = tmp_path/"stage"
    stage.mkdir()
    write_json(stage/"selection.json", dict(pairs=[dict(pair=p) for p in (*X1_PAIRS, "later_pair")]))
    d0, x1, pilot = tmp_path/"d0", tmp_path/"x1", tmp_path/"pilot"
    for k, p in enumerate(X1_PAIRS):
        write_frozen_run(d0, "A0", p, .8062, .8702)
        write_frozen_run(x1, "X1h", p, .79, .84, scale=.5, seed=x1_seed(.80, .838))
        write_frozen_run(x1, "X1f", p, .78, .80, seed=x1_seed(.80, .838), passed=k not in failing, commits=0 if k in no_commit else 20)
        short = f"{p[11:15]}-{p[-8:-4]}"
        metrics = x1_metrics(.7886, .7825)
        labels = dict(seg4_per_label=metrics.pop("seg4_per_label"), seg35_per_label=metrics.pop("seg35_per_label"))
        write_json(pilot/"eval"/"p0n"/short/f"pilot_p0n_{short}"/"metrics.json", dict(
            run_id=f"pilot_p0n_{short}", metrics=metrics, per_label=labels, certificate=dict(passed=True), field_role="native_formal"))
    return d0, x1, pilot, stage


def test_summarize_x1(tmp_path):
    d0, x1, pilot, stage = x1_tree(tmp_path)
    out = tmp_path/"analysis"
    assert EV.summarize_x1(x1, d0, pilot, out, stage=stage) == 0
    dec = json.loads((out/"decision_x1.json").read_text())
    assert dec["arms"]["X1h"]["reading"] == "middle" and dec["arms"]["X1f"]["reading"] == "explains" and dec["pin_problems"] == []
    assert dec["sources"]["pairs"] == X1_PAIRS and dec["means"]["P0n"]["n"] == 5
    t = dec["trajectories"][f"X1h/{X1_PAIRS[0]}"]
    assert (t["commits"], t["inner_accepted"], t["F_certified"], t["E_first"], t["E_last"]) == (2, 10, 1, 20., 15.)
    assert (t["native_reference"], t["native_penalty_scale"]) == ("identity", .5)
    s = dec["seed"]["X1h"]                         # the seed: descriptive, not in the reading
    assert s["n"] == 5 and s["seg4"] == pytest.approx(.838) and s["seg35"] == pytest.approx(.80)
    assert s["arm_minus_seed"]["seg4"]["mean"] == pytest.approx(.002) and s["arm_minus_seed"]["seg4"]["positive"] == 5
    assert s["arm_minus_seed"]["seg35"]["mean"] == pytest.approx(-.01)
    assert s["seed_seg4_minus_midpoint"] == pytest.approx(.012) and s["near_A0_minus_seed_seg4"] == pytest.approx(.0222)
    assert dec["seg35_contributions"]["X1h"]["P0n"]["large_six"] == pytest.approx(6*(.79-.7886)/35)
    assert dec["seg35_contributions"]["X1h"]["P0n"]["other_29"] == pytest.approx(29*(.79-.7886)/35)
    assert dec["launch_loadavg"][f"X1h/{X1_PAIRS[0]}"] == [.25, .5, .75]
    text = (out/"summary_x1.md").read_text()
    for fragment in ("seg35", "seg4", "皮层", "白质", "X1h − A0", "X1f − P0n", "n = 5", "只有方向信息", "读数不是结论", "不是同时段对照",
                     "F 也有下降证书的提交", "差距", "开发集", "X1h 的种子", "只作描述", "X1 − 种子 seg4", "种子 seg4 − 中点", "六个大结构", "其余 29 个",
                     "启动负载", "90 s 截断", "至少", "MSE 比 种子"):
        assert fragment in text, fragment
    assert "差距在别处" not in text
    rows = (out/"long.csv").read_text().splitlines()
    assert any(",X1h," in r and ",seg4," in r for r in rows)
    assert any("frozen_eb808be_T92_identity_s0.5_90s,X1h," in r for r in rows) and any("configC_frozen_eb808be_90s,A0," in r for r in rows)
    with pytest.raises(SystemExit, match="never written under"):
        EV.summarize_x1(x1, d0, pilot, EV.D0/"analysis_x1", stage=stage)
    with pytest.raises(SystemExit, match="never written under"):
        EV.summarize_x1(x1, d0, pilot, pilot/"analysis", stage=stage)


def test_summarize_x1_means_only_over_comparable_runs(tmp_path):
    d0, x1, pilot, stage = x1_tree(tmp_path, failing=(0,), no_commit=(1,))
    out = tmp_path/"analysis"
    assert EV.summarize_x1(x1, d0, pilot, out, stage=stage) == 0
    dec = json.loads((out/"decision_x1.json").read_text())
    assert dec["arms"]["X1f"]["n"] == 3 and dec["arms"]["X1f"]["reading"] == "few" and dec["means"]["X1f"]["n"] == 3
    assert dec["seed"]["X1f"]["n"] == 5 and dec["seed"]["X1f"]["arm_minus_seed"]["seg4"]["n"] == 3
    text = (out/"summary_x1.md").read_text()
    row = next(line for line in text.splitlines() if line.startswith("| X1f（"))
    assert "| 3/5 |" in row and "| 4/5 |" in row      # comparable pairs, certificates passed of all runs
    assert any(line.startswith("| X1f − A0 | seg4 |") and "/ 3 |" in line for line in text.splitlines())


# ---- X2: the driver with a fake machine ---------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def x2():
    return load("gate_a_run_x2", ROOT/"experiments"/"gate_a"/"run_x2.py")


@pytest.fixture(scope="module")
def dry_mod():
    return load("gate_a_m2v1_dry_inputs", ROOT/"experiments"/"m2v1"/"dry_inputs.py")


def x2_common(dry, wall="60"):
    return ["--dry-run", "--data-root", str(dry/"data"), "--allowlist", str(dry/"dev_subjects.json"), "--tissue-root", str(dry/"seg"),
            "--structure-root", str(dry/"structure"), "--segmenter-registry", str(dry/"registry"/"segmenters.json"), "--wall-seconds", wall,
            "--a0-runs-root", str(dry/"refs"/"A0"), "--pilot-root", str(dry/"refs"/"pilot")]


def write_references(dry):
    for pair, short in zip(PAIRS, SHORTS):
        write_json(dry/"refs"/"A0"/pair/"metrics.json", dict(metrics=dict(seg35=.81, seg4=.87, hd95=2., sdlogj_official=.08, sdlogj_brain=.6,
                                                                          seg4_per_label={"1": .82, "3": .90},
                                                                          seg35_per_label={str(k): .81 for k in range(1, 36)}),
                                                             certificate=dict(passed=True, min_jd_lower_bound=1e-6), run_id=f"d0v1_A0_{short}"))
        write_json(dry/"refs"/"pilot"/"eval"/"p0n"/short/f"pilot_p0n_{short}"/"metrics.json", dict(
            metrics=dict(seg35=.79, seg4=.78, hd95=2., sdlogj_official=.06, sdlogj_brain=.4), per_label=dict(seg4_per_label={"1": .62, "3": .77}, seg35_per_label={str(k): .79 for k in range(1, 36)}),
            certificate=dict(passed=True, min_jd_lower_bound=1e-4), run_id=f"pilot_p0n_{short}"))
        write_json(dry/"refs"/"pilot"/"eval"/"p4"/short/f"pilot_p4_{short}"/"metrics.json", dict(
            metrics=dict(seg35=.77, seg4=.80, hd95=2.3, sdlogj_official=.07, sdlogj_brain=.5), per_label=dict(seg4_per_label={"1": .66, "3": .79}, seg35_per_label={str(k): .77 for k in range(1, 36)}),
            certificate=dict(passed=True, min_jd_lower_bound=1e-4), run_id=f"pilot_p4_{short}"))


class X2Fake(FakeSystem):
    """FakeSystem with Dice that depend on the arm's eta and channels, and per-label Dice."""

    def __init__(self, *, seg4=None, **kwargs):
        super().__init__(**kwargs)
        self.seg4 = seg4 or {}

    def run(self, cmd, record):
        code = super().run(cmd, record)
        unit, _, _, argv = self.split(cmd)
        if "-eval-" in unit:
            argv = [str(a) for a in argv]
            ev, rid = Path(argv[argv.index("--out-dir")+1]), argv[argv.index("--run-id")+1]
            path = ev/rid/"metrics.json"
            doc = json.loads(path.read_text())
            arm = rid.split("_")[1]
            seg4 = self.seg4.get(arm, .86)
            doc["metrics"].update(seg35=.81, seg4=seg4)
            doc["per_label"] = dict(seg4_per_label={"1": seg4-.03, "2": seg4, "3": seg4+.02, "4": seg4},
                                    seg35_per_label={str(k): .81 for k in range(1, 36)})
            path.write_text(json.dumps(doc))
        return code


@pytest.fixture
def inputs(dry_mod, tmp_path):
    dry = tmp_path/"dry"
    dry_mod.main(["--out", str(dry), "--wall-seconds", "60", "--extra", "", "--code-sha", head()])
    write_references(dry)
    return dry


@pytest.fixture
def driver(x2, monkeypatch):
    sha = head()

    def prepare(self):
        self.code, self.code_sha = ROOT, sha
        self.allowlist = self.allowlist or self.o.allowlist
    monkeypatch.setattr(x2.X2Arm, "prepare_code", prepare)
    monkeypatch.setattr(x2.X2Arm, "write_env", lambda self: None)

    def go(root, system, dry, *extra):
        return x2.main(["--root", str(root), *x2_common(dry), *extra], system=system)
    return go


def launch_text(root, arm, short):
    return next((root/"launch").glob(f"run_{arm}-{short}_*.txt")).read_text()


def test_x2_arms_order_records_summary_and_restart(driver, inputs, tmp_path):
    root = tmp_path/"x2"
    system = X2Fake()
    assert driver(root, system, inputs) == 0, (root/"queue.log").read_text()[-3000:]
    expected = []
    for i, short in enumerate(SHORTS):
        order = ["a", "b", "c", "d"]
        order = order[i:]+order[:i]
        expected += [f"tag-gax2dry-{a}-{short}" for a in order]
    assert runs_of(system) == expected
    for aid, channel, nu, tau in (("a", "tissue", 3.87, [1., .5, .5, .5]), ("b", "tissue", 1.37, [1., 1., 1., 1.]),
                                  ("c", "structure", 1.37, [1., 1., 1., 1.]), ("d", "structure", 1.37, [1., 1., 1., 1.])):
        record = json.loads((root/"eta"/f"eta_{aid}.json").read_text())
        assert (record["alpha"], record["nu"], record["tau"], record["channel_source"]) == (.01, nu, tau, channel)
        assert (record["sigma_domain"], record["support_threshold"]) == (("support_shared", .05) if aid == "d" else ("brain", None))
        assert ("诊断点" in record["eta_role"]) == (aid == "d") and "C04-6" in json.loads((root/"eta"/"eta_d.json").read_text())["eta_role"]
        text = launch_text(root, aid, SHORTS[1])
        for piece in (f"--channel-source {channel}", f"--eta-json {root/'eta'/f'eta_{aid}.json'}", "--logvol-points gauss27+closed27",
                      "--jd-cell-estimate certificate", "--wall-seconds 60", "--rounds 1,1,1,1", "--device cpu", "--nice=19",
                      "--property=MemoryMax=9G", "--pair stage_L2R_val_v1/"):
            assert piece in text, (aid, piece)
        assert ("--structure-root" in text) == (channel == "structure")
        assert "--init-field" not in text and "--max-accepted" not in text and "certificate_bernstein" not in text
        assert ("--sigma-domain support_shared --support-threshold 0.05" in text) == (aid == "d") and ("--sigma-domain" in text) == (aid == "d")
    batch = json.loads((root/"batch.json").read_text())
    assert batch["pairs"] == list(PAIRS) and batch["arm_order"] == ["a", "b", "c", "d"] and batch["method"] == "bayesreg_gate_a_x2"
    assert batch["arms"]["c"]["expected_args"]["channel_source"] == "structure"
    assert batch["arms"]["a"]["expected_args"]["channel_source"] == "tissue" and "structure_root" not in batch["arms"]["a"]["expected_args"]
    for aid in "abc":
        expected_args = batch["arms"][aid]["expected_args"]
        assert (expected_args["sigma_domain"], expected_args["support_threshold"]) == ("brain", None), aid
        assert (batch["arms"][aid]["sigma_domain"], batch["arms"][aid]["support_threshold"]) == ("brain", None), aid
    arm_d, arm_c = batch["arms"]["d"], batch["arms"]["c"]
    assert (arm_d["expected_args"]["sigma_domain"], arm_d["expected_args"]["support_threshold"]) == ("support_shared", .05)
    assert (arm_d["sigma_domain"], arm_d["support_threshold"]) == ("support_shared", .05) and "诊断点" in arm_d["eta_role"] and "诊断点" in arm_d["note"]
    assert arm_d["eta"] == arm_c["eta"] and arm_d["eval_arm"] == "X2d" and arm_d["unit_prefix"] == "tag-gax2dry-d-"
    differ = {k for k in {*arm_d["expected_args"], *arm_c["expected_args"]} if arm_d["expected_args"].get(k) != arm_c["expected_args"].get(k)}
    assert differ == {"sigma_domain", "support_threshold", "eta_json"}          # X2d is X2c with the sigma domain (and its own eta record)
    differ = {k for k in {*arm_d["configuration"], *arm_c["configuration"]} if arm_d["configuration"].get(k) != arm_c["configuration"].get(k)}
    assert differ == {"sigma_domain", "support_threshold"}
    s = json.loads((root/"summary.json").read_text())
    assert s["counts"]["ran"] == 12 and s["counts"]["args_mismatch"] == 0 and s["counts"]["eta_mismatch"] == 0
    assert s["arm_means"]["a"]["n"] == 3 and s["arm_means"]["a"]["cortex"] == pytest.approx(.83) and s["reference_means"]["A0"]["n"] == 3
    assert s["reference_means"]["P0n"]["white_matter"] == pytest.approx(.77)
    assert s["comparisons"]["b"]["A0"]["seg4"]["median"] == pytest.approx(-.01) and s["comparisons"]["b"]["P0n"]["seg4"]["positive"] == 3
    assert s["extra_comparisons"]["c_vs_b"]["comparison"]["seg4"]["n"] == 3
    assert all(line.startswith("只描述（可比 3/5") for arm in s["readings"].values() for line in arm.values())    # three synthetic pairs
    assert all(("诊断点" in line) == (aid == "d") for aid, arm in s["readings"].items() for line in arm.values())
    assert s["extra_comparisons"]["d_vs_c"]["comparison"]["seg4"]["n"] == 3 and s["extra_comparisons"]["d_vs_c"]["comparison"]["seg4"]["median"] == pytest.approx(0.)
    p4 = s["extra_comparisons"]["d_vs_P4"]
    assert (p4["first"], p4["second"]) == ("X2d", "P4") and p4["comparison"]["seg4"]["n"] == 3 and p4["comparison"]["seg4"]["median"] == pytest.approx(.06)
    assert p4["seg35_split"]["large_six"] == pytest.approx(6*.04/35) and p4["seg35_split"]["other_29"] == pytest.approx(29*.04/35)
    assert s["p4"]["means"]["seg4"] == pytest.approx(.80) and s["p4"]["code"] == "ffc6844" and sorted(s["reference_means"]) == ["A0", "P0n"]
    assert sorted(s["comparisons"]["d"]) == ["A0", "P0n"] and sorted(s["seg35_split"]["a"]) == ["A0", "P0n"]    # P4 goes only with X2d
    assert s["seg35_large_means"]["P4"]["2"] == pytest.approx(.77)
    assert s["rows"]["d"][0]["support"][0]["N_S"] == 60. and s["rows"]["c"][0]["support"] is None
    assert s["reproduction"]["state"].startswith("未判") and s["reproduction"]["qualifier"] is None    # no 0438 <- 0439 among them
    assert s["dry_run"] is True and set(s["bars_check"]) == {"cortex", "white_matter"}
    assert s["seg35_split"]["a"]["P0n"]["large_six"] == pytest.approx(6*.02/35) and s["seg35_split"]["a"]["P0n"]["other_29"] == pytest.approx(29*.02/35)
    assert s["seg35_large_means"]["P0n"]["2"] == pytest.approx(.79) and s["rows"]["a"][0]["launch_loadavg"] == [0., 0., 0.]
    counts = s["rows"]["a"][0]["counts"]
    assert counts["native"]["accepted"] == 20 and counts["all"]["rejected"] == 40 and counts["rejections_by_reason"]["JACOBIAN_VIOLATION"] == 40
    md = (root/"summary.md").read_text()
    for piece in ("seg35", "seg4", "皮层", "白质", "X2c − X2b", "n = 5", "只有方向信息", "读数不是结论", "VAL4C", "G3", "接受步", "拒绝", "受阻层", "各层读数",
                  "预算收尾", "不是同时段对照", "不要直接比", "不能用来否定目标函数", "原生接受步中位", "六个大结构", "其余 29 个", "复现核对", "启动负载", "ffc6844",
                  "没有重新验证", "seg35 拆分", "X2d − X2c", "X2d − P4", "σ 估计域", "support_shared（θ 0.05）", "诊断点，不是合同选出的 η", "重做探针与扫描",
                  "按层上限", "certificate_bernstein", "P4（旁证", "σ 估计域的支撑记录", "不可比"):
        assert piece in md, piece
    assert "门槛与 A0、P0n 同批对的中点不符" not in md           # a dry run: synthetic references
    assert s["counts"]["non_brain_without_support"] == 0 and s["non_brain_without_support"] == [] and "结束了却没有 support 记录" not in md
    readings =md[md.index("## 读数（预登记，修订 1、2"):md.index("各层读数")]
    assert readings[:readings.index("**X2d**")].count("诊断点") == 0          # in the readings only X2d carries it, on each of its lines
    assert readings[readings.index("**X2d**"):].count("诊断点，不是合同选出的 η") == 3 and "重做探针与扫描" in readings[readings.index("**X2d**"):]
    again = X2Fake()
    assert driver(root, again, inputs) == 0 and again.units == []
    assert "(unchanged)" in (root/"queue.log").read_text()


def x2_mean(**change):
    return dict(dict(n=5, seg4=.85, seg35=.80, cortex=.7191, white_matter=.8366), **change)


def test_x2_reading_lines_are_separate_and_need_five_of_five(x2):
    sx = x2.sx2
    assert sx.READING_NO == "未达到（900 s，现行路径）" and sx.READING_LARGE == "目标函数改动在现行求解器上把皮层、白质带回了 A0 与 P0n 差距的一半以上"
    all_reached = dict(seg4=sx.READING_SEG4, large_structures=sx.READING_LARGE, seg35=sx.READING_SEG35)
    assert sx.readings_of(x2_mean()) == all_reached
    for change, line in ((dict(seg4=.8499), "seg4"), (dict(seg35=.7999), "seg35"), (dict(cortex=.719), "large_structures"),
                         (dict(white_matter=.8365), "large_structures"), (dict(cortex=None), "large_structures")):
        assert sx.readings_of(x2_mean(**change)) == dict(all_reached, **{line: sx.READING_NO}), change
    for n in (4, 3, 0):
        readings = sx.readings_of(x2_mean(n=n, seg4=.95, seg35=.95, cortex=.95, white_matter=.95))
        assert all(line.startswith(f"只描述（可比 {n}/5") for line in readings.values()), n
    assert sx.readings_of(x2_mean(n=4, seg4=.5))["seg4"].startswith("只描述")     # described only, below the bar too


def test_x2d_readings_use_the_same_bars_and_carry_the_diagnostic_qualifier(x2):
    sx = x2.sx2
    mark = "（诊断点，不是合同选出的 η）"
    assert sx.arm_qualifier(dict(sigma_domain="support_shared")) == sx.DIAGNOSTIC and sx.arm_qualifier(dict(sigma_domain="brain")) is None
    reached = sx.readings_of(x2_mean(), sx.DIAGNOSTIC)
    assert reached == {k: v+mark for k, v in sx.readings_of(x2_mean()).items()}
    missed = sx.readings_of(x2_mean(cortex=.719), sx.DIAGNOSTIC)
    assert missed["large_structures"] == sx.READING_NO+mark and missed["seg4"] == sx.READING_SEG4+mark and missed["seg35"] == sx.READING_SEG35+mark
    assert all(line.startswith("只描述（可比 4/5") and line.endswith(mark) for line in sx.readings_of(x2_mean(n=4), sx.DIAGNOSTIC).values())


def test_x2_bars_are_the_midpoints_of_a0_and_p0n(x2):
    sx = x2.sx2
    means = dict(A0=dict(cortex=.8168, white_matter=.9033), P0n=dict(cortex=.6213, white_matter=.7698))      # the plan's table
    check = sx.bars_check(means)
    assert check["cortex"]["ok"] and check["white_matter"]["ok"]
    assert (.8168+.6213)/2 == pytest.approx(sx.CORTEX_BAR, abs=5.01e-5) and (.9033+.7698)/2 == pytest.approx(sx.WM_BAR, abs=5.01e-5)    # .71905, .83655
    exact = dict(A0=dict(cortex=.81675, white_matter=.90328), P0n=dict(cortex=.62133, white_matter=.7698))    # exact means: within 1e-4
    assert sx.bars_check(exact)["cortex"]["ok"]
    assert not sx.bars_check(dict(means, P0n=dict(cortex=.70, white_matter=.7698)))["cortex"]["ok"]
    assert not sx.bars_check(dict(means, P0n=dict(cortex=None, white_matter=.7698)))["cortex"]["ok"]


def test_x2_reproduction_check(x2):
    sx = x2.sx2
    pairs = [sx.G3_REF["pair"], "another"]

    def state(value, comparable=True, arm="a", names=pairs):
        return sx.reproduction({arm: [dict(comparable=comparable, metrics=dict(seg4=value)), dict(comparable=True, metrics=dict(seg4=.5))]}, names)
    assert sx.G3_SEG4_BAR == .8445 and state(.8445)["state"] == sx.REPRO_YES and state(.8445)["qualifier"] is None
    for result in (state(.8444), state(.95, comparable=False), state(None)):
        assert result["state"] == sx.REPRO_NO and result["qualifier"] == sx.QUALIFIER
    assert state(.95, arm="b")["state"].startswith("未判") and state(.95, names=["x", "y"])["state"].startswith("未判")


def test_x2_qualifier_goes_on_b_c_and_d(driver, inputs, tmp_path, x2, monkeypatch):
    monkeypatch.setitem(x2.sx2.G3_REF, "pair", PAIRS[0])
    root = tmp_path/"x2"
    assert driver(root, X2Fake(seg4=dict(a=.80)), inputs) == 0, (root/"queue.log").read_text()[-3000:]
    s = json.loads((root/"summary.json").read_text())
    assert s["reproduction"]["state"] == x2.sx2.REPRO_NO and s["reproduction"]["value"] == pytest.approx(.80)
    md = (root/"summary.md").read_text()
    reading = md[md.index("## 读数"):md.index("各层读数")]
    assert x2.sx2.REPRO_NO in reading and reading.count(x2.sx2.QUALIFIER) == 3
    assert reading.index("**X2a**") < reading.index("**X2b**") < reading.index(x2.sx2.QUALIFIER) < reading.index("**X2c**")
    assert x2.sx2.QUALIFIER in reading[reading.index("**X2d**"):]


def test_x2_load_gate_is_on_gpu_run_units_only(x2, tmp_path):
    class Loaded(X2Fake):
        load = "2.00 1.00 1.00"

        def loadavg(self):
            return self.load
    system = Loaded()
    assert x2.parse_args(["--root", str(tmp_path)]).wait_minutes == 30
    (tmp_path/"x2").mkdir()
    arm = x2.X2Queue(x2.parse_args(["--root", str(tmp_path/"x2")]), system).arms["a"]
    assert "1-minute load 2.00 >= 1.5" in arm.busy_reason(1000, True) and arm.busy_reason(1000, False) == ""
    system.load = "1.50 0 0"
    assert "1-minute load 1.50 >= 1.5" in arm.busy_reason(1000, True)
    system.load = "1.49 9.00 9.00"
    assert arm.busy_reason(1000, True) == ""
    system.load, system.on_sleep = "3.00 0 0", (lambda s: setattr(s, "load", "1.00 0 0"))
    arm.gate("run x", 1000, True)                    # polls, then goes on
    assert system.slept == 1
    system.load, system.on_sleep = "3.00 0 0", None
    with pytest.raises(Exception, match="1-minute load 3.00"):
        arm.gate("run x", 1000, True)                # --wait-minutes used up: STOP
    assert system.slept == 31
    hold = tmp_path/"x2"/"hold"
    hold.write_text("")
    assert "hold file" in arm.busy_reason(1000, True)      # the inherited gate comes first


def test_x2_launch_load_same_pairs_and_split(x2, tmp_path):
    sx = x2.sx2
    path = tmp_path/"launch.txt"
    path.write_text("cmd\nbefore 2026-11-01T03:16:11Z: uptime 878519.51; loadavg 3.65 4.86 4.10; MemAvailable 12486 MiB; /home free 581 GB\n")
    assert sx.launch_load(path) == [3.65, 4.86, 4.10] and sx.launch_load(tmp_path/"none") is None
    (tmp_path/"other.txt").write_text("no load here\n")
    assert sx.launch_load(tmp_path/"other.txt") is None

    def row(pair, seg4, comparable=True, labels=None):
        return dict(pair=pair, comparable=comparable, metrics=dict(seg4=seg4), seg35_labels=labels or {})
    arm = [row("p", .9, labels={"1": .8, "3": .6}), row("q", .9, comparable=False, labels={"1": .1, "3": .1}), row("r", .9, labels={"1": .8, "3": .6})]
    a0 = [row("p", .80, labels={"1": .5, "3": .5}), row("q", .70, labels={"1": .9, "3": .9}), row("r", .90, labels={"1": .5, "3": .5})]
    same = sx.same_pairs_means(arm, a0)
    assert same["n"] == 2 and same["seg4"] == pytest.approx(.85)      # review I5: the arm's own pairs, not the hard pair it lost
    split = sx.seg35_split(arm, a0)
    assert split["large_six"] == pytest.approx(.3/35) and split["other_29"] == pytest.approx(.1/35) and split["n_other_above"] == 1
    assert sx.label_means(arm, {1: "x"})[1] == pytest.approx(.8)


def test_x2_config_mismatch_stops(driver, inputs, tmp_path):
    root = tmp_path/"x2"
    system = X2Fake(mutate=("-a-", dict(channel_source="structure")))
    assert driver(root, system, inputs) == 8
    assert "args differ from the plan" in (root/"queue.log").read_text()


@pytest.mark.parametrize("arm,change,key", [
    ("d", dict(sigma_domain="brain", support_threshold=None), "sigma_domain"),     # the run lost the switches of arm d
    ("d", dict(sigma_domain="support"), "sigma_domain"),
    ("d", dict(support_threshold=.1), "support_threshold"),
    ("d", dict(support_threshold=None), "support_threshold"),
    ("c", dict(sigma_domain="support_shared", support_threshold=.05), "sigma_domain"),   # a brain arm that ran under support_shared
    ("a", dict(support_threshold=.05), "support_threshold"),
])
def test_x2_sigma_domain_mismatch_stops(driver, inputs, tmp_path, arm, change, key):
    root = tmp_path/"x2"
    system = X2Fake(mutate=(f"-{arm}-", change))
    assert driver(root, system, inputs) == 8
    log = (root/"queue.log").read_text()
    assert "args differ from the plan" in log and f"'{key}'" in log, log[-2000:]
    assert not (root/"summary.json").exists()
    assert f"tag-gax2dry-eval-{arm}-{SHORTS[0]}" not in system.units       # stopped before its evaluation


def test_x2_arm_d_is_arm_c_with_the_sigma_domain_and_the_base_arm_stays_brain(x2, tmp_path):
    assert x2.DEFAULT_ARMS == "a,b,c,d" and x2.arm_eta(x2.ARMS["d"]) == dict(alpha=.01, nu=1.37, tau=[1., 1., 1., 1.])
    d, c = x2.ARMS["d"], x2.ARMS["c"]
    assert (d["sigma"], d["theta"]) == (x2.pl.ARMS["p4"]["sigma"], x2.pl.ARMS["p4"]["theta"]) == ("support_shared", .05)    # as the pilot's P4
    assert {k for k in {*d, *c} if d.get(k) != c.get(k)} == {"label", "note", "sigma", "theta", "eta_role"}
    queue = x2.X2Queue(x2.parse_args(["--root", str(tmp_path/"x2")]), X2Fake())
    assert (queue.base.support, queue.base.sigma_domain, queue.base.support_threshold) == (False, "brain", None)
    assert (queue.arms["d"].support, queue.arms["d"].sigma_domain, queue.arms["d"].support_threshold) == (True, "support_shared", .05)
    for aid in "abc":
        assert (queue.arms[aid].support, queue.arms[aid].sigma_domain, queue.arms[aid].support_threshold) == (False, "brain", None), aid
    assert queue.base.need_flags == ("--sigma-domain", "support_shared", "--support-threshold")     # the base arm alone checks the code tree
    args = queue.arms["d"].run_args(PAIRS[0], "eta.json")
    assert args[args.index("--sigma-domain")+1] == "support_shared" and args[args.index("--support-threshold")+1] == "0.05"
    assert "--sigma-domain" not in queue.base.run_args(PAIRS[0], "eta.json") and "--sigma-domain" not in queue.arms["c"].run_args(PAIRS[0], "eta.json")
    without = x2.X2Queue(x2.parse_args(["--root", str(tmp_path/"x2"), "--arms", "a,b,c"]), X2Fake())
    assert without.base.need_flags == () and sorted(without.arms) == ["a", "b", "c"]


def test_x2_readings_at_full_count_come_from_each_arms_own_means(driver, inputs, tmp_path, x2, monkeypatch):
    """Review GATEA_rev2 N4 (a), (b): with REQUIRED_N = the three synthetic pairs the summary reaches "reached" and "not reached"; each arm's
    lines read its own means, never P4's or another arm's, and d's records carry d's run id."""
    sx = x2.sx2
    mark = "（诊断点，不是合同选出的 η）"
    monkeypatch.setattr(sx, "REQUIRED_N", 3)
    root = tmp_path/"x2"
    assert driver(root, X2Fake(seg4=dict(a=.86, b=.86, c=.86, d=.84)), inputs) == 0, (root/"queue.log").read_text()[-2000:]
    s = json.loads((root/"summary.json").read_text())
    assert s["readings"]["c"] == dict(seg4=sx.READING_SEG4, large_structures=sx.READING_LARGE, seg35=sx.READING_SEG35)
    # X2d: seg4 .84 < .85; cortex .81 and white matter .86 pass; seg35 .81 passes: "not reached" on its own line only
    assert s["readings"]["d"] == dict(seg4=sx.READING_NO+mark, large_structures=sx.READING_LARGE+mark, seg35=sx.READING_SEG35+mark)
    assert s["arm_means"]["d"]["seg4"] == pytest.approx(.84) and s["p4"]["means"]["seg4"] == pytest.approx(.80)
    assert s["extra_comparisons"]["d_vs_c"]["comparison"]["seg4"]["median"] == pytest.approx(-.02)
    assert s["extra_comparisons"]["d_vs_P4"]["comparison"]["seg4"]["median"] == pytest.approx(.04)
    assert s["rows"]["d"][0]["run_id"] == f"gax2_d_{SHORTS[0]}"
    doc = json.loads((root/"eval"/"d"/SHORTS[0]/f"gax2_d_{SHORTS[0]}"/"metrics.json").read_text())
    assert doc["run_id"] == f"gax2_d_{SHORTS[0]}"
    md = (root/"summary.md").read_text()
    assert f"seg4 均值 0.8400（≥ 0.85）：{sx.READING_NO}{mark}；" in md and f"seg4 均值 0.8600（≥ 0.85）：{sx.READING_SEG4}；" in md


def test_x2_support_table_columns_and_the_gf_note(driver, inputs, tmp_path, x2):
    """Review GATEA_rev2 N3, N4 (c), (d), N8: X2Fake's records are N_q 1000, N_S 60, N_c/N_q .05 and .01, one empty channel on level 0,
    sigma_S^2 .003 at the level start, .004 used, .005 after the round."""
    sx = x2.sx2
    root = tmp_path/"x2"
    assert driver(root, X2Fake(), inputs) == 0, (root/"queue.log").read_text()[-2000:]
    md = (root/"summary.md").read_text()
    assert "| 运行 | 层 | N_c/N_q | 空支撑通道数 | N_S/N_q | σ_S²（层首） | σ_S²（轮用 / 轮后） |" in md
    assert f"| X2d-{SHORTS[0]} | 0 | 0.050，0.010 | 1 | 0.060 | 0.003 | 0.004 / 0.005 |" in md
    assert f"| X2d-{SHORTS[2]} | 3 | 0.050，0.010 | 0 | 0.060 | 0.003 | 0.004 / 0.005 |" in md
    assert sx.support_ratio(dict(N_S=60., N_q=1000.)) == pytest.approx(.06)
    assert sx.support_ratio(dict(N_S=None, N_q=1000.)) is None and sx.support_ratio(dict(N_q=1000.)) is None and sx.support_ratio(dict(N_S=60., N_q=0.)) is None
    assert md.count(sx.GF_NOTE) == 1 and md.index("## 逐对逐臂") < md.index(sx.GF_NOTE) < md.index("## X2a − A0 的配对差")
    brain_only = tmp_path/"x2_brain"
    assert driver(brain_only, X2Fake(), inputs, "--arms", "a,b,c") == 0
    assert sx.GF_NOTE not in (brain_only/"summary.md").read_text() and "σ 估计域的支撑记录" not in (brain_only/"summary.md").read_text()


class X2FakeNoSupport(X2Fake):
    """A run of `unit` ends normally, but its hyper.json has no support records (review GATEA_rev2 N7: the config.json check cannot see it)."""

    def __init__(self, *, unit, **kwargs):
        super().__init__(**kwargs)
        self.without_support = unit

    def run(self, cmd, record):
        code = super().run(cmd, record)
        unit, _, _, argv = self.split(cmd)
        if unit == self.without_support:
            argv = [str(a) for a in argv]
            path = Path(argv[argv.index("--out")+1])/"hyper.json"
            doc = json.loads(path.read_text())
            doc.pop("support_levels")
            path.write_text(json.dumps(doc))
        return code


def test_x2_counts_non_brain_runs_without_support_records(driver, inputs, tmp_path, x2):
    root = tmp_path/"x2"
    system = X2FakeNoSupport(unit=f"tag-gax2dry-d-{SHORTS[1]}")
    assert driver(root, system, inputs) == 0, (root/"queue.log").read_text()[-2000:]     # recorded, not a STOP: the args were right
    s = json.loads((root/"summary.json").read_text())
    assert s["counts"]["non_brain_without_support"] == 1 and s["non_brain_without_support"] == [f"d-{SHORTS[1]}"]
    md = (root/"summary.md").read_text()
    warning = md[:md.index("## 各臂")]                    # at the top of the summary
    assert "结束了却没有 support 记录" in warning and f"d-{SHORTS[1]}" in warning
    assert "非 brain 臂没有 support 记录的运行 1。" in md


def test_x2_support_count_leaves_out_brain_arms_and_killed_runs(driver, inputs, tmp_path):
    root = tmp_path/"x2"
    assert driver(root, X2Fake(drop=[f"tag-gax2dry-d-{SHORTS[0]}"]), inputs) == 0, (root/"queue.log").read_text()[-2000:]
    s = json.loads((root/"summary.json").read_text())
    assert s["counts"]["failed_no_status"] == 1 and s["counts"]["non_brain_without_support"] == 0     # a failure, counted as one already
    assert s["rows"]["a"][0]["support"] is None and s["rows"]["c"][0]["support"] is None              # brain arms have none: not counted


def test_x2_arm_d_failing_everywhere_does_not_stop_and_reads_describe_only(driver, inputs, tmp_path):
    root = tmp_path/"x2"
    system = X2Fake(drop=[f"tag-gax2dry-d-{s}" for s in SHORTS])
    assert driver(root, system, inputs) == 0, (root/"queue.log").read_text()[-2000:]
    s = json.loads((root/"summary.json").read_text())
    assert s["counts"]["failed_no_status"] == 3 and s["counts"]["ran"] == 9 and s["arm_means"]["d"]["n"] == 0
    assert all(line.startswith("只描述（可比 0/5") and "诊断点" in line for line in s["readings"]["d"].values())
    assert s["extra_comparisons"]["d_vs_c"]["comparison"]["seg4"]["n"] == 0 and s["extra_comparisons"]["d_vs_P4"]["comparison"]["seg4"]["n"] == 0
    assert sorted(p.name for p in (root/"failures").iterdir()) == sorted(f"run_d-{x}.json" for x in SHORTS)
    assert "σ 估计域的支撑记录" not in (root/"summary.md").read_text()


def test_x2_three_failures_in_a_row_across_arms_stop(driver, inputs, tmp_path):
    root = tmp_path/"x2"
    system = X2Fake(drop=[f"tag-gax2dry-{a}-{SHORTS[2]}" for a in "cda"])      # pair index 2 runs c, d, a, b
    assert driver(root, system, inputs) == 8
    assert "3 failed runs in a row" in (root/"queue.log").read_text()
    assert f"tag-gax2dry-b-{SHORTS[2]}" not in runs_of(system)


def test_x2_missing_p4_record_only_narrows_the_side_comparison(driver, inputs, tmp_path, x2, monkeypatch):
    monkeypatch.setattr(x2.sx2, "REQUIRED_N", 3)
    (inputs/"refs"/"pilot"/"eval"/"p4"/SHORTS[1]/f"pilot_p4_{SHORTS[1]}"/"metrics.json").unlink()
    root = tmp_path/"x2"
    assert driver(root, X2Fake(), inputs) == 0
    s = json.loads((root/"summary.json").read_text())
    assert s["extra_comparisons"]["d_vs_P4"]["comparison"]["seg4"]["n"] == 2 and s["p4"]["means"]["n"] == 2
    assert s["readings"]["d"]["seg4"].startswith(x2.sx2.READING_SEG4) and s["arm_means"]["d"]["n"] == 3


def test_x2_restart_with_three_arms_on_a_four_arm_root_stops(driver, inputs, tmp_path):
    root = tmp_path/"x2"
    assert driver(root, X2Fake(), inputs, "--until", "eta") == 0
    system = X2Fake()
    assert driver(root, system, inputs, "--arms", "a,b,c") == 8
    assert "differs from the recomputed record" in (root/"queue.log").read_text() and system.units == []


def test_x2_eta_records_of_a_root_cannot_change_domain(driver, inputs, tmp_path):
    root = tmp_path/"x2"
    assert driver(root, X2Fake(), inputs, "--until", "eta") == 0
    doc = json.loads((root/"eta"/"eta_d.json").read_text())
    doc["sigma_domain"] = "brain"
    (root/"eta"/"eta_d.json").write_text(json.dumps(doc))
    system = X2Fake()
    assert driver(root, system, inputs) == 8
    assert "eta_d.json differs from the recomputed record" in (root/"queue.log").read_text() and system.units == []


def test_x2_only_arm_d_selected(driver, inputs, tmp_path):
    root = tmp_path/"x2"
    system = X2Fake()
    assert driver(root, system, inputs, "--arms", "d") == 0
    assert runs_of(system) == [f"tag-gax2dry-d-{s}" for s in SHORTS]
    s = json.loads((root/"summary.json").read_text())
    assert s["extra_comparisons"].keys() == {"d_vs_P4"} and s["reproduction"]["state"].startswith("未判")


def test_x2_rotation_of_four_arms_over_five_pairs(x2, tmp_path):
    queue = x2.X2Queue(x2.parse_args(["--root", str(tmp_path/"x2")]), X2Fake())
    orders = [queue.order_of(i) for i in range(5)]
    assert orders == [list("abcd"), list("bcda"), list("cdab"), list("dabc"), list("abcd")]
    assert {a: sum(o[0] == a for o in orders) for a in "abcd"} == dict(a=2, b=1, c=1, d=1)     # the plan, revision 2
    before = (lambda x, y: sum(o.index(x) < o.index(y) for o in orders))
    assert before("c", "d") == 4 and before("a", "b") == 4 and before("b", "c") == 4


def test_x2_failed_certificate_stops(driver, inputs, tmp_path):
    root = tmp_path/"x2"

    class Failing(X2Fake):
        def run(self, cmd, record):
            code = super().run(cmd, record)
            unit, _, _, argv = self.split(cmd)
            if "-eval-" in unit:
                argv = [str(a) for a in argv]
                path = Path(argv[argv.index("--out-dir")+1])/argv[argv.index("--run-id")+1]/"metrics.json"
                doc = json.loads(path.read_text())
                doc["certificate"]["passed"] = False
                path.write_text(json.dumps(doc))
            return code
    assert driver(root, Failing(), inputs) == 8
    assert "hard constraint 1" in (root/"queue.log").read_text()


def test_x2_gate_and_failures(driver, inputs, tmp_path):
    root = tmp_path/"x2_busy"
    assert driver(root, X2Fake(busy=["tag-other.service"]), inputs) == 8
    assert "tag-other.service" in (root/"queue.log").read_text()
    root = tmp_path/"x2_driver"
    assert driver(root, X2Fake(drivers=[4242]), inputs) == 8
    assert "4242" in (root/"queue.log").read_text()
    root = tmp_path/"x2_drop"
    drop = [f"tag-gax2dry-{a}-{s}" for a in "abc" for s in SHORTS]
    assert driver(root, X2Fake(drop=drop), inputs, "--max-consecutive-failures", "2") == 8
    assert "2 failed runs in a row" in (root/"queue.log").read_text()


def test_x2_refusals(x2, inputs, capsys):
    with pytest.raises(SystemExit):
        x2.parse_args(["--arms", "a,z"])
    with pytest.raises(SystemExit):
        x2.parse_args(["--no-start-after", "05:00"])
    for root in (x2.RESULTS/"d0_v1"/"x", x2.RESULTS/"path_pilot_v1"/"x", x2.RESULTS/"m2v1_dev", x2.RESULTS, x2.RESULTS/"frozen_eb808be",
                 x2.RESULTS/"frozen_eb808be"/"x", x2.RESULTS/"fb_audit_v1", x2.RESULTS/"gap_decomp_v1"/"x"):
        queue = x2.X2Queue(x2.parse_args(["--root", str(root)]), X2Fake())
        assert "overlaps" in queue.root_checks(), root
    queue = x2.X2Queue(x2.parse_args(["--root", str(x2.RESULTS/"gate_a_x2_dry"), "--dry-run"]), X2Fake())
    assert "must lie outside" in queue.root_checks()
    assert x2.X2Queue(x2.parse_args(["--root", str(x2.REAL_ROOT)]), X2Fake()).root_checks() == ""


def test_x2_eta_values_are_the_plans(x2):
    assert x2.arm_eta(x2.ARMS["a"]) == dict(alpha=.01, nu=3.87, tau=[1., .5, .5, .5])
    assert x2.arm_eta(x2.ARMS["b"]) == dict(alpha=.01, nu=1.37, tau=[1., 1., 1., 1.]) == x2.arm_eta(x2.ARMS["c"])
    assert (x2.ARMS["a"]["channel"], x2.ARMS["b"]["channel"], x2.ARMS["c"]["channel"]) == ("tissue", "tissue", "structure")


@pytest.mark.slow
def test_x2_dry_run_in_process(x2, dry_mod, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    out = tmp_path/"dry"
    sha = head()
    dry_mod.main(["--out", str(out), "--wall-seconds", "60", "--extra", "", "--code-sha", sha])
    write_references(out)

    def prepare(self):
        self.code, self.code_sha = ROOT, sha
        self.allowlist = self.allowlist or str(out/"dev_subjects.json")
        self.check_code_tree()
    monkeypatch.setattr(x2.X2Arm, "prepare_code", prepare)
    monkeypatch.setattr(x2.X2Arm, "write_env", lambda self: None)
    root = tmp_path/"x2"
    deadline = (datetime.now(timezone.utc)+timedelta(hours=6)).isoformat(timespec="minutes")
    args = ["--root", str(root), *x2_common(out), "--no-start-after", deadline]
    system = InlineSystem(clock=datetime.now(timezone.utc))
    assert x2.main(args, system=system) == 0, (root/"queue.log").read_text()[-4000:]
    assert len(runs_of(system)) == 12 and len(system.units) == 24
    for config in (root/"runs").glob("*/*/config.json"):
        document = json.loads(config.read_text())
        arm = config.parent.parent.name
        assert document["args"]["channel_source"] == ("structure" if arm in "cd" else "tissue")
        assert document["eta"]["nu"] == (3.87 if arm == "a" else 1.37) and document["eta"]["alpha"] == .01
        assert (document["args"]["sigma_domain"], document["args"]["support_threshold"]) == (("support_shared", .05) if arm == "d" else ("brain", None))
        assert "init_field" not in document["args"]
    s = json.loads((root/"summary.json").read_text())
    assert s["counts"]["args_mismatch"] == 0 and s["counts"]["eta_mismatch"] == 0 and s["counts"]["ran"] == 12
    assert all(r["support"] for r in s["rows"]["d"] if r["state"] == "ran") and not any(r["support"] for r in s["rows"]["c"])
    for row in (r for a in s["arm_order"] for r in s["rows"][a]):
        if row["exported"]:
            assert row["certificate"]["passed"] is True, row["short"]
            assert row["metrics"]["cortex"] is not None and row["metrics"]["white_matter"] is not None
    md = (root/"summary.md").read_text()
    assert "各层读数" in md and "皮层" in md and "诊断点，不是合同选出的 η" in md and "σ 估计域的支撑记录" in md and "X2d − P4" in md
    again = InlineSystem(clock=datetime.now(timezone.utc))
    assert x2.main(args, system=again) == 0 and again.units == []
