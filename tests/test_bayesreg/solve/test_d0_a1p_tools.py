"""The arms A1Ph, A1Ps (runs/20261102-a1p/plan.md; work package WST): experiments/d0/make_inputs.py, evaluate.py summarize-a1p,
run_queue.sh --a1p, check_seg4_map.py, on synthetic files only (tmp_path; no data/, no real image, label or prediction).  The
table configs/bayesreg/seg4_of_seg35_v1.json and its check file are repository files (counts only).

- the table: every seg35 label in one seg4 label, exactly the check file's result, and different from the unverified tissue4_ref
  on labels 6, 17, 25, 33 only; a changed table or check file STOPs;
- with a one-hot posterior both colourings are A1 of the labels seg4_of_seg35(seg35): A1Ph bit for bit, A1Ps to 1e-12; no
  label reaches the colouring; ties and absent classes;
- the written inputs equal an independent recomputation from probs36.npz; labels and initial_u copied bitwise; the record says
  oracle false, the policy and the table; every A4p check still stops the arm (changed probs, training subject, flipped
  prediction, missing Atropos);
- --only-pair: partial record, the other pairs not written, the queue's start check refuses it; never under d0_v1/ or d0_v1_a4/,
  never mixed with A1-A3;
- the R-route rule (threshold, certificates, missing pairs) and the summary (seg35 and seg4, per structure, references not
  concurrent);
- run_queue.sh --a1p: the order, the settings, each start check, nothing written by the dry run; D0 and --a4 order unchanged;
- check_seg4_map.py: synthetic labels, the guard (non-development subject refused), a label that splits between two seg4 labels
  shows share < 1.
"""
import hashlib
import json
import subprocess

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from jd_curl.three_d.nifti import file_sha256

from ..io_seg._helpers import DEV, write_json
from ..io_seg._helpers import synthetic_folds  # noqa: E402   (C23.9-13 (8), revision 6e)
from .test_d0_a4_tools import (EV, FAKE_PAIRS, MI, PAIRS, QUEUE, ROOT, SHAPE, env, image, load, onehot,  # noqa: F401
                               queue_tree, rehash, seg35)

CHECK = load("d0_check_seg4_map", ROOT/"experiments"/"d0"/"check_seg4_map.py")
ARMS = ("A1Ph", "A1Ps")
TISSUE4_DIFFERENT = {6, 17, 25, 33}


def table():
    lookup, record = MI.load_seg4_map()
    return lookup, record


def test_table_is_what_the_check_file_found_and_not_tissue4_ref():
    lookup, record = table()
    assert lookup[0] == 0 and sorted(set(lookup[1:])) == [1, 2, 3, 4]
    assert [k for k in range(36) if lookup[k] == 1] == [2, 21]              # seg4 Cortex is the cerebral cortex only
    assert lookup[1] == lookup[20] == lookup[5] == lookup[24] == lookup[13] == lookup[17] == lookup[33] == 3
    assert lookup[6] == lookup[25] == 2 and lookup[18] == lookup[19] == 4
    assert record["check_subjects"] == 20 and record["seg4_names"]["1"] == "Cortex"
    reference, status, names = CHECK.reference_map(ROOT/"configs"/"bayesreg"/"structure_groups_v1.json")
    assert status == "reference_unverified" and names == ["cortex", "subcortical_gm", "white_matter", "csf"]
    assert {k for k in range(1, 36) if reference[k] != lookup[k]} == TISSUE4_DIFFERENT
    check = json.loads((ROOT/"runs"/"20261102-a1p"/"seg4_label_check.json").read_text())
    assert {d["seg35"] for d in check["disagreements_with_tissue4_ref"]} == TISSUE4_DIFFERENT
    assert check["voxels_seg4_positive"] == check["voxels_seg35_positive"]


@pytest.mark.parametrize("case", ["wrong_value", "missing_label", "maps_to_background", "check_sha", "check_disagrees"])
def test_table_refusals(tmp_path, case):
    doc = json.loads(MI.SEG4_MAP.read_text())
    check = json.loads((ROOT/doc["check_file"]).read_text())
    if case == "wrong_value":
        doc["seg4_of_seg35"]["6"] = 1
    elif case == "missing_label":
        del doc["seg4_of_seg35"]["35"]
    elif case == "maps_to_background":
        doc["seg4_of_seg35"]["7"] = 0
    elif case == "check_disagrees":
        check["per_seg35_label"]["7"]["share"] = .99
    path = tmp_path/"table.json"
    check_path = tmp_path/"check.json"
    check_path.write_text(json.dumps(check))
    doc["check_file"] = "check.json"
    doc["check_file_sha256"] = "0"*64 if case == "check_sha" else file_sha256(check_path)
    path.write_text(json.dumps(doc))
    with pytest.raises(SystemExit):
        MI.load_seg4_map(path, root=tmp_path)


def test_merge_sums_to_one_and_groups_by_the_table():
    lookup, _ = table()
    P = np.random.default_rng(1).uniform(0, 1, (36,)+SHAPE)
    P /= P.sum(axis=0)
    S = MI.merge_seg4(P, lookup)
    assert S.shape == (5,)+SHAPE and np.allclose(S.sum(axis=0), 1.)
    assert np.array_equal(S[1], P[2]+P[21]) and np.array_equal(S[0], P[0])
    with pytest.raises(SystemExit):
        MI.merge_seg4(S, lookup)


def test_one_hot_posterior_gives_a1_of_the_table_labels():
    lookup, _ = table()
    I = image(DEV[0])
    labels = seg35(DEV[0])
    truth = lookup[labels]
    a1, a1_consts = MI.colour("A1", I, truth, None, None)
    S = MI.merge_seg4(onehot(labels), lookup)
    hard, hard_consts = MI.colour("A1Ph", I, None, None, S)
    soft, soft_consts = MI.colour("A1Ps", I, None, None, S)
    assert hard.tobytes() == a1.tobytes()
    assert hard_consts["c_by_seg4_label"] == a1_consts["c_by_seg4_label"]
    assert np.max(np.abs(soft-a1)) <= 1e-12
    for k in range(1, 5):
        assert soft_consts["c_by_seg4_label"][k] == pytest.approx(a1_consts["c_by_seg4_label"][k], abs=1e-12)
    assert sum(hard_consts["predicted_voxels_by_seg4_label"].values()) == I.size


@pytest.mark.parametrize("arm", ARMS)
def test_colouring_takes_no_label_and_checks_the_shape(arm):
    I = image(DEV[0])
    S = np.ones((5,)+SHAPE)/5
    with pytest.raises(ValueError):
        MI.colour(arm, I, seg35(DEV[0]), None, S)
    with pytest.raises(ValueError):
        MI.colour(arm, I, None, seg35(DEV[0]), S)
    with pytest.raises(SystemExit):
        MI.colour(arm, I, None, None, np.ones((36,)+SHAPE)/36)


def test_absent_class_and_ties():
    I = image(DEV[0])
    S = np.zeros((5,)+SHAPE)
    S[0], S[1] = .5, .5                                 # a tie everywhere goes to the smaller label: background
    hard, consts = MI.colour("A1Ph", I, None, None, S)
    assert np.all(hard == 0) and all(consts["c_by_seg4_label"][k] is None for k in range(1, 5))
    soft, consts = MI.colour("A1Ps", I, None, None, S)
    assert consts["c_by_seg4_label"][2] is None and consts["posterior_mass_by_seg4_label"][2] == 0
    expected = gaussian_filter(float((S[1]*I).sum()/S[1].sum())*S[1], sigma=.75, mode="nearest", truncate=4.0)
    assert np.max(np.abs(soft-expected)) <= 1e-12


def build(env, arm, **changes):  # noqa: F811
    kwargs = dict(stage=env["stage"], allowlist=env["allowlist"], unet=env["unet"])
    kwargs.update(changes)
    seg_root = kwargs.pop("seg_root", env["seg_root"])
    MI.build_arm(arm, env["root"]/"inputs", seg_root, **kwargs)
    return env["root"]/"inputs"/arm


def recompute(arm, I, probs16):
    """Independent recomputation of the plan's formula from the stored float16 posterior."""
    lookup, _ = table()
    P = probs16.astype(np.float64)
    P /= P.sum(axis=0)
    S = np.stack([P[lookup == k].sum(axis=0) for k in range(5)])
    J = np.zeros_like(I)
    if arm == "A1Ph":
        L = np.argmax(S, axis=0)
        for k in range(1, 5):
            if np.any(L == k):
                J[L == k] = I[L == k].mean()
    else:
        for k in range(1, 5):
            J += (S[k]*I).sum()/S[k].sum()*S[k]
    return gaussian_filter(J, sigma=.75, mode="nearest", truncate=4.0)


@pytest.mark.parametrize("arm", ARMS)
@synthetic_folds("env")
def test_inputs_written_and_recomputed(env, arm):  # noqa: F811
    out = build(env, arm)
    record = json.loads((out/"d0_inputs.json").read_text())
    assert record["oracle"] is False and "partial" not in record and record["seg4_map"]["check_subjects"] == 20
    assert record["structure_registry"]["status"] == "registered" and record["alignment_qc"]["threshold"] == .8
    for fixed_subject, moving_subject in PAIRS:
        name = f"{fixed_subject}__{moving_subject}"
        d = out/MI.STAGE_NAME/name
        manifest = json.loads((d/"manifest.json").read_text())
        pre = manifest["preprocessing_d0"]
        assert pre["arm"] == arm and pre["oracle"] is False and pre["recipe"] == MI.RECIPES[arm]
        assert manifest["pair_sha256"] == file_sha256(d/"pair.npz")
        with np.load(env["stage"]/name/"pair.npz") as z, np.load(d/"pair.npz") as y:
            assert tuple(y.files) == MI.KEY_ORDER
            for key in ("fixed_seg4", "fixed_seg35", "moving_seg4", "moving_seg35", "initial_u"):
                assert y[key].tobytes() == z[key].tobytes(), key
            for side, subject in (("fixed", fixed_subject), ("moving", moving_subject)):
                with np.load(env["root"]/"pred"/subject/"probs36.npz") as p:
                    expected = recompute(arm, z[side], p["p"])
                assert np.max(np.abs(y[side]-expected)) <= 1e-12, side
                colour = pre["colour"][side]
                assert colour["policy"] == "all_folds_mean" and colour["alignment_qc"]["passed"] is True
                assert "c_by_seg4_label" in colour


@pytest.mark.parametrize("arm", ARMS)
@synthetic_folds("env")
def test_a4p_checks_still_stop_the_arm(env, arm):  # noqa: F811
    directory = env["root"]/"pred"/DEV[1]
    with np.load(directory/"probs36.npz") as z:
        probs = z["p"].copy()
    np.savez_compressed(directory/"probs36.npz", p=probs[:, ::-1])         # hash no longer the provenance's
    with pytest.raises(SystemExit, match="probs36|sha256"):
        build(env, arm)
    assert not (env["root"]/"inputs"/arm/"d0_inputs.json").exists()


@synthetic_folds("env")
def test_flipped_prediction_is_refused_by_the_alignment_qc(env):  # noqa: F811
    directory = env["root"]/"pred"/DEV[1]
    for name in ("probs36.npz", "structure_groups.npz"):
        with np.load(directory/name) as z:
            p = z["p"].copy()
        np.savez_compressed(directory/name, p=p[:, ::-1, ::-1, ::-1])
    rehash(directory)
    with pytest.raises(SystemExit, match="alignment QC"):
        build(env, "A1Ps")


def test_missing_atropos_and_training_subject_are_refused_before_anything_is_written(env):  # noqa: F811
    with pytest.raises(SystemExit, match="Atropos"):
        build(env, "A1Ph", seg_root=None)
    assert not (env["root"]/"inputs"/"A1Ph").exists()
    folds = json.loads(env["folds"].read_text())
    folds["subjects"][DEV[0]] = dict(fold="A", role="train")
    write_json(env["folds"], folds)
    env["unet"] = MI.UNetSource(pred_root=env["unet"].pred_root, folds=env["folds"],
                                folds_sha256=hashlib.sha256(env["folds"].read_bytes()).hexdigest(),
                                models_root=env["unet"].models_root, image_root=env["unet"].image_root)
    with pytest.raises(SystemExit, match="subjects refused"):
        build(env, "A1Ph")
    assert not (env["root"]/"inputs"/"A1Ph").exists()


@synthetic_folds("env")
def test_only_pair_is_partial(env):  # noqa: F811
    name = f"{PAIRS[0][0]}__{PAIRS[0][1]}"
    out = build(env, "A1Ph", only_pair=name)
    record = json.loads((out/"d0_inputs.json").read_text())
    assert record["partial"] is True and list(record["pairs"]) == [name]
    assert (out/MI.STAGE_NAME/name/"pair.npz").is_file() and not (out/MI.STAGE_NAME/f"{PAIRS[1][0]}__{PAIRS[1][1]}").exists()
    with pytest.raises(SystemExit, match="not in"):
        build(env, "A1Ps", only_pair="no_such_pair")


def test_arm_combinations_and_directories(env):  # noqa: F811
    with pytest.raises(SystemExit, match="without A1-A3"):
        MI.main(["--arms", "A1Ph", "A1"])
    with pytest.raises(SystemExit, match="on its own"):
        MI.main(["--arms", "A1Ph", "A4p"])
    for root in (MI.D0V1, MI.D0A4):
        with pytest.raises(SystemExit, match="never written under"):
            MI.main(["--arms", "A1Ph", "--inputs", str(root/"inputs_x")])


# ---- R-route and the summary ---------------------------------------------------------------------------------------------

PAIR_NAMES = [f"p{i:02d}" for i in range(19)]


def metrics(value, seg4=None):
    return dict(seg35=value, seg4=value+.06 if seg4 is None else seg4, dice_official=value, hd95=1.8, sdlogj_official=.08,
                sdlogj_brain=.6, num_foldings_official=0, pct_J_lt_half_brain=.1, cell_centre_J_nonpositive=0,
                seg4_per_label={str(k): value+.01*k for k in range(1, 5)},
                seg35_per_label={str(k): value+(.02 if k in (2, 21, 1, 20, 5, 24) else 0.) for k in range(1, 36)})


def fake(value, *, passed=True):
    return dict(metrics=metrics(value), certificate=dict(passed=passed))


def world(a1ph=.846, a1ps=.83):
    runs = {}
    for p in PAIR_NAMES:
        runs[("A0", p)], runs[("A1", p)], runs[("A4p", p)] = fake(.82), fake(.857), fake(.825)
        runs[("A1Ph", p)], runs[("A1Ps", p)] = fake(a1ph), fake(a1ps)
    return runs


def test_route_threshold_and_certificates():
    dec = EV.decide_a1p(world(), PAIR_NAMES)
    assert dec["routes"] == ["A1Ph"] and dec["arms"]["A1Ph"]["route"] and not dec["arms"]["A1Ps"]["route"]
    assert "另开一条简单路线" in dec["conclusion"]
    assert dec["recovery_descriptive"]["A1Ph"]["seg35"]["recovery_vs_A1"] == pytest.approx(.026/.037)
    assert dec["recovery_descriptive"]["A1Ph"]["seg4"]["recovery_mean_vs_A1"] == pytest.approx(.026/.037)
    assert EV.decide_a1p(world(.845, .845), PAIR_NAMES)["routes"] == ["A1Ph", "A1Ps"]
    below = EV.decide_a1p(world(.8449, .84), PAIR_NAMES)
    assert below["routes"] == [] and "不开这条路线" in below["conclusion"]
    runs = world()
    runs[("A1Ph", "p03")]["certificate"]["passed"] = False
    assert EV.decide_a1p(runs, PAIR_NAMES)["routes"] == []


def test_route_missing_pairs():
    runs = world()
    for p in PAIR_NAMES[:2]:
        del runs[("A1Ph", p)]
    dec = EV.decide_a1p(runs, PAIR_NAMES)
    assert dec["arms"]["A1Ph"]["n_with_field"] == 17 and dec["routes"] == ["A1Ph"]
    del runs[("A1Ph", PAIR_NAMES[2])]
    dec = EV.decide_a1p(runs, PAIR_NAMES)
    assert dec["arms"]["A1Ph"]["valid"] is False and dec["routes"] == [] and "不可判" in dec["conclusion"]


def write_run(root, arm, pair, value, *, load=(1.2, 2.5, 2.)):
    d = root/arm/pair
    d.mkdir(parents=True)
    write_json(d/"metrics.json", dict(run_id=f"{arm}_{pair}", run_dir=str(d), arm=arm, pair=pair, metrics=metrics(value),
                                      certificate=dict(passed=True, min_jd_lower_bound=1e-6), self_check=dict(passed=True),
                                      native_commits=30, e2e_seconds=88.5, success=False, evaluator=dict(sha256=f"sha-{arm}")))
    (d/"log.txt").write_text(f"2026-11-02T10:00:00Z launch tag-x; loadavg {load[0]} {load[1]} {load[2]}; x\n")


def test_summarize_a1p(tmp_path):
    stage = tmp_path/"stage"
    stage.mkdir()
    write_json(stage/"selection.json", dict(pairs=[dict(pair=p) for p in PAIR_NAMES]))
    d0, a4, a1p = tmp_path/"d0", tmp_path/"a4", tmp_path/"a1p"
    for p in PAIR_NAMES:
        for arm, value in (("A0", .82), ("A1", .857), ("A2", .79), ("A3", .85)):
            write_run(d0, arm, p, value)
        write_run(a4, "A4p", p, .825, load=(.4, .8, 1.))
        write_run(a1p, "A1Ph", p, .846, load=(.5, .9, 1.))
        write_run(a1p, "A1Ps", p, .83, load=(.5, .9, 1.))
    out = tmp_path/"analysis"
    assert EV.summarize_a1p(a1p, d0, a4, out, stage=stage) == 0
    dec = json.loads((out/"decision_a1p.json").read_text())
    assert dec["routes"] == ["A1Ph"] and dec["launch_loadavg_median"]["A1Ph"] == dict(one_min=.5, five_min=.9)
    assert dec["sources"]["evaluator_sha256"]["A1Ps"] == ["sha-A1Ps"]
    cortex = dec["per_structure"]["A1Ph"]["A0"]["seg4"]["1"]
    assert cortex["mean_diff"] == pytest.approx(.026) and cortex["positive"] == 19
    assert dec["per_structure"]["A1Ph"]["A4p"]["seg35"]["21"]["mean_diff"] == pytest.approx(.021)
    parts = dec["seg35_contributions_vs_A0"]["A1Ph"]
    assert parts["large_six"]+parts["other_29"] == pytest.approx(.026) and parts["n_other_above"] == 29
    text = (out/"summary_a1p.md").read_text()
    for fragment in ("seg35", "seg4", "皮层", "白质", "左大脑皮层", "A1Ph − A4′", "A1Ps − A1", "开发集", "单次观测", "不是同时段对照",
                     "预登记判据", "评测脚本版本"):
        assert fragment in text, fragment
    rows = (out/"long.csv").read_text().splitlines()
    assert any(",A1Ph," in r and ",seg4," in r for r in rows) and not any(",A2," in r for r in rows)
    for root in (EV.D0, EV.D0A4):
        with pytest.raises(SystemExit, match="never written under"):
            EV.summarize_a1p(a1p, d0, a4, root/"analysis_a1p", stage=stage)


# ---- run_queue.sh --a1p on the synthetic tree ------------------------------------------------------------------------------

@pytest.fixture
def a1p_tree(queue_tree):  # noqa: F811
    records = {}
    a4_doc = json.loads(queue_tree["record"].read_text())
    for arm in ARMS:
        path = queue_tree["results"]/"d0_v1_a1p"/"inputs"/arm/"d0_inputs.json"
        path.parent.mkdir(parents=True)
        write_json(path, dict(a4_doc, arm=arm))
        records[arm] = path
    return dict(queue_tree, records=records)


def run_a1p(tree, *args, **env):  # noqa: F811
    done = subprocess.run(["bash", str(QUEUE), "--a1p", *args], env=dict(tree["env"], **env), capture_output=True, text=True,
                          timeout=120)
    log = tree["results"]/"d0_v1_a1p"/"queue.log"
    return done.returncode, done.stdout, log.read_text() if log.exists() else ""


def test_run_queue_syntax_and_a1p_mode():
    assert subprocess.run(["bash", "-n", str(QUEUE)], capture_output=True).returncode == 0
    text = QUEUE.read_text()
    assert "D0=$RESULTS/d0_v1_a1p; ARMS=(A1Ph A1Ps); SELF_RE='^tag-d0a1p-'" in text and "EXTRA_BUSY=('tag-d0a4-*')" in text
    assert "summarize-a1p --a1p-runs-root" in text and text.count('exec 9> "$LINK_LOCK"') == 1


def test_run_queue_a1p_dry_run_lists_the_rotating_order(a1p_tree):
    code, out, log = run_a1p(a1p_tree, "--dry-run")
    assert code == 0 and log == ""
    expected = []
    for i, p in enumerate(FAKE_PAIRS):
        for j in range(2):
            expected.append(f"{len(expected)+1} {ARMS[(i+j) % 2]} {p}")
    assert out.splitlines() == expected and len(expected) == 38
    assert not (a1p_tree["results"]/"d0_v1_a1p"/"runs").exists()


A1P_START_STOPS = {
    "segmenter training": (dict(FAKE_UNITS="tag-unet35-train.service loaded active running x\\n"), "STOP: units still active: tag-unet35-train"),
    "a D0 unit": (dict(FAKE_UNITS="tag-d0-A0-0438-0439.service loaded active running x\\n"), "STOP: units still active: tag-d0-A0"),
    "an A4 unit": (dict(FAKE_UNITS="tag-d0a4-A4p-0438-0439.service loaded active running x\\n"), "STOP: units still active: tag-d0a4"),
    "GPU compute process": (dict(FAKE_PIDS="4242\\n"), "STOP: GPU not idle at start"),
    "registry modified": ("modify", "STOP: the repository registry"),
    "inputs from another registry": ("record_sha", "STOP: the A1Ph inputs were not made with the current"),
    "second arm without inputs": ("no_record", "STOP: the A1Ps inputs were not made with the current"),
    "dry-run inputs": ("partial", "STOP: the A1Ph inputs were not made with the current"),
}


@pytest.mark.parametrize("case", sorted(A1P_START_STOPS))
def test_run_queue_a1p_start_checks_stop(a1p_tree, case):
    change, fragment = A1P_START_STOPS[case]
    env = change if isinstance(change, dict) else {}
    if change == "modify":
        a1p_tree["registry"].write_text(a1p_tree["registry"].read_text()+"\n")
    elif change == "record_sha" or change == "partial":
        path = a1p_tree["records"]["A1Ph"]
        doc = json.loads(path.read_text())
        if change == "partial":
            doc["partial"] = True
        else:
            doc["structure_registry"]["sha256"] = "0"*64
        write_json(path, doc)
    elif change == "no_record":
        a1p_tree["records"]["A1Ps"].unlink()
    code, _, log = run_a1p(a1p_tree, **env)
    assert code == 1 and fragment in log, log
    assert "worktree HEAD" not in log and not list((a1p_tree["results"]/"d0_v1_a1p"/"runs").glob("*/*"))


def test_run_queue_a1p_start_checks_pass_on_a_quiet_machine(a1p_tree):
    code, _, log = run_a1p(a1p_tree)
    assert code == 1 and "STOP: worktree HEAD is not eb808be" in log, log
    assert "A1Ph registry check:" in log and "A1Ps registry check:" in log and "; ok" in log
    assert not (a1p_tree["results"]/"d0_v1").exists() and not (a1p_tree["results"]/"d0_v1_a4"/"runs").exists()


# ---- check_seg4_map.py -----------------------------------------------------------------------------------------------------

def label_stage(root, seg4_of, *, subjects=(DEV[0], DEV[1])):
    stage = root/"stage"
    d = stage/f"{subjects[0]}__{subjects[1]}"
    d.mkdir(parents=True)
    arrays = {}
    for side, subject in zip(("fixed", "moving"), subjects):
        labels = seg35(subject)
        arrays[f"{side}_seg35"] = labels.astype(float)
        arrays[f"{side}_seg4"] = seg4_of(labels).astype(float)
    np.savez_compressed(d/"pair.npz", **arrays)
    write_json(d/"manifest.json", dict(fixed_subject=subjects[0], moving_subject=subjects[1]))
    write_json(stage/"selection.json", dict(pairs=[dict(pair=d.name)]))
    return stage


def seg4_file(root):
    path = root/"seg4_labels.txt"
    path.write_text("#ID Label R G B A\n\n0 Unknown 0 0 0 0\n1 Cortex 1 1 1 0\n2 Subcortical-Gray-Matter 1 1 1 0\n3 White-Matter 1 1 1 0\n"
                    "4 CSF 1 1 1 0\n")
    return path


@pytest.fixture
def check_env(env):  # noqa: F811
    return dict(unet=env["unet"], allowlist=env["allowlist"], root=env["root"])


def test_check_tool_finds_the_table_and_a_split_label(check_env):
    root, lookup = check_env["root"], table()[0]
    stage = label_stage(root, lambda labels: lookup[labels])
    out = root/"check.json"
    groups = ROOT/"configs"/"bayesreg"/"structure_groups_v1.json"
    record = CHECK.run(stage, out, groups, allowlist=check_env["allowlist"], unet=check_env["unet"], seg4_table=seg4_file(root))
    assert {d["seg35"] for d in record["disagreements_with_tissue4_ref"]} == TISSUE4_DIFFERENT
    assert all(r["share"] == 1. for r in record["per_seg35_label"].values() if r["voxels"])
    assert record["voxels_seg4_positive_seg35_0"] == 0
    with pytest.raises(SystemExit, match="exists"):
        CHECK.run(stage, out, groups, allowlist=check_env["allowlist"], unet=check_env["unet"], seg4_table=seg4_file(root))

    def split(labels):
        values = lookup[labels].copy()
        values[(labels == 7) & (np.indices(labels.shape)[0] % 2 == 0)] = 1
        return values
    (root/"stage").rename(root/"stage_a")
    stage = label_stage(root, split)
    other = CHECK.run(stage, root/"check2.json", groups, allowlist=check_env["allowlist"], unet=check_env["unet"],
                      seg4_table=seg4_file(root))
    assert other["per_seg35_label"]["7"]["share"] < 1.


def test_check_tool_guard(check_env):
    root, lookup = check_env["root"], table()[0]
    from ..io_seg._helpers import SEALED
    stage = label_stage(root, lambda labels: lookup[labels], subjects=(DEV[0], SEALED))
    groups = ROOT/"configs"/"bayesreg"/"structure_groups_v1.json"
    with pytest.raises(SystemExit, match="subjects refused"):
        CHECK.run(stage, root/"out.json", groups, allowlist=check_env["allowlist"], unet=check_env["unet"], seg4_table=seg4_file(root))
    assert not (root/"out.json").exists()
