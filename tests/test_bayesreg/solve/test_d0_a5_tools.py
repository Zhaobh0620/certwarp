"""The arms A5, A5P (runs/20261103-a5/plan.md, docs/bayesreg/05_D0协议.md §14; work package A5): experiments/d0/make_inputs.py,
evaluate.py summarize-a5, run_queue.sh --a5, on synthetic files only (tmp_path; no data/, no real image, label or prediction).  The colour
table is the repository file; the table below is typed from the brief, not read from it.

- the table: content, pinned sha256, left-right symmetry, a changed file STOPs;
- the formula against an independent recomputation; the same labels give A5 and A5P bit for bit (one-hot posterior); ties go to the
  smaller label; the colouring takes an image and a label map, no pair object, and A5P no label: changed label members leave its images
  bit for bit the same; a colouring with a label outside 0..35 or without white matter STOPs;
- the written inputs: recomputed, labels and initial_u copied bitwise, oracle flag, the table in the records; every A4p check stops A5P;
  an arm directory is never overwritten; never under d0_v1/, d0_v1_a4/, d0_v1_a1p/, never mixed with other arms;
- the first gate reads A5 and A1 only (strictly greater; missing pairs, a failed certificate, a pin that does not reproduce: no expansion);
  the second gate per arm and the combined reading; summarize-a5 with 5 pairs and with 19, references from the long.csv of their batch,
  only the same pairs, the earlier summary kept;
- run_queue.sh --a5: the pair-major rotating order, --first-5, each start check, the first gate before the pairs after the first 5, and
  --first-5 followed by the whole queue reruns nothing and overwrites nothing.
"""
import csv
import hashlib
import inspect
import json
import subprocess

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from jd_curl.three_d.nifti import file_sha256

from ..io_seg._helpers import DEV, write_json
from ..io_seg._helpers import synthetic_folds  # noqa: E402   (C23.9-13 (8), revision 6e)
from .test_d0_a4_tools import (EV, FAKE_PAIRS, MI, PAIRS, QUEUE, ROOT, SHAPE, env, image, onehot,  # noqa: F401
                               queue_tree, rehash, seg35)

ARMS = ("A5", "A5P")
TABLE_SHA256 = "ee1d8160bf0d6966df8dc8bfe9bf367e7c6069b34a47a3edf18c58839ed85289"
BRIEF_TABLE = {(1, 20): 1., (2, 21): .62, (3, 22): .28, (4, 23): .12, (5, 24): 1., (6, 25): .6, (7, 26): .55, (8, 27): .62, (9, 28): .48,
               (10, 29): .75, (11,): .28, (12,): .28, (13,): .85, (14, 30): .46, (15, 31): .78, (16, 32): .8, (17, 33): .7, (18, 34): .12,
               (19, 35): .82}
F = {label: f for labels, f in BRIEF_TABLE.items() for label in labels}


def labels_with_white_matter(subject=DEV[0]):
    """The synthetic seg35 with labels 1 and 20 on some voxels whatever the random draw."""
    labels = seg35(subject).copy()
    labels[0, 0, 0], labels[1, 1, 1] = 1, 20
    return labels


def recompute(I, L):
    """The brief's formula, written out independently of make_inputs."""
    w = I[(L == 1) | (L == 20)].mean()
    J = np.zeros(I.shape)
    for label, f in F.items():
        J[L == label] = f*w
    return gaussian_filter(J, sigma=.75, mode="nearest", truncate=4.0), w


# ---- the table -------------------------------------------------------------------------------------------------------------

def test_table_content_hash_and_symmetry():
    assert file_sha256(MI.COLOUR_TABLE_A5) == TABLE_SHA256 == MI.COLOUR_TABLE_A5_SHA256
    table, record = MI.load_colour_table()
    assert table == F and sorted(table) == list(range(1, 36))
    assert record["table_sha256"] == TABLE_SHA256 and record["f_by_seg35_label"]["10"] == .75
    names = json.loads((ROOT/"configs"/"bayesreg"/"structure_groups_v1.json").read_text())["label_table"]["names"]
    for labels in BRIEF_TABLE:
        if len(labels) == 2:
            assert names[str(labels[0])].replace("Left-", "") == names[str(labels[1])].replace("Right-", ""), labels
        else:
            assert not names[str(labels[0])].startswith(("Left-", "Right-")), labels
    doc = json.loads(MI.COLOUR_TABLE_A5.read_text())
    for fragment in ("colour_survey.py", "no truth label read"):
        assert fragment in doc["provenance"]["basis"]
    assert "107" in doc["provenance"]["lead_count"] and "28" in doc["provenance"]["lead_count"]


@pytest.mark.parametrize("case", ["left_right", "two_rows", "missing", "zero", "bad_midline"])
def test_table_refusals(tmp_path, case):
    doc = json.loads(MI.COLOUR_TABLE_A5.read_text())
    rows = doc["rows"]
    if case == "left_right":
        rows[0]["labels"] = [1, 21]
    elif case == "two_rows":
        rows[1]["labels"] = [1, 20]
    elif case == "missing":
        del rows[-1]
    elif case == "zero":
        rows[3]["f"] = 0.
    else:
        rows[10]["labels"] = [14]
    path = tmp_path/"table.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(SystemExit):
        MI.load_colour_table(path, file_sha256(path))
    with pytest.raises(SystemExit, match="pinned"):
        MI.load_colour_table(path)                                  # the pinned sha256 catches any change first


# ---- the colouring ---------------------------------------------------------------------------------------------------------

def test_formula_against_an_independent_recomputation():
    I, L = image(DEV[0]), labels_with_white_matter()
    J, consts = MI.colour_from_table(I, L, F)
    want, w = recompute(I, L)
    assert J.tobytes() == want.tobytes() and J.dtype == np.float64
    assert consts["wm_mean"] == w and consts["wm_labels"] == [1, 20]
    assert consts["c_by_seg35_label"][10] == .75*w and consts["c_by_seg35_label"][1] == w
    assert sum(consts["voxels_by_seg35_label"].values()) == I.size and consts["voxels_by_seg35_label"][0] == np.count_nonzero(L == 0)


def test_smoothing_is_a1s():
    J = np.random.default_rng(0).uniform(0, 1, SHAPE)
    assert MI.smooth(J).tobytes() == gaussian_filter(J, sigma=.75, mode="nearest", truncate=4.0).tobytes()


@pytest.mark.parametrize("storage", [np.float64, np.int64, np.int16])
def test_one_hot_posterior_gives_a5_bit_for_bit(storage):
    I, L = image(DEV[0]), labels_with_white_matter()
    truth, c_truth = MI.colour("A5", I, seg35(DEV[0]), L.astype(storage), None, table=F)
    pred, c_pred = MI.colour("A5P", I, None, None, onehot(L), table=F)
    assert truth.tobytes() == pred.tobytes()
    for key in ("wm_mean", "c_by_seg35_label", "voxels_by_seg35_label"):
        assert c_truth[key] == c_pred[key], key
    assert "oracle" in c_truth["label_source"] and "argmax" in c_pred["label_source"]


def test_argmax_ties_go_to_the_smaller_label():
    I = image(DEV[0])
    P = np.zeros((36,)+SHAPE)
    P[2], P[9] = .5, .5                                             # a tie everywhere: label 2
    P[:, 0, 0, 0] = 0
    P[20, 0, 0, 0], P[1, 0, 0, 0] = .5, .5                          # white matter ties: label 1
    P[:, 1, 1, 1] = 0
    P[0, 1, 1, 1], P[1, 1, 1, 1] = .5, .5                           # a tie with the background: background
    _, consts = MI.colour("A5P", I, None, None, P, table=F)
    voxels = consts["voxels_by_seg35_label"]
    assert voxels[1] == 1 and voxels[20] == 0 and voxels[9] == 0 and voxels[0] == 1 and voxels[2] == I.size-2
    assert consts["wm_mean"] == I[0, 0, 0]


def test_colouring_takes_a_label_map_and_a5p_no_label():
    assert list(inspect.signature(MI.colour_from_table).parameters) == ["I", "L", "table"]
    I, L = image(DEV[0]), labels_with_white_matter()
    for kwargs in (dict(seg4=L), dict(seg35=L)):
        with pytest.raises(ValueError, match="no label"):
            MI.colour("A5P", I, kwargs.get("seg4"), kwargs.get("seg35"), onehot(L), table=F)
    with pytest.raises(SystemExit, match="posterior shape"):
        MI.colour("A5P", I, None, None, np.ones((5,)+SHAPE)/5, table=F)


@pytest.mark.parametrize("change,fragment", [("outside", "outside 0..35"), ("fraction", "outside 0..35"), ("no_white", "no cerebral white matter"),
                                             ("shape", "shape")])
def test_bad_label_maps_stop(change, fragment):
    I, L = image(DEV[0]), labels_with_white_matter().astype(np.float64)
    if change == "outside":
        L[2, 2, 2] = 36
    elif change == "fraction":
        L[2, 2, 2] = 3.5
    elif change == "no_white":
        L[np.isin(L, (1, 20))] = 2
    else:
        L = L[:-1]
    with pytest.raises(SystemExit, match=fragment):
        MI.colour_from_table(I, L, F)


# ---- make_inputs -----------------------------------------------------------------------------------------------------------

def build(env, arm, inputs="inputs", **changes):  # noqa: F811
    kwargs = dict(stage=env["stage"], allowlist=env["allowlist"], unet=env["unet"] if arm == "A5P" else None)
    kwargs.update(changes)
    seg_root = kwargs.pop("seg_root", env["seg_root"])
    MI.build_arm(arm, env["root"]/inputs, seg_root, **kwargs)
    return env["root"]/inputs/arm


@pytest.mark.parametrize("arm", ARMS)
@synthetic_folds("env")
def test_inputs_written_and_recomputed(env, arm):  # noqa: F811
    out = build(env, arm)
    record = json.loads((out/"d0_inputs.json").read_text())
    assert record["oracle"] is (arm == "A5") and "partial" not in record and record["seed"] is None and record["z_offsets"] is None
    assert record["colour_table"]["table_sha256"] == TABLE_SHA256 and record["sigma_voxel"] == .75
    if arm == "A5P":
        assert record["structure_registry"]["status"] == "registered" and record["alignment_qc"]["threshold"] == .8
    else:
        assert "structure_registry" not in record and "alignment_qc" not in record
    for fixed_subject, moving_subject in PAIRS:
        name = f"{fixed_subject}__{moving_subject}"
        d = out/MI.STAGE_NAME/name
        manifest = json.loads((d/"manifest.json").read_text())
        pre = manifest["preprocessing_d0"]
        assert pre["arm"] == arm and pre["oracle"] is (arm == "A5") and pre["recipe"] == MI.RECIPES[arm] and pre["seed"] is None
        assert pre["colour_table"]["table_sha256"] == TABLE_SHA256 and manifest["pair_sha256"] == file_sha256(d/"pair.npz")
        with np.load(env["stage"]/name/"pair.npz") as z, np.load(d/"pair.npz") as y:
            assert tuple(y.files) == MI.KEY_ORDER
            for key in ("fixed_seg4", "fixed_seg35", "moving_seg4", "moving_seg35", "initial_u"):
                assert y[key].tobytes() == z[key].tobytes(), key
            for side, subject in (("fixed", fixed_subject), ("moving", moving_subject)):
                if arm == "A5":
                    L = z[f"{side}_seg35"]
                else:
                    with np.load(env["root"]/"pred"/subject/"probs36.npz") as p:
                        L = np.argmax(p["p"].astype(np.float64), axis=0)
                want, w = recompute(z[side], L)
                assert y[side].tobytes() == want.tobytes(), side
                colour = pre["colour"][side]
                assert colour["wm_mean"] == w and colour["c_by_seg35_label"]["2"] == .62*w
                if arm == "A5P":
                    assert colour["policy"] == "all_folds_mean" and colour["alignment_qc"]["passed"] is True


def relabel_stage(env):  # noqa: F811
    """Change every label member of the stage (the images stay): the manifests' hashes follow."""
    for fixed_subject, moving_subject in PAIRS:
        d = env["stage"]/f"{fixed_subject}__{moving_subject}"
        with np.load(d/"pair.npz") as z:
            arrays = {k: np.array(z[k]) for k in z.files}
        for key in ("fixed_seg4", "fixed_seg35", "moving_seg4", "moving_seg35"):
            arrays[key] = arrays[key][::-1, ::-1, ::-1]
        np.savez_compressed(d/"pair.npz", **{k: arrays[k] for k in MI.KEY_ORDER})
        manifest = json.loads((d/"manifest.json").read_text())
        manifest["pair_sha256"] = file_sha256(d/"pair.npz")
        write_json(d/"manifest.json", manifest)


@synthetic_folds("env")
def test_a5p_images_do_not_depend_on_the_labels(env, monkeypatch):  # noqa: F811
    seen = []
    real = MI.colour

    def spy(arm, I, seg4, seg35, P, table=None):
        seen.append((arm, seg4, seg35))
        return real(arm, I, seg4, seg35, P, table=table)
    monkeypatch.setattr(MI, "colour", spy)
    before = build(env, "A5P", inputs="before")
    oracle_before = build(env, "A5", inputs="before")
    assert seen and all(s[1] is None and s[2] is None for s in seen if s[0] == "A5P")
    relabel_stage(env)
    after = build(env, "A5P", inputs="after")
    oracle_after = build(env, "A5", inputs="after")
    name = f"{PAIRS[0][0]}__{PAIRS[0][1]}"
    with np.load(before/MI.STAGE_NAME/name/"pair.npz") as x, np.load(after/MI.STAGE_NAME/name/"pair.npz") as y, \
            np.load(oracle_before/MI.STAGE_NAME/name/"pair.npz") as o, np.load(oracle_after/MI.STAGE_NAME/name/"pair.npz") as q:
        assert x["fixed"].tobytes() == y["fixed"].tobytes() and x["moving"].tobytes() == y["moving"].tobytes()
        assert x["fixed_seg35"].tobytes() != y["fixed_seg35"].tobytes()         # the copy for the evaluation follows the stage
        assert o["fixed"].tobytes() != q["fixed"].tobytes()                     # control: the oracle arm does read them


@synthetic_folds("env")
def test_a4p_checks_stop_a5p(env):  # noqa: F811
    directory = env["root"]/"pred"/DEV[1]
    with np.load(directory/"probs36.npz") as z:
        probs = z["p"].copy()
    np.savez_compressed(directory/"probs36.npz", p=probs[:, ::-1])               # hash no longer the provenance's
    with pytest.raises(SystemExit, match="probs36|sha256"):
        build(env, "A5P")
    assert not (env["root"]/"inputs"/"A5P"/"d0_inputs.json").exists()


@synthetic_folds("env")
def test_flipped_prediction_is_refused_by_the_alignment_qc(env):  # noqa: F811
    directory = env["root"]/"pred"/DEV[1]
    for name in ("probs36.npz", "structure_groups.npz"):
        with np.load(directory/name) as z:
            p = z["p"].copy()
        np.savez_compressed(directory/name, p=p[:, ::-1, ::-1, ::-1])
    rehash(directory)
    with pytest.raises(SystemExit, match="alignment QC"):
        build(env, "A5P")
    assert not (env["root"]/"inputs"/"A5P"/"d0_inputs.json").exists()


def test_missing_atropos_and_training_subject_are_refused_before_anything_is_written(env):  # noqa: F811
    with pytest.raises(SystemExit, match="Atropos"):
        build(env, "A5P", seg_root=None)
    assert not (env["root"]/"inputs"/"A5P").exists()
    folds = json.loads(env["folds"].read_text())
    folds["subjects"][DEV[0]] = dict(fold="A", role="train")
    write_json(env["folds"], folds)
    env["unet"] = MI.UNetSource(pred_root=env["unet"].pred_root, folds=env["folds"],
                                folds_sha256=hashlib.sha256(env["folds"].read_bytes()).hexdigest(),
                                models_root=env["unet"].models_root, image_root=env["unet"].image_root)
    with pytest.raises(SystemExit, match="subjects refused"):
        build(env, "A5P")
    assert not (env["root"]/"inputs"/"A5P").exists()


@pytest.mark.parametrize("arm", ARMS)
def test_table_changed_after_the_pin_stops_before_anything_is_written(env, arm, tmp_path):  # noqa: F811
    other = tmp_path/"table.json"
    other.write_text(MI.COLOUR_TABLE_A5.read_text()+"\n")
    with pytest.raises(SystemExit, match="pinned"):
        build(env, arm, colour_table=other)
    assert not (env["root"]/"inputs"/arm).exists()


@pytest.mark.parametrize("arm", ARMS)
@synthetic_folds("env")
def test_overwrite_protection_and_partial_record(env, arm):  # noqa: F811
    name = f"{PAIRS[0][0]}__{PAIRS[0][1]}"
    out = build(env, arm, only_pair=name)
    record = json.loads((out/"d0_inputs.json").read_text())
    assert record["partial"] is True and list(record["pairs"]) == [name]
    assert not (out/MI.STAGE_NAME/f"{PAIRS[1][0]}__{PAIRS[1][1]}").exists()
    before = file_sha256(out/MI.STAGE_NAME/name/"pair.npz")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        build(env, arm)
    assert file_sha256(out/MI.STAGE_NAME/name/"pair.npz") == before


def test_arm_combinations_and_directories():
    for arms in (["A5", "A1"], ["A5P", "A1Ph"], ["A5", "A5P", "A3"]):
        with pytest.raises(SystemExit, match="without any other arm"):
            MI.main(["--arms", *arms])
    for root in (MI.D0V1, MI.D0A4, MI.D0A1P):
        with pytest.raises(SystemExit, match="never written under"):
            MI.main(["--arms", "A5", "--inputs", str(root/"inputs_x")])
    assert MI.D0A5.name == "d0_v1_a5"


# ---- the gates -------------------------------------------------------------------------------------------------------------

NAMES = [f"OASIS_OAS1_{900+i:04d}_MR1__OASIS_OAS1_{901+i:04d}_MR1" for i in range(19)]


def metric_values(value, sdlogj=.08):
    return dict(seg35=value, seg4=value+.06, dice_official=value, hd95=1.8, sdlogj_official=sdlogj, sdlogj_brain=.6, num_foldings_official=0,
                pct_J_lt_half_brain=.1, cell_centre_J_nonpositive=0, seg4_per_label={str(k): value+.01*k for k in range(1, 5)},
                seg35_per_label={str(k): value+(.03 if k in (4, 23, 18, 34, 19, 35) else .02 if k in (2, 21, 1, 20, 5, 24) else 0.) for k in range(1, 36)})


def fake(value, *, passed=True, sdlogj=.08):
    return dict(metrics=metric_values(value, sdlogj), certificate=dict(passed=passed, min_jd_lower_bound=1e-6), self_check=dict(passed=True),
                classification={"class": "ok", "kind": None, "reason": None}, native_commits=30, e2e_seconds=88.)


def world(a5=.89, a5p=.85, a1=.8509, pairs=NAMES):
    runs = {("A1", p): fake(a1) for p in NAMES}
    runs.update({("A5", p): fake(a5) for p in pairs})
    runs.update({("A5P", p): fake(a5p) for p in pairs})
    return runs


def test_first_gate_expands_only_above_a1():
    gate = EV.decide_a5_gate1(world(a5=.8510), NAMES)
    assert gate["expand"] is True and gate["status"] == "expand" and gate["n"] == 5 and "两臂都扩到 19 对" in gate["text"]
    assert gate["pairs"] == NAMES[:5] and gate["A1_seg4"] == pytest.approx(.9109) and gate["A5_seg4"] == pytest.approx(.911)
    equal = EV.decide_a5_gate1(world(a5=.8509), NAMES)                      # strictly greater
    assert equal["expand"] is False and equal["status"] == "stop" and "换颜色表没有超过 A1" in equal["text"] and "交 BZ" in equal["text"]
    assert EV.decide_a5_gate1(world(a5=.84), NAMES)["expand"] is False


def test_first_gate_uses_only_the_first_five_pairs_and_never_a5p():
    runs = world(a5=.84)
    for p in NAMES[5:]:
        runs[("A5", p)] = fake(.99)                                         # later pairs do not enter
    assert EV.decide_a5_gate1(runs, NAMES)["expand"] is False
    base = EV.decide_a5_gate1(world(a5=.86), NAMES)
    for a5p in (.5, .99):
        assert EV.decide_a5_gate1(world(a5=.86, a5p=a5p), NAMES) == base
    no_a5p = {k: m for k, m in world(a5=.86).items() if k[0] != "A5P"}
    assert EV.decide_a5_gate1(no_a5p, NAMES) == base


@pytest.mark.parametrize("case", ["missing_run", "no_field", "certificate", "no_a1", "pin", "reference"])
def test_first_gate_not_decidable(case):
    runs = world(a5=.87)
    problems = ()
    if case == "missing_run":
        del runs[("A5", NAMES[2])]
    elif case == "no_field":
        runs[("A5", NAMES[2])] = dict(metrics=None)
    elif case == "certificate":
        runs[("A5", NAMES[2])] = fake(.87, passed=False)
    elif case == "no_a1":
        del runs[("A1", NAMES[0])]
    elif case == "pin":
        runs = world(a5=.87, a1=.852)
    else:
        problems = ("x",)
    gate = EV.decide_a5_gate1(runs, NAMES, problems)
    assert gate["expand"] is False and gate["status"] == "not_decidable" and "不可判" in gate["text"]


@pytest.mark.parametrize("a5,a5p,reading,route,reach", [
    (.885, .845, "scalar_continues", True, True),
    (.90, .86, "scalar_continues", True, True),
    (.885, .8449, "segmenter_bottleneck", False, True),
    (.8849, .85, "scalar_exhausted", True, False),
    (.87, .83, "scalar_exhausted", False, False),
])
def test_second_gate_thresholds_and_reading(a5, a5p, reading, route, reach):
    gate = EV.decide_a5_gate2(world(a5, a5p), NAMES)
    assert gate["evaluated"] is True and gate["reading"] == reading and gate["reading_text"] == EV.A5_READINGS[reading]
    assert gate["A5P_opens_route"] is route and gate["A5_could_reach_86"] is reach
    assert gate["arms"]["A5"]["threshold"] == .885 and gate["arms"]["A5P"]["threshold"] == .845


def test_second_gate_preconditions_per_arm():
    runs = world()
    runs[("A5P", NAMES[3])]["certificate"]["passed"] = False
    gate = EV.decide_a5_gate2(runs, NAMES)
    assert gate["reading"] == "preconditions" and gate["A5P_opens_route"] is False and gate["A5_could_reach_86"] is True
    assert gate["arms"]["A5P"]["certificates_all_passed"] is False and gate["arms"]["A5"]["certificates_all_passed"] is True
    runs = world()
    for p in NAMES:
        runs[("A5", p)] = fake(.89, sdlogj=.1001)
    gate = EV.decide_a5_gate2(runs, NAMES)
    assert gate["reading"] == "preconditions" and gate["arms"]["A5"]["sdlogj_ok"] is False and gate["A5_could_reach_86"] is False
    assert gate["A5P_opens_route"] is True                                  # each arm on its own
    runs = world(a5=.87)                                                    # below its threshold anyway: the reading stands
    for p in NAMES:
        runs[("A5", p)] = fake(.87, sdlogj=.2)
    assert EV.decide_a5_gate2(runs, NAMES)["reading"] == "scalar_exhausted"
    runs = world()
    for p in NAMES:
        runs[("A5", p)] = fake(.89, sdlogj=.0999)
    assert EV.decide_a5_gate2(runs, NAMES)["arms"]["A5"]["sdlogj_ok"] is True


def test_second_gate_missing_pairs():
    runs = world()
    for p in NAMES[:2]:
        runs[("A5P", p)] = dict(metrics=None)                               # method failures: 17 final maps still decide
    gate = EV.decide_a5_gate2(runs, NAMES)
    assert gate["arms"]["A5P"]["n_with_field"] == 17 and gate["reading"] == "scalar_continues"
    runs[("A5P", NAMES[2])] = dict(metrics=None)
    gate = EV.decide_a5_gate2(runs, NAMES)
    assert gate["arms"]["A5P"]["valid"] is False and gate["reading"] == "not_decidable" and gate["A5P_opens_route"] is False
    runs = world()
    for p in NAMES[:3]:
        runs[("A5", p)] = dict(metrics=None)
    assert EV.decide_a5_gate2(runs, NAMES)["reading"] == "not_decidable"
    del runs[("A5", NAMES[0])]                                              # a run that has not finished
    assert EV.decide_a5_gate2(runs, NAMES)["reading"] == "not_complete"
    five = EV.decide_a5_gate2(world(pairs=NAMES[:5]), NAMES)
    assert five["evaluated"] is False and five["reading"] == "not_complete" and five["A5P_opens_route"] is False and "第二关未判" in five["text"]


# ---- summarize-a5 ---------------------------------------------------------------------------------------------------------

def write_run(root, arm, pair, value, *, run_id=None, cls="ok", final=True, sdlogj=.08):
    d = root/arm/pair
    d.mkdir(parents=True)
    doc = dict(run_id=run_id or f"{arm}_{pair}", run_dir=str(d), arm=arm, pair=pair, certificate=dict(passed=final, min_jd_lower_bound=1e-6 if final else None),
               metrics=metric_values(value, sdlogj) if final else None, self_check=dict(passed=True), status="native_budget_exhausted", native_commits=30,
               e2e_seconds=88.5, success=False, evaluator=dict(sha256=f"sha-{arm}"), classification={"class": cls, "kind": None, "reason": None})
    write_json(d/"metrics.json", doc)
    (d/"log.txt").write_text("2026-11-03T10:00:00Z launch tag-x; loadavg 0.5 0.9 1.0; x\n")


def write_batch(root, values):
    """A reference batch: runs/<arm>/<pair>/metrics.json and its analysis/long.csv (the same writer as the real batches)."""
    for arm, value in values.items():
        for p in NAMES:
            write_run(root/"runs", arm, p, value)
    (root/"analysis").mkdir()
    with open(root/"analysis"/"long.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set", "split", "method", "arm", "pair", "measure", "value", "run_id"])
        w.writeheader()
        w.writerows(EV.long_rows(EV.chosen_runs(root/"runs", tuple(values))))


@pytest.fixture
def batches(tmp_path):
    stage = tmp_path/"stage"
    stage.mkdir()
    write_json(stage/"selection.json", dict(pairs=[dict(pair=p) for p in NAMES]))
    d0, a4, a1p = tmp_path/"d0", tmp_path/"a4", tmp_path/"a1p"
    write_batch(d0, dict(A0=.8196, A1=.8509, A2=.79, A3=.8465))
    write_batch(a4, dict(A0=.1, A4p=.8247))                                 # its A0 is not read: the reference is the batch of origin
    write_batch(a1p, dict(A1Ph=.8312, A1Ps=.8311))
    return dict(stage=stage, d0=d0, a4=a4, a1p=a1p, a5=tmp_path/"a5"/"runs", out=tmp_path/"a5"/"analysis")


def summarize(b):
    assert EV.summarize_a5(b["a5"], b["d0"], b["a4"], b["a1p"], b["out"], stage=b["stage"]) == 0
    return json.loads((b["out"]/"decision_a5.json").read_text()), (b["out"]/"summary_a5.md").read_text()


def test_summarize_five_pairs_then_nineteen(batches, capsys):
    for p in NAMES[:5]:
        write_run(batches["a5"], "A5", p, .86)
        write_run(batches["a5"], "A5P", p, .83)
    dec, text = summarize(batches)
    assert dec["n_pairs_in_batch"] == 5 and dec["batch"] == NAMES[:5]
    assert dec["gate1"]["expand"] is True and dec["gate1"]["A1_seg35"] == pytest.approx(.8509)
    assert dec["gate2"]["evaluated"] is False and dec["gate2"]["reading"] == "not_complete"
    row = dec["comparisons"]["A5 - A1"]["seg35"]
    assert row["n"] == 5 and row["positive"] == 5 and row["mean_diff"] == pytest.approx(.0091)
    assert dec["comparisons"]["A5P - A1Ph"]["seg4"]["mean_diff"] == pytest.approx(-.0012)
    assert dec["comparisons"]["A5 - A5P"]["seg35"]["mean_diff"] == pytest.approx(.03)
    assert dec["comparisons"]["A5 - A0"]["sdlogj_official"]["mean_diff"] == pytest.approx(0)
    assert dec["seg35_groups"]["small_six"]["labels"] == [4, 18, 19, 23, 34, 35] and dec["seg35_groups"]["large_six"]["labels"] == [1, 2, 5, 20, 21, 24]
    assert len(dec["seg35_groups"]["other_23"]["labels"]) == 23
    assert dec["seg35_groups"]["small_six"]["means"]["A5"] == pytest.approx(.89) and dec["seg35_groups"]["other_23"]["means"]["A1Ph"] == pytest.approx(.8312)
    parts = dec["seg35_groups"]
    assert sum(parts[g]["paired"]["A5 - A3"]["contribution_to_macro"] for g in parts) == pytest.approx(.86-.8465)
    assert dec["seg4_structures"]["1"]["paired"]["A5 - A1"]["mean_diff"] == pytest.approx(.0091) and dec["seg4_structures"]["3"]["name"] == "白质"
    assert dec["anomalies"] == [] and dec["sources"]["evaluator_sha256"]["A4p"] == ["sha-A4p"]
    assert dec["launch_loadavg"][f"A5/{NAMES[0]}"] == [.5, .9, 1.]
    for fragment in ("开发集", "单次观测", "seg35", "seg4", "第一关", "第二关", "组合读数", "不是同时段对照", "oracle", "小六个", "大六个", "其余 23 个", "皮层", "白质",
                     "Left-Vessel", "A5 − A4′", "A5P − A1Ph", "只有方向信息", "最小下界", "e2e", "受阻或异常", "没有。", "评测脚本版本"):
        assert fragment in text, fragment
    rows = list(csv.DictReader(open(batches["out"]/"long.csv", newline="")))
    assert {r["arm"] for r in rows} == {"A5", "A5P", "A0", "A1", "A3", "A4p", "A1Ph"} and {r["pair"] for r in rows} == set(NAMES[:5])
    assert not [d for d in batches["out"].iterdir() if d.name.startswith("previous_")] and dec["previous_summary_moved_to"] is None
    first_decision = (batches["out"]/"decision_a5.json").read_bytes()

    for p in NAMES[5:]:
        write_run(batches["a5"], "A5", p, .89)
        write_run(batches["a5"], "A5P", p, .85)
    dec, text = summarize(batches)
    assert dec["n_pairs_in_batch"] == 19 and dec["gate1"]["expand"] is True and dec["gate1"]["n"] == 5
    assert dec["gate1"]["A5_seg35"] == pytest.approx(.86)                   # still the first 5 pairs
    assert dec["gate2"]["evaluated"] is True
    assert dec["gate2"]["arms"]["A5"]["seg35_mean"] == pytest.approx((5*.86+14*.89)/19) and dec["gate2"]["arms"]["A5"]["n_with_field"] == 19
    assert dec["gate2"]["reading"] == "scalar_exhausted"                    # (5 x .86 + 14 x .89)/19 = .8821 < .885
    assert "只有方向信息" not in text
    kept = [d for d in batches["out"].iterdir() if d.name.startswith("previous_")]
    assert len(kept) == 1 and (kept[0]/"decision_a5.json").read_bytes() == first_decision and dec["previous_summary_moved_to"] == str(kept[0])
    assert (kept[0]/"long.csv").is_file() and (kept[0]/"summary_a5.md").is_file()


def test_summarize_only_compares_the_same_pairs(batches):
    for p in NAMES[:3]:
        write_run(batches["a5"], "A5", p, .86)
    write_run(batches["a5"], "A5P", NAMES[0], .83)
    dec, text = summarize(batches)
    assert dec["batch"] == NAMES[:3] and dec["comparisons"]["A5 - A1"]["seg35"]["n"] == 3 and dec["comparisons"]["A5P - A1"]["seg35"]["n"] == 1
    assert dec["gate1"]["status"] == "not_decidable" and dec["gate1"]["expand"] is False
    assert [a["pair"] for a in dec["anomalies"]] == NAMES[1:3] and all(a["arm"] == "A5P" for a in dec["anomalies"])
    assert "没有 metrics.json" in text
    rows = list(csv.DictReader(open(batches["out"]/"long.csv", newline="")))
    assert {r["pair"] for r in rows} == set(NAMES[:3])


def test_summarize_reports_failed_runs(batches):
    for p in NAMES[:5]:
        write_run(batches["a5"], "A5", p, .86)
        write_run(batches["a5"], "A5P", p, .83)
    write_run(batches["a5"], "A5P", NAMES[5], 0., cls="method_failure", final=False)
    dec, text = summarize(batches)
    assert dec["anomalies"][-1]["pair"] == NAMES[5] and "无终态场" in dec["anomalies"][-1]["what"] and "method_failure" in dec["anomalies"][-1]["what"]
    assert "method_failure" in text and dec["gate1"]["expand"] is True


def test_summarize_references_come_from_their_own_long_csv(batches):
    """The A0 of the a4 batch (.1) is never read; a reference's metrics.json that disagrees with its long.csv shuts the first gate."""
    for p in NAMES[:5]:
        write_run(batches["a5"], "A5", p, .86)
        write_run(batches["a5"], "A5P", p, .83)
    dec, _ = summarize(batches)
    assert dec["comparisons"]["A5 - A0"]["seg35"]["mean_diff"] == pytest.approx(.86-.8196) and dec["reference_problems"] == []
    path = batches["d0"]/"runs"/"A1"/NAMES[1]/"metrics.json"
    doc = json.loads(path.read_text())
    doc["metrics"]["seg35"] += .01
    write_json(path, doc)
    dec, text = summarize(batches)
    assert dec["gate1"]["expand"] is False and len(dec["reference_problems"]) == 1 and "参照与预登记或 metrics.json 不符" in text
    (batches["d0"]/"analysis"/"long.csv").unlink()
    with pytest.raises(SystemExit, match="long.csv"):
        summarize(batches)


def test_summary_never_under_a_reference_batch(batches):
    for root in (batches["d0"], batches["a4"], batches["a1p"], EV.D0, EV.D0A4, EV.D0A1P):
        with pytest.raises(SystemExit, match="never written under"):
            EV.summarize_a5(batches["a5"], batches["d0"], batches["a4"], batches["a1p"], root/"analysis_a5", stage=batches["stage"])


def test_evaluate_run_accepts_the_arms():
    assert EV.A5_ARMS == ARMS and EV.A5_GATE1_PIN == .8509 and EV.A5_ROUTE_THRESHOLD == .845 and EV.A5_ORACLE_THRESHOLD == .885


# ---- run_queue.sh --a5 on a synthetic tree --------------------------------------------------------------------------------

@pytest.fixture
def a5_tree(queue_tree):  # noqa: F811
    tree = queue_tree
    table = tree["repo"]/"configs"/"bayesreg"/"colour_table_a5_v1.json"
    table.write_text(json.dumps(dict(rows=[])))
    (tree["repo"]/".gitignore").write_text("data/\nresults/\n")
    tree["git"]("add", "configs/bayesreg/colour_table_a5_v1.json", ".gitignore")
    tree["git"]("commit", "-q", "-m", "colour table")
    registry = json.loads(tree["record"].read_text())["structure_registry"]
    records = {}
    for arm in ARMS:
        path = tree["results"]/"d0_v1_a5"/"inputs"/arm/"d0_inputs.json"
        path.parent.mkdir(parents=True)
        doc = dict(arm=arm, colour_table=dict(table_sha256=file_sha256(table)))
        if arm == "A5P":
            doc["structure_registry"] = registry
        write_json(path, doc)
        records[arm] = path
    bin_dir = tree["results"].parent/"bin"
    (bin_dir/"systemctl").write_text('#!/bin/bash\ncase "$*" in\n  *is-active*) exit 3;;\n  *) printf "%b" "${FAKE_UNITS:-}";;\nesac\n')
    (bin_dir/"systemd-run").write_text(f'#!/bin/bash\ntouch "{bin_dir}/systemd_run_called"\nexit 1\n')
    (bin_dir/"systemd-run").chmod(0o755)
    return dict(tree, table=table, records=records, a5=tree["results"]/"d0_v1_a5", bin=bin_dir)


def run_a5(tree, *args, **env):  # noqa: F811
    done = subprocess.run(["bash", str(QUEUE), "--a5", *args], env=dict(tree["env"], CUDA_VISIBLE_DEVICES="", **env), capture_output=True, text=True, timeout=180)
    log = tree["a5"]/"queue.log"
    return done.returncode, done.stdout, log.read_text() if log.exists() else ""


def order(pairs):
    expected = []
    for i, p in enumerate(pairs):
        for j in range(2):
            expected.append(f"{len(expected)+1} {ARMS[(i+j) % 2]} {p}")
    return expected


def test_queue_dry_run_lists_the_pair_major_rotating_order(a5_tree):
    code, out, log = run_a5(a5_tree, "--dry-run")
    assert code == 0 and log == "" and out.splitlines() == order(FAKE_PAIRS) and len(out.splitlines()) == 38
    assert out.splitlines()[:4] == [f"1 A5 {FAKE_PAIRS[0]}", f"2 A5P {FAKE_PAIRS[0]}", f"3 A5P {FAKE_PAIRS[1]}", f"4 A5 {FAKE_PAIRS[1]}"]
    code, out, _ = run_a5(a5_tree, "--first-5", "--dry-run")
    assert code == 0 and out.splitlines() == order(FAKE_PAIRS[:5]) and len(out.splitlines()) == 10
    assert not (a5_tree["a5"]/"runs").exists() and not (a5_tree["a5"]/"analysis").exists()


def test_first_5_goes_with_a5_only(a5_tree):
    done = subprocess.run(["bash", str(QUEUE), "--a1p", "--first-5", "--dry-run"], env=a5_tree["env"], capture_output=True, text=True, timeout=60)
    assert done.returncode == 2 and "--first-5 goes with --a5" in done.stdout


def test_queue_settings():
    text = QUEUE.read_text()
    assert "D0=$RESULTS/d0_v1_a5; ARMS=(A5 A5P); SELF_RE='^tag-d0a5-'; RIDP=d0a5; UNITP=${UNITP:-tag-d0a5}" in text
    assert "EXTRA_BUSY=('tag-d0a4-*' 'tag-d0a1p-*' 'tag-gax1-*')" in text
    assert "summarize-a5 --a5-runs-root" in text and text.count('exec 9> "$LINK_LOCK"') == 1
    assert subprocess.run(["bash", "-n", str(QUEUE)], capture_output=True).returncode == 0


A5_START_STOPS = {
    "segmenter training": (dict(FAKE_UNITS="tag-unet35-train.service loaded active running x\\n"), "STOP: units still active: tag-unet35-train"),
    "a D0 unit": (dict(FAKE_UNITS="tag-d0-A0-0438-0439.service loaded active running x\\n"), "STOP: units still active: tag-d0-A0"),
    "an A4 unit": (dict(FAKE_UNITS="tag-d0a4-A4p-0438-0439.service loaded active running x\\n"), "STOP: units still active: tag-d0a4"),
    "an A1P unit": (dict(FAKE_UNITS="tag-d0a1p-A1Ph-0438-0439.service loaded active running x\\n"), "STOP: units still active: tag-d0a1p"),
    "an X1 unit": (dict(FAKE_UNITS="tag-gax1-X1h-0438-0439.service loaded active running x\\n"), "STOP: units still active: tag-gax1"),
    "GPU compute process": (dict(FAKE_PIDS="4242\\n"), "STOP: GPU not idle at start"),
    "registry modified": ("modify_registry", "STOP: the repository registry"),
    "table modified": ("modify_table", "STOP: the colour table"),
    "A5P from another registry": ("record_registry", "STOP: the A5P inputs were not made with the current"),
    "A5 from another table": ("record_table_A5", "STOP: the A5 inputs were not made with the current colour table"),
    "A5P from another table": ("record_table_A5P", "STOP: the A5P inputs were not made with the current colour table"),
    "A5 without inputs": ("no_record_A5", "STOP: the A5 inputs were not made with the current colour table"),
    "A5P without inputs": ("no_record_A5P", "STOP: the A5P inputs were not made with the current colour table"),
    "A5 dry-run inputs": ("partial_A5", "STOP: the A5 inputs were not made with the current colour table, or are partial"),
    "A5P dry-run inputs": ("partial_A5P", "STOP: the A5P inputs were not made with the current colour table, or are partial"),
}


@pytest.mark.parametrize("case", sorted(A5_START_STOPS))
def test_queue_start_checks_stop(a5_tree, case):
    change, fragment = A5_START_STOPS[case]
    env = change if isinstance(change, dict) else {}
    if change == "modify_registry":
        a5_tree["registry"].write_text(a5_tree["registry"].read_text()+"\n")
    elif change == "modify_table":
        a5_tree["table"].write_text(a5_tree["table"].read_text()+"\n")
    elif change == "record_registry":
        doc = json.loads(a5_tree["records"]["A5P"].read_text())
        doc["structure_registry"]["sha256"] = "0"*64
        write_json(a5_tree["records"]["A5P"], doc)
    elif isinstance(change, str) and change.startswith("record_table_"):
        path = a5_tree["records"][change.rsplit("_", 1)[1]]
        doc = json.loads(path.read_text())
        doc["colour_table"]["table_sha256"] = "0"*64
        write_json(path, doc)
    elif isinstance(change, str) and change.startswith("no_record_"):
        a5_tree["records"][change.rsplit("_", 1)[1]].unlink()
    elif isinstance(change, str) and change.startswith("partial_"):
        path = a5_tree["records"][change.rsplit("_", 1)[1]]
        write_json(path, dict(json.loads(path.read_text()), partial=True))
    code, _, log = run_a5(a5_tree, "--first-5", **env)
    assert code == 1 and fragment in log, log
    assert "worktree HEAD" not in log and not list((a5_tree["a5"]/"runs").glob("*/*"))


def test_queue_start_checks_pass_on_a_quiet_machine(a5_tree):
    code, _, log = run_a5(a5_tree, "--first-5")
    assert code == 1 and "STOP: worktree HEAD is not eb808be" in log, log
    assert "A5 colour table check:" in log and "A5P colour table check:" in log and "A5P registry check:" in log and "A5 registry check:" not in log
    assert "first gate" not in log and "10 runs in order" in log
    for name in ("d0_v1", "d0_v1_a1p"):
        assert not (a5_tree["results"]/name).exists()


@pytest.mark.parametrize("decision,opens", [(None, False), (dict(gate1=dict(expand=False, text="no")), False), (dict(gate1=dict(expand=True, text="yes")), True)])
@pytest.mark.parametrize("args", [(), ("--only", "A5P:7")])
def test_queue_first_gate_guards_the_later_pairs(a5_tree, decision, opens, args):
    if decision is not None:
        path = a5_tree["a5"]/"analysis"/"decision_a5.json"
        path.parent.mkdir(parents=True)
        write_json(path, decision)
    code, _, log = run_a5(a5_tree, *args)
    assert code == 1
    if opens:
        assert "STOP: worktree HEAD is not eb808be" in log and "expand True" in log, log
    else:
        assert "STOP: the first gate has not opened" in log and "worktree HEAD" not in log, log
    assert not list((a5_tree["a5"]/"runs").glob("*/*"))


def frozen_tree(tree):
    """A frozen worktree that passes the start checks: a clone at the tag eb808be, an empty source list, and the arm inputs of 19 pairs
    (arbitrary bytes, the queue only compares hashes)."""
    repo, results = tree["repo"], tree["results"]
    (repo/"experiments"/"d0").mkdir(parents=True, exist_ok=True)
    write_json(repo/"experiments"/"d0"/"configC_reference.json", dict(freeze_source_py_sha256={}))
    tree["git"]("tag", "eb808be")
    subprocess.run(["git", "clone", "-q", str(repo), str(results/"frozen_eb808be")], check=True, capture_output=True)
    selection = (repo/"data"/"three_d"/"stage_L2R_val_v1"/"selection.json").read_text()
    for arm in ARMS:
        stage = tree["a5"]/"inputs"/arm/"stage_L2R_val_v1"
        stage.mkdir(parents=True)
        (stage/"selection.json").write_text(selection)
        files = {}
        for p in FAKE_PAIRS:
            d = stage/p
            d.mkdir()
            (d/"pair.npz").write_bytes(f"{arm} {p}".encode())
            write_json(d/"manifest.json", dict(pair=p, pair_sha256=file_sha256(d/"pair.npz"), preprocessing_d0=dict(arm=arm)))
            files[f"stage_L2R_val_v1/{p}/pair.npz"] = file_sha256(d/"pair.npz")
            files[f"stage_L2R_val_v1/{p}/manifest.json"] = file_sha256(d/"manifest.json")
        doc = json.loads(tree["records"][arm].read_text())
        write_json(tree["records"][arm], dict(doc, files_sha256=files))


def finished_run(tree, arm, pair):
    d = tree["a5"]/"runs"/arm/pair
    d.mkdir(parents=True)
    (d/"queue_done.json").write_text('{"summary": "done before"}\n')
    (d/"metrics.json").write_text('{"run_id": "before"}\n')
    return d


def snapshot(directory):
    return {str(p): file_sha256(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def test_first_5_then_all_reruns_nothing(a5_tree):
    """--first-5 with its 10 runs done starts no run and summarizes; then the whole queue skips those 10 and goes on with item 11; nothing in
    their directories changes.  Item 11 is an interrupted run directory, so the queue stops there, before any unit."""
    frozen_tree(a5_tree)
    for p in FAKE_PAIRS[:5]:
        for arm in ARMS:
            finished_run(a5_tree, arm, p)
    runs = a5_tree["a5"]/"runs"
    before = snapshot(runs)
    code, out, log = run_a5(a5_tree, "--first-5")
    assert code == 0 and "start checks passed" in log and "10 runs in order" in log and "queue finished: 10 items" in log, log
    assert "summary exit" in log and out.count("skip ") == 10 and snapshot(runs) == before
    path = a5_tree["a5"]/"analysis"/"decision_a5.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    code, out, log = run_a5(a5_tree)
    assert code == 1 and "STOP: the first gate has not opened" in log, log       # the summary of the fake repository wrote no decision
    write_json(path, dict(gate1=dict(expand=True, text="yes")))
    interrupted = runs/ARMS[(5+0) % 2]/FAKE_PAIRS[5]
    interrupted.mkdir(parents=True)
    code, out, log = run_a5(a5_tree)
    assert code == 8 and "38 runs in order" in log and f"{interrupted} exists without exit_code" in log and "queue stopped at item 11/38" in log, log
    assert out.count("skip ") == 10 and snapshot(runs) == before
    assert sorted(p.name for p in (runs/"A5").iterdir()) == FAKE_PAIRS[:5] and sorted(p.name for p in (runs/"A5P").iterdir()) == FAKE_PAIRS[:6]
    assert not (a5_tree["bin"]/"systemd_run_called").exists()
