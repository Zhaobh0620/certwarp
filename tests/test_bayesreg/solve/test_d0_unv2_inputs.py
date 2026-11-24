"""experiments/d0/make_inputs.py for A5Pv2 (work package UNV2, runs/20261103-unv2/plan.md), on synthetic files only (the fixtures of
test_d0_a4_tools; no data/, no real image, label or prediction):

- the default path is bit for bit that of the script before this work package (base commit 82a9db7, read from git): the same pair.npz,
  the same manifest and record but for the script's own hash and the time; parser defaults and the record of UNetSource unchanged;
- --segmenter-id, --registry, --pred-root, --inputs: the same colouring from the same posterior, the other segmenter in the provenance and
  the records, the other registry registered, committed and clean; the checks of §12.4 still stop it;
- the command line: all four or none, A5P only, never under the directories of the other arms.
"""
import importlib.util
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from jd_curl.three_d.nifti import file_sha256

from bayesreg.seg import structure_layer

from ..io_seg._helpers import DEV, FOLD0, FOLD1, model_entry
from . import test_d0_a4_tools as A4
from ..io_seg._helpers import synthetic_folds  # noqa: E402   (C23.9-13 (8), revision 6e)
from .test_d0_a4_tools import MI, PAIRS, ROOT, SUBJECTS, env, rehash, write_prediction  # noqa: F401
from .test_run_pair_structure import write_registry

BASE_COMMIT = "82a9db7"                     # the commit this work package starts from
V2_ID = "unet35_v2_full"


def base_script(tmp_path):
    """The module experiments/d0/make_inputs.py as committed at BASE_COMMIT."""
    source = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE_COMMIT}:experiments/d0/make_inputs.py"], capture_output=True, text=True)
    if source.returncode != 0:
        pytest.skip(f"git show {BASE_COMMIT} is not available here: {source.stderr.strip()}")
    path = tmp_path/"base"/"experiments"/"d0"/"make_inputs.py"
    path.parent.mkdir(parents=True)
    path.write_text(source.stdout)
    spec = importlib.util.spec_from_file_location("d0_make_inputs_base", path)
    module = importlib.util.module_from_spec(spec)
    before = list(sys.path)
    spec.loader.exec_module(module)
    sys.path[:] = before                    # the module puts its own (temporary) root first
    return module


def strip(document):
    """The record without the script's own hash and the time, and without the hash of each manifest.json: a manifest holds the script's
    hash, so its file hash differs by construction; the manifests themselves are compared below, without that one field."""
    out = {k: v for k, v in document.items() if k not in ("script_sha256", "created_utc")}
    out["files_sha256"] = {k: v for k, v in document["files_sha256"].items() if not k.endswith("manifest.json")}
    return out


def strip_manifest(manifest):
    pre = dict(manifest["preprocessing_d0"])
    pre.pop("script_sha256")
    return dict(manifest, preprocessing_d0=pre)


@pytest.mark.parametrize("arm", ["A5", "A5P", "A4p", "A1Ph"])
@synthetic_folds("env")
def test_default_path_is_bit_for_bit_the_base_script(env, tmp_path, arm):  # noqa: F811
    base = base_script(tmp_path)
    assert base.RECIPES == MI.RECIPES and base.COLOUR_TABLE_A5_SHA256 == MI.COLOUR_TABLE_A5_SHA256
    unet_arm = arm in MI.UNET_ARMS
    kwargs = dict(stage=env["stage"], allowlist=env["allowlist"], colour_table=MI.COLOUR_TABLE_A5, seg4_map=MI.SEG4_MAP)
    if unet_arm:
        source = env["unet"]
        fields = dict(pred_root=source.pred_root, folds=source.folds, folds_sha256=source.folds_sha256, models_root=source.models_root,
                      image_root=source.image_root)
    else:
        fields = None
    old_root, new_root = env["root"]/"old", env["root"]/"new"
    real_load = base.load_seg4_map                  # its check file is found relative to the code tree, not this temporary one
    base.load_seg4_map = lambda path=MI.SEG4_MAP, root=ROOT: real_load(path, root)
    base.build_arm(arm, old_root, env["seg_root"], unet=base.UNetSource(**fields) if unet_arm else None, **kwargs)
    MI.build_arm(arm, new_root, env["seg_root"], unet=MI.UNetSource(**fields) if unet_arm else None, **kwargs)
    old_record = json.loads((old_root/arm/"d0_inputs.json").read_text())
    new_record = json.loads((new_root/arm/"d0_inputs.json").read_text())
    assert strip(old_record) == strip(new_record)
    assert list(old_record) == list(new_record)
    for fixed, moving in PAIRS:
        rel = f"{MI.STAGE_NAME}/{fixed}__{moving}"
        old_pair, new_pair = old_root/arm/rel/"pair.npz", new_root/arm/rel/"pair.npz"
        assert file_sha256(old_pair) == file_sha256(new_pair)
        old_manifest = json.loads((old_root/arm/rel/"manifest.json").read_text())
        new_manifest = json.loads((new_root/arm/rel/"manifest.json").read_text())
        assert strip_manifest(old_manifest) == strip_manifest(new_manifest)


def test_default_parser_and_source():
    source = MI.UNetSource()
    assert source.segmenter_id is None and source.registry is None and source.pred_root == MI.UNET_ROOT/"pred"
    assert source.record() == dict(pred_root=str(MI.UNET_ROOT/"pred"), folds=str(MI.FOLDS), models_root=str(MI.UNET_ROOT),
                                   image_root="/mnt/d/data")
    assert MI.D0A5.name == "d0_v1_a5" and MI._recipe("A5P", source) == MI.RECIPES["A5P"] and MI._recipe("A5P", None) == MI.RECIPES["A5P"]


@pytest.fixture
def v2(env, tmp_path, monkeypatch):  # noqa: F811
    """The v2 registry, committed in the test's repository, and the v2 predictions: the same posteriors as v1, another segmenter, and the
    three members expected tonight (A0, A1 in fold A, B0 in fold B)."""
    monkeypatch.setattr(A4, "SEG_ID", V2_ID)
    models = {f"{V2_ID}_foldA": model_entry(V2_ID, 0, sorted(env["fold0"]+env["fold1"]), members=2),          # env's training draw (C23.9-13 (8), revision 6e)
              f"{V2_ID}_foldB": model_entry(V2_ID, 1, sorted(env["fold0"]+env["fold1"]), members=1)}
    registry = write_registry(env["registry"].parent/"segmenters_v2.json", models, ensemble=False)
    A4.commit_registry(registry, "v2 registry")
    monkeypatch.setattr(structure_layer, "FORMAL_REGISTRIES", (env["registry"], registry))   # stand in for the repository's two files
    for s in SUBJECTS:
        write_prediction(tmp_path/"pred_v2", s, models)
    source = env["unet"]
    unet = MI.UNetSource(pred_root=tmp_path/"pred_v2", folds=source.folds, folds_sha256=source.folds_sha256, models_root=source.models_root,
                         image_root=source.image_root, segmenter_id=V2_ID, registry=registry)
    return dict(registry=registry, models=models, unet=unet, pred=tmp_path/"pred_v2")


def build(env, unet, inputs, arm="A5P", **changes):  # noqa: F811
    MI.build_arm(arm, env["root"]/inputs, changes.pop("seg_root", env["seg_root"]), stage=env["stage"], allowlist=env["allowlist"], unet=unet, **changes)
    return env["root"]/inputs/arm


@synthetic_folds("env")
def test_a5pv2_same_colouring_other_segmenter(env, v2):  # noqa: F811
    v1_out = build(env, env["unet"], "inputs_v1")
    v2_out = build(env, v2["unet"], "inputs_v2")
    record = json.loads((v2_out/"d0_inputs.json").read_text())
    assert record["structure_registry"]["path"] == str(v2["registry"]) and record["structure_registry"]["status"] == "registered"
    assert record["structure_registry"]["git"]["tracked"] is True and record["structure_registry"]["git"]["clean"] is True
    assert record["unet_source"]["segmenter_id"] == V2_ID and record["unet_source"]["registry"] == str(v2["registry"])
    assert V2_ID in record["recipe"] and "unet35_v1_ens" not in record["recipe"]
    assert record["colour_table"]["table_sha256"] == MI.COLOUR_TABLE_A5_SHA256 and record["alignment_qc"]["threshold"] == .8
    for fixed, moving in PAIRS:
        rel = f"{MI.STAGE_NAME}/{fixed}__{moving}"
        with np.load(v1_out/rel/"pair.npz") as a, np.load(v2_out/rel/"pair.npz") as b:
            for key in MI.KEY_ORDER:
                assert a[key].tobytes() == b[key].tobytes(), key
        manifest = json.loads((v2_out/rel/"manifest.json").read_text())
        assert V2_ID in manifest["preprocessing"] and manifest["preprocessing_d0"]["colour"]["fixed"]["segmenter_id"] == V2_ID
        assert manifest["preprocessing_d0"]["recipe"] == MI.RECIPES["A5P"].replace("unet35_v1_ens", V2_ID)
        assert len(manifest["preprocessing_d0"]["colour"]["fixed"]["members"]) == 3
    v1_record = json.loads((v1_out/"d0_inputs.json").read_text())
    assert v1_record["recipe"] == MI.RECIPES["A5P"] and "segmenter_id" not in v1_record["unet_source"]


def test_a5pv2_refuses_the_v1_predictions(env, v2):  # noqa: F811
    source = v2["unet"]
    crossed = MI.UNetSource(pred_root=env["unet"].pred_root, folds=source.folds, folds_sha256=source.folds_sha256, models_root=source.models_root,
                            image_root=source.image_root, segmenter_id=V2_ID, registry=source.registry)
    with pytest.raises(SystemExit, match="refused"):
        build(env, crossed, "inputs_x")
    assert not (env["root"]/"inputs_x"/"A5P"/"d0_inputs.json").exists()


def _with_registry(v2, other):
    source = v2["unet"]
    return MI.UNetSource(pred_root=source.pred_root, folds=source.folds, folds_sha256=source.folds_sha256, models_root=source.models_root,
                         image_root=source.image_root, segmenter_id=V2_ID, registry=other)


@pytest.mark.parametrize("change", ["modified", "untracked", "missing"])
def test_a5pv2_needs_its_registry_committed_and_clean(env, v2, monkeypatch, change):  # noqa: F811
    path = v2["registry"]
    if change == "modified":
        path.write_text(path.read_text()+"\n")
    elif change == "untracked":
        other = path.with_name("segmenters_v3.json")
        shutil.copyfile(path, other)
        monkeypatch.setattr(structure_layer, "FORMAL_REGISTRIES", (env["registry"], path, other))   # on the list: the git check must catch it
        v2["unet"] = _with_registry(v2, other)
    else:
        path.unlink()
    with pytest.raises(SystemExit, match="repository registry"):
        build(env, v2["unet"], "inputs_x")
    assert not (env["root"]/"inputs_x"/"A5P").exists()


def test_a5pv2_refuses_a_committed_clean_registry_off_the_list(env, v2):  # noqa: F811
    other = v2["registry"].with_name("segmenters_v3.json")
    shutil.copyfile(v2["registry"], other)
    A4.commit_registry(other, "another registry")
    with pytest.raises(SystemExit, match="not one of the repository registries"):
        build(env, _with_registry(v2, other), "inputs_x")
    assert not (env["root"]/"inputs_x"/"A5P").exists()


def _evaluation(env, v2, *, run=True, roots_v2=None, form="bayesreg.unet35.eval_seg.v1"):  # noqa: F811
    directory = v2["pred"].parent/"eval_seg"
    directory.mkdir()
    (directory/"eval_seg.json").write_text(json.dumps(dict(
        format=form, roots=dict(v1="x", v2=str(roots_v2 or v2["pred"])),
        decision=dict(run_a5pv2=run, v2_minus_v1=.001 if run else -.01, text="t"))))
    return directory/"eval_seg.json"


def _main(env, v2, monkeypatch, inputs="inputs_main", extra=()):  # noqa: F811
    """make_inputs.main as the lead runs it, on the synthetic stage and allowlist; returns the arm directory."""
    real = MI.build_arm
    monkeypatch.setattr(MI, "build_arm", lambda arm, root, seg_root, **kw: real(arm, root, seg_root, allowlist=env["allowlist"], **kw))
    out = env["root"]/inputs
    assert MI.main(["--arms", "A5P", "--segmenter-id", V2_ID, "--registry", str(v2["registry"]), "--pred-root", str(v2["pred"]),
                    "--inputs", str(out), "--stage", str(env["stage"]), "--seg-root", str(env["seg_root"]), *extra]) == 0
    return out/"A5P"


@synthetic_folds("env")
def test_main_records_the_v2_models_root_everywhere_and_the_gate(env, v2, monkeypatch):  # noqa: F811
    evaluation = _evaluation(env, v2)
    arm = _main(env, v2, monkeypatch)
    record = json.loads((arm/"d0_inputs.json").read_text())
    root = str(v2["pred"].parent)
    assert record["unet_source"]["models_root"] == root and record["unet_source"]["pred_root"] == str(v2["pred"])
    assert record["unet_source"]["folds"] == str(MI.FOLDS)                      # D24 reads the training subjects from there; the same 106
    assert record["pairs"] and all(p["structure_layer"]["source"]["models_root"] == root for p in record["pairs"].values())
    assert all(p["structure_layer"]["source"]["formal_registry"] == str(v2["registry"]) for p in record["pairs"].values())
    assert all(p["structure_layer"]["source"]["segmenter_id"] == V2_ID for p in record["pairs"].values())
    gate = record["unet_source"]["a5pv2_gate"]
    assert gate["passed_by"] == "the rule" and gate["run_a5pv2"] is True and gate["bz_decision"] is None
    assert gate["eval_sha256"] == file_sha256(evaluation)


@pytest.mark.parametrize("case,fragment", [("missing", "no evaluation"), ("false", "does not allow"), ("foreign", "not an evaluation of"),
                                           ("format", "not an evaluation of")])
def test_main_stops_unless_the_rule_allows_a5pv2(env, v2, monkeypatch, case, fragment):  # noqa: F811
    if case == "false":
        _evaluation(env, v2, run=False)
    elif case == "foreign":
        _evaluation(env, v2, roots_v2=env["root"]/"elsewhere")
    elif case == "format":
        _evaluation(env, v2, form="other")
    with pytest.raises(SystemExit, match=fragment):
        _main(env, v2, monkeypatch)
    assert not (env["root"]/"inputs_main").exists()


@synthetic_folds("env")
def test_only_the_bz_decision_goes_past_the_rule(env, v2, monkeypatch):  # noqa: F811
    _evaluation(env, v2, run=False)
    arm = _main(env, v2, monkeypatch, extra=["--bz-decision-run-a5pv2", "decision_log 11-04"])
    gate = json.loads((arm/"d0_inputs.json").read_text())["unet_source"]["a5pv2_gate"]
    assert gate["passed_by"] == "BZ decision" and gate["bz_decision"] == "decision_log 11-04"
    assert gate["run_a5pv2"] is False and "does not allow" in gate["problem"]
    with pytest.raises(SystemExit, match="A5Pv2 only"):
        MI.main(["--arms", "A5P", "--bz-decision-run-a5pv2", "x"])


def test_a5pv2_takes_only_its_segmenter():
    with pytest.raises(SystemExit, match="must be unet35_v2_full"):
        MI.main(["--arms", "A5P", "--segmenter-id", "something_else", "--registry", "r.json", "--pred-root", "p", "--inputs", "/tmp/x"])


@synthetic_folds("env")
def test_a5pv2_every_check_of_a5p_stops_it(env, v2):  # noqa: F811
    directory = v2["pred"]/DEV[1]
    for name in ("probs36.npz", "structure_groups.npz"):
        with np.load(directory/name) as z:
            p = z["p"].copy()
        np.savez_compressed(directory/name, p=p[:, ::-1, ::-1, ::-1])
    rehash(directory)
    with pytest.raises(SystemExit, match="alignment QC"):
        build(env, v2["unet"], "inputs_qc")
    assert not (env["root"]/"inputs_qc"/"A5P"/"d0_inputs.json").exists()
    with pytest.raises(SystemExit, match="Atropos"):
        build(env, v2["unet"], "inputs_at", seg_root=None)
    other = env["root"]/"table.json"
    other.write_text(MI.COLOUR_TABLE_A5.read_text()+"\n")
    with pytest.raises(SystemExit, match="pinned"):
        build(env, v2["unet"], "inputs_tb", colour_table=other)


GOOD = ["--segmenter-id", V2_ID, "--registry", "r.json", "--pred-root", "p", "--inputs", "/home/rmcfz/seq_vp_tag_results/d0_v1_a5v2/inputs"]


@pytest.mark.parametrize("argv,fragment", [
    (["--arms", "A5P", "--segmenter-id", V2_ID], "all of"),
    (["--arms", "A5P", "--registry", "r.json"], "all of"),
    (["--arms", "A5P", "--segmenter-id", V2_ID, "--registry", "r.json", "--pred-root", "p"], "all of"),
    (["--arms", "A5P", "--segmenter-id", V2_ID, "--registry", "r.json", "--inputs", "/tmp/x"], "all of"),
    (["--arms", "A5", *GOOD], "A5P only"),
    (["--arms", "A5", "A5P", *GOOD], "A5P only"),
    (["--arms", "A4p", *GOOD], "A5P only"),
    (["--arms", "A1Ph", *GOOD], "A5P only"),
    ([*GOOD], "A5P only"),
])
def test_command_line_refusals(argv, fragment):
    with pytest.raises(SystemExit, match=fragment):
        MI.main(argv)


@pytest.mark.parametrize("root", [MI.D0V1, MI.D0A4, MI.D0A1P, MI.D0A5])
def test_a5pv2_is_never_written_under_the_other_arms(root):
    with pytest.raises(SystemExit, match="never written under"):
        MI.main(["--arms", "A5P", "--segmenter-id", V2_ID, "--registry", "r.json", "--pred-root", "p", "--inputs", str(root/"inputs_v2")])


def test_pred_root_is_the_old_option_too(env, monkeypatch):  # noqa: F811
    seen = {}
    monkeypatch.setattr(MI, "build_arm", lambda arm, inputs, seg_root, **kw: seen.update(arm=arm, unet=kw["unet"], inputs=inputs))
    assert MI.main(["--arms", "A5P", "--pred-root", "/x/pred", "--inputs", "/tmp/somewhere"]) == 0
    assert str(seen["unet"].pred_root) == "/x/pred" and seen["unet"].segmenter_id is None
    assert MI.main(["--arms", "A5P", "--unet-pred-root", "/y/pred", "--inputs", "/tmp/somewhere"]) == 0
    assert str(seen["unet"].pred_root) == "/y/pred"
    assert MI.main(["--arms", "A5P"]) == 0
    assert seen["unet"].pred_root == MI.UNET_ROOT/"pred" and seen["inputs"] == MI.D0A5/"inputs"          # the defaults


def test_structure_source_default_record_has_no_new_key(tmp_path):
    source = structure_layer.StructureSource(root=tmp_path, registry_path=tmp_path/"r", allowlist_path=tmp_path/"a", models_root=tmp_path, image_root=tmp_path)
    assert "formal_registry" not in source.record()
    other = structure_layer.StructureSource(root=tmp_path, registry_path=tmp_path/"r", allowlist_path=tmp_path/"a", models_root=tmp_path, image_root=tmp_path,
                                            formal_registry=tmp_path/"r")
    assert other.record()["formal_registry"] == str(tmp_path/"r")
    assert structure_layer.registry_status(tmp_path/"r")["status"] == structure_layer.STAGING
    formal = structure_layer.registry_status(tmp_path/"r", tmp_path/"r")
    assert formal["status"] == structure_layer.REGISTERED and str(tmp_path/"r") in formal["note"]
    assert structure_layer.registry_status(structure_layer.REPO_REGISTRY)["note"] == "正式登记（configs/bayesreg/segmenters.json）"
