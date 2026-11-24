"""The full-training-set segmenter unet35_v2_full (work package UNV2, runs/20261103-unv2/任务书.md; 04_分割器.md §14).

Real repository files are read for subject ids only (folds.json, folds_full_v2.json, train_subjects.txt, dev_subjects.json, the
sealed lists); no image, no label, nothing under data/. Everything else is synthetic (tmp_path, ids OASIS_OAS1_09xx_MR1).
- the export rule: the file is the rule's bytes, 106 subjects, 98 + 8 per fold, calib outside train, no development, T0 or
  hold-out subject; the parsers refuse each other's files and malformed ones;
- v1 does not move: every pin, the snapshot paths, the cli defaults, the repository registry's sha256;
- the pins of v2: pass on the pinned files, each departure refused, also a folds.json the file was not derived from;
- registration and member checks: at least one member per fold, equal contacts, append-only;
- --dry-run of the profile with two members (one per fold) and three (A0, B0, A1): train -> calibrate -> register -> infer, provenance
  accepted on a development pair, the ensemble is the plain mean over the members, refusals (training subject, T0, sealed, label read);
- experiments/unv2/chain.sh: syntax, refusals, member order, deadline (0 to 4 members), failure stops and is not rerun; one real synthetic
  chain stopped by its deadline after A0, B0, A1: registration, inference, the structure layer of a development pair, evaluation,
  idempotent; another stopped after A0, B0: through the evaluation with two members; eval_seg's outputs and its rule;
- the flip is gone: no option, no branch.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import types

import numpy as np
import pytest
import torch

from jd_curl.validation import InputValidationError

from bayesreg.eval.truth import load_sealed_lists
from bayesreg.io.guard import SubjectSealed, load_dev_allowlist
from bayesreg.io.pair import ProvenanceRejected, load_pair, load_segmenter_registry, subject_list_sha256
from bayesreg.seg import structure_layer
from bayesreg.seg.structure_layer import StructureLayer, StructureSource
from bayesreg.seg.unet import cli, pins
from bayesreg.seg.unet.data import (L2R_VAL_RANGE, LabelAccessRefused, LabelGuard, FullFoldSplit, OasisFiles, derive_full_folds,
                                    full_folds_document, load_folds, load_full_folds, normalize_image, parse_folds, parse_full_folds,
                                    read_image, read_labels, render_full_folds)
from bayesreg.seg.unet.infer import load_member, read_probabilities, segment_subject
from bayesreg.seg.unet.pins import PinnedInputRefused, load_pinned_full_inputs, load_pinned_inputs
from bayesreg.seg.unet.registry import (MODEL_PREFIX_V2, SEGMENTER_ID_V2, build_entries, find_member_files, register, verify_members)
from bayesreg.seg.unet.snapshot import CODE_ROOT, CodeNotClean, SNAPSHOT_PATHS, SNAPSHOT_PATHS_V2, SNAPSHOT_PROFILES, make_snapshot
from bayesreg.seg.unet.synthetic import allowlist_document, folds_document, oasis, write_json, write_pair
from bayesreg.seg.unet.train import predict_logits

from ._helpers import A_CALIB, A_TRAIN, B_CALIB, B_TRAIN, DEV, T0, TINY, TRAIN_SOURCE, TRAINING

SPLITS = CODE_ROOT/"splits"/"bayesreg_v1"
FOLDS_V1, FOLDS_V2 = SPLITS/"folds.json", SPLITS/"folds_full_v2.json"
ALLOWLIST = CODE_ROOT/"configs"/"bayesreg"/"dev_subjects.json"
CHAIN = CODE_ROOT/"experiments"/"unv2"/"chain.sh"
EXPORT = CODE_ROOT/"experiments"/"unv2"/"export_folds.py"
EVAL_SEG = CODE_ROOT/"experiments"/"unv2"/"eval_seg.py"


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EV = load_script("unv2_eval_seg", EVAL_SEG)
EX = load_script("unv2_export_folds", EXPORT)


def v1_doc():
    return folds_document(A_TRAIN, A_CALIB, B_TRAIN, B_CALIB)


def v2_doc():
    return full_folds_document(parse_folds(v1_doc()))


# ---- the export rule on the repository's files (ids only) ------------------------------------------------------------------------

def test_repository_file_is_the_rule_applied_to_folds_json():
    assert FOLDS_V2.read_bytes() == render_full_folds(full_folds_document(load_folds(FOLDS_V1)))
    assert hashlib.sha256(FOLDS_V2.read_bytes()).hexdigest() == pins.PINNED_FULL_FOLDS_SHA256
    assert hashlib.sha256(FOLDS_V1.read_bytes()).hexdigest() == pins.PINNED_FOLDS_SHA256
    assert EX.main(["--check"]) == 0
    document = json.loads(FOLDS_V2.read_text())
    assert document["derived_from"] == dict(file="splits/bayesreg_v1/folds.json", sha256=pins.PINNED_FOLDS_SHA256)


def test_repository_split_counts_and_roles():
    v1, v2 = load_folds(FOLDS_V1, expect_counts=(45, 8)), load_full_folds(FOLDS_V2, expect_counts=(98, 8))
    everyone = set(v1.training_subjects)
    assert len(everyone) == 106 and v2.training_subjects == frozenset(everyone)
    assert everyone == set((SPLITS/"train_subjects.txt").read_text().split())
    for fold, other in (("A", "B"), ("B", "A")):
        train, calib = set(v2.subjects(fold, "train")), set(v2.subjects(fold, "calib"))
        assert len(train) == 98 and len(calib) == 8 and not train & calib and train | calib == everyone
        assert v2.subjects(fold, "calib") == v1.subjects(fold, "calib")     # the v1 calib of that fold, unchanged
        assert calib <= set(v2.subjects(other, "train"))                    # and in the other fold's training data
        assert v2.label_contact(fold) == sorted(everyone)


def test_repository_split_has_no_development_t0_or_holdout_subject():
    v2 = load_full_folds(FOLDS_V2)
    allowlist = load_dev_allowlist(ALLOWLIST)
    for source in allowlist.sources:
        if source["name"] != "bayesreg_train_v1":
            assert not set(source["subjects"]) & v2.training_subjects, source["name"]
    assert not v2.training_subjects & allowlist.never_for_training
    sealed = load_sealed_lists(CODE_ROOT)                                   # B-line hold-out, H1, H3, H4 (ids only)
    assert {sealed.which(s) for s in v2.training_subjects} == {None}
    numbers = [int(s.split("_")[2]) for s in v2.training_subjects]
    assert not [n for n in numbers if L2R_VAL_RANGE[0] <= n <= L2R_VAL_RANGE[1]]
    guard = LabelGuard(v2, allowlist, training_source="bayesreg_train_v1")  # constructs: every rule of the guard holds
    calib_a = v2.subjects("A", "calib")[0]
    guard.check(calib_a, "A")
    guard.check(calib_a, "B")                                               # the calib of A is training data of B
    for subject in ("OASIS_OAS1_0438_MR1", sorted(allowlist.never_for_training)[0]):
        with pytest.raises(LabelAccessRefused):
            guard.check(subject)
    assert guard.reads == []


def test_derivation_on_synthetic_folds():
    folds = derive_full_folds(parse_folds(v1_doc()))
    assert folds["A"]["calib"] == A_CALIB and folds["B"]["calib"] == B_CALIB
    assert folds["A"]["train"] == sorted(set(TRAINING)-set(A_CALIB)) and folds["B"]["train"] == sorted(set(TRAINING)-set(B_CALIB))
    split = parse_full_folds(v2_doc())
    assert isinstance(split, FullFoldSplit) and split.counts() == {f: dict(train=5, calib=1) for f in "AB"}
    assert split.record()["derived_from"]["sha256"] == parse_folds(v1_doc()).sha256
    with pytest.raises(InputValidationError, match="both folds"):
        split.fold_of(A_CALIB[0])
    with pytest.raises(InputValidationError):
        derive_full_folds(split)                                            # the rule takes a v1 split only


def test_parsers_refuse_each_others_files(tmp_path):
    with pytest.raises(InputValidationError, match="listed twice"):
        load_folds(FOLDS_V2)
    with pytest.raises(InputValidationError, match="schema"):
        load_full_folds(FOLDS_V1)
    path = tmp_path/"full.json"
    path.write_bytes(render_full_folds(v2_doc()))
    with pytest.raises(InputValidationError, match="expected"):
        load_full_folds(path, expect_counts=(98, 8))
    assert load_full_folds(path, expect_counts=(5, 1)).sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("change,fragment", [
    ("calib_in_train", "both in train and calib"),
    ("unsorted", "sorted and distinct"),
    ("not_canonical", "canonical"),
    ("different_cover", "cover different"),
    ("same_calib", "calib of both"),
    ("extra_key", "unexpected keys"),
    ("no_fold", "exactly"),
])
def test_malformed_full_folds_are_refused(change, fragment):
    doc = json.loads(json.dumps(v2_doc()))
    folds = doc["folds"]
    if change == "calib_in_train":
        folds["A"]["train"] = sorted(folds["A"]["train"]+folds["A"]["calib"])
    elif change == "unsorted":
        folds["A"]["train"] = folds["A"]["train"][::-1]
    elif change == "not_canonical":
        folds["A"]["train"][0] = "oasis_1"
    elif change == "different_cover":
        folds["B"]["train"] = folds["B"]["train"][1:]
    elif change == "same_calib":
        moved = folds["A"]["calib"][0]
        folds["B"]["train"] = sorted(set(folds["B"]["train"]+folds["B"]["calib"])-{moved})
        folds["B"]["calib"] = [moved]
    elif change == "extra_key":
        doc["extra"] = 1
    else:
        del folds["B"]
    with pytest.raises(InputValidationError, match=fragment):
        parse_full_folds(doc)


def test_export_script_never_overwrites(tmp_path):
    source = write_json(tmp_path/"folds.json", v1_doc())
    out = tmp_path/"out.json"
    assert EX.main(["--folds", str(source), "--out", str(out)]) == 0
    first = out.read_bytes()
    assert EX.main(["--folds", str(source), "--out", str(out), "--check"]) == 0
    with pytest.raises(FileExistsError):
        EX.main(["--folds", str(source), "--out", str(out)])
    assert out.read_bytes() == first and first == render_full_folds(full_folds_document(load_folds(source)))
    out.write_bytes(first+b"\n")
    assert EX.main(["--folds", str(source), "--out", str(out), "--check"]) == 1


# ---- v1 does not move ------------------------------------------------------------------------------------------------------------

def test_v1_pins_item_by_item():
    assert pins.PINNED_FOLDS_SHA256 == "98e9e676053d4e9386bc775c5d4cfc6be097afca9919c655ccb554d6236bf245"
    assert pins.TRAINING_SOURCE == "bayesreg_train_v1"
    assert pins.PINNED_COUNTS == (45, 8)
    assert pins.SYNTHETIC_MIN == 900
    assert pins.DEFAULT_FOLDS == CODE_ROOT/"splits"/"bayesreg_v1"/"folds.json"
    assert pins.DEFAULT_ALLOWLIST == CODE_ROOT/"configs"/"bayesreg"/"dev_subjects.json"
    assert pins.REAL_DATA_ROOTS == (Path("/mnt/d/data"), Path("/home/rmcfz/code/python_projects/seq_vp_3d/data"))


def test_v1_snapshot_paths():
    assert SNAPSHOT_PATHS == ("bayesreg", "jd_curl", "configs", "experiments/segnet", "splits/bayesreg_v1/folds.json",
                              "splits/bayesreg_v1/train_subjects.txt")
    assert SNAPSHOT_PROFILES["v1"] == SNAPSHOT_PATHS and SNAPSHOT_PATHS_V2[:len(SNAPSHOT_PATHS)] == SNAPSHOT_PATHS
    assert set(SNAPSHOT_PATHS_V2)-set(SNAPSHOT_PATHS) == {"experiments/unv2", "splits/bayesreg_v1/folds_full_v2.json"}
    assert not [p for p in SNAPSHOT_PATHS_V2 if "holdout" in p]


@pytest.mark.parametrize("argv,want", [
    (["train", "--fold", "A", "--seed", "0"], dict(folds=pins.DEFAULT_FOLDS, expect_counts="45,8", out=cli.DEFAULT_OUT)),
    (["calibrate", "--fold", "A", "--seed", "0"], dict(folds=pins.DEFAULT_FOLDS, expect_counts="45,8", out=cli.DEFAULT_OUT)),
    (["check-members"], dict(folds=pins.DEFAULT_FOLDS, expect_counts="45,8", out=cli.DEFAULT_OUT)),
    (["register"], dict(folds=pins.DEFAULT_FOLDS, expect_counts="45,8", out=cli.DEFAULT_OUT, registry=cli.DEFAULT_REGISTRY,
                        segmenter_id="unet35_v1_ens", prefix="unet35_v1", seeds="0,1")),
    (["infer"], dict(out=cli.DEFAULT_OUT, registry=cli.DEFAULT_REGISTRY, segmenter_id="unet35_v1_ens", all_dev=False)),
])
def test_v1_cli_defaults(argv, want):
    args = cli._apply_profile(cli.build_parser().parse_args(argv))
    assert args.profile == "v1" and not cli._full(args)
    for key, value in want.items():
        assert getattr(args, key) == value, key
    assert cli.DEFAULT_OUT == Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v1")
    assert cli.DEFAULT_REGISTRY == CODE_ROOT/"configs"/"bayesreg"/"segmenters.json"


def test_profile_v2_defaults_and_the_position_of_the_option():
    args = cli._apply_profile(cli.build_parser().parse_args(["register", "--profile", "v2_full"]))
    assert args.folds == pins.DEFAULT_FULL_FOLDS and args.expect_counts == "98,8" and args.segmenter_id == "unet35_v2_full"
    assert args.prefix == "unet35_v2_full" and args.registry == CODE_ROOT/"configs"/"bayesreg"/"segmenters_v2.json"
    assert args.out == Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v2_full")
    with pytest.raises(SystemExit, match="after the subcommand"):           # would otherwise silently mean v1
        cli.main(["--profile", "v2_full", "check-inputs", "--synthetic"])


def test_v1_registry_file_is_untouched_and_the_v2_file_holds_only_v2():
    assert hashlib.sha256((CODE_ROOT/"configs"/"bayesreg"/"segmenters.json").read_bytes()).hexdigest() == \
        "adc5cc5d53930f8f63e3c9f223eac233910d0dc753ed04aed3744508aa55e8e8"      # the A5P queue checks this file
    v2 = CODE_ROOT/"configs"/"bayesreg"/"segmenters_v2.json"
    if v2.exists():                                                             # committed after the review of a real chain
        registry = load_segmenter_registry(v2)
        assert sorted(registry.models) == ["unet35_v2_full_foldA", "unet35_v2_full_foldB"]
        assert {e["segmenter_id"] for e in registry.models.values()} == {"unet35_v2_full"}
        training = set((SPLITS/"train_subjects.txt").read_text().split())
        assert all(set(e["label_contact_subjects"]) == training for e in registry.models.values())


def test_formal_registries_are_a_fixed_list(tmp_path):
    assert structure_layer.FORMAL_REGISTRIES == (CODE_ROOT/"configs"/"bayesreg"/"segmenters.json",
                                                 CODE_ROOT/"configs"/"bayesreg"/"segmenters_v2.json",
                                                 CODE_ROOT/"configs"/"bayesreg"/"segmenters_nn935_v1.json",     # FX's one file more, runs/20261105-fx/plan.md
                                                 CODE_ROOT/"configs"/"bayesreg"/"segmenters_hs_v1.json",       # HS's one file more, runs/20261106-hs/plan.md; still a fixed list
                                                 CODE_ROOT/"configs"/"bayesreg"/"segmenters_r2_v1.json")       # R2's one file more, runs/20261110-r2/plan.md
    assert structure_layer.is_formal_registry(CODE_ROOT/"configs"/"bayesreg"/"segmenters_v2.json")
    other = write_json(tmp_path/"segmenters_v2.json", dict(schema="bayesreg.segmenters.v1", models={}))
    assert not structure_layer.is_formal_registry(other)
    with pytest.raises(ProvenanceRejected, match="not one of the repository registries"):
        structure_layer.open_registry(other, formal_registry=other)
    with pytest.raises(ProvenanceRejected, match="not one of the repository registries"):
        structure_layer.open_registry(other, allow_staging=True, formal_registry=other)


def test_a_profile_keeps_to_its_own_segmenter_and_registry(tmp_path):
    v2_registry = str(cli.PROFILES["v2_full"].registry)
    for argv, fragment in (
            (["infer", "--subjects", DEV[0], "--segmenter-id", "unet35_v2_full"], "goes with --profile v2_full"),
            (["infer", "--subjects", DEV[0], "--registry", v2_registry], "belongs to the other profile"),
            (["infer", "--profile", "v2_full", "--all-dev", "--segmenter-id", "unet35_v1_ens"], "goes with --profile v2_full"),
            (["infer", "--profile", "v2_full", "--all-dev", "--registry", str(cli.DEFAULT_REGISTRY)], "belongs to the other profile"),
            (["infer", "--profile", "v2_full", "--all-dev", "--dev-source", "stage_E11_native_v1"], "--dev-source is stage_L2R_val_v1"),
            (["infer", "--subjects", DEV[0], "--dev-source", "stage_T0_tune_v1"], "--dev-source is stage_L2R_val_v1"),
            (["register", "--profile", "v2_full", "--registry", str(cli.DEFAULT_REGISTRY)], "belongs to the other profile"),
            (["register", "--registry", v2_registry], "belongs to the other profile"),
            (["register", "--segmenter-id", "unet35_v2_full"], "goes with --profile v2_full")):
        with pytest.raises(SystemExit, match=fragment):                         # before the machine gate, before anything is read
            cli.main(argv)


def test_gate_only_reads_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "machine_gate", lambda device: dict(passed=True, device=device))

    def no_reads(*args, **kwargs):
        raise AssertionError("nothing may be read by --gate-only")
    monkeypatch.setattr(pins, "load_pinned_allowlist", no_reads)
    assert cli.main(["infer", "--profile", "v2_full", "--all-dev", "--gate-only", "--device", "cpu", "--out", str(tmp_path)]) == 0
    assert '"passed": true' in capsys.readouterr().out and not list(tmp_path.iterdir())


def test_v1_profile_refuses_a_registry_that_registers_v2(dry):
    work = Path(dry["work"])
    with pytest.raises(SystemExit, match="--profile v2_full is required"):      # the staged file, under any name
        cli.main(["infer", "--synthetic", "--subjects", DEV[0], "--registry", str(work/"segmenters.json"), "--allowlist",
                  str(work/"dev_subjects.json"), "--data-root", str(work/"data"), "--out", str(work/"models"), "--device", "cpu",
                  "--pred-out", str(work/"pred_v1_profile")])
    assert not (work/"pred_v1_profile").exists()


def test_v1_registration_rules_stay():
    def members(fold_seeds, contact):
        return [dict(fold=f, seed=s, member_sha256=hashlib.sha256(f"{f}{s}".encode()).hexdigest(), label_contact_subjects=contact,
                     label_contact_subjects_sha256=subject_list_sha256(contact)) for f, s in fold_seeds]
    a, b = sorted(A_TRAIN+A_CALIB), sorted(B_TRAIN+B_CALIB)
    one_each_disjoint = members([("A", 0)], a)+members([("B", 0)], b)
    with pytest.raises(InputValidationError, match="expected"):             # v1: every seed
        build_entries(one_each_disjoint)
    shared = sorted(TRAINING)
    both = members([("A", 0), ("A", 1), ("B", 0), ("B", 1)], shared)
    with pytest.raises(InputValidationError, match="cross-fitting"):        # v1: disjoint folds
        build_entries(both)


# ---- the pins of v2 --------------------------------------------------------------------------------------------------------------

CLEAN = dict(kind="git_worktree", clean=True, commit="0"*40, root=str(CODE_ROOT), imports_inside_root=True)


@pytest.fixture()
def pinned_v2(tmp_path, monkeypatch):
    """Both pins redirected to synthetic files (the real files are never opened here)."""
    root = tmp_path/"src"
    (root/"lists").mkdir(parents=True)
    items = []
    for name, ids, never in (("synthetic_dev", DEV, False), ("synthetic_t0", T0, True), (TRAIN_SOURCE, TRAINING, False)):
        path = root/"lists"/f"{name}.txt"
        path.write_text("".join(f"{s}\n" for s in ids))
        items.append(dict(name=name, dataset="OASIS", path=f"lists/{name}.txt", sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          never_for_training=never, subjects=list(ids)))
    union = []
    for item in items:
        union += [s for s in item["subjects"] if s not in union]
    allow = write_json(tmp_path/"dev_subjects.json", dict(schema="bayesreg.dev_subjects.v1", subjects={"OASIS": union}, sources=items))
    splits = tmp_path/"splits"
    splits.mkdir()
    v1 = write_json(splits/"folds.json", v1_doc())
    v2 = splits/"folds_full_v2.json"
    v2.write_bytes(render_full_folds(full_folds_document(load_folds(v1))))
    monkeypatch.setattr(pins, "PINNED_FOLDS_SHA256", hashlib.sha256(v1.read_bytes()).hexdigest())
    monkeypatch.setattr(pins, "PINNED_FULL_FOLDS_SHA256", hashlib.sha256(v2.read_bytes()).hexdigest())
    monkeypatch.setattr(pins, "DEFAULT_ALLOWLIST", allow)
    monkeypatch.setattr(pins, "TRAINING_SOURCE", TRAIN_SOURCE)
    monkeypatch.setattr(pins, "PINNED_COUNTS", (2, 1))
    monkeypatch.setattr(pins, "PINNED_FULL_COUNTS", (5, 1))
    return root, allow, v1, v2


def test_full_pins_pass_on_the_pinned_files(pinned_v2):
    root, allow, _, v2 = pinned_v2
    split, allowlist, record = load_pinned_full_inputs(v2, allow, training_source=TRAIN_SOURCE, expect_counts=(5, 1),
                                                       source_root=root, code=CLEAN)
    assert record["mode"] == "pinned_full" and record["folds_sha256"] == pins.PINNED_FULL_FOLDS_SHA256
    assert record["derivation"]["source_sha256"] == pins.PINNED_FOLDS_SHA256
    assert LabelGuard(split, allowlist, training_source=TRAIN_SOURCE).split is split


@pytest.mark.parametrize("case", ["file_changed", "v1_changed", "source_name", "counts", "dirty", "allowlist_path", "v1_file_given",
                                  "not_derived"])
def test_full_pins_refuse_departures(pinned_v2, tmp_path, monkeypatch, case):
    root, allow, v1, v2 = pinned_v2
    kwargs = dict(training_source=TRAIN_SOURCE, expect_counts=(5, 1), source_root=root, code=CLEAN)
    path = v2
    if case == "file_changed":
        v2.write_bytes(v2.read_bytes()+b"\n")
    elif case == "v1_changed":
        v1.write_text(v1.read_text()+"\n")
    elif case == "source_name":
        kwargs["training_source"] = "synthetic_dev"
    elif case == "counts":
        kwargs["expect_counts"] = (98, 8)
    elif case == "dirty":
        kwargs["code"] = dict(CLEAN, clean=False, n_dirty=1, dirty_paths=["?? x"])
    elif case == "allowlist_path":
        allow = write_json(tmp_path/"copy.json", json.loads(allow.read_text()))
    elif case == "v1_file_given":
        path = v1
    else:                                                                   # pinned and well-formed, but not the rule's output
        doc = json.loads(v2.read_text())
        a, b = doc["folds"]["A"], doc["folds"]["B"]
        x, y = a["calib"][0], a["train"][0]
        a["calib"], a["train"] = [y], sorted(set(a["train"])-{y} | {x})
        b["train"] = sorted(set(TRAINING)-set(b["calib"]))
        v2.write_bytes(render_full_folds(doc))
        monkeypatch.setattr(pins, "PINNED_FULL_FOLDS_SHA256", hashlib.sha256(v2.read_bytes()).hexdigest())
    with pytest.raises((PinnedInputRefused, CodeNotClean)):
        load_pinned_full_inputs(path, allow, **kwargs)


def test_v1_pins_refuse_the_full_file(pinned_v2):
    root, allow, _, v2 = pinned_v2
    with pytest.raises(PinnedInputRefused, match="not the pinned"):
        load_pinned_inputs(v2, allow, training_source=TRAIN_SOURCE, expect_counts=(2, 1), source_root=root, code=CLEAN)


def test_real_full_pins_are_the_committed_split():
    assert pins.PINNED_FULL_COUNTS == (98, 8) and pins.DEFAULT_FULL_FOLDS == FOLDS_V2
    with pytest.raises(PinnedInputRefused, match="not the pinned"):         # refused before parsing
        load_pinned_full_inputs(Path(__file__), ALLOWLIST, code=CLEAN)
    with pytest.raises(PinnedInputRefused, match="counts"):
        load_pinned_full_inputs(FOLDS_V2, ALLOWLIST, expect_counts=(45, 8), code=CLEAN)


def test_snapshot_profile_archives_the_v2_files(tmp_path):
    repo = tmp_path/"repo"
    for rel in SNAPSHOT_PATHS_V2:
        target = repo/rel if "." in Path(rel).name else repo/rel/"x.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x\n")
    (repo/"splits"/"bayesreg_v1"/"holdout_B_subjects.txt").write_text("not archived\n")
    run = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(run+["add", "-A"], check=True)
    subprocess.run(run+["commit", "-q", "-m", "init"], check=True)
    record = make_snapshot(repo, tmp_path/"snap", paths=SNAPSHOT_PROFILES["v2_full"])
    assert "splits/bayesreg_v1/folds_full_v2.json" in record["files"] and "experiments/unv2/x.txt" in record["files"]
    assert "splits/bayesreg_v1/holdout_B_subjects.txt" not in record["files"]


# ---- registration ----------------------------------------------------------------------------------------------------------------

def _members(fold_seeds, contact=None):
    contact = sorted(TRAINING) if contact is None else contact
    return [dict(fold=f, seed=s, member_sha256=hashlib.sha256(f"{f}{s}".encode()).hexdigest(), label_contact_subjects=contact,
                 label_contact_subjects_sha256=subject_list_sha256(contact)) for f, s in fold_seeds]


@pytest.mark.parametrize("fold_seeds,counts", [([("A", 0), ("B", 0)], (1, 1)), ([("A", 0), ("A", 1), ("B", 0)], (2, 1)),
                                               ([("A", 0), ("A", 1), ("B", 0), ("B", 1)], (2, 2))])
def test_registration_with_at_least_one_member_per_fold(tmp_path, fold_seeds, counts):
    entries = build_entries(_members(fold_seeds), segmenter_id=SEGMENTER_ID_V2, prefix=MODEL_PREFIX_V2, full_training_set=True)
    assert sorted(entries) == ["unet35_v2_full_foldA", "unet35_v2_full_foldB"]
    assert (len(entries["unet35_v2_full_foldA"]["member_model_sha256"]), len(entries["unet35_v2_full_foldB"]["member_model_sha256"])) == counts
    registry = load_segmenter_registry(write_json(tmp_path/"segmenters.json", dict(schema="bayesreg.segmenters.v1", models=entries)))
    assert registry.models["unet35_v2_full_foldA"]["segmenter_id"] == "unet35_v2_full"
    assert registry.models["unet35_v2_full_foldB"]["fold"] == 1


@pytest.mark.parametrize("members,fragment", [
    (_members([("A", 0), ("A", 1)]), "at least one member per fold"),
    (_members([("B", 0)]), "at least one member per fold"),
    (_members([("A", 0)])+_members([("B", 2)]), "at least one member per fold"),
    (_members([("A", 0)])+_members([("B", 0)], sorted(TRAINING)[1:]), "contacts differ"),
])
def test_registration_refuses(members, fragment):
    with pytest.raises(InputValidationError, match=fragment):
        build_entries(members, segmenter_id=SEGMENTER_ID_V2, prefix=MODEL_PREFIX_V2, full_training_set=True)


@pytest.fixture(scope="module")
def dry(tmp_path_factory):
    return cli.dry_run_v2(tmp_path_factory.mktemp("dry_v2"), device="cpu", config=TINY, log=lambda *_: None)


def _dry_inputs(run):
    work = Path(run["work"])
    return work, load_full_folds(work/"folds_full_v2.json"), load_dev_allowlist(work/"dev_subjects.json")


def test_dry_run_passes_all_checks(dry):
    assert dry["ok"], json.dumps(dry, indent=1, default=str)
    assert dry["counts"] == {f: dict(train=5, calib=1) for f in "AB"} and dry["pair_check"] == "accepted"
    assert dry["contact"] == sorted(TRAINING) and dry["registry"] == ["unet35_v2_full_foldA", "unet35_v2_full_foldB"]
    assert dry["members"]["A0"]["calib"] == A_CALIB and not set(dry["members"]["A0"]["train"]) & set(A_CALIB)
    assert dry["members"]["B0"]["calib"] == B_CALIB and set(A_CALIB) <= set(dry["members"]["B0"]["train"])
    assert all(v.startswith("refused") for v in dry["refusals"].values()), dry["refusals"]
    assert sorted(dry["members"]) == ["A0", "A1", "B0"]
    for subject in DEV:
        assert dry["inferred"][subject]["policy"] == "all_folds_mean" and dry["inferred"][subject]["n_members"] == 3


def test_three_members_are_registered_two_and_one(dry):
    registry = load_segmenter_registry(Path(dry["work"])/"segmenters.json")
    assert len(registry.models["unet35_v2_full_foldA"]["member_model_sha256"]) == 2
    assert len(registry.models["unet35_v2_full_foldB"]["member_model_sha256"]) == 1
    ensemble = json.loads((Path(dry["work"])/"models"/"ensemble.json").read_text())
    assert sorted((m["fold"], m["seed"]) for m in ensemble["members"]) == [("A", 0), ("A", 1), ("B", 0)]
    assert ensemble["member_check"]["passed"] is True


def test_registration_is_append_only(dry, tmp_path):
    work, split, allowlist = _dry_inputs(dry)
    document = json.loads((work/"segmenters.json").read_text())
    document["models"]["unet35_v2_full_foldB"]["member_model_sha256"] = ["0"*64]
    path = write_json(tmp_path/"segmenters.json", document)
    before = path.read_bytes()
    with pytest.raises(InputValidationError, match="append-only"):
        register(work/"models", path, split=split, allowlist=allowlist, training_source=TRAIN_SOURCE, log=lambda *_: None,
                 segmenter_id=SEGMENTER_ID_V2, prefix=MODEL_PREFIX_V2, full_training_set=True)
    assert path.read_bytes() == before


def test_three_member_ensemble_is_the_plain_mean_over_the_members(dry):
    """Independent recomputation from the member files: each member counts once, so fold A weighs 2/3 here."""
    work = Path(dry["work"])
    subject = DEV[0]
    registry = load_segmenter_registry(work/"segmenters.json")
    stored, prov = read_probabilities(work/"models"/"pred"/subject, "probs36")
    assert len(prov["members"]) == 3 and sorted(prov["model_ids"]) == ["unet35_v2_full_foldA", "unet35_v2_full_foldB"]
    values, _ = read_image(OasisFiles(work/"data"), subject)
    available = find_member_files(work/"models")
    total, n = 0., 0
    for entry in registry.models.values():
        for sha in entry["member_model_sha256"]:
            model, blob = load_member(available[sha], sha, "cpu", entry=entry)
            image, _ = normalize_image(values, percentile=blob["percentile"])
            total, n = total+torch.softmax(predict_logits(model, image, device="cpu", amp=False)/blob["temperature"], dim=1)[0], n+1
    assert n == 3
    assert np.max(np.abs(stored-(total/n).numpy())) < 2e-3                  # float16 storage


def test_one_member_per_fold_goes_through(tmp_path):
    run = cli.dry_run_v2(tmp_path/"w", device="cpu", config=TINY, log=lambda *_: None, members=(("A", 0), ("B", 0)))
    assert run["ok"], json.dumps(run, indent=1, default=str)
    assert all(v["n_members"] == 2 for v in run["inferred"].values())
    registry = load_segmenter_registry(tmp_path/"w"/"segmenters.json")
    assert [len(e["member_model_sha256"]) for _, e in sorted(registry.models.items())] == [1, 1]


def test_members_of_one_fold_are_not_registrable(tmp_path):
    with pytest.raises(InputValidationError, match="at least one member per fold"):
        cli.dry_run_v2(tmp_path/"w", device="cpu", config=TINY, log=lambda *_: None, members=(("A", 0),))
    assert not (tmp_path/"w"/"segmenters.json").exists()


def test_the_flip_is_gone():
    from bayesreg.seg.unet import groups, infer
    import inspect
    assert not {"tta", "lateral_check"} & set(inspect.signature(segment_subject).parameters)
    for module, names in ((infer, ("flip_input", "flip_logits_back", "left_right_axis", "check_lateral_order", "TTA_MODES")),
                          (groups, ("load_left_right_swap",))):
        assert not [n for n in names if hasattr(module, n)]
    with pytest.raises(SystemExit):
        cli.main(["infer", "--tta", "flip", "--synthetic"])
    assert _bash("status", "--flip").returncode == 2
    assert "flip" not in CHAIN.read_text() and "flip" not in EVAL_SEG.read_text()


def test_member_records_write_the_label_contact(dry):
    work = Path(dry["work"])
    for fold, seed, calib in (("A", 0, A_CALIB), ("B", 0, B_CALIB), ("A", 1, A_CALIB)):
        member = work/"models"/f"fold{fold}_s{seed}"
        record = json.loads((member/"train_record.json").read_text())
        assert record["label_contact_subjects"] == sorted(TRAINING) and record["calib_subjects"] == calib
        assert sorted(r["subject"] for r in record["label_guard"]["label_reads"]) == sorted(TRAINING)
        assert record["label_guard"]["folds_layout"] == "full_v2" and record["config"]["max_iters"] == TINY["max_iters"]
        assert record["label_guard"]["derived_from"]["sha256"] == parse_folds(v1_doc()).sha256
        calibration = json.loads((member/"calibration.json").read_text())
        assert [r["subject"] for r in calibration["label_guard"]["label_reads"]] == calib


def test_member_check_on_the_full_split(dry, tmp_path):
    work, split, allowlist = _dry_inputs(dry)
    members, check = verify_members(work/"models", split=split, allowlist=allowlist, training_source=TRAIN_SOURCE)
    assert check["passed"] and {(m["fold"], m["seed"]) for m in members} == {("A", 0), ("B", 0), ("A", 1)}
    assert all(m["n_contact"] == 6 for m in check["members"])
    wrong = write_json(tmp_path/"folds.json", v1_doc())
    from bayesreg.seg.unet.registry import MemberCheckFailed
    with pytest.raises(MemberCheckFailed):                                  # the v1 folds do not describe these members
        verify_members(work/"models", split=load_folds(wrong), allowlist=allowlist, training_source=TRAIN_SOURCE)


def test_v2_infer_subject_rule(dry):
    _, _, allowlist = _dry_inputs(dry)
    assert cli.v2_infer_subjects([], allowlist, all_dev=True, source="synthetic_dev") == DEV
    assert cli.v2_infer_subjects([DEV[1]], allowlist, source="synthetic_dev") == [DEV[1]]
    for subjects, source in (([T0[0]], "synthetic_dev"), ([TRAINING[0]], "synthetic_dev"), ([DEV[0]], "synthetic_t0"), ([DEV[0]], "nope")):
        with pytest.raises(SystemExit):
            cli.v2_infer_subjects(subjects, allowlist, source=source)
    with pytest.raises(SystemExit, match="no subjects"):
        cli.v2_infer_subjects([], allowlist, source="synthetic_dev")


def test_infer_all_dev_goes_with_the_profile_and_the_chain_commands_refuse_the_others(tmp_path):
    with pytest.raises(SystemExit, match="--all-dev"):
        cli.main(["infer", "--all-dev", "--synthetic"])
    for command in (["evaluate-calibration"], ["check-tissue4", "--report", str(tmp_path/"r.json")]):
        with pytest.raises(SystemExit, match="not for --profile"):
            cli.main([*command, "--profile", "v2_full", "--synthetic", "--folds", "x", "--allowlist", "y"])


def test_guard_with_the_full_split_refuses_before_opening_a_file(tmp_path):
    split = parse_full_folds(v2_doc())
    allowlist = load_dev_allowlist(write_json(tmp_path/"a.json", allowlist_document(
        [("synthetic_dev", DEV, False), ("synthetic_t0", T0, True), (TRAIN_SOURCE, TRAINING, False)])))
    guard = LabelGuard(split, allowlist, training_source=TRAIN_SOURCE)
    files = types.SimpleNamespace(label_path=lambda *a: (_ for _ in ()).throw(AssertionError("a label file was opened")))
    for subject in (DEV[0], T0[0], oasis(910), "bad"):
        with pytest.raises(LabelAccessRefused):
            read_labels(files, subject, guard)
    assert guard.reads == []


# ---- the chain ------------------------------------------------------------------------------------------------------------------

def _bash(*args, env=None, timeout=300, path_prefix=None):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    environment.update(env or {})
    if path_prefix:
        environment["PATH"] = f"{path_prefix}:{environment['PATH']}"
    return subprocess.run(["bash", str(CHAIN), *args], capture_output=True, text=True, timeout=timeout, env=environment)


def test_chain_syntax_usage_and_refusals(tmp_path):
    assert subprocess.run(["bash", "-n", str(CHAIN)], capture_output=True).returncode == 0
    usage = _bash()
    assert usage.returncode == 2 and "launch" in usage.stdout and "dry-run" in usage.stdout
    outside = _bash("chain", "--out", str(tmp_path))
    assert outside.returncode == 1 and "chain must run from the snapshot" in (tmp_path/"chain.log").read_text()
    naive = _bash("status", "--out", str(tmp_path), "--no-start-after", "2026-11-04T06:00:00")
    assert naive.returncode == 1 and "explicit offset" in naive.stdout
    garbled = _bash("status", "--out", str(tmp_path), "--no-start-after", "someday+01:00")
    assert garbled.returncode == 1 and "not a time" in garbled.stdout
    assert _bash("status", "--bad").returncode == 2
    assert _bash("dry-run").returncode == 2
    text = CHAIN.read_text()
    for needle in ("tag-unv2-chain", "MIN_FREE_MIB:-14336", "status --porcelain", "--profile v2_full", "snapshot make", "--require-snapshot",
                   "check-members", "--all-dev", "experiments/unv2/eval_seg.py", "setsid nohup", "MemoryMax=12G", "grep -v '^tag-unv2-'"):
        assert needle in text, needle
    assert "seq-vp-" in text and "--unit seq-vp" not in text


def _shim(tmp_path, *, counter=1000000000, step=1000):
    """A `date` whose +%s advances by `step` seconds per call; everything else goes to the real date."""
    bin_dir = tmp_path/"bin"
    bin_dir.mkdir()
    state = tmp_path/"clock"
    state.write_text(str(counter))
    real = shutil.which("date")
    script = bin_dir/"date"
    script.write_text(f'#!/bin/bash\nif [ "$1" = "+%s" ]; then n=$(( $(cat {state}) + {step} )); echo $n > {state}; echo $n; else exec {real} "$@"; fi\n')
    script.chmod(0o755)
    return bin_dir


def _iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stub(tmp_path, *, fail=None):
    """The Python command line replaced by one that records its arguments; `fail`: a word that makes it fail."""
    log = tmp_path/"calls.txt"
    script = tmp_path/"stub.sh"
    condition = f'case "$*" in *"{fail}"*) exit 7;; esac' if fail else ":"
    script.write_text(f'#!/bin/bash\necho "$*" >> {log}\n{condition}\nexit 0\n')
    return script, log


def _calls(log):
    return [line.split() for line in log.read_text().splitlines()] if log.exists() else []


def _order(log):
    return [f"{c[1]}_{c[c.index('--fold')+1]}{c[c.index('--seed')+1]}" for c in _calls(log) if c[1] in ("train", "calibrate")]


FULL_ORDER = ["train_A0", "calibrate_A0", "train_B0", "calibrate_B0", "train_A1", "calibrate_A1", "train_B1", "calibrate_B1"]


def test_chain_order_and_steps_without_a_deadline(tmp_path):
    stub, log = _stub(tmp_path)
    result = _bash("dry-run", "--work", str(tmp_path/"w"), "--stub-cli", str(stub))
    assert result.returncode == 0, result.stdout
    assert _order(log) == FULL_ORDER
    words = [c[1] for c in _calls(log)]
    assert words[0] == "check-inputs" and "--require-snapshot" in _calls(log)[0]
    assert words[-4:] == ["check-members", "register", "infer", "eval_seg"]
    assert all("--profile" in c and "v2_full" in c for c in _calls(log) if c[1] != "eval_seg")
    infers = [c for c in _calls(log) if c[1] == "infer"]
    assert len(infers) == 1 and "--all-dev" in infers[0]
    evaluation = next(c for c in _calls(log) if c[1] == "eval_seg")
    assert evaluation[evaluation.index("--v2")+1].endswith("/pred")
    assert "dry-run finished" in result.stdout


@pytest.mark.parametrize("deadline_steps,started,registered", [(.5, [], False), (1.5, ["A0"], False), (2.5, ["A0", "B0"], True),
                                                               (3.5, ["A0", "B0", "A1"], True), (9.5, ["A0", "B0", "A1", "B1"], True)])
def test_chain_deadline(tmp_path, deadline_steps, started, registered):
    """Each look at the clock advances it by 1000 s, one look per member about to start; the member at the deadline does not start."""
    stub, log = _stub(tmp_path)
    start = 1000000000
    deadline = _iso(start+int(deadline_steps*1000))
    result = _bash("dry-run", "--work", str(tmp_path/"w"), "--stub-cli", str(stub), "--no-start-after", deadline,
                   path_prefix=_shim(tmp_path, counter=start))
    assert _order(log) == [x for m in started for x in (f"train_{m}", f"calibrate_{m}")], result.stdout
    words = [c[1] for c in _calls(log)]
    assert ("register" in words) is registered and ("infer" in words) is registered
    if registered:
        assert result.returncode == 0, result.stdout
    else:
        assert result.returncode == 1 and "not registrable" in result.stdout
    for member in {"A0", "B0", "A1", "B1"}-set(started):
        assert f"{member} is not started" in result.stdout


def test_chain_failure_stops_and_is_not_rerun(tmp_path):
    stub, log = _stub(tmp_path, fail="--fold B --seed 0")
    work = tmp_path/"w"
    first = _bash("dry-run", "--work", str(work), "--stub-cli", str(stub))
    assert first.returncode == 1 and "train_B0 failed" in first.stdout
    assert _order(log) == ["train_A0", "calibrate_A0", "train_B0"]
    state = work/"out"/"chain_state"
    assert (state/"train_B0.started").exists() and not (state/"train_B0.done").exists()
    stub2, log2 = _stub(_subdir(tmp_path, "second"))
    second = _bash("dry-run", "--work", str(work), "--stub-cli", str(stub2))
    assert second.returncode == 1 and "did not finish" in second.stdout
    assert _order(log2) == []                                               # A0 skipped, B0 stops the chain; nothing is rerun


def _subdir(path, name):
    (path/name).mkdir()
    return path/name


def test_chain_no_new_member_after_the_registration(tmp_path):
    stub, log = _stub(tmp_path)
    work = tmp_path/"w"
    assert _bash("dry-run", "--work", str(work), "--stub-cli", str(stub), "--no-start-after",
                 _iso(1000000000+2500), path_prefix=_shim(tmp_path)).returncode == 0
    assert _order(log) == FULL_ORDER[:4]
    stub2, log2 = _stub(_subdir(tmp_path, "second"))
    again = _bash("dry-run", "--work", str(work), "--stub-cli", str(stub2))
    assert again.returncode == 0 and "A1 is not started" in again.stdout and "B1 is not started" in again.stdout
    assert _order(log2) == [] and [c[1] for c in _calls(log2)] == ["check-inputs"]


def _machine(tmp_path, *, units="", free="20000", util="3", list_fails=False):
    """Fake systemctl, systemd-run and nvidia-smi for the waiter: nothing starts, the calls of systemd-run are recorded."""
    bin_dir = tmp_path/"machine"
    bin_dir.mkdir()
    started = tmp_path/"systemd_run.txt"
    (bin_dir/"systemctl").write_text('#!/bin/bash\ncase "$*" in\n  *is-active*) exit 1;;\n  *list-units*) '
                                     + ("exit 1" if list_fails else f"printf '%b' '{units}'; exit 0")+';;\nesac\nexit 0\n')
    (bin_dir/"systemd-run").write_text(f'#!/bin/bash\necho "$*" >> {started}\nexit 0\n')
    (bin_dir/"nvidia-smi").write_text(f'#!/bin/bash\ncase "$*" in\n  *name,memory*) echo fake;;\n  *memory.free*) echo "{free}";;\n'
                                      f'  *utilization.gpu*) echo "{util}";;\nesac\n')
    for script in bin_dir.iterdir():
        script.chmod(0o755)
    return bin_dir, started


FUTURE, PAST = "2099-01-01T00:00:00Z", "2000-01-01T00:00:00Z"


def _waiter(tmp_path, deadline, *, env=None, **machine):
    bin_dir, started = _machine(tmp_path, **machine)
    out = tmp_path/"out"
    result = _bash("wait-and-start", "--out", str(out), "--no-start-after", deadline, path_prefix=bin_dir,
                   env=dict(dict(UNV2_SMI=str(bin_dir/"nvidia-smi"), MIN_AVAIL_GIB="0"), **(env or {})))
    return result, started, out


@pytest.mark.parametrize("units", ["", "tag-unv2-other.service loaded active running x\\n"])
def test_waiter_starts_the_unit_from_the_snapshot_with_the_deadline(tmp_path, units):
    result, started, out = _waiter(tmp_path, FUTURE, units=units)
    assert result.returncode == 0, result.stdout
    line = started.read_text()
    assert "--user --unit tag-unv2-chain --collect" in line and f"WorkingDirectory={out}/snapshot" in line and "MemoryMax=12G" in line
    assert line.rstrip().endswith(f"{out}/snapshot/experiments/unv2/chain.sh chain --out {out} --no-start-after {FUTURE}")
    assert "CUDA_VISIBLE_DEVICES" not in line and "unit tag-unv2-chain started" in result.stdout


@pytest.mark.parametrize("machine,reason", [
    (dict(units="tag-d0a5-queue.service loaded active running x\\n"), "units active: tag-d0a5-queue"),
    (dict(units="seq-vp-chain.service loaded active running x\\n"), "units active: seq-vp-chain"),
    (dict(free="100"), "GPU free 100 MiB"),
    (dict(free=""), "no free-memory reading"),
    (dict(util="90"), "GPU utilization 90%"),
    (dict(util=""), "no utilization reading"),
    (dict(list_fails=True), "systemctl failed"),
])
def test_waiter_does_not_start_while_busy_and_gives_up_at_the_deadline(tmp_path, machine, reason):
    result, started, out = _waiter(tmp_path, PAST, **machine)
    assert result.returncode == 1 and reason in result.stdout and "passed while waiting" in result.stdout and "nothing started" in result.stdout
    assert not started.exists()
    assert "passed while waiting; nothing started" in (out/"waiter_gave_up").read_text()


def test_waiter_refuses_after_the_deadline_and_below_the_disk_floor(tmp_path):
    late, started, out = _waiter(_subdir(tmp_path, "late"), PAST)
    assert late.returncode == 1 and "has passed; nothing started" in late.stdout and not started.exists()
    assert "has passed" in (out/"waiter_gave_up").read_text()
    full, started, out = _waiter(_subdir(tmp_path, "full"), FUTURE, env=dict(HOME_FLOOR_GB="100000000"))
    assert full.returncode == 1 and "/home free" in full.stdout and not started.exists()
    assert "/home free" in (out/"waiter_gave_up").read_text()
    shown = _bash("status", "--out", str(out))
    assert "WAITER GAVE UP" in shown.stdout and "/home free" in shown.stdout


def test_a_new_waiter_removes_an_old_marker(tmp_path):
    (tmp_path/"out").mkdir()
    (tmp_path/"out"/"waiter_gave_up").write_text("old\n")
    result, started, out = _waiter(tmp_path, FUTURE)
    assert result.returncode == 0 and started.exists() and not (out/"waiter_gave_up").exists()


def test_launch_refuses_the_test_hook(tmp_path):
    result = _bash("launch", "--out", str(tmp_path/"out"), "--no-start", env=dict(UNV2_SMI="/x"))
    assert result.returncode == 1 and "UNV2_SMI is set" in result.stdout and not (tmp_path/"out").exists()


def test_chain_probes_the_inference_gate_before_its_step():
    text = CHAIN.read_text()
    probe, step = text.index("--gate-only"), text.index("step infer ")
    assert probe < step and "no marker left" in text


@pytest.fixture(scope="module")
def real_chain(tmp_path_factory):
    """One whole synthetic chain with the real command line, tiny members on the CPU, stopped by its deadline after A0, B0, A1: the shape
    expected of the real chain tonight (about 3 members)."""
    base = tmp_path_factory.mktemp("chain")
    work = base/"w"
    result = _bash("dry-run", "--work", str(work), "--no-start-after", _iso(1000000000+3500), timeout=590, path_prefix=_shim(base))
    return work, result


def test_real_chain_with_three_members_runs_through(real_chain):
    work, result = real_chain
    assert result.returncode == 0, result.stdout+result.stderr
    assert "B1 is not started" in result.stdout and "calibrated members: fold A 2, fold B 1" in result.stdout
    out = work/"out"
    for member in ("foldA_s0", "foldB_s0", "foldA_s1"):
        assert (out/member/"member.pt").is_file()
    assert not (out/"foldB_s1").exists()
    registry = load_segmenter_registry(out/"segmenters.json")
    assert sorted(registry.models) == ["unet35_v2_full_foldA", "unet35_v2_full_foldB"]
    assert [len(e["member_model_sha256"]) for _, e in sorted(registry.models.items())] == [2, 1]
    assert sorted(p.name for p in (out/"pred").iterdir()) == DEV
    prov = json.loads((out/"pred"/DEV[0]/"provenance.json").read_text())
    assert prov["segmenter_id"] == "unet35_v2_full" and prov["version"]["segmenter"] == "unet35_v2_full"
    assert len(prov["members"]) == 3 and prov["policy"] == "all_folds_mean"
    assert sorted((m["fold"], m["seed"]) for m in prov["members"]) == [("A", 0), ("A", 1), ("B", 0)]


def test_the_preregistered_decision_is_in_the_log_and_in_status(real_chain):
    work, result = real_chain
    assert result.returncode == 0
    document = json.loads((work/"out"/"eval_seg"/"eval_seg.json").read_text())
    word = str(document["decision"]["run_a5pv2"]).lower()
    assert f"decision.run_a5pv2 = {word}" in result.stdout
    assert f"decision.run_a5pv2 = {word}" in (work/"out"/"decision_run_a5pv2.txt").read_text()
    shown = _bash("status", "--out", str(work/"out"))
    assert f"PREREGISTERED RULE:" in shown.stdout and f"decision.run_a5pv2 = {word}" in shown.stdout


@pytest.fixture(scope="module")
def real_chain_two_members(tmp_path_factory):
    """The same chain with the real command line, stopped by its deadline after A0, B0: the smallest ensemble the chain registers. Each look at
    the clock advances it by 1000 s, one look per member about to start: A0 at +1000, B0 at +2000, A1 at +3000 is past +2500 and does not start."""
    base = tmp_path_factory.mktemp("chain_two")
    work = base/"w"
    result = _bash("dry-run", "--work", str(work), "--no-start-after", _iso(1000000000+2500), timeout=590, path_prefix=_shim(base))
    return work, result


def test_real_chain_with_two_members_runs_through_the_evaluation(real_chain_two_members):
    """Review of the UNV2 fix round, 次要 5: two members from the first step to eval_seg and the preregistered decision."""
    work, result = real_chain_two_members
    assert result.returncode == 0, result.stdout+result.stderr
    assert "calibrated members: fold A 1, fold B 1" in result.stdout
    assert "A1 is not started" in result.stdout and "B1 is not started" in result.stdout
    out = work/"out"
    assert (out/"foldA_s0"/"member.pt").is_file() and (out/"foldB_s0"/"member.pt").is_file()
    assert not (out/"foldA_s1").exists() and not (out/"foldB_s1").exists()
    state = out/"chain_state"
    assert not (state/"train_A1.started").exists() and not (state/"train_B1.started").exists()
    assert (state/"register.done").is_file() and (state/"infer.done").is_file() and (state/"eval_seg.done").is_file()
    registry = load_segmenter_registry(out/"segmenters.json")
    assert sorted(registry.models) == ["unet35_v2_full_foldA", "unet35_v2_full_foldB"]
    assert [len(e["member_model_sha256"]) for _, e in sorted(registry.models.items())] == [1, 1]
    assert sorted(p.name for p in (out/"pred").iterdir()) == DEV
    for subject in DEV:
        prov = json.loads((out/"pred"/subject/"provenance.json").read_text())
        assert prov["segmenter_id"] == "unet35_v2_full" and prov["policy"] == "all_folds_mean" and len(prov["members"]) == 2
        assert sorted((m["fold"], m["seed"]) for m in prov["members"]) == [("A", 0), ("B", 0)]
    document = json.loads((out/"eval_seg"/"eval_seg.json").read_text())
    assert document["n_members"]["v2"] == [2] and document["arms"] == ["v1", "v2"] and document["subjects"] == DEV
    assert all(document["predictions"][s]["v2"]["n_members"] == 2 for s in DEV)
    assert "v2 2" in (out/"eval_seg"/"summary.md").read_text()
    word = str(document["decision"]["run_a5pv2"]).lower()
    assert f"decision.run_a5pv2 = {word}" in result.stdout and f"decision.run_a5pv2 = {word}" in (out/"decision_run_a5pv2.txt").read_text()


@pytest.mark.parametrize("how", ["formal", "staging"])
def test_structure_layer_accepts_the_three_member_prediction(real_chain, tmp_path, monkeypatch, how):
    """What a later arm reads: every check of the structure layer on the chain's outputs, as the registered file and as the staged one."""
    work, first = real_chain
    assert first.returncode == 0
    out = work/"out"
    monkeypatch.setattr(structure_layer, "FORMAL_REGISTRIES", (out/"segmenters.json",))      # stands in for the repository file
    allowlist = load_dev_allowlist(work/"dev_subjects.json")
    pair = load_pair(write_pair(tmp_path/"pair", work/"data", DEV[0], DEV[1]), allowlist=allowlist)
    choice = dict(formal_registry=out/"segmenters.json") if how == "formal" else dict(allow_staging=True)
    layer = StructureLayer(pair, StructureSource(root=out/"pred", registry_path=out/"segmenters.json", allowlist_path=work/"dev_subjects.json",
                                                 models_root=out, image_root=work/"data", segmenter_id="unet35_v2_full", **choice))
    assert layer.registry_record["status"] == (structure_layer.REGISTERED if how == "formal" else structure_layer.STAGING)
    for side in ("fixed", "moving"):
        groups, _, record = layer.load(side)
        assert groups.shape[0] == 14 and record["models"]["policy"] == "all_folds_mean" and len(record["models"]["members"]) == 3


def test_real_chain_is_idempotent(real_chain):
    work, first = real_chain
    assert first.returncode == 0
    before = {p: p.stat().st_mtime_ns for p in (work/"out").rglob("*") if p.is_file() and "chain_state" not in p.parts}
    again = _bash("dry-run", "--work", str(work), timeout=120)
    assert again.returncode == 0 and "skip train_A0 (done)" in again.stdout and "skip eval_seg (done)" in again.stdout
    assert "registered already: B1 is not started" in again.stdout
    assert {p: p.stat().st_mtime_ns for p in before} == before


def test_eval_seg_outputs_of_the_real_chain(real_chain):
    work, first = real_chain
    assert first.returncode == 0
    directory = work/"out"/"eval_seg"
    document = json.loads((directory/"eval_seg.json").read_text())
    assert document["arms"] == ["v1", "v2"] and document["subjects"] == DEV
    assert set(document["paired"]) == {"v2_minus_v1"}
    assert set(document["paired"]["v2_minus_v1"]) == {"seg35", "seg4", "small_six", "big_six", "other_23"}
    assert document["paired"]["v2_minus_v1"]["seg35"]["n"] == 2
    assert document["decision"]["mean_v2"] == document["means"]["v2"]["seg35"]
    assert document["decision"]["run_a5pv2"] is (document["means"]["v1"]["seg35"]-document["means"]["v2"]["seg35"] <= .002)
    assert set(document["truth_sha256"]) == set(DEV) and document["predictions"][DEV[0]]["v2"]["n_members"] == 3
    assert document["n_members"]["v2"] == [3]
    assert len((directory/"per_label.csv").read_text().splitlines()) == 36
    assert len((directory/"per_subject.csv").read_text().splitlines()) == 1+2*2
    text = (directory/"summary.md").read_text()
    assert "seg4" in text and "Dice35" in text and "开发集" in text and "v2 3" in text and "集成规模" in text
    with pytest.raises(FileExistsError):                                    # never overwritten
        EV.evaluate(v2_root=work/"out"/"pred", v1_root=work/"pred_v1", out_dir=directory, data_root=work/"data",
                    allowlist_path=work/"dev_subjects.json", folds_path=work/"folds.json", dev_source="synthetic_dev")


# ---- eval_seg --------------------------------------------------------------------------------------------------------------------

def test_eval_seg_label_groups_partition_the_35_labels():
    labels = EV.SMALL_SIX+EV.BIG_SIX+EV.OTHER_23
    assert sorted(labels) == list(range(1, 36)) and len(EV.OTHER_23) == 23
    assert EV.SMALL_SIX == (4, 23, 18, 34, 19, 35) and EV.BIG_SIX == (1, 2, 5, 20, 21, 24)


def test_eval_seg_dice_by_hand():
    true = np.array([0, 1, 1, 2, 2, 2, 0, 0])
    pred = np.array([0, 1, 2, 2, 2, 0, 0, 3])
    dice = EV.label_dice(pred, true, 5)
    # label 1: truth {1, 2}, prediction {1}: 2*1/3; label 2: truth {3, 4, 5}, prediction {2, 3, 4}: 2*2/6; label 3: only predicted, 0; label 4: nan
    assert dice[0] == pytest.approx(2/3) and dice[1] == pytest.approx(4/6) and dice[2] == 0. and np.isnan(dice[3])
    lookup = np.arange(36) % 5
    row = EV.summarize_subject(np.ones((3, 3, 3), dtype=np.uint8), np.ones((3, 3, 3), dtype=np.uint8), np.ones((3, 3, 3), dtype=np.uint8), lookup)
    assert row["seg35"] == 1. and row["seg4"] == 1. and row["groups"]["big_six"] == 1. and np.isnan(row["groups"]["small_six"])
    assert EV.no_nan(dict(a=[float("nan"), 1.], b=(2., float("inf")))) == dict(a=[None, 1.], b=[2., None])


def test_eval_seg_paired():
    p = EV.paired([.8, .7, .9, np.nan], [.7, .8, .9, .5])
    assert p["n_positive"] == 1 and p["n_negative"] == 1 and p["n_zero"] == 1 and p["n"] == 3 and p["n_undefined"] == 1
    assert p["mean"] == pytest.approx(0.) and p["median"] == pytest.approx(0.)


@pytest.mark.parametrize("v2,v1,run", [(.86, .85, True), (.85, .85, True), (.8481, .85, True), (.8479, .85, False), (.80, .85, False)])
def test_eval_seg_the_preregistered_rule(v2, v1, run):
    gate = EV.decision(v2, v1)
    assert gate["run_a5pv2"] is run and gate["v2_minus_v1"] == pytest.approx(v2-v1) and (("does not run" in gate["text"]) is (not run))
    assert EV.RUN_A5PV2_MARGIN == .002


@pytest.mark.parametrize("change", ["sealed", "never", "training", "off", "no_source"])
def test_eval_seg_admits_development_subjects_only(tmp_path, change):
    never_clash = [DEV[0]] if change == "never" else T0
    allowlist = load_dev_allowlist(write_json(tmp_path/"a.json", allowlist_document(
        [("synthetic_dev", DEV, False), ("synthetic_t0", never_clash, True), (TRAIN_SOURCE, TRAINING, False)])))
    sealed = types.SimpleNamespace(which=lambda s: "H1" if change == "sealed" and s == DEV[1] else None)
    training = {DEV[0]} if change == "training" else set(TRAINING)
    source = "nope" if change == "no_source" else "synthetic_dev"
    if change == "off":
        allowlist = types.SimpleNamespace(sources=allowlist.sources, never_for_training=frozenset(), allows=lambda s: False)
    with pytest.raises(SubjectSealed):
        EV.admit_subjects(allowlist, source, sealed, training)
    calm = types.SimpleNamespace(which=lambda s: None)
    if change in ("sealed", "no_source"):
        assert EV.admit_subjects(allowlist, "synthetic_dev", calm, set(TRAINING)) == DEV


def _prediction_dir(root, subject, *, segmenter="unet35_v2_full", policy="all_folds_mean"):
    from jd_curl.three_d.nifti import file_sha256
    directory = root/subject
    directory.mkdir(parents=True)
    probs = np.zeros((36, 4, 4, 4), dtype=np.float16)
    probs[1] = 1
    np.savez_compressed(directory/"probs36.npz", p=probs)
    write_json(directory/"provenance.json", dict(layer="structure", segmenter_id=segmenter, subject=subject, policy=policy,
                                                 test_labels_seen=False, members=[], input_sha256={"aligned_norm.nii.gz": "0"*64},
                                                 output_sha256={"probs36.npz": file_sha256(directory/"probs36.npz")}))
    return directory


@pytest.mark.parametrize("change,fragment", [("segmenter", "segmenter"), ("policy", "policy"), ("hash", "sha256"), ("subject", "structure layer of")])
def test_eval_seg_refuses_a_prediction_it_cannot_trust(tmp_path, change, fragment):
    subject = DEV[0]
    directory = _prediction_dir(tmp_path, subject, segmenter="other" if change == "segmenter" else "unet35_v2_full",
                                policy="cross_fit_other_folds" if change == "policy" else "all_folds_mean")
    if change == "hash":
        np.savez_compressed(directory/"probs36.npz", p=np.zeros((36, 4, 4, 4), dtype=np.float16)+1/36)
    if change == "subject":
        document = json.loads((directory/"provenance.json").read_text())
        document["subject"] = DEV[1]
        write_json(directory/"provenance.json", document)
    with pytest.raises(Exception, match=fragment):
        EV.read_prediction(tmp_path, subject, expected_segmenter="unet35_v2_full")
