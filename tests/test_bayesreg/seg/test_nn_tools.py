"""The tools of work package NN (runs/20261105-nn/任务书.md, experiments/nn). sci310, CPU, synthetic small volumes (ids OASIS_OAS1_09xx_MR1); nnunetv2 is never imported, its source is read as text where the tools depend on a name or a log line.

Real repository files are read for subject ids only; no image and no label of a real subject. No test starts a unit, an nnU-Net command or a GPU: the real steps are called in a subprocess with a blocking systemd-run first on PATH and must stop at the snapshot check; the steps themselves are run with run_watched replaced by a stub.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time
import types

import nibabel as nib
import numpy as np
import pytest

from bayesreg.eval.truth import load_sealed_lists
from bayesreg.io.guard import SubjectSealed, load_dev_allowlist
from bayesreg.io.pair import ProvenanceRejected
from bayesreg.seg.unet.data import LabelAccessRefused, LabelGuard, OasisFiles, load_folds
from bayesreg.seg.unet.snapshot import CODE_ROOT, code_record
from jd_curl.three_d.nifti import load_scalar
from jd_curl.validation import InputValidationError

NN = CODE_ROOT/"experiments"/"nn"
sys.path[:0] = [str(CODE_ROOT/"experiments"/"unv2"), str(NN)]
import decide_nn as DECIDE  # noqa: E402
import eval_nn as EVAL  # noqa: E402
import nn_common as NC  # noqa: E402
import nn_lock  # noqa: E402
import prepare_nn as PREP  # noqa: E402
import run_nn as RUN  # noqa: E402
import synthetic_tree as TREE  # noqa: E402

SEG = EVAL.eval_seg
SITE = NC.NN_ENV/"lib/python3.10/site-packages/nnunetv2"
REAL_AFFINE = np.array([[-1., 0., 0., 80.], [0., 0., 1., -112.], [0., -1., 0., 96.], [0., 0., 0., 1.]])     # read from the header of a training subject


def quiet(_):
    pass


# ---- the subject lists -------------------------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_lists():
    """Ids only: the repository's lists."""
    code = code_record()
    admitted = PREP.admitted_training_subjects(code=code)
    return dict(admitted=admitted, forbidden=PREP.forbidden_sets(admitted["sealed"], admitted["allowlist"]))


def test_real_training_list_is_clean(real_lists):
    admitted = real_lists["admitted"]
    assert len(admitted["subjects"]) == 106 and all(v == 0 for v in admitted["overlaps"].values()) and set(admitted["overlaps"]) == set(real_lists["forbidden"])
    assert all(not set(admitted["subjects"]) & set(ids) for ids in real_lists["forbidden"].values())       # counted here, not read from the record
    assert admitted["list_sizes"]["holdout_B_v1"] == 100 and admitted["list_sizes"]["H1"] == 100 and admitted["list_sizes"]["H4"] == 40 and admitted["list_sizes"]["H3"] == 18 and admitted["list_sizes"]["H2"] == 4
    assert admitted["list_sizes"]["never_for_training"] == 24 and admitted["list_sizes"]["allowlist:stage_T0_tune_v1"] == 24 and admitted["list_sizes"]["allowlist:stage_L2R_val_v1"] == 20
    assert "allowlist:bayesreg_train_v1" not in real_lists["forbidden"]      # the allowlist also lists the training set; only that source is left out


@pytest.mark.parametrize("name", ["holdout_B_v1", "H1", "H3", "H4", "H2", "never_for_training", "allowlist:stage_L2R_val_v1", "allowlist:stage_T0_tune_v1", "allowlist:stage_E11_native_v1", "allowlist:stage_LPBA40_v1_tune"])
def test_one_subject_of_any_list_is_refused(real_lists, name):
    """The counterexample: one training subject replaced by one id of the list, for every list."""
    subjects, forbidden = real_lists["admitted"]["subjects"], real_lists["forbidden"]
    swapped = subjects[:-1]+[sorted(forbidden[name])[0]]
    with pytest.raises(PREP.SubjectsRefused, match=f"in {name}"):
        PREP.check_training_subjects(swapped, forbidden)
    assert PREP.check_training_subjects(subjects, forbidden) == subjects


def test_the_check_has_teeth_only_through_the_lists(real_lists):
    """The same swapped list passes against no list at all (an OASIS id): the refusal above comes from the list."""
    subjects, forbidden = real_lists["admitted"]["subjects"], real_lists["forbidden"]
    swapped = subjects[:-1]+[sorted(forbidden["holdout_B_v1"])[0]]
    assert sorted(swapped) == PREP.check_training_subjects(swapped, {})


def test_counts_repeats_spelling_and_other_files_are_refused(real_lists):
    subjects = real_lists["admitted"]["subjects"]
    with pytest.raises(PREP.SubjectsRefused, match="105 subjects"):
        PREP.check_training_subjects(subjects[:-1], {})
    with pytest.raises(PREP.SubjectsRefused, match="listed twice"):
        PREP.check_training_subjects(subjects[:-1]+[subjects[0]], {})
    with pytest.raises(PREP.SubjectsRefused, match="not canonical"):
        PREP.check_training_subjects(subjects[:-1]+["oasis_oas1_0022_mr1"], {})
    with pytest.raises(PREP.SubjectsRefused, match="differs from the folds"):
        PREP.check_training_subjects(subjects, {}, folds_subjects=subjects[:-1])
    with pytest.raises(PREP.SubjectsRefused, match="differs from the allowlist training source"):
        PREP.check_training_subjects(subjects, {}, allowlist_source=subjects[1:])


def test_train_list_is_pinned(tmp_path):
    copy = tmp_path/"train_subjects.txt"
    copy.write_bytes(PREP.TRAIN_LIST.read_bytes()+b"OASIS_OAS1_0438_MR1\n")
    with pytest.raises(PREP.SubjectsRefused, match="pinned"):
        PREP.read_train_list(copy)
    assert len(PREP.read_train_list()) == 106


def test_a_missing_source_is_refused_not_ignored():
    """A check against a list that is not there says nothing."""
    sealed = types.SimpleNamespace(subjects={})
    full = [dict(name="bayesreg_train_v1", subjects=["a"])]+[dict(name=name, subjects=["x"]*count) for name, count in PREP.REQUIRED_SOURCES.items()]
    allowlist = types.SimpleNamespace(sources=full, never_for_training=frozenset(["x"]))
    assert "allowlist:stage_T0_tune_v1" in PREP.forbidden_sets(sealed, allowlist)
    with pytest.raises(PREP.SubjectsRefused, match="no source 'stage_T0_tune_v1'"):
        PREP.forbidden_sets(sealed, types.SimpleNamespace(sources=[s for s in full if s["name"] != "stage_T0_tune_v1"], never_for_training=frozenset()))
    with pytest.raises(PREP.SubjectsRefused, match="has 1 subjects, expected 24"):
        PREP.forbidden_sets(sealed, types.SimpleNamespace(sources=[dict(s, subjects=["x"]) if s["name"] == "stage_T0_tune_v1" else s for s in full], never_for_training=frozenset()))
    with pytest.raises(PREP.SubjectsRefused, match="no training source"):
        PREP.forbidden_sets(sealed, types.SimpleNamespace(sources=full[1:], never_for_training=frozenset()))


# ---- the raw directory and dataset.json ---------------------------------------------------------------------------------------------------------------------

SUBJECTS = [f"OASIS_OAS1_{n:04d}_MR1" for n in (941, 942, 943)]


@pytest.fixture
def raw(tmp_path):
    data = tmp_path/"data"
    for subject in SUBJECTS:
        (data/subject).mkdir(parents=True)
        (data/subject/"aligned_norm.nii.gz").write_bytes(b"image "+subject.encode())
        (data/subject/"aligned_seg35.nii.gz").write_bytes(b"label "+subject.encode())
    root = tmp_path/"nnUNet_raw"
    root.mkdir()
    return root, data, PREP.build_raw(root, SUBJECTS, data)


def test_raw_holds_symbolic_links_only(raw):
    root, data, record = raw
    dataset = root/"Dataset935_OASIS35"
    assert record["built"] is True and record["n_links"] == 6 and sorted(p.name for p in dataset.iterdir()) == ["dataset.json", "imagesTr", "labelsTr"]
    files = [p for folder in ("imagesTr", "labelsTr") for p in (dataset/folder).iterdir()]
    assert len(files) == 6 and all(p.is_symlink() for p in files)
    assert sorted(p.name for p in (dataset/"imagesTr").iterdir()) == [f"{s}_0000.nii.gz" for s in SUBJECTS] and sorted(p.name for p in (dataset/"labelsTr").iterdir()) == [f"{s}.nii.gz" for s in SUBJECTS]
    assert Path(os.readlink(dataset/"imagesTr"/f"{SUBJECTS[0]}_0000.nii.gz")) == data/SUBJECTS[0]/"aligned_norm.nii.gz" and Path(os.readlink(dataset/"labelsTr"/f"{SUBJECTS[1]}.nii.gz")) == data/SUBJECTS[1]/"aligned_seg35.nii.gz"
    assert PREP.build_raw(root, SUBJECTS, data)["built"] is False              # a second run verifies
    assert not list(root.glob(".partial*"))


def intrude_regular_file(dataset):
    target = dataset/"imagesTr"/f"{SUBJECTS[0]}_0000.nii.gz"
    target.unlink()
    target.write_bytes(b"a copy of the data")


def intrude_wrong_target(dataset):
    target = dataset/"labelsTr"/f"{SUBJECTS[0]}.nii.gz"
    target.unlink()
    os.symlink("/mnt/d/data/OASIS_OAS1_0438_MR1/aligned_seg35.nii.gz", target)


def intrude_extra_case(dataset):
    os.symlink("/mnt/d/data/OASIS_OAS1_0438_MR1/aligned_norm.nii.gz", dataset/"imagesTr"/"OASIS_OAS1_0438_MR1_0000.nii.gz")


def intrude_missing_case(dataset):
    (dataset/"labelsTr"/f"{SUBJECTS[2]}.nii.gz").unlink()


def intrude_top_level(dataset):
    (dataset/"imagesTs").mkdir()


def intrude_dataset_json(dataset):
    document = json.loads((dataset/"dataset.json").read_text())
    document["numTraining"] = 107
    (dataset/"dataset.json").write_text(json.dumps(document))


@pytest.mark.parametrize("intrude", [intrude_regular_file, intrude_wrong_target, intrude_extra_case, intrude_missing_case, intrude_top_level, intrude_dataset_json])
def test_raw_that_differs_is_refused(raw, intrude):
    root, data, _ = raw
    intrude(root/"Dataset935_OASIS35")
    with pytest.raises(PREP.RawRefused):
        PREP.build_raw(root, SUBJECTS, data)


def test_raw_refuses_a_missing_source_file_and_builds_nothing(tmp_path):
    root = tmp_path/"raw"
    root.mkdir()
    with pytest.raises(PREP.RawRefused, match="missing"):
        PREP.build_raw(root, SUBJECTS, tmp_path/"no_data")
    assert not list(root.iterdir())


def test_dataset_json_content(raw):
    root, _, _ = raw
    document = json.loads((root/"Dataset935_OASIS35"/"dataset.json").read_text(encoding="utf-8"))
    assert document["channel_names"] == {"0": "T1"} and document["numTraining"] == 3 and document["file_ending"] == ".nii.gz" and document["overwrite_image_reader_writer"] == "SimpleITKIO"
    labels = document["labels"]
    assert len(labels) == 36 and sorted(labels.values()) == list(range(36)) and labels["background"] == 0
    assert labels["Left-Cerebral-White-Matter"] == 1 and labels["Right-Cerebral-Cortex"] == 21 and labels["Brain-Stem"] == 13 and labels["Right-Choroid-Plexus"] == 35 and "Unknown" not in labels
    assert sum(k.startswith(("Left-", "Right-")) for k in labels) == 32       # why the trainer does not mirror


# ---- gt_segmentations (hard rule 8) ---------------------------------------------------------------------------------------------------------------------------------

@pytest.fixture
def gt(tmp_path):
    """gt_segmentations as nnU-Net leaves it: a copy of each label file."""
    data = tmp_path/"data"
    directory = tmp_path/"preprocessed"/"Dataset935_OASIS35"/"gt_segmentations"
    directory.mkdir(parents=True)
    for subject in SUBJECTS:
        (data/subject).mkdir(parents=True)
        (data/subject/"aligned_seg35.nii.gz").write_bytes(b"label "+subject.encode())
        (directory/f"{subject}.nii.gz").write_bytes(b"label "+subject.encode())
    return directory, data


def test_gt_copies_become_links(gt):
    directory, data = gt
    assert directory == PREP.gt_directory(directory.parents[1])
    with pytest.raises(PREP.RawRefused, match="not symbolic links"):
        PREP.verify_gt_links(directory, SUBJECTS, data)
    record = PREP.link_gt_segmentations(directory, SUBJECTS, data)
    assert record["replaced"] == 3 and record["n_links"] == 3 and record["bytes_freed"] > 0 and sorted(p.name for p in directory.iterdir()) == [f"{s}.nii.gz" for s in SUBJECTS]
    assert all(p.is_symlink() for p in directory.iterdir()) and Path(os.readlink(directory/f"{SUBJECTS[0]}.nii.gz")) == data/SUBJECTS[0]/"aligned_seg35.nii.gz"
    assert (directory/f"{SUBJECTS[0]}.nii.gz").read_bytes() == b"label "+SUBJECTS[0].encode()
    assert PREP.link_gt_segmentations(directory, SUBJECTS, data)["replaced"] == 0 and PREP.verify_gt_links(directory, SUBJECTS, data)["n_links"] == 3


def test_a_copy_that_differs_refuses_all_of_it(gt):
    directory, data = gt
    (directory/f"{SUBJECTS[2]}.nii.gz").write_bytes(b"something else")
    with pytest.raises(PREP.RawRefused, match="differ from their source"):
        PREP.link_gt_segmentations(directory, SUBJECTS, data)
    assert not any(p.is_symlink() for p in directory.iterdir())               # nothing was replaced


def test_gt_with_other_cases_is_refused(gt):
    directory, data = gt
    (directory/"OASIS_OAS1_0438_MR1.nii.gz").write_bytes(b"x")
    with pytest.raises(PREP.RawRefused, match="only present"):
        PREP.link_gt_segmentations(directory, SUBJECTS, data)
    (directory/"OASIS_OAS1_0438_MR1.nii.gz").unlink()
    (directory/f"{SUBJECTS[0]}.nii.gz").unlink()
    with pytest.raises(PREP.RawRefused, match="missing"):
        PREP.link_gt_segmentations(directory, SUBJECTS, data)
    with pytest.raises(PREP.RawRefused, match="no such directory"):
        PREP.verify_gt_links(directory/"nowhere", SUBJECTS, data)


def test_verify_gt_links_refuses_a_link_elsewhere(gt):
    directory, data = gt
    PREP.link_gt_segmentations(directory, SUBJECTS, data)
    target = directory/f"{SUBJECTS[0]}.nii.gz"
    target.unlink()
    os.symlink("/mnt/d/data/OASIS_OAS1_0438_MR1/aligned_seg35.nii.gz", target)
    with pytest.raises(PREP.RawRefused, match="point elsewhere"):
        PREP.verify_gt_links(directory, SUBJECTS, data)


def test_link_gt_records_once(gt, tmp_path, monkeypatch):
    directory, data = gt
    monkeypatch.setattr(RUN, "LOG_FILE", None)
    out = tmp_path/"out"
    root = out/"nnUNet_preprocessed"/"Dataset935_OASIS35"
    root.mkdir(parents=True)
    directory.rename(root/"gt_segmentations")
    RUN.link_gt(out, SUBJECTS, code=dict(commit="c"), data_root=data)
    first = (out/"gt_links_record.json").read_text()
    assert json.loads(first)["replaced"] == 3 and json.loads(first)["code"]["commit"] == "c"
    RUN.link_gt(out, SUBJECTS, code=dict(commit="d"), data_root=data)
    assert (out/"gt_links_record.json").read_text() == first


def test_require_preprocessed_needs_the_links(gt, tmp_path, monkeypatch):
    directory, data = gt
    out = tmp_path/"out"
    root = out/"nnUNet_preprocessed"/"Dataset935_OASIS35"
    root.mkdir(parents=True)
    directory.rename(root/"gt_segmentations")
    (root/"nnUNetPlans_3d_fullres").mkdir()
    for subject in SUBJECTS:
        (root/"nnUNetPlans_3d_fullres"/f"{subject}.pkl").write_bytes(b"")
    with pytest.raises(RUN.Stop, match="prepare and preprocess come first"):
        RUN.require_preprocessed(out)
    (out/"preprocess_record.json").write_text("{}")
    (out/"nnUNet_raw").mkdir()
    PREP.build_raw(out/"nnUNet_raw", SUBJECTS, _images_too(data))
    monkeypatch.setattr(PREP, "admitted_training_subjects", lambda code=None: dict(subjects=SUBJECTS))
    monkeypatch.setattr(NC, "DATA_ROOT", data)
    with pytest.raises(PREP.RawRefused, match="not symbolic links"):
        RUN.require_preprocessed(out)
    PREP.link_gt_segmentations(root/"gt_segmentations", SUBJECTS, data)
    assert RUN.require_preprocessed(out) == SUBJECTS
    (root/"nnUNetPlans_3d_fullres"/"OASIS_OAS1_0438_MR1.pkl").write_bytes(b"")
    with pytest.raises(RUN.Stop, match="not the training subjects"):
        RUN.require_preprocessed(out)


def _images_too(data):
    for subject in SUBJECTS:
        (data/subject/"aligned_norm.nii.gz").write_bytes(b"image")
    return data


# ---- the label check -----------------------------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    work = tmp_path_factory.mktemp("nn_tree")
    TREE.build(work)
    folds, allowlist = load_folds(work/"folds.json"), load_dev_allowlist(work/"dev_subjects.json")
    return types.SimpleNamespace(work=work, data=work/"data", files=OasisFiles(work/"data"), guard=LabelGuard(folds, allowlist, training_source=TREE.SOURCE))


def test_inspect_subject_reads_labels_of_training_subjects_only(tree):
    entry = PREP.inspect_subject(TREE.TRAINING[0], tree.files, tree.guard)
    assert entry["shape"] == [32, 32, 32] and set(entry["labels_present"]) <= set(range(36)) and entry["image"]["head_sha256"] == hashlib.sha256((tree.data/TREE.TRAINING[0]/"aligned_norm.nii.gz").read_bytes()[:PREP.HEAD_BYTES]).hexdigest() and entry["image"]["whole_file"] is True
    for subject in (TREE.DEV[0], TREE.T0[0]):
        with pytest.raises(LabelAccessRefused):
            PREP.inspect_subject(subject, tree.files, tree.guard)


def test_a_label_outside_the_table_stops(tree, tmp_path):
    subject = TREE.TRAINING[1]
    (tmp_path/subject).mkdir()
    image = nib.load(tree.data/subject/"aligned_norm.nii.gz")
    labels = np.zeros((16, 16, 16), dtype=np.uint8)
    labels[0, 0, 0] = 36
    for name, values in (("aligned_seg35.nii.gz", labels), ("aligned_norm.nii.gz", labels.astype(np.float32))):
        volume = nib.Nifti1Image(values, image.affine)
        volume.set_qform(image.affine, code=1)
        volume.set_sform(image.affine, code=1)
        volume.header.set_xyzt_units("mm")
        nib.save(volume, tmp_path/subject/name)
    with pytest.raises(InputValidationError, match="outside 0..35"):
        PREP.inspect_subject(subject, OasisFiles(tmp_path), tree.guard)


def test_geometry_mismatch_stops(tree, tmp_path):
    subject = TREE.TRAINING[2]
    (tmp_path/subject).mkdir()
    image = nib.load(tree.data/subject/"aligned_norm.nii.gz")
    nib.save(image, tmp_path/subject/"aligned_norm.nii.gz")
    moved = image.affine.copy()
    moved[0, 3] += 1
    nib.save(nib.Nifti1Image(np.zeros(image.shape, dtype=np.uint8), moved), tmp_path/subject/"aligned_seg35.nii.gz")
    with pytest.raises(ValueError, match="differ in shape or affine"):
        PREP.inspect_subject(subject, OasisFiles(tmp_path), tree.guard)


def test_label_summary():
    good = PREP.label_summary([dict(subject="a", labels_present=list(range(36))), dict(subject="b", labels_present=[0, 1, 2])])
    assert good["ok"] and good["absent_labels"] == [] and good["subjects_with_label"]["1"] == 2 and good["labels_not_in_every_subject"]["35"] == 1
    assert PREP.label_summary([dict(subject="a", labels_present=[0, 1])])["absent_labels"] == list(range(2, 36))
    wrong = PREP.label_summary([dict(subject="a", labels_present=[0, 40])])
    assert not wrong["ok"] and wrong["outside"] == [("a", 40)]


def test_plan_summary_reads_the_plan_file(tmp_path):
    plans = dict(plans_name="nnUNetPlans", transpose_forward=[0, 1, 2], image_reader_writer="SimpleITKIO", configurations={"3d_fullres": dict(patch_size=[1, 2, 3], batch_size=2, spacing=[1., 1., 1.], median_image_size_in_voxels=[4., 5., 6.], normalization_schemes=["ZScoreNormalization"], use_mask_for_norm=[True], architecture=dict(network_class_name="x.PlainConvUNet", arch_kwargs=dict(n_stages=6, features_per_stage=[32, 64])))})
    (tmp_path/"plans.json").write_text(json.dumps(plans))
    (tmp_path/"fingerprint.json").write_text(json.dumps(dict(shapes_after_crop=[[1, 2, 3], [3, 4, 5]], median_relative_size_after_cropping=.6)))
    out = PREP.plan_summary(tmp_path/"plans.json", tmp_path/"fingerprint.json")
    assert out["patch_size"] == [1, 2, 3] and out["batch_size"] == 2 and out["network_class"] == "x.PlainConvUNet" and out["features_per_stage"] == [32, 64] and out["shapes_after_crop_median"] == [2., 3., 4.]


# ---- the trainers, as nnU-Net 2.8.1 has them (its source, read as text) --------------------------------------------------------------------------------------

def test_the_trainer_names_exist_in_the_installed_nnunet():
    assert (SITE.parent/"nnunetv2-2.8.1.dist-info").is_dir()                 # the installed version, not a sentence of ours
    source = (SITE/"training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs_NoMirroring.py").read_text(encoding="utf-8")
    assert "class nnUNetTrainer_250epochs_NoMirroring(nnUNetTrainer):" in source and "self.num_epochs = 250" in source and "mirror_axes = None" in source and "self.inference_allowed_mirroring_axes = None" in source
    assert "class nnUNetTrainer_100epochs_NoMirroring" not in source                                # hence ext_trainers/
    assert "mirror_axes = (0, 1, 2)" in (SITE/"training/nnUNetTrainer/nnUNetTrainer.py").read_text(encoding="utf-8")   # why not the stock variants
    assert NC.TRAINERS == {250: "nnUNetTrainer_250epochs_NoMirroring", 100: "nnUNetTrainer_100epochs_NoMirroring"}


def test_the_external_trainer_does_not_mirror():
    tree_ = ast.parse((NC.EXT_TRAINERS/"nn_trainers_v1.py").read_text(encoding="utf-8"))
    classes = [node for node in tree_.body if isinstance(node, ast.ClassDef)]
    assert [c.name for c in classes] == ["nnUNetTrainer_100epochs_NoMirroring"]
    text = ast.unparse(classes[0])
    assert "self.num_epochs = 100" in text and "mirror_axes = None" in text and "self.inference_allowed_mirroring_axes = None" in text
    assert "nnUNet_extTrainer" in (SITE/"utilities/find_objects.py").read_text(encoding="utf-8")


def test_the_log_lines_the_driver_watches_exist_in_the_installed_nnunet():
    source = (SITE/"training/nnUNetTrainer/nnUNetTrainer.py").read_text(encoding="utf-8")
    assert 'f"Epoch time: {' in source and 'self.print_to_log_file(f"predicting {k}")' in source and 'args = (f"{dt_object}:", *args)' in source
    on_train_end = source[source.index("def on_train_end"):source.index("def train_step")]
    run_training = source[source.index("    def run_training(self):"):]
    assert 'self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))' in on_train_end and "self.on_train_end()" in run_training
    entry = (SITE/"run/run_training.py").read_text(encoding="utf-8")
    body = entry[entry.index("        if not only_run_validation:"):entry.index("def run_training_entry")]
    assert body.index("nnunet_trainer.run_training()") < body.index("nnunet_trainer.perform_actual_validation(")     # the order of the calls: checkpoint_final is written before the first `predicting`
    assert source[source.index("def perform_actual_validation"):].index('self.print_to_log_file(f"predicting {k}")') > 0
    assert "'nnUNet_compile' not in os.environ.keys()" in source and "num_processes=max(1, allowed_num_processes // 2)" in source


def test_infer_script_imports_nothing_of_this_repository():
    tree_ = ast.parse((NN/"infer_nn.py").read_text(encoding="utf-8"))
    modules = {n.module.split(".")[0] for n in ast.walk(tree_) if isinstance(n, ast.ImportFrom)} | {a.name.split(".")[0] for n in ast.walk(tree_) if isinstance(n, ast.Import) for a in n.names}
    assert modules <= {"argparse", "datetime", "hashlib", "json", "os", "pathlib", "platform", "sys", "time", "importlib", "torch", "nnunetv2"}
    assert "nnunetv2" not in sys.modules                                    # these tests never import it


# ---- the environment and the command lines ---------------------------------------------------------------------------------------------------------------------

def test_nn_env_keeps_sci310_out():
    base = dict(PATH="/home/rmcfz/miniconda3/envs/sci310/bin:/home/rmcfz/miniconda3/condabin:/usr/bin:/bin", PYTHONPATH="/home/x/seq_vp_tag", CONDA_PREFIX="/home/rmcfz/miniconda3/envs/sci310", CONDA_DEFAULT_ENV="sci310", PYTHONHOME="/x", CUDA_VISIBLE_DEVICES="3", HOME="/home/rmcfz")
    env = NC.nn_env(Path("/results/nn"), base=base)
    assert "sci310" not in " ".join(env.values()) and "PYTHONPATH" not in env and "PYTHONHOME" not in env and "CONDA_PREFIX" not in env and env["HOME"] == "/home/rmcfz"
    assert env["PATH"].split(":")[0] == str(NC.NN_BIN) and env["PATH"].split(":")[1:] == ["/usr/bin", "/bin"] and env["PYTHONNOUSERSITE"] == "1"
    assert env["nnUNet_raw"] == "/results/nn/nnUNet_raw" and env["nnUNet_preprocessed"] == "/results/nn/nnUNet_preprocessed" and env["nnUNet_results"] == "/results/nn/nnUNet_results" and env["nnUNet_extTrainer"] == str(NC.EXT_TRAINERS)
    assert env["nnUNet_n_proc_DA"] == "6" and NC.N_PROC_DA_VALIDATION == 3 and env["nnUNet_def_n_proc"] == "2" and env["nnUNet_compile"] == "f" and env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert "CUDA_VISIBLE_DEVICES" not in NC.nn_env(Path("/results/nn"), base=base, gpu=True)
    assert NC.nn_env(Path("/r"), base=base, results="/r/smoke")["nnUNet_results"] == "/r/smoke"


def test_nn_commands():
    plan = NC.plan_argv()
    assert plan[:3] == ["nice", "-n", "19"] and plan[3] == str(NC.NN_BIN/"nnUNetv2_plan_and_preprocess") and plan[plan.index("-np")+1] == "2" and plan[plan.index("-npfp")+1] == "2" and plan[plan.index("-c")+1] == "3d_fullres" and plan[plan.index("-d")+1] == "935" and "--verify_dataset_integrity" in plan and "--c" not in plan
    train = NC.train_argv("nnUNetTrainer_250epochs_NoMirroring")
    assert train == [str(NC.NN_BIN/"nnUNetv2_train"), "935", "3d_fullres", "all", "-tr", "nnUNetTrainer_250epochs_NoMirroring", "-p", "nnUNetPlans", "-device", "cuda"]
    assert NC.train_argv("t", resume=True)[-1] == "--c" and NC.train_argv("t", device="cpu")[-1] == "cpu"
    infer = NC.infer_argv(model_dir="m", input_dir="i", out_dir="o")
    assert infer[:4] == [str(NC.NN_PY), "-I", "-B", str(NN/"infer_nn.py")] and infer[infer.index("--device")+1] == "cuda"
    assert NC.fold_directory("T", results=Path("/r")) == Path("/r/Dataset935_OASIS35/T__nnUNetPlans__3d_fullres/fold_all")


@pytest.mark.parametrize("step,unit,memory", [("smoke", "tag-nn-smoke", "12G"), ("train", "tag-nn-train", "12G"), ("train-continue", "tag-nn-train", "12G"), ("infer", "tag-nn-infer", "8G"), ("eval", "tag-nn-eval", "6G")])
def test_start_strings(step, unit, memory):
    argv = RUN.launch_argv(step)
    text = " ".join(argv)
    assert argv[:2] == ["systemd-run", "--user"] and f"--unit={unit}" in argv and f"--property=MemoryMax={memory}" in argv and "pytest" not in text and "seq-vp-" not in text
    assert f"--working-directory={NC.OUT}/snapshot" in argv and f"{NC.OUT}/snapshot/experiments/nn/run_nn.py" in argv and NC.PY in argv
    assert "--setenv=OMP_NUM_THREADS=1" in argv and "--setenv=OPENBLAS_NUM_THREADS=1" in argv and "--setenv=MKL_NUM_THREADS=1" in argv and "--setenv=NUMEXPR_NUM_THREADS=1" in argv
    assert any(a.startswith("--property=RuntimeMaxSec=") for a in argv)
    assert ("--continue" in argv) == (step == "train-continue")
    assert ("--property=RuntimeMaxSec=39600" in argv) == step.startswith("train")   # about 11 hours: 10 of the rule, the gate, what `Epoch time` leaves out


def test_start_string_rules():
    good = RUN.launch_argv("smoke")
    assert RUN.check_launch(good) == good
    for name, change in (("test runner", lambda a: a+["pytest"]), ("unit", lambda a: [x.replace("tag-nn-", "seq-vp-") for x in a]), ("unit", lambda a: [x.replace("tag-nn-smoke", "tag-ul-smoke") for x in a]), ("working directory", lambda a: [x.replace(str(NC.RESULTS), "/tmp") for x in a]), ("MemoryMax", lambda a: [x for x in a if not x.startswith("--property=MemoryMax")]), ("MemoryMax", lambda a: [x for x in a if not x.startswith("--property=RuntimeMaxSec")])):
        with pytest.raises(ValueError, match=name):
            RUN.check_launch(change(good))
    with pytest.raises(ValueError, match="no step"):
        RUN.launch_argv("anything")
    assert RUN.launch_argv("eval", bz_go="BZ, 9 am")[-2:] == ["--bz-go", "BZ, 9 am"]


def test_snapshot_paths_hold_what_the_steps_read():
    assert not [p for p in RUN.SNAPSHOT_PATHS_NN if "holdout" in p]
    assert all((CODE_ROOT/p).exists() for p in RUN.SNAPSHOT_PATHS_NN)
    assert "experiments/nn" in RUN.SNAPSHOT_PATHS_NN and "experiments/unv2" in RUN.SNAPSHOT_PATHS_NN and "splits/bayesreg_v1/train_subjects.txt" in RUN.SNAPSHOT_PATHS_NN


# ---- the epoch rule -------------------------------------------------------------------------------------------------------------------------------------------

def measured(seconds, host=6., gpu=8000):
    return dict(epoch_seconds=seconds, n_proc_da=6, host_peak_pss_gib=host, gpu_peak_used_mib=gpu, gpu_mean_utilization_percent=70.)


@pytest.mark.parametrize("seconds,epochs", [(60., 250), (115.3, 250), (144., 250), (144.01, 100), (300., 100)])
def test_epoch_rule(seconds, epochs):
    rule = RUN.epochs_rule(measured([500., seconds, seconds, seconds]))     # the first epoch has the warm-up and is not used
    assert rule["ok"] and rule["epochs"] == epochs and rule["trainer"] == NC.TRAINERS[epochs] and rule["epoch_seconds_used"] == seconds
    assert rule["hours_for_250"] == pytest.approx(250*seconds/3600) and rule["hours_for_chosen"] == pytest.approx(epochs*seconds/3600) and rule["limit_hours"] == 10. and rule["n_proc_da"] == 6


def test_epoch_rule_takes_the_median_from_the_second_epoch():
    assert RUN.SMOKE_EPOCHS == 4
    assert RUN.epochs_rule(measured([900., 100., 200., 150.]))["epoch_seconds_used"] == 150.
    assert RUN.epochs_rule(measured([900., 140., 150., 141.]))["epochs"] == 250      # one slow epoch does not flip it
    assert RUN.epochs_rule(measured([100., 100., 100., 100., 700., 700., 700.]))["epochs"] == 100


def test_epoch_rule_budgets_and_too_few_epochs():
    assert RUN.epochs_rule(measured([100.]*3))["ok"] is False and RUN.epochs_rule(measured([100.]*3))["trainer"] is None
    high = RUN.epochs_rule(measured([100.]*4, host=11.5))
    assert not high["ok"] and "host peak" in high["reasons"][0]
    assert not RUN.epochs_rule(measured([100.]*4, gpu=23000))["ok"]
    assert not RUN.epochs_rule(measured([100.]*4, gpu=None))["ok"]


def write_decision(out, seconds=100.):
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out)/"smoke_measure.json").write_text(json.dumps(measured([seconds]*4)))
    return RUN.decide_epochs(out, code=dict(kind="snapshot", commit="c"*40))


def test_epoch_decision_is_written_once_read_and_recomputed(tmp_path):
    with pytest.raises(RUN.Stop, match="smoke comes first"):
        RUN.decide_epochs(tmp_path, code={})
    with pytest.raises(RUN.Stop, match="decide-epochs come first"):
        RUN.read_decision(tmp_path)
    document = write_decision(tmp_path, 90.)
    assert document["trainer"] == NC.TRAINERS[250] and document["code"]["commit"] == "c"*40
    assert RUN.read_decision(tmp_path)["epochs"] == 250
    with pytest.raises(RUN.Stop, match="never overwritten"):
        RUN.decide_epochs(tmp_path, code={})
    edited = json.loads((tmp_path/"epochs_decision.json").read_text())
    (tmp_path/"epochs_decision.json").write_text(json.dumps(dict(edited, trainer=NC.TRAINERS[100], epochs=100)))
    with pytest.raises(RUN.Stop, match="does not follow"):
        RUN.read_decision(tmp_path)
    (tmp_path/"epochs_decision.json").write_text(json.dumps(edited))
    (tmp_path/"smoke_measure.json").unlink()
    with pytest.raises(RUN.Stop, match="nothing to be checked against"):
        RUN.read_decision(tmp_path)
    (tmp_path/"epochs_decision.json").write_text(json.dumps(dict(ok=False, reasons=["host"])))
    with pytest.raises(RUN.Stop, match="did not pass"):
        RUN.read_decision(tmp_path)


def test_retire_smoke_keeps_a_failed_smoke_and_clears_the_way(tmp_path, monkeypatch):
    monkeypatch.setattr(RUN, "LOG_FILE", None)
    with pytest.raises(ValueError, match="reason"):
        RUN.retire_smoke(tmp_path, "  ")
    with pytest.raises(RUN.Stop, match="no smoke to retire"):
        RUN.retire_smoke(tmp_path, "why")
    (tmp_path/"smoke_measure.json").write_text(json.dumps(measured([100.])))
    (tmp_path/"epochs_decision.json").write_text(json.dumps(dict(ok=False)))
    NC.smoke_results_root(tmp_path).mkdir()
    (NC.smoke_results_root(tmp_path)/"training_log_x.txt").write_text("evidence")
    (tmp_path/"logs").mkdir()
    (tmp_path/"logs"/"smoke.log").write_text("evidence")
    target = RUN.retire_smoke(tmp_path, "the driver died")
    assert target.parent == tmp_path/"smoke_attempts" and not (tmp_path/"smoke_measure.json").exists() and not (tmp_path/"epochs_decision.json").exists() and not NC.smoke_results_root(tmp_path).exists() and not (tmp_path/"logs"/"smoke.log").exists()
    assert (target/"nnUNet_results_smoke"/"training_log_x.txt").read_text() == "evidence" and (target/"smoke.log").read_text() == "evidence" and json.loads((target/"retired.json").read_text())["reason"] == "the driver died" and json.loads((target/"smoke_measure.json").read_text())["epoch_seconds"] == [100.]


def test_a_smoke_that_passed_is_not_retired(tmp_path, monkeypatch):
    monkeypatch.setattr(RUN, "LOG_FILE", None)
    (tmp_path/"smoke_measure.json").write_text(json.dumps(measured([100.]*4)))
    with pytest.raises(RUN.Stop, match="this smoke passed"):
        RUN.retire_smoke(tmp_path, "I prefer another number")
    assert (tmp_path/"smoke_measure.json").is_file()
    (tmp_path/"smoke_measure.json").write_text(json.dumps(measured([100.])))
    (tmp_path/"train_runs").mkdir()
    with pytest.raises(RUN.Stop, match="the training has started"):
        RUN.retire_smoke(tmp_path, "why")


# ---- watching the training ---------------------------------------------------------------------------------------------------------------------------------------

def log_line(text):
    return f"2026-11-05 01:02:03.123456: {text} \n"


def test_training_progress_reads_the_log_lines(tmp_path):
    old = tmp_path/"training_log_2026_11_4_01_00_00.txt"
    old.write_text(log_line("Epoch time: 999.0 s")+log_line("predicting OASIS_OAS1_0022_MR1"))
    os.utime(old, (1000, 1000))
    new = tmp_path/"training_log_2026_11_5_01_00_00.txt"
    new.write_text(log_line("Epoch 0")+log_line("Epoch time: 101.25 s")+log_line("Epoch 1")+log_line("Epoch time: 98.5 s")+log_line("Using predicting in a sentence is not a start"))
    progress = RUN.training_progress(tmp_path, 2000.)
    assert progress == dict(epoch_seconds=[101.25, 98.5], validating=False)
    assert RUN.training_progress(tmp_path, 0.)["validating"] is True and len(RUN.training_progress(tmp_path, 0.)["epoch_seconds"]) == 3
    with new.open("a") as stream:
        stream.write(log_line("predicting OASIS_OAS1_0022_MR1"))
    assert RUN.training_progress(tmp_path, 2000.)["validating"] is True


def child_script(tmp_path, body):
    path = tmp_path/"child.py"
    path.write_text(body)
    return [sys.executable, "-I", "-B", str(path)]


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


GROUP_CHILD = "import subprocess, sys, time\nchild = subprocess.Popen([sys.executable, '-I', '-c', 'import time; time.sleep(60)'])\nprint('grandchild', child.pid, flush=True)\ntime.sleep(60)\n"


def test_run_watched_stops_on_the_condition_and_takes_the_whole_group(tmp_path):
    flag = tmp_path/"flag"
    argv = child_script(tmp_path, GROUP_CHILD)
    log_path = tmp_path/"logs"/"child.log"

    def condition():
        if "grandchild" in log_path.read_text():
            flag.write_text("x")
        return flag.exists()

    outcome = RUN.run_watched(argv, env=dict(os.environ), log_path=log_path, cwd=tmp_path, should_stop=condition, poll=.05, grace=2.)
    assert outcome["stopped"] == "condition" and outcome["returncode"] != 0 and outcome["host_peak_pss_gib"] >= 0
    pid = int(next(line for line in log_path.read_text().splitlines() if line.startswith("grandchild")).split()[1])
    deadline = time.monotonic()+5
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(.05)
    assert not alive(pid)
    assert "stopping: the stop condition held" in log_path.read_text() and "start: " in log_path.read_text() and "end: exit" in log_path.read_text()


GROUP_IGNORES_SIGTERM = "import signal, subprocess, sys, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nchild = subprocess.Popen([sys.executable, '-I', '-c', 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print(1, flush=True); time.sleep(60)'], stdout=subprocess.DEVNULL)\nprint('grandchild', child.pid, flush=True)\ntime.sleep(60)\n"


def test_stop_reaches_a_group_that_ignores_sigterm(tmp_path):
    """The last SIGKILL of stop_group: resource_tracker ignores SIGTERM."""
    argv = child_script(tmp_path, GROUP_IGNORES_SIGTERM)
    log_path = tmp_path/"child.log"
    outcome = RUN.run_watched(argv, env=dict(os.environ), log_path=log_path, cwd=tmp_path, should_stop=lambda: "grandchild" in log_path.read_text(), poll=.05, grace=.5)
    pid = int(next(line for line in log_path.read_text().splitlines() if line.startswith("grandchild")).split()[1])
    assert outcome["returncode"] == -signal.SIGKILL and outcome["group_left"] == [] and not alive(pid) and RUN.group_members(outcome["pid"]) == []


def test_the_crash_of_the_sandbox_is_a_sigkill(tmp_path):
    outcome = RUN.run_watched(child_script(tmp_path, "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(60)\n"), env=dict(os.environ), log_path=tmp_path/"c.log", cwd=tmp_path, should_stop=lambda: True, first_signal=signal.SIGKILL, poll=.05, grace=30.)
    assert outcome["returncode"] == -signal.SIGKILL and outcome["wall_seconds"] < 25.


def test_empty_group_kills_what_a_finished_child_left(tmp_path):
    script = child_script(tmp_path, "import signal, sys, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(60)\n")
    process = subprocess.Popen(script, start_new_session=True)
    try:
        assert process.pid in RUN.group_members(process.pid)
        assert RUN.empty_group(process.pid, timeout=.1) == []
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
    assert RUN.group_members(process.pid) == []


def test_log_span_seconds():
    assert RUN.log_span_seconds("") == 0. and RUN.log_span_seconds(log_line("one")) == 0.
    text = "2026-11-05 01:00:00.500000: a \nno stamp here\n2026-11-05 01:30:00: b \n2026-11-05 02:00:00.500000: c \n"
    assert RUN.log_span_seconds(text) == 3600.


def test_run_watched_memory_limits(tmp_path):
    argv = child_script(tmp_path, "import time\ntime.sleep(60)\n")
    over = RUN.run_watched(argv, env=dict(os.environ), log_path=tmp_path/"a.log", cwd=tmp_path, memory_limit_gib=0., poll=.05, grace=2.)
    assert over["stopped"] == "memory" and "PSS" in over["stop_reason"]
    floor = RUN.run_watched(argv, env=dict(os.environ), log_path=tmp_path/"b.log", cwd=tmp_path, available_floor_gib=1.5, available=lambda: .5, poll=.05, grace=2.)
    assert floor["stopped"] == "memory" and "MemAvailable" in floor["stop_reason"]


def test_run_watched_reports_a_failed_child(tmp_path):
    outcome = RUN.run_watched(child_script(tmp_path, "import sys\nsys.exit(3)\n"), env=dict(os.environ), log_path=tmp_path/"c.log", cwd=tmp_path, poll=.05)
    assert outcome["returncode"] == 3 and outcome["stopped"] is None


def test_gpu_reading():
    assert RUN.gpu_reading(run=lambda *a, **k: types.SimpleNamespace(stdout="1234, 87\n")) == (1234, 87)
    assert RUN.gpu_reading(run=lambda *a, **k: types.SimpleNamespace(stdout="[N/A], 87\n")) is None
    assert RUN.gpu_reading(run=lambda *a, **k: types.SimpleNamespace(stdout="")) is None


def test_run_watched_averages_the_gpu_utilization(tmp_path, monkeypatch):
    monkeypatch.setattr(RUN, "GPU_SAMPLE_SECONDS", 0.)
    readings = iter([(1000, 50), (3000, 90)])
    calls = []

    def condition():
        calls.append(1)
        return len(calls) >= 3

    outcome = RUN.run_watched(child_script(tmp_path, "import time\ntime.sleep(60)\n"), env=dict(os.environ), log_path=tmp_path/"g.log", cwd=tmp_path, should_stop=condition, sample_gpu=True, gpu=lambda: next(readings, None), poll=.05, grace=2.)
    assert outcome["gpu_peak_used_mib"] == 3000 and outcome["gpu_mean_utilization_percent"] == 70. and outcome["gpu_samples"] == 2


# ---- the gate ---------------------------------------------------------------------------------------------------------------------------------------------------

def gate_run(units="", gpu="20000, 3", apps=""):
    def run(argv, **kwargs):
        if argv[0] == "systemctl":
            return types.SimpleNamespace(returncode=0, stdout=units, stderr="")
        return types.SimpleNamespace(returncode=0, stdout=apps if "--query-compute-apps=pid" in argv else gpu, stderr="")
    return run


OPEN = dict(load=lambda: (.5, 0, 0), disk=lambda p: types.SimpleNamespace(free=500e9), available=lambda: 12.)


def test_gate_branches():
    assert RUN.gate_problems("tag-nn-train", run=gate_run(), **OPEN) == []
    assert RUN.gate_problems("tag-nn-train", run=gate_run(units="tag-nn-train.service loaded active running x\n"), **OPEN) == []
    assert "units active: tag-ul-train.service" in RUN.gate_problems("tag-nn-train", run=gate_run(units="tag-ul-train.service loaded active running x\n"), **OPEN)[0]
    assert "GPU free" in RUN.gate_problems("x", run=gate_run(gpu="12000, 3"), **OPEN)[0]
    assert "utilization" in RUN.gate_problems("x", run=gate_run(gpu="20000, 50"), **OPEN)[0]
    assert "compute processes" in RUN.gate_problems("x", run=gate_run(apps="123\n"), **OPEN)[0]
    assert "no reading" in RUN.gate_problems("x", run=gate_run(gpu="[N/A], 3"), **OPEN)[0]
    assert "load" in RUN.gate_problems("x", run=gate_run(), **dict(OPEN, load=lambda: (2., 0, 0)))[0]
    assert "MemAvailable" in RUN.gate_problems("x", run=gate_run(), **dict(OPEN, available=lambda: 8.))[0]
    assert any("/mnt/c" in p for p in RUN.gate_problems("x", run=gate_run(), **dict(OPEN, disk=lambda p: types.SimpleNamespace(free=500e9 if p == "/home" else 10*2**30))))
    assert any("/home" in p for p in RUN.gate_problems("x", run=gate_run(), **dict(OPEN, disk=lambda p: types.SimpleNamespace(free=50e9 if p == "/home" else 500e9))))


def test_waiting_for_the_gate_and_for_memory_stop():
    with pytest.raises(RUN.Stop, match="nothing started"):
        RUN.wait_for_gate("x", 0, problems=lambda own: ["closed"], sleep=lambda s: None)
    RUN.wait_for_gate("x", 0, problems=lambda own: [], sleep=lambda s: None)
    assert RUN.wait_for_memory(4., 0, available=lambda: 5.) == 5.
    with pytest.raises(RUN.Stop, match="stayed under"):
        RUN.wait_for_memory(4., 0, available=lambda: 3., sleep=lambda s: None)
    seen = iter([3., 3., 6.])
    assert RUN.wait_for_memory(4., 5, available=lambda: next(seen), sleep=lambda s: None) == 6.


# ---- the steps, with the child replaced -------------------------------------------------------------------------------------------------------------------------

@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    """The steps on an empty results root: the snapshot, the gate and the preprocessing checks pass, run_watched is replaced by what the test gives."""
    lock = tmp_path/"locks"/"nn_dev_read.lock"
    monkeypatch.setattr(nn_lock, "DEV_READ_LOCK", lock)
    monkeypatch.setattr(RUN, "require_snapshot", lambda: dict(kind="snapshot", commit="c"*40))
    monkeypatch.setattr(RUN, "require_preprocessed", lambda out, code=None: [])
    monkeypatch.setattr(RUN, "wait_for_gate", lambda *a, **k: None)
    monkeypatch.setattr(RUN, "env_record", lambda out, env: None)
    monkeypatch.setattr(RUN, "LOG_FILE", None)
    out = tmp_path/"out"
    return types.SimpleNamespace(out=out, lock=lock, args=types.SimpleNamespace(out=out, wait_minutes=0, resume=False, final_validation=False))


def outcome_of(**changes):
    return dict(dict(returncode=-15, stopped="condition", stop_reason="x", wall_seconds=3600., pid=1, group_left=[], host_peak_pss_gib=6., gpu_peak_used_mib=8000, gpu_mean_utilization_percent=70., gpu_samples=10, cgroup_peak_gib=7., log="l"), **changes)


def timed_lines(lines, *, hour=1):
    """Log lines a minute apart, in the format of print_to_log_file."""
    return "".join(f"2026-11-05 {hour:02d}:{i:02d}:03.123456: {line} \n" for i, line in enumerate(lines))


def fake_training(fold_dir, *, lines, checkpoints=(), results=None, log_name="training_log_x.txt", hour=1):
    def run(argv, *, env, log_path, cwd, should_stop=None, **kwargs):
        results_seen.append(dict(argv=argv, env=env, should_stop=should_stop, kwargs=kwargs))
        Path(fold_dir).mkdir(parents=True, exist_ok=True)
        (Path(fold_dir)/log_name).write_text(timed_lines(lines, hour=hour))
        for name in checkpoints:
            (Path(fold_dir)/name).write_bytes(b"weights")
        if should_stop is not None:
            results_seen[-1]["stop_now"] = should_stop()
        return outcome_of(**(results or {}))
    results_seen = []
    run.seen = results_seen
    return run


FOUR_EPOCHS = ["Epoch time: 120.0 s", "Epoch time: 100.0 s", "Epoch time: 110.0 s", "Epoch time: 90.0 s"]


def test_smoke_measures_and_the_rule_follows(stubbed, monkeypatch):
    fold_dir = NC.fold_directory(NC.TRAINERS[250], results=NC.smoke_results_root(stubbed.out))
    run = fake_training(fold_dir, lines=FOUR_EPOCHS, checkpoints=["checkpoint_best.pth"])
    monkeypatch.setattr(RUN, "run_watched", run)
    RUN.cmd_smoke(stubbed.args)
    measure = json.loads((stubbed.out/"smoke_measure.json").read_text())
    assert measure["epoch_seconds"] == [120., 100., 110., 90.] and measure["trainer"] == NC.TRAINERS[250] and measure["gpu_peak_used_mib"] == 8000 and measure["checkpoint_bytes"] == {"checkpoint_best.pth": 7}
    assert measure["n_proc_da"] == 6 and measure["n_proc_da_validation"] == 3 and measure["gpu_mean_utilization_percent"] == 70. and measure["gpu_samples"] == 10 and measure["host_peak_pss_gib"] == 6. and measure["cgroup_peak_gib"] == 7.
    assert run.seen[0]["env"]["nnUNet_results"] == str(NC.smoke_results_root(stubbed.out)) and run.seen[0]["env"]["nnUNet_n_proc_DA"] == "6" and run.seen[0]["stop_now"] is True and run.seen[0]["argv"] == NC.train_argv(NC.TRAINERS[250])
    with pytest.raises(RUN.Stop, match="the smoke ran"):
        RUN.cmd_smoke(stubbed.args)
    document = RUN.decide_epochs(stubbed.out, code=dict(commit="c"))
    assert document["trainer"] == NC.TRAINERS[250] and document["epoch_seconds_used"] == 100.


def test_smoke_stops_at_the_fourth_epoch_only(stubbed, monkeypatch):
    fold_dir = NC.fold_directory(NC.TRAINERS[250], results=NC.smoke_results_root(stubbed.out))
    run = fake_training(fold_dir, lines=FOUR_EPOCHS[:3], results=dict(returncode=1, stopped=None))
    monkeypatch.setattr(RUN, "run_watched", run)
    with pytest.raises(RUN.Stop, match="did not reach 4"):
        RUN.cmd_smoke(stubbed.args)
    assert run.seen[0]["stop_now"] is False


def test_smoke_that_did_not_reach_four_epochs_stops_and_keeps_the_record(stubbed, monkeypatch):
    fold_dir = NC.fold_directory(NC.TRAINERS[250], results=NC.smoke_results_root(stubbed.out))
    monkeypatch.setattr(RUN, "run_watched", fake_training(fold_dir, lines=["Epoch time: 120.0 s"], results=dict(returncode=1, stopped=None)))
    with pytest.raises(RUN.Stop, match="did not reach"):
        RUN.cmd_smoke(stubbed.args)
    assert (stubbed.out/"smoke_measure.json").is_file()
    assert not RUN.decide_epochs(stubbed.out, code={})["ok"]
    RUN.retire_smoke(stubbed.out, "the first smoke failed")                   # the way to run it again
    monkeypatch.setattr(RUN, "run_watched", fake_training(fold_dir, lines=FOUR_EPOCHS))
    RUN.cmd_smoke(stubbed.args)
    assert RUN.read_json(stubbed.out/"smoke_measure.json")["epoch_seconds"] == [120., 100., 110., 90.]


def test_a_smoke_that_died_without_a_measure_is_not_started_over(stubbed, monkeypatch):
    NC.smoke_results_root(stubbed.out).mkdir(parents=True)
    monkeypatch.setattr(RUN, "run_watched", lambda *a, **k: pytest.fail("no child may start"))
    with pytest.raises(RUN.Stop, match="retire-smoke"):
        RUN.cmd_smoke(stubbed.args)


def test_train_needs_the_decision_and_refuses_when_the_lock_is_there(stubbed, monkeypatch):
    monkeypatch.setattr(RUN, "run_watched", lambda *a, **k: pytest.fail("no child may start"))
    with pytest.raises(RUN.Stop, match="decide-epochs come first"):
        RUN.cmd_train(stubbed.args)
    write_decision(stubbed.out)
    nn_lock.take(dict(out_dir="a"))
    with pytest.raises(nn_lock.DevReadLocked):
        RUN.cmd_train(stubbed.args)


def test_train_stops_at_the_final_validation_and_records(stubbed, monkeypatch):
    write_decision(stubbed.out)
    (stubbed.out/"prepare_manifest.json").write_text(json.dumps(dict(label_contact_subjects=["a", "b"])))
    fold_dir = NC.fold_directory(NC.TRAINERS[250], results=NC.results_root(stubbed.out))
    run = fake_training(fold_dir, lines=["Epoch time: 100.0 s", "Epoch time: 90.0 s", "Epoch time: 80.0 s", "predicting OASIS_OAS1_0022_MR1"], checkpoints=[NC.CHECKPOINT])
    monkeypatch.setattr(RUN, "run_watched", run)
    RUN.cmd_train(stubbed.args)
    assert run.seen[0]["argv"] == NC.train_argv(NC.TRAINERS[250]) and run.seen[0]["stop_now"] is True and run.seen[0]["env"]["nnUNet_results"] == str(NC.results_root(stubbed.out)) and "CUDA_VISIBLE_DEVICES" not in run.seen[0]["env"] and run.seen[0]["kwargs"]["sample_gpu"] is True
    record = json.loads((stubbed.out/"train_record.json").read_text())
    assert record["trainer"] == NC.TRAINERS[250] and record["epochs"] == 250 and record["epochs_logged"] == 3 and record["epoch_seconds_median"] == 90. and record["stopped_before_final_validation"] is True and record["final_checkpoint_sha256"] == hashlib.sha256(b"weights").hexdigest() and record["n_label_contact"] == 2 and record["compile"] is False
    assert record["runs_started"] == 1 and record["resumed_runs"] == 0 and record["runs_without_record"] == 0 and record["wall_seconds"] == 180. and record["driver_wall_seconds_recorded"] == 3600. and record["gpu_mean_utilization_percent"] == 70.
    with pytest.raises(RUN.Stop, match="has finished"):
        RUN.cmd_train(stubbed.args)


def test_train_failure_keeps_the_run_record_and_says_how_to_go_on(stubbed, monkeypatch):
    write_decision(stubbed.out)
    fold_dir = NC.fold_directory(NC.TRAINERS[250], results=NC.results_root(stubbed.out))
    monkeypatch.setattr(RUN, "run_watched", fake_training(fold_dir, lines=["Epoch time: 100.0 s", "x"], results=dict(returncode=1, stopped=None), checkpoints=["checkpoint_latest.pth"], log_name="training_log_a.txt"))
    with pytest.raises(RUN.Stop, match="--continue"):
        RUN.cmd_train(stubbed.args)
    assert len(list((stubbed.out/"train_runs").glob("*.started.json"))) == 1 and len(list((stubbed.out/"train_runs").glob("*.json"))) == 2 and not (stubbed.out/"train_record.json").exists()
    with pytest.raises(RUN.Stop, match="a new training over it is refused"):       # without --continue, over the checkpoint of that run
        RUN.cmd_train(stubbed.args)
    time.sleep(1.1)                                                                # the runs are named by the second
    run = fake_training(fold_dir, lines=["Epoch time: 95.0 s", "predicting x"], checkpoints=[NC.CHECKPOINT], log_name="training_log_b.txt", hour=2)
    monkeypatch.setattr(RUN, "run_watched", run)
    RUN.cmd_train(types.SimpleNamespace(**dict(vars(stubbed.args), resume=True)))
    assert run.seen[0]["argv"][-1] == "--c"
    record = json.loads((stubbed.out/"train_record.json").read_text())
    assert record["resumed_runs"] == 1 and record["runs_started"] == 2 and record["runs_without_record"] == 0 and record["wall_seconds"] == 120. and record["epochs_logged"] == 2 and record["epoch_seconds_median"] == 97.5


def test_a_resumed_run_does_not_count_the_epochs_of_the_old_log(stubbed, monkeypatch):
    write_decision(stubbed.out)
    fold_dir = NC.fold_directory(NC.TRAINERS[250], results=NC.results_root(stubbed.out))
    fold_dir.mkdir(parents=True)
    (fold_dir/"checkpoint_latest.pth").write_bytes(b"x")
    old = fold_dir/"training_log_a.txt"
    old.write_text(timed_lines(["Epoch time: 1.0 s", "predicting x"]))
    os.utime(old, (time.time()-1.5, time.time()-1.5))                              # a resume soon after the crash
    monkeypatch.setattr(RUN, "run_watched", fake_training(fold_dir, lines=["Epoch time: 95.0 s"], checkpoints=[NC.CHECKPOINT], log_name="training_log_b.txt", results=dict(returncode=0, stopped=None)))
    RUN.cmd_train(types.SimpleNamespace(**dict(vars(stubbed.args), resume=True)))
    last = next(p for p in (stubbed.out/"train_runs").glob("*.json") if not p.name.endswith(".started.json"))
    assert json.loads(last.read_text())["epoch_seconds"] == [95.]


def test_a_run_killed_from_outside_is_counted_from_its_started_record_and_its_log(stubbed, monkeypatch):
    """RuntimeMaxSec or OOM: the driver dies with the child, only the started record and the log of that run are left."""
    write_decision(stubbed.out)
    fold_dir = NC.fold_directory(NC.TRAINERS[250], results=NC.results_root(stubbed.out))
    fold_dir.mkdir(parents=True)
    (stubbed.out/"train_runs").mkdir()
    (stubbed.out/"train_runs"/"20261105T010000Z.started.json").write_text(json.dumps(dict(trainer=NC.TRAINERS[250], epochs=250, resumed=False)))
    (fold_dir/"training_log_a.txt").write_text(timed_lines(["Epoch time: 100.0 s", "Epoch time: 100.0 s", "Epoch time: 100.0 s"]))
    (fold_dir/"checkpoint_latest.pth").write_bytes(b"x")
    os.utime(fold_dir/"training_log_a.txt", (1000, 1000))
    monkeypatch.setattr(RUN, "run_watched", fake_training(fold_dir, lines=["Epoch time: 90.0 s", "predicting x"], checkpoints=[NC.CHECKPOINT], log_name="training_log_b.txt", hour=3))
    RUN.cmd_train(types.SimpleNamespace(**dict(vars(stubbed.args), resume=True)))
    record = json.loads((stubbed.out/"train_record.json").read_text())
    assert record["runs_started"] == 2 and record["resumed_runs"] == 1 and record["runs_without_record"] == 1 and record["epochs_logged"] == 4 and "counts again" in record["epochs_logged_note"] and record["wall_seconds"] == 120.+60. and record["driver_wall_seconds_recorded"] == 3600.
    summary = DECIDE.summary_markdown(DECIDE.decide(made_evaluation(.001), record, dict(MADE_INFER, checkpoint_sha256=record["final_checkpoint_sha256"])))
    assert "1 次运行没有写完记录" in summary


def test_check_resume(tmp_path):
    with pytest.raises(RUN.Stop, match="no checkpoint"):
        RUN.check_resume(tmp_path, True)
    assert RUN.check_resume(tmp_path, False) == []
    (tmp_path/"checkpoint_latest.pth").write_bytes(b"x")
    assert RUN.check_resume(tmp_path, True) == ["checkpoint_latest.pth"]
    with pytest.raises(RUN.Stop, match="refused"):
        RUN.check_resume(tmp_path, False)


def test_final_validation_can_be_kept(stubbed, monkeypatch):
    write_decision(stubbed.out, 200.)
    fold_dir = NC.fold_directory(NC.TRAINERS[100], results=NC.results_root(stubbed.out))
    run = fake_training(fold_dir, lines=["Epoch time: 100.0 s"], checkpoints=[NC.CHECKPOINT], results=dict(returncode=0, stopped=None))
    monkeypatch.setattr(RUN, "run_watched", run)
    RUN.cmd_train(types.SimpleNamespace(**dict(vars(stubbed.args), final_validation=True)))
    assert run.seen[0]["should_stop"] is None and json.loads((stubbed.out/"train_record.json").read_text())["stopped_before_final_validation"] is False


def test_dev_input_holds_links_to_images_only(tmp_path):
    data = tmp_path/"data"
    (data/DEV_ONE).mkdir(parents=True)
    (data/DEV_ONE/"aligned_norm.nii.gz").write_bytes(b"image")
    RUN.build_dev_input(tmp_path/"input", [DEV_ONE], data)
    entries = list((tmp_path/"input").iterdir())
    assert [p.name for p in entries] == [f"{DEV_ONE}_0000.nii.gz"] and entries[0].is_symlink() and Path(os.readlink(entries[0])) == data/DEV_ONE/"aligned_norm.nii.gz"
    RUN.build_dev_input(tmp_path/"input", [DEV_ONE], data)                     # the same links: accepted
    (tmp_path/"input"/"extra.nii.gz").write_bytes(b"x")
    with pytest.raises(RUN.Stop, match="something else"):
        RUN.build_dev_input(tmp_path/"input", [DEV_ONE], data)
    with pytest.raises(RUN.Stop, match="no such file"):
        RUN.build_dev_input(tmp_path/"other", ["OASIS_OAS1_0999_MR1"], data)


DEV_ONE = "OASIS_OAS1_0951_MR1"


def test_infer_starts_from_the_finished_training_only(stubbed, monkeypatch):
    with pytest.raises(RUN.Stop, match="has not finished"):
        RUN.cmd_infer(stubbed.args)
    stubbed.out.mkdir(parents=True, exist_ok=True)
    final = stubbed.out/"final.pth"
    final.write_bytes(b"weights")
    (stubbed.out/"train_record.json").write_text(json.dumps(dict(trainer=NC.TRAINERS[250], final_checkpoint=str(final), final_checkpoint_sha256="0"*64)))
    monkeypatch.setattr(RUN, "admitted_dev_subjects", lambda out, code: [DEV_ONE])
    monkeypatch.setattr(RUN, "run_watched", lambda *a, **k: pytest.fail("no child may start"))
    with pytest.raises(RUN.Stop, match="sha256 differs"):
        RUN.cmd_infer(stubbed.args)
    (stubbed.out/"train_record.json").write_text(json.dumps(dict(trainer=NC.TRAINERS[250], final_checkpoint=str(final), final_checkpoint_sha256=hashlib.sha256(b"weights").hexdigest())))
    seen = {}
    monkeypatch.setattr(RUN, "build_dev_input", lambda directory, subjects, root: seen.update(subjects=subjects))

    def fake(argv, **kwargs):
        seen.update(argv=argv, env=kwargs["env"])
        return outcome_of(returncode=0, stopped=None)
    monkeypatch.setattr(RUN, "run_watched", fake)
    RUN.cmd_infer(stubbed.args)
    assert seen["subjects"] == [DEV_ONE] and seen["argv"][:3] == [str(NC.NN_PY), "-I", "-B"] and "CUDA_VISIBLE_DEVICES" not in seen["env"] and str(stubbed.out/"pred_dev") in seen["argv"]
    with pytest.raises(RUN.Stop, match="the inference ran"):
        RUN.cmd_infer(stubbed.args)


def test_infer_refuses_without_the_links(stubbed, monkeypatch):
    stubbed.out.mkdir(parents=True, exist_ok=True)
    (stubbed.out/"train_record.json").write_text(json.dumps(dict(trainer=NC.TRAINERS[250], final_checkpoint="x", final_checkpoint_sha256="0"*64)))

    def refuse(out, code=None):
        raise PREP.RawRefused("2 are not symbolic links")

    monkeypatch.setattr(RUN, "require_preprocessed", refuse)
    monkeypatch.setattr(RUN, "admitted_dev_subjects", lambda out, code: pytest.fail("refused before this"))
    monkeypatch.setattr(RUN, "run_watched", lambda *a, **k: pytest.fail("no child may start"))
    with pytest.raises(PREP.RawRefused, match="symbolic links"):
        RUN.cmd_infer(stubbed.args)


def test_admitted_dev_subjects_on_the_real_lists(tmp_path):
    (tmp_path/"prepare_manifest.json").write_text(json.dumps(dict(label_contact_subjects=PREP.read_train_list())))
    subjects = RUN.admitted_dev_subjects(tmp_path, code=code_record())
    assert len(subjects) == 20 and subjects[0] == "OASIS_OAS1_0438_MR1" and subjects[-1] == "OASIS_OAS1_0457_MR1"
    (tmp_path/"prepare_manifest.json").write_text(json.dumps(dict(label_contact_subjects=["OASIS_OAS1_0438_MR1"])))
    with pytest.raises(SubjectSealed, match="label contact"):
        RUN.admitted_dev_subjects(tmp_path, code=code_record())
    (tmp_path/"prepare_manifest.json").unlink()
    with pytest.raises(SubjectSealed, match="missing"):
        RUN.admitted_dev_subjects(tmp_path, code=code_record())


def test_preprocessed_identifiers(tmp_path):
    folder = RUN.preprocessed_data_directory(tmp_path)
    folder.mkdir(parents=True)
    for name in ("b.pkl", "b.b2nd", "b_seg.b2nd", "a.pkl"):
        (folder/name).write_bytes(b"")
    assert RUN.preprocessed_identifiers(tmp_path) == ["a", "b"]


# ---- the sandbox: real steps stop here, the dry run -----------------------------------------------------------------------------------------------------------------

@pytest.fixture
def blocked_env(tmp_path):
    bin_dir = tmp_path/"bin"
    bin_dir.mkdir()
    shim = bin_dir/"systemd-run"
    marker = tmp_path/"systemd_run_called"
    shim.write_text(f"#!/bin/sh\necho called >> {marker}\nexit 99\n")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    return dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1"), marker


@pytest.mark.parametrize("step", ["smoke", "decide-epochs", "train", "infer", "eval"])
def test_real_steps_stop_outside_a_snapshot(tmp_path, blocked_env, step):
    env, marker = blocked_env
    result = subprocess.run([sys.executable, "-B", str(NN/"run_nn.py"), step, "--out", str(tmp_path/"out")], env=env, capture_output=True, text=True, timeout=180, cwd=str(CODE_ROOT))
    assert result.returncode == 8 and "STOP" in result.stdout and ("snapshot" in result.stdout or "clean" in result.stdout)
    assert not marker.exists() and not (tmp_path/"out").exists()


def test_print_launch_starts_nothing(blocked_env):
    env, marker = blocked_env
    result = subprocess.run([sys.executable, "-B", str(NN/"run_nn.py"), "print-launch", "train"], env=env, capture_output=True, text=True, timeout=180, cwd=str(CODE_ROOT))
    assert result.returncode == 0 and result.stdout.startswith("systemd-run --user --unit=tag-nn-train") and "pytest" not in result.stdout
    assert shlex.split(result.stdout) == RUN.launch_argv("train") and not marker.exists()


def test_dry_run_end_to_end(tmp_path, blocked_env):
    env, marker = blocked_env
    result = subprocess.run([sys.executable, "-B", str(NN/"run_nn.py"), "--dry-run", "--scratch", str(tmp_path/"work")], env=env, capture_output=True, text=True, timeout=300, cwd=str(CODE_ROOT))
    assert result.returncode == 0, result.stdout[-3000:]+result.stderr[-3000:]
    assert "dry run ok" in result.stdout and "--unit=tag-nn-smoke" in result.stdout and "--unit=tag-nn-train" in result.stdout and "--unit=tag-nn-infer" in result.stdout and "--unit=tag-nn-eval" in result.stdout
    assert "pytest" not in result.stdout.replace(str(tmp_path), "") and not marker.exists()


def test_the_evaluation_step_runs_eval_then_decide(eval_tree, stubbed, monkeypatch):
    """cmd_eval with the two real scripts as children, the synthetic switch added to the first: what comes out of one goes into the other."""
    out = stubbed.out
    (out/"pred_dev").mkdir(parents=True)
    for subject in TREE.DEV:
        shutil.copy(eval_tree/"pred_nn"/f"{subject}.nii.gz", out/"pred_dev"/f"{subject}.nii.gz")
    shutil.copy(eval_tree/"manifest.json", out/"prepare_manifest.json")
    (out/"train_record.json").write_text(json.dumps(MADE_TRAIN))
    (out/"pred_dev"/"infer_record.json").write_text(json.dumps(MADE_INFER))
    extra = ["--synthetic", "--v2-root", str(eval_tree/"pred_v1"), "--control-segmenter", "unet35_v1_ens", "--reference", str(eval_tree/"reference.json"), "--allowlist", str(eval_tree/"dev_subjects.json"), "--folds", str(eval_tree/"folds.json"), "--dev-source", "synthetic_dev", "--data-root", str(eval_tree/"data")]
    eval_original, decide_original = RUN.eval_argv, RUN.decide_argv
    monkeypatch.setattr(RUN, "eval_argv", lambda o, bz_go=None: eval_original(o, bz_go=bz_go)+extra)
    monkeypatch.setattr(RUN, "decide_argv", lambda o: decide_original(o, u1_eval=o/"no_u1.json"))
    RUN.cmd_eval(types.SimpleNamespace(out=out, bz_go=None))
    decision = json.loads((out/"analysis"/"decision_nn.json").read_text(encoding="utf-8"))
    assert decision["reading"] == "not_decidable" and decision["not_decidable_because"] == ["2 名受试者，不是 20"] and (out/"eval_nn"/"eval_nn.json").is_file() and (out/"analysis"/"summary_nn.md").is_file() and not stubbed.lock.exists()


def test_a_failed_evaluation_does_not_reach_the_rule(stubbed, monkeypatch):
    out = stubbed.out
    out.mkdir(parents=True)
    monkeypatch.setattr(RUN, "eval_argv", lambda o, bz_go=None: [sys.executable, "-I", "-c", "import sys; sys.exit(3)"])
    marker = out/"the_rule_ran"
    monkeypatch.setattr(RUN, "decide_argv", lambda o: [sys.executable, "-I", "-c", f"open({str(marker)!r}, 'w').close()"])
    with pytest.raises(RUN.Stop, match="eval_nn failed"):
        RUN.cmd_eval(types.SimpleNamespace(out=out, bz_go=None))
    assert not marker.exists()


# ---- the read lock (as test_ul) -------------------------------------------------------------------------------------------------------------------------------------

@pytest.fixture
def lock(tmp_path, monkeypatch):
    path = tmp_path/"locks"/"nn_dev_read.lock"
    monkeypatch.setattr(nn_lock, "DEV_READ_LOCK", path)
    return path


def test_lock_path_is_fixed_and_outside_every_results_root():
    path = nn_lock.DEV_READ_LOCK
    assert path == Path("/home/rmcfz/seq_vp_tag_results/seg/nn_dev_read.lock") and NC.OUT not in path.parents
    assert all("--lock" not in (NN/name).read_text(encoding="utf-8") for name in ("eval_nn.py", "run_nn.py"))


def test_lock_functions(lock):
    nn_lock.check()
    with pytest.raises(ValueError, match="nothing to unlock"):
        nn_lock.check("BZ, some day")
    first = nn_lock.take(dict(out_dir="a"))
    assert first["event"] == "first_read" and lock.is_file()
    with pytest.raises(nn_lock.DevReadLocked, match="read once"):
        nn_lock.check()
    with pytest.raises(nn_lock.DevReadLocked, match="not read"):
        nn_lock.take(dict(out_dir="b"))
    assert [r["out_dir"] for r in nn_lock.describe()] == ["a"]
    for blank in ("", "  "):
        with pytest.raises(ValueError, match="words"):
            nn_lock.check(blank)
    nn_lock.check("BZ, 11-05 9 am: read it again")
    second = nn_lock.take(dict(out_dir="c"), bz_go="BZ, 11-05 9 am: read it again")
    assert second["event"] == "bz_approved_reread" and [r["out_dir"] for r in nn_lock.describe()] == ["a", "c"] and nn_lock.describe()[1]["bz_go"].startswith("BZ")


# ---- the evaluation ------------------------------------------------------------------------------------------------------------------------------------------------------

def write_prediction(directory, subject, data, *, values=None, affine=None):
    Path(directory).mkdir(parents=True, exist_ok=True)
    truth = nib.load(Path(data)/subject/"aligned_seg35.nii.gz")
    array = np.asarray(truth.dataobj).astype(np.uint8) if values is None else values
    nib.save(nib.Nifti1Image(array, truth.affine if affine is None else affine), Path(directory)/f"{subject}.nii.gz")


@pytest.fixture(scope="module")
def eval_tree(tmp_path_factory):
    """The synthetic tree with a reference in the layout of the real eval_seg.json, predictions, and the manifest."""
    work = tmp_path_factory.mktemp("nn_eval")
    TREE.build(work)
    lookup = SEG.load_seg4_lookup()
    rows = {}
    for subject in TREE.DEV:
        top, _ = SEG.read_prediction(work/"pred_v1", subject, expected_segmenter="unet35_v1_ens")
        row = SEG.summarize_subject(top, SEG.read_truth(work/"data", subject, "seg35")[0], SEG.read_truth(work/"data", subject, "seg4")[0], lookup)
        rows[subject] = dict(v1=row, v2=row)
        write_prediction(work/"pred_nn", subject, work/"data")
    means = {arm: {m: float(np.mean([rows[s][arm][m] for s in TREE.DEV])) for m in ("seg35", "seg4")} for arm in ("v1", "v2")}
    (work/"reference.json").write_text(json.dumps(SEG.no_nan(dict(subjects=TREE.DEV, rows=rows, means=means))))
    (work/"manifest.json").write_text(json.dumps(dict(label_contact_subjects=TREE.TRAINING)))
    return work


def eval_kwargs(work, out_dir, **changes):
    return dict(dict(pred_root=work/"pred_nn", manifest=work/"manifest.json", out_dir=out_dir, v2_root=work/"pred_v1", control_segmenter="unet35_v1_ens", reference_path=work/"reference.json", data_root=work/"data", allowlist_path=work/"dev_subjects.json", folds_path=work/"folds.json", dev_source="synthetic_dev", synthetic=True, argv=["test"], log=quiet), **changes)


@pytest.fixture
def real_like(eval_tree, lock, monkeypatch):
    """The real path of evaluate on the synthetic tree: only the pins point at it. No real subject, no file of a real subject."""
    monkeypatch.setattr(EVAL, "REFERENCE", eval_tree/"reference.json")
    monkeypatch.setattr(EVAL, "REFERENCE_SHA256", hashlib.sha256((eval_tree/"reference.json").read_bytes()).hexdigest())
    monkeypatch.setattr(SEG, "ALLOWLIST", eval_tree/"dev_subjects.json")
    monkeypatch.setattr(SEG, "DEV_SOURCE", "synthetic_dev")
    monkeypatch.setattr(SEG, "FOLDS", eval_tree/"folds.json")
    monkeypatch.setattr(SEG, "DATA_ROOT", eval_tree/"data")
    monkeypatch.setattr(EVAL, "MANIFEST", eval_tree/"manifest.json")
    monkeypatch.setattr(EVAL, "V2_ROOT", eval_tree/"pred_v1")
    monkeypatch.setattr(EVAL, "V2_SEGMENTER", "unet35_v1_ens")
    monkeypatch.setattr(EVAL, "LR_PAIRS", ())                                  # the synthetic labels are not anatomical; test_left_right_check has the check itself
    monkeypatch.setattr(EVAL, "load_pinned_allowlist", lambda path: (load_dev_allowlist(path), dict(allowlist_path=str(path))))
    monkeypatch.setattr(EVAL, "require_clean_code", lambda code: code)
    return lock


def real_kwargs(work, out_dir, **changes):
    return eval_kwargs(work, out_dir, synthetic=False, **changes)


def test_synthetic_evaluation(eval_tree, lock, tmp_path):
    document = EVAL.evaluate(**eval_kwargs(eval_tree, tmp_path/"e1"))
    assert document["subjects"] == TREE.DEV and document["control"]["passed"] is True and document["dev_read"] is None and not lock.exists()
    means = document["means"]
    assert means["nn"]["seg35"] == pytest.approx(1.) and set(means) == {"nn", "v2_control", "reference_v2", "reference_v1"} and set(means["nn"]) == set(EVAL.METRICS)
    row = document["rows"]["nn"][TREE.DEV[0]]
    assert row["cortex"] == row["seg4_per_label"][0] and row["white_matter"] == row["seg4_per_label"][2]
    assert "cortex" in (tmp_path/"e1"/"per_subject.csv").read_text().splitlines()[0] and (tmp_path/"e1"/"per_label.csv").is_file()
    assert document["paired"]["nn_minus_reference_v2"]["seg35"]["n"] == 2
    with pytest.raises(FileExistsError, match="never overwritten"):
        EVAL.evaluate(**eval_kwargs(eval_tree, tmp_path/"e1"))


def test_focus_subject_is_reported_on_its_own(eval_tree, tmp_path, monkeypatch):
    monkeypatch.setattr(EVAL, "FOCUS_SUBJECT", TREE.DEV[1])
    focus = EVAL.evaluate(**eval_kwargs(eval_tree, tmp_path/"e1"))["focus"]
    assert focus["subject"] == TREE.DEV[1] and set(focus["row"]) == {"nn", "v2_control", "reference_v2", "reference_v1"} and focus["row"]["nn"]["seg35"] == pytest.approx(1.) and "seg4" in focus["others_mean"]["nn"]


def test_the_real_constants_of_the_evaluation():
    assert EVAL.MIN_FOREGROUND_ON_IMAGE == .95 and EVAL.LR_PAIRS == ((1, 20), (2, 21)) and EVAL.MANIFEST == NC.OUT/"prepare_manifest.json" and EVAL.FOCUS_SUBJECT == "OASIS_OAS1_0440_MR1" and EVAL.V2_SEGMENTER == "unet35_v2_full" and EVAL.REFERENCE == Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v2_full/eval_seg/eval_seg.json") and SEG.DEV_SOURCE == "stage_L2R_val_v1"
    assert EVAL.REFERENCE.parent == EVAL.V2_ROOT.parent/"eval_seg"


def test_a_prediction_with_errors_changes_the_numbers(eval_tree, tmp_path):
    subject = TREE.DEV[0]
    root = tmp_path/"pred"
    labels = np.asarray(nib.load(eval_tree/"data"/subject/"aligned_seg35.nii.gz").dataobj).astype(np.uint8)
    labels[labels == 2] = 1
    write_prediction(root, subject, eval_tree/"data", values=labels)
    write_prediction(root, TREE.DEV[1], eval_tree/"data")
    document = EVAL.evaluate(**eval_kwargs(eval_tree, tmp_path/"e1", pred_root=root))
    assert document["rows"]["nn"][subject]["seg35"] < 1. and document["rows"]["nn"][TREE.DEV[1]]["seg35"] == pytest.approx(1.)


def test_the_control_catches_a_reference_that_differs(eval_tree, tmp_path):
    reference = json.loads((eval_tree/"reference.json").read_text())
    for subject in TREE.DEV:
        reference["rows"][subject]["v2"]["seg35"] -= .01
    reference["means"]["v2"]["seg35"] -= .01
    (tmp_path/"reference.json").write_text(json.dumps(reference))
    document = EVAL.evaluate(**eval_kwargs(eval_tree, tmp_path/"e1", reference_path=tmp_path/"reference.json"))
    assert document["control"]["passed"] is False and document["control"]["differences"]["seg35"] == pytest.approx(.01)


def test_reference_is_pinned_and_its_means_must_follow_from_its_rows(eval_tree, tmp_path):
    with pytest.raises(ValueError, match="pinned"):
        EVAL.read_reference(eval_tree/"reference.json", pinned_sha256="0"*64)
    reference = json.loads((eval_tree/"reference.json").read_text())
    reference["means"]["v2"]["seg35"] += .01
    (tmp_path/"reference.json").write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="does not follow"):
        EVAL.read_reference(tmp_path/"reference.json")


def test_geometry_and_labels_of_a_prediction_are_checked(eval_tree, tmp_path):
    subject, data = TREE.DEV[0], eval_tree/"data"
    truth = nib.load(data/subject/"aligned_seg35.nii.gz")
    moved = truth.affine.copy()
    moved[0, 3] += 1.
    write_prediction(tmp_path/"a", subject, data, affine=moved)
    with pytest.raises(ValueError, match="affine differs"):
        EVAL.read_nn_prediction(tmp_path/"a", data, subject)
    write_prediction(tmp_path/"b", subject, data, values=np.zeros((8, 8, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="shape"):
        EVAL.read_nn_prediction(tmp_path/"b", data, subject)
    write_prediction(tmp_path/"c", subject, data, values=np.full(truth.shape, 36, dtype=np.uint8))
    with pytest.raises(ValueError, match="not labels"):
        EVAL.read_nn_prediction(tmp_path/"c", data, subject)
    write_prediction(tmp_path/"d", subject, data, values=np.zeros(truth.shape, dtype=np.float32))
    with pytest.raises(ValueError, match="not labels"):
        EVAL.read_nn_prediction(tmp_path/"d", data, subject)


def lateral_top(shape=(40, 40, 60), *, swapped=False):
    """An argmax in the order of load_scalar: left labels at low x (the last axis), right labels at high x."""
    top = np.zeros(shape, dtype=np.uint8)
    left, right = (slice(5, 25), slice(35, 55)) if not swapped else (slice(35, 55), slice(5, 25))
    top[5:20, 5:35, left], top[20:35, 5:35, left] = 1, 2
    top[5:20, 5:35, right], top[20:35, 5:35, right] = 20, 21
    return top


def test_left_right_check():
    assert EVAL.left_right_check(lateral_top()) == pytest.approx(30.) and EVAL.left_right_check(lateral_top(), ()) is None
    with pytest.raises(ValueError, match="expected the left at low x"):
        EVAL.left_right_check(lateral_top(swapped=True))
    with pytest.raises(ValueError, match="expected the left at low x"):        # another axis separates them most: axes permuted
        EVAL.left_right_check(lateral_top().transpose(2, 1, 0))
    empty = lateral_top()
    empty[empty == 21] = 0
    with pytest.raises(ValueError, match="no voxels"):
        EVAL.left_right_check(empty)


def write_real_axes(path, values):
    volume = nib.Nifti1Image(values, REAL_AFFINE)
    volume.set_qform(REAL_AFFINE, code=1)
    volume.set_sform(REAL_AFFINE, code=1)
    volume.header.set_xyzt_units("mm")
    nib.save(volume, path)


def native_of(top):
    """An array in the order of load_scalar back to the file's axes for REAL_AFFINE, by nibabel's own inverse."""
    transform = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt(("R", "A", "S")), nib.orientations.io_orientation(REAL_AFFINE))
    return nib.orientations.apply_orientation(top.transpose(2, 1, 0), transform)


def test_read_prediction_checks_the_orientation_without_a_truth(tmp_path):
    subject = "OASIS_OAS1_0961_MR1"
    (tmp_path/"data"/subject).mkdir(parents=True)
    (tmp_path/"pred").mkdir()
    top = lateral_top()
    write_real_axes(tmp_path/"data"/subject/"aligned_norm.nii.gz", native_of((top > 0).astype(np.float32)))
    write_real_axes(tmp_path/"pred"/f"{subject}.nii.gz", native_of(top))
    read, record = EVAL.read_nn_prediction(tmp_path/"pred", tmp_path/"data", subject)
    assert np.array_equal(read, top) and record["foreground_on_image"] == 1. and record["smallest_lr_separation_vox"] == pytest.approx(30.)
    write_real_axes(tmp_path/"pred"/f"{subject}.nii.gz", native_of(lateral_top(swapped=True)))   # a flip left and right: the foreground still lies on the image
    with pytest.raises(ValueError, match="expected the left at low x"):
        EVAL.read_nn_prediction(tmp_path/"pred", tmp_path/"data", subject)
    write_real_axes(tmp_path/"data"/subject/"aligned_norm.nii.gz", native_of(np.roll(top > 0, 15, axis=0).astype(np.float32)))     # a prediction that does not sit on its image
    write_real_axes(tmp_path/"pred"/f"{subject}.nii.gz", native_of(top))
    assert EVAL.read_nn_prediction(tmp_path/"pred", tmp_path/"data", subject, min_on_image=0.)[1]["foreground_on_image"] < EVAL.MIN_FOREGROUND_ON_IMAGE
    with pytest.raises(ValueError, match="orientation"):
        EVAL.read_nn_prediction(tmp_path/"pred", tmp_path/"data", subject)
    write_real_axes(tmp_path/"data"/subject/"aligned_norm.nii.gz", native_of(np.roll(top > 0, 2, axis=0).astype(np.float32)))      # between the old .9 and .95: the flips measured in the review fall here or lower
    assert .9 < EVAL.read_nn_prediction(tmp_path/"pred", tmp_path/"data", subject, min_on_image=0.)[1]["foreground_on_image"] < .95
    with pytest.raises(ValueError, match="orientation"):
        EVAL.read_nn_prediction(tmp_path/"pred", tmp_path/"data", subject)


def test_orientation_equals_load_scalar_on_the_real_axes(tmp_path):
    rng = np.random.default_rng(0)
    values = rng.integers(0, 36, (8, 9, 10), dtype=np.uint8)
    volume = nib.Nifti1Image(values, REAL_AFFINE)
    volume.set_qform(REAL_AFFINE, code=1)
    volume.set_sform(REAL_AFFINE, code=1)
    volume.header.set_xyzt_units("mm")
    nib.save(volume, tmp_path/"v.nii.gz")
    assert np.array_equal(EVAL.orient_like_load_scalar(values, REAL_AFFINE), load_scalar(tmp_path/"v.nii.gz", labels=True).values.astype(np.uint8))


def test_t0_sealed_and_label_contact_are_refused_before_any_file(eval_tree, tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(EVAL, "read_nn_prediction", lambda *a, **k: opened.append(a))
    monkeypatch.setattr(SEG, "read_truth", lambda *a, **k: opened.append(a))
    with pytest.raises(SubjectSealed):
        EVAL.evaluate(**eval_kwargs(eval_tree, tmp_path/"e1", dev_source="synthetic_t0"))
    (tmp_path/"manifest.json").write_text(json.dumps(dict(label_contact_subjects=[TREE.DEV[0]])))
    with pytest.raises(SubjectSealed, match="label contact"):
        EVAL.evaluate(**eval_kwargs(eval_tree, tmp_path/"e2", manifest=tmp_path/"manifest.json"))
    assert not opened and not (tmp_path/"e1").exists() and not (tmp_path/"e2").exists()


@pytest.mark.parametrize("changes", [dict(dev_source="other"), dict(reference_path="other.json"), dict(folds_path="other.json"), dict(manifest="other.json"), dict(data_root="other"), dict(v2_root="other"), dict(control_segmenter="other"), dict(allowlist_path="other.json")])
def test_real_path_refuses_other_inputs(eval_tree, real_like, tmp_path, changes):
    with pytest.raises(ValueError, match="the plan's"):
        EVAL.evaluate(**real_kwargs(eval_tree, tmp_path/"e1", **changes))
    assert not real_like.exists()


def test_real_path_locks_before_the_first_truth_and_refuses_any_new_directory(eval_tree, real_like, tmp_path, monkeypatch):
    original = SEG.read_truth
    states = []
    monkeypatch.setattr(SEG, "read_truth", lambda *a, **k: (states.append(real_like.exists()), original(*a, **k))[1])
    document = EVAL.evaluate(**real_kwargs(eval_tree, tmp_path/"e1"))
    assert states and all(states)
    lines = nn_lock.describe()
    assert len(lines) == 1 and lines[0]["event"] == "first_read" and lines[0]["out_dir"] == str(tmp_path/"e1") and document["dev_read"] == lines[0]
    before = real_like.read_bytes()
    opened = []
    monkeypatch.setattr(SEG, "read_truth", lambda *a, **k: opened.append(a))
    monkeypatch.setattr(EVAL, "read_nn_prediction", lambda *a, **k: opened.append(a))
    with pytest.raises(nn_lock.DevReadLocked, match="read once"):
        EVAL.evaluate(**real_kwargs(eval_tree, tmp_path/"e2"))
    with pytest.raises(nn_lock.DevReadLocked, match="read once"):
        EVAL.evaluate(**real_kwargs(eval_tree, tmp_path/"e3", pred_root=tmp_path/"another_root"))
    assert not opened and not (tmp_path/"e2").exists() and not (tmp_path/"e3").exists() and real_like.read_bytes() == before


BEFORE_THE_LOCK = dict(missing=(FileNotFoundError, "No such file"), geometry=(ValueError, "shape"), control_reader=(ValueError, "the stored prediction is broken"), control_file=(ProvenanceRejected, "sha256 differs"), control_provenance=(ValueError, "the provenance is not of the structure layer"))   # each fails for its own reason, not because a pin was missed


@pytest.mark.parametrize("breakage", sorted(BEFORE_THE_LOCK))
def test_failure_before_the_first_truth_leaves_no_lock(eval_tree, real_like, tmp_path, monkeypatch, breakage):
    root = tmp_path/"pred"
    for subject in TREE.DEV:
        write_prediction(root, subject, eval_tree/"data")
    if breakage == "missing":
        (root/f"{TREE.DEV[1]}.nii.gz").unlink()
    elif breakage == "geometry":
        write_prediction(root, TREE.DEV[1], eval_tree/"data", values=np.zeros((8, 8, 8), dtype=np.uint8))
    elif breakage in ("control_file", "control_provenance"):                   # the second subject: the check runs before the lock, not when the loop reaches it
        control = tmp_path/"control"
        shutil.copytree(eval_tree/"pred_v1", control)
        monkeypatch.setattr(EVAL, "V2_ROOT", control)
        directory = control/TREE.DEV[1]
        if breakage == "control_file":
            with (directory/"probs36.npz").open("ab") as stream:
                stream.write(b"broken")
        else:
            document = json.loads((directory/"provenance.json").read_text())
            (directory/"provenance.json").write_text(json.dumps(dict(document, segmenter_id="another")))
    kwargs = dict(v2_root=EVAL.V2_ROOT)
    with monkeypatch.context() as patch:
        if breakage == "control_reader":
            def broken(*args, **kwargs):
                raise ValueError("the stored prediction is broken")
            patch.setattr(SEG, "read_prediction", broken)
        kind, pattern = BEFORE_THE_LOCK[breakage]
        with pytest.raises(kind, match=pattern):
            EVAL.evaluate(**real_kwargs(eval_tree, tmp_path/"e1", pred_root=root, **kwargs))
    assert not real_like.exists()


def test_failure_after_the_first_truth_keeps_the_lock(eval_tree, real_like, tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("the truth file failed")

    with monkeypatch.context() as patch:
        patch.setattr(SEG, "read_truth", broken)
        with pytest.raises(OSError, match="truth file"):
            EVAL.evaluate(**real_kwargs(eval_tree, tmp_path/"e1"))
    assert real_like.is_file()
    with pytest.raises(nn_lock.DevReadLocked):
        EVAL.evaluate(**real_kwargs(eval_tree, tmp_path/"e2"))


def test_bz_go_reads_again_and_is_recorded(eval_tree, real_like, tmp_path):
    with pytest.raises(ValueError, match="nothing to unlock"):
        EVAL.evaluate(**real_kwargs(eval_tree, tmp_path/"e0", bz_go="BZ, some day"))
    EVAL.evaluate(**real_kwargs(eval_tree, tmp_path/"e1"))
    with pytest.raises(ValueError, match="words"):
        EVAL.evaluate(**real_kwargs(eval_tree, tmp_path/"e2", bz_go="  "))
    words = "BZ, 11-05 9 am: read it again"
    document = EVAL.evaluate(**real_kwargs(eval_tree, tmp_path/"e3", bz_go=words))
    assert document["dev_read"]["event"] == "bz_approved_reread" and document["dev_read"]["bz_go"] == words and [r["event"] for r in nn_lock.describe()] == ["first_read", "bz_approved_reread"]
    summary = DECIDE.summary_markdown(DECIDE.decide(dict(made_evaluation(.006), dev_read=document["dev_read"]), MADE_TRAIN, MADE_INFER))
    assert words in summary and "再次读取" in summary


def test_synthetic_path_does_not_touch_the_lock(eval_tree, lock, tmp_path):
    lock.parent.mkdir(parents=True)
    lock.write_text("a lock of a real read\n")
    assert EVAL.evaluate(**eval_kwargs(eval_tree, tmp_path/"e1"))["dev_read"] is None and lock.read_text() == "a lock of a real read\n"
    with pytest.raises(ValueError, match="synthetic"):
        EVAL.evaluate(**eval_kwargs(eval_tree, tmp_path/"e2", bz_go="BZ"))


def test_synthetic_switch_cannot_reach_the_real_data_root(eval_tree, tmp_path):
    with pytest.raises(Exception, match="(?i)data root|real"):
        EVAL.evaluate(**eval_kwargs(eval_tree, tmp_path/"e1", data_root=Path("/mnt/d/data")))


def test_main_returns_8_when_locked(eval_tree, real_like, tmp_path, capsys):
    real_like.parent.mkdir(parents=True, exist_ok=True)
    real_like.write_text(json.dumps(dict(utc="u", out_dir="o"))+"\n")
    assert EVAL.main(["--pred-root", str(eval_tree/"pred_nn"), "--manifest", str(eval_tree/"manifest.json"), "--out-dir", str(tmp_path/"e")]) == 8 and "STOP" in capsys.readouterr().out and not (tmp_path/"e").exists()


# ---- the rule ------------------------------------------------------------------------------------------------------------------------------------------------------------

REFERENCE_MEANS = dict(seg35=.9275, seg4=.9591, small_six=.82, big_six=.957, other_23=.948, cortex=.9, white_matter=.93)
V1_MEANS = dict(seg35=.9260, seg4=.9584, small_six=.816, big_six=.956, other_23=.947, cortex=.89, white_matter=.92)
MADE_TRAIN = dict(trainer=NC.TRAINERS[250], epochs=250, wall_seconds=7200., epoch_seconds_median=100., host_peak_pss_gib=6., gpu_peak_used_mib=8000, stopped_before_final_validation=True, resumed_runs=0, final_checkpoint_sha256="a"*64)
MADE_INFER = dict(trainer=NC.TRAINERS[250], fold="all", checkpoint="checkpoint_final.pth", configuration="3d_fullres", checkpoint_sha256="a"*64, allowed_mirroring_axes=None, tta_views=1, use_mirroring=True, seconds_median=20., seconds_max=30., seconds_first=30., gpu_peak_allocated_gib=5., patch_size=[128, 128, 128])


def made_evaluation(delta, subjects=20):
    paired = dict(mean=delta, median=delta, n_positive=15, n_negative=5, n_zero=0, n=20, n_undefined=0)
    nn = {k: v+delta for k, v in REFERENCE_MEANS.items()}
    return dict(subjects=[f"s{i}" for i in range(subjects)], means=dict(nn=nn, v2_control=dict(REFERENCE_MEANS), reference_v2=dict(REFERENCE_MEANS), reference_v1=dict(V1_MEANS)), paired=dict(nn_minus_reference_v2={m: paired for m in EVAL.METRICS}, nn_minus_reference_v1={m: paired for m in EVAL.METRICS}), control=dict(differences=dict(seg35=0., seg4=0.), tolerance=1e-9, passed=True),
                focus=dict(subject="OASIS_OAS1_0440_MR1", row=dict(nn=dict(nn), reference_v2=dict(REFERENCE_MEANS)), others_mean=dict(nn=dict(nn), reference_v2=dict(REFERENCE_MEANS))))


@pytest.mark.parametrize("delta,reading", [(.0051, "clearly_better"), (.005, "clearly_better"), (.0049, "middle"), (.0035, "middle"), (.0021, "middle"), (.002, "not_better"), (.0019, "not_better"), (0., "not_better"), (-.01, "not_better")])
def test_rule_branches(delta, reading):
    assert DECIDE.reading_of(delta) == reading


def test_rule_on_undefined():
    assert DECIDE.reading_of(None) == "not_decidable" and DECIDE.reading_of(float("nan")) == "not_decidable"


@pytest.mark.parametrize("delta,reading", [(.006, "clearly_better"), (.003, "middle"), (.001, "not_better")])
def test_decide_reads_the_rule_on_the_unrounded_means(delta, reading):
    document = DECIDE.decide(made_evaluation(delta), MADE_TRAIN, MADE_INFER)
    assert document["reading"] == reading and document["delta_seg35"] == pytest.approx(delta) and document["delta_seg4"] == pytest.approx(delta) and document["not_decidable_because"] == []
    assert document["against_v1"]["seg35"] == pytest.approx(.9275+delta-.9260) and document["against_u1"] is None


def test_decide_text_of_the_better_branch_depends_on_the_epochs():
    long_ = DECIDE.decide(made_evaluation(.006), MADE_TRAIN, MADE_INFER)
    assert "接到配准路径" in long_["reading_text"] and "硬约束 7" in long_["reading_text"] and "训满" in long_["reading_text"]
    short_train, short_infer = dict(MADE_TRAIN, trainer=NC.TRAINERS[100], epochs=100), dict(MADE_INFER, trainer=NC.TRAINERS[100])
    short_ = DECIDE.decide(made_evaluation(.006), short_train, short_infer)
    assert "接到配准路径" in short_["reading_text"] and "训满" not in short_["reading_text"]
    assert DECIDE.decide(made_evaluation(.001), MADE_TRAIN, MADE_INFER)["reading_text"] == "不比现在的集成好" and DECIDE.decide(made_evaluation(.003), MADE_TRAIN, MADE_INFER)["reading_text"] == "中间带"


@pytest.mark.parametrize("arguments,fragment", [
    ((None, MADE_TRAIN, MADE_INFER), "没有评测结果"),
    ((made_evaluation(.006, subjects=19), MADE_TRAIN, MADE_INFER), "19 名受试者"),
    ((made_evaluation(.006), None, MADE_INFER), "没有 train_record.json"),
    ((made_evaluation(.006), MADE_TRAIN, None), "没有 infer_record.json"),
    ((made_evaluation(.006), dict(MADE_TRAIN, trainer="nnUNetTrainer_250epochs"), dict(MADE_INFER, trainer="nnUNetTrainer_250epochs")), "不是本计划的"),
    ((made_evaluation(.006), MADE_TRAIN, dict(MADE_INFER, trainer=NC.TRAINERS[100])), "trainer 不同"),
    ((made_evaluation(.006), MADE_TRAIN, dict(MADE_INFER, fold=0)), "fold all"),
    ((made_evaluation(.006), MADE_TRAIN, dict(MADE_INFER, checkpoint="checkpoint_best.pth")), "checkpoint_final"),
    ((made_evaluation(.006), MADE_TRAIN, dict(MADE_INFER, configuration="3d_lowres")), "3d_fullres"),
    ((made_evaluation(.006), MADE_TRAIN, dict(MADE_INFER, checkpoint_sha256="b"*64)), "权重"),
    ((dict(made_evaluation(.006), control=dict(differences=dict(seg35=.01), tolerance=1e-9, passed=False)), MADE_TRAIN, MADE_INFER), "控制不符"),
    ((made_evaluation(.006), MADE_TRAIN, dict(MADE_INFER, allowed_mirroring_axes=[0, 1], tta_views=4)), "没有 TTA"),
    ((made_evaluation(.006), MADE_TRAIN, dict(MADE_INFER, tta_views=4)), "没有 TTA"),
    ((made_evaluation(.006), MADE_TRAIN, {k: v for k, v in MADE_INFER.items() if k != "tta_views"}), "没有 TTA"),
])
def test_not_decidable(arguments, fragment):
    document = DECIDE.decide(*arguments)
    assert document["reading"] == "not_decidable" and document["delta_seg35"] is None and any(fragment in reason for reason in document["not_decidable_because"])
    assert "不可判" in DECIDE.summary_markdown(document)


def test_undefined_metric_is_not_decidable():
    evaluation = made_evaluation(.006)
    evaluation["means"]["nn"]["seg4"] = None
    assert any("seg4" in reason for reason in DECIDE.decide(evaluation, MADE_TRAIN, MADE_INFER)["not_decidable_because"])


def test_no_option_can_change_a_reading():
    """The reading follows from the three files: nothing added on the command line can turn it into not decidable after the results are seen."""
    source = (NN/"decide_nn.py").read_text(encoding="utf-8")
    assert "--reason" not in source
    tree_ = ast.parse(source)
    options = [n.args[0].value for n in ast.walk(tree_) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"]
    assert sorted(options) == ["--eval", "--infer-record", "--out-dir", "--train-record", "--u1-eval"]


def test_summary_describes_what_the_plan_lists():
    document = DECIDE.decide(made_evaluation(.03), MADE_TRAIN, MADE_INFER, u1_means=dict(seg35=.93, seg4=.96))
    text = DECIDE.summary_markdown(document)
    assert "seg35" in text and "seg4" in text and "小六个" in text and "皮层" in text and "白质" in text and "OASIS_OAS1_0440_MR1" in text and "U1" in text and "SV" in text and "不是配准结果" in text and "没有镜像 TTA" in text and "证伪不了" in text
    assert document["against_u1"]["seg35"] == pytest.approx(.9575-.93)
    places = document["sv_place"]
    assert places[0]["goal"] == .87 and places[0]["at_or_above_high"] is True and places[1]["goal"] == .88 and places[1]["at_or_above_low"] is False
    assert "没有 eval_u1.json" in DECIDE.summary_markdown(DECIDE.decide(made_evaluation(.03), MADE_TRAIN, MADE_INFER))


def test_decide_files_are_never_overwritten_and_nan_is_null(tmp_path):
    evaluation = made_evaluation(.006)
    evaluation["means"]["nn"]["small_six"] = float("nan")
    document = DECIDE.decide(evaluation, MADE_TRAIN, MADE_INFER)
    json_path, md_path = DECIDE.write(tmp_path/"a", document)
    assert json.loads(json_path.read_text(encoding="utf-8"))["means"]["nn"]["small_six"] is None and "读数" in md_path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="never overwritten"):
        DECIDE.write(tmp_path/"a", document)


def test_decide_main_on_missing_files(tmp_path, capsys):
    assert DECIDE.main(["--eval", str(tmp_path/"none.json"), "--train-record", str(tmp_path/"none.json"), "--infer-record", str(tmp_path/"none.json"), "--out-dir", str(tmp_path/"d")]) == 0
    assert json.loads((tmp_path/"d"/"decision_nn.json").read_text(encoding="utf-8"))["reading"] == "not_decidable" and "not_decidable" in capsys.readouterr().out


def test_read_u1(tmp_path):
    assert DECIDE.read_u1(tmp_path/"none.json") is None and DECIDE.read_u1(None) is None
    (tmp_path/"u.json").write_text(json.dumps(dict(means=dict(U1=dict(seg35=.93)))))
    assert DECIDE.read_u1(tmp_path/"u.json") == dict(seg35=.93)
