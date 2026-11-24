"""Folds parsing, the segmenter label guard (refusal before any file is opened), normalization and
augmentation."""
import numpy as np
import pytest
import torch

from bayesreg.seg.unet.data import (LabelAccessRefused, LabelGuard, augment, load_folds, normalize_image, parse_folds,
                                    random_patch, read_labels, spatial_theta)
from bayesreg.seg.unet.synthetic import folds_document, oasis, write_json
from jd_curl.validation import InputValidationError

from ._helpers import (A_CALIB, A_TRAIN, B_CALIB, B_TRAIN, DEV, HOLDOUT, SEALED, T0, TRAIN_SOURCE, TRAINING,
                       allowlist_sources, make_allowlist, make_guard, make_split)


class NoFiles:
    """OasisFiles stand-in: any attempt to locate a file fails the test."""

    def image_path(self, subject):
        raise AssertionError(f"image of {subject} opened")

    def label_path(self, subject, kind="seg35"):
        raise AssertionError(f"labels of {subject} opened")


# ---- folds ----------------------------------------------------------------------------------------

def test_layout_a_counts_and_label_contact():
    split = make_split()
    assert split.layout == "subjects_map" and split.counts() == {"A": {"train": 2, "calib": 1},
                                                                "B": {"train": 2, "calib": 1}}
    assert split.label_contact("A") == sorted(A_TRAIN+A_CALIB) and split.subjects("B", "calib") == B_CALIB
    assert split.sealed_lists == (("bline_holdout", 1),)


def test_layouts_b_and_c_give_the_same_assignment():
    a = make_split()
    records = [dict(subject=s, fold=f, role=r) for s, (f, r) in a.assignment.items()]
    b = parse_folds({"training_records": records})
    c = parse_folds({"training": {"folds": {"A": {"train": A_TRAIN, "calibration": A_CALIB},
                                            "B": {"train": B_TRAIN, "calib": B_CALIB}}}})
    assert b.layout == "record_list" and c.layout == "training/fold_lists"
    assert a.assignment == b.assignment == c.assignment


@pytest.mark.parametrize("change", ["duplicate", "bad_fold", "bad_role", "missing_calib", "holdout_overlap", "lpba",
                                    "two_layouts"])
def test_malformed_folds_are_refused(change):
    doc = folds_document(A_TRAIN, A_CALIB, B_TRAIN, B_CALIB, HOLDOUT)
    if change == "duplicate":
        doc = {"folds": {"A": {"train": A_TRAIN, "calib": A_CALIB}, "B": {"train": B_TRAIN+A_TRAIN[:1], "calib": B_CALIB}}}
    elif change == "bad_fold":
        doc["subjects"][A_TRAIN[0]]["fold"] = "C"
    elif change == "bad_role":
        doc["subjects"][A_TRAIN[0]]["role"] = "test"
    elif change == "missing_calib":
        doc["subjects"][B_CALIB[0]]["role"] = "train"
    elif change == "holdout_overlap":
        doc["bline_holdout"].append(A_TRAIN[0])
    elif change == "lpba":
        doc["subjects"]["S01"] = dict(fold="A", role="train")
    elif change == "two_layouts":
        doc["folds"] = {"A": {"train": A_TRAIN, "calib": A_CALIB}, "B": {"train": B_TRAIN, "calib": B_CALIB}}
    with pytest.raises(InputValidationError):
        parse_folds(doc)


def test_load_folds_checks_expected_counts(tmp_path):
    path = write_json(tmp_path/"folds.json", folds_document(A_TRAIN, A_CALIB, B_TRAIN, B_CALIB))
    assert load_folds(path, expect_counts=(2, 1)).sha256
    with pytest.raises(InputValidationError):
        load_folds(path, expect_counts=(45, 8))


# ---- label guard ----------------------------------------------------------------------------------

@pytest.mark.parametrize("subject", [DEV[0], T0[0], SEALED, HOLDOUT[0], "S01", "OASIS_OAS1_901_MR1", oasis(440)])
def test_guard_refuses_non_training_subjects_before_opening_files(tmp_path, subject):
    guard = make_guard(tmp_path)
    with pytest.raises(LabelAccessRefused):
        read_labels(NoFiles(), subject, guard)
    assert guard.reads == []


def test_guard_refuses_wrong_fold(tmp_path):
    guard = make_guard(tmp_path)
    guard.check(A_TRAIN[0], "A")
    with pytest.raises(LabelAccessRefused):
        read_labels(NoFiles(), A_TRAIN[0], guard, fold="B")


def test_guard_requires_a_label_guard():
    with pytest.raises(InputValidationError):
        read_labels(NoFiles(), A_TRAIN[0], None)


@pytest.mark.parametrize("case", ["t0_in_training", "dev_in_training", "off_allowlist", "l2r_range", "denied",
                                  "no_training_source", "source_larger", "source_smaller", "source_never"])
def test_guard_refuses_a_split_that_touches_protected_subjects(tmp_path, case):
    a_train, sources, denied, name = list(A_TRAIN), None, (), TRAIN_SOURCE
    if case == "t0_in_training":
        a_train.append(T0[0])
        sources = allowlist_sources(TRAINING+[T0[0]])
    elif case == "dev_in_training":
        a_train.append(DEV[0])
        sources = allowlist_sources(TRAINING+[DEV[0]])
    elif case == "off_allowlist":
        a_train.append(SEALED)
    elif case == "l2r_range":
        a_train.append(oasis(450))
        sources = allowlist_sources(TRAINING+[oasis(450)])
    elif case == "denied":
        denied = (A_TRAIN[0],)
    elif case == "no_training_source":
        name = "bayesreg_train_v1"                                  # not in the synthetic allowlist
    elif case == "source_larger":                                   # named source lists one more subject
        sources = allowlist_sources(TRAINING+[oasis(912)])
    elif case == "source_smaller":
        sources = allowlist_sources(TRAINING[:-1], extra=[("synthetic_rest", TRAINING[-1:], False)])
    elif case == "source_never":
        sources = [s for s in allowlist_sources() if s[0] != TRAIN_SOURCE]+[(TRAIN_SOURCE, TRAINING, True)]
    split = make_split(a_train=a_train)
    allowlist = make_allowlist(tmp_path, sources)
    with pytest.raises(LabelAccessRefused):
        LabelGuard(split, allowlist, training_source=name, denied=denied)


def test_guard_needs_the_training_source_named():
    with pytest.raises(TypeError):
        LabelGuard(make_split(), None)                              # training_source is keyword-only, required


def test_guard_record_lists_reads(tmp_path):
    record = make_guard(tmp_path).record()
    assert record["label_reads"] == [] and record["folds_layout"] == "subjects_map"
    assert record["l2r_val_range"] == [438, 457] and record["training_source"] == TRAIN_SOURCE


# ---- normalization and augmentation ---------------------------------------------------------------

def test_normalize_image_divides_by_brain_percentile():
    values = np.zeros((4, 4, 4))
    values[1:3, 1:3, 1:3] = np.arange(1, 9).reshape(2, 2, 2)
    image, record = normalize_image(values, percentile=50)
    assert record["brain_voxels"] == 8 and record["scale"] == pytest.approx(4.5)
    assert image.dtype == np.float32 and float(image.max()) == pytest.approx(8/4.5)
    with pytest.raises(InputValidationError):
        normalize_image(np.zeros((3, 3, 3)))


def test_spatial_theta_identity_and_centre_fixed():
    assert np.allclose(spatial_theta((5, 7, 9), (0, 0, 0), (0, 0, 0)), np.eye(3, 4))
    theta = spatial_theta((5, 7, 9), np.deg2rad([5, -7, 9]), (0, 0, 0))
    assert np.allclose(theta[:, 3], 0)                # the volume centre maps to itself


def test_augment_keeps_labels_integer_background_zero_and_no_flip():
    rng = np.random.default_rng(0)
    labels = torch.zeros(1, 12, 12, 12, dtype=torch.long)
    labels[0, 3:9, 3:9, 2:6] = 7                       # a left-side block
    labels[0, 3:9, 3:9, 6:10] = 26                     # its right-side neighbour
    image = (labels.float()/30)[:, None]
    config = dict(p_spatial=1., max_rotation_deg=0., max_translation_vox=0., p_gamma=1., p_scale_shift=1.)
    aug_image, aug_labels, record = augment(image, labels, rng, config)
    assert torch.equal(aug_labels, labels)             # identity geometry: no flip, labels untouched
    assert float(aug_image[labels[:, None] == 0].abs().max()) < 1e-5 and "gamma" in record
    _, moved, _ = augment(image, labels, rng, dict(p_spatial=1., max_rotation_deg=10., max_translation_vox=2.))
    assert set(torch.unique(moved).tolist()) <= {0, 7, 26}


def test_random_patch_shape_and_foreground_centre():
    rng = np.random.default_rng(1)
    labels = torch.zeros(1, 16, 16, 16, dtype=torch.long)
    labels[0, 12:, 12:, 12:] = 3
    image = labels[:, None].float()
    x, y = random_patch(image, labels, (8, 8, 8), rng, p_foreground=1.)
    assert x.shape == (1, 1, 8, 8, 8) and y.shape == (1, 8, 8, 8) and bool((y == 3).any())
