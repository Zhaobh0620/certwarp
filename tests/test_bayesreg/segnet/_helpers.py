"""Synthetic folds / allowlists / subjects for the segmenter tests (tmp_path only; no data/ access).

Ids OASIS_OAS1_09xx_MR1 are canonical spellings that do not exist in OASIS (real ids end at 0457).
"""
import numpy as np
import torch

from bayesreg.io.guard import load_dev_allowlist
from bayesreg.seg.unet.data import LabelGuard, Subject, parse_folds
from bayesreg.seg.unet.synthetic import allowlist_document, folds_document, oasis, write_json

A_TRAIN, A_CALIB = [oasis(901), oasis(902)], [oasis(903)]
B_TRAIN, B_CALIB = [oasis(904), oasis(905)], [oasis(906)]
TRAINING = A_TRAIN+A_CALIB+B_TRAIN+B_CALIB
DEV, T0, SEALED, HOLDOUT = [oasis(907), oasis(908)], [oasis(909)], oasis(910), [oasis(911)]
TRAIN_SOURCE = "synthetic_bline_train"
TINY = dict(base_channels=4, depth=2, max_iters=4, eval_every=2, warmup_iters=1, min_iters=4, log_every=1, amp=False)

torch.set_num_threads(1)


def allowlist_sources(training=TRAINING, extra=()):
    return [("synthetic_dev", DEV, False), ("synthetic_t0", T0, True), (TRAIN_SOURCE, training, False)]+list(extra)


def make_allowlist(tmp_path, sources=None, name="dev_subjects.json"):
    return load_dev_allowlist(write_json(tmp_path/name, allowlist_document(sources or allowlist_sources())))


def make_split(a_train=A_TRAIN, a_calib=A_CALIB, b_train=B_TRAIN, b_calib=B_CALIB, holdout=HOLDOUT):
    return parse_folds(folds_document(a_train, a_calib, b_train, b_calib, holdout))


def make_guard(tmp_path, split=None, sources=None, training_source=TRAIN_SOURCE):
    return LabelGuard(split or make_split(), make_allowlist(tmp_path, sources), training_source=training_source)


def tiny_subject(name, shape=(16, 16, 16), seed=0, n_classes=36):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, shape).astype(np.uint8)
    image = (labels/n_classes+rng.normal(0, .01, shape)).astype(np.float32)
    image[labels == 0] = 0
    return Subject(name, image, labels, image_sha256="0"*64, label_sha256="1"*64,
                   normalization=dict(percentile=99.5, scale=1.))
