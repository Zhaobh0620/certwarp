"""Shared by the probes of work package SEGP (runs/20261105-segprobe/任务书.md): the admission of the development subjects and a read-back of probs36 that keeps the
float16 array and normalizes slab by slab, so that two posteriors of 36 channels stay within the 3 GiB of the brief.

normalized_slab divides as bayesreg.seg.channels.read_back_probabilities does (float64, by the channel sum) and check_raw makes the same checks before any division; the
result is equal to that function bit by bit (tests/test_bayesreg/segprobe/test_flip_tta.py). Every function of the evaluation comes from experiments/unv2/eval_seg.py unchanged.
"""
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO/"experiments/unv2")]
import eval_seg  # noqa: E402
from bayesreg.eval.truth import load_sealed_lists  # noqa: E402
from bayesreg.io.guard import load_dev_allowlist  # noqa: E402
from bayesreg.io.pair import verify_segmentation_files  # noqa: E402
from bayesreg.seg.channels import RAW_SUM_TOL  # noqa: E402
from bayesreg.seg.unet.data import load_folds  # noqa: E402
from bayesreg.seg.unet.pins import source_root_for  # noqa: E402
from bayesreg.seg.unet.snapshot import code_record  # noqa: E402

V1_ROOT = Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v1/pred")
V2_ROOT = Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v2_full/pred")
V1_ID, V2_ID = "unet35_v1_ens", "unet35_v2_full"
OUT_ROOT = Path("/home/rmcfz/seq_vp_tag_results/segprobe_v1")
REFERENCE = Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v2_full/eval_seg/eval_seg.json")
SLAB = 16
# seg35 labels of the two tissues the brief names, and the classes of seg4 (configs/bayesreg/seg4_of_seg35_v1.json)
CORTEX_LABELS, WM_LABELS = (2, 21), (1, 20)
SEG4_CORTEX, SEG4_WM = 1, 3


def admit_dev_subjects(*, data_root=eval_seg.DATA_ROOT, allowlist_path=eval_seg.ALLOWLIST, folds_path=eval_seg.FOLDS, dev_source=eval_seg.DEV_SOURCE, sealed_repo=None):
    """(subjects, record): the 20 subjects of the allowlist source, each checked against the sealed lists and the training set before anything of a subject is opened
    (eval_seg.admit_subjects: SubjectSealed otherwise). record: what was checked."""
    code = code_record()
    sealed = load_sealed_lists(Path(sealed_repo) if sealed_repo is not None else source_root_for(code))
    allowlist = load_dev_allowlist(allowlist_path)
    folds = load_folds(folds_path)
    subjects = eval_seg.admit_subjects(allowlist, dev_source, sealed, folds.training_subjects)
    return subjects, dict(code=code, dev_source=dev_source, allowlist_sha256=allowlist.sha256, folds_sha256=folds.sha256, sealed_lists=sealed.record, data_root=str(data_root))


def check_raw(raw, name, *, slab=SLAB):
    """The checks of read_back_probabilities on a raw (C, Z, Y, X) array, slab by slab: storage type, finite, not negative, no voxel with zero total, the raw channel sum within
    RAW_SUM_TOL before any division. Returns the largest deviation of the sum from 1."""
    dtype = str(raw.dtype)
    if raw.dtype.kind != "f" or dtype not in RAW_SUM_TOL:
        raise ValueError(f"{name}: probabilities stored as one of {sorted(RAW_SUM_TOL)} required, got {raw.dtype}")
    largest = 0.
    for start in range(0, raw.shape[1], slab):
        part = raw[:, start:start+slab].astype(np.float64)
        if not np.all(np.isfinite(part)) or np.any(part < 0):
            raise ValueError(f"{name}: non-finite or negative probabilities")
        total = part.sum(axis=0)
        if np.any(total <= 0):
            raise ValueError(f"{name}: a voxel with zero total probability")
        largest = max(largest, float(np.max(np.abs(total-1))))
    if not largest <= RAW_SUM_TOL[dtype]:
        raise ValueError(f"{name}: raw channel sum deviates by {largest:.3e} > {RAW_SUM_TOL[dtype]:g}")
    return largest


def load_raw(root, subject, *, expected_segmenter):
    """(raw float16 (C, Z, Y, X), record): the checks of eval_seg.read_prediction (provenance of this subject, policy, output hashes, the raw sums), without the conversion to float64."""
    directory = Path(root)/subject
    provenance = json.loads((directory/"provenance.json").read_text())
    if provenance.get("subject") != subject or provenance.get("layer") != "structure":
        raise ValueError(f"{directory}: the provenance is not of the structure layer of {subject}")
    if provenance.get("segmenter_id") != expected_segmenter:
        raise ValueError(f"{directory}: segmenter {provenance.get('segmenter_id')!r}, expected {expected_segmenter!r}")
    if provenance.get("policy") != "all_folds_mean" or provenance.get("test_labels_seen") is not False:
        raise ValueError(f"{directory}: policy {provenance.get('policy')!r}, test_labels_seen {provenance.get('test_labels_seen')!r}")
    verified = verify_segmentation_files(provenance, directory)
    with np.load(directory/"probs36.npz", allow_pickle=False) as npz:
        raw = npz["p"]
    if raw.shape[0] != eval_seg.N_CLASSES:
        raise ValueError(f"{directory}/probs36.npz: {raw.shape[0]} classes, expected {eval_seg.N_CLASSES}")
    deviation = check_raw(raw, f"{directory}/probs36.npz")
    return raw, dict(directory=str(directory), probs36_sha256=verified.get("probs36.npz"), provenance_sha256=eval_seg.sha_file(directory/"provenance.json"),
                     segmenter_id=provenance.get("segmenter_id"), n_members=len(provenance.get("members", [])), raw_max_abs_sum_deviation=deviation,
                     image_sha256=provenance.get("input_sha256", {}).get(eval_seg.IMAGE_FILE))


def normalized_slab(raw, start, stop):
    """float64 (C, z, Y, X) of the slab: divided by the channel sum, as read_back_probabilities."""
    part = raw[:, start:stop].astype(np.float64)
    part /= part.sum(axis=0)
    return part


def argmax_of_mean(raws, *, slab=SLAB):
    """uint8 argmax of the mean over `raws` of the normalized posteriors; ties: the smallest label (np.argmax). With one raw array it is eval_seg.read_prediction's argmax."""
    shape = raws[0].shape
    if any(r.shape != shape for r in raws):
        raise ValueError(f"posteriors of different shapes: {[r.shape for r in raws]}")
    out = np.empty(shape[1:], dtype=np.uint8)
    for start in range(0, shape[1], slab):
        stop = min(start+slab, shape[1])
        total = normalized_slab(raws[0], start, stop)
        for other in raws[1:]:
            total += normalized_slab(other, start, stop)
        out[start:stop] = np.argmax(total/len(raws), axis=0).astype(np.uint8)
    return out


def write_json(path, document):
    Path(path).write_text(json.dumps(eval_seg.no_nan(document), indent=1, allow_nan=False, ensure_ascii=False)+"\n", encoding="utf-8")


def peak_rss_gib():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
