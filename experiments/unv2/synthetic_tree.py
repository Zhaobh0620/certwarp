"""The synthetic tree of `experiments/unv2/chain.sh dry-run` (no real subject; ids OASIS_OAS1_09xx_MR1, which OASIS does not have):

    <work>/data/<subject>/{aligned_norm,aligned_seg35,aligned_seg4}.nii.gz, <work>/folds.json, <work>/folds_full_v2.json,
    <work>/dev_subjects.json, <work>/pred_v1/<subject>/ (the "v1" predictions eval_seg compares with: the truth with noise, in the
    layout of bayesreg.seg.unet.infer).

Each fold of folds.json has 2 train + 1 calib subjects, so each fold of folds_full_v2.json has 5 train + 1 calib (the chain's
--expect-counts 5,1).

    python -B experiments/unv2/synthetic_tree.py --work DIR
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bayesreg.seg.unet.data import IMAGE_FILE, full_folds_document, parse_folds, render_full_folds  # noqa: E402
from bayesreg.seg.unet.groups import group_probabilities, load_grouping  # noqa: E402
from bayesreg.seg.unet.synthetic import allowlist_document, folds_document, oasis, synthetic_volume, write_json, write_subjects  # noqa: E402

SHAPE = (32, 32, 32)
A_TRAIN, A_CALIB, B_TRAIN, B_CALIB = [oasis(901), oasis(902)], [oasis(903)], [oasis(904), oasis(905)], [oasis(906)]
DEV, T0 = [oasis(907), oasis(908)], [oasis(909)]
TRAINING = A_TRAIN+A_CALIB+B_TRAIN+B_CALIB
SOURCE = "synthetic_bline_train"


def fake_v1_prediction(root, data_root, subject, grouping):
    """A prediction directory with the fields eval_seg reads: probabilities peaked on the truth with noise (float16)."""
    from jd_curl.three_d.nifti import file_sha256, load_scalar
    labels = load_scalar(Path(data_root)/subject/"aligned_seg35.nii.gz", labels=True).values.astype(np.int64)
    rng = np.random.default_rng(int(subject.split("_")[2]))
    noise = rng.uniform(0, 1, (36,)+labels.shape)
    probs = .55*(np.arange(36)[:, None, None, None] == labels[None])+.45*noise/noise.sum(axis=0)
    probs = (probs/probs.sum(axis=0)).astype(np.float32)
    directory = Path(root)/subject
    directory.mkdir(parents=True)
    np.savez_compressed(directory/"probs36.npz", p=probs.astype(np.float16))
    np.savez_compressed(directory/"structure_groups.npz", p=group_probabilities(probs, grouping, dtype=np.float32).astype(np.float16))
    image_sha = hashlib.sha256((Path(data_root)/subject/IMAGE_FILE).read_bytes()).hexdigest()
    write_json(directory/"provenance.json", dict(
        layer="structure", segmenter_id="unet35_v1_ens", subject=subject, policy="all_folds_mean", test_labels_seen=False,
        model_ids=["unet35_v1_foldA", "unet35_v1_foldB"], members=[], input_sha256={IMAGE_FILE: image_sha},
        output_sha256={name: file_sha256(directory/name) for name in ("probs36.npz", "structure_groups.npz")}))


def build(work):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    grouping, tissue4 = load_grouping(), load_grouping(name="tissue4_ref")
    write_subjects(work/"data", TRAINING+DEV+T0, SHAPE, tissue4=tissue4)
    v1 = folds_document(A_TRAIN, A_CALIB, B_TRAIN, B_CALIB)
    write_json(work/"folds.json", v1)
    (work/"folds_full_v2.json").write_bytes(render_full_folds(full_folds_document(parse_folds(v1))))
    write_json(work/"dev_subjects.json", allowlist_document(
        [("synthetic_dev", DEV, False), ("synthetic_t0", T0, True), (SOURCE, TRAINING, False)]))
    for subject in DEV:
        fake_v1_prediction(work/"pred_v1", work/"data", subject, grouping)
    return work


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args(argv)
    build(args.work)
    print(json.dumps(dict(work=str(args.work), training=TRAINING, dev=DEV, t0=T0), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
