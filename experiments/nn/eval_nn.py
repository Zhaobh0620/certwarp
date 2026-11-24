"""Segmentation-level evaluation of the nnU-Net model of work package NN (runs/20261105-nn/plan.md). sci310 only, no nnunetv2; the functions of experiments/unv2/eval_seg.py (admit_subjects, summarize_subject, read_prediction, read_truth, paired): the 20 development subjects, the written argmax against the truth, Dice per label, seg35 = nanmean over labels 1..35, the small six, the big six, the other 23, seg4 through configs/bayesreg/seg4_of_seg35_v1.json, the cortex and the white matter (seg4 labels 1 and 3, the definition of experiments/sv/run_sv.py), per subject; OASIS_OAS1_0440_MR1 on its own.

Arms:
  nn           <pred-root>/<subject>.nii.gz, the argmax nnU-Net wrote. Its shape and affine must be the image's (a header read); it is put into the array order of load_scalar (RAS, zyx) with the orientation the image's affine gives. The arm of the reading.
  v2_control   the registered predictions of unet35_v2_full (<v2-root>/<subject>/probs36.npz, eval_seg.read_prediction) through this same code. A control of the evaluator: no network runs, so it must reproduce the committed numbers of the reference to CONTROL_TOL.
Reference: /home/rmcfz/seq_vp_tag_results/seg/unet35_v2_full/eval_seg/eval_seg.json, pinned by sha256: the per-subject numbers of the arms v1 and v2 (.9260 / .9584 and .9275 / .9591 in the brief).

Order of the checks: the read lock (nn_lock.py), first of all on real data; then, before any file of a subject is opened, every subject in the allowlist source stage_L2R_val_v1, not never_for_training, on no sealed list, no training subject, outside the model's label contact (prepare_manifest.json), and the same subjects as the reference; then, before the lock, for all 20 subjects: the nn arm (file, shape and affine, label range; the share of its foreground on non-zero image voxels; the left and right cerebral labels on their sides along RAS x, the largest separation being x: the checks of orientation that need no truth) and the control arm (provenance, sha256 of its files). A failure there costs no read. The lock is taken immediately before the first truth label.
Real data: the reference pinned; the allowlist, folds, manifest, data root, control root and development source the plan's, clean code; --pred-root is free, the lock holds whatever it says. --synthetic: synthetic ids and a non-real data root only. Outputs in --out-dir (never overwritten): eval_nn.json, per_subject.csv, per_label.csv.

    python -B experiments/nn/eval_nn.py --pred-root DIR --manifest FILE --out-dir DIR [--bz-go TEXT]
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO/"experiments/unv2"), str(REPO/"experiments/nn")]
import eval_seg  # noqa: E402
import nn_common  # noqa: E402
import nn_lock  # noqa: E402
from bayesreg.eval.truth import load_sealed_lists  # noqa: E402
from bayesreg.io.guard import SubjectSealed, load_dev_allowlist, refuse_sealed  # noqa: E402
from bayesreg.seg.unet.data import IMAGE_FILE, load_folds  # noqa: E402
from bayesreg.seg.unet.pins import check_synthetic_data_root, check_synthetic_subjects, load_pinned_allowlist, source_root_for  # noqa: E402
from bayesreg.seg.unet.snapshot import code_record, require_clean_code  # noqa: E402

FORMAT = "bayesreg.nn.eval.v1"
REFERENCE = Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v2_full/eval_seg/eval_seg.json")
REFERENCE_SHA256 = "698dcc39708313fcec61a663d0def019b443f139d937912b186734a80def84e4"
V2_ROOT = Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v2_full/pred")
V2_SEGMENTER = "unet35_v2_full"
CONTROL_TOL = 1e-9              # the same stored probabilities and the same truth: nothing may differ
GEOMETRY_TOL = 1e-3             # mm; SimpleITK's writer goes through float32 quaternions
MANIFEST = nn_common.OUT/"prepare_manifest.json"
MIN_FOREGROUND_ON_IMAGE = .95   # the share of the predicted foreground on non-zero image voxels. Measured (docs/bayesreg/reviews/NN_tools_复核_20261105.md, revision round; truth labels and v2 argmax taken as predictions, single observation): not flipped .9996 and above; flipped top-bottom .885-.919, front-back and both lower. A left-right flip stays near 1: LR_PAIRS stops that one
LR_PAIRS = ((1, 20), (2, 21))   # Left and Right Cerebral-White-Matter, Cerebral-Cortex (the table of configs/bayesreg/structure_groups_v1.json), as bayesreg.seg.unet.large.check_flip_axis: left at low RAS x, x the axis that separates them most
MIN_LR_SEPARATION_VOX = 10.
FOCUS_SUBJECT = "OASIS_OAS1_0440_MR1"
ARMS = ("nn", "v2_control")
METRICS = ("seg35", "seg4")+tuple(eval_seg.GROUPS)+("cortex", "white_matter")
N_CLASSES = eval_seg.N_CLASSES


def metrics_of(summary):
    """The flat row of one subject from eval_seg.summarize_subject (or a stored row of the same form): cortex and white matter are the seg4 labels 1 and 3."""
    per_seg4 = summary["seg4_per_label"]
    return dict(seg35=summary["seg35"], seg4=summary["seg4"], **summary["groups"], cortex=per_seg4[0], white_matter=per_seg4[2], per_label=summary["per_label"], seg4_per_label=per_seg4)


def clean(value):
    """A stored null (undefined Dice) as nan, for the means."""
    return float("nan") if value is None else value


def mean_of(rows, metric):
    return eval_seg.nanmean([clean(r[metric]) for r in rows])


def read_reference(path, *, pinned_sha256=None):
    """The rows of the arms v1 and v2 of an eval_seg.json; ValueError if the file is not the pinned one or its means do not follow from its rows."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if pinned_sha256 is not None and digest != pinned_sha256:
        raise ValueError(f"{path}: sha256 {digest} is not the pinned {pinned_sha256}")
    document = json.loads(raw.decode("utf-8"))
    subjects = list(document["subjects"])
    rows = {arm: [metrics_of({k: document["rows"][s][arm][k] for k in ("seg35", "seg4", "groups", "seg4_per_label", "per_label")}) for s in subjects] for arm in ("v1", "v2")}
    means = {arm: {m: mean_of(rows[arm], m) for m in METRICS} for arm in rows}
    for arm in rows:
        for metric in ("seg35", "seg4"):
            if abs(means[arm][metric]-document["means"][arm][metric]) > 1e-12:
                raise ValueError(f"{path}: the mean of {arm} {metric} does not follow from its rows")
    return dict(path=str(path), sha256=digest, subjects=subjects, rows=rows, means=means)


def orient_like_load_scalar(values, affine):
    """A native array into what jd_curl.three_d.nifti.load_scalar returns for this affine: RAS storage, axes zyx."""
    transform = nib.orientations.ornt_transform(nib.orientations.io_orientation(affine), nib.orientations.axcodes2ornt(("R", "A", "S")))
    return nib.orientations.apply_orientation(values, transform).transpose(2, 1, 0)


def left_right_check(top, pairs=LR_PAIRS, *, min_separation=MIN_LR_SEPARATION_VOX):
    """On an argmax map in the order of load_scalar (zyx, RAS): for each (left, right) pair the left label lies at low x and x separates them most. The smallest separation in voxels, None without pairs; ValueError otherwise. No truth is needed: a prediction flipped left and right stops here, before the one read."""
    smallest = None
    for left, right in pairs:
        points = {label: np.argwhere(top == label) for label in (left, right)}
        if not all(len(p) for p in points.values()):
            raise ValueError(f"labels {left}, {right}: no voxels of one of them in the prediction")
        separation = points[right].mean(axis=0)-points[left].mean(axis=0)
        if separation[2] < min_separation or abs(separation[2]) <= abs(separation[0]) or abs(separation[2]) <= abs(separation[1]):
            raise ValueError(f"labels {left}, {right}: centroid difference (z, y, x) {separation.tolist()}; expected the left at low x, x the largest")
        smallest = float(separation[2]) if smallest is None else min(smallest, float(separation[2]))
    return smallest


def read_nn_prediction(pred_root, data_root, subject, *, lr_pairs=LR_PAIRS, min_on_image=MIN_FOREGROUND_ON_IMAGE):
    """(argmax uint8 in the order of load_scalar, record). ValueError unless the file has the image's shape and affine, holds integers in 0..35, its foreground lies on the image (min_on_image) and left and right are on their sides (lr_pairs). Reads the image, no label.
    A subject of a sealed list is refused before anything is opened (C23.6-2, D104 (a): bayesreg.io.guard.refuse_sealed)."""
    refuse_sealed(subject, "eval_nn.read_nn_prediction")
    path = Path(pred_root)/f"{subject}.nii.gz"
    image = nib.load(Path(data_root)/subject/IMAGE_FILE)
    prediction = nib.load(path)
    if prediction.shape != image.shape:
        raise ValueError(f"{path}: shape {prediction.shape}, the image's is {image.shape}")
    if not np.allclose(prediction.affine, image.affine, rtol=0, atol=GEOMETRY_TOL):
        raise ValueError(f"{path}: affine differs from the image's:\n{prediction.affine}\n{image.affine}")
    values = np.asarray(prediction.dataobj)
    if not np.issubdtype(values.dtype, np.integer) or values.min() < 0 or values.max() >= N_CLASSES:
        raise ValueError(f"{path}: {values.dtype}, values {values.min()}..{values.max()}: not labels 0..{N_CLASSES-1}")
    foreground = values > 0
    on_image = float((foreground & (np.asarray(image.dataobj) > 0)).sum()/max(1, foreground.sum()))
    if on_image < min_on_image:
        raise ValueError(f"{path}: {on_image:.3f} of the predicted foreground lies on non-zero image voxels, under {min_on_image}: orientation?")
    top = orient_like_load_scalar(values.astype(np.uint8), image.affine)
    return top, dict(path=str(path), sha256=eval_seg.sha_file(path), n_labels=int(np.unique(values).size), foreground_on_image=on_image, smallest_lr_separation_vox=left_right_check(top, lr_pairs))


def check_control_files(root, subjects, segmenter):
    """Before the lock, every subject of the control arm: the checks of eval_seg.read_prediction on the provenance, and the sha256 of its files (no decompression). A broken file costs no read of the truth."""
    for subject in subjects:
        directory = Path(root)/subject
        provenance = json.loads((directory/"provenance.json").read_text(encoding="utf-8"))
        if provenance.get("subject") != subject or provenance.get("layer") != "structure" or provenance.get("segmenter_id") != segmenter or provenance.get("policy") != "all_folds_mean" or provenance.get("test_labels_seen") is not False:
            raise ValueError(f"{directory}: the provenance is not of the structure layer of {subject}, {segmenter}, all_folds_mean, no test labels seen")
        eval_seg.verify_segmentation_files(provenance, directory)


def label_contact(manifest):
    """The subjects the model has seen labels of, from prepare_manifest.json."""
    document = json.loads(Path(manifest).read_text(encoding="utf-8"))
    contact = document.get("label_contact_subjects")
    if not contact:
        raise ValueError(f"{manifest}: no label_contact_subjects")
    return sorted(contact), hashlib.sha256(Path(manifest).read_bytes()).hexdigest()


def read_inputs(*, reference_path, data_root, allowlist_path, dev_source, synthetic, folds_path, manifest, v2_root, control_segmenter):
    """(allowlist, its record, the reference): pinned on real data; synthetic ids and a non-real data root otherwise."""
    if synthetic:
        check_synthetic_data_root(data_root)
        allowlist = load_dev_allowlist(allowlist_path)
        check_synthetic_subjects([s for ids in allowlist.subjects.values() for s in ids], "the allowlist")
        return allowlist, dict(mode="synthetic", allowlist_path=str(allowlist_path)), read_reference(reference_path)
    pinned = ((reference_path, REFERENCE), (allowlist_path, eval_seg.ALLOWLIST), (folds_path, eval_seg.FOLDS), (manifest, MANIFEST), (data_root, eval_seg.DATA_ROOT), (v2_root, V2_ROOT))
    if dev_source != eval_seg.DEV_SOURCE or control_segmenter != V2_SEGMENTER or any(Path(given).resolve() != Path(plan).resolve() for given, plan in pinned):
        raise ValueError("real data: the reference, the allowlist, the folds, the manifest, the data root, the control and the development source are the plan's; other values only with --synthetic")
    allowlist, record = load_pinned_allowlist(allowlist_path)
    return allowlist, {k: v for k, v in record.items() if k != "code"}, read_reference(reference_path, pinned_sha256=REFERENCE_SHA256)


def evaluate(*, pred_root, manifest, out_dir, v2_root=V2_ROOT, control_segmenter=V2_SEGMENTER, reference_path=REFERENCE, data_root=eval_seg.DATA_ROOT, allowlist_path=eval_seg.ALLOWLIST, folds_path=eval_seg.FOLDS, dev_source=eval_seg.DEV_SOURCE, sealed_repo=None, synthetic=False, argv=None, log=print, bz_go=None):
    if synthetic and bz_go is not None:
        raise ValueError("--bz-go is for the read lock of real data; synthetic runs do not touch it")
    if not synthetic:
        nn_lock.check(bz_go)                            # the first thing on real data: no other option leads around it
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} exists and is not empty; the evaluation is never overwritten")
    code = code_record()
    if not synthetic:
        require_clean_code(code)
    allowlist, allowlist_record, reference = read_inputs(reference_path=reference_path, data_root=data_root, allowlist_path=allowlist_path, dev_source=dev_source, synthetic=synthetic, folds_path=folds_path, manifest=manifest, v2_root=v2_root, control_segmenter=control_segmenter)
    sealed = load_sealed_lists(Path(sealed_repo) if sealed_repo is not None else source_root_for(code))
    folds = load_folds(folds_path)
    subjects = eval_seg.admit_subjects(allowlist, dev_source, sealed, folds.training_subjects)   # before any file of a subject
    contact, manifest_sha256 = label_contact(manifest)
    overlap = sorted(set(subjects) & set(contact))
    if overlap:
        raise SubjectSealed(f"{overlap} in the label contact of the model; not evaluated")
    if subjects != reference["subjects"]:
        raise ValueError(f"the subjects differ from the reference's: {sorted(set(subjects) ^ set(reference['subjects']))[:5]}")
    lookup = eval_seg.load_seg4_lookup()
    predictions = {subject: read_nn_prediction(pred_root, data_root, subject, lr_pairs=() if synthetic else LR_PAIRS) for subject in subjects}     # all of them before the lock: a missing file, a wrong geometry or a flip costs no read. The synthetic labels are not anatomical, so left and right are not checked there
    check_control_files(v2_root, subjects, control_segmenter)
    rows, records, truth_sha, dev_read = {arm: [] for arm in ARMS}, {arm: {} for arm in ARMS}, {}, None
    for subject in subjects:
        nn_top, records["nn"][subject] = predictions.pop(subject)
        control_top, records["v2_control"][subject] = eval_seg.read_prediction(v2_root, subject, expected_segmenter=control_segmenter)
        if not synthetic and dev_read is None:          # the moment before the first truth label; a failure before it leaves no lock
            dev_read = nn_lock.take(dict(pred_root=str(pred_root), out_dir=str(out_dir), command=argv, code_commit=code.get("commit")), bz_go=bz_go)
        true35, sha35 = eval_seg.read_truth(data_root, subject, "seg35")
        true4, sha4 = eval_seg.read_truth(data_root, subject, "seg4")
        truth_sha[subject] = dict(seg35=sha35, seg4=sha4)
        for arm, top in (("nn", nn_top), ("v2_control", control_top)):
            rows[arm].append(metrics_of(eval_seg.summarize_subject(top, true35, true4, lookup)))
        log(f"{subject}: "+", ".join(f"{arm} {rows[arm][-1]['seg35']:.4f} / {rows[arm][-1]['seg4']:.4f}" for arm in ARMS))
    all_rows = dict(rows, reference_v2=reference["rows"]["v2"], reference_v1=reference["rows"]["v1"])
    means = {arm: {m: mean_of(r, m) for m in METRICS} for arm, r in all_rows.items()}
    column = {arm: {m: [clean(row[m]) for row in r] for m in METRICS} for arm, r in all_rows.items()}
    paired = {name: {m: eval_seg.paired(column["nn"][m], column[other][m]) for m in METRICS} for name, other in (("nn_minus_reference_v2", "reference_v2"), ("nn_minus_reference_v1", "reference_v1"))}
    differences = {m: means["v2_control"][m]-means["reference_v2"][m] for m in ("seg35", "seg4")}
    per_label = {arm: [eval_seg.nanmean([clean(row["per_label"][k]) for row in r]) for k in range(N_CLASSES-1)] for arm, r in all_rows.items()}
    focus = {}
    if FOCUS_SUBJECT in subjects:
        index = subjects.index(FOCUS_SUBJECT)
        keep = [i for i in range(len(subjects)) if i != index]
        focus = dict(subject=FOCUS_SUBJECT, row={arm: {m: clean(r[index][m]) for m in METRICS} for arm, r in all_rows.items()}, others_mean={arm: {m: eval_seg.nanmean([clean(r[i][m]) for i in keep]) for m in METRICS} for arm, r in all_rows.items()})
    document = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), command=argv, code=code, synthetic=synthetic, dev_source=dev_source, subjects=subjects, arms=list(ARMS), means=means, paired=paired, per_label_mean=per_label, focus=focus,
                    control=dict(differences=differences, tolerance=CONTROL_TOL, passed=all(abs(v) <= CONTROL_TOL for v in differences.values())), reference=dict(path=reference["path"], sha256=reference["sha256"]), predictions=records, manifest_sha256=manifest_sha256, n_label_contact=len(contact),
                    allowlist=allowlist_record, allowlist_sha256=allowlist.sha256, folds_sha256=folds.sha256, sealed_lists=sealed.record, seg4_map_sha256=eval_seg.sha_file(eval_seg.SEG4_MAP), groups={k: list(v) for k, v in eval_seg.GROUPS.items()}, truth_sha256=truth_sha, dev_read=dev_read,
                    rows={arm: {s: r[i] for i, s in enumerate(subjects)} for arm, r in rows.items()}, note="开发集，单次观测；分割级读数，不是配准结果。真值只在本评测读取。")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"eval_nn.json").write_text(json.dumps(eval_seg.no_nan(document), indent=1, allow_nan=False)+"\n", encoding="utf-8")
    with (out_dir/"per_subject.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject", "arm", *METRICS])
        for arm, r in all_rows.items():
            for index, subject in enumerate(subjects):
                writer.writerow([subject, arm, *[clean(r[index][m]) for m in METRICS]])
    with (out_dir/"per_label.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "group", *[f"mean_{arm}" for arm in all_rows]])
        for index, label in enumerate(range(1, N_CLASSES)):
            group = next(name for name, labels in eval_seg.GROUPS.items() if label in labels)
            writer.writerow([label, group, *[per_label[arm][index] for arm in all_rows]])
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="prepare_manifest.json: the model's label contact")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, default=V2_ROOT)
    parser.add_argument("--control-segmenter", default=V2_SEGMENTER)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--allowlist", type=Path, default=eval_seg.ALLOWLIST)
    parser.add_argument("--folds", type=Path, default=eval_seg.FOLDS)
    parser.add_argument("--dev-source", default=eval_seg.DEV_SOURCE)
    parser.add_argument("--data-root", type=Path, default=eval_seg.DATA_ROOT)
    parser.add_argument("--sealed-repo", type=Path, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--bz-go", default=None, help="BZ's words and the time: approves another read of the development truth for nnU-Net (recorded in the lock file and in eval_nn.json)")
    args = parser.parse_args(argv)
    try:
        document = evaluate(pred_root=args.pred_root, manifest=args.manifest, out_dir=args.out_dir, v2_root=args.v2_root, control_segmenter=args.control_segmenter, reference_path=args.reference, data_root=args.data_root, allowlist_path=args.allowlist, folds_path=args.folds, dev_source=args.dev_source, sealed_repo=args.sealed_repo, synthetic=args.synthetic, bz_go=args.bz_go, argv=sys.argv if argv is None else ["eval_nn.py", *argv])
    except nn_lock.DevReadLocked as exc:
        print(f"STOP: {exc}", flush=True)
        return 8
    print(json.dumps(dict(means={a: document["means"][a] for a in ("nn", "reference_v2")}, control=document["control"]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
