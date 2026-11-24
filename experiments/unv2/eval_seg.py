"""Segmentation-level evaluation of unet35_v2_full against unet35_v1_ens (work package UNV2, runs/20261103-unv2/plan.md).

For each of the 20 development subjects (the allowlist source stage_L2R_val_v1, 0438-0457) and each set of predictions
(<root>/<subject>/probs36.npz as bayesreg.seg.unet.infer writes it, provenance and output hashes checked, read back by
bayesreg.seg.channels.read_back_probabilities): L = argmax_l P_l (ties: the smallest l, as the arm A5P), hard Dice against the
truth aligned_seg35, label by label; Dice35 = nanmean over labels 1..35; the same nanmean over the small six {4, 23, 18, 34, 19,
35}, the big six {1, 2, 5, 20, 21, 24} and the other 23. seg4: L mapped by configs/bayesreg/seg4_of_seg35_v1.json against the truth
aligned_seg4, labels 1..4 and their mean. Paired per subject: v2 - v1: mean, median, counts of positive, negative and zero
differences.

The preregistered rule (plan.md): the mean Dice35 of v2 more than .002 below that of v1 -> do not run A5Pv2, hand over to BZ
(decision.run_a5pv2, computed on the unrounded means). This is a segmentation-level reading on development subjects, single
observation; it is not a registration result.

The truth is read for this evaluation only (project rules, hard rule 7) and only here: before any label file is opened every subject must
be in that allowlist source, not never_for_training, on no sealed list (bayesreg.eval.truth.load_sealed_lists: the B-line hold-out,
H1, H3, H4; LPBA40 is not an OASIS id) and no training subject of the folds file; SubjectSealed otherwise. Outputs in --out-dir (never
overwritten): eval_seg.json, per_subject.csv, per_label.csv, summary.md.

    python -B experiments/unv2/eval_seg.py --v2 DIR --v1 DIR --out-dir DIR [--data-root DIR]
                                           [--allowlist FILE] [--folds FILE] [--dev-source NAME] [--sealed-repo DIR]
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from jd_curl.three_d.nifti import load_scalar  # noqa: E402

from bayesreg.eval.truth import load_sealed_lists  # noqa: E402
from bayesreg.io.guard import SubjectSealed, load_dev_allowlist, refuse_sealed  # noqa: E402
from bayesreg.io.pair import verify_segmentation_files  # noqa: E402
from bayesreg.seg.channels import read_back_probabilities  # noqa: E402
from bayesreg.seg.unet.data import IMAGE_FILE, LABEL_FILES, load_folds  # noqa: E402
from bayesreg.seg.unet.pins import source_root_for  # noqa: E402
from bayesreg.seg.unet.snapshot import code_record  # noqa: E402

N_CLASSES = 36
N_SEG4 = 5
SMALL_SIX = (4, 23, 18, 34, 19, 35)
BIG_SIX = (1, 2, 5, 20, 21, 24)
OTHER_23 = tuple(l for l in range(1, N_CLASSES) if l not in SMALL_SIX+BIG_SIX)
GROUPS = dict(small_six=SMALL_SIX, big_six=BIG_SIX, other_23=OTHER_23)
RUN_A5PV2_MARGIN = .002         # plan.md: v2 below v1 by more than this: A5Pv2 does not run
DEV_SOURCE = "stage_L2R_val_v1"
ALLOWLIST = ROOT/"configs/bayesreg/dev_subjects.json"
FOLDS = ROOT/"splits/bayesreg_v1/folds.json"
SEG4_MAP = ROOT/"configs/bayesreg/seg4_of_seg35_v1.json"
DATA_ROOT = Path("/mnt/d/data")
FORMAT = "bayesreg.unet35.eval_seg.v1"


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label_dice(pred, true, n_classes):
    """Hard Dice of labels 1..n_classes-1 (array, nan where a label is in neither map)."""
    pred, true = np.asarray(pred, dtype=np.int64).ravel(), np.asarray(true, dtype=np.int64).ravel()
    if pred.shape != true.shape:
        raise ValueError(f"label maps of different sizes: {pred.shape}, {true.shape}")
    overlap = np.bincount(true[true == pred], minlength=n_classes)
    size = np.bincount(true, minlength=n_classes)+np.bincount(pred, minlength=n_classes)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(size > 0, 2.*overlap/size, np.nan)[1:n_classes]


def nanmean(values):
    """Mean over the defined values; nan, without a warning, when there are none (a group with no label in this subject)."""
    values = np.asarray(values, dtype=np.float64)
    defined = values[np.isfinite(values)]
    return float(defined.mean()) if defined.size else float("nan")


def no_nan(node):
    """The document for JSON: an undefined value is null, never the non-standard NaN."""
    if isinstance(node, dict):
        return {k: no_nan(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [no_nan(v) for v in node]
    return None if isinstance(node, float) and not np.isfinite(node) else node


def summarize_subject(pred35, true35, true4, lookup):
    """The row of one subject and one prediction: per-label Dice, Dice35, the three groups, seg4."""
    per_label = label_dice(pred35, true35, N_CLASSES)
    seg4 = label_dice(lookup[pred35], true4, N_SEG4)
    groups = {name: nanmean(per_label[[l-1 for l in labels]]) for name, labels in GROUPS.items()}
    return dict(seg35=nanmean(per_label), seg4=nanmean(seg4), groups=groups, per_label=per_label.tolist(), seg4_per_label=seg4.tolist())


def paired(a, b):
    """a - b per subject: mean, median, counts (positive, negative, zero; nan differences are left out and counted)."""
    d = np.asarray(a, dtype=np.float64)-np.asarray(b, dtype=np.float64)
    ok = d[np.isfinite(d)]
    return dict(mean=float(ok.mean()) if ok.size else None, median=float(np.median(ok)) if ok.size else None,
                n_positive=int((ok > 0).sum()), n_negative=int((ok < 0).sum()), n_zero=int((ok == 0).sum()), n=int(ok.size),
                n_undefined=int(d.size-ok.size))


def admit_subjects(allowlist, source, sealed, training):
    """The subjects of the allowlist source, each checked before any label is read; SubjectSealed otherwise."""
    sources = {s["name"]: s for s in allowlist.sources}
    if source not in sources:
        raise SubjectSealed(f"the allowlist has no source {source!r}")
    subjects = list(sources[source]["subjects"])
    for subject in subjects:
        which = sealed.which(subject)
        if which is not None:
            raise SubjectSealed(f"{subject} is on the sealed list {which}")
        if subject in allowlist.never_for_training or not allowlist.allows(subject) or subject in training:
            raise SubjectSealed(f"{subject}: not a development subject of the evaluation (never_for_training, off the "
                                "allowlist, or a B-line training subject)")
    return subjects


def load_seg4_lookup(path=SEG4_MAP):
    table = {int(k): int(v) for k, v in json.loads(Path(path).read_text())["seg4_of_seg35"].items()}
    if sorted(table) != list(range(N_CLASSES)):
        raise ValueError(f"{path}: the table must cover the labels 0..{N_CLASSES-1}")
    return np.array([table[k] for k in range(N_CLASSES)])


def read_prediction(root, subject, *, expected_segmenter=None):
    """(argmax map uint8, record): provenance of this subject, output files re-hashed, the raw channel sums checked."""
    directory = Path(root)/subject
    provenance = json.loads((directory/"provenance.json").read_text())
    if provenance.get("subject") != subject or provenance.get("layer") != "structure":
        raise ValueError(f"{directory}: the provenance is not of the structure layer of {subject}")
    if expected_segmenter is not None and provenance.get("segmenter_id") != expected_segmenter:
        raise ValueError(f"{directory}: segmenter {provenance.get('segmenter_id')!r}, expected {expected_segmenter!r}")
    if provenance.get("policy") != "all_folds_mean" or provenance.get("test_labels_seen") is not False:
        raise ValueError(f"{directory}: policy {provenance.get('policy')!r}, test_labels_seen "
                         f"{provenance.get('test_labels_seen')!r}: not a prediction of a subject outside every label contact")
    verified = verify_segmentation_files(provenance, directory)
    probs, read_back = read_back_probabilities(directory/"probs36.npz")
    if probs.shape[0] != N_CLASSES:
        raise ValueError(f"{directory}/probs36.npz: {probs.shape[0]} classes, expected {N_CLASSES}")
    argmax = np.argmax(probs, axis=0).astype(np.uint8)                  # ties: the smallest label
    del probs
    record = dict(directory=str(directory), provenance_sha256=sha_file(directory/"provenance.json"),
                  probs36_sha256=verified.get("probs36.npz"), segmenter_id=provenance.get("segmenter_id"),
                  model_ids=provenance.get("model_ids"), n_members=len(provenance.get("members", [])),
                  raw_max_abs_sum_deviation=read_back["raw_max_abs_sum_deviation"],
                  image_sha256=provenance.get("input_sha256", {}).get(IMAGE_FILE),
                  code_commit=(provenance.get("environment", {}).get("code") or {}).get("commit"))
    return argmax, record


def read_truth(data_root, subject, kind):
    """The truth label map of `subject` (C23.6-2, D104 (a): a subject of a sealed list is refused here, before the file is
    opened, whatever the caller admitted; bayesreg.io.guard.refuse_sealed, step (3) of C23.4-2)."""
    refuse_sealed(subject, "eval_seg.read_truth")
    volume = load_scalar(Path(data_root)/subject/LABEL_FILES[kind], labels=True)
    return volume.values.astype(np.uint8), volume.metadata["sha256"]


def decision(mean_v2, mean_v1, margin=RUN_A5PV2_MARGIN):
    difference = mean_v2-mean_v1
    run = not v2_below_v1(mean_v2, mean_v1, margin)
    return dict(rule=f"mean Dice35 of v2 more than {margin} below that of v1: A5Pv2 does not run, hand over to BZ "
                     "(runs/20261103-unv2/plan.md)",
                mean_v2=mean_v2, mean_v1=mean_v1, v2_minus_v1=difference, run_a5pv2=run,
                text=("A5Pv2 may run (a reading, not a conclusion)" if run else "A5Pv2 does not run: hand over to BZ"))


def v2_below_v1(mean_v2, mean_v1, margin=RUN_A5PV2_MARGIN):
    return mean_v1-mean_v2 > margin


def fmt(value, digits=4):
    return "n/a" if value is None or (isinstance(value, float) and not np.isfinite(value)) else f"{value:.{digits}f}"


def summary_markdown(document):
    arms, rows = document["arms"], document["rows"]
    members = "，".join(f"{arm} {'/'.join(str(n) for n in counts)}" for arm, counts in document["n_members"].items())
    lines = ["# unet35_v2_full 对 unet35_v1_ens：分割级评测（开发集，单次观测）", "",
             f"受试者 {len(document['subjects'])} 人（来源 {document['dev_source']}）。argmax 对真值；seg35 与 seg4 同报。这是分割级读数，不是配准结果。", "",
             f"集成成员数：{members}。两臂成员数不同时，差值里同时有训练数据量和集成规模两个因素，不能全归因于数据量。", ""]
    lines += ["| 臂 | Dice35 | seg4 | 小六个 | 大六个 | 其余 23 个 |", "|---|---|---|---|---|---|"]
    for arm in arms:
        def mean(key, sub=None):
            values = [(r[arm][key] if sub is None else r[arm][key][sub]) for r in rows.values()]
            return nanmean(values)
        lines.append(f"| {arm} | {fmt(mean('seg35'))} | {fmt(mean('seg4'))} | {fmt(mean('groups', 'small_six'))} | "
                     f"{fmt(mean('groups', 'big_six'))} | {fmt(mean('groups', 'other_23'))} |")
    lines += ["", "逐受试者配对差（均值、中位差、正 / 负 / 零）：", "",
              "| 比较 | 指标 | 均值差 | 中位差 | 正 | 负 | 零 |", "|---|---|---|---|---|---|---|"]
    for name, comparison in document["paired"].items():
        for metric, p in comparison.items():
            lines.append(f"| {name} | {metric} | {fmt(p['mean'], 5)} | {fmt(p['median'], 5)} | {p['n_positive']} | "
                         f"{p['n_negative']} | {p['n_zero']} |")
    gate = document["decision"]
    lines += ["", f"**预登记规则**：{gate['rule']}。v2 − v1 = {fmt(gate['v2_minus_v1'], 5)}。{gate['text']}。", ""]
    lines += ["逐标签表在 per_label.csv。", ""]
    return "\n".join(lines)


def evaluate(*, v2_root, v1_root, out_dir, data_root=DATA_ROOT, allowlist_path=ALLOWLIST, folds_path=FOLDS,
             dev_source=DEV_SOURCE, sealed_repo=None, seg4_map=SEG4_MAP, v2_segmenter="unet35_v2_full",
             v1_segmenter="unet35_v1_ens", argv=None, log=print):
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} exists and is not empty; the evaluation is never overwritten")
    code = code_record()
    sealed = load_sealed_lists(Path(sealed_repo) if sealed_repo is not None else source_root_for(code))
    allowlist = load_dev_allowlist(allowlist_path)
    folds = load_folds(folds_path)
    subjects = admit_subjects(allowlist, dev_source, sealed, folds.training_subjects)   # before any label is read
    lookup = load_seg4_lookup(seg4_map)
    roots = dict(v1=(v1_root, v1_segmenter), v2=(v2_root, v2_segmenter))
    rows, records, truth_sha = {}, {}, {}
    for subject in subjects:
        predictions = {arm: read_prediction(root, subject, expected_segmenter=segmenter)
                       for arm, (root, segmenter) in roots.items()}
        images = {r["image_sha256"] for _, r in predictions.values()}
        if len(images) != 1:
            raise ValueError(f"{subject}: the predictions were made from different images: {sorted(images)}")
        true35, sha35 = read_truth(data_root, subject, "seg35")
        true4, sha4 = read_truth(data_root, subject, "seg4")
        truth_sha[subject] = dict(seg35=sha35, seg4=sha4)
        rows[subject] = {arm: summarize_subject(pred, true35, true4, lookup) for arm, (pred, _) in predictions.items()}
        records[subject] = {arm: record for arm, (_, record) in predictions.items()}
        log(f"{subject}: "+", ".join(f"{a} seg35 {rows[subject][a]['seg35']:.4f} seg4 {rows[subject][a]['seg4']:.4f}" for a in roots))

    def column(arm, metric):
        if metric in ("seg35", "seg4"):
            return [rows[s][arm][metric] for s in subjects]
        return [rows[s][arm]["groups"][metric] for s in subjects]

    metrics = ("seg35", "seg4")+tuple(GROUPS)
    comparisons = dict(v2_minus_v1=("v2", "v1"))
    paired_rows = {name: {m: paired(column(a, m), column(b, m)) for m in metrics} for name, (a, b) in comparisons.items()}
    means = {arm: {m: nanmean(column(arm, m)) for m in metrics} for arm in roots}
    per_label_mean = {arm: [nanmean([rows[s][arm]["per_label"][k] for s in subjects]) for k in range(N_CLASSES-1)] for arm in roots}
    document = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), command=argv, code=code,
                    dev_source=dev_source, subjects=subjects, arms=list(roots), roots={a: str(r) for a, (r, _) in roots.items()},
                    allowlist_path=str(allowlist_path), allowlist_sha256=allowlist.sha256, folds_sha256=folds.sha256,
                    sealed_lists=sealed.record, seg4_map_sha256=sha_file(seg4_map), groups={k: list(v) for k, v in GROUPS.items()},
                    n_members={arm: sorted({records[s][arm]["n_members"] for s in subjects}) for arm in roots},
                    means=means, per_label_mean=per_label_mean, paired=paired_rows, decision=decision(means["v2"]["seg35"], means["v1"]["seg35"]),
                    truth_sha256=truth_sha, predictions=records, rows=rows,
                    note="开发集，单次观测；分割级读数，不是配准结果。真值只在本评测读取。")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"eval_seg.json").write_text(json.dumps(no_nan(document), indent=1, allow_nan=False)+"\n", encoding="utf-8")
    with (out_dir/"per_subject.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject", "arm", *metrics])
        for subject in subjects:
            for arm in roots:
                writer.writerow([subject, arm]+[rows[subject][arm][m] if m in ("seg35", "seg4") else rows[subject][arm]["groups"][m] for m in metrics])
    with (out_dir/"per_label.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "group", *[f"mean_{a}" for a in roots], *[f"{n}_mean" for n in comparisons], *[f"{n}_n_positive" for n in comparisons]])
        for index, label in enumerate(range(1, N_CLASSES)):
            group = next(name for name, labels in GROUPS.items() if label in labels)
            diffs = {name: paired([rows[s][a]["per_label"][index] for s in subjects], [rows[s][b]["per_label"][index] for s in subjects])
                     for name, (a, b) in comparisons.items()}
            writer.writerow([label, group, *[per_label_mean[a][index] for a in roots], *[diffs[n]["mean"] for n in comparisons],
                             *[diffs[n]["n_positive"] for n in comparisons]])
    (out_dir/"summary.md").write_text(summary_markdown(document), encoding="utf-8")
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v2", type=Path, required=True, help="predictions of unet35_v2_full")
    parser.add_argument("--v1", type=Path, required=True, help="predictions of unet35_v1_ens")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--allowlist", type=Path, default=ALLOWLIST)
    parser.add_argument("--folds", type=Path, default=FOLDS)
    parser.add_argument("--dev-source", default=DEV_SOURCE)
    parser.add_argument("--sealed-repo", type=Path, default=None, help="default: the snapshot's source repository")
    parser.add_argument("--v1-segmenter", default="unet35_v1_ens")
    args = parser.parse_args(argv)
    document = evaluate(v2_root=args.v2, v1_root=args.v1, out_dir=args.out_dir, data_root=args.data_root,
                        allowlist_path=args.allowlist, folds_path=args.folds, dev_source=args.dev_source, sealed_repo=args.sealed_repo,
                        v1_segmenter=args.v1_segmenter, argv=sys.argv if argv is None else ["eval_seg.py", *argv])
    print(json.dumps(dict(means=document["means"], decision=document["decision"]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
