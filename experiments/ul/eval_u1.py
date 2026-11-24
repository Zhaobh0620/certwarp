"""Segmentation-level evaluation of the arm U1 (work package UL, runs/20261104-ul/plan.md), on the terms of the single members of unet35_v2_full (runs/20261103-unv2/v2_member_subsets.py): the 20 development subjects, argmax against the truth,
Dice per label, seg35 = nanmean over labels 1..35, the small six, the big six, the other 23, seg4 through configs/bayesreg/seg4_of_seg35_v1.json (experiments/unv2/eval_seg.py summarize_subject, the same functions); plus the forward seconds per subject.

Arms, one forward pass each per subject, no prediction file is kept:
  U1             the member of --member-root, sliding window as trained: the window, the overlap .5 and the Gaussian window of member.pt. The arm of the reading.
  U1_uniform     the same member and windows, plain mean of the logits. Described only: what the Gaussian window changes.
  v2_A0_control  member foldA_s0 of unet35_v2_full (fold A, seed 0: the same training subjects as U1) through this same code, whole volume as registered. A control of the evaluator: its numbers must reproduce the reference.
Reference: the committed per-subject numbers of member A0 in runs/20261103-unv2/v2_member_subsets.json. There "A0" is fold A, seed 0; the .9230 / .9566 of the plan's brief is the mean over the four single members (`curve`, n = 1), not this member.

Order of the checks, before any image or label is opened: every subject in the allowlist source stage_L2R_val_v1, not never_for_training (T0), on no sealed list (B-line hold-out, H1, H3, H4; eval_seg.admit_subjects), no training subject, and outside the label contact of both members; SubjectSealed
otherwise. Real data: the repository allowlist re-hashed, the reference and the control member pinned by sha256, clean code. --synthetic: synthetic ids and a non-real data root only, no pins.
Forward seconds: train.predict_logits and the argmax, with the device synchronized around them; reading the image, normalizing and the Dice are outside. Outputs in --out-dir (never overwritten): eval_u1.json, per_subject.csv, per_label.csv.

The development truth is read once for U1, across results roots: on real data the lock file of dev_lock.py (a fixed path, no option changes it) refuses a second read, whatever --member-root or --out-dir say; only BZ's approval, --bz-go "<words and the time>", goes on, and it is recorded. Synthetic runs do not touch the lock.

--control-only (plan revision 1; review, minor 7): the arm v2_A0_control alone, before the training, so that a failure of the real path or an evaluator that does not reproduce the reference shows before 4.5 hours of training and before the one read of U1. No member of U1 is needed. Its numbers were read long ago, so it adds nothing about U1, and it takes no lock. Output eval_control.json with `passed`;
exit 8 if it did not pass. It checks the whole-volume path only; the sliding window has unit tests.

    python -B experiments/ul/eval_u1.py --member-root DIR --out-dir DIR [--device cuda] [--bz-go TEXT]
    python -B experiments/ul/eval_u1.py --control-only --out-dir DIR [--device cuda]
    python -B experiments/ul/eval_u1.py --synthetic --member-root DIR --control-member DIR --reference FILE --allowlist FILE --folds FILE --dev-source NAME --data-root DIR --out-dir DIR --device cpu
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO/"experiments/unv2"), str(REPO/"experiments/ul")]
from decide_ul import CONTROL_TOL  # noqa: E402
import dev_lock  # noqa: E402
import eval_seg  # noqa: E402
from bayesreg.eval.truth import load_sealed_lists  # noqa: E402
from bayesreg.io.guard import SubjectSealed, load_dev_allowlist  # noqa: E402
from bayesreg.io.pair import subject_list_sha256  # noqa: E402
from bayesreg.seg.unet.data import FOLD_INDEX, OasisFiles, load_folds, normalize_image, read_image  # noqa: E402
from bayesreg.seg.unet.infer import load_member  # noqa: E402
from bayesreg.seg.unet.large import GIB, PROFILE, cap_gpu_memory, windows_per_volume  # noqa: E402
from bayesreg.seg.unet.pins import check_synthetic_data_root, check_synthetic_subjects, load_pinned_allowlist, source_root_for  # noqa: E402
from bayesreg.seg.unet.snapshot import code_record  # noqa: E402
from bayesreg.seg.unet.train import file_sha256, predict_logits  # noqa: E402

FORMAT = "bayesreg.unet35.eval_u1.v1"
FORMAT_CONTROL = "bayesreg.unet35.eval_u1_control.v1"
FOLD, SEED, MEMBER_DIR = "A", 0, "foldA_s0"
REFERENCE = REPO/"runs/20261103-unv2/v2_member_subsets.json"
REFERENCE_SHA256 = "da54e15ac594ea63b0e8966409d76773a896d60a0a254d27c42495cf98a1c6ee"
REFERENCE_MEMBER = "A0"                                 # fold A, seed 0 (blob fold + seed in v2_member_subsets.py)
CONTROL_MEMBER = Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v2_full/foldA_s0")
CONTROL_SHA256 = "756d8c1c241a292a538a6ddadcf5c271b589b8d76bc05bbca9d2f35a82795acd"     # configs/bayesreg/segmenters_v2.json, unet35_v2_full_foldA, first member
MEMBER_ROOT = Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v3_large")
GPU_CAP_GIB = 20.
ARMS = ("U1", "U1_uniform", "v2_A0_control")
METRICS = ("seg35", "seg4")+tuple(eval_seg.GROUPS)


def read_reference(path, *, pinned_sha256=None, member=REFERENCE_MEMBER):
    """The per-subject rows of the reference member and the summaries; ValueError if the file is not what was pinned or its summary does not follow from its rows."""
    raw = Path(path).read_bytes()
    if pinned_sha256 is not None and hashlib.sha256(raw).hexdigest() != pinned_sha256:
        raise ValueError(f"{path}: sha256 is not the pinned {pinned_sha256}")
    document = json.loads(raw.decode("utf-8"))
    rows, subjects = document["per_subject"][member], document["subjects"]
    if len(rows) != len(subjects):
        raise ValueError(f"{path}: {len(rows)} rows of {member} for {len(subjects)} subjects")
    means = dict(seg35=float(np.mean([r["seg35"] for r in rows])), seg4=float(np.mean([r["seg4"] for r in rows])), **{g: float(np.mean([r["groups"][g] for r in rows])) for g in eval_seg.GROUPS})
    if abs(means["seg35"]-document["summary"][member]["seg35"]) > 1e-12 or abs(means["seg4"]-document["summary"][member]["seg4"]) > 1e-12:
        raise ValueError(f"{path}: the summary of {member} does not follow from its rows")
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), member=member, subjects=list(subjects), rows=rows, means=means, single_member_mean=document["curve"].get("1"), ensemble=document["curve"].get(str(len(document["members"]))),
                members=document["members"])


def member_files(directory):
    """(path of member.pt, sha256, the record member.json) of a member directory, re-hashed."""
    directory = Path(directory)
    record = json.loads((directory/"member.json").read_text(encoding="utf-8"))
    path = directory/record["member_file"]
    sha = file_sha256(path)
    if sha != record["member_sha256"]:
        raise ValueError(f"{path}: sha256 differs from {directory/'member.json'}")
    if subject_list_sha256(record["label_contact_subjects"]) != record["label_contact_subjects_sha256"]:
        raise ValueError(f"{directory/'member.json'}: label_contact_subjects do not match their sha256")
    return path, sha, record


def load_arm(directory, device):
    """(model, blob, record) through infer.load_member: the sha256 again, then what member.pt carries against member.json."""
    path, sha, record = member_files(directory)
    entry = dict(member_model_sha256=[sha], fold=FOLD_INDEX[record["fold"]], label_contact_subjects=record["label_contact_subjects"], label_contact_subjects_sha256=record["label_contact_subjects_sha256"])
    model, blob = load_member(path, sha, device, entry=entry)
    return model, blob, record


def check_contact(subjects, contact, name):
    """No evaluated subject in the member's label contact."""
    overlap = sorted(set(subjects) & set(contact))
    if overlap:
        raise SubjectSealed(f"{overlap} in the label contact of {name}; not evaluated")


def admit(allowlist, source, sealed, training, contacts):
    """The subjects of the evaluation, all checks before any file of a subject is opened. contacts: {arm: label contact}."""
    subjects = eval_seg.admit_subjects(allowlist, source, sealed, training)
    for name, contact in contacts.items():
        check_contact(subjects, contact, name)
    return subjects


def timed_forward(model, image, *, device, amp, window, sigma):
    """(argmax uint8 numpy, seconds)."""
    device = torch.device(device)
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    logits = predict_logits(model, image, device=device, amp=amp, window=window, gaussian_sigma=sigma)
    top = logits.argmax(1)[0].to(torch.uint8).cpu().numpy()
    if cuda:
        torch.cuda.synchronize(device)
    return top, time.perf_counter()-started


def mean_of(rows, metric):
    return eval_seg.nanmean([r[metric] if metric in ("seg35", "seg4") else r["groups"][metric] for r in rows])


def column(rows, metric):
    return [r[metric] if metric in ("seg35", "seg4") else r["groups"][metric] for r in rows]


def seconds_summary(values):
    return dict(mean=float(np.mean(values)), median=float(np.median(values)), minimum=float(np.min(values)), maximum=float(np.max(values)), n=len(values), per_subject=[float(v) for v in values])


def read_inputs(*, reference_path, data_root, allowlist_path, dev_source, synthetic):
    """(allowlist, its record, the reference): pinned on real data; synthetic ids and a non-real data root otherwise."""
    if synthetic:
        check_synthetic_data_root(data_root)
        allowlist = load_dev_allowlist(allowlist_path)
        check_synthetic_subjects([s for ids in allowlist.subjects.values() for s in ids], "the allowlist")
        return allowlist, dict(mode="synthetic", allowlist_path=str(allowlist_path)), read_reference(reference_path)
    if Path(reference_path).resolve() != REFERENCE.resolve() or dev_source != eval_seg.DEV_SOURCE or Path(allowlist_path).resolve() != eval_seg.ALLOWLIST.resolve():
        raise ValueError("real data: the reference, the allowlist and the development source are the plan's; other values only with --synthetic")
    allowlist, record = load_pinned_allowlist(allowlist_path)
    return allowlist, {k: v for k, v in record.items() if k != "code"}, read_reference(reference_path, pinned_sha256=REFERENCE_SHA256)


def evaluate_control(*, out_dir, reference_path=REFERENCE, control_member=CONTROL_MEMBER, data_root=eval_seg.DATA_ROOT, allowlist_path=eval_seg.ALLOWLIST, folds_path=eval_seg.FOLDS, dev_source=eval_seg.DEV_SOURCE, sealed_repo=None, synthetic=False, device="cuda", argv=None, log=print):
    """--control-only: member A0 through the same functions as the arm v2_A0_control, against the reference; `passed` with the tolerance of the reading. Takes no lock, needs no member of U1."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} exists and is not empty; the evaluation is never overwritten")
    code = code_record()
    allowlist, allowlist_record, reference = read_inputs(reference_path=reference_path, data_root=data_root, allowlist_path=allowlist_path, dev_source=dev_source, synthetic=synthetic)
    control_path, control_sha, control_record = member_files(control_member)
    if not synthetic and control_sha != CONTROL_SHA256:
        raise ValueError(f"{control_path}: sha256 {control_sha} is not the pinned control member")
    if (control_record["fold"], control_record["seed"]) != (FOLD, SEED):
        raise ValueError("the control is fold A, seed 0")
    sealed = load_sealed_lists(Path(sealed_repo) if sealed_repo is not None else source_root_for(code))
    folds = load_folds(folds_path)
    subjects = admit(allowlist, dev_source, sealed, folds.training_subjects, dict(v2_A0_control=control_record["label_contact_subjects"]))
    if subjects != reference["subjects"]:
        raise ValueError(f"the subjects differ from the reference's: {sorted(set(subjects) ^ set(reference['subjects']))[:5]}")
    device = torch.device(device)
    if device.type == "cuda":
        cap_gpu_memory(device, GPU_CAP_GIB)
    model, blob, _ = load_arm(control_member, device)
    window = tuple(blob["window"]) if blob.get("window") else None
    lookup = eval_seg.load_seg4_lookup()
    files = OasisFiles(Path(data_root))
    rows = []
    for subject in subjects:
        values, _ = read_image(files, subject)
        image, _ = normalize_image(values, percentile=blob["percentile"])
        top, seconds = timed_forward(model, image, device=device, amp=blob["amp"], window=window, sigma=None)
        true35, _ = eval_seg.read_truth(data_root, subject, "seg35")
        true4, _ = eval_seg.read_truth(data_root, subject, "seg4")
        rows.append(dict(eval_seg.summarize_subject(top, true35, true4, lookup), forward_seconds=seconds, subject=subject))
        log(f"{subject}: v2_A0_control {rows[-1]['seg35']:.4f} / {rows[-1]['seg4']:.4f} ({seconds:.2f} s)")
    means = {m: mean_of(rows, m) for m in METRICS}
    differences = {m: means[m]-reference["means"][m] for m in ("seg35", "seg4")}
    document = dict(format=FORMAT_CONTROL, created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), command=argv, code=code, synthetic=synthetic, subjects=subjects, means=means, reference=reference["means"], reference_sha256=reference["sha256"], differences=differences, tolerance=CONTROL_TOL,
                    passed=all(abs(v) <= CONTROL_TOL for v in differences.values()), member_sha256=control_sha, seconds=seconds_summary([r["forward_seconds"] for r in rows]), allowlist=allowlist_record, device=str(device), gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/GIB if device.type == "cuda" else None,
                    per_subject={r["subject"]: {k: r[k] for k in ("seg35", "seg4", "groups", "forward_seconds")} for r in rows}, note="只有 v2 的 A0，这些数早就读过，不增加任何 U1 的信息；不取读取锁。")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"eval_control.json").write_text(json.dumps(eval_seg.no_nan(document), indent=1, allow_nan=False)+"\n", encoding="utf-8")
    return document


def evaluate(*, member_root, out_dir, reference_path=REFERENCE, control_member=CONTROL_MEMBER, data_root=eval_seg.DATA_ROOT, allowlist_path=eval_seg.ALLOWLIST, folds_path=eval_seg.FOLDS, dev_source=eval_seg.DEV_SOURCE, sealed_repo=None, synthetic=False,
             device="cuda", argv=None, log=print, bz_go=None):
    if synthetic and bz_go is not None:
        raise ValueError("--bz-go is for the read lock of real data; synthetic runs do not touch it")
    if not synthetic:
        dev_lock.check(bz_go)                            # the first thing on real data: no other option leads around it
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} exists and is not empty; the evaluation is never overwritten")
    code = code_record()
    allowlist, allowlist_record, reference = read_inputs(reference_path=reference_path, data_root=data_root, allowlist_path=allowlist_path, dev_source=dev_source, synthetic=synthetic)
    u1_dir = Path(member_root)/MEMBER_DIR
    u1_path, u1_sha, u1_record = member_files(u1_dir)
    control_path, control_sha, control_record = member_files(control_member)
    if not synthetic and control_sha != CONTROL_SHA256:
        raise ValueError(f"{control_path}: sha256 {control_sha} is not the pinned control member")
    if (u1_record["fold"], u1_record["seed"]) != (FOLD, SEED) or (control_record["fold"], control_record["seed"]) != (FOLD, SEED):
        raise ValueError("U1 and the control are fold A, seed 0 (the same training subjects); a different member is not this reading")
    train_record = json.loads((u1_dir/"train_record.json").read_text(encoding="utf-8"))
    if train_record.get("recipe") != PROFILE:
        raise ValueError(f"{u1_dir}: recipe {train_record.get('recipe')!r}, expected {PROFILE!r}")
    sealed = load_sealed_lists(Path(sealed_repo) if sealed_repo is not None else source_root_for(code))
    folds = load_folds(folds_path)
    subjects = admit(allowlist, dev_source, sealed, folds.training_subjects, dict(U1=u1_record["label_contact_subjects"], v2_A0_control=control_record["label_contact_subjects"]))
    if subjects != reference["subjects"]:                # before any subject file is opened: a mismatch means this is not the reference's evaluation
        raise ValueError(f"the subjects differ from the reference's: {sorted(set(subjects) ^ set(reference['subjects']))[:5]}")
    device = torch.device(device)
    if device.type == "cuda":
        cap_gpu_memory(device, GPU_CAP_GIB)
    u1, u1_blob, _ = load_arm(u1_dir, device)
    control, control_blob, _ = load_arm(control_member, device)
    window, sigma = tuple(u1_blob["window"]), u1_blob.get("window_gaussian_sigma")
    if sigma is None:
        raise ValueError(f"{u1_path}: no window_gaussian_sigma: not a member of {PROFILE}")
    control_window = tuple(control_blob["window"]) if control_blob.get("window") else None
    lookup = eval_seg.load_seg4_lookup()
    files = OasisFiles(Path(data_root))
    specs = dict(U1=(u1, u1_blob, window, sigma), U1_uniform=(u1, u1_blob, window, None), v2_A0_control=(control, control_blob, control_window, None))
    rows, truth_sha, image_sha, shape, dev_read = {arm: [] for arm in ARMS}, {}, {}, None, None
    for subject in subjects:
        values, meta = read_image(files, subject)
        image_sha[subject] = meta["sha256"]
        shape = tuple(meta["grid_shape_zyx"])
        tops, images = {}, {}
        for arm, (model, blob, arm_window, arm_sigma) in specs.items():
            if blob["percentile"] not in images:
                images[blob["percentile"]] = normalize_image(values, percentile=blob["percentile"])[0]
            tops[arm] = timed_forward(model, images[blob["percentile"]], device=device, amp=blob["amp"], window=arm_window, sigma=arm_sigma)
        del values, images
        if not synthetic and dev_read is None:           # the moment before the first truth label; a failure before it leaves no lock
            dev_read = dev_lock.take(dict(member_sha256=u1_sha, member_root=str(member_root), out_dir=str(out_dir), command=argv, code_commit=code.get("commit")), bz_go=bz_go)
        true35, sha35 = eval_seg.read_truth(data_root, subject, "seg35")
        true4, sha4 = eval_seg.read_truth(data_root, subject, "seg4")
        truth_sha[subject] = dict(seg35=sha35, seg4=sha4)
        for arm, (top, seconds) in tops.items():
            rows[arm].append(dict(eval_seg.summarize_subject(top, true35, true4, lookup), forward_seconds=seconds, subject=subject))
        log(f"{subject}: "+", ".join(f"{arm} {rows[arm][-1]['seg35']:.4f} / {rows[arm][-1]['seg4']:.4f} ({rows[arm][-1]['forward_seconds']:.2f} s)" for arm in ARMS))
    means = {arm: {m: mean_of(rows[arm], m) for m in METRICS} for arm in ARMS}
    reference_columns = {m: [(r[m] if m in ("seg35", "seg4") else r["groups"][m]) for r in reference["rows"]] for m in METRICS}
    paired = {f"{arm}_minus_reference": {m: eval_seg.paired(column(rows[arm], m), reference_columns[m]) for m in METRICS} for arm in ARMS}
    per_label = {arm: [eval_seg.nanmean([r["per_label"][k] for r in rows[arm]]) for k in range(eval_seg.N_CLASSES-1)] for arm in ARMS}
    n_windows, starts = windows_per_volume(shape, window, .5)
    document = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), command=argv, code=code, synthetic=synthetic, dev_source=dev_source, subjects=subjects, arms=list(ARMS), means=means, paired=paired, per_label_mean=per_label,
                    seconds={arm: seconds_summary([r["forward_seconds"] for r in rows[arm]]) for arm in ARMS}, reference={k: v for k, v in reference.items() if k != "rows"}, reference_rows=reference["rows"],
                    window=dict(window=list(window), overlap=.5, gaussian_sigma=sigma, volume=list(shape), n_windows=n_windows, starts=starts), control_window=list(control_window) if control_window else None,
                    members=dict(U1=dict(directory=str(u1_dir), member_sha256=u1_sha, fold=u1_record["fold"], seed=u1_record["seed"], temperature=u1_blob["temperature"], best_iteration=u1_blob["best_iteration"]),
                                 v2_A0_control=dict(directory=str(control_member), member_sha256=control_sha, fold=control_record["fold"], seed=control_record["seed"], temperature=control_blob["temperature"], best_iteration=control_blob["best_iteration"])),
                    allowlist=allowlist_record, allowlist_sha256=allowlist.sha256, folds_sha256=folds.sha256, sealed_lists=sealed.record, seg4_map_sha256=eval_seg.sha_file(eval_seg.SEG4_MAP), groups={k: list(v) for k, v in eval_seg.GROUPS.items()},
                    image_sha256=image_sha, truth_sha256=truth_sha, dev_read=dev_read, device=str(device), gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/GIB if device.type == "cuda" else None,
                    rows={arm: {r["subject"]: {k: r[k] for k in ("seg35", "seg4", "groups", "per_label", "seg4_per_label", "forward_seconds")} for r in rows[arm]} for arm in ARMS},
                    note="开发集，单次观测；分割级读数，不是配准结果。真值只在本评测读取。")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"eval_u1.json").write_text(json.dumps(eval_seg.no_nan(document), indent=1, allow_nan=False)+"\n", encoding="utf-8")
    with (out_dir/"per_subject.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject", "arm", *METRICS, "forward_seconds"])
        for index, subject in enumerate(subjects):
            for arm in ARMS:
                row = rows[arm][index]
                writer.writerow([subject, arm, row["seg35"], row["seg4"], *[row["groups"][g] for g in eval_seg.GROUPS], row["forward_seconds"]])
            row = reference["rows"][index]
            writer.writerow([subject, f"reference_{reference['member']}", row["seg35"], row["seg4"], *[row["groups"][g] for g in eval_seg.GROUPS], ""])
    with (out_dir/"per_label.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "group", *[f"mean_{a}" for a in ARMS]])
        for index, label in enumerate(range(1, eval_seg.N_CLASSES)):
            group = next(name for name, labels in eval_seg.GROUPS.items() if label in labels)
            writer.writerow([label, group, *[per_label[a][index] for a in ARMS]])
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--member-root", type=Path, default=MEMBER_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--control-member", type=Path, default=CONTROL_MEMBER)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--allowlist", type=Path, default=eval_seg.ALLOWLIST)
    parser.add_argument("--folds", type=Path, default=eval_seg.FOLDS)
    parser.add_argument("--dev-source", default=eval_seg.DEV_SOURCE)
    parser.add_argument("--data-root", type=Path, default=eval_seg.DATA_ROOT)
    parser.add_argument("--sealed-repo", type=Path, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--control-only", action="store_true", help="only the control member against the reference; no lock, no member of U1")
    parser.add_argument("--bz-go", default=None, help="BZ's words and the time: approves another read of the development truth for U1 (recorded in the lock file and in eval_u1.json)")
    args = parser.parse_args(argv)
    common = dict(out_dir=args.out_dir, reference_path=args.reference, control_member=args.control_member, data_root=args.data_root, allowlist_path=args.allowlist, folds_path=args.folds, dev_source=args.dev_source, sealed_repo=args.sealed_repo, synthetic=args.synthetic, device=args.device,
                  argv=sys.argv if argv is None else ["eval_u1.py", *argv])
    if args.control_only:
        if args.bz_go is not None:
            parser.error("--bz-go does not go with --control-only: the control takes no lock")
        document = evaluate_control(**common)
        print(json.dumps({k: document[k] for k in ("passed", "differences", "tolerance", "means")}, indent=1))
        return 0 if document["passed"] else 8
    try:
        document = evaluate(member_root=args.member_root, bz_go=args.bz_go, **common)
    except dev_lock.DevReadLocked as exc:
        print(f"STOP: {exc}", flush=True)
        return 8
    print(json.dumps(dict(means=document["means"], seconds={a: {k: v for k, v in s.items() if k != "per_subject"} for a, s in document["seconds"].items()}, window=document["window"]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
