"""The nnU-Net probabilities of arm (b) of work package UF (runs/20261105-uf/plan.md): NX's step 1 (runs/20261105-fx/plan.md) with this package's root. Only images are read, never a truth label; neither seg/nn_dev_read.lock nor seg/uf_dev_read.lock is touched; nothing here imports torch or nnunetv2.

    infer     <out>/nn/dev_input (links to the 20 images only), then experiments/nn/infer_nn.py with --save-probabilities, unchanged, run by experiments/fl/nx_infer_child.py (the 9 GiB device cap) in the environment nnunet (nn_common.nn_env, gpu on): <out>/nn/raw/<subject>.{npz,pkl,nii.gz}, infer_record.json; <out>/nn/infer.log, infer_check.json.
              The weights are NN's (nnunet_v1/train_record.json: trainer, checkpoint_final.pth re-hashed); the record must show them, the 20 subjects, saved probabilities, no mirroring axes, one view, a reserved peak <= 9 GiB.
    convert   one subject at a time, nx_seg's functions: nnU-Net's (classes, SimpleITK order) into load_scalar's (classes, nz, ny, nx), float16 with the argmax of this run kept (one ULP where float16 makes a tie), <out>/nn/probs/<subject>/probs36.npz (key p), read back by bayesreg.seg.channels.read_back_probabilities.
              The gate (FX plan revision 1): the argmax of what is stored, as the reader reads it, against nnunet_v1/pred_dev (the labels NN read): voxels that differ / all voxels <= 1e-4 per subject, else GateFailed and no output of that subject stays (its probs36.npz is removed; raw/ is kept and
              convert_stop.json records why, so the evidence can be rebuilt from raw/: hard rule 4). <out>/nn/convert/<subject>.json, <out>/nn/convert_record.json.
No provenance in the registry's format is written: nnU-Net is not registered here (the brief: no registration), and these files feed the one evaluation of eval_uf.py only.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path[:0] = [str(HERE), str(EXPERIMENTS/"fl"), str(EXPERIMENTS/"ef"), str(EXPERIMENTS/"mc0"), str(EXPERIMENTS/"nn")]

import nn_common as nc  # noqa: E402
import nx_seg  # noqa: E402
import uf_common as uc  # noqa: E402

PROBS_FILE = nx_seg.PROBS_FILE
INFER_PEAK_GIB = 9.                             # nx_infer_child.LIMIT_BYTES; run_fx.INFER_PEAK_GIB
FALLBACK_LINE = "Prediction on device was unsuccessful"      # nnunetv2 2.8.1, predict_from_raw_data.py (run_fx.FALLBACK_LINE)
CHILD = EXPERIMENTS/"fl"/"nx_infer_child.py"
GateFailed = nx_seg.GateFailed


class NNStepFailed(Exception):
    """The nnU-Net step did not give what the plan needs; kept, not rerun."""


def nn_dir(out):
    return Path(out)/"nn"


def read_nn_training(nn_out=uc.NN_OUT):
    """nnunet_v1/train_record.json, with checkpoint_final.pth re-hashed against it."""
    path = Path(nn_out)/"train_record.json"
    if not path.is_file():
        raise NNStepFailed(f"{path}: no such file; NN's training record is required")
    train = json.loads(path.read_text(encoding="utf-8"))
    if nx_seg.file_sha256(train["final_checkpoint"]) != train["final_checkpoint_sha256"]:
        raise NNStepFailed(f"{train['final_checkpoint']}: sha256 differs from {path}")
    return train


def infer_argv(train, *, out, nn_out=uc.NN_OUT):
    """infer_nn.py in the environment nnunet with --save-probabilities, behind nx_infer_child.py (run_fx.nx_infer_argv with this package's directories)."""
    argv = nc.infer_argv(model_dir=nc.trainer_directory(train["trainer"], results=nc.results_root(nn_out)), input_dir=nn_dir(out)/"dev_input", out_dir=nn_dir(out)/"raw")+["--save-probabilities"]
    return argv[:3]+[str(CHILD)]+argv[3:]


def infer_fallbacks(text):
    """(lines, subjects): where nnU-Net fell back to the CPU for the results of a subject. A copy of experiments/fl/run_fx.py infer_fallbacks (that module imports the whole FX line); a test compares them."""
    lines, subjects, pending = 0, [], False
    for line in text.splitlines():
        if FALLBACK_LINE in line:
            lines, pending = lines+1, True
            continue
        head, _, tail = line.partition(": ")
        if pending and head.startswith("OASIS_") and tail.endswith(" s"):
            subjects.append(head)
            pending = False
    return dict(n_lines=lines, fallback_subjects=subjects)


def infer_problems(record, train, subjects):
    """Why infer_record.json is not this plan's inference (empty: it is)."""
    problems = []
    if not record:
        return ["no infer_record.json"]
    if record.get("checkpoint_sha256") != train["final_checkpoint_sha256"]:
        problems.append("not NN's checkpoint_final")
    if record.get("saved_probabilities") is not True:
        problems.append("no probabilities saved")
    if sorted(record.get("subjects") or []) != sorted(subjects):
        problems.append("not the 20 development subjects")
    if record.get("allowed_mirroring_axes") is not None or record.get("tta_views") != 1:
        problems.append("mirroring axes or more than one view")
    if (record.get("gpu_peak_reserved_gib") or 0.) > INFER_PEAK_GIB:
        problems.append(f"reserved peak above {INFER_PEAK_GIB:g} GiB")
    return problems


def infer(out, subjects, *, nn_out=uc.NN_OUT, data_root=nc.DATA_ROOT, run=subprocess.run, env=None):
    """The inference, once: a dev_input that exists is never redone. Returns the check record; NNStepFailed otherwise (the files are kept)."""
    directory = nn_dir(out)
    inputs = directory/"dev_input"
    if inputs.exists():
        raise NNStepFailed(f"{inputs} exists: the inference started before; never rerun, the lead decides")
    train = read_nn_training(nn_out)
    inputs.mkdir(parents=True)
    for subject in subjects:
        (inputs/f"{subject}_0000.nii.gz").symlink_to(Path(data_root)/subject/nx_seg.IMAGE_FILE)
    with open(directory/"infer.log", "ab") as log:
        code = run(infer_argv(train, out=out, nn_out=nn_out), env=nc.nn_env(nn_out, gpu=True) if env is None else env, cwd=str(directory), stdout=log, stderr=subprocess.STDOUT).returncode
    path = directory/"raw"/"infer_record.json"
    record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    problems = ([f"infer_nn.py exit {code}"] if code != 0 else [])+infer_problems(record, train, subjects)
    check = dict(schema="bayesreg.uf.nn_infer_check.v1", returncode=code, problems=problems, **infer_fallbacks((directory/"infer.log").read_text(errors="replace")), gpu_peak_allocated_gib=record.get("gpu_peak_allocated_gib"), gpu_peak_reserved_gib=record.get("gpu_peak_reserved_gib"),
                 device_cap_gib=INFER_PEAK_GIB, checkpoint_sha256=train["final_checkpoint_sha256"], trainer=train["trainer"], subjects=list(subjects), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    uc.write_json_atomic(directory/"infer_check.json", check)
    if problems:
        raise NNStepFailed(f"the nnU-Net inference is not the plan's: {problems}; kept, not rerun")
    return check


def convert_subject(subject, *, raw_dir, pred_dev, image_path, out_root, log=lambda text: None):
    """One subject: the gate on the raw argmax, probs36.npz, its read-back, the gate on the stored argmax. Returns the subject's record; GateFailed leaves nothing of the subject. Never overwritten."""
    from bayesreg.seg.channels import read_back_probabilities
    started = time.perf_counter()
    out = Path(out_root)/subject
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty; never overwritten")
    labels, affine = nx_seg.read_labels_file(Path(pred_dev)/f"{subject}.nii.gz", image_path)
    again, _ = nx_seg.read_labels_file(Path(raw_dir)/f"{subject}.nii.gz", image_path)
    n_label_files = int((labels != again).sum())                       # a record; the gate is on what is stored
    del again
    with np.load(Path(raw_dir)/f"{subject}.npz", allow_pickle=False) as npz:
        probabilities = npz["probabilities"]
    raw_dtype = str(probabilities.dtype)
    p = nx_seg.orient_probabilities(probabilities, affine)
    del probabilities
    if p.shape[1:] != labels.shape:
        raise GateFailed(f"{subject}: probabilities {p.shape} on labels {labels.shape}")
    own = p.argmax(axis=0)
    n_voxels = int(labels.size)
    n_argmax = int((own != labels).sum())
    nx_seg.check_gate(subject, "the argmax of the raw probabilities", n_argmax, n_voxels)
    p16, n_ties = nx_seg.float16_keeping_argmax(p, own)
    del p
    out.mkdir(parents=True, exist_ok=True)
    try:
        nx_seg.save_npz(out/PROBS_FILE, p16)
        shape = p16.shape
        del p16
        back, read = read_back_probabilities(out/PROBS_FILE, shape=shape)
        stored = back.argmax(axis=0)                                    # ties: the smallest label, as every reader of probs36
        del back
        n_back = int((stored != labels).sum())
        nx_seg.check_gate(subject, f"the argmax of the stored {PROBS_FILE}", n_back, n_voxels)
        if int((stored != own).sum()):
            raise GateFailed(f"{subject}: the read-back argmax differs from the argmax of the probabilities: the conversion is wrong")
    except GateFailed:
        (out/PROBS_FILE).unlink(missing_ok=True)
        out.rmdir()
        raise
    seconds = time.perf_counter()-started
    record = dict(subject=subject, gate="passed", tolerance=nx_seg.GATE_TOLERANCE, n_voxels=n_voxels, n_differing=n_back, fraction_differing=n_back/n_voxels, n_differing_raw_argmax=n_argmax, n_differing_label_file=n_label_files, n_float16_ties_fixed=n_ties, raw_dtype=raw_dtype,
                  raw_max_abs_sum_deviation=read["raw_max_abs_sum_deviation"], shape=list(shape), probs36_sha256=nx_seg.file_sha256(out/PROBS_FILE), image_sha256=nx_seg.file_sha256(image_path), seconds=seconds)
    log(f"{subject}: gate passed, {n_back} of {n_voxels} voxels differ from pred_dev, {n_ties} voxels with one ULP, {seconds:.0f} s")
    return record


def convert(out, subjects, *, nn_out=uc.NN_OUT, data_root=nc.DATA_ROOT, pred_dev=None, log=print):
    """Every subject through convert_subject (a subject done is not done again); then convert_record.json. GateFailed stops the rest (convert_stop.json)."""
    directory = nn_dir(out)
    pred_dev = Path(nn_out)/"pred_dev" if pred_dev is None else Path(pred_dev)
    train = read_nn_training(nn_out)
    record_path = directory/"raw"/"infer_record.json"
    infer_record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
    problems = infer_problems(infer_record, train, subjects)
    if problems:
        raise NNStepFailed(f"{record_path}: {problems}")
    done = directory/"convert"
    done.mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        if (done/f"{subject}.json").is_file():
            continue
        try:
            row = convert_subject(subject, raw_dir=directory/"raw", pred_dev=pred_dev, image_path=Path(data_root)/subject/nx_seg.IMAGE_FILE, out_root=directory/"probs", log=log)
        except GateFailed as failed:
            uc.write_json_atomic(directory/"convert_stop.json", dict(subject=subject, reason=str(failed), raw_kept=str(directory/"raw"), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
            raise
        uc.write_json_atomic(done/f"{subject}.json", row)
    rows = {s: json.loads((done/f"{s}.json").read_text(encoding="utf-8")) for s in subjects}
    document = dict(schema="bayesreg.uf.nn_convert.v1", gate="passed", rule="the argmax of the stored probs36 against nnunet_v1/pred_dev, voxels that differ / all voxels <= the tolerance, per subject (runs/20261105-fx/plan.md revision 1)", tolerance=nx_seg.GATE_TOLERANCE, subjects=list(subjects),
                    n_differing_by_subject={s: rows[s]["n_differing"] for s in subjects}, n_voxels_by_subject={s: rows[s]["n_voxels"] for s in subjects}, n_differing_total=sum(rows[s]["n_differing"] for s in subjects), n_voxels_total=sum(rows[s]["n_voxels"] for s in subjects),
                    probs36_sha256={s: rows[s]["probs36_sha256"] for s in subjects}, image_sha256={s: rows[s]["image_sha256"] for s in subjects}, checkpoint_sha256=train["final_checkpoint_sha256"], trainer=train["trainer"], infer_record_sha256=nx_seg.file_sha256(record_path),
                    pred_dev=str(pred_dev), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    uc.write_json_atomic(directory/"convert_record.json", document)
    return document


def convert_problems(record, subjects):
    """Why convert_record.json is not the 20 subjects through the gate (nx_seg.convert_problems, plus the stored files' hashes), as a list."""
    problems = nx_seg.convert_problems(record or {}, list(subjects))
    if not problems and sorted((record.get("probs36_sha256") or {})) != sorted(subjects):
        problems.append("convert_record.json: no sha256 of a stored probs36 for every subject")
    return problems


def dev_subjects(*, allowlist_path, folds_path, dev_source, synthetic, sealed_repo=None):
    """The subjects of the development source, each checked before anything of them is opened (experiments/unv2/eval_seg.py admit_subjects: the allowlist source, not never_for_training, on no sealed list, no training subject). Ids only."""
    sys.path.insert(0, str(EXPERIMENTS/"unv2"))
    import eval_seg
    from bayesreg.eval.truth import load_sealed_lists
    from bayesreg.io.guard import load_dev_allowlist
    from bayesreg.seg.unet.data import load_folds
    from bayesreg.seg.unet.pins import check_synthetic_subjects, load_pinned_allowlist, source_root_for
    from bayesreg.seg.unet.snapshot import code_record
    if synthetic:
        allowlist = load_dev_allowlist(allowlist_path)
        check_synthetic_subjects([s for ids in allowlist.subjects.values() for s in ids], "the allowlist")
    else:
        allowlist, _ = load_pinned_allowlist(allowlist_path)
    sealed = load_sealed_lists(Path(sealed_repo) if sealed_repo is not None else source_root_for(code_record()))
    return eval_seg.admit_subjects(allowlist, dev_source, sealed, load_folds(folds_path).training_subjects)


def main(argv=None):
    """`infer` on real data only (the chain's unit); `convert` also with --synthetic (the dry run and the tests)."""
    import argparse
    sys.path.insert(0, str(EXPERIMENTS/"unv2"))
    import eval_seg
    from bayesreg.seg.unet.pins import check_synthetic_data_root
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("infer", "convert"))
    parser.add_argument("--out", type=Path, default=uc.OUT)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--allowlist", type=Path, default=eval_seg.ALLOWLIST)
    parser.add_argument("--folds", type=Path, default=eval_seg.FOLDS)
    parser.add_argument("--dev-source", default=eval_seg.DEV_SOURCE)
    parser.add_argument("--data-root", type=Path, default=nc.DATA_ROOT)
    parser.add_argument("--nn-out", type=Path, default=uc.NN_OUT)
    parser.add_argument("--pred-dev", type=Path, default=None, help="default: <nn-out>/pred_dev")
    args = parser.parse_args(argv)
    pred_dev = args.pred_dev or args.nn_out/"pred_dev"
    if args.synthetic:
        if args.step == "infer":
            parser.error("infer runs nnU-Net on real data only")
        check_synthetic_data_root(args.data_root)
    elif (args.allowlist.resolve(), args.folds.resolve(), args.dev_source, args.data_root.resolve(), args.nn_out.resolve(), pred_dev.resolve()) != (eval_seg.ALLOWLIST.resolve(), eval_seg.FOLDS.resolve(), eval_seg.DEV_SOURCE, nc.DATA_ROOT.resolve(), uc.NN_OUT.resolve(), (uc.NN_OUT/"pred_dev").resolve()):
        parser.error("real data: the allowlist, the folds, the development source, the data root and NN's results are the plan's; other values only with --synthetic")
    subjects = dev_subjects(allowlist_path=args.allowlist, folds_path=args.folds, dev_source=args.dev_source, synthetic=args.synthetic)
    if not args.synthetic and len(subjects) != uc.N_SUBJECTS:
        print(f"STOP: {len(subjects)} development subjects, not {uc.N_SUBJECTS}", flush=True)
        return 8
    try:
        if args.step == "infer":
            print(json.dumps(infer(args.out, subjects, nn_out=args.nn_out, data_root=args.data_root), indent=1))
        else:
            document = convert(args.out, subjects, nn_out=args.nn_out, data_root=args.data_root, pred_dev=pred_dev)
            print(json.dumps({k: document[k] for k in ("gate", "tolerance", "n_differing_total", "n_voxels_total")}, indent=1))
    except (NNStepFailed, GateFailed) as failed:
        print(f"STOP: {failed}", flush=True)
        return 8
    return 0


if __name__ == "__main__":
    sys.exit(main())
