"""A check of the tools of work package NN against the real nnU-Net code, on small synthetic volumes in a scratch directory: no real subject, no GPU, no unit. sci310; the nnU-Net commands are children in the environment nnunet, with nnUNet_raw and the others pointing into the scratch directory. nice 19, four CPU cores, MemAvailable >= 4 GiB before it starts.

The complete loop (docs/bayesreg/reviews/NN_tools_复核_20261105.md, important 3), with the real functions of run_nn.py, so that the path of the end of the training is seen before the 10th hour:
    prepare_nn.build_raw (symbolic links, the real dataset.json, the axes and signs of the real images) -> nnUNetv2_plan_and_preprocess with the real arguments -> prepare_nn.link_gt_segmentations -> run_nn.preprocessed_identifiers, prepare_nn.plan_summary, run_nn.network_parameters on what it wrote
    -> both real trainers found by name, their mirror axes (the class's own answer, the real checkpoint is below)
    -> run_nn.train_once with the real run_watched and training_progress: SIGKILL to the process group at the start of the 3rd epoch, a crash (the run's record, the group empty, no process of the sandbox left, checkpoint_latest there, no checkpoint_final; a new training over it refused)
    -> run_nn.train_once with --c, stopped at the first `predicting` line by the real condition (SIGTERM to the group: spawn pool, resource_tracker; the group empty, no process left) -> checkpoint_final.pth loads, its inference_allowed_mirroring_axes is None, checkpoint_latest gone -> merge_train_runs and its fields
    -> infer_nn.py on those weights -> eval_nn.read_nn_prediction, orient_like_load_scalar against load_scalar -> nnUNetv2_train --val (what --final-validation would do) reading gt_segmentations through the links.
What differs from the real commands: -device cpu; a trainer in the scratch directory (subclass of nnUNetTrainer_250epochs_NoMirroring with 4 epochs of 2 + 1 steps, a checkpoint every epoch); nnUNet_n_proc_DA 1. What it cannot show: AMP, the GPU, memory and seconds at real size (the smoke does that). It trains only on synthetic volumes, and the weights mean nothing.

    python -B experiments/nn/sandbox_check.py --work DIR
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(HERE), str(HERE.parent/"unv2")]

import eval_nn  # noqa: E402
import nn_common as nc  # noqa: E402
import prepare_nn  # noqa: E402
import run_nn  # noqa: E402
from bayesreg.seg.unet.synthetic import oasis, synthetic_volume  # noqa: E402
from jd_curl.three_d.nifti import load_scalar  # noqa: E402

SHAPE = (48, 40, 56)                    # native xyz; small, not the real 160, 192, 224
AFFINE = np.array([[-1., 0., 0., 24.], [0., 0., 1., -28.], [0., -1., 0., 20.], [0., 0., 0., 1.]])       # the axes and signs of the real aligned_norm (read from a training subject's header)
TRAIN = [oasis(n) for n in range(921, 927)]
DEV = [oasis(n) for n in (931, 932)]
CORES = {12, 13, 14, 15}                # nnU-Net asks for every core on the CPU (run_training_entry); it gets these
TRAINER, EPOCHS = "nnUNetTrainer_sandbox_NoMirroring", 4
TRAINER_SOURCE = """import torch
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs_NoMirroring import nnUNetTrainer_250epochs_NoMirroring


class nnUNetTrainer_sandbox_NoMirroring(nnUNetTrainer_250epochs_NoMirroring):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 4
        self.num_iterations_per_epoch = 2
        self.num_val_iterations_per_epoch = 1
        self.save_every = 1
"""
CLASS_HELPER = """
import json, sys
import torch
from nnunetv2.run.run_training import get_trainer_from_args
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
out = {}
for name in sys.argv[1:]:
    trainer_class = recursive_find_trainer_class_by_name(name)
    trainer = get_trainer_from_args('935', '3d_fullres', 'all', name, 'nnUNetPlans', False, device=torch.device('cpu'))
    axes = trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()[3]
    out[name] = dict(module=trainer_class.__module__, num_epochs=trainer.num_epochs, mirror_axes=axes, inference_allowed_mirroring_axes=trainer.inference_allowed_mirroring_axes)
print('HELPER ' + json.dumps(out))
"""
CHECKPOINT_HELPER = """
import json, sys
import torch
checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
print('HELPER ' + json.dumps(dict(trainer_name=checkpoint['trainer_name'], current_epoch=checkpoint['current_epoch'], inference_allowed_mirroring_axes=checkpoint['inference_allowed_mirroring_axes'], n_weights=len(checkpoint['network_weights']), configuration=checkpoint['init_args']['configuration'], fold=checkpoint['init_args']['fold'])))
"""


def require(condition, message):
    if not condition:
        raise SystemExit(f"NOT OK: {message}")
    print(f"ok: {message}", flush=True)


def write_subjects(data_root, subjects):
    for subject in subjects:
        image, labels = synthetic_volume(SHAPE, 2000+int(subject.split("_")[2]))
        directory = Path(data_root)/subject
        directory.mkdir(parents=True, exist_ok=True)
        for name, values, dtype in (("aligned_norm.nii.gz", image, np.float32), ("aligned_seg35.nii.gz", labels, np.uint8)):
            volume = nib.Nifti1Image(np.asarray(values, dtype=dtype), AFFINE)
            volume.set_qform(AFFINE, code=1)
            volume.set_sform(AFFINE, code=1)
            volume.header.set_xyzt_units("mm")
            nib.save(volume, directory/name)


def run(argv, *, env, cwd, log_path):
    with Path(log_path).open("a", encoding="utf-8") as stream:
        stream.write("$ "+" ".join(str(a) for a in argv)+"\n")
        stream.flush()
        result = subprocess.run([str(a) for a in argv], env=env, cwd=str(cwd), stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit(f"exit {result.returncode}: {argv}; log {log_path}\n"+Path(log_path).read_text(encoding="utf-8", errors="replace")[-3000:])


def helper(source, *arguments, env, cwd, log_path):
    result = subprocess.run([str(nc.NN_PY), "-I", "-B", "-c", source, *[str(a) for a in arguments]], env=env, cwd=str(cwd), capture_output=True, text=True)
    with Path(log_path).open("a", encoding="utf-8") as stream:
        stream.write(result.stdout+result.stderr)
    lines = [line for line in result.stdout.splitlines() if line.startswith("HELPER ")]
    if result.returncode != 0 or not lines:
        raise SystemExit(f"helper failed: {result.stderr[-3000:]}")
    return json.loads(lines[-1][len("HELPER "):])


def processes_of(results):
    """Every living process, in any group, whose environment points at this sandbox's results root: an orphan that left the group would show here."""
    marker = f"nnUNet_results={results}".encode()
    found = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                if marker in (entry/"environ").read_bytes() and (entry/"stat").read_text().rsplit(")", 1)[1].split()[0] != "Z":
                    found.append(int(entry.name))
            except OSError:
                continue
    return found


def wait_no_processes(results, timeout=15.):
    deadline = time.monotonic()+timeout
    while processes_of(results) and time.monotonic() < deadline:
        time.sleep(.1)
    return processes_of(results)


def last_run_record(out):
    return json.loads(sorted(p for p in (Path(out)/"train_runs").glob("*.json") if not p.name.endswith(".started.json"))[-1].read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args(argv)
    work = args.work.resolve()
    if nc.RESULTS in work.parents or work == nc.OUT:
        raise SystemExit(f"{work}: a scratch directory is required, not the results root")
    os.nice(19-os.nice(0))
    os.sched_setaffinity(0, CORES)
    run_nn.wait_for_memory(4., 60)
    out, data_root, log_path, trainers = work/"out", work/"data", work/"sandbox.log", work/"sandbox_trainers"
    out.mkdir(parents=True)
    trainers.mkdir()
    (trainers/"nn_trainer_sandbox.py").write_text(TRAINER_SOURCE, encoding="utf-8")
    write_subjects(data_root, TRAIN+DEV)
    names = prepare_nn.label_names()

    print("== raw, plan, preprocess")
    print("raw:", json.dumps(prepare_nn.build_raw(nc.raw_root(out), TRAIN, data_root, names)))
    require(all(p.is_symlink() for folder in ("imagesTr", "labelsTr") for p in (nc.raw_root(out)/nc.DATASET/folder).iterdir()), "raw holds links")
    env = nc.nn_env(out, gpu=False)
    require(env.get("PYTHONPATH") is None and "sci310" not in env["PATH"] and env["CUDA_VISIBLE_DEVICES"] == "" and env["nnUNet_n_proc_DA"] == "6", "the environment of the real commands: no sci310, no GPU, 6 augmentation processes")
    run(nc.plan_argv(), env=env, cwd=out, log_path=log_path)
    gt = prepare_nn.gt_directory(nc.preprocessed_root(out))
    require(all(not p.is_symlink() for p in gt.iterdir()), "nnU-Net wrote copies into gt_segmentations")
    linked = prepare_nn.link_gt_segmentations(gt, TRAIN, data_root)
    require(linked["replaced"] == len(TRAIN) and all(p.is_symlink() for p in gt.iterdir()), f"link_gt_segmentations: {linked}")
    require(prepare_nn.link_gt_segmentations(gt, TRAIN, data_root)["replaced"] == 0 and prepare_nn.verify_gt_links(gt, TRAIN, data_root)["n_links"] == len(TRAIN), "a second run changes nothing, verify_gt_links passes")
    require(run_nn.preprocessed_identifiers(out) == sorted(TRAIN), "the preprocessed cases are the training list")
    plans = nc.preprocessed_root(out)/nc.DATASET/f"{nc.PLANS}.json"
    summary = prepare_nn.plan_summary(plans, nc.preprocessed_root(out)/nc.DATASET/"dataset_fingerprint.json")
    summary["parameters"] = run_nn.network_parameters(plans, env=env)
    print("plan (synthetic sizes):", json.dumps({k: summary[k] for k in ("patch_size", "batch_size", "transpose_forward", "n_stages", "features_per_stage", "parameters")}))

    print("\n== the trainers, the class's own answer")
    classes = helper(CLASS_HELPER, nc.TRAINERS[nc.EPOCHS_LONG], nc.TRAINERS[nc.EPOCHS_SHORT], env=env, cwd=out, log_path=log_path)
    print(json.dumps(classes))
    require(classes[nc.TRAINERS[250]]["num_epochs"] == 250 and classes[nc.TRAINERS[100]]["num_epochs"] == 100 and classes[nc.TRAINERS[100]]["module"] == "nn_trainers_v1" and all(c["mirror_axes"] is None for c in classes.values()), "250 from nnU-Net, 100 through nnUNet_extTrainer, no mirror axes")

    print("\n== the loop: a crash")
    results = nc.results_root(out)
    train_env = dict(env, nnUNet_n_proc_DA="1", nnUNet_extTrainer=f"{nc.EXT_TRAINERS}{os.pathsep}{trainers}")
    fold_dir = nc.fold_directory(TRAINER, results=results)
    since = time.time()-run_nn.LOG_MARGIN_SECONDS
    third_epoch =re.compile(r": Epoch 2 *$", re.MULTILINE)     # on_epoch_end of the 2nd epoch has written checkpoint_latest.pth before this line

    def crash():
        return any(third_epoch.search(p.read_text(encoding="utf-8", errors="replace")) for p in fold_dir.glob("training_log_*.txt") if p.stat().st_mtime >= since)

    (out/"prepare_manifest.json").write_text(json.dumps(dict(label_contact_subjects=TRAIN)), encoding="utf-8")
    try:
        run_nn.train_once(out, TRAINER, epochs=EPOCHS, resume=False, final_validation=False, env=train_env, device="cpu", stop_hook=crash, first_signal=signal.SIGKILL, poll=.2)
        raise SystemExit("NOT OK: the crash did not stop the run")
    except run_nn.Stop as exc:
        print(f"Stop: {exc}")
    first = last_run_record(out)
    require(first["outcome"]["returncode"] == -signal.SIGKILL and first["outcome"]["stopped"] == "condition" and first["stopped_before_final_validation"] is False, f"the crash: exit {first['outcome']['returncode']}, epochs logged {first['epoch_seconds']}")
    require(first["outcome"]["group_left"] == [] and run_nn.group_members(first["outcome"]["pid"]) == [] and wait_no_processes(results) == [], "after the SIGKILL: the group is empty, no process of the sandbox left")
    require((fold_dir/"checkpoint_latest.pth").is_file() and not (fold_dir/nc.CHECKPOINT).exists() and not (out/"train_record.json").exists(), "checkpoint_latest there, no checkpoint_final, no train_record")
    try:
        run_nn.check_resume(fold_dir, False)
        raise SystemExit("NOT OK: a new training over the crashed run was not refused")
    except run_nn.Stop as exc:
        print(f"Stop: {str(exc)[-60:]}")

    print("\n== the loop: --c, up to `predicting`")
    time.sleep(1.1)                     # the runs are named by the second
    record = run_nn.train_once(out, TRAINER, epochs=EPOCHS, resume=True, final_validation=False, env=train_env, device="cpu", poll=.2)
    second = last_run_record(out)
    require(second["outcome"]["argv"][-1] == "--c" and second["outcome"]["argv"][second["outcome"]["argv"].index("-device")+1] == "cpu", "the command: the real one with -device cpu and --c")
    require(len(second["epoch_seconds"]) == EPOCHS-2, "this run's record counts its own epochs: the 3rd and the 4th")
    require(second["outcome"]["stopped"] == "condition" and second["stopped_before_final_validation"] is True and any(re.search(r": predicting ", p.read_text(encoding="utf-8", errors="replace")) for p in fold_dir.glob("training_log_*.txt")), f"stopped at `predicting`: exit {second['outcome']['returncode']}, epochs of this run {second['epoch_seconds']}")
    require(second["outcome"]["group_left"] == [] and run_nn.group_members(second["outcome"]["pid"]) == [] and wait_no_processes(results) == [], "after the stop: the group is empty (spawn pool, resource_tracker), no process of the sandbox left")
    require((fold_dir/nc.CHECKPOINT).is_file() and not (fold_dir/"checkpoint_latest.pth").exists() and (fold_dir/"checkpoint_best.pth").is_file(), "checkpoint_final and checkpoint_best there, checkpoint_latest gone")
    print("checkpoints:", {p: (fold_dir/p).stat().st_size for p in run_nn.checkpoints_in(fold_dir)})
    checkpoint = helper(CHECKPOINT_HELPER, fold_dir/nc.CHECKPOINT, env=train_env, cwd=out, log_path=log_path)
    print("checkpoint_final:", json.dumps(checkpoint))
    require(checkpoint["inference_allowed_mirroring_axes"] is None and checkpoint["current_epoch"] == EPOCHS and checkpoint["trainer_name"] == TRAINER and checkpoint["fold"] == nc.FOLD, "the real checkpoint loads: the mirror axes are None after a real on_train_start, all epochs, fold all")
    print("train_record:", json.dumps({k: v for k, v in record.items() if k not in ("final_checkpoint",)}))
    require(record["trainer"] == TRAINER and record["epochs"] == EPOCHS and record["epochs_logged"] == EPOCHS and record["runs_started"] == 2 and record["resumed_runs"] == 1 and record["runs_without_record"] == 0 and record["stopped_before_final_validation"] is True and record["n_label_contact"] == len(TRAIN) and record["wall_seconds"] > 0 and record["final_checkpoint_sha256"] == run_nn.file_sha256(fold_dir/nc.CHECKPOINT), "merge_train_runs: 2 runs, 1 resumed, 4 epochs logged, the sha256 of the file")

    print("\n== infer_nn.py on those weights")
    run_nn.build_dev_input(out/"dev_input", DEV, data_root)
    run(nc.infer_argv(model_dir=nc.trainer_directory(TRAINER, results=results), input_dir=out/"dev_input", out_dir=out/"pred_dev", device="cpu"), env=train_env, cwd=out, log_path=log_path)
    inferred = json.loads((out/"pred_dev"/"infer_record.json").read_text(encoding="utf-8"))
    print("infer record:", json.dumps({k: inferred[k] for k in ("trainer", "configuration", "fold", "checkpoint", "use_mirroring", "allowed_mirroring_axes", "tta_views", "nnunetv2", "torch", "seconds_median", "saved_probabilities", "test_labels_seen")}))
    require(inferred["allowed_mirroring_axes"] is None and inferred["tta_views"] == 1 and inferred["checkpoint_sha256"] == record["final_checkpoint_sha256"] and inferred["configuration"] == nc.CONFIGURATION, "one view; the weights of the training record")
    print("written:", sorted(p.name for p in (out/"pred_dev").iterdir()))
    for subject in DEV:
        top, entry = eval_nn.read_nn_prediction(out/"pred_dev", data_root, subject, lr_pairs=(), min_on_image=0.)
        written = nib.load(out/"pred_dev"/f"{subject}.nii.gz")
        truth = load_scalar(data_root/subject/"aligned_seg35.nii.gz", labels=True)
        same_order = eval_nn.orient_like_load_scalar(np.asarray(nib.load(data_root/subject/"aligned_seg35.nii.gz").dataobj).astype(np.uint8), AFFINE)
        print(subject, "prediction", written.shape, written.header.get_data_dtype(), "qform/sform codes", int(written.header["qform_code"]), int(written.header["sform_code"]), "units", written.header.get_xyzt_units()[0], "foreground on image", round(entry["foreground_on_image"], 3), "labels", entry["n_labels"])
        require(top.shape == truth.values.shape and np.array_equal(same_order, truth.values.astype(np.uint8)) and float(np.abs(written.affine-AFFINE).max()) == 0., f"{subject}: the affine of the input, the order of load_scalar")

    print("\n== nnUNetv2_train --val: what --final-validation would do, through the links of gt_segmentations")
    validation = run_nn.run_watched([*nc.train_argv(TRAINER, device="cpu"), "--val"], env=train_env, log_path=out/"logs"/"val.log", cwd=out, poll=.2)
    summary_file = fold_dir/"validation"/"summary.json"
    require(validation["returncode"] == 0 and summary_file.is_file() and validation["group_left"] == [] and all(p.is_symlink() for p in gt.iterdir()), f"--val finished in {validation['wall_seconds']:.1f} s, summary.json written, the links still links")
    require(wait_no_processes(results) == [], "no process of the sandbox left")
    print("foreground mean Dice of untrained weights (means nothing):", round(json.loads(summary_file.read_text(encoding="utf-8"))["foreground_mean"]["Dice"], 4))
    print(f"\nfull loop ok; peak PSS of the group: crash {first['outcome']['host_peak_pss_gib']:.2f} GiB, resume {second['outcome']['host_peak_pss_gib']:.2f} GiB; {os.fspath(work)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
