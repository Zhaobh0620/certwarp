"""Inference of work package NN, one subject at a time so that each has its own seconds. Runs in the conda environment nnunet (python -I), and imports nothing of this repository: nn_common.infer_argv starts it, with the environment of nn_common.nn_env.

    python -I -B experiments/nn/infer_nn.py --model-dir DIR --input-dir DIR --out-dir DIR [--device cuda] [--save-probabilities]

--input-dir holds <subject>_0000.nii.gz (images only; run_nn.py makes the links, after the subjects were admitted). The model: fold all, checkpoint_final.pth, nnU-Net's defaults of nnUNetv2_predict (tile step .5, Gaussian window, use_mirroring=True; which axes is the checkpoint's: inference_allowed_mirroring_axes, None for the trainers without mirroring). Written: <out-dir>/<subject>.nii.gz (argmax, the geometry of the input), infer_record.json. Never overwritten.
Seconds per subject: the whole of predict_from_files_sequential for that subject (reading, preprocessing, sliding window, resampling, writing), with the device synchronized around it. --save-probabilities is off: the evaluation needs the argmax only, and 36 classes for 20 subjects are gigabytes; if the registration path needs them, they are made then.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

FOLD, CHECKPOINT, FORMAT = "all", "checkpoint_final.pth", "bayesreg.nn.infer_record.v1"
SUFFIX = "_0000.nii.gz"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True, help="<trainer>__nnUNetPlans__3d_fullres")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-probabilities", action="store_true")
    args = parser.parse_args(argv)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        print(f"STOP: {args.out_dir} exists and is not empty; predictions are never overwritten", flush=True)
        return 8
    images = sorted(p for p in args.input_dir.iterdir() if p.name.endswith(SUFFIX))
    if not images or len(images) != len(list(args.input_dir.iterdir())):
        print(f"STOP: {args.input_dir} must hold <subject>{SUFFIX} files and nothing else", flush=True)
        return 8
    checkpoint = args.model_dir/f"fold_{FOLD}"/CHECKPOINT
    if not checkpoint.is_file():
        print(f"STOP: {checkpoint}: no such file; the training has not finished", flush=True)
        return 8

    from importlib.metadata import version
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    device = torch.device(args.device)
    if device.type == "cuda":                               # as nnUNetv2_predict: more threads do not help on the GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    predictor = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=True, perform_everything_on_device=device.type == "cuda", device=device, verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(args.model_dir), use_folds=(FOLD,), checkpoint_name=CHECKPOINT)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_subject = {}
    for image in images:
        subject = image.name[:-len(SUFFIX)]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        predictor.predict_from_files_sequential([[str(image)]], [str(args.out_dir/subject)], save_probabilities=args.save_probabilities, overwrite=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        per_subject[subject] = time.perf_counter()-started
        print(f"{subject}: {per_subject[subject]:.1f} s", flush=True)
        if not (args.out_dir/f"{subject}.nii.gz").is_file():
            print(f"STOP: {subject}: no prediction was written", flush=True)
            return 8
    values = sorted(per_subject.values())
    axes = predictor.allowed_mirroring_axes
    tta_views = 2**len(axes) if predictor.use_mirroring and axes else 1      # predict_from_raw_data._internal_maybe_mirror_and_predict: the plain view and every combination of the axes
    document = dict(tta_views=tta_views,format=FORMAT, created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), command=sys.argv if argv is None else ["infer_nn.py", *argv], python=platform.python_version(), nnunetv2=version("nnunetv2"), torch=torch.__version__, cuda=torch.version.cuda, device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                    model_dir=str(args.model_dir), trainer=predictor.trainer_name, configuration=args.model_dir.name.split("__")[-1], fold=FOLD, checkpoint=CHECKPOINT, checkpoint_sha256=sha256(checkpoint), tile_step_size=predictor.tile_step_size, use_gaussian=predictor.use_gaussian, use_mirroring=predictor.use_mirroring, allowed_mirroring_axes=predictor.allowed_mirroring_axes,
                    patch_size=predictor.configuration_manager.patch_size, compile=os.environ.get("nnUNet_compile"), saved_probabilities=args.save_probabilities, subjects=list(per_subject), seconds_per_subject=per_subject, seconds_median=values[len(values)//2], seconds_max=values[-1], seconds_first=per_subject[next(iter(per_subject))],
                    gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type == "cuda" else None, gpu_peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30 if device.type == "cuda" else None, test_labels_seen=False, note="只读图像，不读标签。")
    (args.out_dir/"infer_record.json").write_text(json.dumps(document, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({k: document[k] for k in ("trainer", "allowed_mirroring_axes", "tta_views", "seconds_median", "seconds_max", "gpu_peak_allocated_gib")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
