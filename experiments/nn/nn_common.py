"""The fixed names of work package NN (runs/20261105-nn/任务书.md, plan.md) and the environment of the nnU-Net commands. Standard library only: this file is imported by the tools that run in sci310 and by their tests, and it never imports nnunetv2.

Two environments, never mixed (runs/20261104-nnunet-env/README.md): the tools of this directory run in sci310 (subject checks, labels, evaluation, the rule); every nnU-Net command runs in the conda environment nnunet, started as a child with the environment of `nn_env`: no PYTHONPATH, no user site, no other conda environment on PATH.

The trainers (read in /home/rmcfz/miniconda3/envs/nnunet/lib/python3.10/site-packages/nnunetv2/training/nnUNetTrainer/variants/training_length/, nnunetv2 2.8.1).
nnUNetTrainer_250epochs and nnUNetTrainer_100epochs exist, but they mirror on all three axes in training (nnUNetTrainer.py, mirror_axes = (0, 1, 2)) without swapping label ids. 32 of the 35 labels of seg35 are Left-* or Right-*, and the axis of left and right is one of the three (the images are axis-aligned: array axis 2 of SimpleITKIO, the first
axis of the NIfTI file), so those trainers would show the same anatomy under both labels and could not tell left from right. So this work package uses the variants without mirroring: nnUNetTrainer_250epochs_NoMirroring exists in 2.8.1; nnUNetTrainer_100epochs_NoMirroring does not, it is the one class of ext_trainers/, found through nnUNet_extTrainer. Those trainers
set inference_allowed_mirroring_axes = None, so inference has no mirror TTA (plan.md, the deviation from the brief; the lead decides).
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MAIN_REPO = Path("/home/rmcfz/code/python_projects/seq_vp_tag")
NN_ENV = Path("/home/rmcfz/miniconda3/envs/nnunet")
NN_BIN = NN_ENV/"bin"
NN_PY = NN_BIN/"python"
PY = "/home/rmcfz/miniconda3/envs/sci310/bin/python"
RESULTS = Path("/home/rmcfz/seq_vp_tag_results")
OUT = RESULTS/"nnunet_v1"
DATA_ROOT = Path("/mnt/d/data")

DATASET_ID = 935
DATASET = "Dataset935_OASIS35"
CONFIGURATION = "3d_fullres"
PLANS = "nnUNetPlans"
FOLD = "all"
CHECKPOINT = "checkpoint_final.pth"
EPOCHS_LONG, EPOCHS_SHORT = 250, 100
TRAINERS = {EPOCHS_LONG: "nnUNetTrainer_250epochs_NoMirroring", EPOCHS_SHORT: "nnUNetTrainer_100epochs_NoMirroring"}
EXT_TRAINERS = HERE/"ext_trainers"

N_PROC = 2                      # the brief: preprocessing -np 2; also nnU-Net's default pool (nnUNet_def_n_proc), which would be 8
N_PROC_DA = 6                   # the lead, ~1:50 am CST 11-05, in place of the brief's <= 4 (docs/bayesreg/reviews/NN_tools_复核_20261105.md, important 1): 16 logical cores, MemoryMax 12G unchanged, the 11 GiB of the smoke's rule still checks it. It changes the throughput, not the recipe
N_PROC_DA_VALIDATION = max(1, N_PROC_DA//2)     # nnUNetTrainer.get_dataloaders starts this many for validation
THREADS = 1                    # nnU-Net's own entry sets 1 when it is run as a script; the console script does not reach that line
CONDA_VARIABLES = ("PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PROMPT_MODIFIER", "CONDA_SHLVL", "CONDA_EXE", "CONDA_PYTHON_EXE", "_CE_M", "_CE_CONDA", "VIRTUAL_ENV")


def raw_root(out=OUT):
    return Path(out)/"nnUNet_raw"


def preprocessed_root(out=OUT):
    return Path(out)/"nnUNet_preprocessed"


def results_root(out=OUT):
    return Path(out)/"nnUNet_results"


def smoke_results_root(out=OUT):
    """The smoke's own root: its checkpoints can never be taken for the training's or the inference's."""
    return Path(out)/"nnUNet_results_smoke"


def trainer_directory(trainer, *, results):
    return Path(results)/DATASET/f"{trainer}__{PLANS}__{CONFIGURATION}"


def fold_directory(trainer, *, results):
    return trainer_directory(trainer, results=results)/f"fold_{FOLD}"


def nn_env(out=OUT, *, results=None, gpu=False, base=None):
    """The environment of an nnU-Net command. gpu=False hides the GPU. No compile (nnUNet_compile=f): torch.compile is nnU-Net's default, the plan does not use it, see plan.md."""
    env = {k: v for k, v in (os.environ if base is None else base).items() if k not in CONDA_VARIABLES}
    path = [p for p in env.get("PATH", "/usr/bin:/bin").split(":") if p and "/miniconda3/" not in p and "/anaconda" not in p]
    env.update(PATH=":".join([str(NN_BIN), *path]), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", nnUNet_raw=str(raw_root(out)), nnUNet_preprocessed=str(preprocessed_root(out)), nnUNet_results=str(results_root(out) if results is None else results), nnUNet_extTrainer=str(EXT_TRAINERS), nnUNet_n_proc_DA=str(N_PROC_DA), nnUNet_def_n_proc=str(N_PROC),
               nnUNet_compile="f", OMP_NUM_THREADS=str(THREADS), OPENBLAS_NUM_THREADS=str(THREADS), MKL_NUM_THREADS=str(THREADS), NUMEXPR_NUM_THREADS=str(THREADS), TORCHINDUCTOR_COMPILE_THREADS="1")
    if gpu:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def plan_argv():
    """Fingerprint, the integrity check, the plan and the preprocessing of 3d_fullres, two processes everywhere (-npfp is the fingerprint's and the integrity check's, -np the preprocessing's: both would default to more). nice 19."""
    return ["nice", "-n", "19", str(NN_BIN/"nnUNetv2_plan_and_preprocess"), "-d", str(DATASET_ID), "-c", CONFIGURATION, "-npfp", str(N_PROC), "-np", str(N_PROC), "--verify_dataset_integrity", "--no_pbar"]


def train_argv(trainer, *, resume=False, device="cuda"):
    """The stock entry. --c: nnU-Net continues from checkpoint_final, else checkpoint_latest (every 50 epochs), else checkpoint_best. device: "cuda" in every real step; sandbox_check.py alone passes "cpu"."""
    return [str(NN_BIN/"nnUNetv2_train"), str(DATASET_ID), CONFIGURATION, FOLD, "-tr", trainer, "-p", PLANS, "-device", device, *(["--c"] if resume else [])]


def infer_argv(*, model_dir, input_dir, out_dir, device="cuda"):
    """infer_nn.py in the environment nnunet; -I: no PYTHONPATH, no user site, no script directory on sys.path."""
    return [str(NN_PY), "-I", "-B", str(HERE/"infer_nn.py"), "--model-dir", str(model_dir), "--input-dir", str(input_dir), "--out-dir", str(out_dir), "--device", device]
