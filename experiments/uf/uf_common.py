"""The fixed names and numbers of work package UF (runs/20261105-uf/任务书.md, plan.md): the recipe v3_large of UL trained to the end of its poly schedule, four members that match unet35_v2_full one to one, then one segmentation-level reading on the development set.
Standard library only: imported by the driver, the evaluation, the reading and their tests.

The members: (fold, seed) of folds_full_v2.json, the names of unet35_v2_full and of runs/20261103-unv2/v2_member_subsets.json; trained in v2's order A0, B0, A1, B1 (one member per fold after two).
The three changes against U1 (runs/20261104-ul): max_iters 40000 (so the poly schedule ends: lr = 5e-4 x (1/39800)^.9, about 3.6e-8, at the last step, it = 39999), no early stop (min_iters = patience = max_iters), and the member takes the weights of the last step (large.py, checkpoint "last"), not the best calib evaluation.
Everything else is the recipe v3_large as large.TrainConfigLarge has it (base 32, patch (128, 160, 112), batch 2, deep supervision, the left-right flip with the labels swapped, class-balanced patches, the Gaussian window, AdamW, the augmentation).
"""
import hashlib
import json
from pathlib import Path

RESULTS = Path("/home/rmcfz/seq_vp_tag_results")
SEG = RESULTS/"seg"
OUT = SEG/"unet35_v4_full"                       # this package's root; unet35_v1, unet35_v2_full, unet35_v3_large, nnunet_v1 are only read
V1_ROOT, V2_ROOT, U1_ROOT = SEG/"unet35_v1", SEG/"unet35_v2_full", SEG/"unet35_v3_large"
V1_PRED, V2_PRED = V1_ROOT/"pred", V2_ROOT/"pred"
NN_OUT = RESULTS/"nnunet_v1"                     # nnU-Net of work package NN: its weights, train_record.json, prepare_manifest.json and pred_dev/, only read
PY = "/home/rmcfz/miniconda3/envs/sci310/bin/python"
MAIN_REPO = Path("/home/rmcfz/code/python_projects/seq_vp_tag")

PROFILE = "v3_large"
MEMBERS = (("A", 0), ("B", 0), ("A", 1), ("B", 1))        # the order of the chain (v2's)
V2_NAME = {f"fold{f}_s{s}": f"{f}{s}" for f, s in MEMBERS}  # foldA_s0 -> A0 of v2_member_subsets.json
MAX_ITERS = 40000                                # the brief and runs/20261104-ul/result.md 读法 3: about 40 000 steps, the poly schedule to its end; U1 measured .226 s per step over 39 000 steps (estimate in the dry run)
EVAL_EVERY = 500
MAX_WALL_SECONDS = 14400.                        # 4 h per member, about 1.6 x the estimate (about 2.5 h); a member stopped by it did not finish the schedule, and the chain stops
TRAIN_CONFIG = dict(max_iters=MAX_ITERS, min_iters=MAX_ITERS, patience=MAX_ITERS, eval_every=EVAL_EVERY, checkpoint="last", max_wall_seconds=MAX_WALL_SECONDS)

N_SUBJECTS = 20
WORTH_BAR = .003                                 # the brief: Delta >= +.003 reads "worth connecting to Hx24"
CONTROL_TOL_STORED = 1e-9                        # v1_control, v2_control: stored probs36 through the same CPU code, nothing may differ (experiments/nn/eval_nn.py CONTROL_TOL; the lead, revision 1 of plan.md, review minor 1)
CONTROL_TOL_FORWARD = 2e-4                       # U1_control: a forward pass on the GPU, AMP and cuDNN may move a few voxels (UL's, experiments/ul/decide_ul.py CONTROL_TOL)
CONTROL_TOLERANCES = dict(v1_control=CONTROL_TOL_STORED, v2_control=CONTROL_TOL_STORED, U1_control=CONTROL_TOL_FORWARD)
CONTROL_METRICS = ("seg35", "seg4")
NN_WEIGHT = .5                                   # arm (b): 1/2 of UF4's mean probability plus 1/2 of nnU-Net's softmax; fixed before any reading, never tuned
FOCUS_SUBJECT = "OASIS_OAS1_0440_MR1"
FIRST5_PAIRS = tuple((f"OASIS_OAS1_{n:04d}_MR1", f"OASIS_OAS1_{n+1:04d}_MR1") for n in range(438, 443))   # the first 5 development pairs (experiments/mc0/mc0_core.PAIRS); a pair is the mean of its two subjects, as the .9172 of OX
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 20000, 20261105

# The references (all read only, pinned by sha256 on the real path)
REF_ENSEMBLES = V2_ROOT/"eval_seg"/"eval_seg.json"               # rows of the v1 and v2 ensembles, from their stored probs36
REF_ENSEMBLES_SHA256 = "698dcc39708313fcec61a663d0def019b443f139d937912b186734a80def84e4"
REF_MEMBERS_REL = "runs/20261103-unv2/v2_member_subsets.json"    # the single members A0 A1 B0 B1 of v2
REF_MEMBERS_SHA256 = "da54e15ac594ea63b0e8966409d76773a896d60a0a254d27c42495cf98a1c6ee"
REF_U1 = U1_ROOT/"eval_u1"/"eval_u1.json"                        # the rows of U1
REF_U1_SHA256 = "8321d8d42d54af9ae02633aaea492f04ac3e06caad24d1ed271f4d878a67c85a"
U1_MEMBER = U1_ROOT/"foldA_s0"
U1_MEMBER_SHA256 = "c7bfcac82c8d83d0beb447918093165130053eb7881dc83ce23081c42f6d43dc"
V2_REGISTRY_REL = "configs/bayesreg/segmenters_v2.json"

# The snapshot of the chain: UL's paths and what the nnU-Net step and the evaluation import (nx_seg imports fast_channels, mc0_core and mc0p_core at its top; nothing of them is called but nx_seg's own functions)
SNAPSHOT_EXTRA = ("experiments/uf", "experiments/nn", "experiments/fl/nx_seg.py", "experiments/fl/nx_infer_child.py", "experiments/ef/fast_channels.py", "experiments/mc0/mc0_core.py", "experiments/mc0/mc0p_core.py")


def member_name(fold, seed):
    return f"fold{fold}_s{seed}"


MEMBER_NAMES = tuple(member_name(f, s) for f, s in MEMBERS)


def planned_config(overrides=None):
    """The recorded config a UF member must have: TrainConfigLarge's defaults (the recipe v3_large, unchanged) with the changes of the plan (TRAIN_CONFIG; the dry run and the tests pass their tiny overrides), as JSON. Imports torch (large.py) when called."""
    from dataclasses import asdict
    from bayesreg.seg.unet.large import TrainConfigLarge
    return json.loads(json.dumps(asdict(TrainConfigLarge(**(TRAIN_CONFIG if overrides is None else overrides)))))


def folds_problems(split, v2):
    """Before any training: the fold of each member in folds_full_v2.json against its v2 namesake's records (train, calib, label contact). Ids only."""
    problems = []
    for fold, seed in MEMBERS:
        name = member_name(fold, seed)
        if split.subjects(fold, "train") != v2[name]["train"] or split.subjects(fold, "calib") != v2[name]["calib"] or split.label_contact(fold) != v2[name]["contact"]:
            problems.append(f"{name}: the folds file gives other subjects than v2's {name} trained on")
    return problems


def write_text_atomic(path, text, *, overwrite=False):
    """Written to <path>.tmp, then moved over (review minor 13): a reader sees the whole file or none. FileExistsError if the file is there, unless overwrite."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; never overwritten")
    tmp = Path(str(path)+".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def write_json_atomic(path, document, *, overwrite=False, **dumps):
    return write_text_atomic(path, json.dumps(document, indent=1, **dumps)+"\n", overwrite=overwrite)


def directory_bytes(path):
    """The bytes of the files under path (symbolic links not followed); 0 when it does not exist."""
    path = Path(path)
    return sum(p.lstat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink()) if path.is_dir() else 0


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def v2_member_lists(v2_root=V2_ROOT, *, registry):
    """{name: fold, seed, train, calib, label contact, member_sha256, train_record_sha256} of the four members of unet35_v2_full, ids only: member.json's member_sha256 listed in the registry under the member's fold, and train_record.json re-hashed against member.json.
    ValueError otherwise: these are the members the plan matches UF's with, one to one."""
    document = json.loads(Path(registry).read_text(encoding="utf-8"))
    registered = {}
    for entry in document["models"].values():
        if entry.get("segmenter_id") == "unet35_v2_full":
            registered.setdefault(entry["fold"], set()).update(entry["member_model_sha256"])
    out = {}
    for fold, seed in MEMBERS:
        name = member_name(fold, seed)
        directory = Path(v2_root)/name
        member = json.loads((directory/"member.json").read_text(encoding="utf-8"))
        if (member.get("fold"), member.get("seed")) != (fold, seed):
            raise ValueError(f"{directory}: member.json is fold {member.get('fold')} seed {member.get('seed')}, not {name}")
        if member["member_sha256"] not in registered.get({"A": 0, "B": 1}[fold], set()):
            raise ValueError(f"{directory}: member {member['member_sha256'][:12]}... is not registered in {registry} under fold {fold}")
        if sha256_file(directory/"train_record.json") != member["train_record_sha256"]:
            raise ValueError(f"{directory}/train_record.json: sha256 differs from member.json")
        record = json.loads((directory/"train_record.json").read_text(encoding="utf-8"))
        if (record.get("fold"), record.get("seed")) != (fold, seed):
            raise ValueError(f"{directory}/train_record.json: not {name}")
        out[name] = dict(fold=fold, seed=seed, train=sorted(record["train_subjects"]), calib=sorted(record["calib_subjects"]), contact=list(record["label_contact_subjects"]), member_sha256=member["member_sha256"], train_record_sha256=member["train_record_sha256"])
    return out


def member_record_problems(train_record, name, v2, *, expected_config, max_iters=MAX_ITERS):
    """Why the train record of UF member `name` is not the plan's (empty: it is): its fold and seed; the recipe v3_large with the config of the plan (the defaults of TrainConfigLarge and the three changes, `expected_config`); early stop off; the poly schedule run to its last step;
    the weights of the last step; the training and calibration subjects and the label contact of its v2 namesake `v2` (v2_member_lists)."""
    fold, seed = name.removeprefix("fold").split("_s")
    seed = int(seed)
    problems = []
    if (train_record.get("fold"), train_record.get("seed")) != (fold, seed):
        problems.append(f"{name}: the record is fold {train_record.get('fold')} seed {train_record.get('seed')}")
    if train_record.get("recipe") != PROFILE:
        problems.append(f"{name}: recipe {train_record.get('recipe')!r}, not {PROFILE}")
    config = train_record.get("config") or {}
    if config != expected_config:
        problems.append(f"{name}: the config differs from the plan's on {sorted(k for k in set(config) | set(expected_config) if config.get(k) != expected_config.get(k))}")
    if not (config.get("min_iters", 0) >= config.get("max_iters", 1) and config.get("patience", 0) >= config.get("max_iters", 1)):
        problems.append(f"{name}: early stop not off (min_iters {config.get('min_iters')}, patience {config.get('patience')}, max_iters {config.get('max_iters')})")
    if train_record.get("stopped") != "max_iters" or train_record.get("iterations_run") != max_iters:
        problems.append(f"{name}: stopped {train_record.get('stopped')!r} after {train_record.get('iterations_run')} steps, not the {max_iters} steps of the schedule")
    if train_record.get("checkpoint_choice") != "last" or train_record.get("checkpoint_file") != "model_last.pt" or train_record.get("checkpoint_iteration") != max_iters:
        problems.append(f"{name}: the member is not the last step (checkpoint {train_record.get('checkpoint_choice')!r}, {train_record.get('checkpoint_iteration')})")
    if sorted(train_record.get("train_subjects") or []) != v2["train"] or sorted(train_record.get("calib_subjects") or []) != v2["calib"]:
        problems.append(f"{name}: the training or calibration subjects are not those of v2's {name}")
    if list(train_record.get("label_contact_subjects") or []) != v2["contact"]:
        problems.append(f"{name}: the label contact is not that of v2's {name}")
    return problems
