"""Command line of the structure-layer segmenter (docs/bayesreg/04_分割器.md §9).

    python -m bayesreg.seg.unet.cli train --fold A --seed 0 [--out ...] [--set key=<JSON>]
    python -m bayesreg.seg.unet.cli calibrate --fold A --seed 0 [--out ...]
    python -m bayesreg.seg.unet.cli check-inputs [--require-snapshot]
    python -m bayesreg.seg.unet.cli check-members [--out ...]
    python -m bayesreg.seg.unet.cli register [--out ...] [--registry configs/bayesreg/segmenters.json]
    python -m bayesreg.seg.unet.cli evaluate-calibration [--out ...]
    python -m bayesreg.seg.unet.cli infer --subjects OASIS_OAS1_0438_MR1 ... [--registry ...] [--skip-machine-gate]
    python -m bayesreg.seg.unet.cli check-tissue4 --report FILE [--n-subjects 3]
    python -m bayesreg.seg.unet.cli --dry-run [--work DIR] [--dry-device cpu|cuda] [--profile v2_full]

--profile selects the segmenter (default v1, everything below is v1). --profile v2_full is unet35_v2_full (04_分割器.md §14):
folds splits/bayesreg_v1/folds_full_v2.json (pinned; re-derived from folds.json), counts 98,8, members in
/home/rmcfz/seq_vp_tag_results/seg/unet35_v2_full, registry configs/bayesreg/segmenters_v2.json, at least one member per
fold to register; `infer` only for the development subjects of the allowlist source stage_L2R_val_v1 (--all-dev: all of
them); evaluate-calibration and check-tissue4 are refused.

--profile v3_large is the larger network of work package UL (large.py, runs/20261104-ul/plan.md, 04_分割器.md §15): the folds and pins of v2_full, members in
/home/rmcfz/seq_vp_tag_results/seg/unet35_v3_large, `train` runs large.train_large_member, `calibrate` and `check-members` as they are; evaluate-calibration, check-tissue4, `register` and `infer`
are refused (one member, not registered; its reading is experiments/ul/run_ul.py). The defaults of v1 and v2_full do not change.

--profile r2_small is the recipe of work package R2 (small.py, runs/20261110-r2/plan.md): v3_large with an emphasis on the target labels of the pp diagnosis (`--set target_labels=[...]`, required),
the folds and pins of v2_full; `train` runs small.train_small_member, `calibrate` and `check-members` as they are; register, infer, evaluate-calibration and check-tissue4 are refused (the members are
registered together with UF4's by experiments/r2/, after its readings).

Real data (default; pins.py): the code must be a verified snapshot (train, calibrate) or a clean git
worktree (the rest); the folds file must have the pinned sha256; the allowlist must be this tree's
configs/bayesreg/dev_subjects.json and its sources must re-hash (against the snapshot's source
repository, or --source-root); the training source is "bayesreg_train_v1" and equals the folds; counts
45,8. `--synthetic` drops the pins for synthetic subjects (OASIS_OAS1_09xx_MR1 and above) and a
non-real data root only.

`infer` without --synthetic first runs `machine_gate` (A4 review item 10): it refuses while a timed run or the
training is active (systemd user units seq-vp-*, tag-d0-*, tag-d0a4-*, tag-unet35-train*) or, on cuda, while the
GPU has a compute process or more than 20% utilization; --skip-machine-gate overrides it (recorded in the output).

Defaults: folds splits/bayesreg_v1/folds.json, allowlist configs/bayesreg/dev_subjects.json, registry
configs/bayesreg/segmenters.json (all relative to the code tree), data /mnt/d/data (read-only),
models and predictions under /home/rmcfz/seq_vp_tag_results/seg/unet35_v1.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from .pins import (DEFAULT_ALLOWLIST, DEFAULT_FOLDS, DEFAULT_FULL_FOLDS, PINNED_FOLDS_SHA256, PINNED_FULL_FOLDS_SHA256,
                   TRAINING_SOURCE)
from .snapshot import CODE_ROOT

REPO = CODE_ROOT
DEFAULT_REGISTRY = REPO/"configs"/"bayesreg"/"segmenters.json"
DEFAULT_GROUPS = REPO/"configs"/"bayesreg"/"structure_groups_v1.json"
DEFAULT_OUT = Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v1")
MAX_TISSUE4_SUBJECTS = 3
V2 = "v2_full"
V2_DEV_SOURCE = "stage_L2R_val_v1"              # the only subjects unet35_v2_full segments (0438-0457)
V3 = "v3_large"
V5 = "r2_small"                                 # work package R2 (small.py); V4 would read as the segmenter unet35_v4_ens, which is v3_large trained by UF


@dataclass(frozen=True)
class Profile:
    """The defaults of one segmenter; an option given on the command line wins."""
    folds: Path
    expect_counts: str
    out: Path
    registry: Path
    segmenter_id: str
    prefix: str


PROFILES = {
    "v1": Profile(DEFAULT_FOLDS, "45,8", DEFAULT_OUT, DEFAULT_REGISTRY, "unet35_v1_ens", "unet35_v1"),
    V2: Profile(DEFAULT_FULL_FOLDS, "98,8", Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v2_full"),
                REPO/"configs"/"bayesreg"/"segmenters_v2.json", "unet35_v2_full", "unet35_v2_full"),
    V3: Profile(DEFAULT_FULL_FOLDS, "98,8", Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v3_large"),
                REPO/"configs"/"bayesreg"/"segmenters_v3.json", "unet35_v3_large", "unet35_v3_large"),
    V5: Profile(DEFAULT_FULL_FOLDS, "98,8", Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_r2_full"),
                REPO/"configs"/"bayesreg"/"segmenters_r2_v1.json", "unet35_r2_members", "unet35_r2_members"),     # register refused: the ids are never written by cli
}
# infer machine gate (A4 review item 10): timed runs and GPU jobs that a real inference must not overlap
TIMED_UNIT_PATTERNS = ("seq-vp-*", "tag-d0-*", "tag-d0a4-*", "tag-unet35-train*")
GPU_UTIL_MAX = 20
SMI = Path("/usr/lib/wsl/lib/nvidia-smi")


def _log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def _counts(text):
    if text.lower() == "none":
        return None
    n_train, n_calib = (int(v) for v in text.split(","))
    return n_train, n_calib


def _full(args):
    return getattr(args, "profile", "v1") == V2


def _on_full_folds(args):
    """The profiles on folds_full_v2.json and its pins; everything else that depends on _full stays v2_full's."""
    return getattr(args, "profile", "v1") in (V2, V3, V5)


def _refuse_for_v3(args, why, why_r2):
    """The commands v3_large and r2_small do not have; the reason is the profile's own (R2 review m9)."""
    profile = getattr(args, "profile", "v1")
    if profile in (V3, V5):
        raise SystemExit(f"{args.command}: not for --profile {profile}: {why if profile == V3 else why_r2}")


def _apply_profile(args):
    """Fill the options left unset with the defaults of --profile (v1 when not given)."""
    profile = PROFILES[getattr(args, "profile", "v1")]
    for name in ("folds", "expect_counts", "out", "registry", "segmenter_id", "prefix"):
        if hasattr(args, name) and getattr(args, name) is None:
            setattr(args, name, getattr(profile, name))
    return args


def _inputs(args):
    """(split, allowlist, inputs record): pinned for real data, synthetic-only checks with --synthetic."""
    from .data import load_full_folds
    from .pins import load_pinned_full_inputs, load_pinned_inputs, load_synthetic_inputs
    counts = _counts(args.expect_counts)
    if args.synthetic:
        return load_synthetic_inputs(args.folds, args.allowlist, data_root=args.data_root,
                                     training_source=args.training_source, expect_counts=counts,
                                     **(dict(loader=load_full_folds) if _on_full_folds(args) else {}))
    load = load_pinned_full_inputs if _on_full_folds(args) else load_pinned_inputs
    return load(args.folds, args.allowlist, training_source=args.training_source, expect_counts=counts,
                source_root=args.source_root)


def _guard(args):
    from .data import LabelGuard
    split, allowlist, record = _inputs(args)
    return LabelGuard(split, allowlist, training_source=args.training_source), record


def _require_snapshot(args):
    if not args.synthetic:
        from .snapshot import require_snapshot
        require_snapshot()


def _pinned_sha(args):
    if args.synthetic:
        return None
    return PINNED_FULL_FOLDS_SHA256 if _on_full_folds(args) else PINNED_FOLDS_SHA256


def _refuse_for_v2(args, why):
    if _on_full_folds(args):
        raise SystemExit(f"{args.command}: not for --profile {args.profile}: {why}")


def _check_segmenter_choice(args):
    """A profile uses its own segmenter and never the other's repository registry (UNV2 review, minor 1 and 2): the v2 rules of
    `infer` cannot be left by naming v2 under --profile v1, and v2 is never appended to the v1 registry the A5P queue checks."""
    v2 = PROFILES[V2]
    if _full(args) != (args.segmenter_id == v2.segmenter_id):
        raise SystemExit(f"{args.command}: segmenter {args.segmenter_id!r} under --profile {args.profile}: "
                         f"{v2.segmenter_id} goes with --profile {V2} and only with it")
    other = DEFAULT_REGISTRY if _full(args) else v2.registry
    if Path(args.registry).resolve() == other.resolve():
        raise SystemExit(f"{args.command}: registry {args.registry} belongs to the other profile than {args.profile}")


def _config(args):
    config = {}
    if args.config:
        config.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    for item in args.set or []:
        key, _, value = item.partition("=")
        config[key] = json.loads(value)
    return config


def cmd_train(args):
    from .data import OasisFiles
    from .train import train_member
    _require_snapshot(args)
    guard, record = _guard(args)
    trainer, config = train_member, _config(args)
    if args.profile in (V3, V5):
        if args.profile == V5:
            from .small import train_small_member as trainer
        else:
            from .large import train_large_member as trainer
        if args.synthetic:
            config.setdefault("flip_axis_check", False)
        elif "flip_axis_check" in config:
            raise SystemExit(f"train --profile {args.profile}: flip_axis_check cannot be set on real data, whatever its value; it is off for synthetic subjects only")
    result = trainer(files=OasisFiles(Path(args.data_root)), guard=guard, fold=args.fold, seed=args.seed,
                     models_root=args.out, config=config, device=args.device, log=_log, command=sys.argv,
                     inputs_record=record)
    return dict(best_calib_mean_dice35=result["best_calib_mean_dice35"], stopped=result["stopped"])


def cmd_calibrate(args):
    from .calibrate import calibrate_member
    from .data import OasisFiles
    _require_snapshot(args)
    guard, record = _guard(args)
    member, calibration = calibrate_member(files=OasisFiles(Path(args.data_root)), guard=guard, fold=args.fold,
                                           seed=args.seed, models_root=args.out, device=args.device,
                                           max_voxels=args.max_voxels, log=_log, command=sys.argv,
                                           inputs_record=record)
    return dict(temperature=member["temperature"], member_sha256=member["member_sha256"], fit=calibration["fit"])


def cmd_check_inputs(args):
    """The pinned-input checks and the label guard's construction, without reading any label or image."""
    if args.require_snapshot:
        _require_snapshot(args)
    guard, record = _guard(args)
    code = record.get("code") or {}
    return dict(mode=record["mode"], folds_sha256=record["folds_sha256"], counts=record["counts"],
                allowlist_sha256=record["allowlist_sha256"], training_source=record["training_source"],
                allowlist_sources_verified=sorted(record.get("allowlist_sources_verified", {})),
                code=dict(kind=code.get("kind"), commit=code.get("commit"), root=code.get("root")),
                guard=dict(n_training_subjects=len(guard.split.training_subjects), label_reads=len(guard.reads)))


def cmd_check_members(args):
    from .registry import verify_members
    from .train import write_json
    split, allowlist, record = _inputs(args)
    _, check = verify_members(args.out, split=split, allowlist=allowlist, training_source=args.training_source,
                              pinned_folds_sha256=_pinned_sha(args))
    check.update(inputs=record, command=sys.argv, checked_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    report = Path(args.report) if args.report else Path(args.out)/f"member_check_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    if report.exists():
        raise FileExistsError(f"{report} exists; never overwritten")
    write_json(report, check)
    return dict(passed=check["passed"], report=str(report), members=[(m["fold"], m["seed"]) for m in check["members"]])


def cmd_register(args):
    from .registry import register
    _refuse_for_v3(args, "U1 is one member and is not registered; the lead registers a later ensemble (runs/20261104-ul/plan.md)",
                   "R2's members are registered together with UF4's by experiments/r2/run_r2f.py register, after the segmentation-level gate and the review (runs/20261110-r2/plan.md)")
    _check_segmenter_choice(args)
    split, allowlist, _ = _inputs(args)
    seeds = tuple(int(v) for v in args.seeds.split(","))
    registry = register(args.out, args.registry, split=split, allowlist=allowlist, training_source=args.training_source,
                        pinned_folds_sha256=_pinned_sha(args), segmenter_id=args.segmenter_id, prefix=args.prefix,
                        seeds=seeds, log=_log, **(dict(full_training_set=True) if _full(args) else {}))
    return dict(registry=registry.path, sha256=registry.sha256, models=sorted(registry.models))


def cmd_evaluate_calibration(args):
    from .crosscal import evaluate_cross_fold_calibration
    from .data import OasisFiles
    _refuse_for_v2(args, "every member saw the labels of the other fold's calib subjects")
    guard, record = _guard(args)
    document = evaluate_cross_fold_calibration(
        files=OasisFiles(Path(args.data_root)), guard=guard, models_root=args.out, split=guard.split,
        allowlist=guard.allowlist, training_source=args.training_source, device=args.device,
        pinned_folds_sha256=_pinned_sha(args), max_voxels=args.max_voxels, out_path=args.report, log=_log,
        command=sys.argv, inputs_record=record)
    return {fold: dict(ensemble=r["ensemble"], members=[(m["nll"], m["ece"]) for m in r["members"]],
                       ensemble_temperature=r["ensemble_temperature_information_only"]["temperature"])
            for fold, r in document["results"].items()}


class MachineBusy(SystemExit):
    """The machine gate of `infer` refused (a timed run or GPU job in progress); nothing was read."""


def machine_gate(device, *, run=subprocess.run, smi=SMI):
    """The gate of a real `infer` (A4 review item 10; the inference uses the GPU and must not overlap a timed run):
    no active or activating systemd user unit matching TIMED_UNIT_PATTERNS (paper-1 chains seq-vp-*, the D0 and A4'
    queues and their runs tag-d0-* / tag-d0a4-*, the segmenter training tag-unet35-train*), and for a cuda device no
    GPU compute process and utilization <= GPU_UTIL_MAX %. A query that fails refuses too. Returns the record;
    MachineBusy otherwise. `run` and `smi` are replaceable for tests."""
    record = dict(check="infer machine gate (A4 review item 10)", unit_patterns=list(TIMED_UNIT_PATTERNS),
                  device=str(device), checked_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    try:
        units = run(["systemctl", "--user", "list-units", "--state=active,activating", "--no-legend", "--plain",
                     *TIMED_UNIT_PATTERNS], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise MachineBusy(f"infer: cannot list systemd user units ({type(exc).__name__}: {exc}); refused "
                          "(--skip-machine-gate to override)") from exc
    if units.returncode != 0:
        raise MachineBusy(f"infer: systemctl --user list-units failed ({units.stderr.strip()}); refused "
                          "(--skip-machine-gate to override)")
    active = sorted(line.split()[0] for line in units.stdout.splitlines() if line.strip())
    record["active_units"] = active
    if active:
        raise MachineBusy(f"infer: timed runs or training in progress ({', '.join(active)}); not started (the inference "
                          "uses the GPU and must not overlap a timed run)")
    if str(device).startswith("cuda"):
        try:
            apps = run([str(smi), "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True,
                       timeout=60)
            util = run([str(smi), "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True,
                       text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as exc:
            raise MachineBusy(f"infer: nvidia-smi failed ({type(exc).__name__}: {exc}); refused") from exc
        pids = [p.strip() for p in apps.stdout.splitlines() if p.strip().isdigit()]
        reading = (util.stdout.strip().splitlines() or [""])[0].strip()
        if apps.returncode != 0 or util.returncode != 0 or not reading.isdigit():
            raise MachineBusy(f"infer: nvidia-smi gave no reading (compute apps exit {apps.returncode}, utilization "
                              f"{reading!r}); refused (--skip-machine-gate to override)")
        record.update(gpu_compute_pids=pids, gpu_utilization_percent=int(reading), gpu_util_max=GPU_UTIL_MAX)
        if pids or int(reading) > GPU_UTIL_MAX:
            raise MachineBusy(f"infer: GPU busy (compute pids {pids or 'none'}, utilization {reading}% > "
                              f"{GPU_UTIL_MAX}% or another process); not started")
    record["passed"] = True
    return record


def v2_infer_subjects(subjects, allowlist, *, all_dev=False, source=V2_DEV_SOURCE):
    """The subjects unet35_v2_full may segment: only those of the allowlist source `source` (the 20 development subjects
    0438-0457; never T0 or anything else on the allowlist). all_dev: every subject of the source. SystemExit otherwise."""
    sources = {s["name"]: s for s in allowlist.sources}
    if source not in sources or sources[source]["never_for_training"]:
        raise SystemExit(f"infer --profile {V2}: the allowlist has no development source {source!r}")
    listed = list(sources[source]["subjects"])
    if all_dev:
        subjects = list(subjects)+[s for s in listed if s not in subjects]
    outside = sorted(set(subjects)-set(listed))
    if outside:
        raise SystemExit(f"infer --profile {V2}: {outside} not in the development source {source!r}; not segmented")
    if not subjects:
        raise SystemExit("infer: no subjects given")
    return subjects


def cmd_infer(args):
    from ...io.guard import load_dev_allowlist
    from ...io.pair import load_segmenter_registry
    from .data import OasisFiles
    from .groups import load_grouping
    from .infer import segment_subject
    from .pins import check_synthetic_data_root, check_synthetic_subjects, load_pinned_allowlist
    _refuse_for_v3(args, "U1 is evaluated by experiments/ul/run_ul.py, on the development subjects only (runs/20261104-ul/plan.md)",
                   "R2's members are evaluated by experiments/r2/eval_r2.py and its candidate is inferred by experiments/r2/run_r2f.py (runs/20261110-r2/plan.md)")
    subjects = list(args.subjects or [])
    if args.subjects_file:
        subjects += [s.strip() for s in Path(args.subjects_file).read_text().split() if s.strip()]
    if args.all_dev and not _full(args):
        raise SystemExit(f"infer: --all-dev goes with --profile {V2}")
    if not subjects and not args.all_dev:
        raise SystemExit("infer: no subjects given")
    if args.dev_source != V2_DEV_SOURCE and not args.synthetic:
        raise SystemExit(f"infer: --dev-source is {V2_DEV_SOURCE} for real data, got {args.dev_source!r}")
    _check_segmenter_choice(args)
    if args.synthetic or args.skip_machine_gate:
        gate = dict(check="infer machine gate (A4 review item 10)", skipped=True,
                    reason="--synthetic" if args.synthetic else "--skip-machine-gate")
        _log(f"machine gate skipped ({gate['reason']})")
    else:
        gate = machine_gate(args.device)                     # before any image, label or model is read
        _log("machine gate passed: no timed run or training unit active" +
             ("; GPU idle" if str(args.device).startswith("cuda") else ""))
    if args.gate_only:                                       # the chain's probe before its step: nothing is read, no marker left
        return dict(machine_gate=gate)
    if args.synthetic:
        check_synthetic_data_root(args.data_root)
        allowlist = load_dev_allowlist(args.allowlist)
        check_synthetic_subjects([s for ids in allowlist.subjects.values() for s in ids], "the allowlist")
    else:
        allowlist, _ = load_pinned_allowlist(args.allowlist, source_root=args.source_root)
    if _full(args):
        subjects = v2_infer_subjects(subjects, allowlist, all_dev=args.all_dev, source=args.dev_source)
    registry = load_segmenter_registry(args.registry)
    if not _full(args) and any(e["segmenter_id"] == PROFILES[V2].segmenter_id for e in registry.models.values()):
        raise SystemExit(f"infer: {args.registry} registers {PROFILES[V2].segmenter_id}: --profile {V2} is required")
    grouping = load_grouping(args.groups, args.grouping)
    pred_root = Path(args.pred_out) if args.pred_out else Path(args.out)/"pred"
    done = {}
    for subject in subjects:
        provenance = segment_subject(subject, files=OasisFiles(Path(args.data_root)), registry=registry,
                                     allowlist=allowlist, models_root=args.out, grouping=grouping, pred_root=pred_root,
                                     device=args.device, segmenter_id=args.segmenter_id, command=sys.argv, log=_log,
                                     allow_unverified_grouping=args.allow_unverified_grouping)
        done[subject] = dict(policy=provenance["policy"], model_ids=provenance["model_ids"])
    return dict(subjects=done, machine_gate=gate)


# ---- check-tissue4 ---------------------------------------------------------------------------------

def label_adjacency(labels, n_classes=36):
    """Symmetric (n, n) counts of face-adjacent voxel pairs (6-neighbourhood) with labels a != b."""
    labels = np.asarray(labels, dtype=np.int64)
    table = np.zeros((n_classes, n_classes), dtype=np.int64)
    for axis in range(labels.ndim):
        moved = np.moveaxis(labels, axis, 0)
        a, b = moved[:-1].reshape(-1), moved[1:].reshape(-1)
        differ = a != b
        table += np.bincount(a[differ]*n_classes+b[differ], minlength=n_classes*n_classes).reshape(n_classes, n_classes)
    return table+table.T


# "Not adjacent" claims of structure13_v1 (04_分割器.md §2) checked on the same reads.
ADJACENCY_CLAIMS = {
    "accumbens_vs_ventral_dc": [(16, 17), (32, 33), (16, 33), (32, 17)],
    "vessel_vs_choroid_plexus": [(18, 19), (34, 35), (18, 35), (34, 19)],
    "inf_lat_ventricle_vs_3rd_4th_ventricle": [(4, 11), (4, 12), (23, 11), (23, 12), (11, 12)],
}


def cmd_check_tissue4(args):
    """seg35 x seg4 cross-table and seg35 adjacency on at most 3 training subjects (labels through the guard)."""
    from .data import OasisFiles, read_labels
    from .groups import load_grouping
    from .train import write_json
    _refuse_for_v2(args, "the check of the v1 folds; nothing of it depends on the segmenter")
    if not 1 <= args.n_subjects <= MAX_TISSUE4_SUBJECTS:
        raise SystemExit(f"check-tissue4: --n-subjects must be 1..{MAX_TISSUE4_SUBJECTS}")
    report_path = Path(args.report)
    if report_path.exists():
        raise FileExistsError(f"{report_path} exists; never overwritten")
    guard, record = _guard(args)
    reference = load_grouping(args.groups, "tissue4_ref")
    default = load_grouping(args.groups)
    files = OasisFiles(Path(args.data_root))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    subjects = guard.split.subjects("A", "train")[:args.n_subjects]
    table = np.zeros((36, 5), dtype=np.int64)
    adjacency = np.zeros((36, 36), dtype=np.int64)
    for subject in subjects:
        seg35, _ = read_labels(files, subject, guard, kind="seg35")
        seg4, _ = read_labels(files, subject, guard, kind="seg4")
        pairs = seg35.reshape(-1).astype(np.int64)*5+seg4.reshape(-1).astype(np.int64)
        table += np.bincount(pairs, minlength=36*5).reshape(36, 5)
        adjacency += label_adjacency(seg35)
    expected = reference.class_to_channel()
    observed = {}
    for label in range(36):
        if table[label].sum():
            observed[label] = dict(majority_seg4=int(table[label].argmax()),
                                   fraction=float(table[label].max()/table[label].sum()),
                                   expected_seg4=int(expected[label]), voxels=int(table[label].sum()),
                                   seg4_counts=table[label].tolist())
    mismatched = sorted(l for l, o in observed.items() if o["majority_seg4"] != o["expected_seg4"])
    within = {}
    for name, labels in zip(default.group_names, default.group_labels):
        pairs = {f"{a}-{b}": int(adjacency[a, b]) for i, a in enumerate(labels) for b in labels[i+1:] if adjacency[a, b]}
        if pairs:
            within[name] = pairs
    claims = {name: dict(pairs={f"{a}-{b}": int(adjacency[a, b]) for a, b in pairs},
                         contacts=int(sum(adjacency[a, b] for a, b in pairs)))
              for name, pairs in ADJACENCY_CLAIMS.items()}
    report = dict(format="bayesreg.unet35.check_tissue4.v2", subjects=subjects, observed=observed,
                  mismatched_labels=mismatched, verified=not mismatched and len(observed) == 36,
                  note="verified refers to tissue4_ref only; updating its status in structure_groups_v1.json is "
                       "a separate, reviewed change",
                  adjacency=dict(neighbourhood="6 (face)", grouping=default.name, within_group_contacts=within,
                                 claims=claims),
                  guard=guard.record(), inputs=record, command=sys.argv)
    write_json(report_path, report)
    return dict(mismatched_labels=mismatched, verified=report["verified"],
                claims={k: v["contacts"] for k, v in claims.items()}, report=str(report_path))


# ---- dry run --------------------------------------------------------------------------------------

def dry_run(work=None, device="cpu", shape=(32, 32, 32), config=None, log=_log):
    """train -> calibrate -> check -> register -> cross-fold calibration -> infer -> provenance validation
    on synthetic NIfTI subjects (no pins; synthetic ids only)."""
    from ...io.guard import SubjectSealed, load_dev_allowlist
    from ...io.pair import ProvenanceRejected, load_pair, load_segmenter_registry, validate_segmentation_provenance
    from .calibrate import calibrate_member
    from .crosscal import evaluate_cross_fold_calibration
    from .data import LabelAccessRefused, LabelGuard, OasisFiles, load_folds, read_labels
    from .groups import GroupingUnverified, load_grouping
    from .infer import read_probabilities, segment_subject
    from .registry import register
    from .synthetic import allowlist_document, folds_document, oasis, write_json, write_pair, write_subjects
    from .train import train_member

    started = time.perf_counter()
    work = Path(work) if work else Path(tempfile.mkdtemp(prefix="unet35_dryrun_"))
    work.mkdir(parents=True, exist_ok=True)
    a_train, a_calib, b_train, b_calib = [oasis(901), oasis(902)], [oasis(903)], [oasis(904), oasis(905)], [oasis(906)]
    dev, t0, sealed, holdout = [oasis(907), oasis(908)], [oasis(909)], oasis(910), [oasis(911)]
    training = a_train+a_calib+b_train+b_calib
    source = "synthetic_bline_train"
    grouping = load_grouping(DEFAULT_GROUPS)
    write_subjects(work/"data", training+dev+t0+[sealed]+holdout, shape, tissue4=load_grouping(DEFAULT_GROUPS,
                                                                                                 "tissue4_ref"))
    folds_path = write_json(work/"folds.json", folds_document(a_train, a_calib, b_train, b_calib, holdout))
    allow_path = write_json(work/"dev_subjects.json", allowlist_document(
        [("synthetic_dev", dev, False), ("synthetic_t0", t0, True), (source, training, False)]))
    allowlist = load_dev_allowlist(allow_path)
    split = load_folds(folds_path)
    guard = LabelGuard(split, allowlist, training_source=source)
    files = OasisFiles(work/"data")
    models_root = work/"models"
    cfg = dict(base_channels=16, depth=4, max_iters=8, eval_every=4, warmup_iters=2, min_iters=8, log_every=2,
               amp=device.startswith("cuda"))
    cfg.update(config or {})
    summary = dict(work=str(work), device=device, shape_xyz=list(shape), config=cfg, members={})
    for fold in ("A", "B"):
        for seed in (0, 1):
            record = train_member(files=files, guard=guard, fold=fold, seed=seed, models_root=models_root, config=cfg,
                                  device=device, log=log, command=["dry-run"])
            member, calibration = calibrate_member(files=files, guard=guard, fold=fold, seed=seed,
                                                   models_root=models_root, device=device, log=log,
                                                   command=["dry-run"])
            summary["members"][f"{fold}{seed}"] = dict(best_calib_mean_dice35=record["best_calib_mean_dice35"],
                                                       temperature=member["temperature"],
                                                       nll=[calibration["fit"]["nll_before"],
                                                            calibration["fit"]["nll_after"]])
    registry_path = work/"segmenters.json"
    register(models_root, registry_path, split=split, allowlist=allowlist, training_source=source, log=log)
    registry = load_segmenter_registry(registry_path)
    crossfit = evaluate_cross_fold_calibration(files=files, guard=LabelGuard(split, allowlist, training_source=source),
                                               models_root=models_root, split=split, allowlist=allowlist,
                                               training_source=source, device=device, log=log, command=["dry-run"])
    summary["crossfit"] = {f: dict(ensemble=r["ensemble"], models_fold=r["models_fold"])
                           for f, r in crossfit["results"].items()}
    pred_root = models_root/"pred"
    inferred = {}
    for subject in [dev[0], t0[0], a_train[0], a_train[1], b_train[0]]:
        prov = segment_subject(subject, files=files, registry=registry, allowlist=allowlist, models_root=models_root,
                               grouping=grouping, pred_root=pred_root, device=device, command=["dry-run"], log=log)
        groups, _ = read_probabilities(pred_root/subject)
        inferred[subject] = dict(policy=prov["policy"], model_ids=prov["model_ids"],
                                 max_abs_sum_minus_1=float(np.max(np.abs(groups.sum(axis=0)-1))))
    summary["inferred"] = inferred
    refusals = {}
    try:
        read_labels(files, dev[0], guard)
        refusals["label_read_dev"] = "NOT REFUSED"
    except LabelAccessRefused as exc:
        refusals["label_read_dev"] = f"refused: {exc}"
    try:
        segment_subject(sealed, files=files, registry=registry, allowlist=allowlist, models_root=models_root,
                        grouping=grouping, pred_root=pred_root, device=device, log=log)
        refusals["infer_sealed"] = "NOT REFUSED"
    except SubjectSealed as exc:
        refusals["infer_sealed"] = f"refused: {exc}"
    try:
        segment_subject(dev[1], files=files, registry=registry, allowlist=allowlist, models_root=models_root,
                        grouping=load_grouping(DEFAULT_GROUPS, "tissue4_ref"), pred_root=work/"pred_tissue4",
                        device=device, log=log)
        refusals["infer_unverified_grouping"] = "NOT REFUSED"
    except GroupingUnverified as exc:
        refusals["infer_unverified_grouping"] = f"refused: {exc}"
    checks = {}
    for name, (fixed, moving) in dict(dev_t0=(dev[0], t0[0]), same_fold_A=(a_train[0], a_train[1]),
                                       cross_fold=(a_train[0], b_train[0])).items():
        pair = load_pair(write_pair(work/"pairs"/name, work/"data", fixed, moving), allowlist=allowlist)
        try:
            for side, subject in (("fixed", fixed), ("moving", moving)):
                provenance = json.loads((pred_root/subject/"provenance.json").read_text())
                validate_segmentation_provenance(provenance, pair, side=side, layer="structure", registry=registry,
                                                 allowlist=allowlist)
            checks[name] = "accepted"
        except ProvenanceRejected as exc:
            checks[name] = f"provenance_rejected: {exc}"
    crossfit_reads = sorted(r["subject"] for r in crossfit["label_guard"]["label_reads"])
    summary.update(refusals=refusals, pair_checks=checks, crossfit_label_reads=crossfit_reads,
                   wall_seconds=time.perf_counter()-started)
    ok = (all(v.startswith("refused") for v in refusals.values()) and checks["dev_t0"] == "accepted"
          and checks["same_fold_A"] == "accepted" and checks["cross_fold"].startswith("provenance_rejected")
          and inferred[dev[0]]["policy"] == "all_folds_mean"
          and inferred[a_train[0]]["model_ids"] == ["unet35_v1_foldB"]
          and inferred[b_train[0]]["model_ids"] == ["unet35_v1_foldA"]
          and all(v["max_abs_sum_minus_1"] < 1e-12 for v in inferred.values())
          and crossfit_reads == sorted(a_calib+b_calib)
          and crossfit["results"]["A"]["models_fold"] == "B" and crossfit["results"]["B"]["models_fold"] == "A")
    summary["ok"] = bool(ok)
    write_json(work/"dry_run_summary.json", summary)
    return summary


def dry_run_v2(work=None, device="cpu", shape=(32, 32, 32), config=None, log=_log,
               members=(("A", 0), ("B", 0), ("A", 1))):
    """unet35_v2_full on synthetic NIfTI subjects (no pins; synthetic ids only): the folds derived from a synthetic
    folds.json; `members` in that order (default the three of a chain stopped by its deadline), train -> calibrate;
    registration; infer; provenance validation of a development pair; the refusals."""
    from ...io.guard import SubjectSealed, load_dev_allowlist
    from ...io.pair import ProvenanceRejected, load_pair, load_segmenter_registry, validate_segmentation_provenance
    from .calibrate import calibrate_member
    from .data import (LabelAccessRefused, LabelGuard, OasisFiles, full_folds_document, load_full_folds, parse_folds,
                       read_labels, render_full_folds)
    from .groups import load_grouping
    from .infer import ModelSelectionRefused, read_probabilities, segment_subject
    from .registry import MODEL_PREFIX_V2, SEGMENTER_ID_V2, register
    from .synthetic import allowlist_document, folds_document, oasis, write_json, write_pair, write_subjects
    from .train import train_member

    started = time.perf_counter()
    work = Path(work) if work else Path(tempfile.mkdtemp(prefix="unet35_v2_dryrun_"))
    work.mkdir(parents=True, exist_ok=True)
    a_train, a_calib, b_train, b_calib = [oasis(901), oasis(902)], [oasis(903)], [oasis(904), oasis(905)], [oasis(906)]
    dev, t0, sealed = [oasis(907), oasis(908)], [oasis(909)], oasis(910)
    training = a_train+a_calib+b_train+b_calib
    source = "synthetic_bline_train"
    grouping = load_grouping(DEFAULT_GROUPS)
    write_subjects(work/"data", training+dev+t0+[sealed], shape, tissue4=load_grouping(DEFAULT_GROUPS, "tissue4_ref"))
    v1_split = parse_folds(folds_document(a_train, a_calib, b_train, b_calib))
    folds_path = work/"folds_full_v2.json"
    folds_path.write_bytes(render_full_folds(full_folds_document(v1_split)))
    allow_path = write_json(work/"dev_subjects.json", allowlist_document(
        [("synthetic_dev", dev, False), ("synthetic_t0", t0, True), (source, training, False)]))
    allowlist = load_dev_allowlist(allow_path)
    split = load_full_folds(folds_path)
    guard = LabelGuard(split, allowlist, training_source=source)
    files = OasisFiles(work/"data")
    models_root = work/"models"
    cfg = dict(base_channels=16, depth=4, max_iters=8, eval_every=4, warmup_iters=2, min_iters=8, log_every=2,
               amp=device.startswith("cuda"))
    cfg.update(config or {})
    summary = dict(work=str(work), device=device, shape_xyz=list(shape), config=cfg, members={},
                   counts=split.counts(), contact=split.label_contact("A"))
    for fold, seed in members:
        record = train_member(files=files, guard=guard, fold=fold, seed=seed, models_root=models_root, config=cfg,
                              device=device, log=log, command=["dry-run"])
        member, _ = calibrate_member(files=files, guard=guard, fold=fold, seed=seed, models_root=models_root,
                                     device=device, log=log, command=["dry-run"])
        summary["members"][f"{fold}{seed}"] = dict(train=record["train_subjects"], calib=record["calib_subjects"],
                                                   temperature=member["temperature"])
    registry_path = work/"segmenters.json"
    register(models_root, registry_path, split=split, allowlist=allowlist, training_source=source, log=log,
             segmenter_id=SEGMENTER_ID_V2, prefix=MODEL_PREFIX_V2, full_training_set=True)
    registry = load_segmenter_registry(registry_path)
    common = dict(files=files, registry=registry, allowlist=allowlist, models_root=models_root, grouping=grouping,
                  device=device, segmenter_id=SEGMENTER_ID_V2, command=["dry-run"], log=log)
    inferred = {}
    for subject in dev:
        prov = segment_subject(subject, pred_root=models_root/"pred", **common)
        groups, _ = read_probabilities(models_root/"pred"/subject)
        inferred[subject] = dict(policy=prov["policy"], model_ids=prov["model_ids"], n_members=len(prov["members"]),
                                 max_abs_sum_minus_1=float(np.max(np.abs(groups.sum(axis=0)-1))))
    refusals = {}
    for name, subject, wanted in (("infer_training_subject", a_train[0], ModelSelectionRefused),
                                  ("infer_sealed", sealed, SubjectSealed)):
        try:
            segment_subject(subject, pred_root=work/"pred_refused", **common)
            refusals[name] = "NOT REFUSED"
        except wanted as exc:
            refusals[name] = f"refused: {exc}"
    try:
        v2_infer_subjects(list(t0), allowlist, source="synthetic_dev")
        refusals["infer_t0_by_the_v2_rule"] = "NOT REFUSED"
    except SystemExit as exc:
        refusals["infer_t0_by_the_v2_rule"] = f"refused: {exc}"
    try:
        read_labels(files, dev[0], guard)
        refusals["label_read_dev"] = "NOT REFUSED"
    except LabelAccessRefused as exc:
        refusals["label_read_dev"] = f"refused: {exc}"
    pair = load_pair(write_pair(work/"pairs"/"dev", work/"data", dev[0], dev[1]), allowlist=allowlist)
    try:
        for side, subject in (("fixed", dev[0]), ("moving", dev[1])):
            provenance = json.loads((models_root/"pred"/subject/"provenance.json").read_text())
            validate_segmentation_provenance(provenance, pair, side=side, layer="structure", registry=registry,
                                             allowlist=allowlist)
        summary["pair_check"] = "accepted"
    except ProvenanceRejected as exc:
        summary["pair_check"] = f"provenance_rejected: {exc}"
    summary.update(inferred=inferred, refusals=refusals, registry=sorted(registry.models),
                   wall_seconds=time.perf_counter()-started)
    summary["ok"] = bool(all(v.startswith("refused") for v in refusals.values()) and summary["pair_check"] == "accepted"
                         and sorted(registry.models) == [f"{MODEL_PREFIX_V2}_foldA", f"{MODEL_PREFIX_V2}_foldB"]
                         and all(v["policy"] == "all_folds_mean" and v["n_members"] == len(members)
                                 and v["max_abs_sum_minus_1"] < 1e-12 for v in inferred.values()))
    write_json(work/"dry_run_summary.json", summary)
    return summary


# ---- argument parsing -----------------------------------------------------------------------------

def _profile_arg(p):
    p.add_argument("--profile", choices=sorted(PROFILES), default="v1", help=f"segmenter: v1 (default) or {V2}")


def _inputs_args(p):
    _profile_arg(p)
    p.add_argument("--synthetic", action="store_true",
                   help="synthetic subjects only (ids >= OASIS_OAS1_0900_MR1, non-real data root); skips the pins")
    p.add_argument("--folds", type=Path, default=None, help="default: the profile's folds file")
    p.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    p.add_argument("--training-source", default=TRAINING_SOURCE, help="allowlist source = the folds' subjects")
    p.add_argument("--expect-counts", default=None,
                   help="n_train,n_calib per fold, or 'none' (synthetic only); default: the profile's (v1 45,8)")
    p.add_argument("--source-root", type=Path, default=None,
                   help="where data/ and the allowlist sources live (default: the snapshot's source repository)")
    p.add_argument("--data-root", type=Path, default=Path("/mnt/d/data"))


def _out_device(p, *, out=True, device=True):
    if out:
        p.add_argument("--out", type=Path, default=None,
                       help="models root (members in fold<F>_s<seed>/); default: the profile's")
    if device:
        p.add_argument("--device", default="cuda")


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m bayesreg.seg.unet.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="synthetic train->calibrate->register->infer")
    parser.add_argument("--work", type=Path, default=None, help="dry run: work directory (default: a new temp dir)")
    parser.add_argument("--dry-device", default="cpu", help="dry run: device")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="v1", dest="dry_profile",
                        help=f"dry run: v1 (default) or {V2}")
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("train")
    p.add_argument("--fold", required=True, choices=("A", "B"))
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--config", type=Path, help="JSON object of TrainConfig overrides")
    p.add_argument("--set", action="append", help="TrainConfig override key=<JSON value>, repeatable")
    _inputs_args(p)
    _out_device(p)
    p = sub.add_parser("calibrate")
    p.add_argument("--fold", required=True, choices=("A", "B"))
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--max-voxels", type=int, default=None)
    _inputs_args(p)
    _out_device(p)
    p = sub.add_parser("check-inputs", help="pinned-input checks + label-guard construction (no label read)")
    p.add_argument("--require-snapshot", action="store_true")
    _inputs_args(p)
    p = sub.add_parser("check-members", help="registry checks of every member (no registry written)")
    p.add_argument("--report", type=Path, default=None, help="default <out>/member_check_<UTC>.json")
    _inputs_args(p)
    _out_device(p, device=False)
    p = sub.add_parser("register")
    p.add_argument("--registry", type=Path, default=None, help="default: the profile's registry")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--segmenter-id", default=None, help="default: the profile's (v1 unet35_v1_ens)")
    p.add_argument("--prefix", default=None, help="default: the profile's (v1 unet35_v1)")
    _inputs_args(p)
    _out_device(p, device=False)
    p = sub.add_parser("evaluate-calibration", help="fold-B ensemble on fold-A calib and vice versa (NLL, ECE)")
    p.add_argument("--max-voxels", type=int, default=2_000_000, help="voxels for the information-only T_ens fit")
    p.add_argument("--report", type=Path, default=None, help="default <out>/calibration_crossfit.json")
    _inputs_args(p)
    _out_device(p)
    p = sub.add_parser("infer")
    _profile_arg(p)
    p.add_argument("--subjects", nargs="*")
    p.add_argument("--subjects-file", type=Path)
    p.add_argument("--all-dev", action="store_true", help=f"--profile {V2}: every subject of --dev-source")
    p.add_argument("--dev-source", default=V2_DEV_SOURCE,
                   help=f"--profile {V2}: the allowlist source it may segment; another value only with --synthetic")
    p.add_argument("--gate-only", action="store_true", help="run the machine gate and exit; nothing is read")
    p.add_argument("--registry", type=Path, default=None, help="default: the profile's registry")
    p.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    p.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    p.add_argument("--grouping", default=None)
    p.add_argument("--allow-unverified-grouping", action="store_true",
                   help="admit a grouping whose status is not default/ablation/verified (recorded)")
    p.add_argument("--segmenter-id", default=None, help="default: the profile's (v1 unet35_v1_ens)")
    p.add_argument("--pred-out", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=Path("/mnt/d/data"))
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--source-root", type=Path, default=None)
    p.add_argument("--skip-machine-gate", action="store_true",
                   help="skip the check that no timed run / training unit is active and (cuda) the GPU is idle "
                        "(recorded in the output; for the lead only)")
    _out_device(p)
    p = sub.add_parser("check-tissue4", help=f"seg35 x seg4 and seg35 adjacency on <= {MAX_TISSUE4_SUBJECTS} "
                                             "training subjects")
    p.add_argument("--n-subjects", type=int, default=MAX_TISSUE4_SUBJECTS)
    p.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    p.add_argument("--report", type=Path, required=True)
    _inputs_args(p)
    return parser


COMMANDS = {"train": cmd_train, "calibrate": cmd_calibrate, "check-inputs": cmd_check_inputs,
            "check-members": cmd_check_members,
            "register": cmd_register, "evaluate-calibration": cmd_evaluate_calibration, "infer": cmd_infer,
            "check-tissue4": cmd_check_tissue4}


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.dry_profile != "v1" and not args.dry_run:
        raise SystemExit("--profile goes after the subcommand (before it, it selects the --dry-run)")
    if args.dry_run and args.dry_profile == V3:
        raise SystemExit(f"--dry-run --profile {V3}: experiments/ul/run_ul.py --dry-run")
    if args.dry_run and args.dry_profile == V5:
        raise SystemExit(f"--dry-run --profile {V5}: experiments/r2/run_r2.py --dry-run")
    if args.dry_run and args.dry_profile == V2:
        result = dry_run_v2(args.work, device=args.dry_device)
        print(json.dumps(dict(ok=result["ok"], work=result["work"], wall_seconds=round(result["wall_seconds"], 1),
                              counts=result["counts"], inferred=result["inferred"], pair_check=result["pair_check"],
                              refusals=result["refusals"], registry=result["registry"]), indent=1))
        return 0 if result["ok"] else 1
    if args.dry_run:
        result = dry_run(args.work, device=args.dry_device)
        print(json.dumps(dict(ok=result["ok"], work=result["work"], wall_seconds=round(result["wall_seconds"], 1),
                              inferred=result["inferred"], pair_checks=result["pair_checks"],
                              members=result["members"], crossfit=result["crossfit"]), indent=1))
        return 0 if result["ok"] else 1
    if args.command not in COMMANDS:
        build_parser().print_help()
        return 2
    print(json.dumps(COMMANDS[_apply_profile(args).command](args), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
