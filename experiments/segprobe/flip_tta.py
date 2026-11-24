"""SEGP P2: left-right flip test-time augmentation of unet35_v1_ens and unet35_v2_full (runs/20261105-segprobe/任务书.md). A tool: the real run is the lead's, on the GPU, after UL.

Per subject, segmenter and member m: p_m = softmax(z_m/T_m) on the image; q_m = the same on its left-right mirror image, flipped back along x (the last array axis of the
(nz, ny, nx) RAS storage) with the left and right classes swapped. The table comes from bayesreg.seg.unet.large.left_right_label_map (read from the label names; 0, 11, 12, 13 map to themselves). Arms:
  unflipped  the mean of p_m, renormalized (bayesreg.seg.unet.infer.ensemble_probabilities); the control: it must reproduce the committed numbers of the segmenter
  flip_only  the mean of q_m, renormalized
  tta        (unflipped + flip_only)/2, renormalized: one forward each, averaged, which is what the brief asks
Inference reads no truth. The truth is read afterwards, for the evaluation only, by experiments/unv2/eval_seg.py summarize_subject: seg35, seg4, the small six, the big six, the other 23.
Seconds per subject: the two forward passes of every member with the device synchronized around them, and the whole subject with image reading and normalizing (the argmax and the file are outside).

Already measured (runs/20261103-n0/result.md, section g): on the four members of v1 tta - unflipped was -.0052 / -.0008 (1/20, 0/20; seg35 / seg4), rule "not worth using". Those members and the members of v2 are
trained without the flip (bayesreg/seg/unet/data.py augment). The v1 part of this tool repeats that measurement; the v2 part is new. The reading, the rule of N0 (g): mean seg35 gain >= +.002 and at least 15/20 positive.

Before any image is opened: the subjects of the allowlist source (eval_seg.admit_subjects: SubjectSealed for T0, a sealed list or a training subject), the registered members found by sha256. On cuda: the process capped
at --gpu-cap-gib, and refused (GpuBusy) if less than --min-free-gib is free, so that it does not start beside a training (the driver-level OOM of 11-04). This tool starts no unit and no solver. Outputs in --out-dir
(never overwritten): <segmenter>/<subject>.npz (top, top_flip, top_tta, uint8), flip_tta.json, per_subject.csv, summary.md.

    python -B experiments/segprobe/flip_tta.py --dry-run
    python -B experiments/segprobe/flip_tta.py --out-dir DIR [--segmenters v1 v2] [--subjects ID ...] [--device cuda]

--dry-run: no GPU, no image and no truth: admission, the registered members re-hashed and loaded on the CPU, the swap table, and two passes of a tiny UNet3D on a synthetic volume on the CPU. Exit 0 if every check passed.

v4 (work package TTA, runs/20261107-tta/plan.md and 任务书.md): UF4 = unet35_v4_ens, the four members of seg/unet35_v4_full registered in configs/bayesreg/segmenters_hs_v1.json, alone (not with v1 or v2).
  Inference: the arm unflipped is hs_seg.uf4_probabilities itself, the registered inference of UF4 (HS's seg phase: infer.load_member, predict_logits with each member's own amp, window and window_gaussian_sigma,
  softmax(z/T_m), the members in HS's order A0 A1 B0 B1, infer.ensemble_probabilities). flip_only: the same members, loaded again the same way, with the parameters the unflipped pass recorded for that member, on the
  mirror image of the same normalized image; ensemble_probabilities of those logits, then flip_back. tta: (unflipped + flip_only)/2, renormalized. The argmax of every arm is taken as eval_uf and hs_seg take UF4's:
  of the float16 cast (as probs36 is stored), numpy, ties to the smallest label.
  Before any image: the plan of HS (hs_seg.uf_plan) for every subject; the flip premise of the plan (every member's train_record.json, through the sha256 member.json names: recipe v3_large, flip_probability > 0,
  flip_axis_check true, the axis check done, the left-right pairs of training those of this tool; FlipPremiseRefused otherwise); the registered predictions' provenance (seg/unet35_v4_ens/pred: this registry, these members).
  The control (plan.md, the first rule): the unflipped means of the 20 subjects within CONTROL_TOL of eval_uf.json means.UF4, and on every subject the agreement of its argmax with the registered prediction's
  (probs36.npz key "p", float16 (36, nz, ny, nx), read through segprobe_common.load_raw) at least AGREEMENT_MIN. A control that fails makes the reading 不判; the run is not stopped and nothing is rerun.
  The control requires exactly the 20 development subjects 0438-0457; the decision also requires 20 finite seg35 differences. A 不判 is written with worth_using null and 不解读 (plan.md: 不判，不解释).
  The read lock /home/rmcfz/seq_vp_tag_results/seg/tta_dev_read.lock (the fields of uf_dev_read.lock): checked first, taken with O_EXCL immediately before the first truth label; a lock that is there, whatever its
  content, refuses the run (DevReadLocked). Then plan.md's sha256 against PLAN_SHA256 (PlanChanged otherwise). The real run needs --real-run; without it and without --dry-run, v4 is refused before anything is read.
  Outputs in --out-dir (default segprobe_v1/tta_v4, never overwritten): v4/<subject>.npz (top, top_flip, top_tta), flip_tta.json (decision 过门 / 不过门 / 不判), per_subject.csv, summary.md.
  Seconds: v4's forward_s is NOT v1/v2's pure forward time: it is unflipped_s + flip_s, the two whole passes with the image read and the members re-hashed and loaded (hs_seg does not time its forward passes apart);
  flip_forward_s is the pure forward time of the mirror pass. host_peak_rss_gib is resource.getrusage's ru_maxrss (a running maximum).

    python -B experiments/segprobe/flip_tta.py --dry-run --segmenters v4
    python -B experiments/segprobe/flip_tta.py --segmenters v4 --real-run [--out-dir DIR] [--device cuda]

--dry-run --segmenters v4: no GPU, no image of a subject, no truth, no lock written: admission, the registry, HS's plan of every subject, the flip premise, the registered predictions' provenance, then the real members
re-hashed and loaded on the CPU and the whole inference of v4_probabilities (two passes of each member) on a synthetic 32^3 volume in a temporary directory.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segprobe_common as common  # noqa: E402
from segprobe_common import REPO, eval_seg  # noqa: E402
from bayesreg.io.guard import SubjectSealed, load_dev_allowlist  # noqa: E402
from bayesreg.io.pair import ProvenanceRejected, load_segmenter_registry  # noqa: E402
from bayesreg.seg.unet.data import IMAGE_FILE, OasisFiles, normalize_image, read_image  # noqa: E402
from bayesreg.seg.unet.infer import ensemble_probabilities, load_member, select_fold_models  # noqa: E402
from bayesreg.seg.unet.large import FLIP_DIM, GIB, PROFILE, cap_gpu_memory, left_right_label_map  # noqa: E402
from bayesreg.seg.unet.model import build_model  # noqa: E402
from bayesreg.seg.unet.registry import find_member_files  # noqa: E402
from bayesreg.seg.unet.snapshot import require_clean_code  # noqa: E402
from bayesreg.seg.unet.train import environment_record, file_sha256, predict_logits  # noqa: E402

FORMAT = "bayesreg.segprobe.flip_tta.v1"
SEGMENTERS = dict(v1=dict(id=common.V1_ID, registry=REPO/"configs/bayesreg/segmenters.json", models_root=Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v1"), reference=(.9259821660395682, .9583579541106246)),
                  v2=dict(id=common.V2_ID, registry=REPO/"configs/bayesreg/segmenters_v2.json", models_root=Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v2_full"), reference=(.9275412471659505, .9590838826390028)),
                  # TTA (runs/20261107-tta): UF4 as HS registered it; reference = eval_uf.json means.UF4 (seg35, seg4); candidate = the key of experiments/hs/hs_core.CANDIDATES; the registry's sha256 is the file's, recorded
                  v4=dict(id="unet35_v4_ens", registry=REPO/"configs/bayesreg/segmenters_hs_v1.json", models_root=Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v4_full"), reference=(0.9300092857879709, 0.9605822402722083),
                          pred_root=Path("/home/rmcfz/seq_vp_tag_results/seg/unet35_v4_ens/pred"), candidate="UF4"))
ARMS = ("unflipped", "flip_only", "tta")
TOP_KEYS = dict(unflipped="top", flip_only="top_flip", tta="top_tta")
METRICS = ("seg35", "seg4")+tuple(eval_seg.GROUPS)
WORTH_GAIN, WORTH_POSITIVE = .002, 15            # the rule of N0 (g)
CONTROL_TOL = 5e-4                               # the arm unflipped against the committed mean (the GPU is not bitwise repeatable; a report, the run is not stopped)
GPU_CAP_GIB, MIN_FREE_GIB = 14., 12.             # N0: peak allocated 6.64-7.56 GiB, reserved 8.80-9.72 GiB

V4 = "v4"
FORMAT_V4 = "bayesreg.segprobe.flip_tta_v4.v1"
V4_PLAN = REPO/"runs/20261107-tta/plan.md"
PLAN_SHA256 = "1098f388ff8a1c69abaec8f3041370d4d60c30bd511367775666b4da125bf183"     # the preregistration as committed in 8423604 (review m4): a real run refuses another plan.md
V4_OUT = common.OUT_ROOT/"tta_v4"
TTA_DEV_READ_LOCK = Path("/home/rmcfz/seq_vp_tag_results/seg/tta_dev_read.lock")    # beside uf_dev_read.lock; outside any results root, not derived from an option
FIRST_READ = "first_read"
AGREEMENT_MIN = .9999                            # plan.md, the control: per subject, the unflipped argmax against the registered prediction's
N_DEV = 20                                       # the 20 development subjects of the reference means
DEV_SUBJECTS = tuple(f"OASIS_OAS1_{k:04d}_MR1" for k in range(438, 458))           # those subjects (plan.md: 0438-0457; eval_uf.json subjects): the control requires exactly this set (review m2)
PASS, FAIL, UNDECIDED = "过门", "不过门", "不判"     # plan.md, the reading
NOT_INTERPRETED = "不解读"                        # what a 不判 says beside its numbers (plan.md: 不判，不解释)
SEG4_TISSUES = dict(cortex=common.SEG4_CORTEX, white_matter=common.SEG4_WM)          # the seg4 labels 1 and 3 (eval_nn.metrics_of, eval_uf.METRICS)
METRICS_V4 = METRICS+tuple(SEG4_TISSUES)
SYNTHETIC_SUBJECT = "OASIS_OAS1_0901_MR1"        # the dry run's volume: a canonical id that does not exist in OASIS (bayesreg.seg.unet.synthetic)
CODE_FILES = ("experiments/segprobe/flip_tta.py", "experiments/segprobe/segprobe_common.py", "experiments/hs/hs_seg.py", "experiments/hs/hs_core.py", "experiments/unv2/eval_seg.py", "experiments/nn/eval_nn.py",
              "bayesreg/seg/unet/infer.py", "bayesreg/seg/unet/train.py", "bayesreg/seg/unet/large.py")


class GpuBusy(Exception):
    """Too little free GPU memory: something else is on the GPU. Status `gpu_busy`."""
    status = "gpu_busy"


class FlipPremiseRefused(Exception):
    """A v4 member is not recorded as trained with the left-right flip the plan's hypothesis assumes; nothing of a subject was opened. Status `flip_premise_refused`."""
    status = "flip_premise_refused"


class DevReadLocked(Exception):
    """The development truth was read for this reading already (tta_dev_read.lock); it is read once and the file is not deleted. Status `dev_read_locked`."""
    status = "dev_read_locked"


class PlanChanged(Exception):
    """runs/20261107-tta/plan.md is not the preregistration PLAN_SHA256 pins; nothing is read. Status `plan_changed`."""
    status = "plan_changed"


def swap_table():
    """Long tensor lut: lut[l] = the label of the mirrored structure (an involution), from the seg35 label names."""
    lut, _, _ = left_right_label_map()
    return torch.as_tensor(lut.astype(np.int64))


def flip_back(q, lut):
    """q (C, D, H, W): the posterior of the mirror image. Mirrored back along x, then the classes swapped: on the mirror image the network calls the original left structure "right"."""
    return torch.flip(q, (FLIP_DIM,))[lut.to(q.device)]


def renormalized(total, n):
    total = total/n
    return total/total.sum(dim=0, keepdim=True)


def timed(device, function):
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.synchronize(torch.device(device))
    started = time.perf_counter()
    result = function()
    if cuda:
        torch.cuda.synchronize(torch.device(device))
    return result, time.perf_counter()-started


def tta_probabilities(members, image, *, device, lut):
    """(unflipped, flip_only, tta) float32 tensors (C, D, H, W), seconds of the forward passes. members: dicts with model, temperature, amp, window, gaussian_sigma; image (D, H, W) float32 numpy."""
    mirrored = np.flip(image, axis=FLIP_DIM).copy()
    total_p, total_q, seconds = None, None, 0.

    def forward(member, volume):
        z = predict_logits(member["model"], volume, device=device, amp=member["amp"], window=member["window"], gaussian_sigma=member["gaussian_sigma"])
        return torch.softmax(z.float()/float(member["temperature"]), dim=1)[0]

    for member in members:
        p, s = timed(device, lambda m=member: forward(m, image))
        seconds += s
        total_p = p if total_p is None else total_p.add_(p)
        q, s = timed(device, lambda m=member: forward(m, mirrored))
        seconds += s
        q = flip_back(q, lut)
        total_q = q if total_q is None else total_q.add_(q)
    unflipped, flip_only = renormalized(total_p, len(members)), renormalized(total_q, len(members))
    return unflipped, flip_only, renormalized(unflipped+flip_only, 2), seconds


def tops(unflipped, flip_only, tta):
    return dict(top=unflipped.argmax(0).to(torch.uint8).cpu().numpy(), top_flip=flip_only.argmax(0).to(torch.uint8).cpu().numpy(), top_tta=tta.argmax(0).to(torch.uint8).cpu().numpy())


def select_subjects(admitted, requested):
    """`requested` (None: all) must lie among the admitted development subjects; SubjectSealed otherwise, before anything is read."""
    if not requested:
        return list(admitted)
    outside = sorted(set(requested)-set(admitted))
    if outside:
        raise SubjectSealed(f"{outside}: not among the admitted development subjects")
    return [s for s in admitted if s in set(requested)]


def require_free_gpu(device, minimum_gib, *, cuda=torch.cuda):
    """The free memory of the GPU as the driver reports it; GpuBusy below `minimum_gib`, ValueError for a device that is not there."""
    if not cuda.is_available():
        raise ValueError(f"device {device}: CUDA is not available")
    free, total = cuda.mem_get_info(torch.device(device))
    if free/GIB < minimum_gib:
        raise GpuBusy(f"{free/GIB:.1f} GiB of {total/GIB:.1f} GiB free on the GPU, at least {minimum_gib} required: something else is running (GPU training and this job are not run together)")
    return dict(free_gib=free/GIB, total_gib=total/GIB)


def load_members(name, subjects, device, *, allowlist):
    """The members of a registered segmenter for these subjects: policy all_folds_mean for every subject (no member saw their labels), each file re-hashed by load_member."""
    spec = SEGMENTERS[name]
    registry = load_segmenter_registry(spec["registry"])
    selected = [select_fold_models(registry, s, allowlist, segmenter_id=spec["id"]) for s in subjects]
    if any(policy != "all_folds_mean" for _, policy in selected) or any(sorted(m) != sorted(selected[0][0]) for m, _ in selected):
        raise ValueError(f"{name}: the policy or the models differ from all_folds_mean over the subjects")
    available = find_member_files(spec["models_root"])
    members = []
    for model_id, entry in selected[0][0].items():
        for sha in entry["member_model_sha256"]:
            if sha not in available:
                raise ValueError(f"{name}: member {sha[:12]}... of {model_id} not found under {spec['models_root']}")
            model, blob = load_member(available[sha], sha, device, entry=entry)
            members.append(dict(model=model, temperature=blob["temperature"], amp=blob["amp"], window=tuple(blob["window"]) if blob.get("window") else None, gaussian_sigma=blob.get("window_gaussian_sigma"),
                                percentile=blob["percentile"], fold=blob["fold"], seed=blob["seed"], sha256=sha, model_id=model_id))
    if len({m["percentile"] for m in members}) != 1:
        raise ValueError(f"{name}: members disagree on the normalization percentile")
    return members


def member_record(members):
    return [{k: m[k] for k in ("model_id", "fold", "seed", "temperature", "amp", "window", "gaussian_sigma", "percentile", "sha256")} for m in members]


def infer_all(subjects, members_of, *, read_volume, device, lut, out_dir, log=print):
    """One subject at a time, the tops written at once; no truth here. read_volume(subject): (values, metadata). Returns {segmenter: {subject: seconds record}}."""
    seconds = {name: {} for name in members_of}
    for subject in subjects:
        values, meta = read_volume(subject)
        for name, members in members_of.items():
            started = time.perf_counter()
            image, _ = normalize_image(values, percentile=members[0]["percentile"])
            if torch.device(device).type == "cuda":
                torch.cuda.reset_peak_memory_stats(torch.device(device))
            with torch.no_grad():
                unflipped, flip_only, tta, forward = tta_probabilities(members, image, device=device, lut=lut)
            whole = time.perf_counter()-started
            maps = tops(unflipped, flip_only, tta)
            del unflipped, flip_only, tta
            directory = Path(out_dir)/name
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory/f"{subject}.npz", **maps)
            gpu = torch.cuda.max_memory_allocated(torch.device(device))/GIB if torch.device(device).type == "cuda" else None
            seconds[name][subject] = dict(forward_s=forward, subject_s=whole, n_forward=2*len(members), gpu_peak_allocated_gib=gpu, image_sha256=meta.get("sha256"))
            log(f"{subject} {name}: forward {forward:.1f} s, subject {whole:.1f} s", flush=True)
    return seconds


def evaluate_tops(subjects, names, *, out_dir, read_truth, lookup):
    """{segmenter: {subject: {arm: row}}} from the written tops. The truth is opened only here."""
    rows = {name: {} for name in names}
    truth_sha = {}
    for subject in subjects:
        true35, sha35 = read_truth(subject, "seg35")
        true4, sha4 = read_truth(subject, "seg4")
        truth_sha[subject] = dict(seg35=sha35, seg4=sha4)
        for name in names:
            with np.load(Path(out_dir)/name/f"{subject}.npz") as npz:
                rows[name][subject] = {arm: eval_seg.summarize_subject(npz[TOP_KEYS[arm]], true35, true4, lookup) for arm in ARMS}
    return rows, truth_sha


def value(row, metric):
    if metric in SEG4_TISSUES:                   # v4 only (METRICS_V4): cortex and white matter, the seg4 labels 1 and 3
        return row["seg4_per_label"][SEG4_TISSUES[metric]-1]
    return row[metric] if metric in ("seg35", "seg4") else row["groups"][metric]


def tabulate(rows, subjects, references, metrics=METRICS):
    """Means per arm, tta - unflipped per subject, the rule of N0 (g), the control of the arm unflipped."""
    out = {}
    for name, per_subject in rows.items():
        column = {arm: {m: [value(per_subject[s][arm], m) for s in subjects] for m in metrics} for arm in ARMS}
        means = {arm: {m: eval_seg.nanmean(column[arm][m]) for m in metrics} for arm in ARMS}
        paired = {m: eval_seg.paired(column["tta"][m], column["unflipped"][m]) for m in metrics}
        gain = paired["seg35"]
        reference = references.get(name) if len(subjects) == 20 else None
        out[name] = dict(means=means, tta_minus_unflipped=paired, rule=f"mean seg35 gain >= +{WORTH_GAIN} and at least {WORTH_POSITIVE}/20 positive (N0 g)",
                         worth_using=bool(gain["mean"] is not None and gain["mean"] >= WORTH_GAIN and gain["n_positive"] >= WORTH_POSITIVE),
                         control=None if reference is None else dict(committed=reference, recomputed=[means["unflipped"]["seg35"], means["unflipped"]["seg4"]],
                                                                     passed=bool(abs(means["unflipped"]["seg35"]-reference[0]) <= CONTROL_TOL and abs(means["unflipped"]["seg4"]-reference[1]) <= CONTROL_TOL), tol=CONTROL_TOL))
    return out


def summary_markdown(document):
    f = eval_seg.fmt
    lines = ["# SEGP P2：左右翻转推断（开发集，单次观测；分割级，不是配准结果）", ""]
    for name, table in document["tables"].items():
        lines += [f"## {name}", "", "| 臂 | "+" | ".join(METRICS)+" |", "|---|"+"---|"*len(METRICS)]
        lines += [f"| {arm} | "+" | ".join(f(table["means"][arm][m]) for m in METRICS)+" |" for arm in ARMS]
        lines += ["", "| tta − unflipped | 均值差 | 中位差 | 正 | 负 | 零 |", "|---|---|---|---|---|---|"]
        lines += [f"| {m} | {f(p['mean'], 5)} | {f(p['median'], 5)} | {p['n_positive']} | {p['n_negative']} | {p['n_zero']} |" for m, p in table["tta_minus_unflipped"].items()]
        seconds = [r["forward_s"] for r in document["seconds"][name].values()]
        lines += ["", f"每人前向秒数（两个方向，全部成员）：中位 {np.median(seconds):.1f}，最大 {np.max(seconds):.1f}。规则（N0 g）：{table['rule']}：{'值得用' if table['worth_using'] else '不值得用'}。对照：{table['control']}", ""]
    return "\n".join(lines)


def tiny_members(*, seed=0):
    """A tiny UNet3D on the CPU, for the dry run and the tests."""
    torch.manual_seed(seed)
    model = build_model(dict(base_channels=4, depth=2)).eval()
    return [dict(model=model, temperature=1.2, amp=False, window=None, gaussian_sigma=None, percentile=99.5)]


def dry_run(names, subjects, *, allowlist, real_members=True, log=print):
    """No GPU, no image, no truth. Returns the record; AssertionError or ValueError if something is wrong."""
    if V4 in names:
        raise ValueError("v4 has its own dry run, dry_run_v4, alone")
    lut = swap_table()
    lookup, pairs, unpaired = left_right_label_map()
    assert sorted(unpaired) == [0, 11, 12, 13], unpaired
    assert np.array_equal(lookup[lookup], np.arange(36))
    log(f"left-right table: {len(pairs)} pairs {pairs}; to themselves {unpaired}")
    record = dict(lut=lut.tolist(), subjects=subjects)
    if real_members:
        for name in names:
            members = load_members(name, subjects, "cpu", allowlist=allowlist)
            record[name] = member_record(members)
            log(f"{name}: {len(members)} members, temperatures {[round(m['temperature'], 4) for m in members]}, window {members[0]['window']}, percentile {members[0]['percentile']}")
            del members
    image = np.random.default_rng(0).random((16, 16, 16), dtype=np.float32)
    with torch.no_grad():
        unflipped, flip_only, tta, seconds = tta_probabilities(tiny_members(), image, device="cpu", lut=lut)
    for tensor in (unflipped, flip_only, tta):
        assert tensor.shape == (36, 16, 16, 16)
        assert float((tensor.sum(0)-1).abs().max()) < 1e-5
    log(f"synthetic volume (16, 16, 16), tiny UNet3D on the CPU: {seconds:.3f} s for the two passes; the channel sums are 1")
    record["synthetic_seconds"] = seconds
    return record


def run(*, names, out_dir, device, requested=None, gpu_cap_gib=GPU_CAP_GIB, min_free_gib=MIN_FREE_GIB, data_root=eval_seg.DATA_ROOT, argv=None, log=print):
    if V4 in names:
        raise ValueError("v4 runs through run_v4, alone")
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} exists and is not empty; results are never overwritten")
    admitted, admission = common.admit_dev_subjects(data_root=data_root)          # before any file of a subject is opened
    subjects = select_subjects(admitted, requested)
    allowlist = load_dev_allowlist(eval_seg.ALLOWLIST)
    gpu = None
    if torch.device(device).type == "cuda":
        gpu = require_free_gpu(device, min_free_gib)
        cap_gpu_memory(torch.device(device), gpu_cap_gib)
    members_of = {name: load_members(name, subjects, device, allowlist=allowlist) for name in names}
    lut = swap_table()
    files = OasisFiles(root=data_root)
    seconds = infer_all(subjects, members_of, read_volume=lambda s: read_image(files, s), device=device, lut=lut, out_dir=out_dir, log=log)
    rows, truth_sha = evaluate_tops(subjects, names, out_dir=out_dir, read_truth=lambda s, k: eval_seg.read_truth(data_root, s, k), lookup=eval_seg.load_seg4_lookup())
    tables = tabulate(rows, subjects, {n: SEGMENTERS[n]["reference"] for n in names})
    document = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), command=argv, admission=admission, subjects=subjects, device=str(device), gpu=gpu, gpu_cap_gib=gpu_cap_gib,
                    members={n: member_record(m) for n, m in members_of.items()}, lut=lut.tolist(), seconds=seconds, tables=tables, truth_sha256=truth_sha, rows=rows,
                    note="开发集，单次观测；分割级读数，不是配准结果。多个臂看同一批开发受试者，带选择效应。v1 的部分重复 N0 (g)。")
    common.write_json(out_dir/"flip_tta.json", document)
    with (out_dir/"per_subject.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["segmenter", "subject", "arm", *METRICS, "forward_s"])
        for name in names:
            for subject in subjects:
                for arm in ARMS:
                    writer.writerow([name, subject, arm]+[value(rows[name][subject][arm], m) for m in METRICS]+[seconds[name][subject]["forward_s"]])
    (out_dir/"summary.md").write_text(summary_markdown(document), encoding="utf-8")
    return document


# ---- v4: UF4 (unet35_v4_ens) through the registered inference of HS, runs/20261107-tta --------------------------------------------------------------

def hs_seg_module():
    """experiments/hs/hs_seg.py (HS's seg phase: the registered inference of UF4), imported on first use; the v1 and v2 paths never import it."""
    here = str(REPO/"experiments/hs")
    if here not in sys.path:
        sys.path.insert(0, here)
    import hs_seg
    return hs_seg


def sha_of(path):
    return file_sha256(path) if Path(path).is_file() else None


def flip_premise(plan, *, lut=None):
    """Before any image (plan.md: the hypothesis assumes it): for every member of HS's plan [(model id, sha256, member.pt)], member.json beside member.pt names that sha256 and the sha256 of train_record.json; the record is of
    the recipe v3_large, config.flip_probability > 0, config.flip_axis_check true, its flip_check done (large.check_flip_axis on the training labels, not skipped), and its left-right pairs and unpaired labels those of
    left_right_label_map, the table this tool swaps with (the axis is large.FLIP_DIM in both). FlipPremiseRefused with every problem otherwise. Returns {member directory: record}; the flips that train_log.jsonl's end event
    counts are described (that file is not pinned by a hash)."""
    table, pairs, unpaired = left_right_label_map()
    if lut is not None and list(map(int, lut)) != list(map(int, table)):
        raise FlipPremiseRefused("the swap table is not left_right_label_map's")
    records, problems = {}, []
    for model_id, sha, path in plan:
        directory = Path(path).parent
        name = directory.name
        try:
            member = json.loads((directory/"member.json").read_text(encoding="utf-8"))
            train_sha = file_sha256(directory/"train_record.json")
            train = json.loads((directory/"train_record.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"{name}: {exc}")
            continue
        config, check = train.get("config") or {}, train.get("flip_check") or {}
        probability = config.get("flip_probability")
        mine = []
        if member.get("member_sha256") != sha:
            mine.append("member.json names another member.pt")
        if member.get("train_record_sha256") != train_sha:
            mine.append("train_record.json is not the file member.json names (sha256)")
        if train.get("recipe") != PROFILE:
            mine.append(f"recipe {train.get('recipe')!r}, not {PROFILE!r}")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not probability > 0:
            mine.append(f"config.flip_probability {probability!r}: not > 0")
        if config.get("flip_axis_check") is not True:
            mine.append(f"config.flip_axis_check {config.get('flip_axis_check')!r}: not true")
        if check.get("skipped") or not check.get("n_subjects"):
            mine.append(f"flip_check {check!r}: the axis check was not done on training labels")
        if train.get("left_right_pairs") != [[int(a), int(b)] for a, b in pairs] or train.get("unpaired_labels") != [int(u) for u in unpaired]:
            mine.append("the left-right pairs of training are not this tool's swap table")
        flips = None
        log_path = directory/"train_log.jsonl"
        if log_path.is_file():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if '"event": "end"' in line:
                    flips = json.loads(line).get("flips")
        records[name] = dict(model_id=model_id, member_sha256=sha, train_record_sha256=train_sha, recipe=train.get("recipe"), flip_probability=probability, flip_axis_check=config.get("flip_axis_check"),
                             flip_check={k: check.get(k) for k in ("check", "n_subjects", "smallest_x_separation_vox", "min_separation_vox")}, steps=train.get("iterations_run"), batch_size=config.get("batch_size"),
                             flips_logged=flips, problems=mine)
        problems += [f"{name}: {p}" for p in mine]
    if problems:
        raise FlipPremiseRefused(f"not run, no image opened: {problems}")
    return records


def registered_provenance(subjects, *, pred_root, segmenter_id, registry_sha256, member_sha256):
    """Before any image: the registered prediction of every subject (seg/unet35_v4_ens/pred/<subject>/provenance.json) is of this segmenter, this registry file and these members in this order, all_folds_mean,
    test_labels_seen false. ValueError otherwise (the control would be against another registration). Returns {subject: record}; the outputs' hashes are checked when the prediction is read (registered_agreement)."""
    out, problems = {}, []
    for subject in subjects:
        path = Path(pred_root)/subject/"provenance.json"
        try:
            provenance = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"{subject}: {exc}")
            continue
        mine = []
        if provenance.get("subject") != subject or provenance.get("layer") != "structure" or provenance.get("segmenter_id") != segmenter_id:
            mine.append(f"subject {provenance.get('subject')!r}, layer {provenance.get('layer')!r}, segmenter {provenance.get('segmenter_id')!r}")
        if provenance.get("registry_sha256") != registry_sha256:
            mine.append(f"registry sha256 {provenance.get('registry_sha256')}, the registry file's {registry_sha256}")
        if [m.get("member_sha256") for m in provenance.get("members") or []] != list(member_sha256):
            mine.append("the members are not HS's plan, in order")
        if provenance.get("policy") != "all_folds_mean" or provenance.get("test_labels_seen") is not False:
            mine.append(f"policy {provenance.get('policy')!r}, test_labels_seen {provenance.get('test_labels_seen')!r}")
        problems += [f"{subject}: {p}" for p in mine]
        environment = provenance.get("environment") or {}
        out[subject] = dict(provenance_sha256=file_sha256(path), image_sha256=(provenance.get("input_sha256") or {}).get(IMAGE_FILE), probs36_sha256=(provenance.get("output_sha256") or {}).get("probs36.npz"),
                            code_commit=(environment.get("code") or {}).get("commit"), cudnn_benchmark=environment.get("cudnn_benchmark"), cudnn_deterministic=environment.get("cudnn_deterministic"),
                            torch=environment.get("torch"), cuda=environment.get("cuda"), cudnn=environment.get("cudnn"), gpu=environment.get("gpu"))     # to set beside this run's environment (review m7)
    if problems:
        raise ValueError(f"the registered predictions are not this registration of {segmenter_id}: {problems}")
    return out


def v4_inputs(subjects, *, allowlist, spec, lut=None):
    """Before any image: the registry (its sha256 the file's), HS's plan of every subject (hs_seg.uf_plan: the admission, all_folds_mean over every registered model, the members found by sha256), the four members in
    HS's order (hs_core.UF_MEMBERS: A0 A1 B0 B1) and the same for every subject, the flip premise of the members, the registered predictions' provenance."""
    hs = hs_seg_module()
    registry = load_segmenter_registry(spec["registry"])
    available = find_member_files(spec["models_root"])
    plans = {s: hs.uf_plan(registry, spec["candidate"], s, allowlist, available) for s in subjects}
    order = [hs.hc.member_name(f, s) for f, s in hs.hc.UF_MEMBERS]
    first = plans[subjects[0]][0]
    for subject, (plan, _, _) in plans.items():
        if [Path(p).parent.name for _, _, p in plan] != order or [(m, sha) for m, sha, _ in plan] != [(m, sha) for m, sha, _ in first]:
            raise ValueError(f"{subject}: HS's plan {[(m, Path(p).parent.name) for m, _, p in plan]} is not the four members in the order {order}")
    premise = flip_premise(first, lut=lut)
    registered = registered_provenance(subjects, pred_root=spec["pred_root"], segmenter_id=spec["id"], registry_sha256=registry.sha256, member_sha256=[sha for _, sha, _ in first])
    return dict(registry=registry, plans=plans, order=order, premise=premise, registered=registered, members={Path(p).parent.name: sha for _, sha, p in first})


def v4_probabilities(subject, *, files, registry, models, plan, device, lut):
    """(unflipped, flip_only, tta, record): float32 numpy (36, nz, ny, nx) each, for one subject.
    unflipped  hs_seg.uf4_probabilities, the registered inference of UF4, called as HS's seg phase calls it.
    flip_only  the same members, re-hashed and loaded again (infer.load_member), each with the parameters the unflipped pass recorded for it (amp, window, window_gaussian_sigma, temperature, percentile: checked against
               member.pt, ValueError otherwise), on the mirror along FLIP_DIM of the same normalized image (its sha256 and normalization checked against the unflipped pass's); infer.ensemble_probabilities of those
               logits, then flip_back. flip_back permutes voxels and classes, so it commutes with the mean over members and the division by the channel sum: the mean of the members' flipped-back posteriors (v1/v2's
               flip_only) up to the order of the 36-term channel sum.
    tta        (unflipped + flip_only)/2, renormalized (as v1/v2's).
    The seconds of the record are not v1/v2's forward_s: unflipped_s is the whole hs_seg.uf4_probabilities call (the image read and normalized, every member re-hashed and loaded, the forward passes; hs_seg does not
    time its forward passes apart), flip_s the whole flip pass (the image read again, the members re-hashed and loaded again, the forward passes, the mean, flip_back on the host), forward_s = unflipped_s + flip_s;
    flip_forward_s alone is pure forward time (the mirror's predict_logits, the device synchronized around each). An estimate of the unflipped pass's pure forward time is flip_forward_s (the same members, windows, AMP)."""
    hs = hs_seg_module()
    clock = time.perf_counter()
    unflipped, meta, norm, members = hs.uf4_probabilities(subject, files=files, registry=registry, models=models, plan=plan, device=device)      # synchronizes the device before it returns
    unflipped_s = time.perf_counter()-clock
    clock = time.perf_counter()
    values, again = read_image(files, subject)
    image, norm_again = normalize_image(values, percentile=norm["percentile"])
    del values
    if again["sha256"] != meta["sha256"] or norm_again != norm:
        raise ValueError(f"{subject}: the image or its normalization differs between the two passes")
    mirrored = np.flip(image, axis=FLIP_DIM).copy()
    del image
    if [m["member_sha256"] for m in members] != [sha for _, sha, _ in plan]:
        raise ValueError(f"{subject}: the unflipped pass's members are not the plan's")
    forward = dict(seconds=0., calls=0)

    def mirror_logits():
        for (model_id, sha, path), unflipped_member in zip(plan, members):
            model, blob = load_member(path, sha, device, entry=models[model_id])
            used = dict(model_id=model_id, member_sha256=sha, temperature=blob["temperature"], amp=blob["amp"], window=list(blob["window"]), window_gaussian_sigma=blob["window_gaussian_sigma"], percentile=blob["percentile"])
            differ = sorted(k for k, v in used.items() if unflipped_member.get(k) != v)
            if differ:
                raise ValueError(f"{path}: the flip pass would use other {differ} than the unflipped pass")
            z, seconds = timed(device, lambda m=model, b=blob: predict_logits(m, mirrored, device=device, amp=b["amp"], window=tuple(b["window"]), gaussian_sigma=b["window_gaussian_sigma"]))
            forward["seconds"] += seconds
            forward["calls"] += 1
            del model
            yield z, blob["temperature"]

    total = ensemble_probabilities(mirror_logits())
    flip_only = flip_back(total.cpu(), lut)
    del total
    flip_s = time.perf_counter()-clock
    tta = renormalized(torch.from_numpy(unflipped)+flip_only, 2)
    record = dict(unflipped_s=unflipped_s, flip_s=flip_s, forward_s=unflipped_s+flip_s, flip_forward_s=forward["seconds"], n_forward=len(members)+forward["calls"], image_sha256=meta["sha256"], normalization=norm,
                  members=members)
    return unflipped, flip_only.numpy(), tta.numpy(), record


def top16(probabilities):
    """The argmax as eval_uf and hs_seg take UF4's (tops['UF4'], labels): of the float16 cast, numpy, ties to the smallest label; uint8."""
    return np.asarray(probabilities).astype(np.float16).argmax(axis=0).astype(np.uint8)


def tops_v4(unflipped, flip_only, tta):
    return dict(top=top16(unflipped), top_flip=top16(flip_only), top_tta=top16(tta))


def infer_v4(subjects, inputs, *, files, device, lut, out_dir, log=print):
    """One subject at a time: v4_probabilities, the three maps written at once to <out_dir>/v4/<subject>.npz; no truth here. Returns ({subject: seconds record}, the members' records of the first subject).
    host_peak_rss_gib is the process's peak resident memory so far (resource.getrusage ru_maxrss: a running maximum over the subjects, not per subject)."""
    seconds, members = {}, None
    directory = Path(out_dir)/V4
    directory.mkdir(parents=True, exist_ok=True)
    cuda = torch.device(device).type == "cuda"
    for subject in subjects:
        plan, models, _ = inputs["plans"][subject]
        started = time.perf_counter()
        if cuda:
            torch.cuda.reset_peak_memory_stats(torch.device(device))
        with torch.no_grad():
            unflipped, flip_only, tta, record = v4_probabilities(subject, files=files, registry=inputs["registry"], models=models, plan=plan, device=device, lut=lut)
        maps = tops_v4(unflipped, flip_only, tta)
        del unflipped, flip_only, tta
        np.savez_compressed(directory/f"{subject}.npz", **maps)
        members = record["members"] if members is None else members
        record["members"] = [m["member_sha256"] for m in record["members"]]
        record.update(subject_s=time.perf_counter()-started, gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(torch.device(device))/GIB if cuda else None,
                      gpu_peak_reserved_gib=torch.cuda.max_memory_reserved(torch.device(device))/GIB if cuda else None, host_peak_rss_gib=common.peak_rss_gib())
        seconds[subject] = record
        log(f"{subject} v4: unflipped {record['unflipped_s']:.1f} s, flip {record['flip_s']:.1f} s (forward {record['flip_forward_s']:.1f} s), subject {record['subject_s']:.1f} s", flush=True)
    return seconds, members


def registered_agreement(subjects, *, out_dir, pred_root, expected_segmenter, registered=None):
    """No truth. Per subject, the unflipped argmax (v4/<subject>.npz top) against the registered prediction's: probs36.npz key "p", float16 (36, nz, ny, nx), its provenance and output hashes and raw sums checked by
    segprobe_common.load_raw, the argmax of the read-back (argmax_of_mean, = eval_seg.read_prediction's; ties to the smallest label). A prediction that cannot be read gets agreement None and the error: the control does
    not pass; nothing is stopped."""
    out = {}
    for subject in subjects:
        with np.load(Path(out_dir)/V4/f"{subject}.npz") as npz:
            mine = npz["top"]
        try:
            raw, record = common.load_raw(pred_root, subject, expected_segmenter=expected_segmenter)
            theirs = common.argmax_of_mean([raw])
            del raw
            if theirs.shape != mine.shape:
                raise ValueError(f"shape {theirs.shape}, the unflipped map's {mine.shape}")
        except (OSError, KeyError, ValueError, ProvenanceRejected) as exc:
            out[subject] = dict(agreement=None, error=f"{type(exc).__name__}: {exc}")
            continue
        differ = int((theirs != mine).sum())
        union = int(((theirs > 0) | (mine > 0)).sum())
        out[subject] = dict(agreement=(mine.size-differ)/mine.size, n_differ=differ, n_voxels=int(mine.size), n_foreground_union=union, agreement_foreground=(union-differ)/union if union else None,
                            probs36_sha256=record["probs36_sha256"], image_sha256=record["image_sha256"],
                            provenance_unchanged=None if registered is None else record["provenance_sha256"] == (registered.get(subject) or {}).get("provenance_sha256"))
    return out


def v4_control(table, agreement, subjects, reference):
    """plan.md, the first rule: the unflipped means of the 20 subjects within CONTROL_TOL of eval_uf.json means.UF4 (seg35 and seg4), and on every subject an agreement with the registered prediction of at least AGREEMENT_MIN.
    Both parts require the subjects to be exactly DEV_SUBJECTS, the 20 of the reference means (review m2). Reported beside the rule, not part of it (review m1): the largest count of voxels that differ from the
    registered argmax, the smallest agreement on the union of the two foregrounds, how many subjects have the registered argmax on every voxel, how many have the registered image."""
    means = table["means"]["unflipped"]
    difference = [means["seg35"]-reference[0], means["seg4"]-reference[1]]
    complete = len(subjects) == N_DEV and sorted(subjects) == sorted(DEV_SUBJECTS)
    mean_passed = bool(complete and all(abs(d) <= CONTROL_TOL for d in difference))
    records = {s: agreement.get(s) or {} for s in subjects}
    values = {s: r.get("agreement") for s, r in records.items()}
    agreement_passed = bool(complete and all(v is not None and v >= AGREEMENT_MIN for v in values.values()))
    defined = [v for v in values.values() if v is not None]
    differ = [r["n_differ"] for r in records.values() if r.get("n_differ") is not None]
    foreground = [r["agreement_foreground"] for r in records.values() if r.get("agreement_foreground") is not None]
    return dict(rule=f"|unflipped mean - eval_uf.json means.UF4| <= {CONTROL_TOL:g} for seg35 and seg4 over exactly the {N_DEV} development subjects 0438-0457, and the agreement of every subject's unflipped argmax with the registered prediction's >= {AGREEMENT_MIN} (plan.md)",
                committed=list(reference), recomputed=[means["seg35"], means["seg4"]], mean_difference=difference, tol=CONTROL_TOL, mean_passed=mean_passed, agreement=values, agreement_threshold=AGREEMENT_MIN,
                agreement_min=min(defined) if defined else None, n_below=sum(v is None or v < AGREEMENT_MIN for v in values.values()), agreement_passed=agreement_passed, n_subjects=len(subjects), subjects_are_dev=complete,
                passed=mean_passed and agreement_passed,
                reported=dict(n_differ_max=max(differ) if differ else None, agreement_foreground_min=min(foreground) if foreground else None, n_identical_to_registered=sum(d == 0 for d in differ),
                              n_image_same_as_registered=sum(r.get("image_same_as_registered") is True for r in records.values()), n_unreadable=sum(r.get("agreement") is None for r in records.values()),
                              note="reported, not part of the rule"))


def undecided_reasons(table, control):
    """Why the reading is 不判 (empty when it is not): the control failed; or the seg35 difference tta - unflipped is finite for other than N_DEV subjects (plan.md's rule is out of 20; review m2)."""
    reasons = [] if control["passed"] else ["控制不过"]
    n = table["tta_minus_unflipped"]["seg35"]["n"]
    if n != N_DEV:
        reasons.append(f"seg35 的有限差值 {n} 人，不是 {N_DEV} 人")
    return reasons


def v4_decision(table, control):
    """plan.md: 不判 when undecided_reasons gives any; then 过门 when the rule of N0 (g) holds (tta - unflipped seg35 mean >= +.002 and >= 15/20 positive), 不过门 otherwise."""
    if undecided_reasons(table, control):
        return UNDECIDED
    return PASS if table["worth_using"] else FAIL


def check_lock(path):
    """The first thing of a real v4 run: DevReadLocked if the read lock is there, whatever its content (the truth was read for this reading; the file is never deleted or edited). A file that cannot be read or
    is not JSON lines is refused the same way (review m3)."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        try:
            first = next((json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()), {})
            when = f"at {first.get('utc')} (out_dir {first.get('out_dir')})" if isinstance(first, dict) else "(its first line is not a record)"
        except (OSError, ValueError) as exc:
            when = f"(its content is not readable as JSON lines: {type(exc).__name__})"
        raise DevReadLocked(f"{path} exists: the development truth was read for the TTA reading {when}; it is read once, whatever --out-dir says; the file is not deleted")


def check_plan(path=None, *, pinned=None):
    """The sha256 of plan.md against PLAN_SHA256; PlanChanged otherwise (review m4). Returns the sha256."""
    path, pinned = V4_PLAN if path is None else Path(path), PLAN_SHA256 if pinned is None else pinned
    digest = sha_of(path)
    if digest != pinned:
        raise PlanChanged(f"{path}: sha256 {digest}, the preregistration is {pinned}; not run, nothing read")
    return digest


def take_lock(path, record):
    """The moment before the first truth label: the lock created with O_EXCL (a second run loses here even if it passed check_lock). Returns the line written."""
    path = Path(path)
    line = dict(record, event=FIRST_READ, utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), pid=os.getpid())
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise DevReadLocked(f"{path} appeared just before the first truth label; not read") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(line, ensure_ascii=False)+"\n")
    return line


def locked_reader(read_truth, take):
    """read_truth(subject, kind) that calls take() before its first read; the line taken in state['line']."""
    state = {}

    def read(subject, kind):
        if "line" not in state:
            state["line"] = take()
        return read_truth(subject, kind)
    return read, state


def summary_markdown_v4(document):
    f = eval_seg.fmt
    table, control, reported = document["table"], document["control"], document["control"]["reported"]
    lines = ["# TTA：UF4（unet35_v4_ens）加左右翻转 TTA，分割级读数（开发集 20 人，单次观测；不是配准结果；不计入 3 次选择读数）", "", f"**判定：{document['decision']}**（plan.md：先看控制，再看 N0 (g) 的规则 {table['rule']}）", ""]
    if document["decision"] == UNDECIDED:
        lines += [f"**{document['interpretation']}**", ""]
    lines += [f"控制：{'过' if control['passed'] else '不过'}（受试者恰为开发集 20 人：{'是' if control['subjects_are_dev'] else '否'}）。unflipped 均值 seg35 / seg4 {f(control['recomputed'][0], 6)} / {f(control['recomputed'][1], 6)}，"
              f"已提交 {f(control['committed'][0], 6)} / {f(control['committed'][1], 6)}，差 {f(control['mean_difference'][0], 6)} / {f(control['mean_difference'][1], 6)}（限 {control['tol']:g}）；"
              f"与已登记预测的体素一致率最小 {f(control['agreement_min'], 6)}（限 {AGREEMENT_MIN}，低于的 {control['n_below']} 人）。", "",
              f"只报告、不进规则：与登记 argmax 不同的体素数最大 {reported['n_differ_max']}；前景（两者前景之并）一致率最小 {f(reported['agreement_foreground_min'], 6)}；"
              f"argmax 与登记逐体素相同 {reported['n_identical_to_registered']} 人；图像与登记相同 {reported['n_image_same_as_registered']} 人；读不出的登记预测 {reported['n_unreadable']} 人。", "",
              "| 臂 | "+" | ".join(METRICS_V4)+" |", "|---|"+"---|"*len(METRICS_V4)]
    lines += [f"| {arm} | "+" | ".join(f(table["means"][arm][m]) for m in METRICS_V4)+" |" for arm in ARMS]
    lines += ["", "| tta − unflipped | 均值差 | 中位差 | 正 | 负 | 零 |", "|---|---|---|---|---|---|"]
    lines += [f"| {m} | {f(p['mean'], 5)} | {f(p['median'], 5)} | {p['n_positive']} | {p['n_negative']} | {p['n_zero']} |" for m, p in table["tta_minus_unflipped"].items()]
    forward = [r["forward_s"] for r in document["seconds"].values()]
    pure = [r["flip_forward_s"] for r in document["seconds"].values()]
    peak, rss = document["gpu_peak_allocated_gib"], document["host_peak_rss_gib"]
    lines += ["", f"每人推断秒数（两个方向，四个成员，含载入成员与读图，不是纯前向）：中位 {np.median(forward):.1f}，最大 {np.max(forward):.1f}；其中镜像那次的纯前向中位 {np.median(pure):.1f}；"
              f"GPU 峰值（已分配）{'—' if peak is None else f'{peak:.2f} GiB'}；主机常驻内存峰值 {rss:.2f} GiB。", ""]
    return "\n".join(lines)


def run_v4(*, out_dir, device, requested=None, gpu_cap_gib=GPU_CAP_GIB, min_free_gib=MIN_FREE_GIB, data_root=eval_seg.DATA_ROOT, argv=None, log=print, lock_path=TTA_DEV_READ_LOCK, spec=None, require_clean=True):
    """The reading of plan.md on the 20 development subjects. Order: the lock (first), the preregistration's sha256, the output directory, the admission, clean code, every check of v4_inputs (no image yet), the GPU,
    the inference (no truth), the agreement with the registered predictions (no truth), the lock taken, the truth, the tables, the control, the decision. Nothing is stopped by a control that fails; nothing is rerun.
    A 不判 has worth_using null and the interpretation 不解读 in the document and in summary.md (plan.md: 不判，不解释)."""
    spec = SEGMENTERS[V4] if spec is None else spec
    check_lock(lock_path)
    plan_sha = check_plan()
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} exists and is not empty; results are never overwritten")
    admitted, admission = common.admit_dev_subjects(data_root=data_root)          # before any file of a subject is opened
    if require_clean:
        require_clean_code(admission["code"])
    if requested and sorted(requested) != sorted(admitted):
        raise ValueError("v4: the reading is of every admitted development subject (plan.md: 20); --subjects is not for it")
    subjects = list(admitted)
    allowlist = load_dev_allowlist(eval_seg.ALLOWLIST)
    lut = swap_table()
    inputs = v4_inputs(subjects, allowlist=allowlist, spec=spec, lut=lut)
    gpu = None
    if torch.device(device).type == "cuda":
        gpu = require_free_gpu(device, min_free_gib)
        cap_gpu_memory(torch.device(device), gpu_cap_gib)
    cudnn = dict(benchmark=bool(torch.backends.cudnn.benchmark), deterministic=bool(torch.backends.cudnn.deterministic))     # recorded; HS's seg phase ran with both false (the registered provenance's environment, in `registered`)
    environment = dict({k: v for k, v in environment_record(torch.device(device)).items() if k != "code"}, cuda_build=torch.version.cuda)      # torch, cuDNN, CUDA, the GPU's name on cuda (review m7); the code record is in admission
    seconds, members = infer_v4(subjects, inputs, files=OasisFiles(root=data_root), device=torch.device(device), lut=lut, out_dir=out_dir, log=log)     # a torch.device, as hs_seg.infer_all hands it on
    agreement = registered_agreement(subjects, out_dir=out_dir, pred_root=spec["pred_root"], expected_segmenter=spec["id"], registered=inputs["registered"])
    for subject in subjects:
        agreement[subject]["image_same_as_registered"] = seconds[subject]["image_sha256"] == inputs["registered"][subject]["image_sha256"]
    lock_record = dict(reading=str(V4_PLAN.relative_to(REPO)), plan_sha256=plan_sha, segmenter_id=spec["id"], registry_sha256=inputs["registry"].sha256, member_root=str(spec["models_root"]), members=inputs["members"],
                       out_dir=str(out_dir), command=argv, code_commit=(admission.get("code") or {}).get("commit"))
    reader, lock = locked_reader(lambda s, k: eval_seg.read_truth(data_root, s, k), lambda: take_lock(lock_path, lock_record))
    rows, truth_sha = evaluate_tops(subjects, [V4], out_dir=out_dir, read_truth=reader, lookup=eval_seg.load_seg4_lookup())
    table = tabulate(rows, subjects, {V4: spec["reference"]}, metrics=METRICS_V4)[V4]
    table["tta_minus_unflipped_per_subject"] = {m: {s: value(rows[V4][s]["tta"], m)-value(rows[V4][s]["unflipped"], m) for s in subjects} for m in METRICS_V4}
    control = v4_control(table, agreement, subjects, spec["reference"])
    table["control"] = control
    reasons = undecided_reasons(table, control)
    decision = v4_decision(table, control)
    interpretation = None
    if decision == UNDECIDED:
        table["worth_using"] = None                  # not True or False: a 不判 is not read (review m2)
        interpretation = f"不判（{'；'.join(reasons)}）：以下数字只作记录，{NOT_INTERPRETED}。"
    peaks = [r["gpu_peak_allocated_gib"] for r in seconds.values() if r["gpu_peak_allocated_gib"] is not None]
    document = dict(format=FORMAT_V4, created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), command=argv, plan=dict(path=str(V4_PLAN.relative_to(REPO)), sha256=plan_sha, pinned=PLAN_SHA256), decision=decision,
                    undecided_reasons=reasons, interpretation=interpretation, control=control, admission=admission, subjects=subjects, device=str(device), gpu=gpu, gpu_cap_gib=gpu_cap_gib, cudnn=cudnn, environment=environment,
                    gpu_peak_allocated_gib=max(peaks) if peaks else None, host_peak_rss_gib=common.peak_rss_gib(),
                    seconds_note="v4: unflipped_s and flip_s are whole passes (image read, members re-hashed and loaded, forward passes; flip_s also the mean and flip_back), forward_s = their sum, not v1/v2's pure forward time; flip_forward_s is pure forward time of the mirror; host_peak_rss_gib a running maximum",
                    segmenter=dict(name=V4, id=spec["id"], candidate=spec["candidate"], registry=str(spec["registry"]), registry_sha256=inputs["registry"].sha256, models_root=str(spec["models_root"]), pred_root=str(spec["pred_root"]), reference=list(spec["reference"])),
                    members=members, member_order=inputs["order"], flip_premise=inputs["premise"], flip=dict(axis=FLIP_DIM, lut=lut.tolist()), registered=inputs["registered"], agreement=agreement,
                    inference="unflipped = hs_seg.uf4_probabilities (the registered inference of UF4); flip_only = the same members with the same parameters on the mirror image, ensemble_probabilities, flip_back; tta = (unflipped + flip_only)/2 renormalized; every argmax of the float16 cast, ties to the smallest label",
                    code_files={rel: sha_of(REPO/rel) for rel in CODE_FILES}, seconds=seconds, table=table, truth_sha256=truth_sha, dev_read_lock=lock.get("line"), rows=rows,
                    note="开发集 20 人，单次观测；分割级读数，不是配准结果；不计入 3 次选择读数（plan.md）。这 20 人已被反复看过，“只读一次”只对本次 TTA 成立。")
    common.write_json(out_dir/"flip_tta.json", document)
    with (out_dir/"per_subject.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["segmenter", "subject", "arm", *METRICS_V4, "forward_s", "agreement"])
        for subject in subjects:
            for arm in ARMS:
                writer.writerow([V4, subject, arm]+[value(rows[V4][subject][arm], m) for m in METRICS_V4]+[seconds[subject]["forward_s"], agreement[subject]["agreement"] if arm == "unflipped" else ""])
    (out_dir/"summary.md").write_text(summary_markdown_v4(document), encoding="utf-8")
    return document


def dry_run_v4(subjects, *, allowlist, spec=None, lock_path=TTA_DEV_READ_LOCK, log=print):
    """No GPU, no image of a subject, no truth, no lock written. Every check of v4_inputs on the real registry, members and registered predictions; then the real members re-hashed and loaded on the CPU and
    v4_probabilities on a synthetic 32^3 volume written to a temporary directory (the plan of the first subject; each member twice). Returns the record; AssertionError, ValueError or FlipPremiseRefused if something is wrong."""
    from bayesreg.seg.unet.synthetic import write_subjects
    spec = SEGMENTERS[V4] if spec is None else spec
    lut = swap_table()
    inputs = v4_inputs(subjects, allowlist=allowlist, spec=spec, lut=lut)
    log(f"v4: registry {spec['registry']} sha256 {inputs['registry'].sha256}; {len(subjects)} subjects, HS's plan {inputs['order']}; flip premise: "
        + ", ".join(f"{n} p={r['flip_probability']} axis_check={r['flip_axis_check']} flips={r['flips_logged']}" for n, r in inputs["premise"].items()) + f"; {len(inputs['registered'])} registered predictions of this registry")
    plan, models, _ = inputs["plans"][subjects[0]]
    with tempfile.TemporaryDirectory(prefix="tta_v4_dry_") as tmp:
        write_subjects(tmp, [SYNTHETIC_SUBJECT], shape_xyz=(32, 32, 32))
        with torch.no_grad():
            unflipped, flip_only, tta, record = v4_probabilities(SYNTHETIC_SUBJECT, files=OasisFiles(root=Path(tmp)), registry=inputs["registry"], models=models, plan=plan, device="cpu", lut=lut)
    for array in (unflipped, flip_only, tta):
        assert array.shape == (36, 32, 32, 32) and array.dtype == np.float32, array.shape
        assert float(np.max(np.abs(array.sum(axis=0, dtype=np.float64)-1))) < 1e-5
    assert record["n_forward"] == 2*len(plan), record["n_forward"]
    maps = tops_v4(unflipped, flip_only, tta)
    members = [{k: m[k] for k in ("model_id", "member_sha256", "temperature", "amp", "window", "window_gaussian_sigma", "percentile", "checkpoint_choice", "checkpoint_iteration")} for m in record["members"]]
    log(f"synthetic volume (32, 32, 32), the real members on the CPU: unflipped {record['unflipped_s']:.2f} s, flip {record['flip_s']:.2f} s, {record['n_forward']} forward passes; the channel sums are 1; "
        f"labels per arm {[int(np.unique(v).size) for v in maps.values()]}; members {[(m['member_sha256'][:12], round(m['temperature'], 4), m['window'], m['window_gaussian_sigma'], m['amp']) for m in members]}")
    lock_exists = Path(lock_path).exists()
    plan_sha = sha_of(V4_PLAN)
    log(f"read lock {lock_path}: {'exists (a real run would be refused)' if lock_exists else 'absent'}; plan {V4_PLAN.relative_to(REPO)} sha256 {plan_sha}: "
        + ("the pinned preregistration" if plan_sha == PLAN_SHA256 else f"NOT the pinned {PLAN_SHA256} (a real run would be refused)"))
    return dict(subjects=subjects, registry_sha256=inputs["registry"].sha256, order=inputs["order"], premise=inputs["premise"], registered=inputs["registered"], members=members, lock_exists=lock_exists,
                synthetic_seconds=dict(unflipped=record["unflipped_s"], flip=record["flip_s"]), plan_sha256=plan_sha, plan_pinned=plan_sha == PLAN_SHA256)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--segmenters", nargs="+", choices=sorted(SEGMENTERS), default=["v1", "v2"])
    parser.add_argument("--subjects", nargs="*", default=None, help="default: all 20 development subjects")
    parser.add_argument("--out-dir", type=Path, default=None, help=f"default: {common.OUT_ROOT/'p2'} for v1 and v2, {V4_OUT} for v4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-cap-gib", type=float, default=GPU_CAP_GIB)
    parser.add_argument("--min-free-gib", type=float, default=MIN_FREE_GIB)
    parser.add_argument("--real-run", action="store_true", help="v4 only: the one real reading (takes the read lock); without it and without --dry-run, v4 is refused before anything is read (review m8)")
    args = parser.parse_args(argv)
    if V4 in args.segmenters:
        if args.segmenters != [V4]:
            parser.error("v4 runs alone (--segmenters v4)")
        if args.dry_run == args.real_run:
            parser.error("v4: exactly one of --dry-run and --real-run (the real run takes the one read lock of the development truth)")
        if args.dry_run:
            admitted, _ = common.admit_dev_subjects()
            record = dry_run_v4(select_subjects(admitted, args.subjects), allowlist=load_dev_allowlist(eval_seg.ALLOWLIST))
            print(json.dumps(dict(dry_run="passed", segmenters=[V4], subjects=len(record["subjects"]), registry_sha256=record["registry_sha256"], lock_exists=record["lock_exists"], plan_pinned=record["plan_pinned"],
                                  synthetic_seconds=record["synthetic_seconds"])))
            return 0
        document = run_v4(out_dir=V4_OUT if args.out_dir is None else args.out_dir, device=args.device, requested=args.subjects, gpu_cap_gib=args.gpu_cap_gib, min_free_gib=args.min_free_gib, argv=sys.argv if argv is None else ["flip_tta.py", *argv])
        table, control = document["table"], document["control"]
        print(json.dumps(dict(decision=document["decision"], interpretation=document["interpretation"], means=table["means"], gain=table["tta_minus_unflipped"]["seg35"], worth_using=table["worth_using"],
                              control=dict(((k, control[k]) for k in ("passed", "subjects_are_dev", "mean_difference", "agreement_min", "n_below")), **{k: control["reported"][k] for k in ("n_differ_max", "agreement_foreground_min", "n_identical_to_registered")})),
                         indent=1, ensure_ascii=False))
        return 0
    if args.real_run:
        parser.error("--real-run is for v4 only")
    if args.out_dir is None:
        args.out_dir = common.OUT_ROOT/"p2"
    if args.dry_run:
        admitted, _ = common.admit_dev_subjects()
        record = dry_run(args.segmenters, select_subjects(admitted, args.subjects), allowlist=load_dev_allowlist(eval_seg.ALLOWLIST))
        print(json.dumps(dict(dry_run="passed", subjects=len(record["subjects"]), synthetic_seconds=record["synthetic_seconds"])))
        return 0
    document = run(names=args.segmenters, out_dir=args.out_dir, device=args.device, requested=args.subjects, gpu_cap_gib=args.gpu_cap_gib, min_free_gib=args.min_free_gib, argv=sys.argv if argv is None else ["flip_tta.py", *argv])
    print(json.dumps({n: dict(means=t["means"]["tta"], gain=t["tta_minus_unflipped"]["seg35"], worth_using=t["worth_using"], control=t["control"]) for n, t in document["tables"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
