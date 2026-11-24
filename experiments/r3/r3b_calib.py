"""R3b: the calibration diagnosis of the v4 ensemble and the decoding criterion on the 16 out-of-fold calibration subjects of the training set
(runs/20261111-r3/calibration_plan.md; task brief runs/20261111-r3/任务书.md, R3b).

    python -B experiments/r3/r3b_calib.py dry-run                       nothing real is read but the plan and the code; a synthetic run end to end
    python -B experiments/r3/r3b_calib.py rehearse                      CPU, before infer (plan revision 2 item 4): infer's inputs under its spy; no model, image or label
    python -B experiments/r3/r3b_calib.py infer --device cuda           unit tag-r3b-infer (GPU, machine gate first): the two same-fold members on each calibration subject; no truth
    python -B experiments/r3/r3b_calib.py decode                        unit tag-r3b-decode (CPU): argmax, the decoding, the ratio description; no truth
    python -B experiments/r3/r3b_calib.py eval --opened-files-sha256 X  unit tag-r3b-eval (CPU): the read lock, the 16 truths once, every quantity of plan section 5
    python -B experiments/r3/r3b_calib.py decide                        the criterion of plan section 6 -> analysis/decision_calib.json (read by R3a, r3b_decision.py)

Each step runs once and never overwrites; a failed step is not rerun by the tool (hard constraint 4).  infer, decode and eval run under PQ's spy with the
rules of r3b_firewall (plan section 4); opened_files_*.json are described, not judged.  Real runs need a clean checkout with the data/ link, the pinned plan
(r3b_common.PLAN_SHA256) and the pinned inputs; `--synthetic WORK` runs everything on a synthetic tree (r3b_synthetic.build) and never touches the real roots.
"""
import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for extra in (HERE, ROOT/"experiments"/"pq", ROOT/"experiments"/"unv2", ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import r3b_common as rc  # noqa: E402
import r3b_firewall as fw  # noqa: E402
from bayesreg.seg import decode as D  # noqa: E402

PY = "/home/rmcfz/miniconda3/envs/sci310/bin/python"
THREADS = "--setenv=OPENBLAS_NUM_THREADS=1 --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 --setenv=PYTHONDONTWRITEBYTECODE=1"


class Refused(Exception):
    """A check before or during a step failed; the step stops (exit 8) and writes no result."""


@dataclass
class Ctx:
    repo: Path = ROOT
    out: Path = rc.OUT
    results: Path = rc.RESULTS
    images: Path = rc.DATA_ROOT
    data_link: Path = ROOT/"data"
    models: Path = rc.UF_MODELS
    registry: Path = ROOT/rc.REGISTRY_REL
    folds: Path = ROOT/rc.FOLDS_REL
    allowlist: Path = ROOT/rc.ALLOWLIST_REL
    seg4_map: Path = ROOT/rc.SEG4_MAP_REL
    lock: Path = rc.READ_LOCK
    calib: dict = field(default_factory=lambda: {f: tuple(v) for f, v in rc.CALIB_SUBJECTS.items()})
    member_sha: dict = field(default_factory=lambda: {f: tuple(v) for f, v in rc.MEMBER_SHA256.items()})
    criterion: dict = field(default_factory=lambda: dict(rc.CRITERION))
    synthetic: bool = False
    training_source: str = "bayesreg_train_v1"
    expect_counts: tuple = (98, 8)
    source_root: Path = ROOT

    def subjects(self):
        return [s for f in rc.FOLDS for s in self.calib[f]]

    def fold_of(self, subject):
        return next(f for f in rc.FOLDS if subject in self.calib[f])


def real_ctx(repo=ROOT):
    repo = Path(repo)
    return Ctx(repo=repo, data_link=repo/"data", registry=repo/rc.REGISTRY_REL, folds=repo/rc.FOLDS_REL, allowlist=repo/rc.ALLOWLIST_REL,
               seg4_map=repo/rc.SEG4_MAP_REL, source_root=repo)


# ---- shared pieces ------------------------------------------------------------------------------------------------------------------------------
def _code(ctx):
    from bayesreg.seg.unet.snapshot import code_record, require_clean_code
    code = code_record()
    if not ctx.synthetic:
        require_clean_code(code)
        if not (ctx.repo/"data").exists():
            raise Refused(f"{ctx.repo}/data missing: the clean checkout needs the data link (ln -s {ROOT/'data'} or seq_vp_3d/data)")
    return code


def _plan(ctx):
    if ctx.synthetic:
        path = ctx.repo/rc.PLAN_REL
        return dict(path=str(path), sha256=rc.sha256_file(path), pinned=False)
    path, sha = rc.plan_check(ctx.repo)
    return dict(path=str(path), sha256=sha, pinned=True)


def _inputs(ctx):
    """(split, allowlist, record): the pinned loaders on real data, the synthetic ones otherwise."""
    from bayesreg.seg.unet import pins
    from bayesreg.seg.unet.data import load_full_folds
    if ctx.synthetic:
        return pins.load_synthetic_inputs(ctx.folds, ctx.allowlist, data_root=ctx.images, training_source=ctx.training_source,
                                          expect_counts=ctx.expect_counts, loader=load_full_folds)
    return pins.load_pinned_full_inputs(ctx.folds, ctx.allowlist, source_root=ctx.source_root)


def _source_files(ctx):
    """The allowlist's source files (read by the pinned loader to re-hash them); named before the spy from the allowlist (a config file)."""
    document = json.loads(Path(ctx.allowlist).read_text(encoding="utf-8"))
    return [Path(ctx.source_root)/s["path"] for s in document.get("sources", []) if "path" in s]


def _member_dirs(ctx):
    from bayesreg.seg.unet.train import member_dir
    return {(f, s): member_dir(ctx.models, f, s) for f in rc.FOLDS for s in rc.SEEDS}


def _spy(rules):
    import pq_firewall
    from bayesreg.io.sealed import sealed_cache_is_empty
    if not sealed_cache_is_empty():
        raise Refused("the guard's cache of the sealed lists is not empty: their first read must happen under the spy")
    return pq_firewall.Spy(rules)


def _write_opened(path, spy):
    record = spy.record()
    rc.write_json_new(path, record)
    return record


def check_members(ctx, split, registry):
    """Plan section 2: for each fold, the registry entry of SEGMENTER_ID with the pinned member sha256; each member's member.json, train_record.json (hashed),
    fold, seed, the fold's calibration subjects = the split's and none of them in the member's training subjects.  Returns {(fold, seed): dict}."""
    from bayesreg.seg.unet.data import FOLD_INDEX
    if not ctx.synthetic and registry.sha256 != rc.REGISTRY_SHA256:
        raise Refused(f"registry sha256 {registry.sha256} is not the pinned {rc.REGISTRY_SHA256}")
    out = {}
    for fold in rc.FOLDS:
        entries = [(m, e) for m, e in registry.models.items() if e["segmenter_id"] == rc.SEGMENTER_ID and e["fold"] == FOLD_INDEX[fold]]
        if len(entries) != 1:
            raise Refused(f"fold {fold}: {len(entries)} registry entries of {rc.SEGMENTER_ID}")
        model_id, entry = entries[0]
        if set(entry["member_model_sha256"]) != set(ctx.member_sha[fold]):
            raise Refused(f"fold {fold}: the registry's members are not the plan's")
        calib = split.subjects(fold, "calib")
        if tuple(calib) != tuple(sorted(ctx.calib[fold])):
            raise Refused(f"fold {fold}: the folds file's calibration subjects {calib} are not the plan's {sorted(ctx.calib[fold])}")
        for seed, directory in ((s, d) for (f, s), d in _member_dirs(ctx).items() if f == fold):
            member = json.loads((directory/"member.json").read_text(encoding="utf-8"))
            sha = rc.sha256_file(directory/"member.pt")
            if sha != member.get("member_sha256") or sha not in entry["member_model_sha256"]:
                raise Refused(f"{directory}: member.pt sha256 {sha[:12]}... is not member.json's or not a registered member of {model_id}")
            if rc.sha256_file(directory/"train_record.json") != member.get("train_record_sha256"):
                raise Refused(f"{directory}: train_record.json differs from member.json's train_record_sha256")
            record = json.loads((directory/"train_record.json").read_text(encoding="utf-8"))
            if record.get("fold") != fold or record.get("seed") != seed or member.get("fold") != fold or member.get("seed") != seed:
                raise Refused(f"{directory}: fold/seed are not ({fold}, {seed})")
            if sorted(record.get("calib_subjects", [])) != sorted(calib):
                raise Refused(f"{directory}: the member's calibration subjects are not the fold's")
            inside = sorted(set(record.get("train_subjects", [])) & set(calib))
            if inside or not record.get("train_subjects"):
                raise Refused(f"{directory}: calibration subjects in the member's training subjects {inside} (not out of fold)")
            out[(fold, seed)] = dict(directory=directory, sha256=sha, model_id=model_id, entry=entry, temperature=member.get("temperature"),
                                     train_record_sha256=member["train_record_sha256"], n_train=len(record["train_subjects"]))
        seeds = [out[(fold, s)]["sha256"] for s in rc.SEEDS]
        if len(set(seeds)) != len(seeds) or set(seeds) != set(entry["member_model_sha256"]):      # review m6
            raise Refused(f"fold {fold}: the seeds' member files are not distinct, or together not the registry entry's members")
    return out


# ---- the machine gate of infer (plan revision 2 item 3; review m4) ------------------------------------------------------------------------------
INFER_UNIT = "tag-r3b-infer"


def gate_problems(own_unit=INFER_UNIT, *, disk=None):
    """R2's gate (experiments/r2/run_r2.gate_problems: experiments/ul/run_ul.gate_problems plus the C: floor), with R3b's unit and floor.  Empty: infer may
    start.  A query that fails counts as closed."""
    import shutil
    ul = ROOT/"experiments"/"ul"
    if str(ul) not in sys.path:
        sys.path.insert(0, str(ul))
    import run_ul
    problems = list(run_ul.gate_problems(own_unit))
    try:
        free = (disk or shutil.disk_usage)("/mnt/c").free/2**30
        if free < rc.C_FLOOR_GIB:
            problems.append(f"/mnt/c free {free:.1f} GiB < {rc.C_FLOOR_GIB:.0f} GiB")
    except OSError as exc:
        problems.append(f"C: query failed: {exc}")
    return problems


# ---- the rehearsal of the real inputs (plan revision 2 item 4; review m5) ------------------------------------------------------------------------
def rehearse(ctx, *, log=print):
    """Under infer's rules and spy: the pinned inputs, the registry and check_members only (member.json, train_record.json, member.pt hashed); no model loaded,
    no image or label read.  Writes <REHEARSAL_ROOT>/<UTC>/opened_files.json and rehearsal.json; OUT is not touched.  Returns the record."""
    from bayesreg.io.pair import load_segmenter_registry
    from bayesreg.seg.unet.data import IMAGE_FILE
    root = Path(ctx.results)/rc.REHEARSAL_ROOT.name/time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if root.exists():
        raise FileExistsError(f"{root} exists; never overwritten")
    code = _code(ctx)
    plan = _plan(ctx)
    subjects = ctx.subjects()
    files = [ctx.registry, ctx.folds, Path(ctx.folds).parent/"folds.json", ctx.allowlist, *_source_files(ctx)]
    files += [d/n for d in _member_dirs(ctx).values() for n in ("member.pt", "member.json", "train_record.json")]
    files += [Path(ctx.images)/s/IMAGE_FILE for s in subjects]                 # infer's list, unchanged: the rehearsal reads no image all the same
    rules = fw.CalibRules(unit="infer", lock=ctx.lock, repo=ctx.repo, results=ctx.results, data=ctx.data_link, images=ctx.images, out=Path(ctx.out), files=files,
                          own_dirs=[Path(ctx.out)/rc.PROBS_DIR])
    spy = _spy(rules)
    error = None
    try:
        with spy:
            split, allowlist, inputs = _inputs(ctx)
            registry = load_segmenter_registry(ctx.registry)
            members = check_members(ctx, split, registry)
    except Exception as exc:                                                    # recorded, then raised again: the rehearsal's job is to find these
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        root.mkdir(parents=True)
        opened = _write_opened(root/"opened_files.json", spy)
        rc.write_json_new(root/"rehearsal.json", dict(format="bayesreg.r3b.rehearsal.v1", plan=plan, code=code, subjects=subjects, error=error, m6=opened["m6"],
                                                      members=None if error else {f"{f}{s}": m["sha256"] for (f, s), m in members.items()},
                                                      note="infer's inputs under infer's rules; no model, image or label; OUT not touched"))
    log(f"rehearsal: m6 {opened['m6']}, {opened['n_files']} files, records in {root}")
    return dict(root=str(root), m6=opened["m6"], n_files=opened["n_files"])


# ---- step 1: infer ------------------------------------------------------------------------------------------------------------------------------
def infer(ctx, *, device="cuda", gate=None, log=print):
    import torch
    from bayesreg.io.pair import load_segmenter_registry
    from bayesreg.seg.unet.data import IMAGE_FILE, OasisFiles, normalize_image, read_image
    from bayesreg.seg.unet.infer import check_subject_admitted, ensemble_probabilities, load_member
    from bayesreg.seg.unet.large import cap_gpu_memory
    from bayesreg.seg.unet.train import environment_record, predict_logits
    out, probs_root = Path(ctx.out), Path(ctx.out)/rc.PROBS_DIR
    if gate is not None or not ctx.synthetic:                                   # the machine gate before anything is read (no wait: the lead restarts)
        problems = (gate or gate_problems)()
        if problems:
            raise Refused(f"machine gate closed, nothing read: {problems}")
    if (out/rc.INFER_RECORD).exists() or (probs_root.exists() and any(probs_root.iterdir())) or (out/rc.OPENED_INFER).exists():
        raise FileExistsError(f"{out}: infer ran already; never overwritten")
    code = _code(ctx)
    plan = _plan(ctx)
    subjects = ctx.subjects()
    files = [ctx.registry, ctx.folds, Path(ctx.folds).parent/"folds.json", ctx.allowlist, *_source_files(ctx)]
    files += [d/n for d in _member_dirs(ctx).values() for n in ("member.pt", "member.json", "train_record.json")]
    files += [Path(ctx.images)/s/IMAGE_FILE for s in subjects]
    rules = fw.CalibRules(unit="infer", lock=ctx.lock, repo=ctx.repo, results=ctx.results, data=ctx.data_link, images=ctx.images, out=out, files=files, own_dirs=[probs_root])
    spy = _spy(rules)
    started = time.perf_counter()
    rows = {}
    try:
        with spy:
            split, allowlist, inputs = _inputs(ctx)
            registry = load_segmenter_registry(ctx.registry)
            members = check_members(ctx, split, registry)
            device = torch.device(device)
            if device.type == "cuda":
                cap_gpu_memory(device, rc.GPU_CAP_GIB)
            images = OasisFiles(Path(ctx.images))
            for fold in rc.FOLDS:
                loaded = []
                for seed in rc.SEEDS:
                    m = members[(fold, seed)]
                    model, blob = load_member(m["directory"]/"member.pt", m["sha256"], device, entry=m["entry"])
                    if m["temperature"] is None or float(blob["temperature"]) != float(m["temperature"]):          # review m6
                        raise Refused(f"{m['directory']}: member.json temperature {m['temperature']} != member.pt's {blob['temperature']}")
                    loaded.append((seed, model, blob))
                percentiles = {blob["percentile"] for _, _, blob in loaded}
                if len(percentiles) != 1:
                    raise Refused(f"fold {fold}: the members disagree on the normalization percentile {percentiles}")
                percentile = percentiles.pop()
                for subject in ctx.calib[fold]:
                    t0 = time.perf_counter()
                    check_subject_admitted(subject, allowlist)             # the inference guard, before the image is read
                    values, meta = read_image(images, subject)
                    image, norm = normalize_image(values, percentile=percentile)

                    def logits():
                        for _, model, blob in loaded:
                            window = tuple(blob["window"]) if blob.get("window") else None
                            yield (predict_logits(model, image, device=device, amp=blob["amp"], window=window, gaussian_sigma=blob.get("window_gaussian_sigma")),
                                   blob["temperature"])
                    probs = ensemble_probabilities(logits()).cpu().numpy().astype(np.float16)     # as infer.segment_subject stores probs36
                    directory = probs_root/subject
                    rc.save_npz_new(directory/"probs36.npz", p=probs)
                    rc.save_npz_new(directory/"brain.npz", brain=np.asarray(values > 0))
                    row = dict(subject=subject, fold=fold, members=[dict(seed=s, member_sha256=members[(fold, s)]["sha256"], temperature=float(b["temperature"]),
                                                                         model_id=members[(fold, s)]["model_id"]) for s, _, b in loaded],
                               out_of_fold=dict(subject_in_member_calib=True, subject_in_member_train=False,
                                                note="temperature fitted on this fold's calibration subjects (calibration_plan.md section 2)"),
                               image_sha256=meta["sha256"], normalization=norm, grid=dict(shape_zyx=meta["grid_shape_zyx"], spacing_xyz=meta["spacing_xyz"],
                                                                                          origin_xyz=meta["origin_xyz"]),
                               probs36_sha256=rc.sha256_file(directory/"probs36.npz"), brain_sha256=rc.sha256_file(directory/"brain.npz"),
                               shape=list(probs.shape), dtype="float16", seconds=time.perf_counter()-t0)
                    rc.write_json_new(directory/"record.json", row)
                    rows[subject] = row
                    log(f"{subject} (fold {fold}): two members, {row['seconds']:.1f} s")
                    del probs, values, image
                del loaded
    finally:
        opened = _write_opened(out/rc.OPENED_INFER, spy)
    record = dict(format="bayesreg.r3b.infer.v1", plan=plan, code=code, environment=environment_record(device), inputs={k: v for k, v in inputs.items() if k != "code"},
                  registry=dict(path=str(ctx.registry), sha256=registry.sha256), subjects=subjects, rows=rows,
                  members={f"{f}{s}": dict(sha256=m["sha256"], model_id=m["model_id"], temperature=m["temperature"], train_record_sha256=m["train_record_sha256"],
                                           n_train=m["n_train"]) for (f, s), m in members.items()},
                  opened_files_sha256=rc.sha256_file(out/rc.OPENED_INFER), m6=opened["m6"], seconds=time.perf_counter()-started, synthetic=ctx.synthetic)
    rc.write_json_new(out/rc.INFER_RECORD, record)
    return record


# ---- step 2: decode -----------------------------------------------------------------------------------------------------------------------------
def _read_probs(ctx, subject, row):
    from bayesreg.seg.channels import read_back_probabilities
    path = Path(ctx.out)/rc.PROBS_DIR/subject/"probs36.npz"
    if rc.sha256_file(path) != row["probs36_sha256"]:
        raise Refused(f"{path}: sha256 differs from the infer record")
    probs, read_back = read_back_probabilities(path, shape=tuple(row["shape"]))
    return probs, read_back


def decode_step(ctx, *, log=print):
    out = Path(ctx.out)
    decoded = out/rc.DECODED_DIR
    if (out/rc.DECODE_RECORD).exists() or (decoded.exists() and any(decoded.iterdir())) or (out/rc.OPENED_DECODE).exists():
        raise FileExistsError(f"{out}: decode ran already; never overwritten")
    code = _code(ctx)
    plan = _plan(ctx)
    subjects = ctx.subjects()
    implementation = rc.implementation_record()                             # plan revision 2 item 1 (review I1): the decoder's file this unit runs
    files = [out/rc.INFER_RECORD]+[out/rc.PROBS_DIR/s/n for s in subjects for n in ("probs36.npz", "record.json")]
    rules = fw.CalibRules(unit="decode", lock=ctx.lock, repo=ctx.repo, results=ctx.results, data=ctx.data_link, images=ctx.images, out=out, files=files, own_dirs=[decoded])
    spy = _spy(rules)
    rows, started = {}, time.perf_counter()
    try:
        with spy:
            infer_record = json.loads((out/rc.INFER_RECORD).read_text(encoding="utf-8"))
            if infer_record["subjects"] != subjects or infer_record["plan"]["sha256"] != plan["sha256"] or infer_record.get("m6") is not True:
                raise Refused("the infer record is not this plan's run over these subjects, or its spy recorded a breach")
            for subject in subjects:
                t0 = time.perf_counter()
                row = json.loads((out/rc.PROBS_DIR/subject/"record.json").read_text(encoding="utf-8"))
                if row != infer_record["rows"][subject]:
                    raise Refused(f"{subject}: record.json differs from the infer record")
                probs, read_back = _read_probs(ctx, subject, row)
                argmax = D.argmax_labels(probs)
                dec, dec_record = D.decode_with_record(probs, rc.DECODE_LABELS, objective=rc.DECODE_SPEC["objective"], candidate_floor=rc.DECODE_SPEC["candidate_floor"])
                rat, rat_record = D.decode_with_record(probs, rc.DECODE_LABELS, objective="ratio", candidate_floor=rc.DESCRIPTION_SPEC["candidate_floor"])
                del probs
                path = rc.save_npz_new(decoded/f"{subject}.npz", argmax=argmax, decode=dec, ratio=rat)
                rows[subject] = dict(subject=subject, fold=row["fold"], probs36_sha256=row["probs36_sha256"], read_back=read_back, decoded_sha256=rc.sha256_file(path),
                                     decode=dec_record, ratio=rat_record, n_changed_decode=int(np.count_nonzero(dec != argmax)),
                                     n_changed_ratio=int(np.count_nonzero(rat != argmax)), seconds=time.perf_counter()-t0)
                log(f"{subject}: decode changes {rows[subject]['n_changed_decode']} voxels, ratio {rows[subject]['n_changed_ratio']} ({rows[subject]['seconds']:.1f} s)")
    finally:
        opened = _write_opened(out/rc.OPENED_DECODE, spy)
    record = dict(format="bayesreg.r3b.decode.v1", plan=plan, code=code, decoder_spec=rc.DECODE_SPEC, description_spec=rc.DESCRIPTION_SPEC, subjects=subjects, rows=rows,
                  implementation=implementation, decode_py_sha256=implementation["decode_py_sha256"], infer_record_sha256=rc.sha256_file(out/rc.INFER_RECORD), opened_files_sha256=rc.sha256_file(out/rc.OPENED_DECODE), m6=opened["m6"],
                  seconds=time.perf_counter()-started, synthetic=ctx.synthetic)
    rc.write_json_new(out/rc.DECODE_RECORD, record)
    return record


# ---- step 3: eval (the truth, once) -------------------------------------------------------------------------------------------------------------
def lock_check(ctx, bz_go=None):
    path = Path(ctx.lock)
    if bz_go is not None and not bz_go.strip():
        raise ValueError("--bz-go: BZ's words and the time are required")
    if bz_go is not None and not path.exists():
        raise ValueError(f"--bz-go: {path} does not exist, there is nothing to unlock")
    if bz_go is None and path.exists():
        raise Refused(f"{path} exists: the 16 calibration truths were read for R3b already; another read only with BZ's approval (--bz-go); the file is not deleted")


def lock_take(ctx, line, bz_go=None):
    from datetime import datetime, timezone
    path = Path(ctx.lock)
    line = dict(line, event="bz_approved_reread" if bz_go else "first_read", utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), pid=os.getpid(),
                **({"bz_go": bz_go} if bz_go else {}))
    text = json.dumps(line, ensure_ascii=False)+"\n"
    if bz_go:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(text)
        return line
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise Refused(f"{path} appeared just before the first truth label; not read") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(text)
    return line


class Reliability:
    """Equal-width bins on (0, 1] (calibrate.ece_bins convention): count, sum of confidence, sum of hits."""

    def __init__(self, bins=rc.ECE_BINS):
        self.n = np.zeros(bins)
        self.conf = np.zeros(bins)
        self.hit = np.zeros(bins)
        self.bins = bins

    def add(self, conf, hit):
        index = np.clip(np.ceil(conf*self.bins).astype(np.int64)-1, 0, self.bins-1)
        self.n += np.bincount(index, minlength=self.bins)
        self.conf += np.bincount(index, weights=conf, minlength=self.bins)
        self.hit += np.bincount(index, weights=hit.astype(np.float64), minlength=self.bins)

    def ece(self):
        total = self.n.sum()
        return float(np.sum(np.abs(self.hit-self.conf))/total) if total else None

    def table(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            return dict(n=self.n.tolist(), mean_confidence=(self.conf/self.n).tolist(), frequency=(self.hit/self.n).tolist(), ece=self.ece())


def bootstrap(values, *, n=rc.BOOTSTRAP["n"], seed=rc.BOOTSTRAP["seed"]):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if values.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(n, values.size))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paired_summary(deltas):
    d = np.asarray([np.nan if v is None else v for v in deltas], dtype=np.float64)
    ok = d[np.isfinite(d)]
    return dict(mean=float(ok.mean()) if ok.size else None, median=float(np.median(ok)) if ok.size else None, n_positive=int((ok > 0).sum()),
                n_negative=int((ok < 0).sum()), n_zero=int((ok == 0).sum()), n_defined=int(ok.size), n=int(d.size), bootstrap95=bootstrap(ok))


def evaluate(ctx, *, opened_files_sha256, bz_go=None, argv=None, log=print):
    import eval_seg
    from bayesreg.seg.unet.data import LabelGuard, OasisFiles, read_labels, LABEL_FILES
    if ctx.synthetic and bz_go is not None:
        raise ValueError("--bz-go is for the read lock of real data")
    lock_check(ctx, bz_go)                                                 # the first thing (the synthetic ctx has its own lock path)
    out = Path(ctx.out)
    eval_dir = out/rc.EVAL_DIR
    if eval_dir.exists() and any(eval_dir.iterdir()):
        raise FileExistsError(f"{eval_dir} exists and is not empty; never overwritten")
    for name in (rc.OPENED_INFER, rc.OPENED_DECODE):
        if not (out/name).is_file():
            raise Refused(f"{out/name} missing: the lead reads it before the truth is read")
    if rc.sha256_file(out/rc.OPENED_DECODE) != opened_files_sha256:
        raise Refused("--opened-files-sha256 is not the sha256 of opened_files_decode.json (the lead's release, calibration_plan.md section 4)")
    for name in (rc.OPENED_INFER, rc.OPENED_DECODE):
        if json.loads((out/name).read_text(encoding="utf-8")).get("m6") is not True:
            raise Refused(f"{name}: the spy recorded a breach (m6 not true); the truth is not read")
    code = _code(ctx)
    plan = _plan(ctx)
    subjects = ctx.subjects()
    decode_record = json.loads((out/rc.DECODE_RECORD).read_text(encoding="utf-8"))
    infer_record = json.loads((out/rc.INFER_RECORD).read_text(encoding="utf-8"))
    if decode_record["subjects"] != subjects or decode_record["plan"]["sha256"] != plan["sha256"] or decode_record["decoder_spec"] != json.loads(json.dumps(rc.DECODE_SPEC)):
        raise Refused("the decode record is not this plan's decoding of these subjects")
    if decode_record["opened_files_sha256"] != opened_files_sha256 or decode_record["infer_record_sha256"] != rc.sha256_file(out/rc.INFER_RECORD):
        raise Refused("the decode record does not bind the released opened_files_decode.json or the infer record")
    if infer_record.get("opened_files_sha256") != rc.sha256_file(out/rc.OPENED_INFER):                                           # review m7
        raise Refused("opened_files_infer.json is not the one the infer record names")
    implementation = rc.implementation_record()
    recorded = decode_record.get("implementation") or {}
    if recorded.get("decode_py_sha256") != implementation["decode_py_sha256"]:                                                    # review I1
        raise Refused(f"bayesreg/seg/decode.py is not the file the decode unit ran ({recorded.get('decode_py_sha256')} != {implementation['decode_py_sha256']})")
    truth_files = [Path(ctx.images)/s/LABEL_FILES[k] for s in subjects for k in ("seg35", "seg4")]
    files = [out/rc.INFER_RECORD, out/rc.DECODE_RECORD, out/rc.OPENED_INFER, out/rc.OPENED_DECODE, ctx.folds, Path(ctx.folds).parent/"folds.json", ctx.allowlist,
             ctx.seg4_map, *_source_files(ctx)]
    files += [out/rc.PROBS_DIR/s/n for s in subjects for n in ("probs36.npz", "brain.npz", "record.json")]+[out/rc.DECODED_DIR/f"{s}.npz" for s in subjects]
    rules = fw.CalibRules(unit="eval", lock=ctx.lock, repo=ctx.repo, results=ctx.results, data=ctx.data_link, images=ctx.images, out=out, files=files, own_dirs=[eval_dir], truth=truth_files)
    spy = _spy(rules)
    started = time.perf_counter()
    arms = ("argmax", "decode", "ratio")
    rows, truth_sha, volumes, transitions, seg4_table_share, lock_line = {a: {} for a in arms}, {}, {}, {}, {}, None
    label_rel = {(c, floor): Reliability() for c in range(1, D.N_CLASSES) for floor in rc.ECE_FLOORS}
    subject_rel = {s: {c: Reliability() for c in rc.DECODE_LABELS} for s in subjects}
    top_rel = Reliability()
    truth_p = {c: [] for c in rc.DECODE_LABELS}
    try:
        with spy:
            split, allowlist, _ = _inputs(ctx)
            guard = LabelGuard(split, allowlist, training_source=ctx.training_source)
            lookup = eval_seg.load_seg4_lookup(ctx.seg4_map)
            files_obj = OasisFiles(Path(ctx.images))
            for subject in subjects:
                fold = ctx.fold_of(subject)
                row = decode_record["rows"][subject]
                path = out/rc.DECODED_DIR/f"{subject}.npz"
                if rc.sha256_file(path) != row["decoded_sha256"]:
                    raise Refused(f"{path}: sha256 differs from the decode record")
                with np.load(path) as npz:
                    maps = {a: npz[a] for a in arms}
                probs, _ = _read_probs(ctx, subject, infer_record["rows"][subject])
                with np.load(out/rc.PROBS_DIR/subject/"brain.npz") as npz:
                    brain = npz["brain"]
                if rc.sha256_file(out/rc.PROBS_DIR/subject/"brain.npz") != infer_record["rows"][subject]["brain_sha256"]:
                    raise Refused(f"{subject}: brain.npz differs from the infer record")
                if lock_line is None:                                      # the moment before the first truth label (the synthetic ctx: its own lock path)
                    lock_line = lock_take(ctx, dict(out_dir=str(out), plan_sha256=plan["sha256"], code_commit=code.get("commit"), command=argv,
                                                    opened_files_decode_sha256=opened_files_sha256), bz_go=bz_go)
                true35, meta35 = read_labels(files_obj, subject, guard, fold=fold, kind="seg35")
                true4, meta4 = read_labels(files_obj, subject, guard, fold=fold, kind="seg4")
                truth_sha[subject] = dict(seg35=meta35["sha256"], seg4=meta4["sha256"])
                seg4_table_share[subject] = float(np.mean(lookup[true35] == true4))
                for a in arms:
                    rows[a][subject] = eval_seg.summarize_subject(maps[a], true35, true4, lookup)
                flat_p = probs.reshape(D.N_CLASSES, -1)
                flat_t = true35.reshape(-1)
                for c in range(1, D.N_CLASSES):
                    p = flat_p[c]
                    for floor in rc.ECE_FLOORS:
                        keep = p >= floor
                        label_rel[(c, floor)].add(p[keep], flat_t[keep] == c)
                    if c in rc.DECODE_LABELS:
                        keep = p >= rc.ECE_FLOORS[0]
                        subject_rel[subject][c].add(p[keep], flat_t[keep] == c)
                        truth_p[c].append(p[flat_t == c])
                mask = brain.reshape(-1)
                conf = flat_p[:, mask].max(axis=0)
                top_rel.add(conf, maps["argmax"].reshape(-1)[mask] == flat_t[mask])
                volumes[subject] = {str(c): dict(argmax=int(np.count_nonzero(maps["argmax"] == c)), decode=int(np.count_nonzero(maps["decode"] == c)),
                                                 ratio=int(np.count_nonzero(maps["ratio"] == c)), expected=float(flat_p[c].sum()),
                                                 truth=int(np.count_nonzero(true35 == c))) for c in rc.TARGET_LABELS}       # plan revision 2 item 8
                changed = maps["argmax"] != maps["decode"]
                pairs = np.stack([maps["argmax"][changed], maps["decode"][changed]]).astype(np.int64)
                keys, counts = np.unique(pairs[0]*100+pairs[1], return_counts=True)
                transitions[subject] = {f"{k//100}->{k % 100}": int(n) for k, n in zip(keys, counts)}
                del probs, flat_p, maps
                log(f"{subject}: small six argmax {rows['argmax'][subject]['groups']['small_six']:.4f} decode {rows['decode'][subject]['groups']['small_six']:.4f}")
            document = _eval_document(ctx, subjects, rows, volumes, transitions, label_rel, subject_rel, top_rel, truth_p, seg4_table_share)
            document.update(plan=plan, code=code, implementation=recorded, decode_py_sha256=recorded["decode_py_sha256"], truth_sha256=truth_sha, lock=lock_line, decoder_spec=rc.DECODE_SPEC, description_spec=rc.DESCRIPTION_SPEC,
                            decode_record_sha256=rc.sha256_file(out/rc.DECODE_RECORD), infer_record_sha256=rc.sha256_file(out/rc.INFER_RECORD),
                            opened_files_decode_sha256=opened_files_sha256, label_guard=guard.record(), seconds=time.perf_counter()-started, synthetic=ctx.synthetic)
            _write_eval(eval_dir, document)
    finally:
        _write_opened(eval_dir/rc.OPENED_EVAL, spy)
    return document


def _quantiles(values):
    values = np.concatenate(values) if values else np.zeros(0)
    if values.size == 0:
        return None
    return dict(n=int(values.size), quantiles={f"{q:.1f}": float(np.quantile(values, q)) for q in np.linspace(.1, .9, 9)})


def _eval_document(ctx, subjects, rows, volumes, transitions, label_rel, subject_rel, top_rel, truth_p, seg4_table_share):
    def deltas(arm, key):
        out = []
        for s in subjects:
            a = rows["argmax"][s]["groups"]["small_six"] if key == "small_six" else rows["argmax"][s][key]
            b = rows[arm][s]["groups"]["small_six"] if key == "small_six" else rows[arm][s][key]
            out.append(None if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)) else float(b-a))
        return out
    folds = {f: [i for i, s in enumerate(subjects) if s in ctx.calib[f]] for f in rc.FOLDS}
    paired = {}
    for arm in ("decode", "ratio"):
        paired[arm] = {}
        for key in ("small_six", "seg35", "seg4"):
            d = deltas(arm, key)
            paired[arm][key] = dict(all=paired_summary(d), per_subject=dict(zip(subjects, d)),
                                    **{f"fold_{f}": paired_summary([d[i] for i in idx]) for f, idx in folds.items()})
    means = {a: {k: float(np.mean([r[k] for r in rows[a].values()])) for k in ("seg35", "seg4")} | dict(
        small_six=float(np.mean([r["groups"]["small_six"] for r in rows[a].values()]))) for a in rows}
    per_label = {a: {str(c): [rows[a][s]["per_label"][c-1] for s in subjects] for c in range(1, D.N_CLASSES)} for a in rows}
    volume_bias = {}
    for c in rc.TARGET_LABELS:                                              # plan revision 2 item 8: all of R2's targets; beyond the small six, description only
        ratios = [volumes[s][str(c)]["argmax"]/volumes[s][str(c)]["truth"] for s in subjects if volumes[s][str(c)]["truth"] > 0]
        below = sum(r < 1 for r in ratios)
        volume_bias[str(c)] = dict(n=len(ratios), n_argmax_below_truth=int(below),
                                   mean_log_ratio_argmax=float(np.mean(np.log(np.maximum(ratios, 1e-9)))) if ratios else None,
                                   reading="画小" if ratios and below >= 12/16*len(ratios) else ("画大" if ratios and below <= 4/16*len(ratios) else "不一致"),
                                   role="small six (decoded)" if c in rc.DECODE_LABELS else "other target label (description only)")
    return dict(format=rc.EVAL_FORMAT, subjects=subjects, folds={f: list(ctx.calib[f]) for f in rc.FOLDS}, rows=rows, means=means, paired=paired,
                per_label=per_label, volumes=volumes, volume_bias=volume_bias, transitions_argmax_to_decode=transitions,
                reliability=dict(per_label={f"{c}@{floor:g}": r.table() for (c, floor), r in label_rel.items()},
                                 small_six_per_subject={s: {str(c): r.ece() for c, r in d.items()} for s, d in subject_rel.items()},
                                 top_label_brain=top_rel.table(), truth_voxel_probability={str(c): _quantiles(v) for c, v in truth_p.items()}),
                seg4_table_share=seg4_table_share, neighbours=list(rc.NEIGHBOURS))


def _write_eval(eval_dir, document):
    rc.write_json_new(eval_dir/rc.EVAL_FILE, document)
    lines = ["subject,fold,arm,seg35,seg4,small_six"]
    for arm, rows in document["rows"].items():
        for s, r in rows.items():
            fold = next(f for f, ids in document["folds"].items() if s in ids)
            lines.append(f"{s},{fold},{arm},{r['seg35']},{r['seg4']},{r['groups']['small_six']}")
    rc.write_new(eval_dir/"per_subject.csv", "\n".join(lines)+"\n")
    lines = ["arm,label,"+",".join(document["subjects"])]
    for arm, labels in document["per_label"].items():
        for c, values in labels.items():
            lines.append(f"{arm},{c},"+",".join("" if v is None or not np.isfinite(v) else repr(float(v)) for v in values))
    rc.write_new(eval_dir/"per_label.csv", "\n".join(lines)+"\n")
    rc.write_json_new(eval_dir/"reliability.json", document["reliability"])


# ---- step 4: decide -----------------------------------------------------------------------------------------------------------------------------
def decide(document, criterion=None):
    """The verdict of calibration_plan.md section 6 from eval_calib.json (no option changes it)."""
    criterion = dict(rc.CRITERION if criterion is None else criterion)
    reasons = []
    subjects = document.get("subjects") or []
    if len(subjects) != criterion["n_subjects"]:
        reasons.append(f"{len(subjects)} subjects, not {criterion['n_subjects']}")
    paired = (document.get("paired") or {}).get("decode") or {}
    values = {}
    for key in ("small_six", "seg35", "seg4"):
        summary = (paired.get(key) or {}).get("all") or {}
        values[key] = summary.get("mean")
        if values[key] is None or not np.isfinite(values[key]):
            reasons.append(f"mean delta {key} undefined")
        if summary.get("n") != len(subjects):
            reasons.append(f"delta {key} not over every subject")
        if summary.get("n_defined") != len(subjects):                       # plan revision 2 item 9 (review m11): an undefined delta of any subject
            reasons.append(f"delta {key} undefined for {len(subjects)-(summary.get('n_defined') or 0)} subject(s)")
    if document.get("decoder_spec") != json.loads(json.dumps(rc.DECODE_SPEC)):
        reasons.append("the decoder is not the pinned one")
    implementation = document.get("implementation") or {}
    bound = document.get("decode_py_sha256")                                # plan revision 2 item 1 (review I1): the decision binds the decoder's file;
    if not bound or bound != implementation.get("decode_py_sha256"):        # the top-level key decode_py_sha256 is R3a's (r3_common.DECISION_DECODE_KEY)
        reasons.append("the evaluation names no decode.py sha256, or two different ones")
    positive = ((paired.get("small_six") or {}).get("all") or {}).get("n_positive")
    checks = None
    if not reasons:
        checks = dict(small6_mean=dict(value=values["small_six"], bound=criterion["small6_mean_min"], ok=values["small_six"] >= criterion["small6_mean_min"]),
                      small6_positive=dict(value=positive, bound=criterion["small6_positive_min"], ok=positive >= criterion["small6_positive_min"]),
                      seg35_mean=dict(value=values["seg35"], bound=criterion["seg35_mean_min"], ok=values["seg35"] >= criterion["seg35_mean_min"]),
                      seg4_mean=dict(value=values["seg4"], bound=criterion["seg4_mean_min"], ok=values["seg4"] >= criterion["seg4_mean_min"]))
    verdict = "undecidable" if reasons else ("pass" if all(c["ok"] for c in checks.values()) else "not_pass")
    return dict(format=rc.DECISION_FORMAT, verdict=verdict, reasons=reasons, checks=checks, criterion=criterion, decoder_spec=rc.DECODE_SPEC,
                decode_py_sha256=bound, implementation=implementation, plan_sha256=(document.get("plan") or {}).get("sha256"), synthetic=bool(document.get("synthetic")),
                means=document.get("means"), n_subjects=len(subjects),
                rule="calibration_plan.md section 6 and revision 2 item 9: pass iff small-six mean delta >= +.005, >= 10/16 positive, seg35 mean delta >= 0, "
                     "seg4 mean delta >= -.0005, every delta defined for every subject")


def summary_markdown(decision, document):
    lines = [f"# R3b calibration reading ({'synthetic' if decision['synthetic'] else 'training-set calibration subjects, 16 out of fold'})", "",
             f"verdict: **{decision['verdict']}**" + (f" ({'; '.join(decision['reasons'])})" if decision["reasons"] else ""), ""]
    if decision["checks"]:
        for name, c in decision["checks"].items():
            lines.append(f"- {name}: {c['value']} (bound {c['bound']}): {'ok' if c['ok'] else 'not met'}")
    m = document.get("means") or {}
    for arm in ("argmax", "decode", "ratio"):
        if arm in m:
            lines.append(f"- {arm}: seg35 {m[arm]['seg35']:.4f} / seg4 {m[arm]['seg4']:.4f}, small six {m[arm]['small_six']:.4f}")
    lines += ["", "Descriptive quantities (ECE, volumes, ratio arm) are in eval/eval_calib.json; not part of the verdict."]
    return "\n".join(lines)+"\n"


def decide_step(ctx, *, log=print):
    out = Path(ctx.out)
    path = out/rc.EVAL_DIR/rc.EVAL_FILE
    document = json.loads(path.read_text(encoding="utf-8"))
    decision = decide(document, ctx.criterion)
    decision.update(eval_sha256=rc.sha256_file(path), decode_record_sha256=document.get("decode_record_sha256"), code=_code(ctx) if not ctx.synthetic else dict(kind="synthetic"))
    rc.write_json_new(out/rc.ANALYSIS_DIR/rc.DECISION_FILE, decision)
    rc.write_new(out/rc.ANALYSIS_DIR/rc.SUMMARY_FILE, summary_markdown(decision, document))
    log(f"verdict: {decision['verdict']}")
    return decision


# ---- dry run ------------------------------------------------------------------------------------------------------------------------------------
def start_strings(checkout="<the clean checkout with the data link>"):
    base = f"systemd-run --user --collect --working-directory={checkout} --nice=10 {THREADS} --setenv=PYTHONPATH={checkout}"
    return dict(
        rehearse=f"cd {checkout} && CUDA_VISIBLE_DEVICES= {PY} -B experiments/r3/r3b_calib.py rehearse   (CPU, before infer; reads infer's inputs under its spy, no model, image or label)",
        infer=f"{base} --unit=tag-r3b-infer --property=MemoryMax=11G --property=RuntimeMaxSec=3600 {PY} -B experiments/r3/r3b_calib.py infer --device cuda",
        decode=f"{base} --unit=tag-r3b-decode --property=MemoryMax=11G --property=RuntimeMaxSec=3600 --setenv=CUDA_VISIBLE_DEVICES= {PY} -B experiments/r3/r3b_calib.py decode",
        eval=(f"{base} --unit=tag-r3b-eval --property=MemoryMax=11G --property=RuntimeMaxSec=3600 --setenv=CUDA_VISIBLE_DEVICES= {PY} -B experiments/r3/r3b_calib.py eval "
              "--opened-files-sha256 '<sha256 of opened_files_decode.json, after the lead read both opened_files>'"),
        decide=f"cd {checkout} && CUDA_VISIBLE_DEVICES= {PY} -B experiments/r3/r3b_calib.py decide")


def dry_run(log=print):
    problems = []
    log(f"== R3b calibration, dry run ({time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}); nothing real is read but the plan and the code; results/seg/ is not opened; under splits/ only the sealed lists, by the guard itself, in the synthetic part")
    plan = ROOT/rc.PLAN_REL
    sha = rc.sha256_file(plan)
    log(f"plan {rc.PLAN_REL}: sha256 {sha}; pinned {rc.PLAN_SHA256}")
    if rc.PLAN_SHA256 is None:
        problems.append("PLAN_SHA256 not pinned yet (real runs refuse)")
    elif rc.PLAN_SHA256 != sha:
        problems.append("the plan is not the pinned one")
    log(f"decoder (plan section 3): {json.dumps(rc.DECODE_SPEC, ensure_ascii=False)}")
    log(f"criterion (plan section 6): {rc.CRITERION}")
    log(f"subjects (plan section 2): A {list(rc.CALIB_SUBJECTS['A'])}; B {list(rc.CALIB_SUBJECTS['B'])}")
    log(f"members (registry {rc.REGISTRY_REL}, pinned sha256 {rc.REGISTRY_SHA256[:12]}...): A {[s[:12] for s in rc.MEMBER_SHA256['A']]}, B {[s[:12] for s in rc.MEMBER_SHA256['B']]} "
        f"under {rc.UF_MODELS} (not opened here)")
    log(f"outputs {rc.OUT} (exists now: {rc.OUT.exists()}); read lock {rc.READ_LOCK} (exists now: {rc.READ_LOCK.exists()})")
    for name, text in start_strings().items():
        log(f"start {name}: {text}")
    log("infer passes the machine gate first (R2's gate and C: >= 40 GiB) and exits 8 without reading anything when it is closed; if the rehearsal or infer stops "
        "on the spy, rename OUT (keep it) before an authorised rerun (plan revision 2 items 3, 4)")
    log("〔估算〕infer GPU 5-8 min, host about 6 GiB; decode CPU 15-20 min (about 4 s per subject per objective on a full grid, plus the read-back); eval about 5 min; "
        "disk about 1 GB (probs36.npz float16) + 16 MB")
    import subprocess
    import r3b_synthetic
    with tempfile.TemporaryDirectory(prefix="r3b_dry_") as work:
        ctx = r3b_synthetic.build(Path(work))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")

        def step(*args):
            done = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), *args, "--synthetic", work], env=env, capture_output=True, text=True)
            if done.returncode:
                problems.append(f"synthetic {args[0]} exited {done.returncode}: {done.stderr.strip()[-400:]}")
            return done.returncode == 0
        ok = step("rehearse") and step("infer", "--device", "cpu") and step("decode")
        ok = ok and step("eval", "--opened-files-sha256", rc.sha256_file(Path(ctx.out)/rc.OPENED_DECODE)) and step("decide")
        if ok:
            decision = json.loads((Path(ctx.out)/rc.ANALYSIS_DIR/rc.DECISION_FILE).read_text())
            opened = [json.loads((Path(ctx.out)/n).read_text())["m6"] for n in (rc.OPENED_INFER, rc.OPENED_DECODE)]
            opened.append(json.loads((Path(ctx.out)/rc.EVAL_DIR/rc.OPENED_EVAL).read_text())["m6"])
            rehearsals = sorted((Path(ctx.results)/rc.REHEARSAL_ROOT.name).glob("*/opened_files.json"))
            opened.append(json.loads(rehearsals[-1].read_text())["m6"] if rehearsals else None)
            log(f"synthetic run end to end, one process per step as the units (CPU, {len(ctx.subjects())} synthetic calibration subjects, tiny members): "
                f"rehearse, infer, decode, eval, decide; verdict {decision['verdict']}, decode.py sha256 bound {decision.get('decode_py_sha256')}; "
                f"spies m6 (infer, decode, eval, rehearse) {opened}")
            if decision["verdict"] == "undecidable" or not all(opened):
                problems.append("the synthetic run did not reach a verdict cleanly")
    log(f"== dry run problems: {problems or 'none'}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("dry-run")
    for name in ("rehearse", "infer", "decode", "eval", "decide"):
        p = sub.add_parser(name)
        p.add_argument("--synthetic", type=Path, default=None, help="a work directory built by r3b_synthetic.build (tests); never the real roots")
        if name == "infer":
            p.add_argument("--device", default="cuda")
        if name == "eval":
            p.add_argument("--opened-files-sha256", required=True)
            p.add_argument("--bz-go", default=None)
    args = parser.parse_args(argv)
    if args.step == "dry-run":
        return 0 if not dry_run() else 1
    if args.synthetic is not None:
        import r3b_synthetic
        ctx = r3b_synthetic.load(args.synthetic)
    else:
        ctx = real_ctx()
    try:
        if args.step == "rehearse":
            rehearse(ctx)
        elif args.step == "infer":
            infer(ctx, device=args.device)
        elif args.step == "decode":
            decode_step(ctx)
        elif args.step == "eval":
            evaluate(ctx, opened_files_sha256=args.opened_files_sha256, bz_go=args.bz_go, argv=sys.argv)
        else:
            decide_step(ctx)
    except Refused as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 8
    return 0


if __name__ == "__main__":
    sys.exit(main())
