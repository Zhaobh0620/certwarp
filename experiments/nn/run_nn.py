"""Driver of work package NN (runs/20261105-nn/任务书.md, plan.md). This file runs in sci310 and imports no nnunetv2; every nnU-Net command is a child in the environment nnunet (nn_common.nn_env). It starts no unit: the units are started by the lead, from main, with the printed strings.

    python -B experiments/nn/run_nn.py --dry-run [--scratch DIR]
        the start strings, the commands, the gate's reading, the subject lists (ids only), the rules on made-up numbers, the evaluation and the rule on synthetic subjects. No GPU, no unit, no nnU-Net command, nothing written outside the scratch directory.
    python -B experiments/nn/run_nn.py prepare
        foreground, CPU, from a clean tree: the checks of the subject lists, nnUNet_raw of symbolic links, dataset.json, the label check, prepare_manifest.json (prepare_nn.py).
    python -B experiments/nn/run_nn.py preprocess
        foreground or background, plain process, CPU, nice 19, -np 2: waits until MemAvailable >= 4 GiB, runs nnUNetv2_plan_and_preprocess, stops it if the sum of its PSS exceeds 5 GiB or MemAvailable falls under 1.5 GiB; preprocess_record.json with the time, the sizes and the plan; then link-gt.
    python -B experiments/nn/run_nn.py link-gt
        foreground, CPU: nnU-Net's copies of the labels in gt_segmentations become symbolic links to the label files, after a comparison of the whole files (hard rule 8; prepare_nn.link_gt_segmentations), gt_links_record.json. smoke, train and infer refuse without the 106 links.
    python -B experiments/nn/run_nn.py snapshot [--repo DIR]
        foreground, from a clean main: the code snapshot <out>/snapshot of the units.
    python -B experiments/nn/run_nn.py print-launch {smoke,train,train-continue,infer,eval}
    python -B <snapshot>/experiments/nn/run_nn.py smoke                                     unit tag-nn-smoke
        behind the gate: the real trainer in its own results root, stopped after SMOKE_EPOCHS epochs; seconds per epoch, the augmentation processes, GPU memory and mean utilization, host memory into smoke_measure.json.
    python -B <snapshot>/experiments/nn/run_nn.py decide-epochs
        foreground, from the snapshot: from what the smoke measured, by the plan's rule, epochs_decision.json with the code record, written once. `train` reads only that file, and recomputes it from smoke_measure.json: a difference stops it.
    python -B <snapshot>/experiments/nn/run_nn.py retire-smoke --reason TEXT
        foreground: after a smoke that did not pass (no measure, or the rule says not ok): its files move to smoke_attempts/<utc>/ with the reason, nothing is deleted (hard rule 4), and the smoke can run again. Refused when the smoke passed: the rule is not rolled twice.
    python -B <snapshot>/experiments/nn/run_nn.py train [--continue] [--final-validation]   unit tag-nn-train
        behind the gate: nnUNetv2_train 935 3d_fullres all with the trainer of epochs_decision.json. Refuses to start over a run that has checkpoints without --continue, and --continue without a checkpoint. After the last epoch nnU-Net predicts the 106 training cases ("validation" of fold all is the training set): the driver stops it there, unless --final-validation; the log line that starts it comes after checkpoint_final.pth is written.
    python -B <snapshot>/experiments/nn/run_nn.py infer                                     unit tag-nn-infer
        behind the gate: the 20 development subjects (the allowlist source, admitted by eval_seg.admit_subjects), links to their images only, infer_nn.py; <out>/pred_dev/<subject>.nii.gz, infer_record.json with the seconds per subject.
    python -B <snapshot>/experiments/nn/run_nn.py eval [--bz-go TEXT]                       unit tag-nn-eval
        sci310, no GPU: eval_nn.py (the development truth, once) -> decide_nn.py; <out>/eval_nn, <out>/analysis.
    python -B experiments/nn/run_nn.py status

The gate of the GPU steps: no other tag-* or seq-vp-* unit active, 1-minute load < 1.5, no GPU compute process, GPU utilization <= 20 %, GPU free >= 12 GiB (the plan's target is 8), MemAvailable >= 9 GiB, /home free >= 100 GB, /mnt/c free >= 15 GiB; polled every 60 s, closed for --wait-minutes: nothing starts, exit 8.
Exit codes: 0 done, 2 usage, 8 a stop (the reason is printed). A failed nnU-Net command is kept whole and not rerun.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(HERE), str(HERE.parent/"unv2")]

import nn_common as nc  # noqa: E402
import nn_lock  # noqa: E402
from bayesreg.seg.unet.snapshot import SNAPSHOT_PATHS, CodeNotClean, make_snapshot, require_clean_code, require_snapshot, verify_snapshot  # noqa: E402

UNITS = dict(smoke="tag-nn-smoke", train="tag-nn-train", infer="tag-nn-infer", eval="tag-nn-eval")
MEMORY_MAX = dict(smoke="12G", train="12G", infer="8G", eval="6G")
RUNTIME_MAX_SECONDS = dict(smoke=5400, train=39600, infer=7200, eval=3600)     # train: 10 h of the rule + 30 minutes of gate + the rest for what `Epoch time` leaves out (checkpoint_best, the plot); a safety valve, --continue goes on after it
SNAPSHOT_PATHS_NN = SNAPSHOT_PATHS+("experiments/unv2", "experiments/nn")
U1_EVAL = nc.RESULTS/"seg/unet35_v3_large/eval_u1/eval_u1.json"
SMI = "/usr/lib/wsl/lib/nvidia-smi"
MIN_GPU_FREE_MIB, MAX_GPU_UTIL, MAX_LOAD, MIN_HOME_GB, MIN_C_GIB, MIN_AVAILABLE_GIB = 12288, 20, 1.5, 100, 15, 9
POLL_SECONDS, WAIT_MINUTES = 60, 30
PREPROCESS_MIN_AVAILABLE_GIB, PREPROCESS_LIMIT_GIB, PREPROCESS_FLOOR_GIB, PREPROCESS_WAIT_MINUTES = 4., 5., 1.5, 120      # the brief: wait while under 4 GiB; <= 5 GiB during preprocessing
SMOKE_EPOCHS = 4                        # the lead, ~1:50 am CST 11-05 (review, important 1): 4 epochs, the rule on the median from the 2nd on
LIMIT_HOURS = 10.                       # the lead, ~1:50 am CST 11-05, in place of the brief's about 8: .88 is the final goal, not the goal of the 12 hours. More than this for 250 epochs (144 s per epoch) -> the 100-epoch variant
HOST_LIMIT_GIB, GPU_LIMIT_MIB = 11., 22528     # the smoke's budgets: 11 of the 12 GiB of the unit; the GPU has 24 GiB
WATCH_SECONDS, GPU_SAMPLE_SECONDS = 5., 15.
LOG_MARGIN_SECONDS = 1.                 # the logs of a run are those written since just before it started; the file clock is coarse. 5 here counted the old log of a run resumed within seconds (seen in sandbox_check)
EPOCH_TIME = re.compile(r"Epoch time: ([0-9.]+) s")                   # nnUNetTrainer.on_epoch_end
VALIDATION_START = re.compile(r"^\S+ \S+: predicting ", re.MULTILINE)  # nnUNetTrainer.perform_actual_validation: the first line after on_train_end has written checkpoint_final.pth
LOG_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?):", re.MULTILINE)   # nnUNetTrainer.print_to_log_file


class Stop(Exception):
    """A stop of this driver: exit 8, the reason printed, nothing rerun."""


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


LOG_FILE = None


def set_run_log(out):
    global LOG_FILE
    Path(out).mkdir(parents=True, exist_ok=True)
    LOG_FILE = Path(out)/"run_nn.log"


def log(message):
    line = f"{utc()} {message}"
    print(line, flush=True)
    if LOG_FILE is not None:
        with LOG_FILE.open("a", encoding="utf-8") as stream:
            stream.write(line+"\n")


def write_json_once(path, document):
    path = Path(path)
    if path.exists():
        raise Stop(f"{path} exists; never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
    return document


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


# ---- the start strings -------------------------------------------------------------------------------------------------------------------------------

def check_launch(argv):
    """The rules of a start string (the brief): no test runner, the unit's prefix, the results root, the host limit, a runtime limit."""
    argv = [str(a) for a in argv]
    text = " ".join(argv)
    if "pytest" in text:
        raise ValueError(f"a start string must not contain the test runner: {text}")
    units = [a for a in argv if a.startswith("--unit=")]
    if len(units) != 1 or not units[0].startswith("--unit=tag-nn-"):
        raise ValueError(f"the unit must be tag-nn-*: {units}")
    if "seq-vp-" in text:
        raise ValueError("a unit of this project never uses the seq-vp- prefix")
    if not any(a.startswith(f"--working-directory={nc.RESULTS}/") for a in argv):
        raise ValueError(f"the working directory is the snapshot under {nc.RESULTS}")
    if not any(a.startswith("--property=MemoryMax=") for a in argv) or not any(a.startswith("--property=RuntimeMaxSec=") for a in argv):
        raise ValueError("MemoryMax and RuntimeMaxSec are required")
    return argv


def launch_argv(step, *, out=nc.OUT, wait_minutes=WAIT_MINUTES, bz_go=None):
    """The lead's start of a step, as a list; printed by print-launch, never run by this file."""
    resume = step == "train-continue"
    name = "train" if resume else step
    if name not in UNITS:
        raise ValueError(f"no step {step}")
    snapshot = Path(out)/"snapshot"
    extra = {"smoke": ["--wait-minutes", str(wait_minutes)], "train": ["--wait-minutes", str(wait_minutes), *(["--continue"] if resume else [])], "infer": ["--wait-minutes", str(wait_minutes)], "eval": ["--bz-go", bz_go] if bz_go else []}[name]
    return check_launch(["systemd-run", "--user", f"--unit={UNITS[name]}", "--collect", f"--working-directory={snapshot}", f"--property=MemoryMax={MEMORY_MAX[name]}", "--property=MemorySwapMax=2G", f"--property=RuntimeMaxSec={RUNTIME_MAX_SECONDS[name]}", "--setenv=PYTHONDONTWRITEBYTECODE=1",
                         *[f"--setenv={v}={nc.THREADS}" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")], nc.PY, "-B", str(snapshot/"experiments/nn/run_nn.py"), name, "--out", str(out), *extra])


# ---- the gate ------------------------------------------------------------------------------------------------------------------------------------------

def mem_available_gib(path="/proc/meminfo"):
    for line in Path(path).read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])/2**20
    raise ValueError(f"{path}: no MemAvailable")


def gate_problems(own_unit, *, run=subprocess.run, smi=SMI, load=os.getloadavg, disk=shutil.disk_usage, available=mem_available_gib):
    """Why a GPU step may not start now (empty: it may). A query that fails counts as closed."""
    problems = []
    try:
        units = run(["systemctl", "--user", "list-units", "--state=active,activating", "--no-legend", "--plain", "tag-*", "seq-vp-*"], capture_output=True, text=True, timeout=60)
        if units.returncode != 0:
            problems.append(f"systemctl failed: {units.stderr.strip()}")
        else:
            others = sorted(line.split()[0] for line in units.stdout.splitlines() if line.strip() and line.split()[0].removesuffix(".service") != own_unit)
            if others:
                problems.append(f"units active: {' '.join(others)}")
    except (OSError, subprocess.SubprocessError) as exc:
        problems.append(f"systemctl failed: {exc}")
    one_minute = load()[0]
    if not one_minute < MAX_LOAD:
        problems.append(f"1-minute load {one_minute:.2f} >= {MAX_LOAD}")
    try:
        gpu = run([smi, "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60)
        apps = run([smi, "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
        fields = [f.strip() for f in (gpu.stdout.strip().splitlines() or [""])[0].split(",")]
        if gpu.returncode != 0 or apps.returncode != 0 or len(fields) != 2 or not all(f.isdigit() for f in fields):
            problems.append(f"nvidia-smi gave no reading ({gpu.stdout.strip()!r})")
        else:
            free, util = int(fields[0]), int(fields[1])
            if free < MIN_GPU_FREE_MIB:
                problems.append(f"GPU free {free} MiB < {MIN_GPU_FREE_MIB} MiB")
            if util > MAX_GPU_UTIL:
                problems.append(f"GPU utilization {util} % > {MAX_GPU_UTIL} %")
            pids = [p.strip() for p in apps.stdout.splitlines() if p.strip().isdigit()]
            if pids:
                problems.append(f"GPU compute processes {pids}")
    except (OSError, subprocess.SubprocessError) as exc:
        problems.append(f"nvidia-smi failed: {exc}")
    problems += disk_problems(disk)
    host = available()
    if host < MIN_AVAILABLE_GIB:
        problems.append(f"MemAvailable {host:.1f} GiB < {MIN_AVAILABLE_GIB} GiB")
    return problems


def disk_problems(disk=shutil.disk_usage):
    try:
        home, c_drive = disk("/home").free/1e9, disk("/mnt/c").free/2**30
    except OSError as exc:
        return [f"disk query failed: {exc}"]
    return ([f"/home free {home:.0f} GB < {MIN_HOME_GB} GB"] if home < MIN_HOME_GB else [])+([f"/mnt/c free {c_drive:.0f} GiB < {MIN_C_GIB} GiB"] if c_drive < MIN_C_GIB else [])


def wait_for_gate(own_unit, wait_minutes, *, problems=gate_problems, sleep=time.sleep, poll=POLL_SECONDS):
    """Returns when the gate is open; Stop when it stayed closed for wait_minutes."""
    deadline = time.monotonic()+60*wait_minutes
    while True:
        reasons = problems(own_unit)
        if not reasons:
            log("gate open")
            return
        if time.monotonic() >= deadline:
            raise Stop(f"the gate stayed closed for {wait_minutes} minutes: {'; '.join(reasons)}; nothing started")
        log(f"gate closed: {'; '.join(reasons)}")
        sleep(poll)


def wait_for_memory(minimum_gib, wait_minutes, *, available=mem_available_gib, sleep=time.sleep, poll=POLL_SECONDS):
    """The brief: under this much available, wait. Stop when it stayed under for wait_minutes."""
    deadline = time.monotonic()+60*wait_minutes
    while True:
        host = available()
        if host >= minimum_gib:
            log(f"MemAvailable {host:.1f} GiB >= {minimum_gib} GiB")
            return host
        if time.monotonic() >= deadline:
            raise Stop(f"MemAvailable stayed under {minimum_gib} GiB for {wait_minutes} minutes ({host:.1f} GiB now); nothing started")
        log(f"MemAvailable {host:.1f} GiB < {minimum_gib} GiB, waiting")
        sleep(poll)


# ---- a watched child ------------------------------------------------------------------------------------------------------------------------------------

def group_members(pgid, *, proc=Path("/proc")):
    """The living pids of a process group, from /proc/<pid>/stat; a zombie holds nothing and waits for its parent."""
    members = []
    for entry in proc.iterdir():
        if entry.name.isdigit():
            try:
                fields = (entry/"stat").read_text().rsplit(")", 1)[1].split()
            except (OSError, IndexError):
                continue
            if int(fields[2]) == pgid and fields[0] != "Z":
                members.append(int(entry.name))
    return members


def group_pss_gib(pgid):
    """The sum of the PSS of a process group: shared pages are counted once, which the sum of RSS would not do across the augmentation workers."""
    total = 0
    for pid in group_members(pgid):
        try:
            for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
                if line.startswith("Pss:"):
                    total += int(line.split()[1])
                    break
        except OSError:
            continue
    return total/2**20


def gpu_reading(*, run=subprocess.run, smi=SMI):
    """(MiB used, utilization %), or None."""
    try:
        result = run([smi, "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        used, utilization = result.stdout.strip().splitlines()[0].split(",")
        return int(used), int(utilization)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def cgroup_peak_gib():
    """memory.peak of this process's cgroup: inside a unit, what MemoryMax limits (page cache included); None if it cannot be read."""
    try:
        path = Path("/proc/self/cgroup").read_text().strip().split("::")[-1].lstrip("/")
        return int((Path("/sys/fs/cgroup")/path/"memory.peak").read_text())/2**30
    except (OSError, ValueError):
        return None


def stop_group(process, *, grace=30., first_signal=signal.SIGTERM):
    """first_signal to the child's group (SIGTERM; sandbox_check.py's crash: SIGKILL), then SIGKILL. The child has its own session (run_watched)."""
    for sig, patience in ((first_signal, grace), (signal.SIGKILL, 10.)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=patience)
            break
        except subprocess.TimeoutExpired:
            continue
    # the group may outlive its leader (workers)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def empty_group(pgid, *, timeout=10.):
    """Wait for a finished child's group to empty, SIGKILL what is left (a worker that ignores SIGTERM, resource_tracker). Returns the pids still alive after that: an orphan would hold the GPU and close the gate of the next step."""
    deadline = time.monotonic()+timeout
    killed = False
    while True:
        left = group_members(pgid)
        if not left:
            return []
        if time.monotonic() >= deadline:
            if killed:
                return left
            killed, deadline = True, time.monotonic()+timeout
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                return []
        time.sleep(.05)


def run_watched(argv, *, env, log_path, cwd, should_stop=None, memory_limit_gib=None, available_floor_gib=None, sample_gpu=False, poll=WATCH_SECONDS, available=mem_available_gib, gpu=gpu_reading, grace=30., first_signal=signal.SIGTERM):
    """The child in its own session, stdout and stderr to log_path. Every poll: the PSS of its group; should_stop() True -> stopped, `stopped` is "condition"; over memory_limit_gib or MemAvailable under available_floor_gib -> stopped, `stopped` is "memory". Returns what happened; the caller decides what it means. `group_left`: the pids of its group still alive at the end, normally none."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started, started_utc = time.monotonic(), utc()
    peak_pss, peak_gpu, utilization, stopped, reason, next_gpu = 0., None, [], None, None, 0.
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"{utc()} start: {shlex.join(str(a) for a in argv)}\n")
        stream.flush()
        process = subprocess.Popen([str(a) for a in argv], cwd=str(cwd), env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                time.sleep(poll)
                pss = group_pss_gib(process.pid)
                peak_pss = max(peak_pss, pss)
                if sample_gpu and time.monotonic() >= next_gpu:
                    reading = gpu()
                    if reading is not None:
                        peak_gpu = max(peak_gpu or 0, reading[0])
                        utilization.append(reading[1])
                    next_gpu = time.monotonic()+GPU_SAMPLE_SECONDS
                if memory_limit_gib is not None and pss > memory_limit_gib:
                    stopped, reason = "memory", f"the PSS of the group is {pss:.2f} GiB, over {memory_limit_gib} GiB"
                elif available_floor_gib is not None and available() < available_floor_gib:
                    stopped, reason = "memory", f"MemAvailable fell under {available_floor_gib} GiB"
                elif should_stop is not None and should_stop():
                    stopped, reason = "condition", "the stop condition held"
                if stopped:
                    stream.write(f"{utc()} stopping: {reason}\n")
                    stream.flush()
                    stop_group(process, grace=grace, first_signal=first_signal)
                    break
        except BaseException:
            stop_group(process, grace=grace)
            raise
        returncode = process.wait()
        group_left = empty_group(process.pid)
        stream.write(f"{utc()} end: exit {returncode}, group left {group_left}\n")
    return dict(argv=[str(a) for a in argv], returncode=returncode, stopped=stopped, stop_reason=reason, wall_seconds=time.monotonic()-started, started_utc=started_utc, ended_utc=utc(), pid=process.pid, group_left=group_left, host_peak_pss_gib=peak_pss, gpu_peak_used_mib=peak_gpu, gpu_mean_utilization_percent=statistics.mean(utilization) if utilization else None, gpu_samples=len(utilization), cgroup_peak_gib=cgroup_peak_gib(), log=str(log_path))


def training_progress(fold_dir, since):
    """What the training logs written since `since` (a time.time()) say: the seconds of each epoch, and whether the final validation has begun. A resumed run writes a new log; the older ones are left out."""
    texts = []
    for path in sorted(Path(fold_dir).glob("training_log_*.txt")):
        if path.stat().st_mtime >= since:
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
    text = "\n".join(texts)
    return dict(epoch_seconds=[float(v) for v in EPOCH_TIME.findall(text)], validating=VALIDATION_START.search(text) is not None)


def log_span_seconds(text):
    """First to last timestamp of one training log, in seconds; 0 with fewer than two."""
    stamps = [datetime.fromisoformat(value) for value in LOG_TIMESTAMP.findall(text)]
    return (max(stamps)-min(stamps)).total_seconds() if len(stamps) > 1 else 0.


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_bytes(path):
    """The size of a tree, links not followed."""
    return sum(p.lstat().st_size for p in Path(path).rglob("*") if p.is_file() or p.is_symlink()) if Path(path).exists() else 0


def env_record(out, *, env):
    """Versions of the environment nnunet, the GPU, the disk: a small text file next to the results."""
    path = Path(out)/f"env_{stamp()}.txt"
    probe = subprocess.run([str(nc.NN_PY), "-I", "-B", "-c", "import sys, importlib.metadata as m, numpy, torch; print(sys.version.replace(chr(10), ' ')); print('nnunetv2', m.version('nnunetv2'), 'torch', torch.__version__, 'cuda', torch.version.cuda, 'numpy', numpy.__version__, 'SimpleITK', m.version('SimpleITK'), 'blosc2', m.version('blosc2'))"], env=dict(env, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True)
    smi = subprocess.run([SMI, "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"], capture_output=True, text=True)
    lines = [f"recorded: {utc()}", f"python: {nc.NN_PY}", *probe.stdout.splitlines(), *probe.stderr.splitlines(), f"nvidia-smi: {smi.stdout.strip()}", f"disk /home free: {shutil.disk_usage('/home').free/1e9:.0f} GB", f"MemAvailable: {mem_available_gib():.1f} GiB", *(f"{k}={v}" for k, v in sorted(env.items()) if k.startswith("nnUNet") or k in ("CUDA_VISIBLE_DEVICES", "PATH", "PYTHONNOUSERSITE"))]
    Path(out).mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    return path


# ---- prepare, preprocess ----------------------------------------------------------------------------------------------------------------------------------

def preprocessed_data_directory(out):
    return nc.preprocessed_root(out)/nc.DATASET/f"{nc.PLANS}_{nc.CONFIGURATION}"


def preprocessed_identifiers(out):
    """The cases of the preprocessed folder: nnU-Net writes one <case>.pkl for each."""
    return sorted(p.stem for p in preprocessed_data_directory(out).glob("*.pkl"))


def network_parameters(plans_path, *, env, n_outputs=36):
    """The number of parameters of the planned network, built on the CPU in the environment nnunet. None if that fails: it is only for the record."""
    snippet = ("import json, sys\nfrom nnunetv2.utilities.get_network_from_plans import get_network_from_plans\narch = json.load(open(sys.argv[1]))['configurations']['3d_fullres']['architecture']\n"
               f"net = get_network_from_plans(arch['network_class_name'], arch['arch_kwargs'], arch['_kw_requires_import'], 1, {n_outputs}, allow_init=False, deep_supervision=True)\nprint(sum(p.numel() for p in net.parameters()))\n")
    result = subprocess.run([str(nc.NN_PY), "-I", "-B", "-c", snippet, str(plans_path)], env=env, capture_output=True, text=True)
    try:
        return int(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def cmd_preprocess(args):
    import prepare_nn
    code = require_clean_code()
    out = Path(args.out)
    set_run_log(out)
    record_path = out/"preprocess_record.json"
    if record_path.exists():
        raise Stop(f"{record_path} exists; the preprocessing is done once")
    admitted = prepare_nn.admitted_training_subjects(code=code)
    prepare_nn.verify_raw(nc.raw_root(out)/nc.DATASET, admitted["subjects"], nc.DATA_ROOT)
    if not (out/"prepare_manifest.json").is_file():
        raise Stop(f"{out/'prepare_manifest.json'}: no such file; prepare comes first")
    problems = disk_problems()
    if problems:
        raise Stop("; ".join(problems))
    before = wait_for_memory(PREPROCESS_MIN_AVAILABLE_GIB, args.wait_minutes)
    env = nc.nn_env(out, gpu=False)
    env_record(out, env=env)
    outcome = run_watched(nc.plan_argv(), env=env, log_path=out/"logs"/"plan_and_preprocess.log", cwd=out, memory_limit_gib=PREPROCESS_LIMIT_GIB, available_floor_gib=PREPROCESS_FLOOR_GIB)
    log(f"plan_and_preprocess: exit {outcome['returncode']}, {outcome['wall_seconds']:.0f} s, peak PSS {outcome['host_peak_pss_gib']:.2f} GiB, stopped {outcome['stopped']}")
    if outcome["returncode"] != 0 or outcome["stopped"]:
        write_json_once(out/f"preprocess_failed_{stamp()}.json", outcome)
        raise Stop(f"plan_and_preprocess did not finish: exit {outcome['returncode']}, {outcome['stop_reason']}; log {outcome['log']}; kept")
    identifiers = preprocessed_identifiers(out)
    if identifiers != admitted["subjects"]:
        raise Stop(f"the preprocessed cases are not the training subjects: only there {sorted(set(identifiers)-set(admitted['subjects']))[:5]}, missing {sorted(set(admitted['subjects'])-set(identifiers))[:5]}")
    gt_bytes = directory_bytes(nc.preprocessed_root(out)/nc.DATASET/"gt_segmentations")
    plans = nc.preprocessed_root(out)/nc.DATASET/f"{nc.PLANS}.json"
    summary = prepare_nn.plan_summary(plans, nc.preprocessed_root(out)/nc.DATASET/"dataset_fingerprint.json")
    summary["parameters"] = network_parameters(plans, env=env)
    document = dict(created_utc=utc(), code=code, mem_available_before_gib=before, outcome=outcome, sizes_bytes=dict(preprocessed=directory_bytes(nc.preprocessed_root(out)), preprocessed_data=directory_bytes(preprocessed_data_directory(out)), gt_segmentations=gt_bytes, raw_links=directory_bytes(nc.raw_root(out))),
                    n_cases=len(identifiers), plan=summary, note="gt_segmentations 是 nnU-Net 自己从 labelsTr 复制到预处理目录的标签文件（训练受试者）；紧接着由 link-gt 换成符号链接，见 gt_links_record.json。")
    write_json_once(record_path, document)
    log(f"plan: {json.dumps({k: summary[k] for k in ('patch_size', 'batch_size', 'spacing', 'n_stages', 'features_per_stage', 'parameters')})}")
    link_gt(out, admitted["subjects"], code=code)
    return document


def link_gt(out, subjects, *, code, data_root=nc.DATA_ROOT):
    """The copies of the labels become links (prepare_nn.link_gt_segmentations), recorded once; a second run only verifies."""
    import prepare_nn
    out = Path(out)
    record = prepare_nn.link_gt_segmentations(prepare_nn.gt_directory(nc.preprocessed_root(out)), subjects, data_root)
    log(f"gt_segmentations: {json.dumps(record)}")
    path = out/"gt_links_record.json"
    if not path.exists():
        write_json_once(path, dict(record, created_utc=utc(), code=code, data_root=str(data_root), note="硬约束 8：nnU-Net 复制的 106 个标签文件比对全文件后换成符号链接；--final-validation 仍能读。"))
    return record


def cmd_link_gt(args):
    import prepare_nn
    code = require_clean_code()
    set_run_log(args.out)
    return link_gt(args.out, prepare_nn.admitted_training_subjects(code=code)["subjects"], code=code)


def require_preprocessed(out, *, code=None):
    """Before a step that trains or predicts: the links and the preprocessed cases are exactly the training subjects."""
    import prepare_nn
    if not (Path(out)/"preprocess_record.json").is_file():
        raise Stop(f"{Path(out)/'preprocess_record.json'}: no such file; prepare and preprocess come first")
    admitted = prepare_nn.admitted_training_subjects(code=code)
    prepare_nn.verify_raw(nc.raw_root(out)/nc.DATASET, admitted["subjects"], nc.DATA_ROOT)
    identifiers = preprocessed_identifiers(out)
    if identifiers != admitted["subjects"]:
        raise Stop(f"the preprocessed cases are not the training subjects: only there {sorted(set(identifiers)-set(admitted['subjects']))[:5]}, missing {sorted(set(admitted['subjects'])-set(identifiers))[:5]}")
    prepare_nn.verify_gt_links(prepare_nn.gt_directory(nc.preprocessed_root(out)), admitted["subjects"], nc.DATA_ROOT)
    return admitted["subjects"]


# ---- smoke and the epoch rule --------------------------------------------------------------------------------------------------------------------------

def checkpoints_in(fold_dir):
    return sorted(p.name for p in Path(fold_dir).glob("checkpoint_*.pth"))


def cmd_smoke(args):
    code = require_snapshot()
    out = Path(args.out)
    set_run_log(out)
    measure_path = out/"smoke_measure.json"
    if measure_path.exists():
        raise Stop(f"{measure_path} exists; the smoke ran")
    results = nc.smoke_results_root(out)
    if results.exists():
        raise Stop(f"{results} exists and there is no measure: a smoke that did not finish. retire-smoke --reason keeps it and clears the way")
    require_preprocessed(out, code=code)
    wait_for_gate(UNITS["smoke"], args.wait_minutes)
    trainer = nc.TRAINERS[nc.EPOCHS_LONG]
    fold_dir = nc.fold_directory(trainer, results=results)
    env = nc.nn_env(out, results=results, gpu=True)
    env_record(out, env=env)
    since = time.time()-LOG_MARGIN_SECONDS
    outcome = run_watched(nc.train_argv(trainer), env=env, log_path=out/"logs"/"smoke.log", cwd=out, should_stop=lambda: len(training_progress(fold_dir, since)["epoch_seconds"]) >= SMOKE_EPOCHS, sample_gpu=True)
    progress = training_progress(fold_dir, since)
    document = dict(trainer=trainer, epochs_asked=SMOKE_EPOCHS, epoch_seconds=progress["epoch_seconds"], n_proc_da=int(env["nnUNet_n_proc_DA"]), n_proc_da_validation=nc.N_PROC_DA_VALIDATION, outcome=outcome, host_peak_pss_gib=outcome["host_peak_pss_gib"], cgroup_peak_gib=outcome["cgroup_peak_gib"], gpu_peak_used_mib=outcome["gpu_peak_used_mib"], gpu_mean_utilization_percent=outcome["gpu_mean_utilization_percent"], gpu_samples=outcome["gpu_samples"], checkpoint_bytes={p: (fold_dir/p).stat().st_size for p in checkpoints_in(fold_dir)}, code_commit=code.get("commit"), created_utc=utc())
    write_json_once(measure_path, document)
    if len(progress["epoch_seconds"]) < SMOKE_EPOCHS or outcome["stopped"] != "condition":
        raise Stop(f"the smoke did not reach {SMOKE_EPOCHS} epochs: exit {outcome['returncode']}, epochs {progress['epoch_seconds']}; log {outcome['log']}; kept")
    log(f"measured: epoch seconds {progress['epoch_seconds']}, {document['n_proc_da']} augmentation processes, host peak PSS {outcome['host_peak_pss_gib']:.2f} GiB, GPU used peak {outcome['gpu_peak_used_mib']} MiB, GPU mean utilization {outcome['gpu_mean_utilization_percent']} %")
    log(f"next, in the foreground: {nc.PY} -B {HERE/'run_nn.py'} decide-epochs --out {out}")


def epochs_rule(measured, *, limit_hours=LIMIT_HOURS, host_limit=HOST_LIMIT_GIB, gpu_limit=GPU_LIMIT_MIB):
    """The plan's rule on what the smoke measured: 250 epochs at the median seconds of the epochs from the 2nd on (the first has the warm-up) taking more than limit_hours -> the 100-epoch variant. The budgets decide `ok`."""
    seconds = measured["epoch_seconds"]
    if len(seconds) < SMOKE_EPOCHS:
        return dict(ok=False, reasons=[f"{len(seconds)} epochs measured, {SMOKE_EPOCHS} needed"], trainer=None, epochs=None)
    per_epoch = statistics.median(seconds[1:])
    hours = nc.EPOCHS_LONG*per_epoch/3600
    epochs = nc.EPOCHS_SHORT if hours > limit_hours else nc.EPOCHS_LONG
    reasons = []
    if measured.get("host_peak_pss_gib") is None or measured["host_peak_pss_gib"] > host_limit:
        reasons.append(f"host peak {measured.get('host_peak_pss_gib')} GiB above {host_limit} GiB")
    if measured.get("gpu_peak_used_mib") is None or measured["gpu_peak_used_mib"] > gpu_limit:
        reasons.append(f"GPU used peak {measured.get('gpu_peak_used_mib')} MiB above {gpu_limit} MiB")
    return dict(ok=not reasons, reasons=reasons, trainer=nc.TRAINERS[epochs], epochs=epochs, epoch_seconds_used=per_epoch, epoch_seconds_measured=seconds, hours_for_250=hours, hours_for_chosen=epochs*per_epoch/3600, limit_hours=limit_hours, rule=f"{nc.EPOCHS_LONG} * median seconds of the epochs from the 2nd on / 3600 > {limit_hours}: {nc.TRAINERS[nc.EPOCHS_SHORT]}; else {nc.TRAINERS[nc.EPOCHS_LONG]}", n_proc_da=measured.get("n_proc_da"), host_peak_pss_gib=measured.get("host_peak_pss_gib"), gpu_peak_used_mib=measured.get("gpu_peak_used_mib"), gpu_mean_utilization_percent=measured.get("gpu_mean_utilization_percent"))


def decide_epochs(out, *, code):
    measured = read_json(Path(out)/"smoke_measure.json")
    if measured is None:
        raise Stop(f"{Path(out)/'smoke_measure.json'}: no such file; the smoke comes first")
    return write_json_once(Path(out)/"epochs_decision.json", dict(epochs_rule(measured), decided_utc=utc(), command=sys.argv, code=code))


def read_decision(out):
    """The decision, recomputed from what the smoke measured: a file changed by hand stops the training."""
    document = read_json(Path(out)/"epochs_decision.json")
    if document is None:
        raise Stop(f"{Path(out)/'epochs_decision.json'}: no such file; the smoke and decide-epochs come first")
    if not document.get("ok"):
        raise Stop(f"{Path(out)/'epochs_decision.json'}: the rule did not pass: {document.get('reasons')}")
    measured = read_json(Path(out)/"smoke_measure.json")
    if measured is None:
        raise Stop(f"{Path(out)/'smoke_measure.json'}: no such file; the decision has nothing to be checked against")
    again = epochs_rule(measured)
    different = [k for k in ("ok", "trainer", "epochs", "epoch_seconds_used") if again.get(k) != document.get(k)]
    if different:
        raise Stop(f"{Path(out)/'epochs_decision.json'} does not follow from {Path(out)/'smoke_measure.json'}: {different}")
    return document


def retire_smoke(out, reason):
    """After a smoke that did not pass: its files move to smoke_attempts/<utc>/ with the reason (hard rule 4: kept, not deleted), so that the smoke can run again. A smoke that passed is not retired: the rule is not rolled twice."""
    out = Path(out)
    if not reason or not reason.strip():
        raise ValueError("--reason: why the smoke is run again")
    measured = read_json(out/"smoke_measure.json")
    if measured is not None and epochs_rule(measured)["ok"]:
        raise Stop(f"{out/'smoke_measure.json'}: this smoke passed; it is not run again")
    if (out/"train_runs").exists():
        raise Stop(f"{out/'train_runs'} exists: the training has started")
    target = out/"smoke_attempts"/stamp()
    moved = []
    for relative in ("smoke_measure.json", "epochs_decision.json", nc.smoke_results_root(out).name, "logs/smoke.log"):
        if (out/relative).exists():
            target.mkdir(parents=True, exist_ok=True)
            (out/relative).rename(target/Path(relative).name)
            moved.append(relative)
    if not moved:
        raise Stop(f"{out}: no smoke to retire")
    write_json_once(target/"retired.json", dict(reason=reason, moved=moved, retired_utc=utc()))
    log(f"retired to {target}: {moved}; reason: {reason}")
    return target


# ---- train ----------------------------------------------------------------------------------------------------------------------------------------------

def check_resume(fold_dir, resume):
    """Without --continue nnU-Net would start again and overwrite the run's checkpoints; with --continue and no checkpoint it would print a warning and start a new training."""
    found = checkpoints_in(fold_dir)
    if resume and not found:
        raise Stop(f"--continue: {fold_dir} has no checkpoint")
    if found and not resume:
        raise Stop(f"{fold_dir} has {found}: a run is there. Continue it with --continue; a new training over it is refused")
    return found


def merge_train_runs(out, *, trainer):
    """train_record.json, when checkpoint_final.pth is there. A run killed by RuntimeMaxSec or the memory limit leaves only its started record (the driver dies with it), so the runs are counted from the started records and the time and the epochs read from all training_log_*.txt; the peaks come from the runs that have a record."""
    out = Path(out)
    fold_dir = nc.fold_directory(trainer, results=nc.results_root(out))
    started = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out/"train_runs").glob("*.started.json"))]
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out/"train_runs").glob("*.json")) if not p.name.endswith(".started.json")]
    final = fold_dir/nc.CHECKPOINT
    logs = [p.read_text(encoding="utf-8", errors="replace") for p in sorted(fold_dir.glob("training_log_*.txt"))]
    seconds = [float(v) for text_ in logs for v in EPOCH_TIME.findall(text_)]
    utilizations = [run["outcome"]["gpu_mean_utilization_percent"] for run in runs if run["outcome"].get("gpu_mean_utilization_percent") is not None]
    manifest = read_json(out/"prepare_manifest.json") or {}
    contact = manifest.get("label_contact_subjects", [])
    return write_json_once(out/"train_record.json", dict(trainer=trainer, epochs=started[-1]["epochs"], fold=nc.FOLD, configuration=nc.CONFIGURATION, wall_seconds=sum(log_span_seconds(text_) for text_ in logs), wall_source="training_log_*.txt: first to last timestamp of each, summed", driver_wall_seconds_recorded=sum(run["outcome"]["wall_seconds"] for run in runs), epochs_logged=len(seconds), epochs_logged_note="the Epoch time lines of all logs: an epoch redone after a crash counts again, so this can exceed `epochs`", epoch_seconds_median=statistics.median(seconds) if seconds else None,
                                                        host_peak_pss_gib=max(run["outcome"]["host_peak_pss_gib"] for run in runs), gpu_peak_used_mib=max((run["outcome"]["gpu_peak_used_mib"] or 0) for run in runs), gpu_mean_utilization_percent=statistics.mean(utilizations) if utilizations else None, runs_started=len(started), resumed_runs=len(started)-1, runs_without_record=len(started)-len(runs), stopped_before_final_validation=runs[-1]["stopped_before_final_validation"],
                                                        final_checkpoint=str(final), final_checkpoint_sha256=file_sha256(final), final_checkpoint_bytes=final.stat().st_size, n_label_contact=len(contact), label_contact_subjects_sha256=hashlib.sha256("\n".join(contact).encode("utf-8")).hexdigest(), compile=False, created_utc=utc()))


def train_once(out, trainer, *, epochs, resume, final_validation, env, code_commit=None, device="cuda", stop_hook=None, first_signal=signal.SIGTERM, poll=WATCH_SECONDS):
    """One run of nnUNetv2_train: a started record first (a run killed from outside leaves that much), the run, its record, and train_record.json if checkpoint_final.pth is there. Stop when the run failed. stop_hook and first_signal are for sandbox_check.py's crash; device for its CPU."""
    out = Path(out)
    fold_dir = nc.fold_directory(trainer, results=nc.results_root(out))
    check_resume(fold_dir, resume)
    since = time.time()-LOG_MARGIN_SECONDS
    should_stop = stop_hook or (None if final_validation else (lambda: training_progress(fold_dir, since)["validating"]))
    name = stamp()
    base = dict(trainer=trainer, epochs=epochs, resumed=resume, code_commit=code_commit, started_utc=utc())
    write_json_once(out/"train_runs"/f"{name}.started.json", base)
    argv = nc.train_argv(trainer, resume=resume, device=device)
    log(f"train: {shlex.join(argv)}")
    outcome = run_watched(argv, env=env, log_path=out/"logs"/f"train_{name}.log", cwd=out, should_stop=should_stop, sample_gpu=device == "cuda", first_signal=first_signal, poll=poll)
    progress = training_progress(fold_dir, since)
    stopped_before_validation = outcome["stopped"] == "condition" and progress["validating"]
    write_json_once(out/"train_runs"/f"{name}.json", dict(base, outcome=outcome, epoch_seconds=progress["epoch_seconds"], stopped_before_final_validation=stopped_before_validation))
    if not stopped_before_validation and outcome["returncode"] != 0:
        raise Stop(f"nnUNetv2_train failed (exit {outcome['returncode']}); log {outcome['log']}; kept. To go on: run_nn.py train --continue")
    if not (fold_dir/nc.CHECKPOINT).is_file():
        raise Stop(f"{fold_dir/nc.CHECKPOINT}: no such file after the run; log {outcome['log']}; kept")
    return merge_train_runs(out, trainer=trainer)


def cmd_train(args):
    code = require_snapshot()
    out = Path(args.out)
    set_run_log(out)
    decision = read_decision(out)
    if (out/"train_record.json").exists():
        raise Stop(f"{out/'train_record.json'} exists; the training has finished")
    require_preprocessed(out, code=code)
    nn_lock.check()                         # not hours of training for an evaluation that the lock will refuse
    trainer = decision["trainer"]
    check_resume(nc.fold_directory(trainer, results=nc.results_root(out)), args.resume)
    wait_for_gate(UNITS["train"], args.wait_minutes)
    env = nc.nn_env(out, gpu=True)
    env_record(out, env=env)
    record = train_once(out, trainer, epochs=decision["epochs"], resume=args.resume, final_validation=args.final_validation, env=env, code_commit=code.get("commit"))
    log(f"finished: {record['epochs']} epochs, {record['wall_seconds']/3600:.2f} h, median epoch {record['epoch_seconds_median']} s, {out/'train_record.json'}")


# ---- infer -----------------------------------------------------------------------------------------------------------------------------------------------

def admitted_dev_subjects(out, *, code):
    """The 20 development subjects, each checked before anything of them is opened, and outside the model's label contact. Ids only."""
    import eval_seg
    from bayesreg.eval.truth import load_sealed_lists
    from bayesreg.io.guard import SubjectSealed, load_dev_allowlist
    from bayesreg.seg.unet.data import load_folds
    from bayesreg.seg.unet.pins import source_root_for
    subjects = eval_seg.admit_subjects(load_dev_allowlist(eval_seg.ALLOWLIST), eval_seg.DEV_SOURCE, load_sealed_lists(source_root_for(code)), load_folds(eval_seg.FOLDS).training_subjects)
    manifest = read_json(Path(out)/"prepare_manifest.json") or {}
    overlap = sorted(set(subjects) & set(manifest.get("label_contact_subjects", [])))
    if not manifest.get("label_contact_subjects") or overlap:
        raise SubjectSealed(f"the label contact of the model is missing or holds development subjects: {overlap}")
    return subjects


def build_dev_input(directory, subjects, data_root):
    """<directory>/<subject>_0000.nii.gz -> the image. Images only, no copy; a directory that holds anything else is refused."""
    from bayesreg.seg.unet.data import IMAGE_FILE
    directory = Path(directory)
    expected = {f"{s}_0000.nii.gz": Path(data_root)/s/IMAGE_FILE for s in subjects}
    if directory.exists():
        found = {p.name: Path(os.readlink(p)) if p.is_symlink() else None for p in directory.iterdir()}
        if found != expected:
            raise Stop(f"{directory} holds something else than the links to the images of the development subjects")
        return expected
    directory.mkdir(parents=True)
    for name, target in expected.items():
        if not target.is_file():
            raise Stop(f"{target}: no such file")
        os.symlink(target, directory/name)
    return expected


def cmd_infer(args):
    code = require_snapshot()
    out = Path(args.out)
    set_run_log(out)
    record = read_json(out/"train_record.json")
    if record is None:
        raise Stop(f"{out/'train_record.json'}: no such file; the training has not finished")
    nn_lock.check()
    if (out/"infer_host.json").exists():
        raise Stop(f"{out/'infer_host.json'} exists; the inference ran")
    require_preprocessed(out, code=code)
    subjects = admitted_dev_subjects(out, code=code)
    if file_sha256(record["final_checkpoint"]) != record["final_checkpoint_sha256"]:
        raise Stop(f"{record['final_checkpoint']}: sha256 differs from train_record.json")
    build_dev_input(out/"dev_input", subjects, nc.DATA_ROOT)
    wait_for_gate(UNITS["infer"], args.wait_minutes)
    env = nc.nn_env(out, gpu=True)
    env_record(out, env=env)
    argv = nc.infer_argv(model_dir=nc.trainer_directory(record["trainer"], results=nc.results_root(out)), input_dir=out/"dev_input", out_dir=out/"pred_dev")
    outcome = run_watched(argv, env=env, log_path=out/"logs"/"infer.log", cwd=out, sample_gpu=True)
    write_json_once(out/"infer_host.json", outcome)
    if outcome["returncode"] != 0:
        raise Stop(f"infer_nn.py failed (exit {outcome['returncode']}); log {outcome['log']}; kept, not rerun")
    log(f"predictions: {out/'pred_dev'}; wall {outcome['wall_seconds']:.0f} s, host peak PSS {outcome['host_peak_pss_gib']:.2f} GiB")


# ---- eval -------------------------------------------------------------------------------------------------------------------------------------------------

def eval_argv(out, *, bz_go=None, cwd=HERE):
    return [nc.PY, "-B", str(Path(cwd)/"eval_nn.py"), "--pred-root", str(Path(out)/"pred_dev"), "--manifest", str(Path(out)/"prepare_manifest.json"), "--out-dir", str(Path(out)/"eval_nn"), *(["--bz-go", bz_go] if bz_go else [])]


def decide_argv(out, *, u1_eval=U1_EVAL, cwd=HERE):
    return [nc.PY, "-B", str(Path(cwd)/"decide_nn.py"), "--eval", str(Path(out)/"eval_nn"/"eval_nn.json"), "--train-record", str(Path(out)/"train_record.json"), "--infer-record", str(Path(out)/"pred_dev"/"infer_record.json"), "--u1-eval", str(u1_eval), "--out-dir", str(Path(out)/"analysis")]


def cmd_eval(args):
    require_snapshot()
    out = Path(args.out)
    set_run_log(out)
    nn_lock.check(args.bz_go)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    (out/"logs").mkdir(parents=True, exist_ok=True)
    for name, argv in (("eval_nn", eval_argv(out, bz_go=args.bz_go)), ("decide_nn", decide_argv(out))):
        log(f"start {name}: {shlex.join(argv)}")
        with (out/"logs").joinpath("eval.log").open("a", encoding="utf-8") as stream:
            result = subprocess.run(argv, cwd=str(REPO), env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise Stop(f"{name} failed (exit {result.returncode}); log {out/'logs'/'eval.log'}; kept, not rerun")
    log(f"finished: {out/'analysis'/'decision_nn.json'}, summary_nn.md")


# ---- status -----------------------------------------------------------------------------------------------------------------------------------------------

def cmd_status(args):
    out = Path(args.out)
    for name, keys in (("prepare_manifest.json", ("n_subjects", "shapes", "spacings")), ("preprocess_record.json", ("n_cases", "plan")), ("gt_links_record.json", ("n_links", "replaced", "bytes_freed")), ("smoke_measure.json", ("epoch_seconds", "n_proc_da", "host_peak_pss_gib", "gpu_peak_used_mib", "gpu_mean_utilization_percent")), ("epochs_decision.json", ("ok", "reasons", "trainer", "epoch_seconds_used", "hours_for_250", "hours_for_chosen")), ("train_record.json", ("trainer", "wall_seconds", "epochs_logged", "epoch_seconds_median", "resumed_runs", "runs_without_record", "final_checkpoint_sha256")), ("pred_dev/infer_record.json", ("trainer", "tta_views", "seconds_median", "seconds_max")), ("analysis/decision_nn.json", ("reading_text", "delta_seg35", "delta_seg4", "not_decidable_because"))):
        document = read_json(out/name)
        print(f"{name}:", json.dumps({k: document.get(k) for k in keys}, ensure_ascii=False) if document is not None else "(none)")
    for trainer in nc.TRAINERS.values():
        fold_dir = nc.fold_directory(trainer, results=nc.results_root(out))
        if fold_dir.is_dir():
            seconds = training_progress(fold_dir, 0.)["epoch_seconds"]
            print(f"{trainer}: {len(seconds)} epochs logged, last {seconds[-1:]} s, checkpoints {checkpoints_in(fold_dir)}")
    print(f"read lock {nn_lock.DEV_READ_LOCK}:", json.dumps(nn_lock.describe(), ensure_ascii=False) if nn_lock.DEV_READ_LOCK.exists() else "(none)")


# ---- dry run ----------------------------------------------------------------------------------------------------------------------------------------------

def rehearse_rules():
    """The two rules on made-up numbers: no subject, no file of a subject."""
    import decide_nn
    return dict(epochs={f"{s:g} s/epoch": epochs_rule(dict(epoch_seconds=[s+50, s, s, s], host_peak_pss_gib=6., gpu_peak_used_mib=8000))["trainer"] for s in (70., 144., 144.1, 200.)}, reading={f"{d:+.4f}": decide_nn.READINGS[decide_nn.reading_of(d)] for d in (.006, .005, .0035, .002, .001, -.003)})


def lists_on_the_repository(code):
    """Ids only: the real training list and the real development list, through the same checks as the real steps."""
    import eval_seg
    import prepare_nn
    from bayesreg.eval.truth import load_sealed_lists
    from bayesreg.io.guard import load_dev_allowlist
    from bayesreg.seg.unet.data import load_folds
    from bayesreg.seg.unet.pins import source_root_for
    admitted = prepare_nn.admitted_training_subjects(code=code)
    subjects = eval_seg.admit_subjects(load_dev_allowlist(eval_seg.ALLOWLIST), eval_seg.DEV_SOURCE, load_sealed_lists(source_root_for(code)), load_folds(eval_seg.FOLDS).training_subjects)
    return dict(training=len(admitted["subjects"]), checked_against=admitted["list_sizes"], development=len(subjects), development_in_training=len(set(subjects) & set(admitted["subjects"])))


def synthetic_chain(work):
    """eval_nn and decide_nn on the synthetic tree of experiments/unv2 (ids OASIS_OAS1_09xx_MR1): the argmax files are the truth with a few voxels changed, the control is the tree's prediction. Two subjects: the rule must say not decidable."""
    import eval_nn
    import decide_nn
    import eval_seg
    import nibabel as nib
    import numpy as np
    import synthetic_tree
    work = Path(work)
    synthetic_tree.build(work)
    data_root = work/"data"
    subjects = synthetic_tree.DEV
    lookup = eval_seg.load_seg4_lookup()
    rows = {}
    for subject in subjects:
        top, _ = eval_seg.read_prediction(work/"pred_v1", subject, expected_segmenter="unet35_v1_ens")
        rows[subject] = dict(v1=eval_seg.summarize_subject(top, eval_seg.read_truth(data_root, subject, "seg35")[0], eval_seg.read_truth(data_root, subject, "seg4")[0], lookup))
        rows[subject]["v2"] = rows[subject]["v1"]
        truth = nib.load(data_root/subject/"aligned_seg35.nii.gz")
        labels = np.asarray(truth.dataobj).astype(np.uint8)
        labels[::7, ::7, ::7] = 0
        (work/"pred_nn").mkdir(exist_ok=True)
        nib.save(nib.Nifti1Image(labels, truth.affine), work/"pred_nn"/f"{subject}.nii.gz")
    means = {arm: {m: float(np.mean([rows[s][arm][m] for s in subjects])) for m in ("seg35", "seg4")} for arm in ("v1", "v2")}
    (work/"reference.json").write_text(json.dumps(dict(subjects=subjects, rows=rows, means=means)), encoding="utf-8")
    (work/"manifest.json").write_text(json.dumps(dict(label_contact_subjects=synthetic_tree.TRAINING)), encoding="utf-8")
    document = eval_nn.evaluate(pred_root=work/"pred_nn", manifest=work/"manifest.json", out_dir=work/"eval_nn", v2_root=work/"pred_v1", control_segmenter="unet35_v1_ens", reference_path=work/"reference.json", data_root=data_root, allowlist_path=work/"dev_subjects.json", folds_path=work/"folds.json", dev_source="synthetic_dev", synthetic=True, argv=["dry-run"], log=lambda _: None)
    record = dict(trainer=nc.TRAINERS[nc.EPOCHS_LONG], final_checkpoint_sha256="x"*64)
    inferred = dict(trainer=record["trainer"], fold=nc.FOLD, checkpoint=nc.CHECKPOINT, configuration=nc.CONFIGURATION, checkpoint_sha256=record["final_checkpoint_sha256"], allowed_mirroring_axes=None, tta_views=1)
    return decide_nn.decide(document, record, inferred), document


def dry_run(scratch=None):
    work = Path(scratch) if scratch else Path(tempfile.mkdtemp(prefix="nn_dryrun_"))
    work.mkdir(parents=True, exist_ok=True)
    print("== the lead's start strings (printed, not run)")
    print("prepare:", shlex.join([nc.PY, "-B", "experiments/nn/run_nn.py", "prepare"]))
    print("preprocess:", shlex.join([nc.PY, "-B", "experiments/nn/run_nn.py", "preprocess"]))
    print("link-gt:", shlex.join([nc.PY, "-B", "experiments/nn/run_nn.py", "link-gt"]))
    print("snapshot:", shlex.join([nc.PY, "-B", "experiments/nn/run_nn.py", "snapshot"]))
    for step in ("smoke", "train", "train-continue", "infer", "eval"):
        print(f"{step}:", shlex.join(launch_argv(step)))
    print("decide-epochs:", shlex.join([nc.PY, "-B", str(nc.OUT/"snapshot/experiments/nn/run_nn.py"), "decide-epochs"]))
    print("retire-smoke, after a smoke that did not pass:", shlex.join([nc.PY, "-B", str(nc.OUT/"snapshot/experiments/nn/run_nn.py"), "retire-smoke", "--reason", "<why>"]))
    print("\n== the nnU-Net commands (printed, not run)")
    print("plan_and_preprocess:", shlex.join(nc.plan_argv()))
    print("train:", shlex.join(nc.train_argv(nc.TRAINERS[nc.EPOCHS_LONG])))
    print("train, the short variant:", shlex.join(nc.train_argv(nc.TRAINERS[nc.EPOCHS_SHORT])))
    print("continue:", shlex.join(nc.train_argv(nc.TRAINERS[nc.EPOCHS_LONG], resume=True)))
    print("infer:", shlex.join(nc.infer_argv(model_dir=nc.trainer_directory(nc.TRAINERS[nc.EPOCHS_LONG], results=nc.results_root()), input_dir=nc.OUT/"dev_input", out_dir=nc.OUT/"pred_dev")))
    print("eval:", shlex.join(eval_argv(nc.OUT, cwd=nc.OUT/"snapshot/experiments/nn")))
    print("decide:", shlex.join(decide_argv(nc.OUT, cwd=nc.OUT/"snapshot/experiments/nn")))
    env = nc.nn_env(nc.OUT, gpu=True)
    print("environment:", json.dumps({k: v for k, v in sorted(env.items()) if k.startswith("nnUNet") or k in ("PATH", "PYTHONNOUSERSITE", "OMP_NUM_THREADS")}, indent=1))
    print("\n== the gate on this machine now (read only; it decides nothing here)")
    print(json.dumps(gate_problems(UNITS["smoke"]), indent=1))
    print("\n== real steps refused here: smoke, train, infer, eval need a verified snapshot")
    try:
        require_snapshot()
        print("NOT REFUSED")
        refused_here = False
    except CodeNotClean as exc:
        print(f"refused: {str(exc)[:160]}")
        refused_here = True
    print("\n== the rules on made-up numbers")
    rehearsal = rehearse_rules()
    print(json.dumps(rehearsal, indent=1, ensure_ascii=False))
    print("\n== subjects, on the repository's lists (ids only)")
    from bayesreg.seg.unet.snapshot import code_record
    lists = lists_on_the_repository(code_record())
    print(json.dumps(lists, indent=1))
    print(f"\n== the read lock of the development truth, {nn_lock.DEV_READ_LOCK}: {'EXISTS' if nn_lock.DEV_READ_LOCK.exists() else 'none'} (synthetic runs do not touch it)")
    print("\n== eval_nn and decide_nn on synthetic subjects")
    decision, document = synthetic_chain(work)
    print("means:", json.dumps({a: {m: document["means"][a][m] for m in ("seg35", "seg4")} for a in ("nn", "v2_control", "reference_v2")}))
    print("control:", json.dumps(document["control"]))
    print("decision:", json.dumps({k: decision.get(k) for k in ("reading", "reading_text", "not_decidable_because")}, ensure_ascii=False))
    ok = refused_here and document["control"]["passed"] and decision["reading"] == "not_decidable" and decision["not_decidable_because"] == ["2 名受试者，不是 20"] and lists["training"] == 106 and lists["development"] == 20 and lists["development_in_training"] == 0
    print(f"\n== dry run {'ok' if ok else 'NOT OK'}; work {work}")
    return 0 if ok else 8


# ---- command line ---------------------------------------------------------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scratch", type=Path, default=None)
    sub = parser.add_subparsers(dest="command")
    for name in ("prepare", "link-gt", "decide-epochs", "status"):
        sub.add_parser(name).add_argument("--out", type=Path, default=nc.OUT)
    p = sub.add_parser("retire-smoke")
    p.add_argument("--out", type=Path, default=nc.OUT)
    p.add_argument("--reason", required=True)
    p = sub.add_parser("preprocess")
    p.add_argument("--out", type=Path, default=nc.OUT)
    p.add_argument("--wait-minutes", type=int, default=PREPROCESS_WAIT_MINUTES)
    p = sub.add_parser("snapshot")
    p.add_argument("--repo", type=Path, default=nc.MAIN_REPO)
    p.add_argument("--out", type=Path, default=nc.OUT)
    p = sub.add_parser("print-launch")
    p.add_argument("step", choices=["smoke", "train", "train-continue", "infer", "eval"])
    p.add_argument("--out", type=Path, default=nc.OUT)
    p.add_argument("--wait-minutes", type=int, default=WAIT_MINUTES)
    p.add_argument("--bz-go", default=None)
    p = sub.add_parser("smoke")
    p.add_argument("--out", type=Path, default=nc.OUT)
    p.add_argument("--wait-minutes", type=int, default=WAIT_MINUTES)
    p = sub.add_parser("train")
    p.add_argument("--out", type=Path, default=nc.OUT)
    p.add_argument("--wait-minutes", type=int, default=WAIT_MINUTES)
    p.add_argument("--continue", dest="resume", action="store_true", help="nnUNetv2_train --c")
    p.add_argument("--final-validation", action="store_true", help="let nnU-Net predict the 106 training cases at the end")
    p = sub.add_parser("infer")
    p.add_argument("--out", type=Path, default=nc.OUT)
    p.add_argument("--wait-minutes", type=int, default=WAIT_MINUTES)
    p = sub.add_parser("eval")
    p.add_argument("--out", type=Path, default=nc.OUT)
    p.add_argument("--bz-go", default=None, help="BZ's words and the time; see eval_nn.py")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from bayesreg.io.guard import SubjectSealed
    import prepare_nn
    try:
        if args.dry_run:
            return dry_run(args.scratch)
        if args.command == "prepare":
            set_run_log(args.out)
            prepare_nn.prepare(args.out, log=log)
        elif args.command == "preprocess":
            cmd_preprocess(args)
        elif args.command == "link-gt":
            cmd_link_gt(args)
        elif args.command == "retire-smoke":
            set_run_log(args.out)
            retire_smoke(args.out, args.reason)
        elif args.command == "snapshot":
            set_run_log(args.out)
            record = make_snapshot(args.repo, Path(args.out)/"snapshot", paths=SNAPSHOT_PATHS_NN)
            verify_snapshot(Path(args.out)/"snapshot")
            log(f"snapshot {Path(args.out)/'snapshot'} made from commit {record['commit']}")
        elif args.command == "print-launch":
            print(shlex.join(launch_argv(args.step, out=args.out, wait_minutes=args.wait_minutes, bz_go=args.bz_go)))
        elif args.command == "smoke":
            cmd_smoke(args)
        elif args.command == "decide-epochs":
            document = decide_epochs(args.out, code=require_snapshot())
            print(json.dumps({k: document.get(k) for k in ("ok", "reasons", "trainer", "epochs", "epoch_seconds_used", "hours_for_250", "hours_for_chosen", "n_proc_da", "gpu_mean_utilization_percent")}, indent=1))
            if not document["ok"]:
                return 8
        elif args.command == "train":
            cmd_train(args)
        elif args.command == "infer":
            cmd_infer(args)
        elif args.command == "eval":
            cmd_eval(args)
        elif args.command == "status":
            cmd_status(args)
        else:
            build_parser().print_help()
            return 2
    except (Stop, CodeNotClean, nn_lock.DevReadLocked, prepare_nn.SubjectsRefused, prepare_nn.RawRefused, SubjectSealed) as exc:
        log(f"STOP: {exc}")
        return 8
    return 0


if __name__ == "__main__":
    sys.exit(main())
