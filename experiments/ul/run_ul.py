"""Driver of work package UL (runs/20261104-ul/plan.md; docs/bayesreg/04_分割器.md §15). Everything real is started by the lead, from main, with the printed strings; this file starts no unit.

    python -B experiments/ul/run_ul.py --dry-run [--scratch DIR]
        the launch strings and the real chain's commands (printed, not run), the gate's reading, a rehearsal of the rule on the committed numbers, the refusals, and the whole chain on synthetic subjects on the CPU. No GPU, no unit, nothing written outside the scratch directory.
    python -B experiments/ul/run_ul.py prepare [--repo DIR]
        foreground, from a clean repository: the code snapshot <out>/snapshot (made once, verified after), `cli check-inputs --profile v3_large --require-snapshot` from it (no label or image is read), the environment record. No GPU.
    python -B experiments/ul/run_ul.py print-launch {smoke,train} [--patch Z,Y,X]
    python -B <snapshot>/experiments/ul/run_ul.py smoke [--patch Z,Y,X] [--steps 60]       in the unit tag-ul-smoke
        behind the gate: the training entry on the real fold A (training subjects, through the label guard) for a few dozen steps, into <out>/smoke_<patch>; step times, GPU and host peaks. It writes no member anything else uses. Then, once, the control of the evaluator (eval_u1.py --control-only, <out>/control_check): member A0 only, so that the real path
        of the evaluation is seen working, and reproducing the reference, before the training; it takes no read lock.
    python -B experiments/ul/run_ul.py decide-steps [--patch Z,Y,X]
        foreground, no GPU: from what the smoke measured, by the plan's rule: the step count and the limits, <out>/steps_decision_<patch>.json and, if it passes, <out>/steps_decision.json (written once; `train` reads only that file).
    python -B <snapshot>/experiments/ul/run_ul.py train                                     in the unit tag-ul-train
        refuses first if steps_decision.json or a passed control is missing, or the read lock of the development truth exists (dev_lock.py: it would refuse the evaluation 4.5 hours later). Then behind the gate: train (fold A, seed 0, the steps of steps_decision.json) -> calibrate -> check-members -> eval_u1 (the 20 development subjects) -> decide_ul
        (analysis/decision_ul.json, summary_ul.md). A step that started and did not finish stops the chain and is kept whole, not rerun.
    python -B experiments/ul/run_ul.py status

The gate: no other tag-* or seq-vp-* unit active, 1-minute load < 1.5, no GPU compute process, GPU utilization <= 20 %, GPU free >= 21 GiB (the peak of this run is capped at 20 GiB), host MemAvailable >= 9 GiB, /home free >= 100 GB, /mnt/c free >= 15 GiB; polled every 60 s, closed for --wait-minutes: nothing starts, exit 8.
Logs: the steps' output and commands in <out>/chain.log (the smoke's in its own directory); the gate's waits, the steps and every STOP also in <out>/run_ul.log. The units have a RuntimeMaxSec, so a call that hangs does not keep the gates of the other queues closed for ever.
Exit codes: 0 done, 2 usage, 8 a stop (the reason is printed).
"""
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(HERE), str(HERE.parent/"unv2")]

import dev_lock  # noqa: E402
from bayesreg.seg.unet.large import PATCH_U1  # noqa: E402
from bayesreg.seg.unet.snapshot import CODE_ROOT, CodeNotClean, SNAPSHOT_PROFILES, make_snapshot, require_snapshot, verify_snapshot  # noqa: E402

PROFILE = "v3_large"
PY = "/home/rmcfz/miniconda3/envs/sci310/bin/python"
MAIN_REPO = Path("/home/rmcfz/code/python_projects/seq_vp_tag")
RESULTS = Path("/home/rmcfz/seq_vp_tag_results")
OUT = RESULTS/"seg"/"unet35_v3_large"
UNITS = dict(smoke="tag-ul-smoke", train="tag-ul-train")
FOLD, SEED, MEMBER = "A", 0, "foldA_s0"
SMI = "/usr/lib/wsl/lib/nvidia-smi"
MIN_GPU_FREE_MIB, MAX_GPU_UTIL, MAX_LOAD, MIN_HOME_GB, MIN_C_GIB, MIN_AVAILABLE_GIB = 21504, 20, 1.5, 100, 15, 9
POLL_SECONDS, WAIT_MINUTES = 60, 30
GPU_LIMIT_GIB, HOST_LIMIT_GIB, GPU_RESERVED_LIMIT_GIB = 20., 8., 19.     # reserved: the allocator raises at the cap (20 GiB), so a smoke that comes this close is too close for a run 500 times longer
RUNTIME_MAX_SECONDS = dict(smoke=3600, train=20700)        # train: 5 h + 30 minutes of gate + 15 minutes of margin; a hung call must not keep the gates of the other queues closed
CONTROL_DIR = "control_check"
BUDGET_HOURS = 5
RESERVE_SECONDS = 1800                  # calibration, member check, evaluation, decision, the margin of the last step; 04_分割器.md §14.8 has 11-13 s of calibration on v1
TRAIN_WALL_SECONDS = BUDGET_HOURS*3600-RESERVE_SECONDS
SAFETY = .9                             # the step count aims at 90 % of the training window; the wall stop (max_wall_seconds) is the limit
MIN_STEPS = 15000                       # fewer: the smoke's rule refuses, the lead decides (v2 ran 20000 steps of 6.9 M voxels; U1 at the estimate about 30000 of 4.6 M)
EVAL_EVERY = 500
SMOKE_STEPS, SKIP_STEPS = 60, 10
THREADS = 4                             # as experiments/unv2/chain.sh


class Stop(Exception):
    """A stop of this driver: exit 8, the reason printed, nothing rerun."""


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


LOG_FILE = None                         # set by the real steps and the dry run


def set_run_log(out):
    global LOG_FILE
    Path(out).mkdir(parents=True, exist_ok=True)
    LOG_FILE = Path(out)/"run_ul.log"


def log(message):
    line = f"{utc()} {message}"
    print(line, flush=True)
    if LOG_FILE is not None:
        with LOG_FILE.open("a", encoding="utf-8") as stream:
            stream.write(line+"\n")


def patch_name(patch):
    return "x".join(str(p) for p in patch)


def parse_patch(text):
    parts = tuple(int(v) for v in text.split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"{text}: Z,Y,X")
    return parts


def smoke_root(out, patch):
    return Path(out)/f"smoke_{patch_name(patch)}"


# ---- the start strings -------------------------------------------------------------------------------------------------------------------------------

def check_launch(argv):
    """The rules of a start string (the plan's brief, section 4): no test runner in it, the unit's prefix, the results root, the host limit."""
    text = " ".join(str(a) for a in argv)
    if "pytest" in text:
        raise ValueError(f"a start string must not contain the test runner: {text}")
    units = [a for a in argv if str(a).startswith("--unit=")]
    if len(units) != 1 or not units[0].startswith("--unit=tag-ul-"):
        raise ValueError(f"the unit must be tag-ul-*: {units}")
    if "seq-vp-" in text:
        raise ValueError("a unit of this project never uses the seq-vp- prefix")
    if not any(str(a).startswith(f"--working-directory={RESULTS}/") for a in argv):
        raise ValueError(f"the working directory is the snapshot under {RESULTS}")
    if "--property=MemoryMax=8G" not in argv:
        raise ValueError("MemoryMax=8G: the host budget of this run")
    if not any(str(a).startswith("--property=RuntimeMaxSec=") for a in argv):
        raise ValueError("RuntimeMaxSec: a hung call must not keep the gates of the other queues closed")
    return [str(a) for a in argv]


def launch_argv(step, *, out=OUT, patch=None, wait_minutes=WAIT_MINUTES):
    """The lead's start of a step, as a list; printed by print-launch, never run by this file."""
    if step not in UNITS:
        raise ValueError(f"no step {step}")
    snapshot = Path(out)/"snapshot"
    extra = ["--patch", ",".join(str(p) for p in patch)] if patch is not None and step == "smoke" else []
    return check_launch(["systemd-run", "--user", f"--unit={UNITS[step]}", "--collect", f"--working-directory={snapshot}", "--property=MemoryMax=8G", "--property=MemorySwapMax=2G", f"--property=RuntimeMaxSec={RUNTIME_MAX_SECONDS[step]}", "--setenv=PYTHONDONTWRITEBYTECODE=1",
                         *[f"--setenv={name}={THREADS}" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")], PY, "-B", str(snapshot/"experiments/ul/run_ul.py"), step, "--out", str(out), "--wait-minutes", str(wait_minutes), *extra])


# ---- the gate ------------------------------------------------------------------------------------------------------------------------------------------

def mem_available_gib(path="/proc/meminfo"):
    for line in Path(path).read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])/2**20
    raise ValueError(f"{path}: no MemAvailable")


def _disk_gib(path, disk):
    return disk(path).free/2**30


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
    try:
        home, c_drive = _disk_gib("/home", disk)*2**30/1e9, _disk_gib("/mnt/c", disk)
        if home < MIN_HOME_GB:
            problems.append(f"/home free {home:.0f} GB < {MIN_HOME_GB} GB")
        if c_drive < MIN_C_GIB:
            problems.append(f"/mnt/c free {c_drive:.0f} GiB < {MIN_C_GIB} GiB")
    except OSError as exc:
        problems.append(f"disk query failed: {exc}")
    host = available()
    if host < MIN_AVAILABLE_GIB:
        problems.append(f"MemAvailable {host:.1f} GiB < {MIN_AVAILABLE_GIB} GiB")
    return problems


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


# ---- the steps -----------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class Spec:
    """What differs between the real chain and the synthetic dry run."""
    out: Path
    cwd: Path
    python: str = PY
    device: str = "cuda"
    inputs: list = field(default_factory=lambda: [])                # options of cli after --profile (real: none, the profile's pins)
    eval_extra: list = field(default_factory=lambda: [])            # options of eval_u1
    tiny: dict = field(default_factory=dict)                        # TrainConfigLarge overrides of the dry run


def cli_argv(spec):
    return [spec.python, "-B", "-m", "bayesreg.seg.unet.cli"]


def sets(config):
    out = []
    for key, value in config.items():
        out += ["--set", f"{key}={json.dumps(value)}"]
    return out


def train_argv(spec, root, config):
    return cli_argv(spec)+["train", "--profile", PROFILE, "--fold", FOLD, "--seed", str(SEED), *spec.inputs, "--out", str(root), "--device", spec.device, *sets(config)]


def smoke_config(patch, steps):
    return dict(patch_size=list(patch), max_iters=steps, eval_every=max(1, steps//2), warmup_iters=min(10, steps), min_iters=steps, log_every=1, patience=steps)


def control_argv(spec):
    return [spec.python, "-B", str(Path(spec.cwd)/"experiments/ul/eval_u1.py"), "--control-only", "--out-dir", str(Path(spec.out)/CONTROL_DIR), "--device", spec.device, *spec.eval_extra]


def chain_steps(spec, config):
    """(name, argv) of the chain, in order. Nothing else runs and each runs once."""
    out = Path(spec.out)
    return [("train_A0", train_argv(spec, out, config)),
            ("calibrate_A0", cli_argv(spec)+["calibrate", "--profile", PROFILE, "--fold", FOLD, "--seed", str(SEED), *spec.inputs, "--out", str(out), "--device", spec.device]),
            ("check_members", cli_argv(spec)+["check-members", "--profile", PROFILE, *spec.inputs, "--out", str(out)]),
            ("eval_u1", [spec.python, "-B", str(Path(spec.cwd)/"experiments/ul/eval_u1.py"), "--member-root", str(out), "--out-dir", str(out/"eval_u1"), "--device", spec.device, *spec.eval_extra]),
            ("decide_ul", [spec.python, "-B", str(Path(spec.cwd)/"experiments/ul/decide_ul.py"), "--eval", str(out/"eval_u1"/"eval_u1.json"), "--member", str(out/MEMBER), "--steps", str(out/"steps_decision.json"), "--out-dir", str(out/"analysis")])]


def run_step(name, argv, *, state, log_path, cwd, run=subprocess.run, env=None):
    """Once: a finished step is skipped, a step that started and did not finish stops the chain and is kept whole."""
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    done, started = state/f"{name}.done", state/f"{name}.started"
    if done.exists():
        log(f"skip {name} (done)")
        return
    if started.exists():
        raise Stop(f"{name} started at {started.read_text().strip()} and did not finish; kept whole, not rerun")
    started.write_text(utc()+"\n")
    log(f"start {name}: {shlex.join(argv)}")
    with Path(log_path).open("a", encoding="utf-8") as stream:
        stream.write(f"{utc()} start {name}: {shlex.join(argv)}\n")
        stream.flush()
        result = run(argv, cwd=str(cwd), stdout=stream, stderr=subprocess.STDOUT, env=env)
        stream.write(f"{utc()} end {name}: exit {result.returncode}\n")
    if result.returncode != 0:
        raise Stop(f"{name} failed (exit {result.returncode}); log {log_path}; kept, not rerun")
    done.write_text(utc()+"\n")
    log(f"done {name}")


def run_chain(spec, config, *, run=subprocess.run, env=None):
    out = Path(spec.out)
    for name, argv in chain_steps(spec, config):
        run_step(name, argv, state=out/"chain_state", log_path=out/"chain.log", cwd=spec.cwd, run=run, env=env)


# ---- the smoke's rule ------------------------------------------------------------------------------------------------------------------------------------

def read_smoke(member_directory):
    """What the training entry wrote: per-step seconds after the first SKIP_STEPS, the evaluations, the peaks. Stop if the run did not end."""
    directory = Path(member_directory)
    path = directory/"train_log.jsonl"
    if not path.is_file():
        raise Stop(f"{path}: no such file; the smoke run did not start or has no log")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    start = next((r for r in rows if r["event"] == "start"), None)
    end = next((r for r in rows if r["event"] == "end"), None)
    steps = [r for r in rows if r["event"] == "train"]
    evals = [r for r in rows if r["event"] == "eval"]
    if start is None or end is None or not evals or len(steps) <= SKIP_STEPS:
        raise Stop(f"{path}: the smoke run did not finish (start {start is not None}, end {end is not None}, {len(steps)} steps, {len(evals)} evaluations)")
    record = json.loads((directory/"train_record.json").read_text(encoding="utf-8"))
    seconds = [r["step_seconds"] for r in steps if r["it"] >= SKIP_STEPS]
    memory = record["memory"]
    return dict(step_seconds_median=statistics.median(seconds), step_seconds_p90=sorted(seconds)[int(.9*(len(seconds)-1))], n_steps_timed=len(seconds), eval_seconds=[r["eval_seconds"] for r in evals], eval_seconds_median=statistics.median(r["eval_seconds"] for r in evals), data_ready_seconds=start["data_ready_seconds"],
                gpu_peak_allocated_gib=memory.get("gpu_peak_allocated_gib"), gpu_peak_reserved_gib=memory.get("gpu_peak_reserved_gib"), host_peak_rss_gib=memory.get("host_peak_rss_gib"), parameters=record.get("parameters"), parameters_plain=record.get("parameters_plain"), patch_size=record["config"]["patch_size"], batch_size=record["config"]["batch_size"],
                train_log=str(path), flip_check=record.get("flip_check"), device=record.get("environment", {}).get("device"), gpu=record.get("environment", {}).get("gpu"), code_commit=record.get("environment", {}).get("git_sha"))


def steps_rule(measured, *, gpu_limit=GPU_LIMIT_GIB, reserved_limit=GPU_RESERVED_LIMIT_GIB, host_limit=HOST_LIMIT_GIB, train_wall=TRAIN_WALL_SECONDS, safety=SAFETY, min_steps=MIN_STEPS, eval_every=EVAL_EVERY):
    """The plan's rule from what the smoke measured: N = floor(safety * (train_wall - data loading) / (step + evaluation/eval_every)), down to a multiple of eval_every. Passes if the peaks are inside the limits and N >= min_steps."""
    per_step = measured["step_seconds_median"]+measured["eval_seconds_median"]/eval_every
    steps = int(safety*(train_wall-measured["data_ready_seconds"])/per_step)
    steps -= steps % eval_every
    reasons = []
    if measured["gpu_peak_allocated_gib"] is None or measured["gpu_peak_allocated_gib"] > gpu_limit:
        reasons.append(f"GPU peak {measured['gpu_peak_allocated_gib']} GiB above {gpu_limit} GiB")
    if measured.get("gpu_peak_reserved_gib") is None or measured["gpu_peak_reserved_gib"] > reserved_limit:
        reasons.append(f"GPU reserved peak {measured.get('gpu_peak_reserved_gib')} GiB above {reserved_limit} GiB: too close to the cap of {gpu_limit} GiB for the long run")
    if measured["host_peak_rss_gib"] is None or measured["host_peak_rss_gib"] > host_limit:
        reasons.append(f"host peak {measured['host_peak_rss_gib']} GiB above {host_limit} GiB")
    if steps < min_steps:
        reasons.append(f"{steps} steps in the training window, fewer than {min_steps}")
    return dict(ok=not reasons, reasons=reasons, patch_size=measured["patch_size"], max_iters=steps, max_wall_seconds=train_wall, eval_every=eval_every, per_step_seconds=per_step,
                rule=f"N = floor({safety} * ({train_wall} s - data loading) / (median step + median evaluation / {eval_every})), down to a multiple of {eval_every}; GPU peak <= {gpu_limit} GiB, GPU reserved peak <= {reserved_limit} GiB, host peak <= {host_limit} GiB, N >= {min_steps}", measured=measured)


def decide_steps(out, patch):
    measured = read_smoke(smoke_root(out, patch)/MEMBER)
    if tuple(measured["patch_size"]) != tuple(patch):
        raise Stop(f"the smoke run used patch {measured['patch_size']}, asked for {patch}")
    document = dict(steps_rule(measured), decided_utc=utc(), command=sys.argv)
    trial = Path(out)/f"steps_decision_{patch_name(patch)}.json"
    if trial.exists():
        raise Stop(f"{trial} exists; never overwritten")
    trial.write_text(json.dumps(document, indent=1)+"\n", encoding="utf-8")
    final = Path(out)/"steps_decision.json"
    if document["ok"]:
        if final.exists():
            raise Stop(f"{final} exists; the step count is decided once")
        final.write_text(json.dumps(document, indent=1)+"\n", encoding="utf-8")
    return document


def read_decision(out):
    path = Path(out)/"steps_decision.json"
    if not path.is_file():
        raise Stop(f"{path}: no such file; the smoke run and decide-steps come first")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not document.get("ok"):
        raise Stop(f"{path}: the rule did not pass: {document.get('reasons')}")
    return document


def read_control(out):
    """The control of the evaluator must have passed before the training starts."""
    path = Path(out)/CONTROL_DIR/"eval_control.json"
    if not path.is_file():
        raise Stop(f"{path}: no such file; the control runs in the smoke step and comes first")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("passed") is not True:
        raise Stop(f"{path}: the control did not reproduce the reference: {document.get('differences')} (tolerance {document.get('tolerance')}); the training does not start, the lead decides")
    return document


def train_config(decision):
    """The configuration of U1: the defaults of TrainConfigLarge, and what the smoke decided."""
    return dict(patch_size=list(decision["patch_size"]), max_iters=int(decision["max_iters"]), eval_every=int(decision["eval_every"]), max_wall_seconds=float(decision["max_wall_seconds"]), min_iters=min(10000, int(decision["max_iters"])))


# ---- prepare ----------------------------------------------------------------------------------------------------------------------------------------

def env_record(path, snapshot):
    lines = [f"recorded: {utc()}", f"python: {PY}"]
    probe = subprocess.run([PY, "-B", "-c", "import numpy, scipy, torch, nibabel, sys; print(sys.version.replace(chr(10), ' ')); print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'nibabel', nibabel.__version__)"],
                           cwd=str(snapshot), capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    lines += probe.stdout.splitlines()+probe.stderr.splitlines()
    smi = subprocess.run([SMI, "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"], capture_output=True, text=True)
    lines.append(f"nvidia-smi: {smi.stdout.strip()}")
    usage = shutil.disk_usage("/home")
    lines += [f"disk /home free: {usage.free/1e9:.0f} GB", f"MemAvailable: {mem_available_gib():.1f} GiB", f"snapshot: {snapshot}"]
    Path(path).write_text("\n".join(lines)+"\n", encoding="utf-8")


def prepare(out, repo):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out/"snapshot"
    if snapshot.exists():
        record = verify_snapshot(snapshot)
        log(f"reusing snapshot {snapshot} (commit {record['commit']})")
    else:
        record = make_snapshot(repo, snapshot, paths=SNAPSHOT_PROFILES[PROFILE])
        log(f"snapshot {snapshot} made from commit {record['commit']}")
    check = subprocess.run([PY, "-B", "-m", "bayesreg.seg.unet.cli", "check-inputs", "--profile", PROFILE, "--require-snapshot"], cwd=str(snapshot), env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True)
    if check.returncode != 0:
        raise Stop(f"the pinned-input check failed: {check.stdout[-2000:]} {check.stderr[-2000:]}")
    log("pinned inputs checked: no label or image was read")
    env_record(out/f"env_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.txt", snapshot)
    return record


# ---- the real steps ---------------------------------------------------------------------------------------------------------------------------------

def real_spec(out):
    return Spec(out=Path(out), cwd=CODE_ROOT)


def cmd_smoke(args):
    require_snapshot()
    set_run_log(args.out)
    patch = args.patch or PATCH_U1
    wait_for_gate(UNITS["smoke"], args.wait_minutes)
    spec = real_spec(args.out)
    root = smoke_root(args.out, patch)
    root.mkdir(parents=True, exist_ok=True)
    run_step(f"smoke_{patch_name(patch)}", train_argv(spec, root, smoke_config(patch, args.steps)), state=Path(args.out)/"chain_state", log_path=root/"smoke.log", cwd=CODE_ROOT)
    measured = read_smoke(root/MEMBER)
    log("measured: "+json.dumps({k: v for k, v in measured.items() if k != "flip_check"}))
    run_step("control_check", control_argv(spec), state=Path(args.out)/"chain_state", log_path=Path(args.out)/"chain.log", cwd=CODE_ROOT)
    log("control of the evaluator: "+json.dumps(read_control(args.out)["differences"]))
    log(f"next, in the foreground: {PY} -B experiments/ul/run_ul.py decide-steps --out {args.out} --patch {','.join(str(p) for p in patch)}")


def cmd_train(args):
    require_snapshot()
    set_run_log(args.out)
    decision = read_decision(args.out)
    read_control(args.out)
    dev_lock.check()                        # not 4.5 hours of training for an evaluation that the lock will refuse
    wait_for_gate(UNITS["train"], args.wait_minutes)
    log(f"U1: {json.dumps(train_config(decision))}")
    run_chain(real_spec(args.out), train_config(decision))
    log(f"finished: {args.out}/analysis/decision_ul.json, summary_ul.md")


def cmd_status(args):
    out = Path(args.out)
    state = out/"chain_state"
    print("state:", " ".join(sorted(p.name for p in state.iterdir())) if state.is_dir() else "(none)")
    for name in ("steps_decision.json", f"{CONTROL_DIR}/eval_control.json", "analysis/decision_ul.json"):
        path = out/name
        print(f"{name}:", json.dumps({k: json.loads(path.read_text()).get(k) for k in ("ok", "max_iters", "passed", "differences", "reading_text", "delta_seg35", "delta_seg4")}, ensure_ascii=False) if path.is_file() else "(none)")
    print(f"read lock {dev_lock.DEV_READ_LOCK}:", json.dumps(dev_lock.describe(), ensure_ascii=False) if dev_lock.DEV_READ_LOCK.exists() else "(none)")
    log_path = out/"chain.log"
    if log_path.is_file():
        print("== chain.log, last lines")
        print("\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-5:]))


# ---- dry run ---------------------------------------------------------------------------------------------------------------------------------------------

TINY = dict(base_channels=4, depth=2, ds_levels=1, batch_size=2, patch_size=[16, 16, 16], max_iters=4, eval_every=2, warmup_iters=1, min_iters=4, log_every=1, amp=False)
TINY_V2 = dict(base_channels=4, depth=2, max_iters=4, eval_every=2, warmup_iters=1, min_iters=4, log_every=1, amp=False)


def synthetic_reference(path, *, control, subjects, data_root, device="cpu"):
    """A file in the layout of runs/20261103-unv2/v2_member_subsets.json for the synthetic control member, made by the functions of eval_u1."""
    import numpy as np
    import eval_seg
    import eval_u1
    from bayesreg.seg.unet.data import OasisFiles, normalize_image, read_image
    model, blob, _ = eval_u1.load_arm(control, device)
    lookup = eval_seg.load_seg4_lookup()
    rows = []
    for subject in subjects:
        values, _ = read_image(OasisFiles(Path(data_root)), subject)
        image, _ = normalize_image(values, percentile=blob["percentile"])
        top, _ = eval_u1.timed_forward(model, image, device=device, amp=blob["amp"], window=None, sigma=None)
        row = eval_seg.summarize_subject(top, eval_seg.read_truth(data_root, subject, "seg35")[0], eval_seg.read_truth(data_root, subject, "seg4")[0], lookup)
        rows.append({k: row[k] for k in ("seg35", "seg4", "groups")})
    means = dict(seg35=float(np.mean([r["seg35"] for r in rows])), seg4=float(np.mean([r["seg4"] for r in rows])))
    Path(path).write_text(json.dumps(dict(subjects=list(subjects), members=["A0"], summary=dict(A0=dict(n_members=1, **means)), curve={"1": dict(means, n_subsets=1)}, per_subject=dict(A0=rows), note="synthetic"), indent=1)+"\n", encoding="utf-8")
    return path


def rehearse_rule():
    """The rule on made-up U1 means around the committed numbers of A0: no evaluation, no subject file."""
    import decide_ul
    import eval_u1
    reference = eval_u1.read_reference(eval_u1.REFERENCE, pinned_sha256=eval_u1.REFERENCE_SHA256)
    out = {}
    for delta in (.006, .005, .0035, .002, .001, -.003):
        out[f"{delta:+.4f}"] = decide_ul.READINGS[decide_ul.reading_of(delta)]
    return dict(reference_member=reference["member"], reference_seg35=reference["means"]["seg35"], reference_seg4=reference["means"]["seg4"], four_single_members=reference["single_member_mean"], readings=out)


def refusals_on_the_repository_files():
    """Ids only: the real allowlist, sealed lists and folds; nothing of a subject is opened."""
    import eval_seg
    import eval_u1
    from bayesreg.eval.truth import load_sealed_lists
    from bayesreg.io.guard import SubjectSealed, load_dev_allowlist
    from bayesreg.seg.unet.data import load_folds
    from bayesreg.seg.unet.synthetic import allowlist_document, write_json
    allowlist = load_dev_allowlist(eval_seg.ALLOWLIST)
    sealed = load_sealed_lists(CODE_ROOT)
    training = load_folds(eval_seg.FOLDS).training_subjects
    admitted = eval_u1.admit(allowlist, eval_seg.DEV_SOURCE, sealed, training, dict(U1=sorted(training)))
    out = dict(admitted=len(admitted))
    try:
        eval_u1.admit(allowlist, "stage_T0_tune_v1", sealed, training, {})
        out["T0"] = "NOT REFUSED"
    except SubjectSealed as exc:
        out["T0"] = f"refused: {str(exc)[:80]}"
    with tempfile.TemporaryDirectory() as scratch:
        for name, subject in (("holdout_B", sorted(sealed.subjects["holdout_B_v1"])[0]), ("training", sorted(training)[0])):
            bad = load_dev_allowlist(write_json(Path(scratch)/f"{name}.json", allowlist_document([("dev", [admitted[0], subject], False)])))
            try:
                eval_u1.admit(bad, "dev", sealed, training, {})
                out[name] = "NOT REFUSED"
            except SubjectSealed as exc:
                out[name] = f"refused: {str(exc)[:80]}"
    return out


def dry_run(scratch=None):
    import decide_ul
    import synthetic_tree
    work = Path(scratch) if scratch else Path(tempfile.mkdtemp(prefix="ul_dryrun_"))
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    print("== the lead's start strings (printed, not run)")
    print("prepare:", shlex.join([PY, "-B", "experiments/ul/run_ul.py", "prepare"]))
    for step in UNITS:
        print(f"{step}:", shlex.join(launch_argv(step)))
    print("decide-steps:", shlex.join([PY, "-B", "experiments/ul/run_ul.py", "decide-steps"]))
    print("\n== the chain of the unit tag-ul-train (printed, not run; max_iters from steps_decision.json, 30000 as an example)")
    example = train_config(dict(patch_size=list(PATCH_U1), max_iters=30000, eval_every=EVAL_EVERY, max_wall_seconds=TRAIN_WALL_SECONDS))
    for name, argv in chain_steps(Spec(out=OUT, cwd=OUT/"snapshot"), example):
        print(f"{name}: {shlex.join(argv)}")
    print("\n== the gate on this machine now (read only; it decides nothing here)")
    print(json.dumps(gate_problems(UNITS["smoke"]), indent=1))
    print("\n== real steps refused here: smoke and train need a verified snapshot")
    try:
        require_snapshot()
        print("NOT REFUSED")
        refused_here = False
    except CodeNotClean as exc:
        print(f"refused: {str(exc)[:160]}")
        refused_here = True
    print("\n== the rule on made-up deltas against the committed A0 numbers")
    rehearsal = rehearse_rule()
    print(json.dumps(rehearsal, indent=1, ensure_ascii=False))
    print("\n== subjects, on the repository's lists (ids only)")
    refusals = refusals_on_the_repository_files()
    print(json.dumps(refusals, indent=1))
    print(f"\n== the read lock of the development truth, {dev_lock.DEV_READ_LOCK}: {'EXISTS' if dev_lock.DEV_READ_LOCK.exists() else 'none'} (synthetic runs and the control do not touch it)")
    print("\n== the chain on synthetic subjects, CPU")
    synthetic_tree.build(work)
    inputs = ["--synthetic", "--folds", str(work/"folds_full_v2.json"), "--allowlist", str(work/"dev_subjects.json"), "--training-source", synthetic_tree.SOURCE, "--expect-counts", "5,1", "--data-root", str(work/"data")]
    out = work/"out"
    set_run_log(out)
    spec = Spec(out=out, cwd=CODE_ROOT, python=sys.executable, device="cpu", inputs=inputs)
    v2_root = work/"v2"
    v2_common = ["--profile", "v2_full", "--fold", FOLD, "--seed", str(SEED), *inputs, "--out", str(v2_root), "--device", "cpu"]
    for name, argv in (("train_v2_control", [sys.executable, "-B", "-m", "bayesreg.seg.unet.cli", "train", *v2_common, *sets(TINY_V2)]), ("calibrate_v2_control", [sys.executable, "-B", "-m", "bayesreg.seg.unet.cli", "calibrate", *v2_common])):
        run_step(name, argv, state=work/"state", log_path=work/"control.log", cwd=CODE_ROOT, env=env)
    reference = synthetic_reference(work/"reference.json", control=v2_root/MEMBER, subjects=synthetic_tree.DEV, data_root=work/"data")
    spec.eval_extra = ["--synthetic", "--control-member", str(v2_root/MEMBER), "--reference", str(reference), "--allowlist", str(work/"dev_subjects.json"), "--folds", str(work/"folds.json"), "--dev-source", "synthetic_dev", "--data-root", str(work/"data")]
    tiny_patch = tuple(TINY["patch_size"])
    root = smoke_root(out, tiny_patch)
    run_step("smoke", train_argv(spec, root, dict(TINY, **smoke_config(tiny_patch, 12), amp=False)), state=work/"state", log_path=work/"smoke.log", cwd=CODE_ROOT, env=env)
    measured = read_smoke(root/MEMBER)
    decision = dict(steps_rule(measured), note="dry run: CPU, tiny network; the step count is not used")
    print("smoke, measured on the CPU:", json.dumps({k: measured[k] for k in ("step_seconds_median", "eval_seconds_median", "data_ready_seconds", "host_peak_rss_gib", "gpu_peak_allocated_gib", "patch_size")}))
    print("the rule on it:", json.dumps({k: decision[k] for k in ("ok", "reasons", "max_iters", "per_step_seconds")}))
    (out/"steps_decision.json").write_text(json.dumps(decision, indent=1)+"\n", encoding="utf-8")
    run_step("control_check", control_argv(spec), state=work/"state", log_path=out/"chain.log", cwd=CODE_ROOT, env=env)
    control = read_control(out)
    print("control of the evaluator:", json.dumps({k: control[k] for k in ("passed", "differences", "tolerance")}))
    run_chain(spec, dict(TINY), env=env)
    run_chain(spec, dict(TINY), env=env)                                  # a second pass skips everything
    result = json.loads((out/"analysis"/"decision_ul.json").read_text(encoding="utf-8"))
    print("decision:", json.dumps({k: result.get(k) for k in ("reading", "reading_text", "not_decidable_because")}, ensure_ascii=False))
    print((out/"analysis"/"summary_ul.md").read_text(encoding="utf-8"))
    ok = refused_here and control["passed"] and result["reading"] == decide_ul.NOT_DECIDABLE and len(result["not_decidable_because"]) == 1 and "2 名受试者" in result["not_decidable_because"][0] and refusals["admitted"] == 20 and all(refusals[k].startswith("refused") for k in ("T0", "holdout_B", "training"))
    print(f"\n== dry run {'ok' if ok else 'NOT OK'}; work {work}")
    return 0 if ok else 8


# ---- command line ---------------------------------------------------------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scratch", type=Path, default=None)
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("prepare")
    p.add_argument("--repo", type=Path, default=MAIN_REPO)
    p.add_argument("--out", type=Path, default=OUT)
    p = sub.add_parser("print-launch")
    p.add_argument("step", choices=sorted(UNITS))
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--patch", type=parse_patch, default=None)
    p.add_argument("--wait-minutes", type=int, default=WAIT_MINUTES)
    p = sub.add_parser("smoke")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--patch", type=parse_patch, default=None)
    p.add_argument("--steps", type=int, default=SMOKE_STEPS)
    p.add_argument("--wait-minutes", type=int, default=WAIT_MINUTES)
    p = sub.add_parser("decide-steps")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--patch", type=parse_patch, default=None)
    p = sub.add_parser("train")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--wait-minutes", type=int, default=WAIT_MINUTES)
    p = sub.add_parser("status")
    p.add_argument("--out", type=Path, default=OUT)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.dry_run:
            return dry_run(args.scratch)
        if args.command == "prepare":
            prepare(args.out, args.repo)
        elif args.command == "print-launch":
            print(shlex.join(launch_argv(args.step, out=args.out, patch=args.patch, wait_minutes=args.wait_minutes)))
        elif args.command == "smoke":
            cmd_smoke(args)
        elif args.command == "decide-steps":
            document = decide_steps(args.out, args.patch or PATCH_U1)
            print(json.dumps({k: document[k] for k in ("ok", "reasons", "max_iters", "per_step_seconds", "patch_size", "max_wall_seconds")}, indent=1))
            if not document["ok"]:
                return 8
        elif args.command == "train":
            cmd_train(args)
        elif args.command == "status":
            cmd_status(args)
        else:
            build_parser().print_help()
            return 2
    except (Stop, CodeNotClean, dev_lock.DevReadLocked) as exc:
        log(f"STOP: {exc}")
        return 8
    return 0


if __name__ == "__main__":
    sys.exit(main())
