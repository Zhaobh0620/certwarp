"""Driver of work package UF (runs/20261105-uf/任务书.md, plan.md): the recipe v3_large of UL trained to the end of its poly schedule, four members matching unet35_v2_full one to one, the nnU-Net probabilities, one evaluation on the development set, the reading. Everything real is started by the lead with the
printed string; this file starts no unit.

    python -B experiments/uf/run_uf.py --dry-run [--scratch DIR]
        the start strings and the real chain's commands (printed, not run), the members against v2 on the repository's files (ids only), the estimate from U1's measurement, the gate's reading, the inputs of the nnU-Net step (headers and hashes, no voxel), the read lock, the refusals on the repository's lists,
        and the whole chain on synthetic subjects on the CPU. No GPU, no unit, nothing written outside the scratch directory.
    python -B experiments/uf/run_uf.py prepare [--repo DIR]
        foreground, from a clean repository: the code snapshot <out>/snapshot (made once, verified after), `cli check-inputs --profile v3_large --require-snapshot` from it (no label or image is read), <out>/members_plan.json (the folds file against v2's members), the environment record. No GPU.
    python -B experiments/uf/run_uf.py print-launch
    python -B <snapshot>/experiments/uf/run_uf.py chain [--wait-minutes 30]                 in the unit tag-uf-chain
        refuses first if the read lock exists (the evaluation would refuse 10 hours later) or members_plan.json is missing or differs; then, in order, each step once, each behind <out>/hold (if the file exists the chain stops before the next step, nothing started) and,
        on the GPU steps, behind the gate:
          control_check (eval_uf.py --control-only: v1, v2 and U1 against their committed numbers, no lock; a control that does not reproduce stops the chain before the training)
          nn_infer (uf_nn.py infer: nnU-Net with probabilities, environment nnunet) -> nn_convert (uf_nn.py convert: the gate against pred_dev): before the training (plan revision 1); a gate that does not pass stops the chain here, nothing goes on by itself
          for foldA_s0, foldB_s0, foldA_s1, foldB_s1: train (cli train --profile v3_large, the plan's three changes) -> check (in this process: the train record against the plan and the member's v2 namesake; any difference stops the chain) -> calibrate
          check_members (cli check-members) -> eval_uf (the one read of the development truth) -> decide_uf -> STOP.
        Every end of the chain (finished or a STOP) writes <out>/queue_end.json (later ones queue_end_<UTC>_<k>.json) with the reminder that the nnU-Net raw directory waits for the lead to clean it, its path and size.
        A step that started and did not finish stops the chain and is kept whole, not rerun; a step over its time limit is killed with its process group and counts as failed. Nothing is rerun automatically (hard rule 4).
    python -B experiments/uf/run_uf.py status
    python -B experiments/uf/run_uf.py estimate

The gate (experiments/ul/run_ul.py gate_problems, UL's): no other tag-* or seq-vp-* unit active, 1-minute load < 1.5, no GPU compute process, GPU utilization <= 20 %, GPU free >= 21 GiB, host MemAvailable >= 9 GiB, /home free >= 100 GB, /mnt/c free >= 15 GiB; polled every 60 s, closed for --wait-minutes: nothing starts, exit 8 (launch
again later: the finished steps are skipped).
Logs: every step's output and command in <out>/chain.log; the gate's waits, the steps and every STOP also in <out>/run_uf.log. Exit codes: 0 done, 2 usage, 8 a stop (the reason is printed).
"""
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
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
sys.path[:0] = [str(REPO), str(HERE), str(HERE.parent/"ul"), str(HERE.parent/"unv2")]

import uf_common as uc  # noqa: E402
import uf_lock  # noqa: E402
from bayesreg.seg.unet.snapshot import CODE_ROOT, SNAPSHOT_PATHS_V3, CodeNotClean, make_snapshot, require_snapshot, verify_snapshot  # noqa: E402

UNIT = "tag-uf-chain"
SNAPSHOT_PATHS_UF = SNAPSHOT_PATHS_V3+uc.SNAPSHOT_EXTRA
THREADS = 4                                     # as UL's units
MEMORY_MAX, MEMORY_SWAP_MAX = "8G", "2G"        # the brief: host MemoryMax 8G (U1 measured 5.0 GiB in training)
WAIT_MINUTES, POLL_SECONDS = 30, 60
TIMEOUTS = dict(control_check=3600, train=int(uc.MAX_WALL_SECONDS)+1800, calibrate=1800, check_members=1800, nn_infer=7200, nn_convert=7200, eval_uf=7200, decide_uf=600)    # seconds per step; train = the wall stop of the loop + data loading, the last evaluation and the record
GPU_STEPS = ("control_check", "train", "calibrate", "nn_infer", "eval_uf")


def runtime_max_seconds():
    """RuntimeMaxSec of the unit: every step at its time limit and every gate closed to its limit. A safety valve for a driver that hangs, not a budget."""
    steps = sum(TIMEOUTS[k] for k in ("train", "calibrate"))*len(uc.MEMBERS)+sum(TIMEOUTS[k] for k in ("control_check", "check_members", "nn_infer", "nn_convert", "eval_uf", "decide_uf"))
    gated = 2*len(uc.MEMBERS)+3
    return steps+gated*WAIT_MINUTES*60+1800


class Stop(Exception):
    """A stop of this driver: exit 8, the reason printed, nothing rerun."""


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


LOG_FILE = None


def set_run_log(out):
    global LOG_FILE
    Path(out).mkdir(parents=True, exist_ok=True)
    LOG_FILE = Path(out)/"run_uf.log"


def log(message):
    line = f"{utc()} {message}"
    print(line, flush=True)
    if LOG_FILE is not None:
        with LOG_FILE.open("a", encoding="utf-8") as stream:
            stream.write(line+"\n")


# ---- the start string ----------------------------------------------------------------------------------------------------------------------------------

def check_launch(argv):
    """The rules of a start string (the brief): no test runner in it, the unit's prefix, the results root, the host limit, a RuntimeMaxSec."""
    text = " ".join(str(a) for a in argv)
    if "pytest" in text:
        raise ValueError(f"a start string must not contain the test runner: {text}")
    units = [a for a in argv if str(a).startswith("--unit=")]
    if units != [f"--unit={UNIT}"]:
        raise ValueError(f"the unit must be {UNIT}: {units}")
    if "seq-vp-" in text:
        raise ValueError("a unit of this project never uses the seq-vp- prefix")
    if not any(str(a).startswith(f"--working-directory={uc.RESULTS}/") for a in argv):
        raise ValueError(f"the working directory is the snapshot under {uc.RESULTS}")
    if f"--property=MemoryMax={MEMORY_MAX}" not in argv or not any(str(a).startswith("--property=RuntimeMaxSec=") for a in argv):
        raise ValueError(f"MemoryMax={MEMORY_MAX} and a RuntimeMaxSec are required")
    return [str(a) for a in argv]


def launch_argv(*, out=uc.OUT, wait_minutes=WAIT_MINUTES):
    """The lead's start of the chain, as a list; printed by print-launch, never run by this file."""
    snapshot = Path(out)/"snapshot"
    return check_launch(["systemd-run", "--user", f"--unit={UNIT}", "--collect", f"--working-directory={snapshot}", f"--property=MemoryMax={MEMORY_MAX}", f"--property=MemorySwapMax={MEMORY_SWAP_MAX}", f"--property=RuntimeMaxSec={runtime_max_seconds()}", "--setenv=PYTHONDONTWRITEBYTECODE=1",
                         *[f"--setenv={name}={THREADS}" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")], uc.PY, "-B", str(snapshot/"experiments/uf/run_uf.py"), "chain", "--out", str(out), "--wait-minutes", str(wait_minutes)])


# ---- the gate and the hold file --------------------------------------------------------------------------------------------------------------------------

def gate_problems(own_unit=UNIT, **kwargs):
    import run_ul
    return run_ul.gate_problems(own_unit, **kwargs)


def wait_for_gate(wait_minutes, *, problems=None, sleep=time.sleep, poll=POLL_SECONDS):
    """Returns when the gate is open; Stop when it stayed closed for wait_minutes (nothing started, no marker: the next launch takes it up). problems: gate_problems, looked up when called."""
    problems = gate_problems if problems is None else problems
    deadline = time.monotonic()+60*wait_minutes
    while True:
        reasons = problems()
        if not reasons:
            log("gate open")
            return
        if time.monotonic() >= deadline:
            raise Stop(f"the gate stayed closed for {wait_minutes} minutes: {'; '.join(reasons)}; nothing started")
        log(f"gate closed: {'; '.join(reasons)}")
        sleep(poll)


def check_hold(out):
    hold = Path(out)/"hold"
    if hold.exists():
        raise Stop(f"{hold} exists: the chain stops before its next step (nothing started); remove it and launch again")


# ---- the steps -------------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class Spec:
    """What differs between the real chain and the synthetic dry run."""
    out: Path
    cwd: Path
    python: str = uc.PY
    device: str = "cuda"
    inputs: list = field(default_factory=list)                 # options of cli after --profile (real: none, the profile's pins)
    train_config: dict = field(default_factory=lambda: dict(uc.TRAIN_CONFIG))
    nn_extra: list = field(default_factory=list)               # options of uf_nn.py convert (real: none)
    eval_extra: list = field(default_factory=list)             # options of eval_uf.py (real: none)
    v2_root: Path = uc.V2_ROOT
    v2_registry: Path = None
    nn_infer: object = None                                    # None: uf_nn.py infer as a step; the dry run puts a synthetic stand-in here
    gate: bool = True
    wait_minutes: int = WAIT_MINUTES
    env: dict = None

    @property
    def registry(self):
        return Path(self.v2_registry) if self.v2_registry is not None else Path(self.cwd)/uc.V2_REGISTRY_REL

    @property
    def overrides(self):
        return None if self.train_config == uc.TRAIN_CONFIG else self.train_config


def cli_argv(spec):
    return [spec.python, "-B", "-m", "bayesreg.seg.unet.cli"]


def sets(config):
    out = []
    for key, value in config.items():
        out += ["--set", f"{key}={json.dumps(value)}"]
    return out


def chain_steps(spec):
    """(name, kind, argv or None) of the chain, in order. Nothing else runs and each runs once. kind: the key of TIMEOUTS, or "check" (in this process)."""
    out, script = Path(spec.out), Path(spec.cwd)/"experiments"/"uf"
    steps = [("control_check", "control_check", [spec.python, "-B", str(script/"eval_uf.py"), "--control-only", "--out-dir", str(out/"control_check"), "--device", spec.device, *spec.eval_extra]),       # UL's R5: the evaluator's real path seen working before the training; no lock
             # plan revision 1 (review important 1, the lead): the nnU-Net steps need no UF member, so they run before the training; a gate that does not pass stops the chain there, with its record
             ("nn_infer", "nn_infer", [spec.python, "-B", str(script/"uf_nn.py"), "infer", "--out", str(out)] if spec.nn_infer is None else None),
             ("nn_convert", "nn_convert", [spec.python, "-B", str(script/"uf_nn.py"), "convert", "--out", str(out), *spec.nn_extra])]
    for fold, seed in uc.MEMBERS:
        name = uc.member_name(fold, seed)
        common = ["--profile", uc.PROFILE, "--fold", fold, "--seed", str(seed), *spec.inputs, "--out", str(out)]
        steps += [(f"train_{name}", "train", cli_argv(spec)+["train", *common, "--device", spec.device, *sets(spec.train_config)]), (f"check_{name}", "check", None), (f"calibrate_{name}", "calibrate", cli_argv(spec)+["calibrate", *common, "--device", spec.device])]
    steps += [("check_members", "check_members", cli_argv(spec)+["check-members", "--profile", uc.PROFILE, *spec.inputs, "--out", str(out)]),
              ("eval_uf", "eval_uf", [spec.python, "-B", str(script/"eval_uf.py"), "--out-dir", str(out/"eval_uf"), "--member-root", str(out), "--device", spec.device, *spec.eval_extra]),
              ("decide_uf", "decide_uf", [spec.python, "-B", str(script/"decide_uf.py"), "--eval", str(out/"eval_uf"/"eval_uf.json"), "--out-dir", str(out/"analysis")])]
    return steps


def run_command(argv, *, cwd, log_path, timeout, env=None, popen=subprocess.Popen):
    """The command in its own process group, output to log_path; on the time limit the whole group is killed (SIGKILL). Returns (exit code, timed out)."""
    with Path(log_path).open("a", encoding="utf-8") as stream:
        process = popen(argv, cwd=str(cwd), stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            return process.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            return process.returncode, True


def run_step(name, action, *, state, log_path, label=None):
    """Once: a finished step is skipped, a step that started and did not finish stops the chain and is kept whole. action() returns an exit code (0 done)."""
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    done, started = state/f"{name}.done", state/f"{name}.started"
    if done.exists():
        log(f"skip {name} (done)")
        return False
    if started.exists():
        raise Stop(f"{name} started at {started.read_text().strip()} and did not finish; kept whole, not rerun")
    started.write_text(utc()+"\n")
    log(f"start {name}: {label or ''}")
    with Path(log_path).open("a", encoding="utf-8") as stream:
        stream.write(f"{utc()} start {name}: {label or ''}\n")
    code = action()
    with Path(log_path).open("a", encoding="utf-8") as stream:
        stream.write(f"{utc()} end {name}: exit {code}\n")
    if code != 0:
        raise Stop(f"{name} failed (exit {code}); log {log_path}; kept, not rerun")
    done.write_text(utc()+"\n")
    log(f"done {name}")
    return True


def member_check(spec, name):
    """The step check_<name>: the train record against the plan and the v2 namesake (uf_common.member_record_problems). Written to <out>/<name>/uf_check.json; a problem stops the chain."""
    v2 = uc.v2_member_lists(spec.v2_root, registry=spec.registry)
    record = json.loads((Path(spec.out)/name/"train_record.json").read_text(encoding="utf-8"))
    expected = uc.planned_config(spec.overrides)
    problems = uc.member_record_problems(record, name, v2[name], expected_config=expected, max_iters=expected["max_iters"])
    document = dict(schema="bayesreg.uf.member_check.v1", member=name, v2_namesake=uc.V2_NAME[name], v2_member_sha256=v2[name]["member_sha256"], problems=problems, stopped=record.get("stopped"), iterations_run=record.get("iterations_run"), checkpoint_iteration=record.get("checkpoint_iteration"),
                    n_train=len(record.get("train_subjects") or []), n_calib=len(record.get("calib_subjects") or []), utc=utc())
    path = Path(spec.out)/name/"uf_check.json"
    if path.exists():
        raise Stop(f"{path} exists; never overwritten")
    uc.write_json_atomic(path, document)
    if problems:
        log(f"{name}: not the plan's member: {problems}")
        return 8
    return 0


def members_plan(spec, split):
    """<out>/members_plan.json, once: the four members, their subjects from the folds file against their v2 namesakes. Stop if they differ, or if the file exists with other content."""
    v2 = uc.v2_member_lists(spec.v2_root, registry=spec.registry)
    problems = uc.folds_problems(split, v2)
    document = dict(schema="bayesreg.uf.members_plan.v1", folds_sha256=split.sha256, members={uc.member_name(f, s): dict(fold=f, seed=s, v2_namesake=uc.V2_NAME[uc.member_name(f, s)], v2_member_sha256=v2[uc.member_name(f, s)]["member_sha256"], n_train=len(split.subjects(f, "train")),
                                                                                                 n_calib=len(split.subjects(f, "calib")), calib=split.subjects(f, "calib"), same_as_v2=not [p for p in problems if p.startswith(uc.member_name(f, s))]) for f, s in uc.MEMBERS},
                    order=list(uc.MEMBER_NAMES), train_config=spec.train_config, problems=problems)
    if problems:
        raise Stop(f"the members do not match v2's: {problems}")
    path = Path(spec.out)/"members_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=1)+"\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise Stop(f"{path} exists with other content; never overwritten")
    else:
        uc.write_text_atomic(path, text)
    return document


def raw_reminder(out):
    """The nnU-Net raw directory and its size, or None when there is none: the tools never delete it, the lead cleans it after the reading (plan revision 1, review minor 9)."""
    raw = Path(out)/"nn"/"raw"
    if not raw.is_dir():
        return None
    size = uc.directory_bytes(raw)
    return dict(path=str(raw), bytes=size, text=f"nnU-Net raw 待总负责人清理：{raw}，{size/1e9:.2f} GB")


def chain_end(out, stop):
    """<out>/queue_end.json (a later end: queue_end_<UTC>.json, never overwritten): why the chain ended, the reading if there is one, and the reminder of the nnU-Net raw directory, also in the log."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    decision = out/"analysis"/"decision_uf.json"
    reading = json.loads(decision.read_text(encoding="utf-8")) if decision.is_file() else {}
    raw = raw_reminder(out)
    document = dict(schema="bayesreg.uf.queue_end.v1", stop=stop, finished=stop is None and decision.is_file(), reading=reading.get("reading"), reading_text=reading.get("reading_text"), primary_arm=reading.get("primary_arm"), delta_seg35=reading.get("delta_seg35"),
                    nn_raw=raw, done=sorted(p.name[:-5] for p in (out/"chain_state").glob("*.done")) if (out/"chain_state").is_dir() else [], utc=utc())
    path, k = out/"queue_end.json", 0
    while path.exists():
        k += 1
        path = out/f"queue_end_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{k}.json"
    uc.write_json_atomic(path, document, ensure_ascii=False)
    if raw:
        log(raw["text"])
    return path


def run_chain(spec):
    out = Path(spec.out)
    try:
        run_steps(spec)
    except Stop as stop:
        chain_end(out, str(stop))
        raise
    chain_end(out, None)


def run_steps(spec):
    out = Path(spec.out)
    state, chain_log = out/"chain_state", out/"chain.log"
    for name, kind, argv in chain_steps(spec):
        if (state/f"{name}.done").exists():
            log(f"skip {name} (done)")
            continue
        check_hold(out)
        if spec.gate and kind in GPU_STEPS:
            wait_for_gate(spec.wait_minutes)
            check_hold(out)                     # review minor 7: a hold file put there while the gate was closed still stops this step
        if kind == "check":
            member = name.removeprefix("check_")
            action, label = (lambda member=member: member_check(spec, member)), "in this process: the train record against the plan and v2's namesake"
        elif argv is None:
            action, label = (lambda: spec.nn_infer(spec)), "synthetic stand-in of the nnU-Net inference"
        else:
            def action(argv=argv, kind=kind, name=name):
                code, timed_out = run_command(argv, cwd=spec.cwd, log_path=chain_log, timeout=TIMEOUTS[kind], env=spec.env)
                if timed_out:
                    log(f"{name}: over its time limit of {TIMEOUTS[kind]} s; the process group was killed")
                    return 124
                return code
            label = shlex.join(argv)
        run_step(name, action, state=state, log_path=chain_log, label=label)


# ---- the estimate from U1 ------------------------------------------------------------------------------------------------------------------------------

def u1_measurement(member=uc.U1_MEMBER):
    """What U1's training measured (unet35_v3_large/foldA_s0: train_log.jsonl and train_record.json, the training subjects' calib Dice only, nothing of the development set): the median step after the first 10, the median evaluation, the data loading, the whole wall per step."""
    rows = [json.loads(line) for line in (Path(member)/"train_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    record = json.loads((Path(member)/"train_record.json").read_text(encoding="utf-8"))
    steps = [r["step_seconds"] for r in rows if r["event"] == "train" and r["it"] >= 10]
    evals = [r["eval_seconds"] for r in rows if r["event"] == "eval"]
    start = next(r for r in rows if r["event"] == "start")
    return dict(step_seconds_median=statistics.median(steps), eval_seconds_median=statistics.median(evals), n_evaluations=len(evals), data_ready_seconds=start["data_ready_seconds"], iterations_run=record["iterations_run"], wall_seconds=record["wall_seconds"],
                wall_per_step=record["wall_seconds"]/record["iterations_run"], gpu_peak_allocated_gib=record["memory"].get("gpu_peak_allocated_gib"), gpu_peak_reserved_gib=record["memory"].get("gpu_peak_reserved_gib"), host_peak_rss_gib=record["memory"].get("host_peak_rss_gib"), stopped=record["stopped"])


def estimate(measured, *, max_iters=uc.MAX_ITERS, eval_every=uc.EVAL_EVERY):
    """〔估算〕 of one member and of the chain from U1's measurement, two ways: its parts (data, steps, evaluations) and its whole wall per step."""
    parts = measured["data_ready_seconds"]+max_iters*measured["step_seconds_median"]+(max_iters//eval_every)*measured["eval_seconds_median"]
    whole = max_iters*measured["wall_per_step"]
    member = max(parts, whole)
    rest = dict(control_check=300, calibrations=4*120, check_members=60, nn_infer=20*60, nn_convert=20*60, eval_uf=15*60)       # 〔估算〕 upper ends: the control (20 subjects, U1's forward and two read-backs each), calibration (U1's took about a minute), nnU-Net writing 36 float32 classes compressed (about 1 minute per subject), the conversion (NX plan: about 1 minute per subject), the evaluation (five networks and three read-backs per subject)
    rest = sum(rest.values())
    return dict(per_member_hours_parts=parts/3600, per_member_hours_whole=whole/3600, per_member_hours=member/3600, members_hours=4*member/3600, rest_hours=rest/3600, total_hours=(4*member+rest)/3600, wall_stop_hours=uc.MAX_WALL_SECONDS/3600, wall_stop_over_estimate=uc.MAX_WALL_SECONDS/member)


# ---- prepare ---------------------------------------------------------------------------------------------------------------------------------------------

def env_record(path, snapshot):
    lines = [f"recorded: {utc()}", f"python: {uc.PY}"]
    probe = subprocess.run([uc.PY, "-B", "-c", "import numpy, scipy, torch, nibabel, sys; print(sys.version.replace(chr(10), ' ')); print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'nibabel', nibabel.__version__)"],
                           cwd=str(snapshot), capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    lines += probe.stdout.splitlines()+probe.stderr.splitlines()
    smi = subprocess.run(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"], capture_output=True, text=True)
    lines += [f"nvidia-smi: {smi.stdout.strip()}", f"disk /home free: {shutil.disk_usage('/home').free/1e9:.0f} GB", f"snapshot: {snapshot}"]
    Path(path).write_text("\n".join(lines)+"\n", encoding="utf-8")


def real_split():
    from bayesreg.seg.unet.data import load_full_folds
    from bayesreg.seg.unet.pins import DEFAULT_FULL_FOLDS, PINNED_FULL_FOLDS_SHA256
    split = load_full_folds(DEFAULT_FULL_FOLDS)
    if split.sha256 != PINNED_FULL_FOLDS_SHA256:
        raise Stop(f"{DEFAULT_FULL_FOLDS}: sha256 {split.sha256} is not the pinned one")
    return split


def prepare(out, repo):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    set_run_log(out)
    snapshot = out/"snapshot"
    if snapshot.exists():
        record = verify_snapshot(snapshot)
        log(f"reusing snapshot {snapshot} (commit {record['commit']})")
    else:
        record = make_snapshot(repo, snapshot, paths=SNAPSHOT_PATHS_UF)
        log(f"snapshot {snapshot} made from commit {record['commit']}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    check = subprocess.run([uc.PY, "-B", "-m", "bayesreg.seg.unet.cli", "check-inputs", "--profile", uc.PROFILE, "--require-snapshot"], cwd=str(snapshot), env=env, capture_output=True, text=True)
    if check.returncode != 0:
        raise Stop(f"the pinned-input check failed: {check.stdout[-2000:]} {check.stderr[-2000:]}")
    plan = subprocess.run([uc.PY, "-B", str(snapshot/"experiments/uf/run_uf.py"), "members-plan", "--out", str(out)], cwd=str(snapshot), env=env, capture_output=True, text=True)
    if plan.returncode != 0:
        raise Stop(f"the members' plan failed: {plan.stdout[-2000:]} {plan.stderr[-2000:]}")
    log("pinned inputs checked, members_plan.json written: no label or image was read")
    env_record(out/f"env_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.txt", snapshot)
    return record


# ---- the real chain --------------------------------------------------------------------------------------------------------------------------------------

def real_spec(out, wait_minutes=WAIT_MINUTES):
    return Spec(out=Path(out), cwd=CODE_ROOT, wait_minutes=wait_minutes)


def cmd_chain(args):
    require_snapshot()
    set_run_log(args.out)
    spec = real_spec(args.out, args.wait_minutes)
    if not (Path(args.out)/"chain_state"/"eval_uf.done").exists():
        uf_lock.check()                         # not 10 hours of training for an evaluation that the lock will refuse
    plan = Path(args.out)/"members_plan.json"
    if not plan.is_file():
        raise Stop(f"{plan}: no such file; prepare comes first")
    members_plan(spec, real_split())            # the same content again, or Stop
    log(f"UF: {json.dumps(spec.train_config)}; members {list(uc.MEMBER_NAMES)}")
    run_chain(spec)
    log(f"finished: {args.out}/analysis/decision_uf.json, summary_uf.md. STOP (the brief: nothing after the reading)")


def cmd_status(args):
    out = Path(args.out)
    state = out/"chain_state"
    print("state:", " ".join(sorted(p.name for p in state.iterdir())) if state.is_dir() else "(none)")
    print("hold:", "YES" if (out/"hold").exists() else "no")
    for name in uc.MEMBER_NAMES:
        path = out/name/"uf_check.json"
        print(f"{name}:", json.dumps({k: json.loads(path.read_text()).get(k) for k in ("stopped", "iterations_run", "checkpoint_iteration", "problems")}) if path.is_file() else "(no check yet)")
    for name in ("nn/infer_check.json", "nn/convert_record.json", "analysis/decision_uf.json"):
        path = out/name
        print(f"{name}:", json.dumps({k: json.loads(path.read_text()).get(k) for k in ("problems", "gate", "n_differing_total", "reading_text", "primary_arm", "delta_seg35")}, ensure_ascii=False) if path.is_file() else "(none)")
    print(f"read lock {uf_lock.DEV_READ_LOCK}:", json.dumps(uf_lock.describe(), ensure_ascii=False) if uf_lock.DEV_READ_LOCK.exists() else "(none)")
    if (out/"chain.log").is_file():
        print("== chain.log, last lines")
        print("\n".join((out/"chain.log").read_text(encoding="utf-8", errors="replace").splitlines()[-5:]))


# ---- dry run ---------------------------------------------------------------------------------------------------------------------------------------------

TINY_UF = dict(base_channels=4, depth=2, ds_levels=1, batch_size=2, patch_size=[16, 16, 16], max_iters=4, min_iters=4, patience=4, eval_every=2, warmup_iters=1, log_every=1, amp=False, flip_axis_check=False, checkpoint="last", max_wall_seconds=3600.)
TINY_U1 = dict(base_channels=4, depth=2, ds_levels=1, batch_size=2, patch_size=[16, 16, 16], max_iters=4, eval_every=2, warmup_iters=1, min_iters=4, log_every=1, amp=False)
TINY_V2 = dict(base_channels=4, depth=2, max_iters=4, eval_every=2, warmup_iters=1, min_iters=4, log_every=1, amp=False)


def synthetic_member_reference(path, *, v2_root, subjects, data_root):
    """runs/20261103-unv2/v2_member_subsets.json's layout for the synthetic v2 members: each alone, argmax of softmax(z/T) (the script's path), summarized by eval_seg."""
    import numpy as np
    import torch
    import eval_seg
    import eval_u1
    from bayesreg.seg.unet.data import OasisFiles, normalize_image, read_image
    from bayesreg.seg.unet.train import predict_logits
    lookup = eval_seg.load_seg4_lookup()
    names = [uc.V2_NAME[n] for n in sorted(uc.MEMBER_NAMES)]
    models = {uc.V2_NAME[n]: eval_u1.load_arm(Path(v2_root)/n, "cpu")[:2] for n in sorted(uc.MEMBER_NAMES)}
    rows = {name: [] for name in names}
    for subject in subjects:
        values, _ = read_image(OasisFiles(Path(data_root)), subject)
        for name, (model, blob) in models.items():
            image, _ = normalize_image(values, percentile=blob["percentile"])
            z = predict_logits(model, image, device="cpu", amp=blob["amp"], window=tuple(blob["window"]) if blob.get("window") else None)
            top = torch.softmax(z.float()/blob["temperature"], dim=1)[0].argmax(0).to(torch.uint8).numpy()
            row = eval_seg.summarize_subject(top, eval_seg.read_truth(data_root, subject, "seg35")[0], eval_seg.read_truth(data_root, subject, "seg4")[0], lookup)
            rows[name].append({k: row[k] for k in ("seg35", "seg4", "groups")})
    summary = {name: dict(n_members=1, seg35=float(np.mean([r["seg35"] for r in rows[name]])), seg4=float(np.mean([r["seg4"] for r in rows[name]]))) for name in names}
    curve = {"1": dict(seg35=float(np.mean([s["seg35"] for s in summary.values()])), seg4=float(np.mean([s["seg4"] for s in summary.values()])), n_subsets=4)}
    Path(path).write_text(json.dumps(dict(subjects=list(subjects), members=names, summary=summary, curve=curve, per_subject=rows, note="synthetic"), indent=1)+"\n", encoding="utf-8")
    return path


def synthetic_u1_reference(path, *, member, subjects, data_root):
    """eval_u1.json's layout (the parts eval_uf reads) for the synthetic U1: eval_u1.timed_forward, summarized by eval_seg."""
    import numpy as np
    import eval_seg
    import eval_u1
    from bayesreg.seg.unet.data import OasisFiles, normalize_image, read_image
    model, blob, _ = eval_u1.load_arm(member, "cpu")
    lookup = eval_seg.load_seg4_lookup()
    rows = {}
    for subject in subjects:
        values, _ = read_image(OasisFiles(Path(data_root)), subject)
        image, _ = normalize_image(values, percentile=blob["percentile"])
        top, _ = eval_u1.timed_forward(model, image, device="cpu", amp=blob["amp"], window=tuple(blob["window"]), sigma=blob["window_gaussian_sigma"])
        row = eval_seg.summarize_subject(top, eval_seg.read_truth(data_root, subject, "seg35")[0], eval_seg.read_truth(data_root, subject, "seg4")[0], lookup)
        rows[subject] = {k: row[k] for k in ("seg35", "seg4", "groups", "per_label", "seg4_per_label")}
    means = dict(U1=dict(seg35=float(np.mean([r["seg35"] for r in rows.values()])), seg4=float(np.mean([r["seg4"] for r in rows.values()]))))
    Path(path).write_text(json.dumps(dict(subjects=list(subjects), rows=dict(U1=rows), means=means, members=dict(U1=dict(member_sha256=eval_u1.member_files(member)[1])), note="synthetic"), indent=1)+"\n", encoding="utf-8")
    return path


def native_from_load_scalar(values, affine):
    """The inverse of nx_seg.orient_native: an array in load_scalar's order (zyx, RAS) into the NIfTI file's order for this affine."""
    import nibabel as nib
    transform = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt(("R", "A", "S")), nib.orientations.io_orientation(affine))
    return nib.orientations.apply_orientation(values.transpose(2, 1, 0), transform)


def synthetic_nn_outputs(*, nn_out, raw_dir, subjects, data_root, seed=0):
    """What NN and nnU-Net would leave for synthetic subjects: nn_out/{train_record.json, prepare_manifest.json, pred_dev/<subject>.nii.gz} and raw_dir/{<subject>.npz (classes, SimpleITK order), <subject>.nii.gz, infer_record.json}. Probabilities peaked on the synthetic truth with noise."""
    import nibabel as nib
    import numpy as np
    import synthetic_tree
    from jd_curl.three_d.nifti import load_scalar
    nn_out, raw_dir = Path(nn_out), Path(raw_dir)
    (nn_out/"pred_dev").mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = nn_out/"checkpoint_final.pth"
    checkpoint.write_bytes(b"synthetic weights")
    train = dict(trainer="nnUNetTrainer_250epochs_NoMirroring", final_checkpoint=str(checkpoint), final_checkpoint_sha256=uc.sha256_file(checkpoint))
    (nn_out/"train_record.json").write_text(json.dumps(train, indent=1)+"\n", encoding="utf-8")
    (nn_out/"prepare_manifest.json").write_text(json.dumps(dict(label_contact_subjects=synthetic_tree.TRAINING), indent=1)+"\n", encoding="utf-8")
    rng = np.random.default_rng(seed)
    for subject in subjects:
        image = nib.load(Path(data_root)/subject/"aligned_norm.nii.gz")
        labels = load_scalar(Path(data_root)/subject/"aligned_seg35.nii.gz", labels=True).values.astype(np.int64)
        noise = rng.uniform(0, 1, (36,)+labels.shape)
        p = .6*(np.arange(36)[:, None, None, None] == labels[None])+.4*noise/noise.sum(axis=0)
        p = (p/p.sum(axis=0)).astype(np.float32)
        native = np.stack([native_from_load_scalar(p[c], image.affine) for c in range(36)])
        np.savez_compressed(raw_dir/f"{subject}.npz", probabilities=native.transpose(0, 3, 2, 1))
        top = native_from_load_scalar(p.argmax(axis=0).astype(np.uint8), image.affine)
        for target in (nn_out/"pred_dev"/f"{subject}.nii.gz", raw_dir/f"{subject}.nii.gz"):
            nib.save(nib.Nifti1Image(top, image.affine), str(target))
    (raw_dir/"infer_record.json").write_text(json.dumps(dict(checkpoint_sha256=train["final_checkpoint_sha256"], saved_probabilities=True, subjects=list(subjects), allowed_mirroring_axes=None, tta_views=1, gpu_peak_reserved_gib=1., note="synthetic"), indent=1)+"\n", encoding="utf-8")
    return train


def repository_checks():
    """Ids and records only: the members against v2 on the real folds file and v2's records; the refusals of the development subjects' admission on the real lists; the inputs of the nnU-Net step (records, hashes, headers; no voxel)."""
    import eval_seg
    from bayesreg.eval.truth import load_sealed_lists
    from bayesreg.io.guard import SubjectSealed, load_dev_allowlist
    from bayesreg.seg.unet.data import load_folds
    from bayesreg.seg.unet.synthetic import allowlist_document, write_json
    split = real_split()
    v2 = uc.v2_member_lists(uc.V2_ROOT, registry=CODE_ROOT/uc.V2_REGISTRY_REL)
    members = {}
    for fold, seed in uc.MEMBERS:
        name = uc.member_name(fold, seed)
        members[name] = dict(fold=fold, seed=seed, v2=uc.V2_NAME[name], v2_member=v2[name]["member_sha256"][:12], n_train=len(split.subjects(fold, "train")), n_calib=len(split.subjects(fold, "calib")), calib=[s[11:15] for s in split.subjects(fold, "calib")],
                             same_train=split.subjects(fold, "train") == v2[name]["train"], same_calib=split.subjects(fold, "calib") == v2[name]["calib"], same_contact=split.label_contact(fold) == v2[name]["contact"])
    allowlist = load_dev_allowlist(eval_seg.ALLOWLIST)
    sealed = load_sealed_lists(CODE_ROOT)
    training = load_folds(eval_seg.FOLDS).training_subjects
    admitted = eval_seg.admit_subjects(allowlist, eval_seg.DEV_SOURCE, sealed, training)
    refusals = dict(admitted=len(admitted), overlap_with_training=len(set(admitted) & set(training)), t0_in_training=len(set(allowlist.never_for_training) & set(training)))
    try:
        eval_seg.admit_subjects(allowlist, "stage_T0_tune_v1", sealed, training)
        refusals["T0"] = "NOT REFUSED"
    except SubjectSealed as exc:
        refusals["T0"] = f"refused: {str(exc)[:80]}"
    with tempfile.TemporaryDirectory() as scratch:
        for label, subject in (("holdout_B", sorted(sealed.subjects["holdout_B_v1"])[0]), ("training", sorted(training)[0])):
            bad = load_dev_allowlist(write_json(Path(scratch)/f"{label}.json", allowlist_document([("dev", [admitted[0], subject], False)])))
            try:
                eval_seg.admit_subjects(bad, "dev", sealed, training)
                refusals[label] = "NOT REFUSED"
            except SubjectSealed as exc:
                refusals[label] = f"refused: {str(exc)[:80]}"
    nn = {}
    train_path = uc.NN_OUT/"train_record.json"
    if train_path.is_file():
        train = json.loads(train_path.read_text(encoding="utf-8"))
        nn.update(trainer=train["trainer"], checkpoint_sha256_matches=uc.sha256_file(train["final_checkpoint"]) == train["final_checkpoint_sha256"], pred_dev=sorted(p.name[:-7] for p in (uc.NN_OUT/"pred_dev").glob("*.nii.gz")) == sorted(admitted),
                  manifest_contact=len(json.loads((uc.NN_OUT/"prepare_manifest.json").read_text(encoding="utf-8"))["label_contact_subjects"]))
    references = {}
    for label, path, pin in (("ensembles", uc.REF_ENSEMBLES, uc.REF_ENSEMBLES_SHA256), ("members", CODE_ROOT/uc.REF_MEMBERS_REL, uc.REF_MEMBERS_SHA256), ("u1", uc.REF_U1, uc.REF_U1_SHA256), ("u1_member", uc.U1_MEMBER/"member.pt", uc.U1_MEMBER_SHA256)):
        references[label] = Path(path).is_file() and uc.sha256_file(path) == pin
    return dict(members=members, refusals=refusals, nn=nn, references=references, folds_sha256=split.sha256)


def build_synthetic(work, env):
    """The synthetic world of the dry run and the tests (CPU; ids OASIS_OAS1_09xx_MR1, a non-real data root): experiments/unv2/synthetic_tree.py's subjects and folds, four tiny v2 members registered and inferred through cli, a tiny U1, the three references in their real layouts,
    NN's records and nnU-Net's raw files for the two development subjects, and the Spec of the chain with every option pointing there. Nothing of a real subject."""
    import types
    import synthetic_tree
    synthetic_tree.build(work)
    data, allow, source = work/"data", work/"dev_subjects.json", synthetic_tree.SOURCE
    inputs = ["--synthetic", "--folds", str(work/"folds_full_v2.json"), "--allowlist", str(allow), "--training-source", source, "--expect-counts", "5,1", "--data-root", str(data)]
    cli = [sys.executable, "-B", "-m", "bayesreg.seg.unet.cli"]
    setup_state, setup_log = work/"setup_state", work/"setup.log"

    def call(argv):
        return lambda: run_command(argv, cwd=CODE_ROOT, log_path=setup_log, timeout=1800, env=env)[0]

    v2_root = work/"v2"
    for fold, seed in uc.MEMBERS:
        common = ["--profile", "v2_full", "--fold", fold, "--seed", str(seed), *inputs, "--out", str(v2_root), "--device", "cpu"]
        run_step(f"v2_train_{fold}{seed}", call([*cli, "train", *common, *sets(TINY_V2)]), state=setup_state, log_path=setup_log)
        run_step(f"v2_calibrate_{fold}{seed}", call([*cli, "calibrate", *common]), state=setup_state, log_path=setup_log)
    run_step("v2_register", call([*cli, "register", "--profile", "v2_full", *inputs, "--out", str(v2_root), "--registry", str(work/"v2_registry.json")]), state=setup_state, log_path=setup_log)
    run_step("v2_infer", call([*cli, "infer", "--profile", "v2_full", "--synthetic", "--allowlist", str(allow), "--dev-source", "synthetic_dev", "--data-root", str(data), "--all-dev", "--registry", str(work/"v2_registry.json"), "--out", str(v2_root), "--device", "cpu"]), state=setup_state, log_path=setup_log)
    u1_root = work/"u1"
    common = ["--profile", "v3_large", "--fold", "A", "--seed", "0", *inputs, "--out", str(u1_root), "--device", "cpu"]
    run_step("u1_train", call([*cli, "train", *common, *sets(TINY_U1)]), state=setup_state, log_path=setup_log)
    run_step("u1_calibrate", call([*cli, "calibrate", *common]), state=setup_state, log_path=setup_log)
    import eval_seg
    eval_seg.evaluate(v2_root=v2_root/"pred", v1_root=work/"pred_v1", out_dir=work/"ref_ensembles", data_root=data, allowlist_path=allow, folds_path=work/"folds.json", dev_source="synthetic_dev", log=lambda _: None)
    synthetic_member_reference(work/"ref_members.json", v2_root=v2_root, subjects=synthetic_tree.DEV, data_root=data)
    synthetic_u1_reference(work/"ref_u1.json", member=u1_root/"foldA_s0", subjects=synthetic_tree.DEV, data_root=data)
    out = work/"out"
    (work/"train_overrides.json").write_text(json.dumps(TINY_UF)+"\n", encoding="utf-8")
    nn_out = work/"nn_out"
    synthetic_nn_outputs(nn_out=nn_out, raw_dir=work/"nn_raw", subjects=synthetic_tree.DEV, data_root=data)      # NN's records exist before the chain, as nnunet_v1's do; the raw files are what the inference step would write

    def nn_stand_in(spec):
        directory = Path(spec.out)/"nn"
        (directory/"dev_input").mkdir(parents=True)
        shutil.copytree(work/"nn_raw", directory/"raw")
        return 0

    nn_extra = ["--synthetic", "--allowlist", str(allow), "--folds", str(work/"folds.json"), "--dev-source", "synthetic_dev", "--data-root", str(data), "--nn-out", str(nn_out)]
    eval_extra = ["--synthetic", "--train-overrides", str(work/"train_overrides.json"), "--nn-manifest", str(nn_out/"prepare_manifest.json"), "--v1-pred", str(work/"pred_v1"), "--v2-pred", str(v2_root/"pred"), "--u1-member", str(u1_root/"foldA_s0"),
                  "--reference-ensembles", str(work/"ref_ensembles"/"eval_seg.json"), "--reference-members", str(work/"ref_members.json"), "--reference-u1", str(work/"ref_u1.json"), "--v2-root", str(v2_root), "--v2-registry", str(work/"v2_registry.json"),
                  "--allowlist", str(allow), "--folds", str(work/"folds.json"), "--dev-source", "synthetic_dev", "--data-root", str(data)]
    spec = Spec(out=out, cwd=CODE_ROOT, python=sys.executable, device="cpu", inputs=inputs, train_config=dict(TINY_UF), nn_extra=nn_extra, eval_extra=eval_extra, v2_root=v2_root, v2_registry=work/"v2_registry.json", nn_infer=nn_stand_in, gate=False, env=env)
    return types.SimpleNamespace(spec=spec, work=work, data=data, allowlist=allow, v2_root=v2_root, u1_root=u1_root, nn_out=nn_out, inputs=inputs, eval_extra=eval_extra, nn_extra=nn_extra)


def dry_run(scratch=None):
    import decide_uf
    work = Path(scratch) if scratch else Path(tempfile.mkdtemp(prefix="uf_dryrun_"))
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    problems = []
    print("== the lead's start strings (printed, not run)")
    print("prepare:", shlex.join([uc.PY, "-B", "experiments/uf/run_uf.py", "prepare"]))
    print("chain:", shlex.join(launch_argv()))
    print(f"RuntimeMaxSec {runtime_max_seconds()} s ({runtime_max_seconds()/3600:.1f} h): every step at its limit {json.dumps(TIMEOUTS)} and every gate closed to {WAIT_MINUTES} minutes; a safety valve, not a budget")
    print(f"\n== the chain of the unit {UNIT} (printed, not run); hold file {uc.OUT/'hold'}")
    for name, kind, argv in chain_steps(Spec(out=uc.OUT, cwd=uc.OUT/"snapshot")):
        gated = "gate, " if kind in GPU_STEPS else ""
        print(f"{name} [{gated}hold, limit {TIMEOUTS.get(kind, 'in-process')} s]: {shlex.join(argv) if argv else 'in this process: uf_common.member_record_problems'}")
    print("\n== the members against v2 (the real folds file and v2's records; ids only)")
    checks = repository_checks()
    print(json.dumps(checks, indent=1))
    if not all(m["same_train"] and m["same_calib"] and m["same_contact"] and m["n_train"] == 98 and m["n_calib"] == 8 for m in checks["members"].values()):
        problems.append("the members do not match v2's")
    if checks["refusals"]["admitted"] != 20 or not all(checks["refusals"][k].startswith("refused") for k in ("T0", "holdout_B", "training")) or checks["refusals"]["overlap_with_training"] or checks["refusals"]["t0_in_training"]:
        problems.append("the admission of the development subjects")
    if not all(checks["references"].values()):
        problems.append(f"a reference is not the pinned file: {checks['references']}")
    if not (checks["nn"].get("checkpoint_sha256_matches") and checks["nn"].get("pred_dev") and checks["nn"].get("manifest_contact") == 106):
        problems.append(f"the inputs of the nnU-Net step: {checks['nn']}")
    print("\n== the estimate from U1's measurement (unet35_v3_large/foldA_s0, its training subjects only)")
    measured = u1_measurement()
    print(json.dumps(dict(measured=measured, estimate=estimate(measured)), indent=1))
    print("\n== the gate on this machine now (read only; it decides nothing here)")
    print(json.dumps(gate_problems(), indent=1))
    print("\n== the real chain is refused here: it needs a verified snapshot")
    try:
        require_snapshot()
        print("NOT REFUSED")
        problems.append("the chain was not refused outside a snapshot")
    except CodeNotClean as exc:
        print(f"refused: {str(exc)[:160]}")
    print(f"\n== the read lock of the development truth, {uf_lock.DEV_READ_LOCK}: {'EXISTS' if uf_lock.DEV_READ_LOCK.exists() else 'none'} (synthetic runs do not touch it)")
    if uf_lock.DEV_READ_LOCK.exists():
        problems.append("the read lock exists")
    print("\n== the chain on synthetic subjects, CPU")
    synthetic = build_synthetic(work, env)
    spec, out = synthetic.spec, synthetic.spec.out
    set_run_log(out)
    from bayesreg.seg.unet.data import load_full_folds
    plan = members_plan(spec, load_full_folds(work/"folds_full_v2.json"))
    print("members plan (synthetic):", json.dumps({k: {kk: v[kk] for kk in ("fold", "seed", "v2_namesake", "same_as_v2", "n_train", "n_calib")} for k, v in plan["members"].items()}))
    (out/"hold").write_text("dry run: the hold file\n")
    try:
        run_chain(spec)
        problems.append("the hold file did not stop the chain")
    except Stop as exc:
        print(f"hold: {exc}")
    (out/"hold").unlink()
    run_chain(spec)
    run_chain(spec)                                               # a second pass skips everything
    checks_of_members = {n: json.loads((out/n/"uf_check.json").read_text())["problems"] for n in uc.MEMBER_NAMES}
    evaluation = json.loads((out/"eval_uf"/"eval_uf.json").read_text(encoding="utf-8"))
    result = json.loads((out/"analysis"/"decision_uf.json").read_text(encoding="utf-8"))
    print("member checks:", json.dumps(checks_of_members))
    print("controls:", json.dumps(evaluation["controls"]))
    print("gate:", json.dumps({k: evaluation["nn"]["convert_record"][k] for k in ("gate", "tolerance", "n_differing_total", "n_voxels_total")}))
    ends = [json.loads(p.read_text(encoding="utf-8")) for p in [out/"queue_end.json"]+sorted(out.glob("queue_end_*.json"))]
    print("chain ends:", json.dumps([{k: e[k] for k in ("stop", "finished", "reading")} | dict(nn_raw=(e["nn_raw"] or {}).get("text")) for e in ends], ensure_ascii=False))
    order = [line.split()[2].rstrip(":") for line in (out/"chain.log").read_text(encoding="utf-8").splitlines() if len(line.split()) > 2 and line.split()[1] == "start"]
    print("order of the steps:", " ".join(order))
    if order[:4] != ["control_check", "nn_infer", "nn_convert", "train_foldA_s0"] or not ends[-1]["finished"]:
        problems.append("the order of the chain or its end")
    print("decision:", json.dumps({k: result.get(k) for k in ("reading", "reading_text", "not_decidable_because")}, ensure_ascii=False))
    print((out/"analysis"/"summary_uf.md").read_text(encoding="utf-8"))
    ok_synthetic = (not any(checks_of_members.values()) and all(abs(v) == 0. for d in evaluation["controls"]["differences"].values() for v in d.values()) and result["reading"] == decide_uf.NOT_DECIDABLE
                    and result["not_decidable_because"] == ["2 名受试者，不是 20"] and evaluation["nn"]["convert_record"]["gate"] == "passed")
    if not ok_synthetic:
        problems.append("the synthetic chain")
    print(f"\n== dry run problems: {'; '.join(problems) if problems else 'none'}; work {work}")
    return 0 if not problems else 8


# ---- command line ----------------------------------------------------------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scratch", type=Path, default=None)
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("prepare")
    p.add_argument("--repo", type=Path, default=uc.MAIN_REPO)
    p.add_argument("--out", type=Path, default=uc.OUT)
    p = sub.add_parser("members-plan", help="(prepare runs it from the snapshot) members_plan.json")
    p.add_argument("--out", type=Path, default=uc.OUT)
    p = sub.add_parser("print-launch")
    p.add_argument("--out", type=Path, default=uc.OUT)
    p.add_argument("--wait-minutes", type=int, default=WAIT_MINUTES)
    p = sub.add_parser("chain")
    p.add_argument("--out", type=Path, default=uc.OUT)
    p.add_argument("--wait-minutes", type=int, default=WAIT_MINUTES)
    for name in ("status", "estimate"):
        sub.add_parser(name).add_argument("--out", type=Path, default=uc.OUT)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.dry_run:
            return dry_run(args.scratch)
        if args.command == "prepare":
            prepare(args.out, args.repo)
        elif args.command == "members-plan":
            require_snapshot()
            print(json.dumps(members_plan(real_spec(args.out), real_split())["members"], indent=1))
        elif args.command == "print-launch":
            print(shlex.join(launch_argv(out=args.out, wait_minutes=args.wait_minutes)))
        elif args.command == "chain":
            cmd_chain(args)
        elif args.command == "status":
            cmd_status(args)
        elif args.command == "estimate":
            measured = u1_measurement()
            print(json.dumps(dict(measured=measured, estimate=estimate(measured)), indent=1))
        else:
            build_parser().print_help()
            return 2
    except (Stop, CodeNotClean, uf_lock.DevReadLocked) as exc:
        log(f"STOP: {exc}")
        return 8
    return 0


if __name__ == "__main__":
    sys.exit(main())
