"""Side-channel memory sampler of one systemd user unit (work package MEM; reads /proc and /sys only).

    python -B experiments/mem/sample.py --unit tag-mem-M01 --out samples.csv [--interval 1] [--wait-start 120]

Every --interval seconds one CSV row: UTC time, seconds since the first row, the unit's cgroup v2 memory.current,
memory.peak, memory.swap.current, memory.swap.peak and memory.stat anon / file / anon_thp / shmem (MiB), the main
process's VmRSS, VmHWM, VmSwap, RssAnon, RssFile (MiB; /proc/<MainPID>/status), and the GPU's memory.used (MiB, the
whole device: WSL gives no per-process figure; nvidia-smi, every --gpu-every rows).  The sampler waits up to
--wait-start seconds for the unit's cgroup to appear and stops when it disappears (the unit ended).  It never writes
anything except --out (which must not exist) and never signals the unit.
"""
import argparse
import csv
import datetime
import os
from pathlib import Path
import subprocess
import time

APP = Path("/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice")
SMI = "/usr/lib/wsl/lib/nvidia-smi"
MIB = 1024.*1024.
STAT = ("anon", "file", "anon_thp", "shmem")
STATUS = ("VmRSS", "VmHWM", "VmSwap", "RssAnon", "RssFile")
FIELDS = (["utc", "elapsed_s", "cg_mb", "cg_peak_mb", "cg_swap_mb", "cg_swap_peak_mb"] + [f"cg_{k}_mb" for k in STAT]
          + ["pid"] + [f"{k}_mb" for k in STATUS] + ["gpu_used_mb"])


def read_mib(path):
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return None if text == "max" else round(int(text)/MIB, 2)


def cgroup_row(directory):
    row = dict(cg_mb=read_mib(directory/"memory.current"), cg_peak_mb=read_mib(directory/"memory.peak"),
               cg_swap_mb=read_mib(directory/"memory.swap.current"),
               cg_swap_peak_mb=read_mib(directory/"memory.swap.peak"))
    try:
        for line in (directory/"memory.stat").read_text().splitlines():
            key, _, value = line.partition(" ")
            if key in STAT:
                row[f"cg_{key}_mb"] = round(int(value)/MIB, 2)
    except OSError:
        pass
    return row


def main_pid(directory):
    """The first PID of cgroup.procs (systemd-run starts the python process as the unit's only process)."""
    try:
        procs = [int(p) for p in (directory/"cgroup.procs").read_text().split()]
    except (OSError, ValueError):
        return None
    return procs[0] if procs else None


def status_row(pid):
    row = {}
    if pid is None:
        return row
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in STATUS:
                row[f"{key}_mb"] = round(int(rest.split()[0])/1024., 2)
    except OSError:
        pass
    return row


def gpu_used():
    try:
        out = subprocess.run([SMI, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=10, check=False).stdout.strip().splitlines()
        return float(out[0]) if out else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--unit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--interval", type=float, default=1.)
    parser.add_argument("--wait-start", type=float, default=120.)
    parser.add_argument("--gpu-every", type=int, default=2, help="query nvidia-smi every N rows (0: never)")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out}: exists (nothing is overwritten)")
    directory = APP/f"{args.unit}.service"
    deadline = time.monotonic()+args.wait_start
    while not directory.is_dir():
        if time.monotonic() > deadline:
            raise SystemExit(f"{directory}: did not appear within {args.wait_start:g} s")
        time.sleep(.2)
    start = time.monotonic()
    rows = 0
    gpu = None
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        while directory.is_dir():
            tick = time.monotonic()
            row = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"),
                       elapsed_s=round(tick-start, 3))
            row.update(cgroup_row(directory))
            pid = main_pid(directory)
            row["pid"] = pid
            row.update(status_row(pid))
            if args.gpu_every and rows % args.gpu_every == 0:
                gpu = gpu_used()
            row["gpu_used_mb"] = gpu
            if row.get("cg_mb") is None and not directory.is_dir():
                break
            writer.writerow(row)
            handle.flush()
            rows += 1
            time.sleep(max(0., args.interval-(time.monotonic()-tick)))
    print(f"{rows} rows -> {out}")
    return 0


if __name__ == "__main__":
    os.nice(5)
    raise SystemExit(main())
