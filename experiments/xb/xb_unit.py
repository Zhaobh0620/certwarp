"""XB: the in-unit supervisor (prereg revision 1c item 1: the end probe reads memory.events before the unit ends).  The unit's main process; the standard library and holdout_common's discriminator
helpers only (no numpy, no bayesreg):

    python -I -B experiments/xb/xb_unit.py --record-dir DIR --unit NAME -- CMD ARG...

It writes DIR/unit_start.json (the time, the uptime, experiments/he/unit_probe.probe: the cgroup's path and memory files, systemd's properties of the unit), runs CMD as its child, and when the child
ends -- or when systemd sends SIGTERM (RuntimeMaxSec, a stop) -- writes DIR/unit_end.json: the child's exit code or signal, whether this process killed it (after a SIGTERM and 20 s of grace; such a
-9 is not OOM evidence), the times (UTC, uptime, monotonic seconds), the end probe (memory.events of the unit's cgroup while the unit is still alive: oom, oom_kill), the kernel's oom-kill lines and
dxg lines in the window since the start (dmesg; None when it cannot be read), and dmesg_readable.  It does not classify: the chain (and the limit experiment) read systemd's Result, this record and
the journal with holdout_common.classify_unit_end.  Both files are written once.  Exit: the child's code; 128 + the signal when it was killed; 143 after a SIGTERM.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent/'he', HERE.parent/'holdout'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import holdout_common as hc  # noqa: E402
import unit_probe  # noqa: E402

SCHEMA = 'bayesreg.xb.unit.{}.v1'
GRACE_SECONDS = 20.


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def uptime():
    try:
        return float(Path('/proc/uptime').read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def dmesg():
    try:
        done = subprocess.run(['dmesg'], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def write_once(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(document, indent=1, ensure_ascii=False, default=str)+'\n')
    return path


def supervise(cmd, *, record_dir, unit, probe=None, read_dmesg=None, popen=subprocess.Popen, grace=GRACE_SECONDS):
    probe = unit_probe.probe if probe is None else probe
    read_dmesg = dmesg if read_dmesg is None else read_dmesg
    record_dir = Path(record_dir)
    t0, m0 = uptime(), time.monotonic()
    start = probe('start')
    write_once(record_dir/'unit_start.json', dict(schema=SCHEMA.format('start'), unit=unit, utc=utc(), uptime=t0, cmd=list(cmd), pid=os.getpid(), probe=start))
    child = popen(list(cmd))
    terminated, killed_by_me = [], False

    def on_term(signum, frame):
        terminated.append(signum)
        try:
            child.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass
    signal.signal(signal.SIGTERM, on_term)
    while True:
        try:
            rc = child.wait(timeout=1.)
            break
        except subprocess.TimeoutExpired:
            if terminated:
                try:
                    rc = child.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    child.kill()
                    killed_by_me = True
                    rc = child.wait()
                break
    end = probe('end')
    text = read_dmesg()
    document = dict(schema=SCHEMA.format('end'), unit=unit, utc=utc(), uptime=uptime(), uptime_start=t0, seconds=round(time.monotonic()-m0, 3), exit_code=rc,
                    killed_by_signal=-rc if rc is not None and rc < 0 else None, killed_by_supervisor=killed_by_me, sigterm_received=bool(terminated),
                    cgroup=(end or {}).get('cgroup') if isinstance(end, dict) else None, memory_events=(end or {}).get('events') if isinstance(end, dict) else None, probe=end,
                    oom_lines=hc.parse_oom(text, t0 or 0.), dxg_lines=hc.dxg_lines(text, t0 or 0.), dmesg_readable=text is not None,
                    rule='evidence only (holdout_common.classify_unit_end reads it with systemd\'s Result and the journal); memory.events are the unit cgroup\'s counts at the end')
    write_once(record_dir/'unit_end.json', document)
    if terminated:
        return 143
    return 128-rc if rc < 0 else rc


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--' not in argv:
        print('usage: xb_unit.py --record-dir DIR --unit NAME -- CMD ARG...', file=sys.stderr)
        return 2
    split = argv.index('--')
    ap = argparse.ArgumentParser()
    ap.add_argument('--record-dir', required=True)
    ap.add_argument('--unit', required=True)
    o = ap.parse_args(argv[:split])
    cmd = argv[split+1:]
    if not cmd:
        return 2
    return supervise(cmd, record_dir=o.record_dir, unit=o.unit)


if __name__ == '__main__':
    sys.exit(main())
