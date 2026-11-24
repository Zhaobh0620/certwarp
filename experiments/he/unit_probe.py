"""HE2 (runs/20261107-he2/任务书.md item 6; contract C22.B4-6 (d); H2, H3): what the unit of this process is, as the machine reports it.  The standard library only: the HE supervisor and the evaluation
call probe() in their own process; the nested unit of configuration C runs this file with python -I -B at its start and at its end (firewall_c.probed_inner_command).

    python -I -B experiments/he/unit_probe.py --when start|end --out FILE        writes FILE once (JSON); exit 0 whatever was read (a probe never fails a run)

What it reads, and nothing else:
  /proc/self/cgroup                   the cgroup v2 path of this process (the line 0::<path>)
  /sys/fs/cgroup/<path>/memory.*      max, swap.max, peak, swap.peak, current, events, swap.events: the kernel's numbers for the whole cgroup (every process of the unit, children included); memory.peak is
                                      the peak since the cgroup was created, memory.events counts oom and oom_kill.  memory.peak and memory.current are the cgroup's accounting: they include the page
                                      cache the unit's writes left (he_smoke_v2), not only the processes' resident memory
  /sys/fs/cgroup/<path>/memory.stat   only its anon and file lines (HE2 clean): the anonymous and the file-backed bytes now, recorded, never judged
  systemctl --user show <unit>        Id, MemoryMax, MemorySwapMax, OOMPolicy, ...: the properties systemd applied; asked only when the cgroup's leaf is a tag-*.service (this project's units)
What cannot be read is recorded as 读不到 with its reason; nothing is inferred or filled in.  A process that is not in a unit of its own (a test, a terminal) is in its session's cgroup: in_tag_unit false, and
those numbers are not a unit's.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

SCHEMA = 'bayesreg.he.unit_probe.v1'
PROC = '/proc/self/cgroup'
CGROUP_ROOT = '/sys/fs/cgroup'
CGROUP_FILES = ('memory.max', 'memory.swap.max', 'memory.peak', 'memory.swap.peak', 'memory.current', 'memory.events', 'memory.swap.events')
STAT_FILE, STAT_KEYS = 'memory.stat', ('anon', 'file')     # HE2 clean (the smoke result's review): the anonymous and the file (page cache) bytes of the cgroup now; recorded only, never judged
SYSTEMD_PROPERTIES = ('Id', 'MemoryMax', 'MemorySwapMax', 'OOMPolicy', 'CPUQuotaPerSecUSec', 'RuntimeMaxUSec', 'ControlGroup', 'MemoryAccounting')
UNREAD = '读不到'
TAG_UNIT = re.compile(r'^tag-[A-Za-z0-9_.:@-]+\.service$')
SHOW_TIMEOUT = 10.


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def cgroup_path(proc=PROC):
    """(path, None) from the cgroup v2 line 0::<path>, or (None, the reason)."""
    try:
        text = Path(proc).read_text()
    except OSError as exc:
        return None, f'{UNREAD}: {proc}: {type(exc).__name__}: {exc}'
    lines = [line[3:] for line in text.splitlines() if line.startswith('0::')]
    if len(lines) != 1 or not lines[0].startswith('/'):
        return None, f'{UNREAD}: {proc} has no single cgroup v2 line 0::/<path> ({text.strip()[:200]!r})'
    return lines[0], None


def parse_flat(text):
    """memory.events and memory.swap.events: 'key value' lines -> {key: int}; None when a line is not of that form."""
    out = {}
    for line in (text or '').splitlines():
        parts = line.split()
        if len(parts) != 2 or not parts[1].lstrip('-').isdigit():
            return None
        out[parts[0]] = int(parts[1])
    return out


def parse_bytes(text):
    """memory.max and the like: an int, the word max, or None when it is neither."""
    value = (text or '').strip()
    return value if value == 'max' else (int(value) if value.isdigit() else None)


def read_cgroup(path, root=CGROUP_ROOT):
    """({file: raw text}, {file: why it was not read}) for the memory files of the cgroup `path`."""
    files, unread = {}, {}
    directory = Path(root+path)
    for name in CGROUP_FILES:
        try:
            files[name] = (directory/name).read_text()
        except OSError as exc:
            unread[name] = f'{UNREAD}: {type(exc).__name__}: {exc}'
    return files, unread


def read_stat(path, root=CGROUP_ROOT):
    """({anon, file: bytes or None}, None) from the cgroup's memory.stat, or (None, why it was not read).  memory.peak and memory.current count both, with the page cache the unit's writes left (he_smoke_v2);
    this tells them apart at the moment of the probe (memory.stat holds no peak of either)."""
    try:
        text = Path(root+path, STAT_FILE).read_text()
    except OSError as exc:
        return None, f'{UNREAD}: {type(exc).__name__}: {exc}'
    values = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in STAT_KEYS and parts[1].isdigit():
            values[parts[0]] = int(parts[1])
    return {k: values.get(k) for k in STAT_KEYS}, None if all(k in values for k in STAT_KEYS) else f'{UNREAD}: {STAT_FILE} has no {[k for k in STAT_KEYS if k not in values]}'


def systemd_show(unit, run=subprocess.run, timeout=SHOW_TIMEOUT):
    """({property: value}, None) from systemctl --user show, or (None, the reason)."""
    env = dict(os.environ)
    env.setdefault('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}')
    argv = ['systemctl', '--user', 'show', unit, '--no-pager', '--property='+','.join(SYSTEMD_PROPERTIES)]
    try:
        done = run(argv, capture_output=True, text=True, timeout=timeout, env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f'{UNREAD}: {type(exc).__name__}: {exc}'
    if done.returncode:
        return None, f'{UNREAD}: systemctl exit {done.returncode}: {(done.stderr or "").strip()[:300]}'
    props = dict(line.split('=', 1) for line in done.stdout.splitlines() if '=' in line)
    if props.get('Id') != unit:
        return None, f'{UNREAD}: systemctl answered for {props.get("Id")!r}, not {unit!r}'
    return {k: props.get(k) for k in SYSTEMD_PROPERTIES}, None


def probe(when, *, proc=PROC, root=CGROUP_ROOT, show=None):
    """The record of this process's unit now (see the module docstring)."""
    path, why = cgroup_path(proc)
    record = dict(schema=SCHEMA, when=when, utc=utc(), pid=os.getpid(), cgroup=path, cgroup_unread=why, unit=None, in_tag_unit=False, files={}, unread={}, events=None, swap_events=None,
                  peak_bytes=None, max_bytes=None, swap_max_bytes=None, memory_stat=None, memory_stat_unread=None, systemd=None, systemd_unread=None,
                  rule='the kernel\'s cgroup files and systemd\'s properties as read; 读不到 with its reason otherwise; nothing inferred')
    if path is None:
        record.update(systemd_unread=f'{UNREAD}: no cgroup')
        return record
    leaf = path.rstrip('/').rsplit('/', 1)[-1]
    record.update(unit=leaf, in_tag_unit=bool(TAG_UNIT.match(leaf)))
    files, unread = read_cgroup(path, root)
    record.update(files=files, unread=unread, events=parse_flat(files.get('memory.events')) if 'memory.events' in files else None,
                  swap_events=parse_flat(files.get('memory.swap.events')) if 'memory.swap.events' in files else None,
                  peak_bytes=parse_bytes(files.get('memory.peak')) if 'memory.peak' in files else None, max_bytes=parse_bytes(files.get('memory.max')) if 'memory.max' in files else None,
                  swap_max_bytes=parse_bytes(files.get('memory.swap.max')) if 'memory.swap.max' in files else None)
    record['memory_stat'], record['memory_stat_unread'] = read_stat(path, root)
    if record['in_tag_unit']:
        record['systemd'], record['systemd_unread'] = (show or systemd_show)(leaf)
    else:
        record['systemd_unread'] = f'{UNREAD}: not asked: {leaf!r} is not a tag-*.service (not a unit of this project)'
    return record


def write_once(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as f:
        f.write(json.dumps(value, indent=1, ensure_ascii=False)+'\n')
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--when', required=True)
    ap.add_argument('--out', required=True)
    o = ap.parse_args(argv)
    try:
        write_once(o.out, probe(o.when))
    except Exception as exc:                            # noqa: BLE001 - a probe never fails the run it watches
        print(f'unit_probe: {type(exc).__name__}: {exc}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
