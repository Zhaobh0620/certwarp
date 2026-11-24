"""PQH (runs/20261109-pqh/任务书.md; contract C22.H, revision 7): the command line of PQ on the B-line hold-out and of its smoke segments.

    python -B experiments/pq/run_pqh.py dry-run {members,out,members-dev,out-dev,out-dev-ivb,out-dev-ivc,p3-replay}
        no image, no truth, no GPU, nothing written: every check before the lock, each on its own (a problem is reported and the next check goes on), the admission included (on main without BZ's unseal
        record a hold-out unit is refused there), the start string, the estimates.
    python -B experiments/pq/run_pqh.py launch {members,out,members-dev,out-dev,out-dev-ivb,out-dev-ivc,p3-replay} [--resume AUTH] [--print]
        the lead's start: systemd-run --wait of the unit (C22.H5-2, C22.H6-3: MemoryMax, MemorySwapMax 4G, OOMPolicy continue, RuntimeMaxSec, nice 5) whose main process is XB's in-unit supervisor
        experiments/xb/xb_unit.py (unit_start.json, unit_end.json with memory.events and the kernel's oom lines), around `run_pqh.py <unit>`; afterwards systemd's Result, unit_end.json and the journal
        through holdout_common.classify_unit_end (D123: OOM evidence or a timeout -> the unit stops, to the lead; unreadable evidence -> needs_lead_rejudgment).  --print prints the start string only.
    python -B experiments/pq/run_pqh.py members [--resume AUTH] | out [--resume AUTH]      (hold-out; only inside its unit)
    python -B experiments/pq/run_pqh.py members-dev | out-dev | p3-replay                   (the smoke's segments (i), (ii), (iii) under he_smoke_v3; only inside its unit)
    python -B experiments/pq/run_pqh.py out-dev-ivb | out-dev-ivc                           (SM3: the PQ unit on the fallback pairs of the smoke's (iv-b), (iv-c); only inside its unit)
    python -B experiments/pq/run_pqh.py replay-gate                                         (the development replay gate, pqh_replay.main run; only inside tag-hb-pq-replay; review m6)

The two families are subcommands, never an option that changes a path (C23.0-3, C22.H8 (ii)).  Exit codes: 0 done; 2 refused before the lock (or usage); 8 stopped after the lock; 9 a dry run found problems;
20 an infrastructure refusal before the lock; launch: the unit's exit, or 30 when the unit's end is a limit, an infrastructure failure or needs the lead.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pqh_inputs as hin  # noqa: E402
import pqh_pins as hp  # noqa: E402
import pqh_units as hu  # noqa: E402

REPO = HERE.parents[1]
PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'
FROZEN = '/home/rmcfz/seq_vp_tag_results/frozen_eb808be'
COMMANDS = dict(members=('holdout', 'members'), out=('holdout', 'out'), **{'members-dev': ('dev', 'members'), 'out-dev': ('dev', 'out'), 'p3-replay': ('dev', 'p3_replay'),
                                                                              'replay-gate': ('dev', 'replay'), 'out-dev-ivb': ('dev', 'out'), 'out-dev-ivc': ('dev', 'out')})
SEGMENT_OF = {command: segment for segment, command in hp.DEV_OUT_COMMANDS.items()}     # SM3: the dev out commands and the smoke segment whose copy each reads ('out-dev': (ii))
ESTIMATES = dict(members='〔估算〕GPU：开发集 20 人约 6.4 分钟（tag-pq-members），100 人约 30–40 分钟；D: 约 2.75 GB（V 每人约 27.5 MB）',
                 out='〔估算〕CPU：开发集 19 对约 22 分钟（tag-pq-out），50 对约 1 小时，加 D113 (a) 的逐对重建与从 D: 读大文件各几分钟；D: 约 13 GB（每对 pq.npz 约 55 MB、baselines.npz 约 200 MB）',
                 p3_replay='〔估算〕CPU 一对约 2–3 分钟', replay='〔估算〕CPU：19 对的真值与数组，逐对，约 10–20 分钟')
FAILURE_EXIT = 30


def layout_for(mode, code=REPO, command=None):
    """The command's layout: the hold-out's, or the dev mode's of the command's smoke segment (SM3: out-dev-ivb, out-dev-ivc on (iv-b), (iv-c); every other dev command on (ii))."""
    return hin.holdout_layout(code) if mode == 'holdout' else hin.dev_layout(code, segment=SEGMENT_OF.get(command, 'ii'))


def unit_name(command):
    mode, unit = COMMANDS[command]
    if unit == 'replay':
        return hp.REPLAY_UNIT
    if command in SEGMENT_OF:
        return hp.DEV_OUT_UNITS[SEGMENT_OF[command]]
    return (hp.UNITS if mode == 'holdout' else hp.DEV_UNITS)[unit]


def spec_of(command):
    _, unit = COMMANDS[command]
    if unit == 'replay':
        return hp.REPLAY_UNIT_SPEC
    return hp.UNIT_SPECS['members' if unit == 'members' else 'out']


def units_dir(command, code=REPO):
    """Where launch writes the unit's records: <pq records>/units/, the replay gate's under its own root (pq_replay_v1/units/)."""
    mode, unit = COMMANDS[command]
    return Path(hp.REPLAY_ROOT)/'units' if unit == 'replay' else Path(layout_for(mode, code, command).pq_small)/'units'


def start_argv(command, *, code=REPO, record_dir, resume=None):
    """The unit's start string (C22.H5-2, C22.H6-3; HE's nice 5): systemd-run --wait, the in-unit supervisor xb_unit.py, then this file's unit command."""
    mode, unit = COMMANDS[command]
    spec = spec_of(command)
    env = dict(OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, NUMBA_NUM_THREADS=1, PYTHONDONTWRITEBYTECODE=1,
               PYTHONPATH=str(code) if unit == 'members' else f'{FROZEN}:{code}')
    inner = [PY, '-B', str(Path(code)/'experiments/pq/run_pqh.py'), command]+([] if resume is None else ['--resume', str(resume)])
    return ['systemd-run', '--user', '--wait', '--collect', f'--unit={unit_name(command)}', f"--property=MemoryMax={spec['memory_max']}", f"--property=MemorySwapMax={spec['memory_swap_max']}",
            f"--property=OOMPolicy={spec['oom_policy']}", f"--property=RuntimeMaxSec={spec['runtime_max_sec']}", f"--nice={spec['nice']}", f'--working-directory={code}',
            *[f'--setenv={k}={v}' for k, v in env.items()], PY, '-I', '-B', str(Path(code)/'experiments/xb/xb_unit.py'), '--record-dir', str(record_dir), '--unit', unit_name(command), '--', *inner]


def unit_result(text):
    """systemd-run --wait's "Finished with result: <r>" (experiments/xb/xb_chain.unit_result's pattern)."""
    m = re.search(r'^Finished with result: (\S+)', text or '', re.MULTILINE)
    return m.group(1) if m else None


def unit_reading(*, unit, result_line, unit_end, oom_lines, journal):
    """holdout_common.classify_unit_end on the unit's evidence, the way experiments/xb/xb_chain.unit_reading reads it (the supervisor's end probe and oom lines; a child's -9 the supervisor did not send)."""
    import holdout_common as hc
    events = None if unit_end is None else unit_end.get('memory_events')
    cgroup = None if unit_end is None else unit_end.get('cgroup')
    step_kill = bool(unit_end is not None and unit_end.get('exit_code') == -9 and not unit_end.get('killed_by_supervisor'))
    if oom_lines is None and unit_end is not None and unit_end.get('dmesg_readable'):
        oom_lines = unit_end.get('oom_lines')
    return hc.classify_unit_end(unit=unit, systemd_result=result_line, oom_lines=oom_lines, events=events, journal=journal, step_kill=step_kill, unit_cgroup=cgroup)


class System:
    """What the launcher does to and reads from the machine (the tests replace it)."""

    def run(self, argv, log_path):
        with open(log_path, 'ab') as stream:
            return subprocess.run([str(a) for a in argv], stdout=stream, stderr=subprocess.STDOUT).returncode

    def uptime(self):
        return float(Path('/proc/uptime').read_text().split()[0])

    def dmesg(self):
        done = subprocess.run(['dmesg'], capture_output=True, text=True)
        return done.stdout if done.returncode == 0 else None

    def journal(self, unit, since_epoch):
        done = subprocess.run(['journalctl', '--user', '-u', unit, '--since', f'@{int(since_epoch)}', '--no-pager', '-o', 'short-iso'], capture_output=True, text=True)
        return done.stdout+done.stderr

    def now(self):
        return time.time()


def launch(command, *, resume=None, system=None, code=REPO, log=print):
    """Start the unit and read its end (D123 (b), prereg revision 1c item 1).  Writes <pq records>/units/<unit>_<utc>/{launch.log, unit_reading.json} (the supervisor writes unit_start.json and
    unit_end.json there).  Returns the unit's exit code, or FAILURE_EXIT when the end is a limit, an infrastructure failure or needs the lead (the unit stops; nothing is rerun)."""
    import holdout_common as hc
    system = System() if system is None else system
    name = unit_name(command)
    record_dir = units_dir(command, code)/f"{name}_{hc.utc().replace(':', '').replace('-', '')}"
    hin.guard_write(record_dir)
    record_dir.mkdir(parents=True, exist_ok=False)
    argv = start_argv(command, code=code, record_dir=record_dir, resume=resume)
    return run_unit(command, name, argv, record_dir, system=system, log=log)


def run_unit(command, name, argv, record_dir, *, system, log=print):
    """launch() after the start string, unchanged (moved here so that the controlled limit experiment's second batch runs PQ's two units through the launcher's own code, as it runs the review
    unit's STOP through hb_review.unit_stop_record; experiments/xb/xb_limit_batch2.py, P0S LB2 of 11-14): systemd-run --wait of `argv` (its log record_dir/launch.log, the supervisor's records in
    record_dir), the reading (unit_reading: holdout_common.classify_unit_end), record_dir/unit_reading.json once; FAILURE_EXIT when the end is a limit, an infrastructure failure or needs the
    lead, else the unit's exit code.  `record_dir` exists (launch made it)."""
    import holdout_common as hc
    t0, epoch = system.uptime(), system.now()
    log('start: ' + ' '.join(argv))
    rc = system.run(argv, record_dir/'launch.log')
    text = (record_dir/'launch.log').read_text(encoding='utf-8', errors='replace') if (record_dir/'launch.log').is_file() else ''
    end_path = record_dir/'unit_end.json'
    unit_end = json.loads(end_path.read_text(encoding='utf-8')) if end_path.is_file() else None
    status, kind, evidence = unit_reading(unit=name, result_line=unit_result(text), unit_end=unit_end, oom_lines=hc.parse_oom(system.dmesg(), t0), journal=system.journal(name, epoch))
    reading = dict(schema=hp.SCHEMA.format('unit_reading'), command=command, unit=name, argv=argv, returncode=rc, status=status, kind=kind, evidence=evidence, rule='holdout_common.classify_unit_end '
                   f'(rule {hc.LIMIT_RULE}); D123 (b): OOM evidence or a timeout -> the unit stops, to the lead; a resume only through a committed resume record, once; unreadable -> needs_lead_rejudgment',
                   utc=hc.utc())
    hc.write_json_once(hin.guard_write(record_dir/'unit_reading.json'), reading)
    if status is not None:
        log(f'{name}: {status} ({kind}): STOP, to the lead (D123 (b)); {record_dir/"unit_reading.json"}')
        return FAILURE_EXIT
    log(f'{name}: exit {rc}; {record_dir}')
    return rc


# ---- the dry run --------------------------------------------------------------------------------------------------------------------------------
def dry_run(command, *, code=REPO, log=print, deps=None):
    """Every check before the lock, each on its own; nothing written, no lock, no image, no truth, no GPU.  Returns 0 or 9."""
    import holdout_common as hc
    import pq_io as io_
    from bayesreg.io import sealed
    mode, unit = COMMANDS[command]
    if unit == 'replay':
        import pqh_replay
        return pqh_replay.main(['dry-run'])
    layout = layout_for(mode, code, command)
    problems = []

    def item(name, fn):
        try:
            value = fn()
            log(f'{name}: {value}')
            return value
        except Exception as exc:                    # noqa: BLE001 - reported, the next check goes on
            problems.append(f'{name}: {type(exc).__name__}: {exc}')
            log(f'{name}: PROBLEM {type(exc).__name__}: {exc}')
            return None
    log(f'PQH {command} --dry-run ({hc.utc()}): mode {mode}, unit {unit_name(command)}; no image, no truth, no GPU, nothing written')
    def code_item():
        record, found = hu.unit_code_identity(code) if mode == 'holdout' else hu.smoke_code_identity(code)
        if found:
            raise RuntimeError(f'{found}')
        return f"commit {record['commit']}, HEAD {record['head']} ({record['rule']})"
    item('code (hold-out: rule one, P1 from runs/holdout_B_v1/pinned_code.json; smoke: the checkout at its HEAD)', code_item)
    item('plan.md', lambda: io_.check_plan(code))
    item('preregistration', lambda: hu.prereg_problem(code) or 'pinned')
    if mode == 'holdout' and hu.prereg_problem(code):
        problems.append(f'preregistration: {hu.prereg_problem(code)}')
    item('settings_sha256', hp.settings_sha256)
    item('the sealed-list cache is empty (C23.6-5)', lambda: hu.hfw.sealed_cache_problem(command) or 'empty')
    if unit == 'out' or mode == 'holdout':
        item('finalization record (C22.H1-3)', lambda: f'{layout.finalized}: covers {len(hin.read_finalized(layout)[0]["pairs"])} pairs')
    def checkout():
        want = Path(hc.HOLDOUT_ROOT)/'code' if mode == 'holdout' else Path(hp.SMOKE_ROOT)/'code'     # I5: the smoke runs from its own checkout too
        if Path(code).resolve() != want.resolve():
            raise RuntimeError(f'{code} is not the detached checkout {want}')
        return 'yes'
    item('the detached checkout (C22.H0-4; the smoke: C22.H8)', checkout)
    lock = layout.members_lock if unit == 'members' else layout.out_lock
    log(f"lock {lock}: {'EXISTS' if Path(lock).exists() else 'absent'}")
    if unit != 'p3_replay' and Path(lock).exists():
        problems.append(f'{lock} exists')
    if unit == 'out':
        item('the members unit (Hm5, C22.H3-3)', lambda: f"{len(hin.members_view(layout, layout.subjects)[0])} subjects")
        if layout.ev_master_lock is not None:
            log(f"the evaluation's master lock {layout.ev_master_lock}: {'EXISTS (PQ refuses)' if Path(layout.ev_master_lock).exists() else 'absent'}")
    if mode == 'holdout':
        item("HI's hold-out inference (its lock)", lambda: f"{layout.hi_lock}: {'present' if Path(layout.hi_lock).is_file() else 'absent: HI has not run'}")
        if unit == 'members':
            import hi_core as hi
            item('admission: hi_core.load_unseal (BZ\'s unseal record, C23.3)', lambda: hi.load_unseal(Path(sealed.REPO)/hi.UNSEAL_REL, anchor=hp.ANCHOR).record())
            item('the decision of HI\'s development check', lambda: hi.check_decision(hi.DEVCHECK_DECISION)['reading'])
        else:
            fixed, moving = layout.pairs[0].split('__')
            item(f'admission of the first pair {layout.pairs[0]} (holdout_common.admit_holdout_pair)', lambda: hc.admit_holdout_pair(fixed, moving)[0]['unseal'])
    else:
        fixed, moving = hp.DEV_SMOKE_PAIR.split('__')
        item(f'admission of {hp.DEV_SMOKE_PAIR} (holdout_common.admit_dev_pair)', lambda: hc.admit_dev_pair(fixed, moving)[0]['rule'])
    if unit == 'members':
        import pq_members as members
        item('the four member files (pq_pins.MEMBERS, by path and sha256)', lambda: [str(p) for p in members.member_files(layout.models_root).values()])
    item('disks (prereg §5.3)', lambda: hc.disk_reading())
    log('start string (the lead, through launch): '+' '.join(start_argv(command, code=code, record_dir=units_dir(command, code)/'<unit>_<utc>')))
    log(f"estimate: {ESTIMATES['members' if unit == 'members' else ('p3_replay' if unit == 'p3_replay' else 'out')]}")
    log(f'problems: {problems or "none"}')
    return 9 if problems else 0


# ---- main ---------------------------------------------------------------------------------------------------------------------------------------
def run(command, *, resume=None, code=REPO):
    mode, unit = COMMANDS[command]
    layout = layout_for(mode, code, command) if unit != 'replay' else None
    if unit == 'members':
        return hu.members_unit(layout, resume_rel=resume)
    if unit == 'out':
        return hu.out_unit(layout, resume_rel=resume)
    if unit == 'replay':
        import pqh_replay
        return pqh_replay.main(['run'])
    import pqh_smoke
    return pqh_smoke.p3_replay(layout)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    for name in COMMANDS:
        p = sub.add_parser(name)
        if COMMANDS[name][0] == 'holdout':
            p.add_argument('--resume')
    d = sub.add_parser('dry-run')
    d.add_argument('unit', choices=list(COMMANDS))
    la = sub.add_parser('launch')
    la.add_argument('unit', choices=list(COMMANDS))
    la.add_argument('--resume')
    la.add_argument('--print', action='store_true')
    o = ap.parse_args(argv)
    if o.command == 'dry-run':
        return dry_run(o.unit)
    if o.command == 'launch':
        if o.resume is not None and COMMANDS[o.unit][0] != 'holdout':
            print('--resume: the dev mode never resumes (C22.H8)', file=sys.stderr)
            return 2
        if o.print:
            print(' '.join(start_argv(o.unit, record_dir=units_dir(o.unit)/'<unit>_<utc>', resume=o.resume)))
            return 0
        return launch(o.unit, resume=o.resume)
    return run(o.command, resume=getattr(o, 'resume', None))


if __name__ == '__main__':
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    sys.exit(main())
