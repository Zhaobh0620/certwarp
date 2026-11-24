"""The controlled limit experiment, second batch (prereg item 19 / revision 2 item 16, revision 1c item 2; runs/20261107-xb/limit_experiment_plan.md revision 1 item 5; B4 review
docs/bayesreg/reviews/开发集核对计划_复核_20261110.md §5 I5; decision_log 2026-11-10 ~3:16 AM CST; A5a).  Only the tool: running it needs the lead's release and a pinned plan.

    python -B experiments/xb/xb_limit_batch2.py plan --unit {configc,review,pq_members,pq_unit} --trial {runtime,mem_swap0,mem_swap256m} [--out DIR]     the start string; launches nothing
    python -B experiments/xb/xb_limit_batch2.py run  --unit ... --trial ... --out NEW_DIR                                                              one trial (one unit), its reading, its check

Why a file of its own: the first batch's tool, xb_limit_experiment.py (XB's two arms; run at a99724d, plan revision 1 item 6), stays byte for byte as it ran -- a change of it would make the first
batch rerun at P0.  This file imports nothing of it.

The units (the lead: "configuration C as (乙); the three D123 (iii) units as D123 (b)"):
  configc      the configuration C arm's unit: hb_core.unit_argv('configc', ...) with hb_core.CONFIGC_UNIT (12G, CPUQuota 1200 %, NUMBA 8, OOMPolicy continue, RuntimeMaxSec 1800, the GPU) and the trial's
               limits; its swap is 256M in every trial where the arm's is 4G (the lead: 交换一律 256M; mem_swap0 keeps 0).  Expected (乙), as XB's arms: memory -> infra_failure (oom), runtime ->
               unit_limit (RuntimeMaxSec).
  review       the step-5 review certificate unit: hb_core.unit_argv('review', ...) with he_core.EVALUATION_UNIT (8G, swap 4G -> 256M, CPUQuota 800 %, NUMBA 8, nice 5, no GPU) and the trial's limits.
               Expected D123 (b): the same reading (OOM evidence -> infra_failure, a timeout -> unit_limit), the evidence readable (the end probe, systemd's Result, the journal), and the queue's STOP: the
               review queue's tail (experiments/hb/hb_review.py queue: no review.json, no new review_not_finalized) writes review_unit_stopped.json (hb_review.unit_stop_record, here into the trial's
               directory) and stops with hb_review.unit_end_stop_code (20 for infra_failure or unit_limit) -- not a method failure.
  pq_members   PQ's member inference unit and PQ's unit (D123 (iii); P0S LB2, the lead's brief of 11-14): the start string is PQH's launcher's own, experiments/pq/run_pqh.start_argv of the
  pq_unit      hold-out command `members` / `out` (pqh_pins.UNIT_SPECS: 12G / 11G, swap 4G, OOMPolicy continue, RuntimeMaxSec 4800 / 7200, nice 5; the in-unit supervisor xb_unit.py; the unit
               names pqh_pins.UNITS, tag-hb-pq-members and tag-hb-pq-out -- the launcher's own names, as the HB units keep their queues' naming; they hold no pair), with exactly MemoryMax,
               MemorySwapMax and RuntimeMaxSec set to the trial's (swap: as for the others) and the inner command `run_pqh.py <command>` replaced by the load in the same python (pq_trial_argv; a test
               compares the rest of the string).  The run is the launcher's own code after the start string (run_pqh.run_unit, moved out of run_pqh.launch unchanged: systemd-run --wait,
               launch.log, the supervisor's records, the reading with holdout_common.classify_unit_end, <trial>/unit/unit_reading.json once, the exit FAILURE_EXIT).  Expected D123 (b): the same
               readings as the review unit's, and the launcher's STOP -- unit_reading.json with that status and the exit run_pqh.FAILURE_EXIT (30) (PQ_STOP).  Neither start string sets
               CUDA_VISIBLE_DEVICES (the members unit is a GPU unit); the load does not touch the GPU.
The trials (each unit's real start string, only the limits changed and the inner command replaced by xb_synthetic_load.py in sci310's python; no subject file, no frozen run, no GPU work):
  runtime        RuntimeMaxSec=20 (memory as the unit's; swap 256M)          load: sleep 60 s
  mem_swap0      MemoryMax=512M, MemorySwapMax=0                            load: touch 1024 MiB
  mem_swap256m   MemoryMax=512M, MemorySwapMax=256M                         load: touch 1024 MiB
The unit name of configc and review is the queues' own (hb_core.unit_name) of a reserved synthetic pair per trial (TRIAL_PAIRS: OASIS_OAS1_0000_MR1__OASIS_OAS1_000<k>_MR1, no such subject):
tag-hb-c-0000-000<k>, tag-hb-review-0000-000<k>.  PQ's two keep their one name in every trial (above): what keeps a trial's kernel window clear of the previous trial's lines is the machine gate's
>= 10 s since the previous unit ended (dmesg runs about 4 s ahead of /proc/uptime: about 6 s to spare, A5a review m1).  The run of configc and review is hb_core.run_unit itself -- the queues' code:
the start string (he_core.unit_argv_problems checks it carries the trial's values), systemd-run --wait, the Result line, the supervisor's unit_end.json, the dmesg window, the journal,
holdout_common.classify_unit_end (rule (乙)) -- into the trial's directory (batch = the trial directory; units/, unit/, unit.log); PQ's is run_pqh.run_unit (above) into <trial>/unit/.
After the unit (plan revision 1 items 2, 4; the result review's I1, I2): the clock (dmesg's stamps against /proc/uptime: the newest dmesg line's stamp and /proc/uptime read together, and for a memory
trial the OOM line's stamp against the supervisor's uptime at its end; recorded, never used to judge); the additional conditions of plan revision 1 item 2, checked mechanically and written to
check.json; for review, the queue's STOP (stop_record and code), for PQ's two the launcher's (unit_reading.json and its exit); then limit_result.json.  passed: the reading is the expected one,
the supervisor wrote its records, every additional condition holds and, for the three D123 units, the STOP is the expected one.  The gate (holdout_common.machine_gate: no other tag-* / seq-vp-* unit, the GPU idle, MemAvailable, the load, and >= 10 s since the previous unit
ended -- decision_log 2026-11-10 ~3:16 AM CST) runs once before the trial and refuses at once (exit 2), as the first batch's; mem_swap256m and the swap-256M runtime also need SwapFree >= 1 GiB.
The output goes only into NEW_DIR, a new directory under <results>/xb_limit_v1/ (the first batch's root).
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import sys
import time

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
for _p in (EXPERIMENTS/'hb', EXPERIMENTS/'pq', HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import hb_core as hb  # noqa: E402
import pqh_pins as hp  # noqa: E402

hc, he = hb.hc, hb.he
SCHEMA = 'bayesreg.xb.limit_batch2.{}.v1'
SWAP_CAP = '256M'                                   # the lead: 交换一律 256M (the units' 4G never written in a trial)
TRIALS = dict(
    runtime=dict(changes=dict(runtime=20, memory_swap_max=SWAP_CAP), load=['runtime', '--seconds', '60'], expected=('unit_limit', 'RuntimeMaxSec'), swap_free_min_gib=1.),
    mem_swap0=dict(changes=dict(memory_max='512M', memory_swap_max='0'), load=['memory', '--mib', '1024', '--block-mib', '16'], expected=('infra_failure', 'oom'), swap_free_min_gib=0.),
    mem_swap256m=dict(changes=dict(memory_max='512M', memory_swap_max=SWAP_CAP), load=['memory', '--mib', '1024', '--block-mib', '16'], expected=('infra_failure', 'oom'), swap_free_min_gib=1.),
)
TRIAL_PAIRS = dict(runtime='OASIS_OAS1_0000_MR1__OASIS_OAS1_0001_MR1', mem_swap0='OASIS_OAS1_0000_MR1__OASIS_OAS1_0002_MR1', mem_swap256m='OASIS_OAS1_0000_MR1__OASIS_OAS1_0003_MR1')
UNITS = dict(
    configc=dict(kind='configc', res=hb.CONFIGC_UNIT, gpu=True, rule='(乙)', d123=False),
    review=dict(kind='review', res=he.EVALUATION_UNIT, gpu=False, rule='D123 (b)', d123=True),
    pq_members=dict(kind='pq', command='members', res=hp.UNIT_SPECS['members'], rule='D123 (b)', d123=True),
    pq_unit=dict(kind='pq', command='out', res=hp.UNIT_SPECS['out'], rule='D123 (b)', d123=True),
)
D123_STOP = dict(infra_failure=hb.EXIT['infra'], unit_limit=hb.EXIT['infra'])      # hb_review.unit_end_stop_code's values for the expected readings (a test compares)
PQ_STOP = dict(infra_failure=30, unit_limit=30)                                    # run_pqh.FAILURE_EXIT for the expected readings: the launcher's STOP (a test compares)
PQ_CHANGES = dict(runtime='runtime_max_sec', memory_max='memory_max', memory_swap_max='memory_swap_max')     # TRIALS' keys -> pqh_pins.UNIT_SPECS' keys
PQ_PROPERTIES = dict(memory_max='MemoryMax', memory_swap_max='MemorySwapMax', runtime_max_sec='RuntimeMaxSec')      # the three properties a trial sets in PQ's start string
KERNEL_TIME = re.compile(r'^\[\s*([0-9.]+)\]')


def root():
    """<results>/xb_limit_v1, read at call time (the tests' sandbox moves the results root; a real root refused there: holdout_common I7)."""
    return hc.guard_root(Path(hc.RESULTS)/'xb_limit_v1')


def trial_res(unit, trial):
    """The unit's settings with the trial's limits (and the swap cap); PQ's in pqh_pins.UNIT_SPECS' keys."""
    spec = UNITS[unit]
    if trial not in TRIALS:
        raise ValueError(f'trial {trial!r}: one of {sorted(TRIALS)}')
    changes = TRIALS[trial]['changes']
    if spec['kind'] == 'pq':
        return dict(spec['res'], **{PQ_CHANGES[k]: v for k, v in changes.items()})
    return dict(spec['res'], **changes)


def unit_name_of(unit, trial):
    """The unit's name: the HB queues' own for the trial's reserved pair; PQ's launcher's own (pqh_pins.UNITS of the hold-out command)."""
    spec = UNITS[unit]
    if spec['kind'] == 'pq':
        return hp.UNITS[spec['command']]
    return hb.unit_name(spec['kind'], TRIAL_PAIRS[trial])


def inner(trial):
    """The inner command: the synthetic load in sci310's python (hb_core.unit_argv puts `python -B` before it)."""
    return [HERE/'xb_synthetic_load.py', *TRIALS[trial]['load']]


def trial_argv(unit, trial, out):
    """The start string (hb_core.unit_argv, the queues' own; PQ's: pq_trial_argv; he_core.unit_argv_problems checks either)."""
    spec = UNITS[unit]
    out = Path(out)
    if spec['kind'] == 'pq':
        argv = pq_trial_argv(unit, trial, out)
    else:
        argv = hb.unit_argv(spec['kind'], TRIAL_PAIRS[trial], inner(trial), code=hb.REPO, unit_dir=out/'unit', res=trial_res(unit, trial), gpu=spec['gpu'], log=out/'unit.log')
    if 'pytest' in ' '.join(map(str, argv)):
        raise ValueError('start string with a forbidden word')
    return argv


def pq_trial_argv(unit, trial, out):
    """PQ's unit: run_pqh.start_argv of the hold-out command (the launcher's own string, its record dir <out>/unit), with exactly MemoryMax, MemorySwapMax and RuntimeMaxSec set to the trial's
    and the inner command (`<python> -B <code>/experiments/pq/run_pqh.py <command>`) replaced by the synthetic load in the same python; nothing else changes.  ValueError when the launcher's
    string is not of that shape (a property not there once, another inner command) or the result does not carry the trial's values once (he_core.unit_argv_problems)."""
    import run_pqh
    spec = UNITS[unit]
    res = trial_res(unit, trial)
    argv = [str(a) for a in run_pqh.start_argv(spec['command'], code=hb.REPO, record_dir=Path(out)/'unit')]
    for key, prop in PQ_PROPERTIES.items():
        at = [i for i, a in enumerate(argv) if a.startswith(f'--property={prop}=')]
        if len(at) != 1:
            raise ValueError(f"{unit}: --property={prop}= is in the launcher's start string {len(at)} times, not once")
        argv[at[0]] = f'--property={prop}={res[key]}'
    i = argv.index('--')
    launcher_inner = [str(run_pqh.PY), '-B', str(Path(hb.REPO)/'experiments/pq/run_pqh.py'), spec['command']]
    if argv[i+1:] != launcher_inner:
        raise ValueError(f"{unit}: the launcher's inner command {argv[i+1:]} is not {launcher_inner}")
    argv = argv[:i+1]+[str(run_pqh.PY), '-B', *[str(a) for a in inner(trial)]]
    problems = he.unit_argv_problems(argv, dict(memory_max=res['memory_max'], memory_swap_max=res['memory_swap_max'], oom_policy=res['oom_policy'], runtime=res['runtime_max_sec'], nice=res['nice']))
    if problems:
        raise ValueError(f"{unit}: the start string is not the trial's {res}: {problems}")
    return argv


def out_problem(out):
    target = Path(out).resolve()
    base = root().resolve()
    if not target.is_relative_to(base) or target == base:
        return f'{out}: a trial writes into a new directory under {root()}'
    if target.exists():
        return f'{out} exists: a trial writes into a new directory'
    return None


def swap_free_gib(meminfo='/proc/meminfo'):
    try:
        for line in Path(meminfo).read_text().splitlines():
            if line.startswith('SwapFree:'):
                return int(line.split()[1])/1024**2
    except (OSError, ValueError, IndexError):
        pass
    return 0.


class System(hb.System):
    """hb_core.System (the queues' run, dmesg, journal, uptime) and the trial's gate."""

    def trial_gate(self, trial):
        record = hc.machine_gate(own_units=())
        need = TRIALS[trial]['swap_free_min_gib']
        free = swap_free_gib()
        if free < need:
            raise hc.Refused('machine', f'SwapFree {free:.1f} GiB < {need} GiB for {trial}')
        return dict(record, swap_free_gib=round(free, 2))


def dmesg_last_stamp(text):
    """The stamp of the newest line of dmesg's text (seconds), or None."""
    stamps = [float(m.group(1)) for line in (text or '').splitlines() if (m := KERNEL_TIME.match(line))]
    return max(stamps) if stamps else None


def clock_reading(system, end, oom_lines):
    """Plan revision 1 item 2 (the result review's I2): dmesg's stamps against /proc/uptime -- the newest dmesg line's stamp and /proc/uptime read one after the other (their difference is the offset
    minus that line's age: a lower bound when the line is old), and, for a memory trial, the OOM line's stamp minus the supervisor's uptime at its end (the kill comes just before that end: about the
    offset).  Recorded, never used to judge."""
    text = system.dmesg()
    now = system.uptime()
    last = dmesg_last_stamp(text)
    oom_t = [r.get('t') for r in oom_lines or [] if isinstance(r.get('t'), (int, float))]
    sup = (end or {}).get('uptime')
    return dict(uptime_now=now, dmesg_newest_stamp=last, newest_minus_uptime=None if last is None else round(last-now, 3), oom_stamps=oom_t, supervisor_uptime_end=sup,
                oom_minus_supervisor_end=[round(t-sup, 3) for t in oom_t] if isinstance(sup, (int, float)) else None,
                note='dmesg stamps run about 3.6-4.1 s ahead of /proc/uptime here (decision_log 2026-11-10 ~3:16 AM CST); the window of the reading starts at /proc/uptime (not corrected)')


def additional_check(trial, reading, end):
    """Plan revision 1 item 2's additional conditions, mechanically (the first batch checked them by hand, the result review's I1): the evidence readable and, for a memory trial, an OOM line of this
    unit's own limit; for runtime, systemd's timeout and no OOM line in the window.  {items: {name: bool}, passed}."""
    evidence = (reading or {}).get('evidence') or {}
    events = (end or {}).get('memory_events')
    items = dict(dmesg_readable=(end or {}).get('dmesg_readable') is True, memory_events_read=isinstance(events, dict))
    if trial == 'runtime':
        lines = evidence.get('machine_oom_lines')
        items.update(systemd_result_timeout=(reading or {}).get('systemd_result') == 'timeout',
                     no_oom_evidence=evidence.get('oom_evidence') == [] and lines == [],
                     no_oom_line_in_window=(end or {}).get('oom_lines') == [])        # A5a review m9: the plan's oom_lines == [] -- every OOM line of the window, any unit's
    else:
        items.update(oom_kill_counted=isinstance(events, dict) and (events.get('oom_kill') or 0) >= 1,
                     own_limit_dmesg_line=any(i.get('source') == 'dmesg' and i.get('own_limit') is True for i in evidence.get('oom_evidence') or []),
                     no_machine_oom_line=evidence.get('machine_oom_lines') == [])
    return dict(items=items, passed=all(items.values()), rule='plan revision 1 item 2 (and the result review I1: written to a file after every trial)')


def run_pq_unit(unit, trial, out, system):
    """PQ's unit through the launcher's own code after its start string (run_pqh.run_unit: systemd-run --wait, launch.log, unit_reading.json once, FAILURE_EXIT), the record directory
    <out>/unit made first, as run_pqh.launch makes its own; then this tool's note of the unit's end (holdout_common.note_unit_end, as hb_core.run_unit does).  The launcher's messages go to
    <out>/launcher.log.  Returns (reading, the launcher's exit code): the launcher's record, with status_kind (its `kind`) and systemd_result (run_pqh.unit_result of launch.log) beside it."""
    import run_pqh
    spec = UNITS[unit]
    name = unit_name_of(unit, trial)
    record_dir = Path(out)/'unit'
    record_dir.mkdir(exist_ok=False)
    argv = pq_trial_argv(unit, trial, out)

    def log(message):
        with (Path(out)/'launcher.log').open('a', encoding='utf-8') as stream:
            stream.write(f'{hc.utc()} {message}\n')
    code = run_pqh.run_unit(spec['command'], name, argv, record_dir, system=system, log=log)
    hc.note_unit_end(name)                          # A5a review I1: this tool's own record of its last unit (holdout_common.machine_gate's local lower bound), as hb_core.run_unit
    document = hb.read_json(record_dir/'unit_reading.json') or {}
    launch_log = record_dir/'launch.log'
    text = launch_log.read_text(encoding='utf-8', errors='replace') if launch_log.is_file() else ''
    return dict(document, status_kind=document.get('kind'), systemd_result=run_pqh.unit_result(text), unit_reading=str(record_dir/'unit_reading.json')), code


def run_trial(unit, trial, out, *, system=None):
    spec = UNITS[unit]
    system = System() if system is None else system
    problem = out_problem(out)
    if problem:
        raise hc.Refused('output', problem)
    out = Path(out)
    pq = spec['kind'] == 'pq'
    pair = None if pq else TRIAL_PAIRS[trial]
    name = unit_name_of(unit, trial)
    planned_argv = [str(a) for a in trial_argv(unit, trial, out)]     # P0S revision 2 (the code review's minor 11): a start string of the wrong shape raises before the gate and the directory
    gate = system.trial_gate(trial)
    out.mkdir(parents=True)
    started = hc.utc()
    launcher_code = None
    if pq:
        reading, launcher_code = run_pq_unit(unit, trial, out, system)
    else:
        reading = hb.run_unit(spec['kind'], pair, inner(trial), system, batch=out, unit_dir=out/'unit', res=trial_res(unit, trial), gpu=spec['gpu'], code=hb.REPO)
    start = hb.read_json(out/'unit'/'unit_start.json')
    end = hb.read_json(out/'unit'/'unit_end.json')
    own = [r for r in (end or {}).get('oom_lines') or [] if hc.in_unit(r.get('task_memcg'), name, (end or {}).get('cgroup'))]
    clock = clock_reading(system, end, own)          # the supervisor's window (parse_oom's rows carry the kernel's stamp t), this unit's lines only
    check = additional_check(trial, reading, end)
    hc.write_json_once(out/'check.json', dict(schema=SCHEMA.format('check'), unit=unit, trial=trial, **check, utc=hc.utc()))
    expected = TRIALS[trial]['expected']
    got = (reading.get('status'), reading.get('status_kind'))
    supervisor_ok = start is not None and end is not None and 'exit_code' in end and bool(end.get('utc')) and bool(start.get('utc'))
    stop = None
    if spec['d123'] and pq:                         # D123 (b): the launcher's STOP -- its unit_reading.json with the status, and its exit FAILURE_EXIT (run_pqh.launch: the unit stops, to the lead)
        record = Path(reading['unit_reading'])
        stop = dict(code=launcher_code, expected_code=PQ_STOP.get(expected[0]), stop_record=str(record), stop_record_sha256=hb.sha(record) if record.is_file() else None,
                    stop_record_status=reading.get('status'),
                    rule='run_pqh.launch (its part after the start string, run_pqh.run_unit): a unit end that is a limit, an infrastructure failure or needs the lead -> unit_reading.json with '
                         'that status and the exit FAILURE_EXIT (30), STOP to the lead, nothing rerun; D123 (b)')
    elif spec['d123']:                              # D123 (b): the review queue's tail on this reading (no review.json, no review_not_finalized: the synthetic load writes neither)
        import hb_review
        record = hb_review.unit_stop_record(pair, reading, root=out/'review_home')
        stop = dict(code=hb_review.unit_end_stop_code(reading), expected_code=D123_STOP.get(expected[0]), stop_record=str(record), stop_record_sha256=hb.sha(record),
                    rule='hb_review.queue: a review unit that left neither review.json nor a new review_not_finalized -> review_unit_stopped (unit_stop_record) and STOP (unit_end_stop_code); D123 (b)')
    passed = got == expected and supervisor_ok and check['passed'] and (stop is None or stop['code'] == stop['expected_code'])
    document = dict(schema=SCHEMA.format('trial'), unit=unit, kind=spec['kind'], command=spec.get('command'), trial=trial, rule=spec['rule'], pair_of_the_unit_name=pair, unit_name=name,
                    res=trial_res(unit, trial), load=TRIALS[trial]['load'], argv=[str(a) for a in reading.get('argv') or []], argv_is_the_planned=[str(a) for a in reading.get('argv') or []] == planned_argv, gate=gate, started_utc=started, reading=reading,
                    expected=dict(status=expected[0], kind=expected[1]), supervisor_records_written=supervisor_ok, check=check, clock=clock, queue_stop=stop, passed=passed, utc=hc.utc(),
                    rule_text='prereg item 19 / revision 2 item 16, revision 1c item 2, plan revision 1 item 5: each lowered limit triggered once by a synthetic load in the unit\'s real start string; the '
                              'reading of the queues\' own code must be the expected one, the evidence readable (plan revision 1 item 2), and for the D123 units the queue\'s STOP (D123 (b); '
                              'for PQ\'s two units the launcher\'s, P0S LB2)')
    hc.write_json_once(out/'limit_result.json', document)
    return document


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    for name in ('plan', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--unit', required=True, choices=sorted(UNITS))
        p.add_argument('--trial', required=True, choices=sorted(TRIALS))
        p.add_argument('--out', required=name == 'run', default=None)
    o = ap.parse_args(argv)
    out = o.out or str(root()/f'{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}_{o.unit}_{o.trial}')
    if o.command == 'plan':
        print(shlex.join(str(a) for a in trial_argv(o.unit, o.trial, out)))
        return 0
    try:
        doc = run_trial(o.unit, o.trial, out)
    except hc.Refused as exc:
        print(f'refused: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(dict(reading=dict(status=doc['reading'].get('status'), kind=doc['reading'].get('status_kind')), expected=doc['expected'], check=doc['check']['passed'],
                          queue_stop=doc['queue_stop'], passed=doc['passed']), ensure_ascii=False))
    return 0 if doc['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
