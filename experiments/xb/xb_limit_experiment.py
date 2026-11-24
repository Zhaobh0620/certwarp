"""XB: the controlled limit experiment (prereg item 19, revision 1c item 2; plan runs/20261107-xb/limit_experiment_plan.md, a draft).  Only the tool: running it needs the lead's release.

    python -B experiments/xb/xb_limit_experiment.py plan --trial {mem_swap0,mem_swap256m,runtime} --arm ARM [--out DIR]     the unit's start string; launches nothing
    python -B experiments/xb/xb_limit_experiment.py run  --trial ... --arm ARM --out NEW_DIR                              one trial (one unit), then its reading

A trial is a unit built from the arm's real start string (xb_chain.unit_prefix with OOMPolicy continue and the environment as in a cell; xb_chain.supervisor, the same in-unit supervisor) with only
the limits lowered and the load replaced: xb_synthetic_load.py in the arm's interpreter (its conda environment's python for a Python arm, sci310's for a CLI arm) instead of the driver.
  mem_swap0    MemoryMax=512M, MemorySwapMax=0     load: touch 1024 MiB          expected (乙) infra_failure (oom); (甲) unit_limit (MemoryMax)
  mem_swap256m MemoryMax=512M, MemorySwapMax=256M  load: touch 1024 MiB          expected (乙) infra_failure (oom); (甲) unit_limit (MemoryMax)
  runtime      RuntimeMaxSec=20 (memory as a cell)  load: sleep 60 s               expected unit_limit (RuntimeMaxSec) under both rules
The cgroup's own limit stops the load before the machine notices (512 MiB of RAM, at most 256 MiB of swap: the lead's ruling, no trial writes gigabytes to swap); no subject file is read; the output goes
only into NEW_DIR (under /home/rmcfz/seq_vp_tag_results/xb_limit_v1/).  The reading is the chain's (xb_chain.unit_reading: systemd's Result as systemd-run printed it, the supervisor's end probe and
dmesg window, the journal) through holdout_common.classify_unit_end with holdout_common.LIMIT_RULE.  limit_result.json records the start string, systemd's Result, the supervisor's exit_code and
time stamps, the evidence, the reading, the expected reading under both rules, and passed.
"""
import argparse
import json
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import xb_chain  # noqa: E402

xc = xb_chain.xc
hc = xc.hc
def root():
    """/home/rmcfz/seq_vp_tag_results/xb_limit_v1, read at call time (the tests' sandbox moves the results root; in the sandbox a real root is refused: I7)."""
    return hc.guard_root(hc.RESULTS/'xb_limit_v1')


def lowered(**changes):
    props = dict(p.split('=', 1) for p in xc.UNIT_PROPERTIES)
    props.update(changes)
    return tuple(f'{k}={v}' for k, v in props.items())


TRIALS = dict(
    mem_swap0=dict(properties=lowered(MemoryMax='512M', MemorySwapMax='0'), load=['memory', '--mib', '1024', '--block-mib', '16'],
                   expected=dict(yi=('infra_failure', 'oom'), jia=('unit_limit', 'MemoryMax')), swap_free_min_gib=0.),
    mem_swap256m=dict(properties=lowered(MemoryMax='512M', MemorySwapMax='256M'), load=['memory', '--mib', '1024', '--block-mib', '16'],
                      expected=dict(yi=('infra_failure', 'oom'), jia=('unit_limit', 'MemoryMax')), swap_free_min_gib=1.),
    runtime=dict(properties=lowered(RuntimeMaxSec='20'), load=['runtime', '--seconds', '60'],
                 expected=dict(yi=('unit_limit', 'RuntimeMaxSec'), jia=('unit_limit', 'RuntimeMaxSec')), swap_free_min_gib=0.),
)


def unit_of(trial, arm):
    return f'tag-hb-xl-{trial.replace("_", "-")}-{arm}'


def load_python(arm):
    cfg = xc.ARMS[arm]
    return xc.ENVS/cfg['env']/'bin/python' if cfg.get('env') else xc.PY


def trial_argv(trial, arm, out):
    if trial not in TRIALS or arm not in xc.ARMS:
        raise ValueError(f'trial {trial!r} / arm {arm!r}')
    unit = unit_of(trial, arm)
    inner = [str(load_python(arm)), '-I', '-B', str(HERE/'xb_synthetic_load.py'), *TRIALS[trial]['load']]
    argv = xb_chain.unit_prefix(unit, out, properties=TRIALS[trial]['properties'])+xb_chain.supervisor(unit, Path(out)/'unit', inner)
    if 'pytest' in ' '.join(argv):
        raise ValueError('start string with a forbidden word')
    return argv


def out_problem(out):
    target = Path(out).resolve()
    if not target.is_relative_to(root().resolve()) or target == root().resolve():
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


class System(xb_chain.System):
    def trial_gate(self, trial):
        record = hc.machine_gate(own_units=())
        need = TRIALS[trial]['swap_free_min_gib']
        free = swap_free_gib()
        if free < need:
            raise hc.Refused('machine', f'SwapFree {free:.1f} GiB < {need} GiB for {trial}')
        return dict(record, swap_free_gib=round(free, 2))


def run_trial(trial, arm, out, *, system=None, rule=None):
    system = System() if system is None else system
    problem = out_problem(out)
    if problem:
        raise hc.Refused('output', problem)
    gate = system.trial_gate(trial)
    out = Path(out)
    out.mkdir(parents=True)
    unit = unit_of(trial, arm)
    argv = trial_argv(trial, arm, out)
    t0, since = system.uptime(), system.now()
    rc = system.run(argv, out/'systemd_run.txt')
    text = (out/'systemd_run.txt').read_text(errors='replace') if (out/'systemd_run.txt').exists() else ''
    result_line = xb_chain.unit_result(text)
    journal = system.journal(unit, since)
    (out/'journal.txt').write_text(journal or '')
    kernel_text = system.dmesg()
    oom = hc.parse_oom(kernel_text, t0)
    start = xb_chain.read_json(out/'unit'/'unit_start.json')
    end = xb_chain.read_json(out/'unit'/'unit_end.json')
    rule = hc.LIMIT_RULE if rule is None else rule
    status, kind, evidence = xb_chain.unit_reading(unit=unit, result_line=result_line, unit_end=end, oom_lines=oom, journal=journal, rule=rule)
    expected = TRIALS[trial]['expected'][rule]
    supervisor_ok = start is not None and end is not None and 'exit_code' in end and bool(end.get('utc')) and bool(start.get('utc'))
    document = dict(schema=xc.SCHEMA.format('limit_trial'), trial=trial, arm=arm, unit=unit, properties=list(TRIALS[trial]['properties']), load=TRIALS[trial]['load'], argv=argv, gate=gate,
                    systemd_run_exit=rc, systemd_result=result_line, systemd_run_output=str(out/'systemd_run.txt'),
                    supervisor=dict(start_utc=None if start is None else start.get('utc'), **({} if end is None else {k: end.get(k) for k in ('utc', 'uptime_start', 'uptime', 'seconds', 'exit_code', 'killed_by_signal', 'killed_by_supervisor', 'sigterm_received', 'memory_events', 'cgroup', 'dmesg_readable')})),
                    oom_lines=oom, reading=dict(status=status, kind=kind), evidence=evidence, rule=rule, expected=dict(status=expected[0], kind=expected[1]),
                    expected_both_rules={r: dict(status=v[0], kind=v[1]) for r, v in TRIALS[trial]['expected'].items()},
                    passed=(status, kind) == expected and supervisor_ok, supervisor_records_written=supervisor_ok, utc=hc.utc(),
                    rule_text='prereg item 19, revision 1c item 2: each lowered limit triggered once by a synthetic load in the arm\'s unit; the discriminator\'s reading must be the expected one and the '
                              'supervisor must have written exit_code and its time stamps')
    hc.write_json_once(out/'limit_result.json', document)
    return document


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    for name in ('plan', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--trial', required=True, choices=sorted(TRIALS))
        p.add_argument('--arm', required=True, choices=xc.ORDER)
        p.add_argument('--out', required=name == 'run', default=None)
    o = ap.parse_args(argv)
    out = o.out or str(root()/f'{o.trial}_{o.arm}_<utc>')
    if o.command == 'plan':
        print(shlex.join(trial_argv(o.trial, o.arm, out)))
        return 0
    try:
        doc = run_trial(o.trial, o.arm, out)
    except hc.Refused as exc:
        print(f'refused: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(dict(reading=doc['reading'], expected=doc['expected'], passed=doc['passed']), ensure_ascii=False))
    return 0 if doc['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
