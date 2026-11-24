"""A5a: the controlled limit experiment's second batch (experiments/xb/xb_limit_batch2.py; runs/20261107-xb/limit_experiment_plan.md revision 1 items 2, 5, 6; B4 review §5 I5; decision_log 2026-11-10
~3:16 AM CST): the configuration C arm's and the review unit's real start strings with only the limits lowered (swap 256M at most), the queues' own reading (hb_core.run_unit), the additional
conditions written to check.json, the clock recorded, the review queue's STOP (D123 (b)); PQ's two units pending PQH.  And plan revision 1 item 6: what the first batch ran is unchanged.
No unit is started: the system is a stand-in (systemd-run is the PATH shim otherwise).  P0S LB2 (11-14): PQ's two units take PQH's launcher's start string (run_pqh.start_argv) and its own
reading code after it (run_pqh.run_unit), the STOP being the launcher's."""
import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from . import _world as W

for _p in (W.ROOT/'experiments'/'hb', W.ROOT/'experiments'/'pq'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import xb_limit_batch2 as xl2  # noqa: E402
import hb_core as hb  # noqa: E402
import hb_review  # noqa: E402
import pqh_pins as hp  # noqa: E402
import run_pqh  # noqa: E402

hc = W.hc
UNIT = 'tag-hb-c-0000-0002'


def props(argv):
    return {a.split('=', 2)[1]: a.split('=', 2)[2] for a in map(str, argv) if a.startswith('--property=')}


@pytest.mark.parametrize('unit', ['configc', 'review'])
def test_the_trials_keep_the_units_start_string_and_lower_only_the_limits(unit):
    spec = xl2.UNITS[unit]
    out = xl2.root()/'t'
    real = hb.unit_argv(spec['kind'], hc.DEV_PAIRS[0], [W.ROOT/'experiments/hb/hb_configc.py', 'run'], code=hb.REPO, unit_dir=out/'u', res=dict(spec['res']), gpu=spec['gpu'], log=out/'l.log')
    for trial, t in xl2.TRIALS.items():
        argv = [str(a) for a in xl2.trial_argv(unit, trial, out)]
        p, q = props(argv), props(real)
        changed = {k for k in set(p) | set(q) if p.get(k) != q.get(k)}
        assert changed <= {'MemoryMax', 'MemorySwapMax', 'RuntimeMaxSec', 'StandardOutput', 'StandardError'}, changed
        assert p['MemorySwapMax'] in ('0', '256M') and p['OOMPolicy'] == 'continue'               # the lead: no trial may write more than 256M of swap
        assert f'--unit=tag-hb-{"c" if unit == "configc" else "review"}-{hc.short_pair(xl2.TRIAL_PAIRS[trial])}' in argv
        envs = [a for a in argv if a.startswith('--setenv=')]
        assert envs == [a for a in map(str, real) if a.startswith('--setenv=')]                   # the environment as the unit's: PYTHONPATH <frozen>:<code>, the threads, the GPU or not
        i = argv.index('--')
        assert argv[i-5].endswith('experiments/xb/xb_unit.py') and argv[i+1] == str(hb.PY) and argv[i+3].endswith('xb_synthetic_load.py') and argv[i+4:] == t['load']
        assert not any('pair.npz' in a or 'hb_configc.py' in a or 'hb_review.py' in a for a in argv)
        assert not hb.he.unit_argv_problems(argv, xl2.trial_res(unit, trial))
    assert props(xl2.trial_argv(unit, 'mem_swap0', out))['MemoryMax'] == '512M' and props(xl2.trial_argv(unit, 'runtime', out))['RuntimeMaxSec'] == '20'
    assert props(xl2.trial_argv(unit, 'runtime', out))['MemoryMax'] == spec['res']['memory_max']


PQ_UNITS = dict(pq_members=('members', 'tag-hb-pq-members', '12G', '4800'), pq_unit=('out', 'tag-hb-pq-out', '11G', '7200'))


@pytest.mark.parametrize('unit', sorted(PQ_UNITS))
def test_pqs_two_units_keep_the_launchers_start_string_and_lower_only_the_limits(unit):
    """P0S LB2 (the lead's brief of 11-14): PQ's member inference unit and PQ's unit take PQH's launcher's own start string (run_pqh.start_argv of the hold-out command), with only MemoryMax,
    MemorySwapMax and RuntimeMaxSec set to the trial's and the inner command `run_pqh.py <command>` replaced by the synthetic load in the same python; the unit name is the launcher's own."""
    command, name, memory, runtime = PQ_UNITS[unit]
    out = xl2.root()/'t'
    real = [str(a) for a in run_pqh.start_argv(command, code=hb.REPO, record_dir=out/'unit')]
    i_real = real.index('--')
    assert real[i_real+1:] == [run_pqh.PY, '-B', str(hb.REPO/'experiments/pq/run_pqh.py'), command]
    assert props(real)['MemoryMax'] == memory and props(real)['RuntimeMaxSec'] == runtime and props(real)['MemorySwapMax'] == '4G'
    for trial, t in xl2.TRIALS.items():
        argv = [str(a) for a in xl2.trial_argv(unit, trial, out)]
        i = argv.index('--')
        assert i == i_real and len(argv[:i]) == len(real[:i_real])
        changed = [(a, b) for a, b in zip(argv[:i], real[:i_real]) if a != b]
        assert {a.split('=', 2)[1] for a, _ in changed} <= {'MemoryMax', 'MemorySwapMax', 'RuntimeMaxSec'}, changed
        assert all(a.startswith('--property=') and b.startswith(a.rsplit('=', 1)[0]+'=') for a, b in changed)      # the same property, another value
        p = props(argv)
        assert p['MemorySwapMax'] in ('0', '256M') and p['OOMPolicy'] == 'continue' and '--nice=5' in argv       # the lead: no trial may write more than 256M of swap
        assert f'--unit={name}' in argv and argv[i-2:i] == ['--unit', name] and argv[i-4] == '--record-dir' and argv[i-3] == str(out/'unit')
        assert argv[i-5].endswith('experiments/xb/xb_unit.py') and argv[i+1:] == [run_pqh.PY, '-B', str(xl2.HERE/'xb_synthetic_load.py'), *t['load']]
        assert not any('run_pqh.py' in a or 'pair.npz' in a for a in argv)
        res = xl2.trial_res(unit, trial)
        assert not hb.he.unit_argv_problems(argv, dict(memory_max=res['memory_max'], memory_swap_max=res['memory_swap_max'], oom_policy='continue', runtime=res['runtime_max_sec'], nice=5))
    assert props(xl2.trial_argv(unit, 'mem_swap0', out))['MemoryMax'] == '512M' and props(xl2.trial_argv(unit, 'mem_swap0', out))['MemorySwapMax'] == '0'
    assert props(xl2.trial_argv(unit, 'mem_swap256m', out))['MemorySwapMax'] == '256M'
    rt = props(xl2.trial_argv(unit, 'runtime', out))
    assert rt['RuntimeMaxSec'] == '20' and rt['MemoryMax'] == memory and rt['MemorySwapMax'] == '256M'
    assert xl2.unit_name_of(unit, 'runtime') == name == hp.UNITS[command]


def test_pq_start_strings_of_another_shape_are_refused(monkeypatch):
    """pq_trial_argv refuses a launcher string that is not the one it was written for (a property twice or missing, another inner command): ValueError, nothing started."""
    good = run_pqh.start_argv

    def twice(command, **kwargs):
        argv = good(command, **kwargs)
        return argv[:5]+['--property=MemoryMax=1G']+argv[5:]

    def missing(command, **kwargs):
        return [a for a in good(command, **kwargs) if not a.startswith('--property=RuntimeMaxSec=')]

    def other_inner(command, **kwargs):
        return good(command, **kwargs)+['--resume', 'x']
    for fake, words in ((twice, 'MemoryMax= is in'), (missing, 'RuntimeMaxSec= is in'), (other_inner, 'inner command')):
        monkeypatch.setattr(run_pqh, 'start_argv', fake)
        with pytest.raises(ValueError, match=words):
            xl2.trial_argv('pq_members', 'runtime', xl2.root()/'t')
    monkeypatch.setattr(run_pqh, 'start_argv', good)
    assert xl2.trial_argv('pq_members', 'runtime', xl2.root()/'t')


@pytest.mark.parametrize('unit', ['configc', 'review', 'pq_members', 'pq_unit'])
def test_a_start_string_of_the_wrong_shape_leaves_no_trial_directory(unit, monkeypatch):
    """P0S revision 2 (the code review's minor 11): the start string is built before the gate and the trial directory; when it cannot be built (PQ: the launcher's string of another shape;
    HB: the queue's unit_argv refusing), run_trial raises and leaves nothing behind, the gate not asked."""
    if unit.startswith('pq_'):
        good = run_pqh.start_argv
        monkeypatch.setattr(run_pqh, 'start_argv', lambda command, **kwargs: good(command, **kwargs)+['--resume', 'x'])
    else:
        monkeypatch.setattr(hb, 'unit_argv', lambda *a, **k: (_ for _ in ()).throw(ValueError('the start string is not the unit (test)')))
    out = xl2.root()/f'shape_{unit}'

    class NoGate(TrialSystem):
        def trial_gate(self, trial):
            raise AssertionError('the gate was asked before the start string was built')
    with pytest.raises(ValueError):
        xl2.run_trial(unit, 'runtime', out, system=NoGate(end={}, result='timeout', dmesg=''))
    assert not out.exists()


def test_the_plan_command_prints_pqs_start_strings(capsys):
    for unit, (command, name, _, _) in PQ_UNITS.items():
        assert xl2.main(['plan', '--unit', unit, '--trial', 'runtime']) == 0
        text = capsys.readouterr().out
        assert f'--unit={name}' in text and 'xb_synthetic_load.py runtime --seconds 60' in text and 'run_pqh.py' not in text


def test_the_output_and_the_expected_stop_codes():
    assert xl2.out_problem(W.ROOT/'x') and xl2.out_problem(xl2.root()) and xl2.out_problem(xl2.root()/'new') is None
    assert {k: hb_review.unit_end_stop_code(dict(status=k)) for k in xl2.D123_STOP} == xl2.D123_STOP == dict(infra_failure=20, unit_limit=20)
    assert hb_review.unit_end_stop_code(dict(status='needs_lead_rejudgment')) == 25 and hb_review.unit_end_stop_code(None) == 25
    assert xl2.PQ_STOP == {k: run_pqh.FAILURE_EXIT for k in ('infra_failure', 'unit_limit')} and run_pqh.FAILURE_EXIT == 30       # the launcher's STOP for the expected readings
    assert {u for u, s in xl2.UNITS.items() if s['d123']} == {'review', 'pq_members', 'pq_unit'} and not any('pending' in s for s in xl2.UNITS.values())


CGROUP = f'/user.slice/user-1000.slice/user@1000.service/app.slice/{UNIT}.service'


def oom_line(t, *, oom_memcg=CGROUP, constraint='CONSTRAINT_MEMCG', task_memcg=CGROUP):
    return f'[{t:12.6f}] oom-kill:constraint={constraint},nodemask=(null),cpuset=/,mems_allowed=0,oom_memcg={oom_memcg},task_memcg={task_memcg},task=python,pid=7,uid=1000\n'


class TrialSystem(xl2.System):
    """A stand-in machine: run() plays the supervisor (unit_start.json, unit_end.json) and systemd-run's result line; dmesg, journal and uptime are given."""

    def __init__(self, *, end, result, dmesg, uptime=100.):
        self.end, self.result, self.kernel, self.up, self.runs = end, result, dmesg, uptime, []

    def trial_gate(self, trial):
        return dict(passed=True, note='test')

    def run(self, argv, log_path):
        self.runs.append([str(a) for a in argv])
        unit_dir = Path(argv[argv.index('--record-dir')+1])
        pq = any(str(a).startswith('--unit=tag-hb-pq-') for a in argv)
        assert unit_dir.is_dir() == pq                  # PQ's launcher makes its record directory before the unit (run_pqh.launch); hb_core.run_unit leaves it to the supervisor
        unit_dir.mkdir(parents=True, exist_ok=pq)
        (unit_dir/'unit_start.json').write_text(json.dumps(dict(utc='2026-11-10T17:00:00Z', uptime=self.up)))
        (unit_dir/'unit_end.json').write_text(json.dumps(self.end))
        Path(log_path).write_text(f'Running as unit: x.service\nFinished with result: {self.result}\n')
        return 0

    def uptime(self):
        return self.up

    def dmesg(self):
        return self.kernel

    def journal(self, unit, since):
        return ''

    def now(self):
        return 1_800_000_000.


def memory_end(**over):
    end = dict(utc='2026-11-10T17:00:30Z', uptime=104.5, uptime_start=100., seconds=4.5, exit_code=-9, killed_by_signal=9, killed_by_supervisor=False, sigterm_received=False,
               memory_events=dict(max=1500, oom=1, oom_kill=1), cgroup=CGROUP, dmesg_readable=True, oom_lines=[dict(t=108.4, constraint='CONSTRAINT_MEMCG', oom_memcg=CGROUP, task_memcg=CGROUP)],
               dxg_lines=0)
    end.update(over)
    return end


@pytest.mark.parametrize('unit', ['configc', 'review'])
def test_a_memory_trial_reads_as_expected_writes_its_check_and_records_the_clock(unit):
    out = xl2.root()/f'{unit}_mem'
    unit_name = hb.unit_name(xl2.UNITS[unit]['kind'], xl2.TRIAL_PAIRS['mem_swap0'])
    cg = CGROUP.replace(UNIT, unit_name)
    system = TrialSystem(end=memory_end(cgroup=cg, oom_lines=[dict(t=108.4, constraint='CONSTRAINT_MEMCG', oom_memcg=cg, task_memcg=cg)]), result='exit-code',
                         dmesg=oom_line(95., oom_memcg='/other.service', task_memcg='/other.service')+oom_line(108.4, oom_memcg=cg, task_memcg=cg))
    doc = xl2.run_trial(unit, 'mem_swap0', out, system=system)
    assert doc['passed'] is True and (doc['reading']['status'], doc['reading']['status_kind']) == ('infra_failure', 'oom')
    check = json.loads((out/'check.json').read_text())
    assert check['passed'] and check['items']['own_limit_dmesg_line'] and check['items']['oom_kill_counted']
    assert doc['clock']['oom_minus_supervisor_end'] == [3.9] and doc['clock']['dmesg_newest_stamp'] == 108.4
    assert (out/'unit_reading.json').is_file() and (out/'units').is_dir() and json.loads((out/'limit_result.json').read_text())['passed'] is True
    if unit == 'review':
        stop = doc['queue_stop']
        assert stop['code'] == stop['expected_code'] == 20 and Path(stop['stop_record']).is_file()
        record = json.loads(Path(stop['stop_record']).read_text())
        assert record['seq'] == 1 and record['reading']['status'] == 'infra_failure' and Path(stop['stop_record']).is_relative_to(out)
    else:
        assert doc['queue_stop'] is None
    assert '--property=MemoryMax=512M' in system.runs[0] and '--property=MemorySwapMax=0' in system.runs[0]


def test_a_trial_whose_additional_conditions_fail_does_not_pass():
    """Plan revision 1 item 2: the reading alone is not enough -- an OOM line of a parent's memcg (not the unit's own limit) reads infra_failure under (乙) but fails the check."""
    out = xl2.root()/'parent'
    cg = CGROUP.replace(UNIT, hb.unit_name('configc', xl2.TRIAL_PAIRS['mem_swap256m']))
    end = memory_end(cgroup=cg, oom_lines=[dict(t=108.4, constraint='CONSTRAINT_MEMCG', oom_memcg='/user.slice', task_memcg=cg)])
    doc = xl2.run_trial('configc', 'mem_swap256m', out, system=TrialSystem(end=end, result='exit-code', dmesg=oom_line(108.4, oom_memcg='/user.slice', task_memcg=cg)))
    assert (doc['reading']['status'], doc['reading']['status_kind']) == ('infra_failure', 'oom') and doc['passed'] is False
    assert json.loads((out/'check.json').read_text())['items']['own_limit_dmesg_line'] is False


def test_a_runtime_trial_and_an_unreadable_one():
    out = xl2.root()/'rt'
    end = dict(utc='x', uptime=121., uptime_start=100., seconds=21., exit_code=-15, killed_by_signal=15, killed_by_supervisor=False, sigterm_received=True, memory_events=dict(max=0, oom=0, oom_kill=0),
               cgroup=CGROUP, dmesg_readable=True, oom_lines=[], dxg_lines=0)
    doc = xl2.run_trial('review', 'runtime', out, system=TrialSystem(end=end, result='timeout', dmesg=''))
    assert doc['passed'] is True and (doc['reading']['status'], doc['reading']['status_kind']) == ('unit_limit', 'RuntimeMaxSec') and doc['queue_stop']['code'] == 20
    assert '--property=MemorySwapMax=256M' in doc['argv'] and '--property=RuntimeMaxSec=20' in doc['argv']
    assert doc['argv'] == json.loads((out/'unit_reading.json').read_text())['argv'] and doc['argv_is_the_planned'] is True       # A5a review m9: the argv the unit ran (unit_reading.json)
    assert doc['check']['items']['no_oom_line_in_window'] is True
    out3 = xl2.root()/'rt_other_oom'                               # A5a review m9: the plan's oom_lines == [] -- another unit's OOM line in the window fails the runtime trial's check
    other = dict(t=110., constraint='CONSTRAINT_MEMCG', oom_memcg='/user.slice/other.service', task_memcg='/user.slice/other.service')
    doc = xl2.run_trial('review', 'runtime', out3, system=TrialSystem(end=dict(end, oom_lines=[other]), result='timeout', dmesg=''))
    assert doc['check']['items']['no_oom_line_in_window'] is False and doc['passed'] is False
    out2 = xl2.root()/'rt_unread'
    doc = xl2.run_trial('review', 'runtime', out2, system=TrialSystem(end=dict(end, memory_events=None, dmesg_readable=False, oom_lines=None), result='timeout', dmesg=None))
    assert doc['passed'] is False and doc['reading']['status'] == 'needs_lead_rejudgment' and doc['queue_stop']['code'] == 25
    with pytest.raises(hc.Refused):
        xl2.run_trial('review', 'runtime', out2, system=TrialSystem(end=end, result='timeout', dmesg=''))     # a new directory every time


def pq_cgroup(name):
    return f'/user.slice/user-1000.slice/user@1000.service/app.slice/{name}.service'


@pytest.mark.parametrize('unit', sorted(PQ_UNITS))
@pytest.mark.parametrize('trial', ['mem_swap0', 'mem_swap256m'])
def test_a_pq_memory_trial_reads_through_the_launchers_code_and_stops_as_the_launcher(unit, trial, monkeypatch):
    """P0S LB2: a PQ unit's memory trial runs through run_pqh.run_unit (the launcher's code after the start string), reads infra_failure (oom) under (乙), writes the launcher's unit_reading.json
    in <trial>/unit/ with that status, and the launcher's exit is FAILURE_EXIT (30): the D123 (b) STOP; check.json, the clock and limit_result.json as for the other units."""
    command, name, _, _ = PQ_UNITS[unit]
    calls = []
    real = run_pqh.run_unit
    monkeypatch.setattr(run_pqh, 'run_unit', lambda *a, **k: calls.append(a[:2]) or real(*a, **k))
    out = xl2.root()/f'{unit}_{trial}'
    cg = pq_cgroup(name)
    system = TrialSystem(end=memory_end(cgroup=cg, oom_lines=[dict(t=108.4, constraint='CONSTRAINT_MEMCG', oom_memcg=cg, task_memcg=cg)]), result='exit-code',
                         dmesg=oom_line(95., oom_memcg='/other.service', task_memcg='/other.service')+oom_line(108.4, oom_memcg=cg, task_memcg=cg))
    doc = xl2.run_trial(unit, trial, out, system=system)
    assert calls == [(command, name)] and len(system.runs) == 1
    assert doc['passed'] is True and (doc['reading']['status'], doc['reading']['status_kind']) == ('infra_failure', 'oom') and doc['unit_name'] == name and doc['command'] == command
    stop = doc['queue_stop']
    assert stop['code'] == stop['expected_code'] == run_pqh.FAILURE_EXIT and stop['stop_record_status'] == 'infra_failure'
    record = json.loads(Path(stop['stop_record']).read_text())
    assert Path(stop['stop_record']) == out/'unit'/'unit_reading.json' and record['status'] == 'infra_failure' and record['unit'] == name and record['command'] == command
    assert record['argv'] == doc['argv'] == system.runs[0] and doc['argv_is_the_planned'] is True
    assert json.loads((out/'check.json').read_text())['passed'] and doc['clock']['oom_minus_supervisor_end'] == [3.9]
    assert (out/'unit'/'launch.log').is_file() and (out/'launcher.log').is_file() and json.loads((out/'limit_result.json').read_text())['passed'] is True
    assert '--property=MemoryMax=512M' in system.runs[0] and f"--property=MemorySwapMax={'0' if trial == 'mem_swap0' else '256M'}" in system.runs[0]
    assert hc.local_unit_end()['unit'] == name                       # the tool's own note of the unit's end (A5a review I1), as hb_core.run_unit


@pytest.mark.parametrize('unit', sorted(PQ_UNITS))
def test_a_pq_runtime_trial_its_check_and_an_unreadable_one(unit):
    """P0S LB2: RuntimeMaxSec reads unit_limit with the launcher's STOP; systemd's Result is read from the launcher's launch.log (run_pqh.unit_result) for the runtime check; another unit's OOM
    line in the window fails the check (A5a review m9); unreadable evidence reads needs_lead_rejudgment and does not pass (the launcher still stops, 30)."""
    _, name, _, _ = PQ_UNITS[unit]
    cg = pq_cgroup(name)
    end = dict(utc='x', uptime=121., uptime_start=100., seconds=21., exit_code=-15, killed_by_signal=15, killed_by_supervisor=False, sigterm_received=True, memory_events=dict(max=0, oom=0, oom_kill=0),
               cgroup=cg, dmesg_readable=True, oom_lines=[], dxg_lines=0)
    out = xl2.root()/f'{unit}_rt'
    doc = xl2.run_trial(unit, 'runtime', out, system=TrialSystem(end=end, result='timeout', dmesg=''))
    assert doc['passed'] is True and (doc['reading']['status'], doc['reading']['status_kind']) == ('unit_limit', 'RuntimeMaxSec') and doc['reading']['systemd_result'] == 'timeout'
    assert doc['queue_stop']['code'] == doc['queue_stop']['expected_code'] == 30 and doc['check']['items']['systemd_result_timeout'] is True
    assert '--property=MemorySwapMax=256M' in doc['argv'] and '--property=RuntimeMaxSec=20' in doc['argv']
    other = dict(t=110., constraint='CONSTRAINT_MEMCG', oom_memcg='/user.slice/other.service', task_memcg='/user.slice/other.service')
    doc = xl2.run_trial(unit, 'runtime', xl2.root()/f'{unit}_rt_other', system=TrialSystem(end=dict(end, oom_lines=[other]), result='timeout', dmesg=''))
    assert doc['check']['items']['no_oom_line_in_window'] is False and doc['passed'] is False
    out2 = xl2.root()/f'{unit}_rt_unread'
    doc = xl2.run_trial(unit, 'runtime', out2, system=TrialSystem(end=dict(end, memory_events=None, dmesg_readable=False, oom_lines=None), result='timeout', dmesg=None))
    assert doc['passed'] is False and doc['reading']['status'] == 'needs_lead_rejudgment' and doc['queue_stop']['code'] == 30
    with pytest.raises(hc.Refused):
        xl2.run_trial(unit, 'runtime', out2, system=TrialSystem(end=end, result='timeout', dmesg=''))       # a new directory every time


def test_a_pq_trial_whose_limit_was_not_reached_does_not_pass():
    """The load finished without the limit (no OOM evidence, systemd's Result success): the launcher reads no status and returns the unit's exit; the trial does not pass (expected infra_failure)."""
    name = PQ_UNITS['pq_unit'][1]
    end = memory_end(cgroup=pq_cgroup(name), exit_code=0, killed_by_signal=None, memory_events=dict(max=0, oom=0, oom_kill=0), oom_lines=[])
    doc = xl2.run_trial('pq_unit', 'mem_swap0', xl2.root()/'pq_not_reached', system=TrialSystem(end=end, result='success', dmesg=''))
    assert doc['reading']['status'] is None and doc['queue_stop']['code'] == 0 and doc['queue_stop']['expected_code'] == 30 and doc['passed'] is False


# ---- plan revision 1 item 6: what the first batch ran (a99724d) is unchanged ------------------------------------------------------------------------
FIRST_BATCH = 'a99724d'
FUNCTIONS = {'experiments/holdout/holdout_common.py': ('classify_unit_end', 'oom_evidence', 'parse_oom', 'in_unit', 'is_unit_cgroup', 'OOM_LINE', 'KERNEL_TIME', 'JOURNAL_OOM', 'LIMIT_RULE',
                                                       'LIMIT_RULES'),
             'experiments/xb/xb_chain.py': ('unit_prefix', 'supervisor', 'unit_reading', 'unit_result'),
             'experiments/xb/xb_core.py': ('UNIT_PROPERTIES',)}
FILES = ('experiments/xb/xb_unit.py', 'experiments/he/unit_probe.py', 'experiments/xb/xb_limit_experiment.py', 'experiments/xb/xb_synthetic_load.py')


def _named(source, names):
    tree = ast.parse(source)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            out[node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            out[node.targets[0].id] = ast.dump(node, include_attributes=False)
    return out


def test_the_first_batchs_functions_and_files_are_as_they_ran():
    """Plan revision 1 item 6 (B4 review I6): between the first batch's commit and this tree, the discriminator's functions and constants (holdout_common), the chain's start-string and reading
    functions (xb_chain), UNIT_PROPERTIES (xb_core) are the same (AST, docstrings included), and the supervisor, the probe, the first batch's tool and the load are the same bytes.  A5a's changes
    keep them so (the second batch is a file of its own; the 10 s gap is the machine gate's)."""
    done = subprocess.run(['git', '-C', str(W.ROOT), 'cat-file', '-e', f'{FIRST_BATCH}^{{commit}}'], capture_output=True)
    if done.returncode != 0:
        pytest.skip(f'{FIRST_BATCH} is not in this repository')
    for rel, names in FUNCTIONS.items():
        old = subprocess.run(['git', '-C', str(W.ROOT), 'show', f'{FIRST_BATCH}:{rel}'], capture_output=True, text=True, check=True).stdout
        then, now = _named(old, names), _named((W.ROOT/rel).read_text(encoding='utf-8'), names)
        assert sorted(then) == sorted(names) and then == now, (rel, sorted(k for k in names if then.get(k) != now.get(k)))
    for rel in FILES:
        old = subprocess.run(['git', '-C', str(W.ROOT), 'show', f'{FIRST_BATCH}:{rel}'], capture_output=True, check=True).stdout
        assert old == (W.ROOT/rel).read_bytes(), rel
