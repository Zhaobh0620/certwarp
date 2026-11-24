"""Unit limits against machine failures (prereg revision 1c item 1; holdout_common.classify_unit_end): synthetic dmesg text (CONSTRAINT_MEMCG of this unit and of a parent, CONSTRAINT_NONE,
CONSTRAINT_CPUSET), the end probe's memory.events, an unreadable dmesg, a timeout together with an OOM; both rules (乙) and (甲); the driver's step reading; the in-unit supervisor (xb_unit.py) as a
real subprocess with a sleeping child that the test kills or terminates, read by the chain's unit_reading."""
import json
import os
import signal
import subprocess
import time

import pytest

from . import _world as W

hc, xc = W.hc, W.xc
import xb_chain  # noqa: E402

UNIT = 'tag-hb-x-ants_s-0423-0099'
PARENT = '/user.slice/user-1000.slice/user@1000.service/app.slice'
CG = f'{PARENT}/{UNIT}.service'
OTHER = f'{PARENT}/other.service'


def line(t, constraint, oom_memcg, task_memcg, extra=''):
    return f'[ {t:.6f}] oom-kill:constraint={constraint},nodemask=(null),cpuset=/,mems_allowed=0,{extra}oom_memcg={oom_memcg},task_memcg={task_memcg},task=python,pid=4321,uid=1000\n'


MEMCG_OWN = line(100, 'CONSTRAINT_MEMCG', CG, CG)
MEMCG_PARENT = line(100, 'CONSTRAINT_MEMCG', PARENT, CG)
MEMCG_OTHER = line(100, 'CONSTRAINT_MEMCG', OTHER, OTHER)
NONE_OURS = f'[ 100.000000] oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=/,mems_allowed=0,global_oom,task_memcg={CG},task=python,pid=1,uid=1000\n'
NONE_OTHER = f'[ 100.000000] oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=/,mems_allowed=0,global_oom,task_memcg={OTHER},task=x,pid=2,uid=1000\n'
CPUSET = line(100, 'CONSTRAINT_CPUSET', '/', CG)
NOISE = '[ 50.000000] mini_init (246): drop_caches: 1\n[ 100.500000] Memory cgroup out of memory: Killed process 4321 (python)\n'
QUIET = dict(max=0, oom=0, oom_kill=0)


def read(text, *, result='exit-code', events=QUIET, journal='', rule='yi', since=0., step_kill=False, unit_cgroup=CG):
    return hc.classify_unit_end(unit=UNIT, systemd_result=result, oom_lines=hc.parse_oom(text, since), events=events, journal=journal, step_kill=step_kill, unit_cgroup=unit_cgroup, rule=rule)[:2]


def test_the_oom_lines_are_parsed():
    rows = hc.parse_oom(NOISE+MEMCG_OWN+NONE_OTHER+CPUSET, 0.)
    assert [r['constraint'] for r in rows] == ['CONSTRAINT_MEMCG', 'CONSTRAINT_NONE', 'CONSTRAINT_CPUSET']
    assert rows[0]['oom_memcg'] == CG and rows[1]['global_oom'] and hc.in_unit(rows[2]['task_memcg'], UNIT, CG)
    assert hc.parse_oom(MEMCG_OWN, 150.) == [] and hc.parse_oom(None) is None
    assert hc.dxg_lines('[ 1.0] dxgk: create_allocation failed: -12\n', 0.) == 1 and hc.dxg_lines(None) is None
    assert hc.in_unit(CG+'/sub', UNIT) and not hc.in_unit(OTHER, UNIT) and hc.is_unit_cgroup(CG, UNIT) and not hc.is_unit_cgroup(PARENT, UNIT, CG)


@pytest.mark.parametrize('text, kw, yi, jia', [
    (MEMCG_OWN, {}, ('infra_failure', 'oom'), ('unit_limit', 'MemoryMax')),                                 # the unit's own MemoryMax
    (MEMCG_PARENT, {}, ('infra_failure', 'oom'), ('infra_failure', 'oom_not_the_units_own_limit')),        # a parent's memory limit killed a task of the unit
    (NONE_OURS, {}, ('infra_failure', 'machine_oom'), ('infra_failure', 'machine_oom')),                   # the machine
    (NONE_OTHER, {}, ('infra_failure', 'machine_oom'), ('infra_failure', 'machine_oom')),                  # the machine, another task: still a machine event
    (CPUSET, {}, ('infra_failure', 'oom'), ('infra_failure', 'oom_not_the_units_own_limit')),
    (MEMCG_OTHER, {}, (None, None), (None, None)),                                                         # another unit's own limit: not ours
    ('', dict(events=dict(max=3, oom=1, oom_kill=1)), ('infra_failure', 'oom'), ('unit_limit', 'MemoryMax')),     # the end probe alone
    ('', dict(events=dict(max=3, oom=0, oom_kill=1)), ('infra_failure', 'oom'), ('infra_failure', 'oom_not_the_units_own_limit')),
    ('', dict(journal='tag-hb-x-...service: A process of this unit has been killed by the OOM killer.'), ('infra_failure', 'oom'), ('infra_failure', 'oom_not_the_units_own_limit')),
    ('', dict(step_kill=True), ('infra_failure', 'oom'), ('infra_failure', 'oom_not_the_units_own_limit')),
    ('', dict(result='timeout'), ('unit_limit', 'RuntimeMaxSec'), ('unit_limit', 'RuntimeMaxSec')),
    (MEMCG_OWN, dict(result='timeout'), ('infra_failure', 'oom'), ('unit_limit', 'MemoryMax')),            # timeout and OOM together: OOM first
    (None, dict(result='timeout'), ('needs_lead_rejudgment', 'unreadable:dmesg'), ('needs_lead_rejudgment', 'unreadable:dmesg')),     # never read as 0
    ('', dict(result='timeout', events=None), ('needs_lead_rejudgment', 'unreadable:end_probe'), ('needs_lead_rejudgment', 'unreadable:end_probe')),
    ('', dict(result=None), ('needs_lead_rejudgment', 'unreadable:result'), ('needs_lead_rejudgment', 'unreadable:result')),
    (None, dict(events=dict(oom=0, oom_kill=2)), ('infra_failure', 'oom'), ('infra_failure', 'oom_not_the_units_own_limit')),      # OOM evidence decides even with dmesg unreadable
    ('', {}, (None, None), (None, None)),                                                                    # a normal end: the driver's record decides
    ('', dict(result='oom-kill'), ('infra_failure', 'oom'), ('infra_failure', 'oom_not_the_units_own_limit')),     # m3: systemd's Result oom-kill is OOM evidence
])
def test_the_rules(text, kw, yi, jia):
    assert read(text, rule='yi', **kw) == yi
    assert read(text, rule='jia', **kw) == jia


def test_the_default_rule_and_the_memory_guard():
    assert hc.LIMIT_RULE == 'yi' and set(hc.LIMIT_RULES) == {'yi', 'jia'}
    assert hc.classify_unit_end(unit=UNIT, systemd_result='exit-code', oom_lines=[], events=QUIET, memory_guard=True)[:2] == ('infra_failure', 'memory_guard')
    with pytest.raises(ValueError):
        hc.classify_unit_end(unit=UNIT, systemd_result='x', oom_lines=[], events=QUIET, rule='bing')


def test_the_drivers_step_reading(monkeypatch):
    monkeypatch.setenv('XB_UNIT', UNIT)
    step = dict(name='method', returncode=-9, memory_events_before=QUIET, memory_events_after=dict(max=4, oom=1, oom_kill=1), kernel_oom_lines=hc.parse_oom(MEMCG_OWN, 0.))
    assert isinstance(xc.step_outcome(step), hc.InfrastructureFailure)                                   # (乙): an exit -9 is OOM evidence
    assert isinstance(xc.step_outcome(dict(step, memory_events_after=QUIET, kernel_oom_lines=[])), hc.InfrastructureFailure)
    assert isinstance(xc.step_outcome(dict(step, memory_events_before=None, memory_events_after=None, kernel_oom_lines=None)), hc.InfrastructureFailure)
    monkeypatch.setattr(hc, 'LIMIT_RULE', 'jia')
    assert isinstance(xc.step_outcome(step), xc.UnitLimit)                                                # (甲): the unit's own limit
    assert isinstance(xc.step_outcome(dict(step, memory_events_after=QUIET, kernel_oom_lines=hc.parse_oom(MEMCG_PARENT, 0.))), hc.InfrastructureFailure)
    assert isinstance(xc.step_outcome(dict(step, returncode=0)), xc.UnitLimit)                            # I2: an exit 0 is read for OOM evidence too
    monkeypatch.setattr(hc, 'LIMIT_RULE', 'yi')
    assert isinstance(xc.step_outcome(dict(step, returncode=0)), hc.InfrastructureFailure)
    assert isinstance(xc.step_outcome(dict(step, returncode=0, memory_events_after=QUIET, kernel_oom_lines=hc.parse_oom(NONE_OTHER, 0.))), hc.InfrastructureFailure)   # a machine OOM in the window
    assert xc.step_outcome(dict(step, returncode=0, memory_events_after=QUIET, kernel_oom_lines=[])) is None
    assert xc.step_outcome(dict(step, returncode=1, memory_events_after=QUIET, kernel_oom_lines=hc.parse_oom(MEMCG_OTHER, 0.))) is None      # another unit's OOM: the caller classes exit 1


# ---- the in-unit supervisor (a real subprocess) --------------------------------------------------------------------------------------------------
def supervisor(tmp_path, child_seconds=30):
    record = tmp_path/'unit'
    argv = [str(xc.PY), '-I', '-B', str(W.ROOT/'experiments/xb/xb_unit.py'), '--record-dir', str(record), '--unit', UNIT, '--', str(xc.PY), '-I', '-B', '-c', f'import time; time.sleep({child_seconds})']
    proc = subprocess.Popen(argv, start_new_session=True)
    for _ in range(200):
        if (record/'unit_start.json').exists():
            break
        time.sleep(.05)
    return proc, record, json.loads((record/'unit_start.json').read_text())


def child_pid(proc):
    for _ in range(200):
        out = subprocess.run(['pgrep', '-P', str(proc.pid)], capture_output=True, text=True).stdout.split()
        if out:
            return int(out[0])
        time.sleep(.05)
    raise AssertionError('no child')


def finish(proc, expect):
    try:
        assert proc.wait(timeout=60) == expect
    finally:
        if proc.poll() is None:
            proc.kill()


def test_the_supervisor_records_a_killed_child_and_the_reading_is_infrastructure(tmp_path):
    proc, record, start = supervisor(tmp_path)
    os.kill(child_pid(proc), signal.SIGKILL)
    finish(proc, 137)
    end = json.loads((record/'unit_end.json').read_text())
    assert end['exit_code'] == -9 and end['killed_by_signal'] == 9 and not end['killed_by_supervisor'] and end['utc'] and start['utc'] and end['seconds'] >= 0
    assert 'memory_events' in end and 'dmesg_readable' in end
    status, kind, evidence = xb_chain.unit_reading(unit=UNIT, result_line='exit-code', unit_end=end, oom_lines=[], journal='')
    assert (status, kind) == ('infra_failure', 'oom') and any(i['source'] == 'exit -9' for i in evidence['oom_evidence'])


def test_the_supervisor_records_a_sigterm_and_the_reading_is_the_runtime_limit(tmp_path):
    proc, record, start = supervisor(tmp_path)
    child_pid(proc)
    os.kill(proc.pid, signal.SIGTERM)
    finish(proc, 143)
    end = json.loads((record/'unit_end.json').read_text())
    assert end['sigterm_received'] and end['exit_code'] == -15 and not end['killed_by_supervisor']
    # outside a unit the probe reads the test session's cgroup, whose counters are not a fresh unit's: the reading is checked with a quiet probe
    status, kind, _ = xb_chain.unit_reading(unit=UNIT, result_line='timeout', unit_end=dict(end, memory_events=QUIET), oom_lines=[], journal='')
    assert (status, kind) == ('unit_limit', 'RuntimeMaxSec')


def test_the_supervisor_passes_a_normal_exit(tmp_path):
    proc, record, start = supervisor(tmp_path, child_seconds=0)
    finish(proc, 0)
    end = json.loads((record/'unit_end.json').read_text())
    assert end['exit_code'] == 0 and end['killed_by_signal'] is None


def test_the_supervisors_own_kill_after_a_sigterm_is_not_oom_evidence():
    end = dict(exit_code=-9, killed_by_supervisor=True, sigterm_received=True, memory_events=QUIET, cgroup=CG)
    assert xb_chain.unit_reading(unit=UNIT, result_line='timeout', unit_end=end, oom_lines=[], journal='')[:2] == ('unit_limit', 'RuntimeMaxSec')
    assert xb_chain.unit_reading(unit=UNIT, result_line='timeout', unit_end=None, oom_lines=[], journal='')[:2] == ('needs_lead_rejudgment', 'unreadable:end_probe')
