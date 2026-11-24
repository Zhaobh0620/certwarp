"""PQ's driver (experiments/pq/run_pq.py): the start strings, the machine gate on fake answers, the own-unit check, the refusal outside a unit, the reading code's dry run.  No unit is started, no
machine is queried (systemctl and nvidia-smi are replaced by fakes)."""
import subprocess
from types import SimpleNamespace

import pytest

from . import _world as W

import run_pq  # noqa: E402

core, pins = W.core, W.pins


def test_the_start_strings():
    for unit in ('members', 'out', 'eval'):
        argv = run_pq.start_string(unit)
        assert argv[:3] == ['systemd-run', '--user', f'--unit=tag-pq-{unit}'] and 'pytest' not in ' '.join(argv) and argv[-1 if unit != 'members' else -3] == unit
        path = [a for a in argv if a.startswith('--setenv=PYTHONPATH=')][0]
        assert path == (f'--setenv=PYTHONPATH={pins.MAIN_REPO}' if unit == 'members' else f'--setenv=PYTHONPATH={pins.FROZEN}:{pins.MAIN_REPO}')
        assert f'--working-directory={pins.MAIN_REPO}' in argv and '--setenv=OPENBLAS_NUM_THREADS=1' in argv
    assert run_pq.start_string('members')[-2:] == ['--device', 'cuda']
    assert not any(u.startswith('seq-vp-') for u in run_pq.UNITS.values())


def fake_run(units='', util='0', apps=''):
    def run(argv, **kwargs):
        if argv[0] == 'systemctl':
            return subprocess.CompletedProcess(argv, 0, units, '')
        if '--query-gpu=utilization.gpu' in argv:
            return subprocess.CompletedProcess(argv, 0, util+'\n', '')
        return subprocess.CompletedProcess(argv, 0, apps, '')
    return run


@pytest.fixture
def meminfo(tmp_path):
    path = tmp_path/'meminfo'
    path.write_text('MemTotal: 15000000 kB\nMemAvailable:   12000000 kB\n')
    return path


def test_the_machine_gate(meminfo, tmp_path):
    record = run_pq.machine_gate('out', run=fake_run(), meminfo=meminfo, home=tmp_path)
    assert record['passed'] and record['other_units'] == []
    assert run_pq.machine_gate('out', run=fake_run('tag-pq-out.service loaded active running x\n'), meminfo=meminfo, home=tmp_path)['passed']
    with pytest.raises(core.PQRefused, match='other units'):
        run_pq.machine_gate('out', run=fake_run('tag-p3-queue.service loaded active running x\n'), meminfo=meminfo, home=tmp_path)
    with pytest.raises(core.PQRefused, match='other units'):
        run_pq.machine_gate('eval', run=fake_run('seq-vp-chain.service loaded active running x\n'), meminfo=meminfo, home=tmp_path)
    with pytest.raises(core.PQRefused, match='GPU busy'):
        run_pq.machine_gate('members', run=fake_run(util='55'), meminfo=meminfo, home=tmp_path)
    with pytest.raises(core.PQRefused, match='GPU busy'):
        run_pq.machine_gate('members', run=fake_run(apps='1234\n'), meminfo=meminfo, home=tmp_path)
    low = tmp_path/'low'
    low.write_text('MemAvailable:   2000000 kB\n')
    with pytest.raises(core.PQRefused, match='MemAvailable'):
        run_pq.machine_gate('out', run=fake_run(), meminfo=low, home=tmp_path)


def test_the_own_unit(tmp_path):
    cgroup = tmp_path/'cgroup'
    cgroup.write_text('0::/user.slice/user-1000.slice/user@1000.service/app.slice/tag-pq-out.service\n')
    assert run_pq.in_own_unit('out', cgroup) and not run_pq.in_own_unit('eval', cgroup)
    cgroup.write_text('0::/user.slice/user-1000.slice/session-1.scope\n')
    assert not run_pq.in_own_unit('out', cgroup)


def test_a_unit_refuses_outside_its_systemd_unit(monkeypatch, capsys):
    monkeypatch.setattr(run_pq, 'in_own_unit', lambda unit, cgroup='/proc/self/cgroup': False)
    called = []
    monkeypatch.setattr(run_pq, 'out_unit', lambda ctx: called.append(ctx))
    assert run_pq.main(['out']) == 8 and not called
    assert 'tag-pq-out' in capsys.readouterr().err


def test_the_reading_code_dry_run(world):
    lines = []
    assert run_pq.dry_run('decide', ctx=world.ctx, log=lines.append) == 0
    text = '\n'.join(lines)
    assert 'PQ 承重' in text and 'PQ 不承重' in text and 'problems: none' in text


def test_code_state_refuses_a_dirty_tree(tmp_path):
    W.git(tmp_path, 'init', '-q')
    (tmp_path/'a.txt').write_text('x')
    with pytest.raises(core.PQRefused, match='not clean'):
        run_pq.code_state(tmp_path, frozen_required=False)
