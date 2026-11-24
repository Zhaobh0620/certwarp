"""PQH's command line (run_pqh.py): the units' start strings (C22.H5-2, C22.H6-3), the launcher's reading of a unit's end with holdout_common.classify_unit_end (D123 (b): a timeout or OOM evidence
stops, unreadable evidence needs the lead), and the dry run on this tree without BZ's unseal record (both hold-out units refused at the admission; nothing written).  systemd-run is the sandbox's
shim (exit 99); the launcher's machine is a fake (CPU, tmp_path)."""
import json
from pathlib import Path

import pytest

from .. import _p0_pins as P0
from . import _world as W

hp, hu = W.hp, W.hu
import run_pqh  # noqa: E402


@pytest.mark.parametrize('command, memory, runtime, unit', [('members', '12G', 4800, 'tag-hb-pq-members'), ('out', '11G', 7200, 'tag-hb-pq-out'),
                                                            ('members-dev', '12G', 4800, 'tag-hb-pq-members-dev'), ('p3-replay', '11G', 7200, 'tag-hb-pq-p3-replay'),
                                                            ('replay-gate', '8G', 3600, 'tag-hb-pq-replay')])     # m6: the replay gate's own unit
def test_the_start_strings(command, memory, runtime, unit, tmp_path):
    argv = run_pqh.start_argv(command, record_dir=tmp_path/'u', resume='splits/bayesreg_v1/holdout_B_v1_pqh_resume_1.json' if command == 'out' else None)
    assert argv[:4] == ['systemd-run', '--user', '--wait', '--collect'] and f'--unit={unit}' in argv
    for prop in (f'--property=MemoryMax={memory}', '--property=MemorySwapMax=4G', '--property=OOMPolicy=continue', f'--property=RuntimeMaxSec={runtime}', '--nice=5'):
        assert prop in argv
    i = argv.index('--')
    assert argv[i-5].endswith('experiments/xb/xb_unit.py') and argv[i+4] == command
    env = [a for a in argv if a.startswith('--setenv=PYTHONPATH=')][0]
    assert (run_pqh.FROZEN in env) == (command != 'members' and command != 'members-dev')
    if command == 'out':
        assert argv[-2:] == ['--resume', 'splits/bayesreg_v1/holdout_B_v1_pqh_resume_1.json']


class FakeSystem:
    def __init__(self, *, result='success', end=None, dmesg='', rc=0):
        self.result, self.end, self.text, self.rc = result, end, dmesg, rc

    def run(self, argv, log_path):
        record_dir = Path(argv[argv.index('--record-dir')+1])
        Path(log_path).write_text(f'Running as unit: x\nFinished with result: {self.result}\n' if self.result else 'no result line\n')
        if self.end is not None:
            (record_dir/'unit_end.json').write_text(json.dumps(self.end))
        return self.rc

    def uptime(self):
        return 100.

    def dmesg(self):
        return self.text

    def journal(self, unit, since_epoch):
        return ''

    def now(self):
        return 1.7e9


END = dict(memory_events=dict(oom=0, oom_kill=0), cgroup='/user.slice/tag-hb-pq-out-dev.service', exit_code=0, killed_by_supervisor=False, dmesg_readable=True, oom_lines=[])


@pytest.mark.parametrize('system, want', [(FakeSystem(end=END), 0), (FakeSystem(result='timeout', end=END, rc=143), run_pqh.FAILURE_EXIT),
                                          (FakeSystem(result='oom-kill', end=None, rc=137), run_pqh.FAILURE_EXIT), (FakeSystem(end=None, rc=137), run_pqh.FAILURE_EXIT)])
def test_the_launcher_reads_the_units_end(system, want, tmp_path):
    assert run_pqh.launch('out-dev', system=system, log=W.quiet) == want
    reading = json.loads(next(Path(hp.DEV_PQ_ROOT).glob('units/*/unit_reading.json')).read_text())
    if want == 0:
        assert reading['status'] is None
    elif system.result == 'timeout':
        assert (reading['status'], reading['kind']) == ('unit_limit', 'RuntimeMaxSec')
    elif system.result == 'oom-kill':
        assert reading['status'] == 'infra_failure'
    else:
        assert reading['status'] == 'needs_lead_rejudgment'


BEFORE_RUN_UNIT = '3a2775f7'                    # main before P0S LB2 moved launch's tail into run_unit


def test_launch_is_its_old_body_split_at_the_start_string():
    """P0S LB2: run_pqh.launch's statements after the start string moved, unchanged, into run_pqh.run_unit (which the limit experiment's second batch calls for PQ's two units); launch now
    returns run_unit(...) of its own argv and record directory.  Compared on the AST with the launch of BEFORE_RUN_UNIT (docstrings aside)."""
    import ast
    import subprocess
    root = Path(run_pqh.__file__).resolve().parents[2]
    done = subprocess.run(['git', '-C', str(root), 'show', f'{BEFORE_RUN_UNIT}:experiments/pq/run_pqh.py'], capture_output=True, text=True)
    if done.returncode != 0:
        pytest.skip(f'{BEFORE_RUN_UNIT} is not in this repository')

    def body(source, name):
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
        statements = node.body[1:] if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) else node.body
        return node, [ast.dump(s, include_attributes=False) for s in statements]
    _, old = body(done.stdout, 'launch')
    now = Path(run_pqh.__file__).read_text(encoding='utf-8')
    launch_node, new_launch = body(now, 'launch')
    run_node, new_run = body(now, 'run_unit')
    cut = next(i for i, s in enumerate(old) if "id='argv'" in s.split('value=')[0])     # argv = start_argv(...)
    assert new_launch[:cut+1] == old[:cut+1] and len(new_launch) == cut+2
    assert new_launch[-1] == ast.dump(ast.parse('return run_unit(command, name, argv, record_dir, system=system, log=log)').body[0], include_attributes=False)
    assert new_run[0] == old[0] and 'holdout_common' in new_run[0] and new_run[1:] == old[cut+1:]
    assert [a.arg for a in run_node.args.args] == ['command', 'name', 'argv', 'record_dir'] and [a.arg for a in run_node.args.kwonlyargs] == ['system', 'log']


def test_the_dry_run_refuses_both_hold_out_units_at_the_admission(monkeypatch):
    import hi_core
    monkeypatch.setattr(hi_core, 'DEVCHECK_DECISION', None)
    monkeypatch.setattr(hp, 'PREREG_SHA256', None)                  # P0 pinned it (runs/20261114-p0/P0_pins.md); the unpinned state, set back here, as before
    for command, admission in (('out', 'admission of the first pair'), ('members', "admission: hi_core.load_unseal")):
        lines = []
        W.XW.sealed_module._reset_cache()
        assert run_pqh.dry_run(command, log=lines.append) == 9
        text = '\n'.join(lines)
        row = next(line for line in lines if line.startswith(admission))
        assert 'PROBLEM' in row and 'unseal' in row
        assert 'PREREG_SHA256 is None' in text and 'start string' in text
    assert not Path(hp.MEMBERS_LOCK).exists() and not Path(hp.OUT_LOCK).exists()


def test_the_dry_run_at_p0_reports_the_preregistration_pinned(monkeypatch):
    """P0 (runs/20261114-p0/d1_expected.md section 6): with PREREG_SHA256 as committed both hold-out units' dry runs print `preregistration: pinned`, never
    `PREREG_SHA256 is None`, list no preregistration problem, and are still refused at the admission (no unseal record); nothing is locked."""
    import hi_core
    monkeypatch.setattr(hi_core, 'DEVCHECK_DECISION', None)
    P0.assert_pinned(hp.PREREG_SHA256, hp.PREREG_REL)
    for command, admission in (('out', 'admission of the first pair'), ('members', "admission: hi_core.load_unseal")):
        lines = []
        W.XW.sealed_module._reset_cache()
        assert run_pqh.dry_run(command, log=lines.append) == 9
        text = '\n'.join(lines)
        row = next(line for line in lines if line.startswith(admission))
        assert 'PROBLEM' in row and 'unseal' in row
        assert 'preregistration: pinned' in lines and 'PREREG_SHA256 is None' not in text and 'start string' in text
        assert not any(line.startswith('problems:') and "'preregistration:" in line for line in lines)
    assert not Path(hp.MEMBERS_LOCK).exists() and not Path(hp.OUT_LOCK).exists()


def test_the_command_line_prints_the_start_string(capsys):
    assert run_pqh.main(['launch', 'members', '--print']) == 0
    assert 'tag-hb-pq-members' in capsys.readouterr().out
    assert run_pqh.main(['launch', 'out-dev', '--resume', 'x']) == 2
