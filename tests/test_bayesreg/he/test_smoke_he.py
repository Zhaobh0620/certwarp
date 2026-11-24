"""HES (runs/20261106-he/任务书_HES冒烟驱动.md): experiments/he/smoke_he.py on synthetic files only (CPU; no unit is started, no GPU, no truth of a real pair).

- the units: their order, names and start strings; the queue's gate, hold file, stop on a failure, a root that runs once;
- the dry run in its own process: nothing written, torch not imported;
- the forced context of (c)(d): settings_frozen false, the identity checked, marked, and nowhere but <results>/he_smoke*/c|d/0438-0439; the forcing ends the real repair_field at round_cap;
- the check: every item with a passing world and a counterexample (records written by hand; the recomputation of item 4 also through bayesreg.eval on a synthetic pair).
HE2: the references hs_v1 (pass 0) and p3_v1 (pass 2, the ratios, the repair) in the place of fd2_v1 and rw_v1, and the contract's two hashes; items 3 and 10-13 (C22.B4-6 (a)-(e)); the gate of c;
systemd's reading in item 7; the cgroups' oom_kill in item 12.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import pytest

from jd_curl.three_d.grid import make_grid_3d

ROOT = Path(__file__).resolve().parents[3]
for _d in ('he', 'fl', 'ef', 'mc0'):
    sys.path.insert(0, str(ROOT/'experiments'/_d))

import firewall_c as fc  # noqa: E402
import he_core as he  # noqa: E402
import run_hx as R  # noqa: E402
import smoke_he as S  # noqa: E402

run = S.run
PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'
FROZEN = Path('/home/rmcfz/seq_vp_tag_results/frozen_eb808be')
RUNNER = 'py'+'test'                            # the word no start string may hold, never written whole in a string here


@pytest.fixture
def work():
    """A temporary directory whose path does not name the test runner (tmp_path's pytest-of-<user> parent does, and no start string may hold that word: run_mc0.check_launch)."""
    d = Path(tempfile.mkdtemp(prefix='hes_'))
    assert RUNNER not in str(d)
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---- the units -------------------------------------------------------------------------------------------------------------------------------------
def test_the_units_their_order_names_and_start_strings(work):
    assert [s.name for s in S.STEPS] == ['a', 'a-eval', 'b', 'c', 'c-eval', 'd', 'd-eval', 'check']
    assert [s.unit for s in S.STEPS] == [f'tag-he-smoke-{n}' for n in ('a', 'a-eval', 'b', 'c', 'c-eval', 'd', 'd-eval', 'check')]
    assert [s.gpu for s in S.STEPS] == [True, False, True, True, False, True, False, False]
    root = Path('/home/rmcfz/seq_vp_tag_results/he_smoke_v1')                 # only strings are built
    code = root/'code'
    d = {arm: f'{root}/{arm}/{S.SHORT}' for arm in 'abcd'}
    argv = {s.name: S.step_argv(s, code, root) for s in S.STEPS}
    for s in S.STEPS:
        a = argv[s.name]
        text = ' '.join(a)
        assert a[:2] == ['systemd-run', '--user'] and f'--unit={s.unit}' in a and '--wait' in a and '--collect' in a and f'--working-directory={code}' in a
        assert RUNNER not in text and 'seq-vp-' not in text and f'--setenv=PYTHONPATH={S.core.FROZEN}:{code}' in a and '--setenv=OPENBLAS_NUM_THREADS=1' in a
        assert ('--setenv=CUDA_VISIBLE_DEVICES=' in a) == (not s.gpu)
        assert f'--property=StandardOutput=append:{root}/{s.arm}/{s.unit}.log' in a
        res = S.SUPERVISOR if s.kind in ('pair', 'spy', 'forced') else S.EVALUATION
        assert f"--property=RuntimeMaxSec={res['runtime']}" in a and f"--property=MemoryMax={res['memory_max']}" in a and f"--property=CPUQuota={res['quota']}%" in a
        assert f"--setenv=NUMBA_NUM_THREADS={res['numba']}" in a and '--property=MemorySwapMax=4G' in a and f"--nice={res['nice']}" in a
        supervisor = s.kind in ('pair', 'spy', 'forced')
        assert len([x for x in a if 'OOMPolicy' in x]) == 1 and a[a.index('--property=MemorySwapMax=4G')+1] == '--property=OOMPolicy=continue'   # revision 1; HE2 review minor 14: every unit
        assert ('--property=MemoryMax=12G' in a) == supervisor and he.unit_argv_problems(a, res) == []                                     # HE2 review minors 8, 11: built from he_core's unit
    assert argv['a'][-14:] == [S.PY, '-B', f'{code}/experiments/he/run_hx.py', 'pair', '--stage-dir', str(S.STAGE), '--pair', S.PAIR, '--out', d['a'], '--device', 'cuda', '--launcher', 'unit']
    assert argv['b'][-13:] == [S.PY, '-B', f'{code}/experiments/he/spy_pair.py', '--stage-dir', str(S.STAGE), '--pair', S.PAIR, '--out', d['b'], '--device', 'cuda', '--launcher', 'unit']
    assert argv['c'][-8:] == [f'{code}/{S.SCRIPT}', 'forced', '--launcher', 'unit', '--out', d['c'], '--device', 'cuda']
    assert argv['d'][-8:] == [f'{code}/{S.SCRIPT}', 'forced', '--launcher', 'direct', '--out', d['d'], '--device', 'cuda']
    assert argv['a-eval'][-4:] == [f'{code}/experiments/he/run_hx.py', 'evaluate', '--context', f"{d['a']}/context.json"]
    assert argv['c-eval'][-4:] == [f'{code}/{S.SCRIPT}', 'evaluate-forced', '--context', f"{d['c']}/context.json"]
    assert argv['d-eval'][-4:] == [f'{code}/{S.SCRIPT}', 'evaluate-forced', '--context', f"{d['d']}/context.json"]
    assert argv['check'][-4:] == [f'{code}/{S.SCRIPT}', 'check', '--root', str(root)]
    launch = S.queue_launch(root)
    assert launch[:2] == ['systemd-run', '--user'] and f'--unit={S.QUEUE_UNIT}' in launch and '--wait' not in launch and f'--working-directory={S.core.ROOT}' in launch
    assert launch[launch.index('-B')+1] == str(S.core.ROOT/S.SCRIPT) and S.core.ROOT.resolve() == Path(S.__file__).resolve().parents[2]      # SM3 revision 3: the tree that runs this code
    assert launch[-6:] == [S.PY, '-B', str(S.core.ROOT/S.SCRIPT), 'queue', '--root', str(root)] and RUNNER not in ' '.join(launch) and '--setenv=CUDA_VISIBLE_DEVICES=' in launch
    assert S.queue_launch(root, 'abc')[-2:] == ['--commit', 'abc']
    named = work/f'{RUNNER}-x'                                           # a root that names the test runner is refused in every string
    with pytest.raises(ValueError, match='forbidden'):
        S.step_argv(S.STEPS[0], named/'code', named)
    with pytest.raises(ValueError, match='forbidden'):
        S.queue_launch(named)
    assert S.root_problem(root) is None and S.root_problem(work/'he_smoke_v1') is not None and S.root_problem(root.parent/'he_v1') is not None
    assert S.root_problem(work/'he_smoke_v1', results=work) is None
    assert [S.expected_files(s, root)[0].name for s in S.STEPS] == ['line.json', 'evaluation.json', 'spy.json', 'line.json', 'evaluation.json', 'line.json', 'evaluation.json', 'smoke_report.json']


# ---- the queue ---------------------------------------------------------------------------------------------------------------------------------------
class FakeSystem:
    """The machine as the queue sees it; run() leaves the files a unit would (or not, or exits non-zero)."""

    def __init__(self, root, *, rc=None, skip=()):
        self.root, self.rc, self.skip = Path(root), dict(rc or {}), set(skip)
        self.launched, self.clock, self.up = [], 0., 100.
        self.load, self.free_gpu, self.free_home, self.avail, self.free_host, self.other = .2, 23000., 500., 12., 70., []

    def load1(self):
        return self.load

    def units(self):
        return list(self.other)

    def gpu_free_mib(self):
        return self.free_gpu

    def home_free_gb(self):
        return self.free_home

    def mem_available_gib(self):
        return self.avail

    def host_free_gib(self):
        return self.free_host

    def now(self):
        return self.clock

    def sleep(self, seconds):
        self.clock += seconds

    def uptime(self):
        self.up += 1.
        return self.up

    def code_checkout(self, root, commit):
        return Path(root)/'code', commit

    def code_identity(self, code):
        return {}, []

    def run(self, argv, log):
        unit = next(a for a in argv if a.startswith('--unit=')).split('=', 1)[1]
        assert RUNNER not in ' '.join(argv)
        self.launched.append(unit)
        step = next(s for s in S.STEPS if s.unit == unit)
        self.clock += 60.
        if unit not in self.skip:
            for path in S.expected_files(step, self.root):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(dict(summary='1 过') if path.name == 'smoke_report.json' else {}))
        return self.rc.get(unit, 0)


def options(root, **kw):
    return SimpleNamespace(**dict(dict(root=str(root), commit=None, wait_minutes=1, gap_seconds=5, min_host_gib=15.), **kw))


@pytest.fixture
def quiet_inputs(monkeypatch):
    monkeypatch.setattr(S, 'input_problems', lambda *a, **k: [])


def test_the_queue_runs_the_units_one_by_one_and_ends_with_the_check(work, quiet_inputs):
    root = work/'he_smoke_t'
    root.mkdir()
    system, logs = FakeSystem(root), []
    assert S.queue_body(options(root), system, root, logs.append) == 0
    assert system.launched == [s.unit for s in S.STEPS]
    batch = json.loads((root/'batch.json').read_text())
    assert batch['code_sha'] == S.git_commit('HEAD') and batch['plan'] == system.launched and batch['forced_settings_sha256'] == he.settings_sha256(S.forced_settings())
    for s in S.STEPS:
        launched, done = (json.loads((root/'steps'/f'{s.name}.{k}.json').read_text()) for k in ('launched', 'done'))
        assert launched['argv'] == S.step_argv(s, root/'code', root) and launched['uptime_before'] < done['uptime_after'] and done['rc'] == 0 and done['missing'] == []
        assert launched['gate_mem_available_gib'] == (7. if s.name == 'check' else 10.)                                         # revision 1: (d)'s gate; HE2: every unit that can start the frozen run
        assert launched['mem_available_gib_at_launch'] == system.avail == 12.                                                   # HE2 clean: the reading at launch
    assert json.loads((root/'queue_end.json').read_text())['summary'] == '1 过' and system.clock >= 8*65
    assert not list(root.glob('.*.tmp'))
    with pytest.raises(run.Stop, match='runs once'):                                  # never rerun: not even a clean finish starts again
        S.queue_body(options(root), FakeSystem(root), root, logs.append)


@pytest.mark.parametrize('how', ['exit', 'no_record'])
def test_a_failure_stops_the_queue_and_nothing_after_it_starts(work, monkeypatch, quiet_inputs, how):
    monkeypatch.setattr(S, 'RESULTS', work)
    root = work/'he_smoke_t'
    system = FakeSystem(root, rc={'tag-he-smoke-b': 1} if how == 'exit' else None, skip={'tag-he-smoke-b'} if how == 'no_record' else ())
    assert S.queue(options(root), system) == 8
    assert system.launched == ['tag-he-smoke-a', 'tag-he-smoke-a-eval', 'tag-he-smoke-b']
    stop = json.loads((root/'queue_stop.json').read_text())['reason']
    assert 'tag-he-smoke-b' in stop and ('exited 1' in stop if how == 'exit' else 'left no' in stop)
    done = json.loads((root/'steps/b.done.json').read_text())
    assert done['rc'] == (1 if how == 'exit' else 0) and bool(done['missing']) == (how == 'no_record') and not (root/'steps/c.launched.json').exists()
    again = FakeSystem(root)
    assert S.queue(options(root), again) == 8 and again.launched == [] and 'runs once' in json.loads((root/'queue_stop.json').read_text())['reason']
    kept = [json.loads(p.read_text())['reason'] for p in root.glob('queue_stop_*.json')]
    assert kept == [stop]                                                           # the first stop is kept under another name, not overwritten
    assert json.loads((root/'steps/b.done.json').read_text()) == done


def test_the_hold_file_and_the_gate(work, quiet_inputs):
    root = work/'he_smoke_t'
    root.mkdir()
    system, gate = FakeSystem(root), S.GATE
    assert run.busy_reason(system, root, True, gate) is None
    (root/'hold').write_text('')
    assert run.busy_reason(system, root, False, gate) == 'hold file'
    with pytest.raises(run.Stop, match='gate closed for 1 minutes: hold file'):
        S.queue_body(options(root), system, root, lambda text: None)
    assert system.launched == [] and not list((root/'steps').glob('*'))               # nothing launched while the hold file is there
    (root/'hold').unlink()
    system.other = [f'{S.QUEUE_UNIT}.service']
    assert run.busy_reason(system, root, True, gate) is None                          # the queue's own unit
    system.other = ['tag-p3-queue.service']
    assert 'other units' in run.busy_reason(system, root, False, gate)
    system.other = ['seq-vp-chain.service']
    assert 'other units' in run.busy_reason(system, root, False, gate)
    system.other = []
    for attr, value, gpu_only in (('load', 1.5, False), ('free_home', 99., False), ('avail', 6.9, False), ('free_host', 14.9, False), ('free_gpu', 15*1024-1, True)):
        old = getattr(system, attr)
        setattr(system, attr, value)
        assert run.busy_reason(system, root, True, gate) is not None
        assert (run.busy_reason(system, root, False, gate) is None) == gpu_only
        setattr(system, attr, old)
    assert gate.own_units == (f'{S.QUEUE_UNIT}.service',) and gate.load_limit == 1.5
    system.avail = 9.                                                                    # revision 1: (d) asks 10 GiB; HE2 (the gate of c): every unit that can start the frozen run, the check 7
    steps = {s.name: s for s in S.STEPS}
    assert 'MemAvailable < 10.0' in run.busy_reason(system, root, True, S.step_gate(steps['c'], gate)) and 'MemAvailable < 10.0' in run.busy_reason(system, root, True, S.step_gate(steps['d'], gate))
    assert run.busy_reason(system, root, False, S.step_gate(steps['check'], gate)) is None and 'MemAvailable < 10.0' in run.busy_reason(system, root, False, S.step_gate(steps['a-eval'], gate))


def test_a_start_before_any_unit_may_follow_a_stop_and_keeps_it(work, quiet_inputs):
    root = work/'he_smoke_t'
    root.mkdir()
    (root/'queue_stop.json').write_text(json.dumps(dict(reason='gate closed for 30 minutes')))
    system = FakeSystem(root)
    assert S.queue_body(options(root), system, root, lambda text: None) == 0
    assert len(list(root.glob('queue_stop_*.json'))) == 1 and not (root/'queue_stop.json').exists()
    other = work/'he_smoke_u'
    other.mkdir()
    (other/'batch.json').write_text(json.dumps(dict(code_sha='0'*40, plan=S.plan_names())))
    with pytest.raises(run.Stop, match='batch.json is at'):
        S.queue_body(options(other), FakeSystem(other), other, lambda text: None)
    with pytest.raises(run.Stop, match='is not HEAD'):
        S.queue_body(options(other, commit='HEAD~1'), FakeSystem(other), other, lambda text: None)


def test_the_inputs_and_the_identity_stop_the_queue_before_any_unit(work, monkeypatch):
    root = work/'he_smoke_t'
    root.mkdir()
    monkeypatch.setattr(S, 'input_problems', lambda *a, **k: ['references: missing'])
    system = FakeSystem(root)
    with pytest.raises(run.Stop, match='before any unit'):
        S.queue_body(options(root), system, root, lambda text: None)
    monkeypatch.setattr(S, 'input_problems', lambda *a, **k: [])
    system.code_identity = lambda code: ({}, ['the code paths are not clean'])
    with pytest.raises(run.Stop, match='identity'):
        S.queue_body(options(root), system, root, lambda text: None)
    assert system.launched == [] and not (root/'steps').exists()


# ---- the dry run ----------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not (FROZEN/'jd_curl').is_dir(), reason='the frozen worktree is not on this machine')
def test_the_dry_run_in_its_own_process_writes_nothing_and_imports_no_torch():
    root = S.RESULTS/'he_smoke_test_never_written'
    assert not root.exists()
    before = sorted(p.name for p in S.RESULTS.iterdir())
    shim = Path(tempfile.mkdtemp(prefix='hesdry_'))
    (shim/'systemd-run').write_text('#!/bin/sh\nexit 99\n')
    (shim/'systemd-run').chmod(0o755)
    environ = dict(os.environ, PATH=f"{shim}:{os.environ['PATH']}", CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', NUMBA_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
                   PYTHONPATH=f'{FROZEN}:{ROOT}')
    try:
        done = subprocess.run([PY, '-B', str(ROOT/S.SCRIPT), '--dry-run', '--root', str(root)], env=environ, capture_output=True, text=True, timeout=50, cwd=str(ROOT))
    finally:
        shutil.rmtree(shim)
    text = done.stdout
    assert done.returncode in (0, 8), text[-2000:]+done.stderr[-2000:]
    assert 'torch imported by this dry run: False; problems: ' in text and '--unit=tag-he-smoke-queue' in text and '--unit=tag-he-smoke-check ' in text and RUNNER not in text
    assert ('problems: none' in text) == (done.returncode == 0)
    assert not root.exists() and sorted(p.name for p in S.RESULTS.iterdir()) == before


# ---- the forced context (c)(d) ----------------------------------------------------------------------------------------------------------------------
PATHS = dict(allowlist='a', cache='c')


def test_the_forced_values_differ_from_the_frozen_ones_in_the_repair_only():
    import repair
    frozen, forced = he.frozen_settings(), S.forced_settings()
    assert he.is_frozen(frozen) and not he.is_frozen(forced)
    assert sorted(k for k in set(frozen) | set(forced) if he.canonical(frozen.get(k)) != he.canonical(forced.get(k))) == ['name', 'repair', 'smoke_forcing']
    assert {k: v for k, v in forced['repair'].items() if frozen['repair'][k] != v} == dict(max_rounds=1, max_inner=0)
    config = repair.RepairConfig(**forced['repair'])
    assert config.max_rounds == 1 and config.max_inner == 0 and config.wall_seconds == 1200. and config.eps == .05
    assert forced['children'] == frozen['children'] == dict(t88_timeout_seconds=S.T88_TIMEOUT, repair_timeout_seconds=S.REPAIR_TIMEOUT)


class Certificate(SimpleNamespace):
    """A certify_map result with the fields experiments/ef/repair.py reads."""


def fake_certify(passes):
    def certify(u, grid, *, config):
        return Certificate(status='certified_feasible' if passes else 'infeasible', reason='r', min_jd_lower_bound=.5 if passes else -1., n_cells_checked=120, n_infeasible=0 if passes else 3,
                           n_uncertain=0, witness_cell=None, witness_reference_point=None, state_hash=he.array_sha256(u), certificate_version='v')
    return certify


def test_the_forcing_ends_the_real_repair_at_round_cap_with_the_field_unchanged():
    """experiments/ef/repair.repair_field (unchanged) with forced_settings()'s RepairConfig on CPU: no inner step, one round, so round 0 certifies the tapered T88 field itself; it fails -> round_cap, the
    supervisor's repair_not_certified.  A field that already passes would be already_certified: the forcing would not force (smoke_plan.md section 3 item 7)."""
    import repair
    config = repair.RepairConfig(**S.forced_settings()['repair'])
    u = field(6)
    calls = []

    def certify(v, grid, *, config):
        calls.append(he.array_sha256(v))
        return fake_certify(False)(v, grid, config=config)
    result = repair.repair_field(u, GRID, [10, 11], config, certify=certify, certificate_config=None, device='cpu', locate=lambda v, grid, c: np.empty(0, dtype=np.int64))
    assert result.passed is False and result.reason == 'round_cap' and len(result.rounds) == 1
    inner = result.rounds[0]['inner']
    assert inner['iterations'] == 0 and inner['stop'] == 'inner_cap' and not he.hit_wall(dict(reason=result.reason, rounds=result.rounds))
    assert calls == [he.array_sha256(u)]*2 and np.array_equal(result.u, u)                    # the input and round 0 certify the same array
    assert he.repair_decision(dict(status='not_certified', passed=False, reason=result.reason, rounds=result.rounds), None)[1]['class'] == 'repair_not_certified'
    done = repair.repair_field(u, GRID, [10], config, certify=fake_certify(True), certificate_config=None, device='cpu')
    assert done.passed is True and done.reason == 'already_certified'


def test_the_forced_context_checks_the_identity_is_marked_and_writes_only_to_the_smoke_root(work):
    out = work/'he_smoke_t'/'c'/S.SHORT
    ctx = S.forced_context(out, launcher='unit', device='cpu', results=work, paths=PATHS)
    assert ctx['settings_frozen'] is False and ctx['check_identity'] is True and ctx['test_context'] == R.TEST_CONTEXT and ctx['launcher'] == dict(kind='unit')
    assert ctx['pair'] == S.PAIR and ctx['stage_dir'] == str(S.STAGE) and ctx['code_head'] == S.git_commit('HEAD')
    assert R.function_context_problem(ctx) is None and S.forced_context_problem(ctx, results=work) is None and not out.exists()       # make_context writes nothing
    real = S.forced_context(S.RESULTS/'he_smoke_test_never_written'/'d'/S.SHORT, launcher='direct', paths=PATHS)
    assert 'results root' in R.cli_context_problem(real) and not (S.RESULTS/'he_smoke_test_never_written').exists()     # run_hx.py's command line refuses it: hence the forced entries
    for bad, launcher in ((work/'he_smoke_t'/'a'/S.SHORT, 'unit'), (work/'he_v1'/'c'/S.SHORT, 'unit'), (work/'x'/'he_smoke_t'/'c'/S.SHORT, 'unit'),
                          (work/'he_smoke_t'/'c'/S.SHORT, 'direct'), (work/'he_smoke_t'/'d'/S.SHORT, 'unit'), (work/'he_smoke_t'/'c'/'0439-0440', 'unit')):
        assert S.forced_out_problem(bad, launcher, results=work) is not None
        with pytest.raises(ValueError):
            S.forced_context(bad, launcher=launcher, results=work, paths=PATHS)
    frozen = R.make_context(S.STAGE, S.PAIR, out, paths=PATHS, launcher='unit')
    other = S.forced_settings()
    other['repair']['max_rounds'] = 2
    cases = dict(frozen=frozen, unchecked=dict(ctx, check_identity=False), unmarked=dict(ctx, test_context=None), other_pair=dict(ctx, pair=S.core.PAIRS[1]),
                 other_values=dict(ctx, settings=other), other_launcher=dict(ctx, launcher=dict(kind='direct')), elsewhere=dict(ctx, out=str(work/'he_smoke_t'/'b'/S.SHORT)))
    for name, case in cases.items():
        assert S.forced_context_problem(case, results=work) is not None, name


def test_the_forced_entries_refuse_any_other_context_and_write_nothing(work, capsys):
    out = work/'he_smoke_t'/'c'/S.SHORT
    out.mkdir(parents=True)
    ctx = S.forced_context(out, launcher='unit', device='cpu', results=work, paths=PATHS)
    he.write_json(out/'context.json', ctx)                                                   # under tmp, not the results root: the entries refuse it
    for stage in ('t88', 'repair', 'third-cert'):
        assert S.main(['child', stage, '--context', str(out/'context.json')]) == 8
    assert S.main(['evaluate-forced', '--context', str(out/'context.json')]) == 8
    assert 'refused' in capsys.readouterr().err and sorted(p.name for p in out.iterdir()) == ['context.json']
    frozen_out = work/'frozen'/S.SHORT
    frozen_out.mkdir(parents=True)
    he.write_json(frozen_out/'context.json', R.make_context(S.STAGE, S.PAIR, frozen_out, paths=PATHS))
    assert S.main(['child', 't88', '--context', str(frozen_out/'context.json')]) == 8 and sorted(p.name for p in frozen_out.iterdir()) == ['context.json']
    assert S.main(['forced', '--launcher', 'unit', '--out', str(work/'he_smoke_t'/'c'/'new')]) == 2 and not (work/'he_smoke_t'/'c'/'new').exists()
    assert S.main(['check', '--root', str(work/'he_smoke_t')]) == 2


# ---- the check -------------------------------------------------------------------------------------------------------------------------------------
SHA = 'c'*40
GRID = make_grid_3d(nx=7, ny=6, nz=5, hx=1., hy=1., hz=1., unit='mm')


def field(seed):
    u = np.zeros(GRID.shape+(3,))
    u[1:-1, 1:-1, 1:-1] = .01*np.random.default_rng(seed).standard_normal((3, 4, 5, 3))
    return u


def cert_of(u, **extra):
    return dict(status='certified_feasible', reason='positive_enclosure_cover', min_jd_lower_bound=.5, min_margin_lower_bound=.5, n_cells_checked=120, n_infeasible=0, n_uncertain=0, witness_cell=None,
                state_hash=he.array_sha256(u), certificate_version='v', delta_hex='0x0.0p+0', grid_id='g', config=dict(delta='0x0.0p+0', max_depth=48, max_boxes=255, tile_size=65536,
                config_class='PrescreenCertificateConfig', certificate_version='v'), state_hash_matches_array=True, **extra)


def wj(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=he._default))
    return path


def identity_record(sha=SHA, **extra):
    return dict(same=True, problems=[], repo=dict(head=sha, tracked_clean=True, code_paths_clean=True), pinned_head=sha, checked_utc='2026-11-07T01:00:00Z', **extra)


def labels_zero():
    rows = lambda n: dict(per_label=[dict(label=i, fixed_count=0, warped_count=0, dice=None) for i in range(1, n+1)], macro_dice=None, defined_classes=0, classes=list(range(1, n+1)))     # noqa: E731
    return dict(role='r', baseline=dict(seg4=rows(4), seg35=rows(35)), final=dict(seg4=rows(4), seg35=rows(35)))


EVAL_METRICS = dict(seg35=.88, seg4=.92, sdlogj_official=.07, hd95=1.5, dice_official=.88)


RATIOS = [dict(CSF=.8213996780056833, GM=1.3867760887321652), dict(CSF=.7778835787906576, GM=1.512660231173919)]     # the shape of P3's records (passes 1, 2 tissue.ratios_used)
LAST = dict(CSF=.7550035328793033, GM=1.522680394342592)


def passes_of(u0, u1, u2):
    """A T88 run.json's passes as he_c.three_passes writes them (the keys the check reads)."""
    return [dict(pass_index=0, u_sha256=u0, tissue=dict(ratios_used=None, levels=[]), estimate=dict(ratios=RATIOS[0])), dict(pass_index=1, u_sha256=u1, tissue=dict(ratios_used=RATIOS[0]), estimate=dict(ratios=RATIOS[1])),
            dict(pass_index=2, u_sha256=u2, tissue=dict(ratios_used=RATIOS[1]), estimate=dict(ratios=LAST))]


def probe_of(when, peak=6*2**30, oom_kill=0, unit='tag-he-smoke-a.service', memory=12):
    return dict(schema='bayesreg.he.unit_probe.v1', when=when, unit=unit, in_tag_unit=True, peak_bytes=peak, max_bytes=memory*2**30, swap_max_bytes=4*2**30,
                events=dict(low=0, high=0, max=0, oom=0, oom_kill=oom_kill, oom_group_kill=0), unread={}, systemd=dict(Id=unit, MemoryMax=str(memory*2**30), MemorySwapMax=str(4*2**30), OOMPolicy='continue'),
                systemd_unread=None, memory_stat=dict(anon=4*2**30, file=2*2**30), memory_stat_unread=None)


def write_world(work):
    """results/{he_smoke_t, hs_v1, p3_v1, d0_v1} and a stage with the pair's manifest: the records of a smoke in which every item passes.  HE2: the references are HS's T88 field (pass 0) and repair, P3's C
    field (pass 2), ratios and repair; S.REF_PASS0_U_SHA256 and S.REF_PASS2_U_SHA256 are set to this world's arrays (the fixture `refs` restores them)."""
    results, stage = work/'results', work/'stage'
    root = results/'he_smoke_t'
    lam = 2.7936785305661008e-06
    wj(stage/S.PAIR/'manifest.json', dict(pair=S.PAIR, weights=dict(lam=lam, kappa=.002)))
    wj(root/'batch.json', dict(code_sha=SHA, code=str(root/'code'), plan=S.plan_names()))
    for i, s in enumerate(S.STEPS):
        wj(root/'steps'/f'{s.name}.launched.json', dict(step=s.name, argv=S.step_argv(s, root/'code', root), gate_mem_available_gib=S.step_gate(s, S.GATE).mem_available_gib, mem_available_gib_at_launch=11.5,
                                                         uptime_before=100.+10*i))
        wj(root/'steps'/f'{s.name}.done.json', dict(step=s.name, rc=0, uptime_after=105.+10*i))
    pass0_u, t88_u = field(1), field(11)                                                     # pass 0 (HS's field, C off) and pass 2 (P3's field, the one kept)
    pass0, pass1, pass2 = he.array_sha256(pass0_u), 'f'*64, he.array_sha256(t88_u)
    S.REF_PASS0_U_SHA256, S.REF_PASS2_U_SHA256 = pass0, pass2
    hs = results/'hs_v1/runs/SHx24_first5'/S.SHORT
    hs.mkdir(parents=True)
    np.savez(hs/'field_0600.npz', u=pass0_u)
    wj(hs/'run.json', dict(status='done', field=dict(file='field_0600.npz', sha256=S.core.sha_file(hs/'field_0600.npz'), u_sha256=pass0), env=dict(torch='2.5.1', device='GPU'), seconds_total=50.))
    (hs/'cmd.txt').write_text('systemd-run --user --unit=tag-hs-x --property=MemoryMax=8G --property=CPUQuota=800% --setenv=NUMBA_NUM_THREADS=1 python run_hs.py run-one\n')
    hs_r = results/'hs_v1/runs/RSHx24_c600'/S.SHORT
    wj(hs_r/'repair.json', dict(passed=True, rounds=[{}], seconds_total=64., source=dict(field_sha256=S.core.sha_file(hs/'field_0600.npz')), initial_certificate=dict(status='infeasible', n_infeasible=19830)))
    wj(hs_r/'repair_metrics.json', dict(repaired_u_sha256='9'*64, after=dict(metrics=dict(seg35=.8905, seg4=.9275, sdlogj_official=.068))))
    p3 = results/'p3_v1/runs/C_first5'/S.SHORT
    p3.mkdir(parents=True)
    np.savez(p3/'field_0600.npz', u=t88_u)
    p3_field = dict(file='field_0600.npz', sha256=S.core.sha_file(p3/'field_0600.npz'), u_sha256=pass2)
    wj(p3/'run.json', dict(status='done', segmenter_id='unet35_v4_ens', field=p3_field, checkpoints={'600': dict(p3_field)}, passes=passes_of(pass0, pass1, pass2), env=dict(torch='2.5.1', device='GPU'),
                           seconds_total=149., host_peak_rss_gib=6.36, device_peak_reserved_gib=6.98))
    (p3/'cmd.txt').write_text('systemd-run --user --unit=tag-p3-x --property=MemoryMax=8G --property=CPUQuota=800% --setenv=NUMBA_NUM_THREADS=1 python run_p2.py p3 run-one\n')
    p3_r = results/'p3_v1/runs/RC_c600'/S.SHORT
    wj(p3_r/'repair.json', dict(passed=True, rounds=[{}, {}], seconds_total=171., source=dict(field_sha256=p3_field['sha256']), initial_certificate=dict(status='infeasible', n_infeasible=21462, state_hash='6'*64)))
    wj(p3_r/'repair_metrics.json', dict(repaired_u_sha256='5'*64, after=dict(metrics=dict(seg35=.8905, seg4=.9279, sdlogj_official=.069))))
    d0 = results/'d0_v1/runs/A0'/S.PAIR
    wj(d0/'protocol.json', dict(task=dict(name='task')))
    wj(d0/'classification.json', {'class': 'ok', 'evidence': dict(whole_wall_seconds=102.)})
    (d0/'task').mkdir()
    np.savez(d0/'task/final_native.npz', u=field(9))
    wj(d0/'task/final_native.json', dict(sha256='8'*64, role='formal_native_outer_commit', certificate=dict(state_hash='1'*64)))
    forced_sha = he.settings_sha256(S.forced_settings())
    for arm in S.ARMS:
        d = S.run_dir(root, arm)
        forced = arm in ('c', 'd')
        ctx = dict(schema=he.SCHEMA.format('context'), pair=S.PAIR, stage_dir=str(stage), out=str(d), workers=7, check_identity=True, code_head=SHA, paths=dict(allowlist='x'),
                   settings=S.forced_settings() if forced else he.frozen_settings(), settings_frozen=not forced, settings_sha256=forced_sha if forced else he.settings_sha256(he.frozen_settings()),
                   test_context=R.TEST_CONTEXT if forced else None, launcher=dict(kind=S.FORCED_ARMS.get(arm, 'unit')))
        wj(d/'context.json', ctx)
        wj(d/'started.json', {})
        for name in ('identity.json', 'identity_t88.json', 'identity_repair.json', 'identity_end.json'):
            wj(d/name, identity_record())
        (d/'t88').mkdir(parents=True)
        np.savez(d/'t88'/he.FIELD_FILE, u=t88_u)
        wj(d/'t88/run.json', dict(status='done', tissue_prior='moment3', field=dict(file=he.FIELD_FILE, u_sha256=pass2), passes=passes_of(pass0, pass1, pass2), seconds_total=150., seconds_tissue=3.,
                                  env=dict(torch='2.5.1', device='GPU', device_peak_reserved_gib=7., device_peak_allocated_gib=6., host_peak_rss_gib=6.4)))
        wj(d/'repair/repair.json', dict(passed=not forced, reason='round_cap' if forced else 'certified', rounds=[{}] if forced else [{}, {}], initial_certificate=dict(status='infeasible', n_infeasible=21462, state_hash='6'*64),
                                        env=dict(device_peak_reserved_gib=7.5, device_peak_allocated_gib=6.5, host_peak_rss_gib=1.8)))
        u = field(2 if not forced else 3)
        cert = cert_of(u)
        unit = he.unit_record(probe_of('start', unit=f'tag-he-smoke-{arm}.service'), probe_of('end', unit=f'tag-he-smoke-{arm}.service'), he.SUPERVISOR_UNIT)
        line = dict(line_status=he.HX, failure=None, fallback=None, settings_frozen=not forced, seconds_total=300., unit=unit)
        if forced:
            line.update(line_status=he.FALLBACK_C, failure=he.failure('repair_not_certified', 'round_cap'),
                        fallback=dict(kind='configC', file=fc.FINAL, final_native_sha256='7'*64, state_hash=he.array_sha256(u), certificate=cert, configC_classification={'class': 'ok'},
                                      firewall=dict(gate=dict(passed=True), lam_hex=float(lam).hex()), prepare=dict(seconds=20.), candidates=[],
                                      launch=dict(kind=S.FORCED_ARMS[arm], dmesg_read=False, dxg_lines=None, oom_lines=None, seconds=120.)))
            runp = d/'fallback_configc/run'
            wj(runp/'prepared_check.json', dict(ok=True))
            wj(runp/'classification.json', {'class': 'ok', 'evidence': dict(whole_wall_seconds=130., dxg_lines=0, oom_lines=0)})
            (runp/'exit_code').write_text('0\n')
            wj(runp/'protocol.json', dict(task=dict(name='task')))
            wj(runp/'task/labels.json', labels_zero())
            wj(runp/'task/smoothness.json', dict(final=dict(n_brain_cells=0, sdlogj_brain=None, pct_J_lt_half_brain=None, pct_J_gt_two_brain=None, sdlogj=.1)))
            wj(d/'fallback_configc/derived/manifest.json', {})
            nested = 'tag-he-c-0438-0439.service' if arm == 'c' else f'tag-he-smoke-{arm}.service'
            for name, when in zip(fc.PROBE_FILES, ('start', 'end')):
                wj(runp/name, probe_of(when, peak=5*2**30, unit=nested))
        he.write_output(d/'output', u, GRID, cert, line)
        if arm == 'b':
            wj(d/'spy.json', dict(passed=True, positive_condition=True, run_rc=0, breach=None, loader_calls=['bayesreg.io.pair.load_pair']*3, reads_seen=dict(fixed=3, moving=3),
                                  reads_at_the_class=dict(fixed=3, moving=3), forbidden_access=[], evaluation_records_opened=[]))
        if arm in ('a', 'c', 'd'):
            wj(d/'identity_eval_0.json', identity_record())
            wj(d/R.TRUTH_MARKER, dict(state_hash=cert['state_hash']))
            wj(d/'evaluation.json', dict(metrics=EVAL_METRICS, per_label=dict(seg35_per_label={'1': .9}), official_row=dict(dice=.88), order=S.ORDER_RULE+'; the truth once',
                                         review_certificate=dict(consistent=True, differences={}, review=cert),
                                         unit=he.unit_record(probe_of('start', unit=f'tag-he-smoke-{arm}-eval.service', memory=8), probe_of('end', peak=2**30, unit=f'tag-he-smoke-{arm}-eval.service', memory=8),
                                                             he.EVALUATION_UNIT)))
            wj(d/'identity_eval_end_0.json', identity_record())
            ns = time.time_ns()-10**12                                                       # in the past: a later edit of a record makes it newer, as in a real run
            for k, name in enumerate(('identity_eval_0.json', R.TRUTH_MARKER, 'evaluation.json')):
                os.utime(d/name, ns=(ns+k, ns+k))
    return root, results, stage


def echo(ctx, out_dir):
    ev = json.loads((Path(out_dir)/'evaluation.json').read_text())
    return dict(metrics=ev['metrics'], per_label=ev['per_label'], official_row=ev['official_row'], d_sha256='d')


def run_check(root, results, stage, *, recompute=echo, dmesg=lambda: None):
    return S.check(root, results=results, stage=stage, recompute=recompute, dmesg=dmesg, measure=lambda: [1., 1.])


def verdicts(report):
    return {(i['item'], i['name']): i['verdict'] for i in report['items']}


@pytest.fixture(autouse=True)
def refs(monkeypatch):
    """write_world sets the contract's reference hashes to its own arrays; they are restored after each test."""
    monkeypatch.setattr(S, 'REF_PASS0_U_SHA256', S.REF_PASS0_U_SHA256)
    monkeypatch.setattr(S, 'REF_PASS2_U_SHA256', S.REF_PASS2_U_SHA256)


def test_the_contracts_reference_hashes():
    """C22.9-4 (xv): hs_v1's field of the pair, u_sha256 b6f061da...; C22.B4-6 (b): p3_v1's, checkpoints."600".u_sha256 edf3b039... (docs/bayesreg/01_合同.md, written out there)."""
    assert S.REF_PASS0_U_SHA256 == 'b6f061daa0eb8ce2131abea9218e2010a9f9e7a1057f91ec25eb227809adde2f'
    assert S.REF_PASS2_U_SHA256 == 'edf3b0392269ff347ae486d5f408abaa36a03f867fc4e25af6f6bf3f523dcf04'
    text = (ROOT/'docs/bayesreg/01_合同.md').read_text()
    assert S.REF_PASS0_U_SHA256 in text and S.REF_PASS2_U_SHA256 in text and S.DEFAULT_ROOT.name == 'he_smoke_v2'
    import fx_core
    assert S.NATIVE == fx_core.NATIVE == 600


def test_the_check_passes_a_world_in_which_every_item_holds_and_writes_its_report_once(work):
    root, results, stage = write_world(work)
    before = {str(p): p.stat().st_mtime_ns for p in root.rglob('*') if p.is_file()}
    report = run_check(root, results, stage)
    got = {n: v for (n, _), v in verdicts(report).items() if n != 5}
    assert got == {1: S.PASS, 2: S.PASS, 3: S.PASS, 4: S.PASS, 6: S.DESCRIBE, 7: S.PASS, 8: S.DESCRIBE, 9: S.DESCRIBE, 10: S.PASS, 11: S.PASS, 12: S.DESCRIBE, 13: S.PASS}, \
        json.dumps(report['items'], ensure_ascii=False, default=str)[:3000]
    assert [i['verdict'] for i in report['items'] if i['item'] == 5] == [S.PASS, S.PASS]
    assert all(i['criterion'] and 'values' in i for i in report['items'])
    assert [i['values'].get('contract') for i in report['items'] if i['item'] in (3, 10, 11, 12, 13)] == ['C22.B4-6 (a)', 'C22.B4-6 (b)', 'C22.B4-6 (c)', 'C22.B4-6 (d)', 'C22.B4-6 (e)']
    after = {str(p): p.stat().st_mtime_ns for p in root.rglob('*') if p.is_file() and not p.name.startswith(S.REPORT)}
    assert after == before                                                                   # read only on the outputs
    assert json.loads((root/'smoke_report.json').read_text())['summary'] == report['summary'] and '| 4 | repair_and_evaluation | 过 |' in (root/'smoke_report.md').read_text()
    run_check(root, results, stage)
    assert (root/'smoke_report_1.json').is_file() and (root/'smoke_report_1.md').is_file() and json.loads((root/'smoke_report.json').read_text())['utc'] == report['utc']
    i4 = next(i for i in report['items'] if i['item'] == 4)['values']
    assert i4['seg35'] == .88 and i4['seg4'] == .92 and abs(i4['described']['seg35_minus_p3']-(.88-.8905)) < 1e-12 and abs(i4['described']['seg4_minus_hs']-(.92-.9275)) < 1e-12 and i4['n_rounds'] == 2
    i12 = next(i for i in report['items'] if i['item'] == 12)['values']
    assert i12['arms']['c']['frozen_run']['unit_probe_end.json']['unit'] == 'tag-he-c-0438-0439.service' and i12['arms']['a']['supervisor_end']['peak_gib'] == 6. and i12['arms']['a']['evaluation_end']['peak_gib'] == 1.
    assert i12['arms']['a']['supervisor_end']['anon_gib'] == 4. and i12['arms']['a']['supervisor_end']['file_gib'] == 2.                  # HE2 clean: memory.stat, described
    i7 = next(i for i in report['items'] if i['item'] == 7)['values']
    assert i7['systemd_read_for'] == list(S.ARMS) and i7['gates'] == {s.name: (10. if s.name != 'check' else 7.) for s in S.STEPS}
    assert i7['described']['mem_available_gib_at_launch'] == {s.name: 11.5 for s in S.STEPS}                                    # HE2 clean: described, never judged


def c_off_world(work):
    """HB (C22 卷首修订 8, D85): the same world with C off: every arm's T88 record says tissue_prior off and keeps HS's field (no passes)."""
    root, results, stage = write_world(work)
    hs = results/'hs_v1/runs/SHx24_first5'/S.SHORT
    for arm in S.ARMS:
        t = S.run_dir(root, arm)/'t88'
        (t/he.FIELD_FILE).unlink()
        shutil.copyfile(hs/'field_0600.npz', t/he.FIELD_FILE)
        rec = json.loads((t/'run.json').read_text())
        rec.update(tissue_prior='off', passes=None, field=dict(file=he.FIELD_FILE, u_sha256=S.REF_PASS0_U_SHA256))
        (t/'run.json').write_text(json.dumps(rec))
    return root, results, stage


def test_with_c_off_item_3_is_the_field_against_hss_and_items_10_11_do_not_apply(work):
    """HB: the smoke's judgment of the method with C off (C22 卷首修订 8, D85 (a)): item 3 the kept T88 field bit for bit against hs_v1's (b6f061da… in the contract), items 10 and 11 only described (no
    pass 2, no ratios), item 13 on HS's repair (the forcing's precondition with C off).  A field one ulp away is 不过（待查）; a record still naming passes is 不过."""
    root, results, stage = c_off_world(work)
    report = run_check(root, results, stage)
    got = verdicts(report)
    assert got[(3, 't88_field_c_off')] == S.PASS and got[(10, 't88_pass2')] == got[(11, 'c_ratios')] == S.DESCRIBE and got[(13, 'forcing_precondition_c_off')] == S.PASS, report['summary']
    assert report['tissue_prior'] == 'off' and 'Hx24-RF05-F1200-v4；' in report['wording'] and S.c_off(root)
    t = S.run_dir(root, 'a')/'t88'
    u = npz_u_of(t/he.FIELD_FILE)
    u.flat[7] = np.nextafter(u.flat[7], 1.)
    np.savez(t/he.FIELD_FILE, u=u)
    rec = json.loads((t/'run.json').read_text())
    rec['field']['u_sha256'] = he.array_sha256(u)
    (t/'run.json').write_text(json.dumps(rec))
    assert S.item3_field_off(root, results)['verdict'] == S.OPEN
    rec['passes'] = []
    (t/'run.json').write_text(json.dumps(rec))
    assert S.item3_field_off(root, results)['verdict'] == S.FAIL


def npz_u_of(path):
    with np.load(path) as z:
        return np.array(z['u'])


def mutate(path, change):
    path = Path(path)
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value, default=he._default))


def one_ulp(path):
    u = S.npz_u(path)
    u[2, 2, 2, 0] = np.nextafter(u[2, 2, 2, 0], 1.)
    np.savez(path, u=u)


def other_field(root, results, stage):
    """a's kept T88 field (pass 2) one ulp off P3's, its own record following it (the reference still the field its record names): item 10 不过（待查）."""
    path = S.run_dir(root, 'a')/'t88'/he.FIELD_FILE
    one_ulp(path)
    u = he.array_sha256(S.npz_u(path))
    mutate(path.parent/'run.json', lambda v: (v['field'].update(u_sha256=u), v['passes'][2].update(u_sha256=u)))


def other_reference(root, results, stage):
    """HS's field replaced by another array than the one hs_v1's run.json names: item 3 不过 (revision 1, review minor 4)."""
    one_ulp(results/'hs_v1/runs/SHx24_first5'/S.SHORT/'field_0600.npz')


def other_p3_reference(root, results, stage):
    one_ulp(results/'p3_v1/runs/C_first5'/S.SHORT/'field_0600.npz')


def not_the_contracts(root, results, stage):
    """The reference is the field its record names, but not the contract's b6f061da...: 不过."""
    S.REF_PASS0_U_SHA256 = '0'*64


def step_argv_drop(arm, old):
    return lambda root, results, stage: mutate(root/'steps'/f'{arm}.launched.json', lambda v: v.update(argv=[a for a in v['argv'] if a != old]))


def failing_certificate(root, results, stage):
    """cert.json and the review both say the certificate did not pass (they still agree): only the certificate check of item 4 flips."""
    d = S.run_dir(root, 'a')
    mutate(d/'output/cert.json', lambda v: v.update(min_jd_lower_bound=0.))
    mutate(d/'evaluation.json', lambda v: v['review_certificate']['review'].update(min_jd_lower_bound=0.))


def late_marker(root, results, stage):
    d = S.run_dir(root, 'a')
    t = (d/'identity_eval_0.json').stat().st_mtime_ns
    os.utime(d/R.TRUTH_MARKER, ns=(t-5, t-5))


def drop(rel):
    return lambda root, results, stage: (S.run_dir(root, rel[0])/rel[1]).unlink()


def edit(arm, rel, change):
    return lambda root, results, stage: mutate(S.run_dir(root, arm)/rel, change)


def edit_line(arm, change):
    def f(root, results, stage):
        o = S.run_dir(root, arm)/'output/line.json'
        mutate(o, change)
    return f


def step_argv_edit(arm, old, new):
    return lambda root, results, stage: mutate(root/'steps'/f'{arm}.launched.json', lambda v: v.update(argv=[new if a == old else a for a in v['argv']]))


COUNTEREXAMPLES = dict(
    head=(1, edit('c', 'identity_t88.json', lambda v: v['repo'].update(head='0'*40)), S.FAIL),
    dirty=(1, edit('a', 'identity_end.json', lambda v: v['repo'].update(code_paths_clean=False)), S.FAIL),
    no_identity=(1, drop(('b', 'identity.json')), S.FAIL),
    frozen_c=(1, edit('c', 'context.json', lambda v: v.update(settings_frozen=True)), S.FAIL),
    reads=(2, edit('b', 'spy.json', lambda v: v['reads_seen'].update(fixed=2)), S.FAIL),
    records=(2, edit('b', 'spy.json', lambda v: v.update(evaluation_records_opened=['x/metrics_0600.json'])), S.FAIL),
    line_b=(2, edit_line('b', lambda v: v.update(line_status=he.FALLBACK_C)), S.FAIL),
    pass0=(3, edit('a', 't88/run.json', lambda v: v['passes'][0].update(u_sha256='0'*64)), S.OPEN),
    reference=(3, other_reference, S.FAIL),
    contract0=(3, not_the_contracts, S.FAIL),
    no_pass0=(3, edit('a', 't88/run.json', lambda v: v.update(passes=[])), S.FAIL),
    pass2=(10, other_field, S.OPEN),
    reference2=(10, other_p3_reference, S.FAIL),
    inconsistent=(10, edit('a', 't88/run.json', lambda v: v['passes'][2].update(u_sha256='0'*64)), S.FAIL),
    no_t88=(10, drop(('a', 't88/field_0600.npz')), S.FAIL),
    ratios=(11, edit('a', 't88/run.json', lambda v: v['passes'][2]['tissue']['ratios_used'].update(CSF=.7778835787906577)), S.OPEN),
    no_ratios=(11, edit('a', 't88/run.json', lambda v: v['passes'][1]['tissue'].update(ratios_used=None)), S.FAIL),
    oom_supervisor=(12, edit_line('b', lambda v: v['unit']['end']['events'].update(oom_kill=1)), S.FAIL),
    oom_frozen=(12, edit('c', 'fallback_configc/run/unit_probe_end.json', lambda v: v['events'].update(oom_kill=2)), S.FAIL),
    oom_evaluation=(12, edit('d', 'evaluation.json', lambda v: v['unit']['end']['events'].update(oom_kill=1)), S.FAIL),
    no_unit=(12, edit_line('a', lambda v: v.pop('unit')), S.FAIL),
    forcing_reference=(13, lambda root, results, stage: mutate(results/'p3_v1/runs/RC_c600'/S.SHORT/'repair.json', lambda v: v['initial_certificate'].update(status='certified_feasible')), S.FAIL),
    forcing_c=(13, edit('c', 'repair/repair.json', lambda v: v.update(initial_certificate=cert_of(field(5)))), S.FAIL),
    forcing_reason=(13, edit('d', 'repair/repair.json', lambda v: v.update(reason='already_certified')), S.FAIL),
    certificate=(4, failing_certificate, S.FAIL),
    fallback_a=(4, edit_line('a', lambda v: v.update(line_status=he.FALLBACK_C)), S.FAIL),
    review=(4, edit('a', 'evaluation.json', lambda v: v['review_certificate']['review'].update(min_jd_lower_bound=.4)), S.FAIL),
    order=(4, late_marker, S.FAIL),
    seg4=(4, edit('a', 'evaluation.json', lambda v: v['metrics'].pop('seg4')), S.FAIL),
    repair_a=(4, edit('a', 'repair/repair.json', lambda v: v.update(passed=False)), S.FAIL),
    lam=(5, edit_line('c', lambda v: v['fallback']['firewall'].update(lam_hex=float(2.79e-06).hex())), S.FAIL),
    dice=(5, edit('d', 'fallback_configc/run/task/labels.json', lambda v: v['final']['seg35'].update(macro_dice=.1)), S.FAIL),
    brain=(5, edit('c', 'fallback_configc/run/task/smoothness.json', lambda v: v['final'].update(sdlogj_brain=.1)), S.FAIL),
    watchdog=(5, edit('d', 'fallback_configc/run/classification.json', lambda v: v['evidence'].update(whole_wall_seconds=301.)), S.FAIL),
    gate=(5, edit_line('c', lambda v: v['fallback']['firewall']['gate'].update(passed=False)), S.FAIL),
    launcher=(5, edit_line('d', lambda v: v['fallback']['launch'].update(kind='unit')), S.FAIL),
    not_forced=(5, edit('c', 'repair/repair.json', lambda v: v.update(passed=True)), S.FAIL),
    prepared=(5, edit('d', 'fallback_configc/run/prepared_check.json', lambda v: v.update(ok=False)), S.FAIL),
    class_ok=(5, edit_line('c', lambda v: v['fallback']['configC_classification'].update({'class': 'unsure'})), S.FAIL),
    certified_again=(5, edit_line('d', lambda v: v['fallback']['certificate'].update(n_uncertain=1)), S.FAIL),
    forced_context=(5, edit('c', 'context.json', lambda v: v['settings']['repair'].update(max_rounds=2)), S.FAIL),
    peak=(6, edit('b', 'repair/repair.json', lambda v: v['env'].update(device_peak_reserved_gib=9.5)), S.FAIL),
    gpu_oom=(6, edit('b', 't88/run.json', lambda v: v.update(status='cuda_oom')), S.FAIL),
    repair_oom=(6, edit('b', 'repair/repair.json', lambda v: v.update(error='OutOfMemoryError: CUDA out of memory')), S.FAIL),
    runtime=(7, step_argv_edit('c', '--property=RuntimeMaxSec=4800', '--property=RuntimeMaxSec=3600'), S.FAIL),
    memory=(7, step_argv_edit('d', '--property=MemoryMax=12G', '--property=MemoryMax=10G'), S.FAIL),
    swap=(7, step_argv_edit('d', '--property=MemorySwapMax=4G', '--property=MemorySwapMax=2G'), S.FAIL),
    oom_policy=(7, step_argv_drop('b', '--property=OOMPolicy=continue'), S.FAIL),
    oom_stop=(7, step_argv_edit('a', '--property=OOMPolicy=continue', '--property=OOMPolicy=stop'), S.FAIL),
    gate_d=(7, lambda root, results, stage: mutate(root/'steps/d.launched.json', lambda v: v.update(gate_mem_available_gib=7.)), S.FAIL),
    gate_c=(7, lambda root, results, stage: mutate(root/'steps/c.launched.json', lambda v: v.update(gate_mem_available_gib=7.)), S.FAIL),           # HE2: the gate of c
    gate_eval=(7, lambda root, results, stage: mutate(root/'steps/a-eval.launched.json', lambda v: v.update(gate_mem_available_gib=7.)), S.FAIL),     # HE2, H3
    systemd_stop=(7, edit_line('a', lambda v: v['unit']['matches_at_start'].update(OOMPolicy=False)), S.FAIL),                                        # HE2, H2: systemd applied another policy
    kernel_max=(7, edit_line('c', lambda v: v['unit']['kernel_matches_at_start'].update({'memory.max': False})), S.FAIL),                            # HE2 revision 1 (review minor 9)
    kernel_swap=(7, edit_line('d', lambda v: v['unit']['kernel_matches_at_start'].update({'memory.swap.max': False})), S.FAIL),
    eval_oom_policy=(7, step_argv_drop('c-eval', '--property=OOMPolicy=continue'), S.FAIL),                                                            # HE2 revision 1 (review minor 14)
    eval_reading=(7, edit('d', 'evaluation.json', lambda v: v['unit']['kernel_matches_at_start'].update({'memory.max': False})), S.FAIL),
    launch=(8, edit_line('c', lambda v: v['fallback'].pop('launch')), S.FAIL),
    timing=(9, lambda root, results, stage: [p.unlink() for p in S.run_dir(root, 'd').glob('identity*.json')], S.FAIL),
)


MISSING = ('no_t88', 'no_pass0', 'no_ratios', 'no_unit', 'launch', 'timing')                 # the counterexamples that remove a record the item reads
FLIPS = dict(fallback_a='line_status', review='review_consistent', order='order', seg4='seg35_and_seg4', repair_a='repair_passed', lam='lam', dice='labels_none', brain='labels_none',
             watchdog='watchdog', gate='gate_one', launcher='launcher', not_forced='repair_failed', prepared='check_prepared', certificate='certificate', class_ok='class_ok',
             certified_again='certified_again', forced_context='forced_context')     # items 4, 5: the one check each counterexample must flip


@pytest.mark.parametrize('case', sorted(COUNTEREXAMPLES))
def test_every_item_has_a_counterexample(work, case):
    n, change, want = COUNTEREXAMPLES[case]
    root, results, stage = write_world(work)
    change(root, results, stage)
    got = [i for i in run_check(root, results, stage)['items'] if i['item'] == n]
    bad = [i for i in got if i['verdict'] == want]
    assert len(bad) == 1, json.dumps(got, ensure_ascii=False, default=str)[:2000]
    if n == 5:
        assert sorted(i['verdict'] for i in got) == sorted([want, S.PASS])                # the other arm still passes
    values = bad[0]['values']
    assert ('missing' in values) == (case in MISSING)                                          # it fails for its reason, not because a record went away
    if case in FLIPS:
        assert {k for k, ok in values['checks'].items() if not ok} == {FLIPS[case]}, values['checks']


def test_the_t88_items_record_both_sides_without_a_reason(work):
    """Item 3 (C22.B4-6 (a), pass 0 against HS) and item 10 ((b), pass 2 against P3): a difference is 不过（待查） with both sides recorded; a reference that is not its record's field is 不过."""
    root, results, stage = write_world(work)
    edit('a', 't88/run.json', lambda v: v['passes'][0].update(u_sha256='0'*64))(root, results, stage)
    i3 = next(i for i in run_check(root, results, stage)['items'] if i['item'] == 3)
    v = i3['values']
    assert i3['verdict'] == S.OPEN and v['hashes_equal'] is False and v['ours']['u_sha256'] == '0'*64 != v['reference']['u_sha256'] and len(v['reference']['file_sha256']) == 64 and v['ours']['n_passes'] == 3
    assert v['ours']['workers'] == 7 and v['reference']['workers'] is None and 'no workers' in v['reference']['workers_note'] and v['ours']['env']['torch'] == '2.5.1' and v['reference']['env']['device'] == 'GPU'
    assert v['ours']['unit']['values'] == dict(NUMBA_NUM_THREADS='8', CPUQuota='1200%', MemoryMax='12G') and v['reference']['unit']['values'] == dict(NUMBA_NUM_THREADS='1', CPUQuota='800%', MemoryMax='8G')
    assert v['reference_is_the_recorded_field'] is True and v['reference']['contract_u_sha256'] == S.REF_PASS0_U_SHA256 and v['described']['b_c_d_against_a']['b']['same_as_a'] is False
    other_reference(root, results, stage)
    i3 = next(i for i in run_check(root, results, stage)['items'] if i['item'] == 3)
    assert i3['verdict'] == S.FAIL and i3['values']['reference_is_the_recorded_field'] is False and '参照' in i3['note']
    (results/'hs_v1/runs/SHx24_first5'/S.SHORT/'cmd.txt').unlink()
    assert next(i for i in run_check(root, results, stage)['items'] if i['item'] == 3)['values']['reference']['unit']['values'] is None
    root2, results2, stage2 = write_world(work/'second')
    other_field(root2, results2, stage2)
    i10 = next(i for i in run_check(root2, results2, stage2)['items'] if i['item'] == 10)
    v = i10['values']
    assert i10['verdict'] == S.OPEN and v['arrays_equal'] is False and v['ours_consistent'] is True and v['reference_is_the_recorded_field'] is True and v['ours']['u_sha256'] != v['reference']['u_sha256']
    assert v['described']['b_c_d_against_a']['c']['same_as_a'] is False and v['reference']['unit']['values'] == dict(NUMBA_NUM_THREADS='1', CPUQuota='800%', MemoryMax='8G')


def test_the_reference_records_must_name_the_contracts_hashes(work):
    """reference_records (the dry run, the queue's input check): hs_v1's field.u_sha256 the contract's pass 0, p3_v1's checkpoint and field the contract's pass 2, P3's pass 0 HS's field, two ratios."""
    root, results, stage = write_world(work)
    r = S.reference_records(results, stage)
    assert r['t88']['u_sha256'] == S.REF_PASS0_U_SHA256 and r['p3']['u_sha256'] == S.REF_PASS2_U_SHA256 and r['p3']['ratios_used'] == RATIOS and r['p3']['last_estimate'] == LAST
    assert r['p3_repair']['initial_certificate']['status'] == 'infeasible' and r['hs']['values']['seg4'] == .9275 and r['p3_repair']['values']['seg35'] == .8905
    for change in (lambda: mutate(results/'p3_v1/runs/C_first5'/S.SHORT/'run.json', lambda v: v['checkpoints']['600'].update(u_sha256='1'*64)),
                   lambda: mutate(results/'p3_v1/runs/C_first5'/S.SHORT/'run.json', lambda v: v['passes'][0].update(u_sha256='1'*64)),
                   lambda: mutate(results/'p3_v1/runs/C_first5'/S.SHORT/'run.json', lambda v: v['passes'][2]['tissue'].update(ratios_used=None)),
                   lambda: mutate(results/'hs_v1/runs/RSHx24_c600'/S.SHORT/'repair.json', lambda v: v['source'].update(field_sha256='2'*64))):
        root, results, stage = write_world(work/f'w{len(list(work.iterdir()))}')
        change()
        with pytest.raises(ValueError, match='reference records'):
            S.reference_records(results, stage)
    root, results, stage = write_world(work/'last')
    S.REF_PASS2_U_SHA256 = '3'*64
    with pytest.raises(ValueError, match='not the contract'):
        S.reference_records(results, stage)
    assert S.estimates(results) is None


def test_the_queues_input_check_asks_what_the_dry_run_asks(work, monkeypatch):
    """HE2 review minor 10: reference_problems (the anchor; hs_v1's field done and the one its repair read; P3 a done field of the anchor's segmenter; P3's field not passing before its repair) is asked by
    the queue's input check as by the dry run."""
    monkeypatch.setattr(S, 'admission_problems', lambda *a, **k: [])
    root, results, stage = write_world(work/'a')
    assert S.reference_problems(S.reference_records(results, stage)) == [] and S.input_problems(results, stage) == []
    for name, change, word in (('b', lambda r: mutate(r/'p3_v1/runs/RC_c600'/S.SHORT/'repair.json', lambda v: v['initial_certificate'].update(status='certified_feasible')), 'item 13'),
                               ('c', lambda r: mutate(r/'p3_v1/runs/C_first5'/S.SHORT/'run.json', lambda v: v.update(segmenter_id='unet35_v1_ens')), 'item 10'),
                               ('d', lambda r: mutate(r/'hs_v1/runs/SHx24_first5'/S.SHORT/'run.json', lambda v: v.update(status='time_cap')), 'item 3')):
        root, results, stage = write_world(work/name)
        change(results)
        problems = S.input_problems(results, stage)
        assert len(problems) == 1 and problems[0].startswith(word), problems
    monkeypatch.setattr(he, 'segmenter_start_problems', lambda settings, **kw: (['the registry differs'], {}))
    assert S.reference_problems(S.reference_records(results, stage), anchor=False) == ['item 3: the reference field records disagree']
    assert 'anchor: the registry differs' in S.input_problems(results, stage)


def test_a_forcing_that_did_not_force_is_judged_on_its_checks_not_as_missing(work):
    """Revision 1, review minor 2: the T88 field already passes, the repair is already_certified, the line hx_certified with no fallback: item 5 不过 with repair_failed false and the reason in the report."""
    root, results, stage = write_world(work)
    edit('c', 'repair/repair.json', lambda v: v.update(passed=True, reason='already_certified', rounds=[]))(root, results, stage)
    edit_line('c', lambda v: v.update(line_status=he.HX, failure=None, fallback=None))(root, results, stage)
    i5 = [i for i in run_check(root, results, stage)['items'] if i['item'] == 5]
    c = next(i for i in i5 if i['name'].endswith('_c'))
    assert c['verdict'] == S.FAIL and 'missing' not in c['values'] and c['values']['fallback_present'] is False and c['values']['repair_reason'] == 'already_certified'
    assert c['values']['checks']['repair_failed'] is False and c['values']['checks']['forced_context'] is True and next(i for i in i5 if i['name'].endswith('_d'))['verdict'] == S.PASS


def test_a_recomputation_that_differs_or_fails_fails_item_four(work):
    root, results, stage = write_world(work)

    def changed(ctx, out_dir):
        x = echo(ctx, out_dir)
        x['metrics'] = dict(x['metrics'], seg35=.8800000000000001)
        return x

    def broken(ctx, out_dir):
        raise ValueError('no truth here')
    for fn in (changed, broken):
        i4 = next(i for i in run_check(root, results, stage, recompute=fn)['items'] if i['item'] == 4)
        assert i4['verdict'] == S.FAIL and i4['values']['checks']['recomputed_equal'] is False


def test_the_device_item_reads_dmesg_in_the_window_of_the_spy_unit(work):
    root, results, stage = write_world(work)
    window = json.loads((root/'steps/b.launched.json').read_text())['uptime_before']+1.
    outside = '[   10.000000] dxgk: create_allocation failed\n'
    inside = f'[{window:12.6f}] dxgk: create_allocation failed\n'
    assert next(i for i in run_check(root, results, stage, dmesg=lambda: outside)['items'] if i['item'] == 6)['verdict'] == S.DESCRIBE
    i6 = next(i for i in run_check(root, results, stage, dmesg=lambda: outside+inside)['items'] if i['item'] == 6)
    assert i6['verdict'] == S.FAIL and i6['values']['dxg_lines'] == 1 and i6['values']['dmesg_read'] is True
    assert S.dmesg_window(None, 1., 2.) == (None, None) and S.dmesg_window('[ 1.5] Out of memory: Killed process 1\n', 1., 2.) == (0, 1)


def test_the_unit_bound_and_the_frozen_unit_memory():
    assert S.T88_TIMEOUT == he.frozen_settings()['children']['t88_timeout_seconds'] and S.REPAIR_TIMEOUT == he.frozen_settings()['children']['repair_timeout_seconds']
    bound = S.T88_TIMEOUT+S.REPAIR_TIMEOUT+fc.WATCHDOG_SECONDS+fc.KILL_GRACE+S.FULL_CERTIFICATES*S.CERTIFICATE_SECONDS+S.FALLBACK_SETUP_SECONDS
    assert S.SUPERVISOR['runtime'] > bound and S._gib('512M') == .5 and S._gib('x') is None
    assert S.frozen_unit_gib('MemoryMax') == 12. == S._gib(S.SUPERVISOR['memory_max']) and S.frozen_unit_gib('MemorySwapMax') == 4. and S.SUPERVISOR['oom_policy'] == S.OOM_POLICY == 'continue'     # revision 1: D0's
    steps = {s.name: s for s in S.STEPS}
    assert S.step_gate(steps['d'], S.GATE).mem_available_gib == S.D_MEM_AVAILABLE_GIB == 10. == he.FROZEN_RUN_GATE_GIB and S.GATE.mem_available_gib == he.OTHER_GATE_GIB == 7.
    assert [s.name for s in S.STEPS if S.step_gate(s, S.GATE).mem_available_gib == 10.] == ['a', 'a-eval', 'b', 'c', 'c-eval', 'd', 'd-eval'] and S.step_gate(steps['check'], S.GATE) == S.GATE   # HE2: the gate of c
    assert '--property=MemoryMax=12G' in fc.UNIT_RESOURCES and S.unit_property(['--property=MemorySwapMax=4G', '--property=MemoryMax=14G'], 'MemoryMax') == '14G'
    assert S.SUPERVISOR == he.SUPERVISOR_UNIT == he.frozen_settings()['units']['supervisor'] and S.EVALUATION == he.EVALUATION_UNIT == he.frozen_settings()['units']['evaluation']     # HE2 (H2): the production values


def synthetic_pair(stage):
    """A pair of the real allowlist's development subjects with synthetic arrays and truth members (no file of those subjects is read)."""
    from jd_curl.three_d.nifti import file_sha256
    d = stage/S.PAIR
    d.mkdir(parents=True)
    rng = np.random.default_rng(5)
    seg35 = rng.integers(0, 36, GRID.shape).astype(np.int16)
    arrays = dict(fixed=rng.uniform(size=GRID.shape), moving=rng.uniform(size=GRID.shape), fixed_seg35=seg35, moving_seg35=np.roll(seg35, 1, axis=2), fixed_seg4=(seg35 % 5).astype(np.int16),
                  moving_seg4=(np.roll(seg35, 1, axis=2) % 5).astype(np.int16), initial_u=np.zeros(GRID.shape+(3,)))
    np.savez_compressed(d/'pair.npz', **arrays)
    fixed, moving = S.PAIR.split('__')
    wj(d/'manifest.json', dict(pair=S.PAIR, fixed_subject=fixed, moving_subject=moving, shape_zyx=list(GRID.shape), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], unit='mm',
                               sources=dict(fixed=dict(sha256='1'*64), moving=dict(sha256='2'*64)), pair_sha256=file_sha256(d/'pair.npz'), weights=dict(lam=.1)))
    return d


def test_the_recomputation_reads_the_truth_through_the_bench_on_a_synthetic_pair(work):
    from bayesreg.eval import metrics
    from bayesreg.eval import truth as bench_truth
    stage = work/'data/three_d/stage_he_smoke_synthetic'
    pair_dir = synthetic_pair(stage)
    out = work/'out'
    u = field(4)
    he.write_output(out/'output', u, GRID, cert_of(u), dict(line_status=he.HX))
    ctx = dict(stage_dir=str(stage), pair=S.PAIR, paths=dict(allowlist=str(ROOT/'configs/bayesreg/dev_subjects.json')))
    for other in (dict(ctx, pair=S.core.PAIRS[1]), ctx):                                     # revision 1, review minor 3: only the smoke's pair of the smoke's stage
        with pytest.raises(ValueError, match='reads the truth of'):
            S.recompute_metrics(other, out, stage=stage if other is not ctx else None)
    again = S.recompute_metrics(ctx, out, stage=stage)
    sealed = bench_truth.load_sealed_lists()
    _, grid, _, guard = bench_truth.admit_pair(stage.parent, stage.name, S.PAIR, ctx['paths']['allowlist'], sealed)
    labels, _ = bench_truth.load_truth(pair_dir, bench_truth.scheme_for(guard, pair_dir), grid, guard)
    scalars, per_label, official = metrics.evaluate_field(he.u_to_d(u, GRID), labels, metrics.OASIS, (1., 1., 1.))
    assert S.canonical_json(again['metrics']) == S.canonical_json(scalars) and S.canonical_json(again['per_label']) == S.canonical_json(per_label)
    assert 'seg35' in again['metrics'] and 'seg4' in again['metrics'] and again['d_sha256'] == he.array_sha256(he.u_to_d(u, GRID))
