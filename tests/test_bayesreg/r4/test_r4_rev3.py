"""R4 修订 3（差异复核 0c9d1c68 → e16214fc：阻断 1、重要 2、次要 6；总负责人裁决）的测试：
N-B1 评测单元按逻辑读计数（真实的 bench_truth 走合成 NIfTI，不打桩）；N-I1 定稿单元的记录名与写操作的白名单；N-I2 子进程被杀、没有 spy 记录时照 D74；
次要 1 崩溃的子进程照写记录再抛出；次要 2 子进程自己的准入被拒不套正条件；次要 3 开发集每对一次逻辑读；次要 4 单步的 stop；次要 6 嵌套单元与自己的单元名。
合成数据，不读真值，不用 GPU，不起单元。"""
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
for _p in ('r4', 'he', 'mc0', 'ef', 'fl', 'd0', 'unv2'):
    p = str(ROOT/'experiments'/_p)
    if p not in sys.path:
        sys.path.insert(0, p)

import r4_bench as rb  # noqa: E402
import r4_chain as ch  # noqa: E402
import r4_common as rc  # noqa: E402
import r4_eval as re_  # noqa: E402
import r4_lock as lk  # noqa: E402
import r4_spy as spy  # noqa: E402
import r4_steps  # noqa: E402
import run_r4  # noqa: E402
from ._batch import batch_here  # noqa: E402

PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'
FROZEN = Path('/home/rmcfz/seq_vp_tag_results/frozen_eb808be')


def judge_tmp(rules, tmp_path, image_root=None):
    rules.roots = [Path(tmp_path).resolve()]
    if image_root is not None:
        rules.image_root = Path(image_root).resolve()
    return rules


# ---- N-B1：评测单元按逻辑读计数 ----------------------------------------------------------------------------------------------------------------------
def synthetic_bench(tmp_path):
    import synthetic_tree as st
    work = st.build(tmp_path/'world')
    (work/'train_subjects.txt').write_text(''.join(f'{s}\n' for s in st.TRAINING), encoding='utf-8')
    world = rb.BenchWorld(repo=tmp_path, folds_path=work/'folds_full_v2.json', folds_sha256=rc.sha256_file(work/'folds_full_v2.json'), train_list_path=work/'train_subjects.txt',
                          allowlist_path=work/'dev_subjects.json', training_source=st.SOURCE, dev_source='synthetic_dev', synthetic=True)
    return rb.load_bench(world), work


def test_the_bench_evaluation_counts_logical_reads_of_the_real_reader(tmp_path):
    from bayesreg.seg.unet.data import OasisFiles
    bench, work = synthetic_bench(tmp_path)
    root = tmp_path/'r4'
    rules = spy.bench_rules(root, unit='eval-bench-first', stage_dir=root/'bench_stage', pred_root=root/'runs', atropos_root=root/'final', extra=(root/'eval',), record_dirs=(root/'eval',),
                            allow_images=False)
    rules.allow_truth = True
    judge_tmp(rules, tmp_path, work/'data')
    with spy.UnitSpy(rules, expect_reads=False, truth_readers=r4_steps.TRUTH_READERS_BENCH) as s:
        truth, guard = re_.bench_truth(bench, files=OasisFiles(work/'data'))                 # the real LabelGuard, read_labels and load_scalar
    record = s.record()
    keys = [(subject, kind) for subject in bench.ordered_subjects() for kind in ('seg35', 'seg4')]
    assert sorted(truth) == sorted(keys) and len(guard.reads) == len(keys)
    assert spy.truth_reads(record, expected_keys=keys) == [] and record['passed'] is True
    opens = {}
    for row in record['opened']:
        if spy.is_truth(row['path']):
            opens[row['resolved']] = opens.get(row['resolved'], 0)+1
    assert opens and max(opens.values()) > 1                                                 # several raw opens per logical read: the old count would have failed


# ---- N-I1：定稿单元的记录名；写操作只许本单元的输出目录 --------------------------------------------------------------------------------------------------
def test_the_finalize_mismatch_record_is_written_under_the_spy(tmp_path, monkeypatch):
    from dataclasses import dataclass
    import he_core as he
    from jd_curl.three_d.feasibility import PrescreenCertificateConfig
    from jd_curl.three_d.grid import make_grid_3d
    grid = make_grid_3d(nx=5, ny=5, nz=5, hx=1., hy=1., hz=1., unit='mm')
    u = np.zeros(grid.shape+(3,))
    root = tmp_path/'r4'
    out = root/'runs'/'B'/'P__Q'
    out.mkdir(parents=True)
    cert = dict(status='certified_feasible', min_jd_lower_bound=1., n_infeasible=0, n_uncertain=0, state_hash=he.array_sha256(u), process='decision')
    he.write_output(out/'output', u, grid, cert, dict(line_status=rc.HX))
    monkeypatch.setattr(ch, 'admit', lambda ctx: (SimpleNamespace(grid=grid), None))

    @dataclass
    class Result:
        status: str
        min_jd_lower_bound: float
        n_infeasible: int
        n_uncertain: int
        state_hash: str
        n_cells_checked: int = 1
        witness_cell: object = None
        witness_reference_point: object = None

    def certify(v, g):                                                                        # a review that differs (another lower bound) and a third certificate that passes
        return Result('certified_feasible', .5, 0, 0, he.array_sha256(v))
    ctx = dict(out=str(out), root=str(root), role='bench', arm='B', pair='P__Q', stage_dir=str(root/'bench_stage'), settings=dict(r4=dict(spec_sha256='s')),
               paths=dict(pred_root=str(root/'pred'), atropos_root=str(root/'atropos'), cache=str(root/'cache')))
    rules = judge_tmp(spy.pair_rules(ctx, 'finalize'), tmp_path)
    with spy.UnitSpy(rules, expect_reads=False) as s:
        ch.finalize_pair(ctx, certification=(certify, PrescreenCertificateConfig()))
    record = s.record()
    assert (out/'finalize_cert_mismatch_0.json').is_file() and not list(out.glob('review_mismatch_*'))
    assert record['passed'] is True and any(r['mode'] == 'write' and 'finalize_cert_mismatch' in r['path'] for r in record['opened'])
    assert ch.read_final(root, 'bench', 'B', 'P__Q')['line_status'] == rc.HX
    with spy.UnitSpy(rules, expect_reads=False) as s:
        (root/'runs'/'B'/'OTHER__PAIR').mkdir(parents=True)
        (root/'runs'/'B'/'OTHER__PAIR'/'x.json').write_text('{}')                             # a write outside this unit's output directories
    assert s.record()['passed'] is False


# ---- N-I2：子进程被杀、没有 spy 记录 ---------------------------------------------------------------------------------------------------------------------
def test_a_killed_child_without_a_record_passes_the_prelock_by_d74(tmp_path):
    run = tmp_path/'run'
    (run/'t88').mkdir(parents=True)
    rc.write_json(run/'opened_files_supervisor.json', dict(passed=True))
    (run/'output').mkdir()
    rc.write_json(run/'output'/'line.json', dict(line_status=rc.FALLBACK_C, t88=dict(returncode=-9), spy=dict(t88=ch.KILLED_NO_RECORD, repair=None)))
    notes = []
    assert re_.spy_record_problems(run, notes=notes) == [] and notes and notes[0]['note'] == ch.KILLED_NO_RECORD
    (run/'output'/'line.json').unlink()
    rc.write_json(run/'output'/'line.json', dict(line_status=rc.FALLBACK_C, t88=dict(returncode=0), spy=dict(t88='missing')))
    assert re_.spy_record_problems(run)                                                       # exit 0 and no record: a problem
    assert ch.spy_state(run, 't88', -9) == ch.KILLED_NO_RECORD and ch.spy_state(run, 't88', 'timeout') == ch.KILLED_NO_RECORD and ch.spy_state(run, 't88', 1) == 'missing'
    assert ch.child_spy_problem(run, 't88', -9) is None and ch.child_spy_problem(run, 't88', 1)


E2E_KILLED = r'''
import json, sys
from pathlib import Path
import numpy as np
ROOT, TMP = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(ROOT/'experiments'/d) for d in ('r4', 'he', 'mc0', 'ef', 'fl', 'd0')]
import mc0_core as core
core.frozen_api()
exec(compile(Path(sys.argv[3]).read_text(), 'world', 'exec'))
import he_core as he
import r4_eval as re_


def killed_child(stage, c):
    return -9                                                                         # killed by a signal before it wrote anything


def identity(ctx_, pin, **kw):
    u = np.zeros(pin.grid.shape+(3,))
    cert = dict(status='certified_feasible', min_jd_lower_bound=1., n_infeasible=0, n_uncertain=0, state_hash=he.array_sha256(u), process='test identity')
    return u, cert, dict(kind='identity'), he.FALLBACK_ID
ch.fallback_chain = identity
code = run_r4.supervise(ctx, run=lambda c: ch.run_pair(c, child=killed_child))
out = Path(ctx['out'])
line = ch.line_record(out)
notes = []
problems = re_.spy_record_problems(out, notes=notes)
print(json.dumps(dict(code=code, line_status=line['line_status'], spy=line.get('spy'), problems=problems, notes=notes)))
'''


@pytest.mark.skipif(not FROZEN.is_dir(), reason='the frozen tree is not on this machine')
def test_the_supervisor_falls_back_when_its_child_is_killed_without_a_record(tmp_path):
    from .test_r4_review import E2E
    world = E2E.split("ch.repair_main = fake_repair")[0]                                       # the synthetic bench pair and its context, as the end-to-end tests build them
    world = world.replace("ROOT, TMP, MODE = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]", "MODE = 'killed'")
    (tmp_path/'world.py').write_text(world)
    env = dict(os.environ, PYTHONPATH=f'{FROZEN}:{ROOT}', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    done = subprocess.run([PY, '-B', '-c', E2E_KILLED, str(ROOT), str(tmp_path), str(tmp_path/'world.py')], capture_output=True, text=True, env=env, timeout=900, cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-3000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out['code'] == 0 and out['line_status'] == rc.FALLBACK_ID and out['spy']['t88'] == ch.KILLED_NO_RECORD and out['problems'] == [] and out['notes']


# ---- 次要 1、2：子进程的 spy 记录 ------------------------------------------------------------------------------------------------------------------------
def child_ctx(tmp_path):
    root = tmp_path/'r4'
    out = root/'runs'/'B'/'P__Q'
    out.mkdir(parents=True)
    return dict(out=str(out), root=str(root), role='bench', arm='B', pair='P__Q', stage_dir=str(root/'bench_stage'),
                paths=dict(pred_root=str(root/'pred'), atropos_root=str(root/'atropos'), cache=str(root/'cache')))


def test_a_crashing_child_writes_its_record_and_raises(tmp_path, monkeypatch):
    ctx = child_ctx(tmp_path)

    def crash(c):
        (Path(c['out'])/'t88').mkdir()
        raise RuntimeError('a bug in the T88 child')
    monkeypatch.setattr(ch, 't88_main', crash)
    with pytest.raises(RuntimeError):
        ch.run_child_spied(ctx, 't88', rules=judge_tmp(spy.pair_rules(ctx, 't88'), tmp_path))
    record = rc.read_json(Path(ctx['out'])/'t88'/'opened_files.json')
    assert record['crashed'].startswith('RuntimeError')
    assert ch.t88_status_decision(None, 1, ['B'])[0] == 'implementation' and ch.t88_status_decision(None, 1, [])[0] == 'fallback'


def test_a_child_whose_own_admission_refused_reads_nothing_and_passes_its_spy(tmp_path, monkeypatch):
    ctx = child_ctx(tmp_path)

    def refused(c):
        d = Path(c['out'])/'t88'
        d.mkdir()
        rc.write_json(d/'run.json', dict(status='provenance_rejected', reason='admission: BenchRefused: not a bench subject'))
        return 0
    monkeypatch.setattr(ch, 't88_main', refused)
    assert ch.run_child_spied(ctx, 't88', rules=judge_tmp(spy.pair_rules(ctx, 't88'), tmp_path)) == 0
    record = rc.read_json(Path(ctx['out'])/'t88'/'opened_files.json')
    assert record['passed'] is True and record['admission_refused'] is True
    assert ch.t88_status_decision(dict(status='provenance_rejected', reason='admission: x'), 0, ['B'])[0] == 'rejected'


# ---- 次要 3：开发集每对一次逻辑读 ----------------------------------------------------------------------------------------------------------------------------
def test_the_development_reader_is_counted_per_pair(tmp_path, monkeypatch):
    fake = types.ModuleType('_mc0_evaluate')
    fake.truth_labels = lambda pair: ({}, {}, 'sha')
    monkeypatch.setitem(sys.modules, '_mc0_evaluate', fake)
    rules = judge_tmp(spy.dev_rules(tmp_path/'r4', unit='eval-dev-1'), tmp_path)
    with spy.UnitSpy(rules, expect_reads=False, truth_readers=r4_steps.TRUTH_READERS_DEV) as s:
        for pair in ('a__b', 'c__d'):
            sys.modules['_mc0_evaluate'].truth_labels(pair)
    assert spy.truth_reads(s.record(), expected_keys=[('a__b',), ('c__d',)]) == []
    with spy.UnitSpy(rules, expect_reads=False, truth_readers=r4_steps.TRUTH_READERS_DEV) as s:
        sys.modules['_mc0_evaluate'].truth_labels('a__b')
        sys.modules['_mc0_evaluate'].truth_labels('a__b')
    assert spy.truth_reads(s.record(), expected_keys=[('a__b',), ('c__d',)])
    assert sys.modules['_mc0_evaluate'].truth_labels is fake.truth_labels                      # restored


# ---- 次要 4：单步的 stop ----------------------------------------------------------------------------------------------------------------------------------
@pytest.fixture()
def failing_shim(tmp_path, monkeypatch):
    d = tmp_path/'shim'
    d.mkdir()
    script = d/'systemd-run'
    script.write_text('#!/usr/bin/env python3\nimport sys\nsys.exit(1)\n', encoding='utf-8')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv('PATH', f"{d}:{os.environ.get('PATH', '')}")


def test_a_failed_step_names_its_arm_and_whether_the_truth_was_read(tmp_path, failing_shim):
    root = batch_here(tmp_path/'r4')
    kw = dict(root=root, gate=lambda own: (True, {}), gap=0, repo=root, require_committed=False)
    assert run_r4.run_step('eval-dev', ['--read', '1', '--arm', 'B'], **kw) == 9
    stop = rc.read_json(re_.stop_files(root)[0])
    assert stop['arm'] == 'B' and stop['after_lock'] is False and '读真值之前' in stop['reason']
    root2 = batch_here(tmp_path/'r4b')
    lk.take_first(lk.lock_path(root2, lk.DEV1), arms=['B'], code_commit='c', plan_sha256='p', reason='test')
    assert run_r4.run_step('eval-dev', ['--read', '1', '--arm', 'B'], **dict(kw, root=root2, repo=root2)) == 9
    stop = rc.read_json(re_.stop_files(root2)[0])
    assert stop['after_lock'] is True and '建锁之后' in stop['reason']


# ---- 次要 6：嵌套单元与自己的单元名 -----------------------------------------------------------------------------------------------------------------------
def test_configuration_c_runs_in_the_supervisors_unit_and_the_queue_knows_its_own(tmp_path):
    ctx = ch.make_context('bench', 'CTL', 'P__Q', root=tmp_path, stage_dir=tmp_path, check_identity=False, test=True)
    assert ctx['launcher'] == dict(kind='direct')
    with pytest.raises(ValueError):
        ch.make_context('bench', 'CTL', 'P__Q', root=tmp_path, stage_dir=tmp_path, check_identity=False, test=True, launcher='unit')
    (tmp_path/'cgroup').write_text('0::/user.slice/user-1000.slice/user@1000.service/app.slice/tag-r4-queue.service\n')
    assert run_r4.own_unit(tmp_path/'cgroup') == 'tag-r4-queue'
    (tmp_path/'cgroup2').write_text('0::/user.slice/user-1000.slice/session-3.scope\n')
    assert run_r4.own_unit(tmp_path/'cgroup2') is None
    problems, _ = run_r4.machine_gate('tag-r4-queue', run=lambda argv, **k: SimpleNamespace(stdout='tag-r4-queue.service loaded active running\n' if argv[0] == 'systemctl' else '', returncode=0),
                                      meminfo=_write(tmp_path/'mi', 'MemAvailable: 20000000 kB\n'), loadavg=_write(tmp_path/'la', '0.1 0 0 1/1 1\n'),
                                      disk=lambda p: SimpleNamespace(free=10**15))
    assert problems == []                                                                     # its own unit is not "another unit"


def _write(path, text):
    path.write_text(text)
    return path
