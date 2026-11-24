"""R4 修订 4（修订 3 的差异复核：阻断 0、重要 1、次要 6；总负责人裁决）的测试：N3-I1 修复子进程被杀；次要 1 逻辑读与文件绑定；次要 2 npz 里的真值成员；
次要 3 开发集读数记被杀的记录；次要 4 崩溃的子进程（读图之前、之后）的 supervisor 结局；次要 5 台第二读的 stop。合成数据，不读真值，不用 GPU，不起单元。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
for _p in ('r4', 'he', 'mc0', 'ef', 'fl', 'd0', 'unv2'):
    p = str(ROOT/'experiments'/_p)
    if p not in sys.path:
        sys.path.insert(0, p)

import r4_chain as ch  # noqa: E402
import r4_common as rc  # noqa: E402
import r4_eval as re_  # noqa: E402
import r4_lock as lk  # noqa: E402
import r4_spy as spy  # noqa: E402
import r4_steps  # noqa: E402
import run_r4  # noqa: E402

from .test_r4_rev3 import failing_shim, judge_tmp  # noqa: E402,F401

PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'
FROZEN = Path('/home/rmcfz/seq_vp_tag_results/frozen_eb808be')

E2E_MODES = r'''
import json, sys
from pathlib import Path
import numpy as np
ROOT, TMP, CASE = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[4]          # the world script sets MODE for itself
sys.path[:0] = [str(ROOT/'experiments'/d) for d in ('r4', 'he', 'mc0', 'ef', 'fl', 'd0')]
import mc0_core as core
core.frozen_api()
exec(compile(Path(sys.argv[3]).read_text(), 'world', 'exec'))
import he_core as he
import r4_eval as re_


def identity(ctx_, pin, **kw):
    u = np.zeros(pin.grid.shape+(3,))
    cert = dict(status='certified_feasible', min_jd_lower_bound=1., n_infeasible=0, n_uncertain=0, state_hash=he.array_sha256(u), process='test identity')
    return u, cert, dict(kind='identity'), he.FALLBACK_ID
ch.fallback_chain = identity
inproc = ch.InProcessChild()
if CASE == 'repair_killed':
    def child(stage, c):
        if stage == 't88':
            return inproc(stage, c)                                                   # the real T88 under its spy
        (Path(c['out'])/'repair').mkdir()
        (Path(c['out'])/'repair'/'started.json').write_text('{}')
        return -9                                                                     # killed after it started, before repair.json and its spy record
else:
    def crash_before(c):
        raise RuntimeError('a bug before the pair is read')

    def crash_after(c):
        ch.admit(c)
        raise RuntimeError('a bug after the pair is read')
    ch.t88_main = crash_before if CASE == 'crash_before' else crash_after
    child = inproc
code = run_r4.supervise(ctx, run=lambda c: ch.run_pair(c, child=child))
out = Path(ctx['out'])
line = ch.line_record(out)
notes = []
problems = re_.spy_record_problems(out, notes=notes)
print(json.dumps(dict(code=code, line_status=line['line_status'], spy=line.get('spy'), repair=line.get('repair'), problems=problems, notes=notes,
                      implementation=line.get('implementation_error'))))
'''


def run_mode(tmp_path, mode):
    from .test_r4_review import E2E
    world = E2E.split("ch.repair_main = fake_repair")[0].replace("ROOT, TMP, MODE = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]", "MODE = 'world'")
    (tmp_path/'world.py').write_text(world)
    env = dict(os.environ, PYTHONPATH=f'{FROZEN}:{ROOT}', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    done = subprocess.run([PY, '-B', '-c', E2E_MODES, str(ROOT), str(tmp_path), str(tmp_path/'world.py'), mode], capture_output=True, text=True, env=env, timeout=900, cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


# ---- N3-I1：修复子进程被杀 -------------------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not FROZEN.is_dir(), reason='the frozen tree is not on this machine')
def test_a_killed_repair_child_falls_back_and_passes_the_prelock(tmp_path):
    out = run_mode(tmp_path, 'repair_killed')
    assert out['code'] == 0 and out['line_status'] == rc.FALLBACK_ID and out['repair']['returncode'] == -9
    assert out['spy']['repair'] == ch.KILLED_NO_RECORD and out['spy']['t88'] == 'passed'
    assert out['problems'] == [] and [n['stage'] for n in out['notes']] == ['repair']


# ---- 次要 4：崩溃的子进程 ---------------------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not FROZEN.is_dir(), reason='the frozen tree is not on this machine')
def test_a_child_that_crashes_before_reading_the_pair_stops_the_queue(tmp_path):
    out = run_mode(tmp_path, 'crash_before')
    assert out['code'] == ch.STOP_EXIT and out['line_status'] == rc.IMPLEMENTATION and out['implementation']['kind'] == 'spy'


@pytest.mark.skipif(not FROZEN.is_dir(), reason='the frozen tree is not on this machine')
def test_a_current_arm_child_that_crashes_after_reading_falls_back_by_d74(tmp_path):
    out = run_mode(tmp_path, 'crash_after')
    assert out['code'] == 0 and out['line_status'] == rc.FALLBACK_ID and out['spy']['t88'] == 'passed'


# ---- 次要 1：逻辑读与文件绑定 ------------------------------------------------------------------------------------------------------------------------
@pytest.fixture()
def reader(tmp_path):
    for subject in ('S1', 'S2'):
        (tmp_path/'data'/subject).mkdir(parents=True)
        for kind in ('seg35', 'seg4'):
            (tmp_path/'data'/subject/f'aligned_{kind}.nii.gz').write_bytes(b'x')
    module = types.ModuleType('_r4_rev4_reader')

    def read(subject, kind='seg35', other=None, only=None):
        own = only or (subject, kind)                                                          # only：修订 5 第 1 条，这次逻辑读只开 only 这份（别人的，或本人另一种）
        for _ in range(2):
            open(tmp_path/'data'/own[0]/f'aligned_{own[1]}.nii.gz', 'rb').close()
        if other:
            open(tmp_path/'data'/other/f'aligned_{kind}.nii.gz', 'rb').close()
    module.read = read
    sys.modules['_r4_rev4_reader'] = module
    rules = judge_tmp(spy.bench_rules(tmp_path/'r4', unit='eval', stage_dir=tmp_path/'s', pred_root=tmp_path/'s', atropos_root=tmp_path/'s'), tmp_path, tmp_path/'data')
    rules.allow_truth = True
    yield module, rules, (('_r4_rev4_reader', 'read', lambda a, k: (a[0], k.get('kind', 'seg35'))),)
    del sys.modules['_r4_rev4_reader']


def test_a_logical_read_opens_exactly_its_own_truth(reader):
    module, rules, readers = reader
    with spy.UnitSpy(rules, expect_reads=False, truth_readers=readers) as s:
        module.read('S1')
        module.read('S2')
    assert spy.truth_reads(s.record(), expected_keys=[('S1', 'seg35'), ('S2', 'seg35')]) == []
    with spy.UnitSpy(rules, expect_reads=False, truth_readers=readers) as s:
        module.read('S1', other='S2')                                                          # opens another subject's truth inside its logical read
    assert any('not exactly' in p for p in spy.truth_reads(s.record(), expected_keys=[('S1', 'seg35')]))
    with spy.UnitSpy(rules, expect_reads=False, truth_readers=readers) as s:
        module.read('S1')
        module.read('S2')                                                                      # a subject the read does not expect
    assert any('not expected' in p for p in spy.truth_reads(s.record(), expected_keys=[('S1', 'seg35')]))


# ---- 次要 2：npz 里的真值成员 -------------------------------------------------------------------------------------------------------------------------
def test_truth_members_of_an_npz_are_read_only_inside_a_logical_read(tmp_path):
    (tmp_path/'a__b').mkdir()
    path = tmp_path/'a__b'/'pair.npz'                                                              # 修订 5 第 2 条：开发集的成员要来自 <该对>/pair.npz
    np.savez(path, fixed=np.zeros(2), fixed_seg35=np.zeros(2), moving_seg35=np.zeros(2))
    fake = types.ModuleType('_mc0_evaluate')

    def truth_labels(pair):
        with np.load(path) as z:
            return {k: np.array(z[k]) for k in ('fixed_seg35', 'moving_seg35')}, {}, 'sha'
    fake.truth_labels = truth_labels
    saved = sys.modules.get('_mc0_evaluate')
    sys.modules['_mc0_evaluate'] = fake
    try:
        rules = judge_tmp(spy.dev_rules(tmp_path/'r4', unit='eval-dev-1'), tmp_path)
        with spy.UnitSpy(rules, expect_reads=False, truth_readers=r4_steps.TRUTH_READERS_DEV) as s:
            sys.modules['_mc0_evaluate'].truth_labels('a__b')
        assert spy.truth_reads(s.record(), expected_keys=[('a__b',)]) == [] and len(s.record()['truth_members']) == 2
        with spy.UnitSpy(rules, expect_reads=False, truth_readers=r4_steps.TRUTH_READERS_DEV) as s:
            sys.modules['_mc0_evaluate'].truth_labels('a__b')
            with np.load(path) as z:
                np.array(z['moving_seg35'])                                                    # outside truth_labels
        assert any('outside a logical read' in p for p in spy.truth_reads(s.record(), expected_keys=[('a__b',)]))
    finally:
        if saved is None:
            del sys.modules['_mc0_evaluate']
        else:
            sys.modules['_mc0_evaluate'] = saved


# ---- 次要 3：开发集读数记被杀的记录 --------------------------------------------------------------------------------------------------------------------
def test_the_development_reading_records_killed_children(tmp_path, monkeypatch):
    pairs = [f'D{k}' for k in range(19)]

    def prelock(*a, notes=None, **k):
        notes.append(dict(run_dir='x', stage='repair', returncode=-9, note=ch.KILLED_NO_RECORD))
        return {p: {} for p in pairs}, {p: np.zeros((2, 2, 2, 3)) for p in pairs}, {}
    monkeypatch.setattr(re_, 'dev_prelock', prelock)
    monkeypatch.setattr(re_, 'run_hx_admit', lambda stage, pair: (None, None))
    monkeypatch.setattr(re_, 'evaluator', lambda: (_ for _ in ()).throw(RuntimeError('stop after the lock')))
    doc = re_.dev_eval(tmp_path, 'B', 1, pairs=pairs, hs_root=tmp_path/'hs', repo=tmp_path, code_commit='c', plan_sha256='p')
    assert doc['killed_without_spy_record'][0]['stage'] == 'repair'


# ---- 次要 5：台第二读的 stop ---------------------------------------------------------------------------------------------------------------------------
def test_the_second_bench_read_is_after_the_lock_only_with_its_second_line(tmp_path, failing_shim):
    from ._batch import batch_here
    root = batch_here(tmp_path/'r4')
    lock = lk.lock_path(root, lk.BENCH)
    lk.take_first(lock, arms=rc.ARM_ORDER, code_commit='c', plan_sha256='p', reason='first')
    kw = dict(root=root, gate=lambda own: (True, {}), gap=0, repo=root, require_committed=False)
    assert run_r4.run_step('eval-bench', ['--read', 'second', '--combo', 'COMBO-B-C1'], **kw) == 9
    stop = rc.read_json(re_.stop_files(root)[0])
    assert stop['after_lock'] is False and stop['arm'] == 'COMBO-B-C1' and stop['scope'] == 'combo' and '读真值之前' in stop['reason']
    lk.take_second(lock, 'COMBO-B-C1', code_commit='c', plan_sha256='p', truth_sha256={})
    assert run_r4.lock_taken(root, 'eval-bench', ['--read', 'second', '--combo', 'COMBO-B-C1']) is True
    assert run_r4.lock_taken(root, 'eval-bench', ['--read', 'first']) is True and run_r4.lock_taken(root, 'eval-dev', ['--read', '1', '--arm', 'B']) is False
