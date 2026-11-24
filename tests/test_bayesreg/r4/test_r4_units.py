"""R4 的单元（总负责人 2026-11-13 约 1:08 PM CST 的补充）：supervisor、定稿、SDlogJ 三个单元在 spy 之下（白名单与正条件）；Atropos 子进程的 argv 断言（C25.10-5）；
单步命令的启动（tag-r4-<步>、机器门、hold、单元间隔、起过一次不再起、失败写 stop），与一臂的队列共用同一套门。systemd-run 由 PATH shim 冒充，绝不真起单元。"""
import json
import os
from pathlib import Path
import stat
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
for _p in ('r4', 'he', 'mc0', 'ef', 'fl', 'd0'):
    p = str(ROOT/'experiments'/_p)
    if p not in sys.path:
        sys.path.insert(0, p)

import r4_bench as rb  # noqa: E402
import r4_chain as ch  # noqa: E402
import r4_common as rc  # noqa: E402
import r4_eval as re_  # noqa: E402
import r4_spy as spy  # noqa: E402
import run_r4  # noqa: E402
from ._batch import batch_here  # noqa: E402
import bayesreg.io.pair as pair_module  # noqa: E402  (looked up at call time: the spy counts the loader by its module attribute)

from .test_r4_bench import World  # noqa: E402


def test_the_plan_is_pinned():
    assert rc.PLAN_SHA256 == 'b8e62d8b6af7d1471b133b7d6ccc3be35d5bad8f0512358621cc7303e923de2c' and rc.plan_problem() is None


# ---- supervisor、定稿、SDlogJ 在 spy 之下 ----------------------------------------------------------------------------------------------------------
@pytest.fixture()
def world(tmp_path):
    return World(tmp_path, rb.load_bench())


def tmp_rules(tmp_path, root, role='bench'):
    rules = spy.bench_rules(root, unit='t', stage_dir=tmp_path/'stage', pred_root=tmp_path/'pred', atropos_root=root/'atropos') if role == 'bench' \
        else spy.dev_rules(root, unit='t', extra=(root/'dev_final', root/'dev_runs'))
    rules.roots = [tmp_path.resolve()]                                                             # tmp 里的文件也判
    return rules


def ctx_of(world, root, out):
    return dict(role='bench', root=str(root), stage_dir=str(world.tmp/'stage'), pair=world.pair, out=str(out), paths=dict(pred_root=str(world.tmp/'pred'), atropos_root=str(root/'atropos')))


def test_the_supervisor_runs_under_the_spy_with_its_positive_condition(world, tmp_path):
    root = tmp_path/'r4'
    out = root/'runs'/'CTL'/world.pair
    ctx = ctx_of(world, root, out)
    rules = tmp_rules(tmp_path, root)

    def good(c):
        pair_module.load_pair(Path(c['stage_dir'])/c['pair']/'pair.npz', allowlist=rc.ROOT/rc.ALLOWLIST_REL)
        Path(c['out']).mkdir(parents=True)
        return 0
    assert run_r4.supervise(ctx, rules=rules, run=good) == 0
    assert json.loads((out/'opened_files_supervisor.json').read_text())['passed'] is True

    def nothing(c):
        return 0
    assert run_r4.supervise(dict(ctx, out=str(root/'x')), rules=rules, run=nothing) == 8               # 一次也没读到 fixed、moving：正条件不成立

    def truth(c):
        pair_module.load_pair(Path(c['stage_dir'])/c['pair']/'pair.npz', allowlist=rc.ROOT/rc.ALLOWLIST_REL)
        try:
            np.load(Path(c['stage_dir'])/c['pair']/'pair.npz')['fixed_seg35']
        except Exception:                                                                     # noqa: BLE001 - the spy raised Breach; run_pair would record it
            pass
        return 0
    assert run_r4.supervise(dict(ctx, out=str(root/'y')), rules=rules, run=truth) == 8

    def stray(c):
        pair_module.load_pair(Path(c['stage_dir'])/c['pair']/'pair.npz', allowlist=rc.ROOT/rc.ALLOWLIST_REL)
        (tmp_path/'stray.json').write_text('{}')
        open(tmp_path/'stray.json').close()
        return 0
    assert run_r4.supervise(dict(ctx, out=str(root/'z')), rules=rules, run=stray) == 8               # 根之内、白名单之外


def test_the_finalize_unit_spies_each_pair(world, tmp_path):
    root = tmp_path/'r4'
    pairs = [world.pair, 'NO__OUTPUT']
    for pair, status in ((world.pair, rc.HX), ('NO__OUTPUT', rc.IMPLEMENTATION)):
        out = root/'runs'/'B'/pair
        out.mkdir(parents=True)
        rc.write_json(out/'context.json', dict(ctx_of(world, root, out), pair=pair))
        rc.write_json(out/'line.json', dict(line_status=status))
    seen = []

    def finalize(c):
        seen.append(c['pair'])
        if c['pair'] == world.pair:
            pair_module.load_pair(Path(c['stage_dir'])/c['pair']/'pair.npz', allowlist=rc.ROOT/rc.ALLOWLIST_REL)
    assert run_r4.finalize_arm(root, 'bench', 'B', pairs, rules=tmp_rules(tmp_path, root), finalize=finalize) == 0 and seen == pairs
    record = json.loads((root/'final'/'B'/'opened_files_finalize.json').read_text())
    assert record['passed'] and record['pairs']['NO__OUTPUT']['he_spy']['expect_reads'] is False
    (root/'final'/'B'/'opened_files_finalize.json').unlink()
    assert run_r4.finalize_arm(root, 'bench', 'B', pairs, rules=tmp_rules(tmp_path, root), finalize=lambda c: None) == 8      # 有输出的对没读到图：正条件不成立


def dev_final(root, arm, pair, u):
    o = root/'dev_runs'/arm/pair/'output'
    o.mkdir(parents=True)
    np.savez(o/'final.npz', u=u, state_hash=np.array(rc.array_sha256(u)))
    rc.write_json(o/'line.json', dict(line_status=rc.HX, grid=dict(spacing_xyz=[1., 1., 1.])))
    rc.write_json(ch.final_dir(root, 'dev', arm, pair)/'final.json', dict(line_status=rc.HX, output=dict(directory=str(o), final_npz_sha256=rc.sha256_file(o/'final.npz'), state_hash=rc.array_sha256(u))))


def test_the_sdlogj_unit_reads_only_the_outputs(tmp_path, monkeypatch):
    root = tmp_path/'r4'
    pairs = [f'D{k}' for k in range(19)]
    for p in pairs:
        dev_final(root, 'B', p, np.zeros((7, 7, 7, 3)))
    assert run_r4.sdlogj_arm(root, 'B', pairs, rules=tmp_rules(tmp_path, root, 'dev')) == 0
    doc = json.loads((root/'dev_final'/'B'/'feasibility.json').read_text())
    assert doc['spy_passed'] and doc['verdict']['verdict'] == 'feasible'
    monkeypatch.setattr(ch, 'output_array', lambda final: (np.zeros((7, 7, 7, 3)), None))                # 没打开 final.npz：正条件不成立
    (root/'dev_final'/'B'/'feasibility.json').unlink()
    (root/'dev_final'/'B'/'opened_files_sdlogj.json').unlink()
    assert run_r4.sdlogj_arm(root, 'B', pairs, rules=tmp_rules(tmp_path, root, 'dev')) == 8


# ---- Atropos 的 argv（C25.10-5） -------------------------------------------------------------------------------------------------------------------
def atropos_world(tmp_path):
    ants = tmp_path/'ants'
    ants.mkdir()
    out = tmp_path/'atropos'/'S'
    out.mkdir(parents=True)
    for n in ('t1.nii.gz', 'mask.nii.gz', 't1_trunc.nii.gz'):
        (out/n).write_bytes(b'x')
    return ants, out


def test_atropos_gets_only_the_stage_image_files(tmp_path):
    ants, out = atropos_world(tmp_path)
    trunc = [str(ants/'ImageMath'), '3', 't1_trunc.nii.gz', 'TruncateImageIntensity', 't1.nii.gz', '0.01', '0.99', '256', 'mask.nii.gz']
    cmd = [str(ants/'Atropos'), '-d', '3', '-a', 't1_trunc.nii.gz', '-x', 'mask.nii.gz', '-i', 'KMeans[3]', '-k', 'Gaussian', '-c', '[5,0.001]', '-m', '[0.3,1x1x1]', '-r', '0',
           '-o', '[labels.nii.gz,posteriors%02d.nii.gz]', '-v', '1']
    kw = dict(ants=ants, out_root=tmp_path/'atropos', subject='S')
    assert rb.atropos_argv_problems(trunc, out, **kw) == [] and rb.atropos_argv_problems(cmd, out, **kw) == []
    for bad in ([str(ants/'Atropos'), '-a', '/mnt/d/data/S/aligned_norm.nii.gz', '-x', 'mask.nii.gz'], [str(ants/'Atropos'), '-a', 't1_trunc.nii.gz', '-x', '../x/mask.nii.gz'],
                ['/bin/cat', 'aligned_seg35.nii.gz'], [str(ants/'Atropos'), '-a', 'aligned_seg35.nii.gz', '-x', 'mask.nii.gz']):
        assert rb.atropos_argv_problems(bad, out, **kw)
    assert rb.atropos_argv_problems(cmd, tmp_path, **kw)                                           # 工作目录不对


def test_the_atropos_guard_refuses_before_the_subprocess_starts(tmp_path):
    ants, out = atropos_world(tmp_path)
    started = []

    class Real:
        STDOUT = -2

        @staticmethod
        def run(argv, **kw):
            started.append(argv)
            return type('Done', (), dict(returncode=0))()
    guard = rb.AtroposArgv(Real, ants=ants, out_root=tmp_path/'atropos', subject='S')
    guard.run([str(ants/'ImageMath'), '3', 't1_trunc.nii.gz', 'TruncateImageIntensity', 't1.nii.gz', '0.01', '0.99', '256', 'mask.nii.gz'], cwd=out)
    assert len(started) == 1 and guard.STDOUT == -2
    with pytest.raises(rb.BenchRefused):
        guard.run([str(ants/'Atropos'), '-a', '/mnt/d/data/S/aligned_seg35.nii.gz', '-x', 'mask.nii.gz'], cwd=out)
    assert len(started) == 1 and guard.calls[-1]['problems']


def test_bench_atropos_installs_the_guard_around_segment(monkeypatch):
    import atropos_seg
    seen = {}

    class B:
        pair_names = ('S__T',)

        def ordered_subjects(self):
            return ['S', 'T']

    class Pin:
        fixed_subject, moving_subject, provenance = 'S', 'T', dict(pair_sha256='a'*64)
    monkeypatch.setattr(rb, 'bench_admission', lambda bench, **kw: None)
    monkeypatch.setattr(rb, 'load_bench_pair', lambda bench, stage, name: (Pin(), None))

    def segment(subject, entries, out_root, threads, stage_name):
        seen[subject] = (atropos_seg.subprocess, [e[:2] for e in entries])
        if subject == 'T':
            raise SystemExit('Atropos failed')
        return {}
    import subprocess
    with pytest.raises(SystemExit):
        rb.bench_atropos(B(), Path('/nonexistent'), Path('/nonexistent/atropos'), segment=segment)
    assert isinstance(seen['S'][0], rb.AtroposArgv) and seen['S'][0].subject == 'S' and seen['S'][1] == [('S__T', 'fixed')]
    assert isinstance(seen['T'][0], rb.AtroposArgv) and seen['T'][0].subject == 'T'
    assert atropos_seg.subprocess is subprocess                                                     # restored after a failure too


# ---- 单步命令与共用的门（PATH shim 冒充 systemd-run） ---------------------------------------------------------------------------------------------------
@pytest.fixture()
def shim(tmp_path, monkeypatch):
    d = tmp_path/'shim'
    d.mkdir()
    log = tmp_path/'shim_calls.jsonl'
    script = d/'systemd-run'
    script.write_text('#!/usr/bin/env python3\nimport json, os, sys\n'
                      f'open({str(log)!r}, "a").write(json.dumps(sys.argv[1:]) + "\\n")\n'
                      'sys.exit(int(os.environ.get("SHIM_EXIT", "0")))\n', encoding='utf-8')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv('PATH', f"{d}:{os.environ.get('PATH', '')}")
    monkeypatch.delenv('SHIM_EXIT', raising=False)
    return log


def calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def step(name, root, extra=(), gate=None, **kw):
    batch_here(root)
    return run_r4.run_step(name, list(extra), root=root, gate=gate or (lambda own: (True, {})), gap=0, repo=root, require_committed=False, **kw)


def test_a_step_runs_once_in_its_tag_r4_unit(tmp_path, shim):
    root = tmp_path/'r4'
    assert step('prepare-bench', root) == 0
    argv = calls(shim)
    assert len(argv) == 1 and '--unit=tag-r4-prepare-bench' in argv[0] and argv[0][-1] == 'prepare-bench' and '--property=MemoryMax=12G' in argv[0]
    assert step('prepare-bench', root) == 2 and len(calls(shim)) == 1                              # 起过一次：不重跑
    assert step('eval-bench', root, ['--read', 'first']) == 0 and '--unit=tag-r4-eval-bench-read-first' in calls(shim)[-1]


def test_a_failed_step_writes_a_stop_and_blocks_every_unit(tmp_path, shim, monkeypatch):
    root = tmp_path/'r4'
    monkeypatch.setenv('SHIM_EXIT', '1')
    assert step('atropos-bench', root) == 9
    stops = re_.stop_files(root)
    assert len(stops) == 1 and rc.read_json(stops[0])['scope'] == 'bench'
    monkeypatch.setenv('SHIM_EXIT', '0')
    assert step('seg-bench', root) == 9 and len(calls(shim)) == 1                                  # 没处理的 stop：任何单步都不起
    assert run_r4.queue_arm('bench', 'CTL', root=root, pairs=['a__b'], gate=lambda own: (True, {}), gap=0, repo=root, require_committed=False) == 9 and len(calls(shim)) == 1


def test_hold_and_a_closed_gate_start_nothing_and_leave_the_step_startable(tmp_path, shim):
    root = tmp_path/'r4'
    root.mkdir()
    (root/'hold').touch()
    assert step('smoke', root) == 3 and calls(shim) == []
    (root/'hold').unlink()
    assert step('smoke', root, gate=lambda own: (False, dict(problems=['load']))) == 3 and calls(shim) == []
    assert step('smoke', root) == 0 and len(calls(shim)) == 1
    assert len(list((root/'steps').glob('smoke.not_started_*.json'))) == 2


def test_the_queue_and_the_steps_share_one_gate_and_spacing(tmp_path, monkeypatch):
    root = tmp_path/'r4'
    seen = []
    monkeypatch.setattr(run_r4, 'gated_launch', lambda root_, name, args, **kw: (seen.append(name), (0, None))[1])
    step('devcheck', root)
    run_r4.queue_arm('dev', 'B', root=root, pairs=[], gate=lambda own: (True, {}), gap=0, repo=root, require_committed=False)
    assert seen == ['tag-r4-devcheck', 'tag-r4-dev-B-finalize', 'tag-r4-dev-B-sdlogj']
    slept = []
    run_r4.mark_unit_end(root, 'tag-r4-x', 0, now=lambda: 100.)
    run_r4.pace(root, 10, sleep=slept.append, now=lambda: 104.)
    assert slept == [6.]
    run_r4.pace(root, 10, sleep=slept.append, now=lambda: 111.)
    assert slept == [6.]


def test_the_gate_waits_then_stops(monkeypatch):
    t = [0.]
    monkeypatch.setattr(run_r4.time, 'monotonic', lambda: t[0])

    def sleep(s):
        t[0] += s
    ok, record = run_r4.wait_gate('tag-r4-queue', minutes=1, gate=lambda own: (['load 2.0'], {}), sleep=sleep, log=lambda *a: None)
    assert ok is False and record['problems'] == ['load 2.0'] and t[0] >= 60
    assert run_r4.wait_gate('tag-r4-queue', minutes=1, gate=lambda own: ([], {}), sleep=sleep)[0] is True
