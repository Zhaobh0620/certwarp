"""R4 的链与失败的分层（合同 C25.6-10、-11、-12，C25.7-3，C25.8-4，C25.10-5 的锁一条）：以建锁为界。
读真值之前的失败 → STOP、续跑记录（(甲) 总负责人签、(乙) 引 BZ 的话；建锁之前都要 BZ 已看到）；建锁之后的任何结局都是终局（开发集算用掉一次读数）。
临时目录、合成数组、替身（队列的单元、证书）；不读真值、不用 GPU、不起单元。"""
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
for _p in ('r4', 'he', 'mc0', 'ef', 'fl', 'd0'):
    p = str(ROOT/'experiments'/_p)
    if p not in sys.path:
        sys.path.insert(0, p)

import r4_chain as ch  # noqa: E402
import r4_common as rc  # noqa: E402
import r4_eval as re_  # noqa: E402
import r4_lock as lk  # noqa: E402
import run_r4  # noqa: E402
from ._batch import batch_here  # noqa: E402

from .test_r4_options import body_dump, function_def  # noqa: E402

PAIRS = [f'S{k:02d}__S{(k+1) % 16:02d}' for k in range(16)]


class FakeBench:
    def __init__(self):
        self.pair_names = tuple(PAIRS)
        self.pairs = tuple((p, 'A' if k < 8 else 'B', k % 8, *p.split('__')) for k, p in enumerate(PAIRS))


def write_final(root, arm, pair, status=rc.HX, *, u=None, corrupt=False, spy_passed=True):
    run_dir = Path(root)/'runs'/arm/pair
    o = run_dir/'output'
    o.mkdir(parents=True)
    rc.write_json(run_dir/'line.json', dict(line_status=status))
    for rel in ('opened_files_supervisor.json', 't88/opened_files.json'):                         # 复核 B2：锁前核每对的 spy 记录
        rc.write_json(run_dir/rel, dict(passed=spy_passed))
    finalize = ch.final_dir(root, 'bench', arm, pair).parent/'opened_files_finalize.json'
    if not finalize.exists():
        rc.write_json(finalize, dict(passed=True))
    output = None
    if status in rc.OUTPUT_STATUSES:
        u = np.zeros((4, 4, 4, 3)) if u is None else u
        np.savez(o/'final.npz', u=u, state_hash=np.array(rc.array_sha256(u)))
        output = dict(directory=str(o), final_npz_sha256=rc.sha256_file(o/'final.npz'), state_hash=rc.array_sha256(u) if not corrupt else '0'*64)
    return rc.write_json(ch.final_dir(root, 'bench', arm, pair)/'final.json', dict(line_status=status, run_dir=str(run_dir), output=output))


def full_bench(root, statuses=None, corrupt=()):
    for arm in rc.ARM_ORDER:
        for k, pair in enumerate(PAIRS):
            status = (statuses or {}).get((arm, k), rc.FALLBACK_C)
            write_final(root, arm, pair, status, corrupt=(arm, k) in corrupt)


def prelock(root):
    return re_.bench_prelock(root, list(rc.ARM_ORDER), FakeBench(), stage_dir=root, repo=root, require_committed=False, rebuild=lambda *a, **k: None)


@pytest.fixture(autouse=True)
def no_guard(monkeypatch):
    import r4_bench as rb
    monkeypatch.setattr(rb, 'bench_admission', lambda bench, **kw: dict(ok=True))


def resume(root, stop, handling='jia', scope='candidate', arm='B', bz_seen=True, **extra):
    d = Path(root)/re_.RESUME_DIR_REL
    d.mkdir(parents=True, exist_ok=True)
    k = len(list(d.glob('resume_*.json')))
    doc = dict(schema=re_.RESUME_SCHEMA, stop_sha256=rc.sha256_file(stop), handling=handling, scope=scope, arm=arm, signed_by='lead' if handling == 'jia' else None,
               bz_seen=dict(words='看到了', time='2026-11-14 9:00 AM CST') if bz_seen else None, **extra)
    return rc.write_json(d/f'resume_{k}.json', doc)


# ---- 台：建锁之前 -------------------------------------------------------------------------------------------------------------------------------
def test_a_clean_bench_passes_the_prelock_checks(tmp_path):
    full_bench(tmp_path)
    pre = prelock(tmp_path)
    assert pre['excluded'] == {} and pre['undecidable_before_truth'] == {}


def test_ctl_without_an_output_or_any_rejection_stops_before_the_lock(tmp_path):
    full_bench(tmp_path, statuses={(rc.CONTROL, 3): rc.IMPLEMENTATION})
    with pytest.raises(re_.PreLockStop):
        prelock(tmp_path)
    full_bench(tmp_path/'b', statuses={('C2', 5): rc.REJECTED})
    with pytest.raises(re_.PreLockStop, match='hx_provenance_rejected'):
        prelock(tmp_path/'b')


def test_array_mismatch_excludes_a_candidate_but_stops_on_ctl(tmp_path):
    full_bench(tmp_path, corrupt={('A2', 4)})
    assert 'A2' in prelock(tmp_path)['excluded']
    full_bench(tmp_path/'c', corrupt={(rc.CONTROL, 4)})
    with pytest.raises(re_.PreLockStop, match='CTL'):
        prelock(tmp_path/'c')


def test_a_candidates_implementation_error_needs_jia_with_bzs_confirmation(tmp_path):
    full_bench(tmp_path, statuses={('B', 2): rc.IMPLEMENTATION})
    stop = re_.write_stop(tmp_path, dict(arm='B', scope='candidate', reason='soft_argmax_mismatch'))
    with pytest.raises(re_.PreLockStop, match='续跑记录'):                                         # 没有续跑记录：不建锁
        prelock(tmp_path)
    record = resume(tmp_path, stop, bz_seen=False)
    with pytest.raises(re_.PreLockStop, match='BZ 已看到'):                                       # (甲) 可以先签，建锁之前要 BZ 已看到
        prelock(tmp_path)
    record.unlink()
    resume(tmp_path, stop)
    pre = prelock(tmp_path)
    assert 'B' in pre['undecidable_before_truth'] and pre['excluded'] == {}


def test_a_jia_candidate_stopped_mid_arm_is_skipped_not_a_missing_record(tmp_path):
    full_bench(tmp_path, statuses={('B', 2): rc.IMPLEMENTATION})
    for pair in PAIRS[3:]:
        (ch.final_dir(tmp_path, 'bench', 'B', pair)/'final.json').unlink()                       # 队列在第 3 对停下，(甲) 之后这一臂不再跑
    stop = re_.write_stop(tmp_path, dict(arm='B', scope='candidate', reason='mask_error'))
    resume(tmp_path, stop)
    pre = prelock(tmp_path)
    assert 'B' in pre['undecidable_before_truth'] and 'B' not in pre['excluded']
    (ch.final_dir(tmp_path, 'bench', 'C1', PAIRS[4])/'final.json').unlink()
    with pytest.raises(re_.PreLockStop, match='没有定稿记录'):                                     # 别的臂缺记录照旧拒绝
        prelock(tmp_path)


def test_yi_needs_bzs_words_and_scope(tmp_path):
    assert re_.resume_problem(dict(schema=re_.RESUME_SCHEMA, handling='yi', scope='candidate', bz_seen=dict(words='x', time='y')), before_lock=True)
    assert re_.resume_problem(dict(schema=re_.RESUME_SCHEMA, handling='yi', scope='candidate', bz_words='修吧', bz_time='t', bz_seen=dict(words='x', time='y')), before_lock=True)  # 缺改动范围
    assert re_.resume_problem(dict(schema=re_.RESUME_SCHEMA, handling='yi', scope='candidate', bz_words='修吧', bz_time='t', changed_units=['B/p'], why_others_unaffected='只改了 B',
                                   bz_seen=dict(words='x', time='y')), before_lock=True) is None
    assert re_.resume_problem(dict(schema=re_.RESUME_SCHEMA, handling='jia', scope='bench', signed_by='lead'), before_lock=False) is None
    assert re_.resume_problem(dict(schema=re_.RESUME_SCHEMA, handling='jia', scope='bench', signed_by='lead'), before_lock=True)


def test_ctl_or_a_shared_step_under_jia_makes_the_whole_bench_undecidable(tmp_path):
    full_bench(tmp_path)
    stop = re_.write_stop(tmp_path, dict(arm=None, scope='bench', reason='Atropos failed'))
    resume(tmp_path, stop, scope='bench', arm=None)
    with pytest.raises(re_.PreLockStop, match='整台不可判'):
        prelock(tmp_path)


def test_a_missing_final_record_refuses_the_evaluation(tmp_path):
    full_bench(tmp_path)
    (ch.final_dir(tmp_path, 'bench', 'C3', PAIRS[9])/'final.json').unlink()
    with pytest.raises(re_.PreLockStop, match='没有定稿记录'):
        prelock(tmp_path)


# ---- 台：建锁之后 -------------------------------------------------------------------------------------------------------------------------------
def test_after_the_lock_every_outcome_is_final(tmp_path, monkeypatch):
    full_bench(tmp_path)
    monkeypatch.setattr(re_, 'bench_truth', lambda bench, files=None: (_ for _ in ()).throw(FileNotFoundError('truth missing')))
    doc = re_.bench_eval(tmp_path, FakeBench(), read='first', stage_dir=tmp_path, repo=tmp_path, code_commit='c', plan_sha256='p', require_committed=False, rebuild=lambda *a, **k: None)
    assert doc['failed'].startswith('FileNotFoundError') and doc['whole_undecidable'] and lk.lock_path(tmp_path, lk.BENCH).exists()
    with pytest.raises(lk.LockRefused):
        re_.bench_eval(tmp_path, FakeBench(), read='first', stage_dir=tmp_path, repo=tmp_path, code_commit='c', plan_sha256='p', require_committed=False, rebuild=lambda *a, **k: None)


def test_a_prelock_failure_leaves_no_lock(tmp_path):
    full_bench(tmp_path, statuses={(rc.CONTROL, 0): rc.FALLBACK_ID, (rc.CONTROL, 1): rc.REJECTED})
    with pytest.raises(re_.PreLockStop):
        re_.bench_eval(tmp_path, FakeBench(), read='first', stage_dir=tmp_path, repo=tmp_path, code_commit='c', plan_sha256='p', require_committed=False, rebuild=lambda *a, **k: None)
    assert not lk.lock_path(tmp_path, lk.BENCH).exists()


def test_the_second_read_is_the_combination_only_and_needs_the_first(tmp_path, monkeypatch):
    full_bench(tmp_path)
    for pair in PAIRS:
        write_final(tmp_path, 'COMBO-B-C1', pair)
    with pytest.raises(lk.LockRefused):
        re_.bench_eval(tmp_path, FakeBench(), read='second', combo='COMBO-B-C1', stage_dir=tmp_path, repo=tmp_path, code_commit='c', plan_sha256='p', require_committed=False, rebuild=lambda *a, **k: None)
    lk.take_first(lk.lock_path(tmp_path, lk.BENCH), arms=rc.ARM_ORDER, code_commit='c', plan_sha256='p', reason='first')
    rc.write_json(tmp_path/'eval'/'bench_eval_1.json', dict(truth_sha256={'s:seg35': 'a'}))
    rc.write_json(tmp_path/'analysis'/'bench_decision_first.json', dict(combo='COMBO-B-C3', combo_needs_second_read=True))
    with pytest.raises(re_.PreLockStop, match='复核 I4'):                                              # 不是第一读判定计划的组合臂：锁前停，不加第二行
        re_.bench_eval(tmp_path, FakeBench(), read='second', combo='COMBO-B-C1', stage_dir=tmp_path, repo=tmp_path, code_commit='c', plan_sha256='p', require_committed=False, rebuild=lambda *a, **k: None)
    assert len(lk.lines(lk.lock_path(tmp_path, lk.BENCH))) == 1
    (tmp_path/'analysis'/'bench_decision_first.json').unlink()
    rc.write_json(tmp_path/'analysis'/'bench_decision_first.json', dict(combo='COMBO-B-C1', combo_needs_second_read=True))
    monkeypatch.setattr(re_, 'bench_truth', lambda bench, files=None: (_ for _ in ()).throw(RuntimeError('boom')))
    doc = re_.bench_eval(tmp_path, FakeBench(), read='second', combo='COMBO-B-C1', stage_dir=tmp_path, repo=tmp_path, code_commit='c', plan_sha256='p', require_committed=False, rebuild=lambda *a, **k: None)
    assert doc['combo_undecidable'] is True and doc.get('whole_undecidable') is None and len(lk.lines(lk.lock_path(tmp_path, lk.BENCH))) == 2


# ---- 开发集：建锁之前与之后（C25.8-4，D141） -----------------------------------------------------------------------------------------------------------
def test_dev_prelock_failure_spends_nothing_and_after_the_lock_the_read_is_spent(tmp_path, monkeypatch):
    pairs = [f'D{k}' for k in range(19)]
    monkeypatch.setattr(re_, 'run_hx_admit', lambda stage, pair: (None, None))
    with pytest.raises(re_.PreLockStop):
        re_.dev_eval(tmp_path, 'B', 1, pairs=pairs, hs_root=tmp_path/'hs', repo=tmp_path, code_commit='c', plan_sha256='p')
    assert not lk.lock_path(tmp_path, lk.DEV1).exists()
    monkeypatch.setattr(re_, 'dev_prelock', lambda *a, **k: ({p: {} for p in pairs}, {p: np.zeros((2, 2, 2, 3)) for p in pairs}, {}))
    monkeypatch.setattr(re_, 'evaluator', lambda: (_ for _ in ()).throw(RuntimeError('evaluator broke')))
    doc = re_.dev_eval(tmp_path, 'B', 1, pairs=pairs, hs_root=tmp_path/'hs', repo=tmp_path, code_commit='c', plan_sha256='p')
    assert doc['read_spent'] is True and doc['undecidable'] and lk.lock_path(tmp_path, lk.DEV1).exists()
    decision = re_.dev_decide(tmp_path, 1, pairs=pairs)
    assert decision['reading'] == 'undecidable' and decision['read_spent'] is True
    with pytest.raises(lk.LockRefused):
        re_.dev_eval(tmp_path, 'B', 1, pairs=pairs, hs_root=tmp_path/'hs', repo=tmp_path, code_commit='c', plan_sha256='p')


def test_dev_prelock_checklist_names_each_item(tmp_path, monkeypatch):
    pairs = ['D0']
    monkeypatch.setattr(re_, 'run_hx_admit', lambda stage, pair: (_ for _ in ()).throw(ValueError('guard')))
    with pytest.raises(re_.PreLockStop) as info:
        re_.dev_prelock(tmp_path, 'B', pairs=pairs, hs_root=tmp_path, repo=tmp_path, code_problems=['dirty'], plan_problem='plan')
    text = str(info.value)
    for item in ('(1)', '(2)/(5)', '(3)', '(4)', '(6)'):
        assert item in text


# ---- 队列：读真值之前的失败停下（C25.7-3） ------------------------------------------------------------------------------------------------------------
def fake_launch(root, statuses):
    def launch(name, args):
        if args[0] == 'pair':
            role, arm, pair = args[2], args[4], args[6]
            d = Path(root)/('runs' if role == 'bench' else 'dev_runs')/arm/pair
            d.mkdir(parents=True)
            status = statuses.get(pair, rc.HX)
            rc.write_json(d/'line.json', dict(line_status=status))
            return 9 if status in (rc.REJECTED, rc.IMPLEMENTATION) else 0
        return 0
    return launch


def test_the_queue_stops_at_the_first_failure_before_truth_and_writes_a_stop(tmp_path):
    batch_here(tmp_path)
    launched = []
    launch = fake_launch(tmp_path, {PAIRS[2]: rc.IMPLEMENTATION})
    code = run_r4.queue_arm('bench', 'B', root=tmp_path, pairs=PAIRS, launch_unit=lambda n, a: (launched.append(a), launch(n, a))[1], gate=lambda own: (True, {}), gap=0,
                            repo=tmp_path, require_committed=False)
    assert code == 9 and len([a for a in launched if a[0] == 'pair']) == 3
    stops = re_.stop_files(tmp_path)
    assert len(stops) == 1 and rc.read_json(stops[0])['scope'] == 'candidate'
    assert run_r4.queue_arm('bench', 'C1', root=tmp_path, pairs=PAIRS, launch_unit=launch, gate=lambda own: (True, {}), gap=0, repo=tmp_path, require_committed=False) == 9   # 整个队列停着，直到续跑记录
    resume(tmp_path, stops[0], arm='B', bz_seen=False)
    assert run_r4.queue_arm('bench', 'B', root=tmp_path, pairs=PAIRS, launch_unit=launch, gate=lambda own: (True, {}), gap=0, repo=tmp_path, require_committed=False) == 0     # 不碰失败的那一臂
    assert run_r4.queue_arm('bench', 'C1', root=tmp_path, pairs=PAIRS, launch_unit=fake_launch(tmp_path, {}), gate=lambda own: (True, {}), gap=0, repo=tmp_path, require_committed=False) == 0


def test_the_queue_respects_hold_and_the_gate_and_never_reruns_a_half_done_pair(tmp_path):
    batch_here(tmp_path)
    (tmp_path/'hold').touch()
    assert run_r4.queue_arm('bench', 'B', root=tmp_path, pairs=PAIRS, launch_unit=fake_launch(tmp_path, {}), gate=lambda own: (True, {}), gap=0, repo=tmp_path, require_committed=False) == 3
    (tmp_path/'hold').unlink()
    assert run_r4.queue_arm('bench', 'B', root=tmp_path, pairs=PAIRS, launch_unit=fake_launch(tmp_path, {}), gate=lambda own: (False, {}), gap=0, repo=tmp_path, require_committed=False) == 3
    (tmp_path/'runs'/'B'/PAIRS[0]).mkdir(parents=True)
    assert run_r4.queue_arm('bench', 'B', root=tmp_path, pairs=PAIRS, launch_unit=fake_launch(tmp_path, {}), gate=lambda own: (True, {}), gap=0, repo=tmp_path, require_committed=False) == 9


def test_ctl_failures_belong_to_the_whole_bench_and_dev_ones_to_feasibility():
    assert run_r4.stop_scope('bench', rc.CONTROL) == 'bench' and run_r4.stop_scope('bench', 'A1') == 'candidate'
    assert run_r4.stop_scope('bench', 'COMBO-B-C1') == 'combo' and run_r4.stop_scope('dev', 'B') == 'feasibility'


def test_unit_names_and_argv():
    name = run_r4.unit_name('bench', 'COMBO-A1-C3', '0443', '0444')
    assert name.startswith('tag-r4-') and 'seq-vp' not in name
    argv = run_r4.unit_argv(name, ['pair', '--role', 'bench'])
    assert argv[:3] == ['systemd-run', '--user', f'--unit={name}'] and '--property=MemoryMax=12G' in argv and any(a.startswith('--setenv=PYTHONPATH=') for a in argv)


def test_machine_gate_problems(tmp_path):
    (tmp_path/'meminfo').write_text('MemAvailable: 2097152 kB\n')
    (tmp_path/'loadavg').write_text('2.0 1 1 1/1 1\n')

    class Done:
        def __init__(self, out):
            self.stdout, self.returncode = out, 0

    def run(argv, **kw):
        return Done('tag-hs-queue.service loaded active running\n' if argv[0] == 'systemctl' else '1234\n')

    class Disk:
        free = 10*2**30
    problems, _ = run_r4.machine_gate('tag-r4-queue', run=run, meminfo=tmp_path/'meminfo', loadavg=tmp_path/'loadavg', disk=lambda p: Disk())
    assert len(problems) == 6


# ---- 退路一：每对至多一次，各臂共用（C25.6-12） -----------------------------------------------------------------------------------------------------
def test_configc_fallback_is_hes_but_for_the_result_path():
    import ast
    import run_hx
    is_path = lambda s: isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == 'd'          # noqa: E731
    assert body_dump(function_def(ch, 'configc_fallback'), is_path) == body_dump(function_def(run_hx, 'configc_fallback'), is_path)
    copy = function_def(ch, 'configc_fallback')
    assert [a.arg for a in copy.args.kwonlyargs] == ['launcher', 'certify', 'config', 'suffix', 'shared_dir']
    assert any(ast.unparse(s) == 'd = Path(shared_dir)' for s in copy.body)


@dataclass
class Cert:
    status: str
    min_jd_lower_bound: float
    n_infeasible: int
    n_uncertain: int
    state_hash: str
    n_cells_checked: int = 1
    witness_cell: object = None
    witness_reference_point: object = None


class Grid:
    shape = (4, 4, 4)


class Pin:
    grid = Grid()


def test_configc_once_reuses_and_certifies_again(tmp_path):
    from jd_curl.three_d.feasibility import PrescreenCertificateConfig
    ctx = dict(configc_root=str(tmp_path/'configC'), pair='P', out=str(tmp_path/'run'), settings=dict(configc={}))
    shared = tmp_path/'configC'/'P'
    (shared/'run').mkdir(parents=True)
    u = np.zeros((4, 4, 4, 3))
    np.savez(shared/'run'/'final_native.npz', u=u)
    sha = rc.sha256_file(shared/'run'/'final_native.npz')
    rc.write_json(shared/'fallback.json', dict(file='final_native.npz', final_native_sha256=sha, state_hash=rc.array_sha256(u),
                                               candidates=[dict(file='final_native.npz', path=str(shared/'run'/'final_native.npz'), recorded_sha256=sha, passed=True)]))
    calls = []

    def certify(v, grid):
        calls.append(rc.array_sha256(v))
        return Cert('certified_feasible', 1., 0, 0, rc.array_sha256(v))
    got, doc, rec = ch.configc_once(ctx, Pin(), launcher=None, certify=certify, config=PrescreenCertificateConfig(), suffix='')
    assert np.array_equal(got, u) and rec['configC_source'] == 'reused' and calls == [rc.array_sha256(u)] and rec['configC_result_sha256'] == rc.sha256_file(shared/'fallback.json')
    got, _, rec = ch.configc_once(ctx, Pin(), launcher=None, certify=lambda v, g: Cert('certified_infeasible', -1., 3, 0, rc.array_sha256(v)), config=PrescreenCertificateConfig())
    assert got is None and 'did not pass' in rec['error']
    (shared/'fallback.json').unlink()
    with pytest.raises(ch.SharedConfigCUnavailable):
        ch.configc_once(ctx, Pin(), launcher=None, certify=certify, config=PrescreenCertificateConfig())


def test_r4_t88_implementation_errors_stop_the_queue():
    kind, info = ch.t88_status_decision(dict(status=ch.T88_IMPLEMENTATION, reason='soft_argmax_mismatch: 3 voxels'), 0)
    assert kind == 'implementation' and info['kind'] == 'soft_argmax_mismatch'
    assert ch.t88_status_decision(dict(status='provenance_rejected', reason='x'), 0)[0] == 'rejected'
    assert ch.t88_status_decision(dict(status='segmentation_qc', reason='x'), 0)[0] == 'fallback'


def test_contexts_check_the_identity_except_in_tests(tmp_path):
    ctx = ch.make_context('bench', 'A1', PAIRS[0], root=tmp_path, stage_dir=tmp_path, check_identity=False, test=True)
    assert ctx['test_context'] and ctx['settings']['r4']['spec']['sigma_small'] == .42 and ctx['settings']['t88']['pre']['sigma_field'] == .5
    assert ch.make_context('dev', 'C2', 'P', root=tmp_path, stage_dir=tmp_path, check_identity=False, test=True)['settings']['t88']['pre']['iterations'] == [50, 50, 100, 1200]
    with pytest.raises((ValueError, rc.Refused)):
        ch.make_context('bench', 'CTL', PAIRS[0], root=tmp_path, stage_dir=tmp_path, check_identity=False)
    with pytest.raises(rc.Refused):
        ch.make_context('bench', 'CTL', PAIRS[0], root=tmp_path, stage_dir=tmp_path)                 # 真实运行只写在 <R4> 之下
