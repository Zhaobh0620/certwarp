"""R4 的决定代码与读一次锁（合同 C25.5、C25.7、C25.8-3、-4、-5，C25.9-4，C25.10-4；纯函数与临时目录，不读任何受试者文件）。"""
import ast
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT/'experiments'/'r4',):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import r4_common as rc  # noqa: E402
import r4_decide as rd  # noqa: E402
import r4_lock as lk  # noqa: E402

PAIRS = [f'p{k:02d}' for k in range(16)]
SUBJECTS = [f'A{k}' for k in range(8)]+[f'B{k}' for k in range(8)]
PAIR_SUBJECTS = {p: ((('A' if k < 8 else 'B')+str(k % 8)), (('A' if k < 8 else 'B')+str((k+1) % 8))) for k, p in enumerate(PAIRS)}


def rows(seg35, seg4=None, sdlogj=.07, status=rc.HX):
    seg4 = [.9]*16 if seg4 is None else seg4
    return {p: dict(line_status=status, metrics=dict(seg35=a, seg4=b, sdlogj_official=sdlogj)) for p, a, b in zip(PAIRS, seg35, seg4)}


CTL = rows([.85]*16)


def arm_with(delta, n_pos=16, d4=0., sdlogj=.07):
    """均值约为 delta，恰 n_pos 对为正：n_pos 对取 a > 0，其余取 −.001。"""
    a = (16*delta+.001*(16-n_pos))/n_pos
    d = [a]*n_pos+[-.001]*(16-n_pos)
    return rows([.85+x for x in d], [.9+d4]*16, sdlogj=sdlogj)


def screen(arm):
    return rd.screen(arm, CTL, PAIRS, PAIR_SUBJECTS, SUBJECTS)


# ---- 筛门（C25.7-2） --------------------------------------------------------------------------------------------------------------------------
def test_screen_boundaries_are_inclusive():
    zero = rows([0.]*16, [0.]*16)                                    # Δ 恰为字面值（浮点减法不引入误差）
    r = rd.screen(rows([.0015]*16, [-.001]*16, sdlogj=.10), zero, PAIRS, PAIR_SUBJECTS, SUBJECTS)
    assert r['verdict'] == rd.PASS and all(r['gate'].values())
    assert rd.screen(rows([.0014]*16, [0.]*16), zero, PAIRS, PAIR_SUBJECTS, SUBJECTS)['gate']['mean_seg35'] is False
    assert rd.screen(rows([.0015]*16, [-.0011]*16), zero, PAIRS, PAIR_SUBJECTS, SUBJECTS)['gate']['mean_seg4'] is False
    assert rd.screen(rows([.0015]*16, [0.]*16, sdlogj=.1001), zero, PAIRS, PAIR_SUBJECTS, SUBJECTS)['gate']['sdlogj'] is False


def test_screen_counts_pairs_with_positive_delta():
    assert screen(arm_with(.002, n_pos=10))['gate']['count_seg35'] is True
    r = screen(arm_with(.002, n_pos=9))
    assert r['gate']['count_seg35'] is False and r['verdict'] == rd.FAIL and r['delta_seg35']['n_positive'] == 9


def test_screen_needs_16_outputs_and_an_undefined_metric_is_undecidable():
    arm = rows([.86]*16)
    arm['p03'] = dict(line_status=rc.IMPLEMENTATION, metrics=None)
    assert screen(arm)['verdict'] == rd.FAIL
    arm = rows([.86]*16)
    arm['p03']['metrics']['seg35'] = float('nan')
    assert screen(arm)['verdict'] == rd.UNDECIDABLE


def test_screen_records_the_paired_standard_error_and_block_interval():
    r = screen(arm_with(.002, n_pos=12))
    assert r['delta_seg35']['se'] is not None and len(r['block_seg35']) == 2 and r['block_seg35'][0] <= r['delta_seg35']['mean'] <= r['block_seg35'][1]


# ---- 排序、组合、顺序 O（C25.5、C25.7-5、-6） -----------------------------------------------------------------------------------------------------
def results(**means):
    return {a: dict(verdict=rd.PASS, delta_seg35=dict(mean=m)) for a, m in means.items()}


def test_ranking_ties_follow_the_option_order():
    assert rd.ranked(results(C3=.002, B=.002, A2=.003)) == ['A2', 'B', 'C3']


def test_combination_takes_the_top_two_compatible():
    assert rd.planned_combination(rd.ranked(results(B=.004, C1=.003, A1=.002))) == 'COMBO-B-C1'
    assert rd.planned_combination(rd.ranked(results(A1=.004, A2=.003, C3=.002))) == 'COMBO-A1-C3'      # A1, A2 不相容
    assert rd.planned_combination(rd.ranked(results(A1=.004, A2=.003))) is None                        # 找不到相容的：不做组合
    assert rd.planned_combination(rd.ranked(results(B=.004))) is None


def test_combination_used_only_with_the_margin_inclusive():
    res = results(B=.0045, C1=.003)
    assert rd.combination_adopted('COMBO-B-C1', dict(verdict=rd.PASS, delta_seg35=dict(mean=.005)), res) is True     # .0045 + .0005 = .005
    assert rd.combination_adopted('COMBO-B-C1', dict(verdict=rd.PASS, delta_seg35=dict(mean=.00499)), res) is False
    assert rd.combination_adopted('COMBO-B-C1', dict(verdict=rd.FAIL, delta_seg35=dict(mean=.01)), res) is False


def test_order_o_with_and_without_the_combination():
    res = results(B=.004, C1=.003, A1=.002)
    assert rd.order_o(res) == ['B', 'C1', 'A1']
    assert rd.order_o(res, 'COMBO-B-C1', dict(verdict=rd.PASS, delta_seg35=dict(mean=.006))) == ['COMBO-B-C1', 'B', 'C1', 'A1']
    assert rd.order_o(res, 'COMBO-B-C1', dict(verdict=rd.PASS, delta_seg35=dict(mean=.0041))) == ['B', 'C1', 'A1']


def test_arm_spec_union_and_incompatibility():
    assert rc.arm_spec('COMBO-A1-C3')['sigma_field'] == .65 and rc.arm_spec('COMBO-A1-C3')['sigma_small'] == .42
    with pytest.raises(ValueError):
        rc.arm_spec('COMBO-A1-A2')
    assert rc.families('COMBO-A2-B') == frozenset(('A', 'B'))


# ---- 开发集：第 1 选、次选、顺延（C25.7-6、C25.8-3、-5） --------------------------------------------------------------------------------------------
F, NF = rd.FEASIBLE, rd.NOT_FEASIBLE


def test_first_choice_is_the_first_feasible_and_unknown_items_go_to_feasibility():
    o = ['A1', 'B', 'C1']
    assert rd.next_dev_step(o, {}, []) == dict(action='feasibility', item='A1', read_number=1)
    assert rd.next_dev_step(o, {'A1': NF}, [])['item'] == 'B'
    assert rd.next_dev_step(o, {'A1': NF, 'B': F}, []) == dict(action='read', item='B', read_number=1)
    assert rd.next_dev_step(o, {'A1': NF, 'B': NF, 'C1': NF}, [])['action'] == 'stop'


def test_runner_up_skips_the_same_family():
    o = ['A1', 'A2', 'C3', 'B']
    step = rd.next_dev_step(o, {'A1': F}, [('A1', rd.FAIL)])
    assert step == dict(action='feasibility', item='C3', read_number=2)                         # A2 同族跳过；A 与 C3 分族（修订 10c）
    assert rd.next_dev_step(['A2', 'A1'], {'A2': F}, [('A2', rd.FAIL)])['action'] == 'stop'      # 找不到次选：不用第 2 次


def test_runner_up_after_a_combination_skips_its_families():
    o = ['COMBO-A1-C3', 'A1', 'C3', 'A2', 'COMBO-A2-B', 'B']
    step = rd.next_dev_step(o, {'COMBO-A1-C3': F}, [('COMBO-A1-C3', rd.FAIL)])
    assert step['item'] == 'B'
    assert rd.next_dev_step(['C1', 'COMBO-A1-C1', 'A1'], {'C1': F}, [('C1', rd.FAIL)])['item'] == 'A1'       # 含 C1 的组合跳过


def test_reading_outcomes_stop_or_go_on():
    o = ['B', 'C1']
    assert rd.next_dev_step(o, {'B': F}, [('B', rd.PASS)])['action'] == 'stop'
    assert rd.next_dev_step(o, {'B': F}, [('B', rd.UNDECIDABLE)])['action'] == 'stop'
    assert rd.next_dev_step(o, {'B': F, 'C1': F}, [('B', rd.FAIL)]) == dict(action='read', item='C1', read_number=2)
    for second in (rd.PASS, rd.FAIL, rd.UNDECIDABLE):
        assert rd.next_dev_step(o, {'B': F, 'C1': F}, [('B', rd.FAIL), ('C1', second)])['action'] == 'stop'


def test_an_implementation_error_in_feasibility_stops_before_any_shift():
    assert rd.next_dev_step(['B', 'C1'], {'B': rd.FEAS_STOP}, [])['action'] == 'stop'


def test_feasibility_verdict():
    good = {f'd{k}': dict(line_status=rc.HX, sdlogj_official=.07) for k in range(19)}
    assert rd.feasibility_verdict(good)['verdict'] == F
    assert rd.feasibility_verdict(dict(good, d0=dict(line_status=rc.FALLBACK_ID, sdlogj_official=.0)))['verdict'] == F
    assert rd.feasibility_verdict({k: dict(v, sdlogj_official=.11) for k, v in good.items()})['verdict'] == NF
    assert rd.feasibility_verdict(dict(good, d3=dict(line_status=rc.IMPLEMENTATION)))['verdict'] == rd.FEAS_STOP
    assert rd.feasibility_verdict(dict(good, d3=dict(line_status=rc.REJECTED)))['verdict'] == rd.FEAS_STOP


# ---- 开发集的门（C25.8-4） -----------------------------------------------------------------------------------------------------------------------
DEV = [f'd{k:02d}' for k in range(19)]
INDEX = {p: k for k, p in enumerate(DEV)}


def dev_rows(seg35, seg4=.92, sdlogj=.07):
    return {p: dict(line_status=rc.HX, metrics=dict(seg35=v, seg4=seg4, sdlogj_official=sdlogj)) for p, v in zip(DEV, seg35)}


HS = dev_rows([.88]*19)


def test_dev_gate_boundaries(monkeypatch):
    # 二进制有限小数的门槛：19 个相同的值求均值不引入误差，边界可以逐位核（门槛的值另核在 test_thresholds_are_the_contracts）
    monkeypatch.setattr(rc, 'DEV_MEAN35', 2**-8)
    monkeypatch.setattr(rc, 'DEV_MEAN4', -2**-10)
    monkeypatch.setattr(rc, 'SDLOGJ_MAX', .125)
    zero = dev_rows([0.]*19, seg4=0.)
    r = rd.dev_reading(dev_rows([2**-8]*19, seg4=-2**-10, sdlogj=.125), zero, DEV, INDEX)
    assert r['reading'] == rd.PASS and r['better'] is True
    assert rd.dev_reading(dev_rows([2**-8-2**-12]*19, seg4=0.), zero, DEV, INDEX)['reading'] == rd.FAIL
    assert rd.dev_reading(dev_rows([2**-8]*19, seg4=-2**-10-2**-14), zero, DEV, INDEX)['gate']['mean_seg4'] is False
    assert rd.dev_reading(dev_rows([2**-8]*19, seg4=0., sdlogj=.125+2**-20), zero, DEV, INDEX)['gate']['sdlogj'] is False
    spread = [2**-8+(.02 if k % 2 else -.02) for k in range(18)]+[2**-8]
    r = rd.dev_reading(dev_rows(spread, seg4=0.), zero, DEV, INDEX)
    assert r['block_seg35'][0] < 0 and r['better'] is False                                     # "更好"要块下界 > 0


def test_thresholds_are_the_contracts():
    assert (rc.SCREEN_MEAN35, rc.SCREEN_COUNT, rc.SCREEN_MEAN4, rc.SDLOGJ_MAX, rc.COMBO_MARGIN) == (.0015, 10, -.001, .10, .0005)
    assert (rc.DEV_MEAN35, rc.DEV_LOWER, rc.DEV_MEAN4, rc.BETTER_LOWER, rc.CONTROL_TOL) == (.003, -.003, -.001, 0., 2e-4)
    assert (rc.BENCH_BOOT_N, rc.BENCH_BOOT_SEED, rc.DEV_BOOT_N, rc.DEV_BOOT_SEED) == (10000, 20261113, 10000, 20261105)


def test_dev_undecidable_after_the_lock():
    arm = dev_rows([.883]*19)
    del arm['d04']
    assert rd.dev_reading(arm, HS, DEV, INDEX)['reading'] == rd.UNDECIDABLE
    arm = dev_rows([.883]*19)
    arm['d04']['metrics']['sdlogj_official'] = None
    assert rd.dev_reading(arm, HS, DEV, INDEX)['reading'] == rd.UNDECIDABLE


def test_control_tolerance_is_inclusive():
    a = {'x': dict(seg35=.8, seg4=.9, sdlogj_official=.07)}
    assert rd.control_mismatches(a, {'x': dict(seg35=.8+1.5e-4, seg4=.9, sdlogj_official=.07)}) == []
    assert rd.control_mismatches(a, {'x': dict(seg35=.8+2.5e-4, seg4=.9, sdlogj_official=.07)})[0][1] == 'seg35'
    assert rd.control_mismatches(a, {'x': dict(seg35=None, seg4=.9, sdlogj_official=.07)})


# ---- 块 bootstrap -----------------------------------------------------------------------------------------------------------------------------
def test_dev_block_interval_is_hs_statement_for_statement():
    sys.path.insert(0, str(ROOT/'experiments'/'hs'))
    for d in ('fl', 'mc0', 'ef'):
        sys.path.insert(0, str(ROOT/'experiments'/d))
    import decide_hs
    mine, theirs = ast.parse(inspect.getsource(rd.block_interval)).body[0], ast.parse(inspect.getsource(decide_hs.block_interval)).body[0]
    assert ast.dump(ast.Module(mine.body, [])) == ast.dump(ast.Module(theirs.body, []))
    assert inspect.signature(rd.block_interval).parameters['n'].default == decide_hs.hc.BOOT_N
    assert inspect.signature(rd.block_interval).parameters['seed'].default == decide_hs.hc.BOOT_SEED == 20261105
    assert inspect.signature(rd.block_interval).parameters['n_subjects'].default == decide_hs.hc.N_SUBJECTS == 20


def test_ring_bootstrap_weights_pairs_by_their_subject_ids():
    x = np.random.default_rng(0).normal(size=16)
    ps = [PAIR_SUBJECTS[p] for p in PAIRS]
    a = rd.ring_block_interval(x, ps, SUBJECTS, n=2000)
    perm = np.random.default_rng(1).permutation(16)
    b = rd.ring_block_interval(x[perm], [ps[k] for k in perm], SUBJECTS, n=2000)
    assert np.allclose(a, b)                                                                     # 按 id 取权：对的列法不影响
    assert PAIR_SUBJECTS['p07'] == ('A7', 'A0') and PAIR_SUBJECTS['p15'] == ('B7', 'B0')        # 回绕在折内，不把 A s7 与 B s0 配成一对
    # 一个独立的写法：counts[固定]·counts[移动]，权和为 0 重抽
    rng, out, pos = np.random.default_rng(rc.BENCH_BOOT_SEED), [], {s: k for k, s in enumerate(SUBJECTS)}
    while len(out) < 2000:
        c = np.bincount(rng.integers(0, 16, 16), minlength=16)
        w = np.array([c[pos[f]]*c[pos[m]] for f, m in ps])
        if w.sum():
            out.append(float((w*x).sum()/w.sum()))
    assert np.allclose(a, np.percentile(out, [2.5, 97.5]))


# ---- 台的判定：不可判的两层 ----------------------------------------------------------------------------------------------------------------------
def decide(first, **kw):
    return rd.bench_decision(first, pairs=PAIRS, pair_subjects=PAIR_SUBJECTS, subjects=SUBJECTS, **kw)


def test_whole_bench_undecidable_after_the_lock():
    bad = dict(CTL, p02=dict(line_status=rc.HX, metrics=dict(seg35=None, seg4=.9, sdlogj_official=.07)))
    assert decide({rc.CONTROL: bad})['reading'] == rd.UNDECIDABLE
    assert decide(None)['reading'] == rd.UNDECIDABLE
    assert decide({rc.CONTROL: CTL}, whole_undecidable='评测侧的指标比对不符')['stop'] is True


def test_candidate_layers_before_and_after_the_truth():
    first = {rc.CONTROL: CTL, 'B': rows([.854]*16), 'C1': rows([.856]*16), 'C2': rows([.86]*16)}
    bad = rows([.86]*16)
    bad['p01']['metrics']['seg35'] = float('nan')
    first['C3'] = bad
    d = decide(first, excluded={'C2': '数组不符'}, undecidable_before_truth={'A1': '(甲)'})
    r = d['results']
    assert r['C2']['verdict'] == rd.EXCLUDED and r['A1']['verdict'] == rd.UNDECIDABLE and r['A1']['layer'] == 'before_truth'
    assert r['C3']['verdict'] == rd.UNDECIDABLE and r['A2']['verdict'] == rd.UNDECIDABLE
    assert d['ranked'] == ['C1', 'B'] and d['combo'] == 'COMBO-B-C1' and d['combo_needs_second_read'] is True
    assert d['order'] == ['C1', 'B']


def test_no_candidate_passes_stops():
    d = decide({rc.CONTROL: CTL, **{a: rows([.85]*16) for a in rc.CANDIDATES}})
    assert d['reading'] == rd.STOP_NO_CANDIDATE and d['stop'] is True and d['order'] == []


# ---- 读一次锁（C25.6-11、C25.9-4） ---------------------------------------------------------------------------------------------------------------
def test_lock_first_then_only_one_planned_second_line(tmp_path):
    path = lk.lock_path(tmp_path, lk.BENCH)
    lk.check_absent(path)
    lk.take_first(path, arms=rc.ARM_ORDER, code_commit='c', plan_sha256='p', reason='第一读')
    with pytest.raises(lk.LockRefused):
        lk.check_absent(path)
    with pytest.raises(lk.LockRefused):
        lk.take_first(path, arms=rc.ARM_ORDER, code_commit='c', plan_sha256='p', reason='again')
    with pytest.raises(lk.LockRefused):
        lk.take_second(path, 'B', code_commit='c', plan_sha256='p', truth_sha256={})            # 第二行只能是组合臂
    lk.take_second(path, 'COMBO-B-C1', code_commit='c', plan_sha256='p', truth_sha256={'s': 'x'})
    with pytest.raises(lk.LockRefused):
        lk.take_second(path, 'COMBO-B-C3', code_commit='c', plan_sha256='p', truth_sha256={})    # 不再有第三行
    rows_ = lk.lines(path)
    assert [r['read'] for r in rows_] == ['first', 'second'] and all(set(r) >= {'utc', 'local_time', 'arms', 'code_commit', 'plan_sha256', 'reason'} for r in rows_)


def test_second_read_needs_the_first(tmp_path):
    with pytest.raises(lk.LockRefused):
        lk.take_second(lk.lock_path(tmp_path, lk.BENCH), 'COMBO-B-C1', code_commit='c', plan_sha256='p', truth_sha256={})
    assert not lk.lock_path(tmp_path, lk.BENCH).exists()


def test_lock_lines_are_json(tmp_path):
    path = lk.lock_path(tmp_path, lk.DEV1)
    lk.take_first(path, arms=['B'], code_commit='c', plan_sha256='p', reason='开发集第 1 次读数')
    assert json.loads(path.read_text(encoding='utf-8').splitlines()[0])['arms'] == ['B']
