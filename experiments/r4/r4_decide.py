"""R4 的决定代码（合同 C25.5、C25.7、C25.8；plan.md 第 6、7 节）：纯函数，不读任何文件。

台上（第一读、组合臂的第二读之后）：Δ = 臂 − CTL（逐对，方法级，退路按退路的值计入）；筛门；排序；组合；顺序 O；不可判的两层；配对标准误、块区间（两个折内 8 人环）。
开发集：可行性的判定；第 1 选、次选（同族跳过）与顺延；读数的门（Δ = 臂 − HS）；块区间沿用 HS 的那个（decide_hs.block_interval 的逐语句复制，测试对照）。
"""
import math

import numpy as np

import r4_common as rc

MEASURES = ('seg35', 'seg4', 'hd95', 'sdlogj_official', 'sdlogj_brain', 'small_six', 'large_six', 'other_23', 'cortex', 'white_matter')
PASS, FAIL, UNDECIDABLE, EXCLUDED = 'pass', 'not_pass', 'undecidable', 'excluded'
STOP_NO_CANDIDATE = 'stop_no_candidate'


# ---- 块 bootstrap ---------------------------------------------------------------------------------------------------------------------------
def block_interval(values, index, *, n=rc.DEV_BOOT_N, seed=rc.DEV_BOOT_SEED, n_subjects=rc.DEV_N_SUBJECTS):
    """(lo, hi): the 95 % subject-block bootstrap interval of the mean over pairs (runs/20261105-fd2/recompute.py block; experiments/p1/p1_core.block_interval): the 20 subjects resampled, pair i = (subject i, i + 1) weighs the product of their counts.  index: each value's pair number 0..18."""
    x, index = np.asarray(values, dtype=np.float64), np.asarray(index)
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        counts = np.bincount(rng.integers(0, n_subjects, n_subjects), minlength=n_subjects)
        w = counts[index]*counts[index+1]
        if w.sum():
            out.append(float((w*x).sum()/w.sum()))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def ring_block_interval(values, pair_subjects, subjects, *, n=rc.BENCH_BOOT_N, seed=rc.BENCH_BOOT_SEED):
    """台的块区间（C25.7-4 修订 10b）：16 人有放回重抽 16 人；一对的权 = counts[固定图受试者]·counts[移动图受试者]，按每对的两个受试者 id 取（不按相邻下标）；权和为 0 就重抽。
    pair_subjects: 每个值那一对的 (fixed id, moving id)；subjects: 16 人（次序固定）。"""
    x = np.asarray(values, dtype=np.float64)
    position = {s: k for k, s in enumerate(subjects)}
    fixed = np.array([position[f] for f, _ in pair_subjects])
    moving = np.array([position[m] for _, m in pair_subjects])
    m = len(subjects)
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        counts = np.bincount(rng.integers(0, m, m), minlength=m)
        w = counts[fixed]*counts[moving]
        if w.sum():
            out.append(float((w*x).sum()/w.sum()))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def paired_summary(deltas):
    """均值、配对标准误（sd / √n，ddof 1）与配对 t 区间（只描述）。"""
    x = np.asarray(deltas, dtype=np.float64)
    n = len(x)
    mean = float(x.mean())
    se = float(x.std(ddof=1)/math.sqrt(n)) if n > 1 else None
    from scipy.stats import t as student
    half = None if se is None else float(student.ppf(.975, n-1)*se)
    return dict(n=n, mean=mean, se=se, t_interval=None if half is None else [mean-half, mean+half], n_positive=int((x > 0).sum()))


# ---- 一臂的行 -------------------------------------------------------------------------------------------------------------------------------
def defined(row, measures=('seg35', 'seg4', 'sdlogj_official')):
    m = (row or {}).get('metrics') or {}
    return all(isinstance(m.get(k), (int, float)) and math.isfinite(m.get(k)) for k in measures)


def has_output(row):
    return (row or {}).get('line_status') in rc.OUTPUT_STATUSES


def deltas(arm_rows, ref_rows, pairs, measure):
    return [arm_rows[p]['metrics'][measure]-ref_rows[p]['metrics'][measure] for p in pairs]


# ---- 筛门（C25.7-2） ------------------------------------------------------------------------------------------------------------------------
def screen(arm_rows, ctl_rows, pairs, pair_subjects, subjects):
    """一个候选的筛门（含边界，全部满足才算过）。有对指标无定义：这一候选不可判（读真值之后，算没过筛门，C25.7-3）。"""
    if len(pairs) != rc.N_BENCH_PAIRS:
        raise ValueError(f'台上是 {rc.N_BENCH_PAIRS} 对，不是 {len(pairs)}')
    outputs = sum(has_output(arm_rows.get(p)) for p in pairs)
    undefined = [p for p in pairs if has_output(arm_rows.get(p)) and not defined(arm_rows.get(p))]
    if undefined or outputs != len(pairs):
        reason = f'指标无定义的对 {undefined}' if undefined else f'有输出的对 {outputs}/{len(pairs)}'
        return dict(verdict=UNDECIDABLE if undefined else FAIL, reason=reason, outputs=outputs, gate=None)
    d35, d4 = deltas(arm_rows, ctl_rows, pairs, 'seg35'), deltas(arm_rows, ctl_rows, pairs, 'seg4')
    s35, s4 = paired_summary(d35), paired_summary(d4)
    sdlogj = float(np.mean([arm_rows[p]['metrics']['sdlogj_official'] for p in pairs]))
    gate = dict(mean_seg35=s35['mean'] >= rc.SCREEN_MEAN35, count_seg35=s35['n_positive'] >= rc.SCREEN_COUNT, mean_seg4=s4['mean'] >= rc.SCREEN_MEAN4,
                sdlogj=sdlogj <= rc.SDLOGJ_MAX, outputs=outputs == len(pairs))
    return dict(verdict=PASS if all(gate.values()) else FAIL, gate=gate, outputs=outputs, delta_seg35=s35, delta_seg4=s4, sdlogj_official_mean=sdlogj,
                block_seg35=ring_block_interval(d35, [pair_subjects[p] for p in pairs], subjects), block_seg4=ring_block_interval(d4, [pair_subjects[p] for p in pairs], subjects),
                per_pair={p: dict(delta_seg35=a, delta_seg4=b) for p, a, b in zip(pairs, d35, d4)}, describe=describe(arm_rows, ctl_rows, pairs))


DESCRIBED = ('small_six', 'large_six', 'other_23', 'cortex', 'white_matter', 'hd95', 'sdlogj_brain', 'sdlogj_official')


def describe(arm_rows, ctl_rows, pairs):
    """C25.7-4 的描述项（不参与判定）：逐组（小六个、大六个、其余 23 个、皮层、白质）、HD95、脑内与官方 SDlogJ 的平均差；修复前场的 seg35 / seg4（D136）；退路数；修复轮数；任务时钟。"""
    def mean_delta(key):
        values = [arm_rows[p]['metrics'].get(key)-ctl_rows[p]['metrics'].get(key) for p in pairs
                  if isinstance((arm_rows[p].get('metrics') or {}).get(key), (int, float)) and isinstance((ctl_rows[p].get('metrics') or {}).get(key), (int, float))]
        return dict(mean=float(np.mean(values)), n=len(values)) if values else None

    def mean_of(rows, get):
        values = [get(rows[p]) for p in pairs]
        values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
        return float(np.mean(values)) if values else None
    out = {f'delta_{k}': mean_delta(k) for k in DESCRIBED}
    for side, rows in (('arm', arm_rows), ('ctl', ctl_rows)):
        out[f'pre_repair_seg35_{side}'] = mean_of(rows, lambda r: (r.get('pre_repair') or {}).get('seg35'))
        out[f'pre_repair_seg4_{side}'] = mean_of(rows, lambda r: (r.get('pre_repair') or {}).get('seg4'))
        out[f'fallbacks_{side}'] = {s: sum(rows[p].get('line_status') == s for p in pairs) for s in (rc.FALLBACK_C, rc.FALLBACK_ID)}
        out[f'repair_rounds_mean_{side}'] = mean_of(rows, lambda r: r.get('repair_rounds'))
        out[f'task_clock_mean_{side}'] = mean_of(rows, lambda r: r.get('task_clock'))
    return out


def ranked(results):
    """过筛门的候选按 Δseg35 均值从高到低；并列按 A1、A2、B、C1、C2、C3（C25.7-5）。"""
    passed = [a for a in rc.CANDIDATES if (results.get(a) or {}).get('verdict') == PASS]
    return sorted(passed, key=lambda a: (-results[a]['delta_seg35']['mean'], rc.CANDIDATES.index(a)))


def planned_combination(order):
    """两个以上过筛门时：最高者与排在它后面的第一个与它相容的候选（C25.5-1）；找不到就 None（不做组合、不做第二读）。"""
    if len(order) < 2:
        return None
    top = order[0]
    for other in order[1:]:
        if rc.compatible(top, other):
            return rc.combo_name(top, other)
    return None


def combination_adopted(combo, combo_result, results):
    """组合也要过筛门，而且 Δseg35 比其中较好的单项高 ≥ +.0005（含边界）才用（C25.5-3）。"""
    if combo is None or (combo_result or {}).get('verdict') != PASS:
        return False
    best = max(results[p]['delta_seg35']['mean'] for p in rc.combo_parts(combo))
    return combo_result['delta_seg35']['mean'] >= best+rc.COMBO_MARGIN


def order_o(results, combo=None, combo_result=None):
    """顺序 O：过筛门的单项，加上被采用的组合；按台上 Δseg35 从高到低（并列：单项的次序；组合排在同值单项之前）。"""
    items = ranked(results)
    if combination_adopted(combo, combo_result, results):
        items = items+[combo]
        means = dict({a: results[a]['delta_seg35']['mean'] for a in items if a != combo}, **{combo: combo_result['delta_seg35']['mean']})
        items.sort(key=lambda a: (-means[a], -1 if a == combo else rc.CANDIDATES.index(a)))
    return items


def bench_decision(first, second=None, *, pairs, pair_subjects, subjects, excluded=None, undecidable_before_truth=None, whole_undecidable=None):
    """台的判定（bench_decision.json 的主体）。
    first: 第一读的 {arm: {pair: row}}（含 CTL）；second: 第二读的 {combo: {pair: row}} 或 None；excluded: {候选: 原因}（建锁之前的数组核对不符，C25.7-3 特例）；
    undecidable_before_truth: {候选: 原因}（读真值之前的失败按 (甲) 判不可判，BZ 已看到）；whole_undecidable: 整台不可判的原因（读真值之后：不是 16 对、CTL 指标无定义、指标比对不符、异常退出、缺真值）。"""
    excluded, before = dict(excluded or {}), dict(undecidable_before_truth or {})
    out = dict(schema=rc.SCHEMA.format('bench_decision'), pairs=list(pairs), thresholds=dict(mean_seg35=rc.SCREEN_MEAN35, count_seg35=rc.SCREEN_COUNT, mean_seg4=rc.SCREEN_MEAN4, sdlogj=rc.SDLOGJ_MAX, combo_margin=rc.COMBO_MARGIN),
               limits=BENCH_LIMITS)
    if whole_undecidable is None:
        if first is None or rc.CONTROL not in first:
            whole_undecidable = '第一读没有 CTL 的行（评测单元异常退出或没写出）'
        else:
            ctl = first[rc.CONTROL]
            if sorted(ctl) != sorted(pairs) or len(pairs) != rc.N_BENCH_PAIRS:
                whole_undecidable = f'评测不是 {rc.N_BENCH_PAIRS} 对'
            elif any(has_output(ctl[p]) and not defined(ctl[p]) for p in pairs):
                whole_undecidable = 'CTL 有输出、但某对的指标没有定义'
            elif not all(has_output(ctl[p]) for p in pairs):
                whole_undecidable = 'CTL 有对没有输出（应在锁前核掉）'
    if whole_undecidable:
        return dict(out, reading=UNDECIDABLE, reason=whole_undecidable, results={}, order=[], combo=None, stop=True)
    ctl = first[rc.CONTROL]
    results = {}
    for arm in rc.CANDIDATES:
        if arm in excluded:
            results[arm] = dict(verdict=EXCLUDED, reason=excluded[arm])
        elif arm in before:
            results[arm] = dict(verdict=UNDECIDABLE, reason=before[arm], layer='before_truth')
        elif arm not in first:
            results[arm] = dict(verdict=UNDECIDABLE, reason='第一读没有这一臂', layer='after_truth')
        else:
            results[arm] = dict(screen(first[arm], ctl, pairs, pair_subjects, subjects), layer='after_truth')
    order = ranked(results)
    combo = planned_combination(order)
    combo_result = None
    if combo is not None and second is not None:
        combo_result = screen(second[combo], ctl, pairs, pair_subjects, subjects) if combo in second else dict(verdict=UNDECIDABLE, reason='第二读没有组合臂')
    elif combo is not None and second is None:
        combo_result = None
    final_o = order_o(results, combo, combo_result)
    reading = STOP_NO_CANDIDATE if not order else PASS
    return dict(out, reading=reading, results=results, ranked=order, combo=combo, combo_needs_second_read=combo is not None and second is None, combo_result=combo_result,
                combo_adopted=combination_adopted(combo, combo_result, results), order=final_o, stop=not order,
                note='没有候选过筛门：停，不花开发集读数，方法不变（C25.7-5）' if not order else None)


BENCH_LIMITS = ('台上用 UF2，开发集用 UF4；受试者也不同。台上的增益不一定传到开发集。',
                '温度在这 16 人上标定：B 臂直接用温度缩放的概率，是样本内；两个成员平均时温度决定相对权重，各臂的 argmax 也带一点样本内的成分。',
                '这 16 人已被读过多次：v1、v2、U1 的 calib；U1 按他们的标定 Dice 选过步；UF 的 40000 步参考过 U1 的标定数；v1 的交叉拟合标定、R3b 的标定诊断、R3 定组成都读过他们。',
                '16 对、每人出现两次；块 bootstrap 按受试者重抽。',
                '台上的选择有赢家偏差：〔推断〕排第 1 的候选的台上 Δ 偏乐观。')


# ---- 开发集：第 1 选、次选、顺延（C25.7-6、C25.8-3、-5） -------------------------------------------------------------------------------------------
FEASIBLE, NOT_FEASIBLE, FEAS_STOP = 'feasible', 'not_feasible', 'stop_before_truth'


def feasibility_verdict(rows):
    """可行性的判定：任何一对 hx_implementation_error 或 hx_provenance_rejected → 读真值之前的失败，停（不自动记"可行性不过"，C25.8-3）；
    否则 19/19 有输出且方法级官方 SDlogJ 均值 ≤ .10 → 可以读；其余 → 可行性不过（不花读数，顺延）。rows: {pair: dict(line_status, sdlogj_official)}。"""
    bad = {p: r.get('line_status') for p, r in rows.items() if r.get('line_status') in (rc.IMPLEMENTATION, rc.REJECTED) or r.get('line_status') not in rc.LINE_STATUSES}
    if bad:
        return dict(verdict=FEAS_STOP, reason=f'读真值之前的失败：{bad}；停下整个可行性队列，交总负责人与 BZ（C25.8-3、C25.7-3）')
    outputs = sum(r.get('line_status') in rc.OUTPUT_STATUSES for r in rows.values())
    values = [r.get('sdlogj_official') for r in rows.values() if r.get('line_status') in rc.OUTPUT_STATUSES]
    undefined = [p for p, r in rows.items() if r.get('line_status') in rc.OUTPUT_STATUSES and (not isinstance(r.get('sdlogj_official'), (int, float)) or not math.isfinite(r.get('sdlogj_official')))]
    if undefined:                                                                           # 复核次要 4：有输出、SDlogJ 算不出来：读真值之前的失败
        return dict(verdict=FEAS_STOP, reason=f'有输出但官方 SDlogJ 算不出来：{undefined}；停下，交总负责人与 BZ（C25.8-3、C25.7-3）')
    if len(rows) != rc.N_DEV_PAIRS or outputs != rc.N_DEV_PAIRS:
        return dict(verdict=NOT_FEASIBLE, reason=f'有输出 {outputs}/{rc.N_DEV_PAIRS}（共 {len(rows)} 对）', outputs=outputs)
    mean = float(np.mean(values))
    return dict(verdict=FEASIBLE if mean <= rc.SDLOGJ_MAX else NOT_FEASIBLE, outputs=outputs, sdlogj_official_mean=mean)


def next_dev_step(order, feasibility, readings):
    """开发集下一步（纯函数）。feasibility: {项: FEASIBLE | NOT_FEASIBLE | FEAS_STOP}（FEAS_STOP 经 BZ 定 (甲) 之后由总负责人记成 NOT_FEASIBLE 才顺延）；
    readings: [(项, 'pass' | 'not_pass' | 'undecidable')]，至多 2 次。返回 dict(action=feasibility | read | stop, item, read_number, reason)。"""
    if not order:
        return dict(action='stop', reason='O 为空：台上没有候选过筛门')
    if len(readings) > 2:
        raise ValueError('读数至多 2 次')
    for item, state in feasibility.items():
        if state == FEAS_STOP:
            return dict(action='stop', item=item, reason=f'{item} 的可行性一步有读真值之前的失败：交总负责人与 BZ；BZ 定 (甲) 之后记成可行性不过再顺延')
    if readings:
        first_item, first_result = readings[0]
        if first_result == PASS:
            return dict(action='stop', reason='第 1 次过门：不用第 2 次；交 BZ，另起 C22 的修订与预登记修订 3')
        if first_result == UNDECIDABLE:
            return dict(action='stop', reason='第 1 次不可判：停，交总负责人与 BZ（算用掉一次读数）')
        if len(readings) == 2:
            second_item, second_result = readings[1]
            return dict(action='stop', reason={PASS: '第 2 次过门：同第 1 次过门的处理，交 BZ', FAIL: '第 2 次不过门：停，重新冻结，方法不变',
                                               UNDECIDABLE: '第 2 次不可判：停，交总负责人与 BZ'}[second_result])
        return _walk(order, feasibility, start=order.index(first_item)+1, skip=rc.families(first_item), read_number=2)
    return _walk(order, feasibility, start=0, skip=frozenset(), read_number=1)


def _walk(order, feasibility, *, start, skip, read_number):
    for item in order[start:]:
        if rc.families(item) & skip:
            continue                                            # 同族跳过（D132）
        state = feasibility.get(item)
        if state is None:
            return dict(action='feasibility', item=item, read_number=read_number)
        if state == FEASIBLE:
            return dict(action='read', item=item, read_number=read_number)
    if read_number == 1:
        return dict(action='stop', reason='O 用完还没有可读的项：停，交 BZ（C25.8-3）')
    return dict(action='stop', reason='找不到次选：不用第 2 次（D132、D133）')


def runner_up(order, first_choice, feasibility=None):
    """从第 1 选之后往下找，第一个与第 1 选不同族（族集合不相交）的项；feasibility 给出时还要过可行性一步（未知的项先返回它，交可行性一步）。"""
    feasibility = feasibility or {}
    skip = rc.families(first_choice)
    for item in order[order.index(first_choice)+1:]:
        if rc.families(item) & skip:
            continue
        if feasibility.get(item, FEASIBLE) == FEASIBLE:
            return item
        if feasibility.get(item) is None:
            return item
    return None


# ---- 开发集读数的门（C25.8-4） -------------------------------------------------------------------------------------------------------------------
def dev_reading(arm_rows, hs_rows, pairs, pair_index):
    """Δ = 臂 − HS，逐对，方法级。过门：Δseg35 均值 ≥ +.003 且受试者块 95% 下界 > −.003；Δseg4 均值 ≥ −.001；19/19 有输出；方法级官方 SDlogJ 均值 ≤ .10。
    "更好"：过门且块下界 > 0。建锁之后：缺真值、Δ 或 SDlogJ 没有定义 → 不可判（算用掉一次读数）。"""
    if len(pairs) != rc.N_DEV_PAIRS:
        return dict(reading=UNDECIDABLE, reason=f'评测不是 {rc.N_DEV_PAIRS} 对')
    missing = [p for p in pairs if p not in arm_rows or p not in hs_rows]
    if missing:
        return dict(reading=UNDECIDABLE, reason=f'有对没读到：{missing}')
    outputs = sum(has_output(arm_rows[p]) for p in pairs)
    undefined = [p for p in pairs if (has_output(arm_rows[p]) and not defined(arm_rows[p])) or not defined(hs_rows[p])]
    if undefined:
        return dict(reading=UNDECIDABLE, reason=f'Δ 或 SDlogJ 没有定义：{undefined}')
    if outputs != len(pairs):
        return dict(reading=FAIL, reason=f'有输出 {outputs}/{len(pairs)}', outputs=outputs)
    d35, d4 = deltas(arm_rows, hs_rows, pairs, 'seg35'), deltas(arm_rows, hs_rows, pairs, 'seg4')
    index = [pair_index[p] for p in pairs]
    lo, hi = block_interval(d35, index)
    s35, s4 = paired_summary(d35), paired_summary(d4)
    sdlogj = float(np.mean([arm_rows[p]['metrics']['sdlogj_official'] for p in pairs]))
    gate = dict(mean_seg35=s35['mean'] >= rc.DEV_MEAN35, block_lower=lo > rc.DEV_LOWER, mean_seg4=s4['mean'] >= rc.DEV_MEAN4, outputs=outputs == len(pairs), sdlogj=sdlogj <= rc.SDLOGJ_MAX)
    passed = all(gate.values())
    return dict(reading=PASS if passed else FAIL, better=bool(passed and lo > rc.BETTER_LOWER), gate=gate, delta_seg35=s35, delta_seg4=s4, block_seg35=[lo, hi], sdlogj_official_mean=sdlogj,
                per_pair={p: dict(delta_seg35=a, delta_seg4=b) for p, a, b in zip(pairs, d35, d4)})


def control_mismatches(mine, reference, measures=('seg35', 'seg4', 'sdlogj_official'), tol=rc.CONTROL_TOL):
    """评测侧的控制（C25.6-11、C25.8-4）：逐对逐项 |Δ| ≤ 2e-4；返回不符的 [(pair, measure, mine, reference)]（无定义也算不符）。"""
    bad = []
    for pair in sorted(set(mine) | set(reference)):
        a, b = (mine.get(pair) or {}), (reference.get(pair) or {})
        for m in measures:
            x, y = a.get(m), b.get(m)
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)) or not math.isfinite(x) or not math.isfinite(y) or abs(x-y) > tol:
                bad.append((pair, m, x, y))
    return bad
