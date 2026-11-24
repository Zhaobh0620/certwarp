"""TC's reading (runs/20261104-tc/plan.md, by code; a label for the lead and BZ, not a conclusion): analysis/decision_tc.json, summary_tc.md, long_curve.csv.

Only S_P decides.  Delta = the final map - the commit at t = 90 s, mean over the first 5 development pairs, seg35, unrounded; t is the native clock (plan revision 1: not the total time):
  premises not met       not decidable, the reasons written
  Delta <= +.002         90 s 之后基本是平台
  Delta >= +.005         90 s 之后还在涨
  otherwise              中间带
Premises: all 5 pairs of S_P have class ok, a complete curve (its self-check against metrics.json passed), a certified final map that is the last commit, a certified commit at t = 90 s older than the final map, the entry injected as recorded.
t = 30, 60 of S_P get the same words with their own t, as a description.  S_K1 and B1TM are only described.  The comparison with the original runs does not enter the reading: it says whether the curve also holds for them.
"""
import csv
from pathlib import Path
import statistics

import numpy as np

import tc_curve as curve

SCHEMA = 'bayesreg.tc.decision.v1'
N = 5
PLATEAU_BAR, RISING_BAR = .002, .005
READING_TIME = 90
DESCRIBED_TIMES = (30, 60)
READING_ARM = 'S_P'
PLATEAU, RISING, MIDDLE, NOT_DECIDABLE = 'plateau', 'still_rising', 'middle', 'not_decidable'
READINGS = {PLATEAU: '90 s 之后基本是平台', RISING: '90 s 之后还在涨', MIDDLE: '中间带', NOT_DECIDABLE: '不可判'}
DIFFERENCES = (('S_P', 'S_K1'), ('S_P', 'B1TM'))
DIFFERENCE_KEYS = ('seg35', 'seg4', 'cortex', 'small_six')
MEAN_KEYS = ('seg35', 'seg4', 'sdlogj_official', 'sdlogj_brain', 'small_six', 'large_six', 'other_23', 'cortex', 'white_matter')
KEY_NAMES = dict(seg35='seg35', seg4='seg4', sdlogj_official='官方 SDlogJ', sdlogj_brain='脑内 SDlogJ', small_six='小六个', large_six='大六个', other_23='其余 23 个', cortex='皮层', white_matter='白质')
# The real total time, an inference (plan revision 1): the task clock with the hold taken out, the real entry and the U-Net put in.  entry: S_P about 60 s (mc0p_v1c, the brief §2); S_K1 about 38 s (review of SEED19, item 7, itself an inference); B1TM's T88 is already inside the task clock.
# unet: 86 s for the two subjects of a pair (the brief §2, the v2 chain, writing the files included); the provenance of unet35_v1, which these arms use, says a median of 57 s a subject, about 114 s a pair (plan revision 2).  All three arms carry the same number: no difference between arms changes.  TC's measurement will replace it.
REAL_SECONDS = dict(S_P=dict(charge=47., entry=60., unet=86.), S_K1=dict(charge=47., entry=38., unet=86.), B1TM=dict(charge=0., entry=0., unet=86.))
CLOCKS = dict(native='原生层时钟 t：history.json 的 elapsed_seconds，从原生层 solve_sequential 开始算，不含入口、持时、转移证书。判定和差表的 t 都是它，t = 90 不是总共 90 s。',
              task='任务时钟：commits.json 的 visible_seconds，预算 330 s 看的就是它。S_P、S_K1 约等于 t + 50 s（47 s 持时加约 3 s），B1TM 约等于 t + 4.6 s（T88 约 2 s 加约 2.5 s）。',
              real='真实总时间〔推断〕：任务时钟 − 持时 + 真实入口 + U-Net 推断（S_P 入口约 60 s，S_K1 约 38 s，B1TM 的 T88 已在任务时钟里；U-Net 一对约 86–114 s，已含写文件，这里按 86 s 算，三臂同加，差不变）。折算，不是测的；上色不算。')
NOTE = ('开发集前 5 对，每臂每对单次观测（n = 5），读数不是结论，下一步交 BZ。取时刻不晚于 t 的最后一次正式提交（t = 0 是第一次提交）。曲线都是预算收尾的运行上的状态，不是收敛。'
        '它只说明这条路径上时间的价值，不说明更快的路径到不了；不等于用更短的预算重跑（没有核实路径是否随预算变）。差表的 t 行对齐的是原生层时钟：同一个 t，S_P 比 B1TM 多用约 45 s 任务时钟，所以 t 行为正不能读成"花同样的时间 S_P 更好"；任务时钟相同的是 final 行（都用满 330 s，B1TM 的原生层因此约长 45 s）。S_P 与 S_K1 持时相同，同一个 t 的任务时钟相近。'
        '三臂的图像是同一份 U-Net 预测上色（unet35_v1_ens，A5P）；入口不同：S_P 是 K = 35、600 步，S_K1 是 K = 1、600 步，B1TM 是 T88 60 步。47 s 持时不是入口的真实成本。')


def load_runs(root, plan):
    """{(arm, pair): {curve, metrics, reproduction, cleanup}} of what exists."""
    out = {}
    for arm, pair in plan:
        run = Path(root)/'runs'/arm/pair
        rec = {name: curve.read_json(run/f'{name}.json') for name in ('curve', 'metrics', 'reproduction', 'cleanup') if (run/f'{name}.json').is_file()}
        if rec:
            out[(arm, pair)] = rec
    return out


def point_at(rec, t):
    """The point of a run's curve at t (curve.FINAL: the final map), or None."""
    c = (rec or {}).get('curve') or {}
    if t == curve.FINAL:
        return c.get('final')
    return next((p for p in c.get('points') or [] if p.get('t') == t), None)


def value_at(rec, t, key):
    point = point_at(rec, t) or {}
    return (point.get('metrics') or {}).get(key)


def run_premises(arm, pair, rec, short):
    """Why this run cannot enter the reading (empty: it can)."""
    name = f'{arm} {short}'
    if rec is None:
        return [f'{name}: 没有记录']
    metrics = rec.get('metrics')
    if metrics is None:
        return [f'{name}: 没有 metrics.json']
    bad = []
    cls = (metrics.get('classification') or {}).get('class')
    if cls != 'ok':
        bad.append(f'{name}: 分类 {cls}')
    certificate = metrics.get('certificate') or {}
    if not (certificate.get('passed') and certificate.get('state_hash_matches_u') is True and certificate.get('final_npz_sha256_matches') is True):
        bad.append(f'{name}: 终态证书未通过，或没有与导出场绑定')
    if (metrics.get('seed') or {}).get('passed') is not True:
        bad.append(f"{name}: 入口核对未通过 {(metrics.get('seed') or {}).get('problems')}")
    c = rec.get('curve')
    if c is None or c.get('complete') is not True or (c.get('self_check') or {}).get('passed') is not True:
        bad.append(f'{name}: 没有完整的 curve.json')
        return bad
    point = point_at(rec, READING_TIME)
    if point is None or point.get('commit') is None:
        bad.append(f'{name}: t = {READING_TIME} s 没有提交')
    elif point.get('certificate_passed') is not True:
        bad.append(f'{name}: t = {READING_TIME} s 的提交证书未通过')
    elif c['final'].get('elapsed_seconds') is None or c['final']['elapsed_seconds'] <= point['elapsed_seconds']:
        bad.append(f'{name}: 终态不晚于 t = {READING_TIME} s 的提交')
    if c['final'].get('certificate_passed') is not True:
        bad.append(f'{name}: curve.json 的终态证书未通过')
    if c['final'].get('is_last_commit') is not True:     # review m2: the final map's seconds are those of the last commit
        bad.append(f'{name}: 终态不是最后一次提交，它的秒数不能用')
    return bad


def delta_from(runs, arm, pairs, short, t, key='seg35'):
    """Per pair final - t, the mean over all pairs, or None if a value is missing."""
    values = {}
    for pair in pairs:
        a, b = value_at(runs.get((arm, pair)), curve.FINAL, key), value_at(runs.get((arm, pair)), t, key)
        if a is not None and b is not None:
            values[pair] = a-b
    complete = len(values) == len(pairs) == N
    return dict(per_pair={short(p): v for p, v in values.items()}, mean=float(np.mean(list(values.values()))) if complete else None, median=float(statistics.median(values.values())) if complete else None, positive=sum(v > 0 for v in values.values()), n=len(values))


def label_of(delta):
    """The plan's words for a mean difference; the bars included, the mean unrounded."""
    return PLATEAU if delta <= PLATEAU_BAR else RISING if delta >= RISING_BAR else MIDDLE


def words_at(code, t):
    """The plan's words with this t: READINGS speak of t = 90 s, the reading's time."""
    return READINGS[code].replace(f'{READING_TIME} s', f'{t} s')


def signed_mean(delta):
    return '不可算' if delta['mean'] is None else f"{delta['mean']:+.4f}（{delta['positive']}/{delta['n']} 为正）"


def decide(runs, pairs, short):
    reasons = [r for pair in pairs for r in run_premises(READING_ARM, pair, runs.get((READING_ARM, pair)), short(pair))]
    delta = delta_from(runs, READING_ARM, pairs, short, READING_TIME)
    delta4 = delta_from(runs, READING_ARM, pairs, short, READING_TIME, 'seg4')
    if delta['mean'] is None and not reasons:            # review m1: the premises hold and a value is still missing
        reasons.append(f"S_P: 终态或 t = {READING_TIME} s 缺 seg35，只有 {delta['n']} 对，Δ 不可算")
    code = NOT_DECIDABLE if reasons else label_of(delta['mean'])
    if reasons:
        text = READINGS[code]+'：'+'；'.join(reasons)
    else:
        text = f"{READINGS[code]}：Δ(seg35) = 终态 − t = {READING_TIME} s = {delta['mean']:+.4f}，{delta['positive']}/{N} 为正；Δ(seg4) {signed_mean(delta4)}，只描述，不进判定；{READING_TIME} s 是原生层时刻 t，不是总时间"
    described = {str(t): dict(delta=delta_from(runs, READING_ARM, pairs, short, t), delta_seg4=delta_from(runs, READING_ARM, pairs, short, t, 'seg4')) for t in DESCRIBED_TIMES}
    for t, row in zip(DESCRIBED_TIMES, described.values()):
        row['words'] = None if row['delta']['mean'] is None else words_at(label_of(row['delta']['mean']), t)
    return dict(reading_code=code, reading=text, reasons=reasons, delta_seg35=delta, delta_seg4=delta4, described_times=described)


def runs_at(runs, t):
    """{(arm, pair): {metrics}} at time t: the form evaluate.py's group functions read."""
    return {key: dict(metrics=(point_at(rec, t) or {}).get('metrics')) for key, rec in runs.items() if (point_at(rec, t) or {}).get('metrics')}


def real_total_seconds(arm, task_clock):
    """〔推断〕The real total time at this task clock: see REAL_SECONDS.  None without a task clock."""
    if task_clock is None:
        return None
    spec = REAL_SECONDS[arm]
    return task_clock-spec['charge']+spec['entry']+spec['unet']


def mean_task_clock(runs, arm, t, pairs):
    values = [(point_at(runs.get((arm, pair)), t) or {}).get('visible_seconds') for pair in pairs]
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def arm_table(runs, arm, pairs):
    """One row per time: the means of the arm over the pairs it has, and how many; its task clock and, an inference, its real total time."""
    rows = []
    for t in curve.TIMES+(curve.FINAL,):
        points = [point_at(runs.get((arm, pair)), t) for pair in pairs]
        points = [p for p in points if p and p.get('metrics')]
        row = dict(t=t, n=len(points))
        for key in MEAN_KEYS:
            values = [p['metrics'].get(key) for p in points if p['metrics'].get(key) is not None]
            row[key] = float(np.mean(values)) if values else None
        for key in ('elapsed_seconds', 'visible_seconds', 'D', 'GF', 'min_jd_lower_bound'):
            values = [p.get(key) for p in points if p.get(key) is not None]
            row[key] = float(np.mean(values)) if values else None
        row['commit'] = float(np.mean([p['commit'] for p in points if p.get('commit') is not None])) if any(p.get('commit') is not None for p in points) else None
        row['real_total_seconds'] = real_total_seconds(arm, row['visible_seconds'])
        rows.append(row)
    return rows


def paired_difference(runs, a, b, t, key, pairs):
    values = []
    for pair in pairs:
        x, y = value_at(runs.get((a, pair)), t, key), value_at(runs.get((b, pair)), t, key)
        if x is not None and y is not None:
            values.append(x-y)
    return dict(mean=float(np.mean(values)) if values else None, positive=sum(v > 0 for v in values), n=len(values))


def difference_table(runs, a, b, pairs, ev):
    """S_P(t) - S_K1(t) and S_P(t) - B1TM(t), pair by pair: seg35, seg4 as in the run; cortex is seg4 label 1 (evaluate.py label_rows); the small six are the paired difference of the group's mean (evaluate.py a5_group_row).  Each row also gives both arms' task clock and, an inference, real total time: a row is the same native time, not the same total time."""
    rows = []
    for t in curve.TIMES+(curve.FINAL,):
        at = runs_at(runs, t)
        clock_a, clock_b = mean_task_clock(runs, a, t, pairs), mean_task_clock(runs, b, t, pairs)
        row = dict(t=t, task_clock_a=clock_a, task_clock_b=clock_b, real_total_a=real_total_seconds(a, clock_a), real_total_b=real_total_seconds(b, clock_b), seg35=paired_difference(runs, a, b, t, 'seg35', pairs), seg4=paired_difference(runs, a, b, t, 'seg4', pairs))
        cortex = ev.label_rows(at, a, b, 'seg4', (1,), pairs)[1]
        row['cortex'] = dict(mean=cortex['mean_diff'], positive=cortex['positive'], n=cortex['n'])
        group = ev.a5_group_row(at, a, b, ev.A5_GROUPS['small_six'], pairs)
        row['small_six'] = dict(mean=group['mean_diff'], positive=group['positive'], n=group['n'])
        rows.append(row)
    return rows


def anomalies(runs, plan, short):
    out = []
    for arm, pair in plan:
        rec = runs.get((arm, pair))
        c = (rec or {}).get('curve')
        if rec is None:
            continue
        if c is None:
            out.append(dict(arm=arm, pair=short(pair), what='没有 curve.json'))
            continue
        for p in c.get('points') or []:
            if p.get('commit') is None:
                out.append(dict(arm=arm, pair=short(pair), what=p.get('reason')))
            elif (p.get('metrics') or {}).get('sdlogj_official', 0) > .10:
                out.append(dict(arm=arm, pair=short(pair), what=f"t = {p['t']} s 官方 SDlogJ {p['metrics']['sdlogj_official']:.4f} > .10"))
        final = c.get('final') or {}
        if (final.get('metrics') or {}).get('sdlogj_official', 0) > .10:
            out.append(dict(arm=arm, pair=short(pair), what=f"终态官方 SDlogJ {final['metrics']['sdlogj_official']:.4f} > .10"))
        if final.get('is_last_commit') is False:
            out.append(dict(arm=arm, pair=short(pair), what='终态不是最后一次提交'))
        if c.get('n_native_commits', 0) and (final.get('elapsed_seconds') or 0) < curve.TIMES[-1]:
            out.append(dict(arm=arm, pair=short(pair), what=f"原生层在 {final.get('elapsed_seconds'):.1f} s 结束，早于 {curve.TIMES[-1]} s"))
        unrecorded = (rec.get('cleanup') or {}).get('not_deleted_unrecorded') or []
        if unrecorded:
            out.append(dict(arm=arm, pair=short(pair), what=f"{len(unrecorded)} 个中间文件没有 sha256 记录，没有删：{', '.join(row['name'] for row in unrecorded)}"))
    return out


def reproduction_rows(runs, plan, short):
    rows = []
    for arm, pair in plan:
        r = (runs.get((arm, pair)) or {}).get('reproduction')
        if r is None:
            continue
        s, f, at = r['sequence'], r['final'], (r.get('commit_at_time') or {}).get(str(READING_TIME)) or {}
        rows.append(dict(arm=arm, pair=short(pair), n_new=s['n_new'], n_old=s['n_old'], common_prefix=s['common_prefix'], state_common_prefix=s['state_common_prefix'], identical=s['identical'], first_difference=s['first_difference'], tail=s['tail'], entry_same=r['start']['entry_same'],
                         initializer_same=r['start']['initializer_same'], same_final=f['same_file'], field_difference=f['field_difference'], seg35_new_minus_old=f['seg35_new_minus_old'], seg4_new_minus_old=f['seg4_new_minus_old'], commit_at_90_new=at.get('commit_new'), commit_at_90_old=at.get('commit_old'), same_field_at_90=at.get('same_field')))
    return rows


def f(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def signed(row, digits=4):
    if row.get('mean') is None:
        return '—'
    return f"{row['mean']:+.{digits}f}（{row['positive']}/{row['n']}）"


def time_at_90(result, arm):
    """A sentence: what t = 90 s of this arm is on the other two clocks."""
    row = next((r for r in result['tables'].get(arm, ()) if r['t'] == READING_TIME), None)
    if row is None or row['visible_seconds'] is None:
        return f'{arm} 的 t = {READING_TIME} s：没有数'
    return f"{arm} 的 t = {READING_TIME} s：任务时钟约 {row['visible_seconds']:.0f} s，〔推断〕真实总时间约 {row['real_total_seconds']:.0f} s"


def summary_text(result):
    lines = ['# TC 汇总（代码写出，读数不是结论）', '', result['note'], '', f"**读数（只看 S_P）**：{result['reading']}", '', '**三种时钟，不要读混**：', '', f"- {result['clocks']['native']}", f"- {result['clocks']['task']}", f"- {result['clocks']['real']}", '',
             '；'.join(time_at_90(result, arm) for arm in result['tables'])+f'。BZ 说的"只要 90 s"是任务时钟的预算（--seconds 90），不是这里的 t = {READING_TIME}。', '', f"结果根目录 {result['root']}；写出 {result['written_utc']}。", '', '## 判定：Δ = 终态 − t = 90 s（S_P，t 是原生层时刻）', '', '| 对 | Δ seg35 | Δ seg4 |', '|---|---|---|']
    d35, d4 = result['delta_seg35'], result['delta_seg4']
    for pair, value in d35['per_pair'].items():
        lines.append(f"| {pair} | {value:+.4f} | {d4['per_pair'].get(pair, float('nan')):+.4f} |")
    lines += ['', f"均值 Δ seg35 {f(d35['mean'])}（{d35['positive']}/{d35['n']} 为正），Δ seg4 {f(d4['mean'])}（{d4['positive']}/{d4['n']} 为正）。线：≤ +.002 基本是平台，≥ +.005 还在涨。", '', '## 只描述：S_P 的 t = 30、60', '']
    for t, row in result['described_times'].items():
        lines.append(f"- t = {t} s：Δ seg35 {f(row['delta']['mean'])}（{row['delta']['positive']}/{row['delta']['n']}），Δ seg4 {f(row['delta_seg4']['mean'])}（{row['delta_seg4']['positive']}/{row['delta_seg4']['n']}）；同样的线下是"+(row['words'] or '不可算'))
    for arm, table in result['tables'].items():
        lines += ['', f"## {'判定臂' if arm == READING_ARM else '只描述'}：{arm} 各时刻的均值", '', '| t（原生层 s） | 任务时钟（s） | 〔推断〕真实总时间（s） | n | 平均提交序号 | 平均 elapsed | '+' | '.join(KEY_NAMES[k] for k in MEAN_KEYS)+' | D | GF |', '|---|---|---|---|---|---|'+'---|'*(len(MEAN_KEYS)+2)]
        for row in table:
            lines.append(f"| {row['t']} | {f(row['visible_seconds'], 1)} | {f(row['real_total_seconds'], 0)} | {row['n']} | {f(row['commit'], 1)} | {f(row['elapsed_seconds'], 1)} | "+' | '.join(f(row[k]) for k in MEAN_KEYS)+f" | {f(row['D'], 1)} | {f(row['GF'], 3)} |")
    for name, table in result['differences'].items():
        lines += ['', f'## 只描述：{name}（均值，括号里是为正的对数/对数）', '', '每行是同一个原生层时刻，不是同一个总时间；两列时钟写的是"前者 / 后者"。', '', '| t（原生层 s） | 任务时钟（s） | 〔推断〕真实总时间（s） | '+' | '.join(KEY_NAMES[k] for k in DIFFERENCE_KEYS)+' |', '|---|---|---|'+'---|'*len(DIFFERENCE_KEYS)]
        for row in table:
            lines.append(f"| {row['t']} | {f(row['task_clock_a'], 0)} / {f(row['task_clock_b'], 0)} | {f(row['real_total_a'], 0)} / {f(row['real_total_b'], 0)} | "+' | '.join(signed(row[k]) for k in DIFFERENCE_KEYS)+' |')
    lines += ['', '## commits 的 sha256 序列对原运行', '', '原运行：S_P 在 seed_v1，S_K1 在 seed_v1b，B1TM 在 b1_v1（都只读）。逐项相同才说曲线也适用于原运行；不同时写第一个分歧点和差多少。序列相同时，同一个 t 也可能取到不同的提交（两次运行的快慢不同），所以另列 t = 90 s 在两次运行上各取第几次提交。', '',
              '| 臂 | 对 | 提交数（新/原） | 相同前缀 | 逐项相同 | t = 90 s 取第几次（新/原） | 入口 / 初值相同 | 第一个分歧 | 终态 seg35 差 | 终态 seg4 差 | 终态场差（RMS / 最大，mm） |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in result['reproduction']:
        first = r['first_difference']
        where = '—' if first is None else f"第 {first['commit']} 次提交（{f(first['elapsed_new'], 1)} / {f(first['elapsed_old'], 1)} s），D 差 {f(first['D_new_minus_old'], 6)}"
        if r['tail'] is not None:
            where = f"{'新运行' if r['tail']['longer'] == 'new' else '原运行'}多 {r['tail']['n_extra']} 次提交（前缀内逐项相同）"
        field = '同一文件' if r['same_final'] else ('—' if r['field_difference'] is None else f"{r['field_difference']['rms_mm']:.2e} / {r['field_difference']['max_abs_mm']:.2e}")
        lines.append(f"| {r['arm']} | {r['pair']} | {r['n_new']}/{r['n_old']} | {r['common_prefix']} | {r['identical']} | {r['commit_at_90_new']}/{r['commit_at_90_old']} | {r['entry_same']} / {r['initializer_same']} | {where} | {f(r['seg35_new_minus_old'], 5)} | {f(r['seg4_new_minus_old'], 5)} | {field} |")
    lines += ['', '## 异常', '']
    lines += [f"- {a['arm']} {a['pair']}: {a['what']}" for a in result['anomalies']] or ['没有。']
    return '\n'.join(lines)+'\n'


def long_rows(runs, plan, short):
    """One row per arm, pair, time and measure; t final is the final map."""
    for arm, pair in plan:
        rec = runs.get((arm, pair)) or {}
        c = rec.get('curve')
        if c is None:
            continue
        for t in curve.TIMES+(curve.FINAL,):
            point = point_at(rec, t)
            if not point or not point.get('metrics'):
                continue
            base = dict(arm=arm, pair=pair, short=short(pair), t=t, commit=point.get('commit'), elapsed_seconds=point.get('elapsed_seconds'), visible_seconds=point.get('visible_seconds'), run_id=c.get('run_id'))
            for key in MEAN_KEYS:
                if point['metrics'].get(key) is not None:
                    yield dict(base, measure=key, value=point['metrics'][key])
            for key in ('D', 'F', 'GF', 'min_jd_lower_bound'):
                if point.get(key) is not None:
                    yield dict(base, measure=key, value=point[key])


def write_analysis(root, plan, pairs, short, *, ev, utc):
    """Reads the runs of this batch, writes the three files under root/analysis; an earlier summary goes to analysis/previous_<UTC>/.  Returns the decision."""
    root = Path(root)
    runs = load_runs(root, plan)
    arms = list(dict.fromkeys(arm for arm, _ in plan))
    result = dict(schema=SCHEMA, root=str(root), written_utc=utc(), note=NOTE, clocks=CLOCKS, real_seconds=REAL_SECONDS, rule='runs/20261104-tc/plan.md', reading_arm=READING_ARM, **decide(runs, pairs, short))
    result['tables'] = {arm: arm_table(runs, arm, pairs) for arm in arms}
    result['differences'] = {f'{a}(t) − {b}(t)': difference_table(runs, a, b, pairs, ev) for a, b in DIFFERENCES}
    result['reproduction'] = reproduction_rows(runs, plan, short)
    result['anomalies'] = anomalies(runs, plan, short)
    result['sources'] = dict(evaluator_sha256=curve.sha_file(ev.__file__), code_sha256=curve.sha_file(__file__), curve_code_sha256=curve.sha_file(curve.__file__))
    out = root/'analysis'
    out.mkdir(parents=True, exist_ok=True)
    moved = ev._set_aside(out, ('long_curve.csv', 'summary_tc.md', 'decision_tc.json'))
    result['previous_summary_moved_to'] = None if moved is None else str(moved)
    with open(out/'long_curve.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['arm', 'pair', 'short', 't', 'measure', 'value', 'commit', 'elapsed_seconds', 'visible_seconds', 'run_id'])
        writer.writeheader()
        writer.writerows(long_rows(runs, plan, short))
    curve.write_json(out/'decision_tc.json', result)
    (out/'summary_tc.md').write_text(summary_text(result))
    return result
