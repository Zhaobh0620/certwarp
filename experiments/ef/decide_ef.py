"""EF's reading (runs/20261104-ef/plan.md, by code; a label for the lead and BZ, not a conclusion): analysis/decision_ef.json, long.csv, summary_ef.md.

E1, each arm against Q2 (long tier, baseline of this batch), at the accepted alpha:
    Delta seg35 >= -.002 (mean over the 5 pairs)  and  5/5 certified at some alpha of the frozen .9^k rule  and  mean seconds_total <= 60 % of Q2's     "可以更快"
    otherwise                                                                                                                                         "不能更快", with what missed
    the reading: some arm "可以更快" -> 可以更快; every arm read and none -> 不能更快; none and some arm not read -> 没有臂过线，有臂没读全; Q2 not read 5/5 -> 不可判
E3, each fine arm (eps .01, .05; the 5 fine pairs only; the two long pairs are described, never in the reading; the lead, 2026-11-04):
    5/5 certified at alpha = 1 by the evaluation's own certificate  and  mean(seg35 after - before) >= -.002  and  mean seconds of the repair <= 15     "修复可用"
    otherwise                                                                                                                                         "修复不可用", with what missed
    the reading: some arm "修复可用" -> 修复可用; every arm read and none -> 修复不可用; none and some arm not read -> 没有臂过线，有臂没读全
Nothing here reads a run's opinion of itself where the evaluation has one: certificates and Dice come from metrics.json and repair_metrics.json, written by the evaluation unit from the fields.  A cell of E1 is read only with status done (EF_tools_复核 I1, as decide_mc0.valid).
Only described, never in the reading (plan revision 1): the same-path share (seconds of the arm / seconds of F600: both on the fast path); the certified-only means of E3 and the fallback's Dice; the pair-by-pair numbers.
"""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics

import ef_core as ef
import mc0_core as core

SCHEMA = 'bayesreg.ef.decision.v1'
N = 5
SEG35_BAR, SECONDS_SHARE, REPAIR_SECONDS = -.002, .60, 15.
MEASURES = ('seg35', 'seg4', 'sdlogj_official', 'sdlogj_brain', 'small_six', 'large_six', 'other_23', 'cortex', 'white_matter')
READ_FASTER, READ_NOT_FASTER, READ_E1_GAPS, READ_E1_UNDECIDABLE = '可以更快', '不能更快', '没有臂过线，有臂没读全', '不可判：基线 Q2 没有读全 5 对'
READ_REPAIR, READ_NO_REPAIR, READ_E3_GAPS = '修复可用', '修复不可用', '没有臂过线，有臂没读全'
NOTE = ('开发集单次观测，通道来自 unet35_v1_ens 的预测（oracle: false），真值标签只进评测。这是入口探针：没有过原生层，不是方法，不进结果表。入口的 Dice 不等于过原生层之后的 Dice；阳性不说明进方法，进方法先改合同。n = 5。'
        '前 5 对是之前选配用过的。Q2 与其余臂的秒数来自同一批，通道路径不同（Q2 走 mc0p 的旧路径，其余走 E2 的快路径），所以 E1 的比值同时含 E1 与 E2 的省时，分段见表。'
        '带选择效应：E1 从 6 个臂里取"有臂"，E3 从 2 个 ε 里取。')
SAME_PATH_NOTE = ('"对 F600 的同路径比值"只描述，不进判定：该臂秒数 / F600 秒数，两者都走快路径，F600 与 Q2 长档的每步代价相同（mc0p_v1b 原生层 32.1–32.7 s 对 32.1–32.4 s）。秒数比的分母 Q2 走旧路径，所以**过线可能全靠 E2（建通道的实现），不是步数或 K**：'
                  '读数不能写成"步数减半就够"。E2 省下的时间里还有一部分只是去掉了记录项（r_ℓ、第二次上传、与 A5P 文件逐位比较）。标"靠 E2"的臂：过线，但同路径比值 > .60。')
K8_NOTE = ('K8：记录里的"同色 3 体素内"只数这个范围；粗层 σ 最大约 3.97 体素、核半径约 16 体素，同色标签在粗层混得更多。读 K8 的 Dice 时要知道这一点。')
E3_ATTEMPT_NOTE = ('"修复尝试后"含没有过证书的场，这些不是可用的场，不能引用成"修复后 seg35"。可以引用的是"只算过证书的对"那一列；没过证书的对，实际入口是回退规则下的 Dice（见逐对表）。判定的第 2 条按预登记用全部 5 对的差，但第 1 条已经挡住未过证书的情形。')


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def cell_row(root, cell):
    """What the run and the evaluation say of one E1 cell; None if the run never started."""
    d = core.run_dir(root, cell)
    run, metrics, done, ladder = read_json(d/'run.json'), read_json(d/'metrics.json'), read_json(d/'queue_done.json'), read_json(d/'ladder_seconds.json')
    if run is None and metrics is None and done is None:
        return None
    run, done = run or {}, done or {}
    row = dict(run_id=cell.run_id, arm=cell.arm, tier=cell.tier, pair=cell.pair, status=run.get('status', done.get('state', 'failed')), oracle=run.get('oracle'), k=ef.ARMS[cell.arm]['k'], native=ef.ARMS[cell.arm]['native'], sigma_field=ef.ARMS[cell.arm]['sigma_field'],
               seconds_total=run.get('seconds_total'), seconds_labels=run.get('seconds_labels'), seconds_channels=run.get('seconds_channels'), device_peak_reserved_gib=run.get('device_peak_reserved_gib'), seconds_ladder=(ladder or {}).get('seconds'),
               D_multiplier=(run.get('scale') or {}).get('D_multiplier'), metrics=None, accepted_alpha=None, at_accepted=None, n_infeasible_at_1=None, min_lower_bound=None, scale_check=run.get('scale_check'))
    levels = (run.get('optimizer') or {}).get('levels') or []
    if levels:
        row['seconds_levels'] = [level['seconds'] for level in levels]
        if run.get('seconds_channels') is not None and (run.get('optimizer') or {}).get('seconds') is not None:
            row['seconds_pyramid_and_norms'] = run['optimizer']['seconds']-run['seconds_channels']-sum(row['seconds_levels'])
    if metrics is not None:
        certificate = metrics['certificate']
        accepted = next((a for a in certificate['attempts'] if a['alpha'] == certificate['accepted_alpha']), None)
        row.update(metrics={m: metrics['metrics'][m] for m in MEASURES}, accepted_alpha=certificate['accepted_alpha'], at_accepted=metrics.get('at_accepted_alpha'), n_infeasible_at_1=certificate['n_infeasible_at_1'],
                   min_lower_bound=None if accepted is None else accepted['min_jd_lower_bound'], attempts=[dict(alpha=a['alpha'], status=a['status'], n_infeasible=a['n_infeasible'], min_jd_lower_bound=a['min_jd_lower_bound']) for a in certificate['attempts']], tapered_u_sha256=metrics['tapered_u_sha256'])
    return row


def repair_row(root, cell):
    """What the repair run and the evaluation say of one E3 cell."""
    d = core.run_dir(root, cell)
    repair, metrics, done = read_json(d/'repair.json'), read_json(d/'repair_metrics.json'), read_json(d/'queue_done.json')
    if repair is None and metrics is None and done is None:
        return None
    repair, done = repair or {}, done or {}
    row = dict(run_id=cell.run_id, arm=cell.arm, tier=cell.tier, pair=cell.pair, eps=ef.REPAIR_ARMS[cell.arm]['eps'], status=repair.get('status', done.get('state', 'failed')), run_says_passed=repair.get('passed'), reason=repair.get('reason'), seconds_repair=repair.get('seconds_repair'), seconds_warmup=repair.get('seconds_warmup'),
               rounds=len(repair.get('rounds') or []), n_extra_points=max((r.get('n_extra_points', 0) for r in repair.get('rounds') or []), default=0), metrics=metrics is not None, certified_at_1=None)
    if metrics is not None:
        after = metrics['after']
        row.update(certified_at_1=after['certificate']['accepted_alpha'] == 1., n_infeasible_after=after['certificate']['n_infeasible_at_1'], min_lower_bound_after=after['certificate']['attempts'][0]['min_jd_lower_bound'],
                   before=metrics['before'], after=after['metrics'], change=metrics['change'], fallback=metrics['fallback'], delta_seg35=after['metrics']['seg35']-metrics['before']['metrics']['seg35'], delta_seg4=after['metrics']['seg4']-metrics['before']['metrics']['seg4'])
    return row


def collect(root):
    rows, repairs = {}, {}
    for cell in ef.plan_runs():
        if cell.is_repair:
            row = repair_row(root, cell)
            if row is not None:
                repairs[(cell.arm, cell.pair)] = row
        else:
            row = cell_row(root, cell)
            if row is not None:
                rows[(cell.arm, cell.pair)] = row
    return rows, repairs


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def e1_arm(rows, arm, baseline_rows):
    runs = {p: rows.get((arm, p)) for p in core.PAIRS}
    read = {p: r for p, r in runs.items() if r is not None and r['metrics'] is not None and r['status'] == 'done'}
    certified = {p: r for p, r in read.items() if r['accepted_alpha'] is not None}
    both = [p for p in certified if baseline_rows.get(p) is not None and baseline_rows[p].get('at_accepted') is not None]
    d35 = {p: certified[p]['at_accepted']['seg35']-baseline_rows[p]['at_accepted']['seg35'] for p in both}
    d4 = {p: certified[p]['at_accepted']['seg4']-baseline_rows[p]['at_accepted']['seg4'] for p in both}
    seconds = mean(r['seconds_total'] for r in runs.values() if r is not None)
    base_seconds = mean(r['seconds_total'] for r in baseline_rows.values() if r is not None)
    share = seconds/base_seconds if seconds is not None and base_seconds else None
    delta = statistics.fmean(d35.values()) if d35 else None
    complete = len(read) == N
    missed = []
    if len(certified) < N:
        missed.append(f'证书 {len(certified)}/{N}（读到 {len(read)}）')
    if delta is None or delta < SEG35_BAR or len(d35) < N:
        missed.append(f"seg35 差 {'—' if delta is None else f'{delta:+.4f}'}（n = {len(d35)}，门 ≥ {SEG35_BAR}）")
    if share is None or share > SECONDS_SHARE:
        missed.append(f"秒数比 {'—' if share is None else f'{share:.3f}'}（门 ≤ {SECONDS_SHARE}）")
    return dict(arm=arm, n_read=len(read), n_certified=len(certified), complete=complete, delta_seg35_at_accepted=delta, delta_seg4_at_accepted=statistics.fmean(d4.values()) if d4 else None, per_pair_delta_seg35={core.short_name(p): v for p, v in d35.items()},
                positive_seg35=sum(v > 0 for v in d35.values()), mean_seconds=seconds, median_seconds=median([r['seconds_total'] for r in runs.values() if r is not None]), per_pair_seconds=[None if r is None else r['seconds_total'] for r in runs.values()], baseline_mean_seconds=base_seconds, seconds_share=share, faster=not missed, missed=missed,
                statuses=[None if r is None else r['status'] for r in runs.values()],
                alpha=[None if r is None else r['accepted_alpha'] for r in runs.values()], seg35_at_accepted=mean((r['at_accepted'] or {}).get('seg35') for r in certified.values()), seg4_at_accepted=mean((r['at_accepted'] or {}).get('seg4') for r in certified.values()),
                seg35_alpha1=mean(r['metrics']['seg35'] for r in read.values()), seg4_alpha1=mean(r['metrics']['seg4'] for r in read.values()), sdlogj_official_alpha1=mean(r['metrics']['sdlogj_official'] for r in read.values()), sdlogj_brain_alpha1=mean(r['metrics']['sdlogj_brain'] for r in read.values()),
                n_infeasible_at_1=[None if r is None else r['n_infeasible_at_1'] for r in runs.values()], min_lower_bound=[None if r is None else r['min_lower_bound'] for r in runs.values()], mean_seconds_labels=mean(r.get('seconds_labels') for r in runs.values() if r is not None), mean_seconds_channels=mean(r.get('seconds_channels') for r in runs.values() if r is not None),
                mean_seconds_native=mean((r.get('seconds_levels') or [None]*4)[-1] for r in runs.values() if r is not None), mean_seconds_ladder=mean(r.get('seconds_ladder') for r in runs.values() if r is not None), device_peak_reserved_gib=mean(r.get('device_peak_reserved_gib') for r in runs.values() if r is not None))


def e1_reading(rows):
    baseline = {p: rows.get((ef.BASELINE, p)) for p in core.PAIRS}
    baseline_read = {p: r for p, r in baseline.items() if r is not None and r['metrics'] is not None and r['accepted_alpha'] is not None and r['status'] == 'done'}
    deciding = [arm for arm, spec in ef.ARMS.items() if spec['decides']]
    arms = {arm: e1_arm(rows, arm, baseline_read) for arm in deciding}
    same_path = arms['F600']['mean_seconds'] if 'F600' in arms else None
    for r in arms.values():                         # only described: never in `faster`
        r['same_path_share'] = r['mean_seconds']/same_path if r['mean_seconds'] is not None and same_path else None
        r['passes_only_with_e2'] = r['faster'] and (r['same_path_share'] is None or r['same_path_share'] > SECONDS_SHARE)
    baseline_row = dict(n_read=len(baseline_read), seg35_at_accepted=mean(r['at_accepted']['seg35'] for r in baseline_read.values()), seg4_at_accepted=mean(r['at_accepted']['seg4'] for r in baseline_read.values()), seg35_alpha1=mean(r['metrics']['seg35'] for r in baseline_read.values()), seg4_alpha1=mean(r['metrics']['seg4'] for r in baseline_read.values()),
                        sdlogj_official_alpha1=mean(r['metrics']['sdlogj_official'] for r in baseline_read.values()), mean_seconds=mean(r['seconds_total'] for r in baseline.values() if r is not None), median_seconds=median([r['seconds_total'] for r in baseline.values() if r is not None]), per_pair_seconds=[None if r is None else r['seconds_total'] for r in baseline.values()], mean_seconds_channels=mean(r.get('seconds_channels') for r in baseline.values() if r is not None), alpha=[None if r is None else r['accepted_alpha'] for r in baseline.values()],
                        mean_seconds_native=mean((r.get('seconds_levels') or [None]*4)[-1] for r in baseline.values() if r is not None), mean_seconds_ladder=mean(r.get('seconds_ladder') for r in baseline.values() if r is not None))
    only_with_e2 = [arm for arm, r in arms.items() if r['passes_only_with_e2']]
    if len(baseline_read) < N:
        return dict(reading=READ_E1_UNDECIDABLE, faster_arms=[], faster_only_with_e2=[], baseline=baseline_row, arms=arms)
    faster = [arm for arm, r in arms.items() if r['faster']]
    if faster:
        reading = READ_FASTER
    elif all(r['complete'] for r in arms.values()):
        reading = READ_NOT_FASTER
    else:
        reading = READ_E1_GAPS
    return dict(reading=reading, faster_arms=faster, faster_only_with_e2=only_with_e2, baseline=baseline_row, arms=arms)


def e1_headline(e1):
    """The reading with the arms that pass only with E2 named: for whoever reads only the first line (EF_tools_复核 r3).  The reading itself does not change."""
    only = e1.get('faster_only_with_e2') or []
    return f"{e1['reading']}（靠 E2 的臂：{'、'.join(only)}）" if only else e1['reading']


def pair_row(short, row):
    """One pair of a repair arm, for the table: (d) of the review asks for the worst pair, (a) for the extra points."""
    if row is None:
        return dict(pair=short, status='没有运行')
    out = dict(pair=short, status=row['status'], reason=row.get('reason'), seconds_repair=row.get('seconds_repair'), rounds=row.get('rounds'), n_extra_points=row.get('n_extra_points'), certified_at_1=row.get('certified_at_1'))
    if row.get('metrics'):
        before, accepted = row['before'], row['before'].get('at_accepted_alpha') or {}
        fallback = row.get('fallback') or {}
        out.update(seg35_before=before['metrics']['seg35'], seg4_before=before['metrics']['seg4'], before_accepted_alpha=before.get('certificate_accepted_alpha'), seg35_before_accepted=accepted.get('seg35'), seg4_before_accepted=accepted.get('seg4'), seg35_attempt=row['after']['seg35'], seg4_attempt=row['after']['seg4'], delta_seg35=row['delta_seg35'], delta_seg4=row['delta_seg4'],
                   n_infeasible_before=before['n_infeasible_at_1'], n_infeasible_after=row['n_infeasible_after'], min_lower_bound_after=row['min_lower_bound_after'], n_changed_nodes=row['change']['n_changed_nodes'], max_change_mm=row['change']['max_change_mm'], fallback_alpha=fallback.get('accepted_alpha'), fallback_at_accepted=fallback.get('at_accepted_alpha'))
    return out


def e3_arm(repairs, arm):
    runs = {p: repairs.get((arm, p)) for p in core.PAIRS}
    read = {p: r for p, r in runs.items() if r is not None and r['metrics']}
    certified = [p for p, r in read.items() if r['certified_at_1']]
    d35 = {p: r['delta_seg35'] for p, r in read.items()}
    d4 = {p: r['delta_seg4'] for p, r in read.items()}
    delta = statistics.fmean(d35.values()) if d35 else None
    seconds_list = [r['seconds_repair'] for r in runs.values() if r is not None and r['seconds_repair'] is not None]
    seconds = statistics.fmean(seconds_list) if seconds_list else None
    missed = []
    if len(certified) < N:
        missed.append(f'α = 1 证书 {len(certified)}/{N}（读到 {len(read)}）')
    if delta is None or delta < SEG35_BAR or len(d35) < N:
        missed.append(f"修复尝试后 seg35 差 {'—' if delta is None else f'{delta:+.4f}'}（n = {len(d35)}，门 ≥ {SEG35_BAR}）")
    if seconds is None or seconds > REPAIR_SECONDS:
        missed.append(f"修复平均 {'—' if seconds is None else f'{seconds:.1f}'} s（门 ≤ {REPAIR_SECONDS:g}）")
    passed = {p: read[p] for p in certified}
    disagree = [core.short_name(p) for p, r in read.items() if r.get('run_says_passed') is not None and bool(r['run_says_passed']) != bool(r['certified_at_1'])]     # never expected: the evaluation counts
    return dict(arm=arm, eps=ef.REPAIR_ARMS[arm]['eps'], n_read=len(read), complete=len(read) == N, run_and_evaluation_disagree=disagree, n_certified_at_1=len(certified), delta_seg35=delta, delta_seg4=statistics.fmean(d4.values()) if d4 else None, per_pair_delta_seg35={core.short_name(p): v for p, v in d35.items()}, mean_seconds=seconds, max_seconds=max(seconds_list) if seconds_list else None, usable=not missed, missed=missed,
                seg35_before=mean(r['before']['metrics']['seg35'] for r in read.values()), seg4_before=mean(r['before']['metrics']['seg4'] for r in read.values()), seg35_attempt=mean(r['after']['seg35'] for r in read.values()), seg4_attempt=mean(r['after']['seg4'] for r in read.values()),
                n_certified_only=len(passed), seg35_certified_only=mean(r['after']['seg35'] for r in passed.values()), seg4_certified_only=mean(r['after']['seg4'] for r in passed.values()), seg35_before_certified_only=mean(r['before']['metrics']['seg35'] for r in passed.values()), seg4_before_certified_only=mean(r['before']['metrics']['seg4'] for r in passed.values()),
                per_pair=[pair_row(core.short_name(p), runs[p]) for p in core.PAIRS],
                sdlogj_official_after=mean(r['after']['sdlogj_official'] for r in read.values()), sdlogj_brain_after=mean(r['after']['sdlogj_brain'] for r in read.values()), sdlogj_official_before=mean(r['before']['metrics']['sdlogj_official'] for r in read.values()), sdlogj_brain_before=mean(r['before']['metrics']['sdlogj_brain'] for r in read.values()),
                n_infeasible_before=[None if p not in read else read[p]['before']['n_infeasible_at_1'] for p in core.PAIRS], n_infeasible_after=[None if p not in read else read[p]['n_infeasible_after'] for p in core.PAIRS], min_lower_bound_after=[None if p not in read else read[p]['min_lower_bound_after'] for p in core.PAIRS],
                changed_nodes=[None if p not in read else read[p]['change']['n_changed_nodes'] for p in core.PAIRS], max_change_mm=[None if p not in read else read[p]['change']['max_change_mm'] for p in core.PAIRS], rounds=[None if runs[p] is None else runs[p]['rounds'] for p in core.PAIRS], reasons=[None if runs[p] is None else runs[p]['reason'] for p in core.PAIRS])


def e3_reading(repairs):
    arms = {arm: e3_arm(repairs, arm) for arm, spec in ef.REPAIR_ARMS.items() if spec['decides']}
    usable = [arm for arm, r in arms.items() if r['usable']]
    if usable:
        reading = READ_REPAIR
    elif all(r['complete'] for r in arms.values()):
        reading = READ_NO_REPAIR
    else:
        reading = READ_E3_GAPS
    described = []
    for arm, spec in ef.REPAIR_ARMS.items():
        if spec['decides']:
            continue
        for pair in ef.LONG_REPAIR_PAIRS:
            row = repairs.get((arm, pair))
            if row is None:
                described.append(dict(arm=arm, pair=core.short_name(pair), eps=spec['eps'], status='没有运行'))
            else:
                described.append(dict(pair=core.short_name(pair), **{k: row.get(k) for k in ('arm', 'eps', 'status', 'reason', 'seconds_repair', 'rounds', 'n_extra_points', 'certified_at_1', 'n_infeasible_after', 'min_lower_bound_after', 'delta_seg35', 'delta_seg4', 'change', 'fallback', 'before', 'after')}))
    return dict(reading=reading, usable_arms=usable, arms=arms, described_long=described, rule_of_described='长档两对（0439-0440、0455-0456）只描述，不进读数（总负责人 2026-11-04）')


def controls(root):
    """The fields of Q2 and F600 against the runs of mc0p (control_bitwise), and the scale numbers of the K = 35 arms of a pair against those of Q2: a record of E2 on the real files.  Different is reported, nothing stops."""
    out = dict(fields={}, scale={}, not_bitwise=[])
    for (arm, pair), (source, tier) in ef.CONTROLS.items():
        cell = ef.Cell(arm, ef.ARMS[arm]['tier'], pair)
        mine = read_json(core.run_dir(root, cell)/'run.json') or {}
        theirs = read_json(ef.RESULTS/source/'runs'/f'Q2_{tier}'/cell.short/'run.json') or {}
        a, b = (mine.get('field') or {}).get('u_sha256'), (theirs.get('field') or {}).get('u_sha256')
        same = None if a is None or b is None else a == b
        out['fields'][cell.run_id] = dict(against=f'{source}/runs/Q2_{tier}/{cell.short}', control_bitwise=same)
        if same is False:
            out['not_bitwise'].append(cell.run_id)
    for pair in core.PAIRS:
        base = read_json(core.run_dir(root, ef.Cell(ef.BASELINE, 'long', pair))/'run.json') or {}
        base_scale = base.get('scale') or {}
        for arm, spec in ef.ARMS.items():
            if spec['channels'] != 'onehot':
                continue
            cell = ef.Cell(arm, spec['tier'], pair)
            mine = read_json(core.run_dir(root, cell)/'run.json') or {}
            scale = mine.get('scale') or {}
            out['scale'][cell.run_id] = {key+'_equal_Q2': None if scale.get(key) is None or base_scale.get(key) is None else scale[key] == base_scale[key] for key in ('D_multiplier', 'g_norm_Q0', 'g_norm_unscaled')}
    return out


def decide(root, stops=None):
    rows, repairs = collect(root)
    bad = sorted(r['run_id'] for r in rows.values() if r['metrics'] is not None and r['oracle'] is not False)
    if bad:
        raise SystemExit(f'run.json is not oracle false for {bad}; nothing written')
    return dict(schema=SCHEMA, is_conclusion=False, next_step='交总负责人与 BZ', oracle=False, note=NOTE, arms_stopped=stops or {}, rules=dict(seg35_bar=SEG35_BAR, seconds_share=SECONDS_SHARE, repair_seconds=REPAIR_SECONDS, n=N),
                e1=e1_reading(rows), e3=e3_reading(repairs), controls=controls(root)), rows, repairs


def _f(value, nd=4):
    if value is None:
        return '—'
    return f'{value:.{nd}f}' if isinstance(value, float) else str(value)


def _signed(value, nd=4):
    return '—' if value is None else f'{value:+.{nd}f}'


def _list(values, nd=4):
    return '、'.join(_f(v, nd) for v in values)


def summary_text(decision):
    e1, e3, c = decision['e1'], decision['e3'], decision['controls']
    b = e1['baseline']
    L = ['# EF 读数（由 decide_ef.py 写出）', '', f"**E1：{e1_headline(e1)}**；**E3：{e3['reading']}**（不是结论，下一步交总负责人与 BZ）", '', NOTE, '']
    if decision['arms_stopped']:
        L += [f"停掉的臂：{json.dumps(decision['arms_stopped'], ensure_ascii=False)}", '']
    L += ['## E1：入口的步数与 K（接受 α 口径）', '',
          f"基线 Q2 长档（本批，旧通道路径）：接受 α 下 seg35 / seg4 {_f(b['seg35_at_accepted'])} / {_f(b['seg4_at_accepted'])}，α = 1 {_f(b['seg35_alpha1'])} / {_f(b['seg4_alpha1'])}，官方 SDlogJ（α = 1）{_f(b['sdlogj_official_alpha1'])}，平均 {_f(b['mean_seconds'], 1)} s、中位 {_f(b['median_seconds'], 1)} s、逐对 {_list(b['per_pair_seconds'], 1)}（建通道 {_f(b['mean_seconds_channels'], 1)}，原生层 {_f(b['mean_seconds_native'], 1)}，证书阶梯 {_f(b['mean_seconds_ladder'], 1)}，后者在评测单元），α {b['alpha']}", '',
          '| 臂 | 判定 | 证书 | 接受 α 下 seg35 / seg4 | 差 seg35 / seg4（对 Q2） | 秒（均值 / 中位） | 逐对秒 | 秒数比（对 Q2，进判定） | 对 F600 的同路径比值（只描述） | 分段（前两项从运行开始累计）：到标签读完 / 到通道建完 / 原生层 / 证书阶梯 | α = 1 seg35 / seg4 | 官方 / 脑内 SDlogJ（α = 1） | α = 1 不可行单元 | α | 运行状态 | 差在哪 |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for arm, r in e1['arms'].items():
        verdict = '可以更快（靠 E2）' if r['passes_only_with_e2'] else '可以更快' if r['faster'] else '不能更快'
        L.append(f"| {arm} | {verdict} | {r['n_certified']}/{N} | {_f(r['seg35_at_accepted'])} / {_f(r['seg4_at_accepted'])} | {_signed(r['delta_seg35_at_accepted'])} / {_signed(r['delta_seg4_at_accepted'])} | {_f(r['mean_seconds'], 1)} / {_f(r['median_seconds'], 1)} | {_list(r['per_pair_seconds'], 1)} | {_f(r['seconds_share'], 3)} | {_f(r['same_path_share'], 3)} | {_f(r['mean_seconds_labels'], 1)} / {_f(r['mean_seconds_channels'], 1)} / {_f(r['mean_seconds_native'], 1)} / {_f(r['mean_seconds_ladder'], 1)} | {_f(r['seg35_alpha1'])} / {_f(r['seg4_alpha1'])} | {_f(r['sdlogj_official_alpha1'])} / {_f(r['sdlogj_brain_alpha1'])} | {_list(r['n_infeasible_at_1'])} | {r['alpha']} | {'、'.join(sorted({str(x) for x in r['statuses']}))} | {'；'.join(r['missed']) or '—'} |")
    L += ['', SAME_PATH_NOTE, '', K8_NOTE, '', '## E2：对照（只记录）', '', '场的 u 对 mc0p 的同一格（control_bitwise）：'+'；'.join(f"{k} 对 {v['against']}：{v['control_bitwise']}" for k, v in c['fields'].items()), '', f"不是逐位相同的：{c['not_bitwise'] or '无'}", '',
          f"尺度数字对同一对的 Q2 不相等的格：{[k for k, v in (c.get('scale') or {}).items() if any(x is False for x in v.values())] or '无'}（没有读到的不算不等）", '',
          '## E3：局部修复', '', '只有细档 5 对进读数。修复前是 α = 1 的未认证场（taper 之后）；修复尝试后的证书和 Dice 都由评测单元从场算。秒数是修复函数内部（定位、求解、每轮证书），numba 已预热（含只读数组），不含读文件。', '', E3_ATTEMPT_NOTE, '',
          '| 臂 | ε | 判定 | α = 1 证书 | seg35 / seg4 修复前 | 修复尝试后（含未过证书） | 差 | 只算过证书的对：n；修复前 → 修复后 seg35 / seg4 | 官方 SDlogJ 前 / 尝试后 | 脑内 SDlogJ 前 / 尝试后 | 平均 / 最长秒 | 运行与评测不一致 | 差在哪 |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for arm, r in e3['arms'].items():
        L.append(f"| {arm} | {r['eps']} | {'修复可用' if r['usable'] else '修复不可用'} | {r['n_certified_at_1']}/{N} | {_f(r['seg35_before'])} / {_f(r['seg4_before'])} | {_f(r['seg35_attempt'])} / {_f(r['seg4_attempt'])} | {_signed(r['delta_seg35'])} / {_signed(r['delta_seg4'])} | {r['n_certified_only']}；{_f(r['seg35_before_certified_only'])} / {_f(r['seg4_before_certified_only'])} → {_f(r['seg35_certified_only'])} / {_f(r['seg4_certified_only'])} | {_f(r['sdlogj_official_before'])} / {_f(r['sdlogj_official_after'])} | {_f(r['sdlogj_brain_before'])} / {_f(r['sdlogj_brain_after'])} | {_f(r['mean_seconds'], 1)} / {_f(r['max_seconds'], 1)} | {r['run_and_evaluation_disagree'] or '无'} | {'；'.join(r['missed']) or '—'} |")
    L += ['', '### 细档逐对', '', '| 臂 | 对 | 状态 | 证书 α = 1 | 修复前 seg35 / seg4（α = 1） | 修复前接受 α：seg35 / seg4 | 修复尝试后 seg35 / seg4 | 差 seg35 / seg4 | 没过时回退规则的 α：seg35 / seg4 | 秒 | 轮数 | W 点数 | 不可行单元 前 → 后 | 最小认证下界 | 改动节点数 | 最大位移变化 mm | 结束原因 |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for arm, r in e3['arms'].items():
        for row in r['per_pair']:
            fallback = row.get('fallback_at_accepted') or {}
            L.append(f"| {arm} | {row['pair']} | {row['status']} | {row.get('certified_at_1')} | {_f(row.get('seg35_before'))} / {_f(row.get('seg4_before'))} | {row.get('before_accepted_alpha')}：{_f(row.get('seg35_before_accepted'))} / {_f(row.get('seg4_before_accepted'))} | {_f(row.get('seg35_attempt'))} / {_f(row.get('seg4_attempt'))} | {_signed(row.get('delta_seg35'))} / {_signed(row.get('delta_seg4'))} | {'—' if row.get('certified_at_1') else row.get('fallback_alpha')}：{_f(fallback.get('seg35'))} / {_f(fallback.get('seg4'))} | {_f(row.get('seconds_repair'), 1)} | {row.get('rounds')} | {row.get('n_extra_points')} | {row.get('n_infeasible_before')} → {row.get('n_infeasible_after')} | {_f(row.get('min_lower_bound_after'), 5)} | {row.get('n_changed_nodes')} | {_f(row.get('max_change_mm'), 3)} | {row.get('reason')} |")
    described = e3['described_long']
    with_points = sorted({f"{r['arm']} {r['pair']}" for arm in e3['arms'].values() for r in ([dict(p, arm=arm['arm']) for p in arm['per_pair']]) if (r.get('n_extra_points') or 0) > 0}|{f"{r['arm']} {r['pair']}" for r in described if (r.get('n_extra_points') or 0) > 0})
    L += ['', f"用到 W 点（第 2 轮起加的格点最小点）的格：{with_points or '无'}。这些格的目标函数不再只是任务书的 53 点（result.md 要写明）。"]
    L += ['', "### 长档两对（只描述，不进读数）", '', '| 对 | 臂 | ε | 状态 | 秒 | 轮数 | W 点数 | 证书 α = 1 | 修复前 seg35 / seg4（α = 1） | 修复前接受 α 与 seg35 / seg4 | 修复尝试后 seg35 / seg4 | 差（对 α = 1） | 改动节点数 | 最大位移变化 mm | 最小认证下界 |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in described:
        before = r.get('before') or {}
        accepted = before.get('at_accepted_alpha') or {}
        after = r.get('after') or {}
        L.append(f"| {r['pair']} | {r['arm']} | {r['eps']} | {r['status']} | {_f(r.get('seconds_repair'), 1)} | {r.get('rounds')} | {r.get('n_extra_points')} | {r.get('certified_at_1')} | {_f((before.get('metrics') or {}).get('seg35'))} / {_f((before.get('metrics') or {}).get('seg4'))} | {accepted.get('alpha')}：{_f(accepted.get('seg35'))} / {_f(accepted.get('seg4'))} | {_f(after.get('seg35'))} / {_f(after.get('seg4'))} | {_signed(r.get('delta_seg35'))} / {_signed(r.get('delta_seg4'))} | {(r.get('change') or {}).get('n_changed_nodes')} | {_f((r.get('change') or {}).get('max_change_mm'), 3)} | {_f(r.get('min_lower_bound_after'), 5)} |")
    return '\n'.join(L)+'\n'


def write_long(path, rows, repairs):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(('run_id', 'experiment', 'arm', 'tier', 'pair', 'status', 'oracle', 'field', 'alpha', 'measure', 'value'))
        for (arm, pair), row in sorted(rows.items()):
            if row['metrics'] is None:
                continue
            for m in MEASURES:
                w.writerow((row['run_id'], 'E1', arm, row['tier'], pair, row['status'], 'false', 'uncertified_alpha1', 1., m, row['metrics'][m]))
            if row['at_accepted'] is not None:
                for m in ('seg35', 'seg4'):
                    w.writerow((row['run_id'], 'E1', arm, row['tier'], pair, row['status'], 'false', 'accepted_alpha', row['accepted_alpha'], m, row['at_accepted'][m]))
        for (arm, pair), row in sorted(repairs.items()):
            if not row['metrics']:
                continue
            for m in MEASURES:
                w.writerow((row['run_id'], 'E3', arm, row['tier'], pair, row['status'], 'false', 'before_alpha1', 1., m, row['before']['metrics'][m]))
                w.writerow((row['run_id'], 'E3', arm, row['tier'], pair, row['status'], 'false', 'repaired' if row['certified_at_1'] else 'attempt_uncertified', 1., m, row['after'][m]))


def set_aside(out, names):
    """decide_mc0p.set_aside, with a number when two readings fall in one second (the queue writes two, one per phase)."""
    if any((out/n).exists() for n in names):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        aside, k = out/f'previous_{stamp}', 1
        while aside.exists():
            aside, k = out/f'previous_{stamp}_{k}', k+1
        aside.mkdir(parents=True)
        for n in names:
            if (out/n).exists():
                shutil.move(str(out/n), str(aside/n))
    out.mkdir(parents=True, exist_ok=True)


def write_analysis(root, stops=None):
    """analysis/: an earlier version moved to analysis/previous_<UTC>/, never overwritten."""
    root = Path(root)
    decision, rows, repairs = decide(root, stops)
    out = root/'analysis'
    set_aside(out, ('decision_ef.json', 'long.csv', 'summary_ef.md'))
    (out/'decision_ef.json').write_text(json.dumps(decision, indent=1, ensure_ascii=False)+'\n')
    (out/'summary_ef.md').write_text(summary_text(decision), encoding='utf-8')
    write_long(out/'long.csv', rows, repairs)
    return decision
