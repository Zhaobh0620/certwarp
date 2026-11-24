"""FL's reading (runs/20261105-fl/plan.md, by code; a label for the lead and BZ, not a conclusion): analysis/decision_fl.json, summary_fl.md, long.csv; the code's choices analysis/x_selection.json, repair_selection_1.json, repair_selection_2.json.

The premise of a checkpoint (arm, step), first 5 pairs, repaired fields, every number from the evaluation unit:
    5/5 pass the certificate at alpha = 1  and  mean official SDlogJ <= .10  and  on every pair the task clock (channels + T88 + repair with its warm-up) <= 330 s.
The reading, on the best non-oracle checkpoint that satisfies it (mean seg35 of the first 5 pairs):
    >= .8668  这条路本身接近 .88;  >= .8500  值得扩到其余 14 对;  >= .8430  中间带（只留作 S_F 的入口）;  below  到不了标量原生层的水平.
Some checkpoint was repaired and none satisfies the premise: 不可判, with what missed.
No checkpoint was repaired: the table of the lead's ruling (FL_tools_复核 blocking 1, plan revision 1, before any result), in this order:
    1  the third phase is not settled: no x_selection.json, or a sigma chosen and X at 4800 steps not read 5/5      不可判 (X may still pass)
    2  some non-oracle checkpoint of the first phase not read 5/5                                                   不可判 (not read is not below the line)
    3  every non-oracle checkpoint read and every one <= .8430                                                       到不了标量原生层的水平; `best` stays None, the highest checkpoint in the reason
    4  a candidate (2400 steps, or X at 4800) above .8430, and all of those over official SDlogJ .10                 不可判
    5  a candidate above .8430 with official SDlogJ <= .10, and no repair on record                                  不可判 (the choice and the records disagree)
    6  only checkpoints at 600 or 1200 steps above .8430                                                             不可判 (those are not candidates), for the lead
Only repairs of a chosen checkpoint enter the reading; RF150, RF300 and the oracle arm never do.
The choices.  X's sigma: among F, M, S, T at 2400 steps (5/5 read), official SDlogJ <= .10, not an arm whose 2400 steps were repaired without 5/5 certificates at alpha = 1 (the lead, plan revision 1), the highest seg35.  The repairs: among the non-oracle checkpoints at 2400 or 4800 steps (5/5 read), seg35 > .8430 and official SDlogJ <= .10, the 3 with the highest seg35 at alpha = 1 (uncertified).  Ties go to the first in the plan's order.
A cell is read only with status done.  Everything else is only described.
"""
import csv
import json
from pathlib import Path
import statistics

from decide_ef import read_json, set_aside
import fl_core as fl
import mc0_core as core

SCHEMA = 'bayesreg.fl.decision.v1'
N = len(core.PAIRS)
BAR_NEAR, BAR_EXPAND, BAR_SP, SDLOGJ_MAX = .8668, .8500, .8430, .10
SELECT_STEP = 2400
MEASURES = ('seg35', 'seg4', 'hd95', 'sdlogj_official', 'sdlogj_brain', 'small_six', 'large_six', 'other_23', 'cortex', 'white_matter')
TERMS = ('data', 'geometry', 'floor', 'curvature', 'min_det')
S_P = dict(seg35=.8430, seg4=.8984, small_six=.5947, cortex=.8553)     # S_P's final state, first 5 pairs: runs/20261104-seed/result.md, the brief
READ_NEAR, READ_EXPAND, READ_MIDDLE, READ_BELOW, READ_UNDECIDABLE = '这条路本身接近 .88', '值得扩到其余 14 对', '中间带（只留作 S_F 的入口）', '到不了标量原生层的水平', '不可判'
NOTE = ('开发集单次观测，通道来自 unet35_v1_ens 的预测（oracle: false），真值标签只进评测。**不是 F_B，没有经认证的下降，只有终态过证书；不是方法，不进结果表，进方法先改合同，由 BZ 定。** n = 5，前 5 对之前选配用过，主读数放在其余 14 对。')
ORACLE_NOTE = 'O：K = 35 真值 one-hot，oracle: true，隔离在 oracle_isolated，只作上限，不进结果表，不做修复，不进读数。'
SIGMA_NOTE = ('σ_field 的名义值不是平滑量：核是截断采样的，每步方差见表。**读法不按 σ 插值**；T（σ .25）几乎不平滑，它失败只能读"几乎不平滑不行"，不能读"σ .25 不行"。官方 SDlogJ（中心差分）、脑内 SDlogJ 和 T88 的几何项（单元中心）都看不见棋盘模态，所以这几个臂上 SDlogJ ≤ .10 这道门可能没有约束，真正的指标是不可行单元数和修复能否过证书〔推断，没有实测〕。')


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


# ---- what the runs and the evaluation say -----------------------------------------------------------------------------------------------------
def field_rows(root, cell):
    """{step: row} of one field run; empty if it never started.  Certificates and Dice from metrics_<step>.json, written by the evaluation unit from the field."""
    d = fl.run_dir(root, cell)
    run, done = read_json(d/'run.json'), read_json(d/'queue_done.json')
    if run is None and done is None:
        return {}
    run, done = run or {}, done or {}
    status = run.get('status', done.get('state', 'failed'))
    optimizer = run.get('optimizer') or {}
    rows = {}
    for step in fl.checkpoints_of(cell):
        checkpoint = (run.get('checkpoints') or {}).get(str(step)) or {}
        metrics = read_json(d/f'metrics_{step:04d}.json')
        row = dict(run_id=cell.run_id, arm=cell.arm, tier=cell.tier, step=step, pair=cell.pair, status=status, oracle=run.get('oracle'), k=fl.ARMS[cell.arm]['k'], sigma_field=(run.get('config') or {}).get('sigma_field'), seconds_task=checkpoint.get('seconds_task'), seconds_io=checkpoint.get('seconds_io'), terms=checkpoint.get('terms'), u_sha256=checkpoint.get('u_sha256'),
                   seconds_levels=[level['seconds'] for level in optimizer.get('levels') or []], seconds_channels=run.get('seconds_channels'), device_peak_reserved_gib=run.get('device_peak_reserved_gib'), host_peak_rss_gib=run.get('host_peak_rss_gib'), metrics=None, certified=None, n_infeasible=None, min_lower_bound=None)
        if metrics is not None:
            attempt = metrics['certificate']['attempts'][0]
            row.update(metrics={m: metrics['metrics'][m] for m in MEASURES}, certified=metrics['certificate']['accepted_alpha'] == 1., n_infeasible=metrics['certificate']['n_infeasible_at_1'], min_lower_bound=attempt['min_jd_lower_bound'], seconds_certificate=metrics.get('seconds_certificate'), seconds_metrics=metrics.get('seconds_metrics'))
        rows[step] = row
    return rows


def repair_row(root, cell):
    """One repair; None if it never started.  Passing is the evaluation's certificate, from the field."""
    d = fl.run_dir(root, cell)
    repair, metrics, done = read_json(d/'repair.json'), read_json(d/'repair_metrics.json'), read_json(d/'queue_done.json')
    if repair is None and metrics is None and done is None:
        return None
    repair, done = repair or {}, done or {}
    rounds = repair.get('rounds') or []
    source = repair.get('source') or {}
    seconds = [source.get('seconds_task'), repair.get('seconds_repair'), repair.get('seconds_warmup')]
    row = dict(run_id=cell.run_id, arm=cell.base, repair_arm=cell.arm, tier=cell.tier, step=fl.steps_of(cell), pair=cell.pair, eps=repair.get('eps'), status=repair.get('status', done.get('state', 'failed')), oracle=repair.get('oracle'), run_says_passed=repair.get('passed'), reason=repair.get('reason'), seconds_repair=repair.get('seconds_repair'), seconds_warmup=repair.get('seconds_warmup'), seconds_source=source.get('seconds_task'),
               task_clock=None if any(s is None for s in seconds) else sum(seconds), rounds=len(rounds), n_extra_points=max((r.get('n_extra_points', 0) for r in rounds), default=0), n_free=None if not rounds else rounds[-1].get('n_free'), metrics=metrics is not None, certified_at_1=None, device_peak_reserved_gib=repair.get('device_peak_reserved_gib'))
    if metrics is not None:
        after = metrics['after']
        row.update(certified_at_1=after['certificate']['accepted_alpha'] == 1., n_infeasible_before=metrics['before']['n_infeasible_at_1'], n_infeasible_after=after['certificate']['n_infeasible_at_1'], min_lower_bound_after=after['certificate']['attempts'][0]['min_jd_lower_bound'],
                   before=metrics['before']['metrics'], after=after['metrics'], change=metrics['change'], delta_seg35=after['metrics']['seg35']-metrics['before']['metrics']['seg35'], delta_seg4=after['metrics']['seg4']-metrics['before']['metrics']['seg4'])
    return row


def collect(root):
    """(fields {(arm, tier, step, pair): row}, repairs {(arm, tier, step, pair): row}); the repairs under the arm and step of their source, tier 'first5' or 'ef' or 'rest<steps>'."""
    fields, repairs = {}, {}
    rest_fields, rest_repairs = fl.rest_cells(root)
    for cell in fl.plan_fields()+fl.plan_x()+rest_fields:
        for step, row in field_rows(root, cell).items():
            fields[(cell.arm, cell.tier, step, cell.pair)] = row
    for cell in fl.plan_ef_repairs()+fl.repair_cells(fl.chosen_repairs(root))+rest_repairs:
        row = repair_row(root, cell)
        if row is not None:
            repairs[(cell.base, fl.FIRST if cell.tier.startswith('c') else cell.tier, row['step'], cell.pair)] = row
    return fields, repairs


# ---- summaries of a checkpoint --------------------------------------------------------------------------------------------------------------------
def arm_step(fields, arm, step, *, tier=fl.FIRST, pairs=core.PAIRS):
    """The means over the pairs of one checkpoint.  Read: metrics there and status done."""
    runs = {p: fields.get((arm, tier, step, p)) for p in pairs}
    read = {p: r for p, r in runs.items() if r is not None and r['metrics'] is not None and r['status'] == 'done'}
    out = dict(arm=arm, tier=tier, step=step, n_read=len(read), complete=len(read) == len(pairs), sigma_field=next((r['sigma_field'] for r in read.values()), None), k=fl.ARMS[arm]['k'], statuses=[None if r is None else r['status'] for r in runs.values()])
    out.update({m: mean(r['metrics'][m] for r in read.values()) for m in MEASURES})
    out.update(per_pair_seg35={core.short_name(p): r['metrics']['seg35'] for p, r in read.items()}, per_pair_seg4={core.short_name(p): r['metrics']['seg4'] for p, r in read.items()}, n_infeasible=[None if r is None else r['n_infeasible'] for r in runs.values()], n_certified=sum(r['certified'] is True for r in read.values()), min_lower_bound=[None if r is None else r['min_lower_bound'] for r in runs.values()],
               seconds_task=mean(r['seconds_task'] for r in read.values()), max_seconds_task=max((r['seconds_task'] for r in read.values() if r['seconds_task'] is not None), default=None), seconds_channels=mean(r['seconds_channels'] for r in read.values()),
               seconds_levels=[mean(r['seconds_levels'][i] if len(r['seconds_levels']) > i else None for r in read.values()) for i in range(4)], device_peak_reserved_gib=mean(r['device_peak_reserved_gib'] for r in read.values()), host_peak_rss_gib=mean(r['host_peak_rss_gib'] for r in read.values()))
    for term in TERMS:
        out[term] = mean((r['terms'] or {}).get(term) for r in read.values())
    return out


def paired(fields, arm, reference, step, measures=('seg35', 'seg4', 'small_six', 'large_six', 'other_23', 'cortex', 'white_matter', 'sdlogj_official'), *, reference_step=None):
    """arm - reference on the pairs both have read; only described."""
    out = dict(arm=arm, reference=reference, step=step, reference_step=step if reference_step is None else reference_step)
    for m in measures:
        rows = [(fields.get((arm, fl.FIRST, step, p)), fields.get((reference, fl.FIRST, out['reference_step'], p))) for p in core.PAIRS]
        diffs = [a['metrics'][m]-b['metrics'][m] for a, b in rows if a and b and a['metrics'] and b['metrics'] and a['status'] == 'done' and b['status'] == 'done' and a['metrics'][m] is not None and b['metrics'][m] is not None]     # a group with no label present has no Dice
        out[m] = dict(mean=mean(diffs), positive=sum(d > 0 for d in diffs), n=len(diffs))
    return out


def repair_summary(repairs, arm, step, *, tier=fl.FIRST, pairs=core.PAIRS):
    """A repaired checkpoint: the premise of the reading, and the description."""
    rows = {p: repairs.get((arm, tier, step, p)) for p in pairs}
    read = {p: r for p, r in rows.items() if r is not None and r['metrics']}
    certified = {p: r for p, r in read.items() if r['certified_at_1']}
    seg35, sdlogj = mean(r['after']['seg35'] for r in read.values()), mean(r['after']['sdlogj_official'] for r in read.values())
    clocks = [r['task_clock'] for r in rows.values() if r is not None and r['task_clock'] is not None]
    n = len(pairs)
    missed = []
    if len(certified) < n:
        missed.append(f'修复后 α = 1 证书 {len(certified)}/{n}（读到 {len(read)}）')
    if sdlogj is None or sdlogj > SDLOGJ_MAX or len(read) < n:
        missed.append(f"官方 SDlogJ 均值 {'—' if sdlogj is None else f'{sdlogj:.4f}'}（门 ≤ {SDLOGJ_MAX}，读到 {len(read)}）")
    if len(clocks) < n or max(clocks) > fl.TASK_BUDGET_SECONDS:
        missed.append(f"任务时钟最长 {'—' if not clocks else f'{max(clocks):.1f}'} s（门：每对 ≤ {fl.TASK_BUDGET_SECONDS:g} s，有数的 {len(clocks)} 对）")
    disagree = [core.short_name(p) for p, r in read.items() if r.get('run_says_passed') is not None and bool(r['run_says_passed']) != bool(r['certified_at_1'])]     # never expected: the evaluation counts
    out = dict(arm=arm, step=step, tier=tier, n_read=len(read), complete=len(read) == n, n_certified_at_1=len(certified), satisfies_premise=not missed, missed=missed, run_and_evaluation_disagree=disagree, mean_task_clock=mean(clocks), max_task_clock=max(clocks, default=None),
               mean_seconds_repair=mean(r['seconds_repair'] for r in rows.values() if r is not None), max_seconds_repair=max((r['seconds_repair'] for r in rows.values() if r is not None and r['seconds_repair'] is not None), default=None), mean_seconds_warmup=mean(r['seconds_warmup'] for r in rows.values() if r is not None),
               per_pair=[per_pair_repair(core.short_name(p), rows[p]) for p in pairs], delta_seg35=mean(r['delta_seg35'] for r in read.values()), delta_seg4=mean(r['delta_seg4'] for r in read.values()), n_certified_only=len(certified), seg35_certified_only=mean(r['after']['seg35'] for r in certified.values()), seg4_certified_only=mean(r['after']['seg4'] for r in certified.values()))
    out.update({m: mean(r['after'][m] for r in read.values()) for m in MEASURES})
    out.update({f'{m}_before': mean(r['before'].get(m) for r in read.values()) for m in MEASURES})
    return out


def per_pair_repair(short, row):
    if row is None:
        return dict(pair=short, status='没有运行')
    out = dict(pair=short, status=row['status'], reason=row.get('reason'), seconds_repair=row.get('seconds_repair'), seconds_warmup=row.get('seconds_warmup'), task_clock=row.get('task_clock'), rounds=row.get('rounds'), n_extra_points=row.get('n_extra_points'), n_free=row.get('n_free'), certified_at_1=row.get('certified_at_1'))
    if row.get('metrics'):
        out.update(n_infeasible_before=row['n_infeasible_before'], n_infeasible_after=row['n_infeasible_after'], min_lower_bound_after=row['min_lower_bound_after'], seg35_before=row['before']['seg35'], seg4_before=row['before']['seg4'], seg35_after=row['after']['seg35'], seg4_after=row['after']['seg4'], sdlogj_official_after=row['after']['sdlogj_official'], n_changed_nodes=row['change']['n_changed_nodes'], max_change_mm=row['change']['max_change_mm'])
    return out


# ---- the code's choices -----------------------------------------------------------------------------------------------------------------------
def select_x(fields, repairs=None, chosen=()):
    """X's sigma: among F, M, S, T at 2400 steps, 5/5 read, official SDlogJ <= .10, the highest seg35; the first in F, M, S, T on a tie.  An arm whose 2400 steps were chosen for repair and have not 5/5 certificates at alpha = 1 by the evaluation is out, and the next one is taken (the lead, plan revision 1, on FL_tools_复核 important 2): 4800 steps would have more infeasible cells.  None when no arm qualifies."""
    rows = []
    repaired = {c['arm'] for c in chosen if c['step'] == SELECT_STEP}
    for arm in fl.SIGMA_ARMS:
        r = arm_step(fields, arm, SELECT_STEP)
        certified = repair_summary(repairs or {}, arm, SELECT_STEP)['n_certified_at_1'] if arm in repaired else None
        failed_repair = arm in repaired and certified < N
        rows.append(dict(arm=arm, sigma_field=fl.ARMS[arm]['sigma_field'], variance_per_step=fl.field_kernel(fl.ARMS[arm]['sigma_field'])['variance_per_step'], n_read=r['n_read'], seg35=r['seg35'], seg4=r['seg4'], sdlogj_official=r['sdlogj_official'], repaired=arm in repaired, n_certified_at_1=certified, eligible=r['complete'] and r['sdlogj_official'] <= SDLOGJ_MAX and not failed_repair))
    eligible = [r for r in rows if r['eligible']]
    best = max(eligible, key=lambda r: r['seg35']) if eligible else None
    return dict(schema=SCHEMA+'.x_selection', rule=f'F、M、S、T 在 {SELECT_STEP} 步、5/5 读到、官方 SDlogJ ≤ {SDLOGJ_MAX}，做过修复的要 α = 1 证书 5/5，seg35 最高', candidates=rows, arm=None if best is None else best['arm'], sigma_field=None if best is None else best['sigma_field'])


def select_repairs(fields, candidates, *, not_candidates=()):
    """The non-oracle checkpoints with seg35 > .8430 and official SDlogJ <= .10 (5/5 read, alpha = 1, uncertified), the 3 highest seg35, in the order of `candidates` on a tie."""
    rows = []
    for arm, step in candidates:
        if arm in fl.ORACLE_ARMS or step not in fl.REPAIR_STEPS:
            raise ValueError(f'{arm} {step} is not a candidate of the repairs')
        r = arm_step(fields, arm, step)
        rows.append(dict(arm=arm, step=step, n_read=r['n_read'], seg35=r['seg35'], seg4=r['seg4'], sdlogj_official=r['sdlogj_official'], passes=r['complete'] and r['seg35'] > BAR_SP and r['sdlogj_official'] <= SDLOGJ_MAX))
    ranked = sorted((r for r in rows if r['passes']), key=lambda r: -r['seg35'])
    chosen = [dict(arm=r['arm'], step=r['step']) for r in ranked[:fl.REPAIR_TOP]]
    return dict(schema=SCHEMA+'.repair_selection', rule=f'非 oracle 臂的 2400 或 4800 步检查点，5/5 读到，α = 1 seg35 > {BAR_SP} 且官方 SDlogJ ≤ {SDLOGJ_MAX}，取 seg35 最高的 {fl.REPAIR_TOP} 个', candidates=rows, not_candidates=list(not_candidates), chosen=chosen, eps=fl.REPAIR_EPS)


def ensure_selection(root, name, make):
    """The code's choice, once: a file that is there is never written again, whatever the records say by then."""
    existing = fl.read_selection(root, name)
    if existing is not None:
        return existing
    fields, repairs = collect(root)
    value = make(fields, repairs)
    fl.write_selection(root, name, value)
    return value


def ensure_x_selection(root):
    return ensure_selection(root, 'x_selection.json', lambda fields, repairs: select_x(fields, repairs, fl.chosen_repairs(root)))


def ensure_repair_selection(root, round_):
    """Round 1: the 2400-step checkpoints of the first phase.  Round 2: X at 4800 steps; X at 2400 is the control of the arm it took sigma from, the same field."""
    if round_ == 1:
        return ensure_selection(root, 'repair_selection_1.json', lambda fields, repairs: select_repairs(fields, [(arm, SELECT_STEP) for arm in fl.FIRST_PHASE]))
    return ensure_selection(root, 'repair_selection_2.json', lambda fields, repairs: select_repairs(fields, [('X', 4800)], not_candidates=[dict(arm='X', step=SELECT_STEP, reason='与它取 σ 的臂的 2400 步检查点是同一个场（控制 x2400）；不重复修复')]))


# ---- the reading ------------------------------------------------------------------------------------------------------------------------------
def reading_line(seg35):
    if seg35 >= BAR_NEAR:
        return READ_NEAR
    if seg35 >= BAR_EXPAND:
        return READ_EXPAND
    if seg35 >= BAR_SP:
        return READ_MIDDLE
    return READ_BELOW


def nothing_repaired(fields, x_selection):
    """(reading, reason): the table of the module docstring.  The code has chosen and no checkpoint was repaired."""
    last = fl.ARMS['X']['checkpoints'][-1]
    if x_selection is None:
        return READ_UNDECIDABLE, '第三段没有跑完（还没有选 X 的 σ），X 的最后一步还可能过门'
    if x_selection.get('sigma_field') is not None and not arm_step(fields, 'X', last)['complete']:
        return READ_UNDECIDABLE, f"第三段没有跑完（X 的 {last} 步读到 {arm_step(fields, 'X', last)['n_read']}/{N}），它还可能过门"
    first = [arm_step(fields, arm, step) for arm in fl.FIRST_PHASE for step in fl.ARMS[arm]['checkpoints']]
    unread = [f"{r['arm']}@{r['step']}（{r['n_read']}/{N}）" for r in first if not r['complete']]
    if unread:
        return READ_UNDECIDABLE, f"没读到不等于没过线：{'、'.join(unread)} 没有 5/5 读到"
    candidates = [r for r in first if r['step'] == SELECT_STEP]
    every = list(first)
    if x_selection.get('sigma_field') is not None:
        x = arm_step(fields, 'X', last)
        candidates.append(x)
        every.append(x)
    highest = max(every, key=lambda r: r['seg35'])
    if highest['seg35'] <= BAR_SP:
        return READ_BELOW, f"全部 {len(every)} 个非 oracle 检查点 5/5 读到，seg35 都 ≤ {BAR_SP}；最高 {highest['arm']}@{highest['step']} {highest['seg35']:.4f}"
    above = [r for r in candidates if r['seg35'] > BAR_SP]
    if not above:
        return READ_UNDECIDABLE, f"过线的不是修复候选：最高 {highest['arm']}@{highest['step']} {highest['seg35']:.4f}，2400、{last} 步都 ≤ {BAR_SP}；交总负责人"
    if all(r['sdlogj_official'] > SDLOGJ_MAX for r in above):
        return READ_UNDECIDABLE, '过线的候选官方 SDlogJ 都超 .10：'+'、'.join(f"{r['arm']}@{r['step']} seg35 {r['seg35']:.4f}、SDlogJ {r['sdlogj_official']:.4f}" for r in above)
    return READ_UNDECIDABLE, '选择文件与记录不一致：有候选过修复门却没有修复记录：'+'、'.join(f"{r['arm']}@{r['step']}" for r in above if r['sdlogj_official'] <= SDLOGJ_MAX)


def read_all(fields, repairs, chosen, selected=True, x_selection=None):
    """The rule of the module docstring.  `chosen`: the checkpoints the code chose so far.  selected False: the code has not chosen yet (the reading after the first phase is only an interim record).  x_selection: analysis/x_selection.json, None before it is written."""
    rows = [repair_summary(repairs, c['arm'], c['step']) for c in chosen]
    good = [r for r in rows if r['satisfies_premise']]
    cells = [(arm, step) for arm, spec in fl.ARMS.items() if arm not in fl.ORACLE_ARMS for step in spec['checkpoints'] if (arm, step) != ('X', SELECT_STEP)]     # X at 2400 steps is the field of the arm it took sigma from: not one more cell
    n_cells = sum(arm_step(fields, arm, step)['n_read'] > 0 for arm, step in cells)
    n_candidates = sum(arm_step(fields, arm, step)['n_read'] > 0 for arm, step in cells if step in fl.REPAIR_STEPS)
    effect = dict(checkpoints_read=n_cells, candidate_checkpoints=n_candidates, repaired=len(rows), satisfying_the_premise=len(good), text=f'最好的臂是从 {n_cells} 个读到的检查点里取的：{n_candidates} 个是修复的候选（2400 或 4800 步），{len(rows)} 个做了修复，{len(good)} 个满足前提。前 5 对之前选配用过。')
    if not good:
        reading = READ_UNDECIDABLE
        if rows:
            reason = '；'.join(f"{r['arm']}@{r['step']}：{'、'.join(r['missed'])}" for r in rows)
        elif selected:
            reading, reason = nothing_repaired(fields, x_selection)
        else:
            reason = '中间读数：修复阶段还没有跑，代码还没有选检查点'
        return dict(reading=reading, reason=reason, best=None, premise=rows, selection_effect=effect)
    best = max(good, key=lambda r: r['seg35'])
    return dict(reading=reading_line(best['seg35']), reason=None, best=dict(arm=best['arm'], step=best['step'], seg35=best['seg35'], seg4=best['seg4'], sdlogj_official=best['sdlogj_official'], max_task_clock=best['max_task_clock']), premise=rows, selection_effect=effect)


# ---- controls (records) ---------------------------------------------------------------------------------------------------------------------------
def controls(root, x_arm=None, results=None):
    """F at 600 steps against ef_v1's F600; X at 2400 against the arm it took sigma from; O at 600 against mc0_v1's P2 fine (isolated, a record); the scale numbers.  True, False, or None when a side has no record."""
    results = fl.RESULTS if results is None else results
    out = dict(f600={}, x2400={}, o600={}, scale={}, not_bitwise=[])

    def hash_of(arm, step, pair):
        d = fl.run_dir(root, fl.Cell(arm, fl.FIRST, pair))
        return ((read_json(d/'run.json') or {}).get('checkpoints') or {}).get(str(step), {}).get('u_sha256')

    def compare(group, short, a, b):
        same = None if a is None or b is None else a == b
        out[group][short] = same
        if same is False:
            out['not_bitwise'].append(f'{group} {short}')
    for pair in core.PAIRS:
        short = core.short_name(pair)
        ef_record = read_json(fl.ef_dir(results, 'F600', pair)/'run.json') or {}
        compare('f600', short, hash_of('F', 600, pair), (ef_record.get('field') or {}).get('u_sha256'))
        if x_arm is not None:
            compare('x2400', short, hash_of('X', SELECT_STEP, pair), hash_of(x_arm, SELECT_STEP, pair))
        mc0_record = read_json(Path(results)/fl.MC0_ROOT/'runs/P2_fine'/short/'run.json') or {}
        compare('o600', short, hash_of('O', 600, pair), (mc0_record.get('field') or {}).get('u_sha256'))
        base = ((read_json(fl.run_dir(root, fl.Cell('F', fl.FIRST, pair))/'run.json') or {}).get('scale')) or {}
        row = dict(F_D_multiplier_equal_ef=None if base.get('D_multiplier') is None or (ef_record.get('scale') or {}).get('D_multiplier') is None else base['D_multiplier'] == ef_record['scale']['D_multiplier'])
        for arm in ('M', 'S', 'T', 'H', 'Hx', 'I', 'X'):
            scale = (read_json(fl.run_dir(root, fl.Cell(arm, fl.FIRST, pair))/'run.json') or {}).get('scale') or {}
            row[f'{arm}_g_norm_Q0_equal_F'] = None if scale.get('g_norm_Q0') is None or base.get('g_norm_Q0') is None else scale['g_norm_Q0'] == base['g_norm_Q0']
        out['scale'][short] = row
    return out


# ---- the description --------------------------------------------------------------------------------------------------------------------------
def describe(fields, repairs, x_arm):
    arms_of = [arm for arm in fl.ARMS if arm not in fl.ORACLE_ARMS]
    table = [arm_step(fields, arm, step) for arm in arms_of for step in fl.ARMS[arm]['checkpoints']]
    versus_sp = [dict(arm=r['arm'], step=r['step'], **{m: None if r[m] is None else r[m]-S_P[m] for m in S_P}) for r in table]
    steps = [paired(fields, arm, arm, b, reference_step=a) for arm in fl.SIGMA_ARMS+('X',) for a, b in zip(fl.ARMS[arm]['checkpoints'][:-1], fl.ARMS[arm]['checkpoints'][1:])]
    sigmas = [paired(fields, arm, 'F', step) for arm in ('M', 'S', 'T') for step in fl.FULL]
    against_f = [paired(fields, arm, 'F', step) for arm in ('H', 'Hx', 'I') for step in fl.TWO]
    ef_repairs = [dict(repair_summary(repairs, source, None, tier='ef'), arm=arm) for arm, source in fl.EF_REPAIRS.items()]
    return dict(table=table, versus_S_P=versus_sp, S_P=S_P, steps=steps, sigmas=sigmas, against_F=against_f, ef_repairs=ef_repairs, x_arm=x_arm, sigma_kernels=[dict(fl.field_kernel(fl.ARMS[arm]['sigma_field']), arm=arm) for arm in fl.SIGMA_ARMS])


def describe_oracle(fields):
    """The ceiling and the loss of passing from truth to predictions; never in a table of the reading."""
    rows = [arm_step(fields, 'O', step) for step in fl.ARMS['O']['checkpoints']]
    return dict(role=fl.ORACLE_ROLE, rows=rows, transfer_loss=[paired(fields, 'O', 'F', step) for step in fl.ARMS['O']['checkpoints']])


def describe_rest(fields, repairs, root):
    """The other 14 pairs: the lead's queues, only described.  Not one field of them enters the reading."""
    field_cells, _ = fl.rest_cells(root)
    out = []
    for arm, tier in sorted({(c.arm, c.tier) for c in field_cells}):
        steps = fl.steps_of(fl.Cell(arm, tier, fl.REST_PAIRS[0]))
        out.append(dict(arm=arm, tier=tier, step=steps, fields=arm_step(fields, arm, steps, tier=tier, pairs=fl.REST_PAIRS), repairs=repair_summary(repairs, arm, steps, tier=tier, pairs=fl.REST_PAIRS)))
    return out


def reading_code(root):
    """What wrote this reading: the sha256 of the two files that hold its rules, and the batch's commit (FL_tools_复核 important 4)."""
    return dict(decide_fl_sha256=core.sha_file(Path(__file__)), fl_core_sha256=core.sha_file(Path(fl.__file__)), batch_code_sha=(read_json(Path(root)/'batch.json') or {}).get('code_sha'))


def decide(root, stops=None):
    root = Path(root)
    fields, repairs = collect(root)
    bad = sorted(r['run_id'] for r in fields.values() if r['metrics'] is not None and r['arm'] not in fl.ORACLE_ARMS and r['oracle'] is not False)
    if bad:
        raise SystemExit(f'run.json is not oracle false for {bad}; nothing written')
    bad = sorted(r['run_id'] for r in fields.values() if r['metrics'] is not None and r['arm'] in fl.ORACLE_ARMS and r['oracle'] is not True)
    if bad:
        raise SystemExit(f'the oracle arm is not marked oracle true for {bad}; nothing written')
    x = fl.read_selection(root, 'x_selection.json') or {}
    x_arm = x.get('arm')
    result = read_all(fields, repairs, fl.chosen_repairs(root), selected=fl.read_selection(root, 'repair_selection_1.json') is not None, x_selection=fl.read_selection(root, 'x_selection.json'))
    decision = dict(schema=SCHEMA, code=reading_code(root), selections=fl.selection_hashes(root), is_conclusion=False, next_step='交总负责人与 BZ', oracle=False, note=NOTE, arms_stopped=stops or {}, rules=dict(bar_near=BAR_NEAR, bar_expand=BAR_EXPAND, bar_sp=BAR_SP, sdlogj_max=SDLOGJ_MAX, task_budget_seconds=fl.TASK_BUDGET_SECONDS, n=N, repair_eps=fl.REPAIR_EPS, repair_top=fl.REPAIR_TOP),
                    reading=result['reading'], reason=result['reason'], best=result['best'], premise=result['premise'], selection_effect=result['selection_effect'], x_selection=x or None, repair_selections=[fl.read_selection(root, n) for n in ('repair_selection_1.json', 'repair_selection_2.json')], description=describe(fields, repairs, x_arm), rest14=describe_rest(fields, repairs, root), oracle_description=describe_oracle(fields), controls=controls(root, x_arm))
    return decision, fields, repairs


# ---- the words --------------------------------------------------------------------------------------------------------------------------------
def _f(value, nd=4):
    if value is None:
        return '—'
    return f'{value:.{nd}f}' if isinstance(value, float) else str(value)


def _signed(value, nd=4):
    return '—' if value is None else f'{value:+.{nd}f}'


def _pair_cell(row, measure):
    d = row[measure]
    return f"{_signed(d['mean'])}（{d['positive']}/{d['n']}）"


def summary_text(decision):
    d = decision['description']
    best = decision['best']
    L = ['# FL 读数（由 decide_fl.py 写出）', '', f"**读数：{decision['reading']}**（不是结论，下一步交总负责人与 BZ）", '', NOTE, '']
    if best is not None:
        L += [f"最好的满足前提的检查点：{best['arm']} @ {best['step']} 步，修复后 seg35 / seg4 {_f(best['seg35'])} / {_f(best['seg4'])}，官方 SDlogJ {_f(best['sdlogj_official'])}，任务时钟最长 {_f(best['max_task_clock'], 1)} s。", '']
    if decision['reason']:
        L += [f"原因：{decision['reason']}", '']
    L += [decision['selection_effect']['text'], '']
    if decision['arms_stopped']:
        L += [f"停掉的臂：{json.dumps(decision['arms_stopped'], ensure_ascii=False)}", '']
    L += ['## 前提（修复后，证书都由评测单元从场算）', '', '| 检查点 | 满足前提 | α = 1 证书 | seg35 / seg4 修复前 → 修复后 | 只算过证书的对 seg35 / seg4（n） | 官方 / 脑内 SDlogJ | HD95 | 任务时钟 平均 / 最长 s | 修复 平均 / 最长 s | 预热平均 s | 运行与评测不一致 | 差在哪 |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in decision['premise']:
        L.append(f"| {r['arm']} @ {r['step']} | {r['satisfies_premise']} | {r['n_certified_at_1']}/{N} | {_f(r['seg35_before'])} / {_f(r['seg4_before'])} → {_f(r['seg35'])} / {_f(r['seg4'])} | {_f(r['seg35_certified_only'])} / {_f(r['seg4_certified_only'])}（{r['n_certified_only']}） | {_f(r['sdlogj_official'])} / {_f(r['sdlogj_brain'])} | {_f(r['hd95'], 3)} | {_f(r['mean_task_clock'], 1)} / {_f(r['max_task_clock'], 1)} | {_f(r['mean_seconds_repair'], 1)} / {_f(r['max_seconds_repair'], 1)} | {_f(r['mean_seconds_warmup'], 1)} | {r['run_and_evaluation_disagree'] or '无'} | {'；'.join(r['missed']) or '—'} |")
    L += ['', f"修复后对 S_P 终态（前 5 对 {S_P['seg35']} / {S_P['seg4']}，小六个 {S_P['small_six']}，皮层 {S_P['cortex']}）：", '', '| 检查点 | seg35 | seg4 | 小六个 | 皮层 |', '|---|---|---|---|---|']
    for r in decision['premise']:
        L.append(f"| {r['arm']} @ {r['step']} | " + ' | '.join('—' if r[m] is None else _signed(r[m]-S_P[m]) for m in ('seg35', 'seg4', 'small_six', 'cortex')) + ' |')
    L += ['', '## 只描述：检查点（α = 1，taper 后，未修复；不是通过证书的状态）', '', '| 臂 | σ | K | 步 | 读到 | seg35 / seg4 | HD95 | 官方 / 脑内 SDlogJ | 小六个 | 大六个 | 其余 23 个 | 皮层 | 白质 | 不可行单元 | 任务时钟 平均 / 最长 s | 通道与四层秒（原生层含检查点写盘） | D | 几何 | floor | 最小中心 J |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in d['table']:
        L.append(f"| {r['arm']} | {r['sigma_field']} | {r['k']} | {r['step']} | {r['n_read']}/{N} | {_f(r['seg35'])} / {_f(r['seg4'])} | {_f(r['hd95'], 3)} | {_f(r['sdlogj_official'])} / {_f(r['sdlogj_brain'])} | {_f(r['small_six'])} | {_f(r['large_six'])} | {_f(r['other_23'])} | {_f(r['cortex'])} | {_f(r['white_matter'])} | {'、'.join(str(x) for x in r['n_infeasible'])} | {_f(r['seconds_task'], 1)} / {_f(r['max_seconds_task'], 1)} | {_f(r['seconds_channels'], 1)}；{'、'.join(_f(s, 1) for s in r['seconds_levels'])} | {_f(r['data'], 1)} | {_f(r['geometry'], 1)} | {_f(r['floor'], 2)} | {_f(r['min_det'], 3)} |")
    L += ['', f"D、几何、floor 是该步之前最后一次求值（T88 的循环先求能量再走步），K 不同的臂 D 的尺度不同，只在同臂的检查点之间比。D 与 Dice 的关系就读这张表。", '',
          f"## 只描述：对 S_P 终态（前 5 对 {S_P['seg35']} / {S_P['seg4']}，小六个 {S_P['small_six']}，皮层 {S_P['cortex']}；runs/20261104-seed/result.md）", '', '| 臂 | 步 | seg35 | seg4 | 小六个 | 皮层 |', '|---|---|---|---|---|---|']
    for r in d['versus_S_P']:
        L.append(f"| {r['arm']} | {r['step']} | {_signed(r['seg35'])} | {_signed(r['seg4'])} | {_signed(r['small_six'])} | {_signed(r['cortex'])} |")
    L += ['', '## 只描述：σ_field 实际做了什么', '', SIGMA_NOTE, '', '| 臂 | σ_field | 核半径 | 中心权 | ±1 权 | 每步每轴方差（连续值 σ²） | 棋盘模态每步每轴增益 | 600 步后 | 2400 步后 |', '|---|---|---|---|---|---|---|---|---|']
    for r in d['sigma_kernels']:
        L.append(f"| {r['arm']} | {r['sigma_field']} | {r['radius']} | {r['centre']:.4f} | {r['first']:.2e} | {r['variance_per_step']:.5f}（{r['continuous_variance']:.4f}） | {r['checkerboard_gain']:.5f} | {r['checkerboard_after_600']:.2e} | {r['checkerboard_after_2400']:.2e} |")
    for title, rows in (('步数曲线（后一个检查点 − 前一个，同一臂，配对；X 的 2400 步是起点）', d['steps']), ('σ 曲线（对 F，同一步，配对；不按 σ 插值）', d['sigmas']), ('H、Hx、I 相对 F（同一步，配对）', d['against_F'])):
        L += ['', f'## 只描述：{title}', '', '| 臂 | 对照 | 步 | seg35 | seg4 | 小六个 | 大六个 | 其余 23 个 | 皮层 | 白质 | 官方 SDlogJ |', '|---|---|---|---|---|---|---|---|---|---|---|']
        for r in rows:
            L.append(f"| {r['arm']} | {r['reference']}@{r['reference_step']} | {r['step']} | {_pair_cell(r, 'seg35')} | {_pair_cell(r, 'seg4')} | {_pair_cell(r, 'small_six')} | {_pair_cell(r, 'large_six')} | {_pair_cell(r, 'other_23')} | {_pair_cell(r, 'cortex')} | {_pair_cell(r, 'white_matter')} | {_pair_cell(r, 'sdlogj_official')} |")
    L += ['', '## 只描述：修复（ε = .05）：秒数对不可行单元数', '', '| 来源 | 对 | 状态 | 证书 α = 1 | 不可行单元 前 → 后 | 修复秒 | 预热秒 | 任务时钟 | 轮数 | W 点 | 掩膜节点 | 改动节点 | 最大位移变化 mm | 最小认证下界 | seg35 前 → 后 | seg4 前 → 后 | 结束原因 |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    labelled = [(f"{r['arm']}@{r['step']}", r) for r in decision['premise']]+[(r['arm'], r) for r in d['ef_repairs']]
    for name, r in labelled:
        for row in r['per_pair']:
            L.append(f"| {name} | {row['pair']} | {row['status']} | {row.get('certified_at_1')} | {row.get('n_infeasible_before')} → {row.get('n_infeasible_after')} | {_f(row.get('seconds_repair'), 1)} | {_f(row.get('seconds_warmup'), 1)} | {_f(row.get('task_clock'), 1)} | {row.get('rounds')} | {row.get('n_extra_points')} | {row.get('n_free')} | {row.get('n_changed_nodes')} | {_f(row.get('max_change_mm'), 3)} | {_f(row.get('min_lower_bound_after'), 6)} | {_f(row.get('seg35_before'))} → {_f(row.get('seg35_after'))} | {_f(row.get('seg4_before'))} → {_f(row.get('seg4_after'))} | {row.get('reason')} |")
    L += ['', '## 只描述：RF150、RF300（修复 ef_v1 的 F150、F300 场，成本标定，不进读数）', '', '| 臂 | α = 1 证书 | seg35 / seg4 修复前 → 修复后 | 官方 SDlogJ | 修复 平均 / 最长 s | 任务时钟 平均 / 最长 s |', '|---|---|---|---|---|---|']
    for r in d['ef_repairs']:
        L.append(f"| {r['arm']} | {r['n_certified_at_1']}/{N} | {_f(r['seg35_before'])} / {_f(r['seg4_before'])} → {_f(r['seg35'])} / {_f(r['seg4'])} | {_f(r['sdlogj_official'])} | {_f(r['mean_seconds_repair'], 1)} / {_f(r['max_seconds_repair'], 1)} | {_f(r['mean_task_clock'], 1)} / {_f(r['max_task_clock'], 1)} |")
    o = decision['oracle_description']
    L += ['', '## 只描述：O 的上限与传递损失（oracle，隔离，不进结果表）', '', ORACLE_NOTE, '', '| 步 | 读到 | seg35 / seg4 | 官方 SDlogJ | 小六个 | 皮层 | 不可行单元 | O − F：seg35 | seg4 |', '|---|---|---|---|---|---|---|---|---|']
    for r, loss in zip(o['rows'], o['transfer_loss']):
        L.append(f"| {r['step']} | {r['n_read']}/{N} | {_f(r['seg35'])} / {_f(r['seg4'])} | {_f(r['sdlogj_official'])} | {_f(r['small_six'])} | {_f(r['cortex'])} | {'、'.join(str(x) for x in r['n_infeasible'])} | {_pair_cell(loss, 'seg35')} | {_pair_cell(loss, 'seg4')} |")
    if decision['rest14']:
        L += ['', '## 只描述：其余 14 对（总负责人单独启动；没有参与选配；不改上面的读数）', '', '| 臂 | 步 | 场 读到 | α = 1 seg35 / seg4 | 修复 读到 | α = 1 证书 | 修复后 seg35 / seg4 | 官方 SDlogJ | 任务时钟 平均 / 最长 s |', '|---|---|---|---|---|---|---|---|---|']
        for r in decision['rest14']:
            a, b = r['fields'], r['repairs']
            L.append(f"| {r['arm']} | {r['step']} | {a['n_read']}/14 | {_f(a['seg35'])} / {_f(a['seg4'])} | {b['n_read']}/14 | {b['n_certified_at_1']}/14 | {_f(b['seg35'])} / {_f(b['seg4'])} | {_f(b['sdlogj_official'])} | {_f(b['mean_task_clock'], 1)} / {_f(b['max_task_clock'], 1)} |")
    c = decision['controls']
    L += ['', '## 控制（只记录）', '', f"读数的规则：decide_fl.py sha256 {decision['code']['decide_fl_sha256']}，fl_core.py sha256 {decision['code']['fl_core_sha256']}，批次提交 {decision['code']['batch_code_sha']}；选择文件 {decision['selections']}", '', 'F 的 600 步是第一次读检查点，它证明 T88 路径不依赖总步数；"读检查点不改变后面的路径"只有 X 的 2400 步能核（σ 臂读过 600、1200，X 没读），只记录，`--through repair` 时没有。', '', f"F 的 600 步对 ef_v1 的 F600（u 逐位）：{c['f600']}", '', f"X 的 2400 步对它取 σ 的臂（{d['x_arm']}）：{c['x2400'] or '没有 X'}", '', f"O 的 600 步对 mc0_v1 的 P2 细档：{c['o600']}", '', f"不逐位相同的：{c['not_bitwise'] or '无'}", '', f"尺度数字不相等的：{[f'{short} {k}' for short, row in c['scale'].items() for k, v in row.items() if v is False] or '无'}（没有读到的不算不等）", '',
          '## 它证伪不了什么', '', '不是 F_B，没有经认证的下降，只有终态过证书；开发集单次；n = 5。进方法先改合同，由 BZ 定。']
    return '\n'.join(L)+'\n'


def write_long(path, fields, repairs):
    """Non-oracle only: the oracle arm enters no result table."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(('run_id', 'experiment', 'arm', 'tier', 'step', 'pair', 'status', 'oracle', 'field', 'measure', 'value'))
        for (arm, tier, step, pair), row in sorted(fields.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
            if row['metrics'] is None or arm in fl.ORACLE_ARMS:
                continue
            for m in MEASURES:
                w.writerow((row['run_id'], 'field', arm, tier, step, pair, row['status'], 'false', 'alpha1_certified' if row['certified'] else 'alpha1_uncertified', m, row['metrics'][m]))
        for (arm, tier, step, pair), row in sorted(repairs.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0, kv[0][3])):
            if not row['metrics']:
                continue
            for m in MEASURES:
                if row['before'].get(m) is not None:      # ef_v1's metrics have no HD95
                    w.writerow((row['run_id'], 'repair', arm, tier, step, pair, row['status'], 'false', 'before_alpha1', m, row['before'][m]))
                w.writerow((row['run_id'], 'repair', arm, tier, step, pair, row['status'], 'false', 'repaired' if row['certified_at_1'] else 'attempt_uncertified', m, row['after'][m]))


def write_analysis(root, stops=None):
    """analysis/: an earlier version moved to analysis/previous_<UTC>/, never overwritten."""
    root = Path(root)
    decision, fields, repairs = decide(root, stops)
    out = root/'analysis'
    set_aside(out, ('decision_fl.json', 'long.csv', 'summary_fl.md'))
    (out/'decision_fl.json').write_text(json.dumps(decision, indent=1, ensure_ascii=False)+'\n')
    (out/'summary_fl.md').write_text(summary_text(decision), encoding='utf-8')
    write_long(out/'long.csv', fields, repairs)
    return decision
