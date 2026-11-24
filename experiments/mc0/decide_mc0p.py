"""MC0P's reading (runs/20261104-mc0p/plan.md, by code; a label for BZ, not a conclusion): analysis/decision_mc0p.json, long.csv, summary_mc0p.md.

Delta = the best deciding K > 1 arm (Q1, Q2, Q2s) - Q0, same tier, mean official SDlogJ <= .10, n = 5.  The rules are MC0's, revisions 1 and 2, and the same code (decide_mc0.tier_reading,
scope_text); only the arms, the baseline, the two tiers and the words of the reading differ:
  some tier    the best arm: Delta(seg35) >= +.01 and 5/5 positive                                  the channels show headroom on predicted labels
  both tiers   comparable, every deciding arm read, each: Delta(seg35) <= +.005 or fewer than 4 of 5 positive     no headroom on predicted labels
  no tier comparable                                                                                undecidable
  otherwise                                                                                         middle band (reading_basis: effect or not_read)
Only described, never in the reading: the realization ratios Delta_Q / Delta_P against MC0 (Q1 long / P1 long, Q2 long / P2 long, Q2 fine / P2 fine), pair by pair; Q2s - Q2; Q3, Q3x3, Q3h.

Plan revision 2, the retry: write_merged reads the first batch and the retry (mc0p_v1b), each cell from the first batch if done there, otherwise from the retry, and applies the same rules to the merged cells:
analysis/{decision_mc0p_merged.json, summary_mc0p_merged.md, long.csv} under the retry's root, with the first batch's own reading quoted first.  The first batch is only read.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics

import numpy as np

import decide_mc0 as base
import mc0_core as core
import mc0p_core as mc

SCHEMA = 'bayesreg.mc0p.decision.v1'
MERGED_SCHEMA = 'bayesreg.mc0p.decision.merged.v1'
N, MEASURES = base.N, base.MEASURES
READ_NONE = '预测标签上看不到通道余量'
READ_SEE = '预测标签上看得到通道余量'
READ_MIDDLE, READ_MIDDLE_NOT_READ, READ_UNDECIDABLE = base.READ_MIDDLE, base.READ_MIDDLE_NOT_READ, base.READ_UNDECIDABLE
A5P_FINAL = dict(seg35=.8290, seg4=.8894)       # A5P on the same 5 pairs, final state (status block of the project status log, 11-04); Q0 is NOT that state: no native level here
REALIZATION = (('Q1', 'long', 'P1'), ('Q2', 'long', 'P2'), ('Q2', 'fine', 'P2'))
DESCRIBED = tuple(arm for arm, spec in mc.ARMS.items() if not spec['decides'] and arm != mc.BASELINE)
SCALE_KEYS = ('images_equal_a5p_inputs', 'D_factor')
CONDITIONS = ('这一批有两种运行条件：v1 与训练同跑，v1b 单独。T88 的迭代数固定，不是计时运行，运行条件本身不进目标函数；control_bitwise 是这件事的证据（同一格、同一份输入，在两种条件下比较 field.npz 的 u）。'
              '补跑的原因是 v1 的 cuda_oom，〔推断〕驱动层失败，不是结论。规则、阈值、Δ 口径都不变，只是格子多了；同一格两边都有时用 v1 的。')
NOTE = ('开发集前 5 对，单次观测。通道来自 unet35_v1_ens 的预测，oracle: false，真值标签只进评测；但这些场不是方法，不进任何结果表。场没有过原生层。读数不是结论，下一步都交 BZ。'
        '阴性不是逻辑证明：Adam 加场平滑不是原生层，A0 的原生层把皮层又提了约 +.083。T88 没有原生层：阳性不代表过了原生层还在，也不代表可认证。'
        'Q0 与 A5P 的终态（.8290 / .8894）不是同一个东西：图相同，优化器不同，Q0 没有原生层。'
        '尺度只取一个值，只在原生层 u = 0 拉平：K > 1 的 SDlogJ 明显低于 Q0 时比较的是不同工作点，阴性只针对这个尺度；粗层没有拉平（见 r_ℓ）。')


def selection_effect():
    """The cells "看得到" can fire on, under the plan as it stands."""
    cells = sum(len(mc.deciding_arms(t)) for t in mc.TIER_NAMES)
    most = max(len(mc.deciding_arms(t)) for t in mc.TIER_NAMES)
    return dict(tiers=len(mc.TIER_NAMES), most_arms_in_a_tier=most, arm_tier_cells=cells, per_cell_5_of_5_under_zero=1/2**N, rough_upper_bound=cells/2**N,
                text=(f'带选择效应：从 {len(mc.TIER_NAMES)} 档、每档最多 {most} 个臂（共 {cells} 个臂×档）里取最好的，看得到只需一档。每格在零假设下 5/5 为正的概率是 1/{2**N}，'
                      f'粗略上界约 {cells/2**N:.3f}；各格相关，另有 Δ ≥ +.01 的门槛，实际低于这个数〔推断〕。'))


def paired(rows, arm, tier, baseline, measure, pairs):
    """{pair: arm - baseline} over the pairs where both have an evaluated value."""
    out = {}
    for p in pairs:
        a, b = rows.get((arm, tier, p)), rows.get((baseline, tier, p))
        if a is not None and b is not None and a.get('metrics') is not None and b.get('metrics') is not None and a['metrics'].get(measure) is not None and b['metrics'].get(measure) is not None:
            out[p] = a['metrics'][measure]-b['metrics'][measure]
    return out


def ratio(numerator, denominator):
    """Delta_Q / Delta_P; None when Delta_P is not positive: there was nothing to realize."""
    return numerator/denominator if denominator is not None and denominator > 0 else None


def realization(rows, mc0_rows, pairs=core.PAIRS):
    """Delta_Q / Delta_P, pair by pair and as the ratio of the means over the pairs that have both.  Delta_Q against Q0, Delta_P against P0 of MC0: each against its own baseline.  Description only."""
    out = []
    for q_arm, tier, p_arm in REALIZATION:
        row = dict(q_arm=q_arm, p_arm=p_arm, tier=tier)
        for measure in ('seg35', 'seg4'):
            dq, dp = paired(rows, q_arm, tier, mc.BASELINE, measure, pairs), paired(mc0_rows, p_arm, tier, 'P0', measure, pairs)
            both = [p for p in pairs if p in dq and p in dp]
            mq, mp = (statistics.fmean(d[p] for p in both) if both else None for d in (dq, dp))
            row[measure] = dict(n=len(both), mean_delta_Q=mq, mean_delta_P=mp, ratio_of_means=ratio(mq, mp) if both else None,
                                per_pair={core.short_name(p): dict(delta_Q=dq[p], delta_P=dp[p], ratio=ratio(dq[p], dp[p])) for p in both})
        out.append(row)
    return out


def soft_minus_hard(rows, pairs=core.PAIRS):
    """Q2s - Q2, long tier, every measure, pair by pair.  Description only."""
    out = {}
    for measure in MEASURES:
        d = paired(rows, 'Q2s', 'long', 'Q2', measure, pairs)
        out[measure] = dict(n=len(d), mean=statistics.fmean(d.values()) if d else None, positive=sum(x > 0 for x in d.values()), per_pair={core.short_name(p): x for p, x in d.items()})
    return out


def decide(rows, pairs=core.PAIRS, stops=None, mc0_rows=None):
    tiers = {t: base.tier_reading(rows, t, pairs, baseline=mc.BASELINE, arms_of=mc.deciding_arms) for t in mc.TIER_NAMES}
    comparable = [t for t, r in tiers.items() if r['comparable']]
    gaps = len(comparable) < len(tiers) or any(r['excluded'] for r in tiers.values())
    if any(r['headroom'] for r in tiers.values()):
        reading = READ_SEE
    elif len(comparable) == len(tiers) and all(r['no_headroom'] for r in tiers.values()):
        reading = READ_NONE
    elif not comparable:
        reading = READ_UNDECIDABLE
    else:
        reading = READ_MIDDLE_NOT_READ if gaps else READ_MIDDLE
    return dict(schema=SCHEMA, reading=reading, reading_basis='not_read' if reading in (READ_MIDDLE_NOT_READ, READ_UNDECIDABLE) else 'effect', is_conclusion=False, next_step='交 BZ',
                oracle=any(r.get('oracle') is not False for r in rows.values() if r.get('metrics') is not None),      # from each run.json
                tiers=tiers, comparable_tiers=comparable, arms_stopped=stops or {},
                tiers_not_comparable={t: dict(reason=r['reason'], k_arms_not_compared=mc.deciding_arms(t)) for t, r in tiers.items() if not r['comparable']},
                arms_incomplete=sorted({a for r in tiers.values() for a, why in r['excluded'].items() if why == base.INCOMPLETE}), scope_note=base.scope_text(tiers, reading, mc.deciding_arms),
                inputs_check=dict(q0_images_equal_a5p_inputs=sorted({str(r.get('images_equal_a5p_inputs')) for (arm, _, _), r in rows.items() if arm == mc.BASELINE}), a5p_final_not_this_state=A5P_FINAL),
                selection_effect=selection_effect(), rules=dict(sdlogj_max=base.SDLOGJ_MAX, no_headroom_bar=base.NO_HEADROOM_BAR, headroom_bar=base.HEADROOM_BAR, n=N, min_positive=base.MIN_POSITIVE), note=NOTE,
                tables=base.arm_tables(rows, pairs, baseline=mc.BASELINE, arms=mc.ARMS, tiers=mc.TIER_NAMES),
                described=dict(role='只描述，不进判据', realization=realization(rows, mc0_rows or {}, pairs), realization_note=None if mc0_rows else '没有读到 MC0 的结果，兑现比空缺', q2s_minus_q2=soft_minus_hard(rows, pairs), arms=list(DESCRIBED)))


def collect(root):
    return base.collect(root, mc.plan_runs(), SCALE_KEYS, ('oracle', 'retry_of'))


def root_problem(root):
    """Why this is not MC0P's root, or None (plan revision 1): the name begins with mc0p, and its batch.json, if any, is MC0P's.  Pointing `decide` or `evaluate` at mc0_v1 by mistake moves nothing there."""
    root = Path(root)
    problem = mc.v1_write_problem(root)                 # plan revision 2: the first batch is read only
    if problem:
        return problem
    if not root.name.startswith('mc0p'):
        return f"{root}: not MC0P's root (its name must begin with mc0p); nothing read, nothing moved"
    batch = base.read_json(root/'batch.json')
    if batch is not None and batch.get('schema') != mc.BATCH_SCHEMA:
        return f"{root}: not MC0P's root: batch.json has schema {batch.get('schema')}, not {mc.BATCH_SCHEMA}; nothing moved"
    return None


def _fmt(v, nd=4):
    return base._fmt(v, nd)


def summary_text(decision, title='MC0P 读数'):
    against = f'against_{mc.BASELINE}'
    L = [f'# {title}（由 decide_mc0p.py 写出）', '', f"**读数：{decision['reading']}**（reading_basis：{decision['reading_basis']}；不是结论，下一步交 BZ）", '', NOTE, '', decision['selection_effect']['text'], '',
         f"Q0 的图与 A5P 输入逐位相同：{decision['inputs_check']['q0_images_equal_a5p_inputs']}", '']
    if decision['arms_stopped']:
        L += [f"停掉的臂：{json.dumps(decision['arms_stopped'], ensure_ascii=False)}", '']
    if decision['scope_note']:
        L += [decision['scope_note'], '']
    for tier, reading in decision['tiers'].items():
        L += [f'## 档 {tier}', '', (f"最好的进判定臂 {reading['best_arm']}：Δ seg35 {_fmt(reading['delta_seg35'])}（{reading['positive_seg35']}/{N} 为正），"
                                  f"Δ seg4 {_fmt(reading['delta_seg4'])}（{reading['positive_seg4']}/{N}）") if reading['comparable'] else f"不可比：{reading['reason']}", '']
        if reading['comparable']:
            point = reading['working_point']
            L += ['工作点（最好的臂 / Q0 / 差）：' + '；'.join(f"{name} {_fmt(point[key]['best'])} / {_fmt(point[key][mc.BASELINE])} / {_fmt(point[key]['delta'])}" for key, name in (('sdlogj_official', '官方 SDlogJ'), ('sdlogj_brain', '脑内 SDlogJ'))), '']
        L += [f"predicate_on_read_arms（只看读到的臂时，每个臂都没有余量）：{_fmt(reading['predicate_on_read_arms'])}；no_headroom（这一档算没有余量）：{reading['no_headroom']}", '']
        if reading['excluded']:
            L += [f"没读到（这一档不能算\"看不到\"）：{json.dumps(reading['excluded'], ensure_ascii=False)}", '']
        if reading['other_arms_at_headroom_bar']:
            L += [base.other_arms_text(tier, reading), '']
        L += ['| 臂 | 进判定 | D 乘数 | 评测/可读 | seg35 | seg4 | 官方 SDlogJ | 脑内 SDlogJ | 小六个 | 大六个 | 其余 23 个 | 皮层 | 白质 | Δ seg35 对 Q0 | Δ seg4 对 Q0 | Δ 官方 SDlogJ 对 Q0 | Δ 脑内 SDlogJ 对 Q0 | r_ℓ（层 8/4/2/1） | r_ℓ（G_grad 之后） | α | α = 1 不可行单元（status） | 接受 α 下 seg35 / seg4（不是方法） | 显存峰值 GiB |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for arm, row in decision['tables'][tier].items():
            d35, d4, dsd, dbrain = (row[against][m] for m in ('seg35', 'seg4', 'sdlogj_official', 'sdlogj_brain'))
            L.append(f"| {arm} | {'基线' if arm == mc.BASELINE else '是' if mc.ARMS[arm]['decides'] else '否，只描述'} | ×{mc.ARMS[arm]['D_factor']:g} | {row['n_evaluated']}/{row['n_read']} | " + ' | '.join(_fmt(row['means'][m]) for m in MEASURES)
                     + f" | {_fmt(d35['mean'])} ({d35['positive']}/{d35['n']}) | {_fmt(d4['mean'])} ({d4['positive']}/{d4['n']}) | {_fmt(dsd['mean'])} | {_fmt(dbrain['mean'])} | {base._fmt_list(row['level_ratio_raw'])} | {base._fmt_list(row['level_ratio_smoothed'])} | {row['alpha']} | {base._fmt_at_1(row)} | "
                     f"{_fmt(row['at_accepted_alpha']['seg35'])} / {_fmt(row['at_accepted_alpha']['seg4'])} | {_fmt(row['device_peak_reserved_gib'], 2)} |")
        L.append('')
    described = decision['described']
    L += ['## 只描述，不进判据', '', '兑现比 Δ_Q / Δ_P（Δ_Q 对 Q0，Δ_P 对 MC0 的 P0，各对自己的基线；Δ_P 不为正时空缺）：', '',
          '| Q 臂 / P 臂 | 档 | 指标 | n | Δ_Q 均值 | Δ_P 均值 | 均值之比 | 逐对比值（0438-0439 … 0442-0443） |', '|---|---|---|---|---|---|---|---|']
    for row in described['realization']:
        for measure in ('seg35', 'seg4'):
            r = row[measure]
            L.append(f"| {row['q_arm']} / {row['p_arm']} | {row['tier']} | {measure} | {r['n']} | {_fmt(r['mean_delta_Q'])} | {_fmt(r['mean_delta_P'])} | {_fmt(r['ratio_of_means'], 3)} | "
                     + ' / '.join(_fmt(x['ratio'], 3) for x in r['per_pair'].values()) + ' |')
    L.append('')
    if described['realization_note']:
        L += [described['realization_note'], '']
    L += ['Q2s − Q2（长档）：', '', '| 指标 | n | 均值 | 为正 |', '|---|---|---|---|']
    L += [f"| {m} | {r['n']} | {_fmt(r['mean'])} | {r['positive']} |" for m, r in described['q2s_minus_q2'].items()]
    L += ['', f"{'、'.join(described['arms'])} 的各组 Dice 与工作点见档 long 的表，只描述。", '']
    return '\n'.join(L)+'\n'


def set_aside(out, names):
    """An earlier version is moved to out/previous_<UTC>/, never overwritten."""
    if any((out/n).exists() for n in names):
        aside = out/f"previous_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        aside.mkdir(parents=True)
        for n in names:
            if (out/n).exists():
                shutil.move(str(out/n), str(aside/n))
    out.mkdir(parents=True, exist_ok=True)


def write_long(path, rows, *, with_source=False):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(('run_id', 'arm', 'tier', 'pair', 'status', 'K', 'oracle', 'field', 'D_factor', 'D_multiplier', 'accepted_alpha', 'n_infeasible_at_1', 'measure', 'value')+(('source_root',) if with_source else ()))
        for (arm, tier, pair), row in sorted(rows.items()):
            for m in MEASURES if row['metrics'] else ():
                w.writerow((row['run_id'], arm, tier, pair, row['status'], mc.ARMS[arm]['k'], str(row['oracle']).lower(), 'uncertified_alpha1', row['D_factor'], row['D_multiplier'], row['accepted_alpha'], row['n_infeasible_at_1'], m, row['metrics'][m])
                           +((row['source_root'],) if with_source else ()))


def not_oracle_false(rows):
    return sorted(row['run_id'] for row in rows.values() if row['metrics'] is not None and row['oracle'] is not False)


def mc0_rows_of(mc0_root):
    mc0_root = mc.MC0_ROOT if mc0_root is None else Path(mc0_root)
    return base.collect(mc0_root) if mc0_root.is_dir() else None


def write_analysis(root, stops=None, mc0_root=None):
    """Writes analysis/: an earlier version is moved to analysis/previous_<UTC>/, never overwritten.  MC0's results are only read.  STOP, nothing moved: not MC0P's root; a run with metrics whose run.json is not oracle false."""
    root = Path(root)
    problem = root_problem(root)
    if problem:
        raise SystemExit(problem)
    rows = collect(root)
    bad = not_oracle_false(rows)
    if bad:
        raise SystemExit(f'run.json is not oracle false for {bad}; nothing written')
    out = root/'analysis'
    set_aside(out, ('decision_mc0p.json', 'long.csv', 'summary_mc0p.md'))
    decision = decide(rows, stops=stops, mc0_rows=mc0_rows_of(mc0_root))
    (out/'decision_mc0p.json').write_text(json.dumps(decision, indent=1, ensure_ascii=False)+'\n')
    (out/'summary_mc0p.md').write_text(summary_text(decision), encoding='utf-8')
    write_long(out/'long.csv', rows)
    return decision


# ---- the first batch and the retry (plan revision 2) --------------------------------------------------------------------------------------
def merge_rows(v1_rows, retry_rows, v1_root, root):
    """Each cell once, with its source_root: done in the first batch, from there (so the control is the first batch's); otherwise the retry's record if there is one; otherwise the first batch's failed record, which counts as not read."""
    out = {}
    for cell in mc.plan_runs():
        key = (cell.arm, cell.tier, cell.pair)
        first, again = v1_rows.get(key), retry_rows.get(key)
        if first is not None and first['status'] == 'done':
            out[key] = dict(first, source_root=str(v1_root))
        elif again is not None:
            out[key] = dict(again, source_root=str(root))
        elif first is not None:
            out[key] = dict(first, source_root=str(v1_root))
    return out


def field_u_sha256(path):
    with np.load(path) as z:
        return hashlib.sha256(np.asarray(z['u']).tobytes()).hexdigest()


def control_record(v1_root, root):
    """The u of the control's field.npz in both batches.  Different is recorded as it is: no failure, nothing in the reading changes."""
    a, b = (core.run_dir(r, mc.CONTROL)/'field.npz' for r in (v1_root, root))
    done = base.read_json(core.run_dir(root, mc.CONTROL)/'queue_done.json') or {}
    record = dict(cell=mc.CONTROL.run_id, control_bitwise=None, u_sha256_v1=None, u_sha256_v1b=None, control_stop=done.get('control_stop'), role='只记录：不算失败，不改读数，它的失败也不停臂')
    if not a.is_file() or not b.is_file():
        return dict(record, reason=f"没有比较：缺 {[str(p) for p in (a, b) if not p.is_file()]}")
    va, vb = field_u_sha256(a), field_u_sha256(b)
    return dict(record, control_bitwise=va == vb, u_sha256_v1=va, u_sha256_v1b=vb)


def first_batch_reading(v1_root):
    """The first batch's own reading, quoted from its decision_mc0p.json; only read."""
    path = Path(v1_root)/'analysis/decision_mc0p.json'
    if not path.is_file():
        raise SystemExit(f'{path}: the first batch has no reading to quote')
    d = json.loads(path.read_text())
    keys = ('comparable', 'best_arm', 'delta_seg35', 'delta_seg4', 'positive_seg35', 'positive_seg4', 'no_headroom', 'headroom', 'excluded', 'reason')
    return dict(file=str(path), sha256=core.sha_file(path), reading=d['reading'], reading_basis=d['reading_basis'], scope_note=d['scope_note'], tiers={t: {k: r[k] for k in keys} for t, r in d['tiers'].items()})


def control_metrics(v1_rows, retry_rows):
    key = (mc.CONTROL.arm, mc.CONTROL.tier, mc.CONTROL.pair)
    def of(row):
        return None if row is None or row.get('metrics') is None else {m: row['metrics'][m] for m in ('seg35', 'seg4', 'sdlogj_official')}
    return dict(v1=of(v1_rows.get(key)), v1b=of(retry_rows.get(key)))


def decide_merged(v1_rows, retry_rows, v1_root, root, *, control, quoted, stops=None, mc0_rows=None):
    rows = merge_rows(v1_rows, retry_rows, v1_root, root)
    return rows, dict(decide(rows, stops=stops, mc0_rows=mc0_rows), schema=MERGED_SCHEMA, conditions=CONDITIONS, first_batch_reading=quoted, control=dict(control, metrics=control_metrics(v1_rows, retry_rows)),
                      merged=dict(first_batch=str(v1_root), retry=str(root), rule='这一格在 v1 是 done 就用 v1（所以控制格用 v1）；否则用 v1b；v1b 也没有就是 v1 的失败记录，算没读到', cells={row['run_id']: row['source_root'] for row in rows.values()}, statuses={row['run_id']: row['status'] for row in rows.values()}))


def _signed(x):
    return '—' if x is None else f'{x:+.4f}'.replace('0.', '.', 1)


def merged_summary_text(decision):
    quoted, control, merged = decision['first_batch_reading'], decision['control'], decision['merged']
    L = [f"**v1 自己的读数（原样，{quoted['file']}）：{quoted['reading']}**（reading_basis：{quoted['reading_basis']}）", '']
    L += [f"- {t}：最好的臂 {r['best_arm']}，Δ seg35 / seg4 {_signed(r['delta_seg35'])} / {_signed(r['delta_seg4'])}（{r['positive_seg35']}/{N}、{r['positive_seg4']}/{N} 为正）" if r['comparable'] else f"- {t}：不可比：{r['reason']}" for t, r in quoted['tiers'].items()]
    L += ['', f"v1 的 scope_note：{quoted['scope_note']}", '', CONDITIONS, '', f"**control_bitwise: {'—' if control['control_bitwise'] is None else str(control['control_bitwise']).lower()}**（{control['cell']}，比较 field.npz 的 u 的 sha256）。"]
    if control['control_bitwise'] is None:
        L[-1] += control.get('reason', '')+(f"（控制格没有跑完，只记录，不停臂：{control['control_stop']}）" if control['control_stop'] else '')
    elif control['control_bitwise']:
        L[-1] += '两种运行条件下逐位相同。'
    else:
        L[-1] += f"两边不同（v1 {control['u_sha256_v1'][:12]}…，v1b {control['u_sha256_v1b'][:12]}…）。可能是设备不可重复，或者与训练同跑有关；不算失败，不改读数。"
    L += [f"控制格的指标（seg35、seg4、官方 SDlogJ；只描述，读数用 v1 的）：v1 {control['metrics']['v1']}；v1b {control['metrics']['v1b']}", '']
    from_retry = sorted(run_id for run_id, source in merged['cells'].items() if source == merged['retry'])
    L += [f"合并：{len(merged['cells'])} 格有记录，其中 {len(from_retry)} 格来自 v1b：{'、'.join(from_retry) or '无'}。{merged['rule']}。", '', '合并之后：', '']
    return '\n'.join(L)+'\n'+summary_text(decision, title='MC0P 合并读数')


def write_merged(root, v1_root, stops=None, mc0_root=None):
    """`decide --root mc0p_v1b --merge mc0p_v1`: analysis/{decision_mc0p_merged.json, summary_mc0p_merged.md, long.csv} under the retry's root; an earlier version moved aside there.  The first batch is only read.
    STOP, nothing moved: not the approved roots; a record with metrics that is not oracle false; a record of the retry without retry_of."""
    root, v1_root = Path(root), Path(v1_root)
    problem = root_problem(root) or mc.retry_problem(root, v1_root)
    if problem:
        raise SystemExit(problem)
    v1_rows, retry_rows = collect(v1_root), collect(root)
    quoted = first_batch_reading(v1_root)
    control = control_record(v1_root, root)
    rows, decision = decide_merged(v1_rows, retry_rows, v1_root, root, control=control, quoted=quoted, stops=stops, mc0_rows=mc0_rows_of(mc0_root))
    bad = not_oracle_false(rows)
    if bad:
        raise SystemExit(f'run.json is not oracle false for {bad}; nothing written')
    # only cells with a run.json: a run that failed without a record has a row from queue_done.json alone, says nothing, and counts as not read (plan revision 3)
    stray = sorted(row['run_id'] for key, row in retry_rows.items() if row.get('retry_of') != str(mc.RETRY_OF_ROOT) and (core.run_dir(root, mc.Cell(*key))/'run.json').is_file())
    if stray:
        raise SystemExit(f'{root}: these records do not say retry_of {mc.RETRY_OF_ROOT}: {stray}; nothing written')
    out = root/'analysis'
    set_aside(out, ('decision_mc0p_merged.json', 'long.csv', 'summary_mc0p_merged.md'))
    (out/'decision_mc0p_merged.json').write_text(json.dumps(decision, indent=1, ensure_ascii=False)+'\n')
    (out/'summary_mc0p_merged.md').write_text(merged_summary_text(decision), encoding='utf-8')
    write_long(out/'long.csv', rows, with_source=True)
    return decision
