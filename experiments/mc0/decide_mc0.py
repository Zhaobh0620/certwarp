"""MC0's reading (runs/20261103-mc0/plan.md, by code; a label for BZ, not a conclusion): analysis/decision_mc0.json, long.csv, summary_mc0.md.

Delta = the best K > 1 arm - P0, same tier, mean official SDlogJ <= .10, n = 5.  Per tier the arms with 5 finished, evaluated runs and that SDlogJ are
eligible, the best has the largest mean seg35 (ties: the smaller arm name), P0 must meet the same bars or the tier is not comparable.  Then:
  some tier    the best arm: Delta(seg35) >= +.01 and 5/5 positive                                     the channels show headroom on this optimizer
  all 3 tiers  comparable, every K > 1 arm of the tier read, and every one of them: Delta(seg35) <= +.005 or fewer than 4 of 5 positive      no headroom on this optimizer
  no tier comparable                                                                                   undecidable
  otherwise                                                                                            middle band
An arm not read (not finished, or over the SDlogJ bar) can stop a tier from counting as no headroom, never make it count as headroom (plan revision 1; review 阻断 1).
Plan revision 2 (review of the fix round, 重要 1, 次要 1): "no headroom" asks every read arm, not only the best; "headroom" stays on the best arm, and another arm at its bar is named in
scope_note and in the tier's section.  reading_basis says whether a middle band comes from the effect ('effect') or from a tier or arm not read ('not_read', also for undecidable); scope_note
names the tiers not compared and their K > 1 arms.
"""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics

import mc0_core as core

SDLOGJ_MAX, NO_HEADROOM_BAR, HEADROOM_BAR, N, MIN_POSITIVE = .10, .005, .01, 5, 4
A5_FINAL = dict(seg35=.8705, seg4=.9289)        # A5 on the same 5 pairs, final state (d0a5_, analysis/long.csv: .87046, .92893); P0 short must be below both
SCHEMA = 'bayesreg.mc0.decision.v1'
MEASURES = ('seg35', 'seg4', 'sdlogj_official', 'sdlogj_brain', 'small_six', 'large_six', 'other_23', 'cortex', 'white_matter')
READ_NONE = '这个优化器上看不到通道余量'
READ_SEE = '这个优化器上看得到通道余量'
READ_MIDDLE = '中间带（n = 5 的 MDE 约 .012–.02）'
READ_MIDDLE_NOT_READ = '中间带（有档或臂没读到，不一定是效应落在 .005–.01，见 scope_note）'
READ_UNDECIDABLE = '不可判：没有一档可比'
INCOMPLETE = f'没有 {N} 次完成并评测的运行'
NOTE = ('开发集前 5 对，单次观测，全部用真值标签（oracle，只作上限），场没有过原生层。读数不是结论，下一步都交 BZ。'
        '阴性不是逻辑证明：Adam 加场平滑不是原生层，A0 的原生层把皮层又提了约 +.083。'
        '阳性不代表可认证，也不代表预测通道能兑现。这些场不是方法，不进任何结果表。'
        '尺度只取一个值，只在原生层 u = 0 拉平：K > 1 的 SDlogJ 明显低于 P0 时比较的是不同工作点，阴性只针对这个尺度；粗层没有拉平（见 r_ℓ），〔推断〕P3 可能在粗层上吃亏，看 r_ℓ；P3 的阴性不说明权重没用。')


def selection_effect():
    """The cells "看得到" can fire on, under the plan as it stands."""
    arms = [spec['tiers'] for arm, spec in core.ARMS.items() if arm != 'P0']
    cells, most = sum(len(t) for t in arms), max(sum(tier in t for t in arms) for tier in core.ALL_TIERS)
    return dict(tiers=len(core.ALL_TIERS), most_arms_in_a_tier=most, arm_tier_cells=cells, per_cell_5_of_5_under_zero=1/2**N, rough_upper_bound=cells/2**N,
                text=(f'带选择效应：从 {len(core.ALL_TIERS)} 档、每档最多 {most} 个臂（共 {cells} 个臂×档）里取最好的，看得到只需一档。每格在零假设下 5/5 为正的概率是 1/{2**N}，'
                      f'粗略上界约 {cells/2**N:.2f}；各格相关，另有 Δ ≥ +.01 的门槛，实际低于这个数〔推断〕。'))


def tier_arms(tier):
    """The K > 1 arms of a tier."""
    return [arm for arm, spec in core.ARMS.items() if arm != 'P0' and tier in spec['tiers']]


def valid(row):
    """A finished, evaluated run; a run cut by the wall cap or stopped by a guard is kept and described, never read."""
    return row is not None and row.get('status') == 'done' and row.get('metrics') is not None


def series(rows, arm, tier, pairs):
    runs = [rows.get((arm, tier, p)) for p in pairs]
    return runs if len(runs) == N and all(valid(r) for r in runs) else None


def mean_of(runs, measure):
    return statistics.fmean(r['metrics'][measure] for r in runs)


def tier_reading(rows, tier, pairs=core.PAIRS, *, baseline='P0', arms_of=tier_arms):
    """baseline, arms_of: MC0P's reading passes Q0 and its deciding arms (decide_mc0p.py); the defaults are MC0's."""
    rec = dict(tier=tier, comparable=False, reason=None, best_arm=None, delta_seg35=None, delta_seg4=None, positive_seg35=None, positive_seg4=None, n=N,
               differences_seg35=None, no_headroom=False, predicate_on_read_arms=None, headroom=False, excluded={}, arm_effects={}, other_arms_at_headroom_bar=[], delta_sdlogj_official=None, working_point=None)
    base = series(rows, baseline, tier, pairs)
    if base is None:
        return dict(rec, reason=f'{baseline} {INCOMPLETE}')
    if mean_of(base, 'sdlogj_official') > SDLOGJ_MAX:
        return dict(rec, reason=f"{baseline} 官方 SDlogJ 均值 {mean_of(base, 'sdlogj_official'):.4f} > {SDLOGJ_MAX}")
    eligible = {}
    for arm in arms_of(tier):
        runs = series(rows, arm, tier, pairs)
        if runs is None:
            rec['excluded'][arm] = INCOMPLETE
        elif mean_of(runs, 'sdlogj_official') > SDLOGJ_MAX:
            rec['excluded'][arm] = f"官方 SDlogJ 均值 {mean_of(runs, 'sdlogj_official'):.4f} > {SDLOGJ_MAX}"
        else:
            eligible[arm] = runs
    if not eligible:
        return dict(rec, reason='没有符合条件的 K > 1 臂')
    best = max(sorted(eligible), key=lambda arm: mean_of(eligible[arm], 'seg35'))

    def differences(arm, measure):
        return [a['metrics'][measure]-b['metrics'][measure] for a, b in zip(eligible[arm], base)]
    def effect(arm):
        d = differences(arm, 'seg35')
        delta, positive = statistics.fmean(d), sum(x > 0 for x in d)
        return dict(delta_seg35=delta, positive_seg35=positive, no_headroom=bool(delta <= NO_HEADROOM_BAR or positive < MIN_POSITIVE), at_headroom_bar=bool(delta >= HEADROOM_BAR and positive == N))
    def point(measure):
        return {'best': mean_of(eligible[best], measure), baseline: mean_of(base, measure), 'delta': mean_of(eligible[best], measure)-mean_of(base, measure)}
    effects = {arm: effect(arm) for arm in eligible}
    d4 = differences(best, 'seg4')
    predicate = all(e['no_headroom'] for e in effects.values())       # plan revision 2: every arm read, not only the one with the largest mean
    headroom = effects[best]['at_headroom_bar']                       # the rule of "看得到" is the plan's: the best arm only; another arm at the bar is only named
    return dict(rec, comparable=True, best_arm=best, delta_seg35=effects[best]['delta_seg35'], delta_seg4=statistics.fmean(d4), positive_seg35=effects[best]['positive_seg35'], positive_seg4=sum(d > 0 for d in d4),
                differences_seg35=differences(best, 'seg35'), arm_effects=effects, predicate_on_read_arms=predicate, no_headroom=predicate and not rec['excluded'],     # an arm not read: this tier does not count as no headroom
                headroom=headroom, other_arms_at_headroom_bar=[] if headroom else [arm for arm, e in effects.items() if e['at_headroom_bar']], delta_sdlogj_official=point('sdlogj_official')['delta'],
                working_point=dict(sdlogj_official=point('sdlogj_official'), sdlogj_brain=point('sdlogj_brain')))


def mean_ratios(runs, key):
    """The mean r_l per level over the runs with all four levels recorded."""
    values = [r[key] for r in runs if r.get(key) and len(r[key]) == 4 and all(x is not None for x in r[key])]
    return [statistics.fmean(v[level] for v in values) for level in range(4)] if values else None


def mean_present(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def arm_tables(rows, pairs=core.PAIRS, *, baseline='P0', arms=None, tiers=core.ALL_TIERS):
    """Every arm in every tier: means over its evaluated runs and paired differences against the baseline, whatever its status (descriptive; nothing is hidden).
    baseline, arms, tiers: MC0P's reading passes its own; the defaults are MC0's."""
    arms = core.ARMS if arms is None else arms
    against = f'against_{baseline}'
    out = {}
    for tier in tiers:
        out[tier] = {}
        for arm, spec in arms.items():
            if tier not in spec['tiers']:
                continue
            runs = {p: rows.get((arm, tier, p)) for p in pairs}
            good = {p: r for p, r in runs.items() if r is not None and r.get('metrics') is not None}
            row = dict(n_evaluated=len(good), n_read=sum(valid(r) for r in runs.values()), statuses={core.short_name(p): None if r is None else r.get('status') for p, r in runs.items()},
                       means={m: mean_present(r['metrics'][m] for r in good.values()) for m in MEASURES}, **{against: {}},
                       level_ratio_raw=mean_ratios([r for r in runs.values() if r], 'level_ratio_raw'), level_ratio_smoothed=mean_ratios([r for r in runs.values() if r], 'level_ratio_smoothed'),
                       at_accepted_alpha={m: mean_present((r.get('at_accepted_alpha') or {}).get(m) for r in good.values()) for m in ('seg35', 'seg4')},
                       scale=[r.get('D_multiplier') for r in good.values()], alpha=[r.get('accepted_alpha') for r in good.values()],
                       n_infeasible_at_1=[r.get('n_infeasible_at_1') for r in good.values()], status_at_1=[r.get('status_at_1') for r in good.values()], device_peak_reserved_gib=max([r.get('device_peak_reserved_gib') or 0. for r in runs.values() if r], default=None),
                       seconds=[r.get('seconds_total') for r in runs.values() if r])
            for m in MEASURES:
                base = {p: rows.get((baseline, tier, p)) for p in good}
                d = [good[p]['metrics'][m]-base[p]['metrics'][m] for p in good if base[p] is not None and base[p].get('metrics') is not None
                     and good[p]['metrics'][m] is not None and base[p]['metrics'][m] is not None]
                row[against][m] = dict(mean=statistics.fmean(d) if d else None, positive=sum(x > 0 for x in d), n=len(d))
            out[tier][arm] = row
    return out


def _effect_text(effect):
    return f"Δ seg35 {effect['delta_seg35']:+.4f}，{effect['positive_seg35']}/{N} 为正"


def other_arms_text(tier, reading):
    """Plan revision 2: the best arm is below the bar of "看得到", another arm read is at it.  The rule does not change: the tier does not count as headroom, nor as no headroom."""
    named = '、'.join(f"{a}（{_effect_text(reading['arm_effects'][a])}）" for a in reading['other_arms_at_headroom_bar'])
    return (f"{tier}：{named}达到\"看得到\"的门槛（Δ ≥ +{HEADROOM_BAR}、{N}/{N}），但不是均值最高的臂；最好的臂 {reading['best_arm']}（{_effect_text(reading['arm_effects'][reading['best_arm']])}）没有达到。"
            '按预登记规则只判均值最高的臂，这一档不算"看得到"，也不算"看不到"')


def scope_text(tiers, reading, arms_of=tier_arms):
    """What the reading line does not say: the tiers not compared and their K > 1 arms, the arms not read, and what the arms that were read show.  None when there is nothing to add."""
    parts = []
    not_compared = [f"{t}：{r['reason']}，因此没比的 K > 1 臂：{'、'.join(arms_of(t))}" for t, r in tiers.items() if not r['comparable']]
    if not_compared:
        parts.append('不可比的档（这些档不能算"看不到"）：'+'；'.join(not_compared))
    not_read = [f"{t}：{'、'.join(f'{a}（{why}）' for a, why in r['excluded'].items())}" for t, r in tiers.items() if r['excluded']]
    if not_read:
        parts.append('没读到的臂（这些档不能算"看不到"）：'+'；'.join(not_read))
    if reading == READ_MIDDLE_NOT_READ:         # the reading line speaks of what was not read; an arm that was read can be in the band as well
        also = [f"{t}：{'、'.join(f'{a}（{_effect_text(e)}）' for a, e in r['arm_effects'].items() if not e['no_headroom'])}" for t, r in tiers.items() if r['predicate_on_read_arms'] is False]
        if also:
            parts.append('读到的臂本身也有不满足"没有余量"的（这部分不是没读到造成的）：'+'；'.join(also))
    others = [other_arms_text(t, r) for t, r in tiers.items() if r['other_arms_at_headroom_bar']]
    if others:
        parts.append('；'.join(others))
    return '。'.join(parts) if parts else None


def decide(rows, pairs=core.PAIRS, stops=None):
    tiers = {t: tier_reading(rows, t, pairs) for t in core.ALL_TIERS}
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
    basis = 'not_read' if reading in (READ_MIDDLE_NOT_READ, READ_UNDECIDABLE) else 'effect'
    incomplete = sorted({a for r in tiers.values() for a, why in r['excluded'].items() if why == INCOMPLETE})
    p0 = series(rows, 'P0', 'short', pairs)
    check = dict(a5_final=A5_FINAL, p0_short=None if p0 is None else {m: mean_of(p0, m) for m in A5_FINAL},
                 below_a5_final=None if p0 is None else all(mean_of(p0, m) < v for m, v in A5_FINAL.items()), role='P0 短档的绝对值只核对脚本的量级，不进判据',
                 p0_images_equal_a5_inputs=sorted({str(r.get('images_equal_a5_inputs')) for (arm, _, _), r in rows.items() if arm == 'P0'}))
    return dict(schema=SCHEMA, reading=reading, reading_basis=basis, is_conclusion=False, next_step='交 BZ', tiers=tiers, comparable_tiers=comparable, arms_stopped=stops or {},
                tiers_not_comparable={t: dict(reason=r['reason'], k_arms_not_compared=tier_arms(t)) for t, r in tiers.items() if not r['comparable']},
                arms_incomplete=incomplete, scope_note=scope_text(tiers, reading), script_magnitude_check=check, selection_effect=selection_effect(),
                rules=dict(sdlogj_max=SDLOGJ_MAX, no_headroom_bar=NO_HEADROOM_BAR, headroom_bar=HEADROOM_BAR, n=N, min_positive=MIN_POSITIVE), note=NOTE, tables=arm_tables(rows, pairs))


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def collect(root, plan=None, scale_keys=('images_equal_a5_inputs',), run_keys=()):
    """{(arm, tier, pair): row} from each run directory's run.json and metrics.json.  plan, scale_keys, run_keys: MC0P's reading passes its own; the defaults are MC0's."""
    rows = {}
    for cell in (core.plan_runs() if plan is None else plan):
        d = core.run_dir(root, cell)
        run, metrics, ended = read_json(d/'run.json'), read_json(d/'metrics.json'), read_json(d/'queue_done.json')
        if run is None and metrics is None and ended is None:
            continue
        run, ended = run or {}, ended or {}
        scale = run.get('scale') or {}
        certificate = (metrics or {}).get('certificate') or {}
        norms = run.get('level_norms') or {}
        rows[(cell.arm, cell.tier, cell.pair)] = dict(run_id=cell.run_id, status=run.get('status', ended.get('state', 'failed')), metrics=None if metrics is None else metrics.get('metrics'),
                                                      D_multiplier=scale.get('D_multiplier'), **{k: scale.get(k) for k in scale_keys}, **{k: run.get(k) for k in run_keys}, accepted_alpha=certificate.get('accepted_alpha'), n_infeasible_at_1=certificate.get('n_infeasible_at_1'), status_at_1=certificate.get('status_at_1'),
                                                      device_peak_reserved_gib=run.get('device_peak_reserved_gib'), seconds_total=run.get('seconds_total'),
                                                      level_ratio_raw=norms.get('ratio_raw'), level_ratio_smoothed=norms.get('ratio_smoothed'), at_accepted_alpha=(metrics or {}).get('at_accepted_alpha'))
    return rows


def _fmt(v, nd=4):
    return '—' if v is None else (f'{v:.{nd}f}' if isinstance(v, float) else str(v))


def _fmt_list(values):
    return '—' if values is None else ' / '.join(_fmt(v, 3) for v in values)


def _fmt_at_1(row):
    """The cells infeasible at alpha = 1 and the status of that certificate, run by run: a domain failure says infeasible with 0 cells (feasibility.py, the early return)."""
    return '；'.join(f'{_fmt(n)}（{_fmt(s)}）' for n, s in zip(row['n_infeasible_at_1'], row['status_at_1'])) or '—'


def summary_text(decision):
    L = ['# MC0 读数（由 decide_mc0.py 写出）', '', f"**读数：{decision['reading']}**（reading_basis：{decision['reading_basis']}；不是结论，下一步交 BZ）", '', NOTE, '', decision['selection_effect']['text'], '',
         f"核对量级（P0 短档低于 A5 同 5 对终态 .8705 / .9289）：{decision['script_magnitude_check']['p0_short']}，{decision['script_magnitude_check']['below_a5_final']}；P0 的图与 A5 输入逐位相同：{decision['script_magnitude_check']['p0_images_equal_a5_inputs']}", '']
    if decision['arms_stopped']:
        L += [f"停掉的臂：{json.dumps(decision['arms_stopped'], ensure_ascii=False)}", '']
    if decision['scope_note']:
        L += [decision['scope_note'], '']
    for tier, reading in decision['tiers'].items():
        L += [f'## 档 {tier}', '', (f"最好的 K > 1 臂 {reading['best_arm']}：Δ seg35 {_fmt(reading['delta_seg35'])}（{reading['positive_seg35']}/{N} 为正），"
                                  f"Δ seg4 {_fmt(reading['delta_seg4'])}（{reading['positive_seg4']}/{N}）") if reading['comparable'] else f"不可比：{reading['reason']}", '']
        if reading['comparable']:
            point = reading['working_point']
            L += ['工作点（最好的臂 / P0 / 差）：' + '；'.join(f"{name} {_fmt(point[key]['best'])} / {_fmt(point[key]['P0'])} / {_fmt(point[key]['delta'])}" for key, name in (('sdlogj_official', '官方 SDlogJ'), ('sdlogj_brain', '脑内 SDlogJ'))), '']
        L += [f"predicate_on_read_arms（只看读到的臂时，每个臂都没有余量）：{_fmt(reading['predicate_on_read_arms'])}；no_headroom（这一档算没有余量）：{reading['no_headroom']}", '']
        if reading['excluded']:
            L += [f"没读到（这一档不能算\"看不到\"）：{json.dumps(reading['excluded'], ensure_ascii=False)}", '']
        if reading['other_arms_at_headroom_bar']:
            L += [other_arms_text(tier, reading), '']
        L += ['| 臂 | 评测/可读 | seg35 | seg4 | 官方 SDlogJ | 脑内 SDlogJ | 小六个 | 大六个 | 其余 23 个 | 皮层 | 白质 | Δ seg35 对 P0 | Δ seg4 对 P0 | Δ 官方 SDlogJ 对 P0 | Δ 脑内 SDlogJ 对 P0 | r_ℓ（层 8/4/2/1） | r_ℓ（G_grad 之后） | α | α = 1 不可行单元（status） | 接受 α 下 seg35 / seg4（oracle，不是方法） | 显存峰值 GiB |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for arm, row in decision['tables'][tier].items():
            d35, d4, dsd, dbrain = (row['against_P0'][m] for m in ('seg35', 'seg4', 'sdlogj_official', 'sdlogj_brain'))
            L.append(f"| {arm} | {row['n_evaluated']}/{row['n_read']} | " + ' | '.join(_fmt(row['means'][m]) for m in MEASURES)
                     + f" | {_fmt(d35['mean'])} ({d35['positive']}/{d35['n']}) | {_fmt(d4['mean'])} ({d4['positive']}/{d4['n']}) | {_fmt(dsd['mean'])} | {_fmt(dbrain['mean'])} | {_fmt_list(row['level_ratio_raw'])} | {_fmt_list(row['level_ratio_smoothed'])} | {row['alpha']} | {_fmt_at_1(row)} | "
                     f"{_fmt(row['at_accepted_alpha']['seg35'])} / {_fmt(row['at_accepted_alpha']['seg4'])} | {_fmt(row['device_peak_reserved_gib'], 2)} |")
        L.append('')
    return '\n'.join(L)+'\n'


def write_analysis(root, stops=None):
    """Writes analysis/: an earlier version is moved to analysis/previous_<UTC>/, never overwritten."""
    root = Path(root)
    out = root/'analysis'
    names = ('decision_mc0.json', 'long.csv', 'summary_mc0.md')
    if any((out/n).exists() for n in names):
        aside = out/f"previous_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        aside.mkdir(parents=True)
        for n in names:
            if (out/n).exists():
                shutil.move(str(out/n), str(aside/n))
    out.mkdir(parents=True, exist_ok=True)
    rows = collect(root)
    decision = decide(rows, stops=stops)
    (out/'decision_mc0.json').write_text(json.dumps(decision, indent=1, ensure_ascii=False)+'\n')
    (out/'summary_mc0.md').write_text(summary_text(decision), encoding='utf-8')
    with open(out/'long.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(('run_id', 'arm', 'tier', 'pair', 'status', 'K', 'oracle', 'field', 'D_multiplier', 'accepted_alpha', 'n_infeasible_at_1', 'measure', 'value'))
        for (arm, tier, pair), row in sorted(rows.items()):
            for m in MEASURES if row['metrics'] else ():
                w.writerow((row['run_id'], arm, tier, pair, row['status'], core.ARMS[arm]['k'], 'true', 'uncertified_alpha1', row['D_multiplier'], row['accepted_alpha'], row['n_infeasible_at_1'], m, row['metrics'][m]))
    return decision
