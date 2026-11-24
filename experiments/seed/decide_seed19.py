"""SEED19's reading (runs/20261104-seed19/plan.md, by code; a label for the lead and BZ, not a conclusion): analysis/decision_seed19.json, summary_seed19.md, long.csv under seed_v1b.

Delta = S_P - B1TM, pair by pair, unrounded.  Positive is strictly > 0; the bars are included.

Reading one, does the rest replicate (only the other 14 pairs: the first 5 were seen when the configuration was picked).  Delta14 = the mean over the 14 pairs, seg35:
  fewer than 12 pairs with a certified final field, or the mean official SDlogJ of those pairs > .10      not decidable
  Delta14 >= +.005 and at least 11 of 14 positive                                                         the other 14 pairs replicate
  Delta14 <= +.002 or at most 7 of 14 positive                                                            the other 14 pairs do not replicate
  otherwise                                                                                               middle band
Reading two, BZ's target (19 pairs: the first 5 of seed_v1, the other 14 of seed_v1b): seg35 mean >= .86, official SDlogJ mean <= .10, certificates 19/19: reached; otherwise not reached, by how much.  .88 described only.  A missing pair: the count, no judgement; a run without a final field with a passed certificate is a missing pair (plan revision 1).
The arm S_K1, only if it ran (n = 5): S_P - S_K1 >= +.005 and >= 4/5 positive: the gain comes mainly from the channels; S_K1 - B1TM >= +.005 and S_P - S_K1 <= +.002: mainly from the entry's steps and alpha, this does not need K = 35; otherwise middle band.
Rules the brief leaves open and this code fixes (plan.md, "工具补上的规则"; review them): see count_reasons, reading_one.

Everything else is only described.  This is one run per pair on the development set, the first 5 pairs have been seen, not a hold-out result.  The evaluation's functions come from experiments/d0/evaluate.py (read only); B1TM's and B1W_v2's numbers from b1_v1/analysis/long.csv, per-label Dice from the metrics.json of the same run_id.
"""
from pathlib import Path
import statistics

import numpy as np

import decide_seed as first
import run_seed as core
import run_seed19 as batch

SCHEMA = 'bayesreg.seed19.decision.v1'
N_REST, N_ALL = len(batch.REST_PAIRS), len(batch.FIRST_PAIRS)+len(batch.REST_PAIRS)
REPLICATE_BAR, NOT_REPLICATE_BAR, REPLICATE_POSITIVE, NOT_REPLICATE_POSITIVE, MIN_PAIRS, SDLOGJ_MAX = .005, .002, 11, 7, 12, .10
TARGET, STRETCH = .86, .88
K1_BAR, K1_NEAR_BAR, K1_POSITIVE, K1_N = .005, .002, 4, 5
PINS = dict(B1TM=(.8468, .9083))                       # (seg35, seg4) of b1_v1 over the 19 pairs (runs/20261103-b1/result.md, status block of the project status log), to 4 decimals; the rule of decide_seed.py
FIRST5_PIN = (.8430, .8984)                            # S_P of seed_v1 over the first 5 pairs (runs/20261104-seed/result.md)
PIN_TOL = first.PIN_TOL
REPLICATED, NOT_REPLICATED, MIDDLE, NOT_DECIDABLE = 'replicated', 'not_replicated', 'middle', 'not_decidable'
READINGS = {REPLICATED: '其余 14 对复现', NOT_REPLICATED: '其余 14 对没有复现', MIDDLE: '中间带', NOT_DECIDABLE: '不可判'}
REACHED, NOT_REACHED, NOT_JUDGED = 'reached', 'not_reached', 'not_judged'
K1_CHANNELS, K1_STEPS, K1_MIDDLE, K1_NOT_RUN, K1_NOT_DECIDABLE = 'channels', 'steps_and_alpha', 'middle', 'not_run', 'not_decidable'
K1_READINGS = {K1_CHANNELS: '增益主要来自通道', K1_STEPS: '增益主要来自入口的步数与 α，这一条不需要 K = 35', K1_MIDDLE: '中间带', K1_NOT_RUN: 'S_K1 没有跑，不判', K1_NOT_DECIDABLE: 'S_K1 不可判'}
SEG4_NAMES = dict(cortex=1, subcortical=2, white_matter=3, csf=4)      # experiments/d0/evaluate.py SEG4_STRUCTURES
NOTE = ('开发集 19 对，每对单次观测，读数不是结论，下一步交 BZ。前 5 对是挑配置时见过的（B1TM 本身也在这 5 对上选出），不是留出集结果。S_P 的入口和上色都来自 unet35_v1_ens 的预测（oracle: false），分割器用训练集标签训练，真值只进评测。'
        '路径改变不改目标函数，原生层仍在优化标量 D；要进方法先改合同（C12-4a 不覆盖这种入口）。到目标不等于留出集，也不说明它已是方法；没到不说明多通道求解器有用或没用。'
        '入口成本持时 47 s，为了与前 5 对可比不改（复核次要 11：无竞争实测 61 s，方向有利于 S_P）；S_P 的原生层比 B1TM 短约 45 s。预算收尾不是收敛。基线不是同时段（b1_v1）。入口用接受 α 的数，α = 1 的数在括号里。')


def batch_of(pair):
    return 'first_5' if pair in batch.FIRST_PAIRS else 'rest_14'


def seed_root_of(pair, root, results):
    """S_P of the first 5 pairs is read only from seed_v1; the other 14 only from this batch."""
    return Path(results)/batch.FIRST_ROOT if pair in batch.FIRST_PAIRS else Path(root)


def load_runs(root, results):
    """{(arm, pair): metrics.json}: S_P of the 19 pairs (first 5: seed_v1, the others: this root), S_K1 of this root."""
    out = {}
    for pair in batch.FIRST_PAIRS+batch.REST_PAIRS:
        m = core.read_json(seed_root_of(pair, root, results)/'runs/S_P'/pair/'metrics.json')
        if m is not None:
            out[('S_P', pair)] = m
    for pair in batch.FIRST_PAIRS:
        m = core.read_json(Path(root)/'runs/S_K1'/pair/'metrics.json')
        if m is not None:
            out[('S_K1', pair)] = m
    return out


def plan_configuration(arm, results):
    return dict(seconds=core.SECONDS, changes=core.CHANGES, inputs_root=str(batch.inputs_root(arm, results)))


def count_reasons(m, arm, results):
    """Why this run does not count as a certified final field of the reading (empty: it counts).  The rule of decide_seed.py after SEED's revision 1: class ok, at least one native commit (otherwise it did not pass the native level), a passed certificate bound to the exported
    field, the entry injected as recorded, the plan's configuration."""
    if m is None:
        return ['没有 metrics.json']
    if not m.get('metrics'):
        return [f"无终态场（{(m.get('classification') or {}).get('kind')}）"]
    reasons = []
    cls = (m.get('classification') or {}).get('class')
    if cls != 'ok':
        reasons.append(f'分类 {cls}，不算过了原生层')
    native = m.get('native_commits')
    if isinstance(native, bool) or not isinstance(native, int) or native < 1:
        reasons.append(f'原生提交 {native}，没有过原生层')
    if not first.final_ok(m):
        reasons.append('终态证书未通过，或没有与导出场绑定（state_hash、npz sha256）')
    seed = m.get('seed') or {}
    if seed.get('passed') is not True:
        reasons.append(f"入口核对未通过 {seed.get('problems')}")
    have, want = seed.get('arm') or {}, plan_configuration(arm, results)
    if {k: have.get(k) for k in want} != want or have.get('charge_seconds') != core.ENTRY_CHARGE_SECONDS:
        reasons.append(f'配置与计划不同 {have}')
    return reasons


def control_ok(run):
    return bool(run and run.get('metrics') and (run.get('certificate') or {}).get('passed'))


def paired(runs, arm, ref, key, pairs):
    """Per pair arm - ref over the pairs where both have the value; the mean, median, positive count over them."""
    values = {}
    for pair in pairs:
        a, b = (runs.get((arm, pair)) or {}).get('metrics'), (runs.get((ref, pair)) or {}).get('metrics')
        if a is not None and b is not None and a.get(key) is not None and b.get(key) is not None:
            values[pair] = a[key]-b[key]
    return dict(per_pair={core.short_name(p): v for p, v in values.items()}, mean=float(np.mean(list(values.values()))) if values else None, median=float(statistics.median(values.values())) if values else None,
                positive=sum(v > 0 for v in values.values()), n=len(values))


def reading_one(runs, reference_problems, results):
    """The table of the plan on the 14 pairs.  The pairs that count have count_reasons empty and a B1TM with a certified final field; Delta14 is over them, the positive count against the absolute 11 and 7: a pair that does not count is not positive."""
    excluded = {}
    for pair in batch.REST_PAIRS:
        reasons = count_reasons(runs.get(('S_P', pair)), 'S_P', results)
        if not control_ok(runs.get(('B1TM', pair))):
            reasons = reasons+['B1TM 没有通过证书的终态场']
        if reasons:
            excluded[core.short_name(pair)] = reasons
    counted = [p for p in batch.REST_PAIRS if core.short_name(p) not in excluded]
    delta, delta4 = paired(runs, 'S_P', 'B1TM', 'seg35', counted), paired(runs, 'S_P', 'B1TM', 'seg4', counted)
    sdlogj = float(np.mean([runs[('S_P', p)]['metrics']['sdlogj_official'] for p in counted])) if counted else None
    reasons = list(reference_problems)
    if len(counted) < MIN_PAIRS:
        reasons.append(f'只有 {len(counted)} 对有通过证书的终态场（要 ≥ {MIN_PAIRS}）')
    if sdlogj is not None and sdlogj > SDLOGJ_MAX:
        reasons.append(f'这些对的官方 SDlogJ 均值 {sdlogj:.4f} > {SDLOGJ_MAX}')
    if reasons:
        code = NOT_DECIDABLE
    elif delta['mean'] >= REPLICATE_BAR and delta['positive'] >= REPLICATE_POSITIVE:
        code = REPLICATED
    elif delta['mean'] <= NOT_REPLICATE_BAR or delta['positive'] <= NOT_REPLICATE_POSITIVE:
        code = NOT_REPLICATED
    else:
        code = MIDDLE
    text = READINGS[code]+('：'+'；'.join(reasons) if reasons else f"：Δ14(seg35) {delta['mean']:+.4f}，{delta['positive']}/{N_REST} 为正（{len(counted)} 对计入），Δ14(seg4) {delta4['mean']:+.4f}（只描述）")
    return dict(code=code, text=text, reasons=reasons, n=len(counted), excluded=excluded, delta_seg35=delta, delta_seg4=delta4, sdlogj_official_mean=sdlogj)


def reading_two(runs, first5_problems, results):
    """BZ's target on the 19 pairs.  A pair is there when its run has a final field with a passed certificate, the entry check passed and the plan's configuration, the charge included; anything else is a missing pair (plan revision 1): only the count, no judgement.  So at 19 pairs the certificates are 19/19."""
    have, missing = [], {}
    want = plan_configuration('S_P', results)
    for pair in batch.FIRST_PAIRS+batch.REST_PAIRS:
        m = runs.get(('S_P', pair))
        seed = (m or {}).get('seed') or {}
        arm = seed.get('arm') or {}
        why = ('没有 metrics.json' if m is None else '无终态场' if not m.get('metrics') else '终态证书未通过，或没有与导出场绑定' if not first.final_ok(m) else f"入口核对未通过 {seed.get('problems')}" if seed.get('passed') is not True
               else f'配置与计划不同 {arm}' if {k: arm.get(k) for k in want} != want or arm.get('charge_seconds') != core.ENTRY_CHARGE_SECONDS else None)
        if why is None:
            have.append(pair)
        else:
            missing[core.short_name(pair)] = why
    values = {key: float(np.mean([runs[('S_P', p)]['metrics'][key] for p in have])) if have else None for key in ('seg35', 'seg4', 'sdlogj_official')}
    out = dict(n=len(have), missing=missing, seg35=values['seg35'], seg4=values['seg4'], sdlogj_official=values['sdlogj_official'], certificates=len(have), role='开发集单次，前 5 对见过，不是留出集结果', stretch=STRETCH, stretch_gap=None if values['seg35'] is None else STRETCH-values['seg35'], problems=list(first5_problems))
    if len(have) < N_ALL or first5_problems:
        reasons = ([f'有缺对：{len(have)}/{N_ALL} 对有通过证书的终态场'] if len(have) < N_ALL else [])+list(first5_problems)
        return dict(out, code=NOT_JUDGED, gaps=None, text='；'.join(reasons)+'，不判')
    gaps = dict(seg35=max(TARGET-values['seg35'], 0.), sdlogj_official=max(values['sdlogj_official']-SDLOGJ_MAX, 0.))
    reached = values['seg35'] >= TARGET and values['sdlogj_official'] <= SDLOGJ_MAX
    text = f"19 对 seg35 {values['seg35']:.4f} / seg4 {values['seg4']:.4f}，官方 SDlogJ {values['sdlogj_official']:.4f}，证书 {len(have)}/{N_ALL}"
    if reached:
        return dict(out, code=REACHED, gaps=gaps, text=f'到目标：{text}（开发集单次，前 5 对见过，不是留出集）')
    return dict(out, code=NOT_REACHED, gaps=gaps, text=f"没到：{text}；差 seg35 {gaps['seg35']:.4f}，SDlogJ 超 {gaps['sdlogj_official']:.4f}（开发集单次，前 5 对见过，不是留出集）")


def reading_s_k1(runs, root, first5_problems, results):
    """Only if the arm ran: no run directory and no record of it: not run.  n = 5, against S_P of seed_v1 and B1TM."""
    pairs = batch.FIRST_PAIRS
    started = any((Path(root)/'runs/S_K1'/p).exists() for p in pairs) or any(('S_K1', p) in runs for p in pairs)
    if not started:
        return dict(code=K1_NOT_RUN, text=K1_READINGS[K1_NOT_RUN], reasons=[], s_p_minus_s_k1=None, s_k1_minus_b1tm=None, n=K1_N)
    reasons = list(first5_problems)
    for pair in pairs:
        short = core.short_name(pair)
        reasons += [f'S_K1 {short}: {r}' for r in count_reasons(runs.get(('S_K1', pair)), 'S_K1', results)]
        reasons += [f'S_P {short}: {r}' for r in count_reasons(runs.get(('S_P', pair)), 'S_P', results)]
        if not control_ok(runs.get(('B1TM', pair))):
            reasons.append(f'B1TM {short}: 没有通过证书的终态场')
    a, b = paired(runs, 'S_P', 'S_K1', 'seg35', pairs), paired(runs, 'S_K1', 'B1TM', 'seg35', pairs)
    if reasons:
        code = K1_NOT_DECIDABLE
    elif a['mean'] >= K1_BAR and a['positive'] >= K1_POSITIVE:
        code = K1_CHANNELS
    elif b['mean'] >= K1_BAR and a['mean'] <= K1_NEAR_BAR:
        code = K1_STEPS
    else:
        code = K1_MIDDLE
    a4, b4 = paired(runs, 'S_P', 'S_K1', 'seg4', pairs), paired(runs, 'S_K1', 'B1TM', 'seg4', pairs)
    text = K1_READINGS[code]+('：'+'；'.join(reasons) if reasons else f"：seg35 S_P − S_K1 {a['mean']:+.4f}（{a['positive']}/{K1_N} 为正），S_K1 − B1TM {b['mean']:+.4f}（{b['positive']}/{K1_N}）；seg4（只描述）{a4['mean']:+.4f}、{b4['mean']:+.4f}；n = 5")
    return dict(code=code, text=text, reasons=reasons, s_p_minus_s_k1=a, s_k1_minus_b1tm=b, n=K1_N, s_p_minus_s_k1_seg4=a4, s_k1_minus_b1tm_seg4=b4)


# ---- description --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def run_trace(run):
    """A finished run's own records: the transfer alpha, the D at the entry (history's first native row, at that alpha), the D of the last native commit, the native level's measured length (levels.json: level_completed minus level_prepared).  Empty values when a file is not there."""
    run = Path(run)
    protocol = core.read_json(run/'protocol.json') or {}
    task = run/((protocol.get('task') or {}).get('name') or 'missing')
    levels = core.read_json(task/'levels.json') or []
    prepared = next((e for e in levels if e.get('event') == 'level_prepared' and e.get('level') == 3), {})
    completed = next((e for e in levels if e.get('event') == 'level_completed' and e.get('level') == 3), {})
    native = [h for h in core.read_json(task/'history.json') or [] if h.get('level') == 3 and h.get('outer_accepted') is not False]
    a, b = prepared.get('visible_seconds'), completed.get('visible_seconds')
    return dict(alpha=(prepared.get('transfer_recovery') or {}).get('accepted_alpha'), D_entry=native[0].get('D_start') if native else None, D_final=native[-1].get('D_candidate') if native else None, native_seconds=b-a if a is not None and b is not None else None)


def entry_root_of(pair):
    """The batch of S_P's entry of this pair: decide_seed's map for the first 5 (run_seed.S_P_ROOTS), mc0p_v1c for the 14."""
    return core.S_P_ROOTS[core.short_name(pair)] if pair in batch.FIRST_PAIRS else batch.ENTRY_ROOT


def entry_numbers(pair, results, arm='S_P'):
    """The evaluation's numbers of the arm's entry: at the accepted alpha (what entered the native level) and at alpha = 1, after the taper."""
    root, source = (entry_root_of(pair), 'Q2_long') if arm == 'S_P' else (batch.ARMS[arm]['entry_root'], f"{batch.ARMS[arm]['source_arm']}_long")
    m = core.read_json(Path(results)/root/'runs'/source/core.short_name(pair)/'metrics.json') or {}
    x, at = m.get('metrics') or {}, m.get('at_accepted_alpha') or {}
    return dict(root=root, accepted_alpha=(m.get('certificate') or {}).get('accepted_alpha'), seg35=at.get('seg35'), seg4=at.get('seg4'), seg35_alpha1=x.get('seg35'), seg4_alpha1=x.get('seg4'), sdlogj_alpha1=x.get('sdlogj_official'))


def group_table(runs, ev, arm, pairs):
    labels = ev.label_rows(runs, arm, arm, 'seg4', tuple(SEG4_NAMES.values()), pairs)
    return dict({g: ev.a5_group_mean(runs, arm, members, pairs) for g, members in ev.A5_GROUPS.items()}, **{name: labels[k]['arm_mean'] for name, k in SEG4_NAMES.items()})


def group_differences(runs, ev, pairs):
    labels = ev.label_rows(runs, 'S_P', 'B1TM', 'seg4', tuple(SEG4_NAMES.values()), pairs)
    return dict({g: ev.a5_group_row(runs, 'S_P', 'B1TM', members, pairs) for g, members in ev.A5_GROUPS.items()}, **{name: labels[k] for name, k in SEG4_NAMES.items()})


def describe(root, runs, ev, results):
    """The tables that decide nothing."""
    sets = dict(rest_14=list(batch.REST_PAIRS), all_19=list(batch.FIRST_PAIRS+batch.REST_PAIRS), first_5=list(batch.FIRST_PAIRS))
    measures = ('seg35', 'seg4', 'hd95', 'sdlogj_official', 'sdlogj_brain', 'pct_J_lt_half_brain')

    def means(arm, pairs):
        out = {}
        for key in measures:
            values = [runs[(arm, p)]['metrics'][key] for p in pairs if ((runs.get((arm, p)) or {}).get('metrics') or {}).get(key) is not None]
            out[key] = float(np.mean(values)) if values else None
        out['n'] = sum(bool((runs.get((arm, p)) or {}).get('metrics')) for p in pairs)
        return out
    rows = []
    for pair in sets['all_19']:
        short = core.short_name(pair)
        m, c, v2 = runs.get(('S_P', pair)) or {}, runs.get(('B1TM', pair)) or {}, runs.get(('B1W_v2', pair)) or {}
        x, cx, vx = m.get('metrics') or {}, c.get('metrics') or {}, v2.get('metrics') or {}
        mine, theirs = run_trace(seed_root_of(pair, root, results)/'runs/S_P'/pair), run_trace(Path(results)/'b1_v1/runs/B1TM'/pair)
        rows.append(dict(pair=short, batch=batch_of(pair), run_id=m.get('run_id'), control_run_id=c.get('run_id'), cls=(m.get('classification') or {}).get('class'), status=m.get('status'), entry=entry_numbers(pair, results), seg35=x.get('seg35'), seg4=x.get('seg4'), control_seg35=cx.get('seg35'), control_seg4=cx.get('seg4'),
                         hd95=x.get('hd95'), control_hd95=cx.get('hd95'), sdlogj_official=x.get('sdlogj_official'), control_sdlogj_official=cx.get('sdlogj_official'), sdlogj_brain=x.get('sdlogj_brain'), control_sdlogj_brain=cx.get('sdlogj_brain'), certificate=(m.get('certificate') or {}).get('passed'),
                         min_certified_bound=(m.get('certificate') or {}).get('min_jd_lower_bound'), native_commits=m.get('native_commits'), control_native_commits=c.get('native_commits'), alpha=mine['alpha'], control_alpha=theirs['alpha'], D_entry=mine['D_entry'], control_D_entry=theirs['D_entry'], D_final=mine['D_final'], control_D_final=theirs['D_final'],
                         native_seconds=mine['native_seconds'], control_native_seconds=theirs['native_seconds'], e2e=m.get('e2e_seconds'), a5pv2_seg35=vx.get('seg35'), a5pv2_seg4=vx.get('seg4'), a5pv2_sdlogj_official=vx.get('sdlogj_official')))
    return dict(means={name: dict(S_P=means('S_P', pairs), B1TM=means('B1TM', pairs), B1W_v2=means('B1W_v2', pairs)) for name, pairs in sets.items()},
                differences={name: {key: paired(runs, 'S_P', 'B1TM', key, pairs) for key in ('seg35', 'seg4', 'hd95', 'sdlogj_official', 'sdlogj_brain')} for name, pairs in sets.items()},
                against_a5pv2={name: {key: paired(runs, 'S_P', 'B1W_v2', key, pairs) for key in ('seg35', 'seg4')} for name, pairs in sets.items()},
                groups={name: dict(S_P=group_table(runs, ev, 'S_P', pairs), B1TM=group_table(runs, ev, 'B1TM', pairs)) for name, pairs in sets.items()},
                group_differences={name: group_differences(runs, ev, pairs) for name, pairs in sets.items()}, rows=rows, subject_0440=[r for r in rows if r['pair'] in ('0439-0440', '0440-0441')])


def describe_s_k1(root, runs, results):
    """The pairs of S_K1 that ran, beside S_P of seed_v1 and B1TM.  Decides nothing."""
    rows = []
    for pair in batch.FIRST_PAIRS:
        m = runs.get(('S_K1', pair))
        if m is None:
            continue
        x, s, c = m.get('metrics') or {}, (runs.get(('S_P', pair)) or {}).get('metrics') or {}, (runs.get(('B1TM', pair)) or {}).get('metrics') or {}
        trace = run_trace(Path(root)/'runs/S_K1'/pair)
        rows.append(dict(pair=core.short_name(pair), run_id=m.get('run_id'), cls=(m.get('classification') or {}).get('class'), entry=entry_numbers(pair, results, 'S_K1'), seg35=x.get('seg35'), seg4=x.get('seg4'), s_p_seg35=s.get('seg35'), s_p_seg4=s.get('seg4'), control_seg35=c.get('seg35'), control_seg4=c.get('seg4'), sdlogj_official=x.get('sdlogj_official'), hd95=x.get('hd95'),
                         alpha=trace['alpha'], D_entry=trace['D_entry'], D_final=trace['D_final'], native_commits=m.get('native_commits'), native_seconds=trace['native_seconds'], certificate=(m.get('certificate') or {}).get('passed')))
    return rows


def pin_problems(runs):
    """B1TM's means over the 19 pairs against the pin; a different number means a different baseline from the one the plan read."""
    pairs = batch.FIRST_PAIRS+batch.REST_PAIRS
    bad, seen = [], {}
    for arm, (seg35, seg4) in PINS.items():
        values = [(runs.get((arm, p)) or {}).get('metrics') for p in pairs]
        values = [v for v in values if v]
        a, b = (float(np.mean([v[k] for v in values])) if len(values) == len(pairs) else None for k in ('seg35', 'seg4'))
        seen[arm] = dict(seen=(None if a is None else round(a, 5), None if b is None else round(b, 5)), pinned=(seg35, seg4))
        if a is None or b is None or abs(a-seg35) > PIN_TOL or abs(b-seg4) > PIN_TOL:
            bad.append(f'{arm}: 19 对 seg35 {a}, seg4 {b}，计划钉住 {seg35}, {seg4}')
    return bad, seen


def first5_problems(runs):
    values = [(runs.get(('S_P', p)) or {}).get('metrics') for p in batch.FIRST_PAIRS]
    if not all(values):
        return [f'{batch.FIRST_ROOT} 的 S_P 前 5 对不全']
    a, b = (float(np.mean([v[k] for v in values])) for k in ('seg35', 'seg4'))
    return [] if abs(a-FIRST5_PIN[0]) <= PIN_TOL and abs(b-FIRST5_PIN[1]) <= PIN_TOL else [f'{batch.FIRST_ROOT} 的 S_P 前 5 对 seg35 {a}, seg4 {b}，记录是 {FIRST5_PIN}']


def control_configuration_problems(results):
    """B1TM's records against the plan: the same budget, options and inputs root, on the 19 pairs."""
    bad = []
    want = plan_configuration('S_P', results)
    for pair in batch.FIRST_PAIRS+batch.REST_PAIRS:
        b1 = (core.read_json(Path(results)/'b1_v1/runs/B1TM'/pair/'metrics.json') or {}).get('b1') or {}
        have = dict(seconds=b1.get('seconds'), changes=b1.get('changes'), inputs_root=b1.get('inputs_root'))
        if have != want:
            bad.append(f'B1TM {core.short_name(pair)}: configuration {have}')
    return bad


def anomalies(runs, ev, results):
    keys = [('S_P', p) for p in batch.FIRST_PAIRS+batch.REST_PAIRS]+[('S_K1', p) for p in batch.FIRST_PAIRS if ('S_K1', p) in runs]
    out = ev.b1_anomalies(runs, keys)
    for arm, pair in keys:
        seed = (runs.get((arm, pair)) or {}).get('seed') or {}
        e = seed.get('entry_metrics') or {}
        if seed.get('accepted_alpha') is not None and e.get('accepted_alpha') is not None and seed['accepted_alpha'] != e['accepted_alpha']:
            out.append(dict(arm=arm, pair=pair, what=f"转移 α {seed['accepted_alpha']} 与入口评测的 α {e['accepted_alpha']} 不同"))
    return out


# ---- the files ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
f = first.f


def stats(row, digits=4):
    return f"{f(row['mean'], digits)}（中位 {f(row['median'], digits)}，{row['positive']}/{row['n']} 为正）"


def pair_row(r):
    e = r['entry']
    return (f"| {r['pair']} | {'前 5' if r['batch'] == 'first_5' else '其余 14'} | {f(r['seg35'])} / {f(r['seg4'])} | {f(r['control_seg35'])} / {f(r['control_seg4'])} | {f(None if r['seg35'] is None or r['control_seg35'] is None else r['seg35']-r['control_seg35'])} | {f(None if r['seg4'] is None or r['control_seg4'] is None else r['seg4']-r['control_seg4'])} | "
            f"{f(e['seg35'])} / {f(e['seg4'])}（α = 1：{f(e['seg35_alpha1'])} / {f(e['seg4_alpha1'])}） | {f(r['alpha'], 5)} / {f(r['control_alpha'], 5)} | {f(r['D_entry'], 1)} / {f(r['control_D_entry'], 1)} | {f(r['D_final'], 1)} / {f(r['control_D_final'], 1)} | {r['native_commits']} / {r['control_native_commits']} | {f(r['native_seconds'], 1)} / {f(r['control_native_seconds'], 1)} | "
            f"{f(r['hd95'], 3)} / {f(r['control_hd95'], 3)} | {f(r['sdlogj_official'])} / {f(r['control_sdlogj_official'])} | {f(r['sdlogj_brain'], 3)} / {f(r['control_sdlogj_brain'], 3)} | {r['certificate']} | {f(r['a5pv2_seg35'])} / {f(r['a5pv2_seg4'])} |")


def summary_text(result):
    d = result['described']
    one, two, k1 = result['reading_one'], result['reading_two'], result['reading_s_k1']
    names = dict(rest_14='其余 14 对', all_19='19 对', first_5='前 5 对')
    lines = ['# SEED19 汇总（代码写出，读数不是结论）', '', result['note'], '', f"**读数一（其余 14 对）**：{one['text']}", '', f"**读数二（BZ 的目标，19 对）**：{two['text']}", '', f".88（只描述，不判）：{'19 对 seg35 差 '+f(two['stretch_gap']) if two['code'] != NOT_JUDGED else '有缺对，不算'}。", '', f"**S_K1**：{k1['text']}", '', f"结果根目录 {result['root']}；写出 {result['written_utc']}。前 5 对的 S_P 只读 {batch.FIRST_ROOT}。", '']
    if one['excluded']:
        lines += ['没有计入读数一的对：', '']+[f"- {p}：{'；'.join(r)}" for p, r in one['excluded'].items()]+['']
    if two['missing']:
        lines += ['读数二的缺对：', '']+[f'- {p}：{r}' for p, r in two['missing'].items()]+['']
    lines += ['## 只描述：Δ = S_P − B1TM', '', '| 对集 | Δ seg35 | Δ seg4 | Δ HD95 | Δ 官方 SDlogJ | Δ 脑内 SDlogJ |', '|---|---|---|---|---|---|']
    for name, row in d['differences'].items():
        lines.append(f"| {names[name]} | {stats(row['seg35'])} | {stats(row['seg4'])} | {stats(row['hd95'], 3)} | {stats(row['sdlogj_official'])} | {stats(row['sdlogj_brain'], 3)} |")
    lines += ['', '## 只描述：均值（B1W_v2 即 A5Pv2，读 b1_v1）', '', '| 对集 | 臂 | n | seg35 | seg4 | HD95 | 官方 SDlogJ | 脑内 SDlogJ | 脑内 J < .5（%） |', '|---|---|---|---|---|---|---|---|---|']
    for name, arms in d['means'].items():
        for arm, row in arms.items():
            lines.append(f"| {names[name]} | {arm} | {row['n']} | {f(row['seg35'])} | {f(row['seg4'])} | {f(row['hd95'], 3)} | {f(row['sdlogj_official'])} | {f(row['sdlogj_brain'], 3)} | {f(row['pct_J_lt_half_brain'], 2)} |")
    lines += ['', 'S_P − A5Pv2（只算差）：']+[f"- {names[name]}：seg35 {stats(row['seg35'])}；seg4 {stats(row['seg4'])}" for name, row in d['against_a5pv2'].items()]
    lines += ['', '## 只描述：分组（小六个、大六个、其余 23 个是 seg35 的标签组；皮层、皮质下、白质、CSF 是 seg4）', '', '| 对集 | 臂 | 小六个 | 大六个 | 其余 23 个 | 皮层 | 皮质下 | 白质 | CSF |', '|---|---|---|---|---|---|---|---|---|']
    for name, arms in d['groups'].items():
        for arm, row in arms.items():
            lines.append(f"| {names[name]} | {arm} | {f(row['small_six'])} | {f(row['large_six'])} | {f(row['other_23'])} | {f(row['cortex'])} | {f(row['subcortical'])} | {f(row['white_matter'])} | {f(row['csf'])} |")
    lines += ['', '分组的差（S_P − B1TM，均值，为正对数）：']
    for name, row in d['group_differences'].items():
        lines.append(f'- {names[name]}：'+'；'.join(f"{g} {f(row[g].get('mean_diff'))}（{row[g].get('positive')}/{row[g].get('n')}）" for g in ('small_six', 'large_six', 'other_23', 'cortex', 'subcortical', 'white_matter', 'csf')))
    header = '| 对 | 批 | S_P seg35 / seg4 | B1TM seg35 / seg4 | Δ seg35 | Δ seg4 | 入口 seg35 / seg4（接受 α；括号 α = 1） | 转移 α S_P / B1TM | 入口 D S_P / B1TM | 终态 D S_P / B1TM | 原生提交 S_P / B1TM | 原生层 s S_P / B1TM | HD95 S_P / B1TM | 官方 SDlogJ S_P / B1TM | 脑内 SDlogJ S_P / B1TM | 证书 | A5Pv2 seg35 / seg4 |'
    rule = '|'+'---|'*17
    lines += ['', '## 只描述：逐对', '', header, rule]+[pair_row(r) for r in d['rows']]
    lines += ['', '## 只描述：0440 的两对（0440 是分割最差的受试者，.8638 / .9062，runs/20261103-n0）', '', header, rule]+[pair_row(r) for r in d['subject_0440']]
    if result['described_s_k1']:
        lines += ['', '## 只描述：S_K1（K = 1，600 步的入口，前 5 对；S_P 读 seed_v1）', '', '| 对 | S_K1 seg35 / seg4 | S_P seg35 / seg4 | B1TM seg35 / seg4 | 入口 seg35 / seg4（接受 α；括号 α = 1） | 转移 α | 入口 D | 终态 D | 原生提交 | 原生层 s | HD95 | 官方 SDlogJ | 证书 |', '|'+'---|'*13]
        for r in result['described_s_k1']:
            e = r['entry']
            lines.append(f"| {r['pair']} | {f(r['seg35'])} / {f(r['seg4'])} | {f(r['s_p_seg35'])} / {f(r['s_p_seg4'])} | {f(r['control_seg35'])} / {f(r['control_seg4'])} | {f(e['seg35'])} / {f(e['seg4'])}（α = 1：{f(e['seg35_alpha1'])} / {f(e['seg4_alpha1'])}） | {f(r['alpha'], 5)} | {f(r['D_entry'], 1)} | {f(r['D_final'], 1)} | {r['native_commits']} | {f(r['native_seconds'], 1)} | {f(r['hd95'], 3)} | {f(r['sdlogj_official'])} | {r['certificate']} |")
    lines += ['', '## 异常', '']+([f"- {a['arm']} {a['pair']}: {a['what']}" for a in result['anomalies']] or ['没有。'])
    if result['reference_problems'] or result['first5_problems']:
        lines += ['', '## 基线问题', '']+[f'- {p}' for p in result['reference_problems']+result['first5_problems']]
    return '\n'.join(lines)+'\n'


FORBIDDEN_ROOTS = (batch.FIRST_ROOT, core.RETRIED_ROOT, core.RETRY_ROOT, batch.ENTRY_ROOT)


def write_analysis(root, results=None):
    """Reads the runs, seed_v1, the entries' records and b1_v1, writes the three files under root/analysis; an earlier summary is moved to analysis/previous_<UTC>/.  Refuses a root under seed_v1 or the entries' batches.  Returns the decision."""
    root = Path(root)
    results = Path(core.RESULTS if results is None else results)
    for name in FORBIDDEN_ROOTS:
        if (results/name).resolve() == root.resolve() or (results/name).resolve() in root.resolve().parents:
            raise SystemExit(f'{root}: under {name}, which this reading only reads; nothing written')
    ev = core.load_evaluate()
    b1_root = results/'b1_v1'
    pairs = set(batch.FIRST_PAIRS+batch.REST_PAIRS)
    refs = ev.long_csv_runs(b1_root, ('B1TM', 'B1W_v2'), pairs)
    problems = list(ev.attach_reference_labels({k: v for k, v in refs.items() if k[0] == 'B1TM'}, b1_root, ('B1TM',)))+control_configuration_problems(results)
    runs = load_runs(root, results)
    own = dict(runs)
    runs.update(refs)
    pin_bad, seen = pin_problems(runs)
    problems += pin_bad
    first5 = first5_problems(runs)
    result = dict(schema=SCHEMA, root=str(root), written_utc=core.utc(), note=NOTE, rule='runs/20261104-seed19/plan.md', reference_problems=problems, first5_problems=first5, pins=seen, first5_pin=FIRST5_PIN,
                  sources=dict(controls=str(b1_root/'analysis/long.csv'), controls_sha256=core.sha_file(b1_root/'analysis/long.csv'), first_5=str(results/batch.FIRST_ROOT), evaluator_sha256=core.sha_file(core.ROOT/'experiments/d0/evaluate.py'), code_sha256=core.sha_file(Path(__file__))),
                  reading_one=reading_one(runs, problems, results), reading_two=reading_two(runs, first5, results), reading_s_k1=reading_s_k1(runs, root, first5, results))
    result['described'] = describe(root, runs, ev, results)
    result['described_s_k1'] = describe_s_k1(root, runs, results)
    result['anomalies'] = anomalies(runs, ev, results)
    out = root/'analysis'
    out.mkdir(parents=True, exist_ok=True)
    moved = ev._set_aside(out, ('long.csv', 'summary_seed19.md', 'decision_seed19.json'))
    result['previous_summary_moved_to'] = None if moved is None else str(moved)
    first.long_csv(out/'long.csv', own, refs, ev)
    core.write_json(out/'decision_seed19.json', result)
    (out/'summary_seed19.md').write_text(summary_text(result), encoding='utf-8')
    return result
