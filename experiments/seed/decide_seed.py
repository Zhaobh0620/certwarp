"""SEED's reading (runs/20261104-seed/plan.md, by code; a label for the lead and BZ, not a conclusion): analysis/decision_seed.json, long.csv, summary_seed.md.

Delta = S_P - B1TM, pair by pair, the first 5 development pairs; the unrounded means.  Premises: all 5 pairs of S_P have class ok, at least one native commit, a final map with a passed certificate (and an entry injected as recorded, with the plan's configuration),
mean official SDlogJ <= .10; B1TM has the same 5 pairs, certified, from the plan's configuration, its means equal to the pins.  (Plan revision 1: a run without a native commit did not pass the native level, so the sentence of the first reading would be false.)  Then, in this order:
  premises not met                                not decidable, the reasons written
  Delta(seg35) >= +.01 and 5/5 positive           the entry's difference is still there after the native level
  Delta(seg35) <= +.005 or fewer than 4/5 positive    no difference visible after the native level
  otherwise                                       middle band (MDE of n = 5 about .009-.020)
Only described, never read: S_T - B1W_A5 (oracle), seg4, the groups, entry -> final, native commits, alpha, D at the entry, SDlogJ, the line of .86 on the first 5 pairs.

The evaluation's functions come from experiments/d0/evaluate.py (read only).  The controls' numbers come from b1_v1/analysis/long.csv, their per-label Dice from the metrics.json of the same run_id (evaluate.py attach_reference_labels checks they agree).
"""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import numpy as np

import run_seed as core

SCHEMA = 'bayesreg.seed.decision.v1'
N = 5
SDLOGJ_MAX, SEE_BAR, NOT_SEE_BAR, MIN_POSITIVE = .10, .01, .005, 4
PINS = dict(B1TM=(.8341, .8947), B1W_A5=(.8853, .9389))     # (seg35, seg4) of b1_v1 over these 5 pairs, to 4 decimals (b1_v1/analysis/long.csv: .83410, .89467; .88527, .93894)
PIN_TOL = 5.01e-5
NEAR_BAR = .001                                              # a sentence only: the native level of S_P is about 45 s shorter than B1TM's (review M3)
TARGET_19, B1TM_19_PIN = .86, .8468                          # BZ's target on the 19 pairs; B1TM on them (.84682)
PERSISTS, NO_DIFFERENCE, MIDDLE, NOT_DECIDABLE = 'persists', 'no_difference', 'middle', 'not_decidable'
READINGS = {PERSISTS: '入口的差别过了原生层还在', NO_DIFFERENCE: '过了原生层看不到入口的差别', MIDDLE: '中间带（n = 5 的 MDE 约 .009–.020）', NOT_DECIDABLE: '不可判'}
MEASURES = ('seg35', 'seg4', 'hd95', 'sdlogj_official', 'sdlogj_brain', 'pct_J_lt_half_brain')
NOTE = ('开发集前 5 对，单次观测，读数不是结论，下一步交 BZ。这 5 对是之前选配用过的对子，n = 5 只有方向信息（MDE 约 .009–.020）。S_P 用预测标签的入口与 U-Net 预测上色，分割器用训练集标签训练；S_T 用真值（oracle，只作上限）。'
        '路径改变不改目标函数，原生层仍在优化标量 D：阴性不说明多通道求解器没用；阳性不等于 19 对，不等于留出集，要进方法先改合同（C12-4a 不覆盖这种入口）。入口计算约 47 s 计入同一个 330 s 预算（冻结预算没有 285 s，见 plan.md），'
        'S_P 的原生层约 275–279 s〔推断，运行前的估计〕，B1TM 约 320–324 s，所以 no_difference 含少跑约 45 s 的影响，幅度〔推断〕约不到 .001；预算收尾不是收敛。基线不是同时段（b1_v1，2026-11-04）。')


def load_runs(root):
    """{(arm, pair): metrics.json} of this batch."""
    out = {}
    for arm, pair in core.plan_runs():
        m = core.read_json(core.run_dir(root, arm, pair)/'metrics.json')
        if m is not None:
            out[(arm, pair)] = m
    return out


def final_ok(run):
    """A final map counts only with a passed certificate (not as in MC0, whose fields are uncertified).  As in B1 the evaluation does not certify again (evaluate.py definitions, certificate): it reads the run's own certificate of the
    exported state and binds it to the map: `passed` holds state_hash == sha256 of the exported u; final_npz_sha256_matches is the sha256 of final_native.npz against final_native.json."""
    certificate = (run or {}).get('certificate') or {}
    return bool(run and run.get('metrics') and certificate.get('passed') and certificate.get('state_hash_matches_u') is True and certificate.get('final_npz_sha256_matches') is True)


def plan_configuration(arm, results=None):
    return dict(seconds=core.SECONDS, changes=core.CHANGES, inputs_root=str(core.inputs_root(arm, results)))


def control_configuration_problems(results, arm):
    """The control's records against the plan: same budget, same options, same inputs root."""
    bad = []
    control = core.ARMS[arm]['control']
    for pair in core.PAIRS:
        m = core.read_json(Path(results)/'b1_v1/runs'/control/pair/'metrics.json') or {}
        b1 = m.get('b1') or {}
        have = dict(seconds=b1.get('seconds'), changes=b1.get('changes'), inputs_root=b1.get('inputs_root'))
        if have != plan_configuration(arm, results):
            bad.append(f'{control} {core.short_name(pair)}: configuration {have}')
    return bad


def arm_premises(runs, arm, results=None):
    """The reasons this arm cannot be read (empty: it can)."""
    reasons = []
    for pair in core.PAIRS:
        short = core.short_name(pair)
        m = runs.get((arm, pair))
        if m is None:
            reasons.append(f'{arm} {short}: 没有 metrics.json')
            continue
        if not m.get('metrics'):
            reasons.append(f'{arm} {short}: 无终态场（{(m.get("classification") or {}).get("kind")}）')
            continue
        cls = (m.get('classification') or {}).get('class')
        if cls != 'ok':
            reasons.append(f'{arm} {short}: 分类 {cls}，不算过了原生层')
        native = m.get('native_commits')
        if isinstance(native, bool) or not isinstance(native, int) or native < 1:
            reasons.append(f'{arm} {short}: 原生提交 {native}，没有过原生层')
        if not final_ok(m):
            reasons.append(f'{arm} {short}: 终态证书未通过，或没有与导出场绑定（state_hash、npz sha256）')
        seed = m.get('seed') or {}
        if seed.get('passed') is not True:
            reasons.append(f'{arm} {short}: 入口核对未通过 {seed.get("problems")}')
        have = seed.get('arm') or {}
        want = plan_configuration(arm, results)
        if {k: have.get(k) for k in want} != want or have.get('charge_seconds') != core.ENTRY_CHARGE_SECONDS:
            reasons.append(f'{arm} {short}: 配置与计划不同 {have}')
    finals = [runs[(arm, p)]['metrics']['sdlogj_official'] for p in core.PAIRS if (arm, p) in runs and runs[(arm, p)].get('metrics')]
    if len(finals) == N and float(np.mean(finals)) > SDLOGJ_MAX:
        reasons.append(f'{arm}: 官方 SDlogJ 均值 {float(np.mean(finals)):.4f} > {SDLOGJ_MAX}')
    return reasons


def paired(runs, arm, ref, key='seg35'):
    """Per pair arm - ref; the mean over all 5, or None if a value is missing."""
    values = {}
    for pair in core.PAIRS:
        a = (runs.get((arm, pair)) or {}).get('metrics')
        b = (runs.get((ref, pair)) or {}).get('metrics')
        if a is not None and b is not None:
            values[pair] = a[key]-b[key]
    complete = len(values) == N
    return dict(per_pair={core.short_name(p): v for p, v in values.items()}, mean=float(np.mean(list(values.values()))) if complete else None,
                median=float(statistics.median(values.values())) if complete else None, positive=sum(v > 0 for v in values.values()), n=len(values))


def decide(runs, reference_problems=(), results=None):
    """The plan's table.  Unrounded means; the bars included ('>= +.01', '<= +.005'); positive is strictly > 0."""
    reasons = list(reference_problems)+arm_premises(runs, 'S_P', results)
    for pair in core.PAIRS:
        control = runs.get(('B1TM', pair))
        if control is None or not control.get('metrics') or not (control.get('certificate') or {}).get('passed'):
            reasons.append(f'B1TM {core.short_name(pair)}: 基线没有通过证书的终态场')
    delta = paired(runs, 'S_P', 'B1TM')
    if reasons or delta['mean'] is None:
        code = NOT_DECIDABLE
    elif delta['mean'] >= SEE_BAR and delta['positive'] == N:
        code = PERSISTS
    elif delta['mean'] <= NOT_SEE_BAR or delta['positive'] < MIN_POSITIVE:
        code = NO_DIFFERENCE
    else:
        code = MIDDLE
    near = None if delta['mean'] is None else next((bar for bar in (NOT_SEE_BAR, SEE_BAR) if abs(delta['mean']-bar) < NEAR_BAR), None)
    text = READINGS[code]+('：'+'；'.join(reasons) if reasons else f"：Δ(seg35) {delta['mean']:+.4f}，{delta['positive']}/{N} 为正")
    if near is not None and not reasons:                 # a sentence for the reader, the code stays what the table says (plan revision 1, review M3)
        text += f'；Δ 贴着 {near:g} 这条线（差不到 {NEAR_BAR:g}），S_P 的原生层比 B1TM 少约 45 s，这部分的影响〔推断〕约不到 .001，要带着读'
    return dict(reading_code=code, reading=text, reasons=reasons, near_bar=near, delta_seg35=delta, delta_seg4=paired(runs, 'S_P', 'B1TM', 'seg4'))


def mean_of(runs, arm, key):
    values = [(runs.get((arm, p)) or {}).get('metrics') for p in core.PAIRS]
    values = [v[key] for v in values if v is not None and v.get(key) is not None]
    return float(np.mean(values)) if values else None


def pin_problems(runs):
    """The controls' means against the plan's pins: a different number means a different baseline from the one the plan read.  Per control: B1W_A5's stay with the oracle arm (plan revision 1, review M4)."""
    bad, seen = {}, {}
    for arm, (seg35, seg4) in PINS.items():
        a, b = mean_of(runs, arm, 'seg35'), mean_of(runs, arm, 'seg4')
        seen[arm] = (None if a is None else round(a, 5), None if b is None else round(b, 5))
        bad[arm] = [f'{arm}: 前 5 对 seg35 {a}, seg4 {b}，计划钉住 {seg35}, {seg4}'] if a is None or b is None or abs(a-seg35) > PIN_TOL or abs(b-seg4) > PIN_TOL else []
    return bad, seen


def entry_rows(results):
    """{(arm, pair): {seg35, seg4, accepted_alpha}} of the entries, from the long.csv of their batch: alpha as the field stands, the numbers the evaluation wrote (field after the taper, alpha = 1)."""
    out = {}
    for arm, spec in core.ARMS.items():
        roots = {}
        for pair in core.PAIRS:
            roots.setdefault(core.entry_root(arm, pair), set()).add(pair)     # each pair from the batch of its entry, nothing else (plan revision 2)
        for root, pairs in roots.items():
            path = Path(results)/root/'analysis/long.csv'
            if not path.is_file():
                continue
            with open(path, newline='') as f:
                for row in csv.DictReader(f):
                    if row['arm'] == spec['source_arm'] and row['tier'] == core.ENTRY_TIER and row['pair'] in pairs and row['measure'] in ('seg35', 'seg4', 'sdlogj_official'):
                        rec = out.setdefault((arm, row['pair']), dict(run_id=row['run_id'], accepted_alpha=row.get('accepted_alpha')))
                        rec[row['measure']] = float(row['value'])
    return out


def describe(runs, entries, ev):
    """The tables that decide nothing."""
    pairs = list(core.PAIRS)
    arms = ('S_P', 'B1TM', 'S_T', 'B1W_A5')
    means = {arm: {key: mean_of(runs, arm, key) for key in MEASURES} for arm in arms}
    groups = {arm: dict({g: ev.a5_group_mean(runs, arm, labels, pairs) for g, labels in ev.A5_GROUPS.items()},
                        cortex=ev.label_rows(runs, arm, arm, 'seg4', (1,), pairs)[1]['arm_mean'], white_matter=ev.label_rows(runs, arm, arm, 'seg4', (3,), pairs)[3]['arm_mean']) for arm in arms}
    differences = {}
    for arm, ref in (('S_P', 'B1TM'), ('S_T', 'B1W_A5')):
        structures = ev.label_rows(runs, arm, ref, 'seg4', (1, 3), pairs)
        differences[f'{arm}-{ref}'] = dict(seg35=paired(runs, arm, ref), seg4=paired(runs, arm, ref, 'seg4'), sdlogj_official=paired(runs, arm, ref, 'sdlogj_official'), hd95=paired(runs, arm, ref, 'hd95'),
                                           groups={g: ev.a5_group_row(runs, arm, ref, labels, pairs) for g, labels in ev.A5_GROUPS.items()}, cortex=structures[1], white_matter=structures[3])
    rows = []
    for arm in ('S_P', 'S_T'):
        control = core.ARMS[arm]['control']
        for pair in pairs:
            m, c, e = runs.get((arm, pair)) or {}, runs.get((control, pair)) or {}, entries.get((arm, pair)) or {}
            x, seed = m.get('metrics') or {}, m.get('seed') or {}
            e = dict(seed.get('entry_metrics') or {}, **e)      # the evaluation's own record of the entry, when the batch has no long.csv yet
            first = next((a for a in seed.get('transfer_attempts') or [] if a.get('passed')), {})
            rows.append(dict(arm=arm, pair=core.short_name(pair), oracle=core.ARMS[arm]['oracle'], run_id=m.get('run_id'), cls=(m.get('classification') or {}).get('class'), status=m.get('status'),
                             entry_seg35=e.get('seg35'), entry_seg4=e.get('seg4'), entry_sdlogj=e.get('sdlogj_official'), entry_alpha_in_evaluation=e.get('accepted_alpha'), seg35=x.get('seg35'), seg4=x.get('seg4'), hd95=x.get('hd95'),
                             sdlogj_official=x.get('sdlogj_official'), sdlogj_brain=x.get('sdlogj_brain'), native_commits=m.get('native_commits'), alpha=seed.get('accepted_alpha'), n_attempts=len(seed.get('transfer_attempts') or []), min_jd_at_alpha=first.get('min_jd_lower_bound'),
                             D_at_entry=seed.get('entry_D_at_accepted_alpha'), last_D=seed.get('last_D'), certificate=(m.get('certificate') or {}).get('passed'), self_check=(m.get('self_check') or {}).get('passed'),
                             min_certified_bound=(m.get('certificate') or {}).get('min_jd_lower_bound'), e2e=m.get('e2e_seconds'), control_seg35=(c.get('metrics') or {}).get('seg35'), control_seg4=(c.get('metrics') or {}).get('seg4'), control_native_commits=c.get('native_commits')))
    return dict(means=means, groups=groups, differences=differences, rows=rows)


def line_86(runs, refs_19):
    """The line '.86 on the first 5 pairs': B1TM's first 5 lie below its 19 pairs by the offset; a model, not a bar."""
    values = [r['metrics']['seg35'] for r in refs_19.values() if r.get('metrics')]
    all19 = float(np.mean(values)) if len(values) == 19 else None
    first = mean_of(runs, 'B1TM', 'seg35')
    offset = None if all19 is None or first is None else all19-first
    line = None if offset is None else TARGET_19-offset
    arm = mean_of(runs, 'S_P', 'seg35')
    return dict(b1tm_19=all19, b1tm_first5=first, offset=offset, first5_line=line, s_p_first5=arm, s_p_at_or_above=None if arm is None or line is None else arm >= line,
                role='〔模型〕S_P 前 5 对到这条线时才可能在 19 对上到 .86（假定 S_P 与 B1TM 从前 5 对到 19 对的偏移相同）；这不是判据')


def anomalies(runs, ev):
    keys = [(arm, p) for arm in core.ARMS for p in core.PAIRS]
    out = ev.b1_anomalies(runs, keys)
    for arm, pair in keys:
        seed = (runs.get((arm, pair)) or {}).get('seed') or {}
        e = seed.get('entry_metrics') or {}
        if seed.get('accepted_alpha') is not None and e.get('accepted_alpha') is not None and seed['accepted_alpha'] != e['accepted_alpha']:
            out.append(dict(arm=arm, pair=pair, what=f"转移 α {seed['accepted_alpha']} 与入口评测的 α {e['accepted_alpha']} 不同"))
    return out


def f(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def summary_text(result):
    d = result['described']
    lines = ['# SEED 汇总（代码写出，读数不是结论）', '', result['note'], '', f"**读数**：{result['reading']}", '', f"结果根目录 {result['root']}；写出 {result['written_utc']}。", '', '## 判定（Δ = S_P − B1TM，前 5 对）', '',
             '| 对 | S_P seg35 | B1TM seg35 | Δ seg35 | S_P seg4 | B1TM seg4 | Δ seg4 |', '|---|---|---|---|---|---|---|']
    delta35, delta4 = result['delta_seg35']['per_pair'], result['delta_seg4']['per_pair']
    for row in (r for r in d['rows'] if r['arm'] == 'S_P'):
        lines.append(f"| {row['pair']} | {f(row['seg35'])} | {f(row['control_seg35'])} | {f(delta35.get(row['pair']))} | {f(row['seg4'])} | {f(row['control_seg4'])} | {f(delta4.get(row['pair']))} |")
    lines += ['', f"均值 Δ seg35 {f(result['delta_seg35']['mean'])}（{result['delta_seg35']['positive']}/{result['delta_seg35']['n']} 为正），Δ seg4 {f(result['delta_seg4']['mean'])}（{result['delta_seg4']['positive']}/{result['delta_seg4']['n']} 为正）。", '']
    line = result['line_86']
    lines += [f"线（{line['role']}）：B1TM 19 对 {f(line['b1tm_19'])}，前 5 对 {f(line['b1tm_first5'])}，差 {f(line['offset'])}，前 5 对的线 {f(line['first5_line'])}；S_P 前 5 对 {f(line['s_p_first5'])}。", '',
              '## 只描述：各臂均值（S_T、B1W_A5 用真值，oracle，只作上限）', '', '| 臂 | seg35 | seg4 | HD95 | 官方 SDlogJ | 脑内 SDlogJ | 脑内 J < .5（%） |', '|---|---|---|---|---|---|---|']
    names = dict(S_P='S_P', B1TM='B1TM', S_T='S_T（oracle，只作上限）', B1W_A5='B1W_A5（oracle，只作上限）')
    for arm, row in d['means'].items():
        lines.append(f"| {names[arm]} | {f(row['seg35'])} | {f(row['seg4'])} | {f(row['hd95'], 3)} | {f(row['sdlogj_official'])} | {f(row['sdlogj_brain'], 3)} | {f(row['pct_J_lt_half_brain'], 2)} |")
    lines += ['', '## 只描述：分组', '', '| 臂 | 小六个 | 大六个 | 其余 23 个 | 皮层（seg4） | 白质（seg4） |', '|---|---|---|---|---|---|']
    for arm, row in d['groups'].items():
        lines.append(f"| {names[arm]} | {f(row['small_six'])} | {f(row['large_six'])} | {f(row['other_23'])} | {f(row['cortex'])} | {f(row['white_matter'])} |")
    lines += ['', '## 只描述：逐对（入口 → 终态）', '', '| 臂 | 对 | 入口 seg35 / seg4 | 终态 seg35 / seg4 | 转移 α | D（入口，α 处） | 原生提交（对照） | 官方 SDlogJ | 最小认证下界 | 证书 | e2e s |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in d['rows']:
        lines.append(f"| {r['arm']} | {r['pair']} | {f(r['entry_seg35'])} / {f(r['entry_seg4'])} | {f(r['seg35'])} / {f(r['seg4'])} | {f(r['alpha'], 5)} | {f(r['D_at_entry'], 1)} | {r['native_commits']}（{r['control_native_commits']}） | {f(r['sdlogj_official'])} | {r['min_certified_bound']} | {r['certificate']} | {f(r['e2e'], 1)} |")
    lines += ['', '## 异常', '']
    lines += [f"- {a['arm']} {a['pair']}: {a['what']}" for a in result['anomalies']] or ['没有。']
    if result['reference_problems'] or result['oracle_only']['reference_problems']:
        lines += ['', '## 基线问题', ''] + [f'- {p}' for p in result['reference_problems']] + [f'- （oracle，只作上限，不进判定）{p}' for p in result['oracle_only']['reference_problems']]
    return '\n'.join(lines)+'\n'


def long_csv(path, own, refs, ev):
    with open(path, 'w', newline='') as out:
        w = csv.DictWriter(out, fieldnames=['set', 'split', 'method', 'arm', 'pair', 'measure', 'value', 'run_id'])
        w.writeheader()
        w.writerows(dict(r, method=own[(r['arm'], r['pair'])].get('method', r['method'])) for r in ev.long_rows(own))
        for rec in refs.values():
            w.writerows(rec.get('rows', []))


def write_analysis(root, results=None, read_entries=True):
    """Reads the runs and the controls, writes the three files; an earlier summary is moved to analysis/previous_<UTC>/.  Returns the decision."""
    root = Path(root)
    results = Path(core.RESULTS if results is None else results)
    ev = core.load_evaluate()
    pairs = set(core.PAIRS)
    b1_root = results/'b1_v1'
    arms = tuple(spec['control'] for spec in core.ARMS.values())
    refs = ev.long_csv_runs(b1_root, arms, pairs)
    problems, oracle_problems = [], []
    for arm, into in (('S_P', problems), ('S_T', oracle_problems)):             # S_T does not enter the reading: its control's problems do not either
        control = core.ARMS[arm]['control']
        into += list(ev.attach_reference_labels({k: v for k, v in refs.items() if k[0] == control}, b1_root, (control,)))+control_configuration_problems(results, arm)
    runs = load_runs(root)
    own = dict(runs)
    runs.update(refs)
    pin_bad, seen = pin_problems(runs)
    problems += pin_bad['B1TM']
    oracle_problems += pin_bad['B1W_A5']
    all19 = ev.long_csv_runs(b1_root, ('B1TM',), {r['pair'] for r in _selection()})
    result = dict(schema=SCHEMA, root=str(root), written_utc=core.utc(), note=NOTE, rule='runs/20261104-seed/plan.md', reference_problems=problems, pins={k: dict(seen=seen[k], pinned=PINS[k]) for k in PINS},
                  sources=dict(controls=str(b1_root/'analysis/long.csv'), controls_sha256=core.sha_file(b1_root/'analysis/long.csv'), evaluator_sha256=core.sha_file(core.ROOT/'experiments/d0/evaluate.py'), code_sha256=core.sha_file(Path(__file__))),
                  **decide(runs, problems, results))
    entries = entry_rows(results) if read_entries else {}
    result['line_86'] = line_86(runs, all19)
    result['described'] = describe(runs, entries, ev)
    result['oracle_only'] = dict(role='S_T − B1W_A5：用真值，只作上限，不进判定', premises=arm_premises(runs, 'S_T', results), reference_problems=oracle_problems, seg35=paired(runs, 'S_T', 'B1W_A5'), seg4=paired(runs, 'S_T', 'B1W_A5', 'seg4'))
    result['anomalies'] = anomalies(runs, ev)
    out = root/'analysis'
    out.mkdir(parents=True, exist_ok=True)
    moved = ev._set_aside(out, ('long.csv', 'summary_seed.md', 'decision_seed.json'))
    result['previous_summary_moved_to'] = None if moved is None else str(moved)
    long_csv(out/'long.csv', own, refs, ev)
    core.write_json(out/'decision_seed.json', result)
    (out/'summary_seed.md').write_text(summary_text(result))
    return result


def _selection():
    return json.loads((core.MAIN_REPO/'data/three_d/stage_L2R_val_v1/selection.json').read_text())['pairs']
