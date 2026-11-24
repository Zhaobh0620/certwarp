"""R2's segmentation-level reading (runs/20261110-r2/plan.md section 4; by code; a label for the lead and BZ, not a conclusion).  Round 2's first segmentation-level reading (of 2).

One arm (the lead's ruling of 11-10 ~10:42 PM CST: r2s only): the primary arm is R2S.  Delta = R2S - UF4 (both from this evaluation, the same code path), per subject.
    过门        mean Delta seg35 >= +.005, at least 15 of 20 subjects positive, and mean Delta seg4 >= -.001  ->  the 19-pair registration reading of R2S (another release, the lead starts it)
    不过门      otherwise  ->  stop, hand to BZ; no other configuration is tried
    不可判      no evaluation, not 20 subjects, a member missing or not the plan's, the control of UF4 not passed (|mean| > 2e-4, or other truth files), a mean undefined
The selection that remains is the one of T and of the recipe on these same 19 pairs / 20 subjects (written in the plan, section 4.3).  No option turns a reading into 不可判 after the fact.
    python -B experiments/r2/decide_r2.py --eval DIR/eval_r2.json --out-dir DIR/analysis
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE)]
import r2_common as rc  # noqa: E402

SCHEMA = 'bayesreg.r2.decision.v1'
PASS, NOT_PASS, UNDECIDABLE = 'pass', 'not_pass', 'undecidable'
READINGS = {PASS: '过门：做 19 对配准读数（第二轮配准读数第 1 次）', NOT_PASS: '不过门：停，交 BZ，不换配置重试', UNDECIDABLE: '不可判'}
DESCRIBED = ('seg35', 'seg4', 'small_six', 'big_six', 'other_23', 'cortex', 'white_matter')
NOTE = ('开发集 20 名受试者，单次观测，分割级读数，不是配准结果，不是留出集结果，不是对外比较。R2 的成员只用训练集（每折 98 + 8 人）的标签训练。这 20 人已被反复读过；目标标签 T 是在同一批 19 对、20 人的逐标签误差上选出来的，选择效应在内（偏乐观）。单臂：只训 r2s（总负责人 11-10 约 10:42 PM 的裁决）。'
        '这是开发第二轮分割级读数的第 1 次（共 2 次，BZ 2026-11-10）。')


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def column(evaluation, arm, metric):
    rows = evaluation['rows'][arm]
    return np.array([np.nan if rows[s].get(metric) is None else rows[s][metric] for s in evaluation['subjects']], dtype=np.float64)


def describe(x, *, draws=rc.SEG_BOOT_DRAWS, seed=rc.SEG_BOOT_SEED):
    """Mean, median, counts, the t interval and the subject bootstrap (95 %): decide_uf.describe_difference's."""
    from scipy import stats
    x = np.asarray(x, dtype=np.float64)
    ok = x[np.isfinite(x)]
    n = int(ok.size)
    out = dict(mean=float(ok.mean()) if n else None, median=float(np.median(ok)) if n else None, n=n, n_undefined=int(x.size-n), n_positive=int((ok > 0).sum()), n_negative=int((ok < 0).sum()), n_zero=int((ok == 0).sum()))
    if n >= 2:
        half = float(stats.t.ppf(.975, n-1)*ok.std(ddof=1)/math.sqrt(n))
        rng = np.random.default_rng(seed)
        boot = ok[rng.integers(0, n, (draws, n))].mean(axis=1)
        out.update(t95=[out['mean']-half, out['mean']+half], se=float(ok.std(ddof=1)/math.sqrt(n)), bootstrap95=[float(v) for v in np.percentile(boot, [2.5, 97.5])], bootstrap_draws=draws, bootstrap_seed=seed)
    return out


def undecidable_reasons(evaluation):
    if evaluation is None:
        return ['没有评测结果']
    out = []
    if len(evaluation.get('subjects') or []) != rc.N_SUBJECTS:
        out.append(f"{len(evaluation.get('subjects') or [])} 名受试者，不是 {rc.N_SUBJECTS}")
    want = sorted([f'uf/{n}' for n in rc.UF_ORDER]+[f'{r}/{n}' for r in rc.RECIPE_ORDER for n in rc.MEMBER_NAMES])
    if sorted(evaluation.get('members') or {}) != want:
        out.append(f"成员缺失：{sorted(set(want)-set(evaluation.get('members') or {}))}")
    out += [f'成员不是计划训的：{p}' for p in evaluation.get('member_problems') or []]
    control = evaluation.get('control') or {}
    if control.get('tolerance') != rc.CONTROL_TOL_FORWARD:
        out.append(f"评测记下的控制容差 {control.get('tolerance')} 不是 {rc.CONTROL_TOL_FORWARD}")
    for metric in ('seg35', 'seg4'):
        d = (control.get('differences') or {}).get(metric)
        if not finite(d):
            out.append(f'控制缺失：UF4 的 {metric}')
        elif abs(d) > rc.CONTROL_TOL_FORWARD:
            out.append(f'控制不符：UF4 的 {metric} 与 eval_uf.json 差 {d:+.3e}，超过 {rc.CONTROL_TOL_FORWARD:g}')
    if control.get('truth_same') is not True:
        out.append('控制不符：真值文件不是 eval_uf.json 读过的那些')
    means = evaluation.get('means') or {}
    for arm in ('UF4',)+rc.CANDIDATE_ORDER:
        for metric in ('seg35', 'seg4'):
            if not finite((means.get(arm) or {}).get(metric)):
                out.append(f'{arm} 的 {metric} 没有定义')
    return out


def primary_arm(means):
    """One arm: R2S (the lead's ruling); the means are not compared with any other arm."""
    return rc.CANDIDATE_ORDER[0]


def reading_of(d35, d4):
    """The gate on the primary's paired differences (described): mean seg35 >= +.005, positives >= 15, mean seg4 >= -.001 (the bars included)."""
    if not (finite(d35.get('mean')) and finite(d4.get('mean'))):
        return UNDECIDABLE, ['Δ 没有定义']
    why = [w for w, ok in ((f"Δseg35 均值 {d35['mean']:+.5f} < +{rc.SEG_BAR}", d35['mean'] >= rc.SEG_BAR), (f"为正的受试者 {d35['n_positive']}/{d35['n']} < {rc.SEG_POSITIVE_MIN}", d35['n_positive'] >= rc.SEG_POSITIVE_MIN),
                               (f"Δseg4 均值 {d4['mean']:+.5f} < {rc.SEG4_FLOOR}", d4['mean'] >= rc.SEG4_FLOOR)) if not ok]
    return (NOT_PASS, why) if why else (PASS, [])


def pair_level(evaluation, arm, shorts=rc.SHORTS, metric='seg35'):
    rows = evaluation['rows'][arm]
    values = [(rows[a][metric]+rows[b][metric])/2 for a, b in map(rc.pair_subjects, shorts) if rows[a].get(metric) is not None and rows[b].get(metric) is not None]
    return float(np.mean(values)) if len(values) == len(shorts) else None


def expectation(evaluation, primary, slope=rc.OX_SLOPE):
    """〔模型〕the registration-level expectation of the primary against HS: slope x the pair-level segmentation Delta (a pair the mean of its two subjects), 19 pairs and the first 5; slopes .84-.92 described."""
    out = dict(slope=slope, slope_range=[.84, .92], slope_source='OX 读法 2 (.890, two points); the gap analysis review: .84 (the stratified model) to .92 (v4)')
    for name, shorts in (('pairs19', rc.SHORTS), ('first5', rc.SHORTS[:5])):
        a, b = pair_level(evaluation, primary, shorts), pair_level(evaluation, 'UF4', shorts)
        d = None if a is None or b is None else a-b
        out[name] = dict(seg_level_delta=d, expected_delta=None if d is None else slope*d, expected_range=None if d is None else [.84*d, .92*d])
    out['note'] = '〔模型〕不是测量：斜率只在这条线上、v1 → v4 一小段量过；分割级 Δ 来自候选被选出的同一批 20 人，偏乐观。'
    return out


def focus(evaluation, arms):
    subjects = evaluation['subjects']
    if rc.FOCUS_SUBJECT not in subjects:
        return None
    others = [s for s in subjects if s != rc.FOCUS_SUBJECT]
    return {a: dict(subject={m: evaluation['rows'][a][rc.FOCUS_SUBJECT].get(m) for m in ('seg35', 'seg4', 'small_six')},
                    others_mean={m: float(np.nanmean(column(dict(evaluation, subjects=others), a, m))) for m in ('seg35', 'seg4', 'small_six')}) for a in arms}


def decide(evaluation):
    document = dict(schema=SCHEMA, plan=rc.plan_record(REPO), rule=dict(primary='R2S, the one arm (r2s only)', delta='R2S - UF4 (this evaluation), per subject, 20 subjects',
                                             pass_=f'mean Delta seg35 >= +{rc.SEG_BAR}, at least {rc.SEG_POSITIVE_MIN}/20 positive, mean Delta seg4 >= {rc.SEG4_FLOOR}', control_tolerance=rc.CONTROL_TOL_FORWARD),
                    note=NOTE, dev_read=(evaluation or {}).get('dev_read'), targets=(evaluation or {}).get('targets'), is_conclusion=False)
    problems = undecidable_reasons(evaluation)
    if problems:
        return dict(document, reading=UNDECIDABLE, reading_text=READINGS[UNDECIDABLE], why=problems, primary_arm=None, candidate=None)
    means = evaluation['means']
    primary = primary_arm(means)
    comparisons = {}
    for arm in rc.CANDIDATE_ORDER+('r2s_2',):
        comparisons[f'{arm}_minus_UF4'] = {m: describe(column(evaluation, arm, m)-column(evaluation, 'UF4', m)) for m in DESCRIBED}
    singles = {}
    for arm, uf_arm in (('r2s_A', 'UF_A0'), ('r2s_B', 'UF_B0')):
        theirs = (evaluation.get('uf_singles') or {}).get(uf_arm)
        mine = column(evaluation, arm, 'seg35')
        singles[arm] = dict(seg35=float(np.nanmean(mine)), small_six=float(np.nanmean(column(evaluation, arm, 'small_six'))),
                            minus_uf_namesake=None if theirs is None else describe(mine-np.array([np.nan if theirs[s].get('seg35') is None else theirs[s]['seg35'] for s in evaluation['subjects']])))
    d35, d4 = comparisons[f'{primary}_minus_UF4']['seg35'], comparisons[f'{primary}_minus_UF4']['seg4']
    reading, why = reading_of(d35, d4)
    per_label = {arm: evaluation['per_label_mean'][arm] for arm in ('UF4',)+rc.CANDIDATE_ORDER}
    targets = evaluation.get('targets') or []
    document.update(reading=reading, reading_text=READINGS[reading], why=why, primary_arm=primary, candidate=primary if reading == PASS else None, delta_seg35=d35['mean'], delta_seg4=d4['mean'],
                    n_positive=d35['n_positive'], primary_minus_uf4=comparisons[f'{primary}_minus_UF4'], comparisons=comparisons, singles=singles,
                    means={arm: {m: means[arm].get(m) for m in DESCRIBED} for arm in ('UF4',)+rc.CANDIDATE_ORDER+('r2s_2', 'r2s_A', 'r2s_B') if arm in means},
                    targets_mean={arm: float(np.nanmean([v for k, v in enumerate(per_label[arm]) if k+1 in targets and v is not None])) if targets else None for arm in per_label},
                    per_label_delta={arm: [None if a is None or b is None else a-b for a, b in zip(per_label[arm], per_label['UF4'])] for arm in rc.CANDIDATE_ORDER},
                    focus=focus(evaluation, ('UF4',)+rc.CANDIDATE_ORDER), first5_pairs={arm: pair_level(evaluation, arm, rc.SHORTS[:5]) for arm in ('UF4',)+rc.CANDIDATE_ORDER},
                    expectation=expectation(evaluation, primary), control=evaluation['control'], members={k: {kk: v.get(kk) for kk in ('stopped', 'iterations_run', 'checkpoint_iteration', 'best_iteration', 'best_calib_mean_dice35', 'checkpoint_calib_mean_dice35', 'wall_seconds', 'temperature')} for k, v in evaluation['members'].items()},
                    seconds=evaluation.get('seconds'), plan_of_evaluation=evaluation.get('plan'))
    return document


def _f(v, d=4):
    return 'n/a' if not finite(v) else f'{v:.{d}f}'


def _s(v, d=4):
    return 'n/a' if not finite(v) else f'{v:+.{d}f}'


def summary(document):
    L = ['# R2 分割级读数（decide_r2.py 写的；开发集 20 人，单次，分割级）', '', f"**读数：{document['reading_text']}**" + (f"（{'；'.join(document['why'])}）" if document.get('why') else ''), '', document['note'], '']
    if document['reading'] == UNDECIDABLE:
        return '\n'.join(L)+'\n'
    p = document['primary_arm']
    d = document['primary_minus_uf4']
    L += [f"主臂 {p}：Δseg35 {_s(d['seg35']['mean'], 5)}（{d['seg35']['n_positive']}/{d['seg35']['n']} 为正；t 95% [{_s(d['seg35']['t95'][0])}, {_s(d['seg35']['t95'][1])}]，bootstrap [{_s(d['seg35']['bootstrap95'][0])}, {_s(d['seg35']['bootstrap95'][1])}]），Δseg4 {_s(d['seg4']['mean'], 5)}。",
          f"线：Δseg35 ≥ +{rc.SEG_BAR}，≥ {rc.SEG_POSITIVE_MIN}/20 为正，Δseg4 ≥ {rc.SEG4_FLOOR}。", '', '| 臂 | seg35 | seg4 | 小六个 | 大六个 | 其余 23 个 | 皮层 | 白质 | T 的均值 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm, m in document['means'].items():
        L.append(f"| {arm} | {_f(m.get('seg35'))} | {_f(m.get('seg4'))} | {_f(m.get('small_six'))} | {_f(m.get('big_six'))} | {_f(m.get('other_23'))} | {_f(m.get('cortex'))} | {_f(m.get('white_matter'))} | {_f((document.get('targets_mean') or {}).get(arm))} |")
    e = document['expectation']
    L += ['', f"〔模型〕配准级预期（19 对，斜率 {e['slope']}）：{_s(e['pairs19']['expected_delta'])}（斜率 .84–.92：{[_s(v) for v in (e['pairs19']['expected_range'] or [])]}）。{e['note']}", '',
          f"控制：UF4 对 eval_uf.json 的均值差 {document['control']['differences']}，容差 {document['control']['tolerance']}，逐行相同 {document['control']['rows_equal']}/20。"]
    return '\n'.join(L)+'\n'


def write(out_dir, document):
    out_dir = Path(out_dir)
    rc.write_json_once(out_dir/'decision_r2.json', document)
    rc.write_text_once(out_dir/'summary_r2.md', summary(document))
    return out_dir/'decision_r2.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eval', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    evaluation = rc.read_json(args.eval)
    document = decide(evaluation)
    write(args.out_dir, document)
    print(json.dumps({k: document.get(k) for k in ('reading', 'reading_text', 'why', 'primary_arm', 'delta_seg35', 'delta_seg4', 'n_positive')}, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
