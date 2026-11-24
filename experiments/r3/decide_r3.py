"""R3's segmentation-level reading (runs/20261111-r3/plan.md section 5; by code; a label for the lead and BZ, not a conclusion).  Round 2's second and last segmentation-level reading.

One arm: COMBO, the composition compose_r3.py chose on the training set (composition.json, pinned).  Delta = COMBO - UF4 (both from this evaluation, the same code path), per subject.
The gate is BZ's (b) (decision_log 2026-11-11 ~3:38 PM CST, fixed before any R3 training or reading; not R2's +.005 / 15 of 20):
    过门        mean Delta seg35 >= +.002, at least 13 of 20 subjects positive, and mean Delta seg4 >= -.001  ->  the 19-pair registration reading (another release, another package of tools, R3c;
                its adoption by R2's registration bars)
    不过门      otherwise  ->  stop, hand to BZ; no other configuration is tried
    不可判      no evaluation, not 20 subjects, a member missing or not the plan's, the control of UF4 not passed (|mean| > 2e-4, or other truth files), a mean undefined
No option turns a reading into 不可判 after the fact.
    python -B experiments/r3/decide_r3.py --eval DIR/eval_r3.json --out-dir DIR/analysis
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent/'r2')]
import decide_r2 as d2  # noqa: E402
import r3_common as r3c  # noqa: E402

SCHEMA = 'bayesreg.r3.decision.v1'
PASS, NOT_PASS, UNDECIDABLE = d2.PASS, d2.NOT_PASS, d2.UNDECIDABLE
READINGS = {PASS: '过门：做 19 对配准读数（第二轮配准读数第 1 次；工具另起一包）', NOT_PASS: '不过门：停，交 BZ，不换配置重试', UNDECIDABLE: '不可判'}
DESCRIBED = d2.DESCRIBED
PRIMARY = 'COMBO'
NOTE = ('开发集 20 名受试者，单次观测，分割级读数，不是配准结果，不是留出集结果，不是对外比较。新成员（L、nnU-Net）只用训练集（每折 98 + 8 人）的标签训练；组合在训练集标定受试者上折外选出（16 人）。'
        '这 20 人已被反复读过。这是开发第二轮分割级读数的第 2 次，也是最后一次（BZ 2026-11-11）；门槛是 BZ 约 3:38 PM 定的 (b)，在任何 R3 训练与读数之前。')


def describe(x):
    return d2.describe(x, draws=r3c.SEG_BOOT_DRAWS, seed=r3c.SEG_BOOT_SEED)


def undecidable_reasons(evaluation):
    if evaluation is None:
        return ['没有评测结果']
    out = []
    if len(evaluation.get('subjects') or []) != r3c.N_SUBJECTS:
        out.append(f"{len(evaluation.get('subjects') or [])} 名受试者，不是 {r3c.N_SUBJECTS}")
    want = sorted([f'uf/{n}' for n in r3c.UF_ORDER]+[f'{r3c.L_FAMILY}/{n}' for n in r3c.L_NAMES])
    if sorted(evaluation.get('members') or {}) != want:
        out.append(f"成员缺失：{sorted(set(want)-set(evaluation.get('members') or {}))}")
    out += [f'成员不是计划训的：{p}' for p in evaluation.get('member_problems') or []]
    composition = evaluation.get('composition') or {}
    if not composition.get('sha256') or not composition.get('delivered'):
        out.append('没有组合的记录')
    elif composition['delivered'].get('nn') and not evaluation.get('nn'):
        out.append('组合含 nnU-Net，但评测没有 nnU-Net 的核对记录')
    control = evaluation.get('control') or {}
    if control.get('tolerance') != r3c.CONTROL_TOL_FORWARD:
        out.append(f"评测记下的控制容差 {control.get('tolerance')} 不是 {r3c.CONTROL_TOL_FORWARD}")
    for metric in ('seg35', 'seg4'):
        d = (control.get('differences') or {}).get(metric)
        if not d2.finite(d):
            out.append(f'控制缺失：UF4 的 {metric}')
        elif abs(d) > r3c.CONTROL_TOL_FORWARD:
            out.append(f'控制不符：UF4 的 {metric} 与 eval_uf.json 差 {d:+.3e}，超过 {r3c.CONTROL_TOL_FORWARD:g}')
    if control.get('truth_same') is not True:
        out.append('控制不符：真值文件不是 eval_uf.json 读过的那些')
    means = evaluation.get('means') or {}
    for arm in ('UF4', PRIMARY):
        for metric in ('seg35', 'seg4'):
            if not d2.finite((means.get(arm) or {}).get(metric)):
                out.append(f'{arm} 的 {metric} 没有定义')
    return out


def reading_of(d35, d4):
    """BZ's gate (b) on the primary's paired differences (described): mean seg35 >= +.002, positives >= 13, mean seg4 >= -.001 (the bars included; r3_common's constants)."""
    if not (d2.finite(d35.get('mean')) and d2.finite(d4.get('mean'))):
        return UNDECIDABLE, ['Δ 没有定义']
    why = [w for w, ok in ((f"Δseg35 均值 {d35['mean']:+.5f} < +{r3c.SEG_BAR}", d35['mean'] >= r3c.SEG_BAR), (f"为正的受试者 {d35['n_positive']}/{d35['n']} < {r3c.SEG_POSITIVE_MIN}", d35['n_positive'] >= r3c.SEG_POSITIVE_MIN),
                               (f"Δseg4 均值 {d4['mean']:+.5f} < {r3c.SEG4_FLOOR}", d4['mean'] >= r3c.SEG4_FLOOR)) if not ok]
    return (NOT_PASS, why) if why else (PASS, [])


def expectation(evaluation, slope=r3c.OX_SLOPE):
    """〔模型〕slope x the pair-level segmentation Delta (decide_r2.expectation with COMBO as the primary)."""
    return d2.expectation(evaluation, PRIMARY, slope)


def decide(evaluation):
    document = dict(schema=SCHEMA, plan=r3c.plan_record(REPO), rule=dict(primary=f'{PRIMARY}, the composition chosen on the training set', delta=f'{PRIMARY} - UF4 (this evaluation), per subject, 20 subjects',
                                                                      pass_=f'mean Delta seg35 >= +{r3c.SEG_BAR}, at least {r3c.SEG_POSITIVE_MIN}/20 positive, mean Delta seg4 >= {r3c.SEG4_FLOOR}', control_tolerance=r3c.CONTROL_TOL_FORWARD),
                    note=NOTE, dev_read=(evaluation or {}).get('dev_read'), composition=(evaluation or {}).get('composition'), is_conclusion=False)
    problems = undecidable_reasons(evaluation)
    if problems:
        return dict(document, reading=UNDECIDABLE, reading_text=READINGS[UNDECIDABLE], why=problems, primary_arm=None, candidate=None)
    arms = [a for a in evaluation['arms'] if a != 'UF4']
    comparisons = {f'{a}_minus_UF4': {m: describe(d2.column(evaluation, a, m)-d2.column(evaluation, 'UF4', m)) for m in DESCRIBED} for a in arms}
    d35, d4 = comparisons[f'{PRIMARY}_minus_UF4']['seg35'], comparisons[f'{PRIMARY}_minus_UF4']['seg4']
    reading, why = reading_of(d35, d4)
    means = evaluation['means']
    document.update(reading=reading, reading_text=READINGS[reading], why=why, primary_arm=PRIMARY, candidate=evaluation['composition']['winner'] if reading == PASS else None,
                    delta_seg35=d35['mean'], delta_seg4=d4['mean'], n_positive=d35['n_positive'], primary_minus_uf4=comparisons[f'{PRIMARY}_minus_UF4'], comparisons=comparisons,
                    means={a: {m: means[a].get(m) for m in DESCRIBED} for a in evaluation['arms']},
                    per_label_delta=[None if a is None or b is None else a-b for a, b in zip(evaluation['per_label_mean'][PRIMARY], evaluation['per_label_mean']['UF4'])],
                    focus=d2.focus(evaluation, ('UF4', PRIMARY)), first5_pairs={a: d2.pair_level(evaluation, a, r3c.SHORTS[:5]) for a in ('UF4', PRIMARY)}, expectation=expectation(evaluation),
                    control=evaluation['control'], seconds=evaluation.get('seconds'), plan_of_evaluation=evaluation.get('plan'))
    return document


def summary(document):
    L = ['# R3 分割级读数（decide_r3.py 写的；开发集 20 人，单次，分割级）', '', f"**读数：{document['reading_text']}**" + (f"（{'；'.join(document['why'])}）" if document.get('why') else ''), '', document['note'], '']
    if document['reading'] == UNDECIDABLE:
        return '\n'.join(L)+'\n'
    d = document['primary_minus_uf4']
    c = document['composition']
    L += [f"组合：{c['winner']}（{c['delivered']['words']}），composition.json sha256 {c['sha256'][:12]}…", '',
          f"Δseg35 {d2._s(d['seg35']['mean'], 5)}（{d['seg35']['n_positive']}/{d['seg35']['n']} 为正；t 95% [{d2._s(d['seg35']['t95'][0])}, {d2._s(d['seg35']['t95'][1])}]，bootstrap [{d2._s(d['seg35']['bootstrap95'][0])}, {d2._s(d['seg35']['bootstrap95'][1])}]），Δseg4 {d2._s(d['seg4']['mean'], 5)}。",
          f"线：Δseg35 ≥ +{r3c.SEG_BAR}，≥ {r3c.SEG_POSITIVE_MIN}/20 为正，Δseg4 ≥ {r3c.SEG4_FLOOR}。", '', '| 臂 | seg35 | seg4 | 小六个 | 大六个 | 其余 23 个 | 皮层 | 白质 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, m in document['means'].items():
        L.append(f"| {arm} | {d2._f(m.get('seg35'))} | {d2._f(m.get('seg4'))} | {d2._f(m.get('small_six'))} | {d2._f(m.get('big_six'))} | {d2._f(m.get('other_23'))} | {d2._f(m.get('cortex'))} | {d2._f(m.get('white_matter'))} |")
    e = document['expectation']
    L += ['', f"〔模型〕配准级预期（19 对，斜率 {e['slope']}）：{d2._s(e['pairs19']['expected_delta'])}。{e['note']}", '',
          f"控制：UF4 对 eval_uf.json 的均值差 {document['control']['differences']}，容差 {document['control']['tolerance']}，逐行相同 {document['control']['rows_equal']}/20。"]
    return '\n'.join(L)+'\n'


def write(out_dir, document):
    out_dir = Path(out_dir)
    r3c.write_json_once(out_dir/'decision_r3.json', document)
    r3c.write_text_once(out_dir/'summary_r3.md', summary(document))
    return out_dir/'decision_r3.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eval', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    document = decide(r3c.read_json(args.eval))
    write(args.out_dir, document)
    print(json.dumps({k: document.get(k) for k in ('reading', 'reading_text', 'why', 'primary_arm', 'delta_seg35', 'delta_seg4', 'n_positive')}, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
