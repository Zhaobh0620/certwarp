"""SV's reading (runs/20261104-sv/plan.md, revision 1; by code; a label for the lead and BZ, not a conclusion): analysis/decision_sv.json, summary_sv.md, long.csv.

The line: S_P's measured offset from its first 5 pairs to its 19 pairs is +.0132 (.8430, .8562, the brief's rounded numbers), so .88 on the 19 pairs corresponds to .8668 on the first 5.  The line is that pinned number; the unrounded offset (.01325) is recorded next to it.
An arm can be read when all 5 pairs have class ok, at least one native commit, a final map with a passed certificate, an entry injected as recorded, the plan's configuration, oracle true, and a mean official SDlogJ <= .10.  Then, on the unrounded mean seg35, bars included:
  r1_3 readable and >= .8668                                 减三分之一就够
  both readable, r1_3 < .8668 <= r2_3                        减三分之二才够
  both readable and both below                               减三分之二也不够
  anything else, or a problem with the endpoints             不可判, the reasons written
The arms are named by the error they actually remove: 1 - mean(1 - seg35) / mean at f = 0, over the 6 subjects (labels/build_record.json); the reading says that number, seg35 and seg4.  f0440 never enters the reading.
Only described: the endpoints f = 0 (S_P) and f = 1 (S_T) of seed_v1 in the same table, marked as another time; the reductions by group; the labels that vanish; the slope from segmentation-level Dice to final Dice; entry -> final; f0440 against S_P and S_T on its two pairs.
"""
import csv
from pathlib import Path

import numpy as np

import run_sv as core                                        # first: it puts experiments/seed on sys.path
import decide_seed  # noqa: E402    # read only, experiments/seed

SCHEMA = 'bayesreg.sv.decision.v1'
LINE, TARGET_19, OFFSET_PINNED = .8668, .88, .0132
SDLOGJ_MAX = .10
NEAR_LINE = .001                                             # a sentence only, the code stays what the table says
MIN_STEP = .002                                              # a step of the slope with a smaller difference of segmentation-level Dice is not computed: its denominator is noise
PINS = dict(S_P=(.8430, .8984), S_T=(.9000, .9438))          # (seg35, seg4) of seed_v1 over the 5 pairs, to 4 decimals (.84298, .89840; .89995, .94383)
PIN_TOL = 5.01e-5
ENDPOINT_ROOT = 'seed_v1'
SECOND_ROOT = 'seed_v1b'                                     # S_P on the other 14 pairs: only for the record of the offset
REST_PAIRS = tuple(f'OASIS_OAS1_{n:04d}_MR1__OASIS_OAS1_{n+1:04d}_MR1' for n in range(443, 457))
ENDPOINTS = dict(S_P=core.TAG_PRED, S_T=core.TAG_TRUTH)
GROUP_KEYS = ('small_six', 'big_six', 'other_23', 'cortex', 'white_matter')      # segmentation level: experiments/unv2/eval_seg.py names; the same labels as evaluate.py A5_GROUPS
THIRD_ENOUGH, TWO_THIRDS_NEEDED, NOT_ENOUGH, NOT_DECIDABLE = 'third_enough', 'two_thirds_needed', 'two_thirds_not_enough', 'not_decidable'
READINGS = {THIRD_ENOUGH: '减三分之一就够', TWO_THIRDS_NEEDED: '减三分之二才够', NOT_ENOUGH: '减三分之二也不够', NOT_DECIDABLE: '不可判'}
NOTE = ('开发集前 5 对，单次观测，读数不是结论，下一步交总负责人与 BZ。oracle 臂：标签由真值参与构造（provenance oracle_postmix_*），结果只在隔离根目录 segvoi_v1，不进结果表，不是方法。n = 5，这条线的偏移只来自一个臂（S_P）。'
        '后验混合优先改掉低置信的错误，高置信的系统性错误留到最后；真实网络的改进不一定是这个形态，所以"减三分之一""减三分之二"说的是 1 − seg35 按这种方式减少时的值。〔推断〕先改掉的低置信错误更可能是两幅图不一致的部分，留到最后的系统性错误在两幅图上可能部分抵消，所以同样的误差减少，这里的终态涨幅可能高于真实网络，曲线可能偏乐观：读"够"（尤其"减三分之一就够"）要打这个折扣，"不够"相对可靠。'
        '线性折算下 r1_3 只比线低约 .005，与 19 对的最小可检测效应同量级，而这里只有 5 对，这一步的代码可能由噪声决定：主要读 crossing 和表，不要只看代码。端点 f = 0（S_P）、f = 1（S_T）读自 seed_v1，没有重跑，不同时段，入口也来自别的批次。预算收尾不是收敛。')
OPTIMISTIC = '〔推断〕曲线可能偏乐观（留到最后的系统性错误在两幅图上可能部分抵消），这个"够"要打折读'
TABLE_ROWS = (('f = 0', 'S_P', 'seed_v1 的 S_P（预测，oracle false）；另一时段'), ('减 1/3', 'r1_3', '本批 oracle'), ('减 2/3', 'r2_3', '本批 oracle'), ('f = 1', 'S_T', 'seed_v1 的 S_T（真值，oracle）；另一时段'))


def load_runs(root):
    """{(arm, pair): metrics.json} of this batch."""
    out = {}
    for arm, pair in core.plan_cells():
        m = core.read_json(core.run_dir(root, arm, pair)/'metrics.json')
        if m is not None:
            out[(arm, pair)] = m
    return out


def load_endpoints(results):
    """{(S_P or S_T, pair): metrics.json} of seed_v1, read only."""
    out = {}
    for arm in ENDPOINTS:
        for pair in core.PAIRS:
            m = core.read_json(Path(results)/ENDPOINT_ROOT/'runs'/arm/pair/'metrics.json')
            if m is not None:
                out[(arm, pair)] = m
    return out


final_ok = decide_seed.final_ok


def final_value(m, key):
    """A value of a final map counts only with a passed certificate."""
    return m['metrics'].get(key) if final_ok(m) else None


def mean_of(runs, arm, key, pairs):
    """The mean over the pairs of a certified final map, None unless every pair has one."""
    values = [final_value(runs.get((arm, pair)), key) for pair in pairs]
    return None if any(v is None for v in values) else float(np.mean(values))


def plan_configuration(arm, root):
    return dict(seconds=core.seed.SECONDS, changes=core.seed.CHANGES, inputs_root=str(core.inputs_root(arm, root)))


def arm_premises(runs, arm, root):
    """The reasons this arm cannot be read (empty: it can)."""
    reasons = []
    pairs = core.ARMS[arm]['pairs']
    for pair in pairs:
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
        if m.get('oracle') is not True:
            reasons.append(f'{arm} {short}: 记录没有标 oracle true')
        if m.get('run_labels_equal_truth') is not True:
            reasons.append(f'{arm} {short}: 运行里的标签副本与真值不同')
        injected = m.get('seed') or {}
        if injected.get('passed') is not True:
            reasons.append(f'{arm} {short}: 入口核对未通过 {injected.get("problems")}')
        have, want = injected.get('arm') or {}, plan_configuration(arm, root)
        if {k: have.get(k) for k in want} != want or have.get('charge_seconds') != core.seed.ENTRY_CHARGE_SECONDS:
            reasons.append(f'{arm} {short}: 配置与计划不同 {have}')
    finals = [runs[(arm, p)]['metrics']['sdlogj_official'] for p in pairs if (arm, p) in runs and runs[(arm, p)].get('metrics')]
    if len(finals) == len(pairs) and float(np.mean(finals)) > SDLOGJ_MAX:
        reasons.append(f'{arm}: 官方 SDlogJ 均值 {float(np.mean(finals)):.4f} > {SDLOGJ_MAX}')
    return reasons


def endpoint_input_problems(results):
    """What the queue asks of the endpoints before anything runs."""
    runs = load_endpoints(results)
    return [f'{ENDPOINT_ROOT} {arm} {core.short_name(pair)}: no metrics.json' for arm in ENDPOINTS for pair in core.PAIRS if (arm, pair) not in runs]+endpoint_problems(runs)


def endpoint_problems(runs):
    """The endpoints against the plan's pins: a different number means a different baseline from the one the plan read."""
    bad = []
    for arm, (seg35, seg4) in PINS.items():
        a, b = mean_of(runs, arm, 'seg35', core.PAIRS), mean_of(runs, arm, 'seg4', core.PAIRS)
        if a is None or b is None or abs(a-seg35) > PIN_TOL or abs(b-seg4) > PIN_TOL:
            bad.append(f'{arm}: 前 5 对 seg35 {a}, seg4 {b}（每对都要有通过证书的终态场），计划钉住 {seg35}, {seg4}')
    return bad


def reduction_of(build, arm, key='seg35'):
    """The error this arm actually removes (labels/build_record.json), 6 subjects; 0 and 1 at the endpoints by definition.  None before the build."""
    if arm == 'S_P':
        return 0.
    if arm == 'S_T':
        return 1.
    row = ((build or {}).get('arms') or {}).get(arm) or {}
    return (row.get('reduction') or {}).get(key)


def arm_reading(runs, arm, root, build):
    reasons = arm_premises(runs, arm, root)
    seg35, seg4 = mean_of(runs, arm, 'seg35', core.PAIRS), mean_of(runs, arm, 'seg4', core.PAIRS)
    readable = not reasons and seg35 is not None
    arms = (build or {}).get('arms') or {}
    return dict(arm=arm, f=(arms.get(arm) or {}).get('f'), target=core.ARMS[arm]['target'], reduction_seg35=reduction_of(build, arm), reduction_seg4=reduction_of(build, arm, 'seg4'), provenance=core.ARMS[arm]['provenance'], seg35=seg35, seg4=seg4, readable=readable, reached=(seg35 >= LINE) if readable else None, reasons=reasons)


def fmt(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def percent(value):
    return '—' if value is None else f'{100*value:.1f}%'


def arm_words(row):
    return f"{row['arm']}（f = {fmt(row['f'])}，实际减少 seg35 {percent(row['reduction_seg35'])}、seg4 {percent(row['reduction_seg4'])}）终态 seg35 {fmt(row['seg35'])} / seg4 {fmt(row['seg4'])}"


def decide(runs, root, reference_problems=(), build=None):
    """The plan's table.  Unrounded means, the bar included.  Only r1_3 and r2_3: f0440 has no way in."""
    arms = {arm: arm_reading(runs, arm, root, build) for arm in core.READING_ARMS}
    third, two_thirds = arms['r1_3'], arms['r2_3']
    reasons = list(reference_problems)
    if reasons:
        code = NOT_DECIDABLE
    elif third['readable'] and third['reached']:
        code = THIRD_ENOUGH
    elif third['readable'] and two_thirds['readable']:
        code = TWO_THIRDS_NEEDED if two_thirds['reached'] else NOT_ENOUGH
    else:
        code = NOT_DECIDABLE
        reasons = third['reasons']+two_thirds['reasons']
    numbers = f"{arm_words(third)}；{arm_words(two_thirds)}；线 {LINE}（seg35）"
    text = READINGS[code]+'：'+numbers+('。原因：'+'；'.join(reasons) if reasons else '')
    near = [arm for arm, row in arms.items() if row['seg35'] is not None and abs(row['seg35']-LINE) < NEAR_LINE]
    if near and not reasons:                             # a sentence for the reader, the code stays what the table says
        text += f'；{"、".join(near)} 贴着线（差不到 {NEAR_LINE:g}），未舍入的偏移给出的线是 .8667'
    if code == THIRD_ENOUGH and not two_thirds['reached']:
        text += '；r2_3 的 '+('均值低于线' if two_thirds['readable'] else '前提不满足')+'，要带着读'
    if code in (THIRD_ENOUGH, TWO_THIRDS_NEEDED):        # the code stays what the table says; revision 2 of the plan
        text += '；'+OPTIMISTIC
    return dict(reading_code=code, reading=text, reasons=reasons, near_line=near, arms=arms)


def row_of_subject(build, subject, the_tag):
    return (((build or {}).get('subjects') or {}).get(subject, {}).get('labels') or {}).get(the_tag)


def tag_of(arm, subject):
    return ENDPOINTS[arm] if arm in ENDPOINTS else core.label_tag(arm, subject)


def segmentation_of(build, arm, pair, key):
    """The segmentation-level Dice of a pair: the mean of its two subjects' maps against the truth (labels/build_record.json); the endpoints are `pred` and `truth` of the same record.  None before the build, or where a value is undefined."""
    values = []
    for subject in core.subjects_of(pair):
        record = row_of_subject(build, subject, tag_of(arm, subject))
        value = None if record is None else record['dice_against_truth'].get(key)
        if value is None:
            return None
        values.append(value)
    return float(np.mean(values))


def entry_of(m):
    """The entry's own numbers, alpha = 1, after the taper, and the alpha of the frozen transfer (uncertified field: a description)."""
    injected = (m or {}).get('seed') or {}
    e = injected.get('entry_metrics') or {}
    return dict(seg35=e.get('seg35'), seg4=e.get('seg4'), sdlogj_official=e.get('sdlogj_official'), alpha_in_evaluation=e.get('accepted_alpha'), transfer_alpha=injected.get('accepted_alpha'))


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def table_row(runs, build, name, arm, source, pairs):
    """One row of the table: a final number only when every pair has a certified final map."""
    certified = [final_ok(runs.get((arm, p))) for p in pairs]
    entries = [entry_of(runs.get((arm, p))) for p in pairs]
    commits = [(runs.get((arm, p)) or {}).get('native_commits') for p in pairs]

    def segmentation(key):
        values = [segmentation_of(build, arm, p, key) for p in pairs]
        return None if any(v is None for v in values) else float(np.mean(values))
    return dict(name=name, arm=arm, source=source, f=((build or {}).get('arms') or {}).get(arm, {}).get('f'), n_pairs=len(pairs), n_certified=sum(certified), reduction_seg35=reduction_of(build, arm), reduction_seg4=reduction_of(build, arm, 'seg4'), reduction_groups={k: reduction_of(build, arm, k) for k in GROUP_KEYS},
                segmentation_seg35=segmentation('seg35'), segmentation_seg4=segmentation('seg4'), segmentation_groups={k: segmentation(k) for k in GROUP_KEYS}, entry_seg35=mean_or_none([e['seg35'] for e in entries]), entry_seg4=mean_or_none([e['seg4'] for e in entries]),
                final_seg35=mean_of(runs, arm, 'seg35', pairs), final_seg4=mean_of(runs, arm, 'seg4', pairs), hd95=mean_of(runs, arm, 'hd95', pairs), sdlogj_official=mean_of(runs, arm, 'sdlogj_official', pairs), sdlogj_brain=mean_of(runs, arm, 'sdlogj_brain', pairs),
                native_commits=mean_or_none(commits), transfer_alpha=[e['transfer_alpha'] for e in entries])


def slopes(table, measure):
    """From the segmentation-level Dice to the final Dice over the rows that have both: each step (not computed below MIN_STEP of segmentation-level difference: its denominator is noise) and the least-squares slope.  A description."""
    points = [(row['name'], row[f'segmentation_{measure}'], row[f'final_{measure}']) for row in table if row[f'segmentation_{measure}'] is not None and row[f'final_{measure}'] is not None]
    steps = [dict(between=f'{a[0]} → {b[0]}', slope=(b[2]-a[2])/(b[1]-a[1]) if abs(b[1]-a[1]) >= MIN_STEP else None) for a, b in zip(points, points[1:])]
    distinct = len({p[1] for p in points}) >= 2
    return dict(measure=measure, points=len(points), steps=steps, least_squares=float(np.polyfit([p[1] for p in points], [p[2] for p in points], 1)[0]) if distinct else None)


def crossing(table):
    """〔模型〕Where the piecewise-linear curve of the final seg35 against the actual reduction first reaches the line: the reduction and the segmentation-level Dice.  Not a bar."""
    points = [(row['reduction_seg35'], row['segmentation_seg35'], row['final_seg35']) for row in table if row['final_seg35'] is not None and row['reduction_seg35'] is not None]
    if points and points[0][2] >= LINE:
        return dict(reduction_seg35=points[0][0], segmentation_seg35=points[0][1], role='〔模型〕第一个点已经在线上')
    for a, b in zip(points, points[1:]):
        if a[2] < LINE <= b[2]:
            t = (LINE-a[2])/(b[2]-a[2])
            return dict(reduction_seg35=a[0]+(b[0]-a[0])*t, segmentation_seg35=None if a[1] is None or b[1] is None else a[1]+(b[1]-a[1])*t, role='〔模型〕相邻两点线性内插，不是判据')
    return dict(reduction_seg35=None, segmentation_seg35=None, role='〔模型〕表里的点没有跨过这条线')


def line_record(results):
    """The offset behind the pinned line, unrounded: S_P on the 19 pairs minus S_P on the first 5 (seed_v1, seed_v1b, read only); None without those files."""
    first = [core.read_json(Path(results)/ENDPOINT_ROOT/'runs/S_P'/p/'metrics.json') for p in core.PAIRS]
    rest = [core.read_json(Path(results)/SECOND_ROOT/'runs/S_P'/p/'metrics.json') for p in REST_PAIRS]
    record = dict(line=LINE, target_19=TARGET_19, offset_pinned=OFFSET_PINNED, s_p_19=None, s_p_first5=None, offset_unrounded=None, line_unrounded=None,
                  role='线取任务书的 .8668（.88 − .0132，由舍入后的 .8562、.8430 算出）；未舍入的偏移只记录，不改线')
    if all(m is not None and m.get('metrics') for m in first+rest):
        five, nineteen = float(np.mean([m['metrics']['seg35'] for m in first])), float(np.mean([m['metrics']['seg35'] for m in first+rest]))
        record.update(s_p_19=nineteen, s_p_first5=five, offset_unrounded=nineteen-five, line_unrounded=TARGET_19-(nineteen-five))
    return record


def per_pair_rows(runs, build):
    rows = []
    for name, arm, source in TABLE_ROWS+(('一人真值', 'f0440', '本批 oracle，只描述'),):
        pairs = core.ARMS[arm]['pairs'] if arm in core.ARMS else core.PAIRS
        for pair in pairs:
            m = runs.get((arm, pair)) or {}
            x, e = m.get('metrics') or {}, entry_of(m)
            rows.append(dict(name=name, arm=arm, pair=core.short_name(pair), run_id=m.get('run_id'), cls=(m.get('classification') or {}).get('class'), certified=final_ok(m) if m else None, segmentation_seg35=segmentation_of(build, arm, pair, 'seg35'), segmentation_seg4=segmentation_of(build, arm, pair, 'seg4'),
                             entry_seg35=e['seg35'], entry_seg4=e['seg4'], transfer_alpha=e['transfer_alpha'], seg35=x.get('seg35'), seg4=x.get('seg4'), hd95=x.get('hd95'), sdlogj_official=x.get('sdlogj_official'), sdlogj_brain=x.get('sdlogj_brain'), native_commits=m.get('native_commits'),
                             min_certified_bound=(m.get('certificate') or {}).get('min_jd_lower_bound'), e2e=m.get('e2e_seconds'), status=m.get('status')))
    return rows


def segmentation_of_subject(build, subject, the_tag):
    record = row_of_subject(build, subject, the_tag)
    return None if record is None else {k: record['dice_against_truth'].get(k) for k in ('seg35', 'seg4')}


def hard_subject(runs, build, line, stop=None):
    """f0440: only subject 0440 takes the truth, on its two pairs, against S_P (f = 0) and S_T (f = 1) of the same pairs.  Only described: decide() does not read it."""
    pairs = core.ARMS['f0440']['pairs']
    rows = []
    for pair in pairs:
        row = dict(pair=core.short_name(pair), certified=final_ok(runs.get(('f0440', pair))))
        for arm in ('f0440', 'S_P', 'S_T'):
            for key in ('seg35', 'seg4'):
                row[f'{arm}_{key}'] = final_value(runs.get((arm, pair)), key)
        for key in ('seg35', 'seg4'):
            a = row[f'f0440_{key}']
            row[f'delta_{key}_from_S_P'] = None if a is None or row[f'S_P_{key}'] is None else a-row[f'S_P_{key}']
            row[f'to_S_T_{key}'] = None if a is None or row[f'S_T_{key}'] is None else row[f'S_T_{key}']-a
        m = runs.get(('f0440', pair)) or {}
        e = entry_of(m)
        row.update(entry_seg35=e['seg35'], entry_seg4=e['seg4'], transfer_alpha=e['transfer_alpha'], sdlogj_official=(m.get('metrics') or {}).get('sdlogj_official'), native_commits=m.get('native_commits'), segmentation={s: {core.label_tag('f0440', s): segmentation_of_subject(build, s, core.label_tag('f0440', s))} for s in core.subjects_of(pair)})
        rows.append(row)
    deltas = [r['delta_seg35_from_S_P'] for r in rows]
    gain = None if any(d is None for d in deltas) else float(sum(deltas))
    s_p_19 = line.get('s_p_19')
    return dict(role='只描述，不进预登记读数；只有 0440 的标签换成真值（入口通道和上色都换），另一名受试者仍用预测；排在主读数写出之后', subject=core.HARD_SUBJECT, stop=stop, rows=rows, sum_delta_seg35=gain, on_19_pairs=None if gain is None else gain/19, s_p_19_if_only_these_change=None if gain is None or s_p_19 is None else s_p_19+gain/19,
                model_role='〔模型〕只有这两对变、其余 17 对不变时 19 对 seg35 的变化；S_P 19 对读自 seed_v1、seed_v1b')


def vanished_table(build):
    """{arm: {subject: [labels]}} and the smallest volume of a label against the truth, per arm: the description of what the mix took away."""
    out = {}
    for arm in core.ARMS:
        row = ((build or {}).get('arms') or {}).get(arm) or {}
        smallest = None
        for subject, the_tag in (row.get('tags') or {}).items():
            ratios = (row_of_subject(build, subject, the_tag) or {}).get('volume_vs_truth') or []
            values = [v for v in ratios[1:] if v is not None]
            if values:
                smallest = min(values) if smallest is None else min(smallest, min(values))
        out[arm] = dict(vanished=row.get('vanished') or {}, smallest_volume_vs_truth=smallest)
    return out


def control_state(root):
    """control.json and the earlier versions kept beside it; `sentence` is what the head of the summary says."""
    root = Path(root)
    record = core.read_json(root/'control.json')
    history = sorted(p.name for p in root.glob('control_*.json'))
    if record is None:
        sentence = f'控制格 {core.CONTROL_ARM}：没有记录。'
    elif record.get('control_bitwise') is True:
        sentence = f'控制格 {core.CONTROL_ARM}：入口 {core.short_name(core.CONTROL_PAIR)} 对 S_P 自己的入口逐位相同，入口这一步相对 S_P 只差标签。'
    else:
        d = record.get('difference') or {}
        numbers = f"max|Δu| {fmt(d.get('max_abs'), 6)}，RMS {fmt(d.get('rms'), 6)}，不同的值占 {percent(d.get('share_of_values_different'))}{'，'+d['reason'] if d.get('reason') else ''}"
        reason = record.get('accepted_difference')
        sentence = (f'**控制格 {core.CONTROL_ARM} 与 S_P 自己的入口不同（{numbers}），已被接受，理由：{reason}。这批"相对 S_P 只差标签"没有逐位证明，要带着读。**' if reason else f'**控制格 {core.CONTROL_ARM} 与 S_P 自己的入口不同（{numbers}），没有接受记录。**')
    return dict(record=record, history=history, sentence=sentence, bitwise=None if record is None else record.get('control_bitwise') is True)


def anomalies(runs, ev, control=None):
    keys = core.plan_cells()
    out = ev.b1_anomalies(runs, keys)
    for arm, pair in keys:
        injected = (runs.get((arm, pair)) or {}).get('seed') or {}
        e = injected.get('entry_metrics') or {}
        if injected.get('accepted_alpha') is not None and e.get('accepted_alpha') is not None and injected['accepted_alpha'] != e['accepted_alpha']:
            out.append(dict(arm=arm, pair=pair, what=f"转移 α {injected['accepted_alpha']} 与入口评测的 α {e['accepted_alpha']} 不同"))
    if control is not None and control['bitwise'] is False:
        out.append(dict(arm=core.CONTROL_ARM, pair=core.CONTROL_PAIR, what=control['sentence'].strip('*')))
    return out


def summary_text(result):
    lines = ['# SV 汇总（代码写出，读数不是结论）', '', result['note'], '', f"**读数**：{result['reading']}", '', result['control']['sentence'], '', f"结果根目录 {result['root']}；写出 {result['written_utc']}。oracle: true；不进结果表。", '',
             f"线：{result['line']['role']}。线 {result['line']['line']}；未舍入的偏移 {fmt(result['line']['offset_unrounded'], 5)}，对应 {fmt(result['line']['line_unrounded'], 5)}。", '',
             '## 表：实际减少多少误差，终态涨多少（前 5 对均值；减少比例是 6 人）', '', '减少比例 = 1 − 6 人平均的 (1 − Dice) ÷ f = 0 时的同一个量。端点 f = 0、f = 1 不是本批，读自 seed_v1，不同时段，入口也来自别的批次。入口是未认证的场在 α = 1、taper 之后的数。', '',
             '| 行 | 来源 | f | 实际减少 seg35 / seg4 | 分割级 seg35 / seg4 | 入口 seg35 / seg4 | 终态 seg35 / seg4 | 官方 SDlogJ | 脑内 SDlogJ | HD95 | 原生提交（均） | 证书 |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for row in result['table']:
        lines.append(f"| {row['name']} | {row['source']} | {fmt(row['f'])} | {percent(row['reduction_seg35'])} / {percent(row['reduction_seg4'])} | {fmt(row['segmentation_seg35'])} / {fmt(row['segmentation_seg4'])} | {fmt(row['entry_seg35'])} / {fmt(row['entry_seg4'])} | {fmt(row['final_seg35'])} / {fmt(row['final_seg4'])} | {fmt(row['sdlogj_official'])} | {fmt(row['sdlogj_brain'], 3)} | {fmt(row['hd95'], 3)} | {fmt(row['native_commits'], 1)} | {row['n_certified']}/{row['n_pairs']} |")
    lines += ['', '## 分割级分组：实际减少比例（6 人）与 Dice（前 5 对均值）', '', '| 行 | 小六个 | 大六个 | 其余 23 个 | 皮层（seg4） | 白质（seg4） |', '|---|---|---|---|---|---|']
    for row in result['table']:
        lines.append('| '+row['name']+' | '+' | '.join(f"{percent(row['reduction_groups'][k])}，Dice {fmt(row['segmentation_groups'][k])}" for k in GROUP_KEYS)+' |')
    lines += ['', '## 标签消失（L 里 0 体素、预测或真值里有）与最小体积比（对真值）', '', '| 臂 | 消失的标签（受试者: 标签） | 最小体积比 |', '|---|---|---|']
    for arm, row in result['vanished'].items():
        lines.append(f"| {arm} | {row['vanished'] or '没有'} | {fmt(row['smallest_volume_vs_truth'], 3)} |")
    lines += ['', '## 只描述：分割级 Dice 到终态 Dice 的斜率', '']
    for measure, record in result['slopes'].items():
        steps = '，'.join(f"{s['between']}：{fmt(s['slope'], 3)}" for s in record['steps']) or '点不够'
        lines.append(f"- {measure}：{steps}；最小二乘 {fmt(record['least_squares'], 3)}（{record['points']} 个点；分割级差小于 {MIN_STEP} 的一步不算）")
    cross = result['crossing']
    hard = result['hard_subject']
    lines += [f"- 到线的位置（{cross['role']}）：实际减少 {percent(cross['reduction_seg35'])}，分割级 seg35 {fmt(cross['segmentation_seg35'])}", '', '## 只描述：f0440（只有 0440 用真值，入口通道和上色都换；不进读数）', '', hard['role']+'。'+(f"**这一格没有跑完，只记录，读数不受影响**：{hard['stop']['reason']}" if hard['stop'] else ''), '',
              '| 对 | f0440 seg35 / seg4 | S_P（f = 0） | S_T（f = 1） | 相对 S_P | 入口 seg35 / seg4 | 转移 α | 官方 SDlogJ | 原生提交 | 证书 |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in hard['rows']:
        lines.append(f"| {r['pair']} | {fmt(r['f0440_seg35'])} / {fmt(r['f0440_seg4'])} | {fmt(r['S_P_seg35'])} / {fmt(r['S_P_seg4'])} | {fmt(r['S_T_seg35'])} / {fmt(r['S_T_seg4'])} | {fmt(r['delta_seg35_from_S_P'])} / {fmt(r['delta_seg4_from_S_P'])} | {fmt(r['entry_seg35'])} / {fmt(r['entry_seg4'])} | {r['transfer_alpha']} | {fmt(r['sdlogj_official'])} | {r['native_commits']} | {r['certified']} |")
    lines += ['', f"seg35 增量之和 {fmt(hard['sum_delta_seg35'])}；{hard['model_role']}：19 对 +{fmt(hard['on_19_pairs'])}，S_P 19 对变成 {fmt(hard['s_p_19_if_only_these_change'])}。", '',
              '## 只描述：逐对', '', '| 行 | 臂 | 对 | 分割级 seg35 / seg4 | 入口 seg35 / seg4 | 转移 α | 终态 seg35 / seg4 | 官方 SDlogJ | 原生提交 | 最小认证下界 | 证书 | e2e s | run_id |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in result['per_pair']:
        lines.append(f"| {r['name']} | {r['arm']} | {r['pair']} | {fmt(r['segmentation_seg35'])} / {fmt(r['segmentation_seg4'])} | {fmt(r['entry_seg35'])} / {fmt(r['entry_seg4'])} | {r['transfer_alpha']} | {fmt(r['seg35'])} / {fmt(r['seg4'])} | {fmt(r['sdlogj_official'])} | {r['native_commits']} | {r['min_certified_bound']} | {r['certified']} | {fmt(r['e2e'], 1)} | {r['run_id']} |")
    lines += ['', '## 只描述：终态分组（前 5 对均值）', '', '| 臂 | 小六个 | 大六个 | 其余 23 个 | 皮层（seg4） | 白质（seg4） |', '|---|---|---|---|---|---|']
    for arm, row in result['groups'].items():
        lines.append(f"| {arm} | {fmt(row['small_six'])} | {fmt(row['large_six'])} | {fmt(row['other_23'])} | {fmt(row['cortex'])} | {fmt(row['white_matter'])} |")
    lines += ['', '## 异常', '']
    lines += [f"- {a['arm']} {a['pair']}: {a['what']}" for a in result['anomalies']] or ['没有。']
    if result['reference_problems']:
        lines += ['', '## 端点问题', '']+[f'- {p}' for p in result['reference_problems']]
    return '\n'.join(lines)+'\n'


def segmentation_long_rows(build, ev):
    """Segmentation-level rows for long.csv: per subject (in the column `pair`) and label map, the Dice of the groups and each arm's reduction.  run_id names the record they come from."""
    rows = []
    if build is None:
        return rows
    base = dict(set=ev.SET, split=ev.SPLIT, run_id='labels/build_record.json')
    for arm, spec in ((a, core.ARMS[a]) for a in core.ARMS):
        tags = ((build.get('arms') or {}).get(arm) or {}).get('tags') or {}
        for subject, the_tag in tags.items():
            dice = (row_of_subject(build, subject, the_tag) or {}).get('dice_against_truth') or {}
            reduction = (((build.get('arms') or {}).get(arm) or {}).get('reduction_per_subject') or {}).get(subject) or {}
            for key in ('seg35', 'seg4')+GROUP_KEYS:
                if dice.get(key) is not None:
                    rows.append(dict(base, method=spec['provenance'], arm=arm, pair=subject, measure=f'segmentation_{key}', value=dice[key]))
                if reduction.get(key) is not None:
                    rows.append(dict(base, method=spec['provenance'], arm=arm, pair=subject, measure=f'reduction_{key}', value=reduction[key]))
    return rows


def long_csv(path, own, ev, build):
    with open(path, 'w', newline='') as out:
        w = csv.DictWriter(out, fieldnames=['set', 'split', 'method', 'arm', 'pair', 'measure', 'value', 'run_id'])
        w.writeheader()
        w.writerows(dict(r, method=own[(r['arm'], r['pair'])].get('method', r['method'])) for r in ev.long_rows(own))
        w.writerows(segmentation_long_rows(build, ev))


def write_analysis(root, results=None):
    """Reads the runs and the endpoints, writes the three files; an earlier summary is moved to analysis/previous_<UTC>/.  Returns the decision."""
    root = Path(root)
    results = Path(core.RESULTS if results is None else results)
    ev = core.seed.load_evaluate()
    own = load_runs(root)
    runs = dict(own)
    runs.update(load_endpoints(results))
    problems = endpoint_problems(runs)
    build = core.read_json(root/'labels/build_record.json')
    line = line_record(results)
    table = [table_row(runs, build, name, arm, source, core.PAIRS) for name, arm, source in TABLE_ROWS]
    pairs = list(core.PAIRS)
    groups = {arm: dict({g: ev.a5_group_mean(runs, arm, labels, pairs) for g, labels in ev.A5_GROUPS.items()}, cortex=ev.label_rows(runs, arm, arm, 'seg4', (1,), pairs)[1]['arm_mean'], white_matter=ev.label_rows(runs, arm, arm, 'seg4', (3,), pairs)[3]['arm_mean']) for _, arm, _ in TABLE_ROWS}
    control = control_state(root)
    result = dict(schema=SCHEMA, root=str(root), written_utc=core.utc(), oracle=True, role=core.ROLE, note=NOTE, rule='runs/20261104-sv/plan.md', reference_problems=problems, control=control, line=line, **decide(runs, root, problems, build), table=table, slopes={m: slopes(table, m) for m in ('seg35', 'seg4')}, crossing=crossing(table), vanished=vanished_table(build),
                  hard_subject=hard_subject(runs, build, line, core.read_json(root/'f0440_stop.json')), per_pair=per_pair_rows(runs, build), groups=groups, anomalies=anomalies(runs, ev, control), sources=dict(endpoints=str(results/ENDPOINT_ROOT), evaluator_sha256=core.sha_file(core.ROOT/'experiments/d0/evaluate.py'), code_sha256=core.sha_file(Path(__file__))))
    out = root/'analysis'
    out.mkdir(parents=True, exist_ok=True)
    moved = ev._set_aside(out, ('long.csv', 'summary_sv.md', 'decision_sv.json'))
    result['previous_summary_moved_to'] = None if moved is None else str(moved)
    long_csv(out/'long.csv', own, ev, build)
    core.write_json(out/'decision_sv.json', result)
    (out/'summary_sv.md').write_text(summary_text(result), encoding='utf-8')
    return result
