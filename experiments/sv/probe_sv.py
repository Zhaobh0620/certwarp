"""SV, revision 1: the CPU probe that fixes the arms before any GPU (runs/20261104-sv/plan.md revision 1).  On the 6 subjects, for each map against the truth: the reduction of the error 1 - Dice relative to f = 0, overall and by group, each subject and the mean; each label's volume; two f* by bisection.
Maps: the posterior mix on a grid of f, the SDF mix at f = 1/3, 2/3 for comparison only.  It writes analysis/probe_posterior.json and .md under the isolated root and no label map.  The truth labels are read for this segmentation-level comparison, of these 6 subjects only (run_sv.read_subjects).

    run_sv.py probe --root DIR [--workers 1]        (needs PYTHONPATH=<frozen>:<code>, as the build)

It stops rather than overwrites: a probe that exists is evidence.  No registration here: nothing in it is a registration result.
"""
from fractions import Fraction
from pathlib import Path
import resource
import time

import numpy as np

import postmix
import run_sv as core
import sdfmix

SCHEMA = 'bayesreg.sv.probe.v1'
F_GRID = tuple(round(.02*k, 2) for k in range(1, 25))+(.49, .495, .499, .4999)        # .02, .04, ..., .48, then up to the edge of (0, 1/2): an error flips at f = d/(1 + d), so d > .96 needs f > .49
SDF_FRACTIONS = (Fraction(1, 3), Fraction(2, 3))
F_TOL = 1e-4
MAX_BISECTIONS = 30
F_DECIMALS = 4


def f_key(f):
    """The name of an f in the records, and of what is pinned: 4 decimals."""
    return f'{f:.{F_DECIMALS}f}'


def evaluate_map(labels_of, subjects, lookup, volumes):
    """One map per subject against the truth: Dice rows, volumes against the truth and the prediction, labels vanished, where the voxels went."""
    out = {}
    for subject, held in subjects.items():
        labels = labels_of(subject)
        counts = postmix.voxels(labels)
        out[subject] = dict(dice=core.segmentation_row(labels, held['reduced'].truth, held['truth4'], lookup), voxels=counts, vanished=postmix.vanished(counts, volumes[subject]['pred'], volumes[subject]['truth']),
                            volume_vs_truth=postmix.ratios(counts, volumes[subject]['truth']), volume_vs_prediction=postmix.ratios(counts, volumes[subject]['pred']), movement=sdfmix.movement(labels, held['reduced'].pred, held['reduced'].truth))
    return out


def summarize_map(rows, base):
    """A map's row: its Dice per subject and in the mean, the reductions, what vanished."""
    dice = {s: rows[s]['dice'] for s in rows}
    reduction = postmix.error_reduction(dice, base)
    mean_dice = {key: None if any(dice[s].get(key) is None for s in dice) else float(np.mean([dice[s][key] for s in dice])) for key in postmix.MEASURES}
    return dict(per_subject=rows, mean_dice=mean_dice, reduction_mean=reduction['mean'], reduction_per_subject=reduction['per_subject'], vanished={s: rows[s]['vanished'] for s in rows if rows[s]['vanished']})


def is_monotone(values):
    return all(b >= a for a, b in zip(values, values[1:]))


def find_fstar(target, reduction_of, grid_rows):
    """The f whose reduction of the mean of (1 - seg35) first reaches the target.  grid_rows: [(f, reduction)] in order, f = 0 and reduction 0 implied before them.  reduction_of: f -> reduction.  Bisection between the last grid point below the target and the next, down to F_TOL, then to F_DECIMALS decimals: the f that is pinned is the f that is evaluated."""
    points = [(0., 0.)]+list(grid_rows)
    hit = next((k for k, (_, r) in enumerate(points) if r >= target), None)
    if hit is None:
        return dict(target=target, f=None, reason=f'不可达：f = {points[-1][0]} 只减少 {points[-1][1]:.4f}')
    low, high = points[hit-1][0], points[hit][0]
    path = []
    for _ in range(MAX_BISECTIONS):
        if high-low <= F_TOL:
            break
        middle = (low+high)/2
        r = reduction_of(middle)
        path.append(dict(f=middle, reduction=r))
        if r >= target:
            high = middle
        else:
            low = middle
    f = round(high, F_DECIMALS)
    if not 0 < f < .5:
        return dict(target=target, f=None, reason=f'f = {f} 不在 (0, 1/2) 里', path=path)
    return dict(target=target, f=f, bracket=[points[hit-1][0], points[hit][0]], path=path, reduction=reduction_of(f))


def pin_template(result):
    """What run_sv.py pin-fstar writes, for the record of the probe."""
    return {arm: dict(f=row.get('f'), target=row['target'], reduction_seg35=row.get('reduction')) for arm, row in result['fstar'].items()}


def run_probe(root, *, sources=None, reader=None, workers=1, f_grid=F_GRID, log=print):
    """Reads the subjects, computes, writes the two files; returns the result.  sources, reader: tests only."""
    sources = core.Sources() if sources is None else sources
    problem = core.root_problem(root, sources.results)
    if problem:
        raise SystemExit(problem)
    root = Path(root)
    target_json, target_md = root/'analysis'/core.PROBE_JSON, root/'analysis'/core.PROBE_MD
    for path in (target_json, target_md):
        if path.exists():
            raise SystemExit(f'{path} exists; a probe is never overwritten (the lead renames it)')
    if any(not 0 < f < .5 for f in f_grid):
        raise SystemExit(f'the grid must lie in (0, 1/2): {f_grid}')
    tick = time.perf_counter()
    mi = core.core.make_inputs()
    lookup, seg4_record = mi.load_seg4_map()
    subjects = core.read_subjects(sources, core.read_subject_posterior if reader is None else reader, log)
    seconds_read = time.perf_counter()-tick
    peak_read = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
    volumes = {s: dict(pred=postmix.voxels(h['reduced'].pred), truth=postmix.voxels(h['reduced'].truth)) for s, h in subjects.items()}
    base = {s: core.segmentation_row(h['reduced'].pred, h['reduced'].truth, h['truth4'], lookup) for s, h in subjects.items()}
    rows, calls = {}, 0

    def row_at(f):
        nonlocal calls
        key = f_key(f) if round(f, F_DECIMALS) == f else repr(f)
        if key not in rows:
            calls += 1
            rows[key] = dict(f=f, **summarize_map(evaluate_map(lambda s: postmix.mix(subjects[s]['reduced'], f), subjects, lookup, volumes), base))
        return rows[key]

    tick = time.perf_counter()
    grid = [(f, row_at(f)['reduction_mean']['seg35']) for f in f_grid]
    fstar = {}
    for arm, target in core.TARGETS.items():
        fstar[arm] = find_fstar(target, lambda f: row_at(f)['reduction_mean']['seg35'], grid)
        log(f"{arm}: target {target:.4f}, f* {fstar[arm].get('f')}, reduction {fstar[arm].get('reduction')}")
    seconds_sweep = time.perf_counter()-tick
    kept = {f_key(f): row_at(f) for f in f_grid}                     # the grid and the two f*; the rows of the bisection path leave only (f, reduction)
    for row in fstar.values():
        if row.get('f') is not None:
            kept[f_key(row['f'])] = row_at(row['f'])
    tick = time.perf_counter()
    sdf = {}
    mixed = {}
    for subject, held in subjects.items():
        mixed[subject], record = sdfmix.sdfmix(held['reduced'].pred, held['reduced'].truth, SDF_FRACTIONS, workers=workers)
        log(f'{subject}: the SDF mix for comparison, {time.perf_counter()-tick:.0f} s so far; absent labels {record["absent_labels"]}')
    for f in SDF_FRACTIONS:
        sdf[str(f)] = dict(f=str(f), role='SDF 插值，只作对照，不作主臂', **summarize_map(evaluate_map(lambda s: mixed[s][f], subjects, lookup, volumes), base))
    seconds_sdf = time.perf_counter()-tick
    result = dict(schema=SCHEMA, oracle=True, role=core.ROLE, written_utc=core.utc(), subjects=list(subjects), rule='L_f = argmax_l [(1 - f) p_l + f 1(l = truth)], p = unet35_v1_ens all_folds_mean', tie_rule=postmix.TIE_RULE, targets=dict(core.TARGETS), target_tolerance=core.TARGET_TOL,
                  reduction_definition='1 - mean_subjects(1 - Dice at f) / mean_subjects(1 - Dice at f = 0); 6 subjects, not pairs', measures=list(postmix.MEASURES), f_grid=list(f_grid), grid_monotone_seg35=is_monotone([r for _, r in grid]),
                  fstar=fstar, rows=kept, sdf=sdf, base_dice=base, volumes=volumes, disagree_voxels={s: int(h['reduced'].where.size) for s, h in subjects.items()},
                  seconds=dict(read=seconds_read, sweep=seconds_sweep, sdf=seconds_sdf, map_evaluations=calls), peak_rss_gib=dict(after_read=peak_read, total=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20), workers=workers,
                  seg4_map=seg4_record['table_sha256'], script_sha256=core.sha_file(Path(__file__)), run_sv_sha256=core.sha_file(core.HERE/'run_sv.py'), postmix_sha256=core.sha_file(core.HERE/'postmix.py'), sdfmix_sha256=core.sha_file(core.HERE/'sdfmix.py'), subject_records={s: h['record'] for s, h in subjects.items()})
    result['pin_template'] = pin_template(result)
    target_json.parent.mkdir(parents=True, exist_ok=True)
    core.write_json(target_json, result)
    target_md.write_text(summary_text(result), encoding='utf-8')
    return result


def fmt(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def percent(value):
    return '—' if value is None else f'{100*value:.1f}%'


def table_rows(name, row):
    r, d = row['reduction_mean'], row['mean_dice']
    worst = [min((x for x in cells['volume_vs_truth'][1:] if x is not None), default=None) for cells in row['per_subject'].values()]
    worst = [x for x in worst if x is not None]
    gone = sum(len(v) for v in row['vanished'].values())
    return (f"| {name} | {fmt(d['seg35'])} / {fmt(d['seg4'])} | {percent(r['seg35'])} | {percent(r['seg4'])} | {percent(r['small_six'])} | {percent(r['big_six'])} | {percent(r['other_23'])} | {percent(r['cortex'])} | {percent(r['white_matter'])} | {gone} | {fmt(min(worst) if worst else None, 3)} |")


def summary_text(result):
    lines = ['# SV 探针：分割误差减少多少（代码写出，不是配准结果）', '', '开发集 6 名受试者，单次。oracle：真值只用来算这些比较，结果只在隔离根目录，不写标签场，不进结果表。减少比例 = 1 − 6 人平均的 (1 − Dice) ÷ f = 0 时的同一个量。皮层、白质是 seg4 标签 1、3；小六个、大六个、其余 23 个是 seg35 的分组。', '',
             f"规则：{result['rule']}。平局：{result['tie_rule']}。f = 0 等于预测 argmax，f > 1/2 等于真值，所以 f 只取 (0, 1/2)。写出 {result['written_utc']}。", '', '## f*（6 人平均的 1 − seg35）', '', '| 臂 | 目标 | f* | 实际减少 | 差 | 允许 |', '|---|---|---|---|---|---|']
    for arm, row in result['fstar'].items():
        f = row.get('f')
        lines.append(f"| {arm} | {percent(row['target'])} | {fmt(f)} | {percent(row.get('reduction'))} | {fmt(None if f is None else row['reduction']-row['target'])} | ≤ {result['target_tolerance']} |" if f is not None else f"| {arm} | {percent(row['target'])} | 没有 | {row.get('reason')} | | |")
    lines += ['', '**这些数要由总负责人钉入 configs/bayesreg/sv_fstar_v1.json 并写进 plan.md 的修订之后才开 GPU**（run_sv.py pin-fstar）。', '', '## 后验混合，f 上的 6 人平均', '', '| f | seg35 / seg4 | 减少 seg35 | seg4 | 小六个 | 大六个 | 其余 23 个 | 皮层 | 白质 | 消失的标签（人次） | 最小体积比（对真值） |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for key, row in sorted(result['rows'].items(), key=lambda item: item[1]['f']):
        lines.append(table_rows(key, row))
    lines += ['', f"seg35 的减少比例在网格上单调：{result['grid_monotone_seg35']}。", '', '## SDF 插值（只作对照）', '', '| f | seg35 / seg4 | 减少 seg35 | seg4 | 小六个 | 大六个 | 其余 23 个 | 皮层 | 白质 | 消失的标签（人次） | 最小体积比（对真值） |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for key, row in result['sdf'].items():
        lines.append(table_rows(key, row))
    lines += ['', '## 逐人（f*）', '', '| 臂 | 受试者 | seg35 / seg4 | 减少 seg35 | seg4 | 小六个 | 大六个 | 其余 23 个 | 皮层 | 白质 | 消失的标签 | 分歧里改到真值 |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for arm, star in result['fstar'].items():
        if star.get('f') is None:
            continue
        row = result['rows'][f_key(star['f'])]
        for subject, cells in row['per_subject'].items():
            r, d, m = row['reduction_per_subject'][subject], cells['dice'], cells['movement']
            lines.append(f"| {arm} | {subject} | {fmt(d['seg35'])} / {fmt(d['seg4'])} | {percent(r['seg35'])} | {percent(r['seg4'])} | {percent(r['small_six'])} | {percent(r['big_six'])} | {percent(r['other_23'])} | {percent(r['cortex'])} | {percent(r['white_matter'])} | {cells['vanished'] or '无'} | {m['equal_truth']}/{m['disagree']} |")
    seconds, peak = result['seconds'], result['peak_rss_gib']
    lines += ['', '## 时间与内存（本次实测）', '', f"读后验与真值 {seconds['read']:.0f} s，后验混合的扫描与二分 {seconds['sweep']:.0f} s（{seconds['map_evaluations']} 个 f），SDF 对照 {seconds['sdf']:.0f} s（{result['workers']} 个线程）；峰值 RSS {peak['total']:.2f} GiB（读完之后 {peak['after_read']:.2f} GiB）。", '', '每个标签在每个 f 上的体积相对真值、预测的比值，逐人在 probe_posterior.json 的 rows 里（voxels、volume_vs_truth、volume_vs_prediction、vanished）。', '',
              '局限：后验混合优先改掉低置信的错误，高置信的系统性错误留到最后；真实网络的改进不一定是这个形态。〔推断〕先改掉的低置信错误更可能是两幅图不一致的部分，留到最后的系统性错误在两幅图上可能部分抵消，所以同样的误差减少，配准终态的涨幅可能高于真实网络，曲线可能偏乐观：读"够"（尤其"减三分之一就够"）要打这个折扣，"不够"相对可靠。']
    return '\n'.join(lines)+'\n'
