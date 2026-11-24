"""RP's reading (runs/20261105-rp/任务书.md, plan.md; by code; labels for the lead and BZ, not conclusions).  R1 is rw_v1, read by decide_fx's own functions; R2 and R3 are rp_v1/r2 and rp_v1/r3, read by the same functions of decide_fl through another instance each (decide_fx._records).  decide_fx, decide_fl and their batches are not edited.

What a pair of a repetition is (the evaluation's certificate, from the field, configuration C, alpha = 1; the fuse from repair.json's rounds):
    certified       the repaired field passes, the 1200 s fuse was not hit
    fuse            the fuse was hit; the certificate passed or not (both recorded)
    not_certified   the repaired field does not pass, the fuse not hit
    not_read        not run, a process that left no repair.json, not evaluated, or not rw_v1's own repair: it says nothing about the repair (FL review, blocking 1)
Two conventions of a failure (the lead's ruling, about 9 pm CST 11-05; D74 of the C22 draft, for BZ):
    strict (the main reading, the brief)   fuse or not_certified: a fuse hit is a failure even when the certificate then passed
    code (reported with it)                not_certified, or fuse with the certificate not passed: what passed is passed (run_fx.repair_one's status)
The method's field of a pair in a repetition, by 2c(i) with two tiers (D75, for BZ): the repaired field when not a failure; on a failure configuration C's field of d0_v1 (A0) when d0_v1's records qualify it (rp_core.fallback_status; no certificate computed again), otherwise the identity u = 0 evaluated into rp_v1/id; none when neither is there.

Reading 1, the certificates of the 19 x 3 repairs (each convention):
    a check of the controls fails (configuration, source field, code, mark)  不可判
    any of the 57 not read                                                   不可判
    no failure                                                               3 次都 19/19
    otherwise                                                                有失败 m/57 (the pairs and repetitions named)
Reading 2, the method's mean seg35 over the 19 pairs, its range over the 3 repetitions (max - min) (each convention):
    a check fails, or a repetition without the method's field on every pair  不可判
    range <= .001                                                            方法级均值的极差 ≤ .001
    range > .005                                                             方法级均值的极差 > .005
    otherwise                                                                方法级均值的极差在 .001 与 .005 之间
Two more lines, only described, in no reading (revision 1): the method's means with a crashed repair (no repair.json) taken as a failure, so through the fallback (the RP review, minor 2; D74 of the C22 draft counts a run error as a failure); and the decomposition of the range, how much comes from the pairs taken from a fallback and how much from the repaired fields themselves (important 1).
Only descriptions: no reading changes the method.  "修复后" means only over the repairs that pass the certificate; an uncertified attempt is listed and enters no mean.  The certificate of a substituted A0 field is D0's run's own (the frozen configuration C, CPU FP64), not recomputed by RP's evaluation (minor 11; D75 for BZ).
"""
import csv
import json
import math
from pathlib import Path

from decide_ef import read_json, set_aside
import decide_fd2 as fd2_reading
import decide_fl as fl_reading
import decide_fx as fx_reading
import fd2_core as fd2
import fl_core as fl
import fx_core as fx
import mc0_core as core
import rp_core as rp
from fx_core import ALL_PAIRS, RW
from rp_core import BATCHES, REPEATS, RP

REPS = (1,)+REPEATS
N = len(ALL_PAIRS)*len(REPS)
UNDECIDABLE = fx_reading.UNDECIDABLE
CERTIFIED, FUSE, NOT_CERTIFIED, NOT_READ = 'certified', 'fuse', 'not_certified', 'not_read'
STRICT, CODE = 'strict', 'code'
HOWS = (STRICT, CODE)
HOW_WORDS = {STRICT: '严格口径（撞保险丝即失败，主读数）', CODE: '代码口径（证书过即算过）'}
READ_STABLE = '3 次都 19/19'
STABLE_WORDS = '在这 19 对上修复稳定（3 次抽样）'
SPREAD_SMALL, SPREAD_LARGE = .001, .005
READ_SMALL, READ_MIDDLE, READ_LARGE = '方法级均值的极差 ≤ .001', '方法级均值的极差在 .001 与 .005 之间', '方法级均值的极差 > .005'
SPREAD_WORDS = {READ_SMALL: '修复的非确定对方法级 19 对 seg35 均值的影响在 .001 以内（3 次抽样）',
                READ_MIDDLE: '看得到，但小于 19 对开发集的最小可检测效应（约 .005）（3 次抽样）',
                READ_LARGE: '达到 19 对开发集的最小可检测效应（约 .005）：一次 19 对的读数代表不了这条线（3 次抽样）'}
MEASURES = ('seg35', 'seg4', 'sdlogj_official')         # the brief's three
METHOD_MEASURES = rp.FALLBACK_MEASURES
TIMES = ('rounds', 'seconds_repair', 'seconds_warmup', 'task_clock')
BZ_SECONDS = fx_reading.BZ_SECONDS
CLOCK_COLUMN = fx_reading.CLOCK_COLUMN
REPAIR, A0, IDENTITY = 'repair', 'A0', 'identity'
mean = fl_reading.mean
_f, _signed = fl_reading._f, fl_reading._signed
NOTE = ('开发集单次观测的三次重复：同样 19 个 Hx24@600 场（fd2_v1，rw_sources_v1.json 钉住）、同样的修复配置（RF05，wall_seconds 1200 当保险丝）、同样的代码（R1 是 rw_v1 的 85fc80d；R2、R3 的提交核过修复与评测的文件和 85fc80d 逐字节相同）。通道来自 unet35_v1_ens 的预测（oracle: false），真值标签只进评测。'
        '**不是 F_B，没有经认证的下降，只有终态过证书；不进结果表，读数只描述，不改方法。** 3 次是抽样，不是分布。"修复后"只算过证书的修复，未认证的尝试单列。'
        '撞保险丝算不算失败待 BZ 定（C22 草案 D74）：主读数按严格口径（撞保险丝即失败），同时报代码口径（证书过即算过）。方法级（2c(i)，退路两层待 BZ 定，D75）：失败的对先代入配置 C 的场（d0_v1 A0，钉住，按 D0 的记录过证书，不重跑证书），配置 C 的场缺失或记录不合格时代入恒等场 u = 0（J ≡ 1）。**代入的 A0 证书是 D0 运行自己算的（冻结的配置 C，CPU FP64），不是 RP 的评测重算的**（D75 待 BZ）。读数 2 的中间带和上档主要来自代入的次数：一对代入 A0 就让 19 对均值移 .0016–.0049（开发集记录），修复本身的非确定在均值层面约 1e-4；分解见每种口径下的"分解"一行。')

records = {rep: fx_reading._records(BATCHES[rep]) for rep in REPEATS}


# ---- the rows -------------------------------------------------------------------------------------------------------------------------------
def extended(row, d):
    """A repair row with what the controls and the tables need from its files: the configuration, the source, the code, the attempt and the repaired field's hashes."""
    if row is None:
        return None
    d = Path(d)
    repair, metrics = read_json(d/'repair.json') or {}, read_json(d/'repair_metrics.json') or {}
    attempt = ((((metrics.get('after') or {}).get('certificate') or {}).get('attempts') or [{}])[0]) or {}
    data = repair.get('data') or {}
    return dict(row, directory=str(d), config=repair.get('config'), code_sha=repair.get('code_sha'), source_tapered_u_sha256=(repair.get('source') or {}).get('tapered_u_sha256'), attempt_sha256=(repair.get('attempt') or {}).get('sha256'), repaired_u_sha256=metrics.get('repaired_u_sha256'),
                has_repair_json=(d/'repair.json').is_file(), metrics_error=read_json(d/'metrics_error.json') is not None,
                repair_certificate_config=repair.get('certificate_config'), frozen_commit=repair.get('frozen_commit'), pair_sha256=data.get('pair_sha256'), manifest_sha256=data.get('manifest_sha256'), env=repair.get('env'), device_cap=repair.get('device_cap'),
                evaluator_sha256=(metrics.get('evaluator') or {}).get('sha256'), evaluation_certificate=dict(config=attempt.get('config'), certificate_version=attempt.get('certificate_version')) if metrics else None)


def r1_rows(results=None):
    """R1: rw_v1's 19 repairs, by decide_fx.rw_rows.  A pair that rw_rows takes from fd2_v1 (only if RW's control had been bit for bit; it was not) is not R1: not read."""
    root = RP.r1_root(results)
    rows, sources = fx_reading.rw_rows(root, results)
    out = {}
    for pair in ALL_PAIRS:
        d = RW.run_dir(root, RW.cell_of(pair))
        out[pair] = extended(rows[pair], d) if sources[pair] == RW.ROOT_NAME else dict(not_r1=True, source=sources[pair], status=f'{sources[pair]}', metrics=False, has_repair_json=False)
    return out, {core.short_name(p): s for p, s in sources.items()}


def rep_rows(root, rep):
    """R2 or R3: {pair: row or None}."""
    batch, where = BATCHES[rep], RP.rep_root(root, rep)
    out = {}
    for cell in batch.plan():
        d = batch.run_dir(where, cell)
        out[cell.pair] = extended(fx_reading.with_wall(records[rep].repair_row(where, cell), d), d)
    return out


def all_rows(root, results=None):
    r1, sources = r1_rows(results)
    rows = {1: r1}
    for rep in REPEATS:
        rows[rep] = rep_rows(root, rep)
    return rows, sources


def kind(row):
    """(kind, why) of one repair of one repetition."""
    if row is None:
        return NOT_READ, '没有运行'
    if row.get('not_r1'):
        return NOT_READ, f"不是 rw_v1 自己的修复（{row.get('source')}）"
    if not row.get('metrics'):
        if not row.get('has_repair_json'):
            return NOT_READ, f"运行失败，没有 repair.json（状态 {row.get('status')}）"
        return NOT_READ, '评测出错（metrics_error.json）' if row.get('metrics_error') else '没有评测'
    if row.get('wall_hit'):
        return FUSE, '撞 1200 s 保险丝，' + ('证书过了' if row.get('certified_at_1') else '证书没过')
    if row.get('certified_at_1'):
        return CERTIFIED, None
    return NOT_CERTIFIED, '没有通过证书的状态'


def crashed(row):
    """A repair process that ran and left no repair.json (rc != 0, a device out of memory among them): not read for the readings; a failure for the described line."""
    return row is not None and not row.get('not_r1') and not row.get('metrics') and not row.get('has_repair_json')


def failed(row, how):
    """Is this repair a failure of 2c(i) under this convention?  A repair not read is no failure."""
    k = kind(row)[0]
    if how == STRICT:
        return k in (FUSE, NOT_CERTIFIED)
    return k == NOT_CERTIFIED or (k == FUSE and not row.get('certified_at_1'))


def identity_rows(root):
    """{pair: what rp_v1/id holds of the identity u = 0}.  Usable: certified at alpha = 1, oracle false, its metrics there."""
    out = {}
    for pair in ALL_PAIRS:
        d = RP.id_dir(RP.id_root(root), pair)
        m, e = read_json(d/'identity_metrics.json'), read_json(d/'identity_error.json')
        if m is None:
            out[pair] = dict(pair=core.short_name(pair), usable=False, why=f"评测出错：{(e or {}).get('error', '')[:300]}" if e else '没有评测', metrics=None)
            continue
        usable = m.get('certified_at_1') is True and m.get('oracle') is False and bool(m.get('metrics'))
        out[pair] = dict(pair=core.short_name(pair), usable=usable, why=None if usable else '没有通过证书或没有指标', metrics=m.get('metrics'))
    return out


def fallback_rows(results=None, document=None, stage=None):
    """{pair: rp_core.fallback_status}: configuration C's field qualified by d0_v1's records or not."""
    return {pair: rp.fallback_status(pair, results=results, document=document, stage=stage) for pair in ALL_PAIRS}


def method_fields(rows, fallback, identity, how, crash_as_failure=False):
    """{rep: {pair: dict(source, metrics, why)}}: what the method (2c(i), two tiers) outputs for each pair of each repetition under this convention.  crash_as_failure: the described line, a crashed repair goes to the fallback."""
    out = {}
    for rep in REPS:
        out[rep] = {}
        for pair in ALL_PAIRS:
            row = rows[rep][pair]
            k, why = kind(row)
            if k == NOT_READ and not (crash_as_failure and crashed(row)):
                out[rep][pair] = dict(source=None, metrics=None, why=why)
            elif k != NOT_READ and not failed(row, how):
                out[rep][pair] = dict(source=REPAIR, metrics=row['after'], why=None)
            elif fallback[pair]['qualified']:
                out[rep][pair] = dict(source=A0, metrics=fallback[pair]['metrics'], why=why)
            elif identity[pair]['usable']:
                out[rep][pair] = dict(source=IDENTITY, metrics=identity[pair]['metrics'], why=f"{why}；配置 C 的场不合格（{fallback[pair]['why']}），代入恒等")
            else:
                out[rep][pair] = dict(source=None, metrics=None, why=f"{why}；配置 C 的场不合格（{fallback[pair]['why']}），恒等场也没有（{identity[pair]['why']}）")
    return out


# ---- the controls ------------------------------------------------------------------------------------------------------------------------------
def batch_sha(root):
    return (read_json(Path(root)/'batch.json') or {}).get('code_sha')


def controls(rows, root, sources=None):
    """The configuration of every repair that left repair.json (RW's RF05 with wall_seconds 1200, one and the same), its source (the pinned tapered field), its code (R1 85fc80d, R2 and R3 the batch's commit).  A wrong oracle mark writes no reading."""
    pins = RW.sources(sources)
    marked = sorted(r['run_id'] for rep in REPS for r in rows[rep].values() if r is not None and r.get('metrics') and r.get('oracle') is not False)
    if marked:
        raise SystemExit(f'repair.json is not oracle false for {marked}; nothing written')
    configs, problems, code = {}, [], {}
    same = {name: {} for name in ('repair_certificate_config', 'frozen_commit', 'evaluator_sha256', 'evaluation_certificate')}
    by_pair, described = {}, {name: {} for name in ('env', 'device_cap')}
    sha = batch_sha(root)
    for rep in REPS:
        for pair, r in rows[rep].items():
            if r is None or not r.get('has_repair_json'):
                continue
            short = core.short_name(pair)
            configs.setdefault(json.dumps(r['config'], sort_keys=True), []).append(f'R{rep} {short}')
            if r.get('source_tapered_u_sha256') != (pins.get(short) or {}).get('tapered_u_sha256'):
                problems.append(f'R{rep} {short}: the source field is not the pinned one')
            for name in same:
                if name in ('evaluator_sha256', 'evaluation_certificate') and not r.get('metrics'):
                    continue
                same[name].setdefault(json.dumps(r.get(name), sort_keys=True), []).append(f'R{rep} {short}')
            for name in described:
                described[name].setdefault(json.dumps(r.get(name), sort_keys=True), []).append(f'R{rep} {short}')
            by_pair.setdefault(short, set()).add((r.get('pair_sha256'), r.get('manifest_sha256')))
            want = rp.BASE_COMMIT if rep == 1 else sha
            code.setdefault(f'R{rep}', set()).add(r.get('code_sha'))
            if r.get('code_sha') != want:
                problems.append(f"R{rep} {short}: code {r.get('code_sha')}, not {want}")
    if len(configs) > 1:
        problems.append(f'the repairs do not share one configuration: {sorted(configs)}')
    config = json.loads(next(iter(configs))) if configs else None
    if config is not None and config.get('wall_seconds') != fx.FUSE_SECONDS:
        problems.append(f"wall_seconds {config.get('wall_seconds')}, not {fx.FUSE_SECONDS}")
    for name, values in same.items():
        if len(values) > 1:
            problems.append(f'the repairs do not share one {name}: {sorted(values)}')
    for short, values in sorted(by_pair.items()):
        if len(values) > 1:
            problems.append(f'{short}: the repetitions read different data (pair_sha256, manifest_sha256): {sorted(map(str, values))}')
    return dict(config=config, n_configs=len(configs), code={k: sorted(str(s) for s in v) for k, v in code.items()}, batch_code_sha=sha, base_commit=rp.BASE_COMMIT, problems=problems,
                same={name: [json.loads(v) for v in values] for name, values in same.items()}, described={name: {v: sorted(set(x.split()[0] for x in where)) for v, where in values.items()} for name, values in described.items()})


# ---- the tables ---------------------------------------------------------------------------------------------------------------------------------
def spread(values):
    """dict(values in the order R1, R2, R3 with None where there is none, n, mean, range, min, max)."""
    present = [v for v in values if v is not None and isinstance(v, (int, float)) and math.isfinite(v)]
    return dict(values=list(values), n=len(present), mean=mean(present), range=None if not present else max(present)-min(present), min=min(present, default=None), max=max(present, default=None))


def full_mean(values):
    """The mean when every value is there and finite, None otherwise: a method's mean is over all 19 pairs or it is not."""
    values = list(values)
    return mean(values) if values and all(v is not None and math.isfinite(v) for v in values) else None


def per_pair(pair, rows, methods):
    """One pair over the 3 repetitions: certificates, the fuse, the brief's three measures of the certified repairs and of the method (both conventions), rounds, seconds, the task clock, the certified bound, the repaired fields' hashes."""
    by = {rep: rows[rep][pair] for rep in REPS}
    kinds = {rep: kind(r) for rep, r in by.items()}
    read = {rep: r for rep, r in by.items() if kinds[rep][0] != NOT_READ}
    out = dict(pair=core.short_name(pair), kinds={f'R{rep}': kinds[rep][0] for rep in REPS}, why={f'R{rep}': kinds[rep][1] for rep in REPS if kinds[rep][1]}, n_read=len(read), n_certified=sum(bool(r['certified_at_1']) for r in read.values()), n_fuse=sum(kinds[rep][0] == FUSE for rep in REPS),
               n_failures={how: sum(failed(by[rep], how) for rep in REPS) for how in HOWS})
    for m in MEASURES:
        out[m] = spread([r['after'][m] if r is not None and r.get('metrics') and r.get('certified_at_1') else None for r in by.values()])
        for how in HOWS:
            out[f'method_{m}_{how}'] = spread([None if methods[how][rep][pair]['metrics'] is None else methods[how][rep][pair]['metrics'].get(m) for rep in REPS])
    for t in TIMES:
        out[t] = spread([None if r is None or not r.get('has_repair_json') else r.get(t) for r in by.values()])
    bounds = [r['min_lower_bound_after'] if r is not None and r.get('metrics') else None for r in by.values()]
    out['min_lower_bound_after'] = spread(bounds)
    positive = [b for b in bounds if b is not None and b > 0]
    out['lower_bound_ratio'] = max(positive)/min(positive) if len(positive) == len(REPS) else None
    hashes = [r.get('repaired_u_sha256') if r is not None else None for r in by.values()]
    out['repaired_u_sha256'] = hashes
    out['n_distinct_repaired_fields'] = len({h for h in hashes if h is not None})
    out['method_sources'] = {how: [methods[how][rep][pair]['source'] for rep in REPS] for how in HOWS}
    out['attempts'] = [dict(rep=f'R{rep}', seg35=r['after']['seg35'], seg4=r['after']['seg4'], n_infeasible_after=r.get('n_infeasible_after'), min_lower_bound_after=r.get('min_lower_bound_after')) for rep, r in by.items() if r is not None and r.get('metrics') and not r.get('certified_at_1')]
    return out


def method_summary(method):
    """One repetition's method under one convention: complete (every pair has a field), which pairs took A0 or the identity, the means over the 19 pairs."""
    complete = all(method[p]['source'] is not None for p in ALL_PAIRS)
    return dict(complete=complete, substituted_a0=[core.short_name(p) for p in ALL_PAIRS if method[p]['source'] == A0], substituted_identity=[core.short_name(p) for p in ALL_PAIRS if method[p]['source'] == IDENTITY],
                missing=[dict(pair=core.short_name(p), why=method[p]['why']) for p in ALL_PAIRS if method[p]['source'] is None],
                **{m: full_mean(method[p]['metrics'].get(m) for p in ALL_PAIRS) if complete else None for m in METHOD_MEASURES})


def per_rep(rep, rows, methods, root, results=None, crash_methods=None):
    """One repetition: what was read, certified, failed under each convention; the means of the certified repairs; the method's means by 2c(i) (on all 19 pairs, or none); rounds and the task clock of every repair read."""
    kinds = {pair: kind(r) for pair, r in rows.items()}
    read = {p: r for p, r in rows.items() if kinds[p][0] != NOT_READ}
    clocks = [r['task_clock'] for r in read.values() if r.get('task_clock') is not None]
    where = RP.r1_root(results) if rep == 1 else RP.rep_root(root, rep)
    return dict(rep=f'R{rep}', root=str(where), n=len(ALL_PAIRS), n_read=len(read), n_certified=sum(bool(r['certified_at_1']) for r in read.values()), n_fuse=sum(kinds[p][0] == FUSE for p in ALL_PAIRS),
                failures={how: [dict(pair=core.short_name(p), kind=kinds[p][0], why=kinds[p][1]) for p in ALL_PAIRS if failed(rows[p], how)] for how in HOWS}, not_read=[dict(pair=core.short_name(p), why=kinds[p][1]) for p in ALL_PAIRS if kinds[p][0] == NOT_READ],
                repaired=fx_reading.certified_summary(rows), n_two_rounds_or_more=sum((r.get('rounds') or 0) >= 2 for r in read.values()), mean_task_clock_read=mean(clocks), max_task_clock_read=max(clocks, default=None), clock_90=dict(n_within=sum(c <= BZ_SECONDS for c in clocks), n=len(clocks)),
                mean_seconds_repair_read=mean(r.get('seconds_repair') for r in read.values()), method={how: method_summary(methods[how][rep]) for how in HOWS},
                n_crashed=sum(crashed(r) for r in rows.values()), method_crash_as_failure={how: method_summary((crash_methods or methods)[how][rep]) for how in HOWS})


# ---- the readings -------------------------------------------------------------------------------------------------------------------------------
def read_certificates(rows, problems=(), how=STRICT):
    """(reading, reason, statement) of reading 1 under one convention."""
    if problems:
        return UNDECIDABLE, f"核对不过：{'；'.join(problems)}", None
    not_read = [(rep, p, kind(rows[rep][p])[1]) for rep in REPS for p in ALL_PAIRS if kind(rows[rep][p])[0] == NOT_READ]
    if not_read:
        return UNDECIDABLE, '这些修复没有读到，和修复本身无关，不能读成失败：' + '、'.join(f'{core.short_name(p)} R{rep}（{why}）' for rep, p, why in not_read), None
    failures = [(rep, p, kind(rows[rep][p])[1]) for rep in REPS for p in ALL_PAIRS if failed(rows[rep][p], how)]
    if not failures:
        return READ_STABLE, None, STABLE_WORDS
    named = '、'.join(f'{core.short_name(p)} R{rep}（{why}）' for rep, p, why in failures)
    return f'有失败 {len(failures)}/{N}', named, f'有失败：{named}；方法级均值按第 4 条（失败的对先代入配置 C 的场，配置 C 不合格时代入恒等）算'


def read_spread(reps, problems=(), how=STRICT):
    """(reading, reason, statement, range) of reading 2 under one convention: the method's mean seg35 of each repetition."""
    if problems:
        return UNDECIDABLE, f"核对不过：{'；'.join(problems)}", None, None
    methods = [(r['rep'], r['method'][how]) for r in reps]
    missing = [(rep, m) for rep, m in methods if not m['complete']]
    if missing:
        return UNDECIDABLE, '这些重复没有每一对的方法级场：' + '；'.join(f"{rep}（{'、'.join(x['pair']+' '+x['why'] for x in m['missing'])}）" for rep, m in missing), None, None
    values = [m['seg35'] for _, m in methods]
    if not all(v is not None and math.isfinite(v) for v in values):
        return UNDECIDABLE, f'方法级均值不是有限值（{values}）', None, None
    width = max(values)-min(values)
    reading = READ_SMALL if width <= SPREAD_SMALL else READ_LARGE if width > SPREAD_LARGE else READ_MIDDLE
    return reading, None, SPREAD_WORDS[reading], width


def decomposition(method):
    """Only described (revision 1, important 1): {rep: {pair: ...}} of one convention.  The repaired part: the pairs whose method field is the repaired one in all 3 repetitions, their mean seg35 per repetition and its range.  The fallback's part: per repetition, the sum over the pairs taken from a fallback of (that value - the pair's mean repaired seg35 where it was repaired) / 19; None when such a pair has no repaired value anywhere."""
    common = [p for p in ALL_PAIRS if all(method[rep][p]['source'] == REPAIR for rep in REPS)]
    shifts = {}
    for rep in REPS:
        shift, known = 0., True
        for p in ALL_PAIRS:
            if method[rep][p]['source'] in (A0, IDENTITY):
                own = [method[r][p]['metrics']['seg35'] for r in REPS if method[r][p]['source'] == REPAIR]
                if not own:
                    known = False
                    continue
                shift += (method[rep][p]['metrics']['seg35']-mean(own))/len(ALL_PAIRS)
        shifts[f'R{rep}'] = shift if known else None
    common_means = spread([full_mean(method[rep][p]['metrics']['seg35'] for p in common) if common else None for rep in REPS])
    return dict(n_common=len(common), common_seg35=common_means, fallback_shift_seg35=shifts, note='只描述：3 次都用修复场的对，其 seg35 均值的极差是修复本身的非确定；代入的分量是（代入值 − 这一对在别的次修复场的均值）÷ 19')


def readings(rows, reps, problems, how):
    reading, reason, statement = read_certificates(rows, problems, how)
    spread_reading, spread_reason, spread_statement, width = read_spread(reps, problems, how)
    crash = [r['method_crash_as_failure'][how] for r in reps]
    return dict(convention=how, words=HOW_WORDS[how], reading=reading, reason=reason, statement=statement, reading_spread=spread_reading, reason_spread=spread_reason, statement_spread=spread_statement, method_seg35_range=width,
                n_a0_used=sum(len(r['method'][how]['substituted_a0']) for r in reps), n_identity_used=sum(len(r['method'][how]['substituted_identity']) for r in reps), method_means={m: spread([r['method'][how][m] for r in reps]) for m in METHOD_MEASURES},
                crash_as_failure=dict(described_only=True, n_crashed=sum(r['n_crashed'] for r in reps), seg35=spread([c['seg35'] for c in crash]), seg4=spread([c['seg4'] for c in crash]), n_a0_used=sum(len(c['substituted_a0']) for c in crash), n_identity_used=sum(len(c['substituted_identity']) for c in crash)))


def code_record(root):
    """What wrote this reading: the sha256 of the files that hold its rules, and the batch's commit."""
    files = dict(decide_rp=Path(__file__), rp_core=Path(rp.__file__), decide_fx=Path(fx_reading.__file__), fx_core=Path(fx.__file__), decide_fl=Path(fl_reading.__file__), fl_core=Path(fl.__file__), decide_fd2=Path(fd2_reading.__file__), fd2_core=Path(fd2.__file__), rp_fallback=rp.FALLBACK_FILE, rw_sources=RW.SOURCES_FILE)
    return dict({f'{k}_sha256': core.sha_file(v) for k, v in files.items()}, batch_code_sha=batch_sha(root))


def decide(root, stops=None, results=None, sources=None, document=None, stage=None):
    root = Path(root)
    rows, r1_sources = all_rows(root, results)
    check = controls(rows, root, sources)
    fallback = fallback_rows(results, document, stage)
    identity = identity_rows(root)
    methods = {how: method_fields(rows, fallback, identity, how) for how in HOWS}
    crash_methods = {how: method_fields(rows, fallback, identity, how, crash_as_failure=True) for how in HOWS}
    reps = [per_rep(rep, rows[rep], methods, root, results, crash_methods) for rep in REPS]
    strict, code = readings(rows, reps, check['problems'], STRICT), readings(rows, reps, check['problems'], CODE)
    for r, how in ((strict, STRICT), (code, CODE)):
        r['decomposition'] = decomposition(methods[how])
    pairs = [per_pair(p, rows, methods) for p in ALL_PAIRS]
    certified_means = {m: spread([r['repaired'][m] if r['repaired']['n_certified'] == len(ALL_PAIRS) else None for r in reps]) for m in METHOD_MEASURES}
    return dict(schema='bayesreg.rp.decision.v1', code=code_record(root), is_conclusion=False, next_step='交总负责人与 BZ（撞保险丝的口径 D74、退路二 D75 待 BZ 定）', oracle=False, note=NOTE,
                reading=strict['reading'], reason=strict['reason'], statement=strict['statement'], reading_spread=strict['reading_spread'], reason_spread=strict['reason_spread'], statement_spread=strict['statement_spread'], method_seg35_range=strict['method_seg35_range'],
                strict=strict, code_convention=code,
                rules=dict(failure_strict='fuse or not_certified (2c(i), the brief; D74)', failure_code='not_certified, or fuse with the certificate not passed', not_read='no failure: 不可判', spread_small=SPREAD_SMALL, spread_large=SPREAD_LARGE, wall_seconds=fx.FUSE_SECONDS, repair_eps=RW.REPAIR_EPS,
                           fallback="first d0_v1 A0 pg_weak__four__90s/final_native.npz, pinned, qualified by D0 records; its certificate is D0's run's own (frozen configuration C, CPU FP64), not computed again; then the identity u = 0, evaluated into rp_v1/id (D75)",
                           described_only='crash_as_failure, decomposition'),
                arms_stopped=stops or {}, n_repairs=N, per_rep=reps, certified_means=certified_means, per_pair=pairs,
                fallback=dict(per_pair=list(fallback.values()), n_qualified=sum(v['qualified'] for v in fallback.values()), not_qualified=[dict(pair=v['pair'], why=v['why']) for v in fallback.values() if not v['qualified']]),
                identity=dict(per_pair=list(identity.values()), n_usable=sum(v['usable'] for v in identity.values())), controls=check, r1_sources=r1_sources), rows, methods


# ---- the summary ------------------------------------------------------------------------------------------------------------------------------
def _three(s, nd=4):
    return ' / '.join('—' if v is None else _f(v, nd) for v in s['values'])


def _bound(v):
    return '—' if v is None else f'{v:.2g}'


def reading_lines(r):
    L = [f"**{r['words']}** 读数 1（证书）：**{r['reading']}**" + (f"：{r['statement']}" if r['statement'] and r['reading'] == READ_STABLE else ''),
         f"读数 2（方法级 19 对 seg35 均值，3 次的极差）：**{r['reading_spread']}**" + (f"（{_f(r['method_seg35_range'])}）：{r['statement_spread']}" if r['statement_spread'] else '')]
    if r['reason']:
        L.append(f"读数 1 的原因：{r['reason']}")
    if r['reason_spread']:
        L.append(f"读数 2 的原因：{r['reason_spread']}")
    mm = r['method_means']
    L.append(f"方法级 19 对均值（R1 / R2 / R3）：seg35 {_three(mm['seg35'])}（极差 {_f(mm['seg35']['range'])}），seg4 {_three(mm['seg4'])}（极差 {_f(mm['seg4']['range'])}），官方 SDlogJ {_three(mm['sdlogj_official'])}（极差 {_f(mm['sdlogj_official']['range'])}）；代入配置 C {r['n_a0_used']} 次，**代入恒等 {r['n_identity_used']} 次**。")
    dc = r.get('decomposition')
    if dc is not None:
        shifts = ' / '.join('—' if v is None else _signed(v) for v in dc['fallback_shift_seg35'].values())
        L.append(f"分解（只描述）：3 次都用修复场的 {dc['n_common']} 对，seg35 均值 {_three(dc['common_seg35'])}，极差 {_f(dc['common_seg35']['range'])}（修复本身）；代入使各次 19 对均值移动 {shifts}（R1 / R2 / R3，代入值减这一对在别的次修复场的均值，÷ 19）。")
    c = r['crash_as_failure']
    L.append(f"崩溃按失败代入（只描述，不进判定）：崩溃 {c['n_crashed']} 次；方法级 19 对 seg35 {_three(c['seg35'])}（极差 {_f(c['seg35']['range'])}），seg4 {_three(c['seg4'])}；代入配置 C {c['n_a0_used']} 次、恒等 {c['n_identity_used']} 次。")
    return L


def summary_text(d):
    L = ['# RP 读数（由 decide_rp.py 写出）', '', '（不是结论，下一步交总负责人与 BZ）', '', d['note'], '']
    for r in (d['strict'], d['code_convention']):
        L += reading_lines(r)+['']
    L += ['## 每次重复', '', f'| 次 | 读到 | 过证书 | 撞保险丝 | 失败 严格 / 代码 | 修复后 seg35 / seg4（只算过证书的） | 方法级 seg35 / seg4 严格 | 方法级 seg35 / seg4 代码 | 代入 A0 / 恒等（严格） | 两轮以上 | 任务时钟 平均 / 最长 s | {CLOCK_COLUMN} |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in d['per_rep']:
        s, ms, mc = r['repaired'], r['method'][STRICT], r['method'][CODE]
        L.append(f"| {r['rep']} | {r['n_read']}/{r['n']} | {r['n_certified']}/{r['n']} | {r['n_fuse']} | {len(r['failures'][STRICT])} / {len(r['failures'][CODE])} | {_f(s['seg35'])} / {_f(s['seg4'])}（{s['n_certified']} 对） | {_f(ms['seg35'])} / {_f(ms['seg4'])} | {_f(mc['seg35'])} / {_f(mc['seg4'])} | {'、'.join(ms['substituted_a0']) or '无'} / {'、'.join(ms['substituted_identity']) or '无'} | {r['n_two_rounds_or_more']} | {_f(r['mean_task_clock_read'], 1)} / {_f(r['max_task_clock_read'], 1)} | {r['clock_90']['n_within']}/{r['clock_90']['n']} |")
    cm = d['certified_means']
    L += ['', f"修复后 19 对均值（只在那一次 19/19 过证书时有）：seg35 {_three(cm['seg35'])}，seg4 {_three(cm['seg4'])}，官方 SDlogJ {_three(cm['sdlogj_official'])}。", '']
    attempts = [f"{p['pair']} {a['rep']} {_f(a['seg35'])} / {_f(a['seg4'])}（不可行单元 {a['n_infeasible_after']}）" for p in d['per_pair'] for a in p['attempts']]
    if attempts:
        L += ['**未认证的尝试（没有通过证书的状态，不进任何均值）**：' + '；'.join(attempts), '']
    L += ['## 每对（R1 / R2 / R3）', '', '| 对 | 过证书 | 撞保险丝 | seg35（过证书的） | 极差 | seg4（过证书的） | 极差 | 官方 SDlogJ 极差 | 方法级 seg35（严格） | 轮数 | 修复秒 | 任务时钟 s | 认证下界 | 下界 最大 ÷ 最小 | 修复场不同的个数 |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for p in d['per_pair']:
        L.append(f"| {p['pair']} | {p['n_certified']}/3 | {p['n_fuse']} | {_three(p['seg35'])} | {_f(p['seg35']['range'])} | {_three(p['seg4'])} | {_f(p['seg4']['range'])} | {_f(p['sdlogj_official']['range'])} | {_three(p['method_seg35_strict'])} | {_three(p['rounds'], 0)} | {_three(p['seconds_repair'], 1)} | {_three(p['task_clock'], 1)} | {' / '.join(_bound(v) for v in p['min_lower_bound_after']['values'])} | {_f(p['lower_bound_ratio'], 1)} | {p['n_distinct_repaired_fields']} |")
    largest = max((p['seg35']['range'] for p in d['per_pair'] if p['seg35']['range'] is not None), default=None)
    L += ['', f"每对 seg35（过证书的修复）在 3 次之间的极差，最大 {_f(largest)}。认证下界到正就停，不是 J 的真实最小值；下界之比只描述。", '']
    fb, ident = d['fallback'], d['identity']
    L += ['## 退路（2c(i)，两层待 BZ 定）', '', f"配置 C 的场（d0_v1 A0）按 D0 的记录合格：{fb['n_qualified']}/{len(ALL_PAIRS)}；不合格（用恒等）：{'；'.join(x['pair']+'：'+str(x['why']) for x in fb['not_qualified']) or '无'}。**代入的 A0 证书是 D0 运行自己算的（冻结的配置 C，CPU FP64），RP 没有重算**。恒等场（rp_v1/id，只评不合格的对）能用：{ident['n_usable']}。只在失败的对上代入。", '']
    c = d['controls']
    L += ['## 核对', '', f"修复配置：{c['n_configs']} 种（要 1 种，wall_seconds {fx.FUSE_SECONDS:g}）；代码：{c['code']}（R1 要 {rp.BASE_COMMIT[:7]}，R2、R3 要批次的提交 {str(c['batch_code_sha'])[:7]}）；修复的证书配置、冻结提交、评测器、评测证书的配置各 {', '.join(str(len(v)) for v in c['same'].values())} 种（都要 1 种）；问题：{'；'.join(c['problems']) or '无'}。", '',
          '运行环境（只描述；读数量的就是 GPU 上的非确定）：' + '；'.join(f"{name} {v}（{'、'.join(reps)}）" for name, values in c['described'].items() for v, reps in values.items()) + '。', '', f"读数的规则：{d['code']}", '',
          '## 它证伪不了什么', '', '3 次抽样不是分布：3 次都过不说明不会失败，极差小不说明下一次也小。不是 F_B，没有经认证的下降，只有终态过证书；开发集，这条线已经用前 5 对选过配。']
    return '\n'.join(L)+'\n'


LONG_FIELDS = ('rep', 'pair', 'kind', 'why', 'certified_at_1', 'wall_hit', 'rounds', 'seconds_repair', 'seconds_warmup', 'task_clock', 'min_lower_bound_after', 'n_infeasible_after', 'seg35_after', 'seg4_after', 'sdlogj_official_after', 'repaired_u_sha256',
               'method_source_strict', 'method_seg35_strict', 'method_seg4_strict', 'method_sdlogj_official_strict', 'method_source_code', 'method_seg35_code', 'method_seg4_code', 'method_sdlogj_official_code')


def write_long(path, rows, methods):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=LONG_FIELDS)
        w.writeheader()
        for rep in REPS:
            for pair in ALL_PAIRS:
                r = rows[rep][pair] or {}
                k, why = kind(rows[rep][pair])
                after = r.get('after') if r.get('metrics') else None
                line = dict(rep=f'R{rep}', pair=core.short_name(pair), kind=k, why=why or '', certified_at_1=r.get('certified_at_1'), wall_hit=r.get('wall_hit'), rounds=r.get('rounds'), seconds_repair=r.get('seconds_repair'), seconds_warmup=r.get('seconds_warmup'), task_clock=r.get('task_clock'),
                            min_lower_bound_after=r.get('min_lower_bound_after'), n_infeasible_after=r.get('n_infeasible_after'), seg35_after=None if after is None else after['seg35'], seg4_after=None if after is None else after['seg4'], sdlogj_official_after=None if after is None else after['sdlogj_official'], repaired_u_sha256=r.get('repaired_u_sha256'))
                for how in HOWS:
                    m = methods[how][rep][pair]
                    line.update({f'method_source_{how}': m['source'], **{f'method_{x}_{how}': None if m['metrics'] is None else m['metrics'].get(x) for x in MEASURES}})
                w.writerow(line)


NAMES = ('decision_rp.json', 'summary_rp.md', 'per_pair_rp.csv')


def write_rp(root, stops=None, results=None, sources=None, document=None, stage=None):
    decision, rows, methods = decide(root, stops, results, sources, document, stage)
    out = Path(root)/'analysis'
    set_aside(out, NAMES)
    (out/NAMES[0]).write_text(json.dumps(decision, indent=1, ensure_ascii=False)+'\n')
    (out/NAMES[1]).write_text(summary_text(decision), encoding='utf-8')
    write_long(out/NAMES[2], rows, methods)
    return decision
