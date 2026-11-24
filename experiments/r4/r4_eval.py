"""R4 的评测单元（合同 C25.6-11、C25.8-4）与锁前核对；HS 19 个输出的重建（不读真值，干跑时总负责人先跑一遍）。

台：全部臂定稿之后，一个评测单元做第一读；组合臂定稿之后另一个单元做计划内的第二读（同一把锁的第二行）。
  锁前（不需要真值）：每臂每对都有定稿记录；任一臂 hx_provenance_rejected → 整台 STOP；CTL 16 对都有输出；输出数组的 sha256 等于定稿记录；CTL 中 hx_certified 的对等于照 HS 评测写法重建的数组
  （run_ef.repaired_field）；每个读真值之前的失败都有已提交的续跑记录且有 BZ 已看到的确认；评测单元再调一次台守卫。候选的数组不符 → 该候选排除（锁照常建）；CTL 不符 → 不建锁、STOP。
  锁后：真值只经 LabelGuard（data.read_labels）读，每人每份一次；指标 = run_fl.field_metrics；CTL 16 个输出的指标与"独立载入 u + LabelGuard 原始数组 + run_fl.field_metrics"逐对比（≤ 2e-4）。
  建锁之后的任何结局都是终局：异常、缺真值、指标比对不符 → 记下，不再读。
开发集：建锁之前的清单（C25.8-4 第 1–6 条）任何一条失败：STOP，不花读数；建锁之后的任何结局都算用掉一次读数（D141）。真值照 HS 的评测写法（experiments/d0/evaluate.py truth_labels）每对读一次；
  同一次读里重算 HS 钉住的 19 个输出，逐对与 decision_hs.json 比（≤ 2e-4）。
"""
import json
from pathlib import Path
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import r4_chain as ch  # noqa: E402
import r4_common as rc  # noqa: E402
import r4_decide as rd  # noqa: E402
import r4_lock as lk  # noqa: E402

core, he = ch.core, ch.he
RESUME_DIR_REL = 'runs/20261113-r4/resume'
RESUME_SCHEMA = rc.SCHEMA.format('resume')
STOP_SCHEMA = rc.SCHEMA.format('stop')


class PreLockStop(rc.Refused):
    """建锁之前的核对不过：不建锁，STOP，交总负责人与 BZ（不花读数）。"""
    status = 'stop_before_lock'


def evaluator():
    return core.load_module('_mc0_evaluate', 'experiments/d0/evaluate.py')


def metrics_of(u, labs, spacing):
    """HS 的评测函数（run_fl.field_metrics：run_mc0.field_metrics 加 HD95）。"""
    import run_fl
    return run_fl.field_metrics(u, labs, spacing, evaluator())


def pre_repair_metrics(u, labs, spacing):
    """D136：修复前场（渐变后的 T88 场）的 seg35 / seg4，只描述。"""
    import run_mc0
    m = run_mc0.field_metrics(u, labs, spacing, evaluator())
    return dict(seg35=m['seg35'], seg4=m['seg4'])


# ---- 续跑记录（C25.7-3，修订 10d、10e） ---------------------------------------------------------------------------------------------------------
def stop_files(root):
    return sorted((Path(root)/'stops').glob('stop_*.json'))


def write_stop(root, record):
    """队列在读真值之前的失败处停下时写的记录（<R4>/stops/stop_<k>.json）；总负责人据此写续跑记录。"""
    d = Path(root)/'stops'
    d.mkdir(parents=True, exist_ok=True)
    k = len(list(d.glob('stop_*.json')))
    return rc.write_json(d/f'stop_{k}.json', dict(record, schema=STOP_SCHEMA, utc=rc.utc(), local_time=rc.local_time()))


def resume_repo_of(root):
    """续跑记录所在的仓库（修订 6）：batch.json 与 dev_batch.json 记的 resume_repo（两份都在时必须相同）；都不在时取默认 rc.RESUME_REPO（batch-init 之前没有单元可起）。"""
    repos = set()
    for name in ('batch.json', 'dev_batch.json'):
        doc = rc.read_json(Path(root)/name)
        if doc is None:
            continue
        if not doc.get('resume_repo'):
            raise rc.Refused(f'{name} 没有记 resume_repo（修订 6：先 make-checkout，再在检出里 batch-init）')
        repos.add(str(Path(doc['resume_repo']).resolve()))
    if len(repos) > 1:
        raise rc.Refused(f'batch.json 与 dev_batch.json 记的 resume_repo 不同：{sorted(repos)}')
    repo = Path(repos.pop()) if repos else rc.RESUME_REPO
    if Path(root).resolve() == rc.R4_ROOT.resolve() and repo.resolve() != rc.RESUME_REPO.resolve():
        raise rc.Refused(f'真实运行的续跑仓库只能是 {rc.RESUME_REPO}，batch 记的是 {repo}（修订 6a I1）')
    return repo


def committed(repo, path):
    rel = Path(path).resolve().relative_to(Path(repo).resolve()).as_posix()
    tracked = subprocess.run(['git', '-C', str(repo), 'ls-files', '--error-unmatch', '--', rel], capture_output=True, text=True).returncode == 0
    status = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain', '--', rel], capture_output=True, text=True)
    return tracked and status.returncode == 0 and not status.stdout.strip()


def resume_records(repo, *, require_committed=True):
    """repo 是续跑仓库（修订 6：main 仓库，不是运行检出）；每份续跑记录要在它的 git 里已提交、工作树里这一份与提交的一致。"""
    d = Path(repo)/RESUME_DIR_REL
    rows = []
    for path in sorted(d.glob('resume_*.json')) if d.is_dir() else []:
        doc = rc.read_json(path)
        rows.append((path, doc, committed(repo, path) if require_committed else True))
    return rows


def resume_problem(doc, *, before_lock):
    """一份续跑记录的问题（None 即合格）。(甲)：总负责人签即可；(乙)：必须引 BZ 的原话与时间，写明改动范围与其余单元为何不受影响。建锁之前每份都要有 BZ 已看到的确认。"""
    if not isinstance(doc, dict) or doc.get('schema') != RESUME_SCHEMA:
        return 'not a resume record'
    handling = doc.get('handling')
    if handling == 'jia':
        if not doc.get('signed_by'):
            return '(甲) 的续跑记录没有签署人'
    elif handling == 'yi':
        if not (doc.get('bz_words') and doc.get('bz_time')):
            return '(乙) 的续跑记录没有引 BZ 的原话与时间'
        if not doc.get('changed_units') or not doc.get('why_others_unaffected'):
            return '(乙) 的续跑记录没有写改动范围与其余单元为何不受影响'
    else:
        return f'handling {handling!r} 不是 jia 或 yi'
    if doc.get('scope') not in ('candidate', 'bench', 'feasibility', 'combo'):
        return f"scope {doc.get('scope')!r}"
    if before_lock and not ((doc.get('bz_seen') or {}).get('words') and (doc.get('bz_seen') or {}).get('time')):
        return '建锁之前：续跑记录没有 BZ 已看到的确认（引 BZ 的话与时间）'
    return None


RESUME_BRANCH = 'refs/heads/main'


def git_text(repo, *args):
    done = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else None


def resume_state(root, repo=None, *, before_lock, require_committed=True, record=None):
    """(problems, {(scope, 臂): 原因}（(甲) 判不可判或可行性不过）, 整台不可判的原因或 None)：每个 stop 都要有一份合格的、已提交的续跑记录（stop_sha256 对上）。
    repo：续跑仓库，None 即 resume_repo_of(root)（修订 6）。
    修订 6a M4：有 stop 要核续跑记录时（require_committed），续跑仓库的当前分支必须是 refs/heads/main；record（给了就填）记续跑仓库、HEAD、symbolic-ref 与每个 stop 用的
    那份记录的 blob id（git rev-parse HEAD:<rel>），供锁前记录与队列日志。"""
    repo = resume_repo_of(root) if repo is None else repo
    records = resume_records(repo, require_committed=require_committed)
    by_stop = {}
    for path, doc, is_committed in records:
        if doc and doc.get('stop_sha256'):
            by_stop.setdefault(doc['stop_sha256'], []).append((path, doc, is_committed))
    problems, candidates, whole = [], {}, None
    stops = stop_files(root)
    used = {}
    if record is not None:
        record.update(resume_repo=str(repo), stops=len(stops), records_used=used)
    if stops and require_committed:
        head, branch = git_text(repo, 'rev-parse', 'HEAD'), git_text(repo, 'symbolic-ref', '-q', 'HEAD')
        if record is not None:
            record.update(head=head, symbolic_ref=branch)
        if branch != RESUME_BRANCH:
            problems.append(f'续跑仓库 {repo} 的当前分支是 {branch}，不是 {RESUME_BRANCH}（修订 6a M4）')
    for stop in stops:
        sha = rc.sha256_file(stop)
        rows = by_stop.get(sha) or []
        if not rows:
            problems.append(f'{stop.name}：没有续跑记录')
            continue
        path, doc, is_committed = rows[-1]
        if record is not None:
            rel = Path(path).resolve().relative_to(Path(repo).resolve()).as_posix() if require_committed else None
            used[stop.name] = dict(record=Path(path).name, blob=git_text(repo, 'rev-parse', f'HEAD:{rel}') if rel else None)
        if not is_committed:
            problems.append(f'{path.name}：续跑记录没有提交或工作树里有改动')
            continue
        problem = resume_problem(doc, before_lock=before_lock)
        if problem:
            problems.append(f'{path.name}：{problem}')
            continue
        if doc['handling'] == 'jia':
            if doc['scope'] in ('candidate', 'combo', 'feasibility') and doc.get('arm') not in (rc.CONTROL, None):
                candidates[(doc['scope'], doc['arm'])] = f'读真值之前的失败按 (甲) 处理（{stop.name}，{path.name}）'
            elif doc['scope'] == 'bench' or doc.get('arm') == rc.CONTROL:
                whole = f'读真值之前的失败在 CTL 或共用步骤，按 (甲) 整台不可判（{stop.name}，{path.name}）'
    return problems, candidates, whole


# ---- 锁前：spy 的记录（复核 B2） --------------------------------------------------------------------------------------------------------------------
def spy_record_problems(run_dir, final=None, notes=None):
    """一对一臂的 spy 记录：supervisor 的 opened_files_supervisor.json、t88/opened_files.json、修复跑过时 repair/opened_files.json，都要在且 passed 为 true。
    修订 3（N-I2）：子进程被信号杀掉或超时（line.json 记的 t88.returncode、repair.returncode 是 'timeout' 或负数）写不出记录：照 D74 放行这种缺失，记进 notes（"被杀、无 spy 记录"）。"""
    run_dir = Path(run_dir)
    problems = []
    line = ch.line_record(run_dir) or {}
    record = rc.read_json(run_dir/'opened_files_supervisor.json')
    if record is None or record.get('passed') is not True:
        problems.append('spy 记录 opened_files_supervisor.json 不在或没有通过')
    for stage in ('t88', 'repair'):
        path = run_dir/stage/'opened_files.json'
        if stage == 'repair' and not (run_dir/'repair').is_dir():
            continue
        record = rc.read_json(path)
        returncode = (line.get(stage) or {}).get('returncode')
        if record is None and ch.killed(returncode) and (line.get('spy') or {}).get(stage) == ch.KILLED_NO_RECORD:
            if notes is not None:
                notes.append(dict(run_dir=str(run_dir), stage=stage, returncode=returncode, note=ch.KILLED_NO_RECORD))
            continue
        if record is None or record.get('passed') is not True:
            problems.append(f'spy 记录 {path.relative_to(run_dir)} 不在或没有通过')
    return problems


def reader_record_problems(run_dir, role, registry_record=None):
    """T88 的读取记录（C22.1-2；复核次要 12 的清单 (5)）：done 的场 oracle 为 false，通道来源对锚（开发集：he_core.channel_source_problems；台：台的 id、policy、登记），
    pair.npz 只读了 fixed、moving、initial_u。T88 没有 done 的对（退路）不核来源。"""
    t88 = rc.read_json(Path(run_dir or '/nonexistent')/'t88'/'run.json')
    if t88 is None:
        line = ch.line_record(run_dir) if run_dir else None
        return [] if (line or {}).get('line_status') in (rc.FALLBACK_C, rc.FALLBACK_ID) else ['没有 t88/run.json']
    if role == 'bench' and registry_record is None:                                      # 修订 8 次要 2：台一侧必须给登记记录（拿 None 比一定报问题）
        raise ValueError('reader_record_problems(role=\'bench\') needs the bench registry record')
    problems = []
    if not set((t88.get('data') or {}).get('keys_read') or []) <= {'fixed', 'moving', 'initial_u'}:
        problems.append(f"pair.npz 读了 {(t88.get('data') or {}).get('keys_read')}")
    if t88.get('status') == 'done':
        if t88.get('oracle') is not False:
            problems.append(f"oracle {t88.get('oracle')}")
        sources = t88.get('channel_sources') or {}
        problems += he.channel_source_problems(sources) if role == 'dev' else ch.bench_source_problems(sources, registry_record)
    return problems


def finalize_spy_problems(root, role, arm):
    record = rc.read_json(Path(root)/('final' if role == 'bench' else 'dev_final')/arm/'opened_files_finalize.json')
    return [] if record is not None and record.get('passed') is True else ['定稿单元的 spy 记录不在或没有通过']


# ---- 台：锁前核对 ------------------------------------------------------------------------------------------------------------------------------------
def ctl_rebuild(root, pair, final, *, stage_dir, bench):
    """CTL 中 hx_certified 的对：照 HS 评测的写法重建输出数组（T88 场 → 渐变 → run_ef.repaired_field），sha256 等于定稿记录（C25.6-11 修订 10c）。返回问题或 None。"""
    import run_ef
    import r4_bench as rb
    run_dir = Path(final['run_dir'])
    t88 = rc.read_json(run_dir/'t88'/'run.json')
    repair = rc.read_json(run_dir/'repair'/'repair.json')
    try:
        pin, _ = rb.load_bench_pair(bench, stage_dir, pair)
        with np.load(run_dir/'t88'/t88['field']['file'], allow_pickle=False) as z:
            raw = np.array(z['u'])
        tapered, _ = core.frozen_api().boundary_taper(raw, pin.grid, core.TAPER)
        tapered = np.ascontiguousarray(tapered)
        if he.array_sha256(tapered) != repair['source']['tapered_u_sha256']:
            return '渐变后的 T88 场与 repair.json 记的不符'
        v = run_ef.repaired_field(tapered, run_dir/'repair', repair)
        return None if he.array_sha256(v) == final['output']['state_hash'] else '重建的数组与定稿记录的输出不符'
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return f'重建失败：{type(exc).__name__}: {exc}'


def bench_prelock(root, arms, bench, *, stage_dir, repo=None, require_committed=True, rebuild=ctl_rebuild):
    """台的锁前核对（C25.6-11）。返回 dict(excluded={候选: 原因}, undecidable_before_truth={候选: 原因}, finals={arm: {pair: final}})；整台的问题 → PreLockStop（不建锁）。"""
    import r4_bench as rb
    resume_log = {}
    resume_problems, jia, whole = resume_state(root, repo, before_lock=True, require_committed=require_committed, record=resume_log)
    before = {arm: why for (scope, arm), why in jia.items() if scope in ('candidate', 'combo')}
    if resume_problems:
        raise PreLockStop('续跑记录：'+'；'.join(resume_problems))
    if whole:
        raise PreLockStop(whole)
    problems, excluded, finals, killed_notes = [], {}, {}, []
    for arm in arms:
        finals[arm] = {}
        for pair in bench.pair_names:
            line = ch.line_record(Path(root)/'runs'/arm/pair) or {}
            if line.get('line_status') == rc.REJECTED:                                  # 复核 I5：所有臂（含 (甲) 的臂）都扫
                problems.append(f'{arm} {pair}：hx_provenance_rejected（按共用步骤在读真值之前的失败，整台 STOP）')
            final = ch.read_final(root, 'bench', arm, pair)
            if final is None:
                if arm not in before:                                                   # (甲)：这一臂不再跑，后面的对没有定稿记录
                    problems.append(f'{arm} {pair}：没有定稿记录')
                continue
            finals[arm][pair] = final
            if final.get('line_status') == rc.REJECTED:
                problems.append(f'{arm} {pair}：hx_provenance_rejected（按共用步骤在读真值之前的失败，整台 STOP）')
            if arm not in before:
                problems += [f'{arm} {pair}：{p}' for p in spy_record_problems(final['run_dir'], final, notes=killed_notes)]
        if arm not in before and finals[arm]:
            problems += [f'{arm}：{p}' for p in finalize_spy_problems(root, 'bench', arm)]
    if problems:
        raise PreLockStop('；'.join(problems))
    for arm in arms:
        if arm in before:
            continue                                                                    # 不可判（读真值之前，(甲)）：数组不读
        for pair, final in finals[arm].items():
            status = final.get('line_status')
            if status in rc.OUTPUT_STATUSES:
                _, problem = ch.output_array(final)
                if problem is None and arm == rc.CONTROL and status == rc.HX:
                    problem = rebuild(root, pair, final, stage_dir=stage_dir, bench=bench)
                if problem:
                    if arm == rc.CONTROL:
                        raise PreLockStop(f'CTL {pair}：{problem}（整台 STOP）')
                    excluded.setdefault(arm, f'{pair}：{problem}（建锁之前的数组核对不符：排除，算没过筛门）')
            elif arm == rc.CONTROL:
                raise PreLockStop(f'CTL {pair}：没有输出（{status}）：读真值之前的失败，STOP，交 BZ')
            elif status == rc.IMPLEMENTATION and arm not in before:
                raise PreLockStop(f'{arm} {pair}：hx_implementation_error 没有 (甲) 的续跑记录')
    for pair in bench.pair_names:
        rb.bench_admission(bench, pair=pair)                                    # 读真值之前对每对再调一次台守卫（C25.6-11）
    return dict(excluded=excluded, undecidable_before_truth=before, finals=finals, killed_without_spy_record=killed_notes, resume_repo=resume_log)


# ---- 台：真值与指标 ----------------------------------------------------------------------------------------------------------------------------------
def bench_truth(bench, *, files=None, guard=None):
    """{(subject, kind): (uint8 数组, 记录)}：每人每份一次，只经 LabelGuard（read_labels：B 线训练受试者）；台守卫先过。"""
    import r4_bench as rb
    from bayesreg.seg.unet.data import LabelGuard, OasisFiles, read_labels
    files = OasisFiles(rc.DATA_ROOT) if files is None else files
    guard = LabelGuard(bench.split, bench.allowlist, training_source=bench.world.training_source) if guard is None else guard
    out = {}
    for subject in bench.ordered_subjects():
        rb.bench_admission(bench, subjects=[subject])
        for kind in ('seg35', 'seg4'):
            out[(subject, kind)] = read_labels(files, subject, guard, kind=kind)
    return out, guard


def labs_for(truth, fixed, moving):
    return dict(fixed_seg35=truth[(fixed, 'seg35')][0], moving_seg35=truth[(moving, 'seg35')][0], fixed_seg4=truth[(fixed, 'seg4')][0], moving_seg4=truth[(moving, 'seg4')][0])


def grid_problem(truth_record, grid_meta):
    want = {k: list(grid_meta[k]) for k in ('shape_zyx', 'spacing_xyz', 'origin_xyz')}
    got = dict(shape_zyx=truth_record['grid_shape_zyx'], spacing_xyz=truth_record['spacing_xyz'], origin_xyz=truth_record['origin_xyz'])
    return None if got == want else f"{truth_record['subject']} {truth_record['kind']}：真值的网格 {got} 不是 pair.npz 的 {want}"


def evaluate_row(final, labs, spacing, grid):
    """R4 的评测包装（C25.6-11）：从定稿记录取 u → 指标；修复前场的 seg35 / seg4（D136）。"""
    if final.get('line_status') not in rc.OUTPUT_STATUSES:
        return dict(line_status=final.get('line_status'), metrics=None)
    u, problem = ch.output_array(final)
    if problem:
        raise ValueError(problem)
    labs = {k: np.asarray(v).astype(np.int16) for k, v in labs.items()}
    row = dict(line_status=final['line_status'], fallback=(final.get('fallback') or {}).get('kind'), metrics=metrics_of(u, labs, spacing), state_hash=final['output']['state_hash'],
               task_clock=final.get('task_clock'), repair_rounds=final.get('repair_rounds'))
    t88 = rc.read_json(Path(final['run_dir'])/'t88'/'run.json')
    if t88 and t88.get('field'):
        with np.load(Path(final['run_dir'])/'t88'/t88['field']['file'], allow_pickle=False) as z:
            raw = np.array(z['u'])
        tapered, _ = core.frozen_api().boundary_taper(raw, grid, core.TAPER)
        row['pre_repair'] = dict(pre_repair_metrics(np.ascontiguousarray(tapered), labs, spacing), note='the T88 field after the taper, before the repair (D136, description only)')
    return row


def independent_control(final, truth, fixed, moving, spacing):
    """对照一侧：u 按定稿记录指向的文件独立载入，标签由 LabelGuard 读出的原始数组构造，直接调 run_fl.field_metrics。"""
    import run_fl
    o = Path(final['output']['directory'])
    with np.load(o/'final.npz', allow_pickle=False) as z:
        u = np.array(z['u'])
    labs = dict(fixed_seg35=truth[(fixed, 'seg35')][0], moving_seg35=truth[(moving, 'seg35')][0], fixed_seg4=truth[(fixed, 'seg4')][0], moving_seg4=truth[(moving, 'seg4')][0])
    m = run_fl.field_metrics(u, labs, spacing, evaluator())
    return {k: m[k] for k in ('seg35', 'seg4', 'sdlogj_official')}


def bench_eval(root, bench, *, read, combo=None, stage_dir, repo=None, code_commit, plan_sha256, files=None, require_committed=True, rebuild=ctl_rebuild, log=print):
    """台的第一读（read='first'，全部七臂）或组合臂计划内的第二读（read='second'）。锁前任何核对不过：PreLockStop，不建锁。建锁之后的任何结局都是终局：写 bench_eval_<n>.json（含失败）。"""
    lock = lk.lock_path(root, lk.BENCH)
    arms = list(rc.ARM_ORDER) if read == 'first' else [combo]
    if read == 'first':
        lk.check_absent(lock)
    else:
        lk.check_second(lock, combo)
    pre = bench_prelock(root, arms, bench, stage_dir=stage_dir, repo=repo, require_committed=require_committed, rebuild=rebuild)
    first_truth = None
    if read == 'second':
        planned = rc.read_json(Path(root)/'analysis'/'bench_decision_first.json') or {}
        if planned.get('combo') != combo or not planned.get('combo_needs_second_read'):
            raise PreLockStop(f'第二读：{combo} 不是第一读判定（bench_decision_first.json）计划的组合臂 {planned.get("combo")}（复核 I4）')
        first = rc.read_json(Path(root)/'eval'/'bench_eval_1.json') or {}
        first_truth = first.get('truth_sha256')
        if not first_truth:
            raise PreLockStop('第二读：第一读的真值 sha256 不在 bench_eval_1.json 里')
        if combo in pre['excluded'] or combo in pre['undecidable_before_truth']:
            raise PreLockStop(f'第二读：组合臂 {combo} 在锁前被排除或不可判：{pre["excluded"].get(combo) or pre["undecidable_before_truth"].get(combo)}')
    n = 1 if read == 'first' else 2
    target = Path(root)/'eval'/f'bench_eval_{n}.json'
    if read == 'first':
        line = lk.take_first(lock, arms=arms, code_commit=code_commit, plan_sha256=plan_sha256, reason='第一读（台，全部臂）')
    else:
        _, line = lk.take_second(lock, combo, code_commit=code_commit, plan_sha256=plan_sha256, truth_sha256=first_truth)
    doc = dict(schema=rc.SCHEMA.format('bench_eval'), read=read, lock_line=line, arms=arms, pairs=list(bench.pair_names), excluded=pre['excluded'], undecidable_before_truth=pre['undecidable_before_truth'],
               killed_without_spy_record=pre.get('killed_without_spy_record'), resume_repo=pre.get('resume_repo'))
    try:
        truth, guard = bench_truth(bench, files=files)
        doc['truth_sha256'] = {f'{s}:{k}': rec['sha256'] for (s, k), (_, rec) in truth.items()}
        if read == 'second' and doc['truth_sha256'] != first_truth:
            raise ValueError('第二读的真值文件 sha256 与第一读不同')
        import r4_bench as rb
        rows, control, grids = {}, {}, {}
        for arm in arms:
            if arm in pre['undecidable_before_truth'] or arm in pre['excluded']:
                continue                                                                # (甲) 的不可判与锁前数组核对排除的候选：不评（复核 B1）
            rows[arm] = {}
            for name, _, _, fixed, moving in bench.pairs:
                final = pre['finals'][arm][name]
                meta = rc.read_json(Path(stage_dir)/name/'manifest.json')
                for subject in (fixed, moving):
                    for kind in ('seg35', 'seg4'):
                        problem = grid_problem(truth[(subject, kind)][1], meta)
                        if problem:
                            raise ValueError(problem)
                spacing = tuple(meta['spacing_xyz'])
                if name not in grids:
                    grids[name] = rb.load_bench_pair(bench, stage_dir, name)[0].grid
                rows[arm][name] = evaluate_row(final, labs_for(truth, fixed, moving), spacing, grids[name])
                if arm == rc.CONTROL:
                    control[name] = independent_control(final, truth, fixed, moving, spacing)
        doc.update(rows=rows, guard=guard.record())
        if read == 'first':
            mine = {p: {k: (rows[rc.CONTROL][p]['metrics'] or {}).get(k) for k in ('seg35', 'seg4', 'sdlogj_official')} for p in bench.pair_names}
            bad = rd.control_mismatches(mine, control)
            doc['control'] = dict(rule='the R4 evaluation wrapper against run_fl.field_metrics on the independently loaded u and LabelGuard\'s raw arrays, |Δ| ≤ 2e-4 (C25.6-11)', mismatches=bad, reference=control)
            if bad:
                doc['whole_undecidable'] = f'评测侧的指标比对不符：{bad[:5]}'
        doc['ended_utc'] = rc.utc()
    except Exception as exc:                                                            # noqa: BLE001 - after the lock every outcome is final
        doc.update(failed=f'{type(exc).__name__}: {exc}'[:2000], traceback=traceback.format_exc()[-3000:], whole_undecidable=f'建锁之后评测单元异常：{type(exc).__name__}' if read == 'first' else None,
                   combo_undecidable=read == 'second', note='after the lock: final, no second read (C25.7-3)')
    rc.write_json(target, doc)
    return doc


def bench_decide(root, bench, *, out=None, repo=None, require_committed=True):
    """读 bench_eval_1.json（与 bench_eval_2.json，若有），写 bench_decision.json。组合臂在第二读之前按 (甲) 判不可判（scope combo 的续跑记录）：组合不采用，O 只含单项，
    照样写 bench_decision.json（复核次要 2）。"""
    first = rc.read_json(Path(root)/'eval'/'bench_eval_1.json')
    second = rc.read_json(Path(root)/'eval'/'bench_eval_2.json')
    whole = None if first is None else first.get('whole_undecidable')
    if first is None:
        whole = '没有 bench_eval_1.json'
    second_rows = None
    if second is not None:
        second_rows = None if second.get('failed') else second.get('rows')
    decision = rd.bench_decision(None if first is None else first.get('rows'), second_rows if second is not None else None, pairs=list(bench.pair_names), pair_subjects=bench.pair_subjects(),
                                 subjects=bench.ordered_subjects(), excluded=(first or {}).get('excluded'), undecidable_before_truth=(first or {}).get('undecidable_before_truth'), whole_undecidable=whole)
    if second is not None and second_rows is None and decision.get('combo'):
        decision.update(combo_result=dict(verdict=rd.UNDECIDABLE, reason=second.get('failed')), combo_adopted=False, order=rd.order_o(decision['results']), combo_needs_second_read=False)
    if second is None and decision.get('combo_needs_second_read'):
        _, jia, _ = resume_state(root, repo, before_lock=False, require_committed=require_committed)
        if ('combo', decision['combo']) in jia:
            decision.update(combo_result=dict(verdict=rd.UNDECIDABLE, reason=jia[('combo', decision['combo'])], layer='before_truth'), combo_adopted=False,
                            order=rd.order_o(decision['results']), combo_needs_second_read=False)
            second = {}
    target = Path(out) if out is not None else Path(root)/'analysis'/('bench_decision.json' if second is not None or not decision.get('combo_needs_second_read') else 'bench_decision_first.json')
    rc.write_json(target, dict(decision, eval_sha256={n: rc.sha256_file(Path(root)/'eval'/f'bench_eval_{n}.json') for n in (1, 2) if (Path(root)/'eval'/f'bench_eval_{n}.json').is_file()}))
    return decision


# ---- 开发集：HS 的 19 个输出（不读真值） ----------------------------------------------------------------------------------------------------------
def hs_dirs(hs_root, pair):
    short = core.short_name(pair)
    fields = sorted(Path(hs_root).glob(f'runs/SHx24_*/{short}/field_0600.npz'))
    repairs = sorted(Path(hs_root).glob(f'runs/RSHx24_*/{short}'))
    if len(fields) != 1 or len(repairs) != 1:
        raise ValueError(f'{short}：HS 的场或修复目录不是恰好一个（{len(fields)}、{len(repairs)}）')
    return fields[0], repairs[0]


def hs_rebuild(pair, grid, *, hs_root=rc.HS_ROOT):
    """HS 的一个输出照 HS 评测的写法重建（field_0600 → 渐变 → run_ef.repaired_field），核 sha256 等于 repair_metrics.json 的 repaired_u_sha256。返回 (v, 记录)；不符 ValueError。
    读 HS 的 repair.json、repair.npz、field_0600.npz 与 repair_metrics.json 里记的散列；不读真值。"""
    import run_ef
    field, rdir = hs_dirs(hs_root, pair)
    repair, metrics = rc.read_json(rdir/'repair.json'), rc.read_json(rdir/'repair_metrics.json')
    with np.load(field, allow_pickle=False) as z:
        raw = np.array(z['u'])
    tapered, _ = core.frozen_api().boundary_taper(raw, grid, core.TAPER)
    tapered = np.ascontiguousarray(tapered)
    if he.array_sha256(tapered) != repair['source']['tapered_u_sha256']:
        raise ValueError(f'{pair}：渐变后的 HS 场与 repair.json 的 tapered_u_sha256 不符')
    v = run_ef.repaired_field(tapered, rdir, repair)
    if he.array_sha256(v) != metrics['repaired_u_sha256']:
        raise ValueError(f'{pair}：重建的 HS 输出与 repaired_u_sha256 不符')
    return v, dict(field=str(field), repair_dir=str(rdir), repair_json_sha256=rc.sha256_file(rdir/'repair.json'), repair_metrics_sha256=rc.sha256_file(rdir/'repair_metrics.json'), repaired_u_sha256=metrics['repaired_u_sha256'])


def dev_pairs(stage=None):
    stage = Path(rc.ROOT/rc.DEV_STAGE_REL if stage is None else stage)
    return [row['pair'] for row in json.loads((stage/'selection.json').read_text(encoding='utf-8'))['pairs']]


def committed_hs_reference(repo=rc.ROOT):
    path = Path(repo)/rc.HS_REFERENCE_REL
    if not path.is_file() or rc.sha256_file(path) != rc.HS_REFERENCE_SHA256:
        return None
    pairs = json.loads(path.read_text(encoding='utf-8'))['p3']['pairs']
    return {s: dict(repair_json_sha256=p.get('repair_json_sha256'), repair_metrics_sha256=p.get('repair_metrics_sha256')) for s, p in pairs.items()}


def dev_state(root=rc.R4_ROOT, repo=None, *, require_committed=True, record=None):
    """(O, 可行性, 读数)：O 取自 bench_decision.json；可行性取自 dev_final/<项>/feasibility.json，FEAS_STOP 经 (甲) 的续跑记录（scope feasibility）之后记成可行性不过；
    读数取自 dev_decision_<n>.json。"""
    decision = rc.read_json(Path(root)/'analysis'/'bench_decision.json') or {}
    order = list(decision.get('order') or [])
    _, jia, _ = resume_state(root, repo, before_lock=False, require_committed=require_committed, record=record)
    feasibility = {}
    for item in order:
        if ('feasibility', item) in jia:                                                 # 复核次要 1：(甲) 之后记成可行性不过，队列在 feasibility.json 之前停下的也一样
            feasibility[item] = rd.NOT_FEASIBLE
            continue
        doc = rc.read_json(Path(root)/'dev_final'/item/'feasibility.json')
        if doc is None:
            continue
        feasibility[item] = doc['verdict']['verdict']
    readings = []
    for n in (1, 2):
        doc = rc.read_json(Path(root)/'analysis'/f'dev_decision_{n}.json')
        if doc is not None:
            readings.append((doc.get('arm'), doc.get('reading')))
    return order, feasibility, readings


def dev_prelock(root, arm, *, pairs, hs_root=rc.HS_ROOT, repo=rc.ROOT, stage_dir=None, code_problems=None, plan_problem=None, read_number=None, state=None, notes=None, resume_repo=None,
                resume_log=None):
    """开发集建锁之前的清单（C25.8-4 第 1–6 条），另加（复核 I4）：这一项的可行性是 FEASIBLE，且 next_dev_step 给出 (read, 这一项, read_number)；每对的 spy 记录都通过（复核 B2）。
    任何一条失败：PreLockStop（STOP，交总负责人与 BZ，不花读数）。返回 (finals, hs 输出, 记录)。"""
    problems = []
    finals = {}
    if read_number is not None:
        order, feasibility, readings = dev_state(root, resume_repo, record=resume_log) if state is None else state    # 修订 6：续跑记录在续跑仓库；repo（运行检出）只给 HS 的参照
        step = rd.next_dev_step(order, feasibility, readings)
        if feasibility.get(arm) != rd.FEASIBLE or step != dict(action='read', item=arm, read_number=read_number):
            problems.append(f'(0) 开发集的下一步是 {step}；{arm} 的可行性 {feasibility.get(arm)}：这次读数不是它')
        feas = rc.read_json(Path(root)/'dev_final'/arm/'feasibility.json') or {}
        if feas.get('spy_passed') is not True:
            problems.append('(0) SDlogJ 单元的 spy 记录没有通过')
        problems += [f'(0) {p}' for p in finalize_spy_problems(root, 'dev', arm)]
    for pair in pairs:
        final = ch.read_final(root, 'dev', arm, pair)
        if final is None or final.get('line_status') not in rc.OUTPUT_STATUSES:
            problems.append(f'(1) {pair}：没有定稿记录或没有输出')
            continue
        _, problem = ch.output_array(final)
        if problem:
            problems.append(f'(1) {pair}：{problem}')
        problems += [f'(1) {pair}：{p}' for p in spy_record_problems(final['run_dir'], final, notes=notes)] if final.get('run_dir') else [f'(1) {pair}：定稿记录没有 run_dir']
        problems += [f'(5) {pair}：{p}' for p in reader_record_problems(final.get('run_dir'), 'dev')]
        finals[pair] = final
    reference = committed_hs_reference(repo)
    if reference is None:
        problems.append(f'(3) {rc.HS_REFERENCE_REL} 不是钉住的那份')
    decision = Path(hs_root)/'analysis'/'decision_hs.json'
    if not decision.is_file() or rc.sha256_file(decision) != rc.HS_DECISION_SHA256:
        problems.append('(4) decision_hs.json 的 sha256 不是 0b85dd06…')
    hs_outputs, hs_record = {}, {}
    stage = Path(rc.ROOT/rc.DEV_STAGE_REL if stage_dir is None else stage_dir)
    for pair in pairs:
        try:
            pin, _ = run_hx_admit(stage, pair)
            hs_outputs[pair], hs_record[pair] = hs_rebuild(pair, pin.grid, hs_root=hs_root)
            short = core.short_name(pair)
            if reference is not None and ((reference.get(short) or {}).get('repair_json_sha256') != hs_record[pair]['repair_json_sha256']
                                          or (reference.get(short) or {}).get('repair_metrics_sha256') != hs_record[pair]['repair_metrics_sha256']):
                problems.append(f'(3) {short}：HS 的记录不是已提交参照钉住的那份')
        except Exception as exc:                                                        # noqa: BLE001 - (2), (5): before the lock, a STOP
            problems.append(f'(2)/(5) {pair}：{type(exc).__name__}: {exc}')
    problems += [f'(6) {p}' for p in (code_problems or [])]+([f'(6) {plan_problem}'] if plan_problem else [])
    if problems:
        raise PreLockStop('开发集锁前清单：'+'；'.join(problems))
    return finals, hs_outputs, hs_record


def run_hx_admit(stage, pair):
    """开发集的准入：HE 的两道锁（封存名单的名字与 manifest，再 load_pair 的白名单）。"""
    import run_hx
    ctx = dict(stage_dir=str(stage), pair=pair, paths=dict(allowlist=str(rc.ROOT/rc.ALLOWLIST_REL)), pair_set='dev')
    record = run_hx.sealed_admission(ctx)
    return run_hx.load_images(ctx), record


def hs_finals(root, read_number, hs_outputs):
    """HS 的 19 个重建输出写成与臂一样的定稿记录（<R4>/dev_eval/hs_outputs_<n>_<k>/<pair>/output/final.npz），评测走同一条路径（evaluate_row：output_array、int16 的标签、
    run_fl.field_metrics；复核次要 12）。建锁之前写；不是真值。"""
    base = Path(root)/'dev_eval'
    base.mkdir(parents=True, exist_ok=True)
    k = len(list(base.glob(f'hs_outputs_{read_number}_*')))
    out = {}
    for pair, u in hs_outputs.items():
        o = base/f'hs_outputs_{read_number}_{k}'/pair/'output'
        o.mkdir(parents=True)
        digest = he.array_sha256(u)
        np.savez(o/'final.npz', u=np.ascontiguousarray(u), state_hash=np.array(digest))
        out[pair] = dict(line_status=rc.HX, run_dir=str(o.parent), output=dict(directory=str(o), final_npz_sha256=rc.sha256_file(o/'final.npz'), state_hash=digest),
                         note='HS (hs_v1) rebuilt as HS evaluated it (run_ef.repaired_field), sha256 = repaired_u_sha256')
    return out


def dev_eval(root, arm, read_number, *, pairs=None, hs_root=rc.HS_ROOT, repo=rc.ROOT, code_commit, plan_sha256, code_problems=None, plan_problem=None, truth=None, state=None, resume_repo=None):
    """开发集第 read_number 次读数。建锁之前的清单不过：PreLockStop（不花读数）。建锁之后的任何结局都算用掉一次读数：写 dev_eval_<n>.json（含失败），决定代码读它。"""
    pairs = dev_pairs() if pairs is None else list(pairs)
    lock = lk.lock_path(root, lk.DEV1 if read_number == 1 else lk.DEV2)
    lk.check_absent(lock)
    killed_notes, resume_log = [], {}
    finals, hs_outputs, hs_record = dev_prelock(root, arm, pairs=pairs, hs_root=hs_root, repo=repo, code_problems=code_problems, plan_problem=plan_problem, read_number=read_number, state=state,
                                                notes=killed_notes, resume_repo=resume_repo, resume_log=resume_log)
    hs_final = hs_finals(root, read_number, hs_outputs)
    for pair in pairs:
        run_hx_admit(Path(rc.ROOT/rc.DEV_STAGE_REL), pair)                              # 读真值之前再调一次守卫
    line = lk.take_first(lock, arms=[arm], code_commit=code_commit, plan_sha256=plan_sha256, reason=f'开发集第 {read_number} 次读数（{arm}）')
    doc = dict(schema=rc.SCHEMA.format('dev_eval'), read_number=read_number, arm=arm, lock_line=line, pairs=pairs, hs=hs_record, read_spent=True, killed_without_spy_record=killed_notes, resume_repo=resume_log)
    try:
        ev = evaluator()
        reader = ev.truth_labels if truth is None else truth
        rows, hs_rows, sha = {}, {}, {}
        for pair in pairs:
            grid = run_hx_admit(Path(rc.ROOT/rc.DEV_STAGE_REL), pair)[0].grid
            labs, manifest, pair_sha = reader(pair)                                      # 每对一次
            sha[pair] = pair_sha
            spacing = tuple(manifest['spacing_xyz'])
            rows[pair] = evaluate_row(finals[pair], labs, spacing, grid)
            m = evaluate_row(hs_final[pair], labs, spacing, grid)['metrics']                  # the same function, the same path (复核次要 12)
            hs_rows[pair] = {k: m[k] for k in ('seg35', 'seg4', 'sdlogj_official')}
            del labs
        decision = rc.read_json(Path(hs_root)/'analysis'/'decision_hs.json')
        pinned = {r['pair']: r for r in decision.get('per_pair') or []}
        reference = {p: {k: ((pinned.get(core.short_name(p)) or {}).get('metrics') or {}).get(k) for k in ('seg35', 'seg4', 'sdlogj_official')} for p in pairs}
        bad = rd.control_mismatches(hs_rows, reference)
        doc.update(rows=rows, hs_recomputed=hs_rows, hs_pinned=reference, truth_pair_sha256=sha, control=dict(mismatches=bad, rule='HS 的 19 个输出在同一次读里重算，与 decision_hs.json 逐对比，|Δ| ≤ 2e-4'),
                   undecidable='评测侧的指标比对不符' if bad else None, ended_utc=rc.utc())
    except Exception as exc:                                                            # noqa: BLE001 - after the lock: the read is spent (D141)
        doc.update(failed=f'{type(exc).__name__}: {exc}'[:2000], traceback=traceback.format_exc()[-3000:], undecidable=f'建锁之后评测单元异常：{type(exc).__name__}')
    rc.write_json(Path(root)/'dev_eval'/f'dev_eval_{read_number}.json', doc)
    return doc


def dev_decide(root, read_number, *, pairs=None):
    doc = rc.read_json(Path(root)/'dev_eval'/f'dev_eval_{read_number}.json')
    pairs = dev_pairs() if pairs is None else list(pairs)
    if doc is None or doc.get('undecidable') or doc.get('failed'):
        decision = dict(reading=rd.UNDECIDABLE, reason=(doc or {}).get('undecidable') or (doc or {}).get('failed') or '没有 dev_eval 记录', read_spent=True)
    else:
        hs_pinned = {p: dict(line_status=rc.HX, metrics=v) for p, v in doc['hs_pinned'].items()}
        decision = dict(rd.dev_reading(doc['rows'], hs_pinned, pairs, {p: k for k, p in enumerate(pairs)}), read_spent=True)
    decision.update(schema=rc.SCHEMA.format('dev_decision'), read_number=read_number, arm=(doc or {}).get('arm'), utc=rc.utc())
    rc.write_json(Path(root)/'analysis'/f'dev_decision_{read_number}.json', decision)
    return decision
