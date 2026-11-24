"""TC (runs/20261104-tc/plan.md; docs/bayesreg/notes/提速与换预分割器规划_20261104.md §5): what one run of the frozen solver has at each native time.  Rules and record functions only; the driver is run_tc.py.

The clock (F = /home/rmcfz/seq_vp_tag_results/frozen_eb808be, eb808be; read on the code and on the 15 original runs):
  history.json row, elapsed_seconds   F/jd_curl/three_d/sequential.py:198 `started = perf_counter()` at the start of solve_sequential, :620 `elapsed_seconds=perf_counter()-started`.  solve_sequential is called once per level, after the
                                      level's preparation.  So it counts only that level's solve.  It leaves out the entry (T88, or the injected entry and its 47 s hold), the transfer certificates and everything before them.
                                      In all these runs history.json has only level 3 rows.  The rule still takes level == 3 and outer_accepted only.
  commits.json row, solver_commit_seconds   the same number (F/scripts/run_stage_3d_e19_mri.py:1385 `solver_commit_seconds=row['elapsed_seconds']`); checked equal, row by row, or the curve stops.
  commits.json row, visible_seconds   the task clock (:1385 `perf_counter()-started` of the worker), recorded next to it, to read a native time as task time.  visible - elapsed on the originals: S_P 49.5-50.3 s, B1TM 4.3-4.9 s.
"""
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

SCHEMA_CURVE = 'bayesreg.tc.curve.v1'
SCHEMA_REPRODUCTION = 'bayesreg.tc.reproduction.v1'
ROOT_NAME = 'tc_v1'
TIMES =(0, 15, 30, 45, 60, 90, 120, 150, 210, 270)    # seconds of the native level; 0 means the first commit
FINAL = 'final'
NATIVE_LEVEL = 3
SELF_CHECK_TOL = 1e-12
SELF_CHECK_KEYS = ('seg35', 'seg4', 'sdlogj_official', 'sdlogj_brain')
CLOCK = ('history.json elapsed_seconds of the level 3 accepted rows: seconds from the start of the native level\'s solve_sequential (F/jd_curl/three_d/sequential.py:198, 620), native level only: not the entry, not its hold, '
         'not the transfer certificates; visible_seconds is the task clock')
RULE = 't = 0: the first native commit; t > 0: the last native commit with elapsed_seconds <= t; none: no point at t'


class CurveError(Exception):
    """The curve of this run cannot be trusted: the queue stops and the commits stay."""


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False, default=float)+'\n')
    tmp.replace(path)


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def task_dir(run):
    return Path(run)/read_json(Path(run)/'protocol.json')['task']['name']


def certificate_passed(certificate):
    """As evaluate.py for a final map: certified_feasible, lower bound > 0, no infeasible or uncertain cell.  A zero or an unknown bound does not pass."""
    certificate = certificate or {}
    lower = certificate.get('min_jd_lower_bound')
    return bool(certificate.get('status') == 'certified_feasible' and lower is not None and lower > 0 and certificate.get('n_infeasible') == 0 and certificate.get('n_uncertain') == 0)


def native_commits(task):
    """The native formal commits of a task directory, in order, each with its row of history.json: [{index (1-based), path, sha256, state_hash, elapsed_seconds, visible_seconds, D, F, GF, min_jd_lower_bound, passed}].
    CurveError unless the two files agree: as many accepted level 3 rows as level 3 commits, the same seconds, strictly increasing."""
    task = Path(task)
    commits = [c for c in read_json(task/'commits.json') if c.get('level') == NATIVE_LEVEL and c.get('native') is True]
    rows = [r for r in read_json(task/'history.json') if r.get('level') == NATIVE_LEVEL and r.get('outer_accepted') is True]
    if len(commits) != len(rows):
        raise CurveError(f'{task}: {len(commits)} native commits, {len(rows)} accepted level 3 rows')
    out, last = [], -math.inf
    for k, (commit, row) in enumerate(zip(commits, rows)):
        elapsed = row.get('elapsed_seconds')
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= last:
            raise CurveError(f'{task}: commit {k+1}: elapsed_seconds {elapsed} after {last}')
        if commit.get('solver_commit_seconds') != elapsed:
            raise CurveError(f"{task}: commit {k+1}: solver_commit_seconds {commit.get('solver_commit_seconds')} != elapsed_seconds {elapsed}")
        if commit.get('role') != 'formal_outer_commit':
            raise CurveError(f"{task}: commit {k+1}: role {commit.get('role')}")
        last = elapsed
        certificate = commit.get('certificate') or {}
        out.append(dict(index=k+1, path=commit['path'], sha256=commit['sha256'], state_hash=certificate.get('state_hash'), elapsed_seconds=elapsed, visible_seconds=commit.get('visible_seconds'), D=row.get('D_candidate'), F=row.get('F_candidate'),
                        GF=row.get('GF_candidate'), min_jd_lower_bound=certificate.get('min_jd_lower_bound'), passed=certificate_passed(certificate)))
    return out


def select_commit(elapsed, t):
    """The position in `elapsed` (increasing seconds of the commits) of the commit taken for time t, or None: see RULE.  Exactly t counts as not later than t."""
    if not elapsed:
        return None
    if t == 0:
        return 0
    k = bisect_right(elapsed, t)-1
    return k if k >= 0 else None


def choose(commits, times=TIMES):
    """{t: position or None}."""
    elapsed = [c['elapsed_seconds'] for c in commits]
    return {t: select_commit(elapsed, t) for t in times}


# ---- evaluation of one field ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
def evaluate_field(ev, u, labs, spacing):
    """The metrics of evaluate.py evaluate_run on one field, by the same functions (ev = experiments/d0/evaluate.py), with the same keys, without HD95 and the official Dice (those need surface distances; the final map's are in metrics.json).
    Groups: ev.A5_GROUPS over seg35 per label, the mean over the labels present, as ev.a5_group_mean."""
    d = ev.disp_voxels(u, spacing)
    w35, w4 = ev.warp_labels_l2r(labs['moving_seg35'], d), ev.warp_labels_l2r(labs['moving_seg4'], d)
    seg35 = ev.dice_labels(labs['fixed_seg35'], w35.astype(np.float64), labels=list(range(1, 36)))
    seg4 = ev.dice_labels(labs['fixed_seg4'], w4.astype(np.float64), labels=list(range(1, 5)))
    sd, folds, neg = ev.sdlogj_l2r(d)
    smooth = ev.brain_smoothness(u, spacing, labs['fixed_seg4'] > 0)
    per35, per4 = {r['label']: r['dice'] for r in seg35['per_label']}, {r['label']: r['dice'] for r in seg4['per_label']}
    metrics = dict(seg35=seg35['macro_dice'], seg4=seg4['macro_dice'], sdlogj_official=sd, num_foldings_official=folds, central_det_nonpositive_fraction=neg, **smooth, seg4_per_label=per4, seg35_per_label=per35)
    for name, labels in ev.A5_GROUPS.items():
        values = [per35[k] for k in labels if per35.get(k) is not None]
        metrics[name] = float(np.mean(values)) if values else None
    metrics['cortex'], metrics['white_matter'] = per4.get(1), per4.get(3)
    return metrics


def load_commit_field(path, commit):
    """u of one commit file; CurveError unless the file has the recorded sha256, a member u, whose bytes hash to the commit's certificate state_hash, and the certificate passed."""
    path = Path(path)
    if not path.is_file():
        raise CurveError(f'{path}: no such file')
    got = sha_file(path)
    if got != commit['sha256']:
        raise CurveError(f"{path}: sha256 {got} != commits.json {commit['sha256']}")
    with np.load(path) as z:
        if 'u' not in z:
            raise CurveError(f'{path}: no member u')
        u = np.array(z['u'])
    state = hashlib.sha256(np.ascontiguousarray(u, dtype=np.float64).view(np.uint8)).hexdigest()
    if state != commit['state_hash']:
        raise CurveError(f"{path}: state hash {state} != the certificate's {commit['state_hash']}")
    if not commit['passed']:
        raise CurveError(f'{path}: its certificate did not pass')
    return u


def curve_of_run(run, arm, pair, run_id, ev, *, clock):
    """curve.json of a finished run whose metrics.json exists: one evaluation per commit chosen, and one of the final map; a commit chosen by two times is evaluated once.  Reads the truth labels of the pair (ev.truth_labels), for the evaluation only.
    CurveError on any disagreement; nothing is deleted here."""
    run = Path(run)
    task = task_dir(run)
    commits = native_commits(task)
    labs, manifest, truth_sha = ev.truth_labels(pair)
    spacing = tuple(manifest['spacing_xyz'])
    chosen = choose(commits)
    done, seconds = {}, []
    points = []
    for t in TIMES:
        k = chosen[t]
        if k is None:
            points.append(dict(t=t, commit=None, reason=f'no native commit by t = {t} s'))
            continue
        commit = commits[k]
        if k not in done:
            tick = clock()
            done[k] = evaluate_field(ev, load_commit_field(task/commit['path'], commit), labs, spacing)
            seconds.append(clock()-tick)
        points.append(dict(t=t, commit=commit['index'], path=commit['path'], sha256=commit['sha256'], state_hash=commit['state_hash'], elapsed_seconds=commit['elapsed_seconds'], visible_seconds=commit['visible_seconds'], age_seconds=None if t == 0 else t-commit['elapsed_seconds'],
                           D=commit['D'], F=commit['F'], GF=commit['GF'], min_jd_lower_bound=commit['min_jd_lower_bound'], certificate_passed=commit['passed'], metrics=done[k]))
    final_record = read_json(task/'final_native.json')
    certificate = final_record.get('certificate') or {}
    with np.load(task/'final_native.npz') as z:
        u = np.array(z['u'])
    state = hashlib.sha256(np.ascontiguousarray(u, dtype=np.float64).view(np.uint8)).hexdigest()
    if state != certificate.get('state_hash'):
        raise CurveError(f"{task}/final_native.npz: state hash {state} != its certificate's {certificate.get('state_hash')}")
    tick = clock()
    metrics = evaluate_field(ev, u, labs, spacing)
    seconds.append(clock()-tick)
    last = commits[-1] if commits else None
    final = dict(t=FINAL, final_native_sha256=final_record.get('sha256'), state_hash=state, is_last_commit=None if last is None else last['state_hash'] == state, commit=None if last is None else last['index'], elapsed_seconds=None if last is None else last['elapsed_seconds'],
                 visible_seconds=None if last is None else last['visible_seconds'], D=None if last is None else last['D'], F=None if last is None else last['F'], GF=None if last is None else last['GF'], min_jd_lower_bound=certificate.get('min_jd_lower_bound'),
                 certificate_passed=certificate_passed(certificate), metrics=metrics)
    recorded = (read_json(run/'metrics.json').get('metrics') or {})
    differences = {key: metrics[key]-recorded[key] for key in SELF_CHECK_KEYS if key in recorded}
    self_check = dict(against='metrics.json of this run (evaluate_run)', keys=list(SELF_CHECK_KEYS), differences=differences, tolerance=SELF_CHECK_TOL, passed=len(differences) == len(SELF_CHECK_KEYS) and all(abs(v) <= SELF_CHECK_TOL for v in differences.values()))
    if not self_check['passed']:
        raise CurveError(f'{run}: the curve\'s evaluation of the final map differs from metrics.json: {differences}')
    if not final['certificate_passed']:
        raise CurveError(f'{task}: the final map\'s certificate did not pass')
    return dict(schema=SCHEMA_CURVE, run_id=run_id, arm=arm, pair=pair, run_dir=str(run), clock=CLOCK, rule=RULE, times=list(TIMES), n_native_commits=len(commits), truth_sha256=truth_sha, n_fields_evaluated=len(seconds), evaluation_seconds=seconds,
                points=points, final=final, self_check=self_check, complete=True, evaluator_sha256=sha_file(ev.__file__))


# ---- the comparison with the original run -------------------------------------------------------------------------------------------------------------------------------------------------------
def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def compare_sequences(new, old):
    """new, old: native_commits of the two runs.  The sha256 sequence of commits.json, position by position: how long the common prefix is; the first difference and how large it is at that commit (D, F, GF, seconds, lower bound); which run went on
    after the other ended.  The state hash (the bytes of u) is compared too: equal fields in different files would say the file, not the path, differs."""
    k = common_prefix([c['sha256'] for c in new], [c['sha256'] for c in old])
    shorter = min(len(new), len(old))
    record = dict(n_new=len(new), n_old=len(old), common_prefix=k, state_common_prefix=common_prefix([c['state_hash'] for c in new], [c['state_hash'] for c in old]), identical=k == len(new) == len(old), first_difference=None, tail=None)
    record['one_is_prefix_of_the_other'] = k == shorter and len(new) != len(old)
    if k < shorter:
        a, b = new[k], old[k]
        record['first_difference'] = dict(commit=k+1, sha256_new=a['sha256'], sha256_old=b['sha256'], state_hash_new=a['state_hash'], state_hash_old=b['state_hash'], elapsed_new=a['elapsed_seconds'], elapsed_old=b['elapsed_seconds'], D_new=a['D'], D_old=b['D'],
                                          D_new_minus_old=None if a['D'] is None or b['D'] is None else a['D']-b['D'], F_new=a['F'], F_old=b['F'], GF_new=a['GF'], GF_old=b['GF'], min_jd_new=a['min_jd_lower_bound'], min_jd_old=b['min_jd_lower_bound'])
    elif len(new) != len(old):
        longer, rest = ('new', new) if len(new) > len(old) else ('old', old)
        record['tail'] = dict(longer=longer, n_extra=len(rest)-shorter, elapsed_last_common=rest[shorter-1]['elapsed_seconds'] if shorter else None, elapsed_last=rest[-1]['elapsed_seconds'])
    return record


def level_files(task):
    """{event: sha256} of the level files of a task directory, in levels.json: the entry and the certified initializer, before any native commit."""
    levels = read_json(Path(task)/'levels.json')
    return {e['event']: e.get('sha256') for e in levels if e.get('event') in ('preoptimizer_done', 'level_prepared')}


def field_difference(path_new, path_old):
    """(rms, max abs) of the difference of the two u, in mm; None when the files have other shapes."""
    with np.load(path_new) as z:
        a = np.array(z['u'])
    with np.load(path_old) as z:
        b = np.array(z['u'])
    if a.shape != b.shape:
        return None
    d = a-b
    return dict(rms_mm=float(np.sqrt(np.mean(d*d))), max_abs_mm=float(np.abs(d).max()))


def compare_runs(new_run, old_run):
    """reproduction.json: this run against the original of the same arm and pair; the original is only read.  The numbers of the final maps come from the two metrics.json."""
    new_run, old_run = Path(new_run), Path(old_run)
    new_task, old_task = task_dir(new_run), task_dir(old_run)
    new, old = native_commits(new_task), native_commits(old_task)
    record = dict(schema=SCHEMA_REPRODUCTION, new=str(new_run), original=str(old_run), sequence=compare_sequences(new, old))
    a, b = level_files(new_task), level_files(old_task)
    record['start'] = dict(new=a, original=b, entry_same=a.get('preoptimizer_done') == b.get('preoptimizer_done'), initializer_same=a.get('level_prepared') == b.get('level_prepared'))
    new_final, old_final = read_json(new_task/'final_native.json'), read_json(old_task/'final_native.json')
    same = new_final.get('sha256') == old_final.get('sha256')
    x, y = (read_json(new_run/'metrics.json').get('metrics') or {}), (read_json(old_run/'metrics.json').get('metrics') or {})
    record['final'] = dict(sha256_new=new_final.get('sha256'), sha256_old=old_final.get('sha256'), same_file=same, field_difference=None if same else field_difference(new_task/'final_native.npz', old_task/'final_native.npz'),
                           seg35_new=x.get('seg35'), seg35_old=y.get('seg35'), seg35_new_minus_old=None if x.get('seg35') is None or y.get('seg35') is None else x['seg35']-y['seg35'], seg4_new=x.get('seg4'), seg4_old=y.get('seg4'),
                           seg4_new_minus_old=None if x.get('seg4') is None or y.get('seg4') is None else x['seg4']-y['seg4'], sdlogj_official_new=x.get('sdlogj_official'), sdlogj_official_old=y.get('sdlogj_official'))
    record['curve_applies_to_original'] = bool(record['sequence']['identical'])
    record['curve_applies_to_original_up_to_commit'] = record['sequence']['common_prefix']
    record['commit_at_time'] = commits_at_times(new, old, record['sequence']['common_prefix'])
    return record


def commits_at_times(new, old, common, times=TIMES):
    """{str(t): which commit the rule takes at t on each run, and the seconds}.  The same sequence does not make the same commit at the same t: the runs may go at another pace (review m7).  same_field: the same position, inside the common prefix."""
    a, b = choose(new, times), choose(old, times)
    out = {}
    for t in times:
        x, y = (None if a[t] is None else new[a[t]]), (None if b[t] is None else old[b[t]])
        out[str(t)] = dict(commit_new=None if x is None else x['index'], commit_old=None if y is None else y['index'], elapsed_new=None if x is None else x['elapsed_seconds'], elapsed_old=None if y is None else y['elapsed_seconds'],
                           same_field=bool(x and y and x['index'] == y['index'] and x['index'] <= common))
    return out


# ---- the cleanup, after the curve ---------------------------------------------------------------------------------------------------------------------------------------------------------------
def intermediate_files(task):
    """Only committed_*.npz and level_*.npz of the task directory, the same filter as run_seed.cleanup_commits; final_native.npz and every small file stay."""
    return sorted(p for p in Path(task).glob('*.npz') if p.name.startswith(('committed_', 'level_')) and p.name != 'final_native.npz')


def cleanup_after_curve(run, runs_root):
    """Delete this run's own intermediate commits, only under runs_root (<...>/tc_v1/runs of this batch), only after this run's curve.json is complete and its self-check passed, only in a task directory that really is a child of the run (not a symbolic link, no `..` in protocol.json: review I2), only files that commits.json or
    levels.json records with a sha256 (review m3; all files of the 15 originals are).  CurveError otherwise, nothing deleted.  cleanup.json: name, bytes, sha256 of each file as those records say it; the unrecorded ones it names stay."""
    run, runs_root = Path(run).resolve(), Path(runs_root).resolve()
    if runs_root.name != 'runs' or runs_root.parent.name != ROOT_NAME or runs_root not in run.parents or len(run.relative_to(runs_root).parts) != 2:
        raise CurveError(f'{run}: not a run directory of a {ROOT_NAME} root ({runs_root}); nothing deleted')
    if (run/'curve.json').is_file():
        curve = read_json(run/'curve.json')
    else:
        raise CurveError(f'{run}: no curve.json; the commits stay')
    if curve.get('complete') is not True or (curve.get('self_check') or {}).get('passed') is not True:
        raise CurveError(f'{run}: curve.json not complete; the commits stay')
    if not curve.get('run_dir') or Path(curve['run_dir']).resolve() != run:
        raise CurveError(f"{run}: curve.json is of the run {curve.get('run_dir')}; the commits stay")
    if (run/'run_id').is_file() and curve.get('run_id') != (run/'run_id').read_text().strip():
        raise CurveError(f"{run}: curve.json has run_id {curve.get('run_id')}; the commits stay")
    task = task_dir(run).resolve()
    if task.parent != run:
        raise CurveError(f'{run}: its task directory resolves to {task}, not a child of this run; nothing deleted')
    recorded = {c['path']: c.get('sha256') for c in read_json(task/'commits.json')}
    recorded.update({e['path']: e.get('sha256') for e in read_json(task/'levels.json') if e.get('path')})
    gone, unrecorded = [], []
    for p in intermediate_files(task):
        row = dict(name=p.name, bytes=p.stat().st_size, sha256=recorded.get(p.name))
        if row['sha256'] is None:
            unrecorded.append(row)
            continue
        gone.append(row)
        p.unlink()
    kept = [dict(name=p.name, bytes=p.stat().st_size) for p in sorted(task.glob('*.npz'))]
    write_json(run/'cleanup.json', dict(rule='TC plan: after curve.json is complete, only committed_*.npz and level_*.npz of the task directory of this run with a recorded sha256 are deleted (classes ok and handover_only); everything else is kept', deleted=gone, deleted_bytes=sum(g['bytes'] for g in gone), kept_npz=kept, not_deleted_unrecorded=unrecorded))
    return len(gone)


# ---- the time ledger of the 19 runs of S_P (only read) --------------------------------------------------------------------------------------------------------------------------------------------
def ledger_row(root_name, run):
    """One run's backend_timings, counts and what they need to be read: from summary.json and history.json of the task directory."""
    run = Path(run)
    task = task_dir(run)
    summary = read_json(task/'summary.json')
    history = [r for r in read_json(task/'history.json') if r.get('level') == NATIVE_LEVEL]
    prepared = next((e for e in read_json(task/'levels.json') if e.get('event') == 'level_prepared'), {})
    levels = (summary.get('costs') or {}).get('levels') or []
    native = next((c for c in levels if c.get('level') == NATIVE_LEVEL), {})
    return dict(root=root_name, pair=run.name, e2e_seconds=summary.get('e2e_seconds'), solve_call_wall_seconds=summary.get('solve_call_wall_seconds'), final_export_seconds=summary.get('final_export_seconds'), stored_array_io_seconds=summary.get('stored_array_io_seconds'),
                native_commits=summary.get('native_commits'), native_elapsed_last=history[-1]['elapsed_seconds'] if history else None, prepare_seconds=prepared.get('prepare_seconds'), accepted_inner_steps=sum(r.get('inner_accepted_steps') or 0 for r in history),
                rejected_reasons=native.get('rejected_reasons') or {}, backend_timings=summary.get('backend_timings') or {}, counts=summary.get('counts') or {}, evaluation_counts=summary.get('evaluation_counts') or {}, timing_scope=summary.get('backend_timing_scope'))


def ledger_statistics(rows, key):
    """{name: dict(mean, min, max, n)} over the runs, for a dictionary-valued key of the rows."""
    names = sorted({name for row in rows for name in row[key]})
    out = {}
    for name in names:
        values = [row[key][name] for row in rows if isinstance(row[key].get(name), (int, float))]
        out[name] = dict(mean=float(np.mean(values)), min=float(np.min(values)), max=float(np.max(values)), n=len(values))
    return out


def time_ledger(sources):
    """sources: [(root name, run directory)].  Returns the record."""
    rows = [ledger_row(*source) for source in sources]
    scalar = ('e2e_seconds', 'solve_call_wall_seconds', 'final_export_seconds', 'stored_array_io_seconds', 'native_commits', 'native_elapsed_last', 'prepare_seconds', 'accepted_inner_steps')
    wide = [dict(row, _scalars={k: row[k] for k in scalar}) for row in rows]
    stats = dict(scalars=ledger_statistics(wide, '_scalars'), backend_timings=ledger_statistics(rows, 'backend_timings'), counts=ledger_statistics(rows, 'counts'), evaluation_counts=ledger_statistics(rows, 'evaluation_counts'), rejected_reasons=ledger_statistics(rows, 'rejected_reasons'))
    seconds_per_step = [row['native_elapsed_last']/row['accepted_inner_steps'] for row in rows if row['accepted_inner_steps'] and row['native_elapsed_last']]
    return dict(schema='bayesreg.tc.time_ledger.v1', n_runs=len(rows), timing_scope=sorted({str(row['timing_scope']) for row in rows}), statistics=stats, native_seconds_per_accepted_step=dict(mean=float(np.mean(seconds_per_step)), min=float(np.min(seconds_per_step)), max=float(np.max(seconds_per_step))) if seconds_per_step else None, rows=rows)


def ledger_text(ledger):
    """The tables of the ledger, in the words of the brief; the timers are nested and are not added up."""
    def line(name, row, digits=1):
        return f"| {name} | {row['mean']:.{digits}f} | {row['min']:.{digits}f} | {row['max']:.{digits}f} |"
    s = ledger['statistics']
    head = ['| 项 | 均值 | 最小 | 最大 |', '|---|---|---|---|']
    lines = [f"# S_P 的时间账（{ledger['n_runs']} 次运行，只读 summary.json、history.json、levels.json）", '', f"计时器范围：{'; '.join(ledger['timing_scope'])}。计时器是嵌套的，不能相加。开发集单次观测。", '', '## 运行级（秒、个）', ''] + head
    lines += [line(name, row, 2) for name, row in s['scalars'].items()]
    if ledger['native_seconds_per_accepted_step']:
        row = ledger['native_seconds_per_accepted_step']
        lines.append(f"| 原生层秒数 / 接受步 | {row['mean']:.3f} | {row['min']:.3f} | {row['max']:.3f} |")
    lines += ['', '## backend_timings（秒，按均值从大到小）', ''] + head
    lines += [line(name, row, 2) for name, row in sorted(s['backend_timings'].items(), key=lambda item: -item[1]['mean'])]
    lines += ['', '## counts', ''] + head + [line(name, row) for name, row in s['counts'].items() if row['max'] > 0]
    lines += ['', '## evaluation_counts', ''] + head + [line(name, row) for name, row in s['evaluation_counts'].items()]
    lines += ['', '## 原生层被拒的试探（costs.levels[3].rejected_reasons）', ''] + head + [line(name, row) for name, row in s['rejected_reasons'].items()]
    return '\n'.join(lines)+'\n'


def write_ledger(sources, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = time_ledger(sources)
    write_json(out_dir/'time_ledger_sp19.json', ledger)
    (out_dir/'time_ledger_sp19.md').write_text(ledger_text(ledger))
    return ledger


def remove_tree(path, root):
    """shutil.rmtree of a scratch directory of the measure, only below root."""
    path, root = Path(path).resolve(), Path(root).resolve()
    if root not in path.parents:
        raise CurveError(f'{path}: not below {root}; nothing deleted')
    shutil.rmtree(path)
