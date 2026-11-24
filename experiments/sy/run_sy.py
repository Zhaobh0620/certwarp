"""S_Y (runs/20261105-sy/任务书.md, plan.md): the repaired field of Hx24 at 600 steps through the frozen native level, the first 5 pairs, with the curve.  Configuration as S_P: A5P's inputs, 330 s, --geometry-weight-levels 1,1,0.1,0.01, the rest configuration C, the 47 s hold (sy_core.py).  Only the injected field changes.  Not F_B, not a method.

    python -B experiments/sy/run_sy.py taper-check [--json FILE]
        for each pair, the raw field and the repair's nodes, raw', the frozen boundary_taper of raw' against the repaired field by hash, no repair node in the band, the frozen certificate of that field (configuration C, CPU); and the certificate of S_P's and S_X's final maps from their fields.  Reads only; writes only --json.  Exit 8 if any does not pass.  It sets the frozen tree on its own sys.path.  Behind the gate when the queue runs it; by hand only on an idle machine.
    python -B experiments/sy/run_sy.py --dry-run [--scratch DIR]
        prints the 5 runs, the band pair by pair on the real records, their command lines against S_P's cmd.txt, the real frozen parser and prepare on each (a scratch copy of the frozen source; prepare copies pair.npz as bytes, nothing opens its members), the protocol against S_P's, the entry record against S_P's, what the gate says now, a rehearsal of the reading.  No solver, no certificate, no GPU, no unit, no torch, no truth label, nothing written outside the scratch directory.
    python -B experiments/sy/run_sy.py queue [--wait-minutes 30] [--min-host-gib 15]
        the lead's start, in the unit printed by the dry run.  The cheap read-only checks; the gate; the taper check of every pair (8 threads, about two minutes); only then raw' under <root>/entries.  Then 5 runs, serial, one unit tag-sy-S_Y-<ffff>-<mmmm> per run (MemoryMax 12G, as S_P), each behind the gate: no $ROOT/hold, no other tag-* or seq-vp-* unit, 1-minute load < 1.5, no GPU compute process, GPU utilization <= 20%, GPU free >= 15 GiB, /home free >= 100 GiB, /mnt/c free >= --min-host-gib; closed for --wait-minutes: STOP.  60 s between runs.  A run directory is never rerun; failures are kept.
        After each run: classify, evaluate_run, the certificate of the final map from its field in a process of its own (the evaluation's, CPU), the entry's trace, curve.json, then, class ok only, this run's own committed_*.npz and level_*.npz go (cleanup.json).  STOP keeps them.  Any run that does not end ok stops the queue, a later start too (plan.md: 失败即停).  At the end the reading (decide_sy.py); nothing else starts.  After a STOP what has finished is summed up.
    python -B experiments/sy/run_sy.py decide         analysis/decision_sy.json, summary_sy.md, long_curve.csv
    python -B experiments/sy/run_sy.py evaluate-one RUN --arm S_Y --pair PAIR --run-id ID     the queue's step: experiments/d0/evaluate.py evaluate_run, plus the entry's record
    python -B experiments/sy/run_sy.py certify-one RUN --arm S_Y --pair PAIR --run-id ID      the queue's step: field_certificate.json; no image, no label
    python -B experiments/sy/run_sy.py curve-one RUN --arm S_Y --pair PAIR --run-id ID        the queue's step: curve.json; deletes nothing

The only root is sy_v1: every writing command refuses anything else; fd2_v1, s2_v1, seed_v1*, tc_v1, frozen_eb808be are only read (the queue points the frozen worktree's data link at A5P's inputs during prepare, under its lock, as every batch before it).  Exit codes: 0 done, 2 usage, another driver on the root or another root, 8 STOP or a failed check.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT/'experiments/s2'))
sys.path.insert(2, str(ROOT/'experiments/seed'))
sys.path.insert(3, str(ROOT/'experiments/tc'))

import numpy as np  # noqa: E402

import decide_s2 as ds2  # noqa: E402
import decide_sy  # noqa: E402
import run_s2 as rs2  # noqa: E402
import run_seed as core  # noqa: E402
import run_tc as tc  # noqa: E402
import sy_core as sy  # noqa: E402
import tc_curve as curve  # noqa: E402

ROOT_NAME = sy.ROOT_NAME
QUEUE_UNIT = 'tag-sy-queue'
DEFAULT_MIN_HOST_GIB = rs2.DEFAULT_MIN_HOST_GIB          # as S2; can be raised, not lowered
GPU_FREE_MIB = rs2.GPU_FREE_MIB                          # as S2
CERTIFICATE_THREADS = rs2.TAPER_THREADS                  # a thread count cannot change a certificate
MAX_FIELDS = rs2.MAX_FIELDS
Stop, EntryError, GateClosed = core.Stop, core.EntryError, rs2.GateClosed
ENTRY_SAME_KEYS = ('pair', 'oracle', 'shape_zyx', 'source_k', 'charge_seconds')
# What the rehearsal must give on the real records: runs/20261105-fd2/result.md (Hx24 against S_P, against S_X; the task clock; 3 repairs at the wall), runs/20261105-s2/result.md (S_X against its entry; the geometric means of the first commits' lower bounds) and docs/bayesreg/reviews/S_X_结果_复核_20261105.md (the windows), to the digits they print.  The dry run checks them; a test's synthetic world sets this to None.
REHEARSAL_PINS = dict(s_p=(.0187, .0059), s_x=(.0091, .0032), hd95_s_p=-.2110, hd95_s_x=-.0122, entry_mean=143.6, entry_max=170.3, window_s_x=.404, window_s_p=.411, s_x_entry=(.0019, .0185), s_x_entry_sdlogj=.0189, wall=3, bound_s_x_first=2.8e-6, bound_s_p_first=5.3e-5)


def keep_if_not_passed(path, passed):
    """Hard rule 4 (review of the tools, minor 2): a record that did not pass is not overwritten by a later start, it stays under its time.  Returns where it went, or None (no record, or it passed)."""
    path = Path(path)
    doc = core.read_json(path)
    if doc is None or passed(doc):
        return None
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target, n = path.with_name(f'{path.stem}_{stamp}{path.suffix}'), 1
    while target.exists():
        n += 1
        target = path.with_name(f'{path.stem}_{stamp}_{n}{path.suffix}')
    path.rename(target)
    return target


def metrics_passed(doc):
    return (doc.get('certificate') or {}).get('passed') is True and (doc.get('seed') or {}).get('passed') is True


def method_name(arm):
    return f"frozen_eb808be_sy_{core.SECONDS}s_gw{'_'.join(f'{x:g}' for x in core.GEOMETRY)}_{arm}"


def new_run_id(arm, pair):
    return f"sy_{arm}_{sy.short_name(pair)}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def prepare_argv(arm, pair, run, results=None):
    sy.require_cell(arm, pair)
    return core.check_launch([core.PY, '-B', core.frozen_dir(results)/'scripts/run_stage_3d_e48_pg.py', 'prepare', run, *core.OPTS, '--stage', sy.stage_name(arm), '--pair', pair])


def unit_argv(arm, pair, run, root):
    """S_P's unit (run_seed.unit_argv: the same resources and watchdog, the run's own copy of inject_run.py), under this batch's name."""
    return core.unit_argv(arm, pair, run, root, unit=sy.unit_name(arm, pair))


def queue_launch(root=None, repo=core.MAIN_REPO, min_host_gib=DEFAULT_MIN_HOST_GIB):
    """The lead's start string of the queue."""
    root = sy.default_root() if root is None else root
    return core.check_launch(['systemd-run', '--user', f'--unit={QUEUE_UNIT}', '--collect', f'--working-directory={repo}', '--property=MemoryMax=9G', '--setenv=PYTHONDONTWRITEBYTECODE=1',
                              '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1', '--setenv=MKL_NUM_THREADS=1', '--setenv=NUMEXPR_NUM_THREADS=1',
                              core.PY, '-B', 'experiments/sy/run_sy.py', 'queue', '--root', str(root), '--min-host-gib', f'{min_host_gib:g}'])


# ---- the certificates ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def certificate_row(result):
    return dict(status=result.status, reason=result.reason, min_jd_lower_bound=result.min_jd_lower_bound, n_cells_checked=result.n_cells_checked, n_infeasible=result.n_infeasible, n_uncertain=result.n_uncertain, state_hash=result.state_hash, grid_id=result.grid_id, certificate_version=result.certificate_version)


def check_cell(arm, pair, *, results, certify, config, frozen_taper, make_grid, clock=time.perf_counter):
    """One row of the check, or a row with `refused`: nothing here writes.  run_s2.check_cell for this batch's source: passed only if the record's hashes hold, no repair node is in the band, the frozen taper of raw' is the repaired field and the frozen certificate passes that field, with its own state hash."""
    row = dict(arm=arm, pair=pair, short=sy.short_name(pair), passed=False)
    manifest = sy.input_files(pair, results)[3]
    try:
        prime, source, record = sy.construct(arm, pair, manifest, results)
    except sy.BandError as band:
        return dict(row, refused=str(band), kind='band', construction=band.record)
    except EntryError as error:
        return dict(row, refused=str(error), kind='entry')
    grid = rs2.grid_of(manifest, make_grid)
    seed, _ = frozen_taper(prime, grid, sy.s2.TAPER)
    seed = np.ascontiguousarray(seed)
    frozen_sha = sy.array_sha(seed)
    tick = clock()
    result = certify(seed, grid, config=config)
    seconds = clock()-tick
    certificate = certificate_row(result)
    reference = rs2.reference_grid_id(pair, results)
    row.update(construction=record, repaired_u_sha256=source['repaired_sha256'], raw_field=str(source['raw_path']), repair_npz=str(source['repair_path']), frozen_taper_sha256=frozen_sha, frozen_taper_is_repaired=frozen_sha == source['repaired_sha256'], certificate=certificate, certificate_seconds=seconds, grid_id_of_s_p=reference, charge_seconds=source['charge_seconds'], lower_bound_as_the_evaluation=result.min_jd_lower_bound == source['evaluation_lower_bound'])
    problems = []
    if not row['frozen_taper_is_repaired']:
        problems.append(f"the frozen taper of raw' is {frozen_sha}, the repaired field {source['repaired_sha256']}")
    if not curve.certificate_passed(certificate):
        problems.append(f"the certificate did not pass: {result.status}, {result.reason}, lower bound {result.min_jd_lower_bound}, {result.n_infeasible} infeasible, {result.n_uncertain} uncertain")
    if result.state_hash != source['repaired_sha256']:
        problems.append(f"the certificate's state hash {result.state_hash} is not the repaired field {source['repaired_sha256']}")
    if reference is not None and reference != result.grid_id:
        problems.append(f"grid id {result.grid_id}, S_P's transfer certificate on this pair {reference}")
    return dict(row, problems=problems, passed=not problems)


def certify_final(run, grid, *, certify, config, clock=time.perf_counter):
    """The certificate of a run's final map, computed here from the field, with the frozen code and configuration C; the run's own record is only compared.  Opens only the member u of final_native.npz.  passed: certificate_passed, and its state hash is this field's.  ValueError without an exported map."""
    task = curve.task_dir(run)
    path = task/'final_native.npz'
    if not path.is_file():
        raise ValueError(f'{path}: no exported map')
    with np.load(path) as z:
        u = np.ascontiguousarray(z['u'], dtype=np.float64)
    state = sy.array_sha(u)
    tick = clock()
    result = certify(u, grid, config=config)
    seconds = clock()-tick
    certificate = certificate_row(result)
    solver = (core.read_json(task/'final_native.json') or {}).get('certificate') or {}
    return dict(certificate, file=str(path), file_sha256=core.sha_file(path), field_state_hash=state, seconds=seconds, solver_lower_bound=solver.get('min_jd_lower_bound'), solver_state_hash=solver.get('state_hash'), lower_bound_as_the_solver=solver.get('min_jd_lower_bound') == result.min_jd_lower_bound,
                solver_grid_id=solver.get('grid_id'), same_grid_as_the_solver=solver.get('grid_id') == result.grid_id, solver_certificate_version=solver.get('certificate_version'), same_version_as_the_solver=solver.get('certificate_version') == result.certificate_version,
                passed=curve.certificate_passed(certificate) and result.state_hash == state, who='the evaluation, from the field: frozen certify_map, configuration C, CPU; not the run\'s record')


def reference_rows(*, results, certify, config, make_grid, clock=time.perf_counter):
    """S_P's and S_X's final maps, from their fields: only read.  Their metrics.json carry the solvers' records; the brief wants every 过证书 of the summary computed by the evaluation."""
    rows = []
    for pair in sy.PAIRS:
        grid = rs2.grid_of(sy.input_files(pair, results)[3], make_grid)
        for arm, run in (('S_P', sy.reference_dir(pair, results)), ('S_X', sy.s_x_dir(pair, results))):
            row = dict(arm=arm, pair=pair, short=sy.short_name(pair), run=str(run), passed=False)
            try:
                rows.append(dict(row, **certify_final(run, grid, certify=certify, config=config, clock=clock)))
            except (OSError, ValueError, KeyError) as error:
                rows.append(dict(row, refused=str(error)))
    return rows


def taper_check(cells, *, results=None, out=print, certifier=None, clock=time.perf_counter):
    """Every cell and reference, also after a refusal: the report says all of them.  Returns the record; `passed` only if everything passed."""
    certify, config, frozen_taper, make_grid = rs2.frozen_certifier(results) if certifier is None else certifier
    rows = []
    for arm, pair in cells:
        row = check_cell(arm, pair, results=results, certify=certify, config=config, frozen_taper=frozen_taper, make_grid=make_grid, clock=clock)
        rows.append(row)
        if 'refused' in row:
            out(f"{arm} {row['short']}: REFUSED ({row['kind']}): {row['refused']}")
            continue
        c, r = row['certificate'], row['construction']
        out(f"{arm} {row['short']}: nodes {r['n_repair_nodes']} (moved {r['n_changed_nodes']}), in the band {r['n_nodes_in_band']} (moved {r['n_changed_nodes_in_band']}), nearest a face {r['min_face_distance']}; raw_prime {r['raw_prime_sha256'][:16]}; frozen taper of raw_prime {row['frozen_taper_sha256'][:16]}, the repaired field {row['repaired_u_sha256'][:16]}: bitwise {row['frozen_taper_is_repaired'] and r['bitwise_equal_to_repaired']}; "
            f"certificate {c['status']} ({c['reason']}), lower bound {c['min_jd_lower_bound']}, cells {c['n_cells_checked']}, infeasible {c['n_infeasible']}, uncertain {c['n_uncertain']}, state hash is the repaired field {c['state_hash'] == row['repaired_u_sha256']}, lower bound bitwise as the evaluation's record {row['lower_bound_as_the_evaluation']}, grid id as S_P {row['grid_id_of_s_p'] == c['grid_id'] if row['grid_id_of_s_p'] else 'no record'}, {row['certificate_seconds']:.2f} s; passed {row['passed']}"+(f"; PROBLEMS {row['problems']}" if row['problems'] else ''))
    references = reference_rows(results=results, certify=certify, config=config, make_grid=make_grid, clock=clock)
    for row in references:
        out(f"{row['arm']} {row['short']}, final map: "+(f"REFUSED: {row['refused']}" if 'refused' in row else f"certificate {row['status']}, lower bound {row['min_jd_lower_bound']}, bitwise as the solver's record {row['lower_bound_as_the_solver']}, {row['seconds']:.2f} s; passed {row['passed']}"))
    return dict(schema='bayesreg.sy.taper_check.v1', utc=core.utc(), taper_width=sy.s2.TAPER, certificate_config=str(config), rows=rows, references=references, passed=bool(rows) and all(r['passed'] for r in rows) and all(r['passed'] for r in references), n_nodes_in_band=sum((r.get('construction') or {}).get('n_nodes_in_band', 0) for r in rows), band_refusals=[r['short'] for r in rows if r.get('kind') == 'band'], references_failed=[f"{r['arm']} {r['short']}" for r in references if not r['passed']])


def run_taper_check(o, out=print):
    cells = sy.plan_runs()
    out(f'S_Y taper check ({core.utc()}): CPU, reads only; CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")!r}, NUMBA_NUM_THREADS={os.environ.get("NUMBA_NUM_THREADS")!r}; {len(cells)} cells')
    record = taper_check(cells, out=out)
    out(f"passed {record['passed']}; repair nodes in the band in all cells {record['n_nodes_in_band']}; refused for the band {record['band_refusals']}; references that did not pass {record['references_failed']}")
    if o.json:
        core.write_json(o.json, record)
    return 0 if record['passed'] else 8


# ---- the machine ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class System(rs2.System):
    """S2's machine; the steps after a run are this batch's script, and there is one more."""

    def step(self, name, run, arm, pair, run_id, env=None):
        with open(Path(run)/'log.txt', 'ab') as f:
            return subprocess.run(core.check_launch([core.PY, '-B', HERE/'run_sy.py', name, run, '--arm', arm, '--pair', pair, '--run-id', run_id]), stdout=f, stderr=subprocess.STDOUT, env=env).returncode

    def certificate_env(self, root):
        """CPU, 8 threads, the caches under this batch's root (as S2's taper check): certify_map alone does not enable them."""
        cache = Path(root)/'cache'
        return dict(os.environ, CUDA_VISIBLE_DEVICES='', NUMBA_NUM_THREADS=str(CERTIFICATE_THREADS), PYTHONDONTWRITEBYTECODE='1', JD_CURL_JIT_CACHE=str(cache/'numba_jit'), JD_CURL_FIRST_USE_CACHE=str(cache/'first_use'))

    def certify(self, run, arm, pair, run_id):
        """In a process of its own: it sets the frozen tree on its sys.path; evaluate.py takes the repository's."""
        return self.step('certify-one', run, arm, pair, run_id, env=self.certificate_env(Path(run).parents[2]))

    def taper_check(self, out, log_path):
        with open(log_path, 'ab') as f:
            return subprocess.run(core.check_launch([core.PY, '-B', HERE/'run_sy.py', 'taper-check', '--json', out]), stdout=f, stderr=subprocess.STDOUT, env=self.certificate_env(Path(out).parent)).returncode


def busy_reason(system, root, min_host_gib):
    """Why no run may start now, or None: S2's gate with this queue's unit (run_seed.busy_reason, and GPU free >= 15 GiB).  A training that holds the GPU closes it twice: its unit is an other tag-* unit, and it is a compute process."""
    reason = core.busy_reason(system, root, queue_unit=QUEUE_UNIT, host_floor_gib=min_host_gib)
    if reason is None and system.gpu_free_mib() < GPU_FREE_MIB:
        return f'GPU free < {GPU_FREE_MIB} MiB'
    return reason


def wait_for_gate(system, root, minutes, log, min_host_gib):
    started = system.now()
    while True:
        reason = busy_reason(system, root, min_host_gib)
        if reason is None:
            return
        if system.now()-started >= minutes*60:
            raise GateClosed(f'gate closed for {minutes} minutes: {reason}')
        log(f'waiting: {reason}')
        system.sleep(core.POLL_SECONDS)


# ---- one run ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def write_records(run, arm, pair, unit, run_id, entry, files, env_text, repo, results=None):
    """The records of protocol §8 (git.txt, env.txt, data.txt, seed), sy_arm.json, the entry's record and the wrapper's copy, as S2."""
    run = Path(run)
    d, pair_sha, manifest_sha, manifest = files
    git = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(['git', '-C', str(repo), 'diff', '--stat'], capture_output=True, text=True).stdout
    files_sha = '\n'.join(f'{p.relative_to(ROOT)} {core.sha_file(p)}' for pattern in ('experiments/sy/*.py', 'experiments/s2/*.py', 'experiments/tc/tc_curve.py', 'experiments/seed/*.py', 'experiments/d0/*.py', 'experiments/d0/*.json') for p in sorted(ROOT.glob(pattern)))
    frozen = core.frozen_dir(results)
    frozen_head = subprocess.run(['git', '-C', str(frozen), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    frozen_status = subprocess.run(['git', '-C', str(frozen), 'status', '--porcelain'], capture_output=True, text=True).stdout
    untracked = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain', '--', 'experiments/sy', 'experiments/s2', 'experiments/tc', 'experiments/seed', 'experiments/d0', 'docs/bayesreg', 'configs/bayesreg', 'runs'], capture_output=True, text=True).stdout
    (run/'git.txt').write_text(f'repo: {repo}\nrepo HEAD: {git}\nrepo git diff --stat:\n{diff}\nrepo untracked or changed (experiments/sy, experiments/s2, experiments/tc, experiments/seed, experiments/d0, docs/bayesreg, configs/bayesreg, runs):\n{untracked}\n'
                               f'frozen worktree: {frozen}\nfrozen HEAD: {frozen_head}\nfrozen commit expected: {core.FROZEN_COMMIT}\nfrozen git status --porcelain (must be empty):\n{frozen_status}\nfile sha256:\n{files_sha}\n')
    (run/'env.txt').write_text(env_text)
    made = manifest['preprocessing_d0']
    (run/'data.txt').write_text('\n'.join([f'run_id: {run_id}', f'arm: {arm}', f'pair: {pair}', f'unit: {unit}', f"compared with: {sy.reference_dir(pair, results)} and {sy.s_x_dir(pair, results)} (only read)", f"fixed subject: {manifest['fixed_subject']}", f"moving subject: {manifest['moving_subject']}",
                                           f'arm stage directory (prepare source): {d.parent}', f'arm input pair.npz: {d}/pair.npz sha256 {pair_sha}', f'arm input manifest.json: {d}/manifest.json sha256 {manifest_sha}', f"original pair.npz sha256: {made['source_pair_sha256']}", f"original manifest sha256: {made['source_manifest_sha256']}", f"colouring: {made['recipe']}", 'oracle: False',
                                           f"entry field (raw prime): {entry['field_path']} sha256 {entry['field_sha256']} bytes sha256 {entry['u_sha256']}", f"its taper, the repaired field: {entry['repaired_u_sha256']}", f"source raw field: {entry['source_raw_field']} sha256 {entry['source_raw_field_sha256']}", f"repair: {entry['repair_npz']} sha256 {entry['repair_npz_sha256']}; eps {entry['repair_eps']}",
                                           f"entry source run: {entry['source_run_id']}; K {entry['source_k']}; {entry['source_tier']}; channels {core.PREDICTION_SEGMENTER}, {core.PREDICTION_POLICY}; oracle {entry['oracle']}", f"the hold: {entry['charge_seconds']} s; the entry's real task clock: {entry['hold']['real_task_clock']} s", 'truth labels for evaluation: the original stage pair.npz (evaluate.py checks its sha256)'])+'\n')
    (run/'seed').write_text(f'{arm} (runs/20261105-sy/plan.md): the inputs of A5P, no new colouring; the entry field has no random step; solver: configuration C has no random number generator or seed option; GPU reductions may still differ between runs\n')
    core.write_json(run/'sy_arm.json', dict(arm=arm, seconds=core.SECONDS, changes=core.CHANGES, inputs_root=str(core.inputs_root('S_P', results)), compared_with=[str(sy.reference_dir(pair, results)), str(sy.s_x_dir(pair, results))], stage=sy.stage_name(arm), charge_seconds=entry['charge_seconds'], hold=entry['hold'], watchdog_seconds=core.WATCHDOG, method=method_name(arm), input_oracle=False, times=list(curve.TIMES), rule=curve.RULE, clock=curve.CLOCK, wrapper_sha256=core.sha_file(core.HERE/'inject_run.py')))
    core.write_json(run/'seed_entry.json', entry)
    core.write_json(run/'seed_arm.json', dict(arm=arm, seconds=core.SECONDS, changes=core.CHANGES, inputs_root=str(core.inputs_root('S_P', results)), entry_root=entry.get('source_root'), source_arm=entry.get('source_arm'), charge_seconds=entry['charge_seconds'], watchdog_seconds=core.WATCHDOG, method=method_name(arm), input_oracle=False, wrapper_sha256=core.sha_file(core.HERE/'inject_run.py')))
    shutil.copyfile(core.HERE/'inject_run.py', run/'seed_inject.py')


def finish_run(root, run, arm, pair, run_id, t0, system, log):
    """Classify, evaluate, certify the final map from its field, check the entry, the curve, then this run's own commits go.  Returns the state: ok, handover_only, method_failure, infrastructure.  Stop on anything else, with the commits still there."""
    run = Path(run)
    record = system.classify(run, t0)
    cls = (record or {}).get('class')
    with open(run/'log.txt', 'a') as f:
        f.write(f'{core.utc()} class {cls}\n')
    if cls == 'infrastructure':
        core.write_json(run/'queue_done.json', dict(run_id=run_id, state=cls, kind=record.get('kind'), reason=record.get('reason'), done_utc=core.utc()))
        log(f"run {arm} {pair}: infrastructure failure ({record.get('kind')}: {record.get('reason')}); kept whole, not rerun")
        return cls
    if cls not in ('ok', 'handover_only', 'method_failure'):
        raise Stop(f"{run} classified {cls} ({(record or {}).get('reason')}); kept whole, nothing rerun")
    if system.evaluate(run, arm, pair, run_id) != 0:
        raise Stop(f'evaluation failed for {run} (see log.txt); the commits stay')
    metrics = core.read_json(run/'metrics.json') or {}
    trace = metrics.get('seed') or {}
    if trace.get('passed') is not True:                  # also for a method failure, and for a transfer that is not alpha = 1 on the repaired field
        raise Stop(f"{run}: the entry was not injected as recorded: {trace.get('problems')}; class {cls}; kept whole")
    certificate = {}
    if (curve.task_dir(run)/'final_native.npz').is_file():
        if system.certify(run, arm, pair, run_id) != 0:
            raise Stop(f'the certificate of the final map from its field failed for {run} (see log.txt); the commits stay')
        certificate = core.read_json(run/'field_certificate.json') or {}
        if certificate.get('passed') is not True:        # the run's own record said otherwise, or the final map is not certified: a person looks
            raise Stop(f"{run}: the certificate computed from the final map did not pass: {certificate.get('status')}, lower bound {certificate.get('min_jd_lower_bound')}; class {cls}; kept whole")
    elif cls == 'ok':
        raise Stop(f'{run}: class ok without an exported map; kept whole')
    if cls == 'ok' and not (run/'cleanup.json').is_file():      # cleanup.json exists: an earlier start got this far and only queue_done.json is missing
        if system.curve(run, arm, pair, run_id) != 0:
            raise Stop(f'the curve failed for {run} (see log.txt); the commits stay')
        try:
            sy.cleanup_after_curve(run, Path(root)/'runs', curve)
        except (curve.CurveError, OSError) as error:
            raise Stop(f'the cleanup of {run} stopped: {error}') from None
        unrecorded = (core.read_json(run/'cleanup.json') or {}).get('not_deleted_unrecorded') or []
        if unrecorded:
            log(f"{arm} {pair}: {len(unrecorded)} intermediate files without a recorded sha256 stay: {[row['name'] for row in unrecorded]}")
    elif cls == 'handover_only' and not (run/'curve.json').is_file():      # kept whole (classify_run.py, hard rule 4); the curve only describes the certified initializer, and its failure changes nothing
        if system.curve(run, arm, pair, run_id) != 0:
            log(f'{arm} {pair}: handover_only, no curve (see log.txt); kept whole')
    x = metrics.get('metrics') or {}
    summary = f"{cls} status {metrics.get('status')} seg35 {x.get('seg35')} seg4 {x.get('seg4')} sdlogj {x.get('sdlogj_official')} cert from the field {certificate.get('status')}/{certificate.get('passed')} alpha {trace.get('accepted_alpha')} native {metrics.get('native_commits')} fields evaluated {(core.read_json(run/'curve.json') or {}).get('n_fields_evaluated')}"
    core.write_json(run/'queue_done.json', dict(run_id=run_id, state=cls, summary=summary, done_utc=core.utc()))
    log(f'done {arm} {pair}: {summary}')
    return cls


def one_run(o, system, root, arm, pair, entry, files, log, env_text, results, repo, first):
    """Returns the state, or None if nothing ran because it had already ended."""
    run = sy.run_dir(root, arm, pair)
    if (run/'queue_done.json').is_file():
        return None
    unit = sy.unit_name(arm, pair)
    if run.exists():                                     # never rerun: an ended unit is post-processed, anything else stops
        if (run/'exit_code').is_file():
            log(f'resume: {run} has an exit_code; post-processing')
            return finish_run(root, run, arm, pair, (run/'run_id').read_text().strip(), (run/'uptime_before').read_text().strip(), system, log)
        raise Stop(f'{run} exists without exit_code or queue_done.json (an incomplete attempt, kept whole)')
    if not first:
        system.sleep(core.BETWEEN_RUNS_SECONDS)
    wait_for_gate(system, root, o.wait_minutes, log, o.min_host_gib)
    run.parent.mkdir(parents=True, exist_ok=True)
    argv = prepare_argv(arm, pair, run, results)
    target = files[0].parent
    prepare_log = Path(root)/'prepare.log'
    code = system.prepare(argv, target, prepare_log, results)
    if code != 0 or not run.is_dir():
        raise Stop(f'prepare exit {code} for {run} (see {prepare_log})')
    check = system.check_prepared(run, pair, sy.stage_name(arm), files[1], files[2], rs2.config_entry(results))
    core.write_json(run/'prepared_check.json', check)
    if not check.get('ok'):
        raise Stop(f"prepared run differs from the configuration or the arm's inputs: {check.get('problems')}")
    run_id = new_run_id(arm, pair)
    (run/'run_id').write_text(run_id+'\n')
    write_records(run, arm, pair, unit, run_id, entry, files, env_text, repo, results)
    sd = unit_argv(arm, pair, run, root)
    (run/'cmd.txt').write_text(f'# prepare (run from any directory; the frozen script resolves its root from its own path)\n{shlex.join(argv)}\n# worktree link during prepare: {core.frozen_dir(results)}/data/three_d/stage_L2R_val_v1 -> {target}\n# run\n{shlex.join(sd)}\n')
    (run/'service_unit.txt').write_text(unit+'\n')
    t0 = system.uptime()
    (run/'uptime_before').write_text(f'{t0}\n')
    log(f"run {arm} {pair} unit {unit}; load {system.load1()}; /home free {system.free_gib('/home'):.0f} GiB; /mnt/c free {system.free_gib('/mnt/c'):.0f} GiB")
    with open(run/'log.txt', 'a') as f:
        f.write(f'{core.utc()} launch {unit}; {system.state()}\n')
    code = system.launch(sd, run/'log.txt')
    with open(run/'log.txt', 'a') as f:
        f.write(f'{core.utc()} unit ended, systemd-run exit {code}; {system.state()}\n')
    return finish_run(root, run, arm, pair, run_id, t0, system, log)


def reference_problems(results=None):
    """What the reading and the comparisons need is there before any run: S_P's final maps and its entries' records, TC's curves, S_X's records, their exported maps.  Hx24's are source_of's.  Only read."""
    problems = []
    results = core.RESULTS if results is None else results
    for pair in sy.PAIRS:
        final, again = ds2.reference_paths(pair, results, sy.PAIRS)
        missing = [name for name in ('metrics.json', 'seed_entry.json', 'protocol.json') if not (final/name).is_file()]
        if missing:
            problems.append(f'S_P {sy.short_name(pair)}: no {missing} in {final}')
        if not (again/'curve.json').is_file():
            problems.append(f'S_P {sy.short_name(pair)}: no curve.json in {again}')
        s_x = sy.s_x_dir(pair, results)
        missing = [name for name in ('metrics.json', 'curve.json', 'seed_entry.json', 'protocol.json') if not (s_x/name).is_file()]
        if missing:
            problems.append(f'S_X {sy.short_name(pair)}: no {missing} in {s_x}')
        for arm, run in (('S_P', final), ('S_X', s_x)):       # the taper check certifies them from their fields
            if (run/'protocol.json').is_file() and not (curve.task_dir(run)/'final_native.npz').is_file():
                problems.append(f'{arm} {sy.short_name(pair)}: no final_native.npz in {run}')
    return problems


def start_checks(o, system, root, results, repo, log):
    """Everything that can be known before the first run, in this order: the cheap read-only checks; the gate (the next step is 8 threads, and a queue may wait for its turn behind a training); the taper check of every cell (a repair node in the band stops here); only then raw' under <root>/entries.  Returns {(arm, pair): (entry record, input files)}."""
    plan = sy.plan_runs()
    problems = list(system.frozen_problems(results))
    problems += [f'inputs: {p}' for p in system.inputs_problems('S_P', repo, results)]
    problems += reference_problems(results)
    files = {}
    for arm, pair in plan:
        try:
            files[(arm, pair)] = sy.input_files(pair, results)
            sy.source_of(pair, files[(arm, pair)][3], results)
        except EntryError as error:
            problems.append(str(error))
    if problems:
        raise Stop('start checks: '+' | '.join(problems))
    wait_for_gate(system, root, o.wait_minutes, log, o.min_host_gib)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = Path(root)/f'taper_check_{stamp}.json'
    code = system.taper_check(out, Path(root)/'taper_check.log')
    record = core.read_json(out) or {}
    if code != 0 or record.get('passed') is not True:
        bad = [f"{r['arm']} {r['short']}: {r.get('refused') or r.get('problems')}" for r in record.get('rows', []) if not r.get('passed')]+[f"{r['arm']} {r['short']}, final map: {r.get('refused') or r.get('status')}" for r in record.get('references', []) if not r.get('passed')]
        raise Stop(f'the taper check did not pass (exit {code}; {out}; {Path(root)/"taper_check.log"}): '+' | '.join(bad))
    log(f"taper check passed on {len(record['rows'])} cells and {len(record['references'])} final maps: {out}; repair nodes in the band: {record.get('n_nodes_in_band')}")
    prepared = {}
    for arm, pair in plan:
        try:
            prepared[(arm, pair)] = (sy.write_entry(root, arm, pair, files[(arm, pair)][3], results), files[(arm, pair)])
        except EntryError as error:
            raise Stop(f'entry {arm} {sy.short_name(pair)}: {error}') from None
    log(f'start checks passed: frozen worktree, inputs record, the references, all {len(prepared)} cells and their entries under {Path(root)/"entries"}')
    return prepared


def load_evaluate():
    ev = core.load_evaluate()
    ev.STAGE = core.MAIN_REPO/'data/three_d/stage_L2R_val_v1'    # a worktree has no data link
    return ev


def write_analysis(root, *, ev=None, results=None, pin=decide_sy.HX24_PIN):
    """The reading, after fd2_v1's records are checked against the pins (review of the tools, minor 6: `decide` by hand reads them too).  SystemExit otherwise, nothing written."""
    results = core.RESULTS if results is None else results
    problems = sy.pin_problems(results)
    if problems:
        raise SystemExit('the reading refuses: fd2_v1 is not what the pins say: '+' | '.join(problems))
    return decide_sy.write_analysis(root, sy.plan_runs(), sy.PAIRS, sy.short_name, ev=load_evaluate() if ev is None else ev, utc=core.utc, results=results, pin=pin)


def batch_record(root, o, repo):
    """batch.json: what defines the batch is checked at every start; a difference is a STOP before any run."""
    fixed = dict(schema='bayesreg.sy.batch.v1', seconds=core.SECONDS, changes=core.CHANGES, frozen_commit=core.FROZEN_COMMIT, hold=sy.HOLD, times=list(curve.TIMES), arm=sy.ARM, source=sy.SOURCE_LABEL, pins_sha256=core.sha_file(sy.__file__), reference_final='seed_v1', reference_curve='tc_v1', s_x='s2_v1', decide_sha256=curve.sha_file(decide_sy.__file__),
                 code_sha256=decide_sy.code_hashes())        # review of the tools, important 2: what the runs and the reading go through, run_sy.py itself included
    head = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    this = dict(utc=core.utc(), min_host_gib=o.min_host_gib, repo_head=head)
    batch = core.read_json(Path(root)/'batch.json')
    if batch is not None:
        different = {k: (batch.get(k), v) for k, v in fixed.items() if batch.get(k) != v}
        if 'code_sha256' in different:
            different['code_sha256'] = decide_sy.code_changed(batch.get('code_sha256') or {}, fixed['code_sha256'])
        if different:
            raise Stop(f'batch.json differs from this batch: {different}')
    core.write_json(Path(root)/'batch.json', dict(fixed, started_utc=this['utc'] if batch is None else batch.get('started_utc'), starts=([] if batch is None else batch.get('starts', []))+[this]))


def not_ok_text(arm, pair, state):
    return f'{arm} {sy.short_name(pair)} ended {state}: a run that does not end ok stops the queue (plan.md); kept whole, not rerun; the reading is not decidable; this root is over (a new one needs a new plan)'


def known_failure(root):
    """Why this root is over, or None: a run of the plan ended other than ok (review of the tools, minor 1: said at once, not behind the gate and the certificates).  A run with no queue_done.json is not known to have failed: an ok run whose steps after it stopped is post-processed again."""
    for arm, pair in sy.plan_runs():
        state = (core.read_json(sy.run_dir(root, arm, pair)/'queue_done.json') or {}).get('state')
        if state not in (None, 'ok'):
            return not_ok_text(arm, pair, state)
    return None


def queue_body(o, system, root, log, results, repo, *, ev=None, pin=decide_sy.HX24_PIN):
    plan = sy.plan_runs()
    reason = known_failure(root)
    if reason:
        raise Stop(reason)
    (root/'cache/numba_jit').mkdir(parents=True, exist_ok=True)
    (root/'cache/first_use').mkdir(parents=True, exist_ok=True)
    prepared = start_checks(o, system, root, results, repo, log)
    env_text = core.environment_text(repo) if isinstance(system, System) else 'test\n'
    batch_record(root, o, repo)
    first = True
    try:
        for arm, pair in plan:
            entry, files = prepared[(arm, pair)]
            state = one_run(o, system, root, arm, pair, entry, files, log, env_text, results, repo, first)
            done = (core.read_json(sy.run_dir(root, arm, pair)/'queue_done.json') or {}).get('state')
            if done != 'ok':                             # the plan's rule: the reading needs all 5, a person looks before another 7 minutes go
                raise Stop(not_ok_text(arm, pair, done))
            if state is not None:
                first = False
    except Stop:                                         # what finished is summed up, then STOP
        if any((sy.run_dir(root, a, p)/'metrics.json').is_file() for a, p in plan):
            try:
                result = write_analysis(root, ev=ev, results=results, pin=pin)
                log(f"the queue stops; reading of what finished: {result['reading'][:400]}")
            except (Exception, SystemExit) as error:
                log(f'the queue stops; the summary failed: {error!r}')
        raise
    try:
        result = write_analysis(root, ev=ev, results=results, pin=pin)
    except (Exception, SystemExit) as error:             # the runs stay; by hand: run_sy.py decide
        raise Stop(f'the reading failed: {error!r}') from None
    log(f"reading: {result['reading']}")
    log('Handed to the lead: no next step starts here')
    return 0


def queue(o, system=None, results=None, repo=None, ev=None, pin=decide_sy.HX24_PIN):
    results = core.RESULTS if results is None else results
    repo = core.MAIN_REPO if repo is None else repo
    problem = sy.root_problem(o.root, results)
    if problem is None and not o.min_host_gib >= DEFAULT_MIN_HOST_GIB:     # can be raised, not lowered; written so that nan is refused too
        problem = f'--min-host-gib {o.min_host_gib:g}: the floor is at least {DEFAULT_MIN_HOST_GIB:g} GiB; nothing read, nothing moved'
    if problem:                                          # before the machine and before the lock file
        print(problem, file=sys.stderr)
        return 2
    system = System() if system is None else system
    root = Path(o.root)
    root.mkdir(parents=True, exist_ok=True)
    lock = open(root/'queue.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f'another driver is on {root}', file=sys.stderr)
        return 2

    def log(text):
        with open(root/'queue.log', 'a') as f:
            f.write(f'{core.utc()} {text}\n')
    log(f'start: root {root}; repo {repo}; {len(sy.plan_runs())} runs in order; /mnt/c floor {o.min_host_gib:g} GiB')
    if (root/'queue_stop.json').exists():                # evidence, kept under its time
        (root/'queue_stop.json').rename(root/f"queue_stop_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    try:
        return queue_body(o, system, root, log, results, repo, ev=ev, pin=pin)
    except Stop as stop:
        log(f'STOP: {stop}')
        core.write_json(root/'queue_stop.json', dict(reason=str(stop), utc=core.utc()))
        return 8


# ---- the steps after a run --------------------------------------------------------------------------------------------------------------------------------------------------------------
def evaluate_one(o, *, ev=None):
    sy.require_cell(o.arm, o.pair)
    ev = load_evaluate() if ev is None else ev
    rec = ev.evaluate_run(Path(o.run), o.arm, o.pair, o.run_id, method=method_name(o.arm))
    trace, _ = sy.entry_trace(o.run)
    rec['seed'] = dict(trace, arm=core.read_json(Path(o.run)/'seed_arm.json'))
    rec['sy'] = core.read_json(Path(o.run)/'sy_arm.json')
    kept = keep_if_not_passed(Path(o.run)/'metrics.json', metrics_passed)
    if kept:
        print(f'{o.arm} {sy.short_name(o.pair)}: the earlier metrics.json did not pass, kept as {kept.name}')
    core.write_json(Path(o.run)/'metrics.json', rec)
    x, c = rec.get('metrics') or {}, rec.get('certificate') or {}
    print(f"{o.arm} {sy.short_name(o.pair)}: seg35 {x.get('seg35')} seg4 {x.get('seg4')} SDlogJ {x.get('sdlogj_official')} solver's cert {c.get('status')} passed {c.get('passed')} alpha {trace.get('accepted_alpha')} entry check {trace.get('passed')}")
    return 0


def certify_one(o, *, results=None, certifier=None):
    """field_certificate.json: the certificate of the final map, from the field.  No image, no label; its grid from the manifest."""
    sy.require_cell(o.arm, o.pair)
    certify, config, _, make_grid = rs2.frozen_certifier(results) if certifier is None else certifier
    grid = rs2.grid_of(sy.input_files(o.pair, results)[3], make_grid)
    try:
        record = certify_final(o.run, grid, certify=certify, config=config)
    except (OSError, ValueError, KeyError) as error:
        print(f'{o.arm} {sy.short_name(o.pair)}: no certificate: {error}', flush=True)
        return 8
    kept = keep_if_not_passed(Path(o.run)/'field_certificate.json', lambda doc: doc.get('passed') is True)
    if kept:
        print(f'{o.arm} {sy.short_name(o.pair)}: the earlier field_certificate.json did not pass, kept as {kept.name}')
    core.write_json(Path(o.run)/'field_certificate.json', dict(record, run_id=o.run_id, arm=o.arm, pair=o.pair, certificate_config=str(config), utc=core.utc(), env=dict(CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'), NUMBA_NUM_THREADS=os.environ.get('NUMBA_NUM_THREADS'))))
    print(f"{o.arm} {sy.short_name(o.pair)}: certificate from the field {record['status']}, lower bound {record['min_jd_lower_bound']}, bitwise as the solver's record {record['lower_bound_as_the_solver']}, {record['seconds']:.2f} s; passed {record['passed']}")
    return 0


def curve_one(o, *, ev=None):
    """curve.json; deletes nothing (finish_run cleans up afterwards)."""
    sy.require_cell(o.arm, o.pair)
    ev = load_evaluate() if ev is None else ev
    try:
        record = curve.curve_of_run(o.run, o.arm, o.pair, o.run_id, ev, clock=time.perf_counter)
    except (curve.CurveError, OSError) as error:         # OSError: a run that never finished an outer step may have no commits.json at all
        print(f'{o.arm} {sy.short_name(o.pair)}: the curve stops: {error}', flush=True)
        return 8
    core.write_json(Path(o.run)/'curve.json', record)
    print(f"{o.arm} {sy.short_name(o.pair)}: curve of {record['n_fields_evaluated']} fields, {sum(record['evaluation_seconds']):.1f} s")
    return 0


# ---- the dry run ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def command_difference(arm, pair, results=None):
    """This batch's two command lines against S_P's cmd.txt on this pair, with the run directory, the root, the unit and the stage put in place: the tokens that still differ."""
    root = sy.default_root(results)
    run, old = sy.run_dir(root, arm, pair), sy.reference_dir(pair, results)
    old_root = old.parents[2]
    old_prepare, old_unit = tc.command_tokens((old/'cmd.txt').read_text())
    old_unit_name = (old/'service_unit.txt').read_text().strip()
    swap = [(str(old), str(run)), (str(old_root), str(root)), (old_unit_name, sy.unit_name(arm, pair))]
    new_prepare, new_unit = [str(t) for t in prepare_argv(arm, pair, run, results)], [str(t) for t in unit_argv(arm, pair, run, root)]
    differ = {}
    for name, new, was in (('prepare', new_prepare, tc.normalized(old_prepare, swap)), ('unit', new_unit, tc.normalized(old_unit, swap))):
        if new != was:
            differ[name] = [(a, b) for a, b in zip(new, was) if a != b]+([f'lengths {len(new)} {len(was)}'] if len(new) != len(was) else [])
    return differ


def entry_difference(entry, pair, results=None):
    """This cell's entry record against S_P's seed_entry.json on this pair: the keys that may differ are the injection's own (all the rest of the record); the hold must be S_P's."""
    old = core.read_json(sy.reference_dir(pair, results)/'seed_entry.json') or {}
    same = {key: (old.get(key), entry.get(key)) for key in ENTRY_SAME_KEYS}
    bad = {key: pair_of for key, pair_of in same.items() if pair_of[0] != pair_of[1]}
    return dict(differs_in=sorted(k for k in set(old) | set(entry) if old.get(k) != entry.get(k)), hold=(old.get('charge_seconds'), entry.get('charge_seconds')), must_be_equal_but_differ=bad)


def rehearse_reading(scratch, out, results=None, ev=None, pins=None, pin=decide_sy.HX24_PIN):
    """The reading on a made-up batch: S_Y's final maps are the metrics of Hx24's own repaired fields (fd2_v1), so every Delta is 0 and the code must say 原生层加不了 seg35; the rest of the records are copies of S_X's (s2_v1).  Formats and the comparison arms only; no field, no label; deleted with the scratch directory.  On the way, the numbers S_Y's comparisons must give are FD2's and the S_X review's: `pins` (the real records only)."""
    results = core.RESULTS if results is None else results
    ev = load_evaluate() if ev is None else ev
    root = scratch/sy.ROOT_NAME
    for arm, pair in sy.plan_runs():
        source_run, target = sy.s_x_dir(pair, results), sy.run_dir(root, arm, pair)
        target.mkdir(parents=True)
        files = sy.input_files(pair, results)
        source = sy.source_of(pair, files[3], results)
        doc = core.read_json(source_run/'metrics.json')
        after = core.read_json(sy.repair_dir(pair, results)/'repair_metrics.json')['after']
        repaired = source['repaired_sha256']
        doc.update(arm=arm, metrics=after['metrics'], seed=dict(doc.get('seed') or {}, passed=True, problems=[], accepted_alpha=1., charge_seconds=sy.HOLD, transfer_attempts=[dict(alpha=1., passed=True, state_hash=repaired)], repaired_u_sha256=repaired))
        core.write_json(target/'metrics.json', doc)
        solver = doc['certificate']
        core.write_json(target/'field_certificate.json', dict(status=solver['status'], passed=True, state_hash=solver['state_hash'], file_sha256=solver['final_npz_sha256'], min_jd_lower_bound=solver['min_jd_lower_bound'], lower_bound_as_the_solver=True))
        shutil.copyfile(source_run/'curve.json', target/'curve.json')
        shutil.copyfile(source_run/'protocol.json', target/'protocol.json')
        task = curve.task_dir(target)
        task.mkdir(parents=True)
        for name in ('summary.json', 'history.json', 'inner_attempts.json'):
            shutil.copyfile(curve.task_dir(source_run)/name, task/name)
        core.write_json(target/'seed_entry.json', dict(charge_seconds=sy.HOLD, source_seconds_total=source['seconds_entry'], seconds_repair=source['seconds_repair'], seconds_warmup=source['seconds_warmup']))
        core.write_json(target/'queue_done.json', dict(state='ok', summary='rehearsal'))
    result = write_analysis(root, ev=ev, results=results, pin=pin)
    d = result['description']
    windows = d['rejection']['windows']
    first = decide_sy.window_name(0, 160)
    t = d['time']
    out(f"reading on the metrics of Hx24's own repaired fields: code {result['reading_code']} (Δ seg35 {result['delta_seg35']['mean']}, Δ seg4 {result['delta_seg4']['mean']}); files {sorted(p.name for p in (root/'analysis').iterdir())}")
    out(f"the comparison arms on the way (real records; the rows named S_Y hold Hx24's numbers here, not results of this batch): against S_P seg35 / seg4 {d['against_s_p']['seg35']['mean']:+.4f} / {d['against_s_p']['seg4']['mean']:+.4f}, HD95 {d['against_s_p']['hd95']['mean']:+.4f}; against S_X {d['against_s_x']['seg35']['mean']:+.4f} / {d['against_s_x']['seg4']['mean']:+.4f}, HD95 {d['against_s_x']['hd95']['mean']:+.4f} (runs/20261105-fd2/result.md: +.0187 / +.0059, −.2110; +.0091 / +.0032, −.0122)")
    out(f"S_X against its entry Hx@2400, the reference of the reading: seg35 / seg4 {d['s_x_against_its_entry']['seg35']['mean']:+.4f} / {d['s_x_against_its_entry']['seg4']['mean']:+.4f}, official SDlogJ {d['s_x_against_its_entry']['sdlogj_official']['mean']:+.4f} (runs/20261105-s2/result.md: +.0019 / +.0185, +.0189)")
    out(f"the geometric means of the first commits' lower bounds: S_X {d['bounds']['s_x_first']}, S_P (tc_v1) {d['bounds']['s_p_first']} (runs/20261105-s2/result.md: 2.8e-6, 5.3e-5); repairs at FD2's wall among the 5 entries: {t['wall']} (runs/20261105-fd2/result.md: 3)")
    out(f"the rejected share in the first 160 s: S_X {windows['s_x'][first] if windows['s_x'] else None}, S_P {windows['s_p'][first] if windows['s_p'] else None} (the review of S_X: .404, .411)")
    out(f"the entry's real task clock: mean {t['hx24']['entry']}, longest {t['hx24']['entry_max']} s (runs/20261105-fd2/result.md: 143.6 / 170.3; the hold stays {sy.HOLD} s); the estimate of FL's reading 9 for the arms that ran: S_X {t['s_x']['total']} s (its result: about 436), S_P {t['s_p']['total']} s (about 445; here the task clock with the hold taken out, about 280 s, not the rounded 277 s), Hx24 alone {t['hx24']['total']} s")
    if result['reading_code'] != decide_sy.NONE or result['delta_seg35']['mean'] != 0.:
        raise SystemExit(f"the rehearsal of the reading did not give what it must: {result['reading']}")
    pins = REHEARSAL_PINS if pins is None else pins
    if pins:
        have = dict(s_p=(d['against_s_p']['seg35']['mean'], d['against_s_p']['seg4']['mean']), s_x=(d['against_s_x']['seg35']['mean'], d['against_s_x']['seg4']['mean']), hd95_s_p=d['against_s_p']['hd95']['mean'], hd95_s_x=d['against_s_x']['hd95']['mean'], entry_mean=t['hx24']['entry'], entry_max=t['hx24']['entry_max'], window_s_x=windows['s_x'][first], window_s_p=windows['s_p'][first],
                    s_x_entry=(d['s_x_against_its_entry']['seg35']['mean'], d['s_x_against_its_entry']['seg4']['mean']), s_x_entry_sdlogj=d['s_x_against_its_entry']['sdlogj_official']['mean'], wall=t['wall'], bound_s_x_first=d['bounds']['s_x_first'], bound_s_p_first=d['bounds']['s_p_first'])
        tolerance = dict(entry_mean=.051, entry_max=.051, window_s_x=5.1e-4, window_s_p=5.1e-4, wall=0., bound_s_x_first=5.1e-8, bound_s_p_first=5.1e-7)
        for key, want in pins.items():
            for a, b in zip(have[key] if isinstance(want, tuple) else (have[key],), want if isinstance(want, tuple) else (want,)):
                if abs(a-b) > tolerance.get(key, 5.1e-5):
                    raise SystemExit(f'the rehearsal read {key} = {have[key]}, the documents say {want}')
        out('all of these as the documents say')


def dry_run(scratch=None, out=print, results=None, repo=None, system=None, ev=None, pins=None, pin=decide_sy.HX24_PIN, prepare=None, torch_counts=True):
    """The scratch directory goes, however the dry run ends.  torch_counts: torch in this process is a problem; a test's process may hold it from elsewhere (the test then forbids the import itself)."""
    results = core.RESULTS if results is None else results
    repo = core.MAIN_REPO if repo is None else repo
    made = scratch is None
    scratch = Path(tempfile.mkdtemp(prefix='sy_dry_')) if made else Path(scratch)
    if not made and scratch.exists():
        raise SystemExit(f'{scratch} exists (nothing is overwritten)')
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        return dry_run_steps(scratch, out, results, repo, System() if system is None else system, ev, pins, pin, frozen_prepare if prepare is None else prepare, torch_counts)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def frozen_prepare(tokens, cwd, env):
    """The real parser and prepare, on the scratch copy: it copies pair.npz and manifest.json as bytes and hashes them; no member is opened (F/scripts/run_stage_3d_e48_pg.py:592-612: np.load only under a contrast mix, which this configuration does not have)."""
    return subprocess.run(tokens, cwd=cwd, env=env, capture_output=True, text=True)


def dry_run_steps(scratch, out, results, repo, system, ev, pins, pin, prepare, torch_counts):
    root = sy.default_root(results)
    link = core.frozen_dir(results)/'data/three_d/stage_L2R_val_v1'
    link_before = os.readlink(link)
    existed = root.exists()
    plan = sy.plan_runs()
    out(f'S_Y dry run ({core.utc()}): no solver, no certificate, no GPU (CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")!r}), no unit; everything written under a scratch directory, deleted at the end.')
    out(f'{len(plan)} runs: S_Y on the first 5 pairs; source {sy.FD2_ROOT} {sy.REPAIR_ARM}_{sy.REPAIR_TIER}; --seconds {core.SECONDS}; hold {sy.HOLD} s, as S_P, a constant; watchdog {core.WATCHDOG} s; root {root}; /mnt/c floor {DEFAULT_MIN_HOST_GIB:g} GiB, free now {system.free_gib("/mnt/c"):.0f} GiB; /home free now {system.free_gib("/home"):.0f} GiB; times {curve.TIMES} and the final map')
    problems = []
    selection = json.loads((core.MAIN_REPO/'data/three_d/stage_L2R_val_v1/selection.json').read_text())
    names = [row['pair'] for row in selection['pairs']]
    out(f"selection.json first 5 == the plan's pairs: {tuple(names[:5]) == sy.PAIRS}")
    if tuple(names[:5]) != sy.PAIRS:
        problems.append('selection.json differs from the plan')
    sys.path.insert(0, str(ROOT))
    from bayesreg.io.guard import check_pair_subjects, load_dev_allowlist
    allowlist = load_dev_allowlist(ROOT/'configs/bayesreg/dev_subjects.json')

    out("\n==== the band and the entries, pair by pair, on the real records: the raw field, repair.npz's nodes and values, their sha256 against the pins; raw' on the scratch disk, taper(raw') against the repaired field by hash and bytes; the wrapper's own load_field.  The certificates are the queue's taper check, behind the gate: not run here ====")
    copy = scratch/'frozen_copy'
    n_files = core.frozen_copy(copy, results)
    out(f'frozen source copied: {n_files} files equal to the freeze hashes; the worktree link stays')
    cells = {}
    entry_scratch = scratch/'entries'
    for arm, pair in plan:
        files = sy.input_files(pair, results)
        check_pair_subjects(files[3]['fixed_subject'], files[3]['moving_subject'], allowlist)
        try:
            entry = sy.write_entry(entry_scratch, arm, pair, files[3], results)
        except sy.BandError as band:
            problems.append(f'{arm} {sy.short_name(pair)}: BAND: {band}')
            out(f'{arm} {sy.short_name(pair)}: BAND: {band}; counts {band.record}')
            continue
        except EntryError as error:
            problems.append(str(error))
            out(f'{arm} {sy.short_name(pair)}: REFUSED: {error}')
            continue
        differ = entry_difference(entry, pair, results)
        if differ['must_be_equal_but_differ']:
            problems.append(f"{arm} {pair}: entry record {differ['must_be_equal_but_differ']}")
        c = entry['construction']
        out(f"{arm} {sy.short_name(pair)}: raw {entry['source_raw_field']} + {entry['repair_npz']}; nodes {c['n_repair_nodes']}, moved {c['n_changed_nodes']}, in the band {c['n_nodes_in_band']} (moved {c['n_changed_nodes_in_band']}), nearest a face {c['min_face_distance']} (the band is < {sy.s2.TAPER}); taper(raw') {entry['tapered_sha256'][:16]} = the repaired field {entry['repaired_u_sha256'][:16]}: {entry['tapered_sha256'] == entry['repaired_u_sha256']}, bitwise {c['bitwise_equal_to_repaired']}; hold {entry['charge_seconds']} s, the entry's real task clock {entry['hold']['real_task_clock']:.1f} s; evaluation's lower bound {entry['evaluation_lower_bound']}; subjects on the allowlist; "
            f"entry record against S_P's seed_entry.json differs in {differ['differs_in']}; hold {differ['hold']}; must be equal: {differ['must_be_equal_but_differ'] or 'equal'}")
        cells[(arm, pair)] = (entry, files)
        sy.entry_paths(entry_scratch, arm, pair)[0].unlink()             # 158 MiB each: only the record is needed from here on

    out('\n==== the input record, colour table and registry checks (main checkout) ====')
    bad = system.inputs_problems('S_P', repo, results)
    out(f"{core.inputs_root('S_P', results)}: {'ok' if not bad else bad}")
    problems += bad
    bad = reference_problems(results)
    out(f"S_P's and S_X's records (seed_v1, tc_v1, s2_v1): {'there' if not bad else bad}")
    problems += bad

    out(f'\n==== the {len(plan)} runs: the real parser and prepare of the frozen source (scratch copy), check_prepared, the protocol against S_P of the same pair; the two command lines against its cmd.txt; then the unit that the queue would start ====')
    prepared = scratch/'prepared'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    scratch_link = copy/'data/three_d/stage_L2R_val_v1'
    scratch_link.parent.mkdir(parents=True, exist_ok=True)
    for k, (arm, pair) in enumerate(plan, 1):
        entry, files = cells.get((arm, pair), (None, None))
        if files is None:
            continue
        shown = sy.run_dir(root, arm, pair)
        out(f"# item {k}: arm {arm}, pair {pair}, inputs {files[0]}, watchdog {core.WATCHDOG} s, unit {sy.unit_name(arm, pair)}, run id sy_{arm}_{sy.short_name(pair)}_<UTC>, compared with {sy.reference_dir(pair, results)} and {sy.s_x_dir(pair, results)}")
        out(shlex.join(prepare_argv(arm, pair, shown, results)))
        out(shlex.join(unit_argv(arm, pair, shown, root)))
        if scratch_link.is_symlink():
            scratch_link.unlink()
        scratch_link.symlink_to(files[0].parent)
        here = prepared/arm/pair
        here.parent.mkdir(parents=True, exist_ok=True)
        tokens = prepare_argv(arm, pair, here, results)
        tokens[2] = str(copy/'scripts/run_stage_3d_e48_pg.py')
        tick = time.perf_counter()
        done = prepare(tokens, scratch, env)
        if done.returncode != 0:
            raise SystemExit(f'prepare refused {arm} {pair}: {done.stdout[-1500:]} {done.stderr[-1500:]}')
        check = system.check_prepared(here, pair, sy.stage_name(arm), files[1], files[2], rs2.config_entry(results))
        if not check['ok']:
            raise SystemExit(f"check_prepared refused {arm} {pair}: {check['problems']}")
        differs = core.protocol_differences(core.read_json(here/'protocol.json'), core.read_json(sy.reference_dir(pair, results)/'protocol.json'))
        commands = command_difference(arm, pair, results)
        out(f'parser, prepare and check_prepared ok ({time.perf_counter()-tick:.1f} s); protocol against S_P differs in: {json.dumps(differs, ensure_ascii=False)}; the command lines differ in: {json.dumps(commands, ensure_ascii=False)}')
        if set(differs) != {'stage'} or differs['stage'] != ('SEED-S_P', sy.stage_name(arm)):
            problems.append(f'{arm} {pair}: the protocol differs from S_P in {sorted(differs)}')
        if set(commands) != {'prepare'} or commands['prepare'] != [(sy.stage_name(arm), 'SEED-S_P')]:
            problems.append(f'{arm} {pair}: the command lines differ from S_P: {commands}')
        shutil.rmtree(here)

    out('\n==== the gate, as it stands now (the queue would wait behind it; nothing is started) ====')
    reason = busy_reason(system, root, DEFAULT_MIN_HOST_GIB)
    out(f"{'closed: '+reason if reason else 'open'}")

    out("\n==== the reading (made-up batch from Hx24's own metrics and S_X's records: formats only, not results) ====")
    rehearse_reading(scratch, out, results, ev, pins, pin)

    out('\n==== the disk ====')
    rows = rs2.disk_estimate(plan, results)
    out(f"intermediate commits per run, from S_P's cleanup.json on the same pairs: {min(rows.values()):.2f}-{max(rows.values()):.2f} GiB (serial, each deleted after its curve, before the next starts); stays per run: as TC, about 262 MiB; raw' under {root}/entries: 158 MiB per cell, kept: 5 cells 0.8 GiB; about 2.1 GiB in all (the brief's estimate says about 1 GiB)")
    out(f"evaluation: {MAX_FIELDS} fields at most a run, one certificate from the field; S_X's runs took about 7.5 minutes each with these steps: about 38 minutes, plus a few minutes for the 15 certificates of the taper check and the 5 of the evaluation")

    out('\n==== the start strings (the lead) ====')
    out(shlex.join(queue_launch(root, repo)))
    out('the taper check runs inside the queue, behind the gate; by hand, only on an idle machine: '+shlex.join([core.PY, '-B', 'experiments/sy/run_sy.py', 'taper-check']))

    if os.readlink(link) != link_before:
        problems.append("the frozen worktree's link changed")
    if root.exists() != existed:
        problems.append(f'{root} was created')
    if torch_counts and 'torch' in sys.modules:
        problems.append('torch was imported')
    out(f"\nthe frozen worktree's link unchanged: {link_before}; {root} {'exists' if existed else 'not created'}; no solver run; no certificate run; the scratch directory goes when this ends")
    out(f"torch imported by this dry run: {'torch' in sys.modules}; problems: {problems or 'none'}")
    return 8 if problems else 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['--dry-run']:
        d = argparse.ArgumentParser()
        d.add_argument('--dry-run', action='store_true')
        d.add_argument('--scratch', default=None)
        return dry_run(scratch=d.parse_args(argv).scratch)
    p = argparse.ArgumentParser(description='S_Y: see the module docstring')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('taper-check').add_argument('--json', default=None)
    q = sub.add_parser('queue')
    q.add_argument('--root', default=str(sy.default_root()))
    q.add_argument('--wait-minutes', type=int, default=30)
    q.add_argument('--min-host-gib', type=float, default=DEFAULT_MIN_HOST_GIB)
    sub.add_parser('decide').add_argument('--root', default=str(sy.default_root()))
    for name in ('evaluate-one', 'certify-one', 'curve-one'):
        e = sub.add_parser(name)
        e.add_argument('run')
        e.add_argument('--arm', required=True)
        e.add_argument('--pair', required=True)
        e.add_argument('--run-id', required=True)
    o = p.parse_args(argv)
    if o.command == 'taper-check':
        return run_taper_check(o)
    if o.command == 'queue':
        return queue(o)
    if o.command == 'decide':
        problem = sy.root_problem(o.root)
        if problem:
            print(problem, file=sys.stderr)
            return 2
        result = write_analysis(o.root)
        if result['decide_code']['same'] is False:
            print(f"decide_sy.py has changed since the queue's first start: {result['decide_code']}", file=sys.stderr)
        return 0
    sy.require_cell(o.arm, o.pair)                       # before any file
    problem = sy.own_run_problem(o.run, o.arm, o.pair)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    return dict(zip(('evaluate-one', 'certify-one', 'curve-one'), (evaluate_one, certify_one, curve_one)))[o.command](o)


if __name__ == '__main__':
    sys.exit(main())
