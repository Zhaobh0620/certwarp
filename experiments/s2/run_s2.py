"""S2 (runs/20261105-s2/任务书.md, plan.md): the repaired entries through the frozen native level, with the curve.  Configuration as S_P: A5P's inputs, 330 s, --geometry-weight-levels 1,1,0.1,0.01, the rest configuration C.  Only the injected field changes (S_X: and the hold).  Not F_B, not a method.

    python -B experiments/s2/run_s2.py taper-check [--arms S_Pp S_F] [--json FILE] [--sx-root R --sx-arm A --sx-checkpoint N --sx-pins FILE --sx-pins-sha256 H]
        the second taper (s2_core.py): for each cell, the raw field and the repair's nodes, raw', the frozen boundary_taper of raw' against the repaired field by hash, no repair node in the band, the frozen certificate of that field (configuration C, CPU).  Reads only; writes only --json.  Exit 8 if any cell does not pass: a node in the band stops here.  It sets the frozen tree on its own sys.path.
    python -B experiments/s2/run_s2.py --dry-run [--scratch DIR] [--sx-*]
        prints all the runs, their command lines against S_P's cmd.txt, the real frozen parser and prepare on each (a scratch copy of the frozen source), the protocol against S_P's, the entry record against S_P's, a rehearsal of the reading, the disk estimate.  No solver, no GPU, no unit, nothing written outside the scratch directory.
    python -B experiments/s2/run_s2.py sx-pins --sx-root R --sx-arm A --sx-checkpoint N
        prints the pins file of S_X, read from FL's records, on stdout (the lead saves it and takes its sha256); writes nothing, judges nothing: the queue and taper-check do that on the file.
    python -B experiments/s2/run_s2.py queue [--wait-minutes 30] [--min-host-gib 15] [--sx-*]
        the lead's start, in the unit printed by the dry run.  Start checks first: the cheap read-only ones (the frozen worktree, the inputs record, S_P's records, the records of every cell); then the gate; then taper-check on every cell of the plan (8 threads for about a minute), and only then raw' under <root>/entries.  Anything wrong stops before any run.  Then S_Pp (2 runs), S_F (5 runs), with the options S_X (5 runs): serial, one unit tag-s2-<arm>-<ffff>-<mmmm> per run (MemoryMax 12G, as S_P), each behind the gate: no $ROOT/hold,
        no other tag-* or seq-vp-* unit, 1-minute load < 1.5, no GPU compute process, GPU utilization <= 20%, GPU free >= 15 GiB, /home free >= 100 GiB, /mnt/c free >= --min-host-gib; closed for --wait-minutes: STOP.  60 s between runs.  A run directory is never rerun; failures are kept.
        After each run: classify, evaluate_run, the entry's trace, curve.json, then, class ok only, this run's own committed_*.npz and level_*.npz go (cleanup.json).  STOP keeps them: a prepare or check failure, a class `unsure`, an evaluation or curve failure, an entry not injected as recorded or a transfer that is not alpha = 1 on the repaired field, 3 infrastructure failures in a row, S_F's first two pairs both with 0 native commits.  At the end the reading (decide_s2.py); nothing else starts.  After a STOP what has finished is summed up.
    python -B experiments/s2/run_s2.py decide         analysis/decision_s2.json, summary_s2.md, long_curve.csv
    python -B experiments/s2/run_s2.py evaluate-one RUN --arm ARM --pair PAIR --run-id ID     the queue's step: experiments/d0/evaluate.py evaluate_run, plus the entry's record
    python -B experiments/s2/run_s2.py curve-one RUN --arm ARM --pair PAIR --run-id ID        the queue's step: curve.json; deletes nothing

The only root is s2_v1: every writing command refuses anything else; ef_v1, mc0p_v1*, seed_v1*, tc_v1, fl_v1 are only read.  Exit codes: 0 done, 2 usage, another driver on the root or another root, 8 STOP or a failed check.
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
sys.path.insert(1, str(ROOT/'experiments/seed'))
sys.path.insert(2, str(ROOT/'experiments/tc'))

import numpy as np  # noqa: E402

import decide_s2  # noqa: E402
import run_seed as core  # noqa: E402
import run_tc as tc  # noqa: E402
import s2_core as s2  # noqa: E402
import tc_curve as curve  # noqa: E402

ROOT_NAME = s2.ROOT_NAME
QUEUE_UNIT = 'tag-s2-queue'
DEFAULT_MIN_HOST_GIB = 15.                               # as TC; can be raised, not lowered
GPU_FREE_MIB = 15*1024                                   # as TC
TAPER_THREADS = 8                                        # the queue's certificates, behind the gate (plan revision 1); a thread count cannot change a certificate
MAX_FIELDS = len(curve.TIMES)+1
Stop, EntryError = core.Stop, core.EntryError
SX_FLAGS = ('--sx-root', '--sx-arm', '--sx-checkpoint', '--sx-pins', '--sx-pins-sha256')
ENTRY_PINS = (.0152, .0235, .0184, .0268, .0161, .0165, .0186, .0212)  # what the reading's entry columns must give on the real records: runs/20261104-ef/result.md, read 2 and the long arms; to 4 decimals; the dry run checks them


class GateClosed(Stop):
    """The gate stayed closed for its time."""


def method_name(arm):
    return f"frozen_eb808be_s2_{core.SECONDS}s_gw{'_'.join(f'{x:g}' for x in core.GEOMETRY)}_{arm}"


def new_run_id(arm, pair):
    return f"s2_{arm}_{s2.short_name(pair)}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def config_entry(results=None):
    """S_P's configuration, in b1_spec's form."""
    return core.b1_entry('S_P', results)


def prepare_argv(arm, pair, run, results=None):
    s2.require_cell(arm, pair)
    return core.check_launch([core.PY, '-B', core.frozen_dir(results)/'scripts/run_stage_3d_e48_pg.py', 'prepare', run, *core.OPTS, '--stage', s2.stage_name(arm), '--pair', pair])


def unit_argv(arm, pair, run, root):
    """S_P's unit (run_seed.unit_argv: the same resources and watchdog, the run's own copy of inject_run.py), under this batch's name."""
    return core.unit_argv(arm, pair, run, root, unit=s2.unit_name(arm, pair))


def sx_argv(sx):
    return [] if sx is None else [token for flag, value in zip(SX_FLAGS, (sx.root, sx.arm, sx.checkpoint, sx.pins, sx.pins_sha256)) for token in (flag, str(value))]


def queue_launch(root=None, repo=core.MAIN_REPO, min_host_gib=DEFAULT_MIN_HOST_GIB, sx=None):
    """The lead's start string of the queue."""
    root = s2.default_root() if root is None else root
    return core.check_launch(['systemd-run', '--user', f'--unit={QUEUE_UNIT}', '--collect', f'--working-directory={repo}', '--property=MemoryMax=9G', '--setenv=PYTHONDONTWRITEBYTECODE=1',
                              '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1', '--setenv=MKL_NUM_THREADS=1', '--setenv=NUMEXPR_NUM_THREADS=1',
                              core.PY, '-B', 'experiments/s2/run_s2.py', 'queue', '--root', str(root), '--min-host-gib', f'{min_host_gib:g}', *sx_argv(sx)])


# ---- the taper check ------------------------------------------------------------------------------------------------------------------------------------------------------------------
def frozen_certifier(results=None):
    """(certify_map, configuration C, boundary_taper, make_grid_3d) of the frozen tree: STOP unless jd_curl comes from it.  certify_map's own default is not configuration C: passed explicitly, as run_mc0.certificate_config (a test compares them)."""
    frozen = core.frozen_dir(results)
    sys.path.insert(0, str(frozen))
    import jd_curl
    if frozen.resolve() not in Path(jd_curl.__file__).resolve().parents:
        raise SystemExit(f'jd_curl comes from {jd_curl.__file__}, not from {frozen}')
    from jd_curl.three_d.feasibility import PrescreenCertificateConfig, certify_map
    from jd_curl.three_d.grid import make_grid_3d
    from jd_curl.three_d.preopt import boundary_taper
    return certify_map, PrescreenCertificateConfig(delta=0., max_depth=48, max_boxes=255, tile_size=65536), boundary_taper, make_grid_3d


def grid_of(manifest, make_grid_3d):
    """The grid of the manifest, as bayesreg.io.pair reads it; no image is opened."""
    (nz, ny, nx), (hx, hy, hz), (x0, y0, z0) = manifest['shape_zyx'], manifest['spacing_xyz'], manifest.get('origin_xyz', (0., 0., 0.))
    return make_grid_3d(nx=nx, ny=ny, nz=nz, hx=hx, hy=hy, hz=hz, x0=x0, y0=y0, z0=z0, unit='mm')


def reference_grid_id(pair, results=None):
    """The grid id of S_P's transfer certificate on this pair (only read), or None."""
    run = s2.reference_dir(pair, results)
    if not (run/'protocol.json').is_file():
        return None
    levels = core.read_json(curve.task_dir(run)/'levels.json') or []
    prepared = next((e for e in levels if e.get('event') == 'level_prepared'), {})
    attempts = (prepared.get('transfer_recovery') or {}).get('attempts') or [{}]
    return (attempts[0].get('certificate') or {}).get('grid_id')


def check_cell(arm, pair, *, results, sx, certify, config, frozen_taper, make_grid, clock=time.perf_counter):
    """One row of the check, or a row with `refused`: nothing here writes.  passed only if the record's hashes hold, no repair node is in the band, the frozen taper of raw' is the repaired field and the frozen certificate passes that field, with its own state hash."""
    row = dict(arm=arm, pair=pair, short=s2.short_name(pair), passed=False)
    manifest = s2.input_files(pair, results)[3]
    try:
        prime, source, record = s2.construct(arm, pair, manifest, results, sx)
    except s2.BandError as band:
        return dict(row, refused=str(band), kind='band', construction=band.record)
    except EntryError as error:
        return dict(row, refused=str(error), kind='entry')
    grid = grid_of(manifest, make_grid)
    seed, _ = frozen_taper(prime, grid, s2.TAPER)
    seed = np.ascontiguousarray(seed)
    frozen_sha = s2.array_sha(seed)
    tick = clock()
    result = certify(seed, grid, config=config)
    seconds = clock()-tick
    certificate = dict(status=result.status, reason=result.reason, min_jd_lower_bound=result.min_jd_lower_bound, n_cells_checked=result.n_cells_checked, n_infeasible=result.n_infeasible, n_uncertain=result.n_uncertain, state_hash=result.state_hash, grid_id=result.grid_id, certificate_version=result.certificate_version)
    reference = reference_grid_id(pair, results)
    row.update(construction=record, repaired_u_sha256=source['repaired_sha256'], raw_field=str(source['raw_path']), repair_npz=str(source['repair_path']), frozen_taper_sha256=frozen_sha, frozen_taper_is_repaired=frozen_sha == source['repaired_sha256'], certificate=certificate, certificate_seconds=seconds, grid_id_of_s_p=reference, charge_seconds=source['charge_seconds'], lower_bound_as_the_evaluation=None if source['evaluation_lower_bound'] is None else result.min_jd_lower_bound == source['evaluation_lower_bound'])
    problems = []
    if not row['frozen_taper_is_repaired']:
        problems.append(f"the frozen taper of raw' is {frozen_sha}, the repaired field {source['repaired_sha256']}")
    if not curve.certificate_passed(certificate):
        problems.append(f"the certificate did not pass: {result.status}, {result.reason}, lower bound {result.min_jd_lower_bound}, {result.n_infeasible} infeasible, {result.n_uncertain} uncertain")
    if result.state_hash != source['repaired_sha256']:
        problems.append(f"the certificate's state hash {result.state_hash} is not the repaired field {source['repaired_sha256']}")
    if reference is not None and reference != result.grid_id:
        problems.append(f"grid id {result.grid_id}, S_P's transfer certificate on this pair {reference}")
    if arm == 'S_Pp':                                    # "only alpha differs": the same raw field as S_P's entry on this pair
        bad = s_pp_source_problems(pair, source, results)
        row['raw_field_is_s_ps_entry'] = not bad
        problems += bad
    return dict(row, problems=problems, passed=not problems)


def s_pp_source_problems(pair, source, results=None):
    """S_Pp's source field must be the very field S_P was given on this pair."""
    recorded = core.read_json(s2.reference_dir(pair, results)/'seed_entry.json') or {}
    return [] if recorded.get('field_sha256') == source['raw_sha256'] else [f"S_P's entry on this pair is {recorded.get('field_sha256')}, this cell's source {source['raw_sha256']}"]


def taper_check(cells, *, results=None, sx=None, out=print, certifier=None):
    """Every cell, also after a refusal: the report says all of them.  Returns the record; `passed` only if every cell passed."""
    certify, config, frozen_taper, make_grid = frozen_certifier(results) if certifier is None else certifier
    rows = []
    for arm, pair in cells:
        row = check_cell(arm, pair, results=results, sx=sx, certify=certify, config=config, frozen_taper=frozen_taper, make_grid=make_grid)
        rows.append(row)
        if 'refused' in row:
            out(f"{arm} {row['short']}: REFUSED ({row['kind']}): {row['refused']}")
            continue
        c, r = row['certificate'], row['construction']
        out(f"{arm} {row['short']}: nodes {r['n_repair_nodes']} (moved {r['n_changed_nodes']}), in the band {r['n_nodes_in_band']} (moved {r['n_changed_nodes_in_band']}), nearest a face {r['min_face_distance']}; raw_prime {r['raw_prime_sha256'][:16]}; frozen taper of raw_prime {row['frozen_taper_sha256'][:16]}, the repaired field {row['repaired_u_sha256'][:16]}: bitwise {row['frozen_taper_is_repaired'] and r['bitwise_equal_to_repaired']}; "
            f"certificate {c['status']} ({c['reason']}), lower bound {c['min_jd_lower_bound']}, cells {c['n_cells_checked']}, infeasible {c['n_infeasible']}, uncertain {c['n_uncertain']}, state hash is the repaired field {c['state_hash'] == row['repaired_u_sha256']}, lower bound bitwise as the evaluation's record {row['lower_bound_as_the_evaluation']}, grid id as S_P {row['grid_id_of_s_p'] == c['grid_id'] if row['grid_id_of_s_p'] else 'no record'}, {row['certificate_seconds']:.2f} s; "+(f"raw field is S_P's entry on this pair {row['raw_field_is_s_ps_entry']}; " if 'raw_field_is_s_ps_entry' in row else '')+f"passed {row['passed']}"+(f"; PROBLEMS {row['problems']}" if row['problems'] else ''))
    return dict(schema='bayesreg.s2.taper_check.v1', utc=core.utc(), taper_width=s2.TAPER, certificate_config=str(config), rows=rows, passed=bool(rows) and all(r['passed'] for r in rows), n_nodes_in_band=sum((r.get('construction') or {}).get('n_nodes_in_band', 0) for r in rows), band_refusals=[r['short'] for r in rows if r.get('kind') == 'band'])


def run_taper_check(o, out=print):
    sx = sx_from(o)
    wanted = o.arms if o.arms else list(s2.QUEUE_ARMS)+(['S_X'] if sx else [])
    cells = [(arm, pair) for arm in wanted for pair in s2.ARMS[arm]['pairs']] if all(a in s2.ARMS for a in wanted) else None
    if cells is None:
        raise SystemExit(f'S2 has only the arms {tuple(s2.ARMS)}')
    out(f'S2 taper check ({core.utc()}): CPU, reads only; CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")!r}, NUMBA_NUM_THREADS={os.environ.get("NUMBA_NUM_THREADS")!r}; {len(cells)} cells')
    record = taper_check(cells, sx=sx, out=out)
    out(f"passed {record['passed']}; repair nodes in the band in all cells {record['n_nodes_in_band']}; refused for the band {record['band_refusals']}")
    if o.json:
        core.write_json(o.json, record)
    return 0 if record['passed'] else 8


def print_pins(o, out=print, results=None):
    """The pins file of S_X from FL's records, on stdout; only the root of FL is read."""
    root = s2.fl_root(results)
    if Path(o.sx_root).resolve() != root.resolve():
        raise SystemExit(f'--sx-root {o.sx_root}: S_X reads only {root}')
    try:
        out(json.dumps(s2.sx_pins(root, o.sx_arm, o.sx_checkpoint), indent=1, ensure_ascii=False))
    except EntryError as error:
        raise SystemExit(str(error)) from None
    return 0


# ---- the machine ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class System(core.System):
    """SEED's machine; the arms' inputs and the steps after a run are this batch's."""

    def inputs_problems(self, arm, repo, results=None):
        return core.b1_spec.check_one_input(config_entry(results), repo)

    def step(self, name, run, arm, pair, run_id):
        with open(Path(run)/'log.txt', 'ab') as f:
            return subprocess.run(core.check_launch([core.PY, '-B', HERE/'run_s2.py', name, run, '--arm', arm, '--pair', pair, '--run-id', run_id]), stdout=f, stderr=subprocess.STDOUT).returncode

    def evaluate(self, run, arm, pair, run_id):
        return self.step('evaluate-one', run, arm, pair, run_id)

    def curve(self, run, arm, pair, run_id):
        return self.step('curve-one', run, arm, pair, run_id)

    def taper_check(self, out, sx, log_path, arms):
        """run_s2.py taper-check on every cell, CPU, in a process of its own (it sets the frozen tree on its sys.path); the record goes to `out`.  The caches under this batch's root, as its units (review minor 9); certify_map alone does not enable them."""
        cache = Path(out).parent/'cache'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NUMBA_NUM_THREADS=str(TAPER_THREADS), PYTHONDONTWRITEBYTECODE='1', JD_CURL_JIT_CACHE=str(cache/'numba_jit'), JD_CURL_FIRST_USE_CACHE=str(cache/'first_use'))
        with open(log_path, 'ab') as f:
            return subprocess.run(core.check_launch([core.PY, '-B', HERE/'run_s2.py', 'taper-check', '--arms', *arms, '--json', out, *sx_argv(sx)]), stdout=f, stderr=subprocess.STDOUT, env=env).returncode


def busy_reason(system, root, min_host_gib):
    """Why no run may start now, or None: SEED's gate with this queue's unit and the floor, and GPU free >= 15 GiB (as TC)."""
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
    """The records of protocol §8 (git.txt, env.txt, data.txt, seed), s2_arm.json, the entry's record and the wrapper's copy, as S_P."""
    run = Path(run)
    d, pair_sha, manifest_sha, manifest = files
    git = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(['git', '-C', str(repo), 'diff', '--stat'], capture_output=True, text=True).stdout
    files_sha = '\n'.join(f'{p.relative_to(ROOT)} {core.sha_file(p)}' for pattern in ('experiments/s2/*.py', 'experiments/tc/tc_curve.py', 'experiments/seed/*.py', 'experiments/d0/*.py', 'experiments/d0/*.json') for p in sorted(ROOT.glob(pattern)))
    frozen = core.frozen_dir(results)
    frozen_head = subprocess.run(['git', '-C', str(frozen), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    frozen_status = subprocess.run(['git', '-C', str(frozen), 'status', '--porcelain'], capture_output=True, text=True).stdout
    untracked = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain', '--', 'experiments/s2', 'experiments/tc', 'experiments/seed', 'experiments/d0', 'docs/bayesreg', 'configs/bayesreg', 'runs'], capture_output=True, text=True).stdout
    (run/'git.txt').write_text(f'repo: {repo}\nrepo HEAD: {git}\nrepo git diff --stat:\n{diff}\nrepo untracked or changed (experiments/s2, experiments/tc, experiments/seed, experiments/d0, docs/bayesreg, configs/bayesreg, runs):\n{untracked}\n'
                               f'frozen worktree: {frozen}\nfrozen HEAD: {frozen_head}\nfrozen commit expected: {core.FROZEN_COMMIT}\nfrozen git status --porcelain (must be empty):\n{frozen_status}\nfile sha256:\n{files_sha}\n')
    (run/'env.txt').write_text(env_text)
    made = manifest['preprocessing_d0']
    (run/'data.txt').write_text('\n'.join([f'run_id: {run_id}', f'arm: {arm}', f'pair: {pair}', f'unit: {unit}', f"compared with: {s2.reference_dir(pair, results)} (only read)", f"fixed subject: {manifest['fixed_subject']}", f"moving subject: {manifest['moving_subject']}",
                                           f'arm stage directory (prepare source): {d.parent}', f'arm input pair.npz: {d}/pair.npz sha256 {pair_sha}', f'arm input manifest.json: {d}/manifest.json sha256 {manifest_sha}', f"original pair.npz sha256: {made['source_pair_sha256']}", f"original manifest sha256: {made['source_manifest_sha256']}", f"colouring: {made['recipe']}", 'oracle: False',
                                           f"entry field (raw prime): {entry['field_path']} sha256 {entry['field_sha256']} bytes sha256 {entry['u_sha256']}", f"its taper, the repaired field: {entry['repaired_u_sha256']}", f"source raw field: {entry['source_raw_field']} sha256 {entry['source_raw_field_sha256']}", f"repair: {entry['repair_npz']} sha256 {entry['repair_npz_sha256']}; eps {entry['repair_eps']}",
                                           f"entry source run: {entry['source_run_id']}; K {entry['source_k']}; tier {entry['source_tier']}; oracle {entry['oracle']}", 'truth labels for evaluation: the original stage pair.npz (evaluate.py checks its sha256)'])+'\n')
    (run/'seed').write_text(f'{arm} (runs/20261105-s2/plan.md): the inputs of A5P, no new colouring; the entry field has no random step; solver: configuration C has no random number generator or seed option; GPU reductions may still differ between runs\n')
    arm_record = dict(arm=arm, seconds=core.SECONDS, changes=core.CHANGES, inputs_root=str(core.inputs_root('S_P', results)), compared_with=str(s2.reference_dir(pair, results)), stage=s2.stage_name(arm), charge_seconds=entry['charge_seconds'], watchdog_seconds=core.WATCHDOG, method=method_name(arm), input_oracle=False, role=s2.ARMS[arm]['role'], decides=s2.ARMS[arm]['decides'], times=list(curve.TIMES), rule=curve.RULE, clock=curve.CLOCK, wrapper_sha256=core.sha_file(core.HERE/'inject_run.py'))
    core.write_json(run/'s2_arm.json', arm_record)
    core.write_json(run/'seed_entry.json', entry)
    core.write_json(run/'seed_arm.json', dict(arm=arm, seconds=core.SECONDS, changes=core.CHANGES, inputs_root=str(core.inputs_root('S_P', results)), entry_root=entry.get('source_root'), source_arm=entry.get('source_arm'), charge_seconds=entry['charge_seconds'], watchdog_seconds=core.WATCHDOG, method=method_name(arm), input_oracle=False, wrapper_sha256=core.sha_file(core.HERE/'inject_run.py')))
    shutil.copyfile(core.HERE/'inject_run.py', run/'seed_inject.py')


def finish_run(root, run, arm, pair, run_id, t0, system, log):
    """Classify, evaluate, check the entry, the curve, then this run's own commits go.  Returns the state: ok, handover_only, method_failure, infrastructure.  Stop on anything else, with the commits still there."""
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
    if trace.get('passed') is not True:                  # also for a method failure, and for a transfer that is not alpha = 1 on the repaired field: the walk of the injection path must hold before another 6 minutes go
        raise Stop(f"{run}: the entry was not injected as recorded: {trace.get('problems')}; class {cls}; kept whole")
    if cls == 'ok' and not (run/'cleanup.json').is_file():      # cleanup.json exists: an earlier start got this far and only queue_done.json is missing
        if system.curve(run, arm, pair, run_id) != 0:
            raise Stop(f'the curve failed for {run} (see log.txt); the commits stay')
        try:
            s2.cleanup_after_curve(run, Path(root)/'runs', curve)
        except (curve.CurveError, OSError) as error:
            raise Stop(f'the cleanup of {run} stopped: {error}') from None
        unrecorded = (core.read_json(run/'cleanup.json') or {}).get('not_deleted_unrecorded') or []
        if unrecorded:
            log(f"{arm} {pair}: {len(unrecorded)} intermediate files without a recorded sha256 stay: {[row['name'] for row in unrecorded]}")
    elif cls == 'handover_only' and not (run/'curve.json').is_file():      # kept whole (classify_run.py, hard rule 4): this is the failure the probe looks for; no commit to take, the curve only describes the certified initializer, and its failure stops nothing
        if system.curve(run, arm, pair, run_id) != 0:
            log(f'{arm} {pair}: handover_only, no curve (see log.txt); kept whole')
    x, c = metrics.get('metrics') or {}, metrics.get('certificate') or {}
    summary = f"{cls} status {metrics.get('status')} seg35 {x.get('seg35')} seg4 {x.get('seg4')} sdlogj {x.get('sdlogj_official')} cert {c.get('status')}/{c.get('passed')} alpha {trace.get('accepted_alpha')} native {metrics.get('native_commits')} fields evaluated {(core.read_json(run/'curve.json') or {}).get('n_fields_evaluated')}"
    core.write_json(run/'queue_done.json', dict(run_id=run_id, state=cls, summary=summary, done_utc=core.utc()))
    log(f'done {arm} {pair}: {summary}')
    return cls


def one_run(o, system, root, arm, pair, entry, files, log, env_text, results, repo, first):
    """Returns the state, or None if nothing ran because it had already ended."""
    run = s2.run_dir(root, arm, pair)
    if (run/'queue_done.json').is_file():
        return None
    unit = s2.unit_name(arm, pair)
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
    check = system.check_prepared(run, pair, s2.stage_name(arm), files[1], files[2], config_entry(results))
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


def reference_problems(plan, results=None):
    """What the reading needs of S_P is there before any run: its final maps and its entries' records, TC's curves for the first 5 pairs.  Only read."""
    problems = []
    for pair in dict.fromkeys(pair for _, pair in plan):
        final, again = decide_s2.reference_paths(pair, core.RESULTS if results is None else results, s2.PAIRS)
        missing = [name for name in ('metrics.json', 'seed_entry.json', 'protocol.json') if not (final/name).is_file()]
        if missing:
            problems.append(f'S_P {s2.short_name(pair)}: no {missing} in {final}')
        if pair in s2.PAIRS and not (again/'curve.json').is_file():
            problems.append(f'S_P {s2.short_name(pair)}: no curve.json in {again}')
    return problems


def start_checks(o, system, root, results, repo, log, plan, sx):
    """Everything that can be known before the first run, in this order: the cheap read-only checks; the gate (review I1: the next step is 8 threads, and a queue may wait for its turn behind a timed run); the taper check of every cell (a repair node in the band stops here); only then raw' under <root>/entries.  Returns {(arm, pair): (entry record, input files)}."""
    problems = list(system.frozen_problems(results))
    problems += [f'inputs: {p}' for p in system.inputs_problems('S_P', repo, results)]
    problems += reference_problems(plan, results)
    files = {}
    for arm, pair in plan:
        try:
            files[(arm, pair)] = s2.input_files(pair, results)
            s2.cell_source(arm, pair, files[(arm, pair)][3], results, sx)
        except EntryError as error:
            problems.append(str(error))
    if problems:
        raise Stop('start checks: '+' | '.join(problems))
    wait_for_gate(system, root, o.wait_minutes, log, o.min_host_gib)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = Path(root)/f'taper_check_{stamp}.json'
    code = system.taper_check(out, sx, Path(root)/'taper_check.log', list(dict.fromkeys(arm for arm, _ in plan)))
    record = core.read_json(out) or {}
    if code != 0 or record.get('passed') is not True:
        bad = [f"{r['arm']} {r['short']}: {r.get('refused') or r.get('problems')}" for r in record.get('rows', []) if not r.get('passed')]
        raise Stop(f'the taper check did not pass (exit {code}; {out}; {Path(root)/"taper_check.log"}): '+' | '.join(bad))
    log(f"taper check passed on {len(record['rows'])} cells: {out}; repair nodes in the band: {record.get('n_nodes_in_band')}")
    prepared = {}
    for arm, pair in plan:
        try:
            prepared[(arm, pair)] = (s2.write_entry(root, arm, pair, files[(arm, pair)][3], results, sx), files[(arm, pair)])
        except EntryError as error:
            raise Stop(f'entry {arm} {s2.short_name(pair)}: {error}') from None
    log(f'start checks passed: frozen worktree, inputs record, S_P\'s records, all {len(prepared)} cells and their entries under {Path(root)/"entries"}')
    return prepared


def trailing_infrastructure(root, plan):
    n = 0
    for arm, pair in plan:
        done = core.read_json(s2.run_dir(root, arm, pair)/'queue_done.json')
        if done is None:
            continue
        n = n+1 if done.get('state') == 'infrastructure' else 0
    return n


def load_evaluate():
    ev = core.load_evaluate()
    ev.STAGE = core.MAIN_REPO/'data/three_d/stage_L2R_val_v1'    # a worktree has no data link
    return ev


def write_analysis(root, *, ev=None, results=None):
    results = core.RESULTS if results is None else results
    return decide_s2.write_analysis(root, s2.plan_runs(with_x=True), s2.PAIRS, s2.S_PP_PAIRS, s2.short_name, ev=load_evaluate() if ev is None else ev, utc=core.utc, results=results)


def batch_record(root, o, repo, sx, plan):
    """batch.json: what defines the batch is checked at every start; a difference is a STOP before any run.  S_X may join later: it is recorded per start."""
    fixed = dict(schema='bayesreg.s2.batch.v1', seconds=core.SECONDS, changes=core.CHANGES, frozen_commit=core.FROZEN_COMMIT, hold_s_p=s2.HOLD_S_P, times=list(curve.TIMES), arms=list(s2.QUEUE_ARMS), reference_final='seed_v1, seed_v1b', reference_curve='tc_v1', decide_sha256=curve.sha_file(decide_s2.__file__))     # review minor 4: the reading's code is pinned at the first start
    head = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    this = dict(utc=core.utc(), min_host_gib=o.min_host_gib, repo_head=head, plan=[f'{a} {s2.short_name(p)}' for a, p in plan], s_x=None if sx is None else dict(root=sx.root, arm=sx.arm, checkpoint=sx.checkpoint, pins=sx.pins, pins_sha256=sx.pins_sha256))
    batch = core.read_json(Path(root)/'batch.json')
    if batch is not None:
        different = {k: (batch.get(k), v) for k, v in fixed.items() if batch.get(k) != v}
        if different:
            raise Stop(f'batch.json differs from this batch: {different}')
    core.write_json(Path(root)/'batch.json', dict(fixed, started_utc=this['utc'] if batch is None else batch.get('started_utc'), starts=([] if batch is None else batch.get('starts', []))+[this]))


def queue_body(o, system, root, log, results, repo, sx, *, ev=None):
    plan = s2.plan_runs(with_x=sx is not None)
    (root/'cache/numba_jit').mkdir(parents=True, exist_ok=True)
    (root/'cache/first_use').mkdir(parents=True, exist_ok=True)
    prepared = start_checks(o, system, root, results, repo, log, plan, sx)
    env_text = core.environment_text(repo) if isinstance(system, System) else 'test\n'
    batch_record(root, o, repo, sx, plan)
    first = True
    try:
        for arm, pair in plan:
            entry, files = prepared[(arm, pair)]
            state = one_run(o, system, root, arm, pair, entry, files, log, env_text, results, repo, first)
            if arm == 'S_F' and pair == s2.PAIRS[1]:     # also on a later start: a second start does not step over the rule
                reason = s2.two_empty_s_f(root)
                if reason:
                    raise Stop(reason)
            if state is None:
                continue
            first = False
            if trailing_infrastructure(root, plan) >= core.MAX_CONSECUTIVE_INFRA:
                raise Stop(f'{core.MAX_CONSECUTIVE_INFRA} infrastructure failures in a row')
    except Stop:                                         # what finished is summed up, then STOP
        if any((s2.run_dir(root, a, p)/'metrics.json').is_file() for a, p in plan):
            try:
                result = write_analysis(root, ev=ev, results=results)
                log(f"the queue stops; reading of what finished: {result['reading'][:400]}")
            except (Exception, SystemExit) as error:
                log(f'the queue stops; the summary failed: {error!r}')
        raise
    try:
        result = write_analysis(root, ev=ev, results=results)
    except (Exception, SystemExit) as error:             # the runs stay; by hand: run_s2.py decide
        raise Stop(f'the reading failed: {error!r}') from None
    log(f"reading: {result['reading']}")
    log('Handed to the lead: no next step starts here')
    return 0


def queue(o, system=None, results=None, repo=None, ev=None):
    results = core.RESULTS if results is None else results
    repo = core.MAIN_REPO if repo is None else repo
    problem = s2.root_problem(o.root, results)
    if problem is None and not o.min_host_gib >= DEFAULT_MIN_HOST_GIB:     # can be raised, not lowered; written so that nan is refused too
        problem = f'--min-host-gib {o.min_host_gib:g}: the floor is at least {DEFAULT_MIN_HOST_GIB:g} GiB; nothing read, nothing moved'
    if problem:                                          # before the machine and before the lock file
        print(problem, file=sys.stderr)
        return 2
    sx = sx_from(o)
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
    log(f"start: root {root}; repo {repo}; {len(s2.plan_runs(with_x=sx is not None))} runs in order{'' if sx is None else f', with S_X from {sx.arm}@{sx.checkpoint}'}; /mnt/c floor {o.min_host_gib:g} GiB")
    if (root/'queue_stop.json').exists():                # evidence, kept under its time
        (root/'queue_stop.json').rename(root/f"queue_stop_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    try:
        return queue_body(o, system, root, log, results, repo, sx, ev=ev)
    except Stop as stop:
        log(f'STOP: {stop}')
        core.write_json(root/'queue_stop.json', dict(reason=str(stop), utc=core.utc()))
        return 8


# ---- the steps after a run --------------------------------------------------------------------------------------------------------------------------------------------------------------
def evaluate_one(o, *, ev=None):
    s2.require_cell(o.arm, o.pair)
    ev = load_evaluate() if ev is None else ev
    rec = ev.evaluate_run(Path(o.run), o.arm, o.pair, o.run_id, method=method_name(o.arm))
    trace, _ = s2.entry_trace(o.run)
    rec['seed'] = dict(trace, arm=core.read_json(Path(o.run)/'seed_arm.json'))
    rec['s2'] = core.read_json(Path(o.run)/'s2_arm.json')
    core.write_json(Path(o.run)/'metrics.json', rec)
    x, c = rec.get('metrics') or {}, rec.get('certificate') or {}
    print(f"{o.arm} {s2.short_name(o.pair)}: seg35 {x.get('seg35')} seg4 {x.get('seg4')} SDlogJ {x.get('sdlogj_official')} cert {c.get('status')} passed {c.get('passed')} alpha {trace.get('accepted_alpha')} entry check {trace.get('passed')}")
    return 0


def curve_one(o, *, ev=None):
    """curve.json; deletes nothing (finish_run cleans up afterwards)."""
    s2.require_cell(o.arm, o.pair)
    ev = load_evaluate() if ev is None else ev
    try:
        record = curve.curve_of_run(o.run, o.arm, o.pair, o.run_id, ev, clock=time.perf_counter)
    except (curve.CurveError, OSError) as error:         # OSError: a run that never finished an outer step may have no commits.json at all
        print(f'{o.arm} {s2.short_name(o.pair)}: the curve stops: {error}', flush=True)
        return 8
    core.write_json(Path(o.run)/'curve.json', record)
    print(f"{o.arm} {s2.short_name(o.pair)}: curve of {record['n_fields_evaluated']} fields, {sum(record['evaluation_seconds']):.1f} s")
    return 0


# ---- the dry run ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def command_difference(arm, pair, results=None):
    """This batch's two command lines against S_P's cmd.txt on this pair, with the run directory, the root, the unit and the stage put in place: the tokens that still differ."""
    root = s2.default_root(results)
    run, old = s2.run_dir(root, arm, pair), s2.reference_dir(pair, results)
    old_root = old.parents[2]
    old_prepare, old_unit = tc.command_tokens((old/'cmd.txt').read_text())
    old_unit_name = (old/'service_unit.txt').read_text().strip()
    swap = [(str(old), str(run)), (str(old_root), str(root)), (old_unit_name, s2.unit_name(arm, pair))]
    new_prepare, new_unit = [str(t) for t in prepare_argv(arm, pair, run, results)], [str(t) for t in unit_argv(arm, pair, run, root)]
    differ = {}
    for name, new, was in (('prepare', new_prepare, tc.normalized(old_prepare, swap)), ('unit', new_unit, tc.normalized(old_unit, swap))):
        if new != was:
            differ[name] = [(a, b) for a, b in zip(new, was) if a != b]+([f'lengths {len(new)} {len(was)}'] if len(new) != len(was) else [])
    return differ


ENTRY_SAME_KEYS = ('pair', 'oracle', 'shape_zyx', 'source_k')


def entry_difference(arm, entry, pair, results=None):
    """This cell's entry record against S_P's seed_entry.json on this pair: the keys that may differ are the injection's own (all the rest of the record); the hold only for S_X."""
    old = core.read_json(s2.reference_dir(pair, results)/'seed_entry.json') or {}
    same = {key: (old.get(key), entry.get(key)) for key in ENTRY_SAME_KEYS if arm != 'S_X' or key != 'source_k'}
    bad = {key: pair_of for key, pair_of in same.items() if pair_of[0] != pair_of[1]}
    return dict(differs_in=sorted(k for k in set(old) | set(entry) if old.get(k) != entry.get(k)), hold=(old.get('charge_seconds'), entry.get('charge_seconds')), must_be_equal_but_differ=bad)


def disk_estimate(plan, results=None):
    """From S_P's cleanup.json on the same pairs: the intermediate files per run; raw' and what stays are added by the dry run."""
    rows = {}
    for pair in dict.fromkeys(p for _, p in plan):
        record = core.read_json(s2.reference_dir(pair, results)/'cleanup.json') or {}
        rows[pair] = record.get('deleted_bytes', 0)/2**30
    return rows


def rehearse_reading(scratch, out, results=None):
    """The reading on a made-up batch: S_F and S_Pp are copies of S_P's own records (seed_v1, seed_v1b; TC's curves), with the entry's record made up, so every Delta at the final map is 0 and the code must say 过了原生层看不到.  Formats only; only codes and the file names are printed; deleted with the scratch directory."""
    results = core.RESULTS if results is None else results
    root = scratch/s2.ROOT_NAME
    for arm, pair in s2.plan_runs():
        final, again = decide_s2.reference_paths(pair, results, s2.PAIRS)
        target = s2.run_dir(root, arm, pair)
        target.mkdir(parents=True)
        doc = core.read_json(final/'metrics.json')
        hash_ = core.read_json(s2.repair_dir(arm, pair, results)/'repair_metrics.json')['repaired_u_sha256']
        doc['seed'] = dict(doc.get('seed') or {}, passed=True, problems=[], accepted_alpha=1., transfer_attempts=[dict(alpha=1., passed=True, state_hash=hash_)], repaired_u_sha256=hash_)
        core.write_json(target/'metrics.json', doc)
        if (again/'curve.json').is_file():
            shutil.copyfile(again/'curve.json', target/'curve.json')
        shutil.copyfile(final/'protocol.json', target/'protocol.json')
        task = curve.task_dir(target)
        task.mkdir(parents=True)
        for name in ('summary.json', 'history.json'):
            shutil.copyfile(curve.task_dir(final)/name, task/name)
        core.write_json(target/'seed_entry.json', dict(repair_dir=str(s2.repair_dir(arm, pair, results)), charge_seconds=s2.HOLD_S_P))
    result = decide_s2.write_analysis(root, s2.plan_runs(with_x=True), s2.PAIRS, s2.S_PP_PAIRS, s2.short_name, ev=load_evaluate(), utc=core.utc, results=results)
    description = result['description']
    retention = {row['key']: row['delta_entry'] for row in description['retention']}
    accepted = {row['key']: row['delta_entry_accepted'] for row in description['retention'] if row['delta_entry_accepted']}
    gains = [(row['pair'], row['seg35_entry_gain'], row['seg4_entry_gain']) for row in description['s_pp']['rows']]
    out(f"reading on copies of S_P's own final maps: code {result['reading_code']} (Δ seg35 {result['delta_seg35']['mean']}, Δ seg4 {result['delta_seg4']['mean']}); same-t rows {len(description['same_t'])}; same-clock rows {len(description['same_clock'])}; files {sorted(p.name for p in (root/'analysis').iterdir())}")
    out(f"real numbers on the way (the entries' own records, not results of this batch): Δ at the entry S_F against S_P, alpha = 1, n {retention['seg35']['n']}: {retention['seg35']['mean']:+.4f} / {retention['seg4']['mean']:+.4f} (runs/20261104-ef/result.md: +.0152 / +.0235), against S_P's accepted alpha {accepted['seg35']['mean']:+.4f} / {accepted['seg4']['mean']:+.4f} (+.0184 / +.0268); S_Pp's entry against S_P's accepted alpha: "+'; '.join(f'{name} {a:+.4f} / {b:+.4f}' for name, a, b in gains)+' (the brief: +.0161 / +.0165, +.0186 / +.0212)')
    if result['reading_code'] != decide_s2.NOT_VISIBLE:
        raise SystemExit(f"the rehearsal of the reading did not give what it must: {result['reading']}")
    pinned = (retention['seg35']['mean'], retention['seg4']['mean'], accepted['seg35']['mean'], accepted['seg4']['mean'], *[value for row in gains for value in row[1:]])
    if any(abs(have-want) > 5.1e-5 for have, want in zip(pinned, ENTRY_PINS)):
        raise SystemExit(f'the entry differences read here {pinned} are not the ones the plan names {ENTRY_PINS}')


def dry_run(scratch=None, out=print, results=None, repo=None, sx=None):
    """The scratch directory goes, however the dry run ends."""
    results = core.RESULTS if results is None else results
    repo = core.MAIN_REPO if repo is None else repo
    made = scratch is None
    scratch = Path(tempfile.mkdtemp(prefix='s2_dry_')) if made else Path(scratch)
    if not made and scratch.exists():
        raise SystemExit(f'{scratch} exists (nothing is overwritten)')
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        return dry_run_steps(scratch, out, results, repo, sx)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def dry_run_steps(scratch, out, results, repo, sx):
    root = s2.default_root(results)
    link = core.frozen_dir(results)/'data/three_d/stage_L2R_val_v1'
    link_before = os.readlink(link)
    existed = root.exists()
    plan = s2.plan_runs(with_x=True)
    out(f'S2 dry run ({core.utc()}): no solver, no GPU (CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")!r}), no unit; everything written under a scratch directory, deleted at the end.')
    out(f'{len(s2.plan_runs())} runs in the queue: S_Pp on 2 pairs, S_F on the first 5; S_X on the first 5 only with its options ({"given" if sx else "not given: its entries are not checked here, its commands are"}); --seconds {core.SECONDS}; hold {s2.HOLD_S_P} s; watchdog {core.WATCHDOG} s; root {root}; /mnt/c floor {DEFAULT_MIN_HOST_GIB:g} GiB, free now {shutil.disk_usage("/mnt/c").free/2**30:.0f} GiB; /home free now {shutil.disk_usage("/home").free/2**30:.0f} GiB; times {curve.TIMES} and the final map')
    problems = []
    selection = json.loads((core.MAIN_REPO/'data/three_d/stage_L2R_val_v1/selection.json').read_text())
    names = [row['pair'] for row in selection['pairs']]
    out(f"selection.json first 5 == the plan's pairs: {tuple(names[:5]) == s2.PAIRS}; 0455-0456 in selection.json, not in the first 5: {s2.EXCEPTION_PAIR in names[5:]}")
    if tuple(names[:5]) != s2.PAIRS or s2.EXCEPTION_PAIR not in names[5:]:
        problems.append('selection.json differs from the plan')
    sys.path.insert(0, str(ROOT))
    from bayesreg.io.guard import check_pair_subjects, load_dev_allowlist
    allowlist = load_dev_allowlist(ROOT/'configs/bayesreg/dev_subjects.json')

    out('\n==== the entries: what the frozen path will be given (raw prime on the scratch disk, its file and bytes hashes, the wrapper\'s own load_field; the certificates are taper_check.out) ====')
    copy = scratch/'frozen_copy'
    n_files = core.frozen_copy(copy, results)
    out(f'frozen source copied: {n_files} files equal to the freeze hashes; the worktree link stays')
    cells = {}
    entry_scratch = scratch/'entries'
    for arm, pair in plan:
        files = s2.input_files(pair, results)
        check_pair_subjects(files[3]['fixed_subject'], files[3]['moving_subject'], allowlist)
        if arm == 'S_X' and sx is None:
            cells[(arm, pair)] = (None, files)
            out(f"S_X {s2.short_name(pair)}: its entry is pinned before the start, from FL's best arm: not checked here; subjects on the allowlist")
            continue
        try:
            entry = s2.write_entry(entry_scratch, arm, pair, files[3], results, sx)
        except EntryError as error:
            problems.append(str(error))
            out(f'{arm} {s2.short_name(pair)}: REFUSED: {error}')
            continue
        differ = entry_difference(arm, entry, pair, results)
        if differ['must_be_equal_but_differ']:
            problems.append(f"{arm} {pair}: entry record {differ['must_be_equal_but_differ']}")
        if arm != 'S_X' and differ['hold'] != (s2.HOLD_S_P, s2.HOLD_S_P):
            problems.append(f"{arm} {pair}: the hold {differ['hold']}")
        c = entry['construction']
        out(f"{arm} {s2.short_name(pair)}: raw {entry['source_raw_field']} + {entry['repair_npz']}; nodes {c['n_repair_nodes']}, moved {c['n_changed_nodes']}, in the band {c['n_nodes_in_band']}, nearest a face {c['min_face_distance']}; taper(raw') {entry['tapered_sha256'][:16]} = the repaired field {entry['repaired_u_sha256'][:16]}: {entry['tapered_sha256'] == entry['repaired_u_sha256']}; hold {entry['charge_seconds']} s; subjects on the allowlist; "
            f"entry record against S_P's seed_entry.json differs in {differ['differs_in']}; hold {differ['hold']}; must be equal: {differ['must_be_equal_but_differ'] or 'equal'}")
        cells[(arm, pair)] = (entry, files)
        s2.entry_paths(entry_scratch, arm, pair)[0].unlink()             # 158 MiB each: only the record is needed from here on

    out('\n==== the input record, colour table and registry checks (main checkout) ====')
    system = System()
    bad = system.inputs_problems('S_P', repo, results)
    out(f"{core.inputs_root('S_P', results)}: {'ok' if not bad else bad}")
    problems += bad
    bad = reference_problems(plan, results)
    out(f"S_P's records (seed_v1, seed_v1b, tc_v1): {'there' if not bad else bad}")
    problems += bad

    out(f'\n==== all {len(plan)} possible runs: the real parser and prepare of the frozen source (scratch copy), check_prepared, the protocol against S_P of the same pair; the two command lines against its cmd.txt; then the unit that the queue would start ====')
    prepared = scratch/'prepared'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    scratch_link = copy/'data/three_d/stage_L2R_val_v1'
    scratch_link.parent.mkdir(parents=True, exist_ok=True)
    for k, (arm, pair) in enumerate(plan, 1):
        entry, files = cells.get((arm, pair), (None, None))
        if files is None:
            continue
        shown = s2.run_dir(root, arm, pair)
        out(f"# item {k}: arm {arm}, pair {pair}, inputs {files[0]}, watchdog {core.WATCHDOG} s, unit {s2.unit_name(arm, pair)}, run id s2_{arm}_{s2.short_name(pair)}_<UTC>, compared with {s2.reference_dir(pair, results)}")
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
        done = subprocess.run(tokens, cwd=scratch, env=env, capture_output=True, text=True)
        if done.returncode != 0:
            raise SystemExit(f'prepare refused {arm} {pair}: {done.stdout[-1500:]} {done.stderr[-1500:]}')
        check = system.check_prepared(here, pair, s2.stage_name(arm), files[1], files[2], config_entry(results))
        if not check['ok']:
            raise SystemExit(f"check_prepared refused {arm} {pair}: {check['problems']}")
        differs = core.protocol_differences(core.read_json(here/'protocol.json'), core.read_json(s2.reference_dir(pair, results)/'protocol.json'))
        commands = command_difference(arm, pair, results)
        out(f'parser, prepare and check_prepared ok ({time.perf_counter()-tick:.1f} s); protocol against S_P differs in: {json.dumps(differs, ensure_ascii=False)}; the command lines differ in: {json.dumps(commands, ensure_ascii=False)}')
        if set(differs) != {'stage'} or differs['stage'] != ('SEED-S_P', s2.stage_name(arm)):
            problems.append(f'{arm} {pair}: the protocol differs from S_P in {sorted(differs)}')
        if set(commands) != {'prepare'} or commands['prepare'] != [(s2.stage_name(arm), 'SEED-S_P')]:
            problems.append(f'{arm} {pair}: the command lines differ from S_P: {commands}')
        shutil.rmtree(here)

    out("\n==== the reading (made-up batch from S_P's own final maps: formats only, not results) ====")
    rehearse_reading(scratch, out, results)

    out('\n==== the disk ====')
    rows = disk_estimate(plan, results)
    out(f"intermediate commits per run, from S_P's cleanup.json on the same pairs: {min(rows.values()):.2f}-{max(rows.values()):.2f} GiB (serial, each deleted after its curve, before the next starts); stays per run: as TC, about 262 MiB; raw' under {root}/entries: 158 MiB per cell, kept: 7 cells 1.1 GiB, with S_X 12 cells 1.9 GiB")
    out(f"evaluation: {MAX_FIELDS} fields at most a run; TC's runs took about 7.5 minutes each with these steps: S_Pp about 15 minutes, S_F about 38 minutes, S_X about 38 minutes")

    out('\n==== the start strings (the lead) ====')
    out(shlex.join(queue_launch(root, repo)))
    out('the pins file of S_X, after FL, from its records (read only, on stdout): '+shlex.join([core.PY, '-B', 'experiments/s2/run_s2.py', 'sx-pins', '--sx-root', str(Path(results)/s2.FL_ROOT), '--sx-arm', '<arm>', '--sx-checkpoint', '2400']))
    out('with S_X (the five options together; plan.md revision 1):')
    example = s2.SXOptions(str(Path(results)/s2.FL_ROOT), '<arm>', 2400, '<pins.json>', '<its sha256>')
    out(shlex.join(queue_launch(root, repo, sx=example if sx is None else sx)))
    out('the taper check, before the queue or by hand: '+shlex.join([core.PY, '-B', 'experiments/s2/run_s2.py', 'taper-check', *sx_argv(sx)]))

    if os.readlink(link) != link_before:
        problems.append("the frozen worktree's link changed")
    if root.exists() != existed:
        problems.append(f'{root} was created')
    out(f"\nthe frozen worktree's link unchanged: {link_before}; {root} {'exists' if existed else 'not created'}; no solver run; the scratch directory goes when this ends")
    out(f"torch imported by this dry run: {'torch' in sys.modules}; problems: {problems or 'none'}")
    return 8 if problems else 0


def add_sx(parser):
    parser.add_argument('--sx-root', default=None)
    parser.add_argument('--sx-arm', default=None)
    parser.add_argument('--sx-checkpoint', type=int, default=None)
    parser.add_argument('--sx-pins', default=None)
    parser.add_argument('--sx-pins-sha256', default=None)


def sx_from(o):
    return s2.sx_options(getattr(o, 'sx_root', None), getattr(o, 'sx_arm', None), getattr(o, 'sx_checkpoint', None), getattr(o, 'sx_pins', None), getattr(o, 'sx_pins_sha256', None))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['--dry-run']:
        d = argparse.ArgumentParser()
        d.add_argument('--dry-run', action='store_true')
        d.add_argument('--scratch', default=None)
        add_sx(d)
        parsed = d.parse_args(argv)
        return dry_run(scratch=parsed.scratch, sx=sx_from(parsed))
    p = argparse.ArgumentParser(description='S2: see the module docstring')
    sub = p.add_subparsers(dest='command', required=True)
    pins = sub.add_parser('sx-pins')
    pins.add_argument('--sx-root', required=True)
    pins.add_argument('--sx-arm', required=True)
    pins.add_argument('--sx-checkpoint', type=int, required=True)
    t = sub.add_parser('taper-check')
    t.add_argument('--arms', nargs='*', default=None)
    t.add_argument('--json', default=None)
    add_sx(t)
    q = sub.add_parser('queue')
    q.add_argument('--root', default=str(s2.default_root()))
    q.add_argument('--wait-minutes', type=int, default=30)
    q.add_argument('--min-host-gib', type=float, default=DEFAULT_MIN_HOST_GIB)
    add_sx(q)
    sub.add_parser('decide').add_argument('--root', default=str(s2.default_root()))
    for name in ('evaluate-one', 'curve-one'):
        e = sub.add_parser(name)
        e.add_argument('run')
        e.add_argument('--arm', required=True)
        e.add_argument('--pair', required=True)
        e.add_argument('--run-id', required=True)
    o = p.parse_args(argv)
    if o.command == 'sx-pins':
        return print_pins(o)
    if o.command == 'taper-check':
        return run_taper_check(o)
    if o.command == 'queue':
        return queue(o)
    if o.command == 'decide':
        problem = s2.root_problem(o.root)
        if problem:
            print(problem, file=sys.stderr)
            return 2
        result = write_analysis(o.root)
        if result['decide_code']['same'] is False:
            print(f"decide_s2.py has changed since the queue's first start: {result['decide_code']}", file=sys.stderr)
        return 0
    s2.require_cell(o.arm, o.pair)                       # before any file
    problem = s2.own_run_problem(o.run, o.arm, o.pair)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    return evaluate_one(o) if o.command == 'evaluate-one' else curve_one(o)


if __name__ == '__main__':
    sys.exit(main())
