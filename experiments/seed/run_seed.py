"""SEED (runs/20261104-seed/任务书.md, plan.md; docs/bayesreg/05_D0协议.md §16): the frozen native level from a K = 35 entry field.  The solver is configuration C with B1TM's options, the images are the scalar colourings;
only the entry changes, and it is paid out of the same 330 s.  Not F_B, not a method yet: a path experiment on the first 5 development pairs.

    python -B experiments/seed/run_seed.py --dry-run [--scratch DIR] [--skip-entries S_P]
        prints all 10 runs, their command lines, the real frozen parser and prepare on each (a scratch copy of the frozen source), the protocol against the control's, the entry checks (S_P: pending until MC0P
        has run), a rehearsal of the reading on the control's numbers.  No solver, no GPU, no unit, nothing written outside the scratch directory.
    python -B experiments/seed/run_seed.py queue [--root DIR] [--wait-minutes 30]
        the lead's start, in the unit printed by the dry run.  Start checks first (frozen worktree, inputs records, every entry, the controls): a missing file stops before any run.  Then S_P on 5 pairs and S_T on 5 pairs,
        serial, each behind the gate: no $ROOT/hold, no other tag-* or seq-vp-* unit, 1-minute load < 1.5, no GPU compute process, GPU utilization <= 20%, GPU free >= 14 GiB, /home free >= 100 GiB, /mnt/c free >= 26 GiB; closed
        for --wait-minutes: STOP.  60 s between runs.  One unit tag-seed-<arm>-<ffff>-<mmmm> per run (MemoryMax 12G, as the B1 runs).  A run directory is never rerun; failures are kept.  STOP on: a prepare or check failure, a class
        `unsure`, an evaluation failure, an entry not injected as recorded (whatever the class), a failed reading, 3 infrastructure failures in a row.  At the end the reading (decide_seed.py); nothing else starts.
    python -B experiments/seed/run_seed.py decide [--root DIR]       analysis/decision_seed.json, long.csv, summary_seed.md
    python -B experiments/seed/run_seed.py evaluate-one RUN --arm ARM --pair PAIR --run-id ID     the queue's step: experiments/d0/evaluate.py evaluate_run, plus the record of the entry

Exit codes: 0 done, 2 usage or another driver on the root, 8 STOP or a failed check.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT/'experiments/d0'))

import numpy as np  # noqa: E402

import b1_spec  # noqa: E402    # read only, experiments/d0
import inject_run  # noqa: E402

PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'
SMI = '/usr/lib/wsl/lib/nvidia-smi'
MAIN_REPO = Path('/home/rmcfz/code/python_projects/seq_vp_tag')
RESULTS = Path('/home/rmcfz/seq_vp_tag_results')
FROZEN_NAME = 'frozen_eb808be'
FROZEN_COMMIT = 'eb808be401fde7ed6fee73c2d86212c8c463a513'
DEFAULT_ROOT = RESULTS/'seed_v1'
QUEUE_UNIT = 'tag-seed-queue'
SEED19_ROOT_NAME = 'seed_v1b'                          # SEED19's root (run_seed19.py): this entry's queue and decide refuse it
PAIRS = tuple(f'OASIS_OAS1_{n:04d}_MR1__OASIS_OAS1_{n+1:04d}_MR1' for n in range(438, 443))    # the first 5 of selection.json; nothing else runs

# ---- the configuration ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# B1TM's: configuration C (experiments/d0/run_queue.sh OPTS, PRE) with --geometry-weight-levels 1,1,0.1,0.01 and --seconds 330.
SECONDS = 330
GEOMETRY = [1., 1., .1, .01]
CHANGES = dict(geometry_weight_levels=GEOMETRY)
# The brief asks for --seconds 285 so that entry + run take as long as B1TM.  The frozen tree has no such budget: F/scripts/run_stage_3d_e48_pg.py:24 (BUDGETS 30, 45, 60, 90, 120, 180, 240, 330, 600, 3600), :405-406 (prepare refuses),
# :925 (argparse choices).  So the budget stays 330 s and the entry is paid inside the task clock: the injected function holds it for ENTRY_CHARGE_SECONDS.  Native cutoff 329 - 47 = 282 s, against 284 s for a 285 s budget; total 330 s, as B1TM.
# 47 s: mc0_v1 P2 long, seconds_total 46.35, 46.47, 46.85, 46.92, 47.92 (idle GPU, median 46.85), rounded up; Q2 is the same optimizer and K, and MC0P runs beside the training, so its own seconds are not the cost.
ENTRY_CHARGE_SECONDS = 47.
WATCHDOG = b1_spec.watchdog_seconds(SECONDS)
PRE = 'fp32-v1:iterations=50/50/100/60,lr=0.5,cap=0.5,sigma_grad=1,sigma_field=1,mu=1/1/0.01/0.003,alpha=1e-3/1e-3/1e-3/0,kappa_pre=10,j0=0.2,seconds=60,taper=4'
OPTS = ('--strength', 'weak', '--lean-commits', '--resolution-exit',
        '--pg-metric-policy-levels', 'legacy_q1_jd_affine_always_v1,legacy_q1_jd_affine_always_v1,image_shift_ggn_equilibrated_jd_affine_v3,image_shift_ggn_equilibrated_jd_affine_v3',
        '--level-shares', '0.020833,0.0625,0.145833,1', '--geometry-weight-levels', '1,1,0.1,0.01', '--curvature-weight-levels', '1e-3,1e-3,1e-3,0',
        '--throughput', 'v41', '--export-reserve', '1', '--seconds', str(SECONDS), '--pre-optimizer', PRE)
TAPER = 4

ARMS = dict(S_P=dict(inputs='d0_v1_a5/inputs/A5P', input_arm='A5P', source_arm='Q2', source_prefix='mc0p_Q2_long_', source_schema='bayesreg.mc0p.run.v1', oracle=False, control='B1TM', pending='MC0P'),
            S_T=dict(inputs='d0_v1_a5/inputs/A5', input_arm='A5', entry_root='mc0_v1', source_arm='P2', source_prefix='mc0_P2_long_', source_schema='bayesreg.mc0.run.v1', oracle=True, control='B1W_A5', pending='MC0'))
ENTRY_TIER, ENTRY_K = 'long', 35
# S_P's entries, fixed pair by pair, never "whichever exists" (plan revision 2): in MC0P's first batch Q2 long 0441-0442 ran out of device memory and 0442-0443 was skipped when the arm stopped; a retry writes these two under mc0p_v1b, with `retry_of`.
# mc0p_v1b also holds a control cell, Q2 long 0438-0439: it is a record of the retry, not an entry.
RETRIED_ROOT, RETRY_ROOT = 'mc0p_v1', 'mc0p_v1b'
S_P_ROOTS = {'0438-0439': RETRIED_ROOT, '0439-0440': RETRIED_ROOT, '0440-0441': RETRIED_ROOT, '0441-0442': RETRY_ROOT, '0442-0443': RETRY_ROOT}
LONG_TIER = dict(iterations=[50, 50, 100, 600], lr=.5, cap=.5, sigma_grad=1., sigma_field=1., mu=[1., 1., .01, .003], alpha=[1e-3, 1e-3, 1e-3, 0.], kappa_pre=10., j0=.2)     # experiments/mc0/mc0_core.py PRE_C, TIERS['long']
PREDICTION_SEGMENTER, PREDICTION_POLICY = 'unet35_v1_ens', 'all_folds_mean'        # experiments/mc0/mc0p_core.py SEGMENTER_ID, POLICY (a test compares them)
PAIR_IMAGE_KEYS = ('fixed', 'moving', 'initial_u')                                 # experiments/mc0/mc0p_core.py IMAGE_KEYS

# ---- the gate ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
LOAD_LIMIT, HOME_FLOOR_GIB = 1.5, 100
STORAGE_RESERVE_GIB, PENDING_GIB = 10, 16      # F's guard: /mnt/c >= 10 GiB + pending (F/scripts/run_stage_3d_e19_mri.py:460, 546); pending: one run's commits, 14.8 GiB in B1TM 0438-0439 (its cleanup.json)
GPU_FREE_MIB, GPU_UTIL_LIMIT = 14*1024, 20
POLL_SECONDS, BETWEEN_RUNS_SECONDS = 30, 60
MAX_CONSECUTIVE_INFRA = 3


class Stop(Exception):
    """The queue stops here; everything so far stays."""


class EntryError(Exception):
    """This entry field may not be used."""


class EntryMissing(EntryError):
    """The entry is not there (yet)."""


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False, default=float)+'\n')
    tmp.replace(path)


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


sha_file = inject_run.sha_file


def short_name(pair):
    return '-'.join(side.split('_')[2] for side in pair.split('__'))


def require_pair(pair):
    """Only the first 5 development pairs: checked before any file of the pair is opened."""
    if pair not in PAIRS:
        raise SystemExit(f'SEED runs only the first 5 development pairs {[short_name(p) for p in PAIRS]}; refused: {pair}')
    return pair


def require_arm(arm):
    if arm not in ARMS:
        raise SystemExit(f'SEED has no arm {arm}; only {tuple(ARMS)}')
    return arm


def plan_runs():
    """S_P on the 5 pairs, then S_T on the 5 pairs (the brief's order): (arm, pair)."""
    return [(arm, pair) for arm in ARMS for pair in PAIRS]


def run_dir(root, arm, pair):
    return Path(root)/'runs'/require_arm(arm)/require_pair(pair)


def unit_name(arm, pair):
    return f'tag-seed-{arm}-{short_name(pair)}'


def new_run_id(arm, pair):
    return f"seed_{arm}_{short_name(pair)}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def inputs_root(arm, results=None):
    return Path(RESULTS if results is None else results)/ARMS[arm]['inputs']


def b1_entry(arm, results=None):
    """The arm in b1_spec's form: b1_spec.check_entry refuses a budget the frozen parser does not have."""
    return b1_spec.check_entry(dict(arm=arm, inputs_root=str(inputs_root(arm, results)), seconds=SECONDS, changes=CHANGES))


def method_name(arm):
    return f"frozen_eb808be_seed_{SECONDS}s_gw{'_'.join(f'{x:g}' for x in GEOMETRY)}_entry{ARMS[arm]['source_arm']}long{'_oracle' if ARMS[arm]['oracle'] else ''}"


def check_launch(argv):
    """The start string of a process never holds the word of the test runner (the machine's gates look for it; gate-12 incident)."""
    if 'pytest' in ' '.join(str(a) for a in argv):
        raise ValueError(f'start string with a forbidden word: {argv}')
    return [str(a) for a in argv]


def frozen_dir(results=None):
    return Path(RESULTS if results is None else results)/FROZEN_NAME


def prepare_argv(arm, pair, run, results=None):
    return check_launch([PY, '-B', frozen_dir(results)/'scripts/run_stage_3d_e48_pg.py', 'prepare', run, *OPTS, '--stage', f'SEED-{arm}', '--pair', require_pair(pair)])


def unit_argv(arm, pair, run, root, *, unit=None):
    """run_queue.sh build_sd_cmd, the same resources and watchdog as the B1 330 s runs; the one difference is the script: the run's own copy of inject_run.py, which hands over to the snapshot's frozen entry.
    unit: SEED19's entry (run_seed19.py) passes its own name; the default is SEED's."""
    inner = (f"printf '%s\\n' \"$(date -u +%FT%TZ)\" > '{run}/supervisor_started_utc.txt'; timeout --signal=TERM --kill-after=15s {WATCHDOG}s {PY} -B '{run}/seed_inject.py' '{run}' > '{run}/run.log' 2>&1; ec=$?; "
             f"printf '%s\\n' \"$ec\" > '{run}/exit_code'; printf '%s\\n' \"$(date -u +%FT%TZ)\" > '{run}/supervisor_finished_utc.txt'; exit $ec")
    return check_launch(['systemd-run', '--user', f'--unit={unit_name(arm, pair) if unit is None else unit}', '--wait', '--collect', f'--working-directory={run}/snapshot',
                         '--property=CPUQuota=1200%', '--property=MemoryMax=12G', '--property=MemorySwapMax=4G',
                         '--setenv=OPENBLAS_NUM_THREADS=8', '--setenv=OMP_NUM_THREADS=8', '--setenv=MKL_NUM_THREADS=8', '--setenv=NUMEXPR_NUM_THREADS=8',
                         '--setenv=NUMBA_NUM_THREADS=8', '--setenv=PYTHONDONTWRITEBYTECODE=1',
                         f'--setenv=JD_CURL_JIT_CACHE={root}/cache/numba_jit', f'--setenv=JD_CURL_FIRST_USE_CACHE={root}/cache/first_use',
                         '/bin/bash', '-c', inner])


def queue_launch(root=DEFAULT_ROOT, repo=MAIN_REPO):
    """The lead's start string of the queue."""
    return check_launch(['systemd-run', '--user', f'--unit={QUEUE_UNIT}', '--collect', f'--working-directory={repo}', '--property=MemoryMax=9G', '--setenv=PYTHONDONTWRITEBYTECODE=1',
                         '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1', '--setenv=MKL_NUM_THREADS=1', '--setenv=NUMEXPR_NUM_THREADS=1',
                         PY, '-B', 'experiments/seed/run_seed.py', 'queue', '--root', str(root)])


# ---- the inputs and the entry ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
def input_files(arm, pair, results=None):
    """(pair.npz, manifest.json, their sha256, the manifest) of the arm's inputs, checked against the inputs record and the arm; EntryError otherwise."""
    root = inputs_root(arm, results)
    record = read_json(root/'d0_inputs.json')
    if record is None:
        raise EntryError(f'{root}/d0_inputs.json: no such file')
    d = root/'stage_L2R_val_v1'/require_pair(pair)
    if not (d/'pair.npz').is_file() or not (d/'manifest.json').is_file():
        raise EntryError(f'{d}: no pair.npz or manifest.json')
    pair_sha, manifest_sha = sha_file(d/'pair.npz'), sha_file(d/'manifest.json')
    listed = record.get('files_sha256') or {}
    if listed.get(f'stage_L2R_val_v1/{pair}/pair.npz') != pair_sha or listed.get(f'stage_L2R_val_v1/{pair}/manifest.json') != manifest_sha:
        raise EntryError(f'{d}: the files differ from {root}/d0_inputs.json')
    manifest = json.loads((d/'manifest.json').read_text())
    made = manifest.get('preprocessing_d0') or {}
    if manifest.get('pair') != pair or manifest.get('pair_sha256') != pair_sha or made.get('arm') != ARMS[arm]['input_arm'] or made.get('oracle') is not ARMS[arm]['oracle'] or record.get('oracle') is not ARMS[arm]['oracle']:
        raise EntryError(f"{d}: manifest pair {manifest.get('pair')}, arm {made.get('arm')}, oracle {made.get('oracle')}; expected {pair}, {ARMS[arm]['input_arm']}, {ARMS[arm]['oracle']}")
    return d, pair_sha, manifest_sha, manifest


def entry_root(arm, pair):
    """The batch an entry may come from: S_P_ROOTS for S_P, mc0_v1 for S_T."""
    require_arm(arm)
    require_pair(pair)
    return S_P_ROOTS[short_name(pair)] if arm == 'S_P' else ARMS[arm]['entry_root']


def entry_dir(arm, pair, results=None):
    """The one place an entry field may come from: the long tier of Q2 (mc0p_v1 or mc0p_v1b, by the pair) or P2 (mc0_v1), the same pair."""
    spec = ARMS[require_arm(arm)]
    return Path(RESULTS if results is None else results)/entry_root(arm, pair)/'runs'/f"{spec['source_arm']}_{ENTRY_TIER}"/short_name(require_pair(pair))


def retry_points_at_first_batch(value):
    """`retry_of` of a mc0p_v1b record: a path, a batch name, or a dict holding one, with mc0p_v1 as a path component.  The retry tool is on another branch: the form is taken loosely, the batch it names is not."""
    texts = [value] if isinstance(value, str) else list(value.values()) if isinstance(value, dict) else []
    return any(isinstance(text, str) and RETRIED_ROOT in Path(text).parts for text in texts)


def npz_header(path, member='u'):
    """(members, shape, dtype) reading only the npy header of the member."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if member+'.npy' not in names:
            return names, None, None
        with z.open(member+'.npy') as f:
            version = np.lib.format.read_magic(f)
            read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, _, dtype = read(f)
    return names, tuple(shape), str(dtype)


def channel_sources_problems(run):
    """S_P's entry: the marks MC0P computes from what its channels were read from (experiments/mc0/mc0p_core.py oracle_mark; run_mc0p.py run_one, channel_sources), read again here: not only the run's own `oracle`."""
    sources = run.get('channel_sources')
    if not isinstance(sources, dict):
        return ['run.json has no channel_sources']
    bad = []
    if sources.get('oracle') is not False:
        bad.append(f"channel_sources oracle {sources.get('oracle')}")
    if sources.get('truth_labels_read') is not False:
        bad.append(f"channel_sources truth_labels_read {sources.get('truth_labels_read')}")
    keys = set(sources.get('pair_keys_read') or [])
    if not {'fixed', 'moving'} <= keys or keys-set(PAIR_IMAGE_KEYS):
        bad.append(f'channel_sources pair_keys_read {sorted(keys)}: only the images')
    sides = sources.get('sides') or {}
    if set(sides) != {'fixed', 'moving'} or any(side.get('segmenter_id') != PREDICTION_SEGMENTER or side.get('policy') != PREDICTION_POLICY for side in sides.values()):
        bad.append('channel_sources sides: not both images from '+PREDICTION_SEGMENTER+', '+PREDICTION_POLICY)
    return bad


def boundary_taper_sha256(u, shape, width=TAPER):
    """The sha256 of F/jd_curl/three_d/preopt.py boundary_taper(u, grid, width), the same arithmetic (a test compares the bytes with the frozen function): what the frozen seed_hash must be if the path tapers exactly once."""
    rs = []
    for n in shape:
        k = np.arange(n)
        rs.append(np.minimum(np.minimum(k, n-1-k)/float(width), 1.))
    w = rs[0][:, None, None]*rs[1][None, :, None]*rs[2][None, None, :]
    return hashlib.sha256((w[..., None]*np.asarray(u, dtype=np.float64)).tobytes()).hexdigest()


def check_entry(arm, pair, manifest, results=None):
    """The record for seed_entry.json, or EntryError (EntryMissing if the files are not there).  The path, then the run record: the pair, the source arm and tier, finished, oracle as the arm says, the run's images from the same original pair as the
    arm's inputs; then the file: sha256, one member u, float64, the grid's shape, bytes sha256, finite, zero faces, and the taper against the evaluation's record of the field."""
    spec = ARMS[require_arm(arm)]
    d = entry_dir(arm, pair, results)
    field_path, run_path = d/'field.npz', d/'run.json'
    if not field_path.is_file() or not run_path.is_file():
        if entry_root(arm, pair) == RETRY_ROOT:
            raise EntryMissing(f"{arm} 入口 {short_name(pair)} 待补跑: {field_path} 或 {run_path.name} 不存在（{RETRY_ROOT}；{RETRIED_ROOT} 这一对的记录不算入口）")
        raise EntryMissing(f"{arm} 入口待 {spec['pending']}: {field_path} 或 {run_path.name} 不存在（{spec['source_arm']} 长档还没有跑完，或这一对失败了）")
    run = json.loads(run_path.read_text())
    short = short_name(pair)
    problems = []
    if run.get('pair') != pair:
        problems.append(f"run.json pair {run.get('pair')}")
    if run.get('arm') != spec['source_arm'] or run.get('tier') != ENTRY_TIER or run.get('k') != ENTRY_K:
        problems.append(f"run.json arm {run.get('arm')} tier {run.get('tier')} K {run.get('k')}, expected {spec['source_arm']} {ENTRY_TIER} {ENTRY_K}")
    if run.get('status') != 'done':
        problems.append(f"run.json status {run.get('status')}: only a finished run is an entry")
    if run.get('oracle') is not spec['oracle']:
        problems.append(f"run.json oracle {run.get('oracle')}, the arm {arm} must have {spec['oracle']}")
    if run.get('run_id') != spec['source_prefix']+short:
        problems.append(f"run.json run_id {run.get('run_id')}, expected {spec['source_prefix']+short}")
    if run.get('schema') != spec['source_schema']:
        problems.append(f"run.json schema {run.get('schema')}, expected {spec['source_schema']}")
    config = run.get('config') or {}
    if {k: config.get(k) for k in LONG_TIER} != LONG_TIER:
        problems.append(f"run.json config {config}: not the long tier")
    if not spec['oracle']:                               # S_T's entry is MC0's record: it has no channel_sources, and its truth labels are the point
        problems += channel_sources_problems(run)
        if entry_root(arm, pair) == RETRY_ROOT and not retry_points_at_first_batch(run.get('retry_of')):
            problems.append(f"run.json retry_of {run.get('retry_of')!r}: a record of {RETRY_ROOT} must point at {RETRIED_ROOT}")
        if entry_root(arm, pair) == RETRIED_ROOT and 'retry_of' in run:
            problems.append(f"run.json retry_of {run.get('retry_of')!r}: a retry's record under {RETRIED_ROOT}")
    made = manifest.get('preprocessing_d0') or {}
    data = run.get('data') or {}
    if data.get('pair_sha256') != made.get('source_pair_sha256') or data.get('manifest_sha256') != made.get('source_manifest_sha256'):
        problems.append(f"the field comes from original pair {data.get('pair_sha256')}, the inputs of {arm} from {made.get('source_pair_sha256')}")
    if data.get('fixed_subject') != manifest.get('fixed_subject') or data.get('moving_subject') != manifest.get('moving_subject'):
        problems.append(f"the field's subjects {data.get('fixed_subject')}, {data.get('moving_subject')}; the inputs' {manifest.get('fixed_subject')}, {manifest.get('moving_subject')}")
    field = run.get('field') or {}
    if field.get('file') != 'field.npz' or not field.get('sha256') or not field.get('u_sha256'):
        problems.append(f'run.json has no field record: {field}')
    if problems:
        raise EntryError(f'{d}: '+'; '.join(problems))
    got = sha_file(field_path)
    if got != field['sha256']:
        raise EntryError(f"{field_path}: sha256 {got} != run.json {field['sha256']}")
    shape = tuple(manifest['shape_zyx'])
    names, stored, dtype = npz_header(field_path)
    if names != ['u.npy'] or stored != shape+(3,) or dtype not in ('<f8', 'float64'):
        raise EntryError(f'{field_path}: members {names}, shape {stored}, dtype {dtype}; expected only u, {shape+(3,)}, float64')
    record = dict(arm=arm, pair=pair, field_path=str(field_path), field_sha256=got, u_sha256=field['u_sha256'], source_run_id=run['run_id'], oracle=spec['oracle'], charge_seconds=ENTRY_CHARGE_SECONDS, shape_zyx=list(shape))
    try:
        u = inject_run.load_field(record)                # the wrapper's own check, once more when it starts
    except inject_run.Refused as refused:
        raise EntryError(str(refused)) from None
    tapered = boundary_taper_sha256(u, shape)
    metrics = read_json(d/'metrics.json') or {}
    if metrics.get('tapered_u_sha256') not in (None, tapered):
        raise EntryError(f"{d}: the evaluation tapered this field to {metrics.get('tapered_u_sha256')}, here {tapered}")
    certificate = metrics.get('certificate') or {}
    return dict(record, tapered_sha256=tapered, tapered_recorded_by_evaluation=metrics.get('tapered_u_sha256'), source_seconds_total=run.get('seconds_total'), source_code_sha=run.get('code_sha'),
                source_status=run.get('status'), source_k=run.get('k'), source_tier=run.get('tier'), source_arm=run.get('arm'), source_root=entry_root(arm, pair), retry_of=run.get('retry_of'), role=inject_run.ROLE,
                entry_metrics=dict(seg35=(metrics.get('metrics') or {}).get('seg35'), seg4=(metrics.get('metrics') or {}).get('seg4'), sdlogj_official=(metrics.get('metrics') or {}).get('sdlogj_official'),
                                   accepted_alpha=certificate.get('accepted_alpha')))


def injection_trace(run):
    """What the frozen run recorded about its entry, against seed_entry.json: (record, problems).  No problem means: the event preoptimizer_done carries the injected record; raw_hash is the field's bytes; seed_hash is ONE taper of them."""
    run = Path(run)
    entry = read_json(run/'seed_entry.json') or {}
    protocol = read_json(run/'protocol.json') or {}
    task = run/((protocol.get('task') or {}).get('name') or 'missing')
    levels = read_json(task/'levels.json') or []
    done = next((e for e in levels if e.get('event') == 'preoptimizer_done'), None)
    prepared = next((e for e in levels if e.get('event') == 'level_prepared'), None)
    history = read_json(task/'history.json') or []
    problems = []
    pre = (done or {}).get('preoptimizer') or {}
    if done is None:
        problems.append('no event preoptimizer_done')
    else:
        if pre.get('policy') != inject_run.POLICY:
            problems.append(f"preoptimizer policy {pre.get('policy')}: the injected function did not run")
        if pre.get('raw_hash') != entry.get('u_sha256'):
            problems.append(f"raw_hash {pre.get('raw_hash')} != the field's bytes {entry.get('u_sha256')}")
        if pre.get('seed_hash') != entry.get('tapered_sha256'):
            problems.append(f"seed_hash {pre.get('seed_hash')} != one taper of the field {entry.get('tapered_sha256')}")
        if pre.get('taper_L') != TAPER:
            problems.append(f"taper_L {pre.get('taper_L')}")
    recovery = (prepared or {}).get('transfer_recovery') or {}
    first = history[0] if history else {}
    record = dict(field_path=entry.get('field_path'), field_sha256=entry.get('field_sha256'), u_sha256=entry.get('u_sha256'), source_run_id=entry.get('source_run_id'), oracle=entry.get('oracle'), tapered_sha256=entry.get('tapered_sha256'),
                  raw_hash=pre.get('raw_hash'), seed_hash=pre.get('seed_hash'), taper_L=pre.get('taper_L'), charge_seconds=pre.get('charge_seconds'), injected_seconds=pre.get('seconds'), injected_seconds_total=pre.get('seconds_total'),
                  transfer_status=recovery.get('status'), accepted_alpha=recovery.get('accepted_alpha'),
                  transfer_attempts=[dict(alpha=a.get('alpha'), passed=a.get('certificate_passed'), min_jd_lower_bound=(a.get('certificate') or {}).get('min_jd_lower_bound'), n_infeasible=(a.get('certificate') or {}).get('n_infeasible'),
                                          state_hash=(a.get('certificate') or {}).get('state_hash')) for a in recovery.get('attempts', [])],
                  entry_D_at_accepted_alpha=first.get('D_start'), entry_F_at_accepted_alpha=first.get('F_start'), last_D=(history[-1].get('D_candidate') if history else None), entry_metrics=entry.get('entry_metrics'),
                  problems=problems, passed=not problems)
    return record, problems


# ---- the machine -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class System:
    """What the queue reads from and does to the machine (tests replace it)."""

    def load1(self):
        return float(Path('/proc/loadavg').read_text().split()[0])

    def units(self):
        done = subprocess.run(['systemctl', '--user', 'list-units', '--state=active,activating', '--no-legend', '--plain', 'tag-*', 'seq-vp-*'], capture_output=True, text=True)
        if done.returncode:
            return [f'systemctl failed with exit {done.returncode}']     # the gate stays closed: an error is not an empty machine
        return [line.split()[0] for line in done.stdout.splitlines() if line.strip()]

    def gpu_free_mib(self):
        out = subprocess.run([SMI, '--query-gpu=memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
        try:
            free = float(out.split()[0])
        except (IndexError, ValueError):                # empty, "[N/A]", an error text: no reading; the gate stays closed
            return 0.
        return free if math.isfinite(free) else 0.

    def gpu_busy(self):
        """Compute processes and utilization above the limit, as run_queue.sh wait_turn (GPU jobs that are not systemd units); None when free."""
        apps = subprocess.run([SMI, '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True).stdout.split()
        apps = [a for a in apps if a.isdigit()]
        out = subprocess.run([SMI, '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.split()
        util = float(out[0]) if out and out[0].replace('.', '', 1).isdigit() else 0.
        return f'compute pids {apps}, utilization {util}%' if apps or util > GPU_UTIL_LIMIT else None

    def free_gib(self, path):
        try:
            return shutil.disk_usage(path).free/2**30
        except OSError:                                  # no reading: the gate stays closed, as for the GPU
            return 0.

    def now(self):
        return time.monotonic()

    def sleep(self, seconds):
        time.sleep(seconds)

    def frozen_problems(self, results=None):
        """The start checks of run_queue.sh on the frozen worktree: the commit, clean, the 235 sources equal to the freeze."""
        w = frozen_dir(results)
        head = subprocess.run(['git', '-C', str(w), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        status = subprocess.run(['git', '-C', str(w), 'status', '--porcelain'], capture_output=True, text=True).stdout.strip()
        reference = json.loads((ROOT/'experiments/d0/configC_reference.json').read_text())['freeze_source_py_sha256']
        live = {str(p.relative_to(w)): sha_file(p) for d in ('jd_curl', 'scripts') for p in (w/d).rglob('*.py') if '__pycache__' not in p.parts}
        return [text for text, bad in ((f'frozen worktree HEAD {head}, expected {FROZEN_COMMIT}', head != FROZEN_COMMIT), (f'frozen worktree not clean: {status[:200]}', bool(status)),
                                       ('frozen worktree source differs from the freeze', live != reference)) if bad]

    def inputs_problems(self, arm, repo, results=None):
        """b1_spec.check_one_input: the colour table and, unless oracle inputs, the registry: committed, clean, the sha256 of the record."""
        return b1_spec.check_one_input(b1_entry(arm, results), repo)

    def prepare(self, argv, target, log, results=None):
        """The worktree link at the arm's stage during `prepare` (flock, one lock with run_queue.sh), restored right after.  Returns the exit code."""
        results = RESULTS if results is None else results
        w = frozen_dir(results)
        link = w/'data/three_d/stage_L2R_val_v1'
        real = (MAIN_REPO/'data/three_d/stage_L2R_val_v1').resolve()

        def point(at):
            tmp = link.with_name(link.name+'.tmp')
            if tmp.is_symlink() or tmp.exists():
                tmp.unlink()
            tmp.symlink_to(at)
            tmp.replace(link)
        with open(Path(results)/f'.frozen_{FROZEN_COMMIT[:7]}_link.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                point(target)
                if os.readlink(link) != str(target):
                    raise Stop(f'cannot point {link} at {target}')
                with open(log, 'ab') as f:
                    return subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT).returncode
            finally:
                point(real)

    def check_prepared(self, run, pair, stage, pair_sha, manifest_sha, entry):
        spec = importlib.util.spec_from_file_location('_seed_check_prepared', ROOT/'experiments/d0/check_prepared.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.check(Path(run), pair, stage, pair_sha, manifest_sha, b1_entry=entry)

    def launch(self, argv, log):
        with open(log, 'ab') as f:
            return subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT).returncode

    def classify(self, run, t0):
        """experiments/d0/classify_run.py, as run_queue.sh postprocess; returns the record."""
        out = Path(run)/'classification.json'
        if not out.exists():
            kernel = subprocess.run(['dmesg'], capture_output=True, text=True).stdout

            def count(words):
                n = 0
                for line in kernel.splitlines():
                    if any(w in line for w in words) and line.startswith('['):
                        try:
                            n += float(line[1:line.index(']')]) >= float(t0)
                        except ValueError:
                            pass
                return n
            with open(Path(run)/'log.txt', 'ab') as f:
                subprocess.run([PY, '-B', str(ROOT/'experiments/d0/classify_run.py'), str(run), '--watchdog', str(WATCHDOG), '--dxg', str(count(('create_allocation failed',))),
                                '--oom', str(count(('Out of memory', 'oom-kill', 'Killed process'))), '--out', str(out)], stdout=f, stderr=subprocess.STDOUT)
        return read_json(out)

    def evaluate(self, run, arm, pair, run_id):
        with open(Path(run)/'log.txt', 'ab') as f:
            return subprocess.run(check_launch([PY, '-B', HERE/'run_seed.py', 'evaluate-one', run, '--arm', arm, '--pair', pair, '--run-id', run_id]), stdout=f, stderr=subprocess.STDOUT).returncode

    def uptime(self):
        return Path('/proc/uptime').read_text().split()[0]

    def state(self):
        """The machine at a start or an end, as run_queue.sh writes it in log.txt."""
        memory = next((line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable')), '0')
        gpu = subprocess.run([SMI, '--query-gpu=name,clocks.sm,pstate,temperature.gpu,utilization.gpu,memory.used', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
        return f"loadavg {' '.join(Path('/proc/loadavg').read_text().split()[:3])}; MemAvailable {int(memory)//1024} MiB; gpu {gpu}"


def busy_reason(system, root, *, queue_unit=QUEUE_UNIT, host_floor_gib=None):
    """Why no run may start now, or None.  queue_unit, host_floor_gib: SEED19's entry (run_seed19.py) passes its own; the defaults are SEED's gate, /mnt/c >= 26 GiB, unchanged."""
    if (Path(root)/'hold').exists():
        return 'hold file'
    others = [u for u in system.units() if u != f'{queue_unit}.service']
    if others:
        return f'other units: {others}'
    load = system.load1()
    if load >= LOAD_LIMIT:
        return f'1-minute load {load} >= {LOAD_LIMIT}'
    if system.free_gib('/home') < HOME_FLOOR_GIB:
        return f'/home free < {HOME_FLOOR_GIB} GiB'
    if host_floor_gib is None and system.free_gib('/mnt/c') < STORAGE_RESERVE_GIB+PENDING_GIB:
        return f'/mnt/c free < {STORAGE_RESERVE_GIB} GiB + {PENDING_GIB} GiB pending'
    if host_floor_gib is not None and system.free_gib('/mnt/c') < host_floor_gib:
        return f'/mnt/c free < {host_floor_gib} GiB'
    busy = system.gpu_busy()
    if busy:
        return f'GPU busy: {busy}'
    if system.gpu_free_mib() < GPU_FREE_MIB:
        return f'GPU free < {GPU_FREE_MIB} MiB'
    return None


def wait_for_gate(system, root, minutes, log, **gate):
    started = system.now()
    while True:
        reason = busy_reason(system, root, **gate)
        if reason is None:
            return
        if system.now()-started >= minutes*60:
            raise Stop(f'gate closed for {minutes} minutes: {reason}')
        log(f'waiting: {reason}')
        system.sleep(POLL_SECONDS)


# ---- one run ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def write_records(run, arm, pair, unit, run_id, entry, files, env_text, repo, results=None):
    """The records of protocol §8 (git.txt, env.txt, data.txt, seed), for this arm; the entry's own record and the wrapper's copy."""
    run = Path(run)
    d, pair_sha, manifest_sha, manifest = files
    git = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(['git', '-C', str(repo), 'diff', '--stat'], capture_output=True, text=True).stdout
    files_sha = '\n'.join(f'{p.relative_to(ROOT)} {sha_file(p)}' for pattern in ('experiments/seed/*.py', 'experiments/d0/*.py', 'experiments/d0/*.json') for p in sorted(ROOT.glob(pattern)))
    frozen = frozen_dir(results)
    frozen_head = subprocess.run(['git', '-C', str(frozen), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    frozen_status = subprocess.run(['git', '-C', str(frozen), 'status', '--porcelain'], capture_output=True, text=True).stdout
    untracked = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain', '--', 'experiments/seed', 'experiments/d0', 'docs/bayesreg', 'configs/bayesreg', 'runs'], capture_output=True, text=True).stdout
    (run/'git.txt').write_text(f'repo: {repo}\nrepo HEAD: {git}\nrepo git diff --stat:\n{diff}\nrepo untracked or changed (experiments/seed, experiments/d0, docs/bayesreg, configs/bayesreg, runs):\n{untracked}\n'
                               f'frozen worktree: {frozen}\nfrozen HEAD: {frozen_head}\nfrozen commit expected: {FROZEN_COMMIT}\nfrozen git status --porcelain (must be empty):\n{frozen_status}\nfile sha256:\n{files_sha}\n')
    (run/'env.txt').write_text(env_text)
    made = manifest['preprocessing_d0']
    (run/'data.txt').write_text('\n'.join([f'run_id: {run_id}', f'arm: {arm}', f'pair: {pair}', f'unit: {unit}', f"fixed subject: {manifest['fixed_subject']}", f"moving subject: {manifest['moving_subject']}",
                                           f'arm stage directory (prepare source): {d.parent}', f'arm input pair.npz: {d}/pair.npz sha256 {pair_sha}', f'arm input manifest.json: {d}/manifest.json sha256 {manifest_sha}',
                                           f"original pair.npz sha256: {made['source_pair_sha256']}", f"original manifest sha256: {made['source_manifest_sha256']}", f"colouring: {made['recipe']}",
                                           f"oracle (truth labels used for the colouring and the entry): {ARMS[arm]['oracle']}", f"entry field: {entry['field_path']} sha256 {entry['field_sha256']} bytes sha256 {entry['u_sha256']}",
                                           f"entry source run: {entry['source_run_id']}; K {entry['source_k']}; tier {entry['source_tier']}; oracle {entry['oracle']}", 'truth labels for evaluation: the original stage pair.npz (evaluate.py checks its sha256)'])+'\n')
    (run/'seed').write_text(f'{arm} (runs/20261104-seed/plan.md): the inputs of {ARMS[arm]["input_arm"]}, no new colouring; the entry field has no random step (run.json: none); solver: configuration C has no random number generator or seed option; GPU reductions may still differ between runs\n')
    write_json(run/'seed_entry.json', entry)
    write_json(run/'seed_arm.json', dict(arm=arm, seconds=SECONDS, changes=CHANGES, inputs_root=str(inputs_root(arm, results)), entry_root=entry_root(arm, pair), source_arm=ARMS[arm]['source_arm'], charge_seconds=ENTRY_CHARGE_SECONDS,
                                         watchdog_seconds=WATCHDOG, method=method_name(arm), input_oracle=ARMS[arm]['oracle'], control=ARMS[arm]['control'], wrapper_sha256=sha_file(HERE/'inject_run.py')))
    shutil.copyfile(HERE/'inject_run.py', run/'seed_inject.py')


def cleanup_commits(run):
    """Protocol §6: only the intermediate maps committed_*.npz and level_*.npz of the task directory go (classes ok and handover_only); recorded in cleanup.json."""
    run = Path(run)
    task = run/read_json(run/'protocol.json')['task']['name']
    gone, kept = [], []
    for p in sorted(task.glob('*.npz')):
        if p.name.startswith(('committed_', 'level_')) and p.name != 'final_native.npz':
            gone.append(dict(name=p.name, bytes=p.stat().st_size))
            p.unlink()
        else:
            kept.append(dict(name=p.name, bytes=p.stat().st_size))
    write_json(run/'cleanup.json', dict(rule='D0 protocol section 6: only committed_*.npz and level_*.npz of the task directory are deleted (runs classified ok or handover_only); everything else is kept', deleted=gone, deleted_bytes=sum(g['bytes'] for g in gone), kept_npz=kept))


def finish_run(run, arm, pair, run_id, t0, system, log):
    """Classify, evaluate, check the entry, clean up.  Returns the state: ok, handover_only, method_failure, infrastructure.  Stop on anything else."""
    run = Path(run)
    record = system.classify(run, t0)
    cls = (record or {}).get('class')
    with open(run/'log.txt', 'a') as f:
        f.write(f'{utc()} class {cls}\n')
    if cls == 'infrastructure':
        write_json(run/'queue_done.json', dict(run_id=run_id, state=cls, kind=record.get('kind'), reason=record.get('reason'), done_utc=utc()))
        log(f"run {arm} {pair}: infrastructure failure ({record.get('kind')}: {record.get('reason')}); kept whole, not rerun")
        return cls
    if cls not in ('ok', 'handover_only', 'method_failure'):
        raise Stop(f"{run} classified {cls} ({(record or {}).get('reason')}); kept whole, nothing rerun")
    if system.evaluate(run, arm, pair, run_id) != 0:
        raise Stop(f'evaluation failed for {run} (see log.txt)')
    metrics = read_json(run/'metrics.json') or {}
    trace = metrics.get('seed') or {}
    if trace.get('passed') is not True:                  # also for a method failure: a wrapper or entry problem must not pass as a method result and use up the batch (review I1)
        raise Stop(f"{run}: the entry was not injected as recorded: {trace.get('problems')}; class {cls}; kept whole")
    if cls != 'method_failure':
        cleanup_commits(run)
    x, c = metrics.get('metrics') or {}, metrics.get('certificate') or {}
    summary = f"{cls} status {metrics.get('status')} seg35 {x.get('seg35')} seg4 {x.get('seg4')} sdlogj {x.get('sdlogj_official')} cert {c.get('status')}/{c.get('passed')} alpha {trace.get('accepted_alpha')} native {metrics.get('native_commits')} e2e {metrics.get('e2e_seconds')}"
    write_json(run/'queue_done.json', dict(run_id=run_id, state=cls, summary=summary, done_utc=utc()))
    log(f'done {arm} {pair}: {summary}')
    return cls


def one_run(o, system, root, arm, pair, entry, files, log, env_text, results, repo, first):
    """Returns the state, or None if nothing ran because it had already ended."""
    run = run_dir(root, arm, pair)
    if (run/'queue_done.json').is_file():
        return None
    unit = unit_name(arm, pair)
    if run.exists():                                     # never rerun: an ended unit is post-processed, anything else stops
        if (run/'exit_code').is_file():
            log(f'resume: {run} has an exit_code; post-processing')
            return finish_run(run, arm, pair, (run/'run_id').read_text().strip(), (run/'uptime_before').read_text().strip(), system, log)
        raise Stop(f'{run} exists without exit_code or queue_done.json (an incomplete attempt, kept whole)')
    if not first:
        system.sleep(BETWEEN_RUNS_SECONDS)
    wait_for_gate(system, root, o.wait_minutes, log)
    run.parent.mkdir(parents=True, exist_ok=True)
    argv = prepare_argv(arm, pair, run, results)
    target = files[0].parent
    prepare_log = Path(root)/'prepare.log'
    code = system.prepare(argv, target, prepare_log, results)
    if code != 0 or not run.is_dir():
        raise Stop(f'prepare exit {code} for {run} (see {prepare_log})')
    check = system.check_prepared(run, pair, f'SEED-{arm}', files[1], files[2], b1_entry(arm, results))
    write_json(run/'prepared_check.json', check)
    if not check.get('ok'):
        raise Stop(f"prepared run differs from B1TM's configuration or the arm's inputs: {check.get('problems')}")
    run_id = new_run_id(arm, pair)
    (run/'run_id').write_text(run_id+'\n')
    write_records(run, arm, pair, unit, run_id, entry, files, env_text, repo, results)
    sd = unit_argv(arm, pair, run, root)
    (run/'cmd.txt').write_text(f'# prepare (run from any directory; the frozen script resolves its root from its own path)\n{shlex.join(argv)}\n# worktree link during prepare: {frozen_dir(results)}/data/three_d/stage_L2R_val_v1 -> {target}\n# run\n{shlex.join(sd)}\n')
    (run/'service_unit.txt').write_text(unit+'\n')
    t0 = system.uptime()
    (run/'uptime_before').write_text(f'{t0}\n')
    log(f"run {arm} {pair} unit {unit}; load {system.load1()}; /home free {system.free_gib('/home'):.0f} GiB; /mnt/c free {system.free_gib('/mnt/c'):.0f} GiB")
    with open(run/'log.txt', 'a') as f:
        f.write(f'{utc()} launch {unit}; {system.state()}\n')
    code = system.launch(sd, run/'log.txt')
    with open(run/'log.txt', 'a') as f:
        f.write(f'{utc()} unit ended, systemd-run exit {code}; {system.state()}\n')
    return finish_run(run, arm, pair, run_id, t0, system, log)


def start_checks(system, results, repo, log):
    """Everything that can be known before the first run: STOP here rather than in the middle.  Returns {(arm, pair): (entry record, input files)}."""
    problems = list(system.frozen_problems(results))
    prepared = {}
    for arm in ARMS:
        problems += [f'{arm}: {p}' for p in system.inputs_problems(arm, repo, results)]
        for pair in PAIRS:
            try:
                files = input_files(arm, pair, results)
                prepared[(arm, pair)] = (check_entry(arm, pair, files[3], results), files)
                if prepared[(arm, pair)][0]['tapered_recorded_by_evaluation'] is None:
                    log(f"{arm} {short_name(pair)}: the source evaluation has no record of the taper yet; the field's taper is checked against the frozen formula only")
            except EntryError as error:
                problems.append(str(error))
    b1 = Path(results)/'b1_v1'
    if not (b1/'analysis/long.csv').is_file():
        problems.append(f'{b1}/analysis/long.csv: no such file (the controls of the reading)')
    problems += [f"control {ARMS[arm]['control']} {short_name(pair)}: no metrics.json" for arm, pair in plan_runs() if not (b1/'runs'/ARMS[arm]['control']/pair/'metrics.json').is_file()]
    if problems:
        raise Stop('start checks: '+' | '.join(problems))
    log(f'start checks passed: frozen worktree, inputs records, the controls, all {len(prepared)} entries')
    return prepared


def environment_text(repo):
    versions = subprocess.run([PY, '-B', '-c', "import importlib,sys\nprint('python',sys.version.replace(chr(10),' '))\nfor m in ('numpy','scipy','numba','torch','triton','nibabel','surface_distance'):\n    x=importlib.import_module(m);print(m,getattr(x,'__version__','?'))\nimport torch;print('torch.cuda',torch.version.cuda,'cudnn',torch.backends.cudnn.version(),'device',torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"], capture_output=True, text=True).stdout
    smi = subprocess.run([SMI, '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout
    cpu = next((line.split(':', 1)[1].strip() for line in subprocess.run(['lscpu'], capture_output=True, text=True).stdout.splitlines() if line.startswith('Model name')), '?')
    memory = next((line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal')), '0')
    tables = ''.join(f'{name}: {Path(repo)/name} sha256 {sha_file(Path(repo)/name)}\n' for name in ('configs/bayesreg/colour_table_a5_v1.json', 'configs/bayesreg/segmenters.json') if (Path(repo)/name).is_file())
    return f'{versions}nvidia-smi: {smi}cpu: {cpu}; nproc {os.cpu_count()}\nkernel: {os.uname().release}\nmemory: {int(memory)//2**20} GiB total\n{tables}repo: {repo}\njit cache and first-use cache: the --setenv lines of cmd.txt (not ~/.cache/seq_vp_3d)\n'


def trailing_infrastructure(root):
    """Infrastructure failures in a row at the end of what has run, in plan order."""
    n = 0
    for arm, pair in plan_runs():
        done = read_json(run_dir(root, arm, pair)/'queue_done.json')
        if done is None:
            continue
        n = n+1 if done.get('state') == 'infrastructure' else 0
    return n


def queue_body(o, system, root, log, results, repo):
    prepared = start_checks(system, results, repo, log)
    env_text = environment_text(repo) if isinstance(system, System) else 'test\n'
    (root/'cache/numba_jit').mkdir(parents=True, exist_ok=True)
    (root/'cache/first_use').mkdir(parents=True, exist_ok=True)
    if read_json(root/'batch.json') is None:
        write_json(root/'batch.json', dict(schema='bayesreg.seed.batch.v1', started_utc=utc(), plan=[f'{a} {short_name(p)}' for a, p in plan_runs()], seconds=SECONDS, changes=CHANGES, charge_seconds=ENTRY_CHARGE_SECONDS, frozen_commit=FROZEN_COMMIT))
    first = True
    for arm, pair in plan_runs():
        state = one_run(o, system, root, arm, pair, *prepared[(arm, pair)], log, env_text, results, repo, first)
        if state is None:
            continue
        first = False
        if trailing_infrastructure(root) >= MAX_CONSECUTIVE_INFRA:
            raise Stop(f'{MAX_CONSECUTIVE_INFRA} infrastructure failures in a row')
    import decide_seed
    try:
        result = decide_seed.write_analysis(root, results=results)
    except (Exception, SystemExit) as error:             # the runs stay; by hand: run_seed.py decide
        raise Stop(f'the reading failed: {error!r}') from None
    log(f"reading: {result['reading']}. Handed to the lead: no next step starts here")
    return 0


def queue(o, system=None, results=None, repo=None):
    system = System() if system is None else system
    results = RESULTS if results is None else results
    repo = MAIN_REPO if repo is None else repo
    root = Path(o.root)
    if not root.name.startswith('seed'):
        print(f'{root}: the root of this experiment is named seed*', file=sys.stderr)
        return 2
    root.mkdir(parents=True, exist_ok=True)
    lock = open(root/'queue.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f'another driver is on {root}', file=sys.stderr)
        return 2

    def log(text):
        with open(root/'queue.log', 'a') as f:
            f.write(f'{utc()} {text}\n')
    log(f'start: root {root}; repo {repo}; {len(plan_runs())} runs in order')
    if (root/'queue_stop.json').exists():                # evidence, kept under its time; the file of that name always means this start
        (root/'queue_stop.json').rename(root/f"queue_stop_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    try:
        return queue_body(o, system, root, log, results, repo)
    except Stop as stop:
        log(f'STOP: {stop}')
        write_json(root/'queue_stop.json', dict(reason=str(stop), utc=utc()))
        return 8


# ---- evaluation ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def load_evaluate():
    """experiments/d0/evaluate.py, read only: the same metrics and definitions as every earlier arm."""
    name = '_seed_evaluate'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT/'experiments/d0/evaluate.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def evaluate_one(o):
    require_arm(o.arm)
    require_pair(o.pair)
    ev = load_evaluate()
    rec = ev.evaluate_run(Path(o.run), o.arm, o.pair, o.run_id, method=method_name(o.arm))
    trace, _ = injection_trace(o.run)
    rec['seed'] = dict(trace, arm=read_json(Path(o.run)/'seed_arm.json'))
    write_json(Path(o.run)/'metrics.json', rec)
    x, c = rec.get('metrics') or {}, rec.get('certificate') or {}
    print(f"{o.arm} {short_name(o.pair)}: seg35 {x.get('seg35')} seg4 {x.get('seg4')} SDlogJ {x.get('sdlogj_official')} cert {c.get('status')} passed {c.get('passed')} alpha {trace.get('accepted_alpha')} entry check {trace.get('passed')}")
    return 0


# ---- the dry run --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def frozen_copy(target, results=None):
    """A scratch copy of the frozen source, checked against the freeze hashes: it holds its own data link (the worktree's link is not touched)."""
    reference = json.loads((ROOT/'experiments/d0/configC_reference.json').read_text())['freeze_source_py_sha256']
    w = frozen_dir(results)
    for name in ('jd_curl', 'scripts'):
        shutil.copytree(w/name, target/name, ignore=shutil.ignore_patterns('__pycache__'))
    live = {str(p.relative_to(target)): sha_file(p) for d in ('jd_curl', 'scripts') for p in (target/d).rglob('*.py')}
    if live != reference:
        raise SystemExit('the scratch copy differs from the freeze hashes')
    return len(live)


def pending_text(pending):
    """What is still missing, in the words of the plan: the retry's pairs by their fixed subject."""
    parts = []
    for arm, pairs in pending.items():
        retry = [p for p in pairs if entry_root(arm, p) == RETRY_ROOT]
        rest = [p for p in pairs if entry_root(arm, p) != RETRY_ROOT]
        if rest:
            parts.append(f"{arm} 入口 {'、'.join(p.split('_')[2] for p in rest)} 待 {ARMS[arm]['pending']}")
        if retry:
            parts.append(f"{arm} 入口 {'、'.join(p.split('_')[2] for p in retry)} 待补跑")
    return '；'.join(parts)


def protocol_differences(new, old):
    """The keys of two protocol.json files that differ, task fields one by one."""
    out = {k: (old.get(k), new.get(k)) for k in sorted(set(new) | set(old)) if k != 'task' and new.get(k) != old.get(k)}
    out.update({f'task.{k}': (old['task'].get(k), new['task'].get(k)) for k in sorted(set(new['task']) | set(old['task'])) if new['task'].get(k) != old['task'].get(k)})
    return out


def rehearse_reading(scratch, out, results=None):
    """The reading on the control's own numbers (copies of B1TM's and B1W_A5's metrics.json as S_P, S_T): every delta is 0, so the code must say no difference.  Formats and pins only; deleted with the scratch directory."""
    import decide_seed
    results = RESULTS if results is None else results
    root = scratch/'seed_rehearsal'
    for arm in ARMS:
        for pair in PAIRS:
            doc = read_json(Path(results)/'b1_v1/runs'/ARMS[arm]['control']/pair/'metrics.json')
            doc.update(arm=arm, run_id=f'rehearsal_{arm}_{short_name(pair)}', method=method_name(arm), seed=dict(passed=True, problems=[], accepted_alpha=None, arm=dict(seconds=SECONDS, changes=CHANGES, inputs_root=str(inputs_root(arm, results)), charge_seconds=ENTRY_CHARGE_SECONDS)))
            target = root/'runs'/arm/pair
            target.mkdir(parents=True)
            write_json(target/'metrics.json', doc)
    result = decide_seed.write_analysis(root, results=results, read_entries=False)      # no number of an entry is opened for this rehearsal
    out(f"reading on copies of the control: {result['reading_code']}; reference problems {result['reference_problems']}; pins reproduced: B1TM {result['pins']['B1TM']}, B1W_A5 {result['pins']['B1W_A5']}; line in the first 5 pairs {result['line_86']['first5_line']}; files {sorted(p.name for p in (root/'analysis').iterdir())}")
    if result['reference_problems'] or result['reading_code'] != decide_seed.NO_DIFFERENCE:
        raise SystemExit('the rehearsal of the reading did not give what it must')


def rehearse_evaluation(scratch, out, results=None):
    """evaluate.py evaluate_run on one finished B1TM run (read only): the same metrics as its metrics.json, under the method name of the arm.  This reads the truth labels of one development pair, for the evaluation only."""
    ev = load_evaluate()
    ev.STAGE = MAIN_REPO/'data/three_d/stage_L2R_val_v1'
    results = RESULTS if results is None else results
    run = Path(results)/'b1_v1/runs/B1TM'/PAIRS[0]
    old = read_json(run/'metrics.json')
    tick = time.perf_counter()
    new = ev.evaluate_run(run, 'S_P', PAIRS[0], 'rehearsal', method=method_name('S_P'))
    same = json.loads(json.dumps(new['metrics'], default=float)) == old['metrics'] and new['certificate'] == old['certificate'] and new['self_check']['passed'] == old['self_check']['passed']
    out(f'evaluate_run on {run.name} of B1TM: metrics, certificate and self-check equal to its metrics.json: {same}; method {new["method"]}; {time.perf_counter()-tick:.0f} s')
    if not same:
        raise SystemExit('evaluate_run does not reproduce the metrics of the run')


def dry_run(scratch=None, out=print, results=None, repo=None, skip_entries=()):
    """skip_entries: arms whose entry files are not opened (--skip-entries S_P: while MC0P is running, its files are not read).  The scratch directory goes, however the dry run ends."""
    results = RESULTS if results is None else results
    repo = MAIN_REPO if repo is None else repo
    made = scratch is None
    scratch = Path(tempfile.mkdtemp(prefix='seed_dry_')) if made else Path(scratch)
    if not made and scratch.exists():
        raise SystemExit(f'{scratch} exists (nothing is overwritten)')
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        return dry_run_steps(scratch, out, results, repo, skip_entries)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def dry_run_steps(scratch, out, results, repo, skip_entries):
    root = Path(results)/'seed_v1'
    link = frozen_dir(results)/'data/three_d/stage_L2R_val_v1'
    link_before = os.readlink(link)
    existed = root.exists()
    out(f'SEED dry run ({utc()}): no solver, no GPU (CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")!r}), no unit; everything written under a scratch directory, deleted at the end.')
    out(f'{len(plan_runs())} runs; --seconds {SECONDS}; --geometry-weight-levels {",".join(f"{x:g}" for x in GEOMETRY)}; entry charge {ENTRY_CHARGE_SECONDS} s inside the clock; watchdog {WATCHDOG} s; root {root}')
    problems = []

    selection = json.loads((MAIN_REPO/'data/three_d/stage_L2R_val_v1/selection.json').read_text())
    first = tuple(row['pair'] for row in selection['pairs'][:5])
    out(f"selection.json first 5 == the plan's pairs: {first == PAIRS}")
    if first != PAIRS:
        problems.append('the first 5 pairs of selection.json differ')
    sys.path.insert(0, str(ROOT))
    from bayesreg.io.guard import check_pair_subjects, load_dev_allowlist
    allowlist = load_dev_allowlist(ROOT/'configs/bayesreg/dev_subjects.json')

    out('\n==== the entries (real files: run.json, npy header, file sha256, the field bytes; no optimization) ====')
    copy = scratch/'frozen_copy'
    n_files = frozen_copy(copy, results)
    out(f'frozen source copied: {n_files} files equal to the freeze hashes; the worktree link stays')
    entries = {}
    pending = {}
    for arm, pair in plan_runs():
        files = input_files(arm, pair, results)
        check_pair_subjects(files[3]['fixed_subject'], files[3]['moving_subject'], allowlist)
        if arm in skip_entries:
            pending.setdefault(arm, []).append(pair)
            out(f"{arm} {short_name(pair)}: not opened (--skip-entries): {entry_dir(arm, pair, results)}")
            continue
        try:
            entries[(arm, pair)] = check_entry(arm, pair, files[3], results)
            e = entries[(arm, pair)]
            taper = 'the evaluation has no record yet' if e['tapered_recorded_by_evaluation'] is None else e['tapered_sha256'] == e['tapered_recorded_by_evaluation']
            out(f"{arm} {short_name(pair)}: {e['field_path']} sha256 {e['field_sha256'][:16]} bytes {e['u_sha256'][:16]} source {e['source_run_id']} oracle {e['oracle']} seconds_total {e['source_seconds_total']:.2f}; taper check against the evaluation's record: "
                f"{taper}; subjects on the allowlist; the evaluation accepted alpha {e['entry_metrics']['accepted_alpha']}")
        except EntryMissing as missing:
            pending.setdefault(arm, []).append(pair)
            out(f'{arm} {short_name(pair)}: {missing}')
        except EntryError as error:
            problems.append(str(error))
            out(f'{arm} {short_name(pair)}: REFUSED: {error}')
    if pending:
        out(f"pending entries: { {a: len(p) for a, p in pending.items()} }; {pending_text(pending)}: not an error of the dry run; the queue stops at its start checks, before any run, until they exist")
    out('wrong sources are refused: S_T on an oracle-false record, S_P on an oracle-true record, the wrong pair: tests/test_bayesreg/solve/test_seed_tools.py')

    out('\n==== the colour table and registry checks on the real inputs records (main checkout) ====')
    system = System()
    for arm in ARMS:
        bad = system.inputs_problems(arm, repo, results)
        out(f'{arm}: {inputs_root(arm, results)}: {"ok" if not bad else bad}')
        problems += bad

    out('\n==== all 10 runs: the real parser and prepare of the frozen source (scratch copy), check_prepared with the arm, the protocol against the control of the same pair; then the unit that the queue would start ====')
    prepared = scratch/'prepared'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    scratch_link = copy/'data/three_d/stage_L2R_val_v1'
    scratch_link.parent.mkdir(parents=True, exist_ok=True)
    for k, (arm, pair) in enumerate(plan_runs(), 1):
        files = input_files(arm, pair, results)
        shown = run_dir(root, arm, pair)
        out(f"# item {k}: arm {arm}, pair {pair}, inputs {files[0]}, entry {entry_dir(arm, pair, results)}/field.npz, control {ARMS[arm]['control']}, watchdog {WATCHDOG} s, unit {unit_name(arm, pair)}, run id seed_{arm}_{short_name(pair)}_<UTC>")
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
        check = system.check_prepared(here, pair, f'SEED-{arm}', files[1], files[2], b1_entry(arm, results))
        if not check['ok']:
            raise SystemExit(f"check_prepared refused {arm} {pair}: {check['problems']}")
        control = Path(results)/'b1_v1/runs'/ARMS[arm]['control']/pair/'protocol.json'
        differs = protocol_differences(read_json(here/'protocol.json'), read_json(control))
        out(f"parser, prepare and check_prepared ok ({time.perf_counter()-tick:.1f} s); protocol against {ARMS[arm]['control']} of this pair differs only in: {json.dumps(differs, ensure_ascii=False)}")
        if set(differs) != {'stage'}:
            problems.append(f"{arm} {pair}: the protocol differs from {ARMS[arm]['control']} in {sorted(differs)}")
        shutil.rmtree(here)

    out('\n==== the queue start string (the lead) ====')
    out(shlex.join(queue_launch(root, repo)))

    out('\n==== negative controls on the frozen parser and the protocol check ====')
    tokens = prepare_argv('S_P', PAIRS[0], prepared/'refused', results)
    tokens[2] = str(copy/'scripts/run_stage_3d_e48_pg.py')
    tokens[tokens.index('--seconds')+1] = '285'
    done = subprocess.run(tokens, cwd=scratch, env=env, capture_output=True, text=True)
    out(f"--seconds 285 (the brief's number): exit {done.returncode}; {done.stderr.strip().splitlines()[-1] if done.stderr.strip() else ''}; directory made: {(prepared/'refused').exists()}")
    if done.returncode == 0:
        problems.append('the frozen parser took --seconds 285')
    files = input_files('S_P', PAIRS[0], results)
    tokens = prepare_argv('S_P', PAIRS[0], prepared/'gw', results)
    tokens[2] = str(copy/'scripts/run_stage_3d_e48_pg.py')
    tokens[tokens.index('--geometry-weight-levels')+1] = '1,1,0.1,0.03'
    scratch_link.unlink()
    scratch_link.symlink_to(files[0].parent)
    subprocess.run(tokens, cwd=scratch, env=env, capture_output=True, text=True, check=True)
    check = system.check_prepared(prepared/'gw', PAIRS[0], 'SEED-S_P', files[1], files[2], b1_entry('S_P', results))
    out(f"configuration C's geometry weight against the arm: ok {check['ok']}: {check['problems']}")
    if check['ok']:
        problems.append('check_prepared took the wrong geometry weight')

    out('\n==== the reading on the real baseline (copies of the control as the arms: formats and pins only, not results) ====')
    rehearse_reading(scratch, out, results)
    out('\n==== experiments/d0/evaluate.py evaluate_run on one finished B1TM run (read only) ====')
    rehearse_evaluation(scratch, out, results)

    if os.readlink(link) != link_before:
        problems.append("the frozen worktree's link changed")
    if root.exists() != existed:
        problems.append(f'{root} was created')
    out(f"\nthe frozen worktree's link unchanged: {link_before}; {root} {'exists' if existed else 'not created'}; no solver run; the scratch directory goes when this ends")
    out(f"torch imported by this dry run: {'torch' in sys.modules}; problems: {problems or 'none'}")
    return 8 if problems else 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['--dry-run']:
        d = argparse.ArgumentParser()
        d.add_argument('--dry-run', action='store_true')
        d.add_argument('--scratch', default=None)
        d.add_argument('--skip-entries', nargs='*', default=[], choices=tuple(ARMS))
        parsed = d.parse_args(argv)
        return dry_run(scratch=parsed.scratch, skip_entries=tuple(parsed.skip_entries))
    p = argparse.ArgumentParser(description='SEED: see the module docstring')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('queue')
    q.add_argument('--root', default=str(DEFAULT_ROOT))
    q.add_argument('--wait-minutes', type=int, default=30)
    sub.add_parser('decide').add_argument('--root', default=str(DEFAULT_ROOT))
    e = sub.add_parser('evaluate-one')
    e.add_argument('run')
    e.add_argument('--arm', required=True)
    e.add_argument('--pair', required=True)
    e.add_argument('--run-id', required=True)
    o = p.parse_args(argv)
    if o.command in ('queue', 'decide') and Path(o.root).name == SEED19_ROOT_NAME:
        print(f'{o.root}: the root of SEED19: experiments/seed/run_seed19.py; nothing read, nothing moved', file=sys.stderr)
        return 2
    if o.command == 'queue':
        return queue(o)
    if o.command == 'decide':
        import decide_seed
        decide_seed.write_analysis(o.root)
        return 0
    return evaluate_one(o)


if __name__ == '__main__':
    sys.exit(main())
