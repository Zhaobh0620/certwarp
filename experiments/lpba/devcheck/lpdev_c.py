"""LP-D (c) (runs/20261119-lpba/任务书_D_开发集逐位对照.md; prereg §12 item 4 (c), §8.1 item 9; contract C26.5-11 (c) with revision 11b item 4, C26.4 L39, C26.5-12 N56): the method entry,
in HE's development mode, on the development pair 0438 <- 0439, once; then the comparison.  Two conditions only (the preregistration's): the T88 field t88/field_0600.npz's u is bitwise HS's
T88 field of the same pair, and the pair's line status is hx_certified (fallback one or two, or any other status, does not pass).  The repaired field is not compared, only recorded.

The run (not started by this package): experiments/he/run_hx.py pair --stage-dir <stage_L2R_val_v1> --pair OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1 --out <c_run>/0438-0439 --device cuda
--launcher unit, in the unit tag-lp-devc with HE's supervisor unit (he_core.SUPERVISOR_UNIT: MemoryMax 12G, MemorySwapMax 4G, OOMPolicy continue, CPUQuota 1200 %, RuntimeMaxSec 4800,
NUMBA 8, nice 5), PYTHONPATH <frozen eb808be>:<code>: the development pair set (run_hx.make_context's default pair_set 'dev'), so every process of the pair admits it through
run_hx.sealed_admission (bayesreg.eval.truth's sealed lists on the names and the manifest's subjects) and load_pair with the development allowlist (C26.5-11 revision 11b item 4); the frozen
settings (settings_sha256 414eccd3...); a new --out under the results root (run_hx refuses one that exists: the run is once by its own rule).  It runs no evaluation (run_hx.py evaluate is not
called): no truth label is read.  `launch-string` prints the start string; `gate` checks, once, what HE's queues check before a supervisor unit (holdout_common.machine_gate with MemAvailable >=
he_core.FROZEN_RUN_GATE_GIB: no other tag-* or seq-vp-* unit, the GPU idle, the load, the gap after the previous unit), the hold, the disks and the run's output absent -- it starts nothing.

The reference (HS, the first selection reading; runs/20261106-hs/result.md): <RESULTS>/hs_v1/runs/SHx24_first5/0438-0439/run.json (status done, the pair, field and checkpoints['600'] naming
field_0600.npz with its file sha256 and u_sha256) and field_0600.npz (member u); both files' sha256 are code constants (HS_RUN_SHA256, HS_FIELD_SHA256; review minor 8), checked before the
lock.  Nothing else of hs_v1 is read (metrics_0600.json beside it is an evaluation: the spy refuses its name).

The comparison (`compare`, once: the lock <RESULTS>/lpba40_devcheck_v1/locks/devcheck_c.lock).  Before the lock: the admission (hp_core.admit_dev on the pair: sealed lists, check_subject on the
development allowlist, its source stage_L2R_val_v1); the pins (lpba_common.check_pins, recorded; review minor 3); the hold; the reference against its pinned sha256; and the run ended (review
I1, run_ended): the unit tag-lp-devc neither active nor activating (nor deactivating, reloading: systemctl --user show, ActiveState; a query that fails refuses) and at least one of the run's
terminal files there (TERMINAL_FILES: output/line.json, line.json, line_not_written.json, unit_refused.json).  After the lock every file of the run may be missing: a missing one is a reason
of "not passed" in the record, never a refusal (I1).  The checks:
  the run's context.json   the pair, the development stage, pair set dev, the identity checked, the frozen settings with the pinned settings_sha256, no test context, code_head = the commit
                           this comparison runs on (review minor 2; C26.9-3: (c) and the feasibility on one commit);
  identity*.json           identity.json there with no problem; identity_t88.json, identity_repair.json, identity_end.json, where there, with no problem (review minor 2);
  t88/run.json             schema bayesreg.he.t88.v1, status done, oracle false, the pair, field field_0600.npz; the file's sha256 and u's sha256 the record's;
  hs run.json              status done, the pair, the run id, oracle false, field_0600.npz, its sha256 and u_sha256 (the same in checkpoints['600']); the file's sha256 and u's sha256 the record's;
  the fields               u (HE) and u (HS) bitwise: dtype, shape, C-order bytes                                                        -> condition 1
  line.json                output/line.json (an output) or line.json (none): the pair, line_status hx_certified                          -> condition 2
  described                output/cert.json (process, status, state_hash, min_jd_lower_bound), line.json's repair, fuse, fallback and files, line_not_written.json and unit_refused.json
                           where there, both runs' labels_sha256 (the T88 input channels) and whether they are equal (review minor 8) -- not judged.
passed = conditions 1 and 2, every record above there and consistent, and the spy passed; `reasons` lists why not.

    python -B experiments/lpba/devcheck/lpdev_c.py dry-run --out DIR [--code DIR]     the admission, the reference (stat only), the run's output, the start string; DIR/dry_run.json
    python -B experiments/lpba/devcheck/lpdev_c.py launch-string --code DIR            the start string of the run (tag-lp-devc); prints only
    python -B experiments/lpba/devcheck/lpdev_c.py gate                                the gate before the run, once; prints only
    python -B experiments/lpba/devcheck/lpdev_c.py compare                             only after systemd-run --wait returned: <RESULTS>/lpba40_devcheck_v1/c/devcheck_c.json
Exit codes: lpdev_common.EXIT (gate: 0 passed, 8 refused -- any exception there, the disk guard's included; review minor 4).
"""
import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import lpdev_common as dc  # noqa: E402

lc, hc, hp = dc.lc, dc.hc, dc.hp
import he_core as he  # noqa: E402

import numpy as np  # noqa: E402

SCHEMA = 'bayesreg.lpba40.devcheck_c.v1'
SCRIPT = 'experiments/lpba/devcheck/lpdev_c.py'
UNIT = 'tag-lp-devc'
FIXED, MOVING = 'OASIS_OAS1_0438_MR1', 'OASIS_OAS1_0439_MR1'
PAIR_NAME = f'{FIXED}__{MOVING}'
SHORT = '0438-0439'
DEV_STAGE = hc.DEV_STAGE
HS_REFERENCE_REL = f'hs_v1/runs/SHx24_first5/{SHORT}'        # hs_core.ROOT_NAME, fx_core.Batch.run_dir(root, Cell('SHx24', 'first5', pair)); found on 2026-11-19 by path, not by a listing
HS_RUN_ID = f'hs_SHx24_first5_{SHORT}'
FIELD_FILE = he.FIELD_FILE                                   # field_0600.npz, in both
T88_SCHEMA = he.SCHEMA.format('t88')
CONTEXT_SCHEMA = he.SCHEMA.format('context')
RECORD = 'devcheck_c.json'
STOPPED = 'devcheck_c_stopped.json'
HS_RUN_SHA256 = '544ee1a453be59f3e28a756a569fcd1412351f5f3344552187d9648dd5bd9e64'       # hs_v1/runs/SHx24_first5/0438-0439/run.json, read 2026-11-19 (review minor 8)
HS_FIELD_SHA256 = 'b5be9924cf48e1722a8008a2ec7b61385d1ed894819a4111f0a84eee8e9c53b2'     # its field_0600.npz (= that run.json's field.sha256 and checkpoints['600'].sha256)
TERMINAL_FILES = ('output/line.json', 'line.json', 'line_not_written.json', 'unit_refused.json')    # run_hx.run_pair's ends (review I1): an output, no output, a check failed at the end, the unit's guard
RUNNING_STATES = ('active', 'activating', 'deactivating', 'reloading')
IDENTITY_FILES = ('identity.json', 'identity_t88.json', 'identity_repair.json', 'identity_end.json')       # run_hx.check_identity's records of the supervisor and its two children


def hs_reference():
    return Path(lc.RESULTS)/HS_REFERENCE_REL


def run_out():
    """The run's --out (new; run_hx refuses one that exists)."""
    return dc.roots().c_run/SHORT


def reference_paths():
    return [hs_reference()/'run.json', hs_reference()/FIELD_FILE]


def run_paths():
    o = run_out()
    return [o/'context.json', *(o/n for n in IDENTITY_FILES), o/'t88'/'run.json', o/'t88'/FIELD_FILE, *(o/rel for rel in TERMINAL_FILES), o/'output'/'cert.json']


def unit_log():
    return Path(lc.RESULTS)/f'{UNIT}_unit.log'                 # outside every output directory (which must be new); the results root exists (systemd makes no directory)


def launch_argv(code):
    """The start string of the run (module docstring): HE's supervisor unit (he_core.SUPERVISOR_UNIT, checked by he_core.unit_argv_problems), PYTHONPATH <frozen>:<code>, single-threaded BLAS, the
    log appended to unit_log().  ValueError when the string is not the unit's."""
    code = Path(code)
    res = he.SUPERVISOR_UNIT
    env = dict(OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, NUMBA_NUM_THREADS=res['numba'], PYTHONDONTWRITEBYTECODE=1,
               PYTHONPATH=f'{he.core.FROZEN}:{code}')
    log = unit_log()
    argv = ['systemd-run', '--user', f'--unit={UNIT}', '--wait', '--collect', f'--working-directory={code}', f"--property=MemoryMax={res['memory_max']}",
            f"--property=MemorySwapMax={res['memory_swap_max']}", f"--property=OOMPolicy={res['oom_policy']}", f"--nice={res['nice']}", f"--property=CPUQuota={res['quota']}%",
            f"--property=RuntimeMaxSec={res['runtime']}", f'--property=StandardOutput=append:{log}', f'--property=StandardError=append:{log}', *[f'--setenv={k}={v}' for k, v in env.items()],
            dc.PY, '-B', str(code/'experiments'/'he'/'run_hx.py'), 'pair', '--stage-dir', str(DEV_STAGE), '--pair', PAIR_NAME, '--out', str(run_out()), '--device', 'cuda', '--launcher', 'unit']
    problems = he.unit_argv_problems(argv, res)
    if problems or not UNIT.startswith('tag-') or UNIT.startswith('seq-vp-'):
        raise ValueError(f'{UNIT}: the start string is not HE\'s supervisor unit {res}: {problems}')
    return argv


def launch_string(code):
    return ' '.join(launch_argv(code))


def gate(*, run=None):
    """Before the run, once (module docstring): the hold, the run's output absent, holdout_common.machine_gate (MemAvailable >= he_core.FROZEN_RUN_GATE_GIB), the disks.  Returns the record;
    Refused otherwise.  Starts nothing."""
    record = dict(hold=lc.check_hold('lpba40_feas'))
    if run_out().exists():
        raise lc.Refused('output', f'{run_out()} exists: the run is once (run_hx.py pair refuses an --out that exists)')
    record['machine'] = hc.machine_gate(own_units=(), mem_min_gib=he.FROZEN_RUN_GATE_GIB, **({} if run is None else dict(run=run)))
    record['disk'] = hc.disk_guard(need_d_gib=0, need_home_gib=2.)
    return record


def code_tree_reads():
    """lpba_common.code_reads (the pins lpba_common.check_pins reads, the sealed lists) and the development allowlist."""
    return lc.code_reads()+[lc.repo_root()/hp.ALLOWLIST_REL]


def policy(*, out=None):
    r = dc.roots()
    return dc.Policy(name='lpba devcheck (c)', read_exact=code_tree_reads()+reference_paths()+run_paths(), write_inside=[r.c if out is None else out], write_exact=[r.c_lock],
                     npz_members={hs_reference()/FIELD_FILE: ('u',), run_out()/'t88'/FIELD_FILE: ('u',)}, results=hc.RESULTS)


def load_u(path):
    with np.load(path, allow_pickle=False) as z:
        return np.array(z['u'])


def check_context(ctx, code_commit):
    """The run was the frozen method in the development mode on this pair, on the commit this comparison runs on (module docstring).  {check: bool}."""
    stage = ctx.get('stage_dir')
    return dict(schema=ctx.get('schema') == CONTEXT_SCHEMA, pair=ctx.get('pair') == PAIR_NAME, stage=stage is not None and Path(stage).resolve() == Path(DEV_STAGE).resolve(),
                pair_set_dev=ctx.get('pair_set', 'dev') == 'dev', identity_checked=ctx.get('check_identity') is True, frozen=ctx.get('settings_frozen') is True,
                settings_sha256=ctx.get('settings_sha256') == lc.SETTINGS_SHA256 == he.settings_sha256(ctx.get('settings') or {}), not_a_test_context=ctx.get('test_context') is None,
                out=Path(str(ctx.get('out'))).resolve() == run_out().resolve(), code_head_is_this_commit=bool(code_commit) and ctx.get('code_head') == code_commit)


def unit_state(*, run=None):
    """The unit's ActiveState (systemctl --user show tag-lp-devc --property=ActiveState --value; a unit that is gone reads inactive).  A query that fails or gives nothing: Refused (fail closed)."""
    run = subprocess.run if run is None else run
    try:
        done = run(['systemctl', '--user', 'show', UNIT, '--property=ActiveState', '--value'], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise lc.Refused('unit', f'systemctl cannot be asked about {UNIT}: {type(exc).__name__}: {exc}') from None
    state = (done.stdout or '').strip()
    if done.returncode != 0 or not state:
        raise lc.Refused('unit', f'systemctl --user show {UNIT}: returncode {done.returncode}, ActiveState {state!r}: the end of the run cannot be told')
    return state


def run_ended(*, run=None):
    """Review I1, before the lock: the unit not running (RUNNING_STATES) and at least one of TERMINAL_FILES there.  Returns the record; Refused otherwise (nothing spent)."""
    state = unit_state(run=run)
    if state in RUNNING_STATES:
        raise lc.Refused('unit', f'{UNIT} is {state}: the run has not ended; compare only after systemd-run --wait returned')
    there = [rel for rel in TERMINAL_FILES if (run_out()/rel).is_file()]
    if not there:
        raise lc.Refused('input', f'{run_out()}: none of {list(TERMINAL_FILES)} is there: the run has not ended, or never started (has the run ended?)')
    return dict(unit=UNIT, active_state=state, terminal_files=there, rule='the unit neither active nor activating (nor deactivating, reloading) and one of TERMINAL_FILES there (review I1)')


def check_reference():
    """Before the lock: the reference's run.json and field_0600.npz have the pinned sha256 (review minor 8).  Returns the record; Refused otherwise."""
    got = {}
    for path, want in ((hs_reference()/'run.json', HS_RUN_SHA256), (hs_reference()/FIELD_FILE, HS_FIELD_SHA256)):
        got[path.name] = hc.sha256_file(path) if path.is_file() else None
        if got[path.name] != want:
            raise lc.Refused('reference', f'{path}: sha256 {got[path.name]} is not the pinned {want}')
    return dict(reference=str(hs_reference()), sha256=got, rule='HS_RUN_SHA256, HS_FIELD_SHA256 (review minor 8)')


def read_optional(path):
    return dc.read_json(path) if Path(path).is_file() else None


def field_of(directory, record, *, what):
    """(u, its record of checks): the field file's sha256 and u's sha256 against the run's record."""
    field = record.get('field') or {}
    path = Path(directory)/FIELD_FILE
    file_sha = hc.sha256_file(path)
    u = load_u(path)
    u_sha = lc.array_sha256(u)
    checks = dict(file_named=field.get('file') == FIELD_FILE, file_sha256=file_sha == field.get('sha256'), u_sha256=u_sha == field.get('u_sha256'))
    return u, dict(what=what, path=str(path), file_sha256=file_sha, u_sha256=u_sha, recorded=dict(field), checks=checks)


def compare_run(*, code_commit, log=print):
    """After the lock: the comparison of the module docstring.  A file of the run that is missing is a reason of "not passed" (review I1).  Returns its record."""
    o, ref = run_out(), hs_reference()
    reasons = []
    ctx = read_optional(o/'context.json')
    context = check_context(ctx, code_commit) if ctx is not None else dict(present=False)
    identities = {name: read_optional(o/name) for name in IDENTITY_FILES}
    identity = {name: (doc is not None and doc.get('problems') == []) if name == 'identity.json' else (doc is None or doc.get('problems') == []) for name, doc in identities.items()}
    t88 = read_optional(o/'t88'/'run.json')
    t88_checks = dict(present=False) if t88 is None else dict(schema=t88.get('schema') == T88_SCHEMA, status_done=t88.get('status') == he.T88_DONE, oracle_false=t88.get('oracle') is False,
                                                              pair=t88.get('pair') == PAIR_NAME)
    u, field = (None, dict(present=False, checks=dict(present=False))) if t88 is None or not (o/'t88'/FIELD_FILE).is_file() else \
        field_of(o/'t88', t88, what='the run (HE, development mode)')
    hs = dc.read_json(ref/'run.json')
    checkpoint = (hs.get('checkpoints') or {}).get('600') or {}
    hs_checks = dict(status_done=hs.get('status') == 'done', pair=hs.get('pair') == PAIR_NAME, run_id=hs.get('run_id') == HS_RUN_ID, oracle_false=hs.get('oracle') is False,
                     checkpoint_is_field=all(checkpoint.get(k) == (hs.get('field') or {}).get(k) for k in ('file', 'sha256', 'u_sha256')))
    u_hs, hs_field = field_of(ref, hs, what='HS (hs_v1), the reference')
    if u is None:
        fields = dict(bitwise_equal=False, reason=f'the run has no T88 field ({o/"t88"/FIELD_FILE} or t88/run.json missing)')
    else:
        fields = dc.arrays_equal(u, u_hs)
    del u, u_hs
    line_path = next((o/rel for rel in ('output/line.json', 'line.json') if (o/rel).is_file()), None)
    line = read_optional(line_path) if line_path is not None else {}
    cert = read_optional(o/'output'/'cert.json')
    ends = {rel: read_optional(o/rel) for rel in ('line_not_written.json', 'unit_refused.json')}
    conditions = dict(t88_field_bitwise_hs=fields['bitwise_equal'],
                      line_status_hx_certified=line_path is not None and line_path.parent.name == 'output' and line.get('line_status') == he.HX and line.get('pair') == PAIR_NAME)
    consistent = dict(context=bool(context) and all(context.values()), identity=all(identity.values()), t88_record=all(t88_checks.values()), t88_field=all(field['checks'].values()),
                      hs_record=all(hs_checks.values()), hs_field=all(hs_field['checks'].values()))
    missing = [rel for rel in ('context.json', 'identity.json', 't88/run.json', f't88/{FIELD_FILE}') if not (o/rel).is_file()]
    reasons += [f'missing: {rel}' for rel in missing]
    reasons += [f'condition not met: {k}' for k, v in conditions.items() if not v]
    reasons += [f'not consistent: {k}' for k, v in consistent.items() if not v]
    log(f"T88 field bitwise HS: {fields['bitwise_equal']} ({fields.get('n_elements_differ')} elements differ, max {fields.get('max_abs_difference')}); line status {line.get('line_status')!r}; "
        f"reasons {reasons}")
    files = [dc.file_record(p) for p in [o/'context.json', *(o/n for n in IDENTITY_FILES), o/'t88'/'run.json', o/'output'/'cert.json', *(o/rel for rel in TERMINAL_FILES), ref/'run.json']
             if p.is_file()]
    files += [dict(path=f['path'], sha256=f['file_sha256']) for f in (field, hs_field) if f.get('path')]
    labels = dict(run=None if t88 is None else t88.get('labels_sha256'), hs=hs.get('labels_sha256'))
    labels['equal'] = labels['run'] is not None and labels['run'] == labels['hs']
    described = dict(identity_problems={n: None if d is None else d.get('problems') for n, d in identities.items()}, code_head=None if ctx is None else ctx.get('code_head'),
                     code_commit=code_commit, cert=None if cert is None else {k: cert.get(k) for k in ('process', 'status', 'state_hash', 'min_jd_lower_bound', 'passed')},
                     line={k: line.get(k) for k in ('line_status', 'repair', 'fuse', 'fallback', 'failure', 'rejection', 'decision_certificate', 'output', 't88', 'seconds_total')},
                     ends={rel: None if d is None else {k: d.get(k) for k in ('reason', 'line_status', 'problems', 'rejection')} for rel, d in ends.items()},
                     t88=None if t88 is None else {k: t88.get(k) for k in ('status', 'reason', 'seconds_task', 'seconds_total', 'settings_sha256', 'env')}, labels_sha256=labels,
                     note='the repaired field is not compared (C26.5-11 (c)); recorded only; labels_sha256 (the T88 input channels) described, not judged (review minor 8)')
    return dict(pair=PAIR_NAME, run_out=str(o), reference=str(ref), context=context, identity=identity, t88=t88_checks, field=field, hs=hs_checks, hs_field=hs_field, fields=fields,
                line_file=None if line_path is None else str(line_path), conditions=conditions, consistent=consistent, missing=missing, reasons=reasons, described=described, files_read=files,
                passed=not reasons)


def compare(*, code, dry_run=False, out=None, launch_code=None, systemctl=None, log=print):
    """The order: the code; the spy; the admission (hp_core.admit_dev on the pair); the pins; the hold; the record's directory new, the lock absent; (a dry run: stat only, and the start
    string); the reference against its pinned sha256; the run ended (unit and terminal files, review I1); the lock; the comparison; the record.  `systemctl`: subprocess.run's stand-in
    (tests).  Returns the record."""
    r = dc.roots()
    out_dir, lock = r.c, r.c_lock
    lc.guard_write(out_dir, lock)
    record = dict(schema=SCHEMA, tool=SCRIPT, task=dc.TASK, contract=dc.CONTRACT, pair=PAIR_NAME, run_out=str(run_out()), reference=str(hs_reference()), started_utc=dc.utc(),
                  out=str(out_dir), lock=str(lock), dry_run=dry_run, code=code, unit=UNIT)
    record['preload'] = lc.preload()
    with dc.spying(policy(out=out if dry_run else None)) as spy:
        _, admission = hp.admit_dev(pairs=[(FIXED, MOVING)])
        record['admission'] = admission
        record['pins'] = lc.check_pins()                       # review minor 3
        record['hold'] = lc.check_hold('lpba40_feas')
        if out_dir.exists():
            raise lc.Refused('output', f'{out_dir} exists: the comparison runs once')
        if lock.exists():
            raise lc.Refused('once', f'{lock} exists: the comparison ran; it runs once')
        if dry_run:
            record['planned'] = dict(reference=dc.planned(reference_paths()), run=dc.planned(run_paths()), run_out_exists=run_out().exists())
            record['launch'] = launch_string(launch_code or (code or {}).get('root') or lc.REPO)
            record['spy'] = spy.record()
            return record
        record['reference_check'] = check_reference()
        record['run_ended'] = run_ended(run=systemctl)
        line = hc.lock_line(tool=SCRIPT, out=str(out_dir), run_out=str(run_out()), reference=str(hs_reference()), code_commit=(code or {}).get('commit'), event='first_run')
        record['lock_line'] = hc.take_lock(lock, line)
        with dc.after_lock(record, spy, out_dir, lock, STOPPED):        # review minor 5: everything after the lock, the record's write included
            out_dir.mkdir(parents=True)
            record['comparison'] = compare_run(code_commit=(code or {}).get('commit'), log=log)
            record['spy'] = spy.record(expect=reference_paths()+[p for p in (run_out()/'t88'/FIELD_FILE, run_out()/'t88'/'run.json', run_out()/'context.json') if p.is_file()])
            record['ended_utc'] = dc.utc()
            record['passed'] = bool(record['spy']['passed']) and record['comparison']['passed']
            record['conclusion'] = 'passed' if record['passed'] else 'not passed'
            hc.write_json_once(out_dir/RECORD, record)
    return record


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog=SCRIPT, description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    d = sub.add_parser('dry-run')
    d.add_argument('--out', type=Path, required=True, help='a new or empty directory outside the results root and the code trees (e.g. mktemp -d)')
    d.add_argument('--code', type=Path, default=None, help='the checkout the run will use (for the start string; default this code tree)')
    s = sub.add_parser('launch-string')
    s.add_argument('--code', type=Path, required=True, help='the clean detached checkout the unit runs (PYTHONPATH <frozen>:<code>)')
    sub.add_parser('gate')
    sub.add_parser('compare')
    return ap.parse_args(argv)


def main(argv=None, *, code=None, out=print):
    a = parse_args(argv)
    if a.command == 'launch-string':
        out(launch_string(a.code))
        return dc.EXIT['passed']
    if a.command == 'gate':
        record, exit_code = dc.guarded(gate, dry=False, out_dir=None, schema=SCHEMA, out=out)      # review minor 4: the disk guard's InfrastructureFailure is a refusal too
        if exit_code is not None:
            return exit_code
        out(f"gate passed: MemAvailable {record['machine'].get('mem_available_gib')} GiB, load {record['machine'].get('load1')}; disks {record['disk'].get('free_gib')}; start the unit now "
            f"(launch-string), nothing was started here")
        return dc.EXIT['passed']
    dry = a.command == 'dry-run'
    if dry:
        problem = lc.dry_run_out_problem(a.out)
        if problem:
            out(problem)
            return dc.EXIT['usage']
    else:
        problem = dc.bytecode_problem()
        if problem:
            out(f'refused: {problem}')
            return dc.EXIT['refused']
    record, exit_code = dc.guarded(lambda: compare(code=dc.check_code(hp.code_identity() if code is None else code, dry_run=dry), dry_run=dry, out=a.out if dry else None,
                                                   launch_code=getattr(a, 'code', None), log=out), dry=dry, out_dir=a.out if dry else None, schema=SCHEMA, out=out)
    if exit_code is not None:
        return exit_code
    if dry:
        dc.write_dry_run(a.out, dict(record, schema=SCHEMA+'.dry_run'))
        ref, runs = record['planned']['reference'], record['planned']['run']
        out(f"dry run passed: the pair admitted; reference files there {sum(p['exists'] for p in ref)}/{len(ref)}; the run's output exists {record['planned']['run_out_exists']}; "
            f"spy breaches {len(record['spy']['breaches'])}; start string: {record['launch']}")
        return dc.EXIT['passed']
    c = record['comparison']
    out(f"done: {record['conclusion']} (T88 field bitwise HS {c['conditions']['t88_field_bitwise_hs']}, hx_certified {c['conditions']['line_status_hx_certified']}; {record['out']}/{RECORD})")
    return dc.EXIT['passed'] if record['passed'] else dc.EXIT['not_passed']


if __name__ == '__main__':
    sys.exit(main())
