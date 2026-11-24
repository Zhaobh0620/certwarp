"""PQH, the smoke's segment (iii) (contract C22.H8 (iii), D120 (a), D124 (a)): P3 replay.

The same PQ computation as the hold-out unit (pqh_inputs.pq_arrays, pq_io.compute_pair's order) on P3's 0438-0439: v rebuilt by the development package's pq_io.bind_field (plan.md's pins), the registered
v4 posteriors through pq_io.read_side, pq_v1/members' V (pq_io.members_record, read_variance); every array of its pq.npz and baselines.npz against pq_v1/runs/0438-0439's, bit for bit, after their files
are checked against pq_v1/outputs_manifest.txt, whose own sha256 is the released 86b1e298... (dd89633).  (iii) uses P3's v4 + C field on purpose (revision 7c): it checks the new entry's arithmetic against
the development reading, not the method's field (segment (ii) does that).  Dev mode only, once (DEV_P3_REPLAY_LOCK), under the spy with the dev mode's rules plus these files; it writes only
he_smoke_v3/pq/p3_replay/.  The reading: '逐位相同' or '不过（待查）' (C22.H8: no explanation, no rerun; a new root after a fix).  No truth, no evaluation record is read.
"""
from pathlib import Path
import sys
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pq_io as io_  # noqa: E402
import pq_pins as pins  # noqa: E402
import pqh_firewall as hfw  # noqa: E402
import pqh_inputs as hin  # noqa: E402
import pqh_pins as hp  # noqa: E402
import pqh_units as hu  # noqa: E402

P3_SHORT = '0438-0439'
P3_REPLAY_DIR = 'p3_replay'
P3_RELEASE_SHA256 = '86b1e298516bf31b2b79a4288f3aa2942a19ceb4eee4266d8c7bf37a12337b06'      # pq_v1/outputs_manifest.txt, released in dd89633 (runs/20261107-pq/outputs_manifest.sha256)
READ_PASS, READ_FAIL = '逐位相同', '不过（待查）'


def contexts(layout, taper):
    """(the pair's context, the 19 pairs' context): pq_io.Context on the development roots (plan.md's pins); the second only for the members record of the 20 subjects (pq_io.members_record)."""
    common = dict(repo=Path(layout.code), source_root=Path(hp.MAIN_REPO), results=Path(hp.RESULTS), images=Path(hp.DATA_IMAGES), taper=taper)
    return io_.Context(shorts=(P3_SHORT,), **common), io_.Context(**common)


def rules(layout, ctx, lock_path):
    r = hu.rules_for(layout, 'out')
    files = [ctx.t88_dir(P3_SHORT)/'run.json', ctx.t88_dir(P3_SHORT)/'field_0600.npz', ctx.repair_dir(P3_SHORT)/'repair.json', ctx.repair_dir(P3_SHORT)/'repair.npz',
             Path(hp.PQ_V1)/'outputs_manifest.txt', Path(hp.PQ_V1)/'members'/hin.MEMBERS_RECORD, lock_path]
    files += [Path(hp.PQ_V1)/'members'/s/io_.VAR_FILE for s in pins.subjects_of(P3_SHORT)]
    files += [Path(hp.PQ_V1)/'runs'/P3_SHORT/n for n in (hu.PQ_NPZ, hu.BASELINES)]
    r.allow_files(files, "smoke (iii): P3's 0438-0439 (its field and repair records), the development V and pq_v1's two arrays with their list (C22.H8 (iii))")
    return r


def compare(mine, path, keys, listed, rel):
    """Rows of one reference file: its sha256 as listed, then each array bit for bit (dtype, shape, bytes)."""
    rows = {}
    file_ok = listed.get(rel) == io_.sha_file(path)
    with np.load(path, allow_pickle=False) as z:
        for k in keys:
            theirs = np.array(z[k])
            same = bool(file_ok and theirs.dtype == mine[k].dtype and theirs.shape == mine[k].shape and theirs.tobytes() == np.ascontiguousarray(mine[k]).tobytes())
            rows[f'{path.name}:{k}'] = dict(equal=same, file_listed=file_ok, mine_sha256=io_.sha_array(mine[k]), theirs_sha256=io_.sha_array(theirs), dtype=str(mine[k].dtype))
    return rows


def p3_replay(layout, *, deps=None, release_sha256=P3_RELEASE_SHA256, lock_path=None, contexts_of=None):
    """Segment (iii) (module docstring).  Returns the exit code (0: 逐位相同 and the spy passed).  contexts_of(layout, taper): the tests' development contexts (default: contexts)."""
    deps = hu.Deps() if deps is None else deps
    log = deps.log
    if layout.mode != 'dev':
        raise ValueError('the P3 replay is a segment of the smoke (dev mode)')
    hu.preload('out')
    checks = {}
    problem = hfw.sealed_cache_problem('p3_replay')
    if problem:
        return hu.refuse(layout, 'p3_replay', None, checks, hu.Refusal('hm6', problem), log)
    if deps.taper is None:
        import mc0_core
        try:
            deps.taper = mc0_core.frozen_api().boundary_taper
        except SystemExit as exc:                   # m7: jd_curl not the frozen tree's; a refusal record, not an exit without one
            return hu.refuse(layout, 'p3_replay', None, checks, hu.Refusal('frozen', f'SystemExit: {exc}'), log)
    ctx, ctx_all = (contexts if contexts_of is None else contexts_of)(layout, deps.taper)
    lock_path = Path(hp.DEV_P3_REPLAY_LOCK if lock_path is None else lock_path)
    spy = hfw.spy(rules(layout, ctx, lock_path))
    spy.install()
    out_dir = Path(layout.pq_small)/P3_REPLAY_DIR
    lock = None
    try:
        try:
            checks['plan'] = io_.check_plan(layout.code)
            problem = hu.own_unit_problem(hp.DEV_UNITS['p3_replay'], deps)
            if problem:
                raise hu.Refusal('unit', problem)
            hu.code_check(layout, deps, None, checks)                 # I5: the smoke's checkout, clean; the code imported from it (m7)
            if out_dir.exists():
                raise hu.Refusal('once', f'{out_dir} exists')
            manifest = Path(hp.PQ_V1)/'outputs_manifest.txt'
            digest = io_.sha_file(manifest)
            if digest != release_sha256:
                raise hu.Refusal('reference', f'{manifest}: sha256 {digest} is not the released {release_sha256}')
            listed = io_.parse_manifest(manifest.read_text(encoding='utf-8'))
            fixed, moving = pins.subjects_of(P3_SHORT)
            record, allowlist = hu.hc().admit_dev_pair(fixed, moving)
            checks['admission'] = record
            hu.disk_and_machine(layout, 'out', deps, checks, unit_name=hp.DEV_UNITS['p3_replay'])     # SM3: the gate's own unit is the one this process runs in
            lock = hu.take_lock(dict(kind='first_run', lock=lock_path, authorization=None), dict(tool=hp.TOOL, unit='p3_replay', mode='dev', code_commit=checks['code'].get('commit'),
                                                                                                          code_head=checks['code'].get('head')))
        except (Exception, SystemExit) as exc:      # noqa: BLE001 - m7: SystemExit too
            return hu.refuse(layout, 'p3_replay', spy, checks, hu.as_refusal(exc), log)
        reading, rows, error = None, {}, None
        try:
            lookup = io_.load_lookup(ctx.repo, hp.SEG4_TABLE_SHA256)
            pin = io_.load_pair_checked(ctx, P3_SHORT, allowlist)
            v, field_record = io_.bind_field(ctx, P3_SHORT, pin.grid)
            layer = io_.structure_layer(ctx, pin)
            members = io_.members_record(ctx_all)
            shape = tuple(pin.grid.shape)
            V_F, V_M = (io_.read_variance(ctx_all, members, s, shape) for s in (pin.fixed_subject, pin.moving_subject))
            labels = field_record['labels_sha256']
            pq, base, _ = hin.pq_arrays(v=v, pin=pin, read_fixed=lambda: io_.read_side(ctx, layer, pin, 'fixed', labels), read_moving=lambda: io_.read_side(ctx, layer, pin, 'moving', labels),
                                        V_F=V_F, V_M=V_M, lookup=lookup)
            mine = dict(pq, **base)
            root = Path(hp.PQ_V1)/'runs'/P3_SHORT
            rows.update(compare(mine, root/hu.PQ_NPZ, ('q', 'q4'), listed, f'runs/{P3_SHORT}/{hu.PQ_NPZ}'))
            rows.update(compare(mine, root/hu.BASELINES, tuple(base), listed, f'runs/{P3_SHORT}/{hu.BASELINES}'))
            reading = READ_PASS if rows and all(r['equal'] for r in rows.values()) else READ_FAIL
        except (Exception, SystemExit) as exc:      # noqa: BLE001 - kept: the reading is 不过（待查）
            error, reading = f'{type(exc).__name__}: {exc}\n{traceback.format_exc()[-3000:]}', READ_FAIL
    finally:
        spy.remove()
    opened = spy.record()
    hu.make_dir(out_dir)
    hu.write_record(out_dir/'opened_files.json', dict(opened, mode='dev'), log)
    hu.write_record(out_dir/'p3_replay.json', dict(schema=hp.SCHEMA.format('p3_replay'), contract='C22.H8 (iii)', pair=P3_SHORT, reading=reading, rows=rows, error=error, lock=lock, checks=checks,
                                                           spy_m6=opened['m6'], host=hu.host(), utc=hu.utc(),
                                                           rule="the same PQ computation on P3's v4 + C field: every array of pq.npz and baselines.npz bit for bit pq_v1/runs/0438-0439's (their files as "
                                                                'listed in pq_v1/outputs_manifest.txt, released 86b1e298...)'))
    log(f"p3 replay: {reading}; spy m6 {opened['m6']}")
    return hu.EXIT['ok'] if reading == READ_PASS and opened['m6'] else hu.EXIT['stopped']
