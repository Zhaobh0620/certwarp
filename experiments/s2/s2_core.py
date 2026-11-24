"""S2 (runs/20261105-s2/任务书.md, plan.md): the arms, the guards and the field of the entry: a repaired field through the frozen native level.  Not F_B, not a method: a path experiment on development pairs.

The second taper (F = /home/rmcfz/seq_vp_tag_results/frozen_eb808be, eb808be):
  F/jd_curl/three_d/multiresolution.py:392-400   raw, record = preoptimizer(fixed, moving, grid); seed, _ = boundary_taper(raw, grid, 4): the frozen path tapers whatever it is given, once;
  F/jd_curl/three_d/preopt.py:617-624            nodes within 4 of a face are multiplied by w < 1; every other node by exactly 1.0;
  experiments/ef/run_ef.py:216, 262, 317-325     the repair started from the TAPERED field and stored (free nodes, their values); repair_metrics.json evaluates "the repaired field (no second taper)".
So a repaired field handed over as it is would have w^2 in the band, and what enters the native level would not be the field the certificate passed.  What is handed over is raw' = the raw field with only the repair's nodes replaced by their values.
If no repair node lies in the band, taper(raw') is the repaired field, bit for bit: w = 1.0 exactly at the repair nodes (1.0 * x == x for every float64), and everywhere else raw' = raw.  This module checks it by hash and refuses otherwise: a node in the band is never used quietly.

The wrapper is experiments/seed/inject_run.py, unchanged: raw' is an ordinary entry file (one member u, zero faces) under <root>/entries, so its checks and run_seed.injection_trace apply as they are, with tapered_sha256 = the repaired field's hash.
Only numpy and the records here: no jd_curl, no torch, no pair.npz, no label key.  The certificate of the check is run_s2.py taper-check.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT/'experiments/seed'))
sys.path.insert(2, str(ROOT/'experiments/tc'))

import numpy as np  # noqa: E402

import run_seed as core  # noqa: E402
import run_seed19 as s19  # noqa: E402

ROOT_NAME = 's2_v1'
EF_ROOT, FL_ROOT = 'ef_v1', 'fl_v1'
TAPER = core.TAPER
PAIRS = core.PAIRS
EXCEPTION_PAIR = 'OASIS_OAS1_0455_MR1__OASIS_OAS1_0456_MR1'                                     # of the 14 other pairs: only in S_Pp (the 2 pairs of S_P with alpha .9)
S_PP_PAIRS = (PAIRS[1], EXCEPTION_PAIR)
HOLD_S_P = core.ENTRY_CHARGE_SECONDS                                                            # 47 s, as S_P: S_Pp and S_F differ from S_P only in the injected field
EPS = .05
SOURCE_SCHEMA, REPAIR_SCHEMA, REPAIR_METRICS_SCHEMA = 'bayesreg.mc0p.run.v1', 'bayesreg.ef.repair.v1', 'bayesreg.ef.repair_metrics.v1'
# The one place each source field is (experiments/ef/ef_core.py SOURCES; a test compares them); read only.
SOURCES = {('fine', PAIRS[0]): 'mc0p_v1', ('fine', PAIRS[1]): 'mc0p_v1', ('fine', PAIRS[2]): 'mc0p_v1', ('fine', PAIRS[3]): 'mc0p_v1b', ('fine', PAIRS[4]): 'mc0p_v1b',
           ('long', PAIRS[1]): 'mc0p_v1', ('long', EXCEPTION_PAIR): 'mc0p_v1c'}
# decides: enters the reading (decide_s2.py).  S_X has no fixed source: see sx_rows.
ARMS = dict(S_Pp=dict(pairs=S_PP_PAIRS, ef_arm='RL05', tier='long', decides=False, role='只描述：长档修复场（RL05），只差 α；也是注入路径的走通（走通的是路径，不是 S_F 的 J 区间）'),
            S_F=dict(pairs=PAIRS, ef_arm='RF05', tier='fine', decides=True, role='判定臂：细档 F600 + 修复（RF05，ε = .05）'),
            S_X=dict(pairs=PAIRS, ef_arm=None, tier=None, decides=False, role='只描述：FL 最好的臂的修复场；持时 = seconds_task + seconds_repair 的中位，不含 warm-up，预算总数不变'))
QUEUE_ARMS = ('S_Pp', 'S_F')                                                                    # S_X only with its explicit options
SX_ARMS = ('F', 'S', 'T', 'H', 'Hx', 'I', 'X')                                                  # FL's arms; O is oracle and stays in its quarantine
SX_K = dict(F=35, S=35, T=35, H=35, Hx=35, I=36, X=35)                                          # experiments/fl/fl_core.py ARMS
SX_CHANNELS = ('onehot', 'onehot+intensity')                                                    # never 'truth'
SX_CHECKPOINTS = (2400, 4800)                                                                   # experiments/fl/fl_core.py REPAIR_STEPS: FL repairs only these
SX_MAX_HOLD = 200.                                                                              # the lead, plan revision 1: above it the native level does not reach t about 126 s; so X at 4800 steps is no entry of S_X
FL_ORACLE_DIR, FL_FIRST = 'oracle_isolated', 'first5'                                           # experiments/fl/fl_core.py ORACLE_DIR, FIRST
FL_RUN_SCHEMA, FL_REPAIR_SCHEMA, FL_REPAIR_METRICS_SCHEMA = 'bayesreg.fl.run.v1', 'bayesreg.fl.repair.v1', 'bayesreg.fl.repair_metrics.v1'
SX_ROW_KEYS = ('raw_field', 'raw_field_sha256', 'repair_npz', 'repair_npz_sha256', 'run_json_sha256', 'repair_json_sha256', 'repair_metrics_json_sha256', 'tapered_u_sha256', 'repaired_u_sha256', 'source_run_id')
EntryError, EntryMissing = core.EntryError, core.EntryMissing


class BandError(EntryError):
    """A repair node lies in the taper's band: the frozen path would change it.  Stop and report; never used.  record: the counts."""

    def __init__(self, text, record):
        super().__init__(text)
        self.record = record


def short_name(pair):
    return core.short_name(pair)


def require_cell(arm, pair):
    """Only the cells of the plan: checked before any file of the pair is opened."""
    if arm not in ARMS:
        raise SystemExit(f'S2 has no arm {arm}; only {tuple(ARMS)}')
    if pair not in ARMS[arm]['pairs']:
        raise SystemExit(f"S2 runs {arm} only on {[short_name(p) for p in ARMS[arm]['pairs']]}; refused: {pair}")
    return arm, pair


def plan_runs(with_x=False):
    """S_Pp, then S_F; S_X only when asked for: (arm, pair)."""
    return [(arm, pair) for arm in QUEUE_ARMS+(('S_X',) if with_x else ()) for pair in ARMS[arm]['pairs']]


def default_root(results=None):
    return Path(core.RESULTS if results is None else results)/ROOT_NAME


def root_problem(root, results=None):
    """Why this batch may not write at this root, or None: only its own."""
    return None if Path(root).resolve() == default_root(results).resolve() else f'{root}: this batch writes only to {default_root(results)}; nothing read, nothing moved'


def run_dir(root, arm, pair):
    require_cell(arm, pair)
    return Path(root)/'runs'/arm/pair


def own_run_problem(run, arm, pair, results=None):
    """Why this directory is not this batch's run of this arm and pair, or None: evaluate-one and curve-one write only there."""
    want = run_dir(default_root(results), arm, pair)
    return None if Path(run).resolve() == want.resolve() else f'{run}: this step writes only to {want}; nothing read, nothing moved'


def unit_name(arm, pair):
    require_cell(arm, pair)
    return f'tag-s2-{arm}-{short_name(pair)}'


def stage_name(arm):
    return f'S2-{arm}'


def reference_dir(pair, results=None):
    """S_P of the same pair: only read.  seed_v1 the first 5 pairs, seed_v1b 0455-0456."""
    name = 'seed_v1' if pair in PAIRS else s19.ROOT_NAME
    return Path(core.RESULTS if results is None else results)/name/'runs/S_P'/pair


def input_files(pair, results=None):
    """A5P's inputs of this pair, against the inputs record: every arm has them, as S_P."""
    return core.input_files('S_P', pair, results) if pair in PAIRS else s19.input_files('S_P', pair, results)


# ---- the taper and raw' --------------------------------------------------------------------------------------------------------------------------------------------------------------
def taper_weights(shape, width=TAPER):
    """w of F/jd_curl/three_d/preopt.py boundary_taper, the same arithmetic (a test compares the bytes with the frozen function)."""
    rs = []
    for n in shape:
        k = np.arange(n)
        rs.append(np.minimum(np.minimum(k, n-1-k)/float(width), 1.))
    return rs[0][:, None, None]*rs[1][None, :, None]*rs[2][None, None, :]


def tapered(u, width=TAPER):
    return taper_weights(u.shape[:3], width)[..., None]*np.asarray(u, dtype=np.float64)


def array_sha(u):
    """As the certificate's state_hash and run_seed.boundary_taper_sha256: the bytes of a float64 C-ordered array."""
    return hashlib.sha256(np.ascontiguousarray(u, dtype=np.float64).view(np.uint8)).hexdigest()


def face_distance(nodes, shape):
    """The distance in nodes of each global node id to the nearest face of the grid."""
    z, y, x = np.unravel_index(nodes, shape)
    return np.minimum.reduce([np.minimum(c, n-1-c) for c, n in zip((z, y, x), shape)])


def read_raw(path, sha256, shape):
    """The raw field: file sha256, one member u, float64, this shape, finite, zero faces.  Read only.  EntryError otherwise."""
    path = Path(path)
    if not path.is_file():
        raise EntryMissing(f'{path}: no such file')
    got = core.sha_file(path)
    if got != sha256:
        raise EntryError(f'{path}: sha256 {got} != {sha256}')
    names, stored, dtype = core.npz_header(path)
    if names != ['u.npy'] or stored != tuple(shape)+(3,) or dtype not in ('<f8', 'float64'):
        raise EntryError(f'{path}: members {names}, shape {stored}, dtype {dtype}; expected only u, {tuple(shape)+(3,)}, float64')
    with np.load(path) as z:
        raw = np.array(z['u'])
    if not np.all(np.isfinite(raw)):
        raise EntryError(f'{path}: not finite')
    if any(np.any(side) for side in (raw[0], raw[-1], raw[:, 0], raw[:, -1], raw[:, :, 0], raw[:, :, -1])):
        raise EntryError(f'{path}: the faces are not zero')
    raw.setflags(write=False)
    return raw


def read_repair(path, sha256, shape):
    """(nodes, values) of a repair.npz (experiments/ef/run_ef.py:262): file sha256, exactly these members, sorted distinct node ids of this grid, (n, 3) finite float64.  No node at all is allowed: repair_field on a field that already passes stores two empty arrays (repair.py, already_certified), and raw' is then raw.  EntryError otherwise."""
    path = Path(path)
    if not path.is_file():
        raise EntryMissing(f'{path}: no such file')
    got = core.sha_file(path)
    if got != sha256:
        raise EntryError(f'{path}: sha256 {got} != {sha256}')
    with zipfile.ZipFile(path) as z:
        names = sorted(z.namelist())
    if names != ['nodes.npy', 'values.npy']:
        raise EntryError(f'{path}: members {names}, expected nodes, values')
    with np.load(path) as z:
        nodes, values = np.array(z['nodes']), np.array(z['values'])
    n_nodes = int(np.prod(shape))
    if nodes.ndim != 1 or not np.issubdtype(nodes.dtype, np.integer) or values.dtype != np.float64 or values.shape != (len(nodes), 3):
        raise EntryError(f'{path}: nodes {nodes.dtype} {nodes.shape}, values {values.dtype} {values.shape}')
    if len(nodes) and (nodes[0] < 0 or nodes[-1] >= n_nodes or np.any(np.diff(nodes) <= 0)):
        raise EntryError(f'{path}: the node ids are not sorted, distinct ids of a grid of {n_nodes} nodes')
    if not np.all(np.isfinite(values)):
        raise EntryError(f'{path}: values not finite')
    return nodes.astype(np.int64), values


def raw_prime(raw, nodes, values, *, tapered_sha256, repaired_sha256, width=TAPER, label=''):
    """(raw', record): raw with only the repair's nodes replaced by their values, after checking, on the arrays:
      taper(raw) is the field the repair started from (hash);
      that field with the repair's nodes is the repaired field (hash);
      no repair node is in the band: BandError otherwise, with the counts: what is moved in the band says whether anything would really change;
      taper(raw') is the repaired field, bit for bit (hash and bytes).
    No repair node at all: raw' is raw, recorded as empty_repair.  The record says the counts, never the arrays."""
    shape = raw.shape[:3]
    w = taper_weights(shape, width)
    start = w[..., None]*raw
    got = array_sha(start)
    if got != tapered_sha256:
        raise EntryError(f'{label}: the taper of the raw field is {got}, the repair started from {tapered_sha256}')
    repaired = start.copy()
    before = repaired.reshape(-1, 3)[nodes].copy()
    repaired.reshape(-1, 3)[nodes] = values
    got = array_sha(repaired)
    if got != repaired_sha256:
        raise EntryError(f'{label}: the tapered field with the repair nodes is {got}, the repaired field is {repaired_sha256}')
    changed = np.any(before != values, axis=1)
    band = w.reshape(-1)[nodes] < 1.
    distance = face_distance(nodes, shape)
    record = dict(n_repair_nodes=len(nodes), n_changed_nodes=int(changed.sum()), n_nodes_in_band=int(band.sum()), n_changed_nodes_in_band=int((band & changed).sum()), min_face_distance=int(distance.min()) if len(nodes) else None, empty_repair=not len(nodes), taper_width=width)
    if record['n_nodes_in_band']:
        raise BandError(f"{label}: {record['n_nodes_in_band']} repair nodes in the taper's band ({record['n_changed_nodes_in_band']} of them moved; nearest a face at {record['min_face_distance']}, the band is < {width}): the frozen path would multiply them by w < 1; stopped, nothing injected", record)
    prime = np.array(raw, copy=True)
    prime.reshape(-1, 3)[nodes] = values
    again = w[..., None]*prime
    got = array_sha(again)
    if got != repaired_sha256 or again.tobytes() != repaired.tobytes():
        raise EntryError(f"{label}: taper(raw') is {got}, the repaired field is {repaired_sha256}")
    prime.setflags(write=False)
    return prime, dict(record, raw_prime_sha256=array_sha(prime), taper_of_raw_prime_sha256=got, bitwise_equal_to_repaired=True)


# ---- the fixed sources: ef_v1's repairs --------------------------------------------------------------------------------------------------------------------------------------------------
def repair_dir(arm, pair, results=None):
    """The one place a repair of this arm and pair may come from."""
    require_cell(arm, pair)
    spec = ARMS[arm]
    return Path(core.RESULTS if results is None else results)/EF_ROOT/'runs'/f"{spec['ef_arm']}_{spec['tier']}"/short_name(pair)


def source_dir(arm, pair, results=None):
    """The one place the raw field of this arm and pair may come from."""
    require_cell(arm, pair)
    tier = ARMS[arm]['tier']
    return Path(core.RESULTS if results is None else results)/SOURCES[(tier, pair)]/'runs'/f'Q2_{tier}'/short_name(pair)


def ef_problems(arm, pair, record, metrics, run, manifest, results=None):
    """What is wrong with the records of a repair of ef_v1 and its source for this cell: the cell's pair, arm, tier, eps, certified at alpha = 1 by the evaluation, oracle false, the source of the plan, the same original images as the arm's inputs."""
    spec = ARMS[arm]
    short = short_name(pair)
    run_id = f"ef_{spec['ef_arm']}_{spec['tier']}_{short}"
    problems = []
    for name, have, want in (('schema', record.get('schema'), REPAIR_SCHEMA), ('run_id', record.get('run_id'), run_id), ('arm', record.get('arm'), spec['ef_arm']), ('tier', record.get('tier'), spec['tier']), ('pair', record.get('pair'), pair), ('oracle', record.get('oracle'), False),
                             ('status', record.get('status'), 'certified'), ('passed', record.get('passed'), True), ('eps', record.get('eps'), EPS)):
        if have != want:
            problems.append(f'repair.json {name} {have!r}, expected {want!r}')
    source = record.get('source') or {}
    where = source_dir(arm, pair, results)
    if str(source.get('directory')) != str(where):
        problems.append(f"repair.json source {source.get('directory')}: this cell's field may only come from {where}")
    certificate = record.get('certificate') or {}
    if certificate.get('status') != 'certified_feasible' or not (certificate.get('min_jd_lower_bound') or 0) > 0 or certificate.get('n_infeasible') != 0 or certificate.get('n_uncertain') != 0:
        problems.append(f'repair.json certificate {certificate}: did not pass')
    for name, have, want in (('schema', metrics.get('schema'), REPAIR_METRICS_SCHEMA), ('run_id', metrics.get('run_id'), run_id), ('pair', metrics.get('pair'), pair), ('oracle', metrics.get('oracle'), False), ('fallback', metrics.get('fallback'), None), ('repaired hash', metrics.get('repaired_u_sha256'), certificate.get('state_hash'))):
        if have != want:
            problems.append(f'repair_metrics.json {name} {have!r}, expected {want!r}')
    ladder = (metrics.get('after') or {}).get('certificate') or {}
    if ladder.get('accepted_alpha') != 1. or ladder.get('status_at_1') != 'certified_feasible' or ladder.get('n_infeasible_at_1') != 0 or ladder.get('n_uncertain_at_1') != 0:
        problems.append(f"repair_metrics.json: the evaluation's certificate at alpha = 1: {ladder.get('accepted_alpha')}, {ladder.get('status_at_1')}, {ladder.get('n_infeasible_at_1')}, {ladder.get('n_uncertain_at_1')}")
    for name, have, want in (('schema', run.get('schema'), SOURCE_SCHEMA), ('run_id', run.get('run_id'), f"mc0p_Q2_{spec['tier']}_{short}"), ('pair', run.get('pair'), pair), ('arm', run.get('arm'), 'Q2'), ('tier', run.get('tier'), spec['tier']), ('k', run.get('k'), 35), ('oracle', run.get('oracle'), False), ('status', run.get('status'), 'done')):
        if have != want:
            problems.append(f'source run.json {name} {have!r}, expected {want!r}')
    problems += [f'source run.json: {p}' for p in core.channel_sources_problems(run)]
    field = run.get('field') or {}
    if field.get('sha256') != source.get('field_sha256'):
        problems.append(f"source run.json field {field.get('sha256')}, the repair's source {source.get('field_sha256')}")
    made, data = manifest.get('preprocessing_d0') or {}, run.get('data') or {}
    if data.get('pair_sha256') != made.get('source_pair_sha256') or data.get('manifest_sha256') != made.get('source_manifest_sha256'):
        problems.append(f"the field comes from original pair {data.get('pair_sha256')}, the inputs from {made.get('source_pair_sha256')}")
    if data.get('fixed_subject') != manifest.get('fixed_subject') or data.get('moving_subject') != manifest.get('moving_subject'):
        problems.append(f"the field's subjects {data.get('fixed_subject')}, {data.get('moving_subject')}; the inputs' {manifest.get('fixed_subject')}, {manifest.get('moving_subject')}")
    return problems


def ef_source(arm, pair, manifest, results=None):
    """The record of one fixed cell: the files to read and what they must hash to, or EntryError (EntryMissing if a file is not there)."""
    d, s = repair_dir(arm, pair, results), source_dir(arm, pair, results)
    record, metrics, run = core.read_json(d/'repair.json'), core.read_json(d/'repair_metrics.json'), core.read_json(s/'run.json')
    if record is None or metrics is None or run is None:
        raise EntryMissing(f'{arm} {short_name(pair)}: no repair.json or repair_metrics.json in {d}, or no run.json in {s}')
    problems = ef_problems(arm, pair, record, metrics, run, manifest, results)
    if problems:
        raise EntryError(f'{arm} {short_name(pair)}: '+'; '.join(problems))
    source = record['source']
    return dict(raw_path=s/'field.npz', raw_sha256=source['field_sha256'], repair_path=d/'repair.npz', repair_sha256=record['attempt']['sha256'], tapered_sha256=source['tapered_u_sha256'], repaired_sha256=metrics['repaired_u_sha256'], source_run_id=run['run_id'], oracle=run['oracle'], source_root=SOURCES[(ARMS[arm]['tier'], pair)], source_k=run['k'], source_tier=run['tier'], source_arm=run['arm'],
                charge_seconds=HOLD_S_P, seconds_entry=run.get('seconds_total'), seconds_repair=record.get('seconds_repair'), seconds_warmup=record.get('seconds_warmup'), hold=None, entry_metrics=entry_metrics_of(metrics), repair_dir=str(d), repair_eps=record['eps'], evaluation_lower_bound=record['certificate'].get('min_jd_lower_bound'))


def entry_metrics_of(metrics):
    x = (metrics.get('after') or {}).get('metrics') or {}
    return dict(seg35=x.get('seg35'), seg4=x.get('seg4'), sdlogj_official=x.get('sdlogj_official'), accepted_alpha=((metrics.get('after') or {}).get('certificate') or {}).get('accepted_alpha'))


# ---- S_X: FL's best arm, named on the command line, pinned before the start -----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SXOptions:
    root: str
    arm: str
    checkpoint: int
    pins: str
    pins_sha256: str


def sx_options(root=None, arm=None, checkpoint=None, pins=None, pins_sha256=None):
    """SXOptions when all five are given, None when none is; anything else: refused."""
    given = (root, arm, checkpoint, pins, pins_sha256)
    if all(v is None for v in given):
        return None
    if any(v is None for v in given):
        raise SystemExit('S_X needs all of --sx-root --sx-arm --sx-checkpoint --sx-pins --sx-pins-sha256, or none of them')
    return SXOptions(str(root), str(arm), int(checkpoint), str(pins), str(pins_sha256))


def fl_root(results=None):
    return Path(core.RESULTS if results is None else results)/FL_ROOT


def fl_field_dir(root, arm, pair):
    """experiments/fl/fl_core.py run_dir of a field run of the first 5 pairs; O is under oracle_isolated and never here."""
    return Path(root)/'runs'/f'{arm}_{FL_FIRST}'/short_name(pair)


def fl_repair_dir(root, arm, step, pair):
    """experiments/fl/fl_core.py run_dir of Cell('R'+arm, 'c<step>', pair)."""
    return Path(root)/'runs'/f'R{arm}_c{step}'/short_name(pair)


def sx_guard(arm, step):
    if arm not in SX_ARMS:
        raise EntryError(f'S_X: --sx-arm {arm}: only {SX_ARMS} (O is oracle and stays in its quarantine)')
    if step not in SX_CHECKPOINTS:
        raise EntryError(f'S_X: --sx-checkpoint {step}: only {SX_CHECKPOINTS}, the checkpoints FL repairs')


def sx_pins(root, arm, step):
    """The pins file's document, read from FL's records of this arm and checkpoint: nothing is written by hand, and nothing is judged here (sx_rows does that, on the file).  EntryMissing if a record is not there."""
    sx_guard(arm, step)
    rows = {}
    for pair in PAIRS:
        field, d = fl_field_dir(root, arm, pair), fl_repair_dir(root, arm, step, pair)
        paths = dict(run_json_sha256=field/'run.json', repair_json_sha256=d/'repair.json', repair_metrics_json_sha256=d/'repair_metrics.json')
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise EntryMissing(f'S_X {short_name(pair)}: no {missing}')
        run, repair, metrics = core.read_json(paths['run_json_sha256']), core.read_json(paths['repair_json_sha256']), core.read_json(paths['repair_metrics_json_sha256'])
        checkpoint = (run.get('checkpoints') or {}).get(str(step)) or {}
        rows[short_name(pair)] = dict(raw_field=str(field/str(checkpoint.get('file'))), raw_field_sha256=checkpoint.get('sha256'), repair_npz=str(d/'repair.npz'), repair_npz_sha256=(repair.get('attempt') or {}).get('sha256'), **{key: core.sha_file(path) for key, path in paths.items()},
                                      tapered_u_sha256=(repair.get('source') or {}).get('tapered_u_sha256'), repaired_u_sha256=metrics.get('repaired_u_sha256'), source_run_id=run.get('run_id'))
    return dict(root=str(root), arm=arm, checkpoint=step, oracle=False, pairs=rows)


def fl_problems(arm, step, pair, row, root, run, repair, metrics):
    """What is wrong with FL's three records of this cell against the pins and each other (the S_X counterpart of ef_problems): oracle false in each, the channels really predicted, this arm, checkpoint and pair, certified at alpha = 1 by the evaluation, the same files.  The records' format: experiments/fl/run_fl.py run_field, repair_one, evaluate_repair."""
    short = short_name(pair)
    problems = []

    def expect(where, items):
        problems.extend(f'{where} {name} {have!r}, expected {want!r}' for name, have, want in items if have != want)
    checkpoint = (run.get('checkpoints') or {}).get(str(step)) or {}
    expect('run.json', (('schema', run.get('schema'), FL_RUN_SCHEMA), ('run_id', run.get('run_id'), f'fl_{arm}_{FL_FIRST}_{short}'), ('arm', run.get('arm'), arm), ('tier', run.get('tier'), FL_FIRST), ('pair', run.get('pair'), pair), ('k', run.get('k'), SX_K[arm]), ('oracle', run.get('oracle'), False), ('status', run.get('status'), 'done'), ('source_run_id of the pins', row['source_run_id'], run.get('run_id')),
                        ('the checkpoint\'s sha256', checkpoint.get('sha256'), row['raw_field_sha256']), ('the checkpoint\'s file', checkpoint.get('file'), Path(row['raw_field']).name)))
    if run.get('channels') not in SX_CHANNELS:
        problems.append(f"run.json channels {run.get('channels')!r}: only {SX_CHANNELS}")
    problems += [f'run.json: {p}' for p in core.channel_sources_problems(run)]
    source, certificate, attempt = repair.get('source') or {}, repair.get('certificate') or {}, repair.get('attempt') or {}
    expect('repair.json', (('schema', repair.get('schema'), FL_REPAIR_SCHEMA), ('run_id', repair.get('run_id'), f'fl_R{arm}_c{step}_{short}'), ('arm', repair.get('arm'), f'R{arm}'), ('tier', repair.get('tier'), f'c{step}'), ('pair', repair.get('pair'), pair), ('oracle', repair.get('oracle'), False), ('status', repair.get('status'), 'certified'), ('passed', repair.get('passed'), True), ('eps', repair.get('eps'), EPS),
                           ('source directory', source.get('directory'), str(fl_field_dir(root, arm, pair))), ('source file', source.get('file'), Path(row['raw_field']).name), ('source field_sha256', source.get('field_sha256'), row['raw_field_sha256']), ('source tapered_u_sha256', source.get('tapered_u_sha256'), row['tapered_u_sha256']), ('source seconds_task', source.get('seconds_task'), checkpoint.get('seconds_task')),
                           ('attempt sha256', attempt.get('sha256'), row['repair_npz_sha256']), ('certificate state_hash', certificate.get('state_hash'), row['repaired_u_sha256'])))
    if certificate.get('status') != 'certified_feasible' or not (certificate.get('min_jd_lower_bound') or 0) > 0 or certificate.get('n_infeasible') != 0 or certificate.get('n_uncertain') != 0:
        problems.append(f'repair.json certificate {certificate}: did not pass')
    expect('repair_metrics.json', (('schema', metrics.get('schema'), FL_REPAIR_METRICS_SCHEMA), ('run_id', metrics.get('run_id'), f'fl_R{arm}_c{step}_{short}'), ('pair', metrics.get('pair'), pair), ('oracle', metrics.get('oracle'), False), ('repaired_u_sha256', metrics.get('repaired_u_sha256'), row['repaired_u_sha256'])))
    ladder = (metrics.get('after') or {}).get('certificate') or {}
    if ladder.get('accepted_alpha') != 1. or ladder.get('status_at_1') != 'certified_feasible' or ladder.get('n_infeasible_at_1') != 0 or ladder.get('n_uncertain_at_1') != 0:
        problems.append(f"repair_metrics.json: the evaluation's certificate at alpha = 1: {ladder.get('accepted_alpha')}, {ladder.get('status_at_1')}, {ladder.get('n_infeasible_at_1')}, {ladder.get('n_uncertain_at_1')}")
    for name, value in (('seconds_task', checkpoint.get('seconds_task')), ('seconds_repair', repair.get('seconds_repair')), ('seconds_warmup', repair.get('seconds_warmup'))):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            problems.append(f'{name} {value!r}: not a number of seconds')
    return problems


def sx_rows(options, results=None):
    """{pair: dict(row, run, repair, metrics, seconds_task, seconds_repair, seconds_warmup)}, after the checks that need no field.  The pins: the file's sha256 as pinned on the command line, the arm (not O), the checkpoint, the root is fl_v1, the same in the file, all 5 pairs and no other.
    The paths: no segment oracle_isolated, inside the root (nothing under a link that leaves it), exactly FL's directories of this arm and checkpoint: a path to another arm's files is refused whatever the pins say.  The records: FL's run.json, repair.json, repair_metrics.json with the pinned sha256, against fl_problems.  EntryError otherwise."""
    root = fl_root(results)
    if Path(options.root).resolve() != root.resolve():
        raise EntryError(f'S_X: --sx-root {options.root}: its entry may only come from {root}')
    sx_guard(options.arm, options.checkpoint)
    pins = Path(options.pins)
    if not pins.is_file():
        raise EntryMissing(f'S_X: {pins}: no such file')
    got = core.sha_file(pins)
    if got != options.pins_sha256:
        raise EntryError(f'S_X: {pins}: sha256 {got} != --sx-pins-sha256 {options.pins_sha256}')
    doc = json.loads(pins.read_text())
    for name, have, want in (('root', doc.get('root'), str(root)), ('arm', doc.get('arm'), options.arm), ('checkpoint', doc.get('checkpoint'), options.checkpoint), ('oracle', doc.get('oracle'), False)):
        if have != want:
            raise EntryError(f'S_X: {pins}: {name} {have!r}, expected {want!r}')
    rows = doc.get('pairs')
    if not isinstance(rows, dict) or set(rows) != {short_name(p) for p in PAIRS}:
        raise EntryError(f'S_X: {pins}: pairs {sorted(rows) if isinstance(rows, dict) else rows}, expected exactly the first 5')
    out = {}
    for pair in PAIRS:
        row = rows[short_name(pair)]
        missing = [k for k in SX_ROW_KEYS if k not in row]
        if missing:
            raise EntryError(f'S_X {short_name(pair)}: {pins}: missing {missing}')
        field, d = fl_field_dir(root, options.arm, pair), fl_repair_dir(root, options.arm, options.checkpoint, pair)
        for key, want in (('raw_field', field), ('repair_npz', d)):
            where = Path(row[key])
            if FL_ORACLE_DIR in where.parts or FL_ORACLE_DIR in where.resolve().parts:
                raise EntryError(f'S_X {short_name(pair)}: {key} {row[key]}: {FL_ORACLE_DIR} is the oracle arm\'s, it never enters')
            if root.resolve() not in where.resolve().parents:
                raise EntryError(f'S_X {short_name(pair)}: {key} {row[key]}: its entry may only come from {root}')
            if where.parent != want or where.resolve().parent != want.resolve():
                raise EntryError(f'S_X {short_name(pair)}: {key} {row[key]}: --sx-arm {options.arm} at {options.checkpoint} steps reads only from {want}')
        if Path(row['repair_npz']).name != 'repair.npz':
            raise EntryError(f"S_X {short_name(pair)}: repair_npz {row['repair_npz']}: expected repair.npz")
        records = {}
        for name, path, key in (('run', field/'run.json', 'run_json_sha256'), ('repair', d/'repair.json', 'repair_json_sha256'), ('metrics', d/'repair_metrics.json', 'repair_metrics_json_sha256')):
            if not path.is_file():
                raise EntryMissing(f'S_X {short_name(pair)}: {path}: no such file')
            got = core.sha_file(path)
            if got != row[key]:
                raise EntryError(f'S_X {short_name(pair)}: {path}: sha256 {got} != the pins {row[key]}')
            records[name] = core.read_json(path)
        problems = fl_problems(options.arm, options.checkpoint, pair, row, root, records['run'], records['repair'], records['metrics'])
        if problems:
            raise EntryError(f'S_X {short_name(pair)}: '+'; '.join(problems))
        seconds_task = records['run']['checkpoints'][str(options.checkpoint)]['seconds_task']
        out[pair] = dict(row=row, **records, seconds_task=seconds_task, seconds_repair=records['repair']['seconds_repair'], seconds_warmup=records['repair']['seconds_warmup'])
    return out


def sx_hold(rows):
    """The hold of S_X and its account, from FL's records (plan revision 1): max(47, the median over the 5 pairs of seconds_task + seconds_repair, rounded up): no warm-up, as S_F's 57-70 s; the median with the warm-up is reported, it sets nothing.  The budget's total stays.  EntryError above SX_MAX_HOLD."""
    without = {short_name(pair): rec['seconds_task']+rec['seconds_repair'] for pair, rec in rows.items()}
    with_warmup = {short_name(pair): rec['seconds_task']+rec['seconds_repair']+rec['seconds_warmup'] for pair, rec in rows.items()}
    median = statistics.median(without.values())
    hold = max(HOLD_S_P, float(math.ceil(median)))
    if hold > SX_MAX_HOLD:
        raise EntryError(f'S_X: hold {hold} s (the median of seconds_task + seconds_repair is {median:.1f} s): more than {SX_MAX_HOLD} s, the native level would not reach t about 126 s; refused')
    return dict(seconds=hold, rule='max(47, ceil(median(seconds_task + seconds_repair))), no warm-up', median_without_warmup=median, median_with_warmup=statistics.median(with_warmup.values()), without_warmup=without, with_warmup=with_warmup)


def sx_source(pair, rows, options, manifest):
    """The source of one S_X cell.  The original images as the arm's inputs; the oracle mark from the record just checked."""
    rec, hold = rows[pair], sx_hold(rows)
    row, run, repair, metrics = rec['row'], rec['run'], rec['repair'], rec['metrics']
    made, problems = manifest.get('preprocessing_d0') or {}, []
    for data in (run.get('data') or {}, repair.get('data') or {}):
        if data.get('pair_sha256') != made.get('source_pair_sha256') or data.get('manifest_sha256') != made.get('source_manifest_sha256'):
            problems.append(f"the field comes from original pair {data.get('pair_sha256')}, the inputs from {made.get('source_pair_sha256')}")
        if data.get('fixed_subject') != manifest.get('fixed_subject') or data.get('moving_subject') != manifest.get('moving_subject'):
            problems.append(f"the field's subjects {data.get('fixed_subject')}, {data.get('moving_subject')}; the inputs' {manifest.get('fixed_subject')}, {manifest.get('moving_subject')}")
    if problems:
        raise EntryError(f'S_X {short_name(pair)}: '+'; '.join(problems))
    return dict(raw_path=Path(row['raw_field']), raw_sha256=row['raw_field_sha256'], repair_path=Path(row['repair_npz']), repair_sha256=row['repair_npz_sha256'], tapered_sha256=row['tapered_u_sha256'], repaired_sha256=row['repaired_u_sha256'], source_run_id=row['source_run_id'], oracle=run['oracle'], source_root=FL_ROOT, source_k=run['k'], source_tier=f'{options.arm}@{options.checkpoint}', source_arm=options.arm,
                charge_seconds=hold['seconds'], seconds_entry=rec['seconds_task'], seconds_repair=rec['seconds_repair'], seconds_warmup=rec['seconds_warmup'], hold=hold, entry_metrics=entry_metrics_of(metrics), repair_dir=str(Path(row['repair_npz']).parent), repair_eps=repair['eps'], evaluation_lower_bound=repair['certificate'].get('min_jd_lower_bound'))


def cell_source(arm, pair, manifest, results=None, sx=None):
    """The source of one cell: fixed for S_Pp, S_F; from the pins for S_X, which needs its options."""
    require_cell(arm, pair)
    if arm != 'S_X':
        return ef_source(arm, pair, manifest, results)
    if sx is None:
        raise EntryError('S_X needs --sx-root --sx-arm --sx-checkpoint --sx-pins --sx-pins-sha256')
    return sx_source(pair, sx_rows(sx, results), sx, manifest)


def construct(arm, pair, manifest, results=None, sx=None):
    """(raw', source, record) of one cell: only the raw field and the repair's two arrays are read.  EntryError, BandError otherwise."""
    source = cell_source(arm, pair, manifest, results, sx)
    shape = tuple(manifest['shape_zyx'])
    raw = read_raw(source['raw_path'], source['raw_sha256'], shape)
    nodes, values = read_repair(source['repair_path'], source['repair_sha256'], shape)
    prime, record = raw_prime(raw, nodes, values, tapered_sha256=source['tapered_sha256'], repaired_sha256=source['repaired_sha256'], label=f'{arm} {short_name(pair)}')
    return prime, source, record


# ---- the entry file: raw' under <root>/entries, once ----------------------------------------------------------------------------------------------------------------------------------------------
def entry_paths(root, arm, pair):
    require_cell(arm, pair)
    d = Path(root)/'entries'/arm/short_name(pair)
    return d/'field.npz', d/'entry.json'


def entry_record(arm, pair, field_path, prime, source, record, manifest):
    """The record for seed_entry.json: inject_run.ENTRY_KEYS, and what run_seed.injection_trace and this batch read."""
    return dict(arm=arm, pair=pair, field_path=str(field_path), field_sha256=core.sha_file(field_path), u_sha256=record['raw_prime_sha256'], source_run_id=source['source_run_id'], oracle=source['oracle'], charge_seconds=float(source['charge_seconds']), shape_zyx=list(manifest['shape_zyx']),
                tapered_sha256=record['taper_of_raw_prime_sha256'], repaired_u_sha256=source['repaired_sha256'], tapered_recorded_by_evaluation=source['repaired_sha256'], source_raw_field=str(source['raw_path']), source_raw_field_sha256=source['raw_sha256'], source_tapered_u_sha256=source['tapered_sha256'],
                repair_npz=str(source['repair_path']), repair_npz_sha256=source['repair_sha256'], repair_dir=source['repair_dir'], repair_eps=source['repair_eps'], construction=record, source_seconds_total=source['seconds_entry'], seconds_repair=source['seconds_repair'], seconds_warmup=source['seconds_warmup'], hold=source['hold'], evaluation_lower_bound=source['evaluation_lower_bound'], source_root=source['source_root'], source_k=source['source_k'], source_tier=source['source_tier'], source_arm=source['source_arm'],
                retry_of=None, role=core.inject_run.ROLE, entry_metrics=source['entry_metrics'])


def write_entry(root, arm, pair, manifest, results=None, sx=None):
    """The record of the entry of this cell; its raw' is written once, under this batch's root only.  Exists already: it must be what this construction gives, nothing is overwritten.  EntryError otherwise, nothing injected."""
    field_path, record_path = entry_paths(root, arm, pair)
    prime, source, record = construct(arm, pair, manifest, results, sx)
    if field_path.exists() or record_path.exists():
        kept = core.read_json(record_path)
        if kept is None or not field_path.is_file():
            raise EntryError(f'{field_path.parent}: an incomplete entry, kept; nothing overwritten')
        fresh = dict(kept, u_sha256=record['raw_prime_sha256'], repaired_u_sha256=source['repaired_sha256'], charge_seconds=float(source['charge_seconds']), field_sha256=core.sha_file(field_path))
        if fresh != kept:
            raise EntryError(f'{field_path.parent}: the entry there differs from this construction; nothing overwritten')
    else:
        field_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = field_path.with_name('field_tmp.npz')
        np.savez(tmp, u=prime)
        tmp.replace(field_path)
        core.write_json(record_path, entry_record(arm, pair, field_path, prime, source, record, manifest))
        kept = core.read_json(record_path)
    try:
        core.inject_run.load_field(kept)                 # the wrapper's own check, once more when it starts
    except core.inject_run.Refused as refused:
        raise EntryError(str(refused)) from None
    return kept


def entry_trace(run):
    """run_seed.injection_trace, and what this batch adds: the transfer accepted alpha = 1 at its first attempt, on exactly the repaired field, with the evaluation's lower bound; the hold as recorded.  (record, problems).  No problem means the field the certificate passed is the field the native level started from."""
    record, problems = core.injection_trace(run)
    entry = core.read_json(Path(run)/'seed_entry.json') or {}
    extra = []
    repaired = entry.get('repaired_u_sha256')
    if repaired is None or repaired != entry.get('tapered_sha256'):
        extra.append(f"seed_entry.json: repaired_u_sha256 {repaired}, tapered_sha256 {entry.get('tapered_sha256')}")
    if record.get('seed_hash') != repaired:
        extra.append(f"seed_hash {record.get('seed_hash')} != the repaired field {repaired}")
    if record.get('accepted_alpha') != 1.:
        extra.append(f"the transfer accepted alpha {record.get('accepted_alpha')}, not 1")
    attempts = record.get('transfer_attempts') or []
    first = attempts[0] if attempts else {}
    if first.get('alpha') != 1. or first.get('passed') is not True or first.get('state_hash') != repaired:
        extra.append(f"the first transfer attempt: alpha {first.get('alpha')}, passed {first.get('passed')}, state hash {first.get('state_hash')}; the repaired field {repaired}")
    if record.get('charge_seconds') != entry.get('charge_seconds'):
        extra.append(f"the hold {record.get('charge_seconds')} != seed_entry.json {entry.get('charge_seconds')}")
    if first.get('min_jd_lower_bound') != entry.get('evaluation_lower_bound'):     # the same certificate, bit for bit: the frozen path's configuration is the one the evaluation and the taper check used
        extra.append(f"the transfer certificate's lower bound {first.get('min_jd_lower_bound')} != the evaluation's {entry.get('evaluation_lower_bound')}")
    problems = list(problems)+extra
    return dict(record, repaired_u_sha256=repaired, problems=problems, passed=not problems), problems


def native_commit_count(run):
    """The native formal commits of a finished run, from commits.json of its task directory (it stays after the cleanup).  No such file: 0."""
    protocol = core.read_json(Path(run)/'protocol.json') or {}
    commits = core.read_json(Path(run)/((protocol.get('task') or {}).get('name') or 'missing')/'commits.json') or []
    return sum(c.get('level') == 3 and c.get('native') is True for c in commits)


def two_empty_s_f(root):
    """The pre-registered rule that saves about 30 minutes (plan revision 1, the review's minor 2): why the queue stops, or None.  S_F's first two pairs have both ended with 0 native commits.  An infrastructure failure says nothing of this and does not count; what ran stays; the reading is not decidable."""
    for pair in PAIRS[:2]:
        run = run_dir(root, 'S_F', pair)
        done = core.read_json(run/'queue_done.json')
        if done is None or done.get('state') == 'infrastructure' or native_commit_count(run) != 0:
            return None
    return f'S_F {short_name(PAIRS[0])}, {short_name(PAIRS[1])}: both 0 native commits; the rule of plan revision 1 stops the queue, what ran stays, the reading is not decidable'


# ---- the cleanup ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def cleanup_after_curve(run, runs_root, curve):
    """tc_curve.cleanup_after_curve with this batch's root in its guard (that function names tc_v1; experiments/tc stays as it is): delete this run's own intermediate commits, only under <...>/s2_v1/runs, only after this run's curve.json is complete and its self-check passed, only in a task directory that really is a child of the run, only files that commits.json or levels.json records with a sha256.
    curve.CurveError otherwise, nothing deleted.  A test compares what the two functions do."""
    run, runs_root = Path(run).resolve(), Path(runs_root).resolve()
    if runs_root.name != 'runs' or runs_root.parent.name != ROOT_NAME or runs_root not in run.parents or len(run.relative_to(runs_root).parts) != 2:
        raise curve.CurveError(f'{run}: not a run directory of a {ROOT_NAME} root ({runs_root}); nothing deleted')
    if not (run/'curve.json').is_file():
        raise curve.CurveError(f'{run}: no curve.json; the commits stay')
    doc = curve.read_json(run/'curve.json')
    if doc.get('complete') is not True or (doc.get('self_check') or {}).get('passed') is not True:
        raise curve.CurveError(f'{run}: curve.json not complete; the commits stay')
    if not doc.get('run_dir') or Path(doc['run_dir']).resolve() != run:
        raise curve.CurveError(f"{run}: curve.json is of the run {doc.get('run_dir')}; the commits stay")
    if (run/'run_id').is_file() and doc.get('run_id') != (run/'run_id').read_text().strip():
        raise curve.CurveError(f"{run}: curve.json has run_id {doc.get('run_id')}; the commits stay")
    task = curve.task_dir(run).resolve()
    if task.parent != run:
        raise curve.CurveError(f'{run}: its task directory resolves to {task}, not a child of this run; nothing deleted')
    recorded = {c['path']: c.get('sha256') for c in curve.read_json(task/'commits.json')}
    recorded.update({e['path']: e.get('sha256') for e in curve.read_json(task/'levels.json') if e.get('path')})
    gone, unrecorded = [], []
    for p in curve.intermediate_files(task):
        row = dict(name=p.name, bytes=p.stat().st_size, sha256=recorded.get(p.name))
        if row['sha256'] is None:
            unrecorded.append(row)
            continue
        gone.append(row)
        p.unlink()
    kept = [dict(name=p.name, bytes=p.stat().st_size) for p in sorted(task.glob('*.npz'))]
    curve.write_json(run/'cleanup.json', dict(rule='S2 plan, as TC: after curve.json is complete, only committed_*.npz and level_*.npz of the task directory of this run with a recorded sha256 are deleted (class ok only: handover_only and method_failure stay whole); everything else is kept', deleted=gone, deleted_bytes=sum(g['bytes'] for g in gone), kept_npz=kept, not_deleted_unrecorded=unrecorded))
    return len(gone)
