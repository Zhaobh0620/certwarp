"""S_Y (runs/20261105-sy/任务书.md, plan.md): the arm, the guards and the field of the entry: the repaired field of Hx24 at 600 steps (fd2_v1's RHx24_c600) through the frozen native level, with S_P's 47 s hold.  Not F_B, not a method: a path experiment on development pairs.

Files of its own, not parameters on experiments/s2 (the brief leaves the choice): s2_core.py, decide_s2.py and run_s2.py stay byte for byte, so S2's and S_X's readings, the sha256 of decide_s2.py in s2_v1/batch.json and their tests cannot move.  S_X's guards are not this arm's anyway: S_X takes FL's arms and a hold from the records, capped at 200 s; S_Y has one fixed source and one fixed hold, no options.  From s2_core this module only calls: raw', the readers, the record of an entry, the trace.

The second taper: s2_core.py's docstring; raw_prime is that function, unchanged: raw' is the raw field with only the repair's nodes replaced, a repair node in the taper's band is a BandError, and taper(raw') must be the repaired field bit for bit.
The hold: 47 s, S_P's, a constant.  It is not what the entry costs: that is seconds_task + seconds_repair + seconds_warmup, about 144 s on average here, and it is written next to the hold in every entry record and read back by the time account of decide_sy.py.  So the native level has about S_P's length (about 277 s), which is what the brief asks: the comparison is about what the native level adds, not about the total time.
Only numpy and the records here: no jd_curl, no torch, no image, no label key.  The certificates are run_sy.py's.
"""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT/'experiments/s2'))
sys.path.insert(2, str(ROOT/'experiments/seed'))
sys.path.insert(3, str(ROOT/'experiments/tc'))

import numpy as np  # noqa: E402

import run_seed as core  # noqa: E402
import s2_core as s2  # noqa: E402

ROOT_NAME = 'sy_v1'
FD2_ROOT, S2_ROOT = 'fd2_v1', s2.ROOT_NAME       # only read
ARM = 'S_Y'
PAIRS = s2.PAIRS
HOLD = s2.HOLD_S_P                               # 47 s, S_P's: a constant, never from the records
HOLD_RULE = "47 s, S_P's, a constant: not the cost of the entry (real_task_clock is that); the native level then has about S_P's length"
SOURCE_ARM, REPAIR_ARM, STEP = 'Hx24', 'RHx24', 600
FIELD_TIER, REPAIR_TIER, FIELD_FILE = 'first5', 'c600', 'field_0600.npz'
K, CHANNELS, D_FACTOR = 35, 'onehot', 24.        # experiments/fl/fd2_core.py ARMS
EPS = s2.EPS
RUN_SCHEMA, REPAIR_SCHEMA, REPAIR_METRICS_SCHEMA = s2.FL_RUN_SCHEMA, s2.FL_REPAIR_SCHEMA, s2.FL_REPAIR_METRICS_SCHEMA
SOURCE_LABEL = f'{SOURCE_ARM}@{STEP}'
# fd2_v1, read 2026-11-05 about 11:45 am CST, before any S_Y run: a test compares the table with the records; the dry run hashes every file.
PINS = {'0438-0439': dict(raw_field='6f6f29727d0ce4414fe75ebc922d20ce63f387d2cd7e564547b484276c9611b3', repair_npz='b40eab395d2a14202a675dd62dbac193e6c974440829efe0921a7415189669ee', run_json='1759b2f3756779cca19b718c8f03e8bfea151182b6d06cf8d807f73db9fa1186', repair_json='3637cd8d594a52e5e17a19627e453d8063aec0b0c4de9a0b868b29df3d4dc265', repair_metrics_json='cc030d41201d8fec1ad7e61871bf0e0d008e7573de4b6ff7cb7637c98ea79d69', tapered_u='1f21da3b9b9daf269d3dab4951f193ed51773b521aa43fa39a936475ea6bb7f7', repaired_u='b825a1af3807b97c01ad1dc54b10b4b770fb63b7bf502787fd66b65bb33ea9bd'),
        '0439-0440': dict(raw_field='4e34c38789e9978e87486625c510eaada47a91bb67f68d95b0f987b359eb0356', repair_npz='4bef654bfd78fa9c34568d03b63e8ad328066eabc6d73f6e7f9c551be9d8486b', run_json='2b724617ce65433227e95d028582e9c250e04e17af74d826c7424e6778da511e', repair_json='275df1b9be659212adfbb5db66c5f08ca876debadd39da4501d4c44a1dee87a7', repair_metrics_json='2f86b2d3bcc14e9a1a68625360d6e57c897fdced88dcfac6b102c13e12fcacc7', tapered_u='cdf9e1ba97f091a954a777dc914fa8f429ef9708736a342c4b5203fa93cc64b4', repaired_u='c68f4698b977bb3b0aa954e12180f7fc17f5fb3c622fd758a98ed15617bf607e'),
        '0440-0441': dict(raw_field='18a9b257a34d677bb22484c83f03db4f9aefeb7d4e6ce302e6b6a0211dd7ad14', repair_npz='d3a71b380212ea8e389b51aa6467447f7a71f64f6b3d92ac97ba72f129853f6f', run_json='73f8d25811ec2d6848a049030df1811c5964b27ff3353a3a0a0df649f79667cb', repair_json='f260f546b3d93cd24ce71519537ad33f6661adf8136e166a4253361bc201eba4', repair_metrics_json='5e5206fdcf19f58666f474fac2b3a7e6628bdada2f83b62e2674e61afbaa02f5', tapered_u='f212b2c29c8fb168b27d180da8361e442c0cf2c04a15470cd718275ebf9929a2', repaired_u='b7c2893effdde02e7290bf31da448ec77d0112229bcbef96211bae7af8ccbe7b'),
        '0441-0442': dict(raw_field='fe0c0885e3f36e499dcf8e21ddaa66a59f41e8dc1152c7ef0de39d30835cdd42', repair_npz='8d6ecca9dfba43babd4543e09c4338c5bd2b94d1061c7ab9500382db8ba0ee9a', run_json='5d716f75b69f97fbddf6b3f556487a1082dce4daa3246cd89cc09d1b997a20a9', repair_json='726aad87dcef210a0d39d5e0b72254294272a22d174f591fc57bb4d633b1d284', repair_metrics_json='15bade07de1138355d45245200add4bfc0a1d8784fd99814d9580d1707adaca1', tapered_u='f83366e8fca0e1cf5ae47731bebcf01442b2768b4219097e4be83994f4d154e4', repaired_u='816b821eca42877685bc8970a1f719a02e589c6685b3fb9c9b3bcdc977991b96'),
        '0442-0443': dict(raw_field='75dba12bba0a123478d0dd4f12c4affb7eefd4aa17dd5b0b424dac45bcdf09d5', repair_npz='801d62b33fc7845a0eff050a3b93506824f0d0a3e8134cafed1d1938494f072b', run_json='aadd1bc11008a7900a0764560fdf01b6d29fb61b7173e170a906a2ce08472294', repair_json='3e8b54c5e4ac8ce83b5dd29746a531581cbffc10d1cada511c80acf7a47ea0be', repair_metrics_json='529c366988857b2dd8c42ab9dfe15ebd3890f2d8c79caf534a02d41bfe60c46b', tapered_u='f2ec2b04e5f9f6fe3a0143636a6fe6f03ac8283db379610974b7e44e1b69dd1b', repaired_u='d6f90f03068da9a8268e85786a85fab1aea9de7129cbd65866f851403407444c')}
EntryError, EntryMissing, BandError = s2.EntryError, s2.EntryMissing, s2.BandError
short_name, input_files, reference_dir, entry_trace, native_commit_count, array_sha = s2.short_name, s2.input_files, s2.reference_dir, s2.entry_trace, s2.native_commit_count, s2.array_sha


def require_cell(arm, pair):
    """Only the cells of the plan: checked before any file of the pair is opened.  The other 14 pairs are not here: their band needs its own rule and a new plan (the brief)."""
    if arm != ARM:
        raise SystemExit(f'S_Y has no arm {arm}; only {ARM}')
    if pair not in PAIRS:
        raise SystemExit(f'S_Y runs only on {[short_name(p) for p in PAIRS]}; refused: {pair}')
    return arm, pair


def plan_runs():
    return [(ARM, pair) for pair in PAIRS]


def default_root(results=None):
    return Path(core.RESULTS if results is None else results)/ROOT_NAME


def root_problem(root, results=None):
    """Why this batch may not write at this root, or None: only its own."""
    return None if Path(root).resolve() == default_root(results).resolve() else f'{root}: this batch writes only to {default_root(results)}; nothing read, nothing moved'


def run_dir(root, arm, pair):
    require_cell(arm, pair)
    return Path(root)/'runs'/arm/pair


def own_run_problem(run, arm, pair, results=None):
    """Why this directory is not this batch's run of this pair, or None: the steps after a run write only there."""
    want = run_dir(default_root(results), arm, pair)
    return None if Path(run).resolve() == want.resolve() else f'{run}: this step writes only to {want}; nothing read, nothing moved'


def unit_name(arm, pair):
    require_cell(arm, pair)
    return f'tag-sy-{arm}-{short_name(pair)}'


def stage_name(arm):
    return f'SY-{arm}'


def s_x_dir(pair, results=None):
    """S_X of the same pair: only read."""
    return Path(core.RESULTS if results is None else results)/S2_ROOT/'runs/S_X'/pair


# ---- the fixed source: fd2_v1's records, against the pins ------------------------------------------------------------------------------------------------------------------------------------
def field_dir(pair, results=None):
    """The one place the raw field of this pair may come from (experiments/fl/fd2_core.py run_dir of Cell('Hx24', 'first5', pair))."""
    require_cell(ARM, pair)
    return Path(core.RESULTS if results is None else results)/FD2_ROOT/'runs'/f'{SOURCE_ARM}_{FIELD_TIER}'/short_name(pair)


def repair_dir(pair, results=None):
    """The one place the repair of this pair may come from (fd2_core.py run_dir of Cell('RHx24', 'c600', pair))."""
    require_cell(ARM, pair)
    return Path(core.RESULTS if results is None else results)/FD2_ROOT/'runs'/f'{REPAIR_ARM}_{REPAIR_TIER}'/short_name(pair)


def source_problems(pair, pin, run, repair, metrics, manifest, results=None):
    """What is wrong with fd2_v1's three records of this pair, against the pins and each other: oracle false in each, the channels predicted by unet35_v1_ens (run_seed.channel_sources_problems), D factor 24 at 600 steps, certified at alpha = 1 by the evaluation, the same original images as A5P's inputs.  The records' format: experiments/fl/run_fl.py run_field, repair_one, evaluate_repair."""
    short = short_name(pair)
    problems = []

    def expect(where, items):
        problems.extend(f'{where} {name} {have!r}, expected {want!r}' for name, have, want in items if have != want)
    checkpoint = (run.get('checkpoints') or {}).get(str(STEP)) or {}
    expect('run.json', (('schema', run.get('schema'), RUN_SCHEMA), ('run_id', run.get('run_id'), f'fd2_{SOURCE_ARM}_{FIELD_TIER}_{short}'), ('arm', run.get('arm'), SOURCE_ARM), ('tier', run.get('tier'), FIELD_TIER), ('pair', run.get('pair'), pair), ('k', run.get('k'), K), ('channels', run.get('channels'), CHANNELS), ('D factor', run.get('D_factor'), D_FACTOR), ('native iterations', run.get('native_iterations'), STEP), ('oracle', run.get('oracle'), False), ('status', run.get('status'), 'done'),
                        ("the checkpoint's file", checkpoint.get('file'), FIELD_FILE), ("the checkpoint's sha256", checkpoint.get('sha256'), pin['raw_field'])))
    problems += [f'run.json: {p}' for p in core.channel_sources_problems(run)]
    source, certificate, attempt = repair.get('source') or {}, repair.get('certificate') or {}, repair.get('attempt') or {}
    expect('repair.json', (('schema', repair.get('schema'), REPAIR_SCHEMA), ('run_id', repair.get('run_id'), f'fd2_{REPAIR_ARM}_{REPAIR_TIER}_{short}'), ('arm', repair.get('arm'), REPAIR_ARM), ('tier', repair.get('tier'), REPAIR_TIER), ('pair', repair.get('pair'), pair), ('oracle', repair.get('oracle'), False), ('status', repair.get('status'), 'certified'), ('passed', repair.get('passed'), True), ('eps', repair.get('eps'), EPS),
                           ('source directory', source.get('directory'), str(field_dir(pair, results))), ('source file', source.get('file'), FIELD_FILE), ('source field_sha256', source.get('field_sha256'), pin['raw_field']), ('source tapered_u_sha256', source.get('tapered_u_sha256'), pin['tapered_u']), ('attempt sha256', attempt.get('sha256'), pin['repair_npz']), ('certificate state_hash', certificate.get('state_hash'), pin['repaired_u'])))
    if certificate.get('status') != 'certified_feasible' or not (certificate.get('min_jd_lower_bound') or 0) > 0 or certificate.get('n_infeasible') != 0 or certificate.get('n_uncertain') != 0:
        problems.append(f'repair.json certificate {certificate}: did not pass')
    expect('repair_metrics.json', (('schema', metrics.get('schema'), REPAIR_METRICS_SCHEMA), ('run_id', metrics.get('run_id'), f'fd2_{REPAIR_ARM}_{REPAIR_TIER}_{short}'), ('pair', metrics.get('pair'), pair), ('oracle', metrics.get('oracle'), False), ('repaired_u_sha256', metrics.get('repaired_u_sha256'), pin['repaired_u'])))
    ladder = (metrics.get('after') or {}).get('certificate') or {}
    if ladder.get('accepted_alpha') != 1. or ladder.get('status_at_1') != 'certified_feasible' or ladder.get('n_infeasible_at_1') != 0 or ladder.get('n_uncertain_at_1') != 0:
        problems.append(f"repair_metrics.json: the evaluation's certificate at alpha = 1: {ladder.get('accepted_alpha')}, {ladder.get('status_at_1')}, {ladder.get('n_infeasible_at_1')}, {ladder.get('n_uncertain_at_1')}")
    for name, value in (('seconds_task', checkpoint.get('seconds_task')), ('seconds_repair', repair.get('seconds_repair')), ('seconds_warmup', repair.get('seconds_warmup'))):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or value < 0:
            problems.append(f'{name} {value!r}: not a number of seconds')
    made = manifest.get('preprocessing_d0') or {}
    for name, data in (('run.json', run.get('data') or {}), ('repair.json', repair.get('data') or {})):
        if data.get('pair_sha256') != made.get('source_pair_sha256') or data.get('manifest_sha256') != made.get('source_manifest_sha256'):
            problems.append(f"{name}: the field comes from original pair {data.get('pair_sha256')}, the inputs from {made.get('source_pair_sha256')}")
        if data.get('fixed_subject') != manifest.get('fixed_subject') or data.get('moving_subject') != manifest.get('moving_subject'):
            problems.append(f"{name}: the field's subjects {data.get('fixed_subject')}, {data.get('moving_subject')}; the inputs' {manifest.get('fixed_subject')}, {manifest.get('moving_subject')}")
    return problems


def hold_record(seconds_task, seconds_repair, seconds_warmup):
    """The hold and, next to it, what the entry really took (FD2's task clock).  The hold does not depend on them."""
    return dict(seconds=HOLD, rule=HOLD_RULE, real_task_clock=seconds_task+seconds_repair+seconds_warmup, seconds_task=seconds_task, seconds_repair=seconds_repair, seconds_warmup=seconds_warmup)


def read_pinned(pair, results=None):
    """(run, repair, metrics): fd2_v1's three records of this pair, each against its pin.  EntryError (EntryMissing if a file is not there) otherwise."""
    require_cell(ARM, pair)
    pin = PINS[short_name(pair)]
    f, d = field_dir(pair, results), repair_dir(pair, results)
    records = []
    for path, key in ((f/'run.json', 'run_json'), (d/'repair.json', 'repair_json'), (d/'repair_metrics.json', 'repair_metrics_json')):
        if not path.is_file():
            raise EntryMissing(f'S_Y {short_name(pair)}: {path}: no such file')
        got = core.sha_file(path)
        if got != pin[key]:
            raise EntryError(f'S_Y {short_name(pair)}: {path}: sha256 {got} != the pin {pin[key]}')
        records.append(json.loads(path.read_text()))
    return tuple(records)


def pin_problems(results=None):
    """The records of the 5 pairs that are not what the pins say (the reading reads repair_metrics.json: review of the tools, minor 6).  Empty: all are."""
    problems = []
    for pair in PAIRS:
        try:
            read_pinned(pair, results)
        except EntryError as error:
            problems.append(str(error))
    return problems


def source_of(pair, manifest, results=None):
    """The source of the one cell of this pair, in the form s2_core.entry_record takes, or EntryError (EntryMissing if a file is not there).  Only the records are read here; the two field files are hashed when they are read."""
    require_cell(ARM, pair)
    pin = PINS[short_name(pair)]
    f, d = field_dir(pair, results), repair_dir(pair, results)
    run, repair, metrics = read_pinned(pair, results)
    problems = source_problems(pair, pin, run, repair, metrics, manifest, results)
    if problems:
        raise EntryError(f'S_Y {short_name(pair)}: '+'; '.join(problems))
    seconds_task = run['checkpoints'][str(STEP)]['seconds_task']
    return dict(raw_path=f/FIELD_FILE, raw_sha256=pin['raw_field'], repair_path=d/'repair.npz', repair_sha256=pin['repair_npz'], tapered_sha256=pin['tapered_u'], repaired_sha256=pin['repaired_u'], source_run_id=run['run_id'], oracle=run['oracle'], source_root=FD2_ROOT, source_k=run['k'], source_tier=SOURCE_LABEL, source_arm=SOURCE_ARM,
                charge_seconds=HOLD, seconds_entry=seconds_task, seconds_repair=repair['seconds_repair'], seconds_warmup=repair['seconds_warmup'], hold=hold_record(seconds_task, repair['seconds_repair'], repair['seconds_warmup']), entry_metrics=s2.entry_metrics_of(metrics), repair_dir=str(d), repair_eps=repair['eps'], evaluation_lower_bound=repair['certificate'].get('min_jd_lower_bound'))


def construct(arm, pair, manifest, results=None):
    """(raw', source, record) of the cell: only the raw field and the repair's two arrays are read.  EntryError, BandError otherwise: a repair node in the band is reported with its counts and never used."""
    require_cell(arm, pair)
    source = source_of(pair, manifest, results)
    shape = tuple(manifest['shape_zyx'])
    raw = s2.read_raw(source['raw_path'], source['raw_sha256'], shape)
    nodes, values = s2.read_repair(source['repair_path'], source['repair_sha256'], shape)
    prime, record = s2.raw_prime(raw, nodes, values, tapered_sha256=source['tapered_sha256'], repaired_sha256=source['repaired_sha256'], label=f'{arm} {short_name(pair)}')
    return prime, source, record


# ---- the entry file: raw' under <root>/entries, once ----------------------------------------------------------------------------------------------------------------------------------------------
def entry_paths(root, arm, pair):
    require_cell(arm, pair)
    d = Path(root)/'entries'/arm/short_name(pair)
    return d/'field.npz', d/'entry.json'


def write_entry(root, arm, pair, manifest, results=None):
    """The record of the entry of this cell; its raw' is written once, under this batch's root only.  Exists already: it must be what this construction gives, nothing is overwritten.  EntryError otherwise, nothing injected."""
    field_path, record_path = entry_paths(root, arm, pair)
    prime, source, record = construct(arm, pair, manifest, results)
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
        core.write_json(record_path, s2.entry_record(arm, pair, field_path, prime, source, record, manifest))
        kept = core.read_json(record_path)
    try:
        core.inject_run.load_field(kept)                 # the wrapper's own check, once more when it starts
    except core.inject_run.Refused as refused:
        raise EntryError(str(refused)) from None
    return kept


# ---- the cleanup ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def cleanup_after_curve(run, runs_root, curve):
    """s2_core.cleanup_after_curve with this batch's root in its guard (that function names s2_v1; experiments/s2 stays as it is): delete this run's own intermediate commits, only under <...>/sy_v1/runs, only after this run's curve.json is complete and its self-check passed, only in a task directory that really is a child of the run, only files that commits.json or levels.json records with a sha256.
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
    curve.write_json(run/'cleanup.json', dict(rule='S_Y plan, as S2 and TC: after curve.json is complete, only committed_*.npz and level_*.npz of the task directory of this run with a recorded sha256 are deleted (class ok only: everything else stays whole); everything else is kept', deleted=gone, deleted_bytes=sum(g['bytes'] for g in gone), kept_npz=kept, not_deleted_unrecorded=unrecorded))
    return len(gone)
