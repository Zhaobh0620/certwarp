"""HP (runs/20261107-hp/任务书.md item 1, 2): the B-line hold-out stage -- pair.npz and manifest.json of the 50 pairs of splits/bayesreg_v1/holdout_B_pairs.tsv -- and
its development check.

Contract C23.12-3 and prereg revision 1 item 1 (docs/bayesreg/notes/留出集预登记_20261107.md; read as the lead ruled on 2026-11-09, review m2: no field derived from
labels or naming them; values computed from the intensities and the grid, constant weights and descriptive strings may stay): the stage holds no label.  pair.npz has
exactly fixed, moving and an all-zero initial_u (float64, (nz, ny, nx, 3)), written by np.savez_compressed in the order of the development stage without its label
members; manifest.json has the fields of archive/seq_vp_3d/scripts/prepare_stage_3d_l2r_val.py without those derived from labels or naming them
(runs/20261107-hp/实现说明.md, field table).  This process reads no label: aligned_norm.nii.gz only, only through OasisFiles.image_path of bayesreg.io.unseal.bind_files
(C23.4-1), with jd_curl.three_d.nifti.load_scalar (labels=False), the reader of the development stage; no aligned_seg file is opened or hashed.  The grid check is the
original script's: both images on one grid (require_common_grid), shape (192, 224, 160) zyx, spacing 1 mm.

    python -B experiments/holdout/prep_stage.py run --dry-run --out DIR [--unseal PATH] [--resume PATH]
        the decision (recorded, not refused), the admission (the pair list, the sealed lists, the unseal record through SG, the pinned allowlist, the rows), the
        resume authorization if given, the output and lock checks; no image, nothing written but DIR/dry_run.json.  Without BZ's record (main today) it is refused at
        the admission (stage unseal).  DIR: new or empty, outside the results root, the D: directory, the data roots and the code trees.
    python -B experiments/holdout/prep_stage.py run --unseal splits/bayesreg_v1/unseal_holdout_B_v1.json [--resume splits/bayesreg_v1/holdout_B_v1_hp_prep_resume_<k>.json]
        after BZ unseals (the dedicated branch, a clean tree) and after the development check is decided (PREP_DEVCHECK_DECISION), in the unit tag-hb-prep: the 50
        pairs into holdout_common.STAGE_ROOT (new), once (the lock holdout_common.LOCK_ROOT/hp_prep_stage.lock, O_EXCL, after every check), under the spy.  After a
        failure only with a committed resume authorization (holdout_common.load_resume; at most once): the failed attempt is kept whole as <stage>.attempt1.
    python -B experiments/holdout/prep_stage.py devcheck [--dry-run --out DIR]
        the development check (runs/20261107-hp/prep_devcheck_plan.md; refused while PREP_DEVCHECK_PLAN_SHA256 is None): the same code, admitted by the development
        allowlist, writes the 19 development pairs into <results>/hp_devcheck_v1/prep/stage and compares them with data/three_d/stage_L2R_val_v1 (fixed, moving,
        initial_u bit for bit through bayesreg.io.pair's npz hook, one member at a time; the manifest's grid and source sha256 through its whitelisting decoder;
        load_pair's keys).  Once (its own lock); no resume.

Order of a hold-out run; no image is opened before step 9: 1. the test-sandbox guard on the stage root and the locks (hp_core.guard_write); 2. the code (one clean
tree; hp_core.check_code, in main; rule one, hp_core.check_pinned, in main: P1 from the committed runs/holdout_B_v1/pinned_code.json -- A5a; run_holdout refuses a code
record without it; the lock line's code_commit is P1); 3. the guard's cache of the sealed lists empty, the spy installed (C23.6-5); 4. the decision of the development check
(PREP_DEVCHECK_DECISION; review I1); 5. the resume authorization, if given; 6. the admission (hp_core.admit_holdout: all 50 rows, ids, H1-H4, load_unseal, bind_unseal,
check_subject, the rows, bind_files); 7. the stage root new and the lock not there (a first run), or the lock there and the resume lock not (a resume); 8. the machine
gate (holdout_common.machine_gate: no other tag-* or seq-vp-* unit, GPU idle, MemAvailable >= 6 GiB, load < 1.5); 9. the lock (or the resume lock; a resume then keeps
the failed attempt as <stage>.attempt1); 10. the stage, pair after pair in row order: selection.json first, completion.json and SHA256SUMS last; 11. prep_record.json (the
spy's record with every open).  A failure after the lock leaves prep_stopped.json and the partial stage (kept, not rerun: project rules, hard rule 4).
Exit codes: 0 done (or a dry run that passed), 2 usage, 8 refused, stopped by the spy, or a check failed (the grid, the source images, a reader's InputValidationError;
the stopped record kept), 1 any other failure (prep_stopped.json when after the lock).
"""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import holdout_common as hc  # noqa: E402
import hp_core as core  # noqa: E402

from jd_curl.three_d.nifti import file_sha256, load_scalar, require_common_grid  # noqa: E402
from jd_curl.validation import InputValidationError  # noqa: E402
from bayesreg.io.guard import SubjectSealed  # noqa: E402

SCHEMA = 'bayesreg.hp.prep.v1'
SCRIPT = 'experiments/holdout/prep_stage.py'
TOOL, ARM = 'hp_prep', 'stage'                  # the resume authorization's tool and arm (holdout_common.load_resume; pair 'all', pair set 'holdout')
UNIT = 'tag-hb-prep'
DEVCHECK_UNIT = 'tag-hb-prep-devcheck'
LOCK_NAME = 'hp_prep_stage.lock'
DATA_ROOT = Path('/mnt/d/data')                 # bayesreg.seg.unet.data.DEFAULT_DATA_ROOT; the development stage was made from it (prepare_stage_3d_l2r_val.py:24-27)
EXPECTED_SHAPE_ZYX = (192, 224, 160)            # prepare_stage_3d_l2r_val.py:47
EXPECTED_SPACING_XYZ = (1., 1., 1.)
SIDES = ('fixed', 'moving')
SOURCE_DROPPED = ('labels',)                    # load_scalar's flag labels=False: its name names labels (the brief and C23.12-3); the reader is recorded as READER
READER = 'jd_curl.three_d.nifti.load_scalar (intensity image) through OasisFiles.image_path'
PREPROCESSING = 'pure RAS storage reorientation only; no resampling/smoothing/intensity adjustment'      # prepare_stage_3d_l2r_val.py:60, word for word
MAPPING = 'fixed -> moving pullback'
DATASET_HOLDOUT = 'OASIS native aligned full-volume, B-line hold-out pair (splits/bayesreg_v1/holdout_B_pairs.tsv)'
DATASET_DEV = 'OASIS native aligned full-volume, Learn2Reg OASIS validation pair (HP development check copy, images only)'
RULE_HOLDOUT = 'the B-line hold-out pairs: splits/bayesreg_v1/holdout_B_pairs.tsv in its row order; fixed <- moving (pullback)'
RULE_DEV = 'the 19 development pairs of stage_L2R_val_v1 in their order (HP development check)'
STOPPED = 'prep_stopped.json'
RECORD = 'prep_record.json'
DEVCHECK_RECORD = 'devcheck_prep.json'
SUMS = 'SHA256SUMS'
RUNTIME_S = 2*3600
DEVCHECK_PLAN_REL = 'runs/20261107-hp/prep_devcheck_plan.md'
PREP_DEVCHECK_PLAN_SHA256 = 'aee9ff25eaee3f467de664c66c229f169b56f77f2a19fb7f40f57fb30f2f8c97'     # P0 (11-14): the final plan (PT 5ef8243); the development check refuses None or another plan
READ_EQUAL, READ_DIFFERENT = '逐位相同 {n}/{n}', '不逐位（交总负责人）'
# Review I1 (the lead's ruling of 2026-11-09): filled by a reviewed commit (P1) after the development check ran on P0 and was read (prep_devcheck_plan.md §4), as a plain dict
# literal (contract C24.3-6 (a): rule two takes only a literal; hp_core.check_decision reads it through a read-only copy) with exactly hp_core.DECISION_KEYS, code_commit the
# commit the check ran on (A5a); only the reading 逐位相同 19/19 lets the hold-out run start.  None (today): the hold-out run refuses (stage decision).
PREP_DEVCHECK_DECISION = {
    'devcheck_json': '/home/rmcfz/seq_vp_tag_results/hp_devcheck_v1/prep/devcheck_prep.json',
    'devcheck_sha256': 'ab6a22199311088534f809afa2d336742ed0aa337670833030c84b7464c75b34',
    'reading': '逐位相同 19/19',
    'plan_sha256': 'aee9ff25eaee3f467de664c66c229f169b56f77f2a19fb7f40f57fb30f2f8c97',
    'code_commit': '318e72ad3d097bcbd03db180a56c0144ee4344bc',
    'decided_by': 'lead (total responsible), 2026-11-15, per prep_devcheck_plan.md §4',
    'review': 'docs/bayesreg/reviews/D58_复核_20261115.md (sha256 9629f2495a4f099347871e4eb3d606a22f2484838745e65e494fe1145f946c4b; 0 blocking, 1 important, 8 minor; reading holds)',
    'bz_words': None,
}     # P1 (11-15): filled after the check ran on P0'' 318e72ad (C1, D5); bz_words None: the reading is 逐位相同, no words of BZ needed (plan §4, revision 1 item 6)


class StageCheckFailed(Exception):
    """A grid or source check of the original script failed: the run stops (record kept)."""


def devcheck_root():
    return Path(hc.RESULTS)/core.DEVCHECK_DIR/'prep'


def devcheck_lock():
    return Path(hc.RESULTS)/core.DEVCHECK_DIR/'locks'/'prep_devcheck.lock'


def say(text):
    print(text, flush=True)


def dump(path, value):
    """archive/seq_vp_3d/scripts/run_stage_3d_c.py:32-36 dump (the development stage's JSON writer), copied: indent 2, no NaN, numpy scalars as items, a temporary
    file in the same directory replaced into place."""
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda obj: obj.item() if isinstance(obj, np.generic) else str(obj))+'\n')
    temp.replace(path)


def write_pair(out, fixed, moving, files, *, dataset, extra):
    """One pair as prepare_stage_3d_l2r_val.py:41-64 writes it, without the label members and the records derived from them: both images through files.image_path and
    load_scalar on one grid, an all-zero initial_u, pair.npz (fixed, moving, initial_u), manifest.json.  Returns the manifest."""
    arrays, sources, reference, grid = {}, {}, None, None
    for side, subject in zip(SIDES, (fixed, moving)):
        volume = load_scalar(files.image_path(subject))
        if reference is None:
            reference = volume
        try:
            grid = require_common_grid(reference, volume)
        except InputValidationError as exc:
            raise StageCheckFailed(f'{fixed}, {moving}: {exc} (prepare_stage_3d_l2r_val.py:47)') from None
        if tuple(grid.shape) != tuple(EXPECTED_SHAPE_ZYX) or tuple(grid.spacing) != tuple(EXPECTED_SPACING_XYZ):
            raise StageCheckFailed(f'{subject}: grid {grid.shape} spacing {grid.spacing}, expected {EXPECTED_SHAPE_ZYX} and {EXPECTED_SPACING_XYZ} (prepare_stage_3d_l2r_val.py:47)')
        arrays[side] = volume.values
        sources[side] = {k: v for k, v in volume.metadata.items() if k not in SOURCE_DROPPED}
    arrays['initial_u'] = np.zeros(grid.shape+(3,))
    q = grid.node_weights()
    initial_mse = float(np.sum(q*(arrays['moving']-arrays['fixed'])**2, dtype=np.longdouble)/np.sum(q, dtype=np.longdouble))
    L = np.prod([a[-1]-a[0] for a in grid.axes])**(1/3)
    np.savez_compressed(out/'pair.npz', **arrays)
    record = dict(dataset=dataset, pair=hc.pair_name(fixed, moving), fixed_subject=fixed, moving_subject=moving, mapping=MAPPING,
                  shape_zyx=grid.shape, spacing_xyz=grid.spacing, origin_xyz=[float(a[0]) for a in grid.axes], unit='mm', preprocessing=PREPROCESSING, reader=READER,
                  sources=sources, pair_sha256=file_sha256(out/'pair.npz'), baseline_native_mse=initial_mse,
                  weights=dict(lam=.1/L**2, kappa=.002, strong_mu_nu=1., weak_mu_nu=.001), **extra)
    dump(out/'manifest.json', record)
    if not all(file_sha256(s['source']) == s['sha256'] for s in sources.values()):
        raise StageCheckFailed(f'{record["pair"]}: a source image changed while the pair was written')
    return record


def stage_files(pairs):
    return ['selection.json']+[f'{hc.pair_name(f, m)}/{n}' for f, m in pairs for n in ('pair.npz', 'manifest.json')]+['completion.json']


def prepare(stage_root, pairs, files, *, rule, dataset, rows=None, pair_list=None, log=say):
    """The stage: stage_root made here (new), selection.json, the pairs in order, completion.json, SHA256SUMS of every file named by stage_files.  Returns the summary."""
    stage_root = Path(stage_root)
    core.guard_write(stage_root)
    started = perf_counter()
    stage_root.mkdir(parents=True, exist_ok=False)
    rows = [None]*len(pairs) if rows is None else list(rows)
    schedule = [dict(pair=hc.pair_name(f, m), fixed=str(Path(files.root)/f), moving=str(Path(files.root)/m), **({} if r is None else dict(row=r)))
                for (f, m), r in zip(pairs, rows)]
    dump(stage_root/'selection.json', dict(rule=rule, pairs=schedule, pair_list=pair_list, data_root=str(files.root), no_metric_based_selection=True))
    manifests, seconds = {}, {}
    for (fixed, moving), row in zip(pairs, rows):
        name = hc.pair_name(fixed, moving)
        t0 = perf_counter()
        out = stage_root/name
        out.mkdir()
        extra = dict(pair_list=pair_list) if row is None else dict(holdout_row=row, pair_list=pair_list)
        manifests[name] = write_pair(out, fixed, moving, files, dataset=dataset, extra=extra)
        seconds[name] = round(perf_counter()-t0, 3)
        log(f'{name}: {seconds[name]:.1f} s')
    dump(stage_root/'completion.json', dict(complete=True, pairs=len(pairs), elapsed_seconds=perf_counter()-started, no_optimizer_runs=True))
    sums = ''.join(f'{file_sha256(stage_root/n)}  {n}\n' for n in sorted(stage_files(pairs)))
    (stage_root/SUMS).write_text(sums, encoding='utf-8')
    return dict(stage=str(stage_root), pairs=len(pairs), seconds=seconds, elapsed_seconds=round(perf_counter()-started, 3), sha256sums_sha256=file_sha256(stage_root/SUMS),
                pair_sha256={n: m['pair_sha256'] for n, m in manifests.items()})


# ---- the decision, the resume -------------------------------------------------------------------------------------------------------------------
def check_decision(decision=None):
    """Review I1: PREP_DEVCHECK_DECISION (read at call time) names the development check's record (devcheck_root()/devcheck_prep.json) by its sha256, the pinned plan,
    the reading 逐位相同 19/19 and the check's lock line (hp_core.check_decision).  Refused (stage decision) otherwise."""
    decision = PREP_DEVCHECK_DECISION if decision is None else decision
    n = len(hc.DEV_PAIRS)

    def extra(record):
        return [] if record.get('n') == n and record.get('n_equal') == n else [f"the check compared {record.get('n')} pairs, {record.get('n_equal')} equal; {n} and {n} expected"]
    return core.check_decision(decision, constant='PREP_DEVCHECK_DECISION', record_path=devcheck_root()/DEVCHECK_RECORD, lock_path=devcheck_lock(),
                               plan_sha256=PREP_DEVCHECK_PLAN_SHA256, schema=SCHEMA+'.devcheck', accepted=(READ_EQUAL.format(n=n),), extra=extra)


def previous_record(stage_root, lock):
    """Where the failed run left its stopped record: in the stage root, or beside the lock when the stage root was never made (the default when there is neither)."""
    for path in (Path(stage_root)/STOPPED, Path(lock).parent/STOPPED):
        if path.is_file():
            return path
    return Path(stage_root)/STOPPED


def run_complete(stage_root):
    """Prereg §5.4 "完整记录" of the data preparation: prep_record.json in the stage root with passed true."""
    path = Path(stage_root)/RECORD
    try:
        return bool(json.loads(path.read_text(encoding='utf-8')).get('passed'))
    except (OSError, ValueError):
        return False


# ---- the hold-out run ----------------------------------------------------------------------------------------------------------------------------
def holdout_policy(stage_root, lock, *, out=None, resume_path=None):
    """The spy's policy of a hold-out run before the admission: the id lists and records, the decision's two records, the resume authorization with the lock and the
    previous record it names; the images are added after the admission (allow_read)."""
    import bayesreg.io.sealed as sealed
    from bayesreg.io.unseal import DECISION_LOG_REL, UNSEAL_REL
    tree = Path(sealed.REPO)
    reads = [tree/hc.PAIRS_TSV_REL, tree/UNSEAL_REL, tree/core.ALLOWLIST_REL, tree/DECISION_LOG_REL, *core.allowed_list_paths(), devcheck_root()/DEVCHECK_RECORD,
             devcheck_lock()]
    if resume_path is not None:
        auth = Path(resume_path) if Path(resume_path).is_absolute() else Path(hc.AUTH_ROOT)/resume_path
        reads += [auth, lock, previous_record(stage_root, lock)]
    writes = [stage_root] if out is None else [out]
    return core.Policy(name='hp prep (hold-out)', read_exact=reads, write_inside=writes, write_exact=[lock, hc.resume_lock_path(lock)], results=hc.RESULTS)


def run_holdout(*, unseal_path, code, data_root=None, dry_run=False, out=None, resume_path=None, log=say):
    """The module docstring's order.  Returns the run's record (dry run: the admission's); holdout_common.Refused, hp_core.Breach, StageCheckFailed otherwise."""
    stage_root, lock = Path(hc.STAGE_ROOT), Path(hc.LOCK_ROOT)/LOCK_NAME
    core.guard_write(stage_root, lock, hc.resume_lock_path(lock), core.attempt_path(stage_root))
    problem = core.pinned_code_problem(code)
    if problem and not dry_run:                     # A5a: rule one passed in main (hp_core.check_pinned), else nothing of the hold-out is read
        raise hc.Refused('code', problem)
    data_root = Path(DATA_ROOT if data_root is None else data_root)
    record = dict(schema=SCHEMA, tool=f'{SCRIPT} run', started_utc=hc.utc(), stage=str(stage_root), lock=str(lock), data_root=str(data_root), dry_run=dry_run, code=code,
                  unseal_path=None if unseal_path is None else str(unseal_path), resume_path=None if resume_path is None else str(resume_path))
    record['preload'] = core.preload()
    policy = holdout_policy(stage_root, lock, out=out if dry_run else None, resume_path=resume_path)
    with core.spying(policy) as spy:
        try:
            record['decision'] = check_decision()
        except hc.Refused as exc:
            if not dry_run:
                raise
            record['decision'] = dict(refused=str(exc))
        resume = core.load_resume(resume_path, tool=TOOL, arm=ARM, lock=lock, previous_record=previous_record(stage_root, lock), complete=lambda: run_complete(stage_root))
        admission = core.admit_holdout(None, unseal_path=unseal_path, data_root=data_root)
        if [tuple(p) for p in admission.pairs] != [tuple(r) for r in admission.rows]:
            raise hc.Refused('admission', 'the stage is every row of the pair list, in its order')
        record['admission'] = admission.record
        hc.check_once(lock, resume=resume)
        if resume is None and stage_root.exists():
            raise hc.Refused('output', f'{stage_root} exists: the stage is written once, never overwritten (after a failure: a committed resume authorization, prereg §5.4)')
        if resume is not None and core.attempt_path(stage_root).exists():
            raise hc.Refused('resume', f'{core.attempt_path(stage_root)} exists: a failed attempt was kept already')
        subjects = list(dict.fromkeys(s for p in admission.pairs for s in p))
        images = [admission.files.image_path(s) for s in subjects]           # ids through the bound guard again; nothing opened
        record['images'] = dict(n=len(images), root=str(data_root))
        if dry_run:
            record['launch'] = core.launch_line(unit=UNIT, script=SCRIPT, args=['run', '--unseal', str(unseal_path)]+([] if resume is None else ['--resume', str(resume_path)]),
                                                runtime_s=RUNTIME_S)
            record['spy'] = spy.record()
            return record
        record['machine'] = hc.machine_gate(own_units=(UNIT,))
        line = hc.lock_line(tool=f'{SCRIPT} run', stage=str(stage_root), pairs_tsv_sha256=hc.PAIRS_TSV_SHA256, unseal_sha256=admission.unseal.sha256,
                            decision_sha256=record['decision']['devcheck_sha256'], code_commit=(code or {}).get('commit'),
                            event='authorized_resume' if resume else 'first_run', resume_sha256=None if resume is None else resume.sha256)
        record['lock_line'] = hc.take_lock(resume.lock_path if resume else lock, line)
        if resume is not None:
            record['resume'] = dict(path=resume.path, sha256=resume.sha256, kept=core.keep_failed_attempt(stage_root))
        spy.policy.allow_read(images)
        try:
            record['stage_summary'] = prepare(stage_root, admission.pairs, admission.files, rule=RULE_HOLDOUT, dataset=DATASET_HOLDOUT, rows=admission.row_numbers,
                                              pair_list=dict(path=hc.PAIRS_TSV_REL, sha256=hc.PAIRS_TSV_SHA256), log=log)
        except BaseException as exc:
            record.update(stopped=f'{type(exc).__name__}: {exc}', spy=spy.record(expect=images), ended_utc=hc.utc())
            spy.remove()
            hc.write_json_once((stage_root if stage_root.is_dir() else lock.parent)/STOPPED, record)
            raise
        record['spy'] = spy.record(expect=images+[Path(core._sealed().REPO)/hc.PAIRS_TSV_REL])
    record['ended_utc'] = hc.utc()
    record['passed'] = bool(record['spy']['passed'])
    hc.write_json_once(stage_root/RECORD, record)
    return record


# ---- the development check -----------------------------------------------------------------------------------------------------------------------
def dev_pairs():
    return [tuple(name.split('__')) for name in hc.DEV_PAIRS]


def check_plan(repo=None):
    """The development check reads its plan's sha256 against PREP_DEVCHECK_PLAN_SHA256 (None: refused, the plan is a draft)."""
    import bayesreg.io.sealed as sealed
    path = Path(sealed.REPO if repo is None else repo)/DEVCHECK_PLAN_REL
    if PREP_DEVCHECK_PLAN_SHA256 is None:
        raise hc.Refused('plan', f'{DEVCHECK_PLAN_REL} is a draft: PREP_DEVCHECK_PLAN_SHA256 is None until the lead pins it (a reviewed commit)')
    digest = file_sha256(path) if path.is_file() else None
    if digest != PREP_DEVCHECK_PLAN_SHA256:
        raise hc.Refused('plan', f'{DEVCHECK_PLAN_REL}: sha256 {digest} is not the pinned {PREP_DEVCHECK_PLAN_SHA256}')
    return dict(path=DEVCHECK_PLAN_REL, sha256=digest)


def read_member(path, key):
    """One member of a pair.npz through bayesreg.io.pair's archive hook (_open_npz), the archive opened for this one member; the archive is not listed."""
    from bayesreg.io.pair import _open_npz
    with _open_npz(path) as npz:
        return np.asarray(npz[key])


def compare_pair(new_dir, old_dir, allowlist):
    """The new pair against the development stage's: the members fixed, moving, initial_u (dtype, shape, bytes); the manifest's grid and source sha256 through the
    loader's whitelisting decoder; load_pair's keys and arrays (prereg revision 1a item 21).  No label member, no label-derived manifest field is read.  The old pair is
    a pair of the development stage written in the code (holdout_common.DEV_STAGE); its caller admitted both subjects (admit_dev)."""
    from bayesreg.io.pair import _read_manifest, load_pair
    members = {}
    for key in core.PAIR_MEMBERS:
        a, b = read_member(new_dir/'pair.npz', key), read_member(old_dir/'pair.npz', key)
        members[key] = dict(dtype_new=str(a.dtype), dtype_old=str(b.dtype), shape_new=list(a.shape), shape_old=list(b.shape),
                            bitwise=a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes())
    _, meta_new, sel_new = _read_manifest(new_dir/'manifest.json')
    _, meta_old, sel_old = _read_manifest(old_dir/'manifest.json')
    grid = {k: dict(new=meta_new[k], old=meta_old[k], equal=meta_new[k] == meta_old[k]) for k in ('shape_zyx', 'spacing_xyz', 'origin_xyz', 'unit')}
    sources = {s: dict(new=sel_new['source_sha256'][s], old=sel_old['source_sha256'][s], equal=sel_new['source_sha256'][s] == sel_old['source_sha256'][s]) for s in SIDES}
    subjects = dict(equal=(sel_new['fixed_subject'], sel_new['moving_subject']) == (sel_old['fixed_subject'], sel_old['moving_subject']))
    p_new, p_old = load_pair(new_dir/'pair.npz', allowlist=allowlist), load_pair(old_dir/'pair.npz', allowlist=allowlist)
    loaded = dict(keys_new=p_new.provenance['keys_read'], keys_old=p_old.provenance['keys_read'], keys_equal=p_new.provenance['keys_read'] == p_old.provenance['keys_read'],
                  arrays_bitwise=all(getattr(p_new, s).tobytes() == getattr(p_old, s).tobytes() for s in SIDES))
    equal = all(m['bitwise'] for m in members.values()) and all(g['equal'] for g in grid.values()) and all(s['equal'] for s in sources.values()) and subjects['equal'] \
        and loaded['keys_equal'] and loaded['arrays_bitwise']
    return dict(members=members, grid=grid, sources_sha256=sources, subjects=subjects, load_pair=loaded, equal=equal)


def devcheck_policy(out_root, lock, dev_stage, pairs, *, extra_out=None):
    import bayesreg.io.sealed as sealed
    tree = Path(sealed.REPO)
    reads = [tree/core.ALLOWLIST_REL, tree/DEVCHECK_PLAN_REL, *core.allowed_list_paths(), Path(dev_stage)/'selection.json']
    for fixed, moving in pairs:
        reads += [Path(dev_stage)/hc.pair_name(fixed, moving)/n for n in ('pair.npz', 'manifest.json')]
    return core.Policy(name='hp prep (development check)', read_exact=reads, write_inside=[out_root] if extra_out is None else [extra_out], write_exact=[lock], results=hc.RESULTS)


def run_devcheck(*, code, data_root=None, dry_run=False, out=None, log=say):
    out_root, lock, dev_stage = devcheck_root(), devcheck_lock(), Path(hc.DEV_STAGE)
    core.guard_write(out_root, lock)
    data_root = Path(DATA_ROOT if data_root is None else data_root)
    pairs = dev_pairs()
    record = dict(schema=SCHEMA+'.devcheck', tool=f'{SCRIPT} devcheck', started_utc=hc.utc(), out=str(out_root), lock=str(lock), dev_stage=str(dev_stage),
                  data_root=str(data_root), dry_run=dry_run, code=code)
    record['preload'] = core.preload()
    with core.spying(devcheck_policy(out_root, lock, dev_stage, pairs, extra_out=out if dry_run else None)) as spy:
        try:
            record['plan'] = check_plan()
        except hc.Refused as exc:
            if not dry_run:
                raise
            record['plan'] = dict(refused=str(exc))
        allowlist, record['admission'] = core.admit_dev(pairs)
        selection = json.loads((dev_stage/'selection.json').read_text(encoding='utf-8'))
        names = [row['pair'] for row in selection['pairs']]
        if names != list(hc.DEV_PAIRS):
            raise hc.Refused('admission', f'{dev_stage}/selection.json lists {names[:3]}..., not the 19 development pairs in order')
        if out_root.exists():
            raise hc.Refused('output', f'{out_root} exists: the development check writes a new directory')
        if lock.exists():
            raise hc.Refused('once', f'{lock} exists: the development check runs once')
        from bayesreg.seg.unet.data import OasisFiles
        files = OasisFiles(root=data_root)
        subjects = list(dict.fromkeys(s for p in pairs for s in p))
        images = [files.image_path(s) for s in subjects]
        if dry_run:
            record['launch'] = core.launch_line(unit=DEVCHECK_UNIT, script=SCRIPT, args=['devcheck'], runtime_s=RUNTIME_S)
            record['spy'] = spy.record()
            return record
        record['machine'] = hc.machine_gate(own_units=(DEVCHECK_UNIT,))
        record['lock_line'] = hc.take_lock(lock, hc.lock_line(tool=f'{SCRIPT} devcheck', out=str(out_root), plan_sha256=record['plan']['sha256'],
                                                              code_commit=(code or {}).get('commit')))
        spy.policy.allow_read(images)
        try:
            record['stage_summary'] = prepare(out_root/'stage', pairs, files, rule=RULE_DEV, dataset=DATASET_DEV, pair_list=dict(path=f'{dev_stage}/selection.json'), log=log)
            rows = {hc.pair_name(f, m): compare_pair(out_root/'stage'/hc.pair_name(f, m), dev_stage/hc.pair_name(f, m), allowlist) for f, m in pairs}
        except BaseException as exc:
            record.update(stopped=f'{type(exc).__name__}: {exc}', spy=spy.record(expect=images), ended_utc=hc.utc())
            spy.remove()
            hc.write_json_once((out_root if out_root.is_dir() else lock.parent)/STOPPED, record)
            raise
        record['spy'] = spy.record(expect=images)
    n_equal = sum(r['equal'] for r in rows.values())
    record.update(pairs=rows, n_equal=n_equal, n=len(rows), reading=READ_EQUAL.format(n=len(rows)) if n_equal == len(rows) and record['spy']['passed'] else READ_DIFFERENT,
                  ended_utc=hc.utc())
    hc.write_json_once(out_root/DEVCHECK_RECORD, record)
    return record


# ---- the command line ----------------------------------------------------------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog=SCRIPT)
    sub = ap.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--unseal', type=Path, default=None)
    run.add_argument('--resume', type=Path, default=None)
    run.add_argument('--dry-run', action='store_true')
    run.add_argument('--out', type=Path, default=None)
    dev = sub.add_parser('devcheck')
    dev.add_argument('--dry-run', action='store_true')
    dev.add_argument('--out', type=Path, default=None)
    a = ap.parse_args(argv)
    if a.dry_run and a.out is None:
        ap.error('--dry-run writes its record into --out DIR (a new or empty directory outside the results roots, e.g. mktemp -d)')
    return a


def main(argv=None, *, code=None, out=say):
    a = parse_args(argv)
    if a.dry_run:
        problem = core.dry_run_out_problem(a.out)
        if problem:
            out(f'{problem}: a dry run writes into a new or empty directory outside the results root, the D: directory, the data roots and the code trees')
            return 2
    try:
        code = core.code_identity() if code is None else code
        code_check = core.check_code(code, require_clean=not a.dry_run)
        if a.command == 'run':                      # A5a (contract C24.3 rule one): the hold-out run's code is the pinned commit's; the development check is not (C24.3-10)
            code = core.check_pinned(code, dry_run=a.dry_run)
    except hc.Refused as exc:
        out(f'refused ({exc.stage}): {exc}')
        return 8
    decision_none = a.command == 'run' and PREP_DEVCHECK_DECISION is None       # A5a diff review n1: a hold-out run refuses at stage decision while it is None (devcheck does not read it)

    def stopped_dry(stage, message, **extra):         # A5a review m8: a dry run that stops writes the same record (passed, problems, expected, unexpected, p0_ok)
        if a.dry_run:
            a.out.mkdir(parents=True, exist_ok=True)
            hc.write_json_once(a.out/'dry_run.json', dict(schema=SCHEMA+'.dry_run', command=a.command, refused=dict(stage=stage, message=message), utc=hc.utc(), code=code, **extra,
                                                          **core.dry_run_fields(code, code_check=code_check, refusal=(stage, message), decision_none=decision_none)))
    try:
        if a.command == 'run':
            record = run_holdout(unseal_path=a.unseal, code=dict(code, check=code_check), dry_run=a.dry_run, out=a.out, resume_path=a.resume)
        else:
            record = run_devcheck(code=dict(code, check=code_check), dry_run=a.dry_run, out=a.out)
    except hc.Refused as exc:
        out(f'refused ({exc.stage}): {exc}')
        stopped_dry(exc.stage, str(exc), spy=getattr(exc, 'hp_spy', None))
        return 8
    except core.Breach as exc:
        out(f'stopped by the spy: {exc}')
        stopped_dry('spy', str(exc))
        return 8
    except (StageCheckFailed, InputValidationError, SubjectSealed) as exc:      # the original script's checks, the readers' and the guard's (review m9)
        out(f'stopped: {type(exc).__name__}: {exc}')
        stopped_dry(type(exc).__name__, str(exc))
        return 8
    if a.dry_run:
        a.out.mkdir(parents=True, exist_ok=True)
        fields = core.dry_run_fields(code, code_check=code_check, record=record, decision_none=decision_none)
        hc.write_json_once(a.out/'dry_run.json', dict(record, schema=SCHEMA+'.dry_run', command=a.command, **fields))
        out(f"dry run passed the admission ({a.command}); launch: {record.get('launch')}; problems: {fields['problems'] or 'none'}; P0 reading: "
            f"{'only the expected refusals' if fields['p0_ok'] else 'unexpected problems'} (the record, not the exit code)")
        return 0
    if a.command == 'devcheck':
        out(f"devcheck: done; reading {record['reading']} ({record['n_equal']}/{record['n']} pairs equal); spy passed {record['spy']['passed']}")
        return 0
    out(f"run: done; {record['stage_summary']['pairs']} pairs in {record['stage']}; spy passed {record['spy']['passed']}")
    return 0 if record['passed'] else 8


if __name__ == '__main__':
    sys.exit(main())
