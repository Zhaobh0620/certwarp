"""LP-A item 2 (runs/20261119-lpba/任务书_A_数据准入HI与Atropos.md): the LPBA40 stage -- the preregistration's §2.2 P1-P7 -- for one of the two modes, 'lpba40_test' (the 17
pairs of splits/lpba40_test_v1/test_pairs.tsv) or 'lpba40_feas' (S01<-S02, S02<-S01).  Contract C26.5-2 (L27), C26.2, C26.3-5; prereg §2.2, §7.2 (data preparation row),
§8.5, §12 item 5.  The structure is experiments/holdout/prep_stage.py's (no label read, a one-shot lock, the spy, a dry run); nothing of it is changed.

What is read (the spy allows nothing else; lpba_common.Policy: hp_core's spy, the name rule widened to 'label'):
  the code tree: the three lists, the LPBA allowlist (pinned path and sha256), the preregistration, the frozen files, the c record (pinned sha256), the guard's sealed lists;
  /mnt/d/data/LPBA40/nii/subjects/<s>/delineation/image.nii.gz of the mode's subjects only (opened after the admission, after the lock); no label file is opened or hashed,
  no other file of /mnt/d/data, no file of the old stage_LPBA40_v1 (C26.5-1 禁读); the run's own stage directory (its mm copies read back, its pair.npz loaded back).
What is computed, per pair, from the preregistration's §2.2 word for word (parameters fixed before any truth is read):
  P1 mm_copy: the image copied to <stage>/_nifti_mm/<s>_image.nii.gz with only the xyzt units set to mm (nibabel 5.4.0: Nifti1Image(np.asanyarray(dataobj), affine,
     header.copy()), header.set_xyzt_units('mm'); archive/seq_vp_3d/scripts/prepare_stage_3d_lpba40.py with_mm, the image only); the copy's data checked bit for bit
     against the original's (dtype, shape, bytes), its affine, qform and sform (with their codes) equal to the original's; both files' sha256 recorded.  Once per subject.
  P2 jd_curl.three_d.nifti.load_scalar(<copy>) (labels=False): RAS storage, zyx, no resampling.
  P3 v = np.flip(a, 2).copy(); the grid (shape, spacing, origin) is P2's.
  P4 both images on one grid (require_common_grid), shape (181, 217, 181), spacing 1 mm, values finite and >= 0; else the run stops (StageCheckFailed).
  P5 p = np.percentile(v[v > 0], 99.5) (linear), img = v / p * c, no clipping; c from the committed c record (lpba_common.load_c_record: its sha256 pinned, the value read
     from its float.hex).  p and c recorded.
  P6 pair.npz with exactly fixed, moving (float64, zyx) and an all-zero initial_u (float64, (nz, ny, nx, 3)), np.savez_compressed; manifest.json with only the grid
     (shape_zyx, spacing_xyz, origin_xyz, unit), the subjects, sources.<side>.sha256 = the input_sha256, source_files.<side> (the original and the mm copy: path and file
     sha256), p and c, the flip, weights (experiments/he/firewall_c.frozen_weights) -- and pair_sha256, which bayesreg.io.pair.load_pair requires (C03.1); nothing derived
     from labels or naming them (no label_sets, no baseline_label_metrics).
  P7 input_sha256 = the sha256 of the stage array itself (float64, C order; he_core.array_sha256).  After a pair is written it is loaded back through
     bayesreg.io.pair.load_pair with the LPBA allowlist (the readers' path, admission included): both arrays' sha256 must be the manifest's sources.<side>.sha256.  P2-P5
     run afresh for every pair (from the mm copy), so a subject in two pairs (the feasibility pairs) gets its array twice: the two must be the same bit for bit, else the
     run stops (prereg §2.2 P7, §7.2).
The stage: <roots(pair_set).stage>/ (new): _nifti_mm/, LPBA40_<f>__LPBA40_<m>/{pair.npz, manifest.json}, SHA256SUMS of every file written (last); then prep_record.json
(the run's record, spy included).  A failure after the lock leaves prep_stopped.json and the partial stage (kept, not rerun: project rules, hard rule 4); no resume here.

Order of a run: 1. the test-sandbox guard; 2. the code (lpba_common.check_code: one clean tree; a formal run also rule one, C26.9-4, LP-C2's lpba_common.rule_one --
it refuses unless it passes); 3. the spy installed (the guard's cache of the sealed lists empty, C23.6-5); 4. the admission (lpba_common.admit: pins, the pinned LPBA allowlist, the mode's
names, sealed lists, check_subject per subject: L27); 5. the c record; 6. the stage root new, the lock absent, the hold file absent; 7. the machine gate and the disk guard;
8. the lock (O_EXCL); 9. the images allowed one by one; the stage; 10. the record.

    python -B experiments/lpba/prep_lpba.py --pair-set lpba40_feas --dry-run --out DIR
        steps 1-6 without the gates and the lock: the admission and the pins; each planned image only stat'ed (exists, size); nothing written but DIR/dry_run.json.
    python -B experiments/lpba/prep_lpba.py --pair-set lpba40_feas      (unit tag-lp-feas-prep; a formal run: --pair-set lpba40_test, unit tag-lp-prep)
Exit codes: 0 done (or a dry run that passed), 2 usage, 8 refused, stopped by the spy or a check failed (the record kept), 1 any other failure.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import lpba_common as lc  # noqa: E402

hc, hp = lc.hc, lc.hp

from jd_curl.three_d.nifti import file_sha256, load_scalar, require_common_grid  # noqa: E402
from jd_curl.validation import InputValidationError  # noqa: E402
from bayesreg.io.guard import SubjectSealed  # noqa: E402

SCHEMA = 'bayesreg.lpba40.prep.v1'
SCRIPT = 'experiments/lpba/prep_lpba.py'
UNIT_TOOL = 'prep'      # the unit name's last part (tag-lp-[feas-]prep); not an authorization tool: holdout_common's TOOL names those (test_pinned_code); LP-C2 integration
LOCK_NAME = 'lpba_prep.lock'
MM_DIR = '_nifti_mm'
EXPECTED_SHAPE_ZYX = (181, 217, 181)                # prereg §2.2 P4
EXPECTED_SPACING_XYZ = (1., 1., 1.)
PERCENTILE = 99.5                                   # prereg §2.2 P5 (numpy's default linear method)
SIDES = ('fixed', 'moving')
SUMS, RECORD, STOPPED = 'SHA256SUMS', 'prep_record.json', 'prep_stopped.json'
RUNTIME_S = 3600
NEED_HOME_GIB = dict(lpba40_test=4., lpba40_feas=1.)    # 〔估算〕17 pairs of two compressed float64 181x217x181 arrays and 34 mm copies, with room
FLIP = dict(rule='v = np.flip(a, 2).copy()', axis=2, grid='the grid metadata (shape, spacing, origin) is P2\'s, unchanged by the flip', source='prereg §2.2 P3, §2.3')
INTENSITY_RULE = 'p = np.percentile(v[v > 0], 99.5) (linear); img = v / p * c; no clipping (prereg §2.2 P5)'


def launch_line(pair_set, *, code=None, log=None):
    """The start string of this entry's unit for `pair_set` (hp_core.launch_line: its default MemoryMax 8G, swap 0, RuntimeMaxSec RUNTIME_S): the dry run prints it; the feasibility chain runs it
    with the chain's log and wraps only the command with the end probe (LP-C2 integration review minor 3: the feasibility unit is the formal one's)."""
    return hp.launch_line(unit=lc.unit_name(pair_set, UNIT_TOOL), script=SCRIPT, args=['--pair-set', pair_set], runtime_s=RUNTIME_S, code=code, log=log)


class StageCheckFailed(Exception):
    """A check of P1-P7 failed: the run stops (record kept)."""


def say(text):
    print(text, flush=True)


array_sha256 = lc.array_sha256                 # review minor 4: he_core.array_sha256 itself (prereg §2.2 P7)


def dump(path, value):
    """prep_stage.dump: indent 2, no NaN, a temporary file in the same directory replaced into place; never over an existing file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'{path} exists: stage files are never overwritten')
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda obj: obj.item() if isinstance(obj, np.generic) else str(obj))+'\n')
    temp.replace(path)


# ---- P1-P5 ----------------------------------------------------------------------------------------------------------------------------------------
def mm_copy(src, dst):
    """P1 (prereg §2.2; with_mm of prepare_stage_3d_lpba40.py, the image only): the copy with only the xyzt units set to mm; data bit for bit, affine, qform and sform (and
    their codes) checked equal to the original's; both sha256 recorded.  StageCheckFailed otherwise.  Never over an existing file."""
    import nibabel as nib
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(f'{dst} exists: an mm copy is written once')
    original = nib.load(str(src))
    zooms = [float(z) for z in original.header.get_zooms()[:3]]
    if any(abs(z-1.) > 1e-6 for z in zooms):
        raise StageCheckFailed(f'{src}: voxel size {zooms}, 1 mm expected (with_mm)')
    data = np.asanyarray(original.dataobj)
    out = nib.Nifti1Image(data, original.affine, original.header.copy())
    out.header.set_xyzt_units('mm')
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(dst))
    copy = nib.load(str(dst))
    back = np.asanyarray(copy.dataobj)
    problems = []
    if not (back.dtype == data.dtype and back.shape == data.shape and back.tobytes() == data.tobytes()):
        problems.append('the data differ')
    if not np.array_equal(copy.affine, original.affine):
        problems.append('the affine differs')
    for name, get in (('qform', lambda im: im.header.get_qform(coded=True)), ('sform', lambda im: im.header.get_sform(coded=True))):
        (a, ca), (b, cb) = get(original), get(copy)
        if ca != cb or (a is None) != (b is None) or (a is not None and not np.array_equal(a, b)):
            problems.append(f'the {name} or its code differs')
    if copy.header.get_xyzt_units()[0] != 'mm':
        problems.append(f'the copy\'s spatial unit is {copy.header.get_xyzt_units()[0]}')
    if problems:
        raise StageCheckFailed(f'{src} -> {dst}: {problems} (P1)')
    return dict(original=dict(path=str(src), sha256=file_sha256(src)), mm_copy=dict(path=str(dst), sha256=file_sha256(dst)),
                units_before=list(original.header.get_xyzt_units()), units_after=list(copy.header.get_xyzt_units()), data_bitwise=True, affine_equal=True,
                qform_sform_equal=True, dtype=str(data.dtype), native_shape_xyz=list(data.shape))


def stage_image(copy_path):
    """P2 and P3: load_scalar(<mm copy>) (labels False) and v = np.flip(a, 2).copy(); the volume (its grid) is P2's.  Returns (v, volume)."""
    volume = load_scalar(copy_path)
    v = np.flip(volume.values, 2).copy()
    return v, volume


def check_grid(volumes, arrays, name):
    """P4: one grid (require_common_grid), shape (181, 217, 181), spacing 1 mm, values finite and >= 0.  Returns the grid."""
    try:
        grid = require_common_grid(*volumes)
    except InputValidationError as exc:
        raise StageCheckFailed(f'{name}: {exc} (P4)') from None
    if tuple(grid.shape) != EXPECTED_SHAPE_ZYX or tuple(float(s) for s in grid.spacing) != EXPECTED_SPACING_XYZ:
        raise StageCheckFailed(f'{name}: grid {grid.shape} spacing {grid.spacing}, expected {EXPECTED_SHAPE_ZYX} and {EXPECTED_SPACING_XYZ} (P4)')
    for v in arrays:
        if v.shape != tuple(grid.shape) or not np.all(np.isfinite(v)) or np.any(v < 0):
            raise StageCheckFailed(f'{name}: an image is not finite and >= 0 on the grid (P4)')
    return grid


def scale(v, c):
    """P5: p = np.percentile(v[v > 0], 99.5) (linear), img = v / p * c, no clipping.  Returns (img, p)."""
    brain = v[v > 0]
    if brain.size == 0:
        raise StageCheckFailed('P5: no voxel > 0')
    p = float(np.percentile(brain, PERCENTILE))
    if not np.isfinite(p) or p <= 0:
        raise StageCheckFailed(f'P5: p99.5 {p} is not a positive number')
    img = v / p * c
    return img, p


def grid_record(grid):
    return dict(shape_zyx=[int(n) for n in grid.shape], spacing_xyz=[float(s) for s in grid.spacing], origin_xyz=[float(a[0]) for a in grid.axes], unit=grid.unit)


def frozen_weights(grid):
    import firewall_c
    return firewall_c.frozen_weights(grid)


def write_pair(stage_root, fixed, moving, *, copies, c, c_record):
    """P2-P7 for one pair: both images afresh from their mm copies, the grid checks, the scaling, pair.npz and manifest.json.  Returns the pair's record."""
    name = lc.pair_name(fixed, moving)
    out = Path(stage_root)/name
    out.mkdir()
    arrays, volumes, intensity = {}, [], {}
    for side, subject in zip(SIDES, (fixed, moving)):
        v, volume = stage_image(copies[subject]['mm_copy']['path'])
        volumes.append(volume)
        arrays[side] = v
    grid = check_grid(volumes, [arrays[s] for s in SIDES], name)
    sources, files = {}, {}
    for side, subject in zip(SIDES, (fixed, moving)):
        img, p = scale(arrays[side], c)
        if not np.all(np.isfinite(img)) or np.any(img < 0):
            raise StageCheckFailed(f'{name}: {side} after P5 is not finite and >= 0')
        arrays[side] = img
        intensity[side] = dict(subject=subject, p99_5=lc.number(p))
        sources[side] = dict(sha256=array_sha256(img))
        files[side] = dict(subject=subject, original=dict(copies[subject]['original']), mm_copy=dict(copies[subject]['mm_copy']))
    arrays['initial_u'] = np.zeros(tuple(grid.shape)+(3,))
    np.savez_compressed(out/'pair.npz', fixed=arrays['fixed'], moving=arrays['moving'], initial_u=arrays['initial_u'])
    manifest = dict(**grid_record(grid), fixed_subject=fixed, moving_subject=moving, sources=sources, source_files=files,
                    intensity=dict(rule=INTENSITY_RULE, c=dict(lc.number(c), record=dict(path=c_record['path'], sha256=c_record['sha256'])), p=intensity),
                    flip=FLIP, weights=frozen_weights(grid), pair_sha256=file_sha256(out/'pair.npz'))
    dump(out/'manifest.json', manifest)
    return dict(pair=name, fixed=fixed, moving=moving, pair_sha256=manifest['pair_sha256'], manifest_sha256=file_sha256(out/'manifest.json'),
                input_sha256={side: sources[side]['sha256'] for side in SIDES}, p={side: intensity[side]['p99_5'] for side in SIDES})


def read_back(stage_root, rows, allowlist):
    """P7: every pair loaded back through bayesreg.io.pair.load_pair with the LPBA allowlist; each array's sha256 is its manifest's sources.<side>.sha256, and a subject in
    two pairs has the same array in both (StageCheckFailed otherwise).  Returns the record."""
    from bayesreg.io.pair import load_pair
    seen, out = {}, {}
    for row in rows:
        pair = load_pair(Path(stage_root)/row['pair']/'pair.npz', allowlist=allowlist)
        if (pair.fixed_subject, pair.moving_subject) != (row['fixed'], row['moving']):
            raise StageCheckFailed(f"{row['pair']}: load_pair gives {pair.fixed_subject}, {pair.moving_subject}")
        sides = {}
        for side in SIDES:
            digest = array_sha256(getattr(pair, side))
            if digest != pair.source_sha256[side] or digest != row['input_sha256'][side]:
                raise StageCheckFailed(f"{row['pair']}: the {side} array read back has sha256 {digest}, the manifest {pair.source_sha256[side]} (P7)")
            subject = getattr(pair, f'{side}_subject')
            seen.setdefault(subject, set()).add(digest)
            sides[side] = digest
        out[row['pair']] = dict(keys_read=pair.provenance['keys_read'], input_sha256=sides, lpba40_guard=pair.provenance['subject_guard'].get('lpba40_guard'))
    differ = {s: sorted(d) for s, d in seen.items() if len(d) != 1}
    if differ:
        raise StageCheckFailed(f'a subject in two pairs has two different arrays: {differ} (prereg §2.2 P7, §7.2)')
    return dict(pairs=out, subjects={s: next(iter(d)) for s, d in seen.items()}, rule='load_pair with the LPBA allowlist; array sha256 = sources.<side>.sha256; one array per subject')


def stage_files(stage_root):
    root = Path(stage_root)
    return sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != SUMS)


def prepare(stage_root, pairs, *, c, c_record, allowlist, log=say):
    """The stage of the module docstring: the root made here (new), the mm copies (once per subject, in the order of first appearance), the pairs in order, the read back,
    SHA256SUMS last.  Returns the summary."""
    stage_root = Path(stage_root)
    lc.guard_write(stage_root)
    started = perf_counter()
    stage_root.mkdir(parents=True, exist_ok=False)
    copies = {}
    for subject in dict.fromkeys(s for p in pairs for s in p):
        copies[subject] = mm_copy(lc.image_path(subject), stage_root/MM_DIR/f'{subject}_image.nii.gz')
        log(f'{subject}: mm copy {copies[subject]["mm_copy"]["sha256"][:12]}')
    rows, seconds = [], {}
    for fixed, moving in pairs:
        t0 = perf_counter()
        rows.append(write_pair(stage_root, fixed, moving, copies=copies, c=c, c_record=c_record))
        seconds[rows[-1]['pair']] = round(perf_counter()-t0, 3)
        log(f"{rows[-1]['pair']}: {seconds[rows[-1]['pair']]:.1f} s")
    back = read_back(stage_root, rows, allowlist)
    sums = ''.join(f'{file_sha256(stage_root/n)}  {n}\n' for n in stage_files(stage_root))
    (stage_root/SUMS).write_text(sums, encoding='utf-8')
    return dict(stage=str(stage_root), pairs=rows, subjects=copies, read_back=back, seconds=seconds, elapsed_seconds=round(perf_counter()-started, 3),
                sha256sums_sha256=file_sha256(stage_root/SUMS), n_files=len(sums.splitlines()))


# ---- a run ----------------------------------------------------------------------------------------------------------------------------------------
def policy(pair_set, *, out=None):
    r = lc.roots(pair_set)
    lock = r.locks/LOCK_NAME
    return lc.Policy(name=f'lpba prep ({pair_set})', read_exact=lc.code_reads(), write_inside=[r.stage] if out is None else [out], write_exact=[lock], results=hc.RESULTS)


def run(pair_set, *, code, dry_run=False, out=None, log=say):
    """The order of the module docstring.  Returns the run's record (a dry run's: the admission and the planned files)."""
    r = lc.roots(pair_set)
    stage_root, lock = r.stage, r.locks/LOCK_NAME
    lc.guard_write(stage_root, lock)
    unit = lc.unit_name(pair_set, UNIT_TOOL)
    problem = lc.formal_code_problem(code, pair_set, dry_run=dry_run)
    if problem:
        raise lc.Refused('code', problem)
    record = dict(schema=SCHEMA, tool=SCRIPT, pair_set=pair_set, started_utc=lc.utc(), stage=str(stage_root), lock=str(lock), dry_run=dry_run, code=code, unit=unit)
    record['preload'] = lc.preload()
    with lc.spying(policy(pair_set, out=out if dry_run else None)) as spy:
        admission = lc.admit(pair_set)
        record['admission'] = admission.record
        c = lc.load_c_record()
        record['c_record'] = c.record
        if stage_root.exists():
            raise lc.Refused('output', f'{stage_root} exists: the stage is written once, never overwritten')
        if lock.exists():
            raise lc.Refused('once', f'{lock} exists: the data preparation of {pair_set} ran; it runs once')
        record['hold'] = lc.check_hold(pair_set)
        subjects = list(dict.fromkeys(s for p in admission.pairs for s in p))
        images = [lc.image_path(s) for s in subjects]
        if dry_run:
            planned = []
            for path in images:
                try:
                    st = os.stat(path)
                    planned.append(dict(path=str(path), exists=True, size=st.st_size))
                except OSError:
                    planned.append(dict(path=str(path), exists=False))
            record['planned_images'] = planned
            record['launch'] = launch_line(pair_set)
            record['spy'] = spy.record()
            return record
        record['machine'] = hc.machine_gate(own_units=(unit,))
        record['disk'] = hc.disk_guard(need_d_gib=0, need_home_gib=NEED_HOME_GIB[pair_set])
        line = hc.lock_line(tool=SCRIPT, pair_set=pair_set, stage=str(stage_root), prereg_sha256=lc.PREREG_SHA256, c_record_sha256=c.sha256,
                            allowlist_sha256=admission.record['allowlist']['sha256'], code_commit=(code or {}).get('commit'), event='first_run')
        record['lock_line'] = hc.take_lock(lock, line)
        spy.policy.allow_read(images)
        try:
            record['stage_summary'] = prepare(stage_root, admission.pairs, c=c.c, c_record=dict(path=lc.C_RECORD_REL, sha256=c.sha256), allowlist=admission.allowlist, log=log)
        except BaseException as exc:
            record.update(stopped=f'{type(exc).__name__}: {exc}', spy=spy.record(expect=images), ended_utc=lc.utc())
            spy.remove()
            hc.write_json_once((stage_root if stage_root.is_dir() else lock.parent)/STOPPED, record)
            raise
        record['spy'] = spy.record(expect=images+[lc.repo_root()/lc.C_RECORD_REL])
    record['ended_utc'] = lc.utc()
    record['passed'] = bool(record['spy']['passed'])
    hc.write_json_once(stage_root/RECORD, record)
    return record


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog=SCRIPT)
    ap.add_argument('--pair-set', required=True, choices=lc.PAIR_SETS)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args(argv)
    if a.dry_run and a.out is None:
        ap.error('--dry-run writes its record into --out DIR (a new or empty directory outside the results roots, e.g. mktemp -d)')
    return a


def main(argv=None, *, code=None, out=say):
    a = parse_args(argv)
    if a.dry_run:
        problem = lc.dry_run_out_problem(a.out)
        if problem:
            out(f'{problem}: a dry run writes into a new or empty directory outside the results root and the code trees')
            return 2
    elif not sys.flags.dont_write_bytecode:
        out('refused: run with python -B (or PYTHONDONTWRITEBYTECODE): the code tree gets no __pycache__')
        return 8
    try:
        code = lc.check_code(hp.code_identity() if code is None else code, a.pair_set, dry_run=a.dry_run)
        record = run(a.pair_set, code=code, dry_run=a.dry_run, out=a.out)
    except hc.Refused as exc:
        out(f'refused ({exc.stage}): {exc}')
        if a.dry_run:
            a.out.mkdir(parents=True, exist_ok=True)
            hc.write_json_once(a.out/'dry_run.json', dict(schema=SCHEMA+'.dry_run', refused=dict(stage=exc.stage, message=str(exc)), utc=lc.utc(), spy=getattr(exc, 'hp_spy', None)))
        return 8
    except hp.Breach as exc:
        out(f'stopped by the spy: {exc}')
        return 8
    except (StageCheckFailed, InputValidationError, SubjectSealed) as exc:
        out(f'stopped: {type(exc).__name__}: {exc}')
        return 8
    if a.dry_run:
        a.out.mkdir(parents=True, exist_ok=True)
        hc.write_json_once(a.out/'dry_run.json', dict(record, schema=SCHEMA+'.dry_run'))
        out(f"dry run passed ({a.pair_set}): {len(record['admission']['subjects'])} subjects admitted, {sum(p['exists'] for p in record['planned_images'])}/"
            f"{len(record['planned_images'])} images there; spy breaches {len(record['spy']['breaches'])}; launch: {record['launch']}")
        return 0
    out(f"done: {len(record['stage_summary']['pairs'])} pairs in {record['stage']}; spy passed {record['spy']['passed']}")
    return 0 if record['passed'] else 8


if __name__ == '__main__':
    sys.exit(main())
