"""LP-D (a) (runs/20261119-lpba/任务书_D_开发集逐位对照.md; prereg §12 item 4 (a), §8.1 item 10; contract C26.5-11 (a), C26.5-1 (the one exception to 禁读), C26.4 L38, C26.5-12 N55):
the new stage's S01 <- S02 against the old stage_LPBA40_v1/LPBA40_S01__LPBA40_S02/pair.npz.  CPU; after the feasibility's data preparation (prep_lpba.py --pair-set lpba40_feas).

What it checks, per side (fixed S01, moving S02), with the preregistration's words:
  v_new  P3's output, recomputed here with prep_lpba's own code (stage_image: P2 load_scalar of the stage's mm copy _nifti_mm/<s>_image.nii.gz -- P1's output, its sha256 the stage's
         SHA256SUMS and prep_record.json's -- then P3 np.flip(., 2).copy()): prep_lpba.py does not keep v_new;
  p, P5 computed here word for word (literal_p5; review minor 1: not with the code under check): p = float(np.percentile(v_new[v_new > 0], 99.5)) (linear), img = v_new / p * c (c the
         committed c record's, lpba_common.load_c_record).  Then: p's float.hex is prep_record.json's; prep_lpba.scale(v_new, c) gives the same p and bitwise the same img; img is bitwise the
         new stage's array (pair.npz through load_pair with the LPBA allowlist); img's sha256 is prep_record.json's input_sha256 and the manifest's sources.<side>.sha256
         ("核重算的 sha256 等于 prep 记录的");
  old    np.clip(np.flip(v_new, 2) / p, 0, 1), with the literal p, bitwise the old stage's array of the same side (dtype, shape, C-order bytes).
passed = every one of these on both sides, and the spy passed.  The grids (P2's, the new manifest's, the old manifest's) are recorded, not judged.

What it reads (the spy allows nothing else): the code-tree files of lpba_common.admit (the pins, the lists, the LPBA allowlist, the sealed lists, the c record); the new feasibility stage's
SHA256SUMS, prep_record.json, both feasibility pairs' pair.npz and manifest.json (lpba_common.check_stage hashes them) and the mm copies of S01, S02; of the old stage only
LPBA40_S01__LPBA40_S02/manifest.json (through bayesreg.io.pair._read_manifest's whitelisting decoder, inside load_pair: no label_sets, no baseline_label_metrics) and its pair.npz (through
load_pair: its container sha256, then fixed, moving and the all-zero initial_u it checks; never fixed_lpba56, moving_lpba56 or the brain members).  No other file of stage_LPBA40_v1, no
file of /mnt/d/data, no label file.
The admission (lpba_common.admit('lpba40_feas', pairs=[(S01, S02)]): the pins, the pinned LPBA allowlist, a row of feasibility_pairs.tsv, the sealed lists, check_subject) comes before any
file of a subject; the old pair is then loaded through load_pair with the same LPBA allowlist (its directory name checked first: C26.3-6).

    python -B experiments/lpba/devcheck/lpdev_a.py dry-run --out DIR      the admission and the pins; the planned files stat'ed; nothing written but DIR/dry_run.json
    python -B experiments/lpba/devcheck/lpdev_a.py run                    once (unit tag-lp-feas-deva): <feasibility root>/devcheck_a/devcheck_a.json
Exit codes: lpdev_common.EXIT.
"""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import lpdev_common as dc  # noqa: E402

lc, hc, hp = dc.lc, dc.hc, dc.hp
import prep_lpba as prep  # noqa: E402

import numpy as np  # noqa: E402

SCHEMA = 'bayesreg.lpba40.devcheck_a.v1'
SCRIPT = 'experiments/lpba/devcheck/lpdev_a.py'
UNIT_TOOL = 'deva'      # the unit name's last part (tag-lp-feas-deva); not an authorization tool: holdout_common's TOOL names those (test_pinned_code); LP-C2 integration follow-up
PAIR_SET = 'lpba40_feas'
PAIR = ('S01', 'S02')
SIDES = ('fixed', 'moving')
OLD_STAGE = hc.MAIN_REPO/'data'/'three_d'/'stage_LPBA40_v1'      # paper 1's LPBA40 stage (archive/seq_vp_3d/scripts/prepare_stage_3d_lpba40.py); only its pair S01 <- S02 is read (C26.5-1)
OLD_PAIR = 'LPBA40_S01__LPBA40_S02'
OLD_RULE = 'np.clip(np.flip(v_new, 2) / p, 0, 1)'                 # prereg §12 item 4 (a): paper 1 divided the unflipped image by its p99.5 and clipped to [0, 1]
RECORD, STOPPED = 'devcheck_a.json', 'devcheck_a_stopped.json'
RUNTIME_S = 1800
PERCENTILE = 99.5                                                  # prereg §2.2 P5, written here (not taken from prep_lpba)


def launch_line(*, code=None, log=None):
    """The start string of this check's unit tag-lp-feas-deva (hp_core.launch_line: MemoryMax 8G, swap 0, RuntimeMaxSec RUNTIME_S): the dry run prints it; the feasibility chain
    runs it with the chain's log and wraps only the command with the end probe (LP-C2 integration review minor 3).  It runs in the feasibility only (no formal unit)."""
    return hp.launch_line(unit=lc.unit_name(PAIR_SET, UNIT_TOOL), script=SCRIPT, args=['run'], runtime_s=RUNTIME_S, memory_max='8G', code=code, log=log)


def old_paths():
    return [Path(OLD_STAGE)/OLD_PAIR/'pair.npz', Path(OLD_STAGE)/OLD_PAIR/'manifest.json']


def policy(*, out=None):
    r = dc.roots()
    feas_pairs = lc.table(PAIR_SET)[1]
    return dc.Policy(name='lpba devcheck (a)', read_exact=lc.code_reads()+lc.stage_paths(PAIR_SET, feas_pairs)+old_paths(), write_inside=[r.a if out is None else out],
                     write_exact=[r.a_lock], list_inside=[r.feas.stage], results=hc.RESULTS)


def prep_row(prep_record, name):
    rows = [row for row in (prep_record.get('stage_summary') or {}).get('pairs') or [] if row.get('pair') == name]
    if len(rows) != 1:
        raise dc.Stopped(f'prep_record.json lists {len(rows)} rows of {name}')
    return rows[0]


def literal_p5(v_new, c):
    """Prereg §2.2 P5 word for word, written in this tool (review minor 1): p = np.percentile(v_new[v_new > 0], 99.5) (linear), img = v_new / p * c, no clipping.  (img, p)."""
    p = float(np.percentile(v_new[v_new > 0], PERCENTILE))
    return v_new / p * c, p


def side_check(side, subject, *, stage, prep_record, row, new_pair, old_pair, c):
    """The checks of the module docstring for one side.  Returns its record (with passed)."""
    copy = (prep_record['stage_summary']['subjects'].get(subject) or {}).get('mm_copy') or {}
    expected_copy = Path(stage)/lc.MM_DIR/f'{subject}_image.nii.gz'
    copy_sha = hc.sha256_file(expected_copy)
    v_new, volume = prep.stage_image(expected_copy)                      # P2, P3: prep_lpba's own code (decision_log LP-D ruling 3)
    img, p = literal_p5(v_new, c)                                        # P5: the preregistration's formula, written here
    prep_img, prep_p = prep.scale(v_new, c)                              # the code under check, compared with it
    recomputed_sha = lc.array_sha256(img)
    new_array, old_array = getattr(new_pair, side), getattr(old_pair, side)
    recorded_p = (row.get('p') or {}).get(side) or {}
    p5 = dc.arrays_equal(img, new_array)
    prep_scale = dc.arrays_equal(img, prep_img)
    del prep_img
    old = dc.arrays_equal(np.clip(np.flip(v_new, 2) / p, 0, 1), old_array)
    checks = dict(mm_copy_is_prep_s=Path(copy.get('path', '')) == expected_copy and copy.get('sha256') == copy_sha, p_hex_is_prep_s=p.hex() == recorded_p.get('hex'),
                  prep_scale_is_literal=prep_p.hex() == p.hex() and prep_scale['bitwise_equal'], p5_bitwise=p5['bitwise_equal'],
                  recomputed_sha256_is_prep_s=recomputed_sha == (row.get('input_sha256') or {}).get(side),
                  recomputed_sha256_is_manifest_s=recomputed_sha == new_pair.source_sha256[side], old_bitwise=old['bitwise_equal'])
    grid = dict(p2=dict(shape_zyx=list(volume.grid.shape), spacing_xyz=[float(s) for s in volume.grid.spacing], origin_xyz=[float(a[0]) for a in volume.grid.axes]),
                new_manifest=dict(new_pair.grid_metadata), old_manifest=dict(old_pair.grid_metadata))
    grid['same'] = grid['p2']['shape_zyx'] == grid['new_manifest']['shape_zyx'] == grid['old_manifest']['shape_zyx'] and \
        grid['new_manifest'] == grid['old_manifest']
    return dict(side=side, subject=subject, mm_copy=dict(path=str(expected_copy), sha256=copy_sha, prep_record=dict(copy)), p=lc.number(p), prep_p=dict(recorded_p),
                prep_scale=dict(prep_scale, p=lc.number(prep_p)), p5_rule='p = float(np.percentile(v_new[v_new > 0], 99.5)); img = v_new / p * c (lpdev_a.literal_p5)',
                v_new_sha256=lc.array_sha256(v_new), recomputed_p5_sha256=recomputed_sha, prep_input_sha256=(row.get('input_sha256') or {}).get(side),
                manifest_source_sha256=new_pair.source_sha256[side], p5=p5, old_rule=OLD_RULE, old=old, grid=grid, checks=checks, passed=all(checks.values()))


def compare(admission, c, *, log=print):
    """After the lock: the new pair and the old pair through load_pair (the LPBA allowlist), prep_record.json, both sides.  Returns the comparison's record."""
    from bayesreg.io.pair import load_pair
    r = dc.roots()
    stage = r.feas.stage
    name = lc.pair_name(*PAIR)
    prep_record = dc.read_json(stage/lc.PREP_RECORD)
    row = prep_row(prep_record, name)
    new_pair = load_pair(stage/name/'pair.npz', allowlist=admission.allowlist)
    old_pair = load_pair(Path(OLD_STAGE)/OLD_PAIR/'pair.npz', allowlist=admission.allowlist)
    for label, pair in (('new', new_pair), ('old', old_pair)):
        if (pair.fixed_subject, pair.moving_subject) != PAIR:
            raise dc.Stopped(f'the {label} pair names {pair.fixed_subject} <- {pair.moving_subject}, not {PAIR[0]} <- {PAIR[1]}')
    sides = {}
    for side, subject in zip(SIDES, PAIR):
        sides[side] = side_check(side, subject, stage=stage, prep_record=prep_record, row=row, new_pair=new_pair, old_pair=old_pair, c=c.c)
        log(f"{side} {subject}: old bitwise {sides[side]['old']['bitwise_equal']} ({sides[side]['old'].get('n_elements_differ')} differ); P5 bitwise {sides[side]['p5']['bitwise_equal']}; "
            f"checks {sides[side]['checks']}")
    files = [dc.file_record(stage/lc.PREP_RECORD), dc.file_record(stage/lc.STAGE_SUMS), *(dc.file_record(p) for p in old_paths()),
             dc.file_record(stage/name/'pair.npz'), dc.file_record(stage/name/'manifest.json'), *(dict(path=s['mm_copy']['path'], sha256=s['mm_copy']['sha256']) for s in sides.values())]
    return dict(pair=name, old_pair=str(Path(OLD_STAGE)/OLD_PAIR), sides=sides, new_keys_read=list(new_pair.provenance['keys_read']), old_keys_read=list(old_pair.provenance['keys_read']),
                old_guard=old_pair.provenance['subject_guard'], new_guard=new_pair.provenance['subject_guard'], files_read=files,
                passed=all(s['passed'] for s in sides.values()))


def run(*, code, dry_run=False, out=None, log=print):
    """The order: the code; the spy; the admission (lpba_common.admit: pins, the LPBA allowlist, the row, sealed lists, check_subject); the c record; the hold; the output new, the lock absent;
    the stage (lpba_common.check_stage: prep_record.json passed, SHA256SUMS); the old pair's two files there; (dry run: stop here); the machine gate and the disks; the lock; the comparison;
    the record.  Returns the record."""
    r = dc.roots()
    out_dir, lock = r.a, r.a_lock
    lc.guard_write(out_dir, lock)
    unit = lc.unit_name(PAIR_SET, UNIT_TOOL)
    record = dict(schema=SCHEMA, tool=SCRIPT, task=dc.TASK, contract=dc.CONTRACT, pair_set=PAIR_SET, pair=lc.pair_name(*PAIR), old_stage=str(OLD_STAGE), started_utc=dc.utc(),
                  out=str(out_dir), lock=str(lock), dry_run=dry_run, code=code, unit=unit)
    record['preload'] = lc.preload()
    feas_pairs = lc.table(PAIR_SET)[1]
    with dc.spying(policy(out=out if dry_run else None)) as spy:
        admission = lc.admit(PAIR_SET, pairs=[PAIR])
        record['admission'] = admission.record
        c = lc.load_c_record()
        record['c_record'] = c.record
        record['hold'] = lc.check_hold(PAIR_SET)
        if out_dir.exists():
            raise lc.Refused('output', f'{out_dir} exists: the check runs once')
        if lock.exists():
            raise lc.Refused('once', f'{lock} exists: the check ran; it runs once')
        if dry_run:
            record['planned'] = dc.planned(lc.stage_paths(PAIR_SET, feas_pairs)+old_paths())
            record['launch'] = launch_line()
            record['spy'] = spy.record()
            return record
        record['stage_check'] = lc.check_stage(PAIR_SET, feas_pairs)
        missing = [str(p) for p in old_paths() if not p.is_file()]
        if missing:
            raise lc.Refused('input', f'the old pair is not there: {missing}')
        record['machine'] = hc.machine_gate(own_units=(unit,))
        record['disk'] = hc.disk_guard(need_d_gib=0, need_home_gib=.1)
        line = hc.lock_line(tool=SCRIPT, pair_set=PAIR_SET, out=str(out_dir), prereg_sha256=lc.PREREG_SHA256, c_record_sha256=c.sha256, stage_sha256sums=record['stage_check']['sha256sums_sha256'],
                            allowlist_sha256=admission.record['allowlist']['sha256'], code_commit=(code or {}).get('commit'), event='first_run')
        record['lock_line'] = hc.take_lock(lock, line)
        with dc.after_lock(record, spy, out_dir, lock, STOPPED):        # review minor 5: everything after the lock, the record's write included
            out_dir.mkdir(parents=True)
            record['comparison'] = compare(admission, c, log=log)
            expect = old_paths()+[r.feas.stage/lc.pair_name(*PAIR)/'pair.npz']+[r.feas.stage/lc.MM_DIR/f'{s}_image.nii.gz' for s in PAIR]
            record['spy'] = spy.record(expect=expect)
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
    sub.add_parser('run')
    return ap.parse_args(argv)


def main(argv=None, *, code=None, out=print):
    a = parse_args(argv)
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
    record, exit_code = dc.guarded(lambda: run(code=dc.check_code(hp.code_identity() if code is None else code, dry_run=dry), dry_run=dry, out=a.out if dry else None, log=out),
                                   dry=dry, out_dir=a.out if dry else None, schema=SCHEMA, out=out)
    if exit_code is not None:
        return exit_code
    if dry:
        dc.write_dry_run(a.out, dict(record, schema=SCHEMA+'.dry_run'))
        out(f"dry run passed: {len(record['admission']['subjects'])} subjects admitted; planned files there {sum(p['exists'] for p in record['planned'])}/{len(record['planned'])}; "
            f"spy breaches {len(record['spy']['breaches'])}; launch: {record['launch']}")
        return dc.EXIT['passed']
    out(f"done: {record['conclusion']} ({record['out']}/{RECORD})")
    return dc.EXIT['passed'] if record['passed'] else dc.EXIT['not_passed']


if __name__ == '__main__':
    sys.exit(main())
