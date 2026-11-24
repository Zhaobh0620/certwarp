"""HB: the step-8 input lock and the move of the method arm to D: (preregistration §5.2 steps 5, 8, §5.3; runs/20261109-hb/任务书.md items 6, 7; contract C22.H1-1, -3).

    python -B experiments/hb/hb_store.py move [--pairs P,...] [--commit SHA]    after step 5: each finalized method cell moved whole to /mnt/d/seq_vp_tag_holdout_B_v1/method/<pair>/
    python -B experiments/hb/hb_store.py move-resume --pair P --authorization REL [--commit SHA]
                                                                                  a half-moved cell (move.json passed false) moved on, once, by a committed resume record (tool hb, arm move)
    python -B experiments/hb/hb_store.py input-lock [--out REL] [--commit SHA]    step 8: the sha256 of the 100 subjects' probs36.npz, structure_groups.npz, provenance.json (HI's hold-out output) and of
                                                                                  every method cell's t88/field_0600.npz, written to REL in the code tree (new); the lead commits it

field_0600.npz is not deleted (decision_log 2026-11-10 ruling (5): the repair ablation's input; it moves with the cell to D:); the step-8 command delete-fields is gone, and pq-done
with it (decision_log 2026-11-10 ~5:29 AM CST answer (4): no caller).  repair/certified.npz is still deleted at step 4 after the bitwise check (hb_method.certified_check).

The move (C22.H1-1: not before step 5; holdout_common.move_files: every file's sha256 before and after, the source removed only when equal, a destination never overwritten): only a cell whose
terminal.json is the review's (finalized) and that is listed in review/finalized_outputs.json with the same terminal and review sha256, and whose destination holds no file (HB review I3: checked
before anything moves); the record holdout_B_v1/method/<pair>/move.json (once).  The disks are guarded first (D: needs the cell's size + 40 GiB).  A move that stops half way (an OSError, a sha256
that differs) leaves move.json with passed false: `move` then stops (exit 22) instead of skipping the pair, where() refuses instead of naming /home, and the repair is move-resume (a committed
resume record: lock_sha256 and previous_record are move.json's; the files the first move left on /home move on, the files it moved are checked at the destination by sha256_after; any file of the
remaining ones already at the destination refuses, to the lead, but the half copy of the file the first move was copying (problem 'copying', HB diff review n2): a regular file whose sha256 is
not the source's is renamed whole to <file>.partial_<k> (kept) and recorded with its sha256 before and after in move_resume_partials.json, then the file is copied again; the record
move_resume.json; at most once).  After the move the evaluation entry, PQ and ConvexAdam-v4seg know the files by the
finalized record and the move records (C22.H1-3 修订 7b: absolute paths are mapped by the move record, never compared).
The input lock (step 8): reads only the predictions' files (by sha256; nothing decoded) and the field files; the subjects are the pair list's (ids).  A5b item 13 (A5a review I2: HI's first run
and its resume write into one root): HI is read as the union of its runs (hb_core.hi_union: its lock and root, complete over the 100 subjects, consistent, no residue in pred/), each subject's record
subjects/<s>.json is a row (kind hi_record) and its three files' sha256 must be the record's; not so -> refused (stage input_lock).  Nothing under attempt<k>/ (the residue a resume moved out).  The fields' rows name where each file is now (the cell
on /home or on D: after the move).

LP-B (contract C26.6-7; LPBA40 preregistration 06121278... §8.2 step 7, §11; brief runs/20261119-lpba/任务书_B_方法配置C与外部臂.md item 1): an LPBA40 mode does not move (its cells stay under
<root>/work/method/, §11), and its step 7 deletes the fields:
    python -B experiments/hb/hb_store.py lpba-input-lock --pair-set M [--out REL] [--commit SHA]
        the input lock of step 7: the mode's subjects' probs36.npz, structure_groups.npz, provenance.json and HI records (hb_core.lpba_hi: hi_lpba.py's one run, complete, each record against its
        files) and every method cell's t88/field_0600.npz by sha256 and size, with the line status of the mode's finalized_outputs.json (step 5 must be finished); written once.
        Formal: to REL in the code tree (default runs/lpba40_test_v1/hb_input_lock.json: a record the formal mode's rule one lets a later commit add once, C26.9-4); the lead commits it.
        Feasibility (review I2, the lead's ruling (a)): to the feasibility root (<results>/lpba40_feas_v1/hb_input_lock.json), never into the code tree and never committed -- the whole
        feasibility chain runs on one commit (C26.9-3); --out is refused.
    python -B experiments/hb/hb_store.py lpba-delete-fields --pair-set lpba40_test --lock REL [--commit SHA]
        then, the lock committed and unchanged: each hx_certified cell's field_0600.npz whose sha256 and size are the lock's row -- its record field_deleted.json (path, sha256, size, the lock's
        path and sha256) written in the cell's records <root>/method/<pair>/ before the file goes; a fallback pair keeps every file (§8.2 step 7: 走退路的对全留); a field whose sha256 or size is not
        the lock's is kept and refuses (to the lead).  Nothing else is deleted.  The feasibility mode deletes nothing (review I2: its two pairs keep every field).
Both check the formal mode's rule one themselves (review minor 7: lpba_arms.formal_code_problem, LP-C2's rule one, C26.9-4; fail closed), whoever calls them.
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import hb_core as hb  # noqa: E402

hc, he = hb.hc, hb.he
ARM = hb.ARM
PRED_FILES = ('probs36.npz', 'structure_groups.npz', 'provenance.json')
LOCK_SCHEMA = hb.SCHEMA.format('input_lock')
DEFAULT_LOCK_REL = 'runs/holdout_B_v1/hb_input_lock.json'           # a record committed after the pinned commit: under runs/holdout_B_v1/ (decision_log 2026-11-09 df8541d)


def finalized():
    """The review's finalized_outputs.json (C22.H1-3), or None."""
    return hb.read_json(hb.review_root()/'finalized_outputs.json')


def where(pair):
    """The method cell now: the working cell (no move.json), the D: store (move.json passed, or a passed move_resume.json after it); Refused for a half-moved cell (HB review I3: never /home by
    default when the files are split; the repair is move-resume)."""
    move = hb.read_json(hb.record_dir(ARM, pair)/'move.json')
    if move is None:
        return hb.cell(pair)
    resumed = hb.read_json(hb.record_dir(ARM, pair)/'move_resume.json')
    if move.get('passed') or (resumed or {}).get('passed'):
        return Path(hc.big_dir(ARM, pair))
    raise hc.Refused('where', f"{pair}: half moved ({hb.record_dir(ARM, pair)/'move.json'} passed false{'' if resumed is None else ', move_resume.json passed false'}): repair with "
                              'hb_store.py move-resume and a committed resume record, to the lead')


def destination_files(pair):
    """The files already under the pair's D: store (HB review I3: a move starts only into an empty or absent directory)."""
    dst = Path(hc.big_dir(ARM, pair))
    return sorted(str(p.relative_to(dst)) for p in dst.rglob('*') if p.is_file() or p.is_symlink()) if dst.exists() else []


def move_pair(pair, *, final=None, disk=None):
    """One finalized method cell to D: (module docstring).  Returns the move record; Refused or InfrastructureFailure otherwise."""
    final = finalized() if final is None else final
    if (hb.record_dir(ARM, pair)/'move.json').exists():
        raise hc.Refused('move', f"{hb.record_dir(ARM, pair)/'move.json'} exists: a cell moves once")
    if final is None:
        raise hc.Refused('move', 'review/finalized_outputs.json is not there: step 5 is not finished (C22.H1-1: no move before it)')
    row = (final.get('pairs') or {}).get(pair)
    cell_dir = hb.cell(pair)
    term = hb.terminal_path(cell_dir)
    if row is None or row.get('terminal') is None or not term.is_file() or hb.sha(term) != row['terminal']['sha256']:
        raise hc.Refused('move', f'{pair}: its terminal.json is not the one finalized_outputs.json names')
    review = cell_dir/'review'/'review.json'
    if row.get('review') is not None and (not review.is_file() or hb.sha(review) != row['review']['sha256']):
        raise hc.Refused('move', f'{pair}: its review.json is not the one finalized_outputs.json names')
    record = hb.record_dir(ARM, pair)/'move.json'
    there = destination_files(pair)
    if there:
        raise hc.Refused('move', f'{pair}: {len(there)} files already at {hc.big_dir(ARM, pair)} ({there[:5]}): nothing moved (HB review I3); to the lead')
    size = sum(p.stat().st_size for p in cell_dir.rglob('*') if p.is_file())/2**30
    (disk or (lambda: hc.disk_guard(need_d_gib=size, need_home_gib=0.)))()
    out = hc.move_files(cell_dir, hc.big_dir(ARM, pair), record)
    hb.remove_empty_tree(cell_dir)
    return out


PARTIALS_RECORD = 'move_resume_partials.json'


def partial_name(target):
    """<file>.partial_<k>: the first k = 1, 2, ... not there (HB diff review n2)."""
    k = 1
    while Path(f'{target}.partial_{k}').exists() or Path(f'{target}.partial_{k}').is_symlink():
        k += 1
    return Path(f'{target}.partial_{k}')


def move_resume(pair, rel, *, disk=None):
    """A half-moved cell moved on (module docstring): the committed resume record of tool hb, arm move (holdout_common.load_resume with move.json as both the lock and the previous record; its own
    lock move.json.resume, at most once); the files the first move reported ok are at the destination with their sha256_after; none of the files left on /home is at the destination, but the
    one file move.json left at problem 'copying' (HB diff review n2: the copy D: stopped half way): its half copy at the destination, a regular file whose sha256 is not the source's (the source
    still its sha256_before), is renamed whole to <file>.partial_<k> (kept, not deleted) and recorded in move_resume_partials.json (written once: the sha256 before and after the rename, the
    source's); then holdout_common.move_files of exactly the remaining files, recorded in move_resume.json.  A destination file equal to the source, or one of another row, refuses (to the lead).
    Returns the record; Refused otherwise (nothing moved, nothing renamed)."""
    rec_dir = hb.record_dir(ARM, pair)
    first = hb.read_json(rec_dir/'move.json')
    if first is None or first.get('passed'):
        raise hc.Refused('move_resume', f'{pair}: only a cell whose move.json is there with passed false is moved on')
    resume = hc.load_resume(rel, tool=hb.TOOL, arm='move', pair=pair, pair_set='holdout', lock=rec_dir/'move.json', previous_record=rec_dir/'move.json', complete=lambda: False)
    cell_dir, dst = hb.cell(pair), Path(hc.big_dir(ARM, pair))
    problems = []
    rows = first.get('files') or []
    for row in rows:
        if row.get('ok'):
            target = Path(row['dst'])
            if not target.is_file() or hb.sha(target) != row.get('sha256_after'):
                problems.append(f"{row['file']}: not at {target} with its sha256_after")
    copying = {row['file']: row for row in rows if not row.get('ok') and row.get('problem') == 'copying'}
    remaining = sorted(str(p.relative_to(cell_dir)) for p in cell_dir.rglob('*') if p.is_file()) if cell_dir.is_dir() else []
    partials = []
    for name in remaining:
        target = dst/name
        if not (target.exists() or target.is_symlink()):
            continue
        row = copying.get(name)
        if row is None:
            problems.append(f'{name}: already at the destination')
            continue
        source_sha = hb.sha(cell_dir/name)
        half_sha = hb.sha(target) if target.is_file() and not target.is_symlink() else None
        if source_sha != row.get('sha256_before'):
            problems.append(f'{name}: the source on /home is not the sha256_before of move.json ({source_sha} vs {row.get("sha256_before")})')
        elif half_sha is None:
            problems.append(f'{name}: at the destination but not a regular file')
        elif half_sha == source_sha:
            problems.append(f'{name}: the destination file equals the source (sha256 {source_sha}): not a half copy, to the lead')
        else:
            partials.append(dict(file=name, half=str(target), sha256_half=half_sha, bytes_half=target.stat().st_size, source=str(cell_dir/name), sha256_source=source_sha))
    if problems or not remaining:
        raise hc.Refused('move_resume', f'{pair}: {problems or "nothing left on /home"}: nothing moved, to the lead')
    size = sum((cell_dir/n).stat().st_size for n in remaining)/2**30
    (disk or (lambda: hc.disk_guard(need_d_gib=size, need_home_gib=0.)))()
    hc.take_lock(resume.lock_path, hc.lock_line(arm='move', pair=pair, resume=dict(path=resume.path, sha256=resume.sha256), move_json_sha256=hb.sha(rec_dir/'move.json')))
    if partials:
        for row in partials:                            # HB diff review n2: the half copy kept whole under a new name, its sha256 before and after the rename
            kept = partial_name(row['half'])
            Path(row['half']).rename(kept)
            row.update(kept=str(kept), sha256_kept=hb.sha(kept), same=hb.sha(kept) == row['sha256_half'])
        hb.write_once(rec_dir/PARTIALS_RECORD, dict(schema=hb.SCHEMA.format('move_resume_partials'), pair=pair, move_json_sha256=hb.sha(rec_dir/'move.json'),
                                                    resume=dict(path=resume.path, sha256=resume.sha256), partials=partials, utc=hb.utc(),
                                                    rule='HB diff review n2: a half copy left at the destination by the first move (problem copying) renamed whole to <file>.partial_<k>, kept; '
                                                         'its sha256 before and after the rename'))
        bad = [row['file'] for row in partials if not row['same']]
        if bad:
            raise hc.Refused('move_resume', f'{pair}: the renamed half copies {bad} changed in the rename: nothing moved, to the lead')
    out = hc.move_files(cell_dir, dst, rec_dir/'move_resume.json', names=remaining)
    hb.remove_empty_tree(cell_dir)
    return out


def input_lock(out_rel, *, repo=None, pred_root=None):
    """Step 8: the rows of the lock (module docstring), written once to <code tree>/out_rel.  Returns the path."""
    repo = Path(hb.REPO if repo is None else repo)
    pred_root = Path(he.HOLDOUT_SOURCES['pred_root'] if pred_root is None else pred_root)
    if not str(out_rel).startswith(hc.LATER_RECORDS) or '..' in Path(out_rel).parts:
        raise hc.Refused('input_lock', f'{out_rel}: the lock is a record committed after the pinned commit, so it goes under {hc.LATER_RECORDS} (decision_log 2026-11-09 df8541d)')
    out = repo/out_rel
    if out.exists():
        raise hc.Refused('input_lock', f'{out} exists: the lock is written once')
    pairs = hb.rows()
    subjects = sorted({s for p in pairs for s in hc.split_pair(p)})
    if len(subjects) != 100:
        raise hc.Refused('input_lock', f'{len(subjects)} subjects in the pair list, 100 expected')
    hi, problems = hb.hi_union(subjects, pred_root=pred_root)        # A5b item 13: HI's first run and its resume in the one root, complete, each record against its files
    if problems:
        raise hc.Refused('input_lock', f"HI's hold-out inference is not complete over its runs or a record does not hold (A5a review I2; A5b item 13): {problems[:5]} ({len(problems)})")
    lines = []
    for s in subjects:
        for name in PRED_FILES:
            path = pred_root/s/name
            lines.append(dict(kind='prediction', subject=s, path=str(path), sha256=hi['records'][s]['sha256'][name]))
        row = hi['records'][s]
        lines.append(dict(kind='hi_record', subject=s, path=row['record'], sha256=row['record_sha256'], run=row['run']))
    final = finalized() or {}
    for p in pairs:
        cell_dir = where(p)
        path = cell_dir/'t88'/he.FIELD_FILE
        lines.append(dict(kind='method_field', pair=p, path=str(path), sha256=hb.sha(path) if path.is_file() else None, line_status=((final.get('pairs') or {}).get(p) or {}).get('line_status')))
    missing = [r for r in lines if r['sha256'] is None and r['kind'] == 'prediction']
    if missing:
        raise hc.Refused('input_lock', f'{len(missing)} prediction files are missing: {missing[:3]}')
    doc = dict(schema=LOCK_SCHEMA, finalized_outputs=None if not final else dict(path=str(hb.review_root()/'finalized_outputs.json'), sha256=hb.sha(hb.review_root()/'finalized_outputs.json')),
               rows=lines, hi=dict(lock=hi['lock'], root=hi['root'], runs=hi['runs'], rule=hi['rule']), utc=hb.utc(),
               rule='prereg §5.2 step 8: the 100 subjects\' probs36, structure_groups, provenance and the method arm\'s field_0600.npz by sha256 (the fields are kept: decision_log 2026-11-10 ruling (5)); '
                    'HI\'s record of each subject (subjects/<s>.json, kind hi_record) by sha256, its files those of the record (A5b item 13); nothing under attempt<k>/')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(doc, indent=1, ensure_ascii=False)+'\n')
    return out


# ---- LP-B: the LPBA40 modes' step 7 ---------------------------------------------------------------------------------------------------------------
LPBA_FIELD_DELETED = 'field_deleted.json'


def lpba_lock_rel(pair_set):
    """The default place of an LPBA mode's input lock: runs/<root name>/hb_input_lock.json (runs/lpba40_test_v1/ for the formal mode: C26.9-4's records)."""
    return f'runs/{hb.lpba().root(pair_set).name}/hb_input_lock.json'


def lpba_input_lock(pair_set, out_rel=None, *, repo=None):
    """LP-B, step 7's lock (module docstring): formal, written once to <code tree>/out_rel; feasibility, to <feasibility root>/hb_input_lock.json (never the code tree).  Returns the path;
    Refused (stage input_lock, or code: rule one) otherwise."""
    la = hb.lpba()
    repo = Path(hb.REPO if repo is None else repo)
    problem = la.formal_code_problem(pair_set, repo=repo)
    if problem:
        raise hc.Refused('code', problem)
    if pair_set == la.FEAS:
        if out_rel is not None:
            raise hc.Refused('input_lock', f'{out_rel}: the feasibility input lock is written to its root only, never into the code tree (review I2: one commit for the whole chain)')
        out = la.root(pair_set)/'hb_input_lock.json'
    else:
        out_rel = lpba_lock_rel(pair_set) if out_rel is None else out_rel
        prefix = f'runs/{la.root(pair_set).name}/'
        if not str(out_rel).startswith(prefix) or '..' in Path(out_rel).parts:
            raise hc.Refused('input_lock', f'{out_rel}: the {pair_set} input lock is a record under {prefix} (C26.9-4)')
        out = repo/out_rel
    if out.exists():
        raise hc.Refused('input_lock', f'{out} exists: the lock is written once')
    pairs = hb.rows_of(pair_set)
    subjects = sorted({s for p in pairs for s in la.split(p)})
    final_path = hb.review_root(pair_set)/'finalized_outputs.json'
    final = hb.read_json(final_path)
    if not isinstance(final, dict) or sorted((final.get('pairs') or {})) != sorted(pairs):
        raise hc.Refused('input_lock', f'{final_path}: step 5 has not finished for every pair of {pair_set} (C26.6-6: the method output is fixed first)')
    hi, problems = hb.lpba_hi(pair_set, subjects)
    if problems:
        raise hc.Refused('input_lock', f"HI's {pair_set} output is not complete or a record does not hold: {problems[:5]} ({len(problems)})")
    pred_root = Path(la.lc.roots(pair_set).pred_root)
    lines = []
    for s in subjects:
        for name in PRED_FILES:
            lines.append(dict(kind='prediction', subject=s, path=str(pred_root/s/name), sha256=hi['records'][s]['sha256'][name]))
        row = hi['records'][s]
        lines.append(dict(kind='hi_record', subject=s, path=row['record'], sha256=row['record_sha256']))
    for p in pairs:
        path = hb.cell(p, pair_set)/'t88'/he.FIELD_FILE
        lines.append(dict(kind='method_field', pair=p, path=str(path), sha256=hb.sha(path) if path.is_file() else None, bytes=path.stat().st_size if path.is_file() else None,
                          line_status=((final.get('pairs') or {}).get(p) or {}).get('line_status')))
    missing = [r for r in lines if r['sha256'] is None and r['kind'] == 'prediction']
    if missing:
        raise hc.Refused('input_lock', f'{len(missing)} prediction files are missing: {missing[:3]}')
    doc = dict(schema=LOCK_SCHEMA, pair_set=pair_set, finalized_outputs=dict(path=str(final_path), sha256=hb.sha(final_path)), rows=lines, hi=dict(lock=hi['lock'], root=hi['root'], rule=hi['rule']),
               utc=hb.utc(), rule='LPBA40 prereg §8.2 step 7: the subjects\' probs36, structure_groups, provenance and HI records and the method arm\'s field_0600.npz by sha256 (and size), then the '
                               'fields of the hx_certified pairs deleted (lpba-delete-fields; the fallback pairs keep every file)')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(doc, indent=1, ensure_ascii=False)+'\n')
    return out


def lpba_delete_fields(pair_set, lock_rel, *, repo=None):
    """LP-B, step 7 after the lock (module docstring): the lock committed and unchanged in the code tree; each row of kind method_field: hx_certified -> the file's sha256 and size the row's, its record
    written in the cell's records, then the file deleted; another line status -> kept (recorded).  Returns the rows; Refused (stage delete_fields) when the lock does not hold or a field is not the
    lock's (that field and every later one kept; to the lead)."""
    repo = Path(hb.REPO if repo is None else repo)
    la = hb.lpba()
    if pair_set != la.FORMAL:
        raise hc.Refused('delete_fields', f'{pair_set}: the feasibility mode deletes no field (review I2: its pairs keep every file; nothing of its chain is committed)')
    problem = la.formal_code_problem(pair_set, repo=repo)
    if problem:
        raise hc.Refused('code', problem)
    path = repo/lock_rel
    if not str(lock_rel).startswith(f'runs/{la.root(pair_set).name}/'):
        raise hc.Refused('delete_fields', f'{lock_rel}: not a record of {pair_set}')
    try:
        tracked = hc._hi().tracked_unchanged(path, repo, stage='delete_fields')
    except hc._hi().HIRefused as exc:
        raise hc.Refused('delete_fields', f'the input lock is not committed and unchanged (§8.2 step 7: committed first): {exc}') from None
    lock = hb.read_json(path)
    if not isinstance(lock, dict) or lock.get('schema') != LOCK_SCHEMA or lock.get('pair_set') != pair_set:
        raise hc.Refused('delete_fields', f'{lock_rel}: not the {pair_set} input lock')
    out = []
    for row in lock.get('rows') or []:
        if row.get('kind') != 'method_field':
            continue
        pair, field = row['pair'], Path(row['path'])
        if field != hb.cell(pair, pair_set)/'t88'/he.FIELD_FILE:
            raise hc.Refused('delete_fields', f'{pair}: the lock names {field}, not the cell\'s field')
        if row.get('line_status') != he.HX:
            out.append(dict(pair=pair, path=str(field), deleted=False, kept=f"{row.get('line_status')}: a fallback pair (or no output) keeps every file (§8.2 step 7)"))
            continue
        record = hb.record_dir(ARM, pair, pair_set)/LPBA_FIELD_DELETED
        if not field.is_file():
            done = hb.read_json(record)
            if isinstance(done, dict) and done.get('sha256') == row['sha256']:
                out.append(dict(pair=pair, path=str(field), deleted=True, already=True))
                continue
            raise hc.Refused('delete_fields', f'{field}: not there and no record of its deletion')
        size, digest = field.stat().st_size, hb.sha(field)
        if digest != row['sha256'] or size != row.get('bytes'):
            raise hc.Refused('delete_fields', f'{field}: sha256 {digest} / {size} bytes is not the lock\'s {row["sha256"]} / {row.get("bytes")}: kept, to the lead')
        hb.write_once(record, dict(schema=hb.SCHEMA.format('field_deleted'), pair=pair, pair_set=pair_set, path=str(field), sha256=digest, bytes=size, lock=dict(path=str(lock_rel), sha256=hb.sha(path), git=tracked),
                                   utc=hb.utc(), rule='LPBA40 prereg §8.2 step 7, C26.6-7: written before the field goes; its sha256 and size are the committed input lock\'s'))
        field.unlink()
        out.append(dict(pair=pair, path=str(field), deleted=True, sha256=digest, bytes=size, record=str(record)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    m = sub.add_parser('move')
    m.add_argument('--pairs', default=None)
    mr = sub.add_parser('move-resume')
    mr.add_argument('--pair', required=True)
    mr.add_argument('--authorization', required=True)
    lk = sub.add_parser('input-lock')
    lk.add_argument('--out', default=DEFAULT_LOCK_REL)
    ll = sub.add_parser('lpba-input-lock')
    ll.add_argument('--pair-set', required=True, choices=hb.hold.LPBA_PAIR_SETS)
    ll.add_argument('--out', default=None)
    ld = sub.add_parser('lpba-delete-fields')
    ld.add_argument('--pair-set', required=True, choices=('lpba40_test',))
    ld.add_argument('--lock', required=True)
    for p in (m, mr, lk, ll, ld):
        p.add_argument('--commit', default=None, help='only a check: must be P1 of runs/holdout_B_v1/pinned_code.json (rule one); an LPBA mode: P_L (formal) or HEAD (feasibility)')
    o = ap.parse_args(argv)
    try:
        import hb_method
        pair_set = getattr(o, 'pair_set', 'holdout')
        _, problems = hb_method.start_checks(o.commit) if not hb.is_lpba(pair_set) else hb_method.lpba_start_checks(pair_set, o.commit)      # the code (holdout_common.pinned_code_problems), the pins, the unseal gate; LP-B: an LPBA mode's
        if problems:
            raise hc.Refused('start', f'{problems}')
        if o.command == 'lpba-input-lock':
            print(lpba_input_lock(o.pair_set, o.out))
            return 0
        if o.command == 'lpba-delete-fields':
            print(json.dumps(lpba_delete_fields(o.pair_set, o.lock), indent=1))
            return 0
        if o.command == 'move':
            final = finalized()
            for pair in (o.pairs.split(',') if o.pairs else hb.rows()):
                done = hb.read_json(hb.record_dir(ARM, pair)/'move.json')
                if done is not None:
                    if done.get('passed') or (hb.read_json(hb.record_dir(ARM, pair)/'move_resume.json') or {}).get('passed'):
                        continue
                    raise hc.Refused('move', f'{pair}: half moved (move.json passed false): repair with move-resume and a committed resume record (HB review I3); the batch stops here')
                print(pair, move_pair(pair, final=final)['passed'])
            return 0
        if o.command == 'move-resume':
            print(o.pair, move_resume(o.pair, o.authorization)['passed'])
            return 0
        print(input_lock(o.out))                       # input-lock
        return 0
    except FileExistsError as exc:                      # HB review m11: a record written once that is there already
        print(f'refused: {exc}', file=sys.stderr)
        return hb.EXIT['refused']
    except (hc.Refused, hc.InfrastructureFailure) as exc:
        print(f'refused: {exc}', file=sys.stderr)
        return hb.EXIT['refused']


if __name__ == '__main__':
    sys.exit(main())
