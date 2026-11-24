"""The pre-repair displacement of the repair ablation (prereg revision 2 item 4; contract revision 9 draft C24.2-3; runs/20261110-a5/任务书.md, A5b item 3 (a)).

    python -B experiments/hb/hb_prerepair.py run [--pairs P,...] [--commit SHA]     after step 9 (every timed run of configuration C, the external arms, the descriptive arms and the 90 s tier
                                                                                    ended), before the step-10 manifest; CPU, in its own unit tag-hb-prerepair
    python -B experiments/hb/hb_prerepair.py dry-run [--commit SHA]                 the start checks only (rule one, the pins, the unseal gate, the frozen API); nothing of a pair is read
    python -B experiments/hb/hb_prerepair.py launch-string [--pairs P,...] [--commit SHA]   the lead's start string of the unit tag-hb-prerepair (PYTHONPATH <frozen tree>:<code>)

What it writes, per pair of the pair list, once (O_EXCL), never into the finalized and moved method cell:
  /home/rmcfz/seq_vp_tag_results/holdout_B_v1/prerepair/<pair>/prerepair.json     the record (schema bayesreg.hb.prerepair.v1; the evaluation entry reads it, bayesreg.eval.holdout.check_ablation)
  /mnt/d/seq_vp_tag_holdout_B_v1/prerepair/<pair>/prerepair_disp.npz               d = he_core.u_to_d(boundary_taper(u)) (key d, (3, nz, ny, nx) voxels, as output/disp.npz), only when
                                                                                    the record passes
The pre-repair field is the repair's input (C22.2-6, C22.3-1): HE's repair child takes t88/field_0600.npz's u and calls fz.boundary_taper(raw, pin.grid, settings.t88.taper)
(experiments/he/run_hx.py source_checked); this tool makes the same call with the same function (mc0_core.frozen_api, the frozen tree's preopt.py at its pinned sha256), the frozen settings'
taper (he_core.frozen_settings; settings_sha256 = hb_method.SETTINGS_SHA256, checked at the start) and the same grid (bayesreg.io.pair._read_manifest of the pair's stage manifest, as
load_pair builds it; pair.npz is never opened).  The conversion is he_core.u_to_d, the one he_core.write_output uses for output/disp.npz (write_output itself needs a passing certificate,
which a pre-repair field has not).

The inputs, from the finalized record and the move records (the interface alignment revisions 1, 2; never a path template): review/finalized_outputs.json's row of the pair (its cell and
part.t88_repair), then the move.json rows (or move.json's ok rows with a passed move_resume.json) of t88/field_0600.npz, t88/run.json, repair/repair.json: each file read at the row's dst on
D:, its sha256 = the row's sha256_after = the finalized record's; field_0600.npz's also = the step-8 input lock's row (runs/holdout_B_v1/hb_input_lock.json, committed).  t88/run.json says done
with this field (field.sha256) and the settings_sha256 of the method; its data.manifest_sha256 is the stage manifest's sha256.
The three checks (C24.2-3), all must hold before the displacement is written:
  field_sha256       the field file's sha256 = repair/repair.json's source.field_sha256;
  u_sha256           the C-order sha256 of its u (he_core.array_sha256) = source.u_sha256;
  tapered_u_sha256   the sha256 of boundary_taper(u) = source.tapered_u_sha256.
Any other check that does not hold (no finalized row, no repair input -- T88 not done or repair.json without source, a move or sha256 that differs, the admission) is recorded the same way:
passed false with the problems; no displacement; the pair is left out of the ablation (the evaluation entry lists why), F1-F4 untouched.  A pair whose record exists is skipped; a
displacement without a record (a killed run between the two writes) refuses that pair (to the lead; a rerun only by holdout_common's resume, at most once -- not written here).
A5b review minor 5 (infrastructure, not the pair's): an input file a move row names on D: that is not there (D: not mounted, a file gone) writes no record; the run stops (exit 20, to the lead)
and the pair can be prepared later.  A5b review minor 3: any other exception of the run (a record that is not JSON, np.load, the frozen API) is recorded in the run record with its traceback
and exit 23, never exit 0; the frozen API (mc0_core.frozen_api: jd_curl from the frozen tree, PYTHONPATH <frozen>:<code>) is checked at the start, before any pair.  A5b review minor 4:
--pairs must be rows of the pair list (checked after the unseal gate, before any pair).
The admission (C23.4-2) per pair in this process: holdout_common.admit_holdout_pair (load_unseal, bind_unseal, check_subject, the pair list in its direction), then the stage manifest's subjects.
The start: hb_method.start_checks (rule one: P1 from runs/holdout_B_v1/pinned_code.json; the frozen settings' sha256; the preregistration pin; the unseal gate), the machine gate (no other
tag-* / seq-vp-* unit: not beside the step-9 timed units), and the spy (C23.6-5: the guard's cache of the sealed lists empty when it is installed; every open under the hold-out roots
recorded; a label file, the subjects' data root, a pair.npz, the evaluation's directories and anything under the hold-out roots but the method's records and cells, the review records, the
stage manifests and this tool's own directories refused at the open).  Nothing of a truth label, an evaluation record or another arm is read.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import hb_core as hb  # noqa: E402

hc, he = hb.hc, hb.he
ARM = hb.ARM
SCHEMA = hb.SCHEMA.format('prerepair')
UNIT = 'tag-hb-prerepair'
RECORD, DISP = 'prerepair.json', 'prerepair_disp.npz'       # bayesreg.eval.holdout.PREREPAIR_RECORD, PREREPAIR_DISP (a test compares)
CHECKS = ('field_sha256', 'u_sha256', 'tapered_u_sha256')     # bayesreg.eval.holdout.PREREPAIR_CHECKS
INPUTS = ('t88/field_0600.npz', 't88/run.json', 'repair/repair.json')
LOCK_REL = 'runs/holdout_B_v1/hb_input_lock.json'            # hb_store.DEFAULT_LOCK_REL (a test compares)
MOVE_NAME, MOVE_RESUME_NAME = 'move.json', 'move_resume.json'
DATA_ROOT = Path('/mnt/d/data')                              # the subjects' files and labels (hb_review.forbidden_roots)
EXIT = dict(done=0, infra=20, pair_refused=21, refused=22, error=23)


class InputMissing(Exception):
    """A5b review minor 5: an input file on D: is not there -- infrastructure, not the pair's: no record is written, the run stops (exit 20)."""


def record_root():
    return Path(hc.guard_root(Path(hc.HOLDOUT_ROOT)/'prerepair'))


def big_root():
    return Path(hc.guard_root(Path(hc.BIG_ROOT)/'prerepair'))


def paths(pair):
    hc.safe_name(pair, 'pair')
    return record_root()/pair/RECORD, big_root()/pair/DISP


# ---- the spy (C23.6-5) ----------------------------------------------------------------------------------------------------------------------------
class Breach(PermissionError):
    """An open the spy does not allow: raised at the open, so nothing is read."""


class Spy:
    """Python's audit event "open": every open under the hold-out roots (/home's holdout_B_v1 and D:'s) and the data root recorded; refused at the open: the data root, a label file name
    (aligned_seg*, a .nii with seg), any pair.npz, the evaluation's directories (holdout_B_v1/eval, <results>/ev), and under the hold-out roots anything but method/ (the records),
    work/method/, review/, stage/<pair>/manifest.json, prerepair/ (and on D: method/, prerepair/).
    The code tree (XF, 2026-11-15; runs/20261115-d1/实现说明_XF.md): this tool runs from the detached checkout <results>/holdout_B_v1/code (C22.H0-4; launch_string's working directory), which
    lies under the hold-out root /home's holdout_B_v1.  A READ whose two forms (as written and resolved) lie in the running code tree (`code`, default hb_core.REPO: the tree this module is in)
    is allowed after the name rule and the evaluation's and data roots, before the hold-out roots' rule -- the start's lazy imports (hb_method, bayesreg.io.code_identity, he_core's modules),
    the preregistration, the pair list, the unseal record, the allowlist, the sealed lists and the input lock are files of that tree.  Before XF every one of them was a breach there ('code' is
    not one of the hold-out root's allowed entries).  A write into the code tree is still refused.  The code tree's own data/ is not code (XF revision 2, review I-1): a path whose form as
    written or resolved lies under <code>/data (in either form of the code root) is judged as before -- under the hold-out root ('code' not allowed) -- whether data/ is a link (seq_vp_3d/data in
    the main checkout's layout) or a real directory; hb_review's hold-out deny holds hb.REPO/data the same way."""

    def __init__(self, code=None):
        self.events, self.breaches, self.active = [], [], False
        self.home, self.big, self.results = Path(hc.HOLDOUT_ROOT).resolve(), Path(hc.BIG_ROOT).resolve(), Path(hc.RESULTS).resolve()
        code = Path(hb.REPO if code is None else code)
        self.code, self.code_forms = code.resolve(), (Path(os.path.abspath(code)), code.resolve())
        self.data_forms = tuple(dict.fromkeys(c/'data' for c in self.code_forms))           # XF revision 2: <code>/data in both forms of the code root

    def in_code_tree(self, path):
        """XF: the path as written (absolute) and resolved both lie in the running code tree, and neither form lies under the code tree's data/ (revision 2)."""
        p = Path(os.path.abspath(path))
        real = Path(os.path.realpath(p))
        if any(form == d or d in form.parents for form in (p, real) for d in self.data_forms):
            return False
        return any(p == c or c in p.parents for c in self.code_forms) and (real == self.code or self.code in real.parents)

    def why(self, path, write=False):
        p = Path(os.path.abspath(path))
        name = p.name
        if name.startswith('aligned_seg') or ('seg' in name and ('.nii' in name)) or name == 'pair.npz':
            return 'a label file or a pair.npz'
        for root in (DATA_ROOT, self.results/'ev', self.home/'eval'):
            if p == root or root in p.parents:
                return f'under {root}'
        if not write and self.in_code_tree(p):
            return None
        for root, allowed in ((self.home, ('method', 'review', 'stage', 'prerepair', 'work', 'locks')), (self.big, ('method', 'prerepair'))):
            if root in p.parents:
                rel = p.relative_to(root).parts
                if rel[0] not in allowed or (rel[0] == 'work' and (len(rel) < 2 or rel[1] != ARM)) or (rel[0] == 'stage' and rel[-1] != 'manifest.json'):
                    return f'under {root} but not a file this tool reads'
        return None

    def watched(self, path):
        p = Path(os.path.abspath(path))
        return any(p == r or r in p.parents for r in (self.home, self.big, DATA_ROOT))

    def saw(self, path, mode, flags=0):
        if not self.active or path is None or isinstance(path, int):
            return
        raw = os.fsdecode(path)
        why = self.why(raw, write=is_write(mode, flags))
        if why or self.watched(raw):
            self.events.append(dict(path=os.path.abspath(raw), mode=None if mode is None else str(mode), refused=why))
        if why:
            self.breaches.append(dict(path=raw, why=why))
            raise Breach(f'hb_prerepair may not open {raw}: {why}')

    def record(self):
        return dict(events=self.events, breaches=self.breaches, passed=not self.breaches, limits='os.open and mmap are not seen; the audit event "open" only (C23.6-5)')


_SPY = None
_HOOKED = False
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND


def is_write(mode, flags):
    """XF: the audit event 'open' gives (path, mode, flags); a write by its mode (w, a, x, +) or, for os.open (mode None), by its flags (as experiments/xb/spy/xb_policy.is_write)."""
    if isinstance(mode, str):
        return any(c in mode for c in 'wax+')
    return bool((flags or 0) & WRITE_FLAGS)


def _audit(event, args):
    spy = _SPY
    if spy is not None and event == 'open' and args:
        spy.saw(args[0], args[1] if len(args) > 1 else None, args[2] if len(args) > 2 else 0)


def install_spy(code=None):
    """C23.6-5: the guard's copy of the sealed lists must be empty (its first read happens under the spy); the hook stays for the process, inactive while no spy is set.
    code: the running code tree whose reads are allowed (Spy; default hb_core.REPO -- the tests give a code tree of their own)."""
    global _SPY, _HOOKED
    from bayesreg.io.sealed import sealed_cache_is_empty
    if not sealed_cache_is_empty():
        raise hc.Refused('spy', 'the sealed lists are already cached in this process: start a new process (C23.6-5)')
    _SPY = Spy(code)
    _SPY.active = True
    if not _HOOKED:
        sys.addaudithook(_audit)
        _HOOKED = True
    return _SPY


def remove_spy():
    global _SPY
    spy, _SPY = _SPY, None
    if spy is not None:
        spy.active = False
    return spy


# ---- the inputs ------------------------------------------------------------------------------------------------------------------------------------
def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def finalized():
    path = hb.review_root()/'finalized_outputs.json'
    if not path.is_file():
        raise hc.Refused('finalized', f'{path} is not there: step 5 has not finalized the method arm')
    doc = read_json(path)
    if not isinstance(doc, dict) or doc.get('schema') != hb.SCHEMA.format('finalized_outputs') or not isinstance(doc.get('pairs'), dict):
        raise hc.Refused('finalized', f'{path}: not a finalized record of HB')
    return path, doc


def input_lock(repo):
    """The step-8 input lock (hb_store.input_lock's format), committed and unchanged in the code tree `repo`: {pair: (path, sha256)} of its method_field rows, and the record."""
    path = Path(repo)/LOCK_REL
    try:
        git = hc._hi().tracked_unchanged(path, Path(repo), stage='input_lock')
    except hc._hi().HIRefused as exc:
        raise hc.Refused('input_lock', f'{LOCK_REL}: not committed and unchanged ({exc})') from None
    doc = read_json(path)
    if not isinstance(doc, dict) or doc.get('schema') != hb.SCHEMA.format('input_lock') or not isinstance(doc.get('rows'), list):
        raise hc.Refused('input_lock', f'{LOCK_REL}: not HB\'s input lock')
    rows = {r.get('pair'): r.get('sha256') for r in doc['rows'] if isinstance(r, dict) and r.get('kind') == 'method_field'}
    return rows, dict(path=str(path), sha256=hb.sha(path), last_commit=git.get('last_commit'))


def moved(pair, frow):
    """{file: row} of the move records (the interface alignment revision 2): move.json passed -> its rows; passed false -> its ok rows and move_resume.json's (which must have passed); every
    file in one ok row; src under the finalized cell, dst under holdout_common.big_dir.  (rows, records, problem)."""
    rec_dir = hb.record_dir(ARM, pair)
    dst_dir, cell_dir = Path(hc.big_dir(ARM, pair)), Path(frow['cell'])
    first_path = rec_dir/MOVE_NAME
    if not first_path.is_file():
        return None, {}, f'{first_path}: no move record (the cell was not moved to D:)'
    records = {str(first_path): hb.sha(first_path)}
    first = read_json(first_path)
    docs = [first]
    if first.get('passed') is not True:
        second_path = rec_dir/MOVE_RESUME_NAME
        if not second_path.is_file():
            return None, records, f'{first_path}: passed {first.get("passed")!r} and no {MOVE_RESUME_NAME}'
        second = read_json(second_path)
        records[str(second_path)] = hb.sha(second_path)
        if second.get('passed') is not True:
            return None, records, f'{second_path}: passed {second.get("passed")!r}'
        docs.append(second)
    rows, problems = {}, []
    for doc in docs:
        if os.path.realpath(str(doc.get('dst_dir'))) != os.path.realpath(str(dst_dir)) or os.path.realpath(str(doc.get('src_dir'))) != os.path.realpath(str(cell_dir)):
            problems.append(f'a move record from {doc.get("src_dir")} to {doc.get("dst_dir")}, not {cell_dir} to {dst_dir}')
        for row in doc.get('files') or []:
            if isinstance(row, dict) and row.get('ok') is True:
                if row.get('file') in rows:
                    problems.append(f"{row.get('file')}: in two ok rows")
                rows[row.get('file')] = row
    return (None if problems else rows), records, '; '.join(problems) or None


# ---- one pair ----------------------------------------------------------------------------------------------------------------------------------------
def prepare(pair, *, final, final_record, lock_rows, lock_record, code, taper):
    """The record of one pair (module docstring) and, when it passes, its prerepair_disp.npz.  Returns the record (written)."""
    record_path, disp_path = paths(pair)
    record = dict(schema=SCHEMA, pair=pair, passed=False, checks={k: None for k in CHECKS}, problems=[], inputs={}, output=None, finalized=final_record, input_lock=lock_record,
                  move_records={}, code=code, taper=taper, settings_sha256=None, line_status=None, utc_start=hb.utc(),
                  rule='prereg revision 2 item 4, C24.2-3: u_to_d(boundary_taper(t88/field_0600.npz u)) with the three checks against repair/repair.json source; described only, not certified')

    def done(problem=None):
        if problem:
            record['problems'].append(problem)
        record['passed'] = not record['problems'] and all(record['checks'][k] is True for k in CHECKS)
        record['utc'] = hb.utc()
        hc.write_json_once(record_path, record)
        return record
    frow = (final.get('pairs') or {}).get(pair)
    if not isinstance(frow, dict) or not isinstance(frow.get('cell'), str):
        return done(f'the finalized record has no row with a cell for {pair}')
    record['line_status'] = frow.get('line_status')
    fixed, moving = hc.split_pair(pair)
    try:
        admission, _ = hc.admit_holdout_pair(fixed, moving)       # C23.4-2 in this process (ids only), before any file of the pair is read
    except hc.Refused as exc:
        return done(f'admission: {exc}')
    record['admission'] = admission
    pinned =((frow.get('part') or {}).get('t88_repair') or {}) if isinstance(frow.get('part'), dict) else {}
    missing = [n for n in INPUTS if not pinned.get(n)]
    if missing:
        return done(f'no repair input: the finalized record names no {missing} (T88 not done, or no step-5 review of an output)')
    rows, move_records, problem = moved(pair, frow)
    record['move_records'] = move_records
    if problem:
        return done(problem)
    files = {}
    for name in INPUTS:
        row = rows.get(name)
        if row is None or row.get('sha256_after') != pinned[name]:
            return done(f'{name}: no ok move row at the finalized sha256 {pinned[name]}')
        if not isinstance(row.get('dst'), str):
            return done(f'{name}: the move row has no dst (the move record\'s format)')
        path = Path(row['dst'])
        if not path.is_file():
            raise InputMissing(f'{pair} {name}: {path} (the move row\'s dst) is not there: no record written; to the lead (A5b review minor 5)')
        digest = hb.sha(path)
        record['inputs'][name] = dict(path=str(path), sha256=digest)
        if digest != pinned[name]:
            return done(f'{name}: {path} has sha256 {digest}, the finalized record {pinned[name]}')
        files[name] = path
    if lock_rows.get(pair) != record['inputs']['t88/field_0600.npz']['sha256']:
        return done(f"t88/field_0600.npz: the input lock's row {lock_rows.get(pair)!r} is not its sha256")
    run = read_json(files['t88/run.json'])
    record['settings_sha256'] = run.get('settings_sha256')
    field = run.get('field') if isinstance(run.get('field'), dict) else {}
    if run.get('status') != he.T88_DONE or field.get('sha256') != record['inputs']['t88/field_0600.npz']['sha256']:
        return done(f"t88/run.json: status {run.get('status')!r} with field sha256 {field.get('sha256')!r}: not done with this field")
    import hb_method
    if run.get('settings_sha256') != hb_method.SETTINGS_SHA256:
        return done(f"t88/run.json: settings_sha256 {run.get('settings_sha256')!r} is not the method's {hb_method.SETTINGS_SHA256}")
    source = read_json(files['repair/repair.json']).get('source')
    if not isinstance(source, dict) or not all(isinstance(source.get(k), str) for k in CHECKS):
        return done('repair/repair.json has no source with field_sha256, u_sha256, tapered_u_sha256 (the repair child did not get past source_checked)')
    from bayesreg.io.pair import _read_manifest
    manifest = Path(hc.STAGE_ROOT)/pair/'manifest.json'
    manifest_sha = hb.sha(manifest) if manifest.is_file() else None
    record['inputs']['stage/manifest.json'] = dict(path=str(manifest), sha256=manifest_sha)
    if manifest_sha is None or manifest_sha != (run.get('data') or {}).get('manifest_sha256'):
        return done(f"{manifest}: sha256 {manifest_sha} is not t88/run.json's data.manifest_sha256 {(run.get('data') or {}).get('manifest_sha256')!r}")
    grid, metadata, selected = _read_manifest(manifest)
    if (selected['fixed_subject'], selected['moving_subject']) != (fixed, moving):
        return done(f"{manifest}: the subjects {selected['fixed_subject']} <- {selected['moving_subject']} are not the pair's")
    record['grid'] = metadata
    record['checks']['field_sha256'] = record['inputs']['t88/field_0600.npz']['sha256'] == source['field_sha256']
    with np.load(files['t88/field_0600.npz'], allow_pickle=False) as z:
        raw = np.array(z['u'])
    record['checks']['u_sha256'] = he.array_sha256(raw) == source['u_sha256']
    fz = he.core.frozen_api()
    u, _ = fz.boundary_taper(raw, grid, taper)
    u = np.ascontiguousarray(u)
    record['checks']['tapered_u_sha256'] = he.array_sha256(u) == source['tapered_u_sha256']
    failed = [k for k in CHECKS if record['checks'][k] is not True]
    if failed:
        return done(f'the checks {failed} against repair/repair.json source do not hold')
    d = he.u_to_d(u, grid)
    disp_path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(convention='pull-back: warped(p) = moving(p + d(p)), p = (z, y, x) voxel index (he_core.write_output\'s)', units='voxels', layout='(3, nz, ny, nx)', grid=he.grid_record(grid),
                field='boundary_taper(t88/field_0600.npz u): the repair\'s input, not certified', tapered_u_sha256=he.array_sha256(u), d_sha256=he.array_sha256(d))
    with open(hc.guard_root(disp_path), 'xb') as stream:          # O_EXCL: written once
        np.savez(stream, d=d, meta=np.array(json.dumps(meta)))
    record['output'] = dict(path=str(disp_path), sha256=hb.sha(disp_path), d_sha256=meta['d_sha256'], tapered_u_sha256=meta['tapered_u_sha256'])
    return done()


# ---- the run ---------------------------------------------------------------------------------------------------------------------------------------------
def start(commit=None, *, repo=None):
    """hb_method.start_checks (rule one, the frozen settings, the preregistration, the unseal gate), the frozen API (mc0_core.frozen_api: it needs jd_curl from the frozen tree; A5b review
    minor 3, checked here before any pair) and the taper of the frozen settings.  (record, problems)."""
    import hb_method
    record, problems = hb_method.start_checks(commit)
    try:
        fz = he.core.frozen_api()
        record['frozen_api'] = dict(boundary_taper=callable(getattr(fz, 'boundary_taper', None)), frozen_tree=str(he.core.FROZEN))
        if not record['frozen_api']['boundary_taper']:
            problems.append('the frozen API has no boundary_taper')
    except (Exception, SystemExit) as exc:                        # noqa: BLE001 - mc0_core.frozen_api exits when jd_curl is not the frozen tree's
        record['frozen_api'] = f'{type(exc).__name__}: {exc}'
        problems.append(f'the frozen API (mc0_core.frozen_api; PYTHONPATH <frozen tree>:<code>): {type(exc).__name__}: {exc}')
    record['taper'] = he.frozen_settings()['t88']['taper']
    record['repo'] = str(hb.REPO if repo is None else repo)
    return record, problems


def launch_string(args, *, repo=None):
    """A5b review minor 3: the lead's start string of this tool's unit (from the detached checkout of P1, as hb_core.queue_launch): CPU only, MemoryMax 8G, MemorySwapMax 2G 〔估算: one pair's
    field, tapered field and displacement at a time, about 165 MB each〕, nice 5, PYTHONPATH <frozen tree>:<code> (mc0_core.frozen_api).  The lead starts it after step 9, before step 10."""
    repo = Path(hb.REPO if repo is None else repo)
    argv = ['systemd-run', '--user', f'--unit={UNIT}', '--collect', f'--working-directory={repo}', '--property=MemoryMax=8G', '--property=MemorySwapMax=2G', '--nice=5',
            '--setenv=PYTHONDONTWRITEBYTECODE=1', '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1', '--setenv=CUDA_VISIBLE_DEVICES=',
            f'--setenv=PYTHONPATH={he.core.FROZEN}:{repo}', str(hb.PY), '-B', 'experiments/hb/hb_prerepair.py', 'run', *args]
    if 'pytest' in ' '.join(argv):
        raise ValueError(f'start string with a forbidden word: {argv}')
    return argv


def run(*, pairs=None, commit=None, repo=None, gate=None, echo=print):
    """Every pair (or `pairs`): the start, the machine gate, the spy, the finalized record and the input lock, then prepare() per pair not done.  Writes record_root()/runs/<stamp>.json.  Exit."""
    repo = Path(hb.REPO if repo is None else repo)
    out = dict(schema=hb.SCHEMA.format('prerepair_run'), utc_start=hb.utc(), pairs={}, pid=os.getpid())
    spy = None
    code = EXIT['done']
    try:
        spy = install_spy()                             # first: the start's unseal gate is the guard's first read of the sealed lists (C23.6-5)
        record, problems = start(commit, repo=repo)
        out['start'] = record
        if problems:
            raise hc.Refused('start', f'{problems}')
        out['machine'] = (gate or (lambda: hc.machine_gate(own_units=(UNIT,))))()
        final_path, final = finalized()
        final_record = dict(path=str(final_path), sha256=hb.sha(final_path))
        lock_rows, lock_record = input_lock(repo)
        code_record = dict(commit=record['code']['commit'], head=record['code'].get('head'), pinned_code=record['code'].get('pinned_code'))
        listed = hb.rows()
        unknown = [p for p in (pairs or []) if p not in listed]
        if unknown:
            raise hc.Refused('pairs', f'not rows of the pair list: {unknown} (A5b review minor 4: nothing is written for them)')
        for pair in (pairs or listed):
            record_path, disp_path = paths(pair)
            if record_path.exists():
                out['pairs'][pair] = dict(state='skipped', reason=f'{record_path} exists (written once)')
                continue
            if disp_path.exists():
                out['pairs'][pair] = dict(state='refused', reason=f'{disp_path} exists without its record: to the lead (a rerun only by a committed resume)')
                code = EXIT['pair_refused']
                continue
            doc = prepare(pair, final=final, final_record=final_record, lock_rows=lock_rows, lock_record=lock_record, code=code_record, taper=record['taper'])
            out['pairs'][pair] = dict(state='written', passed=doc['passed'], problems=doc['problems'], record=str(record_path), sha256=hb.sha(record_path))
            echo(f"{pair}: {'passed' if doc['passed'] else 'not passed'} {doc['problems'][:1]}")
    except (hc.Refused, hb.Stop) as exc:
        out['refused'] = str(exc)
        code = EXIT['refused']
        echo(f'refused: {exc}')
    except Breach as exc:
        out['refused'] = f'spy: {exc}'
        code = EXIT['refused']
        echo(f'refused by the spy: {exc}')
    except InputMissing as exc:
        out['infrastructure'] = str(exc)
        code = EXIT['infra']
        echo(f'infrastructure: {exc}')
    except (Exception, SystemExit) as exc:                        # noqa: BLE001 - A5b review minor 3: recorded, never exit 0
        import traceback
        out['error'] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()[-4000:])
        code = EXIT['error']
        echo(f'error: {type(exc).__name__}: {exc}')
    finally:
        spy = remove_spy() if spy is not None else None
        out.update(spy=None if spy is None else spy.record(), utc=hb.utc(), exit=code)
        if 'start' in out:
            hc.write_json_once(hb.numbered(record_root()/'runs', f'run_{hb.stamp()}_{os.getpid()}'), out)
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--pairs', default=None)
    d = sub.add_parser('dry-run')
    ls = sub.add_parser('launch-string')
    ls.add_argument('--pairs', default=None)
    for p in (r, d, ls):
        p.add_argument('--commit', default=None, help='only a check: must be P1 of runs/holdout_B_v1/pinned_code.json (rule one)')
    o = ap.parse_args(argv)
    if o.command == 'launch-string':
        print(' '.join(launch_string([*(['--pairs', o.pairs] if o.pairs else []), *(['--commit', o.commit] if o.commit else [])])))
        return 0
    if o.command == 'dry-run':
        record, problems = start(o.commit)
        print(json.dumps(dict(record, problems=problems), indent=1, default=str))
        return EXIT['refused'] if problems else 0
    return run(pairs=o.pairs.split(',') if o.pairs else None, commit=o.commit)


__all__ = ['Breach', 'CHECKS', 'DISP', 'EXIT', 'INPUTS', 'InputMissing', 'LOCK_REL', 'RECORD', 'SCHEMA', 'Spy', 'UNIT', 'big_root', 'finalized', 'input_lock', 'install_spy',
           'launch_string', 'moved', 'paths', 'prepare', 'record_root', 'remove_spy', 'run', 'start']


if __name__ == '__main__':
    sys.exit(main())
