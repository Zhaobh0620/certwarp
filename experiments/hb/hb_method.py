"""HB step 4 (preregistration §5.2 step 4; runs/20261109-hb/任务书.md item 3): the method arm Hx24-RF05-F1200-v4 on the 50 hold-out pairs, one HE supervisor unit at a time, in the pair list's order.

    python -B experiments/hb/hb_method.py dry-run                                   the queue's start checks (the code, the pins, the unseal gate); nothing written, no list of the hold-out read
    python -B experiments/hb/hb_method.py queue [--commit SHA] [--pairs P,...] [--resume PAIR=REL ...]
    python -B experiments/hb/hb_method.py launch-string [--commit SHA]               the lead's start string of the queue's unit (tag-hb-he-queue)
    python -B experiments/hb/hb_method.py no-resume --pair P --authorization REL [--commit SHA]     the terminal record of an infrastructure failure BZ does not let rerun (a committed 未批补跑 record)

The queue, in this order: the code (holdout_common.pinned_code_problems: P1 from the committed runs/holdout_B_v1/pinned_code.json, never HEAD, --commit only a check, every commit of P1..HEAD
checked; the locks' commits P1; C22.H0-4; the pinned commit is handed to HE as --pinned), HE's frozen settings the pinned ones (SETTINGS_SHA256), the preregistration pinned (PREREG_SHA256; None refuses a batch),
the unseal gate (hb_core.unseal_gate: load_unseal before any list is read), then the pair list (holdout_common.holdout_pairs, its sha256); when a pair is to run, HI's hold-out predictions
(hb_core.hi_union, A5b item 13: the first run and its resume in the one root complete over the list's 100 subjects, consistent, no residue in pred/, and each record of the subjects of the pairs to
run against its files' sha256; otherwise refused, 22, before any unit: HE's units read pred/<s>/ by id); the batch's disk guard; per pair:
  its state     terminal.json in the cell -> final (skipped), except hx_provenance_rejected (P20: not written; a rerun only by a committed resume record) and hx_implementation_error until BZ decides
                (C22.5-6: a committed no_resume record of the cell, BZ's (甲), makes it final and the others go on; a resume record, (乙), reruns it); step4.json of a pair that went on -> done
                (skipped); a lock without either -> STOP unless a committed resume record of this pair is given
                (prereg §5.4: only a cell without a complete record; at most once; the first attempt's cell directory renamed whole to <pair>.attempt<k>, every file's sha256 recorded, so that HE writes
                the resumed attempt where the first would have);
  the hold file holdout_B_v1/hold waits; the gate (hb_core.gate_once: no other tag-* or seq-vp-* unit, the GPU idle, MemAvailable >= 10 GiB, load; up to 15 minutes, else STOP); the disks (both
                sides); the cell's lock (O_EXCL, after every check; locks/hb__method__<pair>.lock);
  the unit      tag-hb-he-<ffff>-<mmmm>: he_core.SUPERVISOR_UNIT's values (MemoryMax 12G, MemorySwapMax 4G, OOMPolicy continue, CPUQuota 1200 %, RuntimeMaxSec 4800, NUMBA 8, nice 5), PYTHONPATH
                <frozen>:<code>, the GPU; its main process experiments/xb/xb_unit.py around `run_hx.py holdout-pair --pair P` (every process of the method admits the pair itself, he_holdout);
  afterwards    (prereg revision 1c item 1) HE's own line.json decides: output/line.json (an output: hx_certified, hx_fallback_configC, hx_fallback_identity) or the cell's line.json (no output:
                hx_provenance_rejected -> STOP, P20; hx_implementation_error -> STOP, C22.5-6; both get the cell's terminal.json).  HE left no line.json -> **infra_failure** always (revision
                1c item 1; runs/20261109-align item 4): STOP, one resume by a committed resume record (prereg §5.4); holdout_common.classify_unit_end's reading of the unit is recorded as a
                description only.  A resumed attempt without a line.json again: the cell's terminal.json infra_failure with the resume record (no output, no third run).
  certified.npz of an hx_certified pair: its u and output/final.npz's u bit for bit (dtype, shape, bytes and the state hash) -> its sha256 recorded, the file deleted; not the same -> both kept, STOP.
                A fallback pair keeps every file (prereg §5.3).  Nothing of a cell moves to D: here (C22.H1-1: after step 5).
  step4.json    the pair's record in holdout_B_v1/method/<pair>/ (once per attempt): the unit, its reading, HE's exit and line.json, the certified check, whether the batch went on.
Exit: 0 every pair done or skipped; 20 infrastructure (the unit's evidence, a precheck that did not pass in 15 minutes, the disks); 21 a STOP of the rules above (a certified mismatch included);
22 refused at the start; 24 a pair needs a committed resume record; 25 the lead's rejudgment; 26 hx_implementation_error or hx_provenance_rejected.

LP-B (contract C26.6, C26.8; LPBA40 preregistration 06121278... §8.2 step 4, §8.3; brief runs/20261119-lpba/任务书_B_方法配置C与外部臂.md item 1): --pair-set lpba40_test | lpba40_feas runs the
same queue on an LPBA40 mode -- the mode's table (experiments/lpba/lpba_arms.rows), its places (hb_core's pair-set places: <root>/work/method/<pair>, <root>/method/<pair>, <root>/locks, <root>/batches),
the units tag-lp-he-<Sff>-<Smm> (tag-lp-feas-he-...) around `run_hx.py lpba-pair --pair-set M --pair P` (every process admits the pair itself through the LPBA branch, he_holdout.lpba_admitted_images),
the hold file lpba40_test_v1/hold.  The start checks: the code (lpba_arms.code_record: formal, rule one -- LP-C2's, C26.9-4, fail closed; feasibility, a clean checkout), HE's frozen settings the pinned
ones (414eccd3..., C26.6-1), the pins (lpba_common.check_pins: the preregistration, the lists, the frozen files, the c record), the pinned LPBA allowlist; no unseal gate (LPBA40 is not in SG).  HI's
LPBA40 output (hb_core.lpba_hi) complete over the mode's subjects before a unit starts.  Per pair as above: no line.json is infra_failure, the batch stops (C26.6-8); certified.npz deleted after the bitwise
check (C26.6-7); hx_provenance_rejected and hx_implementation_error stop the batch (C26.8-2).  A resume or a no_resume record of an LPBA mode is read only when holdout_common knows its authorizations
(C26.8-3, D148: LP-C2's for the formal mode; the feasibility mode has none, C26.6-2): otherwise a stopped cell stays stopped.  Review I3 (C26.8-5): every process of an LPBA pair runs under
the allow-list spy (run_hx.lpba_spied); a spy_breach_<k>.json anywhere in the cell stops the batch, whatever line.json says.  Review I1: batch.json has the key pair_set only for an LPBA batch.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import hb_core as hb  # noqa: E402

import numpy as np  # noqa: E402

hc, he = hb.hc, hb.he
ARM = hb.ARM
# The pins (prereg §5.1 items 2, 11; revision 1 item 4 item 17): HE's frozen settings after HB (tissue_prior off, the names, review_certificate_in) and the BZ-adopted preregistration after its revision 2.
# SETTINGS_SHA256 is written here by HB (runs/20261109-hb/实现说明.md); PREREG_SHA256 at P0 (11-14), after BZ adopted revision 2 on 11-10 (None or another file: a hold-out batch refuses).
SETTINGS_SHA256 = '414eccd37fe264d081e87bb63235790dc3a1ec40655c57ac64337bf84992f35d'
PREREG_REL = 'docs/bayesreg/notes/留出集预登记_20261107.md'
PREREG_SHA256 = 'df7b497887dc7b8290493c040635edf7334e9ae3be584e31a89177080836f493'
STEP4 = 'step4.json'


def settings_problem():
    s = he.frozen_settings()
    digest = he.settings_sha256(s)
    if digest != SETTINGS_SHA256:
        return f'HE\'s frozen settings have sha256 {digest}, not the pinned {SETTINGS_SHA256} (prereg §5.1 item 2)'
    if s.get('tissue_prior') != 'off' or s.get('name') != 'Hx24-RF05-F1200-v4':
        return f"HE's frozen settings name {s.get('name')} with tissue_prior {s.get('tissue_prior')}, not the method Hx24-RF05-F1200-v4 with tissue_prior off (C22 revision 8)"
    return None


def lpba_pins_problem():
    """LP-B (C26.6-3): the pins of an LPBA40 entry (lpba_common.check_pins: the preregistration 06121278..., the lists, the frozen files 7c4965eb..., d3313868..., the arm table, the method's settings
    414eccd3..., the c record) and the pinned LPBA allowlist (by its path and sha256).  None, or the problem."""
    la = hb.lpba()
    try:
        la.lc.check_pins()
        la.lc.load_lpba_allowlist()
    except la.lc.Refused as exc:
        return str(exc)
    return None


def prereg_problem(repo=None):
    if PREREG_SHA256 is None:
        return 'PREREG_SHA256 is None: the preregistration adopted by BZ after its revision 2 is not pinned (prereg revision 1a item 1); a hold-out batch does not start'
    path = Path(hb.REPO if repo is None else repo)/PREREG_REL
    digest = hb.sha(path) if path.is_file() else None
    return None if digest == PREREG_SHA256 else f'{PREREG_REL}: sha256 {digest} is not the pinned {PREREG_SHA256}'


# ---- one pair -----------------------------------------------------------------------------------------------------------------------------------
def line_of(cell_dir):
    """(path, document) of HE's line.json of the cell: output/line.json (an output) or the cell's line.json (no output); (None, None) when HE wrote neither."""
    for path in (Path(cell_dir)/he.OUTPUT_DIR/'line.json', Path(cell_dir)/'line.json'):
        if path.is_file():
            return path, hb.read_json(path)
    return None, None


def npz_u(path):
    with np.load(path, allow_pickle=False) as z:
        return np.ascontiguousarray(np.array(z['u']))


def certified_check(cell_dir, line_status, *, delete=True, record_path=None):
    """The brief's item 3: for hx_certified, repair/certified.npz's u against output/final.npz's u bit for bit; the same -> certified.npz's sha256 recorded and the file deleted; not the same (or a file
    missing) -> both kept, passed False.  A fallback pair: nothing checked or deleted (its files all kept)."""
    cell_dir = Path(cell_dir)
    certified, final = cell_dir/'repair'/'certified.npz', cell_dir/he.OUTPUT_DIR/'final.npz'
    if line_status != he.HX:
        return dict(applies=False, passed=True, kept=True, note=f'{line_status}: every file of a fallback pair is kept (prereg §5.3)', certified_present=certified.is_file())
    record = dict(applies=True, certified=str(certified), final=str(final), certified_sha256=None, final_sha256=None, deleted=False)
    if not certified.is_file() or not final.is_file():
        return dict(record, passed=False, problem='repair/certified.npz or output/final.npz is not there')
    record.update(certified_sha256=hb.sha(certified), final_sha256=hb.sha(final))
    a, b = npz_u(certified), npz_u(final)
    with np.load(final, allow_pickle=False) as z:
        stored = str(z['state_hash'].item())
    same = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    record.update(u_sha256_certified=he.array_sha256(a), u_sha256_final=he.array_sha256(b), final_state_hash=stored, same=same)
    del a, b
    if not same or record['u_sha256_final'] != stored:
        return dict(record, passed=False, problem='certified.npz\'s u is not output/final.npz\'s (or not its state hash): both kept, to the lead')
    if delete:
        if record_path is not None:                     # HB review m8: the record of the deletion is written before the file goes
            hb.write_once(record_path, dict(record, schema=hb.SCHEMA.format('certified_deleted'), passed=True, utc=hb.utc(), rule='written before repair/certified.npz is deleted'))
        certified.unlink()
        record['deleted'] = True
    return dict(record, passed=True, rule='prereg §5.2 step 4: the same u bit for bit -> certified.npz (an intermediate) deleted, its sha256 recorded')


def rename_attempt(pair, record_root, pair_set='holdout'):
    """A resumed pair: the first attempt's cell directory renamed whole to <pair>.attempt<k> (every file's sha256 before and after recorded, nothing deleted), so that HE writes the resumed attempt into the
    cell's place.  Returns the record (None when there was no cell directory)."""
    cell_dir = hb.cell(pair, pair_set)
    if not cell_dir.exists():
        return None
    files = {str(p.relative_to(cell_dir)): hb.sha(p) for p in sorted(cell_dir.rglob('*')) if p.is_file()}
    target = hb.attempt_dir(cell_dir.with_name(f'{pair}.attempt0'))
    cell_dir.rename(target)
    after = {str(p.relative_to(target)): hb.sha(p) for p in sorted(target.rglob('*')) if p.is_file()}
    doc = dict(schema=hb.SCHEMA.format('attempt_rename'), pair=pair, src=str(cell_dir), dst=str(target), files=files, same=files == after, utc=hb.utc(),
               rule='prereg §5.4: a resumed pair\'s first attempt is kept whole under another name; the resumed attempt writes the cell')
    hb.write_once(hb.numbered(record_root, 'attempt_rename'), doc)
    return doc


def run_one(pair, system, *, batch, log, code, resume=None, pair_set='holdout'):
    """One pair's unit and what it left.  Returns the step-4 record; Stop when the batch must stop."""
    record_root = hb.record_dir(ARM, pair, pair_set)
    record_root.mkdir(parents=True, exist_ok=True)
    lock = hb.lock_path(ARM, pair, pair_set)
    try:
        gate = hb.gate_of(system, 'method', pair_set)
        need_d, need_home = hb.UNIT_NEED_GIB['method']
        disk = system.disk(need_d, need_home)
    except (hc.InfrastructureFailure, hc.Refused) as exc:
        raise hb.Stop(hb.EXIT['infra'], f'{pair}: before the unit: {exc}') from None
    if resume is None:
        hc.check_once(lock)
        lock_file = lock
    else:
        hc.check_once(lock, resume=resume)
        lock_file = Path(resume.lock_path)
    line = hc.lock_line(arm=ARM, pair=pair, pair_set=pair_set, batch=str(batch), code=code.get('commit'), settings_sha256=SETTINGS_SHA256, unit=hb.unit_name('method', pair, pair_set),
                        resume=None if resume is None else dict(path=resume.path, sha256=resume.sha256))
    hc.take_lock(lock_file, line)
    renamed = rename_attempt(pair, record_root, pair_set) if resume is not None else None
    unit_dir = hb.attempt_dir(record_root/('unit_resume' if resume is not None else 'unit'))
    if hb.is_lpba(pair_set):                            # LP-B (C26.6-2): the LPBA subcommand; every process of the pair admits it through the LPBA branch
        inner = [Path(hb.REPO)/'experiments/he/run_hx.py', 'lpba-pair', '--pair-set', pair_set, '--pair', pair, '--device', 'cuda', '--pinned', code.get('commit')]
    else:
        inner = [Path(hb.REPO)/'experiments/he/run_hx.py', 'holdout-pair', '--pair', pair, '--device', 'cuda', '--pinned', code.get('commit')]     # the batch's pinned commit: every process of the pair checks the code directories against it
    log(f'{pair}: unit {hb.unit_name("method", pair, pair_set)} (gate {gate.get("passed")}, disk {disk.get("free_gib")})')
    reading = hb.run_unit('method', pair, inner, system, batch=batch, unit_dir=unit_dir, res=dict(he.SUPERVISOR_UNIT), gpu=True, code=hb.REPO, pair_set=pair_set)
    cell_dir = hb.cell(pair, pair_set)
    path, doc = line_of(cell_dir)
    breaches = sorted(str(p) for p in cell_dir.glob('**/spy_breach_*.json')) if hb.is_lpba(pair_set) and cell_dir.is_dir() else []      # diff review minor 1: read before anything is deleted
    step = dict(schema=hb.SCHEMA.format('step4'), arm=ARM, pair=pair, attempt='resume' if resume is not None else 'first', resume=None if resume is None else dict(path=resume.path, sha256=resume.sha256),
                renamed=renamed, lock=dict(path=str(lock_file), sha256=hb.sha(lock_file)), unit=hb.unit_name('method', pair, pair_set), unit_reading=reading, he_exit=reading.get('child_exit'), cell=str(cell_dir),
                line_json=None if path is None else dict(path=str(path), sha256=hb.sha(path), line_status=(doc or {}).get('line_status')), certified=None, utc=hb.utc())
    stop = None
    if path is None:                                    # prereg revision 1c item 1, runs/20261109-align item 4: no line.json is always infra_failure; the unit's reading is described only
        kind = f"no_line_json:he_exit_{reading.get('child_exit')}:systemd_{reading.get('systemd_result')}"
        step.update(status='infra_failure', kind=kind, unit_reading_described=dict(status=reading['status'], kind=reading['status_kind'],
                                                                                     note='holdout_common.classify_unit_end, described only: the method arm without line.json is infra_failure whatever it reads'))
        stop = hb.Stop(hb.EXIT['infra'], f'{pair}: HE left no line.json (infra_failure; the unit read {reading["status"]} {reading["status_kind"]}): the batch stops; one resume by a committed resume '
                                         'record (prereg §5.4, revision 1c item 1)')
        if resume is not None:                          # prereg §5.4: the resume failed again -> no output (infrastructure), no third run; the cell is written with the resume record
            cell_dir.mkdir(parents=True, exist_ok=True)
            hb.write_terminal(cell_dir, pair=pair, status='infra_failure', written_by='experiments/hb/hb_method.py (the resumed attempt left no line.json either)',
                              rerun_record=dict(path=str(Path(resume.path).resolve().relative_to(Path(hc.AUTH_ROOT).resolve())), sha256=resume.sha256),
                              reason='no output (infrastructure): the resumed attempt failed again; no third run (prereg §5.4)')
    else:
        status = (doc or {}).get('line_status')
        step.update(status=status, kind='he_line_json')
        if status in (he.REJECTED, he.IMPLEMENTATION):
            hb.write_terminal(cell_dir, pair=pair, status=status, written_by='experiments/hb/hb_method.py (HE\'s line.json: no output)', line_json=step['line_json'],
                              reason=(doc or {}).get('rejection') or (doc or {}).get('implementation_error'))
            stop = hb.Stop(hb.EXIT['implementation'], f'{pair}: {status}: the batch stops (P20; C22.5-6), to the lead and BZ; nothing is rerun without BZ\'s authorization')
        elif status in he.OUTPUT_STATUSES and breaches:
            step['certified'] = dict(passed=None, kept=True, rule='LP-B diff review minor 1: a spy breach keeps every file of the cell (certified.npz not deleted; project rules, hard rule 4)')
        elif status in he.OUTPUT_STATUSES:
            step['certified'] = certified_check(cell_dir, status, record_path=record_root/('certified_deleted.json' if resume is None else 'certified_deleted_resume.json'))
            if not step['certified']['passed']:
                stop = hb.Stop(hb.EXIT['stop'], f"{pair}: {step['certified'].get('problem')}")
        else:
            stop = hb.Stop(hb.EXIT['rejudge'], f'{pair}: line.json names the status {status!r}, none of {he.LINE_STATUSES}')
    if breaches:                                        # LP-B review I3 (C26.8-5): a process of the pair reported its allow-list spy's breach: the batch stops, whatever line.json says
        step['spy_breach'] = breaches
        stop = hb.Stop(hb.EXIT['stop'], f'{pair}: the allow-list spy of a process stopped a read ({breaches}): the batch stops (C26.8-5), a tool error first; to BZ, no resume (diff review minor 3)')
    step['went_on'] = stop is None
    hb.write_once(record_root/(STEP4 if resume is None else 'step4_resume.json'), step)
    log(f'{pair}: {step["status"]}' + ('' if stop is None else f' -> STOP ({stop})'))
    if stop is not None:
        raise stop
    return step


# ---- the queue ----------------------------------------------------------------------------------------------------------------------------------
def step_record(pair, pair_set='holdout'):
    """The record of the pair's last attempt: step4_resume.json when the pair was resumed, else step4.json (HB review m9 (i): BZ's decision binds the attempt it is about)."""
    resumed = hb.record_dir(ARM, pair, pair_set)/'step4_resume.json'
    return resumed if resumed.is_file() else hb.record_dir(ARM, pair, pair_set)/STEP4


def bz_declined(pair, pair_set='holdout'):
    """The committed no_resume (未批补跑) record of this cell whose record is the last attempt's record (step_record: step4_resume.json after a resume, else step4.json), or None:
    hb_core.bz_decision_auth (A5b review I2: the selection the finalized record's bz_decision names and the evaluation entry finds by its sha256).  An hx_implementation_error cell: BZ's (甲), its
    reason naming the cell's case of C22.5-6 and (甲) only; an hx_provenance_rejected cell (P20): its reason naming P20 (prereg revision 2 item 17 (f)); a record that does not is not taken."""
    step_path = step_record(pair, pair_set)
    if not step_path.is_file():
        return None
    status = (hb.read_json(hb.terminal_path(hb.cell(pair, pair_set))) or {}).get('line_status')
    return hb.bz_decision_auth(ARM, pair, step_path, line_status=status, kind=hb.implementation_kind(hb.cell(pair, pair_set)), pair_set=pair_set)


def state(pair, pair_set='holdout'):
    """final | done | stopped | not_run (module docstring).  A terminal record is final, except hx_provenance_rejected (P20: not written; a rerun by a committed resume record) and
    hx_implementation_error without BZ's decision (C22.5-6: the queue stops; (甲) is a committed no_resume record of the cell, after which it is final; (乙) a committed resume record)."""
    term = hb.read_json(hb.terminal_path(hb.cell(pair, pair_set)))
    if term is not None:
        if term.get('line_status') == he.REJECTED:          # P20; the lead's item 13: BZ's committed no_resume record (no rerun) makes it written
            return 'final' if bz_declined(pair, pair_set) is not None else 'stopped'
        if term.get('line_status') == he.IMPLEMENTATION and (hb.cell(pair, pair_set)/'review'/'review.json').exists() is False and bz_declined(pair, pair_set) is None:
            return 'stopped'
        return 'final'
    step = hb.read_json(hb.record_dir(ARM, pair, pair_set)/STEP4)
    resumed = hb.read_json(hb.record_dir(ARM, pair, pair_set)/'step4_resume.json')
    for doc in (resumed, step):
        if doc is not None:
            return 'done' if doc.get('went_on') else 'stopped'          # the method arm takes no rejudgment record (decision_log 2026-11-10 N1): resume / no_resume only
    return 'stopped' if hb.lock_path(ARM, pair, pair_set).exists() else 'not_run'


def load_resume(pair, rel, pair_set='holdout'):
    """holdout_common.load_resume for a method cell: tool hb, the cell's lock, its step4.json (or none when the queue died), not complete (no line.json that went on).  An hx_implementation_error
    cell: the resume is BZ's (乙) and its reason must name the cell's case of C22.5-6 and (乙) (runs/20261109-align item 3); Refused otherwise."""
    if hb.is_lpba(pair_set):
        breach = hb.lpba().spy_breach_problem(hb.cell(pair, pair_set), hb.record_dir(ARM, pair, pair_set))
        if breach:                                      # LP-B diff review minor 3: a spy breach goes to BZ, never through a resume
            raise hc.Refused('resume', f'{pair} ({pair_set}): {breach}')
        if hb.lpba_resume_problem(pair_set):
            raise hc.Refused('resume', hb.lpba_resume_problem(pair_set))
    resume = hc.load_resume(rel, tool=hb.TOOL, arm=ARM, pair=pair, pair_set=pair_set, lock=hb.lock_path(ARM, pair, pair_set), previous_record=hb.record_dir(ARM, pair, pair_set)/STEP4,
                            complete=lambda: state(pair, pair_set) in ('final', 'done'))
    kind = hb.implementation_kind(hb.cell(pair, pair_set))
    if (hb.read_json(hb.terminal_path(hb.cell(pair, pair_set))) or {}).get('line_status') == he.IMPLEMENTATION:
        problem = hb.decision_reason_problem(resume.document, kind, '(乙)')
        if problem:
            raise hc.Refused('resume', f'{pair}: {problem}')
    return resume


def existing_pins(pair_set='holdout'):
    """The pinned commits the HB locks already name (locks/hb__<arm>__<pair>.lock and review/review.lock: their first line's `code`): one pinned commit for every HB unit (C22.H0-4)."""
    pins = set()
    lock_root = hb.lpba().lock_dir(pair_set) if hb.is_lpba(pair_set) else Path(hc.LOCK_ROOT)
    paths = sorted(lock_root.glob('hb__*__*.lock'))+[hb.review_root(pair_set)/'review.lock']
    for path in paths:
        try:
            line = json.loads(Path(path).read_text(encoding='utf-8').splitlines()[0])
        except (OSError, ValueError, IndexError):
            continue
        if isinstance(line, dict) and line.get('code'):
            pins.add(line['code'])
    return pins


def start_checks(commit=None, code_run=None):
    """The queue's start (module docstring): (record, problems).  Nothing written.  The code: holdout_common.pinned_code_problems -- P1 from the committed runs/holdout_B_v1/pinned_code.json
    (never HEAD; --commit only a check), every commit of P1..HEAD checked (rule one); the HB locks' commits must all be P1 (C22.H0-4: one pinned commit from step 4 to 7)."""
    record, problems = dict(), []
    pins = existing_pins()
    record['existing_pins'] = sorted(pins)
    if len(pins) > 1:
        problems.append(f'the HB locks name more than one pinned commit: {sorted(pins)} (C22.H0-4)')
    try:
        record['code'] = hb.code_record(commit, run=code_run)          # P1 from runs/holdout_B_v1/pinned_code.json; --commit only a check (decision_log 2026-11-10 ~2:59 AM CST (1))
        if pins and record['code']['commit'] not in pins:
            problems.append(f"the pinned commit {record['code']['commit']} is not the one the HB locks name {sorted(pins)} (C22.H0-4)")
    except hb.Stop as exc:
        problems.append(str(exc))
    for name, fn in (('settings', settings_problem), ('preregistration', prereg_problem)):
        problem = fn()
        record[name] = problem
        if problem:
            problems.append(problem)
    try:
        record['unseal'] = hb.unseal_gate()
    except hc.Refused as exc:
        record['unseal'] = str(exc)
        problems.append(str(exc))
    return record, problems


def lpba_start_checks(pair_set, commit=None, code_run=None):
    """LP-B: the start of an LPBA mode's queue (module docstring's last paragraph): (record, problems); nothing written, no file of a subject read.  The code (lpba_arms.code_record: formal,
    rule one -- LP-C2's, C26.9-4, fail closed; feasibility, a clean checkout, the commit HEAD), the LPBA locks' commits all the batch's (C22.H0-4's rule), HE's frozen settings the pinned ones
    (C26.6-1), the pins and the pinned LPBA allowlist (lpba_pins_problem, C26.6-3)."""
    la = hb.lpba()
    record, problems = dict(pair_set=pair_set), []
    pins = existing_pins(pair_set)
    record['existing_pins'] = sorted(pins)
    if len(pins) > 1:
        problems.append(f'the {pair_set} HB locks name more than one commit: {sorted(pins)} (C22.H0-4)')
    record['code'], code_problems = la.code_record(pair_set, commit, run=code_run)
    problems += code_problems
    if pins and record['code'].get('commit') not in pins:
        problems.append(f"the commit {record['code'].get('commit')} is not the one the {pair_set} HB locks name {sorted(pins)} (C22.H0-4)")
    for name, fn in (('settings', settings_problem), ('pins', lpba_pins_problem)):
        problem = fn()
        record[name] = problem
        if problem:
            problems.append(problem)
    return record, problems


def checks_of(pair_set, commit=None, code_run=None):
    """LP-B: the start checks of a pair set -- the hold-out's start_checks(commit, code_run) (looked up at call time, called as before), an LPBA mode's lpba_start_checks."""
    if hb.is_lpba(pair_set):
        return lpba_start_checks(pair_set, commit, code_run)
    return start_checks(commit, code_run)


def queue(*, pairs=None, resumes=None, commit=None, system=None, echo=print, code_run=None, pair_set='holdout'):
    system = hb.System() if system is None else system
    resumes = dict(resumes or {})
    record, problems = start_checks(commit, code_run) if not hb.is_lpba(pair_set) else lpba_start_checks(pair_set, commit, code_run)
    if problems:
        echo(f'refused at the start: {problems}')
        return hb.EXIT['refused']
    lpba = hb.is_lpba(pair_set)
    batch = hb.new_batch_dir('method', pair_set)
    log = hb.Log(batch, echo)
    record.update(schema=hb.SCHEMA.format('batch'), kind='method', settings_sha256=SETTINGS_SHA256, prereg_sha256=hb.lpba().lc.PREREG_SHA256 if lpba else PREREG_SHA256, utc_start=hb.utc(),
                  disk_start=hc.disk_reading(), cells=[], gate=dict(mem_gib=hb.GATE_MEM_GIB, wait_seconds=hb.GATE_WAIT_SECONDS['method']), unit=dict(he.SUPERVISOR_UNIT))
    if lpba:
        record['pair_set'] = pair_set                   # LP-B review I1: only an LPBA batch has the key (the hold-out's batch.json keys unchanged)
    code = 0
    try:
        listed = hb.rows_of(pair_set)
        plan = list(listed)
        if pairs:
            unknown = [p for p in pairs if p not in plan]
            if unknown:
                raise hb.Stop(hb.EXIT['refused'], f'not rows of the pair list: {unknown}')
            plan = [p for p in plan if p in pairs]
        todo = [p for p in plan if state(p, pair_set) == 'not_run']
        runs = [p for p in plan if state(p, pair_set) not in ('final', 'done')]
        if runs and lpba:                               # LP-B: HI's LPBA40 output complete over the mode's subjects before any unit reads pred/<s>/
            split = hb.lpba().split
            record['hi'], hi_problems = hb.lpba_hi(pair_set, sorted({s for p in listed for s in split(p)}), hashed=sorted({s for p in runs for s in split(p)}))
            if hi_problems:
                raise hb.Stop(hb.EXIT['refused'], f"HI's {pair_set} predictions are not complete or a record does not hold: {hi_problems[:5]} ({len(hi_problems)}); no unit started")
        elif runs:                                      # A5b item 13 (the A5a notes' table, HE's row): HI complete over its runs before any unit reads pred/<s>/
            record['hi'], hi_problems = hb.hi_union(sorted({s for p in listed for s in hc.split_pair(p)}), hashed=sorted({s for p in runs for s in hc.split_pair(p)}))
            if hi_problems:
                raise hb.Stop(hb.EXIT['refused'], f"HI's hold-out predictions are not the union of its runs over the pair list's subjects, or a record does not hold (A5a review I2; A5b item 13): "
                                                  f'{hi_problems[:5]} ({len(hi_problems)}); no unit started')
        try:
            record['disk_batch'] = system.disk(0., len(todo)*hb.KEEP_HOME_GIB['method']+hb.UNIT_NEED_GIB['method'][1])
        except hc.InfrastructureFailure as exc:
            raise hb.Stop(hb.EXIT['infra'], f'the batch disk guard: {exc}') from None
        log(f'start: {len(plan)} pairs, {len(todo)} to run')
        for pair in plan:
            st = state(pair, pair_set)
            row = dict(pair=pair, state_before=st)
            record['cells'].append(row)
            if st in ('final', 'done'):
                log(f'skip {pair}: {st}')
                continue
            resume = None
            if st == 'stopped':
                if pair not in resumes:
                    raise hb.Stop(hb.EXIT['resume'], f'{pair}: a lock without a record that went on; a rerun needs a committed resume record (prereg §5.4)')
                try:
                    resume = load_resume(pair, resumes[pair], pair_set)
                except hc.Refused as exc:
                    raise hb.Stop(hb.EXIT['refused'], f'{pair}: {exc}') from None
            elif pair in resumes:
                raise hb.Stop(hb.EXIT['refused'], f'{pair}: a resume record for a pair that never took its lock')
            while hb.hold_path(pair_set).exists():
                log('hold file present; waiting')
                system.sleep(60)
            step = run_one(pair, system, batch=batch, log=log, code=record['code'], resume=resume, pair_set=pair_set)
            row['status'] = step['status']
    except hb.Stop as exc:
        code = exc.code
        record['stop'] = dict(code=code, reason=str(exc))
        log(f'STOP ({code}): {exc}; nothing rerun; to the lead')
    except hc.Refused as exc:
        code = hb.EXIT['refused']
        record['stop'] = dict(code=code, reason=str(exc))
        log(f'STOP ({code}): {exc}')
    except Exception as exc:                            # noqa: BLE001 - HB review m11: the batch record is written whatever stopped the queue
        code = hb.EXIT['stop']
        record['stop'] = dict(code=code, reason=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()[-4000:])
        log(f'STOP ({code}): an exception of the queue: {type(exc).__name__}: {exc}')
    record.update(utc_end=hb.utc(), disk_end=hc.disk_reading(), exit=code)
    hb.write_once(batch/'batch.json', record)
    if code == 0:
        log('end')
    return code


def no_resume(pair, rel, pair_set='holdout'):
    """The cell's terminal record infra_failure with the committed 未批补跑 record (prereg revision 1c item 4, holdout_common.no_resume_authorization): only for a pair that stopped without HE's line.json."""
    if hb.is_lpba(pair_set) and hb.lpba_resume_problem(pair_set):
        raise hc.Refused('no_resume', hb.lpba_resume_problem(pair_set))
    step_path = step_record(pair, pair_set)             # HB review m9 (i): the last attempt's record
    step = hb.read_json(step_path)
    implementation = step is not None and step.get('status') == he.IMPLEMENTATION and not step.get('went_on')
    rejected = step is not None and step.get('status') == he.REJECTED and not step.get('went_on')       # the lead's item 13: P20, BZ does not let it rerun
    if step is None or step.get('went_on') or (step.get('line_json') is not None and not (implementation or rejected)):
        raise hc.Refused('no_resume', f'{pair}: only a pair whose unit left no line.json (or ended in hx_implementation_error or hx_provenance_rejected) and stopped the batch takes a no-resume record')
    auth = hc.no_resume_authorization(tool=hb.TOOL, arm=ARM, pair=pair, pair_set=pair_set, record=step_path, complete=lambda: False,
                                      accept=lambda a: Path(a.path).resolve() == (Path(hc.AUTH_ROOT)/rel).resolve())
    if auth is None:
        raise hc.Refused('no_resume', f'{rel}: not a committed no_resume record of this pair whose record is {step_path} (sha256 now)')
    if rejected:                                        # P20 (the lead's item 13): the terminal record (hx_provenance_rejected) stays; state() reads this record: written, the others go on
        problem = hb.rejudgment_io.p20_reason_problem(auth.document.get('reason'))      # A5b review I2: the reason names P20, as the evaluation entry requires
        if problem:
            raise hc.Refused('no_resume', f'{pair}: {problem}')
        return hb.terminal_path(hb.cell(pair, pair_set))
    if implementation:                                  # C22.5-6, BZ's (甲): the pair stays without an output (its terminal record is HE's hx_implementation_error), the others go on: state() reads this record
        problem = hb.decision_reason_problem(auth, hb.implementation_kind(hb.cell(pair, pair_set)), '(甲)')
        if problem:
            raise hc.Refused('no_resume', f'{pair}: {problem}')
        return hb.terminal_path(hb.cell(pair, pair_set))
    cell_dir = hb.cell(pair, pair_set)
    cell_dir.mkdir(parents=True, exist_ok=True)
    return hb.write_terminal(cell_dir, pair=pair, status='infra_failure', written_by='experiments/hb/hb_method.py no-resume', rerun_record=dict(path=rel, sha256=auth.sha256),
                             reason=f"{step.get('status')} ({step.get('kind')}); BZ does not let it rerun (未批补跑)", evidence=dict(step4=str(step_path), step4_sha256=hb.sha(step_path)))


def parse_resumes(values):
    out = {}
    for value in values or []:
        pair, _, rel = value.partition('=')
        if not rel:
            raise SystemExit(f'--resume {value!r}: PAIR=REL')
        out[pair] = rel
    return out


PAIR_SET_CHOICES = ('holdout',)+hb.hold.LPBA_PAIR_SETS     # LP-B: the hold-out (default) and the two LPBA40 modes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    d = sub.add_parser('dry-run')
    q = sub.add_parser('queue')
    q.add_argument('--commit', default=None)
    q.add_argument('--pairs', default=None)
    q.add_argument('--resume', action='append', default=[])
    ls = sub.add_parser('launch-string')
    ls.add_argument('--commit', default=None)
    n = sub.add_parser('no-resume')
    n.add_argument('--pair', required=True)
    n.add_argument('--authorization', required=True)
    n.add_argument('--commit', default=None, help='only a check: must be P1 of runs/holdout_B_v1/pinned_code.json (rule one)')
    for p in (d, q, ls, n):
        p.add_argument('--pair-set', choices=PAIR_SET_CHOICES, default='holdout', help='LP-B: an LPBA40 mode (lpba40_test, lpba40_feas); the hold-out by default')
    o = ap.parse_args(argv)
    if o.command == 'dry-run':
        record, problems = checks_of(o.pair_set)
        print(json.dumps(dict(record, problems=problems), indent=1, default=str))
        return hb.EXIT['refused'] if problems else 0
    if o.command == 'launch-string':
        args = ['queue', *(['--commit', o.commit] if o.commit else []), *(['--pair-set', o.pair_set] if o.pair_set != 'holdout' else [])]
        print(' '.join(hb.queue_launch('method', args, pair_set=o.pair_set)))
        return 0
    if o.command == 'no-resume':
        _, problems = checks_of(o.pair_set, o.commit)     # the code (holdout_common.pinned_code_problems), the pins, the unseal gate: as every HB start
        if problems:
            print(f'refused at the start: {problems}', file=sys.stderr)
            return hb.EXIT['refused']
        try:
            print(no_resume(o.pair, o.authorization, o.pair_set))
        except (hc.Refused, FileExistsError) as exc:      # HB review m11: a terminal record already there
            print(f'refused: {exc}', file=sys.stderr)
            return hb.EXIT['refused']
        return 0
    return queue(pairs=o.pairs.split(',') if o.pairs else None, resumes=parse_resumes(o.resume), commit=o.commit, pair_set=o.pair_set)


if __name__ == '__main__':
    sys.exit(main())
