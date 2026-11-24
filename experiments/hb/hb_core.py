"""HB (runs/20261109-hb/任务书.md; contract C22.H0, C22.H1, C22.5-6, C23.8-2; preregistration docs/bayesreg/notes/留出集预登记_20261107.md §2.2, §5.2 steps 4-8, §5.3, §5.4 and revisions 1-1c): what the three
hold-out queues of HB share -- the method arm (step 4, hb_method.py), the step-5 review certificate unit (hb_review.py) and the configuration C arm (hb_configc.py) -- and the two tools after them
(hb_store.py: the step-8 input lock, the move to D:).

Names and places (holdout_common's roots; the lead's storage rule, decision_log 2026-11-07 about 5:10 PM CST):
  the method arm 'method'      its cell is HE's output directory holdout_B_v1/work/method/<pair>/ (HE writes it; step 5 adds review/ and terminal.json); the queue's records per pair in
                               holdout_B_v1/method/<pair>/ (unit/, unit_reading.json, step4.json, attempt renames); the whole cell moves to /mnt/d/seq_vp_tag_holdout_B_v1/method/<pair>/ only after step 5
                               (C22.H1-1: not before, the third certificate's fallback runs in the cell), by hb_store.py move
  the review                   holdout_B_v1/review/: input_manifest.json, review.lock (the batch lock, O_EXCL), <pair>.lock (each pair once), finalized_outputs.json (C22.H1-3), supplements/
  configuration C 'configc'    its cell holdout_B_v1/work/configc/<pair>/ (the run, output/), records holdout_B_v1/configc/<pair>/ (result.json: the terminal record), moved to D: by the driver
  units                        tag-hb-he-<ffff>-<mmmm>, tag-hb-review-<ffff>-<mmmm>, tag-hb-c-<ffff>-<mmmm>; the queues' own units tag-hb-he-queue, tag-hb-review-queue, tag-hb-c-queue.  HE's own
                               nested units keep their names (tag-he-c-<pair>, tag-he-c3-<pair>; preregistration revision 1c item 5).
  locks                        holdout_B_v1/locks/hb__<arm>__<pair>.lock (one per cell, O_EXCL, never edited or deleted)
  batches                      holdout_B_v1/batches/hb_<kind>_<UTC>/ (batch.json, queue.log, units/, refusals/)
Every unit's main process is experiments/xb/xb_unit.py (python -I -B: unit_start.json, the child, unit_end.json with memory.events read while the unit lives, the kernel's oom-kill lines in the window);
its end is read with holdout_common.classify_unit_end (rule (乙) by default; preregistration revision 1c item 1; decision_log 2026-11-09).
The terminal record of a method cell (terminal.json in the cell; the evaluation entry's "written", D107): written once, by the review unit for every finalized pair (the final line status, the review
record), by the method queue for a pair HE ended without an output (hx_provenance_rejected, hx_implementation_error; the batch stops), by hb_method.py no-resume for an infrastructure failure BZ does
not let rerun (a committed 未批补跑 record).  A cell without terminal.json is not written.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REPO = EXPERIMENTS.parent
for _dir in ('he', 'holdout'):
    if str(EXPERIMENTS/_dir) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS/_dir))
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))

import holdout_common as hc  # noqa: E402
import he_core as he  # noqa: E402
import he_holdout as hold  # noqa: E402

from bayesreg.io import rejudgment as rejudgment_io  # noqa: E402

SCHEMA = 'bayesreg.hb.{}.v1'
TOOL = 'hb'                                     # the committed authorizations: splits/bayesreg_v1/holdout_B_v1_hb_<kind>_<k>.json (holdout_common.load_authorization)
ARM = hold.ARM                                  # 'method'
CONFIGC_ARM = 'configc'
PY = Path('/home/rmcfz/miniconda3/envs/sci310/bin/python')
XB_UNIT = 'experiments/xb/xb_unit.py'           # the in-unit supervisor (relative to the code tree)
KINDS = ('method', 'review', 'configc')
UNIT_PREFIX = dict(method='tag-hb-he-', review='tag-hb-review-', configc='tag-hb-c-')
QUEUE_UNIT = dict(method='tag-hb-he-queue', review='tag-hb-review-queue', configc='tag-hb-c-queue')
# The gates (C22.H1-8, D126; prereg §5.3, §5.4): every unit of the three can start configuration C's frozen run (HE's fallback one, the third certificate's fallback, the arm itself), so MemAvailable
# >= he_core.FROZEN_RUN_GATE_GIB (10 GiB, HE2's "gate of c"); the GPU idle; no other tag-* or seq-vp-* unit (the queue's own only when this process runs in it); the 1-minute load < 1.5.  The waits:
# 15 minutes for the arms (prereg §5.4: a precheck that does not pass in 15 minutes is an infrastructure failure), 30 minutes for the review (D126 (a): written here, recorded in the batch record, not
# an option).
GATE_MEM_GIB = he.FROZEN_RUN_GATE_GIB
GATE_WAIT_SECONDS = dict(method=hc.GATE_WAIT_SECONDS, review=30*60, configc=hc.GATE_WAIT_SECONDS)
# (D:, /home) GiB a unit needs while it runs (prereg §5.3, §6.2): HE's output, field and repair (~.6 GB) and a fallback's configuration C run (~5.5 GB) on /home; nothing on D: (the move comes after)
UNIT_NEED_GIB = dict(method=(0., 6.5), review=(0., 6.), configc=(0., 6.5))
KEEP_HOME_GIB = dict(method=.6, review=0., configc=0.)       # per pair, kept on /home until the move (the method arm: step 5 first, C22.H1-1)
# The units' settings: the method's supervisor and the review are HE's frozen units (he_core.SUPERVISOR_UNIT, EVALUATION_UNIT: C22.H1-8 "复用 units.evaluation，不改"); configuration C's arm is D0's
# frozen unit (firewall_c.UNIT_RESOURCES: CPUQuota 1200 %, MemoryMax 12G, MemorySwapMax 4G) with OOMPolicy continue, 8 threads and a RuntimeMaxSec of its own (the prepare, the 300 s watchdog, the
# certificate, the move: 1800 s; the lead may change it).
CONFIGC_UNIT = dict(memory_max='12G', memory_swap_max='4G', oom_policy='continue', quota=1200, numba=8, runtime=1800, nice=None)
EXIT = dict(done=0, infra=20, refused=22, stop=21, resume=24, rejudge=25, implementation=26)


class Stop(Exception):
    """The batch stops (prereg §5.4; nothing is rerun; the evidence stays)."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def write_once(path, document):
    """holdout_common.write_json_once (the real roots refused in the test sandbox)."""
    return hc.write_json_once(path, document)


def sha(path):
    return hc.sha256_file(path)


# ---- the method cell ----------------------------------------------------------------------------------------------------------------------------
# LP-B (contract C26.6, C26.2-3, -5): every place below takes the pair set, 'holdout' by default (the hold-out's places and names, unchanged); an LPBA40 mode ('lpba40_test', 'lpba40_feas') has the
# hold-out's layout under its own root (experiments/lpba/lpba_arms.py: <root>/work/<arm>/<pair>, <root>/<arm>/<pair>, <root>/locks, <root>/batches, <root>/review; nothing moves to D:) and the
# units tag-lp-he-, tag-lp-review-, tag-lp-c- (tag-lp-feas-...).
LPBA_UNIT_KIND = dict(method='he', review='review', configc='c')


def is_lpba(pair_set):
    return pair_set in hold.LPBA_PAIR_SETS


def lpba():
    """experiments/lpba/lpba_arms.py (through he_holdout, imported when an LPBA mode is first met)."""
    return hold._lpba()


def cell(pair, pair_set='holdout'):
    """The method cell: HE's output directory of the pair (holdout_common.work_dir('method', pair); an LPBA mode: he_holdout.lpba_out_dir)."""
    if is_lpba(pair_set):
        return Path(hold.lpba_out_dir(pair_set, pair))
    return Path(hold.out_dir(pair))


def record_dir(arm, pair, pair_set='holdout'):
    if is_lpba(pair_set):
        return lpba().record_dir(arm, pair, pair_set)
    return Path(hc.record_dir(arm, pair))


def lock_path(arm, pair, pair_set='holdout'):
    if is_lpba(pair_set):
        return lpba().lock_dir(pair_set)/f'hb__{arm}__{pair}.lock'
    return Path(hc.LOCK_ROOT)/f'hb__{arm}__{pair}.lock'


def review_root(pair_set='holdout'):
    if is_lpba(pair_set):
        return lpba().root(pair_set)/'review'
    return Path(hc.guard_root(Path(hc.HOLDOUT_ROOT)/'review'))


def hold_path(pair_set='holdout'):
    """The hold file a queue waits on (the hold-out's holdout_B_v1/hold; an LPBA mode's lpba40_test_v1/hold, C26.2-3)."""
    if is_lpba(pair_set):
        return lpba().hold(pair_set)
    return Path(hc.HOLDOUT_ROOT)/'hold'


def unit_name(kind, pair, pair_set='holdout'):
    if is_lpba(pair_set):
        return lpba().unit_name(LPBA_UNIT_KIND[kind], pair_set, pair)
    return f'{UNIT_PREFIX[kind]}{hc.short_pair(pair)}'


def queue_unit(kind, pair_set='holdout'):
    """The queue's own unit: tag-hb-he-queue, tag-hb-review-queue, tag-hb-c-queue; an LPBA mode's tag-lp[-feas]-he-queue, -review-queue, -c-queue."""
    if is_lpba(pair_set):
        return lpba().unit_name(LPBA_UNIT_KIND[kind], pair_set)
    return QUEUE_UNIT[kind]


def terminal_path(cell_dir):
    return Path(cell_dir)/'terminal.json'


TERMINAL_STATUSES = (*he.OUTPUT_STATUSES, he.REJECTED, he.IMPLEMENTATION, 'infra_failure')


def write_terminal(cell_dir, *, pair, status, written_by, output_dir=None, review=None, line_json=None, rerun_record=None, reason=None, evidence=None):
    """The method cell's terminal record (module docstring), once.  status: an output status (output_dir 'output', the review record), hx_provenance_rejected or hx_implementation_error (no output; the
    line.json or the review record says why), or infra_failure (with rerun_record: the committed 未批补跑 record)."""
    if status not in TERMINAL_STATUSES:
        raise ValueError(f'terminal status {status!r}: one of {TERMINAL_STATUSES}')
    if (status in he.OUTPUT_STATUSES) != (output_dir is not None):
        raise ValueError('an output status names its output directory; no other status does')
    doc = dict(schema=SCHEMA.format('terminal'), arm=ARM, pair=pair, line_status=status, output_dir=output_dir, review_record=review, line_json=line_json, rerun_record=rerun_record, reason=reason,
               evidence=evidence, written_by=written_by, utc=utc(),
               rule='the method cell\'s terminal record (experiments/hb/hb_core.py): the evaluation entry reads line_status (D107); output statuses have output/ and the step-5 review record')
    return write_once(terminal_path(cell_dir), doc)


def remove_empty_tree(directory):
    """After a move (every file moved, sha256 before and after equal): the directories left empty, deepest first, then the directory itself if empty.  Never a file."""
    directory = Path(directory)
    if not directory.is_dir():
        return
    for p in sorted((q for q in directory.rglob('*') if q.is_dir()), key=lambda q: len(q.parts), reverse=True):
        if not any(p.iterdir()):
            p.rmdir()
    if not any(directory.iterdir()):
        directory.rmdir()


CONFIGC_JUDGMENTS = ('unit_limit', 'no_output')    # the configuration C arm's needs_lead_rejudgment (decision_log 2026-11-10 N1): written; infrastructure only classifies
CONFIGC_RESUMED_JUDGMENTS = ('unit_limit', 'no_output', 'infrastructure')     # N2 (i): after a resume, infrastructure again is written too (revision 1a item 2 (iii)): no output, no third run


def rejudged(arm, pair, record, accept, *, pair_set='holdout'):
    """The lead's item 13 (decision_log 2026-11-10 NI2, N1, N2): the cell's committed rejudgment record (holdout_common.rejudgment) that judges `record` (its sha256 now) with a judgment in
    `accept`: the cell is written and the batch goes past it.  (found or None, problems).  Used by the configuration C queue only (the method queue takes no rejudgment; the review's are a trace)."""
    found, problems = hc.rejudgment(arm, pair, record=record, pair_set=pair_set)
    if found is not None and found['document']['judgment'] not in accept:
        return None, [f"{found['path']}: judgment {found['document']['judgment']!r} does not let the batch go past (only {accept}; infrastructure only classifies)"]
    return found, problems


def implementation_kind(cell_dir):
    """'a' or 'b' (C22.5-6) of a cell whose terminal record is hx_implementation_error: the step-5 review record's, else HE's line.json's; None otherwise."""
    cell_dir = Path(cell_dir)
    for path in (cell_dir/'review'/'review.json', cell_dir/'line.json'):
        doc = read_json(path)
        if doc is not None and doc.get('line_status') == he.IMPLEMENTATION:
            return (doc.get('implementation_error') or {}).get('kind')
    return None


def decision_reason_problem(auth, kind, which):
    """runs/20261109-align item 3: BZ's decision on an hx_implementation_error (甲: a no_resume record; 乙: a resume record) names in its reason which case of C22.5-6 it is: 'C22.5-6 (a)' or
    'C22.5-6 (b)', the cell's own (implementation_kind, which must be known here), and which decision ('(甲)' for no_resume, '(乙)' for resume).  The criterion is bayesreg.io.rejudgment's
    decision_reason_problem (A5b review I2: one criterion for HB, holdout_common and the evaluation entry).  None, or why not."""
    reason = (auth.document if hasattr(auth, 'document') else auth).get('reason')
    if kind not in ('a', 'b'):
        return f'the cell\'s case of C22.5-6 is not known ({kind!r}): BZ\'s decision {which} is not taken (runs/20261109-align item 3)'
    return rejudgment_io.decision_reason_problem(reason, kind, which)


def bz_decision_auth(arm, pair, record, *, line_status, kind=None, accept=None, pair_set='holdout'):
    """BZ's decision on a method cell that ended without an output (A5b review I2, the lead's ruling 11-10 ~9:22 PM CST): the one selection HB writes into the finalized record's bz_decision
    (hb_review.finalize) and the evaluation entry then finds by that sha256.  holdout_common.no_resume_authorization (tool hb, `arm`, the pair, holdout; its record `record` at its sha256 now;
    note 未批补跑; the first in name order that holds) whose reason meets bayesreg.io.rejudgment's criterion: hx_provenance_rejected -> it names P20 (p20_reason_problem); hx_implementation_error
    -> (甲) and the cell's case of C22.5-6 (decision_reason_problem; the case must be known: kind None takes no record, as hb_method.no_resume refuses one -- A5b diff review
    minor 1).  `accept`: a further condition (a withdrawal's given file).  None when none holds, or for another line status."""
    def holds(auth):
        reason = auth.document.get('reason')
        if line_status == he.REJECTED:
            good = rejudgment_io.p20_reason_problem(reason) is None
        elif line_status == he.IMPLEMENTATION:
            good = decision_reason_problem(auth, kind, '(甲)') is None          # kind not 'a' or 'b': refused (the case of the cell is not known)
        else:
            good = False
        return good and (accept is None or accept(auth))
    if is_lpba(pair_set) and not lpba_authorizations(pair_set):
        return None                                     # LP-B: no authorization directory of this mode (C26.8-3: LP-C2's; the feasibility mode has none): no decision, the pair stays stopped
    return hc.no_resume_authorization(tool=TOOL, arm=arm, pair=pair, pair_set=pair_set, record=record, complete=lambda: False, accept=holds)


def lpba_authorizations(pair_set):
    """LP-B: whether holdout_common knows the committed authorizations of an LPBA mode (AUTH_DIRS; C26.8-3, D148: splits/lpba40_test_v1/lpba40_test_v1_*, LP-C2's; the feasibility mode has none,
    C26.6-2).  Without them a stopped LPBA cell stays stopped (no resume, no decision record is read)."""
    return pair_set in getattr(hc, 'AUTH_DIRS', {}) and pair_set in getattr(hc, 'AUTH_PREFIX', {})


def lpba_resume_problem(pair_set):
    """None when an LPBA mode's resume or no_resume records can be read (lpba_authorizations), else why not (to the lead: refused, nothing rerun)."""
    if lpba_authorizations(pair_set):
        return None
    return (f'{pair_set}: no committed authorizations of this mode are known to holdout_common (AUTH_DIRS; C26.8-3, D148: the formal mode\'s are LP-C2\'s, the feasibility mode has none -- '
            'a feasibility failure is fixed in the code and run again on a new root, C26.8-5): nothing is resumed')


def file_rows(directory, names):
    """{name: sha256 or None (not there)} of files relative to `directory`."""
    directory = Path(directory)
    return {n: (sha(directory/n) if (directory/n).is_file() else None) for n in names}


OUTPUT_FILES = ('output/final.npz', 'output/disp.npz', 'output/cert.json', 'output/line.json')
INPUT_FILES = OUTPUT_FILES+('t88/run.json', 't88/field_0600.npz', 'repair/repair.json', 'repair/repair.npz')     # C22.H1-8 the input manifest; C22.H1-3 the finalized record


# ---- the gate, the disks, the code ---------------------------------------------------------------------------------------------------------------
def gate_once(kind, *, run=None, pair_set='holdout'):
    """holdout_common.machine_gate with this queue's own unit (excluded only when this process runs in it) and MemAvailable >= GATE_MEM_GIB.  Refused (stage machine) otherwise."""
    return hc.machine_gate(own_units=(queue_unit(kind, pair_set),), mem_min_gib=GATE_MEM_GIB, run=run)


def gate_of(system, kind, pair_set='holdout'):
    """system.gate of a queue: the hold-out's and the development's call as before (system.gate(kind)); an LPBA mode's with its pair set (its own queue unit)."""
    return system.gate(kind, pair_set) if is_lpba(pair_set) else system.gate(kind)


def unseal_gate():
    """A hold-out queue's first check, before any list of the hold-out is read: bayesreg.io.unseal.load_unseal on the pinned record of bayesreg's code tree (C23.3-1); without a valid record (main today)
    hc.Refused (stage unseal).  The units admit each pair again in their own processes (he_holdout.admitted_images, hb_configc)."""
    from bayesreg.io import sealed as sealed_module
    from bayesreg.io import unseal as unseal_module
    from bayesreg.io.guard import SubjectSealed
    try:
        unseal = unseal_module.load_unseal(Path(sealed_module.REPO)/unseal_module.UNSEAL_REL)
    except SubjectSealed as exc:
        raise hc.Refused('unseal', f'no valid unseal object ({exc}); the hold-out is sealed until BZ unseals it') from None
    if unseal.sealed_list != hc.UNSEALED_LIST:
        raise hc.Refused('unseal', f'the unseal object unseals {unseal.sealed_list!r}, not {hc.UNSEALED_LIST!r}')
    return unseal.record()


def code_record(commit=None, run=None):
    """The code identity of a hold-out batch or unit (rule one; decision_log 2026-11-10 ~2:59 AM CST (1)): holdout_common.pinned_code_problems -- the pinned commit P1 from the committed
    runs/holdout_B_v1/pinned_code.json, never HEAD; `commit` (--commit) only compared with it.  Returns holdout_common.code_identity's record with `commit` = P1 and `pinned_code` the check;
    Stop (refused) otherwise."""
    code = hc.code_identity(run=run)
    record, problems = hc.pinned_code_problems(commit, run=run)
    if problems:
        raise Stop(EXIT['refused'], f'the code is not the pinned commit\'s (rule one; runs/holdout_B_v1/pinned_code.json): {problems}')
    return dict(code, head=code.get('commit'), commit=record['pinned'], pinned_code=record)


# ---- the unit ------------------------------------------------------------------------------------------------------------------------------------
def unit_argv(kind, pair, inner, *, code, unit_dir, res, gpu, log, pair_set='holdout'):
    """systemd-run with the unit's settings `res` (he_core.unit_argv_problems checks the string carries each value once), the environment (PYTHONPATH <frozen>:<code>, single-threaded BLAS, NUMBA
    threads of the unit, no GPU for a CPU unit), the log appended, and the in-unit supervisor xb_unit.py around `inner` (python -B and its arguments)."""
    code = Path(code)
    unit = unit_name(kind, pair, pair_set)
    env = dict(OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, NUMBA_NUM_THREADS=res['numba'], PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=f'{he.core.FROZEN}:{code}')
    if not gpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    argv = ['systemd-run', '--user', f'--unit={unit}', '--wait', '--collect', f'--working-directory={code}', f"--property=MemoryMax={res['memory_max']}",
            f"--property=MemorySwapMax={res['memory_swap_max']}", f"--property=OOMPolicy={res['oom_policy']}", *([] if res.get('nice') is None else [f"--nice={res['nice']}"]),
            f"--property=CPUQuota={res['quota']}%", f"--property=RuntimeMaxSec={res['runtime']}", f'--property=StandardOutput=append:{log}', f'--property=StandardError=append:{log}',
            *[f'--setenv={k}={v}' for k, v in env.items()], str(PY), '-I', '-B', str(code/XB_UNIT), '--record-dir', str(unit_dir), '--unit', unit, '--', str(PY), '-B', *[str(a) for a in inner]]
    problems = he.unit_argv_problems(argv, res)
    if problems:
        raise ValueError(f'{unit}: the start string is not the unit {res}: {problems}')
    if 'pytest' in ' '.join(argv):
        raise ValueError(f'start string with a forbidden word: {argv}')
    return argv


def numbered(directory, stem, suffix='.json'):
    """<directory>/<stem><suffix>, or <stem>_2<suffix>, ...: the first not there."""
    path, k = Path(directory)/f'{stem}{suffix}', 1
    while path.exists():
        k += 1
        path = Path(directory)/f'{stem}_{k}{suffix}'
    return path


def attempt_dir(base):
    """base, or base_2, base_3, ...: the first not there (an earlier attempt's records stay)."""
    base = Path(base)
    candidate, k = base, 1
    while candidate.exists():
        k += 1
        candidate = base.with_name(f'{base.name}_{k}')
    return candidate


class System:
    """What a queue does to and reads from the machine (tests replace it)."""

    def run(self, argv, log_path):
        with open(log_path, 'ab') as stream:
            return subprocess.run([str(a) for a in argv], stdout=stream, stderr=subprocess.STDOUT).returncode

    def uptime(self):
        return float(Path('/proc/uptime').read_text().split()[0])

    def dmesg(self):
        try:
            done = subprocess.run(['dmesg'], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout if done.returncode == 0 else None

    def journal(self, unit, since_epoch):
        try:
            done = subprocess.run(['journalctl', '--user', '-u', unit, '--since', f'@{int(since_epoch)}', '--no-pager', '-o', 'short-iso'], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout+done.stderr

    def gate(self, kind, pair_set='holdout'):
        return hc.wait_for(lambda: gate_once(kind, pair_set=pair_set), seconds=GATE_WAIT_SECONDS[kind])

    def disk(self, need_d, need_home):
        return hc.disk_guard(need_d_gib=need_d, need_home_gib=need_home)

    def sleep(self, seconds):
        time.sleep(seconds)

    def now(self):
        return time.time()


def unit_result(text):
    m = re.search(r'^Finished with result: (\S+)', text or '', re.MULTILINE)
    return m.group(1) if m else None


def run_unit(kind, pair, inner, system, *, batch, unit_dir, res, gpu, code, pair_set='holdout'):
    """One unit: the start string, systemd-run --wait, then its evidence (systemd's Result line, the supervisor's unit_end.json, the kernel's lines in the window, the journal) read with
    holdout_common.classify_unit_end.  Writes <unit_dir>/../unit_reading*.json (once) and units/<unit>.txt; returns the reading."""
    unit = unit_name(kind, pair, pair_set)
    units = Path(batch)/'units'
    units.mkdir(parents=True, exist_ok=True)
    log = Path(unit_dir).parent/f'{Path(unit_dir).name}.log'
    argv = unit_argv(kind, pair, inner, code=code, unit_dir=unit_dir, res=res, gpu=gpu, log=log, pair_set=pair_set)
    t0, since = system.uptime(), system.now()
    out_txt, k = units/f'{unit}.txt', 1
    while out_txt.exists():
        k += 1
        out_txt = units/f'{unit}_{k}.txt'
    rc = system.run(argv, out_txt)
    hc.note_unit_end(unit)                          # A5a review I1: this queue's own record of its last unit (holdout_common.machine_gate's local lower bound)
    journal = system.journal(unit, since)
    out_txt.with_suffix('.journal.txt').write_text(journal if journal is not None else '')
    text = out_txt.read_text(errors='replace') if out_txt.exists() else ''
    result_line = unit_result(text)
    kernel_text = system.dmesg()
    oom = hc.parse_oom(kernel_text, t0)
    dxg = hc.dxg_lines(kernel_text, t0)
    unit_end = read_json(Path(unit_dir)/'unit_end.json')
    if oom is None and unit_end is not None and unit_end.get('dmesg_readable'):
        oom = unit_end.get('oom_lines')
    if dxg is None and unit_end is not None and unit_end.get('dmesg_readable'):
        dxg = unit_end.get('dxg_lines')
    events = None if unit_end is None else unit_end.get('memory_events')
    cgroup = None if unit_end is None else unit_end.get('cgroup')
    step_kill = bool(unit_end is not None and unit_end.get('exit_code') == -9 and not unit_end.get('killed_by_supervisor'))
    status, why, evidence = hc.classify_unit_end(unit=unit, systemd_result=result_line, oom_lines=oom, events=events, journal=journal, step_kill=step_kill, unit_cgroup=cgroup)
    reading = dict(schema=SCHEMA.format('unit_reading'), kind=kind, pair=pair, unit=unit, argv=argv, systemd_run_exit=rc, systemd_result=result_line, status=status, status_kind=why, evidence=evidence,
                   dxg_lines=dxg, supervisor=None if unit_end is None else {k: unit_end.get(k) for k in ('utc', 'uptime', 'uptime_start', 'seconds', 'exit_code', 'killed_by_signal', 'killed_by_supervisor',
                                                                                                         'sigterm_received')},
                   child_exit=None if unit_end is None else unit_end.get('exit_code'), unit_dir=str(unit_dir), systemd_run_output=str(out_txt), log=str(log), utc=utc(),
                   rule='holdout_common.classify_unit_end (rule %s): OOM evidence -> infra_failure; a timeout without it -> unit_limit; unreadable evidence -> needs_lead_rejudgment' % hc.LIMIT_RULE)
    write_once(Path(unit_dir).parent/f'{Path(unit_dir).name}_reading.json', reading)     # unit_dir is an attempt's own directory (attempt_dir): one reading per attempt
    return reading


# ---- batches -------------------------------------------------------------------------------------------------------------------------------------
def new_batch_dir(kind, pair_set='holdout'):
    base = (lpba().batch_root(pair_set) if is_lpba(pair_set) else Path(hc.batch_root()))/f'hb_{kind}_{stamp()}'
    candidate = attempt_dir(base)
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


class Log:
    def __init__(self, batch, echo=print):
        self.path, self.echo = Path(batch)/'queue.log', echo

    def __call__(self, message):
        line = f'{utc()} {message}'
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(line+'\n')
        if self.echo is not None:
            self.echo(line)


def queue_launch(kind, args, *, repo=None, pair_set='holdout'):
    """The lead's start string of a queue's own unit (from the detached checkout of C22.H0-4): CPU only, 2G, nice 5.  LP-B: an LPBA mode's queue unit (queue_unit)."""
    repo = Path(REPO if repo is None else repo)
    script = {'method': 'experiments/hb/hb_method.py', 'review': 'experiments/hb/hb_review.py', 'configc': 'experiments/hb/hb_configc.py'}[kind]
    argv = ['systemd-run', '--user', f'--unit={queue_unit(kind, pair_set)}', '--collect', f'--working-directory={repo}', '--property=MemoryMax=2G', '--nice=5', '--setenv=PYTHONDONTWRITEBYTECODE=1',
            '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1', '--setenv=CUDA_VISIBLE_DEVICES=', f'--setenv=PYTHONPATH={he.core.FROZEN}:{repo}', str(PY), '-B', script, *args]
    if 'pytest' in ' '.join(argv):
        raise ValueError(f'start string with a forbidden word: {argv}')
    return argv


def rows():
    """The hold-out pairs in the pair list's order (holdout_common.holdout_pairs: its sha256 checked; ids only)."""
    return [hc.pair_name(f, m) for f, m in hc.holdout_pairs()]


def rows_of(pair_set='holdout'):
    """LP-B: the pairs of a pair set -- the hold-out's rows() (looked up at call time, as before), an LPBA mode's table (lpba_arms.rows)."""
    return lpba().rows(pair_set) if is_lpba(pair_set) else rows()


# ---- HI's hold-out predictions: the first run and its resume taken together, in the one root (A5b item 13; runs/20261110-a5/实现说明_A5a.md 6.2) ------------------------------------------
HI_PRED_FILES = ('probs36.npz', 'structure_groups.npz', 'provenance.json')     # hb_store.PRED_FILES (a test compares)
HI_RECORD_OUTPUTS = ('probs36.npz', 'structure_groups.npz')                     # the record's output_sha256 (hi_core.infer_holdout); provenance.json by provenance_sha256 when the record has one


def hi_union(subjects, *, hashed=None, pred_root=None, lock=None):
    """(record, problems): HI's hold-out inference read as the union of its first run and its resume in the one root (A5a review I2; the A5a notes' table, rows HE and hb_store).  pred_root:
    he_core.HOLDOUT_SOURCES['pred_root'] (read at call time), its parent the root; lock: hi_core.HOLDOUT_LOCK (read at call time).
      - the lock is there and the root is one of its runs' (hi_core.run_outs);
      - the runs' records together are every subject of `subjects` (hi_core.complete_subjects), and hi_core.run_consistency_problems has none;
      - no residue in pred/ (hi_core.residue_subjects: a prediction without its record, a killed run's that no resume moved out);
      - each subject of `hashed` (default: all of `subjects`) has its record <root>/subjects/<s>.json naming it, whose output_sha256 of probs36.npz and structure_groups.npz are those files' sha256
        now and whose provenance_sha256 (when it has one) is provenance.json's (xb_core.check_hi_record's rule over the three files); a subject without a record has nothing of it hashed.
    Nothing under attempt<k>/ (the residue a resume moved out) is read.  record: lock, root, runs, missing, residue, records {subject: {record, record_sha256, run, sha256: {file: sha256 or None}}}."""
    hi = hc._hi()
    lock = Path(hi.HOLDOUT_LOCK if lock is None else lock)
    root = Path(he.HOLDOUT_SOURCES['pred_root'] if pred_root is None else pred_root).parent
    subjects = list(subjects)
    hashed = subjects if hashed is None else list(hashed)
    record = dict(rule='A5b item 13: HI\'s first run and its resume taken together in the one root (A5a review I2): complete, consistent, no residue in pred/, each record against its files',
                  lock=str(lock), root=str(root), runs=[], missing=None, residue=None, records={})
    if not lock.is_file():
        return record, [f'{lock}: HI\'s hold-out inference has not run (its lock is not there)']
    problems = []
    record['runs'] = [str(o) for o in hi.run_outs(lock)]
    if root.resolve() not in {Path(o).resolve() for o in record['runs']}:
        problems.append(f'{root} is not a run root of {lock} ({record["runs"]})')
    _, record['missing'] = hi.complete_subjects(lock, subjects)
    if record['missing']:
        problems.append(f'HI\'s runs taken together have not written {len(record["missing"])} subjects: {record["missing"][:5]}')
    problems += [f'HI\'s runs: {x}' for x in hi.run_consistency_problems(lock)]
    record['residue'] = hi.residue_subjects(root)
    if record['residue']:
        problems.append(f'predictions in {root/"pred"} without a record (a killed run\'s residue, not moved out by a resume): {record["residue"][:5]}')
    for s in hashed:
        path = root/'subjects'/f'{s}.json'
        row = dict(record=str(path), record_sha256=None, run=None, sha256={name: None for name in HI_PRED_FILES})
        record['records'][s] = row
        doc = read_json(path) if path.is_file() else None
        if not isinstance(doc, dict) or doc.get('subject') != s:
            problems.append(f'{path}: HI\'s record of {s} is not there or does not name it')
            continue
        row.update(record_sha256=sha(path), run=doc.get('run'))
        pred = root/'pred'/s
        row['sha256'] = {name: sha(pred/name) if (pred/name).is_file() else None for name in HI_PRED_FILES}
        outputs = doc.get('output_sha256') if isinstance(doc.get('output_sha256'), dict) else {}
        bad = [name for name in HI_RECORD_OUTPUTS if row['sha256'][name] is None or row['sha256'][name] != outputs.get(name)]
        if row['sha256']['provenance.json'] is None or (doc.get('provenance_sha256') is not None and doc['provenance_sha256'] != row['sha256']['provenance.json']):
            bad.append('provenance.json')
        if bad:
            problems.append(f'{pred}: {bad} missing or not the sha256 of the record {path}')
    return record, problems


# ---- LP-B: HI's LPBA40 output (experiments/hi/hi_lpba.py, LP-A) -------------------------------------------------------------------------------------
LPBA_HI_LOCK = 'lpba_hi.lock'                   # hi_lpba.LOCK_NAME (a test compares)


def lpba_hi(pair_set, subjects, *, hashed=None):
    """(record, problems): HI's LPBA40 inference of a mode as hi_lpba.py leaves it (one run, no resume): its lock <root>/locks/lpba_hi.lock; <hi_out>/hi_record.json passed, of this pair set, naming
    every subject of `subjects`; no hi_stopped.json; each subject of `hashed` (default all) with its record subjects/<s>.json naming it, whose output_sha256 of probs36.npz and structure_groups.npz
    are those files' sha256 now and whose provenance_sha256 is provenance.json's.  Reads the records and hashes the files; decodes nothing."""
    la = lpba()
    r = la.lc.roots(pair_set)
    root, lock = Path(r.hi_out), Path(r.locks)/LPBA_HI_LOCK
    subjects = list(subjects)
    hashed = subjects if hashed is None else list(hashed)
    record = dict(rule='LP-B: HI\'s LPBA40 run (hi_lpba.py) complete over the subjects, each record against its files', lock=str(lock), root=str(root), records={})
    problems = []
    if not lock.is_file():
        return record, [f'{lock}: HI\'s {pair_set} inference has not run (its lock is not there)']
    doc = read_json(root/'hi_record.json')
    if not isinstance(doc, dict) or doc.get('passed') is not True or doc.get('pair_set') != pair_set:
        problems.append(f"{root/'hi_record.json'}: not there, not passed, or not of {pair_set}")
    elif not set(subjects) <= set(doc.get('rows') or {}):
        problems.append(f"{root/'hi_record.json'}: its rows do not name every subject {sorted(set(subjects)-set(doc.get('rows') or {}))[:5]}")
    if (root/'hi_stopped.json').exists():
        problems.append(f"{root/'hi_stopped.json'} is there: HI's run stopped")
    for s in hashed:
        path = root/'subjects'/f'{s}.json'
        row = dict(record=str(path), record_sha256=None, sha256={name: None for name in HI_PRED_FILES})
        record['records'][s] = row
        rec = read_json(path) if path.is_file() else None
        if not isinstance(rec, dict) or rec.get('subject') != s:
            problems.append(f'{path}: HI\'s record of {s} is not there or does not name it')
            continue
        row['record_sha256'] = sha(path)
        pred = Path(r.pred_root)/s
        row['sha256'] = {name: sha(pred/name) if (pred/name).is_file() else None for name in HI_PRED_FILES}
        outputs = rec.get('output_sha256') if isinstance(rec.get('output_sha256'), dict) else {}
        bad = [name for name in HI_RECORD_OUTPUTS if row['sha256'][name] is None or row['sha256'][name] != outputs.get(name)]
        if row['sha256']['provenance.json'] is None or rec.get('provenance_sha256') != row['sha256']['provenance.json']:
            bad.append('provenance.json')
        if bad:
            problems.append(f'{pred}: {bad} missing or not the sha256 of the record {path}')
    return record, problems


__all__ = ['ARM', 'CONFIGC_ARM', 'CONFIGC_UNIT', 'EXIT', 'GATE_MEM_GIB', 'GATE_WAIT_SECONDS', 'INPUT_FILES', 'KINDS', 'Log', 'OUTPUT_FILES', 'QUEUE_UNIT', 'SCHEMA', 'Stop', 'System', 'TERMINAL_STATUSES',
           'TOOL', 'UNIT_NEED_GIB', 'UNIT_PREFIX', 'HI_PRED_FILES', 'HI_RECORD_OUTPUTS', 'attempt_dir', 'bz_decision_auth', 'cell', 'code_record', 'decision_reason_problem',
           'file_rows', 'gate_once', 'hi_union', 'lock_path', 'new_batch_dir',
           'queue_launch', 'read_json', 'record_dir', 'review_root',
           'rows', 'run_unit', 'sha', 'terminal_path', 'unit_argv', 'unit_name', 'unit_result', 'unseal_gate', 'utc', 'write_once', 'write_terminal',
           'LPBA_HI_LOCK', 'LPBA_UNIT_KIND', 'gate_of', 'hold_path', 'is_lpba', 'lpba', 'lpba_authorizations', 'lpba_hi', 'lpba_resume_problem', 'queue_unit', 'rows_of']
