"""XB chain (prereg §2.3 (d); runs/20261107-xb/任务书.md item 1 (d)): the external baseline arms on the hold-out (or development) pairs, one driver unit at a time, by arm (prereg §5.2 P15), each arm's
pairs in the pair list's order.  Not BL2VAL's run_bl2val.sh: no path of seq_vp_3d or of BL2's D: root, units named tag-hb-x-<arm>-<fixed>-<moving> (never seq-vp-*), both disks guarded (prereg §5.3), no
speed canary (P21), no evaluation in the chain (the evaluation entry, after every arm is written, is bayesreg.eval.holdout's; C23.12).

    python -B experiments/xb/xb_chain.py plan   --pair-set {holdout,dev} --arms A,B,... [--pairs P,...]          the cells, their state and each unit's start string; launches nothing
    python -B experiments/xb/xb_chain.py queue  --pair-set {holdout,dev} --arms A,B,... [--pairs P,...] [--resume ARM=PAIR=REL ...]
    python -B experiments/xb/xb_chain.py launch-string --pair-set ... --arms ...                                   the lead's start string of the chain's own unit (tag-hb-x-chain)

Each cell: its state (cell_state: a complete terminal record -> skipped; a lock or a record that is not complete -> STOP unless a committed resume authorization for it is given, or final with a
committed no_resume authorization (I3); refusal records of it and no lock -> STOP unless a committed rerun authorization covers them (I1)); a hold file (<record root>/hold) waits; the machine
gate (no other tag-* / seq-vp-* unit, GPU idle, MemAvailable, load; up to 15 minutes, else STOP) and the disk guard; the unit (UNIT_PROPERTIES: MemoryMax 10G, MemorySwapMax 0, OOMPolicy continue,
RuntimeMaxSec 1800); afterwards dmesg (dxg create_allocation failures and OOM kills in the window), the unit's journal, systemd-run's result line (kept in units/<unit>.txt), the supervisor's end probe and the driver's
terminal record.  Every unit gets unit_reading.json (holdout_common.classify_unit_end, rule (乙) by default: OOM evidence of the unit -> infra_failure; a timeout without it -> unit_limit; unreadable
evidence -> needs_lead_rejudgment).  A unit that left no terminal record after its lock gets one written here with that reading; a reading of infra_failure or needs_lead_rejudgment that contradicts
the record the driver wrote keeps that record and writes overridden.json beside it (I2: xb_core.effective_record reads it first; the batch stops; the cell needs a resume or a no_resume
authorization; a resumed attempt so overridden is final with no output (infrastructure), never ok).  ConvexAdam-v4seg's cells get --method-run, the pinned path of the method's T88 run.json
(xb_core.method_t88_run; a hold-out pair: the method arm's cell on D: after step 5 and the move, known by HB's finalized and move records and checked by the finalized sha256 (HB review m13); before that the driver refuses, I4).  Every unit gets XB_BATCH_DIR, where the driver's refusal records go.
What continues: ok, method_failure, unit_limit; wrapper_error only under WRAPPER_ERROR_POLICY identity.  What stops the batch (prereg §5.4; the evidence stays, nothing is rerun): infra_failure, a
wrapper_error under the stop policy, needs_lead_rejudgment, a refusal of the driver before its lock (its refusal record in <batch>/refusals/, copied into the cell's record directory: A5a), any dxg
line, a unit ended by a signal.  A hold-out batch stops at its start when the code tree is not clean, the code is not the pinned commit's (rule one, holdout_common.pinned_code_problems: P1 from
runs/holdout_B_v1/pinned_code.json; A5a) or the preregistration is not pinned (I5).
Written without an output, the batch going past it (A5a; prereg revision 2 items 17 (f), 18): a needs_lead_rejudgment record judged unit_limit or no_output by a committed rejudgment record
(holdout_common.rejudgment; infrastructure only classifies); a cell refused before its lock (again after a rerun) whose newest refusal record BZ's committed no_resume names (P20).  batch.json records each arm's frozen and environment checks (m14).  P19: an arm whose
arm_stopped.json appears runs no further pair.  Exit: 0 every cell done or skipped; 20 infrastructure; 22 refused; 23 wrapper error; 24 a cell needs a resume record; 25 the lead's rejudgment.

LP-B (contract C26.5-6, C26.8; LPBA40 preregistration 06121278... §4.3, §8.2 steps 6, 9, §8.7; brief item 3): --pair-set lpba40_test | lpba40_feas runs xb_core.LPBA_ARMS only (another arm refused at
the start) on the mode's table (lpba_arms.rows), its root (records, work, locks, batches; no store), its units tag-lp-x-<arm>-<Sff>-<Smm> and the chain's tag-lp-x-chain (tag-lp-feas-...), the hold file
lpba40_test_v1/hold.  The start: the code (lpba_arms.code_record: formal, rule one -- LP-C2's, C26.9-4, fail closed; feasibility, a clean checkout) and the pins (lpba_common.check_pins) in place of
the hold-out's rule one and preregistration.  P19 at the second suspicious pair (xb_core.p19_stop_at).  A rerun, resume or no_resume record of an LPBA mode is read only when holdout_common knows its
authorizations (xb_core.authorizations_problem; C26.8-3, D148: LP-C2's; the feasibility mode has none).
"""
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import xb_core as xc  # noqa: E402
import xb_drive  # noqa: E402

hc = xc.hc
STOP = dict(infra=20, refused=22, wrapper=23, resume=24, rejudge=25)


class Stop(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class System:
    """What the chain does to and reads from the machine (tests replace it)."""

    def run(self, argv, log_path):
        with open(log_path, 'ab') as stream:
            return subprocess.run([str(a) for a in argv], stdout=stream, stderr=subprocess.STDOUT).returncode

    def uptime(self):
        return float(Path('/proc/uptime').read_text().split()[0])

    def dmesg(self):
        done = subprocess.run(['dmesg'], capture_output=True, text=True)
        return done.stdout if done.returncode == 0 else None

    def journal(self, unit, since_epoch):
        done = subprocess.run(['journalctl', '--user', '-u', unit, '--since', f'@{int(since_epoch)}', '--no-pager', '-o', 'short-iso'], capture_output=True, text=True)
        return done.stdout+done.stderr

    def gate(self, pair_set=None):
        """The machine gate with the chain's own unit excluded (LP-B: an LPBA mode's chain unit, xb_core.chain_unit)."""
        return hc.wait_for(lambda: hc.machine_gate(own_units=(xc.CHAIN_UNIT if pair_set is None else xc.chain_unit(pair_set),)))

    def disk(self, need_d, need_home):
        return hc.disk_guard(need_d_gib=need_d, need_home_gib=need_home)

    def pinned_code(self):
        """Rule one (contract C24.3; A5a): holdout_common.pinned_code_problems -- P1 from the committed runs/holdout_B_v1/pinned_code.json, never HEAD; (record, problems)."""
        return hc.pinned_code_problems()

    def sleep(self, seconds):
        time.sleep(seconds)

    def now(self):
        return time.time()


def unit_prefix(unit, cwd, properties=None, extra_env=None):
    """systemd-run with the unit's settings (prereg §2.3 (g): MemoryMax 10G, MemorySwapMax 0, OOMPolicy continue, RuntimeMaxSec 1800) and the environment: the part of a cell's start string that the
    controlled limit experiment keeps (xb_limit_experiment.py).  `extra_env`: XB_BATCH_DIR for a cell."""
    env = dict(XB_UNIT=unit, XB_UNIT_PROPERTIES=' '.join(xc.UNIT_PROPERTIES), OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, PYTHONDONTWRITEBYTECODE=1,
               **(extra_env or {}))
    properties = xc.UNIT_PROPERTIES if properties is None else tuple(properties)
    env['XB_UNIT_PROPERTIES'] = ' '.join(properties)
    return ['systemd-run', '--user', '--wait', '--collect', f'--unit={unit}', *[f'--property={p}' for p in properties], f'--working-directory={cwd}',
            *[f'--setenv={k}={v}' for k, v in env.items()]]


def supervisor(unit, record_dir, inner, *, code=xc.REPO):
    """The unit's main process: xb_unit.py (writes unit_start.json, runs `inner`, writes unit_end.json with memory.events and the kernel's oom lines)."""
    return [str(xc.PY), '-I', '-B', str(Path(code)/'experiments/xb/xb_unit.py'), '--record-dir', str(record_dir), '--unit', unit, '--', *[str(a) for a in inner]]


def unit_record_dir(arm, pair_set, pair, resume=False):
    """The base directory of the supervisor's records of an attempt: <record>/unit (a resume: unit_resume)."""
    return xc.cell_paths(arm, pair_set, pair)['record']/('unit_resume' if resume else 'unit')


def unit_dirs(arm, pair_set, pair, resume=False):
    """Every attempt's supervisor directory of the cell: <base>, <base>_2, <base>_3, ... (a cell refused before its lock and run again under a rerun authorization has more than one; I1)."""
    base = unit_record_dir(arm, pair_set, pair, resume)
    out = [base] if base.is_dir() else []
    k = 2
    while base.with_name(f'{base.name}_{k}').is_dir():
        out.append(base.with_name(f'{base.name}_{k}'))
        k += 1
    return out


def attempt_unit_dir(arm, pair_set, pair, resume=False):
    """The next attempt's supervisor directory: the first of <base>, <base>_2, ... not there (the supervisor writes unit_start.json once; an earlier attempt's records stay)."""
    base = unit_record_dir(arm, pair_set, pair, resume)
    candidate, k = base, 1
    while candidate.exists():
        k += 1
        candidate = base.with_name(f'{base.name}_{k}')
    return candidate


def unit_argv(arm, pair_set, pair, *, code=xc.REPO, resume_rel=None, cwd=None, batch=None, unit_dir=None, pinned=None):
    """A cell's start string: the unit (unit_prefix, XB_BATCH_DIR = the batch), the supervisor, the driver; ConvexAdam-v4seg with --method-run, the pinned path of the method's T88 run.json on this pair
    when the pair set has one (I4; without it the driver refuses the cell before its lock)."""
    unit = xc.unit_name(arm, pair, pair_set)
    cwd = xc.roots(pair_set)[0] if cwd is None else cwd
    method = xc.method_t88_run(pair_set, pair) if xc.ARMS[arm]['kind'] == 'v4seg' else None
    inner = [str(xc.PY), '-I', '-B', str(Path(code)/'experiments/xb/xb_drive.py'), 'run', '--arm', arm, '--pair-set', pair_set, '--pair', pair, *(['--resume-record', resume_rel] if resume_rel else []),
             *(['--method-run', str(method)] if method is not None else []), *(['--pinned', pinned] if pinned is not None and xc.is_lpba(pair_set) else [])]     # LP-B review minor 5
    unit_dir = attempt_unit_dir(arm, pair_set, pair, bool(resume_rel)) if unit_dir is None else unit_dir
    argv = unit_prefix(unit, cwd, extra_env=None if batch is None else dict(XB_BATCH_DIR=batch))+supervisor(unit, unit_dir, inner, code=code)
    if 'pytest' in ' '.join(argv):
        raise ValueError(f'start string with a forbidden word: {argv}')        # run_mc0.check_launch
    return argv


def chain_launch_string(pair_set, arms, *, code=xc.REPO):
    return ['systemd-run', '--user', f'--unit={xc.chain_unit(pair_set)}', '--collect', f'--working-directory={code}', '--property=MemoryMax=2G', '--nice=5', '--setenv=PYTHONDONTWRITEBYTECODE=1',
            '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1', str(xc.PY), '-B', str(Path(code)/'experiments/xb/xb_chain.py'), 'queue', '--pair-set', pair_set, '--arms', ','.join(arms)]


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def unit_flagged(arm, pair_set, pair, resume=False):
    """The chain's unit reading of a cell says infra_failure or needs_lead_rejudgment (the unit's OOM evidence or unreadable evidence; run_one writes overridden.json when it contradicts the record)."""
    for directory in unit_dirs(arm, pair_set, pair, resume):
        reading = read_json(directory/'unit_reading.json')
        if reading is not None and reading.get('locked', True) and reading.get('status') in ('infra_failure', 'needs_lead_rejudgment'):
            return True                             # the reading of the attempt that took the lock (an attempt refused before it read nothing of the pair)
    return False


def effective_path(paths):
    """The path of the first attempt's effective terminal record: overridden.json when there (I2), else result.json (which may not exist)."""
    return paths['overridden'] if paths['overridden'].is_file() else paths['result']


def no_resume(arm, pair_set, pair, paths=None):
    """I3: the committed no_resume authorization (未批补跑) of the cell's first attempt, or None (holdout_common.no_resume_authorization on the effective record)."""
    paths = xc.cell_paths(arm, pair_set, pair) if paths is None else paths
    if xc.authorizations_problem(pair_set) is not None:
        return None                                 # LP-B: no authorizations of this mode known
    return hc.no_resume_authorization(tool=xb_drive.TOOL, arm=arm, pair=pair, pair_set=pair_set, record=effective_path(paths), complete=lambda: xc.attempt_complete(paths))


REJUDGED_WRITTEN = ('unit_limit', 'no_output')     # prereg revision 2 item 18 (decision_log 2026-11-10 NI2, N1, N2): these judgments write the cell; infrastructure only classifies


def rejudged(arm, pair_set, pair, paths=None):
    """A5a (prereg revision 2 item 18; decision_log 2026-11-10 ~2:28 AM CST NI2, NI3, ~2:45 AM CST N1, N2): the committed rejudgment record of the cell's first attempt (holdout_common.rejudgment,
    the one reader: runs/holdout_B_v1/rejudgments/<arm>/<pair>_<k>.json, the highest k, bound to the effective record by its sha256 now).  (found or None, problems): found only when its judgment is
    unit_limit (the lead's) or no_output (BZ's words): the cell is written (no output), the batch goes past it; infrastructure only classifies -- a rerun still needs a committed resume record, and a
    resumed attempt is final whatever it reads (n5: the chain does not wait for a rejudgment after the resume)."""
    paths = xc.cell_paths(arm, pair_set, pair) if paths is None else paths
    found, problems = hc.rejudgment(arm, pair, record=effective_path(paths), pair_set=pair_set)
    if found is not None and found['document']['judgment'] not in REJUDGED_WRITTEN:
        return None, [f"{found['path']}: judgment {found['document']['judgment']!r} only classifies (a rerun needs a committed resume record); only {list(REJUDGED_WRITTEN)} write the cell"]
    return found, problems


def rejudged_infrastructure(arm, pair_set, pair, paths=None):
    """A5a review m5: the cell's committed rejudgment record holds and judges infrastructure (it only classifies: the cell then needs a committed resume record, STOP resume 24, not rejudge 25).
    The record (path, sha256, k) or None."""
    paths = xc.cell_paths(arm, pair_set, pair) if paths is None else paths
    found, _ = hc.rejudgment(arm, pair, record=effective_path(paths), pair_set=pair_set)
    return None if found is None or found['document']['judgment'] != 'infrastructure' else dict(path=found['path'], sha256=found['sha256'], k=found['k'])


def final_source(arm, pair_set, pair):
    """A5a review m5: what made a cell final, for batch.json (prereg revision 2 step 10's list): the resumed attempt's record, BZ's no_resume of the first attempt, the rejudgment record
    (unit_limit, no_output) or BZ's P20 no_resume on the newest refusal record's copy -- dict(by, path, sha256), or None (not final)."""
    paths = xc.cell_paths(arm, pair_set, pair)
    state, _ = cell_state(arm, pair_set, pair)
    if state != 'final':
        return None
    second = paths['resume_overridden'] if paths['resume_overridden'].is_file() else paths['resume_result']
    if second.is_file():
        return dict(by='resumed attempt (prereg §5.4)', path=str(second), sha256=xc.sha256(second))
    if xc.effective_record(paths) is not None:
        auth = no_resume(arm, pair_set, pair, paths)
        if auth is not None:
            return dict(by='no_resume (I3)', path=auth.path, sha256=auth.sha256, record=str(effective_path(paths)), record_sha256=xc.sha256(effective_path(paths)))
        found, _ = rejudged(arm, pair_set, pair, paths)
        if found is not None:
            return dict(by=f"rejudgment {found['document']['judgment']} (prereg revision 2 item 18)", path=found['path'], sha256=found['sha256'], record=str(effective_path(paths)),
                        record_sha256=xc.sha256(effective_path(paths)))
        return None
    declined = declined_refusal(arm, pair_set, pair, paths)
    if declined is None:
        return None
    return dict(by='no_resume P20 (prereg revision 2 item 17 (f))', path=declined.path, sha256=declined.sha256, record=declined.document.get('record'), record_sha256=declined.document.get('record_sha256'))


def declined_refusal(arm, pair_set, pair, paths=None):
    """A5a (prereg revision 2 item 17 (f); decision_log 2026-11-10 ~2:52, ~2:53 AM CST): a cell refused before its lock that BZ does not let run again (P20): holdout_common.declined_refusal on the
    copies of its refusal records in its record directory (tool xb).  The authorization or None; Refused when the copies' sequence numbers do not tell the newest."""
    paths = xc.cell_paths(arm, pair_set, pair) if paths is None else paths
    if xc.authorizations_problem(pair_set) is not None:
        return None                                 # LP-B: no authorizations of this mode known
    return hc.declined_refusal(paths['record'], tool=xb_drive.TOOL, arm=arm, pair=pair, pair_set=pair_set)


def cell_state(arm, pair_set, pair):
    """(state, the effective terminal record): done (a complete record, not overridden, no flagged unit reading); final (a resumed attempt that did not complete, a committed no_resume
    authorization, a committed rejudgment unit_limit or no_output of a needs_lead_rejudgment record (prereg revision 2 item 18), or a cell refused before its lock whose newest refusal record BZ's
    committed no_resume names (P20, item 17 (f)): no further run, prereg §5.4); needs_resume (a lock or a record that is not complete, the overridden ones included: I2); refused (refusal records of
    the cell before any lock: a rerun needs a committed rerun authorization, I1); not_run; arm_stopped.  The effective record is overridden.json when there is one (xb_core.effective_record)."""
    paths = xc.cell_paths(arm, pair_set, pair)
    if paths['arm_stop'].exists():
        return 'arm_stopped', None
    second = xc.effective_record(paths, resume=True)
    if second is not None:
        return ('done' if xc.complete(second) and not unit_flagged(arm, pair_set, pair, True) else 'final'), second
    first = xc.effective_record(paths)
    if first is not None:
        if xc.complete(first) and not unit_flagged(arm, pair_set, pair):
            return 'done', first
        if no_resume(arm, pair_set, pair, paths) is not None:
            return 'final', first
        if first.get('status') == 'needs_lead_rejudgment' and rejudged(arm, pair_set, pair, paths)[0] is not None:
            return 'final', first
        return 'needs_resume', first
    if paths['lock'].exists():
        return 'needs_resume', None
    if hc.cell_refusals(xc.roots(pair_set)[0], arm=arm, pair=pair, pair_set=pair_set):
        try:
            declined = declined_refusal(arm, pair_set, pair, paths)
        except hc.Refused:
            declined = None                         # the copies do not tell the newest: the queue says so (refused)
        return ('final' if declined is not None else 'refused'), None
    return 'not_run', None


def cells(pair_set, arms, pairs=None):
    if xc.is_lpba(pair_set):                        # LP-B: the mode's table; only the four LPBA40 arms
        other = [a for a in arms if a not in xc.LPBA_ARMS]
        if other:
            raise Stop(STOP['refused'], f'{other}: not LPBA40 arms {xc.LPBA_ARMS} (LPBA40 prereg §4.3, §4.5)')
        rows = xc.lpba().rows(pair_set)
    elif pair_set == 'holdout':
        rows = [hc.pair_name(f, m) for f, m in hc.holdout_pairs()]
    else:
        rows = list(hc.DEV_PAIRS)
    if pairs:
        unknown = [p for p in pairs if p not in rows]
        if unknown:
            raise Stop(STOP['refused'], f'pairs not in the plan of {pair_set}: {unknown}')
        rows = [p for p in rows if p in pairs]
    for arm in arms:
        if arm not in xc.ARMS:
            raise Stop(STOP['refused'], f'unknown arm {arm}')
    return [(arm, pair) for arm in arms for pair in rows]


def unit_result(text):
    m = re.search(r'^Finished with result: (\S+)', text or '', re.MULTILINE)
    return m.group(1) if m else None


def read_lock(path):
    path = Path(path)
    if not path.is_file():
        return None
    lines = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    return dict(path=str(path), line=lines[0] if lines else None, sha256=xc.sha256(path))


def unit_reading(*, unit, result_line, unit_end, oom_lines, journal, rule=None):
    """holdout_common.classify_unit_end on one unit's evidence (prereg revision 1c item 1): systemd's Result (None when systemd-run printed none), the supervisor's end probe (memory.events at the
    end, the unit's cgroup path; None without unit_end.json), the kernel's oom-kill lines in the window (the chain's dmesg; the supervisor's when the chain's could not be read), the journal, and the
    supervisor's child exit -9 (unless the supervisor sent it itself after a SIGTERM)."""
    events = None if unit_end is None else unit_end.get('memory_events')
    cgroup = None if unit_end is None else unit_end.get('cgroup')
    step_kill = bool(unit_end is not None and unit_end.get('exit_code') == -9 and not unit_end.get('killed_by_supervisor'))
    if oom_lines is None and unit_end is not None and unit_end.get('dmesg_readable'):
        oom_lines = unit_end.get('oom_lines')
    return hc.classify_unit_end(unit=unit, systemd_result=result_line, oom_lines=oom_lines, events=events, journal=journal, step_kill=step_kill, unit_cgroup=cgroup, rule=rule)


RULE_TEXT = ('prereg revision 1c item 1 (holdout_common.classify_unit_end, rule {rule}; yi is the default): the driver\'s memory guard or a CONSTRAINT_NONE line in the window -> infra_failure; '
             'OOM evidence of the unit (memory.events oom / oom_kill, its dmesg oom-kill lines, the journal, systemd Result oom-kill, an exit -9 the supervisor did not send) -> under yi infra_failure, '
             'under jia unit_limit when it is the unit\'s own memory limit and infra_failure otherwise; unreadable evidence -> needs_lead_rejudgment; a timeout without OOM evidence -> unit_limit')


def override_record(arm, pair_set, pair, doc, *, status, kind, evidence, unit, result_line, unit_dir, resume_rel=None):
    """I2: the unit's evidence contradicts the terminal record the driver wrote: that record stays as it is; overridden.json beside it (written once) is the attempt's terminal record from now on
    (xb_core.effective_record), with the reading's status, no disp (the field is not an output: prereg §5.4), and for a resumed attempt no_output_class infrastructure (final, never ok)."""
    paths = xc.cell_paths(arm, pair_set, pair)
    record_path = paths['resume_result'] if resume_rel else paths['result']
    target = paths['resume_overridden'] if resume_rel else paths['overridden']
    document = dict(schema=xc.SCHEMA.format('result'), status=status, arm=arm, pair=pair, pair_set=pair_set, fixed_subject=doc.get('fixed_subject'), moving_subject=doc.get('moving_subject'),
                    failure=dict(kind=kind, reason=f'the unit\'s evidence reads {status} ({kind}) although the driver wrote {doc.get("status")}', evidence=evidence),
                    overrides=dict(path=str(record_path), sha256=xc.sha256(record_path), status=doc.get('status'), disp=doc.get('disp')), disp=None, export_check=None, lock=doc.get('lock'),
                    resume=doc.get('resume'), no_output_class='infrastructure' if resume_rel else None, wrapper_error_policy=None,
                    unit=dict(name=unit, systemd_result=result_line, reading=str(unit_dir/'unit_reading.json')), utc=hc.utc(),
                    written_by='xb_chain.py (the unit\'s evidence contradicts the driver\'s record)', rule=RULE_TEXT.format(rule=hc.LIMIT_RULE)+'; I2: the driver\'s record stays, this one supersedes it')
    hc.write_json_once(target, document)
    return target, document


def write_unit_record(arm, pair_set, pair, *, status, kind, reason, unit, unit_rc, result_line, journal, kernel, evidence, resume_rel=None, unit_dir=None):
    """The terminal record of a cell whose driver was stopped after its lock and wrote none (RuntimeMaxSec, an OOM kill of the driver, a signal, a restart)."""
    paths = xc.cell_paths(arm, pair_set, pair)
    lock = read_lock(hc.resume_lock_path(paths['lock']) if resume_rel else paths['lock'])
    doc = dict(schema=xc.SCHEMA.format('result'), status=status, arm=arm, pair=pair, pair_set=pair_set, failure=dict(kind=kind, reason=reason, evidence=evidence), disp=None, export_check=None, lock=lock,
               resume=None if not resume_rel else dict(path=resume_rel), no_output_class=({'infra_failure': 'infrastructure', 'wrapper_error': 'wrapper'}.get(status)) if resume_rel else None,
               unit=dict(name=unit, exit=unit_rc, systemd_result=result_line, journal=journal, properties=list(xc.UNIT_PROPERTIES), unit_end=str((unit_record_dir(arm, pair_set, pair, bool(resume_rel)) if unit_dir is None else unit_dir)/'unit_end.json')),
               kernel=kernel, utc=hc.utc(), wrapper_error_policy=xc.WRAPPER_ERROR_POLICY if status == 'wrapper_error' else None, written_by='xb_chain.py (the driver left no terminal record after its lock)',
               rule=RULE_TEXT.format(rule=hc.LIMIT_RULE))
    hc.write_json_once(paths['resume_result'] if resume_rel else paths['result'], doc)
    return doc


def paths_of(arm, pair_set, pair):
    return xc.cell_paths(arm, pair_set, pair)


def overrides(driver_status, unit_status):
    """I2: whether the unit's reading supersedes the driver's record: OOM evidence (infra_failure) over any other status; unreadable evidence (needs_lead_rejudgment) over a status that is not already
    one of the two (an infrastructure failure the driver saw stays one)."""
    if unit_status == 'infra_failure':
        return driver_status != 'infra_failure'
    if unit_status == 'needs_lead_rejudgment':
        return driver_status not in ('infra_failure', 'needs_lead_rejudgment')
    return False


def run_one(arm, pair_set, pair, system, *, batch, log, resume_rel=None, pinned=None):
    """One cell's unit and what it left.  Returns the terminal record's status; Stop when the batch must stop."""
    unit = xc.unit_name(arm, pair, pair_set)
    units = Path(batch)/'units'
    units.mkdir(parents=True, exist_ok=True)
    gate = system.gate(pair_set) if xc.is_lpba(pair_set) else system.gate()
    need_d, need_home = xb_drive.need_gib(arm, pair_set)
    disk = system.disk(need_d, need_home)
    t0, since = system.uptime(), system.now()
    Path(xc.roots(pair_set)[0]).mkdir(parents=True, exist_ok=True)
    log(f'run {unit} start: gate {gate.get("passed")}, disk {disk.get("free_gib")}')
    unit_dir = attempt_unit_dir(arm, pair_set, pair, bool(resume_rel))
    lock_path = Path(hc.resume_lock_path(paths_of(arm, pair_set, pair)['lock'])) if resume_rel else paths_of(arm, pair_set, pair)['lock']
    argv = unit_argv(arm, pair_set, pair, resume_rel=resume_rel, batch=batch, unit_dir=unit_dir, pinned=pinned)
    out_txt = units/f'{unit}.txt'
    rc = system.run(argv, out_txt)
    hc.note_unit_end(unit)                          # A5a review I1: this queue's own record of its last unit (the gate's local lower bound)
    journal_path = units/f'{unit}.journal.txt'
    journal = system.journal(unit, since)
    journal_path.write_text(journal if journal is not None else '')
    text = out_txt.read_text(errors='replace') if out_txt.exists() else ''
    result_line = unit_result(text)
    kernel_text = system.dmesg()
    oom = hc.parse_oom(kernel_text, t0)
    dxg = hc.dxg_lines(kernel_text, t0)
    kernel = None if kernel_text is None else dict(dxg=dxg, oom_lines=oom)
    log(f'run {unit} end: exit {rc}, systemd result {result_line}, kernel {kernel}')
    paths = xc.cell_paths(arm, pair_set, pair)
    record_path = paths['resume_result'] if resume_rel else paths['result']
    unit_end = read_json(unit_dir/'unit_end.json')
    doc = read_json(record_path)
    if dxg is None and unit_end is not None and unit_end.get('dmesg_readable'):
        dxg = unit_end.get('dxg_lines')                 # the supervisor's window when the chain could not read dmesg
    status_u, kind_u, evidence = unit_reading(unit=unit, result_line=result_line, unit_end=unit_end, oom_lines=oom, journal=journal)
    locked = lock_path.exists()
    if locked or (unit_dir/'unit_start.json').exists():
        hc.write_json_once(unit_dir/'unit_reading.json', dict(schema=xc.SCHEMA.format('unit_reading'), unit=unit, status=status_u, kind=kind_u, evidence=evidence, systemd_result=result_line, locked=locked,
                                                            systemd_run_output=str(out_txt), systemd_run_exit=rc, journal=str(journal_path), dxg_lines=dxg,
                                                            supervisor=None if unit_end is None else {k: unit_end.get(k) for k in ('utc', 'uptime', 'uptime_start', 'seconds', 'exit_code', 'killed_by_signal', 'killed_by_supervisor', 'sigterm_received')},
                                                            utc=hc.utc()))
    if dxg:
        raise Stop(STOP['infra'], f'{unit}: {dxg} dxg create_allocation failure lines during the run')
    if doc is None:
        if not locked:
            code = STOP['refused'] if rc == xc.EXIT['refused'] else STOP['infra']
            try:                                    # A5a (decision_log 2026-11-10 ~2:53 AM CST): the refusal records copied into the cell's record directory (the evaluation entry reads only that)
                kept, copy_problem = hc.keep_refusals(xc.roots(pair_set)[0], paths['record'], arm=arm, pair=pair, pair_set=pair_set), None
            except (hc.Refused, OSError) as exc:
                kept, copy_problem = [], f'; copying them failed: {exc}'
            raise Stop(code, f'{unit}: the driver stopped before its lock (exit {rc}, {result_line}); its refusal record is in {Path(batch)/"refusals"}, copied to '
                             f'{kept[-1]["path"] if kept else "-"}{copy_problem or ""}; see {out_txt}')
        status, kind = status_u, kind_u
        if status is None:
            if unit_end is not None and unit_end.get('killed_by_signal') is None and unit_end.get('exit_code') is not None:
                status, kind = 'wrapper_error', f'driver_exit_without_record:{unit_end.get("exit_code")}'     # the driver exited by itself after its lock and wrote no record: our code
            else:
                status, kind = 'needs_lead_rejudgment', f'no_record:{result_line}'
        reason = f'{unit}: the driver left no terminal record (systemd {result_line}, exit {rc}): {status} ({kind})'
        doc = write_unit_record(arm, pair_set, pair, status=status, kind=kind, reason=reason, unit=unit, unit_rc=rc, result_line=result_line, journal=str(journal_path), kernel=kernel,
                                evidence=evidence, resume_rel=resume_rel, unit_dir=unit_dir)
    elif overrides(doc.get('status'), status_u):
        target, _ = override_record(arm, pair_set, pair, doc, status=status_u, kind=kind_u, evidence=evidence, unit=unit, result_line=result_line, unit_dir=unit_dir, resume_rel=resume_rel)
        raise Stop(STOP['infra'] if status_u == 'infra_failure' else STOP['rejudge'],
                   f'{unit}: the unit\'s evidence reads {status_u} ({kind_u}) although the driver wrote {doc.get("status")}: {target} supersedes {record_path} (kept); '
                   + ('the resumed cell is final, no output (infrastructure)' if resume_rel else 'the cell needs a committed resume or no_resume authorization'))
    elif result_line in ('signal', 'core-dump'):
        raise Stop(STOP['infra'], f'{unit}: the driver ended by a signal ({result_line}) after writing {record_path}')
    status = doc.get('status')
    log(f'{arm} {pair}: {status}' + (f" ({(doc.get('failure') or {}).get('kind')})" if doc.get('failure') else ''))
    if status in ('ok', 'method_failure', 'unit_limit'):
        return status
    if status == 'wrapper_error' and doc.get('wrapper_error_policy') == 'identity':
        return status
    if status == 'wrapper_error':
        raise Stop(STOP['wrapper'], f'{unit}: wrapper error under the stop policy: {(doc.get("failure") or {}).get("reason")}')
    if status == 'needs_lead_rejudgment':
        raise Stop(STOP['rejudge'], f'{unit}: needs the lead\'s rejudgment: {(doc.get("failure") or {}).get("reason")}')
    raise Stop(STOP['infra'], f'{unit}: {status}: {(doc.get("failure") or {}).get("reason")}')


def batch_record(pair_set, arms, *, code_run=None):
    """batch.json's head: the preregistration (sha256 now, the pinned value, the problem), the frozen configuration (sha256 now and pinned, m14), the code; queue adds each arm's checks."""
    prereg = xc.REPO/xc.PREREG_REL
    frozen = xc.REPO/xc.FROZEN_CONFIG_REL
    if xc.is_lpba(pair_set):                        # LP-B review minor 2: an LPBA batch records the LPBA40 preregistration (its pins are checked by queue: lpba_common.check_pins)
        lc = xc.lpba().lc
        path = xc.REPO/lc.PREREG_REL
        digest = xc.sha256(path) if path.is_file() else None
        preregistration = dict(path=lc.PREREG_REL, sha256=digest, pinned_sha256=lc.PREREG_SHA256, problem=None if digest == lc.PREREG_SHA256 else f'{lc.PREREG_REL}: sha256 {digest} is not the pinned one')
    else:
        preregistration = dict(path=xc.PREREG_REL, sha256=xc.sha256(prereg) if prereg.is_file() else None, pinned_sha256=xc.PREREG_SHA256, problem=xc.prereg_problem())
    return dict(schema=xc.SCHEMA.format('batch'), pair_set=pair_set, arms=list(arms),
                preregistration=preregistration,
                frozen=dict(path=xc.FROZEN_CONFIG_REL, sha256=xc.sha256(frozen) if frozen.is_file() else None, pinned_sha256=xc.FROZEN_CONFIG_SHA256),
                code=hc.code_identity(run=code_run), frozen_tree=None, arm_checks={},
                note='frozen_tree: none (the external arms do not use the paper-1 frozen tree); arm_checks: xb_core.check_frozen and check_environment of each arm at the start (m14)')


def new_batch_dir(parent):
    """batches/xb_<UTC>, with a suffix when a batch of the same second is there (never reused)."""
    base = Path(parent)/f'xb_{xc.utc()}'
    candidate, k = base, 1
    while candidate.exists():
        k += 1
        candidate = base.with_name(f'{base.name}_{k}')
    return candidate


def queue(pair_set, arms, *, pairs=None, resumes=None, system=None, batch=None, log=None):
    system = System() if system is None else system
    resumes = dict(resumes or {})
    record_root = xc.roots(pair_set)[0]
    batch = Path(batch) if batch is not None else new_batch_dir(Path(record_root)/'batches')
    batch.mkdir(parents=True, exist_ok=False)
    chain_log = batch/'chain.log'

    def say(msg):
        line = f'{hc.utc()} {msg}'
        with chain_log.open('a', encoding='utf-8') as stream:
            stream.write(line+'\n')
        (log or print)(line)
    record = batch_record(pair_set, arms)
    record.update(utc_start=hc.utc(), disk_start=hc.disk_reading(), cells=[])
    code = 0
    try:
        plan = cells(pair_set, arms, pairs)
        if pair_set == 'holdout' and not record['code']['clean']:
            raise Stop(STOP['refused'], f"the code tree is not clean: {record['code']}")
        if pair_set == 'holdout':                       # A5a: rule one (contract C24.3; decision_log 2026-11-10 ~2:59 AM CST (1)): the code is the pinned commit's, P1 from pinned_code.json
            record['pinned_code'], issues = system.pinned_code()
            if issues:
                raise Stop(STOP['refused'], f'rule one (the pinned commit of runs/holdout_B_v1/pinned_code.json): {issues}')
        if pair_set == 'holdout' and record['preregistration']['problem']:
            raise Stop(STOP['refused'], f"I5: {record['preregistration']['problem']}")
        if xc.is_lpba(pair_set):                        # LP-B (C26.9, C26.6-3): the code and the pins of the mode
            record['lpba_code'], issues = xc.lpba().code_record(pair_set, run=getattr(system, 'code_run', None))
            if issues:
                raise Stop(STOP['refused'], f'the code ({pair_set}): {issues}')
            try:
                record['lpba_pins'] = xc.lpba().lc.check_pins()
            except hc.Refused as exc:
                raise Stop(STOP['refused'], f'the pins ({pair_set}): {exc}') from None
        for arm in arms:
            try:
                record['arm_checks'][arm] = dict(frozen=xc.check_frozen(arm))
                record['arm_checks'][arm]['environment'] = xc.check_environment(arm)
            except hc.Refused as exc:
                record['arm_checks'].setdefault(arm, {})['refused'] = str(exc)
                raise Stop(STOP['refused'], f'{arm}: {exc}') from None
        todo = [(a, p) for a, p in plan if cell_state(a, pair_set, p)[0] in ('not_run', 'refused')]
        need_d = sum(xb_drive.need_gib(a, pair_set)[0] for a, _ in todo)
        need_home = sum(xb_drive.need_gib(a, pair_set)[1]-1. for a, _ in todo)+1.
        try:
            record['disk_batch'] = system.disk(need_d, need_home)
        except hc.InfrastructureFailure as exc:
            raise Stop(STOP['infra'], f'the batch disk guard: {exc}') from None
        say(f'start: {len(plan)} cells, {len(todo)} to run; need D: {need_d:.1f} GiB, /home {need_home:.1f} GiB')
        for arm, pair in plan:
            state, _ = cell_state(arm, pair_set, pair)
            row = dict(arm=arm, pair=pair, state_before=state)
            record['cells'].append(row)
            if state == 'arm_stopped':
                say(f'skip {arm} {pair}: the arm is stopped (P19)')
                continue
            if state in ('done', 'final'):
                if state == 'final':
                    row['final_by'] = final_source(arm, pair_set, pair)      # A5a review m5: which record, with its sha256
                say(f'skip {arm} {pair}: {state}')
                continue
            resume_rel = resumes.get((arm, pair))
            if state == 'needs_resume' and resume_rel is None:
                if ((xc.effective_record(xc.cell_paths(arm, pair_set, pair)) or {}).get('status')) == 'needs_lead_rejudgment':
                    classified = rejudged_infrastructure(arm, pair_set, pair)
                    if classified is not None:          # A5a review m5: rejudged infrastructure only classifies; the cell waits for a committed resume record
                        row['rejudgment'] = classified
                        raise Stop(STOP['resume'], f"{arm} {pair}: needs_lead_rejudgment, rejudged infrastructure ({classified['path']}): it only classifies; a rerun needs a committed resume "
                                                   f'record (prereg revision 2 item 18, §5.4)')
                    _, problems = rejudged(arm, pair_set, pair)
                    raise Stop(STOP['rejudge'], f'{arm} {pair}: needs_lead_rejudgment: a committed rejudgment record runs/holdout_B_v1/rejudgments/{arm}/{pair}_<k>.json (prereg revision 2 item 18: '
                                                f'unit_limit or no_output writes the cell; infrastructure only classifies, a rerun then needs a committed resume record): {problems or "none committed"}')
                raise Stop(STOP['resume'], f'{arm} {pair}: a lock or record without a complete record; a rerun needs a committed resume record (prereg §5.4)')
            if state in ('not_run', 'refused') and resume_rel is not None:
                raise Stop(STOP['refused'], f'{arm} {pair}: a resume record for a cell that never took its lock')
            if state == 'refused':
                try:                                    # A5a: the copies first (a chain stopped before it copied them), then BZ's P20 no_resume, then a rerun authorization
                    row['refusal_copies'] = hc.keep_refusals(record_root, xc.cell_paths(arm, pair_set, pair)['record'], arm=arm, pair=pair, pair_set=pair_set)
                except hc.Refused as exc:
                    raise Stop(STOP['refused'], f'{arm} {pair}: its refusal records could not be copied: {exc}') from None
                try:
                    declined, declined_problem = declined_refusal(arm, pair_set, pair), None
                except hc.Refused as exc:                # the newest refusal record is not known: no P20 here; a rerun authorization still covers the cell (I1)
                    declined, declined_problem = None, str(exc)
                if declined is not None:
                    row['no_resume'] = dict(path=declined.path, sha256=declined.sha256)
                    say(f'skip {arm} {pair}: refused before its lock, BZ\'s no_resume (P20): written, no output')
                    continue
                try:
                    if xc.authorizations_problem(pair_set) is not None:
                        raise hc.Refused(hc.RERUN_STAGE, xc.authorizations_problem(pair_set))
                    row['rerun_authorization'] = hc.rerun_authorization(record_root, tool=xb_drive.TOOL, arm=arm, pair=pair, pair_set=pair_set)
                except hc.Refused as exc:
                    raise Stop(STOP['refused'], f'{arm} {pair}: {exc} (refused again after a rerun: BZ\'s committed no_resume record naming the newest refusal record\'s copy in '
                                                f'{xc.cell_paths(arm, pair_set, pair)["record"]}/{hc.REFUSAL_COPIES} writes the cell, P20)'
                                                + (f'; P20 not readable: {declined_problem}' if declined_problem else '')) from None
            while (xc.lpba().hold(pair_set) if xc.is_lpba(pair_set) else Path(record_root)/'hold').exists():
                say('hold file present; waiting')
                system.sleep(60)
            try:
                row['status'] = run_one(arm, pair_set, pair, system, batch=batch, log=say, resume_rel=resume_rel,
                                        pinned=(record.get('lpba_code') or {}).get('commit') if xc.is_lpba(pair_set) else None)
            except (hc.InfrastructureFailure, hc.Refused) as exc:
                raise Stop(STOP['infra'], f'{arm} {pair}: before the unit: {exc}') from None
    except Stop as exc:
        code = exc.code
        record['stop'] = dict(code=code, reason=str(exc))
        say(f'STOP ({code}): {exc}; nothing rerun; waiting for BZ')
    record.update(utc_end=hc.utc(), disk_end=hc.disk_reading(), exit=code)
    hc.write_json_once(batch/'batch.json', record)
    if code == 0:
        say('end')
    return code


def collect_intermediates(pair_set, arms, out):
    """One manifest (the cleanup script's format) of the intermediates.tsv of every done cell of `arms` whose terminal record is ok (prereg §5.3: deleted only after the audit, by the reviewed step
    experiments/xb/delete_verified.sh on a manifest the lead approves).  Each cell's manifest is checked against the sha256 its record names.  Written once to `out`; returns a summary."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'{out} exists')
    rows, cells_used = [], []
    for arm, pair in cells(pair_set, arms):
        state, doc = cell_state(arm, pair_set, pair)
        if state != 'done' or (doc or {}).get('status') != 'ok':
            continue
        cleanup = doc.get('cleanup') or {}
        manifest = Path(cleanup.get('manifest', ''))
        if not manifest.is_file() or xc.sha256(manifest) != cleanup.get('manifest_sha256'):
            raise Stop(STOP['refused'], f'{arm} {pair}: the intermediates manifest {manifest} is missing or not the one its record names')
        lines = manifest.read_text(encoding='utf-8').splitlines()
        if not lines or lines[0]+'\n' != xc.MANIFEST_HEADER:
            raise Stop(STOP['refused'], f'{manifest}: not a manifest')
        rows += [line for line in lines[1:] if line.strip()]
        cells_used.append(f'{arm} {pair}')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        stream.write(xc.MANIFEST_HEADER)
        for line in rows:
            stream.write(line+'\n')
    return dict(out=str(out), sha256=xc.sha256(out), n_cells=len(cells_used), n_files=len(rows), bytes=sum(int(r.split('\t')[1]) for r in rows))


def parse_resumes(values):
    out = {}
    for value in values or []:
        parts = value.split('=', 2)
        if len(parts) != 3:
            raise SystemExit(f'--resume {value!r}: ARM=PAIR=REL')
        out[(parts[0], parts[1])] = parts[2]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect-intermediates', help='one manifest of the intermediates of the done cells, for the lead to approve before delete_verified.sh')
    c.add_argument('--pair-set', required=True, choices=xb_drive.PAIR_SETS)
    c.add_argument('--arms', required=True)
    c.add_argument('--out', required=True)
    for name in ('plan', 'queue', 'launch-string'):
        p = sub.add_parser(name)
        p.add_argument('--pair-set', required=True, choices=xb_drive.PAIR_SETS)
        p.add_argument('--arms', required=True)
        p.add_argument('--pairs', default=None)
        if name == 'queue':
            p.add_argument('--resume', action='append')
    o = ap.parse_args(argv)
    arms = [a for a in o.arms.split(',') if a]
    if o.command == 'collect-intermediates':
        try:
            print(collect_intermediates(o.pair_set, arms, o.out))
        except (Stop, hc.Refused, FileExistsError) as exc:
            print(f'refused: {exc}', file=sys.stderr)
            return STOP['refused']
        return 0
    pairs = [p for p in o.pairs.split(',') if p] if o.pairs else None
    if o.command == 'launch-string':
        print(shlex.join(chain_launch_string(o.pair_set, arms)))
        return 0
    if o.command == 'plan':
        try:
            plan = cells(o.pair_set, arms, pairs)
        except (Stop, hc.Refused) as exc:
            print(f'refused: {exc}', file=sys.stderr)
            return STOP['refused']
        for arm, pair in plan:
            state, _ = cell_state(arm, o.pair_set, pair)
            print(f'{arm} {pair} {state}: {shlex.join(unit_argv(arm, o.pair_set, pair))}')
        return 0
    return queue(o.pair_set, arms, pairs=pairs, resumes=parse_resumes(o.resume))


if __name__ == '__main__':
    sys.exit(main())
