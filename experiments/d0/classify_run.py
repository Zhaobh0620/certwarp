"""D0: classify one finished run. Copied with provenance from the paper-1 H1RUN classifier
archive/seq_vp_3d/results/verification/3d_stage_H1RUN_v1/analysis/h1_classify.py (sha256 b0ba3539...f517; POSTFREEZE
failure classes, user 2026-10-21); the only change is the script name written into the record. Read only (small JSON /
text files of the run). Prints one line "<class> <kind>: <reason>" and writes the record to --out (exclusive create).

Classes (run_queue.sh acts on the class; D0 protocol section 6):
  ok              exit 0, metrics_role formal_native_outer_commit, final_native.json/.npz present -> evaluate, continue
  handover_only   exit 0, a certified map but no native commit -> recorded (method result), evaluate, continue, kept whole
  method_failure  handover_failure / no_certified_map / timeout / method_error -> recorded, continue, kept whole
  infrastructure  dxg / oom / no_exit_code / guard / signal -> STOP the queue (run kept whole, nothing rerun)
  unsure          anything else -> STOP the queue
Usage: classify_run.py RUN_DIR --watchdog S --dxg N --oom N --out FILE"""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

METHOD_EXCEPTIONS = ('ValueError', 'InputValidationError', 'RuntimeError', 'AssertionError', 'FloatingPointError',
                     'ZeroDivisionError', 'ArithmeticError', 'OverflowError', 'LinAlgError')
INFRA_MARKERS = ('CampaignResourceError', 'KeyboardInterrupt', 'MemoryError', 'OutOfMemoryError', 'out of memory',
                 'CUDA', 'cuda error', 'cudaError', 'No space left', 'Input/output error', 'Errno', 'Segmentation fault',
                 'core dumped', 'Fatal Python error', 'Killed', 'source changed', 'input identity mismatch',
                 'input changed during run', 'OSError', 'IOError', 'BrokenPipeError', 'PermissionError', 'TimeoutError',
                 'InterruptedError', 'SystemError')
EXC_LINE = re.compile(r'^([A-Za-z_][\w.]*)(?::\s?(.*))?$')
GRACE = 15        # launch_stage_3d_e48.sh: timeout --signal=TERM --kill-after=15s
SLACK = 2         # supervisor time stamps have 1 s resolution


def rd(p):
    return json.loads(Path(p).read_text())


def whole_wall(run):
    try:
        t0, t1 = (datetime.fromisoformat((run/f).read_text().strip().replace('Z', '+00:00'))
                  for f in ('supervisor_started_utc.txt', 'supervisor_finished_utc.txt'))
        return (t1-t0).total_seconds()
    except Exception:
        return None


def last_traceback(text):
    """(exception lines, last exception line) of the last 'Traceback (most recent call last):' block of run.log, or
    (None, None). Exception lines = the unindented lines after the header (the exception lines of the block and of the
    exceptions chained into it); the indented frame and source lines are left out, so code text cannot match a marker."""
    lines = text.splitlines()
    starts = [i for i, l in enumerate(lines) if l.startswith('Traceback (most recent call last):')]
    if not starts:
        return None, None
    first = [i for i in starts if i < starts[-1]]
    begin = starts[-1]
    # a chained exception prints its own Traceback header after "During handling ..." / "The above exception ...": take the
    # whole chain, i.e. from the first header that is followed only by chain separators and further headers
    for i in reversed(first):
        between = lines[i:begin]
        if any(l.startswith(('During handling of the above exception', 'The above exception was the direct cause')) for l in between):
            begin = i
        else:
            break
    unindented = [l for l in lines[begin:] if l and not l.startswith((' ', '\t'))
                  and not l.startswith(('Traceback (most recent call last):', 'During handling', 'The above exception'))]
    exc = None
    for l in unindented:
        if EXC_LINE.match(l):
            exc = l                                       # the last unindented exception line
    return '\n'.join(unindented), exc


def classify(run, watchdog, dxg, oom):
    ev = dict(dxg_lines=dxg, oom_lines=oom, watchdog_seconds=watchdog)
    if dxg > 0:
        return 'infrastructure', 'dxg', f'{dxg} new dxg create_allocation failure lines during the run', ev
    if oom > 0:
        return 'infrastructure', 'oom', f'{oom} new kernel OOM lines during the run', ev
    ecf = run/'exit_code'
    if not ecf.exists():
        return 'infrastructure', 'no_exit_code', 'no exit_code file (unit ended without one, disk floor or restart)', ev
    ec_text = ecf.read_text().strip()
    ev['exit_code'] = ec_text
    try:
        ec = int(ec_text)
    except ValueError:
        return 'unsure', 'exit_code', f'exit_code {ec_text!r} is not an integer', ev
    wall = whole_wall(run)
    ev['whole_wall_seconds'] = wall
    ev['started_json'] = (run/'started.json').exists()
    proto = rd(run/'protocol.json') if (run/'protocol.json').exists() else {}
    task = run/((proto.get('task') or {}).get('name') or 'missing_task_name')
    log = (run/'run.log').read_text(errors='replace') if (run/'run.log').exists() else ''
    block, exc = last_traceback(log)
    ev['exception_line'] = exc
    if block is not None and 'CampaignResourceError' in block:
        return 'infrastructure', 'guard', f'storage / host memory guard: {exc}', ev
    if ec == 0:
        if not (task/'summary.json').exists() or not (run/'completion.json').exists() or not (task/'worker_completion.json').exists():
            return 'unsure', 'records', 'exit 0 without summary.json / completion.json / worker_completion.json', ev
        s = rd(task/'summary.json')
        ev.update(status=s.get('status'), reason=s.get('reason'), metrics_role=s.get('metrics_role'),
                  native_commits=s.get('native_commits'))
        if s.get('status') == 'gpu_certificate_mismatch':
            return 'unsure', 'gpu_certificate_mismatch', f"status gpu_certificate_mismatch: {s.get('reason')}", ev
        role = s.get('metrics_role')
        if role == 'formal_native_outer_commit':
            if (task/'final_native.json').exists() and (task/'final_native.npz').exists():
                return 'ok', 'native', f"{s.get('status')}; {s.get('native_commits')} native commits", ev
            return 'unsure', 'records', 'native commits but no final_native.json / final_native.npz', ev
        if role == 'certified_native_initializer_NOT_native_outer_commit':
            return 'handover_only', 'no_native_commit', f"{s.get('status')}: {s.get('reason')}", ev
        if role == 'no_native_result':
            kind = 'handover_failure' if s.get('status') == 'transfer_rejected' else 'no_certified_map'
            return 'method_failure', kind, f"{s.get('status')}: {s.get('reason')}", ev
        return 'unsure', 'metrics_role', f'unknown metrics_role {role!r}', ev
    if ec == 124:
        if wall is not None and wall >= watchdog-SLACK:
            return 'method_failure', 'timeout', f'watchdog {watchdog} s fired (exit 124, whole wall {wall:.0f} s)', ev
        return 'unsure', 'exit_124', f'exit 124 but whole wall {wall} s < watchdog {watchdog} s', ev
    if ec == 137 and wall is not None and wall >= watchdog+GRACE-SLACK:
        return 'method_failure', 'timeout', f'watchdog {watchdog} s fired, killed after the {GRACE} s grace (exit 137)', ev
    if ec >= 128:
        return 'infrastructure', 'signal', f'exit {ec} (signal {ec-128}) outside the watchdog (whole wall {wall} s)', ev
    if not ev['started_json']:
        return 'unsure', 'before_task_clock', f'exit {ec} before the task clock (no started.json): {exc}', ev
    if exc is None:
        return 'unsure', 'no_traceback', f'exit {ec} without a traceback in run.log', ev
    markers = [m for m in INFRA_MARKERS if m in block]
    ev['infra_markers'] = markers
    if markers:
        return 'unsure', 'infra_marker', f'exit {ec}: {exc} (markers {markers})', ev
    name = EXC_LINE.match(exc).group(1).rsplit('.', 1)[-1]
    ev['exception_type'] = name
    if name in METHOD_EXCEPTIONS:
        return 'method_failure', 'method_error', f'exit {ec}: {exc}', ev
    return 'unsure', 'exception_type', f'exit {ec}: {exc} ({name} is not a method exception)', ev


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--watchdog', type=float, required=True)
    ap.add_argument('--dxg', type=int, required=True)
    ap.add_argument('--oom', type=int, required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    run = Path(a.run)
    try:
        cls, kind, reason, ev = classify(run, a.watchdog, a.dxg, a.oom)
    except Exception as exc:                       # a broken record set is itself a reason to stop
        cls, kind, reason, ev = 'unsure', 'classifier_error', f'{type(exc).__name__}: {exc}', {}
    rec = dict(script='classify_run.py (D0; from h1_classify.py)', script_sha256=hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
               classified_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), run=run.name, run_dir=str(run),
               **{'class': cls}, kind=kind, reason=reason, evidence=ev)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(rec, f, indent=1, default=str)
    print(f'{cls} {kind}: {reason}'.replace('\n', ' ')[:400])
    return 0


if __name__ == '__main__':
    sys.exit(main())
