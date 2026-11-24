"""LP-F (decision_log 2026-11-20, the lead's ruling 1 on the LPBA40 run handbook; LPBA40 preregistration 06121278... §8.6; contract C26.8-7): the dxg kernel-line monitor of the
LPBA40 runs, in the code tree.  A port of the hold-out's last monitor (the "m version", decision_log 2026-11-18 lines 498, 500): its evaluator dxg_eval_m.py and its watcher watch_m.sh,
read from the hold-out run's record directory of step 9's last arm (SOURCE below: path and sha256, the sha256 that decision_log line 500 names).  Neither file is imported or copied
whole: the rule, the texts and the δ handling are re-written here around the LPBA40 roots, units and records.  Revision 1 (LP-F review 3/2/9, docs/bayesreg/reviews/LP-F_复核_20261120.md;
the lead's rulings, decision_log 2026-11-20 ~3:42 PM CST) works to the general ruling that every case in which the m version holds must hold here too; the relaxations that remain
are listed at the end, each with its ruling.

    python -B experiments/lpba/dxg_monitor.py now                                               the uptime and the epoch to pass as --since and --start-epoch (read just before a step starts)
    python -B experiments/lpba/dxg_monitor.py delta --out F.json [--tries 3] [--drift .15]      δ measured with nothing running (refuses while a tag-* / seq-vp-* unit is active)
    python -B experiments/lpba/dxg_monitor.py merge-delta A.json B.json [C.json ...]             the merged measured interval (every measurement of a step) for the last re-judgement
    python -B experiments/lpba/dxg_monitor.py before --mode M --since S --start-epoch E --out F    cells_before.json (R3) and its problems_sha256, for --ack-before
    python -B experiments/lpba/dxg_monitor.py eval  --mode M --since S --delta LO,HI --start-epoch E --commit SHA [--active 0|1] [--no-hold] [--log F ...] [--expect-unit U ...]
                                                     [--pending F] [--ack F] [--ack-before SHA256]
    python -B experiments/lpba/dxg_monitor.py watch --mode M --since S --delta LO,HI --start-epoch E --commit SHA --out DIR (--watch-unit U ... | --watch-pid P) [--log F ...]
                                                     [--wait-start 300] [--expect-unit U ...] [--ack F] [--ack-before SHA256] [--continue-from OLD_OUT]
    python -B experiments/lpba/dxg_monitor.py selftest                                          the synthetic cases (a temporary results root; the real hold never touched)

Modes: 'lpba40_test' (the formal run: units tag-lp-*, not tag-lp-feas-*, not tag-lp-dev*; records under lpba_common.roots('lpba40_test').root; step 11's tag-lp-eval included, a CPU
cell), 'lpba40_feas' (the feasibility pair: tag-lp-feas-*; lpba_common.roots('lpba40_feas').root), 'devcheck' (the development checks (b), (c): tag-lp-devb, tag-lp-devc and (c)'s nested
tag-he-c-*; <RESULTS>/lpba40_devcheck_v1).  The hold file is lpba_common.roots(...).hold, read at call time from lpba_common.RESULTS: <RESULTS>/lpba40_test_v1/hold for every mode
(C26.2-3, prereg §8.7: one file pauses both modes; lpdev_common: the development checks too); made with its directory when that is not there yet (touch_hold).  The monitor touches it on
an event and never removes it; it never starts, stops or kills a unit.

How it runs: an ordinary background process (setsid nohup ... &, its own log), never a systemd unit: every entry's machine gate waits while any other tag-* or seq-vp-* unit is
active, so a monitor in a tag-* unit would stop the run it watches (decision_log 2026-11-20, main cf0cfc1a).  `watch` and `delta` refuse inside a tag-* or seq-vp-* service or scope
(/proc/self/cgroup, as feas_chain.chain_unit).  `watch` may start before the step (a systemd-run --wait step blocks the runner's shell): it waits up to --wait-start seconds for a
watched unit or process to become active (none in time: an event, the hold, exit 5), then polls every 20 s while it is, then makes the last evaluation FINAL_SETTLE_S after the step
ended (again 2 x FINAL_SETTLE_S later if a cell still shows no end: the manager's last lines may lag).  Arguments that do not parse (--since, --delta, --start-epoch, --commit,
--ack-before, --continue-from, non-finite numbers included) hold, write an EVENT and an END line, and exit 4 (the m version's evaluator exit; review B3 (a)); a refused --out holds
and writes only to stderr (nothing into a refused directory); a watch refused inside a unit holds and exits 4; any exception of an evaluation holds and is an EVENT with rc 4, the
watch going on; anything else that would end the loop (SystemExit, KeyboardInterrupt) holds, writes EVENT and END and exits 4.  The END line and watch_end.json carry the exit code.
Exit codes: eval 0 no event, 3 events (held), 4 an argument or the evaluator (held); watch 0, 4, 5 (not started, held).  The run commands: runs/20261119-lpba/LP-F_运行命令草稿.md.

Acknowledgements (the LP-F revision 1 diff review I1; decision_log 2026-11-20 ~4:13 PM CST; revision 2 diff review B1, I1, ~4:38 PM CST): every event has a stable id (event_id: its
kind, its cell -- for a journal failure line the window open when it was written --, its source line (a kernel line's time, whole text and order among lines of the same time and
text; a journal line's time and text; a log line's number and text; a record's path and sha256; a condition's name and the start of its episode -- the first poll it held, carried
from poll to poll by the watcher, so that a condition gone and back is a new event (revision 3 diff review I1; a one-off eval starts the episode at its own now)) and its cell's
window starts), listed in doc 'event_items'.  Each evaluation re-reads the ack file (--ack;
watch's default <out>/ack.json), JSON {"schema": "bayesreg.lpba40.dxg_ack.v1", "entries": [{"event_id", "cell", "kind", "decision_log", "bz_words"}]}: an entry releases the one
event it names (cell and kind must be that event's; "-" for an event without a cell); a later event of the same kind is judged anew.  An entry needs the decision_log line (not
empty).  Without BZ's words (bz_words empty) it is accepted only on the lead's white list (LEAD_KINDS) and only when the cell has no kernel event in its windows (driver_stop: none
in the whole scan):
  status              the tool's own status in LEAD_STATUSES and no spy, guard, wrapper or breach cause in the record, nor an admission or provenance refusal after the lock
                      (input_refused_after_lock:*, input_rejected: prereg §8.3, lines 757, 760; revision 3 diff review I2) (lead_status): external arm method_failure, infra_failure
                      (decision_log line 756 (4), line 759 (I2); xb_core.EXIT) and unit_limit (line 757 (B1): unit_limit and infrastructure may cite line 520 for a rerun;
                      decision_log 2026-11-20 ~4:52 PM CST (1), revision 3a); configuration C infra_failure, unit_limit (line 760; hb_configc's result statuses); the method arm
                      never (decision_log 2026-11-20 ~5:05 PM CST (I3), ~5:15 PM CST (3): its infrastructure stops go to BZ); hi, devb, devc never;
  driver_stop         the exit code of the STOP line in LEAD_STOP_CODES: the external arms' chain 20, 24, 25 (line 759 (I2); xb_chain.STOP), configuration C's queue 20, 24
                      (line 760; hb_core.EXIT; 21 stop of many causes, 22, 25, 26 not); the method arm's queue and the review's queue never (~5:15 PM CST (2));
  records_missing, started_unfinished (decision_log ~4:29 PM CST (1)).
Every other kind (reading, unit_failed, driver_failed, log_stop, arm_stopped, refusal, cells, expected, source, before, the kernel kinds) needs BZ's words.  The code cannot check that
bz_words are BZ's own words nor that the decision_log line records the tool's judgement (infrastructure or unit_limit): the handbook has the line quote them, and each step's
independent review compares them (~4:38 PM CST (I1)).  Rejected entries are listed (doc 'ack.rejected') and their events hold.  While a watcher runs, the lead adds entries and the
next poll reads them; then the hold may be removed.  A watcher that takes over (a step resumed after a disposed hold, a watcher restarted) passes --continue-from <the old --out>:
the same mode, --since, --start-epoch, --commit and ack file, a δ interval that holds the old one, and the old watcher ended -- so its scan starts where the old one started and no
kernel line falls outside every scan (I1 (ii)).  The watchers of one step share a monitor root (--out, or its parent when that holds a watcher, as $MON/r2 under $MON): a new watcher
refuses (hold, exit 4) while one of the same mode there has no END line, or when one is there and --continue-from is not given (~4:38 PM CST, minor).
Expected units (I2 of the revision 1 diff review): at watch's last evaluation every --watch-unit and --expect-unit, and at an `eval` every --expect-unit and every unit of --pending
<pending.json>, must have a window in the user journal; an empty user journal (0 entries since --start-epoch) at a last evaluation or an `eval --active 0` holds.

The rule (BZ 2026-11-17 ~11:59 AM CST and 2026-11-18 ~2:00 PM CST, both "按建议", decision_log lines 473, 489; the m version's whole-text classes, decision_log line 498):
  classes   kernel lines from `journalctl -k -o short-monotonic` (normalised to the dmesg form), kernel time >= SINCE + min(0, δ_lo) (the step's start uptime; no offset added when δ_lo
            >= 0, as the m version; with a negative δ the scan starts |δ_lo| earlier, so that a line written after the start but stamped before SINCE on the slower kernel clock is
            counted -- revision 4).  By the whole message after the
            '[ t]' stamp: W = W_TEXT, P = P_TEXT, qai = QAI_TEXTS, EAGAIN = EAGAIN_TEXTS; CA = any line with 'create_allocation' and 'fail'; any other line with 'dxg' or a DXG_KEYS word ->
            dxg_other; the rest non_dxg, the WSL 'mini_init (<n>): drop_caches: 1' class (DROP) whitelisted.
  readings  a kernel time t is read twice as an uptime: t - δ_lo and t - δ_hi (δ measured: `delta`, at least two tries that agree; while a step runs both ends are widened by the drift
            allowance DRIFT -- δ_lo - .15, δ_hi + .15 (revision 4a: the drift's direction at this boot is not known; the m version widened the upper end only) --; BZ 11-18 (5): no
            floors; a δ that is not δ_lo <= δ_hi is refused, never truncated).  δ is signed (LP-F revision 4; BZ 11-18 (5) asks for the measured interval and no floor, not for
            δ >= 0; decision_log 2026-11-20 ~6:45 PM CST): this boot's kernel clock runs about 31.6 s behind /proc/uptime -- eight measurements 75 s apart, 6:36-6:45 PM CST, δ in
            [-31.69, -31.58], no drift over 9 minutes (runs/20261120-lpba-run/delta_diag_20261120.jsonl).
  cells     two sources, both read (review B1; the lead's ruling B1):
            (1) the records on disk, as the m version: every in-unit supervisor directory (experiments/xb/xb_unit.py's unit_start.json, unit_end.json) of the mode's root whose start uptime
                (min of unit_start.uptime and unit_end.uptime_start) is >= SINCE -- an external arm's <root>/<arm>/<pair>/unit*, the method's and configuration C's
                <root>/{method,configc}/<pair>/unit*, the review's <root>/method/<pair>/unit_review*; window [start, unit_end.uptime];
            (2) the user journal of systemd's manager (`journalctl --user -b 0 -o json --output-fields=...`, its readability first checked as holdout_common.journal_alive): every unit of
                the mode except the drivers (*-queue, x-chain), window [Started, Consumed / Deactivated / Failed / Main process exited], CLOCK_MONOTONIC moved to /proc/uptime's clock by
                CLOCK_BOOTTIME - CLOCK_MONOTONIC.
            A supervised unit seen in one source only (beyond GRACE_S while the step is active; at the last evaluation always), an unreadable supervisor directory started since the
            step, a user journal that cannot be read, or a watched unit active for GRACE_S without its Started line -> hold.  Units with no supervisor (hi, devb, devc, prep, atropos,
            deva, input-lock, eval, the nested he-c / he-c3 / tag-he-c units) are known from the journal only: their records carry no uptime (see the end).  A running window ends at now + 5.
            Each criterion takes the stricter of the two windows: a non-dxg line holds when either reading lies in either window (their union); a W, P or EAGAIN line belongs to a
            supervised cell only when both readings lie in both windows (their intersection; the external arm's end is therefore unit_end.uptime, as the m version), else it is
            outside or split and holds.  A window without an end reaches the end of the scan (now + 5), also at the last evaluation, for the non-dxg lines and for its cell's kernel
            events (revision 2 diff review B2): a cell that started and was killed keeps its kernel lines.
  GPU steps the finest GPU segment that a cell's records time on /proc/uptime's clock (the lead's ruling I2), by kind:
            xb       (x-<arm>-<Sf>-<Sm>)     result.json's method.steps carrying max_gpu_mem_used_mib: [uptime_start, uptime_start + seconds_wall]; the anchor is its end (the m version)
            method   (he-<Sf>-<Sm>)          none: run_hx records T88's and the repair's seconds, not their uptimes (the frozen worker's launch uptime t0 is only used for its dmesg read)
            configc  (c-<Sf>-<Sm>)           none: hb_configc reads fc.uptime() at the frozen worker's launch for the kernel lines only, never records it
            review, hi, devb, devc, nested, every other unit
                                             none: hi_lpba / lpdev_b record seconds_wall per subject without its uptime; HE's (c) run likewise; the rest are CPU units
            A W, P or EAGAIN line in a cell without a GPU segment holds (no anchor, no step): no cell is exempted by its whole span.
  CA        any -> hold.
  W, P      both readings in one cell (the intersection above), and within TOL = 1 s of a GPU segment's end, and the cell's (b) and (c) pass -> recorded only; else hold.  While the
            watched step is active a line after the last known window, or in a cell not yet settled, is pending; the last evaluation (active 0) holds on anything still pending (m.2).
  EAGAIN    both readings inside one GPU segment of one cell (no tolerance) and the cell's (b) and (c) pass -> recorded only; else hold; pending as W/P.
  dxg_other any -> hold.
  non_dxg   either reading in any cell's window (either source) -> hold, unless DROP; outside every window -> recorded only; after the last window while active -> pending (placed again
            next poll).
  sources   the kernel journal unreadable or with no line, its first line after SINCE, dmesg unreadable, the journal short of dmesg in some class over their overlap (CCATS, re-read up to 3 times
            1 s apart) -> hold (m: journal source, coverage, BZ 11-18 (3) dmesg unreadable holds, m.3).
  (b)       the unit ended normally: external arm -- its effective record ok, no overridden.json, the unit reading's status and kind null, systemd success, systemd-run exit 0, the supervisor's
            child exit 0, no signal, no SIGTERM, not killed by the supervisor, no resume attempt (the m version, plus the last); method, configuration C and the review -- the unit reading's
            status and status_kind null, systemd success, systemd-run exit 0, the supervisor's child as above; the rest -- a stop line in the user journal and no failure line.
  (c)       the tool's own checks of the cell: external arm -- the m version's list with the D: moves taken out (LPBA40 moves nothing, prereg §11): spy passed, export check not suspicious,
            no move, disp.npz in the cell's work directory with the recorded sha256, the lock one line naming --commit (F or P_L) with the sha256 the record names, machine gate passed,
            the cleanup manifest's sha256, step exit 0, no memory-guard kill, no OOM line, no dxg line of the unit; method -- step4.json went on, an output line status, no spy breach, the lock
            one line naming --commit; configuration C -- result.json ok, no overridden.json, no spy_breach*.json, the lock one line naming --commit, no OOM / dxg line of the unit; hi, devb,
            devc -- the step's record passed (hi_record.json, devcheck_b.json, devcheck_c.json); any other cell -- none known, so (c) fails.  (Without a GPU segment a cell's W/P/EAGAIN hold
            whatever (b) and (c) read.)
  m.2       the last evaluation (review B2): a cell started (either source) without an end in either source, or without its records -- external arm result.json and unit_end.json; method
            unit_end.json and step4.json (step4_resume.json after a resume); configuration C unit_end.json and result.json (result_resume.json); review unit_end.json; hi hi_record.json;
            devb devcheck_b.json; devc devcheck_c.json -> hold.
  deferred  (the lead's ruling I1) a cell whose (b)(c) record is written after its unit by a step that checks the hold first -- devc (lpdev_c.py compare, DEFERRED_KINDS): at `watch`'s
            last evaluation its still pending W/P/EAGAIN lines and its missing record are written to <out>/pending.json, not held (the compare step only looks at the hold); after the
            compare has written devcheck_c.json the runner must run `eval --active 0` once more, which judges them as any cell (no GPU segment: a W/P/EAGAIN there holds), and holds when
            the record is missing or did not pass.  Every other event of that evaluation holds as usual.
  state     also hold, whatever the dxg lines: a failure line of any unit of the mode in the user journal (status not 0/, Failed with result, code=killed, code=dumped, Failed to start); a
            STOP line in a driver unit's own journal output or in a --log file; an external arm's cell whose effective status is not ok or that has overridden.json or a settled unit reading with a
            status, its arm_stopped.json; a method cell whose step4.json did not go on; a configuration C cell whose result is not ok; hi, devb, devc records that did not pass; a refusal
            record written since --start-epoch under the mode root's batches/*/refusals/; any exception of the evaluator (exit 4).
What it reads: the two journals, dmesg, /proc/uptime, the mode root's supervisor directories one level at a time, small JSON records of the mode's cells (no npz is opened; disp.npz is only
hashed, and only for an external arm's cell holding a W/P or EAGAIN line), the --log files.  It writes the hold file (an event) and, in `watch`, its own --out directory (outside every
mode root and the code tree): watch_start.json, refusals_before.json, poll.txt, dxg_live.jsonl, pending.json.

Changes against the m version (the relaxations among them are listed after this paragraph): the hold path, the unit names, the records and their roots (above); the user journal as a second source of cells and
windows, each criterion taking the stricter one (B1); GPU segments and (b)(c) for the other kinds of cell (the m version had XB cells only; I2); the D: move checks of (c) replaced by
"nothing moved, disp.npz in place" (LPBA40 keeps everything on /home); the lock's code commit is --commit (the m version's P1); the state checks generalised from one XB chain to the
mode's units; ids: the m version refused printing an OASIS-like token, here such a token is masked (S01-S40 are not hold-out ids, handbook §13.3) and the refusal stays as the last guard;
δ is measured by `delta` (the hold-out's procedure, decision_log line 477: /proc/uptime, one nvidia-smi, /proc/uptime, the new qai line's kernel time; δ in [t - u1 - .01, t - u0]);
while a step runs both ends are widened by DRIFT (.15, the m version's last value, decision_log line 500): δ_lo_live = δ_lo - DRIFT, δ_hi_live = δ_hi + DRIFT (revision 4a; the m version
widened the upper end only -- a stricter reading: the interval of readings only grows).
Relaxations against the m version, each with its ruling (decision_log 2026-11-20 ~3:42 PM CST, the LP-F review rulings; ~4:13 PM CST, the revision 1 diff review rulings):
  R1 (B3 (b))  refusal records count only when written at or after --start-epoch (the m version counted this arm's at any time): a rerun arm would otherwise hold from its start.  `watch`
               writes every earlier refusal record of the mode root (path, mtime, sha256) into <out>/refusals_before.json at its start, for the runner and the lead.
  R2 (I1)      a devc cell's pending W/P/EAGAIN lines and its missing devcheck_c.json at `watch`'s last evaluation go to pending.json instead of the hold; the `eval --active 0
               --pending <pending.json>` after lpdev_c.py compare judges them and holds (the m version had no such cell).
  R3 (B1 of the revision 1 diff review)  the evaluations judge the cells that started at or after --since; the m version judged every row of the arm's plan at every poll.  The cells of
               the mode root whose latest attempt started before --since are listed in cells_before.json (watch writes it at its start; `before` writes the same bytes): each with its
               status as the tool reads it, its records complete or not, every arm_stopped.json, the single-step records written before --start-epoch, and, as reference rows (problem
               false), the earlier attempts before --since of a cell of this step.  A problem row (not ok, not complete, arm_stopped) holds at every evaluation unless --ack-before
               names problems_sha256 (the mode and the problem rows only, each with its record's, reading's and unit_end's sha256; revision 2 diff review minor 4, revision 3
               diff review minor 2), which the lead gives only after recording the disposal in the decision log;
               no problem row needs no ack, and one disposal holds until a problem row changes.  A record directory without any attempt is a problem row only when it was made
               before --start-epoch (its creation time, statx; revision 5): two queues make it before the unit's supervisor writes unit_start.json -- the method queue
               (hb_method.run_one:157) before its machine gate, up to 15 min; configuration C's (hb_configc.run_one:515) after its gate, just before run_unit -- and feas-r2
               held on such a gap at 13:01:29 CST 2026-11-21 (decision_log that day ~1:03 PM CST).  (An external arm's record directory is made inside its unit, by
               xb_unit.write_once together with unit/ (xb_unit.py:54): no gap there.)  One made since then is listed under not_started while the step runs, no event; at the
               last evaluation (active 0) each one still without an attempt is an event (not_started, not on the lead's white list; revision 5a, B1 of the revision 5 review:
               the method queue ended by SIGTERM in its machine gate leaves no STOP line and no failure line).  A normal step leaves none: a queue goes on only after
               systemd-run --wait returns, and the last evaluation comes after the watched units.  One whose creation time cannot be read stays a problem row.
  R4 (I1 (i))  an accepted acknowledgement (above) takes its events off the hold; the m version had none (BZ removed the holds by hand).
  R5 (revision 4)  a negative δ is judged: the m version's evaluator asserted 0 <= δ_lo (exit 4, hold).  BZ 11-18 (5) asks for the measured interval and no floor, not for δ >= 0;
               decision_log 8f0e5fef (2026-11-20 ~6:45 PM CST): this boot's kernel clock runs about 31.6 s behind /proc/uptime.
Known limits (decision_log 2026-11-20 ~4:01 PM CST (2)): units without an in-unit supervisor (hi, devb, devc, prep, atropos, deva, input-lock, eval, the nested he-c / he-c3 /
tag-he-c units) are known from the user journal only -- their records carry no uptime -- and the last evaluation checks the records m.2 asks for (hi, devb, devc); the nested units
have no record on disk to pair with, so they take no part in the two-source check.  A resumed external-arm cell fails (b) (no_resume_attempt): a W/P/EAGAIN line there holds.
A configuration C cell refused before its lock and started again under a rerun authorization is unit_2 too, yet writes result.json (not result_resume.json): read as a resume (k >= 2),
its record is not found, and the last evaluation holds by m.2 -- to the lead (decision_log 2026-11-20 ~4:52 PM CST (3)).
The tools' own kernel readers (holdout_common.parse_oom, dxg_lines) compare kernel times with an uptime start, as if δ were about 0: with a negative δ they do not see the kernel
lines of the first |δ| seconds of each unit or step.  This monitor scans from SINCE + min(0, δ_lo) and sees them (CA, non-dxg and global OOM lines there hold as usual); the
backstop for what the tools' own classification may miss (decision_log 2026-11-20 ~7:02 PM CST, 70714418; not done by this monitor): at each step's close the lead goes through the
recorded-only W/P/EAGAIN of the monitor's output, and one whose reading falls within |δ_lo| + 1 s of a unit's or a step's start is treated as a hold and goes to BZ.
A condition's episode is sampled at the polls (decision_log 2026-11-20 ~5:15 PM CST (4)): a condition that clears and comes back between two polls, across an evaluation that
raised, or in the gap before a takeover, is read as one episode; in the last two cases the hold is already there.  A one-off eval (final_merged, after_compare) starts every
condition's episode at its own now, so it cannot use an acknowledgement the watcher's events got: a condition still present at the closing eval goes to BZ (~5:15 PM CST (1)).
The record directories' creation time (birth, revision 5; the revision 5 review m2, m4, m5; decision_log 2026-11-21 ~1:42 PM CST):
  m2  a symbolic link is read without following it (AT_SYMLINK_NOFOLLOW): a link made since --start-epoch to a directory without any attempt made before it would read as this
      step's, while iterdir, is_dir and attempts follow the link.  The tools never make such links and the hard rules forbid swapping directories; not handled.
  m4  the kernel stamps inodes with its coarse real-time clock, a few ms behind time.time(): a directory made within about 6 ms after --start-epoch reads as made before it
      (a false hold, the stricter side; the handbook's order now -> before -> watch -> the step puts seconds between).  A wall clock set back across --start-epoch by more than a
      directory's age would read an old directory as this step's; R1's refusal mtimes and disk_cells' attempt mtimes rest on the same clock.
  m5  struct statx is unpacked little-endian ('<I', '<qI'): this machine is x86_64.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import tempfile
import time
import traceback

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import lpba_common as lc  # noqa: E402

hc = lc.hc
SOURCE = dict(evaluator=dict(path='/home/rmcfz/seq_vp_tag_results/holdout_run_v1/step_9_unigradicon_io_eq/dxg_eval_m.py',
                             sha256='4e8928198b94940986d371093675bd0d189df6099c696a9c2b4603602e236955'),
              watcher=dict(path='/home/rmcfz/seq_vp_tag_results/holdout_run_v1/step_9_unigradicon_io_eq/watch_m.sh',
                           sha256='eed0b67470d29346cc3374e8dc3ed986f41d3a0d90ee0d036b7a47371fa1f138'),
              decision_log='docs/decision_log.md lines 489 (BZ 11-18), 498 (the m version), 500 (these two files\' sha256)')
SCHEMA = 'bayesreg.lpba40.dxg_monitor.v1'
MODES = ('lpba40_test', 'lpba40_feas', 'devcheck')
DEVCHECK_DIR = 'lpba40_devcheck_v1'              # = lpdev_common.DEVCHECK_DIR (a test compares)
TOL = 1.0                                        # W/P: a reading within 1 s of a GPU segment's end (BZ 11-17 (a))
DRIFT = .15                                      # while a step runs: δ_lo - DRIFT, δ_hi + DRIFT (the m version's last value, decision_log line 500; both ends: revision 4a)
DELTA_AGREE = .5                                 # revision 4: the tries of one measurement agree when max(lo) - min(hi) <= .5 s (decision_log 2026-11-20 ~6:45 PM CST (i))
POLL_S = 20.
WAIT_START_S = 300.                              # watch: how long to wait for the watched unit or process to become active before the first evaluation
FINAL_SETTLE_S = 5.                              # watch: the wait between the step's end and the last evaluation (review minor 3)
GRACE_S = 15.                                    # a cell in one source only, or a watched unit without its Started line, is an event only after this long while the step is active
SETTLE_S = 3.                                    # a record younger than this is not read as final while the step is active (the m version's 3 s)
EXIT_EVENT, EXIT_ERROR, EXIT_NOT_STARTED = 3, 4, 5
LIMIT = 10*2**20
DISP_LIMIT = 2**30
IDS = re.compile(r'OASIS_OAS1_\d{4}_MR1|\b\d{4}-\d{4}\b|\d{4}_MR1')
KT = re.compile(r'^\[\s*([0-9.]+)\]')
JPREFIX = re.compile(r'^(\[\s*[0-9.]+\])\s+\S+\s+kernel:')          # journal short-monotonic '[ t] host kernel: msg' -> dmesg form '[ t] msg'
JCMD = ('journalctl', '-k', '--no-pager', '-q', '-o', 'short-monotonic')
USER_FIELDS = ('__MONOTONIC_TIMESTAMP', '_BOOT_ID', 'USER_UNIT', '_SYSTEMD_USER_UNIT', 'MESSAGE')      # review minor 9: no other field of an entry is read
UCMD = ('journalctl', '--user', '-b', '0', '--no-pager', '-q', '-o', 'json', f'--output-fields={",".join(USER_FIELDS)}')
CATS = ('W', 'P', 'CA', 'qai', 'eagain', 'dxg_other')
CCATS = CATS + ('non_dxg', 'drop_caches')
W_TEXT = 'misc dxg: dxgk: dxgvmb_send_sync_msg: wait_for_completion failed: fffffe00'          # BZ 11-17's W class (m.1)
P_TEXT = 'misc dxg: dxgk: process_completion_packet: did not find packet to complete'           # BZ 11-17's P class (m.1)
QAI_TEXTS = frozenset({'misc dxg: dxgk: dxgkio_query_adapter_info: Ioctl failed: -22'})         # the qai text (m.1): counted only
EAGAIN_TEXTS = frozenset({'misc dxg: dxgk: dxgvmb_send_sync_msg: vmbus_sendpacket failed: fffffff5',
                          'misc dxg: dxgk: dxgvmb_send_create_process: create_process failed -11'})      # BZ 11-18 question 2 (m.1)
DXG_KEYS = ('dxg', 'wait_for_completion', 'did not find packet', 'query_adapter_info', 'create_process', 'vmbus_sendpacket', 'create_allocation')
DROP = re.compile(r'^mini_init \(\d+\): drop_caches: 1$')          # BZ 11-18 question 3: the only whitelisted non-dxg class
FAIL_LINE = re.compile(r'status=(?!0/)|Failed with result|code=killed|code=dumped|Failed to start')     # the m version's chain-journal test
STOP_KEYS = ('Consumed ', 'Deactivated successfully', 'Failed with result', 'Main process exited')
DRIVER = re.compile(r'.+-queue|x-chain')
KINDS = (('xb', re.compile(r'x-(?P<arm>[a-z0-9_]+)-(?P<f>S\d{2})-(?P<m>S\d{2})')),
         ('method', re.compile(r'he-(?P<f>S\d{2})-(?P<m>S\d{2})')),
         ('configc', re.compile(r'c-(?P<f>S\d{2})-(?P<m>S\d{2})')),
         ('review', re.compile(r'review-(?P<f>S\d{2})-(?P<m>S\d{2})')),
         ('nested', re.compile(r'he-c3?-.+')),
         ('hi', re.compile(r'hi')))
SUPERVISED = {'xb': ('unit', 'unit_resume'), 'method': ('unit', 'unit_resume'), 'configc': ('unit', 'unit_resume'), 'review': ('unit_review', 'unit_review_redo')}
UNIT_PART = {'method': 'he', 'configc': 'c', 'review': 'review'}            # lpba_arms.unit_name's kinds (an external arm's is x-<arm>)
DEFERRED_KINDS = ('devc',)                       # (b)(c) record written after the unit by a step that checks the hold first (lpdev_c.py compare; the lead's ruling I1)
PAIR_DIR = re.compile(r'LPBA40_S\d{2}__LPBA40_S\d{2}')
PROC_CGROUP = Path('/proc/self/cgroup')
OWN_UNIT = re.compile(r'/((?:tag-|seq-vp-)[^/\s]+\.(?:service|scope))(?=/|$)', re.MULTILINE)      # feas_chain.TAG_UNIT widened to seq-vp-* (the machine gates count both)
DEVCHECK_UNITS = {'tag-lp-devb': 'devb', 'tag-lp-devc': 'devc'}
NESTED_DEV_PREFIX = 'tag-he-c-'                  # (c)'s nested configuration C unit in HE's development mode (handbook §4.3, LP-D review minor 9)
OUTPUT_STATUSES = ('hx_certified', 'hx_fallback_configC', 'hx_fallback_identity')      # = he_core.OUTPUT_STATUSES (a test compares)
XB_ARMS = ('ants_s', 'fireants_greedy', 'fireants_syn', 'convexadam_v4seg')            # = xb_core.LPBA_ARMS (a test compares)
ACK_SCHEMA = 'bayesreg.lpba40.dxg_ack.v1'
EVENT_KINDS = ('source', 'cells', 'expected', 'unit_failed', 'driver_failed', 'driver_stop', 'log_stop', 'refusal', 'before', 'status', 'reading', 'arm_stopped', 'ca', 'wp',
               'wp_pending', 'eagain', 'dxg_other', 'non_dxg', 'started_unfinished', 'records_missing', 'not_started', 'ack')
LEAD_KINDS = frozenset({'status', 'driver_stop', 'records_missing', 'started_unfinished'})      # the lead's white list (decision_log 2026-11-20 ~4:38 PM CST (I1)); the rest need BZ's words
LEAD_STATUSES = {'xb': ('method_failure', 'infra_failure', 'unit_limit'),    # line 756 (4), line 759 (I2): method failure, infrastructure failure (xb_core.EXIT); unit_limit: line 757 (B1)
                                                                              # and decision_log 2026-11-20 ~4:52 PM CST (1) (revision 3a)
                                                                # the method arm: none (decision_log 2026-11-20 ~5:05 PM CST (I3), ~5:15 PM CST (3): to BZ)
                 'configc': ('infra_failure', 'unit_limit')}    # line 760: configuration C's infrastructure failure and unit limit (hb_configc result statuses)
LEAD_STOP_CODES = {'chain': (20, 24, 25),                       # line 759 (I2): xb_chain STOP codes 20 infra, 24 resume, 25 rejudgment (xb_chain.STOP); 22, 23 not
                   'configc_queue': (20, 24)}                   # line 760: hb_core.EXIT infra 20, resume 24; 21 (stop, many causes), 22, 25, 26 not.  The method arm's and the
                                                                # review's queues: none (~5:15 PM CST (2): their STOP lines need BZ's words)


# ---- small readers ------------------------------------------------------------------------------------------------------------------------------
def rj(p):
    p = Path(p)
    try:
        if p.stat().st_size > LIMIT:
            return None
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def sha(p, limit=LIMIT):
    p = Path(p)
    try:
        if not p.is_file() or p.stat().st_size > limit:
            return None
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def mtime(p):
    try:
        return Path(p).stat().st_mtime
    except OSError:
        return None


STATX_BTIME, AT_FDCWD, AT_SYMLINK_NOFOLLOW = 0x800, -100, 0x100


def birth(p):
    """The creation time of p (statx(2)'s stx_btime, wall-clock seconds since the epoch, the clock --start-epoch is on), or None when the C library, the kernel or the file system
    does not give it (revision 5).  Unlike mtime and ctime it does not move when something is written into the directory later.  struct statx: stx_mask at 0, stx_btime
    (s64 tv_sec, u32 tv_nsec) at 80."""
    try:
        fn = ctypes.CDLL(None, use_errno=True).statx
    except (OSError, AttributeError):
        return None
    buf = ctypes.create_string_buffer(256)
    try:
        rc = fn(ctypes.c_int(AT_FDCWD), os.fsencode(str(p)), ctypes.c_int(AT_SYMLINK_NOFOLLOW), ctypes.c_uint(STATX_BTIME), buf)
    except (OSError, ValueError, TypeError):
        return None
    if rc != 0 or not struct.unpack_from('<I', buf.raw, 0)[0] & STATX_BTIME:
        return None
    sec, nsec = struct.unpack_from('<qI', buf.raw, 80)
    return sec + nsec*1e-9


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def mask(text):
    """Numbers and hex masked (as the m version's keys); an OASIS-like id masked whole."""
    return IDS.sub('<id>', re.sub(r'\b0x[0-9a-f]+\b|\b\d+\b', '<n>', str(text)))


def msg(line):
    return line.split(']', 1)[-1].strip()


def kind(line):
    """The m version's kind(): W, P, qai, EAGAIN by the whole message; CA; any other line with 'dxg' or a DXG_KEYS word -> dxg_other; else non_dxg."""
    if 'create_allocation' in line and 'fail' in line:
        return 'CA'
    m = msg(line)
    if m == W_TEXT:
        return 'W'
    if m == P_TEXT:
        return 'P'
    if m in QAI_TEXTS:
        return 'qai'
    if m in EAGAIN_TEXTS:
        return 'eagain'
    if any(k in line for k in DXG_KEYS):
        return 'dxg_other'
    return 'non_dxg'


def ckind(line):
    k = kind(line)
    return 'drop_caches' if k == 'non_dxg' and DROP.match(msg(line)) else k


def kernel_lines(text, normalise):
    out = []
    for line in (text or '').splitlines():
        m = KT.match(line)
        if m:
            out.append((float(m.group(1)), JPREFIX.sub(r'\1', line) if normalise else line))
    return out


# ---- the mode: roots, units ---------------------------------------------------------------------------------------------------------------------
def check_mode(mode):
    if mode not in MODES:
        raise SystemExit(f'mode {mode!r}: one of {MODES}')
    return mode


def mode_root(mode):
    check_mode(mode)
    return Path(lc.RESULTS)/DEVCHECK_DIR if mode == 'devcheck' else lc.roots(mode).root


def hold_path(mode):
    """lpba_common's hold file, read at call time: <RESULTS>/lpba40_test_v1/hold for every mode (C26.2-3)."""
    check_mode(mode)
    return lc.roots('lpba40_test' if mode == 'devcheck' else mode).hold


def touch_hold(path):
    """Touch the hold file, making its directory first when it is not there yet (the development checks and the feasibility run before the formal root exists; `touch $T/hold` by
    hand would fail there).  Only the formal root's directory itself may be made here; no entry asks for <RESULTS>/lpba40_test_v1 to be new (its stage/ must be)."""
    path = Path(path)
    lc.guard_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def unit_of(mode, name):
    """(role, kind, info) of a unit name of the mode -- role 'cell', 'driver' or None (not this mode's)."""
    if mode == 'devcheck':
        if name in DEVCHECK_UNITS:
            return 'cell', DEVCHECK_UNITS[name], {}
        if name.startswith(NESTED_DEV_PREFIX):
            return 'cell', 'nested', {}
        return None, None, {}
    prefix = lc.unit_name(mode, '')
    if not name.startswith(prefix) or (mode == 'lpba40_test' and (name.startswith(lc.unit_name('lpba40_feas', '')) or name.startswith('tag-lp-dev'))):
        return None, None, {}
    rest = name[len(prefix):]
    if DRIVER.fullmatch(rest):
        return 'driver', 'driver', {}
    for k, rx in KINDS:
        m = rx.fullmatch(rest)
        if m:
            info = m.groupdict()
            if 'f' in info:
                info['pair'] = lc.pair_name(info['f'], info['m'])
            return 'cell', k, info
    return 'cell', 'other', {}


def expected_unit(mode, k, arm, pair):
    """The unit name of a supervised cell (lpba_arms.unit_name's form): tag-lp[-feas]-<x-arm | he | c | review>-<Sf>-<Sm>."""
    f, m = pair.split('__')
    short = f"{f.split('_')[-1]}-{m.split('_')[-1]}"
    return lc.unit_name(mode, f"{'x-'+arm if k == 'xb' else UNIT_PART[k]}-{short}")


def journal_text(e, key='MESSAGE'):
    v = e.get(key)
    if isinstance(v, list):
        try:
            return bytes(v).decode('utf-8', 'replace')
        except (TypeError, ValueError):
            return ''
    return '' if v is None else str(v)


# ---- source 2: the user journal -----------------------------------------------------------------------------------------------------------------
def unit_events(mode, entries, offset, since):
    """Windows of the mode's units from systemd's own messages (USER_UNIT set), in /proc/uptime's clock; failure lines; STOP lines of the drivers' own output."""
    windows, failures, stops, foreign = [], [], [], set()
    open_ = {}
    for e in sorted(entries, key=lambda x: int(x.get('__MONOTONIC_TIMESTAMP') or 0)):
        try:
            t_us = int(e['__MONOTONIC_TIMESTAMP'])
            t = t_us/1e6 + offset
        except (KeyError, TypeError, ValueError):
            continue
        text = journal_text(e)
        manager_unit = e.get('USER_UNIT')
        own_unit = e.get('_SYSTEMD_USER_UNIT')
        if manager_unit and str(manager_unit).endswith('.service'):
            name = str(manager_unit)[:-len('.service')]
            role, k, info = unit_of(mode, name)
            if role is None:
                if name.startswith(('tag-', 'seq-vp-')):
                    foreign.add(mask(name))
                continue
            if text.startswith('Started '):
                w = dict(unit=name, role=role, kind=k, info=info, start=t, start_us=t_us, end=None, failed=[], started_line=True)
                windows.append(w)
                open_[name] = w
                continue
            is_fail, is_stop = bool(FAIL_LINE.search(text)), any(key in text for key in STOP_KEYS)
            w = open_.get(name)
            if w is None and not (is_fail or is_stop):
                continue                        # e.g. 'Starting ...' before 'Started ...': no window of its own
            if w is None:                       # a stop or failure line of a unit whose start is before the journal's --since: the step's start stands in for it
                w = dict(unit=name, role=role, kind=k, info=info, start=since, start_us=None, end=None, failed=[], started_line=False)
                windows.append(w)
                open_[name] = w
            if is_fail:                         # revision 3 (B1): the failure belongs to the window open when it was written
                w['failed'].append(mask(text)[:160])
                failures.append(dict(unit=mask(name), name=name, role=role, text=mask(text)[:160], raw=text, t=t, t_us=t_us, window_start=w['start'], window_start_us=w['start_us']))
            if is_stop:
                w['end'] = t if w['end'] is None else max(w['end'], t)
        elif own_unit and str(own_unit).endswith('.service') and 'STOP' in text:
            name = str(own_unit)[:-len('.service')]
            role = unit_of(mode, name)[0]
            if role == 'driver':                # revision 3 (I1): the exit code before masking; the line's own time and text for its event id
                m = re.search(r'STOP \((\d+)\)', text)
                stops.append(dict(unit=mask(name), name=name, text=mask(text)[:160], raw=text, t=t, t_us=t_us, code=int(m.group(1)) if m else None))
    return windows, failures, stops, sorted(foreign)


# ---- source 1: the supervisor records on disk -----------------------------------------------------------------------------------------------------
def attempts(base):
    """base, base_2, base_3, ... that exist (hb_core.attempt_dir's names)."""
    out, k = [], 1
    while True:
        d = Path(base) if k == 1 else Path(base).with_name(f'{Path(base).name}_{k}')
        if not d.is_dir():
            break
        out.append(d)
        k += 1
    return out


def disk_cells(mode, since, start_epoch):
    """Every in-unit supervisor directory of the mode root started at or after `since` (the m version's cells: min(unit_start.uptime, unit_end.uptime_start) .. unit_end.uptime); a
    directory whose start cannot be read but that was made since --start-epoch is returned with start None (an event).  One level at a time, no recursive walk."""
    if mode == 'devcheck':
        return []
    root = mode_root(mode)
    specs = [('xb', arm, root/arm) for arm in XB_ARMS] + [('method', 'method', root/'method'), ('configc', 'configc', root/'configc'), ('review', 'method', root/'method')]
    out = []
    for k, arm, base in specs:
        if not base.is_dir():
            continue
        for rec in sorted(p for p in base.iterdir() if p.is_dir() and PAIR_DIR.fullmatch(p.name)):
            rows = []
            for prefix in SUPERVISED[k]:
                for att in attempts(rec/prefix):
                    us, ue = rj(att/'unit_start.json'), rj(att/'unit_end.json')
                    starts = [x for x in ((us or {}).get('uptime'), (ue or {}).get('uptime_start')) if num(x)]
                    unit = str((us or {}).get('unit') or (ue or {}).get('unit') or expected_unit(mode, k, arm, rec.name)).removesuffix('.service')
                    end = ue.get('uptime') if ue and num(ue.get('uptime')) else None
                    rows.append(dict(kind=k, arm=arm, pair=rec.name, rec=rec, att=att, unit=unit, start=min(starts) if starts else None, end=end, mtime=mtime(att),
                                     unreadable=not starts, latest=False))
            if rows:                            # the latest attempt of the cell (I1 (iii): configuration C's unit_<k> with k largest): the largest start, an unreadable one last made
                max(rows, key=lambda r: (r['start'] if r['start'] is not None else float('inf'), r['mtime'] or 0))['latest'] = True
            for r in rows:
                if r['unreadable'] and (r['mtime'] or 0) >= start_epoch:
                    out.append(r)
                elif not r['unreadable'] and r['start'] >= since:
                    out.append(r)
    return sorted(out, key=lambda d: (d['unit'], d['start'] if d['start'] is not None else float('inf')))


def merge_cells(mode, windows, disk):
    """The cells of both sources: a supervised unit's journal windows and supervisor directories paired in their order of start (same unit name); the rest from the journal only."""
    cells = []
    jcells = [w for w in windows if w['role'] == 'cell']
    names = list(dict.fromkeys([w['unit'] for w in jcells] + [d['unit'] for d in disk]))
    for name in names:
        js = [w for w in jcells if w['unit'] == name]
        ds = [d for d in disk if d['unit'] == name]
        role, k, info = unit_of(mode, name)
        if k is None and ds:                    # a supervisor's unit name of another form: the directory's own kind
            k, info = ds[0]['kind'], dict(arm=ds[0]['arm'], pair=ds[0]['pair'])
        if k in SUPERVISED or ds:
            for i in range(max(len(js), len(ds))):
                j = js[i] if i < len(js) else None
                d = ds[i] if i < len(ds) else None
                if d is not None and not info.get('pair'):
                    info = dict(info, pair=d['pair'], arm=d['arm'])
                cells.append(dict(unit=name, kind=k if k in SUPERVISED else (d or {}).get('kind', k), info=dict(info), supervised=True,
                                  jw=None if j is None else (j['start'], j['end']), jw_us=None if j is None else j['start_us'], failed=[] if j is None else j['failed'], disk=d,
                                  dw=None if d is None or d['start'] is None else (d['start'], d['end'])))
        else:
            for j in js:
                cells.append(dict(unit=name, kind=k, info=dict(info), supervised=False, jw=(j['start'], j['end']), jw_us=j['start_us'], failed=j['failed'], disk=None, dw=None))
    seen = {}
    for c in cells:
        seen[c['unit']] = seen.get(c['unit'], 0) + 1
        c['id'] = f"{IDS.sub('<id>', c['unit'])}#{seen[c['unit']]}"
    return cells


def lock_lines(path):
    try:
        lines = Path(path).read_text().splitlines()
        return lines, (json.loads(lines[0]) if lines else {})
    except (OSError, ValueError):
        return [], {}


def supervisor_b(ue):
    ue = ue or {}
    return dict(sup_exit_0=ue.get('exit_code') == 0, no_signal=ue.get('killed_by_signal') is None, no_sigterm=ue.get('sigterm_received') is not True,
                not_killed_by_supervisor=not ue.get('killed_by_supervisor'))


def ok_c(c):
    return all(v is not False and v is not None for v in c.values())


# ---- per kind: the records, the GPU segments, (b), (c), m.2 ---------------------------------------------------------------------------------------
def first_existing(*paths):
    for p in paths:
        doc = rj(p)
        if doc is not None:
            return p, doc
    return None, None


def attempt_index(att):
    """unit -> 1, unit_2 -> 2, ... (hb_core.attempt_dir's names); unit_resume* -> its own index."""
    tail = att.name.rsplit('_', 1)[-1]
    return int(tail) if tail.isdigit() else 1


def attempt_records(mode, k, rec, att, latest):
    """The records of one attempt of a supervised cell, by each tool's own rule (the LP-F revision 1 diff review I1 (iii); revision 2 diff review I2):
      xb       a unit_resume* attempt reads resume/overridden.json, else resume/result.json (xb_core.effective_record(resume=True)), its work <pair>.resume, its lock <lock>.resume;
               the first run's attempts overridden.json, else result.json;
      configc  an attempt is the resume (hb_configc.run_one: attempt_dir(record/'unit'), resumed_paths) when it is unit_<k> with k >= 2, or the latest attempt while
               <lock>.resume (holdout_common.resume_lock_path; written before the run) is there: it reads overridden_resume.json, else result_resume.json -- also while that
               resume still runs (never the first run's result.json) -- and its lock <lock>.resume; the first attempt the first run's;
      method   a unit_resume* attempt reads step4_resume.json (lock <lock>.resume), the first run's step4.json;
      review   unit_end.json only.
    Returns dict(doc_path, doc, over, resumed, lock, work, result)."""
    root = mode_root(mode)
    if k == 'xb':
        resumed = att.name.startswith('unit_resume')
        arm = rec.parent.name
        base = rec/'resume' if resumed else rec
        over_p, over = first_existing(base/'overridden.json')
        res_p = base/'result.json'
        doc_p, doc = (over_p, over) if over is not None else (res_p, rj(res_p))
        lock = root/'locks'/f'xb__{arm}__{rec.name}.lock'
        lock = lock.with_name(lock.name+'.resume') if resumed else lock
        work = root/'work'/arm/(rec.name+'.resume' if resumed else rec.name)
        return dict(doc_path=doc_p, doc=doc, result=rj(res_p), over=over is not None, resumed=resumed, lock=lock, work=work)
    if k == 'configc':
        lock = root/'locks'/f'hb__configc__{rec.name}.lock'
        resume_lock = lock.with_name(lock.name+'.resume')
        resumed = attempt_index(att) >= 2 or (latest and resume_lock.exists())
        names = ('overridden_resume.json', 'result_resume.json') if resumed else ('overridden.json', 'result.json')
        over_p, over = first_existing(rec/names[0])
        doc_p, doc = (over_p, over) if over is not None else (rec/names[1], rj(rec/names[1]))
        return dict(doc_path=doc_p, doc=doc, result=doc, over=over is not None, resumed=resumed, lock=resume_lock if resumed else lock, work=None)
    if k == 'method':
        resumed = att.name.startswith('unit_resume')
        doc_p = rec/('step4_resume.json' if resumed else 'step4.json')
        lock = root/'locks'/f'hb__method__{rec.name}.lock'
        lock = lock.with_name(lock.name+'.resume') if resumed else lock
        return dict(doc_path=doc_p, doc=rj(doc_p), result=None, over=False, resumed=resumed, lock=lock, work=None)
    return dict(doc_path=None, doc={}, result=None, over=False, resumed=False, lock=None, work=None)


def lead_status(k, doc):
    """Whether a status event of this cell is one the lead may acknowledge without BZ's words (revision 2 diff review I1; decision_log 2026-11-20 ~4:38 PM CST (I1); lines 756, 759,
    760): the tool's own status in its LEAD_STATUSES and no spy, guard or wrapper cause recorded."""
    doc = doc or {}
    status = doc.get('status')
    causes = ' '.join(str(x) for x in (doc.get('kind'), (doc.get('failure') or {}).get('kind'), (doc.get('superseded') or {}).get('kind')) if x)
    if any(w in causes for w in ('spy', 'guard', 'wrapper', 'breach', 'input_refused_after_lock', 'input_rejected')) or doc.get('spy_breach'):
        return False                            # revision 3 diff review I2: admission and provenance refusals after the lock go to BZ (prereg §8.3; lines 757, 760)
    return status in LEAD_STATUSES.get(k, ())


def cell_records(mode, cell, commit, now_epoch, active):
    """The cell's GPU segments [(s0, s1)], anchors, settled, (b), (c), state events [dict(kind, text, src, lead)] and the records m.2 asks for that are missing -- by kind."""
    root, k, info, d = mode_root(mode), cell['kind'], cell['info'], cell['disk']
    att = d['att'] if d else None
    out = dict(steps=[], settled=False, b={}, c={}, state=[], missing=[])
    young = []
    ends_known = (cell['jw'] is None or cell['jw'][1] is not None) and (cell['dw'] is None or cell['dw'][1] is not None) and (cell['jw'] is not None or cell['dw'] is not None)
    if k in SUPERVISED:
        pair = info['pair']
        rec = root/(info.get('arm') or 'method')/pair if k == 'xb' else root/('configc' if k == 'configc' else 'method')/pair
        ar = attempt_records(mode, k, rec, att, bool(d and d.get('latest'))) if att else attempt_records(mode, k, rec, rec/'unit', False)
        ue = rj(att/'unit_end.json') if att else None
        reading_path = (att/'unit_reading.json' if k == 'xb' else att.parent/f'{att.name}_reading.json') if att else None
        ur = rj(reading_path) if reading_path else None
        doc = ar['doc']
        if k == 'xb' and doc is None:
            out['missing'] = ['resume/result.json' if ar['resumed'] else 'result.json'] + (['unit_end.json'] if ue is None else [])
        else:
            out['missing'] = [n for n, v in ((ar['doc_path'].name if ar['doc_path'] else None, doc), ('unit_end.json', ue)) if n and v is None]
        young = [mtime(ar['doc_path']) if ar['doc_path'] else None, mtime(reading_path) if reading_path else None]
        out['settled'] = (doc is not None) and ue is not None and ur is not None and ends_known
        doc_d, ue, ur = doc or {}, ue or {}, ur or {}
        lines, l0 = lock_lines(ar['lock']) if ar['lock'] else ([], {})
        src = f"{ar['doc_path']}:{sha(ar['doc_path']) if ar['doc_path'] else None}"
        if k == 'xb':
            res = ar['result'] or {}
            disp = res.get('disp') or {}
            cleanup = res.get('cleanup') or {}
            man = cleanup.get('manifest')
            step = ((res.get('method') or {}).get('steps') or [{}])[0] or {}
            for s in ((res.get('method') or {}).get('steps') or []):
                if isinstance(s, dict) and 'max_gpu_mem_used_mib' in s and num(s.get('uptime_start')) and num(s.get('seconds_wall')):
                    out['steps'].append((s['uptime_start'], s['uptime_start']+s['seconds_wall']))
            out['b'] = dict(status_ok=doc_d.get('status') == 'ok', not_overridden=not ar['over'], ur_status_null=ur.get('status') is None, ur_kind_null=ur.get('kind') is None,
                            systemd_success=ur.get('systemd_result') == 'success', systemd_run_exit_0=ur.get('systemd_run_exit') == 0, no_resume_attempt=not ar['resumed'],
                            **supervisor_b(ue))
            out['c'] = dict(spy_passed=res.get('spy_passed') is True, export_not_suspicious=(res.get('export_check') or {}).get('suspicious') is False,
                            nothing_moved=res.get('move') is None, disp_in_work=disp.get('file') == str(ar['work']/'disp.npz'),
                            disp_sha=bool(disp.get('sha256')) and sha(ar['work']/'disp.npz', DISP_LIMIT) == disp.get('sha256') if out['settled'] and cell.get('needs_c') else None,
                            lock_one_line=len(lines) == 1, lock_code_is_commit=bool(commit) and l0.get('code_commit') == commit,
                            lock_sha_in_record=(res.get('lock') or {}).get('sha256') == sha(ar['lock']),
                            machine_passed=((res.get('checks') or {}).get('machine') or {}).get('passed') is True,
                            manifest_sha_ok=bool(man) and sha(man) == cleanup.get('manifest_sha256'),
                            step_rc_0=step.get('returncode') == 0, no_guard_kill=step.get('memory_guard_killed') is False, no_step_oom=not step.get('kernel_oom_lines'),
                            no_unit_oom=not ue.get('oom_lines'), unit_dxg_0=not ue.get('dxg_lines') and not ur.get('dxg_lines'))
            if ar['over'] or (doc is not None and doc_d.get('status') != 'ok'):
                out['state'].append(dict(kind='status', text=f"{cell['id']}: effective status not ok (status {doc_d.get('status')}, overridden {ar['over']}, resumed {ar['resumed']})",
                                         src=src, lead=lead_status('xb', doc_d)))
            if out['settled'] and ur.get('status') is not None:
                out['state'].append(dict(kind='reading', text=f"{cell['id']}: unit reading status {ur.get('status')}", src=f'{reading_path}:{sha(reading_path)}', lead=False))
        else:
            out['b'] = dict(ur_status_null=ur.get('status') is None, ur_kind_null=ur.get('status_kind') is None, systemd_success=ur.get('systemd_result') == 'success',
                            systemd_run_exit_0=ur.get('systemd_run_exit') == 0, no_resume_attempt=not ar['resumed'], **supervisor_b(ue))
            common = dict(no_unit_oom=not ue.get('oom_lines'), unit_dxg_0=not ue.get('dxg_lines') and not ur.get('dxg_lines'))
            if k == 'method':
                out['c'] = dict(went_on=doc_d.get('went_on') is True, output_status=doc_d.get('status') in OUTPUT_STATUSES, no_spy_breach=not doc_d.get('spy_breach'),
                                lock_one_line=len(lines) == 1, lock_code_is_commit=bool(commit) and l0.get('code') == commit, **common)
                if doc is not None and doc_d.get('went_on') is not True:
                    out['state'].append(dict(kind='status', text=f"{cell['id']}: {ar['doc_path'].name} did not go on (status {doc_d.get('status')})", src=src,
                                             lead=lead_status('method', doc_d)))
            elif k == 'configc':
                breach = bool(rec.is_dir() and any(p.name.startswith('spy_breach') and p.suffix == '.json' for p in rec.iterdir()))
                out['c'] = dict(result_ok=doc_d.get('status') == 'ok', not_overridden=not ar['over'], no_spy_breach=not breach, lock_one_line=len(lines) == 1,
                                lock_code_is_commit=bool(commit) and l0.get('code') == commit, **common)
                if ar['over'] or (doc is not None and doc_d.get('status') != 'ok'):
                    out['state'].append(dict(kind='status', text=f"{cell['id']}: configuration C result not ok (status {doc_d.get('status')}, overridden {ar['over']}, "
                                                                  f"resumed {ar['resumed']})", src=src, lead=lead_status('configc', doc_d) and not breach))
            else:
                out['c'] = dict(tool_check_known=False, **common)
    elif k in ('hi', 'devb', 'devc'):
        path = single_record(mode, k)
        doc = rj(path)
        out['missing'] = [] if doc is not None else [path.name]
        young = [mtime(path)]
        out['settled'] = doc is not None and ends_known
        out['b'] = dict(stop_line=ends_known, no_failure_line=not cell['failed'])
        out['c'] = dict(record_passed=(doc or {}).get('passed') is True)
        if doc is not None and doc.get('passed') is not True:
            out['state'].append(dict(kind='status', text=f"{cell['id']}: {path.name} did not pass", src=f'{path}:{sha(path)}', lead=False))
    else:
        out['settled'] = ends_known
        out['b'] = dict(stop_line=ends_known, no_failure_line=not cell['failed'])
        out['c'] = dict(tool_check_known=False)
    if active and out['settled'] and any(t is not None and t > now_epoch - SETTLE_S for t in young):
        out['settled'] = False                  # the m version: a record written in the last 3 s is not read as final while the step runs
    out['anchors'] = [s1 for _, s1 in out['steps']]
    return out


def single_record(mode, k):
    root = mode_root(mode)
    return {'hi': lambda: lc.roots(mode).hi_out/'hi_record.json', 'devb': lambda: root/'b'/'devcheck_b.json', 'devc': lambda: root/'c'/'devcheck_c.json'}[k]()


def refusal_records(mode):
    """Every file under <mode root>/batches/*/refusals/ (one level each): (path, mtime)."""
    out = []
    batches = mode_root(mode)/'batches'
    if batches.is_dir():
        for b in sorted(batches.iterdir()):
            ref = b/'refusals'
            if ref.is_dir():
                out.extend((p, mtime(p)) for p in sorted(ref.iterdir()))
    return out


# ---- R3: the cells that started before the step (the LP-F revision 1 diff review B1) -----------------------------------------------------------
def before_cells(mode, since, start_epoch):
    """Every cell of the mode root whose latest attempt started before `since`, with its state as each tool reads it (attempt_records), whether its records are complete, and every
    arm_stopped.json and single-step record (hi_record.json, devcheck_b.json, devcheck_c.json) written before --start-epoch.  A row has problem True when its status is not ok, its records
    are not complete, or it is an arm_stopped.json.  Earlier attempts (before `since`) of a cell of this step are listed as reference rows (problem False; revision 2 diff review minor 5).
    A record directory without any attempt is a problem row unless its creation time (birth, statx) is at or after --start-epoch: then it is a cell of this step whose unit has not
    started yet, listed under not_started (no row, not in problems_sha256; revision 5); an unreadable creation time keeps the problem row.
    The document has no time of its own (cells_before_bytes); the sha256 --ack-before names is the problem rows' only (problems_sha256; revision 2 diff review minor 4), so one disposal
    holds until a problem row changes."""
    root = mode_root(mode)
    rows, not_started = [], []
    if mode != 'devcheck':
        specs = [('xb', arm, root/arm) for arm in XB_ARMS] + [('method', 'method', root/'method'), ('configc', 'configc', root/'configc'), ('review', 'method', root/'method')]
        for k, arm, base in specs:
            if not base.is_dir():
                continue
            for rec in sorted(p for p in base.iterdir() if p.is_dir() and PAIR_DIR.fullmatch(p.name)):
                atts = [a for prefix in SUPERVISED[k] for a in attempts(rec/prefix)]
                if not atts:
                    if k == 'review':           # the review's records live in the method's directories: no unit_review* there yet is no cell
                        continue
                    made = birth(rec)
                    if made is not None and made >= start_epoch:
                        # revision 5 (feas-r2, 2026-11-21 13:01:29 CST): two queues make the record directory before the unit's supervisor writes unit/unit_start.json -- the
                        # method queue before its machine gate (hb_method.run_one:157; up to the gate's 15 min), configuration C's after its gate, just before run_unit
                        # (hb_configc.run_one:515); an external arm's is made inside its unit together with unit/ (xb_unit.py:54) -- so a directory made since --start-epoch
                        # (its creation time, not mtime / ctime, which move when anything is written into it) is a cell of this step whose unit has not started: listed, not a
                        # row; the evaluations judge it once its attempt is there (disk_cells); the last evaluation holds on it if it is still listed (revision 5a)
                        not_started.append(dict(kind=k, record=str(rec), birth=made))
                        continue
                    # a record directory without any attempt made before the step (e.g. refused before its lock), or whose creation time cannot be read: its state is not known
                    rows.append(dict(kind=k, record=str(rec), attempt=None, status=None, complete=False, missing=['unit*'], problem=True))
                    continue
                starts = []
                for a in atts:
                    us, ue = rj(a/'unit_start.json'), rj(a/'unit_end.json')
                    s = [x for x in ((us or {}).get('uptime'), (ue or {}).get('uptime_start')) if num(x)]
                    starts.append((min(s) if s else None, a))
                known = [s for s, _ in starts if s is not None]
                if known and max(known) >= since:
                    for s, a in starts:         # a cell of this step: its earlier attempts before --since as reference rows
                        if s is not None and s < since:
                            rows.append(dict(kind=k, record=str(rec), attempt=a.name, status='earlier attempt of a cell of this step', complete=True, missing=[], problem=False))
                    continue
                if not known and min((mtime(a) or 0) for a in atts) >= start_epoch:
                    continue                    # made since the step started, start unreadable: an event of the evaluations (disk_cells)
                latest = max(starts, key=lambda x: (x[0] if x[0] is not None else -1., x[1].name))[1]
                ar = attempt_records(mode, k, rec, latest, True)
                ue = rj(latest/'unit_end.json')
                reading = rj(latest/'unit_reading.json') if k == 'xb' else rj(latest.parent/f'{latest.name}_reading.json')
                doc = ar['doc']
                if k == 'xb':
                    status = (doc or {}).get('status')
                    ok = status == 'ok' and not ar['over'] and (reading or {}).get('status') is None
                elif k == 'method':
                    status = (doc or {}).get('status')
                    ok = (doc or {}).get('went_on') is True
                elif k == 'configc':
                    status = (doc or {}).get('status')
                    ok = status == 'ok' and not ar['over']
                else:
                    status = (reading or {}).get('status')
                    ok = reading is not None and status is None
                missing = [n for n, v in ((ar['doc_path'].name if ar['doc_path'] else None, doc), ('unit_end.json', ue)) if n and v is None]
                reading_path = latest/'unit_reading.json' if k == 'xb' else latest.parent/f'{latest.name}_reading.json'
                rows.append(dict(kind=k, record=str(rec), attempt=latest.name, status=status, resumed=ar['resumed'], complete=not missing, missing=missing,
                                 record_sha256=sha(ar['doc_path']) if ar['doc_path'] else None, reading_sha256=sha(reading_path), unit_end_sha256=sha(latest/'unit_end.json'),
                                 problem=not ok or bool(missing)))
        for arm in XB_ARMS:
            stop = root/arm/'arm_stopped.json'
            if stop.exists() and (mtime(stop) or 0) < start_epoch:
                rows.append(dict(kind='arm_stopped', record=str(stop), attempt=None, status='arm_stopped', complete=True, missing=[], record_sha256=sha(stop), problem=True))
    for k in (('devb', 'devc') if mode == 'devcheck' else ('hi',)):
        path = single_record(mode, k)
        if path.exists() and (mtime(path) or 0) < start_epoch:
            doc = rj(path)
            rows.append(dict(kind=k, record=str(path), attempt=None, status=(doc or {}).get('passed'), complete=doc is not None, missing=[] if doc is not None else [path.name],
                             record_sha256=sha(path), problem=(doc or {}).get('passed') is not True))
    problems = [r for r in rows if r['problem']]
    return dict(schema=SCHEMA+'.cells_before', mode=mode, since=since, start_epoch=start_epoch, rule='R3 (the LP-F revision 1 diff review B1; decision_log 2026-11-20 ~4:13 PM CST)',
                rows=rows, problems=len(problems), problems_sha256=problems_sha256(mode, problems), not_started=not_started)


def problems_sha256(mode, problems):
    """The sha256 --ack-before names: the mode and the problem rows only (canonical JSON), not --since or the ok rows."""
    return hashlib.sha256(json.dumps(dict(mode=mode, problems=problems), sort_keys=True, default=str).encode('utf-8')).hexdigest()


def cells_before_bytes(doc):
    return (json.dumps(doc, indent=1, sort_keys=True, default=str)+'\n').encode('utf-8')


# ---- acknowledgements (the LP-F revision 1 diff review I1 (i); revision 2 diff review B1, I1) ------------------------------------------------------
def event_id(kind_, cell, src, win):
    """A stable id of one event: its kind, its cell, its source (a kernel line's time and whole text; a journal line's time and text; a record's path and sha256; a condition's name) and
    its window's starts (revision 2 diff review B1).  16 hex digits of a sha256."""
    return hashlib.sha256(json.dumps([kind_, cell, src, win], sort_keys=True, default=str).encode('utf-8')).hexdigest()[:16]


def load_ack(path):
    """(entries, problem): the ack file (JSON {"schema": ACK_SCHEMA, "entries": [{"event_id", "cell", "kind", "decision_log", "bz_words"}...]}); no file -> ([], None)."""
    if path is None or not Path(path).exists():
        return [], None
    try:
        doc = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return [], f'ack file {mask(path)} does not parse ({type(exc).__name__})'
    if not isinstance(doc, dict) or doc.get('schema') != ACK_SCHEMA or not isinstance(doc.get('entries'), list):
        return [], f'ack file {mask(path)}: schema {ACK_SCHEMA} with a list "entries" expected'
    return doc['entries'], None


def judge_acks(entries, events, kernel_in, kernel_any):
    """Accepted and rejected entries.  An entry names one event by its event_id (and repeats its cell and kind, which must match) and the decision_log line (not empty); it releases that
    event only.  Without BZ's words (bz_words empty) it is accepted only when the event's own `lead` flag is set -- the white list LEAD_KINDS with each kind's condition (status:
    LEAD_STATUSES; driver_stop: LEAD_STOP_CODES; records_missing, started_unfinished) -- and the cell has no kernel event in its windows (driver_stop: none in the whole scan).
    The code cannot check that bz_words are BZ's own words nor that the decision_log line records the tool's judgement: the handbook has the line quote them and each step's independent
    review compares them (decision_log 2026-11-20 ~4:38 PM CST (I1))."""
    by_id = {e['id']: e for e in events}
    accepted, rejected = [], []
    for a in entries:
        if not isinstance(a, dict):
            rejected.append(dict(entry=mask(a)[:200], why='not an object'))
            continue
        eid, cell, k, where, words = a.get('event_id'), a.get('cell'), a.get('kind'), a.get('decision_log'), a.get('bz_words') or ''
        entry = dict(event_id=eid, cell=cell, kind=k, decision_log=where, bz_words=words)
        ev = by_id.get(eid)
        why = None
        if not isinstance(eid, str) or not isinstance(where, str) or not where.strip() or not isinstance(words, str):
            why = 'event_id, decision_log (not empty) and bz_words (a string) needed'
        elif ev is None:
            why = f'event_id {eid}: no such event in this evaluation (it may come back once it occurs; an id names one event)'
        elif ev['kind'] != k or (ev['cell'] if ev['cell'] is not None else '-') != cell:
            why = f"event_id {eid} is a {ev['kind']} event of {ev['cell'] or '-'}, not {k} of {cell}"
        elif not words.strip():
            if not ev.get('lead'):
                why = f"event {eid} ({ev['kind']}) is not on the lead's white list (LEAD_KINDS with their conditions); BZ's words needed"
            elif ev['kind'] == 'driver_stop' and kernel_any:
                why = 'driver_stop: a kernel event in the scan; BZ\'s words needed'
            elif ev['kind'] != 'driver_stop' and kernel_in.get(ev['cell'], True):
                why = f"cell {ev['cell']}: a kernel event in its windows (or the cell is not known); BZ's words needed"
        if why:
            rejected.append(dict(entry=entry, why=why))
        else:
            accepted.append(entry)
    return accepted, rejected


# ---- one evaluation ---------------------------------------------------------------------------------------------------------------------------
def evaluate(mode, *, since, deltas, active, start_epoch, commit, kernel_text, dmesg_text, user_entries, now_uptime, now_epoch, boot_offset, logs=(), reread=None,
             may_hold=True, user_journal_problem=None, watched=(), defer=(), expect=(), ack_path=None, ack_before=None, conditions=None):
    """One evaluation (the m version's main()): returns (doc, events).  kernel_text None = journalctl -k failed; dmesg_text None = dmesg failed; reread() gives the kernel journal again
    (the consistency re-reads; None: no re-read); user_journal_problem: why the user journal could not be read (journal_alive), or None; watched: units that have been active for
    GRACE_S and must show a Started line; defer: kinds whose open items at the last evaluation go to doc['deferred'] instead of the events (I1); expect: units that must have a window
    in the user journal at the last evaluation (I2 of the revision 1 diff review); ack_path: the ack file (re-read here); ack_before: the problems_sha256 of cells_before that the lead
    acknowledged (R3).  Every event has a stable id (event_id); the events returned are the ones not acknowledged."""
    check_mode(mode)
    lo_d, hi_d = deltas
    if not (num(lo_d) and num(hi_d) and math.isfinite(lo_d) and math.isfinite(hi_d) and lo_d <= hi_d) or not (num(since) and math.isfinite(since)):
        raise ValueError(f'δ {deltas}, since {since}: finite, δ_lo <= δ_hi (signed, revision 4)')
    offsets = (lo_d, hi_d)
    kt_from = since + min(0., lo_d)             # revision 4: the kernel time the scan starts at -- SINCE when δ_lo >= 0 (the m version), |δ_lo| earlier when δ is negative
    events = []

    prior = dict(conditions or {})
    now_cond = {}

    def add(category, kind_, text, cell=None, src=None, win=None, lead=False, cond=False):
        src = src if src is not None else text
        if cond:                                # revision 3 diff review I1: a condition's id carries the start of this episode (the first poll it held, carried by the watcher)
            key = json.dumps([kind_, cell, src], sort_keys=True, default=str)
            now_cond[key] = prior.get(key, now_uptime)
            src = [src, now_cond[key]]
        events.append(dict(category=category, kind=kind_, text=text, cell=cell, lead=lead, id=event_id(kind_, cell, src, win)))

    dm = kernel_lines(dmesg_text, False) if dmesg_text is not None else None
    dm_first = dm[0][0] if dm else None
    dm_last = dm[-1][0] if dm else None
    kj = kernel_lines(kernel_text, True) if kernel_text is not None else None
    consistency = None
    if dm and kj:
        lo, hi = max(kt_from, dm_first), dm_last
        dc = {c: 0 for c in CCATS}
        for t, line in dm:
            if lo <= t <= hi:
                dc[ckind(line)] += 1
        tries = 0
        while True:
            jc = {c: 0 for c in CCATS}
            for t, line in kj:
                if lo <= t <= hi:
                    jc[ckind(line)] += 1
            short = [c for c in CCATS if jc[c] < dc[c]]
            if not short or tries >= 3 or reread is None:
                break
            time.sleep(1)
            tries += 1
            again = reread()
            if not again:
                break
            kj = kernel_lines(again, True)
        consistency = dict(window=[lo, hi], dmesg=dc, journal=jc, short=short, rereads=tries)
    deferred = []
    if not kj:
        add('dxg', 'source', f'kernel journal not readable or empty (lines {None if kj is None else 0})', src='kernel_journal_unreadable', cond=True)
    elif kj[0][0] > kt_from:
        add('dxg', 'source', f'kernel journal earliest line {kj[0][0]} > the scan start {kt_from} (the step start uptime {since}): lines after the start may be missing',
            src=f'kernel_journal_first:{kj[0][0]}')
    if not dm:
        add('dxg', 'source', f'dmesg not readable (read {"failed" if dm is None else "gave no kernel line"}); the journal >= dmesg check could not run', src='dmesg_unreadable', cond=True)
    if consistency is not None and consistency['short']:
        add('dxg', 'source', f"kernel journal has fewer lines than dmesg in kernel {consistency['window']} for {consistency['short']} after {consistency['rereads']} re-read(s)",
            src=f"journal_short:{consistency['short']}", cond=True)

    counts = dict(W=0, P=0, CA=0, qai=0, eagain=0, dxg_other=0, non_dxg=0)
    wp, eag, others, nond, non_dxg_texts, occurrence = [], [], [], [], {}, {}
    for t, line in kj or []:
        if t < kt_from:
            continue
        k = kind(line)
        counts[k] += 1
        text_ = msg(line)
        occurrence[(t, text_)] = occurrence.get((t, text_), 0) + 1
        raw = [text_, occurrence[(t, text_)]]   # revision 3 diff review minor 1: two lines of the same microsecond and text are two events (their order of occurrence)
        if k in ('W', 'P', 'CA'):
            wp.append((k, t, raw))
        elif k == 'eagain':
            eag.append((t, raw))
        elif k == 'dxg_other':
            others.append((t, mask(text_)[:100], raw))
        elif k == 'non_dxg':
            key = mask(text_)[:100]
            non_dxg_texts[key] = non_dxg_texts.get(key, 0) + 1
            nond.append((t, key, bool(DROP.match(text_)), raw))

    # the cells: both sources (B1)
    if user_journal_problem:
        add('cells', 'source', f'user journal not readable: {mask(user_journal_problem)[:300]}', src='user_journal_unreadable', cond=True)
    windows, failures, stops, foreign = unit_events(mode, user_entries, boot_offset, since)
    for name in watched:
        if not any(w['unit'] == name and w.get('started_line') for w in windows):
            add('cells', 'source', f'watched unit {mask(name)} active for {GRACE_S:.0f} s without its Started line in the user journal (entries read {len(user_entries)})',
                src=f'watched_without_started:{name}', cond=True)
    if not active:
        if not user_entries:
            add('cells', 'source', 'last evaluation: the user journal gave 0 entries since the step start (I2 of the revision 1 diff review)', src='user_entries_0', cond=True)
        for name in expect:
            if not any(w['unit'] == name and w.get('started_line') for w in windows):
                add('cells', 'expected', f'last evaluation: expected unit {mask(name)} has no window in the user journal (I2 of the revision 1 diff review)', cell=mask(name),
                    src=f'expected:{name}', cond=True)
    disk = disk_cells(mode, since, start_epoch)
    cells = merge_cells(mode, windows, disk)

    def starts(c):
        return [(c['jw'] or (None,))[0], (c['dw'] or (None,))[0]]

    def id_starts(c):
        """The window starts an event id takes: the journal's own integer microseconds (CLOCK_MONOTONIC, before the boot offset, which jitters) and the supervisor's recorded uptime."""
        return [c.get('jw_us'), (c['dw'] or (None,))[0]]

    for c in cells:
        known = [x for x in starts(c) if x is not None]
        if not known and c['disk'] is not None and c['disk'].get('mtime') is not None:
            known = [now_uptime - (now_epoch - c['disk']['mtime'])]      # minor 4: a directory without a readable start waits its grace from its mtime
        due = not active or not known or now_uptime - max(known) >= GRACE_S
        if c['disk'] is not None and c['disk'].get('unreadable') and due:
            add('cells', 'cells', f"{c['id']}: supervisor directory {mask(c['disk']['att'].name)} made since the step start without a readable start (unit_start.json / unit_end.json)",
                cell=c['id'], src=f"unreadable:{c['disk']['att']}", win=id_starts(c))
        if c['supervised'] and (c['jw'] is None or c['disk'] is None) and due:
            add('cells', 'cells', f"{c['id']}: seen only in {'the user journal' if c['disk'] is None else 'the supervisor records'} (the two sources disagree on the cells)", cell=c['id'],
                src=f"only_in:{'journal' if c['disk'] is None else 'records'}", win=id_starts(c))
    window_cell = {(c['unit'], c['jw'][0]): c['id'] for c in cells if c['jw'] is not None}
    for f in failures:                          # revision 3 (B1): the window open when the failure line was written; never the unit's last window
        if f['role'] == 'driver':
            add('state', 'driver_failed', f"unit {f['unit']}: failure line in the user journal: {f['text']}", cell=f['unit'], src=[f['t_us'], f['raw']])
        else:
            cid = window_cell.get((f['name'], f['window_start']), f"{f['unit']}@{f['window_start']}")
            add('state', 'unit_failed', f"unit {f['unit']}: failure line in the user journal ({cid}): {f['text']}", cell=cid, src=[f['t_us'], f['raw']], win=f['window_start_us'])
    for s in stops:
        driver = 'chain' if s['name'].endswith('x-chain') else 'configc_queue' if s['name'].endswith('-c-queue') else None
        allowed = s['code'] in LEAD_STOP_CODES.get(driver, ())
        add('state', 'driver_stop', f"driver {s['unit']}: STOP line (exit code {s['code']}) in its journal output: {s['text']}", cell=s['unit'], src=[s['t_us'], s['raw']], lead=allowed)
    for log in logs:
        try:
            lines = Path(log).read_text(errors='replace').splitlines()
        except OSError:
            add('state', 'log_stop', f'log {mask(log)}: not readable', cell=mask(log), src='log_unreadable', cond=True)
            continue
        for i, x in enumerate(lines, 1):        # revision 3 (B1): one event per STOP line, its line number and text in the id
            if 'STOP' in x or 'stopped at' in x:
                add('state', 'log_stop', f'log {mask(log)} line {i}: STOP: {mask(x)[:160]}', cell=mask(log), src=[i, x])
    for p, m in refusal_records(mode):
        if (m or 0) >= start_epoch:             # revision 3 (B1): one event per refusal record, its path and sha256 in the id
            add('state', 'refusal', f'refusal record {mask(p.name)} under batches/{mask(p.parent.parent.name)}/refusals/ since the step start', src=[str(p), sha(p)])
    refusals = sum(1 for _, m in refusal_records(mode) if (m or 0) >= start_epoch)
    before = before_cells(mode, since, start_epoch)
    if before['problems'] and ack_before != before['problems_sha256']:
        add('state', 'before', f"{before['problems']} cell(s) that started before the step are not ok, not complete or arm_stopped (problems sha256 {before['problems_sha256']}; R3: "
                               f"--ack-before {'not given' if not ack_before else 'given another sha256'})", src=before['problems_sha256'])

    def end_of(e):
        return e if e is not None else (now_uptime + 5 if active else None)

    def scan_end(e):
        """revision 3 (B2): a window without an end reaches the end of the scan (now + 5), also at the last evaluation, for the kernel events of its cell and the non-dxg lines."""
        return e if e is not None else now_uptime + 5

    def union(c):
        return [(w[0], scan_end(w[1])) for w in (c['jw'], c['dw']) if w is not None and w[0] is not None]

    def strict(c):
        if c['supervised']:
            if c['jw'] is None or c['dw'] is None:
                return None
            e1, e2 = end_of(c['jw'][1]), end_of(c['dw'][1])
            return None if e1 is None or e2 is None else (max(c['jw'][0], c['dw'][0]), min(e1, e2))
        return (c['jw'][0], end_of(c['jw'][1])) if c['jw'][1] is not None or active else None

    known_times = [x for c in cells for w in (c['jw'], c['dw']) if w is not None for x in w if x is not None]
    last_known = max(known_times) if known_times else None

    def after_last(u):
        return active and (last_known is None or u > last_known)

    def place(u):
        """W/P/EAGAIN: the innermost cell whose strict window (both sources) holds u; 'pending' after the last known time while active; else 'outside'."""
        hit = None
        for c in cells:
            w = strict(c)
            if w is not None and w[1] is not None and w[0] <= u <= w[1] and (hit is None or w[0] >= strict(hit)[0]):
                hit = c
        if hit is not None:
            return hit['id']
        return 'pending' if after_last(u) else 'outside'

    def cells_holding(u):
        return [c['id'] for c in cells if any(s <= u <= e for s, e in union(c))]

    # the kernel events of each cell (either reading in either window, an open window reaching the scan's end): the lead's acknowledgements need none (lines 756, 759, 760)
    kernel_in = {c['id']: False for c in cells}
    kernel_times = [t for _, t, _ in wp] + [t for t, _ in eag] + [t for t, _, _ in others] + [t for t, _, d, _ in nond if not d]
    for t in kernel_times:
        for o in offsets:
            for cid in cells_holding(t-o):
                kernel_in[cid] = True
    kernel_any = bool(kernel_times)

    by_id = {c['id']: c for c in cells}
    win_of = {c['id']: id_starts(c) for c in cells}
    per_cell, pending = {}, 0
    for k, t, raw in wp:
        pl = [place(t-o) for o in offsets]
        if k == 'CA':
            cid = pl[0] if pl[0] in by_id else None
            add('dxg', 'ca', f'CA line at kernel {t:.6f} (readings placed {pl[0]} / {pl[1]})', cell=cid, src=[t, raw], win=win_of.get(cid))
            continue
        if 'pending' in pl:
            pending += 1
            continue
        if pl[0] != pl[1] or pl[0] == 'outside':
            add('dxg', 'wp', f'{k} line at kernel {t:.6f}: readings -{offsets[0]} / -{offsets[1]} placed {pl[0]} / {pl[1]} (outside every cell window or split)', src=[t, raw])
            continue
        per_cell.setdefault(pl[0], []).append((k, t, raw))
    eag_cell, eagain = {}, dict(judged=0, pending=0, recorded=0, failed=0)
    for t, raw in eag:
        pl = [place(t-o) for o in offsets]
        if 'pending' in pl:
            eagain['pending'] += 1
            continue
        if pl[0] != pl[1] or pl[0] == 'outside':
            eagain['judged'] += 1
            eagain['failed'] += 1
            add('dxg', 'eagain', f'EAGAIN-class line at kernel {t:.6f}: readings -{offsets[0]} / -{offsets[1]} placed {pl[0]} / {pl[1]} (outside every cell window or split)', src=[t, raw])
            continue
        eag_cell.setdefault(pl[0], []).append((t, raw))
    for cid in set(per_cell) | set(eag_cell):
        by_id[cid]['needs_c'] = True

    recs = {c['id']: cell_records(mode, c, commit, now_epoch, active) for c in cells}
    for c in cells:
        for st in recs[c['id']]['state']:
            add('state', st['kind'], st['text'], cell=c['id'], src=st['src'], win=win_of[c['id']], lead=st['lead'])
    for arm in sorted({c['info']['arm'] for c in cells if c['kind'] == 'xb'}):
        stop = mode_root(mode)/arm/'arm_stopped.json'
        if stop.exists():
            add('state', 'arm_stopped', f'{arm}: arm_stopped.json present (P19)', cell=arm, src=[str(stop), sha(stop)])

    def defer_it(cid):
        return not active and by_id[cid]['kind'] in defer and not recs[cid]['settled']

    judged, eagain_cells, wp_deferred = {}, {}, 0
    for cid, lines in sorted(per_cell.items()):
        rc = recs[cid]
        if not rc['settled']:
            if defer_it(cid):
                wp_deferred += len(lines)
                deferred.append(f'{cid}: {len(lines)} W/P line(s) pending at the last evaluation ({", ".join(f"{k} {t:.6f}" for k, t, _ in lines)})')
            elif not active:
                for k, t, raw in lines:
                    add('dxg', 'wp_pending', f'last evaluation (the step no longer active): {cid}: {k} line at kernel {t:.6f}: W/P line(s) still pending', cell=cid, src=[t, raw],
                        win=win_of[cid])
            else:
                pending += len(lines)
            continue
        offs = {str(o): [] for o in offsets}
        line_a = []
        for k, t, raw in lines:
            best = next(([t-o-a for o in offsets] for a in rc['anchors'] if all(abs(t-o-a) <= TOL for o in offsets)), None)
            line_a.append(best is not None)
            if best is None:
                best = [min((t-o-a for a in rc['anchors']), key=abs) if rc['anchors'] else None for o in offsets]
            for o, dd in zip(offsets, best):
                offs[str(o)].append(None if dd is None else round(dd, 3))
        ok_a = bool(rc['anchors']) and all(line_a)
        ok_bc = all(rc['b'].values()) and ok_c(rc['c'])
        judged[cid] = dict(kind=by_id[cid]['kind'], W=sum(k == 'W' for k, _, _ in lines), P=sum(k == 'P' for k, _, _ in lines), n_gpu_anchors=len(rc['anchors']), a=ok_a,
                           b_failed=[n for n, v in rc['b'].items() if not v], c_failed=[n for n, v in rc['c'].items() if v is False or v is None], offsets=offs, passes=ok_a and ok_bc)
        for (k, t, raw), la in zip(lines, line_a):
            if not (la and ok_bc):              # revision 3 (B1): one event per line
                add('dxg', 'wp', f"{cid}: {k} line at kernel {t:.6f}: W/P lines and the rule fails: a {la} (GPU segments {len(rc['anchors'])}), b failed {judged[cid]['b_failed']}, "
                                 f"c failed {judged[cid]['c_failed']}", cell=cid, src=[t, raw], win=win_of[cid])
    for cid, ts in sorted(eag_cell.items()):
        rc = recs[cid]
        if not rc['settled']:
            if active:
                eagain['pending'] += len(ts)
                continue
            if defer_it(cid):
                deferred.append(f'{cid}: {len(ts)} EAGAIN-class line(s) pending at the last evaluation ({", ".join(f"{t:.6f}" for t, _ in ts)})')
                continue
            eagain['judged'] += len(ts)
            eagain['failed'] += len(ts)
            for t, raw in ts:
                add('dxg', 'eagain', f'{cid}: EAGAIN-class line at kernel {t:.6f} in a cell not settled while the step is no longer active ((b) cannot hold)', cell=cid, src=[t, raw],
                    win=win_of[cid])
            continue
        in_step = [any(all(s0 <= t-o <= s1 for o in offsets) for s0, s1 in rc['steps']) for t, _ in ts]
        ok_bc = all(rc['b'].values()) and ok_c(rc['c'])
        ok = bool(rc['steps']) and all(in_step) and ok_bc
        eagain['judged'] += len(ts)
        eagain['recorded' if ok else 'failed'] += len(ts)
        eagain_cells[cid] = dict(kind=by_id[cid]['kind'], n=len(ts), n_gpu_steps=len(rc['steps']), in_gpu_step=bool(rc['steps']) and all(in_step),
                                 b_failed=[n for n, v in rc['b'].items() if not v], c_failed=[n for n, v in rc['c'].items() if v is False or v is None], recorded_only=ok)
        for (t, raw), ins in zip(ts, in_step):
            if not (ins and ok_bc and rc['steps']):
                add('dxg', 'eagain', f"{cid}: EAGAIN-class line at kernel {t:.6f}: EAGAIN-class line(s) and the rule fails: in a GPU step {ins and bool(rc['steps'])} "
                                     f"(GPU segments {len(rc['steps'])}), b failed {eagain_cells[cid]['b_failed']}, c failed {eagain_cells[cid]['c_failed']}", cell=cid, src=[t, raw],
                    win=win_of[cid])
    for t, key, raw in others:
        add('dxg', 'dxg_other', f'other dxg line at kernel {t:.6f} (not W/P/CA/qai/EAGAIN class): {key}', src=[t, raw])
    nd = dict(in_window=0, outside=0, pending=0, drop_caches=0, drop_caches_in_window=0)
    for t, key, is_drop, raw in nond:
        rd = [t-o for o in offsets]
        holders = sorted({cid for u in rd for cid in cells_holding(u)})
        if is_drop:
            nd['drop_caches'] += 1
            nd['drop_caches_in_window'] += bool(holders)
            continue
        if holders:
            nd['in_window'] += 1
            add('kernel', 'non_dxg', f'non-dxg kernel line at kernel {t:.6f} in a cell window (readings {rd[0]:.3f} / {rd[1]:.3f}; {holders[0]}): {key}', cell=holders[0], src=[t, raw],
                win=win_of[holders[0]])
        elif any(after_last(u) for u in rd):
            nd['pending'] += 1
        else:
            nd['outside'] += 1
    started_unfinished = [c['id'] for c in cells if any(w is not None and w[1] is None for w in (c['jw'], c['dw']))]
    records_missing = {c['id']: recs[c['id']]['missing'] for c in cells if recs[c['id']]['missing']}
    last_eval_open = None
    if not active:
        last_eval_open = dict(wp_pending=pending, eagain_pending=eagain['pending'], started_unfinished=started_unfinished, records_missing=records_missing,
                              not_started=[mask(x['record']) for x in before['not_started']])
        if pending:
            add('dxg', 'wp_pending', f'last evaluation (the step no longer active): {pending} W/P line(s) still pending', src='wp_pending_after_last', cond=True)
        if eagain['pending']:
            add('dxg', 'eagain', f"last evaluation (the step no longer active): {eagain['pending']} EAGAIN-class line(s) still pending", src='eagain_pending_after_last', cond=True)
        for cid in started_unfinished:
            add('state', 'started_unfinished', f'last evaluation (the step no longer active): {cid} started without an end (user journal or supervisor records)', cell=cid,
                src='started_unfinished', win=win_of[cid], lead=True)
        for cid, miss in records_missing.items():
            if defer_it(cid):
                deferred.append(f'{cid}: {miss} not written at the last evaluation (the step after the unit writes it)')
            else:
                add('state', 'records_missing', f'last evaluation (the step no longer active): {cid} started and {miss} not written (m.2)', cell=cid, src=['records_missing', miss],
                    win=win_of[cid], lead=True)
        for x in before['not_started']:         # revision 5a (B1 of the revision 5 review): a record directory of this step whose unit never started -- e.g. the method queue ended
            # by SIGTERM in its machine gate: no STOP line, no failure line, no refusal; not on the lead's white list (BZ's words)
            add('state', 'not_started', f"last evaluation (the step no longer active): record directory {mask(x['record'])} made since the step start (creation time "
                                        f"{x['birth']:.6f}) and no unit attempt in it: its unit never started", cell=mask(f"{x['kind']}/{Path(x['record']).name}"),
                src=['not_started', x['record'], x['birth']])
    # the acknowledgements (I1 (i); revision 2 diff review B1): an accepted entry takes its one event off the hold
    entries, ack_problem = load_ack(ack_path)
    if ack_problem:
        add('state', 'ack', ack_problem, src=f'ack:{sha(ack_path)}', cond=True)
    accepted, rejected = judge_acks(entries, events, kernel_in, kernel_any)
    released = {a['event_id'] for a in accepted}
    acked = [e for e in events if e['id'] in released]
    live = [e for e in events if e['id'] not in released]
    hold = hold_path(mode)
    held = False
    if may_hold and live and not hold.exists():
        touch_hold(hold)
        held = True
    doc = dict(schema=SCHEMA, mode=mode, uptime_now=now_uptime, since=since, offsets=list(offsets), active=active, commit=commit, source='journalctl -k -o short-monotonic',
               kernel_scan_from=kt_from, kernel_lines=None if kj is None else len(kj), kernel_first=kj[0][0] if kj else None, kernel_last=kj[-1][0] if kj else None, dmesg_lines=None if dm is None else len(dm),
               dmesg_first=dm_first, dmesg_last=dm_last, consistency=consistency, counts_since=counts, wp_pending=pending, wp_deferred=wp_deferred,
               wp_cells={c: len(v) for c, v in per_cell.items()}, judged=judged, eagain=eagain, eagain_cells=eagain_cells, non_dxg_judge=nd, non_dxg_texts=non_dxg_texts,
               dxg_other_texts=[k for _, k, _ in others], user_entries=len(user_entries), user_journal_problem=None if not user_journal_problem else mask(user_journal_problem)[:300],
               cells=[dict(id=c['id'], kind=c['kind'], journal=c['jw'], records=c['dw'], settled=recs[c['id']]['settled'], n_gpu_steps=len(recs[c['id']]['steps']),
                           missing=recs[c['id']]['missing'], kernel_events=kernel_in[c['id']]) for c in cells],
               drivers=sorted({IDS.sub('<id>', w['unit']) for w in windows if w['role'] == 'driver'}), foreign_units=foreign, refusals_since_start=refusals,
               started_unfinished=started_unfinished, records_missing=records_missing, last_eval_open=last_eval_open, deferred=deferred, expected=[mask(x) for x in expect],
               cells_before=dict(problems_sha256=before['problems_sha256'], rows=len(before['rows']), problems=before['problems'],
                                 acknowledged=bool(before['problems']) and ack_before == before['problems_sha256'], not_started=len(before['not_started'])),
               ack=dict(path=None if ack_path is None else mask(ack_path), sha256=sha(ack_path) if ack_path else None, accepted=accepted, rejected=rejected),
               acked=[dict(id=e['id'], kind=e['kind'], cell=e['cell'], text=e['text']) for e in acked],
               event_items=[dict(id=e['id'], kind=e['kind'], cell=e['cell'], lead=e['lead'], text=e['text']) for e in live],
               events=[f"{e['category']}: {e['text']}" for e in live], hold=str(hold), hold_touched_now=held, hold_present=hold.exists(), may_hold=may_hold,
               conditions=now_cond)
    if IDS.search(json.dumps(doc, sort_keys=True, default=str)):
        raise ValueError('an id-like token in the output')
    return doc, live


# ---- the machine: the real sources --------------------------------------------------------------------------------------------------------------
def uptime():
    return float(Path('/proc/uptime').read_text().split()[0])


def boot_offset():
    """CLOCK_BOOTTIME - CLOCK_MONOTONIC: the user journal's monotonic times moved to /proc/uptime's clock."""
    return time.clock_gettime(time.CLOCK_BOOTTIME) - time.clock_gettime(time.CLOCK_MONOTONIC)


def read_kernel_journal():
    done = subprocess.run(list(JCMD), capture_output=True, text=True)
    return done.stdout if done.returncode == 0 else None


def read_dmesg():
    done = subprocess.run(['dmesg'], capture_output=True, text=True)
    return done.stdout if done.returncode == 0 else None


def read_user_journal(start_epoch):
    """(entries, problem): holdout_common.journal_alive first (journalctl exits 0 with nothing when it finds no journal: never read as "no unit"), then the entries since --start-epoch."""
    try:
        hc.journal_alive()
    except hc.Refused as exc:
        return [], str(exc)
    done = subprocess.run([*UCMD, '--since', f'@{int(start_epoch)}'], capture_output=True, text=True)
    if done.returncode != 0:
        return [], f'journalctl --user exit {done.returncode}'
    out = []
    for line in done.stdout.splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out, None


def read_file(p):
    try:
        return Path(p).read_text(errors='replace')
    except OSError:
        return None


def file_mode(args):
    return bool(args.journal_file or args.dmesg_file or args.user_journal_file)


def evaluate_now(args, may_hold, *, watched=(), defer=(), expect=(), conditions=None):
    """One evaluation from the machine (or the self-test files named by the options, which force no hold)."""
    kernel = read_file(args.journal_file) if args.journal_file else read_kernel_journal()
    dmesg = read_file(args.dmesg_file) if args.dmesg_file else read_dmesg()
    if args.user_journal_file:
        entries, problem = [json.loads(x) for x in (read_file(args.user_journal_file) or '').splitlines() if x.strip()], None
    else:
        entries, problem = read_user_journal(args.start_epoch)
    return evaluate(args.mode, since=args.since, deltas=args.deltas, active=bool(args.active), start_epoch=args.start_epoch, commit=args.commit, kernel_text=kernel,
                    dmesg_text=dmesg, user_entries=entries, now_uptime=uptime(), now_epoch=time.time(), boot_offset=boot_offset(), logs=args.log or (),
                    reread=None if args.journal_file else read_kernel_journal, may_hold=may_hold and not file_mode(args), user_journal_problem=problem, watched=watched,
                    defer=defer, expect=expect, ack_path=args.ack, ack_before=args.ack_before, conditions=conditions)


def finite(text, name):
    x = float(text)
    if not math.isfinite(x):
        raise ValueError(f'{name} {text!r}: a finite number (minor 3 of the revision 1 diff review)')
    return x


def parse_delta(text):
    """LO,HI finite with LO <= HI, signed (revision 4; the m version asserted 0 <= LO); ValueError otherwise."""
    parts = str(text).split(',')
    if len(parts) != 2:
        raise ValueError(f'--delta {text!r}: LO,HI')
    lo, hi = (finite(x, '--delta') for x in parts)
    if not lo <= hi:
        raise ValueError(f'--delta {text!r}: LO <= HI')
    return lo, hi


def parse_values(args):
    """--since, --start-epoch, --delta, --commit, --ack-before parsed and checked (review B3 (a)): ValueError on any of them."""
    args.since = finite(args.since, '--since')
    args.start_epoch = finite(args.start_epoch, '--start-epoch')
    args.deltas = parse_delta(args.delta)
    if not re.fullmatch(r'[0-9a-f]{40}', str(args.commit)):
        raise ValueError(f'--commit {args.commit!r}: 40 hex digits (F or P_L)')
    if args.ack_before is not None and not re.fullmatch(r'[0-9a-f]{64}', str(args.ack_before)):
        raise ValueError(f'--ack-before {args.ack_before!r}: the 64 hex digits of cells_before.json\'s sha256')
    return args


def pending_units(path):
    """The units a pending.json names (I2 of the revision 1 diff review: the eval after lpdev_c.py compare checks them); ValueError when it does not parse."""
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    units = doc.get('units')
    if not isinstance(units, list) or not units or not all(isinstance(u, str) and u for u in units):
        raise ValueError(f'{path}: "units" (a list of unit names) expected')
    return units


def own_unit(cgroup=None):
    """The tag-* / seq-vp-* unit or scope /proc/self/cgroup shows this process in, None, or '(unreadable: ...)' (refused too)."""
    try:
        text = Path(PROC_CGROUP if cgroup is None else cgroup).read_text(encoding='utf-8')
    except OSError as exc:
        return f'(unreadable: {exc})'
    found = OWN_UNIT.findall(text)
    return found[0] if found else None


def units_active():
    done = subprocess.run(['systemctl', '--user', 'list-units', '--all', 'tag-*', 'seq-vp-*', '--state=active,activating', '--no-legend', '--no-pager'], capture_output=True, text=True)
    if done.returncode != 0:
        return None
    return [x.split()[0] for x in done.stdout.splitlines() if x.strip()]


def measure_delta(tries=3, drift=DRIFT, *, read_uptime=uptime, poke=None, read_journal=read_kernel_journal, sleep=time.sleep, active=units_active):
    """δ (the hold-out's procedure, decision_log line 477; BZ 11-18 (5): no floors): u0 = /proc/uptime, one nvidia-smi (a qai kernel line), u1 = /proc/uptime, t = the first new qai line's
    kernel time: δ in [t - u1 - .01, t - u0].  `tries` times with nothing of tag-* / seq-vp-* running.  Review minor 1: the journal must be read before each try and every try must give
    lo <= hi; otherwise nothing is written (SystemExit); nothing is truncated.  δ is signed (revision 4: the new line is the first qai line after the last one read before the try,
    whatever its kernel time against u0); the tries must agree -- max(lo) - min(hi) <= DELTA_AGREE (.5 s) -- or nothing is written.  nvidia-smi by its full path (hi_core.SMI, the
    machine gate's)."""
    if tries < 2:                               # revision 4a: the tries' agreement is the second guard against reading an old qai line (t >= u0 is gone)
        raise SystemExit(f'delta: --tries {tries}: at least 2 (the tries must agree, DELTA_AGREE)')
    running = active()
    if running is None or running:
        raise SystemExit(f'delta: units active or unreadable ({running}); measure with nothing running')
    poke = poke or (lambda: subprocess.run([str(hc._hi().SMI)], capture_output=True, text=True))      # revision 4: the machine gate's full path
    out = []
    for _ in range(tries):
        text = read_journal()
        if text is None:
            raise SystemExit('delta: the kernel journal could not be read before a try; not measured')
        before = kernel_lines(text, True)
        last = before[-1][0] if before else -1.
        u0 = read_uptime()
        poke()
        u1 = read_uptime()
        t = None
        for _wait in range(4):
            new = [x for x, line in kernel_lines(read_journal(), True) if x > last and kind(line) == 'qai']
            if new:
                t = new[0]
                break
            sleep(1)
        lo, hi = (None, None) if t is None else (round(t-u1-.01, 4), round(t-u0, 4))
        out.append(dict(u0=u0, u1=u1, t=t, lo=lo, hi=hi))
        if t is None or not lo <= hi:
            raise SystemExit(f'delta: a try gave no new qai line or an interval not lo <= hi ({out[-1]}); not measured')
        sleep(1)
    spread = max(x['lo'] for x in out) - min(x['hi'] for x in out)
    if spread > DELTA_AGREE:
        raise SystemExit(f'delta: the tries do not agree (max lo - min hi = {spread:.4f} s > {DELTA_AGREE} s): {[(x["lo"], x["hi"]) for x in out]}; not measured')
    lo, hi = min(x['lo'] for x in out), max(x['hi'] for x in out)
    return dict(schema=SCHEMA+'.delta', tries=out, delta_lo=lo, delta_hi=hi, drift=drift, delta_lo_live=round(lo-drift, 4), delta_hi_live=round(hi+drift, 4),
                watch_arg=f'{lo-drift:.4f},{hi+drift:.4f}', spread=round(spread, 4),
                rule='BZ 2026-11-18 (5): the measured ends, no floors; while a step runs both ends widened by the drift allowance (decision_log lines 489, 500; revision 4a)')


def merge_delta(*docs):
    """The merged measured interval of two or more `delta` records (before, after a takeover, after a step) for the last re-judgement (BZ 11-18: 起前与结束后的合并区间; revision 2
    diff review minor 3: every measurement of the step); no drift, no truncation.  watch_arg (when every record has delta_hi_live): an interval holding every watch interval."""
    if len(docs) < 2:
        raise SystemExit('merge-delta: two or more delta records')
    lo, hi = min(d['delta_lo'] for d in docs), max(d['delta_hi'] for d in docs)
    if not (num(lo) and num(hi) and lo <= hi):
        raise SystemExit(f'merge-delta: {lo},{hi} is not lo <= hi (signed, revision 4)')
    spread = max(d['delta_lo'] for d in docs) - min(d['delta_hi'] for d in docs)
    if spread > DELTA_AGREE:                    # revision 4a: measurements of one step that do not agree (a jump of δ) are not merged; to the lead
        raise SystemExit(f'merge-delta: the measurements do not agree (max lo - min hi = {spread:.4f} s > {DELTA_AGREE} s): '
                         f'{[(d["delta_lo"], d["delta_hi"]) for d in docs]}; not merged, to the lead (交总负责人)')
    out = dict(delta_lo=lo, delta_hi=hi, arg=f'{lo:.4f},{hi:.4f}', n=len(docs), spread=round(spread, 4))
    if all(num(d.get('delta_hi_live')) for d in docs):           # a watcher taking over (--continue-from): an interval that holds every watch interval
        lo_live = min(d['delta_lo_live'] if num(d.get('delta_lo_live')) else d['delta_lo'] for d in docs)
        out['watch_arg'] = f"{lo_live:.4f},{max(d['delta_hi_live'] for d in docs):.4f}"
    return out


def out_problem(out):
    """The watcher's directory: outside every mode root, the HI outputs and the code trees (the feasibility chain wants its root new; the code tree clean)."""
    out = Path(os.path.abspath(out))
    banned = [lc.roots(m).root for m in ('lpba40_test', 'lpba40_feas')] + [lc.roots(m).hi_out for m in ('lpba40_test', 'lpba40_feas')] + \
             [Path(lc.RESULTS)/DEVCHECK_DIR, lc.REPO, Path(hc.MAIN_REPO)]
    for b in banned:
        b = Path(os.path.abspath(b))
        if out == b or b in out.parents:
            return f'{out} is under {b}: the watcher writes outside every mode root and code tree'
    return None


def watched_active(units, pids):
    for u in units:
        q = subprocess.run(['systemctl', '--user', 'is-active', u], capture_output=True, text=True).stdout.strip()
        if q in ('active', 'activating'):
            return True
    for p in pids:
        try:
            os.kill(int(p), 0)
            return True
        except (OSError, ValueError):
            continue
    return False


def poll_line(doc, rc):
    c, nd, e = doc.get('counts_since') or {}, doc.get('non_dxg_judge') or {}, doc.get('eagain') or {}
    try:
        load = Path('/proc/loadavg').read_text().split()[0]
    except OSError:
        load = None
    return (f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} up={doc.get('uptime_now')} active={int(bool(doc.get('active')))} W={c.get('W')} P={c.get('P')} CA={c.get('CA')} "
            f"qai={c.get('qai')} eagain={c.get('eagain')}(rec {e.get('recorded')}, pend {e.get('pending')}) dxg_other={c.get('dxg_other')} non_dxg_in={nd.get('in_window')} "
            f"non_dxg_out={nd.get('outside')} drop_caches={nd.get('drop_caches')} wp_pending={doc.get('wp_pending')} cells={len(doc.get('cells') or [])} "
            f"settled={sum(1 for x in doc.get('cells') or [] if x.get('settled'))} user_entries={doc.get('user_entries')} deferred={len(doc.get('deferred') or [])} "
            f"acked={len(doc.get('acked') or [])} ack_rejected={len((doc.get('ack') or {}).get('rejected') or [])} events={len(doc.get('events') or [])} rc={rc} load={load} "
            f"hold={'PRESENT' if doc.get('hold_present') else 'absent'}")


def _stamp():
    return time.strftime('%H:%M:%S %Z')


def continue_problem(args, old_dir):
    """I1 (ii): a watcher that takes over from an earlier one (--continue-from OLD) scans from the same start: the same mode, --since, --start-epoch, --commit and ack file, a δ interval that
    holds the old one, and the old watcher ended (END in its poll.txt).  None, or the problem."""
    old_dir = Path(old_dir)
    try:
        starts = [json.loads(x) for x in (old_dir/'watch_start.json').read_text().splitlines() if x.strip()]
        poll = (old_dir/'poll.txt').read_text().splitlines()
    except (OSError, ValueError) as exc:
        return f'--continue-from {old_dir}: watch_start.json / poll.txt not readable ({type(exc).__name__})'
    if not starts:
        return f'--continue-from {old_dir}: no watch_start.json line'
    old = starts[-1].get('scan') or {}
    if not poll or not poll[-1].startswith('END'):
        return f'--continue-from {old_dir}: the old watcher has not ended (no END line): stop it first (kill its pid, watch.pid)'
    want = dict(mode=args.mode, since=args.since, start_epoch=args.start_epoch, commit=args.commit, ack=str(Path(args.ack).resolve()))
    for key, value in want.items():
        if old.get(key) != value:
            return f'--continue-from {old_dir}: {key} {value!r} is not the old watcher\'s {old.get(key)!r} (the scan must start where the old one started)'
    lo, hi = args.deltas
    olo, ohi = old.get('deltas') or (None, None)
    if not (num(olo) and num(ohi) and lo <= olo and hi >= ohi):
        return f'--continue-from {old_dir}: δ {lo},{hi} does not hold the old watcher\'s {olo},{ohi}'
    return None


def monitor_root_problem(args, out):
    """The lead's ruling of 2026-11-20 ~4:38 PM CST (minor): the watchers of one step share a monitor root ($MON: --out, or its parent when that holds a watcher, as $MON/r2 under
    $MON).  A watcher of the same mode there without its END line refuses the new one; an earlier watcher of the same mode there refuses it unless --continue-from is given."""
    out = Path(out).resolve()
    root = out.parent if (out.parent/'watch_start.json').exists() else out
    found = []
    for d in [root] + (sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []):
        if d.resolve() == out and not (d/'watch_start.json').exists():
            continue
        if (d/'watch_start.json').exists():
            try:
                starts = [json.loads(x) for x in (d/'watch_start.json').read_text().splitlines() if x.strip()]
            except (OSError, ValueError):
                return f'{d}/watch_start.json does not parse: a watcher there cannot be judged'
            if starts and (starts[-1].get('scan') or {}).get('mode') != args.mode:
                continue
            try:
                ended = (d/'poll.txt').read_text().splitlines()[-1].startswith('END')
            except (OSError, IndexError):
                ended = False
            found.append((d, ended))
    open_ = [str(d) for d, ended in found if not ended]
    if open_:
        return f'a watcher of mode {args.mode} under {root} has no END line ({open_}): stop it (kill <its pid from watch_start.json>) first'
    if found and not args.continue_from:
        return f'an earlier watcher of mode {args.mode} under {root} ({[str(d) for d, _ in found]}): a new one takes over with --continue-from <its --out>'
    return None


def watch(args):
    """The watcher (watch_m.sh's loop, in one process).  Returns its exit code; the END line and watch_end.json name it (minor 2 of the revision 1 diff review)."""
    out = Path(args.out)
    poll = out/'poll.txt'
    rejected_out = out_problem(args.out)

    ours = dict(started=False)

    def finish(code, reason):
        if rejected_out is None and out.is_dir() and (ours['started'] or not (out/'watch_start.json').exists()):
            with open(poll, 'a') as f:
                f.write(f"END {time.strftime('%Y-%m-%d %H:%M:%S %Z')} exit={code} ({reason[:200]})\n")
            (out/'watch_end.json').write_text(json.dumps(dict(exit=code, reason=reason[:500], utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), sort_keys=True)+'\n')
        return code

    def fail(reason, *, code=EXIT_ERROR, end=True):
        hold, held = hold_path(args.mode), False
        if not args.no_hold and not hold.exists():
            touch_hold(hold)
            held = True
        if rejected_out is None and out.is_dir() and (ours['started'] or not (out/'watch_start.json').exists()):     # never into a refused --out or another watcher's directory
            with open(poll, 'a') as f:
                f.write(f'EVENT {_stamp()} rc={code} hold_touched_now={held}: {reason}\n')
        print(f'watch: {reason}; hold touched now: {held}', file=sys.stderr)
        return finish(code, reason) if end else code

    if rejected_out:
        return fail(rejected_out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        parse_values(args)
        args.ack = str(Path(args.ack).resolve()) if args.ack else str((out/'ack.json').resolve())
        if not (args.watch_unit or args.watch_pid):
            raise ValueError('--watch-unit or --watch-pid')
        problem = monitor_root_problem(args, out)
        if problem:
            raise ValueError(problem)
        if args.continue_from:
            problem = continue_problem(args, args.continue_from)
            if problem:
                raise ValueError(problem)
        refusals_before = [dict(path=str(p), mtime=m, sha256=sha(p)) for p, m in refusal_records(args.mode) if (m or 0) < args.start_epoch]
        before = before_cells(args.mode, args.since, args.start_epoch)
    except (ValueError, TypeError, OSError) as exc:
        return fail(f'arguments: {exc}')
    jsonl = out/'dxg_live.jsonl'
    before_path = out/'cells_before.json'
    if not before_path.exists():
        before_path.write_bytes(cells_before_bytes(before))
    before_sha = before['problems_sha256']
    with open(out/'refusals_before.json', 'a') as f:
        f.write(json.dumps(dict(start_epoch=args.start_epoch, rule='R1 (B3 (b)): written before --start-epoch, not counted by the evaluations; listed here', records=refusals_before),
                           sort_keys=True)+'\n')
    scan = dict(mode=args.mode, since=args.since, start_epoch=args.start_epoch, commit=args.commit, ack=args.ack, deltas=list(args.deltas))
    with open(out/'watch_start.json', 'a') as f:
        f.write(json.dumps(dict(schema=SCHEMA+'.watch', argv=sys.argv, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), uptime=uptime(), source=SOURCE,
                                hold=str(hold_path(args.mode)), pid=os.getpid(), refusals_before=len(refusals_before), scan=scan, continue_from=args.continue_from,
                                cells_before=dict(problems_sha256=before_sha, rows=len(before['rows']), problems=before['problems'], ack_before=args.ack_before)),
                           sort_keys=True)+'\n')
    ours['started'] = True
    defer = DEFERRED_KINDS if args.mode == 'devcheck' else ()
    units, pids = args.watch_unit or (), args.watch_pid or ()
    conditions = {}
    if args.continue_from:                      # a takeover carries the old watcher's open conditions (their episodes go on)
        try:
            last = (Path(args.continue_from)/'dxg_live.jsonl').read_text().splitlines()[-1]
            conditions = dict(json.loads(last).get('conditions') or {})
        except (OSError, ValueError, IndexError):
            conditions = {}

    def on_term(signum, frame):                 # a watcher stopped by kill (SIGTERM) holds and writes its END like any other end (I1 (iv): a takeover then starts with --continue-from)
        raise SystemExit(f'signal {signum}')
    signal.signal(signal.SIGTERM, on_term)
    expect_last = tuple(dict.fromkeys(list(units) + list(args.expect_unit or ())))
    try:
        waited, first_active = 0., None
        while args.wait_start > 0 and not watched_active(units, pids) and waited < args.wait_start:
            time.sleep(2.)                      # started before the unit (a systemd-run --wait step blocks the runner's shell): wait for it, then poll as usual
            waited += 2.
        not_started = args.wait_start > 0 and waited >= args.wait_start
        if not_started:
            fail(f'NOT_STARTED: nothing watched became active within {args.wait_start:.0f} s; one last evaluation follows (review minor 2)', code=EXIT_NOT_STARTED, end=False)
        while True:
            args.active = 1 if watched_active(units, pids) else 0
            if args.active and first_active is None:
                first_active = time.time()
            watched = tuple(units) if args.active and first_active is not None and time.time() - first_active >= GRACE_S else ()
            if not args.active:
                time.sleep(FINAL_SETTLE_S)      # review minor 3: the manager's last lines may lag the unit's end
                try:
                    first, _ = evaluate_now(args, may_hold=False, defer=defer, expect=expect_last, conditions=conditions)
                    if first.get('started_unfinished'):
                        time.sleep(2*FINAL_SETTLE_S)
                except Exception:               # noqa: BLE001 - judged again just below, with the hold
                    pass
            try:
                doc, events = evaluate_now(args, may_hold=not args.no_hold, watched=watched, defer=defer if not args.active else (),
                                           expect=expect_last if not args.active else (), conditions=conditions)
                conditions = dict(doc.get('conditions') or {})      # revision 3 diff review I1: an episode keeps its start while it lasts; gone and back, a new one
                rc = EXIT_EVENT if events else 0
                text = json.dumps(doc, sort_keys=True, default=str)
            except Exception as exc:            # noqa: BLE001 - the evaluator cannot vouch -> hold (the m version's exit 4); the watch goes on
                tb = traceback.extract_tb(exc.__traceback__)
                hold, held = hold_path(args.mode), False
                if not args.no_hold and not hold.exists():
                    touch_hold(hold)
                    held = True
                doc = dict(schema=SCHEMA, mode=args.mode, active=bool(args.active), evaluator_error=type(exc).__name__, line=tb[-1].lineno if tb else None, hold_touched_now=held,
                           hold_present=hold.exists(), events=[f'dxg: evaluator exception {type(exc).__name__}'])
                rc, text = EXIT_ERROR, json.dumps(doc, sort_keys=True)
            with open(jsonl, 'a') as f:
                f.write(text+'\n')
            with open(poll, 'a') as f:
                f.write(poll_line(doc, rc)+'\n')
                if rc:
                    f.write(f"EVENT {_stamp()} rc={rc} hold_touched_now={doc.get('hold_touched_now')}: " + ' | '.join(str(e) for e in (doc.get('events') or [])[:8])[:2000]+'\n')
            if not args.active:
                if doc.get('deferred'):
                    (out/'pending.json').write_text(json.dumps(dict(schema=SCHEMA+'.pending', mode=args.mode, deferred=doc['deferred'],
                                                                    units=sorted({d.split('#', 1)[0] for d in doc['deferred']}), since=args.since, delta=args.delta,
                                                                    start_epoch=args.start_epoch, rule='R2 (I1): run `eval --active 0 --pending <this file>` after the step '
                                                                                                      'that writes the record'), indent=1, sort_keys=True))
                code = EXIT_NOT_STARTED if not_started else 0
                return finish(code, f'last evaluation above with active 0; deferred {len(doc.get("deferred") or [])}')
            time.sleep(args.interval)
    except BaseException as exc:                # noqa: BLE001 - review B3 (a): nothing ends the watch without the hold
        return fail(f'the watch ended by {type(exc).__name__}: {str(exc)[:200]}')


# ---- the self-test (synthetic journals and records in a temporary results root) -------------------------------------------------------------------
def _user(t, unit, text, own=False):
    return {'__MONOTONIC_TIMESTAMP': str(int(round(t*1e6))), ('_SYSTEMD_USER_UNIT' if own else 'USER_UNIT'): f'{unit}.service', 'MESSAGE': text}


def _window(unit, t0, t1=None):
    out = [_user(t0, unit, f'Started {unit}.service - x')]
    if t1 is not None:
        out.append(_user(t1, unit, f'{unit}.service: Consumed 1s CPU time.'))
    return out


def _kline(t, text, journal=True):
    return f'[{t:12.6f}] host kernel: {text}' if journal else f'[{t:12.6f}] {text}'


def _write(p, doc):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc) if not isinstance(doc, str) else doc)
    return p


SELF_COMMIT = 'f'*40


def _rmtree(path):
    """The self-test's own temporary files only (a path under its mkdtemp root)."""
    path = Path(path)
    if path.exists():
        for q in sorted(path.rglob('*'), key=lambda x: len(x.parts), reverse=True):
            q.unlink() if q.is_file() or q.is_symlink() else q.rmdir()
        path.rmdir()


def _supervisor(att, unit, win):
    _write(att/'unit_start.json', dict(uptime=win[0], unit=unit))
    if win[1] is not None:
        _write(att/'unit_end.json', dict(exit_code=0, killed_by_signal=None, sigterm_received=False, killed_by_supervisor=False, oom_lines=[], dxg_lines=[],
                                         uptime_start=win[0], uptime=win[1], seconds=round(win[1]-win[0], 3), unit=unit))


def _xb_cell(root, arm, pair, step=(1000., 30.), *, win=None, status='ok', gpu=True, commit=SELF_COMMIT, mode='lpba40_test', result=True):
    """An external-arm cell (the LPBA40 layout: records <root>/<arm>/<pair>, work <root>/work/<arm>/<pair>, nothing moved): supervisor window `win` (default step +- a little),
    one GPU step; with result=True a finished cell that passes (b) and (c)."""
    win = win or (step[0]-5, step[0]+step[1]+2)
    rec, work = Path(root)/arm/pair, Path(root)/'work'/arm/pair
    unit = expected_unit(mode, 'xb', arm, pair)
    _supervisor(rec/'unit', unit, win)
    if win[1] is None or not result:
        return rec
    work.mkdir(parents=True, exist_ok=True)
    (work/'disp.npz').write_bytes(b'disp')
    lock = _write(Path(root)/'locks'/f'xb__{arm}__{pair}.lock', json.dumps(dict(code_commit=commit))+'\n')
    man = _write(rec/'intermediates.tsv', 'path\tsize_bytes\tsha256\n')
    s = dict(returncode=0, memory_guard_killed=False, kernel_oom_lines=[], uptime_start=step[0], seconds_wall=step[1])
    if gpu:
        s['max_gpu_mem_used_mib'] = 1000
    _write(rec/'result.json', dict(status=status, spy_passed=True, export_check=dict(suspicious=False), move=None, disp=dict(file=str(work/'disp.npz'), sha256=sha(work/'disp.npz')),
                                   lock=dict(sha256=sha(lock)), checks=dict(machine=dict(passed=True)), cleanup=dict(manifest=str(man), manifest_sha256=sha(man)),
                                   method=dict(steps=[s])))
    _write(rec/'unit'/'unit_reading.json', dict(status=None, kind=None, systemd_result='success', systemd_run_exit=0, dxg_lines=None))
    return rec


def _hb_cell(root, k, pair, win=(2000., 2100.), *, commit=SELF_COMMIT, went_on=True, mode='lpba40_test'):
    rec = Path(root)/k/pair
    unit = expected_unit(mode, k, 'method' if k == 'method' else k, pair)
    _write(Path(root)/'locks'/f'hb__{k}__{pair}.lock', json.dumps(dict(code=commit))+'\n')
    _supervisor(rec/'unit', unit, win)
    _write(rec/'unit_reading.json', dict(status=None, status_kind=None, systemd_result='success', systemd_run_exit=0, dxg_lines=None))
    if k == 'method':
        _write(rec/'step4.json', dict(went_on=went_on, status='hx_certified'))
    else:
        _write(rec/'result.json', dict(status='ok'))
    return rec


def selftest(log=print):
    """The synthetic cases: one per class and rule (journal and dmesg texts, user-journal windows, supervisor records), each with its expected events and hold.  Runs in a temporary
    results root (lpba_common.RESULTS moved there for the duration); returns (n_pass, n_fail, rows)."""
    saved = lc.RESULTS
    tmp = Path(tempfile.mkdtemp(prefix='lpf_dxg_selftest_'))
    rows = []
    try:
        lc.RESULTS = tmp/'results'
        mode = 'lpba40_test'
        root = mode_root(mode)
        pre = lc.unit_name(mode, '')
        P = {k: lc.pair_name(*k.split('-')) for k in ('S03-S04', 'S05-S06', 'S07-S08', 'S09-S10', 'S11-S12', 'S13-S14', 'S15-S16', 'S17-S18')}
        fa, an = f'{pre}x-fireants_syn-S03-S04', f'{pre}x-ants_s-S05-S06'
        _xb_cell(root, 'fireants_syn', P['S03-S04'], (1000., 30.), win=(995., 1032.))
        _xb_cell(root, 'ants_s', P['S05-S06'], (1100., 30.), win=(1095., 1132.), gpu=False)
        _hb_cell(root, 'method', P['S03-S04'], (2000., 2100.))
        hi_rec = _write(lc.roots(mode).hi_out/'hi_record.json', dict(passed=True))
        boot = '[    0.000000] host kernel: Linux version x'
        base_user = ([_user(990., pre+'x-chain', 'Started '+pre+'x-chain.service - chain')] + _window(fa, 995., 1032.4) + _window(an, 1095., 1132.3)
                     + _window(pre+'he-S03-S04', 1999.9, 2100.4) + _window(pre+'hi', 2999., 3100.))
        d = (.70, .85)                          # δ_lo, δ_hi: a kernel time t reads as t - .70 and t - .85
        start_epoch = time.time()-5.

        def case(name, lines, *, want=(), user=None, dmesg=None, active=False, check=None, logs=(), pre_hold=False, kernel_none=False, problem=None, watched=(), defer=(),
                 may_hold=True, ack=None, ack_before=None, expect=(), deltas=None, epoch=None):
            """want: the substrings, one per expected event, every event matched by one (no want: no event and no hold).  epoch: another --start-epoch (revision 5)."""
            hold = hold_path(mode)
            if hold.exists():
                hold.unlink()
            if pre_hold:
                hold.touch()
            kern = None if kernel_none else '\n'.join([boot]+[_kline(t, x) for t, x in lines])
            dm = '\n'.join(['[    0.000000] Linux version x']+[_kline(t, x, False) for t, x in lines]) if dmesg is None else dmesg
            try:
                doc, events = evaluate(mode, since=900., deltas=d if deltas is None else deltas, active=active, start_epoch=start_epoch if epoch is None else epoch, commit=SELF_COMMIT,
                                       kernel_text=kern, dmesg_text=dm,
                                       user_entries=base_user if user is None else user, now_uptime=4000., now_epoch=time.time()+60, boot_offset=0., logs=logs,
                                       user_journal_problem=problem, watched=watched, defer=defer, may_hold=may_hold, expect=expect,
                                       ack_path=None if ack is None else _write(tmp/'ack'/f'{len(rows)}.json', dict(schema=ACK_SCHEMA, entries=ack)), ack_before=ack_before)
                texts = doc['events']
                expect_hold = (bool(want) and may_hold) or pre_hold
                ok = (len(texts) == len(want) and all(any(w in t for t in texts) for w in want) and all(any(w in t for w in want) for t in texts)
                      and hold.exists() is expect_hold and doc['hold_touched_now'] is (bool(want) and may_hold and not pre_hold) and (check is None or bool(check(doc))))
                why = '' if ok else f'events {texts[:6]}; hold {hold.exists()}, touched {doc["hold_touched_now"]}'
            except Exception as exc:            # noqa: BLE001
                ok, why = False, f'{type(exc).__name__}: {exc}'
            rows.append(dict(case=name, passed=ok, why=why))
            log(f"{'PASS' if ok else 'FAIL'} {name}{'' if ok else ' -- '+why}")

        E0, E1 = sorted(EAGAIN_TEXTS)
        F, A, H = f'{fa}#1', f'{an}#1', f'{pre}he-S03-S04#1'
        case('S0 qai only -> counted, nothing else', [(1010., next(iter(QAI_TEXTS)))],
             check=lambda x: x['counts_since']['qai'] == 1 and not x['judged'] and len(x['cells']) == 4 and all(c['settled'] for c in x['cells']))
        case('S1 W at the GPU step end of an ok external-arm cell -> recorded', [(1030.8, W_TEXT)], check=lambda x: x['judged'][F]['passes'] and x['judged'][F]['W'] == 1)
        case('S2 P in a method cell (no GPU segment on the uptime clock, I2) -> hold', [(2050., P_TEXT)], want=[H+': P line at kernel 2050.000000: W/P lines and the rule fails: a False (GPU segments 0)'])
        case('S3 W in a GPU cell far from every GPU step end -> hold', [(1012., W_TEXT)], want=['W/P lines and the rule fails: a False'])
        case('S4 W outside every cell window -> hold', [(1500., W_TEXT)], want=['outside every cell window or split'])
        case('S5 W in a CPU cell (no GPU step) -> hold', [(1129.8, W_TEXT)], want=[A+': W line at kernel 1129.800000: W/P lines and the rule fails: a False'])
        case('S6 EAGAIN inside the GPU step of an ok cell -> recorded', [(1015., E0)], check=lambda x: x['eagain']['recorded'] == 1 and x['eagain_cells'][F]['recorded_only'])
        case('S7 EAGAIN in the cell window, outside its GPU step -> hold', [(999.5, E1)], want=['EAGAIN-class line(s) and the rule fails: in a GPU step False'])
        case('S8 EAGAIN outside every window -> hold', [(1600., E0)], want=['EAGAIN-class line at kernel'])
        case('S9 EAGAIN in hi (no GPU segment on the uptime clock, I2) -> hold', [(3050., E0)], want=['EAGAIN-class line(s) and the rule fails: in a GPU step False (GPU segments 0)'])
        case('S10 CA -> hold', [(1500., 'misc dxg: dxgk: create_allocation failed: -12')], want=['CA line at kernel'])
        case('S11 a W variant (another errno) -> dxg_other -> hold', [(1030.8, W_TEXT.replace('fffffe00', 'fffffe01'))], want=['other dxg line'])
        case('S12 a qai variant (-2) -> dxg_other -> hold', [(1010., 'misc dxg: dxgk: dxgkio_query_adapter_info: Ioctl failed: -2')], want=['other dxg line'])
        case('S13 an EAGAIN variant (-110) -> dxg_other -> hold', [(1015., 'misc dxg: dxgk: dxgvmb_send_create_process: create_process failed -110')], want=['other dxg line'])
        case('S14 a non-dxg line in a cell window -> hold', [(1010., 'hv_balloon: unhandled message: type: 7')], want=['non-dxg kernel line'])
        case('S15 a non-dxg line outside every window -> recorded', [(1500., 'hv_balloon: unhandled message: type: 7')], check=lambda x: x['non_dxg_judge']['outside'] == 1)
        case('S16 a non-dxg line in a driver\'s window only (not a cell) -> recorded', [(993., 'hv_balloon: unhandled message: type: 7')],
             check=lambda x: x['non_dxg_judge']['outside'] == 1 and x['drivers'] == [f'{pre}x-chain'])
        case('S17 drop_caches: 1 in a cell window -> recorded (whitelist)', [(1010., 'mini_init (123): drop_caches: 1')], check=lambda x: x['non_dxg_judge']['drop_caches_in_window'] == 1)
        case('S18 drop_caches: 3 in a cell window -> hold', [(1010., 'mini_init (123): drop_caches: 3')], want=['non-dxg kernel line'])
        case('S19 dmesg unreadable -> hold', [], dmesg='', want=['dmesg not readable'])
        case('S20 the kernel journal short of dmesg (a W only in dmesg) -> hold', [], dmesg='\n'.join(['[    0.000000] Linux version x', _kline(1500., W_TEXT, False)]),
             want=['fewer lines than dmesg'])
        case('S21 a failure line of a unit of the mode -> hold', [], user=base_user+[_user(3100.1, pre+'hi', pre+"hi.service: Failed with result 'exit-code'.")],
             want=['failure line in the user journal'])
        case('S22 W after the last window while active -> pending, no hold', [(3500., W_TEXT)], active=True, check=lambda x: x['wp_pending'] == 1)
        case('S23 the same W at the last evaluation -> hold', [(3500., W_TEXT)], want=['outside every cell window or split'])
        _xb_cell(root, 'fireants_greedy', P['S07-S08'], (3205., 100.), win=(3200., None))
        running = base_user + _window(f'{pre}x-fireants_greedy-S07-S08', 3200.)
        case('S24 W and EAGAIN in a running cell while active -> pending', [(3260., W_TEXT), (3250., E0)], active=True, user=running,
             check=lambda x: x['wp_pending'] == 1 and x['eagain']['pending'] == 1)
        case('S25 (B2, m.2) a cell started and not ended at the last evaluation -> hold', [], user=running,
             want=['started without an end', "['result.json', 'unit_end.json'] not written (m.2)"])
        _rmtree(root/'fireants_greedy')
        case("S26 a STOP line in a driver unit's own output -> hold", [], user=base_user+[_user(1200., pre+'x-chain', 'S03-S04: STOP (infra)', own=True)],
             want=['STOP line (exit code None) in its journal output'])
        _hb_cell(root, 'configc', P['S09-S10'], (1032.05, 1040.))
        two = base_user + _window(f'{pre}c-S09-S10', 1032.0, 1040.2)
        case('S27 W whose two readings fall in two cells -> hold', [(1032.80, W_TEXT)], user=two, want=['placed tag-lp-c-S09-S10#1 / '+F])
        _rmtree(root/'configc')
        hi_rec.write_text(json.dumps(dict(passed=False)))
        case('S28 hi whose record did not pass -> hold', [], want=['hi_record.json did not pass'])
        hi_rec.write_text(json.dumps(dict(passed=True)))
        lock = root/'locks'/f"xb__fireants_syn__{P['S03-S04']}.lock"
        good = lock.read_text()
        lock.write_text(json.dumps(dict(code_commit='e'*40))+'\n')
        case('S29 W in an external-arm cell whose lock names another commit ((c) fails) -> hold', [(1030.8, W_TEXT)], want=['lock_code_is_commit'])
        lock.write_text(good)
        case('S30 a --log file with a STOP line -> hold', [], logs=[str(_write(tmp/'chain.log', 'cell 1 ok\nSTOP: infra\n'))], want=['line 2: STOP'])
        case('S31 the hold already there stays, not touched again', [(1500., W_TEXT)], want=['outside every cell window or split'], pre_hold=True)
        step4 = root/'method'/P['S03-S04']/'step4.json'
        good4 = step4.read_text()
        step4.write_text(json.dumps(dict(went_on=False, status='hx_certified')))
        case('S32 a method cell whose step4.json did not go on -> hold', [], want=['step4.json did not go on'])
        step4.write_text(good4)
        res = root/'fireants_syn'/P['S03-S04']/'result.json'
        good_r = res.read_text()
        res.write_text(json.dumps(dict(json.loads(good_r), status='method_failure')))
        case('S33 an external-arm cell whose status is not ok -> hold', [], want=['effective status not ok'])
        res.write_text(good_r)
        _write(root/'batches'/'xb_x'/'refusals'/'r.json', '{}')
        case('S34 a refusal record written since the step start -> hold', [], want=['refusal record r.json'])
        (root/'batches'/'xb_x'/'refusals'/'r.json').unlink()
        case('S35 the kernel journal unreadable -> hold', [], kernel_none=True, want=['kernel journal not readable'])
        _xb_cell(root, 'fireants_greedy', P['S07-S08'], (3305., 90.), win=(3300., 3400.), result=False)
        ended = base_user + _window(f'{pre}x-fireants_greedy-S07-S08', 3300., 3400.2)
        G = f'{pre}x-fireants_greedy-S07-S08#1'
        case('S36 (B2) W in an ended cell without result.json at the last evaluation -> hold', [(3350., W_TEXT)], user=ended,
             want=['W/P line(s) still pending', G+" started and ['result.json'] not written (m.2)"])
        case('S37 EAGAIN in the same cell at the last evaluation -> hold', [(3350., E0)], user=ended,
             want=['in a cell not settled while the step is no longer active', G+" started and ['result.json'] not written (m.2)"])
        case('S38 the same W while active -> pending, no hold', [(3350., W_TEXT)], user=ended, active=True, check=lambda x: x['wp_pending'] == 1)
        case('S39 (B2) journal Started and Consumed, no failure line, result.json missing -> hold at the last evaluation', [], user=ended,
             want=[G+" started and ['result.json'] not written (m.2)"])
        case("S40 a 'Starting ...' line before 'Started ...' opens no window of its own", [], user=ended+[_user(2998.9, pre+'hi', 'Starting '+pre+'hi.service - x...')],
             want=[G+" started and ['result.json'] not written (m.2)"], check=lambda x: [c['id'] for c in x['cells']].count(f'{pre}hi#1') == 1)
        _rmtree(root/'fireants_greedy')
        # B1: the two sources
        case('B1a (B1 (a)) user journal with 0 entries: every supervised cell seen only in the records -> hold; a non-dxg line in a record window holds',
             [(1010., 'hv_balloon: unhandled message: type: 7')], user=[],
             want=[F+': seen only in the supervisor records', A+': seen only in the supervisor records', H+': seen only in the supervisor records', 'non-dxg kernel line',
                   'the user journal gave 0 entries'])
        case('B1a2 user journal not readable (journal_alive) -> hold', [], problem='journalctl --user -o json -n 1 gave 0 entries, not one', want=['user journal not readable'])
        case('B1a3 a watched unit active without its Started line -> hold', [], active=True, watched=(pre+'x-chain-missing',), want=['without its Started line'])
        case('B1a4 a cell only in the user journal -> hold', [], user=base_user+_window(f'{pre}c-S15-S16', 1700., 1710.),
             want=['seen only in the user journal', "['result.json', 'unit_end.json'] not written (m.2)"])
        _xb_cell(root, 'fireants_syn', P['S11-S12'], (1500.5, 30.), win=(1495., 1531.3))
        late = base_user + _window(f'{pre}x-fireants_syn-S11-S12', 1495., 1532.)
        case('B1b (B1 (b)) W after unit_end.uptime, before the journal end, within 1 s of the GPU end -> hold (the m version: outside the window)', [(1532.2, W_TEXT)], user=late,
             want=['outside every cell window or split'])
        _xb_cell(root, 'fireants_syn', P['S13-S14'], (1605., 20.), win=(1600., 1630.))
        late2 = late + _window(f'{pre}x-fireants_syn-S13-S14', 1602., 1630.3)
        case('B1c (B1 (c)) journal Started late: a non-dxg line in the supervisor window before it -> hold', [(1601.2, 'hv_balloon: unhandled message: type: 7')], user=late2,
             want=['non-dxg kernel line'])
        # revision 4: a signed δ (this boot: the kernel clock about 31.6 s behind /proc/uptime); kernel time = uptime + δ, δ in [-31.69, -31.57]
        N = (-31.69, -31.57)

        def k_of(u):
            return round(u - 31.63, 6)
        case('NEG1 (revision 4) δ < 0: W at the GPU step end of an ok external-arm cell -> recorded', [(k_of(1030.1), W_TEXT)], user=late2, deltas=N,
             check=lambda x: x['judged'][F]['passes'])
        case('NEG2 δ < 0: W far from every GPU step end -> hold', [(k_of(1012.), W_TEXT)], user=late2, deltas=N, want=['W/P lines and the rule fails: a False'])
        case('NEG3 δ < 0: EAGAIN inside the GPU step -> recorded', [(k_of(1015.), E0)], user=late2, deltas=N, check=lambda x: x['eagain']['recorded'] == 1)
        case('NEG4 δ < 0: EAGAIN in the window, outside the GPU step -> hold', [(k_of(999.), E0)], user=late2, deltas=N,
             want=['EAGAIN-class line(s) and the rule fails: in a GPU step False'])
        case('NEG5 δ < 0: a non-dxg line in a cell window -> hold', [(k_of(1010.), 'hv_balloon: unhandled message: type: 7')], user=late2, deltas=N, want=['non-dxg kernel line'])
        case('NEG6 δ < 0: a non-dxg line outside every window -> recorded', [(k_of(1580.), 'hv_balloon: unhandled message: type: 7')], user=late2, deltas=N,
             check=lambda x: x['non_dxg_judge']['outside'] == 1)
        case('NEG7 δ < 0: a W written after the start but stamped before SINCE on the slower kernel clock -> counted, hold', [(k_of(905.), W_TEXT)], user=late2, deltas=N,
             want=['outside every cell window or split'], check=lambda x: x['counts_since']['W'] == 1 and x['kernel_scan_from'] == 900. - 31.69)
        case('POS1 δ > 0 unchanged: the scan starts at SINCE (a line stamped before it is not counted)', [(899.5, W_TEXT)], user=late2,
             check=lambda x: x['counts_since']['W'] == 0 and x['kernel_scan_from'] == 900.)
        NL = (round(N[0] - DRIFT, 4), round(N[1] + DRIFT, 4))         # revision 4a: the watch interval `delta` prints (both ends widened)
        case('NEG8 (revision 4a) the watch interval (both ends widened by DRIFT): NEG1 still recorded, the scan starts at SINCE + δ_lo - DRIFT', [(k_of(1030.1), W_TEXT)], user=late2,
             deltas=NL, check=lambda x: x['judged'][F]['passes'] and x['kernel_scan_from'] == round(900. + NL[0], 6))
        case('POS2 (revision 4a) δ_lo .09 widened to -.06: the scan starts .06 s before SINCE (stricter) -> a W stamped there counted, hold', [(899.97, W_TEXT)], user=late2,
             deltas=(round(.09 - DRIFT, 4), .54 + DRIFT), want=['outside every cell window or split'],
             check=lambda x: x['counts_since']['W'] == 1 and x['kernel_scan_from'] == round(900. + .09 - DRIFT, 6))
        # revision 2: R3 (cells before the step), acknowledgements (I1 (i)), the latest records of a resumed cell (I1 (iii)), the expected units (I2)
        _xb_cell(root, 'fireants_greedy', P['S15-S16'], (500., 30.), win=(495., 532.), status='method_failure')
        before = before_cells(mode, 900., start_epoch)
        bsha = before['problems_sha256']
        case('R3a (revision 1 diff review B1) a cell before the step with a status not ok -> hold unless --ack-before names cells_before', [], user=late2,
             want=['started before the step are not ok'], check=lambda x: x['cells_before']['problems'] == 1 and not x['cells'][0]['id'].startswith(f'{pre}x-fireants_greedy'))
        case('R3b the same with --ack-before <its sha256> -> no event', [], user=late2, ack_before=bsha, check=lambda x: x['cells_before']['acknowledged'])
        case('R3c --ack-before naming another sha256 -> hold', [], user=late2, ack_before='0'*64, want=['given another sha256'])
        _rmtree(root/'fireants_greedy')
        case('R3d cells before the step all ok -> no ack needed', [], user=late2, check=lambda x: x['cells_before']['problems'] == 0 and x['cells_before']['rows'] == 0)
        # revision 5 (feas-r2, 2026-11-21 13:01:29 CST): the queue made method/<pair>/ and its unit had not written unit/unit_start.json yet (the method queue waits at its machine gate
        # after making the directory, up to 15 min); the m version's R3 had no such row, revision 1's read it as a cell before the step with its records missing
        fresh = [root/'method'/P['S17-S18'], root/'configc'/P['S17-S18'], root/'fireants_syn'/P['S17-S18']]
        fresh[0].mkdir(parents=True)
        case('R3e (revision 5) a method record directory made since the step start, no unit/ yet (the gate wait) -> no event, listed as not started', [], user=late2, active=True,
             check=lambda x: x['cells_before']['problems'] == 0 and x['cells_before']['rows'] == 0 and x['cells_before']['not_started'] == 1)
        for f_ in fresh[1:]:
            f_.mkdir(parents=True)
        case('R3f (revision 5) the same for configuration C (made just before run_unit) and, as code-path coverage, an external arm (its unit makes it with unit/) -> no event while '
             'the step runs', [], user=late2, active=True, check=lambda x: x['cells_before']['problems'] == 0 and x['cells_before']['not_started'] == 3)
        case('R3i (revision 5a, B1) the three still without an attempt at the last evaluation (e.g. the method queue ended by SIGTERM in its gate: no STOP, no failure line) -> hold, '
             'one event each', [], user=late2, want=[f"/{b}/{P['S17-S18']} made since the step start" for b in ('method', 'configc', 'fireants_syn')],
             check=lambda x: x['cells_before']['problems'] == 0 and len(x['last_eval_open']['not_started']) == 3
             and all(e['kind'] == 'not_started' and not e['lead'] for e in x['event_items']))
        case('R3g (revision 5) the same three directories made before the step (--start-epoch after their creation) -> hold: a cell before the step, records not complete', [],
             user=late2, epoch=time.time()+60., want=['started before the step are not ok'], check=lambda x: x['cells_before']['problems'] == 3 and x['cells_before']['not_started'] == 0)
        saved_birth = globals()['birth']
        globals()['birth'] = lambda p_: None
        try:
            case('R3h (revision 5) a creation time that cannot be read -> the problem row stays, hold', [], user=late2, want=['started before the step are not ok'],
                 check=lambda x: x['cells_before']['problems'] == 3 and x['cells_before']['not_started'] == 0)
        finally:
            globals()['birth'] = saved_birth
        for f_ in fresh[1:]:
            _rmtree(f_)
        _hb_cell(root, 'method', P['S17-S18'], (2200., 2300.))      # the unit ran and ended: a normal step's close leaves not_started empty
        case('R3j (revision 5a) the same method directory once its unit ran and ended -> the last evaluation has no event, not_started empty', [],
             user=late2 + _window(f'{pre}he-S17-S18', 2199.9, 2300.4), check=lambda x: x['cells_before']['not_started'] == 0 and x['last_eval_open']['not_started'] == [])
        _rmtree(fresh[0])
        (root/'locks'/f"hb__method__{P['S17-S18']}.lock").unlink()
        def probe(lines, user, active=False):
            """The events of one evaluation without the hold: {(kind, cell): [ids]} (the ack entries name event ids)."""
            doc, _ = evaluate(mode, since=900., deltas=d, active=active, start_epoch=start_epoch, commit=SELF_COMMIT, kernel_text='\n'.join([boot]+[_kline(t, x) for t, x in lines]),
                              dmesg_text='\n'.join(['[    0.000000] Linux version x']+[_kline(t, x, False) for t, x in lines]), user_entries=user, now_uptime=4000.,
                              now_epoch=time.time()+60, boot_offset=0., may_hold=False)
            out = {}
            for e in doc['event_items']:
                out.setdefault((e['kind'], e['cell'] if e['cell'] is not None else '-'), []).append(e['id'])
            return out

        def entry(ids, kind_, cell, words=''):
            return dict(event_id=ids[(kind_, cell)][0], cell=cell, kind=kind_, decision_log='docs/decision_log.md line 7xx: the tool judged it so', bz_words=words)

        res.write_text(json.dumps(dict(json.loads(good_r), status='method_failure')))
        nd_line = [(1010., 'hv_balloon: unhandled message: type: 7')]
        ids0, ids1 = probe([], late2), probe(nd_line, late2)
        lead_ack = [entry(ids0, 'status', F)]
        case('ACK1 (I1 (i)) the lead acknowledges an XB method_failure in a cell without kernel events, by its event_id -> no hold', [], user=late2, ack=lead_ack,
             check=lambda x: len(x['acked']) == 1 and len(x['ack']['accepted']) == 1)
        case('ACK2 the same acknowledgement with a non-dxg line in that cell -> rejected, hold', nd_line, user=late2, ack=[entry(ids1, 'status', F)],
             want=['effective status not ok', 'non-dxg kernel line'], check=lambda x: len(x['ack']['rejected']) == 1)
        case('ACK3 kernel and status events of that cell acknowledged with BZ\'s words -> no hold', nd_line, user=late2,
             ack=[entry(ids1, 'status', F, 'BZ: 按建议'), entry(ids1, 'non_dxg', F, 'BZ: 按建议')], check=lambda x: len(x['acked']) == 2)
        only = late2+_window(f'{pre}c-S15-S16', 1700., 1710.)
        ids4 = probe([], only)
        case('ACK4 an acknowledgement of a kind the lead may not confirm (cells), without BZ\'s words -> rejected, hold', [], user=only,
             ack=[entry(ids4, 'status', F), entry(ids4, 'cells', f'{pre}c-S15-S16#1')],
             want=['seen only in the user journal', "['result.json', 'unit_end.json'] not written (m.2)"], check=lambda x: len(x['ack']['rejected']) == 1)
        case('ACK5 an acknowledgement without its decision_log line -> rejected, hold', [], user=late2, ack=[dict(lead_ack[0], decision_log='')], want=['effective status not ok'])
        res.write_text(json.dumps(dict(json.loads(good_r), status='wrapper_error')))
        ids7 = probe([], late2)
        case('ACK7 (revision 2 diff review I1) the lead acknowledges an XB wrapper_error (not on LEAD_STATUSES) -> rejected, hold', [], user=late2, ack=[entry(ids7, 'status', F)],
             want=['effective status not ok (status wrapper_error'])
        res.write_text(json.dumps(dict(json.loads(good_r), status='infra_failure', failure=dict(kind='spy_breach'))))
        ids7b = probe([], late2)
        case('ACK7b an XB infra_failure caused by a spy breach -> rejected, hold', [], user=late2, ack=[entry(ids7b, 'status', F)], want=['effective status not ok (status infra_failure'])
        res.write_text(json.dumps(dict(json.loads(good_r), status='unit_limit')))
        ids11, ids11k = probe([], late2), probe(nd_line, late2)
        case('ACK11 (revision 3a) the lead acknowledges an XB unit_limit in a cell without kernel events -> accepted, no hold', [], user=late2, ack=[entry(ids11, 'status', F)],
             check=lambda x: len(x['acked']) == 1 and not x['ack']['rejected'])
        case('ACK11b the same with a non-dxg line in that cell -> rejected, hold', nd_line, user=late2, ack=[entry(ids11k, 'status', F)],
             want=['effective status not ok (status unit_limit', 'non-dxg kernel line'], check=lambda x: len(x['ack']['rejected']) == 1)
        res.write_text(good_r)                  # ACK11b left the XB record at unit_limit

        def seq_case(name, fn):
            hold = hold_path(mode)
            if hold.exists():
                hold.unlink()
            try:
                ok, why = fn()
            except Exception as exc:            # noqa: BLE001
                ok, why = False, f'{type(exc).__name__}: {exc}'
            rows.append(dict(case=name, passed=ok, why=why))
            log(f"{'PASS' if ok else 'FAIL'} {name}{'' if ok else ' -- '+why}")

        def cond_run(dmesg_ok, state, now, ackfile=None, hold_ok=False):
            dm_ = '\n'.join(['[    0.000000] Linux version x']) if dmesg_ok else ''
            return evaluate(mode, since=900., deltas=d, active=True, start_epoch=start_epoch, commit=SELF_COMMIT, kernel_text=boot, dmesg_text=dm_, user_entries=late2,
                            now_uptime=now, now_epoch=time.time()+60, boot_offset=0., conditions=state, ack_path=ackfile, may_hold=hold_ok)[0]

        def cond_episodes():
            d1 = cond_run(False, {}, 4000.)
            e1 = next(e for e in d1['event_items'] if 'dmesg not readable' in e['text'])
            ackfile = _write(tmp/'ack'/'cond.json', dict(schema=ACK_SCHEMA, entries=[dict(event_id=e1['id'], cell='-', kind='source', decision_log='line 7xx',
                                                                                      bz_words='BZ: 这一次照规则只记录')]))
            d2 = cond_run(False, d1['conditions'], 4020., ackfile)
            d3 = cond_run(True, d2['conditions'], 4040., ackfile)
            d4 = cond_run(False, d3['conditions'], 9000., ackfile, hold_ok=True)
            e4 = [e for e in d4['event_items'] if 'dmesg not readable' in e['text']]
            ok = len(d2['acked']) == 1 and not d2['events'] and not d3['conditions'] and len(e4) == 1 and e4[0]['id'] != e1['id'] and hold_path(mode).exists()
            return ok, f"d2 acked {len(d2['acked'])}, d3 conditions {d3['conditions']}, d4 {e4}"
        seq_case('COND1 (revision 3 diff review I1) BZ acknowledged "dmesg not readable"; it cleared and came back: a new episode, a new id, hold', cond_episodes)

        def same_microsecond():
            two = [(1580., W_TEXT), (1580., W_TEXT)]
            ids_ = probe(two, late2)[('wp', '-')]
            if len(ids_) != 2 or ids_[0] == ids_[1]:
                return False, f'ids {ids_}'
            ackfile = _write(tmp/'ack'/'dup.json', dict(schema=ACK_SCHEMA, entries=[dict(event_id=ids_[0], cell='-', kind='wp', decision_log='line', bz_words='BZ')]))
            doc_ = evaluate(mode, since=900., deltas=d, active=False, start_epoch=start_epoch, commit=SELF_COMMIT, kernel_text='\n'.join([boot]+[_kline(t, x) for t, x in two]),
                            dmesg_text='\n'.join(['[    0.000000] Linux version x']+[_kline(t, x, False) for t, x in two]), user_entries=late2, now_uptime=4000.,
                            now_epoch=time.time()+60, boot_offset=0., ack_path=ackfile, may_hold=False)[0]
            return len(doc_['acked']) == 1 and len(doc_['events']) == 1, f"acked {len(doc_['acked'])} events {doc_['events']}"
        seq_case('DUP1 (revision 3 diff review minor 1) two W lines of the same microsecond and text: two ids; one acknowledged, the other holds', same_microsecond)
        stop_rq = late2+[_user(1200., pre+'review-queue', 'Started '+pre+'review-queue.service - q'),
                         _user(1250., pre+'review-queue', 'STOP (20): S03-S04: the unit\'s evidence; nothing rerun; to the lead', own=True),
                         _user(1260., pre+'c-queue', 'Started '+pre+'c-queue.service - q'),
                         _user(1270., pre+'c-queue', 'STOP (20): S05-S06: infra; nothing rerun; to the lead', own=True)]
        ids13 = probe([], stop_rq)
        case('ACK13 (revision 3c) STOP (20) of the review queue without BZ\'s words -> rejected; of configuration C\'s queue -> accepted', [], user=stop_rq,
             ack=[dict(event_id=ids13[('driver_stop', f'{pre}review-queue')][0], cell=f'{pre}review-queue', kind='driver_stop', decision_log='line 760', bz_words=''),
                  dict(event_id=ids13[('driver_stop', f'{pre}c-queue')][0], cell=f'{pre}c-queue', kind='driver_stop', decision_log='line 760', bz_words='')],
             want=[f'driver {pre}review-queue: STOP line (exit code 20)'], check=lambda x: len(x['acked']) == 1 and len(x['ack']['rejected']) == 1)
        step4 = root/'method'/P['S03-S04']/'step4.json'
        good4 = step4.read_text()
        step4.write_text(json.dumps(dict(went_on=False, status='infra_failure', kind='no_line_json:he_exit_None:systemd_failed')))
        ids14 = probe([], late2)
        case('ACK14 (revision 3c) a method arm infra_failure (no line.json; no STOP line in the journal) without BZ\'s words -> rejected, hold', [], user=late2,
             ack=[entry(ids14, 'status', H)], want=['step4.json did not go on (status infra_failure)'], check=lambda x: len(x['ack']['rejected']) == 1)
        step4.write_text(good4)
        res.write_text(json.dumps(dict(json.loads(good_r), status='infra_failure', failure=dict(kind='input_refused_after_lock:provenance'))))
        ids12 = probe([], late2)
        case('ACK12 (revision 3 diff review I2) an XB infra_failure from input_refused_after_lock:provenance -> the lead\'s acknowledgement rejected, hold', [], user=late2,
             ack=[entry(ids12, 'status', F)], want=['effective status not ok (status infra_failure'], check=lambda x: len(x['ack']['rejected']) == 1)
        res.write_text(json.dumps(dict(json.loads(good_r), status='infra_failure', failure=dict(kind='input_rejected'))))
        ids12b = probe([], late2)
        case('ACK12b an XB infra_failure from input_rejected -> the lead\'s acknowledgement rejected, hold', [], user=late2, ack=[entry(ids12b, 'status', F)],
             want=['effective status not ok (status infra_failure'], check=lambda x: len(x['ack']['rejected']) == 1)
        res.write_text(good_r)
        w1 = [(1580., W_TEXT)]
        ids6 = probe(w1, late2)
        case('ACK6 (revision 2 diff review B1) BZ acknowledged one W outside every window by its event_id; a later W of the same kind still holds', w1+[(2500., W_TEXT)], user=late2,
             ack=[entry(ids6, 'wp', '-', 'BZ: 照规则只记录')], want=['W line at kernel 2500.000000'], check=lambda x: len(x['acked']) == 1)
        stop20 = late2+[_user(1200., pre+'x-chain', 'S03-S04: STOP (20): infra; nothing rerun; waiting for BZ', own=True)]
        stop23 = stop20+[_user(1300., pre+'x-chain', 'S05-S06: STOP (23): wrapper; nothing rerun; waiting for BZ', own=True)]
        ids8 = probe([], stop23)
        e20, e23 = ids8[('driver_stop', f'{pre}x-chain')]
        case('ACK8 (revision 2 diff review I1) the lead acknowledges STOP (20) of the chain -> accepted; STOP (23) (wrapper, one code many causes) -> rejected', [], user=stop23,
             ack=[dict(event_id=e20, cell=f'{pre}x-chain', kind='driver_stop', decision_log='line 759', bz_words=''),
                  dict(event_id=e23, cell=f'{pre}x-chain', kind='driver_stop', decision_log='line 759', bz_words='')],
             want=['STOP line (exit code 23)'], check=lambda x: len(x['acked']) == 1 and len(x['ack']['rejected']) == 1)
        _xb_cell(root, 'fireants_greedy', P['S07-S08'], (3205., 100.), win=(3200., None))
        killed = late2 + _window(f'{pre}x-fireants_greedy-S07-S08', 3200.)
        G7 = f'{pre}x-fireants_greedy-S07-S08#1'
        nd_late = [(3250., 'hv_balloon: unhandled message: type: 7')]
        ids9 = probe(nd_late, killed)
        case('ACK9 (revision 2 diff review B2) a cell started and never ended: at the last evaluation its window reaches the scan end, its non-dxg line holds and the lead\'s '
             'acknowledgements of started_unfinished and records_missing are rejected', nd_late, user=killed,
             ack=[entry(ids9, 'started_unfinished', G7), entry(ids9, 'records_missing', G7)],
             want=['started without an end', 'not written (m.2)', 'non-dxg kernel line'], check=lambda x: len(x['ack']['rejected']) == 2 and x['cells'][-1]['kernel_events'])
        ids9b = probe([], killed)
        case('ACK9b the same cell without kernel lines: the lead\'s two acknowledgements accepted', [], user=killed,
             ack=[entry(ids9b, 'started_unfinished', G7), entry(ids9b, 'records_missing', G7)], check=lambda x: len(x['acked']) == 2)
        _rmtree(root/'fireants_greedy')
        twice = late2 + _window(f'{pre}prep', 3300., 3310.) + [_user(3310.1, pre+'prep', f"{pre}prep.service: Failed with result 'exit-code'.")] \
            + _window(f'{pre}prep', 3400., 3410.) + [_user(3410.1, pre+'prep', f"{pre}prep.service: Failed with result 'exit-code'.")]
        ids10 = probe([], twice)
        case('ACK10 (revision 2 diff review B1) a unit failing in two windows: two unit_failed events, each of its own window; one acknowledged with BZ\'s words, the other holds',
             [], user=twice, ack=[entry(ids10, 'unit_failed', f'{pre}prep#1', 'BZ: 照规则')], want=[f'failure line in the user journal ({pre}prep#2)'],
             check=lambda x: len(x['acked']) == 1)
        res.write_text(good_r)
        cr = root/'configc'/P['S15-S16']
        _supervisor(cr/'unit', expected_unit(mode, 'configc', 'configc', P['S15-S16']), (600., 700.))
        _write(cr/'unit_reading.json', dict(status='infra_failure', status_kind='x', systemd_result='success', systemd_run_exit=0))
        _write(cr/'result.json', dict(status='infra_failure'))
        _supervisor(cr/'unit_2', expected_unit(mode, 'configc', 'configc', P['S15-S16']), (1700., 1710.))
        _write(cr/'unit_2_reading.json', dict(status=None, status_kind=None, systemd_result='success', systemd_run_exit=0))
        _write(cr/'result_resume.json', dict(status='ok'))
        _write(root/'locks'/f"hb__configc__{P['S15-S16']}.lock.resume", json.dumps(dict(code=SELF_COMMIT))+'\n')
        res2 = late2+_window(f'{pre}c-S15-S16', 1700., 1710.2)
        case('RES1 (I1 (iii)) configuration C resumed in unit_2: result_resume.json (ok) is read, not the first run\'s infra_failure', [], user=res2,
             check=lambda x: x['cells_before']['problems'] == 0 and any(c['id'] == f'{pre}c-S15-S16#1' and c['settled'] for c in x['cells']))
        _rmtree(root/'configc')
        _supervisor(cr/'unit', expected_unit(mode, 'configc', 'configc', P['S15-S16']), (600., 700.))
        _write(cr/'unit_reading.json', dict(status='infra_failure', status_kind='x', systemd_result='success', systemd_run_exit=0))
        _write(cr/'result.json', dict(status='infra_failure'))
        _supervisor(cr/'unit_2', expected_unit(mode, 'configc', 'configc', P['S15-S16']), (1700., None))
        running_c = late2+_window(f'{pre}c-S15-S16', 1700.)
        case('RES4 (revision 2 diff review I2) configuration C resume still running in unit_2: the first run\'s infra_failure is not read', [], user=running_c, active=True,
             check=lambda x: not any(e['kind'] == 'status' for e in x['event_items']))
        _rmtree(root/'configc')
        xr = root/'fireants_greedy'/P['S15-S16']
        _supervisor(xr/'unit', expected_unit(mode, 'xb', 'fireants_greedy', P['S15-S16']), (600., 700.))
        _write(xr/'unit'/'unit_reading.json', dict(status='infra_failure', kind='x', systemd_result='failed', systemd_run_exit=1))
        _write(xr/'result.json', dict(status='infra_failure'))
        _supervisor(xr/'unit_resume', expected_unit(mode, 'xb', 'fireants_greedy', P['S15-S16']), (1750., 1790.))
        _write(xr/'unit_resume'/'unit_reading.json', dict(status=None, kind=None, systemd_result='success', systemd_run_exit=0))
        _write(xr/'resume'/'result.json', dict(status='ok'))
        res3 = late2+_window(f'{pre}x-fireants_greedy-S15-S16', 1750., 1790.2)
        case('RES2 (I1 (iii)) an external arm resumed: resume/result.json (ok) is read, not the first run\'s', [], user=res3,
             check=lambda x: x['cells_before']['problems'] == 0 and any(c['id'] == f'{pre}x-fireants_greedy-S15-S16#1' and c['settled'] for c in x['cells']))
        _write(xr/'resume'/'result.json', dict(status='infra_failure'))
        case('RES3 the resumed external arm whose resume/result.json is not ok -> hold', [], user=res3, want=['effective status not ok (status infra_failure, overridden False, resumed True)'])
        _rmtree(root/'fireants_greedy')
        case('EXP1 (I2) an expected unit without a window in the user journal at the last evaluation -> hold', [], user=late2, expect=(pre+'hi', pre+'eval'),
             want=[f'expected unit {pre}eval has no window'])
        case('EXP2 the same while active -> not yet', [], user=late2, expect=(pre+'eval',), active=True)
        # the evaluation unit of step 11 (a CPU cell) and the deferred devc (I1)
        case('E11 W in the evaluation unit tag-lp-eval (a CPU cell) -> hold', [(3720., W_TEXT)], user=late2+_window(pre+'eval', 3700., 3800.),
             want=[f'{pre}eval#1: W line at kernel 3720.000000: W/P lines and the rule fails: a False (GPU segments 0)'])
        lc_mode = mode
        mode = 'devcheck'
        dev = _window('tag-lp-devc', 1000., 1100.)
        case('I1a devc at watch\'s last evaluation: W and the missing devcheck_c.json deferred, no hold', [(1050., W_TEXT)], user=dev, defer=DEFERRED_KINDS,
             check=lambda x: x['wp_deferred'] == 1 and len(x['deferred']) == 2)
        _write(mode_root('devcheck')/'c'/'devcheck_c.json', dict(passed=True))
        case('I1b the eval after compare (no deferral): the same W holds (no GPU segment)', [(1050., W_TEXT)], user=dev, want=['tag-lp-devc#1: W line at kernel 1050.000000: W/P lines and the rule fails: a False'])
        _write(mode_root('devcheck')/'c'/'devcheck_c.json', dict(passed=False))
        case('I1c the eval after compare: devcheck_c.json did not pass -> hold', [], user=dev, want=['devcheck_c.json did not pass'])
        (mode_root('devcheck')/'c'/'devcheck_c.json').unlink()
        case('I1d the eval after compare: devcheck_c.json still missing -> hold', [], user=dev, want=["['devcheck_c.json'] not written (m.2)"])
        _write(mode_root('devcheck')/'c'/'devcheck_c.json', dict(passed=False))
        case('I1e (I2) the eval after compare with an empty user journal: devc expected (pending.json) -> hold', [(1050., 'hv_balloon: unhandled message: type: 7')], user=[],
             expect=('tag-lp-devc',), want=['the user journal gave 0 entries', 'expected unit tag-lp-devc has no window'])
        mode = lc_mode
    finally:
        lc.RESULTS = saved
        for p in sorted(tmp.rglob('*'), key=lambda q: len(q.parts), reverse=True):
            p.unlink() if p.is_file() or p.is_symlink() else p.rmdir()
        tmp.rmdir()
    n_pass = sum(r['passed'] for r in rows)
    return n_pass, len(rows)-n_pass, rows


# ---- the command line ----------------------------------------------------------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('now')
    d = sub.add_parser('delta')
    d.add_argument('--out', required=True)
    d.add_argument('--tries', type=int, default=3)
    d.add_argument('--drift', type=float, default=DRIFT)
    m = sub.add_parser('merge-delta')
    m.add_argument('records', nargs='+')
    for name in ('eval', 'watch'):
        p = sub.add_parser(name)
        p.add_argument('--mode', required=True, choices=MODES)
        p.add_argument('--since', required=True)            # parsed by parse_values (B3 (a): a value that does not parse holds)
        p.add_argument('--delta', required=True)
        p.add_argument('--start-epoch', required=True)
        p.add_argument('--commit', required=True)
        p.add_argument('--log', action='append')
        p.add_argument('--no-hold', action='store_true')
        p.add_argument('--journal-file')
        p.add_argument('--dmesg-file')
        p.add_argument('--user-journal-file')
        p.add_argument('--expect-unit', action='append')            # I2 of the revision 1 diff review
        p.add_argument('--ack')                                     # I1 (i): the ack file (watch: default <out>/ack.json)
        p.add_argument('--ack-before')                              # R3: the sha256 of cells_before.json the lead disposed of
        if name == 'eval':
            p.add_argument('--active', type=int, choices=(0, 1), default=0)
            p.add_argument('--pending')                             # I2: the units of a pending.json are expected (the eval after lpdev_c.py compare)
        else:
            p.add_argument('--out', required=True)
            p.add_argument('--watch-unit', action='append')
            p.add_argument('--watch-pid', action='append')
            p.add_argument('--interval', type=float, default=POLL_S)
            p.add_argument('--wait-start', type=float, default=WAIT_START_S)
            p.add_argument('--continue-from')                       # I1 (ii): an earlier watcher's --out; the same scan start
    sub.add_parser('selftest')
    b = sub.add_parser('before')                                    # R3: cells_before.json for the lead (the same bytes the watch writes)
    b.add_argument('--mode', required=True, choices=MODES)
    b.add_argument('--since', required=True)
    b.add_argument('--start-epoch', required=True)
    b.add_argument('--out', required=True)
    return ap


VALUE_OPTIONS = ('--since', '--delta', '--start-epoch', '--commit', '--ack-before')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    for i in range(len(argv)-1):                # '--delta -0.1,0.8' would read as an option: joined, so that parse_values sees the value and refuses it with the hold (B3 (a))
        if argv[i] in VALUE_OPTIONS and not argv[i+1].startswith('--'):
            argv[i:i+2] = [f'{argv[i]}={argv[i+1]}', '']
    argv = [a for a in argv if a != '']
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:                   # B3 (a): eval or watch that cannot even parse its arguments holds (the hold file is the same for every mode), exit 4
        if exc.code in (0, None) or not argv or argv[0] not in ('eval', 'watch') or '--no-hold' in argv or any(a.split('=')[0] in ('--journal-file', '--dmesg-file',
                                                                                                                                       '--user-journal-file') for a in argv):
            raise
        touch_hold(hold_path('lpba40_test'))
        print(f'{argv[0]}: the arguments do not parse; hold touched ({hold_path("lpba40_test")})', file=sys.stderr)
        return EXIT_ERROR
    if args.cmd == 'now':
        print(f'{uptime():.2f} {time.time():.6f}')
        return 0
    if args.cmd == 'delta' and own_unit() is not None:
        raise SystemExit(f'delta: this process is in {own_unit()}: run it as an ordinary background process, never in a unit (the machine gates count every tag-* unit)')
    if args.cmd == 'watch' and own_unit() is not None:                  # minor 5: a watcher refused in a unit holds (the step would run unwatched)
        if not args.no_hold:
            touch_hold(hold_path(args.mode))
        print(f'watch: this process is in {own_unit()}: run it as an ordinary background process (setsid nohup), never in a unit; hold touched', file=sys.stderr)
        return EXIT_ERROR
    if args.cmd == 'before':
        since, epoch = finite(args.since, '--since'), finite(args.start_epoch, '--start-epoch')
        if Path(args.out).exists():
            raise SystemExit(f'{args.out} exists')
        doc = before_cells(args.mode, since, epoch)
        Path(args.out).write_bytes(cells_before_bytes(doc))
        print(f"{doc['problems_sha256']}  problems {doc['problems']}  rows {len(doc['rows'])}  not_started {len(doc['not_started'])}  {args.out}")
        return 0
    if args.cmd == 'delta':
        if Path(args.out).exists():
            raise SystemExit(f'{args.out} exists')
        doc = measure_delta(args.tries, args.drift)
        Path(args.out).write_text(json.dumps(doc, indent=1, sort_keys=True))
        print(json.dumps(doc, sort_keys=True))
        return 0
    if args.cmd == 'merge-delta':
        print(json.dumps(merge_delta(*(json.loads(Path(r).read_text()) for r in args.records)), sort_keys=True))
        return 0
    if args.cmd == 'selftest':
        n_pass, n_fail, _ = selftest()
        print(f'selftest: {n_pass} passed, {n_fail} failed')
        return 0 if n_fail == 0 else 1
    if args.cmd == 'eval':
        try:
            parse_values(args)
            expect = list(args.expect_unit or ()) + (pending_units(args.pending) if args.pending else [])
            doc, events = evaluate_now(args, may_hold=not args.no_hold, expect=tuple(dict.fromkeys(expect)))
        except BaseException as exc:            # noqa: BLE001 - the evaluator cannot vouch -> hold (review B3 (a): a δ or a value that does not parse too)
            hold, held = hold_path(args.mode), False
            if not args.no_hold and not file_mode(args) and not hold.exists():
                touch_hold(hold)
                held = True
            print(json.dumps(dict(schema=SCHEMA, evaluator_error=type(exc).__name__, reason=mask(str(exc))[:300], hold_touched_now=held,
                                  events=[f'dxg: evaluator exception {type(exc).__name__}']), sort_keys=True))
            return EXIT_ERROR
        print(json.dumps(doc, sort_keys=True, default=str))
        return EXIT_EVENT if events else 0
    return watch(args)


if __name__ == '__main__':
    sys.exit(main())
