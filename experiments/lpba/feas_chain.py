"""LP-C2 item 3 (contract C26.5-9, C26.8-5; prereg §7, §12 item 14): the feasibility chain of the LPBA40 second test set -- the two pairs S01<-S02, S02<-S01 (pair set
'lpba40_feas'), no truth, every entry once in its own tag-lp-feas-* unit, the stop conditions of prereg §7.2 read from each entry's own records after its unit, a new root.

The steps (prereg §8.2's order; §7.1 "所有臂各走一遍"; LP-C2 integration item 5 wired LP-D's check (a) and LP-B's queues; integration review minor 3: LP-A's, LP-D's and the
evaluation's units are their entries' own launch lines, prep_lpba.launch_line, lpdev_a.launch_line, hi_lpba.launch_line, atropos_lpba.launch_line, lpba_eval.launch_line):
  prep        experiments/lpba/prep_lpba.py --pair-set lpba40_feas          (LP-A)  stage/prep_record.json passed
  devcheck_a  experiments/lpba/devcheck/lpdev_a.py run (unit tag-lp-feas-deva) (LP-D; prereg §12 item 4 (a))  <feas root>/devcheck_a/devcheck_a.json passed
  hi          experiments/hi/hi_lpba.py --pair-set lpba40_feas              (LP-A)  <hi_out>/hi_record.json passed
  atropos     experiments/lpba/atropos_lpba.py --pair-set lpba40_feas       (LP-A)  atropos/atropos_record.json passed; feasibility_stop.stop -> STOP to BZ (a WM ratio outside
                                                                                     [.8, 1.25]: a design question, always to BZ; C26.8-5, 裁决 B1)
  method      hb_method.py queue --pair-set lpba40_feas in its queue unit tag-lp-feas-he-queue (LP-B; hb_core.queue_launch's start string, waited)   the cells (below), the batch records
  review      hb_review.py queue ... in tag-lp-feas-review-queue (LP-B)      <feas root>/review/finalized_outputs.json over both pairs (step 5: the method output fixed)
  convexadam_v4seg   xb_chain.py queue --pair-set lpba40_feas --arms convexadam_v4seg in tag-lp-feas-x-chain (LP-B; xb_chain.chain_launch_string, waited)
  input_lock  hb_store.py lpba-input-lock --pair-set lpba40_feas (LP-B, step 7; review I2 (a): <feas root>/hb_input_lock.json, nothing committed, no field deleted)
  configc     hb_configc.py queue ... in tag-lp-feas-c-queue (LP-B)
  ants_s, fireants_greedy, fireants_syn      xb_chain.py queue ... --arms <arm> in tag-lp-feas-x-chain, one arm at a time (LP-B)
  eval        python -m bayesreg.eval.lpba_eval evaluate --pair-set lpba40_feas  (LP-C2; the no-truth mode)  lpba_eval.json: not stopped, nothing failed, no label file opened;
              every arm's field metrics, BL1 certificate and 24 tetrahedra computed, the grid (181, 217, 181), the brain mask not empty
An LP-B step's unit is its entry's own start string (the unit name its machine gate exempts, its MemoryMax 2G, nice, PYTHONPATH) with --wait, the log and the end probe added
(launch_unit_argv); its queue starts the per-pair units (MemoryMax: the method's supervisor 12G, the review 8G, configuration C 12G, XB 10G -- LP-B's constants, the same in both modes),
whose end probes (xb_unit.py's unit_end.json) the chain reads after the step (check_inner_units).  The code commit of an LP-B step: its batch records' (batch.json's code record) and every
cell record that names one; the input lock's record names none: the code tree's HEAD after its unit, clean (commit_from 'tree').
A step whose command is None makes the whole chain refuse before it starts (fail closed: "所有臂各走一遍").
The stop conditions (prereg §7.2; C26.8-5):
  every step     its unit's exit is not 0, or its record is missing or says not passed -> stop (first look for a tool error; none found: to BZ);
  method         (1) a cell's line status hx_provenance_rejected or hx_implementation_error, or its terminal record infra_failure or needs_lead_rejudgment -> stop, to BZ;
                 (2) a side's alignment QC Dice < .50 -> recorded, the chain goes on (not a stop; to be looked at for a tool error);
                 (3) both directions fell back because of the alignment QC (line.json failure class segmentation_qc) -> STOP to BZ, no formal run (裁决 I2);
                 a fallback in one direction only, a QC between .50 and .80: the method's behaviour, recorded, no stop;
  configc, external arms   status wrapper_error, infra_failure or needs_lead_rejudgment -> stop; an export check that is suspicious (correlation < .999) -> stop (on the
                 feasibility pairs always looked at as a tool error);
  disk           the feasibility roots' size extrapolated to 17 pairs (x 17/2) plus 20 GiB above C:'s free space -> stop, to BZ (prereg §11; holdout_common.disk_guard's floor).
Not stops (prereg §7.2 末): a fallback that is not both directions' QC fallback, a QC in [.50, .80), the time of a step under its unit's limit.
  memory         (LP-C2 review minor 3; diff review minor 1; integration item 5: the per-pair units of LP-B's queues too) each unit's end probe (experiments/he/unit_probe.py: the
                 cgroup's memory.peak, memory.max and memory.events): OOM evidence (oom or oom_kill > 0) -> stop, to BZ -- also after a unit whose exit is not 0, from the probe as far as
                 it was written (integration review minor 1); memory.max not the unit's MemoryMax -> stop, to the lead (integration review minor 4); the peak (it counts the page cache
                 and is capped by MemoryMax) at the limit without OOM evidence -> recorded, stop, to the lead to look at; a peak, memory.max or events that cannot be read -> stop, to
                 the lead; the per-pair units' records one a pair, others skipped and recorded (integration review minor 5).  The feasibility evaluation runs in the no-truth mode:
                 its peak is no bound of the formal evaluation's;
  code           (review minor 4) every step's own record names the code commit it ran on; it must be the chain's HEAD at its start (C26.9-3: the feasibility pairs on one commit);
  any exception of the chain itself -> stop with its record (review minor 2).
A unit's exit that is not 0 still has its step's check run first: a stop the check hands to BZ (the WM ratio: atropos_lpba exits 8 then) is the stop; otherwise the exit is (review I2).
Every unit: the hold file (lpba_common.check_hold); holdout_common.machine_gate (no other tag-* / seq-vp-* unit, memory, load, the 10 s gap) through holdout_common.wait_for (polled, 15 minutes,
then precheck_timeout: a stop) (review I1) and the hold file again after it passed (diff review minor 2); lpba_common.disk_guard; after the unit holdout_common.note_unit_end.
Where the chain runs (diff review I, the lead's choice (a)): from a shell or with setsid, never inside a systemd unit -- inside a tag-* unit or scope (/proc/self/cgroup; integration
review minor 2) it refuses with exit 8 and writes nothing; its own process is exempt from no machine gate.
The units' spec (diff review minor 1, extended by integration review minor 3 from MemoryMax to the whole unit): each step's feasibility unit is its entry's own start string, and it
must equal the same entry's start string of the formal mode lpba40_test (spec_problems: every option -- MemoryMax, MemorySwapMax, RuntimeMaxSec, CPUQuota, OOMPolicy, nice, the
environment -- and the command, but for the unit's name, the pair set and the log; lpdev_a's (a) has its one unit; the input lock has no formal unit), else the chain refuses before
its first unit; a per-pair unit's memory.max not the formal per-pair unit's MemoryMax -> stop, to the lead.
The root: lpba40_feas_v1/ and seg/unet35_v4_ens_lpba40_feas_v1/ must be new (C26.5-9: a rerun uses new roots; `retire` renames the old ones, recorded, never deletes).  The
chain's record: <feas root>/chain/chain.json (and chain_stopped.json on a stop), written once.

    python -B experiments/lpba/feas_chain.py check            # the commands present? the roots new?  nothing started
    python -B experiments/lpba/feas_chain.py run              # the chain (from a clean detached checkout of the commit under test, C26.9-1)
    python -B experiments/lpba/feas_chain.py retire --tag T   # rename the old feasibility roots to <name>_<T>, recorded
Exit codes: 0 every step passed with no stop, 4 a stop (chain_stopped.json says which and to whom), 2 refused before the first unit (a missing command, roots not new, the code not
clean, a unit spec not the formal unit's), 8 refused because the chain runs inside a tag-* unit (nothing written), 1 other.
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

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import lpba_common as lc  # noqa: E402

hc, hp = lc.hc, lc.hp
PAIR_SET = 'lpba40_feas'
FORMAL_PAIR_SET = 'lpba40_test'          # the formal mode: its units' MemoryMax are the ones the feasibility units must have
SCHEMA = 'bayesreg.lpba40.feas_chain.v1'
QC_LOW = .50                      # prereg §7.2 method (2)
QC_FALLBACK_CLASS = 'segmentation_qc'      # he_core.failure('segmentation_qc', ...) of a QC fallback (he_core.t88_decision)
STOP_LINE_STATUSES = ('hx_provenance_rejected', 'hx_implementation_error')
STOP_TERMINAL = ('infra_failure', 'needs_lead_rejudgment')
STOP_ARM_STATUSES = ('wrapper_error', 'infra_failure', 'needs_lead_rejudgment')
FULL_GRID = (181, 217, 181)       # prereg §2.2 P4
PAIRS_FORMAL, PAIRS_FEAS = 17, 2
SUBJECTS_FORMAL, SUBJECTS_FEAS = 34, 2         # review minor 3: the per-subject products (HI, Atropos, the mm copies) scale by 34/2, the per-pair ones by 17/2
PYTHON = '/home/rmcfz/miniconda3/envs/sci310/bin/python'
PROC_CGROUP = Path('/proc/self/cgroup')
TAG_UNIT = re.compile(r'/(tag-[^/\s]+\.(?:service|scope))(?=/|$)', re.MULTILINE)      # review minor 2: a tag-* service or scope (systemd-run --scope)
# The diff review's I (the lead's choice (a)): the chain runs from a shell or setsid, never inside a unit -- every entry's machine gate knows only its own unit, so a chain unit would be
# "another unit" to all of them.  run_chain refuses (exit 8, nothing written) when /proc/self/cgroup shows this process inside a tag-* unit; the machine gate exempts nothing.
PROBE = 'experiments/he/unit_probe.py'


EXPERIMENTS = HERE.parent


def _entry(sub, name):
    """An entry module of this code tree's experiments/<sub>/ (LP-B's hb_core, xb_chain), imported as those entries import each other."""
    import importlib
    path = str(EXPERIMENTS/sub)
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module(name)


def hb_queue_launch(kind):
    """LP-C2 integration item 5: an HB queue's own start string, LP-B's (hb_core.queue_launch: the unit tag-lp[-feas]-he-queue, -review-queue, -c-queue, MemoryMax 2G, nice 5, PYTHONPATH
    <frozen>:<code>) for the queue command the lead's launch-string prints (`queue --pair-set M`)."""
    def launch(pair_set, code, *, log=None):
        return _entry('hb', 'hb_core').queue_launch(kind, ['queue', '--pair-set', pair_set], repo=code, pair_set=pair_set)
    launch.__qualname__ = f'hb_queue_launch({kind!r})'
    return launch


def xb_chain_launch(arm):
    """LP-C2 integration item 5: XB's chain unit for one arm, LP-B's (xb_chain.chain_launch_string: tag-lp[-feas]-x-chain, MemoryMax 2G, nice 5), as the hold-out ran one chain per arm."""
    def launch(pair_set, code, *, log=None):
        return _entry('xb', 'xb_chain').chain_launch_string(pair_set, [arm], code=code)
    launch.__qualname__ = f'xb_chain_launch({arm!r})'
    return launch


def entry_line(sub, name):
    """LP-C2 integration review minor 3: an LP-A or LP-D entry's own launch line (its launch_line: hp_core.launch_line with the entry's MemoryMax, MemorySwapMax, RuntimeMaxSec,
    CPUQuota, nice, threads and PYTHONPATH; --wait and the log already in it), given the chain's log; the chain wraps only its command with the end probe.  lpdev_a's (a) has one
    unit, tag-lp-feas-deva, whatever the pair set."""
    def launch(pair_set, code, *, log=None):
        module = _entry(sub, name)
        line = module.launch_line(code=code, log=log) if name == 'lpdev_a' else module.launch_line(pair_set, code=code, log=log)
        return shlex.split(line)
    launch.__qualname__ = f'entry_line({name!r})'
    return launch


def eval_line(pair_set, code, *, log=None):
    """LP-C2 integration review minor 3: the evaluation entry's own launch line (bayesreg.eval.lpba_eval.launch_line: UNIT_SPEC, the hold-out's tag-hb-eval -- MemoryMax 10G, swap 4G,
    OOMPolicy continue, CPUQuota 800%, RuntimeMaxSec 72 h, nice 5, NUMBA 2) for `evaluate --pair-set <pair_set> --out-dir <the chain's eval_out>`, given the chain's log."""
    from bayesreg.eval import lpba_eval as E
    return shlex.split(E.launch_line(pair_set, places()['eval_out'], code=code, log=log))


def hb_inner_memory(kind):
    """The MemoryMax of the per-pair units an HB queue starts (LP-B's constants, the same in every mode): the method's supervisor he_core.SUPERVISOR_UNIT, the review's
    he_core.EVALUATION_UNIT, configuration C's hb_core.CONFIGC_UNIT."""
    def memory():
        hb = _entry('hb', 'hb_core')
        return dict(method=hb.he.SUPERVISOR_UNIT, review=hb.he.EVALUATION_UNIT, configc=hb.CONFIGC_UNIT)[kind]['memory_max']
    memory.__qualname__ = f'hb_inner_memory({kind!r})'
    return memory


def xb_inner_memory():
    """The MemoryMax of XB's per-pair units: xb_core.UNIT_PROPERTIES (prereg §2.3 (g))."""
    found = [p.split('=', 1)[1] for p in _entry('xb', 'xb_core').UNIT_PROPERTIES if p.startswith('MemoryMax=')]
    if len(found) != 1:
        raise ValueError(f'xb_core.UNIT_PROPERTIES: {len(found)} MemoryMax (one expected)')
    return found[0]


@dataclass(frozen=True)
class Step:
    name: str
    tool: str                     # the unit is lpba_common.unit_name('lpba40_feas', tool): tag-lp-feas-<tool> (an LP-B queue: its own queue unit's name; LP-D (a): lpdev_a's own unit)
    package: str
    argv: tuple                   # python's argv after -B (a script relative to the code tree, or -m <module>); None: not in this code tree yet; () with `launch`
    runtime_s: int
    memory_max: str               # None with `launch`: the start string's (memory_max_of)
    peak_scale: float = 1.     # the formal run's peak over the feasibility unit's 〔推断〕: one unit holds one pair (or works subject by subject, pair by pair), so its peak does not grow with the count
    launch: object = None         # LP-C2 integration item 5: (pair_set, code) -> the entry's own start string (an LP-B queue or chain unit); the chain adds --wait, the log and the end probe
    inner_memory: object = None   # () -> the MemoryMax of the per-pair units the step's queue starts (their end probes are read after the step: inner_unit_problems)
    commit_from: str = 'record'   # 'record': its records name the code commit; 'tree': they name none (the input lock) -- the code tree's HEAD, clean, after the unit
    frozen_path: bool = False     # PYTHONPATH <frozen>:<code> (LP-B's tools, as hb_core.queue_launch); else <code>
    waits: bool = False           # review minor 3: `launch` is an entry's own launch line that already waits and logs to the log it is given (LP-A, LP-D, the evaluation); else an
                                  # LP-B start string that does neither (split_start)


XB_STEP_ARMS = ('convexadam_v4seg', 'ants_s', 'fireants_greedy', 'fireants_syn')       # xb_core.LPBA_ARMS in prereg §8.2's order (steps 6, 9; a test compares the set)
STEPS = (
    Step('prep', 'prep', 'LP-A', (), None, None, launch=entry_line('lpba', 'prep_lpba'), waits=True),
    Step('devcheck_a', 'deva', 'LP-D (prereg §12 item 4 (a))', (), None, None, launch=entry_line('lpba/devcheck', 'lpdev_a'), waits=True),
    Step('hi', 'hi', 'LP-A', (), None, None, launch=entry_line('hi', 'hi_lpba'), waits=True),
    Step('atropos', 'atropos', 'LP-A', (), None, None, launch=entry_line('lpba', 'atropos_lpba'), waits=True),
    Step('method', 'he-queue', 'LP-B', (), None, None, launch=hb_queue_launch('method'), inner_memory=hb_inner_memory('method')),
    Step('review', 'review-queue', 'LP-B', (), None, None, launch=hb_queue_launch('review'), inner_memory=hb_inner_memory('review')),
    Step('convexadam_v4seg', 'x-chain', 'LP-B', (), None, None, launch=xb_chain_launch('convexadam_v4seg'), inner_memory=xb_inner_memory),
    Step('input_lock', 'input-lock', 'LP-B', ('experiments/hb/hb_store.py', 'lpba-input-lock', '--pair-set', PAIR_SET), 1800, '2G', commit_from='tree', frozen_path=True),
    Step('configc', 'c-queue', 'LP-B', (), None, None, launch=hb_queue_launch('configc'), inner_memory=hb_inner_memory('configc')),
    *(Step(arm, 'x-chain', 'LP-B', (), None, None, launch=xb_chain_launch(arm), inner_memory=xb_inner_memory) for arm in XB_STEP_ARMS[1:]),
    Step('eval', 'eval', 'LP-C2', (), None, None, launch=eval_line, waits=True),
)
NEED_HOME_GIB = 3.                # 〔估算〕prereg §11: the feasibility pairs about 2.5 GB


class ChainError(RuntimeError):
    """An exception of a step's check after a unit whose exit was not 0 (its message holds the exit code)."""


class Stop(Exception):
    """A stop condition of prereg §7.2: `to` is 'BZ' or 'lead (tool error first)'."""

    def __init__(self, step, reason, *, to='lead (tool error first; none found: BZ)', kind='check'):
        super().__init__(f'{step}: {reason}')
        self.step, self.reason, self.to, self.kind = step, reason, to, kind


# ---- the places ------------------------------------------------------------------------------------------------------------------------------------
def places():
    r = lc.roots(PAIR_SET)
    return dict(root=r.root, hi_out=r.hi_out, chain=r.root/'chain', eval_out=r.root/'eval'/'run_1')


def missing_commands(steps=STEPS):
    return [f'{s.name} ({s.package})' for s in steps if s.argv is None and s.launch is None]


def roots_problem():
    p = places()
    there = [str(p[k]) for k in ('root', 'hi_out') if p[k].exists()]
    return None if not there else f'{there} exist: a feasibility rerun uses new roots (C26.5-9); rename the old ones first (feas_chain.py retire)'


def retire(tag):
    """Renames the feasibility roots to <name>_<tag> (kept, never deleted) and writes retired.json inside each.  Returns the record."""
    if not tag or '/' in tag or tag.startswith('.'):
        raise ValueError(f'tag {tag!r}: a plain name')
    p, out = places(), []
    for key in ('root', 'hi_out'):
        old = p[key]
        if not old.exists():
            continue
        new = old.with_name(f'{old.name}_{tag}')
        if new.exists():
            raise FileExistsError(f'{new} exists')
        lc.guard_write(new)
        old.rename(new)
        hc.write_json_once(new/'retired.json', dict(schema=SCHEMA+'.retired', old=str(old), new=str(new), tag=tag, utc=lc.utc()))
        out.append(dict(old=str(old), new=str(new)))
    return dict(retired=out, utc=lc.utc())


# ---- the units ----------------------------------------------------------------------------------------------------------------------------------------
def step_command(step, *, code):
    """The step's own command: python -B <argv> (a script of the code tree, or -m <module>; '{eval_out}' and the other places filled in)."""
    argv = [a.format(**{k: str(v) for k, v in places().items()}) for a in step.argv]
    if argv[0] != '-m':
        argv[0] = str(Path(code)/argv[0])
    return [PYTHON, '-B', *argv]


def probe_path(step):
    return places()['chain']/f'{step.name}_probe_end.json'


def unit_log(step):
    """The unit's output: <chain>/<step>_unit.log (the XB steps share one unit name)."""
    return places()['chain']/f'{step.name}_unit.log'


def probe_command(step, *, code):
    return shlex.join([PYTHON, '-I', '-B', str(Path(code)/PROBE), '--when', 'end', '--out', str(probe_path(step))])


def unit_argv(step, *, code):
    """systemd-run --user --wait --collect ... /bin/bash -c '<the step's command>; ec=$?; <the end probe>; exit $ec' (hp_core.launch_line's properties: MemoryMax, MemorySwapMax 0,
    RuntimeMaxSec, the threads, PYTHONPATH the code tree, <frozen>:<code> for LP-B's tools).  The end probe (experiments/he/unit_probe.py, python -I -B) writes the unit's cgroup numbers
    (memory.peak) to <chain>/<step>_probe_end.json, after the command and before the unit ends (firewall_c.probed_inner_command's way).  A step with `launch` (an LP-B queue or chain unit;
    review minor 3: an LP-A or LP-D entry's or the evaluation's own launch line): its entry's own start string, waited (launch_unit_argv).  Only the input lock (no formal unit) is
    started from `argv` with this chain's properties."""
    if step.launch is not None:
        return launch_unit_argv(step, code=code)
    unit = lc.unit_name(PAIR_SET, step.tool)
    log = unit_log(step)
    env = dict(OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, NUMBA_NUM_THREADS=1, PYTHONDONTWRITEBYTECODE=1,
               PYTHONPATH=f'{_entry("hb", "hb_core").he.core.FROZEN}:{code}' if step.frozen_path else str(code))
    inner = f'{shlex.join(step_command(step, code=code))}; ec=$?; {probe_command(step, code=code)}; exit $ec'
    return ['systemd-run', '--user', f'--unit={unit}', '--wait', '--collect', f'--working-directory={code}', '--nice=10', f'--property=MemoryMax={step.memory_max}',
            '--property=MemorySwapMax=0', f'--property=RuntimeMaxSec={step.runtime_s}', f'--property=StandardOutput=append:{log}', f'--property=StandardError=append:{log}',
            *[f'--setenv={k}={v}' for k, v in env.items()], '/bin/bash', '-c', inner]


LOG_PROPERTIES = ('--property=StandardOutput=', '--property=StandardError=')


def split_start(argv, *, unit, log=None):
    """(options, command) of an entry's start string: systemd-run --user <options> <command>; ValueError when it is not one of `unit`.  `log` None (an LP-B start string): ValueError
    when it already waits or logs.  `log` given (review minor 3: an entry's own launch line, hp_core.launch_line's form, built with the chain's log): it must wait once and append its
    output and errors to exactly that log; the options are returned without the --wait and the two log properties (launch_unit_argv puts them back)."""
    argv = [str(a) for a in argv]
    if argv[:2] != ['systemd-run', '--user']:
        raise ValueError(f'not a systemd-run --user start string: {argv[:3]}')
    k = next((i for i in range(2, len(argv)) if not argv[i].startswith('--')), None)
    if k is None:
        raise ValueError(f'a start string without a command: {argv}')
    options, command = argv[2:k], argv[k:]
    if options.count(f'--unit={unit}') != 1 or sum(o.startswith('--unit=') for o in options) != 1:
        raise ValueError(f'the start string is not of the unit {unit}: {options}')
    logs = [o for o in options if o.startswith(LOG_PROPERTIES)]
    if log is None:
        if '--wait' in options or logs:
            raise ValueError(f'the start string already waits or logs: {options}')
        return options, command
    want = sorted(f'{p}append:{log}' for p in LOG_PROPERTIES)
    if options.count('--wait') != 1 or sorted(logs) != want:
        raise ValueError(f'an entry\'s launch line must wait once and log to {log}: {options}')
    return [o for o in options if o != '--wait' and not o.startswith(LOG_PROPERTIES)], command


def launch_unit_argv(step, *, code):
    """LP-C2 integration item 5: an LP-B step's unit is its entry's own start string (step.launch: hb_core.queue_launch, xb_chain.chain_launch_string -- the unit name its own machine gate
    exempts, its MemoryMax, nice and PYTHONPATH) with --wait after --unit, the log appended, and its command wrapped as the other steps' are: bash -c '<command>; ec=$?; <the end probe>;
    exit $ec'.  Nothing else of the start string changes.  Review minor 3: an LP-A or LP-D entry's or the evaluation's own launch line (step.waits) is built with the chain's log and
    already waits; it is taken the same way (its --wait and log in the same places), so its unit's properties are the entry's own."""
    unit = lc.unit_name(PAIR_SET, step.tool)
    log = unit_log(step)
    options, command = split_start(step.launch(PAIR_SET, code, log=log), unit=unit, log=log if step.waits else None)
    k = options.index(f'--unit={unit}')
    inner = f'{shlex.join(command)}; ec=$?; {probe_command(step, code=code)}; exit $ec'
    return ['systemd-run', '--user', *options[:k+1], '--wait', *options[k+1:], f'--property=StandardOutput=append:{log}', f'--property=StandardError=append:{log}', '/bin/bash', '-c', inner]


def start_memory_max(argv):
    """The MemoryMax of a start string (its one --property=MemoryMax=)."""
    found = [str(a).split('=', 2)[2] for a in argv if str(a).startswith('--property=MemoryMax=')]
    if len(found) != 1:
        raise ValueError(f'{len(found)} MemoryMax in the start string (one expected)')
    return found[0]


def memory_max_of(step, *, code=None, pair_set=PAIR_SET):
    """The MemoryMax of the step's unit: its memory_max, or (a step with `launch`) the one of its entry's start string for `pair_set`."""
    if step.launch is None:
        return step.memory_max
    return start_memory_max(step.launch(pair_set, Path(lc.REPO if code is None else code), log=unit_log(step)))


def run_unit(argv):
    done = subprocess.run(argv, capture_output=True, text=True)
    return dict(returncode=done.returncode, stdout=done.stdout[-2000:], stderr=done.stderr[-2000:])


# ---- the checks after each step (prereg §7.2) --------------------------------------------------------------------------------------------------------------
def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def commit_of(doc):
    """The code commit an entry's record names: code_commit, code_head, code (a commit string, or a code record with commit or head), checks.code.commit; None when it names none."""
    if not isinstance(doc, dict):
        return None
    for key in ('code_commit', 'code_head'):
        if isinstance(doc.get(key), str):
            return doc[key]
    code = doc.get('code')
    if isinstance(code, str):
        return code
    if isinstance(code, dict):
        for key in ('commit', 'head'):
            if isinstance(code.get(key), str):
                return code[key]
    checks = doc.get('checks') if isinstance(doc.get('checks'), dict) else {}
    inner = checks.get('code') if isinstance(checks.get('code'), dict) else {}
    return inner.get('commit') if isinstance(inner.get('commit'), str) else None


def _passed_record(step, path):
    doc = _json(path)
    if doc is None or doc.get('passed') is not True:
        raise Stop(step, f'{path}: missing or not passed ({None if doc is None else doc.get("passed")})')
    return doc


def check_prep():
    doc = _passed_record('prep', lc.roots(PAIR_SET).stage/lc.PREP_RECORD)
    return dict(record=str(lc.roots(PAIR_SET).stage/lc.PREP_RECORD), passed=True, pairs=[r.get('pair') for r in (doc.get('stage_summary') or {}).get('pairs') or []],
                code_commit=commit_of(doc))


def check_hi():
    path = lc.roots(PAIR_SET).hi_out/'hi_record.json'
    doc = _passed_record('hi', path)
    return dict(record=str(path), passed=True, rows=len(doc.get('rows') or []), code_commit=commit_of(doc))


def check_atropos():
    path = lc.roots(PAIR_SET).atropos/'atropos_record.json'
    doc = _json(path)
    if doc is None:
        raise Stop('atropos', f'{path}: missing')
    stop = doc.get('feasibility_stop') or {}
    ratios = {s: (w.get('wm_ratio') or {}).get('decimal') for s, w in (doc.get('wm') or {}).items()}
    if stop.get('stop') is True or any(r is not None and not (.8 <= float(r) <= 1.25) for r in ratios.values()):
        raise Stop('atropos', f'a WM ratio outside [.8, 1.25]: {ratios} (prereg §7.2: a design question)', to='BZ', kind='wm_ratio')
    missing = [s for s in lc.FEAS_SUBJECTS if ratios.get(s) is None]
    if missing:                                                          # review minor 2: a ratio not recorded is not "in the band"
        raise Stop('atropos', f'{path}: no WM ratio of {missing} (prereg §7.2: both subjects\' ratios are read)', kind='wm_ratio_missing')
    if doc.get('passed') is not True:
        raise Stop('atropos', f'{path}: not passed')
    return dict(record=str(path), passed=True, wm_ratio=ratios, code_commit=commit_of(doc))


def _arm_table():
    from bayesreg.eval import lpba_eval as E
    table, _ = E.load_arm_table(PAIR_SET)
    return E, {a['name']: a for a in table['arms']}


def _rows():
    from bayesreg.eval import lpba_eval as E
    return E.pair_rows(PAIR_SET, lc.read_lists())


def check_method(arms=None, rows=None):
    """Prereg §7.2 method (1)-(3) from each cell's terminal.json, line.json and t88/run.json."""
    if arms is None:
        _, arms = _arm_table()
    rows = _rows() if rows is None else rows
    arm = arms['method']
    out, qc_fallback, low = {}, [], []
    for row in rows:
        cell = Path(arm['record_dir'].format(pair=row[3], arm='method'))
        terminal = _json(cell/'terminal.json') or {}
        line = _json(cell/'output'/'line.json') or _json(cell/'line.json') or {}
        t88 = _json(cell/'t88'/'run.json') or {}
        status = terminal.get('line_status') or line.get('line_status')
        if not status:
            raise Stop('method', f'{row[3]}: no terminal status (no terminal.json or line.json)')
        if status in STOP_LINE_STATUSES or status in STOP_TERMINAL:
            raise Stop('method', f'{row[3]}: {status} (prereg §7.2 method (1))', to='BZ', kind='method_1')
        failure = line.get('failure') or {}
        qc_rows = [r for r in (t88.get('alignment_qc') or []) if isinstance(r, dict)]+[s.get('alignment_qc') for s in (t88.get('side_refusals') or {}).values() if isinstance(s, dict) and s.get('alignment_qc')]
        dice = [r.get('dice') for r in qc_rows if isinstance(r, dict) and isinstance(r.get('dice'), (int, float))]
        fallback_by_qc = status in ('hx_fallback_configC', 'hx_fallback_identity') and failure.get('class') == QC_FALLBACK_CLASS
        if fallback_by_qc:
            qc_fallback.append(row[3])
        if any(d < QC_LOW for d in dice):
            low.append(row[3])
        out[row[3]] = dict(line_status=status, failure_class=failure.get('class'), qc_dice=dice, fallback_by_qc=fallback_by_qc,
                           code_commit=commit_of(terminal) or commit_of(line) or commit_of(t88))
    if len(qc_fallback) == len(rows):
        raise Stop('method', f'both directions fell back because of the alignment QC {qc_fallback} (prereg §7.2 method (3)): no formal run', to='BZ', kind='method_3')
    batches = batch_records('method', BATCH_PATTERNS['method'])
    return dict(cells=out, qc_below_050=low, note='a QC below .50: recorded, the chain goes on (prereg §7.2 method (2)); first look for a tool error' if low else None,
                batches=batches['batches'], code_commit=_one_commit('method', [batches['code_commit'], *(c['code_commit'] for c in out.values())]))


def check_arm(name, arms=None, rows=None):
    """Configuration C and the external arms: the terminal record's status and the export check (prereg §7.2)."""
    if arms is None:
        _, arms = _arm_table()
    rows = _rows() if rows is None else rows
    arm = arms[name]
    out = {}
    for row in rows:
        base = Path(arm['record_dir'].format(pair=row[3], arm=name))
        path = next((base/rel for rel in arm['terminal']['files'] if (base/rel).is_file()), None)
        doc = _json(path) if path is not None else None
        if doc is None:
            raise Stop(name, f'{row[3]}: no terminal record in {base}')
        status = doc.get('status')
        if status in STOP_ARM_STATUSES:
            raise Stop(name, f'{row[3]}: status {status} (prereg §7.2)', kind='arm_status')
        suspicious = ((doc.get('export_check') or {}).get('suspicious'))
        if arm['export_check'] is not None and suspicious is not False:
            raise Stop(name, f'{row[3]}: the export check suspicious ({suspicious}): on the feasibility pairs a tool error to look for (prereg §7.2)', kind='export')
        out[row[3]] = dict(status=status, terminal=str(path), export_suspicious=suspicious, code_commit=commit_of(doc))
    batches = batch_records(name, BATCH_PATTERNS.get(name, 'xb_*'), arms=None if name in BATCH_PATTERNS else [name])
    return dict(cells=out, batches=batches['batches'], code_commit=_one_commit(name, [batches['code_commit'], *(c['code_commit'] for c in out.values())]))


# ---- LP-C2 integration item 5: the records of LP-D's check (a) and of LP-B's queues ---------------------------------------------------------------------------
BATCH_PATTERNS = dict(method='hb_method_*', review='hb_review_*', configc='hb_configc_*')     # hb_core.new_batch_dir, hb_configc.queue; XB's xb_<UTC> (xb_chain.new_batch_dir) by its arms
DEVCHECK_A_RECORD = ('devcheck_a', 'devcheck_a.json')                                         # lpdev_common.roots().a / lpdev_a.RECORD (a test compares)
INPUT_LOCK_NAME = 'hb_input_lock.json'                                                        # hb_store.lpba_input_lock's feasibility place (a test compares)


def batch_records(step, pattern, *, arms=None):
    """The batch records (batch.json) of the step's queue under <feas root>/batches/<pattern> (the roots are new: every one is this chain's), XB's chosen by their arms; the one commit their
    code records name (holdout_common.code_identity's / lpba_arms.code_record's commit).  Stop when there is none, or one has no batch.json."""
    base = places()['root']/'batches'
    found = []
    for path in sorted(base.glob(pattern)):
        if not path.is_dir():
            continue
        doc = _json(path/'batch.json')
        if doc is None:
            raise Stop(step, f'{path}/batch.json: missing or not readable (the queue did not finish its batch record)', kind='batch')
        if arms is not None and list(doc.get('arms') or []) != list(arms):
            continue
        found.append((path, doc))
    if not found:
        raise Stop(step, f'no batch record {base}/{pattern}{"" if arms is None else f" of the arms {arms}"}', kind='batch')
    return dict(batches=[str(p) for p, _ in found], code_commit=_one_commit(step, [commit_of(doc) for _, doc in found]))


def check_devcheck_a():
    """LP-D's check (a) (prereg §12 item 4 (a)): <feas root>/devcheck_a/devcheck_a.json passed; the commit of its code record."""
    path = places()['root'].joinpath(*DEVCHECK_A_RECORD)
    doc = _passed_record('devcheck_a', path)
    return dict(record=str(path), passed=True, code_commit=commit_of(doc))


def check_review(rows=None):
    """Step 5's end (C26.6-6, C22.H1-3): <feas root>/review/finalized_outputs.json of this mode over both pairs, every pair with its review record and an output status of the arm table
    (a pair without one stopped the chain at the method step: prereg §7.2 method (1)); the commit its code record and the review batches name."""
    from bayesreg.eval import lpba_eval as E
    rows = _rows() if rows is None else rows
    path = E.finalized_path(PAIR_SET)
    doc = _json(path)
    if doc is None:
        raise Stop('review', f'{path}: missing (the method output is not fixed)', kind='finalized')
    pairs = doc.get('pairs') or {}
    names = [row[3] for row in rows]
    if doc.get('pair_set') != PAIR_SET or sorted(pairs) != sorted(names):
        raise Stop('review', f"{path}: pair set {doc.get('pair_set')}, pairs {sorted(pairs)}; {PAIR_SET} {sorted(names)} expected", kind='finalized')
    _, arms = _arm_table()
    output = set(arms['method']['terminal']['classes']['output'])
    bad = {p: (pairs[p] or {}).get('line_status') for p in names if (pairs[p] or {}).get('line_status') not in output or not (pairs[p] or {}).get('review')}
    if bad:
        raise Stop('review', f'{path}: pairs without an output status or a review record {bad}', kind='finalized')
    batches = batch_records('review', BATCH_PATTERNS['review'])
    return dict(record=str(path), pairs={p: pairs[p].get('line_status') for p in names}, batches=batches['batches'],
                code_commit=_one_commit('review', [batches['code_commit'], commit_of(doc)]))


def check_input_lock(rows=None):
    """Step 7 in the feasibility mode (LP-B review I2, the lead's ruling (a)): <feas root>/hb_input_lock.json, written once by hb_store.py lpba-input-lock -- this mode's, a prediction row of
    each of the two subjects' three files and of each HI record, a method_field row of each pair (sha256 there for every hx_certified pair).  Its record names no code commit (commit_from
    'tree')."""
    rows = _rows() if rows is None else rows
    path = places()['root']/INPUT_LOCK_NAME
    doc = _json(path)
    if doc is None:
        raise Stop('input_lock', f'{path}: missing', kind='input_lock')
    lines = [r for r in doc.get('rows') or [] if isinstance(r, dict)]
    predictions = {(r.get('subject'), Path(str(r.get('path'))).name) for r in lines if r.get('kind') == 'prediction' and isinstance(r.get('sha256'), str)}
    want = {(s, n) for s in lc.FEAS_SUBJECTS for n in ('probs36.npz', 'structure_groups.npz', 'provenance.json')}
    records = {r.get('subject') for r in lines if r.get('kind') == 'hi_record' and isinstance(r.get('sha256'), str)}
    fields = {r.get('pair'): r for r in lines if r.get('kind') == 'method_field'}
    problems = []
    if doc.get('schema') != 'bayesreg.hb.input_lock.v1' or doc.get('pair_set') != PAIR_SET:
        problems.append(f"schema {doc.get('schema')}, pair set {doc.get('pair_set')}")
    if not want <= predictions or records != set(lc.FEAS_SUBJECTS):
        problems.append(f'the predictions {sorted(want-predictions)} or the HI records {sorted(set(lc.FEAS_SUBJECTS)-records)} are not locked')
    names = [row[3] for row in rows]
    if sorted(fields) != sorted(names) or any(f.get('line_status') == 'hx_certified' and not isinstance(f.get('sha256'), str) for f in fields.values()):
        problems.append(f'the method fields {sorted(fields)} ({names} expected; an hx_certified field\'s sha256 there)')
    if problems:
        raise Stop('input_lock', f'{path}: {problems}', kind='input_lock')
    return dict(record=str(path), sha256=hc.sha256_file(path), rows=len(lines), code_commit=None)



def check_eval(eval_out=None):
    eval_out = places()['eval_out'] if eval_out is None else Path(eval_out)
    summary = _json(eval_out/'lpba_eval.json')
    if summary is None:
        raise Stop('eval', f'{eval_out}/lpba_eval.json: missing')
    if summary.get('stopped') or summary.get('failed') or summary.get('not_reached'):
        raise Stop('eval', f"stopped {summary.get('stopped')}, failed {summary.get('failed')}, not reached {summary.get('not_reached')}")
    if summary.get('label_files_opened') or summary.get('truth_mode') is not False:
        raise Stop('eval', f"the no-truth mode opened labels {summary.get('label_files_opened')} (truth_mode {summary.get('truth_mode')})", kind='label')
    out = {}
    for path in sorted((eval_out/'pairs').glob('*.json')):
        rec = _json(path) or {}
        if rec.get('complete') is not True or rec.get('truth_read') is not False:
            raise Stop('eval', f'{path}: not complete or a truth read')
        if tuple((rec.get('grid') or {}).get('shape_zyx') or ()) != FULL_GRID:
            raise Stop('eval', f"{path}: grid {(rec.get('grid') or {}).get('shape_zyx')} is not {FULL_GRID} (prereg §2.2 P4)")
        if not (rec.get('brain') or {}).get('nodes'):
            raise Stop('eval', f'{path}: the brain mask is empty')
        for arm, entry in (rec.get('arms') or {}).items():
            if entry.get('outcome') != 'output':
                continue
            m = entry.get('metrics') or {}
            if not all(isinstance(m.get(k), (int, float)) for k in ('sdlogj_official', 'sdlogj_brain')) or entry.get('certificate') is None or entry.get('tet24') is None:
                raise Stop('eval', f'{path}: {arm}: a quantity was not computed (SDlogJ, brain log J, BL1 or the 24 tetrahedra)')
        out[rec.get('pair')] = sorted(rec.get('arms') or {})
    if len(out) != PAIRS_FEAS:
        raise Stop('eval', f'{len(out)} pair records, {PAIRS_FEAS} expected')
    return dict(pairs=out, code_commit=commit_of(((summary.get('record') or {}).get('identity') or {})))


def _one_commit(step, commits):
    """The one commit the records of a step name (None when they name none, a stop when they name two)."""
    found = sorted({c for c in commits if c})
    if len(found) > 1:
        raise Stop(step, f'its records name two code commits {found} (C26.9-3: one commit)', kind='code_commit')
    return found[0] if found else None


def tree_bytes(path):
    path = Path(path)
    return 0 if not path.exists() else sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def check_disk(usage=None):
    """Prereg §11, §7.2 磁盘与内存: the feasibility roots' sizes extrapolated to the formal run -- the per-subject products (HI's output root, atropos/, the stage's mm copies) x 34/2, the
    rest (the stage's pairs, the arms' cells, the evaluation) x 17/2 (review minor 3) -- plus 20 GiB within C:'s free space and /home's (no D:).  Stop, to BZ, otherwise."""
    p, r = places(), lc.roots(PAIR_SET)
    per_subject = tree_bytes(p['hi_out'])+tree_bytes(r.atropos)+tree_bytes(r.stage/lc.MM_DIR)
    per_pair = max(0, tree_bytes(p['root'])-tree_bytes(r.atropos)-tree_bytes(r.stage/lc.MM_DIR))
    need = (per_subject*SUBJECTS_FORMAL/SUBJECTS_FEAS+per_pair*PAIRS_FORMAL/PAIRS_FEAS)/hc.GIB
    free = hc.disk_reading(usage)
    record = dict(per_subject_gib=per_subject/hc.GIB, per_pair_gib=per_pair/hc.GIB, extrapolated_formal_gib=need, free_gib=free, floor_gib=hc.C_FLOOR_GIB,
                  rule='per subject x 34/2 + per pair x 17/2, + 20 GiB <= free(C:) and free(/home)')
    for side in ('c', 'home'):
        if free.get(side) is None or need+hc.C_FLOOR_GIB > free[side]:
            raise Stop('disk', f'{record}', to='BZ', kind='disk')
    return record


GIB_UNITS = dict(K=2**10, M=2**20, G=2**30, T=2**40)


def limit_bytes(text):
    """systemd's MemoryMax text ('10G', '512M', a number of bytes) in bytes."""
    text = str(text).strip()
    return int(float(text[:-1])*GIB_UNITS[text[-1]]) if text[-1:] in GIB_UNITS else int(text)


FEAS_ONLY = ('devcheck_a',)                    # LP-D (a) runs in the feasibility only: its own unit (lpdev_a.launch_line) is the one compared with
NOT_FORMAL = ('input_lock',)                   # the formal step 7 is the lead's command (hb_store.py lpba-input-lock), no unit; the chain's unit is a short CPU one
SPEC_FREE = ('--unit=', *LOG_PROPERTIES)       # what differs between the modes or is the chain's: the unit's name and the log (and --wait)


def unit_spec(step, pair_set, *, code):
    """Review minor 3: the spec of the step's unit in `pair_set` -- its start string's options (MemoryMax, MemorySwapMax, RuntimeMaxSec, CPUQuota, OOMPolicy, nice, the environment,
    the working directory, ...) without the unit's name, --wait and the log, sorted; its command with the pair set's name written '<pair_set>'."""
    argv = [str(a) for a in step.launch(pair_set, Path(code), log=unit_log(step))]
    k = next((i for i in range(2, len(argv)) if not argv[i].startswith('--')), len(argv))
    return dict(options=sorted(o for o in argv[2:k] if o != '--wait' and not o.startswith(SPEC_FREE)), command=['<pair_set>' if a == pair_set else a for a in argv[k:]])


def formal_specs(steps=STEPS, *, code=None):
    """{step: the formal unit's spec} (unit_spec in the formal mode lpba40_test; LP-D's (a) its own unit's; None for a step started from `argv`: the input lock)."""
    code = Path(lc.REPO if code is None else code)
    return {s.name: (None if s.launch is None else unit_spec(s, PAIR_SET if s.name in FEAS_ONLY else FORMAL_PAIR_SET, code=code)) for s in steps}


def spec_problems(steps, formal=None, *, code=None):
    """Review minor 3 (diff review minor 1 extended from MemoryMax to the whole unit): every step's feasibility unit has the formal unit's spec -- the same entry's start string in the
    two modes, equal but for the unit's name, the pair set and the log (the input lock excepted: no formal unit); [problems]."""
    code = Path(lc.REPO if code is None else code)
    formal = formal_specs(steps, code=code) if formal is None else formal
    problems = []
    for step in steps:
        if step.name in NOT_FORMAL:
            continue
        want = formal.get(step.name)
        if step.launch is None or want is None:
            problems.append(f"{step.name}: the formal unit's start string is not known in this code tree")
            continue
        mine = unit_spec(step, PAIR_SET, code=code)
        if mine != want:
            options = dict(feasibility=sorted(set(mine['options'])-set(want['options'])), formal=sorted(set(want['options'])-set(mine['options'])))
            command = None if mine['command'] == want['command'] else dict(feasibility=mine['command'], formal=want['command'])
            problems.append(f"{step.name}: the feasibility unit is not the formal unit's: options {options}, command {command}")
    return problems


def chain_unit(cgroup=None):
    """The tag-* unit /proc/self/cgroup shows this process in, or None (the diff review's I: the chain never runs inside a unit)."""
    try:
        text = Path(PROC_CGROUP if cgroup is None else cgroup).read_text(encoding='utf-8')
    except OSError as exc:
        return f'(unreadable: {exc})'
    found = TAG_UNIT.findall(text)
    return found[0] if found else None


def check_peak(step, memory_max=None):
    """Prereg §7.2 磁盘与内存 (review minor 3; diff review minor 1): the unit's end probe -- OOM evidence (memory.events oom or oom_kill > 0) -> stop, to BZ; the cgroup's memory.max not
    the unit's MemoryMax (integration review minor 4: the probe read another cgroup, or the limit did not apply) -> stop, to the lead; the peak x peak_scale at or above its MemoryMax
    without OOM evidence (memory.peak counts the page cache and is capped by the limit) -> recorded, stop, to the lead to look at; the peak, memory.max or the events unread -> stop,
    to the lead."""
    path = probe_path(step)
    probe = _json(path)
    peak = (probe or {}).get('peak_bytes')
    events = (probe or {}).get('events')
    probe_max = (probe or {}).get('max_bytes')
    if not isinstance(peak, (int, float)) or not isinstance(events, dict) or not isinstance(probe_max, (int, float)):
        raise Stop(step.name, f'{path}: the unit\'s peak, memory.max or memory.events cannot be read ({None if probe is None else probe.get("unread")})', kind='memory_unread')
    memory_max = step.memory_max if memory_max is None else memory_max
    limit = limit_bytes(memory_max)
    oom = {k: int(events.get(k) or 0) for k in ('oom', 'oom_kill')}
    record = dict(probe=str(path), peak_bytes=int(peak), peak_scale=step.peak_scale, extrapolated_bytes=int(peak*step.peak_scale), limit_bytes=limit, memory_max=memory_max,
                  probe_max_bytes=int(probe_max), events=oom)
    if any(oom.values()):
        raise Stop(step.name, f'{record}: OOM evidence in the unit (prereg §7.2: the formal run would reach its limit)', to='BZ', kind='memory_oom')
    if int(probe_max) != limit:
        raise Stop(step.name, f'{record}: the cgroup\'s memory.max is not the unit\'s MemoryMax {memory_max} (review minor 4)', kind='memory_max')
    if peak*step.peak_scale >= limit:
        raise Stop(step.name, f'{record}: the peak (page cache included) reached MemoryMax without OOM evidence: recorded, for the lead to look at', kind='memory_at_limit')
    return record


XB_UNIT_END_SCHEMA = 'bayesreg.xb.unit.end.v1'      # experiments/xb/xb_unit.py's unit_end.json (the in-unit supervisor of every HB and XB per-pair unit; a test compares)


def unit_ends():
    """Every unit_end.json under the feasibility root (the in-unit supervisor's records of the per-pair units LP-B's queues start)."""
    root = places()['root']
    return set() if not root.exists() else {str(p) for p in root.rglob('unit_end.json')}


def check_inner_units(step, before, n_pairs=None):
    """LP-C2 integration item 5 (prereg §7.2 磁盘与内存; diff review minor 1): the per-pair units the step's queue started -- the unit_end.json files that appeared under the feasibility
    root during the step -- each read as the chain's own units are: OOM evidence (oom or oom_kill > 0) in any of them -> stop, to BZ (read before any other stop of these units); its
    end probe's memory.max not the formal per-pair unit's MemoryMax (step.inner_memory) -> stop, to the lead; the peak at its limit without OOM evidence -> stop, to the lead; a peak, a
    limit or the events unread -> stop, to the lead.  Integration review minor 5: a unit_end.json that is not xb_unit.py's (another schema, not JSON) is skipped and recorded, and the
    records read must be one a pair (`n_pairs`, the mode's table) -> else stop, to the lead.  A step whose queue starts no unit (inner_memory None) has none to read."""
    new = sorted(unit_ends()-set(before))
    if step.inner_memory is None:
        return dict(units=[], unexpected=new) if new else None
    n_pairs = len(_rows()) if n_pairs is None else n_pairs
    want = limit_bytes(step.inner_memory())
    out, skipped, problems = [], [], []
    for path in new:
        doc = _json(path)
        if not isinstance(doc, dict) or doc.get('schema') != XB_UNIT_END_SCHEMA:
            skipped.append(dict(unit_end=path, schema=doc.get('schema') if isinstance(doc, dict) else None))
            continue
        probe = doc.get('probe') if isinstance(doc.get('probe'), dict) else {}
        peak, limit, events = probe.get('peak_bytes'), probe.get('max_bytes'), probe.get('events')
        if not isinstance(peak, (int, float)) or not isinstance(limit, (int, float)) or not isinstance(events, dict):
            problems.append(Stop(step.name, f'{path}: the unit\'s peak, MemoryMax or memory.events cannot be read ({probe.get("unread")})', kind='memory_unread'))
            continue
        oom = {k: int(events.get(k) or 0) for k in ('oom', 'oom_kill')}
        record = dict(unit_end=path, unit=doc.get('unit'), peak_bytes=int(peak), limit_bytes=int(limit), formal_limit_bytes=want, events=oom)
        out.append(record)
        if any(oom.values()):
            raise Stop(step.name, f'{record}: OOM evidence in a per-pair unit (prereg §7.2: the formal run would reach its limit)', to='BZ', kind='memory_oom')
        if int(limit) != want:
            problems.append(Stop(step.name, f'{record}: the unit\'s MemoryMax is not the formal per-pair unit\'s {step.inner_memory()}', kind='memory_max'))
        elif peak >= limit:
            problems.append(Stop(step.name, f'{record}: a per-pair unit\'s peak (page cache included) reached MemoryMax without OOM evidence: recorded, for the lead to look at',
                                 kind='memory_at_limit'))
    if problems:
        raise problems[0]
    if len(out) != n_pairs:
        raise Stop(step.name, f'{len(out)} per-pair unit records ({XB_UNIT_END_SCHEMA}) read, {n_pairs} pairs: one unit a pair (review minor 5); skipped {skipped}', kind='inner_count')
    return dict(units=out, skipped=skipped, formal_memory_max=step.inner_memory())


CHECKS = dict(prep=check_prep, devcheck_a=check_devcheck_a, hi=check_hi, atropos=check_atropos, method=check_method, review=check_review, input_lock=check_input_lock, eval=check_eval,
              **{name: (lambda name=name: check_arm(name)) for name in ('configc', 'ants_s', 'fireants_greedy', 'fireants_syn', 'convexadam_v4seg')})


def check_of(step, checks=None):
    """The step's check; a step without one stops the chain (fail closed)."""
    checks = CHECKS if checks is None else checks
    if step.name in checks:
        return checks[step.name]

    def none():
        raise Stop(step.name, f'{step.name} has no check of its own in this chain yet (its tool and records are not in the code tree)', kind='no_check')
    return none


# ---- the chain --------------------------------------------------------------------------------------------------------------------------------------------
def gate(step, *, machine=None, wait=None, log=print):
    """Before each unit (prereg §8.7): the hold file (a stop), the machine gate through holdout_common.wait_for (no other tag-* / seq-vp-* unit than the chain's own; memory, load, the 10 s
    gap; polled until it passes, InfrastructureFailure(precheck_timeout) after 15 minutes: review I1), the disk guard."""
    machine = hc.machine_gate if machine is None else machine
    hold = lc.check_hold(PAIR_SET)
    passed = hc.wait_for(lambda: machine(own_units=()), log=log, **(wait or {}))      # the chain is in no unit (chain_unit): nothing exempted
    hold_after = lc.check_hold(PAIR_SET)                                             # diff review minor 2: the hold file again after the wait
    return dict(hold=hold, machine=passed, hold_after_wait=hold_after, disk=lc.disk_guard(PAIR_SET, NEED_HOME_GIB))


def start_code(code, code_identity=None):
    """The chain's code at its start (C26.9-1, -3): holdout_common.code_identity of the code tree -- clean, its HEAD the commit every step's record must name."""
    code_identity = hc.code_identity if code_identity is None else code_identity
    record = code_identity(repo=code)
    problem = None if record.get('clean') and record.get('commit') else f'the code tree is not clean or has no commit ({record}): the feasibility pairs run from a clean checkout (C26.9-1)'
    return record, problem


def stop_record(exc):
    return dict(step=exc.step, reason=exc.reason, to=exc.to, kind=exc.kind)


def probe_on_exit(step, rc):
    """Integration review minor 1: after a unit whose exit is not 0, its own end probe as far as it was written (a unit killed by its limit may have none): OOM evidence (memory.events
    oom or oom_kill > 0) -> stop, to BZ, as for a per-pair unit; otherwise the reading is recorded and the exit's stop follows."""
    path = probe_path(step)
    probe = _json(path)
    events = (probe or {}).get('events')
    if not isinstance(events, dict):
        return dict(probe=str(path), read=False, unread=None if probe is None else probe.get('unread'))
    oom = {k: int(events.get(k) or 0) for k in ('oom', 'oom_kill')}
    record = dict(probe=str(path), read=True, peak_bytes=probe.get('peak_bytes'), max_bytes=probe.get('max_bytes'), events=oom)
    if any(oom.values()):
        raise Stop(step.name, f'{record}: OOM evidence in the unit, which ended with exit {rc} (prereg §7.2: the formal run would reach its limit)', to='BZ', kind='memory_oom')
    return record


def run_step(step, row, *, code, head, runner, machine, wait, usage, checks, log, code_identity=None):
    """One step: the gate, the unit, holdout_common.note_unit_end; the per-pair units its queue started (integration item 5: OOM evidence there, to BZ, is the stop at once; their other
    stops come after the step's own check, kept in row['inner_stop'] and named in an exit's stop: integration review minor 1); after an exit that is not 0 the unit's own end probe
    (OOM evidence there: to BZ at once; review minor 1); then its check -- run also after an exit that is not 0, and a stop it hands to BZ is the stop (review I2) --; the code commit
    its records name (review minor 4); the unit's peak (review minor 3); the disk."""
    row['gate'] = gate(step, machine=machine, wait=wait, log=log)
    argv = unit_argv(step, code=code)
    row['argv'] = argv
    row['memory_max'] = memory = memory_max_of(step, code=code)
    before = unit_ends()
    log(f'{step.name}: {shlex.join(argv)}')
    try:
        row['unit'] = runner(argv)
    finally:
        row['unit_end'] = hc.note_unit_end(lc.unit_name(PAIR_SET, step.tool))
    rc = row['unit'].get('returncode')
    inner_stop = None
    try:
        row['inner_units'] = check_inner_units(step, before)
    except Stop as exc:
        row['inner_units'] = dict(stop=stop_record(exc))                 # review minor 1: written whatever the stop
        if exc.to == 'BZ':                                               # a per-pair unit's OOM evidence: to BZ before the queue's own exit or records
            raise
        inner_stop = exc
        row['inner_stop'] = stop_record(exc)
    also = '' if inner_stop is None else f'; the per-pair units: {inner_stop.kind}: {inner_stop.reason}'
    if rc != 0:
        row['probe_on_exit'] = probe_on_exit(step, rc)                   # review minor 1: the unit's own OOM evidence goes to BZ, as a per-pair unit's does
    try:
        row['check'] = check_of(step, checks)()
    except Stop as exc:
        if rc != 0 and exc.to != 'BZ':
            raise Stop(step.name, f'the unit ended with exit {rc}; its check: {exc.reason}{also}', kind='exit') from None
        raise
    except Exception as exc:
        if rc != 0:                                                      # diff review minor 3: the exit code stays in the stop record
            raise ChainError(f'the unit ended with exit {rc}; its check raised {type(exc).__name__}: {exc}{also}') from exc
        raise
    if rc != 0:
        raise Stop(step.name, f'the unit ended with exit {rc}{also}', kind='exit')
    if inner_stop is not None:
        raise inner_stop
    commit = row['check'].get('code_commit') if isinstance(row['check'], dict) else None
    if step.commit_from == 'tree':                                       # integration item 5: a record that names no commit (the input lock): the code tree's HEAD after the unit, clean
        tree, problem = start_code(code, code_identity)
        row['tree_after'] = tree
        if problem:
            raise Stop(step.name, problem, kind='code_commit')
        commit = tree.get('commit')
    row['code_commit'] = commit
    if commit != head:
        raise Stop(step.name, f'its record names the code commit {commit!r}, not the chain\'s HEAD {head} (C26.9-3: the feasibility pairs on one commit)', kind='code_commit')
    row['peak'] = check_peak(step, memory)
    row['disk'] = check_disk(usage)


def run_chain(*, steps=STEPS, code=None, runner=run_unit, machine=None, wait=None, usage=None, checks=None, code_identity=None, cgroup=None, formal=None, log=print):
    """The chain (module docstring).  Returns (exit code, record).  `runner(argv)` starts one unit and waits (the tests give one that writes records); `wait`: holdout_common.wait_for's
    seconds, poll, now, sleep (the tests' clock); `checks` the steps' checks (CHECKS); `code_identity` holdout_common.code_identity; `formal` the formal units' specs (formal_specs)."""
    code = Path(lc.REPO if code is None else code)
    p = places()
    record = dict(schema=SCHEMA, pair_set=PAIR_SET, code=str(code), started_utc=lc.utc(), steps=[], places={k: str(v) for k, v in p.items()})
    inside = chain_unit(cgroup)
    if inside is not None:                                               # the diff review's I (a): never inside a unit; exit 8, nothing written
        record.update(refused=dict(unit=inside, rule='the chain runs from a shell or setsid, never inside a tag-* unit (its entries\' machine gates would refuse)'))
        log(f'refused: this process is inside the unit {inside}: run the chain from a shell or with setsid')
        return 8, record
    missing = missing_commands(steps)
    problem = roots_problem()
    identity, code_problem = start_code(code, code_identity)
    record['code_identity'] = identity
    spec = spec_problems([s for s in steps if s.argv is not None or s.launch is not None], formal, code=code)
    if missing or problem or code_problem or spec:
        record.update(refused=dict(missing_commands=missing, roots=problem, code=code_problem, unit_spec=spec))
        log(f'refused: commands missing {missing}; roots {problem}; code {code_problem}; unit spec {spec}')
        return 2, record
    head = identity['commit']
    lc.guard_write(p['chain'])
    p['chain'].mkdir(parents=True)
    stop = None
    for step in steps:
        row = dict(step=step.name, unit=lc.unit_name(PAIR_SET, step.tool), started_utc=lc.utc())
        record['steps'].append(row)
        try:
            run_step(step, row, code=code, head=head, runner=runner, machine=machine, wait=wait, usage=usage, checks=checks, log=log, code_identity=code_identity)
        except Stop as exc:
            stop = dict(step=exc.step, reason=exc.reason, to=exc.to, kind=exc.kind, utc=lc.utc())
        except hc.InfrastructureFailure as exc:
            stop = dict(step=step.name, reason=f'{type(exc).__name__}: {exc}', to='lead (tool error first; none found: BZ)', kind=exc.kind, utc=lc.utc())
        except hc.Refused as exc:
            stop = dict(step=step.name, reason=f'{type(exc).__name__}: {exc}', to='lead (tool error first; none found: BZ)', kind=exc.stage, utc=lc.utc())
        except Exception as exc:                                         # noqa: BLE001 - review minor 2: the chain's own exception is a stop with its record too
            import traceback
            stop = dict(step=step.name, reason=f'{type(exc).__name__}: {exc}', to='lead (tool error first; none found: BZ)', kind='exception', traceback=traceback.format_exc(),
                        utc=lc.utc())
        row['ended_utc'] = lc.utc()
        if stop is not None:
            break
    record.update(stop=stop, passed=stop is None, ended_utc=lc.utc())
    if stop is not None:
        hc.write_json_once(p['chain']/'chain_stopped.json', stop)
    hc.write_json_once(p['chain']/'chain.json', record)
    log(f"feasibility chain: {'passed' if stop is None else 'stopped at ' + stop['step'] + ' (' + stop['to'] + '): ' + stop['reason']}")
    return (0 if stop is None else 4), record


def main(argv=None, *, out=print):
    ap = argparse.ArgumentParser(prog='experiments/lpba/feas_chain.py')
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('check')
    sub.add_parser('run')
    r = sub.add_parser('retire')
    r.add_argument('--tag', required=True)
    a = ap.parse_args(argv)
    if a.command == 'check':
        missing, problem = missing_commands(), roots_problem()
        out(f'commands missing: {missing or "none"}; roots: {problem or "new"}')
        return 0 if not missing and not problem else 2
    if a.command == 'retire':
        out(json.dumps(retire(a.tag)))
        return 0
    if not sys.flags.dont_write_bytecode:
        out('refused: run with python -B (or PYTHONDONTWRITEBYTECODE)')
        return 2
    code, _ = run_chain(log=out)
    return code


if __name__ == '__main__':
    sys.exit(main())
