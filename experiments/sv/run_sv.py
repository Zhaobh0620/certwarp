"""SV (docs/bayesreg/notes/提速与换预分割器规划_20261104.md §7; runs/20261104-sv/plan.md, revision 1): how much final Dice does a better segmenter buy?  The S_P pipeline (K = 35 entry, then the frozen native level, 330 s with a charge of 47 s) on label maps between the prediction and the truth.
An ORACLE probe on the first 5 development pairs: `oracle: true` in every record, an isolated root segvoi_v1, in no result table, not a method.

The arms (entry channels and colouring both from the arm's labels; the rule of the maps: postmix.py; sdfmix.py stays for the probe's comparison only):
    r1_3, r2_3   L_f of every subject, with the one f* that reduces the mean (1 - seg35) of the 6 subjects by 1/3, 2/3: the first 5 pairs, 10 runs.  They enter the preregistered reading (decide_sv.py).  f* comes from the CPU probe (probe_sv.py) and is pinned by the lead in configs/bayesreg/sv_fstar_v1.json.
    f0440        only subject 0440 takes the truth, the other subject of the pair its prediction: 0439-0440 and 0440-0441, 2 runs.  Description only: it starts after the reading is written, and its failure is only recorded.
    f0control    the prediction on both sides, the entry of 0438-0439 only: against mc0p_v1 Q2 long bit for bit, before the two hours (it proves the entry differs from S_P's in the labels only).
    The endpoints f = 0 (S_P) and f = 1 (S_T) are read from seed_v1, not run again.

    python -B experiments/sv/run_sv.py --dry-run [--scratch DIR]
        prints all the runs, their command lines, the real frozen parser and prepare on each (a scratch copy of the frozen source, A5P's inputs in a stage of the 5 pairs), the protocol against S_P's, a rehearsal of the reading on copies of the endpoints.
        No solver, no GPU, no unit, nothing written outside the scratch directory; no truth label is decoded.
    run_sv.py probe --root DIR [--workers 1]         the lead's CPU step before anything else, in the unit printed by the dry run: analysis/probe_posterior.json and .md, no label map
    run_sv.py pin-fstar --root DIR --by "WHO, WHEN"  writes configs/bayesreg/sv_fstar_v1.json from the probe (add only); the lead commits it and adds it to the plan's revision
    python -B experiments/sv/run_sv.py queue [--root DIR] [--wait-minutes 30] [--accept-control-difference "WHO, WHY"]
        the lead's start, in the unit printed by the dry run.  Start checks first: a missing input, an f* not pinned, or a probe that does not back it stops before any run.  Code: a detached worktree $ROOT/code of HEAD (tracked files clean).  Then, serial, each behind the gate (no $ROOT/hold,
        no other tag-* or seq-vp-* unit, 1-minute load < 1.5, no GPU compute process, GPU utilization <= 20%, GPU free >= 15 GiB, MemAvailable >= 7 GiB, /home free >= 100 GiB, /mnt/c free >= 15 GiB; closed for --wait-minutes: STOP) with 60 s between units:
        tag-sv-build (labels, inputs, and the actual reduction against the target: more than .03 off stops before any GPU; CPU), tag-sv-k1check (the K = 1 check of MC0P on the GPU), tag-sv-entry-f0control-0438-0439 (the control), tag-sv-entry-<arm>-<ffff>-<mmmm> (10 entries),
        tag-sv-entryeval (their taper, ladder and metrics; CPU), tag-sv-<arm>-<ffff>-<mmmm> (10 native runs, each evaluated), the reading.  Then f0440: 2 entries, tag-sv-entryeval-f0440, 2 native runs, and the summary again; a stop there is only recorded (f0440_stop.json), the reading does not change.
        A directory that exists is never rerun; failures are kept.  STOP on: a failed build, a failed K = 1 check, a control that differs, an entry that is not `done`, a failed entry evaluation, the stops of SEED (protocol §16.5, §16.6) on a native run.  After a STOP among the native runs, what finished is summed up.
    run_sv.py build --root DIR                       the labels and the inputs; with the probe the only place the truth members of a pair.npz are read for anything but the metrics
    run_sv.py run-entry --arm ARM --pair PAIR --root DIR [--device cuda] [--code-sha SHA]       one entry: field.npz, run.json
    run_sv.py evaluate-entries --root DIR            every entry without metrics.json: as run_mc0p.py evaluate, the truth labels for the metrics only
    run_sv.py evaluate-one RUN --arm ARM --pair PAIR --run-id ID                one native run: experiments/d0/evaluate.py evaluate_run, plus the record of the entry
    run_sv.py decide --root DIR                      analysis/decision_sv.json, summary_sv.md, long.csv

probe, build, run-entry and evaluate-entries need PYTHONPATH=<frozen>:<code> (the units set it).  The only root is segvoi_v1, outside the repository: everything that writes refuses any other.  Exit codes: 0 done, 2 usage, another driver on the root or another root, 8 STOP or a failed check.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in ('experiments/d0', 'experiments/seed', 'experiments/mc0', 'experiments/sv'):
    if str(ROOT/directory) not in sys.path:
        sys.path.insert(0, str(ROOT/directory))

import numpy as np  # noqa: E402

import mc0_core as core  # noqa: E402    # read only, experiments/mc0
import mc0p_core as mc  # noqa: E402
import postmix  # noqa: E402
import run_mc0 as run  # noqa: E402
import run_mc0p as mc0p  # noqa: E402
import run_seed as seed  # noqa: E402    # read only, experiments/seed
import sdfmix  # noqa: E402

SCRIPT = 'experiments/sv/run_sv.py'
RESULTS = seed.RESULTS
ROOT_NAME = 'segvoi_v1'
DEFAULT_ROOT = RESULTS/ROOT_NAME
QUEUE_UNIT, BUILD_UNIT, K1_UNIT, EVAL_UNIT, EVAL_UNIT_F0440, PROBE_UNIT = 'tag-sv-queue', 'tag-sv-build', 'tag-sv-k1check', 'tag-sv-entryeval', 'tag-sv-entryeval-f0440', 'tag-sv-probe'
ROLE = 'oracle_probe_of_the_value_of_a_better_segmenter_NOT_a_method_NOT_in_any_result_table'
PAIRS = core.PAIRS                                     # the first 5 of selection.json; nothing else runs
SUBJECTS = tuple(f'OASIS_OAS1_{n:04d}_MR1' for n in range(438, 444))    # their 6 subjects: the only subjects this package reads a label of
HARD_SUBJECT = 'OASIS_OAS1_0440_MR1'                   # the lead: the worst-segmented subject (.8638 / .9062), the two pairs carry the gap of S_P to .86
TIER, ENTRY_FORMULA = 'long', 'Q2'                     # Q2 long of MC0P: K = 35, the smoothed one-hot of each label 1..35, the scale rule against the colouring of the same labels
PROVENANCE_PREFIX = 'oracle_postmix_'
TAG_PRED, TAG_TRUTH = 'pred', 'truth'                  # f = 0 and f > 1/2 of the mix: each is checked bit for bit against its source on the real files
TARGETS = dict(r1_3=1/3, r2_3=2/3)                     # the reduction of the mean (1 - seg35) of the 6 subjects that names each arm
TARGET_TOL = .03                                       # the actual reduction, more than this off the target: nothing on the GPU starts
ARMS = dict(r1_3=dict(target=TARGETS['r1_3'], pairs=PAIRS, decides=True, provenance=PROVENANCE_PREFIX+'r1_3'),
            r2_3=dict(target=TARGETS['r2_3'], pairs=PAIRS, decides=True, provenance=PROVENANCE_PREFIX+'r2_3'),
            f0440=dict(target=None, pairs=(PAIRS[1], PAIRS[2]), decides=False, provenance=PROVENANCE_PREFIX+'f0440'))
READING_ARMS = tuple(arm for arm, spec in ARMS.items() if spec['decides'])
DESCRIPTION_ARMS = tuple(arm for arm, spec in ARMS.items() if not spec['decides'])
CONTROL_ARM, CONTROL_PAIR = 'f0control', PAIRS[0]      # an entry only, not an arm of the reading or the description
PINS_FILE = ROOT/'configs/bayesreg/sv_fstar_v1.json'   # add only; written by pin-fstar, committed by the lead
PINS_SCHEMA = 'bayesreg.sv.fstar.v1'
PROBE_JSON, PROBE_MD = 'probe_posterior.json', 'probe_posterior.md'
BUILD_QUOTA, BUILD_RUNTIME = 200, 2*3600
PROBE_QUOTA, PROBE_NICE, PROBE_RUNTIME = 100, 19, 2*3600
MEMORY_MAX, NICE = '8G', 5                             # MC0P's K = 35 unit
RESERVE_MIN_MEM_GIB, GPU_FREE_MIB, HOME_FLOOR_GIB, HOST_FLOOR_GIB, LOAD_LIMIT = 7., 15*1024, 100, 15, 1.5     # the gate of the brief (§4); MemAvailable as the strict gate of mc0p_v1b
BETWEEN_RUNS_SECONDS = 60
Stop = seed.Stop
EntryError, EntryMissing = seed.EntryError, seed.EntryMissing


class GateClosed(Stop):
    """The gate stayed closed for its time."""


read_json, write_json, sha_file, short_name, utc = seed.read_json, seed.write_json, seed.sha_file, seed.short_name, seed.utc


@dataclass(frozen=True)
class Sources:
    """Where the inputs are; only tests pass another one."""
    results: Path = RESULTS
    stage: Path = core.STAGE
    pred_root: Path = RESULTS/'seg/unet35_v1/pred'     # mc0p_core.default_unet: the predictions of A5P's images
    seg_root: Path = mc.SEG_ROOT
    allowlist: Path = core.ALLOWLIST
    pins: Path = PINS_FILE


@dataclass(frozen=True)
class EntryCell(core.Cell):
    @property
    def run_id(self):
        return f'sv_entry_{self.arm}_{self.short}'

    @property
    def unit(self):
        return f'tag-sv-entry-{self.arm}-{self.short}'


# ---- what may be read and written ------------------------------------------------------------------------------------------------------------------------------------------------------------------
def require_pair(pair):
    """Only the first 5 development pairs: checked before any file of the pair is opened."""
    if pair not in PAIRS:
        raise SystemExit(f'SV runs only the first 5 development pairs {[short_name(p) for p in PAIRS]}; refused: {pair}')
    return pair


def require_subject(subject):
    """Only the 6 subjects of those pairs: the truth labels of no other subject enter this package (hard constraints 6, 7)."""
    if subject not in SUBJECTS:
        raise SystemExit(f'SV reads labels only of {SUBJECTS}; refused: {subject}')
    return subject


def require_cell(arm, pair):
    """Only the planned cells: checked before any file of the pair is opened."""
    if arm == CONTROL_ARM:
        if pair != CONTROL_PAIR:
            raise SystemExit(f'{CONTROL_ARM} runs only on {short_name(CONTROL_PAIR)}; refused: {pair}')
        return arm, pair
    if arm not in ARMS:
        raise SystemExit(f'SV has no arm {arm}; only {tuple(ARMS)}')
    if pair not in ARMS[arm]['pairs']:
        raise SystemExit(f"SV runs {arm} only on {[short_name(p) for p in ARMS[arm]['pairs']]}; refused: {pair}")
    return arm, pair


def subjects_of(pair):
    require_pair(pair)
    fixed, moving = pair.split('__')
    return require_subject(fixed), require_subject(moving)


def default_root(results=None):
    return Path(RESULTS if results is None else results)/ROOT_NAME


def inside(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    return path == parent or parent in path.parents


def root_problem(root, results=None):
    """Why this package may not write at this root, or None: only its isolated root, never inside a repository checkout."""
    if Path(root).resolve() != default_root(results).resolve():
        return f'{root}: this package writes only to {default_root(results)}; nothing read, nothing moved'
    if inside(root, ROOT) or inside(root, seed.MAIN_REPO):
        return f'{root}: inside the repository; these labels never go into git'
    return None


def label_tag(arm, subject):
    """The name of the label map this arm takes for this subject: r1_3, r2_3 for all; in f0440 `truth` for 0440 alone and `pred` for every other subject; `pred` in the control."""
    require_subject(subject)
    if arm == CONTROL_ARM:
        return TAG_PRED
    if arm not in ARMS:
        raise SystemExit(f'SV has no arm {arm}; only {tuple(ARMS)}')
    if arm in TARGETS:
        return arm
    return TAG_TRUTH if subject == HARD_SUBJECT else TAG_PRED


def side_provenance(the_tag):
    """oracle_postmix_<tag>.  The firewall of the method refuses it (bayesreg.io.pair.FORBIDDEN_ID): it can never pass for a registered segmenter."""
    return PROVENANCE_PREFIX+the_tag


def tags_needed(subject):
    """The maps of a subject the arms read: only these are written."""
    cells = [(arm, pair) for arm, spec in ARMS.items() for pair in spec['pairs']]+[(CONTROL_ARM, CONTROL_PAIR)]
    return sorted({label_tag(arm, subject) for arm, pair in cells if subject in subjects_of(pair)})


def label_path(root, subject, the_tag, results=None):
    """The one place an L_f is written: the isolated root.  Refuses any other root and any other subject."""
    problem = root_problem(root, results)
    if problem:
        raise SystemExit(problem)
    return Path(root)/'labels'/require_subject(subject)/f'L_{the_tag}.npz'


def entry_dir(root, arm, pair):
    require_cell(arm, pair)
    return Path(root)/'entries'/arm/short_name(pair)


def run_dir(root, arm, pair):
    require_cell(arm, pair)
    return Path(root)/'runs'/arm/pair


def inputs_root(arm, root):
    return Path(root)/'inputs'/arm


def plan_cells(arms=None):
    """(arm, pair) of the given arms, or of all: the arms of the reading first, the description last."""
    return [(arm, pair) for arm, spec in ARMS.items() for pair in spec['pairs'] if arms is None or arm in arms]


def unit_name(arm, pair):
    return f'tag-sv-{arm}-{short_name(pair)}'


def new_run_id(arm, pair):
    return f"sv_{arm}_{short_name(pair)}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def method_name(arm):
    return f"frozen_eb808be_seed_{seed.SECONDS}s_gw{'_'.join(f'{x:g}' for x in seed.GEOMETRY)}_entry{ENTRY_FORMULA}long_{ARMS[arm]['provenance']}"


def b1_entry(arm, root):
    return seed.b1_spec.check_entry(dict(arm=arm, inputs_root=str(inputs_root(arm, root)), seconds=seed.SECONDS, changes=seed.CHANGES))


# ---- the pinned f* --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def load_pins(path=None):
    """The pinned document, None without the file."""
    return read_json(PINS_FILE if path is None else path)


def pin_value_problems(pins):
    """What is wrong with the pinned values themselves."""
    if not isinstance(pins, dict) or pins.get('schema') != PINS_SCHEMA or pins.get('oracle') is not True:
        return [f'the pins: schema {pins.get("schema") if isinstance(pins, dict) else pins}, expected {PINS_SCHEMA}, oracle true']
    arms = pins.get('arms') or {}
    if set(arms) != set(TARGETS):
        return [f'the pins name {sorted(arms)}, expected {sorted(TARGETS)}']
    bad = []
    for arm, target in TARGETS.items():
        f = arms[arm].get('f')
        if isinstance(f, bool) or not isinstance(f, (int, float)) or not math.isfinite(f) or not 0 < f < .5:
            bad.append(f'the pin of {arm}: f = {f!r}, must lie in (0, 1/2)')
        if arms[arm].get('target') != target:
            bad.append(f"the pin of {arm}: target {arms[arm].get('target')}, expected {target}")
    if not bad and not arms['r1_3']['f'] < arms['r2_3']['f']:
        bad.append(f"the pins: f of r1_3 {arms['r1_3']['f']} must be below f of r2_3 {arms['r2_3']['f']}")
    return bad


def fstar_problems(root, sources):
    """Why no GPU may start: an f* not pinned, or pinned without a probe that shows it within TARGET_TOL of its target.  The build checks it once more on the maps it writes."""
    pins = load_pins(sources.pins)
    if pins is None:
        return [f'f* 未钉：{sources.pins} 不存在（先跑探针 probe，再 pin-fstar，提交，写进 plan.md 的修订）']
    bad = pin_value_problems(pins)
    if bad:
        return bad
    path = Path(root)/'analysis'/PROBE_JSON
    if not path.is_file():
        return [f'{path}: 探针的结果不存在，钉住的 f* 没有依据']
    if sha_file(path) != pins.get('probe_sha256'):
        bad.append(f"{path}: sha256 {sha_file(path)} != the pins' probe_sha256 {pins.get('probe_sha256')}")
    probe = json.loads(path.read_text())
    rows = probe.get('rows') or {}
    for arm, target in TARGETS.items():
        star = ((probe.get('fstar') or {}).get(arm) or {}).get('f')
        if pins['arms'][arm]['f'] != star:
            bad.append(f"{arm}: the pin {pins['arms'][arm]['f']} is not the f* of the probe, {star}")
        row = rows.get(f"{pins['arms'][arm]['f']:.4f}")
        reduction = None if row is None else (row.get('reduction_mean') or {}).get('seg35')
        if reduction is None:
            bad.append(f"the probe has no row for f = {pins['arms'][arm]['f']} of {arm}")
        elif abs(reduction-target) > TARGET_TOL:
            bad.append(f"{arm}: the probe's reduction {reduction:.4f} at f = {pins['arms'][arm]['f']} is more than {TARGET_TOL} off the target {target:.4f}")
    return bad


def pin_fstar(root, by, *, sources=None, log=print):
    """Writes the pins from the probe.  Add only: an existing file stays."""
    sources = Sources() if sources is None else sources
    problem = root_problem(root, sources.results)
    if problem:
        raise SystemExit(problem)
    path = Path(root)/'analysis'/PROBE_JSON
    probe = read_json(path)
    if probe is None:
        raise SystemExit(f'{path}: no probe')
    if Path(sources.pins).exists():
        raise SystemExit(f'{sources.pins} exists; add only: the code reads this path alone, so to change a pin the lead renames the file (it stays) and writes a revision of the plan')
    fstar = probe.get('fstar') or {}
    missing = {arm: (fstar.get(arm) or {}).get('reason') for arm in TARGETS if (fstar.get(arm) or {}).get('f') is None}
    if missing:
        raise SystemExit(f'the probe found no f* for {missing}: nothing written; the plan says what the lead decides then (revision 2)')
    off = {arm: fstar[arm].get('reduction') for arm, target in TARGETS.items() if fstar[arm].get('reduction') is None or abs(fstar[arm]['reduction']-target) > TARGET_TOL}
    if off:
        raise SystemExit(f'the reduction at f* is more than {TARGET_TOL} off the target: {off}; nothing written')
    pins = dict(schema=PINS_SCHEMA, oracle=True, pinned_by=by, pinned_utc=utc(), probe_sha256=sha_file(path), probe_file=str(path), role=ROLE, tolerance=TARGET_TOL,
                arms={arm: dict(f=fstar[arm]['f'], target=target, reduction_seg35_in_probe=fstar[arm]['reduction']) for arm, target in TARGETS.items()})
    bad = pin_value_problems(pins)
    if bad:
        raise SystemExit('; '.join(bad))
    write_json(sources.pins, pins)
    log(f'{sources.pins}: written; commit it and add the two f* to the revision of runs/20261104-sv/plan.md')
    return 0


# ---- start strings ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def task_unit_argv(unit, code, inner, *, log, gpu=True, numba=1, quota=800, runtime=run.RUN_RUNTIME_MAX):
    return run.unit_argv(unit, code, inner, log=log, gpu=gpu, numba=numba, quota=quota, runtime=runtime, script=SCRIPT, memory_max=MEMORY_MAX, nice=NICE)


def probe_argv(repo=seed.MAIN_REPO, root=None, workers=1):
    """The lead's start string of the CPU probe: one core, nice 19, no GPU, from the main checkout (it comes before the queue and its worktree)."""
    root = DEFAULT_ROOT if root is None else root
    return run.unit_argv(PROBE_UNIT, repo, ['probe', '--root', str(root), '--workers', str(workers)], log=f'{root}/probe.log', gpu=False, numba=1, quota=PROBE_QUOTA*workers, runtime=PROBE_RUNTIME, script=SCRIPT, memory_max=MEMORY_MAX, nice=PROBE_NICE)


def build_argv(code, root):
    return task_unit_argv(BUILD_UNIT, code, ['build', '--root', str(root)], log=f'{root}/build.log', gpu=False, quota=BUILD_QUOTA, runtime=BUILD_RUNTIME)


def k1_argv(code, root):
    return mc0p.k1_argv(code, root, K1_UNIT)


def entry_argv(cell, code, root, sha='n/a'):
    return task_unit_argv(cell.unit, code, ['run-entry', '--arm', cell.arm, '--pair', cell.pair, '--root', str(root), '--code-sha', sha], log=entry_dir(root, cell.arm, cell.pair)/'log.txt')


def entry_eval_argv(code, root, unit=EVAL_UNIT):
    return task_unit_argv(unit, code, ['evaluate-entries', '--root', str(root)], log=f'{root}/entry_eval.log', gpu=False, numba=mc0p.EVAL_THREADS, quota=mc0p.EVAL_QUOTA, runtime=run.EVAL_RUNTIME_MAX)


def prepare_argv(arm, pair, run_path, results=None):
    """The prepare line of S_P with the stage's name of this arm: the options are run_seed.OPTS."""
    require_cell(arm, pair)
    return seed.check_launch([seed.PY, '-B', seed.frozen_dir(results)/'scripts/run_stage_3d_e48_pg.py', 'prepare', run_path, *seed.OPTS, '--stage', f'SV-{arm}', '--pair', pair])


def unit_argv(arm, pair, run_path, root):
    """The unit of S_P with this arm's name: resources, watchdog, the run's own copy of inject_run.py."""
    require_cell(arm, pair)
    return seed.unit_argv(arm, pair, run_path, root, unit=unit_name(arm, pair))


def queue_launch(root=None, repo=seed.MAIN_REPO):
    """The lead's start string of the queue."""
    root = DEFAULT_ROOT if root is None else root
    return seed.check_launch(['systemd-run', '--user', f'--unit={QUEUE_UNIT}', '--collect', f'--working-directory={repo}', '--property=MemoryMax=9G', '--setenv=PYTHONDONTWRITEBYTECODE=1',
                              '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1', '--setenv=MKL_NUM_THREADS=1', '--setenv=NUMEXPR_NUM_THREADS=1',
                              seed.PY, '-B', SCRIPT, 'queue', '--root', str(root)])


# ---- the subjects: posterior, truth ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def read_images(pair, sources=None):
    """The firewall's view of a planned pair: load_pair, `fixed` and `moving`; subject guard, sha256 against the manifest."""
    sources = Sources() if sources is None else sources
    return mc.read_images(require_pair(pair), stage=Path(sources.stage), allowlist=sources.allowlist)


def read_stage_pair(pair, stage=None):
    """(all the members of the pair.npz, its manifest): the one function of this package that opens the truth members for anything but the metrics.  Only a planned pair; never called with another subject."""
    d = Path(core.STAGE if stage is None else stage)/require_pair(pair)
    manifest = json.loads((d/'manifest.json').read_text())
    if (manifest.get('fixed_subject'), manifest.get('moving_subject')) != subjects_of(pair):
        raise SystemExit(f"{pair}: the manifest names {manifest.get('fixed_subject')}, {manifest.get('moving_subject')}")
    if sha_file(d/'pair.npz') != manifest['pair_sha256']:
        raise SystemExit(f'{pair}: pair.npz sha256 differs from its manifest')
    mi = core.make_inputs()
    with np.load(d/'pair.npz') as z:
        if tuple(z.files) != mi.KEY_ORDER:
            raise SystemExit(f'{pair}: unexpected members {z.files}')
        return {k: np.array(z[k]) for k in z.files}, manifest


def read_subject_posterior(pin, side):
    """(P (36, nz, ny, nx) float64, record) of one image: the checks of mc0p_core.read_predicted in its order (D24 admission, StructureLayer, read-back, group sums, alignment QC; make_inputs, read only), which drops P after its argmax.  STOP otherwise.  It takes the PairInput, which holds no label."""
    mi = core.make_inputs()
    unet = mc.default_unet()
    subject = getattr(pin, side+'_subject')
    admission = mi.a4_admission([pin.fixed_subject, pin.moving_subject], core.ALLOWLIST, unet)
    layer = mi.pair_structure_layer(pin, core.ALLOWLIST, unet)
    P, record, u_wm = mi.unet_posterior(layer, side)
    support = mi.support_dice_record(getattr(pin, side), P[0])
    a_wm, atropos = mi.atropos_wm(mc.SEG_ROOT, pin, side)
    qc = mi.alignment_qc(u_wm, a_wm)
    del u_wm, a_wm
    if not qc['passed']:
        raise SystemExit(f"SV {subject}: alignment QC {qc['dice']} < {qc['threshold']} (protocol §12.4 item 5)")
    return P, dict(record, subject=subject, alignment_qc=qc, support_dice_record=support, atropos_qc=atropos, subject_admission=admission, segmenter=unet.record(), label_firewall=pin.provenance['label_firewall'], pair_keys_read=list(pin.provenance['keys_read']))


def read_subjects(sources, reader, log):
    """{subject: dict(reduced, truth4, record)} of the 6 subjects: each subject's posterior is read once, at its first pair, and reduced at once, so that one stack at most is in memory.  The truth of each subject in its two pairs must agree.  reader: (pin, side) -> (P, record)."""
    held = {}
    for pair in PAIRS:
        arrays, _ = read_stage_pair(pair, sources.stage)
        pin = None
        for side, subject in zip(('fixed', 'moving'), subjects_of(pair)):
            truth35, truth4 = sdfmix.label_map(arrays[f'{side}_seg35'], f'{subject} truth'), np.asarray(arrays[f'{side}_seg4']).astype(np.int16)
            if subject in held:
                if held[subject]['reduced'].truth.tobytes() != truth35.tobytes() or held[subject]['truth4'].tobytes() != truth4.tobytes():
                    raise SystemExit(f'{subject}: the truth of the two pairs that hold this subject disagree')
                continue
            if pin is None:
                pin = read_images(pair, sources)
                if tuple(float(s) for s in pin.grid.spacing) != (1., 1., 1.):
                    raise SystemExit(f'{pair}: spacing {pin.grid.spacing}: the distances of the comparison are for a grid of 1 mm')
            if not np.array_equal(arrays[side], getattr(pin, side)):
                raise SystemExit(f'{pair}:{side}: the firewall view differs from the archive member')
            tick = time.perf_counter()
            P, record = reader(pin, side)
            reduced = postmix.reduce_posterior(P, truth35)
            del P
            held[subject] = dict(reduced=reduced, truth4=truth4, record=record, pair=pair, side=side)
            log(f'{subject}: posterior read and reduced in {time.perf_counter()-tick:.1f} s; {reduced.where.size} voxels disagree')
        del arrays
    return held


def segmentation_row(labels, truth35, truth4, lookup):
    """experiments/unv2/eval_seg.py summarize_subject, the definitions of v1 and v2's table: the segmentation-level Dice of a label map against the truth, with the cortex and the white matter (seg4 labels 1, 3).  An undefined value is None."""
    eval_seg = core.load_module('_sv_eval_seg', 'experiments/unv2/eval_seg.py')
    row = eval_seg.summarize_subject(labels, truth35, truth4, lookup)
    return eval_seg.no_nan(dict(seg35=row['seg35'], seg4=row['seg4'], **row['groups'], cortex=row['seg4_per_label'][0], white_matter=row['seg4_per_label'][2], per_label=row['per_label'], seg4_per_label=row['seg4_per_label']))


def check_endpoints(subject, mixed, reduced):
    """On the real maps, not only in a test: `pred` is the argmax, `truth` is the truth, bit for bit (f = 0 and f > 1/2 of the mix); voxels where they agree do not change at any f."""
    if mixed[TAG_PRED].tobytes() != reduced.pred.tobytes():
        raise SystemExit(f'{subject}: L_f at f = 0 is not the prediction bit for bit')
    if mixed[TAG_TRUTH].tobytes() != reduced.truth.tobytes():
        raise SystemExit(f'{subject}: L_f at f > 1/2 is not the truth bit for bit')


def write_labels(root, subject, the_tag, labels, results=None):
    path = label_path(root, subject, the_tag, results)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f'{path} exists; never overwritten')
    np.savez_compressed(path, labels=labels)
    return dict(file=str(path), sha256=sha_file(path), labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest())


def build_maps(sources, reader, pins, log):
    """In memory, nothing written: {subject: {tag: L}}, their rows, the reduction per arm.  SystemExit when an arm's actual reduction is more than TARGET_TOL off its target."""
    mi = core.make_inputs()
    lookup, seg4_record = mi.load_seg4_map()
    subjects = read_subjects(sources, reader, log)
    fractions = {TAG_PRED: 0., TAG_TRUTH: postmix.check_f(.75), **{arm: float(pins['arms'][arm]['f']) for arm in TARGETS}}      # any f > 1/2 is the truth
    mixed, rows = {}, {}
    for subject, held in subjects.items():
        reduced = held['reduced']
        mixed[subject] = {the_tag: postmix.mix(reduced, f) for the_tag, f in fractions.items()}
        check_endpoints(subject, mixed[subject], reduced)
        truth_counts, pred_counts = postmix.voxels(reduced.truth), postmix.voxels(reduced.pred)
        rows[subject] = {}
        for the_tag, labels in mixed[subject].items():
            movement = sdfmix.movement(labels, reduced.pred, reduced.truth)
            if movement['agreeing_voxels_changed'] or movement['third_label']:
                raise SystemExit(f'{subject} {the_tag}: {movement}: the mix took a third label or changed a voxel where the two agree')
            counts = postmix.voxels(labels)
            rows[subject][the_tag] = dict(f=fractions[the_tag] if the_tag != TAG_TRUTH else 'any f > 1/2', provenance=side_provenance(the_tag), oracle=True, dice_against_truth=segmentation_row(labels, reduced.truth, held['truth4'], lookup), movement=movement, voxels=counts,
                                          vanished=postmix.vanished(counts, pred_counts, truth_counts), volume_vs_truth=postmix.ratios(counts, truth_counts), volume_vs_prediction=postmix.ratios(counts, pred_counts))
        log(f"{subject}: at f* seg35 / seg4 r1_3 {rows[subject]['r1_3']['dice_against_truth']['seg35']:.4f} / {rows[subject]['r1_3']['dice_against_truth']['seg4']:.4f}, r2_3 {rows[subject]['r2_3']['dice_against_truth']['seg35']:.4f} / {rows[subject]['r2_3']['dice_against_truth']['seg4']:.4f}")
    base = {s: rows[s][TAG_PRED]['dice_against_truth'] for s in rows}
    reductions = {arm: postmix.error_reduction({s: rows[s][arm]['dice_against_truth'] for s in rows}, base) for arm in TARGETS}
    for arm, target in TARGETS.items():
        actual = reductions[arm]['mean']['seg35']
        if actual is None or abs(actual-target) > TARGET_TOL:
            raise SystemExit(f'{arm}: f = {fractions[arm]} reduces the mean (1 - seg35) of the 6 subjects by {actual}, more than {TARGET_TOL} off the target {target:.4f}: nothing on the GPU starts; the probe or the pin is wrong; nothing written')
    return subjects, mixed, rows, reductions, fractions, seg4_record


def endpoint_inputs(results):
    root = Path(results)/'d0_v1_a5/inputs'
    return root/'A5P'/mi_stage(), root/'A5'/mi_stage()


def mi_stage():
    return core.make_inputs().STAGE_NAME


def check_endpoint_images(pair, fixed, moving, mixed, table, results):
    """The colouring of `pred` is A5P's input of this pair and that of `truth` A5's, bit for bit, on the real files (their `fixed`, `moving` only)."""
    mi = core.make_inputs()
    a5p, a5 = endpoint_inputs(results)
    sides = subjects_of(pair)
    for the_tag, where in ((TAG_PRED, a5p), (TAG_TRUTH, a5)):
        images = [mi.colour_from_table(image, mixed[subject][the_tag], table)[0] for image, subject in zip((fixed, moving), sides)]
        if core.equals_a5_input(pair, images[0], images[1], inputs=where) is not True:
            raise SystemExit(f'{pair}: the colouring of {the_tag} is not {where} bit for bit, or that file is missing')


def recipe(arm, fractions=None):
    tail = ('w = mean(I[L in {1,20}]), J = sum_(l=1..35) f_l w 1[L==l] with f_l of configs/bayesreg/colour_table_a5_v1.json, background 0, gaussian sigma .75 voxel (mode nearest, truncate 4)')
    if arm in TARGETS:
        return (f"oracle posterior mix (runs/20261104-sv/plan.md revision 1): L = argmax_l [(1-f) p_l + f 1(l = truth)], f = {(fractions or {}).get(arm)}, p = unet35_v1_ens all_folds_mean, ties: smallest l; "+tail)
    return (f'oracle one-subject truth (runs/20261104-sv/plan.md): {HARD_SUBJECT} L = truth seg35, the other subject L = argmax of unet35_v1_ens; '+tail)


def build_inputs(root, sources, mixed, fractions, table, table_record, subjects):
    """<root>/inputs/<arm>/stage_L2R_val_v1/<pair>/{pair.npz, manifest.json}, as make_inputs.build_arm: the members in order, `fixed` and `moving` the colouring of the arm's labels, the labels copied bitwise (the frozen run writes its labels.json from them, evaluate.py compares them with the truth)."""
    mi = core.make_inputs()
    files, pairs = {arm: {} for arm in ARMS}, {arm: {} for arm in ARMS}
    stage_name = mi.STAGE_NAME
    for arm in ARMS:
        out = inputs_root(arm, root)/stage_name
        out.mkdir(parents=True)
        for name in ('selection.json', 'completion.json'):
            shutil.copyfile(Path(sources.stage)/name, out/name)
            files[arm][f'{stage_name}/{name}'] = sha_file(out/name)
    for pair in PAIRS:
        arrays, manifest = read_stage_pair(pair, sources.stage)
        fixed_subject, moving_subject = subjects_of(pair)
        check_endpoint_images(pair, arrays['fixed'], arrays['moving'], mixed, table, sources.results)
        for arm, spec in ARMS.items():
            if pair not in spec['pairs']:
                continue
            record = dict(source_pair_sha256=manifest['pair_sha256'], source_manifest_sha256=sha_file(Path(sources.stage)/pair/'manifest.json'))
            members = dict(arrays)
            for side, subject in zip(('fixed', 'moving'), (fixed_subject, moving_subject)):
                the_tag = label_tag(arm, subject)
                image, consts = mi.colour_from_table(arrays[side], mixed[subject][the_tag], table)
                if image.dtype != np.float64 or not np.all(np.isfinite(image)):
                    raise SystemExit(f'{arm} {pair}:{side}: bad coloured image')
                members[side] = image
                record[side] = dict(subject=subject, tag=the_tag, f=fractions[the_tag] if the_tag != TAG_TRUTH else 'any f > 1/2', provenance=side_provenance(the_tag), **consts, image_min=float(image.min()), image_max=float(image.max()))
            dest = inputs_root(arm, root)/stage_name/pair
            dest.mkdir()
            np.savez_compressed(dest/'pair.npz', **{k: members[k] for k in mi.KEY_ORDER})
            with np.load(dest/'pair.npz') as z:
                if tuple(z.files) != mi.KEY_ORDER or any(z[k].tobytes() != members[k].tobytes() or z[k].dtype != members[k].dtype for k in mi.KEY_ORDER):
                    raise SystemExit(f'{arm} {pair}: written pair.npz does not reproduce the arrays')
            new = dict(manifest)
            new['pair_sha256'] = sha_file(dest/'pair.npz')
            new['preprocessing_d0'] = dict(arm=arm, recipe=recipe(arm, fractions), oracle=True, provenance=spec['provenance'], role=ROLE, protocol='runs/20261104-sv/plan.md', script='experiments/sv/run_sv.py', script_sha256=sha_file(Path(__file__)), seed=None, z_offsets=None,
                                           source_pair_sha256=record['source_pair_sha256'], source_manifest_sha256=record['source_manifest_sha256'], source_preprocessing=manifest.get('preprocessing'), source_baseline_native_mse=new.pop('baseline_native_mse', None),
                                           colour=dict(fixed=record['fixed'], moving=record['moving']), colour_table=table_record,
                                           note='fixed/moving replaced by the coloured images; labels and initial_u copied bitwise; baseline_label_metrics unchanged (identity map, same labels)')
            new['preprocessing'] = f"SV {arm}: {recipe(arm, fractions)}; on top of: {manifest.get('preprocessing')}"
            (dest/'manifest.json').write_text(json.dumps(new, indent=1)+'\n')
            files[arm][f'{stage_name}/{pair}/pair.npz'] = new['pair_sha256']
            files[arm][f'{stage_name}/{pair}/manifest.json'] = sha_file(dest/'manifest.json')
            pairs[arm][pair] = dict(record, posterior_records={s: subjects[s]['record'] for s in (fixed_subject, moving_subject)})
    for arm, spec in ARMS.items():
        (inputs_root(arm, root)/'d0_inputs.json').write_text(json.dumps(dict(arm=arm, recipe=recipe(arm, fractions), oracle=True, provenance=spec['provenance'], role=ROLE, stage=stage_name, source_stage=str(sources.stage), created_utc=utc(), script_sha256=sha_file(Path(__file__)), seed=None,
                                                                              sigma_voxel=mi.SIGMA, z_offsets=None, files_sha256=files[arm], pairs=pairs[arm], colour_table=table_record), indent=1, default=float)+'\n')


def build_all(root, *, sources=None, reader=None, log=print):
    """The labels and the inputs, or SystemExit.  build_record.json is written last: without it the build did not finish, and a second start stops (nothing is overwritten).  Nothing is written until the actual reductions have passed.  sources, reader: tests only."""
    sources = Sources() if sources is None else sources
    problem = root_problem(root, sources.results)
    if problem:
        raise SystemExit(problem)
    root = Path(root)
    for name in ('labels', 'inputs'):
        if (root/name).exists():
            raise SystemExit(f'{root/name} exists; never overwritten (a failed build is kept; the lead renames it)')
    pins = load_pins(sources.pins)
    if pins is None:
        raise SystemExit(f'f* 未钉：{sources.pins} 不存在')
    bad = pin_value_problems(pins)
    if bad:
        raise SystemExit('; '.join(bad))
    mi = core.make_inputs()
    table, table_record = mi.load_colour_table(mi.COLOUR_TABLE_A5, core.COLOUR_TABLE_SHA256)
    tick = time.perf_counter()
    subjects, mixed, rows, reductions, fractions, seg4_record = build_maps(sources, read_subject_posterior if reader is None else reader, pins, log)
    seconds_labels = time.perf_counter()-tick
    records = {}
    for subject in SUBJECTS:
        needed = tags_needed(subject)
        stored = {the_tag: dict(row, stored=the_tag in needed, **(write_labels(root, subject, the_tag, mixed[subject][the_tag], sources.results) if the_tag in needed else {})) for the_tag, row in rows[subject].items()}
        reduced = subjects[subject]['reduced']
        records[subject] = dict(subject=subject, oracle=True, role=ROLE, tie_rule=postmix.TIE_RULE, shape_zyx=list(reduced.pred.shape), disagree_voxels=int(reduced.where.size), read_from=dict(pair=subjects[subject]['pair'], side=subjects[subject]['side']), voxels_truth=postmix.voxels(reduced.truth), voxels_prediction=postmix.voxels(reduced.pred), labels=stored)
    build_inputs(root, sources, mixed, fractions, table, table_record, subjects)
    arms = {}
    for arm, spec in ARMS.items():
        subjects_here = sorted({s for pair in spec['pairs'] for s in subjects_of(pair)})
        arms[arm] = dict(provenance=spec['provenance'], decides=spec['decides'], pairs=list(spec['pairs']), target=spec['target'], f=fractions.get(arm), tags={s: label_tag(arm, s) for s in subjects_here}, reduction=reductions[arm]['mean'] if arm in reductions else None, reduction_per_subject=reductions[arm]['per_subject'] if arm in reductions else None,
                         vanished={s: rows[s][label_tag(arm, s)]['vanished'] for s in subjects_here if rows[s][label_tag(arm, s)]['vanished']})
    write_json(root/'labels/build_record.json', dict(schema='bayesreg.sv.build.v1', oracle=True, role=ROLE, written_utc=utc(), rule='L_f = argmax_l [(1 - f) p_l + f 1(l = truth)]', tie_rule=postmix.TIE_RULE, subjects=records, arms=arms, target_tolerance=TARGET_TOL, pins_sha256=sha_file(sources.pins), seg4_map=seg4_record, seconds_labels=seconds_labels, seconds_total=time.perf_counter()-tick,
                                                     peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, script_sha256=sha_file(Path(__file__)), postmix_sha256=sha_file(HERE/'postmix.py')))
    return 0


def build_cli(o):
    core.frozen_api()                                    # before anything else imports jd_curl: make_inputs puts the checkout first on sys.path
    return build_all(o.root)


def probe_cli(o):
    core.frozen_api()
    import probe_sv
    probe_sv.run_probe(o.root, workers=o.workers)
    return 0


def read_labels(root, subject, the_tag):
    """An L_f of the isolated root, against build_record.json: the file's sha256, its bytes, oracle true, the provenance of its tag."""
    path = label_path(root, subject, the_tag, Path(root).parent)
    record = ((read_json(Path(root)/'labels/build_record.json') or {}).get('subjects') or {}).get(subject, {}).get('labels', {}).get(the_tag)
    if record is None or record.get('oracle') is not True or record.get('provenance') != side_provenance(the_tag) or not record.get('stored') or not path.is_file():
        raise SystemExit(f'{path}: not a finished record of the build: {None if record is None else {k: record.get(k) for k in ("oracle", "provenance", "stored")}}')
    if sha_file(path) != record['sha256']:
        raise SystemExit(f'{path}: sha256 differs from build_record.json')
    with np.load(path) as z:
        labels = np.array(z['labels'])
    if labels.dtype != sdfmix.LABEL_DTYPE or hashlib.sha256(labels.tobytes()).hexdigest() != record['labels_sha256']:
        raise SystemExit(f'{path}: bytes differ from build_record.json')
    return labels, record


def arm_labels(root, arm, pin):
    """mc0p_core.Predicted of this arm: what the channels of one pair may use besides its two images, marked oracle, computed from where the maps come from.  The firewall's view holds no label; the maps come only from the isolated root."""
    labels, sides = {}, {}
    for side, subject in zip(('fixed', 'moving'), (pin.fixed_subject, pin.moving_subject)):
        the_tag = label_tag(arm, subject)
        labels[side], record = read_labels(root, subject, the_tag)
        sides[side] = dict(subject=subject, tag=the_tag, f=record['f'], provenance=record['provenance'], file=record['file'], sha256=record['sha256'], labels_sha256=record['labels_sha256'])
    return mc.Predicted(labels, None, dict(oracle=True, truth_labels_read=True, truth_read_how='through the L_f files of the isolated root, built under labels/; this process decodes no label member of a pair.npz', provenance=ARMS[arm]['provenance'] if arm in ARMS else PROVENANCE_PREFIX+arm, pair_keys_read=list(pin.provenance['keys_read']),
                                           label_firewall=pin.provenance['label_firewall'], sides=sides))


def images_equal_inputs(arm, pair, q0, root, results=None):
    """The two images of the labels the channels were made from, against the arm's inputs (the control: A5P's), bit for bit: entry and colouring come from one label map."""
    where = endpoint_inputs(Path(RESULTS if results is None else results))[0] if arm == CONTROL_ARM else inputs_root(arm, root)/core.make_inputs().STAGE_NAME
    return core.equals_a5_input(pair, q0[0][0], q0[1][0], inputs=where)


# ---- one entry -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ENTRY_SCHEMA = 'bayesreg.sv.entry.v1'
ENTRY_METRICS_SCHEMA = 'bayesreg.sv.entry_metrics.v1'


def run_entry(o, *, sources=None):
    """As run_mc0p.run_one for Q2 long (mc0p_core.q0_channels, prepare_stacks, mc0_core.p0_level_norms, optimize_k: the same calls, the same tier, the same cap of 9 GiB); only the label maps differ, and the record.  sources: tests only."""
    t0 = time.perf_counter()
    sources = Sources() if sources is None else sources
    arm, pair = require_cell(o.arm, o.pair)
    problem = root_problem(o.root, sources.results)
    if problem:
        raise SystemExit(problem)
    cell = EntryCell(arm, TIER, pair)
    out = entry_dir(o.root, arm, pair)
    fz = core.frozen_api()
    import torch
    if (out/'started.json').exists():
        raise SystemExit(f'{out}: this run has started before; a run is never rerun')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'started.json', dict(run_id=cell.run_id, utc=utc(), argv=sys.argv, pid=os.getpid()))
    device_cap = mc0p.limit_device_memory(torch, o.device)
    pin = read_images(pair, sources)
    config = fz.PreoptConfig(**core.tier_config(TIER))
    status, reason, u, record, scale, seconds_channels, level_norms, channel_sources = 'done', None, None, None, None, None, None, None
    try:
        predicted = arm_labels(o.root, arm, pin)
        channel_sources = predicted.record
        table = core.load_table()
        q0 = mc.q0_channels(pin, predicted, table)
        equal = images_equal_inputs(arm, pair, q0, o.root, sources.results)
        scale = dict(images_equal_sv_inputs=equal)
        if equal is not True:
            status, reason = 'input_mismatch', f"the colouring of the arm's labels against its inputs: {equal} (None: the file is missing); no optimization"
        else:
            stacks, prepared = mc.prepare_stacks(ENTRY_FORMULA, pin, predicted, q0, o.device, workers=core.SETUP_WORKERS)
            scale.update(prepared)
            reference = core.p0_level_norms(pin.fixed, pin.moving, predicted.labels['fixed'], predicted.labels['moving'], pin.grid, o.device, config.sigma_grad, table=table, workers=core.SETUP_WORKERS)
            del q0, predicted
            seconds_channels = time.perf_counter()-t0
            u, record = core.optimize_k(stacks, pin.grid, config, device=o.device, started=t0, observer=run.vram_guard(torch, o.device, mc.VRAM_LIMIT_BYTES), workers=core.SETUP_WORKERS, record_level_norms=True)
            del stacks
            record['role'] = ROLE
            level_norms = mc.level_norm_record(record['level_g_norm'], reference)
            if record['stopped'] is not None:
                status, reason = 'time_cap', f"the wall cap {core.RUN_CAP_SECONDS} s at {record['stopped']}"
    except run.VramLimit as error:
        status, reason = 'vram_stop', str(error)
    except torch.cuda.OutOfMemoryError as error:
        status, reason = 'cuda_oom', str(error)[:500]
    field = None
    if u is not None:
        np.savez(out/'field.npz', u=u)
        field = dict(file='field.npz', sha256=core.sha_file(out/'field.npz'), u_sha256=hashlib.sha256(u.tobytes()).hexdigest())
    cuda = torch.device(o.device).type == 'cuda'
    write_json(out/'run.json', dict(
        schema=ENTRY_SCHEMA, run_id=cell.run_id, arm=arm, tier=TIER, formula=ENTRY_FORMULA, pair=pair, k=mc.ARMS[ENTRY_FORMULA]['k'], oracle=True, provenance=channel_sources['provenance'] if channel_sources else None, status=status, reason=reason, role=ROLE,
        config=core.tier_config(TIER), scale=scale, channel_sources=channel_sources, level_norms=level_norms, optimizer=record, field=field, taper=core.TAPER, code_sha=o.code_sha, frozen_commit=core.FROZEN_COMMIT, random_seed='none: no random step',
        data=dict(pair_sha256=pin.provenance['pair_sha256'], manifest_sha256=pin.provenance['manifest_sha256'], fixed_subject=pin.fixed_subject, moving_subject=pin.moving_subject, stage=str(sources.stage)),
        env=dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0) if cuda else 'cpu'), device_cap=device_cap,
        device_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if cuda else None, device_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if cuda else None,
        host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, seconds_channels=seconds_channels, seconds_total=time.perf_counter()-t0, ended_utc=utc()))
    return 0


def evaluate_entries(o, *, sources=None):
    """As run_mc0p.evaluate_all: taper, .9^k certificate (CPU FP64, a record only), the metrics of evaluate.py; here the truth labels enter for the metrics.  Every arm's entry that has a field and no metrics.json; the control has none.  A failure is kept and the others go on.  sources: tests only."""
    problem = root_problem(o.root, (Sources() if sources is None else sources).results)
    if problem:
        raise SystemExit(problem)
    fz = core.frozen_api()
    ev = core.load_module('_mc0_evaluate', 'experiments/d0/evaluate.py')
    from jd_curl.three_d.feasibility import certify_map
    config = mc0p.certificate_config()
    failures, loaded = 0, {}
    for arm, pair in plan_cells():
        cell, d = EntryCell(arm, TIER, pair), entry_dir(o.root, arm, pair)
        if not (d/'field.npz').is_file() or (d/'metrics.json').exists() or (d/'metrics_error.json').exists():
            continue
        try:
            if (read_json(d/'run.json') or {}).get('oracle') is not True:
                raise ValueError('run.json is not marked oracle true')
            if pair not in loaded:
                loaded = {pair: run.read_pair(pair, mc0p.TRUTH_MEMBERS)}      # one pair in memory
            run.evaluate_one(d, cell, loaded[pair][1], loaded[pair][0], fz, ev, certify_map, config, oracle=True, schema=ENTRY_METRICS_SCHEMA)
        except Exception as error:                       # kept; the others go on; not retried
            write_json(d/'metrics_error.json', dict(run_id=cell.run_id, error=repr(error)[:2000], utc=utc()))
            failures += 1
    return 8 if failures else 0


# ---- the inputs and the entry of a native run --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def input_files(arm, pair, root):
    """(pair.npz directory, their sha256, the manifest) of the arm's inputs, checked against the record and the arm; EntryError otherwise."""
    require_cell(arm, pair)
    base = inputs_root(arm, root)
    record = read_json(base/'d0_inputs.json')
    if record is None:
        raise EntryError(f'{base}/d0_inputs.json: no such file')
    stage = core.make_inputs().STAGE_NAME
    d = base/stage/pair
    if not (d/'pair.npz').is_file() or not (d/'manifest.json').is_file():
        raise EntryError(f'{d}: no pair.npz or manifest.json')
    pair_sha, manifest_sha = sha_file(d/'pair.npz'), sha_file(d/'manifest.json')
    listed = record.get('files_sha256') or {}
    if listed.get(f'{stage}/{pair}/pair.npz') != pair_sha or listed.get(f'{stage}/{pair}/manifest.json') != manifest_sha:
        raise EntryError(f'{d}: the files differ from {base}/d0_inputs.json')
    manifest = json.loads((d/'manifest.json').read_text())
    made = manifest.get('preprocessing_d0') or {}
    if manifest.get('pair') != pair or manifest.get('pair_sha256') != pair_sha or made.get('arm') != arm or made.get('oracle') is not True or record.get('oracle') is not True or made.get('provenance') != ARMS[arm]['provenance']:
        raise EntryError(f"{d}: manifest pair {manifest.get('pair')}, arm {made.get('arm')}, oracle {made.get('oracle')}, provenance {made.get('provenance')}; expected {pair}, {arm}, True, {ARMS[arm]['provenance']}")
    return d, pair_sha, manifest_sha, manifest


def channel_problems(arm, pair, sources):
    """The marks of an entry against the arm: oracle true, this arm's provenance, each side's subject and label map."""
    if not isinstance(sources, dict):
        return ['run.json has no channel_sources']
    bad = []
    if sources.get('oracle') is not True:
        bad.append(f"channel_sources oracle {sources.get('oracle')}")
    if sources.get('provenance') != ARMS[arm]['provenance']:
        bad.append(f"channel_sources provenance {sources.get('provenance')}")
    keys = set(sources.get('pair_keys_read') or [])
    if not {'fixed', 'moving'} <= keys or keys-set(mc.IMAGE_KEYS):
        bad.append(f'channel_sources pair_keys_read {sorted(keys)}: only the images')
    sides = sources.get('sides') or {}
    for side, subject in zip(('fixed', 'moving'), subjects_of(pair)):
        got = sides.get(side) or {}
        the_tag = label_tag(arm, subject)
        if got.get('subject') != subject or got.get('tag') != the_tag or got.get('provenance') != side_provenance(the_tag):
            bad.append(f'channel_sources {side} {got}: expected {subject}, {the_tag}')
    return bad


def check_entry(arm, pair, manifest, root):
    """The record for seed_entry.json, or EntryError (EntryMissing if the files are not there).  The one place: <root>/entries/<arm>/<ffff>-<mmmm>."""
    d = entry_dir(root, arm, pair)
    field_path, run_path = d/'field.npz', d/'run.json'
    if not field_path.is_file() or not run_path.is_file():
        raise EntryMissing(f'{arm} {short_name(pair)}: 入口不存在: {field_path} 或 {run_path.name}')
    record = json.loads(run_path.read_text())
    cell = EntryCell(arm, TIER, pair)
    problems = []
    if record.get('schema') != ENTRY_SCHEMA:
        problems.append(f"run.json schema {record.get('schema')}")
    if record.get('pair') != pair or record.get('arm') != arm or record.get('tier') != TIER or record.get('k') != mc.ARMS[ENTRY_FORMULA]['k'] or record.get('run_id') != cell.run_id:
        problems.append(f"run.json pair {record.get('pair')} arm {record.get('arm')} tier {record.get('tier')} K {record.get('k')} run_id {record.get('run_id')}")
    if record.get('status') != 'done':
        problems.append(f"run.json status {record.get('status')}: only a finished run is an entry")
    if record.get('oracle') is not True:
        problems.append(f"run.json oracle {record.get('oracle')}: every entry of this package is oracle true")
    config = record.get('config') or {}
    if {k: config.get(k) for k in seed.LONG_TIER} != seed.LONG_TIER:
        problems.append(f'run.json config {config}: not the long tier')
    if (record.get('scale') or {}).get('images_equal_sv_inputs') is not True:
        problems.append(f"run.json scale: images_equal_sv_inputs {(record.get('scale') or {}).get('images_equal_sv_inputs')}")
    problems += channel_problems(arm, pair, record.get('channel_sources'))
    made = manifest.get('preprocessing_d0') or {}
    data = record.get('data') or {}
    if data.get('pair_sha256') != made.get('source_pair_sha256') or data.get('manifest_sha256') != made.get('source_manifest_sha256'):
        problems.append(f"the field comes from original pair {data.get('pair_sha256')}, the inputs of {arm} from {made.get('source_pair_sha256')}")
    field = record.get('field') or {}
    if field.get('file') != 'field.npz' or not field.get('sha256') or not field.get('u_sha256'):
        problems.append(f'run.json has no field record: {field}')
    if problems:
        raise EntryError(f'{d}: '+'; '.join(problems))
    got = sha_file(field_path)
    if got != field['sha256']:
        raise EntryError(f"{field_path}: sha256 {got} != run.json {field['sha256']}")
    shape = tuple(manifest['shape_zyx'])
    names, stored, dtype = seed.npz_header(field_path)
    if names != ['u.npy'] or stored != shape+(3,) or dtype not in ('<f8', 'float64'):
        raise EntryError(f'{field_path}: members {names}, shape {stored}, dtype {dtype}; expected only u, {shape+(3,)}, float64')
    entry = dict(arm=arm, pair=pair, field_path=str(field_path), field_sha256=got, u_sha256=field['u_sha256'], source_run_id=record['run_id'], oracle=True, charge_seconds=seed.ENTRY_CHARGE_SECONDS, shape_zyx=list(shape))
    try:
        u = seed.inject_run.load_field(entry)            # the wrapper's own check, once more when it starts
    except seed.inject_run.Refused as refused:
        raise EntryError(str(refused)) from None
    tapered = seed.boundary_taper_sha256(u, shape)
    metrics = read_json(d/'metrics.json') or {}
    if metrics.get('tapered_u_sha256') != tapered:
        raise EntryError(f"{d}: the evaluation tapered this field to {metrics.get('tapered_u_sha256')}, here {tapered} (an entry without its evaluation does not start a native run)")
    certificate = metrics.get('certificate') or {}
    x = metrics.get('metrics') or {}
    return dict(entry, tapered_sha256=tapered, tapered_recorded_by_evaluation=metrics.get('tapered_u_sha256'), source_seconds_total=record.get('seconds_total'), source_code_sha=record.get('code_sha'), source_status=record.get('status'), source_k=record.get('k'), source_tier=record.get('tier'),
                source_arm=arm, source_root=str(root), provenance=ARMS[arm]['provenance'], role=seed.inject_run.ROLE,
                entry_metrics=dict(seg35=x.get('seg35'), seg4=x.get('seg4'), sdlogj_official=x.get('sdlogj_official'), accepted_alpha=certificate.get('accepted_alpha'), n_infeasible_at_1=certificate.get('n_infeasible_at_1'), at_accepted_alpha=metrics.get('at_accepted_alpha')))


# ---- the machine ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class System(seed.System, run.System):
    """SEED's machine for the native runs (prepare, check_prepared, launch, classify), MC0's for the other units (run, code_checkout); the arms' inputs and the evaluation step are this package's.  code: the detached worktree of this batch."""
    code = None

    def inputs_problems(self, arm, repo, root):
        return seed.b1_spec.check_one_input(b1_entry(arm, root), repo)

    def evaluate(self, run_path, arm, pair, run_id):
        with open(Path(run_path)/'log.txt', 'ab') as f:
            return subprocess.run(seed.check_launch([seed.PY, '-B', Path(self.code)/SCRIPT, 'evaluate-one', run_path, '--arm', arm, '--pair', pair, '--run-id', run_id]), stdout=f, stderr=subprocess.STDOUT).returncode

    def registry_problems(self):
        """What the predictions' registry must be when the build reads them: committed and clean."""
        mi = core.make_inputs()
        from bayesreg.seg.structure_layer import registry_git_state
        git = registry_git_state(mi._registry_of(mc.default_unet()))
        return [] if git.get('tracked') is True and git.get('clean') is True else [f'the repository registry is not committed and clean: {git}']

    def pin_problems(self, sources):
        """The pinned file must be committed and clean: $ROOT/code holds HEAD, and the build reads it there."""
        path = Path(sources.pins)
        rel = str(path.resolve().relative_to(ROOT)) if inside(path, ROOT) else None
        return [f'{path}: not committed or not clean'] if rel is None or not seed.b1_spec._tracked_and_clean(ROOT, rel) else []


def busy_reason(system, root):
    """Why no unit may start now, or None: the gate of the brief (§4), for every unit."""
    if (Path(root)/'hold').exists():
        return 'hold file'
    others = [u for u in system.units() if u != f'{QUEUE_UNIT}.service']
    if others:
        return f'other units: {others}'
    load = system.load1()
    if load >= LOAD_LIMIT:
        return f'1-minute load {load} >= {LOAD_LIMIT}'
    if system.free_gib('/home') < HOME_FLOOR_GIB:
        return f'/home free < {HOME_FLOOR_GIB} GiB'
    if system.free_gib('/mnt/c') < HOST_FLOOR_GIB:
        return f'/mnt/c free < {HOST_FLOOR_GIB} GiB'
    if system.mem_available_gib() < RESERVE_MIN_MEM_GIB:
        return f'MemAvailable < {RESERVE_MIN_MEM_GIB} GiB'
    busy = system.gpu_busy()
    if busy:
        return f'GPU busy: {busy}'
    if system.gpu_free_mib() < GPU_FREE_MIB:
        return f'GPU free < {GPU_FREE_MIB} MiB'
    return None


def wait_for_gate(system, root, minutes, log):
    started = system.now()
    while True:
        reason = busy_reason(system, root)
        if reason is None:
            return
        if system.now()-started >= minutes*60:
            raise GateClosed(f'gate closed for {minutes} minutes: {reason}')
        log(f'waiting: {reason}')
        system.sleep(seed.POLL_SECONDS)


class Pace:
    """60 s between any two units this start launches."""

    def __init__(self, system):
        self.system, self.first = system, True

    def __call__(self):
        if not self.first:
            self.system.sleep(BETWEEN_RUNS_SECONDS)
        self.first = False


# ---- the start checks --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def control_reference(results=None):
    """The u sha256 of Q2 long 0438-0439 in mc0p_v1 (S_P's own entry, read only), None without the record."""
    record = read_json(seed.entry_dir('S_P', CONTROL_PAIR, RESULTS if results is None else results)/'run.json') or {}
    return (record.get('field') or {}).get('u_sha256')


def input_problems(sources):
    """What must exist before anything runs.  Ids, provenance records, file presence; no label member is decoded."""
    import decide_sv
    bad = []
    stage = Path(sources.stage)
    selection = read_json(stage/'selection.json')
    if selection is None:
        return [f'{stage}/selection.json: no such file']
    first = tuple(row['pair'] for row in selection['pairs'][:5])
    if first != PAIRS:
        bad.append(f'the first 5 pairs of {stage}/selection.json differ from the plan: {first}')
    from bayesreg.io.guard import check_pair_subjects, load_dev_allowlist
    from bayesreg.io.pair import _read_manifest          # the whitelisting reader: label-derived records are dropped while the JSON is decoded
    guard = load_dev_allowlist(sources.allowlist)
    for pair in PAIRS:
        d = stage/pair
        if not (d/'pair.npz').is_file() or not (d/'manifest.json').is_file():
            bad.append(f'{d}: no pair.npz or manifest.json')
            continue
        try:
            _, _, manifest = _read_manifest(d/'manifest.json')
            check_pair_subjects(manifest['fixed_subject'], manifest['moving_subject'], guard)
        except Exception as error:                       # a refusal is a problem of the start checks, not a crash
            bad.append(f'{d}: {type(error).__name__}: {error}')
            continue
        if (manifest['fixed_subject'], manifest['moving_subject']) != subjects_of(pair):
            bad.append(f"{pair}: the manifest names {manifest['fixed_subject']}, {manifest['moving_subject']}")
    for subject in SUBJECTS:
        p = Path(sources.pred_root)/subject
        missing = [name for name in ('provenance.json', 'probs36.npz', 'structure_groups.npz') if not (p/name).is_file()]
        if missing:
            bad.append(f'{subject}: prediction {p}: missing {missing}')
        else:
            prov = json.loads((p/'provenance.json').read_text())
            if prov.get('policy') != mc.POLICY or prov.get('segmenter_id') != mc.SEGMENTER_ID or prov.get('test_labels_seen') is not False:
                bad.append(f"{subject}: prediction not {mc.SEGMENTER_ID} {mc.POLICY}: {prov.get('segmenter_id')}, {prov.get('policy')}, test_labels_seen {prov.get('test_labels_seen')}")
        if not (Path(sources.seg_root)/subject/'provenance.json').is_file():
            bad.append(f'{subject}: no Atropos posteriors under {sources.seg_root} (the alignment QC needs them)')
    for where in endpoint_inputs(sources.results):
        bad += [f'{where/pair}: no pair.npz (the check of the colouring at f = 0, f > 1/2)' for pair in PAIRS if not (where/pair/'pair.npz').is_file()]
    bad += decide_sv.endpoint_input_problems(sources.results)
    if control_reference(sources.results) is None:
        bad.append(f"{seed.entry_dir('S_P', CONTROL_PAIR, sources.results)}/run.json: no field record (the reference of the control {CONTROL_ARM})")
    try:
        core.load_table()
    except (SystemExit, OSError, ValueError) as error:
        bad.append(f'the colour table: {error}')
    return bad


# ---- the steps --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def write_records(run_path, arm, pair, unit, run_id, entry, files, env_text, code, results=None):
    """run_seed.write_records for this arm: the records of protocol §8, the entry's own record and the wrapper's copy."""
    run_path = Path(run_path)
    d, pair_sha, manifest_sha, manifest = files
    git = subprocess.run(['git', '-C', str(code), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(['git', '-C', str(code), 'diff', '--stat'], capture_output=True, text=True).stdout
    files_sha = '\n'.join(f'{p.relative_to(code)} {sha_file(p)}' for pattern in ('experiments/sv/*.py', 'experiments/seed/*.py', 'experiments/mc0/*.py', 'experiments/d0/*.py', 'experiments/d0/*.json') for p in sorted(Path(code).glob(pattern)))
    frozen = seed.frozen_dir(results)
    frozen_head = subprocess.run(['git', '-C', str(frozen), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    frozen_status = subprocess.run(['git', '-C', str(frozen), 'status', '--porcelain'], capture_output=True, text=True).stdout
    (run_path/'git.txt').write_text(f'code: {code}\ncode HEAD: {git}\ncode git diff --stat:\n{diff}\nfrozen worktree: {frozen}\nfrozen HEAD: {frozen_head}\nfrozen commit expected: {seed.FROZEN_COMMIT}\nfrozen git status --porcelain (must be empty):\n{frozen_status}\nfile sha256:\n{files_sha}\n')
    (run_path/'env.txt').write_text(env_text)
    made = manifest['preprocessing_d0']
    (run_path/'data.txt').write_text('\n'.join([f'run_id: {run_id}', f'arm: {arm}', f'pair: {pair}', f'unit: {unit}', f"fixed subject: {manifest['fixed_subject']}", f"moving subject: {manifest['moving_subject']}",
                                                f'arm stage directory (prepare source): {d.parent}', f'arm input pair.npz: {d}/pair.npz sha256 {pair_sha}', f'arm input manifest.json: {d}/manifest.json sha256 {manifest_sha}',
                                                f"original pair.npz sha256: {made['source_pair_sha256']}", f"original manifest sha256: {made['source_manifest_sha256']}", f"colouring: {made['recipe']}",
                                                f"oracle: true; provenance: {ARMS[arm]['provenance']}; {ROLE}", f"entry field: {entry['field_path']} sha256 {entry['field_sha256']} bytes sha256 {entry['u_sha256']}",
                                                f"entry source run: {entry['source_run_id']}; K {entry['source_k']}; tier {entry['source_tier']}", 'truth labels for evaluation: the original stage pair.npz (evaluate.py checks its sha256)'])+'\n')
    (run_path/'seed').write_text(f'{arm} (runs/20261104-sv/plan.md): no random step in the labels (ties: the smallest label) or in the entry; solver: configuration C has no random number generator or seed option; GPU reductions may still differ between runs\n')
    write_json(run_path/'seed_entry.json', entry)
    write_json(run_path/'seed_arm.json', dict(arm=arm, seconds=seed.SECONDS, changes=seed.CHANGES, inputs_root=str(inputs_root(arm, Path(run_path).parents[2])), entry_root=str(entry_dir(Path(run_path).parents[2], arm, pair)), source_arm=arm, charge_seconds=seed.ENTRY_CHARGE_SECONDS,
                                              watchdog_seconds=seed.WATCHDOG, method=method_name(arm), input_oracle=True, oracle=True, provenance=ARMS[arm]['provenance'], decides=ARMS[arm]['decides'], role=ROLE, wrapper_sha256=sha_file(Path(code)/'experiments/seed/inject_run.py')))
    shutil.copyfile(Path(code)/'experiments/seed/inject_run.py', run_path/'seed_inject.py')


def one_native(o, system, root, arm, pair, log, env_text, results, code, pace):
    """run_seed.one_run for this arm: the checks, prepare, the records, the unit, the classification, the evaluation.  Returns the state, or None if it had already ended."""
    run_path = run_dir(root, arm, pair)
    if (run_path/'queue_done.json').is_file():
        return None
    unit = unit_name(arm, pair)
    if run_path.exists():                                # never rerun: an ended unit is post-processed, anything else stops
        if (run_path/'exit_code').is_file():
            log(f'resume: {run_path} has an exit_code; post-processing')
            return seed.finish_run(run_path, arm, pair, (run_path/'run_id').read_text().strip(), (run_path/'uptime_before').read_text().strip(), system, log)
        raise Stop(f'{run_path} exists without exit_code or queue_done.json (an incomplete attempt, kept whole)')
    try:
        files = input_files(arm, pair, root)
        entry = check_entry(arm, pair, files[3], root)
    except EntryError as error:
        raise Stop(str(error)) from None
    pace()
    wait_for_gate(system, root, o.wait_minutes, log)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    argv = prepare_argv(arm, pair, run_path, results)
    target = files[0].parent
    prepare_log = Path(root)/'prepare.log'
    rc = system.prepare(argv, target, prepare_log, results)
    if rc != 0 or not run_path.is_dir():
        raise Stop(f'prepare exit {rc} for {run_path} (see {prepare_log})')
    check = system.check_prepared(run_path, pair, f'SV-{arm}', files[1], files[2], b1_entry(arm, root))
    write_json(run_path/'prepared_check.json', check)
    if not check.get('ok'):
        raise Stop(f"prepared run differs from S_P's configuration or the arm's inputs: {check.get('problems')}")
    run_id = new_run_id(arm, pair)
    (run_path/'run_id').write_text(run_id+'\n')
    write_records(run_path, arm, pair, unit, run_id, entry, files, env_text, code, results)
    sd = unit_argv(arm, pair, run_path, root)
    (run_path/'cmd.txt').write_text(f'# prepare (run from any directory; the frozen script resolves its root from its own path)\n{shlex.join(argv)}\n# worktree link during prepare: {seed.frozen_dir(results)}/data/three_d/stage_L2R_val_v1 -> {target}\n# run\n{shlex.join(sd)}\n')
    (run_path/'service_unit.txt').write_text(unit+'\n')
    t0 = system.uptime()
    (run_path/'uptime_before').write_text(f'{t0}\n')
    log(f"run {arm} {pair} unit {unit}; load {system.load1()}; /home free {system.free_gib('/home'):.0f} GiB; /mnt/c free {system.free_gib('/mnt/c'):.0f} GiB")
    with open(run_path/'log.txt', 'a') as f:
        f.write(f'{utc()} launch {unit}; {system.state()}\n')
    rc = system.launch(sd, run_path/'log.txt')
    with open(run_path/'log.txt', 'a') as f:
        f.write(f'{utc()} unit ended, systemd-run exit {rc}; {system.state()}\n')
    return seed.finish_run(run_path, arm, pair, run_id, t0, system, log)


def trailing_infrastructure(root, arms):
    """Infrastructure failures in a row at the end of what has run, in plan order."""
    n = 0
    for arm, pair in plan_cells(arms):
        done = read_json(run_dir(root, arm, pair)/'queue_done.json')
        if done is None:
            continue
        n = n+1 if done.get('state') == 'infrastructure' else 0
    return n


def step_build(o, system, root, code, log, pace):
    if (root/'labels/build_record.json').is_file():
        return
    for name in ('labels', 'inputs'):
        if (root/name).exists():
            raise Stop(f'{root/name} exists without labels/build_record.json (a build that did not finish, kept whole; the lead renames it)')
    pace()
    wait_for_gate(system, root, o.wait_minutes, log)
    log(f'launch {BUILD_UNIT}; load {system.load1()}')
    rc = system.run(build_argv(code, root), root/'build.systemd_run.log')
    log(f'ended {BUILD_UNIT}: rc {rc}')
    if rc != 0 or not (root/'labels/build_record.json').is_file():
        raise Stop(f'{BUILD_UNIT}: exit {rc}, no labels/build_record.json (see {root}/build.log); kept whole; no GPU has started')


def step_k1(o, system, root, code, log, pace):
    k1 = root/'k1_check_gpu.json'
    if not k1.is_file():
        pace()
        wait_for_gate(system, root, o.wait_minutes, log)
        log(f'launch {K1_UNIT}')
        system.run(k1_argv(code, root), root/'k1_check_gpu.systemd_run.log')
    check = read_json(k1)
    if check is None or check.get('bitwise') is not True:
        raise Stop(f'the K = 1 check on the GPU did not pass: {check}')


def step_entry(o, system, root, cell, code, sha, log, pace):
    d = entry_dir(root, cell.arm, cell.pair)
    done = read_json(d/'queue_done.json')
    if done is not None and done.get('state') == 'done':
        return
    if d.exists():
        raise Stop(f'{d} exists and is not a finished entry: {done} (kept whole, never rerun)')
    pace()
    wait_for_gate(system, root, o.wait_minutes, log)
    d.mkdir(parents=True)
    argv = entry_argv(cell, code, root, sha)
    (d/'cmd.txt').write_text(shlex.join(argv)+'\n')
    log(f'launch {cell.unit}; load {system.load1()}')
    tick = system.now()
    rc = system.run(argv, d/'systemd_run.log')
    seconds = system.now()-tick
    record = read_json(d/'run.json')
    state = 'failed' if rc != 0 or record is None else record['status']
    write_json(d/'queue_done.json', dict(run_id=cell.run_id, unit=cell.unit, rc=rc, unit_seconds=seconds, state=state, ended_utc=utc()))
    log(f'ended {cell.run_id}: rc {rc}, {seconds:.0f} s, {state}')
    if state != 'done':
        raise Stop(f"{cell.run_id}: {state}: {(record or {}).get('reason')}; kept whole, never rerun: no native run starts without its entry")
    if record.get('oracle') is not True:
        raise Stop(f"{cell.run_id}: run.json oracle {record.get('oracle')}")
    if (record.get('scale') or {}).get('images_equal_sv_inputs') is not True:
        raise Stop(f"{cell.run_id}: the colouring of its labels is not the arm's input bit for bit")


def control_difference(root, results):
    """How far the control's field is from S_P's own entry (that field only read): max |du|, RMS, the share of the values that differ; a reason instead when a field is missing or has another shape."""
    ours, theirs = entry_dir(root, CONTROL_ARM, CONTROL_PAIR)/'field.npz', seed.entry_dir('S_P', CONTROL_PAIR, results)/'field.npz'
    if not ours.is_file() or not theirs.is_file():
        return dict(max_abs=None, rms=None, share_of_values_different=None, reason=f"no field: {[str(p) for p in (ours, theirs) if not p.is_file()]}")
    with np.load(ours) as a, np.load(theirs) as b:
        u, v = a['u'], b['u']
    if u.shape != v.shape:
        return dict(max_abs=None, rms=None, share_of_values_different=None, reason=f'shapes {u.shape}, {v.shape}')
    d = u-v
    return dict(max_abs=float(np.abs(d).max()), rms=float(np.sqrt(np.mean(d*d))), share_of_values_different=float(np.count_nonzero(d))/d.size, reason=None)


def write_control(root, record):
    """control.json.  An earlier record that says something else is kept under its time: nothing is overwritten (the lead's acceptance above all)."""
    path = Path(root)/'control.json'
    earlier = read_json(path)
    if earlier is not None:
        if {k: v for k, v in earlier.items() if k != 'utc'} == {k: v for k, v in record.items() if k != 'utc'}:
            return
        stamp, k = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'), 0
        keep = Path(root)/f'control_{stamp}.json'
        while keep.exists():
            k += 1
            keep = Path(root)/f'control_{stamp}_{k}.json'
        path.rename(keep)
    write_json(path, record)


def step_control(o, system, root, code, sha, log, pace, results):
    """The entry with the prediction on both sides: the same field as S_P's own entry, bit for bit, or this path is not S_P's but for the labels.  An acceptance of a difference stays with that field: a later start without the option neither loses it nor stops again."""
    step_entry(o, system, root, EntryCell(CONTROL_ARM, TIER, CONTROL_PAIR), code, sha, log, pace)
    reference = control_reference(results)
    u = ((read_json(entry_dir(root, CONTROL_ARM, CONTROL_PAIR)/'run.json') or {}).get('field') or {}).get('u_sha256')
    same = u is not None and u == reference
    earlier = read_json(root/'control.json') or {}
    given = (getattr(o, 'accept_control_difference', None) or '').strip() or None
    carried = earlier.get('accepted_difference') if earlier.get('u_sha256') == u and earlier.get('u_sha256_reference') == reference else None
    accepted = None if same else (given or carried)
    difference = None if same else control_difference(root, results)
    write_control(root, dict(oracle=True, role=ROLE, cell=EntryCell(CONTROL_ARM, TIER, CONTROL_PAIR).run_id, control_bitwise=same, u_sha256=u, u_sha256_reference=reference, reference=str(seed.entry_dir('S_P', CONTROL_PAIR, results)), difference=difference, accepted_difference=accepted, utc=utc()))
    if same:
        log(f'control {CONTROL_ARM}: bit for bit against {reference}')
        return
    numbers = f"max |du| {difference['max_abs']}, RMS {difference['rms']}, share of values {difference['share_of_values_different']}{'; '+difference['reason'] if difference['reason'] else ''}"
    if not accepted:
        raise Stop(f'the control {CONTROL_ARM}: u {u} != {reference} of S_P\'s entry ({numbers}): this path may differ from S_P\'s in more than the labels; control.json; nothing else on the GPU starts (the lead reads it; --accept-control-difference "WHO, WHY" goes past it)')
    log(f'control {CONTROL_ARM}: DIFFERS from {reference} ({numbers}); ACCEPTED: {accepted}; "only the labels differ" is not proved for this batch; the decision and the summary say so')


def step_entry_evaluation(o, system, root, code, log, pace, arms, unit):
    missing = [(a, p) for a, p in plan_cells(arms) if not (entry_dir(root, a, p)/'metrics.json').is_file()]
    if not missing:
        return
    pace()
    wait_for_gate(system, root, o.wait_minutes, log)
    log(f'launch {unit}: {len(missing)} entries')
    rc = system.run(entry_eval_argv(code, root, unit), root/f'{unit}.systemd_run.log')
    still = [(a, short_name(p)) for a, p in plan_cells(arms) if not (entry_dir(root, a, p)/'metrics.json').is_file()]
    log(f'ended {unit}: rc {rc}')
    if rc != 0 or still:
        raise Stop(f'{unit}: exit {rc}; entries without metrics.json: {still} (metrics_error.json in their directories); no native run starts')


def run_arms(o, system, root, arms, code, sha, log, pace, sources, env_text, eval_unit):
    """Entries, their evaluation, the native runs of these arms."""
    for arm, pair in plan_cells(arms):
        step_entry(o, system, root, EntryCell(arm, TIER, pair), code, sha, log, pace)
    step_entry_evaluation(o, system, root, code, log, pace, arms, eval_unit)
    problems = [f'{arm}: {p}' for arm in arms for p in system.inputs_problems(arm, code, root)]
    if problems:
        raise Stop('inputs records: '+' | '.join(problems))
    for arm, pair in plan_cells(arms):
        state = one_native(o, system, root, arm, pair, log, env_text, sources.results, code, pace)
        if state is not None and trailing_infrastructure(root, arms) >= seed.MAX_CONSECUTIVE_INFRA:
            raise Stop(f'{seed.MAX_CONSECUTIVE_INFRA} infrastructure failures in a row')


def reading_parts(result):
    """What may not change after the reading is written: the code, the sentence, the two arms' rows."""
    return result['reading_code'], result['reading'], result['arms']


def queue_body(o, system, root, log, sources, repo):
    import decide_sv
    problems = list(system.frozen_problems(sources.results))+input_problems(sources)+list(system.registry_problems())+fstar_problems(root, sources)
    if not problems:
        problems += list(system.pin_problems(sources))
    if problems:
        raise Stop('start checks: '+' | '.join(problems))
    log('start checks passed: frozen worktree, stage, predictions, Atropos, endpoints of seed_v1, A5 and A5P inputs, the control\'s reference, colour table, registry, f* pinned and backed by the probe')
    code, sha = system.code_checkout(root, None)
    system.code = code
    batch = read_json(root/'batch.json')
    plan = [f'{a} {short_name(p)}' for a, p in plan_cells()]
    if batch is None:
        write_json(root/'batch.json', dict(schema='bayesreg.sv.batch.v1', oracle=True, role=ROLE, code_sha=sha, frozen_commit=core.FROZEN_COMMIT, started_utc=utc(), plan=plan, seconds=seed.SECONDS, changes=seed.CHANGES, charge_seconds=seed.ENTRY_CHARGE_SECONDS, reading_arms=list(READING_ARMS), description_arms=list(DESCRIPTION_ARMS),
                                           fstar={arm: load_pins(sources.pins)['arms'][arm]['f'] for arm in TARGETS}, pins_sha256=sha_file(sources.pins), gate=dict(load_limit=LOAD_LIMIT, gpu_free_mib=GPU_FREE_MIB, home_floor_gib=HOME_FLOOR_GIB, host_floor_gib=HOST_FLOOR_GIB, mem_available_gib=RESERVE_MIN_MEM_GIB)))
    elif batch.get('code_sha') != sha:
        raise Stop(f"batch.json is at {batch.get('code_sha')}, the code at {sha}")
    elif batch.get('plan') != plan:
        raise Stop('batch.json has another plan')
    pace = Pace(system)
    step_build(o, system, root, code, log, pace)
    step_k1(o, system, root, code, log, pace)
    step_control(o, system, root, code, sha, log, pace, sources.results)
    env_text = seed.environment_text(code) if isinstance(system, System) else 'test\n'
    (root/'cache/numba_jit').mkdir(parents=True, exist_ok=True)
    (root/'cache/first_use').mkdir(parents=True, exist_ok=True)
    try:
        run_arms(o, system, root, READING_ARMS, code, sha, log, pace, sources, env_text, EVAL_UNIT)
    except Stop:                                         # what finished is summed up; the stop stays a stop
        if any((run_dir(root, a, p)/'metrics.json').is_file() for a, p in plan_cells(READING_ARMS)):
            try:
                result = decide_sv.write_analysis(root, results=sources.results)
                log(f"the queue stops; reading of what finished: {result['reading']}")
            except (Exception, SystemExit) as error:
                log(f'the queue stops; the summary failed: {error!r}')
        raise
    try:
        result = decide_sv.write_analysis(root, results=sources.results)
    except (Exception, SystemExit) as error:             # the runs stay; by hand: run_sv.py decide
        raise Stop(f'the reading failed: {error!r}') from None
    log(f"reading: {result['reading']}")
    reading = reading_parts(result)
    # the description: after the reading is written; whatever goes wrong here is recorded and changes nothing above
    stop_file = root/'f0440_stop.json'
    if stop_file.exists():
        stop_file.rename(root/f"f0440_stop_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    try:
        run_arms(o, system, root, DESCRIPTION_ARMS, code, sha, log, pace, sources, env_text, EVAL_UNIT_F0440)
    except (Exception, SystemExit) as stop:
        why = str(stop) if isinstance(stop, (Stop, run.Stop)) else repr(stop)
        write_json(stop_file, dict(reason=why, utc=utc(), role='f0440 只描述：它失败只记录，不影响读数'))
        log(f'f0440 stops, recorded, the reading stays: {why}')
    try:
        result = decide_sv.write_analysis(root, results=sources.results)
    except (Exception, SystemExit) as error:
        log(f'the summary with f0440 failed: {error!r}; the reading written before stays in analysis/previous_*')
        return 0
    if reading_parts(result) != reading:                 # cannot happen by the code: said loudly if it does
        raise Stop(f"the reading changed after f0440: {reading[1]} -> {result['reading']}")
    log('Handed to the lead: no next step starts here')
    return 0


def queue(o, system=None, sources=None, repo=None):
    system = System() if system is None else system
    sources = Sources() if sources is None else sources
    repo = seed.MAIN_REPO if repo is None else repo
    problem = root_problem(o.root, sources.results)
    if problem:                                          # before the lock file: nothing is created or touched
        print(problem, file=sys.stderr)
        return 2
    root = Path(o.root)
    root.mkdir(parents=True, exist_ok=True)
    lock = open(root/'queue.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f'another driver is on {root}', file=sys.stderr)
        return 2

    def log(text):
        with open(root/'queue.log', 'a') as f:
            f.write(f'{utc()} {text}\n')
    log(f'start: root {root}; repo {repo}; {len(plan_cells(READING_ARMS))} reading entries and native runs, then {len(plan_cells(DESCRIPTION_ARMS))} of the description; oracle true')
    if (root/'queue_stop.json').exists():                # evidence, kept under its time; the file of that name always means this start
        (root/'queue_stop.json').rename(root/f"queue_stop_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    try:
        return queue_body(o, system, root, log, sources, repo)
    except (Stop, run.Stop) as stop:                     # run.Stop: code_checkout is MC0's
        log(f'STOP: {stop}')
        write_json(root/'queue_stop.json', dict(reason=str(stop), utc=utc()))
        return 8


# ---- evaluation of a native run ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def evaluate_one(o):
    require_cell(o.arm, o.pair)
    ev = seed.load_evaluate()
    ev.STAGE = core.STAGE                                # $ROOT/code has no data link
    rec = ev.evaluate_run(Path(o.run), o.arm, o.pair, o.run_id, method=method_name(o.arm))
    trace, _ = seed.injection_trace(o.run)
    arm_record = read_json(Path(o.run)/'seed_arm.json')
    rec['seed'] = dict(trace, arm=arm_record)
    rec['oracle'] = (read_json(Path(o.run)/'seed_entry.json') or {}).get('oracle') is True and (arm_record or {}).get('oracle') is True      # what the run's own records say
    rec['sv'] = dict(provenance=ARMS[o.arm]['provenance'], decides=ARMS[o.arm]['decides'], role=ROLE, plan='runs/20261104-sv/plan.md')
    write_json(Path(o.run)/'metrics.json', rec)
    x, c = rec.get('metrics') or {}, rec.get('certificate') or {}
    print(f"{o.arm} {short_name(o.pair)}: seg35 {x.get('seg35')} seg4 {x.get('seg4')} SDlogJ {x.get('sdlogj_official')} cert {c.get('status')} passed {c.get('passed')} alpha {trace.get('accepted_alpha')} entry check {trace.get('passed')}")
    return 0


# ---- the dry run ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def command_differences(arm, pair, results=None):
    """The prepare line and the unit of S_P of this pair (its cmd.txt, read only), with only the run directory, the stage, the unit and the cache root replaced, against this arm's: the tokens that still differ; None without the file."""
    results = RESULTS if results is None else results
    path = Path(results)/'seed_v1/runs/S_P'/pair/'cmd.txt'
    if not path.is_file():
        return None
    old_run, old_root = str(path.parent), str(Path(results)/'seed_v1')
    new_root = default_root(results)
    new_run = str(run_dir(new_root, arm, pair))
    lines = [shlex.split(line) for line in path.read_text().splitlines() if line and not line.startswith('#')]

    def replaced(tokens):
        return [t.replace(old_run, new_run).replace(old_root, str(new_root)).replace('SEED-S_P', f'SV-{arm}').replace(seed.unit_name('S_P', pair), unit_name(arm, pair)) for t in tokens]
    theirs = (replaced(lines[0]), replaced(lines[1]))
    ours = ([str(t) for t in prepare_argv(arm, pair, new_run, results)], [str(t) for t in unit_argv(arm, pair, new_run, new_root)])
    return [(a, b) for theirs_tokens, ours_tokens in zip(theirs, ours) for a, b in zip(theirs_tokens+['<end>'], ours_tokens+['<end>']) if a != b]


def rehearse_reading(scratch, out, results):
    """The reading on copies of the endpoints as the arms: S_P's copies lie under the line, S_T's over it.  Formats and pins only; only codes are printed (the two means are the endpoints' own, known); deleted with the scratch directory."""
    import decide_sv
    results = Path(results)
    for source, expect in (('S_P', decide_sv.NOT_ENOUGH), ('S_T', decide_sv.THIRD_ENOUGH)):
        here = scratch/f'rehearsal_{source}'/ROOT_NAME
        for arm, pair in plan_cells():
            doc = read_json(results/'seed_v1/runs'/source/pair/'metrics.json')
            doc.update(arm=arm, run_id=f'rehearsal_{arm}_{short_name(pair)}', method=method_name(arm), oracle=True)
            doc['seed'] = dict(doc['seed'], passed=True, problems=[], arm=dict(seconds=seed.SECONDS, changes=seed.CHANGES, inputs_root=str(inputs_root(arm, here)), charge_seconds=seed.ENTRY_CHARGE_SECONDS, oracle=True))
            target = run_dir(here, arm, pair)
            target.mkdir(parents=True)
            write_json(target/'metrics.json', doc)
        result = decide_sv.write_analysis(here, results=results)
        out(f"reading on copies of {source} as the arms: {result['reading_code']}; reference problems {result['reference_problems']}; files {sorted(p.name for p in (here/'analysis').iterdir())}; f0440 in the reading: {'f0440' in json.dumps(result['arms'])}")
        if result['reference_problems'] or result['reading_code'] != expect:
            raise SystemExit('the rehearsal of the reading did not give what it must')


def dry_run(scratch=None, out=print, sources=None, repo=None):
    """The scratch directory goes, however the dry run ends."""
    sources = Sources() if sources is None else sources
    repo = seed.MAIN_REPO if repo is None else repo
    made = scratch is None
    scratch = Path(tempfile.mkdtemp(prefix='sv_dry_')) if made else Path(scratch)
    if not made and scratch.exists():
        raise SystemExit(f'{scratch} exists (nothing is overwritten)')
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        return dry_run_steps(scratch, out, sources, repo)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def dry_run_steps(scratch, out, sources, repo):
    results = Path(sources.results)
    root = default_root(results)
    link = seed.frozen_dir(results)/'data/three_d/stage_L2R_val_v1'
    link_before = os.readlink(link)
    existed = root.exists()
    code = root/'code'
    out(f'SV dry run ({utc()}): no solver, no GPU (CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")!r}), no unit; everything written under a scratch directory, deleted at the end.  No label member of a pair.npz is decoded: `prepare` copies the files; the inputs here are A5P\'s, because the arms\' do not exist before the build.')
    counts = ', '.join(f"{arm} {len(spec['pairs'])} pairs" for arm, spec in ARMS.items())
    out(f"rule of the maps: L_f = argmax_l [(1 - f) p_l + f 1(l = truth)], ties {postmix.TIE_RULE}; {len(plan_cells())} entries and {len(plan_cells())} native runs and the control entry {CONTROL_ARM} {short_name(CONTROL_PAIR)}, root {root} (oracle true, isolated, in no result table): {counts}; reading only on {READING_ARMS}, {DESCRIPTION_ARMS} after it; --seconds {seed.SECONDS}; --geometry-weight-levels {','.join(f'{x:g}' for x in seed.GEOMETRY)}; entry charge {seed.ENTRY_CHARGE_SECONDS} s inside the clock; watchdog {seed.WATCHDOG} s")
    out(f'gate: no other tag-* or seq-vp-* unit, load < {LOAD_LIMIT}, no GPU compute process, GPU free >= {GPU_FREE_MIB} MiB, MemAvailable >= {RESERVE_MIN_MEM_GIB} GiB, /home >= {HOME_FLOOR_GIB} GiB, /mnt/c >= {HOST_FLOOR_GIB} GiB; {BETWEEN_RUNS_SECONDS} s between units; /home free now {shutil.disk_usage("/home").free/2**30:.0f} GiB, /mnt/c free now {shutil.disk_usage("/mnt/c").free/2**30:.0f} GiB')
    problems = list(input_problems(sources))
    out(f"\n==== the start checks of the queue (ids, provenance records, file presence; nothing decoded) ====\nproblems: {problems or 'none'}")
    system = System()
    registry = system.registry_problems()
    frozen = system.frozen_problems(results)
    out(f"the predictions' registry committed and clean: {not registry}; frozen worktree at {seed.FROZEN_COMMIT[:7]}, clean, sources equal to the freeze: {not frozen}")
    problems += registry+frozen
    pinned = fstar_problems(root, sources)
    out(f"f*: {pinned or 'pinned and backed by the probe'}{' (pending, not an error of the dry run: the queue stops at its start checks, before any run, until the probe has run and the lead has pinned)' if pinned else ''}")
    out(f"the reference of the control: {control_reference(results)} ({seed.entry_dir('S_P', CONTROL_PAIR, results)}/run.json)")
    mi = core.make_inputs()
    unet = mc.default_unet()
    admission = mi.a4_admission(list(SUBJECTS), sources.allowlist, unet)
    out(f"D24 admission of the {len(admission['subjects'])} subjects (ids only): folds sha256 {admission['folds_sha256']}; the only subjects: {', '.join(SUBJECTS)}")
    for subject in SUBJECTS:
        arms_of = {arm: label_tag(arm, subject) for arm, spec in ARMS.items() if any(subject in subjects_of(p) for p in spec['pairs'])}
        out(f'  {subject}: label map by arm {arms_of}; maps stored {tags_needed(subject)}')
    frozen_preopt = core.sha_file(core.FROZEN/'jd_curl/three_d/preopt.py') == core.FROZEN_PREOPT_SHA256
    out(f'frozen {core.FROZEN}: preopt.py sha256 equal to the pin: {frozen_preopt}; colour table sha256 {core.COLOUR_TABLE_SHA256} pinned and read')
    if not frozen_preopt:
        problems.append('preopt.py differs from the pin')

    out('\n==== the CPU probe, before anything else (the lead; no GPU; no label map written; run the mkdir first) ====')
    out(f'mkdir -p {root}')
    out(shlex.join(str(t) for t in probe_argv(repo, root)))
    out(f'then, with the f* it found: {seed.PY} -B {SCRIPT} pin-fstar --root {root} --by "WHO, WHEN"; git add {PINS_FILE.relative_to(ROOT)}; the f* into the revision of the plan; commit')

    out('\n==== the start strings of the units before the native runs ====')
    out(shlex.join(build_argv(code, root)))
    out(shlex.join(k1_argv(code, root)))
    out(shlex.join(entry_argv(EntryCell(CONTROL_ARM, TIER, CONTROL_PAIR), code, root, '<sha>')))
    for arm, pair in plan_cells():
        out(shlex.join(entry_argv(EntryCell(arm, TIER, pair), code, root, '<sha>')))
    out(shlex.join(entry_eval_argv(code, root)))
    out(shlex.join(entry_eval_argv(code, root, EVAL_UNIT_F0440)))

    out(f"\n==== all {len(plan_cells())} native runs: the real parser and prepare of the frozen source (scratch copy; a stage holding only the 5 pairs, A5P's files), check_prepared with the arm, the protocol and the two command lines against S_P of the same pair ====")
    copy = scratch/'frozen_copy'
    n_files = seed.frozen_copy(copy, results)
    out(f'frozen source copied: {n_files} files equal to the freeze hashes; the worktree link stays')
    five = scratch/'stage_of_5_pairs'
    five.mkdir()
    for name in ('selection.json', 'completion.json'):
        shutil.copyfile(Path(sources.stage)/name, five/name)
    for pair in PAIRS:
        (five/pair).symlink_to(seed.input_files('S_P', pair, results)[0])
    scratch_link = copy/'data/three_d/stage_L2R_val_v1'
    scratch_link.parent.mkdir(parents=True, exist_ok=True)
    scratch_link.symlink_to(five)
    prepared = scratch/'prepared'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    for k, (arm, pair) in enumerate(plan_cells(), 1):
        shown = run_dir(root, arm, pair)
        spec = ARMS[arm]
        out(f"# item {k}: arm {arm} ({'enters the reading' if spec['decides'] else 'description only, after the reading'}), provenance {spec['provenance']}, pair {pair}, label maps {label_tag(arm, subjects_of(pair)[0])}, {label_tag(arm, subjects_of(pair)[1])}, inputs {inputs_root(arm, root)}, entry {entry_dir(root, arm, pair)}/field.npz, unit {unit_name(arm, pair)}, run id sv_{arm}_{short_name(pair)}_<UTC>")
        out(shlex.join(str(t) for t in prepare_argv(arm, pair, shown, results)))
        out(shlex.join(str(t) for t in unit_argv(arm, pair, shown, root)))
        here = prepared/arm/pair
        here.parent.mkdir(parents=True, exist_ok=True)
        tokens = [str(t) for t in prepare_argv(arm, pair, here, results)]
        tokens[2] = str(copy/'scripts/run_stage_3d_e48_pg.py')
        tick = time.perf_counter()
        done = subprocess.run(tokens, cwd=scratch, env=env, capture_output=True, text=True)
        if done.returncode != 0:
            raise SystemExit(f'prepare refused {arm} {pair}: {done.stdout[-1500:]} {done.stderr[-1500:]}')
        files = seed.input_files('S_P', pair, results)
        check = system.check_prepared(here, pair, f'SV-{arm}', files[1], files[2], b1_entry(arm, root))
        if not check['ok']:
            raise SystemExit(f"check_prepared refused {arm} {pair}: {check['problems']}")
        control = results/'seed_v1/runs/S_P'/pair/'protocol.json'
        differs = seed.protocol_differences(read_json(here/'protocol.json'), read_json(control))
        tokens_differ = command_differences(arm, pair, results)
        out(f"parser, prepare and check_prepared ok ({time.perf_counter()-tick:.1f} s); protocol against S_P of this pair differs only in: {json.dumps(differs, ensure_ascii=False)}; command lines against S_P's cmd.txt, names replaced, differ in: {tokens_differ}")
        if set(differs) != {'stage'}:
            problems.append(f'{arm} {pair}: the protocol differs from S_P in {sorted(differs)}')
        if tokens_differ != []:
            problems.append(f'{arm} {pair}: the command lines differ from S_P in {tokens_differ}')
        shutil.rmtree(here)
    out("the real runs differ from these in the inputs only: input_sha256 and pair_source name the arm's files (check_prepared compares them with the arm's record at the start of each run)")

    out('\n==== the queue start string (the lead) ====')
    out(shlex.join(queue_launch(root, repo)))

    out('\n==== negative controls ====')
    tokens = [str(t) for t in prepare_argv(READING_ARMS[0], PAIRS[0], prepared/'gw', results)]
    tokens[2] = str(copy/'scripts/run_stage_3d_e48_pg.py')
    tokens[tokens.index('--geometry-weight-levels')+1] = '1,1,0.1,0.03'
    subprocess.run(tokens, cwd=scratch, env=env, capture_output=True, text=True, check=True)
    files = seed.input_files('S_P', PAIRS[0], results)
    check = system.check_prepared(prepared/'gw', PAIRS[0], f'SV-{READING_ARMS[0]}', files[1], files[2], b1_entry(READING_ARMS[0], root))
    out(f"configuration C's geometry weight against the arm: ok {check['ok']}: {check['problems']}")
    if check['ok']:
        problems.append('check_prepared took the wrong geometry weight')
    for call, what in ((lambda: require_subject('OASIS_OAS1_0444_MR1'), 'subject 0444'), (lambda: require_pair('OASIS_OAS1_0443_MR1__OASIS_OAS1_0444_MR1'), 'pair 0443-0444'), (lambda: require_cell('f0440', PAIRS[0]), 'f0440 on 0438-0439'), (lambda: label_path(results/'d0_v1_a5', SUBJECTS[0], 'r1_3', results), 'labels under d0_v1_a5')):
        try:
            call()
            problems.append(f'{what} was not refused')
            out(f'{what}: NOT refused')
        except SystemExit:
            out(f'{what}: refused')

    out('\n==== the reading on the real endpoints (copies as the arms: formats and pins only, not results) ====')
    rehearse_reading(scratch, out, results)

    if os.readlink(link) != link_before:
        problems.append("the frozen worktree's link changed")
    if root.exists() != existed:
        problems.append(f'{root} was created')
    out(f"\nthe frozen worktree's link unchanged: {link_before}; {root} {'exists' if existed else 'not created'}; no solver run; the scratch directory goes when this ends")
    out(f"torch imported by this dry run: {'torch' in sys.modules}; problems: {problems or 'none'}")
    return 8 if problems else 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['--dry-run']:
        d = argparse.ArgumentParser()
        d.add_argument('--dry-run', action='store_true')
        d.add_argument('--scratch', default=None)
        return dry_run(scratch=d.parse_args(argv).scratch)
    p = argparse.ArgumentParser(description='SV: see the module docstring')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('queue')
    q.add_argument('--root', default=str(DEFAULT_ROOT))
    q.add_argument('--wait-minutes', type=int, default=30)
    q.add_argument('--accept-control-difference', default=None)
    sub.add_parser('build').add_argument('--root', default=str(DEFAULT_ROOT))
    b = sub.add_parser('probe')
    b.add_argument('--root', default=str(DEFAULT_ROOT))
    b.add_argument('--workers', type=int, default=1)
    i = sub.add_parser('pin-fstar')
    i.add_argument('--root', default=str(DEFAULT_ROOT))
    i.add_argument('--by', required=True)
    r = sub.add_parser('run-entry')
    r.add_argument('--arm', required=True)
    r.add_argument('--pair', required=True)
    r.add_argument('--root', default=str(DEFAULT_ROOT))
    r.add_argument('--device', default='cuda')
    r.add_argument('--code-sha', default='n/a')
    sub.add_parser('evaluate-entries').add_argument('--root', default=str(DEFAULT_ROOT))
    e = sub.add_parser('evaluate-one')
    e.add_argument('run')
    e.add_argument('--arm', required=True)
    e.add_argument('--pair', required=True)
    e.add_argument('--run-id', required=True)
    sub.add_parser('decide').add_argument('--root', default=str(DEFAULT_ROOT))
    o = p.parse_args(argv)
    if o.command == 'queue':
        return queue(o)
    if o.command == 'build':
        return build_cli(o)
    if o.command == 'probe':
        return probe_cli(o)
    if o.command == 'pin-fstar':
        return pin_fstar(o.root, o.by)
    if o.command == 'run-entry':
        return run_entry(o)
    if o.command == 'evaluate-entries':
        return evaluate_entries(o)
    if o.command == 'evaluate-one':
        require_cell(o.arm, o.pair)
        if Path(o.run).resolve() != run_dir(DEFAULT_ROOT, o.arm, o.pair).resolve():     # it writes <RUN>/metrics.json: only this package's run of this arm and pair
            print(f'{o.run}: evaluate-one writes only to {run_dir(DEFAULT_ROOT, o.arm, o.pair)}; nothing read, nothing moved', file=sys.stderr)
            return 2
        return evaluate_one(o)
    problem = root_problem(o.root)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    import decide_sv
    decide_sv.write_analysis(o.root)
    return 0


if __name__ == '__main__':
    sys.exit(main())
