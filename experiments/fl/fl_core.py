"""FL (runs/20261105-fl/任务书.md, plan.md; docs/bayesreg/notes/追088_12小时规划_20261105.md §2): the arms, the cells and the guards of K-channel T88 with local repair and no scalar native level.  Not a method: no F_B, no certified descent, only the final field passes the certificate; fields in no result table.

Cells.  A field run: Cell(arm, 'first5', pair), first 5 development pairs, the arm's checkpoints; Cell(arm, 'rest<steps>', pair), the other 14 pairs, only by the lead's separate queue (--pairs rest14), one field at <steps>.
A repair: Cell('R'+arm, 'c<steps>', pair) of that checkpoint of the first 5 pairs, only when the code chose it (analysis/repair_selection_*.json); Cell('R'+arm, 'rest<steps>', pair); Cell('RF150' or 'RF300', 'ef', pair) of ef_v1's F150, F300 fields, read only.
Oracle: arm O only, truth one-hot, `oracle: true`, written only under <root>/oracle_isolated; never repaired, in no table of the reading.  No function of this module reads a label except read_truth, for the evaluation.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

import mc0_core as core

RESULTS = Path('/home/rmcfz/seq_vp_tag_results')
DEFAULT_ROOT = RESULTS/'fl_v1'
ORACLE_DIR = 'oracle_isolated'
EF_ROOT, MC0_ROOT = 'ef_v1', 'mc0_v1'           # read only: the sources of RF150, RF300 and the controls
ROLE = 'K-channel T88 and local repair with predicted channels, no scalar native level: a probe, NOT oracle, NOT a method, NOT in any result table'
ORACLE_ROLE = 'oracle ceiling of the same path: truth one-hot channels, isolated, NOT a method, NOT in any result table'
BATCH_SCHEMA = 'bayesreg.fl.batch.v1'
SELECTIONS_LOG = 'selections.sha256'
FIRST = 'first5'
NATIVE_LEVEL = 3
COARSE_ITERATIONS = (50, 50, 100)               # MC0's tiers
RUN_CAP_SECONDS = core.RUN_CAP_SECONDS          # 900 s: above every arm (X about 280 s); the 330 s is a premise of the reading, never a stop of a run.  The brief allows raising it here; nothing needs it
TASK_BUDGET_SECONDS = 330.
REPAIR_EPS = .05                                # the rest of the settings: RepairConfig's defaults, as ef_v1's RF05
REPAIR_STEPS = (2400, 4800)                     # the checkpoints a repair may start from
REPAIR_TOP = 3
SIGMA_ARMS = ('F', 'M', 'S', 'T')               # X takes its sigma_field from these
ORACLE_ARMS = ('O',)
REST_PAIRS = tuple(f'OASIS_OAS1_{n:04d}_MR1__OASIS_OAS1_{n+1:04d}_MR1' for n in range(443, 457))     # runs/20261104-seed19: selection.json, pairs 6..19 (the dry run compares them; a test compares them with experiments/mc0/run_mc0pc.py and experiments/seed/run_seed19.py)
FULL, TWO = (600, 1200, 2400), (600, 2400)
# channels: 'onehot' G_.75 of argmax of the predictions (ef_v1's F600), 'onehot+intensity' the image first (the rule of P1, Q1: unsmoothed, no weight of its own), 'truth' MC0's P2.  weight: an arm of mc0p_core.ARMS whose rule of w_l is used.  sigma_field None: chosen by the code (decide_fl.select_x).
ARMS = dict(F=dict(k=35, sigma_field=.5, native=2400, checkpoints=FULL, channels='onehot', weight=None, D_factor=1.),
            M=dict(k=35, sigma_field=.42, native=2400, checkpoints=FULL, channels='onehot', weight=None, D_factor=1.),     # the lead, plan revision 1: between F and S by the variance per step (field_kernel), not by the name
            S=dict(k=35, sigma_field=.35, native=2400, checkpoints=FULL, channels='onehot', weight=None, D_factor=1.),
            T=dict(k=35, sigma_field=.25, native=2400, checkpoints=FULL, channels='onehot', weight=None, D_factor=1.),
            H=dict(k=35, sigma_field=.5, native=2400, checkpoints=TWO, channels='onehot', weight='Q3h', D_factor=1.),
            Hx=dict(k=35, sigma_field=.5, native=2400, checkpoints=TWO, channels='onehot', weight='Q3h', D_factor=2.),
            I=dict(k=36, sigma_field=.5, native=2400, checkpoints=TWO, channels='onehot+intensity', weight=None, D_factor=1.),
            O=dict(k=35, sigma_field=.5, native=2400, checkpoints=TWO, channels='truth', weight=None, D_factor=1.),
            X=dict(k=35, sigma_field=None, native=4800, checkpoints=(2400, 4800), channels='onehot', weight=None, D_factor=1.))
FIRST_PHASE = ('F', 'M', 'S', 'T', 'H', 'Hx', 'I')     # the arms of predicted channels of the first phase; O after them, in the same phase: a STOP there still means no evaluation until a restart (nothing is lost, a restart reruns nothing)
EF_REPAIRS = dict(RF150='F150', RF300='F300')


@dataclass(frozen=True)
class Cell(core.Cell):
    @property
    def run_id(self):
        return f'fl_{self.arm}_{self.tier}_{self.short}'

    @property
    def unit(self):
        return f'tag-fl-{self.arm.lower()}-{self.tier}-{self.short}'

    @property
    def is_repair(self):
        return self.arm in EF_REPAIRS or (self.arm[:1] == 'R' and self.arm[1:] in ARMS)

    @property
    def is_oracle(self):
        return self.arm in ORACLE_ARMS

    @property
    def base(self):
        """The arm whose field a repair starts from; the arm itself otherwise."""
        return EF_REPAIRS.get(self.arm, self.arm[1:]) if self.is_repair else self.arm


def field_kernel(sigma):
    """What sigma_field does in each step, by the formula of mc0_core._gaussian_1d (a test compares): radius ceil(3 sigma), normalized.  The ladder of the names is not the ladder of the smoothing: the kernel is sampled and truncated.  variance per step per axis; checkerboard_gain per step per axis, the node-level mode the cell-centre terms and the official SDlogJ do not see."""
    radius = int(np.ceil(3*sigma))
    x = np.arange(-radius, radius+1, dtype=np.float64)
    k = np.exp(-.5*(x/sigma)**2)
    k /= k.sum()
    gain = float((k*np.cos(np.pi*x)).sum())
    return dict(sigma_field=sigma, radius=radius, centre=float(k[radius]), first=float(k[radius+1]), variance_per_step=float((k*x*x).sum()), continuous_variance=sigma**2, checkerboard_gain=gain, checkerboard_after_600=gain**600, checkerboard_after_2400=gain**2400)


def _steps_in(tier, prefix):
    match = re.fullmatch(prefix+r'(\d+)', tier)
    return None if match is None else int(match.group(1))


def steps_of(cell):
    """The step of a repair's source, or the last native step of a field run."""
    if cell.tier == FIRST:
        return ARMS[cell.arm]['native']
    if cell.tier == 'ef':
        return None
    return _steps_in(cell.tier, 'rest') or _steps_in(cell.tier, 'c')


def checkpoints_of(cell):
    return ARMS[cell.arm]['checkpoints'] if cell.tier == FIRST else (steps_of(cell),)


def _refuse(arm, tier, pair):
    raise SystemExit(f'FL has no cell arm {arm} tier {tier} pair {pair}: the first 5 development pairs {[core.short_name(p) for p in core.PAIRS]}, the other 14 only with --pairs rest14 and never the oracle arm')


def require_cell(arm, tier, pair):
    """Checked before any file of the pair is opened.  Every cell of every entry comes through here."""
    first, rest = pair in core.PAIRS, pair in REST_PAIRS
    if not (first or rest):
        _refuse(arm, tier, pair)
    if arm in ARMS:
        if tier == FIRST:
            ok = first
        else:
            ok = rest and _steps_in(tier, 'rest') in ARMS[arm]['checkpoints'] and arm not in ORACLE_ARMS
    elif arm in EF_REPAIRS:
        ok = tier == 'ef' and first
    elif arm[:1] == 'R' and arm[1:] in ARMS and arm[1:] not in ORACLE_ARMS:
        spec = ARMS[arm[1:]]
        steps, rest_steps = _steps_in(tier, 'c'), _steps_in(tier, 'rest')
        ok = (first and steps in REPAIR_STEPS and steps in spec['checkpoints']) or (rest and rest_steps in spec['checkpoints'])
    else:
        ok = False
    if not ok:
        _refuse(arm, tier, pair)
    return Cell(arm, tier, pair)


def require_truth_pair(pair):
    """The evaluation's guard: only a pair of the plan."""
    if pair not in core.PAIRS and pair not in REST_PAIRS:
        _refuse('evaluation', '-', pair)
    return pair


def run_dir(root, cell):
    """<root>/runs/<arm>_<tier>/<ffff>-<mmmm>; the oracle arm only under <root>/oracle_isolated."""
    base = Path(root)/ORACLE_DIR if cell.is_oracle else Path(root)
    return base/'runs'/f'{cell.arm}_{cell.tier}'/cell.short


def root_problem(root):
    """Why this entry may not write at this root, or None: only its own."""
    return None if Path(root).resolve() == Path(DEFAULT_ROOT).resolve() else f'{root}: FL writes only to {DEFAULT_ROOT}; nothing read, nothing moved'


def out_problem(out, cell):
    """Why a run may not write at this directory, or None: exactly its own, which keeps the oracle arm in the isolated directory and every other arm out of it."""
    want = run_dir(DEFAULT_ROOT, cell)
    return None if Path(out).resolve() == want.resolve() else f'{out}: {cell.run_id} writes only to {want}'


# ---- the plan ---------------------------------------------------------------------------------------------------------------------------------
def plan_fields():
    """The first phase, in order.  Pair-major, the arm order rotating with the pair index (MC0P's rule, so that no arm is always first or always after the same one), then the oracle arm's 5 runs."""
    arms = FIRST_PHASE
    runs = [Cell(arms[(i+j) % len(arms)], FIRST, pair) for i, pair in enumerate(core.PAIRS) for j in range(len(arms))]
    return runs+[Cell(arm, FIRST, pair) for arm in ORACLE_ARMS for pair in core.PAIRS]


def plan_x():
    return [Cell('X', FIRST, pair) for pair in core.PAIRS]


def plan_ef_repairs():
    return [Cell(arm, 'ef', pair) for arm in EF_REPAIRS for pair in core.PAIRS]


def repair_cells(chosen):
    """The repairs of the chosen checkpoints [{arm, step}, ...], best first, each on the first 5 pairs."""
    return [Cell('R'+c['arm'], f"c{c['step']}", pair) for c in chosen for pair in core.PAIRS]


def plan_rest(arm, steps):
    """The lead's separate queue: the fields of the other 14 pairs, then their repairs."""
    fields = [require_cell(arm, f'rest{steps}', pair) for pair in REST_PAIRS]
    return fields, [require_cell('R'+arm, f'rest{steps}', pair) for pair in REST_PAIRS]


class SelectionChanged(SystemExit):
    """A choice of the code that is not the file it wrote."""


def selection_hashes(root):
    """{file name: sha256}: written when the code chose, one line each."""
    path = Path(root)/'analysis'/SELECTIONS_LOG
    if not path.is_file():
        return {}
    return {name: sha for sha, name in (line.split() for line in path.read_text().splitlines() if line.strip())}


def write_selection(root, name, value):
    """The code's choice: the file and its sha256, once.  Returns the sha256."""
    out = Path(root)/'analysis'
    out.mkdir(parents=True, exist_ok=True)
    if (out/name).exists():
        raise SelectionChanged(f'{out/name} is there: a choice is written once')
    text = json.dumps(value, indent=1, ensure_ascii=False)+'\n'
    (out/name).write_text(text, encoding='utf-8')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    with open(out/SELECTIONS_LOG, 'a') as f:
        f.write(f'{sha} {name}\n')
    return sha


def read_selection(root, name):
    """The code's choice, None if it has not chosen.  A file that is not what it wrote stops whoever asks: the repairs, X's sigma, the reading."""
    path = Path(root)/'analysis'/name
    if not path.is_file():
        return None
    sha = core.sha_file(path)
    if selection_hashes(root).get(name) != sha:
        raise SelectionChanged(f'{path}: sha256 {sha} is not the one written when the code chose ({selection_hashes(root).get(name)}); nothing is read from it')
    return json.loads(path.read_text(encoding='utf-8'))


def x_sigma(root):
    """X's sigma_field as the code chose it; None before the choice, or when no arm qualified."""
    return (read_selection(root, 'x_selection.json') or {}).get('sigma_field')


def chosen_repairs(root):
    """Every checkpoint the code chose so far."""
    out = []
    for name in ('repair_selection_1.json', 'repair_selection_2.json'):
        out += (read_selection(root, name) or {}).get('chosen', [])
    return out


def rest_cells(root):
    """The cells of the lead's queues so far, from the directories."""
    fields, repairs = [], []
    for d in sorted((Path(root)/'runs').glob('*_rest*')):
        arm, tier = d.name.split('_', 1)
        if arm in ARMS:
            fields += [Cell(arm, tier, pair) for pair in REST_PAIRS]
        else:
            repairs += [Cell(arm, tier, pair) for pair in REST_PAIRS]
    return fields, repairs


def all_cells(root):
    """Everything this root's plan holds now: the static cells and what the code chose, in the order of the phases."""
    fields, repairs = rest_cells(root)
    return plan_fields()+plan_ef_repairs()+repair_cells(chosen_repairs(root))+(plan_x() if x_sigma(root) is not None else [])+fields+repairs


def sigma_of(cell, root):
    """The sigma_field of a field run.  STOP when X has none."""
    sigma = ARMS[cell.arm]['sigma_field']
    if sigma is None:
        sigma = x_sigma(root)
        if sigma not in tuple(ARMS[arm]['sigma_field'] for arm in SIGMA_ARMS):
            raise SystemExit(f'{cell.run_id}: analysis/x_selection.json under {root} gives sigma_field {sigma}: the code chose none of {SIGMA_ARMS}')
    return sigma


def arm_config(cell, sigma):
    """The PRE of configuration C, as MC0's tiers, with the cell's native steps and sigma_field.  The path does not depend on the native steps: see plan.md."""
    return dict(core.PRE_C, seconds=RUN_CAP_SECONDS, iterations=COARSE_ITERATIONS+(steps_of(cell),), sigma_field=sigma)


def ef_dir(results, arm, pair):
    """A field run of ef_v1: read only."""
    return Path(results)/EF_ROOT/'runs'/f'{arm}_fine'/core.short_name(pair)


def read_images(cell):
    """The PairInput of a cell of the plan: load_pair's firewall view, `fixed` and `moving` (subject guard, sha256 against the manifest)."""
    require_cell(cell.arm, cell.tier, cell.pair)
    from bayesreg.io.pair import load_pair
    return load_pair(core.STAGE/cell.pair/'pair.npz', allowlist=core.ALLOWLIST)


def read_truth(pair, members):
    """The named truth label members.  Only the evaluation calls this (the oracle arm has its own reader, fl_oracle.py); no channel path or repair path does (a test reads their source)."""
    require_truth_pair(pair)
    from bayesreg.io.pair import load_pair
    path = core.STAGE/pair/'pair.npz'
    pin = load_pair(path, allowlist=core.ALLOWLIST)
    with np.load(path) as z:
        return pin, {k: np.array(z[k]) for k in members}
