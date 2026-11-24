"""FD2 (runs/20261105-fd2/任务书.md, plan.md): FD's extension, the D factor above 6.  The arms, the cells and the guards of its own batch, root fd2_v1; fd_v1 and fl_v1 are only read.  Not a method: no F_B, no certified descent, only the final field passes the certificate; fields in no result table.

The rules of FD and FL do not move: fd_core.py, decide_fd.py, fl_core.py, decide_fl.py are not edited; their entries refuse every cell of this batch.  This module has the names of fl_core that run_fl.py and decide_fl.py use (a test reads them from their source), so that run_fd2.py and decide_fd2.py can run another instance of those two modules on this batch, as FD does.

Cells.  A field run: Cell(arm, 'first5', pair), 600 native steps, one checkpoint; Hx6 only on 0438-0439, the control against fd_v1's Hx6 at 600 steps; Hx6L 1200 native steps, checkpoints 600 and 1200, only described.  Cell(arm, 'rest600', pair), the other 14 pairs, only by the lead's separate queue, only Hx8, Hx12, Hx16, Hx24.
A repair: Cell('R'+arm, 'c600', pair), only when the code chose the arm (analysis/repair_selection_1.json), never of Hx6 or Hx6L; Cell('R'+arm, 'rest600', pair).
No oracle arm.  No function of this module reads a label except read_truth (fl_core's), for the evaluation.
"""
from dataclasses import dataclass
from pathlib import Path

import fl_core as fl
import mc0_core as core

RESULTS = fl.RESULTS
DEFAULT_ROOT = RESULTS/'fd2_v1'
FD_ROOT_NAME = 'fd_v1'                          # read only: the control and the reference
EF_ROOT, MC0_ROOT = fl.EF_ROOT, fl.MC0_ROOT     # read only, in the input checks
ROLE = 'FD2: FL working point probe, D factor above 6, K-channel T88 and local repair, predicted channels, no scalar native level; NOT oracle, NOT a method, NOT in any result table'
ORACLE_ROLE = 'no oracle arm in FD2'
BATCH_SCHEMA = 'bayesreg.fd2.batch.v1'
FIRST = fl.FIRST
NATIVE_LEVEL = fl.NATIVE_LEVEL
COARSE_ITERATIONS = fl.COARSE_ITERATIONS
NATIVE = 600
LONG = 1200
RUN_CAP_SECONDS = fl.RUN_CAP_SECONDS
TASK_BUDGET_SECONDS = fl.TASK_BUDGET_SECONDS
BZ_SECONDS = 90.                                # BZ's own line: a column, not a premise
REPAIR_EPS = fl.REPAIR_EPS                      # the rest of the settings: RepairConfig's defaults, as ef_v1's RF05; the wall stays 120 s
REPAIR_WALL_SECONDS = 120.                      # RepairConfig's default, for the words and the estimates; the repair reads its own config
REPAIR_STEPS = (NATIVE,)                        # only this batch's
REPAIR_TOP = 4
ORACLE_ARMS = ()
SIGMA_ARMS = ()
EF_REPAIRS = {}
FULL, TWO = (NATIVE,), (NATIVE,)
REST_PAIRS = fl.REST_PAIRS
CONTROL_ARM = 'Hx6'
DESCRIBED_ARM = 'Hx6L'
NEW_ARMS = ('Hx8', 'Hx12', 'Hx16', 'Hx24')      # in the order of the D factor: the order of the repairs
FIRST_PHASE = NEW_ARMS
CONTROL_PAIR = core.PAIRS[0]                    # 0438-0439


def _arm(factor, **changes):
    """FL's Hx (Q3h, sigma_field .5, K = 35) at this D factor, 600 native steps, one checkpoint unless said."""
    return dict(dict(fl.ARMS['Hx'], native=NATIVE, checkpoints=(NATIVE,), D_factor=factor), **changes)


ARMS = dict(Hx6=_arm(6.), Hx8=_arm(8.), Hx12=_arm(12.), Hx16=_arm(16.), Hx24=_arm(24.), Hx6L=_arm(6., native=LONG, checkpoints=(NATIVE, LONG)))
# run_fl.py and decide_fl.py ask the batch for these; they are about the root and the pairs, not about FL's arms
selection_hashes, write_selection, read_selection, SelectionChanged, SELECTIONS_LOG = fl.selection_hashes, fl.write_selection, fl.read_selection, fl.SelectionChanged, fl.SELECTIONS_LOG
read_truth, require_truth_pair, ef_dir, field_kernel = fl.read_truth, fl.require_truth_pair, fl.ef_dir, fl.field_kernel


@dataclass(frozen=True)
class Cell(core.Cell):
    @property
    def run_id(self):
        return f'fd2_{self.arm}_{self.tier}_{self.short}'

    @property
    def unit(self):
        return f'tag-fd2-{self.arm.lower()}-{self.tier}-{self.short}'

    @property
    def is_repair(self):
        return self.arm[:1] == 'R' and self.arm[1:] in ARMS

    @property
    def is_oracle(self):
        return False

    @property
    def base(self):
        """The arm whose field a repair starts from; the arm itself otherwise."""
        return self.arm[1:] if self.is_repair else self.arm


def steps_of(cell):
    """The step of a repair's source, or the last native step of a field run."""
    if cell.tier == FIRST:
        return ARMS[cell.arm]['native']
    return fl._steps_in(cell.tier, 'rest') or fl._steps_in(cell.tier, 'c')


def checkpoints_of(cell):
    return ARMS[cell.arm]['checkpoints'] if cell.tier == FIRST else (steps_of(cell),)


def _refuse(arm, tier, pair):
    raise SystemExit(f'FD2 has no cell arm {arm} tier {tier} pair {pair}: arms {NEW_ARMS} on the first 5 development pairs {[core.short_name(p) for p in core.PAIRS]}, the other 14 only with --pairs rest14; the control {CONTROL_ARM} only on {core.short_name(CONTROL_PAIR)}, {DESCRIBED_ARM} only on the first 5; those two never repaired')


def require_cell(arm, tier, pair):
    """Checked before any file of the pair is opened.  Every cell of every entry comes through here."""
    first, rest = pair in core.PAIRS, pair in REST_PAIRS
    if not (first or rest):
        _refuse(arm, tier, pair)
    if arm == CONTROL_ARM:
        ok = tier == FIRST and pair == CONTROL_PAIR
    elif arm == DESCRIBED_ARM:
        ok = tier == FIRST and first
    elif arm in NEW_ARMS:
        ok = (tier == FIRST and first) or (tier == f'rest{NATIVE}' and rest)
    elif arm[:1] == 'R' and arm[1:] in NEW_ARMS:
        ok = (tier == f'c{NATIVE}' and first) or (tier == f'rest{NATIVE}' and rest)
    else:
        ok = False
    if not ok:
        _refuse(arm, tier, pair)
    return Cell(arm, tier, pair)


def run_dir(root, cell):
    return Path(root)/'runs'/f'{cell.arm}_{cell.tier}'/cell.short


def root_problem(root):
    """Why this entry may not write at this root, or None: only its own."""
    return None if Path(root).resolve() == Path(DEFAULT_ROOT).resolve() else f'{root}: FD2 writes only to {DEFAULT_ROOT}; nothing read, nothing moved'


def out_problem(out, cell):
    """Why a run may not write at this directory, or None: exactly its own, so never under fd_v1 or fl_v1."""
    want = run_dir(DEFAULT_ROOT, cell)
    return None if Path(out).resolve() == want.resolve() else f'{out}: {cell.run_id} writes only to {want}'


def fd_root(results=None):
    """fd_v1: read only, by FD's own decide_fd and by run_fd2's control."""
    return Path(RESULTS if results is None else results)/FD_ROOT_NAME


# ---- the plan ---------------------------------------------------------------------------------------------------------------------------------
def plan_control():
    return [Cell(CONTROL_ARM, FIRST, CONTROL_PAIR)]


def plan_new():
    """The 20 runs, in order.  Pair-major, the arm order rotating with the pair index (MC0P's rule)."""
    arms = NEW_ARMS
    return [Cell(arms[(i+j) % len(arms)], FIRST, pair) for i, pair in enumerate(core.PAIRS) for j in range(len(arms))]


def plan_described():
    """Hx6L on the first 5 pairs, after the others: a stop costs description only."""
    return [Cell(DESCRIBED_ARM, FIRST, pair) for pair in core.PAIRS]


def plan_fields():
    return plan_control()+plan_new()+plan_described()


def plan_x():
    return []


def plan_ef_repairs():
    return []


def repair_cells(chosen):
    """The repairs of the chosen arms [{arm, step}, ...], each on the first 5 pairs."""
    return [Cell('R'+c['arm'], f"c{c['step']}", pair) for c in chosen for pair in core.PAIRS]


def plan_rest(arm, steps):
    """The lead's separate queue: the fields of the other 14 pairs, then their repairs."""
    fields = [require_cell(arm, f'rest{steps}', pair) for pair in REST_PAIRS]
    return fields, [require_cell('R'+arm, f'rest{steps}', pair) for pair in REST_PAIRS]


def x_sigma(root):
    """FD2 has no X."""
    return None


def chosen_repairs(root):
    """The arms through the gate, by D factor; the queue stops at the first that does not reach 5/5."""
    return (read_selection(root, 'repair_selection_1.json') or {}).get('chosen', [])


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
    """Everything this root's plan holds now, in the order of the phases."""
    fields, repairs = rest_cells(root)
    return plan_fields()+repair_cells(chosen_repairs(root))+fields+repairs


def sigma_of(cell, root):
    return ARMS[cell.arm]['sigma_field']


def arm_config(cell, sigma):
    """The PRE of configuration C, as FD's arms."""
    return dict(core.PRE_C, seconds=RUN_CAP_SECONDS, iterations=COARSE_ITERATIONS+(steps_of(cell),), sigma_field=sigma)


def read_images(cell):
    """The PairInput of a cell of the plan: load_pair's firewall view, `fixed` and `moving` (subject guard, sha256 against the manifest)."""
    require_cell(cell.arm, cell.tier, cell.pair)
    from bayesreg.io.pair import load_pair
    return load_pair(core.STAGE/cell.pair/'pair.npz', allowlist=core.ALLOWLIST)
