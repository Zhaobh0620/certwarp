"""FD (runs/20261105-fd/任务书.md, plan.md): FL's working point, the D factor above 2.  The arms, the cells and the guards of its own batch, root fd_v1; fl_v1 is only read.  Not a method: no F_B, no certified descent, only the final field passes the certificate; fields in no result table.

The FL rules do not move: fl_core.py and decide_fl.py are not edited, FL's REPAIR_STEPS stay (2400, 4800), and FL's own entry refuses every cell of this batch.  This module has the names of fl_core that run_fl.py and decide_fl.py use (a test reads them from their source), so that run_fd.py and decide_fd.py can run a second instance of those two modules on this batch instead of copying them.

Cells.  A field run: Cell(arm, 'first5', pair), 600 native steps, one checkpoint; Hx only on 0438-0439, the control against fl_v1's Hx at 600 steps.  Cell(arm, 'rest600', pair), the other 14 pairs, only by the lead's separate queue, never Hx.
A repair: Cell('R'+arm, 'c600', pair), only when the code chose the arm (analysis/repair_selection_1.json), never of the control Hx; Cell('R'+arm, 'rest600', pair).
Only described: Cell('RHx600', 'c600', pair), a repair of fl_v1's own Hx at 600 steps (the lead, plan revision 2), written under fd_v1; never a candidate, never in the reading, its failure changes nothing.
No oracle arm.  No function of this module reads a label except read_truth (fl_core's), for the evaluation.
"""
from dataclasses import dataclass
from pathlib import Path

import fl_core as fl
import mc0_core as core

RESULTS = fl.RESULTS
DEFAULT_ROOT = RESULTS/'fd_v1'
FL_ROOT_NAME = 'fl_v1'                          # read only: the control and the references
EF_ROOT, MC0_ROOT = fl.EF_ROOT, fl.MC0_ROOT     # read only, in the input checks
ROLE = 'FL working point probe: K-channel T88 with a higher D factor and local repair, predicted channels, no scalar native level; NOT oracle, NOT a method, NOT in any result table'
ORACLE_ROLE = 'no oracle arm in FD'
BATCH_SCHEMA = 'bayesreg.fd.batch.v1'
FIRST = fl.FIRST
NATIVE_LEVEL = fl.NATIVE_LEVEL
COARSE_ITERATIONS = fl.COARSE_ITERATIONS
NATIVE = 600
RUN_CAP_SECONDS = fl.RUN_CAP_SECONDS
TASK_BUDGET_SECONDS = fl.TASK_BUDGET_SECONDS
REPAIR_EPS = fl.REPAIR_EPS                      # the rest of the settings: RepairConfig's defaults, as ef_v1's RF05
REPAIR_STEPS = (NATIVE,)                        # only this batch's: FL's stay (2400, 4800)
REPAIR_TOP = 2
ORACLE_ARMS = ()
SIGMA_ARMS = ()
EF_REPAIRS = {}
FL_REPAIRS = dict(RHx600='Hx')                  # the lead, plan revision 2: only described; the field is fl_v1's, the repair and its records are this root's
FULL, TWO = (NATIVE,), (NATIVE,)
REST_PAIRS = fl.REST_PAIRS
CONTROL_ARM = 'Hx'
NEW_ARMS = ('Hx3', 'Hx4', 'Hx6', 'Fx2', 'Fx4')
FIRST_PHASE = NEW_ARMS
CONTROL_PAIR = core.PAIRS[0]                    # 0438-0439


def _arm(source, **changes):
    """An arm of FL at 600 native steps, one checkpoint: the channels, the weights and sigma_field are FL's."""
    return dict(fl.ARMS[source], native=NATIVE, checkpoints=(NATIVE,), **changes)


ARMS = dict(Hx=_arm('Hx'),
            Hx3=_arm('Hx', D_factor=3.),
            Hx4=_arm('Hx', D_factor=4.),
            Hx6=_arm('Hx', D_factor=6.),
            Fx2=_arm('F', D_factor=2.),
            Fx4=_arm('F', D_factor=4.))
# run_fl.py and decide_fl.py ask the batch for these; they are about the root and the pairs, not about FL's arms
selection_hashes, write_selection, read_selection, SelectionChanged, SELECTIONS_LOG = fl.selection_hashes, fl.write_selection, fl.read_selection, fl.SelectionChanged, fl.SELECTIONS_LOG
read_truth, require_truth_pair, ef_dir, field_kernel = fl.read_truth, fl.require_truth_pair, fl.ef_dir, fl.field_kernel


@dataclass(frozen=True)
class Cell(core.Cell):
    @property
    def run_id(self):
        return f'fd_{self.arm}_{self.tier}_{self.short}'

    @property
    def unit(self):
        return f'tag-fd-{self.arm.lower()}-{self.tier}-{self.short}'

    @property
    def is_repair(self):
        return self.arm in FL_REPAIRS or (self.arm[:1] == 'R' and self.arm[1:] in ARMS)

    @property
    def is_oracle(self):
        return False

    @property
    def base(self):
        """The arm whose field a repair starts from; the arm itself otherwise."""
        return FL_REPAIRS.get(self.arm, self.arm[1:]) if self.is_repair else self.arm


def steps_of(cell):
    """The step of a repair's source, or the last native step of a field run."""
    if cell.tier == FIRST:
        return ARMS[cell.arm]['native']
    return fl._steps_in(cell.tier, 'rest') or fl._steps_in(cell.tier, 'c')


def checkpoints_of(cell):
    return (steps_of(cell),)


def _refuse(arm, tier, pair):
    raise SystemExit(f'FD has no cell arm {arm} tier {tier} pair {pair}: arms {NEW_ARMS} on the first 5 development pairs {[core.short_name(p) for p in core.PAIRS]}, the other 14 only with --pairs rest14; the control {CONTROL_ARM} only on {core.short_name(CONTROL_PAIR)}, never repaired')


def require_cell(arm, tier, pair):
    """Checked before any file of the pair is opened.  Every cell of every entry comes through here."""
    first, rest = pair in core.PAIRS, pair in REST_PAIRS
    if not (first or rest):
        _refuse(arm, tier, pair)
    if arm == CONTROL_ARM:
        ok = tier == FIRST and pair == CONTROL_PAIR
    elif arm in NEW_ARMS:
        ok = (tier == FIRST and first) or (tier == f'rest{NATIVE}' and rest)
    elif arm[:1] == 'R' and arm[1:] in NEW_ARMS:
        ok = (tier == f'c{NATIVE}' and first) or (tier == f'rest{NATIVE}' and rest)
    elif arm in FL_REPAIRS:
        ok = tier == f'c{NATIVE}' and first
    else:
        ok = False
    if not ok:
        _refuse(arm, tier, pair)
    return Cell(arm, tier, pair)


def run_dir(root, cell):
    return Path(root)/'runs'/f'{cell.arm}_{cell.tier}'/cell.short


def root_problem(root):
    """Why this entry may not write at this root, or None: only its own."""
    return None if Path(root).resolve() == Path(DEFAULT_ROOT).resolve() else f'{root}: FD writes only to {DEFAULT_ROOT}; nothing read, nothing moved'


def out_problem(out, cell):
    """Why a run may not write at this directory, or None: exactly its own, so never under fl_v1."""
    want = run_dir(DEFAULT_ROOT, cell)
    return None if Path(out).resolve() == want.resolve() else f'{out}: {cell.run_id} writes only to {want}'


def fl_root(results=None):
    """fl_v1: read only, by the original decide_fl and by run_fd's control."""
    return Path(RESULTS if results is None else results)/FL_ROOT_NAME


# ---- the plan ---------------------------------------------------------------------------------------------------------------------------------
def plan_control():
    return [Cell(CONTROL_ARM, FIRST, CONTROL_PAIR)]


def plan_new():
    """The 25 runs, in order.  Pair-major, the arm order rotating with the pair index (MC0P's rule)."""
    arms = NEW_ARMS
    return [Cell(arms[(i+j) % len(arms)], FIRST, pair) for i, pair in enumerate(core.PAIRS) for j in range(len(arms))]


def plan_fields():
    return plan_control()+plan_new()


def plan_x():
    return []


def plan_fl_repairs():
    """RHx600 on the first 5 pairs, after the chosen repairs: only described."""
    return [Cell(arm, f'c{NATIVE}', pair) for arm in FL_REPAIRS for pair in core.PAIRS]


def plan_ef_repairs():
    """The name decide_fl's collect asks for: the repairs that are only described, whose place FL's RF150, RF300 have there."""
    return plan_fl_repairs()


def repair_cells(chosen):
    """The repairs of the chosen arms [{arm, step}, ...], best first, each on the first 5 pairs."""
    return [Cell('R'+c['arm'], f"c{c['step']}", pair) for c in chosen for pair in core.PAIRS]


def plan_rest(arm, steps):
    """The lead's separate queue: the fields of the other 14 pairs, then their repairs."""
    fields = [require_cell(arm, f'rest{steps}', pair) for pair in REST_PAIRS]
    return fields, [require_cell('R'+arm, f'rest{steps}', pair) for pair in REST_PAIRS]


def x_sigma(root):
    """FD has no X."""
    return None


def chosen_repairs(root):
    """The arms the code chose so far."""
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
    return plan_fields()+repair_cells(chosen_repairs(root))+plan_fl_repairs()+fields+repairs


def sigma_of(cell, root):
    return ARMS[cell.arm]['sigma_field']


def arm_config(cell, sigma):
    """The PRE of configuration C, as FL's arms, at 600 native steps: F's and Hx's 600-step fields of fl_v1 had the same path (the path does not depend on the total steps, runs/20261105-fl/plan.md)."""
    return dict(core.PRE_C, seconds=RUN_CAP_SECONDS, iterations=COARSE_ITERATIONS+(steps_of(cell),), sigma_field=sigma)


def read_images(cell):
    """The PairInput of a cell of the plan: load_pair's firewall view, `fixed` and `moving` (subject guard, sha256 against the manifest)."""
    require_cell(cell.arm, cell.tier, cell.pair)
    from bayesreg.io.pair import load_pair
    return load_pair(core.STAGE/cell.pair/'pair.npz', allowlist=core.ALLOWLIST)
