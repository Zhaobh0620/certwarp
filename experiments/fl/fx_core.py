"""FX (runs/20261105-fx/任务书.md, plan.md): the arms, the cells and the guards of three batches on the line of Hx24@600.  Not a method: no F_B, no certified descent, only the final field passes the certificate; fields in no result table.  Each batch below stands where run_fl.py and decide_fl.py expect the module fl_core (a test reads the names they ask from their source); fd2_core, fd_core, fl_core are not edited.

RW  root rw_v1.  Repairs of fd2_v1's Hx24 fields with wall_seconds 1200.  Cell('RHx24', 'c600' or 'rest600', pair), the tier of the field's pair in fd2_v1.  fd2_v1 is read only, the fields pinned by rw_sources_v1.json.
OX  root ox_v1, isolated.  Cell('O_Hx24', 'first5', pair), first 5 pairs only, truth one-hot through fl_oracle.prepare_hx, `oracle: true`; never repaired.
NX  root nx_v1.  Cell('NHx24', 'first5' or 'rest600', pair), Cell('RNHx24', 'c600' or 'rest600', pair).  Channels from the registered nn935_v1 predictions; repairs with RW's configuration.

No function of this module reads a label except read_truth (fl_core's), for the evaluation.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import fd2_core as fd2
import fl_core as fl
import mc0_core as core

RESULTS = fl.RESULTS
NATIVE = 600
SOURCE_ARM = 'Hx24'                             # fd2_v1: Hx24@600, read only
D_FACTOR = 24.
FD2_ROOT_NAME = 'fd2_v1'
ALL_PAIRS = core.PAIRS+fl.REST_PAIRS
FUSE_SECONDS = 1200.                            # the brief: wall_seconds of RW and NX's repairs, a fuse and not a budget
OLD_WALL_SECONDS = 120.                         # RF05's, what fd2_v1 ran with
REPAIR_RUNTIME_SECONDS = 1800                   # the brief: RuntimeMaxSec of a repair unit with the fuse
CERTIFIED = 'certified'
NOT_CERTIFIED = 'not_certified_fallback_rule_not_run'     # run_fl's name failed_back_to_frozen_rule reads as if the .9^k rule had run; it never does here (FD2 review, minor 16)
SPEC = dict(k=35, sigma_field=.5, native=NATIVE, checkpoints=(NATIVE,), channels='onehot', weight='Q3h', D_factor=D_FACTOR)


def pair_of(n):
    return f'OASIS_OAS1_{n:04d}_MR1__OASIS_OAS1_{n+1:04d}_MR1'


@dataclass(frozen=True)
class _Cell(core.Cell):
    PREFIX = None
    ARMS = {}
    ORACLE_ARMS = ()

    @property
    def run_id(self):
        return f'{self.PREFIX}_{self.arm}_{self.tier}_{self.short}'

    @property
    def unit(self):
        return f'tag-{self.PREFIX}-{self.arm.lower()}-{self.tier}-{self.short}'

    @property
    def is_repair(self):
        return self.arm[:1] == 'R' and self.arm[1:] in self.ARMS

    @property
    def is_oracle(self):
        return self.arm in self.ORACLE_ARMS

    @property
    def base(self):
        """The arm whose field a repair starts from; the arm itself otherwise."""
        return self.arm[1:] if self.is_repair else self.arm


class Batch:
    """What run_fl.py and decide_fl.py ask of fl_core, for one root.  The names about the root and the pairs are fl_core's own functions."""
    PREFIX, ROOT_NAME, ARMS, ROLE, ORACLE_ROLE, Cell = None, None, {}, None, 'no oracle arm in this batch', None
    FIRST, NATIVE_LEVEL, COARSE_ITERATIONS, RUN_CAP_SECONDS, TASK_BUDGET_SECONDS, REPAIR_EPS, REST_PAIRS, SELECTIONS_LOG = fl.FIRST, fl.NATIVE_LEVEL, fl.COARSE_ITERATIONS, fl.RUN_CAP_SECONDS, fl.TASK_BUDGET_SECONDS, fl.REPAIR_EPS, fl.REST_PAIRS, fl.SELECTIONS_LOG
    EF_ROOT, MC0_ROOT = fl.EF_ROOT, fl.MC0_ROOT
    REPAIR_STEPS, REPAIR_TOP, FULL, TWO = (NATIVE,), 1, (NATIVE,), (NATIVE,)
    ORACLE_ARMS, SIGMA_ARMS, EF_REPAIRS, FIRST_PHASE = (), (), {}, ()
    SelectionChanged = fl.SelectionChanged
    __file__ = __file__
    selection_hashes, write_selection, read_selection = staticmethod(fl.selection_hashes), staticmethod(fl.write_selection), staticmethod(fl.read_selection)
    read_truth, require_truth_pair, ef_dir, field_kernel = staticmethod(fl.read_truth), staticmethod(fl.require_truth_pair), staticmethod(fl.ef_dir), staticmethod(fl.field_kernel)

    def __init__(self):
        self.RESULTS = RESULTS
        self.DEFAULT_ROOT = RESULTS/self.ROOT_NAME

    @property
    def BATCH_SCHEMA(self):
        return f'bayesreg.{self.PREFIX}.batch.v1'

    def steps_of(self, cell):
        if cell.tier == self.FIRST:
            return self.ARMS[cell.arm]['native']
        return fl._steps_in(cell.tier, 'rest') or fl._steps_in(cell.tier, 'c')

    def checkpoints_of(self, cell):
        return self.ARMS[cell.arm]['checkpoints'] if cell.tier == self.FIRST else (self.steps_of(cell),)

    def run_dir(self, root, cell):
        return Path(root)/'runs'/f'{cell.arm}_{cell.tier}'/cell.short

    def root_problem(self, root):
        """Why this entry may not write at this root, or None: only its own."""
        return None if Path(root).resolve() == Path(self.DEFAULT_ROOT).resolve() else f'{root}: {self.PREFIX.upper()} writes only to {self.DEFAULT_ROOT}; nothing read, nothing moved'

    def out_problem(self, out, cell):
        """Why a run may not write at this directory, or None: exactly its own, so never under fd2_v1 or any other root."""
        want = self.run_dir(self.DEFAULT_ROOT, cell)
        return None if Path(out).resolve() == want.resolve() else f'{out}: {cell.run_id} writes only to {want}'

    def plan_rest(self, arm, steps):
        raise SystemExit(f'{self.PREFIX.upper()} has no separate queue of the other 14 pairs (run_fl.rest_body is not used here)')

    def plan_x(self):
        return []

    def plan_ef_repairs(self):
        return []

    def plan_fields(self):
        return []

    def x_sigma(self, root):
        return None

    def chosen_repairs(self, root):
        return []

    def repair_cells(self, chosen):
        return [self.Cell('R'+c['arm'], f"c{c['step']}", pair) for c in chosen for pair in core.PAIRS]

    def rest_cells(self, root):
        return [], []

    def sigma_of(self, cell, root):
        return self.ARMS[cell.arm]['sigma_field']

    def arm_config(self, cell, sigma):
        """The PRE of configuration C, as FD2's arms."""
        return dict(core.PRE_C, seconds=self.RUN_CAP_SECONDS, iterations=self.COARSE_ITERATIONS+(self.steps_of(cell),), sigma_field=sigma)

    def read_images(self, cell):
        """The PairInput of a cell of the plan: load_pair's firewall view, `fixed` and `moving` (subject guard, sha256 against the manifest)."""
        self.require_cell(cell.arm, cell.tier, cell.pair)
        from bayesreg.io.pair import load_pair
        return load_pair(core.STAGE/cell.pair/'pair.npz', allowlist=core.ALLOWLIST)

    def refuse(self, arm, tier, pair):
        raise SystemExit(f'{self.PREFIX.upper()} has no cell arm {arm} tier {tier} pair {pair}: {self.CELLS_TEXT}')


# ---- RW ---------------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RWCell(_Cell):
    PREFIX = 'rw'
    ARMS = {SOURCE_ARM: SPEC}


_CONTROL_PAIR = pair_of(439)                    # one round under the old configuration
_WALL_PAIRS = tuple(pair_of(n) for n in (438, 440, 441, 451, 456))
_REUSED_PAIRS = tuple(p for p in ALL_PAIRS if p not in _WALL_PAIRS and p != _CONTROL_PAIR)


class RWBatch(Batch):
    PREFIX, ROOT_NAME, ARMS, Cell = 'rw', 'rw_v1', {SOURCE_ARM: SPEC}, RWCell
    ROLE = 'RW: local repair of fd2_v1 Hx24@600 fields with wall_seconds 1200, predicted channels: a probe, NOT oracle, NOT a method, NOT in any result table; 120 s result stays in fd2_v1, the wall was removed after seeing that failure'
    REPAIR_ARM = 'R'+SOURCE_ARM
    CONTROL_PAIR, WALL_PAIRS, REUSED_PAIRS = _CONTROL_PAIR, _WALL_PAIRS, _REUSED_PAIRS
    CONTROL_FILE = 'rw_control.json'
    SOURCES_FILE = Path(__file__).resolve().parent/'rw_sources_v1.json'
    CELLS_TEXT = f'only {REPAIR_ARM} on the 19 development pairs, tier c600 on the first 5, rest600 on the other 14'

    def tier_of(self, pair):
        return f'c{NATIVE}' if pair in core.PAIRS else f'rest{NATIVE}'

    def fd2_tier_of(self, pair):
        return fd2.FIRST if pair in core.PAIRS else f'rest{NATIVE}'

    def require_cell(self, arm, tier, pair):
        """Checked before any file of the pair is opened."""
        if pair not in ALL_PAIRS or arm != self.REPAIR_ARM or tier != self.tier_of(pair):
            self.refuse(arm, tier, pair)
        return self.Cell(arm, tier, pair)

    def cell_of(self, pair):
        return self.Cell(self.REPAIR_ARM, self.tier_of(pair), pair)

    def plan_control(self):
        return [self.cell_of(self.CONTROL_PAIR)]

    def plan_wall(self):
        return [self.cell_of(p) for p in self.WALL_PAIRS]

    def plan_reused(self):
        return [self.cell_of(p) for p in self.REUSED_PAIRS]

    def control_choice(self, root):
        return self.read_selection(root, self.CONTROL_FILE)

    def all_cells(self, root):
        """What this root's plan holds now: the control, the 5 pairs that hit the wall, and the other 13 only when the control was not bit for bit."""
        choice = self.control_choice(root)
        return self.plan_control()+self.plan_wall()+(self.plan_reused() if choice is not None and choice['same'] is False else [])

    def fd2_root(self, results=None):
        return Path(self.RESULTS if results is None else results)/FD2_ROOT_NAME

    def fd2_field_cell(self, pair):
        return fd2.Cell(SOURCE_ARM, self.fd2_tier_of(pair), pair)

    def fd2_repair_cell(self, pair):
        return fd2.Cell(self.REPAIR_ARM, self.tier_of(pair), pair)

    def sources(self, path=None):
        """{short name of a pair: the pins of its field and of its 120 s repair}."""
        return json.loads(Path(self.SOURCES_FILE if path is None else path).read_text())['pairs']


# ---- OX ---------------------------------------------------------------------------------------------------------------------------------------
OX_ARM = 'O_Hx24'
OX_SPEC = dict(SPEC, channels='truth_onehot')


@dataclass(frozen=True)
class OXCell(_Cell):
    PREFIX = 'ox'
    ARMS = {OX_ARM: OX_SPEC}
    ORACLE_ARMS = (OX_ARM,)


class OXBatch(Batch):
    PREFIX, ROOT_NAME, ARMS, Cell, ORACLE_ARMS = 'ox', 'ox_v1', {OX_ARM: OX_SPEC}, OXCell, (OX_ARM,)
    ROLE = ORACLE_ROLE = 'OX: oracle ceiling of Hx24@600: truth one-hot channels, Q3h on truth volumes, D factor 24; isolated root, NOT a method, NOT in any result table, NOT an entry point'
    CELLS_TEXT = f'only {OX_ARM}, tier first5, on the first 5 development pairs'

    def require_cell(self, arm, tier, pair):
        if arm != OX_ARM or tier != self.FIRST or pair not in core.PAIRS:
            self.refuse(arm, tier, pair)
        return self.Cell(arm, tier, pair)

    def plan_fields(self):
        return [self.Cell(OX_ARM, self.FIRST, pair) for pair in core.PAIRS]

    def all_cells(self, root):
        return self.plan_fields()

    def read_images(self, cell):
        raise SystemExit(f'{cell.run_id}: the oracle arm reads through fl_oracle.prepare_hx only')


# ---- NX ---------------------------------------------------------------------------------------------------------------------------------------
NX_ARM = 'NHx24'


@dataclass(frozen=True)
class NXCell(_Cell):
    PREFIX = 'nx'
    ARMS = {NX_ARM: SPEC}


class NXBatch(Batch):
    PREFIX, ROOT_NAME, ARMS, Cell, FIRST_PHASE = 'nx', 'nx_v1', {NX_ARM: SPEC}, NXCell, (NX_ARM,)
    ROLE = 'NX: Hx24@600 on the channels of nnU-Net (nn935_v1), repair with the fuse of RW: a probe, NOT oracle, NOT a method, NOT in any result table'
    REPAIR_ARM = 'R'+NX_ARM
    SEGMENTER_ID, MODEL_ID, FOLD = 'nn935_v1', 'nn935_v1_foldall', 0     # the schema's fold is an integer; the model is nnU-Net's fold all, its id says so
    REGISTRY_REL = 'configs/bayesreg/segmenters_nn935_v1.json'
    REST_FILE = 'rest14_selection.json'
    CELLS_TEXT = f'{NX_ARM} tier first5 and {REPAIR_ARM} tier c600 on the first 5 development pairs; both tier rest600 on the other 14'

    def __init__(self):
        super().__init__()
        self.SEG_DIR = RESULTS/'seg'/self.SEGMENTER_ID
        self.PRED_ROOT = self.SEG_DIR/'pred'

    def require_cell(self, arm, tier, pair):
        first, rest = pair in core.PAIRS, pair in fl.REST_PAIRS
        if arm == NX_ARM:
            ok = (tier == self.FIRST and first) or (tier == f'rest{NATIVE}' and rest)
        elif arm == self.REPAIR_ARM:
            ok = (tier == f'c{NATIVE}' and first) or (tier == f'rest{NATIVE}' and rest)
        else:
            ok = False
        if not ok:
            self.refuse(arm, tier, pair)
        return self.Cell(arm, tier, pair)

    def plan_fields(self):
        return [self.Cell(NX_ARM, self.FIRST, pair) for pair in core.PAIRS]

    def plan_repairs(self):
        return [self.Cell(self.REPAIR_ARM, f'c{NATIVE}', pair) for pair in core.PAIRS]

    def plan_rest_fields(self):
        return [self.Cell(NX_ARM, f'rest{NATIVE}', pair) for pair in fl.REST_PAIRS]

    def plan_rest_repairs(self):
        return [self.Cell(self.REPAIR_ARM, f'rest{NATIVE}', pair) for pair in fl.REST_PAIRS]

    def chosen_repairs(self, root):
        """decide_fl.collect asks: the repairs of the first 5 pairs are always in the plan here."""
        return [dict(arm=NX_ARM, step=NATIVE)]

    def rest_choice(self, root):
        return self.read_selection(root, self.REST_FILE)

    def rest_cells(self, root):
        choice = self.rest_choice(root)
        return (self.plan_rest_fields(), self.plan_rest_repairs()) if choice is not None and choice['go'] is True else ([], [])

    def all_cells(self, root):
        fields, repairs = self.rest_cells(root)
        return self.plan_fields()+self.plan_repairs()+fields+repairs

    def planned_pairs(self, root):
        return ALL_PAIRS if self.rest_cells(root)[0] else core.PAIRS

    def registry_path(self, repo=None):
        return Path(core.ROOT if repo is None else repo)/self.REGISTRY_REL


RW, OX, NX = RWBatch(), OXBatch(), NXBatch()
BATCHES = dict(rw=RW, ox=OX, nx=NX)
