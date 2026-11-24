"""EF (docs/bayesreg/notes/提速与换预分割器规划_20261104.md §6; runs/20261104-ef/plan.md): the arms, the cells and the guards of the entry probe.  Not a method: predicted channels (unet35_v1_ens, `oracle: false`), fields in no result table, nothing passes the native level.

E1  K = 35 (argmax, G_.75) with fewer native steps, or K = 8 vector colouring.  Q2 is the baseline: MC0P's Q2 long, by MC0P's own run_one and old channel path, the only arm that uses it; its seconds are the denominator of the reading.
E2  the other arms use the faster channels (fast_channels.py), bit for bit; only the seconds differ.  F600 is Q2 fine again and its field is compared with mc0p's; so is Q2's.
E3  local repair (repair.py) of fields that exist: Q2 fine on the first 5 pairs; Q2 long of the two pairs that miss alpha = 1 by 2 cells.

Pairs: the first 5 development pairs, and the one exception the lead named (2026-11-04, by way of SEED19_结果_复核_20261104.md, important 5): 0455-0456, only in the arms RL01, RL05 (E3, long tier).  require_cell is the only way in.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import mc0_core as core
import mc0p_core as mc

RESULTS = Path('/home/rmcfz/seq_vp_tag_results')
DEFAULT_ROOT = RESULTS/'ef_v1'
ROLE = 'entry probe with predicted channels, NOT oracle, NOT a method, NOT in any result table'
BATCH_SCHEMA = 'bayesreg.ef.batch.v1'
BASELINE = 'Q2'
EXCEPTION_PAIR = 'OASIS_OAS1_0455_MR1__OASIS_OAS1_0456_MR1'     # the lead, 2026-11-04: only in E3, long tier
LONG_REPAIR_PAIRS = (core.PAIRS[1], EXCEPTION_PAIR)              # 0439-0440, 0455-0456: alpha = 1 misses by 2 cells
# E1.  channels: 'old' the path of mc0p_core (only Q2), 'onehot' its K = 35 numbers by fast_channels, 'colour8' vector_colour.  decides: enters the reading of E1.
ARMS = dict(Q2=dict(k=35, native=600, sigma_field=1., tier='long', channels='old', decides=False),
            L150=dict(k=35, native=150, sigma_field=1., tier='long', channels='onehot', decides=True),
            L300=dict(k=35, native=300, sigma_field=1., tier='long', channels='onehot', decides=True),
            F150=dict(k=35, native=150, sigma_field=.5, tier='fine', channels='onehot', decides=True),
            F300=dict(k=35, native=300, sigma_field=.5, tier='fine', channels='onehot', decides=True),
            F600=dict(k=35, native=600, sigma_field=.5, tier='fine', channels='onehot', decides=True),
            K8=dict(k=8, native=600, sigma_field=1., tier='long', channels='colour8', decides=True))
# E3: the source tier and eps.  The fine arms enter the reading of E3; the long arms are described only (the lead, 2026-11-04).
REPAIR_ARMS = dict(RF01=dict(tier='fine', eps=.01, decides=True), RF05=dict(tier='fine', eps=.05, decides=True), RL01=dict(tier='long', eps=.01, decides=False), RL05=dict(tier='long', eps=.05, decides=False))
# Where each source field is (run.json status done; read only).  A cell not here is not repaired.
SOURCES = {('fine', core.PAIRS[0]): 'mc0p_v1', ('fine', core.PAIRS[1]): 'mc0p_v1', ('fine', core.PAIRS[2]): 'mc0p_v1', ('fine', core.PAIRS[3]): 'mc0p_v1b', ('fine', core.PAIRS[4]): 'mc0p_v1b',
           ('long', core.PAIRS[1]): 'mc0p_v1', ('long', EXCEPTION_PAIR): 'mc0p_v1c'}
# The control of E2 and of the baseline: u of the field against the run that made it (cuda; mc0p's control was bitwise across its two runs).
CONTROLS = {('Q2', pair): ('mc0p_v1b' if pair in (core.PAIRS[0], core.PAIRS[3], core.PAIRS[4]) else 'mc0p_v1', 'long') for pair in core.PAIRS}
CONTROLS.update({('F600', pair): (SOURCES[('fine', pair)], 'fine') for pair in core.PAIRS})


@dataclass(frozen=True)
class Cell(core.Cell):
    @property
    def run_id(self):
        return f'ef_{self.arm}_{self.tier}_{self.short}'

    @property
    def unit(self):
        return f'tag-ef-{self.arm.lower()}-{self.tier}-{self.short}'

    @property
    def is_repair(self):
        return self.arm in REPAIR_ARMS


def _refuse(arm, tier, pair):
    raise SystemExit(f'EF runs the first 5 development pairs {[core.short_name(p) for p in core.PAIRS]}, and 0455-0456 only in the arms RL01, RL05 (E3, long tier); refused: {arm} {tier} {pair}')


def require_cell(arm, tier, pair):
    """Checked before any file of the pair is opened."""
    if arm in ARMS:
        if pair not in core.PAIRS:
            _refuse(arm, tier, pair)
        if tier != ARMS[arm]['tier']:
            raise SystemExit(f'EF has no cell arm {arm} tier {tier}')
    elif arm in REPAIR_ARMS:
        if tier != REPAIR_ARMS[arm]['tier']:
            raise SystemExit(f'EF has no cell arm {arm} tier {tier}')
        if (tier, pair) not in SOURCES:
            _refuse(arm, tier, pair)
    else:
        raise SystemExit(f'EF has no cell arm {arm} tier {tier}')
    return Cell(arm, tier, pair)


def require_truth_pair(pair):
    """The evaluation's guard: only a pair of the plan."""
    if pair not in core.PAIRS and pair != EXCEPTION_PAIR:
        _refuse('evaluation', '-', pair)
    return pair


def plan_runs():
    """Every run, in order.  E1: pair-major, the arm order rotating with the pair index (MC0P's rule, so that no arm is always first or always after the same one).  Q2 of the first pair first: it checks the images.  Then E3: the fine arms pair by pair, then the long arms."""
    arms = tuple(ARMS)
    runs = [Cell(arms[(i+j) % len(arms)], ARMS[arms[(i+j) % len(arms)]]['tier'], pair) for i, pair in enumerate(core.PAIRS) for j in range(len(arms))]
    for tier in ('fine', 'long'):
        pairs = core.PAIRS if tier == 'fine' else LONG_REPAIR_PAIRS
        runs += [Cell(arm, tier, pair) for pair in pairs for arm, spec in REPAIR_ARMS.items() if spec['tier'] == tier]
    return runs


def arm_config(arm):
    """The PRE of configuration C, as MC0's tiers (mc0_core.PRE_C, RUN_CAP_SECONDS), with the arm's native steps and sigma_field."""
    spec = ARMS[arm]
    return dict(core.PRE_C, seconds=core.RUN_CAP_SECONDS, iterations=(50, 50, 100, spec['native']), sigma_field=spec['sigma_field'])


def source_dir(cell, results=None):
    """The run directory of the field a repair cell starts from."""
    return Path(RESULTS if results is None else results)/SOURCES[(cell.tier, cell.pair)]/'runs'/f'Q2_{cell.tier}'/cell.short


def read_images(cell):
    """The PairInput of a cell of the plan: load_pair's firewall view, `fixed` and `moving` (subject guard, sha256 against the manifest).  For a repair only its grid is used."""
    require_cell(cell.arm, cell.tier, cell.pair)
    from bayesreg.io.pair import load_pair
    return load_pair(core.STAGE/cell.pair/'pair.npz', allowlist=core.ALLOWLIST)


def read_truth(pair, members):
    """The named truth label members.  Only the evaluation calls this; no channel path or repair path does (a test reads their source)."""
    require_truth_pair(pair)
    from bayesreg.io.pair import load_pair
    path = core.STAGE/pair/'pair.npz'
    pin = load_pair(path, allowlist=core.ALLOWLIST)
    with np.load(path) as z:
        return pin, {k: np.array(z[k]) for k in members}


def root_problem(root):
    """Why this entry may not write at this root, or None: only its own."""
    return None if Path(root).resolve() == Path(DEFAULT_ROOT).resolve() else f'{root}: EF writes only to {DEFAULT_ROOT}; nothing read, nothing moved'
