"""Arm O of FL (runs/20261105-fl/plan.md): the oracle ceiling, MC0's P2 and its firewall as they are.  The ONLY channel path of FL that reads truth labels; only the first 5 development pairs (run_mc0.read_pair), `oracle: true`, written only under <root>/oracle_isolated (fl_core.run_dir, out_problem), never repaired, in no table of the reading.

The channels and the scale are core.prepare_stacks('P2', ...): G_.75 of the truth one-hot, ||g_D|| at u = 0 equal to the truth colouring's P0.  At 600 steps the field should be the one of mc0_v1/runs/P2_fine (a record, decide_fl.controls).
"""
import hashlib
from types import SimpleNamespace

import numpy as np

import mc0_core as core
import run_mc0 as run

TRUTH_CHANNEL_MEMBERS = ('fixed_seg35', 'moving_seg35')


def prepare(pair, device, workers=1):
    """(pin, [F, M], scale record, channel sources, labels' sha256): oracle true, computed from what was read."""
    pin, labs = run.read_pair(pair, TRUTH_CHANNEL_MEMBERS)
    digest = {side: hashlib.sha256(np.ascontiguousarray(labs[f'{side}_seg35']).tobytes()).hexdigest() for side in ('fixed', 'moving')}
    stacks, scale = core.prepare_stacks('P2', pin.fixed, pin.moving, labs['fixed_seg35'], labs['moving_seg35'], device, workers=workers)
    sources = dict(oracle=True, truth_labels_read=True, pair_keys_read=list(pin.provenance['keys_read'])+list(TRUTH_CHANNEL_MEMBERS), label_firewall='this arm is the oracle: run_mc0.read_pair reads the named truth members', segmenter=None)
    return pin, stacks, dict(scale, D_factor=1.), sources, digest


def prepare_hx(pair, device, spec, table, workers=1):
    """FX's arm O_Hx24 (runs/20261105-fx/任务书.md): (pin, [F, M], scale record, channel sources, labels' sha256), oracle true.  The numbers of fl_channels.prepare_stacks with the truth maps in the place of the predicted ones, the one function the arm Hx24 runs: G_.75 one-hot, Q3h with V_l of the truth fixed map, Q0 the A5 colouring of the truth maps, s = D_factor x ||g_Q0|| / ||g_arm||.  Chosen over core.prepare_stacks('P2', ...) with weights added afterwards: only the labels differ from Hx24 then.  Only the first 5 development pairs (run_mc0.read_pair)."""
    import fl_channels as channels
    pin, labs = run.read_pair(pair, TRUTH_CHANNEL_MEMBERS)
    digest = {side: hashlib.sha256(np.ascontiguousarray(labs[f'{side}_seg35']).tobytes()).hexdigest() for side in ('fixed', 'moving')}
    truth = SimpleNamespace(labels=dict(fixed=labs['fixed_seg35'], moving=labs['moving_seg35']))
    stacks, scale = channels.prepare_stacks(spec, pin, truth, table, device, workers=workers)
    sources = dict(oracle=True, truth_labels_read=True, pair_keys_read=list(pin.provenance['keys_read'])+list(TRUTH_CHANNEL_MEMBERS), label_firewall='this arm is the oracle: run_mc0.read_pair reads the named truth members', segmenter=None)
    return pin, stacks, scale, sources, digest
