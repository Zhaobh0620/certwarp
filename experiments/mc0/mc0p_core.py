"""MC0P (runs/20261104-mc0p/任务书.md, plan.md): the arms and the channels of the multi-channel probe on PREDICTED labels.  The optimizer, the scale rule's norms, the tiers and
the first 5 development pairs are MC0's (mc0_core.py), unchanged.  Not a method and not an oracle: `oracle: false`, an uncertified field, in no result table.

The firewall: a channel comes only from the two images of bayesreg.io.pair.load_pair (the loader that never decodes another member of pair.npz) and from the predictions of the
registered segmenter unet35_v1_ens, policy all_folds_mean, read after every check A5P's images went through (make_inputs.py: D24 admission, StructureLayer, read-back, group sums,
alignment QC).  No function of this module takes a pair.npz path or a label member; the truth labels reach only the evaluation (run_mc0p.evaluate_all).

Q0..Q3 are the formulas of P0..P3 with the truth seg35 replaced by L = argmax_l P_l (ties: the smallest l, as A5P): core.channels takes a label map, never a pair, so the same code
runs.  New here: Q2s (G_.75 of the posterior channels 1..35, no argmax), Q3h (w_l proportional to V_l^(-1/2)), and the D factor of Q3x3 (s = 3 ||g_Q0|| / ||g_arm||).
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

import mc0_core as core

SEG_ROOT = Path('/home/rmcfz/seq_vp_tag_results/seg/atropos_v1')     # the Atropos posteriors of the alignment QC: A5P's d0_inputs.json atropos_root
A5P_INPUTS = Path('/home/rmcfz/seq_vp_tag_results/d0_v1_a5/inputs/A5P/stage_L2R_val_v1')     # read only: Q0's two images must be these, bit for bit
MC0_ROOT = Path('/home/rmcfz/seq_vp_tag_results/mc0_v1')             # read only: the Delta_P of the realization ratios
ROLE = 'uncertified_probe_with_predicted_channels_NOT_oracle_NOT_a_method_NOT_in_any_result_table'
VRAM_LIMIT_BYTES = 9*2**30                                           # the brief: per-process cap, MC0's P2 reserved 6.98 GiB
BATCH_SCHEMA = 'bayesreg.mc0p.batch.v1'
IMAGE_KEYS = ('fixed', 'moving', 'initial_u')                        # the only members of pair.npz a channel may come from: load_pair's whitelist (initial_u is all zero)
SEGMENTER_ID, POLICY = 'unet35_v1_ens', 'all_folds_mean'             # the predictions of A5P's images
BASELINE = 'Q0'
TIER_NAMES = ('long', 'fine')
LONG, BOTH = ('long',), ('long', 'fine')
# base: the MC0 arm whose channel formula it is (None: Q2s).  weight: the rule of w_l.  D_factor: on top of the scale rule.  decides: enters the reading.
ARMS = dict(Q0=dict(k=1, tiers=BOTH, base='P0', weight=None, D_factor=1., decides=False),
            Q1=dict(k=14, tiers=LONG, base='P1', weight=None, D_factor=1., decides=True),
            Q2=dict(k=35, tiers=BOTH, base='P2', weight=None, D_factor=1., decides=True),
            Q2s=dict(k=35, tiers=LONG, base=None, weight=None, D_factor=1., decides=True),
            Q3=dict(k=35, tiers=LONG, base='P3', weight='inverse', D_factor=1., decides=False),
            Q3x3=dict(k=35, tiers=LONG, base='P3', weight='inverse', D_factor=3., decides=False),
            Q3h=dict(k=35, tiers=LONG, base='P3', weight='inverse_sqrt', D_factor=1., decides=False))
FIRST_PHASE = ('Q0', 'Q1', 'Q2', 'Q2s')                              # the baseline and the arms that enter the reading
SECOND_PHASE = ('Q3', 'Q3x3', 'Q3h')                                 # description only


@dataclass(frozen=True)
class Cell(core.Cell):
    @property
    def run_id(self):
        return f'mc0p_{self.arm}_{self.tier}_{self.short}'

    @property
    def unit(self):
        return f'tag-mc0p-{self.arm.lower()}-{self.tier}-{self.short}'


def require_cell(arm, tier, pair):
    core.require_pair(pair)
    if arm not in ARMS or tier not in ARMS[arm]['tiers']:
        raise SystemExit(f'MC0P has no cell arm {arm} tier {tier}')
    return Cell(arm, tier, pair)


def plan_runs():
    """Every run, in order: first the baseline and the arms that enter the reading, then the arms only described, so that a cap can only cost description.  In each phase pair-major,
    the arm order rotating with the pair index as in MC0, the tiers in the brief's order."""
    runs = []
    for arms in (FIRST_PHASE, SECOND_PHASE):
        for i, pair in enumerate(core.PAIRS):
            for j in range(len(arms)):
                arm = arms[(i+j) % len(arms)]
                runs += [Cell(arm, tier, pair) for tier in ARMS[arm]['tiers']]
    return runs


# ---- the one retry the lead approved (plan revision 2) ----------------------------------------------------------------------------------
RETRY_OF_ROOT = Path('/home/rmcfz/seq_vp_tag_results/mc0p_v1')       # the first batch: read only, not a byte of it moves
RETRY_ROOT = Path('/home/rmcfz/seq_vp_tag_results/mc0p_v1b')
EXPECTED_NOT_DONE = 20                                               # the lead's count from queue_end.json; the plan itself comes from the records


@dataclass(frozen=True)
class RetryCell(Cell):
    """The same cell, run_id and run directory layout; only the unit is this batch's."""
    @property
    def unit(self):
        return f'tag-mc0pb-{self.arm.lower()}-{self.tier}-{self.short}'


CONTROL = RetryCell('Q2', 'long', core.PAIRS[0])                     # done in v1; run again to compare the field's u


def under(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    return path == parent or parent in path.parents


def v1_write_problem(path):
    """Why nothing may be written at this path, or None."""
    return f'{path}: under the first batch {RETRY_OF_ROOT}, which is read only' if under(path, RETRY_OF_ROOT) else None


def retry_problem(root, retry_of):
    """Why this is not the approved retry, or None: only this first batch and only this new root."""
    if Path(retry_of).resolve() != RETRY_OF_ROOT.resolve():
        return f'{retry_of}: the retry is of {RETRY_OF_ROOT} only'
    if Path(root).resolve() != RETRY_ROOT.resolve():
        return f'{root}: the retry goes to {RETRY_ROOT} only'
    return None


def v1_status(v1_root, cell):
    """The status in the first batch's run.json; None when it has none (a run skipped after its arm stopped, for instance)."""
    path = core.run_dir(v1_root, cell)/'run.json'
    return json.loads(path.read_text()).get('status') if path.is_file() else None


def retry_plan(v1_root=None):
    """The control first, then every cell whose status in the first batch is not done: the arms that enter the reading, then the arms only described, so that a cap or a hold costs description first.
    Arm by arm, then tier, then pair.  From the records, nothing typed in."""
    v1_root = RETRY_OF_ROOT if v1_root is None else Path(v1_root)
    if v1_status(v1_root, CONTROL) != 'done':
        raise SystemExit(f'{CONTROL.run_id} is not done in {v1_root}: there is nothing to compare the control with')
    not_done = [c for c in plan_runs() if v1_status(v1_root, c) != 'done']
    arms = list(ARMS)
    not_done.sort(key=lambda c: (c.arm in SECOND_PHASE, arms.index(c.arm), TIER_NAMES.index(c.tier), core.PAIRS.index(c.pair)))
    return [CONTROL]+[RetryCell(c.arm, c.tier, c.pair) for c in not_done]


def deciding_arms(tier):
    """The K > 1 arms of a tier that enter the reading."""
    return [arm for arm, spec in ARMS.items() if spec['decides'] and tier in spec['tiers']]


# ---- what a channel may be made of ---------------------------------------------------------------------------------------------------
def read_images(pair, *, stage=None, allowlist=None):
    """The PairInput of a planned pair: load_pair's firewall view, `fixed` and `moving` (subject guard, sha256 against the manifest).  Only the first 5 development pairs."""
    core.require_pair(pair)
    from bayesreg.io.pair import load_pair
    return load_pair((core.STAGE if stage is None else stage)/pair/'pair.npz', allowlist=core.ALLOWLIST if allowlist is None else allowlist)


@dataclass
class Predicted:
    """What the channels of one pair may use besides its two images: labels {side: (nz, ny, nx) int16}, soft {side: (35, nz, ny, nx) float64 G_.75 of P_1..P_35} or None, record."""
    labels: dict
    soft: dict
    record: dict


def predicted_labels(P):
    """L = argmax_l P_l, ties to the smallest l: the label map of A5P (make_inputs.colour)."""
    return np.argmax(P, axis=0).astype(np.int16)


def soft_channels(P, workers=1):
    """Q2s: G_.75 of the channels 1..35 of the (36, nz, ny, nx) posterior, no argmax, background left out.  Written in place over P: the host never holds a second copy of the stack."""
    mi = core.make_inputs()
    out = P[1:]

    def fill(c):
        out[c] = mi.smooth(out[c])
    core._pool_map(fill, range(out.shape[0]), workers)
    return out


def default_unet():
    """The source of A5P's images: unet35_v1_ens, /home/rmcfz/seq_vp_tag_results/seg/unet35_v1/pred, configs/bayesreg/segmenters.json, the pinned folds file."""
    return core.make_inputs().UNetSource()


def read_predicted(pin, *, soft=False, unet=None, seg_root=None, allowlist=None, workers=1):
    """The predictions of the pair after every check of A5P (the functions of make_inputs.py, read only; §12.4), STOP otherwise.  It takes the PairInput, which holds no label."""
    mi = core.make_inputs()
    unet = default_unet() if unet is None else unet
    seg_root = SEG_ROOT if seg_root is None else seg_root
    allowlist = core.ALLOWLIST if allowlist is None else allowlist
    admission = mi.a4_admission([pin.fixed_subject, pin.moving_subject], allowlist, unet)
    layer = mi.pair_structure_layer(pin, allowlist, unet)
    labels, stacks, sides = {}, {}, {}
    for side in ('fixed', 'moving'):
        P, record, u_wm = mi.unet_posterior(layer, side)
        support = mi.support_dice_record(getattr(pin, side), P[0])
        a_wm, atropos = mi.atropos_wm(seg_root, pin, side)
        qc = mi.alignment_qc(u_wm, a_wm)
        del u_wm, a_wm
        if not qc['passed']:
            raise SystemExit(f"MC0P {pin.fixed_subject} <- {pin.moving_subject}:{side}: alignment QC {qc['dice']} < {qc['threshold']} (protocol §12.4 item 5)")
        labels[side] = predicted_labels(P)
        if soft:
            stacks[side] = soft_channels(P, workers)
        del P
        sides[side] = dict(record, subject=getattr(pin, side+'_subject'), alignment_qc=qc, support_dice_record=support, atropos_qc=atropos)
    keys_read = list(pin.provenance['keys_read'])
    return Predicted(labels, stacks if soft else None, dict(
        **oracle_mark(keys_read, sides), pair_keys_read=keys_read, label_firewall=pin.provenance['label_firewall'], segmenter=unet.record(), atropos_root=str(seg_root),
        subject_admission=admission, structure_layer=layer.record(), sides=sides))


def oracle_mark(keys_read, sides):
    """The marks of a run, computed from what its channels were read from, never written by hand (plan revision 1).  oracle is False only when pair.npz gave both images and no member
    outside IMAGE_KEYS, and both images' predictions are unet35_v1_ens, all_folds_mean.  truth_labels_read: some other member of pair.npz was read."""
    other = sorted(set(keys_read)-set(IMAGE_KEYS))
    images = {'fixed', 'moving'} <= set(keys_read)
    predictions = set(sides) == {'fixed', 'moving'} and all(s.get('segmenter_id') == SEGMENTER_ID and s.get('policy') == POLICY for s in sides.values())
    return dict(oracle=not (images and predictions and not other), truth_labels_read=bool(other))


def oracle_of_run(record):
    """False, the value metrics.json and long.csv take from run.json, after computing it again from the run's channel_sources.  ValueError otherwise: the field is not read."""
    sources = record.get('channel_sources') or {}
    mark = oracle_mark(sources.get('pair_keys_read') or [], sources.get('sides') or {})
    if record.get('oracle') is not False or mark['oracle'] is not False:
        raise ValueError(f"run.json oracle {record.get('oracle')}; computed from its channel_sources: {mark}")
    return False


def channel_weights(arm, fixed_labels):
    """(w_l, V_l), V_l the voxels of label l in the FIXED image's PREDICTED label map.  Q3: 1/V_l, the rule of P3 (core.channel_weights).  Q3h: V_l^(-1/2), the same rule otherwise: a label
    absent there gets 0, normalized to mean 1 over the present labels, for reading only.  Other arms: None, None."""
    rule = ARMS[arm]['weight']
    if rule is None:
        return None, None
    if rule == 'inverse':
        return core.channel_weights('P3', fixed_labels)
    voxels = np.array([np.count_nonzero(fixed_labels == label) for label in core.LABELS_35], dtype=np.float64)
    present = voxels > 0
    if not present.any():
        raise SystemExit(f'{arm}: no label in the fixed image')
    w = np.zeros(len(core.LABELS_35))
    w[present] = 1/np.sqrt(voxels[present])
    return w/w[present].mean(), voxels


def q0_channels(pin, predicted, table):
    """(F, M), Q0's two (1, nz, ny, nx) images: the A5 colouring of the predicted label maps, make_inputs.colour_from_table as for A5P."""
    return tuple(core.channels('P0', getattr(pin, side), predicted.labels[side], table=table) for side in ('fixed', 'moving'))


def arm_channels(arm, pin, predicted, *, groups=None, workers=1):
    """The two (K, nz, ny, nx) float64 stacks of a K > 1 arm, before the weights and the scale.  Q2s hands over the soft stacks of `predicted`."""
    if arm == 'Q2s':
        if predicted.soft is None:
            raise SystemExit('Q2s: the predictions were read without the soft channels')
        return predicted.soft['fixed'], predicted.soft['moving']
    base = ARMS[arm]['base']
    return tuple(core.channels(base, getattr(pin, side), predicted.labels[side], groups=groups, workers=workers) for side in ('fixed', 'moving'))


def prepare_stacks(arm, pin, predicted, q0, device, *, groups=None, workers=1):
    """([F, M], record): the arm's two stacks, weighted and scaled by MC0's rule against Q0 (||g_D|| at u = 0, native level), then the arm's D_factor.  Q0 is never multiplied."""
    reference = core.data_grad_norm(q0[0], q0[1], device)
    record = dict(D_factor=ARMS[arm]['D_factor'], g_norm_Q0=reference, channel_weights=None, fixed_voxels=None)
    if arm == BASELINE:
        return list(q0), dict(record, D_multiplier=1., channel_multiplier=1., g_norm_unscaled=reference, g_norm_scaled=reference)
    F, M = arm_channels(arm, pin, predicted, groups=groups, workers=workers)
    w, voxels = channel_weights(arm, predicted.labels['fixed'])
    if w is not None:
        root = np.sqrt(w)[:, None, None, None]
        F *= root
        M *= root
    unscaled = core.data_grad_norm(F, M, device)
    s = core.scale_factor(reference, unscaled)*record['D_factor']
    F *= np.sqrt(s)
    M *= np.sqrt(s)
    scaled = core.data_grad_norm(F, M, device)
    if abs(scaled/(reference*record['D_factor'])-1) > core.SCALE_TOL:
        raise SystemExit(f"scale rule: ||g_D|| after scaling {scaled} differs from {record['D_factor']} x Q0 {reference}")
    return [F, M], dict(record, D_multiplier=s, channel_multiplier=float(np.sqrt(s)), g_norm_unscaled=unscaled, g_norm_scaled=scaled,
                        channel_weights=None if w is None else w.tolist(), fixed_voxels=None if voxels is None else voxels.tolist())


def level_norm_record(arm, reference=None):
    """core.level_norm_record with the baseline's name: r_l against Q0."""
    record = core.level_norm_record(arm, reference)
    record[BASELINE] = record.pop('P0')
    return record
