"""HE2 (runs/20261107-he2/任务书.md item 4; contract C22.B1, C22.B4 items 1-5 and 7, D83, D86): the tissue-dependent geometry prior C in the T88 child of the HE entry (method Hx24-RF05-F1200-v4-C3,
tissue_prior moment3).  The rules are experiments/p2/p2_core.py's own functions, imported, not copied: class_index, tissue_levels, TissueWeights, Hooks, optimize_p2 (which calls mc0_core.optimize_k itself when
there are no hooks: pass 0 is the frozen T88, C off), moment_ratios, sha_bytes, C_PASSES, C_CLASSES, EstimateFailed, ChannelSet.mark.

Two parts are copied, each with its source, because run_p2.py holds them inside its driver (importing run_p2 runs two instances of run_fl.py and the P2/P3 queue's machinery into the method's process):
  read_tissue    experiments/p2/run_p2.py:150-163 read_tissue at 090c985, every call in the same order; each step is wrapped so that a refusal says which step refused (D86 needs that:
                 simplex_with_background and brain_mask both raise InputValidationError).
  three_passes   the pass loop of run_p2.run_field for the arm C (em False, tissue True) at 090c985, run_p2.py:200-248: the same calls in the same order with the same arguments, without what is not C
                 (the EM), without the checkpoint keeper (HE writes the field run_hx returns, as for C off) and without the control of pass 0 against a reference field (C22.B1-1: the method reads no other
                 run; the smoke compares, C22.B4-6 (a)).
A test runs both against run_p2's own read_tissue and run_field on a synthetic pair, bit for bit (tests/test_bayesreg/he/test_he_c.py).

D86 (C22.B1-8, BZ 2026-11-07 "4默认"): the refusals of the tissue preparation are split by step:
  (1) the input or the configuration: the tissue layer's provenance or file check (load_tissue_layer: ProvenanceRejected), the class names or wm_index (class_index: ValueError)
        -> provenance_rejected: hx_provenance_rejected, no output, no fallback (D74 (4)); the queue goes on
  (2) the implementation: a coarse grid that is not four_level_grids' (tissue_levels: 'tissue pyramid: the coarse grid differs ...')
        -> implementation_error: hx_provenance_rejected with the reason implementation_error, to the lead; the queue stops (run_hx exit 9); nothing is rerun
  (3) the data: an empty brain mask (brain_mask: InputValidationError), a level without an Omega_B cell (tissue_levels: 'tissue level <k>: no cell inside the brain mask'); and estimate_failed (the passes)
        -> an entry failure, the fallbacks (D74 (2))
  (4) anything else (an OSError reading an Atropos file, memory, the simplex check, any other step's exception)
        -> tissue_error: an entry failure, the fallbacks, needs_lead_rejudgment
      The lead's ruling of 2026-11-07 (HE2 review, minors 1 and 2; docs/bayesreg/reviews/HE2_工具_复核_20261107.md): these stay in (4) as they are: the simplex check failing
      (simplex_with_background's InputValidationError), the tissue layer's files not parsed (provenance.json not JSON: json.loads' JSONDecodeError inside load_tissue_layer) and load_scalar's
      InputValidationError (a posterior NIfTI the reader refuses); only load_tissue_layer's own ProvenanceRejected is (1).  The four-level grids of the pair (step grids) are (4) too.
No function here reads a label: the channels come from the PairInput of load_pair, the tissue from the fixed image's Atropos layer.
"""
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
for _dir in ('p2', 'p1'):                                           # appended: no module already found first (he, mc0, ef, fl) is shadowed
    if str(HERE.parent/_dir) not in sys.path:
        sys.path.append(str(HERE.parent/_dir))

import p2_core as p2  # noqa: E402

OFF, MOMENT3 = 'off', 'moment3'
MODES = (OFF, MOMENT3)
REJECTED, IMPLEMENTATION, DATA, OTHER = 'provenance_rejected', 'implementation_error', 'tissue_data', 'tissue_error'
ESTIMATE_FAILED = 'estimate_failed'
COARSE_GRID = re.compile(r'^tissue pyramid: the coarse grid differs from four_level_grids$')     # p2_core.tissue_levels' two messages (a test reads them from the source)
NO_BRAIN_CELL = re.compile(r'^tissue level \d+: no cell inside the brain mask$')
D86_RULE = ('D86 (C22.B1-8): the input or the configuration (provenance or file check, class names or wm_index) -> provenance_rejected, no fallback; the implementation (a coarse grid) -> implementation_error, '
            'no fallback, the queue stops; the data (an empty brain mask, a level without an Omega_B cell) and estimate_failed -> an entry failure, the fallbacks; anything else -> tissue_error, the fallbacks, '
            'needs_lead_rejudgment')
ROLE = ('C22.2 T88 of the HE entry, method Hx24-RF05-F1200-v4 (C22 revision 8: tissue_prior off; with tissue_prior moment3 the ablation -C3 of C22.B1): uncertified, the repair and its certificate '
        'decide; predicted channels, not oracle; the output is the certified final state, not F_B')


def constants():
    """The constants of C (C22.B1, D83) as the frozen settings hold them (C22.B4-4); a test compares them with p2_core's and bayesreg.seg.channels' own."""
    return dict(passes=p2.C_PASSES, ratios_learned=list(p2.C_CLASSES), tau_background='tau_WM (the background class weighs as white matter)', moments_on='all cells of the native grid (Omega_B described)',
                normalization='per level, the mean of a over the Omega_B cells (all 8 nodes in the level mask) is 1', levels=4, coarse_levels='C03.6-12 (downsample_array, the simplex renormalized, the mask at >= 1/2)',
                b='not learned: the curl weight is 1', estimated_on="each pass's own native field at step 600, untapered, unrepaired; pass k+1 uses pass k's ratios; the ratios of pass 2's field are recorded only",
                restart='every pass from u = 0, Adam reset (the frozen code)', tissue_side='fixed', brain_threshold=.5, smoothing=dict(sigma_voxels=1., truncate=4., mode='nearest', eps=1e-3),
                energy='mu/2 [sum_K a_K (J_K - 1)^2 + sum_K |c_K|^2] (p2_core.pre_energy_p2), a_K = m / (P_bg + P_WM + rho_CSF P_CSF + rho_GM P_GM)', code='experiments/p2/p2_core.py at 090c985 (D83: frozen as the code)')


class TissueRefusal(Exception):
    """A refusal of the tissue preparation, with its D86 status and the step that refused."""

    def __init__(self, status, step, error):
        super().__init__(f'{step}: {type(error).__name__}: {error}')
        self.status, self.step, self.error = status, step, error

    def record(self):
        return dict(status=self.status, step=self.step, exception=type(self.error).__name__, message=str(self.error)[:1500], needs_lead_rejudgment=self.status == OTHER,
                    stop_queue=self.status == IMPLEMENTATION, fallback=self.status in (DATA, OTHER), rule=D86_RULE)


def _step(name, refusals, call):
    """call(); an exception becomes a TissueRefusal: the first (type, status[, message pattern]) of `refusals` that matches, else OTHER (D86 (4))."""
    try:
        return call()
    except Exception as error:                          # noqa: BLE001 - every exception of the step is classified here, none passes unrecorded
        for kind, status, *pattern in refusals:
            if isinstance(error, kind) and (not pattern or pattern[0].match(str(error))):
                raise TissueRefusal(status, name, error) from error
        raise TissueRefusal(OTHER, name, error) from error


@dataclass
class Tissue:
    brain: np.ndarray
    ptilde: np.ndarray
    names: list
    wm_index: int
    index: dict
    record: dict
    levels: list = None
    seconds: float = None


def refusals():
    """D86 by step: [(exception type, status[, message pattern])], the first that matches; nothing matching: tissue_error (D86 (4)).  A test reads this table against C22.B1-8 and the lead's ruling of
    2026-11-07 (the module docstring: tissue_layer is (1) for ProvenanceRejected only; simplex, grids, tissue_channels have no entry, so all of theirs is (4))."""
    from bayesreg.io.pair import ProvenanceRejected
    from jd_curl.validation import InputValidationError
    return dict(tissue_layer=[(ProvenanceRejected, REJECTED)], simplex=[], brain_mask=[(InputValidationError, DATA)], tissue_channels=[], class_index=[(ValueError, REJECTED)], grids=[],
                tissue_levels=[(ValueError, IMPLEMENTATION, COARSE_GRID), (ValueError, DATA, NO_BRAIN_CELL)])


def read_tissue(pin, seg_root):
    """run_p2.read_tissue (experiments/p2/run_p2.py:150-163 at 090c985), every call in the same order: run_pair.load_tissue_layer (C03.1-2/3: provenance, hashes, grid), C03.2 simplex with background,
    C03.3 Omega_B, C03.5 smooth and floor, the class index.  Each step's refusal is a TissueRefusal with its D86 status."""
    from bayesreg.cli import run_pair
    from bayesreg.seg import channels
    table = refusals()
    P, accepted, record = _step('tissue_layer', table['tissue_layer'], lambda: run_pair.load_tissue_layer(pin, 'fixed', seg_root))
    pbar, simplex = _step('simplex', table['simplex'], lambda: channels.simplex_with_background(P))
    del P
    brain, mask = _step('brain_mask', table['brain_mask'], lambda: channels.brain_mask(pbar))
    ptilde, smoothing = _step('tissue_channels', table['tissue_channels'], lambda: channels.tissue_channels(pbar))
    del pbar
    names = list(accepted['class_names'])
    index = _step('class_index', table['class_index'], lambda: p2.class_index(names, accepted['wm_index']))
    return Tissue(brain, ptilde, names, int(accepted['wm_index']), index, dict(record, accepted=dict(accepted), simplex=simplex, brain_mask=mask, smoothing=smoothing, n_brain=int(brain.sum())))


def levels_of(tissue, grid, grids, downsample, workers=1):
    """p2_core.tissue_levels on the four levels (run_p2.run_field's call, run_p2.py:196), its refusals by D86."""
    return _step('tissue_levels', refusals()['tissue_levels'], lambda: p2.tissue_levels(tissue.ptilde, tissue.brain, grid, grids, downsample, tissue.index, workers=workers))


def tissue_of(pin, seg_root, four_level_grids, downsample, workers=1):
    """read_tissue, the pair's four-level grids (four_level_grids(pin.grid), as run_p2.run_field computes them after read_tissue; HE2 review minor 5: inside a step, so that its exception is D86 (4)
    with needs_lead_rejudgment, not a T88 child without run.json), then levels_of; the seconds as run_p2's seconds_tissue."""
    tick = perf_counter()
    tissue = read_tissue(pin, seg_root)
    grids = _step('grids', refusals()['grids'], lambda: four_level_grids(pin.grid))
    tissue.levels = levels_of(tissue, pin.grid, grids, downsample, workers)
    tissue.seconds = perf_counter()-tick
    return tissue


def three_passes(spec, pin, predicted, tissue, config, *, device, started, prepare, table, observer, downsample, scale_check_of, workers, passes):
    """The pass loop of run_p2.run_field for the arm C (run_p2.py:200-248 at 090c985): every pass prepares its channels again (so s is computed again), pass 0 without hooks (optimize_p2 calls
    mc0_core.optimize_k: the frozen T88, C off), pass k > 0 with TissueWeights of pass k-1's ratios; between the passes the moment ratios of the pass's own field; the ratios of the last field recorded only.
    `passes` (a list) receives each pass's row as run_p2 writes it, also when a pass stops or an estimate fails; `prepare`, `table()`, `observer()` and `downsample()` are run_p2's (fl_channels.prepare_stacks,
    mc0_core.load_table, run_mc0.vram_guard, fast_channels.FastDownsample), called once a pass as there.  Returns (u, record, status, reason, scale, scale_check, seconds_channels); EstimateFailed is raised as run_p2 raises it (the caller: estimate_failed)."""
    status, reason, u, record, scale, scale_check, seconds_channels = 'done', None, None, None, None, None, None
    ratios = None
    n_passes = p2.C_PASSES
    for k in range(n_passes):
        tick = perf_counter()
        stacks, scale = prepare(spec, pin, predicted, table(), device, workers=workers)
        seconds_prepare = perf_counter()-tick
        if seconds_channels is None:
            seconds_channels = perf_counter()-started
        weights = p2.TissueWeights(tissue.levels, ratios)
        hooks = None if ratios is None else p2.Hooks(None, weights)
        tick = perf_counter()
        u, record = p2.optimize_p2(stacks, pin.grid, config, device=device, started=started, observer=observer(), workers=workers, record_level_norms=True, downsample=downsample(), checkpoint=None, hooks=hooks)
        del stacks
        scale_check = scale_check_of(record, scale)
        row = dict(pass_index=k, u_sha256=p2.sha_bytes(u), seconds_prepare=seconds_prepare, seconds_optimize=perf_counter()-tick, scale=scale, scale_check=scale_check, stopped=record['stopped'],
                   levels=[dict(level=lv['level'], iterations=lv['iterations'], seconds=lv['seconds'], first=lv['first'], last=lv['last']) for lv in record['levels']], cap_hits=record['cap_hits'], hooks=hooks is not None)
        passes.append(row)
        if k == 0:
            row['control'] = dict(checkable=False, same=None, why='HE reads no other run (C22.B1-1); pass 0 against the C-off T88 field is the smoke\'s comparison (C22.B4-6 (a))', field_kept=False)
        row['tissue'] = weights.record()
        del weights, hooks
        if record['stopped'] is not None:
            status, reason = 'time_cap', f"the wall cap {config.seconds} s at {record['stopped']} (pass {k})"
            break
        if not scale_check['passed']:
            status, reason = 'scale_mismatch', f'pass {k}: ||g_D|| of the native tensors {scale_check}'
            break
        if k < n_passes-1:
            ratios, row['estimate'] = p2.moment_ratios(u, tuple(pin.grid.spacing), tissue.ptilde, tissue.brain, tissue.names, tissue.wm_index, config.j0)
        else:
            try:                                                                        # the ratios of the last pass's field: a record only, never used, never fails the run (run_p2.py:244-248)
                row['estimate'] = p2.moment_ratios(u, tuple(pin.grid.spacing), tissue.ptilde, tissue.brain, tissue.names, tissue.wm_index, config.j0)[1]
            except p2.EstimateFailed as error:
                row['estimate'] = dict(error=str(error)[:500], ratios=None)
    return u, record, status, reason, scale, scale_check, seconds_channels


def summary(passes):
    """C22.B4-5: the three passes' u_sha256, the ratios used (pass 1, pass 2) and the last estimate, for line.json; None without passes."""
    if not passes:
        return None
    used = [(row.get('tissue') or {}).get('ratios_used') for row in passes]
    return dict(n_passes=len(passes), u_sha256=[row.get('u_sha256') for row in passes], ratios_used=used[1:], last_estimate=(passes[-1].get('estimate') or {}).get('ratios'),
                estimates=[(row.get('estimate') or {}).get('ratios') for row in passes], rule='C22.B1-6: pass 0 a = 1; pass k > 0 with the ratios of pass k-1; only the last field is kept')


def mark(segmenter_id, keys_read, sides):
    """The oracle mark for the anchor's segmenter: p2_core.ChannelSet.mark (run_hs.hs_oracle_mark's rule), imported."""
    return p2.ChannelSet(segmenter_id).mark(keys_read, sides)
