"""HB (runs/20261109-hb/任务书.md items 1, 2; contract C23.8-2, C22.H0-1, C22.H1-1; preregistration docs/bayesreg/notes/留出集预登记_20261107.md revision 1a item 6, item 18): HE on the B-line hold-out.

Only the admission changes (C23.8-2: "只改准入，不改算术"); T88, the repair, the certificates and the fallbacks are the same functions with the same frozen settings.
  The context    pair_set 'holdout', made only by run_hx.make_holdout_context: the stage holdout_common.STAGE_ROOT, the output the method arm's working directory holdout_common.work_dir(ARM, pair), the
                 paths of holdout_paths (he_core.HOLDOUT_SOURCES: HI's hold-out predictions, the hold-out Atropos output); run_hx refuses a hold-out context whose stage, output or paths are other
                 (run_hx.holdout_context_problem).  Two modes by the subcommand (run_hx.py pair / holdout-pair), not by an option that changes a path (C23.0-3).
  Every process  of the method -- the supervisor, the t88 and repair children, the third certification -- admits the pair itself (C23.3-5: an unseal object does not cross processes):
                 admitted_images -> holdout_common.admit_holdout_pair (the ids canonical; H1-H4 never; bayesreg.io.unseal.load_unseal and bind_unseal in this process; check_subject on the bound
                 allowlist; the row of holdout_B_pairs.tsv in its direction) -> the manifest's subjects (bayesreg.io.pair's whitelisting decoder) are the pair name's -> load_pair(path, allowlist=<bound>).
                 This takes the place of run_hx.sealed_admission for the hold-out ("sealed_admission 只为 holdout_B_v1 抬起": admit_holdout_pair lets holdout_B_v1 through and nothing else).
  The bound      allowlist goes on to fast_channels.read_predicted (allowlist_of): make_inputs.a4_admission -> a4_admission_unsealed (check_subject, D24 as hi_core.other_layers),
                 pair_structure_layer -> StructureLayer(allowlist=), atropos_wm -> validate_segmentation_provenance(allowlist=).  load_tissue_layer is on C's path only (tissue_prior moment3, the
                 ablation), which the method does not run (tissue_prior off).
Nothing here opens a subject's file before the last check passed; a refusal is HoldoutAdmissionRefused (a SubjectSealed: run_hx's admission refusals, so the supervisor writes hx_provenance_rejected and the
batch queue stops, P20).

LP-B (contract C26.6-2, C26.4 L30, L40-L50; LPBA40 preregistration 06121278... §4.1, §8.2 steps 4-5, §12 item 9): two more pair sets, 'lpba40_test' (formal) and 'lpba40_feas' (feasibility), one
value per mode, each with its own roots and its own pair table (experiments/lpba/lpba_common.roots, table).  Only the admission and the paths change; the method's settings (414eccd3...), its arithmetic
and the certificate are those of the hold-out:
  The context    made only by run_hx.make_lpba_context: the stage lpba_common.roots(pair_set).stage, the output the method cell <root>/work/method/<pair> (lpba_out_dir), the paths of lpba_paths (the mode's
                 HI predictions and Atropos output; the allowlist the pinned LPBA allowlist's path); run_hx.holdout_context_problem refuses an LPBA context whose stage, output, paths or pair are other.
  Every process  admits the pair itself (lpba_admitted_images; C26.6-2's five steps): 1-3 lpba40_pre_manifest (the stage directory the mode's; the pair name a row of the mode's table in its direction; the
                 pins; the pinned LPBA allowlist and both subjects through check_subject on it, the 34 admitted by its one exception, admitted_by_lpba40; no sealed list), which opens no file of the stage;
                 4 the manifest's subjects (bayesreg.io.pair's whitelisting decoder) are the pair name's; 5 load_pair with that allowlist (its name check, check_pair_subjects: the lpba40_guard key).  The record
                 takes the place of admission_second_lock (run.json keeps the key).  run_hx.sealed_admission (truth.check_name refuses these names) and he_holdout's unseal path are not used.
  The allowlist  handed on to fast_channels.read_predicted (allowlist_of) is the pinned LPBA allowlist's path (= the context's paths.allowlist): make_inputs.a4_admission's development branch,
                 pair_structure_layer -> StructureLayer (its file, sha256 = load_pair's guard record), atropos_wm (no sealed subject: no allowlist needed).
A refusal is HoldoutAdmissionRefused (run_hx's admission refusals: hx_provenance_rejected, the queue stops, C26.8-2).
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE.parent/'holdout') not in sys.path:
    sys.path.insert(0, str(HERE.parent/'holdout'))
if str(HERE.parent/'lpba') not in sys.path:             # LP-B: lpba_arms (imported only when an LPBA context is met, _lpba) is findable from every entry that imports this module
    sys.path.append(str(HERE.parent/'lpba'))

import holdout_common as hc  # noqa: E402

from bayesreg.io.guard import SubjectSealed  # noqa: E402

ARM = 'method'                                  # the method arm's name (holdout_common.work_dir, the batch queue's cells; bayesreg.eval.holdout's arm table proposes the same)
LPBA_PAIR_SETS = ('lpba40_test', 'lpba40_feas')         # C26.6-2 (revision 11b): one value per mode (lpba_common.PAIR_SETS; a test compares)
PAIR_SETS = ('dev', 'holdout')+LPBA_PAIR_SETS
_BOUND_OF = {}                                  # id(PairInput) -> (PairInput, the bound allowlist load_pair admitted it through): this process only


class HoldoutAdmissionRefused(SubjectSealed):
    """A hold-out pair refused before anything of its subjects was opened (holdout_common.Refused's stage and message)."""

    def __init__(self, stage, message):
        super().__init__(f'hold-out admission ({stage}): {message}')
        self.stage = stage


def holdout_paths(base):
    """The hold-out context's paths: run_hx.default_paths(out) with pred_root and atropos_root replaced by he_core.HOLDOUT_SOURCES (read at call time)."""
    import he_core as he
    return dict(base, pred_root=str(he.HOLDOUT_SOURCES['pred_root']), atropos_root=str(he.HOLDOUT_SOURCES['atropos_root']))


def pair_set_of(ctx):
    """'dev' for a context without the key (every context made before HB)."""
    return ctx.get('pair_set', 'dev')


def stage_dir():
    return Path(hc.STAGE_ROOT)


def out_dir(pair):
    return Path(hc.work_dir(ARM, pair))


def admitted_images(ctx):
    """(PairInput, record) of a hold-out context, in this process: the module docstring's order.  HoldoutAdmissionRefused, or load_pair's own refusals, otherwise."""
    from bayesreg.io.pair import _read_manifest, load_pair
    pair = ctx['pair']
    if pair_set_of(ctx) != 'holdout' or Path(ctx['stage_dir']) != stage_dir():
        raise HoldoutAdmissionRefused('stage', f"{ctx['stage_dir']} ({pair_set_of(ctx)}) is not the hold-out stage {stage_dir()}")
    try:
        fixed, moving = hc.split_pair(pair)
        hc.safe_name(pair, 'pair')
        record, bound = hc.admit_holdout_pair(fixed, moving)
    except hc.Refused as exc:
        raise HoldoutAdmissionRefused(exc.stage, str(exc)) from None
    path = Path(ctx['stage_dir'])/pair/'pair.npz'
    _, _, selected = _read_manifest(path.parent/'manifest.json')
    if (selected['fixed_subject'], selected['moving_subject']) != (fixed, moving):
        raise HoldoutAdmissionRefused('manifest', f"the manifest names {selected['fixed_subject']} <- {selected['moving_subject']}, not the pair {pair}")
    pin = load_pair(path, allowlist=bound)
    _BOUND_OF[id(pin)] = (pin, bound)
    return pin, dict(rule='he_holdout.admitted_images: holdout_common.admit_holdout_pair (load_unseal, bind_unseal in this process; check_subject; the pair list in its direction), the manifest\'s '
                          'subjects, then load_pair with the bound allowlist (C23.8-2); run_hx.sealed_admission is not used for the hold-out',
                     admission=record, fixed_subject=fixed, moving_subject=moving, allowlist_sha256=bound.sha256)


def allowlist_of(ctx, pin):
    """What fast_channels.read_predicted gets as its allowlist: for a hold-out context the very bound allowlist load_pair admitted `pin` through (this process); for an LPBA context the pinned LPBA
    allowlist's path (C26.6-2: equal to the context's paths.allowlist, checked here) for a PairInput lpba_admitted_images admitted in this process; the context's allowlist path otherwise."""
    if is_lpba(ctx):
        entry = _BOUND_OF.get(id(pin))
        pinned = str(lpba_allowlist_path())
        if entry is None or entry[0] is not pin:
            raise HoldoutAdmissionRefused('allowlist', 'this PairInput was not admitted by he_holdout.lpba_admitted_images in this process')
        if ctx['paths'].get('allowlist') != pinned or Path(entry[1].path).resolve() != Path(pinned).resolve():
            raise HoldoutAdmissionRefused('allowlist', f"the context's allowlist {ctx['paths'].get('allowlist')} is not the pinned LPBA allowlist {pinned} (C26.6-2)")
        return pinned
    if pair_set_of(ctx) != 'holdout':
        return ctx['paths']['allowlist']
    entry = _BOUND_OF.get(id(pin))
    if entry is None or entry[0] is not pin:
        raise HoldoutAdmissionRefused('allowlist', 'this PairInput was not admitted by he_holdout.admitted_images in this process')
    return entry[1]


# ---- LPBA40 (LP-B; contract C26.6-2) ----------------------------------------------------------------------------------------------------------------
def _lpba():
    """experiments/lpba/lpba_arms.py (and lpba_common through it), imported when an LPBA context is first met: a hold-out or development process imports nothing of it (its directory is on sys.path
    from this module's import, appended last)."""
    import lpba_arms
    return lpba_arms


def is_lpba(ctx):
    return pair_set_of(ctx) in LPBA_PAIR_SETS


def lpba_stage_dir(pair_set):
    """The mode's stage (C26.2-3; lpba_common.roots, read at call time)."""
    la = _lpba()
    la.check_pair_set(pair_set)
    path = la.lc.roots(pair_set).stage
    la.lc.guard_write(path)
    return path


def lpba_out_dir(pair_set, pair):
    """The method cell of an LPBA pair: <root>/work/method/<pair> (the arm table's field dir; nothing moves, prereg §11)."""
    return _lpba().work_dir(ARM, pair, pair_set)


def lpba_allowlist_path():
    from bayesreg.io import guard
    return guard.lpba40_allowlist_path()


def lpba_paths(pair_set, base):
    """An LPBA context's paths: run_hx.default_paths(out) with pred_root and atropos_root the mode's (lpba_common.roots) and the allowlist the pinned LPBA allowlist's path (C26.3-5, C26.6-2)."""
    r = _lpba().lc.roots(pair_set)
    return dict(base, pred_root=str(r.pred_root), atropos_root=str(r.atropos), allowlist=str(lpba_allowlist_path()))


def lpba40_pre_manifest(pair_set, pair, stage_dir=None):
    """C26.6-2 steps 1-3 (revision 11b, the second item): the stage directory, when given, is the mode's (step 1); the pair a plain LPBA40_<f>__LPBA40_<m>, a row of the mode's table in its
    direction (step 2); the pins (lpba_common.check_pins: the preregistration, the lists, the frozen files, the method's settings_sha256, the c record); the pinned LPBA allowlist (by its path and
    sha256 only); both subjects on no sealed list and through check_subject on it -- the 34 by its one exception (admitted_by_lpba40), S01 and S02 by steps (2), (3) (step 3); the pair name through
    guard.check_lpba40_names with that allowlist.  Opens no manifest and no file of the stage: code-tree files only.  Returns (record, allowlist); HoldoutAdmissionRefused otherwise."""
    la = _lpba()
    lc = la.lc
    from bayesreg.io.guard import LPBA40_TEST_REJECT_SHA256, SubjectSealed as Sealed, check_lpba40_names
    try:
        la.check_pair_set(pair_set)
        if stage_dir is not None and Path(stage_dir) != lpba_stage_dir(pair_set):
            raise lc.Refused('stage', f'{stage_dir} is not the {pair_set} stage {lpba_stage_dir(pair_set)} (C26.6-2 step 1)')
        hc.safe_name(pair, 'pair')
        fixed, moving = la.split(pair)
        pins = lc.check_pins()
        lists = lc.read_lists()
        allowlist, allowlist_record = lc.load_lpba_allowlist()
        rows, admitted = lc.admit_pairs(pair_set, [(fixed, moving)], allowlist, lists)
        try:
            check_lpba40_names([pair], allowlist)
        except Sealed as exc:
            raise lc.Refused('admission', str(exc)) from None
    except hc.Refused as exc:                           # review minor 3: holdout_common.safe_name's Refused too (lpba_common.Refused is one)
        raise HoldoutAdmissionRefused(exc.stage, str(exc)) from None
    table_rel = lc.TEST_PAIRS_REL if pair_set == la.FORMAL else lc.FEAS_PAIRS_REL
    record = dict(rule='he_holdout.lpba40_pre_manifest (C26.6-2 steps 1-3): the stage directory, the pair a row of the mode\'s table in its direction, the pins, the pinned LPBA allowlist, both subjects '
                       'on no sealed list and through bayesreg.io.guard.check_subject on it; no manifest and no file of the stage opened',
                  pair_set=pair_set, pair=pair, fixed_subject=fixed, moving_subject=moving, stage_dir=None if stage_dir is None else str(stage_dir), stage_constant=str(lpba_stage_dir(pair_set)),
                  pair_table=dict(path=table_rel, sha256=lc.LIST_SHA256[table_rel], row=rows[0], line=f'{fixed}\t{moving}'),
                  check_subject={s: dict(step=r['step'], item=r['item'], admitted_by_lpba40=r['admitted_by_lpba40'], sealed_lists=r['sealed_lists']) for s, r in admitted['subjects'].items()},
                  allowlist=dict(path=allowlist_record['path'], sha256=allowlist_record['sha256']), reject_constant_sha256=LPBA40_TEST_REJECT_SHA256, pins=pins)
    return record, allowlist


def lpba_admitted_images(ctx):
    """(PairInput, record) of an LPBA context in this process: C26.6-2's five steps (the module docstring).  The record takes the place of admission_second_lock (C26.6-2: the rule, the pair set, the
    stage, the pair table's path, sha256 and row, the manifest's subjects, both sides' check_subject, the LPBA allowlist's path and sha256, the constant's sha256, load_pair's lpba40_guard key)."""
    from bayesreg.io.pair import _read_manifest, load_pair
    pair = ctx['pair']
    pre, allowlist = lpba40_pre_manifest(pair_set_of(ctx), pair, ctx['stage_dir'])
    path = Path(ctx['stage_dir'])/pair/'pair.npz'
    _, _, selected = _read_manifest(path.parent/'manifest.json')
    if (selected['fixed_subject'], selected['moving_subject']) != (pre['fixed_subject'], pre['moving_subject']):
        raise HoldoutAdmissionRefused('manifest', f"the manifest names {selected['fixed_subject']} <- {selected['moving_subject']}, not the pair {pair} (C26.6-2 step 4)")
    pin = load_pair(path, allowlist=allowlist)
    _BOUND_OF[id(pin)] = (pin, allowlist)
    record = dict(pre, rule='he_holdout.lpba_admitted_images (C26.6-2): lpba40_pre_manifest (steps 1-3), the manifest\'s subjects (step 4), load_pair with the pinned LPBA allowlist (step 5); '
                            'run_hx.sealed_admission and the unseal path are not used', manifest_subjects=dict(fixed=selected['fixed_subject'], moving=selected['moving_subject']),
                  lpba40_guard=(pin.provenance['subject_guard'] or {}).get('lpba40_guard'), allowlist_sha256=allowlist.sha256)
    return pin, record


__all__ = ['ARM', 'HoldoutAdmissionRefused', 'LPBA_PAIR_SETS', 'PAIR_SETS', 'admitted_images', 'allowlist_of', 'holdout_paths', 'is_lpba', 'lpba40_pre_manifest', 'lpba_admitted_images',
           'lpba_allowlist_path', 'lpba_out_dir', 'lpba_paths', 'lpba_stage_dir', 'out_dir', 'pair_set_of', 'stage_dir']
