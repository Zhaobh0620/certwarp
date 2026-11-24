"""HE (runs/20261106-he/任务书.md; contract C22, revisions 1-3): the firewall-clean entry of the C22 line, one pair of any stage.  Rules, records and checks: he_core.py; fallback one: firewall_c.py.

    PYTHONPATH=<frozen>:<code> python -B experiments/he/run_hx.py pair --stage-dir DIR --pair NAME --out DIR [--device cuda] [--launcher unit|direct] [--unseal-token T]
        the method on one pair (the supervisor).  Admission first, two locks before the archive is opened or hashed (revision 1): bayesreg.eval.truth's sealed lists on the stage's and the pair's names and on
        the manifest's subjects, then load_pair's subject guard, the development allowlist only (an unseal token is recorded by its sha256 and admits nobody); every child checks both again.  Then the T88
        child, the repair child, the decision, the fallbacks (configuration C under the firewall, then the identity), the output:
        <out>/output/{final.npz, disp.npz, cert.json, line.json}.  No step opens a label member, a metrics file or anything an evaluation wrote.  The values are the frozen ones of C22.7 (no option changes them).
        --out: a new directory under /home/rmcfz/seq_vp_tag_results; a directory that exists is never rerun.  --launcher: how configuration C's run is started (unit: systemd-run, as D0; direct: the same
        command in this process's own unit).
    run_hx.py t88 --context FILE         the T88 child: <out>/t88/{run.json, field_0600.npz}
    run_hx.py repair --context FILE      the repair child: <out>/repair/{repair.json, repair.npz, certified.npz, cert.json}
    run_hx.py third-cert --context FILE  C22.3-9 (d): the output certified again in a process that reads no label; pass: the output stands; fail: the next fallback, the old output kept as output_superseded_<k>
                                         (no fallback after it: the output withdrawn whole as output_failed_<k>)
    run_hx.py evaluate --context FILE [--lead-release TEXT]
                                         the evaluation unit, once per pair: the review certificate before any label is read; on a mismatch with cert.json the third certification (a child) first; then the
                                         marker evaluation_started.json, then the truth (bayesreg.eval.truth: sealed lists, allowlist) and the metrics (bayesreg.eval.metrics) from disp.npz: <out>/evaluation.json.
                                         A marker already there: refused; only the lead's words (--lead-release) allow a second read, recorded.
    run_hx.py dry-run --stage-dir DIR --pair NAME      admission and the identity checks; no optimization, nothing written
    run_hx.py holdout-pair --pair NAME [--device cuda]  HB: the method on one pair of the B-line hold-out; the stage, the output (holdout_common.work_dir('method', NAME)) and the paths are fixed
                                         (make_holdout_context); every process admits the pair through he_holdout (load_unseal, bind_unseal in that process; C23.8-2).  The batch queue
                                         experiments/hb/hb_method.py starts it in a unit of its own (tag-hb-he-<pair>).  `pair` refuses the hold-out's roots.
    run_hx.py holdout-dry-run --pair NAME              HB: the hold-out admission and the identity; nothing written

Every process checks at its start that PYTHONPATH is exactly <frozen>:<code>, that the code checkout adds no top-level entry and no importable untracked one, that its code and configuration paths (bayesreg,
experiments, configs, splits, jd_curl, scripts) are committed and clean and experiments/ holds no untracked module, and that the frozen tree is eb808be's (its top level beyond eb808be only data and .git,
nothing untracked under jd_curl/); the supervisor, the third certification and the evaluation check again before they write their result.  A context that skips the identity or carries values other than the
frozen ones exists only through make_context (tests, controls), is marked test_context and writes outside the results root; the command line refuses any other.  Exit codes: 0 done (whatever the line status),
2 usage or a refused start, 8 a failed check, 9 an implementation error of C's tissue preparation (D86 (2): line.json says hx_provenance_rejected, implementation_error; a queue stops).

HE2 (runs/20261107-he2/任务书.md; contract C22 revision 4, BZ 2026-11-07): the method is Hx24-RF05-F1200-v4-C3.
  - The segmenter unet35_v4_ens (D80): its predictions, models and registry (configs/bayesreg/segmenters_hs_v1.json) are the reader's source; the anchor he_core.CONTRACT_SEGMENTER is checked by the supervisor
    and by the T88 child at their start (the settings, the registry read now, its models) and by the T88 child after the channels (the reader's records); the oracle mark is computed for the anchor's id.
  - C (D81 = option I, C22.B1, C22.B4): in the T88 child, the fixed image's Atropos tissue layer and three passes of T88 (he_c.py, p2_core's functions); only pass 2's field is kept; the D86 statuses.
  - H2: the supervisor reads its unit (unit_probe.py: the cgroup's memory files, systemd's properties) at its start and its end, into line.json `unit`; the nested unit of configuration C has
    OOMPolicy=continue and probes itself (firewall_c).  H3: after a failed third certificate, configuration C runs in a unit of its own (tag-he-c3-<pair>, D0's limits), never in the evaluation unit.

HB (runs/20261109-hb/任务书.md; contract C22 卷首修订 8/8b, C22.H, C23.8-2): the method is Hx24-RF05-F1200-v4 (tissue_prior off: the T88 child runs mc0_core.optimize_k once, HS's path; C is the ablation).
  - line.json's review_certificate_in is the frozen setting (he_core.REVIEW_CERTIFICATE_IN: the step-5 review record of the hold-out).
  - C22.5-6: an exception of the supervisor after the decision (the identity's certificate not passing: (a); anything else that leaves no output: (b)) writes line.json hx_implementation_error with the
    reason and the evidence by sha256, and exits 9 (he_core.STOP_QUEUE_EXIT): the queue stops.  No arithmetic changed.
  - The hold-out: holdout-pair and holdout-dry-run (he_holdout.py: the admission only); evaluate refuses a hold-out context (C22.H1-6: bayesreg.eval.holdout evaluates; the review is hb_review.py's).
"""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
from time import perf_counter
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import firewall_c as fc  # noqa: E402
import he_core as he  # noqa: E402
import he_holdout as hold  # noqa: E402

core = he.core
SCRIPT = HERE/'run_hx.py'
PY = fc.PY


# ---- the context ---------------------------------------------------------------------------------------------------------------------------------
def default_paths(out):
    """The sources of the method (C22.1-2 revision 4, D80): unet35_v4_ens's predictions, models and the images they were made from (experiments/p2/reference_unet35_v4_ens.json channels; run_hs.hs_source),
    its registry in the code checkout (the anchor's path), the folds file of UNetSource's default (as run_hs.hs_source), the Atropos posteriors (the alignment QC and C's tissue layer: one root)."""
    import mc0p_core as mc
    u = core.make_inputs().UNetSource()
    v4 = he.V4_SOURCES
    return dict(pred_root=str(v4['pred_root']), models_root=str(v4['models_root']), image_root=str(v4['image_root']), registry=str(he.anchor_registry()), folds=str(u.folds), folds_sha256=u.folds_sha256,
                atropos_root=str(mc.SEG_ROOT), allowlist=str(core.ALLOWLIST), frozen=str(core.FROZEN), cache=str(Path(out).parent/'cache'))


TEST_CONTEXT = 'a test or control context: the identity is not checked and the values may not be the frozen ones; never a result of the method'


def make_context(stage_dir, pair, out, *, device='cuda', settings=None, paths=None, launcher='unit', check_identity=True, workers=None, unseal_token=None, pair_set='dev'):
    """The context of one pair.  check_identity False or values other than the frozen ones: only through this function (tests, controls), and the context says so (test_context); the command line refuses such a
    context (cli_context_problem).  pair_set (HB): 'dev', or 'holdout' through make_holdout_context only (its stage, output and paths fixed; holdout_context_problem)."""
    settings = he.frozen_settings() if settings is None else settings
    paths = default_paths(out) if paths is None else paths
    import hashlib
    frozen = he.is_frozen(settings)
    ctx = dict(schema=he.SCHEMA.format('context'), stage_dir=str(Path(stage_dir)), pair=pair, out=str(Path(out)), device=device, workers=core.SETUP_WORKERS if workers is None else workers,
               settings=settings, settings_sha256=he.settings_sha256(settings), settings_frozen=frozen, paths=paths, launcher=dict(kind=launcher), check_identity=check_identity,
               test_context=TEST_CONTEXT if (check_identity is not True or not frozen) else None, code_head=code_head(),
               unseal_token_sha256=None if unseal_token is None else hashlib.sha256(unseal_token.encode()).hexdigest(), created_utc=he.utc(), method=he.METHOD, pair_set=pair_set)
    problem = function_context_problem(ctx) or holdout_context_problem(ctx)
    if problem:
        raise ValueError(problem)
    return ctx


def make_holdout_context(pair, *, device='cuda', launcher='unit', settings=None, check_identity=True, workers=None, pinned=None):
    """HB (C23.8-2): the context of a hold-out pair: the stage holdout_common.STAGE_ROOT, the output the method arm's working directory (holdout_common.work_dir('method', pair): every timed run on /home,
    prereg §5.3), the hold-out paths (he_holdout.holdout_paths).  No unseal token (an unseal object is made in each process from the committed record, C23.3).
    pinned: the batch's pinned commit (C22.H0-4), only compared with P1 of the committed runs/holdout_B_v1/pinned_code.json (rule one; never HEAD); the context's code_head is that P1 once
    holdout_common.pinned_code_problems passes (checked whenever the identity is, or a commit is given).  Every process of the pair then checks it again (check_identity)."""
    out = hold.out_dir(pair)
    ctx = make_context(hold.stage_dir(), pair, out, device=device, settings=settings, paths=hold.holdout_paths(default_paths(out)), launcher=launcher, check_identity=check_identity, workers=workers,
                       pair_set='holdout')
    if pinned is not None or check_identity:            # rule one: P1 from the committed runs/holdout_B_v1/pinned_code.json (never HEAD); `pinned` only compared with it
        record, problems = hold.hc.pinned_code_problems(pinned, repo=core.ROOT)
        if problems:
            raise ValueError(f'the code is not the pinned commit\'s (runs/holdout_B_v1/pinned_code.json; given {pinned}): {problems}')
        ctx['code_head'] = record['pinned']
    return ctx


def make_lpba_context(pair_set, pair, *, device='cuda', launcher='unit', settings=None, check_identity=True, workers=None, pinned=None):
    """LP-B (C26.6-2, L40): the context of an LPBA40 pair, made only here: the mode's stage, the method cell <root>/work/method/<pair> (he_holdout.lpba_out_dir), the mode's paths
    (he_holdout.lpba_paths: its HI predictions and Atropos output, the pinned LPBA allowlist).  The pair must be a row of the mode's table (holdout_context_problem); the settings the pinned ones
    (C26.6-1: 414eccd3...), unless a marked test context.  The code (C26.9): formal ('lpba40_test') -- rule one, P_L from runs/lpba40_test_v1/pinned_code.json through lpba_arms.pinned_code_problems
    (LP-C2's; fail closed until then), never HEAD, `pinned` only compared with it, code_head = P_L; feasibility ('lpba40_feas') -- not under rule one (C26.9-1): code_head is HEAD, which every process
    of the pair finds again (the development's rule) in a clean checkout.  Checked whenever the identity is, or a commit is given."""
    out = hold.lpba_out_dir(pair_set, pair)
    ctx = make_context(hold.lpba_stage_dir(pair_set), pair, out, device=device, settings=settings, paths=hold.lpba_paths(pair_set, default_paths(out)), launcher=launcher,
                       check_identity=check_identity, workers=workers, pair_set=pair_set)
    if pinned is not None or check_identity:
        la = hold._lpba()
        if pair_set == la.FORMAL:
            record, problems = la.pinned_code_problems(pinned, repo=core.ROOT)
            if problems or not record.get('pinned'):
                raise ValueError(f'the code is not the pinned commit\'s (C26.9-4, runs/lpba40_test_v1/pinned_code.json; given {pinned}): {problems or "no P_L"}')
            ctx['code_head'] = record['pinned']
        elif pinned is not None and pinned != ctx['code_head']:
            raise ValueError(f"--pinned {pinned} is not HEAD {ctx['code_head']} (feasibility, C26.9-1: the checkout's commit)")
    return ctx


def lpba_context_problem(ctx):
    """LP-B (C26.6-2: "holdout_context_problem 的 LPBA 分支按 pair_set 核"): an LPBA context only as make_lpba_context makes it -- the pair a row of its mode's table, the stage, the output and the paths
    the mode's (a formal context cannot carry the feasibility's roots or pairs, nor the reverse), the unit launcher, no unseal token, the settings the pinned ones (C26.6-1) unless a marked test context."""
    pair_set = hold.pair_set_of(ctx)
    la = hold._lpba()
    try:
        if not la.planned(pair_set, ctx.get('pair')):
            return f"{ctx.get('pair')!r} is not a row of the {pair_set} table in its direction (C26.5-3 last item, C26.6-2)"
        want_out = hold.lpba_out_dir(pair_set, ctx['pair'])
        want_stage = hold.lpba_stage_dir(pair_set)
        want_paths = hold.lpba_paths(pair_set, default_paths(want_out))
    except Exception as exc:                                            # noqa: BLE001 - lpba_common.Refused, holdout_common.RealRootInTest
        return f"the {pair_set} pair {ctx.get('pair')!r}: {exc}"
    if Path(ctx['stage_dir']) != want_stage or Path(ctx['out']) != want_out:
        return f"an {pair_set} context runs on {want_stage} into {want_out}, not {ctx['stage_dir']} into {ctx['out']}"
    if ctx.get('paths') != want_paths:
        return f'an {pair_set} context has exactly the paths of he_holdout.lpba_paths (its HI predictions and Atropos output, the pinned LPBA allowlist)'
    if (ctx.get('launcher') or {}).get('kind') != 'unit' or ctx.get('unseal_token_sha256') is not None:
        return f'an {pair_set} context launches configuration C in a unit of its own and carries no unseal token'
    if ctx.get('settings_sha256') != la.SETTINGS_SHA256 and ctx.get('test_context') != TEST_CONTEXT:
        return f"the settings_sha256 {ctx.get('settings_sha256')} is not the pinned {la.SETTINGS_SHA256} (C26.6-1): refused"
    return None


def holdout_context_problem(ctx):
    """None, or why a context's pair set does not hold (HB): 'dev' (or no key: every context before HB) runs as before; 'holdout' only with the stage, the output and the paths make_holdout_context
    gives (a hold-out context cannot point HE at the development's predictions, or a development context at the hold-out stage's admission), the unit launcher.  LP-B: 'lpba40_test' and
    'lpba40_feas' only as make_lpba_context makes them (lpba_context_problem)."""
    pair_set = hold.pair_set_of(ctx)
    if pair_set not in hold.PAIR_SETS:
        return f'pair_set {pair_set!r}: one of {hold.PAIR_SETS}'
    if pair_set == 'dev':
        return None
    if pair_set in hold.LPBA_PAIR_SETS:
        return lpba_context_problem(ctx)
    try:
        want_out = hold.out_dir(ctx['pair'])
    except Exception as exc:                                            # noqa: BLE001 - holdout_common.Refused (not a plain pair name) or RealRootInTest
        return f"the hold-out pair {ctx.get('pair')!r}: {exc}"
    if Path(ctx['stage_dir']) != hold.stage_dir() or Path(ctx['out']) != want_out:
        return f"a hold-out context runs on {hold.stage_dir()} into {want_out}, not {ctx['stage_dir']} into {ctx['out']}"
    if ctx.get('paths') != hold.holdout_paths(default_paths(want_out)):
        return 'a hold-out context has exactly the paths of he_holdout.holdout_paths (the hold-out predictions and Atropos)'
    if (ctx.get('launcher') or {}).get('kind') != 'unit' or ctx.get('unseal_token_sha256') is not None:
        return 'a hold-out context launches configuration C in a unit of its own and carries no unseal token'
    return None


def code_head(root=None):
    """The commit of the code checkout when the context is made (revision 2, review new minor 1): every later identity check of this pair's processes must find the same HEAD."""
    rc, out, _ = he.git(['rev-parse', 'HEAD'], core.ROOT if root is None else root)
    return out.strip() if rc == 0 else None


def under_results(out, results=None):
    results = (he.RESULTS if results is None else Path(results)).resolve()
    out = Path(out).resolve()
    return out == results or results in out.parents


def function_context_problem(ctx, results=None):
    """Why a context that does not check the identity may not run, even through the function interface (revision 2, review new minor 3): it writes under the results root.  A context with values other than
    the frozen ones that does check the identity may (the GPU smoke's forced fallback)."""
    if ctx.get('check_identity') is not True and under_results(ctx['out'], results):
        return f"a context that does not check the identity writes under the results root {he.RESULTS if results is None else results}: refused"
    return None


def context_path(ctx):
    return Path(ctx['out'])/'context.json'


def read_context(path):
    ctx = he.read_json(path)
    if ctx is None or ctx.get('schema') != he.SCHEMA.format('context'):
        raise SystemExit(f'{path}: not an HE context')
    if he.settings_sha256(ctx['settings']) != ctx['settings_sha256'] or he.is_frozen(ctx['settings']) is not ctx['settings_frozen']:
        raise SystemExit(f'{path}: the settings differ from their recorded sha256 or frozen mark')
    return ctx


def cli_context_problem(ctx, results=None):
    """Why a command-line stage may not run on this context (revision 1, review important 2), or None.  A context that does not check the identity, or whose values are not the frozen ones, must be marked as a
    test context and write outside the results root: it can never be taken for a result."""
    results = he.RESULTS if results is None else Path(results)
    problem = holdout_context_problem(ctx)                              # HB: a hold-out context only as make_holdout_context makes it
    if problem:
        return problem
    relaxed = ctx.get('check_identity') is not True or ctx.get('settings_frozen') is not True
    if not relaxed:
        return None
    if ctx.get('test_context') != TEST_CONTEXT:
        return f"the context does not check the identity ({ctx.get('check_identity')}) or its values are not the frozen ones ({ctx.get('settings_frozen')}), and it is not marked as a test context"
    if results.resolve() in Path(ctx['out']).resolve().parents or Path(ctx['out']).resolve() == results.resolve():
        return f"a test context writes under the results root {results}: refused"
    return None


def pair_path(ctx):
    return Path(ctx['stage_dir'])/ctx['pair']/'pair.npz'


def sealed_admission(ctx):
    """The second lock of the admission (revision 1, review important 3), before load_pair opens or hashes the archive: bayesreg.eval.truth's sealed lists (each pinned by sha256; a list missing or changed:
    SealedListError, refused), first on the stage's and the pair's names (no file opened), then on both subjects of the manifest read through the loader's whitelisting decoder.  The allowlist is the first lock
    (load_pair); unsealing, when BZ does it, has to lift both.  Returns the record; SubjectSealed otherwise."""
    from bayesreg.eval import truth as bench_truth
    from bayesreg.io.guard import SubjectSealed
    from bayesreg.io.pair import _read_manifest
    stage_dir = Path(ctx['stage_dir'])
    sealed = bench_truth.load_sealed_lists()
    named = bench_truth.check_name(stage_dir.name, ctx['pair'], sealed)
    _, _, selected = _read_manifest(stage_dir/ctx['pair']/'manifest.json')
    for side in ('fixed', 'moving'):
        which = sealed.which(selected[f'{side}_subject'])
        if which is not None:
            raise SubjectSealed(f"{side} subject {selected[f'{side}_subject']!r} is on the sealed list {which} (the second lock of the admission)")
    return dict(rule='bayesreg.eval.truth sealed lists: the stage and pair names, then the manifest subjects, before the archive is opened', sealed_lists=sealed.record, pair_name_subjects=named,
                fixed_subject=selected['fixed_subject'], moving_subject=selected['moving_subject'])


ADMISSION_REFUSALS = None


def admission_refusals():
    """The exceptions of a refused admission: the subject guards, provenance, a malformed input, a missing file, a sealed list that is not the pinned one, a pair or stage name that is not a plain name."""
    global ADMISSION_REFUSALS
    if ADMISSION_REFUSALS is None:
        from bayesreg.eval.truth import SealedListError
        from bayesreg.io.guard import SubjectSealed
        from bayesreg.io.pair import ProvenanceRejected
        from jd_curl.validation import InputValidationError
        ADMISSION_REFUSALS = (SubjectSealed, SealedListError, ProvenanceRejected, InputValidationError, OSError, ValueError)
    return ADMISSION_REFUSALS


def load_images(ctx, unseal_token=None):
    """The firewall view (bayesreg.io.pair.load_pair): the subject guard on the manifest's ids before the archive is opened or hashed, then fixed, moving, an all-zero initial_u; nothing else."""
    from bayesreg.io.pair import load_pair
    return load_pair(pair_path(ctx), allowlist=ctx['paths']['allowlist'], unseal_token=unseal_token)


def admitted_images(ctx, unseal_token=None):
    """(PairInput, the second lock's record): sealed_admission, then load_images.  Every process of the method opens the pair this way.
    HB (C23.8-2): a hold-out context goes to he_holdout.admitted_images instead (load_unseal and bind_unseal in this process, check_subject, the pair list, load_pair with the bound allowlist).
    LP-B (C26.6-2, L30): an LPBA context goes to he_holdout.lpba_admitted_images (lpba40_pre_manifest, the manifest's subjects, load_pair with the pinned LPBA allowlist); never sealed_admission."""
    if hold.pair_set_of(ctx) == 'holdout':
        return hold.admitted_images(ctx)
    if hold.is_lpba(ctx):
        return hold.lpba_admitted_images(ctx)
    record = sealed_admission(ctx)
    return load_images(ctx, unseal_token=unseal_token), record


def registry_of(ctx):
    """The registry the reader is pointed at: the context's, else the anchor's path in this checkout.  Whichever it is, the anchor checks it (he_core.segmenter_start_problems)."""
    return Path(ctx['paths'].get('registry') or he.anchor_registry())


def unet_source(ctx):
    """make_inputs.UNetSource with the anchor's segmenter id, the registry and the gate record (run_hs.hs_source's fields; C22.9-4 (xv) 'the reader's source')."""
    p = ctx['paths']
    return core.make_inputs().UNetSource(pred_root=Path(p['pred_root']), folds=Path(p['folds']), folds_sha256=p['folds_sha256'], models_root=Path(p['models_root']), image_root=Path(p['image_root']),
                                         segmenter_id=he.CONTRACT_SEGMENTER['segmenter_id'], registry=registry_of(ctx), gate=dict(he.SEGMENTER_GATE))


def tuples(pre):
    return {k: tuple(v) if isinstance(v, list) else v for k, v in pre.items()}


def env_record(torch=None, device='cpu'):
    record = dict(python=sys.version.split()[0], numpy=np.__version__, pid=os.getpid(), host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
    if torch is not None:
        cuda = torch.device(device).type == 'cuda'
        record.update(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0) if cuda else 'cpu', cudnn=torch.backends.cudnn.version() if cuda else None,
                      device_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if cuda else None, device_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if cuda else None,
                      deterministic_algorithms=torch.are_deterministic_algorithms_enabled(), cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    return record


def started(directory, ctx, stage):
    directory = Path(directory)
    directory.mkdir(parents=False, exist_ok=False)                     # never rerun
    he.write_json(directory/'started.json', dict(stage=stage, utc=he.utc(), argv=sys.argv, pid=os.getpid(), settings_sha256=ctx['settings_sha256'], settings_frozen=ctx['settings_frozen']))
    return directory


def enable_jit(ctx):
    from jd_curl.three_d import jit_cache
    return jit_cache.enable(Path(ctx['paths']['cache'])/'jit')


class SideOutcomes:
    """Each side of the channel reader (experiments/ef/fast_channels.py read_side, looked up by read_predicted as a module name; experiments/d0/make_inputs.py alignment_qc, looked up by read_side) on its own
    record (revision 1, review minor 6): 'qc' when that side's own alignment QC (the call made in that side's thread) failed and it refused; 'refused' for any other refusal of that side.  A side whose QC failed
    does not stop the other one: the wrapper returns a placeholder and the other side is still read, so both sides are always known.  Then: any 'refused' side, or a refusal before the sides (admission,
    structure layer) -> provenance_rejected (D74 (4)); only QC refusals -> segmentation_qc (D74 (2)).  The functions are called as they are; no number changes."""

    def __enter__(self):
        import threading
        import fast_channels as fast
        self.fast, self.mi, self.local = fast, core.make_inputs(), threading.local()
        self.real_qc, self.real_side = self.mi.alignment_qc, fast.read_side
        self.rows, self.sides = [], {}
        recorder = self

        def observed(*args, **kwargs):
            row = recorder.real_qc(*args, **kwargs)
            recorder.rows.append(row)
            recorder.local.qc = row
            return row

        def side(layer, pin, name, *args, **kwargs):
            recorder.local.qc = None
            try:
                return recorder.real_side(layer, pin, name, *args, **kwargs)
            except SystemExit as exc:
                qc = recorder.local.qc
                own_qc = qc is not None and qc.get('passed') is False
                recorder.sides[name] = dict(outcome='qc' if own_qc else 'refused', reason=str(exc)[:1500], alignment_qc=qc)
                if own_qc:
                    return None, dict(qc_refused=True)                  # the other side is read too
                raise
        self.mi.alignment_qc, fast.read_side = observed, side
        return self

    def __exit__(self, *exc):
        self.mi.alignment_qc, self.fast.read_side = self.real_qc, self.real_side
        return False

    def qc_refused(self):
        return bool(self.sides) and all(s['outcome'] == 'qc' for s in self.sides.values())

    def reason(self):
        return '; '.join(f"{side}: {s['reason']}" for side, s in sorted(self.sides.items()))[:2000]


# ---- the T88 child -------------------------------------------------------------------------------------------------------------------------------
MARK_RULE = ("the oracle mark computed again for the anchor's segmenter (p2_core.ChannelSet.mark, run_hs.hs_oracle_mark's rule): fast_channels.read_predicted's own mark (mc0p_core.oracle_mark) names "
             "unet35_v1_ens only; the reader's mark is kept as reader_mark (C22.1-2 revision 4)")


def t88_main(ctx, *, _spied=False):
    """run_fl.run_field's flow (experiments/fl/run_fl.py:127-196 at 85fc80d) for one pair of any stage, without checkpoints (the field is the last pass's return, which run_fl compared with its last checkpoint).
    A refusal of the channel reader is recorded here (C22.9-4 (iv)): segmentation_qc when only the alignment QC refused (SideOutcomes), provenance_rejected otherwise; the admission (both locks) again first.
    HE2: the segmenter's anchor at the start (he_core.segmenter_start_problems) and after the channels (channel_source_problems), the mark for the anchor's id; tissue_prior moment3 (the method, C22.B1):
    the fixed image's tissue layer and three passes (he_c.tissue_of, he_c.three_passes: p2_core's functions; run_p2.run_field's arm C), the D86 statuses; tissue_prior off (the ablation's definition,
    C22.B0-2): one pass of mc0_core.optimize_k, the code of revision 3.
    LP-B (review I3): an LPBA context runs under the allow-list spy (lpba_spied; record t88/opened_files_t88_<k>.json)."""
    if hold.is_lpba(ctx) and not _spied:
        return lpba_spied(ctx, 't88', Path(ctx['out'])/'t88', lambda: t88_main(ctx, _spied=True))
    t0 = perf_counter()
    fz = core.frozen_api()
    import torch
    import fast_channels as fast
    import fl_channels as channels
    import he_c
    import run_mc0 as run
    import run_mc0p as mc0p
    s = ctx['settings']
    out = started(Path(ctx['out'])/'t88', ctx, 't88')
    try:
        pin, second_lock = admitted_images(ctx)
    except admission_refusals() as exc:
        he.write_json(out/'run.json', dict(schema=he.SCHEMA.format('t88'), pair=ctx['pair'], stage_dir=ctx['stage_dir'], status=he.T88_REJECTED, reason=f'admission: {type(exc).__name__}: {exc}'[:2000], oracle=None, field=None,
                                           settings_sha256=ctx['settings_sha256'], settings_frozen=ctx['settings_frozen'], ended_utc=he.utc()))
        return 0
    device = ctx['device']
    device_cap = mc0p.limit_device_memory(torch, device, limit=s['device_limit_gib']*2**30)
    pre = s['t88']['pre']
    config = fz.PreoptConfig(**tuples(pre))
    ch = s['channels']
    spec = dict(k=ch['k'], channels=ch['channels'], weight=ch['weight'], D_factor=ch['D_factor'])
    mode = s.get('tissue_prior')
    status, reason, u, record, scale, scale_check, sources, digest, qc_rows, sides, seconds_labels, seconds_channels, seconds_task = 'done', None, None, None, None, None, None, None, [], {}, None, None, None
    rejection_stage = rejection_kind = source_problems = tissue = tissue_failure = None
    passes = []
    anchor_problems, anchor = he.segmenter_start_problems(s, registry=registry_of(ctx))
    if mode not in he_c.MODES:
        status, reason, rejection_stage = he.T88_REJECTED, f'the settings name tissue_prior {mode!r}, none of {he_c.MODES}', 'settings'
    elif anchor_problems:
        status, reason, rejection_stage, rejection_kind = he.T88_REJECTED, f'the segmenter anchor at the start (C22.9-4 (xv)): {anchor_problems}'[:2000], 'segmenter_anchor', 'segmenter_anchor'
    elif ch['sigma'] != fast.SIGMA or ch['colour_table_sha256'] != core.COLOUR_TABLE_SHA256:
        status, reason, rejection_stage = he.T88_REJECTED, f"the settings name channel sigma {ch['sigma']}, colour table {ch['colour_table_sha256'][:12]}: not the reader's (fast_channels, mc0_core)", 'settings'
    else:
        try:
            with SideOutcomes() as outcomes:
                try:
                    predicted = fast.read_predicted(pin, unet=unet_source(ctx), seg_root=Path(ctx['paths']['atropos_root']), allowlist=hold.allowlist_of(ctx, pin), workers=ctx['workers'])
                finally:
                    qc_rows, sides = list(outcomes.rows), dict(outcomes.sides)
            if sides:                                                   # no side was 'refused' (that one raises): only the QC refused
                raise SystemExit(outcomes.reason())
            seconds_labels = perf_counter()-t0
            sources = predicted.record
            reader_mark = {k: sources.get(k) for k in ('oracle', 'truth_labels_read')}
            sources.update(he_c.mark(he.CONTRACT_SEGMENTER['segmenter_id'], sources['pair_keys_read'], sources['sides']), reader_mark=reader_mark, mark_rule=MARK_RULE)
            import run_ef as efrun
            digest = efrun.labels_digest(predicted.labels)
            source_problems = he.channel_source_problems(sources)
            if source_problems:
                status, reason, rejection_stage, rejection_kind = he.T88_REJECTED, f'the segmenter anchor after the channels (C22.9-4 (xv)): {source_problems}'[:2000], 'segmenter_anchor', 'segmenter_anchor'
            elif sources['oracle'] is not False:
                status, reason = 'not_predicted_channels', f"channel sources: oracle {sources['oracle']}, members read {sources['pair_keys_read']}; no optimization"
            elif mode == he_c.OFF:
                stacks, scale = channels.prepare_stacks(spec, pin, predicted, core.load_table(), device, workers=ctx['workers'])
                del predicted
                seconds_channels = perf_counter()-t0
                u, record = core.optimize_k(stacks, pin.grid, config, device=device, started=t0, observer=run.vram_guard(torch, device, s['device_limit_gib']*2**30), workers=ctx['workers'],
                                            record_level_norms=True, downsample=fast.FastDownsample())
                seconds_task = perf_counter()-t0
                del stacks
                scale_check = channels.scale_check_of(record, scale)
                if record['stopped'] is not None:
                    status, reason = 'time_cap', f"the wall cap {pre['seconds']} s at {record['stopped']}"
                elif not scale_check['passed']:
                    status, reason = 'scale_mismatch', f"||g_D|| of the native tensors {scale_check['g_norm_scaled']} differs from {scale_check['target']}"
            else:
                tissue = he_c.tissue_of(pin, Path(ctx['paths']['atropos_root']), fz.four_level_grids, fz.downsample_array, workers=ctx['workers'])
                u, record, status, reason, scale, scale_check, seconds_channels = he_c.three_passes(
                    spec, pin, predicted, tissue, config, device=device, started=t0, prepare=channels.prepare_stacks, table=core.load_table,
                    observer=lambda: run.vram_guard(torch, device, s['device_limit_gib']*2**30), downsample=fast.FastDownsample, scale_check_of=channels.scale_check_of, workers=ctx['workers'], passes=passes)
                seconds_task = perf_counter()-t0
                del predicted
            if status == 'time_cap':
                np.savez(out/'field_stopped.npz', u=u)                        # evidence of a run cut short; never read
        except SystemExit as exc:
            only_qc = bool(sides) and all(x['outcome'] == 'qc' for x in sides.values())
            status, reason = (he.T88_QC if only_qc else he.T88_REJECTED), str(exc)[:2000]
        except he_c.TissueRefusal as refusal:                           # D86: the tissue preparation's refusals by step (he_c)
            tissue_failure = refusal.record()
            status, reason, rejection_stage = refusal.status, str(refusal)[:2000], 'tissue'
            rejection_kind = 'tissue_input' if refusal.status == he_c.REJECTED else None
        except run.VramLimit as error:
            status, reason = 'vram_stop', str(error)
        except torch.cuda.OutOfMemoryError as error:
            status, reason = 'cuda_oom', str(error)[:500]
        except FloatingPointError as error:
            status, reason = 'nonfinite', str(error)[:500]
        except he_c.p2.EstimateFailed as error:                         # a moment ratio or a cell weight without a finite positive value: an entry failure (C22.B1-8, D86 (3))
            status, reason = he.T88_ESTIMATE, str(error)[:500]
    field = final_equals_last_pass = None
    if status == he.T88_DONE:
        np.savez(out/he.FIELD_FILE, u=u)
        field = dict(file=he.FIELD_FILE, sha256=core.sha_file(out/he.FIELD_FILE), u_sha256=he.array_sha256(u))
        final_equals_last_pass = None if not passes else passes[-1]['u_sha256'] == field['u_sha256']
    optimizer = None if record is None else dict(record, role=he_c.ROLE, role_of_the_copied_code=record.get('role'))       # C22.9-4 (xv): this line's wording; the copied code's own string kept beside it
    moment3 = mode == he_c.MOMENT3
    he.write_json(out/'run.json', dict(
        schema=he.SCHEMA.format('t88'), pair=ctx['pair'], stage_dir=ctx['stage_dir'], status=status, reason=reason, rejection_stage=rejection_stage, rejection_kind=rejection_kind,
        oracle=None if sources is None else sources['oracle'], role=he_c.ROLE, method=he.METHOD,
        spec=spec, config=pre, scale=scale, scale_check=scale_check, labels_sha256=digest, channel_sources=sources, segmenter_anchor=dict(start=anchor, start_problems=anchor_problems, after_channels_problems=source_problems),
        alignment_qc=qc_rows, side_refusals=sides, admission_second_lock=second_lock, optimizer=optimizer, field=field, taper=s['t88']['taper'],
        tissue_prior=mode, bayes=dict(em=False, tissue=moment3, passes=he_c.p2.C_PASSES if moment3 else 1), c=s.get('c') if moment3 else None,
        tissue_layer=None if tissue is None else dict(tissue.record, root=ctx['paths']['atropos_root']), tissue_failure=tissue_failure, needs_lead_rejudgment=bool(tissue_failure and tissue_failure['needs_lead_rejudgment']),
        passes=passes if moment3 else None, c_summary=he_c.summary(passes), final_equals_last_pass=final_equals_last_pass, seconds_tissue=None if tissue is None else tissue.seconds,
        seconds_task=seconds_task, seconds_labels=seconds_labels, seconds_channels=seconds_channels, seconds_total=perf_counter()-t0, random_seed='none: no random step', frozen_commit=core.FROZEN_COMMIT,
        data=dict(pair_sha256=pin.provenance['pair_sha256'], manifest_sha256=pin.provenance['manifest_sha256'], fixed_subject=pin.fixed_subject, moving_subject=pin.moving_subject, keys_read=list(pin.provenance['keys_read']),
                  label_firewall=pin.provenance['label_firewall'], subject_guard=pin.provenance['subject_guard']),
        settings_sha256=ctx['settings_sha256'], settings_frozen=ctx['settings_frozen'], env=env_record(torch, device), device_cap=device_cap, ended_utc=he.utc()))
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()                                       # with the repair in the same process (spy_pair), the T88's cached blocks go back to the device (review, smoke item 6); no number changes
    return 0


# ---- the repair child ----------------------------------------------------------------------------------------------------------------------------
def source_checked(t88_dir, fz, pin, ctx):
    """(tapered u, source record).  C22.3-1: t88/run.json says done and oracle false, the field file's sha256 is the record's, u's hash the record's u_sha256; the taper is done here.  Nothing an evaluation
    wrote is read.  ValueError otherwise."""
    t88_dir = Path(t88_dir)
    record = he.read_json(t88_dir/'run.json')
    if record is None or record.get('schema') != he.SCHEMA.format('t88'):
        raise ValueError(f'{t88_dir}: no T88 record')
    if record.get('status') != he.T88_DONE or record.get('oracle') is not False or record.get('pair') != ctx['pair']:
        raise ValueError(f"{t88_dir}: not a done field of this pair with oracle false: {record.get('status')}, {record.get('oracle')}, {record.get('pair')}")
    if record.get('data', {}).get('pair_sha256') != pin.provenance['pair_sha256']:
        raise ValueError(f'{t88_dir}: the field is of another pair.npz')
    field = record['field']
    if core.sha_file(t88_dir/field['file']) != field['sha256']:
        raise ValueError(f"{t88_dir}/{field['file']} differs from run.json")
    with np.load(t88_dir/field['file'], allow_pickle=False) as z:
        raw = np.array(z['u'])
    if he.array_sha256(raw) != field['u_sha256']:
        raise ValueError(f"{t88_dir}/{field['file']}: u differs from run.json u_sha256")
    tick = perf_counter()
    u, _ = fz.boundary_taper(raw, pin.grid, ctx['settings']['t88']['taper'])
    u = np.ascontiguousarray(u)
    return u, dict(directory=str(t88_dir), file=field['file'], field_sha256=field['sha256'], u_sha256=field['u_sha256'], tapered_u_sha256=he.array_sha256(u), seconds_taper=perf_counter()-tick,
                   seconds_task=record.get('seconds_task'), checked='t88/run.json and the field file by their own sha256; the taper in this process (C22.3-1); no evaluation record read')


def repair_main(ctx, *, _spied=False):
    """run_fl.repair_one's flow (experiments/fl/run_fl.py:267-313 at 85fc80d) on the T88 child's field, with the fuse of the settings (RW's 1200 s); the decision certificate is this process's own certificate
    of the final v (C22.4-1), kept whole by RecordingCertify and written as cert.json beside the field.  LP-B (review I3): an LPBA context runs under the allow-list spy (lpba_spied)."""
    if hold.is_lpba(ctx) and not _spied:
        return lpba_spied(ctx, 'repair', Path(ctx['out'])/'repair', lambda: repair_main(ctx, _spied=True))
    t0 = perf_counter()
    fz = core.frozen_api()
    jit = enable_jit(ctx)
    import torch
    from jd_curl.three_d.feasibility import certify_map
    import repair as rp
    import run_ef as efrun
    import run_mc0 as run
    import run_mc0p as mc0p
    s = ctx['settings']
    out = started(Path(ctx['out'])/'repair', ctx, 'repair')
    device = ctx['device']
    record = dict(schema=he.SCHEMA.format('repair'), pair=ctx['pair'], role=rp.ROLE+'; the HE entry: its certificate of the final v is the decision (C22.4-1)', settings_sha256=ctx['settings_sha256'], settings_frozen=ctx['settings_frozen'])
    try:
        pin, second_lock = admitted_images(ctx)
    except admission_refusals() as exc:                 # both locks again: a refusal here is a refusal, never a fallback
        he.write_json(out/'repair.json', dict(record, status=he.T88_REJECTED, passed=False, reason=f'admission: {type(exc).__name__}: {exc}'[:2000], env=env_record(), ended_utc=he.utc()))
        return 0
    if hold.is_lpba(ctx):                               # LP-B (C26.6-2): the repair's record keeps the LPBA admission in admission_second_lock's place (the hold-out's record unchanged)
        record['admission_second_lock'] = second_lock
    try:
        device_cap = mc0p.limit_device_memory(torch, device, limit=s['device_limit_gib']*2**30)
        u, source = source_checked(Path(ctx['out'])/'t88', fz, pin, ctx)
        config = he.certificate_config(s)
        if asdict(config) != asdict(run.certificate_config()):
            raise ValueError(f'the settings certificate {asdict(config)} is not configuration C {asdict(run.certificate_config())}')
        tick = perf_counter()
        efrun.tiny_warmup(certify_map, config, rp.locate_cells)
        seconds_warmup_numba = perf_counter()-tick
        tick = perf_counter()
        warmed = efrun.device_warmup(device, certify_map, config)
        seconds_warmup_device = perf_counter()-tick
        settings = rp.RepairConfig(**s['repair'])
        tick = perf_counter()
        cells = rp.locate_cells(u, pin.grid, config)
        seconds_first_locate = perf_counter()-tick
        recorder = he.RecordingCertify(certify_map)
        tick = perf_counter()
        result = rp.repair_field(u, pin.grid, cells, settings, certify=recorder, certificate_config=config, device=device)
        seconds_repair = seconds_first_locate+perf_counter()-tick
        np.savez(out/'repair.npz', nodes=result.attempt_nodes, values=result.attempt_values)
        decision = None
        if result.passed:
            v = np.ascontiguousarray(result.u)
            digest = he.array_sha256(v)
            final = recorder.of(digest)
            if final is None or not he.passes(final):
                raise ValueError('no passing certificate of the returned field among the calls of certify')
            doc = he.certificate_document(final, config, process=f'the repair process (run_hx.py repair, pid {os.getpid()}; reads no label)', array_sha256_of_u=digest)
            np.savez(out/'certified.npz', u=v)
            he.write_json(out/'cert.json', doc)
            decision = dict(file='certified.npz', sha256=core.sha_file(out/'certified.npz'), state_hash=digest, cert='cert.json', cert_sha256=core.sha_file(out/'cert.json'))
        from jd_curl.three_d import jit_cache
        record.update(status='certified' if result.passed else 'not_certified', passed=result.passed, reason=result.reason, eps=settings.eps, config=asdict(settings), source=source, n_located=len(cells),
                      initial_certificate=result.initial_certificate, certificate=result.certificate, rounds=result.rounds, decision=decision, wall_hit=None,
                      n_certify_calls=len(recorder.results), seconds_warmup=seconds_warmup_numba+seconds_warmup_device, seconds_warmup_numba=seconds_warmup_numba, seconds_warmup_device=seconds_warmup_device, device_warmup=warmed,
                      seconds_first_locate=seconds_first_locate, seconds_repair=seconds_repair, attempt=dict(file='repair.npz', sha256=core.sha_file(out/'repair.npz'), n_free_nodes=len(result.attempt_nodes)),
                      jit_cache_enabled=jit, jit_cache=jit_cache.record(), certificate_config=asdict(config), frozen_commit=core.FROZEN_COMMIT,
                      data=dict(pair_sha256=pin.provenance['pair_sha256'], manifest_sha256=pin.provenance['manifest_sha256'], keys_read=list(pin.provenance['keys_read'])), device_cap=device_cap)
        record['wall_hit'] = he.hit_wall(record)
    except Exception as exc:                            # C22.3-9 (b): the supervisor goes to the fallback; the evidence stays
        record.update(status='error', passed=False, error=f'{type(exc).__name__}: {exc}'[:2000], traceback=traceback.format_exc()[-4000:])
    record.update(env=env_record(torch, device), seconds_total=perf_counter()-t0, ended_utc=he.utc())
    he.write_json(out/'repair.json', record)
    return 0


def certified_consistency(directory, record, grid):
    """True when the repair child's certified.npz and cert.json agree with repair.json and pass; the reason otherwise."""
    if record is None or record.get('passed') is not True:
        return None
    decision = record.get('decision') or {}
    directory = Path(directory)
    try:
        if core.sha_file(directory/'certified.npz') != decision.get('sha256') or core.sha_file(directory/'cert.json') != decision.get('cert_sha256'):
            return 'certified.npz or cert.json differs from repair.json'
        with np.load(directory/'certified.npz', allow_pickle=False) as z:
            u = np.ascontiguousarray(np.array(z['u']))
        cert = he.read_json(directory/'cert.json')
        digest = he.array_sha256(u)
        problem = he.field_problem(u, grid)                     # revision 1, review minor 4: an output that write_output would refuse is a repair error (a fallback), not a crash
        if problem:
            return f'certified.npz {problem}'
        if not (digest == decision.get('state_hash') == cert.get('state_hash') == (record.get('certificate') or {}).get('state_hash')):
            return 'the state hashes of certified.npz, cert.json and repair.json differ'
        if not he.passes(cert) or not he.passes(record.get('certificate') or {}):
            return 'the certificate does not pass'
    except (OSError, ValueError, KeyError) as exc:
        return f'{type(exc).__name__}: {exc}'
    return True


# ---- the children ----------------------------------------------------------------------------------------------------------------------------------
class SubprocessChild:
    """A stage in its own process: python -B run_hx.py <stage> --context FILE, with the timeout of the settings (the units' RuntimeMaxSec of RW); the environment is this process's (PYTHONPATH included)."""
    kind = 'subprocess'

    def __call__(self, stage, ctx):
        timeout = {'t88': ctx['settings']['children']['t88_timeout_seconds'], 'repair': ctx['settings']['children']['repair_timeout_seconds']}.get(stage)
        argv = fc.check_launch([PY, '-B', str(SCRIPT), stage, '--context', str(context_path(ctx))])
        log = Path(ctx['out'])/f'{stage}.log'
        with log.open('a') as f:
            try:
                return subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                f.write(f'\n{he.utc()} killed at the timeout {timeout} s\n')
                return 'timeout'


class InProcessChild:
    """The same stages in this process (tests and controls: one process under one spy)."""
    kind = 'in_process'

    def __call__(self, stage, ctx):
        try:
            return main([stage, '--context', str(context_path(ctx))])
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else 1


def make_launcher(ctx):
    cache = Path(ctx['paths']['cache'])
    kind = ctx['launcher']['kind']
    if kind == 'unit' and hold.is_lpba(ctx):                            # LP-B (C26.2-5): HE's nested unit of an LPBA pair is tag-lp-he-c-<pair> (tag-lp-feas-he-c-<pair>), never tag-he-c-*
        return fc.UnitLauncher(cache, unit_prefix=hold._lpba().unit_prefix('he-c', hold.pair_set_of(ctx)))
    if kind == 'unit':
        return fc.UnitLauncher(cache)
    if kind == 'direct':
        return fc.DirectLauncher(cache)
    raise SystemExit(f'no launcher {kind}')


def certifier(ctx):
    """certify(u, grid) of the settings' configuration C, CPU FP64, in this process; and the configuration."""
    enable_jit(ctx)
    from jd_curl.three_d.feasibility import certify_map
    config = he.certificate_config(ctx['settings'])
    return (lambda u, grid: certify_map(u, grid, config=config)), config


# ---- the fallbacks -----------------------------------------------------------------------------------------------------------------------------------
def configc_fallback(ctx, pin, *, launcher, certify, config, suffix=''):
    """Fallback one (C22.5-2, D75): (u, cert, record) or (None, None, record)."""
    s = ctx['settings']['configc']
    d = Path(ctx['out'])/f'fallback_configc{suffix}'
    d.mkdir(parents=False, exist_ok=False)
    rec = dict(kind='configC', run_dir=str(d/'run'), file=None, final_native_sha256=None, state_hash=None, certificate=None, configC_classification=None, firewall=None, prepare=None, launch=None, candidates=[], error=None,
               rule='D75: classes ok and handover_only; final_native.npz, else the certified native seed; certified again here, in a process that reads no label')
    try:
        rec['firewall'] = fc.derive_pair(pin, d/'derived', pair=ctx['pair'])
        prepared = fc.prepare(d/'derived', d/'run', pair=ctx['pair'], scratch=d, frozen=ctx['paths']['frozen'], opts=tuple(s['opts']), stage_label=s['stage_label'])
        rec['prepare'] = {k: prepared[k] for k in ('slot', 'frozen_files', 'seconds')}
        admission = Path(ctx['out'])/'admission.json'
        he.write_json(d/'run'/'slot.json', dict(stage_dir=ctx['stage_dir'], pair=ctx['pair'], slot=prepared['slot'], slot_is_not_the_stage=True, admission_sha256=core.sha_file(admission) if admission.is_file() else None,
                                                note='protocol.json pair_dataset and pair_source, and prepared_check.json, name the slot of a private copy of the frozen tree (firewall_c.SLOT_STAGE), not this stage; '
                                                     'the admission is load_pair\'s and the sealed lists\' (revision 1, review minor 10)'))
        gate, problems = fc.firewall_check(pin, d/'derived', d/'run', pair=ctx['pair'])
        rec['firewall']['gate'] = gate
        if problems:
            raise RuntimeError(f'the firewall gate: {problems}')
        t0 = fc.uptime()
        tick = perf_counter()
        rec['launch'] = dict(launcher(d/'run', he.short_name(ctx['pair'])), seconds=None)
        rec['launch']['seconds'] = perf_counter()-tick
        dxg, oom = fc.dmesg_counts(t0)
        rec['launch'].update(dxg_lines=dxg, oom_lines=oom, dmesg_read=dxg is not None)
        rec['configC_classification'] = fc.classify(d/'run', dxg=dxg or 0, oom=oom or 0, watchdog=s['watchdog_seconds'])
        for name, path, crec in fc.candidates(d/'run', rec['configC_classification']):
            row = dict(file=name, path=str(path), **{k: crec.get(k) for k in ('recorded_sha256', 'file_sha256', 'role')})
            try:
                u = fc.load_candidate(path, crec, pin.grid)
                problem = he.field_problem(u, pin.grid)         # revision 1, review minor 4: a candidate that cannot be an output is passed over
                if problem:
                    raise ValueError(problem)
                digest = he.array_sha256(u)
                doc = he.certificate_document(certify(u, pin.grid), config, process=f'the HE supervisor (pid {os.getpid()}; reads no label): configuration C field certified again', array_sha256_of_u=digest)
                row.update(state_hash=digest, passed=he.passes(doc) and doc['state_hash'] == digest, status=doc['status'], min_jd_lower_bound=doc['min_jd_lower_bound'])
            except (OSError, ValueError, KeyError) as exc:
                row.update(passed=False, error=f'{type(exc).__name__}: {exc}')
            rec['candidates'].append(row)
            if row['passed']:
                rec.update(file=name, final_native_sha256=crec['file_sha256'], state_hash=digest, certificate=doc)
                he.write_json(d/'fallback.json', rec)
                return u, doc, rec
        rec['error'] = 'no certified state of configuration C' if rec['configC_classification'].get('class') in fc.CANDIDATE_CLASSES else f"class {rec['configC_classification'].get('class')}: no candidate (D75)"
    except Exception as exc:                            # the fallback is unavailable; the evidence stays; the identity follows
        rec['error'] = f'{type(exc).__name__}: {exc}'[:2000]
        rec['traceback'] = traceback.format_exc()[-3000:]
    he.write_json(d/'fallback.json', rec)
    return None, None, rec


class IdentityNotCertified(RuntimeError):
    """C22.5-6 (a): fallback two (u = 0) did not pass the certificate: an implementation error, no output (HB: run_pair writes line.json hx_implementation_error and stops the queue)."""


def identity_fallback(ctx, pin, *, certify, config, suffix=''):
    """Fallback two (D75): u = 0, J = 1; certified here like any output."""
    d = Path(ctx['out'])/f'fallback_identity{suffix}'
    d.mkdir(parents=False, exist_ok=False)
    u = np.zeros(pin.grid.shape+(3,))
    digest = he.array_sha256(u)
    doc = he.certificate_document(certify(u, pin.grid), config, process=f'the HE supervisor (pid {os.getpid()}; reads no label): the identity', array_sha256_of_u=digest)
    rec = dict(kind='identity', run_dir=None, file=None, final_native_sha256=None, state_hash=digest, certificate=doc, configC_classification=None, firewall=None)
    he.write_json(d/'fallback.json', rec)
    if not (he.passes(doc) and doc['state_hash'] == digest):
        raise IdentityNotCertified('the identity did not pass the certificate: an implementation error, no output')
    return u, doc, rec


def fallback_chain(ctx, pin, *, start, launcher, certify, config, suffix=''):
    """(u, cert, fallback record, line status): configuration C, then the identity."""
    before = []
    if start == 'configC':
        u, doc, rec = configc_fallback(ctx, pin, launcher=launcher, certify=certify, config=config, suffix=suffix)
        if u is not None:
            return u, doc, rec, he.FALLBACK_C
        before.append(rec)
    u, doc, rec = identity_fallback(ctx, pin, certify=certify, config=config, suffix=suffix)
    rec['before'] = before
    return u, doc, rec, he.FALLBACK_ID


# ---- the supervisor ------------------------------------------------------------------------------------------------------------------------------
def check_identity(ctx, out, name):
    """The identity checks of a process, recorded as <out>/<name>; (record, problems).  Not checked only in a marked test context (make_context; the command line refuses an unmarked one)."""
    if ctx.get('check_identity') is not True:
        if ctx.get('test_context') != TEST_CONTEXT:
            record, problems = dict(checked=False), ['the identity is not checked and the context is not marked as a test context']
        else:
            record, problems = dict(checked=False, test_context=TEST_CONTEXT, note='not checked: a control or test context (check_identity false)'), []
    else:
        record, problems = he.identity(core.ROOT, frozen=ctx['paths']['frozen'])
        head = (record.get('repo') or {}).get('head')
        if hold.pair_set_of(ctx) == 'holdout':                          # HB (decision_log 2026-11-09, df8541d): the code directories the pinned commit's byte for byte; a later commit may only add
            pinned, pinned_problems = hold.hc.pinned_code_problems(ctx.get('code_head'), repo=core.ROOT)     # authorization, record and decision files (a committed authorization moves HEAD)
            problems = list(problems)+pinned_problems
            record = dict(record, pinned_code=pinned)
        elif hold.pair_set_of(ctx) == 'lpba40_test':                    # LP-B (C26.6-2, L49; C26.9-4): rule one with the LPBA40 constants (lpba_arms.pinned_code_problems; LP-C2's, fail closed until then)
            pinned, pinned_problems = hold._lpba().pinned_code_problems(ctx.get('code_head'), repo=core.ROOT)
            problems = list(problems)+list(pinned_problems)
            if not pinned_problems and pinned.get('pinned') != ctx.get('code_head'):
                problems.append(f"P_L {pinned.get('pinned')} is not the context's pinned commit {ctx.get('code_head')}")
            record = dict(record, pinned_code=pinned)
        elif not ctx.get('code_head') or head != ctx.get('code_head'):          # revision 2, review new minor 1: one commit for every process of the pair (LP-B: lpba40_feas too, C26.9-1)
            problems = list(problems)+[f"HEAD {head} is not the commit pinned when the pair's context was made ({ctx.get('code_head')})"]
        record = dict(record, pinned_head=ctx.get('code_head'))
    he.write_json(Path(out)/name, dict(record, problems=problems))
    return record, problems


def numbered(out, prefix):
    """<prefix>_<k>.json, k the number of such records already there: written once per call."""
    return f"{prefix}_{len(list(Path(out).glob(prefix+'_[0-9]*.json')))}.json"


def line_base(ctx, pin=None):
    return dict(schema=he.SCHEMA.format('line'), method=he.METHOD, pair=ctx['pair'], stage_dir=ctx['stage_dir'], settings_sha256=ctx['settings_sha256'], settings_frozen=ctx['settings_frozen'],
                check_identity=ctx.get('check_identity'), test_context=ctx.get('test_context'), code_head=ctx.get('code_head'),
                grid=None if pin is None else he.grid_record(pin.grid), wording='the output is the certified final state (认证终态), not a formal state; no certified descent; not F_B; no convergence',
                report='three parts (C22.5-4): the pairs of this line, the fallbacks by kind, all pairs (the method-level mean, with fallbacks); seg35 and seg4 both')


def supervisor_unit(ctx):
    """The unit the settings name for the supervisor (he_core.SUPERVISOR_UNIT, frozen settings `units`)."""
    return (ctx['settings'].get('units') or {}).get('supervisor') or {}


def unit_of(ctx, start, probe=None, expected=None):
    """line.json's `unit`: the start probe, a probe now, against the settings' unit (H2; C22.B4-6 (d))."""
    return he.unit_record(start, he.safe_probe('end', probe), supervisor_unit(ctx) if expected is None else expected)


def run_pair(ctx, *, child=None, launcher=None, certification=None, unseal_token=None, probe=None, _spied=False):
    """The supervisor: see the module docstring.  Returns 0 whatever the line status; 2 a refused start; 8 a failed check, no line.json: the code or the frozen tree (identity*.json; at the end
    line_not_written.json), or, HE2, the unit's guard (unit_refused.json: a systemd property or a kernel limit of this process's unit read and not the settings' units.supervisor; a value not read does
    not refuse; HE2 diff review new minor 3); 9 (he_core.STOP_QUEUE_EXIT) after an implementation error of C's tissue preparation (D86 (2)), its line written.  On 8 a queue reads which of
    unit_refused.json and identity*.json is there.  probe: unit_probe.probe's stand-in (tests).
    LP-B (review I3, C26.8-5): an LPBA context runs under the allow-list spy (lpba_spied: opened_files_supervisor_<k>.json); a child's spy breach (its spy_breach_<k>.json) or this process's
    stops the pair: line.json hx_implementation_error, exit he_core.STOP_QUEUE_EXIT, the queue stops."""
    if hold.is_lpba(ctx) and not _spied:
        return lpba_spied(ctx, 'supervisor', Path(ctx['out']), lambda: run_pair(ctx, child=child, launcher=launcher, certification=certification, unseal_token=unseal_token, probe=probe,
                                                                                 _spied=True), on_breach=lambda record: supervisor_breach(ctx, record, t0=perf_counter()))
    t0 = perf_counter()
    out = Path(ctx['out'])
    problem = function_context_problem(ctx) or holdout_context_problem(ctx)     # before anything is written (HB: a hold-out context as make_holdout_context makes it)
    if problem:
        print(f'refused: {problem}', file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=False)                             # never rerun
    he.write_json(context_path(ctx), ctx)
    he.write_json(out/'started.json', dict(utc=he.utc(), argv=sys.argv, pid=os.getpid(), child=getattr(child, 'kind', None), launcher=getattr(launcher, 'kind', None)))
    unit_start = he.safe_probe('start', probe)                          # H2: this process's unit as the machine reports it, before any child
    he.write_json(out/'unit_start.json', unit_start)
    _, problems = check_identity(ctx, out, 'identity.json')
    if problems:
        print(f'refused at the start: {problems}', file=sys.stderr)
        return 8
    unit_problems = he.unit_start_problems(unit_start, supervisor_unit(ctx))     # HE2 review minor 8: a unit whose systemd or kernel readings contradict the settings' unit (OOMPolicy, limits) does not
    if unit_problems:                                                             # start the method; a value not read (读不到) does not stop it
        he.write_json(out/'unit_refused.json', dict(problems=unit_problems, unit=he.unit_record(unit_start, unit_start, supervisor_unit(ctx)), utc=he.utc(),
                                                  rule='H2: the supervisor runs only in a unit whose readable properties are the settings\' units.supervisor (he_core.unit_start_problems)'))
        print(f'refused at the start: the unit {unit_start.get("unit")}: {unit_problems}', file=sys.stderr)
        return 8
    anchor_problems, anchor = he.segmenter_start_problems(ctx['settings'], registry=registry_of(ctx))     # C22.9-4 (xv), before the pair is opened
    he.write_json(out/'segmenter_anchor.json', dict(anchor, problems=anchor_problems))
    if anchor_problems:
        he.write_json(out/'line.json', dict(line_base(ctx), line_status=he.REJECTED, rejection=dict(stage='segmenter_anchor', kind='segmenter_anchor', reason=f'{anchor_problems}'[:2000]), failure=None, fuse=None,
                                            fallback=None, output=None, c=None, unit=unit_of(ctx, unit_start, probe), note='no output; the pair was not opened (D74 (4))', seconds_total=perf_counter()-t0, ended_utc=he.utc()))
        return 0
    fz = core.frozen_api()
    child = SubprocessChild() if child is None else child
    try:
        pin, second_lock = admitted_images(ctx, unseal_token=unseal_token)       # the sealed lists on the names and the manifest, then the allowlist: before the archive is opened or hashed
    except admission_refusals() as exc:
        he.write_json(out/'line.json', dict(line_base(ctx), line_status=he.REJECTED, rejection=dict(stage='admission', reason=f'{type(exc).__name__}: {exc}'[:2000]), failure=None, fuse=None, fallback=None,
                                            output=None, c=None, unit=unit_of(ctx, unit_start, probe), note='no output; nothing of the pair was read beyond the manifest fields of the firewall view (D74 (4))',
                                            seconds_total=perf_counter()-t0, ended_utc=he.utc()))
        return 0
    he.write_json(out/'admission.json', dict(keys_read=list(pin.provenance['keys_read']), label_firewall=pin.provenance['label_firewall'], subject_guard=pin.provenance['subject_guard'], second_lock=second_lock,
                                            pair_sha256=pin.provenance['pair_sha256'], manifest_sha256=pin.provenance['manifest_sha256'], grid=pin.grid_metadata))
    rc_t88 = child('t88', ctx)
    if hold.is_lpba(ctx):
        child_breach(out/'t88')                                         # LP-B (review I3): the child's spy stopped a read: the pair stops here, no fallback
    t88 = he.read_json(out/'t88/run.json')
    kind, info = he.t88_decision(t88, rc_t88)
    repair_record, rc_repair = None, None
    if kind == 'go':
        rc_repair = child('repair', ctx)
        if hold.is_lpba(ctx):
            child_breach(out/'repair')
        repair_record = he.read_json(out/'repair/repair.json')
        kind, info = he.repair_decision(repair_record, certified_consistency(out/'repair', repair_record, pin.grid), rc_repair)
    c_line = None if t88 is None else t88.get('c_summary')                 # C22.B4-5: the three passes' u_sha256, the ratios used, the last estimate
    if kind == 'rejected':
        stop = bool(info.get('stop_queue'))
        he.write_json(out/'line.json', dict(line_base(ctx, pin), line_status=he.REJECTED, rejection=info, failure=None, fuse=None, fallback=None, output=None, c=c_line,
                                            t88=None if t88 is None else dict(status=t88.get('status'), returncode=rc_t88, tissue_failure=t88.get('tissue_failure')),
                                            repair=None if repair_record is None else dict(status=repair_record.get('status'), returncode=rc_repair), stop_queue=stop, unit=unit_of(ctx, unit_start, probe),
                                            note=('no output; an implementation error (D86 (2)): to the lead, the queue stops, nothing is rerun' if stop else 'no output (D74 (4))'), seconds_total=perf_counter()-t0, ended_utc=he.utc()))
        return he.STOP_QUEUE_EXIT if stop else 0
    try:
        certify, config = certifier(ctx) if certification is None else certification
        launcher = make_launcher(ctx) if launcher is None else launcher
        if kind == 'hx':
            with np.load(out/'repair/certified.npz', allow_pickle=False) as z:
                u = np.ascontiguousarray(np.array(z['u']))
            cert, fallback, status, failure_record = he.read_json(out/'repair/cert.json'), None, he.HX, None
        else:
            failure_record = info
            u, cert, fallback, status = fallback_chain(ctx, pin, start='configC', launcher=launcher, certify=certify, config=config)
        line = dict(line_base(ctx, pin), line_status=status, failure=failure_record, fuse=he.fuse_record(repair_record, status), fallback=fallback, rejection=None, third_certificate=None,
                    review_certificate_in=review_certificate_in(ctx), superseded=[], t88=dict(status=None if t88 is None else t88.get('status'), reason=None if t88 is None else t88.get('reason'), returncode=rc_t88, seconds_task=None if t88 is None else t88.get('seconds_task')),
                    repair=None if repair_record is None else dict(status=repair_record.get('status'), reason=repair_record.get('reason'), passed=repair_record.get('passed'), wall_hit=repair_record.get('wall_hit'),
                                                                     rounds=len(repair_record.get('rounds') or []), seconds_repair=repair_record.get('seconds_repair'), seconds_warmup=repair_record.get('seconds_warmup'), returncode=rc_repair),
                    decision_certificate=dict(process=cert.get('process'), state_hash=cert.get('state_hash'), passed=he.passes(cert)), c=c_line,
                    needs_lead_rejudgment=bool(failure_record and failure_record.get('needs_lead_rejudgment')), seconds_total=perf_counter()-t0)
        _, problems = check_identity(ctx, out, 'identity_end.json')
        if problems:
            he.write_json(out/'line_not_written.json', dict(line, reason=f'the frozen tree or the code changed during the run: {problems}'))
            return 8
        line['unit'] = unit_of(ctx, unit_start, probe)                  # the end probe: the cgroup's memory.peak covers the children (they ran in this unit)
        he.write_output(out/he.OUTPUT_DIR, u, pin.grid, cert, line)
    except Exception as exc:                                            # noqa: BLE001 - HB, C22.5-6: no output; line.json says why; the queue stops (exit 9)
        he.write_json(out/'line.json', implementation_error_line(ctx, pin, exc, unit=unit_of(ctx, unit_start, probe), t88=t88, rc_t88=rc_t88, repair_record=repair_record, rc_repair=rc_repair,
                                                                 failure_record=info if kind != 'hx' else None, c_line=c_line, seconds_total=perf_counter()-t0))
        return he.STOP_QUEUE_EXIT
    return 0


def review_certificate_in(ctx):
    """line.json's review_certificate_in: the frozen setting (HB, C22.H0-3, C22.H1-7: the step-5 review record); a context whose settings predate it (HE2's) gets HE2's literal words."""
    return ctx['settings'].get('review_certificate_in') or 'evaluation.json (the evaluation unit writes the review certificate there, after the output is fixed)'


def evidence_files(out, patterns=('fallback_*/fallback.json', 'output', 'output_failed_*', 'output_superseded_*')):
    """[{path, sha256 (files), kind}] of the fallback records and the output directories under `out` (C22.5-6: every piece of evidence by its sha256)."""
    rows = []
    for pattern in patterns:
        for p in sorted(Path(out).glob(pattern)):
            if p.is_file():
                rows.append(dict(path=str(p), sha256=core.sha_file(p), kind='file'))
            elif p.is_dir():
                rows.append(dict(path=str(p), kind='directory', files={str(f.relative_to(p)): core.sha_file(f) for f in sorted(p.rglob('*')) if f.is_file()}))
    return rows


def implementation_error_line(ctx, pin, exc, *, unit, t88, rc_t88, repair_record, rc_repair, failure_record, c_line, seconds_total):
    """HB (C22 卷首修订 8b, C22.5-6): the line.json of a pair whose supervisor raised after the decision: line_status hx_implementation_error, no output, the reason ((a): the identity did not pass its
    certificate; (b): any other implementation error that leaves the pair without an output), the evidence (fallback records with their certificates, any half-written output) by sha256; stop_queue."""
    identity = isinstance(exc, IdentityNotCertified)
    out = Path(ctx['out'])
    fallback_records = [he.read_json(p) for p in sorted(out.glob('fallback_*/fallback.json'))]
    return dict(line_base(ctx, pin), line_status=he.IMPLEMENTATION, rejection=None, failure=failure_record, fuse=None, fallback=None, output=None, c=c_line,
                implementation_error=dict(kind='a' if identity else 'b', rule='C22.5-6 (a): the identity did not pass its certificate' if identity else
                                          'C22.5-6 (b): an implementation error the preregistration does not name, leaving the pair without an output',
                                          exception=f'{type(exc).__name__}: {exc}'[:2000], traceback=traceback.format_exc()[-4000:],
                                          exception_class=type(exc).__name__, possibly_infrastructure=isinstance(exc, (OSError, MemoryError)) or None,    # HB review m4: noted, the class unchanged
                                          note=('an OSError or MemoryError: possibly infrastructure (a full disk, the memory), the lead decides; recorded as C22.5-6 (b) here'
                                                if isinstance(exc, (OSError, MemoryError)) else None),
                                          certificates=[dict(kind=(r or {}).get('kind'), state_hash=(r or {}).get('state_hash'), certificate=(r or {}).get('certificate')) for r in fallback_records],
                                          evidence=evidence_files(out)),
                t88=None if t88 is None else dict(status=t88.get('status'), returncode=rc_t88), repair=None if repair_record is None else dict(status=repair_record.get('status'), returncode=rc_repair),
                stop_queue=True, unit=unit, note='no output; hx_implementation_error (C22.5-6): the queue stops, to the lead and BZ, nothing is rerun without BZ\'s authorization',
                seconds_total=seconds_total, ended_utc=he.utc())


# ---- LP-B: the allow-list spy of an LPBA context's processes (review I3; LPBA40 prereg §8.5 item 7; C26.5-1, C26.8-5) ---------------------------------------
class LpbaSpyBreach(RuntimeError):
    """A child of an LPBA pair (t88, repair) reported that its spy stopped a read (its spy_breach_<k>.json): the pair stops (C26.8-5)."""


def lpba_policy_of(ctx, stage):
    """lpba_arms.ArmPolicy of one process of an LPBA pair (the allow list of experiments/lpba/实现说明_LP-B.md): its own directories <out> and the context's cache (read and written); the pair's
    pair.npz and manifest.json (read); for the t88 child also the two subjects' HI prediction directories, their Atropos output directories and the registered members (read only); the code trees
    (this checkout, bayesreg's, the frozen tree); everything else under the results root and the data roots refused, the libraries not watched."""
    la = hold._lpba()
    from bayesreg.io import sealed
    p = ctx['paths']
    pair_dir = Path(ctx['stage_dir'])/ctx['pair']
    read_inside = []
    if stage == 't88':
        fixed, moving = la.split(ctx['pair'])
        read_inside = [Path(p['pred_root'])/fixed, Path(p['pred_root'])/moving, Path(p['atropos_root'])/fixed, Path(p['atropos_root'])/moving, Path(p['models_root'])]
    return la.ArmPolicy(name=f"{hold.pair_set_of(ctx)} {stage} {ctx['pair']}", read_exact=[pair_dir/'pair.npz', pair_dir/'manifest.json'], read_inside=read_inside,
                        write_inside=[Path(ctx['out']), Path(p['cache'])], results=la.results_roots()+[Path(he.RESULTS)], code_roots=[core.ROOT, Path(sealed.REPO)],
                        frozen_roots=[Path(p['frozen'])])      # diff review I3-a: the frozen tree's results/, docs/, figures/ forbidden


def lpba_spied(ctx, stage, record_dir, body, *, on_breach=None):
    """body() under the allow-list spy (lpba_arms.arm_spying: installed only when the guard's cache of the sealed lists is empty, else refused: exit 2).  After it: the spy's record
    <record_dir>/opened_files_<stage>_<k>.json (when the directory is there).  A breach -- the spy's exception, a child's reported breach (LpbaSpyBreach), or a breach some library swallowed --
    writes <record_dir>/spy_breach_<k>.json and returns on_breach(record) (the supervisor's) or lpba_arms.SPY_BREACH_EXIT."""
    la = hold._lpba()
    breach = None
    try:
        with la.arm_spying(lpba_policy_of(ctx, stage)) as spy:
            try:
                rc = body()
            except (la.lc.hp.Breach, LpbaSpyBreach) as exc:
                breach, rc = f'{type(exc).__name__}: {exc}', la.SPY_BREACH_EXIT
            summary = spy.summary()
    except la.lc.Refused as exc:
        print(f'refused: {exc}', file=sys.stderr)
        return 2
    record_dir = Path(record_dir)
    if record_dir.is_dir():
        he.write_json(record_dir/numbered(record_dir, f'opened_files_{stage}'), summary)
    if breach is None and summary['passed']:
        return rc
    record = dict(stage=stage, pair=ctx['pair'], pair_set=hold.pair_set_of(ctx), exception=breach, breaches=summary['breaches'], utc=he.utc(),
                  rule='LP-B review I3, C26.8-5: a read outside the allow list stops the pair (a tool error first); nothing is rerun')
    record_dir.mkdir(parents=True, exist_ok=True)
    he.write_json(record_dir/numbered(record_dir, 'spy_breach'), record)
    return on_breach(record) if on_breach is not None else la.SPY_BREACH_EXIT


def child_breach(directory):
    """LP-B: raise LpbaSpyBreach when a child's directory holds its spy_breach_<k>.json."""
    found = sorted(Path(directory).glob('spy_breach_*.json'))
    if found:
        raise LpbaSpyBreach(f'{found[-1]}: {(he.read_json(found[-1]) or {}).get("exception")}')


def supervisor_breach(ctx, record, *, t0):
    """LP-B: the supervisor of an LPBA pair after a breach (its own or a child's): line.json hx_implementation_error (no output; when no line.json is there yet), exit STOP_QUEUE_EXIT."""
    out = Path(ctx['out'])
    if out.is_dir() and not (out/'line.json').exists() and not (out/he.OUTPUT_DIR/'line.json').exists():
        he.write_json(out/'line.json', dict(line_base(ctx), line_status=he.IMPLEMENTATION, rejection=None, failure=None, fuse=None, fallback=None, output=None, c=None,
                                            implementation_error=dict(kind='b', rule='LP-B review I3, C26.8-5: the allow-list spy stopped a read (a tool error first; to the lead and BZ)', spy_breach=record,
                                                                      exception_class='SpyBreach'),
                                            stop_queue=True, note='no output; the spy stopped a read: the queue stops, nothing is rerun', seconds_total=perf_counter()-t0, ended_utc=he.utc()))
    return he.STOP_QUEUE_EXIT


# ---- the third certificate ----------------------------------------------------------------------------------------------------------------------------
THIRD_UNIT_PREFIX = 'tag-he-c3'
THIRD_LAUNCHER_RULE = ('H3 (HE2): after a failed third certificate, configuration C runs in a unit of its own (firewall_c.UnitLauncher: D0\'s 12G, swap 4G, OOMPolicy=continue, the unit probe), whatever the '
                       'pair\'s launcher: the third certification runs in the evaluation unit, whose limit (8G) is set for a certificate and the metrics, not for the frozen run')


def third_launcher(ctx):
    """H3: the launcher of the fallback after a failed third certificate: a unit of its own (settings units.third_certificate_fallback = unit)."""
    kind = (ctx['settings'].get('units') or {}).get('third_certificate_fallback', he.THIRD_FALLBACK_LAUNCHER)
    if kind != 'unit':
        raise SystemExit(f'units.third_certificate_fallback {kind!r}: only unit is defined (H3)')
    if hold.is_lpba(ctx):                                               # LP-B (C26.2-5): tag-lp-he-c3-<pair> (tag-lp-feas-he-c3-<pair>)
        return fc.UnitLauncher(Path(ctx['paths']['cache']), unit_prefix=hold._lpba().unit_prefix('he-c3', hold.pair_set_of(ctx)))
    return fc.UnitLauncher(Path(ctx['paths']['cache']), unit_prefix=THIRD_UNIT_PREFIX)


def third_cert_main(ctx, *, launcher=None, certification=None, _spied=False):
    """C22.3-9 (d), the lead's mechanical rule: the output array certified again (the same configuration) in this process, which reads no label; pass: the output stands; fail: the next fallback.  No metric is read.
    LP-B (review I3): an LPBA context runs under the allow-list spy (lpba_spied: opened_files_third_<k>.json in the cell; a breach exits lpba_arms.SPY_BREACH_EXIT, which the review unit reads as its
    child's spy stopping a read)."""
    if hold.is_lpba(ctx) and not _spied:
        return lpba_spied(ctx, 'third', Path(ctx['out']), lambda: third_cert_main(ctx, launcher=launcher, certification=certification, _spied=True))
    out = Path(ctx['out'])
    o = out/he.OUTPUT_DIR
    problem = function_context_problem(ctx) or holdout_context_problem(ctx)
    if problem:
        print(f'refused: {problem}', file=sys.stderr)
        return 2
    _, problems = check_identity(ctx, out, numbered(out, 'identity_third'))
    if problems:
        return 8
    core.frozen_api()
    pin, second_lock = admitted_images(ctx)                             # both locks again
    u, line, cert = he.read_output(o, pin.grid)
    certify, config = certifier(ctx) if certification is None else certification
    digest = he.array_sha256(u)
    doc = he.certificate_document(certify(u, pin.grid), config, process=f'the third certification (run_hx.py third-cert, pid {os.getpid()}; reads no label)', array_sha256_of_u=digest)
    record = dict(schema=he.SCHEMA.format('third_certificate'), rule='C22.3-9 (d): the same array, the same configuration, a process that reads no label; pass -> the output stands, fail -> the next fallback; no metric',
                  output_state_hash=digest, line_status=line['line_status'], certificate=doc, passed=he.passes(doc) and doc['state_hash'] == digest, keys_read=list(pin.provenance['keys_read']), pid=os.getpid(), utc=he.utc())
    if hold.is_lpba(ctx):                                               # LP-B (C26.6-2): the LPBA admission of this process (the hold-out's record unchanged)
        record['admission_second_lock'] = second_lock
    he.write_json(o/'third_certificate.json', record)
    if record['passed']:
        return 0
    start = he.next_fallback(line['line_status'])
    if start is None:                                                   # revision 1, review minor 1: no fallback after this one: the output is withdrawn, kept whole, and nothing is evaluated
        k = len(list(out.glob('output_failed_*')))
        failed = out/f'output_failed_{k}'
        o.rename(failed)
        he.write_json(out/f'output_failed_{k}.json', dict(reason=f"the third certificate of a {line['line_status']} output failed and no fallback follows it: no output (hard rule 1)", third_certificate=record, kept=str(failed), utc=he.utc()))
        return 8
    launcher = third_launcher(ctx) if launcher is None else launcher    # H3: never the pair's direct launcher here (this process is the evaluation unit's child); HE2 review minor 4: chosen
    k = len(list(out.glob('output_superseded_*')))                      # before the output is renamed, so that a refused setting leaves the output where it was
    superseded = out/f'output_superseded_{k}'
    o.rename(superseded)                                                # kept whole
    u, cert, fallback, status = fallback_chain(ctx, pin, start=start, launcher=launcher, certify=certify, config=config, suffix=f'_after_third_{k}')
    new = dict({k_: v for k_, v in line.items() if k_ not in ('output', 'ended_utc')}, line_status=status, fallback=fallback, third_certificate=record, superseded=list(line.get('superseded') or [])+[str(superseded)],
               failure=dict(line.get('failure') or he.failure('repair_not_certified', 'the third certificate (C22.3-9 (d)) did not pass'), third_certificate_failed=True),
               decision_certificate=dict(process=cert.get('process'), state_hash=cert.get('state_hash'), passed=he.passes(cert)),
               third_fallback_launcher=dict(kind=getattr(launcher, 'kind', None), unit_prefix=getattr(launcher, 'unit_prefix', None), rule=THIRD_LAUNCHER_RULE))
    _, problems = check_identity(ctx, out, numbered(out, 'identity_third_end'))       # revision 1, review important 2: before the new output, as the supervisor and the evaluation
    if problems:
        he.write_json(out/numbered(out, 'line_not_written'), dict(new, reason=f'the frozen tree or the code changed during the third certification: {problems}'))
        return 8
    he.write_output(o, u, pin.grid, cert, new)
    return 0


# ---- the evaluation unit ------------------------------------------------------------------------------------------------------------------------------
class ThirdCertChild:
    """The third certification in its own process (it reads no label; the evaluation process has read none yet either)."""

    def __call__(self, ctx):
        argv = fc.check_launch([PY, '-B', str(SCRIPT), 'third-cert', '--context', str(context_path(ctx))])
        with (Path(ctx['out'])/'third-cert.log').open('a') as f:
            return subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT).returncode


TRUTH_MARKER = 'evaluation_started.json'


def evaluate_main(ctx, *, certification=None, third=None, max_rounds=3, lead_release=None, probe=None):
    """Once per pair, after the output is fixed.  Order: the output and its files; the review certificate (CPU FP64, configuration C) compared field by field with cert.json, before any label is read; a mismatch
    without a passing third certificate of this array: the third certification first (a label-free child), then again; then an exclusive marker (evaluation_started.json); then the truth behind the bench's guard
    and the metrics from disp.npz.  A marker already there means the truth may have been read once: refused, unless the lead releases it (lead_release: his words and time), which is recorded and the old
    marker kept (revision 1, review minor 5)."""
    out = Path(ctx['out'])
    problem = function_context_problem(ctx)
    if problem:
        print(f'refused: {problem}', file=sys.stderr)
        return 2
    if hold.pair_set_of(ctx) != 'dev':                                  # HB, C22.H1-6 and prereg revision 1 item 3: a hold-out pair is evaluated by bayesreg.eval.holdout only; this loop does not run there
        print(f"refused: a {hold.pair_set_of(ctx)} context is never evaluated here (the review certificate is the step-5 review unit's, experiments/hb/hb_review.py; the truth bayesreg.eval.holdout's)", file=sys.stderr)
        return 2
    if (out/'evaluation.json').exists():
        print(f'{out}/evaluation.json exists: a pair is evaluated once', file=sys.stderr)
        return 2
    if (out/TRUTH_MARKER).exists() and not lead_release:
        print(f'{out}/{TRUTH_MARKER} exists: the truth of this pair may have been read once; only the lead releases a second read (--lead-release "<his words and time>")', file=sys.stderr)
        return 2
    unit_start = he.safe_probe('start', probe)                          # C22.B4-6 (d): the evaluation unit as the machine reports it (H3's fallback is in a unit of its own)
    _, problems = check_identity(ctx, out, numbered(out, 'identity_eval'))
    if problems:
        return 8
    core.frozen_api()
    from bayesreg.eval import fields, metrics
    from bayesreg.eval import truth as bench_truth
    line0 = he.read_json(out/'line.json')
    stage_dir = Path(ctx['stage_dir'])
    if line0 is not None and line0.get('line_status') in (he.REJECTED, he.IMPLEMENTATION):
        he.write_json(out/'evaluation.json', dict(schema=he.SCHEMA.format('evaluation'), pair=ctx['pair'], line_status=line0.get('line_status'), metrics=None, note='no output: nothing evaluated, no label read',
                                                  check_identity=ctx.get('check_identity'), test_context=ctx.get('test_context'), code_head=ctx.get('code_head'), utc=he.utc()))
        return 0
    sealed = bench_truth.load_sealed_lists()
    pair_dir, grid, _, guard = bench_truth.admit_pair(stage_dir.parent, stage_dir.name, ctx['pair'], ctx['paths']['allowlist'], sealed)     # names, manifest, guard: no label yet
    certify, config = certifier(ctx) if certification is None else certification
    third = ThirdCertChild() if third is None else third
    o = out/he.OUTPUT_DIR
    rounds = []
    for _ in range(max_rounds):
        if not (o/'line.json').is_file():                               # a withdrawn output (output_failed_<k>): nothing to evaluate
            he.write_json(out/numbered(out, 'evaluation_not_written'), dict(rounds=rounds, reason=f'no output: {o} is not there (see output_failed_*.json)', utc=he.utc()))
            return 8
        u, line, cert = he.read_output(o, grid)
        d, disp_record = fields.read_disp(o/'disp.npz', grid.shape)
        roundtrip = fields.roundtrip_check(u, d, tuple(grid.spacing))
        digest = he.array_sha256(u)
        review = he.certificate_document(certify(u, grid), config, process=f'the evaluation unit (run_hx.py evaluate, pid {os.getpid()}), before any label was read', array_sha256_of_u=digest)
        same, diffs = he.same_certificate(review, cert)
        third_record = he.read_json(o/'third_certificate.json')
        resolved = third_record is not None and third_record.get('output_state_hash') == digest and third_record.get('passed') is True     # revision 1, review minor 1: a passing one
        rounds.append(dict(state_hash=digest, line_status=line['line_status'], consistent=same, differences=diffs, third_certificate=third_record))
        if same or resolved:
            break
        he.write_json(o/numbered(o, 'review_mismatch'), dict(review=review, decision=cert, differences=diffs, rule='C22.3-9 (d): the third certification decides, in a process that reads no label', utc=he.utc()))
        rc = third(ctx)
        rounds[-1]['third_returncode'] = rc
    else:
        he.write_json(out/numbered(out, 'evaluation_not_written'), dict(rounds=rounds, reason=f'the review certificate still differs after {max_rounds} rounds'))
        return 8
    if not roundtrip['passed']:
        he.write_json(out/numbered(out, 'evaluation_not_written'), dict(rounds=rounds, roundtrip=roundtrip, reason='disp.npz is not final.npz (C16 round trip)'))
        return 8
    if (out/TRUTH_MARKER).exists():                                     # only with the lead's release (checked at the start): kept under another name, the release recorded
        k = len(list(out.glob('evaluation_release_*.json')))
        (out/TRUTH_MARKER).rename(out/f'evaluation_started_{k}.json')
        he.write_json(out/f'evaluation_release_{k}.json', dict(release=lead_release, previous_marker=f'evaluation_started_{k}.json', utc=he.utc()))
    he.write_json(out/TRUTH_MARKER, dict(utc=he.utc(), pid=os.getpid(), state_hash=digest, line_status=line['line_status'], lead_release=lead_release,
                                         rule='written before the truth is read: a second read of this pair needs the lead\'s release (revision 1, review minor 5)'))
    scheme = bench_truth.scheme_for(guard, pair_dir)
    labels, truth_record = bench_truth.load_truth(pair_dir, scheme, grid, guard)          # the first and only read of a label in this line
    scalars, per_label, official = metrics.evaluate_field(d, labels, scheme, tuple(grid.spacing))
    del labels
    _, problems = check_identity(ctx, out, numbered(out, 'identity_eval_end'))
    if problems:
        he.write_json(out/numbered(out, 'evaluation_not_written'), dict(rounds=rounds, reason=f'the frozen tree or the code changed during the evaluation: {problems}'))
        return 8
    sha = {name: core.sha_file(core.ROOT/'bayesreg/eval'/name) for name in ('metrics.py', 'truth.py', 'fields.py')}
    he.write_json(out/'evaluation.json', dict(
        schema=he.SCHEMA.format('evaluation'), pair=ctx['pair'], stage_dir=ctx['stage_dir'], line_status=line['line_status'], fallback_kind=(line.get('fallback') or {}).get('kind'), metrics=scalars, per_label=per_label, official_row=official,
        review_certificate=dict(consistent=rounds[-1]['consistent'], differences=rounds[-1]['differences'], review=review, decision=dict(process=cert.get('process'), state_hash=cert.get('state_hash'), status=cert.get('status'),
                                min_jd_lower_bound=cert.get('min_jd_lower_bound')), third_certificate=rounds[-1]['third_certificate'], rounds=rounds),
        order='the review certificate of the output before any label was read; the truth once, after the output was fixed (C22.4-1, hard rule 6)', truth=truth_record, guard=guard, disp=disp_record, roundtrip=roundtrip,
        evaluator=dict(modules=sha, definitions='bayesreg.eval.metrics (the bench)'), truth_marker=TRUTH_MARKER, lead_release=lead_release, settings_sha256=ctx['settings_sha256'], settings_frozen=ctx['settings_frozen'],
        unit=unit_of(ctx, unit_start, probe, expected=(ctx['settings'].get('units') or {}).get('evaluation') or {}),
        check_identity=ctx.get('check_identity'), test_context=ctx.get('test_context'), code_head=ctx.get('code_head'), pid=os.getpid(), utc=he.utc()))
    return 0


# ---- the CLI -------------------------------------------------------------------------------------------------------------------------------------------
def out_problem(out, results=None):
    results = he.RESULTS if results is None else Path(results)
    out = Path(out)
    if out.exists():
        return f'{out} exists: a pair is never rerun'
    if results.resolve() not in out.resolve().parents:
        return f'{out}: the outputs go under {results}'
    return None


def dry_run(stage_dir, pair):
    """Admission (both locks) and identity only: no optimization, nothing written."""
    record, problems = he.identity(core.ROOT)
    print(f"identity: {'ok' if not problems else problems}")
    core.frozen_api()
    ctx = make_context(stage_dir, pair, he.RESULTS/'he_dry_run_never_written'/pair)
    try:
        pin, _ = admitted_images(ctx)
        print(f'admission: {pin.fixed_subject} <- {pin.moving_subject}, keys read {pin.provenance["keys_read"]}, grid {pin.grid_metadata}')
    except admission_refusals() as exc:
        print(f'admission refused: {type(exc).__name__}: {exc}')
        return 8
    print(f"settings: frozen {ctx['settings_frozen']}, sha256 {ctx['settings_sha256']}; method {he.NAME}; tissue_prior {ctx['settings'].get('tissue_prior')}")
    anchor_problems, anchor = he.segmenter_start_problems(ctx['settings'], registry=registry_of(ctx))
    print(f"segmenter anchor (C22.9-4 (xv)): {'ok' if not anchor_problems else anchor_problems}; registry {anchor['registry_rel']} sha256 {anchor['registry_sha256']}; models {anchor['model_segmenter_ids']}")
    p = ctx['paths']
    seen = {f'prediction {side}': (Path(p['pred_root'])/getattr(pin, side+'_subject')/'provenance.json').is_file() for side in ('fixed', 'moving')}
    seen.update({'atropos fixed (QC and C)': (Path(p['atropos_root'])/pin.fixed_subject/'provenance.json').is_file(), 'atropos moving (QC)': (Path(p['atropos_root'])/pin.moving_subject/'provenance.json').is_file()})
    print(f"sources (only whether provenance.json is there; nothing of them read): {seen}; pred_root {p['pred_root']}; atropos_root {p['atropos_root']}")
    return 0 if not problems and not anchor_problems else 8


def holdout_dry_run(pair):
    """HB: the hold-out admission of one pair (he_holdout.admitted_images: load_unseal, bind_unseal, check_subject, the pair list, the manifest, load_pair) and the identity; no optimization, nothing written.
    Without an unseal record (main today) the admission refuses at the unseal step, before anything of a subject is opened."""
    record, problems = he.identity(core.ROOT)
    print(f"identity: {'ok' if not problems else problems}")
    core.frozen_api()
    try:
        ctx = make_holdout_context(pair)
    except ValueError as exc:
        print(f'context refused: {exc}')
        return 8
    print(f"context: pair_set {ctx['pair_set']}, stage {ctx['stage_dir']}, out {ctx['out']} (not written); settings sha256 {ctx['settings_sha256']}, frozen {ctx['settings_frozen']}; method {he.NAME}; "
          f"tissue_prior {ctx['settings'].get('tissue_prior')}; pred_root {ctx['paths']['pred_root']}; atropos_root {ctx['paths']['atropos_root']}")
    try:
        pin, _ = admitted_images(ctx)
        print(f'admission: {pin.fixed_subject} <- {pin.moving_subject}, keys read {pin.provenance["keys_read"]}, grid {pin.grid_metadata}')
    except admission_refusals() as exc:
        print(f'admission refused: {type(exc).__name__}: {exc}')
        return 8
    return 0 if not problems else 8


def holdout_roots_problem(stage_dir, out):
    """HB: the development subcommand never writes into the hold-out's roots or reads its stage (a hold-out pair runs through holdout-pair only).  LP-B: nor the LPBA40 modes' roots (an LPBA pair runs
    through lpba-pair only)."""
    roots = [Path(he.RESULTS)/'holdout_B_v1', Path(hold.hc.HOLDOUT_ROOT), Path(hold.hc.BIG_ROOT)]
    for path in (Path(stage_dir), Path(out)):
        resolved = path.resolve()
        if any(resolved == r.resolve() or r.resolve() in resolved.parents for r in roots):
            return f'{path} is under a hold-out root {[str(r) for r in roots]}: a hold-out pair runs only through `run_hx.py holdout-pair` (HB)'
    lpba = [Path(he.RESULTS)/name for name in LPBA_ROOT_NAMES]
    for path in (Path(stage_dir), Path(out)):
        resolved = path.resolve()
        if any(resolved == r.resolve() or r.resolve() in resolved.parents for r in lpba):
            return f'{path} is under an LPBA40 root {[str(r) for r in lpba]}: an LPBA40 pair runs only through `run_hx.py lpba-pair` (LP-B)'
    return None


LPBA_ROOT_NAMES = ('lpba40_test_v1', 'lpba40_feas_v1')        # C26.2-3: the LPBA40 modes' roots under the results root (lpba_common.roots; a test compares)


def lpba_dry_run(pair_set, pair):
    """LP-B: the LPBA admission of one pair (he_holdout.lpba_admitted_images: lpba40_pre_manifest, the manifest's subjects, load_pair) and the identity; no optimization, nothing written.  The
    formal mode's rule one is LP-C2's (lpba_arms.pinned_code_problems fails closed until then): its dry run is refused at the context."""
    record, problems = he.identity(core.ROOT)
    print(f"identity: {'ok' if not problems else problems}")
    core.frozen_api()
    try:
        ctx = make_lpba_context(pair_set, pair)
    except ValueError as exc:
        print(f'context refused: {exc}')
        return 8
    print(f"context: pair_set {ctx['pair_set']}, stage {ctx['stage_dir']}, out {ctx['out']} (not written); settings sha256 {ctx['settings_sha256']}, frozen {ctx['settings_frozen']}; method {he.NAME}; "
          f"tissue_prior {ctx['settings'].get('tissue_prior')}; pred_root {ctx['paths']['pred_root']}; atropos_root {ctx['paths']['atropos_root']}; allowlist {ctx['paths']['allowlist']}")
    try:
        pin, _ = hold.lpba_admitted_images(ctx)                        # the LPBA branch of admitted_images (C26.6-2), called by its own name
        print(f'admission: {pin.fixed_subject} <- {pin.moving_subject}, keys read {pin.provenance["keys_read"]}, grid {pin.grid_metadata}')
    except admission_refusals() as exc:
        print(f'admission refused: {type(exc).__name__}: {exc}')
        return 8
    return 0 if not problems else 8


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('pair')
    p.add_argument('--stage-dir', required=True)
    p.add_argument('--pair', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--launcher', choices=('unit', 'direct'), default='unit')
    p.add_argument('--unseal-token', default=None)
    for name in ('t88', 'repair', 'third-cert', 'evaluate'):
        q = sub.add_parser(name)
        q.add_argument('--context', required=True)
        if name == 'evaluate':
            q.add_argument('--lead-release', default=None, help="the lead's words and time that release a second read of the truth of this pair (evaluation_started.json is there); recorded")
    d = sub.add_parser('dry-run')
    d.add_argument('--stage-dir', required=True)
    d.add_argument('--pair', required=True)
    h = sub.add_parser('holdout-pair', help='HB: the method on one pair of the B-line hold-out (the stage, the output and the paths fixed: make_holdout_context); the batch queue (experiments/hb) starts it')
    h.add_argument('--pair', required=True)
    h.add_argument('--device', default='cuda')
    h.add_argument('--pinned', default=None, help='the batch\'s pinned commit (C22.H0-4): only compared with P1 of runs/holdout_B_v1/pinned_code.json (never HEAD)')
    hd = sub.add_parser('holdout-dry-run', help='HB: the hold-out admission and the identity of one pair; nothing written')
    hd.add_argument('--pair', required=True)
    lp = sub.add_parser('lpba-pair', help='LP-B: the method on one LPBA40 pair of a mode (the stage, the output and the paths fixed: make_lpba_context); the LPBA method queue (experiments/hb/hb_method.py '
                                          '--pair-set) starts it')
    lp.add_argument('--pair-set', required=True, choices=hold.LPBA_PAIR_SETS)
    lp.add_argument('--pair', required=True)
    lp.add_argument('--device', default='cuda')
    lp.add_argument('--pinned', default=None, help='the batch\'s commit: formal, only compared with P_L (C26.9-4); feasibility, HEAD')
    ld = sub.add_parser('lpba-dry-run', help='LP-B: the LPBA admission and the identity of one pair; nothing written')
    ld.add_argument('--pair-set', required=True, choices=hold.LPBA_PAIR_SETS)
    ld.add_argument('--pair', required=True)
    o = ap.parse_args(argv)
    if o.command == 'pair':
        problem = out_problem(o.out) or holdout_roots_problem(o.stage_dir, o.out)
        if problem:
            print(problem, file=sys.stderr)
            return 2
        return run_pair(make_context(o.stage_dir, o.pair, o.out, device=o.device, launcher=o.launcher, unseal_token=o.unseal_token), unseal_token=o.unseal_token)
    if o.command == 'holdout-pair':
        try:
            ctx = make_holdout_context(o.pair, device=o.device, pinned=o.pinned)
        except ValueError as exc:
            print(f'refused: {exc}', file=sys.stderr)
            return 2
        problem = out_problem(ctx['out'])
        if problem:
            print(problem, file=sys.stderr)
            return 2
        return run_pair(ctx)
    if o.command == 'lpba-pair':
        try:
            ctx = make_lpba_context(o.pair_set, o.pair, device=o.device, pinned=o.pinned)
        except ValueError as exc:
            print(f'refused: {exc}', file=sys.stderr)
            return 2
        problem = out_problem(ctx['out'])
        if problem:
            print(problem, file=sys.stderr)
            return 2
        return run_pair(ctx)
    if o.command == 'dry-run':
        return dry_run(o.stage_dir, o.pair)
    if o.command == 'holdout-dry-run':
        return holdout_dry_run(o.pair)
    if o.command == 'lpba-dry-run':
        return lpba_dry_run(o.pair_set, o.pair)
    ctx = read_context(o.context)
    problem = cli_context_problem(ctx)                                 # revision 1, review important 2
    if problem:
        print(f'refused: {problem}', file=sys.stderr)
        return 8
    if o.command in ('t88', 'repair'):
        _, problems = check_identity(ctx, ctx['out'], f'identity_{o.command}.json')
        if problems:
            return 8
        return t88_main(ctx) if o.command == 't88' else repair_main(ctx)
    if o.command == 'third-cert':
        return third_cert_main(ctx)
    return evaluate_main(ctx, lead_release=o.lead_release)


if __name__ == '__main__':
    sys.exit(main())
