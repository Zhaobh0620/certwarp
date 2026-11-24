"""HB step 5 (contract C22.H1, revision 7 adopted by BZ 2026-11-09: D110 (a), D111 (a), D123 (b), D126 (a); preregistration §5.2 step 5): the review certificate unit of the method arm's outputs.

    python -B experiments/hb/hb_review.py dry-run                                     the queue's start checks (the code, the pins, the unseal gate); nothing written
    python -B experiments/hb/hb_review.py queue [--commit SHA] [--redo PAIR=REL ...]  the lead starts it in tag-hb-review-queue (launch-string)
    python -B experiments/hb/hb_review.py unit --pair P --batch DIR [--redo REL]      one pair's review (the queue's unit tag-hb-review-<ffff>-<mmmm>)
    python -B experiments/hb/hb_review.py third-cert --context FILE --batch DIR       the third certification's child (C22.H1-7 (甲): its own entry, the spy first; refused before the spy:
                                                                                       exit THIRD_REFUSED_EXIT = 3, HB diff review n3)
    python -B experiments/hb/hb_review.py withdraw --pair P --authorization REL [--commit SHA]
                                         BZ withdraws a pair's output (D111 (ii), D123 (ii); the lead's item 11): a committed no_resume record (arm review, C22.5-6 (b) and (甲)) -> terminal.json
                                         hx_implementation_error, no output; nothing else changed
    python -B experiments/hb/hb_review.py dev-unit --cell DIR --root ROOT [--batch-sha SHA]
                                         development mode (C22.H1-7 last item; the smoke's (ii), (iv)): a copy of a smoke pair under ROOT = <results>/he_smoke*/<segment>; no unseal object, the
                                         development admission; records under ROOT/review/; never the hold-out's paths

The queue (C22.H1-1, -8): the code (holdout_common.pinned_code_problems against the pinned commit; C22.H0-4), the pins (hb_method.SETTINGS_SHA256, PREREG_SHA256), the unseal gate, the pair list;
every pair's step 4 has ended (hb_method.state final or done, HB review I1: hx_provenance_rejected and an hx_implementation_error without BZ's (甲) stop step 5 before anything is written; a pair
written without an output is recorded only, not reviewed; D110); the input manifest (review/input_manifest.json: every pair's output files,
t88/run.json, field_0600.npz, repair.json, repair.npz by sha256) and then the batch lock review/review.lock (O_EXCL, written once); per pair, in the pair list's order: the gate (no other unit, the GPU
idle, MemAvailable >= 10 GiB, load; waited for up to 30 minutes (D126), else STOP with the pair's output untouched), the disks; the unit with he_core.EVALUATION_UNIT's values (8G, swap 4G, OOMPolicy
continue, CPUQuota 800 %, NUMBA 8, RuntimeMaxSec 1800, nice 5), no GPU, its main process experiments/xb/xb_unit.py.  A unit's end: review.json written -> next pair; its own STOP record -> STOP; killed or
ended without either (holdout_common.classify_unit_end: OOM, timeout, unreadable) -> D123 (b): not a method failure, STOP, a redo only with a committed authorization.  When every pair is finalized:
review/finalized_outputs.json (C22.H1-3).

The unit (C22.H1-2): the identity (this checkout; the context's code_head = the batch's commit, C22.H1-8), the pair's files against the input manifest (round 1 only; a redo against its supplement,
C22.H1-8 修订 7b; read under the spy, HB review m2), the spy (spies.Spies on the stage's pair.npz, and ReviewPolicy: C22.H1-8's forbidden list only, no default refusal, HB review m3; review and
third_certificate records are this step's own and allowed, C22.H1-8 修订 7b) installed after the check that the sealed-list cache is empty (C23.6-5; Hm6: not empty -> STOP 27, to BZ; a breach ->
STOP 28, to BZ), the admission in this process (he_holdout: load_unseal, bind_unseal,
check_subject, the pair list; the images only), then the pair's lock review/<pair>.lock (each pair once; a redo review/<pair>.lock.resume; after every check, before the field is read), then up to
MAX_ROUNDS = 3 rounds (D111 (a)):
  read the output (he_core.read_output), line.json's output.final / disp / cert against the files (sha256, state_hash, d_sha256: C22.H1-2 修订 7a), on the hold-out line.json's review_certificate_in and
  the context's settings_sha256 the pinned ones (C22.H1-7), disp.npz's round trip (bayesreg.eval.fields; C16); any of these not holding -> the pair is not finalized, line_status unchanged, STOP;
  the review certificate (the decision configuration C22.4-1, CPU FP64) compared with cert.json field by field (he_core.same_certificate); the same, or a passing third_certificate.json of this array
  (run_hx.evaluate_main's rule) -> finalized; otherwise review_mismatch_<k>.json in output/ and the third certification in a child (run_hx.third_cert_main, its arithmetic unchanged: pass -> the output
  stands; fail -> the next fallback, the old output output_superseded_<k>; configuration C in tag-he-c3-<pair>; the identity's output withdrawn as output_failed_<k>).
  No child after the last round (HB review m5): the third round's mismatch stops (D111 (a)).
  The child killed (a signal): OOM evidence of this unit (its memory.events, its dmesg lines: tag-he-c3's are another unit's, C22.H1-4 修订 7d) -> D123, STOP, not a round; no evidence ->
  needs_lead_rejudgment, STOP, not a round.  Then the child's records, every round (HB review I2): the sealed-list cache not empty (exit 27) -> STOP 27; the child stopped before it read pair.npz
  (refused by its context or identity, or an exception before the read: a clean report without any access to the pair -- pair.npz not among its opened files either, HB diff review 2 r2 -- and a
  non-zero exit, or exit 3 without a report; HB diff review n3; the cause by the exception's class: input, admission or exception) -> STOP 29, not a firewall event, to the lead; its spy stopped a read (exit 7), no spy report, or a report with passed false or breaches (a read the chain swallowed) -> STOP 28; none is a round, 27 and 28 go
  to BZ.  Another non-zero exit counts as a round (third_returncode).  No
  output/line.json after the child: an output_failed_<k>.json (the identity's third certificate did not pass) -> hx_implementation_error C22.5-6 (a); no record at all (review_failed_<k>.json written
  here) -> (a), or (b) when the round's third_cert_error_<k>.json is another exception (HB review m4; its class recorded, OSError and MemoryError marked possibly infrastructure); the terminal record
  says so, both records copied to /home (item 10), STOP, to BZ.  A fallback line a third certification's chain wrote is bound to its fallback.json (kind, state hash; HB review m6 (i)).
Finalized: <cell>/review/review.json (C22.H1-5's fields: consistent, differences, review, decision, third_certificate, rounds; C22.H1-3's per-pair part) and the cell's terminal.json; both or the
pair is not finalized (HB review m7).  Not finalized: <cell>/review/review_not_finalized.json (why; the redo kind), copied to <review root>/<pair>/ on /home (item 10), and the queue stops.
The stop records of a pair (review_not_finalized, review_unit_stopped) carry one sequence number `seq` (next_stop_seq); the newest -- the one a redo, a withdrawal and a rejudgment bind -- is the
largest (by_seq; HB diff review n4: never the files' mtime or names).
The hold-out policy forbids the whole data/ of the main checkout and of the code tree (HB diff review n5); the other arms' names are the constant XB_ARMS (equal to xb_core.ARMS by a test; NB1: the
review imports nothing of experiments/xb, which is not on its sys.path in production).
D123 (b) redo (C22.H1-4 修订 7d): the queue with --redo PAIR=REL (a committed resume authorization of tool hb, arm review): a supplement of the cell's state now, then the unit with --redo; when the old
output was renamed and no new output exists (killed after the rename), the half directories are kept (output/ -> output_killed_<k>) and the fallback chain goes on from the failed third certificate's
record (continue_fallback: the record's sha256, passed false, its state hash the superseded array's; the next fallback with the suffix _after_third_<k>_redo<j>; nothing certified again), then the review.

LP-B (contract C26.6-6, C26.4 L31, L41, L47; LPBA40 preregistration 06121278... §4.1 "复核证书", §8.2 step 5): --pair-set lpba40_test | lpba40_feas reviews an LPBA40 mode's method outputs the same way
(mode 'lpba40_test' / 'lpba40_feas'): its review root <root>/review, its cells and records (hb_core's pair-set places), the units tag-lp-review-<Sff>-<Smm>; the settings and line.json's
review_certificate_in the pinned ones as on the hold-out; the unit's admission is run_hx.admitted_images' LPBA branch in this process (lpba40_pre_manifest, the manifest's subjects, load_pair with the
pinned LPBA allowlist), the third certification's child admits again in its own (C26.4 L47); the policy (lpba_policy) the hold-out's forbidden list with the mode's root in place of holdout_B_v1
(its eval/, the other arms' cells, records, locks and batches forbidden; nothing of the hold-out read).  When every pair is finalized: finalized_outputs.json of the mode -- the method output is fixed
(C26.6-6).  A redo or a withdrawal of an LPBA mode needs its committed authorizations (C26.8-3, D148: LP-C2's; the feasibility mode has none).
"""
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import hb_core as hb  # noqa: E402

hc, he = hb.hc, hb.he
import run_hx as R  # noqa: E402  (experiments/he is on sys.path through hb_core)
import spies as spying  # noqa: E402

hold = R.hold

import hb_method  # noqa: E402

ARM = hb.ARM
MAX_ROUNDS = 3                                  # D111 (a): HE's three rounds
THIRD_ERROR_EXIT = 9                            # the third certification's child raised (its record in review/third_cert_error_<k>.json)
THIRD_REFUSED_EXIT = 3                          # HB diff review n3: the child refused before its spy (its context, or the policy could not be built); nothing of the pair opened
SPY_BREACH_EXIT = 7
STOP_EXIT = dict(not_finalized=21, implementation=26, killed=20, rejudge=25, sealed_cache=27, spy=28, third_refused=29)
# 27, 28: a firewall event of the unit or its third certification's child: to BZ (C22.H1-8).  29 (HB diff review n3): the child stopped before it read pair.npz (refused by its context or its identity,
# or an exception before the read) with a clean spy report, or with none by THIRD_REFUSED_EXIT: not a firewall event, not a round; the pair is not finalized, to the lead (a redo or a withdrawal)
REVIEW_RECORD, NOT_FINALIZED, OPENED = 'review.json', 'review_not_finalized.json', 'opened_files.json'


# ---- the spy (C22.H1-8; C22.H4-3) --------------------------------------------------------------------------------------------------------------------
# HB review m3 and the lead's ruling (decision_log 2026-11-10 ~1:45 AM CST): the review unit and its third certification use C22.H1-8's forbidden list only, no default refusal (C22.H4-3's
# "禁读": the label files and pair.npz's seg members, the evaluation outputs and locks, the other arms' outputs and records, the development reading products, the files of H1-H4, T0 and the training
# subjects).  Everything else is read and recorded in opened_files.json.
LABEL_NAME = re.compile(r'(^aligned_seg)|(seg[^/]*\.nii(\.gz)?$)', re.IGNORECASE)          # bayesreg/seg/unet/data.py's LABEL_FILES and any .nii(.gz) whose name holds seg (C22.H4-3)
EVALUATION_WORDS = tuple(w for w in spying.EVALUATION_RECORDS if w not in ('review', 'third_certificate'))     # metrics, evaluation, labels, smoothness: evaluation records; the review's own are allowed
EVALUATION_LOCK_NAME = re.compile(r'(^holdout_eval\.lock$)|(\.reeval\.lock$)|(^reeval_[0-9]+\.lock$)|(^truth_gate\.lock$)|(_dev_read\.lock$)|(^evaluation_started\.json$)')
RESULT_DOC_NAME = re.compile(r'^result.*\.md$')                                               # C22.H4-3: the result documents under docs/ and runs/
OTHER_ARM_LOCK = re.compile(r'^(xb__|hb__configc__)')                                         # the locks of XB's arms and of the configuration C arm (locks/)
OTHER_ARM_BATCH = re.compile(r'^(xb_|hb_configc_)')                                           # their batches (batches/)
SEALED_STAGES = ('stage_H1_v1', 'stage_H4_v1', 'stage_H3_IBSR18_v1', 'stage_IBSR18_v1', 'stage_LPBA40_v1', 'stage_T0_tune_v1')     # H1-H4 and T0 (project rules, hard rule 6)
RESULTS_NOT_DEVELOPMENT = ('holdout_B_v1', 'frozen_eb808be', 'seg', 'tests', 'logs')          # the results root's entries that are not development reading products (seg: see below)


# HB diff review NB1 (decision_log 2026-11-10 ~5:47 AM CST): the other arms' names written here, so that the review unit and its third certification import nothing of XB (experiments/xb is not on
# their sys.path: the unit's PYTHONPATH is <frozen>:<code tree>, the script's directory experiments/hb); tests/test_bayesreg/hb/test_hb_production_imports.py checks XB_ARMS == xb_core.ARMS.
XB_ARMS = ('ants_s', 'ants_so', 'convexadam', 'convexadam_v4seg', 'deeds', 'fireants_greedy', 'fireants_greedy_eq', 'fireants_syn', 'fireants_syn_eq', 'niftyreg', 'unigradicon_io',
           'unigradicon_io_eq', 'unigradicon_noio', 'voxelmorph')


def other_arms():
    """Every arm of the hold-out but the method's: the configuration C arm (hb_core.CONFIGC_ARM) and XB's arms (XB_ARMS, equal to xb_core.ARMS by a test)."""
    return tuple(sorted({hb.CONFIGC_ARM, *XB_ARMS}))


@dataclass(frozen=True)
class ReviewPolicy:
    """What the review unit and its third certification may not read (C22.H1-8: the forbidden list, no default refusal).  Everything not forbidden is allowed (and recorded).
    deny: forbidden roots.  results: the results root (abs and real path); under it every entry but RESULTS_NOT_DEVELOPMENT and `keep` is a development reading product, and under its seg/ every entry
    but the hold-out's (a name holding holdout_B_v1) -- a rule on the path, so that an entry made later is covered too.  prefixed: (directory, name pattern) pairs: the other arms' locks and batches.
    Names (both modes): label files, evaluation records, evaluation and development-read locks, result documents."""
    name: str
    deny: tuple
    results: tuple = ()
    keep: tuple = ()
    prefixed: tuple = ()
    not_development: tuple = RESULTS_NOT_DEVELOPMENT        # LP-B: the results root's entries that are not development reading products (an LPBA mode: its own root in place of holdout_B_v1)
    seg_name: str = 'holdout_B_v1'                          # LP-B: the seg/ entries a mode may read hold this name (the hold-out's; an LPBA mode's root name)
    label_name: str = ''                                    # LP-B review minor 1: an extra label-file name rule (an LPBA mode: LPBA_LABEL_NAME); '' none (the hold-out's, the development's)
    deny_except: tuple = ()                                 # LP-B diff review minor 2: (forbidden root, the part of it not forbidden by that root) pairs -- the formal root's code/ (the checkout, C26.9-5)
                                                            # for the feasibility mode; LP-G: the code tree's archive/ and each sealed list under it, exact files (both LPBA modes); () for the
                                                            # hold-out and the development (unchanged).  The other forbidden roots still apply inside it.
    read_only: tuple = ()                                   # LP-G revision 1 (the LP-G review's minor 1): files that may be read but never opened for writing (both forms) -- an LPBA mode's sealed lists
                                                            # (sealed_list_forms); () for the hold-out and the development (unchanged: why(path, write) is then why(path)).

    def why(self, path, write=False):
        raw = os.path.abspath(str(path))
        real = os.path.realpath(raw)
        if write and self.read_only and (raw in self.read_only or real in self.read_only):
            return f'{raw}: a sealed list, read only (LP-G revision 1): no open for writing'
        for base in dict.fromkeys((os.path.basename(raw), os.path.basename(real))):
            if LABEL_NAME.search(base) or (self.label_name and re.search(self.label_name, base, re.IGNORECASE)):
                return f'{base}: a label file by its name'
            if base.endswith('.json') and any(w in base for w in EVALUATION_WORDS):
                return f'{base}: an evaluation record by its name'
            if EVALUATION_LOCK_NAME.search(base):
                return f'{base}: an evaluation lock or a development read lock by its name'
            if RESULT_DOC_NAME.match(base):
                return f'{base}: a result document by its name'
        under = lambda q, d: q == d or q.startswith(d.rstrip('/')+'/')     # noqa: E731
        for p in dict.fromkeys((raw, real)):
            hit = next((d for d in self.deny if under(p, d) and not any(root == d and under(p, part) for root, part in self.deny_except)), None)
            if hit:
                return f'{p}: under the forbidden root {hit}'
            if any(p == k or p.startswith(k.rstrip('/')+'/') for k in self.keep):
                continue
            for r in self.results:
                if p.startswith(r.rstrip('/')+'/'):
                    parts = p[len(r.rstrip('/'))+1:].split('/')
                    if parts[0] not in self.not_development:
                        return f'{p}: a development reading product (the results root\'s {parts[0]}/)'
                    if parts[0] == 'seg' and len(parts) > 1 and self.seg_name not in parts[1]:
                        return f'{p}: a development reading product (seg/{parts[1]}/)'
            for directory, pattern in self.prefixed:
                if p.startswith(directory.rstrip('/')+'/'):
                    first = p[len(directory.rstrip('/'))+1:].split('/', 1)[0]
                    if re.match(pattern, first):
                        return f'{p}: another arm\'s lock or batch'
        return None


def _paths(items):
    out = []
    for item in items:
        out += [os.path.abspath(str(item)), os.path.realpath(str(item))]
    return tuple(dict.fromkeys(out))


def forbidden_roots():
    """The forbidden roots of both modes beside the results rule: the evaluation's directories (holdout_B_v1/eval: its out-dir, master and pair locks; ev/: the truth gate), the stages of H1-H4
    and T0, the subjects' data root (/mnt/d/data: the training subjects', H1-H4's and T0's files, the labels), and the reviews under docs/.  The hold-out mode adds the whole data/ of the main
    checkout and of the code tree (holdout_policy)."""
    return [hc.MAIN_REPO/'data'/'three_d'/s for s in SEALED_STAGES]+[Path('/mnt/d/data'), Path(hc.RESULTS)/'holdout_B_v1'/'eval', Path(hc.RESULTS)/'ev',
                                                                      Path(hb.REPO)/'docs'/'bayesreg'/'reviews']


def holdout_policy():
    """C22.H1-8 for the hold-out: forbidden_roots, the whole data/ of the main checkout and of the code tree (HB diff review n5), the development products rule, and the other arms' cells, records and
    stores (holdout_B_v1/<arm>/, work/<arm>/, D:'s <arm>/), locks and batches."""
    root, big = Path(hc.HOLDOUT_ROOT), Path(hc.BIG_ROOT)
    deny = forbidden_roots()+[hc.MAIN_REPO/'data', Path(hb.REPO)/'data']        # HB diff review n5: the review of the hold-out reads nothing under data/ (its links followed by _paths)
    for arm in other_arms():
        deny += [root/arm, root/'work'/arm, big/arm, Path(hc.REAL_BIG)/arm]
    prefixed = tuple((d, OTHER_ARM_LOCK.pattern) for d in _paths([root/'locks']))+tuple((d, OTHER_ARM_BATCH.pattern) for d in _paths([root/'batches']))
    return ReviewPolicy('holdout review', _paths(deny), results=_paths([hc.RESULTS, he.RESULTS]), prefixed=prefixed)


def dev_policy(root, stage_pair_dir):
    """The development review (he_smoke*/<segment>): forbidden_roots, the development products rule with the segment and this process's own code tree kept, and the hold-out's roots (nothing of
    the hold-out is read in this mode).
    SM3 (runs/20261114-p0/任务书.md): the smoke runs from the detached checkout <results>/he_smoke*/code (C22.H8 "冒烟各单元也从一份分离检出跑"), which lies under the results root; without it in
    `keep` the development products rule forbade the code tree's sealed lists, allowlist and records that the admission reads under the spy (the hold-out's checkout holdout_B_v1/code is not a
    development product by RESULTS_NOT_DEVELOPMENT).  The plan's allowed items of (ii) name the code tree (runs/20261109-hb/smoke_plan.md section 2).  The forbidden roots (the reviews under the
    code tree's docs/, the sealed stages through a data/ link) are judged before `keep`, so they stay forbidden."""
    del stage_pair_dir                                  # the development stage is not forbidden (only SEALED_STAGES are)
    deny = forbidden_roots()+[Path(hc.HOLDOUT_ROOT), Path(hc.BIG_ROOT), Path(hc.REAL_BIG)]
    return ReviewPolicy('development review', _paths(deny), results=_paths([hc.RESULTS, he.RESULTS]), keep=_paths([root, hb.REPO]))


LPBA_LABEL_NAME = r'label[^/]*\.nii(\.gz)?$'                # LP-B review minor 1: LPBA40's truth is <s>/delineation/label.nii.gz (the old stage's copies S??_label.nii.gz); a NIfTI by its name


def sealed_list_forms():
    """LP-G: the files the guard reads for the sealed lists, each in two forms -- holdout_common.sealed_list_paths (resolved; ReviewSpy.report's) and as bayesreg.io.sealed opens it
    (<sealed.REPO>/<path>, absolute; read at call time) -- by _paths (absolute and real path).  Paths only: nothing is opened."""
    from bayesreg.io import sealed
    return _paths([*hc.sealed_list_paths(), *(Path(sealed.REPO)/spec['path'] for spec in sealed.SEALED_LISTS)])


def sealed_list_exceptions(root):
    """LP-G (decision_log 2026-11-20; C22.H4-3: the four lists are not forbidden to the review unit; C23.6-5): deny_except pairs (a form of the forbidden root `root`, a sealed list under that
    form) for every sealed list that lies under it -- exact files, so the rest of the root (the rest of plan/ among it) stays forbidden.  () when no list lies under it."""
    under = lambda q, d: q == d or q.startswith(d.rstrip('/')+'/')     # noqa: E731  (ReviewPolicy.why's rule)
    lists = sealed_list_forms()
    return tuple((d, s) for d in _paths([root]) for s in lists if under(s, d))


def lpba_policy(pair_set):
    """LP-B (C26.6-6): C22.H1-8's forbidden list for an LPBA40 mode's review: forbidden_roots, the whole data/ of the main checkout and of the code tree, the mode's eval/, the other arms' cells,
    records, locks and batches under the mode's root, the other LPBA mode's root, and the hold-out's roots (nothing of the hold-out is read here); the development products rule with the mode's root in
    place of holdout_B_v1 (seg/ entries holding the mode's root name only).  Review I4: the code tree kept (keep, as dev_policy: the detached checkout lies under the results root; the forbidden
    roots are judged before keep).  Diff review minor 2 (C26.9-5): the checkout of both modes is <formal root>/code, so the feasibility mode's denial of the formal root leaves its code/ out
    (deny_except; the code tree's data/ and reviews stay forbidden by their own roots).  Review minor 1: a NIfTI whose name holds 'label' is a label file (LPBA_LABEL_NAME; not every name holding
    'label': the frozen tree's scripts/check_labels_*.py is copied by a third certification's configuration C fallback).  LP-B quick diff review minor 1 (LP-C2 integration item 7): the code
    tree's and the main checkout's archive/ and docs/bayesreg/reviews/ (lpba_arms.CODE_TREE_FORBIDDEN) are forbidden roots too, as in ArmPolicy and XB's LPBA driver policy.  LP-G (decision_log
    2026-11-20; C22.H4-3, C23.6-5): the code tree's archive/ (both forms) has the sealed lists that lie under it -- three in production, under archive/.../3d_stage_HOLDOUT_v1/plan/ -- as
    exact exceptions in deny_except (sealed_list_exceptions); the rest of archive/, the rest of plan/ among it, stays forbidden.  The hold-out's and the development's deny_except stay ().
    LP-G revision 1 (the LP-G review's minor 1): the four sealed lists (both forms, sealed_list_forms) are read_only -- ReviewSpy refuses an open of one for writing (opens_for_writing).  The
    hold-out's and the development's read_only stay ()."""
    root = Path(hb.lpba().root(pair_set))
    formal = Path(hb.lpba().root('lpba40_test'))
    other = Path(hb.lpba().root('lpba40_feas' if pair_set == 'lpba40_test' else 'lpba40_test'))
    deny = forbidden_roots()+[hc.MAIN_REPO/'data', Path(hb.REPO)/'data', root/'eval', other, Path(hc.HOLDOUT_ROOT), Path(hc.BIG_ROOT), Path(hc.REAL_BIG)]
    deny += [Path(t)/rel for t in dict.fromkeys((hc.MAIN_REPO, Path(hb.REPO))) for rel in hb.lpba().CODE_TREE_FORBIDDEN]       # integration item 7 (quick diff review minor 1)
    deny_except = tuple((d, c) for d in _paths([other]) for c in _paths([formal/'code'])) if other == formal else ()
    deny_except += sealed_list_exceptions(Path(hb.REPO)/'archive')                     # LP-G (C22.H4-3, C23.6-5): the guard's lists under the code tree's archive/, one by one
    for arm in other_arms():
        deny += [root/arm, root/'work'/arm]
    prefixed = tuple((d, OTHER_ARM_LOCK.pattern) for d in _paths([root/'locks']))+tuple((d, OTHER_ARM_BATCH.pattern) for d in _paths([root/'batches']))
    not_development = tuple(n for n in RESULTS_NOT_DEVELOPMENT if n != 'holdout_B_v1')+(root.name,)
    return ReviewPolicy(f'{pair_set} review', _paths(deny), results=_paths([hc.RESULTS, he.RESULTS]), keep=_paths([hb.REPO]), prefixed=prefixed, not_development=not_development,
                        seg_name=root.name, label_name=LPBA_LABEL_NAME, deny_except=deny_except, read_only=sealed_list_forms())


def opens_for_writing(args, kwargs):
    """LP-G revision 1: whether open(file, *args, **kwargs) -- builtins.open, io.open, or Python 3.10's pathlib._NormalAccessor.open (io.open) -- opens for writing: a mode holding w, a, x or +
    (spies.Spies' own test, the other way round), or os.open-style flags with a write bit.  The default mode is 'r'."""
    mode = args[0] if args else kwargs.get('mode', kwargs.get('flags', 'r'))
    if isinstance(mode, str):
        return any(c in mode for c in 'wxa+')
    if isinstance(mode, int):
        return bool(mode & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC))
    return False


class ReviewSpy(spying.Spies):
    """spies.Spies (the original pair.npz watched through np.load, NpzFile and zipfile; every open recorded) and the policy's rules on every open (a Breach, recorded).  os.open and mmap are not seen
    (recorded in the record's limits).  The sealed lists are allowed reads, recorded each time (C23.6-5; revision 6f)."""

    def __init__(self, original, policy):
        super().__init__(original)
        self.policy, self.breaches = policy, []

    def install(self):
        super().install()
        import builtins
        import io
        import pathlib
        spy = self

        def guarded(real):
            def wrapped(file, *args, **kwargs):
                if isinstance(file, (str, Path)):
                    why = spy.policy.why(file, write=opens_for_writing(args, kwargs))        # LP-G revision 1: the mode matters only to a policy's read_only (an LPBA mode's sealed lists)
                    if why:
                        spy.breaches.append(dict(path=str(file), why=why))
                        raise spying.Breach(f'the review unit may not open {file}: {why}')
                return real(file, *args, **kwargs)
            return wrapped
        self._inner_open, self._inner_io_open = builtins.open, io.open
        builtins.open, io.open = guarded(builtins.open), guarded(io.open)
        accessor = getattr(pathlib, '_NormalAccessor', None)
        if accessor is not None and self.real_path_open is not None:
            self._inner_path_open = accessor.__dict__.get('open')
            pathlib._NormalAccessor.open = staticmethod(guarded(self._inner_path_open.__func__ if hasattr(self._inner_path_open, '__func__') else self._inner_path_open))
        return self

    def evaluation_records(self):
        """C22.H1-8 修订 7b: the review's own records (review_mismatch_<k>.json, third_certificate.json) are not evaluation records here; metrics, evaluation, labels, smoothness still are."""
        return [p for p in self.read_opened if Path(p).suffix == '.json' and any(w in Path(p).name for w in EVALUATION_WORDS)]

    def report(self, *, expect_reads=True):
        record = self.record(expect_reads=expect_reads)
        sealed = set(hc.sealed_list_paths())
        opened = [os.path.realpath(p) for p in self.opened]
        policy = dict(name=self.policy.name, deny=list(self.policy.deny), results=list(self.policy.results), keep=list(self.policy.keep), prefixed=[list(x) for x in self.policy.prefixed],
                      rule='C22.H1-8: the forbidden list only, no default refusal (HB review m3)')
        if (self.policy.not_development, self.policy.seg_name, self.policy.label_name) != (RESULTS_NOT_DEVELOPMENT, 'holdout_B_v1', ''):          # LP-B: an LPBA mode's policy records its fields (the hold-out's record unchanged)
            policy.update(not_development=list(self.policy.not_development), seg_name=self.policy.seg_name, label_name=self.policy.label_name,
                          deny_except=[list(x) for x in self.policy.deny_except], read_only=list(self.policy.read_only))
        return dict(record, policy=policy, breaches=list(self.breaches), opened=list(self.opened),
                    opened_realpath=opened, original_realpath=os.path.realpath(record['original']) if isinstance(record.get('original'), str) and record.get('original') else None,
                    sealed_lists_opened=sorted({p for p in opened if p in sealed}), n_sealed_list_opens=sum(p in sealed for p in opened),
                    limits='os.open and mmap are not watched; a file opened by a child process (classify_run, the frozen worker in tag-he-c3) is not seen (C22.H9, revision 7d)',
                    passed=record['passed'] and not self.breaches)


# ---- the records -------------------------------------------------------------------------------------------------------------------------------------
def line_json_problems(o, line, u):
    """C22.H1-2 (修订 7a): line.json's output.final / disp / cert against the files: their sha256, final's state_hash = u's hash, disp's d_sha256 = disp.npz's d."""
    from bayesreg.eval import fields
    o = Path(o)
    out = (line or {}).get('output') or {}
    problems = []
    for key, name in (('final', 'final.npz'), ('disp', 'disp.npz'), ('cert', 'cert.json')):
        row = out.get(key) or {}
        if row.get('file') != name or row.get('sha256') != hb.sha(o/name):
            problems.append(f'line.json output.{key} {row} is not {name} with its sha256 now')
    if (out.get('final') or {}).get('state_hash') != he.array_sha256(u):
        problems.append('line.json output.final.state_hash is not the hash of final.npz\'s u')
    try:
        d, _ = fields.read_disp(o/'disp.npz')
        if (out.get('disp') or {}).get('d_sha256') != he.array_sha256(d):
            problems.append('line.json output.disp.d_sha256 is not disp.npz\'s d')
    except Exception as exc:                            # noqa: BLE001 - fields.FieldError and the like
        problems.append(f'disp.npz: {type(exc).__name__}: {exc}')
    return problems


def tree_rows(directory):
    directory = Path(directory)
    return {str(p.relative_to(directory)): hb.sha(p) for p in sorted(directory.rglob('*')) if p.is_file()} if directory.is_dir() else None


def evidence_rows(cell_dir):
    """File by file (C22.H1-2 修订 7c, C22.H1-3; HB review m6 (ii), (iii)): the current output/ (every file: third_certificate.json, review_mismatch_<k>.json beside the four of the line), every
    output_superseded_*, output_failed_*, output_killed_* and fallback_* directory (step 4's fallback_configc/, fallback_identity/ with their fallback.json, and the third certification's), the
    review's records, and a resumed step 4's first attempt (the sibling <pair>.attempt<k>, kept on /home: it does not move with the cell)."""
    cell_dir = Path(cell_dir)
    rows = {}
    for pattern in ('output', 'output_superseded_*', 'output_failed_*', 'output_killed_*', 'fallback_*'):
        for p in sorted(cell_dir.glob(pattern)):
            rows[p.name] = tree_rows(p) if p.is_dir() else hb.sha(p)
    for p in sorted((cell_dir/'review').glob('*.json')) if (cell_dir/'review').is_dir() else []:
        rows[f'review/{p.name}'] = hb.sha(p)
    for p in sorted(cell_dir.parent.glob(f'{cell_dir.name}.attempt*')):
        rows[f'attempts/{p.name}'] = dict(path=str(p), files=tree_rows(p))
    return rows


def finalized_part(cell_dir, *, line, review_doc):
    """C22.H1-3's per-pair part: the output files, the state hash, the T88 and repair files (PQ's inputs, C22.Q1-2), the fallback record and its state hash, the evidence directories."""
    cell_dir = Path(cell_dir)
    fallback = (line or {}).get('fallback') or None
    return dict(line_status=line.get('line_status'), output=hb.file_rows(cell_dir, hb.OUTPUT_FILES), state_hash=((line.get('output') or {}).get('final') or {}).get('state_hash'),
                t88_repair=hb.file_rows(cell_dir, ('t88/run.json', 't88/field_0600.npz', 'repair/repair.json', 'repair/repair.npz')),
                fallback=None if not fallback else dict(kind=fallback.get('kind'), state_hash=fallback.get('state_hash'), certificate_passed=he.passes(fallback.get('certificate') or {}),
                                                        run_dir=fallback.get('run_dir')),
                superseded=list(line.get('superseded') or []), evidence=evidence_rows(cell_dir), **review_doc)      # C22.H1-5: consistent, differences, review, decision, third_certificate, rounds at the top


# ---- the mode (holdout | dev) -------------------------------------------------------------------------------------------------------------------------
@dataclass
class Mode:
    name: str                                   # holdout | dev
    root: Path                                  # the review's records: holdout_B_v1/review, or <smoke segment>/review
    batch_sha: str | None
    policy: ReviewPolicy
    segment: Path | None = None                 # dev: the smoke segment <results>/he_smoke*/<segment> (the third certification's child is told it, not the records' directory)

    def lock(self, pair, redo=False):
        return self.root/(f'{pair}.lock' if not redo else f'{pair}.lock.resume')

    def manifest(self):
        return self.root/'input_manifest.json'


def holdout_mode(batch_sha):
    return Mode('holdout', hb.review_root(), batch_sha, holdout_policy())


PINNED_MODES = ('holdout',)+hold.LPBA_PAIR_SETS          # LP-B: the modes whose settings and review_certificate_in are the pinned ones (C22.H1-7; C26.6-1)


def lpba_mode(pair_set, batch_sha):
    """LP-B: an LPBA40 mode's review (its name is its pair set; its root <root>/review; lpba_policy)."""
    return Mode(pair_set, hb.review_root(pair_set), batch_sha, lpba_policy(pair_set))


def mode_of(pair_set, batch_sha):
    return lpba_mode(pair_set, batch_sha) if hb.is_lpba(pair_set) else holdout_mode(batch_sha)


def pair_set_of_mode(mode):
    """The pair set of a mode's cells: an LPBA mode's own name, else the hold-out's (the development mode reviews one given cell and never asks)."""
    return mode.name if hb.is_lpba(mode.name) else 'holdout'


DEV_ROOT_PREFIX = 'he_smoke'
DEV_STAGE_NAME = 'stage_L2R_val_v1'             # the development stage of holdout_common.DEV_PAIRS (its 19 pairs)


def dev_root_problem(root, cell_dir, results=None):
    """A development review writes only under <results>/he_smoke*/<segment>/ and reviews a cell under it (C22.H8: only in the new root, on copies)."""
    results = Path(he.RESULTS if results is None else results).resolve()
    root, cell_dir = Path(root).resolve(), Path(cell_dir).resolve()
    smoke = next((p for p in [root, *root.parents] if p.parent == results), None)
    if smoke is None or not smoke.name.startswith(DEV_ROOT_PREFIX) or root == smoke:
        return f'{root}: a development review runs under {results}/{DEV_ROOT_PREFIX}*/<segment>'
    if root not in cell_dir.parents:
        return f'{cell_dir} is not under {root}'
    return None


# ---- the third certification's child (C22.H1-7 (甲)) -----------------------------------------------------------------------------------------------
class ThirdChild:
    """python -B hb_review.py third-cert --context FILE --batch DIR --mode M [--root R] in its own process (no timeout of its own: the unit's RuntimeMaxSec is its limit, C22.H1-4 修订 7c)."""

    def __init__(self, mode, batch):
        self.mode, self.batch = mode, batch

    def __call__(self, ctx):
        argv = [str(hb.PY), '-B', str(HERE/'hb_review.py'), 'third-cert', '--context', str(R.context_path(ctx)), '--batch', str(self.batch), '--mode', self.mode.name, '--root', str(self.mode.segment or self.mode.root)]
        R.fc.check_launch(argv)
        with (Path(ctx['out'])/'review'/'third-cert.log').open('a') as f:
            return subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT).returncode


def third_child_main(ctx, mode, *, launcher=None, certification=None):
    """The third certification in this process (the child): the spy installed after the sealed-list cache is found empty, then run_hx.third_cert_main (its identity, its admission, its certificate, its
    fallback chain: unchanged).  An exception (the identity not passing in the chain: C22.5-6 (a)) -> review/third_cert_error_<k>.json, exit THIRD_ERROR_EXIT.  The spy's record -> review/opened_files_third_<k>.json."""
    from bayesreg.io.sealed import sealed_cache_is_empty
    rev = Path(ctx['out'])/'review'
    rev.mkdir(exist_ok=True)
    if not sealed_cache_is_empty():
        hb.write_once(hb.numbered(rev, 'third_sealed_cache'), dict(reason='the sealed-list cache is not empty before the spy (C23.6-5; Hm6): to BZ', utc=hb.utc()))
        return STOP_EXIT['sealed_cache']
    spy = ReviewSpy(R.pair_path(ctx), mode.policy).install()
    rc, error = None, None
    try:
        rc = R.third_cert_main(ctx, launcher=launcher, certification=certification)
    except spying.Breach as exc:
        rc, error = SPY_BREACH_EXIT, dict(kind='spy', exception=str(exc))
    except Exception as exc:                            # noqa: BLE001 - recorded; the parent reads the cell
        rc, error = THIRD_ERROR_EXIT, dict(kind='implementation_error' if isinstance(exc, R.IdentityNotCertified) else 'exception', exception=f'{type(exc).__name__}: {exc}'[:2000],
                                           exception_class=type(exc).__name__, exception_mro=[c.__name__ for c in type(exc).__mro__], refusal_family=refusal_family(exc),
                                           possibly_infrastructure=isinstance(exc, (OSError, MemoryError)), traceback=traceback.format_exc()[-4000:])
    finally:
        spy.remove()
    if error is not None:
        hb.write_once(hb.numbered(rev, 'third_cert_error'), dict(error, evidence=evidence_rows(ctx['out']), utc=hb.utc()))
    hb.write_once(hb.numbered(rev, 'opened_files_third'), dict(spy.report(), rc=rc))
    return rc


# ---- D123 (b): on along the fallback chain from the failed third certificate's record -------------------------------------------------------------
def continue_fallback(ctx, *, k, redo, record_sha256, launcher=None, certification=None):
    """C22.H1-4 修订 7d: never restores the old output, never certifies it again.  output_superseded_<k>/third_certificate.json: its sha256 the supplement's (record_sha256), passed false, its
    output_state_hash the hash of output_superseded_<k>/final.npz's u.  A half output/ is renamed output_killed_<k> (kept); the next fallback (he_core.next_fallback of the superseded line status) with the
    suffix _after_third_<k>_redo<redo> (a new directory: a killed attempt's half fallback directory stays), then the new output (run_hx.third_cert_main's new line).  ValueError when a check does not hold."""
    out = Path(ctx['out'])
    sup = out/f'output_superseded_{k}'
    rec_path = sup/'third_certificate.json'
    if not rec_path.is_file() or hb.sha(rec_path) != record_sha256:
        raise ValueError(f'{rec_path}: not there or not the record of the supplement ({record_sha256})')
    record = he.read_json(rec_path)
    if record.get('passed') is not False:
        raise ValueError(f'{rec_path}: passed is {record.get("passed")!r}: only a third certificate that did not pass is continued (a passed one resolved the pair)')
    u_old, line_old, _ = he.read_output(sup)
    if record.get('output_state_hash') != he.array_sha256(u_old):
        raise ValueError(f'{rec_path}: output_state_hash is not the hash of {sup}/final.npz\'s u')
    start = he.next_fallback(line_old['line_status'])
    if start is None:
        raise ValueError(f"{line_old['line_status']}: no fallback after it (the identity's failed third certificate withdraws the output: output_failed_<k>)")
    o = out/he.OUTPUT_DIR
    killed = None
    if o.exists():
        killed = out/f'output_killed_{k}'
        if killed.exists():
            killed = hb.attempt_dir(killed)
        o.rename(killed)
    pin, _ = R.admitted_images(ctx)
    certify, config = R.certifier(ctx) if certification is None else certification
    launcher = R.third_launcher(ctx) if launcher is None else launcher
    suffix = f'_after_third_{k}_redo{redo}'
    u, cert, fallback, status = R.fallback_chain(ctx, pin, start=start, launcher=launcher, certify=certify, config=config, suffix=suffix)
    new = dict({k_: v for k_, v in line_old.items() if k_ not in ('output', 'ended_utc')}, line_status=status, fallback=fallback, third_certificate=record,
               superseded=list(line_old.get('superseded') or [])+[str(sup)], failure=dict(line_old.get('failure') or he.failure('repair_not_certified', 'the third certificate (C22.3-9 (d)) did not pass'),
                                                                                       third_certificate_failed=True),
               decision_certificate=dict(process=cert.get('process'), state_hash=cert.get('state_hash'), passed=he.passes(cert)),
               third_fallback_launcher=dict(kind=getattr(launcher, 'kind', None), unit_prefix=getattr(launcher, 'unit_prefix', None), rule=R.THIRD_LAUNCHER_RULE),
               continued=dict(rule='C22.H1-4 修订 7d: on along the fallback chain from the failed third certificate\'s record; nothing certified again', record=str(rec_path), record_sha256=record_sha256,
                              suffix=suffix, killed_output=None if killed is None else str(killed)))
    _, problems = R.check_identity(ctx, out, R.numbered(out, 'identity_continue_end'))
    if problems:
        raise ValueError(f'the frozen tree or the code changed: {problems}')
    he.write_output(o, u, pin.grid, cert, new)
    return dict(status=status, suffix=suffix, killed_output=None if killed is None else str(killed))


# ---- one pair ----------------------------------------------------------------------------------------------------------------------------------------
class NotFinalized(Exception):
    def __init__(self, kind, reason, code=STOP_EXIT['not_finalized'], **extra):
        super().__init__(f'{kind}: {reason}')
        self.kind, self.reason, self.code, self.extra = kind, reason, code, extra


def third_child_records(rev, before, rc):
    """HB review I2 (C22.H1-8): the records the third certification's child wrote this round (review/*.json not there before it): third_sealed_cache_<k> (the cache was not empty), opened_files_third_<k>
    (its spy's report), third_cert_error_<k> (an exception).  A sealed-cache record or exit 27 -> NotFinalized sealed_cache (27); the child stopped before it read pair.npz (refused_before_reading:
    a clean report without any access to the pair and a non-zero exit, or THIRD_REFUSED_EXIT without a report) -> NotFinalized third_child_refused (29; HB diff review n3: not a firewall event; its
    exit, the identity_third*.json records and the log named); exit SPY_BREACH_EXIT, no spy report, or a report with passed false or breaches -> NotFinalized spy (28): the pair is not finalized,
    no round counted, the batch stops, to BZ.  A breach the chain swallowed (run_hx.configc_fallback's except Exception) is in the report.
    Returns {name: (path, document)} of the round's records."""
    new = {}
    for p in sorted(Path(rev).glob('*.json')):
        if p.name in before:
            continue
        for stem in ('third_sealed_cache', 'opened_files_third', 'third_cert_error'):
            if p.name.startswith(stem+'_') or p.name == stem+'.json':
                new[stem] = (p, he.read_json(p) or {})
    if rc == STOP_EXIT['sealed_cache'] or 'third_sealed_cache' in new:
        raise NotFinalized('third_sealed_cache', f'the third certification\'s child found the sealed-list cache not empty before its spy (exit {rc}; C23.6-5, Hm6): to BZ', code=STOP_EXIT['sealed_cache'],
                           records={k: str(v[0]) for k, v in new.items()})
    report = new.get('opened_files_third')
    error = new.get('third_cert_error')
    refused = refused_before_reading(rc, None if report is None else report[1], None if error is None else error[1])
    if refused is not None:                             # HB diff review n3: not a firewall event
        identity = {p.name: hb.sha(p) for p in sorted(Path(rev).parent.glob('identity_third*.json'))}
        raise NotFinalized('third_child_refused', f"the third certification's child stopped before it read pair.npz ({refused}, exit {rc}): not a firewall event, not a round; to the lead (a redo "
                                                  'or a withdrawal)', code=STOP_EXIT['third_refused'], child_exit=rc, cause=refused, records={k: str(v[0]) for k, v in new.items()},
                           identity_records=identity, log=str(Path(rev)/'third-cert.log'))
    if rc == SPY_BREACH_EXIT or report is None or report[1].get('passed') is not True or report[1].get('breaches'):
        why = ('its spy stopped a read' if rc == SPY_BREACH_EXIT else 'it wrote no spy report' if report is None else
               f"its spy's report: passed {report[1].get('passed')!r}, breaches {report[1].get('breaches')}, forbidden {report[1].get('forbidden_access')}")
        raise NotFinalized('third_spy', f'the third certification\'s child (exit {rc}): {why} (C22.H1-8): to BZ, not a round', code=STOP_EXIT['spy'], records={k: str(v[0]) for k, v in new.items()})
    return new


REFUSAL_FAMILIES = ('input', 'admission')       # HB diff review 2 r2: the cause of a child that raised before it read pair.npz, by its exception's class (third_cert_error_<k>.json)


def refusal_family(exc):
    """HB diff review 2 r2: 'input' for jd_curl's InputValidationError (a malformed input), 'admission' for the rest of run_hx.admission_refusals() (the subject guards, provenance, a missing file, a
    sealed list that is not the pinned one, a name that is not plain), None for any other exception.  Written by the child into third_cert_error_<k>.json beside the class and its MRO."""
    from jd_curl.validation import InputValidationError
    if isinstance(exc, InputValidationError):
        return 'input'
    return 'admission' if isinstance(exc, R.admission_refusals()) else None


def error_family(error):
    """The family of a third_cert_error_<k>.json document (`error`): its refusal_family when it is one of REFUSAL_FAMILIES (the child wrote it), else read from the class names it recorded
    (exception_mro, else exception_class) against InputValidationError and the classes of run_hx.admission_refusals(); None when neither tells."""
    if not isinstance(error, dict):
        return None
    if error.get('refusal_family') in REFUSAL_FAMILIES:
        return error['refusal_family']
    names = error.get('exception_mro') if isinstance(error.get('exception_mro'), list) else [error.get('exception_class')]
    names = [n for n in names if isinstance(n, str)]
    if 'InputValidationError' in names:
        return 'input'
    return 'admission' if set(names) & {c.__name__ for c in R.admission_refusals()} else None


def _opened_the_original(report):
    """HB diff review 2 r2: whether the spy report's `opened` (every path the child opened through builtins.open and io.open, as given) holds the report's `original` (the pair.npz watched),
    compared after both are resolved: its bytes were read even when no loader asked a member (load_pair's container sha256)."""
    resolved, target = report.get('opened_realpath'), report.get('original_realpath')
    if isinstance(resolved, list) and isinstance(target, str) and target:     # A5a review m10: resolved by the child itself (its working directory), as report() writes them
        return target in resolved
    original = report.get('original')
    if not isinstance(original, str) or not original:
        return True                                     # fail closed: not known, so not "before reading"
    target = os.path.realpath(original)
    return any(isinstance(p, str) and os.path.realpath(p) == target for p in report.get('opened') or [])


def refused_before_reading(rc, report, error=None):
    """HB diff review n3: why the third certification's child stopped before it read pair.npz, or None.  Exit THIRD_REFUSED_EXIT without a report: refused before its spy ('before_spy');
    a whole spy report (its keys and `opened`, `original`) that did not pass, with no breach, no forbidden access, no evaluation record opened, no access at all to pair.npz (no loader call, no
    read at the guard, the class or zipfile) and -- HB diff review 2 r2 -- pair.npz not among the files it opened (its bytes not read through open either), and a non-zero exit: 'context' (exit
    2), 'identity' (exit 8), THIRD_ERROR_EXIT by the class of the round's exception (`error`, its third_cert_error_<k>.json; error_family): 'input' (InputValidationError), 'admission' (the
    rest of run_hx.admission_refusals()), 'exception' (any other); or 'exit <rc>'.  Anything else (a breach, a read the loaders did not account for, the pair's bytes opened, no report with
    another exit) is the spy's (STOP 28)."""
    if report is None:
        return 'before_spy' if rc == THIRD_REFUSED_EXIT else None
    if not isinstance(rc, int) or rc in (0, SPY_BREACH_EXIT, STOP_EXIT['sealed_cache']) or report.get('passed') is True:
        return None                                     # a passed report has its positive condition: the pair was read
    if any(key not in report for key in ('loader_calls', 'original_access', 'npz_access', 'n_zip_opens', 'breaches', 'forbidden_access', 'evaluation_records_opened', 'opened', 'original')):
        return None                                     # not a whole spy report: the spy's case
    if report.get('breaches') or report.get('forbidden_access') or report.get('evaluation_records_opened'):
        return None
    if report.get('loader_calls') or report.get('original_access') or report.get('npz_access') or report.get('n_zip_opens'):
        return None
    if _opened_the_original(report):
        return None                                     # r2: the pair's bytes were opened (e.g. hashed) before the child stopped: not "before it read pair.npz"
    if rc == THIRD_ERROR_EXIT:
        return error_family(error) or 'exception'
    return {2: 'context', 8: 'identity'}.get(rc, f'exit {rc}')


def fallback_record(out, line, digest):
    """C22.H1-8 修订 7b: a fallback line written by a third certification is bound to its fallback.json (fallback_<kind>_after_third_<k>[_redo<j>]/fallback.json): the line's fallback state_hash is the
    output's, and such a file holds that kind and state_hash.  The file (path, sha256); NotFinalized fallback_json otherwise."""
    fb = (line or {}).get('fallback') or {}
    if fb.get('state_hash') != digest:
        raise NotFinalized('fallback_json', f"line.json's fallback state_hash {fb.get('state_hash')} is not the output's {digest}")
    for p in sorted(Path(out).glob('fallback_*_after_third_*/fallback.json'), reverse=True):
        doc = he.read_json(p) or {}
        if doc.get('state_hash') == digest and doc.get('kind') == fb.get('kind'):
            return dict(path=str(p), sha256=hb.sha(p), kind=doc.get('kind'), state_hash=digest)
    raise NotFinalized('fallback_json', f'no fallback_*_after_third_*/fallback.json of kind {fb.get("kind")} with the output\'s state hash {digest}')


def manifest_row(mode, pair, redo=None):
    """The pair's rows of the input manifest (round 1) or, for a redo, of its supplement."""
    path = mode.root/'supplements'/f'{pair}_{redo}.json' if redo is not None else mode.manifest()
    doc = hb.read_json(path)
    if doc is None or pair not in (doc.get('pairs') or {}):
        raise NotFinalized('input_manifest', f'{path} has no row of {pair}')
    return path, doc['pairs'][pair]


def check_inputs(cell_dir, row):
    now = hb.file_rows(cell_dir, list(row.get('files') or {}))
    if now != row.get('files'):
        diff = {k: (row['files'].get(k), now.get(k)) for k in row['files'] if row['files'].get(k) != now.get(k)}
        raise NotFinalized('input_manifest', f'the cell\'s files differ from the manifest: {diff}')
    return now


def evidence_of_kill(start_probe, end_probe, oom_lines):
    """C22.H1-4 修订 7c, 7d: the OOM evidence of this unit (its own cgroup's memory.events between the two probes; dmesg lines whose task is in this unit's cgroup -- tag-he-c3 is another unit)."""
    start_ev = (start_probe or {}).get('events')
    end_ev = (end_probe or {}).get('events')
    events = hc.events_delta(start_ev, end_ev)
    cgroup = (end_probe or {}).get('cgroup') or (start_probe or {}).get('cgroup')
    items = hc.oom_evidence(unit=(end_probe or {}).get('unit'), unit_cgroup=cgroup, oom_lines=oom_lines, events=events)
    return dict(events=events, cgroup=cgroup, oom_lines=oom_lines, items=items, readable=dict(dmesg=oom_lines is not None, events=events is not None))


def default_kill_evidence(t0):
    import unit_probe
    text = hb.System().dmesg()
    return lambda start: evidence_of_kill(start, unit_probe.probe('end'), hc.parse_oom(text if text is not None else None, t0))


def review_pair(ctx, mode, *, batch, redo=None, redo_doc=None, certification=None, third=None, probe=None, kill_evidence=None, install_spy=True, launcher=None):
    """One pair (module docstring).  Returns 0 (finalized) or a STOP_EXIT code; writes review.json + terminal.json, or review_not_finalized.json.  certification, third, probe, kill_evidence, launcher:
    the tests' stand-ins (the certificate, the third certification's child, the unit probe, the kill evidence, configuration C's launcher for a continued chain)."""
    from bayesreg.eval import fields
    from bayesreg.io.sealed import sealed_cache_is_empty
    import unit_probe
    probe = unit_probe.probe if probe is None else probe
    out = Path(ctx['out'])
    pair = ctx['pair']
    rev = out/'review'
    rev.mkdir(exist_ok=True)
    o = out/he.OUTPUT_DIR
    if (rev/REVIEW_RECORD).exists():
        return finish_not_finalized(rev, NotFinalized('once', f'{rev/REVIEW_RECORD} exists: a pair is finalized once'), mode, pair)
    start_probe = probe('start')
    spy = None
    rounds, kept = [], {}
    try:
        if mode.name == 'dev' and (pair not in hc.DEV_PAIRS or Path(ctx['stage_dir']).name != DEV_STAGE_NAME):          # runs/20261109-align item 8: one of stage_L2R_val_v1's 19 pairs
            raise NotFinalized('dev_pair', f"{pair} of {ctx['stage_dir']} is not one of the {len(hc.DEV_PAIRS)} pairs of {DEV_STAGE_NAME} (C22.H8 (ii): the development mode's pair check)")
        identity, problems = R.check_identity(ctx, rev, R.numbered(rev, 'identity_review'))
        if problems:
            raise NotFinalized('identity', f'{problems}')
        if mode.batch_sha is not None and ctx.get('code_head') != mode.batch_sha:
            raise NotFinalized('code_head', f"the context's code_head {ctx.get('code_head')} is not the batch's commit {mode.batch_sha} (C22.H1-8)")
        if (hb.is_lpba(mode.name) or hold.is_lpba(ctx)) and mode.name != hold.pair_set_of(ctx):        # LP-B: an LPBA mode reviews only its own pair set's contexts, and only it reviews them
            raise NotFinalized('pair_set', f"the context's pair set {hold.pair_set_of(ctx)} is not the review's mode {mode.name}")
        if mode.name in PINNED_MODES:
            if ctx.get('settings_sha256') != hb_method.SETTINGS_SHA256 or he.settings_sha256(ctx['settings']) != hb_method.SETTINGS_SHA256:
                raise NotFinalized('settings', f"the context's settings_sha256 {ctx.get('settings_sha256')} is not the pinned {hb_method.SETTINGS_SHA256} (C22.H1-7)")
        if install_spy:                                 # HB review m2: the spy before the input manifest's reads of the cell, so that they are in opened_files.json
            if not sealed_cache_is_empty():
                raise NotFinalized('sealed_cache', 'the sealed-list cache is not empty before the spy (C23.6-5; Hm6): to BZ', code=STOP_EXIT['sealed_cache'])
            spy = ReviewSpy(R.pair_path(ctx), mode.policy).install()
        manifest_path, row = manifest_row(mode, pair, redo)
        kept['inputs'] = dict(manifest=str(manifest_path), manifest_sha256=hb.sha(manifest_path), files=check_inputs(out, row))
        pin, admission = R.admitted_images(ctx)                         # this process's own admission (C23.3-5): load_unseal, bind_unseal, check_subject, the pair list; the images only
        grid = pin.grid
        hc.take_lock(mode.lock(pair, redo is not None), hc.lock_line(arm='review', pair=pair, mode=mode.name, batch=str(batch), redo=redo, manifest_sha256=kept['inputs']['manifest_sha256']))     # C22.H1-8: after the checks, before the first read of the field
        continued = None
        if redo_doc is not None and redo_doc.get('continue') is not None:
            c = redo_doc['continue']
            try:
                continued = continue_fallback(ctx, k=c['k'], redo=redo, record_sha256=c['record_sha256'], launcher=launcher, certification=certification)
            except R.IdentityNotCertified as exc:
                return implementation_error(ctx, mode, rev, pair, kind='a', reason=f'the continued chain: {exc}', rounds=rounds, kept=kept, spy=spy)
            except ValueError as exc:
                raise NotFinalized('continue', f'the record does not hold, nothing continued: {exc}') from None
        kept['continued'] = continued
        certify, config = R.certifier(ctx) if certification is None else certification
        third = ThirdChild(mode, batch) if third is None else third
        failed_before = {p.name for p in out.glob('output_failed_*.json')}
        child_records = {}
        for index in range(MAX_ROUNDS):
            if not (o/'line.json').is_file():
                new_failed = sorted(p.name for p in out.glob('output_failed_*.json') if p.name not in failed_before)
                if new_failed:
                    return implementation_error(ctx, mode, rev, pair, kind='a', reason=f'the identity output\'s third certificate did not pass: {new_failed} (C22.5-6 (a))', rounds=rounds, kept=kept, spy=spy)
                if o.exists():
                    raise NotFinalized('half_output', f'{o} is there without line.json: a half output (D123); not taken as an output')
                error = child_records.get('third_cert_error')
                kind = 'b' if error is not None and error[1].get('kind') == 'exception' else 'a'      # HB review m4: C22.H1-2 修订 7c's last item: another exception of the child is (b)
                k = len(list(rev.glob('review_failed_*.json')))
                hb.write_once(rev/f'review_failed_{k}.json', dict(reason='output/ is not there and no output_failed_*.json: the chain after the third certificate left no output (C22.H1-2)',
                                                                   kind=kind, third_cert_error=None if error is None else dict(path=str(error[0]), sha256=hb.sha(error[0])),
                                                                   rounds=rounds, evidence=evidence_rows(out), utc=hb.utc()))
                return implementation_error(ctx, mode, rev, pair, kind=kind, reason=f'no output and no output_failed record after the third certification: review_failed_{k}.json'
                                            + ('' if error is None else f" (the child's {error[1].get('exception')})"), rounds=rounds, kept=kept, spy=spy,
                                            exception_class=None if error is None else error[1].get('exception_class'))
            try:
                u, line, cert = he.read_output(o, grid)
            except (OSError, ValueError, KeyError) as exc:
                raise NotFinalized('read_output', f'{type(exc).__name__}: {exc}') from None
            problems = line_json_problems(o, line, u)
            if problems:
                raise NotFinalized('line_json', f'{problems}')
            if mode.name in PINNED_MODES and line.get('review_certificate_in') != ctx['settings'].get('review_certificate_in'):
                raise NotFinalized('review_certificate_in', f"line.json's review_certificate_in {line.get('review_certificate_in')!r} is not the setting's (C22.H1-7)")
            try:
                d, disp_record = fields.read_disp(o/'disp.npz', grid.shape)
            except (OSError, ValueError, KeyError) as exc:                # bayesreg.eval.fields' FieldError among them
                raise NotFinalized('disp', f'{type(exc).__name__}: {exc}') from None
            roundtrip = fields.roundtrip_check(u, d, tuple(grid.spacing))
            if not roundtrip['passed']:
                raise NotFinalized('roundtrip', f'disp.npz is not final.npz (C16 round trip): {roundtrip}')
            digest = he.array_sha256(u)
            if (line.get('third_certificate') or {}).get('passed') is False and line.get('line_status') in (he.FALLBACK_C, he.FALLBACK_ID):
                bound = fallback_record(out, line, digest)            # C22.H1-8 修订 7b (HB review m6 (i)): a line a third certification's chain wrote is bound to its fallback.json too
            else:
                bound = None
            review = he.certificate_document(certify(u, grid), config, process=f'the step-5 review unit (hb_review.py, pid {os.getpid()}; reads no label)', array_sha256_of_u=digest)
            same, diffs = he.same_certificate(review, cert)
            third_record = he.read_json(o/'third_certificate.json')
            resolved = third_record is not None and third_record.get('output_state_hash') == digest and third_record.get('passed') is True
            rounds.append(dict(state_hash=digest, line_status=line['line_status'], consistent=same, differences=diffs, third_certificate=third_record, review=review, roundtrip=roundtrip['passed'],
                               fallback_json=bound))
            if same or resolved:
                break
            he.write_json(o/R.numbered(o, 'review_mismatch'), dict(review=review, decision=cert, differences=diffs, rule='C22.3-9 (d): the third certification decides, in a process that reads no label', utc=hb.utc()))
            if index == MAX_ROUNDS-1:                   # HB review m5: no third certification whose output no round would review; the output stays the one this round reviewed
                break
            tick = hb.System().uptime() if kill_evidence is None else 0.
            child_start = probe('child_start')
            before = {p.name for p in rev.glob('*.json')}
            rc = third(ctx)
            rounds[-1]['third_returncode'] = rc
            if isinstance(rc, int) and (rc < 0 or rc > 128):
                ev = (kill_evidence or default_kill_evidence(tick))(child_start)
                rounds[-1]['third_kill_evidence'] = ev
                if ev['items']:
                    raise NotFinalized('third_child_oom', f'the third certification\'s child was killed with OOM evidence of this unit (exit {rc}): D123, not a round', code=STOP_EXIT['killed'], evidence=ev)
                raise NotFinalized('third_child_signal', f'the third certification\'s child was killed by a signal without OOM evidence (exit {rc}): needs_lead_rejudgment, not a round',
                                   code=STOP_EXIT['rejudge'], evidence=ev)
            child_records = third_child_records(rev, before, rc)          # HB review I2: the child's firewall records, read every round; a firewall event stops (not a round)
            rounds[-1]['third_child_records'] = {k: dict(path=str(v[0]), sha256=hb.sha(v[0])) for k, v in child_records.items()}
        if not (rounds and (rounds[-1]['consistent'] or (rounds[-1]['third_certificate'] or {}).get('passed') is True and (rounds[-1]['third_certificate'] or {}).get('output_state_hash') == rounds[-1]['state_hash'])):
            raise NotFinalized('rounds', f'the review certificate still differs after {MAX_ROUNDS} rounds (D111; no third certification after the last one, HB review m5): to the lead and BZ')
        review_doc = dict(consistent=rounds[-1]['consistent'], differences=rounds[-1]['differences'], review=rounds[-1]['review'],
                          decision=dict(process=cert.get('process'), state_hash=cert.get('state_hash'), status=cert.get('status'), min_jd_lower_bound=cert.get('min_jd_lower_bound')),
                          third_certificate=rounds[-1]['third_certificate'], rounds=rounds)
        spy_record = None
        if spy is not None:
            spy.remove()
            spy_record = spy.report()
            hb.write_once(rev/OPENED, spy_record)
            spy = None
            if not spy_record['passed']:
                raise NotFinalized('spy', f"the spy saw a forbidden read or no positive condition: {spy_record['breaches'] or spy_record['forbidden_access']}", code=STOP_EXIT['spy'])
        doc = dict(schema=hb.SCHEMA.format('review'), mode=mode.name, pair=pair, batch=str(batch), redo=redo, finalized=True,
                   **finalized_part(out, line=line, review_doc=review_doc), inputs=kept['inputs'], continued=kept.get('continued'), admission=admission, identity=identity.get('repo'),
                   settings_sha256=ctx.get('settings_sha256'), code_head=ctx.get('code_head'), review_certificate_in=line.get('review_certificate_in'),
                   context_settings_are_pinned=ctx.get('settings_sha256') == hb_method.SETTINGS_SHA256, spy=None if spy_record is None else dict(file=OPENED, passed=spy_record['passed']),
                   unit=dict(start=start_probe, end=probe('end')), utc=hb.utc(),
                   rule='C22.H1-2..5: the review certificate (C22.4-1, CPU FP64) against cert.json; the mechanical rule of C22.3-9 (d); no metric, no label')
        path = hb.write_once(rev/REVIEW_RECORD, doc)
        hb.write_terminal(out, pair=pair, status=line['line_status'], written_by='experiments/hb/hb_review.py (step 5: finalized)', output_dir=he.OUTPUT_DIR,
                          review=dict(path=f'review/{REVIEW_RECORD}', sha256=hb.sha(path)))
        return 0
    except NotFinalized as exc:
        return finish_not_finalized(rev, exc, mode, pair, rounds=rounds, kept=kept, spy=spy)
    except spying.Breach as exc:
        return finish_not_finalized(rev, NotFinalized('spy', str(exc), code=STOP_EXIT['spy']), mode, pair, rounds=rounds, kept=kept, spy=spy)
    except hc.Refused as exc:
        return finish_not_finalized(rev, NotFinalized(f'refused:{exc.stage}', str(exc)), mode, pair, rounds=rounds, kept=kept, spy=spy)
    except R.admission_refusals() as exc:
        return finish_not_finalized(rev, NotFinalized('admission', f'{type(exc).__name__}: {exc}'), mode, pair, rounds=rounds, kept=kept, spy=spy)
    finally:
        if spy is not None:
            spy.remove()


def finish_not_finalized(rev, exc, mode, pair, *, rounds=(), kept=None, spy=None):
    """The pair is not finalized: its line status unchanged, review_not_finalized[_k].json (why, the redo kind a later authorization names), the spy's record; the queue stops (C22.H1-2, -4)."""
    spy_record = None
    if spy is not None:
        spy.remove()
        spy_record = spy.report()
        hb.write_once(hb.numbered(rev, 'opened_files_not_finalized'), spy_record)
    out = rev.parent
    superseded = sorted((p.name for p in out.glob('output_superseded_*') if p.is_dir() and p.name.rsplit('_', 1)[1].isdigit()), key=lambda n: int(n.rsplit('_', 1)[1]))     # by its number (n4)
    continue_from = None
    if not (out/he.OUTPUT_DIR/'line.json').is_file() and superseded:
        k = int(superseded[-1].rsplit('_', 1)[1])
        rec = out/superseded[-1]/'third_certificate.json'
        continue_from = dict(k=k, record=str(rec), record_sha256=hb.sha(rec) if rec.is_file() else None)
    doc = dict(schema=hb.SCHEMA.format('review_not_finalized'), seq=next_stop_seq(pair, mode.root), mode=mode.name, pair=pair, kind=exc.kind, reason=exc.reason, code=exc.code, extra=exc.extra, rounds=list(rounds), kept=kept,
               spy=spy_record and dict(passed=spy_record['passed'], breaches=spy_record['breaches']), continue_from=continue_from, evidence=evidence_rows(out), utc=hb.utc(),
               rule='C22.H1-2, -4: the pair is not finalized, its line status unchanged, the queue stops; a redo only with a committed authorization (D111 (i), D123 (b))')
    path = hb.write_once(hb.numbered(rev, NOT_FINALIZED.removesuffix('.json')), doc)
    copy_to_home(mode, pair, path)
    return exc.code


def copy_to_home(mode, pair, path):
    """Item 10: a byte copy of one step-5 record under home_records(pair) (same name, written once; the copy's sha256 equal to the cell's).  A withdrawal (withdraw) and BZ's decisions name these
    copies: the cell moves to D:, they do not."""
    home = home_records(pair, mode.root)
    home.mkdir(parents=True, exist_ok=True)
    with open(path, 'rb') as stream:
        data = stream.read()
    with open(home/Path(path).name, 'xb') as stream:
        stream.write(data)
    if hb.sha(home/Path(path).name) != hb.sha(path):
        raise OSError(f'the /home copy of {path} differs')
    return home/Path(path).name


def home_records(pair, root=None):
    """The step-5 records of a pair kept on /home (the lead's item 10; decision_log 2026-11-10 ruling (6)): <the review's root>/<pair>/ (holdout_B_v1/review/<pair>/ on the hold-out), beside the
    batch's records; the cell moves to D:, this directory does not."""
    return Path(hb.review_root() if root is None else root)/pair


def keep_on_home(mode, pair, out):
    """Item 10: byte copies of the cell's review/review.json and terminal.json under home_records (written once), and records.json with both sha256 and the cell's paths; the copies are checked
    against the cell's files.  BZ's (甲) on a step-5 hx_implementation_error names the copy (home_records(pair)/review.json) as its record, so that it holds after the move."""
    home = home_records(pair, mode.root)
    home.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, src in (('review.json', Path(out)/'review'/REVIEW_RECORD), ('terminal.json', hb.terminal_path(out))):
        dst = home/name
        with open(src, 'rb') as stream:
            data = stream.read()
        with open(dst, 'xb') as stream:
            stream.write(data)
        rows[name] = dict(copy=str(dst), source=str(src), sha256=hb.sha(dst), same=hb.sha(dst) == hb.sha(src))
    if not all(r['same'] for r in rows.values()):
        raise NotFinalized('home_records', f'the /home copies differ from the cell\'s records: {rows}')
    hb.write_once(home/'records.json', dict(schema=hb.SCHEMA.format('home_records'), pair=pair, files=rows, utc=hb.utc(),
                                            rule='item 10 (decision_log 2026-11-10 (6)): the step-5 records BZ\'s decisions point to, kept on /home; the cell moves to D:'))
    return rows


def implementation_error(ctx, mode, rev, pair, *, kind, reason, rounds, kept, spy, exception_class=None):
    """C22.5-6 (a) or (b) in the (d) chain (C22.H1-2 修订 7c): line_status hx_implementation_error, no output; the review record and the terminal record say so; the evidence by sha256; both copied to
    /home (keep_on_home, item 10); the queue stops."""
    out = Path(ctx['out'])
    spy_record = None
    if spy is not None:
        spy.remove()
        spy_record = spy.report()
        hb.write_once(rev/OPENED, spy_record)
    doc = dict(schema=hb.SCHEMA.format('review'), mode=mode.name, pair=pair, line_status=he.IMPLEMENTATION, finalized=False,
               implementation_error=dict(kind=kind, reason=reason, exception_class=exception_class, possibly_infrastructure=exception_class in ('OSError', 'MemoryError') or None,
                                         note=('an OSError or MemoryError: possibly infrastructure, the lead decides (HB review m4); the class is not changed here'
                                               if exception_class in ('OSError', 'MemoryError') else None)), rounds=list(rounds),
               inputs=(kept or {}).get('inputs'), evidence=evidence_rows(out), spy=None if spy_record is None else dict(file=OPENED, passed=spy_record['passed']), utc=hb.utc(),
               rule=f'C22.5-6 ({kind}), C22.H1-2 修订 7c: no output; the queue stops; BZ decides before step 11')
    path = hb.write_once(rev/REVIEW_RECORD, doc)
    hb.write_terminal(out, pair=pair, status=he.IMPLEMENTATION, written_by=f'experiments/hb/hb_review.py (step 5: C22.5-6 ({kind}))', review=dict(path=f'review/{REVIEW_RECORD}', sha256=hb.sha(path)),
                      reason=reason, evidence=evidence_rows(out))
    keep_on_home(mode, pair, out)
    return STOP_EXIT['implementation']


# ---- the queue ---------------------------------------------------------------------------------------------------------------------------------------
def step4_state(pair, pair_set='holdout'):
    """'output' (output/line.json of an output status, and the method queue's step4.json of the pair went on: the certified check passed), 'terminal' (the cell's step-4 terminal.json: no output),
    None (step 4 not done for this pair, or it stopped the batch)."""
    cell_dir = hb.cell(pair, pair_set)
    if hb.terminal_path(cell_dir).is_file() and not (cell_dir/'review'/REVIEW_RECORD).is_file():
        return 'terminal'
    line = hb.read_json(cell_dir/he.OUTPUT_DIR/'line.json')
    steps = [hb.read_json(hb.record_dir(ARM, pair, pair_set)/name) for name in ('step4_resume.json', hb_method.STEP4)]
    step = next((s for s in steps if s is not None), None)
    if line is not None and line.get('line_status') in he.OUTPUT_STATUSES and step is not None and step.get('went_on') is True:
        return 'output'
    return None


def input_manifest(pairs, pair_set='holdout'):
    """C22.H1-8: every pair's files by sha256 (the reviewed ones; a step-4 terminal pair: its terminal.json and line.json only, recorded)."""
    out = {}
    for pair in pairs:
        cell_dir = hb.cell(pair, pair_set)
        st = step4_state(pair, pair_set)
        names = hb.INPUT_FILES if st == 'output' else ('terminal.json', 'line.json')
        out[pair] = dict(step4=st, cell=str(cell_dir), files=hb.file_rows(cell_dir, names))
    return out


def supplement(mode, pair, redo, auth):
    """C22.H1-8 修订 7b: before a redo, the cell's state now (every file under output*, fallback*after_third*, review/), chained to the input manifest; the redo kind (review again, or continue the chain)."""
    cell_dir = hb.cell(pair, pair_set_of_mode(mode))
    stop = [path for _, path in by_seq((cell_dir/'review').glob('review_not_finalized*.json'))]          # HB diff review n4: by the records' sequence numbers, never mtime or name
    last = hb.read_json(stop[-1]) if stop else None
    files = {name: hb.sha(cell_dir/name) for name in hb.INPUT_FILES if (cell_dir/name).is_file()}
    doc = dict(schema=hb.SCHEMA.format('supplement'), pair=pair, redo=redo, authorization=dict(path=auth.path, sha256=auth.sha256), chained_to=dict(path=str(mode.manifest()), sha256=hb.sha(mode.manifest())),
               stop_record=None if not stop else dict(path=str(stop[-1]), sha256=hb.sha(stop[-1]), kind=(last or {}).get('kind')),
               continue_from=(last or {}).get('continue_from'), evidence=evidence_rows(cell_dir), utc=hb.utc(),
               pairs={pair: dict(step4='output', cell=str(cell_dir), files=files)})
    path = mode.root/'supplements'/f'{pair}_{redo}.json'
    hb.write_once(path, doc)
    return path, doc


def unit_stop_record(pair, reading, root=None):
    """The lead's item 13: the review unit ended without its own record: review_unit_stopped[_k].json under home_records(pair, root) (the unit's reading), the record a rejudgment of the review
    judges.  `root`: the review's root (None: the queue's, hb_core.review_root(); the controlled limit experiment's second batch writes into its trial directory, A5a)."""
    home = home_records(pair, root)
    home.mkdir(parents=True, exist_ok=True)
    return hb.write_once(hb.numbered(home, 'review_unit_stopped'), dict(schema=hb.SCHEMA.format('review_unit_stopped'), seq=next_stop_seq(pair, root), pair=pair, reading=reading, utc=hb.utc(),
                                                                      rule='D123 (b): the review unit ended without review.json or review_not_finalized; to the lead'))


def unit_end_stop_code(reading):
    """D123 (b) (HB review m7 (ii)): the queue's STOP code after a review unit that left neither review.json nor a new review_not_finalized: a limit or OOM evidence (unit_limit, infra_failure)
    -> 20; anything else (needs_lead_rejudgment, no reading) -> 25.  Not a method failure either way; a redo only with a committed authorization."""
    return hb.EXIT['infra'] if (reading or {}).get('status') in ('infra_failure', 'unit_limit') else hb.EXIT['rejudge']


STOP_PATTERNS = ('review_not_finalized*.json', 'review_unit_stopped*.json')


def _stop_records(pair, root=None):
    home = home_records(pair, root)
    return [q for pattern in STOP_PATTERNS for q in home.glob(pattern)] if home.is_dir() else []


def _seq(path):
    doc = hb.read_json(path)
    seq = doc.get('seq') if isinstance(doc, dict) else None
    return seq if isinstance(seq, int) and not isinstance(seq, bool) and seq >= 1 else None


def next_stop_seq(pair, root=None):
    """HB diff review n4: the sequence number of the pair's next stop record (review_not_finalized, review_unit_stopped; one count for both, on /home): one more than the largest there (1 for the
    first); a record without a valid number is passed over here (writing a stop record never fails on it) and refused by by_seq when the newest is looked for."""
    return 1+max((q for q in map(_seq, _stop_records(pair, root)) if q is not None), default=0)


def by_seq(paths):
    """[(seq, path)] by the records' sequence numbers (HB diff review n4: never mtime, which a copy, a restore or a touch changes; never the name); Refused when a record has no sequence
    number (an integer >= 1) or two have the same one."""
    rows = [(_seq(q), q) for q in paths]
    bad = [str(q) for n, q in rows if n is None]
    numbers = [n for n, _ in rows if n is not None]
    twice = sorted({n for n in numbers if numbers.count(n) > 1})
    if bad or twice:
        raise hc.Refused('stop_record', f'stop records without a sequence number {bad} or sharing one {twice}: the newest is not known; to the lead')
    return sorted(rows, key=lambda r: r[0])


def last_stop_record(pair, root=None):
    """The newest of the pair's /home stop records (review_not_finalized*, review_unit_stopped*) by their sequence numbers (by_seq), or None."""
    rows = by_seq(_stop_records(pair, root))
    return rows[-1][1] if rows else None


SPY_STOP_KINDS = ('spy', 'third_spy')                         # NotFinalized kinds of a spy that stopped a read (the review unit's ReviewSpy, the third certification's)


def lpba_spy_stop_problem(pair, pair_set):
    """LP-B diff review minor 3: an LPBA pair whose review or third certification was stopped by a spy -- a spy breach record in the cell (lpba_arms.spy_breach_problem: the third
    certification's ArmSpy) or a stop record of kind spy / third_spy -- is not redone by an authorization: to BZ.  None otherwise (and for the hold-out: unchanged)."""
    if not hb.is_lpba(pair_set):
        return None
    problem = hb.lpba().spy_breach_problem(hb.cell(pair, pair_set))
    if problem:
        return problem
    kinds = [(str(q), (hb.read_json(q) or {}).get('kind')) for q in sorted(_stop_records(pair, hb.review_root(pair_set)))]
    spied = [q for q, kind in kinds if kind in SPY_STOP_KINDS]
    return None if not spied else f'{hb.lpba().SPY_BREACH_NO_RERUN}: the stop records {spied} name a spy'


def review_rejudgment(pair, pair_set='holdout'):
    """A committed rejudgment record (arm review) of the pair's last /home stop record, any judgment: a trace only (decision_log 2026-11-10 N1; prereg revision 2 item 18: the review unit's
    needs_lead_rejudgment goes by D123 (b) -- a redo -- or a withdrawal; neither side lets a rejudgment write the pair).  (found or None, problems)."""
    record = last_stop_record(pair, hb.review_root(pair_set) if hb.is_lpba(pair_set) else None)
    if record is None:
        return None, []
    if hb.is_lpba(pair_set):
        return hc.rejudgment('review', pair, record=record, pair_set=pair_set)
    return hc.rejudgment('review', pair, record=record)


def review_decision(pair, pair_set='holdout'):
    """BZ's (甲) on a step-5 hx_implementation_error (runs/20261109-align item 3): the committed no_resume record of tool hb, arm review, whose record is the /home copy of the pair's review.json
    (home_records(pair)/review.json, item 10: it holds after the cell moved to D:) and whose reason names the case of C22.5-6 and (甲) only; None when there is none that holds."""
    path = home_records(pair, hb.review_root(pair_set) if hb.is_lpba(pair_set) else None)/REVIEW_RECORD
    doc = hb.read_json(path) or {}
    kind = (doc.get('implementation_error') or {}).get('kind')
    withdrawn = doc.get('withdrawn')
    if withdrawn is not None:                           # item 11: the withdrawal's own no_resume record decided it (withdraw)
        return withdrawal_authorization(pair, Path(withdrawn['record']['path']), rel=withdrawn['authorization']['path'], pair_set=pair_set)
    return hb.bz_decision_auth('review', pair, path, line_status=he.IMPLEMENTATION, kind=kind, pair_set=pair_set)          # A5b review I2: the one selection (hb_core.bz_decision_auth)


WITHDRAW_KIND = 'b'                                     # the lead's item 11: a withdrawal (D111 (ii), D123 (ii)) is recorded as C22.5-6 (b), its reason naming (b) and (甲)


def withdrawal_authorization(pair, record, *, rel=None, pair_set='holdout'):
    """The committed no_resume record (tool hb, arm review, holdout) of a withdrawal: its record the /home copy `record` of the pair's last review_not_finalized record with that sha256, its reason
    naming C22.5-6 (b) and (甲) only; `rel` (relative to the code tree) when a given file is meant.  None when there is none."""
    want = None if rel is None else (Path(hc.AUTH_ROOT)/rel).resolve()
    return hb.bz_decision_auth('review', pair, record, line_status=he.IMPLEMENTATION, kind=WITHDRAW_KIND, accept=lambda auth: want is None or Path(auth.path).resolve() == want, pair_set=pair_set)


def withdraw(pair, rel, *, mode=None):
    """The lead's item 11: BZ withdraws the pair's output (D111 (ii), D123 (ii); C22.5-6): with the committed no_resume record `rel` (tool hb, arm review; reason C22.5-6 (b) and (甲); its record the
    /home copy of the pair's last review_not_finalized record), the cell gets review/review.json (line_status hx_implementation_error, finalized false, implementation_error kind b, withdrawn: the
    authorization and the record) and terminal.json (hx_implementation_error, no output, rerun_record the authorization); both copied to /home (item 10).  Nothing else is changed or deleted (the
    output/ directory and every record stay as evidence).  Refused: a cell with terminal.json or review.json already (only a pair whose review stopped without a record is pending), no
    review_not_finalized record, an authorization that is not committed or whose reason or record does not hold."""
    mode = holdout_mode(None) if mode is None else mode
    pair_set = pair_set_of_mode(mode)
    if hb.is_lpba(pair_set) and hb.lpba_resume_problem(pair_set):
        raise hc.Refused('withdraw', hb.lpba_resume_problem(pair_set))
    cell_dir = hb.cell(pair, pair_set)
    rev = cell_dir/'review'
    if hb.terminal_path(cell_dir).exists() or (rev/REVIEW_RECORD).exists():
        raise hc.Refused('withdraw', f'{pair}: terminal.json or review/review.json is there: only a pair whose review stopped without them is withdrawn')
    home_copy = last_stop_record(pair, mode.root)
    if home_copy is None:
        raise hc.Refused('withdraw', f'{pair}: no stop record on /home (review_not_finalized or review_unit_stopped): nothing pending to withdraw')
    last = rev/home_copy.name                           # a review_not_finalized copy has its original in the cell; a review_unit_stopped record is written on /home only
    if home_copy.name.startswith('review_not_finalized') and (not last.is_file() or hb.sha(home_copy) != hb.sha(last)):
        raise hc.Refused('withdraw', f'{home_copy}: not the /home copy of {last}')
    if not last.is_file():
        last = home_copy
    auth = withdrawal_authorization(pair, home_copy, rel=rel, pair_set=pair_set)
    if auth is None:
        raise hc.Refused('withdraw', f'{rel}: not a committed no_resume record of tool hb, arm review, for {pair}, whose record is {home_copy} (its sha256 now) and whose reason names '
                                     f'C22.5-6 ({WITHDRAW_KIND}) and (甲) only')
    authorization = dict(path=str(Path(auth.path).resolve().relative_to(Path(hc.AUTH_ROOT).resolve())), sha256=auth.sha256)
    stop = hb.read_json(last) or {}
    reason = f"withdrawn by BZ's decision (C22.5-6 ({WITHDRAW_KIND}), (甲)): {auth.document.get('reason')}; the review stopped: {stop.get('kind')}"
    doc = dict(schema=hb.SCHEMA.format('review'), mode=mode.name, pair=pair, line_status=he.IMPLEMENTATION, finalized=False,
               implementation_error=dict(kind=WITHDRAW_KIND, reason=reason, exception_class=None, possibly_infrastructure=None, note=None),
               withdrawn=dict(authorization=authorization, record=dict(path=str(home_copy), sha256=hb.sha(home_copy)), stopped=dict(path=str(last), kind=stop.get('kind'), code=stop.get('code'))),
               evidence=evidence_rows(cell_dir), utc=hb.utc(),
               rule='the lead\'s item 11; D111 (ii), D123 (ii): BZ withdrew the output; no output, every file kept')
    path = hb.write_once(rev/REVIEW_RECORD, doc)
    hb.write_terminal(cell_dir, pair=pair, status=he.IMPLEMENTATION, written_by='experiments/hb/hb_review.py withdraw (BZ\'s withdrawal)', review=dict(path=f'review/{REVIEW_RECORD}', sha256=hb.sha(path)),
                      rerun_record=authorization, reason=reason, evidence=evidence_rows(cell_dir))
    keep_on_home(mode, pair, cell_dir)
    return hb.terminal_path(cell_dir)


def queue(*, commit=None, redos=None, system=None, echo=print, code_run=None, pair_set='holdout'):
    system = hb.System() if system is None else system
    redos = dict(redos or {})
    record, problems = hb_method.checks_of(pair_set, commit, code_run)
    if problems:
        echo(f'refused at the start: {problems}')
        return hb.EXIT['refused']
    batch = hb.new_batch_dir('review', pair_set)
    log = hb.Log(batch, echo)
    record.update(schema=hb.SCHEMA.format('batch'), kind='review', utc_start=hb.utc(), disk_start=hc.disk_reading(), cells=[], max_rounds=MAX_ROUNDS,
                  gate=dict(mem_gib=hb.GATE_MEM_GIB, wait_seconds=hb.GATE_WAIT_SECONDS['review'], rule='D126 (a): 30 minutes, written in the code'), unit=dict(he.EVALUATION_UNIT))
    code = 0
    try:
        mode = mode_of(pair_set, record['code']['commit'])  # HB diff review NB1: in the try, so that whatever stops building the policy is in batch.json
        plan = hb.rows_of(pair_set)
        states = {p: hb_method.state(p, pair_set) for p in plan}
        missing = [(p, st) for p, st in states.items() if st not in ('final', 'done')]
        record['step4_states'] = states
        if missing:                                     # HB review I1: D110's "every pair has its terminal record" is hb_method's own reading: a pair in hx_provenance_rejected (P20) or in
            # hx_implementation_error without BZ's (甲) is stopped, not written; nothing of step 5 is written before every pair is final or done
            raise hb.Stop(hb.EXIT['refused'], f'D110: step 4 has not ended for {len(missing)} pairs (hb_method.state not final or done): {missing[:5]}')
        lock = mode.root/'review.lock'
        if (mode.root/'finalized_outputs.json').exists():
            raise hb.Stop(hb.EXIT['refused'], f"{mode.root/'finalized_outputs.json'} exists: step 5 is finished; nothing runs again")
        if not lock.exists():
            if redos:
                raise hb.Stop(hb.EXIT['refused'], 'a redo before the batch lock exists')
            manifest = dict(schema=hb.SCHEMA.format('input_manifest'), batch=str(batch), code=record['code'], settings_sha256=hb_method.SETTINGS_SHA256, pairs=input_manifest(plan, pair_set), utc=hb.utc())
            if hb.is_lpba(pair_set):
                manifest['pair_set'] = pair_set         # LP-B review I1: only an LPBA mode's manifest has the key (the hold-out's input_manifest.json keys unchanged)
            hb.write_once(mode.manifest(), manifest)
            hc.take_lock(lock, hc.lock_line(arm='review', batch=str(batch), code=record['code'].get('commit'), manifest_sha256=hb.sha(mode.manifest())))
        # the batch lock there (a later start after a STOP): the input manifest is not written again; a pair already locked and not finalized runs only with its redo authorization, a pair never
        # started runs; a pair finalized in hx_implementation_error needs BZ's decision first
        mode.batch_sha = (json.loads(lock.read_text(encoding='utf-8').splitlines()[0]) or {}).get('code')
        if mode.batch_sha != record['code'].get('commit'):
            raise hb.Stop(hb.EXIT['refused'], f"the batch lock's commit {mode.batch_sha} is not this checkout's {record['code'].get('commit')} (C22.H0-4: one checkout from step 4 to 7)")
        for pair in plan:
            row = dict(pair=pair)
            record['cells'].append(row)
            finalized = (hb.cell(pair, pair_set)/'review'/REVIEW_RECORD).is_file()
            if finalized and not hb.terminal_path(hb.cell(pair, pair_set)).is_file():      # HB review m7 (i): a unit killed between the two records; to the lead (D123)
                raise hb.Stop(hb.EXIT['rejudge'], f'{pair}: review/review.json without terminal.json (the unit stopped between the two): D123, to the lead; nothing written here')
            st = step4_state(pair, pair_set)
            if finalized and (hb.read_json(hb.cell(pair, pair_set)/'review'/REVIEW_RECORD) or {}).get('line_status') == he.IMPLEMENTATION:
                decided = review_decision(pair, pair_set)
                if decided is None:
                    raise hb.Stop(hb.EXIT['implementation'], f'{pair}: hx_implementation_error at step 5 (C22.5-6): BZ decides first -- (甲) a committed no_resume record of tool hb, arm review, '
                                                             f'record {home_records(pair, mode.root)/REVIEW_RECORD} (the /home copy), its reason naming the case of C22.5-6 and (甲); (乙) needs a fixed code, i.e. a new pinned commit (C22.H0-4: '
                                                             'the batch stops, to the lead)')
                row.update(state='finalized', decision=dict(path=decided.path, sha256=decided.sha256))
                continue
            if finalized or st == 'terminal':
                row['state'] = 'finalized' if finalized else 'recorded'
                continue
            judged, _ = review_rejudgment(pair, pair_set)
            if judged is not None:                              # N1: a rejudgment of the review unit is a trace only; the pair goes on by a redo (resume) or a withdrawal (no_resume)
                row['rejudgment_trace'] = dict(path=judged['path'], sha256=judged['sha256'], judgment=judged['document']['judgment'])
            redo = None
            stop_record = last_stop_record(pair, mode.root)
            if mode.lock(pair).exists() or stop_record is not None:      # item 13: a unit killed before the pair's lock left its /home stop record: a redo needs an authorization too
                if pair not in redos:
                    raise hb.Stop(hb.EXIT['resume'], f'{pair}: its lock or its stop record is there and it is not finalized: it goes on only by a redo (--redo with a committed resume '
                                                     'authorization, D111 (i), D123 (b)) or a withdrawal (hb_review.py withdraw with a committed no_resume record)')
                rel = redos[pair]
                # the previous record: the newest stop record by its sequence number (HB diff review n4) -- the cell's original of a review_not_finalized, a review_unit_stopped on /home
                original = None if stop_record is None else hb.cell(pair, pair_set)/'review'/stop_record.name
                previous = (original if original is not None and stop_record.name.startswith('review_not_finalized') and original.is_file() else
                            stop_record if stop_record is not None else hb.cell(pair, pair_set)/'review'/NOT_FINALIZED)
                try:
                    if lpba_spy_stop_problem(pair, pair_set):       # LP-B diff review minor 3: a spy breach goes to BZ, never through a redo
                        raise hc.Refused('resume', lpba_spy_stop_problem(pair, pair_set))
                    if hb.is_lpba(pair_set) and hb.lpba_resume_problem(pair_set):
                        raise hc.Refused('resume', hb.lpba_resume_problem(pair_set))
                    auth = hc.load_resume(rel, tool=hb.TOOL, arm='review', pair=pair, pair_set=pair_set, lock=mode.lock(pair) if mode.lock(pair).exists() else stop_record,
                                          previous_record=previous, complete=lambda: (hb.cell(pair, pair_set)/'review'/REVIEW_RECORD).is_file())
                except hc.Refused as exc:
                    raise hb.Stop(hb.EXIT['refused'], f'{pair}: {exc}') from None
                redo = 1
                supplement(mode, pair, redo, auth)
            elif pair in redos:
                raise hb.Stop(hb.EXIT['refused'], f'{pair}: a redo for a pair that was never reviewed')
            while hb.hold_path(pair_set).exists():
                log('hold file present; waiting')
                system.sleep(60)
            try:
                row['gate'] = hb.gate_of(system, 'review', pair_set)
                row['disk'] = system.disk(*hb.UNIT_NEED_GIB['review'])
            except (hc.InfrastructureFailure, hc.Refused) as exc:
                raise hb.Stop(hb.EXIT['infra'], f'{pair}: the gate or the disks before the review unit (its output untouched): {exc}') from None
            unit_dir = hb.attempt_dir(hb.record_dir(ARM, pair, pair_set)/('unit_review' if redo is None else 'unit_review_redo'))
            hb.record_dir(ARM, pair, pair_set).mkdir(parents=True, exist_ok=True)
            inner = [HERE/'hb_review.py', 'unit', '--pair', pair, '--batch', str(batch), *(['--redo', str(redo)] if redo else []), *(['--pair-set', pair_set] if hb.is_lpba(pair_set) else [])]
            stops_before = set(p.name for p in (hb.cell(pair, pair_set)/'review').glob('review_not_finalized*.json'))
            log(f'{pair}: unit {hb.unit_name("review", pair, pair_set)}{" (redo)" if redo else ""}')
            reading = hb.run_unit('review', pair, inner, system, batch=batch, unit_dir=unit_dir, res=dict(he.EVALUATION_UNIT), gpu=False, code=hb.REPO, pair_set=pair_set)
            row['reading'] = dict(status=reading['status'], kind=reading['status_kind'], child_exit=reading['child_exit'])
            if (hb.cell(pair, pair_set)/'review'/REVIEW_RECORD).is_file():
                doc = hb.read_json(hb.cell(pair, pair_set)/'review'/REVIEW_RECORD)
                row['line_status'] = doc.get('line_status')
                if not hb.terminal_path(hb.cell(pair, pair_set)).is_file():                # HB review m7 (i): both records or the pair is not finalized
                    raise hb.Stop(hb.EXIT['infra'] if reading['status'] in ('infra_failure', 'unit_limit') else hb.EXIT['rejudge'],
                                  f"{pair}: review.json without terminal.json (the unit ended {reading['status']}, {reading['status_kind']}): D123, to the lead")
                if doc.get('line_status') == he.IMPLEMENTATION:
                    raise hb.Stop(hb.EXIT['implementation'], f'{pair}: hx_implementation_error in the (d) chain (C22.5-6 (a)): to BZ')
                if reading['status'] is not None:
                    log(f'{pair}: finalized, the unit reading {reading["status"]} recorded')
                log(f'{pair}: finalized {doc.get("line_status")}')
                continue
            new_stops = [q for _, q in by_seq(p for p in (hb.cell(pair, pair_set)/'review').glob('review_not_finalized*.json') if p.name not in stops_before)]
            if new_stops:
                last = hb.read_json(new_stops[-1]) or {}
                raise hb.Stop(int(last.get('code') or hb.EXIT['stop']), f"{pair}: not finalized ({last.get('kind')}: {last.get('reason')}); the unit reading {reading['status']}: to the lead")
            stopped = unit_stop_record(pair, reading, mode.root if hb.is_lpba(pair_set) else None)              # the lead's item 13: a record on /home a rejudgment can judge
            raise hb.Stop(unit_end_stop_code(reading),             # HB review m7 (ii): a limit is D123's like an OOM
                          f"{pair}: the review unit ended without its record ({reading['status']}, {reading['status_kind']}, exit {reading['child_exit']}): D123 (b), not a method failure; a redo only "
                          'with a committed authorization')
        record['finalized'] = finalize(plan, batch, record, pair_set)
    except hb.Stop as exc:
        code = exc.code
        record['stop'] = dict(code=code, reason=str(exc))
        log(f'STOP ({code}): {exc}; nothing rerun; to the lead')
    except hc.Refused as exc:
        code = hb.EXIT['refused']
        record['stop'] = dict(code=code, reason=str(exc))
        log(f'STOP ({code}): {exc}')
    except Exception as exc:                            # noqa: BLE001 - HB review m11: the batch record is written whatever stopped the queue
        code = hb.EXIT['stop']
        record['stop'] = dict(code=code, reason=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()[-4000:])
        log(f'STOP ({code}): an exception of the queue: {type(exc).__name__}: {exc}')
    record.update(utc_end=hb.utc(), disk_end=hc.disk_reading(), exit=code)
    hb.write_once(batch/'batch.json', record)
    if code == 0:
        log('end')
    return code


def finalize(plan, batch, record, pair_set='holdout'):
    """C22.H1-3: finalized_outputs.json once every pair is finalized (or recorded at step 4).  LP-B: an LPBA mode's under its review root (C26.6-6: the method output fixed)."""
    pairs = {}
    for pair in plan:
        cell_dir = hb.cell(pair, pair_set)
        rev = cell_dir/'review'/REVIEW_RECORD
        term = hb.terminal_path(cell_dir)
        doc = hb.read_json(rev)
        term_doc = hb.read_json(term) or {}
        decided, step = None, None
        if (doc or {}).get('line_status') == he.IMPLEMENTATION:
            decided, step = review_decision(pair, pair_set), 5                # BZ's (甲) on a step-5 hx_implementation_error (the queue checked it is there)
        elif doc is None and term_doc.get('line_status') in (he.IMPLEMENTATION, he.REJECTED):
            decided, step = hb_method.bz_declined(pair, pair_set), 4          # HB review I1: BZ's (甲) on a step-4 hx_implementation_error; item 13: BZ's no rerun of a P20 cell (the D110 gate checked)
        pairs[pair] = dict(cell=str(cell_dir), terminal=dict(path=str(term), sha256=hb.sha(term)) if term.is_file() else None, review=dict(path=str(rev), sha256=hb.sha(rev)) if rev.is_file() else None,
                           line_status=(hb.read_json(term) or {}).get('line_status'), part=None if doc is None else {k: doc.get(k) for k in ('output', 'state_hash', 't88_repair', 'fallback', 'superseded', 'evidence')},
                           bz_decision=None if decided is None else dict(step=step, path=decided.path, sha256=decided.sha256))     # not review.json's decision (the decision certificate, C22.H1-5)
    doc = dict(schema=hb.SCHEMA.format('finalized_outputs'), batch=str(batch), code=record.get('code'), frozen_tree=dict(path=str(he.core.FROZEN), commit=he.core.FROZEN_COMMIT), environment=R.env_record(), utc_start=record.get('utc_start'), pairs=pairs, utc=hb.utc(),
               rule='C22.H1-3: from now on these files are not changed; later steps know them by these sha256 (after the move to D:, by the move records)')
    if hb.is_lpba(pair_set):
        doc['pair_set'] = pair_set
    path = hb.review_root(pair_set)/'finalized_outputs.json'
    hb.write_once(path, doc)
    return dict(path=str(path), sha256=hb.sha(path))


# ---- the CLI -----------------------------------------------------------------------------------------------------------------------------------------
def unit_main(pair, batch, redo=None, pair_set='holdout'):
    ctx = R.read_context(hb.cell(pair, pair_set)/'context.json')
    problem = R.cli_context_problem(ctx)
    if problem or hold.pair_set_of(ctx) != pair_set or ctx['pair'] != pair:
        print(f'refused: {problem or f"not the {pair_set} context of this pair"}', file=sys.stderr)
        return 2
    lock = hb.review_root(pair_set)/'review.lock'
    batch_sha = (json.loads(lock.read_text(encoding='utf-8').splitlines()[0]) or {}).get('code') if lock.is_file() else None
    redo_doc = hb.read_json(hb.review_root(pair_set)/'supplements'/f'{pair}_{redo}.json') if redo else None
    if redo_doc is not None and redo_doc.get('continue_from') and not (Path(ctx['out'])/he.OUTPUT_DIR/'line.json').is_file():
        redo_doc = dict(redo_doc, **{'continue': redo_doc['continue_from']})
    return review_pair(ctx, mode_of(pair_set, batch_sha), batch=batch, redo=redo, redo_doc=redo_doc)




def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    dr = sub.add_parser('dry-run')
    q = sub.add_parser('queue')
    q.add_argument('--commit', default=None)
    q.add_argument('--redo', action='append', default=[])
    ls = sub.add_parser('launch-string')
    ls.add_argument('--commit', default=None)
    u = sub.add_parser('unit')
    u.add_argument('--pair', required=True)
    u.add_argument('--batch', required=True)
    u.add_argument('--redo', type=int, default=None)
    t = sub.add_parser('third-cert')
    t.add_argument('--context', required=True)
    t.add_argument('--batch', required=True)
    t.add_argument('--mode', choices=('holdout', 'dev')+hold.LPBA_PAIR_SETS, required=True)
    t.add_argument('--root', required=True)
    w = sub.add_parser('withdraw')
    w.add_argument('--pair', required=True)
    w.add_argument('--authorization', required=True)
    w.add_argument('--commit', default=None)
    for p in (dr, q, ls, u, w):
        p.add_argument('--pair-set', choices=hb_method.PAIR_SET_CHOICES, default='holdout', help='LP-B: an LPBA40 mode (lpba40_test, lpba40_feas); the hold-out by default')
    dv = sub.add_parser('dev-unit')
    dv.add_argument('--cell', required=True)
    dv.add_argument('--root', required=True)
    dv.add_argument('--batch-sha', default=None)
    o = ap.parse_args(argv)
    if o.command == 'dry-run':
        record, problems = hb_method.checks_of(o.pair_set)
        print(json.dumps(dict(record, problems=problems), indent=1, default=str))
        return hb.EXIT['refused'] if problems else 0
    if o.command == 'launch-string':
        args = ['queue', *(['--commit', o.commit] if o.commit else []), *(['--pair-set', o.pair_set] if o.pair_set != 'holdout' else [])]
        print(' '.join(hb.queue_launch('review', args, pair_set=o.pair_set)))
        return 0
    if o.command == 'queue':
        return queue(commit=o.commit, redos=hb_method.parse_resumes(o.redo), pair_set=o.pair_set)
    if o.command == 'unit':
        return unit_main(o.pair, o.batch, o.redo, o.pair_set)
    if o.command == 'withdraw':
        _, problems = hb_method.checks_of(o.pair_set, o.commit)
        if problems:
            print(f'refused at the start: {problems}', file=sys.stderr)
            return hb.EXIT['refused']
        try:
            print(withdraw(o.pair, o.authorization, mode=mode_of(o.pair_set, None)))
        except (hc.Refused, FileExistsError) as exc:
            print(f'refused: {exc}', file=sys.stderr)
            return hb.EXIT['refused']
        return 0
    if o.command == 'third-cert':                      # HB diff review n3: a refusal before the spy (nothing of the pair opened) exits THIRD_REFUSED_EXIT, never 2 (argparse's) or 1 (a crash)
        try:
            ctx = R.read_context(o.context)
            problem = R.cli_context_problem(ctx)
            if problem:
                print(f'refused: {problem}', file=sys.stderr)
                return THIRD_REFUSED_EXIT
            if o.mode == 'holdout':
                if hold.pair_set_of(ctx) != 'holdout':
                    print('refused: not a hold-out context', file=sys.stderr)
                    return THIRD_REFUSED_EXIT
                mode = holdout_mode(None)
            elif o.mode in hold.LPBA_PAIR_SETS:                 # LP-B (C26.4 L47): the child of an LPBA review admits the pair again in its own process (run_hx.third_cert_main's LPBA branch)
                if hold.pair_set_of(ctx) != o.mode or Path(o.root).resolve() != Path(hb.review_root(o.mode)).resolve():
                    print(f'refused: not an {o.mode} context, or not its review root', file=sys.stderr)
                    return THIRD_REFUSED_EXIT
                mode = lpba_mode(o.mode, None)
            else:
                problem = dev_root_problem(o.root, ctx['out'])
                if problem or hold.pair_set_of(ctx) != 'dev':
                    print(f'refused: {problem or "not a development context"}', file=sys.stderr)
                    return THIRD_REFUSED_EXIT
                mode = Mode('dev', Path(o.root)/'review', None, dev_policy(o.root, Path(ctx['stage_dir'])/ctx['pair']), segment=Path(o.root))
        except (Exception, SystemExit) as exc:          # noqa: BLE001 - the context (read_context exits) or the policy, before the spy and anything of the pair: refused, traceback in third-cert.log
            print(f'refused before the spy: {type(exc).__name__}: {exc}', file=sys.stderr)
            traceback.print_exc()
            return THIRD_REFUSED_EXIT
        return third_child_main(ctx, mode)
    # dev-unit
    ctx = R.read_context(Path(o.cell)/'context.json')
    problem = R.cli_context_problem(ctx) or dev_root_problem(o.root, o.cell) or (None if hold.pair_set_of(ctx) == 'dev' and Path(ctx['out']).resolve() == Path(o.cell).resolve() else 'the cell is not its context\'s output')
    if problem:
        print(f'refused: {problem}', file=sys.stderr)
        return 2
    mode = Mode('dev', Path(o.root)/'review', o.batch_sha, dev_policy(o.root, Path(ctx['stage_dir'])/ctx['pair']), segment=Path(o.root))
    if not mode.manifest().is_file():
        hb.write_once(mode.manifest(), dict(schema=hb.SCHEMA.format('input_manifest'), mode='dev', pairs={ctx['pair']: dict(step4='output', cell=str(o.cell), files=hb.file_rows(o.cell, hb.INPUT_FILES))},
                                            utc=hb.utc()))
    return review_pair(ctx, mode, batch=str(mode.root))


if __name__ == '__main__':
    sys.exit(main())
