"""HP (runs/20261107-hp/任务书.md item 3, 4): the tissue layer P_T (ANTs Atropos, three classes) of the 100 subjects of the B-line hold-out, read from the hold-out
stage that prep_stage.py wrote, and its development check.  In the method Hx24-RF05-F1200-v4 the tissue layer serves the alignment QC only (C22.1-3).

The Atropos work is experiments/d0/atropos_seg.py's own, imported and not copied: the same commands (ImageMath TruncateImageIntensity 0.01 0.99 256 in the mask, then
Atropos -d 3 -i KMeans[3] -k Gaussian -c [5,0.001] -m [0.3,1x1x1] -r 0), ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS = 8, the mask image > 0, the QC (every class >= 5 % of the
mask, else STOP with the record kept), the class order by posterior-weighted mean intensity (CSF < GM < WM), the provenance (bayesreg.tissue_provenance.v1), checked by
validate_segmentation_provenance (layer tissue) and verify_segmentation_files: atropos_seg.segment, atropos_seg.load_zyx, atropos_seg.ANTS, PARAMS.
What changes, by wrapping (no line of experiments/d0 is changed):
  - the entries of a subject: [(pair, side, PairInput)] as atropos_seg.subject_images builds them (the stage's selection.json order, load_pair, fixed and moving only),
    but one subject at a time (StageEntries): subject_images loads every pair at once, about 5.5 GB of float64 images for the 50 hold-out pairs.  The manifests' ids
    and the guard on them are checked for every pair before the lock (StageEntries.precheck; review m4), the arrays loaded after it;
  - the admission: SG's unseal object (hp_core.admit_holdout: the 50 rows, H1-H4 never, load_unseal, bind_unseal, check_subject; no dev_subjects.json path); the
    decision of the development check (ATROPOS_DEVCHECK_DECISION; review I1); the stage (check_stage): selection.json exactly those rows in their order, completion.json
    complete, every file of its SHA256SUMS unchanged, and the data preparation's prep_record.json passed, of the same unseal record and the same SHA256SUMS (review m3);
    load_pair takes the bound allowlist;
  - the tissue-layer check inside atropos_seg.segment: a pair with a sealed subject needs the bound allowlist load_pair admitted it through (C23.4-8, D103), which
    segment() does not pass; tissue_layer_check() hands it in for the length of the call (atropos_seg's module name validate_segmentation_provenance bound to the same
    function with allowlist=<that allowlist>, restored after), and the provenance is checked again here with it (both sides of every pair, the files' sha256);
  - the roots: the stage holdout_common.STAGE_ROOT, the output holdout_common.ATROPOS_ROOT/<subject>/ (new), one lock holdout_common.LOCK_ROOT/hp_atropos.lock
    (O_EXCL, after every check), the unit tag-hb-atropos, holdout_common's machine gate (hi_core's, MemAvailable, load); after a failure a resume only with a committed
    authorization (holdout_common.load_resume, at most once; the failed attempt kept whole as <atropos>.attempt1);
  - the order of the subjects: the pair list's rows, fixed then moving (prereg §5.2: rows in order);
  - the spy (hp_core.Spy) over the whole run: the id lists, the record, the allowlist, the decision log, the decision's records, the stage's files, the run's own
    directory and locks; the test-sandbox guard (hp_core.guard_write) first;
  - the code (A5a; contract C24.3 rule one): besides one clean tree, the pinned commit's (hp_core.check_pinned in main; run_holdout refuses a code record without it; the lock
    line's code_commit is P1); the decision's code_commit and rule two from it (hp_core.check_decision).
A subject in two pairs: atropos_seg.segment checks that both arrays and both source sha256 agree (as it does for the development stage); the hold-out pairs share no
subject (prereg §1), so this never fires there; the tests cover it.

    python -B experiments/holdout/atropos_holdout.py run --dry-run --out DIR [--unseal PATH] [--resume PATH]
    python -B experiments/holdout/atropos_holdout.py run --unseal splits/bayesreg_v1/unseal_holdout_B_v1.json [--resume splits/bayesreg_v1/holdout_B_v1_hp_atropos_resume_<k>.json]
        (unit tag-hb-atropos, after the stage and after the development check is decided)
    python -B experiments/holdout/atropos_holdout.py devcheck [--dry-run --out DIR]
        the development check (runs/20261107-hp/atropos_devcheck_plan.md; refused while ATROPOS_DEVCHECK_PLAN_SHA256 is None): this entry on OASIS_OAS1_0438_MR1 and
        OASIS_OAS1_0439_MR1 of stage_L2R_val_v1 (the development allowlist, unbound), compared with seg/atropos_v1/<subject>/: posteriors01-03, labels (bit for bit; else
        the largest difference and the agreement of the argmax and of the labels in the mask), and the inputs t1, mask, t1_trunc.  Once; no resume.
Exit codes: 0 done (or a dry run that passed), 2 usage, 8 refused, stopped by the spy, or a check failed (atropos_seg's STOPs -- QC, an exit of ANTs, the image
differing between pairs --, the loader's and the provenance checks; the stopped record kept), 1 any other failure (atropos_stopped.json when after the lock).
"""
import argparse
from contextlib import contextmanager
import functools
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
for _p in (HERE, EXPERIMENTS/'d0'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import holdout_common as hc  # noqa: E402
import hp_core as core  # noqa: E402
import atropos_seg  # noqa: E402   (experiments/d0/atropos_seg.py, imported, not copied)

from bayesreg.io.guard import SubjectSealed  # noqa: E402
from bayesreg.io.pair import ProvenanceRejected, validate_segmentation_provenance, verify_segmentation_files  # noqa: E402
from jd_curl.three_d.nifti import file_sha256  # noqa: E402
from jd_curl.validation import InputValidationError  # noqa: E402

SCHEMA = 'bayesreg.hp.atropos.v1'
SCRIPT = 'experiments/holdout/atropos_holdout.py'
TOOL, ARM = 'hp_atropos', 'atropos'             # the resume authorization's tool and arm (holdout_common.load_resume; pair 'all', pair set 'holdout')
UNIT = 'tag-hb-atropos'
DEVCHECK_UNIT = 'tag-hb-atropos-devcheck'
LOCK_NAME = 'hp_atropos.lock'
THREADS = 8                                     # atropos_seg's default and D0's (ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS)
N_SUBJECTS = 100
STAGE_NAME = 'holdout_B_v1'                     # provenance image_source.stage
STAGE_FILES = ('selection.json', 'completion.json')
SUMS = 'SHA256SUMS'
PREP_RECORD = 'prep_record.json'                # prep_stage.RECORD
RECORD, STOPPED = 'atropos_record.json', 'atropos_stopped.json'
DEVCHECK_RECORD = 'devcheck_atropos.json'
RUNTIME_S = 3*3600                              # 〔估算〕D0: about 11 s of ANTs per subject (seg/atropos_v1 provenance seconds), 100 subjects plus loading 50 pairs
DEV_SUBJECTS = ('OASIS_OAS1_0438_MR1', 'OASIS_OAS1_0439_MR1')
COMPARED = ('posteriors01.nii.gz', 'posteriors02.nii.gz', 'posteriors03.nii.gz', 'labels.nii.gz')
INPUTS_COMPARED = ('t1.nii.gz', 'mask.nii.gz', 't1_trunc.nii.gz')
DEVCHECK_PLAN_REL = 'runs/20261107-hp/atropos_devcheck_plan.md'
ATROPOS_DEVCHECK_PLAN_SHA256 = '678ea102dfa310222e26ace9ce9da3ac239c5d1372cecf8784e28cf44d706a23'     # P0 (11-14): the final plan (PT 5ef8243); the development check refuses None or another plan
READ_EQUAL = '逐位相同 {n}/{n}'
READ_NONDETERMINISTIC = '输入逐位相同、输出不逐位（Atropos 非确定〔推断〕；交总负责人）'
READ_INPUTS_DIFFER = '输入不逐位（入口不等价；停，交总负责人）'
READ_SPY_FAILED = '不判（spy 不过）'
# Review I1 (the lead's ruling of 2026-11-09): filled by a reviewed commit (P1) after the development check ran on P0 and was read (atropos_devcheck_plan.md §4), as a plain
# dict literal (contract C24.3-6 (a); hp_core.check_decision reads it through a read-only copy) with exactly hp_core.DECISION_KEYS, code_commit the commit the check ran on (A5a).
# The reading 逐位相同 2/2 lets the hold-out run start; the reading 输入逐位相同、输出不逐位 only with BZ's words (bz_words) accepting it; any other reading never.  None
# (today): the hold-out run refuses (stage decision).
ATROPOS_DEVCHECK_DECISION = {
    'devcheck_json': '/home/rmcfz/seq_vp_tag_results/hp_devcheck_v1/atropos/devcheck_atropos.json',
    'devcheck_sha256': '63ce8c7e548547745f2b1b0ea5944c3d480d226eae707660a2c06bacb37128c3',
    'reading': '逐位相同 2/2',
    'plan_sha256': '678ea102dfa310222e26ace9ce9da3ac239c5d1372cecf8784e28cf44d706a23',
    'code_commit': '318e72ad3d097bcbd03db180a56c0144ee4344bc',
    'decided_by': 'lead (total responsible), 2026-11-15, per atropos_devcheck_plan.md §4',
    'review': 'docs/bayesreg/reviews/D58_复核_20261115.md (sha256 9629f2495a4f099347871e4eb3d606a22f2484838745e65e494fe1145f946c4b; 0 blocking, 1 important, 8 minor; reading holds)',
    'bz_words': None,
}     # P1 (11-15): filled after the check ran on P0'' 318e72ad (C2, D5); bz_words None: the reading is 逐位相同 2/2, not 输入逐位相同、输出不逐位 (plan §4, revision 1 item 6)


def say(text):
    print(text, flush=True)


def reference_root():
    return Path(hc.RESULTS)/'seg'/'atropos_v1'


def devcheck_root():
    return Path(hc.RESULTS)/core.DEVCHECK_DIR/'atropos'


def devcheck_lock():
    return Path(hc.RESULTS)/core.DEVCHECK_DIR/'locks'/'atropos_devcheck.lock'


# ---- the wrapper around atropos_seg.segment ------------------------------------------------------------------------------------------------------
@contextmanager
def tissue_layer_check(allowlist):
    """For the length of the block, atropos_seg's validate_segmentation_provenance is bayesreg.io.pair's with allowlist=<allowlist> (C23.4-8: the tissue layer of a
    pair with a sealed subject needs the bound allowlist load_pair admitted it through); restored after, also on an exception.  Refuses to wrap a name that is not
    bayesreg's function (someone else changed it)."""
    original = atropos_seg.validate_segmentation_provenance
    if original is not validate_segmentation_provenance:
        raise RuntimeError('atropos_seg.validate_segmentation_provenance is not bayesreg.io.pair.validate_segmentation_provenance: not wrapped')
    atropos_seg.validate_segmentation_provenance = functools.partial(validate_segmentation_provenance, allowlist=allowlist)
    try:
        yield
    finally:
        atropos_seg.validate_segmentation_provenance = original


class StageEntries:
    """atropos_seg.subject_images, one subject at a time: {subject: [(pair, side, PairInput)]} over the stage's pairs in selection.json order, each pair loaded by the
    same load_pair (fixed and moving only) with the run's allowlist, just before its subjects are segmented.  subject_images loads every pair at once (50 hold-out
    pairs: about 5.5 GB of float64 images); here at most CACHE pairs are held, so a hold-out row is loaded once for its two subjects.  The subjects of a pair are
    taken from its name (<fixed>__<moving>) to know which pairs to load; precheck() (before the lock) checks every manifest's ids against the name and passes them through
    the guard, and pair() checks the loaded pair again."""
    CACHE = 2

    def __init__(self, stage_root, allowlist):
        from bayesreg.io.pair import load_pair
        self.root, self.allowlist, self._load = Path(stage_root), allowlist, load_pair
        selection = json.loads((self.root/'selection.json').read_text(encoding='utf-8'))
        self.names = [row['pair'] for row in selection['pairs']]
        self.where = {}
        for name in self.names:
            parts = name.split('__')
            if len(parts) != 2:
                raise hc.Refused('stage', f'{name!r} is not <fixed>__<moving>')
            for side, subject in zip(('fixed', 'moving'), parts):
                self.where.setdefault(subject, []).append((name, side))
        self._cache = {}

    def subjects(self):
        return list(self.where)

    def precheck(self):
        """Review m4, before the lock: every pair's manifest read through the loader's whitelisting decoder (ids, grid and hashes only), its ids those of the pair's name,
        and both admitted by the guard with the run's allowlist (bayesreg.io.guard.check_pair_subjects: ids only).  Refused (stage stage) or SubjectSealed otherwise."""
        from bayesreg.io.guard import check_pair_subjects
        from bayesreg.io.pair import _read_manifest
        for name in self.names:
            _, _, selected = _read_manifest(self.root/name/'manifest.json')
            if hc.pair_name(selected['fixed_subject'], selected['moving_subject']) != name:
                raise hc.Refused('stage', f"{name}: the manifest names {selected['fixed_subject']}, {selected['moving_subject']}")
            check_pair_subjects(selected['fixed_subject'], selected['moving_subject'], self.allowlist)
        return dict(pairs=len(self.names), checked='manifest ids = pair names; check_pair_subjects with the run\'s allowlist')

    def pair(self, name):
        if name not in self._cache:
            while len(self._cache) >= self.CACHE:
                self._cache.pop(next(iter(self._cache)))
            pair = self._load(self.root/name/'pair.npz', allowlist=self.allowlist)
            if hc.pair_name(pair.fixed_subject, pair.moving_subject) != name:
                raise hc.Refused('stage', f'{name}: the manifest names {pair.fixed_subject}, {pair.moving_subject}')
            self._cache[name] = pair
        return self._cache[name]

    def __getitem__(self, subject):
        return [(name, side, self.pair(name)) for name, side in self.where[subject]]


def segment_subjects(subjects, found, out_root, *, allowlist, stage_name, log=say):
    """atropos_seg.segment for each subject in order, inside tissue_layer_check; then the provenance checked again with the allowlist, for every pair the subject is
    in, and its files' sha256.  `found`: StageEntries.  Returns {subject: summary}."""
    out = {}
    for subject in subjects:
        entries = found[subject]
        t0 = perf_counter()
        with tissue_layer_check(allowlist):
            prov = atropos_seg.segment(subject, entries, out_root, THREADS, stage_name)
        for _, side, pair in entries:
            validate_segmentation_provenance(prov, pair, side=side, layer='tissue', allowlist=allowlist)
        verified = verify_segmentation_files(prov, Path(out_root)/subject)
        out[subject] = dict(seconds=round(perf_counter()-t0, 3), atropos_seconds=prov.get('seconds'), class_names=prov.get('class_names'), wm_index=prov.get('wm_index'),
                            pairs=[f'{n}:{s}' for n, s, _ in entries], qc=prov['checks'].get('qc_min_class_fraction_ge_0.05'),
                            provenance_sha256=file_sha256(Path(out_root)/subject/'provenance.json'), output_sha256=verified)
        log(f'{subject}: {out[subject]["seconds"]:.1f} s {prov.get("class_names")}')
    return out


# ---- the hold-out stage, the decision, the resume -------------------------------------------------------------------------------------------------
def stage_paths(stage_root, pairs):
    root = Path(stage_root)
    return [root/n for n in (*STAGE_FILES, SUMS, PREP_RECORD)]+[root/hc.pair_name(f, m)/n for f, m in pairs for n in ('pair.npz', 'manifest.json')]


def check_stage(stage_root, pairs, *, unseal_sha256):
    """The stage prep_stage.py wrote, for exactly these pairs: selection.json lists them in order, completion.json says complete with their number, SHA256SUMS names
    exactly selection.json, completion.json and each pair's pair.npz and manifest.json, and every one has that sha256 now; prep_record.json (review m3) says passed,
    was admitted with this unseal record and wrote this SHA256SUMS.  Refused (stage stage) otherwise."""
    root = Path(stage_root)
    try:
        selection = json.loads((root/'selection.json').read_text(encoding='utf-8'))
        completion = json.loads((root/'completion.json').read_text(encoding='utf-8'))
        lines = [line for line in (root/SUMS).read_text(encoding='utf-8').splitlines() if line.strip()]
        prep = json.loads((root/PREP_RECORD).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise hc.Refused('stage', f'{root}: the stage cannot be read ({type(exc).__name__}: {exc})') from None
    names = [row.get('pair') for row in selection.get('pairs', [])]
    want = [hc.pair_name(f, m) for f, m in pairs]
    if names != want:
        raise hc.Refused('stage', f'{root}/selection.json lists {len(names)} pairs that are not the admitted rows in their order')
    if completion.get('complete') is not True or completion.get('pairs') != len(pairs):
        raise hc.Refused('stage', f'{root}/completion.json does not say complete with {len(pairs)} pairs')
    listed = dict(reversed(line.split('  ', 1)) for line in lines)
    expected = sorted(['selection.json', 'completion.json']+[f'{n}/{f}' for n in want for f in ('pair.npz', 'manifest.json')])
    if sorted(listed) != expected:
        raise hc.Refused('stage', f'{root}/{SUMS} does not name exactly the stage\'s files')
    changed = [name for name in expected if file_sha256(root/name) != listed[name]]
    if changed:
        raise hc.Refused('stage', f'{root}: {changed[:3]} differ from {SUMS}')
    sums_sha = file_sha256(root/SUMS)
    problems = []
    if prep.get('passed') is not True:
        problems.append('passed is not true')
    if ((prep.get('admission') or {}).get('unseal') or {}).get('sha256') != unseal_sha256:
        problems.append('it was admitted with another unseal record')
    if (prep.get('stage_summary') or {}).get('sha256sums_sha256') != sums_sha:
        problems.append(f'its SHA256SUMS sha256 is not the stage\'s {sums_sha}')
    if problems:
        raise hc.Refused('stage', f'{root}/{PREP_RECORD}: {problems}')
    return dict(stage=str(root), pairs=len(want), sha256sums_sha256=sums_sha, prep_record_sha256=file_sha256(root/PREP_RECORD),
                checked='selection, completion, SHA256SUMS (every file re-hashed), prep_record.json (passed, the unseal record, the SHA256SUMS)')


def check_decision(decision=None):
    """Review I1: ATROPOS_DEVCHECK_DECISION (read at call time) names the development check's record (devcheck_root()/devcheck_atropos.json) by its sha256, the pinned
    plan, a reading this tool accepts (逐位相同 2/2; or 输入逐位相同、输出不逐位 with BZ's words) and the check's lock line, on the two development subjects
    (hp_core.check_decision).  Refused (stage decision) otherwise."""
    decision = ATROPOS_DEVCHECK_DECISION if decision is None else decision

    def extra(record):
        return [] if list(record.get('subjects') or []) == list(DEV_SUBJECTS) else [f"the check's subjects {record.get('subjects')} are not {list(DEV_SUBJECTS)}"]
    return core.check_decision(decision, constant='ATROPOS_DEVCHECK_DECISION', record_path=devcheck_root()/DEVCHECK_RECORD, lock_path=devcheck_lock(),
                               plan_sha256=ATROPOS_DEVCHECK_PLAN_SHA256, schema=SCHEMA+'.devcheck', accepted=(READ_EQUAL.format(n=len(DEV_SUBJECTS)),),
                               accepted_with_bz=(READ_NONDETERMINISTIC,), extra=extra)


def run_complete(out_root):
    """Prereg §5.4 "完整记录" of the Atropos entry: atropos_record.json in the output root with passed true."""
    try:
        return bool(json.loads((Path(out_root)/RECORD).read_text(encoding='utf-8')).get('passed'))
    except (OSError, ValueError):
        return False


def holdout_policy(out_root, lock, *, resume_path=None):
    import bayesreg.io.sealed as sealed
    from bayesreg.io.unseal import DECISION_LOG_REL, UNSEAL_REL
    tree = Path(sealed.REPO)
    reads = [tree/hc.PAIRS_TSV_REL, tree/UNSEAL_REL, tree/core.ALLOWLIST_REL, tree/DECISION_LOG_REL, *core.allowed_list_paths(), devcheck_root()/DEVCHECK_RECORD,
             devcheck_lock()]
    if resume_path is not None:
        auth = Path(resume_path) if Path(resume_path).is_absolute() else Path(hc.AUTH_ROOT)/resume_path
        reads += [auth, lock]
    return core.Policy(name='hp atropos (hold-out)', read_exact=reads, write_inside=[out_root], write_exact=[lock, hc.resume_lock_path(lock)], results=hc.RESULTS)


def order_of(pairs):
    return list(dict.fromkeys(s for p in pairs for s in p))


def run_holdout(*, unseal_path, code, dry_run=False, out=None, resume_path=None, log=say):
    stage_root, out_root, lock = Path(hc.STAGE_ROOT), Path(hc.ATROPOS_ROOT), Path(hc.LOCK_ROOT)/LOCK_NAME
    core.guard_write(out_root, lock, hc.resume_lock_path(lock), core.attempt_path(out_root))
    problem = core.pinned_code_problem(code)
    if problem and not dry_run:                     # A5a: rule one passed in main (hp_core.check_pinned), else nothing of the hold-out is read
        raise hc.Refused('code', problem)
    record = dict(schema=SCHEMA, tool=f'{SCRIPT} run', started_utc=hc.utc(), stage=str(stage_root), out=str(out_root), lock=str(lock), threads=THREADS, dry_run=dry_run,
                  code=code, unseal_path=None if unseal_path is None else str(unseal_path), resume_path=None if resume_path is None else str(resume_path),
                  atropos=dict(bin=str(atropos_seg.ANTS), version=atropos_seg.ANTS_VERSION, parameters=atropos_seg.PARAMS))
    record['preload'] = core.preload()
    policy = holdout_policy(out_root if not dry_run else out, lock, resume_path=resume_path)
    with core.spying(policy) as spy:
        try:
            record['decision'] = check_decision()
        except hc.Refused as exc:
            if not dry_run:
                raise
            record['decision'] = dict(refused=str(exc))
        resume = core.load_resume(resume_path, tool=TOOL, arm=ARM, lock=lock, previous_record=out_root/STOPPED, complete=lambda: run_complete(out_root))
        admission = core.admit_holdout(None, unseal_path=unseal_path)
        if [tuple(p) for p in admission.pairs] != [tuple(r) for r in admission.rows]:
            raise hc.Refused('admission', 'the run is every row of the pair list, in its order')
        record['admission'] = admission.record
        subjects = order_of(admission.pairs)
        if len(subjects) != N_SUBJECTS:
            raise hc.Refused('admission', f'{len(subjects)} subjects in the pair list, {N_SUBJECTS} expected')
        hc.check_once(lock, resume=resume)
        if resume is None and out_root.exists():
            raise hc.Refused('output', f'{out_root} exists: the hold-out tissue layer is written once, never overwritten (after a failure: a committed resume authorization)')
        if resume is not None and core.attempt_path(out_root).exists():
            raise hc.Refused('resume', f'{core.attempt_path(out_root)} exists: a failed attempt was kept already')
        if dry_run:
            record['launch'] = core.launch_line(unit=UNIT, script=SCRIPT, args=['run', '--unseal', str(unseal_path)]+([] if resume is None else ['--resume', str(resume_path)]),
                                                runtime_s=RUNTIME_S, cpu_quota='800%')
            record['spy'] = spy.record()
            return record
        spy.policy.allow_read(stage_paths(stage_root, admission.pairs)+([out_root/STOPPED] if resume is not None else []))
        record['stage_check'] = check_stage(stage_root, admission.pairs, unseal_sha256=admission.unseal.sha256)
        found = StageEntries(stage_root, admission.allowlist)
        if sorted(found.subjects()) != sorted(subjects):
            raise hc.Refused('stage', f'the stage\'s pairs hold {len(found.subjects())} subjects, not the {len(subjects)} of the admitted rows')
        record['stage_precheck'] = found.precheck()
        record['shared_subjects'] = sorted(s for s, e in found.where.items() if len(e) > 1)
        record['machine'] = hc.machine_gate(own_units=(UNIT,))
        line = hc.lock_line(tool=f'{SCRIPT} run', out=str(out_root), stage_sha256sums=record['stage_check']['sha256sums_sha256'], unseal_sha256=admission.unseal.sha256,
                            decision_sha256=record['decision']['devcheck_sha256'], code_commit=(code or {}).get('commit'),
                            event='authorized_resume' if resume else 'first_run', resume_sha256=None if resume is None else resume.sha256)
        record['lock_line'] = hc.take_lock(resume.lock_path if resume else lock, line)
        if resume is not None:
            record['resume'] = dict(path=resume.path, sha256=resume.sha256, kept=core.keep_failed_attempt(out_root))
        out_root.mkdir(parents=True)
        try:
            record['subjects'] = segment_subjects(subjects, found, out_root, allowlist=admission.allowlist, stage_name=STAGE_NAME, log=log)
        except BaseException as exc:
            record.update(stopped=f'{type(exc).__name__}: {exc}', spy=spy.record(), ended_utc=hc.utc())
            spy.remove()
            hc.write_json_once(out_root/STOPPED, record)
            raise
        record['spy'] = spy.record(expect=[p for p in stage_paths(stage_root, admission.pairs) if p.name == 'pair.npz'])
    record.update(ended_utc=hc.utc(), passed=bool(record['spy']['passed']))
    hc.write_json_once(out_root/RECORD, record)
    return record


# ---- the development check -----------------------------------------------------------------------------------------------------------------------
def check_plan(repo=None):
    import bayesreg.io.sealed as sealed
    path = Path(sealed.REPO if repo is None else repo)/DEVCHECK_PLAN_REL
    if ATROPOS_DEVCHECK_PLAN_SHA256 is None:
        raise hc.Refused('plan', f'{DEVCHECK_PLAN_REL} is a draft: ATROPOS_DEVCHECK_PLAN_SHA256 is None until the lead pins it (a reviewed commit)')
    digest = file_sha256(path) if path.is_file() else None
    if digest != ATROPOS_DEVCHECK_PLAN_SHA256:
        raise hc.Refused('plan', f'{DEVCHECK_PLAN_REL}: sha256 {digest} is not the pinned {ATROPOS_DEVCHECK_PLAN_SHA256}')
    return dict(path=DEVCHECK_PLAN_REL, sha256=digest)


def compare_file(new, old, *, mask=None, posterior=False):
    """One file of a subject: file sha256, the stored array (atropos_seg.load_zyx: dtype, shape, bytes); not bitwise -> the largest absolute difference and, in the
    mask, the share of voxels that agree (labels) or whose argmax over the three posteriors agrees (posteriors, filled in by compare_subject)."""
    a, b = atropos_seg.load_zyx(new), atropos_seg.load_zyx(old)
    row = dict(file_sha256_equal=file_sha256(new) == file_sha256(old), dtype_new=str(a.dtype), dtype_old=str(b.dtype), shape_new=list(a.shape), shape_old=list(b.shape))
    row['bitwise'] = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if not row['bitwise'] and a.shape == b.shape:
        row['max_abs_diff'] = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
        if mask is not None and not posterior:
            row['agreement_in_mask'] = float(np.mean(a[mask] == b[mask]))
    return row, a, b


def compare_subject(new_dir, old_dir):
    new_dir, old_dir = Path(new_dir), Path(old_dir)
    inputs = {}
    for name in INPUTS_COMPARED:
        inputs[name], a, _ = compare_file(new_dir/name, old_dir/name)
        if name == 'mask.nii.gz':
            mask = a > 0
    outputs, posts_new, posts_old = {}, [], []
    for name in COMPARED:
        posterior = name.startswith('posteriors')
        outputs[name], a, b = compare_file(new_dir/name, old_dir/name, mask=mask, posterior=posterior)
        if posterior:
            posts_new.append(a)
            posts_old.append(b)
    argmax = None
    if not all(outputs[n]['bitwise'] for n in COMPARED if n.startswith('posteriors')) and all(p.shape == mask.shape for p in posts_new+posts_old):
        argmax = float(np.mean(np.argmax(np.stack(posts_new), 0)[mask] == np.argmax(np.stack(posts_old), 0)[mask]))
    prov_new = json.loads((new_dir/'provenance.json').read_text(encoding='utf-8'))
    prov_old = json.loads((old_dir/'provenance.json').read_text(encoding='utf-8'))
    classes = dict(new=prov_new.get('class_names'), old=prov_old.get('class_names'), equal=prov_new.get('class_names') == prov_old.get('class_names'),
                   wm_index_equal=prov_new.get('wm_index') == prov_old.get('wm_index'))
    inputs_bitwise = all(r['bitwise'] for r in inputs.values())
    outputs_bitwise = all(r['bitwise'] for r in outputs.values()) and classes['equal'] and classes['wm_index_equal']
    return dict(inputs=inputs, outputs=outputs, posterior_argmax_agreement_in_mask=argmax, classes=classes, inputs_bitwise=inputs_bitwise, outputs_bitwise=outputs_bitwise)


def devcheck_reading(rows):
    n = len(rows)
    if all(r['inputs_bitwise'] and r['outputs_bitwise'] for r in rows.values()):
        return READ_EQUAL.format(n=n)
    if all(r['inputs_bitwise'] for r in rows.values()):
        return READ_NONDETERMINISTIC
    return READ_INPUTS_DIFFER


def devcheck_policy(out_root, lock, dev_stage, *, extra_out=None):
    import bayesreg.io.sealed as sealed
    tree = Path(sealed.REPO)
    reads = [tree/core.ALLOWLIST_REL, tree/DEVCHECK_PLAN_REL, *core.allowed_list_paths(), Path(dev_stage)/'selection.json']
    for name in hc.DEV_PAIRS:
        reads += [Path(dev_stage)/name/n for n in ('pair.npz', 'manifest.json')]
    for subject in DEV_SUBJECTS:
        reads += [reference_root()/subject/n for n in (*COMPARED, *INPUTS_COMPARED, 'provenance.json')]
    return core.Policy(name='hp atropos (development check)', read_exact=reads, write_inside=[out_root] if extra_out is None else [extra_out], write_exact=[lock],
                       results=hc.RESULTS)


def run_devcheck(*, code, dry_run=False, out=None, log=say):
    out_root, lock, dev_stage, ref = devcheck_root(), devcheck_lock(), Path(hc.DEV_STAGE), reference_root()
    core.guard_write(out_root, lock)
    record = dict(schema=SCHEMA+'.devcheck', tool=f'{SCRIPT} devcheck', started_utc=hc.utc(), out=str(out_root), lock=str(lock), dev_stage=str(dev_stage), reference=str(ref),
                  subjects=list(DEV_SUBJECTS), threads=THREADS, dry_run=dry_run, code=code)
    record['preload'] = core.preload()
    with core.spying(devcheck_policy(out_root, lock, dev_stage, extra_out=out if dry_run else None)) as spy:
        try:
            record['plan'] = check_plan()
        except hc.Refused as exc:
            if not dry_run:
                raise
            record['plan'] = dict(refused=str(exc))
        allowlist, record['admission'] = core.admit_dev(subjects=DEV_SUBJECTS)
        missing = [str(p) for s in DEV_SUBJECTS for p in (ref/s/n for n in (*COMPARED, *INPUTS_COMPARED, 'provenance.json')) if not p.is_file()]
        if missing:
            raise hc.Refused('reference', f'the reference files are missing: {missing[:3]}')
        if out_root.exists():
            raise hc.Refused('output', f'{out_root} exists: the development check writes a new directory')
        if lock.exists():
            raise hc.Refused('once', f'{lock} exists: the development check runs once')
        if dry_run:
            record['launch'] = core.launch_line(unit=DEVCHECK_UNIT, script=SCRIPT, args=['devcheck'], runtime_s=1800, cpu_quota='800%')
            record['spy'] = spy.record()
            return record
        found = StageEntries(dev_stage, allowlist)
        record['stage_precheck'] = found.precheck()
        record['machine'] = hc.machine_gate(own_units=(DEVCHECK_UNIT,))
        record['lock_line'] = hc.take_lock(lock, hc.lock_line(tool=f'{SCRIPT} devcheck', out=str(out_root), plan_sha256=record['plan']['sha256'],
                                                              code_commit=(code or {}).get('commit')))
        out_root.mkdir(parents=True)
        try:
            record['subjects_run'] = segment_subjects(list(DEV_SUBJECTS), found, out_root, allowlist=allowlist, stage_name=atropos_seg.DEFAULT_STAGE, log=log)
            rows = {s: compare_subject(out_root/s, ref/s) for s in DEV_SUBJECTS}
        except BaseException as exc:
            record.update(stopped=f'{type(exc).__name__}: {exc}', spy=spy.record(), ended_utc=hc.utc())
            spy.remove()
            hc.write_json_once(out_root/STOPPED, record)
            raise
        record['spy'] = spy.record(expect=[ref/s/n for s in DEV_SUBJECTS for n in COMPARED])
    record.update(rows=rows, reading=devcheck_reading(rows) if record['spy']['passed'] else READ_SPY_FAILED, ended_utc=hc.utc())
    hc.write_json_once(out_root/DEVCHECK_RECORD, record)
    return record


# ---- the command line ----------------------------------------------------------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog=SCRIPT)
    sub = ap.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--unseal', type=Path, default=None)
    run.add_argument('--resume', type=Path, default=None)
    run.add_argument('--dry-run', action='store_true')
    run.add_argument('--out', type=Path, default=None)
    dev = sub.add_parser('devcheck')
    dev.add_argument('--dry-run', action='store_true')
    dev.add_argument('--out', type=Path, default=None)
    a = ap.parse_args(argv)
    if a.dry_run and a.out is None:
        ap.error('--dry-run writes its record into --out DIR (a new or empty directory outside the results roots, e.g. mktemp -d)')
    return a


def main(argv=None, *, code=None, out=say):
    a = parse_args(argv)
    if a.dry_run:
        problem = core.dry_run_out_problem(a.out)
        if problem:
            out(f'{problem}: a dry run writes into a new or empty directory outside the results root, the D: directory, the data roots and the code trees')
            return 2
    try:
        code = core.code_identity() if code is None else code
        code_check = core.check_code(code, require_clean=not a.dry_run)
        if a.command == 'run':                      # A5a (contract C24.3 rule one): the hold-out run's code is the pinned commit's; the development check is not (C24.3-10)
            code = core.check_pinned(code, dry_run=a.dry_run)
    except hc.Refused as exc:
        out(f'refused ({exc.stage}): {exc}')
        return 8
    decision_none = a.command == 'run' and ATROPOS_DEVCHECK_DECISION is None       # A5a diff review n1: a hold-out run refuses at stage decision while it is None (devcheck does not read it)

    def stopped_dry(stage, message, **extra):         # A5a review m8: a dry run that stops writes the same record (passed, problems, expected, unexpected, p0_ok)
        if a.dry_run:
            a.out.mkdir(parents=True, exist_ok=True)
            hc.write_json_once(a.out/'dry_run.json', dict(schema=SCHEMA+'.dry_run', command=a.command, refused=dict(stage=stage, message=message), utc=hc.utc(), code=code, **extra,
                                                          **core.dry_run_fields(code, code_check=code_check, refusal=(stage, message), decision_none=decision_none)))
    try:
        if a.command == 'run':
            record = run_holdout(unseal_path=a.unseal, code=dict(code, check=code_check), dry_run=a.dry_run, out=a.out, resume_path=a.resume)
        else:
            record = run_devcheck(code=dict(code, check=code_check), dry_run=a.dry_run, out=a.out)
    except hc.Refused as exc:
        out(f'refused ({exc.stage}): {exc}')
        stopped_dry(exc.stage, str(exc), spy=getattr(exc, 'hp_spy', None))
        return 8
    except core.Breach as exc:
        out(f'stopped by the spy: {exc}')
        stopped_dry('spy', str(exc))
        return 8
    except SystemExit as exc:                   # atropos_seg's STOPs (QC, an exit of ANTs, the image differing between pairs): the record is kept
        out(f'stopped: {exc}')
        stopped_dry('SystemExit', str(exc))
        return 8
    except (ProvenanceRejected, SubjectSealed, InputValidationError) as exc:      # the loader's and the provenance checks (review m9)
        out(f'stopped: {type(exc).__name__}: {exc}')
        stopped_dry(type(exc).__name__, str(exc))
        return 8
    if a.dry_run:
        a.out.mkdir(parents=True, exist_ok=True)
        fields = core.dry_run_fields(code, code_check=code_check, record=record, decision_none=decision_none)
        hc.write_json_once(a.out/'dry_run.json', dict(record, schema=SCHEMA+'.dry_run', command=a.command, **fields))
        out(f"dry run passed the admission ({a.command}); launch: {record.get('launch')}; problems: {fields['problems'] or 'none'}; P0 reading: "
            f"{'only the expected refusals' if fields['p0_ok'] else 'unexpected problems'} (the record, not the exit code)")
        return 0
    if a.command == 'devcheck':
        out(f"devcheck: done; reading {record['reading']}")
        return 0
    out(f"run: done; {len(record['subjects'])} subjects in {record['out']}; spy passed {record['spy']['passed']}")
    return 0 if record['passed'] else 8


if __name__ == '__main__':
    sys.exit(main())
