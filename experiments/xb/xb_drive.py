"""XB driver (runs/20261107-xb/任务书.md): one external baseline arm on one pair, in its own unit (xb_chain.py starts it), or a dry run that checks everything and runs nothing.

    python -I -B experiments/xb/xb_drive.py run     --arm ARM --pair-set {holdout,dev} --pair FIXED__MOVING [--resume-record AUTH] [--method-run T88_RUN_JSON] [--batch-dir DIR]
    python -I -B experiments/xb/xb_drive.py dry-run --arm ARM --pair-set {holdout,dev} --pair FIXED__MOVING --out NEW_EMPTY_DIR [--method-run T88_RUN_JSON]
    python -I -B experiments/xb/xb_drive.py authorization-draft --kind {resume,rerun,no_resume} --arm ARM --pair-set ... --pair ... --written-by ... --words ... --time-local ... --reason ...

The order of a run; nothing of a subject is opened before step 7, and nothing decoded before the lock (step 9):
  1. the arm is one of xb_core.ARMS, the pair a plain <fixed>__<moving>; the driver's spy (xb_spy.DriverSpy) is on, for the whole run;
  2. the code tree clean, the code the pinned commit's (rule one, holdout_common.pinned_code_problems: P1 from the committed runs/holdout_B_v1/pinned_code.json, every commit of P1..HEAD checked;
     contract C24.3, A5a; the lock line's code_commit is P1, its code_head HEAD) and the preregistration pinned (xb_core.PREREG_SHA256, I5), all for a hold-out run, **before the admission** (I1); a
     dry run reports them as problems and goes on (before BZ unseals there is no pinned record: the expected refusal, recorded);
  3. a cell refused before (a refusal record of it in any batch) runs only with a committed rerun authorization that lists those records and their sha256 (I1; C23.12-6, D107);
  4. admission (holdout_common): a hold-out pair needs SG's unseal object, H1-H4 never, the row of holdout_B_pairs.tsv in its direction; a development pair one of the 19 and the allowlist;
  5. the arm is not stopped (P19's arm_stopped.json); the frozen configuration (xb_core.check_frozen, the smoke_cmd.md files too) and the environment (xb_core.check_environment);
  6. a resume authorization if one is given (holdout_common.load_resume: the record resumed is the overriding record when the unit's evidence overrode the driver's, I2), the cell's lock not there (or,
     with a resume, the resume's lock not there); ConvexAdam-v4seg: --method-run is the pinned path of the method's T88 run.json on this pair, with labels_sha256 (I4, m10);
  7. the pair's input: the manifest through the loader's whitelisting decoder, the general guard with the admission's allowlist, the container sha256; ConvexAdam-v4seg: both predictions' provenance
     against the anchor and their probs36's sha256;
  8. the disks (holdout_common.disk_guard, both sides), the machine (holdout_common.machine_gate) and the arm's precheck (MemAvailable, the GPU idle: m8), each waited for up to 15 minutes;
  9. the lock (O_EXCL, its line: what was checked); a dry run stops before it and writes its record to --out;
  10. the derived npz (fixed, moving, byte for byte), ConvexAdam-v4seg's maps (equal to the method's labels), the method under its spies, the field, the export check;
  11. ok: the intermediates listed (nothing deleted), a D: arm's kept files moved (sha256 before and after); the spies passed (else infra_failure, m7); the terminal record (result.json, once); P19.
Every refusal before the lock (steps 1-9, a spy breach and an exception of our own code there included) writes one refusal record into the batch directory (<batch>/refusals/, never overwritten) and
never the cell's result.json (I1); a dry run writes only its dry_run.json -- with the fields every hold-out entry's dry run writes (A5a review m8, holdout_common.dry_run_summary: passed,
problems, expected, unexpected, p0_ok); the checklist reads them, not the exit code (a dry run on P0 exits refused: no pinned record, no unseal record).
Exit codes: 0 ok; 1 method_failure or unit_limit; 2 refused (admission included); 20 infra_failure (before the lock: the disks, the machine, the precheck, a spy breach); 23 wrapper_error.

LP-B (contract C26.5-6, C26.4 L33, C26.6-3, C26.8; LPBA40 preregistration 06121278... §4.3, §6.6, §12 item 11): --pair-set lpba40_test | lpba40_feas.  Only xb_core.LPBA_ARMS run (an other arm is
refused before anything); step 2 for an LPBA mode: the code (lpba_arms.code_record: formal, rule one -- LP-C2's, C26.9-4, fail closed; feasibility, a clean checkout) and the pins (lpba_common.check_pins:
the preregistration 06121278..., the lists, configs/frozen/holdout_B_v1_baselines.json 7c4965eb..., configC_reference.json, the arm table, the method's settings, the c record) in place of the hold-out's
rule one and preregistration; step 4 the admission lpba_common's (the pinned LPBA allowlist by its path and sha256; the pair a row of the mode's table in its direction; both subjects on no sealed
list and through check_subject: the 34 by the allowlist's one exception, S01 and S02 by steps (2), (3)); check_pair_input then takes that allowlist (L33: the manifest's subjects, check_pair_subjects,
the container sha256).  ConvexAdam-v4seg reads the mode's HI predictions and HI's records (subjects/<s>.json) and the method's t88/run.json of the pair (xb_core.method_t88_lpba: the cell's, at the
finalized record's sha256).  Nothing moves (no store); the kept files stay in the work cell, disp.file names them there.  P19: an LPBA40 arm stops at its second suspicious pair (xb_core.p19_stop_at).
"""
import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import xb_core as xc  # noqa: E402

hc = xc.hc
xb_spy = xc.xb_spy
TOOL = 'xb'
DRIVER_FILES = ('xb_drive.py', 'xb_core.py', 'xb_spy.py', 'spy/xb_policy.py', 'spy/xb_spyrun.py', 'v4seg_convention.py', 'm_convexadam_v4seg.py', 'xb_chain.py')


@dataclass
class Deps:
    """What a run asks of the machine (tests replace it)."""
    gate: object = None                         # () -> record; None: holdout_common.machine_gate waited for
    disk_usage: object = None                   # shutil.disk_usage
    env_run: object = None                      # subprocess.run for the environment check
    code_run: object = None                     # subprocess.run for git
    pinned: object = None                       # () -> (record, problems) of rule one (A5a); None: holdout_common.pinned_code_problems(run=code_run) (P1 from runs/holdout_B_v1/pinned_code.json)
    step: object = xc.run_step
    precheck: object = xc.precheck
    log: object = print
    probe: object = None                        # (when) -> record; None: experiments/he/unit_probe.probe
    extra: dict = field(default_factory=dict)


PAIR_SETS = ('holdout', 'dev')+xc.LPBA_PAIR_SETS


def planned(pair_set, fixed, moving):
    """Whether (arm, pair) is a cell of the plan: a row of the pinned pair list, or one of the 19 development pairs; LP-B: a row of an LPBA mode's table in its direction.  Ids only."""
    try:
        if pair_set == 'holdout':
            return (fixed, moving) in hc.holdout_pairs()
        if xc.is_lpba(pair_set):
            return xc.lpba().planned(pair_set, xc.lpba().lc.pair_name(fixed, moving))
        return hc.pair_name(fixed, moving) in hc.DEV_PAIRS
    except hc.Refused:
        return False


def lpba_admission(pair_set, fixed, moving):
    """LP-B (C26.5-6, L33; C26.3-5): the pinned LPBA allowlist (lpba_common.load_lpba_allowlist: by its path and sha256 only), the pair a row of the mode's table in its direction, both subjects on no
    sealed list and through check_subject on that allowlist (lpba_common.admit_pairs: the 34 by its one exception, S01 and S02 by steps (2), (3)).  Returns (record, allowlist); Refused otherwise.
    Opens no file of a subject."""
    lc = xc.lpba().lc
    allowlist, allowlist_record = lc.load_lpba_allowlist()
    rows, admitted = lc.admit_pairs(pair_set, [(fixed, moving)], allowlist)
    return dict(pair_set=pair_set, fixed_subject=fixed, moving_subject=moving, row=rows[0], admission=admitted, allowlist=allowlist_record,
                rule='xb_drive.lpba_admission (C26.5-6, L33): the pinned LPBA allowlist, the mode\'s table, sealed lists, check_subject; check_pair_input then with that allowlist'), allowlist


def admit(pair_set, fixed, moving, deps):
    if pair_set == 'holdout':
        return hc.admit_holdout_pair(fixed, moving)
    if pair_set == 'dev':
        return hc.admit_dev_pair(fixed, moving)
    if xc.is_lpba(pair_set):
        return lpba_admission(pair_set, fixed, moving)
    raise hc.Refused('admission', f'pair set {pair_set!r}: one of {PAIR_SETS}')


def harness_record():
    files = {name: xc.sha256(HERE/name) for name in DRIVER_FILES if (HERE/name).is_file()}
    files.update({f'harness/{p.name}': xc.sha256(p) for p in sorted((HERE/'harness').glob('m_*.py'))})
    files['../holdout/holdout_common.py'] = xc.sha256(xc.EXPERIMENTS/'holdout'/'holdout_common.py')
    return dict(files=files, archive_drive_py=dict(path=f'{xc.ARCHIVE_HARNESS}/drive.py', sha256=xc.ARCHIVE_DRIVE_SHA256), bl2_protocol=dict(path=xc.BL2_PROTOCOL, sha256=xc.BL2_PROTOCOL_SHA256))


def probe(deps, when):
    try:
        if deps.probe is not None:
            return deps.probe(when)
        import unit_probe
        return unit_probe.probe(when)
    except Exception as exc:                        # noqa: BLE001 - a probe never fails a run
        return dict(when=when, error=f'{type(exc).__name__}: {exc}')


def copy_log(src, dst, limit=1_000_000):
    """drive.py copy_log: the method log in the record directory, truncated above 1 MB (the full log stays with the run)."""
    data = Path(src).read_bytes()
    if len(data) <= limit:
        Path(dst).write_bytes(data); return False
    lines = data.decode(errors='replace').splitlines()
    Path(dst).write_text('\n'.join(lines[:300]+[f'... [{len(lines)-1100} lines truncated; full log: {src}] ...']+lines[-800:])+'\n')
    return True


def resumed(path):
    """The resume's sibling of a cell directory (<pair>.resume): a resumed run never writes into the first run's directories."""
    return None if path is None else Path(path).with_name(Path(path).name+'.resume')


def lpba_spy_breach_problem(arm, pair_set, pair, paths, record_root):
    """LP-B diff review minor 3: an LPBA cell whose driver's spy stopped a read -- a refusal record of stage spy_breach, or a record (result, overridden, their resumed ones) whose spy did not
    pass -- is not rerun or resumed by an authorization: to BZ.  None otherwise; None for the hold-out and the development (unchanged)."""
    if not xc.is_lpba(pair_set):
        return None
    found = []
    for row in hc.cell_refusals(record_root, arm=arm, pair=pair, pair_set=pair_set, counted=False):
        try:
            doc = json.loads(Path(row['path']).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get('stage') == 'spy_breach':
            found.append(row['path'])
    for key in ('result', 'overridden', 'resume_result', 'resume_overridden'):
        path = paths.get(key)
        if path is not None and Path(path).is_file():
            try:
                doc = json.loads(Path(path).read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            if isinstance(doc, dict) and (doc.get('spy_passed') is False or (doc.get('failure') or {}).get('kind') in ('spy_not_passed', 'spy_breach')):
                found.append(str(path))
    return None if not found else f'{xc.lpba().SPY_BREACH_NO_RERUN}: {found}'


def policies(arm, pair_set, pair, paths, original, *, extra_exact=(), extra_write=(), extra_inside=()):
    record_root = xc.roots(pair_set)[0]
    from bayesreg.io import sealed as sealed_module
    bayesreg_tree = [] if Path(sealed_module.REPO).resolve() == xc.REPO.resolve() else [Path(sealed_module.REPO)]       # bayesreg's code tree: the unseal record, the pinned allowlist, the decision log
    authorizations = [] if (Path(hc.AUTH_ROOT).resolve() == xc.REPO.resolve() or xc.authorizations_problem(pair_set)) else [hc.authorization_dir(pair_set)]      # the committed authorizations (in the code tree, except in the tests)
    driver = xb_spy.driver_policy(work=paths['work'], record_dir=hc.arm_dir(arm, record_root), lock_dir=paths['lock'].parent, code_root=xc.REPO, results=hc.RESULTS, main_repo=hc.MAIN_REPO,
                                  holdout_root=record_root if xc.is_lpba(pair_set) else hc.HOLDOUT_ROOT,      # LP-B: an LPBA mode's stage/ and atropos/ forbidden like the hold-out's
                                  extra_exact=[original.parent/'manifest.json', *extra_exact], extra_read=[*bayesreg_tree, *authorizations],
                                  extra_write=[p for p in (paths['store'], resumed(paths['store']), resumed(paths['work']), *extra_write, *extra_inside) if p is not None])
    if xc.is_lpba(pair_set):            # LP-B quick diff review minor 1 (LP-C2 integration item 7): the code tree's and the main checkout's archive/ and docs/bayesreg/reviews/, as ArmPolicy and the review's lpba_policy
        driver['forbidden_roots'] = [*driver['forbidden_roots'], *(str(Path(t)/rel) for t in dict.fromkeys((Path(xc.REPO), Path(hc.MAIN_REPO))) for rel in xc.lpba().CODE_TREE_FORBIDDEN)]
        # LP-G revision 1 (decision_log 2026-11-21 ~2:42 AM CST): every file named one by one is read only in an LPBA mode (xb_policy rule 1a, read_exact; allowed_exact stays empty) -- the
        # inputs named above (the manifest, the predictions and HI's records, the method's run.json, the sealed lists) and the archive/ files check_frozen and check_environment open
        # (xb_core.archive_reads: archive/ is a forbidden root here); a write to any of them goes on to the rules below.  The hold-out's and the development's policies are unchanged.
        # archive_reads reads the frozen configuration (configs/frozen/holdout_B_v1_baselines.json, its sha256 checked) once here, before the spy is installed: that read is not in the
        # spy's record (check_frozen reads it again under the spy, and that read is).
        driver['read_exact'] = sorted({*driver['allowed_exact'], *(os.path.realpath(p) for p in xc.archive_reads(arm))})
        driver['allowed_exact'] = []
    return driver


def method_policy(arm, work):
    cfg = xc.ARMS[arm]
    groups = dict(fireants='fireants', ants='ants', deeds='deedsBCV', convexadam='convexadam', unigradicon='unigradicon', voxelmorph='voxelmorph')
    baselines = [xc.BASELINES/groups[g] for g in cfg.get('baselines', [])]
    return xb_spy.method_policy(name=f'xb method {arm}', work=work, code_root=xc.REPO, env_prefix=xc.ENVS/cfg['env'] if cfg.get('env') else None, baselines=baselines, results=hc.RESULTS,
                                main_repo=hc.MAIN_REPO, holdout_root=hc.HOLDOUT_ROOT)


def need_gib(arm, pair_set):
    """(need on D:, need on /home) of one run: the run's kept size on its store, and the run's temporary space plus its kept size on /home for a home-store arm (xb_core.KEEP_GIB, BL2's sizes)."""
    keep = xc.KEEP_GIB[arm]
    if pair_set == 'holdout' and xc.ARMS[arm]['store'] == 'd':
        return keep, 1.+keep
    return 0., 1.+keep


def batch_dir_of(pair_set, given=None):
    """(the batch directory this run's refusal records go to, a problem or None): `given` (--batch-dir) or XB_BATCH_DIR (the chain sets it to its batch), which must lie below
    <record root>/batches; otherwise <record root>/batches/direct (a run started by hand), with the problem noted in the record."""
    root = Path(os.path.realpath(hc.batch_root(xc.roots(pair_set)[0])))
    value = given if given is not None else os.environ.get('XB_BATCH_DIR')
    if not value:
        return root/'direct', None
    path = Path(os.path.realpath(value))
    if root not in path.parents:
        return root/'direct', f'the batch directory {value} is not below {root}: the refusal records go to {root/"direct"}'
    return path, None


def same_file(a, b):
    return a is not None and b is not None and os.path.realpath(a) == os.path.realpath(b)


def method_run_check(arm, pair_set, pair, method_run, *, pinned=None, why=None):
    """I4, m10 (prereg §2.4): ConvexAdam-v4seg needs the method's T88 run.json on this pair, given by --method-run, equal to the pinned path (xb_core.method_t88_run), clear of the spy's name rules,
    a file whose labels_sha256 holds both sides; another arm takes none.  Before the lock; reads that one file.  Returns the record (None for another arm); Refused (stage method_run) otherwise."""
    if xc.ARMS[arm]['kind'] != 'v4seg':
        if method_run is not None:
            raise hc.Refused('method_run', f'--method-run is only for ConvexAdam-v4seg, not {arm}')
        return None
    expected, want = pinned if pinned is not None else (xc.method_t88_run(pair_set, pair), xc.method_t88_sha256(pair_set, pair))     # pinned: read by run_cell before the spy (HB's records)
    if expected is None:
        if pinned is None and pair_set == 'holdout':                  # A5b diff review minor 3: why there is none (a move row without dst: the record's format); run_cell reads it before the spy
            why = xc.method_t88_lookup(pair)[1]
        raise hc.Refused('method_run', f'{pair_set}: the method arm\'s T88 run.json has no pinned path yet (HB pins the hold-out layout): ConvexAdam-v4seg is refused until then (prereg §2.4)'
                                       + (f'; {why}' if why else ''))
    if method_run is None:
        raise hc.Refused('method_run', f'--method-run is required for {arm}: the method\'s T88 run.json on this pair, {expected} (prereg §2.4)')
    if not same_file(method_run, expected):
        raise hc.Refused('method_run', f'{method_run} is not the pinned {expected}')
    why = xb_spy.xb_policy.name_breach(os.path.realpath(expected))
    if why:
        raise hc.Refused('method_run', f'{expected}: {why} (the spy\'s name rules)')
    if not Path(expected).is_file():
        raise hc.Refused('method_run', f'{expected}: no such file (the method arm has not written this pair)')
    try:
        document = json.loads(Path(expected).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise hc.Refused('method_run', f'{expected}: {exc}') from None
    labels = document.get('labels_sha256') if isinstance(document, dict) else None
    if not (isinstance(labels, dict) and all(isinstance(labels.get(k), str) and len(labels[k]) == 64 for k in ('fixed', 'moving'))):
        raise hc.Refused('method_run', f'{expected}: no labels_sha256 of both sides')
    if want is not None and xc.sha256(expected) != want:                  # HB review m13: the finalized record's sha256 of the file
        raise hc.Refused('method_run', f'{expected}: its sha256 is not the finalized record\'s {want}')
    return dict(path=str(expected), sha256=xc.sha256(expected), labels_sha256=dict(labels), rule='prereg §2.4: the maps must be the method\'s labels (checked again after the lock)')


def run_cell(arm, pair_set, pair, *, resume_rel=None, method_run=None, dry_run_out=None, deps=None, batch_dir=None, pinned=None):
    """One cell (the module docstring's order).  Every refusal before the lock is a refusal record in the batch directory (a dry run: its dry_run.json only), never the cell's result.json (I1)."""
    deps = Deps() if deps is None else deps
    log = deps.log
    t_start, utc_start = time.time(), hc.utc()
    try:
        xc.roots(pair_set)
    except hc.Refused as exc:
        log(f'refused: {exc}')
        return xc.EXIT['refused']
    batch, batch_problem = batch_dir_of(pair_set, batch_dir)
    state = dict(locked=False, problems=[], checks={}, spy=None)
    planned_cell = None

    def refuse(stage, reason, *, cls='refusal', code=None):
        code = xc.EXIT['refused'] if code is None else code
        log(f'refused before the lock ({stage}): {reason}')
        spy = state['spy']
        seq = None if dry_run_out is not None else hc.next_refusal_seq(xc.roots(pair_set)[0], arm=arm, pair=pair, pair_set=pair_set)     # P20: "newest" by number (HB diff review n4)
        doc = dict(tool=TOOL, arm=arm, pair=pair, pair_set=pair_set, stage=stage, refusal_class=cls, reason=reason, exit=code, planned=planned_cell, utc_start=utc_start, seq=seq,
                   unit=os.environ.get('XB_UNIT'), batch=str(batch), batch_problem=batch_problem, checks_run=sorted(state['checks']), problems=list(state['problems']),
                   spy=None if spy is None else spy.record(expect_original=False),
                   rule='I1 (C23.12-6, D107): a refusal before the lock is recorded here, beside the cell, never in its result.json; the cell runs again only with a committed rerun authorization '
                        'listing this record and its sha256 (xb_drive.py authorization-draft --kind rerun)')
        if dry_run_out is not None:             # A5a review m8: the refusal is a problem of the record (at the unseal with no record there: the expected one on P0)
            kind = 'no_unseal_record' if stage == 'unseal' and not hc.unseal_record_exists() else f'refused_{stage}'
            write_dry(dry_run_out, dict(doc, refused=dict(stage=stage, reason=reason), checks=state['checks']), refusal=(kind, f'refused ({stage}): {reason}'))
        else:
            log(f'refusal record {hc.write_refusal(batch, doc)}')
        return code

    if arm not in xc.ARMS:
        return refuse('admission', f'unknown arm {arm!r}; choose from {xc.ORDER} (not run: {sorted(xc.NOT_RUN)})')
    if xc.is_lpba(pair_set) and arm not in xc.LPBA_ARMS:          # LP-B (prereg §4.3): only the main family's four external arms on LPBA40
        return refuse('admission', f'{arm} is not one of the LPBA40 arms {xc.LPBA_ARMS} (LPBA40 prereg §4.3, §4.5)')
    try:
        hc.safe_name(pair, 'pair')
        fixed, moving = xc.subjects_of(pair_set, pair)
        paths = xc.cell_paths(arm, pair_set, pair)
        original = xc.pair_path(pair_set, pair)
    except hc.Refused as exc:
        return refuse(exc.stage, str(exc))
    cfg = xc.ARMS[arm]
    planned_cell = planned(pair_set, fixed, moving)
    pred_root = xc.hi_pred_root(pair_set) if cfg['kind'] == 'v4seg' else None
    extra_exact = [pred_root/s/n for s in (fixed, moving) for n in ('provenance.json', 'probs36.npz')] if pred_root is not None else []
    if pred_root is not None and (pair_set == 'holdout' or xc.is_lpba(pair_set)):
        extra_exact += [xc.hi_record_path(pred_root, s) for s in (fixed, moving)]       # A5a review I2: HI's subject record decides that a subject is written (first run + resume); LP-B: hi_lpba's record
    expected = xc.method_t88_run(pair_set, pair) if cfg['kind'] == 'v4seg' else None              # a hold-out pair: HB's finalized and move records, read here before the spy (HB review m13)
    state['method_pin'] = (expected, xc.method_t88_sha256(pair_set, pair) if cfg['kind'] == 'v4seg' else None)
    state['method_pin_why'] = (xc.method_t88_lookup(pair, pair_set)[1] if cfg['kind'] == 'v4seg' and (pair_set == 'holdout' or xc.is_lpba(pair_set)) and expected is None
                               else None)     # before the spy too
    if same_file(method_run, expected) and xb_spy.xb_policy.name_breach(os.path.realpath(expected)) is None:
        extra_exact.append(Path(expected))             # m10: only the pinned path, clear of the name rules, is named one by one
    from bayesreg.io.sealed import sealed_cache_is_empty
    if not sealed_cache_is_empty():                 # C23.6-5: the guard's first read of the sealed lists must happen under the spy
        return refuse('sealed_cache', 'the sealed list cache is already populated before the spy is installed (C23.6-5); a driver runs in a process of its own')
    sealed_paths = hc.sealed_list_paths()
    spy = xb_spy.DriverSpy(policies(arm, pair_set, pair, paths, original, extra_exact=[*extra_exact, *sealed_paths], extra_write=[dry_run_out] if dry_run_out else [],
                                    extra_inside=[hc.batch_root(xc.roots(pair_set)[0])]), original=original, counted=sealed_paths)
    state['spy'] = spy
    spy.install()
    try:
        return _run(arm, pair_set, pair, fixed, moving, cfg, paths, original, pred_root, spy, state, refuse, resume_rel=resume_rel, method_run=method_run, dry_run_out=dry_run_out, deps=deps,
                    t_start=t_start, utc_start=utc_start, pinned=pinned)
    except xb_spy.Breach as exc:
        if state['locked']:
            raise                                   # after the lock after_lock writes the record; a breach that escaped it leaves the chain's unit record (wrapper / rejudgment)
        return refuse('spy_breach', str(exc), cls='infrastructure', code=xc.EXIT['infra_failure'])        # m7: a controlled refusal, not a traceback
    except Exception as exc:                        # noqa: BLE001 - our own code before the lock
        if state['locked']:
            raise
        return refuse('wrapper_before_lock', f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}', cls='wrapper', code=xc.EXIT['wrapper_error'])
    finally:
        spy.remove()


def _run(arm, pair_set, pair, fixed, moving, cfg, paths, original, pred_root, spy, state, refuse, *, resume_rel, method_run, dry_run_out, deps, t_start, utc_start, pinned=None):
    log = deps.log
    checks, problems = state['checks'], state['problems']
    dry = dry_run_out is not None
    record_root = xc.roots(pair_set)[0]

    def problem(stage, message, kind='problem'):
        """A hold-out condition that refuses a run and that a dry run reports and passes (I5; the code tree).  `kind`: holdout_common.dry_run_summary's (A5a review m8)."""
        if not dry:
            raise hc.Refused(stage, message)
        problems.append(dict(stage=stage, problem=message, kind=kind))
        log(f'dry run: problem ({stage}): {message}')
    try:
        # ---- 1. the code tree (I1: before the admission) and the preregistration's pin (I5)
        checks['code'] = code = hc.code_identity(run=deps.code_run)
        if pair_set == 'holdout' and not code['clean']:
            problem('code', f'the code tree is not clean ({code}): a hold-out run runs from a committed tree')
        if pair_set == 'holdout':                   # A5a (contract C24.3 rule one; decision_log 2026-11-09 df8541d, 2026-11-10 ~2:59 AM CST (1), ~2:28 AM CST n8): the code is the pinned commit's
            pinned_record, issues = deps.pinned() if deps.pinned is not None else hc.pinned_code_problems(run=deps.code_run)
            checks['pinned_code'] = pinned_record
            if issues:
                problem('code', f'rule one (the pinned commit of runs/holdout_B_v1/pinned_code.json): {issues}', kind=hc.rule_one_kind(issues))
        if xc.is_lpba(pair_set):                    # LP-B (C26.9, C26.6-3): the code (formal: rule one, LP-C2's, C26.9-4, fail closed; feasibility: clean) and the pins, before the admission
            checks['lpba_code'], issues = xc.lpba().code_record(pair_set, pinned, run=deps.code_run)      # review minor 5: the chain's commit (--pinned) compared with this unit's
            if issues:
                problem('code', f'{issues}', kind='lpba_code')
            try:
                checks['pins'] = xc.lpba().lc.check_pins()
            except hc.Refused as exc:
                problem('pins', str(exc))
            lc = xc.lpba().lc
            checks['preregistration'] = dict(path=lc.PREREG_REL, pinned_sha256=lc.PREREG_SHA256, problem=None if checks.get('pins') else 'the pins did not hold (see pins)')
        else:
            checks['preregistration'] = dict(path=xc.PREREG_REL, pinned_sha256=xc.PREREG_SHA256, problem=xc.prereg_problem())
        if pair_set == 'holdout' and checks['preregistration']['problem']:
            problem('preregistration', checks['preregistration']['problem'])
        # ---- 2. a cell refused before: only with a committed rerun authorization (I1)
        try:
            if lpba_spy_breach_problem(arm, pair_set, pair, paths, record_root):       # LP-B diff review minor 3: a spy breach goes to BZ, never through a rerun
                raise hc.Refused(hc.RERUN_STAGE, f'{arm} {pair} ({pair_set}): {lpba_spy_breach_problem(arm, pair_set, pair, paths, record_root)}')
            if xc.authorizations_problem(pair_set) is not None:        # LP-B: no authorizations of this mode known: a cell refused before is refused again (no rerun record can be read)
                prior = hc.cell_refusals(record_root, arm=arm, pair=pair, pair_set=pair_set)
                if prior:
                    raise hc.Refused(hc.RERUN_STAGE, f'{arm} {pair} ({pair_set}) was refused before its lock {len(prior)} time(s): {xc.authorizations_problem(pair_set)}')
                checks['rerun_authorization'] = None
            else:
                checks['rerun_authorization'] = hc.rerun_authorization(record_root, tool=TOOL, arm=arm, pair=pair, pair_set=pair_set)
        except hc.Refused as exc:
            problem(exc.stage, str(exc))
        # ---- 3. admission
        try:
            admission, allowlist = admit(pair_set, fixed, moving, deps)
        except hc.Refused as exc:
            return refuse(exc.stage, str(exc), code=xc.EXIT['admission_rejected'])
        checks['admission'] = admission
        # ---- 4.-7. the checks before the lock
        if paths['arm_stop'].exists():
            raise hc.Refused('arm_stopped', f'{paths["arm_stop"]} exists (P19): no further pair of {arm} runs')
        checks['frozen'] = frozen = xc.check_frozen(arm)
        config = xc.frozen_config()[0]
        checks['environment'] = xc.check_environment(arm, config=config, run=deps.env_run)
        resume = None
        if resume_rel is not None and lpba_spy_breach_problem(arm, pair_set, pair, paths, record_root):     # LP-B diff review minor 3
            raise hc.Refused('resume', lpba_spy_breach_problem(arm, pair_set, pair, paths, record_root))
        if resume_rel is not None and xc.authorizations_problem(pair_set) is not None:
            raise hc.Refused('resume', xc.authorizations_problem(pair_set))
        if resume_rel is not None:
            previous = paths['overridden'] if paths['overridden'].is_file() else paths['result']        # I2: the record the unit's evidence overrode is resumed through its overriding record
            resume = hc.load_resume(resume_rel, tool=TOOL, arm=arm, pair=pair, pair_set=pair_set, lock=paths['lock'], previous_record=previous, complete=lambda: xc.attempt_complete(paths))
            checks['resume'] = dict(path=resume.path, sha256=resume.sha256, previous_record=resume.previous_record)
        hc.check_once(paths['lock'], resume=resume)
        if resume is not None:
            first = (hc.hi.lock_lines(paths['lock']) or [{}])[0]
            if first.get('frozen_sha256') != frozen['config']['sha256']:
                raise hc.Refused('resume', f"the frozen configuration's sha256 {frozen['config']['sha256']} is not the first run's {first.get('frozen_sha256')}: a rerun runs the same configuration "
                                           f"(the lead, revision 1b: configs/frozen/ must still match)")
        result_path = paths['resume_result'] if resume else paths['result']
        work = paths['work'].with_name(paths['work'].name+'.resume') if resume else paths['work']
        if result_path.exists() or work.exists():
            raise hc.Refused('once', f'{result_path} or {work} exists: a cell writes its record and its work directory once')
        checks['method_run'] = method_run_check(arm, pair_set, pair, method_run, pinned=state.get('method_pin'), why=state.get('method_pin_why'))
        checks['pair_input'] = pair_input = xc.check_pair_input(original, fixed=fixed, moving=moving, allowlist=allowlist)
        checks['predictions'] = ({side: xc.check_prediction(pred_root, s, shape=pair_input['grid']['shape_zyx']) for side, s in (('fixed', fixed), ('moving', moving))}
                                 if pred_root is not None else None)
        if pred_root is not None and (pair_set == 'holdout' or xc.is_lpba(pair_set)):          # A5a review I2 point 3: only a subject HI wrote (its record, in the one root: the first run's or the resume's)
            checks['hi_records'] = {side: xc.check_hi_record(pred_root, c['subject'], c['probs36_sha256'], c['provenance_sha256']) for side, c in checks['predictions'].items()}
    except hc.Refused as exc:
        return refuse(exc.stage, str(exc))
    # ---- 8. the disks, the machine and the arm's precheck (m8: the GPU too, before the lock)
    need_d, need_home = need_gib(arm, pair_set)
    try:
        checks['disk_before'] = hc.disk_guard(need_d_gib=need_d, need_home_gib=need_home, usage=deps.disk_usage)
        own_units = tuple(u for u in (os.environ.get('XB_UNIT'),) if u)
        checks['machine'] = deps.gate() if deps.gate is not None else hc.wait_for(lambda: hc.machine_gate(own_units=own_units, parent_units=(xc.chain_unit(pair_set),)), log=log)
        checks['precheck_before_lock'] = deps.precheck(cfg['device'] == 'cuda', log=log)
    except (hc.InfrastructureFailure, hc.Refused) as exc:
        return refuse(getattr(exc, 'kind', None) or getattr(exc, 'stage', 'machine'), str(exc), cls='infrastructure', code=xc.EXIT['infra_failure'])
    if dry:
        write_dry(dry_run_out, dict(arm=arm, pair=pair, pair_set=pair_set, checks=checks, problems=problems, would_lock=str(resume.lock_path if resume else paths['lock']),
                                    spy=spy.record(expect_original=False),
                                    note='dry run: every check before the lock ran; no lock taken, nothing of the pair decoded, nothing run; a problem refuses a real run'))
        return xc.EXIT['refused'] if problems else 0
    # ---- 9. the lock
    lock_path = Path(resume.lock_path) if resume else paths['lock']
    rerun = checks.get('rerun_authorization')
    prereg_file = xc.REPO/(xc.lpba().lc.PREREG_REL if xc.is_lpba(pair_set) else xc.PREREG_REL)
    code_commit = ((checks.get('pinned_code') or {}).get('pinned') if pair_set == 'holdout' else (checks.get('lpba_code') or {}).get('commit') if xc.is_lpba(pair_set)
                   else checks['code']['commit'])        # a hold-out cell: P1 (rule one); an LPBA cell: P_L (formal) or HEAD (feasibility); HEAD beside it
    line = hc.lock_line(tool=TOOL, arm=arm, pair=pair, pair_set=pair_set, event='resume' if resume else 'first run', record=str(result_path), work=str(work), code_commit=code_commit,
                        code_head=checks['code']['commit'],
                        frozen_sha256=checks['frozen']['config']['sha256'], unseal_sha256=(checks['admission'].get('unseal') or {}).get('sha256'), pair_sha256=checks['pair_input']['pair_sha256'],
                        preregistration_sha256=xc.sha256(prereg_file) if prereg_file.is_file() else None,
                        resume=None if resume is None else dict(path=resume.path, sha256=resume.sha256),
                        rerun_authorization=None if rerun is None else dict(path=rerun['path'], sha256=rerun['sha256']),
                        method_run=None if checks['method_run'] is None else dict(path=checks['method_run']['path'], sha256=checks['method_run']['sha256']))
    try:
        hc.take_lock(lock_path, line)
    except hc.Refused as exc:
        return refuse(exc.stage, str(exc))
    state['locked'] = True
    lock = dict(path=str(lock_path), line=line, sha256=xc.sha256(lock_path))
    log(f'XB arm {arm} pair {pair} ({pair_set}); lock {lock_path}; work {work}')
    return after_lock(arm, pair_set, pair, fixed, moving, cfg, paths, original, pred_root, spy, checks=checks, lock=lock, resume=resume, result_path=result_path, work=work,
                      method_run=None if checks['method_run'] is None else checks['method_run']['path'], deps=deps, t_start=t_start, utc_start=utc_start)


def after_lock(arm, pair_set, pair, fixed, moving, cfg, paths, original, pred_root, spy, *, checks, lock, resume, result_path, work, method_run, deps, t_start, utc_start):
    log = deps.log
    work.mkdir(parents=True, exist_ok=False)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    logfile = work/'method_log.txt'
    rec = dict(status='ok', failure=None, derived=None, maps=None, method=None, disp=None, own_warped=None, export_check=None, cleanup=None, move=None)
    rec['unit'] = dict(name=os.environ.get('XB_UNIT'), properties=os.environ.get('XB_UNIT_PROPERTIES'), expected=list(xc.UNIT_PROPERTIES), probe_start=probe(deps, 'start'))
    shape = tuple(checks['pair_input']['grid']['shape_zyx'])
    d = own = None
    try:
        z, rec['derived'] = xc.derive_inputs(original, work/'inputs.npz', shape=shape)
        if cfg['kind'] == 'v4seg':
            if xc.sha256(method_run) != checks['method_run']['sha256']:
                raise hc.InfrastructureFailure('input_rejected', f'{method_run} changed since the check before the lock')
            rec['maps'] = xc.write_maps(pred_root, fixed, moving, work/'maps.npz', shape=shape, checked=checks['predictions'], method_run=method_run)
        ctx = xc.Ctx(policy=method_policy(arm, work), log=log, step=deps.step, precheck=deps.precheck, inputs=work/'inputs.npz', maps=work/'maps.npz')
        d, own, alts, out = xc.ARM_FUNCTIONS[cfg['kind']](arm, cfg, work, logfile, z, ctx)
        rec['method'] = out
        d = xc.finish_field(d, shape, work)
        if not (work/'own_warped.npz').exists():
            import numpy as np
            np.savez_compressed(work/'own_warped.npz', w=np.asarray(own).astype(np.float32))
        rec['export_check'] = xc.export_check(z['fixed'], z['moving'], d, own, alts)
        log(f"check corr {rec['export_check']['check']['corr']:.8f}, suspicious {rec['export_check']['suspicious']}")
    except Exception as exc:                        # noqa: BLE001 - every outcome after the lock ends in the terminal record
        status, kind = xc.classify(exc)
        rec.update(status=status, failure=dict(kind=kind, reason=str(exc), exception=type(exc).__name__, traceback=traceback.format_exc()))
        if rec['method'] is None and isinstance(getattr(exc, 'out', None), dict):
            rec['method'] = exc.out
        log(f'{status} ({kind}): {exc}')
    del d, own
    if logfile.exists():
        rec['method_log_truncated_in_record_copy'] = copy_log(logfile, result_path.parent/('method_log.txt' if not resume else 'method_log_resume.txt'))
    if rec['status'] == 'ok':
        try:
            rec['disp'] = dict(file=str(work/'disp.npz'), sha256=xc.sha256(work/'disp.npz'))
            rec['own_warped'] = dict(file=str(work/'own_warped.npz'), sha256=xc.sha256(work/'own_warped.npz'))
            manifest = result_path.parent/('intermediates.tsv' if not resume else 'intermediates_resume.tsv')
            listed = xc.intermediates_manifest(work, manifest)
            rec['cleanup'] = dict(manifest=str(manifest), manifest_sha256=xc.sha256(manifest), n_files=len(listed), bytes=sum(r['bytes'] for r in listed), deleted=False,
                                  rule='nothing deleted by the driver; the reviewed deletion step (experiments/xb/delete_verified.sh) deletes from an approved manifest after the audit')
            if paths['store'] is not None:
                move_path = result_path.parent/('moves.json' if not resume else 'moves_resume.json')
                store = paths['store'] if not resume else paths['store'].with_name(paths['store'].name+'.resume')
                names = [n for n in xc.KEEP_FILES if (work/n).is_file()]
                rec['move'] = dict(record=str(move_path), dst=str(store), passed=False)
                moved = hc.move_files(work, store, move_path, names=names)
                rec['move']['passed'] = moved['passed']
                rec['disp'] = dict(file=str(store/'disp.npz'), sha256=next(r['sha256_after'] for r in moved['files'] if r['file'] == 'disp.npz'))
                rec['own_warped'] = dict(file=str(store/'own_warped.npz'), sha256=next(r['sha256_after'] for r in moved['files'] if r['file'] == 'own_warped.npz'))
        except Exception as exc:                    # noqa: BLE001
            status, kind = xc.classify(exc)
            rec.update(status=status, failure=dict(kind=kind, reason=str(exc), exception=type(exc).__name__, traceback=traceback.format_exc(), phase='after the export: cleanup or move (both copies kept)'))
            log(f'{rec["status"]} after the export ({kind}): {exc}')
    try:
        rec['disk_after'] = hc.disk_reading(deps.disk_usage)
    except Exception as exc:                        # noqa: BLE001
        rec['disk_after'] = f'{type(exc).__name__}: {exc}'
    rec['unit']['probe_end'] = probe(deps, 'end')
    spies = dict(driver=spy.record(expect_original=True))
    for step in (rec.get('method') or {}).get('steps', []):
        spies.setdefault('method_steps', []).append(dict(step=step.get('name'), fd=step.get('fd_spy'), method=step.get('method_spy')))
    passed = spy_passed(spies)
    if not passed and rec['status'] in ('ok', 'method_failure', 'unit_limit'):
        # m7: a spy that did not pass (a breach some library swallowed, or no positive condition) makes the run an infrastructure failure, whatever it returned; its disp stays listed, not used
        rec['failure'] = dict(kind='spy_not_passed', reason=f"the spies did not pass: driver breaches {spies['driver']['breaches'][:3]}, positive condition {spies['driver']['positive_condition']}",
                              superseded=dict(status=rec['status'], failure=rec['failure']))
        rec['status'] = 'infra_failure'
        log(f'infra_failure (spy_not_passed): {rec["failure"]["reason"]}')
    document = result_document(arm, pair_set, pair, rec, checks=checks, lock=lock, resume=resume, spies=spies, spy_ok=passed, paths=paths, work=work, t_start=t_start, utc_start=utc_start)
    hc.write_json_once(result_path, document)
    log(f'wrote {result_path} (status {document["status"]})')
    if document['status'] == 'ok' and document['export_check']['suspicious']:
        suspicious = xc.suspicious_pairs(arm, pair_set)
        if len(suspicious) >= xc.p19_stop_at(pair_set) and not paths['arm_stop'].exists():
            rule = ('prereg §4.5 P19: the third pair of an arm whose export check is suspicious (corr < .999) stops the arm; the pairs run stay, described only; F1/F2 record the arm as not judged '
                    '(p = 1); to the lead') if not xc.is_lpba(pair_set) else (
                   'LPBA40 prereg §6.6 (裁决 (8)): the second pair of an arm whose export check is suspicious (corr < .999) stops the arm; the pairs run stay, described only; L1/L2 record the arm '
                   'as not judged (p = 1); to the lead')
            stop = dict(schema=xc.SCHEMA.format('arm_stop'), arm=arm, pair_set=pair_set, utc=hc.utc(), suspicious_pairs=suspicious, at_pair=pair, rule=rule)
            if xc.is_lpba(pair_set):
                stop['stop_at'] = xc.p19_stop_at(pair_set)     # LP-B review I1: only an LPBA arm's record has the key (the hold-out's arm_stopped.json keys unchanged)
            hc.write_json_once(paths['arm_stop'], stop)
            log(f'P19: {arm} stopped ({paths["arm_stop"]})')
    return xc.EXIT[document['status']]


def spy_passed(spies):
    return all(s.get('passed', False) for s in [spies['driver'], *[x for st in spies.get('method_steps', []) for x in (st['fd'], st['method']) if x is not None]])


def result_document(arm, pair_set, pair, rec, *, checks, lock, resume, spies, spy_ok, paths, work, t_start, utc_start):
    cfg = xc.ARMS[arm]
    status = rec['status']
    return dict(
        schema=xc.SCHEMA.format('result'), status=status, arm=arm, pair=pair, pair_set=pair_set, fixed_subject=xc.subjects_of(pair_set, pair)[0], moving_subject=xc.subjects_of(pair_set, pair)[1],
        role=cfg['role'], kind=cfg['kind'],
        env=cfg.get('env', {'ants': 'ANTs 2.6.5 binary', 'deeds': 'deedsBCV binaries'}.get(cfg['kind'])), factor=cfg.get('factor'),
        failure=rec['failure'], wrapper_error_policy=xc.WRAPPER_ERROR_POLICY if status == 'wrapper_error' else None,
        no_output_class=None if (resume is None or status in ('ok', 'method_failure', 'unit_limit')) else {'wrapper_error': 'wrapper', 'infra_failure': 'infrastructure'}.get(status, status),
        disp=rec['disp'], own_warped=rec['own_warped'], export_check=rec['export_check'], lock=lock, resume=None if resume is None else dict(path=resume.path, sha256=resume.sha256),
        derived_inputs=rec['derived'], maps=rec['maps'], method=rec['method'], unit=rec['unit'], spies=spies, spy_passed=spy_ok,
        checks=checks, harness=harness_record(), store=dict(kind=cfg['store'] if pair_set == 'holdout' else 'home', work=str(work), d=None if paths['store'] is None else str(paths['store'])),
        cleanup=rec['cleanup'], move=rec['move'],
        disk_after=rec.get('disk_after'), utc_start=utc_start, utc_end=hc.utc(), seconds_driver_wall=round(time.time()-t_start, 2), timing_definitions=xc.TIMING_DEFINITIONS,
        preregistration=xc.lpba().lc.PREREG_REL if xc.is_lpba(pair_set) else xc.PREREG_REL, rule='the terminal record of one (arm, pair) (prereg §5.4, §5.6): status one of ' + ', '.join(xc.STATUSES) + '; written once; an overridden.json beside it '
                                            '(written by the chain when the unit\'s evidence contradicts it) supersedes it (I2)')


def write_dry(out, document, *, refusal=None):
    """The dry run's record <out>/dry_run.json.  A5a review m8: in the shape every hold-out entry's dry run writes (holdout_common.dry_run_summary: passed, problems as strings, expected,
    unexpected, p0_ok -- the checklist reads these, not the exit code); the problems by stage (the dicts of _run's problem()) are kept as problems_by_stage; `refusal`: (kind, message) of a refusal
    that ended the dry run."""
    by_stage = list(document.get('problems') or [])
    rows = [(p.get('kind', 'problem'), f"{p['stage']}: {p['problem']}") for p in by_stage] + ([refusal] if refusal is not None else [])
    hc.write_json_once(Path(out)/'dry_run.json', dict(document, problems_by_stage=by_stage, schema=xc.SCHEMA.format('dry_run'), utc=hc.utc(), **hc.dry_run_summary(rows)))


def dry_out_problem(out):
    """A dry run writes only into a new or empty directory outside the code tree and the results root (mktemp -d)."""
    target = Path(out).resolve()
    if target.is_relative_to(xc.REPO.resolve()) or target.is_relative_to(hc.RESULTS.resolve()):
        return f'{out}: a dry run writes outside the code tree and the results root'
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        return f'{out} exists and is not an empty directory'
    return None


# ---- the authorizations (I1, I2, I3; prereg §5.4, revision 1c item 4): drafts printed for the lead, committed by the lead; the runner reads only committed ones --------------------------------
def authorization_draft(kind, arm, pair_set, pair, *, written_by, words, time_local, reason):
    """The draft of a committed authorization for one cell (holdout_common.AUTH_KEYS[kind]), its fields read from the cell's records now: resume (the first run's lock and the record to resume: the
    overriding record when the unit's evidence overrode the driver's, else result.json), rerun (every refusal record of the cell, path and sha256), no_resume (the record and its sha256, 未批补跑;
    a cell refused before its lock, P20: the newest copy of its refusal records in its record directory, holdout_common.newest_refusal).
    Returns (suggested path, document); writes nothing: the lead writes the file, BZ's words in it, and commits it (only a committed, unchanged file counts)."""
    if kind not in hc.AUTH_KINDS:
        raise hc.Refused('authorization', f'kind {kind!r}: one of {hc.AUTH_KINDS}')
    if written_by not in hc.RESUME_WRITERS or not all((v or '').strip() for v in (words, time_local, reason)):
        raise hc.Refused(kind, f'written_by one of {hc.RESUME_WRITERS}, BZ\'s words, the local time and the reason are required')
    if xc.authorizations_problem(pair_set) is not None:
        raise hc.Refused(kind, xc.authorizations_problem(pair_set))
    paths = xc.cell_paths(arm, pair_set, pair)
    record = paths['overridden'] if paths['overridden'].is_file() else paths['result']
    doc = dict(schema=f'bayesreg.{TOOL}.{kind}.v1', written_by=written_by, words=words, time_local=time_local, utc=hc.utc(), reason=reason, tool=TOOL, arm=arm, pair=pair, pair_set=pair_set)
    if kind == 'resume':
        if not paths['lock'].is_file():
            raise hc.Refused(kind, f'{paths["lock"]}: no lock, nothing to resume (a cell refused before its lock needs --kind rerun)')
        doc.update(lock_sha256=xc.sha256(paths['lock']), previous_record=str(record), previous_record_sha256=xc.sha256(record) if record.is_file() else None)
    elif kind == 'rerun':
        refusals = hc.cell_refusals(xc.roots(pair_set)[0], arm=arm, pair=pair, pair_set=pair_set)
        if not refusals:
            raise hc.Refused(kind, f'{arm} {pair}: no refusal record')
        doc.update(refusals=[dict(path=r['path'], sha256=r['sha256']) for r in refusals])
    else:
        if not record.is_file() and not paths['lock'].exists():          # P20 (prereg revision 2 item 17 (f); A5a): refused before its lock, again after a rerun: the newest refusal record's copy
            found = hc.newest_refusal(paths['record'])
            if found is None:
                raise hc.Refused(kind, f'{record}: no terminal record, and no refusal record copied into {paths["record"]}/{hc.REFUSAL_COPIES} (xb_chain copies them when the driver refuses)')
            doc.update(record=str(found[0]), record_sha256=xc.sha256(found[0]), note=hc.NO_RESUME_NOTE)
        else:
            if not record.is_file():
                raise hc.Refused(kind, f'{record}: no terminal record')
            if xc.attempt_complete(paths):
                raise hc.Refused(kind, f'{record} is a complete record: nothing to decline')
            doc.update(record=str(record), record_sha256=xc.sha256(record), note=hc.NO_RESUME_NOTE)
    directory = hc.authorization_dir(pair_set)
    k = 1
    while (directory/f'{hc.AUTH_PREFIX[pair_set]}{TOOL}_{kind}_{k}.json').exists():
        k += 1
    return directory/f'{hc.AUTH_PREFIX[pair_set]}{TOOL}_{kind}_{k}.json', doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    q = sub.add_parser('authorization-draft', help='print the draft of a committed authorization (resume, rerun, no_resume) for the lead; writes nothing')
    q.add_argument('--kind', required=True, choices=hc.AUTH_KINDS)
    q.add_argument('--arm', required=True)
    q.add_argument('--pair-set', required=True, choices=PAIR_SETS)
    q.add_argument('--pair', required=True)
    q.add_argument('--written-by', required=True)
    q.add_argument('--words', required=True)
    q.add_argument('--time-local', required=True)
    q.add_argument('--reason', required=True)
    for name in ('run', 'dry-run'):
        p = sub.add_parser(name)
        p.add_argument('--arm', required=True)
        p.add_argument('--pair-set', required=True, choices=PAIR_SETS)
        p.add_argument('--pair', required=True)
        p.add_argument('--method-run', default=None, help='ConvexAdam-v4seg (required): the method\'s t88/run.json on this pair, the pinned path of xb_core.method_t88_run (prereg §2.4)')
        p.add_argument('--batch-dir', default=None, help='where refusal records go (default XB_BATCH_DIR, else <record root>/batches/direct)')
        p.add_argument('--pinned', default=None, help='LP-B: an LPBA cell, the chain\'s commit (formal: P_L; feasibility: HEAD), compared with this unit\'s code (review minor 5)')
        if name == 'run':
            p.add_argument('--resume-record', default=None, help='a committed resume authorization, <authorization dir>/<prefix>xb_resume_<k>.json (holdout: splits/bayesreg_v1/holdout_B_v1_; '
                                                                 'dev: runs/20261107-xb/authorizations/dev_; prereg §5.4)')
        else:
            p.add_argument('--out', required=True)
    o = ap.parse_args(argv)
    if o.command == 'authorization-draft':
        try:
            path, document = authorization_draft(o.kind, o.arm, o.pair_set, o.pair, written_by=o.written_by, words=o.words, time_local=o.time_local, reason=o.reason)
        except hc.Refused as exc:
            print(f'refused: {exc}', file=sys.stderr)
            return 2
        print(f'# suggested path (commit it there; only a committed, unchanged file counts): {path}', file=sys.stderr)
        print(json.dumps(document, indent=1, ensure_ascii=False))
        return 0
    if o.command == 'dry-run':
        problem = dry_out_problem(o.out)
        if problem:
            print(problem, file=sys.stderr)
            return 2
        Path(o.out).mkdir(parents=True, exist_ok=True)
        return run_cell(o.arm, o.pair_set, o.pair, method_run=o.method_run, dry_run_out=Path(o.out).resolve(), batch_dir=o.batch_dir, pinned=o.pinned)
    return run_cell(o.arm, o.pair_set, o.pair, resume_rel=o.resume_record, method_run=o.method_run, batch_dir=o.batch_dir, pinned=o.pinned)


if __name__ == '__main__':
    sys.exit(main())
