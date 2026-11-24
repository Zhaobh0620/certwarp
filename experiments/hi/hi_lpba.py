"""LP-A item 3 (runs/20261119-lpba/任务书_A_数据准入HI与Atropos.md): HI's LPBA40 mode -- the structure-layer inference of unet35_v4_ens (hs_a, all_folds_mean) for the subjects
of one LPBA mode, from the stage arrays.  Contract C26.5-3 (L28, L48), C26.2-3 (the output root written in the code), C26.3-5; prereg §2.2 (P7; "分割器的输入取 stage 影像"),
§2.4 (no temperature recalibration), §7.2 (HI row), §12 item 7.  hi_core.py and run_hi.py are not changed; this module is the mode.

The backend is hs_a as it is (C26.5-3: hs_seg.uf_plan, hs_seg.uf4_probabilities, hs_seg.write_prediction, through hi_core.predict_hs_a and provenance_for): the members' own
normalize_image, the temperatures as registered.  The one difference is the image: hs_seg.uf4_probabilities reads it through bayesreg.seg.unet.data.read_image(files, subject)
(aligned_norm.nii.gz of a data root); here, for the length of the subject's call only, that module attribute is the stage reader (stage_reader): it gives the stage array of
the subject -- fixed or moving of its pair.npz, loaded through bayesreg.io.pair.load_pair with the LPBA allowlist and its sha256 recomputed against the manifest's
sources.<side>.sha256 (P7; a subject in two pairs, as S01 and S02 in the feasibility pairs, must have the same array in both) -- with read_image's metadata keys, sha256 the
array's (so the provenance's input_sha256 is P7's value: C26.5-3).  It refuses any other subject and any other `files`, and the attribute is restored after the call, also on
an exception; the real read_image is checked to be the one bound before (atropos_holdout.tissue_layer_check's way).  No file of /mnt/d/data is opened.

Order of a run (hi_lpba.run):
  1. the output root roots(pair_set).hi_out written in the code (C26.2-3; as hi_core.HOLDOUT_OUT_SUGGESTED), new; the code (lpba_common.check_code: one clean tree; a formal
     run also rule one, C26.9-4, LP-C2's lpba_common.rule_one -- it refuses unless it passes);
  2. the spy installed (lpba_common.Policy, the guard's cache of the sealed lists empty);
  3. the admission (lpba_common.admit: pins, the pinned LPBA allowlist, the mode's names, sealed lists, check_subject; no unseal record: LPBA40 is not in SG, C26.5-3);
  4. the registry against HI's anchor (hi_core.check_registry), every label_contact_subjects on the LPBA allowlist and in no never_for_training source
     (lpba_common.check_contact; L48 infer.select_fold_models checks it again in hs_seg.uf_plan, with check_subject_admitted: L28, L22);
  5. HI's development-check decision (hi_core.check_decision: hs_a, the check's environment), the members re-hashed (hi_core.member_files), every subject's plan
     (hi_core.plan_of = hs_seg.uf_plan: check_subject_admitted, D24 all_folds_mean);
  6. the stage (lpba_common.check_stage); the hold file; the machine gate (holdout_common.machine_gate: hi_core's gate, MemAvailable, load); the device (cuda only, the
     environment of the check: hi_core.prepare_device, check_environment);
  7. the lock <root>/locks/lpba_hi.lock (O_EXCL, after every check);
  8. subject after subject: its stage array(s) loaded and checked, the prediction (hi_core.predict_hs_a inside stage_reader), subjects/<s>.json; hi_record.json at the end, or
     hi_stopped.json on a failure (kept, not rerun; no resume here).
Units: tag-lp-hi (tag-lp-feas-hi).

    python -B experiments/hi/hi_lpba.py --pair-set lpba40_feas --dry-run --out DIR       (no image, no GPU; the admission, the pins, the registry, the members' presence by stat)
    python -B experiments/hi/hi_lpba.py --pair-set lpba40_feas [--device cuda]
Exit codes: 0 done (or a dry run that passed), 2 usage, 8 refused, stopped by the spy or a check failed (the record kept), 1 any other failure.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
for _p in (HERE, EXPERIMENTS/'lpba'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import hi_core as hi  # noqa: E402
import lpba_common as lc  # noqa: E402

hc, hp = lc.hc, lc.hp

from bayesreg.io.guard import SubjectSealed  # noqa: E402
from bayesreg.io.pair import ProvenanceRejected  # noqa: E402
from jd_curl.validation import InputValidationError  # noqa: E402

SCHEMA = 'bayesreg.lpba40.hi.v1'
SCRIPT = 'experiments/hi/hi_lpba.py'
UNIT_TOOL = 'hi'      # the unit name's last part (tag-lp-[feas-]hi); not an authorization tool: holdout_common's TOOL names those (test_pinned_code); LP-C2 integration
LOCK_NAME = 'lpba_hi.lock'
RUNTIME_S = 4*3600
NEED_HOME_GIB = dict(lpba40_test=16., lpba40_feas=2.)       # 〔估算〕probs36 and structure_groups, about 0.4 GiB a subject (run_hi.MIB_PER_SUBJECT 384 at 160x192x224; 181x217x181 is a little larger)
REGISTRY_REL = hi.CONTRACT_SEGMENTER['registry']
GROUPING = 'structure13_v1'
READER = 'experiments/hi/hi_lpba.stage_reader: the stage array (bayesreg.io.pair.load_pair with the LPBA allowlist; sha256 = manifest sources.<side>.sha256, prereg §2.2 P7)'


def launch_line(pair_set, *, code=None, log=None):
    """The start string of this entry's unit for `pair_set` (hp_core.launch_line: MemoryMax 12G, swap 0, RuntimeMaxSec RUNTIME_S): the dry run prints it; the feasibility chain runs it
    with the chain's log and wraps only the command with the end probe (LP-C2 integration review minor 3: the feasibility unit is the formal one's)."""
    return hp.launch_line(unit=lc.unit_name(pair_set, UNIT_TOOL), script=SCRIPT, args=['--pair-set', pair_set], runtime_s=RUNTIME_S, memory_max='12G', code=code, log=log)


class StageFiles:
    """The `files` handed to hs_seg.uf4_probabilities in this mode: it names the stage and the pair set; it has no image_path (no NIfTI is opened)."""

    def __init__(self, pair_set, stage):
        self.pair_set, self.stage, self.root = pair_set, str(stage), str(stage)

    def __repr__(self):
        return f'StageFiles({self.pair_set!r}, {self.stage!r})'


array_sha256 = lc.array_sha256                 # review minor 4: he_core.array_sha256 itself


def stage_arrays(subject, pairs, stage, allowlist):
    """The subject's stage array: from every pair of `pairs` it is in, through load_pair with the LPBA allowlist (fixed or moving), its sha256 recomputed against the manifest's
    sources.<side>.sha256; one array if the subject is in two pairs (else the run stops).  Returns (array, metadata of read_image's keys, record)."""
    from bayesreg.io.pair import load_pair
    found = []
    for fixed, moving in pairs:
        for side, s in (('fixed', fixed), ('moving', moving)):
            if s != subject:
                continue
            path = Path(stage)/lc.pair_name(fixed, moving)/'pair.npz'
            pair = load_pair(path, allowlist=allowlist)
            if (pair.fixed_subject, pair.moving_subject) != (fixed, moving):
                raise lc.Refused('stage', f'{path}: the manifest names {pair.fixed_subject}, {pair.moving_subject}')
            array = getattr(pair, side)
            digest = array_sha256(array)
            if digest != pair.source_sha256[side]:
                raise lc.Refused('stage', f'{path}:{side}: the array sha256 {digest} is not the manifest\'s {pair.source_sha256[side]} (P7)')
            found.append((lc.pair_name(fixed, moving), side, array, digest, pair))
    if not found:
        raise lc.Refused('admission', f'{subject} is in none of the pairs')
    digests = sorted({f[3] for f in found})
    if len(digests) != 1:
        raise lc.Refused('stage', f'{subject}: two different arrays in its pairs {[(f[0], f[1], f[3]) for f in found]} (prereg §2.2 P7)')
    name, side, array, digest, pair = found[0]
    meta = dict(source=f'{Path(stage)/name/"pair.npz"}:{side}', sha256=digest, shape_zyx=list(pair.grid.shape), spacing_xyz=[float(s) for s in pair.grid.spacing], unit=pair.grid.unit,
                labels=False, reader=READER, subject=subject, grid_shape_zyx=list(pair.grid.shape), origin_xyz=[float(a[0]) for a in pair.grid.axes])
    record = dict(subject=subject, input_sha256=digest, pairs=[dict(pair=f[0], side=f[1], sha256=f[3], pair_sha256=f[4].provenance['pair_sha256'],
                                                                    lpba40_guard=f[4].provenance['subject_guard'].get('lpba40_guard')) for f in found])
    return array, meta, record


@contextmanager
def stage_reader(files, subject, array, meta):
    """For the length of the block, bayesreg.seg.unet.data.read_image(files, s) gives (array, meta) for exactly this `files` object and `subject`, refusing anything else;
    restored after, also on an exception.  Refuses to replace a read_image that is not bayesreg's own (someone else changed it)."""
    from bayesreg.seg.unet import data
    original = data.read_image
    if getattr(original, '__module__', None) != 'bayesreg.seg.unet.data' or original.__name__ != 'read_image':
        raise RuntimeError('bayesreg.seg.unet.data.read_image is not bayesreg\'s own: not replaced')
    calls = []

    def read(given_files, given_subject):
        if given_files is not files or given_subject != subject:
            raise lc.Refused('reader', f'the stage reader of {subject} was asked for {given_subject!r} with {given_files!r}')
        calls.append(given_subject)
        return array, dict(meta)
    data.read_image = read
    try:
        yield calls
    finally:
        data.read_image = original


def predict(subject, *, files, array, meta, registry, models, plan, policy, device, grouping, grouping_check, pred_root, code, environment, command, allowlist, hi_block):
    """hi_core.predict_hs_a for one subject with the stage reader in place; the reader must have been asked exactly once."""
    with stage_reader(files, subject, array, meta) as calls:
        row = hi.predict_hs_a(subject, files=files, registry=registry, models=models, plan=plan, policy=policy, device=device, grouping=grouping, grouping_check=grouping_check,
                              pred_root=pred_root, code=code, environment=environment, command=command, allowlist=allowlist, hi=hi_block)
    if calls != [subject] or row['image_sha256'] != meta['sha256']:
        raise lc.Refused('reader', f'{subject}: the stage reader was called {calls}, image sha256 {row["image_sha256"]} (expected {meta["sha256"]})')
    return row


def select_models(registry, subjects, allowlist):
    """L28, L48 by ids only (what hs_seg.uf_plan runs before it looks for the members): infer.check_subject_admitted, then infer.select_fold_models (D24): all_folds_mean over
    every registered model of the anchor's segmenter.  Returns {subject: record}; Refused (stage admission) otherwise."""
    from bayesreg.seg.unet.infer import ModelSelectionRefused, check_subject_admitted, select_fold_models
    out = {}
    for subject in subjects:
        try:
            check_subject_admitted(subject, allowlist)
            models, policy_name = select_fold_models(registry, subject, allowlist, segmenter_id=hi.CONTRACT_SEGMENTER['segmenter_id'])
        except (SubjectSealed, ModelSelectionRefused) as exc:
            raise lc.Refused('admission', f'{subject}: {type(exc).__name__}: {exc}') from None
        if policy_name != 'all_folds_mean' or sorted(models) != sorted(registry.models):
            raise lc.Refused('admission', f'{subject}: policy {policy_name} with {sorted(models)}: every registered model (D24, all_folds_mean)')
        out[subject] = dict(policy=policy_name, models=sorted(models))
    return out


def inputs():
    """The registry and the grouping of HI (code-tree files: hi_core.CONTRACT_SEGMENTER's registry, the groups file)."""
    from bayesreg.io.pair import load_segmenter_registry
    from bayesreg.seg.unet.groups import DEFAULT_GROUPS_PATH, load_grouping
    return load_segmenter_registry(lc.repo_root()/REGISTRY_REL), load_grouping(DEFAULT_GROUPS_PATH, GROUPING), Path(DEFAULT_GROUPS_PATH)


def policy(pair_set, pairs, *, out=None):
    r = lc.roots(pair_set)
    from bayesreg.seg.unet.groups import DEFAULT_GROUPS_PATH
    reads = lc.code_reads()+lc.stage_paths(pair_set, pairs)+[lc.repo_root()/REGISTRY_REL, Path(DEFAULT_GROUPS_PATH), Path(hi.DEVCHECK_OUT)/hi.DEVCHECK_RECORD, Path(hi.DEVCHECK_LOCK)]
    return lc.Policy(name=f'lpba hi ({pair_set})', read_exact=reads, write_inside=[r.hi_out] if out is None else [out], write_exact=[r.locks/LOCK_NAME], read_inside=[(hi.MODELS_ROOT, ('.pt',))], list_inside=[r.stage],
                     results=hc.RESULTS)


def run(pair_set, *, code, device='cuda', dry_run=False, out=None, command=None, log=print):
    r = lc.roots(pair_set)
    out_root, lock = r.hi_out, r.locks/LOCK_NAME
    lc.guard_write(out_root, lock)
    unit = lc.unit_name(pair_set, UNIT_TOOL)
    problem = lc.formal_code_problem(code, pair_set, dry_run=dry_run)
    if problem:
        raise lc.Refused('code', problem)
    command = list(sys.argv if command is None else command)
    record = dict(schema=SCHEMA, tool=SCRIPT, pair_set=pair_set, started_utc=lc.utc(), stage=str(r.stage), out=str(out_root), pred_root=str(r.pred_root), lock=str(lock), dry_run=dry_run,
                  code=code, unit=unit, backend=hi.PRODUCTION_BACKEND, anchor=dict(hi.CONTRACT_SEGMENTER))
    record['preload'] = lc.preload()
    registry, grouping, groups_path = inputs()
    with lc.spying(policy(pair_set, lc.table(pair_set)[1], out=out if dry_run else None)) as spy:
        admission = lc.admit(pair_set)
        record['admission'] = admission.record
        record['registry'] = hi.check_registry(registry)
        record['contact'] = lc.check_contact(registry, admission.allowlist)
        subjects = list(dict.fromkeys(s for p in admission.pairs for s in p))
        if out_root.exists():
            raise lc.Refused('output', f'{out_root} exists: the predictions are written once, never overwritten')
        if lock.exists():
            raise lc.Refused('once', f'{lock} exists: HI of {pair_set} ran; it runs once')
        record['hold'] = lc.check_hold(pair_set)
        from bayesreg.seg.unet.groups import require_inference_grouping
        grouping_check = require_inference_grouping(grouping)
        record['selection'] = select_models(registry, subjects, admission.allowlist)
        if dry_run:                             # code-tree reads only: the decision's record, the members and the stage are checked by the run
            record['planned'] = dict(subjects=subjects, stage_files=[dict(path=str(p), exists=p.exists()) for p in lc.stage_paths(pair_set, admission.pairs)],
                                     not_in_the_dry_run='hi_core.check_decision (a file of the results root), hi_core.member_files (hashes the members under seg/unet35_v4_full), the stage')
            record['launch'] = launch_line(pair_set)
            record['spy'] = spy.record()
            return record
        try:
            record['decision'] = hi.check_decision(hi.DEVCHECK_DECISION)
        except hi.HIRefused as exc:
            raise lc.Refused('decision', str(exc)) from None
        available = hi.member_files(registry, hi.MODELS_ROOT)
        plans = {}
        for subject in subjects:
            try:
                plans[subject] = hi.plan_of(registry, subject, admission.allowlist, available)
            except (SubjectSealed, hi.hs_seg.HSSegFailed) as exc:
                raise lc.Refused('admission', f'{subject}: hs_seg.uf_plan refused ({type(exc).__name__}: {exc})') from None
        record['plans'] = {s: dict(policy=p[2], members=[m[1] for m in p[0]]) for s, p in plans.items()}
        record['stage_check'] = lc.check_stage(pair_set, admission.pairs)
        record['machine'] = hc.machine_gate(own_units=(unit,))
        record['disk'] = hc.disk_guard(need_d_gib=0, need_home_gib=NEED_HOME_GIB[pair_set])
        try:
            device = hi.prepare_device(device)
            environment = hi.check_environment(device, record['decision'])
        except hi.HIRefused as exc:
            raise lc.Refused('machine', str(exc)) from None
        line = hc.lock_line(tool=SCRIPT, pair_set=pair_set, out=str(out_root), stage_sha256sums=record['stage_check']['sha256sums_sha256'], prereg_sha256=lc.PREREG_SHA256,
                            registry_sha256=registry.sha256, devcheck_sha256=record['decision']['devcheck_sha256'], code_commit=(code or {}).get('commit'), event='first_run',
                            n_subjects=len(subjects))
        record['lock_line'] = hc.take_lock(lock, line)
        out_root.mkdir(parents=True)
        files = StageFiles(pair_set, r.stage)
        rows, done = {}, out_root/'subjects'
        done.mkdir()
        hc.write_json_once(out_root/hi.STARTED_RECORD, dict(record, environment=environment, utc=lc.utc()))
        try:
            for subject in subjects:
                array, meta, image_record = stage_arrays(subject, admission.pairs, r.stage, admission.allowlist)
                plan, models, policy_name = plans[subject]
                hi_block = dict(backend=hi.PRODUCTION_BACKEND, purpose=f'LPBA40 second test set ({pair_set}; contract C26.5-3)', pair_set=pair_set,
                                admission=record['admission']['subjects'][subject], image=image_record, anchor=dict(hi.CONTRACT_SEGMENTER),
                                devcheck=dict(json=record['decision']['devcheck_json'], sha256=record['decision']['devcheck_sha256']), contract='C26.5-3; C22.1-2 修订 4',
                                task='runs/20261119-lpba/任务书_A_数据准入HI与Atropos.md', temperature='as registered: not recalibrated on LPBA40 (prereg §2.4)')
                t0 = time.perf_counter()
                row = predict(subject, files=files, array=array, meta=meta, registry=registry, models=models, plan=plan, policy=policy_name, device=device, grouping=grouping,
                              grouping_check=grouping_check, pred_root=r.pred_root, code=code, environment=environment, command=command, allowlist=admission.allowlist, hi_block=hi_block)
                row = dict(row, provenance_sha256=hc.sha256_file(r.pred_root/subject/hi.hs_seg.PROVENANCE_FILE), image=image_record, seconds_wall=round(time.perf_counter()-t0, 3))
                if device.type == 'cuda':
                    import torch
                    row['gpu_peak_allocated_gib'] = torch.cuda.max_memory_allocated(device)/2**30
                hc.write_json_once(done/f'{subject}.json', row)
                rows[subject] = row
                log(f"{subject}: {row['seconds_total']:.1f} s; input_sha256 {meta['sha256'][:12]}")
        except BaseException as exc:
            record.update(stopped=f'{type(exc).__name__}: {exc}', done=sorted(rows), spy=spy.record(), ended_utc=lc.utc())
            spy.remove()
            hc.write_json_once(out_root/hi.STOPPED_RECORD, record)
            raise
        record['spy'] = spy.record(expect=[p for p in lc.stage_paths(pair_set, admission.pairs) if p.name == 'pair.npz'])
    record.update(rows=rows, environment=environment, ended_utc=lc.utc(), passed=bool(record['spy']['passed']) and len(rows) == len(subjects)
                  and all(r_['n_reader_not_own'] == 0 for r_ in rows.values()))
    hc.write_json_once(out_root/hi.INFER_RECORD, record)
    return record


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog=SCRIPT)
    ap.add_argument('--pair-set', required=True, choices=lc.PAIR_SETS)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args(argv)
    if a.dry_run and a.out is None:
        ap.error('--dry-run writes its record into --out DIR (a new or empty directory outside the results roots, e.g. mktemp -d)')
    return a


def main(argv=None, *, code=None, out=print):
    a = parse_args(argv)
    if a.dry_run:
        problem = lc.dry_run_out_problem(a.out)
        if problem:
            out(f'{problem}: a dry run writes into a new or empty directory outside the results root and the code trees')
            return 2
    elif not sys.flags.dont_write_bytecode:
        out('refused: run with python -B (or PYTHONDONTWRITEBYTECODE)')
        return 8
    try:
        code = lc.check_code(hp.code_identity() if code is None else code, a.pair_set, dry_run=a.dry_run)
        record = run(a.pair_set, code=code, device=a.device, dry_run=a.dry_run, out=a.out, command=[SCRIPT]+list(sys.argv[1:] if argv is None else argv))
    except hc.Refused as exc:
        out(f'refused ({exc.stage}): {exc}')
        if a.dry_run:
            a.out.mkdir(parents=True, exist_ok=True)
            hc.write_json_once(a.out/'dry_run.json', dict(schema=SCHEMA+'.dry_run', refused=dict(stage=exc.stage, message=str(exc)), utc=lc.utc(), spy=getattr(exc, 'hp_spy', None)))
        return 8
    except hp.Breach as exc:
        out(f'stopped by the spy: {exc}')
        return 8
    except (hi.HIRefused, hi.hs_seg.HSSegFailed, ProvenanceRejected, SubjectSealed, InputValidationError, SystemExit) as exc:
        out(f'stopped: {type(exc).__name__}: {exc}')
        return 8
    if a.dry_run:
        a.out.mkdir(parents=True, exist_ok=True)
        hc.write_json_once(a.out/'dry_run.json', dict(record, schema=SCHEMA+'.dry_run'))
        out(f"dry run passed ({a.pair_set}): {len(record['planned']['subjects'])} subjects planned (policy {sorted({p['policy'] for p in record['selection'].values()})}); "
            f"spy breaches {len(record['spy']['breaches'])}; launch: {record['launch']}")
        return 0
    out(f"done: {len(record['rows'])} subjects in {record['pred_root']}; passed {record['passed']}")
    return 0 if record['passed'] else 8


if __name__ == '__main__':
    sys.exit(main())
