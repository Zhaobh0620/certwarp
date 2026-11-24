"""LP-D (b) (runs/20261119-lpba/任务书_D_开发集逐位对照.md; prereg §12 item 4 (b), §8.1 item 9; contract C26.5-11 (b), C26.4 L39, C26.5-12 N56): HI's path from a stage array, on the 20
development subjects of data/three_d/stage_L2R_val_v1, bitwise against the registered predictions seg/unet35_v4_ens/pred (probs36, structure_groups).  20/20 the same -> passed.  GPU (about
20 minutes; this package does not run it).

The path is LP-A's HI LPBA mode with an OASIS development stage under it:
  the array   each subject's fixed or moving array of the stage pair.npz files it is in, through bayesreg.io.pair.load_pair with the development allowlist (fixed, moving and the all-zero
              initial_u load_pair checks; no seg key: the spy refuses any other member); one array when the subject is in two pairs (else the check stops).  dev_stage_array builds the
              metadata hi_lpba.stage_arrays builds, key for key (a test compares them on an LPBA stage), but it does not ask the array's sha256 to be the manifest's sources.<side>.sha256: the
              OASIS stages record there the sha256 of the aligned_norm.nii.gz file (paper 1's stage writer; it is the registered predictions' input_sha256), not of the array (LPBA40 prereg
              §2.2 P7 is the LPBA stage's convention).  Both sha256 are recorded; the manifest's is compared with the registered prediction's input_sha256 (recorded, not judged).
  the model   hi_lpba.predict: hi_lpba.stage_reader in place of bayesreg.seg.unet.data.read_image for the subject's call, hi_core.predict_hs_a (hs_seg.uf4_probabilities with the registered
              members, their own normalize_image, the registered temperatures; hs_seg.write_prediction), into <b>/pred/<s>/; the reader asked exactly once.
  the reading the written probs36.npz and structure_groups.npz (their sha256 against the written provenance, bayesreg.io.pair.verify_segmentation_files) against the registered ones
              (hi_core.reference_record before any image -- this segmenter, this registry, these members in order, all_folds_mean, test_labels_seen false --, hi_core.read_reference
              re-hashing them), by hi_core.compare: bitwise (bit patterns), the elements that differ, the argmax agreement.
The admission before any file of a subject: the 19 development pairs through holdout_common/hp_core.admit_dev (on no sealed list, check_subject on the development allowlist, its source
stage_L2R_val_v1, no never_for_training source), the pins (lpba_common.check_pins, recorded: the preregistration, the lists, the frozen files, the method's settings_sha256, the c record;
review minor 3), the registry against HI's anchor, every label contact subject on the allowlist and in no never_for_training source, then D24 per subject
(hi_lpba.select_models: infer.check_subject_admitted, select_fold_models: all_folds_mean); then HI's decision of its own development check (hi_core.check_decision: hs_a), the members re-hashed,
the plans, the registered predictions' provenance; the hold; the machine gate and the disks; the device and its environment (cuda, the decision's).  No truth label anywhere.

    python -B experiments/lpba/devcheck/lpdev_b.py dry-run --out DIR      no image, no GPU, no member: the admission, the registry, D24; the planned files stat'ed
    python -B experiments/lpba/devcheck/lpdev_b.py run [--device cuda]    once (unit tag-lp-devb): <RESULTS>/lpba40_devcheck_v1/b/devcheck_b.json, pred/<s>/, subjects/<s>.json
Exit codes: lpdev_common.EXIT.
"""
import argparse
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import lpdev_common as dc  # noqa: E402

lc, hc, hp = dc.lc, dc.hc, dc.hp
import hi_lpba  # noqa: E402

hi = hi_lpba.hi
hs_seg = hi.hs_seg

SCHEMA = 'bayesreg.lpba40.devcheck_b.v1'
SCRIPT = 'experiments/lpba/devcheck/lpdev_b.py'
UNIT = 'tag-lp-devb'
DEV_STAGE = hc.DEV_STAGE                                   # <main checkout>/data/three_d/stage_L2R_val_v1 (the allowlist source stage_L2R_val_v1)
DEV_PAIRS = tuple(tuple(name.split('__')) for name in hc.DEV_PAIRS)       # the 19 development pairs, (fixed, moving)
DEV_SUBJECTS = hi.DEV_SUBJECTS                             # their 20 subjects, OASIS_OAS1_0438_MR1 ... 0457
REFERENCE_ROOT = hi.REGISTERED_PRED                        # seg/unet35_v4_ens/pred: HS's seg phase, registered; only read
READER = 'experiments/lpba/devcheck/lpdev_b.dev_stage_array: the stage array (bayesreg.io.pair.load_pair with the development allowlist), given to hi_lpba.stage_reader'
RECORD, STOPPED = 'devcheck_b.json', 'devcheck_b_stopped.json'
NEED_HOME_GIB = 10.                                        # 〔估算〕20 subjects x about 0.38 GiB (run_hi.MIB_PER_SUBJECT 384), with room
RUNTIME_S = 2*3600
READ_PASSED, READ_NOT = '逐位相同 {n}/{n}', '不过'


def dev_stage_array(subject, pairs, stage, allowlist, *, name_of=None):
    """The subject's stage array (module docstring, "the array").  name_of: the pair directory's name (default holdout_common.pair_name, <fixed>__<moving>; a test passes
    lpba_common.pair_name to compare with hi_lpba.stage_arrays on an LPBA stage).  Returns (array, metadata with hi_lpba.stage_arrays' keys, record)."""
    from bayesreg.io.pair import load_pair
    name_of = hc.pair_name if name_of is None else name_of
    found = []
    for fixed, moving in pairs:
        for side, s in (('fixed', fixed), ('moving', moving)):
            if s != subject:
                continue
            name = name_of(fixed, moving)
            pair = load_pair(Path(stage)/name/'pair.npz', allowlist=allowlist)
            if (pair.fixed_subject, pair.moving_subject) != (fixed, moving):
                raise dc.Stopped(f'{name}: the manifest names {pair.fixed_subject}, {pair.moving_subject}')
            array = getattr(pair, side)
            found.append((name, side, array, lc.array_sha256(array), pair))
    if not found:
        raise dc.Stopped(f'{subject} is in none of the pairs')
    if len({f[3] for f in found}) != 1:
        raise dc.Stopped(f'{subject}: two different arrays in its pairs {[(f[0], f[1], f[3]) for f in found]}')
    if len({f[4].source_sha256[f[1]] for f in found}) != 1:
        raise dc.Stopped(f'{subject}: its pairs\' manifests name two source files {[(f[0], f[1], f[4].source_sha256[f[1]]) for f in found]}')
    name, side, array, digest, pair = found[0]
    meta = dict(source=f'{Path(stage)/name/"pair.npz"}:{side}', sha256=digest, shape_zyx=list(pair.grid.shape), spacing_xyz=[float(s) for s in pair.grid.spacing], unit=pair.grid.unit,
                labels=False, reader=READER, subject=subject, grid_shape_zyx=list(pair.grid.shape), origin_xyz=[float(a[0]) for a in pair.grid.axes])
    record = dict(subject=subject, input_sha256=digest, manifest_source_sha256=pair.source_sha256[side],
                  pairs=[dict(pair=f[0], side=f[1], sha256=f[3], pair_sha256=f[4].provenance['pair_sha256'], manifest_sha256=f[4].provenance['manifest_sha256'],
                              keys_read=list(f[4].provenance['keys_read'])) for f in found])
    return array, meta, record


def compare_subject(subject, pred_root, reference):
    """The written prediction (its files against its own provenance) against the registered one (hi_core.read_reference), by hi_core.compare."""
    import numpy as np
    from bayesreg.io.pair import verify_segmentation_files
    directory = Path(pred_root)/subject
    provenance = dc.read_json(directory/hs_seg.PROVENANCE_FILE)
    verify_segmentation_files(provenance, directory)
    mine = []
    for name in (hs_seg.PROBS_FILE, hs_seg.GROUPS_FILE):
        with np.load(directory/name, allow_pickle=False) as npz:
            mine.append(npz['p'])
    theirs = hi.read_reference(REFERENCE_ROOT, subject)
    written, registered = dict(provenance['output_sha256']), dict(reference['output_sha256'])
    row = dict(probs36=hi.compare(mine[0], theirs[0]), structure_groups=hi.compare(mine[1], theirs[1]), output_sha256=written, reference_output_sha256=registered,
               files_equal={k: written.get(k) == registered.get(k) for k in registered}, input_sha256=dict(provenance['input_sha256']), reference_image_sha256=reference['image_sha256'])
    del mine, theirs
    row['bitwise_equal'] = bool(row['probs36']['bitwise_equal'] and row['structure_groups']['bitwise_equal'])
    return row


def reading(rows, subjects):
    """passed: every subject done, probs36 and structure_groups bitwise equal (the preregistration's 20/20)."""
    done = [s for s in subjects if s in rows]
    equal = [s for s in done if rows[s]['comparison']['bitwise_equal']]
    passed = len(done) == len(subjects) and len(equal) == len(subjects)
    return dict(text=READ_PASSED.format(n=len(subjects)) if passed else READ_NOT, passed=passed, n_subjects=len(subjects), n_done=len(done), n_bitwise_equal=len(equal),
                not_equal=[s for s in done if s not in equal],
                min_probs36_argmax_agreement=min((rows[s]['comparison']['probs36'].get('argmax_agreement', 0.) for s in done), default=None),
                rule='prereg §12 item 4 (b): probs36 and structure_groups bitwise equal to seg/unet35_v4_ens/pred on every one of the 20 subjects')


def stage_paths(pairs):
    return [Path(DEV_STAGE)/hc.pair_name(f, m)/leaf for f, m in pairs for leaf in ('pair.npz', 'manifest.json')]


def reference_paths(subjects):
    return [Path(REFERENCE_ROOT)/s/leaf for s in subjects for leaf in (hs_seg.PROVENANCE_FILE, hs_seg.PROBS_FILE, hs_seg.GROUPS_FILE)]


def code_tree_reads():
    """The code-tree files of the check: lpba_common.code_reads (the pins lpba_common.check_pins reads -- the preregistration, the lists, the frozen files, the c record -- and the sealed
    lists), the development allowlist, the registry, the groups file."""
    from bayesreg.seg.unet.groups import DEFAULT_GROUPS_PATH
    tree = lc.repo_root()
    return lc.code_reads()+[tree/hp.ALLOWLIST_REL, tree/hi_lpba.REGISTRY_REL, Path(DEFAULT_GROUPS_PATH)]


def policy(subjects, pairs, *, out=None):
    r = dc.roots()
    reads = code_tree_reads()+stage_paths(pairs)+reference_paths(subjects)+[Path(hi.DEVCHECK_OUT)/hi.DEVCHECK_RECORD, Path(hi.DEVCHECK_LOCK)]
    members = {Path(REFERENCE_ROOT)/s/leaf: ('p',) for s in subjects for leaf in (hs_seg.PROBS_FILE, hs_seg.GROUPS_FILE)}
    return dc.Policy(name='lpba devcheck (b)', read_exact=reads, write_inside=[r.b if out is None else out], write_exact=[r.b_lock], read_inside=[(hi.MODELS_ROOT, ('.pt',))],
                     npz_members=members, results=hc.RESULTS)


def run(*, code, device='cuda', dry_run=False, out=None, command=None, log=print):
    """The order of the module docstring.  Returns the record."""
    r = dc.roots()
    out_root, lock = r.b, r.b_lock
    lc.guard_write(out_root, lock)
    command = list(sys.argv if command is None else command)
    subjects, pairs = list(DEV_SUBJECTS), [tuple(p) for p in DEV_PAIRS]
    record = dict(schema=SCHEMA, tool=SCRIPT, task=dc.TASK, contract=dc.CONTRACT, started_utc=dc.utc(), stage=str(DEV_STAGE), reference_root=str(REFERENCE_ROOT), out=str(out_root),
                  pred_root=str(out_root/'pred'), lock=str(lock), dry_run=dry_run, code=code, unit=UNIT, backend=hi.PRODUCTION_BACKEND, anchor=dict(hi.CONTRACT_SEGMENTER))
    record['preload'] = lc.preload()
    registry, grouping, _ = hi_lpba.inputs()
    with dc.spying(policy(subjects, pairs, out=out if dry_run else None)) as spy:
        allowlist, admission = hp.admit_dev(pairs=pairs)
        if sorted(admission['subjects']) != sorted(subjects):
            raise lc.Refused('admission', f"the 19 development pairs hold {len(admission['subjects'])} subjects, not the 20 of DEV_SUBJECTS")
        record['admission'] = admission
        record['pins'] = lc.check_pins()                       # review minor 3: the preregistration, the lists, the frozen files, the method's settings_sha256, the c record
        try:
            record['registry'] = hi.check_registry(registry)
        except hi.HIRefused as exc:
            raise lc.Refused('registry', str(exc)) from None
        record['contact'] = lc.check_contact(registry, allowlist)
        record['hold'] = lc.check_hold('lpba40_feas')
        if out_root.exists():
            raise lc.Refused('output', f'{out_root} exists: the check runs once')
        if lock.exists():
            raise lc.Refused('once', f'{lock} exists: the check ran; it runs once')
        from bayesreg.seg.unet.groups import require_inference_grouping
        grouping_check = require_inference_grouping(grouping)
        record['selection'] = hi_lpba.select_models(registry, subjects, allowlist)
        if dry_run:
            record['planned'] = dict(stage=dc.planned(stage_paths(pairs)), reference=dc.planned(reference_paths(subjects)),
                                     not_in_the_dry_run='hi_core.check_decision, the members (hashed by the run), the registered predictions\' provenance, the device')
            record['launch'] = hp.launch_line(unit=UNIT, script=SCRIPT, args=['run'], runtime_s=RUNTIME_S, memory_max='12G')
            record['spy'] = spy.record()
            return record
        try:
            record['decision'] = hi.check_decision(hi.DEVCHECK_DECISION)
            available = hi.member_files(registry, hi.MODELS_ROOT)
            plans = {s: hi.plan_of(registry, s, allowlist, available) for s in subjects}
            references = {s: hi.reference_record(REFERENCE_ROOT, s, registry=registry, plan=plans[s][0]) for s in subjects}
        except (hi.HIRefused, hs_seg.HSSegFailed) as exc:
            raise lc.Refused(getattr(exc, 'stage', 'plan'), str(exc)) from None
        record['plans'] = {s: dict(policy=p[2], members=[m[1] for m in p[0]]) for s, p in plans.items()}
        record['references'] = references
        record['machine'] = hc.machine_gate(own_units=(UNIT,))
        record['disk'] = hc.disk_guard(need_d_gib=0, need_home_gib=NEED_HOME_GIB)
        try:
            device = hi.prepare_device(device)
            environment = hi.check_environment(device, record['decision'])
        except hi.HIRefused as exc:
            raise lc.Refused('machine', str(exc)) from None
        line = hc.lock_line(tool=SCRIPT, out=str(out_root), stage=str(DEV_STAGE), registry_sha256=registry.sha256, devcheck_sha256=record['decision']['devcheck_sha256'],
                            allowlist_sha256=admission['allowlist_sha256'], code_commit=(code or {}).get('commit'), event='first_run', n_subjects=len(subjects))
        record['lock_line'] = hc.take_lock(lock, line)
        rows = {}
        record['done'] = []
        with dc.after_lock(record, spy, out_root, lock, STOPPED):      # review minor 5: everything after the lock, the record's write included
            out_root.mkdir(parents=True)
            (out_root/'subjects').mkdir()
            files = hi_lpba.StageFiles('dev', DEV_STAGE)
            for subject in subjects:
                t0 = time.perf_counter()
                array, meta, image_record = dev_stage_array(subject, pairs, DEV_STAGE, allowlist)
                plan, models, policy_name = plans[subject]
                hi_block = dict(backend=hi.PRODUCTION_BACKEND, purpose='LPBA40 prereg §12 item 4 (b): HI from a stage array, development check (not a prediction of the method)',
                                pair_set='dev', admission=admission['subjects'][subject], image=image_record, anchor=dict(hi.CONTRACT_SEGMENTER),
                                devcheck=dict(json=record['decision']['devcheck_json'], sha256=record['decision']['devcheck_sha256']), contract=dc.CONTRACT, task=dc.TASK,
                                temperature='as registered')
                row = hi_lpba.predict(subject, files=files, array=array, meta=meta, registry=registry, models=models, plan=plan, policy=policy_name, device=device, grouping=grouping,
                                      grouping_check=grouping_check, pred_root=out_root/'pred', code=code, environment=environment, command=command, allowlist=allowlist, hi_block=hi_block)
                del array
                comparison = compare_subject(subject, out_root/'pred', references[subject])
                rows[subject] = dict(subject=subject, image=image_record, prediction=row, comparison=comparison, seconds_wall=round(time.perf_counter()-t0, 3),
                                     manifest_source_is_reference_image=image_record['manifest_source_sha256'] == references[subject]['image_sha256'])
                hc.write_json_once(out_root/'subjects'/f'{subject}.json', rows[subject])
                record['done'] = sorted(rows)
                p = comparison['probs36']
                log(f"{subject}: probs36 bitwise {p['bitwise_equal']} ({p.get('n_elements_differ')} elements differ, argmax agreement {p.get('argmax_agreement')}); "
                    f"groups bitwise {comparison['structure_groups']['bitwise_equal']}; {rows[subject]['seconds_wall']:.1f} s")
            record['spy'] = spy.record(expect=[p for p in stage_paths(pairs) if p.name == 'pair.npz']+reference_paths(subjects))
            if device.type == 'cuda':
                import torch
                record['gpu_peak_allocated_gib'] = torch.cuda.max_memory_allocated(device)/2**30
            record.update(rows=rows, environment=environment, reading=reading(rows, subjects), ended_utc=dc.utc())
            record['passed'] = bool(record['spy']['passed']) and record['reading']['passed']
            record['conclusion'] = 'passed' if record['passed'] else 'not passed'
            record['files_read'] = files_read(rows, references, subjects)
            record['pred_files'] = {s: dict(rows[s]['comparison']['output_sha256']) for s in subjects}      # review minor 10: pred/<s>/ deleted after the review by these sha256
            hc.write_json_once(out_root/RECORD, record)
    return record


def files_read(rows, references, subjects):
    """The subjects' files the check read, with their sha256: each stage pair's pair.npz and manifest.json (load_pair's own hashes), each registered prediction's provenance.json and its two
    outputs (hashed by hi_core.reference_record and read_reference).  The code-tree files are in the admission's and the registry's records; the members in the plans."""
    stage = {}
    for row in rows.values():
        for x in row['image']['pairs']:
            stage[x['pair']] = x
    out = []
    for name, x in sorted(stage.items()):
        out += [dict(path=str(Path(DEV_STAGE)/name/'pair.npz'), sha256=x['pair_sha256']), dict(path=str(Path(DEV_STAGE)/name/'manifest.json'), sha256=x['manifest_sha256'])]
    for s in subjects:
        out.append(dict(path=str(Path(references[s]['directory'])/hs_seg.PROVENANCE_FILE), sha256=references[s]['provenance_sha256']))
        out += [dict(path=str(Path(references[s]['directory'])/k), sha256=v) for k, v in sorted(references[s]['output_sha256'].items())]
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog=SCRIPT, description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    d = sub.add_parser('dry-run')
    d.add_argument('--out', type=Path, required=True, help='a new or empty directory outside the results root and the code trees (e.g. mktemp -d)')
    r = sub.add_parser('run')
    r.add_argument('--device', default='cuda')
    return ap.parse_args(argv)


def main(argv=None, *, code=None, out=print):
    a = parse_args(argv)
    dry = a.command == 'dry-run'
    if dry:
        problem = lc.dry_run_out_problem(a.out)
        if problem:
            out(problem)
            return dc.EXIT['usage']
    else:
        problem = dc.bytecode_problem()
        if problem:
            out(f'refused: {problem}')
            return dc.EXIT['refused']
    command = [SCRIPT]+list(sys.argv[1:] if argv is None else argv)
    record, exit_code = dc.guarded(lambda: run(code=dc.check_code(hp.code_identity() if code is None else code, dry_run=dry), device='cpu' if dry else a.device, dry_run=dry,
                                               out=a.out if dry else None, command=command, log=out), dry=dry, out_dir=a.out if dry else None, schema=SCHEMA, out=out)
    if exit_code is not None:
        return exit_code
    if dry:
        dc.write_dry_run(a.out, dict(record, schema=SCHEMA+'.dry_run'))
        stage, ref = record['planned']['stage'], record['planned']['reference']
        out(f"dry run passed: {len(record['admission']['subjects'])} subjects admitted (policy {sorted({p['policy'] for p in record['selection'].values()})}); stage files there "
            f"{sum(p['exists'] for p in stage)}/{len(stage)}; registered prediction files there {sum(p['exists'] for p in ref)}/{len(ref)}; spy breaches {len(record['spy']['breaches'])}; "
            f"launch: {record['launch']}")
        return dc.EXIT['passed']
    out(f"done: {record['conclusion']} ({record['reading']['text']}; {record['out']}/{RECORD})")
    return dc.EXIT['passed'] if record['passed'] else dc.EXIT['not_passed']


if __name__ == '__main__':
    sys.exit(main())
