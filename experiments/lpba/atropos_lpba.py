"""LP-A item 4 (runs/20261119-lpba/任务书_A_数据准入HI与Atropos.md): the tissue layer P_T (ANTs Atropos, three classes) of the LPBA40 subjects of one mode, read from the
stage prep_lpba.py wrote.  Contract C26.5-4 (L29), C26.4 L44; prereg §8.2 step 3, §7.2 (Atropos and intensity rows), §12 item 8.  In the method the tissue layer serves
the alignment QC only (C22.1-3).

The Atropos work is experiments/d0/atropos_seg.py's own, through experiments/holdout/atropos_holdout.py's wrapper, both imported and not copied (no line of either is
changed): atropos_seg.segment (the same commands, parameters and threads; the mask image > 0; the QC: every class >= 5 % of the mask, else SystemExit, the record kept; the
classes named by rank CSF < GM < WM; provenance bayesreg.tissue_provenance.v1 whose input_sha256 is the pair's manifest sources.<side>.sha256, i.e. the stage array's
sha256 of P7, and image_source.array_sha256_float64 the array's own sha256), wrapped by atropos_holdout.segment_subjects (atropos_seg's provenance check given the run's
allowlist for the length of the call, then validate_segmentation_provenance (layer tissue) for every pair the subject is in, verify_segmentation_files).  What changes:
  - the admission: lpba_common.admit (pins, the pinned LPBA allowlist, the mode's names, sealed lists, check_subject); every pair's manifest ids against its name and through
    check_pair_subjects with the LPBA allowlist (LpbaEntries.precheck, atropos_holdout.py:170's way: L29), before the lock; the arrays through load_pair with it, after;
  - the stage: lpba_common.check_stage (prep_record.json passed, of this mode, its SHA256SUMS; every pair file unchanged);
  - the roots: roots(pair_set).stage, the output roots(pair_set).atropos/<subject>/ (new), the lock <root>/locks/lpba_atropos.lock (O_EXCL, after every check), the unit
    tag-lp-atropos (tag-lp-feas-atropos), holdout_common's machine gate and disk guard, the hold file; no resume;
  - the subjects: the mode's subjects in the order of the pair table, fixed then moving (S01, S02 for the feasibility mode, each in both of its pairs: atropos_seg.segment
    checks that both arrays and both source sha256 agree);
  - after each subject: its provenance's input_sha256 and image_source.array_sha256_float64 checked equal to the stage array's sha256 (P7); the WM ratio (C26.5-4, prereg
    §5.3 and §7.2): the mean of the stage image over the Atropos WM posterior > .5, divided by W_dev of the committed c record -- recorded; in the feasibility mode a ratio
    outside [.8, 1.25] is the preregistration's stop (§7.2: always to BZ): the record says so and passed is false;
  - the spy (lpba_common.Policy) over the whole run: the code-tree reads of the admission, the stage files, the run's own directory and lock.
atropos_seg's provenance keeps two of its strings for OASIS (it is not changed): the input_sha256 key 'aligned_norm.nii.gz' and dataset 'OASIS'; this entry's record writes
the LPBA40 facts beside them (the stage array, the pair set), and the value of input_sha256 is the stage array's sha256 as P7 asks.

    python -B experiments/lpba/atropos_lpba.py --pair-set lpba40_feas --dry-run --out DIR
    python -B experiments/lpba/atropos_lpba.py --pair-set lpba40_feas       (unit tag-lp-feas-atropos; formal: lpba40_test, tag-lp-atropos)
Exit codes: 0 done (or a dry run that passed), 2 usage, 8 refused, stopped by the spy, a check failed or atropos_seg's STOP (QC; the record kept), 1 any other failure.
"""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
for _p in (HERE, EXPERIMENTS/'holdout', EXPERIMENTS/'d0'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import lpba_common as lc  # noqa: E402
import atropos_holdout as ah  # noqa: E402   (experiments/holdout/atropos_holdout.py, imported, not copied)
import atropos_seg  # noqa: E402            (experiments/d0/atropos_seg.py, imported, not copied)

hc, hp = lc.hc, lc.hp

from bayesreg.io.guard import SubjectSealed  # noqa: E402
from bayesreg.io.pair import ProvenanceRejected  # noqa: E402
from jd_curl.validation import InputValidationError  # noqa: E402

SCHEMA = 'bayesreg.lpba40.atropos.v1'
SCRIPT = 'experiments/lpba/atropos_lpba.py'
UNIT_TOOL = 'atropos'      # the unit name's last part (tag-lp-[feas-]atropos); not an authorization tool: holdout_common's TOOL names those (test_pinned_code); LP-C2 integration
LOCK_NAME = 'lpba_atropos.lock'
RECORD, STOPPED = 'atropos_record.json', 'atropos_stopped.json'
RUNTIME_S = 2*3600
NEED_HOME_GIB = dict(lpba40_test=3., lpba40_feas=1.)        # 〔估算〕t1, mask, t1_trunc, labels and three posteriors per subject
WM_LEVEL = .5
WM_BAND = (.8, 1.25)                                        # prereg §7.2: the feasibility stop (always to BZ)
STAGE_NAME = dict(lpba40_test='lpba40_test_v1', lpba40_feas='lpba40_feas_v1')       # provenance image_source.stage


def launch_line(pair_set, *, code=None, log=None):
    """The start string of this entry's unit for `pair_set` (hp_core.launch_line: MemoryMax 8G, swap 0, RuntimeMaxSec RUNTIME_S, CPUQuota 800%): the dry run prints it; the feasibility chain runs it
    with the chain's log and wraps only the command with the end probe (LP-C2 integration review minor 3: the feasibility unit is the formal one's)."""
    return hp.launch_line(unit=lc.unit_name(pair_set, UNIT_TOOL), script=SCRIPT, args=['--pair-set', pair_set], runtime_s=RUNTIME_S, cpu_quota='800%', code=code, log=log)


def say(text):
    print(text, flush=True)


array_sha256 = lc.array_sha256                 # review minor 4: he_core.array_sha256 itself


class LpbaEntries:
    """atropos_holdout.StageEntries for the LPBA stage, whose pairs are the admitted rows (not a selection.json): {subject: [(pair, side, PairInput)]}, each pair loaded by
    load_pair (fixed and moving only) with the LPBA allowlist just before its subjects are segmented, at most CACHE pairs held.  precheck() (before the lock): every manifest
    through the loader's whitelisting decoder, its ids those of the pair's name, both through check_pair_subjects with the LPBA allowlist (atropos_holdout.py:170's way)."""
    CACHE = 2

    def __init__(self, stage_root, pairs, allowlist):
        from bayesreg.io.pair import load_pair
        self.root, self.allowlist, self._load = Path(stage_root), allowlist, load_pair
        self.names = [lc.pair_name(f, m) for f, m in pairs]
        self.where = {}
        for (fixed, moving), name in zip(pairs, self.names):
            for side, subject in (('fixed', fixed), ('moving', moving)):
                self.where.setdefault(subject, []).append((name, side))
        self.ids = {name: pair for name, pair in zip(self.names, pairs)}
        self._cache = {}

    def subjects(self):
        return list(self.where)

    def precheck(self):
        from bayesreg.io.guard import check_pair_subjects
        from bayesreg.io.pair import _read_manifest
        records = {}
        for name in self.names:
            _, _, selected = _read_manifest(self.root/name/'manifest.json')
            if (selected['fixed_subject'], selected['moving_subject']) != self.ids[name]:
                raise lc.Refused('stage', f"{name}: the manifest names {selected['fixed_subject']}, {selected['moving_subject']}")
            guard = check_pair_subjects(selected['fixed_subject'], selected['moving_subject'], self.allowlist)
            records[name] = dict(lpba40_guard=guard.get('lpba40_guard'), source_sha256=selected['source_sha256'])
        return dict(pairs=len(self.names), checked="manifest ids = the pair's name; check_pair_subjects with the LPBA allowlist", records=records)

    def pair(self, name):
        if name not in self._cache:
            while len(self._cache) >= self.CACHE:
                self._cache.pop(next(iter(self._cache)))
            pair = self._load(self.root/name/'pair.npz', allowlist=self.allowlist)
            if (pair.fixed_subject, pair.moving_subject) != self.ids[name]:
                raise lc.Refused('stage', f'{name}: the manifest names {pair.fixed_subject}, {pair.moving_subject}')
            self._cache[name] = pair
        return self._cache[name]

    def __getitem__(self, subject):
        return [(name, side, self.pair(name)) for name, side in self.where[subject]]


def wm_ratio(subject, entries, out_root, W_dev):
    """C26.5-4 (prereg §5.3, §7.2): the mean of the stage image over the subject's Atropos WM posterior > .5, over W_dev; and the P7 binding of the provenance."""
    prov = json.loads((Path(out_root)/subject/'provenance.json').read_text(encoding='utf-8'))
    name, side, pair = entries[0]
    image = getattr(pair, side)
    digest = array_sha256(image)
    inputs = list((prov.get('input_sha256') or {}).values())
    bound = inputs == [digest] and prov.get('image_source', {}).get('array_sha256_float64') == digest and all(
        array_sha256(getattr(p, s)) == digest and p.source_sha256[s] == digest for _, s, p in entries)
    if not bound:
        raise lc.Refused('provenance', f'{subject}: the provenance input_sha256 {inputs} / array sha256 is not the stage array\'s {digest} (P7)')
    wm_file = prov['posterior_files'][prov['wm_index']]
    wm = atropos_seg.load_zyx(Path(out_root)/subject/wm_file).astype(np.float64)
    if wm.shape != image.shape:
        raise lc.Refused('provenance', f'{subject}: the WM posterior has the shape {wm.shape}, the image {image.shape}')
    mask = wm > WM_LEVEL
    mean = float(image[mask].mean()) if mask.any() else float('nan')
    ratio = mean/W_dev
    return dict(input_sha256=digest, input_bound=True, wm_file=wm_file, wm_voxels=int(mask.sum()), wm_mean=lc.number(mean), W_dev=lc.number(W_dev), wm_ratio=lc.number(ratio),
                in_band=bool(WM_BAND[0] <= ratio <= WM_BAND[1]), band=list(WM_BAND), class_names=prov.get('class_names'),
                ascending_by_file=(prov.get('checks') or {}).get('ascending_by_file'), qc=(prov.get('checks') or {}).get('qc_min_class_fraction_ge_0.05'))


def policy(pair_set, pairs, *, out=None):
    r = lc.roots(pair_set)
    return lc.Policy(name=f'lpba atropos ({pair_set})', read_exact=lc.code_reads()+lc.stage_paths(pair_set, pairs), write_inside=[r.atropos] if out is None else [out],
                     write_exact=[r.locks/LOCK_NAME], list_inside=[r.stage], results=hc.RESULTS)


def run(pair_set, *, code, dry_run=False, out=None, log=say):
    r = lc.roots(pair_set)
    out_root, lock = r.atropos, r.locks/LOCK_NAME
    lc.guard_write(out_root, lock)
    unit = lc.unit_name(pair_set, UNIT_TOOL)
    problem = lc.formal_code_problem(code, pair_set, dry_run=dry_run)
    if problem:
        raise lc.Refused('code', problem)
    record = dict(schema=SCHEMA, tool=SCRIPT, pair_set=pair_set, started_utc=lc.utc(), stage=str(r.stage), out=str(out_root), lock=str(lock), threads=ah.THREADS, dry_run=dry_run,
                  code=code, unit=unit, atropos=dict(bin=str(atropos_seg.ANTS), version=atropos_seg.ANTS_VERSION, parameters=atropos_seg.PARAMS))
    record['preload'] = lc.preload()
    with lc.spying(policy(pair_set, lc.table(pair_set)[1], out=out if dry_run else None)) as spy:
        admission = lc.admit(pair_set)
        record['admission'] = admission.record
        c = lc.load_c_record()
        record['c_record'] = c.record
        if out_root.exists():
            raise lc.Refused('output', f'{out_root} exists: the tissue layer is written once, never overwritten')
        if lock.exists():
            raise lc.Refused('once', f'{lock} exists: the Atropos entry of {pair_set} ran; it runs once')
        record['hold'] = lc.check_hold(pair_set)
        subjects = list(dict.fromkeys(s for p in admission.pairs for s in p))
        if dry_run:
            record['planned'] = dict(subjects=subjects, stage_files=[dict(path=str(p), exists=p.exists()) for p in lc.stage_paths(pair_set, admission.pairs)])
            record['launch'] = launch_line(pair_set)
            record['spy'] = spy.record()
            return record
        record['stage_check'] = lc.check_stage(pair_set, admission.pairs)
        found = LpbaEntries(r.stage, admission.pairs, admission.allowlist)
        record['stage_precheck'] = found.precheck()
        record['shared_subjects'] = sorted(s for s, e in found.where.items() if len(e) > 1)
        record['machine'] = hc.machine_gate(own_units=(unit,))
        record['disk'] = hc.disk_guard(need_d_gib=0, need_home_gib=NEED_HOME_GIB[pair_set])
        line = hc.lock_line(tool=SCRIPT, pair_set=pair_set, out=str(out_root), stage_sha256sums=record['stage_check']['sha256sums_sha256'], prereg_sha256=lc.PREREG_SHA256,
                            c_record_sha256=c.sha256, code_commit=(code or {}).get('commit'), event='first_run')
        record['lock_line'] = hc.take_lock(lock, line)
        out_root.mkdir(parents=True)
        try:
            record['subjects'] = ah.segment_subjects(subjects, found, out_root, allowlist=admission.allowlist, stage_name=STAGE_NAME[pair_set], log=log)
            record['wm'] = {}
            for subject in subjects:
                record['wm'][subject] = wm_ratio(subject, found[subject], out_root, c.W_dev)
                log(f"{subject}: WM ratio {record['wm'][subject]['wm_ratio']['decimal']}")
        except BaseException as exc:
            record.update(stopped=f'{type(exc).__name__}: {exc}', spy=spy.record(), ended_utc=lc.utc())
            spy.remove()
            hc.write_json_once(out_root/STOPPED, record)
            raise
        record['spy'] = spy.record(expect=[p for p in lc.stage_paths(pair_set, admission.pairs) if p.name == 'pair.npz'])
    outside = sorted(s for s, w in record['wm'].items() if not w['in_band'])
    record['feasibility_stop'] = dict(rule='prereg §7.2: in the feasibility mode a WM ratio outside [.8, 1.25] stops, always to BZ', subjects=outside,
                                      applies=pair_set == 'lpba40_feas', stop=pair_set == 'lpba40_feas' and bool(outside))
    record.update(ended_utc=lc.utc(), passed=bool(record['spy']['passed']) and not record['feasibility_stop']['stop'])
    hc.write_json_once(out_root/RECORD, record)
    return record


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog=SCRIPT)
    ap.add_argument('--pair-set', required=True, choices=lc.PAIR_SETS)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args(argv)
    if a.dry_run and a.out is None:
        ap.error('--dry-run writes its record into --out DIR (a new or empty directory outside the results roots, e.g. mktemp -d)')
    return a


def main(argv=None, *, code=None, out=say):
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
        record = run(a.pair_set, code=code, dry_run=a.dry_run, out=a.out)
    except hc.Refused as exc:
        out(f'refused ({exc.stage}): {exc}')
        if a.dry_run:
            a.out.mkdir(parents=True, exist_ok=True)
            hc.write_json_once(a.out/'dry_run.json', dict(schema=SCHEMA+'.dry_run', refused=dict(stage=exc.stage, message=str(exc)), utc=lc.utc(), spy=getattr(exc, 'hp_spy', None)))
        return 8
    except hp.Breach as exc:
        out(f'stopped by the spy: {exc}')
        return 8
    except SystemExit as exc:                   # atropos_seg's STOPs (QC, an exit of ANTs, the image differing between pairs): the record is kept
        out(f'stopped: {exc}')
        return 8
    except (ProvenanceRejected, SubjectSealed, InputValidationError) as exc:
        out(f'stopped: {type(exc).__name__}: {exc}')
        return 8
    if a.dry_run:
        a.out.mkdir(parents=True, exist_ok=True)
        hc.write_json_once(a.out/'dry_run.json', dict(record, schema=SCHEMA+'.dry_run'))
        out(f"dry run passed ({a.pair_set}): {len(record['planned']['subjects'])} subjects; spy breaches {len(record['spy']['breaches'])}; launch: {record['launch']}")
        return 0
    out(f"done: {len(record['subjects'])} subjects in {record['out']}; spy passed {record['spy']['passed']}; feasibility stop {record['feasibility_stop']['stop']}")
    return 0 if record['passed'] else 8


if __name__ == '__main__':
    sys.exit(main())
