"""LP-C2 item 4 (contract C26.9-3 "第 10a 项：P_L 之前的纯准入干跑"; C26.1-4; review B1; C26.11-8): the 34 test ids and the 17 test pairs (in row order and direction), and the
two feasibility pairs, through every entry's admission function -- only ids and code-tree files, before P_L, on the same commit's clean detached checkout as the feasibility pairs.

What is called (each pair or subject on its own; the first problem is recorded, the rest go on):
  1. the pins and names of every LPBA entry: experiments/lpba/lpba_common.admit('lpba40_test') and admit('lpba40_feas') (the preregistration, the lists, the frozen files, the c
     record, the pinned LPBA allowlist, the mode's names, the sealed lists, check_subject per subject; C26.4 L27-L34);
  2. per test subject (34): bayesreg.io.guard.check_subject on the pinned LPBA allowlist (admitted, admitted_by_lpba40 true), bayesreg.seg.unet.infer.check_subject_admitted
     (L3), bayesreg.seg.unet.infer.select_fold_models with the registry of HI (hi_core.CONTRACT_SEGMENTER's; all_folds_mean, L48);
  3. per test pair (17): bayesreg.io.guard.check_pair_subjects (its record holds lpba40_guard, both sides admitted_by_lpba40), bayesreg.io.guard.check_lpba40_names on the pair
     directory's name with the LPBA allowlist (the one name-only function load_pair and truth.check_name share, C26.3-6);
  4. per pair of both modes: the method process's LPBA branch steps 1-3, the one function C26.6-2 asks LP-B to write (lpba40_pre_manifest(pair_set, pair); looked up in
     experiments/he/he_holdout.py, then experiments/he/run_hx.py): pair_set 'lpba40_test' for the 17, 'lpba40_feas' for the two -- it opens no manifest; while it is not in the
     code tree, the dry run records that and does not pass (fail closed);
  5. experiments/d0/make_inputs.a4_admission(the 34, the LPBA allowlist's path, make_inputs.UNetSource()) (L42: the T88 child's admission, the method's UNet source default);
  6. the contrasts (C26.9-3 对照): every one of the 34 refused by check_subject on the development allowlist (step "sealed", item "lpba40 test set") and by check_lpba40_names
     without an allowlist; S01, S02 and S37 refused by the formal entry's names (lpba_common.admit_subjects('lpba40_test', ...)); S03 and S37 refused by the feasibility entry.
The spy (lpba_common.Policy) allows only code-tree files: the lists, the allowlists, the preregistration, the frozen files, the c record, the sealed lists, the registry, the
folds file; no stage, no raw data, no prediction, nothing under the results root (C26.9-3).  The record goes to --out (new; e.g. runs/20261119-lpba/admission_dry_run.json),
written after the spy is removed; the lead commits it and puts its sha256 into the decision log; P_L is fixed only when it passed (C26.9-3).

    python -B experiments/lpba/admission_dry_run.py --out runs/20261119-lpba/admission_dry_run.json
Exit codes: 0 passed, 8 a problem (recorded), 2 usage.
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
for _p in (HERE, EXPERIMENTS/'he', EXPERIMENTS/'hi', EXPERIMENTS/'d0'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import lpba_common as lc  # noqa: E402

hc, hp = lc.hc, lc.hp
SCHEMA = 'bayesreg.lpba40.admission_dry_run.v1'
SCRIPT = 'experiments/lpba/admission_dry_run.py'
PRE_MANIFEST = 'lpba40_pre_manifest'
PRE_MANIFEST_HOMES = ('he_holdout', 'run_hx')          # C26.6-2: LP-B names the module; the first that has the function counts
FORMAL_REFUSED = ('S01', 'S02', 'S37')
FEAS_REFUSED = ('S03', 'S37')


def pre_manifest_function():
    """(function or None, where): LP-B's lpba40_pre_manifest (C26.6-2 steps 1-3), looked up in PRE_MANIFEST_HOMES in order."""
    import importlib
    for name in PRE_MANIFEST_HOMES:
        try:
            module = importlib.import_module(name)
        except Exception as exc:                        # noqa: BLE001 - a module that does not import is recorded, not raised
            return None, f'{name}: {type(exc).__name__}: {exc}'
        function = getattr(module, PRE_MANIFEST, None)
        if callable(function):
            return function, f'{name}.{PRE_MANIFEST}'
    return None, f'{PRE_MANIFEST} is in none of {PRE_MANIFEST_HOMES} (LP-B, C26.6-2): not in this code tree'


def registry_path():
    import hi_core
    return lc.repo_root()/hi_core.CONTRACT_SEGMENTER['registry']


def folds_path():
    import make_inputs
    return Path(make_inputs.UNetSource().folds)


def code_tree_reads():
    """The files the dry run may open: lpba_common.code_reads() (lists, LPBA allowlist, preregistration, frozen files, c record, sealed lists), the development allowlist (the
    contrast), the registry and the folds file."""
    from bayesreg.io.unseal import ALLOWLIST_REL
    return lc.code_reads()+[lc.repo_root()/ALLOWLIST_REL, registry_path(), folds_path()]


def _try(problems, label, call):
    try:
        return call()
    except BaseException as exc:                         # noqa: BLE001 - every problem is recorded and the dry run goes on
        if isinstance(exc, (KeyboardInterrupt, hp.Breach)):
            raise
        problems.append(f'{label}: {type(exc).__name__}: {exc}'[:600])
        return None


def check(*, pre_manifest=None, log=print):
    """The steps of the module docstring.  `pre_manifest`: (function or None, where) of pre_manifest_function(), looked up before the spy (its modules' imports read files
    of the code tree that are not on the spy's list).  Returns the record (passed: no problem at all)."""
    from bayesreg.io import guard
    from bayesreg.io.unseal import ALLOWLIST_REL
    from bayesreg.io.pair import load_segmenter_registry
    from bayesreg.seg.unet import infer
    import make_inputs
    problems, record = [], dict(schema=SCHEMA, tool=SCRIPT, started_utc=lc.utc())
    record['admit'] = {}
    for pair_set in lc.PAIR_SETS:
        admitted = _try(problems, f'lpba_common.admit({pair_set})', lambda ps=pair_set: lc.admit(ps))
        record['admit'][pair_set] = None if admitted is None else dict(pairs=[lc.pair_name(f, m) for f, m in admitted.pairs], subjects=admitted.record['subjects'])
    lists = lc.read_lists()
    allowlist, allowlist_record = lc.load_lpba_allowlist()
    dev = guard.load_dev_allowlist(lc.repo_root()/ALLOWLIST_REL)
    registry = load_segmenter_registry(registry_path())
    import hi_core
    segmenter_id = hi_core.CONTRACT_SEGMENTER['segmenter_id']           # unet35_v4_ens (C22.1-2 修订 4; HI's own constant)
    record.update(allowlist=allowlist_record, development_allowlist=dict(path=ALLOWLIST_REL, sha256=dev.sha256), registry=dict(path=str(registry_path())))
    subjects = {}
    for s in lists.test_subjects:
        row = subjects[s] = {}
        verdict = guard.check_subject(s, allowlist)
        row['check_subject'] = dict(admitted=verdict.admitted, admitted_by_lpba40=verdict.admitted_by_lpba40, step=verdict.step)
        if not (verdict.admitted and verdict.admitted_by_lpba40):
            problems.append(f'{s}: check_subject on the LPBA allowlist: admitted {verdict.admitted}, admitted_by_lpba40 {verdict.admitted_by_lpba40} ({verdict.step} {verdict.message})')
        row['check_subject_admitted'] = _try(problems, f'{s}: infer.check_subject_admitted', lambda s=s: infer.check_subject_admitted(s, allowlist) or 'admitted')
        chosen = _try(problems, f'{s}: infer.select_fold_models', lambda s=s: infer.select_fold_models(registry, s, allowlist, segmenter_id=segmenter_id))
        row['select_fold_models'] = None if chosen is None else dict(models=sorted(chosen[0]), policy=chosen[1])
        if chosen is not None and chosen[1] != 'all_folds_mean':
            problems.append(f'{s}: select_fold_models policy {chosen[1]} (all_folds_mean expected, D24)')
        contrast = guard.check_subject(s, dev)
        row['development_allowlist'] = dict(step=contrast.step, item=contrast.item)
        if (contrast.step, contrast.item) != ('sealed', guard.LPBA40_REJECT_ITEM):
            problems.append(f'{s}: the development allowlist does not refuse it at step sealed ({contrast.step}, {contrast.item})')
    record['subjects'] = subjects
    pairs = {}
    for f, m in lists.test_pairs:
        name = lc.pair_name(f, m)
        row = pairs[name] = {}
        guard_record = _try(problems, f'{name}: check_pair_subjects', lambda f=f, m=m: guard.check_pair_subjects(f, m, allowlist))
        lp = (guard_record or {}).get('lpba40_guard') or {}
        row['check_pair_subjects'] = dict(lpba40_guard=lp)
        if guard_record is not None and not ((lp.get('fixed') or {}).get('admitted_by_lpba40') and (lp.get('moving') or {}).get('admitted_by_lpba40')):
            problems.append(f'{name}: check_pair_subjects has no lpba40_guard with both sides admitted_by_lpba40')
        row['check_lpba40_names'] = _try(problems, f'{name}: check_lpba40_names (LPBA allowlist)', lambda name=name: guard.check_lpba40_names([name], allowlist))
        try:
            guard.check_lpba40_names([name])
            problems.append(f'{name}: check_lpba40_names without an allowlist did not refuse it')
            row['names_without_allowlist'] = 'not refused'
        except guard.SubjectSealed:
            row['names_without_allowlist'] = 'refused (SubjectSealed)'
    record['pairs'] = pairs
    function, where = pre_manifest_function() if pre_manifest is None else pre_manifest
    record['pre_manifest'] = dict(function=where, rows={})
    if function is None:
        problems.append(f'the method LPBA branch steps 1-3: {where}')
    else:
        for pair_set, rows in (('lpba40_test', lists.test_pairs), ('lpba40_feas', lists.feas_pairs)):
            for f, m in rows:
                name = lc.pair_name(f, m)
                got = _try(problems, f'{where}({pair_set}, {name})', lambda ps=pair_set, n=name: function(ps, n))
                record['pre_manifest']['rows'][f'{pair_set}:{name}'] = None if got is None else json.loads(json.dumps(got, default=str))
    record['a4_admission'] = _try(problems, 'make_inputs.a4_admission (the 34, the LPBA allowlist)',
                                  lambda: make_inputs.a4_admission(list(lists.test_subjects), str(guard.lpba40_allowlist_path()), make_inputs.UNetSource()))
    record['a4_admission'] = None if record['a4_admission'] is None else json.loads(json.dumps(record['a4_admission'], default=str))
    contrasts = {}
    for pair_set, refused in (('lpba40_test', FORMAL_REFUSED), ('lpba40_feas', FEAS_REFUSED)):
        for s in refused:
            try:
                lc.admit_subjects(pair_set, [s], allowlist, lists)
                problems.append(f'{s}: the {pair_set} entry admitted it')
                contrasts[f'{pair_set}:{s}'] = 'admitted'
            except hc.Refused as exc:
                contrasts[f'{pair_set}:{s}'] = f'refused ({exc.stage})'
    record['contrasts'] = contrasts
    record.update(problems=problems, passed=not problems, ended_utc=lc.utc())
    for problem in problems[:20]:
        log(f'problem: {problem}')
    return record


def run(*, log=print):
    """The dry run under the spy (only code-tree files); the spy's record into the result."""
    record = dict(preload=lc.preload())
    import hi_core   # noqa: F401  (imported before the spy: an import is a code read)
    import make_inputs   # noqa: F401
    from bayesreg.seg.unet import infer   # noqa: F401
    pre_manifest = pre_manifest_function()
    policy = lc.Policy(name='lpba admission dry run (C26.9-3)', read_exact=code_tree_reads(), write_inside=[], results=hc.RESULTS)
    with lc.spying(policy) as spy:
        record.update(check(pre_manifest=pre_manifest, log=log))
        record['spy'] = spy.record()
    if not record['spy'].get('passed', True):
        record['problems'].append('the spy saw an open outside the code-tree files')
        record['passed'] = False
    return record


def main(argv=None, *, out=print):
    ap = argparse.ArgumentParser(prog=SCRIPT)
    ap.add_argument('--out', type=Path, required=True, help='the record (new), e.g. runs/20261119-lpba/admission_dry_run.json')
    a = ap.parse_args(argv)
    if a.out.exists():
        out(f'{a.out} exists: a new file for every dry run')
        return 2
    try:
        record = run(log=out)
    except hc.Refused as exc:
        record = dict(schema=SCHEMA, refused=dict(stage=exc.stage, message=str(exc)), passed=False, spy=getattr(exc, 'hp_spy', None))
    except hp.Breach as exc:
        record = dict(schema=SCHEMA, refused=dict(stage='spy', message=str(exc)), passed=False)
    except Exception as exc:                                    # noqa: BLE001 - review minor 2: any other exception is recorded too (the record is the dry run's evidence)
        import traceback
        record = dict(schema=SCHEMA, refused=dict(stage='exception', message=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()), passed=False)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'x', encoding='utf-8') as stream:
        stream.write(json.dumps(record, indent=1, ensure_ascii=False, default=str)+'\n')
    out(f"admission dry run: passed {record['passed']}; {len(record.get('problems') or [])} problems; record {a.out}")
    return 0 if record['passed'] else 8


if __name__ == '__main__':
    sys.exit(main())
