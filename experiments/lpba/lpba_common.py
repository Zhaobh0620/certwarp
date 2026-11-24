"""LP-A (runs/20261119-lpba/任务书_A_数据准入HI与Atropos.md): what the LPBA40 entries of this package share -- the data preparation (prep_lpba.py), the HI mode
(experiments/hi/hi_lpba.py) and the Atropos entry (atropos_lpba.py).  Contract C26 (revision 11, 11a, 11b; main 36df068f): C26.2 (lists, pair names, roots, pins, unit
names), C26.3 (the admission constant and its one exception), C26.4 L27-L29, L48, C26.5-1 (the common rules), -2 to -4, C26.6-3 (the frozen pins every entry checks).
Preregistration docs/bayesreg/notes/LPBA40第二测试集预登记_20261119.md (final, revision 3, sha256 06121278...) §2.2, §7, §8.5, §12 items 5-8.

Two modes, one pair-set value each (C26.6-2, revision 11b): 'lpba40_test' (the 34 subjects of splits/lpba40_test_v1/test_subjects.txt and the 17 pairs of test_pairs.tsv,
in their order and direction) and 'lpba40_feas' (S01, S02 and the two pairs of feasibility_pairs.tsv).  Each has its own roots (roots(pair_set): stage, atropos, locks, the
HI output root and its pred/), all derived at call time from the code constant RESULTS (never from an option; the tests point RESULTS into tmp_path).  The hold file
<RESULTS>/lpba40_test_v1/hold pauses both (C26.2-3, prereg §8.7).

The admission of an LPBA entry (admit), in this order, before any file of a subject is opened:
  1. the pins (check_pins): the preregistration's sha256 (PREREG_SHA256); the three lists' sha256 (SHA256SUMS of splits/lpba40_test_v1); the frozen files (C26.6-3: the
     method's settings_sha256 414eccd3..., configs/frozen/holdout_B_v1_baselines.json, experiments/d0/configC_reference.json, the LPBA40 arm table); the c record's sha256
     (C_RECORD_SHA256; prereg §2.2 -- its values are read from it, after the check, never written here);
  2. the LPBA allowlist by its pinned path and sha256 only (load_lpba_allowlist: bayesreg.io.guard.lpba40_allowlist_path(), LPBA40_ALLOWLIST_SHA256 checked before it is
     parsed; its source lpba40_test_v1 exactly test_subjects.txt, never_for_training false, C26.3-5);
  3. the names (C26.3-5 last item): the formal entry takes only the 34 test subjects and the 17 test pairs (S01, S02 and S37-S40 refused), the feasibility entry only S01, S02
     and its two pairs; a pair must be a row of its table in that direction;
  4. every subject on no sealed list and through bayesreg.io.guard.check_subject on the LPBA allowlist: the 34 admitted_by_lpba40 (the one exception), S01 and S02 admitted
     through steps (2), (3) as before (not by the exception).
What it reads: code-tree files only (the lists, the allowlist, the preregistration, the frozen files, the c record, the sealed lists the guard reads).

The spy (Policy): hp_core.Policy (its audit-hook spy, hp_core.Spy, records every open and stops at the first one outside the allow list) with the name rule widened to
'label' (prereg §8.5 item 7; an LPBA40 truth file is label.nii.gz) besides hp_core's names; the data roots /mnt/d/data and the code trees' data/ judged one file at a time.

Rule one of the formal runs (C26.9-4; LP-C2): bayesreg.io.code_identity.unit_identity_problems with the LPBA40 profile (code_identity.LPBA40: P_L from the committed
runs/lpba40_test_v1/pinned_code.json at HEAD, never HEAD by default; every commit of P_L..HEAD only adds the LPBA40 authorizations of splits/lpba40_test_v1/ and the records of
runs/lpba40_test_v1/ once and appends to the decision log; the code P_L's; a clean tree) on CODE_REPO, this module's code tree.  check_code records it (rule_one: problem,
passed, record) and a formal ('lpba40_test') run refuses unless it passed (stage code; formal_code_problem for a direct caller); the feasibility mode is not under rule one
(C26.9-1: a clean detached checkout, code_commit and clean recorded).
The disks and the units (prereg §11, §8.7, §12 item 16; LP-C2): disk_guard is holdout_common.disk_guard with nothing on D: (everything on /home, prereg §11: the C: side need +
20 GiB); unit ends are classified by holdout_common.classify_unit_end with LIMIT_RULE 'yi' (C26.8-2); the machine gate is holdout_common.machine_gate with this mode's unit.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REPO = EXPERIMENTS.parent
for _p in (EXPERIMENTS/'holdout', EXPERIMENTS/'he'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import holdout_common as hc  # noqa: E402
import hp_core as hp  # noqa: E402
import he_core  # noqa: E402

array_sha256 = he_core.array_sha256                     # review minor 4: the one definition (prereg §2.2 P7: he_core.array_sha256), imported, not copied

PREREG_REL = 'docs/bayesreg/notes/LPBA40第二测试集预登记_20261119.md'
PREREG_SHA256 = '06121278d8b229401afea7cb86c841f80ba02f23bc40f8c5f5594c7da8d95643'
SPLITS_REL = 'splits/lpba40_test_v1'
TEST_SUBJECTS_REL = f'{SPLITS_REL}/test_subjects.txt'
TEST_PAIRS_REL = f'{SPLITS_REL}/test_pairs.tsv'
FEAS_PAIRS_REL = f'{SPLITS_REL}/feasibility_pairs.tsv'
LIST_SHA256 = {TEST_SUBJECTS_REL: '161f379290b6d3b0e52118570d9bc23e14197e34e77e5263c87f22a84fd5deb9',          # splits/lpba40_test_v1/SHA256SUMS (decision_log line 534)
               TEST_PAIRS_REL: '1b0c27b54384ac15f010f0c9ebaf6577d77386e9166ce4b04e21da05ab289038',
               FEAS_PAIRS_REL: '2f20a742942cb9538b7a60f1502db9f932b6b606bfbcad5a3e0a733366829ff7'}
C_RECORD_REL = 'runs/20261119-lpba/c_record.json'
C_RECORD_SHA256 = 'b6c14406377b48b52bba53bea2f173ae94f38e539b16ce11d8be3cb965602c74'      # main ee27805f (the runner's formal run; decision_log 2026-11-19 ~1:21 PM CST)
C_RECORD_SCHEMA = 'bayesreg.lpba40.c_record.v1'
SETTINGS_SHA256 = '414eccd37fe264d081e87bb63235790dc3a1ec40655c57ac64337bf84992f35d'      # experiments/hb/hb_method.py:61 (C26.2-4, C26.6-1)
FROZEN_FILES = {'configs/frozen/holdout_B_v1_baselines.json': '7c4965ebed65cfe55009518abca72c021309c366f96b8452b9c622a5f53b15a2',
                'experiments/d0/configC_reference.json': 'd331386890d90c8e5fc0c53410a8ffbde64293b1dea3851ccba87e2c9a81c1c3',
                'configs/frozen/lpba40_test_v1_arm_table.json': 'c47bbbe25d748fa14115c251ea1ceb93514eae9136ae8319ba669eb5453a29ad'}      # decision_log line 534
RESULTS = Path('/home/rmcfz/seq_vp_tag_results')
RAW_ROOT = Path('/mnt/d/data/LPBA40/nii/subjects')         # <Sxx>/delineation/image.nii.gz (prereg §1.1; C26.2-3); only read, image only
IMAGE_REL = 'delineation/image.nii.gz'
PAIR_SETS = ('lpba40_test', 'lpba40_feas')
FEAS_SUBJECTS = ('S01', 'S02')
UNIT_PREFIX = 'tag-lp-'                                  # C26.2-5
LIST_PAIR_SET = 'lpba40_test_v1'                         # the lists' name (splits/lpba40_test_v1)
FORBIDDEN_NAME_PARTS = ('label',)                        # added to hp_core.FORBIDDEN_NAME_PARTS by Policy (the LPBA40 truth file is label.nii.gz)


class Refused(hc.Refused):
    """An LPBA entry refuses before anything of a subject is opened (stage names the check)."""


def utc():
    return hc.utc()


def sha256_file(path):
    return hc.sha256_file(path)


def repo_root():
    """bayesreg's code tree (bayesreg.io.sealed.REPO, read at call time): the tree of the lists, the allowlist, the c record and the guard's sealed lists."""
    from bayesreg.io import sealed
    return Path(sealed.REPO)


def roots(pair_set):
    """The roots of one mode (C26.2-3; revision 11b: a group per pair-set value), from RESULTS at call time."""
    check_pair_set(pair_set)
    base = Path(RESULTS)
    if pair_set == 'lpba40_test':
        root, hi_out = base/'lpba40_test_v1', base/'seg'/'unet35_v4_ens_lpba40_test_v1'
    else:
        root, hi_out = base/'lpba40_feas_v1', base/'seg'/'unet35_v4_ens_lpba40_feas_v1'
    return SimpleNamespace(pair_set=pair_set, root=root, stage=root/'stage', atropos=root/'atropos', locks=root/'locks', hi_out=hi_out, pred_root=hi_out/'pred',
                           hold=base/'lpba40_test_v1'/'hold')


def check_pair_set(pair_set):
    if pair_set not in PAIR_SETS:
        raise Refused('pair_set', f'{pair_set!r}: one of {PAIR_SETS} (C26.6-2)')
    return pair_set


def unit_name(pair_set, tool):
    """tag-lp-<tool> for the formal mode, tag-lp-feas-<tool> for the feasibility mode (C26.2-5)."""
    check_pair_set(pair_set)
    name = f"{UNIT_PREFIX}{'' if pair_set == 'lpba40_test' else 'feas-'}{tool}"
    if not name.startswith(UNIT_PREFIX) or name.startswith('seq-vp-'):
        raise ValueError(name)
    return name


def image_path(subject):
    """/mnt/d/data/LPBA40/nii/subjects/<subject>/delineation/image.nii.gz (RAW_ROOT at call time).  Builds a path, opens nothing."""
    return Path(RAW_ROOT)/subject/IMAGE_REL


def pair_name(fixed, moving):
    """C26.2-2: the stage's pair directory LPBA40_<fixed>__LPBA40_<moving>."""
    return f'LPBA40_{fixed}__LPBA40_{moving}'


def number(x):
    x = float(x)
    return dict(decimal=repr(x), hex=x.hex())


# ---- the pins -------------------------------------------------------------------------------------------------------------------------------------
def _pinned_file(rel, sha256, stage):
    path = repo_root()/rel
    digest = sha256_file(path) if path.is_file() else None
    if digest != sha256:
        raise Refused(stage, f'{rel}: sha256 {digest} is not the pinned {sha256}')
    return path


def read_lists():
    """The three lists of splits/lpba40_test_v1 (sha256 against LIST_SHA256 first): test subjects (34, S03-S36 in order), test pairs (17), feasibility pairs (2)."""
    paths = {rel: _pinned_file(rel, sha, 'lists') for rel, sha in LIST_SHA256.items()}

    def rows(rel):
        lines = [line for line in paths[rel].read_text(encoding='utf-8').splitlines() if line.strip()]
        if lines[0].split('\t') != ['fixed', 'moving']:
            raise Refused('lists', f'{rel}: the header is not fixed<TAB>moving')
        return [tuple(line.split('\t')) for line in lines[1:]]
    subjects = [line for line in paths[TEST_SUBJECTS_REL].read_text(encoding='utf-8').splitlines() if line.strip()]
    test_pairs, feas_pairs = rows(TEST_PAIRS_REL), rows(FEAS_PAIRS_REL)
    from bayesreg.io.guard import LPBA40_TEST_REJECT
    if subjects != [f'S{n:02d}' for n in range(3, 37)] or set(subjects) != set(LPBA40_TEST_REJECT):
        raise Refused('lists', f'{TEST_SUBJECTS_REL}: not S03-S36 in order, or not LPBA40_TEST_REJECT')
    if test_pairs != [(f'S{n:02d}', f'S{n+1:02d}') for n in range(3, 37, 2)] or feas_pairs != [('S01', 'S02'), ('S02', 'S01')]:
        raise Refused('lists', 'the pair tables are not the 17 ordered pairs S03<-S04 ... S35<-S36 and S01<-S02, S02<-S01')
    return SimpleNamespace(test_subjects=subjects, test_pairs=test_pairs, feas_pairs=feas_pairs,
                           record={rel: dict(path=rel, sha256=LIST_SHA256[rel]) for rel in LIST_SHA256})


def table(pair_set, lists=None):
    """(subjects, pairs) of one mode, in order."""
    check_pair_set(pair_set)
    lists = read_lists() if lists is None else lists
    if pair_set == 'lpba40_test':
        return list(lists.test_subjects), list(lists.test_pairs)
    return list(FEAS_SUBJECTS), list(lists.feas_pairs)


def settings_problem():
    """C26.6-1: the method's frozen settings (experiments/he/he_core.frozen_settings) hash to SETTINGS_SHA256.  None, or the problem."""
    import he_core
    digest = he_core.settings_sha256(he_core.frozen_settings())
    return None if digest == SETTINGS_SHA256 else f'the method settings_sha256 {digest} is not the pinned {SETTINGS_SHA256}'


def check_pins():
    """Step 1 of the admission (C26.5-1, C26.6-3): the preregistration, the lists, the frozen files, the method's settings_sha256, the c record.  Refused (stage pins)."""
    record = dict(prereg=dict(path=PREREG_REL, sha256=sha256_file(_pinned_file(PREREG_REL, PREREG_SHA256, 'pins'))))
    record['lists'] = read_lists().record
    record['frozen'] = {rel: sha256_file(_pinned_file(rel, sha, 'pins')) for rel, sha in FROZEN_FILES.items()}
    problem = settings_problem()
    if problem:
        raise Refused('pins', problem)
    record['settings_sha256'] = SETTINGS_SHA256
    record['c_record'] = dict(path=C_RECORD_REL, sha256=sha256_file(_pinned_file(C_RECORD_REL, C_RECORD_SHA256, 'pins')))
    return record


def load_c_record():
    """The committed c record (prereg §2.2; sha256 C_RECORD_SHA256 checked before it is parsed): c and W_dev from their float.hex, which must give the decimal repr too."""
    path = _pinned_file(C_RECORD_REL, C_RECORD_SHA256, 'c_record')
    document = json.loads(path.read_text(encoding='utf-8'))
    if document.get('schema') != C_RECORD_SCHEMA or (document.get('prereg') or {}).get('sha256') != PREREG_SHA256:
        raise Refused('c_record', f'{C_RECORD_REL}: schema {document.get("schema")!r} or its preregistration sha256 is not the pinned one')
    values = {}
    for key in ('c', 'W_dev'):
        item = document.get(key) or {}
        try:
            value = float.fromhex(item['hex'])
        except (KeyError, TypeError, ValueError):
            raise Refused('c_record', f'{C_RECORD_REL}: {key} has no float.hex') from None
        if repr(value) != item.get('decimal') or not value > 0:
            raise Refused('c_record', f'{C_RECORD_REL}: {key} hex {item.get("hex")} and decimal {item.get("decimal")} disagree (or not > 0)')
        values[key] = value
    return SimpleNamespace(c=values['c'], W_dev=values['W_dev'], path=str(path), sha256=C_RECORD_SHA256,
                           record=dict(path=C_RECORD_REL, sha256=C_RECORD_SHA256, c=number(values['c']), W_dev=number(values['W_dev'])))


# ---- the LPBA allowlist and the admission -------------------------------------------------------------------------------------------------------
def load_lpba_allowlist():
    """C26.3-5: the LPBA entries' allowlist, loaded only by its pinned repository-relative path and sha256 (read and hashed before it is parsed); its source lpba40_test_v1
    holds exactly test_subjects.txt (path and sha256 too) and is not never_for_training.  Returns (allowlist, record); Refused (stage allowlist) otherwise."""
    from bayesreg.io import guard
    path = guard.lpba40_allowlist_path()
    raw = path.read_bytes() if path.is_file() else None
    digest = None if raw is None else hashlib.sha256(raw).hexdigest()
    if digest != guard.LPBA40_ALLOWLIST_SHA256:
        raise Refused('allowlist', f'{guard.LPBA40_ALLOWLIST_REL}: sha256 {digest} is not the pinned {guard.LPBA40_ALLOWLIST_SHA256}')
    allowlist = guard.load_dev_allowlist(path)
    if allowlist.sha256 != digest or allowlist.unseal is not None:
        raise Refused('allowlist', f'{guard.LPBA40_ALLOWLIST_REL}: changed while it was read')
    lists = read_lists()
    held = [s for s in allowlist.sources if s['name'] == guard.LPBA40_TEST_SOURCE]
    if len(held) != 1 or list(held[0]['subjects']) != lists.test_subjects or held[0]['path'] != TEST_SUBJECTS_REL \
            or held[0]['sha256'] != LIST_SHA256[TEST_SUBJECTS_REL] or held[0]['never_for_training'] is not False or held[0]['dataset'] != 'LPBA40':
        raise Refused('allowlist', f'{guard.LPBA40_ALLOWLIST_REL}: the source {guard.LPBA40_TEST_SOURCE!r} is not test_subjects.txt with never_for_training false (C26.3-5, D143)')
    return allowlist, dict(path=guard.LPBA40_ALLOWLIST_REL, sha256=digest, source=guard.LPBA40_TEST_SOURCE, reject_constant_sha256=guard.LPBA40_TEST_REJECT_SHA256,
                           rule='loaded by the pinned path and sha256 only (C26.3-5)')


def check_contact(registry, allowlist):
    """C26.3-5, C26.5-3: every label_contact_subjects of every registered model is on the LPBA allowlist and in no never_for_training source (infer.py's provenance check)."""
    off = sorted({s for e in registry.models.values() for s in e['label_contact_subjects'] if not allowlist.lists(s)})
    never = sorted({s for e in registry.models.values() for s in e['label_contact_subjects'] if s in allowlist.never_for_training})
    if off or never:
        raise Refused('admission', f'label contact subjects off the LPBA allowlist {off[:5]} or never_for_training {never[:5]}')
    return dict(models=sorted(registry.models), n_contact=len({s for e in registry.models.values() for s in e['label_contact_subjects']}), on_allowlist=True,
                never_for_training=False)


def admit_subjects(pair_set, subjects, allowlist, lists=None):
    """Steps 3-4 of the module docstring for `subjects` of mode `pair_set`.  Returns {subject: record}; Refused (stage admission or never) otherwise.  Opens no file of a subject."""
    from bayesreg.io import sealed
    from bayesreg.io.guard import LPBA40_TEST_REJECT, check_subject
    from bayesreg.io.sealed import SealedListError
    allowed, _ = table(pair_set, lists)
    out = {}
    try:
        for subject in subjects:
            if subject not in allowed:
                raise Refused('admission', f'{subject!r} is not a subject of {pair_set} ({allowed[0]} ... {allowed[-1]}): refused (C26.3-5 last item)')
            on = sealed.lists_of(sealed.guard_sealed_lists(), subject)
            if on:
                raise Refused('never', f'{subject} is on the sealed lists {list(on)}')
            verdict = check_subject(subject, allowlist)
            if not verdict.admitted:
                raise Refused('admission', f'{subject}: check_subject step {verdict.step} ({verdict.item}): {verdict.message or "not on the allowlist"}')
            if verdict.admitted_by_lpba40 != (subject in LPBA40_TEST_REJECT) or (pair_set == 'lpba40_test') != verdict.admitted_by_lpba40:
                raise Refused('admission', f'{subject}: admitted_by_lpba40 {verdict.admitted_by_lpba40} in {pair_set}')
            out[subject] = dict(step=verdict.step, item=verdict.item, dataset=verdict.dataset, admitted_by_lpba40=verdict.admitted_by_lpba40, sealed_lists=list(verdict.sealed_lists))
    except SealedListError as exc:
        raise Refused('sealed', f'the sealed lists cannot be read as pinned: {exc}') from None
    return out


def admit_pairs(pair_set, pairs, allowlist, lists=None):
    """Step 3 for pairs (a row of the mode's table, in its direction), then step 4 for their subjects.  Returns (rows, record)."""
    _, rows = table(pair_set, lists)
    index = {row: i+1 for i, row in enumerate(rows)}
    numbers = []
    for fixed, moving in pairs:
        if (fixed, moving) not in index:
            other = ' (the table has it the other way round)' if (moving, fixed) in index else ''
            raise Refused('admission', f'{pair_name(fixed, moving)} is not a row of the {pair_set} table{other}')
        numbers.append(index[(fixed, moving)])
    subjects = list(dict.fromkeys(s for p in pairs for s in p))
    return numbers, dict(pair_set=pair_set, pairs=[pair_name(f, m) for f, m in pairs], rows=numbers, subjects=admit_subjects(pair_set, subjects, allowlist, lists))


def admit(pair_set, *, pairs=None, subjects=None):
    """The whole admission of the module docstring for a run of mode `pair_set`: every pair of its table (pairs None) or `pairs`, and/or `subjects`.  Returns a namespace
    (pair_set, pairs, subjects, allowlist, lists, record)."""
    check_pair_set(pair_set)
    pins = check_pins()
    lists = read_lists()
    allowlist, allowlist_record = load_lpba_allowlist()
    table_subjects, table_pairs = table(pair_set, lists)
    pairs = list(table_pairs) if pairs is None and subjects is None else [tuple(p) for p in (pairs or [])]
    record = dict(pair_set=pair_set, pins=pins, allowlist=allowlist_record, rule='lpba_common.admit: pins -> the pinned LPBA allowlist -> the names of the mode -> sealed lists, check_subject')
    if pairs:
        record['rows'], admitted = admit_pairs(pair_set, pairs, allowlist, lists)
        record['admission'] = admitted
    names = list(dict.fromkeys([s for p in pairs for s in p]+list(subjects or [])))
    record['subjects'] = admit_subjects(pair_set, names, allowlist, lists)
    from bayesreg.io import sealed
    record['sealed_lists'] = hc.sealed_record(sealed.guard_sealed_lists())
    return SimpleNamespace(pair_set=pair_set, pairs=pairs, subjects=names, allowlist=allowlist, lists=lists, record=record, table_subjects=table_subjects)


# ---- code, hold, sandbox ----------------------------------------------------------------------------------------------------------------------------
CODE_REPO = REPO              # the code tree rule one checks: this module's (read at call time; the tests point it at a temporary git repository)


def rule_one(*, repo=None, run=None):
    """C26.9-4 (LP-C2): rule one with the LPBA40 constants -- bayesreg.io.code_identity.unit_identity_problems(profile=code_identity.LPBA40) on the code tree.
    (record, problems)."""
    from bayesreg.io import code_identity as ci
    return ci.unit_identity_problems(repo=Path(CODE_REPO if repo is None else repo), run=run, profile=ci.LPBA40)


def pinned_code_problems(pinned=None, *, repo=None, run=None):
    """Rule one for an LPBA40 formal process in the shape of holdout_common.pinned_code_problems (C26.9-2, -4; D149): (record, problems), record['pinned'] being P_L, read from the
    committed runs/lpba40_test_v1/pinned_code.json at HEAD (never HEAD by default; `pinned`, when given, only compared with it); every commit of P_L..HEAD only adds the LPBA40
    authorizations (splits/lpba40_test_v1/lpba40_test_v1_<tool>_<kind>_<k>.json, lpba40_test_v1_reeval_<k>.json) and the records under runs/lpba40_test_v1/ (the step-7 input lock
    hb_input_lock.json, the step-10 manifest and its record-only file among them) once, and appends to docs/decision_log.md; no merge commit; the code P_L's; a clean tree.
    The hook LP-B's entries call (experiments/lpba/lpba_arms.pinned_code_problems)."""
    from bayesreg.io import code_identity as ci
    return ci.unit_identity_problems(pinned, repo=Path(CODE_REPO if repo is None else repo), run=run, profile=ci.LPBA40)


def rule_one_problem(code=None, *, repo=None, run=None):
    """None when rule one for LPBA40 holds, else the problem (its words begin 'rule one for LPBA40').  `code` (the code record) is not used: P_L comes from the committed
    record only, never from HEAD (C26.9-2)."""
    _, problems = rule_one(repo=repo, run=run)
    if problems:
        return f'rule one for LPBA40 (C26.9-4: P_L from runs/lpba40_test_v1/pinned_code.json) does not hold: {problems[:5]}'
    return None


def check_code(code, pair_set, *, dry_run=False):
    """One clean tree (hp_core.check_code; the dry run records only); a formal run also rule one (rule_one; fail closed).  Returns the code record."""
    checked = hp.check_code(code, require_clean=not dry_run)
    record = dict(code, check=checked)
    if pair_set == 'lpba40_test':
        identity, problems = rule_one()
        problem = None if not problems else f'rule one for LPBA40 (C26.9-4: P_L from runs/lpba40_test_v1/pinned_code.json) does not hold: {problems[:5]}'
        record['rule_one'] = dict(problem=problem, passed=problem is None, record=json.loads(json.dumps(identity, default=str)))
        if problem and not dry_run:
            raise Refused('code', problem)
    return record


def formal_code_problem(code, pair_set, *, dry_run=False):
    """A run function's own check (as prep_stage.run_holdout's pinned_code_problem): a formal (lpba40_test) run that is not a dry run takes only a code record whose rule
    one passed (check_code put it there with no problem); fail closed for a direct caller.  None, or the problem."""
    if pair_set != 'lpba40_test' or dry_run:
        return None
    rule = (code or {}).get('rule_one') if isinstance(code, dict) else None
    if not isinstance(rule, dict) or rule.get('problem') is not None or rule.get('passed') is not True:
        return f'a formal run needs a code record whose rule one passed (C26.9-4): {rule!r}'
    return None


def check_hold(pair_set):
    """Prereg §8.7, C26.2-3: the hold file pauses every LPBA unit before it starts.  Refused (stage hold) when it is there."""
    hold = roots(pair_set).hold
    if hold.exists():
        raise Refused('hold', f'{hold} exists: the LPBA40 runs are paused (prereg §8.7)')
    return dict(hold=str(hold), present=False)


SANDBOX_ENV = hc.TEST_SANDBOX_ENV
LIMIT_RULE = 'yi'             # C26.8-2: holdout_common.classify_unit_end's rule (乙) (an OOM evidence -> infra_failure; a timeout without it -> unit_limit; unread -> rejudgment)


def disk_guard(pair_set, need_home_gib, *, usage=None):
    """Prereg §11 (C26.8-5 the disk row; §12 item 16): everything on /home, nothing on D: -- holdout_common.disk_guard(need_d_gib=0, need_home_gib): the C: side need + 20 GiB
    (C_FLOOR_GIB) against free(/mnt/c) and free(/home).  InfrastructureFailure(disk_guard) otherwise (STOP, to BZ).  Returns the record with this mode's roots."""
    check_pair_set(pair_set)
    record = hc.disk_guard(need_d_gib=0, need_home_gib=need_home_gib, usage=usage)
    r = roots(pair_set)
    return dict(record, pair_set=pair_set, roots=dict(root=str(r.root), hi_out=str(r.hi_out)), d_side='not used: everything on /home (prereg §11)')


def classify_unit_end(**kwargs):
    """holdout_common.classify_unit_end with LIMIT_RULE 'yi' (C26.8-2; prereg §8.3), whatever the caller passes as rule."""
    kwargs['rule'] = LIMIT_RULE
    return hc.classify_unit_end(**kwargs)


def guard_write(*paths):
    """In the test sandbox (holdout_common.TEST_SANDBOX_ENV set), refuse any path under the real results root, the D: data or the main checkout (RealRootInTest) before
    anything is written; otherwise nothing."""
    if os.environ.get(SANDBOX_ENV):
        for path in paths:
            target = Path(os.path.abspath(path))
            for real in (hc.REAL_RESULTS, Path('/mnt/d'), hc.MAIN_REPO):
                if target == real or real in target.parents:
                    raise hc.RealRootInTest(f'test sandbox: {path} is under the real {real}; monkeypatch the roots into tmp_path')
    return paths


# ---- the spy ------------------------------------------------------------------------------------------------------------------------------------
class Policy(hp.Policy):
    """hp_core.Policy with 'label' added to the forbidden name parts (prereg §8.5 item 7: label.nii.gz is the LPBA40 truth file; a code file and the run's own output
    directories -- Atropos's class map labels.nii.gz -- are exempt from this one name), and two read-only allowances, never written:
      `read_inside`: (directory, suffixes) pairs: a file under the directory with one of the suffixes may be read, and the directory listed (HI's registered members:
                     <seg/unet35_v4_full>/*/member.pt, found by sha256; review minor 2: the *.pt files only, nothing else of that tree);
      `list_inside`: directories that may be listed, nothing read through this allowance (the stage, whose files check_stage compares with its SHA256SUMS; review minor 3)."""

    def __init__(self, *, read_inside=(), list_inside=(), **kwargs):
        super().__init__(**kwargs)
        self.read_inside = sorted({(Path(root).resolve(), tuple(suffixes)) for root, suffixes in read_inside})
        self.list_inside = sorted({Path(p).resolve() for p in list_inside})

    def _named_label(self, path, write):
        raw, resolved = hp._forms(path)
        if raw is not None and hp._under(resolved, self.write_inside):
            return None                                  # the run's own outputs (Atropos writes its class map labels.nii.gz there): not an LPBA40 truth file
        if raw is not None and not (not write and self._code(resolved)):
            for form in (raw, resolved):
                if any(part in form.name.lower() for part in FORBIDDEN_NAME_PARTS):
                    return f'a label file by its name ({form.name}; LPBA40 truth)'
        return None

    def classify(self, path, write):
        why = self._named_label(path, write)
        if why:
            return 'forbidden', why
        verdict = super().classify(path, write)
        if verdict[0] != 'allowed' and not write and not verdict[1].startswith(('a label or evaluation file', 'a real root')):
            raw, resolved = hp._forms(path)
            if resolved is not None and any(hp._under(resolved, [root]) and resolved != root and resolved.suffix in suffixes for root, suffixes in self.read_inside):
                return 'allowed', 'a read-only input file of the run (read_inside: its directory and suffix)'
        return verdict

    def classify_listing(self, path):
        raw, resolved = hp._forms(path)
        if resolved is not None and (any(hp._under(resolved, [root]) for root, _ in self.read_inside) or hp._under(resolved, self.list_inside)):
            return 'allowed', 'a read-only input directory of the run (read_inside or list_inside: a listing)'
        return super().classify_listing(path)

    def document(self):
        return dict(super().document(), forbidden_name_parts=list(hp.FORBIDDEN_NAME_PARTS)+list(FORBIDDEN_NAME_PARTS),
                    read_inside=[dict(root=str(root), suffixes=list(suffixes)) for root, suffixes in self.read_inside], list_inside=list(map(str, self.list_inside)))


# ---- the stage, as prep_lpba wrote it ---------------------------------------------------------------------------------------------------------------
STAGE_SUMS, PREP_RECORD = 'SHA256SUMS', 'prep_record.json'
MM_DIR = '_nifti_mm'                                     # prep_lpba.MM_DIR (a test compares)


def stage_paths(pair_set, pairs):
    """The stage files a reader of the stage may open: SHA256SUMS, prep_record.json, each pair's pair.npz and manifest.json, the subjects' mm copies (hashed only, by
    check_stage)."""
    stage = roots(pair_set).stage
    copies = [stage/MM_DIR/f'{s}_image.nii.gz' for s in dict.fromkeys(s for p in pairs for s in p)]
    return [stage/STAGE_SUMS, stage/PREP_RECORD]+[stage/pair_name(f, m)/n for f, m in pairs for n in ('pair.npz', 'manifest.json')]+copies


def check_stage(pair_set, pairs):
    """The stage prep_lpba.py wrote for this mode: prep_record.json passed, of this pair set, with the preregistration and the c record pinned here, its pairs exactly
    `pairs` in order, its SHA256SUMS the stage's; every pair's pair.npz and manifest.json listed in SHA256SUMS and unchanged now.  Refused (stage stage) otherwise."""
    stage = roots(pair_set).stage
    try:
        lines = [line for line in (stage/STAGE_SUMS).read_text(encoding='utf-8').splitlines() if line.strip()]
        prep = json.loads((stage/PREP_RECORD).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise Refused('stage', f'{stage}: the stage cannot be read ({type(exc).__name__}: {exc})') from None
    listed = dict(reversed(line.split('  ', 1)) for line in lines)
    want = [pair_name(f, m) for f, m in pairs]
    problems = []
    present = sorted(str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file() and p.name not in (STAGE_SUMS, PREP_RECORD))      # review minor 3 (a listing: list_inside)
    if sorted(listed) != present:
        problems.append(f'SHA256SUMS does not list exactly the stage\'s files (listed only {sorted(set(listed)-set(present))[:3]}, on disk only {sorted(set(present)-set(listed))[:3]})')
    expected = sorted({f'{MM_DIR}/{s}_image.nii.gz' for p in pairs for s in p} | {f'{n}/{leaf}' for n in want for leaf in ('pair.npz', 'manifest.json')})
    if present != expected:
        problems.append(f'the stage\'s files are not the mm copies and pair files of these pairs ({len(present)} files, {len(expected)} expected)')
    if prep.get('passed') is not True or prep.get('pair_set') != pair_set:
        problems.append(f"prep_record.json: passed {prep.get('passed')}, pair set {prep.get('pair_set')!r}")
    summary = prep.get('stage_summary') or {}
    if [row.get('pair') for row in summary.get('pairs') or []] != want:
        problems.append('prep_record.json does not list exactly these pairs in order')
    if summary.get('sha256sums_sha256') != sha256_file(stage/STAGE_SUMS):
        problems.append('prep_record.json names another SHA256SUMS')
    if ((prep.get('admission') or {}).get('pins') or {}).get('prereg', {}).get('sha256') != PREREG_SHA256 or (prep.get('c_record') or {}).get('sha256') != C_RECORD_SHA256:
        problems.append('prep_record.json was made under another preregistration or c record')
    for name in want:
        for leaf in ('pair.npz', 'manifest.json'):
            rel = f'{name}/{leaf}'
            if rel not in listed:
                problems.append(f'{rel} not in SHA256SUMS')
            elif sha256_file(stage/rel) != listed[rel]:
                problems.append(f'{rel} differs from SHA256SUMS')
    for rel in sorted(set(listed) & set(present)):
        if rel.startswith(MM_DIR+'/') and sha256_file(stage/rel) != listed[rel]:
            problems.append(f'{rel} differs from SHA256SUMS')
    if problems:
        raise Refused('stage', f'{stage}: {problems[:5]}')
    return dict(stage=str(stage), pairs=want, sha256sums_sha256=summary['sha256sums_sha256'], prep_record_sha256=sha256_file(stage/PREP_RECORD), n_files=len(present),
                checked='prep_record.json (passed, pair set, pairs, SHA256SUMS, preregistration, c record); SHA256SUMS lists exactly the stage\'s files (the mm copies and the '
                        'pair files of these pairs); every file against SHA256SUMS')


def code_reads():
    """The code-tree files every LPBA admission reads (allowed one by one): the lists, the allowlist, the preregistration, the frozen files, the c record, the sealed lists."""
    from bayesreg.io import guard
    import firewall_c
    tree = repo_root()
    return [tree/rel for rel in (*LIST_SHA256, PREREG_REL, *FROZEN_FILES, C_RECORD_REL, guard.LPBA40_ALLOWLIST_REL)]+hp.allowed_list_paths()+[
        Path(firewall_c.CONFIGC_REFERENCE)]          # he_core.frozen_settings hashes it from firewall_c's tree (settings_problem; the same file as FROZEN_FILES' in a run)


class Spy(hp.Spy):
    """hp_core.Spy, except that a member of an npz file in the run's own output directories may be asked (HI's writer reads back its structure_groups.npz, key p);
    a pair.npz anywhere else still answers fixed, moving and initial_u only."""

    def see_member(self, npz, key):
        source = getattr(getattr(npz, 'fid', None), 'name', None)
        raw, resolved = hp._forms(source)
        if resolved is not None and hp._under(resolved, self.policy.write_inside):
            self.members.append(dict(file=str(source), member=str(key), own_output=True))
            return
        super().see_member(npz, key)


@contextmanager
def spying(policy):
    """hp_core.spying with this module's Spy: installed only when the guard's cache of the sealed lists is empty (C23.6-5), Refused (stage spy) otherwise; removed after;
    an exception leaving the block carries the spy's record as `hp_spy`."""
    from bayesreg.io.sealed import sealed_cache_is_empty
    if not sealed_cache_is_empty():
        raise Refused('spy', 'the guard\'s cache of the sealed lists is populated before the spy was installed (C23.6-5): run the entry in a process of its own')
    spy = Spy(policy).install()
    try:
        yield spy
    except BaseException as exc:
        try:
            exc.hp_spy = spy.record()
        except Exception:                       # noqa: BLE001 - the record is a convenience
            pass
        raise
    finally:
        spy.remove()


def preload():
    """hp_core.preload plus what the LPBA entries import (an import is a code read; the run should record its reading, not its imports)."""
    record = hp.preload()
    import he_core  # noqa: F401
    import firewall_c  # noqa: F401
    he_core.frozen_settings()                 # imports what settings_problem needs, before the spy
    return record


def dry_run_out_problem(out):
    return hp.dry_run_out_problem(out)


__all__ = ['CODE_REPO', 'C_RECORD_REL', 'C_RECORD_SHA256', 'FEAS_SUBJECTS', 'LIMIT_RULE', 'PAIR_SETS', 'PREREG_SHA256', 'Policy', 'RAW_ROOT', 'RESULTS', 'Refused', 'admit',
           'admit_pairs', 'admit_subjects', 'check_code', 'check_contact', 'classify_unit_end', 'disk_guard', 'check_hold', 'check_pins', 'code_reads', 'guard_write', 'image_path', 'load_c_record', 'load_lpba_allowlist', 'number', 'pair_name',
           'pinned_code_problems', 'preload', 'read_lists', 'roots', 'rule_one', 'rule_one_problem', 'spying', 'table', 'unit_name']
