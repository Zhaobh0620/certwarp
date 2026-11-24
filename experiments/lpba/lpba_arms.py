"""LP-B (runs/20261119-lpba/任务书_B_方法配置C与外部臂.md): what the LPBA40 modes of the method arm (HE: experiments/he/run_hx.py, he_holdout.py; its queue, review and
store: experiments/hb/), the configuration C arm (experiments/hb/hb_configc.py) and the four external arms (experiments/xb/) share.  Contract C26 (revision 11, 11a,
11b, 11c): C26.2-3 (the roots), C26.2-5 (the unit names), C26.6, C26.8-1, -7, C26.9 (rule one: LP-C2's lpba_common, through pinned_code_problems below); preregistration
(06121278...) §4, §8.2, §8.3, §8.7, §11.

The places mirror the hold-out's (configs/frozen/lpba40_test_v1_arm_table.json, LP-T0: holdout_B_v1/ -> lpba40_test_v1/, the hold-out's D: places -> lpba40_test_v1/work/,
nothing moves to D:, §11):
  the root of a mode     lpba_common.roots(pair_set).root: <results>/lpba40_test_v1 (formal) or <results>/lpba40_feas_v1 (feasibility), read at call time
  a cell's work          <root>/work/<arm>/<pair>       (holdout_common.work_dir with the mode's root: the method's cell is HE's output directory; the fields stay there)
  a cell's records       <root>/<arm>/<pair>            (holdout_common.record_dir with the mode's root)
  locks, batches         <root>/locks/, <root>/batches/
  the hold file          <results>/lpba40_test_v1/hold  (both modes; C26.2-3, prereg §8.7)
  pair names             LPBA40_<fixed>__LPBA40_<moving> (C26.2-2), a row of the mode's table in its direction (lpba_common.table)
  units                  tag-lp-<kind>-<Sff>-<Smm> and tag-lp-<kind>-queue (formal), tag-lp-feas-<kind>-... (feasibility); kind he, review, c, x-<arm>; HE's own nested
                         units tag-lp-he-c-<pair>, tag-lp-he-c3-<pair> (C26.2-5: never tag-hb-*, tag-he-*, seq-vp-*)
In the test sandbox (holdout_common.TEST_SANDBOX_ENV) every place under the real results root, D: or the main checkout is refused (lpba_common.guard_write) before it is used.

Rule one (C26.9-4) and the authorizations of the formal mode (C26.8-3, D148) are LP-C2's (decision_log line 558 item 3), merged with LP-B (decision_log line 584 ruling 3):
pinned_code_problems is the one hook every LP-B entry calls for the formal mode, and it delegates to lpba_common.pinned_code_problems -- bayesreg.io.code_identity with the
LPBA40 constants, P_L from the committed runs/lpba40_test_v1/pinned_code.json (without that record a formal process is refused by rule one itself); holdout_common knows the
formal mode's authorizations (AUTH_DIRS['lpba40_test'] = splits/lpba40_test_v1, D148), the feasibility mode has none.  The feasibility mode is not under rule one (C26.9-1): a
clean tree, its commit recorded.  (LP-C2 integration review minor 6: the old fail-closed stand-in constant is removed.)
"""
from contextlib import contextmanager
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import lpba_common as lc  # noqa: E402

hc = lc.hc
PAIR_SETS = lc.PAIR_SETS
FORMAL, FEAS = 'lpba40_test', 'lpba40_feas'
SETTINGS_SHA256 = lc.SETTINGS_SHA256           # C26.2-4, C26.6-1: the method's frozen settings (experiments/hb/hb_method.py:61)
PAIR_NAME = re.compile(r'LPBA40_(S[0-9]{2})__LPBA40_(S[0-9]{2})', re.ASCII)
UNIT_KINDS = ('he', 'review', 'c', 'he-c', 'he-c3')     # plus x-<arm> (the external arms)
P19_STOP_AT = 2                                 # prereg §6.6 (裁决 (8)), §12 item 11: the second pair of an external arm whose export check is suspicious stops the arm


def check_pair_set(pair_set):
    return lc.check_pair_set(pair_set)


def root(pair_set):
    """The mode's root (C26.2-3), from lpba_common.RESULTS at call time; refused in the test sandbox when it is a real root."""
    path = lc.roots(pair_set).root
    lc.guard_write(path)
    return path


def split(pair):
    """(fixed, moving) of LPBA40_<fixed>__LPBA40_<moving>; Refused (stage admission) otherwise.  Ids only."""
    m = PAIR_NAME.fullmatch(str(pair))
    if not m:
        raise lc.Refused('admission', f'{pair!r} is not LPBA40_<Sxx>__LPBA40_<Syy> (C26.2-2)')
    return m.group(1), m.group(2)


def short(pair):
    """LPBA40_S03__LPBA40_S04 -> S03-S04 (unit names)."""
    fixed, moving = split(pair)
    return f'{fixed}-{moving}'


def rows(pair_set):
    """The mode's pairs, LPBA40_<fixed>__LPBA40_<moving>, in the table's order (lpba_common.table: the lists' sha256 checked; ids only)."""
    return [lc.pair_name(f, m) for f, m in lc.table(pair_set)[1]]


def planned(pair_set, pair):
    """Whether `pair` is a row of the mode's table in its direction (ids only); False for anything else, a refusal of the lists included."""
    try:
        return pair in rows(pair_set)
    except lc.Refused:
        return False


def work_dir(arm, pair, pair_set):
    path = Path(hc.work_dir(arm, pair, root(pair_set)))
    lc.guard_write(path)
    return path


def record_dir(arm, pair, pair_set):
    path = Path(hc.record_dir(arm, pair, root(pair_set)))
    lc.guard_write(path)
    return path


def arm_dir(arm, pair_set):
    path = Path(hc.arm_dir(arm, root(pair_set)))
    lc.guard_write(path)
    return path


def lock_dir(pair_set):
    return root(pair_set)/'locks'


def batch_root(pair_set):
    return root(pair_set)/'batches'


def hold(pair_set):
    """The hold file of both modes (C26.2-3; prereg §8.7)."""
    path = lc.roots(pair_set).hold
    lc.guard_write(path)
    return path


def unit_name(kind, pair_set, pair=None):
    """tag-lp-<kind>-<Sff>-<Smm> (formal) or tag-lp-feas-<kind>-<Sff>-<Smm> (feasibility); without a pair the queue's own unit tag-lp[-feas]-<kind>-queue (C26.2-5)."""
    if not (kind in UNIT_KINDS or (kind.startswith('x-') and re.fullmatch(r'x-[a-z0-9_]+', kind))):
        raise ValueError(f'unit kind {kind!r}')
    return lc.unit_name(pair_set, f"{kind}-{'queue' if pair is None else short(pair)}")


def unit_prefix(kind, pair_set):
    """The prefix of a unit name whose last part is given by another tool (HE's nested units: firewall_c.UnitLauncher's unit_prefix)."""
    return lc.unit_name(pair_set, kind)


# ---- the code (C26.9) ---------------------------------------------------------------------------------------------------------------------------
def pinned_code_problems(pinned=None, *, repo=None, run=None):
    """Rule one of a formal LPBA40 process, in the shape of holdout_common.pinned_code_problems: (record, problems), record['pinned'] being P_L.  The hook of every LP-B entry;
    LP-C2 (C26.9-4, D149): lpba_common.pinned_code_problems -- bayesreg.io.code_identity.unit_identity_problems with the LPBA40 constants (code_identity.LPBA40): P_L from the
    committed runs/lpba40_test_v1/pinned_code.json, never HEAD (`pinned` only compared with it); every commit of P_L..HEAD only adds the LPBA40 authorizations and the records of
    runs/lpba40_test_v1/ (hb_input_lock.json among them) once and appends to the decision log; the code P_L's; a clean tree."""
    return lc.pinned_code_problems(pinned, repo=repo, run=run)


def detached_problem(root, *, run=None):
    """C26.9-1 (LP-B review minor 5): a feasibility run runs from a detached checkout of its commit that is not the main checkout (holdout_common.MAIN_REPO): None, or why not.  `run`:
    subprocess.run's stand-in (git symbolic-ref -q HEAD: exit 0 on a branch, 1 detached)."""
    run = subprocess.run if run is None else run
    root = Path(root)
    if root.resolve() == Path(hc.MAIN_REPO).resolve():
        return f'{root} is the main checkout: a feasibility run runs from a detached checkout of its commit (C26.9-1)'
    try:
        done = run(['git', '--literal-pathspecs', '-C', str(root), 'symbolic-ref', '-q', 'HEAD'], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return f'git symbolic-ref of {root}: {type(exc).__name__}: {exc}'
    if done.returncode == 0:
        return f'{root} is on the branch {done.stdout.strip()}: a feasibility run runs from a detached checkout (C26.9-1)'
    if done.returncode != 1:
        return f'git symbolic-ref of {root} exited {done.returncode}'
    return None


def code_record(pair_set, pinned=None, *, run=None, dry_run=False, repo=None):
    """The code of a batch or a unit of an LPBA mode: holdout_common.code_identity (HEAD, clean, the untracked code files), and
      formal:       rule one (pinned_code_problems; P_L, never HEAD; `pinned` only compared with it): the record's commit is P_L, head HEAD;
      feasibility:  a clean, detached checkout that is not the main checkout (C26.9-1; detached_problem), its commit recorded; the commit is HEAD, `pinned` (the batch's or the chain's
                    commit) compared with it.
    A dry run records the problems and goes on.  Returns (record, problems); the caller refuses on problems (unless a dry run)."""
    check_pair_set(pair_set)
    code = hc.code_identity(repo=repo, run=run)
    problems = []
    if not code['clean']:
        problems.append(f'the code tree is not clean ({code}): an LPBA40 run runs from a committed tree')
    if pair_set == FORMAL:
        record, issues = pinned_code_problems(pinned, repo=repo, run=run)
        problems += list(issues)
        return dict(code, head=code.get('commit'), commit=record.get('pinned'), pinned_code=record, pair_set=pair_set), problems
    detached = detached_problem(code['root'], run=run)
    if detached:
        problems.append(detached)
    if pinned is not None and pinned != code.get('commit'):
        problems.append(f'--commit {pinned} is not HEAD {code.get("commit")} (feasibility: the commit is the checkout\'s HEAD)')
    return dict(code, head=code.get('commit'), pair_set=pair_set, detached=detached is None,
                rule='C26.9-1: feasibility, not under rule one; a clean detached checkout (not the main checkout), its commit recorded'), problems


def formal_code_problem(pair_set, *, repo=None, run=None):
    """LP-B review minor 7 (as lpba_common.formal_code_problem): a tool of the formal mode that is not started through a queue's start checks checks rule one itself (pinned_code_problems,
    LP-C2's, C26.9-4; fail closed).  None for the feasibility mode, or when rule one passes."""
    if pair_set != FORMAL:
        return None
    _, problems = pinned_code_problems(None, repo=repo, run=run)
    return None if not problems else f'rule one of the formal mode (C26.9-4): {problems}'


# ---- the allow-list spy of the method's processes and the configuration C arm (LP-B review I3; prereg §8.5 item 7; C26.5-1, C26.8-5) -------------------
SPY_BREACH = 'spy_breach.json'
SPY_BREACH_EXIT = 7                             # = hb_review.SPY_BREACH_EXIT (a test compares): the review unit reads it as "its spy stopped a read"
LABEL_PART = 'label'                            # LPBA40's truth file is label.nii.gz (prereg §8.5 item 7; lpba_common.FORBIDDEN_NAME_PARTS)
DRIVE_D = Path('/mnt/d')                        # LP-B diff review I3-a: the whole D: (the data, the hold-out's D: store, paper 1's /mnt/d/seq_vp_3d_results) is a forbidden root
CODE_TREE_FORBIDDEN = ('archive', 'docs/bayesreg/reviews')      # I3-a: in every code tree (and the main checkout): the archive and the reviews are not code
FROZEN_TREE_FORBIDDEN = ('results', 'docs', 'figures')         # I3-a: in the frozen tree (frozen_eb808be under the results root): its results, documents and figures are not code
CODE_CACHE = '__pycache__'                      # a code tree's byte-code and numba caches: code (read and written, no name rule; listed)
RESULT_DOC_NAME = re.compile(r'^result.*\.md$')  # LP-B quick diff review minor 2 (LP-C2 integration item 7): a result document by its name (hb_review.RESULT_DOC_NAME, C22.H4-3; a test compares)
CODE_CONFIGS = ()                               # I3-a: the configuration files of a code tree a process needs whose names the name rule would refuse (relative paths); none needed (the t88 read chain,
                                                # the configuration C driver and the third certification read none; tests/test_bayesreg/hb/test_hb_lpba.py)


class ArmPolicy:
    """What one process of the method arm (supervisor, t88, repair, third certification) or of the configuration C arm may open, judged only where the data and the results lie
    (LP-B review I3: "只管数据根与结果根，库文件不管"; diff review I3-a: the code trees' records are not code).  In this order:
      the run's own directories (write_inside) and lock files (write_exact): read and written, no name rule (the frozen run's labels.json, a private copy of the frozen tree);
      the inputs allowed one by one (read_exact) or as read-only directories (read_inside: a subject's Atropos output, the registered members): read only; read_exact always holds the
        files the guard reads for the sealed lists (hp_core.allowed_list_paths = holdout_common.sealed_list_paths, as prep and XB: C23.6-5, C22.H4-3; LP-G, decision_log 2026-11-20),
        so the three under the code tree's archive/ are read although archive/ is a forbidden root -- these four exact files only, judged in their resolved form; a write to them, any other
        file of archive/ (the rest of plan/ among them) and a listing of plan/ stay forbidden;
      the forbidden roots: the data roots (/mnt/d/data), the whole D: (/mnt/d: the hold-out's D: store, paper 1's /mnt/d/seq_vp_3d_results), seq_vp_3d, the code trees' and the main
        checkout's data/, archive/ and docs/bayesreg/reviews/, the frozen tree's results/, docs/ and figures/;
      the main checkout (holdout_common.MAIN_REPO) whole, but a code tree that lies inside it (LP-B quick diff review minor 2; LP-C2 integration item 7: an LPBA process runs from
        <formal root>/code, C26.9-5, so nothing of the main checkout is its code -- the tests' worktree and main checkout are code trees inside it);
      the code trees (the running checkout, wherever it lies -- <formal root>/code, C26.9-5 --, bayesreg's tree, the frozen tree): a file of a code suffix (hp_core.CODE_SUFFIXES) or in a
        __pycache__ (the byte-code and numba caches) is code, read without the name rule; a configuration file of CODE_CONFIGS too; every other file of a code tree goes through the name rule;
        a write only into a __pycache__; every open but a read of a .py is listed in the record;
      the Python environment and the system: read only;
      a label or evaluation file by its name (hp_core.FORBIDDEN_NAME_PARTS and 'label'), or a result document (RESULT_DOC_NAME: result*.md; quick diff review minor 2): forbidden;
      the results root: forbidden (not allowed one by one); a write into a code tree: forbidden;
      anything else (the caches, /tmp, /dev): not watched, allowed.
    A write into a real root while the test sandbox is set is forbidden first (holdout_common.real_root_problem)."""

    def __init__(self, *, name, read_exact=(), read_inside=(), write_inside=(), write_exact=(), results=(), code_roots=(), frozen_roots=()):
        hp = lc.hp
        r = lambda p: Path(p).resolve()     # noqa: E731
        both = lambda ps: sorted({x for p in ps for x in (Path(os.path.abspath(p)), r(p))})     # noqa: E731
        self.name = name
        self.sealed_lists = {r(p) for p in hp.allowed_list_paths()}       # LP-G (decision_log 2026-11-20): C23.6-5, C22.H4-3 -- the guard's four sealed lists, one by one (three lie under archive/)
        self.read_exact = {r(p) for p in read_exact} | self.sealed_lists
        self.read_inside = both(read_inside)
        self.write_inside = both(write_inside)
        self.write_exact = {r(p) for p in write_exact}
        trees = [*code_roots, *frozen_roots]
        roots = [Path(p) for p in hp.DATA_ROOTS]+[DRIVE_D, hp.SEQ_VP_3D]+[Path(c)/'data' for c in (hc.MAIN_REPO, *trees)]
        roots += [Path(c)/sub for c in (hc.MAIN_REPO, *trees) for sub in CODE_TREE_FORBIDDEN]+[Path(f)/sub for f in frozen_roots for sub in FROZEN_TREE_FORBIDDEN]
        self.forbidden_roots = both(roots)
        self.main_checkout = both([hc.MAIN_REPO])
        self.results = both(results)
        self.code_roots = both(trees)
        self.frozen_roots = both(frozen_roots)
        self.system_roots = both({sys.prefix, sys.base_prefix, sys.exec_prefix, *hp.SYSTEM_READ})
        self.name_parts = tuple(hp.FORBIDDEN_NAME_PARTS)+(LABEL_PART,)
        self.code_suffixes = tuple(hp.CODE_SUFFIXES)
        self.configs = {Path(c)/rel for c in self.code_roots for rel in CODE_CONFIGS}

    @staticmethod
    def _in(raw, resolved, roots):
        under = lc.hp._under
        return under(raw, roots) or under(resolved, roots)

    def classify(self, path, write):
        hp = lc.hp
        raw, resolved = hp._forms(path)
        if raw is None:
            return 'allowed', 'not a path (a descriptor)'
        if write and os.environ.get(hc.TEST_SANDBOX_ENV) and (hp.real_root_problem(raw) or hp.real_root_problem(resolved)):
            return 'forbidden', 'a real root written while the test sandbox is set'
        if resolved in self.write_exact or self._in(raw, resolved, self.write_inside):
            return 'allowed', 'own'
        if not write and (resolved in self.read_exact or self._in(raw, resolved, self.read_inside)):
            return 'allowed', 'input'
        if self._in(raw, resolved, self.forbidden_roots):
            return 'forbidden', ('a data root, D:, the seq_vp_3d tree, or a code tree\'s data/, archive/, reviews (the frozen tree\'s results/, docs/, figures/): not allowed one by one')
        if self._main_checkout(raw, resolved):
            return 'forbidden', 'the main checkout outside the code trees (an LPBA process runs from <formal root>/code, C26.9-5)'
        if self._in(raw, resolved, self.code_roots):
            cache = CODE_CACHE in raw.parts or CODE_CACHE in resolved.parts
            if write:
                return ('allowed', 'code cache') if cache else ('forbidden', 'a write into a code tree (only its __pycache__: the byte-code and numba caches)')
            if raw.suffix == '.py' and resolved.suffix == '.py':
                return 'allowed', 'code'                                            # counted, not listed
            if cache or (raw.suffix in self.code_suffixes and resolved.suffix in self.code_suffixes):
                return 'allowed', 'code (listed)'
            if raw in self.configs or resolved in self.configs:
                return 'allowed', 'code configuration (listed)'
            if self._name_hit(raw, resolved):
                return 'forbidden', f'a label or evaluation file of a code tree by its name ({raw.name}; only code and CODE_CONFIGS are exempt, I3-a)'
            return 'allowed', 'code tree file (listed)'
        if not write and self._in(raw, resolved, self.system_roots):
            return 'allowed', 'system'
        if self._name_hit(raw, resolved):
            return 'forbidden', f'a label or evaluation file by its name ({raw.name})'
        if self._in(raw, resolved, self.results):
            return 'forbidden', 'the results root outside the run\'s own files and inputs'
        return 'allowed', 'not watched'

    def _name_hit(self, raw, resolved):
        return any(part in form.name.lower() for form in (raw, resolved) for part in self.name_parts) or any(RESULT_DOC_NAME.match(form.name) for form in (raw, resolved))

    def _main_checkout(self, raw, resolved):
        """Integration item 7: a form of the path (as given, or resolved) inside the main checkout and inside none of the code trees."""
        under = lc.hp._under
        return any(under(form, self.main_checkout) and not under(form, self.code_roots) for form in (raw, resolved))

    def classify_listing(self, path):
        hp = lc.hp
        raw, resolved = hp._forms(path)
        if raw is None or self._in(raw, resolved, self.write_inside) or self._in(raw, resolved, self.read_inside):
            return 'allowed', 'own or input'
        if self._in(raw, resolved, self.forbidden_roots):
            return 'forbidden', 'a listing of a forbidden root (data, D:, seq_vp_3d, a code tree\'s records)'
        if self._main_checkout(raw, resolved):
            return 'forbidden', 'a listing of the main checkout outside the code trees (integration item 7)'
        if self._in(raw, resolved, self.code_roots):
            return 'allowed', 'code'
        if self._in(raw, resolved, self.results):
            return 'forbidden', 'a listing of the results root outside the run\'s own directories'
        return 'allowed', 'not watched'

    def document(self):
        return dict(name=self.name, kind='LP-B allow list (data and results roots only)', read_exact=sorted(map(str, self.read_exact)), sealed_lists=sorted(map(str, self.sealed_lists)),
                    read_inside=list(map(str, self.read_inside)),
                    write_inside=list(map(str, self.write_inside)), write_exact=sorted(map(str, self.write_exact)), forbidden_roots=list(map(str, self.forbidden_roots)),
                    results=list(map(str, self.results)), code_roots=list(map(str, self.code_roots)), frozen_roots=list(map(str, self.frozen_roots)), name_parts=list(self.name_parts),
                    main_checkout=list(map(str, self.main_checkout)), result_doc_name=RESULT_DOC_NAME.pattern,
                    code_exempt=dict(suffixes=list(self.code_suffixes), cache=CODE_CACHE, configs=sorted(map(str, self.configs))),
                    listed='every open but a read of a .py in a code tree, of the system or of an unwatched place')


class ArmSpy(lc.hp.Spy):
    """hp_core.Spy (the audit hook: open, os.listdir, os.scandir; NpzFile.__getitem__) with ArmPolicy: an allowed open of the code, the system or an unwatched place is counted, not listed
    (a GPU process opens thousands of library files); a member is judged only of a file named pair.npz outside the run's own directories (fixed, moving, initial_u); every other archive
    (the predictions, the fields, the derived pair) passes."""

    def __init__(self, policy):
        super().__init__(policy)
        self.n_opens = 0

    def see_open(self, args):
        hp = lc.hp
        path = args[0] if args else None
        raw, resolved = hp._forms(path)
        if raw is None:
            return
        write = hp._is_write(args[1] if len(args) > 1 else None, args[2] if len(args) > 2 else 0)
        cls, why = self.policy.classify(path, write)
        self.n_opens += 1
        if cls == 'allowed' and why in ('code', 'system', 'not watched'):        # counted, not listed
            return
        self.events.append(dict(path=str(raw), resolved=str(resolved), write=write, cls=cls, why=why))
        if cls != 'allowed':
            self.breaches.append(dict(path=str(raw), resolved=str(resolved), write=write, cls=cls, why=why))
            raise hp.Breach(f'LP-B spy ({self.policy.name}): {cls} open of {raw} ({why})')

    def see_member(self, npz, key):
        hp = lc.hp
        source = getattr(getattr(npz, 'fid', None), 'name', None)
        raw, resolved = hp._forms(source)
        own = resolved is not None and ArmPolicy._in(raw, resolved, self.policy.write_inside)
        self.members.append(dict(file=str(source), member=str(key)))
        if raw is not None and raw.name == 'pair.npz' and not own and key not in hp.PAIR_MEMBERS:
            self.breaches.append(dict(path=str(source), member=str(key), cls='forbidden', why='a pair.npz member other than fixed, moving, initial_u'))
            raise hp.Breach(f'LP-B spy ({self.policy.name}): the member {key!r} of {source} was asked')

    def summary(self):
        return dict(spy='lpba_arms.ArmSpy (hp_core audit hook; ArmPolicy)', policy=self.policy.document(), n_opens=self.n_opens, events=self.events, members=self.members,
                    breaches=list(self.breaches), passed=not self.breaches,
                    rule='LP-B review I3, prereg §8.5 item 7: the data and results roots judged against the allow list, the libraries not watched; a breach stops (C26.8-5)')


@contextmanager
def arm_spying(policy):
    """Install an ArmSpy for `policy`: only when the guard's cache of the sealed lists is empty (C23.6-5) -- Refused (stage spy) otherwise; removed after (also on an exception)."""
    from bayesreg.io.sealed import sealed_cache_is_empty
    if not sealed_cache_is_empty():
        raise lc.Refused('spy', 'the guard\'s cache of the sealed lists is populated before the spy was installed (C23.6-5): run the process on its own')
    spy = ArmSpy(policy).install()
    try:
        yield spy
    finally:
        spy.remove()


SPY_BREACH_NO_RERUN = ('LP-B diff review minor 3 (decision_log 2026-11-19 ~3:51 PM CST): a spy breach is not rerun through an authorization (no resume, no rerun record): the run stops '
                       'and goes to BZ (C26.8-5)')


def spy_breach_records(*directories):
    """The spy breach records (spy_breach*.json) anywhere below the given directories (those that exist), sorted."""
    return sorted(str(p) for d in directories if d is not None and Path(d).is_dir() for p in Path(d).glob('**/spy_breach*.json'))


def spy_breach_problem(*directories):
    """None, or why a resume or rerun authorization of this cell is refused: a spy breach record below one of `directories` (SPY_BREACH_NO_RERUN)."""
    found = spy_breach_records(*directories)
    return None if not found else f'{SPY_BREACH_NO_RERUN}: {found}'


def results_roots():
    """The results root as the tools see it now (holdout_common.RESULTS and lpba_common.RESULTS: the same in production; the tests move them)."""
    return sorted({Path(hc.RESULTS), Path(lc.RESULTS)})


__all__ = ['ArmPolicy', 'ArmSpy', 'FEAS', 'FORMAL', 'P19_STOP_AT', 'PAIR_NAME', 'PAIR_SETS', 'SETTINGS_SHA256', 'SPY_BREACH', 'SPY_BREACH_EXIT', 'arm_dir', 'arm_spying',
           'batch_root', 'check_pair_set', 'code_record', 'detached_problem', 'formal_code_problem', 'hold', 'results_roots', 'spy_breach_problem', 'spy_breach_records',
           'lock_dir', 'pinned_code_problems', 'planned', 'record_dir', 'root', 'rows', 'short', 'split', 'unit_name', 'unit_prefix', 'work_dir']
