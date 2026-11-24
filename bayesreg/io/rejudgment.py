"""The rejudgment records of the B-line hold-out: one format, one reader (prereg revision 2 item 18; decision_log 2026-11-10 ~2:28 AM CST NI2, NI3, ~2:45 AM CST N1, N2; A5 brief supplement 3;
work package A5b item 6).  The core moved here from experiments/holdout/holdout_common.rejudgment, as bayesreg/io/code_identity.py was (bayesreg imports nothing of experiments/): the evaluation entry
(bayesreg.eval.holdout) calls `rejudgment` directly, holdout_common.rejudgment only wraps it (after A5a's merge), and the XB chain (A5a) and HB's configuration C queue keep calling holdout_common's.

A rejudgment record judges the needs_lead_rejudgment of an external arm's or the configuration C arm's cell (N1: the method arm takes none; the review unit's are a trace only):
  runs/holdout_B_v1/rejudgments/<arm>/<pair>_<k>.json in the code tree, k from 1, added only; the highest k counts.
It holds when: committed and unchanged (tracked_unchanged: `git ls-files --error-unmatch` and an empty `git status --porcelain --ignored --untracked-files=all` of that path, hi_core.tracked_unchanged's
rule); strict JSON with exactly REJUDGMENT_KEYS (no resume key: a rerun needs holdout_common's resume record); schema REJUDGMENT_SCHEMA; arm, pair, pair_set the cell's; status a non-empty string;
record {path, sha256}, path the given record (the terminal or stop record it judges) and sha256 that file's now (NI3); evidence a list of {path, sha256}; judgment one of JUDGMENTS; reason,
time_local, utc non-empty; written_by 'lead' or BZ's for infrastructure and unit_limit, BZ's ('BZ', "lead on BZ's behalf") for no_output and for a status hx_provenance_rejected (P20), BZ's words
non-empty then.  `by_seq` orders records by their sequence number `seq` (HB diff review n4: never mtime or names), for the copies of refusal records a P20 no_resume names (A5a); holdout_common's
and hb_configc's refusals_by_seq wrap it (A5b review minor 6).
BZ's decision in a no_resume (or resume) record's reason, one criterion for HB, holdout_common and the evaluation entry (A5b review I2, the lead's ruling 11-10 ~9:22 PM CST): a P20 cell's names
P20 (p20_reason_problem, prereg revision 2 item 17 (f)); an hx_implementation_error's names (甲) or (乙) and exactly one case of C22.5-6 (decision_reason_problem, runs/20261109-align item 3).
git is read only; a git failure is a problem, never an exception.
"""
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType

from .code_identity import git_runner, strict_pairs

REJUDGMENT_DIR_REL = 'runs/holdout_B_v1/rejudgments'          # <arm>/<pair>_<k>.json, k from 1, added only (a commit after P1 may add files under runs/holdout_B_v1/)
LPBA40_REJUDGMENT_DIR_REL = 'runs/lpba40_test_v1/rejudgments'  # LP-C2 (C26.8-4): the LPBA40 second test set's, the same format and reader (pair_set 'lpba40_test')
# LP-C2 (C26.8-4): the directory by the pair set; every other pair set (the hold-out's 'holdout', the default, and XB's development 'dev') keeps REJUDGMENT_DIR_REL as before.
# The LPBA40 feasibility mode has no rejudgment records (C26.6-2: no authorization; a needs_lead_rejudgment there is a stop, C26.8-5): None.
REJUDGMENT_DIRS = MappingProxyType({'lpba40_test': LPBA40_REJUDGMENT_DIR_REL, 'lpba40_feas': None})
REJUDGMENT_SCHEMA = 'bayesreg.holdout.rejudgment.v1'
REJUDGMENT_KEYS = ('schema', 'arm', 'pair', 'pair_set', 'status', 'record', 'evidence', 'judgment', 'reason', 'written_by', 'words', 'time_local', 'utc')     # no resume key (N1)
JUDGMENTS = ('infrastructure', 'unit_limit', 'no_output')
BZ_WRITERS = ('BZ', "lead on BZ's behalf")                        # holdout_common.RESUME_WRITERS (a test compares)
LEAD_WRITERS = ('lead',)+BZ_WRITERS                               # infrastructure and unit_limit: the lead judges from the evidence; no_output and every P20 cell: BZ's
REJUDGMENT_NAME = re.compile(r'(?P<pair>.+)_(?P<k>[1-9][0-9]{0,2})\.json', re.ASCII)
SAFE = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,200}', re.ASCII)   # holdout_common.SAFE (a test compares)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_unchanged(path, repo, *, run=None):
    """(record, problem): the file is inside the code tree `repo`, tracked by git and without any change against HEAD (not untracked, ignored, only staged or modified); hi_core.tracked_unchanged's
    rule with literal pathspecs.  Nothing of the file itself is read.  record: path (relative), tracked, unchanged, last_commit; problem None or why not."""
    root = Path(repo).resolve()
    target = Path(path).resolve()
    if not target.is_relative_to(root):
        return None, f'{path} is not inside the code tree {root}'
    rel = target.relative_to(root).as_posix()
    git, _ = git_runner(root, run)
    rc_t, out_t, err_t = git('ls-files', '--error-unmatch', '--', rel)
    rc_s, out_s, err_s = git('status', '--porcelain', '--ignored', '--untracked-files=all', '--', rel)
    if rc_t != 0 or rc_s != 0 or out_s.strip() or out_t.strip() != rel:
        return None, f'{rel}: not a committed, unchanged file of the code tree (ls-files exit {rc_t} {out_t.strip()!r} {err_t}; status exit {rc_s} {out_s.strip()!r} {err_s})'
    rc_l, out_l, _ = git('log', '-1', '--format=%H', '--', rel)
    return dict(path=rel, tracked=True, unchanged=True, last_commit=(out_l.strip() or None) if rc_l == 0 else None), None


def rejudgment_dir(pair_set):
    """The code-tree directory of a pair set's rejudgment records (REJUDGMENT_DIRS; REJUDGMENT_DIR_REL for every other pair set, unchanged); None: the pair set has none."""
    return REJUDGMENT_DIRS[pair_set] if pair_set in REJUDGMENT_DIRS else REJUDGMENT_DIR_REL


def rejudgment(arm, pair, *, record, pair_set='holdout', root, run=None, directory_rel=None):
    """(found, problems): the cell's rejudgment record (module docstring), the highest k of <root>/runs/holdout_B_v1/rejudgments/<arm>/<pair>_<k>.json (root: the code tree).  found: dict(path,
    sha256, k, document (a mapping proxy), git) or None (none there: problems []; the highest does not hold: problems say why).  Never raises.
    The directory (LP-C2, contract C26.8-4): by the pair set (rejudgment_dir: 'lpba40_test' -> runs/lpba40_test_v1/rejudgments; 'lpba40_feas' has none, (None, [why]);
    every other pair set REJUDGMENT_DIR_REL, as before); `directory_rel` given overrides it."""
    if not (isinstance(arm, str) and SAFE.fullmatch(arm)):
        return None, [f'arm {arm!r} is not a plain name']
    base = Path(root)
    rel = rejudgment_dir(pair_set) if directory_rel is None else directory_rel
    if rel is None:
        return None, [f'pair set {pair_set!r} has no rejudgment records (C26.6-2, C26.8-5: the LPBA40 feasibility mode stops instead)']
    directory = base/rel/arm
    ks = []
    try:
        for q in (directory.iterdir() if directory.is_dir() else ()):
            m = REJUDGMENT_NAME.fullmatch(q.name)
            if m and m['pair'] == pair:
                ks.append((int(m['k']), q))
    except OSError as exc:
        return None, [f'{directory}: not listed ({exc})']
    if not ks:
        return None, []
    k, path = max(ks)
    git, problem = tracked_unchanged(path, base, run=run)
    if problem:
        return None, [f'{path}: not committed and unchanged ({problem})']
    try:
        raw = path.read_bytes()
        document = json.loads(raw.decode('utf-8'), object_pairs_hook=strict_pairs)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return None, [f'{path}: not strict JSON ({exc})']
    if not isinstance(document, dict) or sorted(document) != sorted(REJUDGMENT_KEYS):
        return None, [f'{path}: exactly the keys {list(REJUDGMENT_KEYS)} are required']
    problems = []
    if document['schema'] != REJUDGMENT_SCHEMA:
        problems.append(f"schema {document['schema']!r}")
    if (document['arm'], document['pair'], document['pair_set']) != (arm, pair, pair_set):
        problems.append(f"arm, pair, pair_set {(document['arm'], document['pair'], document['pair_set'])} are not {(arm, pair, pair_set)}")
    if not isinstance(document['status'], str) or not document['status']:
        problems.append('status: a non-empty string')
    rec = document['record']
    if not (isinstance(rec, dict) and set(rec) == {'path', 'sha256'}) or Path(str(rec.get('path'))) != Path(record):
        problems.append(f'record {rec!r}: {{path, sha256}} of {record}')
    else:
        try:
            now = sha256_file(record) if Path(record).is_file() else None
        except OSError:
            now = None
        if now is None or now != rec['sha256']:
            problems.append(f'record: {record} is not there with sha256 {rec["sha256"]}')
    if not (isinstance(document['evidence'], list) and all(isinstance(e, dict) and set(e) == {'path', 'sha256'} for e in document['evidence'])):
        problems.append('evidence: a list of {path, sha256}')
    if document['judgment'] not in JUDGMENTS:
        problems.append(f"judgment {document['judgment']!r} is not one of {JUDGMENTS}")
    problems += [f'{key}: non-empty' for key in ('reason', 'time_local', 'utc') if not isinstance(document[key], str) or not document[key].strip()]
    bz_only = document['judgment'] == 'no_output' or document['status'] == 'hx_provenance_rejected'
    if document['written_by'] not in (BZ_WRITERS if bz_only else LEAD_WRITERS):
        problems.append(f"written_by {document['written_by']!r}: {'BZ' if bz_only else 'the lead or BZ'} for judgment {document['judgment']!r}, status {document['status']!r}")
    if not isinstance(document['words'], str) or (bz_only and not document['words'].strip()):
        problems.append("words: a string; BZ's own words (non-empty) for no_output and a P20 cell")
    if problems:
        return None, [f'{path}: {x}' for x in problems]
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), k=k, document=MappingProxyType(dict(document)), git=git), []


# ---- BZ's decision in a record's reason: one criterion (A5b review I2) -----------------------------------------------------------------------------------
P20_WORD = 'P20'
CHOICE_WORDS = {'(甲)': ('(甲)', '（甲）'), '(乙)': ('(乙)', '（乙）')}       # C22.5-6: (甲) no output, a no_resume record; (乙) a rerun, a resume record (half- or full-width parentheses)
C22_5_6_CASES = ('C22.5-6 (a)', 'C22.5-6 (b)')


def p20_reason_problem(reason):
    """Prereg revision 2 item 17 (f): BZ's no_resume record of a P20 cell (refused, again, before its lock or by HE's provenance check) names P20 in its reason.  None or why not."""
    if P20_WORD in str(reason or ''):
        return None
    return f'the reason {reason!r} does not name {P20_WORD} (prereg revision 2 item 17 (f): BZ\'s decision on a P20 cell names it)'


def decision_reason_problem(reason, kind, which):
    """runs/20261109-align item 3, HB review m9 (ii): BZ's decision on an hx_implementation_error names in its reason `which` ('(甲)': no output, a no_resume record; '(乙)': a rerun, a resume
    record; half- or full-width parentheses) and not the other, and exactly one case of C22.5-6 -- the cell's own when `kind` ('a' or 'b') is given; kind None: whichever one it names (the cell's
    case not known to the caller).  None or why not."""
    reason = str(reason or '')
    if which not in CHOICE_WORDS:
        return f'{which!r} is not (甲) or (乙)'
    other = '(乙)' if which == '(甲)' else '(甲)'
    cases = [c for c in C22_5_6_CASES if c in reason]
    named, crossed = any(w in reason for w in CHOICE_WORDS[which]), any(w in reason for w in CHOICE_WORDS[other])
    if named and not crossed and len(cases) == 1 and (kind is None or cases[0] == f'C22.5-6 ({kind})'):
        return None
    case = f'C22.5-6 ({kind})' if kind is not None else 'one case of C22.5-6'
    return (f'the reason {reason!r} must name {case} and {which}, and neither {other} nor another case (runs/20261109-align item 3, HB review m9; one criterion for HB and the '
            'evaluation entry, A5b review I2)')


def record_seq(path):
    """The sequence number `seq` of a record (an int >= 1), or None (none, not a number, the file unreadable)."""
    try:
        doc = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    seq = doc.get('seq') if isinstance(doc, dict) else None
    return seq if isinstance(seq, int) and not isinstance(seq, bool) and seq >= 1 else None


def by_seq(paths, seq_of=record_seq):
    """([(seq, path)] oldest first, problem or None): by the records' sequence numbers (`seq_of(path)`, default record_seq); a problem when one has none or two share one (the newest is then not
    known: to the lead).  HB diff review n4; A5a's holdout_common.refusals_by_seq."""
    rows = [(seq_of(q), Path(q)) for q in paths]
    bad = [str(q) for n, q in rows if n is None]
    numbers = [n for n, _ in rows if n is not None]
    twice = sorted({n for n in numbers if numbers.count(n) > 1})
    if bad or twice:
        return [], f'records without a sequence number {bad} or sharing one {twice}: the newest is not known; to the lead'
    return sorted(rows, key=lambda r: r[0]), None


__all__ = ['BZ_WRITERS', 'C22_5_6_CASES', 'CHOICE_WORDS', 'JUDGMENTS', 'LEAD_WRITERS', 'LPBA40_REJUDGMENT_DIR_REL', 'P20_WORD', 'REJUDGMENT_DIRS', 'REJUDGMENT_DIR_REL', 'REJUDGMENT_KEYS', 'REJUDGMENT_NAME', 'REJUDGMENT_SCHEMA', 'SAFE',
           'by_seq', 'decision_reason_problem', 'p20_reason_problem', 'record_seq', 'rejudgment', 'rejudgment_dir', 'sha256_file', 'tracked_unchanged']
