"""PQH's firewall (contract C22.H4-3, revision 7, with 7a-7d; C23.6-5 with the 6f correction): the rules of the two hold-out PQ units, on the development package's spy (pq_firewall.Spy, unchanged).

The rule (C22.H4-3): a file may be opened if and only if it is one of the allowed items AND not on the forbidden list; on a conflict the forbidden list wins; a file of the watched scope outside the allowed
items stops the unit at once (pq_firewall.Breach, raised inside the open: Hm6).  Every open is recorded (opened_files.json, described, not judged by the code; the lead reads it before the release).
Watched scope (revision 7a, m15): /mnt/d whole, /home/rmcfz whole but the code (a path whose two forms are .py, or whose two forms lie in a __pycache__, and lie in one of the code trees -- the
detached checkout, this package's tree, bayesreg's tree, the frozen tree: review m8, C22.H4-3 "分离检出里不在监视范围内的只是代码"), the Python environment (/home/rmcfz/miniconda3) and the
runtime caches (revision 7b n11 and 7c r9: pqh_pins.RUNTIME_CACHE_DIRS, provisional until the smoke's segment (i)).  The detached checkout's configs/, splits/, docs/, runs/ are watched (only .py is code).
Allowed (default deny): the files the unit names one by one (the tables of C22.H4-1 and C22.H5-3 for this mode), the four sealed lists by path (revision 7c-7d: every read recorded, any number), the unit's
own output directories, its own locks and refusal records, and its authorizations by their name pattern.  After the finalization record and the move records are read, the out unit adds the method's
files they name (allow_files: each addition is recorded with its reason; the method's own records are input, not evaluation records, C22.H4-3 last item).
Forbidden (at least, C22.H4-3): every label file (aligned_seg35.nii.gz, aligned_seg4.nii.gz, any .nii.gz with seg in its name); pair.npz members other than fixed, moving, initial_u (the spy's member watch);
the evaluation outputs (C23.12's directory and locks, evaluation*.json, metrics_*.json, repair_metrics.json, identity_eval*, cert_eval*); output_superseded_*, output_failed_* (not input); the development
runs and readings (p3_v1/, pq_v1/, the development predictions, *_dev_read.lock) in the hold-out mode; result documents and reviews under docs/ and runs/; analysis products (decision*, summary*, per_pair*).
"""
import os
from pathlib import Path
import re

import pq_firewall as fw

LABEL_FILE_NAMES = fw.LABEL_FILE_NAMES
EVALUATION_NAMES = ('evaluation.json', 'evaluation_started.json', 'repair_metrics.json')
EVALUATION_PREFIXES = ('evaluation_not_written', 'identity_eval', 'cert_eval', 'metrics_')
EVALUATION_DIR_PREFIXES = ('identity_eval', 'cert_eval')          # their outputs may be directories (C22.H4-3 "cert_eval 的输出")
WITHDRAWN_PREFIXES = ('output_superseded_', 'output_failed_')


def _forms(path):
    try:
        raw = Path(os.path.abspath(os.fsdecode(path)))
        return raw, raw.resolve()
    except (TypeError, ValueError, OSError):
        return None, None


def _both(paths):
    out = set()
    for p in paths:
        raw, resolved = _forms(p)
        if raw is not None:
            out |= {raw, resolved}
    return tuple(sorted(out, key=str))


def _under(path, roots):
    return any(path == root or path.is_relative_to(root) for root in roots)


class HoldoutRules:
    """C22.H4-3's rules for one unit in one mode.  watched / unwatched: roots (each kept in both forms); files: allowed files (resolved); dirs: allowed directories (resolved; the unit's own outputs);
    patterns: (directory, regex) pairs: a file directly in the directory whose name matches (the unit's authorizations); forbidden_dirs: directories whose every file is forbidden (both forms)."""

    def __init__(self, *, unit, mode, watched, unwatched=(), files=(), dirs=(), patterns=(), forbidden_dirs=(), result_roots=(), pinned_documents=(), code_trees=()):
        r = lambda p: Path(p).resolve()     # noqa: E731
        self.unit, self.mode = unit, mode
        self.code_trees = _both(code_trees)                    # m8: .py and __pycache__ are code only inside these trees
        self.watched = _both(watched)
        self.unwatched = _both(unwatched)
        self.files = set(r(p) for p in files)
        self.dirs = tuple(sorted({r(d) for d in dirs}, key=str))
        self.patterns = tuple((r(d), re.compile(p)) for d, p in patterns)
        self.forbidden_dirs = _both(forbidden_dirs)
        self.result_roots = _both(result_roots)                # docs/ and runs/ of the code trees: result documents and reviews there are forbidden
        self.pinned_documents = frozenset(r(p) for p in pinned_documents)      # plan.md, the preregistration, the decision log (through load_unseal) and this package's replay plan
        self.added = []

    def allow_files(self, paths, reason):
        """Allow more files (resolved), recorded with the reason (the method's files named by the finalization record, the HI predictions named by HI's records)."""
        new = sorted({Path(p).resolve() for p in paths} - self.files, key=str)
        self.files |= set(new)
        if new:
            self.added.append(dict(reason=reason, n=len(new), files=[str(p) for p in new]))
        return new

    def _watched_form(self, form):
        return _under(form, self.watched) and not _under(form, self.unwatched)

    def in_scope(self, path, resolved=None):
        """Watched when either form lies under a watched root and not under an unwatched one; code (both forms .py, or both in a __pycache__, and both forms inside a code tree) is out of scope."""
        if path is None:
            return False
        resolved = path.resolve() if resolved is None else resolved
        code = _under(path, self.code_trees) and _under(resolved, self.code_trees)
        if code and path.suffix == '.py' and resolved.suffix == '.py':
            return False
        if code and '__pycache__' in path.parts and '__pycache__' in resolved.parts:
            return False
        return self._watched_form(path) or self._watched_form(resolved)

    def forbidden(self, path):
        name, parts = path.name, set(path.parts)
        if name in LABEL_FILE_NAMES or (name.endswith('.nii.gz') and 'seg' in name):
            return f'a label file ({name})'
        if name in EVALUATION_NAMES or name.startswith(EVALUATION_PREFIXES) or any(part.startswith(EVALUATION_DIR_PREFIXES) for part in path.parts[:-1]):
            return f'an evaluation record ({name})'
        if any(part.startswith(WITHDRAWN_PREFIXES) for part in path.parts):
            return 'a withdrawn output (output_superseded_*, output_failed_*: not an input, C22.H4-1)'
        if _under(path, self.forbidden_dirs):
            return 'a directory this unit never reads (the evaluation entry, another reading or the development runs)'
        if name.endswith('_dev_read.lock'):
            return f'a development read lock ({name})'
        if 'analysis' in parts or name.startswith(('decision', 'summary', 'per_pair')) or name == 'long.csv':
            if path.resolve() not in self.pinned_documents:
                return f'an analysis product ({name})'
        if _under(path, self.result_roots) and path.resolve() not in self.pinned_documents:
            if name.startswith('result') or 'reviews' in parts:
                return f'a result document or a review ({name})'
        return None

    def allowed(self, resolved):
        if resolved in self.files or any(resolved.is_relative_to(d) for d in self.dirs):
            return True
        return any(resolved.parent == d and rx.fullmatch(resolved.name) for d, rx in self.patterns)

    def judge(self, path, resolved=None):
        resolved = path.resolve() if resolved is None else resolved
        if not self.in_scope(path, resolved):
            return None
        for form in (path, resolved):
            reason = self.forbidden(form)
            if reason is not None:
                return f'forbidden: {reason}'
        if not self.allowed(resolved):
            return 'outside the allowed items of C22.H4-3 (default deny)'
        return None

    def describe(self):
        return dict(unit=self.unit, mode=self.mode, watched=[str(p) for p in self.watched], unwatched=[str(p) for p in self.unwatched], code_trees=[str(p) for p in self.code_trees],
                    allowed_dirs=[str(d) for d in self.dirs],
                    allowed_patterns=[[str(d), rx.pattern] for d, rx in self.patterns], n_allowed_files=len(self.files), forbidden_dirs=[str(p) for p in self.forbidden_dirs], added=list(self.added),
                    rule='C22.H4-3: a watched file may be opened iff it is allowed and not forbidden; forbidden wins; default deny')


def sealed_list_files(code_tree):
    """The four sealed lists by path (C22.H4-3 revision 7c-7d, C23.6-5): bayesreg.io.sealed.SEALED_LISTS under bayesreg's code tree and under `code_tree` (the same in production: the detached checkout)."""
    from bayesreg.io import sealed
    return sorted({(Path(root)/spec['path']).resolve() for root in (sealed.REPO, code_tree) for spec in sealed.SEALED_LISTS})


def sealed_cache_problem(unit):
    """C23.6-5 (and the 6f correction): the guard's cache of the sealed lists must be empty when the spy is installed, so that the first read happens under it.  None, or the reason to stop (Hm6, C22.H4-3 item 2)."""
    from bayesreg.io.sealed import sealed_cache_is_empty
    if sealed_cache_is_empty():
        return None
    return f'{unit}: the sealed list cache was already populated before the spy was installed (C23.6-5): the guard would read the lists unseen; a unit runs in a process of its own'


def spy(rules):
    """The development package's spy with these rules (pq_firewall.Spy: builtins.open, io.open, os.open, pathlib's opener; NpzFile and ZipFile for the pair.npz members)."""
    return fw.Spy(rules)


Breach = fw.Breach
