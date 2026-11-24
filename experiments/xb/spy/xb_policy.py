"""XB's spy policy (runs/20261107-xb/任务书.md item 1 (e)): which files a process of the external baseline runner may open.  The standard library only: the method processes import it in their own
conda environments (Python 3.10 or 3.11) through xb_spyrun.py, the driver in sci310 through xb_spy.py.

A path is classified, in this order (first match wins):
  1. allowed_exact      files named one by one (the driver: the pair's manifest.json, its predicted maps' files; nobody: the original pair.npz, which only the driver's reader may open while
                        it is reading, xb_spy.ORIGINAL)
  1a. read_exact        (optional key; LP-G revision 1, decision_log 2026-11-21) files named one by one for a READ only: the LPBA driver names its inputs and the archive/ files that
                        check_frozen and check_environment open here (xb_drive.policies; its allowed_exact stays empty); a write to one of them goes on to the rules below (archive/ and the
                        results root are forbidden roots there).  Absent in every other policy (the hold-out's, the development's, the method processes'): the order 1-6 as before.
  2. forbidden_names    outside the code and system roots (name_exempt_roots): a basename exactly pair.npz; a basename holding seg4, seg35, aligned_seg or truth; a .json whose name holds metrics,
                        evaluation, review, third_certificate, labels or smoothness (an evaluation record, as experiments/he/spies.py EVALUATION_RECORDS)
  3. allowed_inside     directories inside a forbidden root that may be used (the run's working directory inside the results root)
  3a. code_roots        (optional key; XF, 2026-11-15) a READ inside the running code tree, unless a forbidden root at or below that code root covers the path -- xb_spy lists the code
                        tree's own data/ among the forbidden roots (xb_spy.code_data_root; XF revision 2), so a data/ that is a real directory is refused here too, and a data/ link resolves out of
                        the tree (seq_vp_3d/data, itself forbidden):
                        the code tree is read wherever it lies -- the hold-out's detached checkout is <results>/holdout_B_v1/code (C22.H0-4) and the P0 checkouts are <results>/p0_code_<sha>,
                        both inside the results root, a forbidden root that contains the code root; a forbidden root containing the code root does not take the code tree's reads away.
                        A write is never allowed here: it goes on to 4 (forbidden under the results root) or 6 (unlisted elsewhere), as before.
  4. forbidden_roots    the stages (data/three_d through the link and its target), the subjects' data (/mnt/d/data), the seq_vp_3d tree, the results root, the hold-out stage and Atropos roots
  5. read_roots / write_roots   the code and system roots, the environment, the baselines, the working directory, the caches
  6. otherwise          unlisted
forbidden and unlisted are both breaches: the process stops (越界就停).  Paths are made absolute against the process's working directory and resolved (symbolic links followed), so a link into a
forbidden root is seen as that root (a data/ link in the code tree resolves out of it: 3a does not apply).  What a policy cannot see: opens from C or C++ code that do not go through Python (ITK,
TensorFlow, CUDA, the CLI tools): xb_spy.FdSampler samples those.
Why 3a (XF, runs/20261115-d1/实现说明_XF.md): D1 on the P0 checkout <results>/p0_code_af593248 stopped at XB's dry run -- the driver's lazy import of bayesreg.io.code_identity under the spy
(CPython's import machinery opens __pycache__/*.pyc with io.open_code, audit event 'open', even under -B) was judged forbidden by 4 before the code root of 5 was reached.  With the code tree
outside the results root (the main checkout, every test) 3a changes nothing: a read there was already allowed by 5.
"""
import json
import os

FORBIDDEN_BASENAMES = ('pair.npz',)
FORBIDDEN_TOKENS = ('seg4', 'seg35', 'aligned_seg', 'truth')
EVALUATION_TOKENS = ('metrics', 'evaluation', 'review', 'third_certificate', 'labels', 'smoothness')
WRITE_MODE_CHARS = 'wax+'
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
KEYS = ('allowed_exact', 'allowed_inside', 'forbidden_roots', 'read_roots', 'write_roots', 'name_exempt_roots')
OPTIONAL_KEYS = ('code_roots', 'read_exact')    # XF: absent in a policy written before it (read as no code root: the order 1-6 without 3a); LP-G revision 1: read_exact (rule 1a), absent: none


def under(path, root):
    return path == root or path.startswith(root.rstrip('/')+'/')


def norm(path, cwd=None):
    """Absolute and resolved (symbolic links followed); None when it is not a path (a file descriptor)."""
    if isinstance(path, int):
        return None
    try:
        text = os.fsdecode(path)
    except TypeError:
        return None
    if not os.path.isabs(text):
        text = os.path.join(cwd or os.getcwd(), text)
    return os.path.realpath(text)


def is_write(mode, flags):
    if isinstance(mode, str):
        return any(c in mode for c in WRITE_MODE_CHARS)
    return bool((flags or 0) & WRITE_FLAGS)


def name_breach(path):
    base = os.path.basename(path)
    low = base.lower()
    if base in FORBIDDEN_BASENAMES:
        return f'the basename {base!r} (a pair archive)'
    for token in FORBIDDEN_TOKENS:
        if token in low:
            return f'the basename {base!r} holds {token!r} (a label or truth file)'
    if low.endswith('.json') and any(t in low for t in EVALUATION_TOKENS):
        return f'the basename {base!r} is an evaluation record'
    return None


class Policy:
    def __init__(self, document):
        missing = [k for k in KEYS if k not in document]
        if missing:
            raise ValueError(f'policy: missing keys {missing}')
        self.document = {k: list(document[k]) for k in KEYS}
        self.document.update({k: list(document[k]) for k in OPTIONAL_KEYS if k in document})
        self.document['name'] = document.get('name', 'unnamed')
        for k in KEYS:
            setattr(self, k, tuple(os.path.realpath(p) for p in document[k]))
        for k in OPTIONAL_KEYS:
            setattr(self, k, tuple(os.path.realpath(p) for p in document.get(k, ())))

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as stream:
            return cls(json.load(stream))

    def in_code_tree(self, path):
        """Rule 3a: `path` (resolved) lies in a code root and no forbidden root at or below that code root covers it (a forbidden root above it -- the results root -- does not count)."""
        return any(under(path, c) and not any(under(path, f) and under(f, c) for f in self.forbidden_roots) for c in self.code_roots)

    def classify(self, path, write=False):
        """('allowed' | 'forbidden' | 'unlisted', why) of a resolved absolute path."""
        if path in self.allowed_exact:
            return 'allowed', 'named file'
        if not write and path in self.read_exact:
            return 'allowed', 'named file (read only)'
        if not any(under(path, r) for r in self.name_exempt_roots):
            why = name_breach(path)
            if why:
                return 'forbidden', why
        if any(under(path, r) for r in self.allowed_inside):
            return 'allowed', 'allowed directory inside a forbidden root'
        if not write and self.in_code_tree(path):
            return 'allowed', 'a read of the running code tree'
        for r in self.forbidden_roots:
            if under(path, r):
                return 'forbidden', f'under the forbidden root {r}'
        roots = self.write_roots if write else self.read_roots+self.write_roots
        if any(under(path, r) for r in roots):
            return 'allowed', 'write root' if write else 'read root'
        return 'unlisted', f"not under any {'write' if write else 'read'} root"
