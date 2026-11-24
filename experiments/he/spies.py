"""HE's spies (contract C22.5-2 gate (2) and the C22 implementation gate "防火墙"): installed in a process that runs the method, they record what it opens and show that they saw it.

Three ways into the ORIGINAL pair.npz are watched (revision 1, review important 1):
  np.load            returns a guard: a member other than fixed, moving, initial_u, the member list (.files and any other attribute), iteration -> recorded, then Breach;
  NpzFile.__getitem__  (the class, whoever built the archive): a member outside the whitelist -> recorded, then Breach; every read is counted;
  zipfile.ZipFile.open (and so ZipFile.read): a member outside the whitelist -> recorded, then Breach.
Every other archive passes untouched (the derived pair of fallback one has zero label members by construction; the frozen worker reads only that file, in its own process).
builtins.open and io.open record every path opened; the evaluation records among them are the JSON files whose names hold one of EVALUATION_RECORDS (an evaluation's metrics, review and third certificate, the
frozen worker's labels.json and smoothness.json).
The positive condition: the two loaders allowed to read the images (bayesreg.io.pair.load_pair and firewall_c.read_image_members) are counted when they are called on the original and return; the guard must have
seen exactly that many reads of fixed and of moving, through the np.load path and at the class, and at least one when reads are expected.  A spy that saw nothing is not a passing spy.
Used by the tests (tests/test_bayesreg/he/_pipeline.py, test_he_tools.py) and by spy_pair.py (a real pair, the lead's GPU smoke).  Nothing here changes a number.
"""
import builtins
import io
import pathlib
from pathlib import Path
import zipfile

import numpy as np

WHITELIST = ('fixed', 'moving', 'initial_u')
EVALUATION_RECORDS = ('metrics', 'evaluation', 'review', 'third_certificate', 'labels', 'smoothness')     # metrics_<step>.json, repair_metrics.json, evaluation.json, review_mismatch_<k>.json, labels.json, smoothness.json, ...


class Breach(AssertionError):
    """The method opened what the firewall forbids."""


class Guarded:
    """An NpzFile stand-in for the original pair.npz: only the whitelisted members, one key at a time."""

    def __init__(self, inner, log):
        self._inner, self._log = inner, log

    def _check(self, key):
        if key not in WHITELIST:
            self._log.append(f'member {key}')
            raise Breach(f'label firewall breach on the original pair.npz: {key!r}')

    def __getitem__(self, key):
        self._check(key)
        self._log.append(f'read {key}')
        return self._inner[key]

    def __contains__(self, key):
        self._check(key)
        return key in self._inner

    def __iter__(self):
        self._log.append('iteration')
        raise Breach('iteration over the original pair.npz')

    def __getattr__(self, name):
        self._log.append(f'attribute {name}')
        raise Breach(f'attribute {name!r} of the original pair.npz')

    def close(self):
        self._inner.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._inner.close()
        return False


def _member(name):
    name = getattr(name, 'filename', name)          # a ZipInfo or a name
    return name[:-4] if isinstance(name, str) and name.endswith('.npy') else name


class Spies:
    def __init__(self, original):
        self.original = Path(original).resolve()
        self.original_access, self.npz_access, self.zip_access, self.opened, self.read_opened, self.loader_calls = [], [], [], [], [], []
        self.real_load, self.real_open, self.real_io_open = np.load, builtins.open, io.open
        self.real_getitem, self.real_zip_open = np.lib.npyio.NpzFile.__getitem__, zipfile.ZipFile.open
        accessor = getattr(pathlib, '_NormalAccessor', None)
        self.real_path_open = None if accessor is None else accessor.__dict__.get('open')
        self.patched = []

    def is_original(self, path):
        try:
            return path is not None and Path(path).resolve() == self.original
        except TypeError:
            return False

    def install(self):
        spies = self

        def load(path, *args, **kwargs):
            inner = spies.real_load(path, *args, **kwargs)
            return Guarded(inner, spies.original_access) if spies.is_original(path) else inner

        def getitem(npz, key):
            if spies.is_original(getattr(getattr(npz, 'zip', None), 'filename', None)):
                if key not in WHITELIST:
                    spies.npz_access.append(f'member {key}')
                    raise Breach(f'label firewall breach on the original pair.npz (NpzFile): {key!r}')
                spies.npz_access.append(f'read {key}')
            return spies.real_getitem(npz, key)

        def zip_open(archive, name, *args, **kwargs):
            if spies.is_original(getattr(archive, 'filename', None)):
                member = _member(name)
                if member not in WHITELIST:
                    spies.zip_access.append(f'member {member}')
                    raise Breach(f'label firewall breach on the original pair.npz (zipfile): {member!r}')
                spies.zip_access.append(f'open {member}')
            return spies.real_zip_open(archive, name, *args, **kwargs)

        def opener(real):
            def wrapped(file, *args, **kwargs):
                if isinstance(file, (str, Path)):
                    spies.opened.append(str(file))
                    mode = args[0] if args else kwargs.get('mode', 'r')
                    if isinstance(mode, str) and not any(c in mode for c in 'wxa+'):
                        spies.read_opened.append(str(file))
                return real(file, *args, **kwargs)
            return wrapped
        np.load = load
        np.lib.npyio.NpzFile.__getitem__ = getitem
        zipfile.ZipFile.open = zip_open
        builtins.open = opener(self.real_open)
        io.open = opener(self.real_io_open)
        if self.real_path_open is not None:                             # Python 3.10: Path.open goes through pathlib._NormalAccessor.open, bound to io.open when pathlib was imported
            pathlib._NormalAccessor.open = staticmethod(opener(self.real_path_open))
        self._count('bayesreg.io.pair', 'load_pair', lambda args, kwargs: args[0] if args else kwargs.get('path'))
        self._count('firewall_c', 'read_image_members', lambda args, kwargs: args[0] if args else kwargs.get('path'))
        return self

    def _count(self, module_name, attr, path_of):
        """Count the calls of an image loader on the original that return."""
        import importlib
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            return
        real = getattr(module, attr)
        spies = self

        def counted(*args, **kwargs):
            out = real(*args, **kwargs)
            if spies.is_original(path_of(args, kwargs)):
                spies.loader_calls.append(f'{module_name}.{attr}')
            return out
        setattr(module, attr, counted)
        self.patched.append((module, attr, real))

    def remove(self):
        np.load, builtins.open, io.open = self.real_load, self.real_open, self.real_io_open
        np.lib.npyio.NpzFile.__getitem__, zipfile.ZipFile.open = self.real_getitem, self.real_zip_open
        if self.real_path_open is not None:
            pathlib._NormalAccessor.open = self.real_path_open
        for module, attr, real in reversed(self.patched):
            setattr(module, attr, real)
        self.patched = []

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.remove()
        return False

    def evaluation_records(self):
        """Evaluation records among the files opened for reading (a process may write its own record, e.g. the third certification its third_certificate.json; source files copied for the frozen run are code,
        not records)."""
        return [p for p in self.read_opened if Path(p).suffix == '.json' and any(w in Path(p).name for w in EVALUATION_RECORDS)]

    def record(self, *, expect_reads=True):
        """passed: nothing forbidden, no evaluation record opened, and the reads seen equal the loaders' calls (at least one when expect_reads; none otherwise)."""
        bad = [a for a in self.original_access if not (a.startswith('read ') and a[5:] in WHITELIST)]
        bad += [a for a in self.npz_access+self.zip_access if a.startswith('member ')]
        loads = len(self.loader_calls)
        seen = {k: self.original_access.count(f'read {k}') for k in ('fixed', 'moving')}
        at_class = {k: self.npz_access.count(f'read {k}') for k in ('fixed', 'moving')}
        positive = (loads >= 1 if expect_reads else loads == 0 and not self.original_access and not self.npz_access and not self.zip_access) and seen['fixed'] == seen['moving'] == at_class['fixed'] == at_class['moving'] == loads
        return dict(original=str(self.original), original_access=list(self.original_access), npz_access=list(self.npz_access), n_zip_opens=len(self.zip_access), loader_calls=list(self.loader_calls),
                    reads_seen=seen, reads_at_the_class=at_class, forbidden_access=bad, evaluation_records_opened=self.evaluation_records(), n_files_opened=len(self.opened), expect_reads=expect_reads,
                    positive_condition=positive, passed=positive and not bad and not self.evaluation_records(),
                    rule='passed = no forbidden access (np.load guard, NpzFile, zipfile), no evaluation record opened, and the reads of fixed and moving seen at the guard and at the class equal the calls of the two image loaders (>= 1 when reads are expected, 0 and no access at all otherwise)')
