"""PQ's firewall (contract C22.Q1-3, revision 5b: "禁读清单 + 允许的目录"; C22.Q7 防火墙; plan.md section 8 m6): a spy installed in the PQ unit and in the members unit.

The rule: a file in the watched scope may be opened if and only if it is in one of the allowed items AND not on the forbidden list; on a conflict the forbidden list wins.  A file of the watched scope
outside the allowed items, or on the forbidden list, stops the unit at once (Breach, raised inside the open).  Every file opened (in scope or not; read or written) is recorded with its mode, and every
member read of a pair.npz with its name; the record is written as opened_files.json, described, not judged by the code (the lead reads it before the truth is read).

Watched scope (C22.Q1-3): the results root; the data roots (the repository's data/ and what it links to, /mnt/d/data); the repository's configs/, splits/, runs/, docs/.  Code files (.py, of the repository
and of the frozen tree), the Python environment and system files are outside it.
Allowed: configs/bayesreg/ (the registry, the allowlist, the seg4 table, the structure grouping, ...); splits/bayesreg_v1/folds.json and train_subjects.txt; the selection.json of the four stage sources of
the allowlist; the 19 pairs' manifest.json and pair.npz (members fixed, moving, initial_u only, through load_pair); the 20 subjects' probs36.npz, structure_groups.npz, provenance.json of
seg/unet35_v4_ens/pred; p3_v1/runs/{C_first5, C_rest600}/<pair>/{run.json, field_0600.npz} and p3_v1/runs/{RC_c600, RC_rest600}/<pair>/{repair.json, repair.npz}; the members unit: the 20 subjects'
/mnt/d/data/<subject>/aligned_norm.nii.gz and the four member files (by path; not their directories); pq_v1/ (this package's own files) except pq_v1/eval/; runs/20261107-pq/plan.md; and (C22.Q1-3
revision 6d, C23.6-5 revision 6g) the sealed lists of contract C23.2, file by file at the paths bayesreg.io.sealed.load_sealed_lists reads (SEALED_LISTS under bayesreg's code tree
sealed.REPO, which in production is the unit's repository; revision 6h, review m6): splits/bayesreg_v1/holdout_B_subjects.txt and, under archive/seq_vp_3d/results/verification/3d_stage_HOLDOUT_v1/plan/, oasis_heldout_pairs_draft.tsv, h4_subjects_20261020.txt and
ibsr18_pairs_20261018.tsv (no directory).  The guard reads them itself (id lists, not a subject's files); every open is recorded as any other.  run_pq checks, before each spy is installed, that the
guard's cache of the lists is empty (bayesreg.io.sealed.sealed_cache_is_empty), so that its first read happens under the spy.
Forbidden (at least): a pair.npz member other than fixed, moving, initial_u; label files under the data roots (aligned_seg35.nii.gz, aligned_seg4.nii.gz, any .nii.gz with 'seg' in its name);
metrics_*.json, repair_metrics.json, evaluation.json; pq_v1/eval/ and any analysis/ directory, decision*, long.csv, per_pair*.csv, summary*; the read locks of seg/ (tta, uf, ul, nn: *_dev_read.lock) and the
outputs they protect (eval_uf/, eval_u1/, eval_seg/, control_check/, TTA's and nnU-Net's evaluations); result*.md and reviews/ under docs/ and runs/ (the pinned plan.md excepted).

Three ways into a pair.npz are watched, as experiments/he/spies.py does: np.load's NpzFile.__getitem__ (the class), zipfile.ZipFile.open (and so ZipFile.read); a member outside fixed, moving, initial_u
is recorded and stops the unit.  builtins.open, io.open, os.open and pathlib's opener record every path.  The spy only watches; it changes no number.
Symbolic links (revision 1, review minor 4): every opened path is judged in two forms, as written (made absolute, '..' folded, links kept) and resolved.  It is in the watched scope when either form is
(each root counts in both of its forms too: the repository's data/ is itself a link to seq_vp_3d/data); it is forbidden when either form is; it is allowed only when the resolved form is an allowed file or
lies in an allowed directory.  So a link inside the scope that leads outside it is stopped, a link named innocently that leads to a forbidden file is stopped, and data/ reached through its link is judged
as its target.  A .py file is out of scope only when both forms are .py.
"""
import builtins
import io
import os
import pathlib
from pathlib import Path
import zipfile

import numpy as np

PAIR_MEMBERS = ('fixed', 'moving', 'initial_u')
LABEL_FILE_NAMES = ('aligned_seg35.nii.gz', 'aligned_seg4.nii.gz')          # bayesreg/seg/unet/data.py:54 LABEL_FILES
EVALUATION_NAMES = ('repair_metrics.json', 'evaluation.json')
LOCK_PREFIXES = ('tta', 'uf', 'ul', 'nn')
PROTECTED_DIRS = ('eval_uf', 'eval_u1', 'eval_seg', 'control_check', 'eval_nn', 'tta_v4', 'analysis')
STAGES = ('stage_L2R_val_v1', 'stage_T0_tune_v1', 'stage_E11_native_v1', 'stage_LPBA40_v1')


class Breach(AssertionError):
    """The unit opened what C22.Q1-3 does not allow; it stops (m6 not passed)."""


def _resolve(path):
    try:
        return Path(os.fsdecode(path)).resolve()
    except (TypeError, ValueError, OSError):
        return None


def _forms(path):
    """(as written: absolute, '..' folded, links kept; resolved), or (None, None)."""
    try:
        raw = Path(os.path.abspath(os.fsdecode(path)))
        return raw, raw.resolve()
    except (TypeError, ValueError, OSError):
        return None, None


def _both(path):
    raw, resolved = _forms(path)
    return {raw, resolved}


def _under(path, roots):
    return any(path == root or path.is_relative_to(root) for root in roots)


class Rules:
    """The allowed items and the forbidden list of C22.Q1-3 for one unit.  roots: repo (the code tree), results, data (the repository's data/, a link in production) and images (/mnt/d/data); each root
    is kept in both forms (as written and resolved), the allowed files in their resolved form."""

    def __init__(self, *, repo, results, data, images, pairs, subjects, member_files=(), p3_dirs=None, unit='out', pq_root=None, pred_root=None, p3_root=None, extra_allowed=()):
        r = lambda p: Path(p).resolve()     # noqa: E731
        self.repo, self.results, self.data, self.images = r(repo), r(results), r(data), r(images)
        self.pq_root = r(pq_root) if pq_root is not None else self.results/'pq_v1'
        self.unit = unit
        self.scope = tuple(sorted(set().union(*(_both(p) for p in (results, data, images, Path(repo)/'configs', Path(repo)/'splits', Path(repo)/'runs', Path(repo)/'docs'))), key=str))
        self.data_roots = tuple(sorted(_both(data) | _both(images), key=str))
        self.repo_forms = tuple(sorted(_both(repo), key=str))
        self.results_forms = tuple(sorted(_both(results), key=str))
        self.pq_forms = tuple(sorted(_both(self.pq_root) | ({Path(os.path.abspath(pq_root))} if pq_root is not None else set()), key=str))
        stage = self.data/'three_d'/'stage_L2R_val_v1'
        files = {self.repo/'splits'/'bayesreg_v1'/'folds.json', self.repo/'splits'/'bayesreg_v1'/'train_subjects.txt', self.repo/'runs'/'20261107-pq'/'plan.md'}
        files |= {self.data/'three_d'/s/'selection.json' for s in STAGES}
        files |= {stage/p/name for p in pairs for name in ('manifest.json', 'pair.npz')}
        pred = r(pred_root) if pred_root is not None else self.results/'seg'/'unet35_v4_ens'/'pred'
        files |= {pred/s/name for s in subjects for name in ('probs36.npz', 'structure_groups.npz', 'provenance.json')}
        p3 = r(p3_root) if p3_root is not None else self.results/'p3_v1'/'runs'
        for short, (t88, rep) in (p3_dirs or {}).items():
            files |= {p3/t88/short/'run.json', p3/t88/short/'field_0600.npz', p3/rep/short/'repair.json', p3/rep/short/'repair.npz'}
        if unit == 'members':
            files |= {self.images/s/'aligned_norm.nii.gz' for s in subjects}
            files |= {r(p) for p in member_files}
        files |= {r(p) for p in extra_allowed}
        # C22.Q1-3 revision 6d, C23.6-5: the sealed lists are allowed reads (id lists, not a subject's files): the guard reads them itself (contract C23.2).
        # The paths are the ones the guard reads: bayesreg.io.sealed's pinned table (read at construction) under bayesreg's code tree sealed.REPO -- in production
        # the same tree as the unit's repository (run_pq.REPO); the branch under ctx.repo is gone (SG code review m6, contract C23 revision 6h).
        from bayesreg.io import sealed as sealed_lists
        self.sealed_lists = frozenset(r(Path(sealed_lists.REPO)/spec['path']) for spec in sealed_lists.SEALED_LISTS)
        files |= self.sealed_lists
        self.files = frozenset(files)
        self.dirs = (self.repo/'configs'/'bayesreg', self.pq_root)

    def in_scope(self, path, resolved=None):
        """path: as written (absolute); resolved: its resolved form (computed when None).  Watched when either form lies under a root (in either of the root's forms)."""
        if path is None:
            return False
        resolved = path.resolve() if resolved is None else resolved
        if path.suffix == '.py' and resolved.suffix == '.py':      # code files of the repository and of the frozen tree
            return False
        return _under(path, self.scope) or _under(resolved, self.scope)

    def forbidden(self, path):
        """The reason the forbidden list names for one form of a path, or None."""
        name = path.name
        parts = set(path.parts)
        if _under(path, self.data_roots):
            if name in LABEL_FILE_NAMES or (name.endswith('.nii.gz') and 'seg' in name):
                return f'a label file ({name})'
        if (name.startswith('metrics_') and name.endswith('.json')) or name in EVALUATION_NAMES:
            return f'an evaluation record ({name})'
        if _under(path, [p/'eval' for p in self.pq_forms]):
            return 'the evaluation unit\'s directory pq_v1/eval/'
        if 'analysis' in parts or name.startswith(('decision', 'summary', 'per_pair')) or name == 'long.csv':
            return f'an analysis product ({name})'
        if _under(path, [p/'seg' for p in self.results_forms]):
            if name.endswith('_dev_read.lock') and name.split('_', 1)[0] in LOCK_PREFIXES:
                return f'a development read lock ({name})'
            if parts & set(PROTECTED_DIRS):
                return 'an output a development read lock protects'
        if _under(path, [p/d for p in self.repo_forms for d in ('docs', 'runs')]) and path not in {p/'runs'/'20261107-pq'/'plan.md' for p in self.repo_forms}:
            if name.startswith('result') or 'reviews' in parts:
                return f'a result document or a review ({name})'
        return None

    def allowed(self, resolved):
        """The resolved form is an allowed file or lies in an allowed directory."""
        return resolved in self.files or any(resolved.is_relative_to(d) for d in self.dirs)

    def judge(self, path, resolved=None):
        """None when the open may go on; the reason otherwise (forbidden wins over allowed).  path: as written (absolute); resolved: its resolved form."""
        resolved = path.resolve() if resolved is None else resolved
        if not self.in_scope(path, resolved):
            return None
        for form in (path, resolved):
            reason = self.forbidden(form)
            if reason is not None:
                return f'forbidden: {reason}'
        if not self.allowed(resolved):
            return 'outside the allowed items of C22.Q1-3'
        return None

    def describe(self):
        return dict(scope=[str(s) for s in self.scope], allowed_dirs=[str(d) for d in self.dirs], n_allowed_files=len(self.files), unit=self.unit,
                    rule='a watched file may be opened iff it is allowed and not forbidden; forbidden wins (C22.Q1-3, revision 5b)')


def _member(name):
    name = getattr(name, 'filename', name)
    return name[:-4] if isinstance(name, str) and name.endswith('.npy') else name


class Spy:
    """Install with `with Spy(rules) as spy:`; spy.record() for opened_files.json."""

    def __init__(self, rules):
        self.rules = rules
        self.opened, self.members, self.breaches = [], [], []
        self.real_open, self.real_io_open, self.real_os_open = builtins.open, io.open, os.open
        self.real_getitem, self.real_zip_open = np.lib.npyio.NpzFile.__getitem__, zipfile.ZipFile.open
        accessor = getattr(pathlib, '_NormalAccessor', None)
        self.real_path_open = None if accessor is None else accessor.__dict__.get('open')
        self.installed = False

    def _see(self, file, mode):
        if isinstance(file, int):
            return
        raw, path = _forms(file)
        if path is None:
            return
        row = dict(path=str(path), mode=str(mode), in_scope=self.rules.in_scope(raw, path))
        if raw != path:
            row['as_opened'] = str(raw)
        self.opened.append(row)
        reason = self.rules.judge(raw, path)
        if reason is not None:
            self.breaches.append(dict(path=str(path), as_opened=str(raw), mode=str(mode), reason=reason))
            raise Breach(f'C22.Q1-3: {raw} ({path}): {reason}')

    def _is_pair(self, archive):
        filename = getattr(archive, 'filename', None)
        raw, path = _forms(filename) if filename is not None else (None, None)
        return path if path is not None and (path.name == 'pair.npz' or raw.name == 'pair.npz') and self.rules.in_scope(raw, path) else None

    def install(self):
        spy = self

        def opener(real, mode_index=1):
            def wrapped(file, *args, **kwargs):
                mode = args[0] if args else kwargs.get('mode', 'r')
                spy._see(file, mode)
                return real(file, *args, **kwargs)
            return wrapped

        def os_open(path, flags, *args, **kwargs):
            spy._see(path, f'os.open flags {flags}')
            return spy.real_os_open(path, flags, *args, **kwargs)

        def getitem(npz, key):
            path = spy._is_pair(getattr(npz, 'zip', None))
            if path is not None:
                spy.members.append(dict(path=str(path), member=str(key), via='NpzFile'))
                if key not in PAIR_MEMBERS:
                    spy.breaches.append(dict(path=str(path), member=str(key), reason='a pair.npz member other than fixed, moving, initial_u'))
                    raise Breach(f'label firewall: {path} member {key!r}')
            return spy.real_getitem(npz, key)

        def zip_open(archive, name, *args, **kwargs):
            path = spy._is_pair(archive)
            if path is not None:
                member = _member(name)
                spy.members.append(dict(path=str(path), member=str(member), via='zipfile'))
                if member not in PAIR_MEMBERS:
                    spy.breaches.append(dict(path=str(path), member=str(member), reason='a pair.npz member other than fixed, moving, initial_u'))
                    raise Breach(f'label firewall: {path} member {member!r}')
            return spy.real_zip_open(archive, name, *args, **kwargs)

        builtins.open = opener(self.real_open)
        io.open = opener(self.real_io_open)
        os.open = os_open
        np.lib.npyio.NpzFile.__getitem__ = getitem
        zipfile.ZipFile.open = zip_open
        if self.real_path_open is not None:                 # Python 3.10: Path.open goes through pathlib._NormalAccessor.open, bound to io.open when pathlib was imported
            pathlib._NormalAccessor.open = staticmethod(opener(self.real_path_open))
        self.installed = True
        return self

    def remove(self):
        if not self.installed:
            return
        builtins.open, io.open, os.open = self.real_open, self.real_io_open, self.real_os_open
        np.lib.npyio.NpzFile.__getitem__, zipfile.ZipFile.open = self.real_getitem, self.real_zip_open
        if self.real_path_open is not None:
            pathlib._NormalAccessor.open = self.real_path_open
        self.installed = False

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.remove()
        return False

    def record(self):
        """opened_files.json: every file opened (path, mode, in scope), the pair.npz members read, the breaches; and the per-pair.npz summary of members.  Described; m6 is 'no breach'."""
        seen = {}
        for row in self.opened:
            entry = seen.setdefault(row['path'], dict(path=row['path'], modes=[], in_scope=row['in_scope'], n_opens=0))
            entry['n_opens'] += 1
            entry['in_scope'] = entry['in_scope'] or row['in_scope']
            if row['mode'] not in entry['modes']:
                entry['modes'].append(row['mode'])
            if 'as_opened' in row and row['as_opened'] != entry.get('as_opened'):
                entry['as_opened'] = row['as_opened']                    # the form as written, when it differs from the resolved one (a link on the way)
        pairs = {}
        for row in self.members:
            pairs.setdefault(row['path'], set()).add(row['member'])
        return dict(files=sorted(seen.values(), key=lambda e: e['path']), pair_members={p: sorted(m) for p, m in sorted(pairs.items())}, breaches=list(self.breaches),
                    n_opens=len(self.opened), n_files=len(seen), n_in_scope=sum(1 for e in seen.values() if e['in_scope']), rules=self.rules.describe(),
                    m6=not self.breaches and all(set(m) <= set(PAIR_MEMBERS) for m in pairs.values()),
                    role='described (C22.Q1-3): the lead reads it before the evaluation unit may read any truth; m6 = no breach')
