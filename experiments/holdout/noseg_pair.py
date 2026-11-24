"""Prereg revision 2 item 9, checklist item 14 (revision 1a item 21; the lead's ruling 4 of 11-14, runs/20261114-p0/任务书.md, package P0S): a development pair with its label members taken away,
and the three checks on it, before any one-shot lock (the D1 group of the D segment).  No label member is opened: the hold-out stage holds no label member (prereg revision 1 item 1, C23.12-3), and
this tool makes a development pair of that shape from stage_L2R_val_v1 without decoding the label members of the original.

    python -B experiments/holdout/noseg_pair.py run --out NEW_DIR [--pair FIXED__MOVING]      (default pair: OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1)

Before any file of the pair is read, in this order (each refusal: exit 2 and a refusal record <results>/item14_noseg_v1/refusals/<utc>_<pid>[_k].json, nothing else written):
  a. the pair's name (<fixed>__<moving>; the default NEW_DIR is built from it);
  b. NEW_DIR a new directory under <results>/item14_noseg_v1/, judged on the resolved path (a symbolic link out of the root is refused; P0S revision 2, the code review's minor 2);
  c. the code: holdout_common.code_identity -- HEAD readable, the tracked files clean and nothing untracked under experiments, bayesreg, jd_curl, configs (as HP, XB and HI refuse a dirty tree;
     minor 3);
  d. admission: the pair is one of the 19 development pairs (holdout_common.admit_dev_pair: ids, the sealed lists, check_pair_subjects on the pinned allowlist -- C23.4-2).
The steps (each one recorded; a step that fails is a problem, the steps that need it are not run; nothing is retried):
  1. strip: bayesreg.io.pair.load_pair on the original (manifest through its whitelisting decoder, the guard, the container sha256, then fixed, moving, initial_u); the original's fixed, moving and
     initial_u (if there) read as stored, one key at a time through the loader's hook bayesreg.io.pair._open_npz; written to NEW_DIR/stage/<pair>/pair.npz (np.savez_compressed, in that order: the
     development stage's order without its label members, as the hold-out stage) and a manifest.json of exactly load_pair's whitelisted fields (bayesreg.io.pair.MANIFEST_FIELDS: the grid, the
     subjects, the two source sha256, pair_sha256 of the copy; minor 4); read back: exactly those members, each equal to the original's in dtype, shape and bytes.
  2. load_pair: load_pair on the copy reads the same keys as on the original (keys_read, initial_u_present) and the same arrays, grid, subjects and source sha256.
  3. firewall_c: experiments/he/firewall_c.derive_pair from the original's view and from the copy's (NEW_DIR/firewall/from_original, from_copy); firewall_check passes on both; the two derived
     pair.npz hold the same members (firewall_c.DERIVED_KEYS) equal in dtype, shape and bytes, and are the same file (sha256).
  4. external runner: XB's two readers of a pair, xb_core.check_pair_input (manifest, guard, container sha256) and xb_core.derive_inputs (fixed, moving -> the method's inputs.npz), run on the copy
     without an error (no sanity() any more: prereg §2.3 (b)), and derive_inputs gives the same arrays and the same file from the copy as from the original (NEW_DIR/xb/).
Through all of it the original is guarded twice (MemberGuard): the loader's hook (asking for any member but fixed, moving, initial_u, or listing or iterating the archive, raises; the members
asked are recorded) and numpy's NpzFile.__getitem__ for that file whoever opened it (a reader that bypasses the hook: only those three members are decoded, each recorded; minor 5).  A refusal
fails the run even when the reader swallowed it.
The record NEW_DIR/item14.json: the code (HEAD, clean), the admission, every step's record, the members asked of and decoded from the original, the refusals, problems, passed.  No lock (the D1
group comes before any one-shot lock); a run writes only NEW_DIR (and a refusal record).  Exit codes: 0 passed; 1 a check failed (the record says which); 2 refused before any file of the pair
was read (usage, the pair's name, the output directory, the code, the admission).
〔估算〕a few seconds of CPU and about 0.2 GB in NEW_DIR on a real pair (three compressed copies of fixed and moving, zeros).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
for _p in (HERE, EXPERIMENTS/'he', EXPERIMENTS/'xb'):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))
import holdout_common as hc  # noqa: E402

SCHEMA = 'bayesreg.holdout.item14.v1'
SCHEMA_REFUSAL = 'bayesreg.holdout.item14_refusal.v1'
ROOT_NAME = 'item14_noseg_v1'
DEFAULT_PAIR = hc.DEV_PAIRS[0]                  # OASIS_OAS1_0438_MR1__OASIS_OAS1_0439_MR1: the smoke's development pair
IMAGE_KEYS = ('fixed', 'moving')
ZERO_KEY = 'initial_u'
ALLOWED_KEYS = IMAGE_KEYS+(ZERO_KEY,)
PAIR_FILE, MANIFEST_FILE, RECORD = 'pair.npz', 'manifest.json', 'item14.json'
RULE = ('prereg revision 2 item 9, checklist item 14 (revision 1a item 21); the lead\'s ruling 4 of 2026-11-14: a development pair without its label members, made without opening them, and '
        'load_pair\'s keys, firewall_c\'s derived pair bit for bit, the external runner without missing-key errors, before any one-shot lock (the D1 group)')


class MemberRefused(AssertionError):
    """The original pair.npz was asked for a member outside ALLOWED_KEYS, or listed, or iterated."""


def root():
    """<results>/item14_noseg_v1, read at call time (the tests move the results root)."""
    return Path(hc.RESULTS)/ROOT_NAME


def in_real_results(path):
    """Whether the resolved `path` is the real results root or under it."""
    real = Path(hc.REAL_RESULTS).resolve()
    target = Path(path).resolve()
    return target == real or real in target.parents


def out_problem(out):
    """None when `out` is a new directory under root(), judged on the resolved paths (as xb_limit_batch2.out_problem; a symbolic link that leads out of the root is refused, and so is a name
    that is there at all, a dangling link included); in the test sandbox the real results root is refused too (resolved)."""
    target = Path(out).resolve()
    base = root().resolve()
    if os.environ.get(hc.TEST_SANDBOX_ENV) and in_real_results(out):
        return f'{out}: under the real results root {hc.REAL_RESULTS} in the test sandbox'
    if base not in target.parents:
        return f'{out}: a run writes into a new directory under {base} (resolved: {target})'
    if os.path.lexists(out) or target.exists():
        return f'{out} exists: a run writes into a new directory'
    return None


def write_refusal(stage, message, *, pair, out):
    """A refusal before any file of the pair was read: <root>/refusals/<utc>_<pid>[_k].json, a new file (never overwritten).  Returns its path."""
    directory = root()/'refusals'
    if os.environ.get(hc.TEST_SANDBOX_ENV) and in_real_results(directory):
        raise hc.RealRootInTest(f'test sandbox: {directory} is under the real results root')
    stem = f"{hc.utc().replace(':', '').replace('-', '')}_{os.getpid()}"
    path, k = directory/f'{stem}.json', 1
    while os.path.lexists(path):
        k += 1
        path = directory/f'{stem}_{k}.json'
    return hc.write_json_once(path, dict(schema=SCHEMA_REFUSAL, stage=stage, message=message, pair=str(pair), out=None if out is None else str(out), utc=hc.utc(),
                                         rule='refused before any file of the pair was read (the pair\'s name, the output directory, the code, the admission); nothing else written'))


class _GuardedNpz:
    def __init__(self, npz, path, log):
        self._npz, self._path, self._log = npz, str(path), log

    def __getitem__(self, key):
        if key not in ALLOWED_KEYS:
            self._log.append(dict(path=self._path, refused=str(key)))
            raise MemberRefused(f'{self._path}: member {key!r} asked (only {list(ALLOWED_KEYS)})')
        self._log.append(dict(path=self._path, member=key))
        return self._npz[key]

    def __contains__(self, key):                # a name in the archive's directory: nothing is decoded (load_pair, firewall_c ask for initial_u this way)
        return key in self._npz

    def _refuse(self, *args, **kwargs):
        raise MemberRefused(f'{self._path}: the archive listed or iterated')

    files = property(_refuse)
    keys = items = values = get = __iter__ = __len__ = _refuse

    def close(self):
        self._npz.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class MemberGuard:
    """While active, the original is guarded at two levels, both restored on exit:
      - bayesreg.io.pair._open_npz (the loader's hook, which load_pair, firewall_c.read_image_members and xb_core.read_image_members call) opens `original` behind a guard: only ALLOWED_KEYS may
        be asked for (each recorded in `log`), membership may be tested, the archive may not be listed or iterated; other paths open as before;
      - numpy's NpzFile.__getitem__ (P0S revision 2, the code review's minor 5): an archive of `original`, however it was opened (np.load directly included), decodes only ALLOWED_KEYS (each
        recorded in `decoded`); any other member raises before it is decoded."""

    def __init__(self, original):
        self.original = os.path.realpath(original)
        self.log = []
        self.decoded = []

    def _is_original(self, npz):
        name = getattr(getattr(npz, 'zip', None), 'filename', None) or getattr(getattr(npz, 'fid', None), 'name', None)
        return isinstance(name, (str, os.PathLike)) and os.path.realpath(name) == self.original

    def __enter__(self):
        from bayesreg.io import pair as pair_module
        self._module, self._real = pair_module, pair_module._open_npz
        self._getitem = np.lib.npyio.NpzFile.__getitem__
        guard = self

        def opened(path):
            npz = self._real(path)
            return _GuardedNpz(npz, path, self.log) if os.path.realpath(path) == self.original else npz

        def getitem(npz, key):
            if guard._is_original(npz):
                if key not in ALLOWED_KEYS:
                    guard.decoded.append(dict(path=guard.original, refused=str(key)))
                    raise MemberRefused(f'{guard.original}: member {key!r} decoded outside the loader\'s hook (only {list(ALLOWED_KEYS)})')
                guard.decoded.append(dict(path=guard.original, member=key))
            return guard._getitem(npz, key)
        pair_module._open_npz = opened
        np.lib.npyio.NpzFile.__getitem__ = getitem
        return self

    def __exit__(self, *exc):
        self._module._open_npz = self._real
        np.lib.npyio.NpzFile.__getitem__ = self._getitem
        return False

    def members(self):
        return [r['member'] for r in self.log if 'member' in r]

    def decoded_members(self):
        return [r['member'] for r in self.decoded if 'member' in r]

    def refused(self):
        return [dict(r, level='hook') for r in self.log if 'refused' in r]+[dict(r, level='NpzFile') for r in self.decoded if 'refused' in r]


def read_members(path):
    """{fixed, moving, initial_u (if there)} of a pair.npz as stored, one key at a time through bayesreg.io.pair._open_npz; nothing else asked for, the archive never listed."""
    from bayesreg.io import pair as pair_module
    out = {}
    with pair_module._open_npz(path) as npz:
        for key in IMAGE_KEYS:
            out[key] = np.array(npz[key])
        if ZERO_KEY in npz:
            out[ZERO_KEY] = np.array(npz[ZERO_KEY])
    return out


def same_array(a, b):
    return a is not None and b is not None and a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def npz_members(path):
    """The members of a file this tool wrote (the copy, a derived pair, the method's inputs): its member names, then each array.  Not for the original."""
    with np.load(path, allow_pickle=False) as z:
        keys = list(z.files)
        return keys, {k: np.array(z[k]) for k in keys}


def strip(original, out_dir, *, pin):
    """Step 1: the copy without label members and its manifest of exactly load_pair's fields (MANIFEST_FIELDS; minor 4); read back and compared with the original's members."""
    from bayesreg.io.pair import MANIFEST_FIELDS, MANIFEST_GRID_KEYS, _read_manifest
    from jd_curl.three_d.nifti import file_sha256
    raw = read_members(original)
    keys = [k for k in ALLOWED_KEYS if k in raw]
    out_dir.mkdir(parents=True)
    np.savez_compressed(out_dir/PAIR_FILE, **{k: raw[k] for k in keys})
    _, metadata, selected = _read_manifest(Path(original).parent/MANIFEST_FILE)
    manifest = dict(fixed_subject=selected['fixed_subject'], moving_subject=selected['moving_subject'], **{k: metadata[k] for k in MANIFEST_GRID_KEYS},
                    sources={side: dict(sha256=selected['source_sha256'][side]) for side in ('fixed', 'moving')}, pair_sha256=file_sha256(out_dir/PAIR_FILE))
    if sorted(manifest) != sorted(MANIFEST_FIELDS):
        raise ValueError(f'the copy\'s manifest fields {sorted(manifest)} are not load_pair\'s {sorted(MANIFEST_FIELDS)}')
    (out_dir/MANIFEST_FILE).write_text(json.dumps(manifest, indent=1)+'\n', encoding='utf-8')
    back_keys, back = npz_members(out_dir/PAIR_FILE)
    problems = [] if back_keys == keys else [f'the copy holds {back_keys}, not {keys}']
    problems += [f'{k}: the copy is not the original member, bit for bit' for k in keys if not same_array(back.get(k), raw[k])]
    record = dict(pair_npz=str(out_dir/PAIR_FILE), pair_sha256=manifest['pair_sha256'], manifest=str(out_dir/MANIFEST_FILE), manifest_sha256=file_sha256(out_dir/MANIFEST_FILE), keys=back_keys,
                  members={k: dict(dtype=str(raw[k].dtype), shape=list(raw[k].shape), sha256=hashlib.sha256(np.ascontiguousarray(raw[k]).tobytes()).hexdigest()) for k in keys},
                  source=str(original), source_pair_sha256=pin.provenance['pair_sha256'], problems=problems, passed=not problems)
    return record


def compare_loads(pin_original, pin_copy):
    """Step 2: what load_pair read from the copy against what it read from the original."""
    a, b = pin_original.provenance, pin_copy.provenance
    items = dict(keys_read=a['keys_read'] == b['keys_read'], initial_u_present=a['initial_u_present'] == b['initial_u_present'],
                 fixed=same_array(pin_original.fixed, pin_copy.fixed), moving=same_array(pin_original.moving, pin_copy.moving),
                 grid_metadata=dict(pin_original.grid_metadata) == dict(pin_copy.grid_metadata),
                 subjects=(pin_original.fixed_subject, pin_original.moving_subject) == (pin_copy.fixed_subject, pin_copy.moving_subject),
                 source_sha256=dict(pin_original.source_sha256) == dict(pin_copy.source_sha256))
    return dict(keys_read=dict(original=a['keys_read'], copy=b['keys_read']), initial_u_present=dict(original=a['initial_u_present'], copy=b['initial_u_present']), items=items,
                problems=[f'load_pair: {k} differs' for k, ok in items.items() if not ok], passed=all(items.values()))


def firewall_step(pin_original, pin_copy, out_dir, *, pair):
    """Step 3: firewall_c's derived pair from the original's view and from the copy's: both pass firewall_check; the same members bit for bit; the same file."""
    import firewall_c as fc
    from jd_curl.three_d.nifti import file_sha256
    records, checks = {}, {}
    for name, pin in (('from_original', pin_original), ('from_copy', pin_copy)):
        records[name] = fc.derive_pair(pin, out_dir/name, pair=pair)
        checks[name], _ = fc.firewall_check(pin, out_dir/name, pair=pair)
    (ka, a), (kb, b) = npz_members(out_dir/'from_original'/PAIR_FILE), npz_members(out_dir/'from_copy'/PAIR_FILE)
    same = {k: same_array(a.get(k), b.get(k)) for k in fc.DERIVED_KEYS}
    files = {name: file_sha256(out_dir/name/PAIR_FILE) for name in records}
    problems = [f'firewall_check ({name}): {c["problems"]}' for name, c in checks.items() if not c['passed']]
    problems += [] if ka == kb == list(fc.DERIVED_KEYS) else [f'derived keys {ka} / {kb} are not {list(fc.DERIVED_KEYS)}']
    problems += [f'{k}: the derived members differ' for k, ok in same.items() if not ok]
    problems += [] if files['from_original'] == files['from_copy'] else [f'the derived pair.npz files differ: {files}']
    return dict(records=records, firewall_check=checks, same_members=same, pair_npz_sha256=files, problems=problems, passed=not problems)


def runner_step(original, copy, out_dir, *, fixed, moving, allowlist, shape):
    """Step 4: XB's two readers of a pair on the copy (no missing-key error), and derive_inputs from the copy against derive_inputs from the original."""
    import xb_core as xc
    from jd_curl.three_d.nifti import file_sha256
    out_dir.mkdir(parents=True)
    checked = xc.check_pair_input(copy, fixed=fixed, moving=moving, allowlist=allowlist)
    arrays, records = {}, {}
    for name, source in (('original', original), ('copy', copy)):
        arrays[name], records[name] = xc.derive_inputs(source, out_dir/f'inputs_{name}.npz', shape=shape)
    same = {k: same_array(arrays['original'].get(k), arrays['copy'].get(k)) for k in xc.IMAGE_MEMBERS}
    files = {name: file_sha256(out_dir/f'inputs_{name}.npz') for name in arrays}
    problems = [f'{k}: the method inputs differ' for k, ok in same.items() if not ok]
    problems += [] if files['original'] == files['copy'] else [f'the method inputs files differ: {files}']
    return dict(check_pair_input=checked, derive_inputs=records, same_members=same, inputs_sha256=files, problems=problems, passed=not problems,
                note='xb_drive reads a pair only through these two (step 7 check_pair_input, step 10 derive_inputs); the method processes get inputs.npz only (prereg §2.3 (a)); sanity() is gone (§2.3 (b))')


def run(pair=DEFAULT_PAIR, out=None, *, code_identity=None, log=print):
    """The whole check (module docstring).  Returns (exit code, record or None)."""
    from bayesreg.io.pair import load_pair

    def refuse(stage, message):
        record = write_refusal(stage, message, pair=pair, out=out)
        log(f'refused ({stage}): {message}; {record}')
        return 2, None
    try:                                            # a. the pair's name, b. the output directory (minor 1: a bad name with no --out is a refusal too)
        fixed, moving = hc.split_pair(pair)
        out = Path(out) if out is not None else root()/f'{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}_{hc.short_pair(pair)}'
    except hc.Refused as exc:
        return refuse(exc.stage, str(exc))
    problem = out_problem(out)
    if problem:
        return refuse('output', problem)
    out = Path(out).resolve()
    code = (code_identity or hc.code_identity)()   # c. the code (minor 3): HEAD readable, clean, nothing untracked in the code directories
    if not code.get('commit') or code.get('clean') is not True:
        return refuse('code', f'the code tree is not clean or its HEAD cannot be read: {code}')
    try:
        admission, allowlist = hc.admit_dev_pair(fixed, moving)        # d. C23.4-2 before any file of the pair
    except hc.Refused as exc:
        return refuse(exc.stage, str(exc))
    original = Path(hc.DEV_STAGE)/pair/PAIR_FILE
    out.mkdir(parents=True)
    copy = out/'stage'/pair/PAIR_FILE
    steps, problems = {}, []
    guard = MemberGuard(original)
    t0 = time.perf_counter()
    with guard:
        def step(name, fn):
            try:
                steps[name] = fn()
            except Exception as exc:                # noqa: BLE001 - recorded; the steps that need this one do not run
                steps[name] = dict(passed=False, error=f'{type(exc).__name__}: {exc}'[:2000])
            if not steps[name]['passed']:
                problems.extend(f'{name}: {p}' for p in (steps[name].get('problems') or [steps[name].get('error') or 'not passed']))
            log(f"{name}: {'passed' if steps[name]['passed'] else 'PROBLEM'}")
            return steps[name]['passed']
        pins = {}

        def load(name, path):
            pins[name] = load_pair(path, allowlist=allowlist)
            return dict(keys_read=pins[name].provenance['keys_read'], pair_sha256=pins[name].provenance['pair_sha256'], passed=True)
        if step('load_original', lambda: load('original', original)) and step('strip', lambda: strip(original, copy.parent, pin=pins['original'])):
            if step('load_copy', lambda: load('copy', copy)):
                step('load_pair', lambda: compare_loads(pins['original'], pins['copy']))
                step('firewall_c', lambda: firewall_step(pins['original'], pins['copy'], out/'firewall', pair=pair))
            step('external_runner', lambda: runner_step(original, copy, out/'xb', fixed=fixed, moving=moving, allowlist=allowlist, shape=tuple(pins['original'].grid.shape)))
    asked, decoded, refused = guard.members(), guard.decoded_members(), guard.refused()
    if refused or not set(asked) <= set(ALLOWED_KEYS) or not set(decoded) <= set(ALLOWED_KEYS):
        problems.append(f'the original was asked for or decoded outside {list(ALLOWED_KEYS)}: {refused or sorted(set(asked) | set(decoded))}')
    wanted = ('load_original', 'strip', 'load_copy', 'load_pair', 'firewall_c', 'external_runner')
    missing = [name for name in wanted if name not in steps]
    passed = not problems and not missing and all(steps[name]['passed'] for name in wanted)
    record = dict(schema=SCHEMA, rule=RULE, pair=pair, code=code, admission=admission, original=str(original), out=str(out), steps=steps, steps_not_run=missing,
                  original_members_asked=asked, original_members_decoded=decoded, original_refused=refused, problems=problems, passed=passed,
                  seconds=round(time.perf_counter()-t0, 3), utc=hc.utc())
    hc.write_json_once(out/RECORD, record)
    log(f'checklist item 14: {"passed" if passed else "NOT passed"}; {out/RECORD}')
    return (0 if passed else 1), record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--out', default=None, help=f'a new directory under <results>/{ROOT_NAME}/ (default <UTC>_<pair>)')
    r.add_argument('--pair', default=DEFAULT_PAIR)
    o = ap.parse_args(argv)
    return run(o.pair, o.out)[0]


if __name__ == '__main__':
    sys.exit(main())
