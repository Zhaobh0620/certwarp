"""experiments/lpba/make_eval_manifest_lpba.py -- step 10 of the LPBA40 second test set (prereg §8.2 step 10; contract C26.4 L51, revision 11e): the script that writes, and checks
before it is committed, runs/lpba40_test_v1/eval_manifest.txt -- the input lines the evaluation entry bayesreg.eval.lpba_eval reads (check_manifest: "exactly the files this entry
reads").  LP-C2 integration review I1 and the lead's ruling (docs/bayesreg/reviews/LP-C2整合_复核_20261120.md): the lines are derived from the entry's own input_roles, never written
by hand (the hold-out's D108 way, runs/20261116-d108/make_eval_manifest.py, in the code tree here because the formal run's checkout is P_L's: rule one allows no new file there).

What it derives, in the entry's own order and with the entry's own functions (bayesreg.eval.lpba_eval, the formal mode lpba40_test only):
  1. --out checked (a new directory: not in the code tree, not under a root the entry's spy watches, not a temporary place); the guard on: an audit hook that raises
     GuardViolation before any open or directory listing of a label file (any path named label.nii.gz / label.nii / label_mm.nii.gz, or under */delineation/label*) and of any
     file under lpba_eval.spy_roots() (RAW_ROOT, the data root, data/three_d, the hold-out's results) except, once they are derived, the original images the stage manifests name.
     Every violation is also recorded (a caller that swallows one cannot hide it).  The entry's own spy (lpba_eval.start_spy: the sealed lists' cache empty) records every open.
  2. rule one (lpba_eval.check_code_identity('lpba40_test'): bayesreg.io.code_identity with the LPBA40 constants, P_L from runs/lpba40_test_v1/pinned_code.json); bayesreg imported
     from this tree; lpba_common.admit (pins, names, sealed lists, check_subject); no master lock yet (the manifest never changes after it); the preregistration and the arm table
     committed and unchanged (their sha256 are the manifest's header); the rows of test_pairs.tsv.
  3. check_cells (every (arm, pair) cell written, D107; the method cell crossed with LP-B's finalized record); image_originals (the stage manifests' source_files.<side>.original);
     input_roles: the input lines are exactly its keys -- roles 'cell' (terminal records, complete files, review records, fields, authorization and stop records, the finalized
     record), 'stage' (each pair's manifest.json and pair.npz) and 'image_header' (each subject's original image.nii.gz, whose header the entry reads).
  4. every input hashed (its bytes; nothing decoded: no array member, no NIfTI read); the entry's checks that need no commit (_cell_sha_checks: the records read in (3) at their
     listed sha256, a field at the sha256 its terminal record names, the finalized record's sha256 of a method file, an image at the sha256 its stage manifest names); the entry's
     spy allows the inputs and the sealed lists (an open before that outside them refuses, as it would refuse the evaluation); check_stage (each stage manifest, verified against
     the list, in its row's direction); the manifest text ('# bayesreg.lpba40.eval_manifest.v1', '# prereg_sha256 <hex>', '# arm_table_sha256 <hex>', '<64 hex>  <absolute path>'
     sorted by path) read back by lpba_eval.parse_manifest and compared with the roles and the sha256 before anything is written.
  5. --check-only: the text into <out>/eval_manifest.txt (nothing in the code tree).  --write: the text into <code tree>/runs/lpba40_test_v1/eval_manifest.txt, created exclusively (an
     existing file refuses before anything is read); read back: the bytes, parse_manifest, the roles, every input hashed a second time.  It never commits: the lead commits it with
     runs/lpba40_test_v1/eval_records.txt (the record-only lines -- the HI outputs among them, which this entry does not read -- written by the lead; the entry checks only that both
     were last committed in one commit, lpba_eval.records_file_state) and writes the commit and the manifest's sha256 into the decision log (prereg §8.2 step 10).
  <out>/report.json: the role, path and sha256 of every line, the counts, the header, the code identity, the guard's and the spy's records, the verdict.  Standard output: counts,
  sha256 values and the verdict.  No truth is read, no label file is opened, no unit is started, nothing under the results root is written but --out (and nothing at all moved).

Usage (from the formal checkout, rule one holding; PYTHONPATH need not be set: the code tree is put first on sys.path):
  PY=/home/rmcfz/miniconda3/envs/sci310/bin/python
  nice -n 19 $PY -B experiments/lpba/make_eval_manifest_lpba.py --check-only --out <a new directory under the results root>
  nice -n 19 $PY -B experiments/lpba/make_eval_manifest_lpba.py --write --out <another new directory>
Exit codes: 0 derived (and written) and checked; 2 refused before writing (nothing written in the code tree); 3 written, then the read-back failed (the written file moved to
<out>/failed/); 4 the guard stopped an open of a label file or under a guarded root, or the evaluation entry's spy refused an open (swallowed or not; LP-C2 final diff review
minor 1) -- a file already written is moved to <out>/failed/ then too; 1 an error of this script.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
TREE = HERE.parent.parent                      # the code tree this script and the bayesreg it imports are in
REPO = TREE                                    # the code tree whose committed files are read and whose runs/lpba40_test_v1/ is written (the same; the tests point it elsewhere)
for _p in (TREE, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SCHEMA = 'bayesreg.lpba40.eval_manifest_maker.v1'
PAIR_SET = 'lpba40_test'                       # the formal mode only: the feasibility mode has no manifest (lpba_eval.own_list)
EXIT = dict(ok=0, internal=1, refused=2, check_failed=3, guard=4)
LABEL_NAMES = ('label.nii.gz', 'label.nii', 'label_mm.nii.gz')       # lpba_eval.LABEL_REL's file and the entry's private copy
LABEL_PART = '/delineation/label'
TMP_ROOTS = (Path('/tmp'), Path('/var/tmp'), Path('/dev/shm'), Path('/run'))     # --out keeps report.json (the evidence): never a temporary place (D108 m4)
OUT_NAME = 'eval_manifest.txt'
FAILED_DIR = 'failed'
GUARDED_EVENTS = frozenset(('open', 'os.listdir', 'os.scandir', 'glob.glob', 'glob.glob/2', 'shutil.copyfile', 'shutil.copytree'))


class Refusal(Exception):
    def __init__(self, stage, message):
        super().__init__(f'{stage}: {message}')
        self.stage = stage


class GuardViolation(RuntimeError):
    """An open or a listing of a label file, or under a guarded root outside the allowed original images: raised before the operating system is asked."""


def say(text):
    print(text, flush=True)


# ---- the guard -----------------------------------------------------------------------------------------------------------------------------------------------------
def _form(path):
    text = os.path.normpath(os.path.abspath(os.fsdecode(path)))
    return text.casefold() if text.startswith('/mnt/') else text           # drvfs is case-insensitive under /mnt/<drive> (D108 m5)


def forms(path):
    return {_form(path), _form(os.path.realpath(os.fsdecode(path)))}


def within(path, root):
    path, root = _form(path), _form(root)
    return path == root or path.startswith(root.rstrip(os.sep)+os.sep)


_GUARD = dict(active=False, roots=(), allowed=frozenset(), hits=[])
_HOOKED = [False]


def is_label(path):
    name = os.path.basename(os.fsdecode(path)).casefold()
    return name in LABEL_NAMES or LABEL_PART in os.fsdecode(path).casefold()


def guard_path(path):
    """GuardViolation (recorded first) for a label file anywhere, or a path under a guarded root that is not one of the allowed original images."""
    if not _GUARD['active']:
        return
    seen = forms(path)
    why = None
    if any(is_label(f) for f in seen):
        why = 'a label file'
    elif any(within(f, r) for f in seen for r in _GUARD['roots']) and not seen & _GUARD['allowed']:
        why = 'under a guarded root (lpba_eval.spy_roots) and not an original image of the stage manifests'
    if why:
        _GUARD['hits'].append(dict(path=os.fsdecode(path), why=why))
        raise GuardViolation(f'{os.fsdecode(path)}: {why}')


def _audit(event, args):
    if not _GUARD['active'] or event not in GUARDED_EVENTS or not args:
        return
    path = args[0]
    if path is None or isinstance(path, int):
        return
    guard_path(path)


def guard_on(roots):
    if not _HOOKED[0]:
        sys.addaudithook(_audit)                 # stays for the process; inactive while _GUARD['active'] is False
        _HOOKED[0] = True
    _GUARD.update(active=True, roots=tuple(str(r) for r in roots), allowed=frozenset(), hits=[])


def guard_allow(paths):
    _GUARD['allowed'] = frozenset(f for p in paths for f in forms(p))


def guard_off():
    _GUARD['active'] = False
    return list(_GUARD['hits'])


# ---- the pieces ------------------------------------------------------------------------------------------------------------------------------------------------------
def sha256_file(path):
    guard_path(path)
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def check_out(out, E):
    out = Path(out)
    if out.exists():
        raise Refusal('out', f'{out} exists: a new directory every run')
    if not out.parent.is_dir():
        raise Refusal('out', f'{out.parent} is not a directory')
    places = [REPO, TREE, *E.spy_roots(), E.root(PAIR_SET), *TMP_ROOTS]
    if any(within(out, r) or within(os.path.realpath(out.parent), r) for r in places):
        raise Refusal('out', f'{out}: not in the code tree, under a root the entry\'s spy watches, under the formal root {E.root(PAIR_SET)} or a temporary place (the report is '
                             'the evidence)')
    return out


def manifest_text(E, prereg_sha256, arm_table_sha256, inputs):
    lines = [f'# {E.SCHEMA_MANIFEST}', f'# prereg_sha256 {prereg_sha256}', f'# arm_table_sha256 {arm_table_sha256}']
    lines += [f'{sha}  {path}' for path, sha in sorted(inputs.items())]
    return '\n'.join(lines)+'\n'


def verify_text(E, raw, *, roles, header, cells, originals, rehash=False):
    """The written (or to-be-written) text read as the entry reads it: parse_manifest, the header, the path set = input_roles, the sha256 checks; `rehash`: every file hashed again."""
    got_header, inputs = E.parse_manifest(raw, 'the generated manifest')
    if got_header != header:
        raise Refusal('verify', f'header {got_header} is not {header}')
    if set(inputs) != set(roles):
        raise Refusal('verify', f'{len(set(inputs)-set(roles))} lines outside the entry\'s input roles, {len(set(roles)-set(inputs))} roles without a line')
    E._cell_sha_checks(cells, originals, inputs, 'the generated manifest')
    if rehash:
        changed = sorted(p for p, sha in inputs.items() if sha256_file(p) != sha)
        if changed:
            raise Refusal('verify', f'{len(changed)} inputs changed since they were hashed: {changed[:5]}')
    return inputs


def derive(E, report):
    """Steps 2-4 of the module docstring.  Returns (text, roles, header, cells, originals, inputs)."""
    from bayesreg.eval import holdout as H
    import bayesreg
    if not within(bayesreg.__file__, TREE):
        raise Refusal('code', f'bayesreg is imported from {bayesreg.__file__}, not from this code tree {TREE}')
    identity = E.check_code_identity(PAIR_SET, REPO)
    report['code_identity'] = dict(pinned=(identity.get('rule_one') or {}).get('pinned'), head=(identity.get('code') or {}).get('head'), clean=(identity.get('code') or {}).get('clean'))
    admitted = E._admit(PAIR_SET)
    if E.master_lock(PAIR_SET).exists():
        raise Refusal('once', f'{E.master_lock(PAIR_SET)} exists: the evaluation has taken its lock; the manifest never changes after it')
    prereg = E.check_prereg(REPO)
    table, table_record = E.load_arm_table(PAIR_SET, REPO)
    rows = E.pair_rows(PAIR_SET, admitted.lists)
    if [(r[1], r[2]) for r in rows] != [tuple(p) for p in admitted.pairs]:
        raise Refusal('pairs', 'the admitted pairs are not the mode\'s table in its order')
    spy = E._SPY
    spy.watch(E.watched_roots(table))
    cells = E.check_cells(table, rows, repo=REPO, pair_set=PAIR_SET)
    originals = E.image_originals(table, rows)
    guard_allow([path for path, _ in originals.values()])
    roles = E.input_roles(table, rows, cells, originals)
    absent = sorted(p for p in roles if not Path(p).is_file())
    if absent:
        raise Refusal('inputs', f'{len(absent)} files the entry reads are not there: {absent[:5]}')
    inputs = {p: sha256_file(p) for p in sorted(roles)}
    E._cell_sha_checks(cells, originals, inputs, 'the generated manifest')
    from bayesreg.io import sealed as sealed_module
    spy.allow(list(inputs)+[Path(sealed_module.REPO)/s['path'] for s in sealed_module.SEALED_LISTS])     # an open outside the inputs before this refuses, as in the entry
    E.check_stage(table, rows, inputs)
    header = dict(prereg_sha256=prereg['sha256'], arm_table_sha256=table_record['sha256'])
    text = manifest_text(E, header['prereg_sha256'], header['arm_table_sha256'], inputs)
    verify_text(E, text.encode('utf-8'), roles=roles, header=header, cells=cells, originals=originals)
    counts = {}
    for role in roles.values():
        counts[role] = counts.get(role, 0)+1
    report.update(pairs=len(rows), arms=len(table['arms']), cells=len(cells), header=header, roles=dict(sorted(counts.items())), n_inputs=len(inputs),
                  lines=[dict(role=roles[p], path=p, sha256=inputs[p]) for p in sorted(inputs)], manifest_sha256=H.sha_bytes(text.encode('utf-8')))
    return text, roles, header, cells, originals, inputs


def move_to_failed(out, written):
    target = Path(out)/FAILED_DIR
    target.mkdir(parents=True, exist_ok=True)
    moved = []
    for path in written:
        if Path(path).exists():
            new = target/Path(path).name
            os.replace(path, new)
            moved.append(dict(old=str(path), new=str(new), sha256=hashlib.sha256(new.read_bytes()).hexdigest()))
    return moved


def run(argv=None):
    parser = argparse.ArgumentParser(prog='experiments/lpba/make_eval_manifest_lpba.py', description=__doc__.split('\n\n')[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only', action='store_true', help='write the manifest into --out only')
    mode.add_argument('--write', action='store_true', help='write runs/lpba40_test_v1/eval_manifest.txt into the code tree (once)')
    parser.add_argument('--out', required=True, type=Path, help='a new directory (report.json; with --check-only also eval_manifest.txt)')
    args = parser.parse_args(argv)
    from bayesreg.eval import lpba_eval as E
    report = dict(schema=SCHEMA, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), mode='write' if args.write else 'check_only', pair_set=PAIR_SET, code=str(REPO))
    try:
        out = check_out(args.out, E)
    except Refusal as exc:
        say(f'refused: {exc}')
        return EXIT['refused']
    out.mkdir()
    target = REPO/E.MANIFEST_REL if args.write else out/OUT_NAME
    report['target'] = str(target)
    written, code = [], EXIT['internal']
    E.stop_spy()
    guard_on(E.spy_roots())
    try:
        if args.write and target.exists():
            raise Refusal('once', f'{target} exists: the manifest is written once (step 10)')
        E.start_spy(E.spy_roots())
        text, roles, header, cells, originals, inputs = derive(E, report)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, 'x', encoding='utf-8', newline='\n') as handle:
            handle.write(text)
        written.append(target)
        raw = target.read_bytes()
        if raw != text.encode('utf-8'):
            raise Refusal('read_back', f'{target}: the bytes read back are not the text written')
        try:
            verify_text(E, raw, roles=roles, header=header, cells=cells, originals=originals, rehash=True)
        except Refusal as exc:
            raise Refusal('read_back', str(exc)) from None
        report.update(verdict='written and checked', written=str(target))
        code = EXIT['ok']
    except GuardViolation as exc:
        report.update(verdict='guard', error=str(exc))
        code = EXIT['guard']
    except (Refusal, E.HoldoutRefused, E.SubjectSealed, E.SealedListError) as exc:
        stage = getattr(exc, 'stage', type(exc).__name__)
        report.update(verdict='refused', stage=stage, error=f'{type(exc).__name__}: {exc}')
        code = EXIT['check_failed'] if written else EXIT['refused']
    except Exception as exc:                                             # noqa: BLE001 - the record keeps it
        report.update(verdict='error', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        code = EXIT['internal']
    finally:
        spy = E.stop_spy()
        hits = guard_off()
    record = None if spy is None else spy.record()
    spy_refused = [] if record is None else list(record['refused'])
    if hits and code != EXIT['guard']:                                   # a violation a caller swallowed (or that another refusal followed) still decides the exit
        report.update(verdict='guard', error=f'{len(hits)} guarded opens were raised and not let through', error_after=report.get('error'))
        code = EXIT['guard']
    if spy_refused and code != EXIT['guard']:                            # final diff review minor 1: the entry's spy's refusals decide the exit as the guard's do
        report.update(verdict='spy', error=f"{len(spy_refused)} opens the evaluation entry's spy refused (SpyRefused; swallowed or followed by another error)",
                      error_after=report.get('error'))
        code = EXIT['guard']
    report['guard_hits'] = hits
    if record is not None:
        report['spy'] = dict(events=len(record['events']), refused=spy_refused, sealed_list_reads=record['sealed_list_reads'])
    if code != EXIT['ok'] and written:
        report['moved_to_failed'] = move_to_failed(out, written)
    report['exit'] = code
    (out/'report.json').write_text(json.dumps(report, indent=1, default=str)+'\n', encoding='utf-8')
    if code == EXIT['ok']:
        say(f"{report['verdict']}: {target}; inputs {report['n_inputs']} (roles {report['roles']}); pairs {report['pairs']}, arms {report['arms']}, cells {report['cells']}; "
            f"manifest sha256 {report['manifest_sha256']}; header {report['header']}; guard hits 0; report {out/'report.json'}")
    else:
        say(f"{report['verdict']} (exit {code}): {report.get('error')}; report {out/'report.json'}")
    return code


if __name__ == '__main__':
    sys.exit(run())
