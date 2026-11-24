"""XB: the regression check of prereg §5.1 item 7 (only the tool; the lead releases the runs): on one development pair, every main-family external arm (FireANTs Greedy, FireANTs SyN, ANTs -t s)
once through the archive's harness with only its path constants changed, and once through the new runner; their disp.npz compared, bitwise or by the largest difference.  No Dice is read.

    python -B experiments/xb/xb_regress.py prepare-old --out NEW_DIR                     the archive's drive.py with three path constants replaced, the four method scripts byte for byte, the diff
    python -B experiments/xb/xb_regress.py old-argv --arm ARM --pair PAIR --out DIR      the old harness's unit start string (one arm, one pair); launches nothing
    python -B experiments/xb/xb_regress.py new-argv --arm ARM --pair PAIR                the new runner's unit start string (xb_chain.unit_argv, pair set dev); launches nothing
    python -B experiments/xb/xb_regress.py compare --arm ARM --pair PAIR --old DIR      both disp.npz (and the foreground difference, the repeats) -> DIR/compare_<arm>_<pair>.json

What "only the path constants" means: ROOT (seq_vp_3d -> this repository's main checkout, whose data/ link and scripts/ the old driver uses), STAGE (the BL2VAL stage -> NEW_DIR, so HARNESS = NEW_DIR/
harness holds the copies), BIG (/mnt/d/seq_vp_3d_results/verification -> NEW_DIR/big).  Nothing else of drive.py changes: it still reads every member of the development pair.npz, runs sanity() on its
labels and writes the sanity Dice into its own result.json and driver log -- the reason those two files are NOT to be opened (regression.md); compare reads the two disp.npz (the old one found by
its directory name under NEW_DIR/big, never through the old result.json), the two method_out.json beside them (only their repeat keys; they hold no Dice), and the member fixed of the new cell's
derived inputs.npz (its foreground mask, A5a review I3).  Development pairs only (holdout_common.DEV_PAIRS): the old driver has no firewall.
"""
import argparse
import difflib
import glob
import json
from pathlib import Path
import re
import shlex
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import xb_chain  # noqa: E402

xc = xb_chain.xc
hc = xc.hc
def root():
    """The development root's regression directory, read at call time (the tests' sandbox moves it; in the sandbox a real root is refused: I7)."""
    return hc.guard_root(xc.DEV_ROOT/'regression')
ARMS = ('fireants_greedy', 'fireants_syn', 'ants_s')
REPLACEMENTS = (
    ("ROOT = Path('/home/rmcfz/code/python_projects/seq_vp_3d')", "ROOT = Path({main!r})                                          # XB regression: path constant"),
    ("STAGE = ROOT/'results/verification/3d_stage_BL2VAL_v1'    # BL2VAL: stage, harness, pair and small-result root", "STAGE = Path({out!r})    # XB regression: path constant"),
    ("BIG = Path('/mnt/d/seq_vp_3d_results/verification')", "BIG = Path({big!r})                                      # XB regression: path constant"),
)
SCRIPTS = ('m_fireants.py', 'm_convexadam.py', 'm_unigradicon.py', 'm_voxelmorph.py')


def old_driver_text(out, *, main=hc.MAIN_REPO, repo=xc.REPO):
    source = (repo/xc.ARCHIVE_HARNESS/'drive.py').read_text(encoding='utf-8')
    if xc.sha256(repo/xc.ARCHIVE_HARNESS/'drive.py') != xc.ARCHIVE_DRIVE_SHA256:
        raise hc.Refused('frozen', 'the archive drive.py is not the pinned one')
    text = source
    for old, new in REPLACEMENTS:
        if text.count(old) != 1:
            raise hc.Refused('regression', f'drive.py: {old!r} is not there exactly once')
        text = text.replace(old, new.format(main=str(main), out=str(out), big=str(Path(out)/'big')))
    return source, text


def prepare_old(out, *, main=hc.MAIN_REPO, repo=xc.REPO):
    out = Path(out)
    if not out.resolve().is_relative_to(root().resolve()) or out.exists():
        raise hc.Refused('output', f'{out}: a new directory under {root()}')
    source, text = old_driver_text(out.resolve(), main=main, repo=repo)
    harness = out/'harness'
    harness.mkdir(parents=True)
    (harness/'drive.py').write_text(text, encoding='utf-8')
    for name in SCRIPTS:
        shutil.copyfile(repo/xc.ARCHIVE_HARNESS/name, harness/name)
        if xc.sha256(harness/name) != xc.sha256(repo/xc.ARCHIVE_HARNESS/name):
            raise hc.Refused('regression', f'{name}: the copy differs')
    diff = ''.join(difflib.unified_diff(source.splitlines(True), text.splitlines(True), 'archive/drive.py', 'old/drive.py'))
    (out/'drive_py.diff').write_text(diff, encoding='utf-8')
    changed = [line for line in diff.splitlines() if line.startswith(('+', '-')) and not line.startswith(('+++', '---'))]
    record = dict(schema=xc.SCHEMA.format('regression_old'), out=str(out), drive_py_archive_sha256=xc.ARCHIVE_DRIVE_SHA256, drive_py_old_sha256=xc.sha256(harness/'drive.py'),
                  scripts={n: xc.sha256(harness/n) for n in SCRIPTS}, changed_lines=changed, n_changed=len(changed), utc=hc.utc(),
                  rule='only ROOT, STAGE and BIG replaced (three lines out, three in); the old driver still reads the labels for its sanity Dice: its result.json and driver log are not opened')
    hc.write_json_once(out/'prepared.json', record)
    return record


def old_argv(arm, pair, out):
    if arm not in ARMS or pair not in hc.DEV_PAIRS:
        raise hc.Refused('regression', f'{arm} / {pair}: a main-family external arm on a development pair')
    out = Path(out)
    unit = f'tag-hb-xr-old-{arm}-{hc.short_pair(pair)}'
    stage = hc.MAIN_REPO/'data'/'three_d'/'stage_L2R_val_v1'       # the development stage the old driver reads (its pair.npz with the labels, for its sanity Dice): census row F (m6)
    env = dict(BL2_PAIR=pair, BL2_PAIR_SET=stage.name, BL2_OUT=str(out/'small'), BL2_UNIT_LIMITS=' '.join(xc.UNIT_PROPERTIES), OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, PYTHONDONTWRITEBYTECODE=1)
    argv = ['systemd-run', '--user', '--wait', '--collect', f'--unit={unit}', *[f'--property={p}' for p in xc.UNIT_PROPERTIES], f'--working-directory={out}',
            *[f'--setenv={k}={v}' for k, v in env.items()], str(xc.PY), '-I', '-B', str(out/'harness'/'drive.py'), arm]
    return argv


def new_argv(arm, pair):
    if arm not in ARMS or pair not in hc.DEV_PAIRS:
        raise hc.Refused('regression', f'{arm} / {pair}: a main-family external arm on a development pair')
    return xb_chain.unit_argv(arm, 'dev', pair)


def old_disp(arm, pair, out):
    """The old run's disp.npz, found by the old driver's directory name (bl2val_<arm>_<short>_<UTC>, drive.py:497), not through its result.json (which holds the sanity Dice)."""
    short = re.sub(r'^OASIS_OAS1_([0-9]+)_MR1__OASIS_OAS1_([0-9]+)_MR1$', r'\1__\2', pair)
    found = sorted(glob.glob(str(Path(out)/'big'/f'bl2val_{arm}_{short}_*'/'disp.npz')))
    if len(found) != 1:
        raise hc.Refused('regression', f'expected one old disp.npz for {arm} {pair} under {out}/big, found {found}')
    return Path(found[0])


REPEAT_KEYS = ('repeat_mean_norm_diff_fixed_fg_voxels', 'repeat_max_abs_diff_voxels')     # m_fireants.py's warm repeat (same process); the only keys read from a method_out.json


def new_disp(arm, pair):
    """The new cell's record (result.json, which holds no Dice): its disp.npz (sha256 checked), the export check's corr, its derived inputs.npz (fixed and moving byte for byte; sha256 checked) and
    the method_out.json beside disp.npz."""
    doc = json.loads(xc.cell_paths(arm, 'dev', pair)['result'].read_text(encoding='utf-8'))
    if doc.get('status') != 'ok' or not doc.get('disp'):
        raise hc.Refused('regression', f'the new run of {arm} {pair} is {doc.get("status")}')
    path = Path(doc['disp']['file'])
    if xc.sha256(path) != doc['disp']['sha256']:
        raise hc.Refused('regression', f'{path}: sha256 differs from the record')
    derived = doc.get('derived_inputs') or {}
    fixed = Path(derived['file']) if derived.get('file') else None
    if fixed is None or not fixed.is_file() or xc.sha256(fixed) != derived.get('sha256'):
        raise hc.Refused('regression', f'{fixed}: the derived inputs of the new run are missing or not as recorded')
    return dict(disp=path, corr=doc['export_check']['check']['corr'], fixed_npz=fixed, method_out=path.parent/'method_out.json')


def repeat_of(path):
    """REPEAT_KEYS of a method_out.json (None for a key it lacks, or for both when the file cannot be read), with its sha256."""
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return dict(file=str(path), sha256=None, error=f'{type(exc).__name__}: {exc}', **dict.fromkeys(REPEAT_KEYS))
    return dict(file=str(path), sha256=xc.sha256(path), **{k: doc.get(k) if isinstance(doc, dict) else None for k in REPEAT_KEYS})


def fixed_foreground(path):
    """fixed > 0 of the new cell's derived inputs.npz (the mask m_fireants.py takes for repeat_mean_norm_diff_fixed_fg_voxels); only the member fixed is read."""
    with np.load(path, allow_pickle=False) as q:
        return np.array(q['fixed']) > 0


def compare(arm, pair, out, *, old_path=None, new_path=None, fixed_npz=None, new_method_out=None):
    """Both disp.npz, compared: bitwise; the largest, mean and fraction of the absolute differences; and the norm of the difference averaged over the fixed image's foreground (fixed > 0 of the new
    cell's derived inputs: the measure m_fireants.py takes for its warm repeat, A5a review I3), beside the two runs' repeat_mean_norm_diff_fixed_fg_voxels (old: method_out.json beside the old
    disp.npz; new: beside the new one).  The tests give old_path / new_path / fixed_npz / new_method_out; None: found as the runs wrote them.  Writes <out>/compare_<arm>_<pair>.json once."""
    old_path = old_disp(arm, pair, out) if old_path is None else Path(old_path)
    if new_path is None:
        found = new_disp(arm, pair)
        new_path, new_corr = found['disp'], found['corr']
        fixed_npz = found['fixed_npz'] if fixed_npz is None else fixed_npz
        new_method_out = found['method_out'] if new_method_out is None else new_method_out
    else:
        new_corr = None
    with np.load(old_path) as q:
        a = np.array(q['d'])
    with np.load(new_path) as q:
        b = np.array(q['d'])
    same_shape = a.shape == b.shape and a.dtype == b.dtype
    bitwise = bool(same_shape and a.tobytes() == b.tobytes())
    delta = a.astype(np.float64)-b.astype(np.float64) if a.shape == b.shape else None
    diff = None if delta is None else np.abs(delta)
    fg = None
    if fixed_npz is not None and delta is not None:
        mask = fixed_foreground(fixed_npz)
        if mask.shape == delta.shape[1:] and mask.any():
            fg = dict(file=str(fixed_npz), sha256=xc.sha256(fixed_npz), n_voxels=int(mask.sum()), mean_norm_diff_fixed_fg_voxels=float(np.sqrt((delta**2).sum(0))[mask].mean()))
        else:
            fg = dict(file=str(fixed_npz), sha256=xc.sha256(fixed_npz), n_voxels=int(mask.sum()), mean_norm_diff_fixed_fg_voxels=None, problem=f'mask {mask.shape} against the field {delta.shape}')
    repeat = dict(old=repeat_of(Path(old_path).parent/'method_out.json'), new=None if new_method_out is None else repeat_of(new_method_out))
    doc = dict(schema=xc.SCHEMA.format('regression_compare'), arm=arm, pair=pair, old=dict(file=str(old_path), sha256=xc.sha256(old_path), shape=list(a.shape), dtype=str(a.dtype)),
               new=dict(file=str(new_path), sha256=xc.sha256(new_path), shape=list(b.shape), dtype=str(b.dtype), export_check_corr=new_corr), bitwise_equal=bitwise,
               max_abs_diff_voxels=None if diff is None else float(diff.max()), mean_abs_diff_voxels=None if diff is None else float(diff.mean()),
               fraction_differing=None if diff is None else float((diff > 0).mean()),
               fixed_fg=fg, mean_norm_diff_fixed_fg_voxels=None if fg is None else fg['mean_norm_diff_fixed_fg_voxels'], repeat=repeat, utc=hc.utc(),
               rule='disp.npz of the old harness (path constants only) against the new runner on the same development pair; bitwise, or the norm of the difference averaged over the fixed '
                    "image's foreground against the two runs' warm repeats where the arm is not deterministic (regression.md section 4, A5a review I3); no Dice read")
    hc.write_json_once(Path(out)/f'compare_{arm}_{pair}.json', doc)
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare-old')
    p.add_argument('--out', required=True)
    for name in ('old-argv', 'new-argv', 'compare'):
        p = sub.add_parser(name)
        p.add_argument('--arm', required=True)
        p.add_argument('--pair', required=True)
        if name != 'new-argv':
            p.add_argument('--out' if name == 'old-argv' else '--old', required=True)
    o = ap.parse_args(argv)
    try:
        if o.command == 'prepare-old':
            print(json.dumps(prepare_old(o.out), indent=1))
        elif o.command == 'old-argv':
            print(shlex.join(old_argv(o.arm, o.pair, o.out)))
        elif o.command == 'new-argv':
            print(shlex.join(new_argv(o.arm, o.pair)))
        else:
            doc = compare(o.arm, o.pair, o.old)
            print(json.dumps({k: doc[k] for k in ('arm', 'pair', 'bitwise_equal', 'max_abs_diff_voxels', 'mean_norm_diff_fixed_fg_voxels')}))
    except hc.Refused as exc:
        print(f'refused: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
