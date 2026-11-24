"""HE (runs/20261106-he/任务书.md, item 2): fallback one of the C22 line, configuration C under the label firewall (contract C22.5-2, D75; C22.9-4 (xi)).

What the frozen worker is handed.  The frozen reader decodes EVERY member of the pair.npz it is given (F:scripts/run_stage_3d_e.py:40-47, `{k: archive[k] for k in archive.files}`), so the method process never gives it the original file.  It writes a derived pair.npz:
    fixed, moving, initial_u      the original members, byte for byte (dtype, shape and bytes), read one key at a time through bayesreg.io.pair._open_npz (the loader's own hook; no other member is opened, the archive is never listed)
    fixed_seg4, fixed_seg35, moving_seg4, moving_seg35     all zero, float64: the frozen worker cannot run without them.  After the solve it computes labels.json and smoothness.json from these names
                                  (F:scripts/run_stage_3d_e19_mri.py:2160-2171 -> F:scripts/run_stage_3d_e11_short.py:47-55): with a member missing that is a KeyError after final_native.npz is written, exit 1, and the
                                  classifier calls the run unsure (KeyError is not one of its method exceptions); with zeros every Dice is None (nifti.dice_labels: a label absent from both is skipped) and the brain
                                  smoothness is None (smoothness_report on an empty mask).  Neither file is an evaluation: the evaluation unit computes the metrics from disp.npz, after the output is fixed.
and a derived manifest.json written from the firewall view only: the grid metadata and the subjects and source hashes of load_pair, the pair name of its directory, weights from the grid by the frozen
formula (every stage's prepare script writes weights = dict(lam=.1/L**2, kappa=.002, strong_mu_nu=1., weak_mu_nu=.001), L the cube root of the volume's extent; the worker reads only lam), and the Dice
record of the zero labels as baseline_label_metrics (the worker copies it into labels.json).  Nothing of the original manifest outside load_pair's whitelist is read.

Where it runs.  The frozen `prepare` takes a pair only from its pre-registered pair sets under its own root (F:scripts/run_stage_3d_e48_pg.py:45-97).  The frozen tree is not touched (no link switch, unlike
D0): a private copy of its jd_curl/ and scripts/ (checked against configC_reference.json:freeze_source_py_sha256) gets the derived pair in the slot data/three_d/stage_L2R_val_v1/<pair> with a
selection.json of that one pair; `prepare` copies the two files into the run directory and snapshots the source; experiments/d0/check_prepared.py then checks the prepared directory against configuration C
unchanged (its stage rule holds because the slot is always the same; the pair's real stage is recorded beside it, never written into the frozen records).  The copy is removed after `prepare`; the run imports
only its snapshot.  The slot name is the pair's name, except the frozen development pair (F:scripts/run_stage_3d_e17_pilot.py:34), which resolve_pair accepts only from E11: it gets SLOT_ALIAS_SUFFIX.

Launch: the D0 unit (experiments/d0/run_queue.sh build_sd_cmd: CPUQuota 1200 %, MemoryMax 12G, MemorySwapMax 4G, 8 threads, the JIT caches, `timeout --signal=TERM --kill-after=15s 300s`, exit_code and the
supervisor's time stamps) or the same inner command without systemd (DirectLauncher).  HE2 (H2, C22.B4-6 (d)): the unit also gets OOMPolicy=continue, and the inner command runs unit_probe.py before
its start stamp and after its finish stamp (unit_probe_start.json, unit_probe_end.json: the cgroup's memory files and systemd's properties as read).  Classification: experiments/d0/classify_run.py, unchanged.  Candidates (D75): final_native.npz, else the certified native
seed of levels.json (level 3, level_prepared, its transfer certificate passed); only for the classes ok and handover_only; nothing of the run is deleted here (D0's cleanup of level_*.npz is not done).
No function of this module reads a label: no member of the original pair.npz but fixed, moving, initial_u is opened.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
for _dir in ('mc0',):
    if str(HERE.parent/_dir) not in sys.path:
        sys.path.insert(0, str(HERE.parent/_dir))

import mc0_core as core  # noqa: E402

PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'
CONFIGC_REFERENCE = core.ROOT/'experiments/d0/configC_reference.json'
SLOT_STAGE = 'stage_L2R_val_v1'                 # a pair set of F:scripts/run_stage_3d_e48_pg.py:45-52, in the private copy only
FROZEN_DEV_PAIR = 'OASIS_OAS1_0001_MR1__OASIS_OAS1_0002_MR1'     # F:scripts/run_stage_3d_e17_pilot.py:34 PAIR: resolve_pair takes it only from the E11 set
SLOT_ALIAS_SUFFIX = '__he_slot'
STAGE_LABEL = 'HE-C'                            # protocol.json stage of every fallback run; check_prepared compares it
IMAGE_MEMBERS = ('fixed', 'moving')
ZERO_MEMBER = 'initial_u'
SEG_KEYS = ('fixed_seg4', 'fixed_seg35', 'moving_seg4', 'moving_seg35')
DERIVED_KEYS = ('fixed', 'fixed_seg4', 'fixed_seg35', 'moving', 'moving_seg4', 'moving_seg35', 'initial_u')     # experiments/d0/make_inputs.py KEY_ORDER (a test compares)
# configuration C, as experiments/d0/run_queue.sh runs A0 (its PRE and OPTS lines; a test parses them and compares)
PRE = 'fp32-v1:iterations=50/50/100/60,lr=0.5,cap=0.5,sigma_grad=1,sigma_field=1,mu=1/1/0.01/0.003,alpha=1e-3/1e-3/1e-3/0,kappa_pre=10,j0=0.2,seconds=60,taper=4'
OPTS = ('--strength', 'weak', '--lean-commits', '--resolution-exit',
        '--pg-metric-policy-levels', 'legacy_q1_jd_affine_always_v1,legacy_q1_jd_affine_always_v1,image_shift_ggn_equilibrated_jd_affine_v3,image_shift_ggn_equilibrated_jd_affine_v3',
        '--level-shares', '0.020833,0.0625,0.145833,1', '--geometry-weight-levels', '1,1,0.1,0.03', '--curvature-weight-levels', '1e-3,1e-3,1e-3,0',
        '--throughput', 'v41', '--export-reserve', '1', '--seconds', '90', '--pre-optimizer', PRE)
WATCHDOG_SECONDS, KILL_GRACE = 300, 15          # run_queue.sh WATCHDOG and its --kill-after
UNIT_RESOURCES = ('--property=CPUQuota=1200%', '--property=MemoryMax=12G', '--property=MemorySwapMax=4G')
# HE2 (H2; runs/20261107-he2/任务书.md item 6): the nested unit's OOM policy, set explicitly.  The user manager's DefaultOOMPolicy is stop: the frozen run killed by the OOM killer would stop the whole unit,
# its shell too, and exit_code and the time stamps would not be written (classify_run would see a run without its records).  With continue the shell writes them.  Not one of D0's properties (run_queue.sh
# sets none), so it is a separate item and UNIT_RESOURCES stays D0's (a test compares it with run_queue.sh).
UNIT_OOM_POLICY = '--property=OOMPolicy=continue'
PROBE = HERE/'unit_probe.py'
PROBE_FILES = ('unit_probe_start.json', 'unit_probe_end.json')      # in the run directory: the unit's cgroup and systemd's properties before the start stamp and after the finish stamp (C22.B4-6 (d))
THREADS = 8
CANDIDATE_CLASSES = ('ok', 'handover_only')     # classify_run.py: the classes with a certified map
FINAL, FINAL_RECORD = 'final_native.npz', 'final_native.json'
SEED_LEVEL, SEED_EVENT = 3, 'level_prepared'


def sha_file(path):
    return core.sha_file(path)


def array_sha256(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def slot_name(pair):
    return pair+SLOT_ALIAS_SUFFIX if pair == FROZEN_DEV_PAIR else pair


# ---- the derived pair ---------------------------------------------------------------------------------------------------------------------------
def read_image_members(path):
    """{fixed, moving, initial_u (if there)}: the raw arrays as stored, one key at a time through the loader's hook (tests replace it with a guard); nothing else of the archive is asked for."""
    from bayesreg.io import pair as pair_module
    out = {}
    with pair_module._open_npz(path) as npz:
        for key in IMAGE_MEMBERS:
            out[key] = np.array(npz[key])
        if ZERO_MEMBER in npz:
            out[ZERO_MEMBER] = np.array(npz[ZERO_MEMBER])
    return out


def frozen_weights(grid):
    """The frozen prepare scripts' weights of a grid (F:scripts/prepare_stage_3d_holdout.py:136-138 weights_for; the same expression in prepare_stage_3d_l2r_val.py:57-62, _lpba40.py:104, _t0_tune.py:102)."""
    L = np.prod([a[-1]-a[0] for a in grid.axes])**(1/3)
    return dict(lam=.1/L**2, kappa=.002, strong_mu_nu=1., weak_mu_nu=.001)


def zero_label_metrics(shape):
    """What the frozen worker copies into labels.json as `baseline`: the frozen dice_labels of the zero labels (every Dice None, macro None)."""
    from jd_curl.three_d.nifti import dice_labels
    zero = np.zeros(shape)
    return {k: dice_labels(zero, zero, labels=list(range(1, n+1))) for k, n in (('seg4', 4), ('seg35', 35))}


def derive_pair(pin, out_dir, *, pair):
    """The firewall view written for the frozen worker: out_dir/pair.npz, manifest.json; returns the record of fallback.firewall (keys, sha256 of both files and of every member, what each is)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    source = Path(pin.provenance['pair_path'])
    raw = read_image_members(source)
    for key in IMAGE_MEMBERS:
        if raw[key].shape != pin.grid.shape or not np.array_equal(np.asarray(raw[key], dtype=np.float64), getattr(pin, key)):
            raise ValueError(f'{source}:{key} is not the array load_pair read')
    initial_present = ZERO_MEMBER in raw
    initial = raw[ZERO_MEMBER] if initial_present else np.zeros(pin.grid.shape+(3,))
    if initial.shape != pin.grid.shape+(3,) or np.any(initial != 0):
        raise ValueError(f'{source}:initial_u is not an all-zero field of the grid')
    zero = np.zeros(pin.grid.shape)
    arrays = dict(fixed=raw['fixed'], moving=raw['moving'], initial_u=initial, **{k: zero for k in SEG_KEYS})
    np.savez_compressed(out_dir/'pair.npz', **{k: arrays[k] for k in DERIVED_KEYS})     # lossless; about 1 GB less on a real pair than np.savez (revision 1, review minor 9); gate (1) compares the members' bytes
    name = slot_name(pair)
    m = pin.grid_metadata
    manifest = dict(pair=name, fixed_subject=pin.fixed_subject, moving_subject=pin.moving_subject, shape_zyx=list(m['shape_zyx']), spacing_xyz=list(m['spacing_xyz']), origin_xyz=list(m['origin_xyz']), unit=m['unit'],
                    sources=dict(fixed=dict(sha256=pin.source_sha256['fixed']), moving=dict(sha256=pin.source_sha256['moving'])), pair_sha256=sha_file(out_dir/'pair.npz'), weights=frozen_weights(pin.grid),
                    baseline_label_metrics=zero_label_metrics(pin.grid.shape), preprocessing=f"HE firewall view of {source} (sha256 {pin.provenance['pair_sha256']}): fixed, moving, initial_u byte for byte, the label members zero",
                    labels_policy='no labels: the label members are zero; labels.json and smoothness.json of this run are not evaluations (the evaluation unit computes them from disp.npz)')
    (out_dir/'manifest.json').write_text(json.dumps(manifest, indent=1, default=float)+'\n')
    record = dict(pair_npz=str(out_dir/'pair.npz'), pair_npz_sha256=manifest['pair_sha256'], manifest=str(out_dir/'manifest.json'), manifest_sha256=sha_file(out_dir/'manifest.json'), keys=list(DERIVED_KEYS),
                  members={k: dict(sha256=array_sha256(arrays[k]), dtype=str(arrays[k].dtype), shape=list(arrays[k].shape), role='zero, so that the frozen worker runs (labels.json, smoothness.json)' if k in SEG_KEYS else
                                   ('the original member, byte for byte' if k != ZERO_MEMBER or initial_present else 'zero: the original has no initial_u')) for k in DERIVED_KEYS},
                  seg_keys_zero=list(SEG_KEYS), source_pair=str(source), source_pair_sha256=pin.provenance['pair_sha256'], source_manifest_sha256=pin.provenance['manifest_sha256'], source_keys_read=list(raw),
                  initial_u_present=initial_present, slot_pair=name, weights=manifest['weights'], lam_hex=float(manifest['weights']['lam']).hex(),
                  weights_rule='F:scripts/prepare_stage_3d_holdout.py:136-138 weights_for(grid); the original manifest\'s weights are not read here (not in load_pair\'s whitelist): the lead compares lam_hex with them outside the method (smoke list)')
    return record


def firewall_check(pin, derived_dir, run_dir=None, *, pair):
    """C22.5-2 gate (1) on a real pair: the arrays handed to the frozen solver (fixed, moving, initial_u, the grid metadata) against the original members, bit for bit; the label members zero; the key set; the
    prepared run's inputs the derived files.  (record, problems).  The original is read through read_image_members only."""
    derived_dir = Path(derived_dir)
    problems = []
    original = read_image_members(pin.provenance['pair_path'])
    with np.load(derived_dir/'pair.npz', allow_pickle=False) as z:
        keys = list(z.files)
        given = {k: np.array(z[k]) for k in keys}
    if keys != list(DERIVED_KEYS):
        problems.append(f'derived keys {keys} != {list(DERIVED_KEYS)}')
    same = {}
    for key in IMAGE_MEMBERS+(ZERO_MEMBER,):
        a = original.get(key)
        b = given.get(key)
        if key == ZERO_MEMBER and a is None:
            same[key] = b is not None and b.dtype == np.float64 and b.shape == pin.grid.shape+(3,) and not np.any(b)
        else:
            same[key] = a is not None and b is not None and a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        if not same[key]:
            problems.append(f'{key}: the derived member is not the original, bit for bit')
    zeros = {k: k in given and not np.any(given[k]) and given[k].shape == pin.grid.shape for k in SEG_KEYS}
    problems += [f'{k}: not an all-zero label member of the grid' for k, ok in zeros.items() if not ok]
    manifest = json.loads((derived_dir/'manifest.json').read_text())
    meta = {k: manifest.get(k) for k in ('shape_zyx', 'spacing_xyz', 'origin_xyz', 'unit')}
    if meta != dict(pin.grid_metadata):
        problems.append(f'grid metadata {meta} != the firewall view {dict(pin.grid_metadata)}')
    if (manifest.get('fixed_subject'), manifest.get('moving_subject'), manifest.get('pair')) != (pin.fixed_subject, pin.moving_subject, slot_name(pair)):
        problems.append('subjects or pair name differ')
    if manifest.get('pair_sha256') != sha_file(derived_dir/'pair.npz'):
        problems.append('the derived manifest does not name the derived pair.npz')
    inputs = None
    if run_dir is not None:
        d = Path(run_dir)/'inputs'/slot_name(pair)
        inputs = {n: sha_file(d/n) if (d/n).is_file() else None for n in ('pair.npz', 'manifest.json')}
        if inputs != {n: sha_file(derived_dir/n) for n in ('pair.npz', 'manifest.json')}:
            problems.append(f'the prepared run inputs {inputs} are not the derived files')
    record = dict(rule='C22.5-2 gate (1): fixed, moving, initial_u and the grid metadata handed to the frozen solver equal the original members bit for bit; the label members are zero; the run reads only the derived files',
                  same=same, label_members_zero=zeros, grid_metadata=meta, prepared_inputs_sha256=inputs, passed=not problems, problems=problems)
    return record, problems


# ---- the private copy and prepare ---------------------------------------------------------------------------------------------------------------
def freeze_hashes():
    return json.loads(CONFIGC_REFERENCE.read_text())['freeze_source_py_sha256']


def frozen_copy(target, frozen=None):
    """jd_curl/ and scripts/ of the frozen tree, __pycache__ left out, checked against the freeze hashes (the rule of experiments/seed/run_seed.py:784-793 frozen_copy).  Returns the number of .py files."""
    frozen = Path(core.FROZEN if frozen is None else frozen)
    target = Path(target)
    for name in ('jd_curl', 'scripts'):
        shutil.copytree(frozen/name, target/name, ignore=shutil.ignore_patterns('__pycache__'))
    live = {str(p.relative_to(target)): sha_file(p) for d in ('jd_curl', 'scripts') for p in (target/d).rglob('*.py')}
    if live != freeze_hashes():
        raise RuntimeError('the private copy of the frozen source differs from the freeze hashes (configC_reference.json)')
    return len(live)


def remove_tree(top):
    """LP-H (decision_log 2026-11-21 ~2:58 PM CST, the LPBA40 feasibility run3 stop): remove the private copy `top` bottom-up by absolute paths -- os.walk (topdown False, links not
    followed; a scandir error raises), os.unlink of each file and link, os.rmdir of each directory, then of `top` -- instead of shutil.rmtree.  shutil.rmtree's fd-based walk
    (shutil._rmtree_safe_fd) opens each subdirectory by its bare name relative to a directory descriptor (os.open(name, O_RDONLY, dir_fd=fd)), and the 'open' audit event carries the bare name
    without the descriptor: an audit-hook spy (hp_core.Spy, lpba_arms.ArmSpy) resolves it against the working directory.  An LPBA process runs from the checkout (C26.9-5; the units'
    --working-directory), so the copy's data/ was judged as <checkout>/data, a forbidden root, and the configuration C cell stopped (spy_breach, exit 20).  Here every directory is listed
    by its absolute path (audit event os.scandir: the run's own directory) and nothing is opened.  Removes the same files shutil.rmtree would when nothing changes the tree meanwhile (it gives
    up rmtree's fd-based guard against a directory swapped for a link during the walk: the private copy lies in the run's own directory and the frozen prepare's child has exited, so no
    one else writes there)."""
    top = os.path.abspath(top)
    if os.path.islink(top) or not os.path.isdir(top):
        raise NotADirectoryError(f'{top}: not a directory (the private copy is a directory, never a link)')

    def fail(exc):
        raise exc
    for root, dirs, files in os.walk(top, topdown=False, onerror=fail):
        for name in files:
            os.unlink(os.path.join(root, name))
        for name in dirs:
            path = os.path.join(root, name)
            if os.path.islink(path):
                os.unlink(path)                     # a link to a directory: the link only (shutil.rmtree's rule)
            else:
                os.rmdir(path)
    os.rmdir(top)


def check_prepared_module():
    return core.load_module('_he_check_prepared', 'experiments/d0/check_prepared.py')


def prepare(derived_dir, run_dir, *, pair, scratch, frozen=None, opts=OPTS, stage_label=STAGE_LABEL, env=None):
    """The frozen `prepare` of configuration C on the derived pair, from a private copy; then check_prepared against configC_reference.json.  Returns the record; RuntimeError on a failed prepare or check."""
    derived_dir, run_dir, scratch = Path(derived_dir), Path(run_dir), Path(scratch)
    name = slot_name(pair)
    copy = scratch/'frozen_copy'
    tick = perf_counter()
    n_files = frozen_copy(copy, frozen)
    slot = copy/'data/three_d'/SLOT_STAGE
    (slot/name).mkdir(parents=True)
    for f in ('pair.npz', 'manifest.json'):
        shutil.copyfile(derived_dir/f, slot/name/f)
    (slot/'selection.json').write_text(json.dumps(dict(pairs=[dict(pair=name)]))+'\n')
    argv = [PY, '-B', str(copy/'scripts/run_stage_3d_e48_pg.py'), 'prepare', str(run_dir), *opts, '--stage', stage_label, '--pair', name]
    child_env = dict(os.environ if env is None else env, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    child_env.pop('PYTHONPATH', None)
    done = subprocess.run(argv, cwd=scratch, env=child_env, capture_output=True, text=True)
    remove_tree(copy)                                   # LP-H: not shutil.rmtree (its dir_fd opens are judged against the working directory by an audit-hook spy)
    if done.returncode:
        raise RuntimeError(f'the frozen prepare failed (exit {done.returncode}): {done.stderr[-1500:]}')
    checked = check_prepared_module().check(run_dir, name, stage_label, sha_file(derived_dir/'pair.npz'), sha_file(derived_dir/'manifest.json'))
    (run_dir/'prepared_check.json').write_text(json.dumps(checked, indent=1)+'\n')
    if not checked['ok']:
        raise RuntimeError(f"check_prepared: {checked['problems']}")
    return dict(argv=argv, slot=f'data/three_d/{SLOT_STAGE}/{name} of a private copy of {frozen or core.FROZEN} (removed after prepare)', frozen_files=n_files, check=checked, seconds=perf_counter()-tick,
                stdout_tail=done.stdout[-500:])


# ---- launch --------------------------------------------------------------------------------------------------------------------------------------
def inner_command(run, python_argv, watchdog=WATCHDOG_SECONDS):
    """experiments/d0/run_queue.sh build_sd_cmd's inner string: the supervisor's time stamps, the watchdog, run.log, exit_code."""
    run = str(run)
    cmd = ' '.join(shlex.quote(a) for a in python_argv)
    return (f"printf '%s\\n' \"$(date -u +%FT%TZ)\" > {shlex.quote(run+'/supervisor_started_utc.txt')}; timeout --signal=TERM --kill-after={KILL_GRACE}s {watchdog}s {cmd} > {shlex.quote(run+'/run.log')} 2>&1; "
            f"ec=$?; printf '%s\\n' \"$ec\" > {shlex.quote(run+'/exit_code')}; printf '%s\\n' \"$(date -u +%FT%TZ)\" > {shlex.quote(run+'/supervisor_finished_utc.txt')}; exit $ec")


def probe_command(run, when, name):
    """unit_probe.py in the unit (python -I -B: the standard library only), its output and errors to its own file; never a reason for the shell to stop."""
    return ' '.join(shlex.quote(a) for a in (PY, '-I', '-B', str(PROBE), '--when', when, '--out', str(Path(run)/name)))+' > /dev/null 2>&1 || true'


def probed_inner_command(run, python_argv, watchdog=WATCHDOG_SECONDS):
    """inner_command, unchanged between its first and last statement, with the unit probe before the start stamp and after the finish stamp (so classify_run's whole wall and the watchdog are inner_command's):
    <run>/unit_probe_start.json, unit_probe_end.json.  The end probe's memory.peak is the cgroup's peak up to then: the nested unit's (UnitLauncher), or the supervisor's own unit (DirectLauncher)."""
    inner = inner_command(run, python_argv, watchdog)
    body, tail = inner.rsplit('; exit $ec', 1)
    if tail:
        raise ValueError('inner_command does not end with its exit')
    return f"{probe_command(run, 'start', PROBE_FILES[0])}; {body}; {probe_command(run, 'end', PROBE_FILES[1])}; exit $ec"


def frozen_env(cache):
    """The unit's environment of a configuration C run (run_queue.sh build_sd_cmd)."""
    cache = Path(cache)
    return dict(OPENBLAS_NUM_THREADS=str(THREADS), OMP_NUM_THREADS=str(THREADS), MKL_NUM_THREADS=str(THREADS), NUMEXPR_NUM_THREADS=str(THREADS), NUMBA_NUM_THREADS=str(THREADS), PYTHONDONTWRITEBYTECODE='1',
                JD_CURL_JIT_CACHE=str(cache/'numba_jit'), JD_CURL_FIRST_USE_CACHE=str(cache/'first_use'))


def frozen_run_argv(run):
    return [PY, '-B', str(Path(run)/'snapshot/scripts/run_stage_3d_e48_pg.py'), 'run', str(run)]


def check_launch(argv):
    """No start string holds the word of the test runner (run_mc0.check_launch's rule)."""
    if 'pytest' in ' '.join(str(a) for a in argv):
        raise ValueError(f'start string with a forbidden word: {argv}')
    return argv


def probes_of(run):
    """The two probe records of a run directory (None for a file that is not there or not JSON)."""
    out = {}
    for name in PROBE_FILES:
        try:
            out[name] = json.loads((Path(run)/name).read_text())
        except (OSError, ValueError):
            out[name] = None
    return out


class UnitLauncher:
    """The D0 unit: systemd-run --user --wait --collect, the frozen run in its snapshot; HE2: OOMPolicy=continue and the unit probe (H2, C22.B4-6 (d))."""
    kind = 'unit'

    def __init__(self, cache, unit_prefix='tag-he-c'):
        self.cache, self.unit_prefix = Path(cache), unit_prefix

    def argv(self, run, short):
        env = frozen_env(self.cache)
        return check_launch(['systemd-run', '--user', f'--unit={self.unit_prefix}-{short}', '--wait', '--collect', f'--working-directory={Path(run)/"snapshot"}', *UNIT_RESOURCES, UNIT_OOM_POLICY,
                             *[f'--setenv={k}={v}' for k, v in env.items()], '/bin/bash', '-c', probed_inner_command(run, frozen_run_argv(run))])

    def __call__(self, run, short):
        argv = self.argv(run, short)
        (Path(run)/'cmd.txt').write_text(shlex.join(argv)+'\n')
        done = subprocess.run(argv, capture_output=True, text=True)
        return dict(kind=self.kind, unit=f'{self.unit_prefix}-{short}', argv=argv, returncode=done.returncode, stderr_tail=done.stderr[-500:], probes=probes_of(run))


class DirectLauncher:
    """The same inner command without systemd: when the supervisor already runs in a unit of its own (the frozen run then shares that unit's cgroup; the probe reads that unit)."""
    kind = 'direct'

    def __init__(self, cache):
        self.cache = Path(cache)

    def argv(self, run, short):
        return check_launch(['/bin/bash', '-c', probed_inner_command(run, frozen_run_argv(run))])

    def __call__(self, run, short):
        argv = self.argv(run, short)
        (Path(run)/'cmd.txt').write_text(shlex.join(argv)+'\n')
        env = dict(PATH=os.environ.get('PATH', '/usr/bin:/bin'), HOME=os.environ.get('HOME', ''), **frozen_env(self.cache))
        done = subprocess.run(argv, cwd=Path(run)/'snapshot', env=env, capture_output=True, text=True)
        return dict(kind=self.kind, argv=argv, returncode=done.returncode, stderr_tail=done.stderr[-500:], probes=probes_of(run))


def dmesg_counts(since_uptime):
    """run_queue.sh dmesg_since: the dxg allocation failures and the OOM lines since the launch (None when dmesg cannot be read)."""
    done = subprocess.run(['dmesg'], capture_output=True, text=True)
    if done.returncode:
        return None, None
    import re
    stamp = re.compile(r'^\[ *([0-9.]+)\]')

    def count(pattern):
        n = 0
        for line in done.stdout.splitlines():
            m = stamp.match(line)
            if m and float(m.group(1)) >= since_uptime and re.search(pattern, line):
                n += 1
        return n
    return count('create_allocation failed'), count('Out of memory|oom-kill|Killed process')


def uptime():
    return float(Path('/proc/uptime').read_text().split()[0])


def classify(run, *, dxg, oom, watchdog=WATCHDOG_SECONDS):
    """experiments/d0/classify_run.py on the run directory, unchanged; writes classification.json (exclusive)."""
    done = subprocess.run([PY, '-B', str(core.ROOT/'experiments/d0/classify_run.py'), str(run), '--watchdog', str(watchdog), '--dxg', str(dxg), '--oom', str(oom), '--out', str(Path(run)/'classification.json')],
                          capture_output=True, text=True)
    record = json.loads((Path(run)/'classification.json').read_text()) if (Path(run)/'classification.json').is_file() else None
    if record is None:
        return dict(**{'class': 'unsure'}, kind='classifier_error', reason=done.stderr[-500:])
    return {k: record.get(k) for k in ('class', 'kind', 'reason', 'script_sha256')}


# ---- the candidates (D75) -------------------------------------------------------------------------------------------------------------------------
def task_dir(run):
    protocol = json.loads((Path(run)/'protocol.json').read_text())
    return Path(run)/protocol['task']['name']


def _passed(cert):
    """The rule of a passing certificate record (C07-3: certified_feasible, lower bound > 0, no infeasible, no uncertain cell); he_core.passes', not imported here."""
    lower = (cert or {}).get('min_jd_lower_bound')
    return bool(isinstance(cert, dict) and cert.get('status') == 'certified_feasible' and lower is not None and lower > 0 and cert.get('n_infeasible') == 0 and cert.get('n_uncertain') == 0)


def candidates(run, classification):
    """[(file name, path, record)] in the order of D75: final_native.npz (its sha256 the one final_native.json names), then the native seed of levels.json (the last level-3 level_prepared event whose transfer
    certificate passes by the C07-3 rule, its sha256 the event's).  Nothing for a class without a certified map.  Each is certified again by the caller before it is output."""
    if classification.get('class') not in CANDIDATE_CLASSES:
        return []
    task = task_dir(run)
    out = []
    final = task/FINAL_RECORD
    if final.is_file() and (task/FINAL).is_file():
        record = json.loads(final.read_text())
        out.append((FINAL, task/FINAL, dict(recorded_sha256=record.get('sha256'), file_sha256=sha_file(task/FINAL), role=record.get('role'), run_certificate=record.get('certificate'))))
    levels = task/'levels.json'
    if levels.is_file():
        events = json.loads(levels.read_text())
        seeds = [e for e in events if e.get('event') == SEED_EVENT and e.get('level') == SEED_LEVEL and _passed(e.get('transfer_certificate'))]
        if seeds:
            e = seeds[-1]
            path = task/e['path']
            if path.is_file():
                out.append((e['path'], path, dict(recorded_sha256=e.get('sha256'), file_sha256=sha_file(path), role='the certified native seed (levels.json level 3 level_prepared)', transfer_certificate=e.get('transfer_certificate'))))
    return out


def load_candidate(path, record, grid):
    """u of a candidate file after its checks: the file's sha256 the recorded one, float64 (nz, ny, nx, 3) of the grid (the caller asks he_core.field_problem too: finite, zero faces).  ValueError otherwise."""
    if record.get('recorded_sha256') is not None and record['recorded_sha256'] != record['file_sha256']:
        raise ValueError(f'{path}: sha256 differs from the run record')
    with np.load(path, allow_pickle=False) as z:
        u = np.ascontiguousarray(np.array(z['u']))
    if u.dtype != np.float64 or u.shape != grid.shape+(3,):
        raise ValueError(f'{path}: u is {u.dtype} {u.shape}, not a float64 field of the grid')
    return u
