"""Synthetic pairs, fields and run directories for the bayesreg.eval tests (tmp_path only; no data/ access).

Subjects: synthetic canonical OASIS ids OASIS_OAS1_09xx_MR1 (no real subject has an id above 0457); LPBA40 S01, S02.
Sealed subjects used by the guard tests are taken from the pinned sealed lists at run time (sealed_id), never written
into the test source (review EV item 12).
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from jd_curl.three_d.nifti import file_sha256

REPO = Path(__file__).resolve().parents[3]
SHAPE = (18, 20, 22)                       # (nz, ny, nx)
DEV = ("OASIS_OAS1_0901_MR1", "OASIS_OAS1_0902_MR1", "OASIS_OAS1_0903_MR1", "OASIS_OAS1_0904_MR1")
DEV_LPBA40 = ("S01", "S02")
LPBA56 = tuple(range(21, 75))+(181, 182)   # 56 synthetic LPBA40 label ids, 181 and 182 among them
STAGE = "stage_synth_v1"


def sealed_id(name):
    """The smallest canonical OASIS id of the sealed list `name` (holdout_B_v1, H1, H4), read at run time."""
    from bayesreg.eval.truth import load_sealed_lists
    return min(s for s in load_sealed_lists().subjects[name] if s.startswith("OASIS_"))


def load_module(name, path, package=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if package is not None:
        module.__package__ = package
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def l2r_script():
    return load_module("_eval_test_evaluate_l2r_style", REPO/"scripts"/"evaluate_l2r_style.py")


def archived_smoothness():
    path = REPO/"archive"/"seq_vp_3d"/"jd_curl"/"three_d"/"smoothness.py"
    return load_module("jd_curl.three_d._eval_test_archived_smoothness", path, package="jd_curl.three_d")


def labels(seed, shape=SHAPE, drop=(), n_labels=35):
    """seg35-like labels: Voronoi cells of random seeds inside an ellipsoid (0 outside); labels in `drop` set to 0."""
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    inside = ((z-(nz-1)/2)/(nz*.42))**2+((y-(ny-1)/2)/(ny*.42))**2+((x-(nx-1)/2)/(nx*.42))**2 <= 1
    seeds = np.stack([rng.uniform(2, n-3, n_labels) for n in shape], axis=1)
    pts = np.stack([z, y, x], axis=-1).astype(float)
    dist = ((pts[..., None, :]-seeds)**2).sum(-1)
    lab = (np.argmin(dist, axis=-1)+1)*inside
    for label in drop:
        lab[lab == label] = 0
    return lab.astype(np.float64)


def seg4_of(seg35):
    return np.where(seg35 > 0, (seg35.astype(int) % 4)+1, 0).astype(np.float64)


def smooth_d(shape=SHAPE, amplitude=(1.2, -.9, .7), freq=1):
    """(3, nz, ny, nx) voxels, zero on the six faces."""
    t = [np.linspace(0., 1., n) for n in shape]
    z, y, x = np.meshgrid(*t, indexing="ij")
    bump = np.sin(np.pi*z)*np.sin(np.pi*y)*np.sin(np.pi*x)
    wave = np.cos(2*np.pi*freq*(x+.5*y))
    d = np.stack([a*bump*(1.+.3*k*wave) for k, a in enumerate(amplitude)])
    d[:, [0, -1]] = 0.                          # sin(pi) is not exactly 0 in floating point
    d[:, :, [0, -1]] = 0.
    d[:, :, :, [0, -1]] = 0.
    return d


def folding_d(shape=SHAPE):
    """A field whose central-difference det is <= 0 somewhere (and whose Q1 map folds)."""
    d = smooth_d(shape, amplitude=(0., 0., 0.))
    nz, ny, nx = shape
    d[2, nz//2, ny//2, nx//2-1] = 2.5           # x-displacement of two neighbours swapped across the centre
    d[2, nz//2, ny//2, nx//2+1] = -2.5
    return d


def pair_arrays(seed=0):
    fixed35 = labels(seed)
    moving35 = np.roll(labels(seed, drop=(7,)), 1, axis=2)       # label 7 missing in the moving image (NaN rule)
    fixed35[fixed35 == 33] = 0
    moving35[moving35 == 33] = 0                                 # label 33 absent from both
    rng = np.random.default_rng(seed+100)
    return dict(fixed=rng.random(SHAPE), moving=rng.random(SHAPE), fixed_seg35=fixed35, moving_seg35=moving35,
                fixed_seg4=seg4_of(fixed35), moving_seg4=seg4_of(moving35), initial_u=np.zeros(SHAPE+(3,)))


def lpba40_pair_arrays(seed=0):
    """LPBA40-like pair: lpba56 labels (ids LPBA56) and a brain mask; one label missing in the moving image."""
    ids = np.array((0,)+LPBA56, dtype=np.float64)
    fixed = ids[labels(seed, n_labels=56).astype(int)]
    moving = ids[np.roll(labels(seed, drop=(9,), n_labels=56), 1, axis=2).astype(int)]
    rng = np.random.default_rng(seed+200)
    return dict(fixed=rng.random(SHAPE), moving=rng.random(SHAPE), fixed_lpba56=fixed, moving_lpba56=moving,
                fixed_brain=(fixed > 0).astype(np.float64), moving_brain=(moving > 0).astype(np.float64))


def write_pair(data_root, pair, fixed, moving, arrays=None, stage=STAGE, spacing=(1., 1., 1.), label_sets=None):
    """data_root/stage/pair/{pair.npz, manifest.json} in the mainline format."""
    folder = Path(data_root)/stage/pair
    folder.mkdir(parents=True)
    arrays = pair_arrays() if arrays is None else arrays
    np.savez_compressed(folder/"pair.npz", **arrays)
    manifest = dict(shape_zyx=list(SHAPE), spacing_xyz=list(spacing), origin_xyz=[0., 0., 0.], unit="mm",
                    pair_sha256=file_sha256(folder/"pair.npz"), fixed_subject=fixed, moving_subject=moving,
                    sources=dict(fixed=dict(sha256="a"*64), moving=dict(sha256="b"*64),
                                 fixed_seg35=dict(sha256="c"*64)),
                    baseline_label_metrics=dict(seg35=.5))
    if label_sets is not None:
        manifest["label_sets"] = label_sets
    (folder/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def write_allowlist(path, subjects=DEV, lpba40=()):
    sets = dict(OASIS=list(subjects))
    sources = [dict(name="synthetic_dev", dataset="OASIS", path="tests/synthetic_dev.json", sha256="0"*64,
                    never_for_training=False, subjects=list(subjects))]
    if lpba40:
        sets["LPBA40"] = list(lpba40)
        sources.append(dict(name="synthetic_lpba40", dataset="LPBA40", path="tests/synthetic_lpba40.json",
                            sha256="1"*64, never_for_training=False, subjects=list(lpba40)))
    document = dict(schema="bayesreg.dev_subjects.v1", subjects=sets, sources=sources)
    Path(path).write_text(json.dumps(document), encoding="utf-8")
    return Path(path)


def d_to_u(d, spacing=(1., 1., 1.)):
    hx, hy, hz = spacing
    u = np.empty(d.shape[1:]+(3,))
    u[..., 2], u[..., 1], u[..., 0] = d[0]*hz, d[1]*hy, d[2]*hx
    return u


def write_configc_run(run_dir, pair, u, arrays=None, spacing=(1., 1., 1.), cert=None, *,
                      metrics_role="formal_native_outer_commit", exit_code=0, complete=True, run_id=None,
                      input_sha256=None, pair_dataset=None, subjects=None):
    """A configuration C run directory as seq_vp_3d writes it (the inputs copy is what the L2R script reads), with the
    supervisor's completion records (exit_code, completion.json, task/worker_completion.json, task/summary.json).
    exit_code None: still running (no exit_code file); complete False: worker_completion.json complete = false.
    subjects: (fixed, moving) written into the inputs manifest (C23.9-13 (11): the synthetic ids the L2R script checks
    before it reads the labels, C23.6-2 revision 6d); None: no ids, as before."""
    run_dir = Path(run_dir)
    (run_dir/"inputs"/pair).mkdir(parents=True)
    (run_dir/"task").mkdir()
    arrays = pair_arrays() if arrays is None else arrays
    np.savez_compressed(run_dir/"inputs"/pair/"pair.npz", **arrays)
    manifest = dict(shape_zyx=list(SHAPE), spacing_xyz=list(spacing))
    if subjects is not None:
        manifest.update(fixed_subject=subjects[0], moving_subject=subjects[1])
    (run_dir/"inputs"/pair/"manifest.json").write_text(json.dumps(manifest))
    protocol = dict(task=dict(name="task", pair=pair))
    if input_sha256 is not None:
        protocol["input_sha256"] = {"pair.npz": input_sha256}
    if pair_dataset is not None:
        protocol["pair_dataset"] = pair_dataset
    (run_dir/"protocol.json").write_text(json.dumps(protocol))
    np.savez(run_dir/"task"/"final_native.npz", u=u)
    digest = hashlib.sha256(np.ascontiguousarray(u).tobytes()).hexdigest()
    cert = dict(status="certified_feasible", min_jd_lower_bound=.5, n_infeasible=0, n_uncertain=0,
                state_hash=digest) if cert is None else cert
    (run_dir/"task"/"final_native.json").write_text(json.dumps(dict(certificate=cert)))
    if exit_code is not None:
        (run_dir/"task"/"summary.json").write_text(json.dumps(dict(status="native_budget_exhausted",
                                                                  metrics_role=metrics_role)))
        (run_dir/"task"/"worker_completion.json").write_text(json.dumps(dict(complete=complete)))
        (run_dir/"completion.json").write_text(json.dumps(dict(status="native_budget_exhausted")))
        (run_dir/"exit_code").write_text(f"{exit_code}\n")
    if run_id is not None:
        (run_dir/"run_id").write_text(run_id+"\n")
    return run_dir


def write_bayesreg_run(run_dir, pair, u, *, stage=STAGE, spacing=(1., 1., 1.), with_disp=True,
                       channel_source="tissue", native_commits=3, pair_provenance=None, status_extra=None,
                       provenance=None):
    """A bayesreg.cli.run_pair run directory (exported with bayesreg.solve.export's writers). hyper.json rounds: one per
    level 0..3, the native one with `native_commits` formal commits (None: no hyper.json); pair_provenance: data.txt's
    pair_provenance (None: no data.txt); status_extra: more status.json fields (e.g. structure_registry);
    provenance: provenance.json (None: none)."""
    from jd_curl.three_d.grid import make_grid_3d
    from bayesreg.solve.export import write_disp, write_final_state, write_json
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True)
    grid = make_grid_3d(nx=SHAPE[2], ny=SHAPE[1], nz=SHAPE[0], hx=spacing[0], hy=spacing[1], hz=spacing[2], unit="mm")
    (run_dir/"cmd.txt").write_text("python -m bayesreg.cli.run_pair (synthetic)\n")
    write_json(run_dir/"config.json", dict(args=dict(pair=f"{stage}/{pair}", channel_source=channel_source)))
    write_json(run_dir/"status.json", dict(status="converged_B", exit_code=0 if with_disp else 2, native=True,
                                           **(status_extra or {})))
    if provenance is not None:
        write_json(run_dir/"provenance.json", provenance)
    if native_commits is not None:
        write_json(run_dir/"hyper.json", dict(rounds=[dict(level=level, round=0, commits=5 if level < 3 else native_commits)
                                                      for level in range(4)]))
    if pair_provenance is not None:
        write_json(run_dir/"data.txt", dict(pair=f"{stage}/{pair}", pair_provenance=pair_provenance))
    digest = write_final_state(run_dir/"final_native.npz", u, grid, status="converged_B")
    if with_disp:
        write_disp(run_dir/"disp.npz", u, grid, state_hash=digest)
        write_json(run_dir/"cert.json", dict(status="certified_feasible", passed=True, state_hash=digest,
                                             min_jd_lower_bound=.5, n_infeasible=0, n_uncertain=0))
    return run_dir, grid
