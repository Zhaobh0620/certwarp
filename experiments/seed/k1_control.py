"""SEED's control (runs/20261104-seed/任务书.md, "K = 1 控制"): on a small synthetic volume, on the CPU, the frozen worker with the production options of configuration C (throughput v41, the worker_options of a B1TM protocol, analytic_grad and
parallel_setup on) runs
    A  plain: the frozen pre-optimizer computes its field;
    B  the field that the frozen pre-optimizer computes, written to a file and injected by inject_run.py, charge 0;
    C  B with the clock held for 2 s: the charge changes no number;
    D  the same field with one interior component changed by 1e-9 mm: the comparison can fail;
    E  a budget equal to the charge: the charge comes out of the task clock.
A and B must be the same bit for bit: the final field, every commit, every accepted row of the history without its timings, the transfer attempts, the seed.  That is the proof that the wrapper touches nothing else.

    PYTHONPATH=<frozen> python -B experiments/seed/k1_control.py --scratch DIR --out JSON

Stops after a fixed number of outer steps (the frozen module constant OUTER_STEPS, as the frozen tests do), so that the wall clock decides nothing in A, B, C, D.  No real image, label or field.  Exit 0 when every check holds.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))     # before the libraries start their pools: a small test, the frozen worker sets its own thread counts; the machine may be training

import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE))
sys.path.insert(2, str(HERE.parents[1]/'experiments/d0'))

import jd_curl  # noqa: E402
import inject_run  # noqa: E402

FROZEN = Path('/home/rmcfz/seq_vp_tag_results/frozen_eb808be')
OUTER_STEPS = 3
TIMING = ('seconds', 'utc', 'elapsed', 'time', 'tick', 'duration')
T73 = 'image_shift_ggn_equilibrated_jd_affine_v3'
SHAPE_XYZ = (38, 36, 34)                        # at least 34 nodes per axis: the 8x level of the frozen pyramid needs 5


def strip(value):
    """Without the timings; the rest stays (eval ids are content hashes)."""
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if not any(t in k for t in TIMING) and k != 'preoptimizer'}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def synthetic():
    from jd_curl.three_d.grid import make_grid_3d
    from jd_curl.three_d.nifti import dice_labels
    nx, ny, nz = SHAPE_XYZ
    grid = make_grid_3d(nx=nx, ny=ny, nz=nz, hx=1., hy=1., hz=1., x0=0., y0=0., z0=0., unit='mm')
    rng = np.random.default_rng(20261104)
    fixed = gaussian_filter(rng.uniform(size=grid.shape), 2.)
    moving = np.roll(fixed, 1, axis=2)
    fixed, moving = ((a-a.min())/(a.max()-a.min()) for a in (fixed, moving))
    labels = (np.arange(np.prod(grid.shape)).reshape(grid.shape) % 5).astype(float)
    arrays = dict(fixed=fixed, moving=moving, initial_u=np.zeros(grid.shape+(3,)))
    for kind in ('seg4', 'seg35'):
        arrays['fixed_'+kind] = labels
        arrays['moving_'+kind] = np.roll(labels, 1, axis=0)
    baseline = {k: dice_labels(arrays['fixed_'+k], arrays['moving_'+k], labels=list(range(1, n+1))) for k, n in (('seg4', 4), ('seg35', 35))}
    return arrays, grid, dict(weights={'lam': .001}, pair_sha256='synthetic', baseline_label_metrics=baseline)


def read(path):
    return json.loads(Path(path).read_text())


def run_summary(out):
    """What the two runs must share."""
    levels = read(out/'levels.json')
    done = next(e for e in levels if e['event'] == 'preoptimizer_done')
    attempts = [strip({k: e[k] for k in ('alpha', 'certificate_passed', 'certificate')}) for e in levels if e['event'] == 'transfer_attempt']
    with np.load(out/'final_native.npz') as z:
        final = z['u'].tobytes()
    summary = read(out/'summary.json')
    return dict(final_sha256=hashlib.sha256(final).hexdigest(), raw_hash=done['preoptimizer']['raw_hash'], seed_hash=done['preoptimizer']['seed_hash'], taper_L=done['preoptimizer']['taper_L'], attempts=attempts,
                commits=strip(read(out/'commits.json')), history=strip(read(out/'history.json')), inner=strip(read(out/'inner_attempts.json')), native_commits=summary['native_commits'], status=summary['status'], metrics_role=summary['metrics_role'],
                policy=done['preoptimizer']['policy'], injected_seconds=done['preoptimizer'].get('seconds'), level_results=[strip({k: v for k, v in e.items() if k in ('level', 'status', 'formal_commits', 'GF', 'final_mse', 'mse_ratio')}) for e in read(out/'level_results.json')])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--scratch', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args(argv)
    if FROZEN.resolve() not in Path(jd_curl.__file__).resolve().parents:
        raise SystemExit(f'jd_curl comes from {jd_curl.__file__}; run with PYTHONPATH={FROZEN}')
    from jd_curl.three_d import preopt
    from scripts import run_stage_3d_e19_mri as run
    from scripts import run_stage_3d_e48_pg as e48
    original = preopt.preoptimize
    tick = time.perf_counter()
    arrays, grid, manifest = synthetic()
    run.e11.run.inputs = lambda: (arrays, grid, manifest)      # the frozen tests' fixture (tests/three_d/test_e19_mri.py data)
    run.OUTER_STEPS = OUTER_STEPS
    a.scratch.mkdir(parents=True, exist_ok=True)
    pre = e48.parse_pre_optimizer('fp32-v1:iterations=4/4/4/6,lr=0.5,cap=0.5,sigma_grad=1,sigma_field=1,mu=1/1/0.01/0.003,alpha=1e-3/1e-3/1e-3/0,kappa_pre=10,j0=0.2,seconds=60,taper=4')
    cfg = dict(throughput='v41', first_level=0, lean_commits=True, pg_metric_policy=None, pg_shift_factor=1., export_reserve_seconds=1., resolution_exit=True, level_shares=[.020833, .0625, .145833, 1.],
               geometry_weight_levels=[1., 1., .1, .01], curvature_weight_levels=[1e-3, 1e-3, 1e-3, 0.], pg_metric_policy_levels=['legacy_q1_jd_affine_always_v1', 'legacy_q1_jd_affine_always_v1', T73, T73], pre_optimizer=pre)
    options = e48.worker_options(cfg)                          # the production options: the frozen function that `run` calls
    task = dict(next(t for t in run.tasks() if t['hierarchy'] == 'four' and t['strength'] == 'weak'), wall_seconds=3600)
    e48.enable_jit_cache('v41')

    def worker(name, task=task):
        out = a.scratch/name
        run.worker(task, out, a.scratch, device='cpu', task_started=None, host_storage=None, **options)
        return out

    results = {}
    plain = run_summary(worker('A_plain'))
    # the field the frozen pre-optimizer computes: the same call as the worker's callable (F/scripts/run_stage_3d_e19_mri.py:1892-1894)
    config = preopt.PreoptConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pre.items() if k not in ('policy', 'taper')})
    profile = run.THROUGHPUT_POLICIES['v41']
    execution = preopt.PreoptExecution(lean_sync=bool(profile.get('preopt_lean_sync')), static_grid=bool(profile.get('preopt_static_grid')), cuda_graph=bool(profile.get('preopt_cuda_graph')),
                                       parallel_setup=bool(profile.get('preopt_parallel_setup')), analytic_grad=bool(profile.get('preopt_analytic_grad')))
    raw, record = original(arrays['fixed'], arrays['moving'], grid, config, device='cpu', execution=execution)
    raw = np.array(raw)
    field_path = a.scratch/'field.npz'
    np.savez(field_path, u=raw)

    def entry(path, u, charge):
        return dict(arm='control', pair='synthetic', field_path=str(path), field_sha256=inject_run.sha_file(path), u_sha256=hashlib.sha256(u.tobytes()).hexdigest(), source_run_id='control', oracle=False, charge_seconds=charge, shape_zyx=list(grid.shape))

    def injected(name, path, u, charge, task=task):
        record = entry(path, u, charge)
        function = inject_run.install(preopt, record, inject_run.load_field(record))
        try:
            return run_summary(worker(name, task)), len(function.calls), record
        finally:
            preopt.preoptimize = original

    same, calls, _ = injected('B_injected', field_path, raw, 0.)
    held, calls_held, _ = injected('C_charge', field_path, raw, 2.)
    other = raw.copy()
    other[grid.shape[0]//2, grid.shape[1]//2, grid.shape[2]//2, 0] += 1e-9
    other_path = a.scratch/'field_other.npz'
    np.savez(other_path, u=other)
    changed, _, _ = injected('D_changed', other_path, other, 0.)
    wall = 8.
    record_e = entry(field_path, raw, wall)
    function = inject_run.install(preopt, record_e, inject_run.load_field(record_e))
    out_e = a.scratch/'E_budget'
    run.worker(dict(task, wall_seconds=wall), out_e, a.scratch, device='cpu', task_started=None, host_storage=None, **options)
    preopt.preoptimize = original
    summary_e = read(out_e/'summary.json')
    levels_e = read(out_e/'level_results.json')

    tapered, _ = preopt.boundary_taper(raw, grid, pre['taper'])
    twice, _ = preopt.boundary_taper(tapered, grid, pre['taper'])
    sha = lambda x: hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
    sys.path.insert(0, str(HERE))
    import run_seed
    # the attributes that decide what the comparison means
    keys = ('final_sha256', 'raw_hash', 'seed_hash', 'taper_L', 'attempts', 'commits', 'history', 'inner', 'native_commits', 'status', 'metrics_role', 'level_results')
    checks = dict(
        production_execution=bool(execution.parallel_setup and execution.analytic_grad and not execution.lean_sync),
        field_of_the_plain_run_is_the_field_injected=plain['raw_hash'] == sha(raw),
        plain_used_the_frozen_function=plain['policy'] == 'fp32-preopt-v1',
        injected_ran_once=calls == 1 and same['policy'] == inject_run.POLICY,
        bitwise_equal_to_plain={k: plain[k] == same[k] for k in keys},
        charge_changes_no_number={k: plain[k] == held[k] for k in keys},
        charge_holds_the_clock=calls_held == 1 and held['injected_seconds'] >= 2.,
        negative_control_differs=changed['final_sha256'] != plain['final_sha256'] and changed['raw_hash'] != plain['raw_hash'],
        moved=plain['native_commits'] >= 1 and bool(np.abs(raw).max() > .01),
        taper_once=plain['seed_hash'] == sha(tapered) and same['seed_hash'] == sha(tapered),
        taper_once_is_not_taper_twice=sha(tapered) != sha(twice) and sha(tapered) != sha(raw),
        replica_of_the_taper_in_run_seed=run_seed.boundary_taper_sha256(raw, grid.shape, pre['taper']) == sha(tapered),
        charge_comes_out_of_the_budget=len(function.calls) == 1 and summary_e['native_commits'] == 0 and summary_e['metrics_role'] != 'formal_native_outer_commit')
    flat = [all(v.values()) if isinstance(v, dict) else bool(v) for v in checks.values()]
    result = dict(bitwise=all(flat), device='cpu', profile='v41', outer_steps=OUTER_STEPS, shape_zyx=list(grid.shape), checks=checks, native_commits=plain['native_commits'], max_abs_u=float(np.abs(raw).max()),
                  accepted_alpha_attempts=[x['alpha'] for x in plain['attempts']], status_of_E=summary_e['status'], level_E=[e.get('status') for e in levels_e], seconds=time.perf_counter()-tick, seconds_injected_with_charge=held['injected_seconds'],
                  record_of_the_frozen_preoptimizer={k: record[k] for k in ('policy', 'device', 'dtype')}, config=asdict(config), not_compared='the GPU path: the same callable and the same injection; the queue checks raw_hash and seed_hash of every real run')
    text = json.dumps(result, indent=1, default=float)
    print(text)
    if a.out:
        a.out.write_text(text+'\n')
    return 0 if result['bitwise'] else 8


if __name__ == '__main__':
    sys.exit(main())
