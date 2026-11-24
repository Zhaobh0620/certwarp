"""SEED (runs/20261104-seed/任务书.md, plan.md; docs/bayesreg/05_D0协议.md §16): experiments/seed/, CPU only, small synthetic volumes and files; no solver on a real image, no GPU, no unit.

- the bitwise control (slow): the frozen worker, production options, on a synthetic volume, without and with the injection (subprocess, PYTHONPATH=<frozen>); the end-to-end control (slow): new processes through the real prepare, check_prepared, e48.run, _run_pair;
- the wrapper: refusals before any frozen code; the hand-over to the frozen entry with the sys.path and argv of `python script`; one taper;
- the entry: only from mc0_v1 P2 long and mc0p_v1 Q2 long, the same pair, the right oracle mark, the sha256, the shape;
- the reading: every branch on made-up numbers; premises; pins; the summary;
- the gate and the queue on a fake machine; only the first 5 pairs; the start strings; the frozen source lines this package relies on.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/seed'))

import decide_seed as DEC  # noqa: E402
import inject_run as INJ  # noqa: E402
import run_seed as RUN  # noqa: E402

FROZEN = Path('/home/rmcfz/seq_vp_tag_results/frozen_eb808be')
PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'
BRIEF_PAIRS = ('0438-0439', '0439-0440', '0440-0441', '0441-0442', '0442-0443')
needs_frozen = pytest.mark.skipif(not (FROZEN/'jd_curl/three_d/preopt.py').is_file(), reason='the frozen worktree is not on this machine')
SHAPE = (8, 9, 10)


def small_field(seed=1):
    rng = np.random.default_rng(seed)
    u = np.zeros(SHAPE+(3,))
    u[1:-1, 1:-1, 1:-1] = rng.normal(size=(SHAPE[0]-2, SHAPE[1]-2, SHAPE[2]-2, 3))*.1
    return u


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_results(tmp, arms=('S_P', 'S_T'), entries=('S_P', 'S_T'), controls=True):
    """A synthetic results tree: the inputs of the arms, the entries, b1_v1."""
    results = Path(tmp)/'results'
    for arm in arms:
        spec = RUN.ARMS[arm]
        root = results/spec['inputs']
        files = {}
        for pair in RUN.PAIRS:
            d = root/'stage_L2R_val_v1'/pair
            d.mkdir(parents=True)
            np.savez(d/'pair.npz', fixed=np.zeros(SHAPE), moving=np.zeros(SHAPE))
            fixed, moving = pair.split('__')
            write_json(d/'manifest.json', dict(pair=pair, pair_sha256=sha(d/'pair.npz'), fixed_subject=fixed, moving_subject=moving, shape_zyx=list(SHAPE), preprocessing_d0=dict(
                arm=spec['input_arm'], oracle=spec['oracle'], source_pair_sha256='orig'+pair, source_manifest_sha256='origm'+pair, recipe='synthetic')))
            files[f'stage_L2R_val_v1/{pair}/pair.npz'] = sha(d/'pair.npz')
            files[f'stage_L2R_val_v1/{pair}/manifest.json'] = sha(d/'manifest.json')
        write_json(root/'d0_inputs.json', dict(arm=spec['input_arm'], oracle=spec['oracle'], files_sha256=files))
    for arm in entries:
        for k, pair in enumerate(RUN.PAIRS):
            make_entry(results, arm, pair, seed=k)
    if controls:
        make_controls(results)
    return results


DROP = object()                                                                 # make_entry(..., key=DROP): the record without that key
LONG_CONFIG = dict(iterations=[50, 50, 100, 600], lr=.5, cap=.5, sigma_grad=1., sigma_field=1., mu=[1., 1., .01, .003], alpha=[.001, .001, .001, 0.], kappa_pre=10., j0=.2, beta1=.9, beta2=.99, eps=1e-8, seconds=900.)


def predicted_sources(**changes):
    """channel_sources as run_mc0p.py writes it (the names read on a real mc0p_v1 run.json)."""
    side = dict(segmenter_id='unet35_v1_ens', policy='all_folds_mean')
    return dict(dict(oracle=False, truth_labels_read=False, pair_keys_read=['fixed', 'moving', 'initial_u'], sides=dict(fixed=dict(side), moving=dict(side)), segmenter=dict(), label_firewall='x'), **changes)


def make_entry(results, the_arm, the_pair, seed=1, u=None, **overrides):
    arm, pair = the_arm, the_pair
    spec = RUN.ARMS[arm]
    d = RUN.entry_dir(arm, pair, results)
    d.mkdir(parents=True, exist_ok=True)
    u = small_field(seed) if u is None else u
    np.savez(d/'field.npz', u=u)
    fixed, moving = pair.split('__')
    run = dict(schema=spec['source_schema'], config=dict(LONG_CONFIG), run_id=spec['source_prefix']+RUN.short_name(pair), arm=spec['source_arm'], tier='long', pair=pair, k=35, oracle=spec['oracle'], status='done', seconds_total=46., code_sha='x',
               **({} if spec['oracle'] else dict(channel_sources=predicted_sources())),      # MC0's records have no channel_sources: S_T's entry is one of them
               field=dict(file='field.npz', sha256=sha(d/'field.npz'), u_sha256=hashlib.sha256(u.tobytes()).hexdigest()),
               data=dict(pair_sha256='orig'+pair, manifest_sha256='origm'+pair, fixed_subject=fixed, moving_subject=moving))
    if RUN.entry_root(arm, pair) == RUN.RETRY_ROOT:
        run['retry_of'] = f"/x/{RUN.RETRIED_ROOT}/runs/Q2_long/{RUN.short_name(pair)}"
    run.update(overrides)
    run = {k: v for k, v in run.items() if v is not DROP}
    write_json(d/'run.json', run)
    write_json(d/'metrics.json', dict(tapered_u_sha256=RUN.boundary_taper_sha256(u, u.shape[:3]), metrics=dict(seg35=.86, seg4=.88, sdlogj_official=.05), certificate=dict(accepted_alpha=1.)))
    return d


def per_labels(value):
    return dict(seg35_per_label={str(k): value for k in range(1, 36)}, seg4_per_label={str(k): value for k in range(1, 5)})


def metric_values(seg35=.84, seg4=.9, sdlogj=.08):
    return dict(seg35=seg35, seg4=seg4, dice_official=seg35, hd95=2., sdlogj_official=sdlogj, sdlogj_brain=.6, num_foldings_official=0., pct_J_lt_half_brain=14., cell_centre_J_nonpositive=0, **per_labels(seg35))


CONTROL_MEANS = dict(B1TM=DEC.PINS['B1TM'], B1W_A5=DEC.PINS['B1W_A5'])


def make_controls(results, means=CONTROL_MEANS, configuration=None):
    b1 = Path(results)/'b1_v1'
    rows = []
    for arm, spec in RUN.ARMS.items():
        control = spec['control']
        for pair in RUN.PAIRS:
            m = dict(run_id=f'b1_{control}_{RUN.short_name(pair)}', arm=control, pair=pair, method='b1', metrics=metric_values(*means[control]), certificate=dict(passed=True, min_jd_lower_bound=1e-4), self_check=dict(passed=True), native_commits=85, e2e_seconds=327.,
                     b1=configuration or dict(seconds=RUN.SECONDS, changes=RUN.CHANGES, inputs_root=str(RUN.inputs_root(arm, results))))
            write_json(b1/'runs'/control/pair/'metrics.json', m)
            for key in ('seg35', 'seg4', 'dice_official', 'hd95', 'sdlogj_official', 'sdlogj_brain', 'num_foldings_official', 'pct_J_lt_half_brain', 'cell_centre_J_nonpositive'):
                rows.append(dict(set='L2R_val', split='dev', method='b1', arm=control, pair=pair, measure=key, value=m['metrics'][key], run_id=m['run_id']))
            for key, value in (('cert_passed', 1), ('cert_min_jd_lower_bound', 1e-4), ('self_check_passed', 1), ('native_commits', 85), ('e2e_seconds', 327.)):
                rows.append(dict(set='L2R_val', split='dev', method='b1', arm=control, pair=pair, measure=key, value=value, run_id=m['run_id']))
    (b1/'analysis').mkdir(parents=True, exist_ok=True)
    import csv
    with open(b1/'analysis/long.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['set', 'split', 'method', 'arm', 'pair', 'measure', 'value', 'run_id'])
        w.writeheader()
        w.writerows(rows)


# ---- the bitwise control -----------------------------------------------------------------------------------------------------------------------------------------------------------
@needs_frozen
@pytest.mark.slow
def test_the_end_to_end_control(tmp_path):
    """experiments/seed/e2e_control.py: new processes through the real prepare, check_prepared, e48.run and _run_pair, on the CPU; its patches and what it does not reach are in its docstring."""
    out = tmp_path/'e2e.json'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    done = subprocess.run([PY, '-B', str(ROOT/'experiments/seed/e2e_control.py'), '--scratch', str(tmp_path/'scratch'), '--out', str(out)], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:]+done.stderr[-3000:]
    result = json.loads(out.read_text())
    assert result['passed'] is True and result['native_commits'] >= 1
    assert subprocess.run(['git', '-C', str(FROZEN), 'status', '--porcelain'], capture_output=True, text=True).stdout == ''


@needs_frozen
@pytest.mark.slow
def test_injecting_the_frozen_field_is_bitwise_the_plain_frozen_run(tmp_path):
    """The whole frozen worker with the production options, on the CPU.  Includes: the charge changes no number and comes out of the task clock; one taper; the comparison can fail."""
    started = time.time()-1
    cache = Path(tempfile.gettempdir())/'seed_k1_control_cache'                # content addressed: only the first run compiles
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(FROZEN),
               JD_CURL_JIT_CACHE=str(cache/'jit'), JD_CURL_FIRST_USE_CACHE=str(cache/'first'), NUMBA_CACHE_DIR=str(cache/'numba'))
    out = tmp_path/'k1.json'
    done = subprocess.run([PY, '-B', str(ROOT/'experiments/seed/k1_control.py'), '--scratch', str(tmp_path/'scratch'), '--out', str(out)], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:]+done.stderr[-3000:]
    result = json.loads(out.read_text())
    assert result['bitwise'] is True and result['profile'] == 'v41' and result['native_commits'] >= 1
    for name, value in result['checks'].items():
        assert all(value.values()) if isinstance(value, dict) else value, name
    assert result['status_of_E'] == 'time_budget_exhausted' and result['level_E'][-1] == 'time_budget_before_solver'
    written = [p for p in (FROZEN/'jd_curl').rglob('*') if p.stat().st_mtime > started]
    assert not written, written                                                # the frozen tree stays read only
    assert subprocess.run(['git', '-C', str(FROZEN), 'status', '--porcelain'], capture_output=True, text=True).stdout == ''


# ---- the frozen lines this package relies on --------------------------------------------------------------------------------------------------------------------------------------------
@needs_frozen
def test_the_frozen_source_is_what_the_package_read():
    """If one of these changed the injection would no longer mean what the plan says."""
    lines = (FROZEN/'scripts/run_stage_3d_e19_mri.py').read_text().splitlines()
    assert "__import__('jd_curl.three_d.preopt', fromlist=['preoptimize']).preoptimize(" in lines[1891]
    assert 'first_use_dry' in lines[1890] and 'if first_use_dry is not None else lambda' in lines[1891]
    assert "fixed_img, moving_img, native_grid, preopt_config, device=device," in lines[1892] and 'execution=preopt_execution' in lines[1893] and "preoptimizer_taper=pre_optimizer.get('taper', 4)" in lines[1894]
    assert "deadline = started+task['wall_seconds']-reserve" in (FROZEN/'scripts/run_stage_3d_e19_mri.py').read_text()
    mr = (FROZEN/'jd_curl/three_d/multiresolution.py').read_text().splitlines()
    assert 'raw, pre_record = preoptimizer(fixed, moving, grid)' in mr[391] and 'as_float64_array(raw, "preoptimizer field", grid.shape+(3,))' in mr[392]
    assert 'zero_boundary(raw)' in mr[393] and 'seed, _ = boundary_taper(raw, grid, preoptimizer_taper)' in mr[395]
    assert sum('boundary_taper(' in line and 'def ' not in line for line in mr) == 1      # one taper on the path
    e48 = (FROZEN/'scripts/run_stage_3d_e48_pg.py').read_text().splitlines()
    assert e48[23].startswith('BUDGETS = (30, 45, 60, 90, 120, 180, 240, 330, 600, 3600)')
    assert 'if seconds not in BUDGETS:' in e48[404] and "choices=BUDGETS" in e48[924]
    pre = (FROZEN/'jd_curl/three_d/preopt.py').read_text().splitlines()
    assert 'def preoptimize(fixed_image, moving_image, grid, config=None, device="cuda", observer=None, execution=None):' in pre[505]
    assert 'def boundary_taper(u, grid, L=4):' in pre[616]
    callers = subprocess.run(['grep', '-rln', r'\.preoptimize(', str(FROZEN/'jd_curl'), str(FROZEN/'scripts')], capture_output=True, text=True).stdout.split()
    assert [Path(p).name for p in callers] == ['run_stage_3d_e19_mri.py']       # the one caller


@needs_frozen
def test_the_brief_budget_is_refused_by_the_frozen_parser_and_330_is_not(tmp_path):
    """Why the budget stays 330 s: --seconds 285 never reaches `prepare`."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
    done = subprocess.run([PY, '-B', str(FROZEN/'scripts/run_stage_3d_e48_pg.py'), 'prepare', str(tmp_path/'x'), '--seconds', '285', '--throughput', 'v41'], env=env, capture_output=True, text=True)
    assert done.returncode == 2 and 'invalid choice: 285' in done.stderr and not (tmp_path/'x').exists()
    with pytest.raises(RUN.b1_spec.SpecError):
        RUN.b1_spec.check_entry(dict(arm='S_P', inputs_root='/x', seconds=285, changes=RUN.CHANGES))
    assert RUN.b1_entry('S_P')['seconds'] == 330


def test_the_budget_and_the_charge_add_up_to_b1tm():
    """Native cutoff 329 - 47 s; the total as B1TM's 330 s; within 2 s of what a 285 s budget would leave."""
    assert RUN.SECONDS == 330 and RUN.SECONDS in RUN.b1_spec.BUDGETS and RUN.ENTRY_CHARGE_SECONDS == 47.
    assert RUN.WATCHDOG == 1100
    cutoff = RUN.SECONDS-1-RUN.ENTRY_CHARGE_SECONDS
    assert cutoff == 282 and 0 <= (285-1)-cutoff <= 2


# ---- the wrapper ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
FAKE_ENTRY = '''import hashlib, json, sys
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
if __name__ == '__main__':
    trace = dict(argv=sys.argv, name=__name__, file=__file__, path=sys.path, modules_jd=sorted(m for m in sys.modules if m.startswith('jd_curl')))
    if sys.argv[1] == 'run' and 'PATH_ONLY' not in __import__('os').environ:
        grid = SimpleNamespace(shape=%(shape)r)
        raw, record = __import__('jd_curl.three_d.preopt', fromlist=['preoptimize']).preoptimize(None, None, grid, 'cfg', device='cuda', execution='ex')
        from jd_curl.three_d.preopt import boundary_taper
        seed, _ = boundary_taper(raw, grid, 4)
        trace.update(record=record, raw=hashlib.sha256(raw.tobytes()).hexdigest(), seed=hashlib.sha256(seed.tobytes()).hexdigest(), writeable=bool(raw.flags.writeable), dtype=str(raw.dtype), shape=list(raw.shape))
    Path(sys.argv[2], 'trace.json').write_text(json.dumps(trace))
''' % dict(shape=SHAPE)


def make_fake_run(tmp_path, u=None, **entry_changes):
    """A run directory whose snapshot has the frozen preopt.py and a small script at the frozen entry's place."""
    run = tmp_path/'run'
    snapshot = run/'snapshot'
    for rel in ('jd_curl/__init__.py', 'jd_curl/validation.py', 'jd_curl/three_d/__init__.py', 'jd_curl/three_d/validation.py', 'jd_curl/three_d/preopt.py'):
        (snapshot/rel).parent.mkdir(parents=True, exist_ok=True)
        (snapshot/rel).write_bytes((FROZEN/rel).read_bytes())
    (snapshot/'scripts').mkdir()
    (snapshot/'scripts/run_stage_3d_e48_pg.py').write_text(FAKE_ENTRY)
    u = small_field() if u is None else u
    np.savez(run/'field.npz', u=u)
    entry = dict(arm='S_T', pair=RUN.PAIRS[0], field_path=str(run/'field.npz'), field_sha256=sha(run/'field.npz'), u_sha256=hashlib.sha256(u.tobytes()).hexdigest(), source_run_id='mc0_P2_long_0438-0439', oracle=True, charge_seconds=.05, shape_zyx=list(SHAPE))
    entry.update(entry_changes)
    write_json(run/'seed_entry.json', entry)
    (run/'seed_inject.py').write_bytes((ROOT/'experiments/seed/inject_run.py').read_bytes())
    return run, u, entry


def wrapper(run, **env):
    return subprocess.run([PY, '-B', str(run/'seed_inject.py'), str(run)], cwd=run/'snapshot', env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', **env), capture_output=True, text=True)


@needs_frozen
def test_the_wrapper_hands_over_with_the_given_field_and_tapers_once(tmp_path):
    run, u, entry = make_fake_run(tmp_path)
    done = wrapper(run)
    assert done.returncode == 0, done.stdout+done.stderr
    trace = json.loads((run/'trace.json').read_text())
    assert trace['raw'] == entry['u_sha256'] and trace['dtype'] == 'float64' and trace['shape'] == list(SHAPE)+[3] and trace['writeable'] is False
    assert trace['seed'] == RUN.boundary_taper_sha256(u, SHAPE) and trace['seed'] != trace['raw']
    assert trace['record']['policy'] == INJ.POLICY and trace['record']['source_run_id'] == entry['source_run_id'] and trace['record']['field_sha256'] == entry['field_sha256'] and trace['record']['seconds'] >= .05
    assert 'calls of the injected function: 1' in done.stdout


@needs_frozen
def test_the_wrapper_starts_the_entry_as_python_would(tmp_path):
    """sys.path, argv, __name__ and __file__ of the frozen entry are those of `python script run RUN`, and the wrapper's directory is not on the path."""
    run, _, _ = make_fake_run(tmp_path)
    assert wrapper(run, PATH_ONLY='1').returncode == 0
    through = json.loads((run/'trace.json').read_text())
    script = run/'snapshot/scripts/run_stage_3d_e48_pg.py'
    direct = subprocess.run([PY, '-B', str(script), 'run', str(run)], cwd=run/'snapshot', env=dict(os.environ, PATH_ONLY='1', PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True)
    assert direct.returncode == 0, direct.stderr
    alone = json.loads((run/'trace.json').read_text())
    assert through['path'] == alone['path'] and through['argv'] == alone['argv'] and through['name'] == alone['name'] == '__main__' and through['file'] == alone['file']
    assert str(run) not in through['path']
    assert through['modules_jd'] == ['jd_curl', 'jd_curl.three_d', 'jd_curl.three_d.preopt', 'jd_curl.three_d.validation', 'jd_curl.validation']     # only what preopt.py itself imports, before the entry
    assert alone['modules_jd'] == []


@needs_frozen
@pytest.mark.parametrize('case', ('field_sha256', 'u_sha256', 'shape', 'missing', 'missing_key'))
def test_the_wrapper_refuses_before_any_frozen_code(tmp_path, case):
    changes = dict(field_sha256=dict(field_sha256='0'*64), u_sha256=dict(u_sha256='0'*64), shape=dict(shape_zyx=[8, 9, 11]), missing=dict(field_path=str(tmp_path/'none.npz'))).get(case, {})
    run, _, entry = make_fake_run(tmp_path, **changes)
    if case == 'missing_key':
        del entry['oracle']
        write_json(run/'seed_entry.json', entry)
    done = wrapper(run)
    assert done.returncode == INJ.REFUSED and 'refused' in done.stdout, done.stdout+done.stderr
    assert not (run/'trace.json').exists() and not (run/'started.json').exists()


@pytest.mark.parametrize('case', ('float32', 'faces', 'nan', 'second_member'))
def test_load_field_refuses(tmp_path, case):
    u = small_field()
    path = tmp_path/'f.npz'
    if case == 'float32':
        np.savez(path, u=u.astype(np.float32))
    elif case == 'faces':
        u[0, 0, 0, 0] = 1.
        np.savez(path, u=u)
    elif case == 'nan':
        u[3, 3, 3, 0] = np.nan
        np.savez(path, u=u)
    else:
        np.savez(path, u=u, other=u)
    stored = np.load(path)['u']
    entry = dict(field_path=str(path), field_sha256=sha(path), u_sha256=hashlib.sha256(stored.tobytes()).hexdigest(), shape_zyx=list(SHAPE))
    with pytest.raises(INJ.Refused):
        INJ.load_field(entry)


def test_the_injected_function_holds_the_clock_and_runs_once():
    now = [100.]
    waited = []
    entry = dict(arm='S_P', pair='p', source_run_id='r', oracle=False, field_path='f', field_sha256='s', u_sha256='b', charge_seconds=47.)
    u = small_field()

    def clock():
        return now[0]

    def wait(seconds):
        waited.append(seconds)
        now[0] += seconds
    fn = INJ.make_injected(entry, u, clock=clock, wait=wait)
    got, record = fn(None, None, SimpleNamespace(shape=SHAPE), 'cfg', device='cuda', execution='ex')
    assert got is u and waited == [47.] and record['seconds'] == 47. and record['policy'] == INJ.POLICY and record['charge_seconds'] == 47.
    json.dumps(record)                                                         # the frozen path dumps it into levels.json
    with pytest.raises(RuntimeError, match='twice'):
        fn(None, None, SimpleNamespace(shape=SHAPE), 'cfg')
    with pytest.raises(RuntimeError, match='grid'):
        INJ.make_injected(entry, u, clock=clock, wait=wait)(None, None, SimpleNamespace(shape=(1, 2, 3)), 'cfg')
    with pytest.raises(RuntimeError, match='observer'):
        INJ.make_injected(entry, u, clock=clock, wait=wait)(None, None, SimpleNamespace(shape=SHAPE), 'cfg', observer=print)


def test_install_replaces_the_one_attribute():
    module = SimpleNamespace(preoptimize='frozen', boundary_taper='frozen taper', PreoptConfig='frozen config')
    fn = INJ.install(module, dict(charge_seconds=0), small_field())
    assert module.preoptimize is fn and module.boundary_taper == 'frozen taper' and module.PreoptConfig == 'frozen config'


# ---- the entry ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def entry_check(results, arm='S_T', pair=None):
    pair = RUN.PAIRS[0] if pair is None else pair
    return RUN.check_entry(arm, pair, RUN.input_files(arm, pair, results)[3], results)


def test_the_entry_paths_are_the_two_of_the_brief():
    assert str(RUN.entry_dir('S_P', RUN.PAIRS[0])) == '/home/rmcfz/seq_vp_tag_results/mc0p_v1/runs/Q2_long/0438-0439'
    assert str(RUN.entry_dir('S_T', RUN.PAIRS[4])) == '/home/rmcfz/seq_vp_tag_results/mc0_v1/runs/P2_long/0442-0443'
    assert str(RUN.inputs_root('S_P')) == '/home/rmcfz/seq_vp_tag_results/d0_v1_a5/inputs/A5P' and str(RUN.inputs_root('S_T')) == '/home/rmcfz/seq_vp_tag_results/d0_v1_a5/inputs/A5'
    with pytest.raises(SystemExit):
        RUN.entry_dir('S_X', RUN.PAIRS[0])


@pytest.mark.parametrize('arm', ('S_P', 'S_T'))
def test_a_good_entry_passes(tmp_path, arm):
    results = make_results(tmp_path, controls=False)
    record = entry_check(results, arm)
    assert record['oracle'] is RUN.ARMS[arm]['oracle'] and record['charge_seconds'] == 47. and record['source_run_id'] == RUN.ARMS[arm]['source_prefix']+'0438-0439' and record['tapered_sha256'] == record['tapered_recorded_by_evaluation']
    assert set(INJ.ENTRY_KEYS) <= set(record)


@pytest.mark.parametrize('arm,override,word', (
    ('S_P', dict(oracle=True), 'oracle'),                                       # S_P's entry must be predicted labels
    ('S_T', dict(oracle=False), 'oracle'),                                      # S_T's entry must be the oracle file
    ('S_P', dict(pair=RUN.PAIRS[1]), 'pair'),                                   # a field of another pair at this path
    ('S_T', dict(arm='P1'), 'arm'),
    ('S_T', dict(tier='fine'), 'tier'),
    ('S_T', dict(k=14), 'K'),
    ('S_T', dict(status='time_cap'), 'status'),
    ('S_T', dict(run_id='mc0_P2_long_0439-0440'), 'run_id'),
    ('S_T', dict(data=dict(pair_sha256='other', manifest_sha256='other', fixed_subject='x', moving_subject='y')), 'original pair'),
    ('S_T', dict(field=dict(file='field.npz')), 'field record'),
    ('S_T', dict(schema='bayesreg.mc0p.run.v1'), 'schema'),
    ('S_P', dict(schema='bayesreg.mc0.run.v1'), 'schema'),
    ('S_T', dict(config=dict(LONG_CONFIG, iterations=[50, 50, 100, 60])), 'config'),      # another tier's field under this name
    ('S_P', dict(config=dict(LONG_CONFIG, sigma_field=.5)), 'config'),
    ('S_P', dict(channel_sources=None), 'channel_sources'),                               # a record of the earlier format
    ('S_P', dict(channel_sources=predicted_sources(oracle=True)), 'channel_sources oracle'),      # the run's own mark says false, what it read says true
    ('S_P', dict(channel_sources=predicted_sources(truth_labels_read=True)), 'truth_labels_read'),
    ('S_P', dict(channel_sources=predicted_sources(pair_keys_read=['fixed', 'moving', 'fixed_seg35'])), 'pair_keys_read'),
    ('S_P', dict(channel_sources=predicted_sources(pair_keys_read=['fixed'])), 'pair_keys_read'),
    ('S_P', dict(channel_sources=predicted_sources(sides=dict(fixed=dict(segmenter_id='other', policy='all_folds_mean'), moving=dict(segmenter_id='unet35_v1_ens', policy='all_folds_mean')))), 'sides'),
))
def test_a_wrong_run_record_is_refused(tmp_path, arm, override, word):
    results = make_results(tmp_path, entries=(), controls=False)
    make_entry(results, arm, RUN.PAIRS[0], **override)
    with pytest.raises(RUN.EntryError, match=word):
        entry_check(results, arm)


def test_the_roots_of_s_p_are_fixed_pair_by_pair():
    assert RUN.S_P_ROOTS == {'0438-0439': 'mc0p_v1', '0439-0440': 'mc0p_v1', '0440-0441': 'mc0p_v1', '0441-0442': 'mc0p_v1b', '0442-0443': 'mc0p_v1b'}
    assert set(RUN.S_P_ROOTS) == {RUN.short_name(p) for p in RUN.PAIRS}
    assert [str(RUN.entry_dir('S_P', p)) for p in RUN.PAIRS[2:4]] == ['/home/rmcfz/seq_vp_tag_results/mc0p_v1/runs/Q2_long/0440-0441', '/home/rmcfz/seq_vp_tag_results/mc0p_v1b/runs/Q2_long/0441-0442']
    assert all(RUN.entry_root('S_T', p) == 'mc0_v1' for p in RUN.PAIRS)


def wrong_root_dir(results, pair):
    """The Q2 long directory of the batch that must NOT serve this pair."""
    other = RUN.RETRIED_ROOT if RUN.entry_root('S_P', pair) == RUN.RETRY_ROOT else RUN.RETRY_ROOT
    return Path(results)/other/'runs/Q2_long'/RUN.short_name(pair)


@pytest.mark.parametrize('index,word', ((0, '入口待 MC0P'), (2, '入口待 MC0P'), (3, '0441-0442 待补跑'), (4, '0442-0443 待补跑')))
def test_a_good_record_under_the_wrong_root_is_not_an_entry(tmp_path, index, word):
    """The control cell of mc0p_v1b for a first pair; a done record of mc0p_v1 for a last pair: never "whichever exists"."""
    results = make_results(tmp_path, entries=(), controls=False)
    pair = RUN.PAIRS[index]
    d = make_entry(results, 'S_P', pair)
    target = wrong_root_dir(results, pair)
    target.parent.mkdir(parents=True)
    d.rename(target)
    with pytest.raises(RUN.EntryMissing, match=word):
        entry_check(results, 'S_P', pair)


def test_the_first_pairs_take_the_first_batch_also_when_the_retry_has_a_control_cell(tmp_path):
    results = make_results(tmp_path, entries=(), controls=False)
    pair = RUN.PAIRS[0]
    make_entry(results, 'S_P', pair, seed=1)
    good = RUN.entry_dir('S_P', pair, results)
    other = wrong_root_dir(results, pair)
    other.parent.mkdir(parents=True)
    other.mkdir()
    np.savez(other/'field.npz', u=small_field(7))
    record = entry_check(results, 'S_P', pair)
    assert record['field_path'] == str(good/'field.npz') and record['source_root'] == 'mc0p_v1' and record['retry_of'] is None


@pytest.mark.parametrize('form', ('/home/rmcfz/seq_vp_tag_results/mc0p_v1/runs/Q2_long/0441-0442', 'mc0p_v1', dict(root='/home/rmcfz/seq_vp_tag_results/mc0p_v1', run_id='mc0p_Q2_long_0441-0442')))
def test_a_retry_record_that_points_at_the_first_batch_is_taken(tmp_path, form):
    results = make_results(tmp_path, entries=(), controls=False)
    make_entry(results, 'S_P', RUN.PAIRS[3], retry_of=form)
    record = entry_check(results, 'S_P', RUN.PAIRS[3])
    assert record['source_root'] == 'mc0p_v1b' and record['retry_of'] == form and '/mc0p_v1b/' in record['field_path']


@pytest.mark.parametrize('form', (DROP, None, '', 'mc0p_v1b', '/x/mc0p_v1b/runs/Q2_long/0441-0442', 'mc0_v1', 'mc0p_v1x', 7, dict(root='mc0p_v1b')))
def test_a_retry_record_without_retry_of_at_the_first_batch_is_refused(tmp_path, form):
    results = make_results(tmp_path, entries=(), controls=False)
    make_entry(results, 'S_P', RUN.PAIRS[4], retry_of=form)
    with pytest.raises(RUN.EntryError, match='retry_of'):
        entry_check(results, 'S_P', RUN.PAIRS[4])


def test_a_first_batch_record_with_retry_of_is_refused(tmp_path):
    results = make_results(tmp_path, entries=(), controls=False)
    make_entry(results, 'S_P', RUN.PAIRS[1], retry_of='/x/mc0p_v1/runs/Q2_long/0439-0440')
    with pytest.raises(RUN.EntryError, match='retry_of'):
        entry_check(results, 'S_P', RUN.PAIRS[1])


def test_a_cuda_oom_record_is_never_taken(tmp_path):
    """MC0P's first batch: 0441-0442 status cuda_oom, no field.  There it is not even read; a first pair or a retry in that state is refused."""
    results = make_results(tmp_path, entries=(), controls=False)
    pair = RUN.PAIRS[3]
    first = wrong_root_dir(results, pair)
    write_json(first/'run.json', dict(run_id='mc0p_Q2_long_0441-0442', arm='Q2', tier='long', pair=pair, k=35, oracle=False, status='cuda_oom', field=None))
    with pytest.raises(RUN.EntryMissing, match='0441-0442 待补跑'):
        entry_check(results, 'S_P', pair)
    make_entry(results, 'S_P', RUN.PAIRS[0], status='cuda_oom')
    with pytest.raises(RUN.EntryError, match='status cuda_oom'):
        entry_check(results, 'S_P', RUN.PAIRS[0])
    make_entry(results, 'S_P', pair, status='cuda_oom')
    with pytest.raises(RUN.EntryError, match='status cuda_oom'):
        entry_check(results, 'S_P', pair)


def test_the_words_of_the_pending_entries():
    assert RUN.pending_text(dict(S_P=list(RUN.PAIRS[3:]))) == 'S_P 入口 0441、0442 待补跑'
    assert RUN.pending_text(dict(S_P=list(RUN.PAIRS))) == 'S_P 入口 0438、0439、0440 待 MC0P；S_P 入口 0441、0442 待补跑'


def test_the_queue_stops_before_any_run_while_the_retry_is_missing(neutral):
    results = make_results(neutral, entries=(), controls=True)
    for arm, pair in RUN.plan_runs():
        if RUN.entry_root(arm, pair) != RUN.RETRY_ROOT:
            make_entry(results, arm, pair)
    root = neutral/'seed_v1'
    system = FakeSystem()
    assert RUN.queue(options(root), system, results, neutral) == 8
    stop = json.loads((root/'queue_stop.json').read_text())['reason']
    assert '0441-0442 待补跑' in stop and '0442-0443 待补跑' in stop and system.launched == [] and not (root/'runs').exists()


def test_the_entry_numbers_come_from_the_batch_of_each_pair(tmp_path):
    def long_csv(root, rows):
        path = tmp_path/root/'analysis/long.csv'
        path.parent.mkdir(parents=True)
        path.write_text('run_id,arm,tier,pair,accepted_alpha,measure,value\n'+''.join(f'r,Q2,long,{p},1.0,seg35,{v}\n' for p, v in rows))
    long_csv('mc0p_v1', [(RUN.PAIRS[0], .1), (RUN.PAIRS[3], .9)])                # the first batch's row of a retried pair is not taken
    long_csv('mc0p_v1b', [(RUN.PAIRS[0], .9), (RUN.PAIRS[3], .4)])               # the control cell's row is not taken
    rows = DEC.entry_rows(tmp_path)
    assert rows[('S_P', RUN.PAIRS[0])]['seg35'] == .1 and rows[('S_P', RUN.PAIRS[3])]['seg35'] == .4


def test_s_t_has_no_channel_sources_and_is_not_refused_for_it(tmp_path):
    results = make_results(tmp_path, entries=('S_T',), controls=False)
    assert 'channel_sources' not in json.loads((RUN.entry_dir('S_T', RUN.PAIRS[0], results)/'run.json').read_text())
    assert entry_check(results, 'S_T')['oracle'] is True


def test_the_marks_and_the_long_tier_are_those_of_the_code_that_writes_them():
    """experiments/mc0/mc0p_core.py, mc0_core.py, run_mc0p.py, run_mc0.py, read only: the names and values this package relies on."""
    sys.path.insert(0, str(ROOT/'experiments/mc0'))
    import mc0_core
    import mc0p_core as mc
    assert RUN.PREDICTION_SEGMENTER == mc.SEGMENTER_ID and RUN.PREDICTION_POLICY == mc.POLICY and RUN.PAIR_IMAGE_KEYS == mc.IMAGE_KEYS
    sides = {side: dict(segmenter_id=mc.SEGMENTER_ID, policy=mc.POLICY) for side in ('fixed', 'moving')}
    keys = ['fixed', 'moving', 'initial_u']
    assert RUN.channel_sources_problems(dict(channel_sources=dict(mc.oracle_mark(keys, sides), pair_keys_read=keys, sides=sides))) == []
    read_labels = keys+['fixed_seg35']
    assert RUN.channel_sources_problems(dict(channel_sources=dict(mc.oracle_mark(read_labels, sides), pair_keys_read=read_labels, sides=sides)))
    other = {side: dict(segmenter_id='other', policy=mc.POLICY) for side in ('fixed', 'moving')}
    assert RUN.channel_sources_problems(dict(channel_sources=dict(mc.oracle_mark(keys, other), pair_keys_read=keys, sides=other)))
    config = json.loads(json.dumps(mc0_core.tier_config('long')))
    assert {k: config[k] for k in RUN.LONG_TIER} == RUN.LONG_TIER
    assert "schema='bayesreg.mc0p.run.v1'" in (ROOT/'experiments/mc0/run_mc0p.py').read_text() and "schema='bayesreg.mc0.run.v1'" in (ROOT/'experiments/mc0/run_mc0.py').read_text()
    assert 'channel_sources=channel_sources' in (ROOT/'experiments/mc0/run_mc0p.py').read_text() and 'channel_sources' not in (ROOT/'experiments/mc0/run_mc0.py').read_text()


def test_sha256_and_shape_are_checked(tmp_path):
    results = make_results(tmp_path, entries=(), controls=False)
    d = make_entry(results, 'S_T', RUN.PAIRS[0])
    np.savez(d/'field.npz', u=small_field(99))                                  # another field at the path
    with pytest.raises(RUN.EntryError, match='sha256'):
        entry_check(results)
    u = np.zeros((8, 9, 11, 3))
    make_entry(results, 'S_T', RUN.PAIRS[0], u=u)
    with pytest.raises(RUN.EntryError, match='shape'):
        entry_check(results)
    make_entry(results, 'S_T', RUN.PAIRS[0], field=dict(file='field.npz', sha256=None, u_sha256=None))
    with pytest.raises(RUN.EntryError):
        entry_check(results)


def test_bytes_and_faces_of_the_field_are_checked(tmp_path):
    results = make_results(tmp_path, entries=(), controls=False)
    d = make_entry(results, 'S_T', RUN.PAIRS[0])
    run = json.loads((d/'run.json').read_text())
    run['field']['u_sha256'] = '0'*64
    write_json(d/'run.json', run)
    with pytest.raises(RUN.EntryError, match='bytes'):
        entry_check(results)
    u = small_field()
    u[0, 1, 1, 0] = 1.
    make_entry(results, 'S_T', RUN.PAIRS[0], u=u)
    with pytest.raises(RUN.EntryError, match='faces'):
        entry_check(results)


def test_a_taper_that_differs_from_the_evaluations_record_is_refused(tmp_path):
    results = make_results(tmp_path, entries=(), controls=False)
    d = make_entry(results, 'S_T', RUN.PAIRS[0])
    write_json(d/'metrics.json', dict(tapered_u_sha256='0'*64))
    with pytest.raises(RUN.EntryError, match='tapered'):
        entry_check(results)


def test_a_taper_record_that_is_not_there_yet_is_said(neutral):
    results = make_results(neutral, entries=('S_T', 'S_P'), controls=True)
    (RUN.entry_dir('S_P', RUN.PAIRS[1], results)/'metrics.json').unlink()
    record = entry_check(results, 'S_P', RUN.PAIRS[1])
    assert record['tapered_recorded_by_evaluation'] is None and record['tapered_sha256']
    lines = []
    RUN.start_checks(FakeSystem(), results, neutral, lines.append)
    assert any('S_P 0439-0440' in line and 'no record of the taper yet' in line for line in lines)


def test_the_pending_entries_say_so(tmp_path):
    results = make_results(tmp_path, entries=('S_T',), controls=False)
    with pytest.raises(RUN.EntryMissing, match='S_P 入口待 MC0P'):
        entry_check(results, 'S_P')


@pytest.mark.parametrize('field,value,word', (('oracle', True, 'oracle'), ('arm', 'A5', 'arm')))
def test_inputs_that_are_not_the_arms_are_refused(tmp_path, field, value, word):
    results = make_results(tmp_path, entries=(), controls=False)
    root = RUN.inputs_root('S_P', results)
    d = root/'stage_L2R_val_v1'/RUN.PAIRS[0]
    manifest = json.loads((d/'manifest.json').read_text())
    manifest['preprocessing_d0'][field] = value
    write_json(d/'manifest.json', manifest)
    record = json.loads((root/'d0_inputs.json').read_text())
    record['files_sha256'][f'stage_L2R_val_v1/{RUN.PAIRS[0]}/manifest.json'] = sha(d/'manifest.json')
    write_json(root/'d0_inputs.json', record)
    with pytest.raises(RUN.EntryError, match=word):
        RUN.input_files('S_P', RUN.PAIRS[0], results)


def test_inputs_that_differ_from_their_record_are_refused(tmp_path):
    results = make_results(tmp_path, entries=(), controls=False)
    d = RUN.inputs_root('S_P', results)/'stage_L2R_val_v1'/RUN.PAIRS[0]
    np.savez(d/'pair.npz', fixed=np.ones(SHAPE))
    with pytest.raises(RUN.EntryError, match='differ'):
        RUN.input_files('S_P', RUN.PAIRS[0], results)


# ---- the record of a run -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def fake_levels(run, entry, policy=INJ.POLICY, raw=None, seed=None):
    task = run/'pg_weak__four__330s'
    task.mkdir(parents=True, exist_ok=True)
    write_json(run/'protocol.json', dict(task=dict(name='pg_weak__four__330s')))
    write_json(run/'seed_entry.json', entry)
    attempts = [dict(alpha=1., certificate_passed=True, certificate=dict(min_jd_lower_bound=.01, n_infeasible=0, state_hash='h'))]
    write_json(task/'levels.json', [dict(event='preoptimizer_done', level=3, preoptimizer=dict(policy=policy, raw_hash=entry['u_sha256'] if raw is None else raw, seed_hash=entry['tapered_sha256'] if seed is None else seed, taper_L=4, charge_seconds=47., seconds=47., seconds_total=47.2)),
                                    dict(event='level_prepared', level=3, transfer_recovery=dict(status='ready', accepted_alpha=1., attempts=attempts))])
    write_json(task/'history.json', [dict(D_start=2000., F_start=2000., D_candidate=1900.), dict(D_candidate=1500.)])


def test_the_trace_of_a_run_that_was_injected(tmp_path):
    entry = dict(u_sha256='b', tapered_sha256='t', field_path='f', field_sha256='s', source_run_id='r', oracle=False, entry_metrics=dict(seg35=.86))
    fake_levels(tmp_path, entry)
    record, problems = RUN.injection_trace(tmp_path)
    assert not problems and record['passed'] is True and record['accepted_alpha'] == 1. and record['entry_D_at_accepted_alpha'] == 2000. and record['last_D'] == 1500. and record['field_path'] == 'f' and record['source_run_id'] == 'r'


@pytest.mark.parametrize('change,word', ((dict(policy='fp32-preopt-v1'), 'did not run'), (dict(raw='x'), 'raw_hash'), (dict(seed='x'), 'seed_hash')))
def test_the_trace_catches_a_run_that_was_not_injected_or_tapered_twice(tmp_path, change, word):
    entry = dict(u_sha256='b', tapered_sha256='t')
    fake_levels(tmp_path, entry, **change)
    record, problems = RUN.injection_trace(tmp_path)
    assert record['passed'] is False and word in ' '.join(problems)


def test_no_record_is_a_problem(tmp_path):
    write_json(tmp_path/'protocol.json', dict(task=dict(name='t')))
    assert 'no event preoptimizer_done' in RUN.injection_trace(tmp_path)[1]


# ---- the reading --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def seed_record(arm, results=None):
    return dict(passed=True, problems=[], accepted_alpha=1., transfer_attempts=[], entry_D_at_accepted_alpha=2000., last_D=1500., arm=dict(seconds=RUN.SECONDS, changes=RUN.CHANGES, inputs_root=str(RUN.inputs_root(arm, results)), charge_seconds=RUN.ENTRY_CHARGE_SECONDS))


def made_up(deltas, seg4=.9, sdlogj=.08, base=.83):
    """S_P and B1TM with the given paired differences of seg35."""
    runs = {}
    for pair, delta in zip(RUN.PAIRS, deltas):
        runs[('B1TM', pair)] = dict(run_id='c', metrics=metric_values(base, seg4), certificate=dict(passed=True), native_commits=85)
        runs[('S_P', pair)] = dict(run_id='s', metrics=metric_values(base+delta, seg4, sdlogj), classification={'class': 'ok'}, certificate=dict(passed=True, state_hash_matches_u=True, final_npz_sha256_matches=True), seed=seed_record('S_P'), native_commits=70)
    return runs


@pytest.mark.parametrize('deltas,code', (
    ([.012]*5, DEC.PERSISTS),
    ([.05, .05, .05, .05, -.001], DEC.MIDDLE),                                  # large but 4/5: not 5/5, and not fewer than 4
    ([.009]*5, DEC.MIDDLE),
    ([.0051]*5, DEC.MIDDLE),
    ([.03, .03, .03, -.001, -.001], DEC.NO_DIFFERENCE),                         # 3/5
    ([.0049]*5, DEC.NO_DIFFERENCE),
    ([0.]*5, DEC.NO_DIFFERENCE),
    ([-.02]*5, DEC.NO_DIFFERENCE),
))
def test_the_reading_table(deltas, code):
    result = DEC.decide(made_up(deltas))
    assert result['reading_code'] == code, result
    assert result['reading'].startswith(DEC.READINGS[code])
    assert result['delta_seg35']['n'] == 5 and abs(result['delta_seg35']['mean']-float(np.mean(deltas))) < 1e-12


def test_the_bars_are_included(monkeypatch):
    """On numbers that are exact in binary (.5 + 2^-k): a mean at the bar counts as at the bar."""
    monkeypatch.setattr(DEC, 'SEE_BAR', 2.**-4)
    monkeypatch.setattr(DEC, 'NOT_SEE_BAR', 2.**-5)
    assert DEC.decide(made_up([2.**-4]*5, base=.5))['reading_code'] == DEC.PERSISTS
    assert DEC.decide(made_up([2.**-5]*5, base=.5))['reading_code'] == DEC.NO_DIFFERENCE
    assert DEC.decide(made_up([2.**-5+2.**-20]*5, base=.5))['reading_code'] == DEC.MIDDLE


def test_a_mean_above_the_bar_rounded_is_not_rounded():
    assert DEC.decide(made_up([.00999]*5))['reading_code'] == DEC.MIDDLE
    assert DEC.decide(made_up([.00501]*5))['reading_code'] == DEC.MIDDLE


@pytest.mark.parametrize('breaking,word', (
    (lambda runs: runs.pop(('S_P', RUN.PAIRS[2])), '没有 metrics.json'),
    (lambda runs: runs[('S_P', RUN.PAIRS[2])].update(metrics=None), '无终态场'),
    (lambda runs: runs[('S_P', RUN.PAIRS[2])].update(certificate=dict(passed=False)), '证书未通过'),
    (lambda runs: runs[('S_P', RUN.PAIRS[2])]['certificate'].update(final_npz_sha256_matches=False), '绑定'),                # the map is not the one the certificate speaks of
    (lambda runs: runs[('S_P', RUN.PAIRS[2])]['certificate'].update(state_hash_matches_u=False), '绑定'),
    (lambda runs: runs[('S_P', RUN.PAIRS[2])]['seed'].update(passed=False), '入口核对'),
    (lambda runs: runs[('S_P', RUN.PAIRS[2])]['seed']['arm'].update(seconds=240), '配置'),
    (lambda runs: runs[('S_P', RUN.PAIRS[2])]['seed']['arm'].update(charge_seconds=0.), '配置'),
    (lambda runs: runs[('B1TM', RUN.PAIRS[2])].update(certificate=dict(passed=False)), '基线'),
    (lambda runs: runs.pop(('B1TM', RUN.PAIRS[2])), '基线'),
))
def test_premises_not_met_are_not_decidable(breaking, word):
    runs = made_up([.02]*5)
    breaking(runs)
    result = DEC.decide(runs)
    assert result['reading_code'] == DEC.NOT_DECIDABLE and word in result['reading']


def test_handover_only_on_every_pair_is_not_read_as_passing_the_native_level():
    """The review's counterexample (I2): five certified entries, no native commit, Delta +.02.  It read 'persists', a false sentence."""
    runs = made_up([.02]*5)
    for pair in RUN.PAIRS:
        runs[('S_P', pair)].update(classification={'class': 'handover_only'}, native_commits=0)
    result = DEC.decide(runs)
    assert result['reading_code'] == DEC.NOT_DECIDABLE and 'handover_only' in result['reading'] and '原生提交 0' in result['reading']


@pytest.mark.parametrize('change,word', (
    (dict(classification={'class': 'handover_only'}, native_commits=0), 'handover_only'),      # one pair is enough
    (dict(classification={'class': 'method_failure'}), 'method_failure'),
    (dict(classification=None), '分类 None'),
    (dict(native_commits=0), '原生提交 0'),                                                     # class ok but no commit: not read either
    (dict(native_commits=None), '原生提交 None'),
    (dict(native_commits=True), '原生提交 True'),
))
def test_each_pair_must_be_class_ok_with_a_native_commit(change, word):
    runs = made_up([.02]*5)
    runs[('S_P', RUN.PAIRS[3])].update(change)
    result = DEC.decide(runs)
    assert result['reading_code'] == DEC.NOT_DECIDABLE and word in result['reading']


def test_one_native_commit_is_enough_and_the_bar_is_on_s_p_only():
    runs = made_up([.02]*5)
    runs[('S_P', RUN.PAIRS[3])]['native_commits'] = 1
    assert DEC.decide(runs)['reading_code'] == DEC.PERSISTS
    runs.update({('S_T', p): dict(metrics=metric_values(.9), classification={'class': 'handover_only'}, native_commits=0, certificate=dict(passed=True)) for p in RUN.PAIRS})
    assert DEC.decide(runs)['reading_code'] == DEC.PERSISTS                     # S_T does not enter the reading
    assert any('handover_only' in reason for reason in DEC.arm_premises(runs, 'S_T'))


@pytest.mark.parametrize('deltas,near', (([.0105]*5, .01), ([.0095]*5, .01), ([.0052]*5, .005), ([.0048]*5, .005), ([.02]*5, None), ([.0075]*5, None)))
def test_a_delta_near_a_bar_says_so_and_keeps_its_code(deltas, near):
    result = DEC.decide(made_up(deltas))
    assert result['near_bar'] == near and ('贴着' in result['reading']) is (near is not None)
    assert result['reading_code'] == (DEC.PERSISTS if deltas[0] >= .01 else DEC.NO_DIFFERENCE if deltas[0] <= .005 else DEC.MIDDLE)


def test_sdlogj_over_the_bar_is_not_decidable_and_at_the_bar_is():
    assert DEC.decide(made_up([.02]*5, sdlogj=.1001))['reading_code'] == DEC.NOT_DECIDABLE
    assert DEC.decide(made_up([.02]*5, sdlogj=.10))['reading_code'] == DEC.PERSISTS


def test_a_reference_problem_is_not_decidable():
    result = DEC.decide(made_up([.02]*5), ['B1TM: 不符'])
    assert result['reading_code'] == DEC.NOT_DECIDABLE and 'B1TM: 不符' in result['reading']


def test_s_t_does_not_enter_the_reading():
    runs = made_up([0.]*5)
    runs.update({('S_T', p): dict(metrics=metric_values(.99), certificate=dict(passed=True)) for p in RUN.PAIRS})
    runs.update({('B1W_A5', p): dict(metrics=metric_values(.5), certificate=dict(passed=True)) for p in RUN.PAIRS})
    assert DEC.decide(runs)['reading_code'] == DEC.NO_DIFFERENCE


def write_arm_runs(root, results, deltas, sdlogj=.08):
    for arm, spec in RUN.ARMS.items():
        base = DEC.PINS[spec['control']]
        for pair, delta in zip(RUN.PAIRS, deltas):
            m = dict(run_id=f'seed_{arm}_{RUN.short_name(pair)}', arm=arm, pair=pair, method=RUN.method_name(arm), classification={'class': 'ok'}, status='native_budget_exhausted', metrics=metric_values(base[0]+delta, base[1]+.001, sdlogj), native_commits=70, e2e_seconds=327.,
                     certificate=dict(passed=True, state_hash_matches_u=True, final_npz_sha256_matches=True, min_jd_lower_bound=1e-4, status='certified_feasible'), self_check=dict(passed=True), seed=seed_record(arm, results))
            write_json(RUN.run_dir(root, arm, pair)/'metrics.json', m)


@pytest.fixture
def analysis_tree(tmp_path, monkeypatch):
    results = make_results(tmp_path, entries=(), controls=True)
    monkeypatch.setattr(DEC, '_selection', lambda: [dict(pair=p) for p in RUN.PAIRS])
    return results, tmp_path/'seed_v1'


def test_the_written_analysis_reports_seg35_and_seg4_and_keeps_the_earlier_version(analysis_tree):
    results, root = analysis_tree
    write_arm_runs(root, results, [.02]*5)
    result = DEC.write_analysis(root, results=results)
    assert result['reading_code'] == DEC.PERSISTS and not result['reference_problems']
    summary = (root/'analysis/summary_seed.md').read_text()
    for word in ('seg35', 'seg4', 'oracle，只作上限', '小六个', '大六个', '其余 23 个', '皮层', '白质', '开发集', '入口 seg35'):
        assert word in summary, word
    assert result['oracle_only']['seg35']['mean'] == pytest.approx(.02) and 'oracle' not in result['reading']
    assert set(result['described']['groups']) == {'S_P', 'B1TM', 'S_T', 'B1W_A5'} and result['described']['groups']['S_P']['cortex'] is not None
    assert result['described']['differences']['S_P-B1TM']['groups']['small_six']['n'] == 5
    assert (root/'analysis/long.csv').read_text().count(RUN.method_name('S_P')) > 0
    assert json.loads((root/'analysis/decision_seed.json').read_text())['reading_code'] == DEC.PERSISTS
    again = DEC.write_analysis(root, results=results)
    assert again['previous_summary_moved_to'] and (Path(again['previous_summary_moved_to'])/'decision_seed.json').is_file()


def test_the_analysis_before_any_run_is_not_decidable(analysis_tree):
    results, root = analysis_tree
    assert DEC.write_analysis(root, results=results)['reading_code'] == DEC.NOT_DECIDABLE


def test_a_baseline_that_is_not_the_pinned_one_is_not_decidable(analysis_tree):
    results, root = analysis_tree
    write_arm_runs(root, results, [.02]*5)
    make_controls(results, means=dict(B1TM=(.84, .8947), B1W_A5=DEC.PINS['B1W_A5']))
    result = DEC.write_analysis(root, results=results)
    assert result['reading_code'] == DEC.NOT_DECIDABLE and any('钉住' in p for p in result['reference_problems'])


def test_a_baseline_of_another_configuration_is_not_decidable(analysis_tree):
    results, root = analysis_tree
    write_arm_runs(root, results, [.02]*5)
    make_controls(results, configuration=dict(seconds=90, changes={}, inputs_root='/x'))
    assert DEC.write_analysis(root, results=results)['reading_code'] == DEC.NOT_DECIDABLE


def test_the_oracle_controls_problems_stay_with_the_oracle_arm(analysis_tree):
    results, root = analysis_tree
    write_arm_runs(root, results, [.02]*5)
    make_controls(results, means=dict(B1TM=DEC.PINS['B1TM'], B1W_A5=(.80, .90)))
    result = DEC.write_analysis(root, results=results)
    assert result['reading_code'] == DEC.PERSISTS and result['reference_problems'] == []
    assert any('B1W_A5' in p for p in result['oracle_only']['reference_problems'])
    assert 'oracle，只作上限，不进判定' in (root/'analysis/summary_seed.md').read_text()


def test_the_pins_are_the_plans():
    assert DEC.PINS == dict(B1TM=(.8341, .8947), B1W_A5=(.8853, .9389)) and DEC.SDLOGJ_MAX == .10 and DEC.SEE_BAR == .01 and DEC.NOT_SEE_BAR == .005 and DEC.MIN_POSITIVE == 4


def test_the_line_of_86_is_a_model_not_a_bar():
    runs = made_up([.02]*5, base=.8341)
    refs = {('B1TM', f'p{k}'): dict(metrics=dict(seg35=.8468)) for k in range(19)}
    line = DEC.line_86(runs, refs)
    assert line['first5_line'] == pytest.approx(.86-(.8468-.8341)) and line['first5_line'] == pytest.approx(.8473) and '不是判据' in line['role'] and '模型' in line['role']
    assert DEC.line_86(runs, {})['first5_line'] is None


# ---- the plan, the pairs, the start strings -----------------------------------------------------------------------------------------------------------------------------------------------------------
def test_only_the_first_5_pairs_and_the_plan_order():
    assert tuple(RUN.short_name(p) for p in RUN.PAIRS) == BRIEF_PAIRS
    plan = RUN.plan_runs()
    assert len(plan) == 10 and [a for a, _ in plan] == ['S_P']*5+['S_T']*5 and [p for _, p in plan[:5]] == list(RUN.PAIRS) == [p for _, p in plan[5:]]


@pytest.mark.parametrize('pair', ('OASIS_OAS1_0443_MR1__OASIS_OAS1_0444_MR1', 'OASIS_OAS1_0020_MR1__OASIS_OAS1_0021_MR1', 'S37__S38', '../x'))
def test_other_pairs_are_refused_before_any_file_is_opened(pair, monkeypatch):
    monkeypatch.setattr('builtins.open', lambda *a, **k: pytest.fail('a file was opened'))
    for call in (lambda: RUN.require_pair(pair), lambda: RUN.entry_dir('S_P', pair), lambda: RUN.run_dir('/x', 'S_P', pair), lambda: RUN.prepare_argv('S_P', pair, '/x')):
        with pytest.raises(SystemExit):
            call()


def test_subjects_are_on_the_allowlist_and_off_the_training_and_holdout_lists():
    from bayesreg.io.guard import check_pair_subjects, load_dev_allowlist
    allowlist = load_dev_allowlist(ROOT/'configs/bayesreg/dev_subjects.json')
    subjects = set()
    for pair in RUN.PAIRS:
        fixed, moving = pair.split('__')
        check_pair_subjects(fixed, moving, allowlist)
        subjects |= {fixed, moving}
    assert len(subjects) == 6
    for name in ('train_subjects.txt', 'holdout_B_subjects.txt'):
        listed = set((ROOT/'splits/bayesreg_v1'/name).read_text().split())
        assert listed and not subjects & listed, name


def test_start_strings():
    root = RUN.DEFAULT_ROOT
    assert str(root) == '/home/rmcfz/seq_vp_tag_results/seed_v1'
    strings = [RUN.queue_launch()]
    for arm, pair in RUN.plan_runs():
        strings += [RUN.prepare_argv(arm, pair, RUN.run_dir(root, arm, pair)), RUN.unit_argv(arm, pair, RUN.run_dir(root, arm, pair), root)]
    for argv in strings:
        assert 'py'+'test' not in shlex.join(argv) and 'seq-vp' not in shlex.join(argv)
    for arm, pair in RUN.plan_runs():
        text = shlex.join(RUN.unit_argv(arm, pair, RUN.run_dir(root, arm, pair), root))
        assert text.startswith(f'systemd-run --user --unit=tag-seed-{arm}-') and '--property=MemoryMax=12G' in text and '--property=MemorySwapMax=4G' in text and '1100s' in text and 'seed_inject.py' in text
        assert 'CUDA_VISIBLE_DEVICES' not in text
    assert shlex.join(RUN.queue_launch()).startswith('systemd-run --user --unit=tag-seed-queue')


def test_a_start_string_with_the_word_is_refused():
    with pytest.raises(ValueError):
        RUN.check_launch(['python', '-m', 'py'+'test'])


def test_the_options_are_configuration_c_with_b1tm_changes():
    """Token by token against run_queue.sh (read only), which is what B1TM ran."""
    text = (ROOT/'experiments/d0/run_queue.sh').read_text()
    assert f'PRE="{RUN.PRE}"' in text
    block = text.split('\nOPTS=(', 1)[1].split('--pre-optimizer "$PRE")', 1)[0]
    tokens = shlex.split(block.replace('\\\n', ' '))
    tokens[tokens.index('--geometry-weight-levels')+1] = '1,1,0.1,0.01'
    tokens[tokens.index('--seconds')+1] = '330'
    assert tokens+['--pre-optimizer', RUN.PRE] == list(RUN.OPTS)


b1tm_cmd = Path('/home/rmcfz/seq_vp_tag_results/b1_v1/runs/B1TM')/RUN.PAIRS[0]/'cmd.txt'


@pytest.mark.skipif(not b1tm_cmd.is_file(), reason='the B1TM run is not on this machine')
def test_the_commands_are_b1tm_but_for_the_names_and_the_script():
    """The prepare line and the unit of B1TM 0438-0439 (its cmd.txt, read only), with only the run directory, the stage, the unit, the cache root and the script replaced."""
    lines = [line for line in b1tm_cmd.read_text().splitlines() if not line.startswith('#')]
    old_run, old_root = str(b1tm_cmd.parent), '/home/rmcfz/seq_vp_tag_results/b1_v1'
    new_root = '/home/rmcfz/seq_vp_tag_results/seed_v1'
    new_run = str(RUN.run_dir(new_root, 'S_P', RUN.PAIRS[0]))
    prepare = [t.replace(old_run, new_run).replace('B1-B1TM', 'SEED-S_P') for t in shlex.split(lines[0])]
    assert prepare == RUN.prepare_argv('S_P', RUN.PAIRS[0], new_run)
    unit = [t.replace(f"'{old_run}/snapshot/scripts/run_stage_3d_e48_pg.py' run '{old_run}'", f"'{new_run}/seed_inject.py' '{new_run}'").replace(old_run, new_run).replace(old_root, new_root).replace('tag-b1-B1TM-', 'tag-seed-S_P-') for t in shlex.split(lines[1])]
    assert unit == RUN.unit_argv('S_P', RUN.PAIRS[0], new_run, new_root)


def test_the_method_name_says_seed_and_oracle():
    assert RUN.method_name('S_P') == 'frozen_eb808be_seed_330s_gw1_1_0.1_0.01_entryQ2long'
    assert RUN.method_name('S_T') == 'frozen_eb808be_seed_330s_gw1_1_0.1_0.01_entryP2long_oracle'


# ---- the gate and the queue on a fake machine ------------------------------------------------------------------------------------------------------------------------------------------------------------
class FakeSystem:
    def __init__(self):
        self.load, self.gib, self.mib, self.gpu, self.unit_list = .5, 500., 24000., None, []
        self.clock, self.sleeps, self.launched = 0., [], []
        self.classes = {}
        self.fail_check, self.fail_eval, self.frozen, self.inputs = False, False, [], []
        self.bad_trace, self.no_entry = set(), set()

    def state(self):
        return 'fake state'

    def load1(self):
        return self.load

    def units(self):
        return list(self.unit_list)

    def gpu_free_mib(self):
        return self.mib

    def gpu_busy(self):
        return self.gpu

    def free_gib(self, path):
        return self.gib[path] if isinstance(self.gib, dict) else self.gib

    def now(self):
        return self.clock

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.clock += seconds

    def frozen_problems(self, results=None):
        return list(self.frozen)

    def inputs_problems(self, arm, repo, results=None):
        return list(self.inputs)

    def prepare(self, argv, target, log, results=None):
        run = Path(argv[argv.index('prepare')+1])
        run.mkdir(parents=True)
        write_json(run/'protocol.json', dict(task=dict(name='pg_weak__four__330s')))
        return 0

    def check_prepared(self, run, pair, stage, pair_sha, manifest_sha, entry):
        assert entry['seconds'] == 330 and entry['changes'] == RUN.CHANGES and stage.startswith('SEED-')
        return dict(ok=not self.fail_check, problems=['made up'] if self.fail_check else [])

    def launch(self, argv, log):
        unit = argv[2].split('=')[1]
        self.launched.append(unit)
        assert 'seed_inject.py' in argv[-1]
        run = Path(argv[argv.index('--wait')+2].split('=')[1]).parent
        entry = json.loads((run/'seed_entry.json').read_text())
        if unit not in self.no_entry:
            fake_levels(run, entry, raw='wrong' if unit in self.bad_trace else None)
        (run/'exit_code').write_text('0\n')
        return 0

    def classify(self, run, t0):
        unit = (Path(run)/'service_unit.txt').read_text().strip()
        return {'class': self.classes.get(unit, 'ok'), 'kind': 'k', 'reason': 'r'}

    def evaluate(self, run, arm, pair, run_id):
        if self.fail_eval:
            return 1
        trace, _ = RUN.injection_trace(run)
        write_json(Path(run)/'metrics.json', dict(run_id=run_id, arm=arm, pair=pair, method=RUN.method_name(arm), classification={'class': 'ok'}, status='s', metrics=metric_values(), certificate=dict(passed=True, min_jd_lower_bound=1e-4), self_check=dict(passed=True), native_commits=70, e2e_seconds=327., seed=dict(trace, arm=json.loads((Path(run)/'seed_arm.json').read_text()))))
        return 0

    def uptime(self):
        return '100.0'


UNITS = [RUN.unit_name(a, p) for a, p in RUN.plan_runs()]


def options(root):
    return RUN.argparse.Namespace(root=str(root), wait_minutes=30)


@pytest.fixture
def neutral():
    """A temporary directory whose path does not hold the word of the test runner: check_launch would refuse the start strings under tmp_path, as it should."""
    with tempfile.TemporaryDirectory(prefix='seed_tools_') as d:
        yield Path(d)


@pytest.fixture
def machine(neutral, monkeypatch):
    results = make_results(neutral, controls=True)
    monkeypatch.setattr(DEC, '_selection', lambda: [dict(pair=p) for p in RUN.PAIRS])
    root = neutral/'seed_v1'
    root.mkdir()
    return SimpleNamespace(results=results, root=root, repo=neutral/'no_repo', system=FakeSystem())


def run_queue(machine, system=None):
    return RUN.queue(options(machine.root), system or machine.system, machine.results, machine.repo)


def test_the_gate(tmp_path):
    system = FakeSystem()
    assert RUN.busy_reason(system, tmp_path) is None
    (tmp_path/'hold').write_text('')
    assert RUN.busy_reason(system, tmp_path) == 'hold file'
    (tmp_path/'hold').unlink()
    system.unit_list = ['tag-seed-queue.service']                                 # its own driver is not another unit
    assert RUN.busy_reason(system, tmp_path) is None
    for other in ('tag-unv2-chain.service', 'seq-vp-x.service', 'tag-mc0p-q2.service'):
        system.unit_list = [other]
        assert other in RUN.busy_reason(system, tmp_path)
    system.unit_list = []
    system.load = 1.5
    assert 'load' in RUN.busy_reason(system, tmp_path)
    system.load = 1.49
    assert RUN.busy_reason(system, tmp_path) is None
    system.gib = {'/home': 99.9, '/mnt/c': 500.}
    assert '/home' in RUN.busy_reason(system, tmp_path)
    system.gib = {'/home': 500., '/mnt/c': 25.9}
    assert '/mnt/c' in RUN.busy_reason(system, tmp_path)
    system.gib = {'/home': 100., '/mnt/c': 26.}
    assert RUN.busy_reason(system, tmp_path) is None
    system.gpu = 'compute pids [1]'
    assert 'GPU' in RUN.busy_reason(system, tmp_path)
    system.gpu = None
    system.mib = 14*1024-1
    assert 'GPU free' in RUN.busy_reason(system, tmp_path)
    system.mib = 14*1024
    assert RUN.busy_reason(system, tmp_path) is None


def test_a_gate_closed_too_long_stops(tmp_path):
    system = FakeSystem()
    system.load = 9.
    with pytest.raises(RUN.Stop, match='gate closed for 1 minutes'):
        RUN.wait_for_gate(system, tmp_path, 1, lambda text: None)
    assert system.sleeps and all(s == RUN.POLL_SECONDS for s in system.sleeps)


@pytest.mark.parametrize('how', ('[N/A]\n', 'Failed to initialize NVML\n', '', 'nan\n'))
def test_a_gpu_reading_that_is_not_a_number_keeps_the_gate_closed(monkeypatch, how):
    monkeypatch.setattr(RUN.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=how, returncode=0))
    assert RUN.System().gpu_free_mib() == 0.


def test_a_failing_systemctl_keeps_the_gate_closed(monkeypatch):
    monkeypatch.setattr(RUN.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='', returncode=1))
    assert RUN.System().units() == ['systemctl failed with exit 1']


def test_the_queue_runs_the_plan_once_and_a_restart_reruns_nothing(machine):
    assert run_queue(machine) == 0
    assert machine.system.launched == UNITS
    pauses = [s for s in machine.system.sleeps if s == RUN.BETWEEN_RUNS_SECONDS]
    assert len(pauses) == 9                                                        # 60 s between runs, none before the first
    run = RUN.run_dir(machine.root, 'S_P', RUN.PAIRS[0])
    for name in ('cmd.txt', 'data.txt', 'env.txt', 'git.txt', 'seed', 'run_id', 'seed_arm.json', 'seed_entry.json', 'seed_inject.py', 'prepared_check.json', 'cleanup.json', 'queue_done.json', 'metrics.json', 'service_unit.txt'):
        assert (run/name).is_file(), name
    assert (run/'seed_inject.py').read_bytes() == (ROOT/'experiments/seed/inject_run.py').read_bytes()
    assert (run/'run_id').read_text().startswith('seed_S_P_0438-0439_')
    assert json.loads((run/'seed_arm.json').read_text())['charge_seconds'] == 47.
    assert json.loads((run/'seed_entry.json').read_text())['source_run_id'] == 'mc0p_Q2_long_0438-0439'
    assert (machine.root/'analysis/decision_seed.json').is_file()
    again = FakeSystem()
    assert run_queue(machine, again) == 0 and again.launched == []
    assert not (machine.root/'queue_stop.json').exists()


def test_a_missing_entry_stops_before_any_run_and_says_what_is_missing(neutral):
    results = make_results(neutral, entries=('S_T',), controls=True)
    root = neutral/'seed_v1'
    system = FakeSystem()
    assert RUN.queue(options(root), system, results, neutral) == 8
    stop = json.loads((root/'queue_stop.json').read_text())['reason']
    assert 'S_P 入口待 MC0P' in stop and system.launched == [] and not (root/'runs').exists()


@pytest.mark.parametrize('attr,value', (('frozen', ['frozen worktree not clean']), ('inputs', ['colour table not committed'])))
def test_failed_start_checks_stop_before_any_run(machine, attr, value):
    setattr(machine.system, attr, value)
    assert run_queue(machine) == 8
    assert machine.system.launched == [] and value[0] in json.loads((machine.root/'queue_stop.json').read_text())['reason']


def test_a_method_failure_is_kept_and_the_queue_goes_on(machine):
    machine.system.classes = {UNITS[0]: 'method_failure'}
    run = RUN.run_dir(machine.root, 'S_P', RUN.PAIRS[0])
    assert run_queue(machine) == 0 and machine.system.launched == UNITS
    assert not (run/'cleanup.json').exists() and json.loads((run/'queue_done.json').read_text())['state'] == 'method_failure'


@pytest.mark.parametrize('how', ('bad_trace', 'no_entry'))
def test_a_method_failure_that_did_not_take_the_entry_stops(machine, how):
    """The review's counterexample (I1): a RuntimeError of the wrapper or an InputValidationError of the entry is a method exception for classify_run.py.  It must not pass as a method result and use up the batch."""
    machine.system.classes = {UNITS[0]: 'method_failure'}
    setattr(machine.system, how, {UNITS[0]})
    assert run_queue(machine) == 8
    assert machine.system.launched == UNITS[:1]
    reason = json.loads((machine.root/'queue_stop.json').read_text())['reason']
    assert 'not injected' in reason and 'method_failure' in reason
    run = RUN.run_dir(machine.root, 'S_P', RUN.PAIRS[0])
    assert not (run/'queue_done.json').exists() and not (run/'cleanup.json').exists()     # kept whole
    again = FakeSystem()
    again.classes = {UNITS[0]: 'method_failure'}
    assert run_queue(machine, again) == 8 and again.launched == []              # a restart does not run past it either


def test_three_infrastructure_failures_in_a_row_stop_and_two_do_not(machine):
    machine.system.classes = {UNITS[0]: 'infrastructure', UNITS[1]: 'infrastructure', UNITS[3]: 'infrastructure', UNITS[4]: 'infrastructure', UNITS[5]: 'infrastructure'}
    assert run_queue(machine) == 8
    assert machine.system.launched == UNITS[:6]                                    # 0, 1 fail, 2 passes, 3, 4, 5 fail: stop
    assert '3 infrastructure failures' in json.loads((machine.root/'queue_stop.json').read_text())['reason']
    run = RUN.run_dir(machine.root, 'S_P', RUN.PAIRS[0])
    assert (run/'exit_code').is_file() and not (run/'metrics.json').exists() and json.loads((run/'queue_done.json').read_text())['state'] == 'infrastructure'
    again = FakeSystem()                                                           # a restart reruns none of them
    again.classes = {}
    assert run_queue(machine, again) == 8 or again.launched[0] == UNITS[6]
    assert UNITS[0] not in again.launched


@pytest.mark.parametrize('cls', ('unsure', None))
def test_an_unsure_run_stops(machine, cls):
    machine.system.classes = {UNITS[2]: cls}
    assert run_queue(machine) == 8
    assert machine.system.launched == UNITS[:3] and 'classified' in json.loads((machine.root/'queue_stop.json').read_text())['reason']


def test_a_failed_check_stops(machine):
    machine.system.fail_check = True
    assert run_queue(machine) == 8
    assert machine.system.launched == [] and 'made up' in json.loads((machine.root/'queue_stop.json').read_text())['reason']


def test_a_failed_evaluation_stops(machine):
    machine.system.fail_eval = True
    assert run_queue(machine) == 8
    assert machine.system.launched == UNITS[:1]


def test_a_run_that_did_not_take_the_entry_stops_and_is_kept(machine):
    machine.system.bad_trace = {UNITS[1]}
    assert run_queue(machine) == 8
    assert machine.system.launched == UNITS[:2] and 'not injected' in json.loads((machine.root/'queue_stop.json').read_text())['reason']
    assert not (RUN.run_dir(machine.root, 'S_P', RUN.PAIRS[1])/'cleanup.json').exists()


def test_a_directory_without_an_end_stops_and_one_with_an_exit_code_is_postprocessed(machine):
    run = RUN.run_dir(machine.root, 'S_P', RUN.PAIRS[0])
    run.mkdir(parents=True)
    assert run_queue(machine) == 8
    assert 'incomplete' in json.loads((machine.root/'queue_stop.json').read_text())['reason'] and machine.system.launched == []
    # an ended unit whose queue was lost: post-processed, not rerun
    entry = RUN.check_entry('S_P', RUN.PAIRS[0], RUN.input_files('S_P', RUN.PAIRS[0], machine.results)[3], machine.results)
    fake_levels(run, entry)
    write_json(run/'seed_arm.json', dict(seconds=330))
    (run/'exit_code').write_text('0\n')
    (run/'run_id').write_text('seed_S_P_old\n')
    (run/'uptime_before').write_text('1.0\n')
    (run/'service_unit.txt').write_text(UNITS[0]+'\n')
    second = FakeSystem()
    assert run_queue(machine, second) == 0 and UNITS[0] not in second.launched and json.loads((run/'queue_done.json').read_text())['run_id'] == 'seed_S_P_old'


def test_the_root_must_be_named_seed_and_one_driver_at_a_time(tmp_path):
    system = FakeSystem()
    assert RUN.queue(options(tmp_path/'other'), system, tmp_path, tmp_path) == 2
    root = tmp_path/'seed_v1'
    root.mkdir()
    import fcntl
    with open(root/'queue.lock', 'w') as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert RUN.queue(options(root), system, tmp_path, tmp_path) == 2


def test_the_log_of_a_run_has_the_machine_state_around_the_unit(machine):
    assert run_queue(machine) == 0
    run = RUN.run_dir(machine.root, 'S_P', RUN.PAIRS[0])
    log = (run/'log.txt').read_text()
    assert f'launch {UNITS[0]}; fake state' in log and 'unit ended, systemd-run exit 0; fake state' in log
    assert 'frozen HEAD' in (run/'git.txt').read_text() and 'frozen git status --porcelain' in (run/'git.txt').read_text()


def test_the_state_reads_memory_and_the_gpu(monkeypatch):
    monkeypatch.setattr(RUN.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='fake gpu\n', returncode=0))
    text = RUN.System().state()
    assert 'loadavg' in text and 'MemAvailable' in text and 'fake gpu' in text


def test_a_disk_that_cannot_be_read_keeps_the_gate_closed(monkeypatch, tmp_path):
    def refuse(path):
        raise OSError('not mounted')
    monkeypatch.setattr(RUN.shutil, 'disk_usage', refuse)
    assert RUN.System().free_gib('/mnt/c') == 0.


def test_a_failed_reading_stops_with_a_record_and_the_runs_stay(machine, monkeypatch):
    def fail(root, results=None):
        raise SystemExit('long.csv missing')
    monkeypatch.setattr(DEC, 'write_analysis', fail)
    assert run_queue(machine) == 8
    assert 'the reading failed' in json.loads((machine.root/'queue_stop.json').read_text())['reason'] and machine.system.launched == UNITS


def test_an_old_stop_record_is_kept_under_its_time(machine):
    machine.system.fail_check = True
    assert run_queue(machine) == 8
    first = json.loads((machine.root/'queue_stop.json').read_text())
    assert run_queue(machine) == 8                                              # the directory of the first attempt is there now: another stop
    kept = list(machine.root.glob('queue_stop_*.json'))
    assert len(kept) == 1 and json.loads(kept[0].read_text()) == first
    assert 'incomplete' in json.loads((machine.root/'queue_stop.json').read_text())['reason']


def test_the_dry_run_cleans_up_however_it_ends(monkeypatch, neutral):
    def fail(*a):
        (neutral/'scratch/left').write_text('x')
        raise SystemExit('stopped')
    monkeypatch.setattr(RUN, 'dry_run_steps', fail)
    with pytest.raises(SystemExit):
        RUN.dry_run(scratch=neutral/'scratch')
    assert not (neutral/'scratch').exists()


def test_the_controls_are_part_of_the_start_checks(machine):
    (machine.results/'b1_v1/runs/B1TM'/RUN.PAIRS[1]/'metrics.json').unlink()
    assert run_queue(machine) == 8
    assert machine.system.launched == [] and 'control B1TM 0439-0440' in json.loads((machine.root/'queue_stop.json').read_text())['reason']


def frozen_link_tree(neutral):
    results = neutral/'results'
    link = results/RUN.FROZEN_NAME/'data/three_d/stage_L2R_val_v1'
    link.parent.mkdir(parents=True)
    link.symlink_to(neutral/'somewhere')
    target = neutral/'A5P'/'stage_L2R_val_v1'
    target.mkdir(parents=True)
    return results, link, target


@pytest.mark.parametrize('code', (0, 7))
def test_prepare_points_the_link_at_the_arm_and_restores_it(neutral, code):
    """The real System.prepare on a link in a temporary tree: at the arm's stage while prepare runs, at the original stage afterwards, whatever prepare did."""
    results, link, target = frozen_link_tree(neutral)
    seen = neutral/'seen.txt'
    script = f"import os,sys; open({str(seen)!r},'w').write(os.readlink({str(link)!r})); sys.exit({code})"
    assert RUN.System().prepare([PY, '-B', '-c', script], target, neutral/'prepare.log', results) == code
    assert seen.read_text() == str(target)
    assert os.readlink(link) == str((RUN.MAIN_REPO/'data/three_d/stage_L2R_val_v1').resolve())
    assert (results/'.frozen_eb808be_link.lock').exists()                         # the lock file of run_queue.sh


def test_a_refused_entry_is_classified_unsure_and_stops_the_queue(tmp_path):
    """The real classify_run.py: the wrapper refused before the task clock (exit 3, no started.json)."""
    write_json(tmp_path/'protocol.json', dict(task=dict(name='t')))
    (tmp_path/'exit_code').write_text('3\n')
    (tmp_path/'run.log').write_text('seed: refused: x\n')
    (tmp_path/'log.txt').write_text('')
    system = RUN.System()
    record = system.classify(tmp_path, system.uptime())                         # from now, as the queue does: t0 = 0 would count every kernel memory line since boot (a real run's OOM made this test fail)
    assert record['class'] == 'unsure' and record['kind'] == 'before_task_clock'


def test_evaluate_one_uses_the_evaluation_and_adds_the_entry_record(tmp_path, monkeypatch):
    entry = dict(u_sha256='b', tapered_sha256='t', field_path='f', field_sha256='s', source_run_id='r', oracle=False)
    fake_levels(tmp_path, entry)
    write_json(tmp_path/'seed_arm.json', dict(seconds=330))
    calls = []

    def evaluate_run(run, arm, pair, run_id, method=None):
        calls.append((Path(run), arm, pair, run_id, method))
        return dict(metrics=dict(seg35=.8, seg4=.9, sdlogj_official=.07), certificate=dict(status='certified_feasible', passed=True))
    monkeypatch.setattr(RUN, 'load_evaluate', lambda: SimpleNamespace(evaluate_run=evaluate_run))
    assert RUN.evaluate_one(SimpleNamespace(run=str(tmp_path), arm='S_P', pair=RUN.PAIRS[0], run_id='seed_x')) == 0
    assert calls == [(tmp_path, 'S_P', RUN.PAIRS[0], 'seed_x', RUN.method_name('S_P'))]
    written = json.loads((tmp_path/'metrics.json').read_text())
    assert written['seed']['passed'] is True and written['seed']['field_path'] == 'f' and written['seed']['arm'] == dict(seconds=330) and written['metrics']['seg35'] == .8
    with pytest.raises(SystemExit):
        RUN.evaluate_one(SimpleNamespace(run=str(tmp_path), arm='A5P', pair=RUN.PAIRS[0], run_id='x'))


def test_the_evaluation_does_not_certify_again():
    """The correction of the brief (3bd30a6): as B1, the run's own certificate, bound by state_hash and the npz sha256; no certify call in evaluate.py or here."""
    text = (ROOT/'experiments/d0/evaluate.py').read_text()
    assert 'certify_map' not in text and "state_hash_matches_u" in text and "final_npz_sha256_matches" in text
    assert 'certify_map' not in (ROOT/'experiments/seed/run_seed.py').read_text() and 'certify_map' not in (ROOT/'experiments/seed/decide_seed.py').read_text()


def test_cleanup_only_touches_intermediate_maps(tmp_path):
    write_json(tmp_path/'protocol.json', dict(task=dict(name='t')))
    task = tmp_path/'t'
    task.mkdir()
    for name in ('committed_0001_L3.npz', 'level_3_level_prepared.npz', 'final_native.npz', 'summary.json'):
        (task/name).write_text('x')
    RUN.cleanup_commits(tmp_path)
    assert sorted(p.name for p in task.iterdir()) == ['final_native.npz', 'summary.json']
    record = json.loads((tmp_path/'cleanup.json').read_text())
    assert [d['name'] for d in record['deleted']] == ['committed_0001_L3.npz', 'level_3_level_prepared.npz']


def test_the_replica_of_the_taper_is_the_formula():
    u = small_field()
    k = [np.minimum(np.minimum(np.arange(n), n-1-np.arange(n))/4., 1.) for n in SHAPE]
    w = k[0][:, None, None]*k[1][None, :, None]*k[2][None, None, :]
    assert RUN.boundary_taper_sha256(u, SHAPE) == hashlib.sha256((w[..., None]*u).tobytes()).hexdigest()
