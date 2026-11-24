"""EF (docs/bayesreg/notes/提速与换预分割器规划_20261104.md §6; runs/20261104-ef/plan.md): experiments/ef/{ef_core,repair,fast_channels,vector_colour,decide_ef,run_ef}.py.  CPU only, synthetic small volumes; no real image, label, prediction or field is read; nothing here starts a unit (the fake machine; the sandbox's systemd-run exits 1 on top of that).

- the repair: its J and gradient against bayesreg.geom and differences (check_repair_gradient.py), the mask, nothing moves outside it, a folded field that certify_map passes afterwards with J > 0 on the 53 points, a cell the 53 points cannot see, what a failure returns: never a pass it was not given;
- E2: labels, Q0, one-hot channels, pyramid, scale and field bit for bit against the old functions; refusals; the old entry untouched;
- E1: the K = 8 colouring keeps labels with a common face apart (and the check can fail), the scale rule;
- the guards: only the first 5 pairs, and exactly one exception; only this root;
- the firewall: a spy on np.load, its counterexample, changed truth labels, the source of the paths;
- the reading, every branch; the gate, the queue on a fake machine, the start strings; the dry run; a synthetic run through everything, in a subprocess.
"""
import ast
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
import torch

from jd_curl.three_d import interval as iv
from jd_curl.three_d.feasibility import PrescreenCertificateConfig, _classify, _sample_jacobians, certify_map, jacobian_intervals
from jd_curl.three_d.grid import make_grid_3d
from jd_curl.three_d.nifti import downsample_array, file_sha256

from bayesreg.geom import closed27, gauss27, node_field, tile_geometry
from bayesreg.grids import four_level_grids
from bayesreg.io import pair as pair_module
from bayesreg.seg import structure_layer

from . import test_d0_a4_tools as a4
from ..io_seg._helpers import DEV, source_sha, write_json
from ..io_seg._helpers import synthetic_folds  # noqa: E402   (C23.9-13 (8), revision 6e)
from .test_d0_a4_tools import MI, PAIRS, SHAPE, env, seg35  # noqa: F401

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT/'experiments/ef/run_ef.py'
FROZEN = Path('/home/rmcfz/seq_vp_tag_results/frozen_eb808be')
PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'
needs_frozen = pytest.mark.skipif(not (FROZEN/'jd_curl/three_d/preopt.py').is_file(), reason='the frozen worktree is not on this machine')


def _load():
    spec = importlib.util.spec_from_file_location('_ef_run', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules['_ef_run'] = module
    spec.loader.exec_module(module)
    return module


RUN = _load()
REAL_PROBLEMS = RUN.safe_input_problems
core, mc, ef, fast, repair, vc, decision, MC0, MC0P = RUN.core, RUN.mc, RUN.ef, RUN.fast, RUN.repair, RUN.vc, RUN.decision, RUN.run, RUN.mc0p
sys.path.insert(0, str(ROOT/'experiments/ef'))
import check_repair_gradient as check  # noqa: E402

CONFIG_C = PrescreenCertificateConfig(delta=0., max_depth=48, max_boxes=255, tile_size=65536)
torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def no_real_machine(monkeypatch):
    """A test never starts a process through the queue's machine: a fake replaces System; the real one raises."""
    def forbidden(*args, **kwargs):
        raise AssertionError('the real System.run was called')
    monkeypatch.setattr(MC0.System, 'run', forbidden)


# ---- the repair ---------------------------------------------------------------------------------------------------------------------------------
def out_to(lines):
    return lambda text: lines.append(text)


@pytest.mark.parametrize('name', ('check_j', 'check_gradient', 'check_dj', 'check_counterexample', 'check_taylor'))
def test_the_derivation_checks(name):
    lines = []
    assert bool(getattr(check, name)(out_to(lines))) is True, '\n'.join(lines)


fold_field = repair.synthetic_folded_field                                               # a smooth random field tapered to zero faces, with some folded cells


def j53(v, grid):
    field = node_field(v, grid)
    everything = slice(0, grid.n_cells)
    return np.concatenate((tile_geometry(field, gauss27(grid), tile=everything).J, tile_geometry(field, closed27(grid), tile=everything).J), axis=1)


def test_union53_is_the_union_of_the_two_sets():
    grid = make_grid_3d(nx=6, ny=7, nz=8, hx=1., hy=1.25, hz=.75, unit='mm')
    points, table = repair.union53(grid)
    assert points.shape == (53, 3) and table.shape == (3, 53, 8) and len({tuple(p) for p in points}) == 53
    assert {tuple(p) for p in gauss27(grid).points} | {tuple(p) for p in closed27(grid).points} == {tuple(p) for p in points}


def test_mask_nodes_are_interior_and_the_cells_touched_are_all_of_them():
    grid = make_grid_3d(nx=7, ny=6, nz=5, hx=1., hy=1., hz=1., unit='mm')
    cell = int(np.ravel_multi_index((2, 2, 3), grid.cell_shape))
    free = repair.mask_nodes([cell], grid, 0)
    want = sorted(np.ravel_multi_index((z, y, x), grid.shape) for z in (2, 3) for y in (2, 3) for x in (3, 4))
    assert list(free) == want
    wide = repair.mask_nodes([cell], grid, 1)
    z, y, x = np.unravel_index(wide, grid.shape)
    assert z.min() == 1 and z.max() == grid.shape[0]-2 and y.min() == 1 and x.min() == 2 and x.max() == 5
    assert set(free) < set(wide)
    corner = int(np.ravel_multi_index((0, 0, 0), grid.cell_shape))
    assert list(repair.mask_nodes([corner], grid, 0)) == [int(np.ravel_multi_index((1, 1, 1), grid.shape))]     # faces stay out
    cells = repair.touched_cells(free, grid)
    moved = np.isin(repair.cell_nodes(grid, np.arange(grid.n_cells)), free).any(axis=1)
    assert list(cells) == list(np.flatnonzero(moved))


def test_locator_holds_the_witness_and_every_failing_cell():
    u, grid = fold_field()
    result = certify_map(u, grid, config=CONFIG_C)
    assert result.status == 'infeasible' and result.n_infeasible > 0
    cells = repair.locate_cells(u, grid, CONFIG_C)
    assert result.witness_cell in cells and len(cells) >= result.n_infeasible
    samples = j53(u, grid).min(axis=1)
    assert set(np.flatnonzero(samples <= 0)) <= set(cells)                              # a negative sample cannot have been proved


def run_repair(u, grid, eps=.01, **kw):
    cells = repair.locate_cells(u, grid, CONFIG_C)
    return cells, repair.repair_field(u, grid, cells, repair.RepairConfig(eps=eps, **kw), certify=certify_map, certificate_config=CONFIG_C, device='cpu')


@pytest.mark.parametrize('eps', (.01, .05))
def test_a_folded_field_is_repaired_and_certify_map_passes_it(eps):
    u, grid = fold_field()
    before = u.copy()
    cells, result = run_repair(u, grid, eps)
    assert np.array_equal(u, before)                                                    # the input is not touched
    assert result.passed is True and result.reason == 'certified'
    again = certify_map(result.u, grid, config=CONFIG_C)                                # not the function's word: the certificate again, on what it returned
    assert repair.accepted(again) and result.certificate['state_hash'] == again.state_hash
    assert j53(result.u, grid).min() > 0
    changed = np.flatnonzero(np.any(result.u.reshape(-1, 3) != u.reshape(-1, 3), axis=1))
    assert set(changed) <= set(result.attempt_nodes)                                    # outside the mask, bit for bit
    outside = np.setdiff1d(np.arange(u.size//3), result.attempt_nodes)
    assert result.u.reshape(-1, 3)[outside].tobytes() == u.reshape(-1, 3)[outside].tobytes()
    assert len(result.attempt_nodes) < u.size//3/2                                      # local, on this field
    faces = np.zeros(grid.shape, dtype=bool)
    faces[1:-1, 1:-1, 1:-1] = True
    assert not np.any(result.u[~faces])
    record = repair.change_record(u, result.u)
    assert 0 < record['n_changed_nodes'] <= len(result.attempt_nodes) and record['max_change_mm'] > 0
    assert result.rounds and result.rounds[0]['rho'] == 1e2 and result.rounds[0]['radius'] == 1 and result.rounds[0]['n_seed'] == len(cells)


def test_a_cell_the_53_points_cannot_see_gets_its_lattice_point():
    """The cell of the derivation: J >= .016 on its 53 points, J < 0 between them.  With eps below that the first round leaves the cell alone, the certificate says no; the next round adds the lattice point."""
    u, grid, cell = check.sampled_positive_cell()
    assert j53(u, grid)[cell].min() >= .01
    p = jacobian_intervals(u.reshape(-1, 3)[repair.cell_nodes(grid, np.array([cell]))], grid.spacing)
    verdict = _classify(iv.lagrange_to_bernstein(p)[0], p[0], 0., float(_sample_jacobians(u.reshape(-1, 3)[repair.cell_nodes(grid, np.array([cell]))], grid.spacing).min()), CONFIG_C)
    assert verdict.status == 'infeasible'
    # only this cell is the problem of the round: the penalty sees nothing on it with the 53 points
    more_cells, more_points = repair.lattice_points(u, grid, [cell], .01, device='cpu')
    assert list(more_cells) == [cell] and len(more_points) == 1 and not np.all(np.isin(more_points[0], (0., .5, 1.)))
    # the penalty of this cell alone: without the point it does not see the dip; with it, it does
    free = repair.mask_nodes([cell], grid, 0)
    table = repair.union53(grid)[1][:, :, :]
    one_cell = repair.Objective(torch, u, u, grid, free, table, device='cpu', tile_cells=64)
    keep = torch.as_tensor(np.searchsorted(one_cell.cells, [cell]))
    one_cell.idx = one_cell.idx[keep]
    assert one_cell.value_and_grad(one_cell.start, 1e2, .01)[2]['active'] == 0
    with_point = repair.Objective(torch, u, u, grid, free, table, device='cpu', tile_cells=64, extras=(more_cells, more_points))
    with_point.idx = with_point.idx[keep]
    assert with_point.value_and_grad(with_point.start, 1e2, .01)[2]['active'] == 1 and with_point.value_and_grad(with_point.start, 1e2, .01)[2]['min_j'] < 0
    _, result = run_repair(u, grid, .01)
    assert result.passed is True and repair.accepted(certify_map(result.u, grid, config=CONFIG_C))


def test_a_field_positive_on_the_53_points_is_not_called_a_pass_and_the_next_round_adds_the_point():
    """certify says no for a cell the penalty cannot see: the rounds go on, and the locator's cell comes back as a penalty point.  What passes is what certify_map passes."""
    u, grid, cell = check.sampled_positive_cell()
    calls = []

    def recording(field, g, *, config):
        calls.append(field.copy())
        return certify_map(field, g, config=config)
    only_this = np.array([cell])
    result = repair.repair_field(u, grid, only_this, repair.RepairConfig(eps=.01, radius0=0, radius_step=0), certify=recording, certificate_config=CONFIG_C, device='cpu')
    assert result.passed is True and repair.accepted(certify_map(result.u, grid, config=CONFIG_C))
    assert len(result.rounds) >= 1 and result.rounds[-1]['certificate']['status'] == 'certified_feasible' and len(calls) == len(result.rounds)+1
    for row in result.rounds[:-1]:
        assert row['certificate']['status'] != 'certified_feasible' and row['n_located'] >= 1


def test_the_points_of_a_failed_round_go_into_the_next_objective(monkeypatch):
    """lattice_points is called on the located cells with the field reached, and the next round's objective has them; the seed grows by what was located and by the witness."""
    u, grid = fold_field()
    cells = repair.locate_cells(u, grid, CONFIG_C)[:5]
    seen, objectives, real, real_objective = [], [], repair.lattice_points, repair.Objective

    def spy(v, g, located, eps, *, device):
        seen.append((v.copy(), list(located), eps))
        return real(v, g, located, 1e9, device=device)                                  # every located cell gets a point

    def objective(*args, **kwargs):
        objectives.append(kwargs['extras'])
        return real_objective(*args, **kwargs)
    monkeypatch.setattr(repair, 'lattice_points', spy)
    monkeypatch.setattr(repair, 'Objective', objective)
    located = np.array([int(cells[0])+1, 7], dtype=np.int64)
    result = repair.repair_field(u, grid, cells, repair.RepairConfig(eps=.05, max_rounds=2, max_inner=3), certify=lambda f, g, *, config: Stub(witness_cell=11), certificate_config=CONFIG_C, device='cpu', locate=lambda v, g, c: located)
    assert len(seen) == 2 and seen[0][1] == list(located) and seen[0][2] == .05 and seen[0][0].tobytes() != u.tobytes()
    assert len(objectives[0][0]) == 0 and result.rounds[0]['n_extra_points'] == 0
    assert list(objectives[1][0]) == [c for c in located if objectives[1][0].tolist().count(c)] and result.rounds[1]['n_extra_points'] == len(objectives[1][0]) == result.rounds[0]['n_new_extra_points']
    assert result.rounds[1]['n_seed'] == len(np.union1d(np.union1d(cells, located), [11])) and result.rounds[1]['rho'] == 1e3 and result.rounds[1]['radius'] == 2


class Stub:
    """A certify result of any status: the repair must believe only what it says."""
    def __init__(self, **kw):
        self.status, self.reason, self.min_jd_lower_bound, self.n_cells_checked, self.n_infeasible, self.n_uncertain, self.witness_cell, self.witness_reference_point, self.state_hash, self.certificate_version = 'infeasible', 'JACOBIAN_VIOLATION', -.1, 1, 1, 0, None, None, 'x', 'stub'
        self.__dict__.update(kw)


def test_a_repair_that_never_passes_returns_the_input_and_no_pass(monkeypatch):
    """The real solve; a certificate that always says no.  Even a field with every J positive is not called a pass."""
    u, grid = fold_field()
    calls = []

    def never(field, grid, *, config):
        calls.append(field.copy())
        return Stub()
    cells = repair.locate_cells(u, grid, CONFIG_C)
    result = repair.repair_field(u, grid, cells, repair.RepairConfig(eps=.05, max_rounds=2, max_inner=12), certify=never, certificate_config=CONFIG_C, device='cpu', locate=lambda v, g, c: cells)
    assert result.passed is False and result.reason == 'round_cap' and len(result.rounds) == 2 and result.certificate['status'] == 'infeasible'
    assert result.u.tobytes() == u.tobytes()                                            # back to the frozen rule's input, not the last attempt
    assert len(calls) == 3 and result.attempt_values.shape == (len(result.attempt_nodes), 3)
    attempt = u.copy()
    attempt.reshape(-1, 3)[result.attempt_nodes] = result.attempt_values
    assert attempt.tobytes() == calls[-1].tobytes() and attempt.tobytes() != u.tobytes()   # the evidence of the failure stays


def test_a_real_certificate_failure_is_reported_as_one():
    """One inner step of a weak penalty cannot repair this field: certify_map says no, and so does the result."""
    u, grid = fold_field(amplitude=6.)
    _, result = run_repair(u, grid, .01, max_rounds=1, max_inner=1, rho0=1e-3)
    assert result.passed is False and result.reason == 'round_cap' and result.certificate['status'] != 'certified_feasible'
    assert result.u.tobytes() == u.tobytes()


@pytest.mark.parametrize('kw', (dict(status='uncertain', n_infeasible=0, n_uncertain=1, min_jd_lower_bound=None), dict(status='certified_feasible', n_infeasible=0, min_jd_lower_bound=0.), dict(status='certified_feasible', n_infeasible=0, min_jd_lower_bound=None), dict(status='certified_feasible', n_infeasible=1, min_jd_lower_bound=.1), dict(status='certified_feasible', n_infeasible=0, n_uncertain=1, min_jd_lower_bound=.1)))
def test_zero_and_undecided_do_not_pass(kw):
    assert repair.accepted(Stub(**kw)) is False


def test_a_field_already_certified_is_returned_as_it_is():
    grid = make_grid_3d(nx=7, ny=7, nz=7, hx=1., hy=1., hz=1., unit='mm')
    u = np.zeros(grid.shape+(3,))
    result = repair.repair_field(u, grid, np.empty(0, dtype=np.int64), repair.RepairConfig(eps=.01), certify=certify_map, certificate_config=CONFIG_C, device='cpu')
    assert result.passed is True and result.reason == 'already_certified' and result.rounds == [] and result.u.tobytes() == u.tobytes() and len(result.attempt_nodes) == 0


def test_the_wall_ends_the_rounds_and_is_not_a_pass():
    u, grid = fold_field()
    cells = repair.locate_cells(u, grid, CONFIG_C)
    now = iter(np.arange(0., 1e6, 100.))
    result = repair.repair_field(u, grid, cells, repair.RepairConfig(eps=.05, wall_seconds=50.), certify=lambda f, g, *, config: Stub(), certificate_config=CONFIG_C, device='cpu', clock=lambda: float(next(now)))
    assert result.passed is False and result.reason == 'wall' and len(result.rounds) == 1 and result.rounds[0]['inner']['stop'] == 'wall'


def test_the_warm_up_passes_a_read_only_field_through_the_locator():
    """EF_tools_复核 B1: the first locate of a repair is on a read-only field, and numba compiles per array type."""
    located, certified = [], []
    RUN.tiny_warmup(lambda u, grid, *, config: certified.append(u.flags.writeable), CONFIG_C, lambda u, grid, config: located.append(u.flags.writeable))
    assert located == [True, False] and certified[-1] is False and True in certified


def test_the_device_warm_up_walks_both_branches_and_the_lattice_on_a_small_field(monkeypatch):
    """EF_tools_复核 R1, on the CPU here: what the first use on a GPU would pay for inside the clock is what this runs.  Table and per-row blocks, each with active points, and lattice_points."""
    blocks_seen, lattices, real_block, real_lattice = [], [], repair.Objective._block, repair.lattice_points

    def block(self, X, idx, D, rho, eps, out, stats, *, per_row=False):
        before = stats['active']
        value = real_block(self, X, idx, D, rho, eps, out, stats, per_row=per_row)
        blocks_seen.append((per_row, stats['active'] > before, out is not None))
        return value

    def lattice(*args, **kwargs):
        got = real_lattice(*args, **kwargs)
        lattices.append(len(got[0]))
        return got
    monkeypatch.setattr(repair.Objective, '_block', block)
    monkeypatch.setattr(repair, 'lattice_points', lattice)
    record = RUN.device_warmup('cpu', certify_map, CONFIG_C)
    assert record['rounds'] == 2 and record['n_per_row_points'] > 0 and record['device'] == 'cpu' and record['n_located'] > 0
    assert (False, True, True) in blocks_seen and (True, True, True) in blocks_seen and lattices


def test_the_device_warm_up_is_before_the_clock_of_the_repair():
    """In repair_one: the warm-ups, then the first locate, which is the first thing seconds_repair counts."""
    source = SCRIPT.read_text()
    body = source[source.index('def repair_one(o):'):source.index('# ---- evaluation')]
    assert body.index('tiny_warmup(certify_map') < body.index('device_warmup(o.device') < body.index('cells = repair.locate_cells(u, pin.grid, config)') < body.index('seconds_repair = seconds_first_locate')
    assert body.index('limit_device_memory') < body.index('device_warmup(o.device')


READ_ONLY_LOCATE = """
import sys, time
sys.path.insert(0, sys.argv[1]); sys.path.insert(1, sys.argv[1]+'/../mc0')
import numpy as np
import mc0_core as core
fz = core.frozen_api()
import repair
import run_ef as run
from jd_curl.three_d.feasibility import certify_map
config = run.run.certificate_config()
if sys.argv[2] == 'warm-up':
    run.tiny_warmup(certify_map, config, repair.locate_cells)
else:                                                    # the control: the kernels compiled for writable arrays only, as the first version of the warm-up did
    small, grid = repair.synthetic_folded_field(n=8)
    certify_map(small, grid, config=config)
    repair.locate_cells(small, grid, config)
raw, grid = repair.synthetic_folded_field(n=12)
u, _ = fz.boundary_taper(raw, grid, core.TAPER)
u = np.ascontiguousarray(u)
assert u.flags.writeable is False                        # what source_checked hands to repair_one
tick = time.perf_counter()
cells = repair.locate_cells(u, grid, config)
seconds = time.perf_counter()-tick
assert len(cells) > 0
print(seconds)
"""


@needs_frozen
def test_after_the_warm_up_the_first_read_only_locate_compiles_nothing(tmp_path):
    """In a new process, as each repair-one is: under .5 s after tiny_warmup (the review measured 4.97 s of compile without it).  The control shows the time can be seen."""
    env_ = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=f'{FROZEN}:{ROOT}')
    seconds = {}
    for mode in ('warm-up', 'control'):
        done = subprocess.run([PY, '-B', '-c', READ_ONLY_LOCATE, str(ROOT/'experiments/ef'), mode], env=env_, cwd=tmp_path, capture_output=True, text=True)
        assert done.returncode == 0, done.stdout[-2000:]+done.stderr[-2000:]
        seconds[mode] = float(done.stdout.split()[-1])
    assert seconds['warm-up'] < .5 < seconds['control'], seconds


def test_change_record_is_from_the_fields():
    a = np.zeros((3, 3, 3, 3))
    b = a.copy()
    b[1, 1, 1] = (3., 4., 0.)
    b[1, 1, 2] = (0., 0., 1.)
    record = repair.change_record(a, b)
    assert record['n_changed_nodes'] == 2 and record['max_change_mm'] == 5. and record['mean_change_mm'] == 3.
    assert repair.change_record(a, a.copy())['n_changed_nodes'] == 0


def test_the_certificate_configuration_is_c():
    config = MC0.certificate_config()
    assert isinstance(config, PrescreenCertificateConfig) and config.delta == 0. and config.max_depth == 48 and config.max_boxes == 255 and config.tile_size == 65536 and config.fallback_kernel == CONFIG_C.fallback_kernel and config.prescreen_kernel == CONFIG_C.prescreen_kernel
    source = SCRIPT.read_text()
    assert source.count('run.certificate_config()') == 3 and 'PrescreenCertificateConfig' not in source     # the repair, the evaluation and the smoke test take it from the one place


# ---- E2: the same numbers ---------------------------------------------------------------------------------------------------------------------------
def blocks(shape=(30, 34, 28), seed=4):
    """A label map with large and small labels, one touching the faces, some absent; bit-for-bit tests need the kernel's reach to touch them."""
    rng = np.random.default_rng(seed)
    L = np.zeros(shape, dtype=np.int16)
    L[4:26, 5:29, 4:24] = 1
    L[8:20, 8:24, 8:20] = 20
    L[0:6, 0:9, 0:7] = 2
    L[-3:, -5:, -4:] = 3
    L[12:14, 12:15, 12:13] = 7
    L[20, 20, 20] = 11
    for label in (13, 14, 15):
        L[tuple(rng.integers(1, s-1) for s in shape)] = label
    return L


def test_constants_of_g75():
    assert (fast.SIGMA, fast.TRUNCATE, fast.RADIUS) == (MI.SIGMA, 4.0, 3)


@pytest.mark.parametrize('seed', (4, 5))
def test_onehot_channels_bit_for_bit(seed):
    L = blocks(seed=seed)
    image = np.random.default_rng(seed).uniform(.1, 1., L.shape)
    old = core.channels('P2', image, L, workers=1)
    assert old.tobytes() == fast.onehot_channels(L, workers=1).tobytes() == fast.onehot_channels(L, workers=3).tobytes()
    assert np.signbit(fast.onehot_channels(L)).sum() == 0


@pytest.mark.parametrize('seed', (4, 5))
def test_q0_bit_for_bit_and_its_refusals(seed):
    L = blocks(seed=seed)
    image = np.random.default_rng(seed).uniform(.1, 1., L.shape)
    table = core.load_table()
    assert core.channels('P0', image, L, table=table).tobytes() == fast.q0_image(image, L, table).tobytes()
    with pytest.raises(SystemExit, match='no cerebral white matter'):
        fast.q0_image(image, np.zeros_like(L), table)
    bad = L.copy()
    bad[0, 0, 0] = 36
    with pytest.raises(SystemExit, match='outside'):
        fast.q0_image(image, bad, table)
    bad[0, 0, 0] = -1
    with pytest.raises(SystemExit, match='outside'):
        fast.q0_image(image, bad, table)


def test_the_pyramid_bit_for_bit_every_level_and_channel():
    grid = make_grid_3d(nx=34, ny=40, nz=44, hx=1., hy=1., hz=1., unit='mm')
    L = blocks((grid.shape[0], grid.shape[1], grid.shape[2]))
    channels = fast.onehot_channels(L)
    levels = four_level_grids(grid)
    downsample = fast.FastDownsample()
    for c in range(len(channels)):
        for level in range(3):
            a, got_a, _ = downsample_array(channels[c], grid, levels[level].shape)
            b, got_b, _ = downsample(channels[c], grid, levels[level].shape)
            assert a.tobytes() == b.tobytes() and got_a.grid_id == got_b.grid_id, (c, level)
    assert len(downsample.boxes) == len(channels)                                       # once per channel


def test_the_pyramid_on_a_channel_touching_every_face():
    grid = make_grid_3d(nx=41, ny=45, nz=49, hx=1., hy=1.25, hz=.75, unit='mm')
    values = gaussian_filter(np.random.default_rng(2).uniform(size=grid.shape), 1.)
    levels = four_level_grids(grid)
    for level in range(3):
        assert downsample_array(values, grid, levels[level].shape)[0].tobytes() == fast.FastDownsample()(values, grid, levels[level].shape)[0].tobytes()


def world_of(env, monkeypatch):  # noqa: F811
    monkeypatch.setattr(core, 'STAGE', env['stage'])
    monkeypatch.setattr(core, 'ALLOWLIST', env['allowlist'])
    monkeypatch.setattr(mc, 'SEG_ROOT', env['seg_root'])
    monkeypatch.setattr(mc, 'default_unet', lambda: env['unet'])
    assert core.PAIRS[0] == f'{PAIRS[0][0]}__{PAIRS[0][1]}'
    return core.PAIRS[0]


@synthetic_folds('env')
def test_labels_and_record_bit_for_bit_against_the_old_reader(env, monkeypatch):  # noqa: F811
    pair = world_of(env, monkeypatch)
    pin = mc.read_images(pair)
    old = mc.read_predicted(pin)
    for workers in (1, 3):
        new = fast.read_predicted(pin, workers=workers)
        assert new.soft is None and old.soft is None
        for side in ('fixed', 'moving'):
            assert old.labels[side].dtype == new.labels[side].dtype and old.labels[side].tobytes() == new.labels[side].tobytes()
            a, b = old.record['sides'][side], new.record['sides'][side]
            assert b['read_path'] == 'fast' and set(a) == set(b)-{'read_path'}
            for key in a:
                if key == 'read_back_probs36':
                    assert a[key]['raw_max_abs_sum_deviation'] == b[key]['raw_max_abs_sum_deviation'] and a[key]['stored_dtype'] == b[key]['stored_dtype'] and a[key]['raw_sum_tol'] == b[key]['raw_sum_tol']
                elif key == 'group_sum_max_abs_deviation':
                    assert a[key] == pytest.approx(b[key], abs=1e-12)
                else:
                    assert a[key] == b[key], key
        assert {k: new.record[k] for k in ('oracle', 'truth_labels_read', 'pair_keys_read', 'label_firewall')} == {k: old.record[k] for k in ('oracle', 'truth_labels_read', 'pair_keys_read', 'label_firewall')}


def test_ties_go_to_the_smaller_label_as_the_old_argmax(tmp_path):
    p = np.zeros((36, 2, 2, 2), dtype=np.float16)
    p[0] = .5
    p[3] = .5                                                                           # a tie with 0: label 0
    p[:, 1, 1, 1] = 0
    p[7, 1, 1, 1], p[5, 1, 1, 1] = .5, .5                                               # a tie: 5
    p[:, 0, 0, 0] = 0
    p[9, 0, 0, 0] = 1.
    path = tmp_path/'probs36.npz'
    np.savez_compressed(path, p=p)
    labels, p0, record, deviation = fast.raw_labels(path, p.shape, np.zeros((1, 2, 2, 2))+1, ((0,)+tuple(range(1, 36)),))
    want = np.argmax(p.astype(np.float64)/p.astype(np.float64).sum(axis=0), axis=0)
    assert labels.tolist() == want.tolist() and labels[1, 1, 1] == 5 and labels[0, 0, 0] == 9 and labels[0, 0, 1] == 0
    assert deviation == 0.


def test_argmax_of_bit_patterns_agrees_on_awkward_values(tmp_path):
    rng = np.random.default_rng(9)
    raw = rng.dirichlet(np.full(36, .05), size=(7, 8, 9))
    raw = np.moveaxis(raw, -1, 0).astype(np.float16)                                    # subnormals and near-ties included
    raw[raw < 2e-7] = 0
    path = tmp_path/'probs36.npz'
    np.savez_compressed(path, p=raw)
    total = raw.astype(np.float64).sum(axis=0)
    labels, _, _, _ = fast.raw_labels(path, raw.shape, np.ones((1,)+raw.shape[1:]), (tuple(range(36)),), workers=3)
    assert labels.tobytes() == np.argmax(raw.astype(np.float64)/total, axis=0).astype(np.int16).tobytes()


def test_special_patterns_go_to_the_old_path_and_refusals_are_refusals(tmp_path):
    good = np.zeros((36, 2, 2, 2), dtype=np.float16)
    good[1] = 1.
    path = tmp_path/'probs36.npz'
    groups = np.ones((1, 2, 2, 2))
    for special in (-0.5, np.nan, np.inf, -0.):
        bad = good.copy()
        bad[2, 0, 0, 0] = special
        np.savez_compressed(path, p=bad)
        assert fast.raw_labels(path, bad.shape, groups, (tuple(range(36)),)) is None     # the old reader decides
    np.savez_compressed(path, p=good.astype(np.float32))
    assert fast.raw_labels(path, good.shape, groups, (tuple(range(36)),)) is None        # another storage
    np.savez_compressed(path, p=good)
    assert fast.raw_labels(path, (36, 2, 2, 3), groups, (tuple(range(36)),)) is None     # another shape
    bad = good.copy()
    bad[1, 0, 0, 0] = .9
    np.savez_compressed(path, p=bad)
    with pytest.raises(SystemExit, match='raw channel sum'):
        fast.raw_labels(path, bad.shape, groups, (tuple(range(36)),))
    bad = good.copy()
    bad[:, 0, 0, 0] = 0
    np.savez_compressed(path, p=bad)
    with pytest.raises(SystemExit, match='zero total'):
        fast.raw_labels(path, bad.shape, groups, (tuple(range(36)),))


@synthetic_folds('env')
def test_stacks_and_scale_bit_for_bit_against_the_old_path(env, monkeypatch):  # noqa: F811
    pair = world_of(env, monkeypatch)
    pin = mc.read_images(pair)
    table = core.load_table()
    predicted = mc.read_predicted(pin)
    q0 = mc.q0_channels(pin, predicted, table)
    old, old_record = mc.prepare_stacks('Q2', pin, predicted, q0, 'cpu')
    new, new_record = fast.prepare_stacks(ef.ARMS['L150'], pin, fast.read_predicted(pin, workers=3), table, 'cpu', workers=3)
    assert old[0].tobytes() == new[0].tobytes() and old[1].tobytes() == new[1].tobytes()
    for key in ('D_multiplier', 'channel_multiplier', 'g_norm_Q0', 'g_norm_unscaled'):
        assert old_record[key] == new_record[key], key


@needs_frozen
def test_field_bit_for_bit_with_the_fast_pyramid(tmp_path):
    code = """
import sys, hashlib
sys.path.insert(0, sys.argv[1]); sys.path.insert(1, sys.argv[1]+'/../mc0')
import numpy as np
import mc0_core as core
fz = core.frozen_api()
import fast_channels as fast
from jd_curl.three_d.grid import make_grid_3d
grid = make_grid_3d(nx=30, ny=28, nz=26, hx=1., hy=1.25, hz=.75, unit='mm')
L = np.zeros(grid.shape, dtype=np.int16); L[5:20, 6:22, 7:24] = 1; L[8:14, 9:15, 10:16] = 20; L[3:6, 3:5, 3:9] = 7
M = np.roll(L, 2, axis=1)
config = fz.PreoptConfig(**dict(core.PRE_C, iterations=(2, 2, 2, 3), seconds=600.))
image = np.zeros(grid.shape)
a = core.optimize_k([core.channels('P2', image, L), core.channels('P2', image, M)], grid, config, device='cpu', workers=1)[0]
b = core.optimize_k([fast.onehot_channels(L), fast.onehot_channels(M)], grid, config, device='cpu', workers=1, downsample=fast.FastDownsample())[0]
assert a.tobytes() == b.tobytes() and np.abs(a).max() > 0
print('same')
"""
    env_ = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=f'{FROZEN}:{ROOT}')
    done = subprocess.run([PY, '-B', '-c', code, str(ROOT/'experiments/ef')], env=env_, cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0 and 'same' in done.stdout, done.stdout+done.stderr


def test_the_old_entry_has_not_changed():
    """No EF module in experiments/mc0; its default arguments as they were."""
    for name in ('mc0_core.py', 'mc0p_core.py', 'run_mc0.py', 'run_mc0p.py', 'run_mc0pc.py', 'decide_mc0.py', 'decide_mc0p.py'):
        tree = ast.parse((ROOT/'experiments/mc0'/name).read_text())
        imported = {a.name for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names} | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert not imported & {'fast_channels', 'vector_colour', 'repair', 'ef_core', 'decide_ef', 'run_ef'}, name


# ---- E1: K = 8 ---------------------------------------------------------------------------------------------------------------------------------------
def face_pairs(L):
    """Labels with a common voxel face, 1..35, by brute force."""
    out = set()
    for axis in range(3):
        a, b = np.moveaxis(L, axis, 0)[:-1].ravel(), np.moveaxis(L, axis, 0)[1:].ravel()
        for x, y in zip(a[a != b], b[a != b]):
            if x and y:
                out.add((min(int(x), int(y)), max(int(x), int(y))))
    return out


def random_labels(shape=(20, 22, 24), seed=7):
    """Blocks of random labels: many contacts."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 36, tuple(s//3+1 for s in shape))
    return np.kron(coarse, np.ones((3, 3, 3), dtype=np.int16))[:shape[0], :shape[1], :shape[2]].astype(np.int16)


def planar_labels(shape=(40, 44, 48)):
    """A label map like the real ones: few contacts each; the real graph needs 7 or 8 colours (runs/20261104-ef/e2_profile.out).  Layers, so K = 8 is enough by construction: only neighbouring layers touch."""
    L = np.zeros(shape, dtype=np.int16)
    z = np.arange(shape[0])
    L[1:-1, 1:-1, 1:-1] = 1+(z[1:-1]*35//shape[0])[:, None, None]
    return L


def test_face_contact_against_brute_force():
    L = random_labels()
    assert {(int(a), int(b)) for a, b in np.argwhere(vc.face_contact(L)) if a < b} == face_pairs(L)


def test_contact_matrix_at_one_holds_the_faces_and_more():
    L = random_labels()
    near = vc.contact_matrix(L, 1)
    assert np.all(near[vc.face_contact(L)]) and near.sum() >= vc.face_contact(L).sum()
    assert np.all(near == near.T) and not near.diagonal().any() and not near[0].any()


def test_labels_with_a_common_face_have_codes_at_distance_sqrt_two():
    """In the predicted labels of the fixed image (and of the moving): the brief's guarantee, on voxels: the two sides of a face differ in the code."""
    fixed, moving = planar_labels(), np.roll(planar_labels(), 2, axis=1)
    colours, record = vc.choose_colouring(fixed, moving, K=8)
    table = vc.codes(colours, 8)
    assert table.shape == (36, 8) and np.all(table[0] == 0) and np.all(table[1:].sum(axis=1) == 1)
    for L in (fixed, moving):
        for a, b in face_pairs(L):
            assert np.linalg.norm(table[a]-table[b]) == pytest.approx(np.sqrt(2.))
        for axis in range(3):
            A, B = np.moveaxis(L, axis, 0)[:-1], np.moveaxis(L, axis, 0)[1:]
            on_face = (A != B) & (A > 0) & (B > 0)
            assert np.all(np.abs(table[A[on_face]]-table[B[on_face]]).sum(axis=1) > 0)
    assert record['K'] == 8 and sorted(sum(record['members'], [])) == list(range(1, 36))


def test_the_separation_check_can_fail():
    L = planar_labels()
    with pytest.raises(vc.ColouringFailed, match='common face'):
        vc.check_separated(L, {label: 0 for label in range(1, 36)})
    assert vc.check_separated(L, vc.choose_colouring(L, L)[0]) == pytest.approx(np.sqrt(2.))


def test_too_few_colours_are_a_failure_not_a_merge():
    L = random_labels()
    assert vc.face_contact(L).sum(axis=1).max() > 8
    with pytest.raises(vc.ColouringFailed, match='do not fit'):
        vc.choose_colouring(L, L, K=2)


def test_the_colouring_is_deterministic_and_the_moving_graph_counts():
    fixed, moving = planar_labels(), np.roll(planar_labels(), 7, axis=0)
    first, _ = vc.choose_colouring(fixed, moving)
    assert first == vc.choose_colouring(fixed, moving)[0]
    moving_only = np.zeros_like(fixed)
    moving_only[5:10, 5:10, 5:10] = 3
    moving_only[10:15, 5:10, 5:10] = 30                                                 # 3 and 30 touch only in the moving image
    colours, _ = vc.choose_colouring(np.zeros_like(fixed), moving_only)
    assert colours[3] != colours[30]


def test_the_channels_of_k8_are_g75_of_the_colour_masks():
    L = planar_labels()
    colours, _ = vc.choose_colouring(L, L)
    out = vc.vector_channels(L, colours, 8, smooth=MI.smooth, workers=2)
    assert out.shape == (8,)+L.shape
    for c in range(8):
        members = [label for label, colour in colours.items() if colour == c]
        assert out[c].tobytes() == MI.smooth(np.isin(L, members).astype(np.float64)).tobytes()


@synthetic_folds('env')
def test_the_scale_rule_of_k8_gives_the_norm_of_q0(env, monkeypatch):  # noqa: F811
    pair = world_of(env, monkeypatch)
    pin = mc.read_images(pair)
    predicted = fast.read_predicted(pin)
    labels = planar_labels(pin.fixed.shape).astype(np.int16)                            # the stand-in posteriors are noise, with too many contacts for 8 colours
    labels[labels == labels[1, 1, 1]] = 1                                               # the white matter Q0's scale asks for
    predicted.labels['fixed'], predicted.labels['moving'] = labels, np.roll(labels, 1, axis=1)
    stacks, record = fast.prepare_stacks(ef.ARMS['K8'], pin, predicted, core.load_table(), 'cpu')
    assert stacks[0].shape[0] == 8 and record['colouring']['K'] == 8
    reference = core.data_grad_norm(*mc.q0_channels(pin, predicted, core.load_table()), 'cpu')
    assert core.data_grad_norm(stacks[0], stacks[1], 'cpu') == pytest.approx(reference, rel=core.SCALE_TOL)
    assert record['g_norm_Q0'] == reference and record['D_multiplier'] == core.scale_factor(reference, record['g_norm_unscaled'])


# ---- the arms and the guards ------------------------------------------------------------------------------------------------------------------------------
def test_the_scale_check_reads_the_native_level_and_a_coarse_stop_has_none():
    norms = [dict(raw=9.)]*3
    assert RUN.scale_check_of(dict(level_g_norm=norms), dict(g_norm_Q0=1.))['passed'] is None        # the wall on a coarse level: no IndexError, no run lost
    assert RUN.scale_check_of(dict(level_g_norm=norms+[dict(raw=1.0005)]), dict(g_norm_Q0=1.))['passed'] is True
    assert RUN.scale_check_of(dict(level_g_norm=norms+[dict(raw=1.01)]), dict(g_norm_Q0=1.))['passed'] is False


def test_the_arms_are_the_briefs():
    assert {arm: (s['k'], s['native'], s['sigma_field']) for arm, s in ef.ARMS.items()} == dict(Q2=(35, 600, 1.), L150=(35, 150, 1.), L300=(35, 300, 1.), F150=(35, 150, .5), F300=(35, 300, .5), F600=(35, 600, .5), K8=(8, 600, 1.))
    assert [a for a, s in ef.ARMS.items() if s['decides']] == ['L150', 'L300', 'F150', 'F300', 'F600', 'K8'] and ef.ARMS['Q2']['channels'] == 'old'
    assert {a: (s['tier'], s['eps'], s['decides']) for a, s in ef.REPAIR_ARMS.items()} == dict(RF01=('fine', .01, True), RF05=('fine', .05, True), RL01=('long', .01, False), RL05=('long', .05, False))
    for arm, spec in ef.ARMS.items():
        config = ef.arm_config(arm)
        assert config['iterations'] == (50, 50, 100, spec['native']) and config['sigma_field'] == spec['sigma_field'] and {k: config[k] for k in ('lr', 'cap', 'sigma_grad', 'mu', 'alpha', 'kappa_pre', 'j0')} == {k: core.PRE_C[k] for k in ('lr', 'cap', 'sigma_grad', 'mu', 'alpha', 'kappa_pre', 'j0')}
    assert ef.arm_config('Q2') == core.tier_config('long') and ef.arm_config('F600') == core.tier_config('fine')      # the baseline and the control are MC0P's tiers


def test_the_plan():
    plan = ef.plan_runs()
    assert len(plan) == len(set(plan)) == 49 and sum(c.is_repair for c in plan) == 14
    assert (plan[0].arm, plan[0].tier, plan[0].pair) == ('Q2', 'long', core.PAIRS[0]) and plan[0].run_id == 'ef_Q2_long_0438-0439' and plan[0].unit == 'tag-ef-q2-long-0438-0439'
    assert all(c.run_id.startswith('ef_') and c.unit.startswith('tag-ef-') and 'seq-vp' not in c.unit for c in plan)
    assert [c.is_repair for c in plan] == [False]*35+[True]*14                          # the entries first: nothing there depends on a repair
    assert {c.pair for c in plan if not c.is_repair} == set(core.PAIRS)
    assert {(c.arm, c.short) for c in plan if c.tier == 'long' and c.is_repair} == {(a, s) for a in ('RL01', 'RL05') for s in ('0439-0440', '0455-0456')}
    assert sum(c.tier == 'fine' and c.is_repair for c in plan) == 10
    assert core.run_dir('/r', ef.Cell('RF01', 'fine', core.PAIRS[0])) == Path('/r/runs/RF01_fine/0438-0439')


OUTSIDE = ('OASIS_OAS1_0443_MR1__OASIS_OAS1_0444_MR1', 'OASIS_OAS1_0454_MR1__OASIS_OAS1_0455_MR1', 'OASIS_OAS1_0020_MR1__OASIS_OAS1_0021_MR1', 'S37__S38', '../x')


@pytest.mark.parametrize('pair', OUTSIDE)
@pytest.mark.parametrize('arm,tier', (('Q2', 'long'), ('L150', 'long'), ('F600', 'fine'), ('K8', 'long'), ('RF01', 'fine'), ('RF05', 'fine'), ('RL01', 'long'), ('RL05', 'long')))
def test_every_other_pair_is_refused_before_a_file_opens(pair, arm, tier, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('a file was opened')
    monkeypatch.setattr(np, 'load', forbidden)
    with pytest.raises(SystemExit, match='first 5'):
        ef.require_cell(arm, tier, pair)
    args = RUN.argparse.Namespace(arm=arm, tier=tier, pair=pair, out='/nonexistent/x/y/z', device='cpu', code_sha='n/a')
    with pytest.raises(SystemExit, match='first 5'):
        (RUN.repair_one if arm in ef.REPAIR_ARMS else RUN.run_one)(args)


@pytest.mark.parametrize('arm,tier', (('Q2', 'long'), ('L150', 'long'), ('L300', 'long'), ('F150', 'fine'), ('F300', 'fine'), ('F600', 'fine'), ('K8', 'long'), ('RF01', 'fine'), ('RF05', 'fine')))
def test_the_exception_pair_is_refused_everywhere_but_in_the_long_repair(arm, tier):
    with pytest.raises(SystemExit, match='first 5'):
        ef.require_cell(arm, tier, ef.EXCEPTION_PAIR)


def test_the_exception_is_exactly_this_cell():
    assert ef.EXCEPTION_PAIR == 'OASIS_OAS1_0455_MR1__OASIS_OAS1_0456_MR1' and ef.EXCEPTION_PAIR not in core.PAIRS
    for arm in ('RL01', 'RL05'):
        assert ef.require_cell(arm, 'long', ef.EXCEPTION_PAIR).is_repair
        assert ef.require_cell(arm, 'long', core.PAIRS[1]).is_repair
        for pair in (core.PAIRS[0], core.PAIRS[2], core.PAIRS[3], core.PAIRS[4]):       # alpha = 1 passes there: nothing to repair
            with pytest.raises(SystemExit, match='first 5'):
                ef.require_cell(arm, 'long', pair)
    allowed = {(c.arm, c.pair) for c in ef.plan_runs() if c.pair == ef.EXCEPTION_PAIR}
    assert allowed == {('RL01', ef.EXCEPTION_PAIR), ('RL05', ef.EXCEPTION_PAIR)}
    assert ef.source_dir(ef.Cell('RL01', 'long', ef.EXCEPTION_PAIR)) == Path('/home/rmcfz/seq_vp_tag_results/mc0p_v1c/runs/Q2_long/0455-0456')
    assert ef.source_dir(ef.Cell('RL05', 'long', core.PAIRS[1])) == Path('/home/rmcfz/seq_vp_tag_results/mc0p_v1/runs/Q2_long/0439-0440')
    assert ef.require_truth_pair(ef.EXCEPTION_PAIR) == ef.EXCEPTION_PAIR
    with pytest.raises(SystemExit, match='first 5'):
        ef.require_truth_pair(OUTSIDE[0])
    with pytest.raises(SystemExit, match='first 5'):
        mc.read_images(ef.EXCEPTION_PAIR)                                               # MC0P's own guard stands
    with pytest.raises(SystemExit, match='first 5'):
        core.require_pair(ef.EXCEPTION_PAIR)


@pytest.mark.parametrize('arm,tier', (('Q2', 'fine'), ('L150', 'fine'), ('F150', 'long'), ('RF01', 'long'), ('RL01', 'fine'), ('Q9', 'long'), ('P2', 'long')))
def test_unknown_cells_are_refused(arm, tier):
    with pytest.raises(SystemExit, match='no cell'):
        ef.require_cell(arm, tier, core.PAIRS[1])


def test_the_sources_are_only_the_read_only_fields():
    assert set(ef.SOURCES.values()) == {'mc0p_v1', 'mc0p_v1b', 'mc0p_v1c'} and len(ef.SOURCES) == 7
    assert {pair for tier, pair in ef.SOURCES if tier == 'fine'} == set(core.PAIRS)
    for name in ('mc0p_v1', 'mc0p_v1b', 'mc0p_v1c', 'mc0_v1', 'seed_v1'):
        assert ef.root_problem(ef.RESULTS/name) is not None
    assert ef.root_problem(ef.DEFAULT_ROOT) is None
    assert ef.DEFAULT_ROOT == Path('/home/rmcfz/seq_vp_tag_results/ef_v1')


@pytest.mark.parametrize('command', ('run-one', 'repair-one'))
def test_a_run_outside_the_root_is_refused(command, tmp_path):
    arm, tier = ('L150', 'long') if command == 'run-one' else ('RF01', 'fine')
    with pytest.raises(SystemExit, match='writes only to'):
        RUN.main([command, '--arm', arm, '--tier', tier, '--pair', core.PAIRS[1], '--out', str(tmp_path/'runs/a/b'), '--device', 'cpu'])


def test_the_entries_refuse_another_root(tmp_path, capsys):
    for command in ('evaluate', 'decide'):
        assert RUN.main([command, '--root', str(tmp_path/'ef_v1')]) == 2
    assert RUN.queue(RUN.argparse.Namespace(root=str(tmp_path/'ef_v1'), commit=None, wait_minutes=1, gap_seconds=0, min_host_gib=15.), FakeSystem()) == 2
    assert 'EF writes only to' in capsys.readouterr().err and not (tmp_path/'ef_v1').exists()


# ---- the firewall -------------------------------------------------------------------------------------------------------------------------------------------
class LabelRead(Exception):
    """A channel path asked for a label member."""


def spy_on_load(monkeypatch):
    """np.load, with every member asked for recorded; a member with seg in its name raises."""
    seen, real = {}, np.load

    class Spy:
        def __init__(self, path, inner):
            self.name, self.inner = Path(path).name, inner

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.inner.close()

        def close(self):
            self.inner.close()

        def __contains__(self, key):
            return key in self.inner

        def __getitem__(self, key):
            seen.setdefault(self.name, set()).add(key)
            if 'seg' in key:
                raise LabelRead(f'{self.name}: {key}')
            return self.inner[key]

        @property
        def files(self):
            if self.name == 'pair.npz':                  # the firewall never lists that archive's members
                raise LabelRead(f'{self.name}: the member list')
            return self.inner.files

    monkeypatch.setattr(np, 'load', lambda path, **kwargs: Spy(path, real(path, **kwargs)))
    return seen


@synthetic_folds('env')
def test_the_fast_channel_path_reads_only_the_images_and_the_predictions(env, monkeypatch):  # noqa: F811
    pair = world_of(env, monkeypatch)
    seen = spy_on_load(monkeypatch)
    pin = ef.read_images(ef.Cell('L150', 'long', pair))
    predicted = fast.read_predicted(pin, workers=2)
    fast.prepare_stacks(ef.ARMS['L150'], pin, predicted, core.load_table(), 'cpu', workers=2)
    assert seen['pair.npz'] == {'fixed', 'moving', 'initial_u'} and seen['probs36.npz'] == {'p'} and seen['structure_groups.npz'] == {'p'}
    assert predicted.record['oracle'] is False and predicted.record['truth_labels_read'] is False and predicted.record['pair_keys_read'] == ['fixed', 'moving', 'initial_u'] and predicted.record['label_firewall'] == 'whitelist-v2'
    assert {s['policy'] for s in predicted.record['sides'].values()} == {'all_folds_mean'} and all(s['alignment_qc']['passed'] for s in predicted.record['sides'].values())
    assert mc.oracle_of_run(dict(oracle=False, channel_sources=predicted.record)) is False


def test_the_counterexample_the_spy_stops_a_reader_of_labels(env, monkeypatch):  # noqa: F811
    """The check above can fail: on the same files, the evaluation's reader trips it, and so does a channel path that asked for the truth."""
    pair = world_of(env, monkeypatch)
    spy_on_load(monkeypatch)
    with pytest.raises(LabelRead, match='fixed_seg4|fixed_seg35|moving_seg'):
        ef.read_truth(pair, RUN.TRUTH_MEMBERS)
    with pytest.raises(LabelRead, match='fixed_seg35'):
        ef.read_truth(pair, ('fixed_seg35',))


@synthetic_folds('env')
def test_changed_truth_labels_change_no_channel(env, monkeypatch):  # noqa: F811
    from .test_d0_a5_tools import relabel_stage
    pair = world_of(env, monkeypatch)
    table = core.load_table()

    def channels():
        pin = ef.read_images(ef.Cell('L150', 'long', pair))
        stacks, _ = fast.prepare_stacks(ef.ARMS['L150'], pin, fast.read_predicted(pin), table, 'cpu')
        return stacks[0].tobytes(), stacks[1].tobytes()
    before = channels()
    with np.load(env['stage']/pair/'pair.npz') as z:
        truth = np.array(z['fixed_seg35'])
    relabel_stage(env)
    with np.load(env['stage']/pair/'pair.npz') as z:
        assert truth.tobytes() != np.array(z['fixed_seg35']).tobytes()
    assert before == channels()


@synthetic_folds('env')
def test_a_missing_atropos_output_stops_the_fast_path_too(env, monkeypatch):  # noqa: F811
    pair = world_of(env, monkeypatch)
    pin = mc.read_images(pair)
    (env['seg_root']/PAIRS[0][0]/'provenance.json').unlink()
    with pytest.raises(SystemExit, match='Atropos'):
        fast.read_predicted(pin)


def test_a_refused_prediction_stops_on_the_fast_path(env, monkeypatch):  # noqa: F811
    pair = world_of(env, monkeypatch)
    pin = mc.read_images(pair)
    (env['unet'].pred_root/PAIRS[0][0]/'probs36.npz').write_bytes(b'not the file')      # its sha256 is no longer the recorded one
    with pytest.raises(SystemExit, match='structure layer refused|sha256|hash'):
        fast.read_predicted(pin)


LABEL_WORDS = ('fixed_seg35', 'moving_seg35', 'fixed_seg4', 'moving_seg4', 'seg35_of', 'read_truth', 'TRUTH_MEMBERS', 'read_pair')


@pytest.mark.parametrize('name', ('fast_channels.py', 'vector_colour.py', 'repair.py'))
def test_the_source_of_the_channel_and_repair_paths_names_no_label_member(name):
    tree = ast.parse((ROOT/'experiments/ef'/name).read_text())
    names = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)} | {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names = {n for n in names if len(n) < 60}                                            # docstrings mention the words of the rule
    assert not names & set(LABEL_WORDS)


def test_in_the_driver_only_the_evaluation_reads_the_truth():
    tree = ast.parse(SCRIPT.read_text())
    readers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            words = {sub.attr for sub in ast.walk(node) if isinstance(sub, ast.Attribute)} | {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}
            if words & {'read_truth', 'TRUTH_MEMBERS'}:
                readers.add(node.name)
    assert readers == {'evaluate_all'}


def test_the_repair_function_takes_no_image():
    import inspect
    assert set(inspect.signature(repair.repair_field).parameters) == {'u', 'grid', 'cells', 'config', 'certify', 'certificate_config', 'device', 'locate', 'clock'}


# ---- the reading ------------------------------------------------------------------------------------------------------------------------------------------------
def fake_row(pair, arm, *, seg35=.82, seg4=.86, alpha=1., seconds=30.):
    metrics = {m: .5 for m in decision.MEASURES}
    metrics.update(seg35=seg35+.01, seg4=seg4+.01, sdlogj_official=.05)
    return dict(run_id=f'ef_{arm}', arm=arm, tier=ef.ARMS[arm]['tier'], pair=pair, status='done', oracle=False, seconds_total=seconds, seconds_labels=3., seconds_channels=8., seconds_levels=[1., 1., 1., 20.], seconds_ladder=1., device_peak_reserved_gib=5., metrics=metrics, accepted_alpha=alpha,
                at_accepted=None if alpha is None else dict(alpha=alpha, seg35=seg35, seg4=seg4), n_infeasible_at_1=0, min_lower_bound=.01)


def e1_rows(arm_kw=None, baseline=None, skip=()):
    rows = {}
    for p in core.PAIRS:
        rows[(ef.BASELINE, p)] = fake_row(p, ef.BASELINE, seconds=60., **(baseline or {}))
        for arm, spec in ef.ARMS.items():
            if spec['decides'] and (arm, p) not in skip:
                rows[(arm, p)] = fake_row(p, arm, **(arm_kw(arm, p) if arm_kw else {}))
    return rows


def slow(arm, pair):
    return dict(seconds=50.)


def test_e1_every_arm_faster():
    result = decision.e1_reading(e1_rows())
    assert result['reading'] == decision.READ_FASTER and set(result['faster_arms']) == {'L150', 'L300', 'F150', 'F300', 'F600', 'K8'}
    assert result['arms']['L150']['seconds_share'] == pytest.approx(.5) and result['arms']['L150']['delta_seg35_at_accepted'] == pytest.approx(0.)


def test_e1_one_arm_is_enough_and_the_others_say_what_missed():
    def kw(arm, pair):
        return dict(seconds=30.) if arm == 'L300' else dict(seconds=50.)
    result = decision.e1_reading(e1_rows(kw))
    assert result['reading'] == decision.READ_FASTER and result['faster_arms'] == ['L300'] and '秒数比' in result['arms']['L150']['missed'][0]


def test_e1_at_the_lines():
    at = decision.e1_reading(e1_rows(lambda arm, pair: dict(seg35=.82-.002, seconds=36.)))
    assert at['arms']['L150']['seconds_share'] == pytest.approx(.6)
    # -.002 and .60 are in; a hair worse is out
    over = decision.e1_reading(e1_rows(lambda arm, pair: dict(seg35=.82-.00201, seconds=30.)))
    assert over['reading'] == decision.READ_NOT_FASTER and 'seg35 差' in over['arms']['L150']['missed'][0]
    over = decision.e1_reading(e1_rows(lambda arm, pair: dict(seconds=36.1)))
    assert over['reading'] == decision.READ_NOT_FASTER and '秒数比' in over['arms']['L150']['missed'][0]


def test_e1_a_pair_not_certified_misses_even_with_more_dice():
    def kw(arm, pair):
        return dict(alpha=None if pair == core.PAIRS[1] else 1., seg35=.9)
    result = decision.e1_reading(e1_rows(kw))
    assert result['reading'] == decision.READ_NOT_FASTER and all('证书 4/5' in r['missed'][0] for r in result['arms'].values())
    assert result['arms']['F150']['n_certified'] == 4 and result['arms']['F150']['alpha'][1] is None


def test_e1_uses_the_accepted_alpha_not_alpha_one():
    def kw(arm, pair):
        return dict(alpha=.53, seg35=.70, seconds=30.)
    result = decision.e1_reading(e1_rows(kw))
    assert result['reading'] == decision.READ_NOT_FASTER and result['arms']['F600']['delta_seg35_at_accepted'] == pytest.approx(-.12)


def test_e1_an_arm_not_read_makes_it_gaps_not_negative():
    rows = e1_rows(lambda arm, pair: dict(seconds=50.), skip=[('K8', core.PAIRS[0])])
    result = decision.e1_reading(rows)
    assert result['reading'] == decision.READ_E1_GAPS and result['arms']['K8']['complete'] is False


def test_e1_a_baseline_not_read_is_undecidable():
    rows = e1_rows()
    del rows[(ef.BASELINE, core.PAIRS[2])]
    assert decision.e1_reading(rows)['reading'] == decision.READ_E1_UNDECIDABLE


@pytest.mark.parametrize('status', ('scale_mismatch', 'time_cap', 'not_predicted_channels'))
def test_e1_a_run_that_is_not_done_is_not_read(status):
    """EF_tools_复核 I1: its field may have good numbers; decide_mc0.valid asks for done, so does this."""
    rows = e1_rows()
    for pair in core.PAIRS:
        rows[('L150', pair)]['status'] = status
    result = decision.e1_reading(rows)
    assert result['arms']['L150']['faster'] is False and result['arms']['L150']['n_read'] == 0 and result['arms']['L150']['complete'] is False and result['arms']['L150']['statuses'] == [status]*5
    assert 'L150' not in result['faster_arms'] and 'L300' in result['faster_arms']


def test_e1_one_pair_not_done_is_not_five_of_five():
    rows = e1_rows(lambda arm, pair: dict(seconds=50.) if arm != 'L150' else {})
    rows[('L150', core.PAIRS[2])]['status'] = 'scale_mismatch'
    result = decision.e1_reading(rows)
    assert result['arms']['L150']['faster'] is False and result['arms']['L150']['n_certified'] == 4 and result['reading'] == decision.READ_E1_GAPS


def test_e1_a_baseline_not_done_is_undecidable():
    rows = e1_rows()
    rows[(ef.BASELINE, core.PAIRS[0])]['status'] = 'time_cap'
    assert decision.e1_reading(rows)['reading'] == decision.READ_E1_UNDECIDABLE


def test_e1_the_same_path_share_is_described_and_decides_nothing():
    """EF_tools_复核 I3.  L300 passes against Q2 on the old path and not against F600 on the fast one: it is marked; the reading is the same with and without the column."""
    seconds = dict(L150=20., L300=33., F150=20., F300=33., F600=40., K8=20.)
    result = decision.e1_reading(e1_rows(lambda arm, pair: dict(seconds=seconds[arm])))
    assert result['reading'] == decision.READ_FASTER and set(result['faster_arms']) == {'L150', 'L300', 'F150', 'F300', 'K8'}
    arms = result['arms']
    assert arms['L300']['seconds_share'] == pytest.approx(.55) and arms['L300']['same_path_share'] == pytest.approx(.825) and arms['L300']['passes_only_with_e2'] is True and arms['L300']['faster'] is True
    assert arms['L150']['same_path_share'] == pytest.approx(.5) and arms['L150']['passes_only_with_e2'] is False
    assert arms['F600']['same_path_share'] == pytest.approx(1.) and arms['F600']['faster'] is False and arms['F600']['passes_only_with_e2'] is False
    assert arms['L300']['per_pair_seconds'] == [33.]*5 and arms['L300']['median_seconds'] == 33.
    assert result['faster_only_with_e2'] == ['L300', 'F300'] and decision.e1_headline(result) == '可以更快（靠 E2 的臂：L300、F300）'


def test_the_first_line_of_the_summary_and_the_queue_records_name_the_arms_that_need_e2(root, monkeypatch):
    """EF_tools_复核 r3: for whoever reads only the first line.  The reading's own word does not change."""
    seconds = dict(L150=20., L300=33., F150=20., F300=33., F600=40., K8=20.)
    rows = e1_rows(lambda arm, pair: dict(seconds=seconds[arm]))
    monkeypatch.setattr(decision, 'collect', lambda path: (rows, {}))
    assert RUN.queue(options(root), FakeSystem()) == 0
    first_line = [line for line in (root/'analysis/summary_ef.md').read_text().splitlines() if line.startswith('**E1')][0]
    assert '**E1：可以更快（靠 E2 的臂：L300、F300）**' in first_line
    assert json.loads((root/'queue_e1_end.json').read_text())['e1'] == decision.READ_FASTER and json.loads((root/'queue_e1_end.json').read_text())['e1_faster_only_with_e2'] == ['L300', 'F300']
    assert json.loads((root/'queue_end.json').read_text())['e1_faster_only_with_e2'] == ['L300', 'F300']
    assert 'E1: 可以更快（靠 E2 的臂：L300、F300）' in (root/'queue.log').read_text()


def repair_rows(arm_kw=None, skip=()):
    rows = {}
    for arm, spec in ef.REPAIR_ARMS.items():
        pairs = core.PAIRS if spec['tier'] == 'fine' else ef.LONG_REPAIR_PAIRS
        for p in pairs:
            if (arm, p) in skip:
                continue
            kw = dict(certified=True, before=.84, after=.84, seconds=10.)
            kw.update(arm_kw(arm, p) if arm_kw else {})
            metrics = {m: .5 for m in decision.MEASURES}
            rows[(arm, p)] = dict(run_id=f'ef_{arm}', arm=arm, tier=spec['tier'], pair=p, eps=spec['eps'], status='certified', reason='certified', seconds_repair=kw['seconds'], rounds=1, n_extra_points=0, metrics=True, certified_at_1=kw['certified'], n_infeasible_after=0, min_lower_bound_after=.001,
                                  before=dict(metrics=dict(metrics, seg35=kw['before'], seg4=.9), n_infeasible_at_1=1000, at_accepted_alpha=dict(alpha=.53, seg35=.7, seg4=.75)), after=dict(metrics, seg35=kw['after'], seg4=.9), change=dict(n_changed_nodes=100, max_change_mm=.5), fallback=None, delta_seg35=kw['after']-kw['before'], delta_seg4=0.)
    return rows


def test_e3_usable():
    result = decision.e3_reading(repair_rows())
    assert result['reading'] == decision.READ_REPAIR and result['usable_arms'] == ['RF01', 'RF05']


def test_e3_one_eps_is_enough():
    result = decision.e3_reading(repair_rows(lambda arm, pair: dict(certified=arm == 'RF05')))
    assert result['reading'] == decision.READ_REPAIR and result['usable_arms'] == ['RF05'] and 'α = 1 证书 0/5' in result['arms']['RF01']['missed'][0]


@pytest.mark.parametrize('kw,word', ((dict(certified=False), '证书'), (dict(after=.8379), '修复尝试后 seg35'), (dict(seconds=15.1), '修复平均')))
def test_e3_not_usable_says_where(kw, word):
    result = decision.e3_reading(repair_rows(lambda arm, pair: kw))
    assert result['reading'] == decision.READ_NO_REPAIR and result['usable_arms'] == [] and word in result['arms']['RF01']['missed'][0]


def test_e3_the_lines_are_in():
    result = decision.e3_reading(repair_rows(lambda arm, pair: dict(after=.838, seconds=15.)))
    assert result['arms']['RF01']['delta_seg35'] == pytest.approx(-.002, abs=1e-12)
    # a mean exactly at -.002 up to rounding may fall either side: the seconds line is exact
    assert '修复平均' not in ' '.join(result['arms']['RF01']['missed'])


def test_e3_a_run_that_disagrees_with_the_evaluation_is_named():
    """The run said it passed, the evaluation's certificate says no: the evaluation counts, and the disagreement is in the record."""
    rows = repair_rows(lambda arm, pair: dict(certified=pair != core.PAIRS[0]))
    for (arm, pair), row in rows.items():
        row['run_says_passed'] = True
    result = decision.e3_reading(rows)
    assert result['arms']['RF01']['n_certified_at_1'] == 4 and result['arms']['RF01']['usable'] is False and result['arms']['RF01']['run_and_evaluation_disagree'] == ['0438-0439']


def test_e3_four_of_five_is_not_five_of_five():
    result = decision.e3_reading(repair_rows(lambda arm, pair: dict(certified=pair != core.PAIRS[3])))
    assert result['reading'] == decision.READ_NO_REPAIR and result['arms']['RF05']['n_certified_at_1'] == 4


def test_e3_a_cell_not_read_is_gaps():
    result = decision.e3_reading(repair_rows(lambda arm, pair: dict(seconds=20.), skip=[('RF01', core.PAIRS[0])]))
    assert result['reading'] == decision.READ_E3_GAPS


def test_the_long_pairs_are_described_and_never_in_the_reading():
    """The lead, 2026-11-04.  A perfect or a failed long tier does not move the reading, either way."""
    good = decision.e3_reading(repair_rows())
    bad_long = decision.e3_reading(repair_rows(lambda arm, pair: dict(certified=False, after=.5, seconds=99.) if arm in ('RL01', 'RL05') else {}))
    assert good['reading'] == bad_long['reading'] == decision.READ_REPAIR
    bad_fine = decision.e3_reading(repair_rows(lambda arm, pair: dict(certified=False) if arm in ('RF01', 'RF05') else {}))
    assert bad_fine['reading'] == decision.READ_NO_REPAIR
    assert sorted(good['arms']) == ['RF01', 'RF05'] and {(r['arm'], r['pair']) for r in good['described_long']} == {(a, p) for a in ('RL01', 'RL05') for p in ('0439-0440', '0455-0456')}
    assert not ef.REPAIR_ARMS['RL01']['decides'] and not ef.REPAIR_ARMS['RL05']['decides']
    row = good['described_long'][0]
    assert row['before']['metrics']['seg35'] == .84 and row['after']['seg35'] == .84 and row['change']['n_changed_nodes'] == 100 and row['change']['max_change_mm'] == .5 and row['before']['metrics']['seg4'] == .9 and row['after']['seg4'] == .9


def test_e3_certified_only_means_are_apart_from_the_attempts():
    """EF_tools_复核 I4.  Three of five pass: the attempts' mean holds two fields no certificate passed, the other mean does not.  The reading is the one the rule gives."""
    def kw(arm, pair):
        return dict(certified=pair in core.PAIRS[:3], after=.90 if pair in core.PAIRS[:3] else .70, before=.84)
    result = decision.e3_reading(repair_rows(kw))
    arm = result['arms']['RF01']
    assert result['reading'] == decision.READ_NO_REPAIR and arm['n_certified_at_1'] == 3 and arm['n_certified_only'] == 3
    assert arm['seg35_attempt'] == pytest.approx((3*.90+2*.70)/5) and arm['seg35_certified_only'] == pytest.approx(.90) and arm['seg35_before_certified_only'] == pytest.approx(.84)
    assert arm['per_pair'][0]['delta_seg35'] == pytest.approx(.06) and arm['per_pair'][4]['delta_seg35'] == pytest.approx(-.14) and arm['per_pair'][4]['certified_at_1'] is False and arm['per_pair'][0]['n_extra_points'] == 0


def test_the_summary_says_what_a_number_is_and_long_csv_marks_the_attempts(tmp_path, monkeypatch):
    rows = e1_rows()
    repairs = repair_rows(lambda arm, pair: dict(certified=pair != core.PAIRS[1]))
    repairs[('RF01', core.PAIRS[0])]['n_extra_points'] = 3
    repairs[('RF01', core.PAIRS[1])]['run_says_passed'] = True
    monkeypatch.setattr(decision, 'collect', lambda root: (rows, repairs))
    monkeypatch.setattr(decision, 'controls', lambda root: dict(fields={}, scale={'ef_L150_long_0438-0439': dict(D_multiplier_equal_Q2=False)}, not_bitwise=[]))
    decision.write_analysis(tmp_path/'ef_v1', stops={})
    text = (tmp_path/'ef_v1/analysis/summary_ef.md').read_text()
    assert decision.e1_headline(dict(reading='可以更快', faster_only_with_e2=['L300', 'F300'])) == '可以更快（靠 E2 的臂：L300、F300）' and decision.e1_headline(dict(reading='不能更快', faster_only_with_e2=[])) == '不能更快'
    for words in ('修复尝试后（含未过证书）', '只算过证书的对', '对 F600 的同路径比值', '过线可能全靠 E2', '逐对秒', '细档逐对', '回退规则', '带选择效应', '同色标签在粗层混得更多', 'W 点数', "['0439-0440']", 'ef_L150_long_0438-0439', 'RF01 0438-0439', '不再只是任务书的 53 点'):
        assert words in text, words
    assert '| 修复后 |' not in text
    with open(tmp_path/'ef_v1/analysis/long.csv') as f:
        lines = list(csv.DictReader(f))
    fields = {(r['arm'], r['pair']): r['field'] for r in lines if r['experiment'] == 'E3' and r['field'] != 'before_alpha1'}
    assert fields[('RF01', core.PAIRS[0])] == 'repaired' and fields[('RF01', core.PAIRS[1])] == 'attempt_uncertified'


def test_the_summary_is_written_with_seg35_and_seg4(tmp_path, monkeypatch):
    rows = e1_rows()
    repairs = repair_rows()
    monkeypatch.setattr(decision, 'collect', lambda root: (rows, repairs))
    monkeypatch.setattr(decision, 'controls', lambda root: dict(fields={'ef_Q2_long_0438-0439': dict(against='mc0p_v1b', control_bitwise=True)}, scale={}, not_bitwise=[]))
    result = decision.write_analysis(tmp_path/'ef_v1', stops={})
    text = (tmp_path/'ef_v1/analysis/summary_ef.md').read_text()
    assert result['oracle'] is False and result['is_conclusion'] is False and decision.READ_FASTER in text and decision.READ_REPAIR in text and 'seg35 / seg4' in text and '只描述' in text and '不是方法' in text
    with open(tmp_path/'ef_v1/analysis/long.csv') as f:
        lines = list(csv.DictReader(f))
    assert {r['oracle'] for r in lines} == {'false'} and {r['experiment'] for r in lines} == {'E1', 'E3'} and {'before_alpha1', 'repaired', 'accepted_alpha', 'uncertified_alpha1'} == {r['field'] for r in lines}
    first = json.loads((tmp_path/'ef_v1/analysis/decision_ef.json').read_text())
    assert first['e1']['reading'] == decision.READ_FASTER
    decision.write_analysis(tmp_path/'ef_v1', stops={})
    assert len(list((tmp_path/'ef_v1/analysis').glob('previous_*/decision_ef.json'))) == 1  # an earlier version moved aside, not overwritten


def test_a_record_that_is_not_oracle_false_writes_no_reading(tmp_path, monkeypatch):
    rows = e1_rows()
    rows[(ef.BASELINE, core.PAIRS[0])]['oracle'] = True
    monkeypatch.setattr(decision, 'collect', lambda root: (rows, {}))
    monkeypatch.setattr(decision, 'controls', lambda root: {})
    with pytest.raises(SystemExit, match='not oracle false'):
        decision.write_analysis(tmp_path/'ef_v1')
    assert not (tmp_path/'ef_v1/analysis').exists()


# ---- the gate, the queue ----------------------------------------------------------------------------------------------------------------------------------------------
class Machine:
    def __init__(self):
        self.load, self.other, self.free_gpu, self.free_home, self.avail, self.free_host = .1, [], 20000., 500., 12., 40.

    def load1(self):
        return self.load

    def units(self):
        return [f'{RUN.QUEUE_UNIT}.service']+list(self.other)

    def gpu_free_mib(self):
        return self.free_gpu

    def home_free_gb(self):
        return self.free_home

    def mem_available_gib(self):
        return self.avail

    def host_free_gib(self):
        return self.free_host

    def now(self):
        return 0.


def test_the_gate_is_the_briefs(tmp_path):
    gate = RUN.GATE
    assert (gate.load_limit, gate.gpu_free_mib, gate.mem_available_gib, gate.host_floor_gib) == (1.5, 15*1024, 7., 15.) and gate.own_units == ('tag-ef-queue.service',) and gate.gpu_hold is None and MC0.HOME_FLOOR_GB == 100
    assert MC0.busy_reason(Machine(), tmp_path, True, gate) is None


@pytest.mark.parametrize('field,value,word', (('load', 1.5, 'load'), ('other', ['tag-other.service'], 'other units'), ('other', ['seq-vp-x.service'], 'other units'), ('free_gpu', 15*1024-1., 'GPU free'), ('free_home', 99., '/home'), ('avail', 6.9, 'MemAvailable'), ('free_host', 14.9, '/mnt/c')))
def test_the_gate_stays_closed(tmp_path, field, value, word):
    machine = Machine()
    setattr(machine, field, value)
    assert word in MC0.busy_reason(machine, tmp_path, True, RUN.GATE)


def test_the_gate_hold_file_and_the_cpu_unit_does_not_ask_for_the_gpu(tmp_path):
    machine = Machine()
    machine.free_gpu = 0.
    assert MC0.busy_reason(machine, tmp_path, False, RUN.GATE) is None
    (tmp_path/'hold').write_text('')
    assert MC0.busy_reason(machine, tmp_path, False, RUN.GATE) == 'hold file'


GOOD_E1 = dict(status='done', oracle=False, scale=dict(images_equal_a5p_inputs=True))
GOOD_E3 = dict(status='certified', passed=True, oracle=False)


class FakeSystem(Machine):
    """The machine: nothing starts; each cell's record is what a behaviour says."""
    def __init__(self, behaviour=None):
        super().__init__()
        self.launched, self.clock, self.behaviour, self.sleeps, self.evaluate = [], 0., behaviour or (lambda cell: (0, None, 10.)), [], None

    def now(self):
        return self.clock

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.clock += seconds

    def code_checkout(self, root, commit):
        return Path(root)/'code', 'fakesha'

    def run(self, argv, log):
        unit = next(a for a in argv if a.startswith('--unit=')).split('=', 1)[1]
        self.launched.append(unit)
        if unit in (RUN.EVAL_UNIT, RUN.EVAL_E3_UNIT):
            return self.evaluate(unit) if self.evaluate else 0
        cell = ef.require_cell(argv[argv.index('--arm')+1], argv[argv.index('--tier')+1], argv[argv.index('--pair')+1])
        rc, record, seconds = self.behaviour(cell)
        if record is None and rc == 0:
            record = dict(GOOD_E3) if cell.is_repair else dict(GOOD_E1)
        self.clock += seconds
        if record is not None:
            MC0.write_json(Path(argv[argv.index('--out')+1])/('repair.json' if cell.is_repair else 'run.json'), record)
        return rc


@pytest.fixture
def root(monkeypatch):
    base = tempfile.mkdtemp(prefix='efq_')              # not the test runner's tmp_path: its name would fail the start-string check
    path = Path(base)/'ef_v1'
    monkeypatch.setattr(ef, 'DEFAULT_ROOT', path)
    monkeypatch.setattr(ef, 'RESULTS', Path('/nonexistent/results'))                     # the reading's controls: no real record is read here
    monkeypatch.setattr(RUN, 'safe_input_problems', lambda *args, **kwargs: [])          # the start checks have their own tests below
    yield path
    shutil.rmtree(base)


def options(path, gap=5):
    return RUN.argparse.Namespace(root=str(path), commit=None, wait_minutes=30, gap_seconds=gap, min_host_gib=15.)


def test_the_queue_runs_e1_then_its_evaluation_then_e3_and_a_restart_reruns_nothing(root):
    system = FakeSystem()
    assert RUN.queue(options(root), system) == 0
    plan = ef.plan_runs()
    e1, e3 = [c.unit for c in plan if not c.is_repair], [c.unit for c in plan if c.is_repair]
    assert system.launched == e1+[RUN.EVAL_UNIT]+e3+[RUN.EVAL_E3_UNIT]
    assert system.sleeps.count(5) == len(e1)-1+len(e3)-1                                 # between the runs of a phase, none before a phase's first
    batch = json.loads((root/'batch.json').read_text())
    assert batch['code_sha'] == 'fakesha' and batch['plan'] == [c.run_id for c in plan] and batch['gap_seconds'] == 5 and batch['gate']['host_floor_gib'] == 15. and batch['phases'] == dict(E1=35, E3=14)
    assert (core.run_dir(root, plan[0])/'cmd.txt').read_text().startswith('systemd-run') and 'py'+'test' not in (core.run_dir(root, plan[0])/'cmd.txt').read_text()
    end = json.loads((root/'queue_end.json').read_text())
    assert end['eval_rc'] == 0 and end['eval_rc_e3'] == 0 and end['stop_e3'] is None and end['e1'] == decision.READ_E1_UNDECIDABLE and end['e3'] == decision.READ_E3_GAPS     # nothing evaluated here: no reading can come from a run's opinion of itself
    assert json.loads((root/'queue_e1_end.json').read_text())['e1'] == decision.READ_E1_UNDECIDABLE
    again = FakeSystem()
    assert RUN.queue(options(root), again) == 0
    assert again.launched == [RUN.EVAL_UNIT, RUN.EVAL_E3_UNIT]


def test_the_gap_can_be_set(root):
    system = FakeSystem()
    assert RUN.queue(options(root, gap=60), system) == 0
    assert set(system.sleeps) == {60}


def test_the_reading_of_e1_is_written_before_any_repair_starts(root):
    """The file is there, with E1's reading, when the first E3 unit starts."""
    seen = []

    def behaviour(cell):
        if cell.is_repair and not seen:
            seen.append(json.loads((root/'analysis/decision_ef.json').read_text())['e1']['reading'])
            seen.append((root/'queue_e1_end.json').is_file())
        return 0, None, 10.
    system = FakeSystem(behaviour)
    assert RUN.queue(options(root), system) == 0
    assert seen == [decision.READ_E1_UNDECIDABLE, True]


def test_three_failures_in_e3_stop_only_e3(root):
    """E1 evaluated and read, E3 stopped after 3, the repairs that ran evaluated, exit 8, the stop on record."""
    system = FakeSystem(lambda cell: (1, None, 5.) if cell.is_repair else (0, None, 5.))
    assert RUN.queue(options(root), system) == 8
    plan = ef.plan_runs()
    e1, e3 = [c.unit for c in plan if not c.is_repair], [c.unit for c in plan if c.is_repair]
    assert system.launched == e1+[RUN.EVAL_UNIT]+e3[:3]+[RUN.EVAL_E3_UNIT]
    assert '3 failures' in json.loads((root/'queue_stop_e3.json').read_text())['reason'] and 'reading of E1 stands' in json.loads((root/'queue_stop.json').read_text())['reason']
    end = json.loads((root/'queue_end.json').read_text())
    assert '3 failures' in end['stop_e3'] and (root/'queue_e1_end.json').is_file() and (root/'analysis/decision_ef.json').is_file() and end['eval_rc'] == 0


def test_a_gate_closed_in_e3_stops_only_e3_and_e3_is_not_evaluated_there(root):
    system = FakeSystem()

    def close_the_gate(unit):
        if unit == RUN.EVAL_UNIT:
            system.load = 3.
        return 0
    system.evaluate = close_the_gate
    assert RUN.queue(options(root), system) == 8
    assert system.launched[-1] == RUN.EVAL_UNIT and RUN.EVAL_E3_UNIT not in system.launched
    end = json.loads((root/'queue_end.json').read_text())
    assert 'gate closed' in end['stop_e3'] and 'not evaluated' in end['end_e3'] and end['eval_rc_e3'] is None and (root/'analysis/decision_ef.json').is_file()


def test_a_stop_in_e1_has_no_evaluation_and_no_e3(root):
    system = FakeSystem(lambda cell: (1, None, 5.))
    assert RUN.queue(options(root), system) == 8
    assert RUN.EVAL_UNIT not in system.launched and not any(u.startswith('tag-ef-r') for u in system.launched) and not (root/'analysis').exists() and not (root/'queue_e1_end.json').exists()


def test_a_failed_evaluation_still_gets_the_readings_and_exit_8(root):
    system = FakeSystem()
    system.evaluate = lambda unit: 1 if unit == RUN.EVAL_UNIT else 0
    assert RUN.queue(options(root), system) == 8
    assert system.launched[-1] == RUN.EVAL_E3_UNIT and 'evaluation unit failed' in json.loads((root/'queue_stop.json').read_text())['reason'] and json.loads((root/'queue_end.json').read_text())['eval_rc'] == 1


def test_an_earlier_stop_record_is_kept_at_a_restart(root):
    root.mkdir(parents=True)
    (root/'queue_stop.json').write_text('{"reason": "before"}')
    (root/'queue_stop_e3.json').write_text('{"reason": "before in E3"}')
    assert RUN.queue(options(root), FakeSystem()) == 0
    kept = {json.loads(path.read_text())['reason'] for path in root.glob('queue_stop*_2*.json')}
    assert kept == {'before', 'before in E3'} and not (root/'queue_stop.json').exists() and not (root/'queue_stop_e3.json').exists()     # no old stop next to stop_e3 null
    assert json.loads((root/'queue_end.json').read_text())['stop_e3'] is None


def test_the_reading_of_e1_stays_in_previous_after_the_second_phase(root):
    """The first reading, moved aside: it has no row of E3; the second has them."""
    assert RUN.queue(options(root), FakeSystem()) == 0
    previous = list((root/'analysis').glob('previous_*/decision_ef.json'))
    assert len(previous) == 1
    first, last = json.loads(previous[0].read_text()), json.loads((root/'analysis/decision_ef.json').read_text())
    assert {row['status'] for row in first['e3']['described_long']} == {'没有运行'} and {row['status'] for row in last['e3']['described_long']} == {'certified'}
    assert (previous[0].parent/'summary_ef.md').is_file() and (previous[0].parent/'long.csv').is_file()


# ---- the start checks ---------------------------------------------------------------------------------------------------------------------------------------
def queue_world(tmp_path, monkeypatch):
    world = dry_world(tmp_path, monkeypatch)
    monkeypatch.setattr(RUN, 'safe_input_problems', REAL_PROBLEMS)
    monkeypatch.setattr(core, 'STAGE', world['stage'])
    monkeypatch.setattr(mc, 'SEG_ROOT', world['seg_root'])
    monkeypatch.setattr(mc, 'A5P_INPUTS', world['a5p'])
    monkeypatch.setattr(mc, 'default_unet', lambda: world['unet'])
    monkeypatch.setattr(ef, 'RESULTS', world['results'])
    return world


def test_with_every_input_there_the_queue_starts(root, tmp_path, monkeypatch):
    queue_world(tmp_path, monkeypatch)
    system = FakeSystem()
    assert RUN.queue(options(root), system) == 0
    assert system.launched[0] == ef.plan_runs()[0].unit and (root/'batch.json').is_file()


@pytest.mark.parametrize('what,word', (('source_field', 'E3 source long 0455-0456'), ('source_status', 'E3 source fine 0442-0443'), ('a5p', "A5P's input"), ('prediction', 'missing'), ('atropos', 'Atropos'), ('manifest', 'manifest'), ('selection', 'selection')))
def test_a_missing_input_stops_before_batch_json_and_before_any_run(root, tmp_path, monkeypatch, what, word):
    """EF_tools_复核 I2: not at run 36."""
    world = queue_world(tmp_path, monkeypatch)
    if what == 'source_field':
        (world['results']/'mc0p_v1c/runs/Q2_long/0455-0456/field.npz').unlink()
    elif what == 'source_status':
        write_json(world['results']/'mc0p_v1b/runs/Q2_fine/0442-0443/run.json', dict(status='cuda_oom', oracle=False, field=dict(sha256='x')))
    elif what == 'a5p':
        (world['a5p']/core.PAIRS[2]/'pair.npz').unlink()
    elif what == 'prediction':
        (tmp_path/'pred'/core.PAIRS[0].split('__')[0]/'probs36.npz').unlink()
    elif what == 'atropos':
        (world['seg_root']/core.PAIRS[1].split('__')[1]/'provenance.json').unlink()
    elif what == 'manifest':
        (world['stage']/core.PAIRS[0]/'manifest.json').unlink()
    else:
        (world['stage']/'selection.json').unlink()
    system = FakeSystem()
    assert RUN.queue(options(root), system) == 8
    assert system.launched == [] and not (root/'batch.json').exists() and not (root/'runs').exists()
    reason = json.loads((root/'queue_stop.json').read_text())['reason']
    assert 'before any run' in reason and word in reason, reason


def test_q0_not_a5p_in_the_baseline_stops_the_queue(root):
    system = FakeSystem(lambda cell: (0, dict(status='input_mismatch', reason='x', oracle=False, scale=dict(images_equal_a5p_inputs=False)), 5.))
    assert RUN.queue(options(root), system) == 8
    assert system.launched == [ef.plan_runs()[0].unit] and 'A5P' in json.loads((root/'queue_stop.json').read_text())['reason'] and not (root/'analysis').exists()


def test_a_record_that_is_not_oracle_false_stops_the_queue(root):
    system = FakeSystem(lambda cell: (0, dict(status='x', oracle=True, reason='y', scale=dict(images_equal_a5p_inputs=True)), 5.))
    assert RUN.queue(options(root), system) == 8
    assert 'oracle True' in json.loads((root/'queue_stop.json').read_text())['reason']


def test_a_device_oom_stops_only_that_arm(root):
    def behaviour(cell):
        return (0, dict(status='cuda_oom', reason='oom', oracle=False, scale=dict(images_equal_a5p_inputs=True)), 10.) if cell.arm == 'L300' else (0, None, 10.)
    system = FakeSystem(behaviour)
    assert RUN.queue(options(root), system) == 0
    assert sum(u.startswith('tag-ef-l300-') for u in system.launched) == 1 and sum(u.startswith('tag-ef-l150-') for u in system.launched) == 5 and sum(u.startswith('tag-ef-rf01-') for u in system.launched) == 5


def test_a_repair_that_fails_stops_nothing(root):
    def behaviour(cell):
        return (0, dict(status='failed_back_to_frozen_rule', passed=False, oracle=False), 10.) if cell.is_repair else (0, None, 10.)
    system = FakeSystem(behaviour)
    assert RUN.queue(options(root), system) == 0
    assert sum(u.startswith('tag-ef-r') for u in system.launched) == 14


def test_failures_are_kept_not_rerun_and_three_in_a_row_stop(root):
    system = FakeSystem(lambda cell: (1, None, 5.))
    assert RUN.queue(options(root), system) == 8
    assert len(system.launched) == 3 and '3 failures' in json.loads((root/'queue_stop.json').read_text())['reason']
    again = FakeSystem()
    assert RUN.queue(options(root), again) == 0
    assert not set(system.launched) & set(again.launched)


def test_a_closed_gate_stops_with_a_record(root):
    system = FakeSystem()
    system.load = 3.
    assert RUN.queue(options(root), system) == 8
    assert system.launched == [] and 'load' in json.loads((root/'queue_stop.json').read_text())['reason'] and system.clock >= 30*60


def test_the_other_queue_keeps_its_gate():
    assert MC0P.GATE.load_limit == 8. and MC0P.RETRY_GATE.host_floor_gib is None and MC0P.RETRY_GATE.load_limit == 1.5


# ---- the start strings -------------------------------------------------------------------------------------------------------------------------------------------------
def test_the_smoke_test_on_the_cpu_and_its_start_string(capsys):
    """The lead runs it with --device cuda before the queue; here the same code on the CPU.  It reads and writes no file."""
    assert RUN.main(['smoke-repair', '--device', 'cpu']) == 0
    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert record['passed'] is True and all(record['checks'].values()) and record['n_infeasible_before'] > 0 and record['device'] == 'cpu' and record['rounds'] and record['energy_relative_difference'] == 0.
    assert record['n_per_row_points'] > 0 and record['checks']['per_row_points_compared'] is True and record['device_warmup']['n_per_row_points'] > 0 and record['seconds_warmup_device'] > 0
    argv = RUN.smoke_launch()
    assert argv[:2] == ['systemd-run', '--user'] and '--unit=tag-ef-smoke' in argv and '--wait' in argv and '--pipe' in argv and argv[-3:] == ['smoke-repair', '--device', 'cuda'] and '--setenv=PYTHONPATH=/home/rmcfz/seq_vp_tag_results/frozen_eb808be:/home/rmcfz/code/python_projects/seq_vp_tag' in argv
    assert 'py'+'test' not in ' '.join(argv) and not any(a.startswith('--setenv=CUDA_VISIBLE_DEVICES') for a in argv)


def test_the_smoke_test_compares_per_row_points_and_a_failed_check_is_exit_8(capsys, monkeypatch):
    """EF_tools_复核 r2, r5.  A repair that does not pass, an objective with no per-row point: each is a failed check and exit 8, whatever else passes.  The solve is stubbed here: the test above runs the real one."""
    monkeypatch.setattr(RUN, 'tiny_warmup', lambda *args: None)
    monkeypatch.setattr(RUN, 'device_warmup', lambda *args: dict(role='stub'))
    real_lattice = repair.lattice_points

    def no_pass(u, grid, cells, config, **kwargs):
        return repair.RepairResult(False, u, 'round_cap', dict(status='infeasible', state_hash='x'), dict(), [], np.empty(0, dtype=np.int64), np.empty((0, 3)))
    monkeypatch.setattr(repair, 'repair_field', no_pass)
    assert RUN.main(['smoke-repair', '--device', 'cpu']) == 8
    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert record['passed'] is False and record['checks']['repair_passed'] is False and record['n_per_row_points'] > 0 and record['checks']['per_row_points_compared'] is True
    monkeypatch.setattr(repair, 'lattice_points', lambda u, grid, cells, eps, *, device: (np.empty(0, dtype=np.int64), np.empty((0, 3))) if eps > 1e8 else real_lattice(u, grid, cells, eps, device=device))
    assert RUN.main(['smoke-repair', '--device', 'cpu']) == 8
    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert record['n_per_row_points'] == 0 and record['checks']['per_row_points_compared'] is False


def test_the_start_strings():
    code = Path('/r/code')
    queue = RUN.queue_launch()
    assert queue[:2] == ['systemd-run', '--user'] and '--unit=tag-ef-queue' in queue and f'--working-directory={core.MAIN_REPO}' in queue and 'experiments/ef/run_ef.py' in queue and '--wait' not in queue
    assert queue[queue.index('--root')+1] == '/home/rmcfz/seq_vp_tag_results/ef_v1' and queue[queue.index('--gap-seconds')+1] == '5' and queue[queue.index('--min-host-gib')+1] == '15'
    e1 = RUN.run_argv(ef.Cell('L150', 'long', core.PAIRS[0]), code, '/r', 'abc')
    assert '--unit=tag-ef-l150-long-0438-0439' in e1 and '--wait' in e1 and '--property=MemoryMax=8G' in e1 and '--nice=5' in e1 and '--setenv=NUMBA_NUM_THREADS=1' in e1 and '--setenv=PYTHONPATH=/home/rmcfz/seq_vp_tag_results/frozen_eb808be:/r/code' in e1
    assert 'run-one' in e1 and '--setenv=CUDA_VISIBLE_DEVICES=' not in e1 and '/r/code/experiments/ef/run_ef.py' in e1
    e3 = RUN.run_argv(ef.Cell('RL01', 'long', ef.EXCEPTION_PAIR), code, '/r', 'abc')
    assert '--unit=tag-ef-rl01-long-0455-0456' in e3 and 'repair-one' in e3 and '--setenv=NUMBA_NUM_THREADS=8' in e3 and '--property=CPUQuota=800%' in e3
    assert '--unit=tag-ef-eval-e3' in RUN.eval_argv(code, '/r', RUN.EVAL_E3_UNIT)
    ev = RUN.eval_argv(code, '/r')
    assert '--unit=tag-ef-eval' in ev and '--setenv=CUDA_VISIBLE_DEVICES=' in ev and 'evaluate' in ev and '--setenv=NUMBA_NUM_THREADS=8' in ev
    for argv in (queue, e1, e3, ev):
        assert 'py'+'test' not in ' '.join(argv) and not any('seq-vp' in a for a in argv)


def test_a_start_string_with_the_word_is_refused():
    with pytest.raises(ValueError, match='forbidden'):
        RUN.run_argv(ef.Cell('L150', 'long', core.PAIRS[0]), Path('/tmp/py'+'test-1/code'), '/r')


def test_nothing_here_stops_or_signals_a_unit():
    for name in ('run_ef.py', 'decide_ef.py', 'ef_core.py', 'fast_channels.py', 'vector_colour.py', 'repair.py'):
        text = (ROOT/'experiments/ef'/name).read_text()
        for word in ('systemctl', 'pkill', 'os.kill', 'SIGTERM', 'mc0p_v1b/hold'):
            assert word not in text, (name, word)


# ---- the dry run --------------------------------------------------------------------------------------------------------------------------------------------------------------
def dry_world(tmp_path, monkeypatch):
    """Files with the real names, synthetic content."""
    stage = tmp_path/'stage'
    a5p = tmp_path/'a5p'
    pairs = core.PAIRS+(ef.EXCEPTION_PAIR,)
    subjects = sorted({s for pair in pairs for s in pair.split('__')})
    for pair in pairs:
        fixed, moving = pair.split('__')
        for parent in (stage, a5p):
            (parent/pair).mkdir(parents=True)
            np.savez(parent/pair/'pair.npz', fixed=np.zeros((6, 7, 8)), moving=np.zeros((6, 7, 8)), fixed_seg35=np.zeros((6, 7, 8), dtype=np.int16))
        write_json(stage/pair/'manifest.json', dict(pair=pair, fixed_subject=fixed, moving_subject=moving, shape_zyx=[6, 7, 8], spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], unit='mm', sources=dict(fixed=dict(sha256='0'*64), moving=dict(sha256='0'*64)), pair_sha256='0'*64))
    write_json(stage/'selection.json', dict(pairs=[dict(pair=p) for p in pairs]))
    for s in subjects:
        d = tmp_path/'pred'/s
        d.mkdir(parents=True)
        np.savez(d/'probs36.npz', p=np.zeros((36, 6, 7, 8), dtype=np.float16))
        np.savez(d/'structure_groups.npz', p=np.zeros((14, 6, 7, 8), dtype=np.float16))
        write_json(d/'provenance.json', dict(segmenter_id='unet35_v1_ens', policy='all_folds_mean', test_labels_seen=False))
        (tmp_path/'atropos'/s).mkdir(parents=True)
        write_json(tmp_path/'atropos'/s/'provenance.json', {})
    results = tmp_path/'results'
    for (tier, pair), name in ef.SOURCES.items():
        d = results/name/'runs'/f'Q2_{tier}'/core.short_name(pair)
        d.mkdir(parents=True)
        np.savez(d/'field.npz', u=np.zeros((6, 7, 8, 3)))
        write_json(d/'run.json', dict(status='done', oracle=False, field=dict(sha256=core.sha_file(d/'field.npz'))))
        write_json(d/'metrics.json', dict(tapered_u_sha256='0'*64, certificate=dict(n_infeasible_at_1=2, accepted_alpha=.9), metrics=dict(seg35=.8, seg4=.85)))
    mi = core.make_inputs()
    return dict(stage=stage, a5p=a5p, seg_root=tmp_path/'atropos', unet=mi.UNetSource(pred_root=tmp_path/'pred'), results=results)


def test_the_dry_run_prints_all_49_and_reads_no_label(tmp_path, monkeypatch):
    world = dry_world(tmp_path, monkeypatch)
    lines = []
    assert RUN.dry_run(out=lines.append, **world) == 0, '\n'.join(lines)
    text = '\n'.join(lines)
    for cell in ef.plan_runs():
        assert cell.run_id in text
    assert 'problems: none' in text and 'selection.json first 5' in text and 'fixed (6, 7, 8)' in text and 'all_folds_mean' in text and 'E3 source long 0455-0456' in text
    assert 'py'+'test' not in text.replace(str(tmp_path), '') and "torch imported by this dry run" in text


@pytest.mark.parametrize('what', ('a5p', 'prediction', 'atropos', 'source', 'source_status', 'policy'))
def test_the_dry_run_names_what_is_missing(tmp_path, monkeypatch, what):
    world = dry_world(tmp_path, monkeypatch)
    if what == 'a5p':
        (world['a5p']/core.PAIRS[2]/'pair.npz').unlink()
    elif what == 'prediction':
        (tmp_path/'pred'/core.PAIRS[0].split('__')[0]/'probs36.npz').unlink()
    elif what == 'atropos':
        (world['seg_root']/core.PAIRS[1].split('__')[1]/'provenance.json').unlink()
    elif what == 'source':
        (world['results']/'mc0p_v1c/runs/Q2_long/0455-0456/field.npz').unlink()
    elif what == 'source_status':
        write_json(world['results']/'mc0p_v1b/runs/Q2_fine/0442-0443/run.json', dict(status='cuda_oom', oracle=False, field=dict(sha256='x')))
    else:
        write_json(tmp_path/'pred'/core.PAIRS[0].split('__')[0]/'provenance.json', dict(segmenter_id='unet35_v1_ens', policy='cross_fit_other_folds', test_labels_seen=False))
    lines = []
    assert RUN.dry_run(out=lines.append, **world) == 8
    assert 'problems: none' not in lines[-1]


# ---- one synthetic run of each kind through everything ----------------------------------------------------------------------------------------------------------------------------------
PIPELINE = """
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1]); sys.path.insert(1, sys.argv[1]+'/../mc0')
import numpy as np
import mc0_core as core
fz = core.frozen_api()
import mc0p_core as mc
import ef_core as ef
import repair
import run_ef as run
world = json.loads(sys.argv[2])
seen, real_load = set(), np.load
class Spy:                                              # the whole of run-one and repair-one under the spy: a member with seg in its name raises, and the run fails
    def __init__(self, path, inner):
        self.name, self.inner = Path(path).name, inner
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        self.inner.close()
    def __contains__(self, key):
        return key in self.inner
    def __getitem__(self, key):
        seen.add(self.name+':'+key)
        if 'seg' in key:
            raise AssertionError('a label member was read: '+key)
        return self.inner[key]
    @property
    def files(self):
        return self.inner.files
root = Path(world['root'])
results = Path(world['results'])
ef.DEFAULT_ROOT, ef.RESULTS = root, results
core.STAGE = Path(world['stage'])
core.SETUP_WORKERS = 2
core.TIERS.update(long=dict(iterations=(2, 2, 2, 4)), fine=dict(iterations=(2, 2, 2, 4), sigma_field=.5))
for arm in ef.ARMS:
    ef.ARMS[arm]['native'] = 4                              # the same steps as Q2's tier above: L150's field must equal Q2's, bit for bit, through the two channel paths
real_config = ef.arm_config
ef.arm_config = lambda arm: dict(real_config(arm), iterations=(2, 2, 2, 4))
mc.SEG_ROOT, mc.A5P_INPUTS, mc.MC0_ROOT = Path(world['seg_root']), Path(world['a5p']), Path(world['mc0_root'])
mi = core.make_inputs()
mc.default_unet = lambda: mi.UNetSource(pred_root=Path(world['pred']))
pair = core.PAIRS[0]
# the sources: a folded field written as an entry run would, with the metrics the evaluation would have written
pin = ef.read_images(ef.Cell('RF01', 'fine', pair))
grid = pin.grid
sources = []
for tier, p, seed in (('fine', pair, 1), ('long', ef.EXCEPTION_PAIR, 2)):
    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter
    raw = np.stack([gaussian_filter(rng.normal(size=grid.shape), 1.2) for _ in range(3)], axis=-1)
    raw *= 4./np.abs(raw).max()
    raw[0], raw[-1], raw[:, 0], raw[:, -1], raw[:, :, 0], raw[:, :, -1] = 0, 0, 0, 0, 0, 0
    d = results/ef.SOURCES[(tier, p)]/'runs'/f'Q2_{tier}'/core.short_name(p)
    d.mkdir(parents=True)
    np.savez(d/'field.npz', u=raw)
    u, _ = fz.boundary_taper(raw, grid, core.TAPER)
    run.run.write_json(d/'run.json', dict(status='done', oracle=False, arm='Q2', tier=tier, run_id='source', code_sha='s', field=dict(sha256=core.sha_file(d/'field.npz'))))
    run.run.write_json(d/'metrics.json', dict(tapered_u_sha256=hashlib.sha256(np.ascontiguousarray(u).tobytes()).hexdigest(), certificate=dict(n_infeasible_at_1=7, accepted_alpha=.81), at_accepted_alpha=dict(alpha=.81, seg35=.1, seg4=.2),
        metrics=dict(seg35=.3, seg4=.4, sdlogj_official=.05, sdlogj_brain=.2, small_six=.1, large_six=.1, other_23=.1, cortex=.1, white_matter=.1)))
repair_calls, real_repair_field = [], repair.repair_field
def spying_repair_field(u, grid, cells, config, **kwargs):
    repair_calls.append(dict(max_inner=config.max_inner, shape=list(u.shape), device=str(kwargs['device'])))
    return real_repair_field(u, grid, cells, config, **kwargs)
repair.repair_field = spying_repair_field
np.load = lambda path, **kwargs: Spy(path, real_load(path, **kwargs))
for arm in ('Q2', 'L150', 'F600', 'K8'):
    tier = ef.ARMS[arm]['tier']
    cell = ef.Cell(arm, tier, pair)
    assert run.main(['run-one', '--arm', arm, '--tier', tier, '--pair', pair, '--out', str(core.run_dir(root, cell)), '--device', 'cpu', '--code-sha', 'x']) == 0
for arm, tier, p in (('RF01', 'fine', pair), ('RL01', 'long', ef.EXCEPTION_PAIR)):
    cell = ef.Cell(arm, tier, p)
    assert run.main(['repair-one', '--arm', arm, '--tier', tier, '--pair', p, '--out', str(core.run_dir(root, cell)), '--device', 'cpu', '--code-sha', 'x']) == 0
real_repair_config = run.repair_config                  # a repair that cannot succeed: the failure's path through the evaluation and the reading
run.repair_config = lambda cell: repair.RepairConfig(eps=ef.REPAIR_ARMS[cell.arm]['eps'], max_rounds=1, max_inner=1, rho0=1e-3)
cell = ef.Cell('RF05', 'fine', pair)
assert run.main(['repair-one', '--arm', 'RF05', '--tier', 'fine', '--pair', pair, '--out', str(core.run_dir(root, cell)), '--device', 'cpu', '--code-sha', 'x']) == 0
run.repair_config = real_repair_config
np.load = real_load
(root/'seen_by_runs.json').write_text(json.dumps(sorted(seen)))
(root/'repair_calls.json').write_text(json.dumps(repair_calls))
assert run.main(['evaluate', '--root', str(root)]) == 0
assert run.main(['decide', '--root', str(root)]) == 0
"""


def write_stage_pair(stage, pair, grid, maps):
    """A pair.npz with the truth different from the predictions, as the real stage has it."""
    fixed, moving = pair.split('__')
    (stage/pair).mkdir(parents=True)
    rng = np.random.default_rng(3)
    truth = {s: np.roll(m, 1, axis=2).astype(np.int16) for s, m in maps.items()}
    arrays = dict(fixed=gaussian_filter(rng.uniform(.1, .7, grid), 1.), fixed_seg4=(truth[fixed] % 5).astype(np.int16), fixed_seg35=truth[fixed], moving=gaussian_filter(rng.uniform(.1, .7, grid), 1.),
                  moving_seg4=(truth[moving] % 5).astype(np.int16), moving_seg35=truth[moving], initial_u=np.zeros(grid+(3,)))
    np.savez_compressed(stage/pair/'pair.npz', **{k: arrays[k] for k in MI.KEY_ORDER})
    write_json(stage/pair/'manifest.json', dict(pair=pair, fixed_subject=fixed, moving_subject=moving, shape_zyx=list(grid), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], unit='mm', sources=dict(fixed=dict(sha256=source_sha(fixed)), moving=dict(sha256=source_sha(moving))), pair_sha256=file_sha256(stage/pair/'pair.npz'), baseline_native_mse=.01, preprocessing='synthetic'))


def pipeline_world(tmp_path, monkeypatch):
    """A stage of the first planned pair and the exception pair on a grid large enough for four levels; predictions of the first pair under the repository's registry (label blocks with few contacts: K = 8 fits), Atropos, A5P's inputs written by make_inputs itself."""
    grid = (26, 28, 30)
    monkeypatch.setattr(a4, 'SHAPE', grid)
    pair = core.PAIRS[0]
    fixed, moving = pair.split('__')
    layers = np.zeros(grid, dtype=np.int64)
    layers[2:-2, 2:-2, 2:-2] = 1
    for k, label in enumerate((1, 20, 2, 21, 7, 11, 13, 14)):
        layers[3+k*2:5+k*2, 2:-2, 2:-2] = label
    maps = {fixed: layers, moving: np.roll(layers, 1, axis=0)}
    stage = tmp_path/'stage_L2R_val_v1'
    write_stage_pair(stage, pair, grid, maps)
    exception_fixed, exception_moving = ef.EXCEPTION_PAIR.split('__')
    write_stage_pair(stage, ef.EXCEPTION_PAIR, grid, {exception_fixed: layers, exception_moving: np.roll(layers, 1, axis=1)})
    write_json(stage/'selection.json', dict(pairs=[dict(pair=pair), dict(pair=ef.EXCEPTION_PAIR)]))
    write_json(stage/'completion.json', dict(done=True))
    models = pair_module.load_segmenter_registry(structure_layer.REPO_REGISTRY).models
    for subject, L in maps.items():
        noise = np.random.default_rng(int(subject.split('_')[2])).uniform(0, 1, (36,)+grid)
        probs = .6*(np.arange(36)[:, None, None, None] == L[None])+.4*noise/noise.sum(axis=0)
        a4.write_prediction(tmp_path/'pred', subject, models, probs=(probs/probs.sum(axis=0)).astype(np.float32))
        wm = np.where(np.isin(L, MI.CEREBRAL_WM), .9, .05)
        csf = .3*(1-wm)
        a4.write_atropos(tmp_path/'atropos', subject, posteriors=np.stack([csf, 1-wm-csf, wm]))
    unet = MI.UNetSource(pred_root=tmp_path/'pred', models_root=tmp_path/'models', image_root=tmp_path/'images')
    only_first = tmp_path/'stage_first'/'stage_L2R_val_v1'
    (only_first/pair).mkdir(parents=True)
    for name in ('pair.npz', 'manifest.json'):
        shutil.copy(stage/pair/name, only_first/pair/name)
    write_json(only_first/'selection.json', dict(pairs=[dict(pair=pair)]))
    write_json(only_first/'completion.json', dict(done=True))
    MI.build_arm('A5P', tmp_path/'inputs', tmp_path/'atropos', stage=only_first, allowlist=core.ALLOWLIST, unet=unet)
    return dict(root=str(tmp_path/'ef_v1'), results=str(tmp_path/'results'), stage=str(stage), seg_root=str(tmp_path/'atropos'), pred=str(tmp_path/'pred'), a5p=str(tmp_path/'inputs/A5P'/MI.STAGE_NAME), mc0_root=str(tmp_path/'no_mc0'))


@needs_frozen
def test_every_kind_of_run_through_everything(tmp_path, monkeypatch):
    world = pipeline_world(tmp_path, monkeypatch)
    env_ = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=f'{FROZEN}:{ROOT}')
    done = subprocess.run([PY, '-B', '-c', PIPELINE, str(ROOT/'experiments/ef'), json.dumps(world)], env=env_, cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:]+done.stderr[-3000:]
    root = Path(world['root'])
    pair = core.PAIRS[0]
    runs = {}
    for arm in ('Q2', 'L150', 'F600', 'K8'):
        cell = ef.Cell(arm, ef.ARMS[arm]['tier'], pair)
        d = core.run_dir(root, cell)
        run, metrics = json.loads((d/'run.json').read_text()), json.loads((d/'metrics.json').read_text())
        runs[arm] = run
        assert run['status'] == 'done' and run['oracle'] is False and metrics['oracle'] is False and metrics['schema'] == 'bayesreg.ef.metrics.v1' and run['field']['sha256'] == core.sha_file(d/'field.npz') and run['run_id'] == cell.run_id
        assert run['seconds_total'] > 0 and run['seconds_channels'] > 0
        first = metrics['certificate']['attempts'][0]
        assert first['alpha'] == 1. and first['config']['tile_size'] == 65536 and first['config']['fallback_workers'] == 8 and set(decision.MEASURES) <= set(metrics['metrics'])
        assert json.loads((d/'ladder_seconds.json').read_text())['seconds'] > 0
    assert runs['L150']['field']['u_sha256'] == runs['Q2']['field']['u_sha256']          # E2 end to end: two channel paths, one field
    assert runs['L150']['scale']['D_multiplier'] == runs['Q2']['scale']['D_multiplier'] and runs['L150']['scale']['g_norm_unscaled'] == runs['Q2']['scale']['g_norm_unscaled']
    assert runs['K8']['k'] == 8 and runs['K8']['optimizer']['channels'] == 8 and runs['K8']['scale_check']['passed'] is True and runs['K8']['scale']['colouring']['K'] == 8
    assert runs['K8']['scale']['g_norm_Q0'] == runs['L150']['scale']['g_norm_Q0'] and runs['F600']['config']['sigma_field'] == .5 and runs['F600']['optimizer']['config']['sigma_field'] == .5
    assert runs['L150']['channel_sources']['sides']['fixed']['read_path'] == 'fast' and runs['Q2']['scale']['images_equal_a5p_inputs'] is True
    for arm, tier, p in (('RF01', 'fine', pair), ('RL01', 'long', ef.EXCEPTION_PAIR)):
        cell = ef.Cell(arm, tier, p)
        d = core.run_dir(root, cell)
        record, metrics = json.loads((d/'repair.json').read_text()), json.loads((d/'repair_metrics.json').read_text())
        assert record['oracle'] is False and record['status'] == 'certified' and record['passed'] is True and record['seconds_repair'] > 0 and record['seconds_warmup'] > 0
        assert record['seconds_warmup'] == record['seconds_warmup_numba']+record['seconds_warmup_device'] and record['seconds_warmup_device'] > 0 and record['device_warmup']['rounds'] == 2 and record['device_warmup']['n_per_row_points'] > 0
        assert record['source']['directory'].endswith(f"{ef.SOURCES[(tier, p)]}/runs/Q2_{tier}/{core.short_name(p)}")
        assert set(record['data']['keys_read']) <= {'fixed', 'moving', 'initial_u'} and record['certificate_config']['max_depth'] == 48 and record['certificate_config']['max_boxes'] == 255 and record['certificate_config']['delta'] == 0.
        after = metrics['after']
        assert after['certificate']['accepted_alpha'] == 1. and after['certificate']['attempts'][0]['status'] == 'certified_feasible'           # the evaluation's own certificate, at alpha = 1
        assert metrics['change']['n_changed_nodes'] > 0 and metrics['change']['max_change_mm'] > 0 and metrics['before']['metrics']['seg35'] == .3 and {'seg35', 'seg4', 'sdlogj_official', 'sdlogj_brain'} <= set(after['metrics']) and metrics['fallback'] is None
    calls = json.loads((root/'repair_calls.json').read_text())                          # each repair-one: first the warm-up on the small field, on the run's device, then the repair
    assert [c['max_inner'] for c in calls] == [2, 300, 2, 300, 2, 1] and {c['device'] for c in calls} == {'cpu'} and [c['shape'] for c in calls[:2]] == [[10, 9, 8, 3], [26, 28, 30, 3]]
    failed = ef.Cell('RF05', 'fine', pair)
    record, metrics = json.loads((core.run_dir(root, failed)/'repair.json').read_text()), json.loads((core.run_dir(root, failed)/'repair_metrics.json').read_text())
    assert record['passed'] is False and record['status'] == 'failed_back_to_frozen_rule' and record['reason'] == 'round_cap' and record['certificate']['status'] != 'certified_feasible'
    assert metrics['after']['certificate']['accepted_alpha'] is None and metrics['after']['certificate']['attempts'][0]['status'] != 'certified_feasible'
    assert metrics['fallback']['accepted_alpha'] == .81 and metrics['fallback']['at_accepted_alpha']['seg35'] == .1 and metrics['change']['n_changed_nodes'] > 0
    result = json.loads((root/'analysis/decision_ef.json').read_text())
    assert result['e1']['reading'] == decision.READ_E1_UNDECIDABLE and result['e3']['reading'] == decision.READ_E3_GAPS and result['oracle'] is False                  # only part of the plan ran
    row = result['e3']['arms']['RF05']['per_pair'][0]
    assert row['certified_at_1'] is False and row['fallback_alpha'] == .81 and row['fallback_at_accepted']['seg35'] == .1 and result['e3']['arms']['RF05']['n_certified_only'] == 0 and result['e3']['arms']['RF05']['seg35_certified_only'] is None
    assert result['e3']['arms']['RF01']['n_certified_only'] == 1 and result['e3']['arms']['RF01']['per_pair'][0]['certified_at_1'] is True
    assert {(r['arm'], r['pair']) for r in result['e3']['described_long'] if r['status'] == 'certified'} == {('RL01', '0455-0456')}
    with open(root/'analysis/long.csv') as f:
        assert {r['oracle'] for r in csv.DictReader(f)} == {'false'}
    assert json.loads((root/'seen_by_runs.json').read_text()) == ['field.npz:u', 'pair.npz:fixed', 'pair.npz:initial_u', 'pair.npz:moving', 'probs36.npz:p', 'structure_groups.npz:p']     # every run of every kind: no truth label


def test_a_run_refuses_to_start_twice(tmp_path, monkeypatch):
    code = ("import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(1, sys.argv[1]+'/../mc0'); import mc0_core as core; core.frozen_api(); import ef_core as ef; from pathlib import Path; root = Path(sys.argv[2]); ef.DEFAULT_ROOT = root; import run_ef as run; d = root/'runs/L150_long/0438-0439'; d.mkdir(parents=True); (d/'started.json').write_text('{}'); "
            "sys.exit(run.main(['run-one', '--arm', 'L150', '--tier', 'long', '--pair', core.PAIRS[0], '--out', str(d), '--device', 'cpu']))")
    env_ = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=f'{FROZEN}:{ROOT}')
    done = subprocess.run([PY, '-B', '-c', code, str(ROOT/'experiments/ef'), str(tmp_path/'ef_v1')], env=env_, cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode != 0 and 'never rerun' in done.stderr
