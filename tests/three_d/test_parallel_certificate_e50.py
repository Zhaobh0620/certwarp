"""E50/S2: the parallel whole-map certificate equals the E33 hoisted certificate."""
from dataclasses import asdict, replace

import numpy as np
import pytest

from jd_curl.three_d.feasibility import (CompiledEnclosureCertificateConfig, ParallelEnclosureCertificateConfig,
                                         certify_map, certificate_version)
from jd_curl.three_d.grid import make_grid_3d

IDENTITY = ('config', 'certificate_version')


def configs(tile, workers=3, depth=48):
    hoisted = CompiledEnclosureCertificateConfig(delta=0., tile_size=tile, max_depth=depth, kernel='numba-hoisted-edge-hull-v1')
    parallel = ParallelEnclosureCertificateConfig(delta=0., tile_size=tile, max_depth=depth, fallback_workers=workers)
    return hoisted, parallel


def same(a, b):
    x, y = asdict(a), asdict(b)
    for key in IDENTITY:
        x.pop(key); y.pop(key)
    assert x == y
    # Exact float endpoints, not approximate equality.
    for key in ('min_jd_lower_bound', 'min_margin_lower_bound'):
        if x[key] is not None:
            assert np.float64(x[key]).tobytes() == np.float64(y[key]).tobytes()


def grid(nx=9, ny=8, nz=7, hx=.7, hy=1.1, hz=.9):
    return make_grid_3d(nx=nx, ny=ny, nz=nz, hx=hx, hy=hy, hz=hz, x0=-1., y0=.5, z0=2., unit='mm')


def field(g, amplitude, seed):
    rng = np.random.default_rng(seed)
    u = np.zeros(g.shape+(3,))
    h = np.array(g.spacing)
    u[2:-2, 2:-2, 2:-2] = amplitude*h*rng.uniform(-1, 1, u[2:-2, 2:-2, 2:-2].shape)
    return u


@pytest.mark.parametrize('amplitude', [0., .05, .25, .38, .45, .6])
@pytest.mark.parametrize('tile', [1, 7, 64, 100000])
@pytest.mark.parametrize('seed', [0, 1])
def test_parallel_certificate_equals_hoisted(amplitude, tile, seed):
    g = grid(); u = field(g, amplitude, seed)
    hoisted, parallel = configs(tile)
    a, b = certify_map(u, g, config=hoisted), certify_map(u, g, config=parallel)
    same(a, b)
    assert b.certificate_version == 'q1-3d-parallel-hoisted-edge-hull-bernstein-v1' == certificate_version(parallel)
    assert b.config == parallel


def test_statuses_are_exercised():
    g = grid(); seen = set(); fallback = 0
    for amplitude in (0., .25, .38, .45, .6):
        for seed in range(4):
            r = certify_map(field(g, amplitude, seed), g, config=configs(64)[1])
            seen.add(r.status); fallback += r.n_fallback_cells
    assert {'certified_feasible', 'infeasible'} <= seen and fallback > 0


def test_depth_budget_uncertain_matches():
    g = grid(); u = field(g, .45, 3)
    for depth in (0, 1, 2):
        hoisted, parallel = configs(16, depth=depth)
        same(certify_map(u, g, config=hoisted), certify_map(u, g, config=parallel))


@pytest.mark.parametrize('workers', [1, 2, 8])
def test_thread_count_cannot_change_the_certificate(workers):
    g = grid(); u = field(g, .38, 5)
    reference = certify_map(u, g, config=configs(32, workers=1)[1])
    other = certify_map(u, g, config=configs(32, workers=workers)[1])
    same(reference, other)


def test_invalid_step_disables_the_whole_tile_like_legacy():
    """In-domain overflow (review finding: the old version was stopped by the domain check).

    Extreme anisotropy makes finite, in-domain displacements overflow the interval
    determinant; the legacy kernel raises for that whole tile, which then uses Bernstein.
    """
    from jd_curl.three_d.compiled_enclosure import parallel_margins
    g = make_grid_3d(nx=6, ny=7, nz=8, hx=1e-160, hy=1., hz=1e154, x0=0., y0=0., z0=0., unit="mm")
    u = np.zeros(g.shape+(3,))
    u[4, 3, 2, 1] = .3          # y displacement, far inside the y extent
    u[4, 3, 2, 2] = .3e154      # z displacement, far inside the z extent
    margin, flag = parallel_margins(u, g, 0.)
    assert flag.any()
    for tile in (1, 7, 13, 64):
        hoisted, parallel = configs(tile)
        a, b = certify_map(u, g, config=hoisted), certify_map(u, g, config=parallel)
        assert a.n_cells_checked == g.n_cells     # the domain check passed: the kernel path ran
        same(a, b)


def test_nonfinite_and_domain_short_circuits_match():
    g = grid(); u = field(g, .05, 4)
    bad = u.copy(); bad[3, 3, 3, 0] = np.nan
    far = u.copy(); far[3, 3, 3, 0] = 1e3
    for case in (bad, far):
        hoisted, parallel = configs(16)
        same(certify_map(case, g, config=hoisted), certify_map(case, g, config=parallel))


def test_parallel_config_validation():
    with pytest.raises(Exception):
        ParallelEnclosureCertificateConfig(delta=0., kernel='numba-hoisted-edge-hull-v1')
    with pytest.raises(Exception):
        ParallelEnclosureCertificateConfig(delta=0., fallback_workers=0)
    # Legacy classes keep their fields (problem identities unchanged).
    assert 'fallback_workers' not in asdict(CompiledEnclosureCertificateConfig(delta=0.))


def legacy_tile_margins(u, g, tile):
    """Per-cell margins of the E33 tiled path, or None where a tile raised."""
    from jd_curl.three_d.cell_q1 import cell_node_indices, cell_tiles
    from jd_curl.three_d.feasibility import compiled_jacobian_enclosure
    out = np.full(g.n_cells, np.nan); raised = np.zeros(g.n_cells, dtype=bool)
    for t in cell_tiles(g, tile):
        vertices = u.reshape(-1, 3)[cell_node_indices(g, t)]
        try:
            out[t] = (compiled_jacobian_enclosure(vertices, g.spacing, kernel='numba-hoisted-edge-hull-v1')-0.).lo
        except FloatingPointError:
            raised[t] = True
    return out, raised


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_kernel_margins_are_bitwise_the_hoisted_endpoints(seed):
    from jd_curl.three_d.compiled_enclosure import parallel_margins
    g = grid(); u = field(g, .5, seed)
    margin, flag = parallel_margins(u, g, 0.)
    legacy, raised = legacy_tile_margins(u, g, 7)
    assert not flag.any() and not raised.any()
    assert margin.tobytes() == legacy.tobytes()


def test_kernel_flags_exactly_the_cells_whose_legacy_arithmetic_fails():
    from jd_curl.three_d.compiled_enclosure import parallel_margins
    g = grid(); u = field(g, .05, 1)
    u[3, 3, 3] = (1e154, -1e154, 1e154)   # overflow in interval products (domain bypassed)
    margin, flag = parallel_margins(u, g, 0.)
    legacy1, raised1 = legacy_tile_margins(u, g, 1)   # one cell per tile: per-cell legacy verdict
    assert np.array_equal(flag, raised1) and flag.any()
    ok = ~flag
    assert margin[ok].tobytes() == legacy1[ok].tobytes()
