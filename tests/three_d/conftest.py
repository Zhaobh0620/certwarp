import pytest

from jd_curl.three_d.grid import make_grid_3d


@pytest.fixture
def grid3d():
    return make_grid_3d(nx=7, ny=6, nz=5, hx=.2, hy=.3, hz=.4,
                        x0=-.3, y0=.2, z0=-.7, unit="mm")
