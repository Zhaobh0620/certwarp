"""T30 independent exact arithmetic checks of the whole-cell preproof."""
from dataclasses import asdict
from itertools import product
import numpy as np
import pytest
from jd_curl.three_d import feasibility as f
from jd_curl.three_d.grid import make_grid_3d
from jd_curl.validation import InputValidationError
from .test_interval_certificate import VERTICES, exact_jacobian, contains, counterexample


@pytest.mark.parametrize('scale',[2.**-40,1.,2.**40])
@pytest.mark.parametrize('seed',[2,61,99])
def test_enclosure_contains_fraction_at_independent_points(scale,seed):
    rng=np.random.default_rng(seed)
    u=(rng.normal(scale=.17,size=(8,3))+[17.,-31.,9.])*scale
    spacing=np.array([.2,.3,.7])*scale
    box=f.jacobian_enclosure(u[None],spacing)[0]
    for point in list(product((0.,1.),repeat=3))+list(rng.uniform(size=(20,3))):
        contains(box,exact_jacobian(u,spacing,point))


@pytest.mark.parametrize('delta',[0.,.01])
@pytest.mark.parametrize('jd',[0.,-.1,2.**-30,.005,.5,1.])
def test_local_strict_floor_no_clipping(delta,jd):
    u=VERTICES*np.array([jd-1,0.,0.])
    out=f.certify_cell(u,(1,1,1),config=f.EnclosureCertificateConfig(delta=delta))
    if jd>delta:
        assert out.status=='certified_feasible'
        assert delta<out.min_jd_lower_bound<=jd
    else:
        assert out.status!='certified_feasible'


@pytest.mark.parametrize('depth,boxes',[(0,255),(48,1),(48,255)])
def test_positive_hull_failure_falls_back(depth,boxes):
    u=counterexample(.1)
    cfg=f.EnclosureCertificateConfig(max_depth=depth,max_boxes=boxes)
    old=f.certify_cell(u,(1,1,1),config=f.AdaptiveCertificateConfig(max_depth=depth,max_boxes=boxes))
    assert f.jacobian_enclosure(u[None],(1,1,1)).lo[0]<0
    assert f.certify_cell(u,(1,1,1),config=cfg)==old


def test_all_27_positive_but_interior_negative_not_accepted():
    x,y,z=VERTICES.T
    u=np.stack((x,y+2*(x-.25)*z,-2*(x-.25)*y-.0625*z),axis=-1)-VERTICES
    assert min(exact_jacobian(u,(1,1,1),p) for p in product((0.,.5,1.),repeat=3))>0
    out=f.certify_cell(u,(1,1,1),config=f.EnclosureCertificateConfig(delta=0))
    assert out.status=='infeasible'
    assert exact_jacobian(u,(1,1,1),out.witness_reference_point)<=0


def grid():
    return make_grid_3d(nx=6,ny=5,nz=7,hx=.2,hy=.3,hz=.7)


@pytest.mark.parametrize('tile',[1,7,8192])
def test_map_cover_counts_identity_and_mixed_witness(tile):
    g=grid();u=np.zeros(g.shape+(3,));cfg=f.EnclosureCertificateConfig(delta=0,tile_size=tile)
    out=f.certify_map(u,g,config=cfg)
    assert out.status=='certified_feasible'
    assert out.n_fast_cells==out.n_cells_checked==out.n_boxes==g.n_cells
    assert out.n_fallback_cells==0 and out.sampled_min_jd is None
    u[1,2,3,0]=-.35  # Admissible sampling domain, locally inverted.
    out=f.certify_map(u,g,config=cfg)
    old=f.certify_map(u,g,config=f.AdaptiveCertificateConfig(delta=0,tile_size=tile))
    assert out.status==old.status=='infeasible'
    assert out.witness_cell==old.witness_cell
    assert out.witness_reference_point==old.witness_reference_point
    assert 0<out.n_fast_cells<g.n_cells
    assert out.n_fast_cells+out.n_fallback_cells==out.n_cells_checked==g.n_cells
    assert out.sampled_min_jd is None


@pytest.mark.parametrize('bad,reason', [('nan','NONFINITE_STATE'),('boundary','NONZERO_BOUNDARY'),('domain','OUTSIDE_DOMAIN')])
def test_global_guards(bad,reason):
    g=grid();u=np.zeros(g.shape+(3,))
    if bad=='nan':u[1,1,1,0]=np.nan
    elif bad=='boundary':u[0,0,0,0]=.1
    else:u[1,1,1,0]=100.
    old=f.certify_map(u,g);out=f.certify_map(u,g,config=f.EnclosureCertificateConfig(delta=0))
    assert out.status==old.status and out.reason==old.reason
    assert out.n_fast_cells==out.n_fallback_cells==0


def test_arithmetic_failure_falls_back_and_unsupported_fails_closed(monkeypatch):
    g=grid();u=np.zeros(g.shape+(3,))
    def fail(*a,**k):raise FloatingPointError('simulated prepass overflow')
    monkeypatch.setattr(f,'jacobian_enclosure',fail)
    out=f.certify_map(u,g,config=f.EnclosureCertificateConfig())
    assert out.status=='certified_feasible' and out.n_fallback_cells==g.n_cells
    assert out.n_fast_cells==0 and out.sampled_min_jd==1.
    monkeypatch.setattr(f.iv,'arithmetic_supported',lambda:False)
    assert f.certify_map(u,g,config=f.EnclosureCertificateConfig()).status=='uncertain'


def test_legacy_schema_and_observer_preserved():
    g=grid();old=f.certify_map(np.zeros(g.shape+(3,)),g)
    assert type(old) is f.FeasibilityResult and 'n_fast_cells' not in asdict(old)
    observed=[]
    out=f.certify_cell(np.zeros((8,3)),g.spacing,config=f.EnclosureCertificateConfig(),point_observer=lambda *args:observed.append(args))
    assert len(observed)==8 and out.sampled_min_jd==1.
    with pytest.raises(InputValidationError):f.EnclosureCertificateConfig(prepass='sampling')
    with pytest.raises(InputValidationError):f.EnclosureCertificateConfig(max_depth=121)


def test_public_geometry_and_transpose_keep_full_input_guards():
    from jd_curl.three_d.geometry import compute_cell_geometry,_compute_cell_geometry
    from jd_curl.three_d.cell_q1 import CellQ1Operators3D,cell_tiles
    from jd_curl.three_d.linearization import _finish
    g=grid();ops=CellQ1Operators3D(g);u=np.zeros(g.shape+(3,));u[-1,-1,-1,0]=np.nan
    with pytest.raises(InputValidationError):compute_cell_geometry(u,ops,tile=slice(0,1))
    with pytest.raises(InputValidationError):
        for tile in cell_tiles(g,7):_compute_cell_geometry(u,ops,tile=tile)
    out=np.zeros(g.shape);out[-1,-1,-1]=np.nan
    with pytest.raises(FloatingPointError):
        ops.derivative_transpose(np.zeros((1,27)),'x',tile=slice(0,1),accumulate_into=out)
    out=np.zeros((3,)+g.shape);out[0,0,0,0]=np.nan
    with pytest.raises(FloatingPointError):_finish(out)
