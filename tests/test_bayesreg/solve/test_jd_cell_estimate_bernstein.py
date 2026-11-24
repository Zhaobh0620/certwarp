"""C09.1-3a jd_cell_estimate "certificate_bernstein" (work package JDTI, 2026-10-31; contract draft JDT, D53; review
docs/bayesreg/reviews/JDT_复核_20261031.md).  The contract's gates 1-13 (layer 0, synthetic data; single synthetic
observations, not convergence statements), one block per gate:

 1  the default "certificate" and the options "closed27", "gauss27+corners" bitwise as before JDTI: golden digests of
    short run_pg / outer solves computed on main 96c8d35 (GOLDEN_96C8D35; the run_pair goldens of test_def_defaults.py
    and the pre-option goldens of test_jd_cell_estimate.py are the other existing gold standards);
 2  synthetic fields (certified_fields, path_fields, "mid", "off"): r'' >= r' (certificate), r'' <= r (closed27),
    0 < b~ <= J53 per cell (raised cells without tolerance), l on S == prescreen on all cells bitwise, l == b on
    prescreen cells, l not positive on fallback cells, r restricted to S == r of max(b, l) on every cell bitwise; some
    field has raised cells, nodes raised against certificate and a cell with b~ >= 2 b;
 3  the "mid" / "off" edge-dip cells: proven by the fallback, b~ = b bitwise, their 8 vertices r'' < .01;
 4  check (b) by injection (monkeypatched prescreen): no exception, the cell falls back to b, one violation counted and
    warned, the run total equals the injections (solve_direct and a whole run_pair run); the four check-(a) violations
    raise CertificateContractError;
 5  cache: hit on the second call, key (certificate key, name, J_act.hex()), check (b) on a hit, alternating with
    certificate gives the separate results;
 6  synthetic problems (F_B edge dip, quadratic "mid" / "off"): steps and ending recorded (expected as certificate);
 7  torch64 (cpu) step sequence = numpy, the new record fields equal; cuda only with JDC_TEST_CUDA=1;
 8  a locally affine compressed field with S cells l <= b: those keep b bitwise, n_cells_raised < |S|, r'' >= r',
    restricted == full max, no violation;
 9  CLI: run_pair refuses the value with the metric off, records it otherwise; the learn CLI accepts and records it;
10  J_act = .5: S changes, another cache key, restricted == full max, bernstein_activation recorded;
11  coefficient table FloatingPointError, subnormal table, extreme spacing: l all NaN, b~ = b, r == certificate's;
12  record fields consistent (n_cells_hull == PROVEN_HULL count == certify_map n_fast_cells, ...);
13  FMA regression of the jd_curl prescreen kernel (inspect_asm / inspect_llvm; also guards the C07-2 prescreen path).
No data/, no GPU (cuda case skipped unless JDC_TEST_CUDA=1).
"""
from dataclasses import replace
import functools
import json
import os
from pathlib import Path
import re

import numpy as np
import pytest

from jd_curl.validation import InputValidationError
from jd_curl.three_d.boundary import restrict_free
from jd_curl.three_d.cell_q1 import VERTICES, CellQ1Operators3D
from jd_curl.three_d import certificate_prescreen as cps
from jd_curl.three_d.certificate_prescreen import PRESCREEN_KERNEL, prescreen_lower_bounds
from jd_curl.three_d.feasibility import certify_map
from jd_curl.three_d.grid import make_grid_3d

from bayesreg.cli import run_pair
from bayesreg.eval import cert_eval
from bayesreg.eval.cert_eval import PROVEN_FALLBACK, PROVEN_HULL, PROVEN_PRESCREEN, cell_pass
from bayesreg.solve import (Caps, CertificateContractError, Commit, JDScaledMetric, JDScalingConfig, NumpyBackend,
                            PGConfig, g0_of, solve_direct, threshold)
from bayesreg.solve import jd_scaling as jds

from . import _jdti_digest as JD
from . import _run_pair_digest as RD
from . import test_jd_cell_estimate as T
from .test_jd_cell_estimate import (C27, CERT, G27, TORCH_CASES, Identity, certified_fields, compare_with_numpy,
                                    dip_problem, fb_dip_problem, run_outer, run_quadratic)
from .test_jd_scaling import aniso_grid, wavy_field

BERN = jds.CERTIFICATE_BERNSTEIN
ROOT = Path(__file__).resolve().parents[3]
GOLDEN_RECORD = ROOT/"tests"/"test_bayesreg"/"_golden"/"golden_96c8d35_jdti.json"
# Source: tests/test_bayesreg/_golden/golden_96c8d35_jdti.json (make_golden_jdti.py --ref 96c8d35, numpy 2.0.2)
GOLDEN_96C8D35 = {
    "outer_a_certificate": "95716aec8e17e586dbd03d0dcd54d5337a180b041309e2ee3ba8c9e3dd1fba98",
    "outer_b_certificate": "19e32fe0215b0a31950859e6d2297ec717f3cfb367eb77084ab67820c4c23e5f",
    "quadratic_mid_certificate": "5ff4fac4b01f23a6a0684c038695773fe5a845d60189c5778c1907452aa60d33",
    "quadratic_mid_closed27": "b265f54415f024a1f61b95cae3f0dccfb257f2fb9d41018f47065994a4af7285",
    "quadratic_mid_gauss27+corners": "96491ff7a02cab1ea4283a8ee14cc082d7bd403883884bda6b26f1f55d47b72a",
    "quadratic_off_certificate": "026db4257b5cb1fdd2913f6b26de4e7fe40fcc528db9b6709a829d0dfec8f495",
    "quadratic_off_closed27": "7ebe0119e05fd4e527e1efa31103be871b134f7e6d52a5b93824cf0669b6d937",
    "quadratic_off_gauss27+corners": "857f7ffbf93176847fcfa2bba00f1a66ae61c0b39ab273aed85d5bcb92dc5f7a",
}
BITWISE = pytest.mark.skipif(np.__version__ != "2.0.2", reason="bitwise golden digests recorded with numpy 2.0.2")
NEW_INT_KEYS = ("n_cells_hull", "n_cells_bernstein", "n_cells_raised", "n_bernstein_j53_violations",
                "n_r_lt_1_certificate", "n_nodes_r_lowered_certificate", "n_nodes_r_raised_vs_certificate",
                "n_r_lt_0p01", "n_r_lt_0p01_certificate", "n_r_lt_0p01_default")
RECORD_KEYS = {"cell_estimate", "n_cells_lowered", "n_cells_without_bound", "min_cert_bound", "cert_state_hash",
               "cert_min_matches", "cell_bounds_cached", "cell_bounds_seconds", "n_cells_hull", "n_cells_bernstein",
               "n_cells_raised", "min_bernstein_bound", "bernstein_cached", "bernstein_seconds",
               "bernstein_activation", "prescreen_kernel", "bernstein_j53_min_margin", "n_bernstein_j53_violations",
               "bernstein_j53_max_excess", "bernstein_j53_violation_cells", "n_nodes_r_lowered", "n_r_lt_1_default",
               "n_r_lt_1_certificate", "n_nodes_r_lowered_certificate", "n_nodes_r_raised_vs_certificate",
               "n_r_lt_0p01", "n_r_lt_0p01_certificate", "n_r_lt_0p01_default", "min_J_a", "fraction_r_lt_1",
               "n_r_lt_1", "n_nodes", "min_r", "seconds"}


@pytest.fixture(autouse=True)
def _empty_cache():
    jds.clear_cell_bounds_cache()
    yield
    jds.clear_cell_bounds_cache()


# ---- helpers ----------------------------------------------------------------------------------------------

def factors(m, grid, activation=jds.J_ACT):
    m = np.asarray(m).reshape(grid.cell_shape)
    return np.minimum(1., np.maximum(jds.node_minimum(m, grid), jds.J_FLOOR)/activation).ravel()


def reference(u, grid, config, activation=jds.J_ACT):
    """The contract's quantities computed independently of the implementation (as the JDT prototype
    experiments/jdt/synthetic_check.py): cell_pass, the prescreen on EVERY cell, S, b~ restricted to S, b~ of max(b, l)
    on every cell, J53, m_K, and the node factors of gauss27+corners, certificate, certificate_bernstein (restricted and
    full) and the 53-point minimum."""
    code, b = cell_pass(u, grid, config)
    table = CellQ1Operators3D(grid, location="closed27_uniform").coefficients
    ell = prescreen_lower_bounds(u, grid, table, np.arange(grid.n_cells, dtype=np.int64))
    S = np.flatnonzero((code == PROVEN_HULL) & (b < activation))
    up = np.isfinite(ell[S]) & (ell[S] > b[S])
    bt = b.copy()
    bt[S[up]] = ell[S][up]
    bt_full = np.where(np.isfinite(ell) & (ell > b), ell, b)
    m53, m_def = jds.closed27_cell_estimate(u, grid)
    J53, m_def = m53.ravel(), m_def.ravel()

    def mm(bb):
        return np.where(np.isfinite(bb) & (bb < m_def), bb, m_def)

    out = dict(code=code, b=b, ell=ell, S=S, raised=S[up], bt=bt, bt_full=bt_full, J53=J53, m_def=m_def)
    out.update({f"r_{k}": factors(v, grid, activation) for k, v in (("def", m_def), ("cert", mm(b)), ("new", mm(bt)),
                                                                     ("full", mm(bt_full)), ("53", J53))})
    return out


def vertices_of(cell):
    return [(cell[0]+z-1, cell[1]+y-1, cell[2]+x-1) for x, y, z in VERTICES]     # interior-node indices


def affine_field(seed=0, spacing=(1., .7, 1.3), s=.4, block=5, n=11):
    """Review JDT appendix item 4: zero faces, u = A (x - c) on an inner block^3 node block, A = -s Q Lambda Q^T (Q a
    random rotation, Lambda = diag(1 + .05 N(0, 1))); spacings with non-dyadic values.  (grid, u, det(I + A))."""
    rng = np.random.default_rng(seed)
    grid = make_grid_3d(nx=n, ny=n, nz=n, hx=spacing[0], hy=spacing[1], hz=spacing[2], unit="mm")
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    A = -s*Q@np.diag(1.+.05*rng.standard_normal(3))@Q.T
    x = grid.identity()
    lo = (n-block)//2
    sl = slice(lo, lo+block)
    u = np.zeros(grid.shape+(3,))
    u[sl, sl, sl] = (x[sl, sl, sl]-x[(n//2,)*3])@A.T
    return grid, u, float(np.linalg.det(np.eye(3)+A))


def wavy12():
    grid = aniso_grid()
    return grid, wavy_field(grid, amplitude=1.2, seed=5)


def certified(grid, u, backend=None):
    backend = NumpyBackend() if backend is None else backend
    cert = backend.certify(u, grid)
    assert cert.passed
    return cert, backend.cert_config


def inject_first(monkeypatch, value=1e3):
    """Monkeypatch jd_scaling's prescreen so that the first cell of every non-empty call returns `value` (above J at
    its 53 points; > b_K since b_K < J_act <= 1): one check-(b) violation per call.  Returns the list of the
    injected cell ids (one per call)."""
    calls = []
    original = jds.prescreen_lower_bounds

    def patched(u, grid, coefficients, cells, **kw):
        out = np.array(original(u, grid, coefficients, cells, **kw))
        if len(out):
            out[0] = value
            calls.append(int(cells[0]))
        return out
    monkeypatch.setattr(jds, "prescreen_lower_bounds", patched)
    return calls


# ---- gate 1: the existing options bitwise ------------------------------------------------------------------------

@BITWISE
def test_gate1_existing_options_bitwise_the_code_before_jdti():
    assert JD.golden_cases(T, jds, PGConfig, JDScalingConfig) == GOLDEN_96C8D35


def test_gate1_golden_constants_match_the_generation_record():
    record = json.loads(GOLDEN_RECORD.read_text(encoding="utf-8"))
    assert record["ref_commit"].startswith("96c8d35") and record["digests"] == GOLDEN_96C8D35
    assert record["environment"]["numpy"] == "2.0.2"
    assert record["environment"]["thread_env"]["OPENBLAS_NUM_THREADS"] == "1"
    import hashlib
    assert hashlib.sha256(Path(JD.__file__).read_bytes()).hexdigest() == record["digest_module_sha256"]


def test_gate1_default_unchanged_and_option_listed():
    assert jds.DEFAULT_CELL_ESTIMATE == CERT and JDScalingConfig().cell_estimate == CERT
    assert PGConfig().jd_cell_estimate == CERT
    assert jds.CELL_ESTIMATES == (G27, C27, CERT, BERN)
    record = JDScalingConfig(cell_estimate=BERN).record()
    assert record["cell_estimate"] == BERN and record["estimate"] == jds.ESTIMATES[BERN]
    assert len(set(jds.ESTIMATES.values())) == 4


# ---- gate 2: synthetic fields, pointwise -------------------------------------------------------------------------

def test_gate2_synthetic_fields_pointwise():
    backend = NumpyBackend()
    cfg = backend.cert_config
    seen = 0
    raised_fields, nodes_raised_fields, doubled_fields = [], [], []
    for name, grid, u in certified_fields():
        cert = backend.certify(u, grid)
        if not cert.passed:
            continue
        seen += 1
        jds.clear_cell_bounds_cache()
        r, rec = jds.scaling_factors(u, grid, estimate=BERN, cert=cert, cert_config=cfg)
        S_impl, ell_impl = jds._BERNSTEIN_CACHE["S"], jds._BERNSTEIN_CACHE["ell"]
        bt, b, _ = jds.bernstein_cell_bounds(u, grid, cert, cfg)
        ref = reference(u, grid, cfg)
        r = r.ravel()
        r_cert = jds.scaling_factors(u, grid, estimate=CERT, cert=cert, cert_config=cfg)[0].ravel()
        r_c27 = jds.scaling_factors(u, grid, estimate=C27)[0].ravel()
        assert np.array_equal(b.ravel(), ref["b"], equal_nan=True), name
        assert np.array_equal(bt, ref["bt"], equal_nan=True), name
        assert np.array_equal(r_cert, ref["r_cert"]), name
        assert np.all(r >= r_cert) and np.all(r <= r_c27), name              # r'' >= r', r'' <= r(closed27)
        assert np.all(np.isfinite(bt)) and np.all(bt > 0), name
        assert np.all(bt <= ref["J53"]), name                                # every cell, no tolerance
        raised = ref["raised"]
        assert np.all(bt[raised] <= ref["J53"][raised]) and np.all(bt[raised] > ref["b"][raised]), name
        assert np.array_equal(S_impl, ref["S"]) and np.array_equal(ell_impl, ref["ell"][ref["S"]], equal_nan=True), name
        pre, fb = ref["code"] == PROVEN_PRESCREEN, ref["code"] == PROVEN_FALLBACK
        assert np.array_equal(ref["ell"][pre], ref["b"][pre]), name         # prescreen cells: l == b bit for bit
        assert not np.any(ref["ell"][fb] > 0), name                          # fallback cells: l not positive
        assert np.array_equal(r, ref["r_new"]) and np.array_equal(r, ref["r_full"]), name
        assert rec["n_cells_raised"] == raised.size and rec["n_bernstein_j53_violations"] == 0, name
        if rec["n_cells_raised"]:
            raised_fields.append(name)
        if rec["n_nodes_r_raised_vs_certificate"]:
            assert rec["n_nodes_r_raised_vs_certificate"] == int(np.count_nonzero(r > r_cert))
            nodes_raised_fields.append(name)
        if raised.size and np.any(bt[raised] >= 2*ref["b"][raised]):
            doubled_fields.append(name)
    assert seen >= 8
    assert raised_fields and nodes_raised_fields and doubled_fields
    assert "wavy1.2" in raised_fields and "edge_dip" in doubled_fields


# ---- gate 3: the edge dips stay visible --------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["mid", "off"])
def test_gate3_edge_dip_cells_keep_their_certificate_bound(kind):
    grid, cell, u, _ = dip_problem(kind=kind)
    cert, cfg = certified(grid, u)
    code, b = cell_pass(u, grid, cfg)
    K = int(np.ravel_multi_index(cell, grid.cell_shape))
    assert code[K] == PROVEN_FALLBACK
    bt, bounds, rec = jds.bernstein_cell_bounds(u, grid, cert, cfg)
    assert bt[K].hex() == b[K].hex() == float(bounds[cell]).hex()
    r = jds.scaling_factors(u, grid, estimate=BERN, cert=cert, cert_config=cfg)[0]
    r_cert = jds.scaling_factors(u, grid, estimate=CERT, cert=cert, cert_config=cfg)[0]
    assert max(r[v] for v in vertices_of(cell)) < .01
    assert all(r[v] == r_cert[v] for v in vertices_of(cell))


# ---- gate 4: check (b) by injection; check (a) raises ------------------------------------------------------------

def test_gate4_injected_violation_falls_back_counts_and_warns(monkeypatch):
    grid, u = wavy12()
    cert, cfg = certified(grid, u)
    bt0, b0, rec0 = jds.bernstein_cell_bounds(u, grid, cert, cfg)
    j53 = jds.closed27_cell_estimate(u, grid)[0].ravel()
    raised = np.flatnonzero(bt0 > b0.ravel())
    assert raised.size >= 2 and rec0["n_bernstein_j53_violations"] == 0
    K = int(raised[1])
    value = float(j53[K]+1e-3)                     # above J at the 53 points of K
    original = jds.prescreen_lower_bounds

    def patched(u_, grid_, coefficients, cells, **kw):
        out = np.array(original(u_, grid_, coefficients, cells, **kw))
        out[np.flatnonzero(cells == K)] = value
        return out
    monkeypatch.setattr(jds, "prescreen_lower_bounds", patched)
    jds.clear_cell_bounds_cache()
    bt, b, rec = jds.bernstein_cell_bounds(u, grid, cert, cfg)          # no exception
    assert bt[K].hex() == b.ravel()[K].hex()
    others = np.arange(grid.n_cells) != K
    assert np.array_equal(bt[others], bt0[others])
    assert rec["n_bernstein_j53_violations"] == 1 and rec["bernstein_j53_violation_cells"] == [K]
    assert rec["bernstein_j53_max_excess"] == value-float(j53[K])
    assert rec["n_cells_raised"] == rec0["n_cells_raised"]-1
    warning = jds.j53_warning_row(dict(rec, at="freeze"))
    assert warning["kind"] == "warning" and warning["n_bernstein_j53_violations"] == 1
    assert warning["bernstein_j53_violation_cells"] == [K] and jds.j53_warning_row(rec0) is None
    # through JDScaledMetric (the S~ record carries the fields)
    metric = JDScaledMetric(Identity(), grid, u, config=JDScalingConfig(cell_estimate=BERN), cert=cert,
                            cert_config=cfg)
    assert metric.computations[0]["n_bernstein_j53_violations"] == 1


def test_gate4_run_total_equals_injections_solve_direct(monkeypatch):
    calls = inject_first(monkeypatch)
    backend = NumpyBackend()
    grid, state, u0 = fb_dip_problem()
    start = Commit(u0, restrict_free(u0, grid), backend.certify(u0, grid))
    rows = []
    result = solve_direct(backend, state, start, target=threshold(g0_of(backend, state)), caps=Caps(max_accepted=30),
                          log=rows.append, pg=PGConfig(jd=JDScalingConfig(cell_estimate=BERN)))
    computations = result.metric["jd_scaling"]["computations"]
    with_S = [c for c in computations if c["n_cells_bernstein"]]
    hits = [c for c in with_S if c["bernstein_cached"]]
    assert with_S and len(calls) == len(with_S)-len(hits)               # one prescreen call per uncached S~
    assert all(c["n_bernstein_j53_violations"] == 1 for c in with_S)     # also on a cache hit
    round_row = dict(metric=result.metric)
    assert jds.j53_violations_total([round_row]) == len(with_S)
    warnings = [r for r in rows if r["kind"] == "warning"]
    scaling_rows = [r for r in rows if r["kind"] == "jd_scaling"]
    assert len(warnings) == len(with_S) == sum(1 for r in scaling_rows if r["n_bernstein_j53_violations"])
    assert all(w["warning"] == jds.J53_WARNING and w["n_bernstein_j53_violations"] == 1 for w in warnings)
    assert result.status in ("cap_reached", "line_search_failed")        # the run goes on (no abort)


def test_gate4_check_a_violations_raise(monkeypatch):
    grid, u = wavy12()
    cert, cfg = certified(grid, u)
    jds.bernstein_cell_bounds(u, grid, cert, cfg)                         # fills both cache levels
    for cached in (True, False):
        if not cached:
            jds.clear_cell_bounds_cache()
        with pytest.raises(CertificateContractError):                     # another configuration version
            jds.bernstein_cell_bounds(u, grid, replace(cert, certificate_version=cert.certificate_version+"-x"), cfg)
        with pytest.raises(CertificateContractError):                     # sha256(u) differs from state_hash
            jds.bernstein_cell_bounds(u*.999, grid, cert, cfg)
        tampered = replace(cert, min_jd_lower_bound=float(np.nextafter(cert.min_jd_lower_bound, 1.)))
        with pytest.raises(CertificateContractError):                     # min b_K != min_jd_lower_bound
            jds.bernstein_cell_bounds(u, grid, tampered, cfg)
    jds.clear_cell_bounds_cache()
    original = cert_eval.cell_pass

    def unproven(u_, grid_, config):
        code, lower = original(u_, grid_, config)
        code = code.copy()
        code[0] = cert_eval.UNCERTAIN
        return code, lower
    monkeypatch.setattr(cert_eval, "cell_pass", unproven)
    with pytest.raises(CertificateContractError):                         # a cell left unproven
        jds.bernstein_cell_bounds(u, grid, cert, cfg)
    with pytest.raises(CertificateContractError):
        jds.scaling_factors(u, grid, estimate=BERN, cert=cert, cert_config=cfg)
    monkeypatch.undo()
    with pytest.raises(InputValidationError):
        JDScaledMetric(Identity(), grid, u, config=JDScalingConfig(cell_estimate=BERN), cert=cert)   # no config


# ---- gate 5: cache ------------------------------------------------------------------------------------------------

def test_gate5_cache_key_hit_check_b_on_hit_and_alternation(monkeypatch):
    grid, u = wavy12()
    cert, cfg = certified(grid, u)
    bt1, _, rec1 = jds.bernstein_cell_bounds(u, grid, cert, cfg)
    assert not rec1["bernstein_cached"] and not rec1["cell_bounds_cached"]
    key = jds._CELL_BOUNDS_CACHE["key"]
    assert jds._BERNSTEIN_CACHE["key"] == (key, BERN, float(jds.J_ACT).hex())
    bt2, _, rec2 = jds.bernstein_cell_bounds(u.copy(), grid, cert, cfg)
    assert rec2["bernstein_cached"] and rec2["cell_bounds_cached"] and np.array_equal(bt1, bt2)
    # certificate first, then the new option: the b_K level hits, the l_S level does not (two-level cache)
    jds.clear_cell_bounds_cache()
    jds.certificate_cell_bounds(u, grid, cert, cfg)
    _, _, rec3 = jds.bernstein_cell_bounds(u, grid, cert, cfg)
    assert rec3["cell_bounds_cached"] and not rec3["bernstein_cached"]
    # check (b) also on a hit: an injected value is counted on the first and on the cached call
    jds.clear_cell_bounds_cache()
    calls = inject_first(monkeypatch)
    _, _, first = jds.bernstein_cell_bounds(u, grid, cert, cfg)
    _, _, second = jds.bernstein_cell_bounds(u, grid, cert, cfg)
    assert len(calls) == 1 and second["bernstein_cached"]
    assert first["n_bernstein_j53_violations"] == second["n_bernstein_j53_violations"] == 1
    monkeypatch.undo()
    # alternating certificate / certificate_bernstein: each equals its separate call
    jds.clear_cell_bounds_cache()
    alone = {e: jds.scaling_factors(u, grid, estimate=e, cert=cert, cert_config=cfg)[0] for e in (CERT,)}
    jds.clear_cell_bounds_cache()
    alone[BERN] = jds.scaling_factors(u, grid, estimate=BERN, cert=cert, cert_config=cfg)[0]
    jds.clear_cell_bounds_cache()
    for estimate in (CERT, BERN, CERT, BERN, BERN, CERT):
        r, rec = jds.scaling_factors(u, grid, estimate=estimate, cert=cert, cert_config=cfg)
        assert np.array_equal(r, alone[estimate]), estimate
        assert rec["cell_estimate"] == estimate
    assert not np.array_equal(alone[CERT], alone[BERN])


# ---- gate 6: synthetic problems -----------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["mid", "off"])
def test_gate6_quadratic_dips_as_certificate(kind):
    """Expected (contract gate 6): as certificate, the step cap is used up (the dip cell keeps b~ = b)."""
    result, rows, metric = run_quadratic(BERN, kind=kind)
    jds.clear_cell_bounds_cache()
    reference_run, _, _ = run_quadratic(CERT, kind=kind)
    accepted = [r for r in rows if r["kind"] == "candidate" and r["accepted"]]
    assert all(r["cert_status"] == "certified_feasible" for r in accepted)
    assert (result.status, result.accepted) == (reference_run.status, reference_run.accepted) == ("max_iterations", 30)
    assert all(c["cell_estimate"] == BERN and c["cert_min_matches"] and c["n_bernstein_j53_violations"] == 0
               for c in metric.computations)


def test_gate6_fb_synthetic_as_certificate():
    backend = NumpyBackend()
    grid, state, u0 = fb_dip_problem()
    start = Commit(u0, restrict_free(u0, grid), backend.certify(u0, grid))
    target = threshold(g0_of(backend, state))
    out = {}
    for estimate in (CERT, BERN):
        jds.clear_cell_bounds_cache()
        out[estimate] = solve_direct(backend, state, start, target=target, caps=Caps(max_accepted=30),
                                     pg=PGConfig(jd=JDScalingConfig(cell_estimate=estimate)))
    new = out[BERN]
    assert new.status == out[CERT].status == "cap_reached" and new.accepted == 30
    assert np.all(np.diff(new.energies) < 0)
    assert jds.j53_violations_total([dict(metric=new.metric)]) == 0


# ---- gate 7: torch64 ----------------------------------------------------------------------------------------------

def _compare_new_fields(runs):
    records = [runs[name][0].metric["jd_scaling"]["computations"] for name in ("numpy", "torch64")]
    for x, y in zip(*records):
        assert x["cell_estimate"] == y["cell_estimate"] == BERN
        assert all(x[key] == y[key] for key in NEW_INT_KEYS), [(k, x[k], y[k]) for k in NEW_INT_KEYS if x[k] != y[k]]
        for key in ("min_bernstein_bound", "bernstein_activation"):
            assert abs(x[key]-y[key]) <= 1e-9*max(1., abs(x[key]))
    assert records[0][0]["n_cells_raised"] == records[1][0]["n_cells_raised"]


@pytest.mark.parametrize("case", TORCH_CASES)
def test_gate7_torch64_cpu_matches_numpy(case, monkeypatch):
    runs = compare_with_numpy("cpu", case, BERN, monkeypatch)
    _compare_new_fields(runs)


@pytest.mark.skipif(os.environ.get("JDC_TEST_CUDA") != "1",
                    reason="cuda only on request (JDC_TEST_CUDA=1, GPU idle; the lead runs it)")
@pytest.mark.parametrize("case", TORCH_CASES)
def test_gate7_torch64_cuda_matches_numpy(case, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("cuda not available")
    runs = compare_with_numpy("cuda", case, BERN, monkeypatch)
    _compare_new_fields(runs)


# ---- gate 8: S cells with l <= b (locally affine compression) -----------------------------------------------------

def test_gate8_affine_field_has_s_cells_not_raised():
    grid, u, det = affine_field()
    assert .1 < det < .45
    cert, cfg = certified(grid, u)
    r, rec = jds.scaling_factors(u, grid, estimate=BERN, cert=cert, cert_config=cfg)
    ref = reference(u, grid, cfg)
    S, ell, b = ref["S"], ref["ell"], ref["b"]
    kept = S[np.isfinite(ell[S]) & (ell[S] <= b[S])]
    assert kept.size >= 10
    bt, _, _ = jds.bernstein_cell_bounds(u, grid, cert, cfg)
    assert np.array_equal(bt[kept], b[kept])                             # b~ = b bit for bit
    assert rec["n_cells_raised"] < rec["n_cells_bernstein"] == S.size
    assert rec["n_cells_raised"] == S.size-kept.size
    r_cert = jds.scaling_factors(u, grid, estimate=CERT, cert=cert, cert_config=cfg)[0]
    assert np.all(r >= r_cert)
    assert np.array_equal(r.ravel(), ref["r_new"]) and np.array_equal(r.ravel(), ref["r_full"])
    assert rec["n_bernstein_j53_violations"] == 0 and rec["bernstein_j53_violation_cells"] == []


# ---- gate 9: CLI --------------------------------------------------------------------------------------------------

def test_gate9_run_pair_parse():
    base = ["--pair", "s/p", "--channel-source", "none", "--out", "x"]
    assert run_pair.parse_args(base+["--jd-cell-estimate", BERN]).jd_cell_estimate == BERN
    with pytest.raises(SystemExit):
        run_pair.parse_args(base+["--metric-jd-scaling", "off", "--jd-cell-estimate", BERN])
    config = run_pair.pg_config("affine_v1", BERN)
    assert config.jd_cell_estimate == BERN and config.record()["jd_scaling"]["cell_estimate"] == BERN


def test_gate9_learn_cli_accepts_and_records(tmp_path, monkeypatch):
    from bayesreg.learn import cli
    from bayesreg.seg import structure_layer
    from ..learn import _guard_fixtures as gf
    from ..learn.test_cli import argv, eta_file
    base = ["--out", "o", "--pairs-root", "p", "--eta0-json", "e"]
    args = cli.parse_args(base+["--jd-cell-estimate", BERN])
    assert args.jd_cell_estimate == BERN and cli.solver_switches(args)["jd_cell_estimate"] == BERN
    paths = gf.write_files(tmp_path/"split")
    monkeypatch.setattr(structure_layer, "REPO_REGISTRY", paths["registry"])
    out = tmp_path/"eb_bern"
    assert cli.main(argv(paths, out, "--dry-run", "--jd-cell-estimate", BERN, eta0=eta_file(tmp_path))) == 0
    record = json.loads((out/"config.json").read_text())
    assert record["switches"]["jd_cell_estimate"] == BERN and record["switches"]["metric_jd_scaling"] == "affine_v1"


def test_gate4_gate9_run_pair_records_switch_total_and_warnings(tmp_path, monkeypatch):
    """A whole run_pair run (26^3 synthetic pair, tissue, numpy) with --jd-cell-estimate certificate_bernstein and an
    injected violation per prescreen call.  J_act is raised to 1 (JDScalingConfig in run_pair) so that S is not empty
    on this mild pair: the switches record the option, every S~ computation with S counts one violation, log.jsonl has
    one warning row per such computation, and status.json bernstein_j53_violations_total equals the sum."""
    calls = inject_first(monkeypatch)
    monkeypatch.setattr(run_pair, "JDScalingConfig", functools.partial(JDScalingConfig, activation=1.))
    base = RD.write_inputs(tmp_path)
    code, out = RD.run_arm(base, tmp_path/"run_bern", ("--jd-cell-estimate", BERN))
    assert code == 0
    read = lambda name: json.loads((out/name).read_text(encoding="utf-8"))      # noqa: E731
    config, prov, hyper, status = read("config.json"), read("provenance.json"), read("hyper.json"), read("status.json")
    assert config["args"]["jd_cell_estimate"] == BERN and prov["switches"]["jd_cell_estimate"] == BERN
    assert config["solver"]["pg"]["jd_cell_estimate"] == BERN
    computations = [c for r in hyper["rounds"] for c in r["metric"]["jd_scaling"]["computations"]]
    with_S = [c for c in computations if c["n_cells_bernstein"]]
    assert with_S and all(c["bernstein_activation"] == 1. for c in computations)
    assert len(calls) == sum(1 for c in with_S if not c["bernstein_cached"])
    assert status["bernstein_j53_violations_total"] == len(with_S) == jds.j53_violations_total(hyper["rounds"])
    rows = [json.loads(line) for line in (out/"log.jsonl").read_text().splitlines()]
    warnings = [r for r in rows if r["kind"] == "warning"]
    assert len(warnings) == len(with_S) and all(w["warning"] == jds.J53_WARNING for w in warnings)
    assert status["export"]["certificate_passed"] is True


# ---- gate 10: non-default J_act ----------------------------------------------------------------------------------

def test_gate10_non_default_activation():
    grid, u = wavy12()
    cert, cfg = certified(grid, u)
    _, _, rec25 = jds.bernstein_cell_bounds(u, grid, cert, cfg, .25)
    r, rec = jds.scaling_factors(u, grid, activation=.5, estimate=BERN, cert=cert, cert_config=cfg)
    assert not rec["bernstein_cached"] and rec["cell_bounds_cached"]          # another key for J_act = .5
    assert jds._BERNSTEIN_CACHE["key"][2] == (.5).hex() and rec["bernstein_activation"] == .5
    ref = reference(u, grid, cfg, activation=.5)
    assert rec["n_cells_bernstein"] == ref["S"].size > rec25["n_cells_bernstein"]
    assert np.array_equal(r.ravel(), ref["r_new"]) and np.array_equal(r.ravel(), ref["r_full"])
    metric = JDScaledMetric(Identity(), grid, u, config=JDScalingConfig(activation=.5, cell_estimate=BERN),
                            cert=cert, cert_config=cfg)
    assert metric.computations[0]["bernstein_activation"] == .5 and np.array_equal(metric.r, r)
    assert metric.computations[0]["bernstein_cached"]


# ---- gate 11: coefficient table failures and extreme spacing ------------------------------------------------------

def _assert_no_lift(u, grid, cert, cfg, activation=jds.J_ACT):
    bt, b, rec = jds.bernstein_cell_bounds(u, grid, cert, cfg, activation)
    assert np.all(np.isnan(jds._BERNSTEIN_CACHE["ell"])) and rec["n_cells_bernstein"] > 0
    assert np.array_equal(bt, b.ravel(), equal_nan=True) and rec["n_cells_raised"] == 0
    assert rec["min_bernstein_bound"] == rec["min_cert_bound"]
    r = jds.scaling_factors(u, grid, activation=activation, estimate=BERN, cert=cert, cert_config=cfg)[0]
    r_cert = jds.scaling_factors(u, grid, activation=activation, estimate=CERT, cert=cert, cert_config=cfg)[0]
    assert np.array_equal(r, r_cert)


def test_gate11_table_floating_point_error(monkeypatch):
    grid, u = wavy12()
    cert, cfg = certified(grid, u)

    class Failing:
        def __init__(self, *args, **kwargs):
            raise FloatingPointError("coefficient table")
    monkeypatch.setattr(jds, "CellQ1Operators3D", Failing)
    _assert_no_lift(u, grid, cert, cfg)


def test_gate11_subnormal_table(monkeypatch):
    grid, u = wavy12()
    cert, cfg = certified(grid, u)

    class Subnormal:
        def __init__(self, grid_, location):
            table = np.array(CellQ1Operators3D(grid_, location=location).coefficients)
            table[tuple(np.argwhere(table != 0)[0])] = 1e-310                # a nonzero subnormal entry
            self.coefficients = table
    monkeypatch.setattr(jds, "CellQ1Operators3D", Subnormal)
    _assert_no_lift(u, grid, cert, cfg)


def test_gate11_extreme_spacing_subnormal_table():
    """tests/three_d/test_certificate_prescreen_e71.py's grid with f/hx subnormal: the real table has a subnormal
    entry, so jd_curl's prescreen returns NaN.  u = 0 passes the certificate (edge hull, b = 1 - 1ulp-ish); J_act = 1
    puts every cell in S."""
    grid = make_grid_3d(nx=5, ny=5, nz=5, hx=4.4e307, hy=1e-150, hz=1e-150, unit="mm")
    D = CellQ1Operators3D(grid, location="closed27_uniform").coefficients
    assert np.abs(D[D != 0]).min() < np.finfo(float).tiny
    u = np.zeros(grid.shape+(3,))
    cert, cfg = certified(grid, u)
    _assert_no_lift(u, grid, cert, cfg, activation=1.)


# ---- gate 12: record fields consistent ----------------------------------------------------------------------------

def test_gate12_record_fields_consistent():
    backend = NumpyBackend()
    cfg = backend.cert_config
    fields = [(name, grid, u) for name, grid, u in certified_fields()]
    grid, u, _ = affine_field()
    fields.append(("affine", grid, u))
    checked = 0
    for name, grid, u in fields:
        cert = backend.certify(u, grid)
        if not cert.passed:
            continue
        checked += 1
        jds.clear_cell_bounds_cache()
        _, rec = jds.scaling_factors(u, grid, estimate=BERN, cert=cert, cert_config=cfg)
        assert set(rec) == RECORD_KEYS, name
        code, _ = cell_pass(u, grid, cfg)
        result = certify_map(u, grid, config=cfg)
        assert rec["n_cells_hull"] == int(np.count_nonzero(code == PROVEN_HULL)) == result.n_fast_cells, name
        assert rec["n_cells_raised"] <= rec["n_cells_bernstein"], name
        assert rec["min_bernstein_bound"] >= rec["min_cert_bound"], name
        assert rec["min_cert_bound"] == cert.min_jd_lower_bound, name
        assert rec["bernstein_activation"] == jds.J_ACT and rec["prescreen_kernel"] == PRESCREEN_KERNEL, name
        assert rec["n_bernstein_j53_violations"] == 0 and rec["bernstein_j53_max_excess"] is None, name
        assert rec["bernstein_j53_violation_cells"] == [], name
        assert (rec["bernstein_j53_min_margin"] is None) == (rec["n_cells_raised"] == 0), name
        if rec["n_cells_raised"]:
            assert rec["bernstein_j53_min_margin"] >= 0, name
        assert rec["cell_estimate"] == BERN and rec["cert_min_matches"] is True, name
        assert rec["n_r_lt_0p01"] <= rec["n_r_lt_0p01_certificate"], name
        assert rec["n_r_lt_1"] <= rec["n_r_lt_1_certificate"], name
    assert checked >= 9


# ---- gate 13: FMA regression of the prescreen kernel ---------------------------------------------------------------

FMA_ASM = re.compile(r"\bv?f(n)?m(add|sub)\w*", re.IGNORECASE)
FMA_INTRINSIC = re.compile(r"llvm\.(fma|fmuladd)\b")
FP_FLAGS = re.compile(r"\b(fadd|fsub|fmul|fdiv|frem|fneg|fcmp|call)\b[^\n]*\b(fast|contract|reassoc)\b")
# JDTI review minor 7 (beyond the contract, which names fast / contract / reassoc): arcp turns a division into a
# multiplication by the reciprocal and afn approximates functions -- either would also break the T75 error bound
FP_FLAGS_EXTRA = re.compile(r"\b(fadd|fsub|fmul|fdiv|frem|fneg|fcmp|call)\b[^\n]*\b(arcp|afn)\b")


def host_has_fma():
    """True when the host CPU has FMA (llvmlite's FeatureMap, {name: bool}; NOT a substring test of its flattened
    string, which lists "-fma" and e.g. "avx512ifma")."""
    import llvmlite.binding as llvm
    return bool(llvm.get_host_cpu_features().get("fma", False))


def compiled_prescreen_kernel():
    """jd_curl's prescreen kernel after one call (cache=False: compiled in this process for the host CPU)."""
    grid = make_grid_3d(nx=5, ny=5, nz=5, hx=1., hy=1., hz=1., unit="mm")
    u = np.zeros(grid.shape+(3,))
    u[1:-1, 1:-1, 1:-1] = .1*np.random.default_rng(0).uniform(-1, 1, u[1:-1, 1:-1, 1:-1].shape)
    table = CellQ1Operators3D(grid, location="closed27_uniform").coefficients
    bounds = prescreen_lower_bounds(u, grid, table, np.arange(grid.n_cells, dtype=np.int64), workers=1)
    assert np.all(np.isfinite(bounds))
    kernel = cps._kernel()
    assert kernel.signatures, "the prescreen kernel was not compiled"
    assert kernel.targetoptions.get("fastmath") is False
    return kernel


def test_gate13_prescreen_kernel_ir_has_no_fma_intrinsic_and_no_fast_math_flags():
    """The LLVM IR check (independent of the host CPU): no llvm.fma / llvm.fmuladd, no fast / contract / reassoc
    flag (contract gate 13), and no arcp / afn either (review minor 7)."""
    kernel = compiled_prescreen_kernel()
    for signature in kernel.signatures:
        llvm = kernel.inspect_llvm(signature)
        assert llvm
        assert not FMA_INTRINSIC.search(llvm)
        bad = [line.strip() for line in llvm.splitlines() if FP_FLAGS.search(line) or FP_FLAGS_EXTRA.search(line)]
        assert not bad, bad[:3]
        assert re.search(r"\bfmul\b", llvm) and re.search(r"\bfadd\b", llvm)      # the IR holds the arithmetic


def test_gate13_prescreen_kernel_asm_has_no_fma_instruction():
    """The machine-code check (contract gate 13): no vfmadd / vfmsub / vfnmadd / vfnmsub.  It is only meaningful on a
    host with FMA -- without it LLVM emits no FMA instruction even when contraction is allowed, and the check would
    pass vacuously (review minor 7) -- so it is skipped there, with the reason; the IR test above still runs."""
    if not host_has_fma():
        pytest.skip("host CPU without FMA (llvmlite get_host_cpu_features()['fma'] is False): the asm check would be "
                    "vacuous; the IR check (test_gate13_prescreen_kernel_ir_...) covers the contraction flags")
    kernel = compiled_prescreen_kernel()
    for signature in kernel.signatures:
        asm = kernel.inspect_asm(signature)
        assert asm
        assert not FMA_ASM.findall(asm), sorted(set(m.group(0) for m in FMA_ASM.finditer(asm)))[:5]


def test_gate13_patterns_catch_fma_and_flags():
    """The patterns of the FMA test find what they must find (asm mnemonics, intrinsics, flags)."""
    assert FMA_ASM.search("\tvfmadd231sd\t%xmm1, %xmm2, %xmm0") and FMA_ASM.search("vfnmsub213pd")
    assert FMA_ASM.search("vfmsub132sd") and FMA_ASM.search("vfnmadd231sd")
    assert not FMA_ASM.search("\tvmulsd\t%xmm1, %xmm2, %xmm0") and not FMA_ASM.search("vaddsd")
    assert FMA_INTRINSIC.search("call double @llvm.fmuladd.f64(double %a, double %b, double %c)")
    assert FMA_INTRINSIC.search("declare double @llvm.fma.f64(double, double, double)")
    assert FP_FLAGS.search("  %.5 = fadd fast double %.3, %.4") and FP_FLAGS.search("%x = fmul contract double %a, %b")
    assert FP_FLAGS.search("%y = fsub reassoc nsz double %a, %b")
    assert not FP_FLAGS.search("  %.5 = fadd double %.3, %.4")
    assert FP_FLAGS_EXTRA.search("%z = fdiv arcp double %a, %b") and FP_FLAGS_EXTRA.search("%w = call afn double @f()")
    assert not FP_FLAGS_EXTRA.search("  %.5 = fdiv double %.3, %.4")
    assert isinstance(host_has_fma(), bool)


# ---- JDTI review minor 1: violations on every path reach the run total and the log ---------------------------------

@pytest.mark.parametrize("estimate, error", [(BERN, RuntimeError), (BERN, ValueError), (CERT, RuntimeError)])
def test_minor1_round_exception_after_a_violation(tmp_path, monkeypatch, estimate, error):
    """A violation is injected at the round's freeze S~ (J_act = 1 so that S is not empty), then the outer solve
    raises.  RuntimeError (a ROUND_EXCEPTIONS member): the run ends level_failed and the exception row carries the
    round's S~ ledger (`jd_scaling`), so status.json's total counts it.  ValueError: it escapes solve_multilevel, the
    run ends with status exception, and the total comes from the warning rows of log.jsonl.  certificate: the
    exception row is that of the code before (no `jd_scaling`), no total."""
    from bayesreg.solve import outer
    calls = inject_first(monkeypatch)
    if estimate == BERN:
        monkeypatch.setattr(run_pair, "JDScalingConfig", functools.partial(JDScalingConfig, activation=1.))

    def failing(*args, **kwargs):
        raise error("injected after the freeze S~ (JDTI review minor 1 test)")
    monkeypatch.setattr(outer, "run_pg", failing)
    base = RD.write_inputs(tmp_path)
    code, out = RD.run_arm(base, tmp_path/"run", ("--jd-cell-estimate", estimate))
    assert code != 0
    status = json.loads((out/"status.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (out/"log.jsonl").read_text().splitlines()]
    warnings = [r for r in rows if r["kind"] == "warning"]
    scaling_rows = [r for r in rows if r["kind"] == "jd_scaling"]
    assert len(scaling_rows) == 1 and scaling_rows[0]["at"] == "round_start"
    if estimate == CERT:
        assert not calls and not warnings and "bernstein_j53_violations_total" not in status
        hyper = json.loads((out/"hyper.json").read_text(encoding="utf-8"))
        assert hyper["rounds"][-1]["status"] == "exception" and "jd_scaling" not in hyper["rounds"][-1]
        return
    assert len(calls) == 1 and len(warnings) == 1 and warnings[0]["n_bernstein_j53_violations"] == 1
    assert status["bernstein_j53_violations_total"] == 1
    if error is RuntimeError:
        assert status["status"] == "level_failed" and "bernstein_j53_violations_source" not in status
        hyper = json.loads((out/"hyper.json").read_text(encoding="utf-8"))
        row = hyper["rounds"][-1]
        assert row["status"] == "exception" and "metric" not in row
        assert [c["n_bernstein_j53_violations"] for c in row["jd_scaling"]["computations"]] == [1]
        assert jds.j53_violations_total(hyper["rounds"]) == 1
        logged = [r for r in rows if r["kind"] == "round" and r["status"] == "exception"]
        assert logged and logged[-1]["jd_scaling"]["computations"][0]["n_bernstein_j53_violations"] == 1
    else:
        assert status["status"] == "exception" and status["bernstein_j53_violations_source"].startswith("log.jsonl")
        assert not (out/"hyper.json").exists()


@pytest.mark.parametrize("estimate", [BERN, CERT])
def test_minor1_bb2_failure_after_a_refresh(monkeypatch, estimate):
    """The refresh succeeds (its record joins the computations), then the BB2 previous direction is not finite (the
    metric's solve returns NaN after the refresh): the solve ends nonfinite_evaluation.  certificate_bernstein logs
    the refresh's jd_scaling row and its warning row before ending; certificate keeps the log of the code before (no
    row on this path)."""
    from bayesreg.solve.stopping import NONFINITE_EVALUATION
    if estimate == BERN:
        inject_first(monkeypatch)
    original_refresh, original_solve = JDScaledMetric.refresh, JDScaledMetric.solve

    def refresh(self, u, **kwargs):
        record = original_refresh(self, u, **kwargs)
        self._jdti_nan = True
        return record

    def solve(self, g):
        return np.full_like(g, np.nan) if getattr(self, "_jdti_nan", False) else original_solve(self, g)
    monkeypatch.setattr(JDScaledMetric, "refresh", refresh)
    monkeypatch.setattr(JDScaledMetric, "solve", solve)
    result, rows, metric = run_quadratic(estimate)
    assert result.status == NONFINITE_EVALUATION and "BB2 previous direction" in result.reason
    assert metric.refreshes == 1 and metric.computations[-1]["at"] == "refresh"
    refresh_rows = [r for r in rows if r["kind"] == "jd_scaling" and r.get("at") == "refresh"]
    warnings = [r for r in rows if r["kind"] == "warning"]
    if estimate == CERT:
        assert not refresh_rows and not warnings
        return
    last = metric.computations[-1]
    assert last["n_bernstein_j53_violations"] == 1
    assert len(refresh_rows) == 1 and refresh_rows[0]["n_bernstein_j53_violations"] == 1
    assert refresh_rows[0]["cert_state_hash"] == last["cert_state_hash"]
    assert len(warnings) == 1 and warnings[0]["at"] == "refresh"
    assert rows.index(warnings[0]) == rows.index(refresh_rows[0])+1 == len(rows)-1
    assert jds.j53_violations_total([dict(metric=dict(jd_scaling=metric.record()))]) == sum(
        c["n_bernstein_j53_violations"] for c in metric.computations)


# ---- JDTI minors review, important 1: no JDScaledMetric outlives its round ----------------------------------------

def _round_with_tracker(option, pg, monkeypatch, fail=False):
    """solve_round on the near-fold problem (3 accepted steps) with a CommitTracker; weak references to every
    JDScaledMetric made.  fail: the inner solve raises after the freeze S~ (as a round exception)."""
    import gc
    import weakref
    from bayesreg.solve import CommitTracker, ProximalParams, default_lambda, solve_round
    from bayesreg.solve import outer
    from .test_jd_scaling import near_fold_problem
    refs = []
    original = JDScaledMetric.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        refs.append(weakref.ref(self))
    monkeypatch.setattr(JDScaledMetric, "__init__", init)
    if fail:
        def failing(*args, **kwargs):
            raise RuntimeError("injected after the freeze S~")
        monkeypatch.setattr(outer, "run_pg", failing)
    backend = NumpyBackend()
    grid, state = near_fold_problem()
    zero = np.zeros(grid.shape+(3,))
    start = Commit(zero, restrict_free(zero, grid), backend.certify(zero, grid))
    tracker = CommitTracker(start)
    params = ProximalParams(lam=default_lambda(grid)) if option == "b" else None
    try:
        result = solve_round(backend, state, start, option=option, target=threshold(g0_of(backend, state)),
                             caps=Caps(max_accepted=3, max_outer=2), params=params, pg=pg, tracker=tracker)
    except RuntimeError:
        result = None
    gc.collect()
    return result, tracker, refs


@pytest.mark.parametrize("option", ["a", "b"])
def test_no_scaled_metric_outlives_a_finished_round_default_estimate(option, monkeypatch):
    """Default estimate (certificate): after solve_round returns normally no JDScaledMetric is alive (weakref), as on
    main before the minors; the tracker holds no ledger."""
    result, tracker, refs = _round_with_tracker(option, PGConfig(), monkeypatch)
    assert result is not None and result.metric["jd_scaling"]["cell_estimate"] == CERT
    assert refs and tracker.ledger is None
    assert [r for r in refs if r() is not None] == []


def test_ledger_kept_only_while_a_bernstein_round_runs(monkeypatch):
    """certificate_bernstein: dropped at a normal return (no live metric); kept after an exception (the exception
    row reads it) -- the positive control of the weakref check."""
    pg = PGConfig(jd=JDScalingConfig(cell_estimate=BERN))
    result, tracker, refs = _round_with_tracker("a", pg, monkeypatch)
    assert result is not None and refs and tracker.ledger is None
    assert [r for r in refs if r() is not None] == []
    monkeypatch.undo()
    result, tracker, refs = _round_with_tracker("a", pg, monkeypatch, fail=True)
    assert result is None and tracker.ledger is not None and len(refs) == 1 and refs[0]() is not None
    assert tracker.ledger.record()["computations"][0]["cell_estimate"] == BERN
    tracker.ledger = None
    import gc
    gc.collect()
    assert refs[0]() is None
    result, tracker, refs = _round_with_tracker("a", PGConfig(), monkeypatch, fail=True)
    assert result is None and tracker.ledger is None                      # certificate: never kept
