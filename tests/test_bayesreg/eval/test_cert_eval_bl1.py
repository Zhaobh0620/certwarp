"""bayesreg.eval.cert_eval.certify_jacobian_only against the archived original, BL1 item 6's certify_jd_only (preregistration §3 "Jacobian 证书"; the lead's
changes of 11-07: no second copy, the existing one tested bit for bit; the archived module is never imported).

The original is archive/seq_vp_3d/results/verification/3d_stage_BL1_v1/analysis/bl1_cert.py (= eb808be results/verification/3d_stage_BL1_v1/analysis/
bl1_cert.py, sha256 332fb21b...).  Importing it would put the live seq_vp_3d tree on sys.path (bl1_cert.py:41-50), so this test does not import or execute
that file: BL1_SOURCE below is its lines 53-54 (`CERT`) and 61-73 (`certify_jd_only`) transcribed character for character -- a test compares the
transcription with the archived text (read as text only) -- and the transcription is executed against this repository's frozen jd_curl (feasibility as F,
numpy, hashlib, dataclasses.replace).
  1. Under BL1's own configuration (ParallelEnclosureCertificateConfig, tile 8192, fallback_workers 2) every field of the original's result equals
     cert_eval's; cert_eval's result type is PrescreenFeasibilityResult (one more field, n_prescreen_cells, 0 here).
  2. Under the C07-2 configuration the evaluation uses (PrescreenCertificateConfig, tile 8192) the original body cannot run as it is (its
     EnclosureFeasibilityResult base has no n_prescreen_cells); it is executed with that one name bound to PrescreenFeasibilityResult (the adaptation cert_eval
     made) and again every field is equal.
  3. The text of cert_eval's body is the original body with exactly the documented substitutions.
Small grids only (numba compiles the certificate kernels on first use)."""
import ast
from dataclasses import fields, replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jd_curl.three_d import feasibility as F
from jd_curl.three_d.grid import make_grid_3d

from bayesreg.eval import cert_eval as C
from bayesreg.solve.backend import certificate_config

REPO = Path(__file__).resolve().parents[3]
ARCHIVE = REPO/"archive"/"seq_vp_3d"/"results"/"verification"/"3d_stage_BL1_v1"/"analysis"/"bl1_cert.py"
ARCHIVE_SHA256 = "332fb21bbf5f2ae3aed52d1e98e20b59223d69806863e1c1e9d7eabdd6fbffbf"     # = git show eb808be:results/.../bl1_cert.py | sha256sum
ARCHIVE_LINES = ((53, 54), (61, 73))
BL1_SOURCE = '''CERT = F.ParallelEnclosureCertificateConfig(delta=0., max_depth=48, max_boxes=255, tile_size=8192,
                                            policy='lowest_bound_first_v1', prepass='edge_hull_v1', fallback_workers=2)
def certify_jd_only(u, grid, config=CERT):
    """certify_map(u, grid, config=config) without its NONZERO_BOUNDARY and INVALID_DOMAIN early returns (neither is a
    Jacobian property): the same owned C-order copy, digest, result record and per-cell parallel core."""
    u = np.array(F.as_float64_array(u, "u", grid.shape+(3,), check_finite=False), copy=True, order="C")
    digest = hashlib.sha256(memoryview(u).cast("B")).hexdigest()
    base = F.EnclosureFeasibilityResult("uncertain", None, None, None, 0, grid.n_cells, 0, 0, 0, None, None, "not_checked",
                                        digest, grid.grid_id, config.delta.hex(), config)
    base = replace(base, certificate_version=F.certificate_version(config))
    if not np.all(np.isfinite(u)):
        return replace(base, reason="NONFINITE_STATE")
    if not F.iv.arithmetic_supported():
        return replace(base, reason="unsupported_arithmetic")
    return F._certify_map_parallel(u, grid, config, base)
'''
SUBSTITUTIONS = (("F.as_float64_array(", "as_float64_array("), ("F.EnclosureFeasibilityResult(", "F.PrescreenFeasibilityResult("), ("F.iv.", "iv."))


def archived_text():
    """The archived lines, read as text (never imported, never executed)."""
    raw = ARCHIVE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ARCHIVE_SHA256
    lines = raw.decode("utf-8").splitlines()
    return "".join("\n".join(lines[a-1:b])+"\n" for a, b in ARCHIVE_LINES)


def archived_nodes():
    tree = ast.parse(BL1_SOURCE)
    cert = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "CERT")
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "certify_jd_only")
    return cert, function


def archived(feasibility=F):
    namespace = dict(np=np, hashlib=hashlib, replace=replace, F=feasibility)
    exec(compile(BL1_SOURCE, "bl1_cert.py (lines 53-54, 61-73, transcribed)", "exec"), namespace)
    return namespace["certify_jd_only"], namespace["CERT"]


def grid_of(n=7, h=1.):
    return make_grid_3d(nx=n, ny=n+1, nz=n+2, hx=h, hy=h, hz=h, unit="mm")


def smooth(grid, amplitude=.25):
    z, y, x = np.meshgrid(*[np.linspace(0, 1, s) for s in grid.shape], indexing="ij")
    bump = np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
    u = np.stack([amplitude*bump, -.5*amplitude*bump, .3*amplitude*bump], axis=-1)
    u[[0, -1]] = 0.
    u[:, [0, -1]] = 0.
    u[:, :, [0, -1]] = 0.
    return u


def fields_of(grid):
    u = smooth(grid)
    faces = u+.05                                                    # nonzero faces (not a Jacobian property)
    outside = u.copy()
    outside[1, 1, 1] = (-6., -6., -6.)                               # a node mapped outside the sampling domain
    fold = u.copy()
    fold[3, 3, 2, 0], fold[3, 3, 4, 0] = 1.6, -1.6                   # two x neighbours swapped: J <= 0 somewhere
    nonfinite = u.copy()
    nonfinite[2, 2, 2, 1] = np.nan
    return dict(zero=np.zeros(grid.shape+(3,)), smooth=u, faces=faces, outside=outside, fold=fold, nonfinite=nonfinite)


def same_fields(a, b):
    out = []
    for f in fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, float) and isinstance(y, float):
            if not (x == y or (np.isnan(x) and np.isnan(y))):
                out.append(f.name)
        elif x != y:
            out.append(f.name)
    return out


def test_the_transcription_is_the_archived_text():
    assert BL1_SOURCE == archived_text()
    cert, function = archived_nodes()
    assert ast.unparse(cert.value).startswith("F.ParallelEnclosureCertificateConfig(delta=0.0, max_depth=48, max_boxes=255, tile_size=8192")


def test_the_archived_module_is_never_imported():
    import sys
    assert not any("bl1_cert" in name for name in sys.modules)
    assert not any("seq_vp_3d" in p and "seq_vp_tag" not in p for p in sys.path)


def test_bit_for_bit_under_the_bl1_configuration():
    original, config = archived()
    assert isinstance(config, F.ParallelEnclosureCertificateConfig) and not isinstance(config, F.CompiledFallbackCertificateConfig)
    assert (config.delta, config.max_depth, config.max_boxes, config.tile_size, config.fallback_workers) == (0., 48, 255, 8192, 2)
    grid = grid_of()
    seen = set()
    for name, u in fields_of(grid).items():
        a = original(u, grid)
        b = C.certify_jacobian_only(u, grid, config)
        assert same_fields(a, b) == [], name
        assert type(a) is F.EnclosureFeasibilityResult and type(b) is F.PrescreenFeasibilityResult and b.n_prescreen_cells == 0
        seen.add((a.status, a.reason))
    assert ("infeasible", "JACOBIAN_VIOLATION") in seen and ("uncertain", "NONFINITE_STATE") in seen
    assert any(s == "certified_feasible" for s, _ in seen)


def test_bit_for_bit_under_the_c07_configuration_of_the_evaluation():
    view = SimpleNamespace(**{k: getattr(F, k) for k in dir(F) if not k.startswith("__")})
    view.EnclosureFeasibilityResult = F.PrescreenFeasibilityResult   # the one adaptation cert_eval made for the prescreen configuration
    original, _ = archived(view)
    config = certificate_config(8192)
    grid = grid_of()
    for name, u in fields_of(grid).items():
        a = original(u, grid, config)
        b = C.certify_jacobian_only(u, grid, config)
        assert type(a) is type(b) is F.PrescreenFeasibilityResult
        assert same_fields(a, b) == [], name
    with pytest.raises(TypeError):
        archived()[0](fields_of(grid)["fold"], grid, config)       # as written, the archived body cannot carry the prescreen count


def test_the_copy_is_the_archived_body_with_the_documented_substitutions():
    _, function = archived_nodes()
    theirs = ast.unparse(ast.Module(body=function.body[1:], type_ignores=[]))      # without the docstring
    for old, new in SUBSTITUTIONS:
        theirs = theirs.replace(old, new)
    tree = ast.parse(Path(C.__file__).read_text(encoding="utf-8"))
    mine = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "certify_jacobian_only")
    body = [n for n in mine.body[1:] if not isinstance(n, (ast.Import, ast.ImportFrom))]
    assert ast.unparse(ast.Module(body=body, type_ignores=[])) == theirs
