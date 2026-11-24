"""C07: the one certificate configuration, the map-level reason -> rejection mapping, the backend wrapper."""
import hashlib

import numpy as np
import pytest

from jd_curl.validation import InputValidationError
from jd_curl.three_d.feasibility import CertificateConfig, PrescreenCertificateConfig, certificate_version

from bayesreg import model
from bayesreg.solve import backend as bk

from . import _synthetic as syn


def test_certificate_config_is_contract_c07():
    config = bk.certificate_config()
    assert isinstance(config, PrescreenCertificateConfig)
    assert config.delta == 0. and config.delta.hex() == "0x0.0p+0"
    assert config.max_depth == 48 and config.max_boxes == 255 and config.tile_size == 8192
    record = bk.certificate_record(config)
    assert record["certificate_version"] == certificate_version(config)
    assert record["delta_hex"] == "0x0.0p+0" and record["tile_size"] == 8192
    assert bk.certificate_config(4096).tile_size == 4096
    for wrong in (CertificateConfig(), PrescreenCertificateConfig(delta=.01),
                  PrescreenCertificateConfig(delta=0., max_depth=12), PrescreenCertificateConfig(delta=0., max_boxes=16)):
        with pytest.raises(InputValidationError):
            bk.check_certificate_config(wrong, 8192)
    with pytest.raises(InputValidationError):
        bk.check_certificate_config(PrescreenCertificateConfig(delta=0.), 4096)    # recorded tile size differs


def b3_2_field(grid):
    """02_理论 B3(2): cell (2, 2, 2), local vertices (1,0,1), (1,1,1) moved by (0, 0, -1.05 h_z)."""
    u = np.zeros(grid.shape+(3,))
    u[3, 2, 3, 2] = u[3, 3, 3, 2] = -1.05*grid.spacing[2]
    return u


def test_certify_reason_mapping_on_real_calls():
    backend = bk.NumpyBackend()
    grid = syn.cube_grid(7)
    zero = np.zeros(grid.shape+(3,))
    ok = backend.certify(zero, grid)
    assert ok.passed and ok.rejection is None and 0 < ok.min_jd_lower_bound <= 1
    assert ok.state_hash == hashlib.sha256(zero.tobytes(order="C")).hexdigest()
    fold = backend.certify(b3_2_field(grid), grid)
    assert not fold.passed and (fold.status, fold.rejection) == ("infeasible", "JACOBIAN_VIOLATION")
    outside = zero.copy()
    outside[3, 3, 3, 0] = 50.
    domain = backend.certify(outside, grid)
    assert not domain.passed and (domain.status, domain.rejection) == ("infeasible", "INVALID_DOMAIN")
    nan = zero.copy()
    nan[3, 3, 3, 1] = np.nan
    nonfinite = backend.certify(nan, grid)
    assert not nonfinite.passed and (nonfinite.status, nonfinite.rejection) == ("uncertain", "NONFINITE_STATE")
    face = zero.copy()
    face[0, 3, 3, 0] = .1
    with pytest.raises(bk.CertificateContractError):
        backend.certify(face, grid)
    assert backend.certificate_calls == 4                     # the NONZERO_BOUNDARY call raised before counting


def test_rejection_reason_table():
    """C07-3: uncertain -> FEASIBILITY_UNCERTAIN (map reason kept by CertOutcome), zero bound never passes.
    (A real subdivision_budget case was not constructed: on random fields near the folding boundary the
    edge-hull / prescreen proved every certified cell even with max_depth 0 or max_boxes 1.)"""
    assert bk.rejection_reason("certified_feasible", "positive_enclosure_cover", .3) is None
    assert bk.rejection_reason("certified_feasible", "positive_bernstein_cover", 0.) == "FEASIBILITY_UNCERTAIN"
    assert bk.rejection_reason("certified_feasible", "positive_bernstein_cover", None) == "FEASIBILITY_UNCERTAIN"
    for reason in ("subdivision_budget", "arithmetic_uncertain", "unsupported_arithmetic"):
        assert bk.rejection_reason("uncertain", reason, None) == "FEASIBILITY_UNCERTAIN"
    assert bk.rejection_reason("uncertain", "NONFINITE_STATE", None) == "NONFINITE_STATE"
    assert bk.rejection_reason("infeasible", "JACOBIAN_VIOLATION", None) == "JACOBIAN_VIOLATION"
    assert bk.rejection_reason("infeasible", "INVALID_DOMAIN", None) == "INVALID_DOMAIN"
    for status, reason in (("infeasible", "NONZERO_BOUNDARY"), ("infeasible", "other"), ("weird", "x")):
        with pytest.raises(bk.CertificateContractError):
            bk.rejection_reason(status, reason, None)


def test_module_backend_is_the_model():
    """A backend over any module with the bayesreg.model names; the certificate stays the CPU call."""
    grid = syn.cube_grid(7)
    moving = syn.moving_image(grid.identity())
    state = model.ModelState(grid=grid, omega0=1., fixed=moving[None]+.01, samplers=model.build_samplers(moving[None], grid),
                             mask=syn.brain_box(grid, 1), sigma2=np.array([.1]), alpha=.1, nu=.1, tau2=np.full(4, 1.),
                             tissue=syn.uniform_tissue(grid))
    reference, wrapped = bk.NumpyBackend(), bk.module_backend(model, name="model-module", device="cpu")
    u = syn.solenoidal_displacement(grid, .3)
    a, b = reference.evaluate(state, u), wrapped.evaluate(state, u)
    assert a.energy == b.energy
    assert np.array_equal(reference.gradient(state, a), wrapped.gradient(state, b))
    assert wrapped.record()["certificate"]["certificate_version"] == reference.record()["certificate"]["certificate_version"]

    class Partial:
        evaluate = staticmethod(model.evaluate)
    with pytest.raises(InputValidationError):
        bk.module_backend(Partial, name="partial", device="cpu")
