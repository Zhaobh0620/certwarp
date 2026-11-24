"""Temperature scaling: on known over-confident logits the fitted T is > 1 (and near the truth), on
under-confident ones < 1; NLL and ECE drop; the 1-D search reports bounds."""
import math

import numpy as np
import pytest
import torch

from bayesreg.seg.unet.calibrate import ece, fit_temperature, nll


def _scaled_logits(scale, n=20000, c=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    true_logits = 2*torch.randn(n, c, generator=g, dtype=torch.float64)
    labels = torch.multinomial(torch.softmax(true_logits, 1), 1, generator=g)[:, 0]
    return (scale*true_logits).float(), labels


def test_overconfident_logits_give_temperature_above_one():
    logits, labels = _scaled_logits(3.)
    t, record = fit_temperature(logits, labels)
    assert t > 1 and abs(t-3.) < .3
    assert record["nll_after"] < record["nll_before"] and record["ece_after"] < record["ece_before"]
    assert not record["at_bound"] and record["n_voxels"] == 20000


def test_underconfident_logits_give_temperature_below_one():
    logits, labels = _scaled_logits(.5, seed=1)
    t, record = fit_temperature(logits, labels)
    assert t < 1 and abs(t-.5) < .08
    assert record["nll_after"] < record["nll_before"]


def test_nll_matches_direct_formula_and_ece_of_perfect_calibration_is_small():
    logits, labels = _scaled_logits(1., n=4000, c=3, seed=2)
    p = torch.softmax(logits.double()/2., 1)
    direct = float(-torch.log(p[torch.arange(4000), labels]).mean())
    assert nll(logits, labels, 2., chunk=1000) == pytest.approx(direct, rel=1e-12)
    assert ece(logits, labels, 1.) < .03


def test_temperature_search_flags_the_bound():
    logits = torch.tensor([[10., 0.], [10., 0.]])
    labels = torch.tensor([0, 1])
    t, record = fit_temperature(logits, labels, bounds=(.5, 2.))
    assert record["at_bound"] and math.isclose(t, 2., rel_tol=1e-3)


def test_fit_rejects_bad_shapes():
    from jd_curl.validation import InputValidationError
    with pytest.raises(InputValidationError):
        fit_temperature(torch.zeros(3, 2), torch.zeros(2, dtype=torch.long))
