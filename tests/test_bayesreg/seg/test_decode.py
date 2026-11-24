"""bayesreg.seg.decode (work package R3b): the interface on synthetic posteriors.  No file, no label, no subject is read; CPU only."""
import numpy as np
import pytest

from bayesreg.seg import decode as D
from jd_curl.validation import InputValidationError


def dirichlet_probs(grid, *, seed=0, alpha=.3, dtype=np.float64):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.full(D.N_CLASSES, alpha), size=int(np.prod(grid))).T.reshape((D.N_CLASSES,)+tuple(grid))
    return (p/p.sum(axis=0)).astype(dtype)


def test_signature_output_and_empty_label_set_is_the_argmax():
    probs = dirichlet_probs((4, 5, 6))
    out = D.decode(probs, labels=())
    assert out.dtype == np.int16 and out.shape == (4, 5, 6)
    assert np.array_equal(out, np.argmax(probs, axis=0))
    labels, record = D.decode_with_record(probs, D.SMALL_SIX)
    assert labels.dtype == np.int16 and labels.shape == (4, 5, 6) and labels.min() >= 0 and labels.max() <= 35
    assert record["decoder_id"] == D.DECODER_ID and record["labels"] == list(D.SMALL_SIX)
    assert D.SMALL_SIX == (4, 18, 19, 23, 34, 35)


def test_one_hot_posterior_is_returned_unchanged():
    rng = np.random.default_rng(1)
    truth = rng.integers(0, D.N_CLASSES, size=(6, 7, 5))
    truth[2:4, 2:5, 1:3] = 18
    probs = np.moveaxis(np.eye(D.N_CLASSES)[truth], -1, 0)
    for objective in D.OBJECTIVES:
        assert np.array_equal(D.decode(probs, objective=objective), truth)


def test_labels_outside_the_set_follow_the_argmax_of_the_rest():
    probs = dirichlet_probs((5, 6, 7), seed=2, alpha=.2)
    out = D.decode(probs)
    rest = [k for k in range(D.N_CLASSES) if k not in D.SMALL_SIX]
    argmax_rest = np.asarray(rest)[np.argmax(probs[rest], axis=0)]
    keep = ~np.isin(out, D.SMALL_SIX)
    assert np.array_equal(out[keep], argmax_rest[keep])


@pytest.mark.parametrize("bad", ["float16", "channels", "negative", "sum", "nan", "list", "above_one"])
def test_input_is_refused(bad):
    probs = dirichlet_probs((3, 3, 3), seed=3)
    if bad == "float16":
        probs = probs.astype(np.float16)
    elif bad == "channels":
        probs = probs[:35]
    elif bad == "negative":
        probs[1, 0, 0, 0] = -1e-9
    elif bad == "sum":
        probs = probs*1.001
    elif bad == "nan":
        probs[2, 1, 1, 1] = np.nan
    elif bad == "list":
        probs = probs.tolist()
    elif bad == "above_one":                           # review r3b-math important 1: p = 1 + 2^-52 passed the sum check and emptied the label
        probs[:, 0, 0, 0] = 0.
        probs[18, 0, 0, 0] = np.nextafter(1., 2.)
    with pytest.raises(InputValidationError):
        D.decode(probs)


@pytest.mark.parametrize("labels", [(0,), (36,), (4, 4), "18", (4.7,), [True], None, 5, (np.float64(4.),)])
def test_label_set_is_checked(labels):
    with pytest.raises(InputValidationError):
        D.decode(dirichlet_probs((2, 2, 2)), labels)


def test_options_are_checked():
    probs = dirichlet_probs((2, 2, 2))
    with pytest.raises(InputValidationError):
        D.decode(probs, objective="dice")
    with pytest.raises(InputValidationError):
        D.decode(probs, candidate_floor=0.)


# ---- the algorithm against references that use none of the module's code (verify-math procedure; runs/20261111-r3/checks/decode_verify.py) ----
import itertools  # noqa: E402

RNG_SEED = 20261111


def _truths(n):
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.float64)


def _enum_expected_dice(p, region):
    """E[Dice(region, T)], every truth T of independent Bernoulli(p) enumerated; Dice(empty, empty) = 1."""
    y = _truths(len(p))
    w = np.prod(np.where(y == 1, p, 1-p), axis=1)
    r = np.asarray(region, dtype=np.float64)
    tau, gamma = r.sum(), y.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(w@np.where(tau+gamma > 0, 2*(y@r)/(tau+gamma), 1.))


def _enum_best_region(p, candidates):
    best, arg = -1., None
    for bits in itertools.product([0, 1], repeat=len(candidates)):
        r = np.zeros(len(p), bool)
        r[[c for c, b in zip(candidates, bits) if b]] = True
        value = _enum_expected_dice(p, r)
        if value > best+1e-15:
            best, arg = value, r
    return best, arg


def _case(rng, n_c, n_t, *, one=False):
    p = rng.uniform(1e-3, 1, n_c)
    if one and n_c:
        p[0] = 1.
    return np.sort(p)[::-1], rng.uniform(0, 1e-3, n_t)


def test_expected_dice_curve_equals_enumeration_of_the_truth():
    rng = np.random.default_rng(RNG_SEED)
    for trial in range(60):
        pc, pt = _case(rng, int(rng.integers(0, 9)), int(rng.integers(0, 3)), one=trial % 3 == 0)
        curve = D.expected_dice_curve(pc, D.tail_moments(pt))
        allp = np.concatenate([pc, pt])
        assert curve.shape == (pc.shape[0]+1,)
        for tau in range(pc.shape[0]+1):
            r = np.zeros(allp.shape[0], bool)
            r[:tau] = True
            assert abs(curve[tau]-_enum_expected_dice(allp, r)) <= 1e-13


def test_top_tau_is_the_optimum_over_every_subset_of_the_candidates():
    rng = np.random.default_rng(RNG_SEED+1)
    for trial in range(25):
        pc, pt = _case(rng, int(rng.integers(1, 9)), int(rng.integers(0, 3)), one=trial % 4 == 0)
        allp = np.concatenate([pc, pt])
        best, _ = _enum_best_region(allp, list(range(pc.shape[0])))
        curve = D.expected_dice_curve(pc, D.tail_moments(pt))
        assert abs(curve.max()-best) <= 1e-13


def _poisson_binomial(p):
    pmf = np.zeros(len(p)+1)
    pmf[0] = 1.
    for k, q in enumerate(p):
        pmf[1:k+2] = pmf[1:k+2]*(1-q)+pmf[0:k+1]*q
        pmf[0] *= 1-q
    return pmf


def test_quadrature_against_the_exact_poisson_binomial_on_a_large_set():
    rng = np.random.default_rng(RNG_SEED+2)
    pc = np.sort(np.clip(rng.beta(.4, .4, 400), 1e-3, 1))[::-1]
    pt = rng.uniform(0, 1e-3, 3000)
    curve = D.expected_dice_curve(pc, D.tail_moments(pt))
    for tau in (1, 57, int(np.argmax(curve)), 400):
        a, b = _poisson_binomial(pc[:tau]), _poisson_binomial(np.concatenate([pc[tau:], pt]))
        ka, kb = np.arange(a.shape[0])[:, None], np.arange(b.shape[0])[None, :]
        assert abs(curve[tau]-float(np.sum(a[:, None]*b[None, :]*2*ka/(tau+ka+kb)))) <= 1e-12


def test_tail_power_sums_equal_the_explicit_product():
    rng = np.random.default_rng(RNG_SEED+3)
    pc = np.sort(rng.uniform(1e-3, 1, 120))[::-1]
    pt = rng.uniform(0, 1e-3, 50000)
    explicit = D.expected_dice_curve(np.concatenate([pc, np.sort(pt)[::-1]]))[:pc.shape[0]+1]
    assert np.max(np.abs(D.expected_dice_curve(pc, D.tail_moments(pt))-explicit)) <= 1e-13


def test_ratio_objective_optimum_and_the_lipton_threshold():
    rng = np.random.default_rng(RNG_SEED+4)
    for _ in range(30):
        pc, pt = _case(rng, int(rng.integers(1, 10)), int(rng.integers(0, 3)))
        total = float(pc.sum()+pt.sum())
        curve = D.ratio_curve(pc, total)
        best = max((2*pc[np.array(b, bool)].sum()/(sum(b)+total) if any(b) else 0.)
                   for b in itertools.product([0, 1], repeat=pc.shape[0]))
        tau = int(np.argmax(curve))
        assert abs(curve[tau]-best) <= 1e-14
        assert np.all(pc[:tau] >= curve[tau]/2-1e-15) and np.all(pc[tau:] <= curve[tau]/2+1e-15)


def _reference_decode(probs, labels, floor):
    """Rule 1 by enumeration over every subset of the candidates, rule 2 voxel by voxel; none of the module's code."""
    n = probs.shape[1]
    accept = {c: _enum_best_region(probs[c], [i for i in range(n) if probs[c, i] >= floor])[1] for c in labels}
    argmax = np.argmax(probs, axis=0)
    out = argmax.copy()
    rest = [k for k in range(D.N_CLASSES) if k not in labels]
    for v in range(n):
        claim = [c for c in sorted(labels) if accept[c][v]]
        if claim:
            out[v] = max(claim, key=lambda c: (probs[c, v], -c))
        elif argmax[v] in labels:
            out[v] = max(rest, key=lambda k: (probs[k, v], -k))
    return out


def _mixed_posterior(n, labels, seed):
    rng = np.random.default_rng(seed)
    probs = np.zeros((D.N_CLASSES, n))
    pool = list(labels)+[0, 3, 14, 22, 17]
    for v in range(n):
        pick = rng.choice(pool, size=3, replace=False)
        probs[pick, v] = rng.dirichlet(np.full(3, .7))
    probs += 1e-6*rng.random(probs.shape)
    return probs/probs.sum(axis=0)


def _giveback_posterior(seed):
    rng = np.random.default_rng(seed)
    probs = np.zeros((D.N_CLASSES, 10))
    core = int(rng.integers(4, 7))
    probs[4, :core] = rng.uniform(.95, .99, core)
    probs[3, :core] = 1-probs[4, :core]
    probs[4, core:core+3] = rng.uniform(.34, .38, 3)
    probs[14, core:core+3] = (1-probs[4, core:core+3])*rng.uniform(.5, .55, 3)
    probs[22, core:core+3] = 1-probs[4, core:core+3]-probs[14, core:core+3]
    probs[19, core+3:] = rng.uniform(0, .2, 10-core-3)
    probs[0, core+3:] = 1-probs[19, core+3:]
    probs += 1e-6*rng.random(probs.shape)
    return probs/probs.sum(axis=0)


def test_whole_decoding_against_an_independent_reimplementation():
    twice = back = 0
    for seed in range(12):
        probs = _mixed_posterior(9, (4, 19, 23), seed)
        out, record = D.decode_with_record(probs, (4, 19, 23))
        assert np.array_equal(out, _reference_decode(probs, (4, 19, 23), D.CANDIDATE_FLOOR))
        twice += record["n_voxels_claimed_twice"] > 0
    for seed in range(6):
        probs = _giveback_posterior(seed)
        out, record = D.decode_with_record(probs, (4, 19))
        assert np.array_equal(out, _reference_decode(probs, (4, 19), D.CANDIDATE_FLOOR))
        back += record["n_given_back"] > 0
    assert twice > 0 and back > 0                      # both branches of rule 2 were exercised


def test_record_counts_add_up_and_the_grid_shape_is_kept():
    probs = dirichlet_probs((6, 5, 4), seed=5, alpha=.15)
    out, record = D.decode_with_record(probs)
    argmax = np.argmax(probs, axis=0)
    assert out.shape == (6, 5, 4) and record["n_voxels_changed"] == int(np.count_nonzero(out != argmax))
    for c in D.SMALL_SIX:
        rec = record["per_label"][str(c)]
        assert rec["final_volume"] == int(np.count_nonzero(out == c))
        assert rec["final_volume"] == rec["argmax_volume"]+rec["n_taken"]-rec["n_given_up"]
        assert rec["final_volume"] == rec["region_volume"]-rec["n_lost_to_other_decoded"]
    import json
    json.dumps(record)                                 # the record is plain JSON


def test_float32_and_float64_input_give_the_same_map_when_the_values_are_the_same():
    probs = dirichlet_probs((5, 4, 6), seed=6, alpha=.2, dtype=np.float32)
    probs = probs/probs.sum(axis=0, dtype=np.float32)
    assert np.array_equal(D.decode(probs), D.decode(probs.astype(np.float64)))


def test_expected_dice_of_any_set_equals_enumeration():
    rng = np.random.default_rng(RNG_SEED+5)
    for _ in range(40):
        n = int(rng.integers(1, 10))
        p = np.where(rng.random(n) < .25, rng.uniform(0, 1e-3, n), rng.uniform(0, 1, n))
        p[rng.random(n) < .1] = 1.
        region = np.flatnonzero(rng.random(n) < .5)
        mask = np.zeros(n, bool)
        mask[region] = True
        assert abs(D.expected_dice_of_set(p, region)-_enum_expected_dice(p, mask)) <= 1e-13


def test_the_record_carries_the_expected_dice_of_the_argmax_set():
    probs = _giveback_posterior(3)
    _, record = D.decode_with_record(probs, (4, 19))
    rec = record["per_label"]["4"]
    argmax_set = np.argmax(probs, axis=0) == 4
    assert abs(rec["expected_dice_argmax_set"]-_enum_expected_dice(probs[4], argmax_set)) <= 1e-13
    assert rec["objective_value"] >= rec["expected_dice_argmax_set"]-1e-15
