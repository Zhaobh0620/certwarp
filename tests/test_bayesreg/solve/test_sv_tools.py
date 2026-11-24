"""SV (docs/bayesreg/notes/提速与换预分割器规划_20261104.md §7; runs/20261104-sv/plan.md, revision 1): experiments/sv/, CPU only, small synthetic volumes and files; no solver on a real image, no GPU, no unit.

- the posterior mix: f = 0 is the argmax and f > 1/2 the truth, bit for bit; the ties; the formula as written; low confidence first; the SDF mix, kept for comparison, with its staircase;
- the probe: the reductions, the volumes, the two f* by bisection, no label map written; the pins;
- the firewall: the 6 subjects, the first 5 pairs, the isolated root, the provenance the method refuses, oracle true in every record; f0440: only 0440 is the truth;
- the reading: every branch on made-up numbers, f0440 outside it, the endpoints in one table marked as another time;
- the gate, the start strings, a missing input or an f* not pinned stopping before any run, the queue on a fake machine, f0440 after the reading.
Anything that would start a unit fails the test (the fixture no_real_units).
"""
from fractions import Fraction
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/sv'))

import decide_sv as DEC  # noqa: E402
import postmix as MIX  # noqa: E402
import probe_sv as PROBE  # noqa: E402
import run_sv as RUN  # noqa: E402
import sdfmix as SDF  # noqa: E402

core, mc, seed = RUN.core, RUN.mc, RUN.seed
FROZEN = Path('/home/rmcfz/seq_vp_tag_results/frozen_eb808be')
PY = '/home/rmcfz/miniconda3/envs/sci310/bin/python'
SHAPE = (26, 28, 24)
ZERO, THIRD, HALF, TWO_THIRDS, ONE = Fraction(0), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(1)
BRIEF_PAIRS = ('0438-0439', '0439-0440', '0440-0441', '0441-0442', '0442-0443')


@pytest.fixture(autouse=True)
def no_real_units(monkeypatch):
    """No test of this file reaches the machine's units; a call that tried fails the test (the sandbox's systemd-run also exits 1)."""
    real = subprocess.run

    def guarded(argv, *args, **kwargs):
        first = argv[0] if isinstance(argv, (list, tuple)) else str(argv).split()[0]
        if Path(str(first)).name in ('systemd-run', 'systemctl'):
            pytest.fail(f'a test tried to reach a unit: {argv}')
        return real(argv, *args, **kwargs)
    monkeypatch.setattr(subprocess, 'run', guarded)


@pytest.fixture
def neutral():
    """A temporary directory whose path does not hold the word of the test runner: check_launch would refuse the start strings under tmp_path, as it should."""
    with tempfile.TemporaryDirectory(prefix='sv_tools_') as d:
        yield Path(d)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic_posterior(pred, truth, number):
    """A 36-class posterior whose argmax is pred; where truth differs, the truth label has the second place with a random margin."""
    rng = np.random.default_rng(number)
    P = rng.uniform(0., .002, (36,)+pred.shape)
    z, y, x = np.indices(pred.shape, sparse=True)
    P[pred, z, y, x] += rng.uniform(.5, 1., pred.shape)
    P[truth, z, y, x] += np.where(pred != truth, rng.uniform(0., .45, pred.shape), 0.)
    return P/P.sum(axis=0)


def quantized_posterior(shape, number):
    """Few distinct values, so that exact ties happen everywhere, as float16 storage makes them."""
    rng = np.random.default_rng(number)
    P = rng.integers(0, 5, (36,)+shape).astype(np.float64)
    P[0] += 1
    return P/P.sum(axis=0)


# ---- the posterior mix -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('number', (1, 2, 3))
def test_f_0_is_the_argmax_of_a5p_bit_for_bit(number):
    shape = (6, 7, 5)
    P = quantized_posterior(shape, number)
    truth = np.random.default_rng(number+9).integers(0, 36, shape)
    labels = MIX.mix(MIX.reduce_posterior(P, truth), 0.)
    assert labels.dtype == np.int16 and labels.tobytes() == mc.predicted_labels(P).tobytes()


@pytest.mark.parametrize('f', (.5000001, .51, .75, 1., 1))
def test_f_above_one_half_is_the_truth_bit_for_bit(f):
    shape = (6, 7, 5)
    P = quantized_posterior(shape, 4)
    truth = np.random.default_rng(5).integers(0, 36, shape).astype(np.float64)
    labels = MIX.mix(MIX.reduce_posterior(P, truth), f)
    assert labels.dtype == np.int16 and labels.tobytes() == truth.astype(np.int16).tobytes()


@pytest.mark.parametrize('f', (0., .0001, .02, .1, .2, .25, 1/3, .4, .49, .5, .6, .75, 1.))
@pytest.mark.parametrize('number', (6, 7))
def test_the_mix_is_the_formula_as_written_with_ties(f, number):
    shape = (9, 8, 7)
    P = quantized_posterior(shape, number)
    truth = np.random.default_rng(number+20).integers(0, 36, shape)
    assert MIX.mix(MIX.reduce_posterior(P, truth), f).tobytes() == MIX.mix_direct(P, truth, f).tobytes()


@pytest.mark.parametrize('a,t', ((3, 7), (7, 3)))
def test_a_tie_goes_to_the_smallest_label_as_np_argmax(a, t):
    """At f = .2 the two values are equal in the floats: .8 * .5 and .8 * .25 + .2."""
    P = np.zeros((36, 1, 1, 1))
    P[a], P[t], P[20] = .5, .25, .25
    truth = np.full((1, 1, 1), t, dtype=np.int16)
    assert .8*.5 == .8*.25+.2
    assert MIX.mix(MIX.reduce_posterior(P, truth), .2)[0, 0, 0] == 3 == MIX.mix_direct(P, truth, .2)[0, 0, 0]
    assert MIX.mix(MIX.reduce_posterior(P, truth), .19)[0, 0, 0] == a and MIX.mix(MIX.reduce_posterior(P, truth), .21)[0, 0, 0] == t


def test_a_tie_in_the_posterior_at_f_0_is_the_argmax_also_when_the_truth_is_one_of_them():
    P = np.zeros((36, 1, 1, 2))
    P[4], P[9] = .5, .5
    truth = np.array([[[9, 4]]], dtype=np.int16)
    assert MIX.mix(MIX.reduce_posterior(P, truth), 0.).tolist() == [[[4, 4]]] == np.argmax(P, axis=0).tolist()


def test_at_one_half_only_a_full_confidence_error_stays_a_tie():
    P = np.zeros((36, 1, 1, 2))
    P[5, 0, 0, 0] = 1.                                                           # p_a = 1, p_t = 0: both values 1/2
    P[5, 0, 0, 1], P[8, 0, 0, 1] = .9, .1
    truth = np.array([[[2, 8]]], dtype=np.int16)
    assert MIX.mix(MIX.reduce_posterior(P, truth), .5).tolist() == [[[2, 8]]]    # the first by the tie: 2 < 5
    truth = np.array([[[9, 8]]], dtype=np.int16)
    assert MIX.mix(MIX.reduce_posterior(P, truth), .5).tolist() == [[[5, 8]]]    # the same tie: 5 < 9
    assert MIX.mix(MIX.reduce_posterior(P, truth), .5000001).tolist() == [[[9, 8]]]


def test_low_confidence_errors_go_first_at_d_over_1_plus_d():
    margins = np.array([.05, .2, .5, .9])
    P = np.zeros((36, 1, 1, 4))
    P[1, 0, 0], P[2, 0, 0] = (1+margins)/2, (1-margins)/2
    truth = np.full((1, 1, 4), 2, dtype=np.int16)
    reduced = MIX.reduce_posterior(P, truth)
    for k, d in enumerate(margins):
        threshold = d/(1+d)
        assert MIX.mix(reduced, threshold-1e-6)[0, 0, k] == 1 and MIX.mix(reduced, threshold+1e-6)[0, 0, k] == 2
    assert MIX.mix(reduced, .1).ravel().tolist() == [2, 1, 1, 1] and MIX.mix(reduced, .3).ravel().tolist() == [2, 2, 1, 1] and MIX.mix(reduced, .4).ravel().tolist() == [2, 2, 2, 1]


def test_the_mix_changes_nothing_where_the_two_agree_and_takes_no_third_label():
    pred = SDF.synthetic_labels(SHAPE, 6)
    truth = SDF.synthetic_labels(SHAPE, 6, shift=2.)
    reduced = MIX.reduce_posterior(synthetic_posterior(pred, truth, 7), truth)
    assert reduced.pred.tobytes() == pred.tobytes() and reduced.truth.tobytes() == truth.tobytes()
    done = []
    for f in (.05, .1, .2, .3, .4, .45):
        labels = MIX.mix(reduced, f)
        row = SDF.movement(labels, pred, truth)
        assert row['agreeing_voxels_changed'] == 0 and row['third_label'] == 0 and row['equal_prediction']+row['equal_truth'] == row['disagree'] > 0
        done.append(row['equal_truth'])
    assert done == sorted(done) and done[0] < done[-1]


def test_the_argument_checks():
    for bad in (True, '1/3', None):
        with pytest.raises(TypeError):
            MIX.check_f(bad)
    for bad in (-.1, 1.1, float('nan')):
        with pytest.raises(ValueError):
            MIX.check_f(bad)
    with pytest.raises(ValueError):
        MIX.reduce_posterior(np.zeros((35, 2, 2, 2)), np.zeros((2, 2, 2), np.int16))
    with pytest.raises(ValueError):
        MIX.reduce_posterior(np.zeros((36, 2, 2, 2)), np.full((2, 2, 2), 36))


def test_the_reduction_of_the_error_by_hand():
    base = dict(a=dict(zip(MIX.MEASURES, [.9]*7)), b=dict(zip(MIX.MEASURES, [.7]*7)))
    rows = dict(a=dict(zip(MIX.MEASURES, [.95]*7)), b=dict(zip(MIX.MEASURES, [.8]*7)))
    got = MIX.error_reduction(rows, base)
    assert got['per_subject']['a']['seg35'] == pytest.approx(.5) and got['per_subject']['b']['seg4'] == pytest.approx(1/3)
    assert got['mean']['seg35'] == pytest.approx(1-(.05+.2)/(.1+.3))             # of the mean error, not the mean of the ratios
    base['a']['small_six'] = None
    assert MIX.error_reduction(rows, base)['mean']['small_six'] is None and MIX.error_reduction(rows, base)['per_subject']['a']['small_six'] is None
    base = dict(a=dict(zip(MIX.MEASURES, [1.]*7)))
    assert MIX.error_reduction(dict(a=dict(zip(MIX.MEASURES, [1.]*7))), base)['mean']['seg35'] is None


def test_volumes_and_vanished_labels():
    labels = np.array([0, 0, 1, 1, 1, 2], dtype=np.int16)
    counts = MIX.voxels(labels)
    assert len(counts) == 36 and counts[:3] == [2, 3, 1]
    pred, truth = MIX.voxels(np.array([0, 1, 1, 3, 3, 3])), MIX.voxels(np.array([0, 0, 1, 2, 4, 4]))
    assert MIX.vanished(counts, pred, truth) == [3, 4]
    assert MIX.ratios(counts, truth)[1:3] == [3., 1.] and MIX.ratios(counts, truth)[5] is None


# ---- the SDF mix (kept for the probe's comparison) ------------------------------------------------------------------------------------------------------------------------------------------------------
def slabs(boundary, left=1, right=2, shape=(5, 5, 32)):
    """The label `left` up to and including x = boundary, `right` after it; every y, z the same."""
    labels = np.full(shape, right, dtype=np.int16)
    labels[:, :, :boundary+1] = left
    return labels


def last_of(labels, label):
    """The last x of a label on the slab example, checked to be a slab."""
    assert (labels == labels[:1, :1, :]).all()
    return int(np.nonzero(labels[0, 0] == label)[0].max())


def brute_sdf(region):
    """The signed distance from its definition: every voxel against every voxel."""
    grid = np.stack(np.indices(region.shape), axis=-1).reshape(-1, 3).astype(float)
    inside, outside = grid[region.ravel()], grid[~region.ravel()]
    out = np.zeros(region.size)
    for k, (point, member) in enumerate(zip(grid, region.ravel())):
        other = outside if member else inside
        distance = np.sqrt(((other-point)**2).sum(axis=1)).min()
        out[k] = -distance if member else distance
    return out.reshape(region.shape)


def test_the_signed_distance_is_the_exact_one():
    rng = np.random.default_rng(1)
    region = rng.uniform(size=(7, 8, 6)) < .3
    got = SDF.signed_distance(region)
    assert np.array_equal(got, brute_sdf(region))
    assert (got[region] <= -1).all() and (got[~region] >= 1).all()
    assert SDF.signed_distance(np.zeros((3, 3, 3), bool)) is None
    full = SDF.signed_distance(np.ones((3, 4, 5), bool))
    assert (full == -np.sqrt(9+16+25)).all()


@pytest.mark.parametrize('seed_value', (1, 2, 3))
def test_sdf_f_0_is_the_prediction_and_f_1_the_truth_bit_for_bit(seed_value):
    pred = SDF.synthetic_labels(SHAPE, seed_value)
    truth = SDF.synthetic_labels(SHAPE, seed_value+50, shift=1.)
    assert pred.tobytes() != truth.tobytes()
    maps, _ = SDF.sdfmix(pred, truth, (ZERO, THIRD, TWO_THIRDS, ONE))
    assert maps[ZERO].dtype == np.int16 and maps[ZERO].tobytes() == pred.tobytes()
    assert maps[ONE].dtype == np.int16 and maps[ONE].tobytes() == truth.tobytes()


def test_sdf_the_end_points_are_exact_when_a_map_lacks_labels_and_is_stored_as_floats():
    pred = SDF.synthetic_labels(SHAPE, 4)
    truth = SDF.synthetic_labels(SHAPE, 4, shift=2.)
    pred[pred == 7] = 0
    truth[truth == 9] = 0
    maps, record = SDF.sdfmix(pred.astype(np.float64), truth.astype(np.float64), (ZERO, ONE))
    assert maps[ZERO].tobytes() == pred.tobytes() and maps[ONE].tobytes() == truth.tobytes()
    assert 7 in record['absent_labels']['prediction'] and 9 in record['absent_labels']['truth']


@pytest.mark.parametrize('layer,third,two_thirds', ((1, 0, 1), (3, 1, 2), (4, 1, 3), (6, 2, 4)))
def test_sdf_is_a_staircase_on_thin_layers(layer, third, two_thirds):
    """Review B1: the k-th voxel of a layer of D flips at f = k/(D + 1).  D = 1 only at f = 1/2: why this is not the rule of an arm."""
    maps, _ = SDF.sdfmix(slabs(10), slabs(10+layer), (THIRD, TWO_THIRDS))
    assert (last_of(maps[THIRD], 1)-10, last_of(maps[TWO_THIRDS], 1)-10) == (third, two_thirds)


def test_sdf_a_layer_of_2_is_decided_by_the_label_numbers():
    maps, _ = SDF.sdfmix(slabs(10), slabs(12), (THIRD, TWO_THIRDS))
    assert (last_of(maps[THIRD], 1)-10, last_of(maps[TWO_THIRDS], 1)-10) == (1, 2)
    maps, _ = SDF.sdfmix(slabs(10, 2, 1), slabs(12, 2, 1), (THIRD, TWO_THIRDS))
    assert (last_of(maps[THIRD], 2)-10, last_of(maps[TWO_THIRDS], 2)-10) == (0, 1)


@pytest.mark.parametrize('left,right', ((1, 2), (2, 1)))
def test_sdf_a_tie_goes_to_the_smallest_label(left, right):
    """Boundaries 9|10 and 14|15, f = 1/2: the two values of x = 12 are equal in exact arithmetic and in the floats (integer weights, integer distances)."""
    maps, _ = SDF.sdfmix(slabs(9, left, right), slabs(14, left, right), (HALF,))
    row = maps[HALF][0, 0]
    assert row[12] == 1
    assert row[11] == left and row[13] == right


def test_sdf_the_formula_by_brute_force():
    rng = np.random.default_rng(5)
    shape = (9, 10, 8)
    pred, truth = rng.integers(0, 4, shape).astype(np.int16), rng.integers(0, 4, shape).astype(np.int16)
    truth[0, 0, 0], pred[0, 0, 0] = 5, 6                                         # a label only each map has
    maps, _ = SDF.sdfmix(pred, truth, (THIRD, TWO_THIRDS))
    for f in (THIRD, TWO_THIRDS):
        wa, wb = f.denominator-f.numerator, f.numerator
        stack, formula = [], []
        for label in SDF.LABELS:
            a, b = pred == label, truth == label
            both = a.any() and b.any()
            stack.append(wa*brute_sdf(a)+wb*brute_sdf(b) if both else np.full(shape, np.inf))
            formula.append((1-float(f))*brute_sdf(a)+float(f)*brute_sdf(b) if both else np.full(shape, np.inf))      # the brief's formula as written
        assert np.array_equal(np.argmin(np.stack(stack), axis=0), maps[f])       # np.argmin: the first minimum, the smallest label
        formula = np.stack(formula)
        chosen = np.take_along_axis(formula, maps[f][None].astype(np.int64), axis=0)[0]
        assert np.all(chosen <= formula.min(axis=0)+1e-12)                       # what was chosen minimizes the formula, up to the rounding of a tie


def test_sdf_where_the_two_agree_nothing_changes():
    pred = SDF.synthetic_labels(SHAPE, 6)
    truth = SDF.synthetic_labels(SHAPE, 6, shift=2.)
    maps, record = SDF.sdfmix(pred, truth, (THIRD, HALF, TWO_THIRDS))
    for f, labels in maps.items():
        assert (labels[pred == truth] == pred[pred == truth]).all()
        row = record['per_fraction'][str(f)]
        assert row['agreeing_voxels_changed'] == 0 and row['disagree'] == int(np.count_nonzero(pred != truth))
        assert row['equal_prediction']+row['equal_truth']+row['third_label'] == row['disagree'] > 0


def test_sdf_a_label_only_one_map_has_stays_at_that_end():
    pred, truth = slabs(9), slabs(9)
    pred[0, 0, 0], truth[0, 0, 1] = 8, 7
    maps, record = SDF.sdfmix(pred, truth, (ZERO, THIRD, TWO_THIRDS, ONE))
    assert (maps[ZERO] == 8).sum() == 1 and (maps[ONE] == 7).sum() == 1
    for f in (THIRD, TWO_THIRDS):
        assert not np.isin(maps[f], (7, 8)).any()
    assert 7 in record['absent_labels']['prediction'] and 8 in record['absent_labels']['truth']
    assert record['absent_rule'] == SDF.ABSENT_RULE and record['tie_rule'] == SDF.TIE_RULE == 'the smallest label'


def test_sdf_a_label_that_fills_the_grid_has_a_finite_value():
    pred = np.full((6, 6, 6), 3, dtype=np.int16)
    truth = pred.copy()
    truth[:2] = 4
    maps, _ = SDF.sdfmix(pred, truth, (ZERO, HALF, ONE))
    assert (maps[ZERO] == 3).all() and maps[ONE].tobytes() == truth.tobytes()
    assert (maps[HALF] == 3).all()                                               # label 4 is absent from the prediction


def test_sdf_no_common_label_is_refused():
    with pytest.raises(ValueError, match='no label is present in both'):
        SDF.sdfmix(slabs(9, 1, 2), slabs(9, 3, 4), (HALF,))


def test_sdf_no_random_step_and_threads_change_nothing():
    pred = SDF.synthetic_labels(SHAPE, 7)
    truth = SDF.synthetic_labels(SHAPE, 7, shift=2.)
    first, _ = SDF.sdfmix(pred, truth, (THIRD, TWO_THIRDS), workers=1)
    again, _ = SDF.sdfmix(pred, truth, (TWO_THIRDS, THIRD), workers=1)
    threads, _ = SDF.sdfmix(pred, truth, (THIRD, TWO_THIRDS), workers=3)
    for f in (THIRD, TWO_THIRDS):
        assert first[f].tobytes() == again[f].tobytes() == threads[f].tobytes()
    assert 'random' not in inspect.getsource(SDF.sdfmix)


@pytest.mark.parametrize('bad', (.5, 1/3, True, '1/3'))
def test_sdf_f_must_be_exact(bad):
    with pytest.raises(TypeError):
        SDF.as_fraction(bad)


@pytest.mark.parametrize('bad', (2, -1, Fraction(4, 3)))
def test_sdf_f_must_lie_between_the_two(bad):
    with pytest.raises(ValueError):
        SDF.as_fraction(bad)


def test_sdf_the_weights_and_the_names():
    assert [SDF.weights(f) for f in (ZERO, THIRD, TWO_THIRDS, ONE)] == [(1, 0), (2, 1), (1, 2), (0, 1)]
    assert [SDF.tag(f) for f in (ZERO, THIRD, TWO_THIRDS, ONE)] == ['f0', 'f1_3', 'f2_3', 'f1']


@pytest.mark.parametrize('values', (np.array([[[.5]]]), np.array([[[36]]]), np.array([[[-1]]])))
def test_a_label_map_with_other_values_is_refused(values):
    with pytest.raises(ValueError):
        SDF.label_map(values)


def test_sdf_shapes_and_fractions_are_checked():
    with pytest.raises(ValueError):
        SDF.sdfmix(np.zeros((2, 2, 2), np.int16), np.zeros((2, 2, 3), np.int16), (HALF,))
    with pytest.raises(ValueError):
        SDF.sdfmix(np.zeros((2, 2, 2), np.int16), np.zeros((2, 2, 2), np.int16), (HALF, HALF))


def test_sdf_is_no_arm_and_the_build_does_not_call_it():
    source = (ROOT/'experiments/sv/run_sv.py').read_text()
    assert 'sdfmix.sdfmix(' not in source and all(spec['provenance'].startswith('oracle_postmix_') for spec in RUN.ARMS.values())
    assert 'sdfmix.sdfmix(' in (ROOT/'experiments/sv/probe_sv.py').read_text()


# ---- a synthetic tree: the stage, the predictions, Atropos, the endpoints' files ------------------------------------------------------------------------------------------------------------------------------------
def subject_maps(shape=SHAPE):
    out = {}
    lookup = core.make_inputs().load_seg4_map()[0]
    for k, subject in enumerate(RUN.SUBJECTS):
        truth, pred = SDF.synthetic_labels(shape, 100+k), SDF.synthetic_labels(shape, 100+k, shift=1.5)
        out[subject] = dict(truth=truth, pred=pred, P=synthetic_posterior(pred, truth, 300+k), truth4=lookup[truth].astype(np.int16), image=np.random.default_rng(200+k).uniform(1., 2., shape))
    return out


def per_labels(value):
    return dict(seg35_per_label={str(k): value for k in range(1, 36)}, seg4_per_label={str(k): value for k in range(1, 5)})


def metric_values(seg35=.84, seg4=.9, sdlogj=.08):
    return dict(seg35=seg35, seg4=seg4, dice_official=seg35, hd95=2., sdlogj_official=sdlogj, sdlogj_brain=.6, num_foldings_official=0., pct_J_lt_half_brain=14., cell_centre_J_nonpositive=0, **per_labels(seg35))


CERTIFIED = dict(passed=True, state_hash_matches_u=True, final_npz_sha256_matches=True, min_jd_lower_bound=1e-4, status='certified_feasible', n_infeasible=0, n_uncertain=0)
CONTROL_U = 'c'*64


def final_record(arm, pair, seg35, seg4=.9, sdlogj=.08, oracle=True, inputs_root='/x', **changes):
    doc = dict(run_id=f'made_up_{arm}_{RUN.short_name(pair)}', arm=arm, pair=pair, method='made up', classification={'class': 'ok', 'kind': 'native', 'reason': 'r'}, status='native_budget_exhausted', native_commits=70, e2e_seconds=325., oracle=oracle, run_labels_equal_truth=True,
               metrics=metric_values(seg35, seg4, sdlogj), certificate=dict(CERTIFIED), self_check=dict(passed=True),
               seed=dict(passed=True, problems=[], accepted_alpha=1., entry_metrics=dict(seg35=seg35-.02, seg4=seg4-.02, sdlogj_official=.05, accepted_alpha=1.), arm=dict(seconds=seed.SECONDS, changes=seed.CHANGES, inputs_root=str(inputs_root), charge_seconds=seed.ENTRY_CHARGE_SECONDS)))
    doc.update(changes)
    return doc


def make_tree(base, maps, shape=SHAPE):
    """<base>/results (the endpoints, A5 and A5P's images, the control's reference), <base>/stage (the 5 pairs), <base>/pred, <base>/seg: every input the queue asks for, of the real names, of a small size."""
    mi = core.make_inputs()
    table = core.load_table()
    results, stage = Path(base)/'results', Path(base)/'stage'
    stage.mkdir(parents=True)
    write_json(stage/'selection.json', dict(pairs=[dict(pair=p) for p in RUN.PAIRS]))
    write_json(stage/'completion.json', dict(note='synthetic'))
    for pair in RUN.PAIRS:
        fixed, moving = RUN.subjects_of(pair)
        arrays = dict(fixed=maps[fixed]['image'], fixed_seg4=maps[fixed]['truth4'].astype(np.float64), fixed_seg35=maps[fixed]['truth'].astype(np.float64), moving=maps[moving]['image'], moving_seg4=maps[moving]['truth4'].astype(np.float64), moving_seg35=maps[moving]['truth'].astype(np.float64), initial_u=np.zeros(tuple(shape)+(3,)))
        d = stage/pair
        d.mkdir()
        np.savez_compressed(d/'pair.npz', **{k: arrays[k] for k in mi.KEY_ORDER})
        write_json(d/'manifest.json', dict(dataset='synthetic', pair=pair, fixed_subject=fixed, moving_subject=moving, shape_zyx=list(shape), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], unit='mm', preprocessing='synthetic', baseline_native_mse=.1,
                                           sources=dict(fixed=dict(source='/x', sha256='a'*64), moving=dict(source='/y', sha256='b'*64)), pair_sha256=sha(d/'pair.npz')))
        for name, key in (('A5P', 'pred'), ('A5', 'truth')):
            images = {side: mi.colour_from_table(maps[subject]['image'], maps[subject][key], table)[0] for side, subject in zip(('fixed', 'moving'), (fixed, moving))}
            target = results/'d0_v1_a5/inputs'/name/mi.STAGE_NAME/pair
            target.mkdir(parents=True)
            np.savez(target/'pair.npz', **images)
        for arm, (seg35, seg4) in DEC.PINS.items():
            write_json(results/DEC.ENDPOINT_ROOT/'runs'/arm/pair/'metrics.json', final_record(arm, pair, seg35, seg4, oracle=arm == 'S_T'))
    write_json(seed.entry_dir('S_P', RUN.CONTROL_PAIR, results)/'run.json', dict(field=dict(u_sha256=CONTROL_U)))
    np.savez(seed.entry_dir('S_P', RUN.CONTROL_PAIR, results)/'field.npz', u=np.zeros(tuple(shape)+(3,)))
    for subject in RUN.SUBJECTS:
        p = Path(base)/'pred'/subject
        p.mkdir(parents=True)
        write_json(p/'provenance.json', dict(policy=mc.POLICY, segmenter_id=mc.SEGMENTER_ID, test_labels_seen=False))
        (p/'probs36.npz').write_bytes(b'x')
        (p/'structure_groups.npz').write_bytes(b'x')
        write_json(Path(base)/'seg'/subject/'provenance.json', dict(segmenter_id='atropos'))
    return RUN.Sources(results=results, stage=stage, pred_root=Path(base)/'pred', seg_root=Path(base)/'seg', pins=Path(base)/'sv_fstar.json')


def posterior_from(maps):
    def read(pin, side):
        return maps[getattr(pin, side+'_subject')]['P'], dict(made_up=True, subject=getattr(pin, side+'_subject'))
    return read


@pytest.fixture(scope='module')
def built():
    """One real probe, pin and build on the synthetic tree: every test that looks at their files; the tests that change them copy them."""
    with tempfile.TemporaryDirectory(prefix='sv_built_') as d:
        base = Path(d)
        maps = subject_maps()
        sources = make_tree(base, maps)
        root = sources.results/RUN.ROOT_NAME
        root.mkdir()
        logs = []
        probe = PROBE.run_probe(root, sources=sources, reader=posterior_from(maps), log=logs.append)
        assert RUN.pin_fstar(root, 'test', sources=sources, log=logs.append) == 0
        assert RUN.build_all(root, sources=sources, reader=posterior_from(maps), log=logs.append) == 0
        yield SimpleNamespace(base=base, maps=maps, sources=sources, root=root, logs=logs, probe=probe)


def pins_of(built):
    return json.loads(Path(built.sources.pins).read_text())


# ---- the probe ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_the_probe_writes_two_files_and_no_label_map(built):
    assert sorted(p.name for p in (built.root/'analysis').iterdir()) == [RUN.PROBE_JSON, RUN.PROBE_MD]
    assert RUN.PROBE_JSON.startswith('probe_') and RUN.PROBE_JSON.endswith('.json') and RUN.PROBE_MD.startswith('probe_') and RUN.PROBE_MD.endswith('.md')
    assert built.probe['oracle'] is True and 'NOT_in_any_result_table' in built.probe['role'] and built.probe['subjects'] == list(RUN.SUBJECTS)
    text = (built.root/'analysis'/RUN.PROBE_JSON).read_text()
    assert len(text) < 3_000_000 and '"labels"' not in text


def test_the_probe_has_every_measure_for_every_subject_and_the_mean(built):
    assert list(built.probe['measures']) == ['seg35', 'seg4', 'small_six', 'big_six', 'other_23', 'cortex', 'white_matter']
    grid = set(PROBE.f_key(f) for f in PROBE.F_GRID)
    assert grid <= set(built.probe['rows']) and min(PROBE.F_GRID) == .02 and max(PROBE.F_GRID) == .4999 and .49 in PROBE.F_GRID and PROBE.F_GRID == tuple(sorted(PROBE.F_GRID))
    for row in list(built.probe['rows'].values())+list(built.probe['sdf'].values()):
        assert set(row['reduction_mean']) == set(built.probe['measures']) == set(row['mean_dice'])
        assert set(row['per_subject']) == set(RUN.SUBJECTS) == set(row['reduction_per_subject'])
        for cells in row['per_subject'].values():
            assert len(cells['voxels']) == 36 and len(cells['volume_vs_truth']) == 36 and len(cells['volume_vs_prediction']) == 36 and isinstance(cells['vanished'], list) and {'seg35', 'seg4', 'cortex', 'white_matter'} <= set(cells['dice'])


def test_the_probes_numbers_are_those_of_the_maps(built):
    f = PROBE.F_GRID[10]
    row = built.probe['rows'][PROBE.f_key(f)]
    lookup = core.make_inputs().load_seg4_map()[0]
    subject = RUN.SUBJECTS[2]
    held = built.maps[subject]
    labels = MIX.mix_direct(held['P'], held['truth'], f)
    dice = RUN.segmentation_row(labels, held['truth'], held['truth4'], lookup)
    base = RUN.segmentation_row(held['pred'], held['truth'], held['truth4'], lookup)
    cells = row['per_subject'][subject]
    assert cells['dice']['seg35'] == pytest.approx(dice['seg35'], abs=0) and cells['voxels'] == MIX.voxels(labels)
    assert row['reduction_per_subject'][subject]['seg35'] == pytest.approx(1-(1-dice['seg35'])/(1-base['seg35']))
    assert cells['volume_vs_truth'][1:] == MIX.ratios(MIX.voxels(labels), MIX.voxels(held['truth']))[1:]


def test_the_probes_reduction_grows_on_the_grid_and_ends_near_all(built):
    values = [built.probe['rows'][PROBE.f_key(f)]['reduction_mean']['seg35'] for f in PROBE.F_GRID]
    assert values[-1] > .9 and values[0] < .2 and values == sorted(values)
    assert built.probe['grid_monotone_seg35'] is True


def test_the_probe_has_the_sdf_comparison(built):
    lookup = core.make_inputs().load_seg4_map()[0]
    assert set(built.probe['sdf']) == {'1/3', '2/3'} and all('对照' in row['role'] for row in built.probe['sdf'].values())
    subject = RUN.SUBJECTS[0]
    maps, _ = SDF.sdfmix(built.maps[subject]['pred'], built.maps[subject]['truth'], (THIRD, TWO_THIRDS))
    assert built.probe['sdf']['1/3']['per_subject'][subject]['dice']['seg35'] == pytest.approx(RUN.segmentation_row(maps[THIRD], built.maps[subject]['truth'], built.maps[subject]['truth4'], lookup)['seg35'], abs=0)


def test_the_two_f_star_are_within_the_tolerance_of_their_targets(built):
    assert set(built.probe['fstar']) == {'r1_3', 'r2_3'}
    for arm, target in RUN.TARGETS.items():
        row = built.probe['fstar'][arm]
        assert 0 < row['f'] < .5 and round(row['f'], 4) == row['f'] and row['target'] == target and abs(row['reduction']-target) <= RUN.TARGET_TOL
        assert built.probe['rows'][PROBE.f_key(row['f'])]['reduction_mean']['seg35'] == row['reduction']
    assert built.probe['fstar']['r1_3']['f'] < built.probe['fstar']['r2_3']['f']
    assert built.probe['pin_template']['r1_3']['f'] == built.probe['fstar']['r1_3']['f']


def test_the_probes_summary_says_what_it_is(built):
    text = (built.root/'analysis'/RUN.PROBE_MD).read_text(encoding='utf-8')
    for word in ('不是配准结果', 'oracle', 'seg35', 'seg4', '小六个', '大六个', '其余 23 个', '皮层', '白质', 'r1_3', 'r2_3', 'SDF', '只作对照', '消失', '体积', '峰值 RSS', '高置信', 'pin-fstar', '逐人', '部分抵消', '偏乐观', '减三分之一就够', '不够'):
        assert word in text, word


def test_the_probe_never_overwrites(built, neutral):
    results = neutral/'results'
    root = results/RUN.ROOT_NAME
    (root/'analysis').mkdir(parents=True)
    (root/'analysis'/RUN.PROBE_JSON).write_text('{}')
    with pytest.raises(SystemExit, match='never overwritten'):
        PROBE.run_probe(root, sources=RUN.Sources(results=results, stage=built.sources.stage), reader=posterior_from(built.maps))
    assert (root/'analysis'/RUN.PROBE_JSON).read_text() == '{}'
    with pytest.raises(SystemExit, match='writes only'):
        PROBE.run_probe(results/'d0_v1_a5', sources=RUN.Sources(results=results, stage=built.sources.stage), reader=posterior_from(built.maps))


def test_the_probes_grid_stays_below_one_half(built, neutral):
    results = neutral/'results'
    with pytest.raises(SystemExit, match='grid'):
        PROBE.run_probe(results/RUN.ROOT_NAME, sources=RUN.Sources(results=results, stage=built.sources.stage), reader=posterior_from(built.maps), f_grid=(.1, .5))


@pytest.mark.parametrize('target', (.1, 1/3, .55, 2/3, .98))
def test_the_bisection_finds_the_first_f_that_reaches_the_target(target):
    curve = lambda f: min(2*f, 1.)**.5                                            # noqa: E731
    calls = []

    def reduction(f):
        calls.append(f)
        return curve(f)
    grid = [(f, curve(f)) for f in PROBE.F_GRID]
    row = PROBE.find_fstar(target, reduction, grid)
    assert 0 < row['f'] < .5 and round(row['f'], 4) == row['f'] and curve(row['f']) == pytest.approx(target, abs=2e-3) and len(row['path']) <= 12
    assert row['reduction'] == curve(row['f'])
    assert row['bracket'][0] <= row['f'] <= row['bracket'][1]+1e-4


def test_the_bisection_says_when_the_target_cannot_be_reached():
    row = PROBE.find_fstar(.9, lambda f: .5, [(f, .5) for f in PROBE.F_GRID])
    assert row['f'] is None and '不可达' in row['reason']


# ---- the pins ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def good_pins():
    return dict(schema=RUN.PINS_SCHEMA, oracle=True, arms=dict(r1_3=dict(f=.1, target=1/3), r2_3=dict(f=.2, target=2/3)), probe_sha256='x')


@pytest.mark.parametrize('change,word', (
    (lambda p: p['arms']['r1_3'].update(f=.5), r'\(0, 1/2\)'), (lambda p: p['arms']['r2_3'].update(f=.75), r'\(0, 1/2\)'), (lambda p: p['arms']['r1_3'].update(f=0.), r'\(0, 1/2\)'), (lambda p: p['arms']['r1_3'].update(f=None), r'\(0, 1/2\)'), (lambda p: p['arms']['r1_3'].update(f=True), r'\(0, 1/2\)'),
    (lambda p: p['arms']['r1_3'].update(f=float('nan')), r'\(0, 1/2\)'), (lambda p: p['arms']['r1_3'].update(target=.3), 'target'), (lambda p: p['arms'].pop('r2_3'), 'expected'), (lambda p: p.update(schema='x'), 'schema'), (lambda p: p.update(oracle=False), 'oracle'), (lambda p: p['arms']['r1_3'].update(f=.3), 'below')))
def test_wrong_pins_are_refused(change, word):
    pins = good_pins()
    assert RUN.pin_value_problems(pins) == []
    change(pins)
    assert any(re.search(word, p) for p in RUN.pin_value_problems(pins)), RUN.pin_value_problems(pins)


def test_the_pins_written_are_those_of_the_probe(built):
    pins = pins_of(built)
    assert RUN.pin_value_problems(pins) == [] and pins['pinned_by'] == 'test' and pins['probe_sha256'] == sha(built.root/'analysis'/RUN.PROBE_JSON) and pins['tolerance'] == .03 == RUN.TARGET_TOL
    for arm in RUN.TARGETS:
        assert pins['arms'][arm]['f'] == built.probe['fstar'][arm]['f'] and pins['arms'][arm]['reduction_seg35_in_probe'] == built.probe['fstar'][arm]['reduction']
    assert RUN.fstar_problems(built.root, built.sources) == []


def test_pins_are_add_only(built):
    with pytest.raises(SystemExit, match='add only'):
        RUN.pin_fstar(built.root, 'again', sources=built.sources)
    assert pins_of(built)['pinned_by'] == 'test'


def test_pins_are_refused_before_they_are_written_when_the_probe_found_no_f_star_or_a_bad_one(built, neutral):
    results = neutral/'results'
    root = results/RUN.ROOT_NAME
    (root/'analysis').mkdir(parents=True)
    sources = RUN.Sources(results=results, stage=built.sources.stage, pins=neutral/'pins.json')
    probe = json.loads((built.root/'analysis'/RUN.PROBE_JSON).read_text())
    probe['fstar']['r2_3'] = dict(target=2/3, f=None, reason='不可达：made up')
    write_json(root/'analysis'/RUN.PROBE_JSON, probe)
    with pytest.raises(SystemExit, match='不可达'):
        RUN.pin_fstar(root, 'x', sources=sources)
    assert not sources.pins.exists()
    probe = json.loads((built.root/'analysis'/RUN.PROBE_JSON).read_text())
    probe['fstar']['r1_3']['reduction'] = 1/3+.031
    write_json(root/'analysis'/RUN.PROBE_JSON, probe)
    with pytest.raises(SystemExit, match='off the target'):
        RUN.pin_fstar(root, 'x', sources=sources)
    assert not sources.pins.exists()


def test_a_pin_that_is_not_the_f_star_of_the_probe_is_refused(built, neutral):
    root, sources, pins = pin_copy(built, neutral)
    probe = json.loads((root/'analysis'/RUN.PROBE_JSON).read_text())
    star = probe['fstar']['r1_3']['f']
    probe['fstar']['r1_3']['f'] = round(star+.0001, 4)                            # the row of the pinned f stays, within the tolerance; only the f* differs
    write_json(root/'analysis'/RUN.PROBE_JSON, probe)
    pins['probe_sha256'] = sha(root/'analysis'/RUN.PROBE_JSON)
    write_json(sources.pins, pins)
    bad = RUN.fstar_problems(root, sources)
    assert len(bad) == 1 and 'is not the f* of the probe' in bad[0]


def pin_copy(built, neutral, **changes):
    """A results tree of its own with the probe; pins as changed."""
    results = neutral/'results'
    root = results/RUN.ROOT_NAME
    (root/'analysis').mkdir(parents=True)
    shutil.copyfile(built.root/'analysis'/RUN.PROBE_JSON, root/'analysis'/RUN.PROBE_JSON)
    sources = RUN.Sources(results=results, stage=built.sources.stage, pins=neutral/'pins.json')
    pins = pins_of(built)
    pins.update(changes)
    write_json(sources.pins, pins)
    return root, sources, pins


def test_fstar_problems_each_way(built, neutral):
    root, sources, pins = pin_copy(built, neutral)
    assert RUN.fstar_problems(root, sources) == []
    pins['probe_sha256'] = 'x'
    write_json(sources.pins, pins)
    assert any('probe_sha256' in p for p in RUN.fstar_problems(root, sources))
    pins['probe_sha256'] = sha(root/'analysis'/RUN.PROBE_JSON)
    pins['arms']['r1_3']['f'] = .3333                                            # no row of the probe at this f
    write_json(sources.pins, pins)
    assert any('no row' in p for p in RUN.fstar_problems(root, sources))
    pins['arms']['r1_3']['f'] = PROBE.F_GRID[-2]                                  # a row far from the target
    pins['arms']['r2_3']['f'] = PROBE.F_GRID[-1]
    write_json(sources.pins, pins)
    assert any('off the target' in p for p in RUN.fstar_problems(root, sources))
    (root/'analysis'/RUN.PROBE_JSON).unlink()
    assert any('探针的结果不存在' in p for p in RUN.fstar_problems(root, sources))
    Path(sources.pins).unlink()
    assert any('f* 未钉' in p for p in RUN.fstar_problems(root, sources))


def test_a_pin_off_the_target_on_the_real_maps_stops_the_build_and_writes_nothing(built, neutral):
    results = neutral/'results'
    shutil.copytree(built.sources.results/'d0_v1_a5', results/'d0_v1_a5')
    sources = RUN.Sources(results=results, stage=built.sources.stage, pins=neutral/'pins.json')
    pins = pins_of(built)
    pins['arms']['r1_3']['f'] = .001
    write_json(sources.pins, pins)
    root = results/RUN.ROOT_NAME
    with pytest.raises(SystemExit, match='off the target'):
        RUN.build_all(root, sources=sources, reader=posterior_from(built.maps), log=lambda text: None)
    assert not (root/'labels').exists() and not (root/'inputs').exists()


def test_a_build_without_pins_stops(built, neutral):
    results = neutral/'results'
    sources = RUN.Sources(results=results, stage=built.sources.stage, pins=neutral/'nowhere.json')
    with pytest.raises(SystemExit, match='未钉'):
        RUN.build_all(results/RUN.ROOT_NAME, sources=sources, reader=posterior_from(built.maps))


# ---- the firewall -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_only_the_first_5_pairs_and_their_6_subjects():
    assert tuple(RUN.short_name(p) for p in RUN.PAIRS) == BRIEF_PAIRS and len(RUN.SUBJECTS) == 6
    assert {s for p in RUN.PAIRS for s in p.split('__')} == set(RUN.SUBJECTS)
    from bayesreg.io.guard import check_pair_subjects, load_dev_allowlist
    allowlist = load_dev_allowlist(ROOT/'configs/bayesreg/dev_subjects.json')
    for pair in RUN.PAIRS:
        check_pair_subjects(*pair.split('__'), allowlist)
    for name in ('train_subjects.txt', 'holdout_B_subjects.txt'):
        listed = set((ROOT/'splits/bayesreg_v1'/name).read_text().split())
        assert listed and not set(RUN.SUBJECTS) & listed, name
    sources = json.loads((ROOT/'configs/bayesreg/dev_subjects.json').read_text())['sources']
    forbidden = [s for s in sources if s.get('never_for_training') or s['name'] == 'bayesreg_train_v1']
    assert {s['name'] for s in forbidden} == {'stage_T0_tune_v1', 'bayesreg_train_v1'}                # T0 is on the allowlist: check_pair_subjects does not see it (review minor 7)
    for source in forbidden:
        assert source['subjects'] and not set(RUN.SUBJECTS) & set(source['subjects']), source['name']


@pytest.mark.parametrize('subject', ('OASIS_OAS1_0437_MR1', 'OASIS_OAS1_0444_MR1', 'OASIS_OAS1_0001_MR1', 'S37', '', '../OASIS_OAS1_0438_MR1'))
def test_other_subjects_are_refused(subject):
    with pytest.raises(SystemExit):
        RUN.require_subject(subject)
    with pytest.raises(SystemExit):
        RUN.label_path(RUN.DEFAULT_ROOT, subject, 'r1_3')
    with pytest.raises(SystemExit):
        RUN.label_tag('r1_3', subject)


@pytest.mark.parametrize('pair', ('OASIS_OAS1_0443_MR1__OASIS_OAS1_0444_MR1', 'OASIS_OAS1_0020_MR1__OASIS_OAS1_0021_MR1', 'S37__S38', '../x'))
def test_other_pairs_are_refused_before_any_file_is_opened(pair, monkeypatch):
    monkeypatch.setattr('builtins.open', lambda *a, **k: pytest.fail('a file was opened'))
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('an array was opened'))
    for call in (lambda: RUN.read_stage_pair(pair), lambda: RUN.read_images(pair), lambda: RUN.subjects_of(pair), lambda: RUN.entry_dir('/x', 'r1_3', pair), lambda: RUN.run_dir('/x', 'r2_3', pair), lambda: RUN.prepare_argv('r1_3', pair, '/x')):
        with pytest.raises(SystemExit):
            call()


def test_the_arm_of_one_subject_has_only_its_two_pairs():
    assert RUN.ARMS['f0440']['pairs'] == (RUN.PAIRS[1], RUN.PAIRS[2]) and [RUN.short_name(p) for p in RUN.ARMS['f0440']['pairs']] == ['0439-0440', '0440-0441']
    for pair in (RUN.PAIRS[0], RUN.PAIRS[3], RUN.PAIRS[4]):
        with pytest.raises(SystemExit):
            RUN.require_cell('f0440', pair)
    with pytest.raises(SystemExit):
        RUN.require_cell('r3_3', RUN.PAIRS[0])
    assert [RUN.label_tag('f0440', s) for s in RUN.SUBJECTS] == ['pred', 'pred', 'truth', 'pred', 'pred', 'pred']
    assert {RUN.label_tag('r1_3', s) for s in RUN.SUBJECTS} == {'r1_3'} and {RUN.label_tag('r2_3', s) for s in RUN.SUBJECTS} == {'r2_3'}                # one f for all subjects


def test_the_control_has_only_its_pair():
    assert RUN.require_cell(RUN.CONTROL_ARM, RUN.CONTROL_PAIR) == (RUN.CONTROL_ARM, RUN.PAIRS[0])
    with pytest.raises(SystemExit):
        RUN.require_cell(RUN.CONTROL_ARM, RUN.PAIRS[1])
    assert RUN.CONTROL_ARM not in RUN.ARMS and all(arm != RUN.CONTROL_ARM for arm, _ in RUN.plan_cells())
    assert RUN.label_tag(RUN.CONTROL_ARM, RUN.SUBJECTS[0]) == 'pred' == RUN.label_tag(RUN.CONTROL_ARM, RUN.SUBJECTS[1])


def test_the_plan():
    cells = RUN.plan_cells()
    assert len(cells) == 12 and [a for a, _ in cells] == ['r1_3']*5+['r2_3']*5+['f0440']*2
    assert [p for _, p in cells[:5]] == [p for _, p in cells[5:10]] == list(RUN.PAIRS)
    assert RUN.READING_ARMS == ('r1_3', 'r2_3') and RUN.DESCRIPTION_ARMS == ('f0440',) and RUN.TARGETS == dict(r1_3=1/3, r2_3=2/3) and RUN.TARGET_TOL == .03
    assert RUN.plan_cells(RUN.READING_ARMS) == cells[:10] and RUN.plan_cells(RUN.DESCRIPTION_ARMS) == cells[10:]


@pytest.mark.parametrize('root', ('/home/rmcfz/seq_vp_tag_results/seed_v1', '/home/rmcfz/seq_vp_tag_results/d0_v1_a5', '/home/rmcfz/seq_vp_tag_results/segvoi_v2', '/tmp/segvoi_v1', str(ROOT/'segvoi_v1')))
def test_other_roots_are_refused(root):
    assert RUN.root_problem(root)
    with pytest.raises(SystemExit):
        RUN.label_path(root, RUN.SUBJECTS[0], 'r1_3')


def test_a_root_inside_the_repository_is_refused():
    assert 'repository' in RUN.root_problem(ROOT/'segvoi_v1', ROOT)
    assert RUN.root_problem(RUN.DEFAULT_ROOT) is None and str(RUN.DEFAULT_ROOT) == '/home/rmcfz/seq_vp_tag_results/segvoi_v1'


def test_the_provenance_is_oracle_and_the_method_refuses_it(built):
    from bayesreg.io.pair import FORBIDDEN_ID, ProvenanceRejected, load_pair, validate_segmentation_provenance
    assert RUN.side_provenance('r1_3') == 'oracle_postmix_r1_3' and RUN.side_provenance('truth') == 'oracle_postmix_truth'
    assert {spec['provenance'] for spec in RUN.ARMS.values()} == {'oracle_postmix_r1_3', 'oracle_postmix_r2_3', 'oracle_postmix_f0440'}
    pin = load_pair(built.sources.stage/RUN.PAIRS[0]/'pair.npz', allowlist=core.ALLOWLIST)
    for spec in RUN.ARMS.values():
        assert FORBIDDEN_ID.search(spec['provenance'])
        with pytest.raises(ProvenanceRejected):
            validate_segmentation_provenance(dict(layer='tissue', segmenter_id=spec['provenance'], model_ids=['x'], test_labels_seen=False, trained_with_labels=False), pin, side='fixed', layer='tissue')


def test_the_truth_members_are_named_in_one_place_of_the_entry_path():
    for function in (RUN.run_entry, RUN.arm_labels, RUN.read_labels, RUN.images_equal_inputs, RUN.read_images, RUN.read_subject_posterior):
        source = inspect.getsource(function)
        assert 'seg35' not in source and 'seg4' not in source and 'read_stage_pair' not in source, function.__name__
    text = (ROOT/'experiments/sv/run_sv.py').read_text()
    assert text.count('read_stage_pair(') == 3                                   # the definition, the subjects of the probe and the build, the inputs of the build
    assert inspect.getsource(RUN.read_subjects).count('read_stage_pair(') == 1 and inspect.getsource(RUN.build_inputs).count('read_stage_pair(') == 1
    assert 'seg35' not in (ROOT/'experiments/sv/postmix.py').read_text().split('MEASURES = ')[0]


def test_the_real_reader_keeps_the_checks_of_read_predicted(monkeypatch):
    """read_subject_posterior against the functions of make_inputs, faked: the order of mc0p_core.read_predicted, the alignment QC stops, the stack it returns is the one the structure layer gave."""
    mi = core.make_inputs()
    P = np.zeros((36, 2, 2, 2))
    calls = []

    def record(name, value):
        def call(*args, **kwargs):
            calls.append(name)
            return value
        return call
    monkeypatch.setattr(mi, 'a4_admission', record('admission', dict(ok=1)))
    monkeypatch.setattr(mi, 'pair_structure_layer', record('layer', SimpleNamespace(record=lambda: dict(layer=1))))
    monkeypatch.setattr(mi, 'unet_posterior', record('posterior', (P, dict(segmenter_id=mc.SEGMENTER_ID, policy=mc.POLICY), np.zeros((2, 2, 2)))))
    monkeypatch.setattr(mi, 'support_dice_record', record('support', dict(dice=.8)))
    monkeypatch.setattr(mi, 'atropos_wm', record('atropos', (np.zeros((2, 2, 2)), dict(wm=1))))
    monkeypatch.setattr(mi, 'alignment_qc', record('qc', dict(passed=True, dice=.9, threshold=.8)))
    pin = SimpleNamespace(fixed_subject=RUN.SUBJECTS[0], moving_subject=RUN.SUBJECTS[1], fixed=np.zeros((2, 2, 2)), moving=np.zeros((2, 2, 2)), provenance=dict(label_firewall='x', keys_read=['fixed', 'moving']))
    got, rec = RUN.read_subject_posterior(pin, 'fixed')
    assert got is P and calls == ['admission', 'layer', 'posterior', 'support', 'atropos', 'qc'] and rec['subject'] == RUN.SUBJECTS[0] and rec['alignment_qc']['passed'] is True and rec['segmenter_id'] == mc.SEGMENTER_ID
    order = inspect.getsource(mc.read_predicted)
    assert order.index('a4_admission') < order.index('pair_structure_layer') < order.index('unet_posterior') < order.index('support_dice_record') < order.index('atropos_wm') < order.index('alignment_qc')
    monkeypatch.setattr(mi, 'alignment_qc', record('qc', dict(passed=False, dice=.5, threshold=.8)))
    with pytest.raises(SystemExit, match='alignment QC'):
        RUN.read_subject_posterior(pin, 'moving')


def test_each_posterior_is_read_once_so_one_stack_at_most_is_in_memory(built):
    reads = []

    def read(pin, side):
        reads.append(getattr(pin, side+'_subject'))
        return posterior_from(built.maps)(pin, side)
    held = RUN.read_subjects(built.sources, read, lambda text: None)
    assert reads == list(RUN.SUBJECTS) == list(held)
    assert not hasattr(held[RUN.SUBJECTS[0]]['reduced'], 'P')


def test_a_subject_whose_two_pairs_disagree_stops(built, neutral):
    stage = neutral/'stage'
    shutil.copytree(built.sources.stage, stage)
    d = stage/RUN.PAIRS[2]
    with np.load(d/'pair.npz') as z:
        arrays = dict(z)
    arrays['fixed_seg35'] = arrays['fixed_seg35'].copy()
    arrays['fixed_seg35'][0, 0, 0] = 35                                          # subject 0440 in its second pair
    np.savez_compressed(d/'pair.npz', **{k: arrays[k] for k in core.make_inputs().KEY_ORDER})
    manifest = json.loads((d/'manifest.json').read_text())
    manifest['pair_sha256'] = sha(d/'pair.npz')
    write_json(d/'manifest.json', manifest)
    with pytest.raises(SystemExit, match='disagree'):
        RUN.read_subjects(RUN.Sources(results=neutral/'results', stage=stage), posterior_from(built.maps), lambda text: None)


# ---- the build ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_the_labels_are_written_only_under_the_isolated_root(built):
    files = [p for p in built.base.rglob('*') if p.is_file() and p.name.startswith('L_')]
    assert files and all(built.root/'labels' in p.parents for p in files)
    assert sorted(p.name for p in (built.root/'labels').iterdir()) == sorted(list(RUN.SUBJECTS)+['build_record.json'])
    assert sorted(p.name for p in (built.root/'labels'/'OASIS_OAS1_0440_MR1').iterdir()) == ['L_r1_3.npz', 'L_r2_3.npz', 'L_truth.npz']
    assert sorted(p.name for p in (built.root/'labels'/'OASIS_OAS1_0439_MR1').iterdir()) == ['L_pred.npz', 'L_r1_3.npz', 'L_r2_3.npz']
    assert sorted(p.name for p in (built.root/'labels'/'OASIS_OAS1_0438_MR1').iterdir()) == ['L_pred.npz', 'L_r1_3.npz', 'L_r2_3.npz']          # the control
    assert sorted(p.name for p in (built.root/'labels'/'OASIS_OAS1_0443_MR1').iterdir()) == ['L_r1_3.npz', 'L_r2_3.npz']


def test_every_record_of_the_build_says_oracle_and_its_provenance(built):
    record = json.loads((built.root/'labels/build_record.json').read_text())
    assert record['oracle'] is True and 'NOT_in_any_result_table' in record['role'] and 'smallest label' in record['tie_rule']
    for subject in RUN.SUBJECTS:
        rows = record['subjects'][subject]
        assert rows['oracle'] is True and set(rows['labels']) == {'pred', 'r1_3', 'r2_3', 'truth'} and 'smallest label' in rows['tie_rule']
        for the_tag, row in rows['labels'].items():
            assert row['provenance'] == 'oracle_postmix_'+the_tag and row['oracle'] is True and {'seg35', 'seg4', 'small_six', 'big_six', 'other_23', 'cortex', 'white_matter', 'per_label'} <= set(row['dice_against_truth'])
            assert len(row['voxels']) == 36 and len(row['volume_vs_truth']) == 36 and isinstance(row['vanished'], list)
    for arm, spec in RUN.ARMS.items():
        inputs = json.loads((RUN.inputs_root(arm, built.root)/'d0_inputs.json').read_text())
        assert inputs['oracle'] is True and inputs['provenance'] == spec['provenance'] and inputs['arm'] == arm
        for pair in spec['pairs']:
            files = RUN.input_files(arm, pair, built.root)
            made = files[3]['preprocessing_d0']
            assert made['oracle'] is True and made['provenance'] == spec['provenance'] and 'oracle' in files[3]['preprocessing'] and made['role'] == RUN.ROLE


def test_the_build_records_the_actual_reduction_of_each_arm(built):
    record = json.loads((built.root/'labels/build_record.json').read_text())
    pins = pins_of(built)
    for arm, target in RUN.TARGETS.items():
        row = record['arms'][arm]
        assert row['f'] == pins['arms'][arm]['f'] and row['target'] == target and abs(row['reduction']['seg35']-target) <= RUN.TARGET_TOL and set(row['reduction']) == set(MIX.MEASURES) and set(row['reduction_per_subject']) == set(RUN.SUBJECTS)
        assert row['reduction']['seg35'] == pytest.approx(built.probe['fstar'][arm]['reduction'], abs=1e-12)           # the build's maps are the probe's maps
    assert record['arms']['f0440']['f'] is None and record['arms']['f0440']['reduction'] is None and record['arms']['f0440']['tags'][RUN.HARD_SUBJECT] == 'truth'
    assert 'pins_sha256' in record


def test_the_segmentation_level_dice_runs_from_the_prediction_to_one(built):
    record = json.loads((built.root/'labels/build_record.json').read_text())
    for subject in RUN.SUBJECTS:
        rows = record['subjects'][subject]['labels']
        for key in ('seg35', 'seg4'):
            assert rows['truth']['dice_against_truth'][key] == 1.
            assert rows['pred']['dice_against_truth'][key] < rows['r1_3']['dice_against_truth'][key] < rows['r2_3']['dice_against_truth'][key] < 1.


def stored(root, subject, the_tag):
    return RUN.read_labels(root, subject, the_tag)[0]


@pytest.mark.parametrize('subject_index', range(6))
def test_the_stored_labels_are_the_formula_with_one_f_for_all(built, subject_index):
    subject = RUN.SUBJECTS[subject_index]
    P, truth = built.maps[subject]['P'], built.maps[subject]['truth']
    for arm in RUN.TARGETS:
        assert stored(built.root, subject, arm).tobytes() == MIX.mix_direct(P, truth, pins_of(built)['arms'][arm]['f']).tobytes()


def test_the_build_stops_when_a_map_is_not_its_source():
    reduced = MIX.reduce_posterior(synthetic_posterior(SDF.synthetic_labels(SHAPE, 1), SDF.synthetic_labels(SHAPE, 1, shift=1.), 3), SDF.synthetic_labels(SHAPE, 1, shift=1.))
    good = dict(pred=reduced.pred.copy(), truth=reduced.truth.copy())
    RUN.check_endpoints('s', good, reduced)
    for key, word in (('pred', 'prediction'), ('truth', 'truth')):
        bad = dict(good)
        bad[key] = good[key].copy()
        bad[key][0, 0, 0] = (bad[key][0, 0, 0]+1) % 36
        with pytest.raises(SystemExit, match=word):
            RUN.check_endpoints('s', bad, reduced)


@pytest.mark.parametrize('pair', (RUN.PAIRS[1], RUN.PAIRS[2]))
def test_in_f0440_only_0440_is_the_truth_and_the_other_subject_is_the_prediction_bit_for_bit(built, pair):
    """The lead's test: entry channels and colouring both come from these maps."""
    from bayesreg.io.pair import load_pair
    pin = load_pair(built.sources.stage/pair/'pair.npz', allowlist=core.ALLOWLIST)
    predicted = RUN.arm_labels(built.root, 'f0440', pin)
    assert predicted.record['oracle'] is True and predicted.record['provenance'] == 'oracle_postmix_f0440'
    seen = 0
    table = core.load_table()
    q0 = mc.q0_channels(pin, predicted, table)
    with np.load(RUN.input_files('f0440', pair, built.root)[0]/'pair.npz') as z:
        for k, side in enumerate(('fixed', 'moving')):
            subject = getattr(pin, side+'_subject')
            truth, pred = built.maps[subject]['truth'], built.maps[subject]['pred']
            labels = predicted.labels[side]
            assert truth.tobytes() != pred.tobytes()
            if subject == RUN.HARD_SUBJECT:
                seen += 1
                assert labels.tobytes() == truth.tobytes() and labels.tobytes() != pred.tobytes()
                assert predicted.record['sides'][side]['provenance'] == 'oracle_postmix_truth'
            else:
                assert labels.tobytes() == pred.tobytes() and labels.tobytes() != truth.tobytes()
                assert predicted.record['sides'][side]['provenance'] == 'oracle_postmix_pred'
            assert z[side].tobytes() == q0[k][0].tobytes()                       # the colouring is of the same labels
            source = 'A5' if subject == RUN.HARD_SUBJECT else 'A5P'
            with np.load(built.sources.results/'d0_v1_a5/inputs'/source/'stage_L2R_val_v1'/pair/'pair.npz') as endpoint:
                assert z[side].tobytes() == endpoint[side].tobytes(), source      # the colouring of the truth is A5's image, that of the prediction A5P's
    assert seen == 1


def test_the_inputs_keep_the_truth_labels_for_the_evaluation_and_the_order(built):
    mi = core.make_inputs()
    for arm, pair in RUN.plan_cells():
        d = RUN.input_files(arm, pair, built.root)[0]
        with np.load(d/'pair.npz') as z, np.load(built.sources.stage/pair/'pair.npz') as original:
            assert tuple(z.files) == mi.KEY_ORDER
            for key in ('fixed_seg4', 'fixed_seg35', 'moving_seg4', 'moving_seg35', 'initial_u'):
                assert z[key].tobytes() == original[key].tobytes()
            assert z['fixed'].tobytes() != original['fixed'].tobytes()


def test_the_inputs_record_passes_the_check_of_the_queues(built):
    for arm in RUN.ARMS:
        assert seed.b1_spec.check_one_input(RUN.b1_entry(arm, built.root), ROOT) == []


def test_a_build_never_overwrites(built, neutral):
    results = neutral/'results'
    root = results/RUN.ROOT_NAME
    (root/'labels').mkdir(parents=True)
    sources = RUN.Sources(results=results, stage=built.sources.stage, pins=built.sources.pins)
    with pytest.raises(SystemExit, match='never overwritten'):
        RUN.build_all(root, sources=sources)
    with pytest.raises(SystemExit, match='writes only'):
        RUN.build_all(results/'d0_v1_a5', sources=sources)


def test_a_build_stops_when_the_colouring_is_not_a5_and_a5p(built, neutral):
    results = neutral/'results'
    shutil.copytree(built.sources.results/'d0_v1_a5', results/'d0_v1_a5')
    path = results/'d0_v1_a5/inputs/A5/stage_L2R_val_v1'/RUN.PAIRS[2]/'pair.npz'
    with np.load(path) as z:
        images = dict(z)
    images['moving'] = images['moving']*(1+1e-12)
    np.savez(path, **images)
    sources = RUN.Sources(results=results, stage=built.sources.stage, pins=built.sources.pins)
    with pytest.raises(SystemExit, match='bit for bit'):
        RUN.build_all(results/RUN.ROOT_NAME, sources=sources, reader=posterior_from(built.maps), log=lambda text: None)


def test_a_tampered_label_file_is_refused(built, neutral):
    root = neutral/RUN.ROOT_NAME
    shutil.copytree(built.root/'labels', root/'labels')
    path = root/'labels'/RUN.SUBJECTS[0]/'L_r1_3.npz'
    assert RUN.read_labels(root, RUN.SUBJECTS[0], 'r1_3')[0].shape == SHAPE
    np.savez_compressed(path, labels=np.zeros(SHAPE, dtype=np.int16))
    with pytest.raises(SystemExit, match='sha256'):
        RUN.read_labels(root, RUN.SUBJECTS[0], 'r1_3')
    with pytest.raises(SystemExit, match='not a finished record'):
        RUN.read_labels(root, RUN.SUBJECTS[5], 'pred')                           # computed and recorded, not stored: 0443 has no arm of the prediction


def test_the_channels_are_those_of_q2_on_the_arms_labels(built):
    """mc0p_core.prepare_stacks, the function of Q2 long, takes the arm's maps as it takes the predicted maps."""
    from bayesreg.io.pair import load_pair
    pair = RUN.PAIRS[1]
    pin = load_pair(built.sources.stage/pair/'pair.npz', allowlist=core.ALLOWLIST)
    predicted = RUN.arm_labels(built.root, 'f0440', pin)
    table = core.load_table()
    q0 = mc.q0_channels(pin, predicted, table)
    assert RUN.images_equal_inputs('f0440', pair, q0, built.root) is True
    assert RUN.images_equal_inputs('r1_3', pair, q0, built.root) is False
    stacks, record = mc.prepare_stacks(RUN.ENTRY_FORMULA, pin, predicted, q0, 'cpu', workers=1)
    assert stacks[0].shape == (35,)+SHAPE and record['D_factor'] == 1.
    direct = core.channels('P2', pin.fixed, predicted.labels['fixed'])*record['channel_multiplier']
    assert np.array_equal(stacks[0], direct)
    assert RUN.channel_problems('f0440', pair, predicted.record) == []


def test_the_control_takes_the_prediction_and_compares_with_a5p(built):
    from bayesreg.io.pair import load_pair
    pin = load_pair(built.sources.stage/RUN.CONTROL_PAIR/'pair.npz', allowlist=core.ALLOWLIST)
    predicted = RUN.arm_labels(built.root, RUN.CONTROL_ARM, pin)
    for side in ('fixed', 'moving'):
        assert predicted.labels[side].tobytes() == built.maps[getattr(pin, side+'_subject')]['pred'].tobytes()
    q0 = mc.q0_channels(pin, predicted, core.load_table())
    assert RUN.images_equal_inputs(RUN.CONTROL_ARM, RUN.CONTROL_PAIR, q0, built.root, built.sources.results) is True
    assert predicted.record['provenance'] == 'oracle_postmix_f0control' and predicted.record['oracle'] is True


def test_the_entry_tier_is_q2_long():
    assert RUN.ENTRY_FORMULA == 'Q2' and RUN.TIER == 'long' and mc.ARMS['Q2']['k'] == 35 and mc.ARMS['Q2']['base'] == 'P2' and mc.ARMS['Q2']['D_factor'] == 1. and 'long' in mc.ARMS['Q2']['tiers']
    config = core.tier_config('long')
    assert {k: list(v) if isinstance(v, tuple) else v for k, v in config.items() if k in seed.LONG_TIER} == seed.LONG_TIER


def test_the_sources_are_the_ones_the_real_readers_use():
    sources = RUN.Sources()
    assert sources.pred_root == Path(mc.default_unet().pred_root) and sources.seg_root == mc.SEG_ROOT and sources.stage == core.STAGE and sources.allowlist == core.ALLOWLIST and sources.results == seed.RESULTS and sources.pins == RUN.PINS_FILE
    assert mc.SEGMENTER_ID == 'unet35_v1_ens' and mc.POLICY == 'all_folds_mean'


# ---- the entry check ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def small_field(number=1):
    rng = np.random.default_rng(number)
    u = np.zeros(SHAPE+(3,))
    u[1:-1, 1:-1, 1:-1] = rng.normal(size=(SHAPE[0]-2, SHAPE[1]-2, SHAPE[2]-2, 3))*.1
    return u


def make_entry(root, arm, pair, /, *, u=None, status='done', evaluated=True, **overrides):
    """What run_entry and evaluate_entries leave, without running them; record changes through overrides."""
    d = RUN.entry_dir(root, arm, pair)
    d.mkdir(parents=True, exist_ok=True)
    u = small_field(RUN.plan_cells().index((arm, pair))+1) if u is None else u
    np.savez(d/'field.npz', u=u)
    made = RUN.input_files(arm, pair, root)[3]['preprocessing_d0']
    sides = {}
    for side, subject in zip(('fixed', 'moving'), RUN.subjects_of(pair)):
        the_tag = RUN.label_tag(arm, subject)
        sides[side] = dict(subject=subject, tag=the_tag, provenance=RUN.side_provenance(the_tag))
    record = dict(schema=RUN.ENTRY_SCHEMA, run_id=RUN.EntryCell(arm, RUN.TIER, pair).run_id, arm=arm, tier=RUN.TIER, formula=RUN.ENTRY_FORMULA, pair=pair, k=35, oracle=True, provenance=RUN.ARMS[arm]['provenance'], status=status, reason=None, config=core.tier_config(RUN.TIER), scale=dict(images_equal_sv_inputs=True),
                  channel_sources=dict(oracle=True, provenance=RUN.ARMS[arm]['provenance'], pair_keys_read=['fixed', 'moving', 'initial_u'], sides=sides), field=dict(file='field.npz', sha256=sha(d/'field.npz'), u_sha256=hashlib.sha256(u.tobytes()).hexdigest()),
                  data=dict(pair_sha256=made['source_pair_sha256'], manifest_sha256=made['source_manifest_sha256']), seconds_total=60., code_sha='x')
    record.update(overrides)
    write_json(d/'run.json', record)
    if evaluated:
        write_json(d/'metrics.json', dict(tapered_u_sha256=seed.boundary_taper_sha256(u, SHAPE), certificate=dict(accepted_alpha=1., n_infeasible_at_1=0), metrics=dict(seg35=.85, seg4=.89, sdlogj_official=.05), at_accepted_alpha=dict(alpha=1., seg35=.85, seg4=.89)))
    return d


@pytest.fixture
def copied(built, neutral):
    root = neutral/RUN.ROOT_NAME
    root.mkdir()
    shutil.copytree(built.root/'inputs', root/'inputs')
    return root


def entry_check(root, arm='r1_3', pair=None):
    pair = RUN.PAIRS[0] if pair is None else pair
    return RUN.check_entry(arm, pair, RUN.input_files(arm, pair, root)[3], root)


@pytest.mark.parametrize('arm,pair', ((('r1_3', RUN.PAIRS[0])), ('r2_3', RUN.PAIRS[4]), ('f0440', RUN.PAIRS[1])))
def test_a_good_entry_passes_and_says_oracle(copied, arm, pair):
    make_entry(copied, arm, pair)
    record = entry_check(copied, arm, pair)
    assert record['oracle'] is True and record['provenance'] == RUN.ARMS[arm]['provenance'] and record['charge_seconds'] == 47. and record['tapered_recorded_by_evaluation'] == record['tapered_sha256']
    assert set(seed.inject_run.ENTRY_KEYS) <= set(record)


@pytest.mark.parametrize('override,word', (
    (dict(oracle=False), 'oracle'), (dict(status='cuda_oom'), 'status'), (dict(arm='r2_3'), 'arm'), (dict(k=14), 'K'), (dict(schema='bayesreg.mc0p.run.v1'), 'schema'), (dict(run_id='x'), 'run_id'), (dict(scale=dict(images_equal_sv_inputs=False)), 'images_equal_sv_inputs'),
    (dict(config=dict(iterations=[50, 50, 100, 60])), 'long tier'), (dict(data=dict(pair_sha256='a'*64, manifest_sha256='b'*64)), 'original pair'), (dict(channel_sources=dict(oracle=True, provenance='oracle_postmix_r2_3', pair_keys_read=['fixed', 'moving'], sides={})), 'provenance'),
    (dict(channel_sources=None), 'channel_sources'), (dict(field=None), 'no field record')))
def test_a_wrong_entry_record_is_refused(copied, override, word):
    make_entry(copied, 'r1_3', RUN.PAIRS[0], **override)
    with pytest.raises(RUN.EntryError, match=word):
        entry_check(copied)


def test_the_marks_of_the_sides_are_checked(copied):
    pair = RUN.PAIRS[1]
    d = make_entry(copied, 'f0440', pair)
    record = json.loads((d/'run.json').read_text())
    record['channel_sources']['sides']['moving']['tag'] = 'pred'                 # 0440 as if it were its prediction
    write_json(d/'run.json', record)
    with pytest.raises(RUN.EntryError, match='OASIS_OAS1_0440_MR1'):
        entry_check(copied, 'f0440', pair)


def test_an_entry_without_files_or_without_its_evaluation_is_refused(copied):
    with pytest.raises(RUN.EntryMissing, match='入口不存在'):
        entry_check(copied)
    make_entry(copied, 'r1_3', RUN.PAIRS[0], evaluated=False)
    with pytest.raises(RUN.EntryError, match='without its evaluation'):
        entry_check(copied)


def test_sha256_shape_and_faces_of_the_field_are_checked(copied):
    d = make_entry(copied, 'r1_3', RUN.PAIRS[0])
    np.savez(d/'field.npz', u=small_field(99))
    with pytest.raises(RUN.EntryError, match='sha256'):
        entry_check(copied)
    u = np.zeros((4, 4, 4, 3))
    d = make_entry(copied, 'r1_3', RUN.PAIRS[0], u=u, evaluated=False)
    with pytest.raises(RUN.EntryError, match='shape'):
        entry_check(copied)
    u = small_field(1)
    u[0, 0, 0, 0] = 1.
    d = make_entry(copied, 'r1_3', RUN.PAIRS[0], u=u)
    with pytest.raises(RUN.EntryError, match='faces'):
        entry_check(copied)


def test_an_inputs_file_that_changed_is_refused(copied):
    d = RUN.input_files('r1_3', RUN.PAIRS[0], copied)[0]
    (d/'manifest.json').write_text((d/'manifest.json').read_text()+' ')
    with pytest.raises(RUN.EntryError, match='differ'):
        RUN.input_files('r1_3', RUN.PAIRS[0], copied)


# ---- the reading ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def made_up(root, third, two_thirds, hard=None, **changes):
    """{(arm, pair): record}: r1_3 and r2_3 at these means (the deviations add up to zero), the endpoints at their pins, f0440 at `hard` if given."""
    runs = {}
    for arm, value in (('r1_3', third), ('r2_3', two_thirds)):
        if value is None:
            continue
        for k, pair in enumerate(RUN.PAIRS):
            runs[(arm, pair)] = final_record(arm, pair, value+(k-2)*.001, inputs_root=RUN.inputs_root(arm, root), **changes)
    for arm, (seg35, seg4) in DEC.PINS.items():
        for pair in RUN.PAIRS:
            runs[(arm, pair)] = final_record(arm, pair, seg35, seg4)
    if hard is not None:
        for pair in RUN.ARMS['f0440']['pairs']:
            runs[('f0440', pair)] = final_record('f0440', pair, hard, inputs_root=RUN.inputs_root('f0440', root))
    return runs


@pytest.mark.parametrize('third,two_thirds,code,words', (
    (.87, .89, DEC.THIRD_ENOUGH, '减三分之一就够'), (.86, .87, DEC.TWO_THIRDS_NEEDED, '减三分之二才够'), (.85, .86, DEC.NOT_ENOUGH, '减三分之二也不够'), (.86, .8667, DEC.NOT_ENOUGH, '减三分之二也不够')))
def test_the_reading_table(third, two_thirds, code, words):
    result = DEC.decide(made_up('/r', third, two_thirds), '/r')
    assert result['reading_code'] == code and result['reading'].startswith(words) and result['reasons'] == []
    assert result['arms']['r1_3']['seg35'] == pytest.approx(third) and result['arms']['r2_3']['seg35'] == pytest.approx(two_thirds)


@pytest.mark.parametrize('third,two_thirds,qualified', ((.87, .89, True), (.86, .87, True), (.85, .86, False)))
def test_a_reading_of_enough_carries_the_direction_of_the_limit(third, two_thirds, qualified):
    reading = DEC.decide(made_up('/r', third, two_thirds), '/r')['reading']
    assert ('偏乐观' in reading) is qualified and ('打折' in reading) is qualified


def test_the_note_says_the_direction_of_the_limit_and_how_to_read_a_line_this_close():
    for word in ('低置信', '高置信', '部分抵消', '偏乐观', '减三分之一就够', '打这个折扣', '相对可靠', 'crossing', '噪声', '〔推断〕'):
        assert word in DEC.NOTE, word


def test_the_words_are_the_leads():
    assert DEC.READINGS == {DEC.THIRD_ENOUGH: '减三分之一就够', DEC.TWO_THIRDS_NEEDED: '减三分之二才够', DEC.NOT_ENOUGH: '减三分之二也不够', DEC.NOT_DECIDABLE: '不可判'}
    assert DEC.LINE == .8668 and DEC.TARGET_19 == .88 and DEC.OFFSET_PINNED == .0132 and round(DEC.TARGET_19-DEC.OFFSET_PINNED, 6) == DEC.LINE and DEC.SDLOGJ_MAX == .10


def test_the_reading_says_the_actual_reduction_and_both_dice():
    build = dict(arms=dict(r1_3=dict(f=.1234, reduction=dict(seg35=.34, seg4=.3)), r2_3=dict(f=.3456, reduction=dict(seg35=.65, seg4=.6))))
    result = DEC.decide(made_up('/r', .86, .87), '/r', build=build)
    for word in ('34.0%', '30.0%', '65.0%', '60.0%', '0.1234', '0.3456', 'seg35 0.8600', 'seg4 0.9000', 'seg35 0.8700', '线 0.8668'):
        assert word in result['reading'], word
    assert result['arms']['r1_3']['reduction_seg35'] == .34 and result['arms']['r2_3']['f'] == .3456


def test_not_decidable_still_says_the_means():
    runs = made_up('/r', .85, .86)
    runs[('r2_3', RUN.PAIRS[0])]['native_commits'] = 0
    reading = DEC.decide(runs, '/r')['reading']
    assert reading.startswith('不可判') and 'seg35 0.8500' in reading and '原生提交' in reading and '原因' in reading


def test_the_line_is_included_and_the_means_are_not_rounded(monkeypatch):
    monkeypatch.setattr(DEC, 'LINE', .875)                                       # a line the mean of five equal floats reaches exactly
    runs = made_up('/r', .86, .875)
    means = [runs[('r2_3', p)]['metrics'] for p in RUN.PAIRS]
    for m in means:
        m['seg35'] = .875
    assert DEC.decide(runs, '/r')['reading_code'] == DEC.TWO_THIRDS_NEEDED
    for m in means:
        m['seg35'] = .875-1e-6                                                   # rounds to the line, is below it
    assert DEC.decide(runs, '/r')['reading_code'] == DEC.NOT_ENOUGH


def test_a_mean_near_the_line_says_so_and_keeps_its_code():
    result = DEC.decide(made_up('/r', .86, .8673), '/r')
    assert result['reading_code'] == DEC.TWO_THIRDS_NEEDED and result['near_line'] == ['r2_3'] and '贴着线' in result['reading']
    assert '贴着线' not in DEC.decide(made_up('/r', .86, .89), '/r')['reading']


@pytest.mark.parametrize('change,word', (
    (dict(metrics=None), '无终态场'), (dict(native_commits=0), '原生提交'), (dict(native_commits=None), '原生提交'), (dict(classification={'class': 'handover_only', 'kind': 'k', 'reason': 'r'}), '分类'), (dict(certificate=dict(CERTIFIED, passed=False)), '证书未通过'),
    (dict(certificate=dict(CERTIFIED, final_npz_sha256_matches=False)), '证书未通过'), (dict(oracle=False), 'oracle'), (dict(run_labels_equal_truth=False), '标签副本'), (dict(seed=dict(passed=False, problems=['p'])), '入口核对')))
@pytest.mark.parametrize('arm', ('r1_3', 'r2_3'))
def test_a_premise_not_met_makes_it_not_decidable(arm, change, word):
    runs = made_up('/r', .85, .86)
    runs[(arm, RUN.PAIRS[2])].update(change)
    result = DEC.decide(runs, '/r')
    assert result['reading_code'] == DEC.NOT_DECIDABLE and any(word in reason for reason in result['reasons']), result['reasons']
    assert result['reading'].startswith('不可判')


def test_a_certificate_not_passed_is_no_final_field():
    runs = made_up('/r', .95, .95)
    runs[('r1_3', RUN.PAIRS[0])]['certificate'] = dict(CERTIFIED, passed=False)
    result = DEC.decide(runs, '/r')
    assert result['arms']['r1_3']['seg35'] is None and result['arms']['r1_3']['readable'] is False and result['reading_code'] == DEC.NOT_DECIDABLE


@pytest.mark.parametrize('third,two_thirds', ((.85, None), (None, .85), (None, .95), (None, None)))
def test_a_missing_arm_is_not_decidable(third, two_thirds):
    assert DEC.decide(made_up('/r', third, two_thirds), '/r')['reading_code'] == DEC.NOT_DECIDABLE


def test_the_first_arm_over_the_line_reads_alone_and_a_problem_of_the_second_is_said():
    result = DEC.decide(made_up('/r', .87, None), '/r')
    assert result['reading_code'] == DEC.THIRD_ENOUGH and '前提不满足' in result['reading']
    result = DEC.decide(made_up('/r', .87, .86), '/r')
    assert result['reading_code'] == DEC.THIRD_ENOUGH and '均值低于线' in result['reading']


def test_sdlogj_over_the_bar_is_not_decidable_and_at_the_bar_is():
    runs = made_up('/r', .85, .86)
    for pair in RUN.PAIRS:
        runs[('r1_3', pair)]['metrics']['sdlogj_official'] = .1001
    assert DEC.decide(runs, '/r')['reading_code'] == DEC.NOT_DECIDABLE
    for pair in RUN.PAIRS:
        runs[('r1_3', pair)]['metrics']['sdlogj_official'] = .10
    assert DEC.decide(runs, '/r')['reading_code'] == DEC.NOT_ENOUGH


def test_another_configuration_is_not_decidable():
    runs = made_up('/r', .85, .86)
    runs[('r1_3', RUN.PAIRS[0])]['seed']['arm']['charge_seconds'] = 0.
    assert any('配置' in r for r in DEC.decide(runs, '/r')['reasons'])
    runs = made_up('/r', .85, .86)
    runs[('r2_3', RUN.PAIRS[0])]['seed']['arm']['inputs_root'] = '/other'
    assert DEC.decide(runs, '/r')['reading_code'] == DEC.NOT_DECIDABLE


def test_an_endpoint_problem_is_not_decidable():
    result = DEC.decide(made_up('/r', .87, .89), '/r', ['S_P: made up'])
    assert result['reading_code'] == DEC.NOT_DECIDABLE and 'S_P: made up' in result['reading']


def test_the_pins_of_the_endpoints():
    assert DEC.PINS == dict(S_P=(.8430, .8984), S_T=(.9000, .9438))
    runs = made_up('/r', .85, .86)
    assert DEC.endpoint_problems(runs) == []
    runs[('S_P', RUN.PAIRS[0])]['metrics']['seg35'] += .01
    assert any('S_P' in p for p in DEC.endpoint_problems(runs))
    runs = made_up('/r', .85, .86)
    runs[('S_T', RUN.PAIRS[0])]['certificate'] = dict(CERTIFIED, passed=False)
    assert any('S_T' in p for p in DEC.endpoint_problems(runs))


@pytest.mark.parametrize('hard', (None, .99, .10))
def test_f0440_never_enters_the_reading(hard):
    result = DEC.decide(made_up('/r', .85, .86, hard=hard), '/r')
    assert result['reading_code'] == DEC.NOT_ENOUGH and set(result['arms']) == {'r1_3', 'r2_3'} and 'f0440' not in json.dumps(result)


def test_f0440_missing_or_broken_changes_no_reading():
    runs = made_up('/r', .86, .87, hard=.9)
    runs[('f0440', RUN.PAIRS[1])]['certificate'] = dict(CERTIFIED, passed=False)
    del runs[('f0440', RUN.PAIRS[2])]
    assert DEC.decide(runs, '/r')['reading_code'] == DEC.TWO_THIRDS_NEEDED


def test_the_slope_and_the_crossing_are_descriptions_of_the_table():
    table = [dict(name='0', reduction_seg35=0., segmentation_seg35=.92, final_seg35=.84, segmentation_seg4=.95, final_seg4=.90), dict(name='a', reduction_seg35=1/3, segmentation_seg35=.94, final_seg35=.86, segmentation_seg4=.96, final_seg4=.91),
             dict(name='b', reduction_seg35=2/3, segmentation_seg35=.97, final_seg35=.88, segmentation_seg4=.98, final_seg4=.93), dict(name='1', reduction_seg35=1., segmentation_seg35=1., final_seg35=.90, segmentation_seg4=1., final_seg4=.94)]
    row = DEC.slopes(table, 'seg35')
    assert row['points'] == 4 and row['steps'][0]['slope'] == pytest.approx(1.) and row['steps'][1]['slope'] == pytest.approx(.02/.03) and row['least_squares'] > 0
    cross = DEC.crossing(table)
    t = (.8668-.86)/.02
    assert cross['reduction_seg35'] == pytest.approx(1/3+(1/3)*t) and cross['segmentation_seg35'] == pytest.approx(.94+.03*t) and '模型' in cross['role']
    assert DEC.crossing([dict(reduction_seg35=0., segmentation_seg35=.9, final_seg35=.8)])['reduction_seg35'] is None
    assert DEC.slopes([dict(name='0', segmentation_seg35=None, final_seg35=None)], 'seg35')['least_squares'] is None


def test_a_step_of_the_slope_with_no_real_denominator_is_not_computed():
    table = [dict(name='0', segmentation_seg35=.9200, final_seg35=.84), dict(name='a', segmentation_seg35=.9201, final_seg35=.86), dict(name='1', segmentation_seg35=1., final_seg35=.90)]
    row = DEC.slopes(table, 'seg35')
    assert row['steps'][0]['slope'] is None and row['steps'][1]['slope'] is not None


# ---- the written analysis --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def write_arm_runs(root, third, two_thirds, hard=None):
    for (arm, pair), doc in made_up(root, third, two_thirds, hard=hard).items():
        if arm in RUN.ARMS:
            write_json(RUN.run_dir(root, arm, pair)/'metrics.json', doc)


def test_the_written_analysis_has_the_endpoints_in_one_table_marked_and_both_dice(built, neutral):
    root = neutral/RUN.ROOT_NAME
    shutil.copytree(built.root/'labels', root/'labels')
    write_arm_runs(root, .86, .88, hard=.85)
    result = DEC.write_analysis(root, results=built.sources.results)
    assert result['reading_code'] == DEC.TWO_THIRDS_NEEDED and result['oracle'] is True and result['reference_problems'] == []
    assert [row['arm'] for row in result['table']] == ['S_P', 'r1_3', 'r2_3', 'S_T']
    assert result['table'][0]['final_seg35'] == pytest.approx(DEC.PINS['S_P'][0]) and result['table'][3]['final_seg35'] == pytest.approx(DEC.PINS['S_T'][0])
    assert all(row['segmentation_seg35'] is not None and row['segmentation_seg4'] is not None for row in result['table']) and result['table'][3]['segmentation_seg35'] == 1.
    assert '另一时段' in result['table'][0]['source'] and '另一时段' in result['table'][3]['source'] and '另一时段' not in result['table'][1]['source']
    assert result['table'][0]['reduction_seg35'] == 0. and result['table'][3]['reduction_seg35'] == 1. and abs(result['table'][1]['reduction_seg35']-1/3) <= .03 and abs(result['table'][2]['reduction_seg35']-2/3) <= .03
    summary = (root/'analysis/summary_sv.md').read_text(encoding='utf-8')
    for word in ('seg35', 'seg4', '不同时段', 'oracle', '不进结果表', '开发集', '不进读数', '分割级', '入口', '小六个', '大六个', '其余 23 个', '皮层', '白质', '斜率', 'f0440', '〔模型〕', '实际减少', '标签消失', '最小体积比', '高置信'):
        assert word in summary, word
    saved = json.loads((root/'analysis/decision_sv.json').read_text())
    assert saved['reading_code'] == DEC.TWO_THIRDS_NEEDED and set(saved['arms']) == {'r1_3', 'r2_3'}
    again = DEC.write_analysis(root, results=built.sources.results)
    assert again['previous_summary_moved_to'] and (Path(again['previous_summary_moved_to'])/'decision_sv.json').is_file()


def test_the_groups_and_what_vanished_reach_the_summary_and_the_long_table(built, neutral):
    root = neutral/RUN.ROOT_NAME
    shutil.copytree(built.root/'labels', root/'labels')
    write_arm_runs(root, .86, .88)
    result = DEC.write_analysis(root, results=built.sources.results)
    record = json.loads((root/'labels/build_record.json').read_text())
    for row in result['table']:
        assert set(row['reduction_groups']) == set(DEC.GROUP_KEYS) == set(row['segmentation_groups']) and row['segmentation_groups']['small_six'] is not None
    assert result['table'][1]['reduction_groups']['big_six'] == record['arms']['r1_3']['reduction']['big_six']
    assert set(result['vanished']) == set(RUN.ARMS) and result['vanished']['r1_3']['vanished'] == record['arms']['r1_3']['vanished'] and result['vanished']['r1_3']['smallest_volume_vs_truth'] is not None
    text = (root/'analysis/long.csv').read_text()
    assert text.count('made_up_r1_3') > 0
    for measure in ('segmentation_seg35', 'segmentation_seg4', 'segmentation_small_six', 'segmentation_big_six', 'segmentation_other_23', 'segmentation_cortex', 'segmentation_white_matter', 'reduction_seg35', 'reduction_small_six'):
        assert measure in text, measure
    assert 'oracle_postmix_r1_3,r1_3,'+RUN.SUBJECTS[0] in text


def test_the_f0440_table_compares_with_the_endpoints_on_its_two_pairs(built, neutral):
    root = neutral/RUN.ROOT_NAME
    shutil.copytree(built.root/'labels', root/'labels')
    write_arm_runs(root, .86, .88, hard=.87)
    hard = DEC.write_analysis(root, results=built.sources.results)['hard_subject']
    assert [r['pair'] for r in hard['rows']] == ['0439-0440', '0440-0441'] and hard['subject'] == RUN.HARD_SUBJECT and '不进预登记读数' in hard['role'] and hard['stop'] is None
    for r in hard['rows']:
        assert r['f0440_seg35'] == pytest.approx(.87) and r['delta_seg35_from_S_P'] == pytest.approx(.87-DEC.PINS['S_P'][0]) and r['to_S_T_seg35'] == pytest.approx(DEC.PINS['S_T'][0]-.87) and r['S_P_seg4'] == pytest.approx(DEC.PINS['S_P'][1])
        assert r['segmentation'][RUN.HARD_SUBJECT]['truth']['seg35'] == 1.
    assert hard['on_19_pairs'] == pytest.approx(2*(.87-DEC.PINS['S_P'][0])/19) and '〔模型〕' in hard['model_role']


def test_the_analysis_before_any_run_is_not_decidable_and_writes(built, neutral):
    root = neutral/RUN.ROOT_NAME
    root.mkdir()
    result = DEC.write_analysis(root, results=built.sources.results)
    assert result['reading_code'] == DEC.NOT_DECIDABLE and (root/'analysis/summary_sv.md').is_file() and result['hard_subject']['sum_delta_seg35'] is None
    assert result['control']['bitwise'] is None and result['control']['record'] is None and '没有记录' in (root/'analysis/summary_sv.md').read_text(encoding='utf-8')


def test_the_line_record_keeps_the_unrounded_offset(neutral):
    results = neutral/'results'
    for arm_root, pairs, value in ((DEC.ENDPOINT_ROOT, RUN.PAIRS, .8430), (DEC.SECOND_ROOT, DEC.REST_PAIRS, .8610)):
        for pair in pairs:
            write_json(results/arm_root/'runs/S_P'/pair/'metrics.json', dict(metrics=dict(seg35=value)))
    record = DEC.line_record(results)
    assert record['line'] == .8668 and record['offset_unrounded'] == pytest.approx((5*.8430+14*.8610)/19-.8430) and record['line_unrounded'] == pytest.approx(.88-record['offset_unrounded'])
    assert DEC.line_record(neutral/'nowhere')['offset_unrounded'] is None


# ---- the start strings ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_start_strings():
    root, code = RUN.DEFAULT_ROOT, RUN.DEFAULT_ROOT/'code'
    strings = [RUN.queue_launch(), RUN.probe_argv(), RUN.build_argv(code, root), RUN.k1_argv(code, root), RUN.entry_eval_argv(code, root), RUN.entry_eval_argv(code, root, RUN.EVAL_UNIT_F0440), RUN.entry_argv(RUN.EntryCell(RUN.CONTROL_ARM, RUN.TIER, RUN.CONTROL_PAIR), code, root, 'sha')]
    for arm, pair in RUN.plan_cells():
        strings += [RUN.entry_argv(RUN.EntryCell(arm, RUN.TIER, pair), code, root, 'sha'), RUN.prepare_argv(arm, pair, RUN.run_dir(root, arm, pair)), RUN.unit_argv(arm, pair, RUN.run_dir(root, arm, pair), root)]
    for argv in strings:
        text = shlex.join(str(t) for t in argv)
        assert 'py'+'test' not in text and 'seq-vp' not in text
        if 'systemd-run' in text:
            assert '--unit=tag-sv-' in text, text
    assert shlex.join(RUN.queue_launch()).startswith('systemd-run --user --unit=tag-sv-queue')
    assert RUN.unit_name('r1_3', RUN.PAIRS[0]) == 'tag-sv-r1_3-0438-0439' and RUN.unit_name('f0440', RUN.PAIRS[2]) == 'tag-sv-f0440-0440-0441'
    assert RUN.EntryCell('f0440', RUN.TIER, RUN.PAIRS[1]).unit == 'tag-sv-entry-f0440-0439-0440' and RUN.EntryCell('r2_3', RUN.TIER, RUN.PAIRS[0]).run_id == 'sv_entry_r2_3_0438-0439'
    assert RUN.new_run_id('r1_3', RUN.PAIRS[0]).startswith('sv_r1_3_0438-0439_') and RUN.new_run_id('f0440', RUN.PAIRS[1]).startswith('sv_f0440_0439-0440_')
    for arm, pair in RUN.plan_cells():
        text = shlex.join(RUN.unit_argv(arm, pair, RUN.run_dir(root, arm, pair), root))
        assert '--property=MemoryMax=12G' in text and '1100s' in text and 'seed_inject.py' in text and f'{root}/cache/numba_jit' in text
        text = shlex.join(str(t) for t in RUN.prepare_argv(arm, pair, RUN.run_dir(root, arm, pair)))
        assert f'--stage SV-{arm}' in text and '--seconds 330' in text and '--geometry-weight-levels 1,1,0.1,0.01' in text
    for argv in (RUN.build_argv(code, root), RUN.entry_eval_argv(code, root)):
        assert '--setenv=CUDA_VISIBLE_DEVICES=' in argv and f'--setenv=PYTHONPATH={core.FROZEN}:{code}' in argv
    entry = RUN.entry_argv(RUN.EntryCell('r1_3', RUN.TIER, RUN.PAIRS[0]), code, root, 'sha')
    assert '--setenv=CUDA_VISIBLE_DEVICES=' not in entry and '--property=MemoryMax=8G' in entry and f'--setenv=PYTHONPATH={core.FROZEN}:{code}' in entry and '--nice=5' in entry


def test_the_probe_unit_has_no_gpu_one_core_nice_19_and_the_frozen_path():
    argv = RUN.probe_argv()
    assert '--unit=tag-sv-probe' in argv and '--setenv=CUDA_VISIBLE_DEVICES=' in argv and '--nice=19' in argv and '--property=CPUQuota=100%' in argv and '--property=MemoryMax=8G' in argv and f'--setenv=PYTHONPATH={core.FROZEN}:{seed.MAIN_REPO}' in argv
    assert argv[argv.index(RUN.seed.PY)+2:] == [f'{seed.MAIN_REPO}/experiments/sv/run_sv.py', 'probe', '--root', str(RUN.DEFAULT_ROOT), '--workers', '1']
    assert '--setenv=OMP_NUM_THREADS=1' in argv and '--setenv=NUMBA_NUM_THREADS=1' in argv


def test_a_start_string_with_the_word_is_refused():
    with pytest.raises(ValueError):
        RUN.prepare_argv('r1_3', RUN.PAIRS[0], '/tmp/py'+'test-of-x/run')


@pytest.mark.skipif(not (FROZEN/'jd_curl/three_d/preopt.py').is_file(), reason='the frozen worktree is not on this machine')
@pytest.mark.parametrize('command', (('probe',), ('build',), ('pin-fstar', '--by', 'x'), ('evaluate-entries',), ('decide',)))
def test_the_command_lines_reach_their_functions_and_refuse_another_root(neutral, command):
    """As the units start them, the frozen tree first: nothing but the root's refusal happens, no data is read."""
    env = dict(os.environ, PYTHONPATH=f'{FROZEN}:{ROOT}', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    done = subprocess.run([PY, '-B', str(ROOT/RUN.SCRIPT), command[0], '--root', str(neutral/'other'), *command[1:]], env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode != 0 and 'writes only' in done.stdout+done.stderr, done.stdout+done.stderr
    assert not (neutral/'other').exists()


def test_the_method_name_says_oracle():
    assert RUN.method_name('r1_3') == 'frozen_eb808be_seed_330s_gw1_1_0.1_0.01_entryQ2long_oracle_postmix_r1_3'
    assert 'oracle' in RUN.method_name('f0440')


s_p_cmd = Path('/home/rmcfz/seq_vp_tag_results/seed_v1/runs/S_P')/RUN.PAIRS[0]/'cmd.txt'


@pytest.mark.skipif(not s_p_cmd.is_file(), reason='the S_P run is not on this machine')
@pytest.mark.parametrize('arm,pair', (('r1_3', RUN.PAIRS[0]), ('r2_3', RUN.PAIRS[4]), ('f0440', RUN.PAIRS[1])))
def test_the_commands_are_s_ps_but_for_the_names(arm, pair):
    assert RUN.command_differences(arm, pair) == []


def test_the_wrapper_of_the_native_run_is_the_one_of_seed():
    assert seed.ENTRY_CHARGE_SECONDS == 47. and seed.SECONDS == 330 and seed.GEOMETRY == [1., 1., .1, .01] and seed.WATCHDOG == 1100


# ---- the evaluation step ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_the_evaluation_step_starts_the_right_script_in_the_runs_log(neutral, monkeypatch):
    run_path = neutral/'run'
    run_path.mkdir()
    seen = []
    monkeypatch.setattr(subprocess, 'run', lambda argv, **kwargs: seen.append(argv) or SimpleNamespace(returncode=0))
    system = RUN.System()
    system.code = Path('/some/code')
    assert system.evaluate(run_path, 'r1_3', RUN.PAIRS[0], 'id') == 0
    assert seen[0][:3] == [seed.PY, '-B', '/some/code/experiments/sv/run_sv.py'] and seen[0][3:] == ['evaluate-one', str(run_path), '--arm', 'r1_3', '--pair', RUN.PAIRS[0], '--run-id', 'id']


def fake_levels(run, entry):
    task = run/'pg_weak__four__330s'
    task.mkdir(parents=True, exist_ok=True)
    attempts = [dict(alpha=1., certificate_passed=True, certificate=dict(min_jd_lower_bound=.01, n_infeasible=0, state_hash='h'))]
    write_json(task/'levels.json', [dict(event='preoptimizer_done', level=3, preoptimizer=dict(policy=seed.inject_run.POLICY, raw_hash=entry['u_sha256'], seed_hash=entry['tapered_sha256'], taper_L=4, charge_seconds=47., seconds=47., seconds_total=47.5)),
                                    dict(event='level_prepared', level=3, transfer_recovery=dict(status='ready', accepted_alpha=1., attempts=attempts))])
    write_json(task/'history.json', [dict(D_start=2000., F_start=2000., D_candidate=1900.), dict(D_candidate=1500.)])


@pytest.mark.parametrize('entry_oracle,arm_oracle,expected', ((True, True, True), (False, True, False), (True, False, False)))
def test_evaluate_one_patches_the_stage_and_takes_oracle_from_the_runs_records(neutral, monkeypatch, entry_oracle, arm_oracle, expected):
    run_path = neutral/'run'
    entry = dict(oracle=entry_oracle, u_sha256='u', tapered_sha256='t', entry_metrics=dict(seg35=.8))
    write_json(run_path/'seed_entry.json', entry)
    write_json(run_path/'seed_arm.json', dict(oracle=arm_oracle))
    write_json(run_path/'protocol.json', dict(task=dict(name='pg_weak__four__330s')))
    fake_levels(run_path, entry)
    fake = SimpleNamespace(STAGE=None)

    def evaluate_run(run, arm, pair, run_id, method=None):
        assert fake.STAGE == core.STAGE                                          # $ROOT/code has no data link
        return dict(run_id=run_id, arm=arm, pair=pair, method=method, metrics=dict(seg35=.8), certificate=dict(status='s', passed=True))
    fake.evaluate_run = evaluate_run
    monkeypatch.setattr(seed, 'load_evaluate', lambda: fake)
    assert RUN.evaluate_one(SimpleNamespace(run=str(run_path), arm='r1_3', pair=RUN.PAIRS[0], run_id='id')) == 0
    doc = json.loads((run_path/'metrics.json').read_text())
    assert doc['oracle'] is expected and doc['seed']['passed'] is True and doc['sv']['provenance'] == 'oracle_postmix_r1_3' and doc['method'] == RUN.method_name('r1_3')


# ---- the gate ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class FakeMachine:
    """The gate's questions; everything open."""

    def __init__(self):
        self.load, self.gib, self.mib, self.mem, self.gpu, self.unit_list = .5, dict(home=500., host=70.), 24000., 12., None, []
        self.clock, self.sleeps = 0., []

    def load1(self):
        return self.load

    def units(self):
        return list(self.unit_list)

    def free_gib(self, path):
        return self.gib['home' if path == '/home' else 'host']

    def mem_available_gib(self):
        return self.mem

    def gpu_busy(self):
        return self.gpu

    def gpu_free_mib(self):
        return self.mib

    def now(self):
        return self.clock

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.clock += seconds


def test_the_gate_opens_and_each_bar_closes_it(neutral):
    system = FakeMachine()
    assert RUN.busy_reason(system, neutral) is None
    (neutral/'hold').write_text('')
    assert RUN.busy_reason(system, neutral) == 'hold file'
    (neutral/'hold').unlink()
    system.unit_list = ['tag-sv-queue.service']                                  # its own driver is not another unit
    assert RUN.busy_reason(system, neutral) is None
    for other in ('tag-tc-queue.service', 'seq-vp-x.service', 'tag-sv-build.service', 'tag-sv-probe.service'):
        system.unit_list = [other]
        assert other in RUN.busy_reason(system, neutral)
    system.unit_list = []
    for attribute, value, word in (('load', 1.5, 'load'), ('mem', 6.9, 'MemAvailable'), ('gpu', 'compute pids', 'GPU busy'), ('mib', 15*1024-1, 'GPU free')):
        kept = getattr(system, attribute)
        setattr(system, attribute, value)
        assert word in RUN.busy_reason(system, neutral), attribute
        setattr(system, attribute, kept)
    assert RUN.busy_reason(system, neutral) is None
    for key, value, word in (('home', 99.9, '/home'), ('host', 14.9, '/mnt/c')):
        kept = system.gib[key]
        system.gib[key] = value
        assert word in RUN.busy_reason(system, neutral)
        system.gib[key] = kept
    system.load, system.mib, system.mem = 1.49, 15*1024, 7.
    system.gib = dict(home=100., host=15.)
    assert RUN.busy_reason(system, neutral) is None


def test_the_gate_waits_then_stops(neutral):
    system = FakeMachine()
    system.load = 3.
    lines = []
    with pytest.raises(RUN.GateClosed, match='gate closed for 2 minutes'):
        RUN.wait_for_gate(system, neutral, 2, lines.append)
    assert system.sleeps and all(s == seed.POLL_SECONDS for s in system.sleeps) and any('waiting' in line for line in lines)


# ---- the queue on a fake machine ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class FakeSystem(FakeMachine):
    def __init__(self, machine):
        super().__init__()
        self.machine = machine
        self.launched = []
        self.frozen, self.registry, self.inputs, self.pin_git = [], [], [], []
        self.classes, self.finals = {}, dict(r1_3=.86, r2_3=.88, f0440=.87)
        self.fail_build, self.bad_k1, self.fail_eval, self.fail_check, self.fail_native_eval, self.unequal_entry = False, False, False, False, False, False
        self.entry_states, self.fail_eval_units, self.control_u = {}, set(), CONTROL_U
        self.close_gate_after = None
        self.reading_when_f0440_started = None

    def load1(self):
        return 9. if self.close_gate_after is not None and len(self.launched) >= self.close_gate_after else self.load

    def state(self):
        return 'fake state'

    def frozen_problems(self, results=None):
        return list(self.frozen)

    def registry_problems(self):
        return list(self.registry)

    def pin_problems(self, sources):
        return list(self.pin_git)

    def inputs_problems(self, arm, repo, root):
        return list(self.inputs)

    def code_checkout(self, root, commit):
        return ROOT, 'fake-sha'

    def run(self, argv, log):
        unit = next(a for a in argv if a.startswith('--unit=')).split('=', 1)[1]
        self.launched.append(unit)
        root = self.machine.root
        if unit == RUN.BUILD_UNIT:
            if self.fail_build:
                (root/'labels').mkdir()
                return 1
            shutil.copytree(self.machine.built.root/'labels', root/'labels')
            shutil.copytree(self.machine.built.root/'inputs', root/'inputs')
            return 0
        if unit == RUN.K1_UNIT:
            write_json(root/'k1_check_gpu.json', dict(bitwise=not self.bad_k1))
            return 0
        if unit in (RUN.EVAL_UNIT, RUN.EVAL_UNIT_F0440):
            if self.fail_eval or unit in self.fail_eval_units:
                return 8
            for arm, pair in RUN.plan_cells():
                d = RUN.entry_dir(root, arm, pair)
                if (d/'field.npz').is_file() and not (d/'metrics.json').exists():
                    with np.load(d/'field.npz') as z:
                        write_json(d/'metrics.json', dict(tapered_u_sha256=seed.boundary_taper_sha256(z['u'], SHAPE), certificate=dict(accepted_alpha=1., n_infeasible_at_1=0), metrics=dict(seg35=.85, seg4=.89, sdlogj_official=.05), at_accepted_alpha=dict(alpha=1., seg35=.85, seg4=.89)))
            return 0
        arm, pair = argv[argv.index('--arm')+1], argv[argv.index('--pair')+1]
        if arm == 'f0440' and self.reading_when_f0440_started is None:           # I1: the reading is on disk before the first unit of f0440
            path = root/'analysis/decision_sv.json'
            self.reading_when_f0440_started = json.loads(path.read_text())['reading_code'] if path.is_file() else 'none'
        state = self.entry_states.get(unit, 'done')
        if state == 'no_record':
            return 1
        if arm == RUN.CONTROL_ARM:
            d = RUN.entry_dir(root, arm, pair)
            np.savez(d/'field.npz', u=np.zeros(SHAPE+(3,)) if self.control_u == CONTROL_U else small_field(77))
            write_json(d/'run.json', dict(status=state, reason=None, oracle=True, scale=dict(images_equal_sv_inputs=True), field=dict(u_sha256=self.control_u)))
            return 0
        overrides = dict(reason='made up') if state != 'done' else {}
        if self.unequal_entry:
            overrides['scale'] = dict(images_equal_sv_inputs=False)
        make_entry(root, arm, pair, status=state, evaluated=False, **overrides)
        return 0

    def prepare(self, argv, target, log, results=None):
        run = Path(argv[argv.index('prepare')+1])
        run.mkdir(parents=True)
        write_json(run/'protocol.json', dict(task=dict(name='pg_weak__four__330s')))
        return 0

    def check_prepared(self, run, pair, stage, pair_sha, manifest_sha, entry):
        assert entry['seconds'] == 330 and entry['changes'] == seed.CHANGES and stage.startswith('SV-') and Path(entry['inputs_root']).parent == self.machine.root/'inputs'
        return dict(ok=not self.fail_check, problems=['made up'] if self.fail_check else [])

    def launch(self, argv, log):
        unit = next(a for a in argv if a.startswith('--unit=')).split('=', 1)[1]
        self.launched.append(unit)
        assert 'seed_inject.py' in argv[-1]
        run = Path(next(a for a in argv if a.startswith('--working-directory=')).split('=', 1)[1]).parent
        fake_levels(run, json.loads((run/'seed_entry.json').read_text()))
        (run/'exit_code').write_text('0\n')
        return 0

    def classify(self, run, t0):
        unit = (Path(run)/'service_unit.txt').read_text().strip()
        return {'class': self.classes.get(unit, 'ok'), 'kind': 'k', 'reason': 'r'}

    def evaluate(self, run, arm, pair, run_id):
        if self.fail_native_eval:
            return 1
        trace, _ = seed.injection_trace(run)
        doc = final_record(arm, pair, self.finals[arm], inputs_root=RUN.inputs_root(arm, self.machine.root), run_id=run_id)
        doc['seed'] = dict(trace, arm=json.loads((Path(run)/'seed_arm.json').read_text()))
        write_json(Path(run)/'metrics.json', doc)
        return 0

    def uptime(self):
        return '100.0'


CONTROL_UNIT = RUN.EntryCell(RUN.CONTROL_ARM, RUN.TIER, RUN.CONTROL_PAIR).unit
ENTRY_UNITS = [RUN.EntryCell(a, RUN.TIER, p).unit for a, p in RUN.plan_cells()]
NATIVE_UNITS = [RUN.unit_name(a, p) for a, p in RUN.plan_cells()]
MAIN_UNITS = [RUN.BUILD_UNIT, RUN.K1_UNIT, CONTROL_UNIT]+ENTRY_UNITS[:10]+[RUN.EVAL_UNIT]+NATIVE_UNITS[:10]
F0440_UNITS = ENTRY_UNITS[10:]+[RUN.EVAL_UNIT_F0440]+NATIVE_UNITS[10:]
ALL_UNITS = MAIN_UNITS+F0440_UNITS


def options(root, **changes):
    return SimpleNamespace(**dict(dict(root=str(root), wait_minutes=30, accept_control_difference=None), **changes))


@pytest.fixture
def machine(built, neutral):
    """The tree of the module fixture, with its endpoints, predictions, pins and probe, under a fresh results directory."""
    results = neutral/'results'
    shutil.copytree(built.sources.results, results, ignore=shutil.ignore_patterns(RUN.ROOT_NAME))
    sources = RUN.Sources(results=results, stage=built.sources.stage, pred_root=built.sources.pred_root, seg_root=built.sources.seg_root, pins=built.sources.pins)
    root = results/RUN.ROOT_NAME
    (root/'analysis').mkdir(parents=True)
    shutil.copyfile(built.root/'analysis'/RUN.PROBE_JSON, root/'analysis'/RUN.PROBE_JSON)
    state = SimpleNamespace(built=built, sources=sources, root=root, results=results)
    state.system = FakeSystem(state)
    return state


def run_queue(machine, **changes):
    return RUN.queue(options(machine.root, **changes), machine.system, machine.sources, Path('/nonexistent/repo'))


def stop_reason(machine):
    return json.loads((machine.root/'queue_stop.json').read_text())['reason']


def reading_code(machine):
    return json.loads((machine.root/'analysis/decision_sv.json').read_text())['reading_code']


def test_the_queue_runs_everything_in_order_and_reads(machine):
    assert run_queue(machine) == 0
    assert machine.system.launched == ALL_UNITS and not (machine.root/'queue_stop.json').exists() and not (machine.root/'f0440_stop.json').exists()
    assert machine.system.sleeps.count(RUN.BETWEEN_RUNS_SECONDS) == len(ALL_UNITS)-1
    decision = json.loads((machine.root/'analysis/decision_sv.json').read_text())
    assert decision['reading_code'] == DEC.TWO_THIRDS_NEEDED and decision['oracle'] is True and decision['reference_problems'] == []
    assert machine.system.reading_when_f0440_started == DEC.TWO_THIRDS_NEEDED     # I1
    batch = json.loads((machine.root/'batch.json').read_text())
    assert batch['oracle'] is True and batch['code_sha'] == 'fake-sha' and batch['reading_arms'] == ['r1_3', 'r2_3'] and batch['description_arms'] == ['f0440'] and len(batch['plan']) == 12 and set(batch['fstar']) == {'r1_3', 'r2_3'}
    for arm, pair in RUN.plan_cells():
        run = RUN.run_dir(machine.root, arm, pair)
        for name in ('seed_entry.json', 'seed_arm.json', 'seed_inject.py', 'cmd.txt', 'data.txt', 'git.txt', 'env.txt', 'prepared_check.json', 'queue_done.json', 'metrics.json', 'cleanup.json'):
            assert (run/name).is_file(), name
        entry, arm_record = (json.loads((run/name).read_text()) for name in ('seed_entry.json', 'seed_arm.json'))
        assert entry['oracle'] is True and arm_record['oracle'] is True and arm_record['provenance'] == RUN.ARMS[arm]['provenance'] and arm_record['decides'] == RUN.ARMS[arm]['decides']
        assert 'oracle: true' in (run/'data.txt').read_text()
        assert (RUN.entry_dir(machine.root, arm, pair)/'field.npz').is_file()    # the entry's field stays
    assert sum(json.loads((RUN.run_dir(machine.root, a, p)/'queue_done.json').read_text())['state'] == 'ok' for a, p in RUN.plan_cells()) == 12
    assert json.loads((machine.root/'control.json').read_text())['control_bitwise'] is True


def test_a_second_start_runs_nothing_again(machine):
    assert run_queue(machine) == 0
    machine.system.launched.clear()
    assert run_queue(machine) == 0
    assert machine.system.launched == []


@pytest.mark.parametrize('remove,word', (
    (lambda m: (m.built.base/'pred'/RUN.SUBJECTS[3]/'probs36.npz').rename(m.built.base/'pred'/RUN.SUBJECTS[3]/'probs36.away'), 'probs36'),
    (lambda m: (m.built.base/'seg'/RUN.SUBJECTS[0]/'provenance.json').rename(m.built.base/'seg'/RUN.SUBJECTS[0]/'provenance.away'), 'Atropos'),
    (lambda m: (m.built.sources.stage/RUN.PAIRS[1]/'pair.npz').rename(m.built.sources.stage/RUN.PAIRS[1]/'pair.away'), 'pair.npz'),
    (lambda m: (m.results/DEC.ENDPOINT_ROOT/'runs/S_T'/RUN.PAIRS[0]/'metrics.json').unlink(), 'S_T'),
    (lambda m: (m.results/DEC.ENDPOINT_ROOT/'runs/S_P'/RUN.PAIRS[4]/'metrics.json').unlink(), 'S_P'),
    (lambda m: shutil.rmtree(m.results/'d0_v1_a5/inputs/A5P'/'stage_L2R_val_v1'/RUN.PAIRS[2]), 'the check of the colouring'),
    (lambda m: (seed.entry_dir('S_P', RUN.CONTROL_PAIR, m.results)/'run.json').unlink(), 'reference of the control')))
def test_a_missing_input_stops_before_any_run(machine, remove, word):
    moved = [(machine.built.base/'pred'/RUN.SUBJECTS[3]/'probs36.away', machine.built.base/'pred'/RUN.SUBJECTS[3]/'probs36.npz'), (machine.built.base/'seg'/RUN.SUBJECTS[0]/'provenance.away', machine.built.base/'seg'/RUN.SUBJECTS[0]/'provenance.json'), (machine.built.sources.stage/RUN.PAIRS[1]/'pair.away', machine.built.sources.stage/RUN.PAIRS[1]/'pair.npz')]
    try:
        remove(machine)
        assert run_queue(machine) == 8
    finally:
        for away, back in moved:                                                 # the module's tree goes back
            if away.exists():
                away.rename(back)
    assert machine.system.launched == [] and word in stop_reason(machine)
    assert not (machine.root/'labels').exists() and not (machine.root/'batch.json').exists() and not (machine.root/'entries').exists()


@pytest.mark.parametrize('attribute,word', (('frozen', 'frozen made up'), ('registry', 'registry made up'), ('pin_git', 'pin not committed made up')))
def test_frozen_tree_registry_or_pin_problems_stop_before_any_run(machine, attribute, word):
    setattr(machine.system, attribute, [word])
    assert run_queue(machine) == 8
    assert machine.system.launched == [] and word in stop_reason(machine)


def test_an_f_star_not_pinned_stops_before_any_run(machine):
    machine.sources = RUN.Sources(results=machine.results, stage=machine.sources.stage, pred_root=machine.sources.pred_root, seg_root=machine.sources.seg_root, pins=machine.results/'nowhere.json')
    assert run_queue(machine) == 8
    assert machine.system.launched == [] and 'f* 未钉' in stop_reason(machine) and not (machine.root/'batch.json').exists()


def test_an_f_star_without_the_probe_stops_before_any_run(machine):
    (machine.root/'analysis'/RUN.PROBE_JSON).unlink()
    assert run_queue(machine) == 8
    assert machine.system.launched == [] and '探针' in stop_reason(machine)


def test_an_f_star_off_its_target_in_the_probe_stops_before_any_run(machine, neutral):
    pins = pins_of(machine.built)
    pins['arms']['r1_3']['f'], pins['arms']['r2_3']['f'] = PROBE.F_GRID[-2], PROBE.F_GRID[-1]
    path = neutral/'off.json'
    write_json(path, pins)
    machine.sources = RUN.Sources(results=machine.results, stage=machine.sources.stage, pred_root=machine.sources.pred_root, seg_root=machine.sources.seg_root, pins=path)
    assert run_queue(machine) == 8
    assert machine.system.launched == [] and 'off the target' in stop_reason(machine)


def test_an_endpoint_off_its_pin_stops_before_any_run(machine):
    path = machine.results/DEC.ENDPOINT_ROOT/'runs/S_P'/RUN.PAIRS[0]/'metrics.json'
    doc = json.loads(path.read_text())
    doc['metrics']['seg35'] += .01
    path.write_text(json.dumps(doc))
    assert run_queue(machine) == 8
    assert machine.system.launched == [] and '钉住' in stop_reason(machine)


def test_another_root_is_refused_and_nothing_is_created(machine, capsys):
    other = machine.results/'seed_v1_copy'
    assert RUN.queue(options(other), machine.system, machine.sources, Path('/nonexistent/repo')) == 2
    assert not other.exists() and machine.system.launched == [] and 'writes only' in capsys.readouterr().err


def test_a_second_driver_is_refused(machine, capsys):
    import fcntl
    with open(machine.root/'queue.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert run_queue(machine) == 2
    assert machine.system.launched == []


def test_a_failed_build_stops_before_any_gpu_and_the_second_start_does_not_build_over_it(machine):
    machine.system.fail_build = True
    assert run_queue(machine) == 8
    assert machine.system.launched == [RUN.BUILD_UNIT] and 'no GPU has started' in stop_reason(machine)
    machine.system.launched.clear()
    assert run_queue(machine) == 8
    assert machine.system.launched == [] and 'did not finish' in stop_reason(machine)
    assert list(machine.root.glob('queue_stop_*.json'))                          # the earlier stop is kept under its time


def test_a_failed_k1_check_stops(machine):
    machine.system.bad_k1 = True
    assert run_queue(machine) == 8
    assert machine.system.launched == [RUN.BUILD_UNIT, RUN.K1_UNIT] and 'K = 1' in stop_reason(machine)


def test_a_control_that_differs_stops_before_the_arms_and_can_be_accepted(machine):
    machine.system.control_u = 'd'*64
    assert run_queue(machine) == 8
    assert machine.system.launched == [RUN.BUILD_UNIT, RUN.K1_UNIT, CONTROL_UNIT] and 'control' in stop_reason(machine) and 'max |du|' in stop_reason(machine)
    record = json.loads((machine.root/'control.json').read_text())
    u = small_field(77)
    assert record['oracle'] is True and record['control_bitwise'] is False and record['u_sha256_reference'] == CONTROL_U and record['accepted_difference'] is None
    assert record['difference']['max_abs'] == float(np.abs(u).max()) and record['difference']['rms'] == pytest.approx(float(np.sqrt(np.mean(u*u)))) and record['difference']['reason'] is None
    assert not list(machine.root.glob('control_*.json'))
    machine.system.launched.clear()
    assert run_queue(machine, accept_control_difference='the lead, because') == 0
    record = json.loads((machine.root/'control.json').read_text())
    assert record['accepted_difference'] == 'the lead, because' and record['oracle'] is True and machine.system.launched[0] == ENTRY_UNITS[0]
    kept = list(machine.root.glob('control_*.json'))
    assert len(kept) == 1 and json.loads(kept[0].read_text())['accepted_difference'] is None      # the record without the acceptance stays
    assert 'ACCEPTED: the lead, because' in (machine.root/'queue.log').read_text()
    decision = json.loads((machine.root/'analysis/decision_sv.json').read_text())
    assert decision['control']['bitwise'] is False and decision['control']['record']['accepted_difference'] == 'the lead, because' and decision['control']['history'] == [kept[0].name]
    assert any(a['arm'] == RUN.CONTROL_ARM and 'the lead, because' in a['what'] for a in decision['anomalies'])
    summary = (machine.root/'analysis/summary_sv.md').read_text(encoding='utf-8')
    head = summary.split('## 表')[0]
    assert 'the lead, because' in head and '没有逐位证明' in head and 'max|Δu|' in head and 'RMS' in head
    assert decision['reading_code'] == DEC.TWO_THIRDS_NEEDED                     # the code does not change


def test_an_acceptance_stays_with_its_field_when_a_later_start_has_no_option(machine):
    machine.system.control_u = 'd'*64
    assert run_queue(machine, accept_control_difference='first reason') == 0
    machine.system.launched.clear()
    assert run_queue(machine) == 0
    assert machine.system.launched == [] and json.loads((machine.root/'control.json').read_text())['accepted_difference'] == 'first reason'
    assert not list(machine.root.glob('control_*.json'))                         # nothing changed, nothing to keep
    assert run_queue(machine, accept_control_difference='second reason') == 0
    assert json.loads((machine.root/'control.json').read_text())['accepted_difference'] == 'second reason'
    kept = list(machine.root.glob('control_*.json'))
    assert len(kept) == 1 and json.loads(kept[0].read_text())['accepted_difference'] == 'first reason'


@pytest.mark.parametrize('empty', ('', '   '))
def test_an_empty_reason_accepts_nothing(machine, empty):
    machine.system.control_u = 'd'*64
    assert run_queue(machine, accept_control_difference=empty) == 8
    assert machine.system.launched[-1] == CONTROL_UNIT and json.loads((machine.root/'control.json').read_text())['accepted_difference'] is None


def test_a_control_that_is_the_same_says_so_and_is_no_anomaly(machine):
    assert run_queue(machine) == 0
    record = json.loads((machine.root/'control.json').read_text())
    assert record['oracle'] is True and record['control_bitwise'] is True and record['difference'] is None and record['accepted_difference'] is None
    decision = json.loads((machine.root/'analysis/decision_sv.json').read_text())
    assert decision['control']['bitwise'] is True and not any(a['arm'] == RUN.CONTROL_ARM for a in decision['anomalies']) and '逐位相同' in (machine.root/'analysis/summary_sv.md').read_text(encoding='utf-8')


def test_the_difference_of_the_control_says_when_it_cannot_be_computed(machine):
    root = machine.root
    d = RUN.entry_dir(root, RUN.CONTROL_ARM, RUN.CONTROL_PAIR)
    d.mkdir(parents=True)
    assert 'no field' in RUN.control_difference(root, machine.results)['reason']
    np.savez(d/'field.npz', u=np.zeros((4, 4, 4, 3)))
    row = RUN.control_difference(root, machine.results)
    assert 'shapes' in row['reason'] and row['max_abs'] is None and row['rms'] is None


def test_a_control_that_is_not_done_stops(machine):
    machine.system.entry_states[CONTROL_UNIT] = 'cuda_oom'
    assert run_queue(machine) == 8
    assert machine.system.launched[-1] == CONTROL_UNIT and 'never rerun' in stop_reason(machine)


@pytest.mark.parametrize('state', ('cuda_oom', 'vram_stop', 'time_cap', 'input_mismatch', 'no_record'))
def test_an_entry_of_a_reading_arm_that_is_not_done_stops_and_is_never_rerun(machine, state):
    unit = ENTRY_UNITS[3]
    machine.system.entry_states[unit] = state
    assert run_queue(machine) == 8
    assert machine.system.launched == [RUN.BUILD_UNIT, RUN.K1_UNIT, CONTROL_UNIT]+ENTRY_UNITS[:4] and 'no native run' in stop_reason(machine)
    done = json.loads((RUN.entry_dir(machine.root, *RUN.plan_cells()[3])/'queue_done.json').read_text())
    assert done['state'] == ('failed' if state == 'no_record' else state)
    machine.system.launched.clear()
    machine.system.entry_states.clear()
    assert run_queue(machine) == 8
    assert machine.system.launched == [] and 'never rerun' in stop_reason(machine)


def test_an_entry_whose_images_are_not_the_inputs_stops(machine):
    machine.system.unequal_entry = True
    assert run_queue(machine) == 8
    assert machine.system.launched[-1] == ENTRY_UNITS[0] and 'bit for bit' in stop_reason(machine)


def test_a_failed_entry_evaluation_stops_before_the_native_runs(machine):
    machine.system.fail_eval = True
    assert run_queue(machine) == 8
    assert machine.system.launched[-1] == RUN.EVAL_UNIT and 'no native run' in stop_reason(machine)
    assert not (machine.root/'runs').exists()


def test_inputs_records_with_a_problem_stop_before_the_native_runs(machine):
    machine.system.inputs = ['made up']
    assert run_queue(machine) == 8
    assert machine.system.launched[-1] == RUN.EVAL_UNIT and 'made up' in stop_reason(machine)


def test_a_prepared_run_that_differs_stops(machine):
    machine.system.fail_check = True
    assert run_queue(machine) == 8
    assert machine.system.launched[-1] == RUN.EVAL_UNIT and 'differs' in stop_reason(machine)


def test_a_method_failure_is_kept_and_the_queue_goes_on(machine):
    machine.system.classes[NATIVE_UNITS[1]] = 'method_failure'
    assert run_queue(machine) == 0
    assert machine.system.launched == ALL_UNITS


def test_class_unsure_of_a_reading_arm_stops_and_what_finished_is_summed_up(machine):
    machine.system.classes[NATIVE_UNITS[2]] = 'unsure'
    assert run_queue(machine) == 8
    assert machine.system.launched[-1] == NATIVE_UNITS[2] and 'unsure' in stop_reason(machine)
    assert reading_code(machine) == DEC.NOT_DECIDABLE
    assert not any(unit in machine.system.launched for unit in F0440_UNITS)       # f0440 waits for the reading
    machine.system.launched.clear()
    machine.system.classes.clear()
    assert run_queue(machine) == 0                                               # the ended unit is post-processed; nothing is launched again for it
    assert machine.system.launched == NATIVE_UNITS[3:10]+F0440_UNITS


def test_infrastructure_failures_in_a_row_stop(machine):
    for unit in NATIVE_UNITS[:3]:
        machine.system.classes[unit] = 'infrastructure'
    assert run_queue(machine) == 8
    assert machine.system.launched[-1] == NATIVE_UNITS[2] and 'infrastructure' in stop_reason(machine)


def test_a_failed_native_evaluation_stops(machine):
    machine.system.fail_native_eval = True
    assert run_queue(machine) == 8
    assert 'evaluation failed' in stop_reason(machine)


# I1: f0440 after the reading, and its failure only recorded
@pytest.mark.parametrize('state', ('cuda_oom', 'no_record'))
def test_an_entry_of_f0440_that_fails_is_only_recorded(machine, state):
    machine.system.entry_states[ENTRY_UNITS[10]] = state
    assert run_queue(machine) == 0
    assert machine.system.launched == MAIN_UNITS+[ENTRY_UNITS[10]] and not (machine.root/'queue_stop.json').exists()
    assert reading_code(machine) == DEC.TWO_THIRDS_NEEDED
    assert 'never rerun' in json.loads((machine.root/'f0440_stop.json').read_text())['reason']
    decision = json.loads((machine.root/'analysis/decision_sv.json').read_text())
    assert decision['hard_subject']['stop'] is not None and '只记录' in (machine.root/'analysis/summary_sv.md').read_text(encoding='utf-8')
    assert all(all(r[key] is None for key in ('f0440_seg35', 'f0440_seg4')) for r in decision['hard_subject']['rows'])


def test_an_entry_evaluation_of_f0440_that_fails_is_only_recorded(machine):
    machine.system.fail_eval_units.add(RUN.EVAL_UNIT_F0440)
    assert run_queue(machine) == 0
    assert machine.system.launched == MAIN_UNITS+ENTRY_UNITS[10:]+[RUN.EVAL_UNIT_F0440] and reading_code(machine) == DEC.TWO_THIRDS_NEEDED and (machine.root/'f0440_stop.json').is_file()


def test_a_native_run_of_f0440_that_fails_is_only_recorded(machine):
    machine.system.classes[NATIVE_UNITS[10]] = 'unsure'
    assert run_queue(machine) == 0
    assert machine.system.launched[-1] == NATIVE_UNITS[10] and reading_code(machine) == DEC.TWO_THIRDS_NEEDED and 'unsure' in json.loads((machine.root/'f0440_stop.json').read_text())['reason']


def test_the_gate_closed_during_f0440_is_only_recorded(machine):
    machine.system.close_gate_after = len(MAIN_UNITS)
    assert run_queue(machine) == 0
    assert machine.system.launched == MAIN_UNITS and reading_code(machine) == DEC.TWO_THIRDS_NEEDED and 'gate closed' in json.loads((machine.root/'f0440_stop.json').read_text())['reason']


def test_any_other_error_of_f0440_is_only_recorded_too(machine, monkeypatch):
    real = machine.system.classify

    def classify(run, t0):
        if 'f0440' in str(run):
            raise ValueError('a truncated record')
        return real(run, t0)
    monkeypatch.setattr(machine.system, 'classify', classify)
    assert run_queue(machine) == 0
    assert reading_code(machine) == DEC.TWO_THIRDS_NEEDED and 'truncated' in json.loads((machine.root/'f0440_stop.json').read_text())['reason'] and not (machine.root/'queue_stop.json').exists()


@pytest.mark.parametrize('change', (dict(reading_code=DEC.NOT_DECIDABLE), dict(reading='another sentence'), dict(arms=dict(r1_3=dict(seg35=.1)))))
def test_a_reading_that_changed_after_f0440_stops(machine, monkeypatch, change):
    real, calls = DEC.write_analysis, []

    def changing(root, results=None):
        result = real(root, results=results)
        calls.append(1)
        return dict(result, **change) if len(calls) == 2 else result
    monkeypatch.setattr(DEC, 'write_analysis', changing)
    assert run_queue(machine) == 8
    assert 'the reading changed' in stop_reason(machine)


def test_the_way_back_after_a_failed_entry_is_to_rename_its_directory(machine):
    """The plan's recovery (revision 2): the failed directory stays under another name, and the next start runs that entry alone."""
    unit, cell = ENTRY_UNITS[3], RUN.plan_cells()[3]
    machine.system.entry_states[unit] = 'cuda_oom'
    assert run_queue(machine) == 8
    machine.system.launched.clear()
    machine.system.entry_states.clear()
    d = RUN.entry_dir(machine.root, *cell)
    d.rename(d.with_name(d.name+'.failed_20261104'))
    assert run_queue(machine) == 0
    assert machine.system.launched[0] == unit and d.is_dir() and (d.with_name(d.name+'.failed_20261104')/'queue_done.json').is_file()
    assert reading_code(machine) == DEC.TWO_THIRDS_NEEDED


def test_a_second_start_keeps_the_earlier_f0440_stop_and_the_reading(machine):
    machine.system.entry_states[ENTRY_UNITS[10]] = 'cuda_oom'
    assert run_queue(machine) == 0
    machine.system.launched.clear()
    assert run_queue(machine) == 0
    assert machine.system.launched == [] and list(machine.root.glob('f0440_stop_*.json')) and reading_code(machine) == DEC.TWO_THIRDS_NEEDED


def test_a_run_directory_without_exit_code_stops(machine):
    (RUN.run_dir(machine.root, *RUN.plan_cells()[0])).mkdir(parents=True)
    assert run_queue(machine) == 8
    assert 'incomplete attempt' in stop_reason(machine)


def test_the_gate_closed_stops_the_queue(machine):
    machine.system.load = 9.
    assert run_queue(machine) == 8
    assert machine.system.launched == [] and 'gate closed' in stop_reason(machine)


@pytest.mark.parametrize('after', (1, 4, 14, 16))
def test_the_gate_closing_in_the_middle_stops_the_queue_there(machine, after):
    machine.system.close_gate_after = after
    assert run_queue(machine) == 8
    assert machine.system.launched == ALL_UNITS[:after] and 'gate closed' in stop_reason(machine)


def test_the_hold_file_stops_the_queue(machine):
    (machine.root/'hold').write_text('')
    assert run_queue(machine) == 8
    assert machine.system.launched == [] and 'hold file' in stop_reason(machine)


def test_another_code_sha_stops(machine):
    write_json(machine.root/'batch.json', dict(code_sha='other', plan=[]))
    assert run_queue(machine) == 8
    assert 'batch.json is at other' in stop_reason(machine)


def test_a_checkout_that_is_not_clean_stops_with_a_record(machine, monkeypatch):
    """code_checkout is MC0's, with MC0's class of stop."""
    def refuse(root, commit):
        raise RUN.run.Stop('tracked files are not clean')
    monkeypatch.setattr(machine.system, 'code_checkout', refuse)
    assert run_queue(machine) == 8
    assert machine.system.launched == [] and 'not clean' in stop_reason(machine)


# ---- the frozen parser; a real entry on the CPU ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
HELPER = '''import json, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1]+'/experiments/sv')
import run_sv as RUN
sources = RUN.Sources(results=Path(sys.argv[2]), stage=Path(sys.argv[3]), pred_root=Path('/none'), seg_root=Path('/none'), pins=Path(sys.argv[5]))
RUN.core.STAGE = sources.stage
RUN.core.TIERS['long'] = dict(iterations=(2, 2, 2, 3))
root = sources.results/RUN.ROOT_NAME
RUN.core.frozen_api()                     # as build_cli and probe_cli: before anything else imports jd_curl
import numpy as np
import probe_sv
stacks = np.load(sys.argv[4])
reader = lambda pin, side: (stacks[getattr(pin, side+'_subject')], dict(made_up=True))
quiet = lambda text: None
probe_sv.run_probe(root, sources=sources, reader=reader, workers=1, log=quiet)
print(RUN.pin_fstar(root, 'test', sources=sources, log=quiet))
print(RUN.build_all(root, sources=sources, reader=reader, log=quiet))
import jd_curl
assert str(Path(jd_curl.__file__).resolve()).startswith('/home/rmcfz/seq_vp_tag_results/frozen_eb808be/')
print(RUN.run_entry(SimpleNamespace(arm=RUN.CONTROL_ARM, pair=RUN.CONTROL_PAIR, root=str(root), device='cpu', code_sha='test'), sources=sources))
print(RUN.run_entry(SimpleNamespace(arm='f0440', pair=RUN.PAIRS[1], root=str(root), device='cpu', code_sha='test'), sources=sources))
print(RUN.evaluate_entries(SimpleNamespace(root=str(root)), sources=sources))
'''


@pytest.mark.slow
@pytest.mark.skipif(not (FROZEN/'jd_curl/three_d/preopt.py').is_file(), reason='the frozen worktree is not on this machine')
def test_a_real_probe_build_and_entry_on_the_cpu_and_its_evaluation(neutral):
    """run_probe, pin_fstar, build_all, run_entry and evaluate_entries with the frozen tree first on the path, as in the units: the real calls of Q2 on a small synthetic volume (the four levels need 40 nodes), with few iterations; the record passes check_entry but for the tier it was given."""
    shape = (40, 42, 40)
    maps = subject_maps(shape)
    sources = make_tree(neutral, maps, shape)
    results = sources.results
    root = results/RUN.ROOT_NAME
    np.savez(neutral/'stacks.npz', **{subject: maps[subject]['P'] for subject in RUN.SUBJECTS})
    (neutral/'helper.py').write_text(HELPER)
    env = dict(os.environ, PYTHONPATH=f'{FROZEN}:{ROOT}', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', JD_CURL_JIT_CACHE=str(neutral/'jit'))
    done = subprocess.run([PY, '-B', str(neutral/'helper.py'), str(ROOT), str(results), str(sources.stage), str(neutral/'stacks.npz'), str(sources.pins)], env=env, capture_output=True, text=True, timeout=1500)
    assert done.returncode == 0, done.stdout[-3000:]+done.stderr[-3000:]
    assert (root/'labels/build_record.json').is_file() and (root/'analysis'/RUN.PROBE_JSON).is_file() and (root/'analysis'/RUN.PROBE_MD).is_file()
    assert json.loads((root/'analysis'/RUN.PROBE_JSON).read_text())['peak_rss_gib']['total'] > 0
    control = json.loads((RUN.entry_dir(root, RUN.CONTROL_ARM, RUN.CONTROL_PAIR)/'run.json').read_text())
    assert control['status'] == 'done' and control['oracle'] is True and control['scale']['images_equal_sv_inputs'] is True and control['provenance'] == 'oracle_postmix_f0control'
    assert not (RUN.entry_dir(root, RUN.CONTROL_ARM, RUN.CONTROL_PAIR)/'metrics.json').exists()      # the control is compared, not evaluated
    d = RUN.entry_dir(root, 'f0440', RUN.PAIRS[1])
    record = json.loads((d/'run.json').read_text())
    assert record['status'] == 'done' and record['oracle'] is True and record['k'] == 35 and record['scale']['images_equal_sv_inputs'] is True and record['channel_sources']['truth_labels_read'] is True
    assert set(record['channel_sources']['pair_keys_read']) <= {'fixed', 'moving', 'initial_u'}
    metrics = json.loads((d/'metrics.json').read_text())
    assert metrics['oracle'] is True and metrics['schema'] == RUN.ENTRY_METRICS_SCHEMA and {'seg35', 'seg4'} <= set(metrics['metrics'])
    assert metrics['certificate']['attempts'] and metrics['certificate']['role'] == 'record only' and metrics['certificate']['attempts'][0]['config']['max_boxes'] == 255 and metrics['certificate']['attempts'][0]['config']['max_depth'] == 48
    files = RUN.input_files('f0440', RUN.PAIRS[1], root)
    with pytest.raises(RUN.EntryError, match='long tier'):                       # the few iterations of this test are not an entry, and the check says so
        RUN.check_entry('f0440', RUN.PAIRS[1], files[3], root)
    write_json(d/'run.json', dict(record, config=core.tier_config(RUN.TIER)))
    entry = RUN.check_entry('f0440', RUN.PAIRS[1], files[3], root)
    assert entry['oracle'] is True and entry['entry_metrics']['seg35'] == metrics['metrics']['seg35']
    assert not (d/'metrics_error.json').exists()
