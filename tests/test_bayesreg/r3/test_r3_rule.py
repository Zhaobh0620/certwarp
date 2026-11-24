"""R3a's fixed names, the closed list of candidates and the composition rule (experiments/r3/r3_common.py), the mixture and the decoding switch (r3_core.py).  CPU, no file of a subject.

- the candidates: five bases, with D only when the switch is on; each a recipe of members per fold; the weights every model equal; the composition delivered the type over both folds;
- the rule: the seg4 floor against UF (its bar included), the band strictly within .0005 of the best, the simplest of the band (fewer models first, then no D, then UF before the others --
  the review's I3 --, then seg35, then the order); the gate is BZ's (b), not R2's;
- the mixture: at w = 1/2 the numbers of eval_uf.half_and_half; without nnU-Net the float16 argmax (eval_r2's); the float64 posterior handed to the decoder sums to 1 within 1e-6 and its argmax
  is the mixture's on every voxel (the review's m6); decoding no label gives that argmax whatever the decoder; with the stub every decoding does;
- the switch (the review's I1): missing, off; on: R3b's decision through its reader (real) or the stand-in (synthetic), the pinned sha256 of decode.py and of the decision, the stub refused on
  real data, a verdict other than pass refused;
- L's expected config refuses UF records that disagree and --set values outside the four keys (the review's m7 c); the seeded trainers override initialize() only (m7 d).
"""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT/'experiments'/'r3', ROOT/'experiments'/'r2', ROOT/'experiments'/'uf', ROOT/'experiments'/'unv2', ROOT/'experiments'/'ul', ROOT/'experiments'/'nn'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import r2_common as r2c  # noqa: E402
import r3_common as r3c  # noqa: E402
import r3_core  # noqa: E402


def test_r2s_objects_are_shared():
    assert r3c.UF_EVAL_SHA256 == r2c.UF_EVAL_SHA256 and r3c.UF_MEMBER_SHA256 is r2c.UF_MEMBER_SHA256 and r3c.SUBJECTS == r2c.SUBJECTS and r3c.SMALL_SIX == r2c.SMALL_SIX
    assert (r3c.SEG_BAR, r3c.SEG_POSITIVE_MIN, r3c.SEG4_FLOOR, r3c.CONTROL_TOL_FORWARD) == (.002, 13, -.001, 2e-4)        # BZ's (b), 11-11 ~3:38 PM CST
    assert (r2c.SEG_BAR, r2c.SEG_POSITIVE_MIN) == (.005, 15)                                                               # R2's, unchanged and not used by R3
    assert r3c.OUT != r2c.OUT and r3c.DEV_READ_LOCK != r2c.DEV_READ_LOCK and r3c.CALIB_READ_LOCK != r3c.DEV_READ_LOCK and r3c.SMOKE != r2c.SMOKE


def test_l_is_ufs_recipe_with_twice_the_steps():
    import uf_common
    config = r3c.l_train_config()
    assert config == dict(uf_common.TRAIN_CONFIG, max_iters=80000, min_iters=80000, patience=80000, max_wall_seconds=28800.)
    assert set(k for k in config if config[k] != uf_common.TRAIN_CONFIG[k]) == set(r3c.UF_CHANGES)
    assert r3c.L_PROFILE == uf_common.PROFILE == 'v3_large' and r3c.L_SEED not in (0, 1, r2c.SEED) and r3c.L_NAMES == ('foldA_s3', 'foldB_s3')


def test_the_nnunet_models_and_their_seeds():
    assert r3c.NN_NAMES == ('nnA_s0', 'nnA_s1', 'nnB_s2', 'nnB_s3')
    assert len({s for _, s in r3c.NN_MODELS}) == 4 and r3c.NN_FOLD_INDEX == {'A': 0, 'B': 1}
    assert [r3c.nn_trainer(s) for _, s in r3c.NN_MODELS] == [f'nnUNetTrainer_250epochs_NoMirroring_r3s{s}' for s in range(4)]
    assert r3c.nn_models_of_fold('A') == ['nnA_s0', 'nnA_s1'] and r3c.nn_model_of('nnB_s3') == ('B', 3)
    with pytest.raises(ValueError):
        r3c.nn_model_of('nnA_s2')


def test_the_ext_trainers_are_the_seeds_of_r3_common():
    import ast
    tree = ast.parse((ROOT/'experiments/r3/ext_trainers/nn_trainers_r3.py').read_text(encoding='utf-8'))
    seeds = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.startswith('nnUNetTrainer_'):
            seeds[node.name] = next(s.value.value for s in node.body if isinstance(s, ast.Assign) and s.targets[0].id == 'SEED')
    assert seeds == {r3c.nn_trainer(s): s for _, s in r3c.NN_MODELS}
    modules = {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert modules <= {'random', 'numpy', 'torch', 'nnunetv2'}
    seeded = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == '_Seeded')
    assert [n.name for n in seeded.body if isinstance(n, ast.FunctionDef)] == ['initialize']                  # the trainer adds the seed only
    assert all(not any(isinstance(n, ast.FunctionDef) for n in node.body) for node in tree.body if isinstance(node, ast.ClassDef) and node.name != '_Seeded')


def test_the_candidates_are_closed():
    assert r3c.candidate_order(False) == ('UF', 'UF+L', 'Lswap', 'UF+N', 'UF+L+N')
    assert r3c.candidate_order(True) == r3c.candidate_order(False)+('UF+D', 'UF+L+D', 'Lswap+D', 'UF+N+D', 'UF+L+N+D')
    assert r3c.unet_members('Lswap', 'B') == [('uf', 'foldB_s0'), ('r3l', 'foldB_s3')]
    assert r3c.unet_members('UF+L+N+D', 'A') == [('uf', 'foldA_s0'), ('uf', 'foldA_s1'), ('r3l', 'foldA_s3')] and r3c.nn_members('UF+L+N', 'A') == ['nnA_s0', 'nnA_s1']
    assert [r3c.n_models(c) for c in r3c.BASE_ORDER] == [2, 3, 2, 4, 5]
    assert r3c.nn_weight('UF+N') == .5 and r3c.nn_weight('UF+L+N') == .4 and r3c.nn_weight('UF+L') == 0.


def test_the_composition_delivered_is_the_type_over_both_folds():
    d = r3c.delivered('UF+L+N+D')
    assert d['unet'] == ['uf/foldA_s0', 'uf/foldA_s1', 'r3l/foldA_s3', 'uf/foldB_s0', 'uf/foldB_s1', 'r3l/foldB_s3'] and d['nn'] == list(r3c.NN_NAMES) and d['nn_weight'] == .4 and d['decode'] is True
    # the family shares are the same over both folds: 6 U-Net, 4 nnU-Net
    assert d['nn_weight'] == len(d['nn'])/(len(d['nn'])+len(d['unet']))
    assert r3c.delivered('Lswap')['unet'] == ['uf/foldA_s0', 'r3l/foldA_s3', 'uf/foldB_s0', 'r3l/foldB_s3']


def M(table):
    return {c: dict(seg35=a, seg4=b) for c, (a, b) in table.items()}


def test_the_best_wins_when_alone():
    out = r3c.choose(M({'UF': (.90, .95), 'UF+L': (.901, .95), 'Lswap': (.899, .95), 'UF+N': (.903, .95), 'UF+L+N': (.9031, .95)}), r3c.candidate_order(False))
    assert out['best'] == 'UF+L+N' and out['band'] == ['UF+L+N', 'UF+N'] and out['winner'] == 'UF+N'        # .0001 apart: the simpler (4 models)


def test_the_band_is_within_five_ten_thousandths():
    order = r3c.candidate_order(False)
    table = {'UF': (.90, .95), 'UF+L': (.9049, .95), 'Lswap': (.80, .95), 'UF+N': (.90541, .95), 'UF+L+N': (.80, .95)}
    out = r3c.choose(M(table), order)
    assert out['best'] == 'UF+N' and out['band'] == ['UF+N'] and out['winner'] == 'UF+N'                     # .00051 apart: outside
    out = r3c.choose(M(dict(table, **{'UF+L': (.90492, .95)})), order)
    assert out['band'] == ['UF+N', 'UF+L'] and out['winner'] == 'UF+L'                                       # .00049 apart: the simpler (3 models < 4)


def test_the_seg4_floor_includes_its_bar():
    order = r3c.candidate_order(False)
    out = r3c.choose(M({'UF': (.90, .95), 'UF+L': (.91, .949), 'Lswap': (.80, .95), 'UF+N': (.92, .9489), 'UF+L+N': (.80, .95)}), order)
    assert 'UF+N' in out['excluded'] and 'UF+L' not in out['excluded'] and out['winner'] == 'UF+L'


def test_fewer_models_before_no_decoding():
    order = r3c.candidate_order(True)
    table = {c: (.80, .95) for c in order}
    table.update({'UF': (.90, .95), 'UF+L+D': (.9100, .95), 'UF+N': (.9098, .95)})
    out = r3c.choose(M(table), order)
    assert set(out['band']) == {'UF+L+D', 'UF+N'} and out['winner'] == 'UF+L+D'
    table.update({'UF+L': (.9097, .95)})
    assert r3c.choose(M(table), order)['winner'] == 'UF+L'                                                   # same models, no D first


def test_uf_before_lswap_on_a_tie_of_size_and_decoding():
    """The review's example (I3): every candidate within the band; Lswap .0002 above UF with the same models and no D must not win."""
    order = r3c.candidate_order(False)
    table = {'UF': (.9300, .95), 'UF+L': (.9298, .95), 'Lswap': (.9302, .95), 'UF+N': (.9299, .95), 'UF+L+N': (.9301, .95)}
    out = r3c.choose(M(table), order)
    assert out['best'] == 'Lswap' and set(out['band']) == set(order) and out['winner'] == 'UF'
    order = r3c.candidate_order(True)
    table = {c: (.80, .95) for c in order}
    table.update({'UF': (.90, .95), 'UF+D': (.9100, .95), 'Lswap+D': (.9102, .95)})
    assert r3c.choose(M(table), order)['winner'] == 'UF+D'
    table.update({'Lswap+D': (.9106, .95)})                                                                    # .0006 apart: Lswap+D alone in the band
    assert r3c.choose(M(table), order)['winner'] == 'Lswap+D'


def test_no_change_and_undefined():
    order = r3c.candidate_order(False)
    assert r3c.choose(M({c: (.90, .95) for c in order}), order)['winner'] == 'UF'
    out = r3c.choose(M({'UF': (float('nan'), .95), 'UF+L': (.91, .95)}), order)
    assert out['winner'] is None and 'UF+L' in out['excluded'] and not out['eligible']


# ---- the mixture ---------------------------------------------------------------------------------------------------------------------------------------
def posterior(rng, shape=(36, 9, 5, 7), peak=None):
    x = rng.gamma(.3, size=shape)
    if peak is not None:
        x[peak] += 3.
    return x/x.sum(axis=0)


def test_the_mixture_at_one_half_is_ufs_half_and_half():
    import eval_uf
    rng = np.random.default_rng(1)
    p16 = posterior(rng).astype(np.float16)
    p_nn = posterior(rng)
    assert np.array_equal(r3_core.mixed_labels(p16, p_nn, .5), eval_uf.half_and_half(p16, p_nn))
    assert np.array_equal(r3_core.mixed_labels(p16), p16.argmax(axis=0).astype(np.uint8))
    with pytest.raises(ValueError):
        r3_core.mixed_labels(p16, None, .4)
    with pytest.raises(ValueError):
        r3_core.mixed_labels(p16, p_nn, 1.)


def test_the_posterior_of_the_decoder():
    from bayesreg.seg.decode import check_probs, decoder_spec
    rng = np.random.default_rng(2)
    p16 = posterior(rng, shape=(36, 19, 6, 7)).astype(np.float16)
    p_nn = posterior(rng, shape=(36, 19, 6, 7))
    for nn, w in ((None, 0.), (p_nn, .4), (p_nn, .5)):
        post = r3_core.mixed_posterior(p16, nn, w)
        assert post.dtype == np.float64 and check_probs(post) <= 1e-6 and post.max() <= 1.
        assert np.array_equal(post.argmax(axis=0), r3_core.mixed_labels(p16, nn, w))                       # every voxel (the review's m6)
        labels, record = r3_core.candidate_labels(p16, nn, w, decoder_spec(()))
        assert labels.dtype == np.uint8 and np.array_equal(labels, r3_core.mixed_labels(p16, nn, w)) and record['decoder_id'] == 'dice_decode_v1'
        if decoder_spec()['stub']:
            assert np.array_equal(r3_core.candidate_labels(p16, nn, w, decoder_spec())[0], r3_core.mixed_labels(p16, nn, w))


# ---- the switch --------------------------------------------------------------------------------------------------------------------------------------
def test_the_switch_off_and_missing(tmp_path):
    path = tmp_path/'decode_switch.json'
    with pytest.raises(r3_core.Refused, match='no decoding switch'):
        r3_core.read_switch(path)
    path.write_text(json.dumps(dict(decode=False, words='x')))
    assert r3_core.read_switch(path)[:2] == (False, None)
    path.write_text(json.dumps(dict(decode='yes')))
    with pytest.raises(r3_core.Refused):
        r3_core.read_switch(path)


def test_the_synthetic_switch_and_its_pins(tmp_path):
    import r3_synthetic
    from bayesreg.seg.decode import decoder_spec
    switch = r3_synthetic.synthetic_switch(tmp_path/'decision.json')
    path = tmp_path/'decode_switch.json'
    path.write_text(json.dumps(switch))
    on, spec, record = r3_core.read_switch(path, synthetic=True)
    assert on and spec == json.loads(json.dumps(decoder_spec())) and record['decode_py']['sha256'] == r3c.sha256_file(r3_core.decode_module_path())
    with pytest.raises(r3_core.Refused, match='synthetic switch here'):
        r3_core.read_switch(path, synthetic=False)
    path.write_text(json.dumps(dict(switch, decode_py=dict(switch['decode_py'], sha256='0'*64))))           # decode.py is not the pinned one
    with pytest.raises(r3_core.Refused, match='not what the decision and decode.py give'):
        r3_core.read_switch(path, synthetic=True)
    path.write_text(json.dumps(switch))
    decision = json.loads((tmp_path/'decision.json').read_text())
    (tmp_path/'decision.json').write_text(json.dumps(dict(decision, note='changed')))                      # the record changed after the switch
    with pytest.raises(r3_core.Refused, match='pinned'):
        r3_core.read_switch(path, synthetic=True)
    with pytest.raises(r3_core.Refused, match='verdict pass'):
        r3_synthetic.synthetic_switch(tmp_path/'other.json', verdict='not_pass')
    (tmp_path/'decision.json').write_text(json.dumps(dict(decision, **{r3c.DECISION_DECODE_KEY: 'f'*64})))
    with pytest.raises(r3_core.Refused, match='not the one the decision recorded'):
        r3_core.switch_document(r3_core.load_decision(tmp_path/'decision.json', synthetic=True), words='x', synthetic=True)


class _FakeR3b:
    """The interface of R3b's experiments/r3/r3b_decision.py (r3b-tools 91217dc7) for the real path of the switch."""
    class DecodeNotAdmitted(Exception):
        pass

    def __init__(self, document):
        self.document, self.calls = document, []

    def load_calibration_decision(self, path, *, require_pass=False, expected_sha256=None, allow_synthetic=False):
        self.calls.append(dict(path=Path(path), require_pass=require_pass, expected_sha256=expected_sha256))
        sha = r3c.sha256_file(path)
        if expected_sha256 is not None and sha != expected_sha256:
            raise ValueError('sha256')
        if require_pass and self.document['verdict'] != 'pass':
            raise self.DecodeNotAdmitted('not pass')
        return dict(self.document, decision_sha256=sha)


def test_the_real_switch_goes_through_r3bs_reader(tmp_path, monkeypatch):
    import bayesreg.seg.decode as decode
    record = tmp_path/'decision_calib.json'
    monkeypatch.setattr(r3c, 'R3B_DECISION', record)
    document = dict(format='r3b', verdict='pass', decoder_spec=decode.decoder_spec(), **{r3c.DECISION_DECODE_KEY: r3c.sha256_file(r3_core.decode_module_path())})
    record.write_text(json.dumps(document))
    fake = _FakeR3b(document)
    monkeypatch.setitem(sys.modules, 'r3b_decision', fake)
    if decode.decoder_spec()['stub']:
        with pytest.raises(r3_core.Refused, match='stub'):
            r3_core.switch_document(dict(r3_core.load_decision(record), path=str(record)), words='x')
    monkeypatch.setattr(decode, 'STUB', False)                                                             # as R3b's final decode.py
    document = dict(document, decoder_spec=decode.decoder_spec())
    record.write_text(json.dumps(document))
    fake.document = document
    switch = r3_core.switch_document(dict(r3_core.load_decision(record), path=str(record)), words='x')
    path = tmp_path/'decode_switch.json'
    path.write_text(json.dumps(switch))
    assert r3_core.read_switch(path)[0] is True and fake.calls[-1] == dict(path=record, require_pass=True, expected_sha256=switch['decision']['sha256'])
    fake.document = dict(document, verdict='not_pass')
    record.write_text(json.dumps(fake.document))
    with pytest.raises(r3_core.Refused, match='refuses D'):
        r3_core.read_switch(path)
    with pytest.raises(r3_core.Refused, match='refuses D'):
        r3_core.load_decision(record)
    with pytest.raises(r3_core.Refused, match="not R3b's decision record"):
        r3_core.load_decision(tmp_path/'elsewhere.json')
    monkeypatch.setitem(sys.modules, 'r3b_decision', None)                                                 # not in this code tree
    with pytest.raises(r3_core.Refused, match='not in this code tree'):
        r3_core.load_decision(record)


# ---- L's expected config ---------------------------------------------------------------------------------------------------------------------------------
def test_ls_expected_config_refuses(tmp_path):
    base = dict(max_iters=40000, min_iters=40000, patience=40000, eval_every=500, checkpoint='last', max_wall_seconds=14400., base_channels=32)
    for name in r3c.UF_ORDER:
        (tmp_path/name).mkdir()
        (tmp_path/name/'train_record.json').write_text(json.dumps(dict(config=base)))
    assert r3_core.l_expected_config(tmp_path) == dict(base, max_iters=80000, min_iters=80000, patience=80000, max_wall_seconds=28800.)
    with pytest.raises(r3_core.Refused, match='four keys|--set values'):
        r3_core.l_expected_config(tmp_path, dict(r3c.l_train_config(), eval_every=250))                     # a --set value outside UF_CHANGES
    (tmp_path/r3c.UF_ORDER[2]/'train_record.json').write_text(json.dumps(dict(config=dict(base, base_channels=16))))
    with pytest.raises(r3_core.Refused, match='do not share'):
        r3_core.l_expected_config(tmp_path)
