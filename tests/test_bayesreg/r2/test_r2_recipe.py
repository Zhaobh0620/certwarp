"""R2's recipe r2_small (bayesreg/seg/unet/small.py) and the hooks of large.train_large_member it uses; the cli profile r2_small.  CPU, tiny subjects in memory or tmp_path; no data/ access.

- the hooks: their defaults are the calls of v3_large before the hooks (the same patches from the same random stream, the same loss bit for bit);
- the loss: with weight 0 it is dice_ce_loss / deep_supervision_loss bit for bit; with weight 1 the extra term is 1 - the mean soft Dice over T, computed here independently, summed with the ds weights;
- the patches: a target-centred patch is centred on a voxel of a present target class; no target present -> v3_large's patch; p_target 0 never centres on a target by this rule;
- the targets are checked before anything is read; a tiny member trains to the end of its schedule with recipe r2_small and the target fields in its record, calibrates and passes the member check;
- the cli: r2_small trains with small.train_small_member, is on the full folds, refuses register, infer, evaluate-calibration and the dry run.
"""
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from jd_curl.validation import InputValidationError

from bayesreg.seg.unet import cli, large, small
from bayesreg.seg.unet.calibrate import calibrate_member
from bayesreg.seg.unet.large import TrainConfigLarge, class_balanced_patch, deep_supervision_loss, ds_weights
from bayesreg.seg.unet.small import TrainConfigSmall, dice_ce_target_loss, require_targets, small_loss, target_patch, train_small_member
from bayesreg.seg.unet.train import dice_ce_loss

from ..segnet._helpers import A_CALIB, A_TRAIN, tiny_subject

TINY_SMALL = dict(base_channels=4, depth=2, ds_levels=1, batch_size=2, patch_size=[8, 8, 8], max_iters=4, min_iters=4, patience=4, eval_every=2, warmup_iters=1, log_every=1, amp=False, flip_axis_check=False,
                  checkpoint='last', target_labels=[4, 18, 19, 23, 34, 35])


def quiet(_):
    pass


def tiny_sets():
    return [tiny_subject(s, shape=(16, 16, 16), seed=i) for i, s in enumerate(A_TRAIN)], [tiny_subject(A_CALIB[0], shape=(16, 16, 16), seed=9)]


def batch(seed=0, shape=(2, 36, 8, 8, 8)):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(shape, generator=g)
    labels = torch.randint(0, 36, (shape[0],)+shape[2:], generator=g)
    return logits, labels


# ---- the hooks of large.py -----------------------------------------------------------------------------------------------------------------------
def test_the_default_hooks_are_the_calls_of_v3_large():
    cfg = TrainConfigLarge(patch_size=(8, 8, 8))
    image = torch.randn(1, 1, 16, 16, 16)
    labels = torch.randint(0, 36, (1, 16, 16, 16))
    for seed in range(5):
        a = large.default_patch(image, labels, (8, 8, 8), np.random.default_rng(seed), cfg)
        b = class_balanced_patch(image, labels, (8, 8, 8), np.random.default_rng(seed), p_foreground=cfg.p_foreground, n_classes=cfg.n_classes)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2] == b[2]
    outputs = [batch(0)[0], torch.randn(2, 36, 4, 4, 4)]
    labels = batch(0)[1]
    a, pa = large.default_loss(outputs, labels, cfg)
    b, pb = deep_supervision_loss(outputs, labels, n_classes=cfg.n_classes, ce_weight=cfg.ce_weight, dice_weight=cfg.dice_weight, smooth=cfg.dice_smooth)
    assert a.item() == b.item() and pa == pb


def test_train_large_member_keeps_its_signature_defaults():
    import inspect
    parameters = inspect.signature(large.train_large_member).parameters
    assert parameters['patch_fn'].default is large.default_patch and parameters['loss_fn'].default is large.default_loss and parameters['recipe'].default == large.PROFILE == 'v3_large'


# ---- the loss --------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_the_loss_with_weight_zero_is_dice_ce_bit_for_bit(seed):
    logits, labels = batch(seed)
    a, pa = dice_ce_target_loss(logits, labels, (4, 18), target_weight=0.)
    b, pb = dice_ce_loss(logits, labels)
    assert a.item() == b.item() and pa['ce'] == pb['ce'] and pa['soft_dice'] == pb['soft_dice']


def test_the_target_term_is_one_minus_the_mean_soft_dice_over_the_targets():
    logits, labels = batch(3)
    targets = (4, 18, 19)
    loss, parts = dice_ce_target_loss(logits, labels, targets, target_weight=1.)
    probs = torch.softmax(logits.float(), dim=1)
    dices = []
    for k in targets:
        p = probs[:, k]
        y = (labels == k).float()
        dices.append(((2*(p*y).sum()+1e-5)/(p.sum()+y.sum()+1e-5)).item())
    base, _ = dice_ce_loss(logits, labels)
    assert parts['target_soft_dice'] == pytest.approx(np.mean(dices), rel=1e-6)
    assert loss.item() == pytest.approx(base.item()+1-np.mean(dices), rel=1e-6)


def test_small_loss_sums_the_levels_with_the_deep_supervision_weights():
    cfg = TrainConfigSmall(target_labels=[4, 18], ds_levels=1)
    logits, labels = batch(4)
    outputs = [logits, torch.randn(2, 36, 4, 4, 4)]
    loss, parts = small_loss(outputs, labels, cfg)
    weights = ds_weights(2)
    want = sum(w*dice_ce_target_loss(o, labels[:, ::2**k, ::2**k, ::2**k], (4, 18), target_weight=1.)[0].item() for k, (o, w) in enumerate(zip(outputs, weights)))
    assert loss.item() == pytest.approx(want, rel=1e-6) and len(parts['level_target_soft_dice']) == 2
    zero = TrainConfigSmall(target_labels=[4, 18], ds_levels=1, target_dice_weight=0.)
    a, _ = small_loss(outputs, labels, zero)
    b, _ = deep_supervision_loss(outputs, labels)
    assert a.item() == b.item()


# ---- the patches -------------------------------------------------------------------------------------------------------------------------------------
def test_a_target_patch_is_centred_on_a_present_target():
    labels = torch.zeros(1, 16, 16, 16, dtype=torch.long)
    labels[0, 3, 4, 5] = 18
    labels[0, 12, 12, 12] = 2
    image = torch.randn(1, 1, 16, 16, 16)
    for seed in range(10):
        x, y, record = target_patch(image, labels, (8, 8, 8), np.random.default_rng(seed), targets=(4, 18), p_target=1.)
        assert record['target'] is True and record['label'] == 18 and (y == 18).any() and x.shape == (1, 1, 8, 8, 8)
        assert record['start'] == [0, 0, 1]                                  # centred on (3, 4, 5), shifted inside the volume


def test_no_target_present_falls_back_to_v3_large():
    labels = torch.full((1, 16, 16, 16), 2, dtype=torch.long)
    image = torch.randn(1, 1, 16, 16, 16)
    x, y, record = target_patch(image, labels, (8, 8, 8), np.random.default_rng(0), targets=(4, 18), p_target=1.)
    assert record['target'] is False and record['target_absent'] is True


def test_p_target_zero_never_centres_on_a_target_by_this_rule():
    labels = torch.randint(0, 36, (1, 16, 16, 16))
    image = torch.randn(1, 1, 16, 16, 16)
    for seed in range(20):
        assert target_patch(image, labels, (8, 8, 8), np.random.default_rng(seed), targets=(4,), p_target=0.)[2]['target'] is False


def test_a_patch_larger_than_the_volume_is_refused():
    with pytest.raises(InputValidationError):
        target_patch(torch.zeros(1, 1, 4, 4, 4), torch.zeros(1, 4, 4, 4, dtype=torch.long), (8, 8, 8), np.random.default_rng(0), targets=(4,), p_target=1.)


@pytest.mark.parametrize('targets', [[], (), [18, 4], [4, 4], [0, 4], [4, 36], ['4'], [True], '4'])
def test_the_targets_are_checked(targets):
    with pytest.raises(InputValidationError):
        require_targets(TrainConfigSmall(target_labels=targets))


@pytest.mark.parametrize('field, value', [('p_target', 1.5), ('p_target', -.1), ('target_dice_weight', -1.), ('target_dice_weight', float('nan')), ('p_target', True)])
def test_the_emphasis_is_checked(field, value):
    with pytest.raises(InputValidationError):
        require_targets(TrainConfigSmall(target_labels=[4], **{field: value}))


# ---- a tiny member ---------------------------------------------------------------------------------------------------------------------------------
def test_a_tiny_member_trains_calibrates_and_records_its_targets(tmp_path):
    train, calib = tiny_sets()
    record = train_small_member(files=None, guard=None, fold='A', seed=2, models_root=tmp_path, config=dict(TINY_SMALL), device='cpu', log=quiet, command=['test'], subjects=(train, calib))
    assert record['recipe'] == 'r2_small' and record['stopped'] == 'max_iters' and record['iterations_run'] == 4 and record['checkpoint_choice'] == 'last' and record['checkpoint_iteration'] == 4
    assert record['config']['target_labels'] == [4, 18, 19, 23, 34, 35] and record['config']['p_target'] == .5 and record['config']['target_dice_weight'] == 1.
    assert json.loads(json.dumps(record['config'])) == json.loads(json.dumps(asdict(TrainConfigSmall(**TINY_SMALL))))
    rows = [json.loads(line) for line in (tmp_path/'foldA_s2'/'train_log.jsonl').read_text().splitlines()]
    steps = [r for r in rows if r['event'] == 'train']
    assert rows[0]['recipe'] == 'r2_small' and all('target_soft_dice' in r and 'level_target_soft_dice' in r for r in steps)
    member, _ = calibrate_member(files=None, guard=None, fold='A', seed=2, models_root=tmp_path, device='cpu', log=quiet, subjects=calib)
    blob = torch.load(tmp_path/'foldA_s2'/'member.pt', map_location='cpu', weights_only=True)
    assert blob['checkpoint_choice'] == 'last' and blob['model_config'] == dict(base_channels=4, depth=2, n_classes=36, max_channels=320) and member['temperature'] > 0


def test_the_targets_are_refused_before_anything_is_read(tmp_path):
    with pytest.raises(InputValidationError):
        train_small_member(files=None, guard=None, fold='A', seed=2, models_root=tmp_path, config=dict(TINY_SMALL, target_labels=[]), device='cpu', log=quiet, subjects=None)
    assert not any(tmp_path.iterdir())


def test_the_v3_large_record_is_unchanged_by_the_hooks(tmp_path):
    train, calib = tiny_sets()
    config = {k: v for k, v in TINY_SMALL.items() if k != 'target_labels'}
    record = large.train_large_member(files=None, guard=None, fold='A', seed=0, models_root=tmp_path, config=config, device='cpu', log=quiet, subjects=(train, calib))
    assert record['recipe'] == 'v3_large' and 'target_labels' not in record['config']


# ---- the cli --------------------------------------------------------------------------------------------------------------------------------------
def test_the_cli_profile_r2_small(monkeypatch):
    called = {}
    monkeypatch.setattr(small, 'train_small_member', lambda **kwargs: (called.update(config=kwargs['config']), dict(best_calib_mean_dice35=0., stopped='x'))[1])
    monkeypatch.setattr(cli, '_require_snapshot', lambda args: None)
    monkeypatch.setattr(cli, '_guard', lambda args: (object(), {}))
    args = cli._apply_profile(cli.build_parser().parse_args(['train', '--fold', 'A', '--seed', '2', '--profile', 'r2_small', '--set', 'target_labels=[4, 18]']))
    cli.cmd_train(args)
    assert called['config'] == dict(target_labels=[4, 18])
    assert cli._on_full_folds(args) and not cli._full(args) and args.expect_counts == '98,8'
    with pytest.raises(SystemExit):
        cli.cmd_train(cli._apply_profile(cli.build_parser().parse_args(['train', '--fold', 'A', '--seed', '2', '--profile', 'r2_small', '--set', 'flip_axis_check=false'])))


@pytest.mark.parametrize('command', [['register'], ['infer', '--subjects', 'OASIS_OAS1_0438_MR1'], ['evaluate-calibration']])
def test_the_cli_refuses_r2_small_where_v3_large_is_refused(command, monkeypatch):
    args = cli._apply_profile(cli.build_parser().parse_args([*command, '--profile', 'r2_small']))
    with pytest.raises(SystemExit):
        cli.COMMANDS[args.command](args)


def test_the_cli_dry_run_of_r2_small_points_to_its_driver():
    with pytest.raises(SystemExit, match='experiments/r2/run_r2.py'):
        cli.main(['--dry-run', '--profile', 'r2_small'])
