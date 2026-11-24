"""UNet3D shapes, padding, the Dice + CE loss against a one-hot reference, hard Dice, and that training
steps lower the loss (CPU, tiny synthetic volumes)."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from bayesreg.seg.unet.model import N_CLASSES, UNet3D, build_model, count_parameters, crop_to, pad_to_multiple
from bayesreg.seg.unet.train import TrainConfig, confusion_dice, dice_ce_loss, predict_logits

from ._helpers import tiny_subject


def test_forward_shape_default_architecture_widths():
    model = UNet3D()
    assert model.widths == (16, 32, 64, 128, 256) and model.multiple == 16 and model.n_classes == 36
    small = UNet3D(base_channels=4, depth=2)
    out = small(torch.zeros(2, 1, 8, 12, 16))
    assert out.shape == (2, N_CLASSES, 8, 12, 16)


def test_forward_rejects_sizes_not_multiple():
    with pytest.raises(ValueError):
        UNet3D(base_channels=4, depth=2)(torch.zeros(1, 1, 8, 10, 8))


def test_pad_and_crop_roundtrip():
    x = torch.arange(5*7*9, dtype=torch.float32).reshape(1, 1, 5, 7, 9)
    padded, shape = pad_to_multiple(x, 4)
    assert padded.shape[2:] == (8, 8, 12) and shape == (5, 7, 9)
    assert torch.equal(crop_to(padded, shape), x) and float(padded[..., 5:, :, :].abs().sum()) == 0


def test_predict_logits_pads_and_crops_and_full_window_equals_whole():
    torch.manual_seed(0)
    model = build_model(dict(base_channels=4, depth=2)).eval()
    image = np.random.default_rng(0).random((6, 9, 11)).astype(np.float32)
    whole = predict_logits(model, image, device="cpu")
    assert whole.shape == (1, 36, 6, 9, 11)
    window = predict_logits(model, image, device="cpu", window=(8, 12, 12))
    assert torch.allclose(whole, window, atol=1e-5)


def _reference_loss(logits, labels, n, smooth):
    p = torch.softmax(logits.double(), 1)
    onehot = F.one_hot(labels, n).permute(0, 4, 1, 2, 3).double()
    dims = (0, 2, 3, 4)
    dice = (2*(p*onehot).sum(dims)+smooth)/(p.sum(dims)+onehot.sum(dims)+smooth)
    return F.cross_entropy(logits.double(), labels)+1-dice[1:].mean()


def test_dice_ce_loss_matches_one_hot_reference_and_gradient():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(1, 6, 4, 5, 3, generator=g, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 6, (1, 4, 5, 3), generator=g)
    loss, parts = dice_ce_loss(logits, labels, n_classes=6, smooth=1e-5)
    reference = _reference_loss(logits, labels, 6, 1e-5)
    assert abs(float(loss)-float(reference)) < 1e-5
    (grad,) = torch.autograd.grad(loss, logits)
    (grad_ref,) = torch.autograd.grad(reference, logits)
    assert torch.allclose(grad.double(), grad_ref, atol=1e-6)
    assert set(parts) == {"ce", "soft_dice"}


def test_confusion_dice_matches_direct_count():
    rng = np.random.default_rng(2)
    true, pred = rng.integers(0, 5, 200), rng.integers(0, 5, 200)
    pred[:3] = 4
    true[true == 3] = 0
    pred[pred == 3] = 0                              # label 3 absent from both -> nan
    dice = confusion_dice(torch.as_tensor(pred), torch.as_tensor(true), 5)
    for k in (1, 2, 4):
        a, b = true == k, pred == k
        assert dice[k-1] == pytest.approx(2*np.sum(a & b)/(a.sum()+b.sum()))
    assert np.isnan(dice[2])


def test_training_steps_lower_the_loss():
    torch.manual_seed(0)
    item = tiny_subject("s", shape=(8, 8, 8), seed=3, n_classes=6)
    model = build_model(dict(base_channels=4, depth=2, n_classes=6))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.as_tensor(item.image)[None, None]
    y = torch.as_tensor(item.labels.astype(np.int64))[None]
    losses = []
    for _ in range(6):
        loss, _ = dice_ce_loss(model(x), y, n_classes=6)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0]


def test_train_config_defaults_match_design():
    cfg = TrainConfig()
    assert (cfg.base_channels, cfg.depth, cfg.patch_size, cfg.amp) == (16, 4, None, True)
    assert count_parameters(build_model(cfg.model_config())) == 5_648_452


@pytest.mark.parametrize("patch_size", [None, (4, 8, 8)])
def test_train_member_pads_odd_volumes_and_supports_patches(tmp_path, patch_size):
    import json
    from bayesreg.seg.unet.train import train_member
    from ._helpers import TINY
    train = [tiny_subject(f"OASIS_OAS1_09{k}_MR1", shape=(6, 10, 9), seed=k) for k in (12, 13)]
    calib = [tiny_subject(f"OASIS_OAS1_09{k}_MR1", shape=(6, 10, 9), seed=k) for k in (14, 15)]
    record = train_member(files=None, guard=None, fold="B", seed=3, models_root=tmp_path, device="cpu",
                          config=dict(TINY, patch_size=patch_size), subjects=(train, calib), log=lambda *_: None)
    assert record["iterations_run"] == TINY["max_iters"] and record["fold_index"] == 1
    assert record["label_contact_subjects"] == ["OASIS_OAS1_0912_MR1", "OASIS_OAS1_0913_MR1", "OASIS_OAS1_0914_MR1",
                                                "OASIS_OAS1_0915_MR1"]
    assert record["label_guard"] is None                # subjects given directly: no guard record (not registrable)
    lines = [json.loads(line) for line in (tmp_path/"foldB_s3"/"train_log.jsonl").read_text().splitlines()]
    events = [line["event"] for line in lines]
    assert events[0] == "start" and events[-1] == "end" and events.count("eval") == 2
    # early stopping evaluates on every calib subject, and only on them
    for history in (record["eval_history"], [line for line in lines if line["event"] == "eval"]):
        assert [len(e["per_subject_dice35"]) for e in history] == [len(calib)]*2


def test_collect_brain_logits_takes_only_brain_voxels():
    from bayesreg.seg.unet.calibrate import collect_brain_logits
    from bayesreg.seg.unet.data import Subject

    class Stub(torch.nn.Module):
        multiple, n_classes = 1, 3

        def forward(self, x):                               # class c logit = (c + 1) * intensity
            return torch.cat([x, 2*x, 3*x], dim=1)

    rng = np.random.default_rng(5)
    image = rng.uniform(.1, 1., (4, 5, 6)).astype(np.float32)
    image[0] = 0
    image[:, :2] = 0                                        # outside the brain (aligned_norm == 0)
    labels = rng.integers(0, 3, image.shape).astype(np.uint8)
    item = Subject("OASIS_OAS1_0950_MR1", image, labels)
    z, y, dice = collect_brain_logits(Stub(), [item], device="cpu", amp=False)
    brain = image > 0
    assert z.shape == (int(brain.sum()), 3) and y.shape == (int(brain.sum()),)
    assert torch.equal(y, torch.as_tensor(labels[brain].astype(np.int64)))
    assert torch.allclose(z[:, 0], torch.as_tensor(image[brain])) and torch.allclose(z[:, 2], 3*z[:, 0])
    z2, y2, _ = collect_brain_logits(Stub(), [item], device="cpu", amp=False, max_voxels=7, seed=1)
    assert z2.shape == (7, 3) and bool((z2[:, 0] > 0).all())
